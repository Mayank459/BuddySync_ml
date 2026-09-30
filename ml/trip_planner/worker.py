"""Planner worker: runs draft, chat and decision jobs from the Postgres queue (design §2), and closes
group decisions whose 24 h deadline has passed.

Dev: the API starts TRIP_WORKERS (default 2) worker threads itself.
Production: run the API with TRIP_WORKERS=0 and scale workers on their own:
    python -m trip_planner.worker --threads 4
"""
import argparse
import logging
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone

from trip_planner import fares, groups, hotels, llm, planner, store
from trip_planner.catalog import load_catalog

MAX_ATTEMPTS = {"draft": 3, "chat": 5, "decision": 3}
SWEEP_S = 30  # how often a worker looks for decisions past their deadline
log = logging.getLogger("trip_worker")


class Conflict(Exception):
    """A card edit landed while this job ran: replay the job on the new version."""


# ---- group decisions ------------------------------------------------------------------------------------
def open_decision(trip, cat, kind, payload, opened_by):
    """Save a decision, count the opener's 'yes', announce it (Spring posts the poll card), settle if decided."""
    options = groups.options_for(trip, kind)
    question = groups.describe(trip, cat, kind, payload)
    deadline = datetime.now(timezone.utc) + groups.DEADLINE
    did = store.create_decision(trip.trip_id, kind, payload, options, question, opened_by, deadline)
    if opened_by is not None and kind != "variant":
        store.vote_decision(did, opened_by, "yes")
    store.emit(trip.trip_id, None, "decision_opened", {"decision_id": did, "kind": kind, "question": question,
                                                        "options": options, "deadline": deadline.isoformat(),
                                                        "opened_by": opened_by})
    settle(store.decision(did), len(trip.members))
    return did


def settle(d, n_members, now=None) -> str:
    """Close the decision if a majority (or the deadline) decides it; a pass queues the apply job."""
    status, winner = groups.resolve(d["options"], d["votes"], n_members, d["deadline"],
                                    now or datetime.now(timezone.utc))
    if status == "open":
        return status
    if store.close_decision(d["decision_id"], status, {"winner": winner, "tally": d["tally"]}):
        if status == "passed":
            store.enqueue(d["trip_id"], "decision", {"decision_id": d["decision_id"], "winner": winner})
        else:
            store.emit(d["trip_id"], None, "decision_resolved", {"decision_id": d["decision_id"], "status": status,
                                                                  "tally": d["tally"]})
    return status


def sweep():
    for d in store.due_decisions():
        settle(d, store.member_count(d["trip_id"]))


# ---- jobs ---------------------------------------------------------------------------------------------------
def run(job):
    trip_id, job_id = job["trip_id"], job["job_id"]
    emit = lambda type_, data=None: store.emit(trip_id, job_id, type_, data)
    trip = store.load_trip(trip_id)
    cat = load_catalog(trip.city)
    start_version, t0 = trip.version, time.monotonic()
    new_decisions, reply, author = [], None, None
    if job["kind"] == "decision":
        d = store.decision(job["payload"]["decision_id"])
        result = groups.apply(trip, cat, d, job["payload"]["winner"])
        if "error" in result:
            store.close_decision(d["decision_id"], "failed", {"error": result["error"]}, from_status="passed")
            emit("decision_resolved", {"decision_id": d["decision_id"], "status": "failed", "error": result["error"]})
            emit("done", {"job_id": job_id})
            return
    elif job["kind"] == "draft":
        reply, model, usage = planner.draft(trip, cat, emit, cache=store.DraftCache())
        if len(trip.members) > 1 and groups.make_variants(trip, cat):  # clans vote on budget / balanced / premium
            new_decisions.append(("variant", {}, None))
    else:
        p = job["payload"]
        emit("status", {"text": "Thinking…"})
        history = store.messages(trip_id, before=p["seq"])
        older = [m for m in history[:-2 * planner.HISTORY_TURNS] if m["seq"] > trip.memory_upto]
        if older:
            planner.fold_memory(trip, older)
        reply, model, usage, new_decisions = planner.chat_turn(trip, cat, p["text"], history, emit,
                                                               actor_id=p["user_id"])
        author = p["user_id"]
    fares.enrich(trip)  # fare ranges on any travel leg that lacks one (cached; no-op without a token)
    hotels.refresh_stay(trip)  # the chosen hotel's rate, if over an hour old (no-op without LITEAPI_KEY)
    if not store.save_trip(trip, start_version):
        raise Conflict()
    if job["kind"] == "decision":
        store.close_decision(d["decision_id"], "applied", {"winner": job["payload"]["winner"],
                                                           "version": trip.version}, from_status="passed")
        emit("decision_resolved", {"decision_id": d["decision_id"], "status": "applied",
                                   "winner": job["payload"]["winner"], "version": trip.version})
    for kind, payload, opened_by in new_decisions:
        open_decision(trip, cat, kind, payload, opened_by)
    if reply is not None:
        seq = store.add_message(trip_id, "assistant", None, reply, model, usage["tokens_in"], usage["tokens_out"],
                                int((time.monotonic() - t0) * 1000))
        emit("message", {"seq": seq, "text": reply, "model": model, "reply_to": author})
        if model == "auto":
            store.log(trip_id, job_id, "fallback", {"kind": job["kind"]})
        if model == "guard":
            store.log(trip_id, job_id, "injection", {"text": job["payload"]["text"][:500]})
    if trip.version != start_version:
        emit("trip_updated", {"version": trip.version})
    emit("done", {"job_id": job_id})


def handle(job):
    try:
        run(job)
        store.finish(job, "done")
    except Conflict:
        store.retry_later(job, 0)
    except llm.Busy as e:
        if job["attempts"] < MAX_ATTEMPTS[job["kind"]]:
            delay = min(60, 15 * job["attempts"])
            store.emit(job["trip_id"], job["job_id"], "status",
                       {"text": f"Planner is busy (free-tier limit). Your message is queued, retrying in {delay}s…"})
            store.log(job["trip_id"], job["job_id"], "rate_limited", {"error": str(e)[:300]})
            store.retry_later(job, delay)
        else:
            _fail(job, "The planner is busy right now. Please send your message again in a few minutes.", e)
    except llm.NotConfigured as e:
        _fail(job, "The trip planner chat isn't configured on the server (no LLM key).", e)
    except Exception as e:  # never leave a trip busy
        _fail(job, "Something went wrong while planning. Please try again.", e)


def _fail(job, user_text, e):
    log.warning("trip job %s failed: %s", job["job_id"], e)
    store.log(job["trip_id"], job["job_id"], "error", {"error": repr(e)[:500], "trace": traceback.format_exc()[-2000:]})
    store.finish(job, "failed", repr(e)[:500])
    if job["kind"] == "decision":
        store.close_decision(job["payload"]["decision_id"], "failed", {"error": repr(e)[:300]}, from_status="passed")
    store.emit(job["trip_id"], job["job_id"], "error", {"text": user_text})
    store.emit(job["trip_id"], job["job_id"], "done", {"job_id": job["job_id"], "failed": True})


def loop(stop: threading.Event, idle_s=0.5):
    last_sweep = 0.0
    while not stop.is_set():
        try:
            if time.monotonic() - last_sweep > SWEEP_S:
                last_sweep = time.monotonic()
                sweep()
            job = store.claim()
        except Exception as e:  # database briefly unreachable
            log.warning("claim failed: %s", e)
            stop.wait(5)
            continue
        if job:
            handle(job)
        else:
            stop.wait(idle_s)


def start(threads: int):
    """Start worker threads; returns a stop() function. No-op when storage isn't configured."""
    stop = threading.Event()
    if threads <= 0:
        return stop.set
    try:
        store.pool()
    except store.StoreUnavailable as e:
        log.warning("trip workers not started: %s", e)
        return stop.set
    for n in range(threads):
        threading.Thread(target=loop, args=(stop,), name=f"trip-worker-{n}-{uuid.uuid4().hex[:4]}", daemon=True).start()
    return stop.set


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    logging.basicConfig(level=logging.INFO)
    stop = start(ap.parse_args().threads)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        stop()
