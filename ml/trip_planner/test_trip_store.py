"""Trip planner against a real Postgres: API, queue ordering, worker, SSE, erasure.

Uses DATABASE_URL if set, otherwise an embedded Postgres from pgserver (skipped if neither is available).
The LLM is faked; workers run inline so the tests are deterministic.
"""
import json
import os

import pytest
from fastapi.testclient import TestClient

from common.utils import load
from trip_planner import llm, store, worker


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    if not os.getenv("DATABASE_URL"):
        pgserver = pytest.importorskip("pgserver")
        srv = pgserver.get_server(str(tmp_path_factory.mktemp("pg")), cleanup_mode="stop")
        os.environ["DATABASE_URL"] = srv.get_uri()
    store.pool.cache_clear()
    from app import app
    yield TestClient(app)  # no `with`: the lifespan (background workers) doesn't start
    store.pool.cache_clear()


@pytest.fixture
def event_id():
    ev = load("events")
    return int(ev[ev.city == "Mumbai"].iloc[0].event_id)


def run_all():
    """Drain the queue inline, like a worker thread would."""
    while (job := store.claim()) is not None:
        worker.handle(job)


def new(client, event_id, user=7):
    r = client.post("/v1/trips", json={"event_id": event_id, "user_id": user, "origin_city": "Pune"})
    assert r.status_code == 202
    return r.json()


def test_create_draft_edit_and_conflicts(client, event_id, monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda *a, **kw: (_ for _ in ()).throw(llm.NotConfigured("no key")))
    b = new(client, event_id)
    tid = b["trip_id"]
    assert b["trip"]["version"] == 0 and b["trip"]["pending_jobs"][0]["kind"] == "draft"
    run_all()
    t = client.get(f"/v1/trips/{tid}", params={"user_id": 7}).json()
    assert t["version"] == 1 and any(i["slot"] == "stay" for i in t["items"]) and t["map"] and not t["pending_jobs"]
    assert client.get(f"/v1/trips/{tid}", params={"user_id": 8}).status_code == 403

    explore = next(i for i in t["items"] if i["slot"] == "explore")
    r = client.delete(f"/v1/trips/{tid}/items/{explore['item_id']}", params={"user_id": 7, "if_version": 1})
    assert r.status_code == 200 and r.json()["trip"]["version"] == 2
    assert client.delete(f"/v1/trips/{tid}/items/{explore['item_id']}",
                         params={"user_id": 7, "if_version": 1}).status_code == 409  # stale
    bad = client.post(f"/v1/trips/{tid}/items", json={"user_id": 7, "if_version": 2, "slot": "stay", "place_id": "x"})
    assert bad.status_code == 400
    assert client.post(f"/v1/trips/{tid}/undo", json={"user_id": 7}).json()["trip"]["version"] == 3
    assert client.get(f"/v1/trips/{tid}/versions/2", params={"user_id": 7}).json()["author"] == "user:7"


def test_queue_runs_one_job_per_trip_in_order(client, event_id, monkeypatch):
    a, b = new(client, event_id)["trip_id"], new(client, event_id)["trip_id"]
    first = store.claim()
    assert first["trip_id"] == a and first["kind"] == "draft"
    second = store.claim()
    assert second["trip_id"] == b  # other trips don't wait
    j2 = store.enqueue(a, "chat", {"text": "later", "user_id": 7, "seq": 0})
    assert store.claim() is None  # trip a is busy, so its next job waits
    store.finish(first, "done")
    store.finish(second, "done")
    nxt = store.claim()
    assert nxt["job_id"] == j2
    store.finish(nxt, "done")


def test_chat_turn_through_worker_and_stream(client, event_id, monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda *a, **kw: (_ for _ in ()).throw(llm.Busy("429")))
    tid = new(client, event_id)["trip_id"]
    run_all()  # draft falls back to code when the LLM is busy
    hotels = json.loads(client.get(f"/v1/trips/{tid}", params={"user_id": 7}).content)
    stay = next(i for i in hotels["items"] if i["slot"] == "stay")

    calls = iter([llm.Reply("openai", "", [llm.Call("c1", "remove_item", {"item_id": stay["item_id"]})],
                            {"role": "assistant", "content": ""}, "fake", 0),
                  llm.Reply("openai", "Removed the hotel; want a cheaper one?", [],
                            {"role": "assistant", "content": ""}, "fake", 0)])
    monkeypatch.setattr(llm, "complete", lambda *a, **kw: next(calls))
    r = client.post(f"/v1/trips/{tid}/messages", json={"user_id": 7, "text": "drop the hotel"}).json()
    run_all()
    with client.stream("GET", f"/v1/trips/{tid}/stream", params={"user_id": 7, "until_job": r["turn_id"]}) as s:
        body = "".join(s.iter_text())
    assert "event: tool" in body and "event: message" in body and "Removed the hotel" in body
    msgs = client.get(f"/v1/trips/{tid}/messages", params={"user_id": 7}).json()["messages"]
    assert [m["role"] for m in msgs][-2:] == ["user", "assistant"] and msgs[-1]["model"] == "fake"
    t = client.get(f"/v1/trips/{tid}", params={"user_id": 7}).json()
    assert not any(i["slot"] == "stay" for i in t["items"])


def test_busy_llm_requeues_chat_then_fails_cleanly(client, event_id, monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda *a, **kw: (_ for _ in ()).throw(llm.Busy("429")))
    tid = new(client, event_id)["trip_id"]
    run_all()
    turn = client.post(f"/v1/trips/{tid}/messages", json={"user_id": 7, "text": "hi"}).json()["turn_id"]
    job = store.claim()
    worker.handle(job)
    assert store.job(turn)["status"] == "queued" and not store.is_busy(tid)  # retried later, trip released
    monkeypatch.setitem(worker.MAX_ATTEMPTS, "chat", 1)
    with store.pool().connection() as c:
        c.execute("UPDATE ml.trip_jobs SET run_after = now() WHERE job_id = %s", (turn,))
    run_all()
    assert store.job(turn)["status"] == "failed"
    types = [e["type"] for e in store.events(tid) if e["job_id"] == turn]
    assert types[-2:] == ["error", "done"] and "status" in types


def test_limits_and_erasure(client, event_id, monkeypatch):
    tid = new(client, event_id, user=42)["trip_id"]
    monkeypatch.setattr("trip_planner.api.MAX_TRIP_MESSAGES", 1)
    assert client.post(f"/v1/trips/{tid}/messages", json={"user_id": 42, "text": "a"}).status_code == 202
    assert client.post(f"/v1/trips/{tid}/messages", json={"user_id": 42, "text": "b"}).status_code == 429
    assert client.get("/v1/trips/users/42/trips").json()["trips"][0]["trip_id"] == tid
    assert client.delete("/v1/trips/users/42").json() == {"messages_deleted": 1, "memberships_removed": 1}
    assert client.get(f"/v1/trips/{tid}", params={"user_id": 42}).status_code == 403


def clan(client, event_id, ids=(21, 22, 23)):
    r = client.post("/v1/trips", json={"event_id": event_id, "user_id": ids[0], "origin_city": "Pune", "clan_id": 5,
                                       "members": [{"user_id": u, "origin_city": "Nashik"} for u in ids[1:]]})
    assert r.status_code == 202
    return r.json()["trip_id"]


def test_clan_draft_opens_variant_vote_and_majority_applies_it(client, event_id, monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda *a, **kw: (_ for _ in ()).throw(llm.Busy("429")))
    tid = clan(client, event_id)
    run_all()
    t = client.get(f"/v1/trips/{tid}", params={"user_id": 22}).json()
    assert len(t["members"]) == 3 and "budget" in t["variants"] and len(t["variants"]) >= 2  # identical bands merge
    assert sum(i["slot"] == "travel_in" for i in t["items"]) == 3  # one leg per member
    d = t["decisions"][0]
    assert d["kind"] == "variant" and d["status"] == "open" and d["opened_by"] is None
    r = client.post(f"/v1/trips/{tid}/decisions/{d['decision_id']}/vote", json={"user_id": 21, "option": "budget"})
    assert r.json()["status"] == "open"  # 1 of 3
    r = client.post(f"/v1/trips/{tid}/decisions/{d['decision_id']}/vote", json={"user_id": 23, "option": "budget"})
    assert r.json()["status"] == "passed"  # 2 of 3: decided early, apply job queued
    run_all()
    t = client.get(f"/v1/trips/{tid}", params={"user_id": 21}).json()
    stay = next(i for i in t["items"] if i["slot"] == "stay")
    assert t["preferences"]["budget_band"] == "budget" and stay["title"] == t["variants"]["budget"]["stay"]
    assert client.get(f"/v1/trips/{tid}/decisions", params={"user_id": 21}).json()["decisions"][0]["status"] == "applied"
    assert client.post(f"/v1/trips/{tid}/decisions/{d['decision_id']}/vote",
                       json={"user_id": 22, "option": "premium"}).status_code == 409  # closed


def test_lock_by_decision_then_deadline_sweep_applies_a_swap(client, event_id, monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda *a, **kw: (_ for _ in ()).throw(llm.Busy("429")))
    tid = clan(client, event_id, ids=(31, 32))
    run_all()
    lock = client.post(f"/v1/trips/{tid}/decisions", json={"user_id": 31, "kind": "lock", "payload": {"slot": "explore"}})
    assert lock.status_code == 201 and lock.json()["tally"]["yes"] == 1 and lock.json()["status"] == "open"
    client.post(f"/v1/trips/{tid}/decisions/{lock.json()['decision_id']}/vote", json={"user_id": 32, "option": "yes"})
    run_all()
    t = client.get(f"/v1/trips/{tid}", params={"user_id": 31}).json()
    assert "explore" in t["locks"]
    item = next(i for i in t["items"] if i["slot"] == "explore")
    r = client.delete(f"/v1/trips/{tid}/items/{item['item_id']}", params={"user_id": 31, "if_version": t["version"]})
    assert r.status_code == 409  # locked: only a group decision can change it

    swap = client.post(f"/v1/trips/{tid}/decisions", json={"user_id": 32, "kind": "unlock",
                                                          "payload": {"slot": "explore"}}).json()
    with store.pool().connection() as c:  # 24 h pass with only the opener's yes
        c.execute("UPDATE ml.trip_decisions SET deadline = now() - interval '1 minute' WHERE decision_id = %s",
                  (swap["decision_id"],))
    worker.sweep()
    run_all()
    t = client.get(f"/v1/trips/{tid}", params={"user_id": 31}).json()
    assert "explore" not in t["locks"]


def test_item_votes_and_members(client, event_id, monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda *a, **kw: (_ for _ in ()).throw(llm.Busy("429")))
    tid = clan(client, event_id, ids=(41, 42))
    run_all()
    t = client.get(f"/v1/trips/{tid}", params={"user_id": 41}).json()
    stay = next(i for i in t["items"] if i["slot"] == "stay")
    client.post(f"/v1/trips/{tid}/items/{stay['item_id']}/vote", json={"user_id": 41, "value": 1})
    tally = client.post(f"/v1/trips/{tid}/items/{stay['item_id']}/vote", json={"user_id": 42, "value": -1}).json()
    assert (tally["up"], tally["down"]) == (1, 1)
    r = client.post(f"/v1/trips/{tid}/members", json={"user_id": 41, "member": {"user_id": 43, "origin_city": "Surat"}})
    assert r.status_code == 200 and len(r.json()["trip"]["members"]) == 3
    assert client.patch(f"/v1/trips/{tid}/members/43", json={"user_id": 41, "origin_city": "Goa"}).status_code == 403
    r = client.patch(f"/v1/trips/{tid}/members/43", json={"user_id": 43, "origin_city": "Goa"})
    assert any(i["title"] == "Goa → Mumbai" for i in r.json()["trip"]["items"])
    assert client.delete(f"/v1/trips/{tid}/members/43", params={"user_id": 43}).status_code == 200
    assert client.get(f"/v1/trips/{tid}", params={"user_id": 43}).status_code == 403


def test_city_without_catalog_is_503(client, event_id, monkeypatch):
    def missing(city):
        raise FileNotFoundError(f"no catalog for {city}")
    monkeypatch.setattr("trip_planner.api.load_catalog", missing)
    r = client.post("/v1/trips", json={"event_id": event_id, "user_id": 7, "origin_city": "Pune"})
    assert r.status_code == 503 and "no catalog" in r.json()["detail"]


def test_voice_endpoint(client, event_id, monkeypatch):
    from trip_planner import stt
    tid = new(client, event_id)["trip_id"]
    monkeypatch.setattr(stt, "transcribe", lambda name, data: {"text": "veg food after the show", "lang": "english",
                                                              "seconds": 2.1})
    r = client.post(f"/v1/trips/{tid}/voice", data={"user_id": 7}, files={"audio": ("v.webm", b"x" * 100, "audio/webm")})
    assert r.status_code == 200 and r.json()["text"] == "veg food after the show"
    bad = client.post(f"/v1/trips/{tid}/voice", data={"user_id": 7}, files={"audio": ("v.txt", b"x", "text/plain")})
    assert bad.status_code == 400
    assert client.post(f"/v1/trips/{tid}/voice", data={"user_id": 99},
                       files={"audio": ("v.webm", b"x", "audio/webm")}).status_code == 403


def test_draft_cache_in_postgres(client, event_id, monkeypatch):
    calls = []

    def complete(task, system, msgs, **kw):
        calls.append(task)
        cands = json.loads(msgs[0]["content"])["candidates"]
        plan = {"stay": cands["stay"][0]["place_id"], "after_show": [cands["after_show"][0]["place_id"]],
                "message": "LLM plan"}
        return llm.Reply("openai", json.dumps(plan), [], {"role": "assistant", "content": ""}, "fake", 0)
    monkeypatch.setattr(llm, "complete", complete)
    run_all()  # jobs left queued by earlier tests
    calls.clear()
    with store.pool().connection() as c:
        c.execute("DELETE FROM ml.draft_cache")
    first = client.post("/v1/trips", json={"event_id": event_id, "user_id": 51, "origin_city": "Pune",
                                           "prefs": {"budget_band": "premium"}}).json()
    second = client.post("/v1/trips", json={"event_id": event_id, "user_id": 52, "origin_city": "Delhi",
                                            "prefs": {"budget_band": "premium"}}).json()
    run_all()
    assert calls == ["draft"]  # the second trip reused the plan
    models = [e["data"]["model"] for tid in (first["trip_id"], second["trip_id"])
              for e in store.events(tid) if e["type"] == "message"]
    assert models == ["fake", "cache:fake"]


def test_click_redirect_logs_and_blocks_open_redirects(client, event_id):
    tid = new(client, event_id)["trip_id"]
    run_all()
    t = client.get(f"/v1/trips/{tid}", params={"user_id": 7}).json()
    assert t["booking_disclosure"].startswith("BuddySync may earn a commission")
    link = next(b for i in t["items"] if i["slot"] == "stay" for b in i["book"])
    r = client.get(link["tracked_url"] + "&user_id=7", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == link["url"]
    assert store.click_counts(tid) == {link["provider"]: 1}
    for evil in ("https://evil.example/phish", "http://www.booking.com/x", "https://booking.com.evil.io/"):
        assert client.get("/v1/go", params={"url": evil}, follow_redirects=False).status_code == 400
