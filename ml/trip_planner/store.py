"""Postgres storage for the trip planner (ml schema): trips, versions, chat, job queue, events, logs.

DATABASE_URL is the Postgres URL. When it is unset (local dev), an embedded Postgres from `pgserver`
starts under data/pgdata. Postgres also runs the job queue (FOR UPDATE SKIP LOCKED) and the event log
that the SSE stream reads, so the planner needs no Redis.
"""
import json
import os
import time
import uuid
from functools import lru_cache
from pathlib import Path

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from common.utils import DATA
from trip_planner.model import Trip

SCHEMA = (Path(__file__).with_name("schema.sql")).read_text(encoding="utf-8")
STALE = "10 minutes"  # a running job older than this belongs to a dead worker


class StoreUnavailable(Exception):
    pass


def J(obj):
    return Jsonb(obj, dumps=lambda o: json.dumps(o, default=str, ensure_ascii=False))


@lru_cache
def pool() -> ConnectionPool:
    url = os.getenv("DATABASE_URL")
    if not url:
        try:
            import pgserver
        except ImportError:
            raise StoreUnavailable("trip planner storage not configured: set DATABASE_URL")
        # ponytail: embedded dev database; production always sets DATABASE_URL
        import logging
        logging.getLogger("trip_store").warning(
            "DATABASE_URL is not set: using an embedded Postgres inside this process. Trips are LOST on every "
            "restart or redeploy of a container. Set DATABASE_URL (e.g. a Render Postgres) in production.")
        url = pgserver.get_server(str(DATA / "pgdata"), cleanup_mode="stop").get_uri()  # stops with the last process
    p = ConnectionPool(url, min_size=1, max_size=int(os.getenv("DB_POOL_SIZE", "10")), open=True,
                       kwargs={"row_factory": dict_row, "autocommit": True})
    with p.connection() as c, c.transaction():
        c.execute("SELECT pg_advisory_xact_lock(7314001)")  # several processes may start at once
        c.execute(SCHEMA)
    return p


def _one(sql, args=()):
    with pool().connection() as c:
        return c.execute(sql, args).fetchone()


def _all(sql, args=()):
    with pool().connection() as c:
        return c.execute(sql, args).fetchall()


# ---- trips --------------------------------------------------------------------------------------------
def _insert_versions(c, trip: Trip):
    for h in trip.history:
        if h["version"] > trip.saved_version:
            c.execute("INSERT INTO ml.trip_versions (trip_id, version, parent, items, prefs, locks, author, reason) "
                      "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                      (trip.trip_id, h["version"], h["parent"], J(h["items"]), J(h["prefs"]), J(h["locks"]),
                       h["author"], h["reason"]))
    trip.saved_version = trip.version


def create_trip(trip: Trip):
    with pool().connection() as c, c.transaction():
        c.execute("INSERT INTO ml.trips (trip_id, event, clan_id, created_by, prefs, items, locks, variants, "
                  "current_version, parent_version) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                  (trip.trip_id, J(trip.event), trip.clan_id, trip.created_by, J(trip.prefs), J(trip.items),
                   J(trip.locks), J(trip.variants), trip.version, trip.parent))
        _sync_members(c, trip)
        _insert_versions(c, trip)


def _sync_members(c, trip: Trip):
    ids = [m["user_id"] for m in trip.members]
    c.execute("DELETE FROM ml.trip_members WHERE trip_id = %s AND NOT (user_id = ANY(%s))", (trip.trip_id, ids))
    for m in trip.members:
        c.execute("INSERT INTO ml.trip_members (trip_id, user_id, first_name, origin_city, role) VALUES (%s,%s,%s,%s,%s) "
                  "ON CONFLICT (trip_id, user_id) DO UPDATE SET first_name = EXCLUDED.first_name, "
                  "origin_city = EXCLUDED.origin_city",
                  (trip.trip_id, m["user_id"], m.get("first_name") or "", m.get("origin_city") or "",
                   m.get("role", "member")))


def load_trip(trip_id: str) -> Trip | None:
    row = _one("SELECT * FROM ml.trips WHERE trip_id = %s", (trip_id,))
    if not row:
        return None
    members = _all("SELECT user_id, first_name, origin_city, role FROM ml.trip_members WHERE trip_id = %s "
                   "ORDER BY joined_at, user_id", (trip_id,))
    history = _all("SELECT version, parent, items, prefs, locks, author, reason FROM ml.trip_versions "
                   "WHERE trip_id = %s ORDER BY version DESC LIMIT 50", (trip_id,))[::-1]
    return Trip(event=row["event"], members=members, prefs=row["prefs"], items=row["items"], locks=row["locks"],
                memory=row["memory"], memory_upto=row["memory_upto"], trip_id=trip_id, clan_id=row["clan_id"],
                created_by=row["created_by"], version=row["current_version"], parent=row["parent_version"],
                history=history, saved_version=row["current_version"], variants=row["variants"],
                votes=item_votes(trip_id), decisions=decisions(trip_id, open_only=True))


def save_trip(trip: Trip, expected_version: int) -> bool:
    """Optimistic write: False if someone else saved a newer version since expected_version."""
    with pool().connection() as c, c.transaction():
        cur = c.execute("UPDATE ml.trips SET prefs=%s, items=%s, locks=%s, variants=%s, memory=%s, memory_upto=%s, "
                        "current_version=%s, parent_version=%s, updated_at=now() "
                        "WHERE trip_id=%s AND current_version=%s",
                        (J(trip.prefs), J(trip.items), J(trip.locks), J(trip.variants), trip.memory,
                         trip.memory_upto, trip.version, trip.parent, trip.trip_id, expected_version))
        if cur.rowcount != 1:
            return False  # nothing written yet, so the empty transaction commits harmlessly
        _sync_members(c, trip)
        _insert_versions(c, trip)
    return True


def is_busy(trip_id: str) -> bool:
    r = _one("SELECT busy FROM ml.trips WHERE trip_id = %s", (trip_id,))
    return bool(r and r["busy"])


def version(trip_id: str, v: int):
    return _one("SELECT version, parent, items, prefs, locks, author, reason, created_at FROM ml.trip_versions "
                "WHERE trip_id = %s AND version = %s", (trip_id, v))


def is_member(trip_id: str, user_id: int) -> bool:
    return _one("SELECT 1 AS ok FROM ml.trip_members WHERE trip_id = %s AND user_id = %s", (trip_id, user_id)) is not None


def trips_for_user(user_id: int) -> list:
    return _all("SELECT t.trip_id, t.event->>'title' AS title, t.event->>'city' AS city, t.updated_at "
                "FROM ml.trips t JOIN ml.trip_members m USING (trip_id) WHERE m.user_id = %s "
                "ORDER BY t.updated_at DESC", (user_id,))


# ---- item votes and group decisions -------------------------------------------------------------------
def vote_item(trip_id, item_id, user_id, value: int):
    """value +1 / -1 sets the member's vote; 0 clears it."""
    if value == 0:
        _all("DELETE FROM ml.trip_votes WHERE trip_id=%s AND item_id=%s AND user_id=%s RETURNING 1",
             (trip_id, item_id, user_id))
    else:
        _one("INSERT INTO ml.trip_votes (trip_id, item_id, user_id, value) VALUES (%s,%s,%s,%s) "
             "ON CONFLICT (trip_id, item_id, user_id) DO UPDATE SET value = EXCLUDED.value, created_at = now() "
             "RETURNING 1", (trip_id, item_id, user_id, value))
    return item_votes(trip_id).get(item_id, {"up": 0, "down": 0, "by": {}})


def item_votes(trip_id) -> dict:
    out = {}
    for r in _all("SELECT item_id, user_id, value FROM ml.trip_votes WHERE trip_id = %s", (trip_id,)):
        v = out.setdefault(r["item_id"], {"up": 0, "down": 0, "by": {}})
        v["up" if r["value"] > 0 else "down"] += 1
        v["by"][str(r["user_id"])] = r["value"]
    return out


def create_decision(trip_id, kind, payload, options, question, opened_by, deadline) -> str:
    decision_id = uuid.uuid4().hex[:12]
    _one("INSERT INTO ml.trip_decisions (decision_id, trip_id, kind, payload, options, question, opened_by, deadline) "
         "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING decision_id",
         (decision_id, trip_id, kind, J(payload), J(options), question, opened_by, deadline))
    return decision_id


def vote_decision(decision_id, user_id, option):
    _one("INSERT INTO ml.decision_votes (decision_id, user_id, option) VALUES (%s,%s,%s) "
         "ON CONFLICT (decision_id, user_id) DO UPDATE SET option = EXCLUDED.option, created_at = now() RETURNING 1",
         (decision_id, user_id, option))


def _with_votes(rows):
    for d in rows:
        votes = _all("SELECT user_id, option FROM ml.decision_votes WHERE decision_id = %s", (d["decision_id"],))
        d["votes"] = {v["user_id"]: v["option"] for v in votes}
        d["tally"] = {o: sum(1 for v in d["votes"].values() if v == o) for o in d["options"]}
    return rows


def decision(decision_id):
    rows = _with_votes(_all("SELECT * FROM ml.trip_decisions WHERE decision_id = %s", (decision_id,)))
    return rows[0] if rows else None


def decisions(trip_id, open_only=False, limit=50) -> list:
    return _with_votes(_all("SELECT * FROM ml.trip_decisions WHERE trip_id = %s AND (NOT %s OR status = 'open') "
                            "ORDER BY created_at DESC LIMIT %s", (trip_id, open_only, limit)))


def due_decisions() -> list:
    return _with_votes(_all("SELECT * FROM ml.trip_decisions WHERE status = 'open' AND deadline <= now() "
                            "ORDER BY deadline LIMIT 100"))


def close_decision(decision_id, status, result=None, from_status="open") -> bool:
    """Move a decision on (open → passed/rejected/tied/expired, passed → applied/failed).
    Only the first caller wins: the vote endpoint and the deadline sweep can race."""
    r = _one("UPDATE ml.trip_decisions SET status = %s, result = %s, resolved_at = now() "
             "WHERE decision_id = %s AND status = %s RETURNING decision_id",
             (status, J(result), decision_id, from_status))
    return r is not None


def member_count(trip_id) -> int:
    return _one("SELECT count(*) AS n FROM ml.trip_members WHERE trip_id = %s", (trip_id,))["n"]


# ---- chat messages ---------------------------------------------------------------------------------------
def add_message(trip_id, role, sender_id, content, model=None, tokens_in=None, tokens_out=None, latency_ms=None) -> int:
    return _one("INSERT INTO ml.trip_messages (trip_id, role, sender_id, content, model, tokens_in, tokens_out, "
                "latency_ms) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING seq",
                (trip_id, role, sender_id, content, model, tokens_in, tokens_out, latency_ms))["seq"]


def messages(trip_id, after=0, before=None) -> list:
    return _all("SELECT seq, role, sender_id, content, model, created_at FROM ml.trip_messages "
                "WHERE trip_id = %s AND seq > %s AND (%s::bigint IS NULL OR seq < %s) ORDER BY seq",
                (trip_id, after, before, before))


def message_counts(trip_id, user_id) -> tuple[int, int]:
    r = _one("SELECT count(*) FILTER (WHERE trip_id = %s) AS trip, "
             "count(*) FILTER (WHERE created_at > now() - interval '1 day') AS day "
             "FROM ml.trip_messages WHERE sender_id = %s AND role = 'user'", (trip_id, user_id))
    return r["trip"], r["day"]


def erase_user(user_id: int) -> dict:
    """DPDP erasure: the user's messages go; trips they were in keep their items (author becomes 'former member')."""
    with pool().connection() as c, c.transaction():
        n_msg = c.execute("DELETE FROM ml.trip_messages WHERE sender_id = %s", (user_id,)).rowcount
        n_mem = c.execute("DELETE FROM ml.trip_members WHERE user_id = %s", (user_id,)).rowcount
        c.execute("UPDATE ml.trips SET items = (SELECT coalesce(jsonb_agg(CASE WHEN i->>'added_by' = %s "
                  "THEN jsonb_set(i, '{added_by}', '\"former member\"') ELSE i END), '[]'::jsonb) "
                  "FROM jsonb_array_elements(items) i) WHERE items @> %s",
                  (f"user:{user_id}", J([{"added_by": f"user:{user_id}"}])))
        c.execute("UPDATE ml.trip_versions SET author = 'former member' WHERE author = %s", (f"user:{user_id}",))
    return {"messages_deleted": n_msg, "memberships_removed": n_mem}


# ---- job queue -------------------------------------------------------------------------------------------
def enqueue(trip_id, kind, payload=None) -> str:
    job_id = uuid.uuid4().hex[:12]
    _one("INSERT INTO ml.trip_jobs (job_id, trip_id, kind, payload) VALUES (%s,%s,%s,%s) RETURNING job_id",
         (job_id, trip_id, kind, J(payload or {})))
    return job_id


_last_sweep = [0.0]


def claim():
    """Oldest runnable job whose trip isn't busy; jobs of one trip run strictly in order."""
    with pool().connection() as c:
        if time.monotonic() - _last_sweep[0] > 30:  # recover jobs from workers that died mid-run
            _last_sweep[0] = time.monotonic()
            c.execute(f"UPDATE ml.trip_jobs SET status='queued' WHERE status='running' "
                      f"AND started_at < now() - interval '{STALE}'")
            c.execute(f"UPDATE ml.trips SET busy=false WHERE busy AND busy_since < now() - interval '{STALE}'")
        with c.transaction():
            row = c.execute(
                "SELECT j.job_id, j.trip_id FROM ml.trip_jobs j JOIN ml.trips t ON t.trip_id = j.trip_id "
                "WHERE j.status = 'queued' AND j.run_after <= now() AND NOT t.busy "
                "AND NOT EXISTS (SELECT 1 FROM ml.trip_jobs o WHERE o.trip_id = j.trip_id "
                "  AND o.status IN ('queued', 'running') AND o.created_at < j.created_at) "
                "ORDER BY j.created_at LIMIT 1 FOR UPDATE OF j, t SKIP LOCKED").fetchone()
            if not row:
                return None
            c.execute("UPDATE ml.trips SET busy = true, busy_since = now() WHERE trip_id = %s", (row["trip_id"],))
            return c.execute("UPDATE ml.trip_jobs SET status = 'running', started_at = now(), attempts = attempts + 1 "
                             "WHERE job_id = %s RETURNING *", (row["job_id"],)).fetchone()


def finish(job, status, error=None):
    with pool().connection() as c, c.transaction():
        c.execute("UPDATE ml.trip_jobs SET status = %s, error = %s, finished_at = now() WHERE job_id = %s",
                  (status, error, job["job_id"]))
        c.execute("UPDATE ml.trips SET busy = false, busy_since = NULL WHERE trip_id = %s", (job["trip_id"],))


def retry_later(job, delay_s: float):
    with pool().connection() as c, c.transaction():
        c.execute("UPDATE ml.trip_jobs SET status = 'queued', run_after = now() + make_interval(secs => %s) "
                  "WHERE job_id = %s", (delay_s, job["job_id"]))
        c.execute("UPDATE ml.trips SET busy = false, busy_since = NULL WHERE trip_id = %s", (job["trip_id"],))


def job(job_id):
    return _one("SELECT * FROM ml.trip_jobs WHERE job_id = %s", (job_id,))


def pending(trip_id) -> list:
    return _all("SELECT job_id, kind, status, run_after FROM ml.trip_jobs WHERE trip_id = %s "
                "AND status IN ('queued', 'running') ORDER BY created_at", (trip_id,))


# ---- draft cache --------------------------------------------------------------------------------------------
class DraftCache:
    """planner.draft's cache in Postgres. A week-old plan is stale (the catalog is refreshed weekly)."""
    TTL = "7 days"

    def get(self, key):
        r = _one(f"UPDATE ml.draft_cache SET hits = hits + 1, used_at = now() WHERE key = %s "
                 f"AND created_at > now() - interval '{self.TTL}' RETURNING plan, model", (key,))
        return r and (r["plan"], r["model"])

    def put(self, key, event_id, plan, model):
        _one("INSERT INTO ml.draft_cache (key, event_id, plan, model) VALUES (%s,%s,%s,%s) ON CONFLICT (key) "
             "DO UPDATE SET plan = EXCLUDED.plan, model = EXCLUDED.model, created_at = now() RETURNING key",
             (key, event_id, J(plan), model))


# ---- events and logs ----------------------------------------------------------------------------------------
def emit(trip_id, job_id, type_, data=None) -> int:
    return _one("INSERT INTO ml.trip_events (trip_id, job_id, type, data) VALUES (%s,%s,%s,%s) RETURNING seq",
                (trip_id, job_id, type_, J(data or {})))["seq"]


def events(trip_id, after=0, limit=200) -> list:
    return _all("SELECT seq, job_id, type, data, at FROM ml.trip_events WHERE trip_id = %s AND seq > %s "
                "ORDER BY seq LIMIT %s", (trip_id, after, limit))


def log_click(trip_id, item_id, user_id, provider, host):
    _one("INSERT INTO ml.link_clicks (trip_id, item_id, user_id, provider, host) VALUES (%s,%s,%s,%s,%s) RETURNING id",
         (trip_id, item_id, user_id, provider, host))


def click_counts(trip_id) -> dict:
    return {r["provider"]: r["n"] for r in _all("SELECT provider, count(*) AS n FROM ml.link_clicks WHERE trip_id = %s "
                                                 "GROUP BY provider", (trip_id,))}


def log(trip_id, job_id, kind, detail=None):
    _one("INSERT INTO ml.planner_log (trip_id, job_id, kind, detail) VALUES (%s,%s,%s,%s) RETURNING id",
         (trip_id, job_id, kind, J(detail or {})))
