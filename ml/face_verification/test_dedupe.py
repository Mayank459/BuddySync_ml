"""Duplicate accounts: matching logic offline, then the template table against a real Postgres
(DATABASE_URL, else an embedded pgserver; skipped if neither is available)."""
import os

import numpy as np
import pytest

from face_verification import api, dedupe


def unit(*v):
    v = np.array(v, np.float32)
    return v / np.linalg.norm(v)


def test_matches_only_above_threshold_most_similar_first():
    me = unit(1, 0, 0)
    others = [(7, unit(0, 1, 0)), (8, unit(0.9, 0.1, 0)), (9, unit(1, 0.02, 0))]
    assert [m["user_id"] for m in dedupe.matches(me, others, 0.9)] == [9, 8]
    assert dedupe.matches(me, [], 0.9) == []


def test_match_sends_to_review_and_a_broken_store_does_not_break_verification(monkeypatch):
    art = {"thresholds": {"pass": 0.37, "review": 0.36, "duplicate": 0.5}, "model_version": "m"}
    monkeypatch.setattr(dedupe, "check_and_keep", lambda *a: [{"user_id": 8, "similarity": 0.8}])
    r = api.duplicate_check(1, unit(1, 0), art)
    assert r["decision"] == "review" and r["reason"] == "possible_duplicate_account"

    def down(*a):
        raise ConnectionError("db down")
    monkeypatch.setattr(dedupe, "check_and_keep", down)
    assert api.duplicate_check(1, unit(1, 0), art) == {"duplicate_check": "unavailable"}
    assert api.duplicate_check(1, unit(1, 0), {**art, "thresholds": {"review": 0.36}}) == {"duplicate_check": "not_calibrated"}


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    from trip_planner import store
    if not os.getenv("DATABASE_URL"):
        pgserver = pytest.importorskip("pgserver")
        srv = pgserver.get_server(str(tmp_path_factory.mktemp("pg")), cleanup_mode="stop")
        os.environ["DATABASE_URL"] = srv.get_uri()
    store.pool.cache_clear()
    yield
    store.pool.cache_clear()


def test_templates_round_trip(db):
    face_a, face_b = unit(1, 0.05, 0), unit(0, 1, 0)
    assert dedupe.check_and_keep(101, face_a, 0.9, "m") == []
    assert dedupe.check_and_keep(102, face_b, 0.9, "m") == []
    assert dedupe.check_and_keep(101, face_a, 0.9, "m") == []        # re-verifying your own account is fine
    hit = dedupe.check_and_keep(103, unit(1, 0.04, 0), 0.9, "m")     # same face, third account
    assert [h["user_id"] for h in hit] == [101]
    assert dedupe.check_and_keep(104, face_a, 0.9, "other-model") == []  # templates from another model don't mix
    assert dedupe.erase(101) and not dedupe.erase(101)
    assert dedupe.check_and_keep(105, face_a, 0.9, "m")[0]["user_id"] == 103
