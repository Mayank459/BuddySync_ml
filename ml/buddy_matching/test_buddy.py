import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from buddy_matching.model import DEFAULT_WEIGHTS, fit_text, rank_candidates, reciprocal


def user(uid, **kw):
    base = dict(user_id=uid, city="Pune", lat=18.52, lng=73.85, age=25, gender="M",
                interests="rock;edm", bio="love rock", vibe="chill", arrive="together", after="food",
                punctuality="early", energy="ambivert", budget=2, languages="english;hindi",
                group_pref="either", verified=True, filter_verified_only=False,
                filter_women_only=False, age_min=18, age_max=40)
    return {**base, **kw}


def test_one_sided_pair_scores_below_balanced():
    w = {"interest": 0, "event_fit": 0, "location": 0, "prefs": 1}
    Z = np.array([[0, 0, 0, 0.9, 0.1], [0, 0, 0, 0.5, 0.5]])
    lopsided, balanced = reciprocal(Z, w)
    assert balanced > lopsided


def test_hard_filters_and_ranking():
    users = pd.DataFrame([
        user(1, gender="F", filter_women_only=True),
        user(2, gender="F", interests="rock;edm"),       # passes, strong match
        user(3, gender="M"),                             # excluded by viewer's women-only filter
        user(4, gender="F", interests="pottery", lat=18.60),  # passes, weaker match
        user(5, gender="F", age=39, age_min=35, age_max=45),  # viewer (25) outside *their* range
    ])
    parts = pd.DataFrame({"event_id": 7, "user_id": [1, 2, 3, 4, 5], "status": "looking_for_buddy"})
    reqs = pd.DataFrame(columns=["viewer_id", "candidate_id", "event_id", "sent", "status"]).astype({"sent": bool})
    art = {"vectorizer": fit_text(users), "scaler": FunctionTransformer(), "weights": DEFAULT_WEIGHTS}
    out = rank_candidates(1, 7, users, parts, reqs, art)
    ids = [c["user_id"] for c in out]
    assert ids == [2, 4], ids
    assert all(0 <= c["match_pct"] <= 100 for c in out)
