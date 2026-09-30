"""Buddy matching: pair features (preprocess) + reciprocal score + ranking.

Shared by train.py and api.py so training and serving compute features the same way.
Components per pair (viewer u, candidate v):
  interest   0.6 * tf-idf cosine(profile text) + 0.4 * jaccard(interest tags)
  event_fit  onboarding-quiz agreement (vibe, arrive, after, punctuality, budget)
  location   exp(-km / 5)
  pref_uv    share of u's soft preferences v meets (language, group size, verified, age)
  pref_vu    same, the other way round
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from common.utils import haversine_km

KEYS = ["interest", "event_fit", "location", "prefs"]
DEFAULT_WEIGHTS = {"interest": 0.40, "event_fit": 0.30, "location": 0.20, "prefs": 0.10}
QUIZ_KEYS = ["vibe", "arrive", "after", "punctuality"]


def profile_text(u) -> str:
    return u["interests"].replace(";", " ") + " " + str(u["bio"])


def fit_text(users: pd.DataFrame) -> TfidfVectorizer:
    # char n-grams cope with Hinglish spelling variants ("mast"/"mastt")
    # ponytail: tf-idf, not multilingual-e5 embeddings; switch when the Hinglish test in the plan says so
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(users.apply(profile_text, axis=1))


def hard_ok(A: pd.DataFrame, B: pd.DataFrame) -> pd.Series:
    """Row-wise: does B pass A's hard filters? Never a score, always a filter."""
    return (((~A.filter_women_only) | (B.gender == "F"))
            & ((~A.filter_verified_only) | B.verified)
            & B.age.between(A.age_min, A.age_max))


def soft_pref(A: pd.DataFrame, B: pd.DataFrame) -> np.ndarray:
    lang = np.array([bool(set(x.split(";")) & set(y.split(";"))) for x, y in zip(A.languages, B.languages)])
    group = (A.group_pref == "either") | (B.group_pref == "either") | (A.group_pref == B.group_pref)
    mid, half = (A.age_max + A.age_min) / 2, ((A.age_max - A.age_min) / 2).clip(lower=1)
    age_fit = (1 - (B.age - mid).abs() / half).clip(0, 1)
    return ((lang + group.values + B.verified.values + age_fit.values) / 4).astype(float)


def features(pairs: pd.DataFrame, users: pd.DataFrame, vec: TfidfVectorizer):
    """pairs has viewer_id, candidate_id. Returns (raw (n,5), A, B, common_tags, dist_km)."""
    U = users.set_index("user_id")
    A = U.loc[pairs["viewer_id"]].reset_index()
    B = U.loc[pairs["candidate_id"]].reset_index()
    XA, XB = vec.transform(A.apply(profile_text, axis=1)), vec.transform(B.apply(profile_text, axis=1))
    cos = np.asarray(XA.multiply(XB).sum(axis=1)).ravel()  # rows are L2-normalised → dot = cosine
    ta, tb = A.interests.str.split(";").map(set), B.interests.str.split(";").map(set)
    common = [x & y for x, y in zip(ta, tb)]
    jac = np.array([len(c) / len(x | y) for c, x, y in zip(common, ta, tb)])
    quiz = sum((A[k] == B[k]).astype(float) for k in QUIZ_KEYS) + 1 - (A.budget - B.budget).abs() / 2
    dist = np.array([haversine_km(*r) for r in zip(A.lat, A.lng, B.lat, B.lng)])
    raw = np.column_stack([0.6 * cos + 0.4 * jac, (quiz / 5).values, np.exp(-dist / 5),
                           soft_pref(A, B), soft_pref(B, A)])
    return raw, A, B, common, dist


def reciprocal(Z: np.ndarray, w: dict) -> np.ndarray:
    """Score both directions, combine with a harmonic mean: one-sided pairs score low."""
    base = w["interest"] * Z[:, 0] + w["event_fit"] * Z[:, 1] + w["location"] * Z[:, 2]
    s_uv, s_vu = base + w["prefs"] * Z[:, 3], base + w["prefs"] * Z[:, 4]
    return 2 * s_uv * s_vu / (s_uv + s_vu + 1e-9)


def distance_bucket(km: float) -> str:
    return "< 2 km" if km < 2 else "2-5 km" if km < 5 else "5-10 km" if km < 10 else "10+ km"


def rank_candidates(viewer_id, event_id, users, participants, requests, art, limit=10):
    pool = participants.loc[(participants.event_id == event_id) & (participants.user_id != viewer_id), "user_id"]
    asked = requests.loc[(requests.viewer_id == viewer_id) & requests.sent, "candidate_id"]
    pool = pd.Series(pool[~pool.isin(asked)].unique())
    if pool.empty:
        return []
    raw, A, B, common, dist = features(pd.DataFrame({"viewer_id": viewer_id, "candidate_id": pool}),
                                       users, art["vectorizer"])
    keep = (hard_ok(A, B) & hard_ok(B, A)).values
    if not keep.any():
        return []
    Z = art["scaler"].transform(raw[keep])
    w = art["weights"]
    R = reciprocal(Z, w)
    # congestion: people with many unanswered requests get shown less (anti-"superstar")
    pending = requests.loc[requests.status == "pending", "candidate_id"].value_counts()
    ids = pool[keep].reset_index(drop=True)
    final = R / (1 + ids.map(pending).fillna(0).values / 5)
    # ponytail: no trust multiplier / exploration slots yet — need report data and user age; see plan §3.3
    kept_common = [c for c, k in zip(common, keep) if k]
    kept_dist = dist[keep]
    out = []
    for i in np.argsort(-final)[:limit]:
        contrib = {"interest": w["interest"] * Z[i, 0], "event_fit": w["event_fit"] * Z[i, 1],
                   "location": w["location"] * Z[i, 2]}
        reasons = []
        for k in sorted(contrib, key=contrib.get, reverse=True)[:2]:
            if k == "interest":
                reasons.append("both into " + ", ".join(sorted(kept_common[i])[:2]) if kept_common[i] else "similar profiles")
            elif k == "event_fit":
                reasons.append("similar plans for the night")
            else:
                reasons.append(f"{distance_bucket(kept_dist[i])} apart")
        out.append({"user_id": int(ids[i]), "match_pct": int(round(100 * R[i])),
                    "distance": distance_bucket(kept_dist[i]), "reasons": reasons,
                    "components": {k: round(float(Z[i, j]), 3) for j, k in enumerate(["interest", "event_fit", "location", "pref_uv", "pref_vu"])}})
    return out
