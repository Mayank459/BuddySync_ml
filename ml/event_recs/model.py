"""Event recommendations: content-based taste + distance + tag popularity.

Every event is new (it's in the future), so we score it through its text and tags,
not through its own history. Shared by train.py and api.py.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from common.utils import haversine_km

ACTION_WEIGHT = {"going": 3.0, "looking_for_buddy": 3.0, "interested": 1.0}
WEIGHTS = {"taste": 0.6, "distance": 0.2, "popularity": 0.2}  # default; train.py tunes these on validation
# ponytail: no trending / "buddies going" / urgency terms yet (plan §5.1); add once live counts exist


def event_text(ev: pd.DataFrame) -> pd.Series:
    return ev.title + " " + ev.category + " " + ev.tags + " " + ev.performers + " " + ev.description


def fit_text(events, users) -> TfidfVectorizer:
    corpus = pd.concat([event_text(events), users.interests.str.replace(";", " ")])
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(corpus)


def popularity(events, inter) -> dict:
    """Interaction share per (city, tag), scaled to [0,1] per city. New events inherit their tag's."""
    m = inter.merge(events[["event_id", "city", "tags"]], on="event_id")
    counts = m.groupby(["city", "tags"]).size()
    return (counts / counts.groupby(level=0).transform("max")).to_dict()


def taste(user, user_inter, events, vec):
    """Profile interests + time-weighted events the user saved/RSVP'd, as one unit vector."""
    v = vec.transform([user.interests.replace(";", " ")]).toarray().ravel()
    if len(user_inter):
        X = vec.transform(event_text(events.set_index("event_id").loc[user_inter.event_id]))
        v = v + X.T.dot(user_inter.status.map(ACTION_WEIGHT).values)
    return normalize(v.reshape(1, -1)).ravel()


def score(user, tvec, cands, vec, pop, w=WEIGHTS):
    cos = vec.transform(event_text(cands)).dot(tvec)
    dist = np.array([haversine_km(user.lat, user.lng, a, b) for a, b in zip(cands.venue_lat, cands.venue_lng)])
    pop_s = np.array([pop.get((c, t), 0.0) for c, t in zip(cands.city, cands.tags)])
    total = w["taste"] * cos + w["distance"] * np.exp(-dist / 10) + w["popularity"] * pop_s
    return total, cos, dist, pop_s
