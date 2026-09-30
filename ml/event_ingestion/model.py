"""Event ingestion: is this listing a duplicate of an event we already have? Which category?

Dedup = blocking (only compare same city, within 24 h, venues within 300 m) + a pair classifier.
Shared by train.py and api.py.
"""
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from common.utils import haversine_km

STOP = {"tickets", "live", "in", "at", "the", "ft", "presents", "a", "an", "of", "night"}
PAIR_FEATURES = ["title_ratio", "title_jaccard", "same_performer", "venue_km", "hours_apart"]


def norm_title(t: str) -> str:
    return " ".join(w for w in re.sub(r"[^a-z0-9 ]", " ", t.lower()).split() if w not in STOP)


def blocked_pairs(left: pd.DataFrame, right: pd.DataFrame, max_km=0.3, max_hours=24):
    """Index pairs (i, j) worth comparing. Cheap filters first, so we never compare everything."""
    out = []
    for i, a in left.iterrows():
        near = right[(right.city == a.city) & ((right.start_time - a.start_time).abs() <= pd.Timedelta(hours=max_hours))]
        out += [(i, j) for j, b in near.iterrows()
                if haversine_km(a.venue_lat, a.venue_lng, b.venue_lat, b.venue_lng) <= max_km]
    return out


def pair_features(a, b) -> list:
    ta, tb = norm_title(a.title), norm_title(b.title)
    wa, wb = set(ta.split()), set(tb.split())
    return [SequenceMatcher(None, ta, tb).ratio(),
            len(wa & wb) / max(1, len(wa | wb)),
            float(a.performers.strip().lower() == b.performers.strip().lower()),
            haversine_km(a.venue_lat, a.venue_lng, b.venue_lat, b.venue_lng),
            abs((a.start_time - b.start_time).total_seconds()) / 3600]


def category_text(df: pd.DataFrame) -> pd.Series:
    return df.title + " " + df.description


def dedup_action(p: float) -> str:
    return "merge" if p >= 0.9 else "review" if p >= 0.5 else "new"
