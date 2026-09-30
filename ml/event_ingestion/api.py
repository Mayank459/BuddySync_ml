from datetime import datetime
from functools import lru_cache

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from common.utils import load, load_artifact
from event_ingestion.model import blocked_pairs, category_text, dedup_action, pair_features

router = APIRouter(prefix="/events", tags=["event ingestion"])


@lru_cache
def artifact():
    try:
        return load_artifact("event_ingestion")
    except FileNotFoundError:
        raise HTTPException(503, "model not trained: run python -m event_ingestion.train")


@lru_cache
def canonical_events():
    ev = load("events")
    ev["start_time"] = pd.to_datetime(ev.start_time)
    return ev


class Listing(BaseModel):
    title: str
    performers: str
    city: str
    venue_lat: float
    venue_lng: float
    start_time: datetime
    description: str = ""


@router.post("/ingest")
def ingest(listing: Listing):
    art, events = artifact(), canonical_events()
    new = pd.DataFrame([listing.model_dump()])
    proba = art["category"].predict_proba(category_text(new))[0]
    best = proba.argmax()
    dup_id, dup_p = None, 0.0
    for _, j in blocked_pairs(new, events):
        p = art["dedup"].predict_proba([pair_features(new.iloc[0], events.loc[j])])[0, 1]
        if p > dup_p:
            dup_id, dup_p = int(events.event_id[j]), float(p)
    return {"category": art["category"].classes_[best], "category_confidence": round(float(proba[best]), 3),
            "category_needs_review": bool(proba[best] < 0.5),  # ponytail: LLM fallback for these, plan §6
            "duplicate_of": dup_id, "duplicate_probability": round(dup_p, 3),
            "action": dedup_action(dup_p) if dup_id else "new"}
