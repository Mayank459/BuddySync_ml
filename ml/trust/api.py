from functools import lru_cache

from fastapi import APIRouter, HTTPException

from common.utils import load
from trust.model import attended, profile, table

router = APIRouter(prefix="/trust", tags=["trust & safety"])


@lru_cache
def tables():
    peer, ratings = load("peer_feedback"), load("event_ratings")
    return set(load("users").user_id), table(peer), attended(peer, ratings)


@router.get("/users/{user_id}")
def trust(user_id: int):
    """Trust signals from post-event feedback: events attended, show-up record, reliability level.
    Face/phone verified come from the app's own records, not from here."""
    users, t, events = tables()
    if user_id not in users:
        raise HTTPException(404, "unknown user")
    return profile(user_id, t, events)
