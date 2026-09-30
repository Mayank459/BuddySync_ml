import uuid
from functools import lru_cache

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from buddy_matching.model import rank_candidates
from common.utils import load, load_artifact

router = APIRouter(prefix="/match", tags=["buddy matching"])


@lru_cache
def artifact():
    try:
        return load_artifact("buddy_matching")
    except FileNotFoundError:
        raise HTTPException(503, "model not trained: run python -m buddy_matching.train")


@lru_cache
def tables():
    return load("users"), load("participants"), load("buddy_requests")


class BuddyQuery(BaseModel):
    user_id: int
    event_id: int
    limit: int = 10


@router.post("/buddies")
def buddies(q: BuddyQuery):
    users, participants, requests = tables()
    if q.user_id not in set(users.user_id):
        raise HTTPException(404, "unknown user")
    art = artifact()
    return {"request_id": str(uuid.uuid4()), "model_version": art["model_version"],
            "candidates": rank_candidates(q.user_id, q.event_id, users, participants, requests, art, q.limit)}
