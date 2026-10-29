import uuid
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from buddy_matching.event_profile import EventProfile
from buddy_matching.model import (
    DEFAULT_WEIGHTS,
    EVENT_SPECIFIC_QUESTIONS,
    FALLBACK_PROVENANCE,
    FIRST_TIME_QUESTIONS,
    KEYS,
    calculate_default_weights_from_answers,
    calculate_event_weights,
    get_user_default_weights,
    get_user_event_weights,
    get_user_taste_profile,
    normalize_weights,
    rank_candidates,
    rank_triadic_candidates,
    reconduct_user_weights,
    reset_user_event_weights,
    save_user_default_weights,
    save_user_event_weights,
    save_user_taste_profile,
)
from buddy_matching.taste import TasteProfile
from common.utils import load, load_artifact
from trust.model import never_again, table

router = APIRouter(prefix="/match", tags=["buddy matching"])


@lru_cache
def artifact():
    try:
        return load_artifact("buddy_matching")
    except FileNotFoundError:
        raise HTTPException(503, "model not trained: run python -m buddy_matching.train")


@lru_cache
def tables():
    return load("users"), load("participants"), load("buddy_requests"), load("events")


@lru_cache
def feedback():
    """(user_id -> reliability, pairs to keep apart) from post-event peer feedback."""
    peer = load("peer_feedback")
    return table(peer).reliability, never_again(peer)


# --- Request Models ---

class BuddyQuery(BaseModel):
    user_id: int
    event_id: int
    limit: int = 10
    custom_weights: Optional[Dict[str, float]] = None
    use_event_adjustments: bool = True


class RecommendQuery(BaseModel):
    user_id: Optional[int] = Field(None, alias="userId")
    event_id: Optional[int] = Field(None, alias="eventId")
    limit: int = 10
    viewer_taste: Optional[Dict[str, Any]] = None
    candidate_tastes: Optional[List[Dict[str, Any]]] = None
    event_profile: Optional[Dict[str, Any]] = None
    custom_weights: Optional[Dict[str, float]] = None
    use_event_adjustments: bool = True
    include_details: bool = False

    model_config = ConfigDict(populate_by_name=True)


class FirstTimeAnswersRequest(BaseModel):
    answers: Optional[Dict[str, str]] = None
    custom_weights: Optional[Dict[str, float]] = None


class EventWeightsRequest(BaseModel):
    answers: Optional[Dict[str, str]] = None
    adjustments: Optional[Dict[str, float]] = None
    custom_weights: Optional[Dict[str, float]] = None


class TasteProfileUpdate(BaseModel):
    interests: Optional[str] = None
    bio: Optional[str] = None
    vibe: Optional[str] = None
    energy: Optional[str] = None
    dimensions: Optional[Dict[str, float]] = None
    confidence: Optional[Dict[str, float]] = None
    tags: Optional[Dict[str, float]] = None


# --- Questionnaire & Onboarding Endpoints ---

@router.get("/questionnaire/first-time")
def get_first_time_questionnaire():
    """Returns the 10 trade-off questions for first-time event listers."""
    return {
        "title": "Amigo Buddy Recommendation — First-Time Questionnaire",
        "description": (
            "Determines what matters to you when choosing a buddy. "
            "Generates your persistent default recommendation profile."
        ),
        "dimensions": [
            {"key": "interest", "name": "Interest", "fallback_pct": 30},
            {"key": "event_fit", "name": "Event Fit", "fallback_pct": 25},
            {"key": "social_fit", "name": "Social Fit", "fallback_pct": 15},
            {"key": "prefs", "name": "Preferences", "fallback_pct": 10},
            {"key": "intent_fit", "name": "Intent Fit", "fallback_pct": 10},
            {"key": "location", "name": "Location", "fallback_pct": 5},
            {"key": "availability", "name": "Availability", "fallback_pct": 5},
        ],
        "fallback_weights": DEFAULT_WEIGHTS,
        "fallback_provenance": FALLBACK_PROVENANCE,
        "questions": FIRST_TIME_QUESTIONS,
    }


@router.get("/questionnaire/event-specific")
def get_event_specific_questionnaire():
    """Returns the short 4-question survey for subsequent events."""
    return {
        "title": "Event-Specific Weight Adjustment Questionnaire",
        "description": "Adjust your preferences for a specific event without modifying your persistent default weights.",
        "questions": EVENT_SPECIFIC_QUESTIONS,
    }


# --- Taste Profile Endpoints ---

@router.get("/users/{user_id}/taste-profile")
def get_taste_profile(user_id: int):
    users, _, _, _ = tables()
    if user_id not in set(users.user_id):
        raise HTTPException(404, "unknown user")
    return get_user_taste_profile(user_id, users=users)


@router.post("/users/{user_id}/taste-profile")
def update_taste_profile(user_id: int, req: TasteProfileUpdate):
    users, _, _, _ = tables()
    if user_id not in set(users.user_id):
        raise HTTPException(404, "unknown user")
    current = get_user_taste_profile(user_id, users=users)
    if req.interests is not None:
        current["interests"] = [t.strip() for t in req.interests.split(";") if t.strip()]
        if not req.tags:
            current["tags"] = {t: 1.0 for t in current["interests"]}
    if req.bio is not None:
        current["bio"] = req.bio
    if req.vibe is not None:
        current["vibe"] = req.vibe
    if req.energy is not None:
        current["energy"] = req.energy
    if req.dimensions is not None:
        current["dimensions"] = {str(k).upper(): float(v) for k, v in req.dimensions.items()}
    if req.confidence is not None:
        current["confidence"] = {str(k).upper(): float(v) for k, v in req.confidence.items()}
    if req.tags is not None:
        current["tags"] = {str(k).lower(): float(v) for k, v in req.tags.items()}
    current["userId"] = int(user_id)
    return save_user_taste_profile(user_id, current)


# --- Default Weights Endpoints ---

@router.get("/users/{user_id}/weights")
def get_weights(user_id: int):
    users, _, _, _ = tables()
    if user_id not in set(users.user_id):
        raise HTTPException(404, "unknown user")
    weights, is_fallback = get_user_default_weights(user_id)
    return {
        "user_id": user_id,
        "default_weights": weights,
        "percentages": {k: round(v * 100, 1) for k, v in weights.items()},
        "is_fallback": is_fallback,
        "source": "system_fallback" if is_fallback else "user_questionnaire",
        "provenance": FALLBACK_PROVENANCE if is_fallback else "Generated from user first-time questionnaire",
    }


@router.post("/users/{user_id}/weights/first-time")
def submit_first_time_weights(user_id: int, req: FirstTimeAnswersRequest):
    users, _, _, _ = tables()
    if user_id not in set(users.user_id):
        raise HTTPException(404, "unknown user")
    weights = save_user_default_weights(user_id, answers=req.answers, custom_weights=req.custom_weights)
    return {
        "user_id": user_id,
        "default_weights": weights,
        "percentages": {k: round(v * 100, 1) for k, v in weights.items()},
        "is_fallback": False,
        "message": "Default recommendation weights saved successfully in user RecommendationProfile.",
    }


@router.post("/users/{user_id}/weights/reconduct")
def reconduct_questionnaire(user_id: int):
    users, _, _, _ = tables()
    if user_id not in set(users.user_id):
        raise HTTPException(404, "unknown user")
    reconduct_user_weights(user_id)
    return {
        "user_id": user_id,
        "status": "reconduct_ready",
        "current_weights": DEFAULT_WEIGHTS,
        "message": "User default recommendation profile reset. Ready to reconduct first-time questionnaire.",
    }


# --- Event-Specific Weights Endpoints ---

@router.get("/users/{user_id}/events/{event_id}/weights")
def get_event_weights(user_id: int, event_id: int):
    users, _, _, events = tables()
    if user_id not in set(users.user_id):
        raise HTTPException(404, "unknown user")
    weights, source = get_user_event_weights(user_id, event_id)
    default_w, _ = get_user_default_weights(user_id)
    return {
        "user_id": user_id,
        "event_id": event_id,
        "effective_weights": weights,
        "effective_percentages": {k: round(v * 100, 1) for k, v in weights.items()},
        "default_weights": default_w,
        "source": source,
        "is_adjusted": source == "event_custom",
    }


@router.post("/users/{user_id}/events/{event_id}/weights")
def set_event_weights(user_id: int, event_id: int, req: EventWeightsRequest):
    users, _, _, events = tables()
    if user_id not in set(users.user_id):
        raise HTTPException(404, "unknown user")
    event_w = save_user_event_weights(
        user_id,
        event_id,
        adjustments=req.adjustments,
        answers=req.answers,
        custom_weights=req.custom_weights,
    )
    default_w, _ = get_user_default_weights(user_id)
    return {
        "user_id": user_id,
        "event_id": event_id,
        "event_weights": event_w,
        "event_percentages": {k: round(v * 100, 1) for k, v in event_w.items()},
        "default_weights": default_w,
        "message": "Event-specific weights applied. Stored default weights preserved without change.",
    }


@router.post("/users/{user_id}/events/{event_id}/weights/reset")
@router.delete("/users/{user_id}/events/{event_id}/weights")
def reset_event_weights(user_id: int, event_id: int):
    users, _, _, events = tables()
    if user_id not in set(users.user_id):
        raise HTTPException(404, "unknown user")
    default_w = reset_user_event_weights(user_id, event_id)
    return {
        "user_id": user_id,
        "event_id": event_id,
        "status": "reset",
        "effective_weights": default_w,
        "message": "Event-specific weights reset. Reverted to persistent user default weights.",
    }


# --- Buddy Recommendation Endpoint ---

@router.post("/buddies")
def buddies(q: BuddyQuery):
    users, participants, requests, events = tables()
    if q.user_id not in set(users.user_id):
        raise HTTPException(404, "unknown user")

    # Resolve effective weights
    if q.custom_weights:
        weights = normalize_weights(q.custom_weights)
        weights_source = "query_custom"
        is_fallback = False
    elif q.use_event_adjustments:
        weights, weights_source = get_user_event_weights(q.user_id, q.event_id)
        is_fallback = (weights_source == "system_fallback")
    else:
        weights, is_fallback = get_user_default_weights(q.user_id)
        weights_source = "system_fallback" if is_fallback else "user_default"

    art = artifact()
    candidates = rank_candidates(
        q.user_id,
        q.event_id,
        users,
        participants,
        requests,
        art,
        q.limit,
        *feedback(),
        weights=weights,
        events=events,
    )

    return {
        "request_id": str(uuid.uuid4()),
        "model_version": art["model_version"],
        "weights_used": weights,
        "weights_source": weights_source,
        "is_fallback": is_fallback,
        "fallback_provenance": FALLBACK_PROVENANCE if is_fallback else None,
        "candidates": candidates,
    }


@router.post("/recommend")
def recommend(q: RecommendQuery):
    """Triadic (user, candidate, event) recommendation endpoint.

    Matches the future ML handoff schema:
      {
        "userId": 1,
        "eventId": 123,
        "recommendations": [
          {"candidateUserId": 42, "score": 0.91},
          {"candidateUserId": 17, "score": 0.86}
        ]
      }
    """
    uid = q.user_id
    eid = q.event_id
    if uid is None or eid is None:
        raise HTTPException(422, "userId and eventId are required")

    users, participants, requests, events = tables()
    if uid not in set(users.user_id):
        raise HTTPException(404, "unknown user")

    # Resolve effective weights
    if q.custom_weights:
        weights = normalize_weights(q.custom_weights)
    elif q.use_event_adjustments:
        weights, _ = get_user_event_weights(uid, eid)
    else:
        weights, _ = get_user_default_weights(uid)

    # Parse optional payloads if supplied directly
    v_taste = TasteProfile.from_dict(q.viewer_taste) if q.viewer_taste else None
    c_tastes = {int(t.get("userId", t.get("user_id", 0))): TasteProfile.from_dict(t)
                for t in q.candidate_tastes} if q.candidate_tastes else None
    ev_prof = EventProfile.from_dict(q.event_profile) if q.event_profile else None

    art = artifact()
    res = rank_triadic_candidates(
        viewer_id=uid,
        event_id=eid,
        users=users,
        participants=participants,
        requests=requests,
        art=art,
        limit=q.limit,
        reliability=feedback()[0],
        avoid=feedback()[1],
        weights=weights,
        events=events,
        viewer_taste=v_taste,
        candidate_tastes=c_tastes,
        event_profile=ev_prof,
    )

    if q.include_details:
        return res

    return {
        "userId": uid,
        "eventId": eid,
        "recommendations": res["recommendations"],
    }

