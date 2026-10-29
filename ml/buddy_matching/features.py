"""Contextual Pairwise Feature Extraction for (user, candidate, event).

Builds the feature vector from:
  1. Taste dimensions (confidence-weighted)
  2. Taste confidence reliability
  3. Explicit interest/tag overlap
  4. Event features & triadic context
  5. User recommendation preferences
  6. Event-specific preferences
  7. Location and availability
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from buddy_matching.event_profile import EventProfile
from buddy_matching.taste import TasteProfile
from common.utils import haversine_km


@dataclass
class TriadicFeatures:
    viewer_id: int
    candidate_id: int
    event_id: int

    # Taste features
    taste_similarity: float = 0.50
    joint_confidence: float = 0.0
    coverage: float = 0.0
    tag_overlap_score: float = 0.0
    shared_tags: List[str] = field(default_factory=list)

    # Event context features
    triadic_event_fit: float = 0.50
    viewer_event_affinity: float = 0.50
    candidate_event_affinity: float = 0.50

    # Social & Behavioral features
    social_fit: float = 0.50
    pref_uv: float = 0.50
    pref_vu: float = 0.50
    intent_fit: float = 0.50

    # Physical / Schedule features
    location_score: float = 0.50
    distance_km: float = 0.0
    availability_score: float = 0.50

    def to_dimension_scores(self) -> Dict[str, float]:
        """Maps extracted signals into the 7 standard recommendation dimensions."""
        # 1. Interest: blended confidence-weighted taste similarity + explicit tag overlap
        interest_score = float(0.50 * self.taste_similarity + 0.50 * self.tag_overlap_score)
        return {
            "interest": round(interest_score, 4),
            "event_fit": round(self.triadic_event_fit, 4),
            "social_fit": round(self.social_fit, 4),
            "pref_uv": round(self.pref_uv, 4),
            "pref_vu": round(self.pref_vu, 4),
            "intent_fit": round(self.intent_fit, 4),
            "location": round(self.location_score, 4),
            "availability": round(self.availability_score, 4),
        }

    def to_vector(self) -> np.ndarray:
        """Returns 8-element numpy vector aligned with the recommendation model scaler."""
        scores = self.to_dimension_scores()
        return np.array([
            scores["interest"],
            scores["event_fit"],
            scores["social_fit"],
            scores["pref_uv"],
            scores["pref_vu"],
            scores["intent_fit"],
            scores["location"],
            scores["availability"],
        ], dtype=float)


def extract_triadic_pair_features(
    viewer_taste: TasteProfile,
    candidate_taste: TasteProfile,
    event: EventProfile,
    viewer_row: Optional[pd.Series | Dict[str, Any]] = None,
    candidate_row: Optional[pd.Series | Dict[str, Any]] = None,
) -> TriadicFeatures:
    """Extracts pairwise contextual features for (viewer, candidate, event)."""
    # 1. Taste dimension similarity with confidence semantics
    taste_res = viewer_taste.confidence_weighted_similarity(candidate_taste)
    tag_res = viewer_taste.tag_overlap(candidate_taste)

    # 2. Triadic event synergy
    event_res = event.triadic_synergy(viewer_taste, candidate_taste)

    # 3. Social fit
    soc_from_taste = viewer_taste.social_compatibility(candidate_taste)

    # Behavioral signals from user rows if available
    v_row = viewer_row if viewer_row is not None else {}
    c_row = candidate_row if candidate_row is not None else {}

    # Soft preferences (language, group size, verified, age)
    lang_match = 1.0
    v_langs = set(str(v_row.get("languages", "english")).split(";"))
    c_langs = set(str(c_row.get("languages", "english")).split(";"))
    if v_langs and c_langs:
        lang_match = 1.0 if bool(v_langs & c_langs) else 0.0

    group_match = 1.0
    vg = str(v_row.get("group_pref", "either"))
    cg = str(c_row.get("group_pref", "either"))
    if vg != "either" and cg != "either" and vg != cg:
        group_match = 0.4

    c_verified = 1.0 if c_row.get("verified", True) else 0.0
    v_verified = 1.0 if v_row.get("verified", True) else 0.0

    pref_uv = float((lang_match + group_match + c_verified + 1.0) / 4.0)
    pref_vu = float((lang_match + group_match + v_verified + 1.0) / 4.0)

    # Intent fit (after-event plans: party, food, home)
    v_after = str(v_row.get("after", "food")).lower()
    c_after = str(c_row.get("after", "food")).lower()
    if v_after == c_after:
        intent_fit = 1.0
    elif (v_after in ("party", "food")) and (c_after in ("party", "food")):
        intent_fit = 0.75
    else:
        intent_fit = 0.30

    # Energy & vibe fit
    v_vibe = str(v_row.get("vibe", "chill")).lower()
    c_vibe = str(c_row.get("vibe", "chill")).lower()
    vibe_match = 1.0 if v_vibe == c_vibe else (0.85 if "depends" in (v_vibe, c_vibe) else 0.40)

    social_fit = float(0.50 * soc_from_taste + 0.50 * vibe_match)

    # Distance & Location
    v_lat = float(v_row.get("lat", event.venue_lat or 19.0))
    v_lng = float(v_row.get("lng", event.venue_lng or 72.8))
    c_lat = float(c_row.get("lat", event.venue_lat or 19.0))
    c_lng = float(c_row.get("lng", event.venue_lng or 72.8))
    dist = haversine_km(v_lat, v_lng, c_lat, c_lng)
    location_score = float(np.exp(-dist / 5.0))

    # Availability & arrival
    v_arr = str(v_row.get("arrive", "together")).lower()
    c_arr = str(c_row.get("arrive", "together")).lower()
    v_punc = str(v_row.get("punctuality", "on_time")).lower()
    c_punc = str(c_row.get("punctuality", "on_time")).lower()
    arr_fit = 1.0 if v_arr == c_arr else 0.50
    punc_fit = 1.0 if v_punc == c_punc else 0.70
    availability_score = float(0.50 * arr_fit + 0.50 * punc_fit)

    return TriadicFeatures(
        viewer_id=viewer_taste.user_id,
        candidate_id=candidate_taste.user_id,
        event_id=event.event_id,
        taste_similarity=taste_res["taste_similarity"],
        joint_confidence=taste_res["joint_confidence"],
        coverage=taste_res["coverage"],
        tag_overlap_score=tag_res["tag_overlap_score"],
        shared_tags=tag_res["shared_tags"],
        triadic_event_fit=event_res["triadic_event_fit"],
        viewer_event_affinity=event_res["user_a_affinity"],
        candidate_event_affinity=event_res["user_b_affinity"],
        social_fit=social_fit,
        pref_uv=pref_uv,
        pref_vu=pref_vu,
        intent_fit=intent_fit,
        location_score=location_score,
        distance_km=round(dist, 1),
        availability_score=availability_score,
    )
