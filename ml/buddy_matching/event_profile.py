"""Amigo Event Profile: Features, User-Event Affinity & Triadic Synergy.

Ranks the triadic context:
    (user, candidate, event)
            ↓
    pairwise & context features
            ↓
    compatibility score
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd

from buddy_matching.taste import TasteProfile

CATEGORY_DIMENSION_MAP = {
    "music": {"MUSIC": 1.0, "LIVE_EVENT": 0.85},
    "festival": {"LIVE_EVENT": 0.95, "PARTY": 0.80, "CROWD_COMFORT": 0.75},
    "theatre": {"THEATRE": 1.0, "ARTS": 0.85, "CREATIVE": 0.75},
    "comedy": {"COMEDY": 1.0, "SOCIAL": 0.70},
    "sports": {"SPORTS": 1.0, "COMPETITIVE": 0.85, "LIVE_EVENT": 0.70},
    "movies": {"MOVIES": 1.0},
    "technology": {"TECHNOLOGY": 1.0, "ACADEMIC": 0.70},
    "food": {"FOOD": 1.0, "EXPLORATION": 0.75},
    "nightlife": {"NIGHTLIFE": 1.0, "PARTY": 0.90},
    "outdoor": {"OUTDOOR": 1.0, "TRAVEL": 0.80},
}


@dataclass
class EventProfile:
    event_id: int
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    title: str = ""
    description: str = ""
    venue_lat: float = 0.0
    venue_lng: float = 0.0
    price_band: int = 2
    hotness: float = 0.5
    target_dimensions: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EventProfile:
        eid = int(data.get("eventId", data.get("event_id", 0)))
        cat = str(data.get("category", "general")).lower()
        raw_tags = data.get("tags", [])
        if isinstance(raw_tags, str):
            tags = [t.strip().lower() for t in raw_tags.split(";") if t.strip()]
        else:
            tags = [str(t).strip().lower() for t in raw_tags]

        title = str(data.get("title", ""))
        desc = str(data.get("description", ""))
        lat = float(data.get("venue_lat", data.get("venueLat", 0.0)))
        lng = float(data.get("venue_lng", data.get("venueLng", 0.0)))
        price = int(data.get("price_band", data.get("priceBand", 2)))
        hotness = float(data.get("hotness", 0.5))

        target_dims = dict(CATEGORY_DIMENSION_MAP.get(cat, {"LIVE_EVENT": 0.6}))
        for t in tags:
            t_clean = t.lower()
            if "music" in t_clean or "rock" in t_clean or "edm" in t_clean:
                target_dims["MUSIC"] = 0.95
            if "food" in t_clean:
                target_dims["FOOD"] = 0.95
            if "comedy" in t_clean:
                target_dims["COMEDY"] = 0.95
            if "theatre" in t_clean or "musical" in t_clean:
                target_dims["THEATRE"] = 0.95
            if "ai" in t_clean or "tech" in t_clean or "code" in t_clean:
                target_dims["TECHNOLOGY"] = 0.95
            if "sports" in t_clean or "cricket" in t_clean or "football" in t_clean:
                target_dims["SPORTS"] = 0.95

        return cls(
            event_id=eid,
            category=cat,
            tags=tags,
            title=title,
            description=desc,
            venue_lat=lat,
            venue_lng=lng,
            price_band=price,
            hotness=hotness,
            target_dimensions=target_dims,
        )

    @classmethod
    def from_event_row(cls, row: pd.Series | Dict[str, Any]) -> EventProfile:
        return cls.from_dict(dict(row))

    def user_affinity(self, user: TasteProfile) -> float:
        """Evaluates user affinity to this event using taste dimensions and explicit tags."""
        # 1. Dimension alignment
        dim_scores = []
        for dim, weight in self.target_dimensions.items():
            score, conf, is_known = user.get_dimension(dim)
            if is_known:
                dim_scores.append(score * weight)
            else:
                dim_scores.append(0.50 * weight)  # neutral prior if unknown

        dim_fit = float(np.mean(dim_scores)) if dim_scores else 0.50

        # 2. Tag alignment
        ev_tags_set = set(self.tags) | {self.category}
        user_tags_set = set(user.tags.keys())
        overlap = ev_tags_set & user_tags_set
        tag_fit = 1.0 if overlap else 0.40

        # 3. Combine
        return float(0.55 * dim_fit + 0.45 * tag_fit)

    def triadic_synergy(self, user_a: TasteProfile, user_b: TasteProfile) -> Dict[str, float]:
        """Evaluates mutual synergy for pair (user_a, user_b) attending this event."""
        aff_a = self.user_affinity(user_a)
        aff_b = self.user_affinity(user_b)

        # Mutual excitement is high only when BOTH users have high affinity for the event
        mutual_base = float(np.sqrt(aff_a * aff_b))
        balance_penalty = float(abs(aff_a - aff_b) * 0.3)
        synergy_score = float(np.clip(mutual_base - balance_penalty, 0.0, 1.0))

        return {
            "triadic_event_fit": round(synergy_score, 4),
            "user_a_affinity": round(aff_a, 4),
            "user_b_affinity": round(aff_b, 4),
        }
