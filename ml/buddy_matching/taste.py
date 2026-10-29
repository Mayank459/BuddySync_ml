"""Amigo Taste Profile: Ingestion, Confidence Semantics & Compatibility.

Implements backend handoff semantics:
  - 0.0–1.0 = preference score.
  - 0.50 + confidence 0 = unknown, not neutral preference.
  - confidence indicates reliability of the corresponding taste score.
  - tags are explicit user interests with intensity weights.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

# The 21 standard taste dimensions provided by the backend
STANDARD_DIMENSIONS = [
    "MUSIC", "NIGHTLIFE", "PARTY", "SPORTS", "MOVIES", "COMEDY", "THEATRE",
    "FOOD", "ARTS", "TECHNOLOGY", "SOCIAL", "TRAVEL", "OUTDOOR", "CREATIVE",
    "ACADEMIC", "GROUP_PREFERENCE", "SOLO_COMFORT", "CROWD_COMFORT",
    "EXPLORATION", "LIVE_EVENT", "COMPETITIVE"
]

# Domain clusters for high-level semantic grouping
DIMENSION_CLUSTERS = {
    "entertainment": ["MUSIC", "NIGHTLIFE", "PARTY", "MOVIES", "COMEDY", "THEATRE", "FOOD", "ARTS"],
    "active": ["SPORTS", "OUTDOOR", "TRAVEL", "COMPETITIVE"],
    "intellectual": ["TECHNOLOGY", "ACADEMIC", "CREATIVE"],
    "social_style": ["SOCIAL", "GROUP_PREFERENCE", "SOLO_COMFORT", "CROWD_COMFORT", "EXPLORATION", "LIVE_EVENT"],
}

# Tag to dimension mappings for fallback/enrichment
TAG_DIMENSION_MAP = {
    "bollywood music": ("MUSIC", 0.9), "indie music": ("MUSIC", 0.9), "edm": ("MUSIC", 0.9),
    "rock": ("MUSIC", 0.9), "hip hop": ("MUSIC", 0.9), "sufi": ("MUSIC", 0.9), "music festivals": ("MUSIC", 0.95),
    "stand-up comedy": ("COMEDY", 0.95), "improv": ("COMEDY", 0.9),
    "cricket": ("SPORTS", 0.95), "football": ("SPORTS", 0.95), "kabaddi": ("SPORTS", 0.9),
    "bollywood films": ("MOVIES", 0.9), "hollywood films": ("MOVIES", 0.9), "anime": ("MOVIES", 0.85),
    "theatre": ("THEATRE", 0.95), "musicals": ("THEATRE", 0.95),
    "photography": ("CREATIVE", 0.85), "pottery": ("ARTS", 0.9), "food festivals": ("FOOD", 0.95),
    "coding": ("TECHNOLOGY", 0.95), "tech": ("TECHNOLOGY", 0.95), "startups": ("TECHNOLOGY", 0.85), "ai": ("TECHNOLOGY", 0.95),
}


@dataclass
class TasteProfile:
    user_id: int
    dimensions: Dict[str, float] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TasteProfile:
        """Parses backend Taste Profile JSON.

        Accepts either camelCase ('userId') or snake_case ('user_id').
        """
        uid = int(data.get("userId", data.get("user_id", 0)))
        raw_dims = data.get("dimensions", {})
        # Normalize dimension keys to uppercase
        dimensions = {str(k).upper(): float(v) for k, v in raw_dims.items()}
        raw_conf = data.get("confidence", {})
        confidence = {str(k).upper(): float(v) for k, v in raw_conf.items()}
        raw_tags = data.get("tags", {})
        if isinstance(raw_tags, list):
            tags = {str(t).lower(): 1.0 for t in raw_tags}
        else:
            tags = {str(k).lower(): float(v) for k, v in raw_tags.items()}

        return cls(user_id=uid, dimensions=dimensions, confidence=confidence, tags=tags)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes back to standard backend JSON representation."""
        return {
            "userId": self.user_id,
            "dimensions": self.dimensions,
            "confidence": self.confidence,
            "tags": self.tags,
        }

    def get_dimension(self, dim: str) -> Tuple[float, float, bool]:
        """Returns (score, confidence, is_known).

        Semantics: 0.50 with confidence 0.0 is unknown, not neutral.
        """
        d_key = dim.upper()
        score = float(self.dimensions.get(d_key, 0.50))
        # If not listed in confidence dict:
        # If score is exactly 0.50, confidence defaults to 0.0 (unknown).
        # If score deviates from 0.50, confidence defaults to 0.50 unless explicitly set.
        if d_key in self.confidence:
            conf = float(self.confidence[d_key])
        else:
            conf = 0.0 if abs(score - 0.50) < 1e-4 else 0.50

        is_known = conf > 0.0
        return score, conf, is_known

    def confidence_weighted_similarity(self, other: TasteProfile) -> Dict[str, float]:
        """Computes confidence-weighted similarity between two taste profiles.

        Weights each dimension by joint confidence (c_u * c_v). Unknown dimensions
        (confidence = 0) do not penalize the match.
        """
        all_dims = set(self.dimensions.keys()) | set(other.dimensions.keys()) | set(STANDARD_DIMENSIONS)
        weighted_sim_sum = 0.0
        total_weight = 0.0
        active_dimension_count = 0

        for dim in all_dims:
            s1, c1, k1 = self.get_dimension(dim)
            s2, c2, k2 = other.get_dimension(dim)

            if not k1 or not k2:
                # One or both users have unknown preference for this dimension: skip.
                continue

            # Joint confidence = product of reliability scores
            joint_conf = c1 * c2
            if joint_conf > 0.0:
                dim_sim = 1.0 - abs(s1 - s2)
                weighted_sim_sum += joint_conf * dim_sim
                total_weight += joint_conf
                active_dimension_count += 1

        if total_weight > 0.0:
            overall_sim = float(weighted_sim_sum / total_weight)
        else:
            overall_sim = 0.50  # Neutral prior if no mutual dimensions known

        # Coverage: share of standard dimensions with mutual evidence
        coverage = float(active_dimension_count / max(1, len(STANDARD_DIMENSIONS)))

        return {
            "taste_similarity": round(overall_sim, 4),
            "joint_confidence": round(float(total_weight), 4),
            "coverage": round(coverage, 4),
            "known_shared_dimensions": active_dimension_count,
        }

    def tag_overlap(self, other: TasteProfile) -> Dict[str, Any]:
        """Computes explicit user interest tag overlap (weighted cosine + Jaccard)."""
        tags_a = {k.lower(): float(v) for k, v in self.tags.items()}
        tags_b = {k.lower(): float(v) for k, v in other.tags.items()}

        common_keys = set(tags_a.keys()) & set(tags_b.keys())
        all_keys = set(tags_a.keys()) | set(tags_b.keys())

        if not all_keys:
            return {"tag_overlap_score": 0.50, "shared_tags": [], "jaccard": 0.0}

        # Weighted dot product & cosine
        dot = sum(tags_a[k] * tags_b[k] for k in common_keys)
        norm_a = np.sqrt(sum(v ** 2 for v in tags_a.values()))
        norm_b = np.sqrt(sum(v ** 2 for v in tags_b.values()))
        cosine = float(dot / (norm_a * norm_b)) if (norm_a > 0 and norm_b > 0) else 0.0
        jaccard = float(len(common_keys) / len(all_keys))

        score = float(0.6 * cosine + 0.4 * jaccard)
        return {
            "tag_overlap_score": round(score, 4),
            "shared_tags": sorted(common_keys),
            "jaccard": round(jaccard, 4),
        }

    def social_compatibility(self, other: TasteProfile) -> float:
        """Evaluates social style compatibility from taste dimensions."""
        # Social dimensions: SOCIAL, GROUP_PREFERENCE, SOLO_COMFORT, CROWD_COMFORT
        soc_dims = ["SOCIAL", "GROUP_PREFERENCE", "SOLO_COMFORT", "CROWD_COMFORT"]
        sims = []
        for d in soc_dims:
            s1, c1, k1 = self.get_dimension(d)
            s2, c2, k2 = other.get_dimension(d)
            if k1 and k2:
                sims.append(1.0 - abs(s1 - s2))
            else:
                sims.append(0.75)  # reasonable default when unmeasured
        return float(np.mean(sims)) if sims else 0.50

    @classmethod
    def from_user_row(cls, row: pd.Series | Dict[str, Any], user_id: int | None = None) -> TasteProfile:
        """Synthesizes a TasteProfile from a legacy user database row."""
        if user_id is not None:
            uid = int(user_id)
        elif "user_id" in row:
            uid = int(row["user_id"])
        elif hasattr(row, "name") and row.name is not None:
            try:
                uid = int(row.name)
            except (ValueError, TypeError):
                uid = 0
        else:
            uid = 0
        interests_str = str(row.get("interests", "")).lower()
        interest_list = [t.strip() for t in interests_str.split(";") if t.strip()]

        tags = {t: 1.0 for t in interest_list}
        dimensions: Dict[str, float] = {d: 0.50 for d in STANDARD_DIMENSIONS}
        confidence: Dict[str, float] = {}

        # Infer dimensions from explicit tags
        for t in interest_list:
            if t in TAG_DIMENSION_MAP:
                dim, score = TAG_DIMENSION_MAP[t]
                dimensions[dim] = max(dimensions[dim], score)
                confidence[dim] = max(confidence.get(dim, 0.0), 0.85)

        # Map behavioral attributes
        energy = str(row.get("energy", "ambivert")).lower()
        if energy == "extrovert":
            dimensions["SOCIAL"] = 0.85
            confidence["SOCIAL"] = 0.90
        elif energy == "introvert":
            dimensions["SOCIAL"] = 0.30
            dimensions["SOLO_COMFORT"] = 0.85
            confidence["SOCIAL"] = 0.90
            confidence["SOLO_COMFORT"] = 0.85
        else:
            dimensions["SOCIAL"] = 0.60
            confidence["SOCIAL"] = 0.70

        vibe = str(row.get("vibe", "chill")).lower()
        if vibe == "front_row":
            dimensions["LIVE_EVENT"] = 0.85
            dimensions["CROWD_COMFORT"] = 0.80
            confidence["LIVE_EVENT"] = 0.80
            confidence["CROWD_COMFORT"] = 0.75
        elif vibe == "chill":
            dimensions["LIVE_EVENT"] = 0.60
            dimensions["CROWD_COMFORT"] = 0.50
            confidence["LIVE_EVENT"] = 0.60
            confidence["CROWD_COMFORT"] = 0.60

        group_pref = str(row.get("group_pref", "either")).lower()
        if group_pref == "group":
            dimensions["GROUP_PREFERENCE"] = 0.85
            confidence["GROUP_PREFERENCE"] = 0.85
        elif group_pref == "duo":
            dimensions["GROUP_PREFERENCE"] = 0.40
            confidence["GROUP_PREFERENCE"] = 0.80

        after = str(row.get("after", "home")).lower()
        if after == "party":
            dimensions["PARTY"] = 0.80
            dimensions["NIGHTLIFE"] = 0.80
            confidence["PARTY"] = 0.80
            confidence["NIGHTLIFE"] = 0.80
        elif after == "food":
            dimensions["FOOD"] = 0.85
            confidence["FOOD"] = 0.80

        return cls(user_id=uid, dimensions=dimensions, confidence=confidence, tags=tags)
