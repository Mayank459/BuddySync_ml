"""Buddy matching: 7-dimension reciprocal score + profile weighting + ranking.

Shared by train.py, api.py, and clan formation.
Components per pair (viewer u, candidate v):
  1. interest     0.6 * tf-idf cosine(profile text) + 0.4 * jaccard(interest tags)
  2. event_fit    event suitability from user taste + event category/tags + budget alignment
  3. social_fit   compatibility in social behaviour and interaction style (energy & vibe)
  4. prefs        directional soft preferences (language, group size, verified, age)
  5. intent_fit   purpose of going and after-event plans (party, food, home)
  6. location     proximity: exp(-km / 5)
  7. availability schedules & arrival alignment (arrive together/venue, punctuality)

Workflow:
  - User setup creates a persistent Taste Profile (general interest fingerprint).
  - First-time event listing triggers a 10-question trade-off questionnaire -> user Default Weights.
  - System fallback weights (30/25/15/10/10/5/5) derived from team survey + AI optimization.
  - Subsequent events allow event-specific adjustments without overwriting user defaults.
  - Reconduct option allows user to retake the first-time questionnaire.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from common.utils import DATA, haversine_km
from trust.model import NEUTRAL
from buddy_matching.taste import TasteProfile, STANDARD_DIMENSIONS
from buddy_matching.event_profile import EventProfile
from buddy_matching.features import TriadicFeatures, extract_triadic_pair_features

KEYS = ["interest", "event_fit", "social_fit", "prefs", "intent_fit", "location", "availability"]
DEFAULT_WEIGHTS = {
    "interest": 0.30,
    "event_fit": 0.25,
    "social_fit": 0.15,
    "prefs": 0.10,
    "intent_fit": 0.10,
    "location": 0.05,
    "availability": 0.05,
}
FALLBACK_WEIGHTS = DEFAULT_WEIGHTS.copy()
FALLBACK_PROVENANCE = (
    "Baseline system fallback weights established via internal survey and AI weighting optimization analysis."
)

QUIZ_KEYS = ["vibe", "arrive", "after", "punctuality"]
INTEREST_GROUPS = {
    "Music": {"bollywood music", "indie music", "edm", "rock", "hip hop", "sufi", "music festivals"},
    "Comedy": {"stand-up comedy", "improv"},
    "Sports": {"cricket", "football", "kabaddi"},
    "Movies": {"bollywood films", "hollywood films", "anime"},
    "Theatre": {"theatre", "musicals"},
    "Photography": {"photography"},
    "Arts & crafts": {"pottery"},
    "Food": {"food festivals"},
    "Technology": {"coding", "tech", "startups", "ai"},
}
GROUP_OF = {t: g for g, tags in INTEREST_GROUPS.items() for t in tags}

# Persistent data store paths
PROFILES_FILE = DATA / "recommendation_profiles.json"
EVENT_WEIGHTS_FILE = DATA / "event_weights.json"
TASTE_PROFILES_FILE = DATA / "taste_profiles.json"


# --- First-Time Questionnaire (10 trade-off questions) ---
FIRST_TIME_QUESTIONS = [
    {
        "id": "q1",
        "dimension": "Common Interests vs Location",
        "question": "Which would you prefer?",
        "options": [
            {"id": "A", "text": "Someone with very similar interests but somewhat farther away.", "impact": {"interest": 0.06, "location": -0.03}},
            {"id": "B", "text": "Someone nearby but with very different interests.", "impact": {"location": 0.06, "interest": -0.03}},
        ],
    },
    {
        "id": "q2",
        "dimension": "Event Compatibility vs Social Fit",
        "question": "For an event you are excited about, which buddy would you prefer?",
        "options": [
            {"id": "A", "text": "Someone genuinely interested in the same event.", "impact": {"event_fit": 0.06, "social_fit": -0.03}},
            {"id": "B", "text": "Someone who matches your personality well but is not particularly interested in the event.", "impact": {"social_fit": 0.06, "event_fit": -0.03}},
        ],
    },
    {
        "id": "q3",
        "dimension": "Social Behaviour",
        "question": "Which would you prefer at an event?",
        "options": [
            {"id": "A", "text": "Energetic person who likes talking and meeting people.", "impact": {"social_fit": 0.05, "intent_fit": 0.02}},
            {"id": "B", "text": "Quieter person who prefers staying with a small group.", "impact": {"social_fit": 0.05, "prefs": 0.02}},
        ],
    },
    {
        "id": "q4",
        "dimension": "Event Experience",
        "question": "What matters more?",
        "options": [
            {"id": "A", "text": "Someone who enjoys the same activities.", "impact": {"event_fit": 0.05, "prefs": -0.02}},
            {"id": "B", "text": "Someone who behaves similarly during the event.", "impact": {"prefs": 0.05, "event_fit": -0.02}},
        ],
    },
    {
        "id": "q5",
        "dimension": "Purpose of Going",
        "question": "Which is closer to your ideal?",
        "options": [
            {"id": "A", "text": "Attend, enjoy the event and leave.", "impact": {"intent_fit": 0.04, "social_fit": -0.02}},
            {"id": "B", "text": "Attend, meet people and spend more time together before/after.", "impact": {"intent_fit": 0.06, "social_fit": 0.03}},
        ],
    },
    {
        "id": "q6",
        "dimension": "Distance",
        "question": "Which would you choose?",
        "options": [
            {"id": "A", "text": "Highly compatible buddy 15–20 km away.", "impact": {"interest": 0.03, "event_fit": 0.03, "location": -0.04}},
            {"id": "B", "text": "Reasonably compatible buddy 2–3 km away.", "impact": {"location": 0.06, "interest": -0.02, "event_fit": -0.02}},
        ],
    },
    {
        "id": "q7",
        "dimension": "Timing & Availability",
        "question": "Which would you prefer?",
        "options": [
            {"id": "A", "text": "Highly compatible buddy with a slightly different schedule.", "impact": {"availability": -0.03, "interest": 0.02, "event_fit": 0.02}},
            {"id": "B", "text": "Slightly less compatible buddy available at exactly the same time.", "impact": {"availability": 0.06, "interest": -0.02}},
        ],
    },
    {
        "id": "q8",
        "dimension": "Arrival Behaviour",
        "question": "Which would you prefer?",
        "options": [
            {"id": "A", "text": "Someone who arrives early/on time.", "impact": {"prefs": 0.04, "availability": 0.03}},
            {"id": "B", "text": "Someone flexible about arrival time.", "impact": {"prefs": -0.02, "availability": -0.02, "social_fit": 0.03}},
        ],
    },
    {
        "id": "q9",
        "dimension": "After-Event Behaviour",
        "question": "After an event, what would you prefer?",
        "options": [
            {"id": "A", "text": "Go home soon after.", "impact": {"intent_fit": 0.03, "prefs": 0.02}},
            {"id": "B", "text": "Continue hanging out / explore / eat together.", "impact": {"intent_fit": 0.06, "prefs": 0.02, "social_fit": 0.02}},
        ],
    },
    {
        "id": "q10",
        "dimension": "Overall Trade-off",
        "question": "Which would you choose?",
        "options": [
            {"id": "A", "text": "Very similar interests and event preferences but farther away.", "impact": {"interest": 0.05, "prefs": 0.03, "location": -0.03}},
            {"id": "B", "text": "Nearby and socially compatible but fewer common interests.", "impact": {"location": 0.05, "social_fit": 0.04, "interest": -0.04}},
        ],
    },
]

# --- Event-Specific Questionnaire (3-4 questions for later events) ---
EVENT_SPECIFIC_QUESTIONS = [
    {
        "id": "eq1",
        "question": "What matters more for this event?",
        "options": [
            {"id": "same_interest", "text": "Same event/interest compatibility", "impact": {"event_fit": 0.04, "interest": 0.03, "social_fit": -0.03}},
            {"id": "social_vibe", "text": "Similar social vibe", "impact": {"social_fit": 0.05, "event_fit": -0.02}},
        ],
    },
    {
        "id": "eq2",
        "question": "How important is meeting someone nearby?",
        "options": [
            {"id": "not_important", "text": "Not important", "impact": {"location": -0.03, "interest": 0.02}},
            {"id": "somewhat_important", "text": "Somewhat important", "impact": {"location": 0.02}},
            {"id": "very_important", "text": "Very important", "impact": {"location": 0.06, "interest": -0.03}},
        ],
    },
    {
        "id": "eq3",
        "question": "What do you want from this event?",
        "options": [
            {"id": "just_watch", "text": "Just watch", "impact": {"intent_fit": 0.02, "social_fit": -0.02}},
            {"id": "watch_and_meet", "text": "Watch + meet people", "impact": {"intent_fit": 0.04, "social_fit": 0.03}},
            {"id": "watch_and_hangout", "text": "Watch + hangout afterwards", "impact": {"intent_fit": 0.06, "social_fit": 0.02, "prefs": 0.02}},
        ],
    },
    {
        "id": "eq4",
        "question": "How important is arriving/leaving together?",
        "options": [
            {"id": "not_important", "text": "Not important", "impact": {"availability": -0.03, "prefs": -0.02}},
            {"id": "somewhat_important", "text": "Somewhat important", "impact": {"availability": 0.02, "prefs": 0.01}},
            {"id": "very_important", "text": "Very important", "impact": {"availability": 0.05, "prefs": 0.03}},
        ],
    },
]


def normalize_weights(w: dict, decimals: int = 4) -> dict:
    """Normalize any dict of weights to ensure all KEYS exist, >= 0.01, and sum exactly to 1.0."""
    filled = {k: max(0.01, float(w.get(k, DEFAULT_WEIGHTS.get(k, 0.1)))) for k in KEYS}
    total = sum(filled.values())
    norm = {k: round(v / total, decimals) for k, v in filled.items()}
    # Exact adjustment on largest weight so sum is strictly 1.0
    diff = round(1.0 - sum(norm.values()), decimals)
    if diff != 0:
        max_k = max(norm, key=norm.get)
        norm[max_k] = round(norm[max_k] + diff, decimals)
    return norm


def calculate_default_weights_from_answers(answers: dict) -> dict:
    """Calculates user default weights from first-time questionnaire answers.

    Answers can be {"q1": "A", "q2": "B", ...}.
    """
    scores = {k: float(DEFAULT_WEIGHTS[k]) for k in KEYS}
    for q in FIRST_TIME_QUESTIONS:
        qid = q["id"]
        chosen = answers.get(qid)
        if not chosen:
            continue
        opt = next((o for o in q["options"] if o["id"] == chosen), None)
        if opt and "impact" in opt:
            for dim, delta in opt["impact"].items():
                if dim in scores:
                    scores[dim] += delta
    # Ensure minimum weight bound so no dimension is zeroed
    bounded = {k: max(0.02, v) for k, v in scores.items()}
    return normalize_weights(bounded)


def calculate_event_weights(default_weights: dict, answers: dict = None,
                            adjustments: dict = None, custom_weights: dict = None) -> dict:
    """Derive event-specific weights without mutating default_weights.

    Supports short event answers, delta adjustments, or explicit custom weights.
    """
    if custom_weights:
        return normalize_weights(custom_weights)

    base = {k: float(default_weights.get(k, DEFAULT_WEIGHTS[k])) for k in KEYS}
    if adjustments:
        for k, delta in adjustments.items():
            if k in base:
                base[k] += float(delta)

    if answers:
        for q in EVENT_SPECIFIC_QUESTIONS:
            qid = q["id"]
            chosen = answers.get(qid)
            if not chosen:
                continue
            opt = next((o for o in q["options"] if o["id"] == chosen), None)
            if opt and "impact" in opt:
                for dim, delta in opt["impact"].items():
                    if dim in base:
                        base[dim] += delta

    bounded = {k: max(0.01, v) for k, v in base.items()}
    return normalize_weights(bounded)


# --- Storage Helpers (JSON backed) ---
def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_user_default_weights(user_id: int) -> tuple[dict, bool]:
    """Returns (default_weights, is_fallback).

    Returns user's saved questionnaire weights if available; otherwise system fallback.
    """
    data = _load_json(PROFILES_FILE)
    uid_str = str(user_id)
    if uid_str in data and "weights" in data[uid_str]:
        return normalize_weights(data[uid_str]["weights"]), False
    return DEFAULT_WEIGHTS.copy(), True


def save_user_default_weights(user_id: int, answers: dict = None, custom_weights: dict = None) -> dict:
    """Calculates and saves user default weights (RecommendationProfile)."""
    if custom_weights:
        weights = normalize_weights(custom_weights)
    elif answers:
        weights = calculate_default_weights_from_answers(answers)
    else:
        weights = DEFAULT_WEIGHTS.copy()

    data = _load_json(PROFILES_FILE)
    data[str(user_id)] = {
        "weights": weights,
        "answers": answers or {},
        "updated_at": str(pd.Timestamp.now()),
    }
    _save_json(PROFILES_FILE, data)
    return weights


def reconduct_user_weights(user_id: int) -> dict:
    """Resets user's recommendation profile so questionnaire can be retaken."""
    data = _load_json(PROFILES_FILE)
    uid_str = str(user_id)
    if uid_str in data:
        del data[uid_str]
        _save_json(PROFILES_FILE, data)
    return DEFAULT_WEIGHTS.copy()


def get_user_event_weights(user_id: int, event_id: int) -> tuple[dict, str]:
    """Returns (effective_weights, source).

    source is one of: 'event_custom', 'user_default', 'system_fallback'.
    """
    ev_data = _load_json(EVENT_WEIGHTS_FILE)
    key = f"{user_id}_{event_id}"
    if key in ev_data and "weights" in ev_data[key]:
        return normalize_weights(ev_data[key]["weights"]), "event_custom"

    user_weights, is_fallback = get_user_default_weights(user_id)
    if not is_fallback:
        return user_weights, "user_default"
    return DEFAULT_WEIGHTS.copy(), "system_fallback"


def save_user_event_weights(user_id: int, event_id: int, adjustments: dict = None,
                            answers: dict = None, custom_weights: dict = None) -> dict:
    """Saves event-specific weights without modifying user default weights."""
    default_w, _ = get_user_default_weights(user_id)
    event_w = calculate_event_weights(default_w, answers=answers, adjustments=adjustments,
                                      custom_weights=custom_weights)
    data = _load_json(EVENT_WEIGHTS_FILE)
    data[f"{user_id}_{event_id}"] = {
        "weights": event_w,
        "adjustments": adjustments or {},
        "answers": answers or {},
        "updated_at": str(pd.Timestamp.now()),
    }
    _save_json(EVENT_WEIGHTS_FILE, data)
    return event_w


def reset_user_event_weights(user_id: int, event_id: int) -> dict:
    """Removes event-specific overrides so user default weights are used."""
    data = _load_json(EVENT_WEIGHTS_FILE)
    key = f"{user_id}_{event_id}"
    if key in data:
        del data[key]
        _save_json(EVENT_WEIGHTS_FILE, data)
    default_w, _ = get_user_default_weights(user_id)
    return default_w


def get_user_taste_profile(user_id: int, users: pd.DataFrame = None) -> dict:
    """Persistent Taste Profile (general interest fingerprint, independent of event listing)."""
    data = _load_json(TASTE_PROFILES_FILE)
    uid_str = str(user_id)
    if uid_str in data:
        prof = data[uid_str]
        if "dimensions" not in prof:
            t = TasteProfile.from_user_row(prof, user_id=user_id)
            prof.update(t.to_dict())
            prof["userId"] = int(user_id)
            data[uid_str] = prof
            _save_json(TASTE_PROFILES_FILE, data)
        return prof

    if users is not None and user_id in set(users.user_id):
        u = users.set_index("user_id").loc[user_id]
        t = TasteProfile.from_user_row(u, user_id=user_id)
        interests = [item.strip() for item in str(u.get("interests", "")).split(";") if item.strip()]
        groups = sorted({interest_group(item) for item in interests})
        profile = {
            "userId": int(user_id),
            "user_id": int(user_id),
            "dimensions": t.dimensions,
            "confidence": t.confidence,
            "tags": t.tags,
            "interests": interests,
            "interest_groups": groups,
            "primary_group": groups[0] if groups else "General",
            "bio": str(u.get("bio", "")),
            "vibe": str(u.get("vibe", "chill")),
            "energy": str(u.get("energy", "ambivert")),
            "generated_at": str(pd.Timestamp.now()),
        }
        data[uid_str] = profile
        _save_json(TASTE_PROFILES_FILE, data)
        return profile

    t = TasteProfile(user_id=int(user_id), dimensions={d: 0.50 for d in STANDARD_DIMENSIONS}, confidence={}, tags={})
    return {
        "userId": int(user_id),
        "user_id": int(user_id),
        "dimensions": t.dimensions,
        "confidence": t.confidence,
        "tags": t.tags,
        "interests": [],
        "interest_groups": [],
        "primary_group": "General",
        "bio": "",
        "vibe": "chill",
        "energy": "ambivert",
        "generated_at": str(pd.Timestamp.now()),
    }


def save_user_taste_profile(user_id: int, profile: dict) -> dict:
    data = _load_json(TASTE_PROFILES_FILE)
    data[str(user_id)] = profile
    _save_json(TASTE_PROFILES_FILE, data)
    return profile


# --- Feature Calculation & Modeling ---

def profile_text(u) -> str:
    return str(u.get("interests", "")).replace(";", " ") + " " + str(u.get("bio", ""))


def fit_text(users: pd.DataFrame) -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(users.apply(profile_text, axis=1))


def hard_ok(A: pd.DataFrame, B: pd.DataFrame) -> pd.Series:
    """Row-wise: does B pass A's hard filters? Never a score, always a filter."""
    return (((~A.filter_women_only) | (B.gender == "F"))
            & ((~A.filter_verified_only) | B.verified)
            & B.age.between(A.age_min, A.age_max))


def soft_pref(A: pd.DataFrame, B: pd.DataFrame) -> np.ndarray:
    """Directional soft preferences (language, group size, verified, age)."""
    lang = np.array([bool(set(str(x).split(";")) & set(str(y).split(";")))
                     for x, y in zip(A.languages, B.languages)])
    group = (A.group_pref == "either") | (B.group_pref == "either") | (A.group_pref == B.group_pref)
    mid, half = (A.age_max + A.age_min) / 2, ((A.age_max - A.age_min) / 2).clip(lower=1)
    age_fit = (1 - (B.age - mid).abs() / half).clip(0, 1)
    return ((lang + group.values + B.verified.values + age_fit.values) / 4).astype(float)


def compute_social_fit(A: pd.DataFrame, B: pd.DataFrame) -> np.ndarray:
    """Compatibility in social behaviour and interaction style (energy & vibe)."""
    # Energy fit
    e_map = {"introvert": 0, "ambivert": 1, "extrovert": 2}
    ea = A.energy.map(lambda x: e_map.get(str(x).lower(), 1))
    eb = B.energy.map(lambda x: e_map.get(str(x).lower(), 1))
    e_diff = (ea - eb).abs()
    energy_fit = (1.0 - e_diff * 0.35).clip(0, 1)

    # Vibe fit
    same_vibe = (A.vibe == B.vibe).astype(float)
    either_depends = ((A.vibe == "depends") | (B.vibe == "depends")).astype(float) * 0.85
    vibe_fit = np.maximum(same_vibe, either_depends)
    diff_mask = (A.vibe != B.vibe) & (A.vibe != "depends") & (B.vibe != "depends")
    vibe_fit = np.where(diff_mask, 0.4, vibe_fit)

    return ((energy_fit.values + vibe_fit) / 2.0).astype(float)


def compute_intent_fit(A: pd.DataFrame, B: pd.DataFrame) -> np.ndarray:
    """Whether both want a similar experience from the event (after-event plans)."""
    same_after = (A.after == B.after).astype(float)
    hangout_both = (((A.after == "party") & (B.after == "food"))
                    | ((A.after == "food") & (B.after == "party"))).astype(float) * 0.75
    home_mismatch = (((A.after == "home") & (B.after != "home"))
                     | ((A.after != "home") & (B.after == "home"))).astype(float) * 0.25
    return (same_after + hangout_both + home_mismatch).clip(0, 1).values.astype(float)


def compute_availability_fit(A: pd.DataFrame, B: pd.DataFrame) -> np.ndarray:
    """Whether schedules and arrival expectations align."""
    arrive_fit = np.where(A.arrive == B.arrive, 1.0, 0.5)
    p_map = {"early": 0, "on_time": 1, "relaxed": 2}
    pa = A.punctuality.map(lambda x: p_map.get(str(x).lower(), 1))
    pb = B.punctuality.map(lambda x: p_map.get(str(x).lower(), 1))
    p_diff = (pa - pb).abs()
    punct_fit = (1.0 - p_diff * 0.3).clip(0, 1).values
    return ((arrive_fit + punct_fit) / 2.0).astype(float)


def compute_event_fit(A: pd.DataFrame, B: pd.DataFrame, pairs: pd.DataFrame,
                      events: pd.DataFrame | None = None) -> np.ndarray:
    """How suitable both users are for this particular event + budget alignment."""
    budget_fit = 1 - (A.budget - B.budget).abs() / 2.0
    if events is None or "event_id" not in pairs.columns:
        return budget_fit.clip(0, 1).values.astype(float)

    try:
        ev_map = events.set_index("event_id")
        eids = pairs["event_id"].values
        known_mask = [eid in ev_map.index for eid in eids]
        if not any(known_mask):
            return budget_fit.clip(0, 1).values.astype(float)

        cats = [str(ev_map.at[eid, "category"]).lower() if k else "" for eid, k in zip(eids, known_mask)]
        tags = [str(ev_map.at[eid, "tags"]).lower() if k else "" for eid, k in zip(eids, known_mask)]

        def user_suitability(interests, cat, tag):
            if not cat and not tag:
                return 0.5
            u_set = set(str(interests).lower().split(";"))
            ev_set = set(tag.split(";")) | {cat}
            ev_set = {t.strip() for t in ev_set if t.strip()}
            return 1.0 if bool(u_set & ev_set) else 0.4

        fit_a = np.array([user_suitability(i, c, t) for i, c, t in zip(A.interests, cats, tags)])
        fit_b = np.array([user_suitability(i, c, t) for i, c, t in zip(B.interests, cats, tags)])
        return (0.4 * budget_fit.values + 0.3 * fit_a + 0.3 * fit_b).clip(0, 1).astype(float)
    except Exception:
        return budget_fit.clip(0, 1).values.astype(float)


def features(pairs: pd.DataFrame, users: pd.DataFrame, vec: TfidfVectorizer,
             events: pd.DataFrame | None = None):
    """Pairs has viewer_id, candidate_id (and optionally event_id).

    Returns:
      raw (n, 8): [interest, event_fit, social_fit, pref_uv, pref_vu, intent_fit, location, availability]
      A, B, common_tags, dist_km
    """
    U = users.set_index("user_id")
    A = U.loc[pairs["viewer_id"]].reset_index()
    B = U.loc[pairs["candidate_id"]].reset_index()

    XA, XB = vec.transform(A.apply(profile_text, axis=1)), vec.transform(B.apply(profile_text, axis=1))
    cos = np.asarray(XA.multiply(XB).sum(axis=1)).ravel()

    ta, tb = A.interests.str.split(";").map(set), B.interests.str.split(";").map(set)
    common = [x & y for x, y in zip(ta, tb)]
    jac = np.array([len(c) / len(x | y) if len(x | y) else 0.0 for c, x, y in zip(common, ta, tb)])

    interest_score = 0.6 * cos + 0.4 * jac
    event_fit_score = compute_event_fit(A, B, pairs, events)
    social_fit_score = compute_social_fit(A, B)
    p_uv, p_vu = soft_pref(A, B), soft_pref(B, A)
    intent_fit_score = compute_intent_fit(A, B)
    dist = np.array([haversine_km(*r) for r in zip(A.lat, A.lng, B.lat, B.lng)])
    location_score = np.exp(-dist / 5.0)
    availability_score = compute_availability_fit(A, B)

    raw = np.column_stack([
        interest_score,
        event_fit_score,
        social_fit_score,
        p_uv,
        p_vu,
        intent_fit_score,
        location_score,
        availability_score,
    ])
    return raw, A, B, common, dist


def reciprocal(Z: np.ndarray, w: dict) -> np.ndarray:
    """Score both directions, combine with harmonic mean: one-sided pairs score low.

    Supports both 8-column and backward-compatible 5-column feature matrices.
    """
    if Z.shape[1] == 5:
        # Legacy 4-dimension compatibility
        base = (w.get("interest", 0.4) * Z[:, 0]
                + w.get("event_fit", 0.3) * Z[:, 1]
                + w.get("location", 0.2) * Z[:, 2])
        s_uv = base + w.get("prefs", 0.1) * Z[:, 3]
        s_vu = base + w.get("prefs", 0.1) * Z[:, 4]
        return 2 * s_uv * s_vu / (s_uv + s_vu + 1e-9)

    w_norm = normalize_weights(w)
    base = (
        w_norm["interest"] * Z[:, 0]
        + w_norm["event_fit"] * Z[:, 1]
        + w_norm["social_fit"] * Z[:, 2]
        + w_norm["intent_fit"] * Z[:, 5]
        + w_norm["location"] * Z[:, 6]
        + w_norm["availability"] * Z[:, 7]
    )
    s_uv = base + w_norm["prefs"] * Z[:, 3]
    s_vu = base + w_norm["prefs"] * Z[:, 4]
    return 2 * s_uv * s_vu / (s_uv + s_vu + 1e-9)


def interest_group(tag: str) -> str:
    return GROUP_OF.get(tag, tag.title())


def shared_interests(a: set, b: set, top: int = 3) -> list:
    """Per interest group both people have: 60% base for sharing group, up to 100% as tags overlap."""
    out = []
    for g in {interest_group(t) for t in a} & {interest_group(t) for t in b}:
        ga, gb = {t for t in a if interest_group(t) == g}, {t for t in b if interest_group(t) == g}
        denom = len(ga | gb)
        pct = round(100 * (0.6 + 0.4 * len(ga & gb) / denom)) if denom else 60
        out.append({"interest": g, "pct": pct, "shared": sorted(ga & gb)})
    return sorted(out, key=lambda x: (-x["pct"], x["interest"]))[:top]


def distance_bucket(km: float) -> str:
    return "< 2 km" if km < 2 else "2-5 km" if km < 5 else "5-10 km" if km < 10 else "10+ km"


def rank_triadic_candidates(
    viewer_id: int,
    event_id: int,
    users: pd.DataFrame,
    participants: pd.DataFrame,
    requests: pd.DataFrame,
    art: dict,
    limit: int = 10,
    reliability: pd.Series | None = None,
    avoid: set = frozenset(),
    weights: dict | None = None,
    events: pd.DataFrame | None = None,
    viewer_taste: TasteProfile | None = None,
    candidate_tastes: dict[int, TasteProfile] | None = None,
    event_profile: EventProfile | None = None,
) -> dict:
    """Ranks candidates across the triadic context: (user, candidate, event).

    Outputs schema matching the Amigo ML handoff specification:
      {
        "userId": 1,
        "eventId": 123,
        "recommendations": [
          {"candidateUserId": 42, "score": 0.91},
          ...
        ]
      }
    """
    pool = participants.loc[(participants.event_id == event_id) & (participants.user_id != viewer_id), "user_id"]
    asked = requests.loc[(requests.viewer_id == viewer_id) & requests.sent, "candidate_id"]
    pool = pool[~pool.isin(asked) & ~pool.map(lambda c: (viewer_id, c) in avoid)]
    pool = pd.Series(pool.unique())
    if pool.empty:
        return {"userId": viewer_id, "eventId": event_id, "recommendations": [], "candidates": []}

    U = users.set_index("user_id")
    A = U.loc[[viewer_id] * len(pool)].reset_index()
    B = U.loc[pool].reset_index()
    keep = (hard_ok(A, B) & hard_ok(B, A)).values
    if not keep.any():
        return {"userId": viewer_id, "eventId": event_id, "recommendations": [], "candidates": []}

    kept_ids = pool[keep].reset_index(drop=True)
    kept_A = A[keep].reset_index(drop=True)
    kept_B = B[keep].reset_index(drop=True)

    # 1. Resolve EventProfile
    if event_profile is None:
        if events is not None and event_id in set(events.event_id):
            event_profile = EventProfile.from_event_row(events.set_index("event_id").loc[event_id])
        else:
            event_profile = EventProfile(event_id=event_id)

    # 2. Resolve Viewer TasteProfile
    if viewer_taste is None:
        raw_tastes = _load_json(TASTE_PROFILES_FILE)
        uid_str = str(viewer_id)
        if uid_str in raw_tastes and "dimensions" in raw_tastes[uid_str]:
            viewer_taste = TasteProfile.from_dict(raw_tastes[uid_str])
        else:
            viewer_taste = TasteProfile.from_user_row(U.loc[viewer_id])

    # 3. Resolve Candidate TasteProfiles & Extract Triadic Features
    raw_tastes = _load_json(TASTE_PROFILES_FILE)
    triadic_list = []
    for i, cid in enumerate(kept_ids):
        cid_int = int(cid)
        if candidate_tastes and cid_int in candidate_tastes:
            c_taste = candidate_tastes[cid_int]
        elif str(cid_int) in raw_tastes and "dimensions" in raw_tastes[str(cid_int)]:
            c_taste = TasteProfile.from_dict(raw_tastes[str(cid_int)])
        else:
            c_taste = TasteProfile.from_user_row(kept_B.iloc[i])

        tf = extract_triadic_pair_features(
            viewer_taste=viewer_taste,
            candidate_taste=c_taste,
            event=event_profile,
            viewer_row=kept_A.iloc[i],
            candidate_row=kept_B.iloc[i],
        )
        triadic_list.append(tf)

    raw = np.array([tf.to_vector() for tf in triadic_list])
    Z = art["scaler"].transform(raw)
    w = normalize_weights(weights if weights is not None else art.get("weights", DEFAULT_WEIGHTS))
    R = reciprocal(Z, w)

    # Congestion penalty: dampen candidates who already have many pending unanswered requests
    pending = requests.loc[requests.status == "pending", "candidate_id"].value_counts()
    final = R / (1 + kept_ids.map(pending).fillna(0).values / 5.0)

    # Reliability adjustment from trust signals
    rel = kept_ids.map(reliability if reliability is not None else {}).fillna(NEUTRAL).values
    final = final * (0.5 + 0.5 * rel)
    final_scores = np.clip(final, 0.0, 1.0)

    elsewhere = participants[participants.event_id != event_id]
    mine = set(elsewhere.event_id[elsewhere.user_id == viewer_id])

    recommendations = []
    candidates = []
    top_indices = np.argsort(-final_scores)[:limit]

    for i in top_indices:
        cid = int(kept_ids[i])
        score_val = round(float(final_scores[i]), 4)
        recommendations.append({
            "candidateUserId": cid,
            "score": score_val,
        })

        tf = triadic_list[i]
        contrib = {
            "interest": w["interest"] * Z[i, 0],
            "event_fit": w["event_fit"] * Z[i, 1],
            "social_fit": w["social_fit"] * Z[i, 2],
            "prefs": w["prefs"] * (Z[i, 3] + Z[i, 4]) / 2.0,
            "intent_fit": w["intent_fit"] * Z[i, 5],
            "location": w["location"] * Z[i, 6],
            "availability": w["availability"] * Z[i, 7],
        }
        reasons = []
        for k in sorted(contrib, key=contrib.get, reverse=True)[:2]:
            if k == "interest":
                reasons.append("high taste harmony" if not tf.shared_tags else f"both into {', '.join(tf.shared_tags[:2])}")
            elif k == "event_fit":
                reasons.append("high mutual excitement for this event")
            elif k == "social_fit":
                reasons.append("matching social & crowd vibe")
            elif k == "prefs":
                reasons.append("aligned event preferences")
            elif k == "intent_fit":
                reasons.append("aligned after-event plans")
            elif k == "location":
                reasons.append(f"{distance_bucket(tf.distance_km)} apart")
            elif k == "availability":
                reasons.append("synchronized timing & schedules")

        candidates.append({
            "user_id": cid,
            "match_pct": int(round(100 * score_val)),
            "distance": distance_bucket(tf.distance_km),
            "reasons": reasons,
            "shared_interests": shared_interests(set(str(U.at[viewer_id, "interests"]).split(";")),
                                                 set(str(U.at[cid, "interests"]).split(";"))),
            "events_in_common": len(mine & set(elsewhere.event_id[elsewhere.user_id == cid])),
            "reliability": round(float(rel[i]), 2),
            "components": {
                "interest": round(float(Z[i, 0]), 3),
                "event_fit": round(float(Z[i, 1]), 3),
                "social_fit": round(float(Z[i, 2]), 3),
                "pref_uv": round(float(Z[i, 3]), 3),
                "pref_vu": round(float(Z[i, 4]), 3),
                "intent_fit": round(float(Z[i, 5]), 3),
                "location": round(float(Z[i, 6]), 3),
                "availability": round(float(Z[i, 7]), 3),
            },
        })

    return {
        "userId": viewer_id,
        "eventId": event_id,
        "recommendations": recommendations,
        "weights_used": w,
        "candidates": candidates,
    }


def rank_candidates(viewer_id, event_id, users, participants, requests, art, limit=10,
                    reliability: pd.Series | None = None, avoid: set = frozenset(),
                    weights: dict | None = None, events: pd.DataFrame | None = None,
                    viewer_taste: TasteProfile | None = None,
                    candidate_tastes: dict[int, TasteProfile] | None = None,
                    event_profile: EventProfile | None = None):
    """Ranks candidates for viewer_id at event_id using effective weights across 7 dimensions."""
    res = rank_triadic_candidates(
        viewer_id=viewer_id,
        event_id=event_id,
        users=users,
        participants=participants,
        requests=requests,
        art=art,
        limit=limit,
        reliability=reliability,
        avoid=avoid,
        weights=weights,
        events=events,
        viewer_taste=viewer_taste,
        candidate_tastes=candidate_tastes,
        event_profile=event_profile,
    )
    return res["candidates"]
