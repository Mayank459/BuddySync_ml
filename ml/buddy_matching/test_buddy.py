import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from buddy_matching.model import (
    DEFAULT_WEIGHTS,
    KEYS,
    calculate_default_weights_from_answers,
    calculate_event_weights,
    features,
    fit_text,
    get_user_default_weights,
    get_user_event_weights,
    get_user_taste_profile,
    normalize_weights,
    rank_candidates,
    reconduct_user_weights,
    reciprocal,
    save_user_default_weights,
    save_user_event_weights,
    shared_interests,
)


def user(uid, **kw):
    base = dict(
        user_id=uid, city="Pune", lat=18.52, lng=73.85, age=25, gender="M",
        interests="rock;edm", bio="love rock", vibe="chill", arrive="together", after="food",
        punctuality="early", energy="ambivert", budget=2, languages="english;hindi",
        group_pref="either", verified=True, filter_verified_only=False,
        filter_women_only=False, age_min=18, age_max=40,
    )
    return {**base, **kw}


def test_one_sided_pair_scores_below_balanced():
    w = {"interest": 0, "event_fit": 0, "location": 0, "prefs": 1}
    Z = np.array([[0, 0, 0, 0.9, 0.1], [0, 0, 0, 0.5, 0.5]])
    lopsided, balanced = reciprocal(Z, w)
    assert balanced > lopsided


def test_hard_filters_and_ranking():
    users = pd.DataFrame([
        user(1, gender="F", filter_women_only=True),
        user(2, gender="F", interests="rock;edm"),       # passes, strong match
        user(3, gender="M"),                             # excluded by viewer's women-only filter
        user(4, gender="F", interests="pottery", lat=18.60),  # passes, weaker match
        user(5, gender="F", age=39, age_min=35, age_max=45),  # viewer (25) outside *their* range
    ])
    parts = pd.DataFrame({"event_id": 7, "user_id": [1, 2, 3, 4, 5], "status": "looking_for_buddy"})
    reqs = pd.DataFrame(columns=["viewer_id", "candidate_id", "event_id", "sent", "status"]).astype({"sent": bool})
    art = {"vectorizer": fit_text(users), "scaler": FunctionTransformer(), "weights": DEFAULT_WEIGHTS}
    out = rank_candidates(1, 7, users, parts, reqs, art)
    ids = [c["user_id"] for c in out]
    assert ids == [2, 4], ids
    assert all(0 <= c["match_pct"] <= 100 for c in out)


def test_shared_interests_by_group():
    out = shared_interests({"edm", "rock", "coding"}, {"edm", "sufi", "ai", "pottery"})
    assert out == [
        {"interest": "Music", "pct": 73, "shared": ["edm"]},
        {"interest": "Technology", "pct": 60, "shared": []},
    ]


def test_feedback_reorders_and_keeps_pairs_apart():
    users = pd.DataFrame([user(1), user(2), user(3), user(4, interests="rock;edm;sufi")])
    parts = pd.DataFrame({"event_id": [7, 7, 7, 7, 8, 8], "user_id": [1, 2, 3, 4, 1, 2],
                          "status": "looking_for_buddy"})
    reqs = pd.DataFrame(columns=["viewer_id", "candidate_id", "event_id", "sent", "status"]).astype({"sent": bool})
    art = {"vectorizer": fit_text(users), "scaler": FunctionTransformer(), "weights": DEFAULT_WEIGHTS}
    out = rank_candidates(1, 7, users, parts, reqs, art, reliability=pd.Series({2: 0.3, 3: 0.95}), avoid={(1, 4)})
    assert [c["user_id"] for c in out] == [3, 2]
    assert out[1]["events_in_common"] == 1 and out[0]["events_in_common"] == 0
    assert out[0]["shared_interests"][0] == {"interest": "Music", "pct": 100, "shared": ["edm", "rock"]}


def test_fallback_weights_dimensions_and_sum():
    assert set(DEFAULT_WEIGHTS.keys()) == set(KEYS)
    assert round(sum(DEFAULT_WEIGHTS.values()), 4) == 1.0000
    assert DEFAULT_WEIGHTS["interest"] == 0.30
    assert DEFAULT_WEIGHTS["event_fit"] == 0.25
    assert DEFAULT_WEIGHTS["social_fit"] == 0.15
    assert DEFAULT_WEIGHTS["prefs"] == 0.10
    assert DEFAULT_WEIGHTS["intent_fit"] == 0.10
    assert DEFAULT_WEIGHTS["location"] == 0.05
    assert DEFAULT_WEIGHTS["availability"] == 0.05


def test_first_time_questionnaire_tradeoffs():
    # User preferring interest over location (Q1: A, Q6: A, Q10: A)
    answers_interest = {"q1": "A", "q2": "A", "q3": "A", "q4": "A", "q5": "A", "q6": "A", "q7": "A", "q8": "A", "q9": "A", "q10": "A"}
    w_interest = calculate_default_weights_from_answers(answers_interest)
    assert round(sum(w_interest.values()), 4) == 1.0000

    # User preferring proximity over interest (Q1: B, Q6: B, Q10: B)
    answers_location = {"q1": "B", "q2": "A", "q3": "A", "q4": "A", "q5": "A", "q6": "B", "q7": "A", "q8": "A", "q9": "A", "q10": "B"}
    w_location = calculate_default_weights_from_answers(answers_location)
    assert round(sum(w_location.values()), 4) == 1.0000

    # Trade-off holds: user choosing location has higher location weight and lower interest weight
    assert w_location["location"] > w_interest["location"]
    assert w_interest["interest"] > w_location["interest"]


def test_event_specific_adjustments_preserve_default_weights():
    uid, eid = 99991, 101
    default_saved = save_user_default_weights(uid, answers={"q1": "A", "q2": "A"})
    initial_defaults, _ = get_user_default_weights(uid)

    # Event adjustments: user wants location much more important for this event
    save_user_event_weights(uid, eid, adjustments={"location": 0.15, "interest": -0.10})
    event_w, src = get_user_event_weights(uid, eid)

    assert src == "event_custom"
    assert event_w["location"] > initial_defaults["location"]

    # Verify default weights are UNCHANGED
    post_defaults, is_fallback = get_user_default_weights(uid)
    assert not is_fallback
    assert post_defaults == initial_defaults, "Default weights must be preserved and never overwritten by event adjustments!"


def test_reconduct_user_weights():
    uid = 99992
    save_user_default_weights(uid, answers={"q1": "B", "q2": "B"})
    _, is_fb = get_user_default_weights(uid)
    assert not is_fb

    reconduct_user_weights(uid)
    reset_w, is_fb_after = get_user_default_weights(uid)
    assert is_fb_after
    assert reset_w == DEFAULT_WEIGHTS


def test_taste_profile_generation():
    users = pd.DataFrame([
        user(99993, interests="coding;ai;rock;food festivals", bio="Tech enthusiast and rocker."),
    ])
    profile = get_user_taste_profile(99993, users=users)
    assert profile["user_id"] == 99993
    assert "Technology" in profile["interest_groups"]
    assert "Music" in profile["interest_groups"]
    assert "coding" in profile["interests"]


def test_components_contain_all_seven_dimensions():
    users = pd.DataFrame([user(1), user(2)])
    parts = pd.DataFrame({"event_id": 10, "user_id": [1, 2], "status": "looking_for_buddy"})
    reqs = pd.DataFrame(columns=["viewer_id", "candidate_id", "event_id", "sent", "status"]).astype({"sent": bool})
    art = {"vectorizer": fit_text(users), "scaler": FunctionTransformer(), "weights": DEFAULT_WEIGHTS}
    out = rank_candidates(1, 10, users, parts, reqs, art)
    assert len(out) == 1
    comps = out[0]["components"]
    for key in ["interest", "event_fit", "social_fit", "pref_uv", "pref_vu", "intent_fit", "location", "availability"]:
        assert key in comps, f"Missing component {key}"
        assert 0.0 <= comps[key] <= 1.0


def test_taste_profile_unknown_semantics_and_confidence():
    from buddy_matching.taste import TasteProfile

    # 0.50 + confidence 0 = unknown, not neutral
    p1 = TasteProfile.from_dict({
        "userId": 1,
        "dimensions": {"MUSIC": 0.85, "FOOD": 0.50, "SPORTS": 0.80},
        "confidence": {"MUSIC": 0.90, "SPORTS": 0.80},  # FOOD confidence is 0
        "tags": {"edm": 1.0, "cricket": 1.0},
    })
    p2 = TasteProfile.from_dict({
        "userId": 2,
        "dimensions": {"MUSIC": 0.85, "FOOD": 0.95, "SPORTS": 0.80},
        "confidence": {"MUSIC": 0.90, "FOOD": 0.95, "SPORTS": 0.80},
        "tags": {"edm": 1.0, "rock": 1.0},
    })

    sim = p1.confidence_weighted_similarity(p2)
    # FOOD (unknown for p1) is ignored; MUSIC and SPORTS match perfectly -> taste_similarity == 1.0
    assert sim["taste_similarity"] == 1.0000
    assert sim["known_shared_dimensions"] == 2


def test_explicit_tag_overlap():
    from buddy_matching.taste import TasteProfile

    p1 = TasteProfile.from_dict({
        "userId": 1,
        "tags": {"cricket": 1.0, "ai": 1.0, "music festivals": 1.0, "anime": 1.0, "edm": 1.0},
    })
    p2 = TasteProfile.from_dict({
        "userId": 2,
        "tags": {"edm": 1.0, "music festivals": 1.0, "rock": 1.0},
    })

    res = p1.tag_overlap(p2)
    assert res["shared_tags"] == ["edm", "music festivals"]
    assert 0.0 < res["tag_overlap_score"] <= 1.0


def test_triadic_event_synergy():
    from buddy_matching.event_profile import EventProfile
    from buddy_matching.taste import TasteProfile

    ev = EventProfile.from_dict({
        "eventId": 101,
        "category": "music",
        "tags": ["edm", "music festivals"],
    })

    # Both users love EDM music festivals
    fan1 = TasteProfile.from_dict({"userId": 1, "dimensions": {"MUSIC": 0.9, "LIVE_EVENT": 0.85}, "confidence": {"MUSIC": 0.9}, "tags": {"edm": 1.0}})
    fan2 = TasteProfile.from_dict({"userId": 2, "dimensions": {"MUSIC": 0.95, "LIVE_EVENT": 0.90}, "confidence": {"MUSIC": 0.9}, "tags": {"edm": 1.0}})
    # User who dislikes live music
    non_fan = TasteProfile.from_dict({"userId": 3, "dimensions": {"MUSIC": 0.2, "LIVE_EVENT": 0.1}, "confidence": {"MUSIC": 0.9}, "tags": {"coding": 1.0}})

    syn_high = ev.triadic_synergy(fan1, fan2)
    syn_low = ev.triadic_synergy(fan1, non_fan)

    assert syn_high["triadic_event_fit"] > syn_low["triadic_event_fit"]


def test_rank_triadic_candidates_output_schema():
    from buddy_matching.model import rank_triadic_candidates

    users = pd.DataFrame([user(1), user(2), user(3)])
    parts = pd.DataFrame({"event_id": 5, "user_id": [1, 2, 3], "status": "looking_for_buddy"})
    reqs = pd.DataFrame(columns=["viewer_id", "candidate_id", "event_id", "sent", "status"]).astype({"sent": bool})
    art = {"vectorizer": fit_text(users), "scaler": FunctionTransformer(), "weights": DEFAULT_WEIGHTS}

    res = rank_triadic_candidates(1, 5, users, parts, reqs, art)

    # Check exact future ML output schema
    assert "userId" in res and res["userId"] == 1
    assert "eventId" in res and res["eventId"] == 5
    assert "recommendations" in res
    assert isinstance(res["recommendations"], list)
    for rec in res["recommendations"]:
        assert "candidateUserId" in rec
        assert "score" in rec
        assert 0.0 <= rec["score"] <= 1.0


def test_reset_user_event_weights():
    from buddy_matching.model import (
        save_user_default_weights,
        save_user_event_weights,
        get_user_event_weights,
        reset_user_event_weights,
    )
    uid, eid = 99994, 202
    save_user_default_weights(uid, answers={"q1": "A", "q2": "A"})
    save_user_event_weights(uid, eid, adjustments={"location": 0.20})
    _, src_before = get_user_event_weights(uid, eid)
    assert src_before == "event_custom"

    reset_w = reset_user_event_weights(uid, eid)
    _, src_after = get_user_event_weights(uid, eid)
    assert src_after == "user_default"


def test_taste_profile_update_and_backend_dimensions():
    from buddy_matching.model import get_user_taste_profile, save_user_taste_profile
    users = pd.DataFrame([user(99995, interests="cricket;ai", bio="Data & sports fan")])
    prof = get_user_taste_profile(99995, users=users)

    assert prof["userId"] == 99995
    assert "SPORTS" in prof["dimensions"]
    assert "TECHNOLOGY" in prof["dimensions"]
    assert prof["tags"]["cricket"] == 1.0
    assert prof["tags"]["ai"] == 1.0

