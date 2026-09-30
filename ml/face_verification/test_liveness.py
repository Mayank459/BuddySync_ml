import time

import pytest

from face_verification import liveness as L


def test_challenge_mix_and_single_use():
    for _ in range(200):
        c = L.new_challenge()
        steps = [s["action"] for s in c["steps"]]
        groups = [L.ACTIONS[a][0] for a in steps]
        assert len(set(steps)) == 3
        assert "expression" in groups and "hand" in groups
        assert sum(a.startswith("fingers") for a in steps) <= 1
    assert L.use_challenge(c["challenge_id"]) == steps
    with pytest.raises(KeyError):
        L.use_challenge(c["challenge_id"])  # second use rejected


def test_expired_challenge():
    c = L.new_challenge()
    L._challenges[c["challenge_id"]]["expires"] = time.time() - 1
    with pytest.raises(TimeoutError):
        L.use_challenge(c["challenge_id"])


def hand(extended: int):
    """Wrist at origin, fingers pointing up; the first `extended` of index..pinky are straight."""
    pts = [(0.0, 0.0)] * 21
    for k, (tip, pip) in enumerate(L.FINGERS):
        pts[pip] = (k * 10.0, -50.0)
        pts[tip] = (k * 10.0, -90.0) if k < extended else (k * 10.0, -30.0)  # folded tip curls back toward the wrist
    return pts


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_finger_count(n):
    assert L.finger_count(hand(n)) == n


def sig(**kw):
    base = {"face": True, "hands": 0, "yaw": 0.0, "eyes_closed": 0.0, "eyes_open": True,
            "jaw": 0.0, "nose_touch": 99, "fingers": [], "eye_w": 60}
    return {**base, **kw}


def frames(*runs):
    """runs of (signal, count) -> (signals, times at 10 fps)"""
    out = [s for s, k in runs for _ in range(k)]
    return out, [i / 10 for i in range(len(out))]


def test_steps_in_order_pass():
    s, t = frames((sig(), 5), (sig(jaw=0.8), 4), (sig(), 3), (sig(nose_touch=0.3, hands=1), 4), (sig(), 3),
                  (sig(eyes_closed=0.9, eyes_open=False), 1), (sig(), 3))
    res = L.check_steps(["open_mouth", "touch_nose", "blink"], s, t)
    assert all(r["done"] for r in res)
    assert res[0]["at_s"] < res[1]["at_s"] < res[2]["at_s"]


def test_wrong_order_fails():
    s, t = frames((sig(), 3), (sig(nose_touch=0.3, hands=1), 4), (sig(), 3), (sig(jaw=0.8), 4), (sig(), 3))
    res = L.check_steps(["open_mouth", "touch_nose"], s, t)
    assert res[0]["done"] and not res[1]["done"]


def test_closed_eyes_photo_is_not_a_blink():
    s, t = frames((sig(eyes_closed=0.9, eyes_open=False), 30))
    assert not L.check_steps(["blink"], s, t)[0]["done"]


def test_brief_flicker_does_not_count_as_held():
    s, t = frames((sig(), 3), (sig(jaw=0.8), 2), (sig(), 3))  # 2 frames < HOLD_FRAMES
    assert not L.check_steps(["open_mouth"], s, t)[0]["done"]


def test_fingers_need_exact_count():
    s, t = frames((sig(fingers=[2], hands=1), 5))
    assert L.check_steps(["fingers_2"], s, t)[0]["done"]
    assert not L.check_steps(["fingers_3"], s, t)[0]["done"]


def test_turn_is_relative_to_start_pose():
    # phone held off to one side: user starts at yaw 0.15. Staying there is not a turn; moving 0.25 away is.
    s, t = frames((sig(yaw=0.15), 6), (sig(yaw=0.2), 5))
    assert not L.check_steps(["turn_head"], s, t)[0]["done"]
    s, t = frames((sig(yaw=0.15), 6), (sig(yaw=-0.12), 4))
    assert L.check_steps(["turn_head"], s, t)[0]["done"]


def test_live_guide_matches_check_steps():
    """The live guide completes steps in order with the same rules as the offline check."""
    from face_verification.liveness import BASELINE_FRAMES, HOLD_FRAMES, LiveGuide, check_steps
    neutral = {"face": True, "yaw": 0.0, "eyes_closed": 0.0, "eyes_open": True, "jaw": 0.0, "nose_touch": 99,
               "fingers": [], "eye_w": 50}
    frames = ([neutral] * BASELINE_FRAMES + [{**neutral, "jaw": 0.9}] * HOLD_FRAMES + [neutral] * 2
              + [{**neutral, "eyes_closed": 0.9, "eyes_open": False}] + [neutral] * 2
              + [{**neutral, "fingers": [2]}] * HOLD_FRAMES)
    times = [n / 10 for n in range(len(frames))]
    steps = ["open_mouth", "blink", "fingers_2"]
    guide = LiveGuide(steps)
    completed = [guide.feed(s, t) for s, t in zip(frames, times)]
    assert guide.finished and sum(completed) == 3
    assert [r["done"] for r in check_steps(steps, frames, times)] == [True, True, True]
    early = LiveGuide(["blink"])  # a blink before the eyes were seen open doesn't count
    assert not any(early.feed(s, 0) for s in [neutral] * BASELINE_FRAMES + [{**neutral, "eyes_closed": 0.9, "eyes_open": False}])
