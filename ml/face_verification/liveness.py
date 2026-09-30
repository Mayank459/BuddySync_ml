"""Active liveness: random challenge -> short live video -> check every action happened, in order, by one person.

MediaPipe (Apache-2.0) gives face landmarks + blendshapes and hand landmarks; each action is simple geometry.
Why each challenge has one facial-expression step and one hand step:
  - a printed/on-screen photo can't blink or open its mouth
  - a pre-recorded video can't know which random steps (and finger count) will be asked
  - hands in front of the face tend to break real-time face-swap deepfakes
Not covered: live deepfakes fed through a virtual camera -> pair with device attestation in the app (plan §7).
"""
import math
import secrets
import threading
import time
from functools import lru_cache

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from face_verification.face import MODELS, embed

URLS = {
    "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}
ACTIONS = {  # action -> (group, prompt shown to the user)
    "blink": ("expression", "Blink a couple of times"),
    "open_mouth": ("expression", "Open your mouth wide"),
    "turn_head": ("motion", "Turn your head to one side, then back to the camera"),
    "touch_nose": ("hand", "Touch the tip of your nose with your index finger"),
    **{f"fingers_{n}": ("hand", f"Hold up {n} finger{'s' * (n > 1)} next to your face (don't count your thumb)")
       for n in (1, 2, 3, 4)},
}
CHALLENGE_TTL_S = 300    # ponytail: generous for manual playground uploads; ~90 s once the app records live
MAX_VIDEO_S, SAMPLE_FPS, MAX_SIDE = 20, 10, 640
HOLD_FRAMES = 3          # an action must hold for 3 sampled frames (~0.3 s); a blink needs 1
BLINK_CLOSED, BLINK_OPEN, JAW_OPEN = 0.5, 0.3, 0.5
TURN = 0.2               # head turn: nose moves this far (as a share of face width) from the user's own start pose
IDENTITY_COVERAGE = 0.6   # share of identity samples that must pass the quality gate
STEP_ID_WINDOW_S = 1.0    # each completed step needs an identity sample this close to it
BASELINE_FRAMES = 5      # start pose = median of the first face frames (user is told to start facing the camera)
NOSE_TOUCH = 0.6         # fingertip-to-nose distance, in eye-widths
# face mesh / hand landmark indices
NOSE, CHEEK_A, CHEEK_B, EYE_A, EYE_B = 1, 234, 454, 33, 263
WRIST, FINGERS = 0, [(8, 6), (12, 10), (16, 14), (20, 18)]  # (tip, pip) for index..pinky

# ---- challenges ----------------------------------------------------------------------------
_challenges, _lock = {}, threading.Lock()  # ponytail: in-memory, single instance; move to Redis with a TTL


def new_challenge() -> dict:
    rng = secrets.SystemRandom()  # unpredictable: the whole defence against replayed videos
    by_group = lambda g: [a for a, (grp, _) in ACTIONS.items() if grp == g]
    steps = [rng.choice(by_group("expression")), rng.choice(by_group("hand"))]
    rest = [a for a in ACTIONS if a not in steps and not (a.startswith("fingers") and any(s.startswith("fingers") for s in steps))]
    steps.append(rng.choice(rest))
    rng.shuffle(steps)
    cid = secrets.token_urlsafe(16)
    with _lock:
        now = time.time()
        for k in [k for k, v in _challenges.items() if v["expires"] < now]:
            del _challenges[k]
        _challenges[cid] = {"steps": steps, "expires": now + CHALLENGE_TTL_S}
    return {"challenge_id": cid, "expires_in_s": CHALLENGE_TTL_S,
            "steps": [{"action": s, "prompt": ACTIONS[s][1]} for s in steps],
            "instructions": f"Record one video of up to {MAX_VIDEO_S} s: start facing the camera, do the steps in "
                            "this order holding each for about a second, then face the camera again."}


def use_challenge(cid: str) -> list:
    """Single use: removed on first attempt, so one challenge can't be retried with different videos."""
    with _lock:
        c = _challenges.pop(cid, None)
    if c is None:
        raise KeyError("unknown or already used challenge")
    if c["expires"] < time.time():
        raise TimeoutError("challenge expired")
    return c["steps"]


# ---- per-frame geometry (pure functions: tested without video) -----------------------------
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def finger_count(hand) -> int:
    """Extended fingers, thumb excluded: a fingertip further from the wrist than its middle joint."""
    return sum(dist(hand[WRIST], hand[t]) > 1.1 * dist(hand[WRIST], hand[p]) for t, p in FINGERS)


def frame_signals(face, blend, hands) -> dict:
    """face: landmark (x, y) pixels or None; blend: {name: score}; hands: list of 21-point (x, y) pixel lists."""
    s = {"face": face is not None, "hands": len(hands)}
    if face is None:
        return s
    eye_w = dist(face[EYE_A], face[EYE_B]) or 1.0
    a, b = sorted([face[CHEEK_A][0], face[CHEEK_B][0]])
    s["yaw"] = (face[NOSE][0] - a) / ((b - a) or 1.0) - 0.5  # signed; ~0 facing camera. A tilted flat photo barely moves it
    s["eyes_closed"] = min(blend.get("eyeBlinkLeft", 0), blend.get("eyeBlinkRight", 0))
    s["eyes_open"] = max(blend.get("eyeBlinkLeft", 0), blend.get("eyeBlinkRight", 0)) < BLINK_OPEN
    s["jaw"] = blend.get("jawOpen", 0)
    s["nose_touch"] = min((dist(h[8], face[NOSE]) / eye_w for h in hands), default=99)
    s["fingers"] = [finger_count(h) for h in hands]
    s["eye_w"] = eye_w
    return s


def baseline(signals: list) -> float:
    first = [s["yaw"] for s in signals if s.get("face")][:BASELINE_FRAMES]
    return float(np.median(first)) if first else 0.0


def turned(s: dict, base: float) -> float:
    return abs(s["yaw"] - base)


def action_now(action: str, s: dict, base: float = 0.0) -> bool:
    if not s.get("face"):
        return False
    if action == "blink":
        return s["eyes_closed"] > BLINK_CLOSED
    if action == "open_mouth":
        return s["jaw"] > JAW_OPEN
    if action == "turn_head":
        return turned(s, base) > TURN
    if action == "touch_nose":
        return s["nose_touch"] < NOSE_TOUCH
    if action.startswith("fingers_"):
        return int(action[-1]) in s["fingers"]
    raise ValueError(action)


def check_steps(steps: list, signals: list, times: list) -> list:
    """Each step must be seen (held HOLD_FRAMES) after the previous one finished. Returns per-step results."""
    out, i, base = [], 0, baseline(signals)
    for action in steps:
        need, run, eyes_were_open, at = (1 if action == "blink" else HOLD_FRAMES), 0, False, None
        while i < len(signals):
            s = signals[i]
            i += 1
            if action == "blink" and not eyes_were_open:  # closed eyes only count after we've seen them open
                eyes_were_open = bool(s.get("eyes_open"))
                continue
            run = run + 1 if action_now(action, s, base) else 0
            if run >= need:
                at = round(times[i - 1], 2)
                break
        out.append({"action": action, "done": at is not None, "at_s": at})
    return out


class LiveGuide:
    """check_steps, one frame at a time, for live on-screen guidance ("Blink now" → ✓ → next step).
    Same thresholds as the server; the server still verifies the recorded video and makes the decision."""

    def __init__(self, steps: list):
        self.steps, self.i, self.run, self.eyes_were_open = list(steps), 0, 0, False
        self.base_yaws, self.done_at = [], []

    @property
    def finished(self) -> bool:
        return self.i >= len(self.steps)

    @property
    def current(self):
        return None if self.finished else self.steps[self.i]

    def feed(self, s: dict, t: float) -> bool:
        """True when this frame completed the current step."""
        if s.get("face") and len(self.base_yaws) < BASELINE_FRAMES:
            self.base_yaws.append(s["yaw"])  # start pose, as in baseline()
            return False
        if self.finished or len(self.base_yaws) < BASELINE_FRAMES:
            return False
        action = self.steps[self.i]
        if action == "blink" and not self.eyes_were_open:  # closed eyes only count after we've seen them open
            self.eyes_were_open = bool(s.get("eyes_open"))
            return False
        self.run = self.run + 1 if action_now(action, s, float(np.median(self.base_yaws))) else 0
        if self.run >= (1 if action == "blink" else HOLD_FRAMES):
            self.done_at.append(round(t, 2))
            self.i, self.run, self.eyes_were_open = self.i + 1, 0, False
            return True
        return False


# ---- video ---------------------------------------------------------------------------------
def _model(name):
    path = MODELS / f"{name}_landmarker.task"
    if not path.exists():
        import requests
        MODELS.mkdir(exist_ok=True)
        r = requests.get(URLS[name], timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    return str(path)


@lru_cache
def landmarkers():
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_model("face")), num_faces=2, output_face_blendshapes=True))
    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_model("hand")), num_hands=2))
    return face, hand


_mp_lock = threading.Lock()  # ponytail: one video at a time per process; run more workers if verifications queue up


def landmarks(bgr):
    h, w = bgr.shape[:2]
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(bgr[..., ::-1]))
    face_lm, hand_lm = landmarkers()
    with _mp_lock:
        fr, hr = face_lm.detect(img), hand_lm.detect(img)
    px = lambda pts: [(p.x * w, p.y * h) for p in pts]
    face = px(fr.face_landmarks[0]) if fr.face_landmarks else None
    blend = {c.category_name: c.score for c in fr.face_blendshapes[0]} if fr.face_blendshapes else {}
    return face, blend, [px(hl) for hl in hr.hand_landmarks], len(fr.face_landmarks)


def analyze_video(path: str, steps: list) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("not a readable video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = fps if 1 <= fps <= 240 else 30.0
    every = max(1, round(fps / SAMPLE_FPS))
    signals, times, embs, emb_times, tried, multi, best, n = [], [], [], [], 0, 0, None, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            t = (n - 1) / fps
            if t > MAX_VIDEO_S:
                raise OverflowError(f"video longer than {MAX_VIDEO_S} s")
            if (n - 1) % every:
                continue
            scale = MAX_SIDE / max(frame.shape[:2])
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            face, blend, hands, n_faces = landmarks(frame)
            s = frame_signals(face, blend, hands)
            signals.append(s)
            times.append(t)
            multi += n_faces > 1
            # identity samples on every 3rd frame at ANY pose (else a second person could hide by staying turned);
            # skip only frames with a hand over the face. SFace copes with pose (LFW mixed-pose pairs: 2% FRR).
            if s.get("face") and len(signals) % 3 == 1:
                tried += 1  # hand-over-face frames count as tried-but-unconfirmed, so a raised hand can't hide a face
                emb, _ = embed(frame) if s["nose_touch"] > 2 else (None, "hand_over_face")
                if emb is not None:
                    embs.append(emb)
                    emb_times.append(t)
                    facing = turned(s, baseline(signals)) < TURN / 2 and s["eyes_open"]
                    if facing and (best is None or s["eye_w"] > best[0]):  # clearest frontal frame -> profile match
                        best = (s["eye_w"], emb)
    finally:
        cap.release()
    if not signals:
        raise ValueError("video has no frames")
    results = check_steps(steps, signals, times)
    for r in results:  # an action only counts if we also confirmed WHO did it (no blurry-stand-in gap)
        r["identity_checked"] = r["done"] and any(abs(t - r["at_s"]) <= STEP_ID_WINDOW_S for t in emb_times)
    return {"steps": results, "frames_analyzed": len(signals),
            "identity_coverage": round(len(embs) / tried, 2) if tried else 0.0,
            "duration_s": round(times[-1], 2), "multiple_faces_share": multi / len(signals),
            "identity_samples": embs, "best_embedding": best[1] if best else None}
