"""Active liveness: random challenge -> short live video -> check every action happened, in order, by one person.

MediaPipe (Apache-2.0) gives face landmarks + blendshapes and hand landmarks; each action is simple geometry.
Why each challenge has one facial-expression step and one hand step:
  - a printed/on-screen photo can't blink, smile or open its mouth
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
    "blink": ("expression", "Blink twice, quickly"),
    "open_mouth": ("expression", "Open your mouth wide"),
    "smile": ("expression", "Give a big smile, lips closed or slightly apart"),
    "turn_head": ("motion", "Turn your head to one side, then back to the camera"),
    "tilt_head": ("motion", "Tilt your head sideways, ear toward your shoulder, keep facing the camera"),
    "touch_nose": ("hand", "Touch the tip of your nose with your index finger"),
    "cover_eye": ("hand", "Cover one eye with your hand"),
    **{f"fingers_{n}": ("hand", f"Hold up {n} finger{'s' * (n > 1)} next to your face (don't count your thumb)")
       for n in (1, 2, 3, 4)},
}
COVERS_FACE = {"touch_nose", "cover_eye"}
CHALLENGE_TTL_S = 300    # ponytail: generous for manual playground uploads; ~90 s once the app records live
MAX_VIDEO_S, SAMPLE_FPS, MAX_SIDE = 20, 10, 640
MAX_FRAMES = MAX_VIDEO_S * 240     # timestamps come from the file: one that never advances must still end
MAX_FRAME_PIXELS = 3840 * 2160     # 4K; bigger frames are refused before decoding (memory)
HOLD_FRAMES = 3          # an action must hold for 3 sampled frames (~0.3 s)
BLINKS, BLINK_WINDOW_S = 2, 2.0  # people blink on their own every few seconds: one blink proves nothing asked
BLINK_CLOSED, BLINK_OPEN, JAW_OPEN = 0.5, 0.3, 0.5
# blink scores run low for some people (glasses: full blinks peaked at 0.39), but the eye aspect ratio still
# drops to under half of their own start value, so a blink also counts on that
EAR_CLOSED, EAR_OPEN = 0.6, 0.8
# smile = mouth corners pulled out vs the user's start; the smile blendshape also fires on a wide-open mouth
# (0.67 seen), and an open mouth widened the mouth only 1.17x, so the jaw must stay mostly shut
SMILE_WIDEN, SMILE_MAX_JAW = 1.2, 0.35
WIDE_MOUTH = 0.35        # jaw above this: not used for identity (the face model sees a different face)
TILT_DEG = 15            # head tilt: eye line rotates this far from the start (seen wandering +-3 deg)
COVER_EYE = 0.25         # a hand point this close to an eye centre, in eye-widths (fingers held beside the face: 0.39)
TURN = 0.2               # head turn: nose moves this far (as a share of face width) from the user's own start pose
IDENTITY_COVERAGE = 0.6   # share of identity samples that must pass the quality gate
STEP_ID_WINDOW_S = 1.0    # each completed step needs an identity sample this close to it
BASELINE_FRAMES = 5      # start pose = median of the first face frames (user is told to start facing the camera)
NOSE_TOUCH = 0.6         # fingertip-to-nose distance, in eye-widths
# face mesh / hand landmark indices
NOSE, CHEEK_A, CHEEK_B, EYE_A, EYE_B = 1, 234, 454, 33, 263
EYES = ((33, 160, 158, 133, 153, 144), (362, 385, 387, 263, 373, 380))  # corner, top, top, corner, bottom, bottom
MOUTH_A, MOUTH_B = 61, 291  # mouth corners
WRIST, FINGERS = 0, [(8, 6), (12, 10), (16, 14), (20, 18)]  # (tip, pip) for index..pinky

# ---- challenges ----------------------------------------------------------------------------
_challenges, _lock = {}, threading.Lock()  # ponytail: in-memory, single instance; move to Redis with a TTL


def new_challenge() -> dict:
    rng = secrets.SystemRandom()  # unpredictable: the whole defence against replayed videos
    by_group = lambda g: [a for a, (grp, _) in ACTIONS.items() if grp == g]
    steps = [rng.choice(by_group("expression")), rng.choice(by_group("hand"))]
    # at most one finger count, and at most one step that covers the face: two (cover eye, then touch nose)
    # kept a hand on the face for ~4 s of an 8.7 s real attempt, and those frames count against identity coverage
    rest = [a for a in ACTIONS if a not in steps
            and not (a.startswith("fingers") and any(s.startswith("fingers") for s in steps))
            and not (a in COVERS_FACE and any(s in COVERS_FACE for s in steps))]
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
    s["ear"] = float(np.mean([(dist(face[e[1]], face[e[5]]) + dist(face[e[2]], face[e[4]]))
                              / (2 * dist(face[e[0]], face[e[3]]) or 1.0) for e in EYES]))
    s["jaw"] = blend.get("jawOpen", 0)
    s["mouth_w"] = dist(face[MOUTH_A], face[MOUTH_B]) / eye_w
    s["roll"] = math.degrees(math.atan2(face[EYE_B][1] - face[EYE_A][1], face[EYE_B][0] - face[EYE_A][0]))
    centres = [np.mean([face[i] for i in e], axis=0) for e in EYES]
    s["eye_cover"] = min((dist(p, c) / eye_w for h in hands for p in h for c in centres), default=99)
    # hand points inside the face (its landmark box, 10% in from each edge): touching the nose put 3 points inside,
    # fingers held beside the face 0 (the old "fingertip within 2 eye-widths of the nose" test flagged those too)
    xs, ys = [p[0] for p in face], [p[1] for p in face]
    mx, my = (max(xs) - min(xs)) * 0.1, (max(ys) - min(ys)) * 0.1
    s["hand_in_face"] = sum(min(xs) + mx < p[0] < max(xs) - mx and min(ys) + my < p[1] < max(ys) - my
                            for h in hands for p in h)
    s["nose_touch"] = min((dist(h[8], face[NOSE]) / eye_w for h in hands), default=99)
    s["fingers"] = [finger_count(h) for h in hands]
    s["eye_w"] = eye_w
    return s


def baseline(signals: list, key: str = "yaw") -> float:
    first = [s[key] for s in signals if s.get("face") and key in s][:BASELINE_FRAMES]
    return float(np.median(first)) if first else 0.0


def baselines(signals: list) -> dict:
    """The user's own start face: actions are judged against it, not against fixed numbers."""
    return {k: baseline(signals, k) for k in ("yaw", "ear", "mouth_w", "roll")}


def eyes_open(s: dict, ear0: float = 0.0) -> bool:
    return bool(s.get("eyes_open")) or bool(ear0 and s.get("ear", 0) > EAR_OPEN * ear0)


def eyes_shut(s: dict, ear0: float = 0.0) -> bool:
    return s["eyes_closed"] > BLINK_CLOSED or bool(ear0 and s.get("ear", ear0) < EAR_CLOSED * ear0)


def turned(s: dict, base: float) -> float:
    return abs(s["yaw"] - base)


def action_now(action: str, s: dict, base: dict) -> bool:
    if not s.get("face"):
        return False
    if action == "blink":
        return eyes_shut(s, base.get("ear", 0.0))
    if action == "open_mouth":
        return s["jaw"] > JAW_OPEN
    if action == "smile":
        return bool(base.get("mouth_w")) and s["mouth_w"] >= SMILE_WIDEN * base["mouth_w"] and s["jaw"] < SMILE_MAX_JAW
    if action == "turn_head":
        return turned(s, base.get("yaw", 0.0)) > TURN
    if action == "tilt_head":  # still facing the camera: a big turn skews the eye line too (-13 deg seen on a turn)
        return abs(s["roll"] - base.get("roll", 0.0)) > TILT_DEG and turned(s, base.get("yaw", 0.0)) < TURN
    if action == "touch_nose":
        return s["nose_touch"] < NOSE_TOUCH
    if action == "cover_eye":
        return s["eye_cover"] < COVER_EYE
    if action.startswith("fingers_"):
        return int(action[-1]) in s["fingers"]
    raise ValueError(action)


class Step:
    """One step's progress, frame by frame. A held action needs HOLD_FRAMES in a row. A blink needs BLINKS
    separate closures within BLINK_WINDOW_S (a natural blink is a single one), counted only after open eyes
    were seen (a closed-eyes photo isn't a blink). Covering an eye often hides the whole face from the
    landmarker: those frames (no face, a hand in view) keep the run going if the hand was seen on the eye."""

    def __init__(self, action: str):
        self.action, self.run, self.seen_open, self.was_shut, self.closures = action, 0, False, False, []
        self.hand_on_eye = False  # cover_eye: seen at least once in the current run

    def feed(self, s: dict, base: dict, t: float) -> bool:
        if self.action == "cover_eye":
            on = action_now("cover_eye", s, base)
            self.run = self.run + 1 if on or (not s.get("face") and s.get("hands", 0) > 0) else 0
            self.hand_on_eye = (self.hand_on_eye or on) and self.run > 0
            return self.run >= HOLD_FRAMES and self.hand_on_eye
        if self.action != "blink":
            self.run = self.run + 1 if action_now(self.action, s, base) else 0
            return self.run >= HOLD_FRAMES
        if not self.seen_open:
            self.seen_open = eyes_open(s, base["ear"])
            return False
        shut = action_now("blink", s, base)
        if shut and not self.was_shut:  # a new closure: the eyes reopened since the last one
            self.closures = [c for c in self.closures if t - c <= BLINK_WINDOW_S] + [t]
        self.was_shut = shut
        return len(self.closures) >= BLINKS


def check_steps(steps: list, signals: list, times: list) -> list:
    """Each step must be completed after the previous one finished. Returns per-step results.
    A missed step doesn't hide the rest: the next step is searched from where the missed one started, so the user
    sees which steps were done. The check still fails unless every step is done, in order."""
    out, i, base = [], 0, baselines(signals)
    for action in steps:
        step, at, start = Step(action), None, i
        while i < len(signals):
            i += 1
            if step.feed(signals[i - 1], base, times[i - 1]):
                at = round(times[i - 1], 2)
                break
        if at is None:
            i = start
        out.append({"action": action, "done": at is not None, "at_s": at})
    return out


class LiveGuide:
    """check_steps, one frame at a time, for live on-screen guidance ("Blink now" → ✓ → next step).
    Same thresholds as the server; the server still verifies the recorded video and makes the decision."""

    def __init__(self, steps: list):
        self.steps, self.i, self.step = list(steps), 0, Step(steps[0]) if steps else None
        self.start_frames, self.done_at = [], []

    def retake(self):
        """Start the current step over (the user is asked to do it again)."""
        if not self.finished:
            self.step = Step(self.steps[self.i])

    @property
    def finished(self) -> bool:
        return self.i >= len(self.steps)

    @property
    def current(self):
        return None if self.finished else self.steps[self.i]

    def feed(self, s: dict, t: float) -> bool:
        """True when this frame completed the current step."""
        if s.get("face") and len(self.start_frames) < BASELINE_FRAMES:
            self.start_frames.append(s)  # start face, as in baselines()
            return False
        if self.finished or len(self.start_frames) < BASELINE_FRAMES:
            return False
        if self.step.feed(s, baselines(self.start_frames), t):
            self.done_at.append(round(t, 2))
            self.i += 1
            self.retake()  # the next step starts fresh
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
    if cap.get(cv2.CAP_PROP_FRAME_WIDTH) * cap.get(cv2.CAP_PROP_FRAME_HEIGHT) > MAX_FRAME_PIXELS:
        cap.release()
        raise OverflowError("video frames larger than 4K")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = fps if 1 <= fps <= 240 else 30.0
    signals, times, embs, emb_times, tried, multi, best, n, next_t = [], [], [], [], 0, 0, None, 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            if n > MAX_FRAMES:
                raise OverflowError(f"video has more than {MAX_FRAMES} frames")
            # the frame's own timestamp: browser recordings (MediaRecorder) have a variable frame rate
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 or (n - 1) / fps
            if t > MAX_VIDEO_S:
                raise OverflowError(f"video longer than {MAX_VIDEO_S} s")
            if t + 1e-3 < next_t:
                continue
            next_t = max(next_t + 1 / SAMPLE_FPS, t)  # ~SAMPLE_FPS whatever the source frame rate
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
            # a wide-open mouth isn't sampled at all: SFace scores the same person 0.20-0.30 then (vs 0.86+ closed),
            # and it hides nothing. The open_mouth step is still identity-checked by samples within STEP_ID_WINDOW_S
            if s.get("face") and len(signals) % 3 == 1 and s["jaw"] <= WIDE_MOUTH:
                tried += 1  # hand-over-face frames count as tried-but-unconfirmed, so a raised hand can't hide a face
                emb, _ = (None, "hand_over_face") if s["hand_in_face"] else embed(frame)
                if emb is not None:
                    embs.append(emb)
                    emb_times.append(t)
                    # the reference everything is compared with: frontal AND neutral. A wide-open-mouth frame as the
                    # reference made the same user's normal face score 0.35 ("face_changed")
                    neutral = s["jaw"] < 0.2 and s["mouth_w"] < 1.1 * baseline(signals, "mouth_w")
                    facing = turned(s, baseline(signals)) < TURN / 2 and s["eyes_open"] and neutral
                    if facing and (best is None or s["eye_w"] > best[0]):  # clearest frontal frame -> profile match
                        best = (s["eye_w"], emb)
    finally:
        cap.release()
    if n == 0:  # opened but nothing decodes, e.g. AV1, which Chrome picks for webm unless told otherwise
        raise ValueError("couldn't decode the video: send H.264 MP4 or VP8 WebM")
    if not signals:
        raise ValueError("video has no frames")
    results = check_steps(steps, signals, times)
    for r in results:  # an action only counts if we also confirmed WHO did it (no blurry-stand-in gap)
        r["identity_checked"] = r["done"] and any(abs(t - r["at_s"]) <= STEP_ID_WINDOW_S for t in emb_times)
    return {"steps": results, "frames_analyzed": len(signals),
            "identity_coverage": round(len(embs) / tried, 2) if tried else 0.0,
            "duration_s": round(times[-1], 2), "multiple_faces_share": multi / len(signals),
            "identity_samples": embs, "best_embedding": best[1] if best else None}
