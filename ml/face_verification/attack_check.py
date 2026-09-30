"""Attack videos built from LFW photos, sent through /v1/verify/liveness. Every one must be rejected.

Run from ml/ (after face_verification.train has cached LFW):  python -m face_verification.attack_check
Writes temp videos to artifacts/attack_*.mp4 and deletes them afterwards.
A real person doing the challenge can't be simulated here: record one and use the playground's Liveness tab.
"""
import cv2
import numpy as np
from fastapi.testclient import TestClient
from sklearn.datasets import fetch_lfw_pairs

from app import app
from common.utils import ARTIFACTS, DATA

lfw = fetch_lfw_pairs(subset="test", color=True, resize=1.0, slice_=None, data_home=DATA)
px = lfw.pairs if lfw.pairs.max() > 1.5 else lfw.pairs * 255
bgr = lambda im: cv2.resize(np.ascontiguousarray(im[..., ::-1]).astype(np.uint8), (500, 500))
same = np.where(lfw.target == 1)[0][0]
A, A2 = bgr(px[same][0]), bgr(px[same][1])
from face_verification.face import embed  # noqa: E402
from face_verification.liveness import frame_signals, landmarks  # noqa: E402


def usable(img):  # sharp enough to embed, and no hand in the photo
    return embed(cv2.resize(img, (360, 360)))[0] is not None and frame_signals(*landmarks(img)[:3]).get("nose_touch", 0) > 2


B = next(bgr(px[i][1]) for i in np.where(lfw.target == 0)[0] if usable(bgr(px[i][1])))  # a different, clean face
BLUR = cv2.GaussianBlur(B, (0, 0), 6)
FPS = 15


def write(name, frames):
    path = ARTIFACTS / f"attack_{name}.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (640, 480))
    for f in frames:
        w.write(f)
    w.release()
    return path


def on_canvas(img, dx=0, dy=0, zoom=1.0, tilt=0.0):
    """Paste the 'printed photo' on a grey background, shifted/zoomed, with a perspective tilt (paper turned)."""
    s = int(360 * zoom)
    ph = cv2.resize(img, (s, s))
    k = tilt * s  # squeeze one vertical edge: what turning a flat photo looks like
    src = np.float32([[0, 0], [s, 0], [s, s], [0, s]])
    dst = np.float32([[0, 0], [s, k], [s, s - k], [0, s]])
    ph = cv2.warpPerspective(ph, cv2.getPerspectiveTransform(src, dst), (s, s), borderValue=(128, 128, 128))
    canvas = np.full((480, 640, 3), 128, np.uint8)
    x, y = 320 - s // 2 + dx, 240 - s // 2 + dy
    canvas[max(0, y):y + s, max(0, x):x + s] = ph[max(0, -y):480 - y, max(0, -x):640 - x][:canvas[max(0, y):y + s, max(0, x):x + s].shape[0], :canvas[max(0, y):y + s, max(0, x):x + s].shape[1]]
    return canvas


n = FPS * 6
attacks = {
    "static_photo": [on_canvas(A)] * n,
    "moving_tilted_photo": [on_canvas(A, dx=int(30 * np.sin(i / 4)), dy=int(15 * np.cos(i / 5)),
                                      zoom=1 + 0.15 * np.sin(i / 7), tilt=0.12 * np.sin(i / 6)) for i in range(n)],
    "person_swap": [on_canvas(A)] * (n // 2) + [on_canvas(B)] * (n // 2),
    "hand_up_stand_in": [on_canvas(A)] * (n // 3) + [on_canvas(bgr(px[np.where(lfw.target == 0)[0][0]][1]))] * (2 * n // 3),
    "blurry_stand_in": [on_canvas(A)] * (n // 3) + [on_canvas(BLUR)] * (2 * n // 3),
}
expected_ok = {"retry", "fail"}  # anything but pass/review
client, failures = TestClient(app), 0
for name, frames in attacks.items():
    path = write(name, frames)
    try:
        c = client.post("/v1/verify/challenge").json()
        with open(path, "rb") as v:
            r = client.post("/v1/verify/liveness", data={"challenge_id": c["challenge_id"]},
                            files=[("video", (path.name, v.read(), "video/mp4")),
                                   ("profile_photos", ("p.jpg", cv2.imencode(".jpg", A2)[1].tobytes(), "image/jpeg"))]).json()
    finally:
        path.unlink()
    ok = r.get("decision") in expected_ok
    failures += not ok
    steps = ", ".join(f"{s['action']}={'Y' if s['done'] else 'n'}" for s in r.get("liveness", {}).get("steps", []))
    print(f"{'PASS' if ok else 'FAIL'}  {name:20s} -> {r.get('decision')} ({r.get('reason')})  steps: {steps}  "
          f"continuity={r.get('min_continuity')}  coverage={r.get('identity_coverage')}")
raise SystemExit(1 if failures else 0)
