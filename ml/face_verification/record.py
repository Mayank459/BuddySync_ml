"""Record a liveness challenge with your webcam and send it to the running API (a real-person test).

Run from ml/ with the API up:   python -m face_verification.record [http://127.0.0.1:8765]
1. A window opens: face the camera and press SPACE to take your profile photo.
2. It fetches a random challenge and counts down 3 s, then records while showing each step on screen.
   Do the steps in order, holding each for about a second. Press SPACE when you're done (max 18 s).
3. It uploads the video and photo to /v1/verify/liveness and prints the verdict.
Press Q or ESC at any time to quit. The recording is saved to data/liveness_test.mp4 for re-runs.
"""
import json
import sys
import time

import cv2
import requests

from common.utils import DATA

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
MAX_S, FPS = 18, 20


def text(img, lines, color=(255, 255, 255)):
    out = img.copy()
    for n, line in enumerate(lines):
        y = 38 + n * 34
        cv2.putText(out, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(out, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return out


def main():
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        sys.exit("no webcam found (is another app using it?)")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    win = "BuddySync liveness test"

    photo = None  # 1. profile photo
    while photo is None:
        ok, frame = cam.read()
        if not ok:
            sys.exit("webcam gave no frame")
        cv2.imshow(win, text(frame, ["Step 1: profile photo", "Face the camera, good light. Press SPACE"]))
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            return
        if k == 32:
            photo = frame.copy()
    ok, jpg = cv2.imencode(".jpg", photo)

    ch = requests.post(f"{BASE}/v1/verify/challenge", timeout=10).json()  # 2. challenge
    steps = [s["prompt"] for s in ch["steps"]]
    print("challenge:", ch["challenge_id"], "|", " → ".join(s["action"] for s in ch["steps"]))
    end = time.time() + 3
    while time.time() < end:
        ok, frame = cam.read()
        lines = ["Get ready: face the camera", f"Recording in {int(end - time.time()) + 1}..."]
        cv2.imshow(win, text(frame, lines + [f"{n + 1}. {s}" for n, s in enumerate(steps)], (0, 255, 255)))
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            return

    path = DATA / "liveness_test.mp4"
    h, w = frame.shape[:2]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    start, frames = time.time(), 0
    while time.time() - start < MAX_S:
        ok, frame = cam.read()
        if not ok:
            break
        out.write(frame)  # the saved video has no overlay
        frames += 1
        elapsed = time.time() - start
        lines = [f"REC {elapsed:4.1f}s  (SPACE = done)", "Start facing the camera, then:"]
        lines += [f"{n + 1}. {s}" for n, s in enumerate(steps)] + ["...then face the camera again"]
        cv2.imshow(win, text(frame, lines, (80, 80, 255)))
        k = cv2.waitKey(max(1, int(1000 / FPS) - 5)) & 0xFF
        if k in (ord("q"), 27):
            out.release()
            return
        if k == 32 and elapsed > 3:
            break
    out.release()
    cam.release()
    cv2.destroyAllWindows()
    print(f"recorded {frames} frames in {time.time() - start:.1f} s → {path}")

    print("analysing… (about 5–10 s)")  # 3. verify
    with open(path, "rb") as v:
        r = requests.post(f"{BASE}/v1/verify/liveness", timeout=120, data={"challenge_id": ch["challenge_id"]},
                          files=[("video", ("liveness_test.mp4", v.read(), "video/mp4")),
                                 ("profile_photos", ("profile.jpg", jpg.tobytes(), "image/jpeg"))])
    print(r.status_code)
    print(json.dumps(r.json(), indent=1))


if __name__ == "__main__":
    main()
