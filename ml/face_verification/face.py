"""Face verification: detect (YuNet) -> quality gate -> align -> embed (SFace) -> cosine.

Both models are Apache-2.0 (OpenCV Zoo). Shared by train.py and api.py.
NOT included: liveness / anti-spoofing. Without it a printed photo or a screen can pass.
Add MiniFASNet + a random head-turn challenge, or AWS Rekognition Face Liveness, before launch (plan §7).
"""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import requests

MODELS = Path(__file__).resolve().parents[1] / "artifacts"
URLS = {
    "yunet": "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
    "sface": "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
}
MIN_FACE_PX, MIN_SHARPNESS, BRIGHTNESS = 60, 30.0, (40, 220)


def model_path(name: str) -> Path:
    path = MODELS / Path(URLS[name]).name
    if not path.exists():
        MODELS.mkdir(exist_ok=True)
        r = requests.get(URLS[name], timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


@lru_cache
def models():
    detector = cv2.FaceDetectorYN.create(str(model_path("yunet")), "", (320, 320), 0.8)
    recognizer = cv2.FaceRecognizerSF.create(str(model_path("sface")), "")
    return detector, recognizer


def embed(img_bgr: np.ndarray):
    """Returns (embedding or None, reason). Rejects images we shouldn't judge rather than guessing."""
    detector, recognizer = models()
    h, w = img_bgr.shape[:2]
    detector.setInputSize((w, h))
    faces = detector.detect(img_bgr)[1]
    if faces is None:
        return None, "no_face"
    face = max(faces, key=lambda f: f[2] * f[3])  # largest face
    if min(face[2], face[3]) < MIN_FACE_PX:
        return None, "face_too_small"
    x, y, fw, fh = (int(max(0, v)) for v in face[:4])
    gray = cv2.cvtColor(img_bgr[y:y + fh, x:x + fw], cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < MIN_SHARPNESS:
        return None, "too_blurry"
    if not BRIGHTNESS[0] <= gray.mean() <= BRIGHTNESS[1]:
        return None, "bad_lighting"
    feat = recognizer.feature(recognizer.alignCrop(img_bgr, face)).ravel()
    return feat / np.linalg.norm(feat), "ok"


def decide(similarity: float, thresholds: dict) -> str:
    """Three bands: only clearly-same faces pass automatically; the grey zone goes to a human."""
    if similarity >= thresholds["pass"]:
        return "pass"
    return "review" if similarity >= thresholds["review"] else "fail"
