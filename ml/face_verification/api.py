import os
import tempfile
from functools import lru_cache

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from common.utils import load_artifact
from face_verification.face import decide, embed
from face_verification.liveness import IDENTITY_COVERAGE, analyze_video, new_challenge, use_challenge

router = APIRouter(prefix="/verify", tags=["face verification"])
MAX_BYTES = 8 * 1024 * 1024
MAX_VIDEO_BYTES = 40 * 1024 * 1024


@lru_cache
def artifact():
    try:
        return load_artifact("face_verification")
    except FileNotFoundError:
        raise HTTPException(503, "thresholds not calibrated: run python -m face_verification.train")


async def read_image(f: UploadFile) -> np.ndarray:
    data = await f.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"{f.filename}: image over 8 MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"{f.filename}: not an image")
    return img


async def profile_scores(profile_photos, emb):
    scores = []
    for photo in profile_photos:
        e, _ = embed(await read_image(photo))
        scores.append(None if e is None else round(float(emb @ e), 3))
    return scores


@router.post("/face")
async def verify(selfie: UploadFile = File(...), profile_photos: list[UploadFile] = File(...)):
    """Face MATCH only, no liveness: never grant the badge on this alone. Use /verify/liveness.
    Images are processed in memory and never written to disk (DPDP: keep no raw biometrics)."""
    art = artifact()
    selfie_emb, reason = embed(await read_image(selfie))
    if selfie_emb is None:
        return {"decision": "retry", "reason": reason, "liveness_checked": False}
    scores = await profile_scores(profile_photos, selfie_emb)
    best = max((s for s in scores if s is not None), default=None)
    if best is None:
        return {"decision": "retry", "reason": "no_face_in_profile_photos", "liveness_checked": False}
    return {"decision": decide(best, art["thresholds"]), "best_similarity": best, "per_photo": scores,
            "model_version": art["model_version"],
            "liveness_checked": False}  # ponytail: no anti-spoofing yet; do not grant the badge on this alone


@router.post("/challenge")
def challenge():
    """Step 1: get random actions to perform. Single use, expires."""
    return new_challenge()


@router.post("/liveness")
async def liveness(challenge_id: str = Form(...), video: UploadFile = File(...),
                   profile_photos: list[UploadFile] = File(...)):
    """Step 2: one video doing the challenge + profile photos -> liveness, same-person check, face match."""
    try:
        steps = use_challenge(challenge_id)  # consumed now, even if the rest fails
    except KeyError as e:
        raise HTTPException(404, str(e))
    except TimeoutError as e:
        raise HTTPException(410, str(e))
    art = artifact()
    data = await video.read(MAX_VIDEO_BYTES + 1)
    if len(data) > MAX_VIDEO_BYTES:
        raise HTTPException(413, "video over 40 MB")
    # OpenCV reads video from a path: temp file, deleted right after analysis (DPDP: no raw biometrics kept)
    fd, path = tempfile.mkstemp(suffix=os.path.splitext(video.filename or "")[1] or ".mp4")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        del data
        res = await run_in_threadpool(analyze_video, path, steps)
    except OverflowError as e:
        raise HTTPException(413, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        os.remove(path)

    t = art["thresholds"]
    samples, best = res.pop("identity_samples"), res.pop("best_embedding")
    continuity = round(min(float(best @ e) for e in samples), 3) if best is not None else None
    passed = all(s["done"] and s["identity_checked"] for s in res["steps"])
    out = {"liveness": {"passed": passed, "steps": res["steps"]}, "same_person_throughout": None,
           "min_continuity": continuity, "frames_analyzed": res["frames_analyzed"],
           "duration_s": res["duration_s"], "identity_coverage": res["identity_coverage"], "model_version": art["model_version"] + "+mp-landmarker-1.0.1",
           "liveness_checked": True}
    if res["multiple_faces_share"] > 0.2:
        return {**out, "decision": "fail", "reason": "multiple_faces"}
    if best is None or len(samples) < 2 or res["identity_coverage"] < IDENTITY_COVERAGE:
        return {**out, "decision": "retry", "reason": "no_clear_frame"}  # too blurry/dark to confirm who is in it
    out["same_person_throughout"] = continuity >= t["review"]
    if not out["same_person_throughout"]:
        return {**out, "decision": "fail", "reason": "face_changed"}
    if not passed:
        return {**out, "decision": "retry", "reason": "challenge_not_completed"}
    scores = await profile_scores(profile_photos, best)
    top = max((s for s in scores if s is not None), default=None)
    if top is None:
        return {**out, "decision": "retry", "reason": "no_face_in_profile_photos"}
    return {**out, "decision": decide(top, t), "best_similarity": top, "per_photo": scores}
