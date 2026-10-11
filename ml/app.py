"""BuddySync ML service: one FastAPI app, one router per feature.

Run from ml/:  uvicorn app:app --port 8000     (docs at http://localhost:8000/docs)
Spring Boot calls these endpoints; if the service is down it falls back to non-ML sorting.
ML_SERVICE_TOKEN set → every /v1 call needs the header X-Service-Token (except /v1/go, which users' browsers open).
"""
import hmac
import os
import threading
import urllib.request
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from buddy_matching.api import router as buddy
from clan_formation.api import router as clans
from event_ingestion.api import router as ingestion
from event_recs.api import router as recs
from face_verification.api import router as face
from moderation.api import router as moderation
from trip_planner import worker
from trip_planner.api import go_router
from trip_planner.api import router as trips
from trust.api import router as trust

PUBLIC = {"/v1/go"}  # exact paths: a future /v1/gofoo must not become public by accident
MAX_BODY = 96 * 1024 * 1024  # largest real request: a 40 MB video + 5 photos of 8 MB; caps temp-disk use too


def keep_awake(url, every=13 * 60):
    """Render's free plan sleeps after 15 min without inbound traffic, so ping our own public URL every 13."""
    done = threading.Event()

    def loop():
        while not done.wait(every):
            try:
                urllib.request.urlopen(url + "/health", timeout=30).close()
            except OSError:
                pass  # next ping retries

    threading.Thread(target=loop, daemon=True).start()
    return done.set


@asynccontextmanager
async def lifespan(app):
    # trip planner jobs: in-process workers for dev; production sets TRIP_WORKERS=0 and runs trip_planner.worker
    stop = worker.start(int(os.getenv("TRIP_WORKERS", "2")))
    url = os.getenv("RENDER_EXTERNAL_URL")  # Render sets it; unset locally, so no pinging
    stop_ping = keep_awake(url) if url else (lambda: None)
    yield
    stop_ping()
    stop()


# with a token set (deployed), the schema isn't public; API.md is the reference
_hide = bool(os.getenv("ML_SERVICE_TOKEN"))
app = FastAPI(title="BuddySync ML", version="0.1", lifespan=lifespan,
              docs_url=None if _hide else "/docs", redoc_url=None, openapi_url=None if _hide else "/openapi.json")
for router in (buddy, clans, recs, ingestion, face, moderation, trust, trips, go_router):
    app.include_router(router, prefix="/v1")


@app.middleware("http")
async def service_token(request: Request, call_next):
    token = os.getenv("ML_SERVICE_TOKEN")
    path = request.url.path
    # body size before anything reads it: Starlette spools every uploaded file to disk in full
    size = request.headers.get("content-length")
    if size is None and request.headers.get("transfer-encoding"):
        return JSONResponse({"detail": "Content-Length required"}, status_code=411)
    if size is not None and (not size.isdigit() or int(size) > MAX_BODY):
        return JSONResponse({"detail": f"request over {MAX_BODY // 2**20} MB"}, status_code=413)
    if token and path.startswith("/v1/") and path not in PUBLIC:
        if not hmac.compare_digest(request.headers.get("x-service-token", ""), token):
            return JSONResponse({"detail": "missing or wrong X-Service-Token"}, status_code=401)
    return await call_next(request)


@app.get("/health")
def health():
    return {"ok": True}
