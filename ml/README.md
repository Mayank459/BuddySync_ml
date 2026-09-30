# BuddySync ML service

Seven features, one folder each. Every folder follows the same steps:
**data → preprocess → train/test split → train → test → deploy (FastAPI router)**.
Design and reasoning: [../ML_IMPLEMENTATION_PLAN.md](../ML_IMPLEMENTATION_PLAN.md).

| Folder | What it does | "Training" | Endpoint |
|---|---|---|---|
| `buddy_matching/` | Rank people at the same event, scoring both directions | tune the 4 weights + logistic regression, split by viewer | `POST /v1/match/buddies` |
| `clan_formation/` | Group 3–6 people so every pair is compatible | none (reuses buddy score); greedy vs random check | `POST /v1/clans/form` |
| `event_recs/` | Personal event feed | taste vectors + tag popularity; weights tuned on a validation window; time split | `POST /v1/recs/events` |
| `event_ingestion/` | Spot duplicate listings across sources, categorise events | pair classifier + tf-idf/logistic regression categoriser | `POST /v1/events/ingest` |
| `face_verification/` | Liveness video challenge (random actions) + same-person check + face match → pass / review / fail / retry | calibrate thresholds on LFW train pairs, test on LFW test pairs; `attack_check.py` runs attack videos | `POST /v1/verify/challenge`, `/v1/verify/liveness` (`/v1/verify/face` = photo match only) |
| `moderation/` | Chat: block / nudge / flag / allow | char-n-gram tf-idf + logistic regression, plus regex scam rules | `POST /v1/moderate/message` |
| `trip_planner/` | Trip around the event: draft, then chat edits (LLM + tools, every edit checked by code), stored in Postgres, run by queued workers ([design](../TRIP_PLANNER_DESIGN.md)) | none; the data step builds a real places catalog from OpenStreetMap | `POST /v1/trips`, `POST /v1/trips/{id}/messages`, `GET /v1/trips/{id}/stream` |

`common/` holds the synthetic data generator and small shared helpers.

## Run

```bash
cd ml
pip install -r requirements.txt

python -m common.synthetic            # 1. data (synthetic until real logs exist)
python -m buddy_matching.train        # 2-5. preprocess, split, train, test, save to artifacts/
python -m event_recs.train
python -m event_ingestion.train
python -m moderation.train
python -m face_verification.train     # downloads LFW (~230 MB) + OpenCV face models, first run only
python -m face_verification.attack_check  # photo / swap / stand-in attack videos: all must be rejected
python -m clan_formation.clans        # sanity check: greedy vs random clans
python -m trip_planner.catalog Mumbai # places catalog per event city

python -m pytest -q                   # checks for every feature (no network or API key needed; trip tests start an embedded Postgres)

# trip planner LLM (without one, drafts are code-only and chat returns an error event). Gemini free tier:
export LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export LLM_MODEL=gemini-3.8-flash,gemini-flash-lite-latest   # comma list = fallbacks on 429/503
export LLM_CHAT_MODEL=gemini-flash-lite-latest,gemini-3.8-flash
export LLM_API_KEY=...                # or ANTHROPIC_API_KEY=... for Claude. Free tiers may train on your data.
# optional second provider, tried last: LLM_FALLBACK_BASE_URL=https://api.groq.com/openai/v1 LLM_FALLBACK_MODEL=qwen/qwen3.8-27b LLM_FALLBACK_API_KEY=...
# trip storage: DATABASE_URL=postgresql://...  (unset = embedded dev Postgres in data/pgdata)
# flight fare ranges: TRAVELPAYOUTS_TOKEN=...  (+ TRAVELPAYOUTS_MARKER=<partner id> for affiliate links)
# hotel rates: LITEAPI_KEY=sand_... or prod_...; merge LiteAPI hotels into a catalog: python -m trip_planner.catalog Mumbai --hotels
# voice input: STT_API_KEY=<groq key>   ·   lock the API to Spring: ML_SERVICE_TOKEN=<secret> (Spring sends X-Service-Token)
# train fares table (already built into data/rail_fares.csv): python -m trip_planner.rail path/to/TAG_2026_Fares.pdf
export USE_LAYA=1                     # optional: semantic scam check in moderation (pip install laya; ~70 ms/msg on CPU)
uvicorn app:app --port 8000           # 6. deploy; open http://127.0.0.1:8000/docs (also runs 2 trip worker threads)
python smoke_test.py http://127.0.0.1:8000   # HTTP checks against the running server
(cd .. && docker build -t buddysync-ml .) && docker run -p 8000:8000 -e DATABASE_URL -e LLM_BASE_URL -e LLM_MODEL -e LLM_API_KEY buddysync-ml
```

**Production trip workers:** run the API with `TRIP_WORKERS=0`, and scale the planner separately with `python -m trip_planner.worker --threads 4`. Both need the same `DATABASE_URL` and LLM variables.

**Trip planner jobs to schedule:**
```bash
python -m trip_planner.pregen --top 20      # nightly (e.g. 02:00): cached drafts for popular upcoming events
python -m trip_planner.evals --offline      # CI: every event × 3 origins, code-only (~4 min); exit 1 on a hard failure
python -m trip_planner.evals --quick        # before a prompt/model change: a small live set (fits a free tier)
python -m trip_planner.evals                # full live set (needs a paid LLM tier)
```

**Try endpoints by hand:** with the API running, start the Streamlit playground (binds to 127.0.0.1 only):
```bash
python -m streamlit run playground.py   # http://127.0.0.1:8501, one tab per endpoint
```

**Windows notes:**
- Call `127.0.0.1`, not `localhost`. `localhost` tries IPv6 first and adds about 2 s to every request.
- If port 8000 is taken, use `--port 8765`.

Measured latency on a laptop (p95): buddies 26 ms, recs 11 ms, moderation 2 ms, ingest 4 ms, clan formation for 45 people 178 ms.

## Read the numbers honestly

The data is **synthetic**: labels come from hidden rules in `common/synthetic.py`. The metrics show that each pipeline works end to end, **not** how good the models are on real users. Perfect scores (dedup, categories, moderation) mean the synthetic data is too easy. They are not results. Replace each `load("...")` in `common/utils.py` with reads from Postgres once the backend logs real data (plan §2).

## Known gaps, marked `ponytail:` in the code

- **Liveness has not yet been tested on a real person.** Attack videos are rejected, but a real recording has never been run. Record a few in the playground (Face → Liveness) and tune the thresholds at the top of `face_verification/liveness.py`. It doesn't stop live deepfakes through a virtual camera; add device attestation in the app. `/v1/verify/face` (photo only) still has no liveness.
- Face thresholds are calibrated on LFW (mostly Western celebrities). Recalibrate on consented Indian pairs and check error rates per skin tone.
- Buddy ranking: no trust multiplier or exploration slots yet (needs report data and account age).
- Event recs: no trending, "buddies going" or urgency terms yet.
- Moderation training data is a tiny synthetic seed. Swap in a real Hinglish abuse dataset and moderator decisions.
- Trip planner:
  - Hotel rates come from LiteAPI (`hotels.py`). The key is a sandbox one, so rates are test inventory labelled "range"; switch to a `prod_` key for verified rates. OSM hotels, food and sights use hand-set estimate bands (`model.py`).
  - Travel time is straight-line × 1.3 until OSRM is deployed.
  - Flights: cheapest cached fares from Travelpayouts (`fares.py`), not live seats. Trains: estimated fares from the Railway Board fare chart with rail distance estimated (`rail.py`); no timetable yet. Buses have links only.
  - **Gemini free tier: 20 requests per day per model** (measured Sep 2026). That's a few chat turns a day in total, so it's for development only. Turns fail over to the next model and provider (e.g. Groq) mid-turn.
  - The last-metro warning is city-level (`transit.py`). Refresh it every quarter.
  - Weather uses Open-Meteo's free, non-commercial tier; get the paid plan before launch.
  - The event stream polls Postgres every 0.5 s. Switch to LISTEN/NOTIFY when streams number in the hundreds.
  - Design §18 tracks what's built. Streams E–H and the rest of A are still to build.
- Place catalogs: run `python -m trip_planner.catalog <City>` for each new event city (Mumbai, Delhi, Bengaluru, Pune and Hyderabad are built).
- Text features use tf-idf. Switch to multilingual-e5 embeddings if the Hinglish test in plan §2.2 shows tf-idf falls short.
