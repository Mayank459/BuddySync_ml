# BuddySync ML

The machine-learning service behind **BuddySync**, an app for finding people to go to live events with, in India.
It ranks buddies, forms clans, recommends events, cleans up event listings, verifies faces with a liveness
challenge, moderates chat, and plans the whole trip around a show.

It's one FastAPI service. The Spring Boot backend calls it; the React app never calls it directly.

```
React ──► Spring Boot ──(X-Service-Token, user_id)──► BuddySync ML (FastAPI, this repo)
                                                          │   ├─ in-process workers (trip drafts and chat turns)
                                                          ▼   │
                                                   PostgreSQL (ml schema): trips, versions, chat, job queue, events
                                                          │
                          external: Groq / Gemini (LLM) · LiteAPI (hotel rates) · Travelpayouts (flight fares)
                                    Open-Meteo (weather) · Whisper on Groq (voice) · OpenStreetMap (places)
```

---

## What's inside

| Feature | Endpoint(s) | How it works |
|---|---|---|
| **Buddy matching** | `POST /v1/match/buddies` | Reciprocal score: each side has to fit the other's preferences. Weights tuned on held-out users. Hard filters are checked both ways, and popular users are ranked down a little to spread requests. |
| **Clan formation** | `POST /v1/clans/form` | Greedy grouping where **every pair** clears a minimum score (a batch job) |
| **Event recommendations** | `POST /v1/recs/events` | A taste vector plus tag popularity, with a diversity cap, tuned on a time-based split |
| **Event ingestion** | `POST /v1/events/ingest` | A duplicate-pair classifier (time, distance, text) and a tf-idf category model |
| **Face verification** | `POST /v1/verify/challenge`, `/liveness`, `/face` | Random actions (blink, touch your nose, fingers…) in one video. Continuity and identity checks, then YuNet + SFace face match. Attack videos are rejected. |
| **Chat moderation** | `POST /v1/moderate/message` | Character n-gram toxicity model plus regex scam rules (UPI IDs, money asks, off-platform moves). Laya semantic check optional. |
| **Trip planner** | `POST /v1/trips` … (23 endpoints) | An LLM with tools over a real places catalog; code enforces every rule. Groups, votes, live prices, Hinglish, voice. |

Full contract: **[ml/API.md](ml/API.md)** (every input, output and error). Design: **[docs](#documentation)**.

### The trip planner in one paragraph
Pick an event and a home city. You get a plan built around **the show**:
- how you get there: a train fare from the official Railway Board chart, or a live cached flight fare
- a hotel near the venue at a **live rate** (LiteAPI)
- food before and after the show, skipping places known to be closed by then
- a way back after a late show, with a **last-metro warning** and a cab link
- things to do the next day, depending on the weather

Then you edit it in chat, in English, Hindi or Hinglish, or by voice. For a clan:
- each member gets their own travel legs and costs
- the group votes on budget, balanced or premium
- anyone can start a decision: a majority passes it, otherwise the most votes win after 24 h
- locked slots can't be changed without another vote, not even by the AI

We never book or take payment. Every **Book** button hands off to the provider through a tracked affiliate link.

---

## Quick start (local)

Python 3.12. Models (`ml/artifacts/`) and demo data (`ml/data/`) are committed, so there's nothing to train first.

```bash
cd ml
pip install -r requirements.txt
python -m uvicorn app:app --port 8765          # API: http://127.0.0.1:8765/docs
python -m streamlit run playground.py          # a tab per feature: http://127.0.0.1:8501
python -m pytest -q                            # 68 tests; no network, no API keys (Postgres is embedded)
```

No database needed locally: without `DATABASE_URL` the trip planner starts an embedded Postgres (`pgserver`) in `ml/data/pgdata/`.
The trip planner's chat needs an LLM key. Everything else works without keys (drafts fall back to a code-only planner).

To give it live prices, a model and voice, copy `ml/demo_env.example.ps1` to `demo_env.ps1`, fill in the keys, and load it (`. .\demo_env.ps1`). For bash, `export` the same variables.

## Deploy on Render

The repo is a Render **Blueprint** ([render.yaml](render.yaml)): one Docker web service and one Postgres.

1. Render dashboard → **New → Blueprint** → choose this repo.
2. Fill in the secrets it asks for:
   - `LLM_API_KEY`: a Groq key
   - `LLM_FALLBACK_API_KEY`: a Gemini key
   - `LITEAPI_KEY`
   - `TRAVELPAYOUTS_TOKEN`
   - `STT_API_KEY`: Groq
3. Deploy. The build installs the requirements (about 5–8 min the first time), and `/health` is the health check.
4. **Service token:** Render generates `ML_SERVICE_TOKEN`. Every `/v1` call must send it as `X-Service-Token` (`/v1/go` is public, because users' browsers open it). Give it to Spring. To run the playground or smoke test against Render, set it locally:
   ```bash
   ML_SERVICE_TOKEN=<token> API_URL=https://<service>.onrender.com python -m streamlit run playground.py
   ML_SERVICE_TOKEN=<token> python smoke_test.py https://<service>.onrender.com
   ```

**Sizing:** about 250 MB for the trip planner alone, and about 430 MB once face verification and liveness have loaded their models.
- The blueprint uses the **free** plan (512 MB) with one process: fine for a demo.
- Use **standard** (2 GB) for real traffic, or to raise `WEB_CONCURRENCY`.
- Free instances sleep after 15 min idle, and the first request then takes about a minute. Open `/health` before a demo.

**Webcam liveness check:** the playground's check uses the camera of the machine *running the playground*. Run the playground on a laptop and point `API_URL` at Render.

**Background jobs:**
- **Drafts and chat turns:** the API runs them itself (`TRIP_WORKERS`, 2 threads).
- **At scale:** run `python -m trip_planner.worker` as a separate Render background worker, and set `TRIP_WORKERS=0` on the web service.
- **Nightly (optional, Render cron job):** `python -m trip_planner.pregen --top 20` pre-builds drafts for popular events.

## Configuration

The main settings. Every variable is in [ml/API.md §7.24](ml/API.md#724-configuration-environment).

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Postgres for the trip planner. Unset → embedded dev database. |
| `ML_SERVICE_TOKEN` | Require `X-Service-Token` on `/v1` calls |
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | OpenAI-compatible LLM. `LLM_MODEL` is a comma list: on a 429 or 5xx the next model is tried at once. |
| `LLM_FALLBACK_*` | Second provider. A chat turn switches to it mid-turn if the first runs out of quota. |
| `ANTHROPIC_API_KEY` | Use Claude instead |
| `LITEAPI_KEY` · `TRAVELPAYOUTS_TOKEN` / `TRAVELPAYOUTS_MARKER` · `STT_API_KEY` | Hotel rates · flight fares and affiliate ID · voice |
| `TRIP_WORKERS`, `WEB_CONCURRENCY` | Worker threads per process · API processes |
| `USE_LAYA=1` | Optional semantic scam and jailbreak check (needs `pip install laya`) |

---

## Repository layout

```
render.yaml                 Render blueprint (web service + Postgres)
Dockerfile                  the service image (build from the repo root)
ML_IMPLEMENTATION_PLAN.md   design for every ML feature: decisions, data, evaluation, roadmap
TRIP_PLANNER_DESIGN.md      the trip planner's full design (§18 tracks what's built)
DEMO.md                     3-minute demo script
ml/
  app.py                    FastAPI app: routers, service-token check, trip workers
  API.md                    the API contract
  playground.py             Streamlit UI for every endpoint (webcam liveness, trip planner, clans…)
  smoke_test.py             HTTP checks against a running server
  buddy_matching/ clan_formation/ event_recs/ event_ingestion/ moderation/ face_verification/
                            one folder per feature: train.py, api.py, tests
  trip_planner/
    model.py                trip, timeline, prices, costs, the rule checker (pure)
    tools.py                what the LLM (and card actions) can do; every edit is checked
    planner.py              draft pipeline, chat loop, injection guard, masking, provider failover
    groups.py               members, group decisions, variants, votes
    llm.py                  Claude or OpenAI-compatible providers, model and provider fallback
    store.py  schema.sql    Postgres: trips, versions, chat, job queue, events, cache
    worker.py               runs drafts, turns and decisions; sweeps decision deadlines
    hotels.py fares.py rail.py weather.py transit.py links.py   data sources and hand-off links
    evals.py pregen.py demo.py                                  quality, cache warm-up, demo kit
  common/                   synthetic data generator, shared helpers
  artifacts/                trained models (small) and face/landmark model files
  data/                     synthetic users, events and chats; OSM + LiteAPI place catalogs; rail fare table
```

## Quality

- **68 unit and integration tests** (`python -m pytest -q`). The trip planner tests run against a real, embedded Postgres, and the LLM is faked.
- **Trip planner evals:**
  - `python -m trip_planner.evals --offline` runs every event × 3 origin cities: **1,990 plans, 0 invented places, 0 broken rules**.
  - `--quick` and the full set run live against the LLM, with edit chats, Hinglish and attack prompts. Latest `--quick` run: 22/23. The one miss was a bug in the eval itself, since fixed.
- **Smoke test** (`python smoke_test.py <url>`): 32 HTTP checks across every feature.
- **Face verification:** `python -m face_verification.attack_check` runs photo, swap and stand-in attack videos, and all must be rejected.

## Engineering decisions

- **The LLM chooses, code decides.** The planner may only pick place IDs its tools returned. Times, prices, opening hours, budgets, locks and safety rules are checked in code on every edit. That makes invented places impossible by construction, not just unlikely.
- **Postgres is the job queue** (`FOR UPDATE SKIP LOCKED`), the per-trip lock and the event log behind the SSE stream. That's one less system to run at our scale. Move to LISTEN/NOTIFY or Redis when streams reach the hundreds.
- **Built for free tiers:**
  - model lists fail over instantly, and turns switch provider mid-turn
  - drafts are cached per event and preferences (origin cities don't change them), and pre-built overnight
  - a code-only planner is used when no LLM is available
  - Gemini's free tier is about 20 requests per day per model. Free tiers are for demos; use a paid tier for real users.
- **Hand off, never transact.** Affiliate links (Aviasales with the Travelpayouts marker, Booking.com, Agoda) through a logged, allow-listed redirect. No PCI, IATA or payment-aggregator exposure.
- **Honest numbers.** Every price carries `estimate`, `range` or `verified`, its source and when it was fetched. Train fares come from the official fare chart with an estimated rail distance, and hotel rates from LiteAPI (a sandbox key gives test inventory).
- **Privacy (DPDP):**
  - phone numbers, emails, UPI IDs and card numbers are masked before text reaches an LLM
  - chats are kept, with erasure on request (`DELETE /v1/trips/users/{id}`)
  - face video is deleted as soon as it's analysed
  - we never infer gender, so single-gender hostels are never suggested

## Known limitations

- The models are trained on **synthetic data**. The pipelines work end to end, but the scores aren't tuned to real users.
- **Places catalogs:** 5 cities (Mumbai, Delhi, Bengaluru, Pune, Hyderabad), built around each city centre.
- **Flights:** cheapest cached fares, not live seats. **Trains:** fare estimates, no timetable.
- **Travel time:** straight-line distance × 1.3 until OSRM is deployed.
- **The last-metro table is city-level** and needs refreshing every quarter.
- **Liveness thresholds** still need tuning on about 30 real recordings per action. The face thresholds are calibrated on LFW and should be recalibrated on consented Indian photo pairs.

## Documentation

| Doc | For |
|---|---|
| [ml/API.md](ml/API.md) | Backend and frontend developers: every endpoint's inputs, outputs and errors |
| [ml/README.md](ml/README.md) | ML developers: training pipelines, scheduled jobs, gaps marked `ponytail:` in the code |
| [ML_IMPLEMENTATION_PLAN.md](ML_IMPLEMENTATION_PLAN.md) | The design and reasoning for every ML feature |
| [TRIP_PLANNER_DESIGN.md](TRIP_PLANNER_DESIGN.md) | The trip planner's finished design, and its build status (§18) |
| [DEMO.md](DEMO.md) | Presenting the trip planner in 3 minutes |
