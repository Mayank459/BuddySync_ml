# BuddySync: ML & Recommendation Implementation Plan

**Scope:** everything the Python/FastAPI ML service owns. That is buddy matching, clan formation, event recommendations, event ingestion ML, face verification, trust & safety ML and the AI trip planner, plus the data and infrastructure they need.
**Status:** plan, 29 Sep 2026 (about week 8 of the 14-week timeline in the Phase I report).
**Basis:** the Phase I report and project overview, plus four research tracks: reciprocal people-matching, event recsys & infrastructure, trust & safety, and AI trip planning. Sources are at the end. Items marked **(verify)** could not be confirmed and should be checked before you rely on them.

---

## 0. Decisions at a glance

| # | Decision | Changes vs. Phase I report |
|---|---|---|
| D1 | Keep the 40/30/20/10 weighted score as the **Phase 0 ranker**. Compute it **in both directions** (u→v and v→u) and combine with a harmonic mean, as in the RECON reciprocal recommender. | Buddy requests are two-sided. A one-sided score recommends people who will never accept. |
| D2 | Recommendation is a **pipeline**: candidates → hard filters → score → re-rank (trust, congestion, exploration). | Replaces the per-candidate "Suitable match? → Next candidate" loop in Figure 4. Candidates are scored as a batch instead. |
| D3 | **Log impressions and outcomes from day 1.** | Without these logs we can never train a model or evaluate one. This is the single most important item. |
| D4 | Scoring evolves **heuristic → logistic regression → LightGBM ranker → two-tower**. Each step needs a measured win before it replaces the one before. | The report said "extend later with interaction data". This defines how. |
| D5 | **PostgreSQL does most of the work**: pgvector for embeddings, PostGIS plus H3 cells for location. No separate vector DB, feature store or Kafka until measured load demands it. | Growth is by read replicas and caching, not new systems. |
| D6 | Text embeddings use **`intfloat/multilingual-e5-small`** (384-d, CPU, ONNX int8). | It handles Hindi and English. Hinglish must be tested on our own data. |
| D7 | Face verification uses **OpenCV YuNet + SFace** (Apache-2.0) plus a **liveness check**. It is not Haar/LBPH. Production can swap in AWS Rekognition Face Liveness behind the same endpoint. | Haar/LBPH can't reliably match faces and a printed photo fools them. The InsightFace weights are non-commercial. |
| D8 | The trip planner is a **deterministic workflow with an LLM step**, not a free-roaming agent. It **hands off to booking sites via deep or affiliate links**. It never books or takes payment itself. | New feature. Amadeus Self-Service shut down in Jul 2026. Transacting in India brings PCI scope, IATA and refund liability. |
| D9 | Store a user's location only as an **H3 cell** (res 8, about 0.7 km²). Never store raw GPS for matching. | Makes the "approximate location" promise a property of the data model. |

---

## 1. Architecture

```
React ──REST/WS──► Spring Boot (owns all business writes, auth, chat)
                        │  sync REST (ids only, small payloads)
                        ▼
              ┌──────────────────────────┐        ┌───────────────────────┐
              │ ML API (FastAPI, stateless│◄──────►│ Redis (phase 2+):      │
              │ N replicas behind LB)     │        │ rec cache, trending,   │
              │  /match /recs /clans      │        │ rate limits, job queue │
              │  /verify /moderate        │        └───────────────────────┘
              └───────────┬──────────────┘
                          │ read-only role on app tables,
                          │ read/write on `ml` schema
                          ▼
              PostgreSQL + pgvector + PostGIS (+ read replica later)
                          ▲
              ┌───────────┴──────────────┐
              │ ML workers / batch jobs   │  cron → Prefect later
              │  embed, dedup, trending,  │  MLflow for models
              │  training, trip planner   │
              └──────────────────────────┘
```

**Rules that keep it scalable:**
1. **Spring sends IDs, not data.** For example `POST /v1/match/buddies {user_id, event_id}`. The ML service reads the features it needs from Postgres itself. Payloads stay small, and Spring never has to know which features a model uses.
2. **The ML API is stateless.** Models load at startup and it scales horizontally. Slow work (embedding, training, trip planning) runs in workers, never in the request path.
3. **Only Spring writes business tables.** The ML service writes only to the `ml` schema: embeddings, scores, logs and model outputs.
4. **Every response carries `model_version` and `request_id`**, and Spring logs them with the impression. That link is what makes offline evaluation possible.
5. **Every ML call has a fallback.** If the ML service is down, Spring shows events sorted by date and distance, and buddies sorted by distance plus shared interests. ML should never be a single point of failure.

**ML service API (v1):**

| Endpoint | Sync/async | Used by |
|---|---|---|
| `POST /v1/match/buddies` `{user_id, event_id, limit}` → ranked candidates with `score`, `components`, `explanation`, `request_id` | sync, p95 < 300 ms | "Find a Buddy" |
| `POST /v1/recs/events` `{user_id, lat_lng→h3, date_from, date_to, limit}` | sync (cached later) | Home feed |
| `POST /v1/clans/suggest` `{user_id, event_id?}` and `POST /v1/clans/form` `{event_id}` | sync / batch | Clans |
| `POST /v1/verify/face` `{user_id, video_ref, challenge_id}` → `{decision: pass/review/fail, scores}` | async (seconds) | Verification |
| `POST /v1/moderate/message` `{conversation_id, sender_id, text}` → `{action: allow/nudge/block/flag, labels}` | sync, p95 < 50 ms | Chat |
| `POST /v1/trips` → `job_id`; `GET /v1/trips/{job_id}` | async | Trip planner |
| Internal jobs: `embed_events`, `dedup_events`, `categorize_events`, `trending`, `train_*` | batch | cron |

---

## 2. Data foundations (build first, week 8)

### 2.1 Interaction log: one table, all surfaces

```sql
CREATE SCHEMA ml;

CREATE TABLE ml.interaction (
  id            bigserial,
  ts            timestamptz NOT NULL DEFAULT now(),
  request_id    uuid,              -- links back to the ML response that produced the impression
  user_id       bigint NOT NULL,   -- actor
  surface       text NOT NULL,     -- 'event_feed' | 'buddy_list' | 'clan_suggest' | 'trip' | 'chat'
  item_type     text NOT NULL,     -- 'event' | 'user' | 'clan' | 'trip_item'
  item_id       bigint NOT NULL,
  event_id      bigint,            -- event context, if any
  action        text NOT NULL,     -- see list below
  position      int,               -- rank shown at (for position-bias correction)
  score         real,
  model_version text,
  explore       boolean DEFAULT false,
  context       jsonb,             -- score components snapshot, dwell ms, rating, reason...
  PRIMARY KEY (id, ts)
) PARTITION BY RANGE (ts);        -- monthly partitions
CREATE INDEX ON ml.interaction (user_id, ts);
CREATE INDEX ON ml.interaction (item_type, item_id, action);
```

**`action` values to log from day 1.** The bold ones are the training labels.
- Events: `impression`, `click`, `save`, `dismiss`, **`rsvp`**, **`ticket_click`**
- Buddies: `impression`, `profile_open`, **`request_sent`**, **`request_accepted`**, `request_declined`, `request_expired`, **`chat_started`**, **`duo_formed`**, **`attended_together`**, **`rated`** (1–5 plus "would go again"), `blocked`, `reported`
- Clans: `impression`, `join_request`, `join_accepted`, `left`

Snapshot the score components in `context` **at serve time**. If you recompute features later, training uses data the model didn't have when it ranked, and the offline numbers look better than they really are.

**When to move off Postgres:** only when writes exceed about 1–5k/s or several consumers need the stream. Then move to Redpanda (Kafka API, single binary).

### 2.2 Embeddings and location

```sql
CREATE EXTENSION vector; CREATE EXTENSION postgis; CREATE EXTENSION pg_trgm;

CREATE TABLE ml.event_embedding (event_id bigint PRIMARY KEY, model text NOT NULL,
  emb vector(384) NOT NULL, updated_at timestamptz DEFAULT now());
CREATE TABLE ml.user_embedding  (user_id  bigint PRIMARY KEY, model text NOT NULL,
  emb vector(384) NOT NULL, updated_at timestamptz DEFAULT now());
CREATE INDEX ON ml.event_embedding USING hnsw (emb vector_cosine_ops);
CREATE INDEX ON ml.user_embedding  USING hnsw (emb vector_cosine_ops);

-- privacy: the ML side only ever sees an H3 cell, never raw coordinates
CREATE TABLE ml.user_location (user_id bigint PRIMARY KEY, h3_r8 text NOT NULL,
  updated_at timestamptz DEFAULT now());
```

- **Embedding model:** `intfloat/multilingual-e5-small`. Exported to ONNX int8 through sentence-transformers it runs about 3× faster on CPU. Remember the `query: ` / `passage: ` prefixes. Store the `model` name next to each vector, so switching models is a controlled re-embed.
- **User text** to embed: bio, interests, hobbies, favourite artists and teams, and the onboarding quiz answers rendered as a sentence.
- **Event text** to embed: title, performers, category, venue, city and description.
- **Hinglish check (week 8):** hand-label about 200 Hinglish/English pairs, such as "kal raat ka show mast tha" vs "great show last night". Check that e5-small ranks them sensibly. If it doesn't, try `BAAI/bge-m3`, which is about 4–5× slower but fine for nightly event batches.
- **Distance:** compute it between H3 cell centres. Display buckets such as "< 2 km" or "2–5 km", never exact values. The "1.2 km away" in the report becomes "about 1 km away".

### 2.3 Onboarding quiz (cold start)

Pure genre overlap is a weak signal for "should these two go to a concert together". Timeleft matches strangers for dinners using a values and personality quiz. Add about 10 quick questions at signup:

| Question | Values |
|---|---|
| Vibe | front-row & dance / chill at the back / depends |
| Arrive | together / meet at venue |
| After the show | head home / food / party |
| Budget band | ₹ / ₹₹ / ₹₹₹ |
| Punctuality | early / on time / relaxed |
| Social energy | introvert / ambivert / extrovert |
| Languages | multi-select |
| Group preference | duo / small group / either |
| Buddy filters (**hard filters, user-chosen**) | verified-only, women-only, age band |

Hard filters are the user's explicit choice. We **never learn gender as a ranking feature**. Meetup's ML team recommends the same.

---

## 3. F1: Buddy matching (reciprocal recommendation)

### 3.1 Pipeline

```
candidates : users with EventParticipant(event_id, status in {interested, going, looking_for_buddy})
                 (per-event pools are small: brute-force scoring is fine for a long time)
hard filter: not self, not blocked either way, not already connected/requested,
             both sides' hard filters (verified-only, women-only, age band, language),
             same show slot/date, within max travel distance, trust >= threshold
score      : reciprocal score R(u,v)            (3.2)
re-rank    : x trust, x congestion penalty, exploration slots, per-user daily exposure cap
output     : top K with score components + human-readable reasons
```

### 3.2 Phase 0 scorer (weeks 8–10)

The report's weights stay. What changes is that each term is **directional**:

```
I(u,v)  = 0.6 * cos(emb_u, emb_v) + 0.4 * jaccard(tags_u, tags_v)      # interests (symmetric)
E(u,v)  = mean of quiz agreement on {vibe, arrive, after, budget, punctuality}
          + bonus if same ticket tier/section                             # event-context compat
L(u,v)  = exp(-dist_km(h3_u, h3_v) / 5)                                   # proximity (symmetric)
P(u→v)  = share of u's soft preferences that v satisfies                  # directional

s(u→v)  = 0.40*I + 0.30*E + 0.20*L + 0.10*P(u→v)
R(u,v)  = 2 * s(u→v) * s(v→u) / (s(u→v) + s(v→u))       # harmonic mean punishes one-sided pairs

final   = R(u,v) * trust(v) * 1 / (1 + pending_incoming(v) / 5)
```

- **"Match %" shown in the UI** is `round(100 * R)`. Trust and congestion affect the order only, never the number the user sees.
- **E is redefined.** In the report, "Event Compatibility" meant "same event". Candidates are already filtered to the same event, so that version would score 1 for everyone. E now measures how compatible two people are *at this event*.
- **Explanations** come from the top components, for example "Both love Arijit Singh · both want food after · ~2 km apart". This is cheap and builds trust.
- **Keep weights in a config file**, with the version recorded in `model_version`. Weight tuning then becomes an experiment you can track.
- **Check:** keep a `test_scorer.py` with asserts. A one-sided pair (s = 0.9 / 0.1) must score below a balanced one (0.5 / 0.5). A blocked user must never appear. Scores must be in [0, 1].

### 3.3 Re-ranking: avoid the "superstar" problem

This is the failure mode every dating app has hit. Pure score ranking sends every request to the same few attractive or active users. They can't answer them all, so total matches **fall**. Tinder retired its Elo desirability score for this reason. LinkedIn penalised recipients with many unanswered invites and saw invites to infrequent members rise by 5.44% and connections to them by 4.8%.
- **Congestion penalty** in `final`, as in 3.2.
- **Exposure cap:** a user appears in at most N lists per day, for example 30.
- **Exploration:** 10–20% of slots go to new or low-exposure users (flag `explore=true`), with a new-user boost that decays over the first 7 days.
- **Request rate limits** (enforced in Spring):
  - at most 20 requests per day
  - at most 10 pending outgoing requests, Hinge-style: no new requests until older ones resolve
  - stricter limits for accounts less than 7 days old
- **Optional "Top pick per event"**, as in Hinge's Most Compatible: run Gale-Shapley stable matching on the per-event score matrix. The pools are small, so this is cheap.

### 3.4 Evolution (each step replaces the previous only after it wins offline and in an A/B test)

| Phase | Trigger | Model | Labels / features |
|---|---|---|---|
| 0 | now | the heuristic above | none |
| 1 | about 5–10k requests with outcomes (estimate) | two `LogisticRegression` models: `p = P(send \| impression)`, `q = P(accept \| request)`, calibrated with `CalibratedClassifierCV`. Rank by `p·q`, then the same re-rank. | Heuristic components, embedding cosine, distance bucket, both users' historical send and accept rates, days since last active, pending inbox size |
| 2 | about 50k+ impressions | `lightgbm.LGBMRanker(objective="lambdarank")`, grouped by (viewer, session) | Graded label: 0 impression, 1 request, 2 accepted, 3 chatted, 4 attended together & rated ≥ 4 |
| 3 | pools too large for brute force (big cities, all-events buddy search) | two-tower model with bidirectional objectives (Hinge 2026) for retrieval via pgvector ANN, with the LightGBM model as re-ranker | Same, plus in-batch negatives |

**Metrics:**
- **Offline:** AUC of p and q; NDCG@10 on accepted requests.
- **Online:** accepted requests per 100 impressions; attended-together rate.
- **Guardrails:**
  - Coverage: the share of active users who got at least 1 request this week.
  - Gini coefficient of incoming requests. If it rises, the superstar problem is back.
  - Block and report rate after a match.

---

## 4. F2: Clan formation & clan recommendation

Two jobs:

**(a) Suggest an existing clan to a user:**
- `score(u, C) = min over members m of R(u, m)`. Least misery means a new member can't be a bad fit for anyone.
- Blend in the clan's interest tags: `0.7 * least_misery + 0.3 * cos(emb_u, clan_centroid)`.

**(b) Form new clans for an event** from users who chose "group" or "either". Batch job, 48 h before the event and daily before that:
```
pool = users going to event E wanting a group, after hard filters
M    = pairwise R matrix (O(n²); fine for per-event pools up to a few thousand)
repeat:
  seed = most-constrained unassigned user (fewest candidates above τ)
  group = {seed}
  while |group| < k (k = 4 default, 3–6):
      add the candidate maximising min_{m in group} M[c, m]
      subject to: every pair >= τ (average-without-misery), composition rules
                  (women-only if any member requires it, introvert/extrovert mix, shared language)
  propose the group → each member accepts (clan forms when ≥ 3 accept)
```
- Finding the optimal partition is NP-hard. A greedy algorithm with a misery threshold is the standard practical answer. **Check:** in `test_clans.py`, no formed group may contain a pair below τ, and no user may be in two groups.
- **Feedback:** after the event, each member rates the others. Timeleft does the same. The ratings feed the phase 1+ labels.

---

## 5. F3: Event recommendations (home feed)

**Key constraint:** every event is new, short-lived and in the future. Collaborative filtering on event IDs barely works. **Learn preferences on stable things** (artist, performer, team, genre, venue, organizer, category, price band) and score each new event through those.

### 5.1 Pipeline
```
candidates : upcoming events in user's city / radius and date window (usually ≤ a few thousand)
             ∪ ANN(user taste vector) ∪ followed artists/venues ∪ city trending ∪ buddies-going
filter     : sold out, dismissed, age-restricted, past
score      : phase 0 weighted (below) → phase 2 LightGBM lambdarank
re-rank    : MMR diversity / max 3 per category in top 10, dedup, partner boosts (labelled)
```

**Phase 0 score:**
```
0.30 * cos(user_taste_vec, event_emb)
+ 0.20 * entity_affinity(user, event.performers/venue/category)
+ 0.15 * exp(-dist_km / 10)
+ 0.15 * trending_norm(event)
+ 0.15 * social(event)      # buddies/connections/clan members going — also shown as the reason
+ 0.05 * urgency(days_until) # ramps up as the event nears
```

- **User taste vector:** a time-decayed weighted mean of the embeddings of the user's interest text and of events they saved or RSVP'd to (RSVP counts 3×, save 1×, half-life 60 days). Recompute nightly, and immediately after onboarding.
- **Cold-start event:** for richer signals, borrow the mean behaviour vector of past events with the same performer, venue or category. This is Airbnb's trick for new listings.
- **Trending:** a job runs every 10 minutes per city: `(saves + 3·rsvps + 5·buddy_requests) / (age_hours + 2)^1.5`. It's stored in a table (Phase 0) or a Redis sorted set (Phase 2).
- **Concert affinity without Spotify:** since Nov 2024, new apps have lost Spotify's recommendation and related-artist endpoints. Extended access now requires 250k+ MAU, and development mode is capped at 5 test users. **Don't depend on Spotify.** Use artist and genre picks in onboarding. Check the Last.fm API later **(verify)**.
- **Model upgrade path:**
  - LightFM (hybrid matrix factorisation over metadata) as a strong baseline once we have about 1k users with RSVPs.
  - LightGBM ranker after that.
  - Evaluate on a **time-based split** (train on the past, test on next week's events), never a random one.

**Metrics:**
- **Offline:** Recall@10/20 and NDCG@10 against RSVPs and saves.
- **Catalog coverage:** the share of upcoming events recommended to anyone. This protects small and college events.
- **Intra-list diversity.**
- **Baselines to beat:** "popular in city" and "sorted by date".

---

## 6. F4: Event ingestion ML (dedup + categorization)

The same concert will arrive from two platforms. Recommend the **canonical event**, and deep-link to the listing the user prefers or the cheapest one.

```
event (canonical)  1 ── *  source_listing (source, external_id, url, price, fetched_at)
```

**Dedup job**, run on every ingest batch:
1. **Blocking:** only compare listings with the same city, date within ±1 day, and venues within 300 m (PostGIS `ST_DWithin`).
2. **Pair score:**
   - `0.4·trigram_sim(title)`, using pg_trgm `similarity()`
   - `+ 0.4·cos(title+performer embeddings)`
   - `+ 0.2·(normalised performer match)`
3. **Thresholds:** ≥ 0.90 auto-merge; 0.75–0.90 goes to the admin review queue (this fits the Admin module); below that, keep separate.
4. **Upgrade** to Splink (unsupervised probabilistic record linkage, runs on DuckDB) when the rules produce too many review items.
5. **Check:** a `test_dedup.py` with about 20 hand-made listing pairs, both known duplicates and known non-duplicates.

**Auto-categorization and tags:**
1. **Zero-shot:** cosine between the event embedding and an embedding of each category description ("Stand-up comedy: live comedians performing jokes…"). Free, on CPU.
2. **Low-confidence cases** (top-1 minus top-2 < 0.05) go to an LLM call (`claude-haiku-4-5`) that returns category plus tags (language, vibe, audience, age limit) as JSON. Cache the result per canonical event.
3. **After about 2k labelled events,** train a logistic regression on the embeddings and use the LLM only as a fallback.

---

## 7. F5: Face verification (Verified badge)

### 7.1 Why not the current plan
- OpenCV **Haar cascades only detect** faces.
- **LBPH** is a pre-deep-learning recogniser. It fails under changes in lighting and pose, and nothing stops someone holding up a printed photo.
- **InsightFace `buffalo_l`** is the most accurate open option, but its weights are licensed **for non-commercial research only**.
- **Stay in OpenCV but use its modern models:** `cv2.FaceDetectorYN` (YuNet) + `cv2.FaceRecognizerSF` (SFace), both Apache-2.0 and fast on CPU. This also keeps the report's "OpenCV" claim true.

### 7.2 Pipeline
```
client  : 3-second selfie video; server issues a random challenge (turn left / turn right / blink)
          Android Play Integrity / iOS App Attest where available (anti-injection)
ML job  : 1. YuNet detect per frame → exactly one face, size/blur (Laplacian var)/brightness gate
          2. passive anti-spoofing on sampled frames: MiniFASNet (minivision Silent-Face, Apache-2.0, ONNX)
          3. challenge check: head yaw from YuNet's 5 landmarks must follow the requested motion
          4. best frame → SFace 128-d embedding → cosine vs each profile photo containing a face
          5. dedup: ANN search against all stored face embeddings → match to *another* account ⇒ review
decision: three bands → pass (badge) / review (admin queue) / fail (retry, max 3/day)
cleanup : delete video + frames immediately after decision; keep embedding + scores + model_version
```

- **Thresholds:** OpenCV suggests a starting cosine of about 0.363 for SFace. **Recalibrate** on a consented set of about 200 same/different pairs collected from real users' phones.
  - Set auto-pass at a strict false-match rate, around 1e-4, and send the grey zone to human review. Bumble also uses human reviewers.
- **Bias:** in NIST's demographic tests, false-match rates vary 10–100× across demographic groups. Measure false-match and false-non-match rates per skin-tone bucket on our own set. Reject dark or blurry selfies before matching, because poor lighting hurts darker skin most.
- **Production swap:** AWS Rekognition Face Liveness at about $0.015 per check, plus CompareFaces at about $0.001 per image. That is roughly $1.6k per 100k verifications. It sits behind the same `/v1/verify/face`. Confirm Face Liveness is available in `ap-south-1` Mumbai **(verify)**. Self-built passive liveness is the weakest part of this pipeline, so swap it before any real launch.
- **What not to do:** Aadhaar biometric authentication. Private use requires government approval (2025 amendment rules). An optional "ID verified" tier through DigiLocker or masked documents is a later, lawyer-reviewed decision.

### 7.3 DPDP Act 2023 / DPDP Rules 2025 (most obligations apply from 13 May 2027; design for them now)
- **Consent:** a separate, itemised screen for face verification that says what is stored (an embedding), why (badge, duplicate detection) and for how long. Withdrawing must be as easy as giving it.
- **Retention:** raw media is deleted right after the decision. The embedding is deleted when the account is deleted. Access logs are kept for 1 year.
- **Security:** keep face embeddings in their own table, with a restricted DB role, encrypted disk and access logging. Note the trade-off: the dedup ANN search needs unencrypted vectors in the index, so protection is by access control, not column encryption.
- **Breach:** notify the Data Protection Board within 72 h.
- **Age:** keep the platform **18+ only**. Minors need verifiable parental consent.

---

## 8. F6: Trust & safety ML

| Component | Phase 0 (MVP) | Later |
|---|---|---|
| **Chat moderation** `/v1/moderate/message` | 1) Regex: phone numbers, UPI IDs `\w+@(ok\w+\|ybl\|paytm\|upi)`, payment links, "WhatsApp/Telegram pe aao"-style moves off-platform, "send money for the ticket". 2) Detoxify (English only). 3) Hindi/Hinglish abuse word list. Actions: allow / **"Are you sure?" nudge** / block / flag. Tinder's version cut inappropriate messages by more than 10%. | Fine-tune **MuRIL / IndicBERT / L3Cube HingBERT** on Hinglish abuse datasets (e.g. LCS2-IIITD). Run an asynchronous LLM check (Llama Guard 4 or `claude-haiku-4-5`) on flagged messages only. **Don't use Perspective API, which shuts down 31 Dec 2026.** |
| **Images in chat** | Bumble **Private Detector** (Apache-2.0) blurs lewd images until the recipient taps to view them. | n/a |
| **"Does this bother you?"** | When a message is flagged, ask the recipient. Tinder saw reports of harassing messages rise 46%. | n/a |
| **Report triage** | Priority = severity (threat or sexual harm first) × reporter credibility × reported user's history × **urgency (hours until they meet at an event)**. Auto-suspend at N independent reports. Feeds the admin queue. | A learned triage model trained on moderator decisions |
| **Trust score** | Falls with valid reports, blocks received per impression, and a spam pattern (high send rate, low accept rate). Used as a ranking multiplier and hard filter in section 3. | n/a |
| **Fake / duplicate accounts** | One account per phone number (OTP). Device fingerprint (FingerprintJS v5, which is MIT-licensed again). Face-embedding dedup (section 7). Perceptual hash of profile photos (`imagehash`) against known scam photos. | Link accounts that share a device, phone, IP or face into a graph, and find connected components with SQL or NetworkX. Move to a graph neural network only if organised rings appear. |

**Scam root cause specific to this app:** "send me money for your ticket". Build ticket cost-splitting into the product, or push users to each buy through the source platform, so there is no reason to transfer money directly.

**Meetup safety (product, not ML, but belongs in the same release):**
- Share live location with a trusted contact during the event window.
- Check-in pings at the start and end of the meetup.
- An SOS button that calls 112 and sends location to trusted contacts.
- The **venue as the default meeting point**.
- Phone numbers are never shared by default.
- A post-event question, "Did you feel safe?", which feeds triage.

---

## 9. F7: AI trip planner ("Plan my trip around this show")

> **The finished design is in [TRIP_PLANNER_DESIGN.md](TRIP_PLANNER_DESIGN.md), 30 Sep 2026.** It replaces the phased plan below, and wins wherever the two disagree. This section stays for the reasoning behind it.

**Short answer to "can we use an AI agent for this?":**
- Yes, and it is a strong differentiator.
- Mindtrip has had an Events feature since Nov 2025, but no general planner combines three things BuddySync can offer:
  - the exact gate and end times
  - planning for your duo or clan (votes, locks, cost split)
  - Indian logistics (trains, Tatkal, INR, getting back late)
- The product lesson from all of them is the same:
  - ground every place and price in live data, because invented places and wrong opening hours are the classic failure
  - keep a human confirming every booking
  - don't try to own payments

### 9.1 What it produces
For an event plus origin city, budget band and the duo or clan members:
1. **Getting there:** train, bus and flight options timed to arrive 3–4 h before gates open. Each is a deep link with a "price as of HH:MM" snapshot.
2. **Stay:** 3 hotels within X km of the venue, grouped by budget band. Check-in and check-out are aligned to the show.
3. **Show day:** what to do before gates open, gate time, the show, and **getting back late** (after 23:00 prefer hotels near the venue or a cab, and flag safety).
4. **Around the stay:** a "must visit" shortlist (3–5), optional extra days, food near the venue, and the weather.
5. **For clans:**
   - Each member's origin gets its own travel plan; the stay is shared.
   - 2–3 variants (budget / balanced / premium) that members vote on in clan chat.
   - Per-person cost split.

### 9.2 Design: a workflow with one LLM step, not a free agent
```
1 skeleton (code)   : from event record → time slots: arrive-by, check-in 14:00, pre-show window,
                      gates, show, late return, check-out 11:00, return window
2 retrieval (code, parallel, cached):
                      hotels_near(venue, dates, budget)   transport(origin→city, date)
                      pois(city, near venue, categories)   travel_time_matrix(points)
                      weather(city, dates)                 opening_hours(place_ids)
3 plan (LLM)        : fill the skeleton, choosing ONLY from the candidate IDs returned in step 2;
                      structured JSON output (strict schema)
4 validate (code)   : every id ∈ retrieved set · opening hours cover slot · travel time feasible
                      (OSRM matrix) · total ≤ budget · arrival ≤ gate time − buffer
                      → on failure, send errors back to the LLM once to repair, else drop the item
5 cache             : store plan per (event_id, origin_city, budget_band, group_size) in Postgres JSONB;
                      prices refreshed on view (flights 15–30 min TTL, hotels 1–6 h)
```
- **Why not an open ReAct-style agent:**
  - The time constraints around a show are rigid, and code handles them reliably.
  - Letting the model only choose from retrieved IDs makes invented places impossible by construction.
  - Cost and latency stay predictable.
  - Expedia moved to specialised agents plus strict evals for the same reasons.
- **Implementation:**
  - A worker in the ML service, driven by a queue: Spring submits a job and gets results over SSE or push.
  - Anthropic Python SDK. Planner model `claude-opus-5` by default. Evaluate `claude-sonnet-5` on the eval set (9.5) as the cheaper option; that choice is the team's call.
  - `claude-haiku-4-5` for summarising POI descriptions and Wikivoyage text in batch.
  - Keep the tool definitions and system prompt as a fixed prefix so prompt caching applies.
  - Pre-generate plans for popular events overnight with the Batch API (50% off).
  - Frameworks like LangGraph are unnecessary at this size.
- **Output item schema:** `{slot_start, slot_end, kind, poi_id|hotel_id|route_id, title, why, source, fetched_at, price_snapshot, deeplink}`.

### 9.3 Data sources (status as researched Sep 2026)

| Need | MVP choice | Notes |
|---|---|---|
| Hotels | **LiteAPI** (sign up yourself, free sandbox, 2–3M hotels, official MCP server); or the Agoda affiliate feed (Travelpayouts' Hotellook closed 20 Oct 2025) | Booking.com's Demand API needs a managed partner contract, and its terms **forbid AI use without written approval**. Expedia Rapid needs an application. |
| Flights | **Travelpayouts** affiliate links. The Data API prices are cached; the real-time Search API needs 50k+ MAU. Skyscanner referral links | **Amadeus Self-Service shut down 17 Jul 2026.** Duffel (sign up yourself, $3 + 1% per order) for live search in v1.5, still handing off. Skyscanner's API requires 100k+ monthly users. Kiwi is invite-only. |
| Trains | Static timetable, deep-linking to IRCTC, ixigo or ConfirmTkt | IRCTC has no public API. Avoid grey-market RapidAPI wrappers. |
| Buses | redBus deep links or affiliate | SeatSeller API is for agents only |
| Places | **Our own POI catalog** for the top 15 event cities, built from OpenStreetMap/Overpass, Wikidata, OpenTripMap (free) and Wikivoyage "See/Eat" sections. Refresh weekly. | Call **Google Places (New)** only to confirm the final 10–15 places: `place_id` lookups are free, opening hours are a paid tier **(verify price)**. Foursquare: 500 free calls then $15 per 1k. |
| Travel time | **OSRM** self-hosted on an India OSM extract (no per-call cost) | Google Routes: 10k free/month, then about $5 per 1k |
| Weather | Open-Meteo | The free tier is non-commercial. The commercial plan is $29/month. |

### 9.4 Booking policy
- **Hand off, don't transact:** deep links and affiliate links, with the user confirming on the provider's site.
- **Why not book ourselves:** we would be the merchant of record. That brings:
  - PCI scope
  - refunds and chargebacks
  - servicing schedule changes
  - IATA's bank-guarantee requirements
  - partners expecting Ministry of Tourism recognition
  - GST handling
- **Revenue:** affiliate commissions, with hotels the largest. LiteAPI lets us set our own markup. Later, possibly a "BuddySync Plus" tier for clan planning.
- **v2 option:** book hotels only, through LiteAPI with a payment provider (Razorpay or Stripe tokenisation keeps PCI at SAQ-A). Only once there's traffic, and only if we're ready to own customer support.

### 9.5 Cost & quality
**Per uncached plan** (estimate; measure with `usage` logging):
- About 50k input tokens, of which about 40k is cached prefix, plus about 6k output.
  - `claude-opus-5`: $5/$25 per MTok, cache reads about 0.1× → about **$0.22**.
  - `claude-sonnet-5`: $2/$10 → about **$0.09**.
- With our own POI catalog and OSRM, the map/API cost is near zero. Careless Google usage would add $0.10–0.60.
- With a ≥ 70% cache hit rate per (event, origin), that's about **₹2–6 per user plan** at about ₹85/USD.

**Evals:** 30 real past events × 3 origin cities. Hard targets:
- **0 invented places** (enforced by the validator)
- **0 infeasible time slots**
- budget respected
- arrival before gates open

Log every failure the validator catches, because the eval set grows from real failures.

### 9.6 Phases
Replaced. No MVP cut: everything ships at launch, and the build order is in [TRIP_PLANNER_DESIGN.md §19](TRIP_PLANNER_DESIGN.md#19-build-order-everything-ships-at-launch-this-is-only-the-order).

---

## 10. Scaling roadmap

| Stage | Infra | ML |
|---|---|---|
| **MVP (≤ 1k users)** | Postgres (pgvector, PostGIS, pg_trgm) + 1 FastAPI container + cron | Heuristic rankers, e5-small ONNX, rule-based dedup, OpenCV verification, regex+Detoxify moderation |
| **~10k users** | + Redis (rec cache with 1–6 h TTL, trending sorted sets, rate limits, job queue); Postgres read replica for ML reads; MLflow (Postgres backend); Prefect replaces cron once there are more than 5 dependent jobs; 2+ ML API replicas | LR p·q buddy model, LightFM/LightGBM event ranker, A/B tests by hashed user ID, Splink, fine-tuned Hinglish moderation, AWS Liveness |
| **~1M users** | Redpanda + a stream job for real-time trending and session features; consider Qdrant only past about 10M vectors or if pgvector filtering struggles; Feast when several models share real-time features; GPU only if a model needs it | Two-tower retrieval + LightGBM re-rank, session vectors, interleaving experiments, graph-based fraud detection |

**Rule:** move up a stage only when a metric (p95 latency, DB CPU, write rate, cache miss cost) says so. Record the trigger in the PR that adds the component.

**Operations from day 1, cheap:**
- A `/health` endpoint.
- Structured logs with `request_id`.
- Timing per endpoint.
- A nightly job that prints last week's key metrics (acceptance rate, coverage, Gini, verification pass/review/fail split).

---

## 11. Timeline (remaining weeks of the semester, then post-semester)

| Week | ML work | Demo-able outcome |
|---|---|---|
| **8** | `ml` schema, interaction log, H3 location, embedding job (e5-small ONNX), onboarding quiz fields; FastAPI skeleton with `/health`, fallbacks, `request_id` | Events and users embedded; logs flowing |
| **9** | F1 Phase 0 buddy scorer + re-ranking + explanations + `test_scorer.py`; Spring integration | "Find a Buddy" shows ranked, explained matches |
| **10** | F3 Phase 0 event feed + trending; F4 dedup + zero-shot categorization | Personalized home feed; duplicate events merged |
| **11** | F5 face verification (YuNet/SFace + MiniFASNet + challenge + 3 bands + dedup) + consent screen; threshold calibration on a consented set | Verified badge end to end |
| **12** | F6 moderation endpoint (regex + Detoxify + nudge), report triage score, trust multiplier; F2 clan suggestion + greedy clan formation | Safe chat; auto-suggested clans |
| **13** | Offline evaluation report: synthetic plus pilot data, baselines vs Phase 0, metrics from sections 3–5; **stretch:** trip planner MVP for 1 city | Evaluation chapter for the final report |
| **14** | Hardening, load test (e.g. locust against `/match` at 50 RPS), documentation, demo | Final demo |
| **Post-semester** | Trip planner MVP → v1.5; LR p·q buddy model once logs suffice; LightGBM; AWS Liveness; Hinglish moderation model | n/a |

**Data for the demo:**
- Seed about 500 synthetic users with realistic interest distributions and about 200 real-looking events.
- Run a small pilot, for example with classmates for campus events, to get real impressions and requests.
- Synthetic data is fine for testing the pipeline. Only real outcomes validate the model. **State this clearly in the report.**

---

## 12. Risks & open questions

| Risk | Mitigation |
|---|---|
| Too little interaction data to train anything this semester | Phase 0 heuristics are designed to ship without training data; the pilot generates the first real labels |
| Event API access (BookMyShow and District have no public API) | Out of ML scope, but dedup/categorization must work with any source; user-created events plus permitted feeds for the demo |
| Popularity concentration in matching | Congestion penalty, exposure caps, exploration, Gini guardrail (section 3.3) |
| Face-verification false matches, or bias against darker skin tones | Three bands with human review, per-group error measurement, quality gate, managed liveness before real launch |
| Hinglish text poorly handled by embeddings or moderation | Week 8 Hinglish test; bge-m3 fallback; fine-tuned HingBERT later |
| Trip planner hallucinations and stale prices | Choosing only from retrieved IDs, a code validator, "price as of" labels, handing off to book |
| Legal (DPDP, Aadhaar, Booking.com's AI clause) | Section 7.3; no Aadhaar biometrics; read partner terms before integrating |

**Still unverified (check before relying on them):**
- AWS Face Liveness availability in Mumbai.
- Google Places Pro/Enterprise prices.
- Whether Travelpayouts data is live.
- Affiliate commission rates.
- Last.fm API terms.
- The Phase 1 data-volume threshold.
- The token-cost estimate (measure it).

---

## Sources

**Matching / recsys**
- Hinge Most Compatible (Gale-Shapley): https://techcrunch.com/2018/07/11/hinge-employs-new-algorithm-to-find-your-most-compatible-match-for-you/
- Hinge reciprocal two-tower model (2026): https://dl.acm.org/doi/10.1145/3774935.3807906
- Hinge Your Turn Limits: https://hinge.co/newsroom/your-turn-limits-test
- Tinder matching method: https://www.tinderpressroom.com/powering-tinder-r-the-method-behind-our-matching
- LinkedIn PYMK: https://www.linkedin.com/blog/engineering/recommendations/building-a-large-scale-recommendation-system-people-you-may-know
- LinkedIn PYMK equity: https://www.linkedin.com/blog/engineering/member-customer-experience/optimizing-pymk-for-equity-in-network-creation
- Timeleft matching: https://timeleft.com/post/timeleft-algorithm-the-maestro-of-your-dinners/
- RECON reciprocal recommender: https://dl.acm.org/doi/10.1145/1864708.1864747
- Congestion-aware reciprocal recommendation: https://arxiv.org/html/2409.00720
- Group recommendation strategies (Masthoff): https://pro.unibz.it/projects/schoolrecsys17/JudithMasthoff.pdf
- LightGBM LGBMRanker: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html
- Two-tower sampling-bias correction: https://dl.acm.org/doi/10.1145/3298689.3346996

**Event recommendations / infrastructure**
- Meetup event recommendation (Macedo et al.): https://dl.acm.org/doi/10.1145/2792838.2800187
- Airbnb embeddings: https://dl.acm.org/doi/10.1145/3219819.3219885
- LightFM: https://github.com/lyst/lightfm
- Eugene Yan, discovery system design: https://eugeneyan.com/writing/system-design-for-discovery/
- pgvector 0.8: https://aws.amazon.com/blogs/database/supercharging-vector-search-performance-and-relevance-with-pgvector-0-8-0-on-amazon-aurora-postgresql
- H3: https://h3geo.org/
- Splink: https://moj-analytical-services.github.io/splink/
- multilingual-e5-small: https://huggingface.co/intfloat/multilingual-e5-small
- sentence-transformers ONNX export: https://sbert.net/docs/sentence_transformer/usage/efficiency.html
- Spotify API changes: https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api
- Spotify extended-access criteria: https://developer.spotify.com/blog/2025-04-15-updating-the-criteria-for-web-api-extended-access
- Netflix interleaving: https://netflixtechblog.com/using-interleaving-in-online-experiments-to-accelerate-algorithm-innovation-at-netflix-a04ee392ec55

**Trust & safety**
- OpenCV SFace licence: https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/LICENSE
- InsightFace model zoo (licence): https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md
- Silent-Face-Anti-Spoofing: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
- NIST demographic effects: https://pages.nist.gov/frvt/reports/demographics/nistir_8429.pdf
- AWS Rekognition pricing: https://aws.amazon.com/rekognition/pricing/
- Tinder liveness check: https://policies.tinder.com/faq-mandatory-liveness-check/intl/en/
- DPDP Rules 2025: https://static.pib.gov.in/WriteReadData/specificdocs/documents/2025/nov/doc20251117695301.pdf
- Aadhaar amendment rules (PIB): https://www.pib.gov.in/PressReleasePage.aspx?PRID=2098223
- Bumble Private Detector: https://github.com/bumble-tech/private-detector
- Tinder "Are You Sure?": https://www.tinderpressroom.com/2021-05-20-Tinder-Introduces-Are-You-Sure-,-an-Industry-First-Feature-That-is-Stopping-Harassment-Before-It-Starts
- Hinglish offense detection: https://github.com/LCS2-IIITD/Hinglish_offense_detection-Neurocomputing2021
- Perspective API shutdown: https://medium.com/tisanelabs/goodbye-perspective-api-79da0f237b3f

**Trip planner**
- Google agentic travel booking: https://blog.google/products-and-platforms/products/search/agentic-plans-booking-travel-canvas-ai-mode/
- Expedia acquires Layla: https://skift.com/2026/07/31/expedia-acquired-ai-trip-planner-layla-exclusive/
- MakeMyTrip Myra 2.0: https://www.traveltrendstoday.in/makemytrip-launches-myra-20-an-end-to-end-conversational-booking-assistant
- Amadeus Self-Service shutdown: https://www.phocuswire.com/amadeus-shut-down-self-service-apis-portal-developers
- Duffel pricing: https://duffel.com/pricing
- LiteAPI: https://liteapi.travel/
- Skyscanner Travel API: https://www.partners.skyscanner.net/product/travel-api
- Booking.com Demand API terms: https://vorplabs.com/agent-tools/booking-demand-api
- Google Places billing: https://developers.google.com/maps/documentation/places/web-service/usage-and-billing
- Foursquare changes: https://docs.foursquare.com/developer/reference/upcoming-changes
- Open-Meteo pricing: https://open-meteo.com/en/pricing
- IATA India requirements: https://www.iata.org/en/services/travel-agency-program/accreditation-travel/country-requirements/a52w0000000fy3haaq/pax/
