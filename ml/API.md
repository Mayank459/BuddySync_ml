# BuddySync ML Service: API Reference

Contract between the **Spring Boot backend** and the **Python ML service** (`ml/`).
The React app never calls this service directly. Spring calls it, then decides what to show and store.

- **Base URL:** `http://<host>:<port>/v1`, for example `http://127.0.0.1:8765/v1` locally. On Windows use `127.0.0.1`, not `localhost`, which adds about 2 s per request.
- **Format:** JSON in and out, except face verification (`multipart/form-data`).
- **Auth:** set `ML_SERVICE_TOKEN` on the service, and Spring sends it as the header `X-Service-Token` on every `/v1` call (otherwise `401`). `/v1/go` is the exception: users' browsers open it directly, and it only redirects to known booking sites. Keep the service on the backend's private network anyway.
- **IDs:** `user_id` and `event_id` are the backend's integer IDs. For now the service reads them from the synthetic CSVs in `ml/data/`; later it will read them from Postgres.
- **Interactive docs:** `GET /docs` (Swagger) lists every request schema.

## Contents

| # | Method & path | Purpose | Typical p95 latency |
|---|---|---|---|
| 1 | `POST /v1/match/buddies` | Ranked buddy suggestions for one user at one event | 26 ms |
| 2 | `POST /v1/clans/form` | Propose clans among people at an event | 178 ms (45 people); batch job |
| 3 | `POST /v1/recs/events` | Personal event feed | 11 ms |
| 4 | `POST /v1/events/ingest` | Categorise a new listing and detect duplicates | 4 ms |
| 5a | `POST /v1/verify/challenge` | Get random liveness actions | < 5 ms |
| 5b | `POST /v1/verify/liveness` | Challenge video → liveness + same-person + face match | ~5 s for a 10 s video |
| 5c | `POST /v1/verify/face` | Photo match only (no liveness) | ~100–300 ms (not benchmarked) |
| 6 | `POST /v1/moderate/message` | Check a chat message before delivery | 2 ms (~70 ms with `USE_LAYA=1`) |
| 7.1 | `POST /v1/trips` | Create a trip (queues the draft) | < 100 ms; draft 1–20 s async |
| 7.2 | `GET /v1/trips/{trip_id}` | Read the trip | < 50 ms |
| 7.3 | `GET /v1/trips/{trip_id}/versions/{v}` | One past version | < 50 ms |
| 7.4 | `POST /v1/trips/{trip_id}/messages` | Chat with the planner (queues a turn) | < 50 ms; turn 5 s – 3 min async (free tiers are slow) |
| 7.5 | `GET /v1/trips/{trip_id}/messages` | Chat history | < 50 ms |
| 7.6 | `POST /v1/trips/{trip_id}/voice` | Speech → text | 1–5 s |
| 7.7 | `GET /v1/trips/{trip_id}/stream` | Progress events (SSE) | streaming |
| 7.8–7.11 | `POST/PATCH/DELETE …/items…`, `POST …/undo` | Card actions | < 300 ms (more if a hotel rate is fetched) |
| 7.12 | `GET /v1/trips/{trip_id}/costs` | Cost split | < 50 ms |
| 7.13 | `GET /v1/trips/{trip_id}/timeline-events` | Reminders for Spring | < 50 ms |
| 7.14 | `GET /v1/trips/users/{user_id}/trips` | A user's trips | < 50 ms |
| 7.15–7.17 | `POST/PATCH/DELETE …/members…` | Clan members | < 1 s (fares for new legs) |
| 7.18 | `POST …/items/{item_id}/vote` | Thumbs up or down | < 50 ms |
| 7.19–7.21 | `POST/GET …/decisions…` | Group decisions | < 100 ms |
| 7.22 | `DELETE /v1/trips/users/{user_id}` | Erase a user's trip data | < 100 ms |
| 7.23 | `GET /v1/go` | Booking hand-off (click tracking, 302) | < 50 ms |
| 8 | `GET /health` | Liveness probe | < 5 ms |

---

## Conventions (apply to every endpoint)

### Errors
Every error has the same body:
```json
{ "detail": "human-readable message" }
```

| Status | Meaning | What the backend should do |
|---|---|---|
| `400` | Bad input the service understood (e.g. uploaded file is not an image) | Show the message to the user |
| `403` | The user isn't a member of that trip | Don't show the trip |
| `404` | Unknown `user_id`, `event_id` or `trip_id` | Fix the call; don't retry |
| `413` | Uploaded image larger than 8 MB | Ask the user for a smaller image |
| `422` | Request doesn't match the schema (missing field, wrong type). FastAPI adds a list of field errors to `detail`. | Bug in the caller; don't retry |
| `409` | Trip card action on a stale version, while the planner is editing, or on a locked slot | Re-fetch the trip, then retry |
| `429` | Limit reached (trip message caps) | Show the message |
| `503` | Feature not ready: model not trained, face thresholds not calibrated, city has no places catalog, or trip storage not configured | Use the fallback (below) and alert the team |

### Fallbacks
The ML service must never be a single point of failure. If a call fails or times out (suggested timeout: **1 s** for sync endpoints; trip drafts and chat are async, so their POSTs are fast too):

| Endpoint | Fallback in Spring |
|---|---|
| buddies | Users at the event, sorted by distance, then by number of shared interests |
| recs | Upcoming events in the user's city, sorted by date |
| ingest | Save the listing as a new event with no category; queue for admin review |
| moderate | **Deliver the message** and log it for later review. Don't block chat because ML is down. |
| verify | Keep the user unverified and let them retry later |
| clans | Hide the feature for now |
| trips | The trip still works without an LLM: drafts fall back to a code-only plan, and chat shows the `error` event. Hide the planner only if the trip endpoints themselves fail. |

### Logging (important)
Responses from **buddies** and **recs** include `request_id` and `model_version`. Store both with every impression and every follow-up action: request sent, accepted, RSVP and so on. Without them, the models can never be evaluated or retrained (see ML_IMPLEMENTATION_PLAN.md §2.1).

### Types
- `int`: integer ID.
- `float`: number.
- `datetime`: ISO 8601 without a timezone, e.g. `"2026-10-10T19:00:00"`, interpreted as IST.
- Coordinates: decimal degrees (WGS84).

---

## 1. `POST /v1/match/buddies`

Ranks the people attending the same event as buddy candidates for one user.

### Request
```json
{ "user_id": 1024, "event_id": 214, "limit": 10 }
```

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `user_id` | int | ✓ | | The viewer |
| `event_id` | int | ✓ | | The event they're finding a buddy for |
| `limit` | int | | `10` | Max candidates returned |

### Response `200`
```json
{
  "request_id": "5f0c3c1e-8a55-4b8e-9d2e-2b1f6a0c9e41",
  "model_version": "buddy-phase0-tuned",
  "candidates": [
    {
      "user_id": 655,
      "match_pct": 50,
      "distance": "2-5 km",
      "reasons": ["2-5 km apart", "similar plans for the night"],
      "components": {
        "interest": 0.069, "event_fit": 0.678, "location": 0.83,
        "pref_uv": 0.872, "pref_vu": 0.153
      }
    }
  ]
}
```

| Field | Type | Notes |
|---|---|---|
| `request_id` | string (UUID) | Log with the impression |
| `model_version` | string | Log with the impression |
| `candidates` | array | In display order. May be empty `[]`. |
| `candidates[].user_id` | int | Candidate |
| `candidates[].match_pct` | int 0–100 | **Show this to the user.** Reciprocal compatibility (both directions). |
| `candidates[].distance` | string | Bucket only: `"< 2 km"`, `"2-5 km"`, `"5-10 km"`, `"10+ km"`. Never exact. |
| `candidates[].reasons` | string[] | Up to 2 short explanations to display |
| `candidates[].components` | object of floats 0–1 | For debugging/analytics only; **don't show**. `pref_uv` = candidate fits viewer's preferences, `pref_vu` = the reverse. |

### Behaviour
- **Who can appear:** other users with an EventParticipant row for this event whom the viewer hasn't already sent a request to.
- **Hard filters, checked both ways:** women-only, verified-only, age range. If either side fails the other's filter, the pair is excluded.
- **Order isn't strictly by `match_pct`.** Users with many unanswered incoming requests are ranked lower on purpose, to spread requests beyond the few most popular users.
- **What the score uses:** interests and bio text, onboarding-quiz answers, home-area distance, and soft preferences (language, group size, verified, age). It does **not** use gender (except the user's own women-only filter), photos or looks.

### Errors
- `404` unknown `user_id`
- `422` bad body
- `503` model not trained
- An unknown `event_id` returns `200` with `candidates: []`.

---

## 2. `POST /v1/clans/form`

Splits people at an event who want a group into clans where **every pair** is compatible. Run it as a **scheduled batch job** (e.g. daily, and 48 h before the event), then send each member a "join this clan?" proposal. Don't call it on page load.

### Request
```json
{ "event_id": 214, "size": 4, "min_score": 0.4 }
```

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `event_id` | int | ✓ | | |
| `size` | int | | `4` | Max members per clan (3–6 is sensible) |
| `min_score` | float 0–1 | | `0.4` | Every pair in a clan must score at least this |

### Response `200`
```json
{
  "event_id": 214,
  "clans": [
    { "members": [1332, 1243, 1299], "worst_pair_pct": 43 },
    { "members": [730, 1190, 342, 88], "worst_pair_pct": 41 }
  ]
}
```

| Field | Type | Notes |
|---|---|---|
| `clans` | array | May be empty |
| `clans[].members` | int[] | 3 to `size` user IDs. A user appears in at most one clan. |
| `clans[].worst_pair_pct` | int 0–100 | Score of the least compatible pair in the clan |

### Behaviour
- **Pool:** users at the event with status `looking_for_buddy` whose group preference isn't `duo`.
- **Hard filters:** a pair that fails either side's filters scores 0, so they never end up in the same clan.
- **Too few people:** fewer than 3 eligible gives `clans: []`. People who can't be placed are simply left out of every clan.

### Errors
- `422` bad body
- `503` buddy model not trained (clans reuse the buddy score)

---

## 3. `POST /v1/recs/events`

Personal event feed for the home screen.

### Request
```json
{ "user_id": 1024, "limit": 10, "days_ahead": 60, "max_per_category": 3 }
```

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `user_id` | int | ✓ | | |
| `limit` | int | | `10` | Max events returned |
| `days_ahead` | int | | `60` | Only events starting within this many days |
| `max_per_category` | int | | `3` | Diversity cap per category |

### Response `200`
```json
{
  "request_id": "a3c9e0b2-1f44-4d0b-8e57-3c2d9f1b7a10",
  "model_version": "event-recs-phase0-tuned",
  "events": [
    {
      "event_id": 548,
      "title": "Static Tides Express Live in Mumbai",
      "start_time": "2026-11-08T18:00:00",
      "score": 0.351,
      "reasons": ["matches your taste for hip hop", "0.4 km away"]
    }
  ]
}
```

| Field | Type | Notes |
|---|---|---|
| `request_id`, `model_version` | string | Log with the impression |
| `events` | array | In display order. May be empty. |
| `events[].event_id` | int | |
| `events[].title` | string | |
| `events[].start_time` | datetime | |
| `events[].score` | float | Relative ranking score. **Don't show**; not a percentage. |
| `events[].reasons` | string[] | 0–3 of: taste match, "popular in {city}", "{x} km away" (under 5 km only) |

### Behaviour
- **Filtering:** only events in the **user's city** that start **between now and `days_ahead`**, excluding events the user already has a participant row for.
- **Taste:** built from the user's interests plus events they RSVP'd to or saved, with RSVPs weighted more.

### Errors
- `404` unknown `user_id`
- `422` bad body
- `503` model not trained

---

## 4. `POST /v1/events/ingest`

Call this for **every listing** arriving from an external source or created by a user, **before** saving it as a new event. It returns a category and whether the listing duplicates an existing event.

### Request
```json
{
  "title": "KAVYA TRIO LIVE @ Mumbai | Tickets",
  "performers": "Kavya Trio",
  "city": "Mumbai",
  "venue_lat": 19.0701,
  "venue_lng": 72.8702,
  "start_time": "2026-10-10T19:00:00",
  "description": "An evening of indie music"
}
```

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `title` | string | ✓ | | As given by the source; formatting noise is fine |
| `performers` | string | ✓ | | Comma-separated if several; `""` if unknown |
| `city` | string | ✓ | | Must match the city names used in events |
| `venue_lat`, `venue_lng` | float | ✓ | | Venue coordinates |
| `start_time` | datetime | ✓ | | |
| `description` | string | | `""` | |

### Response `200`
```json
{
  "category": "concert",
  "category_confidence": 0.56,
  "category_needs_review": false,
  "duplicate_of": null,
  "duplicate_probability": 0.0,
  "action": "new"
}
```

| Field | Type | Notes |
|---|---|---|
| `category` | string | One of `concert`, `comedy`, `sports`, `movie`, `theatre`, `workshop`, `festival`, `conference` |
| `category_confidence` | float 0–1 | |
| `category_needs_review` | bool | `true` when confidence < 0.5. Send to the admin queue to confirm the category. |
| `duplicate_of` | int or null | Existing `event_id` this listing most likely duplicates |
| `duplicate_probability` | float 0–1 | |
| `action` | string | What the backend should do (below) |

| `action` | When | Backend should |
|---|---|---|
| `merge` | `duplicate_probability ≥ 0.9` | Attach the listing to event `duplicate_of` as another source. Don't create a new event. |
| `review` | 0.5 – 0.9 | Save it and queue "is this the same as event X?" for an admin |
| `new` | < 0.5, or no nearby event | Create a new event |

### Behaviour
- **Duplicate candidates:** only existing events in the same city, within 300 m and 24 h of this listing, are compared.

### Errors
- `422` bad body
- `503` model not trained

---

## 5. Face verification

**Use 5a + 5b for the Verified badge.** 5c (photo match only) remains for quick checks, but a printed photo can pass it.

Flow:
1. The app calls **5a** and shows the steps.
2. The app records **one live video** of the user doing them. Record it in the app; never accept a gallery upload.
3. The app sends the video plus the profile photos to **5b**.
4. Grant the badge only on `decision: "pass"`.

### 5a. `POST /v1/verify/challenge`: get random actions

**Request:** no body.

**Response `200`:**
```json
{
  "challenge_id": "_SDxcDdSoZsn16uh5Cn2Gg",
  "expires_in_s": 300,
  "steps": [
    { "action": "touch_nose", "prompt": "Touch the tip of your nose with your index finger" },
    { "action": "blink",      "prompt": "Blink a couple of times" },
    { "action": "fingers_1",  "prompt": "Hold up 1 finger next to your face (don't count your thumb)" }
  ],
  "instructions": "Record one video of up to 20 s: start facing the camera, do the steps in this order holding each for about a second, then face the camera again."
}
```

| Field | Type | Notes |
|---|---|---|
| `challenge_id` | string | Send with the video. **Single use**, even if that attempt fails. |
| `expires_in_s` | int | Challenge lifetime in seconds (300 now; ~90 once the app records live) |
| `steps` | array of 3 | Do these **in this order**. Show `prompt` to the user. |
| `steps[].action` | string | `blink`, `open_mouth`, `turn_head`, `touch_nose`, `fingers_1` … `fingers_4` |
| `instructions` | string | Show before recording |

**How steps are picked:**
- **Randomness:** chosen with a cryptographic random generator. There are always 3 steps: at least one facial expression (`blink` / `open_mouth`), at least one hand action (`touch_nose` / `fingers_N`), and at most one finger count.
- **Why:** a photo can't change expression, a pre-recorded video can't know the random steps, and a hand in front of the face tends to break face-swap deepfakes.
- **Head turns:** `turn_head` accepts either side, because front cameras often mirror the image.

### 5b. `POST /v1/verify/liveness`: check the video and match the face

#### Request: `multipart/form-data`

| Part | Type | Required | Notes |
|---|---|---|---|
| `challenge_id` | text | ✓ | From 5a |
| `video` | video file | ✓ | mp4 / mov / webm / avi / mkv. **At most 20 s and 40 MB.** One person, face visible, decent light. |
| `profile_photos` | image file(s) | ✓ | One or more; repeat the field. 8 MB each. |

- **Storage:** the video is written to a temp file (OpenCV needs a file path) and **deleted as soon as it's analysed**. Only the result is kept. Photos are processed in memory only.

```bash
curl -X POST http://127.0.0.1:8765/v1/verify/liveness \
  -F "challenge_id=_SDxcDdSoZsn16uh5Cn2Gg" \
  -F "video=@challenge.mp4" \
  -F "profile_photos=@photo1.jpg"
```

#### Response `200`
```json
{
  "decision": "pass",
  "liveness": {
    "passed": true,
    "steps": [
      { "action": "touch_nose", "done": true, "at_s": 2.1, "identity_checked": true },
      { "action": "blink",      "done": true, "at_s": 4.6, "identity_checked": true },
      { "action": "fingers_1",  "done": true, "at_s": 6.9, "identity_checked": true }
    ]
  },
  "same_person_throughout": true,
  "min_continuity": 0.71,
  "identity_coverage": 0.86,
  "frames_analyzed": 92,
  "duration_s": 9.1,
  "best_similarity": 0.64,
  "per_photo": [0.64],
  "model_version": "yunet2023mar-sface2021dec+mp-landmarker-1.0.1",
  "liveness_checked": true
}
```
*(Illustrative values: no real recording has been tested yet. See Limitations.)*

| Field | Type | Notes |
|---|---|---|
| `decision` | string | `pass` / `review` / `fail` / `retry` (tables below) |
| `reason` | string | Present on `fail` and `retry` |
| `liveness.passed` | bool | Every step done, in order, with the face identity-checked at that moment |
| `liveness.steps[].done` | bool | Action seen, held about 0.3 s (a blink only needs one frame, but open eyes must be seen first), after the previous step |
| `liveness.steps[].at_s` | float or null | Seconds into the video when the step was completed |
| `liveness.steps[].identity_checked` | bool | The face was confirmed within 1 s of this step |
| `same_person_throughout` | bool or null | Every sampled frame, at any pose, matches the clearest frontal frame. `null` if not evaluated. |
| `min_continuity` | float or null | Lowest frame-to-reference similarity |
| `identity_coverage` | float 0–1 | Share of sampled face frames that could be identity-checked. Blurry frames and frames with a hand over the face count against it. |
| `frames_analyzed` | int | Frames checked (sampled at up to about 10 per second, depending on the video's frame rate) |
| `duration_s` | float | Length of the video |
| `best_similarity`, `per_photo` | float, list | Clearest frontal video frame vs each profile photo (only when a match was attempted) |
| `liveness_checked` | bool | Always `true` here |

#### How `decision` is reached (checks run in this order)
| Check | Result | Tell the user / backend action |
|---|---|---|
| More than 20% of frames have 2+ faces | `fail` · `multiple_faces` | "Only you should be in the video" |
| No clear frontal frame, or `identity_coverage` < 0.6 | `retry` · `no_clear_frame` | "Too dark, too blurry, or your face was covered; try again" |
| `min_continuity` < 0.358 | `fail` · `face_changed` | Possible impersonation. Log it and count toward the trust score. |
| A step is missing, out of order, or not identity-checked | `retry` · `challenge_not_completed` | Show which `steps[].done` were false, then get a **new** challenge |
| No usable face in the profile photos | `retry` · `no_face_in_profile_photos` | "Add a clear profile photo of your face" |
| Face match ≥ 0.368 | `pass` | **Grant the Verified badge.** Store `model_version`. |
| Face match 0.358 – 0.368 | `review` | Queue for a human moderator |
| Face match < 0.358 | `fail` | "Doesn't match your profile photos" |

**Retries:** every retry needs a new challenge (5a). Limit attempts in Spring, e.g. 3 per day.

#### Errors
| Status | When |
|---|---|
| `400` | Video can't be decoded, or a photo isn't an image |
| `404` | Unknown or already-used `challenge_id` |
| `410` | Challenge expired |
| `413` | Video over 40 MB or 20 s, or an image over 8 MB |
| `422` | Missing a form field |
| `503` | Face thresholds not calibrated |

**Speed:** about 20 frames/s analysed on a laptop CPU, so a 10 s video takes about 5–8 s. Call it with a **30 s timeout** and show a spinner. One video is processed at a time per server process.

### 5c. `POST /v1/verify/face`: photo match only

Compares a selfie photo with the user's profile photos.

> ⚠️ **No liveness:** a printed or on-screen photo can pass. Every response has `liveness_checked: false`. **Don't grant the Verified badge on this.** Use 5a + 5b.

### Request: `multipart/form-data`

| Part | Type | Required | Notes |
|---|---|---|---|
| `selfie` | image file | ✓ | One image. JPEG or PNG. The face should be clear, front-facing, and well lit. |
| `profile_photos` | image file(s) | ✓ | One or more. Repeat the field name for each photo. |

- **Size limit:** 8 MB per image.
- **Storage:** images are processed in memory and **never stored** by the service (DPDP: raw biometrics aren't retained).

Example:
```bash
curl -X POST http://127.0.0.1:8765/v1/verify/face \
  -F "selfie=@selfie.jpg" \
  -F "profile_photos=@photo1.jpg" \
  -F "profile_photos=@photo2.jpg"
```

### Response `200`: decided
```json
{
  "decision": "pass",
  "best_similarity": 0.821,
  "per_photo": [0.821, null],
  "model_version": "yunet2023mar-sface2021dec",
  "liveness_checked": false
}
```

| Field | Type | Notes |
|---|---|---|
| `decision` | string | `pass`, `review` or `fail` (table below) |
| `best_similarity` | float −1–1 | Highest similarity across the profile photos |
| `per_photo` | (float or null)[] | Same order as uploaded. `null` = no usable face in that photo. |
| `model_version` | string | Store with the verification record |
| `liveness_checked` | bool | Always `false` for now |

| `decision` | Rule (thresholds calibrated on LFW) | Backend should |
|---|---|---|
| `pass` | similarity ≥ 0.368 | Mark face-matched (see liveness warning) |
| `review` | 0.358 – 0.368 | Queue for a human moderator |
| `fail` | < 0.358 | Show "didn't match, try again"; limit to 3 attempts per day |

- **Where the thresholds come from:** they're recalculated whenever `face_verification.train` runs. On LFW they gave 0% strangers accepted and 2.1% real people rejected. Recalibrate on consented Indian photo pairs before launch.

### Response `200`: retry needed
```json
{ "decision": "retry", "reason": "too_blurry", "liveness_checked": false }
```

| `reason` | Meaning | Tell the user |
|---|---|---|
| `no_face` | No face found in the selfie | "We couldn't see your face" |
| `face_too_small` | Face under 60 px | "Move closer to the camera" |
| `too_blurry` | Image too blurry | "Hold still" |
| `bad_lighting` | Too dark or too bright | "Find better light" |
| `no_face_in_profile_photos` | None of the profile photos has a usable face | "Add a clear profile photo of your face" |

### Errors
- `400` a file isn't an image
- `413` a file is over 8 MB
- `422` missing `selfie` or `profile_photos`
- `503` thresholds not calibrated

---

## 6. `POST /v1/moderate/message`

Check every chat message **before delivering it**.

### Request
```json
{ "text": "bhai advance bhej do 500 rs to rahul@okaxis" }
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `text` | string | ✓ | Raw message text. Hindi, English and Hinglish are all supported. |

### Response `200`
```json
{
  "action": "flag",
  "toxicity": 0.196,
  "rules": ["upi_id", "money_request"],
  "model_version": "tox-charlr-v0"
}
```

| Field | Type | Notes |
|---|---|---|
| `action` | string | What to do (table below) |
| `toxicity` | float 0–1 | Probability the message is abusive |
| `rules` | string[] | Pattern rules that matched (table below). May be empty. |
| `model_version` | string | Store with any moderation record |

The first matching row decides the action:

| `action` | When | Backend should |
|---|---|---|
| `block` | toxicity ≥ 0.9 | Don't deliver. Tell the sender the message breaks the guidelines. Count toward the trust score. |
| `flag` | rule `upi_id`, `money_request`, or `laya_scam` (only when `USE_LAYA=1`) | Deliver, show the **recipient** a scam warning ("Never send money for tickets"), and add to the moderator queue |
| `nudge` | toxicity ≥ 0.5, or rule `phone_number`, `link` or `off_platform` | Show the **sender** "Are you sure?" before sending. Deliver if they confirm. |
| `allow` | everything else | Deliver |

| Rule | Matches |
|---|---|
| `phone_number` | Indian mobile numbers (10 digits starting 6–9, optional +91) |
| `upi_id` | UPI handles such as `name@okaxis`, `@ybl`, `@paytm`, `@upi`, `@ibl`, `@axl`, `@apl` |
| `link` | `http://`, `https://`, `www.` |
| `off_platform` | WhatsApp, Telegram, Instagram DM, Snapchat |
| `money_request` | "send/transfer/pay/bhej…" near "money/paise/rs/₹/advance/payment/upi" |
| `laya_scam` | Only with `USE_LAYA=1`: Laya's semantic scam check scores ≥ 0.8 (asks for money, payment or personal details, or moves the chat off the app) |

### Errors
- `422` bad body
- `503` model not trained

---

## 7. Trip planner

Plans a trip around one event: travel legs for each member, a stay near the venue, food before and after the show, getting back after a late show, and places to visit the next day or days. It works for one person or a clan. Design: [TRIP_PLANNER_DESIGN.md](../TRIP_PLANNER_DESIGN.md).

**How it works:**
- **Slow work is queued.** Creating a trip queues a **draft**, and sending a chat message queues a **turn**. Both return `202` straight away. Follow them on the event stream (7.7), then re-read the trip (7.2).
- **Card actions are immediate:** add, move, remove, undo, members. They run in the request and go through the same rules as the planner.
- **Storage:** Postgres (`ml` schema), including every version and the whole chat, kept forever (erasure: 7.22).
- **Every call is made as a user.** Spring passes the signed-in `user_id`, and the service checks trip membership (`403` otherwise).
- **Auth:** with `ML_SERVICE_TOKEN` set, send the header `X-Service-Token` on every call except `/v1/go`.

**Typical flow from Spring:**
1. `POST /v1/trips` returns `{trip_id, job_id, trip}`. Render `trip` (the skeleton, version 0) straight away.
2. Open `GET /v1/trips/{trip_id}/stream?user_id=…&until_job={job_id}`. Show `status` and `tool` as progress lines, and `message` as the planner's chat reply.
3. On `trip_updated` or `done`, call `GET /v1/trips/{trip_id}` and re-render.
4. For a chat message, `POST …/messages` returns `{turn_id}`; then repeat step 2 with `until_job={turn_id}`.

### 7.0 Shared objects

#### Trip view
Returned by 7.2 and, as `trip`, by 7.1 and every card action.

```json
{
  "trip_id": "21aa3e2ba229",
  "version": 5,
  "clan_id": null,
  "event": { "event_id": 9001, "title": "Midnight Echoes Live at NSCI Dome", "city": "Mumbai",
             "venue_name": "NSCI Dome", "venue_lat": 18.9985, "venue_lng": 72.8175,
             "start_time": "2026-10-07 19:00:00", "gates_open": "2026-10-07 17:30:00", "end_time": "2026-10-07 22:45:00" },
  "timeline": { "arrive_by": "2026-10-07 14:30", "check_in": "2026-10-07 14:00", "gates_open": "2026-10-07 17:30",
                "show_start": "2026-10-07 19:00", "show_end": "2026-10-07 22:45", "late_finish": false,
                "check_out": "2026-10-08 11:00", "depart_after": "2026-10-08 18:00",
                "show_day": "2026-10-07 00:00", "days": 1 },
  "members": [ { "user_id": 7, "first_name": "Riya", "origin_city": "Pune", "role": "creator" } ],
  "preferences": { "budget_band": "balanced", "hotel_min_stars": null, "max_distance_km": 5.0, "veg_only": true,
                   "interests": [], "travelling_solo": false, "extra_days": 0, "rooms": null, "lang": "en" },
  "locks": {},
  "items": [ "…Item objects, sorted by time…" ],
  "map": [ { "item_id": "d97bf4324f", "slot": "stay", "name": "Sunbeam Geeta", "lat": 18.9990, "lng": 72.8170 },
           { "item_id": null, "slot": "venue", "name": "NSCI Dome", "lat": 18.9985, "lng": 72.8175 } ],
  "costs": { "…Costs object…": "" },
  "variants": { "budget": { "stay": "Treebo Amber Imperial", "cost_per_person": { "min": 3820, "max": 5260, "currency": "INR" } } },
  "decisions": [ "…open Decision objects…" ],
  "warnings": [ "Cafe Coffee Day is not tagged vegetarian: confirm before going" ],
  "booking_disclosure": "BuddySync may earn a commission when you book through these links. We rank options by fit for your trip, never by commission. We don't book or take payment.",
  "pending_jobs": [ { "job_id": "7205f1c47e45", "kind": "chat", "status": "running", "run_after": "2026-09-30T05:02:45+00:00" } ]
}
```

| Field | Type | Notes |
|---|---|---|
| `trip_id` | string | |
| `version` | int | Goes up by one on every change. `0` is the skeleton before the draft. Send it as `if_version` on card actions. |
| `clan_id` | int or null | |
| `event` | object | Snapshot of the event taken when the trip was created |
| `timeline.arrive_by` | datetime | Gates open − 3 h |
| `timeline.check_in` / `check_out` | datetime | 14:00 on show day / 11:00 on the last day |
| `timeline.gates_open`, `show_start`, `show_end` | datetime | From the event |
| `timeline.late_finish` | bool | The show ends at 23:00 or later, or before 05:00 |
| `timeline.depart_after` | datetime | 18:00 on the last day: after the day's explore time |
| `timeline.show_day` | datetime | Midnight of the show day |
| `timeline.days` | int | 1 + `extra_days` |
| `members[]` | object | `{user_id, first_name, origin_city, role}`. `role` is `creator` or `member`; every member has equal planning rights. |
| `preferences` | object | Table below |
| `locks` | object | `{slot: decision_id}` for slots locked by a group decision |
| `items[]` | Item | Below |
| `map[]` | object | Pins to draw, the venue included |
| `costs` | Costs | Below |
| `variants` | object | Clan trips only: `{budget/balanced/premium: {stay, cost_per_person}}`. Only options that really differ are listed. |
| `decisions[]` | Decision | Open group decisions only (all of them: 7.20) |
| `warnings[]` | string[] | Trip-level soft problems: over budget, late show with a far hotel, last metro before the show ends |
| `booking_disclosure` | string | **Show it next to the Book buttons** |
| `pending_jobs[]` | object | Queued or running jobs (`draft`, `chat`, `decision`). Show "planner is working" while this list isn't empty. |

**Preferences** (send any of them in `prefs` on 7.1; the planner changes them through chat):

| Key | Type | Default | Notes |
|---|---|---|---|
| `budget_band` | string | `balanced` | `budget` (≤ 3★, hostels allowed, ₹6,000 pp cap), `balanced` (3–4★, ₹15,000 pp), `premium` (4★+, no cap) |
| `hotel_min_stars` | int or null | `null` | 1–5 |
| `max_distance_km` | float | `5.0` | Hotel distance from the venue |
| `veg_only` | bool | `false` | |
| `interests` | string[] | `[]` | e.g. `["museum", "fort"]`: used to choose explore places |
| `travelling_solo` | bool | `false` | Prefer a closer hotel and a cab back. Never inferred. |
| `extra_days` | int | `0` | Days after the show, 0–4 |
| `rooms` | int or null | `null` | Default `ceil(members / 2)` |
| `lang` | string | `en` | `en`, `hi` or `hinglish`: the language of the first draft message (chat replies follow the user's language) |

#### Item
```json
{
  "item_id": "d97bf4324f", "slot": "stay", "slot_label": "where to sleep (one hotel)", "day": 0,
  "start": "2026-10-07 14:00", "end": "2026-10-08 11:00", "fixed": false, "member_id": null,
  "ref": { "kind": "hotel", "id": "lp655731e0" }, "title": "Bloom Hotel - Worli",
  "why": "3-star, 0.8 km from the venue", "note": "",
  "price": { "min": 10190, "max": 11200, "currency": "INR", "unit": "night", "confidence": "range",
             "source": "LiteAPI sandbox rates (test inventory)", "fetched_at": "2026-09-30 10:00",
             "room": "Deluxe Room", "board": "Room Only", "refundable": true },
  "book": [ { "provider": "booking", "label": "Booking.com", "url": "https://www.booking.com/searchresults.html?…",
              "tracked_url": "/v1/go?url=…&trip_id=21aa3e2ba229&item_id=d97bf4324f&provider=booking" } ],
  "added_by": "planner", "booking_status": "suggested", "locked": false,
  "votes": { "up": 1, "down": 0, "by": { "8": 1 } },
  "warnings": [],
  "place": { "name": "Bloom Hotel - Worli", "kind": "hotel", "lat": 19.0061, "lng": 72.8162, "stars": 3,
             "cuisine": null, "veg": false, "opening_hours": "unknown: check hours", "km_to_venue": 0.79,
             "maps_url": "https://www.google.com/maps/search/?api=1&query=19.0061,72.8162",
             "guest_rating": null, "reviews": 461, "photo": "https://…jpg", "source": "liteapi" }
}
```

| Field | Type | Notes |
|---|---|---|
| `item_id` | string | Use in card actions and votes |
| `slot` | string | `travel_in`, `stay`, `before_show`, `show`, `after_show`, `return`, `explore`, `travel_out`. `travel_*`, `show` and `return` are built by code; the other four hold catalog places. |
| `slot_label` | string | Human-readable slot description |
| `day` | int | `0` is show day. `explore`: 1 = the day after the show. `travel_out`: the last day. |
| `start` / `end` | datetime or null | Set by code inside the slot's time window. `travel_in` has only `end` (arrive by), `travel_out` only `start` (leave after). |
| `fixed` | bool | A user pinned the start time (7.9) |
| `member_id` | int or null | Only on `travel_in` / `travel_out`: one pair of legs per member who isn't from the event city |
| `ref` | object | `{kind: hotel/place/event/route, id}`. For places, `id` is a catalog `place_id`: `n…`/`w…`/`r…` from OpenStreetMap, or `lp…` from LiteAPI. |
| `title`, `why`, `note` | string | `why` is one line from the planner |
| `price` | Price or null | Below. `null` means no estimate (e.g. a leg from a town we can't price). |
| `book[]` | Link | Hand-off links, below |
| `added_by` | string | `planner`, `auto`, `system`, `group decision` or `user:<id>` |
| `booking_status` | string | `suggested` (booking checklist coming later) |
| `locked` | bool | The slot is locked by a group decision |
| `votes` | object | `{up, down, by: {user_id: ±1}}` (7.18) |
| `warnings[]` | string[] | Problems with this item, e.g. "not tagged vegetarian", "open only 16:00–17:00 of that slot", "the cheapest fare lands after the arrive-by time" |
| `place` | object | Catalog places only. `guest_rating` (out of 10), `reviews`, `photo` and `source: "liteapi"` are only on LiteAPI hotels. |
| `mode` | string | Travel legs only: `train` (AC 3-tier fare, trips under about 700 km) or `flight` |
| `train` | object | Travel legs: `{rail_km: [lo, hi], classes: {SL/3A/2A: {min, max}}, labels, source}`. Estimated from the Railway Board fare chart TAG-2026 plus the Dec 2025 revision; rail distance is estimated at 1.15–1.4 × straight-line distance. Tatkal costs more. |
| `flight` | object | Travel legs with a known airport pair: `{airline, flight_number, departs, arrives, stops, same_day, from, to}`, the cheapest cached fare (Travelpayouts), not a live seat |

**Price:**

| Field | Notes |
|---|---|
| `min`, `max` | INR. If they're equal, show "from ₹min". |
| `unit` | `night` (one room), `person` or `trip` (a shared cab) |
| `confidence` | `estimate`, `range` or `verified`. **Show it next to the price.** |
| `source`, `fetched_at` | Where the price came from, and when |
| `room`, `board`, `refundable` | LiteAPI hotels only |

Where each price comes from:
- **LiteAPI hotels:** a rate for 2 adults in one room for the trip dates, from LiteAPI's retail rate to the public price other sites show. It's a `range` with a sandbox key and `verified` with a production key, and refreshed if over 1 h old.
- **Other hotels, food and sights:** `estimate` bands.
- **Flights:** a `range` over the cheapest fares seen for up to 3 days before the show (inbound) or after the trip (outbound).
- **Trains:** `estimate`.

**Link:** `{provider, label, url, tracked_url, affiliate?}`. Use **`tracked_url`** for the Book button (7.23), prefixed with the ML base URL and with `&user_id=` added. Providers include `booking`, `agoda`, `aviasales` (an affiliate link when `TRAVELPAYOUTS_MARKER` is set), `google_flights`, `irctc`, `redbus`, `uber` and `maps`.

#### Costs
```json
{
  "per_person": { "min": 2770, "max": 6720, "currency": "INR" },
  "lines": [ { "item_id": "1144b76ad2", "title": "Pune → Mumbai", "min": 510, "max": 560, "confidence": "estimate" } ],
  "members": 1, "rooms": 1, "nights": 1,
  "not_included": [ "event ticket", "train / bus / flight tickets (see Book links)" ],
  "per_member": { "7": { "min": 2770, "max": 6720, "currency": "INR" } }
}
```

| Field | Notes |
|---|---|
| `per_person` | The group average. Each member's own legs are spread across the group. |
| `lines[]` | Each priced item's share per person |
| `members`, `rooms`, `nights` | What the split used: the stay is split per room per night, and a cab is shared one per 4 people |
| `not_included[]` | e.g. the event ticket, or legs without a fare |
| `per_member` | `{user_id: {min, max}}`: the shared items plus that member's own travel |

#### Decision
```json
{
  "decision_id": "92d991c4c80f", "kind": "lock", "payload": { "slot": "stay" }, "options": ["yes", "no"],
  "question": "Lock 'stay'?", "opened_by": 8, "deadline": "2026-10-01T05:02:45.557854+00:00",
  "status": "open", "result": null, "tally": { "yes": 1, "no": 0 }, "votes": { "8": "yes" }
}
```

| Field | Notes |
|---|---|
| `kind` / `payload` | `variant` / `{}` · `swap` / `{item_id, place_id}` · `lock` or `unlock` / `{slot}` (a place slot) · `pref` / `{changes: {…preferences…}}` |
| `options` | The variant names for `variant`; `["yes", "no"]` otherwise |
| `question` | Text for the poll card |
| `opened_by` | user_id, or `null` for the planner (the automatic vote after a clan draft) |
| `deadline` | UTC, 24 h after opening |
| `status` | `open` → `passed` (queued to apply) → `applied` / `failed`, or `rejected`, `tied` (the plan stays) or `expired` (no votes) |
| `result` | `{winner, tally}` or `{winner, version}` once applied, or `{error}` |
| `tally` / `votes` | Count per option / `{user_id: option}` |

**The rule:** a strict majority of members passes it immediately (a majority of `no` rejects it). Otherwise, at the deadline, the option with the most votes wins, and a tie or no votes keeps the current plan. The opener's vote counts as `yes` (not for `variant`).

---

### 7.1 `POST /v1/trips`: create a trip

#### Request
```json
{ "event_id": 9001, "user_id": 7, "origin_city": "Pune", "first_name": "Riya", "clan_id": null,
  "members": [ { "user_id": 8, "origin_city": "Bengaluru", "first_name": "Arjun" } ],
  "prefs": { "budget_band": "balanced", "veg_only": false, "extra_days": 0, "lang": "en" } }
```

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `event_id` | int | ✓ | | |
| `user_id` | int | ✓ | | Becomes the creator and first member |
| `origin_city` | string | ✓ | | Where the creator travels from (free text; about 50 Indian cities get fares) |
| `first_name` | string | | `""` | Shown to the planner. Never send phone numbers or emails. |
| `clan_id` | int or null | | `null` | |
| `members[]` | `{user_id, origin_city, first_name?}` | | `[]` | The other clan members |
| `prefs` | object | | defaults | Any preference keys (7.0). Unknown keys → `422`. |

#### Response `202`
```json
{ "trip_id": "21aa3e2ba229", "job_id": "2d31a169dbda", "trip": { "…Trip view, version 0…": "" } }
```

| Field | Notes |
|---|---|
| `trip_id` | |
| `job_id` | The queued draft: follow it on 7.7 with `until_job` |
| `trip` | The skeleton (show and travel legs, version 0) |

**The draft** fills the stay, before and after the show, the way back and the explore days, as one version:
- **From the cache**, if the same event, preferences and group size were planned before. Its `message` event has `model: "cache:…"`, and it takes under 5 s.
- **With the LLM**, 5–20 s.
- **Code only** when the LLM is missing or busy (`model: "auto"`), under 1 s.

Clan trips then get variants and an automatic vote on them.

#### Errors
| Status | When |
|---|---|
| `404` | Unknown `event_id` |
| `422` | Bad body, or unknown preference keys |
| `503` | No places catalog for the event's city, or storage not configured |

### 7.2 `GET /v1/trips/{trip_id}`: read a trip
| Query | Type | Required |
|---|---|---|
| `user_id` | int | ✓ |

**Response `200`:** the Trip view.
**Errors:** `403` not a member · `404` unknown trip.

### 7.3 `GET /v1/trips/{trip_id}/versions/{v}`: one past version
Query `user_id` (required).

**Response `200`:**
```json
{ "version": 1, "parent": 0, "items": [ "…Items as stored (no place, votes or tracked_url)…" ],
  "prefs": { "…": "" }, "locks": {}, "author": "planner", "reason": "first draft",
  "created_at": "2026-09-30T05:02:45+00:00" }
```
- `parent`: the version it was made from. Undo walks back through `parent`.
- `author`: `system`, `planner`, `auto`, `group decision` or `user:<id>`.

**Errors:** `403` · `404` unknown trip or version.

### 7.4 `POST /v1/trips/{trip_id}/messages`: send a chat message
#### Request
```json
{ "user_id": 7, "text": "Mujhe sirf veg khana chahiye" }
```
| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | int | ✓ | Must be a member |
| `text` | string | ✓ | 1–2,000 characters. Any language. |

#### Response `202`
```json
{ "turn_id": "7205f1c47e45", "message_seq": 2, "queued_behind": 0 }
```
| Field | Notes |
|---|---|
| `turn_id` | The queued turn: follow it on 7.7 with `until_job` |
| `message_seq` | The stored user message |
| `queued_behind` | Jobs ahead of it for this trip. Turns run one at a time, in order. |

**What the planner does:**
- It uses tools: read the trip, change preferences, search hotels and places, travel options with fares and Tatkal times, add / move / remove, costs, weather, undo, start a group decision, propose variants.
- It makes at most 12 tool calls per turn.
- It replies in the user's language and script.
- It can only add places from the catalog, and every edit goes through the rules.
- **Masking:** phone numbers, emails, UPI IDs and card numbers are masked before anything reaches the LLM. The stored chat keeps the original.
- **Prompt-injection guard:** a message like "ignore previous instructions" gets a fixed reply with `model: "guard"`, and never reaches the LLM.
- **Busy LLM:** the turn switches to the next model or provider mid-turn. If every one is busy, it re-queues with a growing delay (status: "Planner is busy… retrying in N s"), up to 5 tries, then fails with an `error` event.

#### Errors
| Status | When |
|---|---|
| `403` | Not a member |
| `404` | Unknown trip |
| `422` | Empty or too-long text |
| `429` | 20 messages per member per trip, or 60 per user per day |

### 7.5 `GET /v1/trips/{trip_id}/messages`: chat history
| Query | Type | Required | Default |
|---|---|---|---|
| `user_id` | int | ✓ | |
| `after` | int | | `0`: only messages with `seq` greater than this |

**Response `200`:**
```json
{ "messages": [
  { "seq": 1, "role": "assistant", "sender_id": null, "content": "Here's a first plan for Midnight Echoes…", "model": "auto", "created_at": "2026-09-30T05:02:45+00:00" },
  { "seq": 2, "role": "user", "sender_id": 7, "content": "Only veg food please", "model": null, "created_at": "…" },
  { "seq": 3, "role": "assistant", "sender_id": null, "content": "Done: veg only from now on.", "model": "qwen/qwen3.8-27b", "created_at": "…" } ] }
```
- `model` on assistant rows: the model name, `cache:<model>`, `auto` (the code-only draft) or `guard`.

**Errors:** `403` · `404`.

### 7.6 `POST /v1/trips/{trip_id}/voice`: speech → text
**Request:** `multipart/form-data`

| Part | Type | Required | Notes |
|---|---|---|---|
| `user_id` | text | ✓ | Must be a member |
| `audio` | file | ✓ | `audio/webm`, `ogg`, `mpeg`, `mp4`, `m4a`, `wav`, `flac` or `video/webm`. At most 10 MB. |

**Response `200`:**
```json
{ "text": "veg food after the show", "lang": "english", "seconds": 2.1 }
```
Show the text so the user can check it, then send it with 7.4. It uses Whisper `whisper-large-v3-turbo` on Groq, which handles Hindi and Hinglish.

**Errors:**
| Status | When |
|---|---|
| `400` | Unsupported type |
| `403` / `404` | |
| `413` | Over 10 MB |
| `422` | No words heard |
| `429` | Busy |
| `502` | Upstream error |
| `503` | `STT_API_KEY` not set |

### 7.7 `GET /v1/trips/{trip_id}/stream`: progress (server-sent events)
| Query / header | Type | Required | Notes |
|---|---|---|---|
| `user_id` | int | ✓ | |
| `after` | int | | Only events with `id` greater than this |
| `until_job` | string | | Close the stream after that job's `done` |
| `Last-Event-ID` header | int | | Resume after reconnecting |

**Response `200`** (`text/event-stream`). Each event has `id`, `event` and `data` (JSON, always with `job_id`):
```
id: 6
event: status
data: {"text": "Thinking…", "job_id": "7205f1c47e45"}

id: 7
event: tool
data: {"name": "update_preferences", "text": "Saving your preferences…", "job_id": "7205f1c47e45"}

id: 8
event: message
data: {"seq": 3, "text": "Done: veg only from now on.", "model": "qwen/qwen3.8-27b", "reply_to": 7, "job_id": "7205f1c47e45"}

id: 9
event: trip_updated
data: {"version": 2, "job_id": "7205f1c47e45"}

id: 10
event: done
data: {"job_id": "7205f1c47e45"}
```

| `event` | `data` | Show as |
|---|---|---|
| `status` | `{text}` | A progress line |
| `tool` | `{name, text}` | A progress line |
| `message` | `{seq, text, model, reply_to}` | The planner's chat reply. `reply_to` is the sender's user_id, or `null` for a draft. |
| `trip_updated` | `{version, by?}` | Re-fetch the trip |
| `decision_opened` | `{decision_id, kind, question, options, deadline, opened_by}` | **Spring posts a poll card in clan chat** |
| `decision_voted` | `{decision_id, tally}` | Update the poll card |
| `decision_resolved` | `{decision_id, status, winner?, tally?, version?, error?}` | Close the poll card |
| `votes_updated` | `{item_id, up, down}` | Update the thumbs |
| `error` | `{text}` | An error line in the chat. The job failed, and the user can resend. |
| `done` | `{job_id, failed?}` | That job is finished |

- **Keepalive:** a `: keepalive` comment every 15 s.
- **Card actions** emit `trip_updated` with `job_id: null`.

**Errors:** `403` · `404`.

### 7.8 `POST /v1/trips/{trip_id}/items`: add a place
#### Request
```json
{ "user_id": 7, "if_version": 4, "slot": "explore", "place_id": "n2562631544", "day": 1, "note": "" }
```
| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | int | ✓ | |
| `if_version` | int | ✓ | The version the user is looking at |
| `slot` | string | ✓ | `stay`, `before_show`, `after_show` or `explore`. Adding to `stay` replaces the hotel. |
| `place_id` | string | ✓ | From the catalog (e.g. an item's `ref.id`, or a planner search) |
| `day` | int | | `explore` only (1 = the day after the show) |
| `note` | string | | |

#### Response `200` (all card actions, 7.8–7.10 and 7.15–7.17)
```json
{ "result": { "ok": true, "version": 5, "warnings": [ "Cafe Coffee Day is not tagged vegetarian: confirm before going" ] },
  "trip": { "…Trip view…": "" } }
```

#### Errors (all card actions)
| Status | When | Example `detail` |
|---|---|---|
| `400` | A rule is broken | `"hotels go in 'stay' and only hotels go there"`, `"unknown place_id …"`, `"… is closed Wed 23:00–00:30 (OSM hours: …)"`, `"… has no rooms for these dates; pick another hotel"`, a duplicate |
| `403` | Not a member (for 7.16: not that member) | `"not a member of this trip"` |
| `404` | Unknown trip | `"unknown trip"` |
| `409` | Stale version, the planner is editing right now, or the slot is locked | `"stale version 0: the trip is at v4; refetch"`, `"the planner is updating this trip; retry in a moment"`, `"'stay' is locked by a group decision…"` |
| `422` | Bad body | |

### 7.9 `PATCH /v1/trips/{trip_id}/items/{item_id}`: move or pin a time
```json
{ "user_id": 7, "if_version": 5, "slot": "after_show", "day": 1, "start": "13:00" }
```
| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id`, `if_version` | int | ✓ | |
| `slot` | string | | Move to another place slot. Hotels only move within `stay`. |
| `day` | int | | |
| `start` | string `HH:MM` | | Pin the start time on that day. The item keeps its duration, and gets `fixed: true`. |

**Response and errors:** as in 7.8.

### 7.10 `DELETE /v1/trips/{trip_id}/items/{item_id}`: remove a place
Query: `user_id` and `if_version` (both required). Only place items (`stay`, `before_show`, `after_show`, `explore`) can be removed.
**Response and errors:** as in 7.8.

### 7.11 `POST /v1/trips/{trip_id}/undo`: undo the last change
```json
{ "user_id": 7, "if_version": 5 }
```
- `if_version` is optional. Undo creates a **new** version equal to the previous one, and repeated undo walks further back.
- It's refused with `409` if it would change something the group decided (a lock, or a locked slot).
- `400` `"nothing to undo"`.

**Response and errors:** as in 7.8.

### 7.12 `GET /v1/trips/{trip_id}/costs`: cost split
Query `user_id`. **Response `200`:** the Costs object (7.0). **Errors:** `403` · `404`.

### 7.13 `GET /v1/trips/{trip_id}/timeline-events`: reminders for Spring to schedule
Query `user_id`.

**Response `200`:**
```json
{ "version": 5, "reminders": [
  { "at": "2026-10-06 09:45", "kind": "tatkal", "text": "Tatkal for Wed 07 Oct opens at 10:00 (AC) and 11:00 (non-AC) today: book early on IRCTC", "member_ids": [7], "item_id": "1144b76ad2" },
  { "at": "2026-10-07 16:43", "kind": "leave_for_show", "text": "Leave Sunbeam Geeta by 16:58 to reach NSCI Dome (~2 min) as gates open at 17:30", "member_ids": null, "item_id": null },
  { "at": "2026-10-07 21:45", "kind": "last_metro", "text": "Mumbai's last metro is around 22:30, before the show ends at 22:45: plan a cab back", "member_ids": null, "item_id": null } ] }
```

| Field | Notes |
|---|---|
| `at` | IST, `YYYY-MM-DD HH:MM`. Sorted by time. |
| `kind` | `tatkal` (only for members going by train), `travel`, `check_in`, `leave_for_show`, `gates`, `last_metro`, `book_cab`, `check_out` |
| `member_ids` | `null` means everyone |
| `item_id` | The item it relates to, if any |

Fetch again after every `trip_updated`, and replace the scheduled notifications.

**Errors:** `403` · `404`.

### 7.14 `GET /v1/trips/users/{user_id}/trips`: a user's trips
**Response `200`:**
```json
{ "trips": [ { "trip_id": "21aa3e2ba229", "title": "Midnight Echoes Live at NSCI Dome", "city": "Mumbai", "updated_at": "2026-09-30T05:02:45+00:00" } ] }
```
Newest first. **Errors:** `503` storage not configured.

### 7.15 `POST /v1/trips/{trip_id}/members`: add a member
```json
{ "user_id": 7, "member": { "user_id": 8, "origin_city": "Delhi", "first_name": "Arjun" } }
```
- `user_id`: the member making the change. Spring has already checked the clan's rules.
- `member`: the person joining. They get their own travel legs, priced.

**Response:** as in 7.8. **Errors:** as in 7.8, plus `400` `"already a member"`.

### 7.16 `PATCH /v1/trips/{trip_id}/members/{member_id}`: change your origin city
```json
{ "user_id": 8, "origin_city": "Bengaluru" }
```
Only that member can change it (`403` otherwise). Their travel legs are rebuilt and re-priced.
**Response and errors:** as in 7.8.

### 7.17 `DELETE /v1/trips/{trip_id}/members/{member_id}`: remove a member
Query `user_id` (the member making the change; it can be the member themselves). Their travel legs are removed.
**Response and errors:** as in 7.8, plus `400` `"a trip needs at least one member"` / `"not a member"`.

### 7.18 `POST /v1/trips/{trip_id}/items/{item_id}/vote`: thumbs up or down
```json
{ "user_id": 8, "value": 1 }
```
`value` is `1`, `-1`, or `0` to clear.

**Response `200`:**
```json
{ "up": 1, "down": 0, "by": { "8": 1 } }
```
It doesn't change the trip version, and emits `votes_updated`.

**Errors:** `403` · `404` unknown trip or item · `422` a value outside −1…1.

### 7.19 `POST /v1/trips/{trip_id}/decisions`: open a group decision
```json
{ "user_id": 8, "kind": "lock", "payload": { "slot": "stay" } }
```
| `kind` | `payload` |
|---|---|
| `variant` | `{}`. Needs `variants` on the trip. |
| `swap` | `{item_id, place_id}` |
| `lock` / `unlock` | `{slot}`: `stay`, `before_show`, `after_show` or `explore` |
| `pref` | `{changes: {budget_band: "budget", …}}` |

**Response `201`:** the Decision (7.0), with the opener's `yes` counted. With a single member it's decided at once. It emits `decision_opened`.

**Errors:**
| Status | When |
|---|---|
| `400` | Invalid proposal: `"'stay' is already locked"`, `"no variants to choose from…"`, `"swap needs a place_id from search results"`, an unknown kind |
| `403` / `404` | |

### 7.20 `POST /v1/trips/{trip_id}/decisions/{decision_id}/vote` and `GET /v1/trips/{trip_id}/decisions`
**Vote request:**
```json
{ "user_id": 7, "option": "yes" }
```

**Vote response `200`:** the Decision, with its new `tally` and `status`. On `passed`, a worker applies it within about a second (`decision_resolved`, then `trip_updated`). It emits `decision_voted`.

**Vote errors:**
- `400` option not in `options`
- `403`
- `404` unknown decision
- `409` `"decision is already passed/applied/…"`

**List:** query `user_id`. Returns `{ "decisions": [ …Decision… ] }`, the last 50 of any status, newest first.

### 7.21 Group decisions: what happens when
| Event | Result |
|---|---|
| A clan draft finishes | Variants are built. If at least two differ, a `variant` decision opens (`opened_by: null`). |
| A strict majority votes the same option | `passed` (or `rejected` for `no`), and applied as a new version authored `group decision` |
| 24 h pass without a majority | The option with the most votes wins. A tie → `tied`; no votes → `expired`; the plan stays. Checked every 30 s. |
| A slot is locked | Card actions, the planner and undo can't change it (`409`). Only another decision can. |

### 7.22 `DELETE /v1/trips/users/{user_id}`: erasure (account deletion / DPDP request)
**Response `200`:**
```json
{ "messages_deleted": 12, "memberships_removed": 2 }
```
- **Deleted:** the user's chat messages and memberships.
- **Kept:** items they added stay, with the author set to `former member`.

Call it on account deletion, or on an erasure request. Chats are otherwise kept forever.

### 7.23 `GET /v1/go`: booking hand-off (click tracking)
| Query | Type | Required | Notes |
|---|---|---|---|
| `url` | string | ✓ | The partner URL (already inside `tracked_url`) |
| `trip_id`, `item_id`, `provider` | string | | Already inside `tracked_url` |
| `user_id` | int | | Add it when opening the link |

**Response `302`:** `Location` is the partner URL, and the click is logged.
- Only `https` links to known booking hosts are allowed: Aviasales, Booking.com, Agoda, IRCTC, redBus, Uber, Google, ixigo, ConfirmTkt, BookMyShow, District, AbhiBus, MakeMyTrip, Cleartrip, Travelpayouts.
- It's **public**, with no service token, because users' browsers open it.

**Errors:** `400` `"not a known booking link"` (never an open redirect).

### 7.24 Configuration (environment)
| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | unset → embedded dev Postgres in `data/pgdata` | **Always set in production** |
| `TRIP_WORKERS` | `2` | Worker threads inside the API process. Set `0` in production and run `python -m trip_planner.worker --threads N`. |
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | | OpenAI-compatible provider (Groq or Gemini). `LLM_MODEL` is a comma list of fallback models. |
| `LLM_DRAFT_MODEL`, `LLM_CHAT_MODEL`, `LLM_UTILITY_MODEL` | `LLM_MODEL` | Per-task model lists |
| `LLM_FALLBACK_BASE_URL`, `LLM_FALLBACK_API_KEY`, `LLM_FALLBACK_MODEL` | | Second provider, tried last. A turn switches to it mid-turn if the first runs out of quota. |
| `ANTHROPIC_API_KEY`, `CLAUDE_MODEL` | `claude-opus-5` | Use Claude for every task instead |
| `LITEAPI_KEY` | | Hotel rates for the trip dates (`sand_…` or `prod_…`) |
| `TRAVELPAYOUTS_TOKEN` | | Flight fare ranges |
| `TRAVELPAYOUTS_MARKER` | | Partner ID (783614): Aviasales links become affiliate links |
| `STT_API_KEY`, `STT_BASE_URL`, `STT_MODEL` | Groq, `whisper-large-v3-turbo` | Voice (7.6). Falls back to `LLM_FALLBACK_API_KEY`. |
| `ML_SERVICE_TOKEN` | | Require `X-Service-Token` on `/v1` calls (not on `/v1/go`) |
| `USE_LAYA` | | `1` adds Laya's semantic check to the prompt-injection guard (and to moderation) |

---

## 8. `GET /health`

- **Response `200`:** `{ "ok": true }`
- **Use for:** container liveness and readiness probes. It does not check whether the models are trained. A feature that isn't ready returns `503` on its own endpoint.

---

## Current limitations (affect how you use the API)

| Area | Limitation |
|---|---|
| All | Models are trained on **synthetic data**. Rankings work, but aren't tuned to real users yet. |
| All | No authentication. Keep the service on a private network. |
| buddies | Candidate pool includes every participant, not only those with `looking_for_buddy` (fix pending) |
| verify | **Liveness has not yet been tested on a real recording.** Only the attack videos are verified: all are rejected (still or moving photo, person swap, blurry or hand-covered stand-in). Thresholds (hold time, blink/mouth/turn/nose distances) are first guesses, so tune them on ~30 real attempts per action. It doesn't stop a live deepfake through a virtual camera (add Play Integrity / App Attest in the app) or someone switching between several photos of the victim. Face thresholds are calibrated on LFW, not Indian faces. Challenges are held in memory (single instance). |
| moderate | Toxicity model trained on a tiny seed set, so treat `toxicity` as rough. The pattern rules are reliable. |
| trips | Places catalogs for Mumbai, Delhi, Bengaluru, Pune and Hyderabad. Prices are estimate bands, not live rates, until LiteAPI and Travelpayouts are connected. Few OSM hotels have star ratings, so "5-star only" returns few results, and clan variants can collapse to one or two real options. Travel legs have links but no times or fares yet. The last-metro warning is city-level (earliest line), not per station. The weather uses Open-Meteo's free tier, which is non-commercial only, so production needs its paid plan. The Gemini free tier is often busy: turns wait and retry, and drafts fall back to the code planner. Voice, affiliate links and the click redirect aren't built yet (design §19, streams E–F). |
| responses | Response schemas aren't declared in code, so `/docs` shows inputs only. This file is the source of truth for outputs. |
