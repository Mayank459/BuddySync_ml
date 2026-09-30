# BuddySync Trip Planner: Finished Design

**Status:** target design, 30 Sep 2026. Team decisions confirmed 30 Sep 2026 (§20). It replaces the phased MVP in [ML_IMPLEMENTATION_PLAN.md §9](ML_IMPLEMENTATION_PLAN.md#9-f7-ai-trip-planner-plan-my-trip-around-this-show).
**Scope:** the complete product that launches, not a first cut. §19 gives the order to build it in.
**Basis:**
- the existing planner in `ml/trip_planner/`
- competitor research on Mindtrip, Layla (Expedia), Wanderlog, Winnie, MakeMyTrip Myra, ixigo TARA/ConfirmTkt, EaseMyTrip and Cleartrip
- research into booking and affiliate access in India

Items marked **(verify)** are unconfirmed.

---

## 0. Positioning

**What it is:**
- A trip planner built around one live event.
- It plans getting there, a stay near the venue, the show-day timeline, getting back safely after a late show, and optional extra days.
- It works for a buddy pair (duo) or a group of 3–6 (clan).

**Why it wins.** Nobody else combines all four of these:

| | Mindtrip | Layla | Wanderlog | Winnie | Indian travel booking sites | **BuddySync** |
|---|---|---|---|---|---|---|
| Knows the event (gates, end time) | Events feature since Nov 2025, generic | ✗ | ✗ | ✗ | ✗ | **✓ exact gate and end times** |
| Plans for a group | group chat, merged preferences | ✗ | shared editing, no voting | "coming soon" | ✗ | **✓ members, votes, locks, cost split** |
| India logistics (trains, INR, Tatkal, late-night return) | weak | weak | ✗ | ✓ | ✓ | **✓** |
| Hindi / Hinglish | ✗ | ✗ | ✗ | ✗ | ✓ (Myra, TARA) | **✓** |

**Principles.** Each one is backed by what competitors got wrong:
1. **Every place and every price has a source.** The LLM may only pick from IDs our tools returned, and every price carries a confidence tag. Competitors' most common failure is invented or closed places.
2. **Send a draft first, then refine.** The first reply is a complete plan. Ask at most one question, and only if the plan can't be built without the answer.
3. **Edit only the part that changed.** Change what was asked, keep everything else, and keep a version history with undo. Layla re-plans the whole trip on every change, and users complain it "forgets".
4. **Never lose a trip.** Trips are saved on every change. Losing a saved plan is the top complaint about both Layla and Mindtrip.
5. **Hand off every booking, and be honest about money.** Use affiliate deep links, no dark-pattern trials, and disclose that we earn commission. Rank by fit, never by commission.
6. **Code enforces the rules; the LLM chooses and explains.** Time windows, budgets, opening hours, locks and safety rules are all checked in code.

**Out of scope, deliberately:**
- **Taking payments or issuing tickets.** It needs an RBI payment-aggregator licence or merchant-of-record liability, IATA accreditation, and refund and chargeback support.
- **IRCTC booking.** A direct partner pays a ₹20 lakh fee plus a ₹10 lakh deposit and needs ₹5 crore turnover.
- **An open-ended agent.** A group of cooperating agents costs more tokens and time for no gain at this size.

---

## 1. User experience

### 1.1 Entry points
- **Event page → "Plan my trip"**, for a solo user or a duo.
- **In clan chat**, `@planner plan our trip` creates a clan trip with every member added.
- **Paste a link or screenshot.** A BookMyShow, District or organiser link, or a screenshot, is matched to our event record using the event_ingestion dedup model. Mindtrip's "Start Anywhere" does the same.
- **Shared link.** A read-only view that anyone can open. Signed-in members can edit.

### 1.2 First draft (under 20 s, streamed)
The user gives their origin city, budget band (optional) and dates. The dates default from the event.

They see progress as it happens:
- "Finding hotels near Royal Hall…"
- "Checking trains from Pune…"

Then the draft arrives, with every slot filled:

| Block | Content |
|---|---|
| Getting there | Train, bus and flight options that arrive 3–4 h before gates open. Each has a price tag and a book link. |
| Stay | One chosen hotel plus 2 alternatives, all near the venue, with check-in and check-out aligned to the show |
| Show day | What to do before gates, gate time, the show, food after the show |
| Getting back | Hotel return plan: walk, metro or cab. A late-finish warning compares the show's end with the last metro. |
| Extra days (optional) | 3–5 "worth visiting" places near the stay, grouped by day |
| Going home | Return options timed after check-out |
| Cost | A per-person total as a range, and what's included |

It ends with **at most one** follow-up question (e.g. "Veg only?").

### 1.3 Trip screen (web and mobile layout)
- **Map:** pins for the venue, hotel and every place.
- **Timeline:** Day 0 is travel, then show day, extra days and departure.
- **Chat:** a drawer on mobile, a side panel on desktop.

Every item is a **place card** showing:
- name and kind
- distance from the venue or hotel
- opening hours, with "check hours" when the data is unsure
- a price chip showing its confidence (VERIFIED / RANGE / ESTIMATE)
- an "Open in Maps" link and a **Book** link
- votes and lock state (clans)
- "Why this" in one line from the planner

**Editing:**
- By chat: "swap dinner for something veg", "cheaper hotel", "add a day".
- Directly on a card: remove, move to another slot or time, lock, vote.
- Every edit creates a new version, and there is an **Undo** control and a version history.

### 1.4 Clan planning
- Each member has their own origin city, so each gets their own travel plan. The stay and show-day plan are shared.
- **Any member can pick.** Picking a variant, swapping an item, or locking or unlocking a slot opens a **group decision**:
  - It passes as soon as a strict majority agrees.
  - Otherwise, **after 24 h the option with the most votes wins**.
  - On a tie, the current plan stays.
- For a clan, the draft comes in **3 variants: budget, balanced, premium**, shown as a group-decision card in clan chat.
- Members vote on each item. **Locked** slots can't be changed by the planner (a hard rule). Only another group decision can unlock them.
- A **per-person cost split** covers stay by room occupancy, each member's own transport, and shared food and activities.
- **Settle-up after the trip:**
  - Members log what they actually paid, and see who owes whom.
  - Each debt gets a one-tap UPI payment link.
  - This is a structured feature, not chat text. That way it doesn't trip the moderation rule that flags UPI IDs in chat.

### 1.5 Before and during the trip ("Trip mode")
- **Booking checklist.** Each item that needs booking shows *not booked* or *I booked this*. The user can paste a confirmation email or SMS, and the LLM pulls out the details (train number, hotel, times). Wanderlog does this with email forwarding.
- **Reminders** (Spring notifications on a schedule built from the timeline):
  - Tatkal opens tomorrow at 10:00 (AC) or 11:00 (non-AC)
  - check-in from 14:00
  - leave the hotel by 17:30 to reach the gates
  - last metro 23:20, and the show ends at 23:30, so book a cab
- **Offline copy** of the itinerary and map tiles on mobile.

### 1.6 Language and voice
- The planner replies in the language the user wrote in: English, Hindi or Hinglish, written in Devanagari or the Latin alphabet.
- A **voice input** button uses speech-to-text (§12). Replies are text only; there is no spoken output.

---

## 2. Architecture

```
React (web/mobile) ──REST / SSE / STOMP──► Spring Boot ──(service token, user_id)──► ML API (FastAPI, stateless)
     trip screen, map, cards,               auth, clan chat,                            /v1/trips/*  (enqueue, read)
     chat, votes, settle-up                  notifications,                                 │
                                             expenses & settle-up                           ▼
                                                                         Postgres (ml schema) is also the job queue
                                                                         (FOR UPDATE SKIP LOCKED), the per-trip lock
                                                                         and the event log the SSE stream reads. No Redis.
                                                                                            │
                                                                                            ▼
                                                                              Planner workers (N)
                                                                              draft pipeline · chat turns ·
                                                                              price refresh · reminders feed
                                                                                            │
                          PostgreSQL (+PostGIS)  ◄──────────────────────────────────────────┤
                            app tables (Spring)                                             │
                            ml.trips*, ml.places, ml.price_snapshots (ML)                   ▼
                                                                          External: LLM provider(s), OSRM (self-host),
                                                                          Open-Meteo, Travelpayouts Data API, LiteAPI,
                                                                          Viator Basic API, affiliate link builders
```

**Ownership.** This follows the rule from ML plan §1 that only Spring writes business tables:

| Data | Owner | Why |
|---|---|---|
| Trips, versions, planner messages, votes, locks, place catalog, price snapshots | **ML service** (`ml` schema) | Planner state is model input and output. Spring passes IDs and relays events. |
| Clan chat messages, notifications, expenses and settlements, users, clans, events | **Spring** | Business data and money |

**Request flow for a chat turn:**
1. React sends the message to Spring.
2. Spring checks the user is a trip member and calls `POST /v1/trips/{id}/messages`, which returns `202 {turn_id}`.
3. A worker runs the turn and writes events to `ml.trip_events`: `status`, `tool`, `message`, `trip_updated`, `done` and `error`.
4. `GET /v1/trips/{id}/stream` sends them as server-sent events, and can resume with `Last-Event-ID`. Spring relays them to React over SSE, or to clan chat over STOMP.

A turn takes 7–70 s on real providers, which is why it's never done synchronously.

**Concurrency.**
- **One job at a time per trip.** A worker claims a job with `FOR UPDATE SKIP LOCKED` and sets `ml.trips.busy`. Later messages for that trip queue behind it, in order. Other trips run in parallel. Jobs from a dead worker are re-queued after 10 min.
- **Card actions** (votes, locks, add, move, remove, undo) use **optimistic versioning**. The request sends `if_version`; a mismatch, or an edit while the planner is busy, returns 409 so the client refetches.
- **Why Postgres, not Redis:** at our scale Postgres handles the queue, the lock and the events with one less system to run. Move to Redis or LISTEN/NOTIFY only when open streams reach the hundreds, because the stream currently polls every 0.5 s.

---

## 3. Trip model

```jsonc
Trip {
  trip_id, event_id, owner_id, clan_id?, status: drafting|active|archived,
  members: [{user_id, first_name, origin_city, role: creator|member (equal planning rights), prefs_override?}],
  prefs: {budget_band: budget|balanced|premium, hotel_min_stars?, max_distance_km, veg_only,
          interests[], travelling_solo, extra_days, rooms?},
  timeline: {arrive_by, check_in, gates_open, show_start, show_end, late_finish, check_out, depart_after},
  items: [Item], variants?: {budget|balanced|premium: [Item]}, chosen_variant?,
  locks: {slot: decision_id}, open_decisions: [decision_id], version, updated_at
}
Item {
  item_id, slot: travel_in|stay|before_show|show|after_show|return|explore|travel_out,
  day: 0..n, start?, end?, member_id?,          // member_id only on per-member travel items
  ref: {kind: place|hotel|route|event, id},     // id MUST come from a tool result
  title, why, note, price: Price?, book: Link?, added_by: user_id|planner,
  booking_status: suggested|booked, votes: {user_id: +1|-1}
}
Price { min, max, currency: "INR", unit: night|person|trip, confidence: verified|range|estimate,
        source, fetched_at }
Link  { provider, url, affiliate: bool, link_id }   // url goes via /v1/go/{link_id} for click logging
```

What changes from today's code:
- Slots become **items with times and days**, which lets us support multi-day trips and check opening hours.
- Each item gets an `item_id`, so votes and locks can refer to it.
- **Per-member travel items** are added for clans.

---

## 4. Planning pipeline

### 4.1 Draft (workflow; one structured LLM call and at most one repair)
```
1 skeleton  (code)     event → timeline windows (arrive_by = gates − 3 h, check-in 14:00, check-out 11:00, late_finish = end ≥ 23:00)
2 retrieve  (code, parallel, cached)
                        hotels_near(venue, budget_band, stars)         → 15 candidates with Price
                        places_near(venue|hotel, kinds by slot, veg)   → 15 per slot
                        transport(origin_i → city, date) per member    → train/bus/flight options with Price + Link
                        travel_matrix(candidates)  OSRM                → minutes walk/drive
                        last_transit(city, venue)                      → last metro/local time
                        weather(city, dates)                           → Open-Meteo
3 plan      (LLM)       fill skeleton choosing ONLY candidate IDs; strict JSON schema; "why" per item;
                        clans: 3 variants
4 validate  (code)      §6 rules → hard failures sent back once for repair, else item dropped + warning
5 save + cache          ml.trip_versions v1; cache key (event_id, origin_city, budget_band, group_size)
```
- **Cache.** Popular events get drafts pre-built overnight for the top 5 origin cities using the Batch API. A cache hit returns in under 1 s, and only the price chips are refreshed.
- **If the LLM is down,** step 3 falls back to code: nearest hotel in the budget band with the best stars, the highest-rated places per slot, and the fastest transport. The trip still works and is labelled "auto-planned".

### 4.2 Chat turn (a limited loop of tool calls)
- This is the loop that exists today, extended:
  - at most 12 tool calls per turn
  - the same §6 validator runs after every edit tool
  - warnings go back to the model so it tells the user
- **Context sent each turn:** fixed instructions and tool definitions, then a compact trip snapshot, then the last 10 messages. The snapshot means we never resend the full history. That keeps turns under about 6k input tokens, which also fits Groq's 8k tokens-per-minute free tier.
- Older messages get summarised into `trip.memory`, for example "Riya is vegetarian; Arjun arrives by bus".

### 4.3 Re-planning triggers (code-initiated)

| Trigger | What happens |
|---|---|
| The event's times change (event_ingestion update) | Re-run the timeline checks. Affected items are flagged and the members notified. |
| A member joins or leaves, or their origin changes | Build or remove that member's travel items. Recompute the cost split. |
| A price refresh moves an item out of budget | Add a warning. The planner suggests an alternative on the next turn. |
| A place is closed or removed from the catalog | Flag the item and suggest the closest alternative |

---

## 5. Planner tools (LLM-facing)

| Tool | Purpose | Notes |
|---|---|---|
| `get_trip` | Snapshot: event, timeline, members, preferences, items, locks, votes, warnings, cost summary | Called first on every turn |
| `update_preferences` | Save preferences that should stay applied, trip-wide or per member | Existing |
| `search_hotels` | Hotels near the venue, with Price and a book Link | Adds budget band and price |
| `search_places` | Food and sights by slot, near the venue or hotel | Adds opening-hours fit for the slot time |
| `get_transport_options` | Train, bus and flight options per member, with Price and Link, plus the Tatkal opening time | Replaces `get_transport_links` |
| `add_item` / `remove_item` / `move_item` | Edit the plan, with slot, day and time | The validator runs after each; locked slots are refused |
| `set_variant` | Generate or choose budget / balanced / premium variants | Clans |
| `get_votes` | Vote tallies per item | So the planner can say "3 of 4 prefer X" |
| `estimate_costs` | Per-person split, as ranges | From item Prices |
| `get_weather` | Forecast for the trip dates | Rain, so pick indoor places |
| `undo` | Revert to the previous version | Existing |

Things the model is **not** given: payment, sending messages to other people, arbitrary web search, and writing to anything outside the trip.

---

## 6. Validation rules

| Rule | Type | Check |
|---|---|---|
| Every `ref.id` exists in the catalog or the results of this turn's tools | **hard** | Rejected with an error; no invented places |
| Stay slot holds only hotels; other slots hold no hotels | **hard** | Existing |
| Locked slot unchanged | **hard** | Refused, with "locked by Riya" |
| Arrival ≤ gates − 90 min buffer | **hard** for trains and flights | Suggest an earlier option |
| Item times fall within opening hours | soft (hard if we know it's closed) | OSM `opening_hours` parser (verify: which library); "check hours" when missing |
| Travel time between items fits the gap | soft | OSRM matrix; walking under 1.5 km, otherwise drive |
| Late finish, and the hotel is more than 3 km away or the last metro leaves before show end + 30 min | soft (safety) | Suggest a closer hotel or pre-booking a cab |
| `travelling_solo`: the walk from venue to hotel is under 1 km on main roads, or a cab is advised | soft (safety) | Heuristic on OSM road class (verify) |
| Per-person total ≤ budget band maximum | soft | Warning, and the planner offers cuts |
| Veg only: restaurants and cafés tagged veg | soft | Existing |
| Hotel stars ≥ preference | soft | Existing |

Every validator failure is logged to `ml.planner_log`. These logs are how the eval set grows (§15).

---

## 7. Data sources

| Need | Source | Refresh | Confidence it gives |
|---|---|---|---|
| Places (food, sights, hotels) | Our catalog: OpenStreetMap/Overpass, Wikidata, Wikivoyage "See/Eat". Expands to every city with an upcoming event. | Weekly, plus on demand when a new event city appears | n/a |
| Hotel rates | **LiteAPI** rates (verify: terms and commission) | 1–6 h TTL, and refreshed when viewed | **VERIFIED** with `fetched_at` |
| Hotel rates, fallback | Price bands by city tier × star rating, calibrated monthly | Monthly | **ESTIMATE** |
| Flights | Travelpayouts **Data API** (cached prices) | 15–30 min TTL | **RANGE**, "from ₹X, seen 2 h ago" |
| Trains | Static timetable (Indian Railways open data, verify) plus fares per class from the published distance-based fare table | Monthly | **RANGE** |
| Buses | Per-km band by bus type (sleeper/seater, AC/non-AC) | Monthly | **ESTIMATE** |
| Cabs and last mile | Distance × per-km rate by city | Monthly | **ESTIMATE** |
| Last metro / local train | Hand-maintained table per city and line | Monthly, plus when operators announce changes | n/a |
| Travel time | **OSRM**, self-hosted on an India OSM extract (car and foot profiles) | Monthly extract | n/a |
| Weather | Open-Meteo (commercial plan, about $29/month) | 6 h | n/a |
| Activities near the venue | **Viator Basic API** (free, read-only) | Daily | **RANGE** |
| Event data | Our events table (event_ingestion) | Live | **VERIFIED** |

Prices always show their confidence label and age. The planner is told: "never state a price without its confidence label".

---

## 8. Booking handoff and revenue

| Category | Link target (in order of preference) | Commission (research, Sep 2026) |
|---|---|---|
| Flights | Aviasales via Travelpayouts; fallback Cleartrip or ixigo via Cuelinks | about 1.1–1.3% of the ticket (Travelpayouts); Cleartrip about 4.5% |
| Hotels | **Booking.com** affiliate deep link and **Agoda** (direct or via Travelpayouts), both shown; MakeMyTrip or OYO via Cuelinks as fallback | Booking.com 25–40% of its commission; Agoda up to 5%; OYO 3–7%; MakeMyTrip about ₹135 per sale |
| Trains | ixigo or ConfirmTkt affiliate link (the user logs in with their own IRCTC ID); fallback plain IRCTC link | up to ₹90 per booking (ixigo) |
| Buses | redBus via Cuelinks or INRDeals (**mobile web link**, because in-app sales aren't tracked); AbhiBus | ₹24–60 per sale |
| Activities | Viator, GetYourGuide, Klook | about 8% |
| Event tickets | **BookMyShow or District**: a deep link to the event's own ticket page, from event_ingestion's source URL. BookMyShow goes through the Cuelinks affiliate programme; District has no public programme, so its link is plain. Organiser site as the fallback. | BookMyShow about ₹4.50 per sale, and none in the metros; District none |
| Cabs | Uber universal deep link with pickup and drop pre-filled; Ola deep link (verify) | none |

- **Link builder.** `ml/trip_planner/links.py` fills each link with origin, destination, dates and number of travellers. It adds the affiliate ID from an environment variable and `subid = trip_id` so we can track conversions. Every link goes through `GET /v1/go/{link_id}`, which records the click and then redirects.
- **Booking.com's AI clause.** Its partner terms reportedly require written approval to use it with an AI system (verify). We only use plain affiliate deep links, but request approval anyway when joining. If they refuse, Agoda takes its place.
- **Disclosure.** Each Book button carries a short note: "BuddySync may earn a commission. Ranking is by fit, not commission."
- **Revenue.** Affiliate commissions. No subscription at launch, and never an auto-renewing free trial, since that's the top Trustpilot complaint about Layla and Wanderlog.
- **Taxes.** 18% GST on commissions from Indian affiliate networks once turnover passes ₹20 lakh. Commissions from foreign networks may count as zero-rated exports. Confirm both with a CA.

---

## 9. Groups in detail

| Topic | Rule |
|---|---|
| Members | The clan's members are invited automatically. Each confirms their origin city on first open. |
| Roles | **Every member is equal for planning.** Anyone can edit unlocked slots, chat with the planner, vote, and start a group decision. Membership (add or remove) follows the clan's admin rules in Spring. |
| Group decisions | Anyone can start one: choose a variant, swap an item, lock or unlock a slot, or change a group-wide preference. It passes early with a strict majority (> half of members). Otherwise at 24 h the option with the most votes wins, and a tie keeps the current plan. The deadline is enforced by a worker job, and the result creates a new version authored by "group decision". A member's own travel items are the exception: only that member decides them. |
| Variants | Clan drafts have 3 variants, and choosing one is a group decision (poll card in clan chat). |
| Votes | +1 or −1 per item per member. The planner sees the tallies and prefers well-liked items when re-planning. |
| Locks | Set and removed by group decision. Hard rule in the validator. A locked item shows 🔒 and "locked by group decision". |
| Conflicting preferences | Hard ones (veg, max budget) apply to the whole group, using the strictest member's value. Soft ones (interests) are combined, and the planner explains any trade-off. |
| Rooms | `rooms = ceil(members / 2)` by default; the user can change it. The stay cost is split per room, then per person. |
| Cost split | Per person: own travel, stay share, and shared items divided equally. Shown as min–max. |
| Settle-up (Spring) | Members log actual expenses: payer, amount, and who shared it. Spring computes net balances with a greedy min-cash-flow algorithm. Each debt gets a pay link: `upi://pay?pa=<payee VPA>&pn=<name>&am=<amount>&cu=INR&tn=BuddySync trip`. The payee VPA is stored in their profile, never pasted in chat. |
| Planner in clan chat | `@planner …` messages are relayed with the sender's ID. The reply shows as a bot message with an updated trip card. Messages without `@planner` never go to the LLM. |

---

## 10. Safety and trust

- **Late-night return rules** are in §6. The show-day timeline always includes a return plan when the show ends late.
- **Solo travellers** get the `travelling_solo` preference, which applies closer-hotel and cab rules. It is never inferred from gender.
- **Prompt injection.** User text sent to the planner first passes through the moderation endpoint. It then goes through an injection check, using the regex + Laya semantic guard pattern already used in the CodeBase project. The tools can only affect the trip itself (see §5), so even a successful injection can't reach payments or other users.
- **Refusals.** The planner won't give advice on reselling tickets or scalping, and won't help anyone get around identity checks.
- **Moderation.** Planner replies aren't moderated (they're our own output). Clan chat moderation is unchanged; settle-up stays out of chat by design.

---

## 11. LLM providers, cost and limits

- **One interface for every provider**, which already exists: Claude natively, and any OpenAI-compatible API through `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL`, with a fallback list of models.
- **Separate model per task:**

| Task | Model setting | Default |
|---|---|---|
| Draft (structured JSON, quality matters) | `LLM_DRAFT_MODEL` | `gemini-3.8-flash,gemini-flash-lite-latest` |
| Chat edits | `LLM_CHAT_MODEL` | `gemini-flash-lite-latest,gemini-3.8-flash` |
| Summarising memory, extracting details from pasted confirmations | `LLM_UTILITY_MODEL` | `gemini-flash-lite-latest` |
| Speech-to-text | `STT_MODEL` | Whisper large-v3-turbo (Groq free tier) |

- **Production runs on the Gemini free tier** (team decision). The design works around its limits:
  - **Fallback models:** within Gemini, models fall back on 429 or 503, as they do today. A second free provider (Groq `qwen/qwen3.8-27b`) is the last resort, through `LLM_FALLBACK_BASE_URL`, `LLM_FALLBACK_API_KEY` and `LLM_FALLBACK_MODEL`.
  - **Queue and pacing:** when every model and provider returns 429 or 503, the job goes back in the Postgres queue with a growing delay (15 s, 30 s … up to 60 s; 5 tries for chat, 3 for a draft). The user sees "Planner is busy (free-tier limit), retrying in N s". Check the current requests-per-minute and requests-per-day limits in AI Studio.
  - **Draft cache and nightly pre-generation** matter even more, because every cache hit saves free-tier quota. When quota runs out, drafts fall back to the code-only planner (§4.1).
  - **Quota monitoring:** alert at 80% of the daily quota.
- **Moving to a paid tier later** is a configuration change: paid Gemini, or Claude through `ANTHROPIC_API_KEY`. Nothing else changes.
- **Data consequence of the free tier:** Google may use free-tier prompts and responses to improve its products, and human reviewers may read them. So:
  - the privacy notice and a first-use consent line must say trip chats are processed by Google's Gemini and may be used by Google to improve its services
  - the LLM never receives phone numbers, emails, UPI IDs or exact locations (§12)
  - this is a DPDP consent item. Review it before launch, and move to paid if consent turns out to be insufficient.
- **Budgets:**
  - 20 messages per member per trip
  - 60 messages per user per day
  - context per turn: the instructions, the trip's memory notes and the last 10 user/assistant pairs; the trip itself comes from `get_trip`
  - at most 12 tool calls per turn
  - all counted from `ml.trip_messages`
- **Target cost** is under about ₹5 per trip over its whole life, assuming a 70% draft-cache hit rate and cheap chat models. Measure it from logged `usage`.

---

## 12. Privacy (DPDP Act 2023 / Rules 2025)

- **What the LLM sees.** Members' first names and origin cities only: no phone numbers, emails or exact locations.
- **Retention.** Trip chats, versions and votes are **kept indefinitely** (team decision), so the whole trip history is always available.
- **Erasure rights still apply, as DPDP requires.** When a user deletes their account or asks for erasure, we delete their messages and votes and remove them from trips. Clan trips keep the items, with the author marked "former member". The indefinite retention covers active users only.
- **Provider.** Gemini free tier (§11). Consent text in the app names Google as the processor and says it may use the data to improve its services.
- **What we send the LLM.** User text is scrubbed before it's sent: phone numbers, emails and UPI IDs are masked using the regexes in `moderation/rules.py`.
- **Where data lives.** Postgres in an Indian region. Sending data abroad for LLM processing is covered in the privacy notice.

---

## 13. ML API (called by Spring only; service token plus `user_id`)

| Method | Path | Returns |
|---|---|---|
| POST | `/v1/trips` `{event_id, owner_id, origin_city, clan_id?, prefs?}` | `202 {trip_id, job_id}`; the draft streams in |
| GET | `/v1/trips/{id}` | Trip view: items with price, link and votes; map points; warnings; costs; `version` |
| GET | `/v1/trips/{id}/versions/{v}` | That version, read-only |
| POST | `/v1/trips/{id}/messages` `{user_id, text}` | `202 {turn_id}` |
| GET | `/v1/trips/{id}/stream` | SSE events: `status`, `tool`, `delta`, `trip_updated`, `done`, `error` |
| POST | `/v1/trips/{id}/voice` (audio) | `{text, lang}`; the client then sends it as a message |
| POST | `/v1/trips/{id}/items` / PATCH / DELETE `/items/{item_id}` `{user_id, if_version}` | Direct edits from cards (same validator) |
| POST | `/v1/trips/{id}/items/{item_id}/vote` `{user_id, value}` | Tally |
| POST | `/v1/trips/{id}/decisions` `{user_id, kind: variant\|swap\|lock\|unlock\|pref, payload}` | `{decision_id, deadline}`; posts a poll card to clan chat |
| POST | `/v1/trips/{id}/decisions/{decision_id}/vote` `{user_id, option}` | Tally; resolves early on a strict majority |
| POST / DELETE | `/v1/trips/{id}/members` `{user_id, origin_city}` | Rebuilds that member's travel items |
| POST | `/v1/trips/{id}/undo` `{user_id}` | Previous version |
| POST | `/v1/trips/{id}/bookings` `{user_id, item_id?, pasted_text?}` | Marks the item booked; extracts details |
| GET | `/v1/trips/{id}/costs` | Per-person split |
| GET | `/v1/trips/{id}/timeline-events` | Reminder times for Spring to schedule notifications |
| POST | `/v1/trips/resolve-link` `{url or image}` | The matching `event_id` |
| GET | `/v1/go/{link_id}` | Logs the click and redirects to the partner |

Errors:
- `404`: unknown trip
- `409`: stale version, or a slot is locked
- `429`: a limit was hit
- `503`: no LLM or no catalog. The trip is still created with the code-only draft.

---

## 14. Database (`ml` schema, Postgres + PostGIS)

```sql
ml.trips          (trip_id pk, event_id, owner_id, clan_id, status, prefs jsonb, memory text,
                   current_version int, chosen_variant, created_at, updated_at)
ml.trip_members   (trip_id, user_id, origin_city, role, prefs_override jsonb, pk(trip_id,user_id))
ml.trip_versions  (trip_id, version, items jsonb, prefs jsonb, locks jsonb, author, reason, created_at,
                   pk(trip_id,version))
ml.trip_messages  (trip_id, seq, role, sender_id, content jsonb, provider, model,
                   tokens_in, tokens_out, latency_ms, created_at, pk(trip_id,seq))
ml.trip_votes     (trip_id, item_id, user_id, value smallint, created_at, pk(trip_id,item_id,user_id))
ml.trip_decisions (decision_id pk, trip_id, kind, payload jsonb, options jsonb, opened_by, deadline,
                   status: open|passed|expired|tied, result jsonb, resolved_at)
ml.decision_votes (decision_id, user_id, option, pk(decision_id,user_id))
ml.places         (place_id pk, city, kind, name, geom geography(point), h3_r8, stars, veg, cuisine,
                   opening_hours, wikidata, source, refreshed_at)          -- gist(geom), (city,kind)
ml.price_snapshots(key pk, kind, min_inr, max_inr, unit, confidence, source, fetched_at)
ml.link_clicks    (link_id, trip_id, user_id, provider, clicked_at)
ml.planner_log    (trip_id, turn_id, kind: validator_fail|tool_error|fallback|refusal, detail jsonb, at)
```
Spring owns `trip_expenses(trip_id, payer_id, amount, shared_with[], note)` and the settlements.

---

## 15. Quality: evaluations and monitoring

**Eval suite.** It runs in CI for every prompt, model or tool change.

| Set | Size | Hard targets |
|---|---|---|
| Event × origin drafts | 30 past events × 3 origins | 0 invented places, 0 impossible time slots, arrival before gates, late-return plan present |
| Clan drafts | 10 clans of 3–5 people with mixed origins | per-member travel present, 3 variants, cost split adds up |
| Edit conversations | 40 scripted multi-turn chats (veg, cheaper, lock, undo) | only the requested part changes; locks respected; preferences stick |
| Language | 20 Hinglish or Hindi messages | reply in the same language (checked by language detection) |
| Adversarial | 20 (injection, "book and pay for me", ticket scalping, off-topic) | tools stay within the trip; polite refusal or redirect |
| Past failures | grows from `ml.planner_log` | no repeats |

The reply-quality rubric (relevant, short, one question at most) is scored by an LLM judge, and a human spot-checks 10% of it.

**Online metrics:**
- time to first draft (p50 and p95)
- chat turn latency
- validator failures per turn
- fallback rate
- tokens and ₹ per trip
- share of draft items the user keeps
- click-through and conversion on book links
- clan trips that reach "variant chosen"

---

## 16. Failure modes

| Failure | What the user sees |
|---|---|
| LLM provider down or rate-limited | The fallback model list is tried first. For a draft, a code-only draft labelled "auto-planned". For chat, "Planner is busy, your message is queued". |
| No catalog for the city yet | A catalog build job starts (Overpass, about 1–3 min). Meanwhile the draft has transport, stay estimates and a "limited local info" label. |
| A price source is down | Its chips fall back to ESTIMATE bands |
| An affiliate network is down or the program is paused | The plain provider link |
| OSRM down | Straight-line distance × city factor, with "approx" |
| Two people editing at once | 409, the client refetches, and the user's edit is replayed |

---

## 17. Frontend (React + TypeScript)

| Screen or component | Notes |
|---|---|
| `TripPage` | Map (MapLibre with OSM tiles), `Timeline`, chat panel |
| `PlaceCard` | Details, `PriceChip` (VERIFIED / RANGE / ESTIMATE colours and a tooltip with source and age), Book, Maps, votes, lock |
| `PlannerChat` | Streams SSE events; tool-progress lines; voice button; the answering model shown only in debug builds |
| `VariantPoll` | Clan chat card: 3 variants with per-person cost; vote |
| `CostSplit` | Per-person ranges; after the trip, expenses and UPI settle-up buttons |
| `TripMode` | Booking checklist, reminders, offline copy |
| `SharedTripView` | Read-only public link |

Accessibility: every map pin also appears in the timeline list, and price confidence is shown in text as well as colour.

---

## 18. What exists today (`ml/trip_planner/`)

Updated 30 Sep 2026. The ML side of streams **A–F** and **H** is built. What's left is external: production keys, the paid tiers, OSRM, and the React/Spring work.

| Built | Still to build |
|---|---|
| **Model and checker** (`model.py`): Trip and Item with times, days and IDs; timeline built around the show; code-built show, travel and return legs; per-person and per-member costs. Checks for invented places, wrong slot, duplicates, known-closed hours, locked slots, stars, distance, late finish, last metro, solo return, veg, crowded windows, budget, and flight arrival / departure timing | OSRM travel times (straight-line × 1.3 for now); a timetable-based arrival check for trains |
| **Tools** (`tools.py`): every edit checked; undo that never undoes a group decision; single-gender hostels never suggested; weather; group decisions and variants | |
| **Planner** (`planner.py`): candidates (search widens past 3 km if needed) → one JSON LLM call with the forecast → one repair → code fill; code-only draft; **draft cache** keyed on event, preferences and group size (origins don't matter) plus nightly **pre-generation** (`pregen.py`); chat tool loop that **fails over across providers mid-turn**; masking of phones, emails, UPI IDs and cards; **prompt-injection guard** (regex, plus Laya with `USE_LAYA=1`); memory folding | |
| **LLM layer** (`llm.py`): draft, chat and utility models; model and provider fallback | A paid tier: the Gemini free tier allows **20 requests per day per model** |
| **Storage and API** (`schema.sql`, `store.py`, `worker.py`, `api.py`): trips, versions and chat kept forever, erasure; Postgres queue, per-trip lock and event log (no Redis); retry on busy; async API with SSE; **voice** (`stt.py`, Whisper on Groq); **reminder feed** (`reminders.py`); **click redirect** `/v1/go` with an allow-list and a commission disclosure; **service token** (`ML_SERVICE_TOKEN`) | "Start anywhere" (resolve a BookMyShow/District link); booking checklist and pasted confirmations |
| **Groups** (`groups.py`): members and their legs; group decisions with the majority / 24 h rule; locks; deduplicated clan variants with an automatic vote; item votes; per-member costs | Settle-up and the clan-chat poll cards (Spring) |
| **Data:** OSM catalogs for 5 cities, plus **LiteAPI hotels** (1.1k–3.9k per city) with **rates for the trip dates** (`hotels.py`); **flight fares** from Travelpayouts cached fares with affiliate marker 783614 (`fares.py`); **train fares** from the Railway Board TAG-2026 chart plus the Dec 2025 revision (`rail.py`, `data/rail_fares.csv`); last-metro table (`transit.py`); Open-Meteo weather (`weather.py`) | LiteAPI `prod_` key; hotel affiliate links (join Agoda in Travelpayouts); per-station metro times; a timetable; Open-Meteo commercial plan; catalogs built around each venue, not only the city centre |
| **Evals** (`evals.py`): offline, every event × 3 origins, code-only (1,990 cases, 0 failures); live quick and full sets (edits, Hinglish, attack prompts), with free-tier quota exhaustion reported as "skipped" | Run the full live set on a paid tier; grow it from `ml.planner_log` |
| Streamlit playground on the new API, with clan members, votes and decisions | React screens (stream G) |

---

## 19. Build order (everything ships at launch; this is only the order)

Work that other work depends on comes first. Streams A–C can run in parallel.

| # | Workstream | Contents | Depends on | Size |
|---|---|---|---|---|
| A | **Data** | Catalog for all event cities, price bands, LiteAPI and Travelpayouts clients, train timetable and fares, last-metro table, OSRM deployment, Open-Meteo | none | L |
| B | **Core model and validator** | New Trip and Item model with times, days and IDs; §6 validator; tools from §5; draft pipeline §4.1 with code-only fallback | none | L |
| C | **Storage and async** | `ml` schema, Postgres queue, locks and limits, workers, SSE, full API from §13 | none | M |
| D | **Groups** | Members, per-member travel, variants, votes, locks, cost split; Spring settle-up and clan-chat relay | B, C | L |
| E | **Booking** | `links.py`, affiliate sign-ups, click redirect, disclosure, extracting details from pasted confirmations | A | M |
| F | **Language and voice** | Language-matching instructions, speech-to-text endpoint, Hinglish eval set | B | S |
| G | **Frontend** | All §17 components; Trip mode and reminders (Spring notifications) | C (API contract only), then D | L |
| H | **Evals and ops** | §15 suite in CI, dashboards, cost logging, prompt injection guard | B | M |

**Launch gate:** all hard eval targets met, p95 time to first draft under 20 s, p95 chat turn under 25 s, and cost per trip under ₹5.

---

## 20. Team decisions (confirmed 30 Sep 2026)

| # | Decision | Where it's applied |
|---|---|---|
| 1 | Trip data lives in **PostgreSQL**, in the `ml` schema | §2, §14 |
| 2 | LLM: the **Gemini free tier** in production, with fallbacks, pacing and a consent notice | §11, §12 |
| 3 | Trip chats are **kept indefinitely**; erasure on request or account deletion still applies | §12 |
| 4 | **Any member can pick**; a strict majority passes a decision early, otherwise the option with most votes wins after 24 h | §1.4, §9, §13, §14 |
| 5 | Hand-off links: **BookMyShow or District** for event tickets, **Booking.com** (with Agoda) for hotels | §8 |
| 6 | **No in-app booking** | §0 |

---

## Sources (research, Sep 2026)
- Mindtrip Flights (Sabre + PayPal, 6 May 2026): https://skift.com/2026/05/06/sabre-mindtrip-paypal-launch-agentic-ai-travel-booking/ · Stays (Jul 2026): https://www.prnewswire.com/news-releases/mindtrip-launches-mindtrip-stays-bringing-agentic-ai-to-hotel-search-and-booking-302826228.html · Events (Nov 2025): https://www.prnewswire.com/news-releases/mindtrip-expands-beyond-travel-into-everyday-experiences-with-new-events-feature-helping-users-discover-local-happenings-wherever-they-are-302606680.html
- Expedia acquires Layla (31 Jul 2026): https://skift.com/2026/07/31/expedia-acquired-ai-trip-planner-layla-exclusive/ · Layla reviews: https://www.trustpilot.com/review/layla.ai
- Wanderlog collaboration, budget and expense splitting: https://wanderlog.com/travel-budget-expense-splitting-app · reviews: https://tripstone.app/blog/wanderlog-review
- Winnie: https://trywinnie.com/features · https://trywinnie.com/ai-trip-planner-india
- MakeMyTrip Myra 2.0 (May 2026): https://www.fonearena.com/blog/482670/makemytrip-myra-ai-2-0-features.html · ixigo TARA: https://www.thehansindia.com/business/ixigo-reimagines-travel-with-a-fully-ai-native-app-reinvents-tara-as-a-multimodal-ai-assistant-and-introduces-agentic-travel-flows-1075258 · ConfirmTkt AI Seat Finder: https://analyticsindiamag.com/ai-news/irctc-approved-travel-app-confirmtkt-launches-ai-agent-to-help-book-trains
- Travelpayouts Aviasales terms: https://www.travelpayouts.com/en/offers/aviasales-affiliate-program/ · Real-time API needs 50k MAU: https://support.travelpayouts.com/hc/en-us/articles/30565016140434 · Hotellook closed: https://support.travelpayouts.com/hc/en-us/articles/29534131568530
- Amadeus Self-Service shutdown: https://www.phocuswire.com/amadeus-shut-down-self-service-apis-portal-developers
- IRCTC B2C PSP policy: https://contents.irctc.co.in/en/B2C_PSP_Policy_other_than_Startup_MSME.pdf
- Agoda partners: https://partners.agoda.com/en-us/faq.html · redBus on Cuelinks: https://www.cuelinks.com/campaigns/red-bus-affiliate-program · Viator API access: https://partnerresources.viator.com/travel-commerce/levels-of-access/
- RBI payment aggregator directions (Sep 2025): https://sarafpartners.com/rbi-issues-the-reserve-bank-of-india-regulation-of-payment-aggregators-directions-2025/
