"""Trip planner API (design §13). Called by Spring only: Spring authenticates the user and passes user_id.

Slow work (drafts, chat turns) is queued and runs in workers; clients follow it on the SSE stream.
Card actions (add / move / remove / undo) are quick, run here, and go through the same validator.
"""
import json
import time
from urllib.parse import urlparse

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

from common.utils import load
from trip_planner import fares, groups, links, store, stt, worker
from trip_planner.catalog import load_catalog
from trip_planner.reminders import reminders
from trip_planner.model import DEFAULT_PREFS, PLACE_SLOTS, SLOTS, costs, km_to_venue, new_trip, validate
from trip_planner.tools import PREF_KEYS, Tools

router = APIRouter(prefix="/trips", tags=["trip planner"])
MAX_TRIP_MESSAGES = 20   # per member per trip (cost cap)
MAX_DAILY_MESSAGES = 60  # per user per day, all trips
KEEPALIVE_S, POLL_S = 15, 0.5


class MemberIn(BaseModel):
    user_id: int
    origin_city: str
    first_name: str = ""


class NewTrip(BaseModel):
    event_id: int
    user_id: int
    origin_city: str
    first_name: str = ""
    clan_id: int | None = None
    members: list[MemberIn] = Field(default_factory=list, description="clan trips: the other members")
    prefs: dict = Field(default_factory=dict)


class MemberAdd(BaseModel):
    user_id: int  # the member making the change (Spring has checked the clan's rules)
    member: MemberIn


class OriginPatch(BaseModel):
    user_id: int
    origin_city: str


class ItemVote(BaseModel):
    user_id: int
    value: int = Field(ge=-1, le=1, description="+1, -1, or 0 to clear")


class DecisionIn(BaseModel):
    user_id: int
    kind: str
    payload: dict = Field(default_factory=dict)


class DecisionVote(BaseModel):
    user_id: int
    option: str


class MessageIn(BaseModel):
    user_id: int
    text: str = Field(min_length=1, max_length=2000)


class ItemIn(BaseModel):
    user_id: int
    if_version: int
    slot: str
    place_id: str
    day: int | None = None
    note: str = ""


class ItemPatch(BaseModel):
    user_id: int
    if_version: int
    slot: str | None = None
    day: int | None = None
    start: str | None = None


class UserVersion(BaseModel):
    user_id: int
    if_version: int | None = None


def db():
    try:
        store.pool()
    except store.StoreUnavailable as e:
        raise HTTPException(503, str(e))


def get_trip(trip_id: str, user_id: int):
    db()
    trip = store.load_trip(trip_id)
    if trip is None:
        raise HTTPException(404, "unknown trip")
    if not store.is_member(trip_id, user_id):
        raise HTTPException(403, "not a member of this trip")
    return trip


def view(trip) -> dict:
    """Everything the trip screen renders: timeline, items as cards, map points, costs, warnings."""
    cat = load_catalog(trip.city)
    issues = validate(trip, cat)
    cards = []
    for i in sorted(trip.items, key=lambda i: (i["start"] or i["end"] or "")):
        row = cat.get(i["ref"]["id"]) if i["slot"] in PLACE_SLOTS else None
        i = {**i, "book": [{**b, "tracked_url": links.tracked(b, trip.trip_id, i["item_id"])} for b in i["book"]]}
        card = {**i, "slot_label": SLOTS[i["slot"]], "locked": i["slot"] in trip.locks,
                "votes": trip.votes.get(i["item_id"], {"up": 0, "down": 0, "by": {}}),
                "warnings": [w["message"] for w in issues if w["item_id"] == i["item_id"]]}
        if row:
            card["place"] = {"name": row["name"], "kind": row["kind"], "lat": row["lat"], "lng": row["lng"],
                             "stars": int(row["stars"]) if row["stars"] else None, "cuisine": row["cuisine"],
                             "veg": bool(row["veg"]), "opening_hours": row["opening_hours"] or "unknown: check hours",
                             "km_to_venue": round(km_to_venue(trip, row), 2), "maps_url": links.maps(row),
                             "guest_rating": row.get("rating"), "reviews": row.get("reviews"),
                             "photo": row.get("photo"), "source": row.get("source") or "openstreetmap"}
        cards.append(card)
    tl = trip.timeline
    ev = trip.event
    return {"trip_id": trip.trip_id, "version": trip.version, "clan_id": trip.clan_id,
            "event": {k: ev.get(k) for k in ("event_id", "title", "city", "venue_name", "venue_lat", "venue_lng",
                                             "start_time", "gates_open", "end_time")},
            "timeline": {k: (f"{v:%Y-%m-%d %H:%M}" if hasattr(v, "strftime") else v) for k, v in tl.items()},
            "members": trip.members, "preferences": trip.prefs, "locks": trip.locks, "items": cards,
            "map": [{"item_id": c["item_id"], "slot": c["slot"], "name": c["place"]["name"],
                     "lat": c["place"]["lat"], "lng": c["place"]["lng"]} for c in cards if "place" in c]
                   + [{"item_id": None, "slot": "venue", "name": ev["venue_name"], "lat": ev["venue_lat"],
                       "lng": ev["venue_lng"]}],
            "costs": {**costs(trip), "per_member": groups.member_costs(trip)},
            "variants": {b: {"stay": v["stay"], "cost_per_person": v["cost_per_person"]} for b, v in trip.variants.items()},
            "decisions": [decision_view(d) for d in trip.decisions],
            "warnings": [w["message"] for w in issues if w["level"] == "soft" and not w["item_id"]],
            "booking_disclosure": links.DISCLOSURE,
            "pending_jobs": store.pending(trip.trip_id)}


def decision_view(d) -> dict:
    return {k: d[k] for k in ("decision_id", "kind", "payload", "options", "question", "opened_by", "deadline",
                              "status", "result", "tally", "votes")}


@router.post("", status_code=202)
def create(body: NewTrip):
    db()
    events = load("events")
    hit = events[events.event_id == body.event_id]
    if hit.empty:
        raise HTTPException(404, "unknown event")
    event = json.loads(hit.iloc[[0]].to_json(orient="records"))[0]  # plain JSON types for storage
    try:
        load_catalog(event["city"])
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    unknown = set(body.prefs) - PREF_KEYS
    if unknown:
        raise HTTPException(422, f"unknown preference(s): {sorted(unknown)}")
    members = [{"user_id": body.user_id, "first_name": body.first_name, "origin_city": body.origin_city,
                "role": "creator"}]
    members += [{**m.model_dump(), "role": "member"} for m in body.members if m.user_id != body.user_id]
    trip = new_trip(event, members, {**DEFAULT_PREFS, **body.prefs}, clan_id=body.clan_id, created_by=body.user_id)
    store.create_trip(trip)
    job_id = store.enqueue(trip.trip_id, "draft")
    return {"trip_id": trip.trip_id, "job_id": job_id, "trip": view(trip)}


@router.get("/{trip_id}")
def show(trip_id: str, user_id: int):
    return view(get_trip(trip_id, user_id))


@router.get("/{trip_id}/versions/{v}")
def show_version(trip_id: str, v: int, user_id: int):
    get_trip(trip_id, user_id)
    row = store.version(trip_id, v)
    if not row:
        raise HTTPException(404, "unknown version")
    return row


@router.post("/{trip_id}/voice")
def voice(trip_id: str, user_id: int = Form(...), audio: UploadFile = File(...)):
    """Speech → text. The app shows the text for a quick check, then sends it with POST /messages."""
    import openai
    get_trip(trip_id, user_id)
    if audio.content_type not in stt.TYPES:
        raise HTTPException(400, f"unsupported audio type {audio.content_type}")
    data = audio.file.read(stt.MAX_BYTES + 1)
    if len(data) > stt.MAX_BYTES:
        raise HTTPException(413, "audio longer than about 5 minutes")
    try:
        out = stt.transcribe(audio.filename, data)
    except stt.NotConfigured as e:
        raise HTTPException(503, str(e))
    except openai.RateLimitError:
        raise HTTPException(429, "voice service is busy, retry shortly")
    except (openai.APIConnectionError, openai.APIStatusError) as e:
        raise HTTPException(502, f"voice service error: {type(e).__name__}")
    if not out["text"]:
        raise HTTPException(422, "couldn't hear any words; try again closer to the mic")
    return out


@router.get("/{trip_id}/messages")
def chat_history(trip_id: str, user_id: int, after: int = 0):
    get_trip(trip_id, user_id)
    return {"messages": store.messages(trip_id, after=after)}


@router.post("/{trip_id}/messages", status_code=202)
def send(trip_id: str, body: MessageIn):
    get_trip(trip_id, body.user_id)
    in_trip, today = store.message_counts(trip_id, body.user_id)
    if in_trip >= MAX_TRIP_MESSAGES:
        raise HTTPException(429, f"message limit reached for this trip ({MAX_TRIP_MESSAGES})")
    if today >= MAX_DAILY_MESSAGES:
        raise HTTPException(429, f"daily message limit reached ({MAX_DAILY_MESSAGES})")
    seq = store.add_message(trip_id, "user", body.user_id, body.text)
    job_id = store.enqueue(trip_id, "chat", {"text": body.text, "user_id": body.user_id, "seq": seq})
    return {"turn_id": job_id, "message_seq": seq, "queued_behind": len(store.pending(trip_id)) - 1}


@router.get("/{trip_id}/stream")
def stream(trip_id: str, user_id: int, after: int = 0, until_job: str | None = None,
           last_event_id: int | None = Header(default=None)):
    """Server-sent events: status, tool, message, trip_updated, done, error. Resume with Last-Event-ID."""
    get_trip(trip_id, user_id)
    start = max(after, last_event_id or 0)

    def gen():
        last, quiet = start, 0.0
        # ponytail: polls the events table; switch to LISTEN/NOTIFY + async when streams number in the hundreds
        while True:
            evs = store.events(trip_id, last)
            for e in evs:
                last = e["seq"]
                yield f"id: {e['seq']}\nevent: {e['type']}\ndata: {json.dumps({**e['data'], 'job_id': e['job_id']}, ensure_ascii=False)}\n\n"
                if until_job and e["job_id"] == until_job and e["type"] == "done":
                    return
            quiet = 0.0 if evs else quiet + POLL_S
            if quiet >= KEEPALIVE_S:
                quiet = 0.0
                yield ": keepalive\n\n"
            time.sleep(POLL_S)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def card_edit(trip_id: str, user_id: int, if_version, action):
    """Run one tool as the user, with the same validator; reject while a planner job is editing."""
    trip = get_trip(trip_id, user_id)
    if store.is_busy(trip_id):
        raise HTTPException(409, "the planner is updating this trip; retry in a moment")
    if if_version is not None and if_version != trip.version:
        raise HTTPException(409, f"stale version {if_version}: the trip is at v{trip.version}; refetch")
    start = trip.version
    result = action(Tools(trip, load_catalog(trip.city), author=f"user:{user_id}"))
    if "error" in result:
        raise HTTPException(409 if "locked" in result.get("codes", []) else 400, result["error"])
    fares.enrich(trip)  # new or re-dated travel legs get their fare range (cached)
    if not store.save_trip(trip, start):
        raise HTTPException(409, "the trip changed meanwhile; refetch")
    store.emit(trip_id, None, "trip_updated", {"version": trip.version, "by": user_id})
    return {"result": result, "trip": view(trip)}


@router.post("/{trip_id}/items")
def add_item(trip_id: str, body: ItemIn):
    return card_edit(trip_id, body.user_id, body.if_version,
                     lambda t: t.add_item(body.slot, body.place_id, day=body.day, note=body.note))


@router.patch("/{trip_id}/items/{item_id}")
def move_item(trip_id: str, item_id: str, body: ItemPatch):
    return card_edit(trip_id, body.user_id, body.if_version,
                     lambda t: t.move_item(item_id, slot=body.slot, day=body.day, start=body.start))


@router.delete("/{trip_id}/items/{item_id}")
def remove_item(trip_id: str, item_id: str, user_id: int, if_version: int):
    return card_edit(trip_id, user_id, if_version, lambda t: t.remove_item(item_id))


@router.post("/{trip_id}/undo")
def undo(trip_id: str, body: UserVersion):
    return card_edit(trip_id, body.user_id, body.if_version, lambda t: t.undo())


@router.get("/{trip_id}/timeline-events")
def timeline_events(trip_id: str, user_id: int):
    """Reminders for Spring to schedule (re-fetch after trip_updated): IST times, per member or everyone."""
    trip = get_trip(trip_id, user_id)
    return {"version": trip.version, "reminders": reminders(trip, load_catalog(trip.city))}


@router.get("/{trip_id}/costs")
def trip_costs(trip_id: str, user_id: int):
    trip = get_trip(trip_id, user_id)
    return {**costs(trip), "per_member": groups.member_costs(trip)}


# ---- members (Spring enforces the clan's admin rules; any member may call these) ---------------------
@router.post("/{trip_id}/members")
def add_member(trip_id: str, body: MemberAdd):
    m = body.member
    return card_edit(trip_id, body.user_id, None, lambda t: t.add_member(m.user_id, m.origin_city, m.first_name))


@router.patch("/{trip_id}/members/{member_id}")
def set_origin(trip_id: str, member_id: int, body: OriginPatch):
    if body.user_id != member_id:
        raise HTTPException(403, "only that member can change their own origin city")
    return card_edit(trip_id, body.user_id, None, lambda t: t.set_origin(member_id, body.origin_city))


@router.delete("/{trip_id}/members/{member_id}")
def remove_member(trip_id: str, member_id: int, user_id: int):
    return card_edit(trip_id, user_id, None, lambda t: t.remove_member(member_id))


# ---- votes on items and group decisions -----------------------------------------------------------------
@router.post("/{trip_id}/items/{item_id}/vote")
def vote_item(trip_id: str, item_id: str, body: ItemVote):
    trip = get_trip(trip_id, body.user_id)
    if not trip.find(item_id):
        raise HTTPException(404, "unknown item")
    tally = store.vote_item(trip_id, item_id, body.user_id, body.value)
    store.emit(trip_id, None, "votes_updated", {"item_id": item_id, "up": tally["up"], "down": tally["down"]})
    return tally


@router.post("/{trip_id}/decisions", status_code=201)
def open_decision(trip_id: str, body: DecisionIn):
    """Anyone can open one. The opener's vote counts as 'yes'. Spring posts the poll card in clan chat."""
    trip = get_trip(trip_id, body.user_id)
    cat = load_catalog(trip.city)
    err = groups.check_proposal(trip, cat, body.kind, body.payload)
    if err:
        raise HTTPException(400, err)
    return decision_view(store.decision(worker.open_decision(trip, cat, body.kind, body.payload, body.user_id)))


@router.post("/{trip_id}/decisions/{decision_id}/vote")
def vote_decision(trip_id: str, decision_id: str, body: DecisionVote):
    trip = get_trip(trip_id, body.user_id)
    d = store.decision(decision_id)
    if not d or d["trip_id"] != trip_id:
        raise HTTPException(404, "unknown decision")
    if d["status"] != "open":
        raise HTTPException(409, f"decision is already {d['status']}")
    if body.option not in d["options"]:
        raise HTTPException(400, f"option must be one of {d['options']}")
    store.vote_decision(decision_id, body.user_id, body.option)
    d = store.decision(decision_id)
    store.emit(trip_id, None, "decision_voted", {"decision_id": decision_id, "tally": d["tally"]})
    worker.settle(d, len(trip.members))
    return decision_view(store.decision(decision_id))


@router.get("/{trip_id}/decisions")
def list_decisions(trip_id: str, user_id: int):
    get_trip(trip_id, user_id)
    return {"decisions": [decision_view(d) for d in store.decisions(trip_id)]}


@router.get("/users/{user_id}/trips")
def user_trips(user_id: int):
    db()
    return {"trips": store.trips_for_user(user_id)}


@router.delete("/users/{user_id}")
def erase(user_id: int):
    """Account deletion / DPDP erasure request: chats are otherwise kept forever."""
    db()
    return store.erase_user(user_id)


# ---- hand-off click redirect (mounted at /v1/go) ---------------------------------------------------------
go_router = APIRouter(tags=["trip planner"])


@go_router.get("/go")
def go(url: str, trip_id: str = "", item_id: str = "", provider: str = "", user_id: int | None = None):
    """Log the click, then 302 to the partner. Only known booking hosts: never an open redirect."""
    if not links.allowed(url):
        raise HTTPException(400, "not a known booking link")
    try:
        store.log_click(trip_id or None, item_id or None, user_id, provider or None, urlparse(url).hostname)
    except store.StoreUnavailable:
        pass  # the user still gets to the partner
    return RedirectResponse(url, status_code=302)
