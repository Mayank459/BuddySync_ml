"""Trip model, show-anchored timeline, price estimates, costs and the validator.

Pure code, no I/O (TRIP_PLANNER_DESIGN.md §3, §6, §7). A trip is plain JSON-able data, so the
store saves it as it is. Code owns times, prices and rules; the LLM only chooses places and explains.
"""
import copy
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import opening_hours as oh
import pandas as pd

from common.utils import haversine_km
from trip_planner import links
from trip_planner.transit import LAST_METRO

SLOTS = {
    "travel_in": "getting to the event city (one per member)",
    "stay": "where to sleep (one hotel)",
    "before_show": "food or a short visit between check-in and gates",
    "show": "the event itself",
    "after_show": "food after the show",
    "return": "getting back to the hotel after the show",
    "explore": "places worth visiting on the day(s) after the show",
    "travel_out": "getting home (one per member)",
}
PLACE_SLOTS = ("stay", "before_show", "after_show", "explore")  # filled from the catalog
HOTEL_KINDS = {"hotel", "guest_house", "hostel"}
FOOD_KINDS = {"restaurant", "cafe"}
SIGHT_KINDS = {"attraction", "museum", "viewpoint", "gallery", "zoo", "theme_park",
               "monument", "fort", "castle", "memorial", "park"}
BANDS = {  # ponytail: hand-set bands; retune once real booking data exists
    "budget": {"stars": (None, 3), "kinds": {"hotel", "guest_house", "hostel"}, "cap_inr": 6000},
    "balanced": {"stars": (3, 4), "kinds": {"hotel", "guest_house"}, "cap_inr": 15000},
    "premium": {"stars": (4, None), "kinds": {"hotel"}, "cap_inr": None},
}
DEFAULT_PREFS = {"budget_band": "balanced", "hotel_min_stars": None, "max_distance_km": 5.0, "veg_only": False,
                 "interests": [], "travelling_solo": False, "extra_days": 0, "rooms": None, "lang": "en"}
MAX_EXTRA_DAYS = 4

# ---- prices (ESTIMATE bands, INR) -----------------------------------------------------------
# ponytail: static bands per star/kind; replace hotel bands with LiteAPI rates (VERIFIED) when connected
PRICE_SOURCE = "BuddySync estimate bands (Sep 2026)"
TIER1 = {"Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Pune"}
HOTEL_NIGHT = {None: (1500, 4000), 1: (800, 1500), 2: (1200, 2500), 3: (2500, 5000), 4: (5000, 9000), 5: (9000, 20000)}
KIND_NIGHT = {"hostel": (500, 1200), "guest_house": (1000, 2500)}
PER_PERSON = {"cafe": (150, 400), "restaurant": (300, 900), "museum": (50, 500), "gallery": (0, 300),
              "fort": (25, 600), "monument": (25, 600), "castle": (25, 600), "memorial": (0, 100),
              "park": (0, 100), "viewpoint": (0, 0), "zoo": (100, 500), "theme_park": (800, 2500),
              "attraction": (0, 500)}
CAB_BASE, CAB_PER_KM = 50, (18, 30)


def _price(lo, hi, unit, tier2=False):
    if tier2:
        lo, hi = lo * 0.7, hi * 0.7
    r = lambda x: int(round(x / 50.0) * 50) if x >= 100 else int(round(x / 10.0) * 10)
    return {"min": r(lo), "max": r(hi), "currency": "INR", "unit": unit, "confidence": "estimate",
            "source": PRICE_SOURCE, "fetched_at": None}


def price_of(row: dict, city: str):
    tier2 = city not in TIER1
    if row["kind"] in HOTEL_KINDS:
        stars = int(row["stars"]) if row.get("stars") else None
        lo, hi = KIND_NIGHT.get(row["kind"]) if row["kind"] in KIND_NIGHT and not stars else HOTEL_NIGHT[stars]
        return _price(lo, hi, "night", tier2)
    if row["kind"] in PER_PERSON:
        return _price(*PER_PERSON[row["kind"]], "person", tier2)
    return None


def cab_price(km: float):
    return _price(CAB_BASE + km * CAB_PER_KM[0], CAB_BASE + km * CAB_PER_KM[1], "trip")


def travel_min(km: float) -> int:
    """Door-to-door minutes for a straight-line distance. ponytail: haversine × road factor; OSRM later."""
    road = km * 1.3
    return int(road / 4.5 * 60) if road <= 1.5 else int(road / 20 * 60) + 10


# ---- timeline ----------------------------------------------------------------------------------
def _ts(x) -> datetime:
    return pd.Timestamp(x).to_pydatetime().replace(tzinfo=None)


def timeline(event: dict, prefs: dict) -> dict:
    gates, start, end = _ts(event["gates_open"]), _ts(event["start_time"]), _ts(event["end_time"])
    day = gates.replace(hour=0, minute=0, second=0, microsecond=0)
    check_out = day + timedelta(days=1 + prefs["extra_days"], hours=11)
    return {"arrive_by": gates - timedelta(hours=3), "check_in": day + timedelta(hours=14), "gates_open": gates,
            "show_start": start, "show_end": end, "late_finish": end.hour >= 23 or end.hour < 5,
            "check_out": check_out, "depart_after": check_out.replace(hour=18),  # after the last explore window
            "show_day": day, "days": 1 + prefs["extra_days"]}


def windows(tl: dict) -> dict:
    """(start, end) per slot; explore has one window per day after the show (keys 'explore:1', ...)."""
    w = {"before_show": (max(tl["check_in"], tl["gates_open"] - timedelta(hours=2, minutes=30)),
                         tl["gates_open"] - timedelta(minutes=15)),
         "after_show": (tl["show_end"], tl["show_end"] + timedelta(minutes=90))}
    for d in range(1, tl["days"] + 1):
        base = tl["show_day"] + timedelta(days=d)
        last = d == tl["days"]  # check-out day: explore after leaving the hotel
        w[f"explore:{d}"] = (base + timedelta(hours=11, minutes=30 if last else -60),
                             base + timedelta(hours=17, minutes=30 if last else 60))
    return w


def fmt(t) -> str | None:
    return t.strftime("%Y-%m-%d %H:%M") if t else None


# ---- items and trip ------------------------------------------------------------------------------
def new_item(slot, kind, ref_id, title, *, day=0, why="", note="", added_by="planner", member_id=None,
             start=None, end=None, fixed=False, price=None, book=None) -> dict:
    return {"item_id": uuid.uuid4().hex[:10], "slot": slot, "day": day, "start": fmt(start), "end": fmt(end),
            "fixed": fixed, "member_id": member_id, "ref": {"kind": kind, "id": ref_id}, "title": title,
            "why": why, "note": note, "price": price, "book": book or [], "added_by": added_by,
            "booking_status": "suggested"}


@dataclass
class Trip:
    event: dict
    members: list  # [{user_id, first_name, origin_city, role}]
    prefs: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_PREFS))
    items: list = field(default_factory=list)
    locks: dict = field(default_factory=dict)  # slot -> who/what locked it
    memory: str = ""
    memory_upto: int = 0  # last message seq folded into memory
    trip_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    clan_id: int | None = None
    created_by: int | None = None
    version: int = -1
    parent: int | None = None
    history: list = field(default_factory=list)  # recent versions, oldest first
    saved_version: int = -1  # store: versions above this are not saved yet
    variants: dict = field(default_factory=dict)  # clans: budget/balanced/premium proposals (groups.make_variants)
    votes: dict = field(default_factory=dict)  # read-only here: item_id -> {"up", "down"} (store.trip_votes)
    decisions: list = field(default_factory=list)  # read-only here: open group decisions

    @property
    def city(self):
        return self.event["city"]

    @property
    def timeline(self):
        return timeline(self.event, self.prefs)

    def state(self) -> dict:
        return copy.deepcopy({"items": self.items, "prefs": self.prefs, "locks": self.locks})

    def restore(self, state: dict):
        s = copy.deepcopy(state)
        self.items, self.prefs, self.locks = s["items"], s["prefs"], s["locks"]

    def commit(self, author: str, reason: str, parent=None):
        self.parent = self.version if parent is None else parent
        self.version += 1
        self.history.append({"version": self.version, "parent": self.parent, "author": author,
                             "reason": reason, **self.state()})
        del self.history[:-50]  # ponytail: undo depth 50 in memory; the store keeps every version

    def find(self, item_id):
        return next((i for i in self.items if i["item_id"] == item_id), None)

    def in_slot(self, slot, day=None):
        return [i for i in self.items if i["slot"] == slot and (day is None or i["day"] == day)]

    def stay_row(self, cat):
        stay = self.in_slot("stay")
        return cat.get(stay[0]["ref"]["id"]) if stay else None


def new_trip(event: dict, members: list, prefs: dict | None = None, **kw) -> Trip:
    t = Trip(event=event, members=members, prefs={**copy.deepcopy(DEFAULT_PREFS), **(prefs or {})}, **kw)
    build_auto_items(t)
    t.commit("system", "created")
    return t


def build_auto_items(trip: Trip, cat=None):
    """Travel legs per member and the show itself: code-built from the timeline, never by the LLM."""
    tl, ev = trip.timeline, trip.event
    keep = [i for i in trip.items if i["slot"] not in ("travel_in", "show", "travel_out")]
    auto = [new_item("show", "event", ev.get("event_id"), f"{ev['title']} @ {ev['venue_name']}",
                     start=tl["gates_open"], end=tl["show_end"], note=f"gates {tl['gates_open']:%H:%M}")]
    for m in trip.members:
        origin = m.get("origin_city") or ""
        if not origin or origin.lower() == trip.city.lower():
            continue
        auto.append(new_item("travel_in", "route", f"{origin}->{trip.city}", f"{origin} → {trip.city}",
                             member_id=m["user_id"], end=tl["arrive_by"],
                             note=f"arrive by {tl['arrive_by']:%a %H:%M}",
                             book=links.transport(origin, trip.city, tl["arrive_by"].date())))
        auto.append(new_item("travel_out", "route", f"{trip.city}->{origin}", f"{trip.city} → {origin}",
                             day=tl["days"], member_id=m["user_id"], start=tl["depart_after"],
                             book=links.transport(trip.city, origin, tl["depart_after"].date())))
    # keep a leg's fetched fare and item_id while its route and date are unchanged (fares.enrich fills new ones)
    old = {(i["slot"], i["member_id"], i["ref"]["id"], i["start"], i["end"]): i for i in trip.items
           if i["slot"] in ("travel_in", "travel_out")}
    for n, i in enumerate(auto):
        prev = old.get((i["slot"], i["member_id"], i["ref"]["id"], i["start"], i["end"]))
        if prev:
            auto[n] = prev
    trip.items = auto + keep
    if cat is not None:
        schedule(trip, cat)


def km_to_venue(trip: Trip, row: dict) -> float:
    return haversine_km(trip.event["venue_lat"], trip.event["venue_lng"], row["lat"], row["lng"])


def schedule(trip: Trip, cat):
    """Give every non-fixed item a time inside its slot window, and rebuild the return-to-hotel leg."""
    tl, w = trip.timeline, windows(trip.timeline)
    for i in trip.items:
        if i["slot"] == "stay":
            i["start"], i["end"] = fmt(tl["check_in"]), fmt(tl["check_out"])
        elif i["slot"] == "explore":
            i["day"] = min(max(i["day"], 1), tl["days"])
    for key, (ws, we) in w.items():
        slot, _, day = key.partition(":")
        _spread([i for i in trip.in_slot(slot, int(day) if day else None) if not i["fixed"]], ws, we, cat)
    trip.items = [i for i in trip.items if i["slot"] != "return"]
    hotel = trip.stay_row(cat)
    if hotel:
        km = km_to_venue(trip, hotel)
        walk = km <= 1.2 and not tl["late_finish"] and not trip.prefs["travelling_solo"]
        start = tl["show_end"]
        trip.items.append(new_item(
            "return", "route", f"venue->{hotel['place_id']}",
            f"{'Walk' if walk else 'Cab'} back to {hotel['name']} ({km:.1f} km, ~{travel_min(km)} min)",
            start=start, end=start + timedelta(minutes=travel_min(km)),
            price=None if walk else cab_price(km * 1.3),
            book=[] if walk else links.cab(trip.event, hotel)))


def _spread(items, ws, we, cat):
    if not items:
        return
    rows = [cat.get(i["ref"]["id"]) for i in items]
    gaps = [travel_min(haversine_km(a["lat"], a["lng"], b["lat"], b["lng"])) if a and b else 15
            for a, b in zip(rows, rows[1:])]
    total = (we - ws).total_seconds() / 60
    each = max(20.0, (total - sum(gaps)) / len(items))
    t = ws
    for n, i in enumerate(items):
        i["start"], i["end"] = fmt(t), fmt(t + timedelta(minutes=each))
        t += timedelta(minutes=each + (gaps[n] if n < len(gaps) else 0))


# ---- opening hours ------------------------------------------------------------------------------
def hours_check(row: dict, start: datetime, end: datetime):
    """None if fine or unknown; ('closed', msg) if closed the whole time; ('closes', msg) if open only part of it."""
    expr = row.get("opening_hours")
    if not expr:
        return None
    try:
        kw = {"coords": (row["lat"], row["lng"])} if any(s in expr for s in ("sun", "dawn", "dusk")) else {}
        o = oh.OpeningHours(expr, **kw)
        spans = [(a.replace(tzinfo=None), b.replace(tzinfo=None), s) for a, b, s, _ in o.intervals(start, end)]
    except Exception:
        return None  # unparseable OSM value: the view shows "check hours"
    open_spans = [(a, b) for a, b, s in spans if s == oh.State.OPEN]
    if not open_spans and all(s == oh.State.CLOSED for _, _, s in spans):
        return "closed", f"{row['name']} is closed {start:%a %H:%M}–{end:%H:%M} (OSM hours: {expr})"
    if open_spans and (open_spans[0][0] > start or open_spans[-1][1] < end):
        return "closes", f"{row['name']} is open only {open_spans[0][0]:%H:%M}–{open_spans[-1][1]:%H:%M} of that slot"
    return None


# ---- costs ----------------------------------------------------------------------------------------
def costs(trip: Trip) -> dict:
    tl = trip.timeline
    n = max(1, len(trip.members))
    rooms = trip.prefs.get("rooms") or math.ceil(n / 2)
    nights = max(1, (tl["check_out"].date() - tl["check_in"].date()).days)
    lines, lo, hi = [], 0, 0
    for i in trip.items:
        p = i["price"]
        if not p:
            continue
        if i.get("member_id"):  # one member's own travel leg, averaged over the group (groups.member_costs splits it)
            f = 1 / n
        elif p["unit"] == "night":
            f = nights * rooms / n
        elif p["unit"] == "trip":  # shared cab: one per 4 people
            f = math.ceil(n / 4) / n
        else:
            f = 1
        a, b = int(p["min"] * f), int(p["max"] * f)
        lo, hi = lo + a, hi + b
        lines.append({"item_id": i["item_id"], "title": i["title"], "min": a, "max": b, "confidence": p["confidence"]})
    not_included = sorted({"train / bus / flight tickets (see Book links)" for i in trip.items
                           if i["slot"] in ("travel_in", "travel_out") and not i["price"]}
                          | {"event ticket"})
    return {"per_person": {"min": lo, "max": hi, "currency": "INR"}, "lines": lines, "members": n,
            "rooms": rooms, "nights": nights, "not_included": not_included}


# ---- validator (§6) --------------------------------------------------------------------------------
def validate(trip: Trip, cat, before: dict | None = None) -> list[dict]:
    """Every rule the plan must meet. 'hard' blocks an edit; 'soft' is shown to the user."""
    out, p, tl = [], trip.prefs, trip.timeline

    def add(level, code, msg, item=None):
        out.append({"level": level, "code": code, "message": msg, "item_id": item and item["item_id"]})

    seen = set()
    for i in trip.items:
        if i["slot"] not in PLACE_SLOTS:
            continue
        row = cat.get(i["ref"]["id"])
        if row is None:
            add("hard", "unknown_place", f"unknown place_id {i['ref']['id']}: only use IDs from search results", i)
            continue
        if (i["slot"] == "stay") != (row["kind"] in HOTEL_KINDS):
            add("hard", "wrong_slot", "hotels go in 'stay' and only hotels go there", i)
        key = (i["slot"], i["day"], i["ref"]["id"])
        if key in seen:
            add("hard", "duplicate", f"{row['name']} is already in {i['slot']}", i)
        seen.add(key)
        if i["start"] and i["slot"] != "stay":
            hc = hours_check(row, _ts(i["start"]), _ts(i["end"]))
            if hc:
                add("hard" if hc[0] == "closed" else "soft", f"hours_{hc[0]}", hc[1], i)
        if p["veg_only"] and row["kind"] in FOOD_KINDS and not row.get("veg"):
            add("soft", "not_veg", f"{row['name']} is not tagged vegetarian: confirm before going", i)
    if len(trip.in_slot("stay")) > 1:
        add("hard", "two_stays", "only one hotel per trip")
    for i in trip.items:  # a fare quote, not a booking: the user picks the actual flight, so these are soft
        f = i.get("flight") if i.get("mode") == "flight" else None
        if f and i["slot"] == "travel_in" and i["end"] and f["arrives"] > i["end"]:
            add("soft", "flight_late", f"the cheapest fare ({f['flight_number']}) lands {f['arrives'][5:]}, after "
                                       f"the arrive-by time {i['end'][5:]}: choose an earlier flight", i)
        if f and i["slot"] == "travel_in" and i["end"] and f["arrives"][:10] < i["end"][:10]:
            add("soft", "flight_day_early", f"the cheapest fare seen is {f['flight_number']} on {f['departs'][5:10]}, "
                                            "a day or more early: add a night or look for a same-day flight", i)
        if f and i["slot"] == "travel_out" and i["start"] and f["departs"] < i["start"]:
            add("soft", "flight_early", f"the cheapest fare ({f['flight_number']}) leaves {f['departs'][5:]}, "
                                        f"before the trip ends ({i['start'][5:]}): choose a later flight", i)
    if before is not None:
        for slot in trip.locks:
            ids = lambda items: sorted((i["ref"]["id"], i["day"]) for i in items if i["slot"] == slot)
            if ids(before["items"]) != ids(trip.items):
                add("hard", "locked", f"'{slot}' is locked by a group decision; start a new decision to change it")

    hotel = trip.stay_row(cat)
    if hotel:
        km = km_to_venue(trip, hotel)
        if p["hotel_min_stars"] and (not hotel.get("stars") or hotel["stars"] < p["hotel_min_stars"]):
            rating = f"{int(hotel['stars'])}-star" if hotel.get("stars") else "unrated"
            add("soft", "stars", f"{hotel['name']} is {rating}, below the {p['hotel_min_stars']}-star preference")
        if km > p["max_distance_km"]:
            add("soft", "far_hotel", f"{hotel['name']} is {km:.1f} km from the venue (limit {p['max_distance_km']} km)")
        if tl["late_finish"] and km > 3:
            add("soft", "late_far", f"show ends {tl['show_end']:%H:%M} and {hotel['name']} is {km:.1f} km away: "
                                    "pre-book a cab or pick a closer hotel")
        if p["travelling_solo"] and km > 1:
            add("soft", "solo_walk", f"travelling solo: take a cab for the {km:.1f} km back to {hotel['name']}")
        metro = LAST_METRO.get(trip.city)
        if metro and km > 1.2:
            last = datetime.combine(tl["show_day"].date(), metro["last"])
            if tl["show_end"] + timedelta(minutes=30) > last:
                add("soft", "last_metro", f"show ends {tl['show_end']:%H:%M} but {trip.city}'s last metro is around "
                                          f"{metro['last']:%H:%M} ({metro['note']}): plan a cab back, and check for "
                                          "extended event-night service")

    for key, (ws, we) in windows(tl).items():
        slot, _, day = key.partition(":")
        items = [i for i in trip.in_slot(slot, int(day) if day else None) if not i["fixed"]]
        rows = [cat.get(i["ref"]["id"]) for i in items]
        if len(items) > 1 and all(rows):
            gaps = sum(travel_min(haversine_km(a["lat"], a["lng"], b["lat"], b["lng"])) for a, b in zip(rows, rows[1:]))
            if ((we - ws).total_seconds() / 60 - gaps) / len(items) < 30:
                add("soft", "too_tight", f"too much in {key} ({len(items)} stops, {gaps} min travelling): drop one")

    cap = BANDS[p["budget_band"]]["cap_inr"]
    c = costs(trip)
    if cap and c["per_person"]["max"] > cap:
        add("soft", "budget", f"could cost up to ₹{c['per_person']['max']:,} per person, above the "
                              f"{p['budget_band']} budget of ₹{cap:,} (estimate)")
    return out


def new_hard(after: list[dict], before: list[dict]) -> list[dict]:
    old = {(i["code"], i["item_id"]) for i in before if i["level"] == "hard"}
    return [i for i in after if i["level"] == "hard" and (i["code"], i["item_id"]) not in old]
