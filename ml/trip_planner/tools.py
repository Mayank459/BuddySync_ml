"""Planner tools (design §5): the only way the LLM, or a user's card action, can read or change a trip.

Every edit runs the validator on the result. New hard problems undo the edit and come back as an
error; soft problems come back as warnings for the planner to tell the user.
"""
import json
import re
from datetime import timedelta, timezone

import pandas as pd

from common.utils import haversine_km
from trip_planner import hotels, links
from trip_planner.model import (BANDS, DEFAULT_PREFS, FOOD_KINDS, HOTEL_KINDS, MAX_EXTRA_DAYS, PLACE_SLOTS, _ts,
                                build_auto_items, costs, hours_check, km_to_venue, new_hard, new_item, price_of,
                                schedule, validate, windows)

TOOL_SPECS = [
    {"name": "get_trip", "description": "Current event, timeline, members, preferences, items, locks, warnings and cost.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "update_preferences",
     "description": "Save lasting preferences. Send only fields that change; null clears a field.",
     "input_schema": {"type": "object", "properties": {
         "budget_band": {"type": "string", "enum": list(BANDS)},
         "hotel_min_stars": {"type": ["integer", "null"], "minimum": 1, "maximum": 5},
         "max_distance_km": {"type": "number", "description": "max hotel distance from the venue"},
         "veg_only": {"type": "boolean"},
         "interests": {"type": "array", "items": {"type": "string"}},
         "travelling_solo": {"type": "boolean"},
         "extra_days": {"type": "integer", "minimum": 0, "maximum": MAX_EXTRA_DAYS,
                        "description": "days to stay after the show"},
         "rooms": {"type": ["integer", "null"], "minimum": 1}}}},
    {"name": "search_hotels", "description": "Hotels near the venue with estimated nightly price. Defaults come from preferences.",
     "input_schema": {"type": "object", "properties": {
         "min_stars": {"type": "integer", "minimum": 1, "maximum": 5},
         "max_distance_km": {"type": "number"}, "limit": {"type": "integer", "maximum": 15}}}},
    {"name": "search_places",
     "description": "Places to eat or visit, nearest first, around the venue or the chosen hotel. Pass slot to skip places known to be closed then.",
     "input_schema": {"type": "object", "properties": {
         "kinds": {"type": "array", "items": {"type": "string"},
                   "description": "e.g. restaurant, cafe, museum, park, attraction, monument, viewpoint, fort"},
         "near": {"type": "string", "enum": ["venue", "hotel"]},
         "slot": {"type": "string", "enum": ["before_show", "after_show", "explore"]},
         "day": {"type": "integer", "minimum": 1, "description": "explore day (1 = day after the show)"},
         "max_distance_km": {"type": "number"}, "limit": {"type": "integer", "maximum": 15}}}},
    {"name": "get_transport_options", "description": "Travel legs per member with booking links and the Tatkal opening time.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "add_item", "description": "Put a searched place into a slot. Adding to 'stay' replaces the current hotel.",
     "input_schema": {"type": "object", "required": ["slot", "place_id"], "properties": {
         "slot": {"type": "string", "enum": list(PLACE_SLOTS)}, "place_id": {"type": "string"},
         "day": {"type": "integer", "minimum": 1, "description": "explore only: 1 = day after the show"},
         "why": {"type": "string", "description": "one short line: why this fits"}}}},
    {"name": "remove_item", "description": "Remove an item by item_id.",
     "input_schema": {"type": "object", "required": ["item_id"], "properties": {"item_id": {"type": "string"}}}},
    {"name": "move_item", "description": "Move an item to another slot, day or start time (HH:MM).",
     "input_schema": {"type": "object", "required": ["item_id"], "properties": {
         "item_id": {"type": "string"}, "slot": {"type": "string", "enum": list(PLACE_SLOTS)},
         "day": {"type": "integer", "minimum": 1}, "start": {"type": "string", "description": "HH:MM"}}}},
    {"name": "estimate_costs", "description": "Per-person cost range (INR estimates) and what is not included.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_weather", "description": "Daily forecast for the trip dates (up to 16 days ahead). If indoor_advised, prefer museums, galleries, cafes.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "undo", "description": "Revert the last change.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "start_decision",
     "description": "Group trips: open a vote for the whole group (majority wins; after 24 h the most votes win). "
                    "Use it to change a locked slot, or when members disagree. The person you are talking to opens it.",
     "input_schema": {"type": "object", "required": ["kind"], "properties": {
         "kind": {"type": "string", "enum": ["variant", "swap", "lock", "unlock", "pref"]},
         "item_id": {"type": "string", "description": "swap: the item to replace"},
         "place_id": {"type": "string", "description": "swap: the replacement, from search results"},
         "slot": {"type": "string", "enum": list(PLACE_SLOTS), "description": "lock / unlock"},
         "changes": {"type": "object", "description": "pref: preferences to change"}}}},
    {"name": "propose_variants",
     "description": "Group trips: build budget / balanced / premium versions of the plan and open a vote on them.",
     "input_schema": {"type": "object", "properties": {}}},
]
NAMES = {t["name"] for t in TOOL_SPECS}
LABELS = {"get_trip": "Reading the trip…", "update_preferences": "Saving your preferences…",
          "search_hotels": "Finding hotels near the venue…", "search_places": "Looking for places…",
          "get_transport_options": "Checking travel options…", "add_item": "Updating the plan…",
          "remove_item": "Updating the plan…", "move_item": "Updating the plan…",
          "estimate_costs": "Adding up costs…", "undo": "Undoing the last change…",
          "get_weather": "Checking the weather…",
          "start_decision": "Opening a group vote…", "propose_variants": "Building plan options for the group…"}
PREF_KEYS = set(DEFAULT_PREFS)
IST = timezone(timedelta(hours=5, minutes=30))
# single-gender hostels / PGs: never suggested, because we don't infer anyone's gender (design §10)
RESTRICTED = re.compile(r"\b(?:girls?|ladies|boys|women'?s?|gents|men'?s|pg)\b", re.I)


def price_text(p):
    if not p:
        return None
    tag = {"estimate": "est", "range": "range", "verified": "verified"}[p["confidence"]]
    if not p["max"]:
        return f"free ({tag})"
    if p["min"] == p["max"]:
        return f"from ₹{p['min']:,}/{p['unit']} ({tag})"
    return f"₹{p['min']:,}–{p['max']:,}/{p['unit']} ({tag})"


class Tools:
    def __init__(self, trip, cat, author="planner", batch=False, bypass_locks=False, actor_id=None):
        self.trip, self.cat, self.author = trip, cat, author
        self.batch = batch  # batch: the caller commits once (the first draft is one version, not ten)
        self.bypass_locks = bypass_locks  # only for applying a group decision
        self.actor_id = actor_id  # the member the planner is talking to (opens decisions)
        self.new_decisions = []  # (kind, payload, opened_by): the worker saves and announces these

    def run(self, name, args) -> str:
        fn = getattr(self, name) if name in NAMES else None
        try:
            result = fn(**(args or {})) if fn else {"error": f"unknown tool {name}"}
        except TypeError as e:  # bad arguments from the model
            result = {"error": str(e)}
        return json.dumps(result, default=str, ensure_ascii=False)

    # ---- reads ---------------------------------------------------------------
    def warnings(self):
        return [w["message"] for w in validate(self.trip, self.cat) if w["level"] == "soft"]

    def get_trip(self):
        t, tl = self.trip, self.trip.timeline
        items = []
        for i in sorted(t.items, key=lambda i: (i["start"] or i["end"] or "")):
            row = self.cat.get(i["ref"]["id"]) if i["slot"] in PLACE_SLOTS else None
            items.append({k: v for k, v in {
                "item_id": i["item_id"], "slot": i["slot"], "day": i["day"] or None, "start": i["start"],
                "title": row["name"] if row else i["title"], "place_id": row and row["place_id"],
                "kind": row and row["kind"], "km_to_venue": row and round(km_to_venue(t, row), 1),
                "price": price_text(i["price"]), "member_id": i["member_id"], "why": i["why"] or None,
                "votes": t.votes.get(i["item_id"])}.items() if v})
        c = costs(t)
        out = {"event": {k: t.event[k] for k in ("title", "city", "venue_name")},
               "timeline": {k: (v.strftime("%a %d %b %H:%M") if hasattr(v, "strftime") else v)
                            for k, v in tl.items() if k != "show_day"},
               "members": [{"user_id": m["user_id"], "name": m.get("first_name") or None,
                            "origin_city": m.get("origin_city")} for m in t.members],
               "talking_to": self.actor_id,
               "preferences": t.prefs, "locked_slots": list(t.locks), "items": items,
               "warnings": self.warnings(),
               "cost_per_person": f"₹{c['per_person']['min']:,}–{c['per_person']['max']:,} (estimate)"}
        if t.decisions:
            out["open_group_votes"] = [{"kind": d["kind"], "question": d["question"], "tally": d["tally"],
                                        "closes": f"{d['deadline'].astimezone(IST):%a %H:%M} IST"} for d in t.decisions]
        if t.variants:
            out["variants"] = {b: {"stay": v["stay"], "cost_per_person": v["cost_per_person"]}
                               for b, v in t.variants.items()}
        return out

    def _place_rows(self, df, anchor_row, limit):
        lat, lng = (anchor_row["lat"], anchor_row["lng"]) if anchor_row else (self.trip.event["venue_lat"],
                                                                                self.trip.event["venue_lng"])
        rows = []
        for r in df.to_dict("records"):
            rows.append((haversine_km(lat, lng, r["lat"], r["lng"]), r))
        return sorted(rows, key=lambda x: x[0])[:limit]

    def search_hotels(self, min_stars=None, max_distance_km=None, limit=8):
        p = self.trip.prefs
        min_stars = min_stars or p["hotel_min_stars"]
        max_km = max_distance_km or p["max_distance_km"]
        df = self.cat.df[self.cat.df.kind.isin(HOTEL_KINDS) & ~self.cat.df.name.str.contains(RESTRICTED, na=False)]
        if min_stars:
            df = df[pd.to_numeric(df.stars, errors="coerce").fillna(0) >= min_stars]
        band = BANDS[p["budget_band"]]
        lo, hi = band["stars"]
        def fits(r):
            # unrated OSM hotels count as 3-star; an unrated LiteAPI listing (apartments, homestays) only fits budget
            stars = r["stars"] or (None if hotels.is_lite(r["place_id"]) else 3)
            if stars is None:
                return p["budget_band"] == "budget"
            return r["kind"] in band["kinds"] and (lo is None or stars >= lo) and (hi is None or stars <= hi)

        def rating(r):  # a score from a handful of reviews counts for less
            return (r.get("rating") or 0) * (1 if (r.get("reviews") or 0) >= 20 else 0.6)
        rows = [(km, r) for km, r in self._place_rows(df, None, 400) if km <= max_km]
        rows.sort(key=lambda x: (not fits(x[1]), x[0]))  # budget-band fit first, then nearest
        pool = rows[:max(limit * 3, 20)]
        tl = self.trip.timeline
        live = hotels.rates([r["place_id"] for _, r in pool], tl["check_in"], tl["check_out"])
        sold_out = sum(1 for _, r in pool if r["place_id"] in live and live[r["place_id"]] is None)
        pool = [(km, r) for km, r in pool if not (r["place_id"] in live and live[r["place_id"]] is None)]
        # band fit, then hotels with a live rate, then distance in 1 km steps, then guest rating
        pool.sort(key=lambda x: (not fits(x[1]), not live.get(x[1]["place_id"]), int(x[0]), -rating(x[1])))
        out = []
        for km, r in pool:
            pid = r["place_id"]
            price = hotels.price(live[pid]) if live.get(pid) else price_of(r, self.trip.city)
            out.append({k: v for k, v in {
                "place_id": pid, "name": r["name"], "kind": r["kind"], "km": round(km, 2),
                "stars": int(r["stars"]) if r["stars"] else None, "price": price_text(price),
                "guest_rating": r.get("rating"), "reviews": int(r["reviews"]) if r.get("reviews") else None,
                "refundable": live[pid]["refundable"] if live.get(pid) else None}.items() if v is not None})
            if len(out) >= limit:
                break
        note = None if out else "no hotels match: try fewer stars or a larger distance"
        if sold_out:
            note = (note + "; " if note else "") + f"{sold_out} nearby hotels have no rooms for these dates"
        return {"hotels": out, "note": note}

    def _stay_price(self, row):
        """Live LiteAPI rate for the trip dates if there is one, else the estimate band. 'sold_out' if no rooms."""
        tl = self.trip.timeline
        live = hotels.rates([row["place_id"]], tl["check_in"], tl["check_out"])
        if row["place_id"] in live:
            return hotels.price(live[row["place_id"]]) if live[row["place_id"]] else "sold_out"
        return price_of(row, self.trip.city)

    def search_places(self, kinds=None, near="venue", slot=None, day=1, max_distance_km=5.0, limit=8):
        df = self.cat.df[~self.cat.df.kind.isin(HOTEL_KINDS)]
        if kinds:
            df = df[df.kind.isin(kinds)]
        if self.trip.prefs["veg_only"]:
            df = df[~df.kind.isin(FOOD_KINDS) | df.veg.astype(bool)]
        anchor = self.trip.stay_row(self.cat) if near == "hotel" else None
        win = None
        if slot:
            w = windows(self.trip.timeline)
            win = w.get(f"explore:{min(max(day or 1, 1), self.trip.timeline['days'])}" if slot == "explore" else slot)
        out = []
        for km, r in self._place_rows(df, anchor, 600):
            if km > max_distance_km or len(out) >= limit:
                break
            hc = hours_check(r, *win) if win else None
            if hc and hc[0] == "closed":
                continue
            out.append({k: v for k, v in {
                "place_id": r["place_id"], "name": r["name"], "kind": r["kind"], "km": round(km, 2),
                "cuisine": r["cuisine"], "veg": bool(r["veg"]) if r["kind"] in FOOD_KINDS else None,
                "opening_hours": r["opening_hours"] or "unknown: check hours", "hours_note": hc and hc[1],
                "price": price_text(price_of(r, self.trip.city))}.items() if v is not None})
        return {"places": out}

    def get_transport_options(self):
        from trip_planner import fares
        fares.enrich(self.trip)  # cached fares; a no-op without TRAVELPAYOUTS_TOKEN
        tl = self.trip.timeline
        legs = [{k: v for k, v in {
            "member_id": i["member_id"], "leg": i["title"], "slot": i["slot"],
            "arrive_by" if i["slot"] == "travel_in" else "leave_after": i["end"] or i["start"],
            "priced_as": i.get("mode"), "fare": price_text(i["price"]), "cheapest_flight": i.get("flight"),
            "train_fares_estimate": {i["train"]["labels"][c]: f"₹{v['min']:,}–{v['max']:,}"
                                     for c, v in i["train"]["classes"].items()} if i.get("train") else None,
            "book": [link["url"] for link in i["book"]]}.items() if v is not None}
            for i in self.trip.items if i["slot"] in ("travel_in", "travel_out")]
        travel_day = tl["arrive_by"].date()
        return {"legs": legs,
                "tatkal": f"Tatkal train tickets for {travel_day:%a %d %b} open on "
                          f"{travel_day - timedelta(days=1):%a %d %b}: 10:00 for AC classes, 11:00 for non-AC",
                "note": "flight fares are the cheapest recently seen, not live; train fares are estimates from the "
                        "official fare chart (Tatkal costs more); buses have links only. Users book on the provider's site."}

    def estimate_costs(self):
        return costs(self.trip)

    def get_weather(self):
        from trip_planner import weather
        return weather.for_trip(self.trip)

    # ---- edits -----------------------------------------------------------------
    def _edit(self, change, reason):
        before = self.trip.state()
        issues_before = validate(self.trip, self.cat)
        err = change()
        if err:
            self.trip.restore(before)
            return {"error": err}
        schedule(self.trip, self.cat)
        issues = validate(self.trip, self.cat, None if self.bypass_locks else before)
        hard = new_hard(issues, issues_before)
        if hard:
            self.trip.restore(before)
            return {"error": "; ".join(h["message"] for h in hard), "codes": [h["code"] for h in hard]}
        if not self.batch:
            self.trip.commit(self.author, reason)
        return {"ok": True, "version": self.trip.version, "warnings": [w["message"] for w in issues if w["level"] == "soft"]}

    def update_preferences(self, **changes):
        unknown = set(changes) - PREF_KEYS
        if unknown:
            return {"error": f"unknown preference(s): {sorted(unknown)}"}
        if changes.get("budget_band") not in (None, *BANDS):
            return {"error": f"budget_band must be one of {list(BANDS)}"}
        if "extra_days" in changes:
            changes["extra_days"] = min(max(int(changes["extra_days"] or 0), 0), MAX_EXTRA_DAYS)

        def change():
            self.trip.prefs.update({k: DEFAULT_PREFS[k] if v is None else v for k, v in changes.items()})
            if "extra_days" in changes:
                build_auto_items(self.trip)  # check-out and the return legs move
        return self._edit(change, f"preferences: {', '.join(changes)}")

    def add_item(self, slot, place_id, day=None, why="", note=""):
        row = self.cat.get(place_id)
        if slot not in PLACE_SLOTS:
            return {"error": f"unknown slot {slot}; use one of {list(PLACE_SLOTS)}"}
        if row is None:
            return {"error": f"unknown place_id {place_id}: only use IDs from search results"}
        tl = self.trip.timeline
        price = self._stay_price(row) if slot == "stay" and row["kind"] in HOTEL_KINDS else price_of(row, self.trip.city)
        if price == "sold_out":
            return {"error": f"{row['name']} has no rooms for these dates; pick another hotel", "codes": ["sold_out"]}

        def change():
            if slot == "stay":
                self.trip.items = [i for i in self.trip.items if i["slot"] != "stay"]
            book = links.hotel(row, self.trip.city, tl["check_in"], tl["check_out"], len(self.trip.members)) \
                if slot == "stay" else [{"provider": "maps", "label": "Open in Maps", "url": links.maps(row)}]
            self.trip.items.append(new_item(slot, "hotel" if slot == "stay" else "place", place_id, row["name"],
                                            day=(day or 1) if slot == "explore" else 0, why=why, note=note,
                                            added_by=self.author, price=price, book=book))
        return self._edit(change, f"add {row['name']} to {slot}")

    def remove_item(self, item_id):
        item = self.trip.find(item_id)
        if not item or item["slot"] not in PLACE_SLOTS:
            return {"error": "no such removable item; item_ids come from get_trip"}

        def change():
            self.trip.items = [i for i in self.trip.items if i["item_id"] != item_id]
        return self._edit(change, f"remove {item['title']}")

    def move_item(self, item_id, slot=None, day=None, start=None):
        item = self.trip.find(item_id)
        if not item or item["slot"] not in PLACE_SLOTS:
            return {"error": "no such movable item; item_ids come from get_trip"}

        def change():
            if slot:
                if (slot == "stay") != (item["slot"] == "stay"):
                    return "hotels only move within 'stay'"
                item["slot"] = slot
            if day is not None:
                item["day"] = day
            if start:
                try:
                    hh, mm = (int(x) for x in start.split(":"))
                except ValueError:
                    return "start must be HH:MM"
                base = self.trip.timeline["show_day"] + timedelta(days=item["day"] or 0)
                t0 = base.replace(hour=hh, minute=mm)
                dur = (_ts(item["end"]) - _ts(item["start"])) if item["start"] and item["end"] else timedelta(hours=1)
                item["start"], item["end"], item["fixed"] = f"{t0:%Y-%m-%d %H:%M}", f"{t0 + dur:%Y-%m-%d %H:%M}", True
        return self._edit(change, f"move {item['title']}")

    def undo(self):
        t = self.trip
        target = next((h for h in t.history if h["version"] == t.parent), None) if t.parent is not None else None
        if target is None or t.version == 0 or t.history[-1]["reason"] == "created":
            return {"error": "nothing to undo"}
        slots = set(t.locks) | set(target["locks"])
        ids = lambda items, s: sorted(i["item_id"] for i in items if i["slot"] == s)
        if target["locks"] != t.locks or any(ids(t.items, s) != ids(target["items"], s) for s in slots):
            return {"error": "undo would change something the group decided; start a group decision instead",
                    "codes": ["locked"]}
        t.restore(target)
        t.commit(self.author, f"undo to v{target['version']}", parent=target["parent"])
        return {"ok": True, "version": t.version, "warnings": self.warnings()}

    def replace_item(self, item_id, place_id):
        """Swap one place for another in the same slot and day, as one change."""
        item, row = self.trip.find(item_id), self.cat.get(place_id)
        if not item or item["slot"] not in PLACE_SLOTS or row is None:
            return {"error": "replace needs a plan item_id and a place_id from search results"}
        tl = self.trip.timeline
        price = self._stay_price(row) if item["slot"] == "stay" else price_of(row, self.trip.city)
        if price == "sold_out":
            return {"error": f"{row['name']} has no rooms for these dates", "codes": ["sold_out"]}

        def change():
            self.trip.items = [i for i in self.trip.items if i["item_id"] != item_id]
            book = links.hotel(row, self.trip.city, tl["check_in"], tl["check_out"], len(self.trip.members)) \
                if item["slot"] == "stay" else [{"provider": "maps", "label": "Open in Maps", "url": links.maps(row)}]
            self.trip.items.append(new_item(item["slot"], item["ref"]["kind"], place_id, row["name"], day=item["day"],
                                            added_by=self.author, price=price, book=book))
        return self._edit(change, f"swap {item['title']} for {row['name']}")

    # ---- members (card actions from Spring; membership rules live in Spring) ------------
    def add_member(self, user_id, origin_city, first_name=""):
        if any(m["user_id"] == user_id for m in self.trip.members):
            return {"error": "already a member"}

        def change():
            self.trip.members.append({"user_id": user_id, "first_name": first_name, "origin_city": origin_city,
                                      "role": "member"})
            build_auto_items(self.trip)  # their own travel legs
        return self._edit(change, f"{first_name or f'user {user_id}'} joined")

    def remove_member(self, user_id):
        if not any(m["user_id"] == user_id for m in self.trip.members):
            return {"error": "not a member"}
        if len(self.trip.members) == 1:
            return {"error": "a trip needs at least one member"}

        def change():
            self.trip.members = [m for m in self.trip.members if m["user_id"] != user_id]
            build_auto_items(self.trip)
        return self._edit(change, f"user {user_id} left")

    def set_origin(self, user_id, origin_city):
        m = next((m for m in self.trip.members if m["user_id"] == user_id), None)
        if not m:
            return {"error": "not a member"}

        def change():
            m["origin_city"] = origin_city
            build_auto_items(self.trip)
        return self._edit(change, f"user {user_id} now travels from {origin_city}")

    # ---- group decisions (the worker saves them and posts the poll card) ------------------
    def start_decision(self, kind, item_id=None, place_id=None, slot=None, changes=None):
        from trip_planner import groups
        payload = {k: v for k, v in {"item_id": item_id, "place_id": place_id, "slot": slot,
                                     "changes": changes}.items() if v is not None}
        err = groups.check_proposal(self.trip, self.cat, kind, payload)
        if err:
            return {"error": err}
        self.new_decisions.append((kind, payload, self.actor_id))
        return {"ok": True, "question": groups.describe(self.trip, self.cat, kind, payload),
                "note": "the group votes on it; it applies when a majority agrees, or after 24 h if it leads"}

    def propose_variants(self):
        from trip_planner import groups
        variants = groups.make_variants(self.trip, self.cat)
        if not variants:
            return {"error": "the catalog doesn't have hotels different enough for budget / balanced / premium "
                             "options here; suggest specific alternatives from search_hotels instead"}
        self.new_decisions.append(("variant", {}, self.actor_id))
        return {"ok": True, "variants": {b: {"stay": v["stay"], "cost_per_person": v["cost_per_person"]}
                                         for b, v in variants.items()},
                "note": "a vote is open for the group"}
