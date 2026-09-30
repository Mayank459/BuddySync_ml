"""Group planning (design §1.4, §9): group decisions, variants and applying what the group chose.

Every member is equal. Anyone can open a decision (choose a variant, swap an item, lock or unlock
a slot, change a group-wide preference). It passes as soon as a strict majority agrees; otherwise at
the deadline the option with the most votes wins, and a tie keeps the current plan.
Pure code: the store persists decisions and votes, the worker applies passed ones.
"""
import copy
from collections import Counter
from datetime import datetime, timedelta

from trip_planner.model import BANDS, PLACE_SLOTS, costs
from trip_planner.tools import PREF_KEYS, Tools

DEADLINE = timedelta(hours=24)
KINDS = ("variant", "swap", "lock", "unlock", "pref")


def options_for(trip, kind) -> list[str]:
    return list(trip.variants) if kind == "variant" else ["yes", "no"]


def check_proposal(trip, cat, kind, payload) -> str | None:
    """Why this proposal can't be opened, or None."""
    if kind not in KINDS:
        return f"kind must be one of {list(KINDS)}"
    if kind == "variant":
        return None if trip.variants else "no variants to choose from; ask the planner for options first"
    if kind == "swap":
        item = trip.find(payload.get("item_id"))
        if not item or item["slot"] not in PLACE_SLOTS:
            return "swap needs the item_id of a place in the plan"
        row = cat.get(payload.get("place_id"))
        if row is None:
            return "swap needs a place_id from search results"
        return None
    if kind in ("lock", "unlock"):
        slot = payload.get("slot")
        if slot not in PLACE_SLOTS:
            return f"slot must be one of {list(PLACE_SLOTS)}"
        if kind == "lock" and slot in trip.locks:
            return f"'{slot}' is already locked"
        if kind == "unlock" and slot not in trip.locks:
            return f"'{slot}' is not locked"
        return None
    changes = payload.get("changes") or {}
    if not changes or set(changes) - PREF_KEYS:
        return f"pref needs 'changes' with keys from {sorted(PREF_KEYS)}"
    return None


def describe(trip, cat, kind, payload) -> str:
    """One line for the poll card in clan chat."""
    if kind == "variant":
        return "Which plan should we go with? " + ", ".join(
            f"{b} (₹{v['cost_per_person']['min']:,}–{v['cost_per_person']['max']:,} pp, {v['stay']})"
            for b, v in trip.variants.items())
    if kind == "swap":
        item, row = trip.find(payload["item_id"]), cat.get(payload["place_id"])
        return f"Swap {item['title']} for {row['name']}?"
    if kind in ("lock", "unlock"):
        return f"{kind.title()} '{payload['slot']}'?"
    return "Change preferences: " + ", ".join(f"{k} → {v}" for k, v in payload["changes"].items()) + "?"


def resolve(options, votes: dict, n_members: int, deadline: datetime, now: datetime):
    """(status, winner). status: open | passed | rejected | tied | expired."""
    counts = Counter(v for v in votes.values() if v in options)
    if counts:
        top, top_n = counts.most_common(1)[0]
        if top_n * 2 > n_members:  # strict majority of all members: decided early
            return ("rejected" if top == "no" else "passed"), top
    if now < deadline:
        return "open", None
    if not counts:
        return "expired", None
    best = max(counts.values())
    leaders = [o for o, c in counts.items() if c == best]
    if len(leaders) > 1:
        return "tied", None  # keep the current plan
    return ("rejected" if leaders[0] == "no" else "passed"), leaders[0]


def make_variants(trip, cat) -> dict:
    """Budget / balanced / premium: the current plan with the best-fitting hotel for each band."""
    variants = {}
    for band in BANDS:
        probe = copy.deepcopy(trip)
        probe.prefs["budget_band"] = band
        probe.prefs["hotel_min_stars"] = None
        hotels = Tools(probe, cat).search_hotels(limit=5)["hotels"]
        if not hotels:
            continue
        tools = Tools(probe, cat, author="planner", batch=True)
        for h in hotels:
            if "ok" in tools.add_item("stay", h["place_id"], why=f"best {band} fit near the venue"):
                break
        if not probe.in_slot("stay"):
            continue
        stay = probe.in_slot("stay")[0]
        if any(v["stay_id"] == stay["ref"]["id"] for v in variants.values()):
            continue  # same hotel as a cheaper band: not a real option (few rated hotels in the catalog)
        variants[band] = {"items": [i for i in probe.items if i["slot"] in PLACE_SLOTS], "stay": stay["title"],
                          "stay_id": stay["ref"]["id"], "cost_per_person": costs(probe)["per_person"]}
    trip.variants = variants if len(variants) >= 2 else {}  # a vote needs at least two different plans
    return trip.variants


def apply(trip, cat, decision: dict, winner: str) -> dict:
    """Apply a passed decision as one new version, authored 'group decision'. Locks don't block it."""
    kind, payload = decision["kind"], decision["payload"]
    tools = Tools(trip, cat, author="group decision", bypass_locks=True)
    if kind == "variant":
        chosen = trip.variants.get(winner)
        if not chosen:
            return {"error": f"variant {winner} no longer exists"}

        def change():
            trip.items = [i for i in trip.items if i["slot"] not in PLACE_SLOTS] + copy.deepcopy(chosen["items"])
            trip.prefs["budget_band"] = winner
        return tools._edit(change, f"group chose the {winner} plan")
    if kind == "swap":
        return tools.replace_item(payload["item_id"], payload["place_id"])
    if kind == "lock":
        trip.locks[payload["slot"]] = decision["decision_id"]
        trip.commit("group decision", f"locked {payload['slot']}")
        return {"ok": True, "version": trip.version}
    if kind == "unlock":
        trip.locks.pop(payload["slot"], None)
        trip.commit("group decision", f"unlocked {payload['slot']}")
        return {"ok": True, "version": trip.version}
    return tools.update_preferences(**payload["changes"])


def member_costs(trip) -> dict:
    """Per member: shared items split, plus that member's own items (their travel legs)."""
    c = costs(trip)
    own = {m["user_id"]: [i for i in trip.items if i.get("member_id") == m["user_id"] and i["price"]]
           for m in trip.members}
    out = {}
    for m in trip.members:
        lo = sum(line["min"] for line in c["lines"] if not trip.find(line["item_id"]).get("member_id"))
        hi = sum(line["max"] for line in c["lines"] if not trip.find(line["item_id"]).get("member_id"))
        lo += sum(i["price"]["min"] for i in own[m["user_id"]])
        hi += sum(i["price"]["max"] for i in own[m["user_id"]])
        out[m["user_id"]] = {"min": lo, "max": hi, "currency": "INR"}
    return out
