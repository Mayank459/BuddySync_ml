"""Flight fare ranges for travel legs from Travelpayouts' cached Aviasales prices (design §7).

TRAVELPAYOUTS_TOKEN: Data API token. TRAVELPAYOUTS_MARKER: partner ID that makes the links affiliate links.
The Data API returns the cheapest fares seen recently, not live availability, so prices are a RANGE
(cheapest seen within ±3 days of the travel date) with the time we fetched them, and users book on Aviasales.
ponytail: the real-time Search API needs 50k MAU; cached fares until then.
"""
import os
import time
from datetime import date, datetime, timedelta

import requests

URL = "https://api.travelpayouts.com/aviasales/v3/prices_for_dates"
TTL_S = 30 * 60
NEAR_DAYS = 3
IATA = {"mumbai": "BOM", "delhi": "DEL", "new delhi": "DEL", "bengaluru": "BLR", "bangalore": "BLR", "hyderabad": "HYD",
        "chennai": "MAA", "kolkata": "CCU", "pune": "PNQ", "ahmedabad": "AMD", "goa": "GOI", "jaipur": "JAI",
        "kochi": "COK", "cochin": "COK", "lucknow": "LKO", "chandigarh": "IXC", "indore": "IDR", "nagpur": "NAG",
        "surat": "STV", "vadodara": "BDQ", "bhopal": "BHO", "patna": "PAT", "bhubaneswar": "BBI", "guwahati": "GAU",
        "thiruvananthapuram": "TRV", "trivandrum": "TRV", "coimbatore": "CJB", "visakhapatnam": "VTZ", "vizag": "VTZ",
        "varanasi": "VNS", "amritsar": "ATQ", "srinagar": "SXR", "udaipur": "UDR", "mangaluru": "IXE",
        "mangalore": "IXE", "raipur": "RPR", "ranchi": "IXR", "dehradun": "DED", "madurai": "IXM", "nashik": "ISK"}
_cache: dict = {}


def _month(orig: str, dest: str, ym: str) -> dict:
    """Cheapest cached fare per departure date for one month: {date: row}."""
    key = (orig, dest, ym)
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < TTL_S:
        return hit[1]
    token = os.getenv("TRAVELPAYOUTS_TOKEN")
    if not token:
        return {}
    try:
        r = requests.get(URL, timeout=10, params={"origin": orig, "destination": dest, "departure_at": ym,
                                                   "currency": "inr", "one_way": "true", "sorting": "price",
                                                   "limit": 1000, "token": token})
        r.raise_for_status()
        rows = r.json().get("data", [])
    except (requests.RequestException, ValueError):
        return {}
    by_day = {}
    for row in rows:
        d = date.fromisoformat(row["departure_at"][:10])
        if d not in by_day or row["price"] < by_day[d]["price"]:
            by_day[d] = row
    _cache[key] = (time.monotonic(), by_day)
    return by_day


def quote(origin_city: str, dest_city: str, day: date, direction: str = "before") -> dict | None:
    """direction 'before': inbound, only flights on or before `day`; 'after': outbound, on or after `day`.
    The quoted flight is from the closest date; the range covers every cached date in that window."""
    o, d = IATA.get((origin_city or "").strip().lower()), IATA.get((dest_city or "").strip().lower())
    if not o or not d or o == d:
        return None
    sign = -1 if direction == "before" else 1
    window = [day + timedelta(days=sign * k) for k in range(NEAR_DAYS + 1)]  # closest date first
    rows = {k: v for ym in {f"{x:%Y-%m}" for x in window} for k, v in _month(o, d, ym).items()}
    near = {k: rows[k] for k in window if k in rows}
    if not near:
        return None
    best_day = next(k for k in window if k in near)
    best = near[best_day]
    prices = [r["price"] for r in near.values()]
    dep = datetime.fromisoformat(best["departure_at"]).replace(tzinfo=None)
    arr = dep + timedelta(minutes=best.get("duration_to") or best.get("duration") or 0)
    marker = os.getenv("TRAVELPAYOUTS_MARKER")
    url = "https://www.aviasales.com" + best["link"] + (f"&marker={marker}" if marker else "")
    return {"price": {"min": min(prices), "max": max(prices), "currency": "INR", "unit": "person", "confidence": "range",
                      "source": "cheapest cached fares (Aviasales via Travelpayouts)",
                      "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M")},
            "flight": {"airline": best["airline"], "flight_number": f"{best['airline']} {best['flight_number']}",
                       "departs": f"{dep:%Y-%m-%d %H:%M}", "arrives": f"{arr:%Y-%m-%d %H:%M}",
                       "stops": best.get("transfers", 0), "same_day": best_day == day, "from": o, "to": d},
            "book": {"provider": "aviasales", "label": f"Flight from ₹{best['price']:,}", "url": url,
                     "affiliate": bool(marker)}}


TRAIN_FIRST_KM = 700  # straight-line km: below this the leg's price is the AC 3-tier train fare


def enrich(trip) -> int:
    """Fare ranges on travel legs not priced yet: a train fare (Railway Board chart) and the cheapest
    cached flight. The leg's `price` is the train (AC 3-tier) for trips under ~700 km, else the flight.
    Returns legs priced."""
    from trip_planner import rail
    n = 0
    for i in trip.items:
        if i["slot"] not in ("travel_in", "travel_out") or i.get("mode"):
            continue
        member = next((m for m in trip.members if m["user_id"] == i["member_id"]), None)
        if not member:
            continue
        when = datetime.fromisoformat(i["end"] if i["slot"] == "travel_in" else i["start"]).date()
        a, b = (member["origin_city"], trip.city) if i["slot"] == "travel_in" else (trip.city, member["origin_city"])
        flight = quote(a, b, when, "before" if i["slot"] == "travel_in" else "after")
        train = rail.quote(a, b)
        if flight:
            i["flight"] = flight["flight"]
            i["book"] = [flight["book"]] + [link for link in i["book"] if link["provider"] != "aviasales"]
        if train:
            i["train"] = train
        by_train = train and (not flight or train["rail_km"][0] / rail.RAIL_FACTOR[0] < TRAIN_FIRST_KM)
        if by_train:
            ac3 = train["classes"]["3A"]
            i["price"] = {"min": ac3["min"], "max": ac3["max"], "currency": "INR", "unit": "person",
                          "confidence": "estimate", "source": f"AC 3-tier train fare ({train['source']})",
                          "fetched_at": None}
            i["mode"] = "train"
        elif flight:
            i["price"], i["mode"] = flight["price"], "flight"
        else:
            continue
        n += 1
    return n
