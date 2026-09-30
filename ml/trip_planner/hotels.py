"""Hotel inventory and nightly rates from LiteAPI (design §7).

LITEAPI_KEY: `sand_…` (sandbox, test inventory) or `prod_…`. Catalog hotels from LiteAPI have ids like `lp…`.
Rates are per room (2 adults) per night, as a range: LiteAPI's retail rate → the public price other sites
show (suggestedSellingPrice), since booking is handed off to Booking.com / Agoda. Sandbox rates are labelled
'range' (test data); production rates 'verified'. Cached per hotel for an hour (design: refresh on view).
"""
import os
import time
from datetime import datetime

import requests

BASE = "https://api.liteapi.travel/v3.0"
TTL_S = 3600
CHUNK = 100  # hotels per rates call (~7 s for 100 in the sandbox)
_rates: dict = {}  # (hotel_id, checkin, checkout, adults) -> (fetched_monotonic, fetched_at, rate | None)


def key():
    return os.getenv("LITEAPI_KEY", "")


def is_lite(place_id) -> bool:
    return isinstance(place_id, str) and place_id.startswith("lp")


def hotels_near(lat, lng, radius_m=12000) -> list[dict]:
    """Every LiteAPI hotel within radius_m (paged), for the catalog build."""
    out, offset = [], 0
    while key():
        r = requests.get(f"{BASE}/data/hotels", headers={"X-API-Key": key(), "accept": "application/json"},
                         params={"latitude": lat, "longitude": lng, "radius": radius_m, "limit": 1000,
                                 "offset": offset}, timeout=60)
        r.raise_for_status()
        page = r.json().get("data", [])
        out += page
        if len(page) < 1000:
            break
        offset += 1000
    return out


def rates(hotel_ids, check_in, check_out, adults=2) -> dict:
    """{hotel_id: rate or None (no rooms)} for the stay; unknown to LiteAPI or no key → not in the result."""
    ids = [h for h in dict.fromkeys(hotel_ids) if is_lite(h)]
    if not ids or not key():
        return {}
    ci, co = f"{check_in:%Y-%m-%d}", f"{check_out:%Y-%m-%d}"
    now = time.monotonic()
    fresh = {h: _rates[(h, ci, co, adults)] for h in ids
             if (h, ci, co, adults) in _rates and now - _rates[(h, ci, co, adults)][0] < TTL_S}
    missing = [h for h in ids if h not in fresh]
    nights = max(1, (check_out.date() - check_in.date()).days)
    for n in range(0, len(missing), CHUNK):
        part = missing[n:n + CHUNK]
        try:
            r = requests.post(f"{BASE}/hotels/rates", timeout=60,
                              headers={"X-API-Key": key(), "accept": "application/json",
                                       "content-type": "application/json"},
                              json={"hotelIds": part, "checkin": ci, "checkout": co, "currency": "INR",
                                    "guestNationality": "IN", "occupancies": [{"adults": adults}]})
            r.raise_for_status()
            data = {h["hotelId"]: h for h in r.json().get("data", [])}
        except (requests.RequestException, ValueError):
            continue  # try again next time; callers fall back to estimate bands
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        for h in part:
            rate = _cheapest(data.get(h), nights)
            _rates[(h, ci, co, adults)] = (time.monotonic(), stamp, rate)
            fresh[h] = _rates[(h, ci, co, adults)]
    return {h: (v[2] and {**v[2], "fetched_at": v[1]}) for h, v in fresh.items()}


def _cheapest(hotel, nights):
    if not hotel:
        return None
    best = None
    for rt in hotel.get("roomTypes", []):
        retail = (rt.get("offerRetailRate") or {}).get("amount")
        if retail is None:
            continue
        public = (rt.get("suggestedSellingPrice") or {}).get("amount") or retail
        rate = rt["rates"][0] if rt.get("rates") else {}
        cand = {"min": int(round(retail / nights, -1)), "max": int(round(max(public, retail) / nights, -1)),
                "room": rate.get("name"), "board": rate.get("boardName"),
                "refundable": (rate.get("cancellationPolicies") or {}).get("refundableTag") == "RFN"}
        if best is None or cand["min"] < best["min"]:
            best = cand
    return best


def price(rate: dict) -> dict:
    sandbox = key().startswith("sand_")
    return {"min": rate["min"], "max": rate["max"], "currency": "INR", "unit": "night",
            "confidence": "range" if sandbox else "verified",
            "source": "LiteAPI sandbox rates (test inventory)" if sandbox else "LiteAPI live rates",
            "fetched_at": rate["fetched_at"], "room": rate["room"], "board": rate["board"],
            "refundable": rate["refundable"]}


def refresh_stay(trip) -> bool:
    """Re-price the chosen LiteAPI hotel if its rate is missing or over an hour old."""
    stay = trip.in_slot("stay")
    if not stay or not is_lite(stay[0]["ref"]["id"]):
        return False
    tl = trip.timeline
    r = rates([stay[0]["ref"]["id"]], tl["check_in"], tl["check_out"]).get(stay[0]["ref"]["id"])
    if r:
        stay[0]["price"] = price(r)
        return True
    return False
