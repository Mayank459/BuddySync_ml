"""Daily forecast for the trip dates from Open-Meteo (design §7), cached 6 h per city and date range.

ponytail: Open-Meteo's free API is for non-commercial use; switch to its commercial plan (or set
WEATHER_URL to it) before launch. Forecasts only reach 16 days ahead; beyond that we say so.
"""
import os
import time
from datetime import date, timedelta

import requests

URL = os.getenv("WEATHER_URL", "https://api.open-meteo.com/v1/forecast")
HORIZON_DAYS = 16
TTL_S = 6 * 3600
_cache: dict = {}
CODES = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "cloudy", 45: "fog", 48: "fog", 51: "drizzle",
         53: "drizzle", 55: "drizzle", 61: "light rain", 63: "rain", 65: "heavy rain", 80: "showers",
         81: "showers", 82: "heavy showers", 95: "thunderstorm", 96: "thunderstorm", 99: "thunderstorm"}


def forecast(lat: float, lng: float, start: date, end: date, today: date | None = None) -> dict:
    today = today or date.today()
    if end < today:
        return {"available": False, "note": "the trip dates are in the past"}
    start = max(start, today)
    if start > today + timedelta(days=HORIZON_DAYS - 1):
        return {"available": False, "note": f"forecast opens {HORIZON_DAYS} days before the trip"}
    end = min(end, today + timedelta(days=HORIZON_DAYS - 1))
    key = (round(lat, 2), round(lng, 2), start, end)
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < TTL_S:
        return hit[1]
    try:
        r = requests.get(URL, timeout=8, params={
            "latitude": lat, "longitude": lng, "timezone": "Asia/Kolkata", "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"})
        r.raise_for_status()
        d = r.json()["daily"]
    except (requests.RequestException, KeyError, ValueError):
        return {"available": False, "note": "weather service unavailable right now"}
    days = [{"date": day, "sky": CODES.get(code, "mixed"), "max_c": hi, "min_c": lo, "rain_chance_pct": rain,
             "indoor_advised": (rain or 0) >= 60 or code in (63, 65, 81, 82, 95, 96, 99)}
            for day, code, hi, lo, rain in zip(d["time"], d["weather_code"], d["temperature_2m_max"],
                                               d["temperature_2m_min"], d["precipitation_probability_max"])]
    out = {"available": True, "days": days, "source": "Open-Meteo"}
    _cache[key] = (time.monotonic(), out)
    return out


def for_trip(trip) -> dict:
    tl = trip.timeline
    return forecast(trip.event["venue_lat"], trip.event["venue_lng"], tl["show_day"].date(), tl["check_out"].date())
