"""Demo kit for presenting the trip planner. Run from ml/ on demo day, with the API's env vars set:

    python -m trip_planner.demo            # demo event 7 days from today, checks, pre-built drafts
    python -m trip_planner.demo --days 5   # pick the date offset (weather needs ≤ 16 days ahead)

1. Puts the demo event in data/events.csv, dated --days from today: live flight fares, hotel rates and
   weather only exist for future dates (the synthetic events are mostly in the past).
2. Checks every key and service and prints READY / MISSING.
3. Pre-builds the drafts the demo script uses (draft cache), so the first plan appears instantly on stage
   without spending free-tier LLM quota.
"""
import argparse
import os
from datetime import date, timedelta

import pandas as pd

from common.utils import DATA

DEMO_ID = 9001
SOLO = [{"user_id": 7, "first_name": "Riya", "origin_city": "Pune"}]
CLAN = [{"user_id": 7, "first_name": "Riya", "origin_city": "Delhi"},
        {"user_id": 8, "first_name": "Arjun", "origin_city": "Bengaluru"},
        {"user_id": 9, "first_name": "Sana", "origin_city": "Mumbai"}]


def demo_event(day: date) -> dict:
    d = day.isoformat()
    return {"event_id": DEMO_ID, "category": "concert", "tags": "bollywood music", "city": "Mumbai",
            "performers": "Midnight Echoes", "venue_name": "NSCI Dome", "venue_lat": 18.9985, "venue_lng": 72.8175,
            "title": "Midnight Echoes Live at NSCI Dome",
            "description": "A late-night Bollywood and indie set at NSCI Dome, Worli. bhai chalo saath chalte hain",
            "start_time": f"{d} 19:00:00", "gates_open": f"{d} 17:30:00", "end_time": f"{d} 22:45:00",
            "price_band": 3, "hotness": 99.0}


def put_event(days: int) -> dict:
    path = DATA / "events.csv"
    events = pd.read_csv(path)
    e = demo_event(date.today() + timedelta(days=days))
    events = pd.concat([events[events.event_id != DEMO_ID], pd.DataFrame([e])], ignore_index=True)
    events.to_csv(path, index=False)
    return e


def check(e) -> list[tuple[str, bool, str]]:
    from trip_planner import fares, hotels, llm, store, weather
    from trip_planner.catalog import load_catalog
    out = []

    def row(name, fn):
        try:
            ok, note = fn()
        except Exception as ex:  # a check must never crash the kit
            ok, note = False, f"{type(ex).__name__}: {str(ex)[:120]}"
        out.append((name, ok, note))

    row("database", lambda: (bool(store.pool()), os.getenv("DATABASE_URL") and "DATABASE_URL" or "embedded dev Postgres"))
    row("Mumbai catalog", lambda: (True, f"{len(load_catalog('Mumbai').rows)} places"))

    def ping():
        r = llm.complete("utility", "Reply with one word.", [{"role": "user", "content": "Say OK"}])
        return True, f"{r.model} answered"
    row("LLM", ping if llm.configured() else lambda: (False, "set LLM_BASE_URL / LLM_API_KEY / LLM_MODEL"))
    day = pd.Timestamp(e["start_time"])
    row("hotel rates (LiteAPI)", lambda: (bool(hotels.rates(["lp655c3"], day, day + pd.Timedelta(days=1))),
                                          "St. Regis Mumbai priced" if hotels.key() else "set LITEAPI_KEY"))
    row("flight fares (Travelpayouts)", lambda: (bool(fares.quote("Delhi", "Mumbai", day.date())),
                                                 "Delhi → Mumbai priced" if os.getenv("TRAVELPAYOUTS_TOKEN")
                                                 else "set TRAVELPAYOUTS_TOKEN"))
    row("affiliate marker", lambda: (bool(os.getenv("TRAVELPAYOUTS_MARKER")), "TRAVELPAYOUTS_MARKER=783614"))
    row("weather", lambda: (weather.forecast(e["venue_lat"], e["venue_lng"], day.date(), day.date())["available"],
                            "Open-Meteo forecast for the show day"))
    row("voice (Groq Whisper)", lambda: (bool(os.getenv("STT_API_KEY") or os.getenv("LLM_FALLBACK_API_KEY")),
                                         "STT_API_KEY"))
    return out


def prewarm(e) -> list[str]:
    from trip_planner import planner, store
    from trip_planner.catalog import load_catalog
    from trip_planner.model import DEFAULT_PREFS, new_trip
    cat, cache, done = load_catalog("Mumbai"), store.DraftCache(), []
    for name, members, prefs in (("solo from Pune", SOLO, {}), ("clan of 3", CLAN, {}),
                                 ("solo, Hinglish", SOLO, {"lang": "hinglish"})):
        trip = new_trip(e, members, {**DEFAULT_PREFS, **prefs})
        _, model, _ = planner.draft(trip, cat, cache=cache)
        done.append(f"{name}: {model}")
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--no-prewarm", action="store_true")
    a = ap.parse_args()
    e = put_event(a.days)
    print(f"demo event {DEMO_ID}: {e['title']}, {e['start_time']} (ends {e['end_time'][11:16]})\n")
    results = check(e)
    for name, ok, note in results:
        print(f"  {'READY  ' if ok else 'MISSING'}  {name:30} {note}")
    if not a.no_prewarm:
        print("\npre-building drafts:")
        for line in prewarm(e):
            print("  ", line)
        print("  ('cache:…' = already cached; 'auto' = no LLM available, code plan cached nothing)")
    print("\nnext: start (or keep) uvicorn and the playground → Trip planner tab. Script: ../DEMO.md")
