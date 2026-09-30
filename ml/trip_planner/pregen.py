"""Pre-build drafts overnight for popular upcoming events (design §4.1), so users get a cached plan instantly
and the Gemini free-tier quota is spent at night instead of at peak time.

Run from ml/ (cron, e.g. 02:00 IST):  python -m trip_planner.pregen --top 20
Each (event × budget band × language) plan is cached for 7 days; origin cities don't matter.
"""
import argparse
import time
from datetime import datetime

from common.utils import load
from trip_planner import llm, planner, store
from trip_planner.catalog import load_catalog
from trip_planner.model import DEFAULT_PREFS, new_trip


def run(top=20, bands=("balanced", "budget", "premium"), langs=("en",), group_sizes=(1, 2), pause_s=6.0,
        include_past=False) -> dict:
    events = load("events")
    if not include_past:
        events = events[events.start_time > datetime.now().strftime("%Y-%m-%d %H:%M")]
    events = events.sort_values("hotness", ascending=False).head(top)
    cache, done, skipped = store.DraftCache(), 0, 0
    for e in events.to_dict("records"):
        try:
            cat = load_catalog(e["city"])
        except FileNotFoundError:
            skipped += 1
            continue
        for band in bands:
            for lang in langs:
                for n in group_sizes:
                    trip = new_trip(e, [{"user_id": -i, "origin_city": e["city"]} for i in range(1, n + 1)],
                                    {**DEFAULT_PREFS, "budget_band": band, "lang": lang})
                    if cache.get(planner.cache_key(trip)):
                        continue
                    try:
                        _, model, _ = planner.draft(trip, cat, cache=cache)
                    except llm.Busy:
                        time.sleep(60)  # free tier: back off and move on
                        continue
                    done += model not in ("auto",)
                    time.sleep(pause_s)  # stay under the free tier's requests per minute
    return {"events": len(events), "cached": done, "skipped_no_catalog": skipped}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--include-past", action="store_true", help="dev only: the synthetic events are in the past")
    a = ap.parse_args()
    print(run(top=a.top, include_past=a.include_past))
