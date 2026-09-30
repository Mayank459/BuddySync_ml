import pandas as pd

from event_ingestion.model import blocked_pairs, dedup_action, norm_title


def ev(title, lat=19.0, lng=72.8, hours=0, city="Mumbai"):
    return {"title": title, "performers": "x", "city": city, "venue_lat": lat, "venue_lng": lng,
            "start_time": pd.Timestamp("2026-10-10 19:00") + pd.Timedelta(hours=hours), "description": ""}


def test_norm_title_ignores_source_formatting():
    assert norm_title("KAVYA TRIO LIVE @ MUMBAI | Tickets") == norm_title("Kavya Trio Live in Mumbai")


def test_blocking_keeps_near_same_day_only():
    left = pd.DataFrame([ev("a")])
    right = pd.DataFrame([ev("same place"), ev("other city", city="Pune"),
                          ev("2 km away", lat=19.018), ev("next week", hours=24 * 7)])
    assert blocked_pairs(left, right) == [(0, 0)]


def test_thresholds():
    assert [dedup_action(p) for p in (0.95, 0.7, 0.2)] == ["merge", "review", "new"]
