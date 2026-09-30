import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Trip tests stay offline: no forecast, no cached fares, no hotel rates (tests that need them patch again)."""
    from trip_planner import fares, hotels, weather
    monkeypatch.setattr(weather, "forecast", lambda *a, **kw: {"available": False, "note": "offline test"})
    monkeypatch.setattr(fares, "_month", lambda *a, **kw: {})
    monkeypatch.setattr(hotels, "rates", lambda *a, **kw: {})
