"""Reminder times for Spring to schedule as notifications (design §1.5, "Trip mode"). Pure code.

Each reminder: {at (IST, "YYYY-MM-DD HH:MM"), kind, text, member_ids (None = everyone), item_id?}.
"""
from datetime import datetime, timedelta

from trip_planner.model import fmt, km_to_venue, travel_min
from trip_planner.transit import LAST_METRO

BUFFER = timedelta(minutes=30)  # reach the gates this long after they open is fine; leave with margin


def reminders(trip, cat) -> list[dict]:
    tl, out = trip.timeline, []

    def add(at, kind, text, member_ids=None, item_id=None):
        out.append({"at": fmt(at), "kind": kind, "text": text, "member_ids": member_ids, "item_id": item_id})

    for i in trip.items:
        if i["slot"] == "travel_in":
            day = datetime.fromisoformat(i["end"]).date()
            eve = datetime.combine(day - timedelta(days=1), datetime.min.time())
            if i.get("train") or i.get("mode") == "train":
                add(eve.replace(hour=9, minute=45), "tatkal",
                    f"Tatkal for {day:%a %d %b} opens at 10:00 (AC) and 11:00 (non-AC) today: book early on IRCTC",
                    [i["member_id"]], i["item_id"])
            add(datetime.fromisoformat(i["end"]) - timedelta(hours=6), "travel",
                f"Travel day: {i['title']}. Arrive by {i['end'][11:]}", [i["member_id"]], i["item_id"])
        if i["slot"] == "travel_out":
            add(datetime.fromisoformat(i["start"]) - timedelta(hours=2), "travel",
                f"Heading home: {i['title']} after {i['start'][11:]}", [i["member_id"]], i["item_id"])

    hotel = trip.stay_row(cat)
    if hotel:
        add(tl["check_in"], "check_in", f"Check-in at {hotel['name']} opens now")
        mins = travel_min(km_to_venue(trip, hotel))
        leave = tl["gates_open"] - timedelta(minutes=mins) - BUFFER
        add(leave - timedelta(minutes=15), "leave_for_show",
            f"Leave {hotel['name']} by {leave:%H:%M} to reach {trip.event['venue_name']} "
            f"(~{mins} min) as gates open at {tl['gates_open']:%H:%M}")
        add(tl["check_out"] - timedelta(hours=1), "check_out", f"Check out of {hotel['name']} by {tl['check_out']:%H:%M}")
    else:
        add(tl["gates_open"] - timedelta(hours=1), "leave_for_show", f"Gates open at {tl['gates_open']:%H:%M}")
    add(tl["gates_open"], "gates", f"Gates are open at {trip.event['venue_name']}")

    ret = trip.in_slot("return")
    metro = LAST_METRO.get(trip.city)
    if metro:
        last = datetime.combine(tl["show_day"].date(), metro["last"])
        if tl["show_end"] + BUFFER > last:
            add(last - timedelta(minutes=45), "last_metro",
                f"{trip.city}'s last metro is around {metro['last']:%H:%M}, before the show ends at "
                f"{tl['show_end']:%H:%M}: plan a cab back")
    if ret and ret[0]["title"].startswith("Cab"):
        add(tl["show_end"] - timedelta(minutes=20), "book_cab",
            f"Show ends at {tl['show_end']:%H:%M}: book your cab now. {ret[0]['title']}", None, ret[0]["item_id"])
    return sorted(out, key=lambda r: r["at"])
