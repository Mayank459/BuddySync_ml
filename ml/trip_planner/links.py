"""Hand-off links: we never book, we link to the provider with the search pre-filled (design §8).

ponytail: plain search links; stream E adds affiliate IDs (Travelpayouts, Cuelinks) and the /v1/go click redirect.
"""
from urllib.parse import quote, urlencode, urlparse

# /v1/go redirects only to these hosts (and their subdomains): it must never be an open redirect
ALLOWED_HOSTS = ("aviasales.com", "booking.com", "agoda.com", "irctc.co.in", "redbus.in", "uber.com",
                 "google.com", "ixigo.com", "confirmtkt.com", "bookmyshow.com", "district.in", "abhibus.com",
                 "makemytrip.com", "cleartrip.com", "tp.media", "travelpayouts.com")
DISCLOSURE = ("BuddySync may earn a commission when you book through these links. "
              "We rank options by fit for your trip, never by commission. We don't book or take payment.")


def allowed(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return urlparse(url).scheme == "https" and any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS)


def tracked(link: dict, trip_id: str, item_id: str) -> str:
    """Relative URL of the click redirect; Spring or the app prefixes the ML base URL and adds user_id."""
    return "/v1/go?" + urlencode({"url": link["url"], "trip_id": trip_id, "item_id": item_id,
                                  "provider": link["provider"]})


def transport(origin: str, city: str, date) -> list[dict]:
    q = quote(f"Flights from {origin} to {city} on {date}")
    return [{"provider": "google_flights", "label": "Flights", "url": f"https://www.google.com/travel/flights?q={q}"},
            {"provider": "irctc", "label": "Trains", "url": "https://www.irctc.co.in/nget/train-search"},
            {"provider": "redbus", "label": "Buses", "url": "https://m.redbus.in/"}]


def cab(event: dict, hotel: dict) -> list[dict]:
    params = {"action": "setPickup", "pickup[latitude]": event["venue_lat"], "pickup[longitude]": event["venue_lng"],
              "pickup[nickname]": event["venue_name"], "dropoff[latitude]": hotel["lat"],
              "dropoff[longitude]": hotel["lng"], "dropoff[nickname]": hotel["name"]}
    return [{"provider": "uber", "label": "Uber", "url": "https://m.uber.com/ul/?" + urlencode(params)}]


def maps(row: dict) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={row['lat']},{row['lng']}"


def hotel(row: dict, city: str, check_in, check_out, adults: int) -> list[dict]:
    q = {"ss": f"{row['name']}, {city}", "checkin": f"{check_in:%Y-%m-%d}", "checkout": f"{check_out:%Y-%m-%d}",
         "group_adults": adults}
    return [{"provider": "booking", "label": "Booking.com", "url": "https://www.booking.com/searchresults.html?" + urlencode(q)},
            {"provider": "agoda", "label": "Agoda",
             "url": "https://www.agoda.com/search?" + urlencode({"textToSearch": f"{row['name']} {city}",
                                                                   "checkIn": f"{check_in:%Y-%m-%d}",
                                                                   "checkOut": f"{check_out:%Y-%m-%d}",
                                                                   "adults": adults})}]
