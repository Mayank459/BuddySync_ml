"""Data step for the trip planner: a real places + hotels catalog per city from OpenStreetMap.

The planner may only recommend places from this catalog, which is what stops it inventing places.
Run from ml/:  python -m trip_planner.catalog Mumbai
"""
import sys
from functools import lru_cache

import pandas as pd
import requests

from common.synthetic import CITIES
from common.utils import DATA

OVERPASS = ["https://overpass-api.de/api/interpreter",  # public servers get busy: try mirrors in turn
            "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.private.coffee/api/interpreter"]
QUERY = """[out:json][timeout:120];
(
  nwr["tourism"~"^(attraction|museum|viewpoint|gallery|zoo|theme_park)$"]["name"](around:{r},{lat},{lng});
  nwr["historic"~"^(monument|fort|castle|memorial)$"]["name"](around:{r},{lat},{lng});
  nwr["leisure"="park"]["name"](around:{r},{lat},{lng});
  nwr["amenity"~"^(restaurant|cafe)$"]["name"](around:{r},{lat},{lng});
  nwr["tourism"~"^(hotel|guest_house|hostel)$"]["name"](around:{r},{lat},{lng});
);
out center tags;"""


def kind_of(tags: dict) -> str:
    for key in ("tourism", "amenity", "historic", "leisure"):
        if key in tags:
            return tags[key]
    return "place"


def build(city: str, radius_m=12000) -> pd.DataFrame:
    lat, lng = CITIES[city]
    for url in OVERPASS:
        r = requests.post(url, data={"data": QUERY.format(r=radius_m, lat=lat, lng=lng)},
                          headers={"User-Agent": "BuddySync-trip-planner/0.1 (student project)"}, timeout=180)
        if r.ok:
            break
        print(f"{url}: HTTP {r.status_code}, trying next")
    r.raise_for_status()
    rows = []
    for el in r.json()["elements"]:
        t = el.get("tags", {})
        pos = el.get("center", el)
        stars = t.get("stars", "").rstrip("S")  # OSM uses "4", "5", "3S"
        rows.append({"place_id": f"{el['type'][0]}{el['id']}", "name": t["name"], "kind": kind_of(t),
                     "lat": pos["lat"], "lng": pos["lon"],
                     "stars": int(stars) if stars.isdigit() else None,
                     "opening_hours": t.get("opening_hours"), "cuisine": t.get("cuisine"),
                     "veg": t.get("diet:vegetarian") in ("yes", "only")})
    return pd.DataFrame(rows).drop_duplicates("place_id")


def merge_liteapi(df: pd.DataFrame, city: str, radius_m=12000) -> pd.DataFrame:
    """Add LiteAPI hotels (stars, rating, photo; live rates at search time) and drop OSM hotels they duplicate."""
    import numpy as np
    from trip_planner import hotels
    lat, lng = CITIES[city]
    lite = pd.DataFrame([{"place_id": h["id"], "name": h["name"],
                          "kind": "hostel" if "hostel" in h["name"].lower() else "hotel",
                          "lat": h["latitude"], "lng": h["longitude"], "stars": h.get("stars") or None,
                          "opening_hours": None, "cuisine": None, "veg": False, "rating": h.get("rating") or None,
                          "reviews": h.get("reviewCount") or None, "photo": h.get("main_photo") or None,
                          "source": "liteapi"}
                         for h in hotels.hotels_near(lat, lng, radius_m) if h.get("latitude")])
    df = df[~df.place_id.astype(str).str.startswith("lp")]  # a re-merge replaces the old LiteAPI rows
    if lite.empty:
        return df
    osm_hotels = df.kind.isin({"hotel", "guest_house", "hostel"})
    h = df[osm_hotels]
    # ponytail: O(osm × lite) flat-earth distances; fine for a few thousand hotels per city
    dy = (h.lat.values[:, None] - lite.lat.values[None, :]) * 111_000
    dx = (h.lng.values[:, None] - lite.lng.values[None, :]) * 111_000 * np.cos(np.radians(lat))
    dup = (np.hypot(dx, dy) < 80).any(axis=1)
    df = pd.concat([df[~osm_hotels], h[~dup], lite], ignore_index=True)
    print(f"LiteAPI: {len(lite)} hotels added, {int(dup.sum())} duplicate OSM hotels dropped")
    return df


class Catalog:
    """One city's places, looked up by place_id. Rows are plain dicts (None for missing values)."""

    def __init__(self, df: pd.DataFrame):
        df = df.drop_duplicates("place_id").astype(object)
        self.df = df.where(df.notna(), None)
        self.rows = {r["place_id"]: r for r in self.df.to_dict("records")}

    def get(self, place_id):
        return self.rows.get(place_id)


@lru_cache(maxsize=32)
def load_catalog(city: str) -> Catalog:
    path = DATA / f"places_{city.lower()}.csv"
    if not path.exists():
        raise FileNotFoundError(f"no catalog for {city}: run python -m trip_planner.catalog {city}")
    return Catalog(pd.read_csv(path, dtype={"place_id": str}))


if __name__ == "__main__":
    # python -m trip_planner.catalog Mumbai            OSM places, plus LiteAPI hotels if LITEAPI_KEY is set
    # python -m trip_planner.catalog Mumbai --hotels   only (re)merge LiteAPI hotels into the existing catalog
    import os
    city = sys.argv[1] if len(sys.argv) > 1 else "Mumbai"
    path = DATA / f"places_{city.lower()}.csv"
    df = pd.read_csv(path, dtype={"place_id": str}) if "--hotels" in sys.argv else build(city)
    if os.getenv("LITEAPI_KEY"):
        df = merge_liteapi(df, city)
    DATA.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    print(df.kind.value_counts().head(12).to_string())
    print(f"{len(df)} places, {df.stars.notna().sum()} hotels with star ratings")
