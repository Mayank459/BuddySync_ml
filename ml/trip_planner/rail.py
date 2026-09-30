"""Train fare ranges from the Railway Board basic fare chart (TAG-2026, w.e.f. 01.07.2025), design §7.

fare = basic fare for the distance slab + the 26 Dec 2025 revision (+2 paise/km, Mail/Express)
       + reservation fee + superfast charge (upper end only) + 5% GST on AC classes.
We have no timetable, so the rail distance is estimated as 1.15–1.40 × the straight-line distance:
the result is a range labelled 'estimate'. Tatkal and dynamic-fare trains cost more.
Build the table once from the PDF:  python -m trip_planner.rail path/to/Fares.pdf  → data/rail_fares.csv
Source: https://indianrailways.gov.in/railwayboard/uploads/directorate/coaching/TAG_2026/Fares.pdf
"""
import csv
import re
import sys
from functools import lru_cache

from common.utils import DATA, haversine_km
from trip_planner.geo import coords

PATH = DATA / "rail_fares.csv"
RAIL_FACTOR = (1.15, 1.40)  # rail km per straight-line km
REVISION_PER_KM = 0.02      # 26 Dec 2025: +2 paise/km on Mail/Express classes
SOURCE = "Railway Board fare chart TAG-2026 + Dec 2025 revision; rail distance estimated"
# class: (column(s) in the table, reservation fee, superfast charge, GST applies)
CLASSES = {"SL": (("sleeper",), 20, 30, False),
           "3A": (("ac3",), 40, 45, True),
           "2A": (("ac2_lean", "ac2_peak"), 50, 45, True)}
LABELS = {"SL": "Sleeper", "3A": "AC 3-tier", "2A": "AC 2-tier"}


def build(pdf_path):
    from pypdf import PdfReader
    pages = PdfReader(pdf_path).pages
    rows = []
    for page in pages[7:10]:  # Mail/Express non-AC: two slabs per line
        for m in re.finditer(r"(\d+)\s*[–-]\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", page.extract_text()):
            lo, hi, second, sleeper, first = map(int, m.groups())
            rows += [(lo, hi, "second", second), (lo, hi, "sleeper", sleeper), (lo, hi, "first", first)]
    ac_cols = ("cc", "ac3e", "ac3", "ac2_peak", "ac2_lean", "ac1_peak", "ac1_lean")
    for page in pages[10:16]:  # Mail/Express AC classes
        for line in page.extract_text().splitlines():
            m = re.match(r"^\s*(\d+)\s*[–-]\s*(\d+)" + r"\s+(\d+)" * 7 + r"\s*$", line)
            if m:
                v = list(map(int, m.groups()))
                rows += [(v[0], v[1], c, f) for c, f in zip(ac_cols, v[2:])]
    with open(PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["km_from", "km_to", "class", "basic_fare"])
        w.writerows(sorted(rows, key=lambda r: (r[2], r[0])))
    return len(rows)


@lru_cache
def table() -> dict:
    out = {}
    if not PATH.exists():
        return out
    with open(PATH) as f:
        for r in csv.DictReader(f):
            out.setdefault(r["class"], []).append((int(r["km_from"]), int(r["km_to"]), int(r["basic_fare"])))
    return out


def basic(cls: str, km: float):
    slabs = table().get(cls, [])
    if not slabs:
        return None
    km = max(km, slabs[0][0])
    return next((f for lo, hi, f in slabs if lo <= km <= hi), slabs[-1][2])


def fare(code: str, km: float, upper: bool) -> int | None:
    cols, reservation, superfast, gst = CLASSES[code]
    base = basic(cols[-1] if upper else cols[0], km)
    if base is None:
        return None
    total = base + REVISION_PER_KM * km + reservation + (superfast if upper else 0)
    return int(round(total * (1.05 if gst else 1), -1))


def quote(origin_city: str, dest_city: str) -> dict | None:
    a, b = coords(origin_city), coords(dest_city)
    if not a or not b or not table():
        return None
    straight = haversine_km(*a, *b)
    if straight < 40:  # same metro area: local train or cab, not a long-distance ticket
        return None
    lo_km, hi_km = straight * RAIL_FACTOR[0], straight * RAIL_FACTOR[1]
    classes = {code: {"min": fare(code, lo_km, False), "max": fare(code, hi_km, True)} for code in CLASSES}
    return {"rail_km": [int(lo_km), int(hi_km)], "classes": classes, "labels": LABELS, "source": SOURCE}


if __name__ == "__main__":
    print(build(sys.argv[1]), "fare rows written to", PATH)
