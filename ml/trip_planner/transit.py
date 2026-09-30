"""Last metro / suburban train of the night per city, for the "show ends after the last metro" warning.

City-level and conservative: where lines differ we keep the earliest last departure from the terminals,
because we don't yet know which line serves each venue. Researched Sep 2026; every row has a source.
`verified` is False where only unofficial or older sources were found. Operators often run later on
big event nights (IPL, festivals), so the warning always says to check.
ponytail: per-line table + nearest station per venue once we map stations; refresh these every quarter.
"""
from datetime import time

LAST_METRO = {
    "Mumbai": {"last": time(22, 30), "verified": True,
               "note": "Metro Lines 2A, 7 and 3 end about 22:30; Line 1 about 23:25; suburban locals run until about midnight",
               "source": "https://www.timeout.com/mumbai/news/mumbai-aqua-line-extends-its-operations-for-ganesh-chaturthi-090726"},
    "Delhi": {"last": time(23, 0), "verified": True,
              "note": "last trains leave the terminals about 23:00; DMRC often extends on match nights",
              "source": "https://www.businesstoday.in/latest/trends/story/delhi-metro-to-stay-open-late-for-ipl-fans-last-train-timings-extended-for-april-8-match-check-full-time-table-524475-2026-04-08"},
    "Bengaluru": {"last": time(22, 42), "verified": False,
                  "note": "Namma Metro last trains about 22:42–23:05 (Yellow Line from Bommasandra earliest)",
                  "source": "https://bengalurumetro.in/bangalore-metro-timings.html"},
    "Hyderabad": {"last": time(23, 0), "verified": True,
                  "note": "Red and Blue lines 23:00 from terminals, Green 23:35",
                  "source": "https://hmrl.co.in/train-timings/"},
    "Pune": {"last": time(23, 0), "verified": True,
             "note": "Purple and Aqua lines run until 23:00",
             "source": "https://www.punekarnews.in/pune-metro-extends-operating-hours-to-11-pm-starting-january-2025/"},
    "Chennai": {"last": time(23, 0), "verified": False,
                "note": "Blue and Green lines about 23:00",
                "source": "https://chennaimetrorail.org/time-table/"},
    "Kolkata": {"last": time(22, 30), "verified": True,
                "note": "Blue and Green lines 22:30 (Mon–Sun, experimental timings from 27 Jul 2026)",
                "source": "https://www.newkerala.com/news/a/now-kolkata-metro-trains-start-running-from-am-876.htm"},
}
