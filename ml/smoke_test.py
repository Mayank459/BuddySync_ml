"""Call every endpoint of a *running* ML service over HTTP and check the answers make sense.

Run from ml/ (server started with: uvicorn app:app --port 8000):
    python smoke_test.py [http://localhost:8000]
Exit code 1 if any check fails.
"""
import json
import sys

import cv2
import numpy as np
import pandas as pd
import requests

from common.utils import DATA, load
import common.http  # noqa: F401  (adds X-Service-Token when ML_SERVICE_TOKEN is set)

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"  # not "localhost": ~2 s IPv6 fallback on Windows
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def post(path, **kw):
    return requests.post(BASE + path, timeout=120, **kw)


users, parts, events = load("users").set_index("user_id"), load("participants"), load("events")

# --- service -------------------------------------------------------------------------
check("health", requests.get(BASE + "/health", timeout=10).json() == {"ok": True})
check("openapi docs", requests.get(BASE + "/docs", timeout=10).status_code == 200)

# --- buddy matching ------------------------------------------------------------------
looking = parts[parts.status == "looking_for_buddy"]
viewer_row = looking[looking.user_id.map(users.filter_women_only)].iloc[0]  # a viewer with a hard filter
r = post("/v1/match/buddies", json={"user_id": int(viewer_row.user_id), "event_id": int(viewer_row.event_id)})
cands = r.json().get("candidates", [])
check("buddies: 200 + candidates", r.status_code == 200 and len(cands) > 0, f"{len(cands)} candidates")
check("buddies: women-only filter respected", all(users.gender[c["user_id"]] == "F" for c in cands))
check("buddies: everyone is at the same event",
      set(c["user_id"] for c in cands) <= set(parts[parts.event_id == viewer_row.event_id].user_id))
check("buddies: match % in 0-100 and reasons given", all(0 <= c["match_pct"] <= 100 and c["reasons"] for c in cands))
check("buddies: unknown user -> 404", post("/v1/match/buddies", json={"user_id": 999999, "event_id": 1}).status_code == 404)

# --- clans ---------------------------------------------------------------------------
busiest = int(looking.event_id.value_counts().index[0])
clans = post("/v1/clans/form", json={"event_id": busiest}).json()["clans"]
members = [m for c in clans for m in c["members"]]
check("clans: formed, sizes 3-4, nobody twice",
      clans and all(3 <= len(c["members"]) <= 4 for c in clans) and len(members) == len(set(members)),
      f"{len(clans)} clans")
check("clans: tiny event -> empty list", post("/v1/clans/form", json={"event_id": 999999}).json()["clans"] == [])

# --- event recs ----------------------------------------------------------------------
uid = int(viewer_row.user_id)
recs = post("/v1/recs/events", json={"user_id": uid, "limit": 10}).json()["events"]
ev = events.set_index("event_id")
check("recs: 200 + events", len(recs) > 0, f"{len(recs)} events")
check("recs: all in user's city and in the future",
      all(ev.city[e["event_id"]] == users.city[uid] and pd.Timestamp(e["start_time"]) >= pd.Timestamp.now() for e in recs))
check("recs: max 3 per category",
      pd.Series([ev.category[e["event_id"]] for e in recs]).value_counts().max() <= 3)
check("recs: excludes events user already joined",
      not {e["event_id"] for e in recs} & set(parts[parts.user_id == uid].event_id))

# --- event ingestion -----------------------------------------------------------------
e = events.iloc[5]
dup = post("/v1/events/ingest", json={"title": e.title.upper() + " | Tickets", "performers": e.performers,
                                      "city": e.city, "venue_lat": float(e.venue_lat) + 0.0002,
                                      "venue_lng": float(e.venue_lng), "start_time": e.start_time,
                                      "description": e.description}).json()
check("ingest: reformatted copy -> merge into same event",
      dup["action"] == "merge" and dup["duplicate_of"] == int(e.event_id), f"p={dup['duplicate_probability']}")
check("ingest: category right on the copy", dup["category"] == e.category, dup["category"])
new = post("/v1/events/ingest", json={"title": "Pottery Workshop with Meera", "performers": "Meera",
                                      "city": "Pune", "venue_lat": 18.40, "venue_lng": 73.70,
                                      "start_time": "2026-12-01T11:00:00", "description": "hands-on pottery"}).json()
check("ingest: unseen event -> new, workshop", new["action"] == "new" and new["category"] == "workshop", new["category"])

# --- moderation ----------------------------------------------------------------------
for text, want in [("kal gate 2 pe milte hain", "allow"), ("tu pagal hai kya bewakoof", "block"),
                   ("bhai advance bhej do 500 rs to rahul@okaxis", "flag"),
                   ("add me on whatsapp 9876543210", "nudge")]:
    got = post("/v1/moderate/message", json={"text": text}).json()["action"]
    check(f"moderate: '{text}' -> {want}", got == want, got)

# --- face verification ---------------------------------------------------------------
from sklearn.datasets import fetch_lfw_pairs  # noqa: E402  (cached locally by face_verification.train)

lfw = fetch_lfw_pairs(subset="test", color=True, resize=1.0, slice_=None, data_home=DATA)
px = lfw.pairs if lfw.pairs.max() > 1.5 else lfw.pairs * 255
jpg = lambda img: cv2.imencode(".jpg", np.ascontiguousarray(img[..., ::-1]).astype(np.uint8))[1].tobytes()


def verify(a, b):
    return post("/v1/verify/face", files=[("selfie", ("s.jpg", jpg(a), "image/jpeg")),
                                          ("profile_photos", ("p.jpg", jpg(b), "image/jpeg"))]).json()


same = [verify(*px[i]) for i in np.where(lfw.target == 1)[0][:20]]
diff = [verify(*px[i]) for i in np.where(lfw.target == 0)[0][:20]]
judged_diff = [d for d in diff if d["decision"] != "retry"]
check("face: same person mostly passes",
      sum(d["decision"] == "pass" for d in same) >= 15, f"{sum(d['decision'] == 'pass' for d in same)}/20 pass")
check("face: strangers never pass", all(d["decision"] != "pass" for d in diff),
      f"{len(judged_diff)} judged, {20 - len(judged_diff)} asked to retry (quality gate)")
check("face: liveness flagged as not checked", all(d["liveness_checked"] is False for d in same + diff))
bad = post("/v1/verify/face", files=[("selfie", ("x.txt", b"not an image", "text/plain")),
                                     ("profile_photos", ("p.jpg", jpg(px[0][1]), "image/jpeg"))])
check("face: non-image -> 400", bad.status_code == 400)

# --- trip planner --------------------------------------------------------------------
def follow(trip_id, job_id):
    """Read the SSE stream until the job is done; returns {event_type: [data, ...]}."""
    seen = {}
    with requests.get(f"{BASE}/v1/trips/{trip_id}/stream", params={"user_id": 7, "until_job": job_id},
                      stream=True, timeout=240) as s:
        kind = None
        for line in s.iter_lines(decode_unicode=True):
            if line.startswith("event:"):
                kind = line[7:]
            elif line.startswith("data:"):
                data = json.loads(line[6:])
                if data.get("job_id") == job_id:  # the stream replays earlier jobs' events too
                    seen.setdefault(kind, []).append(data)
    return seen


mumbai = events[events.city == "Mumbai"].iloc[0]
t = post("/v1/trips", json={"event_id": int(mumbai.event_id), "user_id": 7, "origin_city": "Pune"})
created = t.json()
check("trip: create -> 202 + queued draft", t.status_code == 202 and created["trip"]["version"] == 0, t.status_code)
if t.status_code == 202:
    tid = created["trip_id"]
    got = follow(tid, created["job_id"])
    trip = requests.get(f"{BASE}/v1/trips/{tid}", params={"user_id": 7}, timeout=10).json()
    slots = {i["slot"] for i in trip["items"]}
    check("trip: draft fills stay, show, return and explore", {"stay", "show", "return", "explore"} <= slots,
          f"model {got.get('message', [{}])[0].get('model')}")
    check("trip: every place card is a real catalog place", all(i["place"]["maps_url"] for i in trip["items"] if "place" in i))
    check("trip: costs are INR ranges", trip["costs"]["per_person"]["max"] >= trip["costs"]["per_person"]["min"] > 0)
    check("trip: non-member -> 403", requests.get(f"{BASE}/v1/trips/{tid}", params={"user_id": 999999}, timeout=10).status_code == 403)
    m = post(f"/v1/trips/{tid}/messages", json={"user_id": 7, "text": "Only 5 star hotels please."})
    got = follow(tid, m.json()["turn_id"])
    if "error" in got:
        check("trip chat: no LLM -> clear error event", "configured" in got["error"][0]["text"], "live chat NOT tested")
    else:
        trip = requests.get(f"{BASE}/v1/trips/{tid}", params={"user_id": 7}, timeout=10).json()
        check("trip chat: reply + 5-star preference saved",
              got.get("message") and trip["preferences"]["hotel_min_stars"] == 5, got["message"][0]["text"][:120])
check("trip: unknown event -> 404", post("/v1/trips", json={"event_id": 10**9, "user_id": 7,
                                                           "origin_city": "Pune"}).status_code == 404)
check("trip: unknown trip -> 404", requests.get(BASE + "/v1/trips/nope", params={"user_id": 7}, timeout=10).status_code == 404)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
