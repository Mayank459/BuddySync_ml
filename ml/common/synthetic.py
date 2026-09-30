"""Generate synthetic BuddySync data so every pipeline runs end to end.

Metrics on this data measure the *pipeline*, not real-world quality: the labels come
from a hidden rule below. Replace with real logs (see ML_IMPLEMENTATION_PLAN.md §2).

Run from ml/:  python -m common.synthetic
"""
import math
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from common.utils import DATA, haversine_km

random.seed(42)
rng = np.random.default_rng(42)

CITIES = {"Mumbai": (19.0760, 72.8777), "Delhi": (28.6139, 77.2090),
          "Bengaluru": (12.9716, 77.5946), "Pune": (18.5204, 73.8567),
          "Hyderabad": (17.3850, 78.4867)}
CITY_LANG = {"Mumbai": "marathi", "Pune": "marathi", "Bengaluru": "kannada",
             "Hyderabad": "telugu", "Delhi": "hindi"}
CATEGORIES = {
    "concert": ["bollywood music", "indie music", "edm", "rock", "hip hop", "sufi"],
    "comedy": ["stand-up comedy", "improv"],
    "sports": ["cricket", "football", "kabaddi"],
    "movie": ["bollywood films", "hollywood films", "anime"],
    "theatre": ["theatre", "musicals"],
    "workshop": ["photography", "pottery", "coding"],
    "festival": ["food festivals", "music festivals"],
    "conference": ["tech", "startups", "ai"],
}
QUIZ = {"vibe": ["front_row", "chill", "depends"], "arrive": ["together", "at_venue"],
        "after": ["home", "food", "party"], "punctuality": ["early", "on_time", "relaxed"],
        "energy": ["introvert", "ambivert", "extrovert"]}
HINGLISH = ["weekend pe plans chahiye", "ekdum mast vibe", "chalo saath chalte hain",
            "always up for a good show", "new city, new friends", ""]
PERFORMER_A = ["The Monsoon", "DJ Neon", "Kavya", "Blue Lantern", "Aarav", "Midnight Chai",
               "Saffron", "Riya", "Static Tides", "Mehfil"]
PERFORMER_B = ["Project", "Collective", "Band", "Trio", "Live", "Express", "Sessions"]
TEAMS = ["Mavericks", "Dynamos", "Tigers", "Strikers", "Royals", "Titans"]
VENUE_A = ["Skyline", "Harbour", "Lotus", "Royal", "Metro", "Garden", "Riverside", "Sunrise"]
VENUE_B = ["Arena", "Grounds", "Auditorium", "Club", "Hall"]
START = datetime(2026, 6, 1)


def jitter(lat, lng, sd_deg):
    return round(lat + rng.normal(0, sd_deg), 4), round(lng + rng.normal(0, sd_deg), 4)


def make_users(n=1500):
    rows = []
    for uid in range(1, n + 1):
        city = random.choice(list(CITIES))
        lat, lng = jitter(*CITIES[city], 0.05)
        cats = random.sample(list(CATEGORIES), k=2)
        tags = {t for c in cats for t in random.sample(CATEGORIES[c], k=min(2, len(CATEGORIES[c])))}
        gender = rng.choice(["F", "M", "X"], p=[0.45, 0.5, 0.05])
        age = int(rng.integers(18, 41))
        rows.append({
            "user_id": uid, "city": city, "lat": round(lat, 3), "lng": round(lng, 3),  # ~100 m, never raw GPS
            "age": age, "gender": gender,
            "interests": ";".join(sorted(tags)),
            "bio": f"Love {' and '.join(sorted(tags)[:2])}. {random.choice(HINGLISH)}".strip(),
            **{k: random.choice(v) for k, v in QUIZ.items()},
            "budget": int(rng.integers(1, 4)),
            "languages": ";".join(sorted({"english", random.choice(["hindi", CITY_LANG[city]])})),
            "group_pref": random.choice(["duo", "group", "either"]),
            "verified": bool(rng.random() < 0.6),
            "filter_verified_only": bool(rng.random() < 0.2),
            "filter_women_only": bool(gender == "F" and rng.random() < 0.3),
            "age_min": max(18, age - int(rng.integers(4, 10))), "age_max": age + int(rng.integers(4, 10)),
        })
    return pd.DataFrame(rows)


def event_title(cat, tag, performer, city):
    return {
        "concert": random.choice([f"{performer} Live in {city}", f"{performer} - {tag.title()} Night"]),
        "comedy": random.choice([f"{performer}: Stand-up Special", f"Comedy Night ft. {performer}"]),
        "sports": f"{city} {random.choice(TEAMS)} vs {random.choice(TEAMS)} ({tag})",
        "movie": f"{tag.title()} Marathon: {performer} Premiere",
        "theatre": f"{performer} presents a {tag} evening",
        "workshop": f"{tag.title()} Workshop with {performer}",
        "festival": f"{city} {tag.title()} {random.choice(['Fest', 'Carnival'])}",
        "conference": f"{city} {tag.title()} Summit 2026",
    }[cat]


def make_events(n=600):
    venues = {c: [(f"{random.choice(VENUE_A)} {random.choice(VENUE_B)}", *jitter(*CITIES[c], 0.05))
                  for _ in range(8)] for c in CITIES}
    rows = []
    for eid in range(1, n + 1):
        cat = random.choice(list(CATEGORIES))
        tag = random.choice(CATEGORIES[cat])
        city = random.choice(list(CITIES))
        vname, vlat, vlng = random.choice(venues[city])
        performer = f"{random.choice(PERFORMER_A)} {random.choice(PERFORMER_B)}"
        start = START + timedelta(days=int(rng.integers(0, 240)), hours=int(rng.integers(17, 22)))
        rows.append({
            "event_id": eid, "category": cat, "tags": tag, "city": city, "performers": performer,
            "venue_name": vname, "venue_lat": vlat, "venue_lng": vlng,
            "title": event_title(cat, tag, performer, city),
            "description": f"An evening of {tag} in {city}. {random.choice(HINGLISH)}".strip(),
            "start_time": start, "gates_open": start - timedelta(minutes=90),
            "end_time": start + timedelta(hours=3), "price_band": int(rng.integers(1, 4)),
            "hotness": float(rng.lognormal(0, 0.8)),
        })
    ev = pd.DataFrame(rows)
    # siblings: a different show at the same venue on the same day (hard negatives for dedup)
    sib = ev.sample(frac=0.1, random_state=1).copy()
    sib["event_id"] = range(n + 1, n + 1 + len(sib))
    sib["performers"] = [f"{random.choice(PERFORMER_A)} {random.choice(PERFORMER_B)}" for _ in range(len(sib))]
    sib["title"] = [event_title(c, t, p, ci) for c, t, p, ci in zip(sib.category, sib.tags, sib.performers, sib.city)]
    for col in ["start_time", "gates_open", "end_time"]:
        sib[col] = sib[col] + timedelta(hours=2)
    return pd.concat([ev, sib], ignore_index=True)


def make_participants(users, events, per_user=8):
    rows = []
    for u in users.itertuples():
        pool = events[events.city == u.city]
        tags = set(u.interests.split(";"))
        cats = {c for c, ts in CATEGORIES.items() if tags & set(ts)}
        w = np.array([e.hotness * math.exp(2.0 * (e.tags in tags) + 1.0 * (e.category in cats))
                      for e in pool.itertuples()])
        picks = rng.choice(len(pool), size=min(per_user, len(pool)), replace=False, p=w / w.sum())
        for i in picks:
            e = pool.iloc[i]
            rows.append({"event_id": e.event_id, "user_id": u.user_id,
                         "status": rng.choice(["looking_for_buddy", "going", "interested"], p=[0.5, 0.25, 0.25]),
                         "ts": e.start_time - timedelta(days=int(rng.integers(1, 30)))})
    return pd.DataFrame(rows)


def quiz_agree(a, b):
    same = [a[k] == b[k] for k in ["vibe", "arrive", "after", "punctuality"]]
    return (sum(same) + 1 - abs(a["budget"] - b["budget"]) / 2) / 5


def hard_ok(a, b):
    """Does b pass a's hard filters?"""
    return ((not a["filter_women_only"] or b["gender"] == "F")
            and (not a["filter_verified_only"] or b["verified"])
            and a["age_min"] <= b["age"] <= a["age_max"])


def make_buddy_requests(users, participants, shown=10):
    """Hidden truth: interests + quiz agreement matter most, distance a bit, filters are absolute."""
    u = users.set_index("user_id").to_dict("index")
    sig = lambda x: 1 / (1 + math.exp(-x))
    rows = []
    looking = participants[participants.status == "looking_for_buddy"]
    for event_id, grp in looking.groupby("event_id"):
        ids = grp.user_id.tolist()
        for viewer in ids:
            for cand in random.sample([i for i in ids if i != viewer], k=min(shown, len(ids) - 1)):
                a, b = u[viewer], u[cand]
                ta, tb = set(a["interests"].split(";")), set(b["interests"].split(";"))
                jac = len(ta & tb) / len(ta | tb)
                d = haversine_km(a["lat"], a["lng"], b["lat"], b["lng"])
                aff = 2.5 * jac + 2.0 * quiz_agree(a, b) - 0.08 * d - 1.6 + rng.normal(0, 0.5)
                p_send = sig(aff + 0.4 * b["verified"]) * hard_ok(a, b)
                p_acc = sig(aff + rng.normal(0, 0.5)) * hard_ok(b, a)
                sent = rng.random() < p_send
                status = "none"
                if sent:
                    status = "pending" if rng.random() < 0.15 else ("accepted" if rng.random() < p_acc else "declined")
                rows.append({"viewer_id": viewer, "candidate_id": cand, "event_id": event_id,
                             "sent": bool(sent), "status": status})
    return pd.DataFrame(rows)


def make_listings(events):
    """Each event arrives from 1-2 sources; the second copy is formatted differently."""
    def restyle(t):
        t = random.choice([t.upper(), t.replace(" Live in ", " LIVE @ "), t + " | Tickets", t.replace(":", " -"), t])
        return t
    rows, lid = [], 1
    for e in events.itertuples():
        copies = [("partner_a", e.title, 0)]
        if rng.random() < 0.6:
            copies.append(("partner_b", restyle(e.title), random.choice([0, 0, 30, -30])))
        for source, title, shift in copies:
            lat, lng = (e.venue_lat, e.venue_lng) if source == "partner_a" else jitter(e.venue_lat, e.venue_lng, 0.0003)
            rows.append({"listing_id": lid, "event_id": e.event_id, "source": source, "title": title,
                         "performers": e.performers, "category": e.category, "city": e.city,
                         "venue_lat": lat, "venue_lng": lng,
                         "start_time": e.start_time + timedelta(minutes=shift),
                         "description": e.description})
            lid += 1
    return pd.DataFrame(rows)


def make_messages():
    clean = ["kal show pe milte hain", "what time are the gates opening?", "I'll reach by 6",
             "let's meet at gate 2", "did you get the tickets?", "bhai kitne baje nikalna hai",
             "the opening act was amazing", "want to grab food after?", "running 10 min late, sorry",
             "which section are you in?", "mast concert tha yaar", "see you there!",
             "can we meet near the metro station", "is parking available at the venue?",
             "haha that was so funny", "thanks for coming along", "maine ticket le liya",
             "add me on whatsapp, 9876543210", "send the advance for the ticket to my upi"]
    toxic = ["you are such an idiot", "shut up loser", "tu pagal hai kya", "bakwas mat kar",
             "get lost you fool", "chup kar bewakoof", "nobody wants you here", "you are pathetic",
             "stupid girl stop texting", "abey gadhe", "you are so dumb", "go away creep",
             "tere jaisa bewakoof nahi dekha", "worst person ever, get lost"]
    fillers = ["", " ", "!!", " yaar", " bro", " lol", " ??", " .", " seriously"]
    rows = [{"text": random.choice(clean) + random.choice(fillers), "label": 0} for _ in range(900)]
    rows += [{"text": random.choice(toxic) + random.choice(fillers), "label": 1} for _ in range(300)]
    return pd.DataFrame(rows).sample(frac=1, random_state=0)


def main():
    DATA.mkdir(exist_ok=True)
    users, events = make_users(), make_events()
    participants = make_participants(users, events)
    tables = {"users": users, "events": events, "participants": participants,
              "buddy_requests": make_buddy_requests(users, participants),
              "listings": make_listings(events), "messages": make_messages()}
    for name, df in tables.items():
        df.to_csv(DATA / f"{name}.csv", index=False)
        print(f"{name:15s} {len(df):7d} rows")


if __name__ == "__main__":
    main()
