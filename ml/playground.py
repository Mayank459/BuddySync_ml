"""Streamlit playground for trying every ML endpoint by hand.

Start the API first, then this (both from ml/):
    python -m uvicorn app:app --port 8765
    python -m streamlit run playground.py
"""
import importlib.util
import os
import time

import pandas as pd
import requests
import streamlit as st

from common.utils import DATA, load
import common.http  # noqa: F401  (adds X-Service-Token when ML_SERVICE_TOKEN is set)

HAS_CAM = importlib.util.find_spec("cv2") is not None  # webcam flows need OpenCV; absent on the hosted demo

st.set_page_config(page_title="BuddySync ML playground", layout="wide")
BASE = st.sidebar.text_input("API base URL", os.getenv("API_URL", "https://buddysync-ml.onrender.com"))  # local: http://127.0.0.1:8765 (127.0.0.1, not localhost → 2 s delay on Windows)


@st.cache_data
def data():
    return load("users"), load("events"), load("participants")


users, events, parts = data()


def call(method, path, **kw):
    """Call the API; show status + raw JSON; return (status, body)."""
    try:
        r = requests.request(method, BASE + path, timeout=180, **kw)
    except requests.ConnectionError:
        st.error(f"Can't reach {BASE}. Start the API: `python -m uvicorn app:app --port 8765`")
        return None, None
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
    (st.success if r.ok else st.error)(f"{method} {path} → {r.status_code}  ({r.elapsed.total_seconds() * 1000:.0f} ms)")
    with st.expander("Raw response"):
        st.json(body)
    return r.status_code, body


try:
    ok = requests.get(BASE + "/health", timeout=3).json().get("ok")
except Exception:
    ok = None
if ok:
    st.sidebar.success("API is up")
else:
    st.sidebar.error("API is down")
st.sidebar.markdown(f"[Swagger docs]({BASE}/docs) · data: synthetic CSVs in `ml/data/`")

tabs = st.tabs(["Buddies", "Clans", "Event recs", "Ingest", "Face", "Moderation", "Trip planner"])
looking = parts[parts.status == "looking_for_buddy"]
title = events.set_index("event_id").title


def event_label(eid):
    return f"{eid} · {title.get(eid, '?')}"


# --- Buddies -----------------------------------------------------------------------------
with tabs[0]:
    busy = looking.event_id.value_counts().index[:50]
    eid = st.selectbox("Event (top 50 by people looking for a buddy)", busy, format_func=event_label, key="b_ev")
    at_event = looking[looking.event_id == eid].user_id
    uid = st.selectbox("Viewer", at_event, key="b_user")
    u = users.set_index("user_id").loc[uid]
    st.caption(f"Viewer: {u.gender}, {u.age}, {u.city} · interests: {u.interests} · vibe {u.vibe}, after-show {u.after}"
               f" · filters: women-only={u.filter_women_only}, verified-only={u.filter_verified_only}, age {u.age_min}-{u.age_max}")
    limit = st.slider("limit", 1, 30, 10, key="b_lim")
    if st.button("Find buddies", type="primary"):
        _, body = call("POST", "/v1/match/buddies", json={"user_id": int(uid), "event_id": int(eid), "limit": limit})
        if body and body.get("candidates"):
            info = users.set_index("user_id")[["gender", "age", "interests", "vibe", "after", "verified"]]
            df = pd.DataFrame(body["candidates"]).set_index("user_id").join(info)
            df["reasons"] = df.reasons.str.join(" · ")
            st.dataframe(df.drop(columns="components"), use_container_width=True)
        elif body is not None:
            st.info("No candidates.")

# --- Clans -------------------------------------------------------------------------------
with tabs[1]:
    eid = st.selectbox("Event", looking.event_id.value_counts().index[:50], format_func=event_label, key="c_ev")
    c1, c2 = st.columns(2)
    size = c1.slider("max clan size", 3, 6, 4)
    min_score = c2.slider("min pair score", 0.0, 1.0, 0.4, 0.05)
    if st.button("Form clans", type="primary"):
        _, body = call("POST", "/v1/clans/form", json={"event_id": int(eid), "size": size, "min_score": min_score})
        if body and body.get("clans"):
            st.write(f"{len(body['clans'])} clans, {sum(len(c['members']) for c in body['clans'])} people placed")
            st.dataframe(pd.DataFrame(body["clans"]), use_container_width=True)
        elif body is not None:
            st.info("No clans (fewer than 3 compatible people who want a group).")

# --- Event recs --------------------------------------------------------------------------
with tabs[2]:
    uid = st.number_input("user_id", 1, int(users.user_id.max()), 1, key="r_user")
    u = users.set_index("user_id").loc[uid]
    st.caption(f"{u.city} · interests: {u.interests}")
    c1, c2, c3 = st.columns(3)
    limit = c1.slider("limit", 1, 30, 10, key="r_lim")
    days = c2.slider("days_ahead", 7, 240, 60)
    per_cat = c3.slider("max_per_category", 1, 10, 3)
    if st.button("Recommend events", type="primary"):
        _, body = call("POST", "/v1/recs/events", json={"user_id": int(uid), "limit": limit, "days_ahead": days,
                                                         "max_per_category": per_cat})
        if body and body.get("events"):
            df = pd.DataFrame(body["events"]).set_index("event_id").join(events.set_index("event_id")[["category", "tags"]])
            df["reasons"] = df.reasons.str.join(" · ")
            st.dataframe(df, use_container_width=True)
        elif body is not None:
            st.info("No upcoming events in this user's city within the window.")

# --- Ingest ------------------------------------------------------------------------------
with tabs[3]:
    st.write("Paste a listing, or copy an existing event with messy formatting to test duplicate detection.")
    if st.button("Prefill: messy copy of a random existing event"):
        e = events.sample(1).iloc[0]
        st.session_state.update(i_title=e.title.upper() + " | Tickets", i_perf=e.performers, i_city=e.city,
                                i_lat=float(e.venue_lat) + 0.0002, i_lng=float(e.venue_lng),
                                i_start=str(e.start_time).replace(" ", "T"), i_desc=e.description)
        st.info(f"Expect: merge into event {e.event_id} ({e.category})")
    c1, c2 = st.columns(2)
    body = {"title": c1.text_input("title", "Pottery Workshop with Meera", key="i_title"),
            "performers": c2.text_input("performers", "Meera", key="i_perf"),
            "city": c1.selectbox("city", sorted(events.city.unique()), key="i_city"),
            "start_time": c2.text_input("start_time (ISO)", "2026-12-01T11:00:00", key="i_start"),
            "venue_lat": c1.number_input("venue_lat", value=18.52, format="%.5f", key="i_lat"),
            "venue_lng": c2.number_input("venue_lng", value=73.85, format="%.5f", key="i_lng"),
            "description": st.text_area("description", "hands-on pottery", key="i_desc")}
    if st.button("Ingest", type="primary"):
        _, r = call("POST", "/v1/events/ingest", json=body)
        if r and "action" in r:
            c1, c2, c3 = st.columns(3)
            c1.metric("action", r["action"])
            c2.metric("category", r["category"], f"{r['category_confidence']:.0%} confident")
            c3.metric("duplicate of", r["duplicate_of"] or "—", f"p={r['duplicate_probability']}")

# --- Face --------------------------------------------------------------------------------
def webcam_frame():
    import cv2
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    for _ in range(8):  # let exposure settle
        ok, frame = cam.read()
    cam.release()
    return frame if ok else None


def liveness_flow(view, base, max_attempts=3):
    """One continuous flow on this PC's webcam: the camera stays open until the check passes.
    Profile photo (taken automatically when you hold still) → an instruction pops up → do it → OK ✓ next /
    not detected → retake → all done → the server verifies the recording (preview stays live) → pass: done;
    otherwise a fresh challenge. Returns (result, attempts, photo_jpg)."""
    import concurrent.futures as cf
    import tempfile

    import cv2
    from face_verification.liveness import MAX_VIDEO_S, LiveGuide, frame_signals, landmarks
    STEP_S, ATTEMPT_S, GREEN, YELLOW, RED, GREY = 7.0, MAX_VIDEO_S - 3, (80, 220, 80), (0, 220, 255), (60, 60, 255), (170, 170, 170)
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        return {"error": "Couldn't open the webcam (is another app using it?)"}, 0, None
    pool = cf.ThreadPoolExecutor(1)

    def draw(img, lines):
        img = img.copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 16 + 32 * len(lines)), (0, 0, 0), -1)
        for n, (line, color) in enumerate(lines):
            cv2.putText(img, line, (14, 34 + n * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
        view.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")

    def frame_and_signals():
        ok, f = cam.read()
        if not ok:
            raise RuntimeError("the webcam stopped sending frames")
        face, blend, hands, n_faces = landmarks(f)
        return f, frame_signals(face, blend, hands), n_faces

    def hold(seconds, lines):  # keep the preview live while showing a message
        end = time.time() + seconds
        while time.time() < end:
            f, _, _ = frame_and_signals()
            draw(f, lines)

    try:
        # 1. profile photo: taken by itself once one face has looked straight at the camera for ~1 s
        photo, steady, t0, n = st.session_state.get("cam_photo"), 0, time.time(), 0
        while photo is None:
            f, s, n_faces = frame_and_signals()
            n += 1
            good = s.get("face") and n_faces == 1 and s.get("eyes_open") and abs(s.get("yaw", 1)) < 0.08
            steady = steady + 1 if good else 0
            hint = ("Hold still..." if good else "No face found" if not s.get("face")
                    else "Only you in the frame" if n_faces > 1 else "Look straight at the camera")
            draw(f, [("Profile photo", YELLOW), (hint, GREEN if good else RED), ("#" * steady, GREEN)])
            if steady >= 15:
                photo = cv2.imencode(".jpg", f)[1].tobytes()
                st.session_state.cam_photo = photo
        fps = max(5.0, min(30.0, n / max(time.time() - t0, 0.1))) if n > 5 else 15.0
        hold(1.0, [("Profile photo taken  OK", GREEN)])

        result = None
        for attempt in range(1, max_attempts + 1):
            # 2. a fresh random challenge each attempt (each one works only once)
            ch = requests.post(f"{base}/v1/verify/challenge", timeout=10).json()
            prompts = {x["action"]: x["prompt"] for x in ch["steps"]}
            guide = LiveGuide([x["action"] for x in ch["steps"]])
            path = os.path.join(tempfile.gettempdir(), f"liveness_attempt{attempt}.mp4")
            f, _, _ = frame_and_signals()
            out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (f.shape[1], f.shape[0]))
            t0, step_t0, note, finished_at = time.time(), None, None, None
            while True:
                f, s, n_faces = frame_and_signals()
                t = time.time() - t0
                out.write(f)  # the recording has no overlay
                if step_t0 is None and len(guide.base_yaws) >= 5:
                    step_t0 = t  # start pose captured: the first instruction pops up
                if guide.feed(s, t):
                    note, step_t0 = ("OK  -  next:", GREEN, t + 0.9), t
                elif step_t0 is not None and not guide.finished and t - step_t0 > STEP_S:
                    note, step_t0 = ("Not detected  -  retake:", RED, t + 1.8), t  # retake the same step
                    guide.run, guide.eyes_were_open = 0, False
                lines = [(f"Attempt {attempt}/{max_attempts}   step {min(guide.i + 1, len(guide.steps))}"
                          f"/{len(guide.steps)}", GREY)]
                if not s.get("face"):
                    lines.append(("No face: move into the frame", RED))
                elif n_faces > 1:
                    lines.append(("Only you should be in the frame", RED))
                elif step_t0 is None:
                    lines.append(("Look straight at the camera...", YELLOW))
                elif guide.finished:
                    finished_at = finished_at or t
                    lines.append(("All steps done: look at the camera", GREEN))
                else:
                    if note and t < note[2]:
                        lines.append((note[0], note[1]))
                    lines.append((f">> {prompts[guide.current]}", YELLOW))
                    lines.append((f"   {STEP_S - (t - step_t0):3.0f}s", GREY))
                draw(f, lines)
                if finished_at is not None and t - finished_at > 1.5:
                    break  # a final frontal look for the face match
                if t > ATTEMPT_S:
                    break  # the server accepts at most MAX_VIDEO_S seconds
            out.release()
            if not guide.finished:
                hold(2.0, [("Time's up: new instructions coming", RED)])
                continue

            # 3. the server verifies the recording; the preview stays live meanwhile
            with open(path, "rb") as fh:
                video = fh.read()
            job = pool.submit(lambda: requests.post(
                f"{base}/v1/verify/liveness", timeout=120, data={"challenge_id": ch["challenge_id"]},
                files=[("video", ("webcam.mp4", video, "video/mp4")),
                       ("profile_photos", ("webcam.jpg", photo, "image/jpeg"))]).json())
            while not job.done():
                f, _, _ = frame_and_signals()
                draw(f, [("Verifying on the server...", YELLOW)])
            result = job.result()
            if result.get("decision") == "pass":
                hold(2.5, [("VERIFIED  OK", GREEN), (f"face match {result.get('best_similarity')}", GREEN)])
                return result, attempt, photo
            if result.get("decision") == "review":
                hold(2.5, [("Sent for a human review", YELLOW)])
                return result, attempt, photo
            reason = result.get("reason") or result.get("detail") or result.get("decision")
            hold(2.5, [(f"Not passed: {reason}", RED), ("Let's try again...", YELLOW)])
        return result or {"decision": "retry", "reason": "steps not completed in time"}, max_attempts, photo
    finally:
        cam.release()
        pool.shutdown(wait=False)


with tabs[4]:
    mode = st.radio("Mode", ["Liveness (video challenge)", "Photo match only (no liveness)"], horizontal=True)
    use_cam = st.toggle("Use this PC's webcam (photo + challenge video)", value=HAS_CAM, disabled=not HAS_CAM,
                        key="f_cam", help=None if HAS_CAM else "Unavailable on this hosted demo — upload photos/a video instead.")
    photos = [] if use_cam else st.file_uploader("Profile photos", ["jpg", "jpeg", "png"], accept_multiple_files=True,
                                                 key="f_photos")
    if use_cam:
        if st.session_state.get("cam_photo") and st.button("Retake profile photo"):
            st.session_state.pop("cam_photo")
        if st.button("📸 Take profile photo now (optional: the check takes one itself)"):
            import cv2
            frame = webcam_frame()
            if frame is None:
                st.error("No webcam frame: is another app using the camera?")
            else:
                st.session_state.cam_photo = cv2.imencode(".jpg", frame)[1].tobytes()
        if st.session_state.get("cam_photo"):
            st.image(st.session_state.cam_photo, width=160, caption="profile photo (webcam)")
        profile_files = [("profile_photos", ("webcam.jpg", st.session_state.cam_photo, "image/jpeg"))] \
            if st.session_state.get("cam_photo") else []
    else:
        for p in photos or []:
            st.image(p, width=110)
        profile_files = [("profile_photos", (p.name, p.getvalue(), p.type)) for p in photos or []]

    if mode.startswith("Liveness") and use_cam:
        st.caption("One flow, camera stays open until you pass: profile photo (hold still) → an instruction pops up → "
                   "do it → OK, next (or retake) → the server verifies → pass, or new instructions.")
        if st.button("🔴 Start liveness check", type="primary"):
            view = st.empty()
            r, attempts, photo = liveness_flow(view, BASE)
            view.empty()
            if "error" in r:
                st.error(r["error"])
            elif "decision" in r:
                (st.success if r["decision"] == "pass" else st.warning)(
                    f"{r['decision'].upper()} after {attempts} attempt(s)" + (f": {r.get('reason')}" if r.get("reason") else ""))
                c1, c2, c3 = st.columns(3)
                c1.metric("decision", r["decision"], r.get("reason"))
                c2.metric("same person throughout", str(r.get("same_person_throughout")), f"min {r.get('min_continuity')}")
                c3.metric("face match", r.get("best_similarity", "—"), f"coverage {r.get('identity_coverage')}")
                if r.get("liveness"):
                    st.dataframe(pd.DataFrame(r["liveness"]["steps"]), use_container_width=True)
                with st.expander("Raw response"):
                    st.json(r)
            else:
                st.error(r)
    elif mode.startswith("Liveness"):
        st.caption("1) Get a challenge  2) record ONE short video doing the steps in order (webcam/phone camera app)"
                   "  3) upload it before the challenge expires. Each challenge works once.")
        if st.button("Get challenge"):
            _, c = call("POST", "/v1/verify/challenge")
            if c and "challenge_id" in c:
                st.session_state.challenge = c
        c = st.session_state.get("challenge")
        if c:
            st.info(c["instructions"])
            for k, step in enumerate(c["steps"], 1):
                st.markdown(f"**{k}. {step['prompt']}**")
            st.caption(f"challenge {c['challenge_id']} · expires {c['expires_in_s']} s after issue")
        video = st.file_uploader("Challenge video", ["mp4", "mov", "webm", "avi", "mkv"], key="f_video")
        if st.button("Verify liveness", type="primary"):
            if not (c and video and photos):
                st.warning("Need a challenge, a video and at least one profile photo.")
            else:
                with st.spinner("Analysing video…"):
                    _, r = call("POST", "/v1/verify/liveness", data={"challenge_id": c["challenge_id"]},
                                files=[("video", (video.name, video.getvalue(), video.type or "video/mp4"))] + profile_files)
                st.session_state.pop("challenge", None)  # used up either way
                if r and "decision" in r:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("decision", r["decision"], r.get("reason"))
                    c2.metric("same person throughout", str(r.get("same_person_throughout")), f"min {r.get('min_continuity')}")
                    c3.metric("face match", r.get("best_similarity", "—"), f"coverage {r.get('identity_coverage')}")
                    st.dataframe(pd.DataFrame(r["liveness"]["steps"]), use_container_width=True)
    else:
        st.warning("No liveness: a photo of a photo can pass. Testing face matching only.")
        selfie = st.camera_input("Selfie (webcam)") if st.toggle("Use webcam") else st.file_uploader("Selfie", ["jpg", "jpeg", "png"])
        if selfie:
            st.image(selfie, width=200)
        if st.button("Verify photo", type="primary"):
            if not (selfie and photos):
                st.warning("Need a selfie and at least one profile photo.")
            else:
                files = [("selfie", (selfie.name or "selfie.jpg", selfie.getvalue(), selfie.type or "image/jpeg"))]
                _, r = call("POST", "/v1/verify/face", files=files + profile_files)
                if r and "decision" in r:
                    st.metric("decision", r["decision"], r.get("reason") or f"similarity {r.get('best_similarity')}")

# --- Moderation --------------------------------------------------------------------------
with tabs[5]:
    examples = ["kal gate 2 pe milte hain", "tu pagal hai kya bewakoof", "bhai advance bhej do 500 rs to rahul@okaxis",
                "add me on whatsapp 9876543210", "check this out www.example.com"]
    pick = st.selectbox("Example", ["(type your own)"] + examples)
    text = st.text_area("Message", "" if pick == "(type your own)" else pick)
    if st.button("Moderate", type="primary") and text.strip():
        _, r = call("POST", "/v1/moderate/message", json={"text": text})
        if r and "action" in r:
            c1, c2, c3 = st.columns(3)
            c1.metric("action", r["action"])
            c2.metric("toxicity", r["toxicity"])
            c3.metric("rules", ", ".join(r["rules"]) or "none")

# --- Trip planner ------------------------------------------------------------------------
def follow(trip_id, job_id, uid, box):
    """Show the planner's progress from the SSE stream until the job is done."""
    import json
    out = {}
    try:
        with requests.get(f"{BASE}/v1/trips/{trip_id}/stream", params={"user_id": uid, "until_job": job_id},
                          stream=True, timeout=300) as s:
            kind = None
            for line in s.iter_lines(decode_unicode=True):
                if line.startswith("event:"):
                    kind = line[7:]
                elif line.startswith("data:"):
                    data = json.loads(line[6:])
                    out.setdefault(kind, []).append(data)
                    if kind in ("status", "tool"):
                        box.update(label=data["text"])
                    elif kind == "error":
                        box.update(label=data["text"], state="error")
    except requests.RequestException as e:
        st.error(f"stream failed: {e}")
    return out


PRICE_TAG = {"estimate": "est", "range": "range", "verified": "verified"}
with tabs[6]:
    with_catalog = {p.stem.removeprefix("places_") for p in DATA.glob("places_*.csv")}
    planned = events[events.city.str.lower().isin(with_catalog)].sort_values("hotness", ascending=False).event_id
    c1, c2, c3 = st.columns([3, 1, 1])
    eid = c1.selectbox("Event (cities with a places catalog; the demo event is first)", planned,
                       format_func=event_label, key="t_ev")
    origin = c2.text_input("Origin city", "Pune")
    uid = int(c3.number_input("As user_id", value=7, step=1))
    p1, p2, p3, p4, p5 = st.columns(5)
    band = p1.selectbox("Budget", ["budget", "balanced", "premium"], index=1)
    lang = p2.selectbox("Reply language", ["en", "hinglish", "hi"])
    extra = p3.number_input("Extra days", 0, 4, 0)
    veg = p4.checkbox("Veg only")
    clan_ids = p5.text_input("Clan members (user_id:city, …)", "", placeholder="8:Nashik, 9:Surat")
    if st.button("Start new trip", type="primary"):
        members = [{"user_id": int(x.split(":")[0]), "origin_city": (x.split(":") + ["Pune"])[1].strip()}
                   for x in clan_ids.split(",") if x.strip()]
        code, r = call("POST", "/v1/trips", json={"event_id": int(eid), "user_id": uid, "origin_city": origin,
                                                   "members": members, "clan_id": 1 if members else None,
                                                   "prefs": {"budget_band": band, "lang": lang,
                                                             "extra_days": int(extra), "veg_only": veg}})
        if code == 202:
            with st.status("Planning your trip…") as box:
                follow(r["trip_id"], r["job_id"], uid, box)
            st.session_state.trip_id = r["trip_id"]

    tid = st.session_state.get("trip_id")
    if tid:
        t = requests.get(f"{BASE}/v1/trips/{tid}", params={"user_id": uid}, timeout=30).json()
        if "items" not in t:
            st.error(t)
            st.stop()
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Chat")
            msgs = requests.get(f"{BASE}/v1/trips/{tid}/messages", params={"user_id": uid}, timeout=30).json()["messages"]
            for m in msgs:
                with st.chat_message(m["role"]):
                    st.write(m["content"])
                    if m.get("model"):
                        st.caption(m["model"])
            if msg := st.chat_input("e.g. Only 5 star hotels · veg only · ek din aur rukna hai"):
                code, r = call("POST", f"/v1/trips/{tid}/messages", json={"user_id": uid, "text": msg})
                if code == 202:
                    with st.status("Thinking… (free tiers can take up to a minute)") as box:
                        got = follow(tid, r["turn_id"], uid, box)
                    if "error" in got and "configured" in got["error"][0]["text"]:
                        st.info("No LLM configured on the API. Set one before starting uvicorn, e.g. free Gemini:")
                        st.code("export LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/\n"
                                "export LLM_MODEL=gemini-3.8-flash,gemini-flash-lite-latest\n"
                                "export LLM_API_KEY=<your key>   # or ANTHROPIC_API_KEY for Claude", language="bash")
                    else:
                        st.rerun()
        with right:
            st.subheader(f"Trip {tid} · v{t['version']}")
            st.caption(f"{t['event']['title']} @ {t['event']['venue_name']}, {t['event']['city']}")
            cost = t["costs"]["per_person"]
            st.metric("Per person (estimate)", f"₹{cost['min']:,}–{cost['max']:,}",
                      f"{t['costs']['members']} people · {t['costs']['rooms']} room(s) · {t['costs']['nights']} night(s)",
                      delta_color="off")
            if len(t["members"]) > 1:
                st.caption("Members: " + ", ".join(f"{m['user_id']} ({m['origin_city']})" for m in t["members"])
                           + " · switch 'As user_id' to vote as someone else")
                per = t["costs"]["per_member"]
                st.caption("Per member: " + " · ".join(f"{k}: ₹{v['min']:,}–{v['max']:,}" for k, v in per.items()))
            for d in t["decisions"]:
                with st.container(border=True):
                    st.markdown(f"🗳️ **{d['question']}**  \n"
                                + " · ".join(f"{o}: {n}" for o, n in d["tally"].items())
                                + f" · closes {d['deadline'][:16].replace('T', ' ')} UTC")
                    for o, col in zip(d["options"], st.columns(len(d["options"]))):
                        if col.button(o, key=f"dv{d['decision_id']}{o}"):
                            call("POST", f"/v1/trips/{tid}/decisions/{d['decision_id']}/vote",
                                 json={"user_id": uid, "option": o})
                            time.sleep(1.5)  # a passed decision is applied by the worker
                            st.rerun()
            st.map(pd.DataFrame(t["map"]).rename(columns={"lng": "lon"}), size=40)
            for w in t["warnings"]:
                st.warning(w)
            if st.button("Undo last change"):
                call("POST", f"/v1/trips/{tid}/undo", json={"user_id": uid, "if_version": t["version"]})
                st.rerun()
            for i in t["items"]:
                price = i.get("price")
                chip = f" · ₹{price['min']:,}–{price['max']:,}/{price['unit']} ({PRICE_TAG[price['confidence']]})" if price else ""
                when = (i["start"] or i["end"] or "")[5:]
                with st.container(border=True):
                    a, b = st.columns([5, 1])
                    title = f"[{i['title']}]({i['place']['maps_url']})" if "place" in i else i["title"]
                    day = f" · day {i['day']}" if i["day"] else ""
                    a.markdown(f"**{i['slot']}**{day} · {when}  \n{title}{chip}")
                    if i.get("why"):
                        a.caption(i["why"])
                    for w in i["warnings"]:
                        a.warning(w)
                    if i["book"]:
                        a.markdown(" · ".join(f"[{l['label']}]({l['url']})" for l in i["book"]))
                    if i["locked"]:
                        a.caption("🔒 locked by group decision")
                    if "place" in i:
                        v = i["votes"]
                        if b.button(f"👍 {v['up']}", key=f"up{i['item_id']}"):
                            call("POST", f"/v1/trips/{tid}/items/{i['item_id']}/vote", json={"user_id": uid, "value": 1})
                            st.rerun()
                        if b.button(f"👎 {v['down']}", key=f"dn{i['item_id']}"):
                            call("POST", f"/v1/trips/{tid}/items/{i['item_id']}/vote", json={"user_id": uid, "value": -1})
                            st.rerun()
                        if not i["locked"] and b.button("✕", key=f"rm{i['item_id']}", help="Remove"):
                            call("DELETE", f"/v1/trips/{tid}/items/{i['item_id']}",
                                 params={"user_id": uid, "if_version": t["version"]})
                            st.rerun()
