"""Streamlit playground for trying every ML endpoint by hand.

Start the API first, then this (both from ml/):
    python -m uvicorn app:app --port 8765
    python -m streamlit run playground.py
"""
import importlib.util
import json
import os
import time

import pandas as pd
import requests
import streamlit as st

from common.utils import DATA, load
import common.http  # noqa: F401  (adds X-Service-Token when ML_SERVICE_TOKEN is set)

HAS_CAM = importlib.util.find_spec("cv2") is not None  # webcam flows need OpenCV; absent on the hosted demo

st.set_page_config(page_title="BuddySync ML playground", layout="wide")

if "api_base_input" not in st.session_state:
    st.session_state["api_base_input"] = common.http.API

c_sb1, c_sb2 = st.sidebar.columns([3, 1])
BASE = c_sb1.text_input("API base URL", key="api_base_input",
                        help="Fixed: this deployment holds a service token" if common.http.TOKEN else None)
if c_sb2.button("Local", help="Reset to http://127.0.0.1:8765"):
    st.session_state["api_base_input"] = "http://127.0.0.1:8765"
    st.rerun()

if "127.0.0.1" not in BASE and "localhost" not in BASE:
    st.sidebar.warning(f"⚠️ Connected to external host (`{BASE}`). If calls fail with 401, switch to local server `http://127.0.0.1:8765`.")
    if st.sidebar.button("👉 Switch to Local API (127.0.0.1:8765)", key="sb_switch_local"):
        st.session_state["api_base_input"] = "http://127.0.0.1:8765"
        st.rerun()


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
    if r.status_code == 401 and "127.0.0.1" not in BASE and "localhost" not in BASE:
        st.warning(f"⚠️ **401 Unauthorized from `{BASE}`.** The remote server requires an authentication token. Click below to use your active local server where all new changes are running:")
        if st.button("👉 Switch to Local Server (127.0.0.1:8765)", key=f"sw_local_{method}_{path.replace('/', '_')}"):
            st.session_state["api_base_input"] = "http://127.0.0.1:8765"
            st.rerun()
    with st.expander("Raw response"):
        st.json(body)
    return r.status_code, body


try:
    ok = requests.get(BASE + "/health", timeout=3).json().get("ok")
except Exception:
    ok = None
if ok:
    st.sidebar.success("API is up")
    if common.http.TOKEN:  # /health needs no token, so check one cheap read-only call that does
        @st.cache_data(ttl=60)
        def token_accepted(base):
            try:
                return requests.get(base + "/v1/trust/users/1", timeout=10).status_code != 401
            except Exception:
                return None
        accepted = token_accepted(BASE)
        if accepted is False:
            st.sidebar.error("API rejected the service token: ML_SERVICE_TOKEN in this app's secrets must match "
                             "the API's (Render → Environment). Reboot the app after changing secrets.")
        elif accepted:
            st.sidebar.caption("Service token accepted")
else:
    st.sidebar.error("API is down")
st.sidebar.markdown(("" if common.http.TOKEN else f"[Swagger docs]({BASE}/docs) · ")  # hidden when a token is set
                    + "data: synthetic CSVs in `ml/data/`")

tabs = st.tabs(["Buddies", "Clans", "Event recs", "Ingest", "Face", "Moderation", "Trip planner"])
looking = parts[parts.status == "looking_for_buddy"]
title = events.set_index("event_id").title


def event_label(eid):
    return f"{eid} · {title.get(eid, '?')}"


# --- Buddies -----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Amigo / BuddySync Recommendation Engine")
    st.markdown(
        "**Contextual Triadic Architecture:** `(User u, Candidate v, Event e) → Contextual Pairwise Features → Compatibility Score → Candidate Ranking`\n\n"
        "**Weighting Flow:** Persistent Taste Profile (21 dimensions + confidence semantics + explicit tags) $\\longrightarrow$ "
        "First-Time 10-Question Trade-off Questionnaire (User Default Weights) $\\longrightarrow$ "
        "Event-Specific Adjustments (Event Weights) $\\longrightarrow$ "
        "Mutual Event Synergy $S(u, v, e)$ & 7-Dimension Reciprocal Matching."
    )

    if "127.0.0.1" not in BASE and "localhost" not in BASE:
        st.warning(
            f"⚠️ **Notice:** Your Streamlit session is currently pointed at `{BASE}` (which requires authentication and doesn't have your local changes). "
            "Click the button below to connect to your active local server (`http://127.0.0.1:8765`)."
        )
        if st.button("👉 Connect to Local ML Server (http://127.0.0.1:8765)", key="tab0_top_connect_local"):
            st.session_state["api_base_input"] = "http://127.0.0.1:8765"
            st.rerun()

    busy = looking.event_id.value_counts().index[:50]
    c_ev, c_u = st.columns([2, 1])
    eid = c_ev.selectbox("Event (top 50 by people looking for a buddy)", busy, format_func=event_label, key="b_ev")
    at_event = looking[looking.event_id == eid].user_id
    uid = c_u.selectbox("Viewer", at_event, key="b_user")
    u = users.set_index("user_id").loc[uid]
    ev = events.set_index("event_id").loc[eid]

    # Context Header Card
    with st.container():
        ce1, ce2 = st.columns([1, 1])
        with ce1:
            st.markdown(f"🎪 **Selected Event #{eid}: {ev.get('title', 'Unknown')}**")
            st.caption(f"Category: `{ev.get('category')}` · City: `{ev.get('city')}` · Price Band: `{ev.get('price_band')}` · Hotness: `{round(float(ev.get('hotness', 0.5)), 2)}` · Tags: `{ev.get('tags')}`")
        with ce2:
            st.markdown(f"👤 **Viewer #{uid} ({u.get('gender', '?')}, age {u.get('age', '?')})**")
            st.caption(f"City: `{u.get('city')}` · Vibe: `{u.get('vibe')}` · Energy: `{u.get('energy')}` · After: `{u.get('after')}` · Verified: `{'✅' if u.get('verified') else '❌'}`")

    # 1. Persistent Taste Profile
    with st.expander("👤 Step 1: Viewer Persistent Taste Profile (21 Dimensions & Unknown Semantics)", expanded=False):
        t_status, t_body = call("GET", f"/v1/match/users/{int(uid)}/taste-profile")
        if t_body:
            st.info(
                "ℹ️ **Important Semantics (ML Handoff):**\n"
                "- Preference scores range `0.0–1.0`.\n"
                "- A score of `0.50` with confidence `0.0` represents **UNKNOWN** (no user data yet), **NOT** neutral preference.\n"
                "- The matching engine **does not penalize** compatibility on unknown dimensions; similarity is strictly weighted by joint confidence $(c_u \\times c_v)$.\n"
                "- `tags` represent explicit interest tags with intensity weights."
            )

            # High-level summary
            st.markdown(f"**Primary Interest Group:** `{t_body.get('primary_group')}` · **All Groups:** {', '.join(t_body.get('interest_groups', []))}")
            tags_dict = t_body.get("tags", {})
            if tags_dict:
                tag_badges = " ".join([f"`🏷️ {t} ({v:.1f})`" for t, v in tags_dict.items()])
                st.markdown(f"**Explicit Interest Tags:** {tag_badges}")
            else:
                st.markdown(f"**Interests:** `{'; '.join(t_body.get('interests', []))}`")
            st.caption(f"Bio: {t_body.get('bio')} · Vibe: {t_body.get('vibe')} · Energy: {t_body.get('energy')}")

            # 21 Dimensions Table
            dims = t_body.get("dimensions", {})
            confs = t_body.get("confidence", {})
            if dims:
                CLUSTERS = {
                    "entertainment": ["MUSIC", "NIGHTLIFE", "PARTY", "MOVIES", "COMEDY", "THEATRE", "FOOD", "ARTS"],
                    "active": ["SPORTS", "OUTDOOR", "TRAVEL", "COMPETITIVE"],
                    "intellectual": ["TECHNOLOGY", "ACADEMIC", "CREATIVE"],
                    "social_style": ["SOCIAL", "GROUP_PREFERENCE", "SOLO_COMFORT", "CROWD_COMFORT", "EXPLORATION", "LIVE_EVENT"],
                }
                dim_to_cluster = {}
                for clust, members in CLUSTERS.items():
                    for m in members:
                        dim_to_cluster[m] = clust

                dim_rows = []
                for dim_name, score in dims.items():
                    c_val = confs.get(dim_name, 0.0 if abs(score - 0.50) < 1e-4 else 0.50)
                    is_known = c_val > 0.0
                    status_lbl = f"🟢 Measured ({round(c_val * 100)}%)" if is_known else "⚪ Unknown (Non-penalizing)"
                    dim_rows.append({
                        "Dimension": dim_name,
                        "Cluster": dim_to_cluster.get(dim_name, "other"),
                        "Preference Score": round(float(score), 2),
                        "Confidence": round(float(c_val), 2),
                        "Status": status_lbl,
                    })
                df_dims = pd.DataFrame(dim_rows)
                st.dataframe(df_dims, use_container_width=True, hide_index=True)

            # JSON Editor / Inspection
            with st.expander("🛠️ Inspect or Edit Raw Taste Profile JSON (Backend Handoff Schema)"):
                clean_json_str = json.dumps({
                    "userId": t_body.get("userId", int(uid)),
                    "dimensions": dims,
                    "confidence": confs,
                    "tags": tags_dict,
                }, indent=2)
                raw_json_input = st.text_area("Taste Profile JSON", clean_json_str, height=220, key=f"raw_tp_{uid}")
                if st.button("Save Updated Taste Profile", key=f"btn_save_tp_{uid}"):
                    try:
                        parsed = json.loads(raw_json_input)
                        s, resp = call("POST", f"/v1/match/users/{int(uid)}/taste-profile", json={
                            "dimensions": parsed.get("dimensions"),
                            "confidence": parsed.get("confidence"),
                            "tags": parsed.get("tags"),
                            "bio": t_body.get("bio"),
                            "vibe": t_body.get("vibe"),
                            "energy": t_body.get("energy"),
                        })
                        if s == 200:
                            st.success("Taste Profile saved successfully!")
                            st.rerun()
                    except Exception as ex:
                        st.error(f"Invalid JSON: {ex}")

    # 2. First-Time Questionnaire & User Default Weights
    w_status, w_body = call("GET", f"/v1/match/users/{int(uid)}/weights")
    user_defaults = w_body.get("default_weights", {}) if w_body else {}
    is_fallback = w_body.get("is_fallback", True) if w_body else True

    st.markdown("### ⚖️ Step 2: User Default Weights (Recommendation Profile)")
    if is_fallback:
        st.info("ℹ️ **Status: Using System Fallback Weights (30/25/15/10/10/5/5).** User has not yet completed the first-time questionnaire. "
                "Baseline derived from internal user survey and AI weighting optimization.")
    else:
        st.success("✅ **Status: Custom Default Profile active.** Saved from first-time questionnaire.")

    # Show metrics for 7 dimensions
    d_cols = st.columns(7)
    labels = ["Interest", "Event Fit", "Social Fit", "Prefs", "Intent Fit", "Location", "Availability"]
    keys_list = ["interest", "event_fit", "social_fit", "prefs", "intent_fit", "location", "availability"]
    for col, lbl, k in zip(d_cols, labels, keys_list):
        val = user_defaults.get(k, 0.0)
        col.metric(lbl, f"{round(val * 100, 1)}%")

    with st.expander("📝 First-Time Questionnaire (10 Trade-Off Questions) — Set or Reconduct Defaults", expanded=False):
        q_status, q_data = call("GET", "/v1/match/questionnaire/first-time")
        if q_data and "questions" in q_data:
            st.caption("Answer these trade-off questions to generate your persistent default weights:")
            user_answers = {}
            for q in q_data["questions"]:
                st.markdown(f"**{q['id'].upper()} ({q['dimension']}):** {q['question']}")
                opts = [f"{o['id']}. {o['text']}" for o in q["options"]]
                sel = st.radio(q['question'], opts, key=f"q_{uid}_{q['id']}", label_visibility="collapsed")
                user_answers[q["id"]] = sel[0]  # 'A' or 'B'

            c_sub, c_rec = st.columns([1, 1])
            if c_sub.button("Calculate & Save Default Weights", type="primary", key="save_defaults_btn"):
                s, resp = call("POST", f"/v1/match/users/{int(uid)}/weights/first-time", json={"answers": user_answers})
                if s == 200:
                    st.success("Default weights saved! Refreshing profile...")
                    st.rerun()

            if c_rec.button("🔄 Reconduct Questionnaire (Reset to Fallback)", key="reconduct_btn"):
                s, resp = call("POST", f"/v1/match/users/{int(uid)}/weights/reconduct")
                if s == 200:
                    st.warning("Profile reset to system fallback! Ready to reconduct.")
                    st.rerun()

    # 3. Event-Specific Adjustments
    st.markdown("### 🎯 Step 3: Event-Specific Adjustments (Preserves Default Weights)")
    ew_status, ew_body = call("GET", f"/v1/match/users/{int(uid)}/events/{int(eid)}/weights")
    cur_event_weights = ew_body.get("effective_weights", user_defaults) if ew_body else user_defaults
    ev_source = ew_body.get("source", "system_fallback") if ew_body else "system_fallback"

    if ev_source == "event_custom":
        st.warning(f"⚡ **Active: Custom adjustments applied for Event #{eid}.** User's persistent default weights remain untouched.")
    else:
        st.caption(f"Currently using base weights ({ev_source}) for Event #{eid}.")

    adj_tab1, adj_tab2 = st.tabs(["Quick Event Survey (4 Questions)", "Direct Percentage Sliders"])
    with adj_tab1:
        st.caption("Short questions to determine if you want to deviate from your normal weighting for this event:")
        eq_status, eq_data = call("GET", "/v1/match/questionnaire/event-specific")
        event_answers = {}
        if eq_data and "questions" in eq_data:
            for eq in eq_data["questions"]:
                st.markdown(f"**{eq['question']}**")
                opts = [f"{o['id']}: {o['text']}" for o in eq["options"]]
                sel = st.radio(eq['question'], opts, key=f"eq_{uid}_{eid}_{eq['id']}", label_visibility="collapsed")
                event_answers[eq["id"]] = sel.split(":")[0]

            c_apply_eq, c_rst_eq = st.columns([1, 1])
            if c_apply_eq.button("Apply Event Survey Preferences", key="save_event_survey_btn"):
                s, resp = call("POST", f"/v1/match/users/{int(uid)}/events/{int(eid)}/weights", json={"answers": event_answers})
                if s == 200:
                    st.success("Event weights updated! Defaults remain unchanged.")
                    st.rerun()
            if c_rst_eq.button("🔄 Reset Event Overrides to User Defaults", key="reset_event_eq_btn"):
                s, resp = call("POST", f"/v1/match/users/{int(uid)}/events/{int(eid)}/weights/reset")
                if s == 200:
                    st.info("Event overrides cleared! Reverted to user default weights.")
                    st.rerun()

    with adj_tab2:
        st.caption("Slide percentages to adjust priorities for this event (sums auto-normalize to 100%):")
        slider_vals = {}
        sc1, sc2 = st.columns(2)
        for i, (lbl, k) in enumerate(zip(labels, keys_list)):
            cur_p = round(cur_event_weights.get(k, user_defaults.get(k, 0.1)) * 100, 1)
            target_col = sc1 if i < 4 else sc2
            slider_vals[k] = target_col.slider(f"{lbl} (%)", 0.0, 100.0, cur_p, 1.0, key=f"sl_{uid}_{eid}_{k}")

        c_apply_sl, c_rst_sl = st.columns([1, 1])
        if c_apply_sl.button("Save Slider Weights for This Event", key="save_event_sliders_btn"):
            s, resp = call("POST", f"/v1/match/users/{int(uid)}/events/{int(eid)}/weights", json={"custom_weights": slider_vals})
            if s == 200:
                st.success("Event custom weights saved! Default weights preserved.")
                st.rerun()
        if c_rst_sl.button("🔄 Reset Event Overrides to User Defaults", key="reset_event_sl_btn"):
            s, resp = call("POST", f"/v1/match/users/{int(uid)}/events/{int(eid)}/weights/reset")
            if s == 200:
                st.info("Event overrides cleared! Reverted to user default weights.")
                st.rerun()

    # 4. Recommendation Search
    st.markdown("### 👥 Step 4: Recommend Buddies")
    c_ctrl1, c_ctrl2 = st.columns([2, 1])
    limit = c_ctrl1.slider("Candidate Limit", 1, 30, 10, key="b_lim")
    use_adj = c_ctrl2.checkbox("Use Event-Specific Adjustments", value=True, key="b_use_adj",
                               help="When checked, uses event custom weights if set; otherwise user defaults.")

    c_btn1, c_btn2 = st.columns([1, 1])
    btn_recommend_schema = c_btn1.button("🚀 Recommend (Future ML Output Schema)", type="primary", use_container_width=True, key="btn_rec_schema")
    btn_detailed_ranking = c_btn2.button("📊 Detailed Reciprocal Ranking (Diagnostic Mode)", use_container_width=True, key="btn_rec_detailed")

    if btn_recommend_schema:
        st.markdown("#### 📦 Future ML Output Schema (`POST /v1/match/recommend`)")
        _, body = call("POST", "/v1/match/recommend", json={
            "userId": int(uid),
            "eventId": int(eid),
            "limit": limit,
            "use_event_adjustments": use_adj,
        })
        if body and "recommendations" in body:
            st.caption("Matches requested backend contract: `userId`, `eventId`, and ordered list of `{candidateUserId, score}`:")
            st.json(body)

            recs = body.get("recommendations", [])
            if recs:
                rec_df = pd.DataFrame(recs)
                rec_df.rename(columns={"candidateUserId": "user_id"}, inplace=True)
                info = users.set_index("user_id")[["gender", "age", "interests", "vibe", "energy", "after", "verified"]]
                rec_df = rec_df.set_index("user_id").join(info)
                rec_df["score_pct"] = (rec_df["score"] * 100).round(1).astype(str) + "%"
                st.dataframe(rec_df[["score_pct", "gender", "age", "interests", "vibe", "energy", "verified"]], use_container_width=True)
        elif body is not None:
            st.info("No candidates found.")

    if btn_detailed_ranking:
        st.markdown("#### 🔬 Detailed Reciprocal Ranking (`POST /v1/match/buddies`)")
        _, body = call("POST", "/v1/match/buddies", json={
            "user_id": int(uid),
            "event_id": int(eid),
            "limit": limit,
            "use_event_adjustments": use_adj,
        })
        if body and body.get("candidates"):
            st.caption(f"Weights used ({body.get('weights_source')}): " +
                       ", ".join(f"{k}: {round(v * 100, 1)}%" for k, v in body.get("weights_used", {}).items()))
            info = users.set_index("user_id")[["gender", "age", "interests", "vibe", "energy", "after", "verified"]]
            df = pd.DataFrame(body["candidates"]).set_index("user_id").join(info)
            df["reasons"] = df.reasons.str.join(" · ")
            df["shared_interests"] = df.shared_interests.map(lambda xs: ", ".join(f"{x['interest']} {x['pct']}%" for x in xs))
            st.dataframe(df.drop(columns="components"), use_container_width=True)

            with st.expander("🔍 Detailed 7-Dimension Component Scores (Per Candidate)"):
                comp_rows = []
                for c in body["candidates"]:
                    row = {"user_id": c["user_id"], "match_pct": c["match_pct"], **c["components"]}
                    comp_rows.append(row)
                st.dataframe(pd.DataFrame(comp_rows).set_index("user_id"), use_container_width=True)
        elif body is not None:
            st.info("No candidates found.")

    st.markdown("---")
    st.markdown("**Trust signals** (post-event feedback: show-ups, would join again)")
    t_uid = st.number_input("user_id", 1, int(users.user_id.max()), int(uid), key="t_user")
    if st.button("Get trust signals"):
        _, body = call("GET", f"/v1/trust/users/{int(t_uid)}")
        if body and "level" in body:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("level", body["level"], f"reliability {body['reliability']}")
            c2.metric("showed up", body["showed_up"])
            c3.metric("would join again", body["would_join_again"])
            c4.metric("events attended", body["events_attended"])

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
# Live check in the viewer's browser (works on the hosted demo, where the server has no camera):
# preview → Record → do the steps → Stop → the video + a still from its start go back to Python.
browser_recorder = st.components.v2.component("liveness_recorder", html="""
<video autoplay muted playsinline></video>
<ol></ol>
<p class="msg">Starting the camera…</p>
<button disabled>● Record</button>
""", css="""
video { width: 100%; max-width: 480px; border-radius: 8px; background: #000; transform: scaleX(-1); }
.msg { font-weight: 600; }
button { padding: .4rem 1rem; border-radius: 6px; border: 1px solid currentColor; cursor: pointer; font: inherit; }
""", js="""
export default function ({ parentElement, data, setTriggerValue }) {
  const $ = (q) => parentElement.querySelector(q);
  const video = $("video"), btn = $("button"), msg = $(".msg");
  $("ol").replaceChildren(...data.steps.map((s) => Object.assign(document.createElement("li"), { textContent: s })));
  let stream, rec, timer, gone = false;
  const say = (t) => (msg.textContent = t);
  const b64 = (blob) => new Promise((ok) => {
    const r = new FileReader(); r.onload = () => ok(r.result.split(",")[1]); r.readAsDataURL(blob);
  });
  navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false })
    .then((s) => {
      if (gone) return s.getTracks().forEach((t) => t.stop());
      stream = s; video.srcObject = s; btn.disabled = false;
      say("Face the camera, then press Record.");
    })
    .catch((e) => say(`Camera unavailable: ${e.message}. Allow camera access for this site.`));

  btn.onclick = async () => {
    if (rec) return rec.stop();
    const still = document.createElement("canvas");  // profile photo: a frame from the start of the check
    still.width = video.videoWidth; still.height = video.videoHeight;
    still.getContext("2d").drawImage(video, 0, 0);
    const photo = await new Promise((ok) => still.toBlob(ok, "image/jpeg", 0.9));
    // name the codec: left to choose, Chrome records AV1, which the server's OpenCV (Linux) can't decode
    const mime = ["video/webm;codecs=vp8", "video/mp4;codecs=avc1", "video/mp4"].find((m) => MediaRecorder.isTypeSupported(m));
    if (!mime) return say("This browser can't record VP8 WebM or H.264 MP4. Try Chrome, Firefox or Safari.");
    const chunks = [];
    rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 2500000 });
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    rec.onstop = async () => {
      clearInterval(timer); btn.disabled = true; say("Verifying…");
      const blob = new Blob(chunks, { type: rec.mimeType });
      setTriggerValue("recorded", { video: await b64(blob), mime: rec.mimeType, photo: await b64(photo) });
    };
    rec.start(250);
    btn.textContent = "■ Stop";
    const end = Date.now() + data.max_s * 1000;
    say("Look straight at the camera…");
    setTimeout(() => rec.state === "recording" && say("Now do the steps in order, then press Stop."), 1200);
    timer = setInterval(() => Date.now() >= end && rec.state === "recording" && rec.stop(), 250);
  };
  return () => { gone = true; clearInterval(timer); stream?.getTracks().forEach((t) => t.stop()); };
}
""")


def show_liveness(r):
    c1, c2, c3 = st.columns(3)
    c1.metric("decision", r["decision"], r.get("reason"))
    c2.metric("same person throughout", str(r.get("same_person_throughout")), f"min {r.get('min_continuity')}")
    c3.metric("face match", r.get("best_similarity", "—"), f"coverage {r.get('identity_coverage')}")
    if r.get("liveness"):
        st.dataframe(pd.DataFrame(r["liveness"]["steps"]), use_container_width=True)


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
    STEP_S, ATTEMPT_S, REC_FPS, GREEN, YELLOW, RED, GREY = 7.0, MAX_VIDEO_S - 3, 15, (80, 220, 80), (0, 220, 255), (60, 60, 255), (170, 170, 170)
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
        photo, steady = st.session_state.get("cam_photo"), 0
        while photo is None:
            f, s, n_faces = frame_and_signals()
            good = s.get("face") and n_faces == 1 and s.get("eyes_open") and abs(s.get("yaw", 1)) < 0.08
            steady = steady + 1 if good else 0
            hint = ("Hold still..." if good else "No face found" if not s.get("face")
                    else "Only you in the frame" if n_faces > 1 else "Look straight at the camera")
            draw(f, [("Profile photo", YELLOW), (hint, GREEN if good else RED), ("#" * steady, GREEN)])
            if steady >= 15:
                photo = cv2.imencode(".jpg", f)[1].tobytes()
                st.session_state.cam_photo = photo
        hold(1.0, [("Profile photo taken  OK", GREEN)])

        result = None
        for attempt in range(1, max_attempts + 1):
            # 2. a fresh random challenge each attempt (each one works only once)
            ch = requests.post(f"{base}/v1/verify/challenge", timeout=10).json()
            prompts = {x["action"]: x["prompt"] for x in ch["steps"]}
            guide = LiveGuide([x["action"] for x in ch["steps"]])
            path = os.path.join(tempfile.gettempdir(), f"liveness_attempt{attempt}.mp4")
            f, _, _ = frame_and_signals()
            out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), REC_FPS, (f.shape[1], f.shape[0]))
            t0, step_t0, note, finished_at, written = time.time(), None, None, None, 0
            while True:
                f, s, n_faces = frame_and_signals()
                t = time.time() - t0
                while written <= t * REC_FPS:  # paced by the clock, so the file's length is the real length
                    out.write(f)  # the recording has no overlay
                    written += 1
                if step_t0 is None and len(guide.start_frames) >= 5:
                    step_t0 = t  # start pose captured: the first instruction pops up
                if guide.feed(s, t):
                    note, step_t0 = ("OK  -  next:", GREEN, t + 0.9), t
                elif step_t0 is not None and not guide.finished and t - step_t0 > STEP_S:
                    note, step_t0 = ("Not detected  -  retake:", RED, t + 1.8), t  # retake the same step
                    guide.retake()
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
                show_liveness(r)
                with st.expander("Raw response"):
                    st.json(r)
            else:
                st.error(r)
    elif mode.startswith("Liveness") and st.radio("Video", ["Live camera (in your browser)", "Upload a video"],
                                                  horizontal=True, key="f_src").startswith("Live"):
        st.caption("Get a challenge → Record → face the camera, do the steps in order → Stop. "
                   "No profile photos uploaded? A still from the start of the recording is used.")
        if st.button("🔴 Start live check", type="primary"):
            _, c = call("POST", "/v1/verify/challenge")
            if c and "challenge_id" in c:
                st.session_state.live_challenge = c
        c = st.session_state.get("live_challenge")
        if c:
            slot = st.empty()
            with slot:
                rec = browser_recorder(key=c["challenge_id"], on_recorded_change=lambda: None,
                                       data={"steps": [s["prompt"] for s in c["steps"]], "max_s": 18})
            if rec.recorded:
                import base64
                slot.empty()  # recording is in: drop the recorder (and its camera) before the result shows
                st.session_state.pop("live_challenge")  # used up either way
                ext = "mp4" if "mp4" in rec.recorded["mime"] else "webm"
                import tempfile
                with open(os.path.join(tempfile.gettempdir(), f"liveness_browser.{ext}"), "wb") as fh:
                    fh.write(base64.b64decode(rec.recorded["video"]))  # last attempt, for debugging (local temp)
                photo = [("profile_photos", ("live.jpg", base64.b64decode(rec.recorded["photo"]), "image/jpeg"))]
                with st.spinner("Analysing video…"):
                    _, r = call("POST", "/v1/verify/liveness", data={"challenge_id": c["challenge_id"]},
                                files=[("video", (f"live.{ext}", base64.b64decode(rec.recorded["video"]),
                                                  rec.recorded["mime"].split(";")[0]))] + (profile_files or photo))
                if r and "decision" in r:
                    (st.success if r["decision"] == "pass" else st.warning)(
                        r["decision"].upper() + (f": {r.get('reason')}" if r.get("reason") else ""))
                    show_liveness(r)
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
                    show_liveness(r)
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
