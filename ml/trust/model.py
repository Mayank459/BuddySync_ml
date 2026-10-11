"""Reliability: does this person show up, and would the people they went with join them again?

Built from post-event peer feedback. Rates are smoothed towards a prior worth PRIOR_N ratings, so a new
user (no feedback) starts neutral and one bad night doesn't sink anyone. Shared by trust/api.py and
buddy_matching (reliability nudges the ranking; "wouldn't join again" keeps a pair apart).
"""
import pandas as pd

PRIOR_N, SHOW_PRIOR, AGAIN_PRIOR = 4, 0.9, 0.75
NEUTRAL = 0.6 * SHOW_PRIOR + 0.4 * AGAIN_PRIOR  # reliability with no feedback yet
MIN_FEEDBACK = 3                                  # below this the level stays "new"


def table(peer: pd.DataFrame) -> pd.DataFrame:
    """One row per user with feedback: counts, smoothed rates and reliability (0-1)."""
    t = peer.groupby("ratee_id").agg(feedback=("showed_up", "size"), showed=("showed_up", "sum"),
                                     again=("would_join_again", "sum")).rename_axis("user_id")
    t["show_up"] = (t.showed + PRIOR_N * SHOW_PRIOR) / (t.feedback + PRIOR_N)
    t["join_again"] = (t.again + PRIOR_N * AGAIN_PRIOR) / (t.feedback + PRIOR_N)
    t["reliability"] = 0.6 * t.show_up + 0.4 * t.join_again
    return t


def attended(peer: pd.DataFrame, ratings: pd.DataFrame) -> pd.Series:
    """Past events per user: they rated the event, or someone who went with them says they showed up."""
    seen = peer.loc[peer.showed_up, ["ratee_id", "event_id"]].set_axis(["user_id", "event_id"], axis=1)
    return pd.concat([seen, ratings[["user_id", "event_id"]]]).drop_duplicates().groupby("user_id").size()


def level(reliability: float, feedback: int) -> str:
    if feedback < MIN_FEEDBACK:
        return "new"
    return "reliable" if reliability >= 0.85 else "ok" if reliability >= 0.65 else "low"


def never_again(peer: pd.DataFrame) -> set:
    """(rater, ratee) pairs where the rater said they wouldn't join the other again: kept apart both ways."""
    no = peer.loc[~peer.would_join_again, ["rater_id", "ratee_id"]]
    return set(zip(no.rater_id, no.ratee_id)) | set(zip(no.ratee_id, no.rater_id))


def profile(user_id: int, t: pd.DataFrame, events: pd.Series) -> dict:
    if user_id in t.index:
        r = t.loc[user_id]
        n, rel = int(r.feedback), float(r.reliability)
        rates = {"showed_up": f"{int(r.showed)} of {n}", "would_join_again": f"{int(r.again)} of {n}"}
    else:
        n, rel, rates = 0, NEUTRAL, {"showed_up": "0 of 0", "would_join_again": "0 of 0"}
    return {"user_id": user_id, "events_attended": int(events.get(user_id, 0)), "feedback_count": n,
            **rates, "reliability": round(rel, 3), "level": level(rel, n)}
