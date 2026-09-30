import uuid
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from common.utils import load, load_artifact
from event_recs.model import score, taste

router = APIRouter(prefix="/recs", tags=["event recommendations"])


@lru_cache
def artifact():
    try:
        return load_artifact("event_recs")
    except FileNotFoundError:
        raise HTTPException(503, "model not trained: run python -m event_recs.train")


@lru_cache
def tables():
    events, inter = load("events"), load("participants")
    events["start_time"] = pd.to_datetime(events.start_time)
    return load("users").set_index("user_id"), events, inter


class RecsQuery(BaseModel):
    user_id: int
    limit: int = 10
    days_ahead: int = 60
    max_per_category: int = 3


@router.post("/events")
def recommend(q: RecsQuery):
    users, events, inter = tables()
    if q.user_id not in users.index:
        raise HTTPException(404, "unknown user")
    art, user = artifact(), users.loc[q.user_id]
    mine = inter[inter.user_id == q.user_id]
    now = datetime.now()
    cands = events[(events.city == user.city) & (events.start_time >= now)
                   & (events.start_time <= now + timedelta(days=q.days_ahead))
                   & ~events.event_id.isin(mine.event_id)]
    if cands.empty:
        return {"request_id": str(uuid.uuid4()), "model_version": art["model_version"], "events": []}
    total, cos, dist, pop = score(user, taste(user, mine, events, art["vectorizer"]), cands,
                                  art["vectorizer"], art["popularity"], art["weights"])
    out, per_cat = [], {}
    for i in np.argsort(-total):  # diversity: cap events per category
        e = cands.iloc[i]
        if per_cat.get(e.category, 0) >= q.max_per_category:
            continue
        per_cat[e.category] = per_cat.get(e.category, 0) + 1
        reasons = [r for r, on in [(f"matches your taste for {e.tags}", cos[i] > 0.1),
                                   (f"popular in {e.city}", pop[i] > 0.5),
                                   (f"{dist[i]:.1f} km away", dist[i] < 5)] if on]
        out.append({"event_id": int(e.event_id), "title": e.title, "start_time": e.start_time.isoformat(),
                    "score": round(float(total[i]), 3), "reasons": reasons})
        if len(out) == q.limit:
            break
    return {"request_id": str(uuid.uuid4()), "model_version": art["model_version"], "events": out}
