"""Event recs pipeline: data -> preprocess -> time split -> train -> test -> save.

Split by time, never randomly:  [ train | validation | test ]
  - train      : builds taste vectors + tag popularity
  - validation : picks the blend weights
  - test       : final numbers, touched once
Run from ml/:  python -m event_recs.train
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from common.utils import load, save_artifact
from event_recs.model import fit_text, popularity, score, taste

NO_INTER = pd.DataFrame(columns=["event_id", "status"])


def components(users, events, past, future, cutoff, vec):
    """Per user: score components for upcoming city events + which ones they engaged with later."""
    pop, upcoming = popularity(events, past), events[events.start_time >= cutoff]
    by_past, by_future = dict(list(past.groupby("user_id"))), dict(list(future.groupby("user_id")))
    rows = []
    for u in users.itertuples():
        if u.user_id not in by_future:
            continue
        mine = by_past.get(u.user_id, NO_INTER)
        cands = upcoming[(upcoming.city == u.city) & ~upcoming.event_id.isin(mine.event_id)]
        y = cands.event_id.isin(by_future[u.user_id].event_id).values.astype(int)
        if y.sum() and len(cands) > 1:
            _, cos, dist, p = score(u, taste(u, mine, events, vec), cands, vec, pop)
            rows.append((np.column_stack([cos, np.exp(-dist / 10), p]), y))
    return rows


def evaluate(rows, w, k=10):
    wv = np.array([w["taste"], w["distance"], w["popularity"]])
    recall, ndcg = [], []
    for C, y in rows:
        s = C @ wv
        recall.append(y[np.argsort(-s)[:k]].sum() / y.sum())
        ndcg.append(ndcg_score([y], [s], k=k))
    return {"recall10": float(np.mean(recall)), "ndcg10": float(np.mean(ndcg))}


def main():
    # 1. data
    users, events, inter = load("users"), load("events"), load("participants")
    events["start_time"], inter["ts"] = pd.to_datetime(events.start_time), pd.to_datetime(inter.ts)

    # 2. time split
    t_val, t_test = inter.ts.quantile(0.55), inter.ts.quantile(0.7)
    vec = fit_text(events, users)  # event text is public before the event, so no leakage

    # 3. train on [0, t_val), pick weights on [t_val, t_test)
    val_rows = components(users, events, inter[inter.ts < t_val],
                          inter[(inter.ts >= t_val) & (inter.ts < t_test)], t_val, vec)
    grid = [dict(zip(["taste", "distance", "popularity"], np.round(np.array(c) / 10, 1)))
            for c in itertools.product(range(11), repeat=3) if sum(c) == 10]
    best = max(grid, key=lambda w: evaluate(val_rows, w)["ndcg10"])

    # 4. test: everything before t_test is history, score what happened after
    test_rows = components(users, events, inter[inter.ts < t_test], inter[inter.ts >= t_test], t_test, vec)
    variants = {"popularity only": {"taste": 0, "distance": 0, "popularity": 1},
                "taste only": {"taste": 1, "distance": 0, "popularity": 0},
                "untuned 60/20/20": {"taste": 0.6, "distance": 0.2, "popularity": 0.2},
                f"tuned {best['taste']}/{best['distance']}/{best['popularity']}": best}
    print(f"val users {len(val_rows)}  test users {len(test_rows)}  test cutoff {t_test:%Y-%m-%d}")
    print(f"{'model':22s} {'Recall@10':>9s} {'NDCG@10':>8s}")
    metrics = {n: evaluate(test_rows, w) for n, w in variants.items()}
    for n, m in metrics.items():
        print(f"{n:22s} {m['recall10']:9.3f} {m['ndcg10']:8.3f}")

    # 5. save: popularity refit on everything known now, for serving
    path = save_artifact({"vectorizer": vec, "popularity": popularity(events, inter), "weights": best,
                          "model_version": "event-recs-phase0-tuned", "metrics": metrics}, "event_recs")
    print("saved", path)


if __name__ == "__main__":
    main()
