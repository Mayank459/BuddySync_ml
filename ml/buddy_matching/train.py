"""Buddy matching pipeline: data -> preprocess -> split -> train -> test -> save.

Label: the suggestion ended in an accepted buddy request (both sides said yes).
Run from ml/:  python -m buddy_matching.train
"""
import itertools

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ndcg_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import QuantileTransformer

from buddy_matching.model import DEFAULT_WEIGHTS, KEYS, features, fit_text, hard_ok, reciprocal
from common.utils import load, save_artifact


def weight_grid(step=0.05):
    """Every 4-weight combination summing to 1 (each >= step): 969 candidates."""
    n = round(1 / step)
    for a, b, c in itertools.product(range(1, n), repeat=3):
        if n - a - b - c >= 1:
            yield dict(zip(KEYS, np.round(np.array([a, b, c, n - a - b - c]) * step, 2)))


def mean_ndcg(df, scores, k=10):
    df = df.assign(s=scores)
    vals = [ndcg_score([g.y], [g.s], k=k) for _, g in df.groupby(["viewer_id", "event_id"])
            if g.y.sum() > 0 and len(g) > 1]
    return float(np.mean(vals))


def main():
    # 1. data
    users, req = load("users"), load("buddy_requests")

    # 2. preprocess: pair features; drop pairs the hard filters would never show
    vec = fit_text(users)  # text vocabulary only, no labels involved
    raw, A, B, _, _ = features(req, users, vec)
    keep = (hard_ok(A, B) & hard_ok(B, A)).values
    req, raw = req[keep].reset_index(drop=True), raw[keep]
    req["y"] = (req.status == "accepted").astype(int)
    y = req.y.values

    # 3. split by viewer so nobody's suggestions leak across train and test
    tr, te = next(GroupShuffleSplit(test_size=0.2, random_state=0).split(raw, y, groups=req.viewer_id))
    # rescale every component to [0,1] by rank, so a weight means what it says (fit on train only)
    scaler = QuantileTransformer(n_quantiles=200).fit(raw[tr])
    Z = scaler.transform(raw)

    # 4. train: (a) tune the 4 weights, (b) logistic regression = Phase 1 candidate
    tuned = max(weight_grid(), key=lambda w: roc_auc_score(y[tr], reciprocal(Z[tr], w)))
    lr = LogisticRegression(max_iter=1000).fit(Z[tr], y[tr])

    # 5. test
    test = req.iloc[te]
    results = {
        "default 40/30/20/10": reciprocal(Z[te], DEFAULT_WEIGHTS),
        f"tuned {'/'.join(str(round(v * 100)) for v in tuned.values())}": reciprocal(Z[te], tuned),
        "logistic regression": lr.predict_proba(Z[te])[:, 1],
    }
    print(f"train pairs {len(tr)}  test pairs {len(te)}  positive rate {y.mean():.3f}")
    print(f"{'model':28s} {'AUC':>6s} {'NDCG@10':>8s}")
    metrics = {}
    for name, s in results.items():
        metrics[name] = {"auc": roc_auc_score(y[te], s), "ndcg10": mean_ndcg(test, s)}
        print(f"{name:28s} {metrics[name]['auc']:6.3f} {metrics[name]['ndcg10']:8.3f}")

    # 6. save: API serves the tuned weights (explainable); LR is kept for comparison
    path = save_artifact({"vectorizer": vec, "scaler": scaler, "weights": tuned, "lr": lr,
                          "model_version": "buddy-phase0-tuned", "metrics": metrics}, "buddy_matching")
    print("saved", path)


if __name__ == "__main__":
    main()
