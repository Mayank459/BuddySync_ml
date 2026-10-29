"""Buddy matching pipeline: data -> preprocess -> split -> train -> test -> save.

7 Dimensions: interest, event_fit, social_fit, prefs, intent_fit, location, availability.
Label: the suggestion ended in an accepted buddy request (both sides said yes).
Run from ml/:  python -m buddy_matching.train
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ndcg_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import QuantileTransformer

from buddy_matching.model import (
    DEFAULT_WEIGHTS,
    KEYS,
    features,
    fit_text,
    hard_ok,
    normalize_weights,
    reciprocal,
)
from common.utils import load, save_artifact


def mean_ndcg(df, scores, k=10):
    df = df.assign(s=scores)
    vals = [ndcg_score([g.y], [g.s], k=k) for _, g in df.groupby(["viewer_id", "event_id"])
            if g.y.sum() > 0 and len(g) > 1]
    return float(np.mean(vals)) if vals else 0.0


def tune_weights(Z_tr, y_tr, init_weights=DEFAULT_WEIGHTS, max_iter=120) -> dict:
    """Tune 7 recommendation weights via bounded softmax optimization."""
    w0 = np.array([init_weights[k] for k in KEYS])

    def softmax(x):
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    def obj(x):
        w = dict(zip(KEYS, softmax(x)))
        try:
            return -roc_auc_score(y_tr, reciprocal(Z_tr, w))
        except Exception:
            return 0.0

    res = minimize(obj, np.log(w0), method="Nelder-Mead", options={"maxiter": max_iter, "disp": False})
    # enforce minimum floor of 0.02 per dimension
    raw_tuned = dict(zip(KEYS, softmax(res.x)))
    bounded = {k: max(0.02, v) for k, v in raw_tuned.items()}
    return normalize_weights(bounded, decimals=2)


def main():
    # 1. data
    users, req, events = load("users"), load("buddy_requests"), load("events")

    # 2. preprocess: pair features (8 columns); drop pairs hard filters would never show
    vec = fit_text(users)
    raw, A, B, _, _ = features(req, users, vec, events=events)
    keep = (hard_ok(A, B) & hard_ok(B, A)).values
    req, raw = req[keep].reset_index(drop=True), raw[keep]
    req["y"] = (req.status == "accepted").astype(int)
    y = req.y.values

    # 3. split by viewer so nobody's suggestions leak across train and test
    tr, te = next(GroupShuffleSplit(test_size=0.2, random_state=0).split(raw, y, groups=req.viewer_id))
    scaler = QuantileTransformer(n_quantiles=200).fit(raw[tr])
    Z = scaler.transform(raw)

    # 4. train: (a) tune the 7 weights, (b) logistic regression candidate
    tuned = tune_weights(Z[tr], y[tr])
    lr = LogisticRegression(max_iter=1000).fit(Z[tr], y[tr])

    # 5. test
    test = req.iloc[te]
    results = {
        "fallback 30/25/15/10/10/5/5": reciprocal(Z[te], DEFAULT_WEIGHTS),
        f"tuned {'/'.join(str(round(v * 100)) for v in tuned.values())}": reciprocal(Z[te], tuned),
        "logistic regression": lr.predict_proba(Z[te])[:, 1],
    }
    print(f"train pairs {len(tr)}  test pairs {len(te)}  positive rate {y.mean():.3f}")
    print(f"{'model':32s} {'AUC':>6s} {'NDCG@10':>8s}")
    metrics = {}
    for name, s in results.items():
        metrics[name] = {"auc": roc_auc_score(y[te], s), "ndcg10": mean_ndcg(test, s)}
        print(f"{name:32s} {metrics[name]['auc']:6.3f} {metrics[name]['ndcg10']:8.3f}")

    # 6. save
    path = save_artifact({
        "vectorizer": vec,
        "scaler": scaler,
        "weights": tuned,
        "default_weights": DEFAULT_WEIGHTS,
        "keys": KEYS,
        "lr": lr,
        "model_version": "buddy-phase0-tuned-7dim",
        "metrics": metrics,
    }, "buddy_matching")
    print("saved", path)


if __name__ == "__main__":
    main()
