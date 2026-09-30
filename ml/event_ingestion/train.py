"""Event ingestion pipelines: data -> preprocess -> split -> train -> test -> save.

Two small models:
  dedup     : pair classifier on blocked listing pairs (label: same real event)
  category  : tf-idf + logistic regression on title + description
Run from ml/:  python -m event_ingestion.train
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.utils import load, save_artifact
from event_ingestion.model import blocked_pairs, category_text, pair_features


def train_dedup(listings):
    # preprocess: blocked pairs within the listing set (i < j), features, labels
    pairs = [(i, j) for i, j in blocked_pairs(listings, listings) if i < j]
    X = np.array([pair_features(listings.loc[i], listings.loc[j]) for i, j in pairs])
    y = np.array([listings.event_id[i] == listings.event_id[j] for i, j in pairs], dtype=int)
    # blocking recall: how many true duplicates survive the cheap filters at all
    true_dups = sum(len(list(itertools.combinations(g.index, 2))) for _, g in listings.groupby("event_id"))
    print(f"[dedup] blocked pairs {len(pairs)}  duplicates {y.sum()}  blocking recall {y.sum() / true_dups:.3f}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    auto = p >= 0.9
    metrics = {"precision@0.5": precision_score(yte, p >= 0.5), "recall@0.5": recall_score(yte, p >= 0.5),
               "auto_merge_precision": precision_score(yte, auto) if auto.any() else float("nan"),
               "auto_merge_share": float(auto.mean())}
    print("[dedup] " + "  ".join(f"{k} {v:.3f}" for k, v in metrics.items()))
    return model, metrics


def train_category(events):
    Xtr, Xte, ytr, yte = train_test_split(category_text(events), events.category, test_size=0.25,
                                          stratify=events.category, random_state=0)
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                          LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
    proba = model.predict_proba(Xte)
    metrics = {"accuracy": accuracy_score(yte, model.classes_[proba.argmax(1)]),
               "low_confidence_share": float((proba.max(1) < 0.5).mean())}
    print("[category] " + "  ".join(f"{k} {v:.3f}" for k, v in metrics.items()))
    return model, metrics


def main():
    # data
    listings, events = load("listings"), load("events")
    listings["start_time"] = pd.to_datetime(listings.start_time)
    dedup, m1 = train_dedup(listings)
    category, m2 = train_category(events)
    path = save_artifact({"dedup": dedup, "category": category, "model_version": "ingest-v0",
                          "metrics": {"dedup": m1, "category": m2}}, "event_ingestion")
    print("saved", path)


if __name__ == "__main__":
    main()
