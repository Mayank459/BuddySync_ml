"""Shared helpers: where data/models live, loading them, and distance."""
import math
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"


def load(name: str) -> pd.DataFrame:
    # ponytail: CSVs from common/synthetic.py; swap for read-only Postgres queries once the schema exists
    return pd.read_csv(DATA / f"{name}.csv")


def save_artifact(obj, name: str) -> Path:
    ARTIFACTS.mkdir(exist_ok=True)
    path = ARTIFACTS / f"{name}.joblib"
    joblib.dump(obj, path)
    return path


def load_artifact(name: str):
    return joblib.load(ARTIFACTS / f"{name}.joblib")


def haversine_km(lat1, lng1, lat2, lng2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))
