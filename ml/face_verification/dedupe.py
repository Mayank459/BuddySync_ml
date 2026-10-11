"""Duplicate accounts: one face verifying several accounts.

After a liveness check passes, the user's face template (128 numbers from the face model, never an image) is
kept and compared with every other account's. A match never blocks anyone by itself: it sends the check to a
human ("review"). The threshold is far stricter than 1:1 matching because one face is searched against all
accounts (face_verification/train.py: duplicate threshold). Templates are biometric data (DPDP): the app sends
user_id only with the user's consent, and erase() runs when the account is deleted.
"""
import numpy as np

from trip_planner.store import pool

MAX_MATCHES = 5


def matches(emb: np.ndarray, others: list, threshold: float) -> list:
    """others: [(user_id, embedding)]. Accounts whose face is this close, most similar first."""
    if not others:
        return []
    sims = np.array([e for _, e in others], np.float32) @ emb
    return [{"user_id": int(others[i][0]), "similarity": round(float(sims[i]), 3)}
            for i in np.argsort(-sims)[:MAX_MATCHES] if sims[i] >= threshold]


def check_and_keep(user_id: int, emb: np.ndarray, threshold: float, model_version: str) -> list:
    """Compare with every other account's template, then store (or replace) this user's."""
    # ponytail: full scan of all templates per check (~50 MB at 100k users); pgvector + HNSW index past that
    with pool().connection() as c:
        rows = c.execute("SELECT user_id, embedding FROM ml.face_templates WHERE user_id <> %s AND model_version = %s",
                         (user_id, model_version)).fetchall()
        c.execute("INSERT INTO ml.face_templates (user_id, embedding, model_version) VALUES (%s, %s, %s) "
                  "ON CONFLICT (user_id) DO UPDATE SET embedding = EXCLUDED.embedding, "
                  "model_version = EXCLUDED.model_version, updated_at = now()",
                  (user_id, [float(x) for x in emb], model_version))
    return matches(emb, [(r["user_id"], r["embedding"]) for r in rows], threshold)


def erase(user_id: int) -> bool:
    with pool().connection() as c:
        return c.execute("DELETE FROM ml.face_templates WHERE user_id = %s", (user_id,)).rowcount > 0
