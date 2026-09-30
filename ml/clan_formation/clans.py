"""Clan formation: group people going to the same event into clans of 3-6.

No model of its own: it reuses the buddy-matching pair score, then groups greedily so
that *every* pair in a clan clears a threshold ("average without misery").
Run from ml/ to compare against random grouping:  python -m clan_formation.clans
"""
import random

import numpy as np
import pandas as pd

from buddy_matching.model import features, hard_ok, reciprocal


def pair_matrix(user_ids, users, art) -> np.ndarray:
    """Symmetric reciprocal-score matrix; 0 where either side's hard filters fail."""
    n = len(user_ids)
    i, j = np.triu_indices(n, k=1)
    pairs = pd.DataFrame({"viewer_id": np.asarray(user_ids)[i], "candidate_id": np.asarray(user_ids)[j]})
    raw, A, B, _, _ = features(pairs, users, art["vectorizer"])
    R = reciprocal(art["scaler"].transform(raw), art["weights"]) * (hard_ok(A, B) & hard_ok(B, A)).values
    M = np.zeros((n, n))
    M[i, j] = M[j, i] = R
    return M


def form_clans(M: np.ndarray, k=4, tau=0.4, min_size=3):
    """Greedy: seed with the hardest-to-place person, add whoever keeps the group's worst pair highest."""
    unassigned, clans = set(range(len(M))), []
    while len(unassigned) >= min_size:
        options = {u: sum(M[u, v] >= tau for v in unassigned if v != u) for u in unassigned}
        seed = min(options, key=options.get)
        group = [seed]
        while len(group) < k:
            cands = [c for c in unassigned - set(group) if all(M[c, m] >= tau for m in group)]
            if not cands:
                break
            group.append(max(cands, key=lambda c: min(M[c, m] for m in group)))
        unassigned -= {seed}  # seed is tried once either way
        if len(group) >= min_size:
            clans.append(group)
            unassigned -= set(group)
    return clans  # ponytail: O(n^3) greedy; fine for per-event pools of a few hundred


def worst_pair(M, group):
    return min(M[a, b] for x, a in enumerate(group) for b in group[x + 1:])


if __name__ == "__main__":
    from common.utils import load, load_artifact

    users, parts, art = load("users"), load("participants"), load_artifact("buddy_matching")
    wants_group = set(users.loc[users.group_pref != "duo", "user_id"])
    pool = parts[(parts.status == "looking_for_buddy") & parts.user_id.isin(wants_group)]
    event_id = pool.event_id.value_counts().index[0]
    ids = pool.loc[pool.event_id == event_id, "user_id"].tolist()
    M = pair_matrix(ids, users, art)
    clans = form_clans(M)
    random.seed(0)
    shuffled = random.sample(range(len(ids)), len(ids))
    rand = [shuffled[x:x + 4] for x in range(0, len(shuffled) - 3, 4)]
    print(f"event {event_id}: {len(ids)} people -> {len(clans)} clans, "
          f"{sum(map(len, clans))} placed")
    print(f"mean worst-pair score  greedy {np.mean([worst_pair(M, c) for c in clans]):.3f}  "
          f"random {np.mean([worst_pair(M, c) for c in rand]):.3f}")
