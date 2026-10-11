import numpy as np

from clan_formation.clans import form_clans


def test_every_pair_clears_threshold_and_nobody_twice():
    rng = np.random.default_rng(0)
    M = rng.random((40, 40))
    M = (M + M.T) / 2
    clans = form_clans(M, k=4, tau=0.4)
    members = [m for c in clans for m in c]
    assert len(members) == len(set(members))
    assert all(3 <= len(c) <= 4 for c in clans)
    assert all(M[a, b] >= 0.4 for c in clans for a in c for b in c if a != b)


def test_incompatible_pool_forms_nothing():
    assert form_clans(np.zeros((10, 10))) == []


def test_never_again_pair_scores_zero():
    import pandas as pd
    from sklearn.preprocessing import FunctionTransformer

    from buddy_matching.model import DEFAULT_WEIGHTS, fit_text
    from buddy_matching.test_buddy import user
    from clan_formation.clans import pair_matrix
    users = pd.DataFrame([user(1), user(2), user(3)])
    art = {"vectorizer": fit_text(users), "scaler": FunctionTransformer(), "weights": DEFAULT_WEIGHTS}
    M = pair_matrix([1, 2, 3], users, art, avoid={(1, 2), (2, 1)})
    assert M[0, 1] == M[1, 0] == 0 and M[0, 2] > 0
