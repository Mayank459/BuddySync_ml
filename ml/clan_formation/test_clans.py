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
