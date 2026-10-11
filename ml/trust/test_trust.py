import pandas as pd

from trust.model import NEUTRAL, attended, never_again, profile, table


def peer(rows):
    return pd.DataFrame(rows, columns=["event_id", "rater_id", "ratee_id", "showed_up", "would_join_again"])


def test_new_user_is_neutral_and_one_flake_does_not_sink_anyone():
    t = table(peer([(1, 2, 1, False, False)]))
    assert profile(9, t, pd.Series(dtype=int))["reliability"] == round(NEUTRAL, 3)
    one_bad = profile(1, t, pd.Series(dtype=int))
    assert one_bad["level"] == "new" and 0.6 < one_bad["reliability"] < NEUTRAL


def test_levels_follow_the_record():
    good = [(e, 10 + e, 1, True, True) for e in range(6)]
    flaky = [(e, 10 + e, 2, e % 3 == 0, False) for e in range(6)]
    t = table(peer(good + flaky))
    assert profile(1, t, pd.Series(dtype=int))["level"] == "reliable"
    assert profile(2, t, pd.Series(dtype=int))["level"] == "low"
    assert profile(1, t, pd.Series(dtype=int))["showed_up"] == "6 of 6"


def test_attended_counts_each_past_event_once():
    p = peer([(1, 5, 1, True, True), (1, 6, 1, True, True), (2, 5, 1, False, False)])
    ratings = pd.DataFrame({"user_id": [1, 1], "event_id": [1, 3], "rating": [4, 5]})
    assert attended(p, ratings)[1] == 2  # event 1 (rated + seen) and 3; a no-show at event 2 doesn't count


def test_never_again_keeps_pairs_apart_both_ways():
    pairs = never_again(peer([(1, 5, 6, True, False), (1, 7, 8, True, True)]))
    assert (5, 6) in pairs and (6, 5) in pairs and (7, 8) not in pairs
