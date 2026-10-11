import pandas as pd

from event_recs.model import fit_text, taste


def test_ratings_pull_and_push_taste():
    events = pd.DataFrame({"event_id": [1, 2, 3], "title": ["Rock night", "Comedy hour", "Comedy hour two"],
                           "category": ["concert", "comedy", "comedy"], "tags": ["rock", "stand-up comedy", "stand-up comedy"],
                           "performers": ["Static Tides", "Kavya", "Aarav"], "description": ["", "", ""]})
    users = pd.DataFrame({"interests": ["rock;stand-up comedy"]})
    vec = fit_text(events, users)
    user, none = users.iloc[0], pd.DataFrame(columns=["event_id", "status"])
    upcoming = vec.transform(["Comedy hour two comedy stand-up comedy Aarav"])

    def comedy_fit(stars):
        rated = None if stars is None else pd.DataFrame({"event_id": [2], "rating": [stars]})
        return float(upcoming.dot(taste(user, none, events, vec, rated))[0])

    assert comedy_fit(1) < comedy_fit(None) < comedy_fit(5)
