import numpy as np

from face_verification.face import decide, embed

T = {"pass": 0.6, "review": 0.4}


def test_bands():
    assert [decide(s, T) for s in (0.7, 0.5, 0.1)] == ["pass", "review", "fail"]


def test_blank_image_is_rejected_not_guessed():
    emb, reason = embed(np.full((300, 300, 3), 128, np.uint8))
    assert emb is None and reason == "no_face"
