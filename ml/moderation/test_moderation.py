from moderation import api
from moderation.api import decide
from moderation.rules import check


def test_rules():
    assert check("call me on 9876543210") == ["phone_number"]
    assert "upi_id" in check("pay to rahul.k@okaxis")
    assert "money_request" in check("bhai ticket ke liye advance bhej do 500 rs")
    assert "off_platform" in check("add me on WhatsApp")
    assert check("let's meet at gate 2 at 6") == []


def test_actions():
    assert decide(0.95, []) == "block"
    assert decide(0.1, ["upi_id"]) == "flag"
    assert decide(0.1, ["phone_number"]) == "nudge"
    assert decide(0.6, []) == "nudge"
    assert decide(0.1, []) == "allow"


class FakeLaya:
    def __init__(self, p): self.p = p
    def predict(self, text, q): return {"answers": {"scam": {"noul": self.p}}}


def test_laya_scam(monkeypatch):
    monkeypatch.setattr(api, "laya", lambda: FakeLaya(0.98))
    assert api.laya_rules("send me 5000 on paytm first") == ["laya_scam"]
    assert decide(0.1, ["laya_scam"]) == "flag"
    monkeypatch.setattr(api, "laya", lambda: FakeLaya(0.2))
    assert api.laya_rules("see you at gate 2") == []
    monkeypatch.setattr(api, "laya", lambda: None)  # off or not installed -> rules only
    assert api.laya_rules("send me 5000 on paytm first") == []
