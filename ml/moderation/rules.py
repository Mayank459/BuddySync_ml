"""Deterministic scam / contact-sharing rules. Cheap, explainable, and catch most MVP risk."""
import re

RULES = {
    "phone_number": re.compile(r"(?:\+?91[\s-]?)?\b[6-9]\d{9}\b"),
    "upi_id": re.compile(r"\b[\w.\-]{2,}@(?:ok\w+|ybl|paytm|upi|ibl|axl|apl)\b", re.I),
    "link": re.compile(r"https?://|www\.", re.I),
    "off_platform": re.compile(r"\b(whats\s?app|telegram|insta(?:gram)?\s?dm|snap(?:chat)?)\b", re.I),
    "money_request": re.compile(r"\b(send|transfer|bhej\w*|pay)\b.{0,30}\b(money|paise|rs\.?|₹|advance|payment|upi)\b", re.I),
}
FLAG = {"upi_id", "money_request", "laya_scam"}  # likely scam (laya_scam comes from api.laya_rules): warn recipient + moderator queue
NUDGE = {"phone_number", "off_platform", "link"}  # risky this early: warn sender


def check(text: str) -> list[str]:
    return [name for name, rx in RULES.items() if rx.search(text)]
