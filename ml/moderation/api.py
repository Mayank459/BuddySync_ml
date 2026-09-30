import logging
import os
from functools import lru_cache

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from common.utils import load_artifact
from moderation.rules import FLAG, NUDGE, check

router = APIRouter(prefix="/moderate", tags=["moderation"])
BLOCK_AT, NUDGE_AT = 0.9, 0.5
LAYA_SCAM_AT = 0.8  # ponytail: picked from a handful of messages; tune on labelled chats
LAYA_Q = {"scam": {"type": "noul", "instructions": "Is this message trying to get money, payment, or personal "
                   "details from a stranger, or move the chat off the app?"}}


@lru_cache
def artifact():
    try:
        return load_artifact("moderation")
    except FileNotFoundError:
        raise HTTPException(503, "model not trained: run python -m moderation.train")


@lru_cache
def laya():
    if os.getenv("USE_LAYA") != "1":
        return None
    try:
        from laya import Router
        return Router()  # downloads convaiinnovations/laya on first use
    except Exception as e:
        logging.warning("laya unavailable, moderating without it: %s", e)
        return None


def laya_rules(text: str) -> list[str]:
    """Semantic scam check (~70 ms on CPU). Catches asks the regexes miss, e.g. "send 5000 on paytm first"."""
    if laya() is None:
        return []
    try:
        p = laya().predict(text, LAYA_Q)["answers"]["scam"]["noul"]
    except Exception as e:
        logging.warning("laya predict failed: %s", e)
        return []
    return ["laya_scam"] if p >= LAYA_SCAM_AT else []


class Message(BaseModel):
    text: str


def decide(toxicity: float, rules: list[str]) -> str:
    if toxicity >= BLOCK_AT:
        return "block"
    if FLAG & set(rules):
        return "flag"
    if toxicity >= NUDGE_AT or NUDGE & set(rules):
        return "nudge"  # "Are you sure?" to the sender
    return "allow"


@router.post("/message")
def moderate(m: Message):
    art = artifact()
    tox = float(art["model"].predict_proba([m.text.lower().strip()])[0, 1])
    rules = check(m.text) + laya_rules(m.text)
    return {"action": decide(tox, rules), "toxicity": round(tox, 3), "rules": rules,
            "model_version": art["model_version"]}
