"""LLM providers for the trip planner: Claude (native SDK) or OpenAI-compatible APIs (Gemini, Groq, ...).

Config (env), design §11:
  ANTHROPIC_API_KEY                         Claude for every task (CLAUDE_MODEL, default claude-opus-5)
  LLM_BASE_URL + LLM_API_KEY + LLM_MODEL    primary OpenAI-compatible provider
  LLM_DRAFT_MODEL / LLM_CHAT_MODEL / LLM_UTILITY_MODEL   per-task override of LLM_MODEL
  LLM_FALLBACK_BASE_URL + LLM_FALLBACK_API_KEY + LLM_FALLBACK_MODEL   second provider, tried last
Every model setting is a comma list: on 429/5xx the next model is tried at once (no client-side waiting:
free tiers limit each model separately, so e.g. three Groq models give three times the quota), then the next provider.
"""
import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache


class NotConfigured(Exception):
    pass


class Busy(Exception):
    """Every model and provider is rate-limited or overloaded (free tiers): retry later."""


class UpstreamError(Exception):
    pass


@dataclass
class Call:
    id: str
    name: str
    args: dict


@dataclass
class Reply:
    kind: str  # anthropic | openai
    text: str
    calls: list
    message: dict  # assistant message to append to this turn's history
    model: str
    provider: int
    tokens_in: int = 0
    tokens_out: int = 0
    refused: bool = False

    def result_messages(self, results: dict) -> list:
        if self.kind == "anthropic":
            return [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": c.id, "content": results[c.id]}
                                                 for c in self.calls]}]
        return [{"role": "tool", "tool_call_id": c.id, "content": results[c.id]} for c in self.calls]


@dataclass
class Provider:
    kind: str
    client: object
    models: dict = field(default_factory=dict)  # task -> [model, ...]


def _models(prefix, default):
    lists = lambda v: [m.strip() for m in v.split(",") if m.strip()]
    base = lists(os.getenv(default, ""))
    return {t: lists(os.getenv(f"{prefix}_{t.upper()}_MODEL", "")) or base for t in ("draft", "chat", "utility")}


@lru_cache
def providers() -> tuple:
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        m = [os.getenv("CLAUDE_MODEL", "claude-opus-5")]
        return (Provider("anthropic", anthropic.Anthropic(max_retries=1), {t: m for t in ("draft", "chat", "utility")}),)
    out = []
    import openai
    if all(os.getenv(k) for k in ("LLM_BASE_URL", "LLM_API_KEY")) and any(_models("LLM", "LLM_MODEL").values()):
        out.append(Provider("openai", openai.OpenAI(base_url=os.environ["LLM_BASE_URL"],
                                                    api_key=os.environ["LLM_API_KEY"], max_retries=0),
                            _models("LLM", "LLM_MODEL")))
    if all(os.getenv(k) for k in ("LLM_FALLBACK_BASE_URL", "LLM_FALLBACK_API_KEY", "LLM_FALLBACK_MODEL")):
        fb = [m.strip() for m in os.environ["LLM_FALLBACK_MODEL"].split(",")]
        out.append(Provider("openai", openai.OpenAI(base_url=os.environ["LLM_FALLBACK_BASE_URL"],
                                                    api_key=os.environ["LLM_FALLBACK_API_KEY"], max_retries=0),
                            {t: fb for t in ("draft", "chat", "utility")}))
    return tuple(out)


def configured() -> bool:
    return bool(providers())


def openai_tools(specs):
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in specs]


def complete(task, system, messages, tools=None, json_mode=False, pin=None) -> Reply:
    """One model call. pin=provider index keeps a multi-step turn on one provider (message formats differ)."""
    provs = providers()
    if not provs:
        raise NotConfigured("set ANTHROPIC_API_KEY, or LLM_BASE_URL + LLM_API_KEY + LLM_MODEL")
    order = [pin] if pin is not None else range(len(provs))
    busy = None
    for idx in order:
        p = provs[idx]
        for model in p.models[task]:
            try:
                return (_anthropic if p.kind == "anthropic" else _openai)(p, idx, model, system, messages, tools, json_mode)
            except Busy as e:
                busy = e
    raise busy or Busy("no model available")


def _anthropic(p, idx, model, system, messages, tools, json_mode):
    import anthropic
    kw = {"tools": tools} if tools else {}
    try:
        resp = p.client.messages.create(model=model, max_tokens=16000, messages=messages, **kw,
                                        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}])
    except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError) as e:
        raise Busy(str(e))
    except anthropic.APIStatusError as e:
        if e.status_code == 529:
            raise Busy(str(e))
        raise UpstreamError(f"{type(e).__name__}: {e.status_code}")
    calls = [Call(b.id, b.name, b.input) for b in resp.content if b.type == "tool_use"]
    text = "".join(b.text for b in resp.content if b.type == "text")
    return Reply("anthropic", text, calls, {"role": "assistant", "content": resp.content}, resp.model, idx,
                 resp.usage.input_tokens, resp.usage.output_tokens, refused=resp.stop_reason == "refusal")


def _openai(p, idx, model, system, messages, tools, json_mode):
    import openai
    kw = {}
    if tools:
        kw["tools"] = openai_tools(tools)
    if json_mode:
        kw["response_format"] = {"type": "json_object"}
    try:
        resp = p.client.chat.completions.create(model=model, messages=[{"role": "system", "content": system}] + messages, **kw)
    except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError) as e:
        raise Busy(str(e))
    except openai.APIStatusError as e:
        if e.status_code == 413:  # free tiers cap one request's size (Groq: 8k tokens): try the next model/provider
            raise Busy(str(e))
        raise UpstreamError(f"{type(e).__name__}: {e.status_code}")
    msg = resp.choices[0].message
    calls = []
    for c in msg.tool_calls or []:
        try:
            args = json.loads(c.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {"_bad_json": c.function.arguments}
        calls.append(Call(c.id, c.function.name, args))
    u = resp.usage
    return Reply("openai", msg.content or "", calls, msg.model_dump(exclude_none=True),  # keeps Gemini thought signatures
                 resp.model or model, idx, getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0)


def parse_json(text: str) -> dict:
    """Models sometimes wrap JSON in fences or prose: take the outermost object."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("no JSON object in reply")
    return json.loads(m.group(0))
