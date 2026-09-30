"""Voice input (design §1.6): speech → text with Whisper through an OpenAI-compatible API (Groq by default).

STT_API_KEY (falls back to LLM_FALLBACK_API_KEY), STT_BASE_URL (default Groq), STT_MODEL (whisper-large-v3-turbo).
Hindi and Hinglish work; the text goes back to the app, which sends it as a normal chat message.
"""
import os
from functools import lru_cache

MAX_BYTES = 10 * 1024 * 1024  # ~5 min of compressed audio
TYPES = {"audio/webm", "audio/ogg", "audio/mpeg", "audio/mp4", "audio/m4a", "audio/x-m4a", "audio/wav",
         "audio/x-wav", "audio/wave", "audio/flac", "video/webm"}


class NotConfigured(Exception):
    pass


@lru_cache
def client():
    import openai
    key = os.getenv("STT_API_KEY") or os.getenv("LLM_FALLBACK_API_KEY")
    if not key:
        raise NotConfigured("voice input not configured: set STT_API_KEY (e.g. a Groq key)")
    return openai.OpenAI(base_url=os.getenv("STT_BASE_URL", "https://api.groq.com/openai/v1"), api_key=key,
                         max_retries=1)


def transcribe(filename: str, data: bytes) -> dict:
    r = client().audio.transcriptions.create(model=os.getenv("STT_MODEL", "whisper-large-v3-turbo"),
                                             file=(filename or "voice.webm", data), response_format="verbose_json")
    return {"text": (r.text or "").strip(), "lang": getattr(r, "language", None),
            "seconds": getattr(r, "duration", None)}
