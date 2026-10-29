"""Send X-Service-Token on `requests` calls to the API when ML_SERVICE_TOKEN is set (the playground and the smoke
test calling a deployed API, e.g. on Render). Import it once; safe to import again (Streamlit reruns scripts).
The token goes only to API_URL's origin: never to any other host, e.g. a URL typed into the playground."""
import os
from urllib.parse import urlsplit

import requests

TOKEN = os.getenv("ML_SERVICE_TOKEN")
DEFAULT_URL = "https://buddysync-ml.onrender.com" if TOKEN else "http://127.0.0.1:8765"
API = os.getenv("API_URL", DEFAULT_URL)


def _origin(url: str):
    p = urlsplit(url)
    return p.scheme.lower(), (p.hostname or "").lower(), p.port


def token_headers(url: str) -> dict:
    return {"X-Service-Token": TOKEN} if TOKEN and _origin(url) == _origin(API) else {}


if TOKEN and not getattr(requests.Session.request, "_adds_token", False):
    _request = requests.Session.request

    def _with_token(self, method, url, **kw):
        kw["headers"] = {**token_headers(url), **(kw.get("headers") or {})}
        return _request(self, method, url, **kw)

    _with_token._adds_token = True
    requests.Session.request = _with_token
