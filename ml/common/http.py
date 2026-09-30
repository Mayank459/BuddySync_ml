"""Send X-Service-Token on every `requests` call when ML_SERVICE_TOKEN is set (the playground and the smoke
test calling a deployed API, e.g. on Render). Import it once; safe to import again (Streamlit reruns scripts)."""
import os

import requests

TOKEN = os.getenv("ML_SERVICE_TOKEN")

if TOKEN and not getattr(requests.Session.request, "_adds_token", False):
    _request = requests.Session.request

    def _with_token(self, method, url, **kw):
        kw["headers"] = {"X-Service-Token": TOKEN, **(kw.get("headers") or {})}
        return _request(self, method, url, **kw)

    _with_token._adds_token = True
    requests.Session.request = _with_token
