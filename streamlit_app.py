"""Entrypoint for Streamlit Community Cloud (Main file path = streamlit_app.py).

It lives at the repo root so Community Cloud installs the lean client deps in the
root requirements.txt — not ml/requirements.txt (mediapipe, opencv, pgserver, ...),
which it would pick up if the entrypoint lived in ml/. Just runs the playground.
"""
import runpy
import sys
from pathlib import Path

ML = Path(__file__).parent / "ml"
sys.path.insert(0, str(ML))  # so `common.*` imports resolve
runpy.run_path(str(ML / "playground.py"), run_name="__main__")
