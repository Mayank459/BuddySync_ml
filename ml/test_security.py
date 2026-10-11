"""Regression tests for the security review: token scope, upload/decoding limits, body size, public paths,
and the /v1/go allowlist. Each one failed (or would have) before its fix."""
import os
import struct
import tempfile
import zlib
from fractions import Fraction

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as app_module
import common.http as http
from face_verification import liveness
from trip_planner import links

client = TestClient(app_module.app)  # no `with`: the lifespan (trip workers, database) doesn't start


def png_bomb(side=16000) -> bytes:
    """~250 KB on disk, 0.77 GB once decoded."""
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    z = zlib.compressobj(9)
    body = b"".join(z.compress(b"\x00" * (side + 1)) for _ in range(side)) + z.flush()
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", body) + chunk(b"IEND", b""))


def test_token_only_goes_to_the_api(monkeypatch):
    monkeypatch.setattr(http, "TOKEN", "secret")
    monkeypatch.setattr(http, "API", "https://buddysync-ml.onrender.com")
    assert http.token_headers("https://buddysync-ml.onrender.com/v1/match/buddies") == {"X-Service-Token": "secret"}
    for url in ("https://attacker.example/v1/match/buddies", "http://buddysync-ml.onrender.com/v1/x",
                "https://buddysync-ml.onrender.com.attacker.example/", "https://buddysync-ml.onrender.com:8443/"):
        assert http.token_headers(url) == {}, url


def test_image_bomb_is_refused_before_decoding():
    r = client.post("/v1/verify/face", files=[("selfie", ("bomb.png", png_bomb(), "image/png")),
                                              ("profile_photos", ("p.png", png_bomb(100), "image/png"))])
    assert r.status_code == 413 and "megapixels" in r.json()["detail"]


def test_too_many_profile_photos():
    small = png_bomb(100)
    r = client.post("/v1/verify/face", files=[("selfie", ("s.png", small, "image/png"))]
                    + [("profile_photos", (f"{i}.png", small, "image/png")) for i in range(6)])
    assert r.status_code == 413


def test_oversized_or_unsized_body_is_refused():
    big = {"content-length": str(app_module.MAX_BODY + 1), "content-type": "application/json"}
    assert client.post("/v1/moderate/message", content=b"{}", headers=big).status_code == 413
    assert client.post("/v1/moderate/message", content=iter([b"{}"])).status_code == 411  # chunked, no length


def test_public_path_is_exact(monkeypatch):
    monkeypatch.setenv("ML_SERVICE_TOKEN", "secret")
    assert client.get("/v1/gofoo").status_code == 401  # was public by prefix
    assert client.get("/v1/go", params={"url": "https://evil.example/"}).status_code == 400  # public, still checked


def test_go_allowlist_blocks_google_redirectors():
    assert links.allowed("https://www.google.com/maps/search/?api=1&query=1,2")
    assert links.allowed("https://www.google.com/travel/flights?q=x")
    assert links.allowed("https://www.booking.com/searchresults.html?ss=Pune")
    for url in ("https://www.google.com/url?q=https://evil.example", "https://www.google.com/amp/s/evil.example",
                "http://www.booking.com/", "https://booking.com.evil.example/"):
        assert not links.allowed(url), url


def test_video_whose_timestamps_never_advance_is_cut_off():
    av = pytest.importorskip("av")
    path = os.path.join(tempfile.mkdtemp(), "squashed.mp4")
    with av.open(path, "w") as out:
        s = out.add_stream("libx264", rate=30)
        s.width = s.height = 64
        s.time_base = Fraction(1, 1_000_000)
        for n in range(liveness.MAX_FRAMES + 10):  # 1 microsecond apart: the "video" lasts 5 ms
            f = av.VideoFrame.from_ndarray(np.full((64, 64, 3), n % 255, np.uint8), format="rgb24")
            f.pts, f.time_base = n, s.time_base
            for pkt in s.encode(f):
                out.mux(pkt)
        for pkt in s.encode():
            out.mux(pkt)
    with pytest.raises(OverflowError, match="frames"):
        liveness.analyze_video(path, ["blink"])


def test_frames_bigger_than_4k_are_refused():
    import cv2
    path = os.path.join(tempfile.mkdtemp(), "huge.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (4096, 2304))
    for _ in range(2):
        w.write(np.zeros((2304, 4096, 3), np.uint8))
    w.release()
    with pytest.raises(OverflowError, match="4K"):
        liveness.analyze_video(path, ["blink"])
