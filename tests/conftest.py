"""Shared fixtures: a mock-backed app, a logged-in admin client, sketch helpers."""
from __future__ import annotations

import io
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import protocol
from server.app import create_app
from server.backends.mock import MockBackend
from server.config import Config

ADMIN_PIN = "4321"


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(backend="mock", admin_pin=ADMIN_PIN, secret_key="test-secret",
                  data_dir=str(tmp_path / "data"), size=384)


@pytest.fixture
def make_client(config: Config):
    """Factory: ``with make_client(delay_ms) as client`` runs the app incl. lifespan."""
    def factory(delay_ms: int = 1) -> TestClient:
        return TestClient(create_app(config, backend=MockBackend(mock_delay_ms=delay_ms)))
    return factory


@pytest.fixture
def client(make_client):
    with make_client() as c:
        yield c


@pytest.fixture
def admin(client: TestClient) -> TestClient:
    r = client.post("/api/admin/login", json={"pin": ADMIN_PIN})
    assert r.status_code == 200, r.text
    return client


def sketch_png(blank: bool = False, size: int = 256, color: str = "black") -> bytes:
    img = Image.new("RGB", (size, size), "white")
    if not blank:
        ImageDraw.Draw(img).line((20, 20, size - 20, size - 40), fill=color, width=8)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def current_prompt(app) -> str:
    """The prompt the server builds for the kid's current topic and style."""
    current = app.state.session.current()
    return app.state.styles.build_prompt(current["topic"], current["style"])


def sketch_frame(seq: int, blank: bool = False) -> bytes:
    return protocol.pack(protocol.SKETCH, seq, sketch_png(blank))


def recv(ws) -> tuple[str, Any]:
    """Receive one message as ("json", dict) or ("bin", (type, seq, payload))."""
    msg = ws.receive()
    if msg.get("type") == "websocket.close":
        return "close", msg.get("code")
    if msg.get("bytes") is not None:
        return "bin", protocol.unpack(msg["bytes"])
    return "json", json.loads(msg["text"])


def recv_until(ws, pred: Callable[[str, Any], bool], limit: int = 200) -> tuple[str, Any]:
    """Receive until ``pred(kind, value)`` holds; skips status/stats noise."""
    for _ in range(limit):
        kind, value = recv(ws)
        if pred(kind, value):
            return kind, value
        if kind == "close":
            raise AssertionError(f"socket closed (code {value}) before expected message")
    raise AssertionError("expected message never arrived")


def is_json(t: str) -> Callable[[str, Any], bool]:
    return lambda kind, value: kind == "json" and value.get("t") == t


def is_picture(seq: int | None = None) -> Callable[[str, Any], bool]:
    return lambda kind, value: kind == "bin" and (seq is None or value[1] == seq)
