"""Integration test for the /ws/translate route end-to-end: connect, stream
landmark frames, signal a sentence boundary, and confirm a real translation
comes back -- using the fake_bundle fixture as a stand-in for a promoted
model. Also covers the two most important error paths.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
fastapi_testclient = pytest.importorskip("fastapi.testclient")


@pytest.fixture
def promoted_client(fake_bundle, monkeypatch):
    monkeypatch.setenv("SIGNSAGE_MODEL_BUNDLE_DIR", str(fake_bundle))
    monkeypatch.setenv("SIGNSAGE_MODEL_PROMOTED", "true")

    from app.core.config import get_settings

    get_settings.cache_clear()

    from app.ml.models.registry import reset_registry

    reset_registry()

    from app.main import app

    client = fastapi_testclient.TestClient(app)
    yield client

    get_settings.cache_clear()
    reset_registry()


def test_websocket_happy_path_returns_a_translation(promoted_client):
    rng = np.random.default_rng(2)
    with promoted_client.websocket_connect("/ws/translate") as ws:
        for _ in range(30):
            ws.send_json({"type": "landmarks.frame", "landmarks": rng.normal(size=(543, 3)).tolist()})
        ws.send_json({"type": "sentence.boundary"})
        response = ws.receive_json()

    assert response["type"] == "translation.text"
    assert response["accepted"] is True
    assert isinstance(response["text"], str)


def test_websocket_rejects_unknown_message_type(promoted_client):
    with promoted_client.websocket_connect("/ws/translate") as ws:
        ws.send_json({"type": "not.a.real.type"})
        response = ws.receive_json()

    assert response["type"] == "error"
    assert response["code"] == "UNKNOWN_MESSAGE_TYPE"


def test_websocket_rejects_sentence_boundary_with_no_buffered_frames(promoted_client):
    with promoted_client.websocket_connect("/ws/translate") as ws:
        ws.send_json({"type": "sentence.boundary"})
        response = ws.receive_json()

    assert response["type"] == "error"
    assert response["code"] == "EMPTY_SEQUENCE"


def test_rolling_interim_then_final(promoted_client):
    """Streaming semantics: a trailing-window interim arrives with final=false
    while signing continues; sentence.boundary finalizes with final=true.
    Exactly 80 frames hits the default window once, so exactly one interim
    is expected before the boundary."""
    rng = np.random.default_rng(11)
    with promoted_client.websocket_connect("/ws/translate") as ws:
        for _ in range(80):
            ws.send_json({"type": "landmarks.frame", "landmarks": rng.normal(size=(543, 3)).tolist()})
        interim = ws.receive_json()
        ws.send_json({"type": "sentence.boundary"})
        final = ws.receive_json()

    assert interim["type"] == "translation.text" and interim["final"] is False
    assert interim["accepted"] is True
    assert final["type"] == "translation.text" and final["final"] is True


def test_rest_endpoint_refuses_when_model_not_promoted(fake_bundle, monkeypatch):
    """With the fallback disabled, SIGNSAGE_MODEL_PROMOTED=false must refuse
    to serve, even with a perfectly loadable bundle available."""
    monkeypatch.setenv("SIGNSAGE_MODEL_BUNDLE_DIR", str(fake_bundle))
    monkeypatch.setenv("SIGNSAGE_MODEL_PROMOTED", "false")
    monkeypatch.setenv("SIGNSAGE_FALLBACK_ENABLED", "false")

    from app.core.config import get_settings

    get_settings.cache_clear()
    from app.ml.models.registry import reset_registry

    reset_registry()
    from app.main import app

    client = fastapi_testclient.TestClient(app)
    frames = [{"landmarks": np.zeros((543, 3)).tolist()} for _ in range(5)]
    resp = client.post("/api/v1/translation/sign-to-text", json={"frames": frames})

    assert resp.status_code == 503
    assert resp.json()["code"] == "MODEL_NOT_PROMOTED"

    get_settings.cache_clear()
    reset_registry()


def test_rest_endpoint_serves_heuristic_fallback_when_gated(monkeypatch):
    """Default config (unpromoted, no bundle): the heuristic fallback serves
    a 5-sign vocabulary instead of MODEL_NOT_PROMOTED."""
    monkeypatch.delenv("SIGNSAGE_MODEL_BUNDLE_DIR", raising=False)
    monkeypatch.setenv("SIGNSAGE_MODEL_PROMOTED", "false")

    from app.core.config import get_settings

    get_settings.cache_clear()
    from app.ml.models.registry import reset_registry

    reset_registry()
    from app.main import app

    client = fastapi_testclient.TestClient(app)

    caps = client.get("/api/v1/capabilities").json()
    assert caps["sign_to_text_available"] is True
    assert caps["sign_to_text_source"] == "heuristic-fallback"
    assert set(caps["fallback_vocabulary"]) == {"HELLO", "YES", "NO", "YOU", "HELP"}

    # All-zero frames: no hand -> honest rejection, not a fabricated guess.
    frames = [{"landmarks": np.zeros((543, 3)).tolist()} for _ in range(30)]
    resp = client.post("/api/v1/translation/sign-to-text", json={"frames": frames})
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is False
    assert body["text"] == ""

    get_settings.cache_clear()
    reset_registry()
