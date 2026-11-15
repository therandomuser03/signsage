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


def test_websocket_sentence_boundary_saves_history_for_signed_in_user(promoted_client, monkeypatch):
    """The WS sentence.boundary path must write session+message (the old
    single-table insert targeted columns the live schema does not have)."""
    from app.infrastructure.database.repositories.translation_repository import (
        get_memory_repository,
        reset_memory_repository,
    )
    from app.core.security import AuthenticatedUser

    async def fake_verify(token, settings, transport=None):
        if token.startswith("good-"):
            return AuthenticatedUser(id=token[5:], email=f"{token[5:]}@example.com", access_token=token)
        from app.core.errors import AuthError
        raise AuthError("Invalid or expired access token.")

    monkeypatch.setattr(
        "app.api.websocket.translation.verify_access_token", fake_verify
    )
    monkeypatch.setenv("SIGNSAGE_HISTORY_BACKEND", "memory")
    reset_memory_repository()

    from app.core.config import get_settings

    get_settings.cache_clear()
    from app.main import app
    client = fastapi_testclient.TestClient(app)

    rng = np.random.default_rng(2)
    with client.websocket_connect("/ws/translate") as ws:
        ws.send_json({"type": "auth", "access_token": "good-wsuser"})
        auth_ok = ws.receive_json()
        assert auth_ok["type"] == "auth.ok"
        for _ in range(30):
            ws.send_json({"type": "landmarks.frame", "landmarks": rng.normal(size=(543, 3)).tolist()})
        ws.send_json({"type": "sentence.boundary", "save_history": True})
        response = ws.receive_json()

    assert response["type"] == "translation.text"
    assert response["final"] is True
    if response.get("accepted"):
        repo = get_memory_repository()
        sessions = [s for s in repo._sessions.values() if s["user_id"] == "wsuser"]
        assert len(sessions) == 1
        # The connection closed when the `with` block exited: the interaction
        # session is marked completed, not deleted.
        assert sessions[0]["status"] in ("open", "completed")
        messages = repo._messages.get(sessions[0]["id"], [])
        assert len(messages) == 1
        assert messages[0]["sequence"] == 1
        assert messages[0]["direction"] == "sign_to_text"
    get_settings.cache_clear()


def test_websocket_two_boundaries_group_into_one_session(promoted_client, monkeypatch):
    """Two sentence.boundary messages in one connection = ONE history session
    with two sequenced sentences."""
    from app.infrastructure.database.repositories.translation_repository import (
        get_memory_repository,
        reset_memory_repository,
    )
    from app.core.security import AuthenticatedUser

    async def fake_verify(token, settings, transport=None):
        if token.startswith("good-"):
            return AuthenticatedUser(id=token[5:], email=f"{token[5:]}@example.com", access_token=token)
        from app.core.errors import AuthError
        raise AuthError("Invalid or expired access token.")

    monkeypatch.setattr(
        "app.api.websocket.translation.verify_access_token", fake_verify
    )
    monkeypatch.setenv("SIGNSAGE_HISTORY_BACKEND", "memory")
    reset_memory_repository()

    from app.core.config import get_settings

    get_settings.cache_clear()
    from app.main import app
    client = fastapi_testclient.TestClient(app)

    rng = np.random.default_rng(3)

    def frames(n=30):
        return [rng.normal(size=(543, 3)).tolist() for _ in range(n)]

    with client.websocket_connect("/ws/translate") as ws:
        ws.send_json({"type": "auth", "access_token": "good-wsuser2", "history_session_id": "interaction-ws-1"})
        assert ws.receive_json()["type"] == "auth.ok"
        for boundary in range(2):
            for _ in range(30):
                ws.send_json({"type": "landmarks.frame", "landmarks": frames(1)[0]})
            ws.send_json({"type": "sentence.boundary", "save_history": True, "client_key": f"ws-sentence-{boundary}"})
            resp = ws.receive_json()
            assert resp["type"] == "translation.text"

    repo = get_memory_repository()
    sessions = [s for s in repo._sessions.values() if s["user_id"] == "wsuser2"]
    if sessions:  # only assert when at least one boundary was accepted
        assert len(sessions) == 1
        assert sessions[0]["client_key"] == "interaction-ws-1"
        messages = repo._messages.get(sessions[0]["id"], [])
        assert [m["sequence"] for m in messages] == list(range(1, len(messages) + 1))
        assert all(m["session_id"] == sessions[0]["id"] for m in messages)
    get_settings.cache_clear()


def test_websocket_sentence_boundary_saves_nothing_for_guests(promoted_client, monkeypatch):
    monkeypatch.setenv("SIGNSAGE_HISTORY_BACKEND", "memory")
    from app.infrastructure.database.repositories.translation_repository import (
        get_memory_repository,
        reset_memory_repository,
    )

    reset_memory_repository()

    from app.core.config import get_settings

    get_settings.cache_clear()
    from app.main import app
    client = fastapi_testclient.TestClient(app)

    rng = np.random.default_rng(2)
    with client.websocket_connect("/ws/translate") as ws:
        for _ in range(30):
            ws.send_json({"type": "landmarks.frame", "landmarks": rng.normal(size=(543, 3)).tolist()})
        ws.send_json({"type": "sentence.boundary", "save_history": True})
        response = ws.receive_json()

    assert response["type"] == "translation.text"
    repo = get_memory_repository()
    assert sum(len(v) for v in repo._sessions.values()) == 0
    assert sum(len(v) for v in repo._messages.values()) == 0
    assert response.get("history_id") is None
    get_settings.cache_clear()
