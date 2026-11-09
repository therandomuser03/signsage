"""Integration tests for the word-level recognition REST routes."""
from __future__ import annotations

import os

import numpy as np
import pytest

BUNDLE_DIR = os.environ.get(
    "SIGNSAGE_WORD_MODEL_DIR",
    "/mnt/Drive-D/Codes/signsage/models/word_v3",
)


@pytest.fixture
def make_client(monkeypatch):
    """Factory for a TestClient with chosen env."""
    pytest.importorskip("torch")
    from fastapi.testclient import TestClient

    from app.core.config import get_settings
    from app.core.errors import AuthError
    from app.core.security import AuthenticatedUser, clear_auth_cache
    from app.ml.models.registry import reset_registry
    from app.infrastructure.database.repositories.translation_repository import (
        reset_memory_repository,
    )

    async def fake_verify(token, settings, transport=None):
        from app.core.security import AuthenticatedUser

        if token.startswith("good-"):
            return AuthenticatedUser(id=token[5:], email=f"{token[5:]}@example.com", access_token=token)
        raise AuthError("Invalid or expired access token.")

    def _reset():
        get_settings.cache_clear()
        reset_registry()
        reset_memory_repository()
        clear_auth_cache()

    def _make(**env):
        monkeypatch.setenv("SIGNSAGE_SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SIGNSAGE_SUPABASE_PUBLISHABLE_KEY", "pub")
        monkeypatch.setenv("SIGNSAGE_HISTORY_BACKEND", "memory")
        monkeypatch.setenv("SIGNSAGE_MODEL_PROMOTED", "false")
        monkeypatch.setenv("SIGNSAGE_ALLOW_UNPROMOTED_DEV", "true")
        monkeypatch.setattr(
            "app.services.authentication_service.verify_access_token", fake_verify
        )
        monkeypatch.setattr(
            "app.api.websocket.translation.verify_access_token", fake_verify
        )
        for k, v in env.items():
            monkeypatch.setenv(f"SIGNSAGE_{k.upper()}", str(v))
        _reset()
        from app.main import app

        return TestClient(app)

    yield _make
    _reset()


def _payload():
    rng = np.random.default_rng(7)
    return {"frames": [{"landmarks": frame.tolist()} for frame in rng.normal(size=(10, 543, 3))]}


@pytest.mark.skipif(not os.path.isdir(BUNDLE_DIR), reason="models/word_v3/ bundle not present")
def test_word_route_returns_503_when_flag_off(make_client):
    client = make_client(word_mode_enabled="false")
    response = client.post("/api/v1/translation/word", json=_payload())
    assert response.status_code == 503
    assert response.json()["code"] == "WORD_MODE_DISABLED"


@pytest.mark.skipif(not os.path.isdir(BUNDLE_DIR), reason="models/word_v3/ bundle not present")
def test_word_route_ok_when_flag_on(make_client):
    client = make_client(word_mode_enabled="true", word_model_dir=BUNDLE_DIR)
    response = client.post("/api/v1/translation/word", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert set(body) >= {"word", "confidence", "accepted", "top3", "threshold", "model_version", "status"}
    assert body["model_version"] == "word_v3_8w_s123"
    assert body["threshold"] == 0.7


@pytest.mark.skipif(not os.path.isdir(BUNDLE_DIR), reason="models/word_v3/ bundle not present")
def test_compose_route(make_client):
    client = make_client(word_mode_enabled="true", word_model_dir=BUNDLE_DIR)
    response = client.post("/api/v1/translation/compose", json={"words": ["hello", "water"]})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"sentence", "method"}
    assert body["method"] in {"llm", "fallback"}
    assert isinstance(body["sentence"], str)


@pytest.mark.skipif(not os.path.isdir(BUNDLE_DIR), reason="models/word_v3/ bundle not present")
def test_compose_rejects_unknown_word(make_client):
    client = make_client(word_mode_enabled="true", word_model_dir=BUNDLE_DIR)
    response = client.post("/api/v1/translation/compose", json={"words": ["banana"]})
    assert response.status_code == 422


@pytest.mark.skipif(not os.path.isdir(BUNDLE_DIR), reason="models/word_v3/ bundle not present")
def test_compose_saves_history_for_signed_in_user(make_client):
    client = make_client(word_mode_enabled="true", word_model_dir=BUNDLE_DIR)
    from app.infrastructure.database.repositories.translation_repository import (
        get_memory_repository,
    )
    repo = get_memory_repository()

    response = client.post(
        "/api/v1/translation/compose",
        json={"words": ["hello", "help", "me"], "confidences": [0.9, 0.8, 0.7], "save_history": True},
        headers={"Authorization": "Bearer good-user1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["sentence"] == "Hello, help, me."
    assert body["method"] == "fallback"

    sessions = repo._sessions.get("user1", [])
    assert len(sessions) == 1
    assert sessions[0]["direction"] == "sign_to_text"
    assert sessions[0]["status"] == "completed"
    assert sessions[0]["title"].startswith("Word recognition")

    messages = repo._messages.get(sessions[0]["id"], [])
    assert len(messages) == 1
    msg = messages[0]
    assert msg["sequence"] == 1
    assert msg["direction"] == "sign_to_text"
    assert msg["input_text"] == "hello help me"
    assert msg["gloss"] == "hello help me"
    assert msg["output_text"] == "Hello, help, me."
    assert msg["confidence"] == pytest.approx(0.8)  # mean of 0.9, 0.8, 0.7
    assert msg["model_version"] == "word_v3_8w_s123"
    assert msg["result_type"] == "model"


@pytest.mark.skipif(not os.path.isdir(BUNDLE_DIR), reason="models/word_v3/ bundle not present")
def test_compose_guest_makes_zero_history_calls(make_client):
    client = make_client(word_mode_enabled="true", word_model_dir=BUNDLE_DIR)
    from app.infrastructure.database.repositories.translation_repository import (
        get_memory_repository,
    )
    repo = get_memory_repository()

    response = client.post(
        "/api/v1/translation/compose",
        json={"words": ["hello", "help", "me"], "confidences": [0.9, 0.8, 0.7]},
    )
    assert response.status_code == 200
    assert len(repo._sessions) == 0
    assert len(repo._messages) == 0


@pytest.mark.skipif(not os.path.isdir(BUNDLE_DIR), reason="models/word_v3/ bundle not present")
def test_compose_save_history_false_makes_zero_history_calls(make_client):
    client = make_client(word_mode_enabled="true", word_model_dir=BUNDLE_DIR)
    from app.infrastructure.database.repositories.translation_repository import (
        get_memory_repository,
    )
    repo = get_memory_repository()

    response = client.post(
        "/api/v1/translation/compose",
        json={"words": ["hello", "help", "me"], "confidences": [0.9, 0.8, 0.7], "save_history": False},
        headers={"Authorization": "Bearer good-user1"},
    )
    assert response.status_code == 200
    assert len(repo._sessions) == 0
    assert len(repo._messages) == 0


@pytest.mark.skipif(not os.path.isdir(BUNDLE_DIR), reason="models/word_v3/ bundle not present")
def test_compose_confidence_clamped(make_client):
    client = make_client(word_mode_enabled="true", word_model_dir=BUNDLE_DIR)
    from app.infrastructure.database.repositories.translation_repository import (
        get_memory_repository,
    )
    repo = get_memory_repository()

    # confidences > 1 and < 0 should be clamped to 0..1
    response = client.post(
        "/api/v1/translation/compose",
        json={"words": ["water"], "confidences": [1.5, -0.3], "save_history": True},
        headers={"Authorization": "Bearer good-user2"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["sentence"] == "Water."

    sessions = repo._sessions.get("user2", [])
    assert len(sessions) == 1
    messages = repo._messages.get(sessions[0]["id"], [])
    assert len(messages) == 1
    # mean of 1.5 and -0.3 = 0.6, clamped to 0.6
    assert messages[0]["confidence"] == pytest.approx(0.6)


@pytest.mark.skipif(not os.path.isdir(BUNDLE_DIR), reason="models/word_v3/ bundle not present")
def test_compose_no_confidences_means_null_confidence(make_client):
    client = make_client(word_mode_enabled="true", word_model_dir=BUNDLE_DIR)
    from app.infrastructure.database.repositories.translation_repository import (
        get_memory_repository,
    )
    repo = get_memory_repository()

    response = client.post(
        "/api/v1/translation/compose",
        json={"words": ["water"], "save_history": True},
        headers={"Authorization": "Bearer good-user3"},
    )
    assert response.status_code == 200

    sessions = repo._sessions.get("user3", [])
    assert len(sessions) == 1
    messages = repo._messages.get(sessions[0]["id"], [])
    assert len(messages) == 1
    assert messages[0]["confidence"] is None
