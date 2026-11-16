"""Integration tests for the word-level recognition REST routes."""
from __future__ import annotations

import os
from pathlib import Path

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
        # Hermetic: never hit the real Groq API from integration tests.
        monkeypatch.setenv("SIGNSAGE_GROQ_API_KEY", "")
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
    assert set(body) == {"sentence", "method", "history_saved", "history_error"}
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

    sessions = [s for s in repo._sessions.values() if s["user_id"] == "user1"]
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

    sessions = [s for s in repo._sessions.values() if s["user_id"] == "user2"]
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

    sessions = [s for s in repo._sessions.values() if s["user_id"] == "user3"]
    assert len(sessions) == 1
    messages = repo._messages.get(sessions[0]["id"], [])
    assert len(messages) == 1
    assert messages[0]["confidence"] is None


def _capture_files(captures_dir: Path) -> list[Path]:
    if not captures_dir.is_dir():
        return []
    return sorted(captures_dir.rglob("*.npz"))


@pytest.mark.skipif(not os.path.isdir(BUNDLE_DIR), reason="models/word_v3/ bundle not present")
def test_capture_flag_off_ignores_true_label(make_client, tmp_path, monkeypatch):
    captures_dir = tmp_path / "captures_labeled"
    monkeypatch.setattr("app.api.routes.word_translation.CAPTURES_LABELED_DIR", captures_dir)
    client = make_client(
        word_mode_enabled="true",
        word_model_dir=BUNDLE_DIR,
        capture_mode_enabled="false",
    )
    before = _capture_files(captures_dir)
    response = client.post(
        "/api/v1/translation/word", json={**_payload(), "true_label": "water"}
    )
    assert response.status_code == 200
    assert _capture_files(captures_dir) == before  # nothing saved


@pytest.mark.skipif(not os.path.isdir(BUNDLE_DIR), reason="models/word_v3/ bundle not present")
def test_capture_without_true_label_unchanged(make_client, tmp_path, monkeypatch):
    captures_dir = tmp_path / "captures_labeled"
    monkeypatch.setattr("app.api.routes.word_translation.CAPTURES_LABELED_DIR", captures_dir)
    monkeypatch.setattr(
        "app.api.routes.word_translation._is_loopback_address", lambda host: True
    )
    client = make_client(
        word_mode_enabled="true",
        word_model_dir=BUNDLE_DIR,
        capture_mode_enabled="true",
    )
    response = client.post("/api/v1/translation/word", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert set(body) >= {"word", "confidence", "accepted", "top3", "threshold", "model_version", "status"}
    assert _capture_files(captures_dir) == []  # nothing saved without true_label


@pytest.mark.skipif(not os.path.isdir(BUNDLE_DIR), reason="models/word_v3/ bundle not present")
def test_capture_flag_on_loopback_saves(make_client, tmp_path, monkeypatch):
    captures_dir = tmp_path / "captures_labeled"
    monkeypatch.setattr("app.api.routes.word_translation.CAPTURES_LABELED_DIR", captures_dir)
    monkeypatch.setattr(
        "app.api.routes.word_translation._is_loopback_address", lambda host: True
    )
    client = make_client(
        word_mode_enabled="true",
        word_model_dir=BUNDLE_DIR,
        capture_mode_enabled="true",
    )
    response = client.post(
        "/api/v1/translation/word", json={**_payload(), "true_label": "water"}
    )
    assert response.status_code == 200
    files = _capture_files(captures_dir)
    assert len(files) == 1
    with np.load(files[0]) as z:
        assert str(z["true_label"].tolist()[0]) == "water"
        assert z["landmarks"].ndim == 3 and z["landmarks"].shape[1:] == (543, 3)
        assert list(z["top3_labels"].tolist()) and list(z["top3_probs"].tolist())
        assert str(z["model_version"].tolist()[0]) == "word_v3_8w_s123"
        assert isinstance(str(z["timestamp"].tolist()[0]), str)


@pytest.mark.skipif(not os.path.isdir(BUNDLE_DIR), reason="models/word_v3/ bundle not present")
def test_capture_accepts_rest_label(make_client, tmp_path, monkeypatch):
    captures_dir = tmp_path / "captures_labeled"
    monkeypatch.setattr("app.api.routes.word_translation.CAPTURES_LABELED_DIR", captures_dir)
    monkeypatch.setattr(
        "app.api.routes.word_translation._is_loopback_address", lambda host: True
    )
    client = make_client(
        word_mode_enabled="true",
        word_model_dir=BUNDLE_DIR,
        capture_mode_enabled="true",
    )
    response = client.post(
        "/api/v1/translation/word", json={**_payload(), "true_label": "rest"}
    )
    assert response.status_code == 200
    files = _capture_files(captures_dir)
    assert len(files) == 1
    with np.load(files[0]) as z:
        assert str(z["true_label"].tolist()[0]) == "rest"


@pytest.mark.skipif(not os.path.isdir(BUNDLE_DIR), reason="models/word_v3/ bundle not present")
def test_capture_non_loopback_saves_nothing(make_client, tmp_path, monkeypatch):
    captures_dir = tmp_path / "captures_labeled"
    monkeypatch.setattr("app.api.routes.word_translation.CAPTURES_LABELED_DIR", captures_dir)
    monkeypatch.setattr(
        "app.api.routes.word_translation._is_loopback_address", lambda host: False
    )
    client = make_client(
        word_mode_enabled="true",
        word_model_dir=BUNDLE_DIR,
        capture_mode_enabled="true",
    )
    response = client.post(
        "/api/v1/translation/word", json={**_payload(), "true_label": "water"}
    )
    assert response.status_code == 200
    assert _capture_files(captures_dir) == []  # nothing saved from non-loopback


@pytest.mark.skipif(not os.path.isdir(BUNDLE_DIR), reason="models/word_v3/ bundle not present")
def test_capture_invalid_label_returns_422(make_client, tmp_path, monkeypatch):
    captures_dir = tmp_path / "captures_labeled"
    monkeypatch.setattr("app.api.routes.word_translation.CAPTURES_LABELED_DIR", captures_dir)
    monkeypatch.setattr(
        "app.api.routes.word_translation._is_loopback_address", lambda host: True
    )
    client = make_client(
        word_mode_enabled="true",
        word_model_dir=BUNDLE_DIR,
        capture_mode_enabled="true",
    )
    response = client.post(
        "/api/v1/translation/word", json={**_payload(), "true_label": "not_a_word"}
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INVALID_MESSAGE"
    assert _capture_files(captures_dir) == []  # nothing saved for an invalid label


@pytest.mark.skipif(not os.path.isdir(BUNDLE_DIR), reason="models/word_v3/ bundle not present")
def test_compose_history_saved_true_for_signed_in_save(make_client):
    client = make_client(word_mode_enabled="true", word_model_dir=BUNDLE_DIR)
    response = client.post(
        "/api/v1/translation/compose",
        json={"words": ["hello"], "confidences": [0.9], "save_history": True},
        headers={"Authorization": "Bearer good-user9"},
    )
    assert response.status_code == 200
    assert response.json()["history_saved"] is True


@pytest.mark.skipif(not os.path.isdir(BUNDLE_DIR), reason="models/word_v3/ bundle not present")
def test_compose_history_saved_false_for_guest(make_client):
    client = make_client(word_mode_enabled="true", word_model_dir=BUNDLE_DIR)
    response = client.post(
        "/api/v1/translation/compose",
        json={"words": ["hello"], "confidences": [0.9], "save_history": True},
    )
    assert response.status_code == 200
    assert response.json()["history_saved"] is False


@pytest.mark.skipif(not os.path.isdir(BUNDLE_DIR), reason="models/word_v3/ bundle not present")
def test_compose_history_saved_false_when_save_history_off(make_client):
    client = make_client(word_mode_enabled="true", word_model_dir=BUNDLE_DIR)
    response = client.post(
        "/api/v1/translation/compose",
        json={"words": ["hello"], "confidences": [0.9], "save_history": False},
        headers={"Authorization": "Bearer good-user9"},
    )
    assert response.status_code == 200
    assert response.json()["history_saved"] is False


@pytest.mark.skipif(not os.path.isdir(BUNDLE_DIR), reason="models/word_v3/ bundle not present")
def test_compose_collapses_consecutive_duplicate_words(make_client):
    client = make_client(word_mode_enabled="true", word_model_dir=BUNDLE_DIR)
    response = client.post("/api/v1/translation/compose", json={"words": ["water", "water"]})
    assert response.status_code == 200
    assert response.json()["sentence"] == "Water."
    response = client.post("/api/v1/translation/compose", json={"words": ["hello", "hello", "help"]})
    assert response.status_code == 200
    assert response.json()["sentence"] == "Hello, help."
