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
    from app.ml.models.registry import reset_registry

    async def fake_verify(token, settings, transport=None):
        from app.core.security import AuthenticatedUser

        if token.startswith("good-"):
            return AuthenticatedUser(id=token[5:], email=f"{token[5:]}@example.com", access_token=token)
        from app.core.errors import AuthError

        raise AuthError("Invalid or expired access token.")

    def _reset():
        get_settings.cache_clear()
        reset_registry()

    def _make(**env):
        monkeypatch.setenv("SIGNSAGE_SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SIGNSAGE_SUPABASE_PUBLISHABLE_KEY", "pub")
        monkeypatch.setenv("SIGNSAGE_HISTORY_BACKEND", "memory")
        monkeypatch.setenv("SIGNSAGE_MODEL_PROMOTED", "false")
        monkeypatch.setenv("SIGNSAGE_ALLOW_UNPROMOTED_DEV", "true")
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
