"""Integration tests for rate limiting on the word routes."""
from __future__ import annotations

import os

import numpy as np
import pytest

from app.core.ratelimit import _LIMITERS

BUNDLE_DIR = os.environ.get(
    "SIGNSAGE_WORD_MODEL_DIR", "/mnt/Drive-D/Codes/signsage/models/word_v3"
)


@pytest.fixture
def make_client(monkeypatch):
    pytest.importorskip("torch")
    from fastapi.testclient import TestClient

    from app.core.config import get_settings
    from app.ml.models.registry import reset_registry

    def _reset():
        get_settings.cache_clear()
        reset_registry()
        _LIMITERS.clear()

    def _make(**env):
        monkeypatch.setenv("SIGNSAGE_SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SIGNSAGE_SUPABASE_PUBLISHABLE_KEY", "pub")
        monkeypatch.setenv("SIGNSAGE_HISTORY_BACKEND", "memory")
        monkeypatch.setenv("SIGNSAGE_MODEL_PROMOTED", "false")
        monkeypatch.setenv("SIGNSAGE_RATE_LIMIT_ENABLED", "true")
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
def test_word_route_rate_limits_then_blocks(make_client):
    client = make_client(
        word_mode_enabled="true", word_model_dir=BUNDLE_DIR, rate_limit_word="2/min",
    )
    for _ in range(2):
        response = client.post("/api/v1/translation/word", json=_payload())
        assert response.status_code == 200, response.text
    response = client.post("/api/v1/translation/word", json=_payload())
    assert response.status_code == 429
    body = response.json()
    assert body["code"] == "RATE_LIMITED"
    assert "Retry-After" in response.headers
    assert int(response.headers["Retry-After"]) > 0


@pytest.mark.skipif(not os.path.isdir(BUNDLE_DIR), reason="models/word_v3/ bundle not present")
def test_compose_route_rate_limits(make_client):
    client = make_client(
        word_mode_enabled="true", word_model_dir=BUNDLE_DIR, rate_limit_compose="1/min",
    )
    assert client.post("/api/v1/translation/compose", json={"words": ["hello"]}).status_code == 200
    response = client.post("/api/v1/translation/compose", json={"words": ["hello"]})
    assert response.status_code == 429
    assert response.json()["code"] == "RATE_LIMITED"


@pytest.mark.skipif(not os.path.isdir(BUNDLE_DIR), reason="models/word_v3/ bundle not present")
def test_health_and_capabilities_are_exempt(make_client):
    client = make_client(word_mode_enabled="true", word_model_dir=BUNDLE_DIR)
    for _ in range(20):
        assert client.get("/api/v1/health").status_code == 200
        assert client.get("/api/v1/capabilities").status_code == 200
