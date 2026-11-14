"""Integration tests for /api/v1/translation/sentence.

The real model is replaced with a scripted stub for determinism; the live
pipeline (WordSequencePipeline windowing + majority vote + composer) runs
for real.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

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
    from app.core.security import clear_auth_cache
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


def _frames(n=50):
    rng = np.random.default_rng(42)
    return [{"landmarks": frame.tolist()} for frame in rng.normal(size=(n, 543, 3))]


class _ScriptedWordAdapter:
    """Word adapter whose classify() replays a scripted label timeline."""

    labels = ["hello", "me", "you", "yes", "no", "help", "water", "where", "rest"]
    confidence_threshold = 0.7
    model_version = "stub_1"
    signer_note = "stub"

    def __init__(self, script):
        self._script = list(script)

    def classify(self, arr):
        label, conf = self._script.pop(0) if self._script else ("rest", 0.99)
        return SimpleNamespace(
            status="ok",
            label=label,
            confidence=conf,
            motion_energy=0.1,
            top3=[{"label": label, "probability": conf}],
            feature_ms=0.0,
            inference_ms=0.0,
            frames_with_hand=len(arr),
        )

    @classmethod
    def scripted(cls, pairs):
        return cls(pairs)


def _install_stub(monkeypatch, adapter):
    monkeypatch.setattr(
        "app.api.routes.word_translation._get_word_adapter",
        lambda settings: adapter,
    )


def test_sentence_503_when_flag_off(make_client):
    client = make_client(word_mode_enabled="false")
    response = client.post("/api/v1/translation/sentence", json={"frames": _frames()})
    assert response.status_code == 503
    assert response.json()["code"] == "WORD_MODE_DISABLED"


def test_sentence_422_invalid_frames(make_client):
    client = make_client(word_mode_enabled="true", word_model_dir=BUNDLE_DIR)
    response = client.post("/api/v1/translation/sentence", json={"frames": []})
    assert response.status_code == 422


def test_sentence_requires_vocabulary_entries(make_client, monkeypatch):
    """Stub: 3 x hello wins over rest -> gloss [hello] + sentence composed."""
    adapter = _ScriptedWordAdapter.scripted([("hello", 0.99)] * 8)
    _install_stub(monkeypatch, adapter)
    client = make_client(word_mode_enabled="true", word_model_dir=BUNDLE_DIR)
    response = client.post("/api/v1/translation/sentence", json={"frames": _frames()})
    assert response.status_code == 200
    body = response.json()
    assert body["gloss_sequence"] == [{"gloss": "hello", "confidence": 0.99}]
    assert body["sentence"] == "Hello."
    assert body["accepted"] is True


def test_sentence_two_signs(make_client, monkeypatch):
    script = [("hello", 0.99)] * 4 + [("water", 0.97)] * 3
    adapter = _ScriptedWordAdapter.scripted(script)
    _install_stub(monkeypatch, adapter)
    client = make_client(word_mode_enabled="true", word_model_dir=BUNDLE_DIR)
    response = client.post("/api/v1/translation/sentence", json={"frames": _frames()})
    assert response.status_code == 200
    body = response.json()
    assert [g["gloss"] for g in body["gloss_sequence"]] == ["hello", "water"]
    assert body["sentence"].startswith("Hello")
    assert body["accepted"] is True
    for g in body["gloss_sequence"]:
        assert g["confidence"] > 0.7


def test_sentence_rest_only_returns_not_accepted(make_client, monkeypatch):
    adapter = _ScriptedWordAdapter.scripted([("rest", 0.98)] * 12)
    _install_stub(monkeypatch, adapter)
    client = make_client(word_mode_enabled="true", word_model_dir=BUNDLE_DIR)
    response = client.post("/api/v1/translation/sentence", json={"frames": _frames()})
    assert response.status_code == 200
    body = response.json()
    assert body["gloss_sequence"] == []
    assert body["sentence"] == ""
    assert body["accepted"] is False
