"""POST /api/v1/speech/speak — TTS route behaviour."""
from __future__ import annotations

import httpx
import pytest

from app.core.errors import TtsUnavailableError
from app.services import tts_service


def test_speak_returns_wav(make_client, monkeypatch):
    async def fake(text, settings, voice=None, transport=None, provider=None):
        assert text == "Hello, how are you?"
        return b"RIFF" + b"\x00" * 60

    monkeypatch.setattr(tts_service, "synthesize_speech", fake)
    c = make_client()
    r = c.post("/api/v1/speech/speak", json={"text": "Hello, how are you?"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    assert r.content.startswith(b"RIFF")


def test_speak_fallback_signal_when_provider_unavailable(make_client, monkeypatch):
    async def fake(text, settings, voice=None, transport=None, provider=None):
        raise TtsUnavailableError("TTS provider rejected the configured API key.")

    monkeypatch.setattr(tts_service, "synthesize_speech", fake)
    c = make_client()
    r = c.post("/api/v1/speech/speak", json={"text": "Hello."})
    assert r.status_code == 503
    assert r.json()["code"] == "TTS_UNAVAILABLE"


def test_speak_rejects_overlong_input(make_client):
    c = make_client(groq_api_key="test-key", tts_provider="groq")
    r = c.post("/api/v1/speech/speak", json={"text": "x" * 201})
    # Rejected before any network call: 503 TTS_UNAVAILABLE (fallback signal)
    assert r.status_code == 503
    assert r.json()["code"] == "TTS_UNAVAILABLE"


def test_speak_rejects_empty_text(make_client):
    c = make_client()
    assert c.post("/api/v1/speech/speak", json={"text": ""}).status_code == 422


def test_speak_reports_rate_limit_or_unavailable_without_api_key(make_client):
    # No server voice configured → 503 with a machine code, never a 500.
    # (auto resolves to Piper, which is absent in this environment.)
    c = make_client()
    r = c.post("/api/v1/speech/speak", json={"text": "Hello."})
    assert r.status_code == 503
    assert r.json()["code"] == "TTS_UNCONFIGURED"
