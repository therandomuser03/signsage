"""Piper TTS provider: order, codes, cap, fallback, WAV sanity.

The real piper-tts package is not importable in every environment (it needs
onnxruntime, which lags new Python releases), so most tests stub the voice.
The WAV-header test uses the real voice when it is present on disk.
"""
from __future__ import annotations

import io
import wave

import pytest

from app.core.config import Settings
from app.core.errors import TtsUnavailableError
from app.services import tts_service

TEXT = "Can you help me?"


def _settings(**overrides):
    base = dict(
        groq_api_key="test-key",
        groq_tts_api_key=None,
        tts_provider="auto",
        groq_tts_voice="hannah",
        tts_max_input_chars=200,
        piper_voice="en_US-kristin-medium",
        piper_dir="/srv/voices",
        piper_cache="/tmp/piper",
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


class _Chunk:
    def __init__(self, payload: bytes):
        self.audio_int16_bytes = payload


class _FakeVoice:
    def synthesize(self, text):
        return iter([_Chunk(b"\x01\x00" * 256), _Chunk(b"\x02\x00" * 256)])


class _BoomVoice:
    def synthesize(self, text):
        raise RuntimeError("onnx exploded")


def _load_ok(monkeypatch, voice_obj=None):
    monkeypatch.setattr(tts_service, "_load_piper_voice", lambda s: (voice_obj or _FakeVoice(), None))


def _load_unavailable(monkeypatch, reason="piper import failed (ImportError)"):
    monkeypatch.setattr(tts_service, "_load_piper_voice", lambda s: (None, reason))


# ---------- provider order -------------------------------------------------

async def test_auto_uses_piper_when_available(monkeypatch):
    _load_ok(monkeypatch)
    wav = await tts_service.synthesize_speech(TEXT, _settings())
    assert wav.startswith(b"RIFF")


async def test_auto_groq_is_never_called_even_when_key_set(monkeypatch):
    _load_ok(monkeypatch)
    transport_calls = []

    async def boom(*args, **kwargs):  # pragma: no cover - must not be hit
        transport_calls.append(args)
        raise AssertionError("groq path must not run in auto mode")

    monkeypatch.setattr(tts_service, "_synthesize_groq", boom)
    wav = await tts_service.synthesize_speech(TEXT, _settings(groq_api_key="k"))
    assert wav.startswith(b"RIFF")
    assert transport_calls == []


async def test_auto_falls_back_to_503_when_piper_unavailable(monkeypatch):
    _load_unavailable(monkeypatch)
    with pytest.raises(TtsUnavailableError) as excinfo:
        await tts_service.synthesize_speech(TEXT, _settings())
    assert excinfo.value.code == "TTS_UNCONFIGURED"


async def test_ui_pick_of_groq_forces_groq_even_if_piper_available(monkeypatch):
    _load_ok(monkeypatch)
    seen = {}

    async def fake_groq(text, settings, voice, transport):
        seen["text"] = text
        return b"RIFF" + b"\x00" * 60

    monkeypatch.setattr(tts_service, "_synthesize_groq", fake_groq)
    wav = await tts_service.synthesize_speech(TEXT, _settings(tts_provider="groq"))
    assert seen["text"] == TEXT
    assert wav.startswith(b"RIFF")


async def test_per_request_provider_overrides_env(monkeypatch):
    _load_ok(monkeypatch)
    wav = await tts_service.synthesize_speech(TEXT, _settings(tts_provider="groq"), provider="piper")
    assert wav.startswith(b"RIFF")


# ---------- Piper unavailable → clean fallback, never raise ---------------

async def test_piper_import_failure_reports_unavailable(monkeypatch):
    monkeypatch.setattr(tts_service, "_piper_import", lambda: (None, "piper import failed (ImportError)"))
    ok, reason = tts_service.piper_status(_settings())
    assert ok is False
    assert "piper import failed" in (reason or "")
    with pytest.raises(TtsUnavailableError) as excinfo:
        await tts_service.synthesize_speech(TEXT, _settings())
    assert excinfo.value.code == "TTS_UNCONFIGURED"


async def test_piper_synthesis_error_raises_unavailable_only(monkeypatch):
    _load_ok(monkeypatch, voice_obj=_BoomVoice())
    with pytest.raises(TtsUnavailableError) as excinfo:
        await tts_service.synthesize_speech(TEXT, _settings())
    assert excinfo.value.code == "TTS_UNAVAILABLE"


def test_provider_unavailable_when_voice_files_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(tts_service, "_piper_import", lambda: (object, None))
    ok, reason = tts_service.piper_status(_settings(piper_dir=str(tmp_path / "nope"), piper_cache=str(tmp_path / "cache")))
    assert ok is False
    assert "voice files" in (reason or "")


# ---------- text cap --------------------------------------------------------

async def test_text_over_300_chars_is_rejected_in_auto(monkeypatch):
    _load_ok(monkeypatch)
    with pytest.raises(TtsUnavailableError, match="300"):
        await tts_service.synthesize_speech("x" * 301, _settings())


async def test_text_at_300_chars_is_accepted(monkeypatch):
    _load_ok(monkeypatch)
    wav = await tts_service.synthesize_speech("x" * 300, _settings())
    assert wav.startswith(b"RIFF")


# ---------- 503 machine codes (Groq path) ----------------------------------

import httpx


def _mock_transport(status, json_body=None, content=b""):
    def handler(req: httpx.Request) -> httpx.Response:
        if json_body is not None:
            return httpx.Response(status, json=json_body)
        return httpx.Response(status, content=content)

    return httpx.MockTransport(handler)


async def test_terms_required_maps_to_machine_code(monkeypatch):
    transport = _mock_transport(400, {"error": {"code": "model_terms_required", "message": "accept terms"}})
    with pytest.raises(TtsUnavailableError) as excinfo:
        await tts_service.synthesize_speech(
            TEXT, _settings(tts_provider="groq"), transport=transport
        )
    assert excinfo.value.code == "TTS_MODEL_TERMS_REQUIRED"


@pytest.mark.parametrize("status", [401, 403])
async def test_bad_key_maps_to_machine_code(status):
    transport = _mock_transport(status, {"error": {"message": "denied"}})
    with pytest.raises(TtsUnavailableError) as excinfo:
        await tts_service.synthesize_speech(
            TEXT, _settings(tts_provider="groq"), transport=transport
        )
    assert excinfo.value.code == "TTS_BAD_KEY"


async def test_rate_limited_maps_to_machine_code():
    transport = _mock_transport(429, {"error": {"message": "slow down"}})
    with pytest.raises(TtsUnavailableError) as excinfo:
        await tts_service.synthesize_speech(
            TEXT, _settings(tts_provider="groq"), transport=transport
        )
    assert excinfo.value.code == "TTS_RATE_LIMITED"


async def test_unconfigured_groq_maps_to_machine_code():
    with pytest.raises(TtsUnavailableError) as excinfo:
        await tts_service.synthesize_speech(TEXT, _settings(groq_api_key=None, groq_tts_api_key=None, tts_provider="groq"))
    assert excinfo.value.code == "TTS_UNCONFIGURED"


# ---------- WAV header sanity with the real voice when present --------------

def _real_voice_available(settings) -> bool:
    Voice, err = tts_service._piper_import()
    return Voice is not None and tts_service._piper_paths(settings) is not None


@pytest.mark.skipif(not _real_voice_available(_settings()), reason="piper or the real voice is not present locally")
async def test_real_voice_produces_22050hz_mono_wav():
    voice_obj, reason = tts_service._load_piper_voice(_settings())
    assert voice_obj is not None, reason
    wav_bytes = tts_service._piper_wav_bytes(voice_obj, TEXT)
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wav:
        assert wav.getframerate() == 22050
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
    assert wav_bytes.startswith(b"RIFF")
