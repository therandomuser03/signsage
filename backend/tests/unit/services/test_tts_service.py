"""TTS service: every failure mode maps to TtsUnavailableError (503-worthy)."""
import httpx
import pytest

from app.core.config import Settings
from app.core.errors import TtsUnavailableError
from app.services import tts_service

TEXT = "Hello, can you help me?"


def _settings(**overrides):
    base = dict(
        groq_api_key="test-key",
        tts_provider="auto",
        groq_tts_model="canopylabs/orpheus-v1-english",
        groq_tts_voice="hannah",
        tts_max_input_chars=200,
    )
    base.update(overrides)
    return Settings(**base)


def _wav_bytes(n=64):
    return b"RIFF" + b"\x00" * (n - 4)


def _client(handler):
    return httpx.MockTransport(handler)


async def test_synthesize_returns_wav_on_success():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        import json

        calls.append((req.url, req.headers, json.loads(req.content)))
        return httpx.Response(200, content=_wav_bytes(), headers={"content-type": "audio/wav"})

    audio = await tts_service.synthesize_speech(TEXT, _settings(), transport=_client(handler))
    assert audio.startswith(b"RIFF")
    url, headers, body = calls[0]
    assert str(url).endswith("/openai/v1/audio/speech")
    assert headers["authorization"] == "Bearer test-key"
    assert body == {
        "model": "canopylabs/orpheus-v1-english",
        "input": TEXT,
        "voice": "hannah",
        "response_format": "wav",
    }


async def test_unsupported_voice_is_rejected():
    with pytest.raises(TtsUnavailableError):
        await tts_service.synthesize_speech(TEXT, _settings(), voice="not-a-voice")


async def test_text_over_the_provider_limit_is_rejected():
    with pytest.raises(TtsUnavailableError):
        await tts_service.synthesize_speech("x" * 201, _settings())


async def test_empty_text_is_rejected():
    with pytest.raises(TtsUnavailableError):
        await tts_service.synthesize_speech("   ", _settings())


async def test_missing_api_key_raises_unavailable():
    with pytest.raises(TtsUnavailableError):
        await tts_service.synthesize_speech(TEXT, _settings(groq_api_key=None))


async def test_browser_provider_selected_raises_unavailable():
    with pytest.raises(TtsUnavailableError):
        await tts_service.synthesize_speech(TEXT, _settings(tts_provider="browser"))


@pytest.mark.parametrize("status", [429, 401, 403, 500, 502])
async def test_provider_error_statuses_raise_unavailable(status):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "nope"})

    with pytest.raises(TtsUnavailableError):
        await tts_service.synthesize_speech(TEXT, _settings(), transport=_client(handler))


async def test_empty_audio_body_raises_unavailable():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"")

    with pytest.raises(TtsUnavailableError):
        await tts_service.synthesize_speech(TEXT, _settings(), transport=_client(handler))


async def test_model_terms_required_is_surfaced_explicitly():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "requires terms acceptance", "code": "model_terms_required"}},
        )

    with pytest.raises(TtsUnavailableError, match="model_terms_required"):
        await tts_service.synthesize_speech(TEXT, _settings(), transport=_client(handler))


async def test_transport_error_raises_unavailable():
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    with pytest.raises(TtsUnavailableError):
        await tts_service.synthesize_speech(TEXT, _settings(), transport=_client(handler))


def test_tts_provider_available():
    assert tts_service.tts_provider_available(_settings()) is True
    assert tts_service.tts_provider_available(_settings(groq_api_key=None)) is False
    assert tts_service.tts_provider_available(_settings(tts_provider="browser")) is False


async def test_dedicated_tts_key_wins_over_shared_key():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["authorization"] == "Bearer tts-key"
        return httpx.Response(200, content=_wav_bytes(), headers={"content-type": "audio/wav"})

    audio = await tts_service.synthesize_speech(
        TEXT, _settings(groq_api_key="shared-key", groq_tts_api_key="tts-key"), transport=_client(handler)
    )
    assert audio.startswith(b"RIFF")


async def test_shared_key_used_when_no_dedicated_tts_key():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["authorization"] == "Bearer shared-key"
        return httpx.Response(200, content=_wav_bytes(), headers={"content-type": "audio/wav"})

    audio = await tts_service.synthesize_speech(
        TEXT, _settings(groq_api_key="shared-key"), transport=_client(handler)
    )
    assert audio.startswith(b"RIFF")


def test_tts_provider_available_with_dedicated_key_only():
    assert tts_service.tts_provider_available(_settings(groq_api_key=None, groq_tts_api_key="tts-key")) is True
    assert tts_service.tts_provider_available(_settings(groq_api_key=None, groq_tts_api_key=None)) is False
