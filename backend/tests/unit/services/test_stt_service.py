"""STT service: whisper model selection from the key's /models list, codes,
and the never-log-audio-or-text rule. Groq is always mocked."""
from __future__ import annotations

import logging

import httpx
import pytest

from app.core.config import Settings
from app.core.errors import SttUnavailableError
from app.services import stt_service

AUDIO = b"\x1a\x45\xdf\xa3" + b"\x00" * 64  # fake webm
TRANSCRIPT = "hello there"


def _settings(**overrides):
    base = dict(
        groq_api_key="test-key",
        groq_tts_api_key=None,
        groq_base_url="https://api.groq.com/openai/v1",
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


@pytest.fixture(autouse=True)
def _reset_cache():
    stt_service._reset_model_cache()
    yield
    stt_service._reset_model_cache()


def _transport(models, *, whisper_available_models=None, transcribe_status=200, transcribe_body=None):
    models = models if models is not None else ["whisper-large-v3-turbo", "whisper-large-v3", "llama-3.3-70b-versatile"]

    def handler(req: httpx.Request) -> httpx.Response:
        if str(req.url).endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": m} for m in models]})
        if transcribe_status != 200:
            return httpx.Response(transcribe_status, json={"error": {"message": "nope"}})
        body = transcribe_body if transcribe_body is not None else {"text": TRANSCRIPT}
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


async def test_happy_path_returns_text():
    transport = _transport(["whisper-large-v3-turbo"])
    text = await stt_service.transcribe_audio(AUDIO, "clip.webm", "audio/webm", _settings(), transport=transport)
    assert text == TRANSCRIPT


async def test_prefers_whisper_large_v3_turbo_over_other_whispers():
    requests = []

    def handler(req: httpx.Request) -> httpx.Response:
        if str(req.url).endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "whisper-large-v3"}, {"id": "whisper-large-v3-turbo"}]})
        requests.append(req.content)
        return httpx.Response(200, json={"text": TRANSCRIPT})

    await stt_service.transcribe_audio(AUDIO, "clip.webm", "audio/webm", _settings(),
                                       transport=httpx.MockTransport(handler))
    assert b"whisper-large-v3-turbo" in requests[0]
    assert b'name="language"' in requests[0] and b"en" in requests[0]


async def test_falls_back_to_whisper_large_v3_when_turbo_absent():
    requests = []

    def handler(req: httpx.Request) -> httpx.Response:
        if str(req.url).endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "whisper-large-v3"}]})
        requests.append(req.content)
        return httpx.Response(200, json={"text": TRANSCRIPT})

    await stt_service.transcribe_audio(AUDIO, "clip.webm", "audio/webm", _settings(),
                                       transport=httpx.MockTransport(handler))
    assert b"whisper-large-v3" in requests[0]


async def test_no_whisper_in_models_list_is_unconfigured():
    transport = _transport(["llama-3.3-70b-versatile"])
    with pytest.raises(SttUnavailableError) as excinfo:
        await stt_service.transcribe_audio(AUDIO, "clip.webm", "audio/webm", _settings(), transport=transport)
    assert excinfo.value.code == "STT_UNCONFIGURED"


async def test_rate_limited_transcription_maps_code():
    transport = _transport(["whisper-large-v3-turbo"], transcribe_status=429)
    with pytest.raises(SttUnavailableError) as excinfo:
        await stt_service.transcribe_audio(AUDIO, "clip.webm", "audio/webm", _settings(), transport=transport)
    assert excinfo.value.code == "STT_RATE_LIMITED"


async def test_bad_audio_maps_code():
    transport = _transport(["whisper-large-v3-turbo"], transcribe_status=400)
    with pytest.raises(SttUnavailableError) as excinfo:
        await stt_service.transcribe_audio(AUDIO, "clip.webm", "audio/webm", _settings(), transport=transport)
    assert excinfo.value.code == "STT_BAD_AUDIO"


async def test_rejected_key_maps_code():
    transport = _transport(["whisper-large-v3-turbo"], transcribe_status=401)
    with pytest.raises(SttUnavailableError) as excinfo:
        await stt_service.transcribe_audio(AUDIO, "clip.webm", "audio/webm", _settings(), transport=transport)
    assert excinfo.value.code == "STT_UNCONFIGURED"


async def test_missing_key_maps_code():
    with pytest.raises(SttUnavailableError) as excinfo:
        await stt_service.transcribe_audio(
            AUDIO, "clip.webm", "audio/webm", _settings(groq_api_key=None, groq_tts_api_key=None)
        )
    assert excinfo.value.code == "STT_UNCONFIGURED"


async def test_audio_bytes_and_text_never_logged(caplog):
    transport = _transport(["whisper-large-v3-turbo"])
    with caplog.at_level(logging.INFO):
        text = await stt_service.transcribe_audio(AUDIO, "clip.webm", "audio/webm", _settings(), transport=transport)
    assert text == TRANSCRIPT
    assert TRANSCRIPT not in caplog.text
    assert "1a45dfa3" not in caplog.text
    assert AUDIO.hex() not in caplog.text
    assert "test-key" not in caplog.text
