"""Speech-to-text via Groq Whisper. Audio is never stored, audio/text never logged.

Used by the browser-microphone fallback in the Sign Generator: browsers
without (or blocked from) the Web Speech API record a short clip client-side
and POST it here; we forward it to Groq's /audio/transcriptions and return
{text}. The recording is processed in memory only — nothing touches disk and
nothing is retained.

Codes on failure (all HTTP 503): STT_UNCONFIGURED, STT_RATE_LIMITED, STT_BAD_AUDIO.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

from app.core.config import Settings
from app.core.errors import SttUnavailableError

logger = logging.getLogger(__name__)

# The Whisper model to use is read from the key's own /models list (prefer
# turbo) — never hardcoded past what the account actually offers.
_PREFERRED_WHISPER_MODELS = ("whisper-large-v3-turbo", "whisper-large-v3")
_MODEL_CACHE_TTL_SECONDS = 600.0
# base_url -> (timestamp, chosen model or None)
_model_cache: dict[str, tuple[float, Optional[str]]] = {}


def _api_key(settings: Settings) -> Optional[str]:
    # Same chain as TTS: the dedicated key when set, else the shared Groq key.
    return settings.groq_tts_api_key or settings.groq_api_key


def _reset_model_cache() -> None:
    """Test hook: clear the cached /models lookup."""
    _model_cache.clear()


async def _whisper_model(settings: Settings, transport: Optional[httpx.AsyncBaseTransport]) -> str:
    base = settings.groq_base_url.rstrip("/")
    now = time.monotonic()
    cached = _model_cache.get(base)
    if cached and now - cached[0] < _MODEL_CACHE_TTL_SECONDS:
        if cached[1]:
            return cached[1]
        raise SttUnavailableError(
            "No Whisper model is available for this Groq key.", code="STT_UNCONFIGURED"
        )

    api_key = _api_key(settings)
    if not api_key:
        raise SttUnavailableError(
            "Server transcription is not configured (SIGNSAGE_GROQ_TTS_API_KEY or SIGNSAGE_GROQ_API_KEY).",
            code="STT_UNCONFIGURED",
        )
    try:
        async with httpx.AsyncClient(timeout=10.0, transport=transport) as client:
            resp = await client.get(
                base + "/models", headers={"Authorization": f"Bearer {api_key}"}
            )
    except httpx.HTTPError:
        logger.warning("stt: /models request failed (transport error)")
        raise SttUnavailableError("Groq is unreachable.", code="STT_UNCONFIGURED") from None

    if resp.status_code == 429:
        logger.warning("stt: /models rate limited (429)")
        raise SttUnavailableError("Groq rate limit reached, try again shortly.", code="STT_RATE_LIMITED")
    if resp.status_code >= 400:
        logger.warning("stt: /models returned HTTP %d", resp.status_code)
        raise SttUnavailableError(
            f"Groq rejected the configured API key (HTTP {resp.status_code}).",
            code="STT_UNCONFIGURED",
        )
    try:
        ids = [m.get("id", "") for m in resp.json().get("data", [])]
    except ValueError:
        ids = []
    pick = next((m for m in _PREFERRED_WHISPER_MODELS if m in ids), None)
    if pick is None:
        pick = next((i for i in ids if "whisper" in i.lower()), None)
    _model_cache[base] = (now, pick)
    if not pick:
        logger.warning("stt: no whisper model in the key's /models list")
        raise SttUnavailableError("No Whisper model is available for this Groq key.", code="STT_UNCONFIGURED")
    return pick


async def transcribe_audio(
    data: bytes,
    filename: str,
    content_type: str,
    settings: Settings,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> str:
    """Return the transcript for `data`. Raises SttUnavailableError on any failure."""
    api_key = _api_key(settings)
    if not api_key:
        raise SttUnavailableError(
            "Server transcription is not configured (SIGNSAGE_GROQ_TTS_API_KEY or SIGNSAGE_GROQ_API_KEY).",
            code="STT_UNCONFIGURED",
        )
    model = await _whisper_model(settings, transport)
    base = settings.groq_base_url.rstrip("/")
    files = {"file": (filename, data, content_type)}
    form = {"model": model, "language": "en", "response_format": "json"}
    try:
        async with httpx.AsyncClient(timeout=30.0, transport=transport) as client:
            resp = await client.post(
                base + "/audio/transcriptions",
                files=files,
                data=form,
                headers={"Authorization": f"Bearer {api_key}"},
            )
    except httpx.HTTPError:
        logger.warning("stt: transcription request failed (transport error), model=%s", model)
        raise SttUnavailableError("Groq is unreachable.", code="STT_UNCONFIGURED") from None

    if resp.status_code == 429:
        logger.warning("stt: transcription rate limited (429), model=%s", model)
        raise SttUnavailableError("Groq rate limit reached, try again shortly.", code="STT_RATE_LIMITED")
    if resp.status_code in (400, 413, 422):
        logger.warning("stt: transcription rejected the audio (HTTP %d), model=%s", resp.status_code, model)
        raise SttUnavailableError("Groq could not transcribe this audio.", code="STT_BAD_AUDIO")
    if resp.status_code in (401, 403):
        logger.warning("stt: transcription rejected the key (HTTP %d), model=%s", resp.status_code, model)
        raise SttUnavailableError(
            f"Groq rejected the configured API key (HTTP {resp.status_code}).",
            code="STT_UNCONFIGURED",
        )
    if resp.status_code >= 400:
        logger.warning("stt: transcription returned HTTP %d, model=%s", resp.status_code, model)
        raise SttUnavailableError("Server transcription failed.", code="STT_UNCONFIGURED")
    try:
        text = str(resp.json().get("text", "")).strip()
    except ValueError:
        text = ""
    if not text:
        logger.warning("stt: empty transcript, model=%s, audio_bytes=%d", model, len(data))
        raise SttUnavailableError("The audio could not be transcribed.", code="STT_BAD_AUDIO")
    logger.info("stt: transcribed %d bytes with model=%s", len(data), model)  # never the text
    return text
