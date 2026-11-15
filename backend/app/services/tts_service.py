"""Text-to-speech via Groq Orpheus (optional provider).

Design rules:
  * The Groq API key never leaves the backend. Uses SIGNSAGE_GROQ_TTS_API_KEY
    when set (dedicated voice-model key), else SIGNSAGE_GROQ_API_KEY.
  * Only completed, user-visible interpretations are sent -- never
    intermediate predictions -- and the text is never logged.
  * Orpheus input is capped at 200 characters (provider limit); longer text
    is rejected with TtsUnavailableError so the caller can fall back.
  * Every failure mode raises TtsUnavailableError (→ HTTP 503, code
    TTS_UNAVAILABLE) so the frontend can fall back to browser voices.
    The app must never hard-fail because audio is unavailable.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

from app.core.config import Settings
from app.core.errors import TtsUnavailableError

logger = logging.getLogger(__name__)

# Voice personas supported by canopylabs/orpheus-v1-english (checked against
# the Groq docs, 2026: autumn, diana, hannah, austin, daniel, troy).
GROQ_ORPHEUS_VOICES = frozenset({"autumn", "diana", "hannah", "austin", "daniel", "troy"})


def _tts_api_key(settings: Settings) -> Optional[str]:
    """The dedicated TTS key when set, else the shared Groq key."""
    return settings.groq_tts_api_key or settings.groq_api_key


def tts_provider_available(settings: Settings) -> bool:
    """True when the backend is configured to attempt a Groq synthesis."""
    return settings.tts_provider in ("auto", "groq") and bool(_tts_api_key(settings))


async def synthesize_speech(
    text: str,
    settings: Settings,
    voice: Optional[str] = None,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> bytes:
    """Return WAV bytes for `text` from Groq Orpheus.

    Raises TtsUnavailableError on any failure; never raises anything else,
    so callers can treat it as "fall back" unconditionally.
    """
    if settings.tts_provider == "browser":
        raise TtsUnavailableError("Browser TTS selected; backend provider is not used.")
    api_key = _tts_api_key(settings)
    if not api_key:
        raise TtsUnavailableError("TTS provider is not configured (SIGNSAGE_GROQ_TTS_API_KEY or SIGNSAGE_GROQ_API_KEY).")

    text = (text or "").strip()
    if not text:
        raise TtsUnavailableError("Nothing to speak.")
    if len(text) > settings.tts_max_input_chars:
        raise TtsUnavailableError(
            f"Text exceeds the {settings.tts_max_input_chars}-character Orpheus limit."
        )

    voice = (voice or settings.groq_tts_voice).strip().lower()
    if voice not in GROQ_ORPHEUS_VOICES:
        raise TtsUnavailableError(f"Unsupported Orpheus voice {voice!r}.")

    body = {
        "model": settings.groq_tts_model,
        "input": text,
        "voice": voice,
        "response_format": "wav",
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = settings.groq_base_url.rstrip("/") + "/audio/speech"
    if transport is None:
        transport = None  # real network; tests pass a MockTransport explicitly
    try:
        async with httpx.AsyncClient(
            timeout=settings.groq_tts_timeout_seconds, transport=transport
        ) as client:
            resp = await client.post(url, json=body, headers=headers)
    except httpx.HTTPError as exc:
        logger.warning("tts synthesize failed: transport error (%s)", type(exc).__name__)
        raise TtsUnavailableError("TTS provider is unreachable.") from exc

    if resp.status_code == 429:
        logger.warning("tts synthesize failed: rate limited (429)")
        raise TtsUnavailableError("TTS provider rate limit reached.")
    if resp.status_code in (401, 403):
        logger.warning("tts synthesize failed: provider rejected the key (%d)", resp.status_code)
        raise TtsUnavailableError("TTS provider rejected the configured API key.")
    if resp.status_code == 400:
        # Groq returns model_terms_required until an org admin accepts the
        # model's terms in the console (one-time, per organization). Surface
        # the code — it is not secret — so operators can act on it.
        code = None
        try:
            code = resp.json().get("error", {}).get("code")
        except ValueError:
            pass
        if code == "model_terms_required":
            logger.warning("tts synthesize failed: model terms not accepted (model_terms_required)")
            raise TtsUnavailableError(
                "TTS model requires one-time terms acceptance in the Groq console (model_terms_required)."
            )
        logger.warning("tts synthesize failed: provider returned %d", resp.status_code)
        raise TtsUnavailableError(f"TTS provider returned HTTP {resp.status_code}.")
    if resp.status_code >= 400:
        logger.warning("tts synthesize failed: provider returned %d", resp.status_code)
        raise TtsUnavailableError(f"TTS provider returned HTTP {resp.status_code}.")
    if not resp.content:
        logger.warning("tts synthesize failed: empty audio body")
        raise TtsUnavailableError("TTS provider returned no audio.")
    return resp.content
