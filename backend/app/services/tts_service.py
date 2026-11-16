"""Text-to-speech: free local Piper (default) + optional Groq Orpheus.

Providers, resolved from ``tts_provider`` (env SIGNSAGE_TTS_PROVIDER, or the
per-request ``provider`` the frontend sends when the user picks one):

  * ``"auto"``   — Piper when it is importable and its voice files are
                   present/downloadable; otherwise HTTP 503 so the frontend
                   uses browser voices. Groq is NOT used on this path.
  * ``"piper"``  — Piper only; 503 when unavailable.
  * ``"groq"``   — Groq Orpheus only (SIGNSAGE_TTS_PROVIDER=groq, or the UI
                   picked "Groq Orpheus"). Browser fallback on any failure.
  * ``"browser"``— never call the backend (frontend Web Speech API).

Design rules:
  * Piper (piper-tts) runs fully in-process at 22.05 kHz mono; the voice is a
    pair of files (.onnx + .onnx.json) from the public Hugging Face repo
    rhasspy/piper-voices. The Docker image bakes them in at build time
    (/srv/voices); a runtime fallback downloads them once into
    SIGNSAGE_PIPER_CACHE (default /tmp/piper) when absent.
  * Piper input is capped at 300 characters; the loaded voice is cached for
    reuse. Text is never logged.
  * The Groq API key never leaves the backend. Uses SIGNSAGE_GROQ_TTS_API_KEY
    when set (dedicated voice-model key), else SIGNSAGE_GROQ_API_KEY.
  * Orpheus input is capped at 200 characters (provider limit).
  * Every failure mode raises TtsUnavailableError (→ HTTP 503) carrying a short
    machine ``code`` (TTS_MODEL_TERMS_REQUIRED, TTS_BAD_KEY, TTS_RATE_LIMITED,
    TTS_UNCONFIGURED, TTS_UNAVAILABLE) so the frontend can fall back to
    browser voices and say why. The app never hard-fails because audio is
    unavailable, and Piper paths never raise anything else.
"""
from __future__ import annotations

import io
import logging
import re
import wave
from pathlib import Path
from typing import Optional

import httpx
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.errors import TtsUnavailableError

logger = logging.getLogger(__name__)

# Voice personas supported by canopylabs/orpheus-v1-english (checked against
# the Groq docs, 2026: autumn, diana, hannah, austin, daniel, troy).
GROQ_ORPHEUS_VOICES = frozenset({"autumn", "diana", "hannah", "austin", "daniel", "troy"})

PIPER_HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
PIPER_SAMPLE_RATE = 22050  # the selected voice (en_US-kristin-medium) is 22.05 kHz
PIPER_MAX_INPUT_CHARS = 300
PIPER_MIN_ONNX_BYTES = 10 * 1024 * 1024  # a truncated/corrupt download is useless
PIPER_MIN_CONFIG_BYTES = 512
# en_<region>-<name>-<quality>, e.g. en_US-kristin-medium / en_GB-alba-medium
_PIPER_VOICE_RE = re.compile(r"^(en_(?:US|GB))-([a-z0-9_]+)-(medium|high|low|x_low)$")

# Loaded voices are reused across requests. Keyed by the resolved onnx path.
_VOICE_CACHE: dict[str, object] = {}


def _tts_api_key(settings: Settings) -> Optional[str]:
    """The dedicated TTS key when set, else the shared Groq key."""
    return settings.groq_tts_api_key or settings.groq_api_key


def _piper_voice_url(voice: str) -> tuple[str, str]:
    match = _PIPER_VOICE_RE.match(voice)
    if not match:
        raise ValueError(f"unsupported piper voice name {voice!r}")
    lang, name, quality = match.groups()
    base = f"{PIPER_HF_BASE}/en/{lang}/{name}/{quality}/{voice}"
    return f"{base}.onnx", f"{base}.onnx.json"


def _piper_paths(settings: Settings) -> Optional[tuple[Path, Path]]:
    """Voice pair from SIGNSAGE_PIPER_DIR or the runtime cache, no network."""
    voice = (settings.piper_voice or "").strip()
    if not _PIPER_VOICE_RE.match(voice):
        return None
    for directory in (settings.piper_dir, settings.piper_cache):
        if not directory:
            continue
        onnx = Path(directory) / f"{voice}.onnx"
        config = Path(directory) / f"{voice}.onnx.json"
        if onnx.is_file() and config.is_file() and onnx.stat().st_size >= PIPER_MIN_ONNX_BYTES:
            return onnx, config
    return None


def _download_piper_voice(settings: Settings) -> Optional[tuple[Path, Path]]:
    """One-time download into the runtime cache; never logs file contents."""
    voice = (settings.piper_voice or "").strip()
    try:
        onnx_url, config_url = _piper_voice_url(voice)
    except ValueError:
        return None
    cache = Path(settings.piper_cache)
    try:
        cache.mkdir(parents=True, exist_ok=True)
        onnx = cache / f"{voice}.onnx"
        config = cache / f"{voice}.onnx.json"
        for url, dest, minimum in (
            (onnx_url, onnx, PIPER_MIN_ONNX_BYTES),
            (config_url, config, PIPER_MIN_CONFIG_BYTES),
        ):
            tmp = dest.with_suffix(dest.suffix + ".part")
            with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as resp:
                resp.raise_for_status()
                size = 0
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_bytes(1024 * 1024):
                        fh.write(chunk)
                        size += len(chunk)
            if size < minimum:
                tmp.unlink(missing_ok=True)
                logger.error("piper voice download too small (%s bytes)", size)
                return None
            tmp.replace(dest)
        return onnx, config
    except Exception as exc:  # noqa: BLE001 -- any download failure → unavailable, never raise
        logger.error("piper voice download failed (%s)", type(exc).__name__)
        return None


def _piper_import():
    try:
        from piper import PiperVoice

        return PiperVoice, None
    except Exception as exc:  # noqa: BLE001 -- import must never take the app down
        return None, f"piper import failed ({type(exc).__name__})"


def piper_status(settings: Settings) -> tuple[bool, Optional[str]]:
    """(available, reason). Cheap: import check + file presence only, no network, no model load."""
    PiperVoice, err = _piper_import()
    if PiperVoice is None:
        return False, err
    voice = (settings.piper_voice or "").strip()
    if not _PIPER_VOICE_RE.match(voice):
        return False, f"SIGNSAGE_PIPER_VOICE {voice!r} is not a supported voice name"
    if _piper_paths(settings) is not None:
        return True, None
    return False, (
        f"voice files for {voice} not found in SIGNSAGE_PIPER_DIR or SIGNSAGE_PIPER_CACHE "
        "(they are downloaded on first synthesis attempt)"
    )


def piper_available(settings: Settings) -> bool:
    return piper_status(settings)[0]


def _load_piper_voice(settings: Settings):
    PiperVoice, err = _piper_import()
    if PiperVoice is None:
        return None, err
    paths = _piper_paths(settings) or _download_piper_voice(settings)
    if paths is None:
        voice = (settings.piper_voice or "").strip()
        return None, f"voice files for {voice} are not available (SIGNSAGE_PIPER_DIR / SIGNSAGE_PIPER_CACHE)"
    onnx, config = paths
    key = str(onnx)
    cached = _VOICE_CACHE.get(key)
    if cached is not None:
        return cached, None
    try:
        voice_obj = PiperVoice.load(str(onnx), config_path=str(config))
    except Exception as exc:  # noqa: BLE001
        return None, f"voice load failed ({type(exc).__name__})"
    _VOICE_CACHE[key] = voice_obj
    return voice_obj, None


def _piper_wav_bytes(voice_obj, text: str) -> bytes:
    """22.05 kHz mono 16-bit WAV, built in memory. piper-tts API varies by release."""
    if hasattr(voice_obj, "synthesize"):
        frames = b"".join(
            chunk.audio_int16_bytes for chunk in voice_obj.synthesize(text)
        )
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(PIPER_SAMPLE_RATE)
            wav.writeframes(frames)
        return buf.getvalue()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(PIPER_SAMPLE_RATE)
        voice_obj.synthesize_wav(text, wav)
    return buf.getvalue()


def _synthesize_piper(text: str, settings: Settings) -> bytes:
    """Blocking Piper synthesis (run in a threadpool). Raises TtsUnavailableError on any failure."""
    voice_obj, reason = _load_piper_voice(settings)
    if voice_obj is None:
        logger.error("piper tts unavailable: %s", reason)
        raise TtsUnavailableError(f"Server voice is unavailable: {reason}.", code="TTS_UNCONFIGURED")
    try:
        wav_bytes = _piper_wav_bytes(voice_obj, text)
    except Exception as exc:  # noqa: BLE001 -- never raise raw piper/onnx errors
        logger.error("piper tts synthesis failed (%s)", type(exc).__name__)
        raise TtsUnavailableError("Server voice failed to produce audio.") from exc
    if not wav_bytes:
        logger.error("piper tts synthesis returned no audio")
        raise TtsUnavailableError("Server voice returned no audio.")
    return wav_bytes


def tts_provider_available(settings: Settings) -> bool:
    """True when a backend synthesis will be attempted for the configured provider."""
    provider = (settings.tts_provider or "auto").strip().lower()
    if provider == "groq":
        return bool(_tts_api_key(settings))
    if provider in ("auto", "piper"):
        return piper_available(settings)
    return False


async def synthesize_speech(
    text: str,
    settings: Settings,
    voice: Optional[str] = None,
    transport: Optional[httpx.AsyncBaseTransport] = None,
    provider: Optional[str] = None,
) -> bytes:
    """Return WAV bytes for `text`.

    `provider` is the per-request override ("auto" | "piper" | "groq" | "browser")
    the frontend sends when the user picks one; unset means SIGNSAGE_TTS_PROVIDER.
    Raises TtsUnavailableError on any failure; never raises anything else,
    so callers can treat it as "fall back" unconditionally.
    """
    provider = (provider or settings.tts_provider or "auto").strip().lower()
    if provider == "browser":
        raise TtsUnavailableError("Browser TTS selected; backend provider is not used.")

    text = (text or "").strip()
    if not text:
        raise TtsUnavailableError("Nothing to speak.")

    if provider in ("auto", "piper"):
        if len(text) > PIPER_MAX_INPUT_CHARS:
            raise TtsUnavailableError(
                f"Text exceeds the {PIPER_MAX_INPUT_CHARS}-character server-voice limit."
            )
        return await run_in_threadpool(_synthesize_piper, text, settings)

    if provider != "groq":
        raise TtsUnavailableError(f"Unknown TTS provider {provider!r}.")

    return await _synthesize_groq(text, settings, voice, transport)


async def _synthesize_groq(
    text: str,
    settings: Settings,
    voice: Optional[str],
    transport: Optional[httpx.AsyncBaseTransport],
) -> bytes:
    api_key = _tts_api_key(settings)
    if not api_key:
        raise TtsUnavailableError(
            "TTS provider is not configured (SIGNSAGE_GROQ_TTS_API_KEY or SIGNSAGE_GROQ_API_KEY).",
            code="TTS_UNCONFIGURED",
        )

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
        raise TtsUnavailableError("TTS provider rate limit reached.", code="TTS_RATE_LIMITED")
    if resp.status_code in (401, 403):
        logger.warning("tts synthesize failed: provider rejected the key (%d)", resp.status_code)
        raise TtsUnavailableError("TTS provider rejected the configured API key.", code="TTS_BAD_KEY")
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
                "TTS model requires one-time terms acceptance in the Groq console (model_terms_required).",
                code="TTS_MODEL_TERMS_REQUIRED",
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
