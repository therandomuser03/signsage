"""Text-to-speech (Auto-Speak). Piper (free, offline, default) or Groq Orpheus.

The browser Web Speech API remains the free fallback and is used entirely
client-side; this route exists so users get a voice without shipping any key
to the browser. 503 responses carry a machine `code` (TTS_MODEL_TERMS_REQUIRED,
TTS_BAD_KEY, TTS_RATE_LIMITED, TTS_UNCONFIGURED, TTS_UNAVAILABLE) — the
frontend falls back to browser speech and shows the code.
"""
from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, Response, UploadFile
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.core.errors import SttUnavailableError
from app.core.ratelimit import rate_limit_dependency
from app.services import stt_service, tts_service

router = APIRouter(prefix="/speech", tags=["speech"])


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)
    voice: Optional[str] = Field(None, max_length=32)
    # Per-request provider override from the UI picker ("auto" | "piper" |
    # "groq"). Unset → SIGNSAGE_TTS_PROVIDER decides. Browser never reaches here.
    provider: Optional[Literal["auto", "piper", "groq"]] = None


@router.post(
    "/speak",
    responses={
        200: {"description": "WAV audio for the text (audio/wav)."},
        503: {"description": "TTS_UNAVAILABLE — frontend falls back to browser speech."},
    },
    dependencies=[Depends(rate_limit_dependency("tts"))],
)
async def speak(
    payload: SpeakRequest,
    settings: Settings = Depends(get_settings),
) -> Response:
    # Thin handler: the service raises TtsUnavailableError (→ 503) on any failure.
    audio = await tts_service.synthesize_speech(payload.text, settings, payload.voice, provider=payload.provider)
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"Cache-Control": "no-store", "Content-Disposition": "inline"},
    )


# ---------- Server transcription (microphone fallback) ----------

MAX_STT_BYTES = 1_048_576  # 1 MB
_ALLOWED_TYPES = {
    "audio/webm", "video/webm",
    "audio/ogg", "application/ogg",
    "audio/mp4", "video/mp4",
    "audio/wav", "audio/x-wav", "audio/wave",
}
_EXT_TO_TYPE = {"webm": "audio/webm", "ogg": "audio/ogg", "mp4": "audio/mp4", "wav": "audio/wav"}


class TranscribeResponse(BaseModel):
    text: str


@router.post(
    "/transcribe",
    responses={
        200: {"description": "{text} transcript of the clip (never stored, never logged)."},
        503: {"description": "STT_UNCONFIGURED | STT_RATE_LIMITED | STT_BAD_AUDIO — the UI falls back to typing."},
    },
    dependencies=[Depends(rate_limit_dependency("stt"))],
)
async def transcribe(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
) -> TranscribeResponse:
    data = await file.read(MAX_STT_BYTES + 1)
    if len(data) > MAX_STT_BYTES:
        raise SttUnavailableError("Audio exceeds the 1 MB limit.", code="STT_BAD_AUDIO")
    if not data:
        raise SttUnavailableError("Empty audio.", code="STT_BAD_AUDIO")
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type not in _ALLOWED_TYPES:
        filename = file.filename or ""
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if content_type in ("", "application/octet-stream") and ext in _EXT_TO_TYPE:
            content_type = _EXT_TO_TYPE[ext]
        else:
            raise SttUnavailableError(
                "Unsupported audio type; send webm, ogg, mp4 or wav.", code="STT_BAD_AUDIO"
            )
    text = await stt_service.transcribe_audio(
        data, file.filename or "audio.webm", content_type, settings
    )
    return TranscribeResponse(text=text)
