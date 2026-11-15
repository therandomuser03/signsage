"""Text-to-speech (Auto-Speak). Backend proxies Groq Orpheus when configured.

The browser Web Speech API remains the free fallback and is used entirely
client-side; this route exists so users get a natural voice without shipping
any key to the browser. Every failure answers 503 TTS_UNAVAILABLE — the
frontend treats any non-200 as "fall back to browser speech".
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.core.ratelimit import rate_limit_dependency
from app.services import tts_service

router = APIRouter(prefix="/speech", tags=["speech"])


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)
    voice: Optional[str] = Field(None, max_length=32)


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
    audio = await tts_service.synthesize_speech(payload.text, settings, payload.voice)
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"Cache-Control": "no-store", "Content-Disposition": "inline"},
    )
