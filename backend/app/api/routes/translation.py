"""REST Sign->Text (complete recorded sequence). Live capture uses /ws/translate."""
from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends

from app.core.dependencies import get_optional_history_service, get_request_user, get_translation_service
from app.core.errors import InferenceTimeoutError, InvalidLandmarksError
from app.core.security import AuthenticatedUser
from app.domain.translation.entities import Direction, NewHistoryRecord
from app.domain.translation.policies import classify_result, should_persist
from app.schemas.translation import SignToTextRequest, SignToTextResponse
from app.services.history_service import HistoryService
from app.services.translation_service import TranslationService
from app.core.config import Settings, get_settings

router = APIRouter(prefix="/translation", tags=["translation"])


@router.post("/sign-to-text", response_model=SignToTextResponse)
async def translate_sign_to_text(
    payload: SignToTextRequest,
    service: TranslationService = Depends(get_translation_service),
    user: Optional[AuthenticatedUser] = Depends(get_request_user),
    history: Optional[HistoryService] = Depends(get_optional_history_service),
    settings: Settings = Depends(get_settings),
) -> SignToTextResponse:
    from starlette.concurrency import run_in_threadpool

    landmarks = np.array([f.landmarks for f in payload.frames], dtype=np.float32)
    try:
        result = await asyncio.wait_for(
            run_in_threadpool(
                service.translate_sign_to_text, landmarks,
                min_motion=payload.min_motion, min_confidence=payload.min_confidence,
            ),
            timeout=settings.inference_timeout_s,
        )
    except TimeoutError as exc:
        raise InferenceTimeoutError(
            f"Sign->Text inference did not complete within {settings.inference_timeout_s}s."
        ) from exc
    except ValueError as exc:
        # Schema-valid JSON that still violates the canonical representation
        # (e.g. ragged frames numpy accepted): 422, never an unhandled 500.
        raise InvalidLandmarksError(str(exc)) from exc
    history_id = None
    if user and history and payload.save_history and should_persist(result):
        history_id = await history.try_record_word(user, NewHistoryRecord(
            direction=Direction.SIGN_TO_TEXT, output_text=result.text, result_type=classify_result(result),
            model_version=settings.model_version, confidence=result.confidence,
            input_text=None, sequence=1,
            title=f"Sign to text: {result.text[:50]}", status="completed",
        ))
    return SignToTextResponse(
        text=result.text, accepted=result.accepted, reason=result.reason,
        confidence=result.confidence, motion_energy=result.motion_energy,
        result_type=classify_result(result).value, history_id=history_id,
    )
