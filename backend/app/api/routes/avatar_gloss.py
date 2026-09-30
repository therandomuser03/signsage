"""English -> gloss IDs for the frontend's avatar. URL stays /api/v1/text-to-sign;
module names are avatar_gloss to avoid the neural ml/*/text_to_sign.py."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.core.dependencies import get_optional_history_service, get_request_user
from app.core.security import AuthenticatedUser
from app.domain.translation.entities import Direction, NewHistoryRecord, ResultType
from app.schemas.avatar_gloss import AvatarGlossRequest, AvatarGlossResponse, AvatarVocabularyResponse
from app.services.avatar_gloss_service import generate_gloss, load_vocabulary
from app.services.history_service import HistoryService

router = APIRouter(prefix="/text-to-sign", tags=["avatar-gloss"])


@router.get("/vocabulary", response_model=AvatarVocabularyResponse)
def vocabulary(settings: Settings = Depends(get_settings)) -> AvatarVocabularyResponse:
    return AvatarVocabularyResponse(gloss_ids=sorted(load_vocabulary(settings)))


@router.post("", response_model=AvatarGlossResponse)
async def avatar_gloss(
    payload: AvatarGlossRequest,
    user: Optional[AuthenticatedUser] = Depends(get_request_user),
    history: Optional[HistoryService] = Depends(get_optional_history_service),
    settings: Settings = Depends(get_settings),
) -> AvatarGlossResponse:
    result = await generate_gloss(payload.text, settings)
    history_id = None
    if user and history and payload.save_history and result.gloss:
        history_id = await history.try_record(user, NewHistoryRecord(
            direction=Direction.TEXT_TO_SIGN, input_text=payload.text, output_text=" ".join(result.gloss),
            result_type=ResultType.AVATAR_GLOSS, model_version="avatar-gloss-v1",
        ))
    return AvatarGlossResponse(
        gloss=result.gloss, unmatched_words=result.unmatched_words,
        source=result.source, history_id=history_id,
    )
