"""Per-user, session-based translation history. All routes require a signed-in user.

Sessions are the unit of history: one row per translation interaction, with its
completed sentences embedded. Ownership is enforced by the backend (user comes
from the verified token) and by Postgres RLS; client-supplied ids are never
trusted for scoping.
"""
from __future__ import annotations

from typing import Literal, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query

from app.core.dependencies import get_current_user, get_history_service
from app.core.ratelimit import rate_limit_dependency
from app.core.security import AuthenticatedUser
from app.domain.translation.entities import SessionRecord
from app.schemas.history import (
    BulkDeleteRequest,
    DeletedResponse,
    HistoryListResponse,
    HistorySentence,
    HistorySessionDetail,
    HistorySessionSummary,
)
from app.services.history_service import HistoryService

router = APIRouter(prefix="/history", tags=["history"])


def _summary(record: SessionRecord) -> HistorySessionSummary:
    first = record.first_sentence
    return HistorySessionSummary(
        id=record.id,
        direction=record.direction.value,
        title=record.title,
        status=record.status,
        created_at=record.created_at,
        ended_at=record.ended_at,
        sentence_count=record.sentence_count,
        preview=record.preview,
        model_version=(first.model_version if first else "") or "",
        result_type=record.result_type.value,
    )


def _detail(record: SessionRecord) -> HistorySessionDetail:
    sentences = [
        HistorySentence(
            sequence_number=s.sequence_number,
            direction=s.direction.value,
            input_text=s.input_text,
            output_text=s.output_text,
            gloss=s.gloss,
            confidence=s.confidence,
            model_version=s.model_version,
            result_type=s.result_type.value,
            created_at=s.created_at,
        )
        for s in record.sentences
    ]
    return HistorySessionDetail(**_summary(record).model_dump(), sentences=sentences)


@router.get("", response_model=HistoryListResponse)
async def list_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    direction: Optional[Literal["sign_to_text", "text_to_sign"]] = Query(None),
    q: Optional[str] = Query(None, max_length=200),
    sort: Literal["newest", "oldest"] = Query("newest"),
    user: AuthenticatedUser = Depends(get_current_user),
    service: HistoryService = Depends(get_history_service),
) -> HistoryListResponse:
    records, total = await service.list(user, limit, offset, direction, q, sort)
    return HistoryListResponse(
        items=[_summary(r) for r in records],
        total=total, limit=limit, offset=offset,
    )


@router.get(
    "/selfcheck",
    dependencies=[Depends(rate_limit_dependency("history_write"))],
)
async def history_selfcheck(
    user: AuthenticatedUser = Depends(get_current_user),
    service: HistoryService = Depends(get_history_service),
) -> dict:
    """Record a sentence, read its session back, delete it; return step-by-step status."""
    return await service.selfcheck(user)


@router.post(
    "/bulk-delete", response_model=DeletedResponse,
    dependencies=[Depends(rate_limit_dependency("history_write"))],
)
async def bulk_delete_history(
    payload: BulkDeleteRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    service: HistoryService = Depends(get_history_service),
) -> DeletedResponse:
    """Delete many sessions by id — but only the caller's own rows."""
    deleted = await service.delete_many(user, payload.ids)
    return DeletedResponse(deleted=deleted)


@router.get("/{session_id}", response_model=HistorySessionDetail)
async def get_history_session(
    session_id: UUID,
    user: AuthenticatedUser = Depends(get_current_user),
    service: HistoryService = Depends(get_history_service),
) -> HistorySessionDetail:
    record = await service.get(user, str(session_id))
    return _detail(record)


@router.delete(
    "/{session_id}", response_model=DeletedResponse,
    dependencies=[Depends(rate_limit_dependency("history_write"))],
)
async def delete_history_session(
    session_id: UUID,
    user: AuthenticatedUser = Depends(get_current_user),
    service: HistoryService = Depends(get_history_service),
) -> DeletedResponse:
    await service.delete(user, str(session_id))
    return DeletedResponse(deleted=1)


@router.delete(
    "", response_model=DeletedResponse,
    dependencies=[Depends(rate_limit_dependency("history_write"))],
)
async def clear_history(
    user: AuthenticatedUser = Depends(get_current_user),
    service: HistoryService = Depends(get_history_service),
) -> DeletedResponse:
    """Delete ALL sessions belonging to the authenticated user (their sentences cascade)."""
    return DeletedResponse(deleted=await service.clear(user))
