"""Per-user translation history. All routes require a signed-in user."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.core.dependencies import get_current_user, get_history_service
from app.core.ratelimit import rate_limit_dependency
from app.core.security import AuthenticatedUser
from app.schemas.history import DeletedResponse, HistoryItem, HistoryListResponse
from app.services.history_service import HistoryService

router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=HistoryListResponse)
async def list_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: AuthenticatedUser = Depends(get_current_user),
    service: HistoryService = Depends(get_history_service),
) -> HistoryListResponse:
    records, total = await service.list(user, limit, offset)
    return HistoryListResponse(
        items=[HistoryItem(
            id=r.id, direction=r.direction.value, input_text=r.input_text, output_text=r.output_text,
            confidence=r.confidence, model_version=r.model_version, result_type=r.result_type.value,
            created_at=r.created_at, title=r.title,
        ) for r in records],
        total=total, limit=limit, offset=offset,
    )


@router.delete(
    "", response_model=DeletedResponse,
    dependencies=[Depends(rate_limit_dependency("history_write"))],
)
async def clear_history(
    user: AuthenticatedUser = Depends(get_current_user),
    service: HistoryService = Depends(get_history_service),
) -> DeletedResponse:
    return DeletedResponse(deleted=await service.clear(user))


@router.delete(
    "/{record_id}", response_model=DeletedResponse,
    dependencies=[Depends(rate_limit_dependency("history_write"))],
)
async def delete_history_entry(
    record_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
    service: HistoryService = Depends(get_history_service),
) -> DeletedResponse:
    await service.delete(user, record_id)
    return DeletedResponse(deleted=1)
