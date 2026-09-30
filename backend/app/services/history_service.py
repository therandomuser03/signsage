"""Translation-history use-cases.

History is a best-effort side effect of translating: a database outage must
never make a translation fail, so try_record() swallows and logs errors.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from app.core.config import Settings
from app.core.errors import NotFoundError
from app.core.security import AuthenticatedUser
from app.domain.translation.entities import HistoryRecord, NewHistoryRecord
from app.infrastructure.database.repositories.translation_repository import (
    SupabaseTranslationRepository,
    TranslationRepository,
    get_memory_repository,
)

logger = logging.getLogger(__name__)


class HistoryService:
    def __init__(self, repository: TranslationRepository):
        self._repo = repository

    async def try_record(self, user: AuthenticatedUser, record: NewHistoryRecord) -> Optional[str]:
        """Save a record; return its id, or None if saving failed (never raises)."""
        try:
            return (await self._repo.add(user, record)).id
        except Exception as exc:  # noqa: BLE001 -- history must not break translation
            logger.warning("Could not save history for user %s: %s", user.id, exc)
            return None

    async def list(self, user: AuthenticatedUser, limit: int, offset: int) -> Tuple[List[HistoryRecord], int]:
        return await self._repo.list(user, limit, offset)

    async def delete(self, user: AuthenticatedUser, record_id: str) -> None:
        if not await self._repo.delete(user, record_id):
            raise NotFoundError("History entry not found.")

    async def clear(self, user: AuthenticatedUser) -> int:
        return await self._repo.clear(user)


def build_history_service(settings: Settings) -> Optional[HistoryService]:
    """None when history is disabled or not configured -- callers treat that as 'don't save'."""
    if not settings.history_enabled:
        return None
    if settings.history_backend == "memory":
        return HistoryService(get_memory_repository())
    if settings.supabase_url and settings.supabase_publishable_key:
        return HistoryService(SupabaseTranslationRepository.from_settings(settings))
    return None
