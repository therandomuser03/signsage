"""Translation-history use-cases.

History is a best-effort side effect of translating: a database outage must
never make a translation fail. Failures are logged with enough detail to
diagnose (step, HTTP status, PostgREST code/message, request path) and never
with tokens, keys or payload text.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from app.core.config import Settings
from app.core.errors import AuthError, HistoryUnavailableError, NotFoundError
from app.core.security import AuthenticatedUser
from app.domain.translation.entities import HistoryRecord, NewHistoryRecord
from app.infrastructure.database.repositories.translation_repository import (
    SupabaseTranslationRepository,
    TranslationRepository,
    get_memory_repository,
)

logger = logging.getLogger(__name__)


def _log_failure(step: str, exc: Exception) -> None:
    """One warning per failed history operation. Never logs tokens, keys or
    payload text — the path is scheme-relative, and PostgREST errors name
    constraints/columns, not row data."""
    step = getattr(exc, "step", None) or step
    if isinstance(exc, HistoryUnavailableError):
        logger.warning(
            "history %s failed: status=%s path=%s pg_code=%s pg_message=%s",
            step, exc.status, exc.path, exc.pg_code, exc.pg_message,
        )
    elif isinstance(exc, AuthError):
        logger.warning(
            "history %s failed: status=401 path=%s (access token rejected)",
            step, getattr(exc, "path", "unknown"),
        )
    else:
        logger.warning("history %s failed: %s", step, type(exc).__name__)


class HistoryService:
    def __init__(self, repository: TranslationRepository):
        self._repo = repository

    async def try_record(self, user: AuthenticatedUser, record: NewHistoryRecord) -> Optional[str]:
        """Save a record; return its id, or None if saving failed (never raises). Deprecated shape; see add_word_recognition."""
        try:
            return (await self._repo.add(user, record)).id
        except Exception as exc:  # noqa: BLE001 -- history must not break translation
            _log_failure("record", exc)
            return None

    async def try_record_word(self, user: AuthenticatedUser, record: NewHistoryRecord) -> Optional[str]:
        """Save a session + message row; return the session id or None (never raises)."""
        try:
            return await self._repo.add_word_recognition(user, record)
        except Exception as exc:  # noqa: BLE001 -- history must not break translation
            _log_failure("record", exc)
            return None

    async def list(self, user: AuthenticatedUser, limit: int, offset: int) -> Tuple[List[HistoryRecord], int]:
        try:
            return await self._repo.list(user, limit, offset)
        except Exception as exc:
            _log_failure("list", exc)
            raise

    async def delete(self, user: AuthenticatedUser, record_id: str) -> None:
        try:
            deleted = await self._repo.delete(user, record_id)
        except Exception as exc:
            _log_failure("delete", exc)
            raise
        if not deleted:
            raise NotFoundError("History entry not found.")

    async def clear(self, user: AuthenticatedUser) -> int:
        try:
            return await self._repo.clear(user)
        except Exception as exc:
            _log_failure("clear", exc)
            raise


def build_history_service(settings: Settings) -> Optional[HistoryService]:
    """None when history is disabled or not configured -- callers treat that as 'don't save'."""
    if not settings.history_enabled:
        return None
    if settings.history_backend == "memory":
        return HistoryService(get_memory_repository())
    if settings.supabase_url and settings.supabase_publishable_key:
        return HistoryService(SupabaseTranslationRepository.from_settings(settings))
    return None
