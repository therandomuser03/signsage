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
from app.domain.translation.entities import NewHistorySentence, SessionRecord
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

    # ------------------------------------------------------------- recording
    async def record_sentence(
        self, user: AuthenticatedUser, record: NewHistorySentence
    ) -> Tuple[str, bool]:
        """Persist one completed sentence. Returns (session_id, duplicated). Raises on failure."""
        session_id, _, duplicated = await self._repo.record_sentence(user, record)
        return session_id, duplicated

    async def try_record_sentence(
        self, user: AuthenticatedUser, record: NewHistorySentence
    ) -> Optional[str]:
        """Save a sentence; return its session id, or None if saving failed (never raises)."""
        try:
            session_id, _ = await self.record_sentence(user, record)
            return session_id
        except Exception as exc:  # noqa: BLE001 -- history must not break translation
            _log_failure("record", exc)
            return None

    async def try_record_sentence_checked(
        self, user: AuthenticatedUser, record: NewHistorySentence
    ) -> Tuple[bool, Optional[str]]:
        """Like try_record_sentence but also returns a compact error string on failure."""
        try:
            await self.record_sentence(user, record)
            return True, None
        except Exception as exc:  # noqa: BLE001 -- history must not break translation
            _log_failure("record", exc)
            step = getattr(exc, "step", None) or "record"
            status = getattr(exc, "status", None)
            code = getattr(exc, "pg_code", None)
            return False, f"{step.replace(' ', '_')}:{status}:{code}"

    async def close_session(self, user: AuthenticatedUser, session_key: str) -> None:
        """Best-effort: mark the interaction's history session completed. Never raises."""
        try:
            await self._repo.close_session(user, session_key)
        except Exception as exc:  # noqa: BLE001 -- best-effort bookkeeping
            _log_failure("close", exc)

    # --------------------------------------------------------------- listing
    async def list(
        self,
        user: AuthenticatedUser,
        limit: int,
        offset: int,
        direction: Optional[str] = None,
        query: Optional[str] = None,
        order: str = "newest",
    ) -> Tuple[List[SessionRecord], int]:
        try:
            return await self._repo.list_sessions(user, limit, offset, direction, query, order)
        except Exception as exc:
            _log_failure("list", exc)
            raise

    async def get(self, user: AuthenticatedUser, session_id: str) -> SessionRecord:
        try:
            session = await self._repo.get_session(user, session_id)
        except Exception as exc:
            _log_failure("get", exc)
            raise
        if session is None:
            raise NotFoundError("Translation session not found.")
        return session

    # --------------------------------------------------------------- deletion
    async def delete(self, user: AuthenticatedUser, session_id: str) -> None:
        try:
            deleted = await self._repo.delete_session(user, session_id)
        except Exception as exc:
            _log_failure("delete", exc)
            raise
        if not deleted:
            raise NotFoundError("Translation session not found.")

    async def delete_many(self, user: AuthenticatedUser, ids: List[str]) -> int:
        try:
            return await self._repo.delete_sessions(user, ids)
        except Exception as exc:
            _log_failure("bulk delete", exc)
            raise

    async def clear(self, user: AuthenticatedUser) -> int:
        try:
            return await self._repo.clear(user)
        except Exception as exc:
            _log_failure("clear", exc)
            raise

    async def selfcheck(self, user: AuthenticatedUser) -> dict:
        """Record a sentence, read the session back, delete it; step-by-step status."""
        try:
            return await self._repo.selfcheck(user)
        except Exception as exc:  # noqa: BLE001
            _log_failure("selfcheck", exc)
            return {
                "ok": False,
                "steps": [{
                    "step": (getattr(exc, "step", None) or "selfcheck").replace(" ", "_"),
                    "status": getattr(exc, "status", None),
                    "code": getattr(exc, "pg_code", None),
                }],
            }


def build_history_service(settings: Settings) -> Optional[HistoryService]:
    """None when history is disabled or not configured -- callers treat that as 'don't save'."""
    if not settings.history_enabled:
        return None
    if settings.history_backend == "memory":
        return HistoryService(get_memory_repository())
    if settings.supabase_url and settings.supabase_publishable_key:
        return HistoryService(SupabaseTranslationRepository.from_settings(settings))
    return None
