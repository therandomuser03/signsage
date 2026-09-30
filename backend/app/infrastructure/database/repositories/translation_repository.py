"""Translation-history repositories.

TranslationRepository is the interface services depend on. Two implementations:
  * SupabaseTranslationRepository -- production (Postgres + RLS via PostgREST)
  * InMemoryTranslationRepository  -- dev/tests (SIGNSAGE_HISTORY_BACKEND=memory)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Protocol, Tuple

from app.core.config import Settings
from app.core.security import AuthenticatedUser
from app.domain.translation.entities import Direction, HistoryRecord, NewHistoryRecord, ResultType
from app.infrastructure.database.supabase import SupabaseRestClient

TABLE = "translation_sessions"


class TranslationRepository(Protocol):
    async def add(self, user: AuthenticatedUser, record: NewHistoryRecord) -> HistoryRecord: ...
    async def list(self, user: AuthenticatedUser, limit: int, offset: int) -> Tuple[List[HistoryRecord], int]: ...
    async def delete(self, user: AuthenticatedUser, record_id: str) -> bool: ...
    async def clear(self, user: AuthenticatedUser) -> int: ...


def _row_to_record(row: dict) -> HistoryRecord:
    return HistoryRecord(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        direction=Direction(row["direction"]),
        input_text=row.get("input_text"),
        output_text=row.get("output_text") or "",
        confidence=row.get("confidence"),
        model_version=row.get("model_version") or "",
        result_type=ResultType(row["result_type"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


class SupabaseTranslationRepository:
    def __init__(self, client: SupabaseRestClient):
        self._db = client

    @classmethod
    def from_settings(cls, settings: Settings) -> "SupabaseTranslationRepository":
        return cls(SupabaseRestClient(settings))

    async def add(self, user: AuthenticatedUser, record: NewHistoryRecord) -> HistoryRecord:
        body = {
            "user_id": user.id,
            "direction": record.direction.value,
            "input_text": record.input_text,
            "output_text": record.output_text,
            "confidence": record.confidence,
            "model_version": record.model_version,
            "result_type": record.result_type.value,
        }
        resp = await self._db.request("POST", TABLE, user, json=body, prefer="return=representation")
        return _row_to_record(resp.json()[0])

    async def list(self, user: AuthenticatedUser, limit: int, offset: int) -> Tuple[List[HistoryRecord], int]:
        resp = await self._db.request(
            "GET", TABLE, user,
            params={"select": "*", "order": "created_at.desc", "limit": limit, "offset": offset},
            prefer="count=exact",
        )
        rows = [_row_to_record(r) for r in resp.json()]
        # Content-Range looks like "0-9/42" (or "*/0" when empty)
        total = len(rows) + offset
        content_range = resp.headers.get("content-range", "")
        if "/" in content_range and content_range.rsplit("/", 1)[1].isdigit():
            total = int(content_range.rsplit("/", 1)[1])
        return rows, total

    async def delete(self, user: AuthenticatedUser, record_id: str) -> bool:
        resp = await self._db.request(
            "DELETE", TABLE, user, params={"id": f"eq.{record_id}"}, prefer="return=representation"
        )
        return len(resp.json()) > 0

    async def clear(self, user: AuthenticatedUser) -> int:
        resp = await self._db.request(
            "DELETE", TABLE, user, params={"user_id": f"eq.{user.id}"}, prefer="return=representation"
        )
        return len(resp.json())


class InMemoryTranslationRepository:
    def __init__(self) -> None:
        self._by_user: Dict[str, List[HistoryRecord]] = {}

    async def add(self, user: AuthenticatedUser, record: NewHistoryRecord) -> HistoryRecord:
        saved = HistoryRecord(
            id=str(uuid.uuid4()), user_id=user.id, direction=record.direction,
            input_text=record.input_text, output_text=record.output_text,
            confidence=record.confidence, model_version=record.model_version,
            result_type=record.result_type, created_at=datetime.now(timezone.utc),
        )
        self._by_user.setdefault(user.id, []).append(saved)
        return saved

    async def list(self, user: AuthenticatedUser, limit: int, offset: int) -> Tuple[List[HistoryRecord], int]:
        items = sorted(self._by_user.get(user.id, []), key=lambda r: r.created_at, reverse=True)
        return items[offset : offset + limit], len(items)

    async def delete(self, user: AuthenticatedUser, record_id: str) -> bool:
        items = self._by_user.get(user.id, [])
        kept = [r for r in items if r.id != record_id]
        self._by_user[user.id] = kept
        return len(kept) != len(items)

    async def clear(self, user: AuthenticatedUser) -> int:
        return len(self._by_user.pop(user.id, []))


_memory_repository: Optional[InMemoryTranslationRepository] = None


def get_memory_repository() -> InMemoryTranslationRepository:
    global _memory_repository
    if _memory_repository is None:
        _memory_repository = InMemoryTranslationRepository()
    return _memory_repository


def reset_memory_repository() -> None:
    global _memory_repository
    _memory_repository = None
