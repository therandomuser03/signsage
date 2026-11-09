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
MESSAGES_TABLE = "translation_messages"

# Sessions own the translatable text; the live schema keeps it in
# translation_messages, so list() embeds that child table instead of selecting
# nonexistent columns on translation_sessions.
EMBED_SELECT = (
    "id,title,direction,status,created_at,"
    "translation_messages(sequence,input_text,output_text,gloss,"
    "confidence,model_version,result_type)"
)


class TranslationRepository(Protocol):
    async def add(self, user: AuthenticatedUser, record: NewHistoryRecord) -> HistoryRecord: ...
    async def add_word_recognition(
        self, user: AuthenticatedUser, record: NewHistoryRecord
    ) -> str: ...
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
        title=row.get("title"),
    )


def _session_to_record(row: dict) -> HistoryRecord:
    """Map a sessions row with embedded translation_messages to a HistoryRecord.

    The first message by sequence drives the translation fields. A session with
    no message (legacy data) falls back to its title for the output text.
    """
    messages = sorted(row.get("translation_messages") or [], key=lambda m: m.get("sequence") or 0)
    msg = messages[0] if messages else None
    result_type = ResultType.MODEL
    if msg and msg.get("result_type"):
        try:
            result_type = ResultType(msg["result_type"])
        except ValueError:
            result_type = ResultType.MODEL
    return HistoryRecord(
        id=str(row["id"]),
        user_id=str(row.get("user_id") or ""),
        direction=Direction(row["direction"]),
        input_text=msg.get("input_text") if msg else None,
        output_text=(msg.get("output_text") if msg else None) or row.get("title") or "",
        confidence=msg.get("confidence") if msg else None,
        model_version=msg.get("model_version") or "" if msg else "",
        result_type=result_type,
        created_at=datetime.fromisoformat(row["created_at"]),
        title=row.get("title"),
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

    async def add_word_recognition(
        self, user: AuthenticatedUser, record: NewHistoryRecord
    ) -> str:
        """Create one translation_sessions row and one translation_messages row.

        Returns the new session id. Raises on failure; callers (HistoryService)
        swallow and log errors so history never breaks translation.
        """
        # 1) Session row.
        session_body = {
            "user_id": user.id,
            "direction": record.direction.value,
            "status": record.status,
            "title": record.title,
        }
        session_resp = await self._db.request(
            "POST", TABLE, user, json=session_body, prefer="return=representation"
        )
        session_id = str(session_resp.json()[0]["id"])

        # 2) Message row linked to the session.
        message_body = {
            "session_id": session_id,
            "sequence": record.sequence,
            "direction": record.direction.value,
            "input_text": record.input_text,
            "output_text": record.output_text,
            "gloss": record.gloss,
            "confidence": record.confidence,
            "model_version": record.model_version,
            "result_type": record.result_type.value,
        }
        message_resp = await self._db.request(
            "POST", "translation_messages", user, json=message_body, prefer="return=representation"
        )
        return str(message_resp.json()[0]["id"])

    async def list(self, user: AuthenticatedUser, limit: int, offset: int) -> Tuple[List[HistoryRecord], int]:
        resp = await self._db.request(
            "GET", TABLE, user,
            params={"select": EMBED_SELECT, "order": "created_at.desc", "limit": limit, "offset": offset},
            prefer="count=exact",
        )
        rows = [_session_to_record(r) for r in resp.json()]
        # Content-Range looks like "0-9/42" (or "*/0" when empty)
        total = len(rows) + offset
        content_range = resp.headers.get("content-range", "")
        if "/" in content_range and content_range.rsplit("/", 1)[1].isdigit():
            total = int(content_range.rsplit("/", 1)[1])
        return rows, total

    async def delete(self, user: AuthenticatedUser, record_id: str) -> bool:
        # 1) Messages first: nothing is left behind when the FK does not
        #    cascade; a no-rows delete is an ordinary empty response, not an
        #    error. If the FK is ON DELETE CASCADE this simply deletes zero
        #    messages and then the session.
        await self._db.request(
            "DELETE", MESSAGES_TABLE, user, params={"session_id": f"eq.{record_id}"},
        )
        # 2) The session itself (RLS keeps this to the user's own rows).
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
        self._sessions: Dict[str, List[dict]] = {}
        self._messages: Dict[str, List[dict]] = {}

    async def add(self, user: AuthenticatedUser, record: NewHistoryRecord) -> HistoryRecord:
        saved = HistoryRecord(
            id=str(uuid.uuid4()), user_id=user.id, direction=record.direction,
            input_text=record.input_text, output_text=record.output_text,
            confidence=record.confidence, model_version=record.model_version,
            result_type=record.result_type, created_at=datetime.now(timezone.utc),
        )
        self._by_user.setdefault(user.id, []).append(saved)
        return saved

    async def add_word_recognition(
        self, user: AuthenticatedUser, record: NewHistoryRecord
    ) -> str:
        session_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        self._sessions.setdefault(user.id, []).append({
            "id": session_id, "user_id": user.id, "direction": record.direction.value,
            "status": record.status, "title": record.title,
            "created_at": now.isoformat(),
        })
        self._messages.setdefault(session_id, []).append({
            "id": message_id, "session_id": session_id, "sequence": record.sequence,
            "direction": record.direction.value, "input_text": record.input_text,
            "output_text": record.output_text, "gloss": record.gloss,
            "confidence": record.confidence, "model_version": record.model_version,
            "result_type": record.result_type.value,
            "created_at": now.isoformat(),
        })
        self._by_user.setdefault(user.id, []).append(HistoryRecord(
            id=session_id, user_id=user.id, direction=record.direction,
            input_text=record.input_text, output_text=record.output_text,
            confidence=record.confidence, model_version=record.model_version,
            result_type=record.result_type, created_at=now, title=record.title,
        ))
        return session_id

    async def list(self, user: AuthenticatedUser, limit: int, offset: int) -> Tuple[List[HistoryRecord], int]:
        items = sorted(self._by_user.get(user.id, []), key=lambda r: r.created_at, reverse=True)
        return items[offset : offset + limit], len(items)

    async def delete(self, user: AuthenticatedUser, record_id: str) -> bool:
        items = self._by_user.get(user.id, [])
        kept = [r for r in items if r.id != record_id]
        if len(kept) == len(items):
            return False
        self._by_user[user.id] = kept
        self._sessions[user.id] = [s for s in self._sessions.get(user.id, []) if s["id"] != record_id]
        self._messages.pop(record_id, None)
        return True

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
