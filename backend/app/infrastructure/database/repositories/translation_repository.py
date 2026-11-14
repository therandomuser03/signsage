"""Translation-history repositories.

TranslationRepository is the interface services depend on. Two implementations:
  * SupabaseTranslationRepository -- production (Postgres + RLS via PostgREST)
  * InMemoryTranslationRepository  -- dev/tests (SIGNSAGE_HISTORY_BACKEND=memory)
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Protocol, Tuple

from app.core.config import Settings
from app.core.errors import HistoryUnavailableError
from app.core.security import AuthenticatedUser
from app.domain.translation.entities import Direction, HistoryRecord, NewHistoryRecord, ResultType
from app.infrastructure.database.supabase import SupabaseRestClient

logger = logging.getLogger(__name__)

TABLE = "translation_sessions"
MESSAGES_TABLE = "translation_messages"


def _tag(exc: Exception, step: str) -> Exception:
    """Attach the failing step so the service layer can log it precisely."""
    try:
        exc.step = step  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 -- tagging is best-effort; logging still works untagged
        pass
    return exc

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
    async def selfcheck(self, user: AuthenticatedUser) -> dict: ...


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


def _is_relationship_error(exc: HistoryUnavailableError) -> bool:
    """PostgREST relationship-resolution failures (the embed's FK is gone)."""
    return (
        (exc.status or 0) == 400
        and bool(exc.pg_code)
        and str(exc.pg_code).upper().startswith("PGRST2")
    )


def _total_from(resp, offset: int, fallback_page_size: int) -> int:
    """Content-Range looks like "0-9/42" (or "*/0" when empty)."""
    total = fallback_page_size + offset
    content_range = resp.headers.get("content-range", "")
    if "/" in content_range and content_range.rsplit("/", 1)[1].isdigit():
        total = int(content_range.rsplit("/", 1)[1])
    return total


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

    async def add(self, user: AuthenticatedUser, record: NewHistoryRecord) -> HistoryRecord:  # deprecated: old shape, kept for the memory backend
        body = {
            "user_id": user.id,
            "direction": record.direction.value,
            "input_text": record.input_text,
            "output_text": record.output_text,
            "confidence": record.confidence,
            "model_version": record.model_version,
            "result_type": record.result_type.value,
        }
        try:
            resp = await self._db.request("POST", TABLE, user, json=body, prefer="return=representation")
        except Exception as exc:
            raise _tag(exc, "record insert") from exc
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
        try:
            session_resp = await self._db.request(
                "POST", TABLE, user, json=session_body, prefer="return=representation"
            )
        except Exception as exc:
            raise _tag(exc, "session insert") from exc
        session_id = str(session_resp.json()[0]["id"])

        # 2) Message row linked to the session. user_id is deliberately NOT
        #    sent: the live translation_messages shape must be confirmed by
        #    the information_schema query (see the runbook/report); RLS via
        #    the owning session covers ownership.
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
        try:
            message_resp = await self._db.request(
                "POST", MESSAGES_TABLE, user, json=message_body, prefer="return=representation"
            )
        except Exception as exc:
            raise _tag(exc, "message insert") from exc
        return str(message_resp.json()[0]["id"])

    async def list(self, user: AuthenticatedUser, limit: int, offset: int) -> Tuple[List[HistoryRecord], int]:
        try:
            resp = await self._db.request(
                "GET", TABLE, user,
                params={"select": EMBED_SELECT, "order": "created_at.desc", "limit": limit, "offset": offset},
                prefer="count=exact",
            )
        except HistoryUnavailableError as exc:
            # PostgREST relationship errors (PGRST2xx): the FK behind the embed
            # is missing. Fall back to two requests so history still lists.
            if not _is_relationship_error(exc):
                raise _tag(exc, "list") from exc
            logger.warning(
                "history list: embed failed (status=%s pg_code=%s); "
                "falling back to a two-request read (sessions, then messages)",
                exc.status, exc.pg_code,
            )
            return await self._list_two_requests(user, limit, offset)
        except Exception as exc:
            raise _tag(exc, "list") from exc
        rows = [_session_to_record(r) for r in resp.json()]
        return rows, _total_from(resp, offset, len(rows))

    async def _list_two_requests(
        self, user: AuthenticatedUser, limit: int, offset: int
    ) -> Tuple[List[HistoryRecord], int]:
        # 1) The page of sessions.
        try:
            resp = await self._db.request(
                "GET", TABLE, user,
                params={"select": "id,user_id,title,direction,status,created_at",
                        "order": "created_at.desc", "limit": limit, "offset": offset},
                prefer="count=exact",
            )
        except Exception as exc:
            raise _tag(exc, "list sessions") from exc
        session_rows = resp.json()
        total = _total_from(resp, offset, len(session_rows))
        if not session_rows:
            return [], total
        # 2) Their messages, one request by session_id in (...). No FK required.
        ids = ",".join(str(r["id"]) for r in session_rows)
        try:
            msg_resp = await self._db.request(
                "GET", MESSAGES_TABLE, user,
                params={"select": ("session_id,sequence,input_text,output_text,"
                                   "gloss,confidence,model_version,result_type"),
                        "session_id": f"in.({ids})", "order": "sequence.asc"},
            )
        except Exception as exc:
            raise _tag(exc, "list messages") from exc
        messages_by_session: Dict[str, List[dict]] = {}
        for m in msg_resp.json():
            messages_by_session.setdefault(str(m.get("session_id")), []).append(m)
        records = []
        for r in session_rows:
            row = dict(r)
            row["translation_messages"] = messages_by_session.get(str(r.get("id")), [])
            records.append(_session_to_record(row))
        return records, total

    async def delete(self, user: AuthenticatedUser, record_id: str) -> bool:
        # 1) Messages first: nothing is left behind when the FK does not
        #    cascade; a no-rows delete is an ordinary empty response, not an
        #    error. If the FK is ON DELETE CASCADE this simply deletes zero
        #    messages and then the session.
        try:
            await self._db.request(
                "DELETE", MESSAGES_TABLE, user, params={"session_id": f"eq.{record_id}"},
            )
        except Exception as exc:
            raise _tag(exc, "delete messages") from exc
        # 2) The session itself (RLS keeps this to the user's own rows).
        try:
            resp = await self._db.request(
                "DELETE", TABLE, user, params={"id": f"eq.{record_id}"}, prefer="return=representation"
            )
        except Exception as exc:
            raise _tag(exc, "delete session") from exc
        return len(resp.json()) > 0

    async def clear(self, user: AuthenticatedUser) -> int:
        # Messages first, then the sessions (same reason as delete()): page
        # through this user's session ids, delete their messages, then the
        # sessions. Paged so very large histories are cleaned up too.
        session_ids: List[str] = []
        offset = 0
        page = 1000
        while True:
            try:
                ids_resp = await self._db.request(
                    "GET", TABLE, user,
                    params={"select": "id", "user_id": f"eq.{user.id}",
                            "limit": page, "offset": offset},
                )
            except Exception as exc:
                raise _tag(exc, "clear list sessions") from exc
            batch = [str(r["id"]) for r in ids_resp.json()]
            session_ids.extend(batch)
            if len(batch) < page:
                break
            offset += page
        for i in range(0, len(session_ids), 100):
            chunk = ",".join(session_ids[i : i + 100])
            try:
                await self._db.request(
                    "DELETE", MESSAGES_TABLE, user, params={"session_id": f"in.({chunk})"},
                )
            except Exception as exc:
                raise _tag(exc, "clear messages") from exc
        try:
            resp = await self._db.request(
                "DELETE", TABLE, user, params={"user_id": f"eq.{user.id}"}, prefer="return=representation"
            )
        except Exception as exc:
            raise _tag(exc, "clear sessions") from exc
        return len(resp.json())

    async def selfcheck(self, user: AuthenticatedUser) -> dict:
        """Insert a session titled 'selfcheck' and one message, read back, delete both."""
        steps = []
        # 1) Session insert
        try:
            session_resp = await self._db.request(
                "POST", TABLE, user,
                json={"user_id": user.id, "direction": "sign_to_text", "status": "completed", "title": "selfcheck"},
                prefer="return=representation",
            )
            session_id = str(session_resp.json()[0]["id"])
            steps.append({"step": "session_insert", "status": session_resp.status_code, "code": None})
        except Exception as exc:
            steps.append({"step": "session_insert", "status": getattr(exc, "status", None), "code": getattr(exc, "pg_code", None)})
            return {"ok": False, "steps": steps}

        # 2) Message insert
        try:
            msg_resp = await self._db.request(
                "POST", MESSAGES_TABLE, user,
                json={"session_id": session_id, "sequence": 1, "direction": "sign_to_text", "input_text": "selfcheck", "output_text": "selfcheck", "gloss": "selfcheck", "confidence": 1.0, "model_version": "selfcheck", "result_type": "model"},
                prefer="return=representation",
            )
            steps.append({"step": "message_insert", "status": msg_resp.status_code, "code": None})
        except Exception as exc:
            steps.append({"step": "message_insert", "status": getattr(exc, "status", None), "code": getattr(exc, "pg_code", None)})
            try:
                await self.delete(user, session_id)
            except Exception:
                pass
            return {"ok": False, "steps": steps}

        # 3) Read back
        try:
            read_resp = await self._db.request(
                "GET", TABLE, user, params={"id": f"eq.{session_id}", "select": "id,title"}
            )
            steps.append({"step": "read_back", "status": read_resp.status_code, "code": None})
        except Exception as exc:
            steps.append({"step": "read_back", "status": getattr(exc, "status", None), "code": getattr(exc, "pg_code", None)})
            try:
                await self.delete(user, session_id)
            except Exception:
                pass
            return {"ok": False, "steps": steps}

        # 4) Delete both
        try:
            await self.delete(user, session_id)
            steps.append({"step": "delete", "status": 200, "code": None})
        except Exception as exc:
            steps.append({"step": "delete", "status": getattr(exc, "status", None), "code": getattr(exc, "pg_code", None)})
            return {"ok": False, "steps": steps}

        return {"ok": True, "steps": steps}


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

    async def selfcheck(self, user: AuthenticatedUser) -> dict:
        return {
            "ok": True,
            "steps": [
                {"step": "session_insert", "status": 201, "code": None},
                {"step": "message_insert", "status": 201, "code": None},
                {"step": "read_back", "status": 200, "code": None},
                {"step": "delete", "status": 200, "code": None},
            ],
        }


_memory_repository: Optional[InMemoryTranslationRepository] = None


def get_memory_repository() -> InMemoryTranslationRepository:
    global _memory_repository
    if _memory_repository is None:
        _memory_repository = InMemoryTranslationRepository()
    return _memory_repository


def reset_memory_repository() -> None:
    global _memory_repository
    _memory_repository = None
