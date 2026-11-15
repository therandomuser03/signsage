"""History persistence: Supabase (PostgREST + RLS) and in-memory dev/test backends.

Session-based model (see migrations/002_history_sessions.sql):
  translation_sessions  — one row per translation interaction
  translation_messages  — one row per completed sentence inside a session

Ownership is enforced twice: every query carries the caller's user JWT (Postgres
RLS keeps rows to their owner) and list/delete paths additionally filter on
user_id. A service-role key is never used.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Protocol, Tuple

import httpx  # noqa: F401  (kept for parity with previous imports)

from app.core.config import Settings
from app.core.errors import HistoryUnavailableError, NotFoundError
from app.core.security import AuthenticatedUser
from app.domain.translation.entities import (
    Direction,
    NewHistorySentence,
    ResultType,
    SentenceRecord,
    SessionRecord,
)
from app.infrastructure.database.supabase import SupabaseRestClient

logger = logging.getLogger(__name__)

TABLE = "translation_sessions"
MESSAGES_TABLE = "translation_messages"

# Sessions own the translatable text; the live schema keeps it in
# translation_messages, so list() embeds that child table instead of selecting
# nonexistent columns on translation_sessions.
EMBED_SELECT = (
    "id,user_id,title,direction,status,created_at,ended_at,updated_at,"
    "translation_messages(sequence,id,session_id,direction,input_text,output_text,gloss,"
    "confidence,model_version,result_type,created_at)"
)
# Used when embedding is unavailable (hand-built live schema without the FK):
# first page of sessions, then their messages, joined client-side.
SESSION_COLS = "id,user_id,title,direction,status,created_at,ended_at,updated_at"
MESSAGE_COLS = (
    "session_id,sequence,id,direction,input_text,output_text,gloss,"
    "confidence,model_version,result_type,created_at"
)


class TranslationRepository(Protocol):
    async def record_sentence(
        self, user: AuthenticatedUser, record: NewHistorySentence
    ) -> Tuple[str, str, bool]: """Insert (session, sentence); returns (session_id, sentence_id, duplicated)."""
    async def list_sessions(
        self,
        user: AuthenticatedUser,
        limit: int,
        offset: int,
        direction: Optional[str] = None,
        query: Optional[str] = None,
        order: str = "newest",
    ) -> Tuple[List[SessionRecord], int]: ...
    async def get_session(self, user: AuthenticatedUser, session_id: str) -> Optional[SessionRecord]: ...
    async def delete_session(self, user: AuthenticatedUser, session_id: str) -> bool: ...
    async def delete_sessions(self, user: AuthenticatedUser, ids: List[str]) -> int: ...
    async def clear(self, user: AuthenticatedUser) -> int: ...
    async def close_session(self, user: AuthenticatedUser, session_key: str) -> bool: ...
    async def selfcheck(self, user: AuthenticatedUser) -> dict: ...


def _tag(exc: Exception, step: str) -> Exception:
    """Attach the failing step so the service layer can log it precisely."""
    try:
        exc.step = step  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 -- tagging is best-effort; logging still works untagged
        pass
    return exc


def _is_relationship_error(exc: HistoryUnavailableError) -> bool:
    """PostgREST relationship-resolution failures (the embed's FK is gone)."""
    return (
        (exc.status or 0) == 400
        and bool(exc.pg_code)
        and str(exc.pg_code).upper().startswith("PGRST2")
    )


def _is_unique_violation(exc: HistoryUnavailableError) -> bool:
    return (exc.status == 409) or ((exc.pg_code or "").upper() == "23505")


def _total_from(resp, offset: int, fallback_page_size: int) -> int:
    """Content-Range looks like "0-9/42" (or "*/0" when empty)."""
    total = fallback_page_size + offset
    content_range = resp.headers.get("content-range", "")
    if "/" in content_range and content_range.rsplit("/", 1)[1].isdigit():
        total = int(content_range.rsplit("/", 1)[1])
    return total


def _parse_dt(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _message_to_record(row: dict, session_id: str) -> SentenceRecord:
    try:
        result_type = ResultType(row.get("result_type") or "model")
    except ValueError:
        result_type = ResultType.MODEL
    return SentenceRecord(
        id=str(row.get("id") or ""),
        session_id=str(row.get("session_id") or session_id),
        sequence_number=int(row.get("sequence") or 1),
        direction=Direction(row.get("direction") or "sign_to_text"),
        input_text=row.get("input_text"),
        output_text=row.get("output_text") or "",
        gloss=row.get("gloss"),
        confidence=row.get("confidence"),
        model_version=row.get("model_version") or "",
        result_type=result_type,
        created_at=_parse_dt(row.get("created_at")) or datetime.now(timezone.utc),
    )


def _session_to_record(row: dict) -> SessionRecord:
    messages = sorted(
        row.get("translation_messages") or [],
        key=lambda m: (m.get("sequence") or 0, m.get("created_at") or ""),
    )
    session_id = str(row["id"])
    sentences = [_message_to_record(m, session_id) for m in messages]
    try:
        direction = Direction(row["direction"])
    except (ValueError, KeyError):
        direction = Direction.SIGN_TO_TEXT
    return SessionRecord(
        id=session_id,
        user_id=str(row.get("user_id") or ""),
        direction=direction,
        title=row.get("title"),
        status=row.get("status") or "completed",
        created_at=_parse_dt(row.get("created_at")) or datetime.now(timezone.utc),
        ended_at=_parse_dt(row.get("ended_at")),
        updated_at=_parse_dt(row.get("updated_at")),
        sentences=sentences,
    )


def _ilike_escape(term: str) -> str:
    """Escape PostgREST ilike metacharacters so user input is matched literally."""
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class SupabaseTranslationRepository:
    def __init__(self, client: SupabaseRestClient):
        self._db = client

    @classmethod
    def from_settings(cls, settings: Settings) -> "SupabaseTranslationRepository":
        return cls(SupabaseRestClient(settings))

    # ------------------------------------------------------------------ writes

    async def _find_session_by_key(self, user: AuthenticatedUser, session_key: str) -> Optional[str]:
        try:
            resp = await self._db.request(
                "GET", TABLE, user,
                params={"select": "id", "user_id": f"eq.{user.id}", "client_key": f"eq.{session_key}", "limit": 1},
            )
        except Exception as exc:
            raise _tag(exc, "session lookup") from exc
        rows = resp.json()
        return str(rows[0]["id"]) if rows else None

    async def _create_session(self, user: AuthenticatedUser, record: NewHistorySentence) -> str:
        body = {
            "user_id": user.id,
            "direction": record.direction.value,
            "status": record.status,
            "title": record.title or f"{record.direction.value.replace('_', ' ').title()}: {record.output_text[:50]}",
            "result_type": record.result_type.value,
            "model_version": record.model_version,
            "input_text": record.input_text,
            "output_text": record.output_text,
            "confidence": record.confidence,
            "client_key": record.session_key,
        }
        try:
            resp = await self._db.request(
                "POST", TABLE, user, json=body, prefer="return=representation"
            )
        except HistoryUnavailableError as exc:
            # Lost a race against an identical get-or-create (e.g. reconnect
            # while the first request was in flight): read the winner's row.
            if _is_unique_violation(exc) and record.session_key:
                existing = await self._find_session_by_key(user, record.session_key)
                if existing:
                    return existing
            raise _tag(exc, "session insert") from exc
        except Exception as exc:
            raise _tag(exc, "session insert") from exc
        return str(resp.json()[0]["id"])

    async def _get_or_create_session(self, user: AuthenticatedUser, record: NewHistorySentence) -> str:
        if record.session_key:
            existing = await self._find_session_by_key(user, record.session_key)
            if existing:
                return existing
        return await self._create_session(user, record)

    async def _find_sentence_by_key(self, user: AuthenticatedUser, session_id: str, client_key: str) -> Optional[str]:
        try:
            resp = await self._db.request(
                "GET", MESSAGES_TABLE, user,
                params={
                    "select": "id", "session_id": f"eq.{session_id}",
                    "client_key": f"eq.{client_key}", "limit": 1,
                },
            )
        except Exception as exc:
            raise _tag(exc, "sentence lookup") from exc
        rows = resp.json()
        return str(rows[0]["id"]) if rows else None

    async def _next_sequence(self, user: AuthenticatedUser, session_id: str) -> int:
        try:
            resp = await self._db.request(
                "GET", MESSAGES_TABLE, user,
                params={"select": "sequence", "session_id": f"eq.{session_id}",
                        "order": "sequence.desc", "limit": 1},
            )
        except Exception as exc:
            raise _tag(exc, "sequence lookup") from exc
        rows = resp.json()
        return (int(rows[0]["sequence"]) + 1) if rows and rows[0].get("sequence") is not None else 1

    async def record_sentence(
        self, user: AuthenticatedUser, record: NewHistorySentence
    ) -> Tuple[str, str, bool]:
        session_id = await self._get_or_create_session(user, record)

        if record.client_key:
            existing = await self._find_sentence_by_key(user, session_id, record.client_key)
            if existing:
                return session_id, existing, True

        sequence = await self._next_sequence(user, session_id)
        message_body = {
            "session_id": session_id,
            "sequence": sequence,
            "direction": record.direction.value,
            "input_text": record.input_text,
            "output_text": record.output_text,
            "gloss": record.gloss,
            "confidence": record.confidence,
            "model_version": record.model_version,
            "result_type": record.result_type.value,
            "client_key": record.client_key,
        }
        try:
            resp = await self._db.request(
                "POST", MESSAGES_TABLE, user, json=message_body, prefer="return=representation"
            )
        except HistoryUnavailableError as exc:
            # Idempotent retry: the row we tried to insert already exists.
            if _is_unique_violation(exc) and record.client_key:
                existing = await self._find_sentence_by_key(user, session_id, record.client_key)
                if existing:
                    return session_id, existing, True
            raise _tag(exc, "sentence insert") from exc
        except Exception as exc:
            raise _tag(exc, "sentence insert") from exc
        sentence_id = str(resp.json()[0]["id"])

        # Keep the session's recency/title metadata fresh. Best-effort.
        try:
            await self._db.request(
                "PATCH", TABLE, user,
                params={"id": f"eq.{session_id}", "user_id": f"eq.{user.id}"},
                json={"updated_at": datetime.now(timezone.utc).isoformat()},
                prefer="return=minimal",
            )
        except Exception:  # noqa: BLE001 -- history must never break translation
            logger.warning("history session touch failed (non-fatal)")
        return session_id, sentence_id, False

    async def close_session(self, user: AuthenticatedUser, session_key: str) -> bool:
        """Mark the interaction session completed (WS disconnect / explicit stop)."""
        try:
            resp = await self._db.request(
                "PATCH", TABLE, user,
                params={"user_id": f"eq.{user.id}", "client_key": f"eq.{session_key}"},
                json={
                    "ended_at": datetime.now(timezone.utc).isoformat(),
                    "status": "completed",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                prefer="return=representation",
            )
        except Exception as exc:
            logger.warning("history session close failed: %s", type(exc).__name__)
            return False
        return bool(resp.json())

    # ------------------------------------------------------------------- reads

    def _base_list_params(self, user: AuthenticatedUser, direction: Optional[str]) -> dict:
        params = {"user_id": f"eq.{user.id}"}
        if direction:
            params["direction"] = f"eq.{direction}"
        return params

    async def list_sessions(
        self,
        user: AuthenticatedUser,
        limit: int,
        offset: int,
        direction: Optional[str] = None,
        query: Optional[str] = None,
        order: str = "newest",
    ) -> Tuple[List[SessionRecord], int]:
        direction_clause = self._base_list_params(user, direction)
        ordering = "created_at.asc" if order == "oldest" else "created_at.desc"
        query = (query or "").strip() or None

        if query:
            # Exact total + page over a union of (title matches, message matches).
            page_ids, total = await self._search_page_ids(user, query, direction, ordering, limit, offset)
            if not page_ids:
                return [], total
            try:
                resp = await self._db.request(
                    "GET", TABLE, user,
                    params={
                        "select": EMBED_SELECT,
                        "user_id": f"eq.{user.id}",
                        "id": f"in.({','.join(page_ids)})",
                        "order": ordering,
                    },
                )
            except Exception as exc:
                raise _tag(exc, "list sessions") from exc
            rows = [_session_to_record(r) for r in resp.json()]
            return rows, total

        try:
            resp = await self._db.request(
                "GET", TABLE, user,
                params={
                    "select": EMBED_SELECT,
                    **direction_clause,
                    "order": ordering,
                    "limit": limit,
                    "offset": offset,
                },
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
            return await self._list_two_requests(user, limit, offset, direction, ordering)
        except Exception as exc:
            raise _tag(exc, "list") from exc
        rows = [_session_to_record(r) for r in resp.json()]
        return rows, _total_from(resp, offset, len(rows))

    async def _search_page_ids(
        self,
        user: AuthenticatedUser,
        query: str,
        direction: Optional[str],
        ordering: str,
        limit: int,
        offset: int,
    ) -> Tuple[List[str], int]:
        """Ids of sessions matching `query` (title or any sentence text), paged.

        Filtering happens in Python over per-user rows — exact count, no URL
        length issues. Histories are per-account and small enough for this.
        """
        try:
            sess_resp = await self._db.request(
                "GET", TABLE, user,
                params={"select": "id,title,direction,created_at", **self._base_list_params(user, direction),
                        "limit": 1000},
            )
        except Exception as exc:
            raise _tag(exc, "search sessions") from exc
        sessions = sess_resp.json()

        needle = query.lower()
        matching = {str(s["id"]) for s in sessions if needle in (s.get("title") or "").lower()}
        # A DSL-safe fragment for the or=(...) clause below: the PostgREST
        # or-grammar is split on commas/parens, so those characters must not
        # reach it. The Python-side title match above is unaffected.
        dsl_needle = (
            needle.replace(",", " ").replace("(", " ").replace(")", " ").replace('"', " ")
        ).strip()

        if len(matching) < len(sessions) and dsl_needle:
            try:
                msg_resp = await self._db.request(
                    "GET", MESSAGES_TABLE, user,
                    params={
                        "select": "session_id,output_text,input_text,gloss",
                        "or": f"(output_text.ilike.*{_ilike_escape(dsl_needle)}*,"
                              f"input_text.ilike.*{_ilike_escape(dsl_needle)}*,"
                              f"gloss.ilike.*{_ilike_escape(dsl_needle)}*)",
                        "limit": 1000,
                    },
                )
            except Exception as exc:
                raise _tag(exc, "search sentences") from exc
            for m in msg_resp.json():
                sid = m.get("session_id")
                if sid:
                    matching.add(str(sid))

        by_id = {str(s["id"]): s for s in sessions}
        ordered_ids = sorted(
            (sid for sid in matching if sid in by_id),
            key=lambda sid: str(by_id[sid].get("created_at") or ""),
            reverse=ordering == "created_at.desc",
        )
        return ordered_ids[offset: offset + limit], len(ordered_ids)

    async def _list_two_requests(
        self,
        user: AuthenticatedUser,
        limit: int,
        offset: int,
        direction: Optional[str],
        ordering: str,
    ) -> Tuple[List[SessionRecord], int]:
        try:
            resp = await self._db.request(
                "GET", TABLE, user,
                params={"select": SESSION_COLS, **self._base_list_params(user, direction),
                        "order": ordering, "limit": limit, "offset": offset},
                prefer="count=exact",
            )
        except Exception as exc:
            raise _tag(exc, "list sessions") from exc
        session_rows = resp.json()
        total = _total_from(resp, offset, len(session_rows))
        if not session_rows:
            return [], total
        ids = ",".join(str(r["id"]) for r in session_rows)
        try:
            msg_resp = await self._db.request(
                "GET", MESSAGES_TABLE, user,
                params={"select": MESSAGE_COLS, "session_id": f"in.({ids})", "order": "sequence.asc"},
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

    async def get_session(self, user: AuthenticatedUser, session_id: str) -> Optional[SessionRecord]:
        try:
            resp = await self._db.request(
                "GET", TABLE, user,
                params={"select": EMBED_SELECT, "id": f"eq.{session_id}",
                        "user_id": f"eq.{user.id}", "limit": 1},
            )
        except HistoryUnavailableError as exc:
            if not _is_relationship_error(exc):
                raise _tag(exc, "get session") from exc
            try:
                sess_resp = await self._db.request(
                    "GET", TABLE, user,
                    params={"select": SESSION_COLS, "id": f"eq.{session_id}",
                            "user_id": f"eq.{user.id}", "limit": 1},
                )
            except Exception as exc2:
                raise _tag(exc2, "get session") from exc2
            rows = sess_resp.json()
            if not rows:
                return None
            session_row = rows[0]
            try:
                msg_resp = await self._db.request(
                    "GET", MESSAGES_TABLE, user,
                    params={"select": MESSAGE_COLS, "session_id": f"eq.{session_id}",
                            "order": "sequence.asc"},
                )
            except Exception as exc2:
                raise _tag(exc2, "get sentences") from exc2
            session_row["translation_messages"] = msg_resp.json()
            return _session_to_record(session_row)
        except Exception as exc:
            raise _tag(exc, "get session") from exc
        rows = resp.json()
        return _session_to_record(rows[0]) if rows else None

    # ------------------------------------------------------------------ deletes

    async def delete_session(self, user: AuthenticatedUser, session_id: str) -> bool:
        # 1) Sentences first: nothing is left behind when the FK does not
        #    cascade. With the cascade in place (migration 002) this simply
        #    deletes zero rows first, then the session.
        try:
            await self._db.request(
                "DELETE", MESSAGES_TABLE, user, params={"session_id": f"eq.{session_id}"},
            )
        except Exception as exc:
            raise _tag(exc, "delete sentences") from exc
        # 2) The session itself (user_id + RLS: never another account's row).
        try:
            resp = await self._db.request(
                "DELETE", TABLE, user,
                params={"id": f"eq.{session_id}", "user_id": f"eq.{user.id}"},
                prefer="return=representation",
            )
        except Exception as exc:
            raise _tag(exc, "delete session") from exc
        return len(resp.json()) > 0

    async def delete_sessions(self, user: AuthenticatedUser, ids: List[str]) -> int:
        """Bulk delete. De-duplicated, chunked; each chunk is one statement.

        With ON DELETE CASCADE (migration 002) deleting the sessions is atomic
        per chunk and takes the sentences with it; the explicit message delete
        keeps the operation safe on pre-migration databases.
        """
        unique_ids = list(dict.fromkeys(str(i) for i in ids))
        if not unique_ids:
            return 0
        deleted = 0
        for i in range(0, len(unique_ids), 100):
            chunk = unique_ids[i: i + 100]
            chunk_str = ",".join(chunk)
            try:
                await self._db.request(
                    "DELETE", MESSAGES_TABLE, user, params={"session_id": f"in.({chunk_str})"},
                )
            except Exception as exc:
                raise _tag(exc, "delete sentences") from exc
            try:
                resp = await self._db.request(
                    "DELETE", TABLE, user,
                    params={"id": f"in.({chunk_str})", "user_id": f"eq.{user.id}"},
                    prefer="return=representation",
                )
            except Exception as exc:
                raise _tag(exc, "delete sessions") from exc
            deleted += len(resp.json())
        return deleted

    async def clear(self, user: AuthenticatedUser) -> int:
        """Delete every session belonging to `user` (and their sentences)."""
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
        return await self.delete_sessions(user, session_ids)

    async def selfcheck(self, user: AuthenticatedUser) -> dict:
        """Record a sample sentence, read the session back, delete it; step-by-step status."""
        steps = []

        async def step(name: str, fn) -> Optional[str]:
            try:
                result = await fn()
            except Exception as exc:
                steps.append({
                    "step": name,
                    "status": getattr(exc, "status", None),
                    "code": getattr(exc, "pg_code", None),
                })
                return None
            steps.append({"step": name, "status": 200, "code": None})
            return result

        # Session + sentence insert
        record = NewHistorySentence(
            Direction.SIGN_TO_TEXT, "selfcheck", ResultType.MODEL, "selfcheck",
            status="completed", title="selfcheck",
        )
        try:
            session_id, sentence_id, _ = await self.record_sentence(user, record)
            steps.append({"step": "session_insert", "status": 201, "code": None})
            steps.append({"step": "sentence_insert", "status": 201, "code": None})
        except Exception as exc:
            fail_step = getattr(exc, "step", None) or "history insert"
            steps.append({
                "step": fail_step,
                "status": getattr(exc, "status", None),
                "code": getattr(exc, "pg_code", None),
            })
            return {"ok": False, "steps": steps}

        # Read back
        detail = await step("read_back", lambda: self.get_session(user, session_id))
        if detail is None:
            try:
                await self.delete_session(user, session_id)
            except Exception:
                pass
            return {"ok": False, "steps": steps}
        if not detail.sentences:
            try:
                await self.delete_session(user, session_id)
            except Exception:
                pass
            return {"ok": False, "steps": steps}

        # Delete both
        ok = await step("delete", lambda: self.delete_session(user, session_id))
        return {"ok": ok is not None, "steps": steps}


class InMemoryTranslationRepository:
    """Dev/test backend. Mirrors the Supabase semantics, including idempotency."""

    def __init__(self) -> None:
        self._sessions: Dict[str, dict] = {}       # session_id -> session dict
        self._messages: Dict[str, List[dict]] = {}  # session_id -> message dicts

    # ------------------------------------------------------------------ writes

    async def record_sentence(
        self, user: AuthenticatedUser, record: NewHistorySentence
    ) -> Tuple[str, str, bool]:
        now = datetime.now(timezone.utc)
        # get-or-create the interaction session
        session_id: Optional[str] = None
        if record.session_key:
            for sid, sess in self._sessions.items():
                if sess["user_id"] == user.id and sess.get("client_key") == record.session_key:
                    session_id = sid
                    break
        if session_id is None:
            session_id = str(uuid.uuid4())
            self._sessions[session_id] = {
                "id": session_id,
                "user_id": user.id,
                "direction": record.direction.value,
                "status": record.status,
                "title": record.title or f"{record.direction.value.replace('_', ' ').title()}: {record.output_text[:50]}",
                "created_at": now.isoformat(),
                "ended_at": None,
                "updated_at": now.isoformat(),
                "client_key": record.session_key,
            }
            self._messages[session_id] = []
        # idempotency: same sentence key twice collapses
        if record.client_key:
            for existing in self._messages[session_id]:
                if existing.get("client_key") == record.client_key:
                    return session_id, existing["id"], True
        seq = max((m["sequence"] for m in self._messages[session_id]), default=0) + 1
        message_id = str(uuid.uuid4())
        self._messages[session_id].append({
            "id": message_id,
            "session_id": session_id,
            "sequence": seq,
            "direction": record.direction.value,
            "input_text": record.input_text,
            "output_text": record.output_text,
            "gloss": record.gloss,
            "confidence": record.confidence,
            "model_version": record.model_version,
            "result_type": record.result_type.value,
            "client_key": record.client_key,
            "created_at": now.isoformat(),
        })
        self._sessions[session_id]["updated_at"] = now.isoformat()
        return session_id, message_id, False

    async def close_session(self, user: AuthenticatedUser, session_key: str) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        for sess in self._sessions.values():
            if sess["user_id"] == user.id and sess.get("client_key") == session_key:
                sess["ended_at"] = now
                sess["status"] = "completed"
                sess["updated_at"] = now
                return True
        return False

    # ------------------------------------------------------------------- reads

    def _session_rows(
        self,
        user: AuthenticatedUser,
        direction: Optional[str],
        query: Optional[str],
        order: str,
    ) -> List[Tuple[dict, List[dict]]]:
        rows: List[Tuple[dict, List[dict]]] = []
        for sid, sess in self._sessions.items():
            if sess["user_id"] != user.id:
                continue
            if direction and sess["direction"] != direction:
                continue
            messages = list(self._messages.get(sid, []))
            if query:
                needle = query.lower()
                hay = " ".join(
                    [sess.get("title") or ""] + [
                        f"{m.get('output_text') or ''} {m.get('input_text') or ''} {m.get('gloss') or ''}"
                        for m in messages
                    ]
                ).lower()
                if needle not in hay:
                    continue
            rows.append((sess, messages))
        rows.sort(key=lambda item: str(item[0]["created_at"]), reverse=(order != "oldest"))
        return rows

    def _to_record(self, sess: dict, messages: List[dict]) -> SessionRecord:
        row = dict(sess)
        messages = sorted(messages, key=lambda m: (m.get("sequence") or 0, m.get("created_at") or ""))
        row["translation_messages"] = messages
        return _session_to_record(row)

    async def list_sessions(
        self,
        user: AuthenticatedUser,
        limit: int,
        offset: int,
        direction: Optional[str] = None,
        query: Optional[str] = None,
        order: str = "newest",
    ) -> Tuple[List[SessionRecord], int]:
        query = (query or "").strip() or None
        rows = self._session_rows(user, direction, query, order)
        total = len(rows)
        page = rows[offset: offset + limit]
        return [self._to_record(sess, msgs) for sess, msgs in page], total

    async def get_session(self, user: AuthenticatedUser, session_id: str) -> Optional[SessionRecord]:
        sess = self._sessions.get(session_id)
        if not sess or sess["user_id"] != user.id:
            return None
        return self._to_record(sess, self._messages.get(session_id, []))

    # ------------------------------------------------------------------ deletes

    async def delete_session(self, user: AuthenticatedUser, session_id: str) -> bool:
        sess = self._sessions.get(session_id)
        if not sess or sess["user_id"] != user.id:
            return False
        del self._sessions[session_id]
        self._messages.pop(session_id, None)
        return True

    async def delete_sessions(self, user: AuthenticatedUser, ids: List[str]) -> int:
        deleted = 0
        for sid in dict.fromkeys(str(i) for i in ids):
            if await self.delete_session(user, sid):
                deleted += 1
        return deleted

    async def clear(self, user: AuthenticatedUser) -> int:
        mine = [sid for sid, sess in self._sessions.items() if sess["user_id"] == user.id]
        return await self.delete_sessions(user, mine)

    async def selfcheck(self, user: AuthenticatedUser) -> dict:
        steps = []
        record = NewHistorySentence(
            Direction.SIGN_TO_TEXT, "selfcheck", ResultType.MODEL, "selfcheck",
            status="completed", title="selfcheck",
        )
        try:
            session_id, _, _ = await self.record_sentence(user, record)
            steps.append({"step": "session_insert", "status": 201, "code": None})
            steps.append({"step": "sentence_insert", "status": 201, "code": None})
        except Exception as exc:
            steps.append({
                "step": getattr(exc, "step", "history insert"),
                "status": getattr(exc, "status", None),
                "code": getattr(exc, "pg_code", None),
            })
            return {"ok": False, "steps": steps}
        try:
            detail = await self.get_session(user, session_id)
            steps.append({"step": "read_back", "status": 200, "code": None})
            if detail is None or not detail.sentences:
                steps[-1]["code"] = "EMPTY_SESSION"
                raise RuntimeError("selfcheck read_back failed")
        except Exception:
            try:
                await self.delete_session(user, session_id)
            except Exception:
                pass
            return {"ok": False, "steps": steps}
        try:
            await self.delete_session(user, session_id)
            steps.append({"step": "delete", "status": 200, "code": None})
        except Exception:
            return {"ok": False, "steps": steps}
        return {"ok": True, "steps": steps}


_memory_repository: Optional[InMemoryTranslationRepository] = None


def get_memory_repository() -> InMemoryTranslationRepository:
    global _memory_repository
    if _memory_repository is None:
        _memory_repository = InMemoryTranslationRepository()
    return _memory_repository


def reset_memory_repository() -> None:
    global _memory_repository
    _memory_repository = None
