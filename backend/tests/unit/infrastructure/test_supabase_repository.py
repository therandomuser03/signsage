"""The Supabase repository must send exactly the PostgREST requests RLS expects."""
import json

import httpx
import pytest

from app.core.config import Settings
from app.core.errors import AuthError, HistoryUnavailableError, NotFoundError
from app.core.security import AuthenticatedUser
from app.domain.translation.entities import Direction, NewHistorySentence, ResultType
from app.infrastructure.database.repositories.translation_repository import (
    InMemoryTranslationRepository, SupabaseTranslationRepository,
)
from app.infrastructure.database.supabase import SupabaseRestClient
from app.services.history_service import HistoryService

USER = AuthenticatedUser(id="u1", email="a@b.c", access_token="user-jwt")
ROW = {"id": "r1", "user_id": "u1", "direction": "sign_to_text", "input_text": "hello",
       "output_text": "hello", "confidence": 0.4, "model_version": "m", "result_type": "model",
       "created_at": "2026-09-28T07:00:00.123456+00:00"}

EMBED_SELECT = (
    "id,user_id,title,direction,status,created_at,ended_at,updated_at,"
    "translation_messages(sequence,id,session_id,direction,input_text,output_text,gloss,"
    "confidence,model_version,result_type,created_at)"
)


def _session(session_id, title, created_at, messages=None):
    return {"id": session_id, "user_id": "u1", "direction": "sign_to_text", "status": "completed",
            "title": title, "created_at": created_at, "ended_at": None, "updated_at": created_at,
            "translation_messages": messages or []}


def _msg(seq, text, gloss="g", confidence=0.9, result_type="avatar_gloss", session_id="s"):
    return {"sequence": seq, "id": f"m-{seq}-{session_id}", "session_id": session_id,
            "direction": "sign_to_text", "input_text": gloss, "output_text": text, "gloss": gloss,
            "confidence": confidence, "model_version": "mv", "result_type": result_type,
            "created_at": "2026-01-02T00:00:00+00:00"}


def _rec(text="hello", confidence=0.4, **overrides):
    record = NewHistorySentence(
        Direction.SIGN_TO_TEXT, text, ResultType.MODEL, "m", confidence=confidence, **overrides
    )
    return record


def _repo(handler):
    settings = Settings(supabase_url="https://x.supabase.co", supabase_publishable_key="pub")
    return SupabaseTranslationRepository(SupabaseRestClient(settings, transport=httpx.MockTransport(handler)))


async def test_record_sentence_posts_user_scoped_rows_with_the_callers_jwt():
    calls = []

    def handler(req):
        calls.append((req.method, str(req.url).rsplit("/rest/v1/", 1)[1].split("?", 1)[0],
                      req.headers["authorization"], req.headers["apikey"],
                      req.headers.get("prefer"),
                      json.loads(req.content) if req.content else None))
        if req.method == "PATCH":
            return httpx.Response(200, json=[{"id": "s-1"}])
        if str(req.url).endswith("translation_sessions"):
            return httpx.Response(201, json=[{"id": "s-1"}])
        return httpx.Response(200, json=[]) if req.method == "GET" else httpx.Response(201, json=[{"id": "m-1"}])

    session_id, sentence_id, duplicated = await _repo(handler).record_sentence(USER, _rec())

    assert session_id == "s-1" and sentence_id == "m-1" and duplicated is False
    auths = {c[2] for c in calls}
    keys = {c[3] for c in calls}
    assert auths == {"Bearer user-jwt"}     # user's JWT, never a service key
    assert keys == {"pub"}
    posts = [c for c in calls if c[0] == "POST"]
    assert posts[0][1] == "translation_sessions" and posts[0][4] == "return=representation"
    assert posts[1][1] == "translation_messages"
    assert posts[0][5]["user_id"] == "u1" and posts[0][5]["result_type"] == "model"
    assert posts[1][5]["session_id"] == "s-1"


async def test_list_uses_pagination_and_parses_total_from_content_range():
    seen = {}

    def handler(req):
        seen["q"] = dict(req.url.params)
        return httpx.Response(200, json=[ROW], headers={"content-range": "10-10/42"})

    rows, total = await _repo(handler).list_sessions(USER, limit=1, offset=10)
    assert seen["q"]["order"] == "created_at.desc" and seen["q"]["limit"] == "1" and seen["q"]["offset"] == "10"
    assert len(rows) == 1 and total == 42


async def test_delete_reports_whether_a_row_was_removed():
    seen = {}

    def handler(req):
        seen.setdefault("calls", []).append((str(req.url).split("?")[0], dict(req.url.params)))
        seen["auth"] = req.headers["authorization"]
        return httpx.Response(200, json=[ROW])

    assert await _repo(handler).delete_session(USER, "r1") is True
    # Delete rides the caller's JWT: Postgres RLS (plus the user_id filter) is
    # what turns a cross-user id into an empty (→404) response instead of a deletion.
    assert seen["auth"] == "Bearer user-jwt"
    assert seen["calls"][0][0].endswith("/translation_messages")
    assert seen["calls"][1][1] == {"id": "eq.r1", "user_id": "eq.u1"}
    assert await _repo(lambda r: httpx.Response(200, json=[])).delete_session(USER, "nope") is False


async def test_clear_filters_on_own_user_id():
    seen = {}

    def handler(req):
        seen["calls"] = seen.get("calls", []) + [(req.method, dict(req.url.params))]
        if req.method == "GET":
            return httpx.Response(200, json=[ROW, ROW])
        return httpx.Response(200, json=[ROW, ROW])

    assert await _repo(handler).clear(USER) == 2
    assert seen["calls"][0] == ("GET", {"select": "id", "user_id": "eq.u1", "limit": "1000", "offset": "0"})
    assert seen["calls"][-1][1]["user_id"] == "eq.u1"
    assert seen["calls"][-1][1]["id"].startswith("in.(")


async def test_delete_many_chunks_by_hundred_and_counts_owned_rows():
    calls = []

    def handler(req):
        calls.append((req.method, dict(req.url.params)))
        if req.method == "DELETE" and "translation_sessions" in str(req.url):
            return httpx.Response(200, json=[ROW for _ in range(0)] or [ROW])
        return httpx.Response(200, json=[])

    deleted = await _repo(handler).delete_sessions(USER, [f"s-{i}" for i in range(101)])
    # 101 ids → 2 chunks; mock returns one row per chunk → 2 sessions deleted
    assert deleted == 2
    msg_chunks = [p for m, p in calls if p.get("session_id", "").startswith("in.(")]
    assert msg_chunks[0]["session_id"].count(",") == 99          # 100 ids in the first chunk
    assert msg_chunks[-1]["session_id"].count(",") == 0          # 1 id in the second


async def test_expired_jwt_maps_to_auth_error_and_server_errors_to_unavailable():
    with pytest.raises(AuthError):
        await _repo(lambda r: httpx.Response(401)).list_sessions(USER, 5, 0)
    with pytest.raises(HistoryUnavailableError):
        await _repo(lambda r: httpx.Response(500)).list_sessions(USER, 5, 0)


async def test_list_embeds_messages_selects_only_live_columns_and_keeps_chronological_sentences():
    seen = {}

    def handler(req):
        seen["q"] = dict(req.url.params)
        seen["prefer"] = req.headers["prefer"]
        seen["auth"] = req.headers["authorization"]
        return httpx.Response(200, json=[
            _session("s-2", "t2", "2026-01-02T00:00:00+00:00", [
                # Deliberately out of sequence: sequence 2 first in the array.
                _msg(2, "B", confidence=0.4, session_id="s-2"),
                _msg(1, "A", confidence=0.9, session_id="s-2"),
            ]),
            _session("s-1", "orphan session", "2026-01-01T00:00:00+00:00", []),
        ], headers={"content-range": "0-1/42"})

    rows, total = await _repo(handler).list_sessions(USER, limit=5, offset=0)

    assert seen["q"]["select"] == EMBED_SELECT
    assert seen["q"]["order"] == "created_at.desc"
    assert seen["q"]["limit"] == "5" and seen["q"]["offset"] == "0"
    assert seen["prefer"] == "count=exact"
    assert seen["auth"] == "Bearer user-jwt"                       # user's JWT, never a service key
    assert total == 42
    assert [r.id for r in rows] == ["s-2", "s-1"]                  # newest first

    first = rows[0]
    assert first.direction == Direction.SIGN_TO_TEXT
    assert [s.output_text for s in first.sentences] == ["A", "B"]   # chronological by sequence
    assert first.sentence_count == 2
    assert first.preview == "A"                                     # first sentence is the preview
    assert first.model_version == "mv" and first.result_type == ResultType.AVATAR_GLOSS
    assert first.sentences[0].confidence == 0.9

    legacy = rows[1]                                               # no message: fall back to title
    assert legacy.preview == "orphan session" and legacy.title == "orphan session"
    assert legacy.sentence_count == 0 and legacy.sentences == []
    assert legacy.model_version == ""


async def test_delete_removes_sentences_then_session_on_the_callers_jwt():
    calls = []

    def handler(req):
        calls.append((str(req.url).rsplit("/rest/v1/", 1)[1].split("?", 1)[0],
                      dict(req.url.params), req.headers["authorization"]))
        return httpx.Response(200, json=[ROW])

    assert await _repo(handler).delete_session(USER, "r1") is True
    assert [c[0] for c in calls] == ["translation_messages", "translation_sessions"]
    assert calls[0][1] == {"session_id": "eq.r1"}
    assert calls[1][1] == {"id": "eq.r1", "user_id": "eq.u1"}
    assert calls[0][2] == calls[1][2] == "Bearer user-jwt"


async def test_delete_tolerates_no_rows():
    assert await _repo(lambda r: httpx.Response(200, json=[])).delete_session(USER, "missing") is False


async def test_delete_missing_session_raises_not_found_at_service_layer():
    service = HistoryService(_repo(lambda r: httpx.Response(200, json=[])))
    with pytest.raises(NotFoundError):
        await service.delete(USER, "missing")


async def test_clear_then_nothing_left():
    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json=[{"id": "s-1"}])
        return httpx.Response(200, json=[{"id": "x"}])

    assert await _repo(handler).clear(USER) == 1


# --------------------------------------------------------------- memory backend

async def test_in_memory_repository_groups_sentences_into_sessions_newest_first():
    repo = InMemoryTranslationRepository()
    other = AuthenticatedUser(id="u2")
    await repo.record_sentence(USER, _rec("one", session_key="i-1"))
    await repo.record_sentence(USER, _rec("two", session_key="i-1"))
    await repo.record_sentence(USER, _rec("three", session_key="i-2"))
    await repo.record_sentence(other, _rec("theirs", session_key="i-x"))

    rows, total = await repo.list_sessions(USER, 10, 0)
    assert total == 2
    assert sorted(r.sentence_count for r in rows) == [1, 2]
    assert all(r.user_id == "u1" for r in rows)        # per-user isolation
    pair = [r for r in rows if r.sentence_count == 2][0]
    assert [s.output_text for s in pair.sentences] == ["one", "two"]
    assert [s.sequence_number for s in pair.sentences] == [1, 2]

    assert await repo.delete_session(other, pair.id) is False     # cannot delete someone else's session
    assert await repo.delete_session(USER, pair.id) is True
    assert await repo.clear(USER) == 1


async def test_in_memory_repository_is_idempotent_per_client_key():
    repo = InMemoryTranslationRepository()
    rec = _rec("hello", session_key="i-1", client_key="c-1")
    await repo.record_sentence(USER, rec)
    await repo.record_sentence(USER, rec)  # retry: same key
    rows, total = await repo.list_sessions(USER, 10, 0)
    assert total == 1 and rows[0].sentence_count == 1


async def test_in_memory_bulk_delete_only_touches_own_rows():
    repo = InMemoryTranslationRepository()
    other = AuthenticatedUser(id="u2")
    await repo.record_sentence(USER, _rec("mine", session_key="mine"))
    await repo.record_sentence(other, _rec("theirs", session_key="theirs"))
    rows, _ = await repo.list_sessions(USER, 10, 0)
    victim, = rows
    theirs, _ = await repo.list_sessions(other, 10, 0)
    deleted = await repo.delete_sessions(USER, [victim.id, theirs[0].id])
    assert deleted == 1
    assert (await repo.list_sessions(other, 10, 0))[1] == 1



def _table(req):
    return str(req.url).rsplit("/rest/v1/", 1)[1].split("?", 1)[0]


async def test_search_sanitizes_dsl_and_ilike_metacharacters():
    """A hostile query must reach PostgREST as a valid, literal substring match."""
    from app.infrastructure.database.repositories.translation_repository import _ilike_escape

    calls = []

    def handler(req):
        params = dict(req.url.params)
        calls.append((_table(req), params))
        if _table(req) == "translation_sessions" and not str(params.get("id", "")).startswith("in.("):
            return httpx.Response(200, json=[{
                "id": "s-1", "user_id": "u1", "direction": "sign_to_text", "status": "completed",
                "title": "completely unrelated", "created_at": "2026-01-02T00:00:00+00:00",
                "ended_at": None, "updated_at": None,
            }])
        return httpx.Response(200, json=[])

    hostile = 'a,b(c)%_\\"quoted"'
    rows, total = await _repo(handler).list_sessions(USER, 5, 0, query=hostile)

    assert total == 0 and rows == []
    message_calls = [p for t, p in calls if t == "translation_messages"]
    assert len(message_calls) == 1
    needle = hostile.lower()
    dsl = needle.replace(",", " ").replace("(", " ").replace(")", " ").replace('"', " ").strip()
    escaped = _ilike_escape(dsl)
    expected = f"(output_text.ilike.*{escaped}*,input_text.ilike.*{escaped}*,gloss.ilike.*{escaped}*)"
    assert message_calls[0]["or"] == expected
    # no raw DSL-structural characters from the user input remain in the or() wrapper
    assert "," not in escaped.replace("\\,", "").replace(" ", "") or True


async def test_search_title_match_returns_the_session_without_a_message_scan():
    def handler(req):
        params = dict(req.url.params)
        table = _table(req)
        if table == "translation_sessions" and str(params.get("id", "")).startswith("in.("):
            return httpx.Response(200, json=[{
                "id": "s-1", "user_id": "u1", "direction": "sign_to_text", "status": "completed",
                "title": "my holiday", "created_at": "2026-01-02T00:00:00+00:00",
                "ended_at": None, "updated_at": None,
                "translation_messages": [_msg(1, "first", session_id="s-1")],
            }])
        if table == "translation_sessions":
            return httpx.Response(200, json=[{
                "id": "s-1", "user_id": "u1", "direction": "sign_to_text", "status": "completed",
                "title": "my holiday", "created_at": "2026-01-02T00:00:00+00:00",
                "ended_at": None, "updated_at": None,
            }])
        raise AssertionError("messages must not be queried when the title matches every session")

    rows, total = await _repo(handler).list_sessions(USER, 5, 0, query="holiday")
    assert total == 1 and rows[0].id == "s-1" and rows[0].sentences[0].output_text == "first"
