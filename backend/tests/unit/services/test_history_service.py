import pytest

from app.core.config import Settings
from app.core.errors import NotFoundError
from app.core.security import AuthenticatedUser
from app.domain.translation.entities import Direction, NewHistorySentence, ResultType
from app.infrastructure.database.repositories.translation_repository import InMemoryTranslationRepository
from app.services.history_service import HistoryService, build_history_service

USER = AuthenticatedUser(id="u1")
REC = NewHistorySentence(Direction.SIGN_TO_TEXT, "hi", ResultType.MODEL, "m")


def _service():
    return HistoryService(InMemoryTranslationRepository())


async def test_try_record_sentence_returns_session_id_on_success():
    session_id = await _service().try_record_sentence(USER, REC)
    assert session_id


async def test_try_record_sentence_swallows_repository_failures():
    class Broken(InMemoryTranslationRepository):
        async def record_sentence(self, user, record):
            raise RuntimeError("db down")

    assert await HistoryService(Broken()).try_record_sentence(USER, REC) is None


async def test_try_record_sentence_checked_reports_compact_error_on_failure():
    class Broken(InMemoryTranslationRepository):
        async def record_sentence(self, user, record):
            err = RuntimeError("db down")
            err.step = "sentence insert"  # type: ignore[attr-defined]
            err.status = 403  # type: ignore[attr-defined]
            err.pg_code = "42501"  # type: ignore[attr-defined]
            raise err

    saved, error = await HistoryService(Broken()).try_record_sentence_checked(USER, REC)
    assert saved is False
    assert error == "sentence_insert:403:42501"


async def test_record_sentence_is_idempotent_and_scopes_the_interaction():
    service = _service()
    rec = NewHistorySentence(
        Direction.SIGN_TO_TEXT, "hello", ResultType.MODEL, "m",
        session_key="i-1", client_key="c-1",
    )
    first, _ = await service.record_sentence(USER, rec)
    second, _ = await service.record_sentence(USER, rec)
    assert first == second
    sessions, total = await service.list(USER, 10, 0)
    assert total == 1 and sessions[0].sentence_count == 1


async def test_delete_missing_entry_is_not_found():
    with pytest.raises(NotFoundError):
        await _service().delete(USER, "missing")


async def test_delete_many_reports_only_the_callers_sessions():
    service = _service()
    other = AuthenticatedUser(id="u2")
    await service.record_sentence(USER, NewHistorySentence(
        Direction.SIGN_TO_TEXT, "mine", ResultType.MODEL, "m", session_key="mine"))
    await service.record_sentence(other, NewHistorySentence(
        Direction.SIGN_TO_TEXT, "theirs", ResultType.MODEL, "m", session_key="theirs"))
    mine, _ = await service.list(USER, 10, 0)
    theirs, _ = await service.list(other, 10, 0)

    deleted = await service.delete_many(USER, [mine[0].id, theirs[0].id])
    assert deleted == 1
    assert (await service.list(other, 10, 0))[1] == 1


async def test_close_session_is_best_effort():
    service = _service()
    await service.close_session(USER, "nonexistent-key")  # must never raise
    await service.record_sentence(USER, NewHistorySentence(
        Direction.SIGN_TO_TEXT, "hi", ResultType.MODEL, "m", session_key="i-9", status="open"))
    await service.close_session(USER, "i-9")
    sessions, _ = await service.list(USER, 10, 0)
    assert sessions[0].status == "completed" and sessions[0].ended_at is not None


def test_build_history_service_selection():
    assert build_history_service(Settings(history_enabled=False)) is None
    assert build_history_service(Settings(supabase_url=None, supabase_publishable_key=None)) is None
    assert build_history_service(Settings(history_backend="memory")) is not None
    assert build_history_service(Settings(supabase_url="https://x.supabase.co", supabase_publishable_key="k")) is not None
