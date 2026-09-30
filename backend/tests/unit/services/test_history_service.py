import pytest

from app.core.config import Settings
from app.core.errors import NotFoundError
from app.core.security import AuthenticatedUser
from app.domain.translation.entities import Direction, NewHistoryRecord, ResultType
from app.infrastructure.database.repositories.translation_repository import InMemoryTranslationRepository
from app.services.history_service import HistoryService, build_history_service

USER = AuthenticatedUser(id="u1")
REC = NewHistoryRecord(Direction.SIGN_TO_TEXT, "hi", ResultType.MODEL, "m")


async def test_try_record_swallows_repository_failures():
    class Broken(InMemoryTranslationRepository):
        async def add(self, user, record):
            raise RuntimeError("db down")
    assert await HistoryService(Broken()).try_record(USER, REC) is None


async def test_try_record_returns_id_on_success():
    assert await HistoryService(InMemoryTranslationRepository()).try_record(USER, REC)


async def test_delete_missing_entry_is_not_found():
    with pytest.raises(NotFoundError):
        await HistoryService(InMemoryTranslationRepository()).delete(USER, "missing")


def test_build_history_service_selection():
    assert build_history_service(Settings(history_enabled=False)) is None
    assert build_history_service(Settings(supabase_url=None, supabase_publishable_key=None)) is None
    assert build_history_service(Settings(history_backend="memory")) is not None
    assert build_history_service(Settings(supabase_url="https://x.supabase.co", supabase_publishable_key="k")) is not None
