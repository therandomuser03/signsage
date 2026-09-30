from datetime import datetime, timezone

from app.domain.translation.entities import Direction, ResultType, TranslationResult
from app.domain.translation.policies import classify_result, is_presentable, should_persist
from app.domain.translation.value_objects import ConfidenceProxy
from app.domain.users.entities import User
import pytest


def _result(text="hello", accepted=True, direction=Direction.SIGN_TO_TEXT):
    return TranslationResult(direction, text, accepted, "ok", 0.5, 0.1, datetime.now(timezone.utc))


def test_accepted_sign_result_is_a_model_result():
    assert classify_result(_result()) == ResultType.MODEL


def test_accepted_text_to_sign_result_is_avatar_gloss():
    assert classify_result(_result(direction=Direction.TEXT_TO_SIGN)) == ResultType.AVATAR_GLOSS


def test_rejected_or_blank_results_are_failed_and_not_persisted():
    for r in (_result(accepted=False), _result(text="   ")):
        assert classify_result(r) == ResultType.FAILED
        assert not is_presentable(r) and not should_persist(r)


def test_confidence_proxy_rejects_out_of_range():
    with pytest.raises(ValueError):
        ConfidenceProxy(1.5)


def test_user_entity_is_immutable():
    with pytest.raises(Exception):
        User(id="u1").id = "u2"
