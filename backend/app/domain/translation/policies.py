"""Business-level policies for translation results."""
from __future__ import annotations

from app.domain.translation.entities import Direction, ResultType, TranslationResult


def is_presentable(result: TranslationResult) -> bool:
    """Accepted by the ML gate with non-empty text."""
    return result.accepted and bool(result.text.strip())


def classify_result(result: TranslationResult) -> ResultType:
    """Label a result so the UI can keep model output, avatar output and
    failures visually distinct (never present a failure as a translation)."""
    if not is_presentable(result):
        return ResultType.FAILED
    return ResultType.MODEL if result.direction == Direction.SIGN_TO_TEXT else ResultType.AVATAR_GLOSS


def should_persist(result: TranslationResult) -> bool:
    """Only presentable results are worth keeping in a user's history."""
    return is_presentable(result)
