"""Translation domain entities."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class Direction(str, Enum):
    SIGN_TO_TEXT = "sign_to_text"
    TEXT_TO_SIGN = "text_to_sign"


class ResultType(str, Enum):
    """What kind of result this is -- the UI must never present them alike.

    MODEL:        produced by the trained Sign->Text model (only served once promoted)
    AVATAR_GLOSS: rule-based English->gloss lookup driving pre-authored avatar clips
    FAILED:       nothing usable (no motion, empty output, no matching signs)
    """
    MODEL = "model"
    AVATAR_GLOSS = "avatar_gloss"
    FAILED = "failed"


@dataclass
class TranslationResult:
    direction: Direction
    text: str
    accepted: bool
    reason: str
    confidence: float
    motion_energy: float
    created_at: datetime


@dataclass
class NewHistoryRecord:
    direction: Direction
    output_text: str
    result_type: ResultType
    model_version: str
    input_text: Optional[str] = None
    confidence: Optional[float] = None
    # Word-recognition session/message fields (optional; existing flows leave them unset).
    gloss: Optional[str] = None
    sequence: Optional[int] = None
    title: Optional[str] = None
    status: Optional[str] = None


@dataclass
class HistoryRecord:
    id: str
    user_id: str
    direction: Direction
    output_text: str
    result_type: ResultType
    model_version: str
    created_at: datetime
    input_text: Optional[str] = None
    confidence: Optional[float] = None
    title: Optional[str] = None
