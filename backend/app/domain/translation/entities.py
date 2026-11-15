"""Translation domain entities."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


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
class NewHistorySentence:
    """One completed/finalized sentence to persist, scoped to a session.

    Nothing here is an intermediate prediction. Intermediate model output
    never becomes a history row.

    session_key: interaction-scoped idempotency key. All sentences sharing it
        belong to one translation session. The repository get-or-creates the
        session, so WebSocket reconnects / retries never fork a new session.
        When unset, each sentence gets its own fresh session (one-shot
        translations).
    client_key: sentence-level idempotency key. A retry/duplicate delivery of
        the same finalized sentence collapses instead of inserting twice.
    """
    direction: Direction
    output_text: str
    result_type: ResultType
    model_version: str
    input_text: Optional[str] = None
    gloss: Optional[str] = None
    confidence: Optional[float] = None
    session_key: Optional[str] = None
    client_key: Optional[str] = None
    title: Optional[str] = None
    status: str = "open"


@dataclass
class SentenceRecord:
    id: str
    session_id: str
    sequence_number: int
    direction: Direction
    input_text: Optional[str]
    output_text: str
    gloss: Optional[str]
    confidence: Optional[float]
    model_version: str
    result_type: ResultType
    created_at: datetime


@dataclass
class SessionRecord:
    id: str
    user_id: str
    direction: Direction
    title: Optional[str]
    status: str
    created_at: datetime
    ended_at: Optional[datetime]
    updated_at: Optional[datetime]
    sentences: List[SentenceRecord] = field(default_factory=list)

    @property
    def sentence_count(self) -> int:
        return len(self.sentences)

    @property
    def first_sentence(self) -> Optional[SentenceRecord]:
        return self.sentences[0] if self.sentences else None

    @property
    def model_version(self) -> str:
        """Model of the first recorded sentence, or '' when unknown."""
        first = self.first_sentence
        return first.model_version if first else ""

    @property
    def result_type(self) -> ResultType:
        first = self.first_sentence
        return first.result_type if first else ResultType.MODEL

    @property
    def preview(self) -> str:
        first = self.first_sentence
        return first.output_text if first else (self.title or "")
