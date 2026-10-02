"""WebSocket protocol for live Sign->Text  (path: /ws/translate).

client -> server
  auth               {"type":"auth","access_token":"<supabase jwt>"}   (optional; guests skip it)
  landmarks.frame    {"type":"landmarks.frame","landmarks":[[x,y,z] x 543]}
  sentence.boundary  {"type":"sentence.boundary","save_history":true}
server -> client
  auth.ok            {"type":"auth.ok","user_id":"..."}
  translation.text   {"type":"translation.text","text","accepted","reason","confidence","result_type","history_id"}
  error              {"type":"error","code","message"}

Browsers cannot set headers on a WebSocket, so the token travels in the first
message rather than the URL (URLs end up in logs).
"""
from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from app.ml.preprocessing.landmarks import TOTAL_LANDMARKS
from app.schemas.translation import validate_landmark_rows


class UnknownMessageTypeError(ValueError):
    pass


class AuthMessage(BaseModel):
    type: Literal["auth"] = "auth"
    access_token: str = Field(..., min_length=1)


class LandmarksFrameMessage(BaseModel):
    type: Literal["landmarks.frame"] = "landmarks.frame"
    landmarks: List[List[float]] = Field(..., min_length=TOTAL_LANDMARKS, max_length=TOTAL_LANDMARKS)

    _check = field_validator("landmarks")(validate_landmark_rows)


class SentenceBoundaryMessage(BaseModel):
    type: Literal["sentence.boundary"] = "sentence.boundary"
    save_history: bool = True


IncomingMessage = Union[AuthMessage, LandmarksFrameMessage, SentenceBoundaryMessage]


class AuthOkMessage(BaseModel):
    type: Literal["auth.ok"] = "auth.ok"
    user_id: str


class TranslationTextMessage(BaseModel):
    type: Literal["translation.text"] = "translation.text"
    text: str
    accepted: bool
    reason: str
    confidence: float
    result_type: str
    history_id: Optional[str] = None
    # Streaming semantics: False = stabilizing interim from the rolling
    # window (do not persist, may be superseded); True = finalized by an
    # explicit sentence.boundary.
    final: bool = False


class ErrorCode:
    INVALID_JSON = "INVALID_JSON"
    INVALID_MESSAGE = "INVALID_MESSAGE"
    MESSAGE_TOO_LARGE = "MESSAGE_TOO_LARGE"
    SESSION_ERROR = "SESSION_ERROR"
    EMPTY_SEQUENCE = "EMPTY_SEQUENCE"
    UNKNOWN_MESSAGE_TYPE = "UNKNOWN_MESSAGE_TYPE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    UNAUTHORIZED = "UNAUTHORIZED"


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str


_PARSERS = {
    "auth": AuthMessage,
    "landmarks.frame": LandmarksFrameMessage,
    "sentence.boundary": SentenceBoundaryMessage,
}


def parse_incoming(raw: dict) -> IncomingMessage:
    """UnknownMessageTypeError for an unrecognised type; pydantic's
    ValidationError (a ValueError) for a recognised type with a bad body."""
    model = _PARSERS.get(raw.get("type"))
    if model is None:
        raise UnknownMessageTypeError(f"Unknown message type: {raw.get('type')!r}")
    return model.model_validate(raw)
