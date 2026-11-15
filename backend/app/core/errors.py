"""Application-level exceptions + FastAPI exception handlers.

Each error carries a stable `code` (same vocabulary as the WebSocket protocol's
error messages, see schemas/websocket.py:ErrorCode) and an HTTP `status_code`,
so the transport layer only translates -- it never decides.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class SignSageError(Exception):
    code = "INTERNAL_ERROR"
    status_code = 400

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ModelNotPromotedError(SignSageError):
    code = "MODEL_NOT_PROMOTED"
    status_code = 503


class ModelUnavailableError(SignSageError):
    code = "MODEL_UNAVAILABLE"
    status_code = 503


class InvalidLandmarksError(SignSageError):
    code = "INVALID_MESSAGE"
    status_code = 422


class InferenceTimeoutError(SignSageError):
    # Shares the SESSION_ERROR code with the WebSocket protocol (no new error
    # vocabulary, no frontend change): both transports report a timed-out
    # inference as SESSION_ERROR with a message saying so.
    code = "SESSION_ERROR"
    status_code = 503


class AuthError(SignSageError):
    code = "UNAUTHORIZED"
    status_code = 401


class AuthNotConfiguredError(SignSageError):
    code = "AUTH_NOT_CONFIGURED"
    status_code = 503


class AuthUnavailableError(SignSageError):
    code = "AUTH_UNAVAILABLE"
    status_code = 503


class HistoryUnavailableError(SignSageError):
    code = "HISTORY_UNAVAILABLE"
    status_code = 503

    def __init__(
        self,
        message: str,
        *,
        path: Optional[str] = None,
        status: Optional[int] = None,
        pg_code: Optional[str] = None,
        pg_message: Optional[str] = None,
    ):
        super().__init__(message)
        # Details for the history-service log line only; never carries tokens,
        # keys or payload text (PostgREST error bodies name constraints, not data).
        self.path = path
        self.status = status
        self.pg_code = pg_code
        self.pg_message = pg_message


class TtsUnavailableError(SignSageError):
    code = "TTS_UNAVAILABLE"
    status_code = 503


class NotFoundError(SignSageError):
    code = "NOT_FOUND"
    status_code = 404


class RateLimitedError(SignSageError):
    code = "RATE_LIMITED"
    status_code = 429

    def __init__(self, message: str, retry_after: int):
        super().__init__(message)
        self.retry_after = retry_after


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(SignSageError)
    async def _handle_signsage_error(request: Request, exc: SignSageError) -> JSONResponse:
        logger.warning("%s: %s", exc.code, exc.message)
        headers: dict[str, str] = {}
        if isinstance(exc, AuthError):
            headers["WWW-Authenticate"] = "Bearer"
        retry_after = getattr(exc, "retry_after", None)
        if retry_after is not None:
            headers["Retry-After"] = str(retry_after)
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message},
            headers=headers or None,
        )
