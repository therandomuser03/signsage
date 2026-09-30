"""Application-level exceptions + FastAPI exception handlers.

Each error carries a stable `code` (same vocabulary as the WebSocket protocol's
error messages, see schemas/websocket.py:ErrorCode) and an HTTP `status_code`,
so the transport layer only translates -- it never decides.
"""
from __future__ import annotations

import logging

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


class NotFoundError(SignSageError):
    code = "NOT_FOUND"
    status_code = 404


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(SignSageError)
    async def _handle_signsage_error(request: Request, exc: SignSageError) -> JSONResponse:
        logger.warning("%s: %s", exc.code, exc.message)
        headers = {"WWW-Authenticate": "Bearer"} if isinstance(exc, AuthError) else None
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message},
            headers=headers,
        )
