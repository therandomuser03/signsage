"""FastAPI dependency providers.

build_* are plain functions for code outside route handlers (the WebSocket
route); get_* wrap them for Depends() injection.
"""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, Request

from app.core.config import Settings, get_settings
from app.core.errors import (
    AuthError,
    AuthNotConfiguredError,
    HistoryUnavailableError,
    ModelNotPromotedError,
    ModelUnavailableError,
)
from app.core.security import AuthenticatedUser
from app.ml.models.registry import get_sign_to_text_pipeline
from app.services.authentication_service import authenticate_header
from app.services.history_service import HistoryService, build_history_service
from app.services.translation_service import TranslationService
import logging

logger = logging.getLogger(__name__)


def build_translation_service(settings: Settings) -> TranslationService:
    """Transformer when promotable, else the heuristic fallback.

    The transformer gate is unchanged: a NO-GO checkpoint is never served.
    The fallback is checkpoint-free rule logic with its own 5-sign vocabulary,
    reported as "heuristic-fallback" via /capabilities.
    """
    try:
        pipeline = get_sign_to_text_pipeline(settings, allow_unpromoted=settings.allow_unpromoted_dev)
        return TranslationService(pipeline, settings)
    except (ModelNotPromotedError, ModelUnavailableError) as exc:
        if not settings.fallback_enabled:
            raise
        logger.warning("[MODEL] transformer unavailable (%s); serving heuristic fallback.", exc)
        from app.ml.inference.heuristic_signs import HeuristicSignPipeline

        return TranslationService(HeuristicSignPipeline(), settings)


def get_translation_service(settings: Settings = Depends(get_settings)) -> TranslationService:
    return build_translation_service(settings)


async def get_optional_user(
    request: Request, settings: Settings = Depends(get_settings)
) -> Optional[AuthenticatedUser]:
    """Guest -> None. A present-but-invalid token is an error, never silently a guest."""
    return await authenticate_header(request.headers.get("authorization"), settings)


async def get_current_user(
    user: Optional[AuthenticatedUser] = Depends(get_optional_user),
    settings: Settings = Depends(get_settings),
) -> AuthenticatedUser:
    if user is None:
        if not settings.auth_enabled:
            raise AuthNotConfiguredError("Authentication is not configured on this server.")
        raise AuthError("Authentication required.")
    return user


async def get_request_user(
    user: Optional[AuthenticatedUser] = Depends(get_optional_user),
    settings: Settings = Depends(get_settings),
) -> Optional[AuthenticatedUser]:
    """For translation endpoints: guests allowed unless SIGNSAGE_AUTH_REQUIRED=true."""
    if user is None and settings.auth_required:
        raise AuthError("Authentication required.")
    return user


def get_optional_history_service(settings: Settings = Depends(get_settings)) -> Optional[HistoryService]:
    return build_history_service(settings)


def get_history_service(settings: Settings = Depends(get_settings)) -> HistoryService:
    service = build_history_service(settings)
    if service is None:
        raise HistoryUnavailableError("History is disabled or not configured on this server.")
    return service
