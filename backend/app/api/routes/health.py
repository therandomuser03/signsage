"""GET /api/v1/health"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.infrastructure.storage.model_storage import (
    ModelBundleNotConfiguredError,
    ModelBundleNotFoundError,
    resolve_bundle_dir,
)
from app.ml.models.registry import is_loaded
from app.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    try:
        resolve_bundle_dir(settings)
        found = True
    except (ModelBundleNotConfiguredError, ModelBundleNotFoundError):
        found = False
    return HealthResponse(
        status="ok", model_promoted=settings.model_promoted,
        model_bundle_found=found, model_loaded=is_loaded(),
        word_recognition_enabled=settings.word_mode_enabled,
    )
