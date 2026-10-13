"""GET /api/v1/capabilities and /api/v1/models -- what the frontend should
show, hide or label. The frontend reads these instead of guessing."""
from __future__ import annotations

import json
from typing import Dict

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.infrastructure.storage.model_storage import (
    ModelBundleNotConfiguredError,
    ModelBundleNotFoundError,
    resolve_bundle_dir,
)
from app.ml.models.registry import is_loaded
from app.ml.preprocessing.landmarks import TOTAL_LANDMARKS
from app.schemas.common import CapabilitiesResponse, ModelInfo, ModelsResponse
from app.services.avatar_gloss_service import load_vocabulary
from app.services.history_service import build_history_service

router = APIRouter(tags=["capabilities"])

_REPORTED_METRICS = ("wer", "bleu", "rouge_l", "output_diversity_fraction", "top_prediction_freq")


def _bundle_found(settings: Settings) -> bool:
    try:
        resolve_bundle_dir(settings)
        return True
    except (ModelBundleNotConfiguredError, ModelBundleNotFoundError):
        return False


@router.get("/capabilities", response_model=CapabilitiesResponse)
def capabilities(settings: Settings = Depends(get_settings)) -> CapabilitiesResponse:
    from app.ml.inference.heuristic_signs import FALLBACK_VOCABULARY

    reason = None
    source = "unavailable"
    if not settings.model_promoted and not settings.allow_unpromoted_dev:
        reason = "model_not_promoted"
    elif not _bundle_found(settings):
        reason = "model_bundle_missing"
    else:
        source = "transformer"
    fallback_vocab = list(FALLBACK_VOCABULARY) if settings.fallback_enabled else []
    if reason is not None and fallback_vocab:
        # Transformer correctly gated, but the checkpoint-free heuristic
        # fallback serves a 5-sign vocabulary -- report honestly.
        reason = None
        source = "heuristic-fallback"
    return CapabilitiesResponse(
        api_version="v1",
        websocket_path="/ws/translate",
        auth_enabled=settings.auth_enabled,
        auth_required=settings.auth_required,
        history_enabled=build_history_service(settings) is not None,
        sign_to_text_available=reason is None,
        sign_to_text_unavailable_reason=reason,
        sign_to_text_source=source,
        fallback_vocabulary=fallback_vocab,
        text_to_sign_mode="avatar_gloss",
        avatar_vocabulary_size=len(load_vocabulary(settings)),
        landmark_count=TOTAL_LANDMARKS,
        max_frames_per_sequence=settings.max_frames_per_sequence,
    )


@router.get("/models", response_model=ModelsResponse)
def models(settings: Settings = Depends(get_settings)) -> ModelsResponse:
    status = None
    metrics: Dict[str, float] = {}
    try:
        with open(resolve_bundle_dir(settings) / "evaluation_results.json", "r", encoding="utf-8") as f:
            evaluation = json.load(f)
        status = evaluation.get("phase_0_status")
        metrics = {k: float(evaluation[k]) for k in _REPORTED_METRICS if isinstance(evaluation.get(k), (int, float))}
    except (ModelBundleNotConfiguredError, ModelBundleNotFoundError, OSError, ValueError):
        pass
    return ModelsResponse(models=[ModelInfo(
        id=settings.model_version, direction="sign_to_text", promoted=settings.model_promoted,
        loaded=is_loaded(), evaluation_status=status, metrics=metrics,
    )])
