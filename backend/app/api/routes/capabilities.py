"""GET /api/v1/capabilities and /api/v1/models -- what the frontend should
show, hide or label. The frontend reads these instead of guessing."""
from __future__ import annotations

import json
from pathlib import Path
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
from app.ml.word_classifier_adapter import resolve_word_model_dir
from app.schemas.common import CapabilitiesResponse, ModelInfo, ModelsResponse
from app.services.avatar_gloss_service import load_vocabulary
from app.services.history_service import build_history_service

router = APIRouter(tags=["capabilities"])

_REPORTED_METRICS = (
    "wer", "bleu", "rouge_l", "output_diversity_fraction", "top_prediction_freq",
    "accuracy", "macro_recall",
)


def _bundle_found(settings: Settings) -> bool:
    try:
        resolve_bundle_dir(settings)
        return True
    except (ModelBundleNotConfiguredError, ModelBundleNotFoundError):
        return False


def _word_model_version(settings: Settings) -> "str | None":
    """model_version of the word model, read from its meta.json at
    <resolved word model dir>/meta.json -- the same file
    WordClassifierAdapter.load() uses. Never the hardcoded literal."""
    if not settings.word_mode_enabled:
        return None
    try:
        meta = json.loads(
            (Path(resolve_word_model_dir(settings)) / "meta.json").read_text(encoding="utf-8")
        )
    except Exception:  # noqa: BLE001 -- capabilities must not 500 on a bad/missing meta
        return None
    version = meta.get("model_version")
    return str(version) if isinstance(version, str) and version else None


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
        source = "isolated-word" if settings.model_kind == "isolated_words" else "transformer"
    fallback_vocab = list(FALLBACK_VOCABULARY) if settings.fallback_enabled else []
    if reason is not None and fallback_vocab:
        # Transformer correctly gated, but the checkpoint-free heuristic
        # fallback serves a 5-sign vocabulary -- report honestly.
        reason = None
        source = "heuristic-fallback"
    word_model_version = _word_model_version(settings)
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
        word_recognition_enabled=settings.word_mode_enabled,
        word_recognition_vocabulary=(
            ["hello", "me", "you", "yes", "no", "help", "water", "where", "rest"]
            if settings.word_mode_enabled else None
        ),
        word_recognition_threshold=(
            0.7 if settings.word_mode_enabled else None
        ),
        word_recognition_model_version=(
            word_model_version if settings.word_mode_enabled else None
        ),
        word_recognition_label="sign-sequence to sentence interpretation (8 words plus a rest (no-sign) class), adapted to one signer",
    )


@router.get("/models", response_model=ModelsResponse)
def models(settings: Settings = Depends(get_settings)) -> ModelsResponse:
    status = None
    metrics: Dict[str, float] = {}
    try:
        with open(resolve_bundle_dir(settings) / "evaluation_results.json", "r", encoding="utf-8") as f:
            evaluation = json.load(f)
        status = evaluation.get("status", evaluation.get("phase_0_status"))
        values = evaluation.get("test", evaluation)
        metrics = {k: float(values[k]) for k in _REPORTED_METRICS if isinstance(values.get(k), (int, float))}
    except (ModelBundleNotConfiguredError, ModelBundleNotFoundError, OSError, ValueError):
        pass
    return ModelsResponse(models=[ModelInfo(
        id=settings.model_version, direction="sign_to_text", promoted=settings.model_promoted,
        loaded=is_loaded(), evaluation_status=status, metrics=metrics,
    )])
