"""In-process model registry.

Loads the Sign->Text pipeline once (it's expensive: a checkpoint + a
tokenizer) and reuses the same instance across requests. Raises a clear
error rather than silently serving an unpromoted/NO-GO checkpoint unless
the caller explicitly acknowledges that (allow_unpromoted=True, used only
by scripts/model_check.py and tests).
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from app.core.config import Settings, get_settings
from app.core.errors import ModelNotPromotedError, ModelUnavailableError
from app.infrastructure.storage.model_storage import resolve_bundle_dir
from app.infrastructure.storage.model_storage import ModelBundleNotConfiguredError, ModelBundleNotFoundError
from app.ml.adapters.sign_to_text_adapter import ModelLoadError, SignToTextAdapter
from app.ml.contracts.inference_contract import InferenceContractError
from app.ml.contracts.landmark_contract import LandmarkContractError
from app.ml.contracts.model_contract import ModelContractError

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_pipeline = None


def get_sign_to_text_pipeline(settings: Optional[Settings] = None, allow_unpromoted: bool = False) -> SignToTextAdapter:
    """Return the process-wide SignToTextAdapter, loading it on first use.

    Raises ModelNotPromotedError if settings.model_promoted is False and
    allow_unpromoted is not explicitly set -- this is what keeps a NO-GO
    checkpoint (see docs/ai-ml/model-contract.md and backend/models/README.md)
    from being silently wired into user-facing translation.
    """
    global _pipeline
    settings = settings or get_settings()

    if not settings.model_promoted and not allow_unpromoted:
        raise ModelNotPromotedError(
            "The configured Sign->Text checkpoint has not passed the deployment "
            "gate (SIGNSAGE_MODEL_PROMOTED is False). Refusing to serve live "
            "translations. See research/manifests/models/ for the current "
            "run's status."
        )

    with _lock:
        if _pipeline is None:
            try:
                bundle_dir = resolve_bundle_dir(settings)
                logger.info("Loading Sign->Text pipeline from %s", bundle_dir)
                _pipeline = SignToTextAdapter.load(bundle_dir, device=settings.device)
            except (
                ModelBundleNotConfiguredError, ModelBundleNotFoundError, ModelLoadError,
                LandmarkContractError, ModelContractError, InferenceContractError,
            ) as exc:
                raise ModelUnavailableError(f"Sign->Text model could not be loaded: {exc}") from exc
        return _pipeline


def is_loaded() -> bool:
    return _pipeline is not None


def reset_registry() -> None:
    """Test-only: force the next get_sign_to_text_pipeline() call to reload."""
    global _pipeline
    with _lock:
        _pipeline = None
