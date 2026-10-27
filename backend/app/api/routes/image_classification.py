"""REST endpoints for static ISL image classification.

Endpoints:
  POST /api/v1/classification/image   -- classify one image (alphabet or word)

Model bundles are loaded once per process and cached. The bundles point at the
artifact directories produced by research/image_classification/train.py
(artifacts/alphabet/ and artifacts/words/). They are NOT the transformer
Sign->Text bundle and have their own promotion/status tracking.

These endpoints are independent of the Sign->Text transformer path: enabling
them does not affect model_promoted, allow_unpromoted_dev, or the WebSocket
translation route.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import Settings, get_settings
from app.core.errors import InvalidLandmarksError, ModelUnavailableError
from app.ml.adapters.image_classifier_adapter import (
    ImageClassifierAdapter,
    ImageModelLoadError,
    classify_base64,
)
from app.schemas.image_classification import (
    ImageClassifyRequest,
    ImageClassifyResponse,
)

router = APIRouter(prefix="/classification", tags=["classification"])

# Process-wide cache of loaded image classifier adapters.
# Key: (task, bundle_dir). Value: ImageClassifierAdapter.
_adapters: dict[tuple[str, str], ImageClassifierAdapter] = {}


def _get_adapter(task: str, settings: Settings) -> ImageClassifierAdapter:
    if task == "alphabet":
        bundle_dir = settings.image_alphabet_bundle_dir
    elif task == "word":
        bundle_dir = settings.image_word_bundle_dir
    else:
        raise HTTPException(status_code=422, detail=f"Unknown classification task: {task!r}")

    if not bundle_dir:
        raise ModelUnavailableError(
            f"No bundle directory configured for image classification task {task!r}. "
            f"Set SIGNSAGE_IMAGE_{task.upper()}_BUNDLE_DIR."
        )

    key = (task, bundle_dir)
    if key not in _adapters:
        try:
            _adapters[key] = ImageClassifierAdapter.load(
                bundle_dir, task=task,
                model_version=f"signsage-image-{task}",
                device=settings.device,
            )
        except (ImageModelLoadError, FileNotFoundError, ValueError, OSError) as exc:
            raise ModelUnavailableError(
                f"Image classification model ({task}) could not be loaded: {exc}"
            ) from exc
    return _adapters[key]


@router.post("/image", response_model=ImageClassifyResponse)
async def classify_image(
    payload: ImageClassifyRequest,
    settings: Settings = Depends(get_settings),
) -> ImageClassifyResponse:
    adapter = _get_adapter(payload.task, settings)
    try:
        result = classify_base64(adapter, payload.image, min_confidence=payload.min_confidence)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image data: {exc}") from exc
    return ImageClassifyResponse(
        label=result.label,
        accepted=result.accepted,
        reason=result.reason,
        confidence=result.confidence,
        task=result.task,
        index=result.index,
        n_classes=result.n_classes,
        model_version=result.model_version,
        input_size=[result.input_size[0], result.input_size[1]],
    )
