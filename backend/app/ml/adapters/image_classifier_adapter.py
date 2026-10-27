"""Image classification adapter for ISL alphabet / word recognition.

Wraps a SmallImageCNN bundle (model.pt + labels.json) behind a single
classify() call. Mirrors the SignToTextAdapter / IsolatedWordAdapter pattern:
load once, reuse across requests, validate input, return a structured result.

This adapter handles STATIC IMAGE classification only. It does NOT perform
continuous or dynamic sign recognition. Each call classifies one image frame.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from app.ml.models.image_cnn import SmallImageCNN


class ImageModelLoadError(RuntimeError):
    """The image model bundle could not be loaded or is incompatible."""


@dataclass
class ClassificationResult:
    """Result of classifying a single static image frame."""
    label: str
    confidence: float
    accepted: bool
    reason: str
    task: str  # "alphabet" | "word" | "unknown"
    index: int
    n_classes: int
    model_version: str
    input_size: tuple[int, int]  # (height, width) of the preprocessed input


class ImageClassifierAdapter:
    def __init__(self, bundle: SmallImageCNN._BundleModel, task: str,
                 model_version: str, confidence_threshold: float = 0.5):
        self.bundle = bundle
        self.task = task
        self.model_version = model_version
        self.confidence_threshold = confidence_threshold
        self.img_size = bundle.img_size
        self.n_classes = bundle.n_classes

    def classify(self, image_bytes: bytes, min_confidence: Optional[float] = None) -> ClassificationResult:
        """Classify one image (JPEG/PNG bytes).

        Returns a ClassificationResult. If confidence is below the threshold,
        accepted=False and label is the empty string (unsupported/uncertain).
        """
        import numpy as np

        arr = self._preprocess(image_bytes)
        thresh = min_confidence if min_confidence is not None else self.confidence_threshold

        with torch.inference_mode():
            logits = self.bundle.model(arr.to(self.bundle.device))
            probs = torch.softmax(logits, dim=-1)[0]
            conf, idx = probs.max(dim=0)

        confidence = float(conf.item())
        index = int(idx.item())
        label = self.bundle.index_to_label[index]

        accepted = confidence >= thresh
        reason = "ok" if accepted else "low_model_confidence"
        if not accepted:
            label = ""

        return ClassificationResult(
            label=label,
            confidence=confidence,
            accepted=accepted,
            reason=reason,
            task=self.task,
            index=index,
            n_classes=self.bundle.n_classes,
            model_version=self.model_version,
            input_size=(self.img_size, self.img_size),
        )

    @classmethod
    def load(cls, bundle_dir: str | Path, task: str, model_version: str,
             device: Optional[str] = None, confidence_threshold: float = 0.5) -> "ImageClassifierAdapter":
        bundle_dir = Path(bundle_dir)
        if not bundle_dir.is_dir():
            raise ImageModelLoadError(f"Model bundle directory not found: {bundle_dir}")
        try:
            bundle = SmallImageCNN.load_from_bundle(bundle_dir, device=device)
        except (FileNotFoundError, ValueError, OSError) as exc:
            raise ImageModelLoadError(f"Model bundle could not be loaded: {exc}") from exc
        return cls(bundle=bundle, task=task, model_version=model_version,
                   confidence_threshold=confidence_threshold)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, image_bytes: bytes) -> torch.Tensor:
        """Decode image bytes -> RGB -> resize to (img_size, img_size) -> (1,3,H,W) float [0,1]."""
        try:
            with Image.open(BytesIO(image_bytes)) as im:
                im = im.convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
                arr = np.asarray(im, dtype=np.float32) / 255.0
        except Exception as exc:
            raise ValueError(f"Failed to decode image: {exc}") from exc

        # (H, W, C) -> (C, H, W) -> (1, C, H, W)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor


def classify_base64(adapter: ImageClassifierAdapter, b64_image: str,
                    min_confidence: Optional[float] = None) -> ClassificationResult:
    """Convenience: decode a base64 data URI or raw base64 string and classify."""
    if b64_image.startswith("data:"):
        # Strip data URI prefix: data:image/jpeg;base64,<payload>
        _, b64_image = b64_image.split(",", 1)
    image_bytes = base64.b64decode(b64_image)
    return adapter.classify(image_bytes, min_confidence=min_confidence)
