"""Word-level recognition adapter for the demo vocabulary (8 words + rest).

Loads the FINAL-UNEVALUATED cnn-v3 bundle once per process and classifies a
(T, 543, 3) landmark array into top-3 labels with probabilities.

Preprocessing uses the vendored, numpy-only ``sign_features_v3`` (identical
output to research/track_b/features_v3.py) which implements the v3 recipe:
MediaPipe Left/Right labels are IGNORED; hands are tracked to stable physical
slots; ACTIVE/SECONDARY assignment by total wrist travel; canonicalisation
mirrors x about the shoulder midpoint so the active hand is right-hand-active;
gaps <= 5 frames interpolated; resampled to 60 frames after filling.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from app.ml.preprocessing.sign_features_v3 import N_FRAMES as _V3_FRAMES
from app.ml.preprocessing.sign_features_v3 import (
    FEAT_DIM as _V3_FEAT_DIM,
    extract_features as _extract_features,
)


class _WordNet(nn.Module):
    """cnn-v3: 1D-CNN, dropout 0.3, on 122-dim x 60-frame features."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(_V3_FEAT_DIM, 32, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 60, 122)
        x = x.transpose(1, 2)  # (B, 122, 60)
        x = self.net(x).squeeze(-1)  # (B, 64)
        return self.head(x)


@dataclass
class WordClassificationResult:
    label: Optional[str] = None
    confidence: float = 0.0
    top3: list[dict] = field(default_factory=list)
    motion_energy: float = 0.0
    frames_with_hand: int = 0
    status: str = "ok"  # "ok" | "no_hands" | "error"
    feature_ms: float = 0.0
    inference_ms: float = 0.0


class WordClassifierAdapter:
    """Single-process word classifier. Load once, reuse across requests."""

    def __init__(
        self,
        model: _WordNet,
        labels: list[str],
        device: torch.device,
        confidence_threshold: float,
        feature_version: str,
        model_version: str,
        signer_note: str,
    ):
        self.model = model
        self.labels = list(labels)
        self.device = device
        self.confidence_threshold = float(confidence_threshold)
        self.feature_version = feature_version
        self.model_version = model_version
        self.signer_note = signer_note
        self.model.eval()

    @classmethod
    def load(
        cls,
        bundle_dir: str | Path,
        device: str | None = None,
    ) -> "WordClassifierAdapter":
        root = Path(bundle_dir)
        meta_path = root / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"word model meta.json not found: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        labels = meta.get("label_list")
        if not isinstance(labels, list) or len(labels) < 2:
            raise ValueError("word model label list is missing or inconsistent")

        confidence_threshold = float(
            meta.get("recommended_confidence_threshold", 0.7)
        )
        model_version = str(meta.get("model_version", "word_v3_unknown"))
        feature_version = str(
            meta.get("features", {}).get("dim", _V3_FEAT_DIM)
        )
        signer_note = str(meta.get("signer_note", ""))

        checkpoint = root / "model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"word model checkpoint not found: {checkpoint}")
        torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Load weights only (no arbitrary code execution).
        payload = torch.load(checkpoint, map_location=torch_device, weights_only=True)
        state_dict = payload.get("state_dict", payload)

        model = _WordNet(len(labels)).to(torch_device)
        model.load_state_dict(state_dict)
        model.eval()

        return cls(
            model=model,
            labels=labels,
            device=torch_device,
            confidence_threshold=confidence_threshold,
            feature_version=feature_version,
            model_version=model_version,
            signer_note=signer_note,
        )

    @staticmethod
    def _frames_with_hand(landmarks_xyz: np.ndarray) -> int:
        """Count frames where at least one hand (left 501-522 or right 522-543)
        has any nonzero coordinate."""
        if landmarks_xyz.ndim != 3 or landmarks_xyz.shape[1:] != (543, 3):
            return 0
        left = np.any(landmarks_xyz[:, 501:522, :] != 0.0, axis=(1, 2))
        right = np.any(landmarks_xyz[:, 522:543, :] != 0.0, axis=(1, 2))
        return int(np.count_nonzero(left | right))

    def classify(self, landmarks_xyz: np.ndarray) -> WordClassificationResult:
        """Classify a (T, 543, 3) landmark array into top-3 word predictions.

        Raises ValueError on incompatible input (wrong shape, non-numeric,
        too few frames, NaN/inf).
        """
        import numpy as np

        arr = np.asarray(landmarks_xyz)
        if not isinstance(arr, np.ndarray) or arr.ndim != 3:
            raise ValueError("Expected landmarks as numpy.ndarray shaped (T, 543, 3)")
        if arr.shape[1:] != (543, 3):
            raise ValueError(
                f"Expected landmarks shaped (T, 543, 3), received {arr.shape}"
            )
        if arr.dtype.kind not in ("i", "u", "f"):
            raise ValueError(f"Expected numeric landmark dtype, received {arr.dtype}")
        if len(arr) < 2:
            raise ValueError(f"Need at least 2 frames, received {len(arr)}")
        if not np.isfinite(arr).all():
            raise ValueError("Landmarks must contain only finite coordinates")

        frames_with_hand = self._frames_with_hand(arr)
        if frames_with_hand == 0:
            return WordClassificationResult(
                label=None,
                confidence=0.0,
                top3=[],
                motion_energy=0.0,
                frames_with_hand=0,
                status="no_hands",
                feature_ms=0.0,
                inference_ms=0.0,
            )

        # v3 features: (60, 122) float32, resampled to 60 after filling.
        t0 = time.perf_counter()
        features = _extract_features(arr, layout="signsage")  # (60, 122)
        feature_ms = (time.perf_counter() - t0) * 1000
        if features.shape != (_V3_FRAMES, _V3_FEAT_DIM):
            raise ValueError(
                f"Expected features ({_V3_FRAMES}, {_V3_FEAT_DIM}), received {features.shape}"
            )
        if not np.isfinite(features).all():
            raise ValueError("Computed features contain NaN or Inf")

        motion = float(np.mean(np.abs(np.diff(arr.reshape(len(arr), -1), axis=0))))

        t0 = time.perf_counter()
        with torch.inference_mode():
            xb = torch.from_numpy(features[None].astype(np.float32)).to(self.device)
            logits = self.model(xb)[0]
            scores = torch.softmax(logits, dim=-1)
        inference_ms = (time.perf_counter() - t0) * 1000

        top3_scores, top3_idx = torch.topk(scores, k=min(3, len(self.labels)))
        top3 = [
            {
                "label": self.labels[int(i)],
                "probability": float(p),
            }
            for i, p in zip(top3_idx.tolist(), top3_scores.tolist())
        ]

        best_idx = int(top3_idx[0].item())
        best_label = self.labels[best_idx]
        best_conf = float(top3_scores[0].item())

        return WordClassificationResult(
            label=best_label,
            confidence=best_conf,
            top3=top3,
            motion_energy=motion,
            frames_with_hand=frames_with_hand,
            status="ok",
            feature_ms=feature_ms,
            inference_ms=inference_ms,
        )

    def promotion_eligible(self) -> bool:
        """This bundle is FINAL-UNEVALUATED; never eligible for promotion."""
        return False
