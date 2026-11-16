"""EXPERIMENTAL adapter for the FDMSE-ISL 200-word recognition model.

This module is NOT wired into the production pipeline. It exists so the
Colab-trained FDMSE-ISL 200-word bundle can be loaded and smoke-tested by
SignSage engineers before any promotion decision. It mirrors the PUBLIC SHAPE
and SEMANTICS of ``word_classifier_adapter.WordClassifierAdapter`` exactly,
but implements the FDMSE architecture (BiGRU on sign_features_v3 inputs),
because cnn-v3 and this BiGRU are different parameterizations. It provides the
same interface engineers know from production:

    adapter = FDMSEWordAdapter.load(<bundle_dir>)
    result = adapter.classify(landmarks_543)         # (T, 543, 3) float32
    result.label / result.confidence / result.top3 / result.status

Behaviour parity with production (verified by reading both adapters):

| Contract point                    | Production word_v3               | This adapter (FDMSE-ISL)        |
|-----------------------------------|----------------------------------|---------------------------------|
| meta.json keys                    | label_list / model_version / conf| identical, same defaults        |
| device resolution                 | resolve_torch_device             | same helper when importable     |
| weights file                      | model.pt                         | model.pt (same convention)      |
| state_dict vs {"state_dict": ...} | both accepted                    | both accepted                   |
| input tensor                      | (1, 60, 122)                     | (1, 60, 122)                    |
| input landmarks                   | (T, 543, 3) float32              | (T, 543, 3) float32             |
| no_hands short-circuit            | yes (label=None, status=no_hands)| identical logic                 |
| validation errors                 | ValueError on bad shape/dtype    | identical logic                 |
| feature extractor                 | sign_features_v3 extract_features| same, imported from backend     |
| top3 entries                      | {"label", "probability"}         | identical                       |
| confidence threshold              | stored, NOT applied (gate is at route level) | identical — same semantics     |
| promotion_eligible()              | False (FINAL-UNEVALUATED)        | False (EXPERIMENTAL)            |
| load order / ownership            | single process-wide instance     | single instance by convention   |

DO NOT instantiate this from API routes. DO NOT promote on the strength of a
single training run. Promotion requires the gates in docs/ml/MODEL_STATUS.md
plus a fresh paired evaluation on held-out signer captures — this experiment
does not satisfy them (FDMSE-ISL is a frontal, single-viewpoint dataset from a
specific southern-Indian demographic).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    from app.ml.device import resolve_torch_device
    from app.ml.preprocessing.sign_features_v3 import (
        FEAT_DIM,
        N_FRAMES,
        extract_features as _extract_features,
    )
    HAVE_BACKEND = True
except ImportError:  # top-level import (e.g. notebooks / smoke scripts)
    resolve_torch_device = None  # type: ignore[assignment]
    FEAT_DIM, N_FRAMES = 122, 60
    HAVE_BACKEND = False

    def _extract_features(landmarks, n_frames=N_FRAMES):  # type: ignore[override]
        raise RuntimeError(
            "sign_features_v3 not importable in this context; run from the backend "
            "venv (PYTHONPATH=backend) so the production feature extractor is used."
        )

# Keep in sync with the notebook's ModelConfig — the checkpoint carries the
# actual values and meta.json records them; defaults match the notebook.
_DEFAULT_HIDDEN = 128
_DEFAULT_LAYERS = 2
_DEFAULT_DROPOUT = 0.3


class FDMSEWordNet(nn.Module):
    """BiGRU (default hidden 128, 2 layers, dropout 0.3) + linear head."""

    def __init__(self, n_classes: int, hidden: int = _DEFAULT_HIDDEN, layers: int = _DEFAULT_LAYERS, dropout: float = _DEFAULT_DROPOUT):
        super().__init__()
        self.hidden = hidden
        self.rnn = nn.GRU(FEAT_DIM, hidden, layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 60, 122)
        out, h_n = self.rnn(x)   # h_n: (layers * 2, batch, hidden)
        fwd_final = h_n[-2]      # last layer, forward direction
        bwd_final = h_n[-1]      # last layer, backward direction
        return self.head(self.dropout(torch.cat([fwd_final, bwd_final], dim=1)))


@dataclass
class FDMSEWordResult:
    label: str | None = None
    confidence: float = 0.0
    top3: list[dict] = field(default_factory=list)   # [{"label": str, "probability": float}]
    motion_energy: float = 0.0
    frames_with_hand: int = 0
    status: str = "ok"  # "ok" | "no_hands" | "error"
    feature_ms: float = 0.0
    inference_ms: float = 0.0


class FDMSEWordAdapter:
    """Experimental loader for the FDMSE-ISL 200-word bundle (production-shaped)."""

    def __init__(
        self,
        model: FDMSEWordNet,
        labels: list[str],
        device,
        confidence_threshold: float,
        model_version: str,
        signer_note: str,
        feature_version: str,
    ):
        self.model = model
        self.labels = list(labels)
        self.device = device
        self.confidence_threshold = float(confidence_threshold)
        self.model_version = str(model_version)
        self.signer_note = signer_note
        self.feature_version = feature_version
        self.model.eval()

    @classmethod
    def load(cls, bundle_dir: str | Path, device: str | None = None) -> "FDMSEWordAdapter":
        root = Path(bundle_dir)
        meta_path = root / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"FDMSE bundle meta.json not found: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        labels = meta.get("label_list")
        if not isinstance(labels, list) or len(labels) < 2:
            raise ValueError("FDMSE bundle label_list missing or inconsistent")

        confidence_threshold = float(meta.get("recommended_confidence_threshold", 0.7))
        model_version = str(meta.get("model_version", "fdmse_200word_unknown"))
        feature_version = str(meta.get("feature_version", meta.get("features", {}).get("spec", "sign_features_v3")))

        arch = meta.get("architecture", {})
        hidden = int(arch.get("hidden_size", _DEFAULT_HIDDEN))
        layers = int(arch.get("layers", _DEFAULT_LAYERS))
        dropout = float(arch.get("dropout", _DEFAULT_DROPOUT))

        if resolve_torch_device is not None:
            resolved = resolve_torch_device(device)
        else:
            resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        model = FDMSEWordNet(len(labels), hidden, layers, dropout)
        payload = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
        # Accept a raw state_dict, or a one-level wrapped checkpoint ({"model" | "state_dict": ...}).
        if isinstance(payload, dict) and ("model" in payload or "state_dict" in payload):
            state = payload.get("state_dict", payload.get("model"))
        else:
            state = payload
        model.load_state_dict(state)
        model.to(resolved)
        model.eval()

        return cls(
            model=model,
            labels=labels,
            device=resolved,
            confidence_threshold=confidence_threshold,
            model_version=model_version,
            signer_note=str(meta.get("signer_note", "trained on FDMSE-ISL subset")),
            feature_version=feature_version,
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

    def classify(self, landmarks_xyz: np.ndarray) -> FDMSEWordResult:
        """Classify a (T, 543, 3) landmark array into top-3 word predictions.

        Raises ValueError on incompatible input (wrong shape, non-numeric,
        too few frames, NaN/inf) — exactly like the production adapter.
        """
        arr = np.asarray(landmarks_xyz)
        if not isinstance(arr, np.ndarray) or arr.ndim != 3:
            raise ValueError("Expected landmarks as numpy.ndarray shaped (T, 543, 3)")
        if arr.shape[1:] != (543, 3):
            raise ValueError(f"Expected landmarks shaped (T, 543, 3), received {arr.shape}")
        if arr.dtype.kind not in ("i", "u", "f"):
            raise ValueError(f"Expected numeric landmark dtype, received {arr.dtype}")
        if len(arr) < 2:
            raise ValueError(f"Need at least 2 frames, received {len(arr)}")
        if not np.isfinite(arr).all():
            raise ValueError("Landmarks must contain only finite coordinates")

        frames_with_hand = self._frames_with_hand(arr)
        if frames_with_hand == 0:
            return FDMSEWordResult(
                label=None, confidence=0.0, top3=[], motion_energy=0.0,
                frames_with_hand=0, status="no_hands",
                feature_ms=0.0, inference_ms=0.0,
            )

        if not HAVE_BACKEND:
            raise RuntimeError("sign_features_v3 not importable — run this adapter from the backend venv.")

        # v3 features: (60, 122) float32, resampled to 60 after filling.
        t0 = time.perf_counter()
        features = _extract_features(arr, layout="signsage")  # (60, 122)
        feature_ms = (time.perf_counter() - t0) * 1000
        if features.shape != (N_FRAMES, FEAT_DIM):
            raise ValueError(f"Expected features ({N_FRAMES}, {FEAT_DIM}), received {features.shape}")
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
            {"label": self.labels[int(i)], "probability": float(p)}
            for i, p in zip(top3_idx.tolist(), top3_scores.tolist())
        ]

        best_idx = int(top3_idx[0].item())
        best_label = self.labels[best_idx]
        best_conf = float(top3_scores[0].item())

        return FDMSEWordResult(
            label=best_label,
            confidence=best_conf,
            top3=top3,
            motion_energy=motion,
            frames_with_hand=frames_with_hand,
            status="ok",
            feature_ms=feature_ms,
            inference_ms=inference_ms,
        )

    # Backwards-compatible name used by earlier verification scripts.
    def classify_landmarks(self, landmarks_xyz: np.ndarray) -> FDMSEWordResult:
        return self.classify(landmarks_xyz)

    def promotion_eligible(self) -> bool:
        """This bundle is EXPERIMENTAL; never eligible for promotion."""
        return False
