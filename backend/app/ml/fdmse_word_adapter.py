"""FDMSE-ISL isolated-word adapter (2,002-word BiGRU, features = sign_features_v4).

Loads the bundle exported by notebook 02 (SignSage_02_FDMSE_Training_v2):

    model.pt, model_config.json, preprocessing.json, labels.json, metrics.json (+ MODEL_CARD.md)

and classifies a (T, 543, 3) raw browser landmark sequence into the top-3 words. The public shape matches
``WordClassifierAdapter`` (label / confidence / top3 / status / motion_energy / frames_with_hand ...), so
``WordClassifierAdapter.load(dir)`` returns this class when the directory holds ``preprocessing.json``.

Serving recipe (must equal training, see app/ml/preprocessing/sign_features_v4.py):
    resample to 48 frames -> pose-derived face points + isotropic x (video aspect) -> sign_features_v3
    -> per-feature standardisation (mean/std from the bundle) -> BiGRU.

The confidence threshold is stored and NOT applied here (the gate sits at route level, like the other
adapter). It was calibrated on STUDIO video; recalibrate on real webcam captures before relying on it.
This bundle is never promotion-eligible on its own: see docs/MODELS_AND_DATA.md section 6.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from app.ml.device import resolve_torch_device
from app.ml.preprocessing.sign_features_v3 import FEAT_DIM, N_FRAMES
from app.ml.preprocessing.sign_features_v4 import extract_features_v4

DEFAULT_ASPECT = 16.0 / 9.0      # assumed when the client does not send its video size (typical 1280x720 webcam)
_ASPECT_RANGE = (0.3, 5.0)


class SignGRU(nn.Module):
    """Identical to sp_train.SignGRU in the training notebook (parameter names must match)."""

    def __init__(self, n_classes: int, in_dim: int = FEAT_DIM, proj: int = 128, hidden: int = 128,
                 layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, proj), nn.GELU(), nn.Dropout(dropout * 0.5))
        self.gru = nn.GRU(proj, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.attn = nn.Linear(2 * hidden, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: (B, 60, 122), already standardised
        h, _ = self.gru(self.proj(x))
        w = torch.softmax(self.attn(h).squeeze(-1), dim=1).unsqueeze(-1)
        return self.head(torch.cat([(h * w).sum(1), h.max(1).values], dim=1))


@dataclass
class FDMSEWordResult:
    label: Optional[str] = None
    confidence: float = 0.0
    top3: list[dict] = field(default_factory=list)       # [{"label": str, "probability": float}]
    motion_energy: float = 0.0                           # same definition as the other word adapter
    frames_with_hand: int = 0
    status: str = "ok"                                   # "ok" | "no_hands" | "error"
    feature_ms: float = 0.0
    inference_ms: float = 0.0
    # FDMSE-specific extras (extra attributes are ignored by callers that do not know them)
    motion_feature: float = 0.0                          # mean |wrist velocity| of the model's own features
    aspect_used: float = 0.0
    aspect_source: str = "client"                        # "client" | "assumed"
    frames_in: int = 0


class FDMSEWordAdapter:
    """Single-process classifier. Load once, reuse across requests."""

    def __init__(self, model: SignGRU, labels: list[str], device: torch.device, mean: np.ndarray,
                 std: np.ndarray, confidence_threshold: float, model_version: str, spec: dict,
                 extract_frames: int, default_aspect: float, signer_note: str = ""):
        self.model = model
        self.labels = list(labels)
        self.device = device
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.confidence_threshold = float(confidence_threshold)
        self.model_version = model_version
        self.feature_version = "sign_features_v4"
        self.signer_note = signer_note
        self.spec = spec
        self.extract_frames = int(extract_frames)
        self.default_aspect = float(default_aspect)
        self.min_motion_feature: Optional[float] = None   # set when a motion floor has been calibrated
        self.robust_hands = True                          # see sign_features_v4.extract_features_v4
        self.model.eval()

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, bundle_dir: str | Path, device: str | None = None,
             default_aspect: float | None = None) -> "FDMSEWordAdapter":
        root = Path(bundle_dir)
        for name in ("model.pt", "model_config.json", "preprocessing.json", "labels.json", "metrics.json"):
            if not (root / name).is_file():
                raise FileNotFoundError(f"FDMSE bundle file missing: {root / name}")
        arch = json.loads((root / "model_config.json").read_text(encoding="utf-8"))
        pre = json.loads((root / "preprocessing.json").read_text(encoding="utf-8"))
        classes = json.loads((root / "labels.json").read_text(encoding="utf-8"))["classes"]
        metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))

        if pre.get("feature_spec") != "sign_features_v4":
            raise ValueError(f"unsupported feature spec {pre.get('feature_spec')!r}; this adapter serves sign_features_v4")
        if int(pre.get("feat_dim", -1)) != FEAT_DIM or int(pre.get("n_frames", -1)) != N_FRAMES:
            raise ValueError("bundle feature shape does not match sign_features_v3 (60 x 122)")
        mean = np.asarray(pre["standardize_mean"], np.float32)
        std = np.asarray(pre["standardize_std"], np.float32)
        if mean.shape != (FEAT_DIM,) or std.shape != (FEAT_DIM,) or not np.all(std > 0):
            raise ValueError("bundle standardisation vectors are invalid")
        if len(classes) != int(arch["n_classes"]):
            raise ValueError("labels.json and model_config.json disagree on the number of classes")
        detail = pre.get("feature_spec_detail") or {}
        if detail.get("face_source") != "pose":
            raise ValueError("bundle was not built with pose-derived face points (face_source != 'pose')")
        extract_frames = int((pre.get("extraction") or {}).get("n_frames", 48))

        model = SignGRU(int(arch["n_classes"]), FEAT_DIM, int(arch["proj"]), int(arch["hidden"]),
                        int(arch["layers"]), float(arch["dropout"]))
        torch_device = resolve_torch_device(device)
        payload = torch.load(root / "model.pt", map_location="cpu", weights_only=True)   # weights only, no code execution
        model.load_state_dict(payload.get("state_dict", payload) if isinstance(payload, dict) else payload)
        model.to(torch_device).eval()

        threshold = metrics.get("confidence_threshold")
        if threshold is None:
            raise ValueError("bundle has no calibrated confidence_threshold; refusing to guess one")
        digest = hashlib.sha256((root / "model.pt").read_bytes()).hexdigest()[:8]
        if default_aspect is None:
            default_aspect = float(os.environ.get("SIGNSAGE_WORD_DEFAULT_ASPECT", DEFAULT_ASPECT))
        adapter = cls(model, classes, torch_device, mean, std, float(threshold),
                      f"fdmse-isl-{len(classes)}w-v4-{digest}", detail, extract_frames, default_aspect,
                      signer_note="trained on FDMSE-ISL studio video (frontal, 20 signers)")
        motion = metrics.get("motion") or {}
        if motion.get("min_motion_feature") is not None:
            adapter.min_motion_feature = float(motion["min_motion_feature"])
        return adapter

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _frames_with_hand(arr: np.ndarray) -> int:
        if arr.ndim != 3 or arr.shape[1:] != (543, 3):
            return 0
        left = np.any(arr[:, 501:522, :] != 0.0, axis=(1, 2))
        right = np.any(arr[:, 522:543, :] != 0.0, axis=(1, 2))
        return int(np.count_nonzero(left | right))

    def _resolve_aspect(self, aspect: Optional[float]) -> tuple[float, str]:
        if aspect is not None and np.isfinite(aspect) and _ASPECT_RANGE[0] <= float(aspect) <= _ASPECT_RANGE[1]:
            return float(aspect), "client"
        return self.default_aspect, "assumed"

    # ---------------------------------------------------------------- inference
    def classify(self, landmarks_xyz: np.ndarray, aspect: Optional[float] = None) -> FDMSEWordResult:
        """Classify a (T, 543, 3) raw landmark array. `aspect` = video width / height (client-measured).
        Raises ValueError on incompatible input, like the other word adapter."""
        arr = np.asarray(landmarks_xyz)
        if arr.ndim != 3:
            raise ValueError("Expected landmarks as numpy.ndarray shaped (T, 543, 3)")
        if arr.shape[1:] != (543, 3):
            raise ValueError(f"Expected landmarks shaped (T, 543, 3), received {arr.shape}")
        if arr.dtype.kind not in ("i", "u", "f"):
            raise ValueError(f"Expected numeric landmark dtype, received {arr.dtype}")
        if len(arr) < 2:
            raise ValueError(f"Need at least 2 frames, received {len(arr)}")
        if not np.isfinite(arr).all():
            raise ValueError("Landmarks must contain only finite coordinates")

        used_aspect, source = self._resolve_aspect(aspect)
        n_hand = self._frames_with_hand(arr)
        if n_hand == 0:
            return FDMSEWordResult(status="no_hands", aspect_used=used_aspect, aspect_source=source, frames_in=len(arr))

        t0 = time.perf_counter()
        feats = extract_features_v4(arr, used_aspect, n_frames=self.extract_frames,
                                    forehead_k=float(self.spec.get("forehead_k", 1.2)),
                                    chin_k=float(self.spec.get("chin_k", 0.6)), robust=self.robust_hands)
        feature_ms = (time.perf_counter() - t0) * 1000
        if feats.shape != (N_FRAMES, FEAT_DIM):
            raise ValueError(f"Expected features ({N_FRAMES}, {FEAT_DIM}), received {feats.shape}")
        if not np.isfinite(feats).all():
            raise ValueError("Computed features contain NaN or Inf")

        motion = float(np.mean(np.abs(np.diff(arr.reshape(len(arr), -1), axis=0))))
        motion_feature = float(np.mean(np.abs(feats[:, 118:122])))     # the model's own wrist-velocity channels

        t0 = time.perf_counter()
        with torch.inference_mode():
            xb = torch.from_numpy(((feats - self.mean) / self.std)[None].astype(np.float32)).to(self.device)
            scores = torch.softmax(self.model(xb)[0], dim=-1)
        inference_ms = (time.perf_counter() - t0) * 1000

        top_s, top_i = torch.topk(scores, k=min(3, len(self.labels)))
        top3 = [{"label": self.labels[int(i)], "probability": float(p)} for i, p in zip(top_i.tolist(), top_s.tolist())]
        return FDMSEWordResult(
            label=top3[0]["label"], confidence=top3[0]["probability"], top3=top3, motion_energy=motion,
            frames_with_hand=n_hand, status="ok", feature_ms=feature_ms, inference_ms=inference_ms,
            motion_feature=motion_feature, aspect_used=used_aspect, aspect_source=source, frames_in=len(arr))

    def is_moving(self, result: FDMSEWordResult) -> bool:
        """True unless a calibrated motion floor exists and the clip is below it (static pose)."""
        return self.min_motion_feature is None or result.motion_feature >= self.min_motion_feature

    def promotion_eligible(self) -> bool:
        """Never promotable on its own: webcam accuracy and the other gates are unmeasured."""
        return False
