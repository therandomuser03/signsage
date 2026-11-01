"""Adapter for signsage_isolated_words_v1 pose-based word classifier.

Uses backend/app/ml/isolated_words.py (IsolatedWordNet + prepare_isolated_features).
Contract: input landmarks (T, 543, 3) -> (64, 225) -> 6-class prediction.
Evaluation from metadata.json: NO-GO (accuracy 0.611, not signer-independent).
Not promoted to production translation pipeline.
"""
from __future__ import annotations
import json, sys, os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from isolated_words import IsolatedWordNet, prepare_isolated_features, MIN_CONFIDENCE, FEATURE_VERSION, FRAME_COUNT

class IsolatedWordAdapterError(RuntimeError):
    pass

class IsolatedWordAdapter:
    def __init__(self, bundle_dir: str | Path, task: str = "word",
                 device: str = "cpu", confidence_threshold: float = 0.5):
        self.bundle_dir = Path(bundle_dir)
        self.task = task
        metadata_path = self.bundle_dir / "metadata.json"
        if not metadata_path.exists():
            raise IsolatedWordAdapterError(f"No metadata.json in {bundle_dir}")
        meta = json.load(open(metadata_path))
        if meta.get("format") != "signsage_isolated_words_v1":
            raise IsolatedWordAdapterError(f"Unexpected bundle format: {meta.get('format')}")
        self.labels = meta.get("labels", [])
        self.n_classes = len(self.labels)
        self.feature_version = meta.get("feature_version", FEATURE_VERSION)
        self.confidence_threshold = confidence_threshold
        self.device = torch.device(device if (torch.cuda.is_available() or device == "cpu") else "cpu")
        self.model = IsolatedWordNet(self.n_classes)
        state_path = self.bundle_dir / "model.pt"
        state = torch.load(str(state_path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(state.get("state_dict", state), strict=False)
        self.model.to(self.device)
        self.model.eval()

    def classify_landmarks(self, landmarks: np.ndarray, min_confidence: Optional[float] = None) -> dict:
        # landmarks expected (T, 543, 3) numeric, finite
        if not isinstance(landmarks, np.ndarray) or landmarks.ndim != 3 or landmarks.shape[1:] != (543, 3):
            raise IsolatedWordAdapterError(f"Expected landmarks (T,543,3); got {getattr(landmarks,'shape',type(landmarks).__name__)}")
        features = prepare_isolated_features(landmarks)
        tensor = torch.from_numpy(features).unsqueeze(0).to(dtype=torch.float32, device=self.device)
        thresh = min_confidence if min_confidence is not None else self.confidence_threshold
        with torch.inference_mode():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1)[0]
        conf, idx = probs.max(dim=0)
        confidence = float(conf.item())
        index = int(idx.item())
        accepted = confidence >= thresh
        label = self.labels[index] if index < len(self.labels) else ""
        return {
            "label": label if accepted else "",
            "confidence": confidence,
            "accepted": accepted,
            "reason": "ok" if accepted else "low_model_confidence",
            "task": self.task,
            "index": index,
            "n_classes": self.n_classes,
            "model_version": "signsage_isolated_words_v1",
            "input_size": (FRAME_COUNT, 225),
            "evaluation_status": "NO-GO",
            "evaluation_note": "Small vocabulary; signer-independent split not verified; accuracy ~0.61.",
        }
