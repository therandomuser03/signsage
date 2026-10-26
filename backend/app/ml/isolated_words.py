"""Small-vocabulary ISL word classifier and its landmark contract."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn


FEATURE_VERSION = "pose_hands_root_scale_v1"
FRAME_COUNT = 64
FEATURE_DIM = 75 * 3
MIN_CONFIDENCE = 0.5
PROMOTION_MIN_ACCURACY = 0.8
PROMOTION_MIN_MACRO_RECALL = 0.8
PROMOTION_MIN_CLASS_RECALL = 0.6


def prepare_isolated_features(landmarks: np.ndarray) -> np.ndarray:
    """Convert SignSage (T, 543, 3) landmarks to fixed (64, 225) features."""
    if not isinstance(landmarks, np.ndarray) or landmarks.ndim != 3 or landmarks.shape[1:] != (543, 3):
        shape = getattr(landmarks, "shape", type(landmarks).__name__)
        raise ValueError(f"Expected landmarks shaped (T, 543, 3); received {shape}")
    if len(landmarks) < 2 or not np.issubdtype(landmarks.dtype, np.number):
        raise ValueError("Expected at least 2 numeric landmark frames")
    if not np.isfinite(landmarks).all():
        raise ValueError("Landmarks must contain only finite coordinates")

    indices = np.r_[0:33, 501:543]
    selected = np.asarray(landmarks[:, indices, :], dtype=np.float32)
    hips = (selected[:, 23] + selected[:, 24]) * 0.5
    shoulder_width = np.linalg.norm(selected[:, 11, :2] - selected[:, 12, :2], axis=1)
    valid_widths = shoulder_width[shoulder_width > 1e-5]
    scale = float(np.median(valid_widths)) if len(valid_widths) else 1.0
    selected = (selected - hips[:, None, :]) / scale

    sample_indices = np.linspace(0, len(selected) - 1, FRAME_COUNT).round().astype(int)
    return selected[sample_indices].reshape(FRAME_COUNT, FEATURE_DIM)


def split_isolated_rows(rows: list[dict[str, str]], seed: int) -> dict[str, list[dict[str, str]]]:
    """Stratify clips while keeping repeated source recording IDs together."""
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["normalized_word"]].append(row)
        source = row.get("signer", "").strip() or row["video_path"]
        by_source[source].append(row)

    targets: dict[str, Counter] = {name: Counter() for name in ("train", "validation", "test")}
    for label, samples in by_label.items():
        held_out = max(1, round(len(samples) * 0.15))
        targets["train"][label] = len(samples) - held_out * 2
        targets["validation"][label] = held_out
        targets["test"][label] = held_out

    groups = list(by_source.values())
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    result: dict[str, list[dict[str, str]]] = {
        name: [] for name in ("train", "validation", "test")
    }
    counts: dict[str, Counter] = {name: Counter() for name in result}
    for group in groups:
        group_counts = Counter(row["normalized_word"] for row in group)
        candidates = [
            name for name in result
            if all(counts[name][label] + amount <= targets[name][label]
                   for label, amount in group_counts.items())
        ]
        if not candidates:
            raise ValueError("Source-group-aware split cannot satisfy per-class split counts")
        rng.shuffle(candidates)
        selected = min(
            candidates,
            key=lambda name: sum(
                ((counts[name][label] + amount) / targets[name][label]) ** 2
                for label, amount in group_counts.items()
            ),
        )
        result[selected].extend(group)
        counts[selected].update(group_counts)

    if any(counts[name] != targets[name] for name in result):
        raise ValueError("Source-group-aware split did not meet stratified target counts")
    return result


class IsolatedWordNet(nn.Module):
    def __init__(self, class_count: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(FEATURE_DIM, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(96, class_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features.transpose(1, 2))


class IsolatedWordAdapter:
    def __init__(self, model: IsolatedWordNet, labels: list[str], device: torch.device, metadata: dict):
        self.model = model
        self.labels = labels
        self.device = device
        self.metadata = metadata
        self.confidence_threshold = MIN_CONFIDENCE

    @classmethod
    def load(cls, bundle_dir: str | Path, device: str | None = None) -> "IsolatedWordAdapter":
        root = Path(bundle_dir)
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        labels = json.loads((root / "labels.json").read_text(encoding="utf-8"))
        evaluation = json.loads((root / "evaluation_results.json").read_text(encoding="utf-8"))
        if metadata.get("format") != "signsage_isolated_words_v1":
            raise ValueError("Unsupported isolated-word bundle format")
        if labels != metadata.get("labels") or not isinstance(labels, list) or len(labels) < 2:
            raise ValueError("Bundle label mapping is missing or inconsistent")
        if evaluation != metadata.get("evaluation"):
            raise ValueError("Bundle evaluation results do not match metadata")
        if metadata.get("feature_version") != FEATURE_VERSION or metadata.get("feature_dim") != FEATURE_DIM:
            raise ValueError("Bundle preprocessing contract does not match this runtime")
        checkpoint = root / "model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Isolated-word checkpoint not found: {checkpoint}")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != metadata.get("checkpoint_sha256"):
            raise ValueError("Isolated-word checkpoint checksum does not match metadata")
        torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        payload = torch.load(checkpoint, map_location=torch_device, weights_only=True)
        model = IsolatedWordNet(len(labels)).to(torch_device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        adapter = cls(model, labels, torch_device, metadata)
        adapter.confidence_threshold = float(metadata.get("confidence_threshold", MIN_CONFIDENCE))
        with torch.inference_mode():
            probe = model(torch.zeros((1, FRAME_COUNT, FEATURE_DIM), device=torch_device))
        if not torch.isfinite(probe).all():
            raise ValueError("Isolated-word checkpoint failed inference self-test")
        return adapter

    def promotion_eligible(self) -> bool:
        evaluation = self.metadata.get("evaluation", {})
        test = evaluation.get("test", {})
        recalls = test.get("per_class_recall", {})
        return bool(
            self.metadata.get("promotion_eligible") is True
            and evaluation.get("status") == "GO"
            and evaluation.get("tests_passed") is True
            and evaluation.get("signer_independent") is True
            and test.get("accuracy", 0.0) >= PROMOTION_MIN_ACCURACY
            and test.get("macro_recall", 0.0) >= PROMOTION_MIN_MACRO_RECALL
            and len(recalls) == len(self.labels)
            and min(recalls.values(), default=0.0) >= PROMOTION_MIN_CLASS_RECALL
        )

    def transcribe(
        self, landmarks_xyz: np.ndarray, min_motion: float = 0.002, min_confidence: float = 0.0
    ):
        from app.ml.adapters.sign_to_text_adapter import TranscriptionResult
        from app.ml.preprocessing.sequence import compute_motion_energy

        features = prepare_isolated_features(landmarks_xyz)
        motion = compute_motion_energy(landmarks_xyz)
        if motion < min_motion:
            return TranscriptionResult("", False, "insufficient_motion", motion, 0.0)
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(features[None]).to(self.device))
            scores = torch.softmax(logits, dim=-1)[0]
        score, index = scores.max(dim=0)
        confidence = float(score.item())
        if confidence < max(min_confidence, self.confidence_threshold):
            return TranscriptionResult("", False, "low_model_confidence", motion, confidence)
        return TranscriptionResult(self.labels[int(index)], True, "ok", motion, confidence)
