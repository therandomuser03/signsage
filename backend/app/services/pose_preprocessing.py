from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np


FULL_KEYPOINTS = 543
COORDS_PER_KEYPOINT = 3
FULL_FEATURE_DIM = FULL_KEYPOINTS * COORDS_PER_KEYPOINT

# The first 468 landmarks are the face. The remaining 75 are pose + hands:
# pose 33, left hand 21, right hand 21.
FACE_END = 468
NOFACE_KEYPOINTS = list(range(FACE_END, FULL_KEYPOINTS))
NOFACE_FEATURE_INDICES = np.array(
    [3 * i + j for i in NOFACE_KEYPOINTS for j in range(3)],
    dtype=np.int64,
)
NOFACE_FEATURE_DIM = int(NOFACE_FEATURE_INDICES.size)


@dataclass(frozen=True)
class ProcessedPose:
    features: np.ndarray
    true_length: int
    feature_dim: int


class PoseNormalizationStats:
    def __init__(self) -> None:
        self.total_sequences = 0
        self.bad_center_sequences = 0
        self.degenerate_scale_sequences = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "total_sequences": self.total_sequences,
            "bad_center_sequences": self.bad_center_sequences,
            "degenerate_scale_sequences": self.degenerate_scale_sequences,
        }


POSE_NORMALIZATION_STATS = PoseNormalizationStats()


def validate_landmark_array(points: np.ndarray, *, allow_empty: bool = False) -> np.ndarray:
    x = np.asarray(points, dtype=np.float32)

    if x.ndim == 2:
        if x.shape != (FULL_KEYPOINTS, COORDS_PER_KEYPOINT):
            raise ValueError(
                f"Expected one frame with shape (543, 3), received {tuple(x.shape)}."
            )
        x = x[None, ...]

    if x.ndim != 3 or x.shape[1:] != (FULL_KEYPOINTS, COORDS_PER_KEYPOINT):
        raise ValueError(
            f"Expected [frames, 543, 3], received shape={tuple(x.shape)}."
        )

    if not allow_empty and x.shape[0] == 0:
        raise ValueError("Empty landmark sequence.")

    if not np.isfinite(x).all():
        raise ValueError("Landmarks contain NaN or infinite values.")

    return np.ascontiguousarray(x, dtype=np.float32)


def normalize_pose_sequence(points: np.ndarray) -> np.ndarray:
    """Exact training-time root-center + body-scale normalization.

    Source of truth: `normalize_pose_sequence()` in research/notebooks/signsage.ipynb.
    """
    x = validate_landmark_array(points)

    POSE_NORMALIZATION_STATS.total_sequences += 1

    hip = x[:, [23, 24], :]
    center = hip.mean(axis=1, keepdims=True)

    bad_center = ~np.isfinite(center).all(axis=(1, 2))
    if bad_center.any():
        # Matches the notebook's counter exactly: one increment per sequence
        # that had >=1 bad-center frame, not one increment per bad frame.
        POSE_NORMALIZATION_STATS.bad_center_sequences += 1
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            center[bad_center] = np.nanmean(
                x[bad_center, :33, :], axis=1, keepdims=True
            )

    x = x - center
    body = x[:, :33, :]
    scale_per_frame = np.nanstd(body.reshape(body.shape[0], -1), axis=1)
    valid_scale = scale_per_frame[
        np.isfinite(scale_per_frame) & (scale_per_frame > 1e-6)
    ]

    if valid_scale.size == 0:
        POSE_NORMALIZATION_STATS.degenerate_scale_sequences += 1
        scale = 1.0
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            scale = np.nanmedian(valid_scale)
        if not np.isfinite(scale) or scale <= 1e-6:
            POSE_NORMALIZATION_STATS.degenerate_scale_sequences += 1
            scale = 1.0

    x = x / float(scale)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.reshape(x.shape[0], FULL_FEATURE_DIM).astype(np.float32)


def resize_sequence(seq: np.ndarray, target_len: int) -> tuple[np.ndarray, int]:
    if target_len <= 0:
        raise ValueError("target_len must be positive.")

    src_len = int(seq.shape[0])
    if src_len <= 0:
        raise ValueError("Empty pose sequence.")

    true_len = min(src_len, target_len)

    if src_len == target_len:
        return seq.astype(np.float32), true_len

    if src_len > target_len:
        idx = np.linspace(0, src_len - 1, target_len)
        return seq[np.round(idx).astype(int)].astype(np.float32), true_len

    out = np.zeros((target_len, seq.shape[1]), dtype=np.float32)
    out[:src_len] = seq
    return out, true_len


def prepare_landmarks(
    landmarks: np.ndarray,
    *,
    max_frames: int,
    no_face: bool = False,
    input_mode: str = "normalized",
) -> ProcessedPose:
    """Validate, optionally normalize, feature-select, and resize a sequence."""
    if input_mode not in {"normalized", "raw"}:
        raise ValueError("input_mode must be 'normalized' or 'raw'.")

    x = validate_landmark_array(landmarks)

    if input_mode == "raw":
        normalized = normalize_pose_sequence(x)
    else:
        # Contract says client supplies training-compatible normalized landmarks.
        normalized = x.reshape(x.shape[0], FULL_FEATURE_DIM).astype(np.float32)

    if no_face:
        normalized = normalized[:, NOFACE_FEATURE_INDICES]

    resized, true_len = resize_sequence(normalized, max_frames)
    return ProcessedPose(
        features=np.ascontiguousarray(resized, dtype=np.float32),
        true_length=true_len,
        feature_dim=int(resized.shape[1]),
    )
