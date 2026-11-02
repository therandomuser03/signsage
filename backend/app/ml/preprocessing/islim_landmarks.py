"""MediaPipe Holistic (543 x 3) -> ISLiM (T, 94, 4) landmark conversion.

ISLiM contract (verified in research/evaluation/islim_inference_verification/):
  - 94 keypoints, 4 dims (x, y, z, visibility)
  - per-coordinate min/max normalization over the sequence (x, y, z); visibility
    carried raw (0 = not detected, 1 = detected)
  - pad / uniform-sample to max_frames=128 (handled by the encoder, but we
    normalize + select here)
  - model input: (B, T, 94, 4) or (B, 4, 94, T)

SignSage frontend/backend produce 543 MediaPipe Holistic landmarks per frame
(component order: pose 0-32, face 33-500, left hand 501-521, right hand 522-542)
with coordinates only (x, y, z) -- no visibility channel.

Only pose + hands are extracted by the available MediaPipe models
(`pose_landmarker_lite.task`, `hand_landmarker.task`); no `face_landmarker.task`
is present locally, so face slots are populated as not-detected (visibility 0).
This is an honest, documented approximation: the exact ISLiM keypoint ordering
ships with the iSign `.npy` preprocessing at an external path
(`/home/shoaib/phd/Datasets/iSign/misc/iSign_Keypoints/{uid}.npy`) that is not
available in this checkout, so a production deployment must replace the default
94-index selection below with the verified ISLiM ordering before claiming
translation accuracy.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

TOTAL_LANDMARKS = 543
ISLI_KEYPOINTS = 94
ISLI_DIMS = 4  # x, y, z, visibility
MAX_FRAMES = 128

# Canonical MediaPipe Holistic component layout (matches
# backend/app/ml/preprocessing/landmarks.py).
POSE_INDICES = list(range(0, 33))            # 33
LEFT_HAND_INDICES = list(range(501, 522))    # 21
RIGHT_HAND_INDICES = list(range(522, 543))   # 21
# Face landmarks chosen for ISL facial grammar / mouthing (eyes + mouth).
# All are valid MediaPipe face-mesh indices (< 468). ORDER IS NOT the verified
# ISLiM ordering; replace with the iSign-derived ordering when available.
FACE_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                291, 308, 324, 318, 402, 13, 82, 33, 263]  # 19

# Default 94-index selection: pose + face + left hand + right hand = 94.
DEFAULT_ISLI_INDICES: list[int] = (
    POSE_INDICES + FACE_INDICES + LEFT_HAND_INDICES + RIGHT_HAND_INDICES
)

assert len(DEFAULT_ISLI_INDICES) == ISLI_KEYPOINTS, len(DEFAULT_ISLI_INDICES)
assert max(DEFAULT_ISLI_INDICES) < TOTAL_LANDMARKS


def select_islim_indices(indices: Sequence[int] | None = None) -> list[int]:
    """Return the global landmark indices used to build the 94-keypoint input."""
    if indices is None:
        return list(DEFAULT_ISLI_INDICES)
    idx = [int(i) for i in indices]
    if len(idx) != ISLI_KEYPOINTS:
        raise ValueError(f"ISLiM requires exactly {ISLI_KEYPOINTS} indices, got {len(idx)}")
    if any(i < 0 or i >= TOTAL_LANDMARKS for i in idx):
        raise ValueError(f"landmark index out of range [0, {TOTAL_LANDMARKS})")
    return idx


def _detected_mask(arr_94x3: np.ndarray) -> np.ndarray:
    """1.0 where a landmark has any non-zero coordinate, 0.0 otherwise.

    arr_94x3: (T, 94, 3) -> returns (T, 94) per-keypoint detection mask.
    """
    return (np.abs(arr_94x3).sum(axis=-1) > 0).astype(np.float32)


def _minmax_normalize(coords: np.ndarray) -> np.ndarray:
    """Per-coordinate (x/y/z) min/max over the whole sequence; degenerate axis -> zeros.

    coords: (T, 94, 3). Returns (T, 94, 3) with each coordinate scaled to [0,1]
    using the min/max of that coordinate across all frames and keypoints.
    """
    out = np.zeros_like(coords, dtype=np.float32)
    for c in range(coords.shape[2]):  # x, y, z
        col = coords[:, :, c]
        cmin, cmax = float(col.min()), float(col.max())
        if cmax > cmin:
            out[:, :, c] = ((col - cmin) / (cmax - cmin)).astype(np.float32)
        else:
            out[:, :, c] = (col - cmin).astype(np.float32)
    return out


def convert_landmarks_543_to_islim(
    arr_543x3: np.ndarray,
    max_frames: int = MAX_FRAMES,
    indices: Sequence[int] | None = None,
) -> np.ndarray:
    """Convert (T, 543, 3) MediaPipe landmarks to (T, 94, 4) ISLiM input.

    Args:
        arr_543x3: (T, 543, 3) float array of [x, y, z] per landmark.
        max_frames: pad / uniform-sample to this many frames (ISLiM: 128).
        indices: optional 94 global landmark indices (defaults to pose+face+hands).

    Returns:
        (T, 94, 4) float32 array: [x, y, z, visibility] per keypoint, with
        x/y/z min/max normalized over the sequence and visibility in {0, 1}.
    """
    arr = np.asarray(arr_543x3, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (TOTAL_LANDMARKS, 3):
        raise ValueError(
            f"Expected (T, {TOTAL_LANDMARKS}, 3), received {arr.shape}"
        )
    if arr.shape[0] < 2:
        raise ValueError(f"Need at least 2 frames, received {arr.shape[0]}")

    sel = select_islim_indices(indices)
    selected = arr[:, sel, :]  # (T, 94, 3)

    xyz = _minmax_normalize(selected[..., :3])
    visibility = _detected_mask(selected)
    out = np.concatenate([xyz, visibility[..., None]], axis=-1)  # (T, 94, 4)

    T = out.shape[0]
    if T > max_frames:
        idx = np.linspace(0, T - 1, max_frames).round().astype(np.int64)
        out = out[idx]
    elif T < max_frames:
        pad = np.zeros((max_frames - T, ISLI_KEYPOINTS, ISLI_DIMS), dtype=np.float32)
        out = np.concatenate([out, pad], axis=0)
    return out


def to_islim_batch(arr_543x3: np.ndarray, max_frames: int = MAX_FRAMES,
                   indices: Sequence[int] | None = None) -> np.ndarray:
    """Convenience: (T, 543, 3) -> (1, T, 94, 4) batched for the ISLiM adapter."""
    return convert_landmarks_543_to_islim(arr_543x3, max_frames=max_frames,
                                          indices=indices)[None, ...]
