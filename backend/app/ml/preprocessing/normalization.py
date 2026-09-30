"""Pose normalization -- verbatim port of the training notebook's
normalize_pose_sequence() (signsage.ipynb, Section 5a / the notebook's own
exported signsage_inference/preprocessing.py).

Root-centers every frame on the hip midpoint and scales by the per-sequence
body-landmark spread. This MUST stay byte-for-byte identical to the
training-time implementation, or live inference will silently diverge from
what the model was actually trained on.
"""
from __future__ import annotations

import warnings

import numpy as np

from .landmarks import (
    BODY_LANDMARK_COUNT,
    COORDINATES_PER_LANDMARK,
    HIP_LANDMARK_INDICES,
    TOTAL_LANDMARKS,
)


def normalize_pose_sequence(flat: np.ndarray) -> np.ndarray:
    """Root-center and scale one pose sequence.

    Args:
        flat: (frames, 543*3) raw landmark coordinates.
    Returns:
        float32 (frames, 543*3) normalized coordinates.
    """
    x = flat.reshape(flat.shape[0], TOTAL_LANDMARKS, COORDINATES_PER_LANDMARK).astype(np.float32)

    hip = x[:, list(HIP_LANDMARK_INDICES), :]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        center = np.nanmean(hip, axis=1, keepdims=True)

    bad_center = ~np.isfinite(center).all(axis=(1, 2))
    if bad_center.any():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            fallback_center = np.nanmean(x[bad_center, :BODY_LANDMARK_COUNT, :], axis=1, keepdims=True)
        center[bad_center] = fallback_center

    x = x - center

    body = x[:, :BODY_LANDMARK_COUNT, :]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Degrees of freedom")
        scale_per_frame = np.nanstd(body.reshape(body.shape[0], -1), axis=1)

    valid_scale = scale_per_frame[np.isfinite(scale_per_frame) & (scale_per_frame > 1e-6)]
    if valid_scale.size == 0:
        scale = 1.0
    else:
        scale = np.nanmedian(valid_scale)
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = 1.0

    x = x / float(scale)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.reshape(x.shape[0], -1).astype(np.float32)
