"""Temporal resizing and live-frame preparation -- verbatim port of the
training notebook's resize_sequence()/prepare_live_landmarks()/
live_motion_energy() (signsage.ipynb, Section 5a & Section 15).
"""
from __future__ import annotations

import numpy as np

from .landmarks import assert_landmark_dtype, assert_landmark_shape
from .normalization import normalize_pose_sequence


def resize_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Uniformly downsample long sequences; zero-pad short ones."""
    if not isinstance(target_len, int) or target_len <= 0:
        raise ValueError(f"target_len must be a positive int, received {target_len!r}")
    seq = np.asarray(seq)
    if seq.ndim != 2:
        raise ValueError(f"Expected sequence shaped (T, D), received {seq.shape}")
    src_len = seq.shape[0]
    if src_len <= 0:
        raise ValueError("Empty pose sequence")
    if src_len == target_len:
        return seq.astype(np.float32, copy=False)
    if src_len > target_len:
        idx = np.linspace(0, src_len - 1, target_len)
        return seq[np.round(idx).astype(np.int64)].astype(np.float32, copy=False)
    out = np.zeros((target_len, seq.shape[1]), dtype=np.float32)
    out[:src_len] = seq
    return out


def prepare_live_landmarks(landmarks_xyz: np.ndarray, max_frames: int) -> tuple[np.ndarray, int]:
    """Convert raw (T, 543, 3) camera landmarks into the model's input shape.

    Returns (processed, true_len): processed is (max_frames, 1629) float32,
    true_len is min(T, max_frames) -- the real (unpadded) sequence length the
    model needs for its padding mask. Callers must NOT precompute deltas;
    the model does that internally (see ml/models/sign_to_text.py).

    Raises:
        ValueError: on any incompatible representation (wrong container,
            non-numeric dtype, wrong shape, too few frames, bad max_frames).
    """
    arr = np.asarray(landmarks_xyz)
    assert_landmark_dtype(arr)
    assert_landmark_shape(arr)
    arr = arr.astype(np.float32)
    if not isinstance(max_frames, int) or max_frames < 2:
        raise ValueError(f"max_frames must be an int >= 2, received {max_frames!r}")
    if len(arr) < 2:
        raise ValueError(f"Need at least two frames, received {len(arr)}.")
    flat = arr.reshape(len(arr), -1)
    norm = normalize_pose_sequence(flat)
    true_len = min(len(norm), max_frames)
    return resize_sequence(norm, max_frames), true_len


def compute_motion_energy(landmarks_xyz: np.ndarray) -> float:
    """Mean absolute frame-to-frame delta on normalized landmarks.

    Used as a cheap "is anything actually moving" gate before trusting the
    model's output -- a static/no-signing frame sequence should not produce
    a confident-looking translation.
    """
    arr = np.asarray(landmarks_xyz)
    assert_landmark_dtype(arr)
    assert_landmark_shape(arr)
    arr = arr.astype(np.float32)
    if len(arr) < 2:
        return 0.0
    norm = normalize_pose_sequence(arr.reshape(len(arr), -1))
    return float(np.mean(np.abs(np.diff(norm, axis=0))))
