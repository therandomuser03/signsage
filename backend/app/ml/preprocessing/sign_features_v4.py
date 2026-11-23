"""sign_features_v4 - serving-side recipe for the FDMSE-ISL word models.

The model was trained on features built by this exact recipe (see docs/MODELS_AND_DATA.md):

  1. Resample the captured landmark sequence to EXACTLY `n_frames` (48) frames, uniformly over the
     capture. sign_features_v3 pads a shorter sequence by repeating the LAST frame, and its wrist
     velocities are per frame step, so feeding it ~25 live frames instead of the 48 it was trained
     on would change both the padding and the velocity scale.
  2. Fill frames where the pose is missing from the nearest frame that has it.
  3. Multiply x by the video aspect ratio (width / height) so x and y share one unit.
  4. Derive the four face reference points (nose, chin, mouth, forehead) from the POSE head points.
     The face mesh is deliberately ignored: MediaPipe never found a face in the training videos.
  5. Run the unchanged sign_features_v3 on the result.

Any change here must be mirrored in the training code and the golden-vector test.
"""
from __future__ import annotations

import numpy as np

from app.ml.preprocessing.sign_features_v3 import (
    FEAT_DIM,
    N_FRAMES,
    _FACE_CHIN,
    _FACE_FOREHEAD,
    _FACE_MOUTH,
    _FACE_NOSE,
    extract_features as _extract_v3,
)

EXTRACT_FRAMES = 48          # frames the training pipeline sampled per clip
FOREHEAD_K = 1.2
CHIN_K = 0.6
_MAX_LINEAR_JUMP = 0.3       # do not interpolate a hand that "jumped" > 30% of the frame between samples

_POSE = slice(0, 33)
_FACE = slice(33, 501)
_LHAND = slice(501, 522)
_RHAND = slice(522, 543)


def _block_valid(block: np.ndarray) -> np.ndarray:
    """(T, K, 3) -> (T,) True where the block has any non-zero coordinate (zeros = not detected)."""
    return np.any(block != 0.0, axis=(1, 2))


def resample_landmarks(arr: np.ndarray, n: int = EXTRACT_FRAMES) -> np.ndarray:
    """Uniformly resample (T, 543, 3) to (n, 543, 3).

    Each body block (pose, left hand, right hand) is interpolated linearly only when it is present in
    both neighbouring frames and did not jump; otherwise the nearest frame is used, and a block that is
    missing in the nearest frame stays missing (never invented). T == n returns an exact copy."""
    t_in = arr.shape[0]
    if t_in == n:
        return np.array(arr, dtype=np.float32, copy=True)
    out = np.zeros((n, 543, 3), np.float32)
    t = np.linspace(0.0, t_in - 1, n)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, t_in - 1)
    w = (t - i0).astype(np.float32)
    near_idx = np.where(w < 0.5, i0, i1)
    for sl in (_POSE, _LHAND, _RHAND):
        blk = arr[:, sl]
        valid = _block_valid(blk)
        a, b = blk[i0], blk[i1]
        jump = np.abs(a[..., :2] - b[..., :2]).mean(axis=(1, 2))
        linear = valid[i0] & valid[i1] & (jump <= _MAX_LINEAR_JUMP)
        blended = (1.0 - w)[:, None, None] * a + w[:, None, None] * b
        nearest = np.where(valid[near_idx][:, None, None], blk[near_idx], 0.0)
        out[:, sl] = np.where(linear[:, None, None], blended, nearest)
    return out


def _fill_missing_pose(pose: np.ndarray) -> np.ndarray:
    ok = _block_valid(pose)
    if ok.all() or not ok.any():
        return pose
    idx = np.flatnonzero(ok)
    t = np.arange(len(pose))
    return pose[idx[np.abs(t[:, None] - idx[None, :]).argmin(axis=1)]]


def pose_face_reference(pose_xy: np.ndarray, forehead_k: float = FOREHEAD_K, chin_k: float = CHIN_K) -> dict:
    """pose_xy: (T, >=11, >=2) -> {face_mesh_index: (T, 2)} for the four points sign_features_v3 reads.
    Pose indices: 0 nose, 2/5 eye centres, 9/10 mouth corners."""
    nose = pose_xy[:, 0, :2]
    eye = (pose_xy[:, 2, :2] + pose_xy[:, 5, :2]) / 2.0
    mouth = (pose_xy[:, 9, :2] + pose_xy[:, 10, :2]) / 2.0
    axis = eye - mouth
    return {_FACE_NOSE: nose, _FACE_CHIN: mouth - chin_k * axis,
            _FACE_MOUTH: mouth, _FACE_FOREHEAD: eye + forehead_k * axis}


def prepare_landmarks(arr: np.ndarray, aspect: float | None, *, n_frames: int = EXTRACT_FRAMES,
                      forehead_k: float = FOREHEAD_K, chin_k: float = CHIN_K) -> np.ndarray:
    """(T, 543, 3) raw browser landmarks -> (n_frames, 543, 3) laid out as in training."""
    out = resample_landmarks(np.asarray(arr, dtype=np.float32), n_frames)
    out[:, _FACE] = 0.0                                   # the live face mesh is never used
    out[:, _POSE] = _fill_missing_pose(out[:, _POSE])
    if aspect:
        out[:, :, 0] *= float(aspect)                     # zeros (missing landmarks) stay zero
    for idx, pts in pose_face_reference(out[:, _POSE], forehead_k, chin_k).items():
        out[:, 33 + idx, :2] = pts
    return out


def extract_features_v4(arr: np.ndarray, aspect: float | None, *, n_frames: int = EXTRACT_FRAMES,
                        forehead_k: float = FOREHEAD_K, chin_k: float = CHIN_K) -> np.ndarray:
    """(T, 543, 3) raw landmarks -> (60, 122) float32 features for the FDMSE-ISL models."""
    prepared = prepare_landmarks(arr, aspect, n_frames=n_frames, forehead_k=forehead_k, chin_k=chin_k)
    return _extract_v3(prepared, layout="signsage")


__all__ = ["EXTRACT_FRAMES", "FEAT_DIM", "N_FRAMES", "extract_features_v4", "prepare_landmarks",
           "pose_face_reference", "resample_landmarks"]
