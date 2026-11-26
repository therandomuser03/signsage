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
  5. Run sign_features_v3's feature code on the result. The default (robust=True) differs from v3 in ONE place:
     which hand is "active" is decided from frames where the hand was seen (v3 counts a missed frame as a jump to
     the image corner, so a hand with a few dropouts looks like the signing hand). Without missed hands the two
     give identical features.

Any change here must be mirrored in the training code and the golden-vector test.
"""
from __future__ import annotations

import numpy as np

from app.ml.preprocessing.sign_features_v3 import (
    EPS,
    FEAT_DIM,
    N_FRAMES,
    _FACE_CHIN,
    _FACE_FOREHEAD,
    _FACE_MOUTH,
    _FACE_NOSE,
    _FACE_POINTS,
    _INDEX_TIP,
    _LHAND,
    _MIDDLE_TIP,
    _RHAND,
    _WRIST,
    _fill_gaps,
    _mirror_x,
    _resample,
    _shoulder_centre_width,
    _should_mirror,
    _track_hands,
    _xy,
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


# ==========================================================================
# Robust variant of extract_features (sign_features_v4 revision 2; v3 above is unchanged).
# Fixes the two ways v3 mishandles MISSED hands: active-hand choice and mirroring. Identical when no hand is missed.
# ==========================================================================
def _active_secondary_robust(phys, wrists):
    """Which hand is the signing (active) hand. Same rule as v3 (larger wrist travel; ties by presence) but travel is
    measured ONLY between consecutive frames in which the hand was actually seen. v3 measures it on arrays where a
    missed frame is stored as zeros, so every miss adds two fake jumps of about |wrist| and a hand with a few
    dropouts looks like it moves a lot: the wrong hand is chosen and the whole sign may be mirrored."""
    travel = np.zeros(2)
    n_present = np.zeros(2, dtype=int)
    for s in (0, 1):
        present = np.any(wrists[:, s] != 0.0, axis=1)
        n_present[s] = int(present.sum())
        both = present[1:] & present[:-1]
        if both.any():
            step = np.linalg.norm(np.diff(wrists[:, s], axis=0), axis=1)
            travel[s] = float(np.sum(step[both]))
    if travel[0] > travel[1]:
        return 0
    if travel[1] > travel[0]:
        return 1
    return 1 if n_present[1] > n_present[0] else 0


def _mirror_x_present(arr, mid_x):
    """Mirror x about the shoulder centre for landmarks that were DETECTED only. v3 also mirrors the zeros that mean
    'not detected': x = 0 becomes 2 * mid_x, which is non-zero, so every missed hand frame turned into a phantom hand
    at the top of the image after mirroring."""
    out = arr.copy()
    seen = np.any(arr != 0.0, axis=2)                         # (T, K): landmark has any non-zero coordinate
    mirrored = 2.0 * mid_x[:, None] - arr[:, :, 0]
    out[:, :, 0] = np.where(seen, mirrored, 0.0)
    return out


def extract_features_robust(landmarks, layout='signsage', n_frames=N_FRAMES):
    if layout != 'signsage':
        raise ValueError("sign_features_v3 supports layout='signsage' only (543-array)")
    if landmarks.ndim != 3 or landmarks.shape[2] < 2:
        raise ValueError("landmarks must be (T, N, >=3)")
    T = landmarks.shape[0]
    pose = landmarks[:, 0:33, :]
    lh = _xy(landmarks, _LHAND)
    rh = _xy(landmarks, _RHAND)
    face = landmarks[:, 33:500, :2]

    centre, width = _shoulder_centre_width(pose)
    phys, wrists = _track_hands(lh, rh)
    active = _active_secondary_robust(phys, wrists)
    secondary = 1 - active

    mirror = _should_mirror(phys, active, wrists, centre)
    if mirror:
        mid_x = centre[:, 0]
        lh = _mirror_x_present(lh, mid_x)
        rh = _mirror_x_present(rh, mid_x)
        face = _mirror_x_present(face, mid_x)
        phys, wrists = _track_hands(lh, rh)
        active = _active_secondary_robust(phys, wrists)
        secondary = 1 - active

    def hand_part(hfilled, hmask, hsize):
        wrist = hfilled[:, _WRIST, :]
        others = np.delete(hfilled, _WRIST, axis=1)
        wrs = (wrist - centre) / width
        orw = ((others - wrist[:, None, :]) / hsize.reshape(-1, 1, 1))
        wrs = wrs * hmask[:, None]
        orw = orw * hmask[:, None, None]
        return np.concatenate([wrs.reshape(T, -1), orw.reshape(T, -1)], axis=1)

    slot_lm = [np.zeros((T, 21, 2), dtype=np.float64),
               np.zeros((T, 21, 2), dtype=np.float64)]
    for t, phys_t in enumerate(phys):
        for s in (0, 1):
            if phys_t[s] is not None:
                slot_lm[s][t] = phys_t[s]

    l_filled, l_mask = _fill_gaps(slot_lm[0])
    r_filled, r_mask = _fill_gaps(slot_lm[1])

    def hand_size(hfilled):
        return np.linalg.norm(hfilled[:, _WRIST, :] - hfilled[:, _MIDDLE_TIP, :],
                              axis=1, keepdims=True) + EPS

    active_filled = l_filled if active == 0 else r_filled
    active_mask = l_mask if active == 0 else r_mask
    active_size = hand_size(active_filled)
    sec_filled = l_filled if secondary == 0 else r_filled
    sec_mask = l_mask if secondary == 0 else r_mask
    sec_size = hand_size(sec_filled)

    active_hand = hand_part(active_filled, active_mask, active_size)
    sec_hand = hand_part(sec_filled, sec_mask, sec_size)

    h2f = []
    for hp, hmask in ((active_filled[:, _WRIST, :], active_mask),
                      (active_filled[:, _INDEX_TIP, :], active_mask),
                      (sec_filled[:, _WRIST, :], sec_mask),
                      (sec_filled[:, _INDEX_TIP, :], sec_mask)):
        for fidx in _FACE_POINTS:
            h2f.append(((hp - face[:, fidx, :]) / width) * hmask[:, None])
    h2f = np.concatenate(h2f, axis=1)

    def wrist_vel(hfilled, hmask):
        wrist = hfilled[:, _WRIST, :]
        vel = np.zeros_like(wrist)
        if T > 1:
            vel[1:] = wrist[1:] - wrist[:-1]
        return vel * hmask[:, None]

    active_vel = wrist_vel(active_filled, active_mask)
    sec_vel = wrist_vel(sec_filled, sec_mask)

    feat = np.concatenate([active_hand, sec_hand,
                           active_mask.reshape(T, 1), sec_mask.reshape(T, 1),
                           h2f,
                           active_vel, sec_vel], axis=1)
    assert feat.shape[1] == FEAT_DIM, feat.shape[1]
    feat = _resample(feat, n_frames)
    return feat.astype(np.float32)


def extract_features_v4(arr: np.ndarray, aspect: float | None, *, n_frames: int = EXTRACT_FRAMES,
                        forehead_k: float = FOREHEAD_K, chin_k: float = CHIN_K, robust: bool = True) -> np.ndarray:
    """(T, 543, 3) raw landmarks -> (60, 122) float32 features for the FDMSE-ISL models.

    robust=True (default) selects the signing hand from frames where the hand was actually seen; it is identical to
    the legacy sign_features_v3 behaviour when no hand is missed, and far more stable when a webcam misses hands.
    robust=False reproduces the legacy behaviour exactly (used by the golden test of features built without it)."""
    prepared = prepare_landmarks(arr, aspect, n_frames=n_frames, forehead_k=forehead_k, chin_k=chin_k)
    return extract_features_robust(prepared) if robust else _extract_v3(prepared, layout="signsage")


__all__ = ["EXTRACT_FRAMES", "FEAT_DIM", "N_FRAMES", "extract_features_robust", "extract_features_v4",
           "prepare_landmarks", "pose_face_reference", "resample_landmarks"]
