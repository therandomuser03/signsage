"""sign_features_v2.py

Standalone, numpy-only re-implementation of research/track_b/features_v2.py.
Same output: (n_frames, 122) float32 per-frame hand-centric features for a
SignSage (T,543,3) landmark array (face mesh index i at absolute 33+i).

Per-frame vector (122 dims), x,y only:
  left hand  (42): wrist - shoulder_centre / shoulder_width;
                   other 20 landmarks - wrist / hand_size
  right hand (42): same
  left_mask, right_mask (2): 1 if the hand has data after gap-fill
  hand-to-face (32): per hand (L then R), wrist then index tip, -> nose/chin/mouth/forehead,
                     scaled by shoulder width
  wrist velocity (4): frame difference per wrist

Missing-hand handling:
  - hand missing in a frame when all 21 landmarks are zero
  - missing runs <= MAX_GAP frames linearly interpolated (mask stays 1)
  - runs still missing after that keep zero coords + mask=0; every hand-derived
    channel is multiplied by the mask (missing zeros never normalised into shoulder space)
  - output resampled to N_FRAMES after filling
"""
import numpy as np

N_FRAMES = 60
FEAT_DIM = 122
MAX_GAP = 5
EPS = 1e-6

_LHAND = slice(501, 522)
_RHAND = slice(522, 543)
_FACE_NOSE = 33 + 1
_FACE_CHIN = 33 + 152
_FACE_MOUTH = 33 + 13
_FACE_FOREHEAD = 33 + 10
_L_SHOULDER = 11
_R_SHOULDER = 12
_WRIST = 0
_INDEX_TIP = 8
_MIDDLE_TIP = 12
_FACE_POINTS = (_FACE_NOSE, _FACE_CHIN, _FACE_MOUTH, _FACE_FOREHEAD)


def _xy(landmarks, sl):
    return landmarks[:, sl, :2]


def _present(hand_xy):
    return np.any(hand_xy != 0.0, axis=(1, 2))


def _fill_gaps(hand_xy, max_gap=MAX_GAP):
    T = hand_xy.shape[0]
    filled = hand_xy.copy().astype(np.float64)
    present = _present(hand_xy)
    mask = present.copy().astype(np.float64)
    if T == 0:
        return filled, mask
    miss = ~present
    run_start = None
    for t in range(T + 1):
        if t < T and miss[t] and run_start is None:
            run_start = t
        if (t == T or not miss[t]) and run_start is not None:
            s, e = run_start, t - 1
            length = e - s + 1
            if length <= max_gap and s > 0 and e < T - 1:
                a = filled[s - 1]; b = filled[e + 1]
                frac = ((np.arange(s, e + 1, dtype=np.float64) - (s - 1))
                        / ((e + 1) - (s - 1)))
                for k in range(s, e + 1):
                    fk = frac[k - s]
                    filled[k] = a + fk * (b - a)
                mask[s:e + 1] = 1.0
            run_start = None
    return filled, mask


def _shoulder_centre_width(pose):
    lsh = pose[:, _L_SHOULDER, :2]
    rsh = pose[:, _R_SHOULDER, :2]
    centre = (lsh + rsh) / 2.0
    width = np.linalg.norm(lsh - rsh, axis=1, keepdims=True) + EPS
    return centre, width


def _resample(feat, n_frames):
    T = feat.shape[0]
    if T == 0:
        return np.zeros((n_frames, FEAT_DIM), dtype=np.float32)
    if T == n_frames:
        return feat
    if T > n_frames:
        idx = np.linspace(0, T - 1, n_frames).round().astype(int)
        return feat[idx]
    pad = np.repeat(feat[-1:], n_frames - T, axis=0)
    return np.concatenate([feat, pad], axis=0)


def extract_features(landmarks, layout='signsage', n_frames=N_FRAMES):
    if layout != 'signsage':
        raise ValueError("sign_features_v2 supports layout='signsage' only (543-array)")
    if landmarks.ndim != 3 or landmarks.shape[2] < 2:
        raise ValueError("landmarks must be (T, N, >=3)")
    T = landmarks.shape[0]
    pose = landmarks[:, 0:33, :]
    lh = _xy(landmarks, _LHAND)
    rh = _xy(landmarks, _RHAND)
    face = landmarks[:, 33:500, :2]

    centre, width = _shoulder_centre_width(pose)
    l_filled, l_mask = _fill_gaps(lh)
    r_filled, r_mask = _fill_gaps(rh)

    def hand_size(h):
        return np.linalg.norm(h[:, _WRIST, :] - h[:, _MIDDLE_TIP, :],
                              axis=1, keepdims=True) + EPS

    l_size = hand_size(l_filled)
    r_size = hand_size(r_filled)

    def hand_part(hfilled, hmask, hsize):
        wrist = hfilled[:, _WRIST, :]
        others = np.delete(hfilled, _WRIST, axis=1)
        wrs = (wrist - centre) / width
        orw = ((others - wrist[:, None, :]) / hsize.reshape(-1, 1, 1))
        wrs = wrs * hmask[:, None]
        orw = orw * hmask[:, None, None]
        return np.concatenate([wrs.reshape(T, -1), orw.reshape(T, -1)], axis=1)

    lhand = hand_part(l_filled, l_mask, l_size)
    rhand = hand_part(r_filled, r_mask, r_size)

    h2f = []
    for hp, hmask in ((l_filled[:, _WRIST, :], l_mask),
                      (l_filled[:, _INDEX_TIP, :], l_mask),
                      (r_filled[:, _WRIST, :], r_mask),
                      (r_filled[:, _INDEX_TIP, :], r_mask)):
        for fidx in _FACE_POINTS:
            h2f.append(((hp - face[:, fidx, :]) / width) * hmask[:, None])
    h2f = np.concatenate(h2f, axis=1)

    def wrist_vel(hfilled, hmask):
        wrist = hfilled[:, _WRIST, :]
        vel = np.zeros_like(wrist)
        if T > 1:
            vel[1:] = wrist[1:] - wrist[:-1]
        return vel * hmask[:, None]

    l_vel = wrist_vel(l_filled, l_mask)
    r_vel = wrist_vel(r_filled, r_mask)

    feat = np.concatenate([lhand, rhand,
                           l_mask.reshape(T, 1), r_mask.reshape(T, 1),
                           h2f, l_vel, r_vel], axis=1)
    assert feat.shape[1] == FEAT_DIM, feat.shape[1]
    feat = _resample(feat, n_frames)
    return feat.astype(np.float32)
