"""sign_features_v3.py

Standalone, numpy-only re-implementation of research/track_b/features_v3.py.
Same output: (n_frames, 122) float32 per-frame hand-centric features for a
SignSage (T,543,3) landmark array.

MediaPipe Left/Right handedness labels are IGNORED (camera is unmirrored).
  (a) Detected hands (both slots, labels ignored) are assigned to a stable
      physical identity by temporal tracking (nearest wrist to the previous
      frame; two hands in one frame ordered by image x, then continuity).
  (b) ACTIVE hand per clip = larger total wrist travel (tie: more frames
      present); the other is SECONDARY. Features use ACTIVE / SECONDARY slots.
  (c) If the active hand's mean wrist x is image-left of the shoulder midpoint,
      mirror all x coordinates about the per-frame shoulder midpoint so every
      clip looks right-hand-active.

Per-frame vector (122 dims), x,y only:
  active hand  (42): wrist - shoulder_centre / shoulder_width;
                     other 20 landmarks - wrist / hand_size
  secondary hand (42): same
  active_mask, secondary_mask (2): 1 if the hand has data after gap-fill
  hand-to-face (32): per hand (active then secondary), wrist then index tip,
                     -> nose/chin/mouth/forehead, scaled by shoulder width
  wrist velocity (4): frame difference per wrist

Missing-hand handling:
  - hand missing in a frame when all 21 landmarks are zero
  - missing runs <= MAX_GAP frames linearly interpolated (mask stays 1)
  - runs still missing keep zero coords + mask=0; hand-derived channels are
    multiplied by the mask (missing zeros never normalised into shoulder space)
  - output resampled to N_FRAMES after filling
"""
import numpy as np

N_FRAMES = 60
FEAT_DIM = 122
MAX_GAP = 5
EPS = 1e-6

_LHAND = slice(501, 522)   # MediaPipe slot 0 (label ignored)
_RHAND = slice(522, 543)   # MediaPipe slot 1 (label ignored)
_FACE_NOSE = 1             # face mesh index; face array is landmarks[:,33:500,:2]
_FACE_CHIN = 152
_FACE_MOUTH = 13
_FACE_FOREHEAD = 10
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


def _shoulder_centre_width(pose):
    lsh = pose[:, _L_SHOULDER, :2]
    rsh = pose[:, _R_SHOULDER, :2]
    centre = (lsh + rsh) / 2.0
    width = np.linalg.norm(lsh - rsh, axis=1, keepdims=True) + EPS
    return centre, width


def _mirror_x(arr, mid_x):
    out = arr.copy()
    out[:, :, 0] = 2.0 * mid_x[:, None] - out[:, :, 0]
    return out


def _track_hands(lh_xy, rh_xy):
    T = lh_xy.shape[0]
    lh_present = _present(lh_xy)
    rh_present = _present(rh_xy)
    phys = []
    slots = [None, None]
    for t in range(T):
        detected = []
        if lh_present[t]:
            detected.append((lh_xy[t, _WRIST].copy(), lh_xy[t].copy()))
        if rh_present[t]:
            detected.append((rh_xy[t, _WRIST].copy(), rh_xy[t].copy()))
        phys_t = [None, None]
        if len(detected) == 0:
            pass
        elif len(detected) == 1:
            wrist, lm = detected[0]
            best_s, best_d = None, None
            for s in (0, 1):
                if slots[s] is not None:
                    d = float(np.linalg.norm(wrist - slots[s]["wrist"]))
                    if best_d is None or d < best_d:
                        best_s, best_d = s, d
            if best_s is None:
                best_s = 0
            phys_t[best_s] = lm
            slots[best_s] = {"wrist": wrist, "lm": lm}
        else:
            d0, d1 = detected
            if d0[0][0] > d1[0][0]:
                d0, d1 = d1, d0
            if slots[0] is not None and slots[1] is not None:
                c1 = (np.linalg.norm(d0[0] - slots[0]["wrist"])
                      + np.linalg.norm(d1[0] - slots[1]["wrist"]))
                c2 = (np.linalg.norm(d0[0] - slots[1]["wrist"])
                      + np.linalg.norm(d1[0] - slots[0]["wrist"]))
                if c1 <= c2:
                    phys_t[0], phys_t[1] = d0[1], d1[1]
                    slots[0] = {"wrist": d0[0], "lm": d0[1]}
                    slots[1] = {"wrist": d1[0], "lm": d1[1]}
                else:
                    phys_t[0], phys_t[1] = d1[1], d0[1]
                    slots[0] = {"wrist": d1[0], "lm": d1[1]}
                    slots[1] = {"wrist": d0[0], "lm": d0[1]}
            elif slots[0] is not None:
                if np.linalg.norm(d0[0] - slots[0]["wrist"]) <= np.linalg.norm(d1[0] - slots[0]["wrist"]):
                    phys_t[0], phys_t[1] = d0[1], d1[1]
                    slots[0] = {"wrist": d0[0], "lm": d0[1]}
                    slots[1] = {"wrist": d1[0], "lm": d1[1]}
                else:
                    phys_t[0], phys_t[1] = d1[1], d0[1]
                    slots[0] = {"wrist": d1[0], "lm": d1[1]}
                    slots[1] = {"wrist": d0[0], "lm": d0[1]}
            elif slots[1] is not None:
                if np.linalg.norm(d0[0] - slots[1]["wrist"]) <= np.linalg.norm(d1[0] - slots[1]["wrist"]):
                    phys_t[1], phys_t[0] = d0[1], d1[1]
                    slots[1] = {"wrist": d0[0], "lm": d0[1]}
                    slots[0] = {"wrist": d1[0], "lm": d1[1]}
                else:
                    phys_t[1], phys_t[0] = d1[1], d0[1]
                    slots[1] = {"wrist": d1[0], "lm": d1[1]}
                    slots[0] = {"wrist": d0[0], "lm": d0[1]}
            else:
                phys_t[0], phys_t[1] = d0[1], d1[1]
                slots[0] = {"wrist": d0[0], "lm": d0[1]}
                slots[1] = {"wrist": d1[0], "lm": d1[1]}
        phys.append(phys_t)
    wrists = np.zeros((T, 2, 2), dtype=np.float64)
    for t, phys_t in enumerate(phys):
        for s in (0, 1):
            if phys_t[s] is not None:
                wrists[t, s] = phys_t[s][_WRIST]
    return phys, wrists


def _active_secondary(phys, wrists):
    T = wrists.shape[0]
    travel = np.zeros(2)
    n_present = np.zeros(2, dtype=int)
    for s in (0, 1):
        present = np.any(wrists[:, s] != 0.0, axis=1)
        n_present[s] = int(present.sum())
        if n_present[s] > 1:
            travel[s] = float(np.sum(np.linalg.norm(np.diff(wrists[:, s], axis=0), axis=1)))
    if travel[0] > travel[1]:
        return 0
    if travel[1] > travel[0]:
        return 1
    return 1 if n_present[1] > n_present[0] else 0


def _should_mirror(phys, active, wrists, centre):
    T = wrists.shape[0]
    active_present = np.any(wrists[:, active] != 0.0, axis=1)
    sh_valid = np.any(centre != 0.0, axis=1)
    valid = active_present & sh_valid
    if valid.sum() < 2:
        return False
    return float(np.mean(wrists[valid, active, 0])) < float(np.mean(centre[valid, 0]))


def extract_features(landmarks, layout='signsage', n_frames=N_FRAMES):
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
    active = _active_secondary(phys, wrists)
    secondary = 1 - active

    mirror = _should_mirror(phys, active, wrists, centre)
    if mirror:
        mid_x = centre[:, 0]
        lh = _mirror_x(lh, mid_x)
        rh = _mirror_x(rh, mid_x)
        face = _mirror_x(face, mid_x)
        phys, wrists = _track_hands(lh, rh)
        active = _active_secondary(phys, wrists)
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
