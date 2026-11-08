"""research/track_b/features_v2.py

Track B step 2 (research only, S1): v2 hand-centric per-frame features.
Does NOT edit features.py. Offline, CPU, SignSage (T,543,3) layout only.

Per-frame vector (122 dims), x,y only:
  pose 0-32 | face 33-500 | left hand 501-521 | right hand 522-542
  Face mesh index i lives at absolute index 33 + i.

  Left hand  (42): [0,1]   wrist - shoulder_centre / shoulder_width
                   [2..41] other 20 landmarks - wrist / hand_size
                   (hand_size = ||wrist - middle fingertip||)
  Right hand (42): same at [42..83]
  [84] left_mask, [85] right_mask            1 if the hand has data after gap-fill
  Hand-to-face (32), scaled by shoulder width:
    per hand (L then R), wrist then index_tip, each -> nose,chin,mouth,forehead (x,y each)
    [86..117]
  Wrist velocity (4), frame difference:
    [118,119] left  wrist dx,dy   [120,121] right wrist dx,dy

Missing-hand handling:
  - A hand is "missing" in a frame when all 21 landmarks are zero.
  - Missing runs of <= MAX_GAP frames are linearly interpolated (per landmark).
  - Runs still missing after that (long gaps / leading / trailing) keep zero
    coordinates AND mask=0. Every hand-derived channel is multiplied by the mask,
    so missing zeros are never normalised into shoulder space.
  - Interpolated frames keep mask=1 (they carry interpolated coordinate data).

Output: (n_frames, 106) float32, resampled to 60 frames AFTER filling.
"""

import numpy as np

N_FRAMES = 60
FEAT_DIM = 122
MAX_GAP = 5          # frames: gaps at most this long are interpolated
EPS = 1e-6

# absolute slices inside the 543 array
_LHAND = slice(501, 522)   # left hand 501-521
_RHAND = slice(522, 543)   # right hand 522-542
# face mesh index i -> absolute 33 + i (face occupies 33-500)
_FACE_NOSE = 33 + 1
_FACE_CHIN = 33 + 152
_FACE_MOUTH = 33 + 13
_FACE_FOREHEAD = 33 + 10
# pose joints (absolute)
_L_SHOULDER = 11
_R_SHOULDER = 12
# hand landmark indices (relative within a 21-hand)
_WRIST = 0
_INDEX_TIP = 8
_MIDDLE_TIP = 12
_OTHER_LANDMARKS = [i for i in range(21) if i != _WRIST]   # 20 landmarks
_FACE_POINTS = (_FACE_NOSE, _FACE_CHIN, _FACE_MOUTH, _FACE_FOREHEAD)


def _xy(landmarks, sl):
    return landmarks[:, sl, :2]                       # (T, n, 2)


def _present(hand_xy):
    """True per frame where the hand has at least one nonzero coordinate."""
    return np.any(hand_xy != 0.0, axis=(1, 2))        # (T,)


def _fill_gaps(hand_xy, max_gap=MAX_GAP):
    """Return (filled (T,21,2), mask (T,)); mask=1 where data exists after fill."""
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
                a = filled[s - 1]                      # (21,2)
                b = filled[e + 1]                      # (21,2)
                frac = ((np.arange(s, e + 1, dtype=np.float64) - (s - 1))
                        / ((e + 1) - (s - 1)))         # (len,)
                for k in range(s, e + 1):
                    f = frac[k - s]
                    filled[k] = a + f * (b - a)
                mask[s:e + 1] = 1.0                    # interpolated => has data
            # else (long gap / leading / trailing): stays zero, mask stays 0
            run_start = None
    return filled, mask


def _shoulder_centre_width(pose):
    """pose: (T,33,3). Returns centre (T,2), width (T,1)."""
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
    """landmarks: (T, N, 3) float. Returns (n_frames, 106) float32."""
    if layout != 'signsage':
        raise ValueError("features_v2 supports layout='signsage' only (543-array)")
    if landmarks.ndim != 3 or landmarks.shape[2] < 2:
        raise ValueError("landmarks must be (T, N, >=3)")
    T = landmarks.shape[0]
    pose = landmarks[:, 0:33, :]          # (T,33,3)
    lh = _xy(landmarks, _LHAND)           # (T,21,2)
    rh = _xy(landmarks, _RHAND)
    face = landmarks[:, 33:500, :2]       # (T,467,2), face mesh index i at 33+i

    centre, width = _shoulder_centre_width(pose)   # (T,2), (T,1)

    l_filled, l_mask = _fill_gaps(lh)
    r_filled, r_mask = _fill_gaps(rh)

    def hand_size(h):
        return np.linalg.norm(h[:, _WRIST, :] - h[:, _MIDDLE_TIP, :],
                              axis=1, keepdims=True) + EPS   # (T,1)

    l_size = hand_size(l_filled)
    r_size = hand_size(r_filled)

    def hand_part(hfilled, hmask, hsize):
        wrist = hfilled[:, _WRIST, :]                      # (T,2)
        others = np.delete(hfilled, _WRIST, axis=1)        # (T,20,2) all landmarks except wrist
        wrist_rel_shoulder = (wrist - centre) / width      # (T,2)
        others_rel_wrist = ((others - wrist[:, None, :])
                            / hsize.reshape(-1, 1, 1))        # (T,20,2)
        # missing frames => mask 0 => all hand-derived channels zero
        wrist_rel_shoulder = wrist_rel_shoulder * hmask[:, None]
        others_rel_wrist = others_rel_wrist * hmask[:, None, None]
        return np.concatenate([wrist_rel_shoulder.reshape(T, -1),
                               others_rel_wrist.reshape(T, -1)], axis=1)  # (T,42)

    lhand = hand_part(l_filled, l_mask, l_size)    # (T,42)
    rhand = hand_part(r_filled, r_mask, r_size)    # (T,42)

    # hand-to-face: per hand, wrist then index_tip, each to nose/chin/mouth/forehead
    h2f = []
    for hp, hmask in ((l_filled[:, _WRIST, :], l_mask),
                      (l_filled[:, _INDEX_TIP, :], l_mask),
                      (r_filled[:, _WRIST, :], r_mask),
                      (r_filled[:, _INDEX_TIP, :], r_mask)):
        for fidx in _FACE_POINTS:
            vec = (hp - face[:, fidx, :]) / width     # (T,2), from hand to face
            h2f.append(vec * hmask[:, None])
    h2f = np.concatenate(h2f, axis=1)                # (T,16)

    def wrist_vel(hfilled, hmask):
        wrist = hfilled[:, _WRIST, :]                # (T,2)
        vel = np.zeros_like(wrist)
        if T > 1:
            vel[1:] = wrist[1:] - wrist[:-1]
        return vel * hmask[:, None]                  # (T,2)

    l_vel = wrist_vel(l_filled, l_mask)              # (T,2)
    r_vel = wrist_vel(r_filled, r_mask)              # (T,2)

    feat = np.concatenate([lhand, rhand,
                           l_mask.reshape(T, 1), r_mask.reshape(T, 1),
                           h2f,
                           l_vel, r_vel], axis=1)   # (T,106)
    assert feat.shape[1] == FEAT_DIM, feat.shape[1]
    feat = _resample(feat, n_frames)
    return feat.astype(np.float32)


def self_test():
    T = 24
    # Build a (T,543,3) array with a moving left hand, static right hand, shoulders.
    h = np.zeros((T, 543, 3))
    # shoulders: lsh=(10,0.5), rsh=(12,0.5) -> centre=(11,0.5), width=2
    h[:, 11, 0] = 10.0
    h[:, 12, 0] = 12.0
    h[:, 11, 1] = 0.5
    h[:, 12, 1] = 0.5
    # left hand: wrist moves x = 0..23, index tip at wrist+(1,0)
    h[:, 501, 0] = np.arange(T, dtype=np.float64)     # l wrist x
    h[:, 501, 1] = 0.5
    h[:, 509, 0] = np.arange(T, dtype=np.float64) + 1.0   # l index tip (8) -> abs 501+8=509
    h[:, 509, 1] = 0.5
    # right hand present
    h[:, 522, 0] = 20.0
    h[:, 522, 1] = 0.5
    h[:, 530, 0] = 21.0   # r index tip
    h[:, 530, 1] = 0.5
    # face: nose(abs34)=(11,0.4), chin(abs185)=(11,0.7)
    h[:, 34, 0] = 11.0;  h[:, 34, 1] = 0.4
    h[:, 185, 0] = 11.0; h[:, 185, 1] = 0.7

    f = extract_features(h, 'signsage')
    assert f.shape == (N_FRAMES, FEAT_DIM), f.shape
    assert not np.isnan(f).any(), "NaN in features"
    assert not np.isinf(f).any(), "Inf in features"
    # left wrist col 0 at frame 0: (0 - 11)/2 = -5.5 ; frame T-1: (23-11)/2 = 6 (before resample)
    # right wrist col 42: (20-11)/2 = 4.5
    masks = f[:, 84:86]
    assert np.all(masks == 1.0), masks[:5]   # both hands present all frames
    print("features_v2.self_test: basic shape/mask OK")

    # --- all-missing-hand case ---
    h2 = np.zeros((T, 543, 3))
    h2[:, 11, 0] = 10.0; h2[:, 12, 0] = 12.0
    h2[:, 11, 1] = 0.5;  h2[:, 12, 1] = 0.5
    h2[:, 34, 0] = 11.0; h2[:, 34, 1] = 0.4
    h2[:, 185, 0] = 11.0; h2[:, 185, 1] = 0.7
    # hands stay all-zero
    f2 = extract_features(h2, 'signsage')
    assert f2.shape == (N_FRAMES, FEAT_DIM)
    assert not np.isnan(f2).any()
    assert np.allclose(f2, 0.0), f2[np.abs(f2) > 0][:5]   # everything zero (masks 0)
    assert np.all(f2[:, 84] == 0.0) and np.all(f2[:, 85] == 0.0)
    print("features_v2.self_test: all-missing-hand -> all zeros, masks 0 OK")

    # --- gap interpolation: gap of 5 filled, gap of 6 stays missing ---
    h3 = np.zeros((20, 543, 3))
    h3[:, 11, 0] = 10.0; h3[:, 12, 0] = 12.0
    h3[:, 11, 1] = 0.5; h3[:, 12, 1] = 0.5
    h3[:, 34, 0] = 11.0; h3[:, 34, 1] = 0.4
    h3[:, 185, 0] = 11.0; h3[:, 185, 1] = 0.7
    # left wrist x: present at frames 0 and 6, missing 1-5 (gap 5 -> interpolate)
    h3[0, 501, 0] = 0.0;  h3[0, 501, 1] = 0.5
    h3[6, 501, 0] = 10.0; h3[6, 501, 1] = 0.5
    # left wrist x: present at frames 8 and 15, missing 9-14 (gap 6 -> stays missing)
    h3[8, 501, 0] = 100.0; h3[8, 501, 1] = 0.5
    h3[15, 501, 0] = 100.0; h3[15, 501, 1] = 0.5
    f3 = extract_features(h3, 'signsage', n_frames=20)
    lwrist_x = f3[:, 0] * 2.0 + 11.0   # undo normalisation: x = val*width + centre
    # frames 1-5 interpolated between 0 and 10
    for t in range(1, 6):
        expected = 0.0 + (t - 0) / 6.0 * 10.0
        assert abs(lwrist_x[t] - expected) < 1e-4, (t, lwrist_x[t], expected)
        assert f3[t, 84] == 1.0        # interpolated => mask 1
    # frames 9-14 still missing => zero coords => wrist_rel_shoulder zeroed by mask
    for t in range(9, 15):
        assert f3[t, 0] == 0.0, (t, f3[t, 0])
        assert f3[t, 84] == 0.0        # mask 0
    print("features_v2.self_test: gap interpolation OK (gap5 filled, gap6 missing)")
    print("features_v2.self_test: ALL OK")


if __name__ == "__main__":
    self_test()
