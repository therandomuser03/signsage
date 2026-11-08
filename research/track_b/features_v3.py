"""research/track_b/features_v3.py

Track B step 1 (research only): v3 hand-centric per-frame features.
Does NOT edit features.py or features_v2.py. Offline, CPU, SignSage (T,543,3) layout only.

KEY DESIGN DIFFERENCE vs v2: MediaPipe Left/Right handedness labels are IGNORED.
The camera is unmirrored, so handedness flips between sessions and within clips.
Instead:
  (a) Per frame, detected hands (from both slots, labels ignored) are assigned to a
      stable physical identity by temporal tracking:
          - nearest wrist to the previous frame's wrist for that physical hand
          - if two hands appear in one frame, order by image x, then keep continuity
  (b) ACTIVE hand per clip = hand with larger total wrist travel
      (tie: more frames present). The other is SECONDARY.
      Features use ACTIVE / SECONDARY slots instead of L / R.
  (c) If the active hand's mean wrist x is on the image-left of the shoulder
      midpoint, mirror all x coordinates of the clip about the per-frame shoulder
      midpoint (hands and face vectors) so every clip looks right-hand-active.

Per-frame vector (122 dims), x,y only:
  active hand  (42): [0,1]   wrist - shoulder_centre / shoulder_width
                     [2..41] other 20 landmarks - wrist / hand_size
  secondary hand (42): same at [42..83]
  [84] active_mask, [85] secondary_mask          1 if the hand has data after gap-fill
  Hand-to-face (32), scaled by shoulder width:
    per hand (active then secondary), wrist then index_tip, each -> nose/chin/mouth/forehead
    [86..117]
  Wrist velocity (4), frame difference:
    [118,119] active  wrist dx,dy   [120,121] secondary wrist dx,dy

Missing-hand handling (same as v2):
  - A hand is "missing" in a frame when all 21 landmarks are zero.
  - Missing runs of <= MAX_GAP frames are linearly interpolated (per landmark).
  - Runs still missing after that (long gaps / leading / trailing) keep zero
    coordinates AND mask=0. Every hand-derived channel is multiplied by the mask,
    so missing zeros are never normalised into shoulder space.
  - Interpolated frames keep mask=1.
Output: (n_frames, 122) float32, resampled to 60 frames AFTER filling.
"""

import numpy as np

N_FRAMES = 60
FEAT_DIM = 122
MAX_GAP = 5          # frames: gaps at most this long are interpolated
EPS = 1e-6

# absolute slices inside the 543 array
_LHAND = slice(501, 522)   # MediaPipe slot 0 (label ignored)
_RHAND = slice(522, 543)   # MediaPipe slot 1 (label ignored)
# face mesh index i -> absolute 33 + i (face occupies 33-500).
# `face` array is sliced as landmarks[:, 33:500, :2], so row index == face mesh index.
_FACE_NOSE = 1
_FACE_CHIN = 152
_FACE_MOUTH = 13
_FACE_FOREHEAD = 10
# pose joints (absolute)
_L_SHOULDER = 11
_R_SHOULDER = 12
# hand landmark indices (relative within a 21-hand)
_WRIST = 0
_INDEX_TIP = 8
_MIDDLE_TIP = 12
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
                b = filled[e + 1]
                frac = ((np.arange(s, e + 1, dtype=np.float64) - (s - 1))
                        / ((e + 1) - (s - 1)))         # (len,)
                for k in range(s, e + 1):
                    fk = frac[k - s]
                    filled[k] = a + fk * (b - a)
                mask[s:e + 1] = 1.0                    # interpolated => has data
            # else (long gap / leading / trailing): stays zero, mask stays 0
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


# ---------------------------------------------------------------- tracking
def _track_hands(lh_xy, rh_xy):
    """Assign detected hands (labels ignored) to two stable physical slots.

    lh_xy, rh_xy: (T,21,2) from the two MediaPipe slots (Left/Right labels ignored).
    Returns:
      phys: list of T elements; each is [hand0_or_None, hand1_or_None] where each
            hand is (21,2) in ORIGINAL (unmirrored) coords, or None if missing.
      wrists: (T,2,2) wrists for phys[0], phys[1] (zero where missing) - used for travel.
    """
    T = lh_xy.shape[0]
    lh_present = _present(lh_xy)                      # (T,)
    rh_present = _present(rh_xy)

    phys = []                                         # list of [h0, h1]
    # slots[s] = {'wrist':(2,), 'lm':(21,2)} last known hand for physical slot s, or None
    slots = [None, None]

    for t in range(T):
        detected = []                                 # list of (wrist(2,), lm(21,2))
        if lh_present[t]:
            detected.append((lh_xy[t, _WRIST].copy(), lh_xy[t].copy()))
        if rh_present[t]:
            detected.append((rh_xy[t, _WRIST].copy(), rh_xy[t].copy()))

        phys_t = [None, None]
        if len(detected) == 0:
            # both missing this frame; slots keep last-known for continuity
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
                best_s = 0                            # no history: initialise slot 0
            phys_t[best_s] = lm
            slots[best_s] = {"wrist": wrist, "lm": lm}
        else:  # len(detected) == 2
            d0, d1 = detected
            # order by image x (left -> first)
            if d0[0][0] > d1[0][0]:
                d0, d1 = d1, d0
            if slots[0] is not None and slots[1] is not None:
                c1 = (np.linalg.norm(d0[0] - slots[0]["wrist"])
                      + np.linalg.norm(d1[0] - slots[1]["wrist"]))
                c2 = (np.linalg.norm(d0[0] - slots[1]["wrist"])
                      + np.linalg.norm(d1[0] - slots[0]["wrist"]))
                if c1 <= c2:
                    phys_t[0], phys_t[1] = d0[1], d1[1]
                    slots[0], slots[1] = {"wrist": d0[0], "lm": d0[1]}, {"wrist": d1[0], "lm": d1[1]}
                else:
                    phys_t[0], phys_t[1] = d1[1], d0[1]
                    slots[0], slots[1] = {"wrist": d1[0], "lm": d1[1]}, {"wrist": d0[0], "lm": d0[1]}
            elif slots[0] is not None:
                if np.linalg.norm(d0[0] - slots[0]["wrist"]) <= np.linalg.norm(d1[0] - slots[0]["wrist"]):
                    phys_t[0], phys_t[1] = d0[1], d1[1]
                    slots[0], slots[1] = {"wrist": d0[0], "lm": d0[1]}, {"wrist": d1[0], "lm": d1[1]}
                else:
                    phys_t[0], phys_t[1] = d1[1], d0[1]
                    slots[0], slots[1] = {"wrist": d1[0], "lm": d1[1]}, {"wrist": d0[0], "lm": d0[1]}
            elif slots[1] is not None:
                if np.linalg.norm(d0[0] - slots[1]["wrist"]) <= np.linalg.norm(d1[0] - slots[1]["wrist"]):
                    phys_t[1], phys_t[0] = d0[1], d1[1]
                    slots[1], slots[0] = {"wrist": d0[0], "lm": d0[1]}, {"wrist": d1[0], "lm": d1[1]}
                else:
                    phys_t[1], phys_t[0] = d1[1], d0[1]
                    slots[1], slots[0] = {"wrist": d1[0], "lm": d1[1]}, {"wrist": d0[0], "lm": d0[1]}
            else:
                # no history: order by image x -> slot0 = left, slot1 = right
                phys_t[0], phys_t[1] = d0[1], d1[1]
                slots[0], slots[1] = {"wrist": d0[0], "lm": d0[1]}, {"wrist": d1[0], "lm": d1[1]}
        phys.append(phys_t)

    # wrists per slot per frame (zero where missing)
    wrists = np.zeros((T, 2, 2), dtype=np.float64)
    for t, phys_t in enumerate(phys):
        for s in (0, 1):
            if phys_t[s] is not None:
                wrists[t, s] = phys_t[s][_WRIST]
    return phys, wrists


def _active_secondary(phys, wrists):
    """Decide active = larger total wrist travel (tie: more frames present).

    Returns active_slot (0 or 1).
    """
    T = wrists.shape[0]
    travel = np.zeros(2)
    n_present = np.zeros(2, dtype=int)
    for s in (0, 1):
        present = np.any(wrists[:, s] != 0.0, axis=1)   # wrist nonzero == present  (T,)
        n_present[s] = int(present.sum())
        if n_present[s] > 1:
            w = wrists[:, s]                            # (T,2)
            travel[s] = float(np.sum(np.linalg.norm(np.diff(w, axis=0), axis=1)))
    if travel[0] > travel[1]:
        return 0
    if travel[1] > travel[0]:
        return 1
    # tie: more frames present
    if n_present[1] > n_present[0]:
        return 1
    return 0


# ---------------------------------------------------------------- main
def extract_features(landmarks, layout='signsage', n_frames=N_FRAMES):
    """landmarks: (T, N, 3) float. Returns (n_frames, 122) float32.

    Slots are ACTIVE then SECONDARY (MediaPipe L/R labels ignored).
    """
    if layout != 'signsage':
        raise ValueError("features_v3 supports layout='signsage' only (543-array)")
    if landmarks.ndim != 3 or landmarks.shape[2] < 2:
        raise ValueError("landmarks must be (T, N, >=3)")
    T = landmarks.shape[0]
    pose = landmarks[:, 0:33, :]                      # (T,33,3)
    lh = _xy(landmarks, _LHAND)                       # (T,21,2)  slot 0 (label ignored)
    rh = _xy(landmarks, _RHAND)                       # (T,21,2)  slot 1 (label ignored)
    face = landmarks[:, 33:500, :2]                   # (T,467,2)

    centre, width = _shoulder_centre_width(pose)      # (T,2), (T,1)

    phys, wrists = _track_hands(lh, rh)               # temporal tracking, original coords
    active = _active_secondary(phys, wrists)
    secondary = 1 - active

    # --- mirror decision: if active hand's mean wrist x is left of shoulder midpoint
    mirror = _should_mirror(phys, active, wrists, centre)
    if mirror:
        # reflect x about per-frame shoulder midpoint: x' = 2*mid_x - x
        mid_x = centre[:, 0]                          # (T,)
        lh = _mirror_x(lh, mid_x)
        rh = _mirror_x(rh, mid_x)
        face = _mirror_x(face, mid_x)
        # re-track after mirroring (positions changed) and re-pick active
        phys, wrists = _track_hands(lh, rh)
        active = _active_secondary(phys, wrists)
        secondary = 1 - active

    def hand_part(hfilled, hmask, hsize, is_active):
        wrist = hfilled[:, _WRIST, :]                 # (T,2)
        others = np.delete(hfilled, _WRIST, axis=1)   # (T,20,2)
        wrs = (wrist - centre) / width                # (T,2)
        orw = ((others - wrist[:, None, :]) / hsize.reshape(-1, 1, 1))   # (T,20,2)
        wrs = wrs * hmask[:, None]
        orw = orw * hmask[:, None, None]
        return np.concatenate([wrs.reshape(T, -1), orw.reshape(T, -1)], axis=1)   # (T,42)

    # Build per-slot landmark sequences (T,21,2) with None -> zeros
    slot_lm = [np.zeros((T, 21, 2), dtype=np.float64), np.zeros((T, 21, 2), dtype=np.float64)]
    slot_pres = [np.zeros(T, dtype=bool), np.zeros(T, dtype=bool)]
    for t, phys_t in enumerate(phys):
        for s in (0, 1):
            if phys_t[s] is not None:
                slot_lm[s][t] = phys_t[s]
                slot_pres[s][t] = True

    # gap-fill per slot
    l_filled, l_mask = _fill_gaps(slot_lm[0])
    r_filled, r_mask = _fill_gaps(slot_lm[1])
    # keep only frames where the slot was originally present OR interpolated:
    # mask already encodes that. But the slot arrays for the NON-tracked hand are all
    # zero; ensure masks reflect actual tracked presence.
    # Re-derive masks from tracked presence so gap-fill didn't invent presence:
    # _fill_gaps returns mask=1 for present+interpolated, 0 for long/leading/trailing missing.
    # That's exactly what we want. slot_present = (mask == 1) is fine, but interpolated
    # frames are "present enough". We'll use the returned masks directly.

    # hand sizes
    def hand_size(hfilled):
        return np.linalg.norm(hfilled[:, _WRIST, :] - hfilled[:, _MIDDLE_TIP, :],
                              axis=1, keepdims=True) + EPS

    l_size = hand_size(l_filled)
    r_size = hand_size(r_filled)

    # active = slot `active`, secondary = slot `secondary`
    active_filled = l_filled if active == 0 else r_filled
    active_mask = l_mask if active == 0 else r_mask
    active_size = l_size if active == 0 else r_size
    sec_filled = l_filled if secondary == 0 else r_filled
    sec_mask = l_mask if secondary == 0 else r_mask
    sec_size = l_size if secondary == 0 else r_size

    active_hand = hand_part(active_filled, active_mask, active_size, True)     # (T,42)
    sec_hand = hand_part(sec_filled, sec_mask, sec_size, False)                # (T,42)

    # hand-to-face: active wrist, active index, secondary wrist, secondary index
    h2f = []
    for hp, hmask in ((active_filled[:, _WRIST, :], active_mask),
                      (active_filled[:, _INDEX_TIP, :], active_mask),
                      (sec_filled[:, _WRIST, :], sec_mask),
                      (sec_filled[:, _INDEX_TIP, :], sec_mask)):
        for fidx in _FACE_POINTS:
            h2f.append(((hp - face[:, fidx, :]) / width) * hmask[:, None])
    h2f = np.concatenate(h2f, axis=1)                                          # (T,32)

    def wrist_vel(hfilled, hmask):
        wrist = hfilled[:, _WRIST, :]
        vel = np.zeros_like(wrist)
        if T > 1:
            vel[1:] = wrist[1:] - wrist[:-1]
        return vel * hmask[:, None]

    active_vel = wrist_vel(active_filled, active_mask)                         # (T,2)
    sec_vel = wrist_vel(sec_filled, sec_mask)                                  # (T,2)

    feat = np.concatenate([active_hand, sec_hand,
                           active_mask.reshape(T, 1), sec_mask.reshape(T, 1),
                           h2f,
                           active_vel, sec_vel], axis=1)                        # (T,122)
    assert feat.shape[1] == FEAT_DIM, feat.shape[1]
    feat = _resample(feat, n_frames)
    return feat.astype(np.float32)


def _shoulder_centre_width(pose):
    lsh = pose[:, _L_SHOULDER, :2]
    rsh = pose[:, _R_SHOULDER, :2]
    centre = (lsh + rsh) / 2.0
    width = np.linalg.norm(lsh - rsh, axis=1, keepdims=True) + EPS
    return centre, width


def _mirror_x(arr, mid_x):
    """Reflect x coordinate of a (T,n,2) array about per-frame mid_x."""
    out = arr.copy()
    out[:, :, 0] = 2.0 * mid_x[:, None] - out[:, :, 0]
    return out


def _should_mirror(phys, active, wrists, centre):
    """True if active hand's mean wrist x is left of the shoulder midpoint.

    Computed over frames where both active wrist and shoulder midpoint are valid.
    Defaults to False when there isn't enough valid data.
    """
    T = wrists.shape[0]
    active_present = np.any(wrists[:, active] != 0.0, axis=1)   # (T,)
    sh_valid = np.any(centre != 0.0, axis=1)                          # (T,) shoulders present
    valid = active_present & sh_valid
    if valid.sum() < 2:
        return False
    active_mean_wx = float(np.mean(wrists[valid, active, 0]))
    sh_mean_mx = float(np.mean(centre[valid, 0]))
    return active_mean_wx < sh_mean_mx


def self_test():
    T = 24
    h = np.zeros((T, 543, 3), dtype=np.float64)
    h[:, 11, 0] = 10.0; h[:, 12, 0] = 12.0   # shoulders -> midpoint x=11
    h[:, 11, 1] = 0.5; h[:, 12, 1] = 0.5
    # left hand (slot) wrist moves x=0..5.75 on the image-LEFT side of midpoint 11
    h[:, 501, 0] = np.arange(T) * 0.25; h[:, 501, 1] = 0.5
    h[:, 509, 0] = np.arange(T) * 0.25 + 1.0; h[:, 509, 1] = 0.5   # index tip
    # right hand (slot) static on image-RIGHT side
    h[:, 522, 0] = 20.0; h[:, 522, 1] = 0.5
    h[:, 530, 0] = 21.0; h[:, 530, 1] = 0.5
    h[:, 34, 0] = 11.0; h[:, 34, 1] = 0.4     # face nose
    h[:, 185, 0] = 11.0; h[:, 185, 1] = 0.7   # face chin

    f = extract_features(h, 'signsage')
    assert f.shape == (N_FRAMES, FEAT_DIM), f.shape
    assert not np.isnan(f).any() and not np.isinf(f).any()
    # active = left-hand slot (travel 23 > 0), starts image-left of midpoint -> mirrored
    masks = f[:, 84:86]
    assert np.all(masks == 1.0), masks[:5]   # both hands present all frames
    # After canonical mirroring, active hand should be on the RIGHT of shoulder midpoint.
    # active wrist = col 0; shoulder centre col 84+42+42... compute centre from cols.
    # Pose centre cols: in v2, pose was cols 84-98. Here active_hand is cols 0-42 (wrist col0).
    # Shoulder midpoint x: not directly stored; but active wrist col0 after mirroring should be > 0
    # (right of midpoint x=11 mirrored to >11 -> normalised (x-11)/2 > 0).
    assert np.all(f[:, 0] > 0), f[:3, 0]   # active wrist now right of midpoint
    print("features_v3.self_test: basic + mirror canonicalisation OK")

    # --- Test 1: swapping input L/R slots gives IDENTICAL output ---
    h_swapped = h.copy()
    h_swapped[:, 501:522, :] = h[:, 522:543, :]
    h_swapped[:, 522:543, :] = h[:, 501:522, :]
    f_swapped = extract_features(h_swapped, 'signsage')
    assert np.allclose(f, f_swapped, atol=1e-5), np.abs(f - f_swapped).max()
    print("features_v3.self_test: L/R slot swap -> identical output OK")

    # --- Test 2: mirrored copy gives near-identical output ---
    h_mirror = h.copy()
    mid = 11.0
    # mirror ALL x coords of face + both hands about the shoulder midpoint
    h_mirror[:, 33:500, 0] = 2 * mid - h[:, 33:500, 0]
    h_mirror[:, 501:543, 0] = 2 * mid - h[:, 501:543, 0]
    f_mirror = extract_features(h_mirror, 'signsage')
    assert np.allclose(f, f_mirror, atol=1e-4), np.abs(f - f_mirror).max()
    print("features_v3.self_test: mirrored copy -> near-identical output OK")

    # --- Test 3: all-missing-hand case ---
    h2 = np.zeros((T, 543, 3), dtype=np.float64)
    h2[:, 11, 0] = 10.0; h2[:, 12, 0] = 12.0
    h2[:, 11, 1] = 0.5; h2[:, 12, 1] = 0.5
    h2[:, 34, 0] = 11.0; h2[:, 34, 1] = 0.4
    h2[:, 185, 0] = 11.0; h2[:, 185, 1] = 0.7
    f2 = extract_features(h2, 'signsage')
    assert f2.shape == (N_FRAMES, FEAT_DIM)
    assert not np.isnan(f2).any()
    assert np.allclose(f2, 0.0), f2[np.abs(f2) > 0][:5]
    assert np.all(f2[:, 84] == 0.0) and np.all(f2[:, 85] == 0.0)
    print("features_v3.self_test: all-missing-hand -> all zeros, masks 0 OK")

    # --- Test 4: short/long resample ---
    fs = extract_features(h[:5], 'signsage')
    assert fs.shape == (N_FRAMES, FEAT_DIM)
    assert np.allclose(fs[-1], fs[4])
    print("features_v3.self_test: resample OK")
    print("features_v3.self_test: ALL OK")


if __name__ == "__main__":
    self_test()
