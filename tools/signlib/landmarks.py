"""Load/clean/normalize (T,75,3) MediaPipe landmark clips; select representative sample.

Layout: [0:33] pose, [33:54] hand A, [54:75] hand B (labels unreliable on unmirrored
video, so hands are assigned to physical sides by image x, tracked over time).
"""
import numpy as np

POSE_L_SHOULDER, POSE_R_SHOULDER = 11, 12
POSE_L_WRIST, POSE_R_WRIST = 15, 16
POSE_NOSE = 0
HAND_A, HAND_B = slice(33, 54), slice(54, 75)

# MediaPipe Hands landmark order
WRIST = 0
THUMB = (1, 2, 3, 4)
INDEX = (5, 6, 7, 8)
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)
FINGERS = {'thumb': THUMB, 'index': INDEX, 'middle': MIDDLE, 'ring': RING, 'pinky': PINKY}
MCP = {'index': 5, 'middle': 9, 'ring': 13, 'pinky': 17, 'thumb': 2}

def load_clip(path):
    a = np.load(path).astype(np.float64)
    assert a.ndim == 3 and a.shape[1:] == (75, 3), f'unexpected shape {a.shape}'
    return a

def hand_presence(clip):
    """(T,) bool per hand slot A/B."""
    a = np.any(clip[:, HAND_A] != 0, axis=(1, 2))
    b = np.any(clip[:, HAND_B] != 0, axis=(1, 2))
    return a, b

def assign_hands(clip):
    """Return dict physical_side -> (T,21,3) array (zeros where missing).

    Physical side for unmirrored frontal video: image-left = person's RIGHT hand.
    Assignment by wrist image-x relative to pose shoulder midpoint, with
    temporal continuity (nearest wrist to previous frame) to survive crossings.
    """
    T = len(clip)
    pres_a, pres_b = hand_presence(clip)
    mid_x = np.nanmean(np.where(clip[:, [POSE_L_SHOULDER, POSE_R_SHOULDER]].any(axis=2), clip[:, [POSE_L_SHOULDER, POSE_R_SHOULDER], 0], np.nan), axis=1)
    mid_x = np.where(np.isfinite(mid_x), mid_x, 0.5)
    slots = {}
    prev = {'R': None, 'L': None}
    out = {'R': np.zeros((T, 21, 3)), 'L': np.zeros((T, 21, 3))}
    for t in range(T):
        detected = []
        if pres_a[t]:
            detected.append(clip[t, HAND_A])
        if pres_b[t]:
            detected.append(clip[t, HAND_B])
        if not detected:
            continue
        if len(detected) == 1:
            d = detected[0]
            wx = d[WRIST, 0]
            if prev['R'] is not None or prev['L'] is not None:
                dR = np.linalg.norm(d[WRIST] - prev['R']) if prev['R'] is not None else np.inf
                dL = np.linalg.norm(d[WRIST] - prev['L']) if prev['L'] is not None else np.inf
                side = 'R' if dR <= dL else 'L'
            else:
                side = 'R' if wx < mid_x[t] else 'L'
            out[side][t] = d
            prev[side] = d[WRIST].copy()
            continue
        # two hands: image-left (smaller x) = physical RIGHT
        d0, d1 = detected
        if d0[WRIST, 0] <= d1[WRIST, 0]:
            right, left = d0, d1
        else:
            right, left = d1, d0
        out['R'][t] = right
        out['L'][t] = left
        prev['R'] = right[WRIST].copy()
        prev['L'] = left[WRIST].copy()
    return out

def fill_gaps(hand, max_gap=5):
    """Linear-interpolate missing landmarks runs <= max_gap frames, per landmark.
    Returns (filled, frame_presence_mask). Longer gaps stay zero."""
    T = len(hand)
    filled = hand.copy().astype(np.float64)
    for lm in range(hand.shape[1]):
        for c in range(3):
            vals = filled[:, lm, c]
            present = vals != 0
            i = 0
            while i < T:
                if present[i]:
                    i += 1
                    continue
                j = i
                while j < T and not present[j]:
                    j += 1
                if i > 0 and j < T and (j - i) <= max_gap:
                    a, b = filled[i - 1, lm, c], filled[j, lm, c]
                    for k in range(i, j):
                        f = (k - i + 1) / (j - i + 1)
                        filled[k, lm, c] = a + f * (b - a)
                i = j
    mask = np.any(filled != 0, axis=(1, 2)).astype(np.float64)
    return filled, mask

def aspect_correct(arr, width, height):
    """[0,1]-per-axis image coords -> pixel units (x*W, y*H). z scaled by W (MediaPipe
    pose/hand z shares the x scale)."""
    out = arr.copy()
    out[..., 0] *= width
    out[..., 1] *= height
    out[..., 2] *= width
    return out

# MediaPipe z (hip/wrist-relative depth) is unreliable in magnitude on these
# low-res unmirrored clips; compress it so depth contributes weakly rather than
# dominating IK reaches. x,y remain the primary signal (same as the x,y-only
# training pipeline).
Z_DAMP = 0.3

def normalize(hands, pose, avatar_shoulder_width, avatar_shoulder_center,
              width=1.0, height=1.0):
    """Image-normalized coords -> avatar-frame meters, y-up, relative to shoulder center.

    Works in pixel units so per-axis normalization is accounted for.
      x' = (x - mid_x)/sw * A     (image x grows rightward; avatar faces +z, its left is +x)
      y' = (mid_y - y)/sw * A     (flip so y grows up)
      z' = -(z - z_mid)/sw * A    (MediaPipe z: negative = closer to camera; avatar +z faces camera)
    Signer scale A = avatar shoulder width (meters); sw = signer shoulder width in pixels.
    """
    pose_px = aspect_correct(pose, width, height)
    hands_px = {s: (aspect_correct(h, width, height) if h is not None else None) for s, h in hands.items()}
    lsh = pose_px[:, POSE_L_SHOULDER]
    rsh = pose_px[:, POSE_R_SHOULDER]
    valid = np.any(lsh != 0, axis=1) & np.any(rsh != 0, axis=1)
    if valid.any():
        sw = float(np.median(np.linalg.norm(lsh[valid, :2] - rsh[valid, :2], axis=1)))
        mid = np.median((lsh[valid] + rsh[valid]) / 2.0, axis=0)
    else:
        sw, mid = 100.0, np.array([width / 2, height / 2, 0.0])
    sw = max(sw, 1e-4)
    out = {}
    for side, h in hands_px.items():
        if h is None:
            out[side] = None
            continue
        h2 = h.copy()
        h2[:, :, 0] = (h[:, :, 0] - mid[0]) / sw * avatar_shoulder_width
        h2[:, :, 1] = (mid[1] - h[:, :, 1]) / sw * avatar_shoulder_width
        h2[:, :, 2] = -(h[:, :, 2] - mid[2]) / sw * avatar_shoulder_width * Z_DAMP
        # keep frame-level absence as exact zeros so downstream presence tests work
        h2[~np.any(h != 0, axis=(1, 2))] = 0.0
        out[side] = h2
    pose_n = pose_px.copy()
    pose_n[:, 0] = (pose_px[:, 0] - mid[0]) / sw * avatar_shoulder_width
    pose_n[:, 1] = (mid[1] - pose_px[:, 1]) / sw * avatar_shoulder_width
    pose_n[:, 2] = -(pose_px[:, 2] - mid[2]) / sw * avatar_shoulder_width * Z_DAMP
    return out, pose_n, sw

def pose_shoulder_center_width(clip, width=1.0, height=1.0):
    """Median shoulder midpoint (pixels) and shoulder width (pixels)."""
    px = aspect_correct(clip[:, :33], width, height)
    lsh = px[:, POSE_L_SHOULDER]
    rsh = px[:, POSE_R_SHOULDER]
    valid = np.any(lsh != 0, axis=1) & np.any(rsh != 0, axis=1)
    if not valid.any():
        return np.array([width / 2, height / 2, 0.0]), 100.0
    mid = np.median((lsh[valid] + rsh[valid]) / 2.0, axis=0)
    sw = float(np.median(np.linalg.norm(lsh[valid, :2] - rsh[valid, :2], axis=1)))
    return mid, max(sw, 1e-4)

def resample_time(x, n):
    T = len(x)
    if T == n:
        return x.copy()
    idx = np.linspace(0, T - 1, n)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    f = (idx - i0)[:, None, None] if x.ndim == 3 else (idx - i0)[:, None]
    return x[i0] * (1 - f) + x[i1] * f

def clip_feature(clip, n_frames=60, width=1.0, height=1.0):
    """Shoulder-relative feature for DTW/medoid: both wrists + index/middle tips,
    x,y only, in shoulder-width units (aspect-corrected)."""
    hands = assign_hands(clip)
    feats = []
    mid, sw = pose_shoulder_center_width(clip, width, height)
    px_hands = {s: aspect_correct(hands[s], width, height) for s in ('R', 'L')}
    for side in ('R', 'L'):
        h, mask = fill_gaps(px_hands[side])
        for lm in (WRIST, 8, 12):
            p = (h[:, lm, :2] - mid[:2]) / sw
            feats.append(p)
    f = np.concatenate(feats, axis=1)  # (T, 12)
    return resample_time(f, n_frames)

def dtw_distance(a, b):
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = np.linalg.norm(ai - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m] / (n + m)

def medoid_index(clips, dims=None):
    """Index of the clip with minimal summed DTW to the others (time-normalized).

    dims: list of (width, height) per clip for aspect correction."""
    dims = dims or [(1.0, 1.0)] * len(clips)
    feats = [clip_feature(c, width=d[0], height=d[1]) for c, d in zip(clips, dims)]
    n = len(feats)
    totals = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_distance(feats[i], feats[j])
            totals[i] += d
            totals[j] += d
    return int(np.argmin(totals)), feats

def temporal_median(hand, kernel=5):
    """Per-landmark per-coordinate temporal median filter over present frames.
    Suppresses detection spikes without fabricating absent landmarks."""
    if hand is None:
        return None
    T = len(hand)
    out = hand.copy().astype(np.float64)
    half = kernel // 2
    for lm in range(hand.shape[1]):
        for c in range(3):
            vals = hand[:, lm, c]
            idx = np.where(vals != 0)[0]
            if len(idx) < 3:
                continue
            v = vals[idx]
            for i, t in enumerate(idx):
                s, e = max(0, i - half), min(len(idx), i + half + 1)
                out[t, lm, c] = np.median(v[s:e])
    return out
