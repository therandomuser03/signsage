"""Step 3: Feature extraction for ISL word classifier.

Standardized per-frame feature (98 dims, x,y only):
  left hand  (21 x,y)  = 42
  right hand (21 x,y)  = 42
  7 upper-body pose joints x,y = 14   [nose, left eye, right eye, l shoulder, r shoulder, l elbow, r elbow]
Total = 98 dims / frame.

Normalization: shoulder-center + shoulder-width (per frame).
Temporal: resample to 60 frames (uniform subsample if T>=60, edge-pad if T<60).

Accepts either:
  - MediaPipe Holistic (T,75,3): pose 0-32, left hand 33-53, right hand 54-74
  - SignSage           (T,543,3): pose 0-32, face 33-500, left hand 501-521, right hand 522-542
"""
import numpy as np

POSE_JOINTS = [0, 2, 5, 11, 12, 13, 14]  # nose, leye, reye, lsh, rsh, lel, rel
N_POSE_JOINTS = len(POSE_JOINTS)          # 7
N_FRAMES = 60
FEAT_DIM = 21 * 2 + 21 * 2 + N_POSE_JOINTS * 2  # 42+42+14 = 98

LAYOUTS = {
    'holistic': {'pose': (0, 33), 'lhand': (33, 54), 'rhand': (54, 75), 'n_landmarks': 75},
    'signsage': {'pose': (0, 33), 'lhand': (501, 522), 'rhand': (522, 543), 'n_landmarks': 543},
}

# column offsets inside the 98-dim feature vector
_LHAND_COLS = slice(0, 42)
_RHAND_COLS = slice(42, 84)
_POSE_COLS = slice(84, 98)
# shoulder joints within the 7 pose joints -> columns 84 + (3,4)*2
_LSH_COL = slice(90, 92)
_RSH_COL = slice(92, 94)


def _slice(landmarks, layout, key):
    a, b = LAYOUTS[layout][key]
    return landmarks[:, a:b, :]


def extract_features(landmarks, layout='holistic', n_frames=N_FRAMES):
    """landmarks: (T, N, 3) float. Returns (n_frames, 98) float32."""
    layout = layout.lower()
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; expected one of {list(LAYOUTS)}")
    if landmarks.ndim != 3 or landmarks.shape[2] < 2:
        raise ValueError("landmarks must be (T, N, >=3)")
    lh = _slice(landmarks, layout, 'lhand')[:, :, :2]   # (T,21,2)
    rh = _slice(landmarks, layout, 'rhand')[:, :, :2]   # (T,21,2)
    pose = _slice(landmarks, layout, 'pose')            # (T,33,3)
    pj = pose[:, POSE_JOINTS, :2]                       # (T,7,2)
    feat = np.concatenate([
        lh.reshape(lh.shape[0], -1),                    # (T,42)
        rh.reshape(rh.shape[0], -1),                    # (T,42)
        pj.reshape(pj.shape[0], -1),                    # (T,14)
    ], axis=1)                                          # (T,98)
    feat = _normalize(feat)
    feat = _resample(feat, n_frames)
    return feat.astype(np.float32)


def _normalize(feat):
    """Shoulder-center + shoulder-width per frame. feat: (T,98) = 49 (x,y) pairs."""
    T = feat.shape[0]
    pairs = feat.reshape(T, -1, 2)                      # (T,49,2)
    lsh = pairs[:, 45, :]                               # lsh = pair index 45 (col 90-91)
    rsh = pairs[:, 46, :]                               # rsh = pair index 46 (col 92-93)
    center = (lsh + rsh) / 2.0                          # (T,2)
    width = np.linalg.norm(lsh - rsh, axis=1, keepdims=True) + 1e-6  # (T,1)
    pairs = (pairs - center[:, None, :]) / width[:, None]
    return pairs.reshape(T, -1)


def _resample(feat, n_frames):
    T = feat.shape[0]
    if T == n_frames:
        return feat
    if T > n_frames:
        idx = np.linspace(0, T - 1, n_frames).round().astype(int)
        return feat[idx]
    pad = np.repeat(feat[-1:], n_frames - T, axis=0)
    return np.concatenate([feat, pad], axis=0)


def self_test():
    """Index mapping + shape tests only (not accuracy evidence)."""
    T = 10
    # holistic: lhand wrist (idx33) x=1, rhand wrist (idx54) x=2
    # shoulders idx11=10, idx12=12 -> center=11, width=2
    h = np.zeros((T, 75, 3))
    h[:, 33, 0] = 1.0
    h[:, 54, 0] = 2.0
    h[:, 11, 0] = 10.0   # l shoulder
    h[:, 12, 0] = 12.0   # r shoulder
    f = extract_features(h, 'holistic')
    assert f.shape == (N_FRAMES, FEAT_DIM), f.shape
    # normalized: (x - 11) / 2
    assert np.allclose(f[0, 0], -5.0), f[0, 0]          # lhand wrist x -> col 0
    assert np.allclose(f[0, 42], -4.5), f[0, 42]        # rhand wrist x -> col 42
    assert np.allclose(f[0, 90], -0.5), f[0, 90]        # lshoulder -> col 90
    assert np.allclose(f[0, 92], 0.5), f[0, 92]         # rshoulder -> col 92
    # signsage: lhand idx501 x=3, rhand idx522 x=4, same shoulders
    s = np.zeros((T, 543, 3))
    s[:, 501, 0] = 3.0
    s[:, 522, 0] = 4.0
    s[:, 11, 0] = 10.0
    s[:, 12, 0] = 12.0
    f2 = extract_features(s, 'signsage')
    assert f2.shape == (N_FRAMES, FEAT_DIM), f2.shape
    assert np.allclose(f2[0, 0], (3.0 - 11.0) / 2.0), f2[0, 0]
    assert np.allclose(f2[0, 42], (4.0 - 11.0) / 2.0), f2[0, 42]
    assert np.allclose(f2[0, 90], -0.5), f2[0, 90]
    assert np.allclose(f2[0, 92], 0.5), f2[0, 92]
    # resample short (T=5 -> 60 edge pad) and long (T=120 -> 60 subsample)
    short = np.zeros((5, 75, 3))
    short[:, 33, 0] = 9.0   # lhand wrist
    short[:, 11, 0] = 10.0
    short[:, 12, 0] = 12.0
    fs = extract_features(short, 'holistic')
    assert fs.shape == (N_FRAMES, FEAT_DIM)
    assert np.allclose(fs[-1], fs[4])                   # edge pad repeats last frame
    assert np.allclose(fs[0, 0], (9.0 - 11.0) / 2.0)
    long = np.zeros((120, 75, 3))
    long[:, 0, 0] = np.arange(120, dtype=np.float32)    # varying nose
    long[:, 11, 0] = 10.0
    long[:, 12, 0] = 12.0
    fl = extract_features(long, 'holistic')
    assert fl.shape == (N_FRAMES, FEAT_DIM)
    assert np.allclose(fl[0, 0], (0.0 - 11.0) / 2.0)    # first frame kept
    assert not np.allclose(fl, fs)
    print("features.self_test: OK (index mapping + normalization + resample shapes verified)")


if __name__ == '__main__':
    self_test()
