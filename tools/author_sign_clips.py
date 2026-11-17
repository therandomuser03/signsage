#!/usr/bin/env python3
"""Hand-authored keyframe clips for library signs with NO recorded data.

APPROXIMATIONS ONLY — not recorded motion, not validated ISL, not claimed
correct. Each synthetic hand is built in avatar frame, run through the same
retargeter as the recorded pipeline so the rig's limits are respected.
Output clips carry meta.authored=true and must be replaced by signer-reviewed
or recorded clips. ISL signer review is mandatory before any validation claim.
"""
import json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signlib.glb import Rig
from signlib.retarget import Retargeter, smooth_quats, ease_frames, FINGER_BONES
from build_sign_clips import clip_bones

REPO = Path(__file__).resolve().parent.parent
GLB_PATH = REPO / 'frontend/public/models/signsage-human.glb'
OUT_DIR = REPO / 'frontend/public/models/avatar/clips'

FPS = 30
DUR_S = 1.6
FRAMES = int(FPS * DUR_S)  # 48

# handedness frame math: r = "pinky→index" across palm
def palm_frame(f, n, side):
    f = np.asarray(f, float); f /= np.linalg.norm(f)
    n = np.asarray(n, float); n /= np.linalg.norm(n)
    n = n - np.dot(n, f) * f  # orthogonalize
    n /= np.linalg.norm(n)
    # pinky→index direction across the palm. Right hand: index is on the
    # thumb (lateral) side; verify with f=(0,1,0), n=(0,0,1): R needs r=(-1,0,0).
    if side == 'R':
        r = np.cross(n, f)
    else:
        r = np.cross(f, n)
    r /= np.linalg.norm(r)
    return f, r, n

# per-finger: (mcp offset along r in meters, segment lengths p/m/d, curl per joint multipliers)
FINGER_LAYOUT = {
    'index':  (+0.048, (0.052, 0.034, 0.024), (0.9, 1.15, 0.8)),
    'middle': (+0.016, (0.058, 0.038, 0.026), (0.9, 1.15, 0.8)),
    'ring':   (-0.014, (0.052, 0.035, 0.024), (0.9, 1.15, 0.8)),
    'pinky':  (-0.042, (0.042, 0.027, 0.021), (0.9, 1.1, 0.75)),
}
THUMB_LEN = (0.046, 0.034, 0.030)
THUMB_CURL = (0.6, 0.7, 0.5)

def synth_hand(W, f, n, side, curl, thumb_curl=None):
    """21 landmarks (avatar frame, meters). W = wrist position (absolute)."""
    f, r, n = palm_frame(f, n, side)
    lm = np.zeros((21, 3))
    lm[0] = W
    base = W + f * 0.085  # palm length wrist→MCP (anatomical scale)
    # thumb: diagogonal out from palm, base at index side
    tm = THUMB_LEN
    tc = thumb_curl if thumb_curl is not None else curl
    thumb_dir = (f * 0.45 - r * 0.75 + n * 0.18)
    thumb_dir /= np.linalg.norm(thumb_dir)
    p = W + r * 0.046 - f * 0.008 + n * 0.006
    for i, seg_len in enumerate(tm):
        a = THUMB_CURL[i] * tc
        bend_axis = np.cross(thumb_dir, n)
        bend_axis /= np.linalg.norm(bend_axis) + 1e-9
        thumb_dir = rot_axis(thumb_dir, bend_axis, a)
        p = p + thumb_dir * seg_len
        lm[[2, 3, 4][i]] = p
    lm[1] = W + r * 0.030 - f * 0.006  # thumb CMC-ish
    for name, (off, lens, mults) in FINGER_LAYOUT.items():
        idx = {'index': 5, 'middle': 9, 'ring': 13, 'pinky': 17}[name]
        d = f.copy()
        p = base + r * off
        for i in range(3):
            lm[idx + i] = p
            a = mults[i] * curl
            axis = np.cross(d, n)
            axis /= np.linalg.norm(axis) + 1e-9
            d = rot_axis(d, axis, a)
            p = p + d * lens[i]
        lm[idx + 3] = p
    return lm

def rot_axis(v, axis, angle):
    axis = np.asarray(axis, float)
    na = np.linalg.norm(axis)
    if na < 1e-9:
        return v
    axis = axis / na
    v = np.asarray(v, float)
    return (v * np.cos(angle) + np.cross(axis, v) * np.sin(angle)
            + axis * np.dot(axis, v) * (1 - np.cos(angle)))

def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)

def build_sequence(spec):
    """spec: fn(t01) -> dict with keys: W (abs wrist pos), f, n, curl, thumb_curl, side.
    Both-hands signs return dict of two hands. Returns list of (L,R) dict per frame."""
    frames = []
    for i in range(FRAMES):
        t = i / (FRAMES - 1)
        frames.append(spec(t))
    return frames

def frame_hands(frame, anchor):
    """Absolute W -> shoulder-center-relative meters for the retargeter.
    Returns (hands, palms): palms = exact authored palm normals per side."""
    out = {}
    palms = {}
    for side, sd in frame.items():
        W = np.asarray(sd['W'], float)
        lm = synth_hand(W, sd['f'], sd['n'], side, sd['curl'], sd.get('thumb_curl'))
        out[side] = lm - anchor
        palms[side] = np.asarray(sd['n'], dtype=np.float64)
    for side in ('L', 'R'):
        if side not in out:
            out[side] = np.zeros((21, 3))
    return out, palms

def up(t, y):  # vertical ease helper unused; kept minimal
    return y

def please_spec(t):
    # right flat palm circles twice on the chest
    anchor = np.array([0.0, 1.4456, 0.026])
    theta = 2 * np.pi * 2 * t
    W = anchor + np.array([-0.07 + 0.045 * np.cos(theta), -0.16 + 0.045 * np.sin(theta), 0.19])
    return {'R': {'W': W, 'f': [0, 1, -0.15], 'n': [0, 0, -1], 'curl': 0.0, 'thumb_curl': 0.1}}

def food_spec(t):
    # right bunched fingers tap chin/mouth, three times
    anchor = np.array([0.0, 1.4456, 0.026])
    cycle = (t * 3) % 1.0
    lift = smoothstep(min(cycle / 0.45, 1.0)) * smoothstep((1.0 - cycle) / 0.45 if cycle > 0.55 else 1.0)
    W = anchor + np.array([-0.085, -0.24 + 0.27 * lift, 0.17 + 0.05 * lift])
    return {'R': {'W': W, 'f': [0, 0.8, 0.6], 'n': [0, -0.55, 0.84], 'curl': 0.72, 'thumb_curl': 0.55}}

def name_spec(t):
    # right index extended, taps twice toward the chest
    anchor = np.array([0.0, 1.4456, 0.026])
    tap = abs(np.sin(2 * np.pi * 2 * t))
    W = anchor + np.array([-0.115, -0.175, 0.16 + 0.055 * tap])
    return {'R': {'W': W, 'f': [0, 0.15, 1], 'n': [0, 1, 0.1], 'curl': 0.9, 'thumb_curl': 0.35,
                   'index_extended': True}}

def name_spec_hand(frame_spec_t):
    pass

def how_spec(t):
    # both cupped palms up, alternate small rolls
    anchor = np.array([0.0, 1.4456, 0.026])
    roll = np.sin(2 * np.pi * 2 * t) * 0.45
    hands = {}
    for side, sx in (('R', -1), ('L', 1)):
        n = [np.sin(sx * roll), np.cos(roll), 0.15]
        W = anchor + np.array([sx * 0.175, -0.27, 0.23])
        hands[side] = {'W': W, 'f': [0, 0.25, 1], 'n': n, 'curl': 0.28, 'thumb_curl': 0.15}
    return hands

SYNTHS = {'PLEASE': please_spec, 'FOOD': food_spec, 'NAME': name_spec, 'HOW': how_spec}

def main():
    rig = Rig(GLB_PATH)
    rt = Retargeter(rig)
    anchor = rt.shoulder_center_rest
    bones = clip_bones(rig)
    rest_local = {b: rig.rest_local_q[rig.by_name[b]] for b in bones}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for gloss, spec in SYNTHS.items():
        frames_local = []
        for i in range(FRAMES):
            t = i / (FRAMES - 1)
            frame_abs = spec(t)
            h, palms = frame_hands(frame_abs, anchor)
            # NAME: index extended while others curled — special-case via curl vector
            if gloss == 'NAME':
                h = synth_name_hands(frame_abs, anchor)
            local = rt.solve_frame(h, None, palm_overrides=palms)
            frames_local.append({b: local[b].tolist() for b in bones})
        # smooth + ease
        out = []
        for i in range(FRAMES):
            out.append({b: None for b in bones})
        for b in bones:
            seq = smooth_quats([np.array(f[b]) for f in frames_local], window=5)
            for i, q in enumerate(seq):
                out[i][b] = q
        frames_local = [ {b: out[i][b].tolist() for b in bones} for i in range(FRAMES) ]
        frames_local = ease_frames(frames_local, rest_local, blend_in=8, blend_out=8)
        clip = {
            'gloss': gloss,
            'version': 1,
            'fps': FPS,
            'duration': len(frames_local) / FPS,
            'frames': [{b: [round(c, 6) for c in f[b]] for b in bones} for f in frames_local],
            'meta': {
                'source': 'authored-approximation',
                'authored': True,
                'needsReview': True,
                'samplesUsed': 0,
                'sampleFiles': [],
                'note': 'Hand-authored keyframes (no recorded data exists for this sign in the repo). '
                        'NOT validated ISL. Must be replaced/reviewed by a fluent ISL signer.',
            },
        }
        path = OUT_DIR / f'{gloss.lower()}.json'
        path.write_text(json.dumps(clip))
        print(f'{gloss}: {len(frames_local)} frames -> {path}')

def synth_name_hands(frame_abs, anchor):
    """NAME: index extended, other fingers+thumb curled."""
    out = {}
    for side, sd in frame_abs.items():
        W = np.asarray(sd['W'], float)
        f, r, n = palm_frame(sd['f'], sd['n'], side)
        lm = np.zeros((21, 3))
        lm[0] = W
        base = W + f * 0.085  # palm length wrist→MCP (anatomical scale)
        # curled fingers except index
        for name, (off, lens, mults) in FINGER_LAYOUT.items():
            idx = {'index': 5, 'middle': 9, 'ring': 13, 'pinky': 17}[name]
            curl = 0.0 if name == 'index' else 0.92
            d = f.copy()
            p = base + r * off
            for i in range(3):
                lm[idx + i] = p
                a = mults[i] * curl
                axis = np.cross(d, n)
                axis /= np.linalg.norm(axis) + 1e-9
                d = rot_axis(d, axis, a)
                p = p + d * lens[i]
            lm[idx + 3] = p
        # thumb relaxed-curled
        d = (f * 0.45 - r * 0.75 + n * 0.18); d /= np.linalg.norm(d)
        p = W + r * 0.046 - f * 0.008 + n * 0.006
        lm[1] = W + r * 0.030 - f * 0.006
        for i in range(3):
            a = THUMB_CURL[i] * 0.35
            axis = np.cross(d, n); axis /= np.linalg.norm(axis) + 1e-9
            d = rot_axis(d, axis, a)
            p = p + d * THUMB_LEN[i]
            lm[[2, 3, 4][i]] = p
        out[side] = lm - anchor
    for side in ('L', 'R'):
        if side not in out:
            out[side] = np.zeros((21, 3))
    return out

if __name__ == '__main__':
    main()
