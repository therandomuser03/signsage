#!/usr/bin/env python3
"""Build per-sign animation clips for signsage-human.glb from recorded landmarks.

Usage:
  python3 tools/build_sign_clips.py [--check-only] [--signs HELLO,WHERE]

Outputs:
  frontend/public/models/avatar/clips/<gloss>.json   clip data (per-frame local quats)
  docs/verification/retarget_report.json              per-sign stats (FK error, samples)
Manifest update (public/models/avatar/manifest.json) is done by update_manifest.py.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signlib import landmarks as LM
from signlib.glb import Rig
from signlib.retarget import Retargeter, smooth_quats, ease_frames, FINGER_BONES

REPO = Path(__file__).resolve().parent.parent
GLB_PATH = REPO / 'frontend/public/models/signsage-human.glb'
EXTRACTED = REPO / 'research/openhands_pilot/extracted_poses'
LOCAL = REPO / 'data/landmarks'
OUT_DIR = REPO / 'frontend/public/models/avatar/clips'
REPORT = REPO / 'docs/verification/retarget_report.json'

# Sign -> recorded sample config. Only signs with usable recordings are retargeted.
SIGN_SOURCES = {
    'HELLO': {'dirs': [EXTRACTED], 'pattern': 'hello__*.npy', 'fps': 25.0},
    'THANK_YOU': {'dirs': [EXTRACTED], 'pattern': 'thank_you__*.npy', 'fps': 25.0},
    'WHERE': {'dirs': [LOCAL], 'pattern': 'where.npy', 'fps': 30.0},
}

# Bones baked into clips (rest pose kept for everything else)
def clip_bones(rig):
    bones = ['clavicle.L', 'clavicle.R', 'shoulder01.L', 'shoulder01.R',
             'upperarm01.L', 'upperarm02.L', 'lowerarm01.L', 'lowerarm02.L',
             'wrist.L', 'upperarm01.R', 'upperarm02.R', 'lowerarm01.R', 'lowerarm02.R',
             'wrist.R']
    for side in ('L', 'R'):
        for chain in FINGER_BONES.values():
            bones += [f'{b}.{side}' for b in chain]
        bones += [f'metacarpal{i}.{side}' for i in range(1, 5)]
    return [b for b in bones if b in rig.by_name]

def video_dims_for(npy_path: Path):
    """Locate the source video for a .npy and return (width, height); None if unknown."""
    name = npy_path.stem
    sign = name.split('__')[0]
    candidates = [
        REPO / 'backend/data/isl/videos' / sign / f'{name}.mp4',
        REPO / 'backend/data/isl/videos' / f'{name}.mp4',
        REPO / 'data/videos' / f'{name}.mp4',
    ]
    for v in candidates:
        if v.exists():
            out = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                 '-show_entries', 'stream=width,height', '-of', 'csv=p=0', str(v)],
                capture_output=True, text=True).stdout.strip()
            w, h = out.split(',')[:2]
            return int(w), int(h)
    return None

def clean_and_normalize(clip, retarget: Retargeter, width, height):
    """Returns (hands_norm {L,R:(T,21,3) avatar meters, shoulder-center relative}, pose_norm) or None."""
    hands = LM.assign_hands(clip)
    out = {}
    ok = False
    for side in ('L', 'R'):
        filled, mask = LM.fill_gaps(hands[side], max_gap=5)
        if mask.mean() < 0.5:
            filled = None  # hand mostly missing in this sample
        else:
            filled = LM.temporal_median(filled, kernel=5)
        out[side] = filled
        if filled is not None:
            ok = True
    if not ok:
        return None
    if not (np.any(clip[:, LM.POSE_L_SHOULDER] != 0) and np.any(clip[:, LM.POSE_R_SHOULDER] != 0)):
        return None
    hands_norm, pose_norm, sw = LM.normalize(
        out, clip[:, :33], retarget.shoulder_width, retarget.shoulder_center_rest, width, height)
    return hands_norm, pose_norm

def solve_clip(sample, retarget: Retargeter, bones, width, height):
    cleaned = clean_and_normalize(sample, retarget, width, height)
    if cleaned is None:
        return None, 'cleaning failed (shoulders/hands missing)'
    hands_norm, pose_norm = cleaned
    rig = retarget.rig
    frames = []
    for t in range(len(sample)):
        h_t = {s: (hands_norm[s][t] if hands_norm[s] is not None else None) for s in ('L', 'R')}
        local = retarget.solve_frame(h_t, pose_norm[t])
        frames.append({b: local[b].tolist() for b in bones})
    return frames, None

def smooth_clip(frames, bones, window=5):
    out = []
    for i in range(len(frames)):
        out.append({b: None for b in bones})
    for b in bones:
        seq = smooth_quats([np.array(f[b]) for f in frames], window=window)
        for i, q in enumerate(seq):
            out[i][b] = q
    return out

def ease_clip(frames, rest, bones, blend_in=8, blend_out=8):
    rest_local = {b: rest[b] for b in bones}
    for f in frames:
        for b in bones:
            pass
    return ease_frames(frames, rest_local, blend_in, blend_out)

def fk_error(frames, sample, retarget: Retargeter, bones, anchor, width, height):
    """Mean position error (shoulder-width units) vs source landmarks for
    wrist + fingertips of both hands."""
    cleaned = clean_and_normalize(sample, retarget, width, height)
    if cleaned is None:
        return None
    hands_norm, pose_norm = cleaned
    rig = retarget.rig
    errs = []
    for t, f in enumerate(frames):
        local = {b: np.array(f[b]) for b in bones}
        pos, _ = rig.world_positions(local)
        for side in ('L', 'R'):
            h = hands_norm[side]
            if h is None or not np.any(h[t]):
                continue  # hand absent this frame: avatar at rest by design
            target_wrist = anchor + h[t, 0]
            errs.append(np.linalg.norm(pos[f'wrist.{side}'] - target_wrist) / retarget.shoulder_width)
            for tip, bone in ((4, 'finger1-3'), (8, 'finger2-3'), (12, 'finger3-3'), (16, 'finger4-3'), (20, 'finger5-3')):
                target = anchor + h[t, tip]
                errs.append(np.linalg.norm(pos[f'{bone}.{side}'] - target) / retarget.shoulder_width)
    return float(np.mean(errs)) if errs else None

def active_window(sample, width, height, pad=3, frac=0.15):
    """[start, end) frame window of actual hand motion (wrist speed), so dead
    air at the start/end of a recording is not baked into the clip."""
    hands = LM.assign_hands(sample)
    px = {s: LM.aspect_correct(hands[s], width, height) for s in 'RL'}
    wrists = []
    for s in 'RL':
        h, mask = LM.fill_gaps(px[s])
        wrists.append(np.where(mask[:, None] > 0, h[:, 0, :2], np.nan))
    W = np.nanmean(np.stack(wrists), axis=0)
    ok = ~np.isnan(W).any(axis=1)
    if ok.sum() < 5:
        return (0, len(sample))
    Wf = W[ok]
    idx = np.where(ok)[0]
    sp = np.linalg.norm(np.diff(Wf, axis=0), axis=1)
    if sp.max() <= 0:
        return (0, len(sample))
    thr = max(5.0, sp.max() * frac)
    act = np.where(sp > thr)[0]
    if len(act) == 0:
        return (0, len(sample))
    a, b = idx[act[0]], idx[act[-1]] + 1
    return (max(0, a - pad), min(len(sample), b + pad))

def process_sign(gloss, cfg, retarget: Retargeter, bones, check_only=False):
    files = []
    for d in cfg['dirs']:
        files += sorted(d.glob(cfg['pattern']))
    samples = []
    for f in files:
        try:
            samples.append((f, LM.load_clip(f)))
        except Exception as e:
            print(f'  skip {f}: {e}')
    if not samples:
        return {'gloss': gloss, 'status': 'no-data', 'samples': 0}
    dims = [video_dims_for(s[0]) for s in samples]
    if any(d is None for d in dims):
        missing = [s[0].name for s, d in zip(samples, dims) if d is None]
        print(f'  WARNING: no source video dims for {missing}; using square-frame assumption')
        dims = [d or (1.0, 1.0) for d in dims]
    medoid_i, _ = LM.medoid_index([s[1] for s in samples], dims)
    medoid_file, medoid = samples[medoid_i]
    print(f'  {gloss}: {len(samples)} samples, medoid={medoid_file.name} ({len(medoid)} frames)')
    medoid_dims = dims[medoid_i]
    a, b = active_window(medoid, medoid_dims[0], medoid_dims[1])
    if (a, b) != (0, len(medoid)):
        print(f'  trim idle: frames {a}:{b} of {len(medoid)}')
        medoid = medoid[a:b]
    frames, err = solve_clip(medoid, retarget, bones, medoid_dims[0], medoid_dims[1])
    if frames is None:
        return {'gloss': gloss, 'status': f'failed: {err}', 'samples': len(samples)}
    frames = smooth_clip(frames, bones, window=5)
    anchor = retarget.shoulder_center_rest
    fk_before = fk_error(frames, medoid, retarget, bones, anchor, medoid_dims[0], medoid_dims[1])
    frames = ease_clip(frames, {b: retarget.rig.rest_local_q[retarget.rig.by_name[b]] for b in bones}, bones)
    fk_full = None  # ease frames intentionally deviate (rest-blended), so check pre-ease only
    rest = {b: retarget.rig.rest_local_q[retarget.rig.by_name[b]].tolist() for b in bones}
    fps = cfg['fps']
    clip = {
        'gloss': gloss,
        'version': 1,
        'fps': fps,
        'duration': len(frames) / fps,
        'frames': [{b: [round(c, 6) for c in f[b]] for b in bones} for f in frames],
        'meta': {
            'source': 'retarget',
            'authored': False,
            'needsReview': len(samples) < 3,
            'samplesUsed': len(samples),
            'sampleFiles': [s[0].name for s in samples],
            'medoidFile': medoid_file.name,
            'fkMeanErrorShoulderWidths': fk_before,
            'method': 'DTW-medoid of cleaned samples; minimal-arc finger swing; two-bone IK arms; palm-frame wrist twist',
        },
    }
    if not check_only:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / f'{gloss.lower()}.json', 'w') as f:
            json.dump(clip, f)
    return {'gloss': gloss, 'status': 'ok', 'samples': len(samples), 'frames': len(frames),
            'fkMeanErrorShoulderWidths': fk_before, 'medoid': medoid_file.name}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--signs', default=','.join(SIGN_SOURCES))
    ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args()
    t0 = time.time()
    rig = Rig(GLB_PATH)
    retarget = Retargeter(rig)
    bones = clip_bones(rig)
    print(f'bones in clip: {len(bones)}')
    report = []
    for gloss in args.signs.split(','):
        cfg = SIGN_SOURCES[gloss]
        print(f'== {gloss}')
        report.append(process_sign(gloss, cfg, retarget, bones, args.check_only))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({'rig': str(GLB_PATH), 'results': report}, indent=2))
    print(f'\nreport -> {REPORT}  ({time.time()-t0:.1f}s)')
    for r in report:
        print(' ', r['gloss'], r['status'], f"samples={r.get('samples')} fk_err={r.get('fkMeanErrorShoulderWidths')}")

if __name__ == '__main__':
    main()
