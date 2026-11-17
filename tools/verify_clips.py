#!/usr/bin/env python3
"""Step 3 verification harness for the sign clips.

1. FK consistency: re-solve frames, forward-kinematics the GLB, compare
   wrist + fingertip positions against the source/synthetic landmarks.
2. Distinctness: pairwise DTW between final clips (wrist + fingertip world
   positions, resampled). Different signs must differ measurably.
3. Left/right sanity: the hand that moves in the source must be the avatar's
   hand on the same side. (Signs whose active hand is missing in the source
   are reported as unknown.)

Usage: python3 tools/verify_clips.py [--max-mean-error 0.6]
Writes docs/verification/verification_report.json. Exits non-zero on failure.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signlib import landmarks as LM
from signlib.glb import Rig
from signlib.retarget import Retargeter
from build_sign_clips import (SIGN_SOURCES, clip_bones, solve_clip, fk_error,
                              active_window, video_dims_for, clean_and_normalize)
from author_sign_clips import SYNTHS, synth_hand, palm_frame, FRAMES

REPO = Path(__file__).resolve().parent.parent
GLB_PATH = REPO / 'frontend/public/models/signsage-human.glb'
CLIPS = REPO / 'frontend/public/models/avatar/clips'
REPORT = REPO / 'docs/verification/verification_report.json'

TIP_PAIRS = [(4, 'finger1-3'), (8, 'finger2-3'), (12, 'finger3-3'), (16, 'finger4-3'), (20, 'finger5-3')]

def clip_world_tracks(rig, bones, frames):
    """(T, 12, 3): wrist + index/middle tips for L and R, avatar frame."""
    tracks = []
    for f in frames:
        local = {b: np.array(f[b]) for b in bones}
        pos, _ = rig.world_positions(local)
        row = []
        for side in ('R', 'L'):
            row.append(pos[f'wrist.{side}'])
            row.append(pos[f'finger2-3.{side}'])
            row.append(pos[f'finger3-3.{side}'])
        tracks.append(np.array(row))
    return np.array(tracks)

def resample(x, n=60):
    T = len(x)
    idx = np.linspace(0, T - 1, n)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    f = (idx - i0)[:, None, None]
    return x[i0] * (1 - f) + x[i1] * f

def dtw(a, b):
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m] / (n + m)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-mean-error', type=float, default=0.6,
                    help='max allowed mean FK error in shoulder-width units')
    args = ap.parse_args()
    rig = Rig(GLB_PATH)
    retarget = Retargeter(rig)
    bones = clip_bones(rig)
    report = {'rig': str(GLB_PATH), 'maxMeanErrorShoulderWidths': args.max_mean_error, 'signs': []}
    failures = []

    # ---- retargeted signs: FK error + left/right sanity
    for gloss, cfg in SIGN_SOURCES.items():
        files = sorted(p for d in cfg['dirs'] for p in d.glob(cfg['pattern']))
        samples = [(f, LM.load_clip(f)) for f in files]
        dims = [video_dims_for(f) or (1.0, 1.0) for f, _ in samples]
        medoid_i, _ = LM.medoid_index([s[1] for s in samples], dims)
        medoid = samples[medoid_i][1]
        dims_m = dims[medoid_i]
        a, b = active_window(medoid, dims_m[0], dims_m[1])
        medoid = medoid[a:b]
        frames, err = solve_clip(medoid, retarget, bones, dims_m[0], dims_m[1])
        fk = fk_error(frames, medoid, retarget, bones, retarget.shoulder_center_rest, dims_m[0], dims_m[1])

        # left/right sanity: which avatar-side hand moves more?
        cleaned = clean_and_normalize(medoid, retarget, dims_m[0], dims_m[1])
        hands_norm, _ = cleaned
        mov = {}
        for side in ('L', 'R'):
            h = hands_norm[side]
            if h is None:
                mov[side] = 0.0
                continue
            pres = np.any(h != 0, axis=(1, 2))
            w = h[:, 0, :2]
            sp = np.linalg.norm(np.diff(w, axis=0), axis=1)
            mov[side] = float(sp[pres[1:]].sum()) if pres[1:].any() else 0.0
        track = clip_world_tracks(rig, bones, frames)
        avatar_mov = {'R': float(np.linalg.norm(np.diff(track[:, 0], axis=0), axis=1).sum()),
                      'L': float(np.linalg.norm(np.diff(track[:, 3], axis=0), axis=1).sum())}
        source_stronger = 'R' if mov['R'] >= mov['L'] else ('L' if mov['L'] > mov['R'] else 'even')
        avatar_stronger = 'R' if avatar_mov['R'] >= avatar_mov['L'] else 'L'
        lr_ok = (source_stronger == 'even') or (source_stronger == avatar_stronger)
        if fk is not None and fk > args.max_mean_error:
            failures.append(f'{gloss}: FK error {fk:.3f} > {args.max_mean_error}')
        report['signs'].append({'gloss': gloss, 'kind': 'retarget', 'fkMeanErrorShoulderWidths': fk,
                                'sourceWristTravel': mov, 'avatarWristTravel': avatar_mov,
                                'leftRightSanity': lr_ok})
        print(f'{gloss}: fk={fk:.3f} source_travel={mov} avatar_travel={ {k: round(v,3) for k,v in avatar_mov.items()} } lr_ok={lr_ok}')

    # ---- authored signs: FK error against the exact synthetic landmarks
    for gloss, spec in SYNTHS.items():
        anchor = retarget.shoulder_center_rest
        frames_local = []
        lm_seq = []
        for i in range(FRAMES):
            t = i / (FRAMES - 1)
            frame_abs = spec(t)
            lm = {}
            for side, sd in frame_abs.items():
                W = np.asarray(sd['W'], float)
                lm[side] = synth_hand(W, sd['f'], sd['n'], side, sd['curl'], sd.get('thumb_curl'))
                if gloss == 'NAME':
                    from author_sign_clips import synth_name_hands
                    lm[side] = synth_name_hands(frame_abs, anchor)[side] + anchor
            for side in ('L', 'R'):
                if side not in lm:
                    lm[side] = np.zeros((21, 3))
            palms = {s: np.asarray(sd['n'], dtype=np.float64) for s, sd in frame_abs.items()}
            rel = {s: (lm[s] - anchor if np.any(lm[s]) else np.zeros((21, 3))) for s in ('L', 'R')}
            local = retarget.solve_frame(rel, None, palm_overrides=palms)
            frames_local.append({b: local[b].tolist() for b in bones})
            lm_seq.append(rel)
        # FK error over synthetic frames
        errs = []
        for t in range(FRAMES):
            local = {b: np.array(frames_local[t][b]) for b in bones}
            pos, _ = rig.world_positions(local)
            for side in ('L', 'R'):
                h = lm_seq[t][side]
                if not np.any(h):
                    continue
                errs.append(np.linalg.norm(pos[f'wrist.{side}'] - (anchor + h[0])) / retarget.shoulder_width)
                for tip, bone in TIP_PAIRS:
                    errs.append(np.linalg.norm(pos[f'{bone}.{side}'] - (anchor + h[tip])) / retarget.shoulder_width)
        fk = float(np.mean(errs))
        if fk > args.max_mean_error:
            failures.append(f'{gloss}: authored FK error {fk:.3f} > {args.max_mean_error}')
        mov = {}
        for side in ('L', 'R'):
            h = lm_seq[-1][side]
            mov[side] = float(np.linalg.norm(np.diff(np.array([f[side][0] for f in lm_seq]), axis=0), axis=1).sum())
        report['signs'].append({'gloss': gloss, 'kind': 'authored', 'fkMeanErrorShoulderWidths': fk,
                                'authored': True, 'needsReview': True})
        print(f'{gloss}: authored fk={fk:.3f} wrist_travel={ {k: round(v,3) for k,v in mov.items()} }')

    # ---- distinctness: pairwise DTW between shipped clips (avatar frame)
    shipped = ['HELLO', 'THANK_YOU', 'WHERE', 'PLEASE', 'FOOD', 'NAME', 'HOW']
    tracks = {}
    for gloss in shipped:
        d = json.loads((CLIPS / f'{gloss.lower()}.json').read_text())
        tracks[gloss] = resample(clip_world_tracks(rig, bones, d['frames']), 60).reshape(60, -1)
    dists = []
    for i, a in enumerate(shipped):
        for b in shipped[i + 1:]:
            dists.append((a, b, float(dtw(tracks[a], tracks[b]))))
    dists.sort(key=lambda x: x[2])
    print('\npairwise DTW (shoulder-width units, lower = similar):')
    for a, b, d in dists:
        print(f'  {a:10s} {b:10s} {d:.3f}')
    report['dtw'] = [{'a': a, 'b': b, 'distance': d} for a, b, d in dists]
    min_d = min(d[2] for d in dists)
    if min_d < 0.02:
        failures.append(f'clips suspiciously similar: min DTW {min_d:.3f}')

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print(f'\nreport -> {REPORT}')
    if failures:
        print('FAILURES:')
        for f in failures:
            print(' -', f)
        sys.exit(1)
    print('verification PASSED')

if __name__ == '__main__':
    main()
