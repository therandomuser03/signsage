#!/usr/bin/env python3
"""Render the source landmark skeleton of each sign (front orthographic view)
for side-by-side comparison with the avatar renders.

Usage: python3 tools/render_landmarks.py OUT_DIR gloss frac [gloss frac ...]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signlib import landmarks as LM
from build_sign_clips import SIGN_SOURCES, active_window, video_dims_for

REPO = Path(__file__).resolve().parent.parent

# medoid source per sign, matching build_sign_clips (retargeted clips only)
MEDOIDS = {"HELLO": "hello__INCLUDE__00581__MVI_9955.npy",
           "THANK_YOU": "thank_you__INCLUDE__00612__MVI_0059.npy",
           "WHERE": "where.npy"}

HAND_BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
               (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
POSE_BONES = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24)]

def draw(ax, clip, frame_idx, dims):
    W, H = dims
    ax.set_facecolor("#1b2230")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    pose = clip[frame_idx, :33]
    for a, b in POSE_BONES:
        if np.any(pose[a]) and np.any(pose[b]):
            ax.plot([pose[a,0]*W, pose[b,0]*W], [pose[a,1]*H, pose[b,1]*H], c="#5b8dd9", lw=2, alpha=0.8)
    a2, b2 = LM.hand_presence(clip)
    for off, color in ((33, "#f2a541"), (54, "#7bd88a")):
        lm = clip[frame_idx, off:off+21]
        if not np.any(lm):
            continue
        for a, b in HAND_BONES:
            if np.any(lm[a]) and np.any(lm[b]):
                ax.plot([lm[a,0]*W, lm[b,0]*W], [lm[a,1]*H, lm[b,1]*H], c=color, lw=2.5)
        ax.scatter(lm[:,0]*W, lm[:,1]*H, s=14, c=color)
    ax.set_title("source landmarks", color="#cfd6e4", fontsize=10)

def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    args = sys.argv[2:]
    for gloss, frac_s in zip(args[::2], args[1::2]):
        frac = float(frac_s)
        gloss = gloss.upper()
        cfg = SIGN_SOURCES.get(gloss)
        if cfg is None:
            # authored approximation: no source to draw
            fig, ax = plt.subplots(figsize=(5.12, 6.4), dpi=100, facecolor="#1b2230")
            ax.set_facecolor("#1b2230"); ax.axis("off")
            ax.text(0.5, 0.5, "authored approximation\n(no source recording)",
                    color="#cfd6e4", ha="center", va="center", fontsize=14, wrap=True)
            fig.savefig(out / f"{gloss.lower()}_{frac}_lm.png", bbox_inches=None, facecolor="#1b2230")
            plt.close(fig)
            continue
        # resolve medoid file from the manifest clip instead of the table:
        medoid = MEDOIDS[gloss]
        src = None
        for d in cfg['dirs']:
            p = d / medoid
            if p.exists():
                src = p
        clip = LM.load_clip(src)
        dims = video_dims_for(src) or (854, 480)
        # match the pipeline's active-window trim so frac aligns with the clip
        a, b = active_window(clip, dims[0], dims[1])
        T = b - a
        idx = a + round(frac * (T - 1))
        fig, ax = plt.subplots(figsize=(5.12, 6.4), dpi=100, facecolor="#1b2230")
        draw(ax, clip, idx, dims)
        fig.savefig(out / f"{gloss.lower()}_{frac}_lm.png", bbox_inches=None, facecolor="#1b2230")
        plt.close(fig)

if __name__ == "__main__":
    main()
