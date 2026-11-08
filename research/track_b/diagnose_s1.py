"""Diagnose hand detection on S1 recordings only (research/track_b/recordings + data/s1_converted).

Per label:
  - mean fraction of frames with left hand detected / right hand detected
  - active hand (hand present in more frames, averaged over clips)
For "you": per-take detail.
Fraction of clips where the active hand flips (L-only frame AND R-only frame both present).
Also prints the CURRENT baseline CORE model top-10 confusion pairs (results_track_b.json).
Hand detected in a frame = at least one of the 21 hand landmarks nonzero.
"""
import csv, json, os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data" / "s1_converted" / "manifest.csv"
RESULTS = ROOT / "results_track_b.json"

CORE = ["hello", "thank you", "please", "yes", "no", "help", "water", "you", "me", "where"]
REST = "rest"


def hand_present(arr, hand):
    sl = slice(501, 522) if hand == "left" else slice(522, 543)
    return np.any(arr[:, sl, :] != 0.0, axis=(1, 2))


def main():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    print("=" * 72)
    print("HAND-DETECTION DIAGNOSTICS — S1 ONLY  (labels from data/s1_converted)")
    print("=" * 72)
    print(f"{'label':14} {'clips':>5} {'L_pres':>7} {'R_pres':>7} {'active':>7} {'flip_frac':>9}  detail")
    print("-" * 72)

    overall_flip = 0
    overall_clips = 0
    for label in sorted(by_label):
        recs = by_label[label]
        l_pres_list, r_pres_list = [], []
        n_flip = 0
        for r in recs:
            arr = np.load(r["npy"]).astype(np.float32)
            lp = hand_present(arr, "left")
            rp = hand_present(arr, "right")
            l_pres_list.append(lp.mean())
            r_pres_list.append(rp.mean())
            # active hand per frame: L-only / R-only
            has_l_only = bool(np.any(lp & ~rp))
            has_r_only = bool(np.any(rp & ~lp))
            if has_l_only and has_r_only:
                n_flip += 1
        l_mean = float(np.mean(l_pres_list))
        r_mean = float(np.mean(r_pres_list))
        active = "Left" if l_mean >= r_mean else "Right"
        flip_frac = n_flip / len(recs)
        overall_flip += n_flip
        overall_clips += len(recs)
        detail = ""
        if label == "you":
            takes = sorted(recs, key=lambda r: int(r["take"]))
            detail = " | ".join(
                f"take{r['take']} L={hand_present(np.load(r['npy']),'left').mean():.2f} "
                f"R={hand_present(np.load(r['npy']),'right').mean():.2f}"
                for r in takes[:6]
            )
            if len(takes) > 6:
                detail += f" ... ({len(takes)} takes total)"
        print(f"{label:14} {len(recs):>5} {l_mean:7.3f} {r_mean:7.3f} {active:>7} {flip_frac:9.3f}  {detail}")

    print("-" * 72)
    print(f"{'OVERALL':14} {overall_clips:>5} {'':7} {'':7} {'':>7} {overall_flip/overall_clips:9.3f}")
    print("active hand = hand present in more frames (per-clip mean). "
          "flip = clip has >=1 left-only-active frame and >=1 right-only-active frame.")
    print()

    # Current baseline CORE model top-10 confusions
    print("=" * 72)
    print("CURRENT BASELINE CORE MODEL — top-10 confusion pairs (results_track_b.json)")
    print("=" * 72)
    d = json.load(open(RESULTS))
    p = d["pooled"]["CORE"]
    print(f"1D-CNN CORE pooled: acc={p['acc']:.3f} macroF1={p['macro_f1']:.3f} "
          f"baseline(majority)={p['baseline']:.3f} chance(uniform)={1/len(p['per_class']):.3f}")
    for k, (c, i, j) in enumerate(p["top10"], 1):
        print(f"  {k:2d}. {i:14} -> {j:14}  ({c})")
    print()
    print("Per-class F1 (CORE pooled, current baseline):")
    for name, m in sorted(p["per_class"].items(), key=lambda kv: kv[1]["f1"]):
        print(f"  {name:14} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}")


if __name__ == "__main__":
    main()
