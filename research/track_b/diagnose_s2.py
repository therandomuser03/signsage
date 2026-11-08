"""Track B step 1+2 diagnostics for S2 (cross-session).

Step 1:
  - Convert S2 recordings -> data/s2_converted (done via convert_and_validate.py).
  - Verify each JSON has session "S2".
  - Check no S2 clip duplicates an S1 clip (compare timestamps).
  - Print per-label S2 clip counts and hands-missing share.
  - List which CORE labels have NO S2 clips (NOT VERIFIED).
Step 2:
  - For each label present in BOTH sessions, compare S1 vs S2 active-hand label
    (Left/Right), presence rates, and flip rate. Flag labels where they differ.
"""
import csv, json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
S1_MANIFEST = ROOT / "data" / "s1_converted" / "manifest.csv"
S2_MANIFEST = ROOT / "data" / "s2_converted" / "manifest.csv"
CORE10 = ["hello", "thank you", "please", "yes", "no", "help", "water", "you", "me", "where"]


def load_manifest(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def hand_stats(arr):
    """Return (l_pres_frac, r_pres_frac, flip) for a (T,543,3) array."""
    l = np.any(arr[:, 501:522, :] != 0.0, axis=(1, 2))
    r = np.any(arr[:, 522:543, :] != 0.0, axis=(1, 2))
    lf, rf = float(l.mean()), float(r.mean())
    flip = bool(np.any(l & ~r) and np.any(r & ~l))
    return lf, rf, flip


def active_hand(lf, rf):
    return "Left" if lf >= rf else "Right"


def main():
    s1 = load_manifest(S1_MANIFEST)
    s2 = load_manifest(S2_MANIFEST)

    # --- step 1: session verification ---
    bad_session = [r for r in s2 if r["session"] != "S2"]
    print("=" * 70)
    print("STEP 1 — S2 conversion verification")
    print("=" * 70)
    print(f"S2 clips: {len(s2)}  all session='S2': {len(bad_session) == 0}")
    if bad_session:
        for r in bad_session:
            print(f"  BAD SESSION: {r['json']} session={r['session']}")

    # duplicate check vs S1: compare timestamps (and label+take)
    s1_ts = {(r["label"], int(r["take"]), r["timestamp"]) for r in s1}
    s2_ts = [(r["label"], int(r["take"]), r["timestamp"], r["json"]) for r in s2]
    dups = [(lab, tk, ts, j) for (lab, tk, ts, j) in s2_ts if (lab, tk, ts) in s1_ts]
    print(f"S2 clips duplicating an S1 clip (same label/take/timestamp): {len(dups)}")
    for d in dups:
        print(f"  DUP: {d}")

    # per-label S2 counts + hands-missing share
    by_label_s2 = defaultdict(list)
    for r in s2:
        by_label_s2[r["label"]].append(r)

    print("\nPer-label S2 clip counts and hands-missing share:")
    print(f"{'label':14} {'clips':>6} {'L_miss_mean':>12} {'R_miss_mean':>12} {'either>40%':>10}")
    s2_lm_means, s2_rm_means = [], []
    for label in sorted(by_label_s2):
        recs = by_label_s2[label]
        lm = np.mean([float(r["left_miss_frac"]) for r in recs])
        rm = np.mean([float(r["right_miss_frac"]) for r in recs])
        either40 = sum(1 for r in recs if float(r["left_miss_frac"]) > 0.40 or float(r["right_miss_frac"]) > 0.40)
        s2_lm_means.append(lm); s2_rm_means.append(rm)
        print(f"{label:14} {len(recs):>6} {lm:12.3f} {rm:12.3f} {either40:>10}/{len(recs)}")

    # CORE labels with NO S2 clips
    core_no_s2 = [w for w in CORE10 if w not in by_label_s2]
    print(f"\nCORE labels with NO S2 clips (NOT VERIFIED for cross-session): {core_no_s2 if core_no_s2 else 'NONE — all 10 CORE words present in S2'}")
    print(f"S2 labels present: {sorted(by_label_s2.keys())}")
    print(f"S2 rest clips: {len(by_label_s2.get('rest', []))} (rest FPR = NOT VERIFIED if 0)")

    # --- step 2: hand-label check S1 vs S2 ---
    print("\n" + "=" * 70)
    print("STEP 2 — Hand-label check S1 vs S2 (labels in both sessions)")
    print("=" * 70)
    print(f"{'label':14} {'S1 active':>10} {'S1 Lpres':>9} {'S1 Rpres':>9} {'S1 flip':>8} | "
          f"{'S2 active':>10} {'S2 Lpres':>9} {'S2 Rpres':>9} {'S2 flip':>8}  flag")
    by_label_s1 = defaultdict(list)
    for r in s1:
        by_label_s1[r["label"]].append(r)

    any_flag = False
    for label in sorted(set(by_label_s1) & set(by_label_s2)):
        s1_recs = by_label_s1[label]
        s2_recs = by_label_s2[label]
        s1_l, s1_r, s1_f = [], [], []
        for r in s1_recs:
            arr = np.load(r["npy"]).astype(np.float32)
            lf, rf, flip = hand_stats(arr)
            s1_l.append(lf); s1_r.append(rf); s1_f.append(flip)
        s2_l, s2_r, s2_f = [], [], []
        for r in s2_recs:
            arr = np.load(r["npy"]).astype(np.float32)
            lf, rf, flip = hand_stats(arr)
            s2_l.append(lf); s2_r.append(rf); s2_f.append(flip)
        s1_act = active_hand(np.mean(s1_l), np.mean(s1_r))
        s2_act = active_hand(np.mean(s2_l), np.mean(s2_r))
        s1_flip = np.mean(s1_f); s2_flip = np.mean(s2_f)
        flag = (s1_act != s2_act) or (abs(np.mean(s1_l) - np.mean(s2_l)) > 0.20) \
            or (abs(np.mean(s1_r) - np.mean(s2_r)) > 0.20) or (abs(s1_flip - s2_flip) > 0.30)
        if flag:
            any_flag = True
        print(f"{label:14} {s1_act:>10} {np.mean(s1_l):9.3f} {np.mean(s1_r):9.3f} {s1_flip:8.3f} | "
              f"{s2_act:>10} {np.mean(s2_l):9.3f} {np.mean(s2_r):9.3f} {s2_flip:8.3f}  "
              f"{'FLAG' if flag else 'ok'}")

    print(f"\nHand-label check flags (active-hand change >0.20 presence or >0.30 flip diff): "
          f"{'SOME' if any_flag else 'NONE'}")
    print("\nAll numbers PRELIMINARY, OPTIMISTIC — cross-session S1->S2, same signer(s)/setup presumed.")


if __name__ == "__main__":
    main()
