"""Track B step 1 (REPORT_H): Convert S3 recordings -> data/s3_converted, verify
no S3 clip duplicates S1 or S2 (compare timestamps), print per-label counts and
hands-detected rates, list CORE labels with no S3 clips as NOT VERIFIED.
"""
import csv, json, subprocess, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
S3_RECORDINGS = ROOT / "recordings_s3"
S3_OUT = ROOT / "data" / "s3_converted"
S3_MANIFEST = S3_OUT / "manifest.csv"
S1_MANIFEST = ROOT / "data" / "s1_converted" / "manifest.csv"
S2_MANIFEST = ROOT / "data" / "s2_converted" / "manifest.csv"
CORE10 = ["hello", "thank you", "please", "yes", "no", "help", "water", "you", "me", "where"]
REST = "rest"


def hand_detected_frac(arr, hand):
    sl = slice(501, 522) if hand == "left" else slice(522, 543)
    return float(np.any(arr[:, sl, :] != 0.0, axis=(1, 2)).mean())


def main():
    S3_OUT.mkdir(parents=True, exist_ok=True)
    # Convert S3 recordings via the shared converter script.
    cmd = [sys.executable, str(ROOT / "recorder" / "convert_and_validate.py"),
           str(S3_RECORDINGS), "--out", str(S3_OUT)]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    s3 = list(csv.DictReader(open(S3_MANIFEST, encoding="utf-8")))
    s1 = list(csv.DictReader(open(S1_MANIFEST, encoding="utf-8")))
    s2 = list(csv.DictReader(open(S2_MANIFEST, encoding="utf-8")))

    print("=" * 70)
    print("STEP 1 — S3 conversion & duplicate verification")
    print("=" * 70)
    print(f"S3 clips converted: {len(s3)}")
    # session check
    bad_session = [r for r in s3 if r["session"] != "S3"]
    print(f"All session='S3': {len(bad_session) == 0}  ({len(bad_session)} bad)")
    for r in bad_session:
        print(f"  BAD SESSION: {r['json']} session={r['session']}")

    # duplicate check vs S1 and S2: same label+take+timestamp
    def key(r):
        return (r["label"], int(r["take"]), r["timestamp"])
    s1_keys = {key(r) for r in s1}
    s2_keys = {key(r) for r in s2}
    dups1 = [r for r in s3 if key(r) in s1_keys]
    dups2 = [r for r in s3 if key(r) in s2_keys]
    print(f"S3 clips duplicating S1 (label/take/timestamp): {len(dups1)}")
    for r in dups1:
        print(f"  DUP-S1: {r['json']}")
    print(f"S3 clips duplicating S2 (label/take/timestamp): {len(dups2)}")
    for r in dups2:
        print(f"  DUP-S2: {r['json']}")

    # per-label S3 counts + hands-detected rates
    by_label = defaultdict(list)
    for r in s3:
        by_label[r["label"]].append(r)
    print("\nPer-label S3 clip counts and hands-detected rate (mean fraction of frames):")
    print(f"{'label':14} {'clips':>6} {'L_det':>8} {'R_det':>8} {'either_det':>10}")
    for label in sorted(by_label):
        recs = by_label[label]
        ld = np.mean([hand_detected_frac(np.load(r["npy"]).astype(np.float32), "left")
                      for r in recs])
        rd = np.mean([hand_detected_frac(np.load(r["npy"]).astype(np.float32), "right")
                      for r in recs])
        ed = np.mean([max(hand_detected_frac(np.load(r["npy"]).astype(np.float32), "left"),
                          hand_detected_frac(np.load(r["npy"]).astype(np.float32), "right"))
                      for r in recs])
        print(f"{label:14} {len(recs):>6} {ld:8.3f} {rd:8.3f} {ed:10.3f}")

    core_no_s3 = [w for w in CORE10 if w not in by_label]
    print(f"\nCORE labels with NO S3 clips (NOT VERIFIED): "
          f"{core_no_s3 if core_no_s3 else 'NONE — all 10 CORE words present in S3'}")
    print(f"S3 labels present: {sorted(by_label.keys())}")
    print(f"S3 rest clips: {len(by_label.get('rest', []))} "
          f"(rest FPR = NOT VERIFIED if 0)")
    print("\nAll numbers PRELIMINARY — S3 clean held-out test.")


if __name__ == "__main__":
    main()
