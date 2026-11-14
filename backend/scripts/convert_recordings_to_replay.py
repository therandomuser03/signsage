"""Convert existing converted recordings (research/track_b/data/*_converted) into
backend/captures_replay/<session>/<label>/<take>.npz in the format replay_capture.py
expects (array (T,543,3) plus true_label).

Only the 8 words plus rest are converted: hello, me, you, yes, no, help, water,
where, rest.  (thank you and please are excluded — they are not in the model
vocabulary.)

Read-only source: never writes or deletes anything in research/track_b/data/.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]  # backend/scripts/ -> repo root
TRACK_B = REPO_ROOT / "research" / "track_b"
OUT_DIR = REPO_ROOT / "backend" / "captures_replay"

EIGHT = ["hello", "me", "you", "yes", "no", "help", "water", "where"]
REST = "rest"
LABELS = set(EIGHT + [REST])

SESSIONS = ["s1_converted", "s2_converted", "s3_converted"]


def load_manifest(p: Path) -> list[dict]:
    return list(csv.DictReader(open(p, encoding="utf-8")))


def main() -> int:
    total = 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for session_dir in SESSIONS:
        manifest = TRACK_B / "data" / session_dir / "manifest.csv"
        if not manifest.is_file():
            print(f"Skipping missing manifest: {manifest}", file=sys.stderr)
            continue
        rows = load_manifest(manifest)
        session = session_dir.replace("_converted", "").upper()  # S1/S2/S3
        n_session = 0
        for r in rows:
            lab = r["label"]
            if lab not in LABELS:
                continue
            npy_raw = r["npy"]
            npy = Path(npy_raw)
            if not npy.is_absolute():
                # S1/S2 manifests store relative paths like data/s1_converted/...,
                # relative to research/track_b/. S3 stores absolute paths.
                npy = TRACK_B / npy_raw
            if not npy.is_file():
                print(f"  (skip missing npy: {npy})", file=sys.stderr)
                continue
            arr = np.load(npy).astype(np.float32)
            if arr.ndim != 3 or arr.shape[1:] != (543, 3):
                print(f"  (skip bad shape {npy}: {arr.shape})", file=sys.stderr)
                continue
            take = int(r["take"])
            label_dir = OUT_DIR / session / lab
            label_dir.mkdir(parents=True, exist_ok=True)
            out_path = label_dir / f"{take:02d}.npz"
            np.savez(
                out_path,
                landmarks=arr,
                true_label=np.array([lab]),
            )
            n_session += 1
            total += 1
        print(f"{session}: {n_session} clips converted -> {OUT_DIR}/{session.lower()}")
    print(f"Total converted: {total} clips -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
