"""Step 5: convert recorder clips (JSON) -> .npy + manifest.csv, then validate.

Usage:
  python convert_and_validate.py <json_file_or_dir> [--out <dir>] [--features-clip <npy_file>]

Each input JSON is the clip saved by research/track_b/recorder/recorder.html:
{
  "extractor": {...},
  "label": "hello", "session": "S1", "take": 1,
  "timestamp": "...", "duration_s": 2.5, "effective_fps": 12.5,
  "frames": 31,
  "hands_missing_fraction": {"left": 0.0, "right": 0.13},
  "landmarks": [[[x,y,z] x 543] x T]   # (T, 543, 3), SignSage component order
}

For every clip:
  * validate shape == (T, 543, 3)
  * save <stem>.npy (float32)
  * report per-clip hands-missing fractions (recomputed from the array)
  * append a row to manifest.csv

Finally runs research/track_b/features.py (extract_features, layout='signsage')
on one clip and prints the output shape (expect (60, 98)).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# research/track_b is the parent of this file's directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import extract_features  # noqa: E402

POSE = (0, 33)
FACE = (33, 501)
LHAND = (501, 522)   # 21 landmarks
RHAND = (522, 543)   # 21 landmarks
LANDMARK_N = 543


def missing_fraction(arr, start, count):
    """Fraction of frames where every landmark in [start, start+count) is [0,0,0]."""
    if arr.shape[0] == 0:
        return 0.0
    part = arr[:, start:start + count, :]
    return float(np.all(part == 0, axis=(1, 2)).mean())


def validate_shape(arr, source):
    if arr.ndim != 3 or arr.shape[1] != LANDMARK_N or arr.shape[2] != 3:
        raise ValueError(
            f"{source}: expected (T, 543, 3) landmarks, got shape {arr.shape}"
        )
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{source}: landmarks must be numeric, got {arr.dtype}")


def convert_one(json_path, out_dir):
    with open(json_path, encoding="utf-8") as f:
        clip = json.load(f)

    label = clip["label"]
    session = clip["session"]
    take = clip["take"]
    timestamp = clip.get("timestamp")
    effective_fps = clip.get("effective_fps")
    frames_reported = clip.get("frames")
    reported_missing = clip.get("hands_missing_fraction", {})

    arr = np.asarray(clip["landmarks"], dtype=np.float32)
    validate_shape(arr, json_path)

    T = arr.shape[0]
    left_miss = missing_fraction(arr, LHAND[0], LHAND[1] - LHAND[0])
    right_miss = missing_fraction(arr, RHAND[0], RHAND[1] - RHAND[0])

    npy_path = out_dir / (json_path.stem + ".npy")
    np.save(npy_path, arr)

    row = {
        "json": str(json_path),
        "npy": str(npy_path),
        "label": label,
        "session": session,
        "take": take,
        "timestamp": timestamp,
        "effective_fps": effective_fps,
        "frames": T,
        "frames_reported": frames_reported,
        "left_miss_frac": round(left_miss, 6),
        "right_miss_frac": round(right_miss, 6),
        "left_miss_frac_reported": reported_missing.get("left"),
        "right_miss_frac_reported": reported_missing.get("right"),
        "shape": f"{arr.shape}",
    }
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="clip JSON file or directory of clip JSONs")
    ap.add_argument("--out", default=None, help="output dir (default: beside the input)")
    ap.add_argument(
        "--features-clip",
        default=None,
        help=".npy (or JSON) to run features.py extract_features on (default: first converted clip)",
    )
    args = ap.parse_args()

    inp = Path(args.input)
    if inp.is_dir():
        jsons = sorted(inp.glob("*.json"))
    elif inp.is_file() and inp.suffix == ".json":
        jsons = [inp]
    else:
        raise SystemExit(f"input not found or not a JSON file/dir: {inp}")

    out_dir = Path(args.out) if args.out else (inp if inp.is_dir() else inp.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    fieldnames = [
        "json", "npy", "label", "session", "take", "timestamp", "effective_fps",
        "frames", "frames_reported", "left_miss_frac", "right_miss_frac",
        "left_miss_frac_reported", "right_miss_frac_reported", "shape",
    ]

    rows = []
    for json_path in jsons:
        try:
            row = convert_one(json_path, out_dir)
            rows.append(row)
            print(
                f"OK  {json_path.name} -> {row['npy'].split('/')[-1]}  "
                f"shape={row['shape']} frames={row['frames']} "
                f"left_miss={row['left_miss_frac']:.3f} right_miss={row['right_miss_frac']:.3f}"
            )
        except Exception as exc:
            print(f"FAIL {json_path}: {exc}", file=sys.stderr)

    if not rows:
        raise SystemExit("no clips converted")

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nmanifest: {manifest_path} ({len(rows)} clips)")
    print(
        f"hands-missing summary: left mean={np.mean([r['left_miss_frac'] for r in rows]):.3f} "
        f"right mean={np.mean([r['right_miss_frac'] for r in rows]):.3f} "
        f"clips flagged >40%: "
        f"{sum(1 for r in rows if r['left_miss_frac'] > 0.40 or r['right_miss_frac'] > 0.40)}"
    )

    # --- run features.py on one clip (expect (60, 98)) ---
    feat_source = Path(args.features_clip) if args.features_clip else Path(rows[0]["npy"])
    if feat_source.suffix == ".json":
        with open(feat_source, encoding="utf-8") as f:
            arr = np.asarray(json.load(f)["landmarks"], dtype=np.float32)
    else:
        arr = np.load(feat_source)
    validate_shape(arr, feat_source)
    feat = extract_features(arr, layout="signsage")
    print(
        f"features.extract_features({feat_source.name}, layout='signsage') "
        f"-> {tuple(feat.shape)}  (expect (60, 98))"
    )
    assert feat.shape == (60, 98), f"unexpected feature shape {feat.shape}"


if __name__ == "__main__":
    main()
