"""Per-label summary of labeled captures.

Reads backend/captures_labeled/<label>/*.npz and prints, per label: the count
of captures, mean frames, and mean left/right hand-present fractions.

This script only reads files; it never writes or deletes anything.

Usage:
    python scripts/count_labeled_captures.py
    python scripts/count_labeled_captures.py --captures-dir backend/captures_labeled
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]  # backend/scripts/ -> repo root
DEFAULT_CAPTURES_DIR = REPO_ROOT / "backend" / "captures_labeled"

# MediaPipe slots (labels ignored by the v3 recipe): left hand 501-522, right 522-543.
_HAND_LEFT = slice(501, 522)
_HAND_RIGHT = slice(522, 543)


def _hand_fractions(landmarks: np.ndarray) -> tuple[float, float]:
    """Fraction of frames with left / right hand present."""
    n = len(landmarks)
    if n == 0:
        return 0.0, 0.0
    left = np.any(landmarks[:, _HAND_LEFT, :] != 0.0, axis=(1, 2))
    right = np.any(landmarks[:, _HAND_RIGHT, :] != 0.0, axis=(1, 2))
    return float(left.mean()), float(right.mean())


def _mean(rows: list[float]) -> float:
    return float(sum(rows) / len(rows)) if rows else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Count labeled captures per label with mean frames and hand-present fractions.",
    )
    parser.add_argument(
        "--captures-dir",
        default=str(DEFAULT_CAPTURES_DIR),
        help=f"Directory with <label>/*.npz captures (default: {DEFAULT_CAPTURES_DIR})",
    )
    args = parser.parse_args(argv)

    captures_dir = Path(args.captures_dir)
    if not captures_dir.is_dir():
        print(f"No captures directory: {captures_dir}")
        print("No labeled captures found.")
        return 0

    print("=== Labeled capture counts ===")
    print(f"Captures dir: {captures_dir}")

    stats: dict[str, list[tuple[int, float, float]]] = {}
    for label_dir in sorted(p for p in captures_dir.iterdir() if p.is_dir()):
        label = label_dir.name
        rows = []
        for npz in sorted(label_dir.glob("*.npz")):
            try:
                with np.load(npz) as z:
                    landmarks = np.asarray(z["landmarks"])
            except Exception as exc:  # noqa: BLE001 - report and keep going
                print(f"  (skip {npz.name}: {exc})", file=sys.stderr)
                continue
            l, r = _hand_fractions(landmarks)
            rows.append((len(landmarks), l, r))
        if rows:
            stats[label] = rows

    if not stats:
        print("No labeled captures found.")
        return 0

    print(f"{'LABEL':<12} {'COUNT':<7} {'MEAN_T':<9} {'MEAN_LEFT':<12} {'MEAN_RIGHT':<12}")
    total = 0
    all_t, all_l, all_r = [], [], []
    for label in sorted(stats):
        rows = stats[label]
        total += len(rows)
        ts = [row[0] for row in rows]
        ls = [row[1] for row in rows]
        rs = [row[2] for row in rows]
        all_t.extend(ts)
        all_l.extend(ls)
        all_r.extend(rs)
        print(
            f"{label:<12} {len(rows):<7} {_mean(ts):<9.1f} "
            f"{_mean(ls):<12.3f} {_mean(rs):<12.3f}"
        )
    print(
        f"{'TOTAL':<12} {total:<7} {_mean(all_t):<9.1f} "
        f"{_mean(all_l):<12.3f} {_mean(all_r):<12.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())