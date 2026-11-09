"""Read-only summary of word-mode debug captures.

Reads backend/debug_captures/*.npz (sorted by filename timestamp) and prints,
per capture: T frames, left/right hand-present fractions, top-3 predictions
with probabilities, and CORRECT/WRONG when --labels are given.

Then prints per-word accuracy and, for each word, mean T and mean
hand-present fraction beside the same stats for the S3 clips of that word
(research/track_b/data/s3_converted). If that directory lives somewhere else,
the script searches the repository for a directory named s3_converted and
reports what it found (or that it found nothing).

This script only reads files; it never writes or deletes anything.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]  # backend/scripts/ -> repo root
DEFAULT_CAPTURES_DIR = REPO_ROOT / "backend" / "debug_captures"
DEFAULT_S3_DIR = REPO_ROOT / "research" / "track_b" / "data" / "s3_converted"

_HAND_LEFT = slice(501, 522)   # MediaPipe slot 0 (labels ignored by the v3 recipe)
_HAND_RIGHT = slice(522, 543)  # MediaPipe slot 1


def _hand_fractions(landmarks: np.ndarray) -> tuple[float, float, float]:
    """Fraction of frames with left / right / any hand present."""
    n = len(landmarks)
    if n == 0:
        return 0.0, 0.0, 0.0
    left = np.any(landmarks[:, _HAND_LEFT, :] != 0.0, axis=(1, 2))
    right = np.any(landmarks[:, _HAND_RIGHT, :] != 0.0, axis=(1, 2))
    return float(left.mean()), float(right.mean()), float(np.logical_or(left, right).mean())


def _find_s3_dir(repo_root: Path) -> tuple[Path | None, list[Path]]:
    """Locate the S3 converted-clips directory; report all places looked."""
    candidates = [
        repo_root / "research" / "track_b" / "data" / "s3_converted",
        repo_root / "research" / "track_b" / "s3_converted",
        repo_root / "data" / "s3_converted",
    ]
    for c in candidates:
        if c.is_dir():
            return c, candidates
    found = [p for p in repo_root.rglob("s3_converted") if p.is_dir()]
    return (found[0] if found else None), candidates + found


def _load_capture(path: Path) -> dict:
    with np.load(path) as z:
        landmarks = np.asarray(z["landmarks"])
        top3_labels = [str(t) for t in z["top3_labels"].tolist()]
        top3_probs = [float(p) for p in z["top3_probs"].tolist()]
        label = z["label"]
        predicted = str(label.tolist()[0] if hasattr(label, "tolist") else label)
    return {
        "path": path,
        "landmarks": landmarks,
        "top3_labels": top3_labels,
        "top3_probs": top3_probs,
        "predicted": predicted,
    }


def _s3_clips(s3_dir: Path) -> dict[str, list[dict]]:
    """Group S3 clips by word parsed from WORD__S3__takeNN__TS.npy."""
    by_word: dict[str, list[dict]] = {}
    for path in sorted(s3_dir.glob("*.npy")):
        word = path.name.split("__", 1)[0]
        try:
            arr = np.load(path)
            if arr.ndim != 3 or arr.shape[1:] != (543, 3):
                print(f"  (skip {path.name}: unexpected shape {arr.shape})", file=sys.stderr)
                continue
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"  (skip {path.name}: {exc})", file=sys.stderr)
            continue
        l, r, any_h = _hand_fractions(arr)
        by_word.setdefault(word, []).append(
            {"path": path, "t": len(arr), "l": l, "r": r, "any": any_h}
        )
    return by_word


def _mean(rows: list[dict], key: str) -> float:
    return float(sum(row[key] for row in rows) / len(rows)) if rows else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize word-mode debug captures (read-only).",
    )
    parser.add_argument(
        "--captures-dir",
        default=str(DEFAULT_CAPTURES_DIR),
        help=f"Directory with *.npz captures (default: {DEFAULT_CAPTURES_DIR})",
    )
    parser.add_argument(
        "--s3-dir",
        default=str(DEFAULT_S3_DIR),
        help=f"Directory with S3 *.npy clips (default: {DEFAULT_S3_DIR})",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help='True words in timestamp order, e.g. --labels "where,where,you"',
    )
    args = parser.parse_args(argv)

    captures_dir = Path(args.captures_dir)
    if not captures_dir.is_dir():
        print(f"No captures directory: {captures_dir}")
        print("No debug captures found.")
        return 0

    capture_files = sorted(captures_dir.glob("*.npz"))
    if not capture_files:
        print(f"No debug captures found in {captures_dir}")
        return 0

    labels = [w.strip() for w in args.labels.split(",") if w.strip()] if args.labels else []
    if labels and len(labels) != len(capture_files):
        print(
            f"Warning: {len(labels)} labels for {len(capture_files)} captures; "
            "labeling the first min(len) captures in timestamp order.",
            file=sys.stderr,
        )

    print("=== Word-mode debug capture summary ===")
    print(f"Captures dir: {captures_dir}")
    print(f"Captures found: {len(capture_files)} (sorted by filename timestamp)")
    if labels:
        print(f"True labels ({len(labels)}): {', '.join(labels)}")
    else:
        print("True labels: none (--labels not given)")

    # ---- Per capture ----
    rows = []
    for i, path in enumerate(capture_files):
        row = _load_capture(path)
        l, r, any_h = _hand_fractions(row["landmarks"])
        row.update(t=len(row["landmarks"]), l=l, r=r, any=any_h)
        true_word = labels[i] if i < len(labels) else None
        row["true"] = true_word
        if true_word is not None:
            row["correct"] = row["predicted"] == true_word
            verdict = "CORRECT" if row["correct"] else "WRONG"
        else:
            row["correct"] = None
            verdict = "—"
        top3 = ", ".join(
            f"{lab} {prob:.2f}" for lab, prob in zip(row["top3_labels"], row["top3_probs"])
        )
        print(
            f"[{i + 1}] {path.name}  T={row['t']}  L={l:.3f}  R={r:.3f}  "
            f"Any={any_h:.3f}  top3: {top3}  {verdict}"
            + (f" (true={true_word}, predicted={row['predicted']})" if true_word else "")
        )
        rows.append(row)

    # ---- Per word ----
    s3_dir = Path(args.s3_dir)
    if s3_dir.is_dir():
        resolved_s3, searched = s3_dir, []
    else:
        resolved_s3, searched = _find_s3_dir(REPO_ROOT)
    if resolved_s3 is None:
        print(
            f"\nS3 clips dir not found at {s3_dir} (also looked at: "
            f"{', '.join(str(s) for s in searched)}); S3 stats omitted.",
        )
    else:
        note = "" if resolved_s3 == s3_dir else f" (not at {s3_dir}; found at {resolved_s3})"
        print(f"\nS3 clips dir: {resolved_s3}{note}")
    s3_by_word = _s3_clips(resolved_s3) if resolved_s3 else {}

    labeled = [row for row in rows if row["true"] is not None]
    words = sorted({row["true"] for row in labeled} | set(s3_by_word.keys()))
    print("\n--- Per-word summary: labeled captures vs S3 clips ---")
    print(
        f"{'WORD':<10} {'ACC':<8} {'CAPTURES':<26} {'S3 CLIPS':<22}"
    )
    print(
        f"{'':<10} {'correct/total':<8} "
        f"{'n':<4}{'mean_T':<8}{'mean_L':<8}{'mean_R':<8}{'mean_Any':<8} "
        f"{'n':<4}{'mean_T':<8}{'mean_L':<8}{'mean_R':<8}{'mean_Any':<8}"
    )
    for word in words:
        w_rows = [row for row in labeled if row["true"] == word]
        if w_rows:
            correct = sum(1 for row in w_rows if row["correct"])
            acc = f"{correct}/{len(w_rows)} ({correct / len(w_rows) * 100:.0f}%)"
            cap = (
                f"{len(w_rows):<4}"
                f"{_mean(w_rows, 't'):<8.1f}"
                f"{_mean(w_rows, 'l'):<8.3f}"
                f"{_mean(w_rows, 'r'):<8.3f}"
                f"{_mean(w_rows, 'any'):<8.3f}"
            )
        else:
            acc = "—"
            cap = f"{'-':<4}{'-':<8}{'-':<8}{'-':<8}{'-':<8}"
        s3_rows = s3_by_word.get(word, [])
        if s3_rows:
            s3 = (
                f"{len(s3_rows):<4}"
                f"{_mean(s3_rows, 't'):<8.1f}"
                f"{_mean(s3_rows, 'l'):<8.3f}"
                f"{_mean(s3_rows, 'r'):<8.3f}"
                f"{_mean(s3_rows, 'any'):<8.3f}"
            )
        else:
            s3 = f"{'-':<4}{'-':<8}{'-':<8}{'-':<8}{'-':<8}"
        print(f"{word:<10} {acc:<8} {cap} {s3}")

    # Overall accuracy across labeled captures
    if labeled:
        correct = sum(1 for row in labeled if row["correct"])
        print(
            f"\nOverall accuracy: {correct}/{len(labeled)} "
            f"({correct / len(labeled) * 100:.1f}%)"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
