"""Shared helpers for the temporal-crop experiment (REPORT_K).

Builds "tight temporal crop" copies of the S1-S4 word clips:

  crop window = hand-active span (first/last frame with any nonzero hand
  block, i.e. indices 501-521 or 522-542 with any nonzero x/y/z)
  expanded by 400 ms on each side, using the per-clip effective_fps from the
  conversion manifest (frames are not individually timestamped). Window is
  clamped to the clip; clips with no hands are kept whole. Minimum length is
  12 frames (expanded symmetrically, clamped at the clip edges).

A second variant ("Auto-mode-like") applies a seeded uniform jitter of
+/-100 ms to each boundary independently.

Everything here is a mechanical data transform — no model changes.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

LABELS = ["hello", "me", "you", "yes", "no", "help", "water", "where", "rest"]
PAD_MS = 400.0
MIN_FRAMES = 12
JITTER_MS = 100.0
JITTER_SEED = 20261115


def hands_present_mask(arr: np.ndarray) -> np.ndarray:
    """(T,) bool — any nonzero value in the two hand blocks (501-521, 522-542)."""
    return np.any(arr[:, 501:543, :] != 0.0, axis=(1, 2))


def hand_active_span(arr: np.ndarray) -> tuple[int, int]:
    """(first, last) frame indices with a nonzero hand block; whole clip if none."""
    idx = np.flatnonzero(hands_present_mask(arr))
    if len(idx) == 0:
        return 0, arr.shape[0] - 1
    return int(idx[0]), int(idx[-1])


def cropped_window(
    arr: np.ndarray,
    fps: float,
    pad_ms: float = PAD_MS,
    jitter_ms: tuple[float, float] | None = None,
    min_frames: int = MIN_FRAMES,
) -> tuple[int, int]:
    """Return (start, end) inclusive frame indices for the hand-active span + pad.

    jitter_ms: optional (start_jitter_ms, end_jitter_ms), each in [-JITTER_MS, +JITTER_MS];
    positive moves that boundary outward (more context), negative crops tighter.
    """
    T = arr.shape[0]
    span0, span1 = hand_active_span(arr)
    pad0 = pad_ms if jitter_ms is None else pad_ms + jitter_ms[0]
    pad1 = pad_ms if jitter_ms is None else pad_ms + jitter_ms[1]
    start = int(np.floor(span0 - (pad0 / 1000.0) * fps))
    end = int(np.ceil(span1 + (pad1 / 1000.0) * fps))
    # minimum length: expand symmetrically, then clamp, then mop up the clamp side
    length = end - start + 1
    if length < min_frames:
        extra = min_frames - length
        start -= extra // 2
        end += extra - extra // 2
    start = max(0, start)
    end = min(T - 1, end)
    if end - start + 1 < min_frames:
        deficit = min_frames - (end - start + 1)
        if start == 0:
            end = min(T - 1, end + deficit)
        else:
            start = max(0, start - deficit)
    return start, end


def load_manifest(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path, encoding="utf-8")))


def word_rows(rows: list[dict]) -> list[dict]:
    """Rows whose label is one of the 8 words or rest (file order preserved)."""
    return [r for r in rows if r["label"] in LABELS]


def miss_fracs(arr: np.ndarray) -> tuple[float, float]:
    """Recompute left/right hand miss fractions over the whole clip."""
    left = np.any(arr[:, 501:522, :] != 0.0, axis=(1, 2))
    right = np.any(arr[:, 522:543, :] != 0.0, axis=(1, 2))
    t = len(arr)
    return float((~left).sum()) / t, float((~right).sum()) / t


def write_manifest(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


MANIFEST_FIELDS = [
    "json", "npy", "label", "session", "take", "timestamp", "effective_fps",
    "frames", "frames_reported", "left_miss_frac", "right_miss_frac",
    "left_miss_frac_reported", "right_miss_frac_reported", "shape",
    "source_npy", "crop_start_frame", "crop_end_frame",
]


def build_variant(source_dir_name: str, dest_name: str, jitter: bool, write_manifest: bool = False) -> tuple[list[dict], Path]:
    """Build the cropped (or cropped+jittered) copy of one session's 9-class clips.

    Returns (rows, dest_dir). Manifests are written by the caller
    (build_crops.py) so a single combined manifest can be produced for all
    sessions.
    """
    dest = DATA / dest_name
    dest.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict] = []
    suffix = "__crop400j" if jitter else "__crop400"
    manifest_path = DATA / source_dir_name / "manifest.csv"
    for r in word_rows(load_manifest(manifest_path)):
        src_npy = Path(r["npy"])
        arr = np.load(src_npy).astype(np.float32)
        fps = float(r["effective_fps"])
        if jitter:
            # Per-clip deterministic jitter, independent of processing order.
            seed = JITTER_SEED + int(hashlib.sha1(src_npy.stem.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            j_start = float(rng.uniform(-JITTER_MS, JITTER_MS))
            j_end = float(rng.uniform(-JITTER_MS, JITTER_MS))
        else:
            j_start = j_end = 0.0
        start, end = cropped_window(
            arr, fps, PAD_MS, (j_start, j_end) if jitter else None, MIN_FRAMES
        )
        out = arr[start:end + 1]
        stem = src_npy.stem
        out_path = dest / f"{stem}{suffix}.npy"
        np.save(out_path, out)
        left_miss, right_miss = miss_fracs(out)
        row = dict(r)
        row.update({
            "json": r.get("json", ""),
            "npy": str(out_path),
            "frames": len(out),
            "frames_reported": len(out),
            "left_miss_frac": f"{left_miss:.6f}",
            "right_miss_frac": f"{right_miss:.6f}",
            "left_miss_frac_reported": f"{left_miss:.6f}",
            "right_miss_frac_reported": f"{right_miss:.6f}",
            "shape": f"({out.shape[0]}, {out.shape[1]}, {out.shape[2]})",
            "source_npy": str(src_npy),
            "crop_start_frame": start,
            "crop_end_frame": end,
        })
        if jitter:
            row["jitter_start_ms"] = f"{j_start:.1f}"
            row["jitter_end_ms"] = f"{j_end:.1f}"
        rows_out.append(row)
    if write_manifest:
        fields = MANIFEST_FIELDS + (["jitter_start_ms", "jitter_end_ms"] if jitter else [])
        write_manifest(dest / "manifest.csv", rows_out, fields)
    return rows_out, dest
