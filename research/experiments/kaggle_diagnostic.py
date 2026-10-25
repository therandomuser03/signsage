#!/usr/bin/env python3
"""Bounded Kaggle diagnostic for SignSage, using the existing Version 4 assets.

Usage (from a fresh Kaggle session with Version 4 output attached):
    python3 -m signsage.research.experiments.kaggle_diagnostic \
        --input-root /kaggle/input \
        --out /kaggle/working/s2t_diag1 \
        --timeout-seconds 1440 \
        --expected-tokenizer-fingerprint 2d88a5de48318cb9

This is NOT full training. It validates the cache/tokenizer/split identity,
reuses the existing pose shards (read-only), starts fresh weights, runs only
--stages overfit,train, applies a bounded external timeout, and writes
subprocess_stdout_stderr.log, pose_preflight.json, gate_report.json,
training_history.json, and any checkpoint/recovery artifacts. It does NOT
run evaluate/export or claim quality success.

Safety rules preserved:
- No resume from collapsed checkpoints (fresh out, no --resume).
- No overwrite of results.zip or existing artifacts.
- No full dataset download (uses existing shard ZIPs).
- No GPU-heavy execution unless the user explicitly runs with GPU enabled.
- Timeout kills the subprocess; partial artifacts are preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import research.experiments.train_sign_to_text as T


MANIFEST_REL = "s2t_small6/pose_cache/manifest.json"
CSV_REL = "isign/iSign_v1.1.csv"
EXPECTED_CACHE_N = 7500


def resolve_pose_shard(cache_dir: Path, shard_name: str) -> Path:
    """Resolve a manifest shard to either its ZIP or extracted directory."""
    archive = cache_dir / shard_name
    if archive.is_file():
        return archive
    if shard_name.endswith(".zip"):
        extracted = cache_dir / shard_name[:-4]
        if extracted.is_dir():
            return extracted
    raise FileNotFoundError(
        f"Manifest shard {shard_name!r} has neither an archive "
        f"nor a matching extracted directory under {cache_dir}"
    )


def discover_input_root(input_root: str | Path) -> Path:
    """Find the unique attached input root holding the Version 4 assets.

    Kaggle mounts attached datasets under /kaggle/input/<slug>/, so the assets
    may live one level below the nominal input root. Search the given root, its
    immediate dataset subdirectories, and a local mirror (kaggle_input_v4) used
    for off-Kaggle checks. Require exactly one root containing both the
    pose-cache manifest and the iSign CSV. Never fabricates or substitutes inputs.
    """
    root = Path(input_root)
    candidates: list[Path] = []
    seen: set[Path] = set()

    def consider(p: Path) -> None:
        try:
            p = p.resolve()
            if p in seen or not p.is_dir():
                return
            has_manifest = (p / MANIFEST_REL).is_file()
            has_csv = (p / CSV_REL).is_file()
        except OSError:
            return
        seen.add(p)
        if has_manifest and has_csv:
            candidates.append(p)

    consider(root)
    if root.is_dir():
        try:
            children = sorted(root.iterdir())
        except OSError:
            children = []
        for child in children:
            try:
                if child.is_dir():
                    consider(child)
            except OSError:
                continue
    for cand in (Path("kaggle_input_v4"), Path.cwd() / "kaggle_input_v4", root / "kaggle_input_v4"):
        consider(cand)

    if not candidates:
        searched = [str(root)]
        if root.is_dir():
            searched += [str(c) for c in sorted(root.iterdir()) if c.is_dir()]
        raise SystemExit(
            "No input root contains both "
            f"{MANIFEST_REL} and {CSV_REL}. Searched: {searched}. Attach the Version 4 "
            "input dataset (or pass --input-root pointing at it) and re-run."
        )
    if len(candidates) > 1:
        raise SystemExit(f"Ambiguous input roots contain the required data: {candidates}")
    return candidates[0]


def validate_input(root: Path, expected_tokenizer_fingerprint: str) -> dict:
    """Validate all required inputs under root (read-only; never copies shards).

    Returns the loaded manifest plus paths to the CSV, pose-cache dir, tokenizer
    dir, auto_subset.json and feasibility.json. Raises SystemExit with a clear
    message listing missing files when anything is absent or invalid.
    """
    manifest_path = root / MANIFEST_REL
    csv_path = root / CSV_REL
    cache_dir = root / "s2t_small6" / "pose_cache"
    tok_dir = root / "s2t_small6" / "tokenizer"

    required_small = {
        "manifest.json": manifest_path,
        "iSign_v1.1.csv": csv_path,
        "auto_subset.json": root / "s2t_small6" / "auto_subset.json",
        "feasibility.json": root / "s2t_small6" / "feasibility.json",
        "signsage_en_sp.model": tok_dir / "signsage_en_sp.model",
        "signsage_en_sp.vocab": tok_dir / "signsage_en_sp.vocab",
        "tokens.npz": tok_dir / "tokens.npz",
        "tokens_meta.json": tok_dir / "tokens_meta.json",
    }
    missing = [name for name, p in required_small.items() if not p.is_file()]
    if missing:
        raise SystemExit(f"Required inputs missing under {root}: {missing}")

    manifest = T.load_strict_manifest(manifest_path)
    uid_to_shard = manifest.get("uid_to_shard", {})
    shards = {p.name for p in cache_dir.glob("shard-*.zip")}
    uid_to_shard = manifest.get("uid_to_shard", {})
    missing_shards = []
    for shard_name in sorted(set(uid_to_shard.values())):
        try:
            resolve_pose_shard(cache_dir, shard_name)
        except FileNotFoundError:
            missing_shards.append(shard_name)
    if manifest.get("version") != T.POSE_CACHE_VERSION:
        raise SystemExit(
            f"Unsupported pose cache version {manifest.get('version')!r} "
            f"(expected {T.POSE_CACHE_VERSION})"
        )
    if manifest.get("complete") is not True or manifest.get("n") != EXPECTED_CACHE_N:
        raise SystemExit(
            f"Manifest incomplete/mis-sized: complete={manifest.get('complete')!r}, "
            f"n={manifest.get('n')!r} (expected complete=True, n={EXPECTED_CACHE_N})"
        )
    if missing_shards:
        raise SystemExit(
            f"Manifest references missing shard files under {cache_dir}: {missing_shards[:5]}"
        )

    tok_model_path = tok_dir / "signsage_en_sp.model"
    sp = spm.SentencePieceProcessor(model_file=str(tok_model_path))
    fp = T._sp_fingerprint(sp, tok_model_path)
    if fp != expected_tokenizer_fingerprint:
        raise SystemExit(
            f"Tokenizer fingerprint mismatch: {fp} != expected {expected_tokenizer_fingerprint}"
        )
    if (sp.pad_id(), sp.unk_id(), sp.bos_id(), sp.eos_id()) != (0, 1, 2, 3):
        raise SystemExit(
            f"Unexpected tokenizer special ids: "
            f"{(sp.pad_id(), sp.unk_id(), sp.bos_id(), sp.eos_id())} != (0, 1, 2, 3)"
        )

    return {
        "manifest": manifest,
        "csv_path": csv_path,
        "cache_dir": cache_dir,
        "tok_dir": tok_dir,
        "auto_subset": root / "s2t_small6" / "auto_subset.json",
        "feasibility": root / "s2t_small6" / "feasibility.json",
    }


ap = argparse.ArgumentParser(description="Bounded Kaggle diagnostic run")
ap.add_argument("--input-root", default="/kaggle/input")
ap.add_argument("--out", default="/kaggle/working/s2t_diag1")
ap.add_argument("--timeout-seconds", type=int, default=1440)
ap.add_argument("--expected-tokenizer-fingerprint", default="2d88a5de48318cb9")
ap.add_argument("--resume", action="store_true",
                help="ignored by the diagnostic runner (fresh run only); "
                     "kept for CLI parity with train_sign_to_text.py")
ap.add_argument("--embed-init", choices=("scaled", "legacy"), default="scaled",
                help="decoder token-embedding init (scaled default); "
                     "kept for CLI parity with train_sign_to_text.py")
ap.add_argument("--stages", default=None,
                help="ignored by the diagnostic runner (runs overfit,train); "
                     "kept for CLI parity with train_sign_to_text.py")


def main(argv=None):
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"Refusing to overwrite existing diagnostic output: {out}")
    out.mkdir(parents=True)
    (out / "tokenizer").mkdir()

    # Discover the attached Version 4 input root (Kaggle mounts datasets under
    # /kaggle/input/<slug>/; never assume a fixed staging directory).
    input_root = discover_input_root(args.input_root)
    _paths = validate_input(input_root, args.expected_tokenizer_fingerprint)
    manifest = _paths["manifest"]
    csv_path = _paths["csv_path"]
    cache_dir = _paths["cache_dir"]
    tok_dir = _paths["tok_dir"]

    # Validate archive identity (read-only; no payload extraction).
    cleaned, cleaning = T.clean_dataframe(pd.read_csv(csv_path))
    assert not cleaning["duplicate_uid"], "duplicate UID in CSV"
    tr, va, te, split_meta = T.video_split(cleaned, 42)
    parts = T.apply_subset(tr, va, te, 2500, 42)
    expected_uids = {str(uid) for frame in parts for uid in frame["uid"]}
    assert manifest.get("version") == T.POSE_CACHE_VERSION
    assert manifest.get("complete") is True and manifest.get("n") == 7500
    assert set(manifest.get("uid_to_shard", {}).keys()) == expected_uids
    assert T.check_cache_identity(manifest["uid_to_shard"], expected_uids) == ([], [])

    # Copy tokenizer artifacts to fresh out (read-only source preserved).
    for fname in ("signsage_en_sp.model", "signsage_en_sp.vocab",
                  "tokens.npz", "tokens_meta.json", "train_text.txt"):
        src = tok_dir / fname
        if src.is_file():
            shutil.copy2(src, out / "tokenizer" / fname)
    # Copy subset/scope metadata for provenance (not a new feasibility measurement).
    shutil.copy2(_paths["auto_subset"], out / "auto_subset.json")
    shutil.copy2(_paths["feasibility"], out / "source_feasibility_v4.json")

    # Initial preflight uses the first training UID; main switches to full cache.
    probe_uid = str(parts[0]["uid"].iloc[0])
    shard_name = manifest["uid_to_shard"][probe_uid]
    shard_path = resolve_pose_shard(cache_dir, shard_name)
    assert shard_path.exists(), shard_path

    # Verify the specific probe UID is readable through the actual storage
    # representation (ZIP member or directory file), using the same code
    # path as the training loader.
    if shard_path.is_file():
        # ZIP-backed: open archive and check member
        import zipfile
        with zipfile.ZipFile(str(shard_path), mode="r", allowZip64=True) as zf:
            candidates = [
                f"iSign-poses_v1.1/{probe_uid}.pose",
                f"{probe_uid}.pose",
                f"poses/{probe_uid}.pose",
            ]
            if not any(c in zf.namelist() for c in candidates):
                raise SystemExit(
                    f"Probe UID {probe_uid} not found in ZIP archive {shard_path}"
                )
    else:
        # Directory-backed: check file exists in extracted directory
        pose_path = shard_path / f"{probe_uid}.pose"
        if not pose_path.is_file():
            raise SystemExit(
                f"Probe UID {probe_uid} not found in extracted shard {shard_path} "
                f"(expected {pose_path})"
            )

    # Fresh diagnostic command (no resume; bounded; fresh out; no evaluation/export).
    # `run_args` is built once and used for both the flag-parity check and the
    # subprocess command, so there is no off-by-one slice between them.
    run_args = [
        "--csv", str(csv_path),
        "--pose-source", str(shard_path),
        "--pose-cache", str(cache_dir),
        "--out", str(out),
        "--data-mode", "cache",
        "--stages", "overfit,train",
        "--subset", "2500",
        "--seed", "42",
        "--epochs", "8",
        "--batch", "32",
        "--lr", "1e-4",
        "--patience", "2",
        "--decoder-input-dropout", "0.2",
        "--max-hours", "0.4",
        "--embed-init", "scaled",
    ]
    cmd = [sys.executable, "-u", str(T.__file__)] + run_args
    args_parsed = T.parse_args(run_args)
    assert not args_parsed.resume
    assert args_parsed.embed_init == "scaled"

    log_path = out / "subprocess_stdout_stderr.log"
    with log_path.open("w") as log:
        try:
            result = subprocess.run(
                cmd,
                cwd=str(Path(__file__).resolve().parents[3]),
                stdout=log, stderr=subprocess.STDOUT,
                timeout=args.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            print("Exit: TIMEOUT | log:", log_path)
            raise SystemExit(
                f"Diagnostic subprocess exceeded {args.timeout_seconds}s and was killed. "
                f"Preserve outputs under {out}; do not retry automatically."
            ) from exc
    print("Exit:", result.returncode, "| log:", log_path)
    if result.returncode != 0:
        raise SystemExit(
            "Diagnostic failed. Preserve outputs; do not retry automatically. "
            "Exit 124 typically means timeout; 137 can mean forced termination."
        )
    history = json.loads((out / "training_history.json").read_text())
    print("Completed epochs:", len(history))
    print("Run finished; inspect learning/diversity metrics before declaring success.")
    if len(history) < 2:
        raise SystemExit("INCONCLUSIVE: fewer than two completed epochs; not a successful stability result.")


if __name__ == "__main__":
    main()
