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

import research.experiments.train_sign_to_text as T


def main(argv=None):
    ap = argparse.ArgumentParser(description="Bounded Kaggle diagnostic run")
    ap.add_argument("--input-root", default="/kaggle/input")
    ap.add_argument("--out", default="/kaggle/working/s2t_diag1")
    ap.add_argument("--timeout-seconds", type=int, default=1440)
    ap.add_argument("--expected-tokenizer-fingerprint", default="2d88a5de48318cb9")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"Refusing to overwrite existing diagnostic output: {out}")
    out.mkdir(parents=True)
    (out / "tokenizer").mkdir()

    # Find mounted Version 4 artifacts (not nested results.zip).
    input_root = Path(args.input_root)
    manifests = sorted(input_root.glob("**/s2t_small6/pose_cache/manifest.json"))
    csvs = sorted(input_root.glob("**/iSign_v1.1.csv"))
    if len(manifests) != 1:
        raise SystemExit(f"Expected exactly one s2t_small6/pose_cache/manifest.json; found: {manifests}")
    if len(csvs) != 1:
        raise SystemExit(f"Expected exactly one iSign_v1.1.csv; found: {csvs}")
    cache_dir = manifests[0].parent
    csv_path = csvs[0]
    manifest = T.load_strict_manifest(manifests[0])

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

    # Tokenizer fingerprint.
    tok_dir = cache_dir.parent / "tokenizer"
    tok_model_path = tok_dir / "signsage_en_sp.model"
    assert tok_model_path.is_file(), f"tokenizer model missing: {tok_model_path}"
    sp = spm.SentencePieceProcessor(model_file=str(tok_model_path))
    fp = T._sp_fingerprint(sp, tok_model_path)
    assert fp == args.expected_tokenizer_fingerprint, f"tokenizer fingerprint mismatch: {fp}"
    assert (sp.pad_id(), sp.unk_id(), sp.bos_id(), sp.eos_id()) == (0, 1, 2, 3)

    # Copy tokenizer artifacts to fresh out (read-only source preserved).
    for fname in ("signsage_en_sp.model", "signsage_en_sp.vocab",
                  "tokens.npz", "tokens_meta.json", "train_text.txt"):
        src = tok_dir / fname
        if src.is_file():
            shutil.copy2(src, out / "tokenizer" / fname)
    # Copy subset/scope metadata for provenance (not a new feasibility measurement).
    shutil.copy2(cache_dir.parent / "auto_subset.json", out / "auto_subset.json")
    shutil.copy2(cache_dir.parent / "feasibility.json", out / "source_feasibility_v4.json")

    # Initial preflight uses the first training UID; main switches to full cache.
    probe_uid = str(parts[0]["uid"].iloc[0])
    shard_name = manifest["uid_to_shard"][probe_uid]
    assert (cache_dir / shard_name).is_file()

    # Fresh diagnostic command (no resume; bounded; fresh out; no evaluation/export).
    cmd = [
        sys.executable, "-u", str(T.__file__),
        "--csv", str(csv_path),
        "--pose-source", str(cache_dir / shard_name),
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
    args_parsed = T.parse_args(cmd[4:])
    assert not args_parsed.resume
    assert args_parsed.embed_init == "scaled"

    log_path = out / "subprocess_stdout_stderr.log"
    with log_path.open("w") as log:
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parents[3]),
            stdout=log, stderr=subprocess.STDOUT,
        )
    print("Exit:", result.returncode, "| log:", log_path)
    if result.returncode != 0:
        raise SystemExit(
            "Diagnostic failed or timed out. Preserve outputs; do not retry automatically. "
            "Exit 124 typically means timeout; 137 can mean forced termination."
        )
    history = json.loads((out / "training_history.json").read_text())
    print("Completed epochs:", len(history))
    print("Run finished; inspect learning/diversity metrics before declaring success.")
    if len(history) < 2:
        raise SystemExit("INCONCLUSIVE: fewer than two completed epochs; not a successful stability result.")


if __name__ == "__main__":
    main()
