#!/usr/bin/env python3
"""Materialize a SMALL (~5k) local pose subset for the T5 POC (RESEARCH ONLY).

Reuses, in order:
- `clean_dataframe` + `video_split` from research/training/audit_dataset.py
  (official video-level split policy; subset keeps whole videos only)
- `read_normalized_pose` + `resize_sequence` path from
  research/training/train_sign_to_text.py (canonical pose_normalize_v1,
  MAX_FRAMES=160, [160,1629] + true length)

Format: one .npz shard per split chunk (fast, deterministic, inspectable):
    poses   float16 [N,160,1629]   normalized + resized
    lengths int32   [N]            true (unpadded) lengths
    texts   '<U...' [N]            original English translations
    uids    '<U...' [N]
    video_ids '<U...' [N]

Resumability: shards are written atomically (tmp + rename) and the manifest
is updated after EVERY shard. A restart re-validates recorded shards via
sha256, skips rows already materialized, and continues. Relative shard paths
keep the manifest portable across Kaggle re-attaches.

Usage (Kaggle GPU host, HF_TOKEN with iSign access in env):
    python materialize.py --csv /path/iSign_v1.1.csv --pose-source remote \\
        --out ./t5_poc_shards --target-train 4000 --target-val 500 --target-test 500

Local smoke test (no dataset, no token; proves write + manifest + resume):
    python materialize.py --smoke --out /tmp/poc_smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "research" / "training"))

from audit_dataset import clean_dataframe, video_split  # noqa: E402
from train_sign_to_text import (  # noqa: E402
    MAX_FRAMES,
    _open_pose_store,
    load_pose_item,
    preflight_pose_source,
)
from app.ml.preprocessing.sequence import resize_sequence  # noqa: E402

POSE_DIM = 1629
DEFAULT_SHARD_SIZE = 1000
DEFAULT_MAX_SKIP_RATE = 0.05


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_npz(path: Path, **arrays) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def take_video_subset(df: pd.DataFrame, target_rows: int, seed: int) -> pd.DataFrame:
    """Whole videos only, shuffled with fixed seed, until ~target_rows."""
    vids = df["video_id"].dropna().unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(vids)
    picked, count = [], 0
    for v in vids:
        if count >= target_rows:
            break
        picked.append(v)
        count += int((df["video_id"] == v).sum())
    return df[df["video_id"].isin(picked)].reset_index(drop=True)


def synthetic_pose(uid: str, seed: int = 42) -> tuple[np.ndarray, int]:
    """Deterministic synthetic pose: exercises resize/true_len without iSign."""
    rng = np.random.default_rng(abs(hash((uid, seed))) % (2 ** 31))
    frames = 40 + (abs(hash(uid)) % 120)  # varying lengths: 40..159
    raw = rng.normal(size=(frames, POSE_DIM)).astype(np.float32)
    true_len = min(frames, MAX_FRAMES)
    return resize_sequence(raw, MAX_FRAMES).astype(np.float32), true_len


def fetch_real(uid: str, store) -> tuple[np.ndarray, int]:
    frames, true_len = load_pose_item(uid, store)
    if not np.isfinite(frames).all() or true_len <= 0:
        raise ValueError(f"invalid pose for {uid}")
    return frames, true_len


def load_split_manifest(out: Path, name: str) -> dict:
    """Load prior manifest; keep only leading run of valid shards."""
    path = out / f"{name}_manifest.json"
    if not path.is_file():
        return {"split": name, "shards": [], "completed_uids": [],
                "skipped": [], "n": 0}
    try:
        man = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"split": name, "shards": [], "completed_uids": [],
                "skipped": [], "n": 0}
    valid_shards, valid_uids = [], []
    for sh in man.get("shards", []):
        p = out / sh["file"]
        if not p.is_file() or sha256_file(p) != sh.get("sha256"):
            break  # truncate at first invalid; rest are untrusted
        valid_shards.append(sh)
        valid_uids.extend(sh.get("uids", []))
    man["shards"] = valid_shards
    man["completed_uids"] = valid_uids
    man["n"] = sum(s["n"] for s in valid_shards)
    return man


def flush_shard(out: Path, name: str, idx: int, buf: dict, man: dict) -> None:
    poses = np.stack(buf["poses"]).astype(np.float16)
    path = out / f"{name}_{idx:03d}.npz"
    atomic_write_npz(
        path,
        poses=poses,
        lengths=np.asarray(buf["lengths"], dtype=np.int32),
        texts=np.asarray(buf["texts"]),
        uids=np.asarray(buf["uids"]),
        video_ids=np.asarray(buf["video_ids"]),
    )
    man["shards"].append({
        "file": path.name,  # relative: portable across Kaggle re-attaches
        "n": len(buf["uids"]),
        "sha256": sha256_file(path),
        "dtype": "float16",
        "shape": [int(x) for x in poses.shape],
        "uid_first": buf["uids"][0],
        "uid_last": buf["uids"][-1],
        "uids": list(buf["uids"]),
    })
    man["completed_uids"].extend(buf["uids"])
    man["n"] = sum(s["n"] for s in man["shards"])
    atomic_write_json(out / f"{name}_manifest.json", man)
    print(f"[{name}] shard {path.name}: n={len(buf['uids'])}", flush=True)


def materialize_split(df: pd.DataFrame, fetch, out: Path, name: str,
                      seed: int, shard_size: int,
                      max_skip_rate: float) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    man = load_split_manifest(out, name)
    done = set(man["completed_uids"])
    if done:
        print(f"[{name}] resume: {len(done)} rows already materialized, "
              f"{len(man['shards'])} valid shards", flush=True)
    skipped = man.get("skipped", [])
    buf: dict = {k: [] for k in ("poses", "lengths", "texts", "uids", "video_ids")}
    idx = len(man["shards"])
    processed = 0
    for _, row in df.iterrows():
        uid = str(row["uid"])
        if uid in done:
            continue
        text = str(row["text"])
        processed += 1
        if not text.strip():
            skipped.append({"uid": uid, "reason": "empty_text"})
            continue
        try:
            frames, true_len = fetch(uid)
        except Exception as exc:  # noqa: BLE001 - recorded, never silent
            skipped.append({"uid": uid, "reason": f"fetch_failed: {exc}"})
            continue
        buf["poses"].append(frames.astype(np.float16))
        buf["lengths"].append(true_len)
        buf["texts"].append(text)
        buf["uids"].append(uid)
        buf["video_ids"].append(str(row["video_id"]))
        if len(buf["uids"]) >= shard_size:
            flush_shard(out, name, idx, buf, man)
            idx += 1
            buf = {k: [] for k in buf}
    if buf["uids"]:
        flush_shard(out, name, idx, buf, man)
        idx += 1
    man["skipped"] = skipped
    man["seed"] = seed
    man["shard_size"] = shard_size
    total = man["n"] + len(skipped)
    man["skip_rate"] = len(skipped) / total if total else 1.0
    atomic_write_json(out / f"{name}_manifest.json", man)
    print(f"[{name}] kept={man['n']} skipped={len(skipped)} "
          f"skip_rate={man['skip_rate']:.3%} shards={len(man['shards'])}", flush=True)
    if man["n"] == 0:
        raise SystemExit(f"[{name}] FAIL: zero samples materialized.")
    if man["skip_rate"] > max_skip_rate:
        raise SystemExit(f"[{name}] FAIL: skip_rate {man['skip_rate']:.3%} > "
                         f"{max_skip_rate:.3%} -- refusing to claim success.")
    return man


def sanity_print(shard_path: str, k: int = 10, seed: int = 42) -> None:
    """§8 data sanity check: shapes, stats, finiteness, non-empty text."""
    z = np.load(shard_path, allow_pickle=True)
    n = len(z["uids"])
    rng = np.random.default_rng(seed)
    for i in rng.choice(n, size=min(k, n), replace=False):
        p = z["poses"][i].astype(np.float32)
        print(f"uid={z['uids'][i]} video={z['video_ids'][i]} shape={p.shape} "
              f"len={z['lengths'][i]} min={p.min():.3f} max={p.max():.3f} "
              f"mean={p.mean():.4f} std={p.std():.4f} "
              f"naninf={bool((~np.isfinite(p)).any())} text={z['texts'][i]!r}")
        assert p.shape == (MAX_FRAMES, POSE_DIM) and z["lengths"][i] <= MAX_FRAMES
        assert np.isfinite(p).all() and bool(str(z["texts"][i]).strip())


def smoke_dataframe(n_videos: int = 30, per_video: int = 2) -> pd.DataFrame:
    rows = []
    for v in range(n_videos):
        for s in range(per_video):
            rows.append({"uid": f"smoke-v{v:03d}-{s}",
                         "text": f"Smoke test sentence {s} for video {v}",
                         "video_id": f"smoke-v{v:03d}"})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Materialize small T5 POC subset")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--pose-source", default=None,
                    help="local .pose dir, local .zip, or 'remote' (needs HF_TOKEN)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-train", type=int, default=4000)
    ap.add_argument("--target-val", type=int, default=500)
    ap.add_argument("--target-test", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap rows per split (quick real-source test)")
    ap.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    ap.add_argument("--max-skip-rate", type=float, default=DEFAULT_MAX_SKIP_RATE)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="synthetic local run: no CSV, no poses, no token")
    args = ap.parse_args(argv)

    out = Path(args.out)

    if args.smoke:
        print("SMOKE mode: synthetic poses, exercises sharding + manifest + resume.", flush=True)
        cleaned = smoke_dataframe()
        cleaned["video_id"] = cleaned["video_id"].astype(str)
        df_tr, df_va, df_te, split = video_split(cleaned, args.seed)
        fetch = lambda uid: synthetic_pose(uid, args.seed)  # noqa: E731
    else:
        if not args.csv or not args.pose_source:
            raise SystemExit("--csv and --pose-source are required (or use --smoke)")
        df = pd.read_csv(args.csv)
        cleaned, cleaning = clean_dataframe(df)
        if cleaning["duplicate_uid"]:
            raise SystemExit("duplicate UIDs after cleaning -- refusing.")
        df_tr, df_va, df_te, split = video_split(cleaned, args.seed)
        print(f"official video-level split: train={len(df_tr)} val={len(df_va)} "
              f"test={len(df_te)} videos={split['n_videos']}", flush=True)
        kind = "remote" if args.pose_source == "remote" else (
            "zip" if args.pose_source.endswith(".zip") else "dir")
        arg = os.environ.get("HF_TOKEN", "") if kind == "remote" else args.pose_source
        if kind == "remote" and not arg:
            raise SystemExit("remote pose streaming needs HF_TOKEN")
        store = _open_pose_store(kind, arg)
        preflight_pose_source(store, str(df_tr["uid"].iloc[0]))
        fetch = lambda uid: fetch_real(uid, store)  # noqa: E731

    targets = {"train": args.target_train, "val": args.target_val, "test": args.target_test}
    frames = {"train": df_tr, "val": df_va, "test": df_te}
    seeds = {"train": args.seed, "val": args.seed + 1, "test": args.seed + 2}
    report: dict = {"mode": "smoke" if args.smoke else "real",
                    "seed": args.seed, "shard_size": args.shard_size,
                    "subset": {}}
    if not args.smoke:
        report["cleaning"] = cleaning
        report["split"] = split
    for name in ("train", "val", "test"):
        sub = take_video_subset(frames[name], targets[name], seeds[name])
        if args.limit:
            sub = sub.head(args.limit).reset_index(drop=True)
        report["subset"][name] = {
            k: v for k, v in materialize_split(
                sub, fetch, out, name, seeds[name],
                args.shard_size, args.max_skip_rate).items()
            if k != "completed_uids"}
    atomic_write_json(out / "materialize_report.json", report)
    if args.sanity:
        for name in ("train", "val", "test"):
            for shard in report["subset"][name]["shards"][:1]:
                sanity_print(str(out / shard["file"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
