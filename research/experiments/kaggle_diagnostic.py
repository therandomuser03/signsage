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
--stages tokenizer,overfit,train (the tokenizer stage creates the model file
before the overfit gate), applies a bounded external timeout, and writes
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
    """Resolve a manifest shard to either its ZIP or extracted directory.

    Checks:
    1. Direct match: cache_dir / shard_name (file or directory)
    2. If shard_name ends with .zip: cache_dir / shard_name[:-4] as directory or file
    3. If shard_name does not end with .zip: cache_dir / f"{shard_name}.zip" as file
    Blocks path traversal attempts.
    """
    p = cache_dir / shard_name
    try:
        p.resolve().relative_to(cache_dir.resolve())
    except ValueError:
        raise FileNotFoundError(
            f"Manifest shard {shard_name!r} attempts directory traversal outside {cache_dir}"
        )
    if p.is_file() or p.is_dir():
        return p
    if shard_name.endswith(".zip"):
        extracted = cache_dir / shard_name[:-4]
        if extracted.is_dir() or extracted.is_file():
            return extracted
    else:
        archive = cache_dir / f"{shard_name}.zip"
        if archive.is_file() or archive.is_dir():
            return archive
    raise FileNotFoundError(
        f"Manifest shard {shard_name!r} has neither an archive "
        f"nor a matching extracted directory under {cache_dir}"
    )


def discover_input_root(input_root: str | Path) -> Path:
    """Find the unique attached input root holding the Version 4 assets.

    The directory depth and Kaggle slug are not fixed. The manifest and CSV
    must resolve under the same dataset root; separate inputs are never paired.
    """
    root = Path(input_root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Input root is not a directory: {root}")

    candidates: list[Path] = []
    checked: list[str] = []
    manifests = sorted(root.glob("**/pose_cache/manifest.json"))

    for manifest_path in manifests:
        cache_dir = manifest_path.parent
        if cache_dir.parent.name == "s2t_small6":
            candidate = cache_dir.parent.parent
            csv_paths = (candidate / CSV_REL, candidate / "iSign_v1.1.csv")
        else:
            candidate = cache_dir.parent
            csv_paths = (candidate / "isign" / "iSign_v1.1.csv",
                         candidate / "iSign_v1.1.csv")
        checked.extend(str(path) for path in csv_paths)
        if any(path.is_file() for path in csv_paths):
            candidate = candidate.resolve()
            if candidate not in candidates:
                candidates.append(candidate)

    if not candidates:
        csvs = sorted(root.glob("**/iSign_v1.1.csv"))
        raise SystemExit(
            "No input root contains both "
            f"{MANIFEST_REL} and {CSV_REL} under the same root. "
            f"Search root: {root}. Manifest paths found: {[str(p) for p in manifests]}. "
            f"CSV paths found: {[str(p) for p in csvs]}. "
            f"CSV paths checked beside each manifest: {checked}. "
            "Attach the matching dataset version with both artifacts under one root."
        )
    if len(candidates) > 1:
        resolved = [str(c) for c in candidates]
        raise SystemExit(
            f"Ambiguous input roots contain the required data: {resolved}. "
            f"More than one root holds both {MANIFEST_REL} and {CSV_REL}. "
            f"Pass --input-root explicitly or ensure only one dataset is mounted."
        )
    return candidates[0]


def validate_input(root: Path, expected_tokenizer_fingerprint: str) -> dict:
    """Validate all required inputs under root (read-only; never copies shards).

    Performs strict, independent identity checks:
    - Manifest version/completion/count must match expected values.
    - Expected UIDs are derived independently from the CSV + split, never
      from the manifest keys themselves.
    - Every manifest shard must resolve to an actual archive or directory.
    - Tokenizer fingerprint must match the pinned expectation.

    Raises SystemExit with a precise message identifying which artifact is
    missing, mismatched, or corrupt.
    """
    root = Path(root).resolve()
    manifest_path = root / MANIFEST_REL
    csv_path = root / CSV_REL
    cache_dir = root / "s2t_small6" / "pose_cache"
    tok_dir = root / "s2t_small6" / "tokenizer"

    if not manifest_path.is_file() and (root / "pose_cache" / "manifest.json").is_file():
        manifest_path = root / "pose_cache" / "manifest.json"
        cache_dir = root / "pose_cache"
        tok_dir = root / "tokenizer"
    if not csv_path.is_file() and (root / "iSign_v1.1.csv").is_file():
        csv_path = root / "iSign_v1.1.csv"

    auto_subset_path = root / "s2t_small6" / "auto_subset.json"
    if not auto_subset_path.is_file() and (root / "auto_subset.json").is_file():
        auto_subset_path = root / "auto_subset.json"

    feasibility_path = root / "s2t_small6" / "feasibility.json"
    if not feasibility_path.is_file() and (root / "feasibility.json").is_file():
        feasibility_path = root / "feasibility.json"

    required_small = {
        "manifest.json": manifest_path,
        "iSign_v1.1.csv": csv_path,
        "auto_subset.json": auto_subset_path,
        "feasibility.json": feasibility_path,
        "signsage_en_sp.model": tok_dir / "signsage_en_sp.model",
        "signsage_en_sp.vocab": tok_dir / "signsage_en_sp.vocab",
        "tokens.npz": tok_dir / "tokens.npz",
        "tokens_meta.json": tok_dir / "tokens_meta.json",
    }
    missing = [name for name, p in required_small.items() if not p.is_file()]
    if missing:
        # Show the exact paths checked so the user can verify the mount layout.
        checked = {name: str(p) for name, p in required_small.items()}
        raise SystemExit(
            f"Required inputs missing under {root}: filenames={missing}. "
            f"Checked paths: {checked}. Ensure the dataset mount contains both "
            f"{MANIFEST_REL} and {CSV_REL} under the same root."
        )

    # Load manifest with duplicate-key rejection (prevents silent overwrite).
    manifest = T.load_strict_manifest(manifest_path)

    try:
        subset_meta = json.loads(auto_subset_path.read_text())
        feasibility = json.loads(feasibility_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid subset or feasibility metadata under {root}: {exc}") from exc
    if not isinstance(subset_meta, dict) or not isinstance(feasibility, dict):
        raise SystemExit(f"Subset and feasibility metadata must be JSON objects under {root}.")
    try:
        subset_size = int(subset_meta["subset"])
        subset_seed = int(subset_meta.get("subset_seed", 42))
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid subset/seed in {auto_subset_path}: {subset_meta}") from exc
    if subset_size <= 0:
        raise SystemExit(f"Invalid pilot subset size {subset_size} in {auto_subset_path}.")
    if feasibility.get("verdict") != "PASS":
        raise SystemExit(
            f"Feasibility gate is not PASS in {feasibility_path}: "
            f"verdict={feasibility.get('verdict')!r}."
        )

    # Derive expected identities from the authoritative CSV and training split,
    # never from manifest keys.
    cleaned, cleaning = T.clean_dataframe(pd.read_csv(csv_path))
    if cleaning["duplicate_uid"]:
        raise SystemExit(
            f"CSV at {csv_path} contains duplicate UIDs: cannot derive independent identity."
        )
    tr_all, va_all, te_all, split_meta = T.video_split(cleaned, subset_seed)
    tr_sub, va_sub, te_sub = T.apply_subset(
        tr_all, va_all, te_all, subset_size, subset_seed)
    split_for_identity = {"train": tr_sub, "val": va_sub, "test": te_sub}

    expected_uids = {str(uid) for split_name in split_for_identity
                     for uid in split_for_identity[split_name]["uid"]}

    # Strict manifest version check.
    manifest_version = manifest.get("version")
    if manifest_version != T.POSE_CACHE_VERSION:
        raise SystemExit(
            f"Unsupported pose cache version: manifest={manifest_version!r} "
            f"at {manifest_path}; expected {T.POSE_CACHE_VERSION!r}. "
            f"The cache was likely built with a different version of the pipeline."
        )

    # Strict manifest completion, shape, and count checks.
    complete = manifest.get("complete")
    if complete is not True:
        raise SystemExit(
            f"Manifest incomplete: complete={complete!r} at {manifest_path}. "
            f"A partial manifest cannot be trusted for identity validation."
        )

    uid_to_shard = manifest.get("uid_to_shard")
    if not isinstance(uid_to_shard, dict) or any(
        not isinstance(uid, str) or not isinstance(shard, str)
        for uid, shard in uid_to_shard.items()
    ):
        raise SystemExit(
            f"Invalid uid_to_shard mapping in manifest {manifest_path}; "
            "expected string UID keys and shard-name values."
        )
    manifest_uids = set(str(k) for k in uid_to_shard.keys())
    if type(manifest.get("n")) is not int or manifest["n"] != len(manifest_uids):
        raise SystemExit(
            f"Manifest count mismatch at {manifest_path}: "
            f"claims n={manifest.get('n')!r}, uid_to_shard has {len(manifest_uids)} entries."
        )

    # Independent identity validation: compare manifest UIDs to the CSV-derived
    # expected set. Both missing and extra UIDs are reported precisely.
    missing_uids = sorted(expected_uids - manifest_uids)
    extra_uids = sorted(manifest_uids - expected_uids)

    # If the user explicitly expects a full dataset manifest (e.g., full mode),
    # we enforce exact equality. If a subset is configured, we allow a subset
    # match (manifest UIDs must be exactly the subset, no extras, no missing).
    # In either case, the identity is derived independently from the CSV/subset,
    # never from comparing the manifest to itself.
    identity_ok = (missing_uids == [] and extra_uids == [])
    if not identity_ok:
        # Show only the first few mismatches to keep the error concise.
        show_missing = missing_uids[:10]
        show_extra = extra_uids[:10]
        raise SystemExit(
            f"Independent UID identity validation failed at {manifest_path}: "
            f"expected={len(expected_uids):,} (derived from CSV/subset at {csv_path}), "
            f"manifest={len(manifest_uids):,}. "
            f"Missing in manifest (first 10): {show_missing}; "
            f"Extra in manifest (first 10): {show_extra}. "
            f"The manifest does not match the authoritative dataset/subset identity."
        )

    # Check each manifest shard resolves to a real archive or directory.
    # This catches cases where the manifest references shards that are missing,
    # corrupt, or stored as directories when a ZIP was expected (or vice versa).
    missing_shards = []
    resolved_shards_info = []
    for shard_name in sorted(set(uid_to_shard.values())):
        try:
            resolved = resolve_pose_shard(cache_dir, shard_name)
            shard_uids = [uid for uid, shard in uid_to_shard.items() if shard == shard_name]
            if resolved.is_file():
                import zipfile
                if not zipfile.is_zipfile(str(resolved)):
                    raise SystemExit(
                        f"Shard archive {resolved} referenced by manifest is corrupt (not a valid ZIP)."
                    )
                try:
                    with zipfile.ZipFile(str(resolved), mode="r", allowZip64=True) as zf:
                        names = set(zf.namelist())
                        if not names:
                            raise SystemExit(f"Shard archive {resolved} contains no members.")
                        missing_members = [
                            uid for uid in shard_uids
                            if not any(name in names for name in (
                                f"iSign-poses_v1.1/{uid}.pose",
                                f"{uid}.pose",
                                f"poses/{uid}.pose",
                            ))
                        ]
                        if missing_members:
                            raise SystemExit(
                                f"Shard {resolved} is missing pose members mapped by "
                                f"{manifest_path}: {missing_members[:10]}."
                            )
                        # Full CRC check on small/synthetic test archives; sample member check on large
                        if resolved.stat().st_size < 10_000_000:
                            corrupt = zf.testzip()
                            if corrupt is not None:
                                raise SystemExit(
                                    f"Shard archive {resolved} referenced by manifest is corrupt: corrupt member {corrupt}."
                                )
                        else:
                            with zf.open(next(iter(names))) as probe_fh:
                                probe_fh.read(1024)
                except zipfile.BadZipFile:
                    raise SystemExit(
                        f"Shard archive {resolved} referenced by manifest is corrupt (BadZipFile). "
                        f"The archive must be rebuilt or replaced."
                    )
            elif resolved.is_dir():
                missing_members = []
                for uid in shard_uids:
                    member = (resolved / f"{uid}.pose").resolve()
                    try:
                        member.relative_to(resolved.resolve())
                    except ValueError:
                        raise SystemExit(
                            f"Manifest UID {uid!r} escapes extracted shard {resolved}: {member}."
                        )
                    if not member.is_file():
                        missing_members.append(uid)
                if missing_members:
                    raise SystemExit(
                        f"Extracted shard {resolved} is missing pose files mapped by "
                        f"{manifest_path}: {missing_members[:10]}."
                    )
            resolved_shards_info.append({
                "name": shard_name,
                "path": str(resolved),
                "type": "zip" if resolved.is_file() else "directory",
                "exists": resolved.exists(),
            })
        except FileNotFoundError:
            missing_shards.append(shard_name)
            resolved_shards_info.append({
                "name": shard_name,
                "path": str(cache_dir / shard_name),
                "exists": False,
            })

    if missing_shards:
        checked_paths = {s: str(cache_dir / s) for s in missing_shards}
        raise SystemExit(
            f"Manifest references missing shard artifacts: {missing_shards[:10]}. "
            f"Checked under {cache_dir}: {checked_paths}. "
            f"Each manifest shard must exist as either a .zip archive or an extracted directory."
        )

    # Tokenizer fingerprint check (must match the expected value exactly).
    tok_model_path = tok_dir / "signsage_en_sp.model"
    if not tok_model_path.is_file():
        raise SystemExit(
            f"Tokenizer model missing: expected {tok_model_path}. "
            f"The tokenizer artifacts must be present for identity verification."
        )
    sp = spm.SentencePieceProcessor(model_file=str(tok_model_path))
    fp = T._sp_fingerprint(sp, tok_model_path)
    if fp != expected_tokenizer_fingerprint:
        raise SystemExit(
            f"Tokenizer fingerprint mismatch at {tok_model_path}: "
            f"calculated={fp!r} != expected={expected_tokenizer_fingerprint!r}. "
            f"The tokenizer was likely rebuilt with different data or parameters."
        )
    if (sp.pad_id(), sp.unk_id(), sp.bos_id(), sp.eos_id()) != (0, 1, 2, 3):
        raise SystemExit(
            f"Unexpected tokenizer special token IDs: "
            f"(pad, unk, bos, eos) = {(sp.pad_id(), sp.unk_id(), sp.bos_id(), sp.eos_id())} "
            f"; expected (0, 1, 2, 3). Tokenizer does not match the expected contract."
        )

    token_splits = {
        name: (frame["uid"].to_numpy(), frame["text"].to_numpy())
        for name, frame in split_for_identity.items()
    }
    expected_token_meta = {
        "splits": {
            name: {
                "sp": fp,
                "max_len": T.MAX_TOKENS,
                "n": len(uids),
                "uids": T._uids_fingerprint(uids),
                "texts": T._texts_fingerprint(texts),
            }
            for name, (uids, texts) in token_splits.items()
        }
    }
    try:
        token_meta = json.loads((tok_dir / "tokens_meta.json").read_text())
        if token_meta != expected_token_meta:
            raise SystemExit(
                f"Token cache identity mismatch at {tok_dir / 'tokens_meta.json'}: "
                "tokenizer/split fingerprints do not match the CSV-derived training split."
            )
        with np.load(tok_dir / "tokens.npz", allow_pickle=False) as token_cache:
            expected_keys = {
                f"{name}_{suffix}"
                for name in token_splits for suffix in ("ids", "len")
            }
            if set(token_cache.files) != expected_keys:
                raise ValueError(f"keys={sorted(token_cache.files)} expected={sorted(expected_keys)}")
            for name, (uids, _) in token_splits.items():
                ids = token_cache[f"{name}_ids"]
                lengths = token_cache[f"{name}_len"]
                if (ids.ndim != 2 or lengths.shape != (len(uids),)
                        or len(ids) != len(uids)
                        or np.any(lengths < 2) or np.any(lengths > ids.shape[1])
                        or np.any(ids < 0) or np.any(ids >= sp.get_piece_size())):
                    raise ValueError(f"invalid {name} token array shapes or IDs")
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"Invalid token cache at {tok_dir}: {exc}") from exc

    # Verify the probe UID (derived from CSV/subset) resolves correctly through
    # the discovered shard path (ZIP member or directory file). This ensures
    # the diagnostic's --pose-source argument points at a real readable artifact.
    probe_uid = str(split_for_identity.get("train", tr_all)["uid"].iloc[0])
    if probe_uid not in manifest_uids:
        raise SystemExit(
            f"Probe UID {probe_uid} from CSV/subset is not present in manifest. "
            f"The manifest identity does not cover the selected dataset sample."
        )
    shard_name = manifest.get("uid_to_shard", {}).get(probe_uid)
    if shard_name is None:
        raise SystemExit(
            f"Probe UID {probe_uid} has no shard mapping in manifest. "
            f"Every UID in the manifest must map to exactly one shard."
        )
    shard_path = resolve_pose_shard(cache_dir, shard_name)
    if not shard_path.exists():
        raise SystemExit(
            f"Resolved probe shard missing for UID {probe_uid}: "
            f"shard={shard_name}, resolved_path={shard_path}, expected in {cache_dir}."
        )
    # Check the specific member/file exists inside the resolved shard.
    if shard_path.is_file():
        import zipfile
        try:
            with zipfile.ZipFile(str(shard_path), mode="r", allowZip64=True) as zf:
                members = [
                    f"iSign-poses_v1.1/{probe_uid}.pose",
                    f"{probe_uid}.pose",
                    f"poses/{probe_uid}.pose",
                ]
                if not any(c in zf.namelist() for c in members):
                    found_members = [m for m in zf.namelist() if m.endswith(".pose")][:5]
                    raise SystemExit(
                        f"Probe UID {probe_uid} not found in ZIP archive {shard_path}. "
                        f"Searched member names: {members}. First 5 ZIP members: {found_members}. "
                        f"The archive contents do not match the expected dataset format."
                    )
        except zipfile.BadZipFile:
            raise SystemExit(
                f"Shard archive {shard_path} referenced by manifest is corrupt (BadZipFile). "
                f"The archive must be rebuilt or replaced."
            )
    else:
        pose_path = shard_path / f"{probe_uid}.pose"
        if not pose_path.is_file():
            # Check for path-escape attempts: the resolved file must be inside
            # the shard directory and must not escape via traversal.
            try:
                resolved_member = pose_path.resolve()
                resolved_shard = shard_path.resolve()
                # Ensure the resolved member is actually inside the shard root.
                resolved_member.relative_to(resolved_shard)
            except ValueError:
                raise SystemExit(
                    f"Probe UID {probe_uid} attempts directory escape: "
                    f"resolved member {resolved_member} is outside shard directory {resolved_shard}. "
                    f"Rejecting potentially malicious path traversal."
                )
            found_poses = sorted([p.name for p in shard_path.glob("*.pose")])[:5]
            raise SystemExit(
                f"Probe UID {probe_uid} not found in extracted shard {shard_path} "
                f"(expected {pose_path}). First 5 files in directory: {found_poses}."
            )

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "csv_path": csv_path,
        "cache_dir": cache_dir,
        "tok_dir": tok_dir,
        "auto_subset": auto_subset_path,
        "feasibility": feasibility_path,
        "expected_uids": sorted(expected_uids),
        "manifest_uids": sorted(manifest_uids),
        "independent_identity_ok": identity_ok,
        "subset_size": subset_size,
        "subset_seed": subset_seed,
        "resolved_shards": resolved_shards_info,
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

    # Validate all inputs before creating output, so a failed preflight is
    # safe to rerun without leaving a misleading partial output directory.
    input_root = discover_input_root(args.input_root)
    _paths = validate_input(input_root, args.expected_tokenizer_fingerprint)
    manifest = _paths["manifest"]
    csv_path = _paths["csv_path"]
    cache_dir = _paths["cache_dir"]
    tok_dir = _paths["tok_dir"]
    out.mkdir(parents=True)
    (out / "tokenizer").mkdir()

    # The validate_input already performed independent identity checks, strict
    # manifest validation, tokenizer fingerprint verification, and shard
    # resolution. We keep the identity assertions here as a defensive
    # cross-check using the same CSV-derived expected_uids.
    cleaned, cleaning = T.clean_dataframe(pd.read_csv(csv_path))
    assert not cleaning["duplicate_uid"], "duplicate UID in CSV (independent identity broken)"
    tr, va, te, split_meta = T.video_split(cleaned, _paths["subset_seed"])
    parts = T.apply_subset(tr, va, te, _paths["subset_size"], _paths["subset_seed"])
    expected_uids = {str(uid) for frame in parts for uid in frame["uid"]}

    # Independent identity verification (manifest vs CSV/subset, not self-comparison).
    assert _paths.get("independent_identity_ok") is True, (
        f"Independent UID identity failed: expected={len(_paths.get('expected_uids', set()))}, "
        f"manifest={len(_paths.get('manifest_uids', set()))}"
    )
    assert manifest.get("version") == T.POSE_CACHE_VERSION, (
        f"Manifest version mismatch: {manifest.get('version')!r} != {T.POSE_CACHE_VERSION!r}"
    )
    assert manifest.get("complete") is True, (
        f"Manifest incomplete: complete={manifest.get('complete')!r}"
    )
    # Self-consistency: manifest count claim must match its own uid_to_shard length.
    assert manifest.get("n") == len(manifest.get("uid_to_shard", {})), (
        f"Manifest count mismatch: claims n={manifest.get('n')!r} but has "
        f"{len(manifest.get('uid_to_shard', {}))} UIDs"
    )
    # Cross-check identity using the training module's identity helper.
    identity_result = T.check_cache_identity(manifest.get("uid_to_shard", {}), expected_uids)
    assert identity_result == ([], []), (
        f"Cache identity mismatch from independent CSV derivation: missing={identity_result[0]}, extra={identity_result[1]}"
    )

    # Copy tokenizer artifacts to fresh out (read-only source preserved).
    for fname in ("signsage_en_sp.model", "signsage_en_sp.vocab",
                  "tokens.npz", "tokens_meta.json", "train_text.txt"):
        src = tok_dir / fname
        if src.is_file():
            shutil.copy2(src, out / "tokenizer" / fname)
    # Copy subset/scope metadata for provenance (not a new feasibility measurement).
    shutil.copy2(_paths["auto_subset"], out / "auto_subset.json")
    shutil.copy2(_paths["feasibility"], out / "source_feasibility_v4.json")

    # Exercise the training loader on one real sample per referenced shard.
    # This proves each manifest route, rather than merely checking shard names.
    pose_source = T.ShardPoseSource(str(cache_dir))
    probe_uids = []
    probed_shards = set()
    for uid, shard_name in manifest["uid_to_shard"].items():
        if shard_name not in probed_shards:
            probe_uids.append(uid)
            probed_shards.add(shard_name)
    for probe_uid in probe_uids:
        frames, frame_count = T.load_pose_item(probe_uid, pose_source)
        if frames.shape != (T.MAX_FRAMES, T.LANDMARK_DIM) or frames.dtype != np.float32:
            raise SystemExit(
                f"Pose decode failed for UID {probe_uid}: "
                f"shape={frames.shape}, dtype={frames.dtype}."
            )
        if not 0 < frame_count <= T.MAX_FRAMES:
            raise SystemExit(f"Invalid frame count for UID {probe_uid}: {frame_count}.")
    print(f"Decoded {len(probe_uids)} samples across {len(_paths['resolved_shards'])} shards.")

    # Fresh diagnostic command (no resume; bounded; fresh out; no evaluation/export).
    # `run_args` is built once and used for both the flag-parity check and the
    # subprocess command, so there is no off-by-one slice between them.
    run_args = [
        "--csv", str(csv_path),
        "--pose-source", str(cache_dir),
        "--pose-cache", str(cache_dir),
        "--out", str(out),
        "--data-mode", "cache",
        "--stages", "tokenizer,overfit,train",
        "--subset", str(_paths["subset_size"]),
        "--seed", str(_paths["subset_seed"]),
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
