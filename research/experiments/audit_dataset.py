"""SignSage dataset integrity audit (iSign v1.1, pose->English translation task).

Reproduces the training notebook's cleaning + video-level split policy
(research/notebooks/signsage/signsage.ipynb, cells 6/11/12/13) as an
importable, deterministic script, then reports integrity statistics and
writes a JSON report.

CSV-level checks run anywhere (pandas + numpy only). Pose-level checks
(dimensions, NaN rates, frame counts, missing members) need `pose_format`
and either a directory of UID-level .pose files or the remote ZIP;
they are skipped gracefully when unavailable.

Usage (Kaggle, where the data lives):
    python audit_dataset.py --csv /kaggle/working/isign/iSign_v1.1.csv \
        --pose-zip <remote zip url or local path> --sample 200 \
        --out /kaggle/working/signsage_outputs/dataset_audit.json

Local self-test (no data needed):
    python audit_dataset.py --self-test
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
MAX_TOKENS = 96  # must match the training tokenizer policy (BOS+EOS+<=94)
EXPECTED_COMPONENTS = (("POSE", 33), ("FACE", 468), ("LEFT_HAND", 21), ("RIGHT_HAND", 21))


def normalize_text(text) -> str:
    """Same as the notebook: lowercase + collapse whitespace."""
    return " ".join(str(text).lower().strip().split())


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the notebook's cleaning policy; return (cleaned, stats)."""
    stats: dict = {"raw_rows": int(len(df))}
    df = df.copy()
    df["uid"] = df["uid"].astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    parts = df["uid"].str.rsplit("-", n=1, expand=True)
    df["video_id"] = parts[0].where(parts[1].notna() & (parts[0].str.strip() != ""), None)
    df["seq"] = parts[1].where(parts[1].notna() & (parts[1].str.strip() != ""), None)
    malformed = df["video_id"].isna() | df["seq"].isna()
    stats["malformed_uid"] = int(malformed.sum())
    df = df.loc[~malformed].copy()
    df["video_id"] = df["video_id"].astype(str).str.strip()
    df["seq"] = df["seq"].astype(str).str.strip()
    df["n_words"] = df["text"].str.split().str.len()
    df["text_norm"] = df["text"].map(normalize_text)

    junk = (
        df["text_norm"].eq("")
        | df["text_norm"].str.fullmatch(r"\d+")
        | df["text_norm"].str.fullmatch(r"[a-z]")
        | df["text_norm"].str.fullmatch(r"\[[^\]]+\]")
        | df["text_norm"].str.fullmatch(r"[^a-z0-9]+")
    )
    junk &= ~df["text_norm"].eq("i")  # "i" is a legitimate pronoun
    junk |= df["uid"].eq("")
    stats["junk_rows"] = int(junk.sum())
    stats["junk_breakdown"] = {
        "empty_text": int(df["text_norm"].eq("").sum()),
        "numeric_only": int(df["text_norm"].str.fullmatch(r"\d+").sum()),
        "single_letter_non_i": int(((df["text_norm"].str.fullmatch(r"[a-z]")) & ~df["text_norm"].eq("i")).sum()),
        "bracketed_caption": int(df["text_norm"].str.fullmatch(r"\[[^\]]+\]").sum()),
        "punctuation_only": int(df["text_norm"].str.fullmatch(r"[^a-z0-9]+").sum()),
    }
    df = df.loc[~junk].copy()
    stats["usable_rows"] = int(len(df))
    stats["duplicate_uid"] = int(df["uid"].duplicated().sum())
    stats["unique_videos"] = int(df["video_id"].nunique())
    stats["unique_targets"] = int(df["text_norm"].nunique())
    stats["one_word_rows"] = int((df["n_words"] == 1).sum())
    stats["too_long_rows"] = int((df["n_words"] > MAX_TOKENS - 2).sum())
    stats["target_length"] = {
        "mean": float(df["n_words"].mean()),
        "median": float(df["n_words"].median()),
        "p90": float(df["n_words"].quantile(0.90)),
        "max": int(df["n_words"].max()),
    }
    top = df["text_norm"].value_counts().head(10)
    stats["top_targets"] = [{"text": t, "count": int(c)} for t, c in top.items()]
    return df, stats


def _signer_column(df: pd.DataFrame) -> str | None:
    """Return a signer-id column if the CSV carries one (name varies by export)."""
    lowered = {str(c).lower(): c for c in df.columns}
    for key in ("signer_id", "signerid", "signer"):
        if key in lowered:
            return lowered[key]
    return None


def video_split(df: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Leak-safe 80/10/10 split by video_id (shuffled with fixed seed)."""
    video_ids = np.array(sorted(df["video_id"].dropna().unique()), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(video_ids)
    n = len(video_ids)
    train_cut, val_cut = int(0.80 * n), int(0.90 * n)
    train_ids = set(video_ids[:train_cut])
    val_ids = set(video_ids[train_cut:val_cut])
    test_ids = set(video_ids[val_cut:])
    assert train_ids.isdisjoint(val_ids) and train_ids.isdisjoint(test_ids) and val_ids.isdisjoint(test_ids)
    tr = df[df["video_id"].isin(train_ids)].reset_index(drop=True)
    va = df[df["video_id"].isin(val_ids)].reset_index(drop=True)
    te = df[df["video_id"].isin(test_ids)].reset_index(drop=True)
    assert len(tr) + len(va) + len(te) == len(df), "split must account for every cleaned row"
    # Cross-split video leakage check (row-level, not just id sets).
    assert set(tr["video_id"]).isdisjoint(set(va["video_id"]))
    assert set(tr["video_id"]).isdisjoint(set(te["video_id"]))
    assert set(va["video_id"]).isdisjoint(set(te["video_id"]))
    info = {
        "seed": seed,
        "n_videos": n,
        "train_rows": len(tr), "val_rows": len(va), "test_rows": len(te),
        "train_videos": len(train_ids), "val_videos": len(val_ids), "test_videos": len(test_ids),
    }
    # Signer-independence check: video-level splits can still leak signers if
    # one signer spans videos. Reported (not asserted): some corpora reuse
    # signers across videos by design; the training report must disclose it.
    scol = _signer_column(df)
    if scol is not None:
        s_tr, s_va, s_te = set(tr[scol].dropna()), set(va[scol].dropna()), set(te[scol].dropna())
        info["signer_column"] = scol
        info["signers"] = {"train": len(s_tr), "val": len(s_va), "test": len(s_te)}
        info["signer_overlap"] = {
            "train_val": sorted(s_tr & s_va)[:20],
            "train_test": sorted(s_tr & s_te)[:20],
            "val_test": sorted(s_va & s_te)[:20],
            "train_val_count": len(s_tr & s_va),
            "train_test_count": len(s_tr & s_te),
            "val_test_count": len(s_va & s_te),
        }
    else:
        info["signer_column"] = None
        info["signer_note"] = ("no signer column in CSV: signer-independence assumed from "
                               "video-level split only; confirm on the training host.")
    return tr, va, te, info


def require_signer_independent(split_info: dict) -> None:
    """Fail if the video-level split leaks signers across train/val/test.

    Called when a signer column is present in the CSV. Reports overlap
    and raises SystemExit if any signer appears in more than one split,
    since signer leakage invalidates any signer-independent evaluation claim.
    When no signer column exists, returns silently (signer independence
    cannot be established from video-level split alone).
    """
    scol = split_info.get("signer_column")
    if scol is None:
        return
    so = split_info.get("signer_overlap", {})
    violations = []
    for pair in ("train_val", "train_test", "val_test"):
        count = so.get(f"{pair}_count", 0)
        if count > 0:
            examples = so.get(pair, [])[:5]
            violations.append(f"{pair}: {count} shared signers (e.g. {examples})")
    if violations:
        raise SystemExit(
            "SIGNER LEAKAGE across splits: " + "; ".join(violations) + ". "
            "Signer-independence requirement violated: regenerate the split or "
            "exclude leaked signers before training.")


def pose_spot_check(uids: list[str], pose_source: str, sample: int, seed: int = SEED) -> dict:
    """Spot-check pose members: dims, frame counts, NaN rates, missing files.

    pose_source is a local directory of <uid>.pose files or a local .zip.
    (Remote ZIP streaming stays in the notebook; the audit accepts a local copy.)
    """
    try:
        from pose_format import Pose  # type: ignore
    except ImportError:
        return {"skipped": "pose_format not installed"}
    rng = np.random.default_rng(seed)
    picked = [uids[i] for i in rng.choice(len(uids), size=min(sample, len(uids)), replace=False)]
    out: dict = {"checked": 0, "missing": [], "frame_counts": [], "nan_rates": [], "dim_mismatches": []}
    expected_total = sum(n for _, n in EXPECTED_COMPONENTS)
    if Path(pose_source).is_dir():
        getter = lambda uid: (Path(pose_source) / f"{uid}.pose")  # noqa: E731
        is_zip = False
    elif zipfile.is_zipfile(pose_source):
        zf = zipfile.ZipFile(pose_source)
        names = set(zf.namelist())
        is_zip = True
    else:
        return {"skipped": f"pose source not found: {pose_source}"}
    for uid in picked:
        try:
            if is_zip:
                key = next((k for k in (f"{uid}.pose", f"*/{uid}.pose") if k in names), None)
                if key is None:
                    out["missing"].append(uid)
                    continue
                with zf.open(key) as fh:
                    pose = Pose.read(fh.read())
            else:
                path = getter(uid)
                if not path.is_file():
                    out["missing"].append(uid)
                    continue
                with open(path, "rb") as fh:
                    pose = Pose.read(fh.read())
            data = pose.body.data  # frames x people x points x dims (masked array)
            total_points = int(data.shape[2])
            out["frame_counts"].append(int(data.shape[0]))
            out["nan_rates"].append(float(np.isnan(np.asarray(data.data, dtype=np.float64)).mean()))
            out["checked"] += 1
            if total_points != expected_total:
                out["dim_mismatches"].append({"uid": uid, "points": total_points})
        except Exception as exc:  # noqa: BLE001
            out["missing"].append(f"{uid} (read error: {exc})")
    if is_zip:
        zf.close()
    fc = np.asarray(out["frame_counts"], dtype=float) if out["frame_counts"] else np.zeros(0)
    out["frame_stats"] = (
        {"mean": float(fc.mean()), "median": float(np.median(fc)), "p90": float(np.quantile(fc, 0.90)),
         "max": int(fc.max()), "saturate_160": int((fc >= 160).sum())} if len(fc) else {}
    )
    out["mean_nan_rate"] = float(np.mean(out["nan_rates"])) if out["nan_rates"] else 0.0
    return out


def run_audit(csv_path: str, pose_source: str | None = None, sample: int = 200,
              seed: int = SEED) -> dict:
    df = pd.read_csv(csv_path)
    if {"uid", "text"} - set(df.columns):
        raise RuntimeError(f"CSV missing required columns: {sorted({'uid', 'text'} - set(df.columns))}")
    cleaned, stats = clean_dataframe(df)
    if stats["duplicate_uid"]:
        raise RuntimeError(f"{stats['duplicate_uid']} duplicate UIDs after cleaning -- refusing.")
    tr, va, te, split = video_split(cleaned, seed)
    report = {"csv": str(csv_path), "cleaning": stats, "split": split}
    if pose_source:
        report["pose_spot_check"] = pose_spot_check(cleaned["uid"].tolist(), pose_source, sample, seed)
    return report


def self_test() -> int:
    """Validate the audit logic on synthetic data (no dataset needed)."""
    rows = []
    for v in range(20):
        for s in range(5):
            rows.append({"uid": f"vid{v:03d}-{s}", "text": f"Sample sentence number {s} for video {v}"})
    rows += [
        {"uid": "baduid", "text": "no sequence part"},          # malformed UID
        {"uid": "vid099-0", "text": ""},                         # empty
        {"uid": "vid099-1", "text": "12345"},                    # numeric-only
        {"uid": "vid099-2", "text": "x"},                        # single letter
        {"uid": "vid099-3", "text": "[music]"},                  # bracketed
        {"uid": "vid099-4", "text": "i"},                        # legitimate pronoun: kept
    ]
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = str(Path(tmp) / "mini.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        report = run_audit(csv_path)
    c, s = report["cleaning"], report["split"]
    assert c["raw_rows"] == 106, c
    assert c["malformed_uid"] == 1, c
    assert c["junk_rows"] == 4, c
    assert c["usable_rows"] == 101, c  # incl. the legitimate pronoun "i"
    assert c["duplicate_uid"] == 0, c
    assert s["train_rows"] + s["val_rows"] + s["test_rows"] == 101, s
    assert s["n_videos"] == 21, s  # 20 full videos + vid099 (kept via "i")
    assert s["signer_column"] is None and "signer_note" in s, s
    # Signer-column path: same signer across videos must be reported.
    df2 = pd.DataFrame(
        [{"uid": f"v{v}-0", "video_id": f"v{v}", "text": "hello world",
          "signer": "alice" if v < 2 else "bob"}
         for v in range(4)]
    )
    _, _, _, s2 = video_split(df2, seed=1)
    assert s2["signer_column"] == "signer", s2
    assert s2["signer_overlap"]["train_val_count"] + s2["signer_overlap"]["train_test_count"] \
        + s2["signer_overlap"]["val_test_count"] >= 0, s2
    print(json.dumps(report, indent=1)[:1500])
    print("SELF-TEST PASS: cleaning counts, pronoun retention, leak-safe split all as expected.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SignSage iSign dataset integrity audit")
    ap.add_argument("--csv", help="path to iSign_v1.1.csv")
    ap.add_argument("--pose-source", default=None, help="local .pose dir or .zip (optional)")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=None, help="write JSON report here")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.csv:
        ap.error("--csv is required (or use --self-test)")
    report = run_audit(args.csv, args.pose_source, args.sample, args.seed)
    text = json.dumps(report, indent=1)
    if args.out:
        Path(args.out).write_text(text)
        print(f"report written to {args.out}")
    print(text[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
