#!/usr/bin/env python3
"""Evaluate a trained PoseT5 checkpoint on held-out shards (RESEARCH ONLY).

Reports: WER, BLEU-1..4, ROUGE-L, exact match, non-empty rate, prediction
diversity, mean lengths, trivial baseline, 30 qualitative examples,
language-prior/collapse stats, and input-sensitivity tests
(pose swap / zero input / temporal shuffle -> POSE-CONDITIONING check).

Usage:
    python evaluate.py --checkpoint ./t5_poc_5k/t5_pose_projection.pt \\
        --shards ./t5_poc_shards --out ./t5_poc_eval
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer

from model import T5_CHECKPOINT, PoseT5
from train import ShardDataset, collate_pose_text, pose_mask

from functools import partial


@torch.no_grad()
def decode_all(model: PoseT5, loader, tokenizer, device) -> tuple[list, list, list]:
    refs, hyps, uids = [], [], []
    model.eval()
    for poses, lengths, labels, batch_uids in loader:
        poses = poses.to(device)
        mask = pose_mask(lengths, poses.size(1), device)
        preds = model.translate(poses, mask, tokenizer)
        ref_ids = labels.clone()
        ref_ids[ref_ids == -100] = tokenizer.pad_token_id
        refs += tokenizer.batch_decode(ref_ids, skip_special_tokens=True)
        hyps += preds
        uids += batch_uids
    return refs, hyps, uids


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "research" / "experiments"))
from evaluation_metrics import (
    compute_metrics,
    diversity_metrics,
    build_trivial_baseline,
    evaluate_gate,
    normalize_text,
)


def sensitivity(model: PoseT5, loader, tokenizer, device, k: int = 4) -> dict:
    """Swap / zero / shuffle poses; outputs SHOULD change if pose-conditioned."""
    model.eval()
    batch = next(iter(loader))
    poses, lengths, _, _ = batch
    poses = poses[:k].to(device)
    mask = pose_mask(lengths[:k], poses.size(1), device)
    base = model.translate(poses, mask, tokenizer)
    swapped = model.translate(poses.flip(0), mask, tokenizer)
    zeroed = model.translate(torch.zeros_like(poses), mask, tokenizer)
    shuffled = model.translate(
        poses[:, torch.randperm(poses.size(1), device=poses.device)], mask, tokenizer)
    same = lambda a, b: float(np.mean([x.strip() == y.strip() for x, y in zip(a, b)]))  # noqa: E731
    return {"base": base, "swapped": swapped, "zeroed": zeroed, "shuffled": shuffled,
            "frac_same_swap": same(base, swapped),
            "frac_same_zero": same(base, zeroed),
            "frac_same_shuffle": same(base, shuffled),
            "pose_conditioned": bool(same(base, swapped) < 1.0
                                     or same(base, zeroed) < 1.0
                                     or same(base, shuffled) < 1.0)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate PoseT5 checkpoint")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--examples", type=int, default=30)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = T5Tokenizer.from_pretrained(T5_CHECKPOINT)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = PoseT5().to(device)
    model.load_state_dict(ckpt["state_dict"])

    shard_dir = Path(args.shards)
    test_ds = ShardDataset(sorted(map(str, shard_dir.glob("test_*.npz"))))
    loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                        collate_fn=partial(collate_pose_text, tokenizer=tokenizer))
    refs, hyps, uids = decode_all(model, loader, tokenizer, device)

    metrics = compute_metrics(refs, hyps)
    div = diversity_metrics(hyps, refs)
    trivial = build_trivial_baseline(refs)
    trivial_metrics = compute_metrics(refs, trivial)
    sens = sensitivity(model, loader, tokenizer, device)
    gate = evaluate_gate(metrics, trivial_metrics, div)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(refs), size=min(args.examples, len(refs)), replace=False)
    examples = [{"uid": uids[i], "ref": refs[i], "hyp": hyps[i]} for i in idx]

    report = {"test_metrics": metrics, "trivial_baseline": trivial_metrics,
              "diversity": div, "sensitivity": sens, "examples": examples,
              "gate": gate,
              "beats_trivial_wer": gate["beats_trivial_wer"],
              "beats_trivial_bleu": gate["beats_trivial_bleu"],
              "beats_trivial_chrf": gate["beats_trivial_chrf"],
              "pose_conditioned": sens["pose_conditioned"],
              "collapsed": div["collapsed"]}
    (out / "eval_report.json").write_text(json.dumps(report, indent=1))

    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("examples", "sensitivity")}, indent=1))
    print(f"\n--- {len(examples)} held-out examples (random) ---")
    for e in examples:
        print(f"UID={e['uid']}\n  REF: {e['ref']}\n  HYP: {e['hyp']}")
    print("\n--- sensitivity ---")
    for i, (b, s, z, f) in enumerate(zip(sens["base"], sens["swapped"],
                                         sens["zeroed"], sens["shuffled"])):
        print(f"[{i}] base={b!r} swap={s!r} zero={z!r} shuffle={f!r}")
    print(f"\npose_conditioned={sens['pose_conditioned']} "
          f"collapsed={report['collapsed']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
