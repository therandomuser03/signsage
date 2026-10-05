#!/usr/bin/env python3
"""T5-small pose-to-text training: gates + staged short experiment.

Modes:
  --synthetic-gate   CPU wiring probe on structured synthetic poses (no dataset)
  --real-gate        BINDING gate: first 64 TRAIN rows from materialized shards;
                     Stage A (frozen T5) then Stage B (joint), PASS only on
                     >=95% exact match via greedy translate(). FAIL = hard STOP.
  --shards DIR       5k experiment: staged fine-tune with val loss, best-keep,
                     patience stopping, optimizer/scheduler resume, AMP on CUDA.

Usage (local CPU wiring probe, no dataset):
    python train.py --synthetic-gate --out ./t5_poc_gate
    (committed record: research/evaluation/t5_poc_gate/gate_report.json)

Usage (Kaggle GPU host, after materialize.py):
    python train.py --real-gate --shards ./t5_poc_shards --out ./t5_poc_gate64
    python train.py --shards ./t5_poc_shards --out ./t5_poc_5k --epochs 8
    python train.py --shards ./t5_poc_shards --out ./t5_poc_5k --resume  # after interrupt

Checkpoint records: T5 checkpoint id, projection config, tokenizer id,
normalization version, split manifest, seed, metrics, git revision.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import T5Tokenizer

from model import MAX_FRAMES, MAX_TOKENS, POSE_FEATURE_DIM, T5_CHECKPOINT, PoseT5

SEED = 42
REAL_GATE_N = 64
REAL_GATE_EM = 0.95


def log_memory(tag: str) -> None:
    """One-line host+GPU memory diagnostic (never spams per batch)."""
    try:
        import resource
        rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        # Linux ru_maxrss is KiB; macOS is bytes. Heuristic normalize:
        if sys.platform == "darwin":
            rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
        rss = f"rss_max={rss_gb:.2f}G"
    except Exception:  # noqa: BLE001
        try:
            with open("/proc/self/status") as fh:
                vm = next(l for l in fh if l.startswith("VmRSS"))
            rss = f"rss={int(vm.split()[1]) / 1e6:.2f}G"
        except Exception:  # noqa: BLE001
            rss = "rss=?"
    if torch.cuda.is_available():
        try:
            al = torch.cuda.memory_allocated() / 1e9
            rs = torch.cuda.memory_reserved() / 1e9
            gpu = f" cuda_alloc={al:.2f}G cuda_res={rs:.2f}G"
        except Exception:  # noqa: BLE001
            gpu = ""
    else:
        gpu = ""
    print(f"[mem:{tag}] {rss}{gpu}", flush=True)


def tokenize_batch(tokenizer, texts: list[str]) -> torch.Tensor:
    enc = tokenizer(texts, padding="max_length", max_length=MAX_TOKENS,
                    truncation=True, return_tensors="pt")
    labels = enc.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return labels


class ShardDataset(Dataset):
    """Loads materialize.py .npz shards (poses fp16 -> fp32 on the fly).

    Each shard member is loaded ONCE per file (np.load returns a fresh
    full array on every __getitem__ call, so per-row z["poses"][i] access
    would decompress the whole shard N times and pin N full-size bases).
    Rows are copied out so the per-shard base can be released immediately.
    """

    def __init__(self, shard_paths: list[str]):
        self.rows = []
        for sp in shard_paths:
            with np.load(sp, allow_pickle=True) as z:
                poses = z["poses"]
                lengths = z["lengths"]
                texts = z["texts"]
                uids = z["uids"]
                for i in range(len(uids)):
                    self.rows.append((np.array(poses[i], copy=True),
                                      int(lengths[i]),
                                      str(texts[i]), str(uids[i])))
            del poses, lengths, texts, uids
            gc.collect()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, length, text, uid = self.rows[i]
        return torch.from_numpy(p.astype(np.float32)), length, text, uid


class GateDataset(Dataset):
    """Tiny in-memory 64-row gate set (fp16 rows -> fp32 on the fly)."""

    def __init__(self, rows: list):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, length, text, uid = self.rows[i]
        return torch.from_numpy(p.astype(np.float32)), length, text, uid


def build_gate_dataset(shard_dir: Path, n: int) -> GateDataset:
    """Collect only the first n TRAIN rows across sorted train_*.npz shards.

    Reads shard-by-shard, copies just the needed rows, and releases each
    shard's arrays before opening the next one. Never touches val/test
    shards and never holds more than one shard + n small rows at once.
    """
    shard_paths = sorted(shard_dir.glob("train_*.npz"))
    if not shard_paths:
        raise SystemExit(f"no train_*.npz shards in {shard_dir}")
    rows: list = []
    shards_touched = 0
    for sp in shard_paths:
        if len(rows) >= n:
            break
        with np.load(str(sp), allow_pickle=True) as z:
            poses = z["poses"]
            lengths = z["lengths"]
            texts = z["texts"]
            uids = z["uids"]
            take = min(n - len(rows), len(uids))
            for i in range(take):
                rows.append((np.array(poses[i], copy=True),
                             int(lengths[i]),
                             str(texts[i]), str(uids[i])))
        shards_touched += 1
        del poses, lengths, texts, uids
        gc.collect()
    if len(rows) < n:
        raise SystemExit(f"only {len(rows)} train rows, need {n}")
    print(f"gate dataset: {len(rows)} rows from {shards_touched} shard(s) "
          f"(of {len(shard_paths)} train shards)", flush=True)
    return GateDataset(rows)


class SyntheticGateDataset(Dataset):
    """Structured synthetic poses: per-sample offset+phase so inputs are
    mutually distinct and pose-conditioned (wiring/memorization probe only)."""

    TEXTS = ["hello world", "good morning everyone", "thank you very much",
             "how are you today", "see you tomorrow morning", "i love you",
             "have a nice day", "what is your name"]

    def __init__(self, n: int = 32, frames: int = MAX_FRAMES, seed: int = SEED):
        self.n, self.frames = n, frames
        rng = np.random.default_rng(seed)
        self.base = rng.normal(size=(POSE_FEATURE_DIM,)).astype(np.float32)
        self.texts = [self.TEXTS[i % len(self.TEXTS)] + f" number {i}" for i in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        t = np.arange(self.frames, dtype=np.float32)[:, None]
        phase = float(i + 1) / self.n
        pose = (self.base[None, :] * (0.5 + phase)
                + 0.3 * np.sin(2 * np.pi * phase * t / self.frames
                               + np.linspace(0, 3, POSE_FEATURE_DIM)[None, :]))
        return torch.from_numpy(pose.astype(np.float32)), self.frames, self.texts[i], f"synth-{i}"


def collate_pose_text(batch, tokenizer):
    poses = torch.stack([b[0] for b in batch])
    lengths = torch.tensor([b[1] for b in batch], dtype=torch.long)
    labels = tokenize_batch(tokenizer, [b[2] for b in batch])
    uids = [b[3] for b in batch]
    return poses, lengths, labels, uids


def pose_mask(lengths: torch.Tensor, max_len: int, device) -> torch.Tensor:
    return (torch.arange(max_len, device=device).unsqueeze(0)
            < lengths.to(device).unsqueeze(1))


def autocast_ctx(device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def make_scaler(device):
    try:
        return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    except TypeError:  # older torch
        return torch.cuda.amp.GradScaler(enabled=device.type == "cuda")


def step_batch(model, poses, lengths, labels, device, opt, scaler) -> float:
    poses, labels = poses.to(device), labels.to(device)
    mask = pose_mask(lengths, poses.size(1), device)
    opt.zero_grad(set_to_none=True)
    with autocast_ctx(device):
        loss = model(poses, mask, labels).loss
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite training loss.")
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    return float(loss.detach())


def run_stage(model: PoseT5, loader, tokenizer, device, steps: int, lr: float,
              log_every: int = 25) -> list[float]:
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=1e-4)
    scaler = make_scaler(device)
    model.train()
    losses, step = [], 0
    while step < steps:
        for poses, lengths, labels, _ in loader:
            losses.append(step_batch(model, poses, lengths, labels, device, opt, scaler))
            step += 1
            if step % log_every == 0:
                print(f"  step {step}/{steps} loss={losses[-1]:.4f}", flush=True)
            if step >= steps:
                break
    return losses


@torch.no_grad()
def eval_teacher_loss(model: PoseT5, loader, device) -> float:
    """Mean teacher-forced cross-entropy (fast val signal; ignore_index=-100)."""
    model.eval()
    crit = nn.CrossEntropyLoss(ignore_index=-100)
    total, n = 0.0, 0
    for poses, lengths, labels, _ in loader:
        poses, labels = poses.to(device), labels.to(device)
        mask = pose_mask(lengths, poses.size(1), device)
        with autocast_ctx(device):
            logits = model(poses, mask, labels).logits
        total += float(crit(logits.reshape(-1, logits.size(-1)),
                            labels.reshape(-1)).detach())
        n += 1
    return total / max(1, n)


@torch.no_grad()
def exact_match(model: PoseT5, loader, tokenizer, device) -> tuple[float, list]:
    model.eval()
    refs, hyps = [], []
    for poses, lengths, labels, _ in loader:
        poses = poses.to(device)
        mask = pose_mask(lengths, poses.size(1), device)
        preds = model.translate(poses, mask, tokenizer)
        ref_ids = labels.clone()
        ref_ids[ref_ids == -100] = tokenizer.pad_token_id
        refs += tokenizer.batch_decode(ref_ids, skip_special_tokens=True)
        hyps += preds
    em = float(np.mean([r.strip() == h.strip() for r, h in zip(refs, hyps)]))
    return em, list(zip(refs, hyps))[:8]


def build_schedule(opt, total_steps: int, warmup_frac: float = 0.06):
    warmup = max(1, int(warmup_frac * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def ckpt_payload(model, opt=None, scheduler=None, extra: dict | None = None) -> dict:
    payload = {
        "state_dict": model.state_dict(),
        "t5_checkpoint": T5_CHECKPOINT,
        "projection": {"in_dim": POSE_FEATURE_DIM, "d_model": 512,
                       "arch": "Linear-LN-GELU-Linear-LN"},
        "tokenizer": T5_CHECKPOINT,
        "normalization_version": "pose_normalize_v1",
        "max_frames": MAX_FRAMES, "max_tokens": MAX_TOKENS,
        "run_at": datetime.now(timezone.utc).isoformat(),
    }
    if opt is not None:
        payload["optimizer_state"] = opt.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if extra:
        payload.update(extra)
    return payload


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=Path(__file__).parents[2],
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def run_gate(ds, loader, tokenizer, device, model, out: Path, mode: str,
             n: int, seed: int, a_steps: int, b_steps: int,
             lr_a: float, lr_b: float, t0: float,
             em_threshold: float) -> int:
    log_memory("gate_start")
    print(f"STAGE A: frozen T5, projection only ({a_steps} steps)", flush=True)
    model.set_t5_frozen(True)
    hist_a = run_stage(model, loader, tokenizer, device, a_steps, lr_a)
    log_memory("gate_stage_a_done")
    em_a, _ = exact_match(model, loader, tokenizer, device)
    print(f"STAGE A: loss {hist_a[0]:.4f}->{hist_a[-1]:.4f} exact_match={em_a:.1%}", flush=True)
    print(f"STAGE B: joint fine-tune ({b_steps} steps)", flush=True)
    model.set_t5_frozen(False)
    hist_b = run_stage(model, loader, tokenizer, device, b_steps, lr_b)
    em_b, ex_b = exact_match(model, loader, tokenizer, device)
    log_memory("gate_stage_b_done")
    print(f"STAGE B: loss {hist_b[0]:.4f}->{hist_b[-1]:.4f} exact_match={em_b:.1%}", flush=True)
    for r, h in ex_b:
        print(f"  REF={r!r} HYP={h!r}", flush=True)
    passed = em_b >= em_threshold
    report = {"mode": mode, "gate_n": n, "seed": seed,
              "stage_a": {"loss_first": hist_a[0], "loss_last": hist_a[-1],
                          "exact_match": em_a},
              "stage_b": {"loss_first": hist_b[0], "loss_last": hist_b[-1],
                          "exact_match": em_b},
              "threshold": em_threshold, "passed": passed,
              "git_rev": git_rev(), "wall_clock_s": time.time() - t0}
    (out / "gate_report.json").write_text(json.dumps(report, indent=1))
    if passed:
        print(f"GATE PASSED (EM {em_b:.1%} >= {em_threshold:.0%})", flush=True)
        return 0
    print(f"GATE FAILED (EM {em_b:.1%} < {em_threshold:.0%}) -- STOP. "
          f"Diagnose before any 5k experiment.", flush=True)
    return 1


def train_5k(model, train_loader, val_loader, device, out: Path, args,
             tokenizer, t0: float) -> int:
    scaler = make_scaler(device)
    last_path = out / "last.pt"
    best_path = out / "best.pt"
    history: list = []
    start_stage, start_epoch = "A", 1
    best, bad = float("inf"), 0
    opt, scheduler = None, None

    def new_opt(lr):
        o = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=lr, weight_decay=1e-4)
        steps = max(1, len(train_loader)) * args.epochs
        return o, build_schedule(o, steps)

    if args.resume and last_path.exists():
        rec = torch.load(last_path, map_location=device)
        model.load_state_dict(rec["state_dict"])
        start_stage = rec.get("stage", "A")
        start_epoch = int(rec.get("epoch", 0)) + 1
        best = float(rec.get("best", best))
        history = rec.get("history", [])
        print(f"resumed at stage {start_stage} epoch {start_epoch} "
              f"(best val {best:.4f})", flush=True)

    for stage, lr, frozen in (("A", args.lr_a, True), ("B", args.lr_b, False)):
        if stage < start_stage:
            continue
        model.set_t5_frozen(frozen)
        if opt is None:  # fresh or stage boundary: rebuild for new param set
            opt, scheduler = new_opt(lr)
        if args.resume and last_path.exists() and stage == start_stage and opt is not None:
            try:
                rec = torch.load(last_path, map_location=device)
                if rec.get("stage") == stage and "optimizer_state" in rec:
                    opt.load_state_dict(rec["optimizer_state"])
                    if rec.get("scheduler_state"):
                        scheduler.load_state_dict(rec["scheduler_state"])
            except Exception as exc:  # noqa: BLE001 - resume best-effort, restart stage
                print(f"resume state unloadable ({exc}); restarting stage {stage}", flush=True)
                opt, scheduler = new_opt(lr)
        bad = 0
        for epoch in range(start_epoch if stage == start_stage else 1, args.epochs + 1):
            model.train()
            tr_loss, n = 0.0, 0
            for poses, lengths, labels, _ in train_loader:
                tr_loss += step_batch(model, poses, lengths, labels, device, opt, scaler)
                scheduler.step()
                n += 1
            tr_loss /= max(1, n)
            va_loss = eval_teacher_loss(model, val_loader, device)
            history.append({"stage": stage, "epoch": epoch,
                            "train_loss": tr_loss, "val_loss": va_loss})
            print(f"stage {stage} epoch {epoch}/{args.epochs}: "
                  f"train={tr_loss:.4f} val={va_loss:.4f}", flush=True)
            torch.save(ckpt_payload(model, opt, scheduler, {
                "stage": stage, "epoch": epoch, "best": best,
                "history": history, "seed": args.seed, "git_rev": git_rev()}), last_path)
            if va_loss < best - 1e-4:
                best, bad = va_loss, 0
                torch.save(ckpt_payload(model, opt, scheduler, {
                    "stage": stage, "epoch": epoch, "best": best,
                    "history": history, "seed": args.seed,
                    "git_rev": git_rev()}), best_path)
                print(f"  new best (val {best:.4f})", flush=True)
            else:
                bad += 1
                if bad >= args.patience:
                    print("  early stopping: val loss stopped improving.", flush=True)
                    break
        start_epoch = 1  # subsequent stages always start at epoch 1
        opt, scheduler = None, None  # force rebuild for next stage's param set
    if not best_path.exists():
        raise SystemExit("no best checkpoint produced.")
    (out / "train_report.json").write_text(json.dumps(
        {"seed": args.seed, "git_rev": git_rev(), "best_val": best,
         "history": history, "wall_clock_s": time.time() - t0}, indent=1))
    print(f"best checkpoint: {best_path} (val {best:.4f})", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="T5-small pose-to-text training")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", default=None, help="materialize.py output dir")
    ap.add_argument("--synthetic-gate", action="store_true")
    ap.add_argument("--real-gate", action="store_true",
                    help="binding 64-sample gate on TRAIN shards (hard STOP on fail)")
    ap.add_argument("--gate-n", type=int, default=REAL_GATE_N)
    ap.add_argument("--gate-threshold", type=float, default=REAL_GATE_EM)
    ap.add_argument("--stage-a-steps", type=int, default=150)
    ap.add_argument("--stage-b-steps", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr-a", type=float, default=3e-4)
    ap.add_argument("--lr-b", type=float, default=1e-4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log_memory("startup")
    if os.environ.get("HF_TOKEN"):
        print("HF_TOKEN configured: YES", flush=True)
    else:
        print("HF_TOKEN configured: NO (using cached weights if available)", flush=True)
    tokenizer = T5Tokenizer.from_pretrained(T5_CHECKPOINT)
    model = PoseT5().to(device)
    print(f"params={model.trainable_summary()} "
          f"amp={'on' if device.type == 'cuda' else 'off (CPU fallback)'}", flush=True)
    log_memory("model")
    collate = partial(collate_pose_text, tokenizer=tokenizer)

    if args.synthetic_gate:
        ds = SyntheticGateDataset(n=args.gate_n)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
        return run_gate(ds, loader, tokenizer, device, model, out, "synthetic-gate",
                        args.gate_n, args.seed, args.stage_a_steps, args.stage_b_steps,
                        args.lr_a, args.lr_b, t0, em_threshold=0.5)

    if args.real_gate:
        if not args.shards:
            raise SystemExit("--real-gate needs --shards <materialize.py dir>")
        shard_dir = Path(args.shards)
        log_memory("shard_discovery")
        gate_ds = build_gate_dataset(shard_dir, args.gate_n)
        log_memory("gate_dataset")
        # Gate safety: single-process loading, no worker duplication of data,
        # no persistent workers, no aggressive prefetch, no pinned copies.
        loader = DataLoader(gate_ds, batch_size=args.batch, shuffle=True,
                            collate_fn=collate, num_workers=0,
                            pin_memory=False)
        print(f"REAL GATE: {args.gate_n} materialized train rows "
              f"(hard STOP below {args.gate_threshold:.0%} EM)", flush=True)
        try:
            return run_gate(gate_ds, loader, tokenizer, device, model, out, "real-gate",
                            args.gate_n, args.seed, args.stage_a_steps, args.stage_b_steps,
                            args.lr_a, args.lr_b, t0, em_threshold=args.gate_threshold)
        finally:
            del loader, gate_ds
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_memory("gate_cleanup")

    if not args.shards:
        raise SystemExit("need --shards <materialize.py dir>, --synthetic-gate, or --real-gate")
    shard_dir = Path(args.shards)
    log_memory("shard_discovery")
    train_ds = ShardDataset(sorted(map(str, shard_dir.glob("train_*.npz"))))
    val_ds = ShardDataset(sorted(map(str, shard_dir.glob("val_*.npz"))))
    log_memory("full_dataset")
    if not len(train_ds) or not len(val_ds):
        raise SystemExit("empty train or val shards -- refusing.")
    workers = 2 if device.type == "cuda" else 0
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate, num_workers=workers,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            collate_fn=collate, num_workers=workers,
                            pin_memory=device.type == "cuda")
    print(f"5k run: train={len(train_ds)} val={len(val_ds)}", flush=True)
    return train_5k(model, train_loader, val_loader, device, out, args, tokenizer, t0)


if __name__ == "__main__":
    sys.exit(main())
