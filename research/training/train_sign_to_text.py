#!/usr/bin/env python3
"""SignSage v2 sign-to-text training — corrected post-352868924 configuration.

What this is: a reproducible, resumable training job for continuous ISL
pose-sequence -> English translation on iSign v1.1. It reuses the backend's
 OWN model, preprocessing, and masks (single canonical implementation), keeps
every training mechanic the 352868924 audit verified as correct
(shift / masks / ignore_index / video-level split / tokenizer procedure),
and changes only what the root-cause analysis justified (see
research/training/READINESS.md): pre-norm, larger capacity, label smoothing,
longer schedule, per-epoch diversity collapse gate, hard overfit gate.

Stages (select with --stages subset, default all):
    audit -> tokenizer -> overfit-gate -> train -> evaluate -> export

    python train_sign_to_text.py --csv /path/iSign_v1.1.csv \\
        --pose-source /path/to/poses.zip --out /kaggle/working/s2t_v2
    python train_sign_to_text.py --smoke --out /tmp/s2t_smoke   # CPU plumbing check

Resumable: recovery snapshots every --snapshot-batches + best kept;
re-run with --resume to continue. Never overwrites run 352868924.
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import math
import os
import random
import shutil
import sys
import time
import warnings
import zipfile
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).parent))

from app.ml.models.sign_to_text import SignTransformer, causal_mask, padding_mask  # noqa: E402
from app.ml.preprocessing.landmarks import (  # noqa: E402
    COORDINATES_PER_LANDMARK,
    TOTAL_LANDMARKS,
)
from app.ml.preprocessing.normalization import normalize_pose_sequence  # noqa: E402
from app.ml.preprocessing.sequence import resize_sequence  # noqa: E402
from audit_dataset import clean_dataframe, normalize_text, video_split  # noqa: E402

SEED = 42
MAX_FRAMES = 160
MAX_TOKENS = 96
LANDMARK_DIM = TOTAL_LANDMARKS * COORDINATES_PER_LANDMARK  # 1629
MODEL_INPUT_DIM = LANDMARK_DIM * 2  # model concatenates frame deltas
FULL_FEATURE_MASK = list(range(LANDMARK_DIM))

# v2 model capacity (root-cause fix: 352868924's d192x3L underfit 100k pairs).
D_MODEL, NHEAD, NLAYERS, FF = 256, 4, 4, 1024
DROPOUT = 0.15
NORM_FIRST = True  # pre-norm: stable from step 0 on short budgets
LABEL_SMOOTHING = 0.1
PEAK_LR = 3e-4
WARMUP_FRACTION = 0.06
MIN_LR_RATIO = 0.1
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# Decoding constraints (identical to the serving contract).
MAX_DECODE_TOKENS = 94
MIN_OUTPUT_TOKENS = 1
NO_REPEAT_NGRAM_SIZE = 3
REPETITION_PENALTY = 1.15

# Collapse gate: unique-fraction on a val decode sample; STOP below this.
DIVERSITY_FLOOR = 0.05
DIVERSITY_SAMPLE = 512


def log(msg: str) -> None:
    print(f"[s2t-v2 {datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Pose sources
# --------------------------------------------------------------------------

class PoseSource:
    def read_member_bytes(self, uid: str) -> bytes:
        raise NotImplementedError


class DirPoseSource(PoseSource):
    def __init__(self, root: str):
        self.root = Path(root)

    def read_member_bytes(self, uid: str) -> bytes:
        path = self.root / f"{uid}.pose"
        if not path.is_file():
            raise FileNotFoundError(f"UID not found in pose dir: {uid}")
        return path.read_bytes()


class ZipPoseSource(PoseSource):
    def __init__(self, path: str):
        self.zf = zipfile.ZipFile(path, mode="r", allowZip64=True)
        self.names = set(self.zf.namelist())

    def read_member_bytes(self, uid: str) -> bytes:
        for cand in (f"iSign-poses_v1.1/{uid}.pose", f"{uid}.pose", f"poses/{uid}.pose"):
            if cand in self.names:
                with self.zf.open(cand, "r") as fh:
                    return fh.read()
        raise FileNotFoundError(f"UID not found in pose archive: {uid}")


class RemoteConcatFile(io.RawIOBase):
    """Seekable view over the 4-part iSign pose ZIP on Hugging Face (ported
    from signsage.ipynb cell 8; streams members via bounded HTTP ranges)."""

    PARTS = [
        "iSign-poses_v1.1_part_aa",
        "iSign-poses_v1.1_part_ab",
        "iSign-poses_v1.1_part_ac",
        "iSign-poses_v1.1_part_ad",
    ]

    def __init__(self, token: str, chunk_mb: int = 32):
        import requests
        from huggingface_hub import hf_hub_url

        super().__init__()
        self._requests = requests
        self.urls = [hf_hub_url("Exploration-Lab/iSign", p, repo_type="dataset") for p in self.PARTS]
        self.sizes = [self._probe(u, token) for u in self.urls]
        self.starts = np.cumsum([0] + self.sizes[:-1]).tolist()
        self.total_size = int(sum(self.sizes))
        self.pos = 0
        self.token = token
        self.chunk_size = chunk_mb * 1024 * 1024
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}", "User-Agent": "SignSage-v2/1.0"})
        from collections import OrderedDict
        self.cache: OrderedDict = OrderedDict()

    def _probe(self, url: str, token: str) -> int:
        r = self._requests.get(url, headers={"Authorization": f"Bearer {token}", "Range": "bytes=0-0"},
                               timeout=(20, 60), allow_redirects=True)
        if r.status_code != 206:
            raise RuntimeError(f"Could not probe remote part size: HTTP {r.status_code}")
        return int(r.headers.get("Content-Range", "0/0").rsplit("/", 1)[-1])

    def readable(self): return True

    def seekable(self): return True

    def tell(self): return self.pos

    def seek(self, offset, whence=os.SEEK_SET):
        self.pos = min(int(offset if whence == os.SEEK_SET else self.pos + offset), self.total_size)
        return self.pos

    def read(self, n=-1):
        if self.pos >= self.total_size:
            return b""
        n = min(int(n if n and n >= 0 else self.total_size - self.pos), self.total_size - self.pos)
        out = bytearray()
        while len(out) < n:
            idx = max(0, min(int(np.searchsorted(self.starts, self.pos, side="right") - 1), len(self.urls) - 1))
            local = self.pos - self.starts[idx]
            take = min(n - len(out), self.sizes[idx] - local)
            key = (idx, (local // self.chunk_size) * self.chunk_size)
            if key not in self.cache:
                cs, ce = key[1], min(self.sizes[idx] - 1, key[1] + self.chunk_size - 1)
                last = None
                for attempt in range(5):
                    try:
                        r = self.session.get(self.urls[idx], headers={"Range": f"bytes={cs}-{ce}"},
                                             timeout=(20, 180), allow_redirects=True)
                        if r.status_code != 206:
                            raise RuntimeError(f"range request failed: HTTP {r.status_code}")
                        self.cache[key] = r.content
                        while len(self.cache) > 8:
                            self.cache.popitem(last=False)
                        break
                    except (self._requests.exceptions.Timeout, self._requests.exceptions.ConnectionError) as exc:
                        last = exc
                        time.sleep(1.5 * 2 ** attempt)
                else:
                    raise ConnectionError(f"range fetch failed after retries: {key}") from last
            out.extend(self.cache[key][local - key[1]:local - key[1] + take])
            self.pos += take
        return bytes(out)


class RemotePoseSource(PoseSource):
    def __init__(self, token: str):
        self.zf = zipfile.ZipFile(RemoteConcatFile(token), mode="r", allowZip64=True)
        self.names = set(self.zf.namelist())

    def read_member_bytes(self, uid: str) -> bytes:
        for cand in (f"iSign-poses_v1.1/{uid}.pose", f"{uid}.pose"):
            if cand in self.names:
                with self.zf.open(cand, "r") as fh:
                    return fh.read()
        raise FileNotFoundError(f"UID not found in remote pose archive: {uid}")


def read_normalized_pose(uid: str, store: PoseSource) -> np.ndarray:
    """Raw .pose -> (frames, 1629) via the canonical pose_normalize_v1 path."""
    from pose_format import Pose

    pose = Pose.read(store.read_member_bytes(uid))
    data = np.asarray(np.ma.filled(pose.body.data, 0.0), dtype=np.float32)
    if data.ndim != 4 or data.shape[1] < 1:
        raise ValueError(f"Unexpected pose tensor {data.shape} for {uid}")
    flat = data[:, 0, :TOTAL_LANDMARKS, :].reshape(data.shape[0], -1).astype(np.float32)
    return normalize_pose_sequence(flat)


def load_pose_item(uid: str, store: PoseSource) -> tuple[np.ndarray, int]:
    raw = read_normalized_pose(uid, store)[:, FULL_FEATURE_MASK]
    true_len = min(len(raw), MAX_FRAMES)
    return resize_sequence(raw, MAX_FRAMES).astype(np.float32), true_len


# --------------------------------------------------------------------------
# Tokenizer (same procedure as 352868924: SP unigram 16k on train text only)
# --------------------------------------------------------------------------

def train_tokenizer(train_texts: list[str], out_dir: Path, vocab_size: int = 16000):
    import sentencepiece as spm

    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = out_dir / "train_text.txt"
    with open(corpus, "w", encoding="utf-8") as fh:
        for text in train_texts:
            fh.write(normalize_text(text).replace("\n", " ") + "\n")
    prefix = out_dir / "signsage_en_sp"
    model_path = prefix.with_suffix(".model")
    if not model_path.exists():
        spm.SentencePieceTrainer.train(
            input=str(corpus), model_prefix=str(prefix), vocab_size=vocab_size,
            model_type="unigram", character_coverage=1.0,
            normalization_rule_name="nmt_nfkc_cf",
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            pad_piece="<pad>", unk_piece="<unk>", bos_piece="<bos>", eos_piece="<eos>",
            hard_vocab_limit=False, shuffle_input_sentence=False, input_sentence_size=0,
            seed_sentencepiece_size=1000000, num_threads=max(1, min(8, os.cpu_count() or 1)),
        )
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    assert (sp.pad_id(), sp.unk_id(), sp.bos_id(), sp.eos_id()) == (0, 1, 2, 3)
    return sp


def encode_text(sp, text: str, max_len: int = MAX_TOKENS) -> list[int]:
    ids = sp.encode(normalize_text(text), out_type=int, add_bos=True, add_eos=True)
    if len(ids) > max_len:
        ids = ids[:max_len - 1] + [sp.eos_id()]
    return ids


def decode_ids(sp, ids) -> str:
    clean = []
    for idx in ids:
        idx = int(idx)
        if idx == sp.eos_id():
            break
        if idx not in (sp.bos_id(), sp.pad_id()):
            clean.append(idx)
    return normalize_text(sp.decode(clean)) if clean else ""


# --------------------------------------------------------------------------
# Dataset / collate (same shift semantics as the verified baseline)
# --------------------------------------------------------------------------

class SignToTextDataset(Dataset):
    def __init__(self, uids: np.ndarray, texts: np.ndarray, sp, store: PoseSource | None):
        self.uids = uids
        self.sp = sp
        self.store = store
        log(f"Pre-tokenizing {len(texts):,} captions...")
        self.token_ids = [np.asarray(encode_text(sp, t), dtype=np.int64) for t in texts]

    def __len__(self): return len(self.uids)

    def __getitem__(self, idx):
        uid = str(self.uids[idx])
        if self.store is None:  # smoke mode: deterministic synthetic poses
            rng = np.random.default_rng(abs(hash(uid)) % (2 ** 31))
            frames = rng.normal(size=(MAX_FRAMES, LANDMARK_DIM)).astype(np.float32)
            length = MAX_FRAMES
        else:
            frames, length = load_pose_item(uid, self.store)
        return (torch.from_numpy(np.ascontiguousarray(frames)), torch.tensor(length),
                torch.from_numpy(self.token_ids[idx].copy()).long())


def collate_sign_text(batch):
    frames, lengths, tokens = zip(*batch)
    frames = torch.stack(frames)
    lengths = torch.stack(lengths)
    max_tok = max(len(t) for t in tokens)
    tok_pad = torch.full((len(tokens), max_tok), 0, dtype=torch.long)
    for i, t in enumerate(tokens):
        tok_pad[i, :len(t)] = t
    return frames, lengths, tok_pad[:, :-1], tok_pad[:, 1:]


# --------------------------------------------------------------------------
# Train / eval
# --------------------------------------------------------------------------

def build_schedule(opt, total_steps: int):
    warmup = max(1, int(WARMUP_FRACTION * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def run_epoch(model, loader, opt, crit, device, scheduler=None, train: bool = True) -> float:
    model.train(train)
    total = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    amp = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
    scaler = getattr(run_epoch, "_scaler", None)
    if train and scaler is None:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
        run_epoch._scaler = scaler  # type: ignore[attr-defined]
    if train:
        opt.zero_grad(set_to_none=True)
    with ctx:
        for frames, lengths, tin, tout in loader:
            frames, lengths, tin, tout = (t.to(device, non_blocking=True) for t in (frames, lengths, tin, tout))
            src_pad = padding_mask(lengths, frames.size(1))
            tgt_mask = causal_mask(tin.size(1), frames.device)
            with amp:
                logits = model(frames, src_pad, tin, tgt_mask)
                loss = crit(logits.reshape(-1, logits.size(-1)), tout.reshape(-1))
            if train:
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss.")
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
            total += loss.item()
    return total / max(1, len(loader))


@torch.no_grad()
def batch_greedy(model, loader, sp, device, max_decode: int = MAX_DECODE_TOKENS):
    """Constrained batched greedy decode (same bans as serving)."""
    model.eval()
    refs, hyps, confs = [], [], []
    pad, unk, bos, eos = sp.pad_id(), sp.unk_id(), sp.bos_id(), sp.eos_id()
    for frames, lengths, tin, tout in loader:
        frames, lengths = frames.to(device, non_blocking=True), lengths.to(device, non_blocking=True)
        src_pad = padding_mask(lengths, frames.size(1))
        memory = model.encode(frames, src_pad)
        ids = torch.full((frames.size(0), 1), bos, dtype=torch.long, device=device)
        finished = torch.zeros(frames.size(0), dtype=torch.bool, device=device)
        sum_logp = torch.zeros(frames.size(0), device=device)
        steps = torch.zeros(frames.size(0), dtype=torch.long, device=device)
        gen = []
        for step in range(max_decode):
            logits = model.decode(ids, memory, causal_mask(ids.size(1), device), src_pad)[:, -1].float()
            logits[:, [pad, bos, unk]] = -torch.inf
            if step < MIN_OUTPUT_TOKENS:
                logits[:, eos] = -torch.inf
            lp = torch.log_softmax(logits, dim=-1)
            prob, nxt = torch.max(lp, dim=-1)
            active = ~finished
            sum_logp += torch.where(active, prob, torch.zeros_like(prob))
            steps += active.long()
            nxt = torch.where(finished, torch.full_like(nxt, eos), nxt)
            gen.append(nxt.detach().cpu().numpy())
            finished |= nxt.eq(eos)
            ids = torch.cat([ids, nxt[:, None]], dim=1)
            if finished.all():
                break
        gen = np.stack(gen, axis=1)
        conf = torch.exp(sum_logp / steps.clamp(min=1)).detach().cpu().numpy()
        for r, row in enumerate(tout.cpu().tolist()):
            refs.append(decode_ids(sp, row))
            hyps.append(decode_ids(sp, gen[r].tolist()))
            confs.append(float(conf[r]))
    return refs, hyps, confs


def diversity_metrics(hyps: list[str]) -> dict:
    top, count = Counter(hyps).most_common(1)[0] if hyps else ("", 0)
    return {
        "unique": len(set(hyps)),
        "unique_fraction": len(set(hyps)) / max(1, len(hyps)),
        "top_prediction": top,
        "top_prediction_freq": count / max(1, len(hyps)),
    }


def compute_metrics(refs: list[str], hyps: list[str]) -> dict:
    import jiwer
    import sacrebleu
    from rouge_score import rouge_scorer

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return {
        "WER": float(jiwer.wer(refs, hyps)),
        "BLEU": float(sacrebleu.corpus_bleu(hyps, [refs]).score),
        "ROUGE-L": float(np.mean([rouge.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)])),
        "exact_match": float(np.mean([normalize_text(r) == normalize_text(h) for r, h in zip(refs, hyps)])),
        "nonempty_output": float(np.mean([bool(normalize_text(h)) for h in hyps])),
    }


# --------------------------------------------------------------------------
# Export (same bundle layout the backend already loads)
# --------------------------------------------------------------------------

def export_bundle(model, sp, out_dir: Path, run_id: str, train_cfg: dict,
                  eval_report: dict, contract_src: Path | None) -> Path:
    bundle = out_dir / "signsage_deployment_bundle"
    (bundle / "checkpoint").mkdir(parents=True, exist_ok=True)
    (bundle / "tokenizer").mkdir(parents=True, exist_ok=True)
    mc = {
        "input_dim": MODEL_INPUT_DIM, "raw_feature_dim": LANDMARK_DIM,
        "vocab_size": sp.get_piece_size(),
        "d_model": D_MODEL, "nhead": NHEAD, "layers": NLAYERS, "ff": FF,
        "dropout": DROPOUT, "norm_first": NORM_FIRST,
        "max_frames": MAX_FRAMES, "max_tokens": MAX_TOKENS,
    }
    torch.save({"state_dict": model.state_dict(), "model_config": mc,
                "vocab": {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}},
               bundle / "checkpoint" / "signsage_transformer_full.pt")
    shutil.copy(sp_train_artifacts["model"], bundle / "tokenizer" / "signsage_en_sp.model")
    if contract_src and contract_src.is_file():
        shutil.copy(contract_src, bundle / "landmark_contract.json")
    else:
        bundle.joinpath("landmark_contract.json").write_text(json.dumps({
            "total_landmark_count": TOTAL_LANDMARKS, "coordinates_per_landmark": COORDINATES_PER_LANDMARK,
            "raw_feature_dim": LANDMARK_DIM,
            "component_order": ["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"],
            "component_ranges": {"POSE_LANDMARKS": [0, 33], "FACE_LANDMARKS": [33, 501],
                                 "LEFT_HAND_LANDMARKS": [501, 522], "RIGHT_HAND_LANDMARKS": [522, 543]},
            "normalization_version": "pose_normalize_v1", "max_frames": MAX_FRAMES,
            "final_model_input_dim": MODEL_INPUT_DIM, "full_feature_mask": FULL_FEATURE_MASK,
        }, indent=1))
    bundle.joinpath("sign_to_text_deployment.json").write_text(json.dumps({
        "project": "SignSage", "task": "ISL_to_English", "dataset": "iSign v1.1",
        "feature_dim": LANDMARK_DIM, "model_input_dim": MODEL_INPUT_DIM,
        "max_frames": MAX_FRAMES, "max_tokens": MAX_TOKENS, "vocab_size": sp.get_piece_size(),
        "tokenizer": "SentencePiece unigram trained on training targets only",
        "tokenizer_model": "tokenizer/signsage_en_sp.model",
        "special_token_ids": {"pad": 0, "unk": 1, "bos": 2, "eos": 3},
        "normalization_version": "pose_normalize_v1",
        "model_architecture": {"input_dim": MODEL_INPUT_DIM, "raw_feature_dim": LANDMARK_DIM,
                               "vocab_size": sp.get_piece_size(), **{k: mc[k] for k in
                               ("d_model", "nhead", "layers", "ff", "dropout", "norm_first",
                                "max_frames", "max_tokens")}},
        "decoding": {"decoder_runtime": "constrained_greedy", "max_decode_tokens": MAX_DECODE_TOKENS,
                     "min_output_tokens": MIN_OUTPUT_TOKENS,
                     "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE, "repetition_penalty": REPETITION_PENALTY},
        "training_config": train_cfg,
        "checkpoint": "checkpoint/signsage_transformer_full.pt",
    }, indent=1))
    bundle.joinpath("evaluation_results.json").write_text(json.dumps(eval_report, indent=1))
    manifest = {
        "model_id": "signsage-s2t", "run_id": run_id, "architecture": "spatial-temporal-transformer",
        "predecessor": "352868924 (NO-GO baseline; results preserved, not overwritten)",
        "dataset": {"source": "iSign v1.1 (Hugging Face)", **train_cfg.get("data_counts", {})},
        "metrics": eval_report.get("test_metrics", {}),
        "diversity": eval_report.get("diversity", {}),
        "gate": eval_report.get("gate", {}),
        "status": "unpromoted — set SIGNSAGE_MODEL_PROMOTED=true only after the quality gate passes",
    }
    out_dir.joinpath(f"signsage-s2t_v{run_id}.yaml").write_text(
        "# SignSage run manifest (generated by train_sign_to_text.py)\n"
        + json.dumps(manifest, indent=1) + "\n")
    return bundle


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

sp_train_artifacts: dict = {}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="SignSage v2 sign-to-text training")
    ap.add_argument("--csv", default=None, help="path to iSign_v1.1.csv")
    ap.add_argument("--pose-source", default=None,
                    help="local .pose dir, local .zip, or 'remote' (HF streaming, needs HF_TOKEN)")
    ap.add_argument("--contract", default="research/evaluation/reports/352868924/landmark_contract.json",
                    help="landmark contract to copy into the bundle (representation unchanged)")
    ap.add_argument("--out", required=True, help="output dir for checkpoints/bundle/report")
    ap.add_argument("--stages", default="all",
                    help="comma subset of: audit,tokenizer,overfit,train,evaluate,export")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=PEAK_LR)
    ap.add_argument("--max-hours", type=float, default=8.0, help="graceful stop budget")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="CPU plumbing check on synthetic data")
    ap.add_argument("--seed", type=int, default=SEED)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    import pandas as pd

    args = parse_args(argv)
    stages = set(args.stages.split(",")) if args.stages != "all" else {"audit", "tokenizer", "overfit", "train", "evaluate", "export"}
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log(f"run {run_id} on {device}; stages={sorted(stages)}")

    # ---- data ----
    if args.smoke:
        log("SMOKE mode: synthetic poses + tiny vocab targets (no dataset).")
        vocab_texts = [f"hello world {i}" for i in range(8)]
        df_tr = pd.DataFrame({"uid": [f"smoke-{i}" for i in range(8)], "text": vocab_texts})
        df_va = df_tr.iloc[:4].reset_index(drop=True)
        df_te = df_tr.iloc[:4].reset_index(drop=True)
        store = None
    else:
        if not args.csv:
            raise SystemExit("--csv is required (or use --smoke)")
        import pandas as pd  # noqa: F811

        df = pd.read_csv(args.csv)
        cleaned, cleaning = clean_dataframe(df)
        if cleaning["duplicate_uid"]:
            raise SystemExit("duplicate UIDs after cleaning -- refusing.")
        df_tr_all, df_va_all, df_te_all, split = video_split(cleaned, args.seed)
        df_tr, df_va, df_te = df_tr_all, df_va_all, df_te_all
        log(f"data: {cleaning['usable_rows']:,} usable; "
            f"train={len(df_tr):,} val={len(df_va):,} test={len(df_te):,}")
        if "audit" in stages:
            (out / "dataset_audit.json").write_text(json.dumps(
                {"cleaning": cleaning, "split": split,
                 "note": "run audit_dataset.py --pose-source ... for pose-level checks"}, indent=1))
        if args.pose_source == "remote":
            token = os.environ.get("HF_TOKEN")
            if not token:
                raise SystemExit("remote pose streaming needs HF_TOKEN")
            store = RemotePoseSource(token)
        elif args.pose_source and zipfile.is_zipfile(args.pose_source):
            store = ZipPoseSource(args.pose_source)
        elif args.pose_source:
            store = DirPoseSource(args.pose_source)
        else:
            raise SystemExit("--pose-source is required (dir | zip | remote)")

    # ---- tokenizer ----
    tok_dir = out / "tokenizer"
    if "tokenizer" in stages or args.smoke:
        train_texts = df_tr["text"].astype(str).tolist()
        vocab_size = 40 if args.smoke else 16000
        sp = train_tokenizer(train_texts, tok_dir, vocab_size)
        sp_train_artifacts["model"] = str(tok_dir / "signsage_en_sp.model")
        log(f"tokenizer vocab={sp.get_piece_size()}")
    else:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=str(tok_dir / "signsage_en_sp.model"))
        sp_train_artifacts["model"] = str(tok_dir / "signsage_en_sp.model")

    def make_loader(frame, train: bool, batch: int | None = None):
        ds = SignToTextDataset(frame["uid"].to_numpy(), frame["text"].to_numpy(), sp, store)
        return DataLoader(ds, batch_size=batch or args.batch, shuffle=train,
                          collate_fn=collate_sign_text,
                          num_workers=2 if device.type == "cuda" else 0,
                          pin_memory=device.type == "cuda")

    # ---- model ----
    def build_model():
        return SignTransformer(MODEL_INPUT_DIM, sp.get_piece_size(), MAX_FRAMES, MAX_TOKENS,
                               pad_idx=sp.pad_id(), d_model=D_MODEL, nhead=NHEAD, layers=NLAYERS,
                               ff=FF, dropout=DROPOUT, norm_first=NORM_FIRST).to(device)

    # ---- overfit gate (mandatory) ----
    if "overfit" in stages:
        log("overfit gate: 64 memorization samples must reach >=95% exact.")
        import pandas as pd  # noqa: F811

        sub = df_tr.sample(min(64, len(df_tr)), random_state=args.seed).reset_index(drop=True)
        gate_loader = make_loader(sub, True, batch=16)
        gate = build_model()
        gate_opt = torch.optim.AdamW(gate.parameters(), lr=1e-3)
        gate_crit = nn.CrossEntropyLoss(ignore_index=sp.pad_id())
        exact = 0.0
        for step in range(600):
            run_epoch(gate, gate_loader, gate_opt, gate_crit, device)
            if (step + 1) % 100 == 0:
                refs, hyps, _ = batch_greedy(gate, gate_loader, sp, device)
                exact = float(np.mean([normalize_text(r) == normalize_text(h) for r, h in zip(refs, hyps)]))
                log(f"  gate step {step + 1}: exact={exact:.1%}")
                if exact >= 0.95:
                    break
        if exact < 0.95:
            raise SystemExit(f"OVERFIT GATE FAILED (exact={exact:.1%}) -- fix implementation, do not train.")
        log("overfit gate PASSED.")

    # ---- train ----
    best_path = out / "checkpoints" / "signsage_transformer_full.pt"
    history: list = []
    if "train" in stages:
        train_loader = make_loader(df_tr, True)
        val_loader = make_loader(df_va, False)
        model = build_model()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
        steps_per_epoch = math.ceil(len(train_loader))
        scheduler = build_schedule(opt, max(1, steps_per_epoch * args.epochs))
        crit = nn.CrossEntropyLoss(ignore_index=sp.pad_id(), label_smoothing=LABEL_SMOOTHING)
        start_epoch, best, bad = 1, float("inf"), 0
        rec_path = out / "checkpoints" / "signsage_transformer_recovery.pt"
        if args.resume and rec_path.exists():
            rec = torch.load(rec_path, map_location=device)
            model.load_state_dict(rec["state_dict"])
            opt.load_state_dict(rec["optimizer_state_dict"])
            if rec.get("scheduler_state_dict"):
                scheduler.load_state_dict(rec["scheduler_state_dict"])
            start_epoch = int(rec.get("epoch", 0)) + 1
            best = float(rec.get("best", best))
            log(f"resumed at epoch {start_epoch} (best val {best:.4f})")
        div_loader_sample = make_loader(df_va.sample(min(DIVERSITY_SAMPLE, len(df_va)),
                                                      random_state=args.seed).reset_index(drop=True), False)
        for epoch in range(start_epoch, args.epochs + 1):
            tr_loss = run_epoch(model, train_loader, opt, crit, device, scheduler, train=True)
            va_loss = run_epoch(model, val_loader, opt, crit, device, train=False)
            history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})
            log(f"epoch {epoch:02d}/{args.epochs}: train={tr_loss:.4f} val={va_loss:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}")
            # Diversity collapse gate on real decodes (not teacher-forced loss).
            _, dhyps, _ = batch_greedy(model, div_loader_sample, sp, device)
            div = diversity_metrics(dhyps)
            log(f"  diversity: unique={div['unique']}/{len(dhyps)} ({div['unique_fraction']:.2%}) "
                f"top={div['top_prediction']!r} ({div['top_prediction_freq']:.1%})")
            if epoch >= 2 and div["unique_fraction"] < DIVERSITY_FLOOR:
                (out / "COLLAPSED.txt").write_text(
                    f"epoch {epoch}: unique_fraction={div['unique_fraction']:.4f} < {DIVERSITY_FLOOR}; "
                    f"top={div['top_prediction']!r} ({div['top_prediction_freq']:.1%})\n")
                raise SystemExit("COLLAPSE GATE FIRED -- stopped instead of burning more GPU. See COLLAPSED.txt.")
            payload = {"state_dict": model.state_dict(), "epoch": epoch, "val_loss": va_loss, "best": best,
                       "optimizer_state_dict": opt.state_dict(), "scheduler_state_dict": scheduler.state_dict()}
            torch.save(payload, rec_path)
            if va_loss < best - 1e-4:
                best, bad = va_loss, 0
                torch.save(payload, best_path)
                log(f"  new best (val {best:.4f})")
            else:
                bad += 1
                if bad >= args.patience:
                    log("early stopping: val loss stopped improving.")
                    break
            if (time.time() - t_start) / 3600 > args.max_hours:
                log("wall-clock budget reached -- stopping gracefully (resume with --resume).")
                break
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if not best_path.exists():
            raise SystemExit("no best checkpoint produced.")
    else:
        if not best_path.exists():
            raise SystemExit("train stage skipped but no checkpoint found; nothing to evaluate/export.")

    # ---- evaluate ----
    eval_report: dict = {"history": history}
    if "evaluate" in stages:
        model = build_model()
        model.load_state_dict(torch.load(best_path, map_location=device)["state_dict"])
        test_loader = make_loader(df_te, False)
        refs, hyps, confs = batch_greedy(model, test_loader, sp, device)
        test_metrics = compute_metrics(refs, hyps)
        div = diversity_metrics(hyps)
        pool = df_tr.loc[df_tr["text"].astype(str).str.split().str.len() >= 2, "text"].map(normalize_text)
        trivial = [pool.value_counts().idxmax()] * len(refs)
        trivial_metrics = compute_metrics(refs, trivial)
        samples = [{"ref": r, "hyp": h, "conf": c}
                   for r, h, c in zip(refs[:30], hyps[:30], confs[:30])]
        lengths = {"gen_mean": float(np.mean([len(normalize_text(h).split()) for h in hyps])),
                   "ref_mean": float(np.mean([len(normalize_text(r).split()) for r in refs]))}
        gate = {
            "beats_trivial_wer": test_metrics["WER"] < trivial_metrics["WER"],
            "beats_trivial_bleu": test_metrics["BLEU"] > trivial_metrics["BLEU"],
            "diversity_ok": div["unique_fraction"] >= DIVERSITY_FLOOR,
            "exact_match_nonzero": test_metrics["exact_match"] > 0,
        }
        gate["GO"] = all(gate.values())
        eval_report.update({"test_metrics": test_metrics, "trivial_baseline": trivial_metrics,
                            "diversity": div, "lengths": lengths, "samples": samples, "gate": gate,
                            "mean_confidence_proxy": float(np.mean(confs))})
        log(f"test: WER={test_metrics['WER']:.3f} BLEU={test_metrics['BLEU']:.2f} "
            f"unique={div['unique_fraction']:.2%} GO={gate['GO']}")
        (out / "evaluation_report.json").write_text(json.dumps(eval_report, indent=1))

    # ---- export ----
    if "export" in stages:
        model = build_model()
        model.load_state_dict(torch.load(best_path, map_location=device)["state_dict"])
        train_cfg = {"architecture": {"d_model": D_MODEL, "nhead": NHEAD, "layers": NLAYERS,
                                      "ff": FF, "dropout": DROPOUT, "norm_first": NORM_FIRST},
                     "label_smoothing": LABEL_SMOOTHING, "peak_lr": args.lr,
                     "batch": args.batch, "seed": args.seed, "history": history}
        bundle = export_bundle(model, sp, out, run_id, train_cfg, eval_report,
                               Path(args.contract) if args.contract else None)
        log(f"bundle exported: {bundle}")
    log("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
