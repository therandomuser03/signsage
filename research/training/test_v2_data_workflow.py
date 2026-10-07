#!/usr/bin/env python3
"""V2 data-workflow validation (no full training, no network, CPU only).

Covers the Phase A-H rework of train_sign_to_text.py:
  1. pose-cache materialization on a small subset (round-trip exactness)
  2. interrupted/resume simulation (present samples are never re-fetched)
  3. corrupt partial shards are ignored and rebuilt
  4. tokenizer cache reuse (second call never re-encodes)
  5. checkpoint resume payload (opt/sched/epoch/bad/history/RNG round-trip)
  6. batch_greedy() == serving greedy_decode_ids() parity
  7. overfit gate still passes (via --smoke + synthetic pose files)

Usage: python3 research/training/test_v2_data_workflow.py [--gate]
--gate includes the slow overfit-gate subprocess check (default: skip it;
run it explicitly before claiming validation).
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).parent))

import train_sign_to_text as T  # noqa: E402
from train_sign_to_text import (  # noqa: E402
    MAX_FRAMES,
    LANDMARK_DIM,
    MODEL_INPUT_DIM,
    MAX_TOKENS,
    MAX_DECODE_TOKENS,
    MIN_OUTPUT_TOKENS,
    NO_REPEAT_NGRAM_SIZE,
    REPETITION_PENALTY,
    PoseSource,
    ShardPoseSource,
    DirPoseSource,
    SignToTextDataset,
    collate_sign_text,
    batch_greedy,
    load_pose_item,
    load_or_build_token_cache,
    materialize_pose_cache,
    cache_coverage,
)


def make_pose_blob(seed: int, frames: int = 25) -> bytes:
    from pose_format import Pose
    from pose_format.numpy import NumPyPoseBody
    from pose_format.pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions

    comps = []
    for name, n in [("POSE_LANDMARKS", 33), ("FACE_LANDMARKS", 468),
                    ("LEFT_HAND_LANDMARKS", 21), ("RIGHT_HAND_LANDMARKS", 21)]:
        comps.append(PoseHeaderComponent(name=name, points=[f"{name}_{i}" for i in range(n)],
                                         limbs=[], colors=[], point_format="XYZC"))
    header = PoseHeader(version=0.1,
                        dimensions=PoseHeaderDimensions(width=100, height=100, depth=100),
                        components=comps)
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(frames, 1, 543, 3)).astype(np.float32)
    conf = np.ones((frames, 1, 543), dtype=np.float32)
    buf = io.BytesIO()
    Pose(header, NumPyPoseBody(fps=30, data=data, confidence=conf)).write(buf)
    return buf.getvalue()


class FakeRemote(PoseSource):
    def __init__(self, blobs: dict, fail_after: int | None = None):
        self.blobs = blobs
        self.fetches = 0
        self.fail_after = fail_after

    def read_member_bytes(self, uid: str) -> bytes:
        self.fetches += 1
        if self.fail_after is not None and self.fetches > self.fail_after:
            raise ConnectionError("simulated session interruption")
        return self.blobs[uid]


def test_materialize_roundtrip(tmp: Path) -> None:
    uids = [f"vid{i:03d}-0" for i in range(12)]
    blobs = {u: make_pose_blob(i, frames=10 + i) for i, u in enumerate(uids)}
    # reference decode path (pre-existing DirPoseSource) for exactness compare
    rawdir = tmp / "raw"
    rawdir.mkdir(parents=True)
    for u, b in blobs.items():
        (rawdir / f"{u}.pose").write_bytes(b)
    remote = FakeRemote(blobs)
    cache = tmp / "pose_cache"
    materialize_pose_cache(uids, remote, cache, shard_samples=5, verify_every=4)
    assert remote.fetches == 12, remote.fetches
    assert len(list(cache.glob("shard-*.zip"))) == 3
    man = json.loads((cache / "manifest.json").read_text())
    assert man["complete"] is True and len(man["uid_to_shard"]) == 12
    cached = ShardPoseSource(str(cache))
    ref = DirPoseSource(str(rawdir))
    for u in uids:
        f1, l1 = load_pose_item(u, cached)
        f2, l2 = load_pose_item(u, ref)
        assert f1.shape == (MAX_FRAMES, LANDMARK_DIM) and f1.dtype == np.float32
        assert l1 == l2 and np.array_equal(f1, f2), f"cache decode differs for {u}"
    print("PASS materialize round-trip exact (12 samples, 3 shards)")


def test_parallel_materialize(tmp: Path) -> None:
    import threading
    uids = [f"vid{i:03d}-0" for i in range(20)]
    blobs = {u: make_pose_blob(300 + i, frames=10 + (i % 5)) for i, u in enumerate(uids)}
    rawdir = tmp / "raw"
    rawdir.mkdir(parents=True)
    for u, b in blobs.items():
        (rawdir / f"{u}.pose").write_bytes(b)
    lock = threading.Lock()
    made, fetched = [], [0]

    def factory():
        with lock:
            made.append(1)
        return FakeRemote(blobs)

    orig_fetch = FakeRemote.read_member_bytes

    def counted(self, uid):
        with lock:
            fetched[0] += 1
        return orig_fetch(self, uid)

    FakeRemote.read_member_bytes = counted
    try:
        cache = tmp / "pose_cache"
        materialize_pose_cache(uids, FakeRemote(blobs), cache, shard_samples=6,
                               verify_every=7, jobs=4, source_factory=factory)
    finally:
        FakeRemote.read_member_bytes = orig_fetch
    assert fetched[0] == 20, fetched
    assert len(made) >= 2, made  # actually parallel: one source per thread
    man = json.loads((cache / "manifest.json").read_text())
    assert man["complete"] is True and len(man["uid_to_shard"]) == 20
    cached = ShardPoseSource(str(cache))
    ref = DirPoseSource(str(rawdir))
    for u in uids:
        f1, l1 = load_pose_item(u, cached)
        f2, l2 = load_pose_item(u, ref)
        assert l1 == l2 and np.array_equal(f1, f2), f"parallel cache differs for {u}"
    print("PASS parallel materialize (jobs=4, 20 samples, layout identical)")


def test_resume_skips_present(tmp: Path) -> None:
    uids = [f"vid{i:03d}-0" for i in range(10)]
    blobs = {u: make_pose_blob(100 + i) for i, u in enumerate(uids)}
    cache = tmp / "pose_cache"
    first = FakeRemote(blobs)
    materialize_pose_cache(uids[:6], first, cache, shard_samples=4, verify_every=1000)
    assert first.fetches == 6
    # simulate next Kaggle session: fresh process state, same cache dir
    second = FakeRemote(blobs)
    materialize_pose_cache(uids, second, cache, shard_samples=4, verify_every=1000)
    assert second.fetches == 4, second.fetches  # only the missing 4
    present, missing = cache_coverage(cache, uids)
    assert (present, missing) == (10, 0)
    print("PASS interrupted/resume simulation (6 reused, 4 fetched)")


def test_corrupt_shard_rebuilt(tmp: Path) -> None:
    uids = [f"vid{i:03d}-0" for i in range(4)]
    blobs = {u: make_pose_blob(200 + i) for i, u in enumerate(uids)}
    cache = tmp / "pose_cache"
    cache.mkdir(parents=True)
    (cache / "shard-00000.zip").write_bytes(b"truncated-zip-bytes")
    remote = FakeRemote(blobs)
    materialize_pose_cache(uids, remote, cache, shard_samples=10, verify_every=1000)
    assert remote.fetches == 4, remote.fetches
    cached = ShardPoseSource(str(cache))
    for u in uids:
        f, _ = load_pose_item(u, cached)
        assert f.shape == (MAX_FRAMES, LANDMARK_DIM)
    print("PASS corrupt partial shard ignored and rebuilt")


def test_token_cache_reuse(tmp: Path) -> None:
    import sentencepiece as spm

    tok_dir = tmp / "tokenizer"
    tok_dir.mkdir(parents=True)
    texts = [f"hello world number {i} and some more words here" for i in range(30)]
    (tok_dir / "train_text.txt").write_text("\n".join(texts) + "\n")
    spm.SentencePieceTrainer.train(
        input=str(tok_dir / "train_text.txt"), model_prefix=str(tok_dir / "signsage_en_sp"),
        vocab_size=32, model_type="unigram", character_coverage=1.0,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<bos>", eos_piece="<eos>",
        hard_vocab_limit=False)
    sp = spm.SentencePieceProcessor(model_file=str(tok_dir / "signsage_en_sp.model"))
    uids = np.array([f"u{i}" for i in range(30)])
    splits = {"train": (uids, np.array(texts))}
    first = load_or_build_token_cache(sp, tok_dir / "signsage_en_sp.model", tok_dir, splits)
    assert (tok_dir / "tokens.npz").is_file()
    # strong reuse proof: encoding path must not run on the second call
    orig = T.encode_text
    T.encode_text = lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-encoded!"))
    try:
        second = load_or_build_token_cache(sp, tok_dir / "signsage_en_sp.model", tok_dir, splits)
    finally:
        T.encode_text = orig
    assert all(np.array_equal(a, b) for a, b in zip(first["train"], second["train"]))
    # changed texts must rebuild, not silently reuse
    splits2 = {"train": (uids, np.array([t + " extra" for t in texts]))}
    third = load_or_build_token_cache(sp, tok_dir / "signsage_en_sp.model", tok_dir, splits2)
    assert not all(np.array_equal(a, b) for a, b in zip(first["train"], third["train"]))
    print("PASS token cache reuse (no re-encode; mismatch rebuilds)")


def test_checkpoint_resume_keys(tmp: Path) -> None:
    out = tmp / "s2t"
    cmd = [sys.executable, str(Path(__file__).parent / "train_sign_to_text.py"),
           "--smoke", "--out", str(out), "--epochs", "1", "--stages", "tokenizer,train"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    rec = out / "checkpoints" / "signsage_transformer_recovery.pt"
    assert rec.is_file()
    payload = torch.load(rec, map_location="cpu", weights_only=False)
    for key in ("state_dict", "optimizer_state_dict", "scheduler_state_dict",
                "epoch", "best", "bad", "history", "rng"):
        assert key in payload, f"recovery missing {key}"
    assert payload["epoch"] == 1 and len(payload["history"]) == 1
    assert set(payload["rng"]) == {"python", "numpy", "torch", "cuda"}
    cmd2 = [sys.executable, str(Path(__file__).parent / "train_sign_to_text.py"),
            "--smoke", "--out", str(out), "--epochs", "2", "--stages", "train", "--resume"]
    r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=900)
    assert r2.returncode == 0, r2.stdout[-2000:] + r2.stderr[-2000:]
    assert "resumed at epoch 2" in r2.stdout, r2.stdout[-2000:]
    payload2 = torch.load(rec, map_location="cpu", weights_only=False)
    assert payload2["epoch"] == 2 and len(payload2["history"]) == 2, payload2["history"]
    print("PASS checkpoint resume (epoch/bad/history/opt/sched/rng round-trip)")


def test_decode_parity() -> None:
    import sentencepiece as spm
    from torch.utils.data import DataLoader
    from app.ml.models.sign_to_text import SignTransformer
    from app.ml.inference.sign_to_text import greedy_decode_ids
    from app.ml.contracts.model_contract import DecodingConfig

    torch.manual_seed(7)
    np.random.seed(7)
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        texts = [f"the quick brown fox {i} jumps over" for i in range(40)]
        (tp / "c.txt").write_text("\n".join(texts) + "\n")
        spm.SentencePieceTrainer.train(
            input=str(tp / "c.txt"), model_prefix=str(tp / "sp"), vocab_size=40,
            model_type="unigram", character_coverage=1.0,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            pad_piece="<pad>", unk_piece="<unk>", bos_piece="<bos>", eos_piece="<eos>",
            hard_vocab_limit=False)
        sp = spm.SentencePieceProcessor(model_file=str(tp / "sp.model"))
    device = torch.device("cpu")
    model = SignTransformer(MODEL_INPUT_DIM, sp.get_piece_size(), MAX_FRAMES, MAX_TOKENS,
                            pad_idx=sp.pad_id(), d_model=16, nhead=4, layers=1,
                            ff=32, dropout=0.0, norm_first=True).to(device)
    model.eval()
    rng = np.random.default_rng(3)
    frames = torch.from_numpy(rng.normal(size=(3, MAX_FRAMES, LANDMARK_DIM)).astype(np.float32))
    lengths = torch.tensor([160, 25, 7])
    toks = [torch.tensor(T.encode_text(sp, texts[i]), dtype=torch.long) for i in range(3)]
    loader = DataLoader(
        [(frames[i], lengths[i], torch.tensor(T.encode_text(sp, texts[i]), dtype=torch.long))
         for i in range(3)],
        batch_size=3, collate_fn=collate_sign_text)
    refs, hyps, confs = batch_greedy(model, loader, sp, device)
    special = {"pad": sp.pad_id(), "unk": sp.unk_id(), "bos": sp.bos_id(), "eos": sp.eos_id()}
    decoding = DecodingConfig(max_decode_tokens=MAX_DECODE_TOKENS,
                              min_output_tokens=MIN_OUTPUT_TOKENS,
                              no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                              repetition_penalty=REPETITION_PENALTY)
    for i in range(3):
        ids, _ = greedy_decode_ids(model, frames[i], int(lengths[i]), special, decoding, device)
        expect = T.decode_ids(sp, ids[1:])
        assert hyps[i] == expect, f"row {i}: batch={hyps[i]!r} serving={expect!r}"
    assert len(confs) == 3 and all(np.isfinite(confs))
    print("PASS decoder parity (batch_greedy == serving on 3 rows incl. constraints)")


def test_overfit_gate() -> None:
    """Slow: real gate (d256, 600 checks max) on synthetic pose files + tiny CSV."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        posed = td / "poses"
        posed.mkdir()
        rows = []
        vocab = ["hello world", "good morning everybody", "thank you very much",
                 "how are you today", "see you later friend"]
        for i in range(20):
            uid = f"gatevid{i:02d}-0"
            (posed / f"{uid}.pose").write_bytes(make_pose_blob(500 + i, frames=15 + (i % 10)))
            rows.append({"uid": uid, "text": vocab[i % len(vocab)]})
        import pandas as pd
        csv = td / "mini.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        out = td / "s2t"
        cmd = [sys.executable, str(Path(__file__).parent / "train_sign_to_text.py"),
               "--csv", str(csv), "--pose-source", str(posed), "--out", str(out),
               "--stages", "tokenizer,prepare,overfit"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        assert "overfit gate PASSED" in r.stdout, r.stdout[-3000:]
        # gate must have run against the LOCAL cache, not direct dir reads
        assert "LOCAL CACHE" in r.stdout, r.stdout[-3000:]
    print("PASS overfit gate (local cache, >=95% exact)")


def _measure_run(td: Path, n_rows: int = 30, measure_samples: int = 10,
                 extra: list | None = None):
    """Tiny CSV + synthetic pose dir, run --stages measure, return (report, out)."""
    import pandas as pd
    posed = td / "poses"
    posed.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_rows):
        uid = f"mv{i:02d}-0"
        (posed / f"{uid}.pose").write_bytes(make_pose_blob(700 + i, frames=12 + (i % 20)))
        rows.append({"uid": uid, "text": f"sample sentence number {i}"})
    csv = td / "m.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    out = td / "s2t"
    cmd = [sys.executable, str(Path(__file__).parent / "train_sign_to_text.py"),
           "--csv", str(csv), "--pose-source", str(posed), "--out", str(out),
           "--measure-samples", str(measure_samples),
           "--stages", "measure"] + (extra or [])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    rep = json.loads((out / "measure_report.json").read_text())
    return rep, out


def test_measure_report_structure(tmp: Path) -> None:
    rep, out = _measure_run(tmp, extra=["--persistence-target-gib", "50"])
    required = {"n_sampled", "requested", "stride", "per_split_sampled", "true_len",
                "mean_zero_padding_fraction", "uncompressed_bytes", "compressed_bytes",
                "compression_ratio", "avg_compressed_bytes_per_sample",
                "unpadded_bytes", "unpadded_compressed_bytes", "padding_bytes",
                "dense_data_compression_ratio",
                "projected_compressed_gib_full_123779", "projected_compressed_gib_by_split",
                "split_counts", "persistence_target_gib", "safety_margin_gib", "verdict",
                "zstd_level", "representation"}
    assert required <= set(rep), required - set(rep)
    assert rep["n_sampled"] == 10 and rep["requested"] == 10
    assert sum(rep["per_split_sampled"].values()) == 10
    tl = rep["true_len"]
    assert tl["min"] <= tl["median"] <= tl["max"] and tl["p90"] <= tl["p95"] <= tl["max"]
    assert rep["uncompressed_bytes"] == 10 * 160 * 1629 * 4
    assert rep["compressed_bytes"] < rep["uncompressed_bytes"]
    assert rep["compression_ratio"] > 1.0
    assert rep["padding_bytes"] == rep["uncompressed_bytes"] - rep["unpadded_bytes"] >= 0
    assert rep["projected_compressed_gib_full_123779"] > 0
    assert set(rep["projected_compressed_gib_by_split"]) == {"train", "val", "test"}
    assert rep["verdict"] in ("FIT", "BORDERLINE", "DOES_NOT_FIT")
    assert rep["safety_margin_gib"] is not None
    # measurement creates NO cache, NO tokenizer, NO checkpoints
    assert not (out / "pose_cache").exists()
    assert not (out / "tokenizer").exists()
    assert not (out / "checkpoints").exists()
    print("PASS measure report structure (+FIT verdict, no cache created)")


def test_measure_unknown_target(tmp: Path) -> None:
    rep, _ = _measure_run(tmp)
    assert rep["verdict"] == "UNKNOWN-no-target" and rep["safety_margin_gib"] is None
    print("PASS measure without target stays UNKNOWN (no hardcoded limit)")


def test_zstd_roundtrip_unit() -> None:
    import zstandard
    rng = np.random.default_rng(11)
    for shape in ((160, 1629), (37, 1629)):
        a = np.ascontiguousarray(rng.normal(size=shape).astype(np.float32))
        raw = a.tobytes()
        comp = zstandard.ZstdCompressor(level=3).compress(raw)
        assert zstandard.ZstdDecompressor().decompress(comp) == raw
        assert len(comp) < len(raw)  # smooth data must compress
    print("PASS zstd byte-exact round-trip unit")


def test_measure_deterministic(tmp: Path) -> None:
    rep1, _ = _measure_run(tmp / "a")
    rep2, _ = _measure_run(tmp / "b")
    rep1.pop("elapsed_s")
    rep2.pop("elapsed_s")
    assert rep1 == rep2, "measure must be deterministic for identical inputs"
    print("PASS measure deterministic sample accounting")


def test_no_training_config_change() -> None:
    assert (T.D_MODEL, T.NHEAD, T.NLAYERS, T.FF) == (256, 4, 4, 1024)
    assert T.DROPOUT == 0.15 and T.LABEL_SMOOTHING == 0.1
    assert T.PEAK_LR == 3e-4 and T.WARMUP_FRACTION == 0.06
    assert T.WEIGHT_DECAY == 1e-4 and T.GRAD_CLIP == 1.0
    assert T.MAX_FRAMES == 160 and T.MAX_TOKENS == 96 and T.SEED == 42
    assert T.LANDMARK_DIM == 543 * 3 and T.MODEL_INPUT_DIM == 3258
    assert T.NO_REPEAT_NGRAM_SIZE == 3 and T.REPETITION_PENALTY == 1.15
    assert T.DIVERSITY_FLOOR == 0.05 and T.DIVERSITY_SAMPLE == 512
    args = T.parse_args(["--out", "x", "--smoke"])
    assert (args.epochs, args.patience, args.batch) == (12, 3, 64)
    assert args.measure_samples == 2000 and args.subset == 0
    print("PASS no model/training configuration changes")


def main() -> int:
    only_gate = "--gate" in sys.argv
    if only_gate:
        test_overfit_gate()
        return 0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        test_materialize_roundtrip(td / "t1")
        test_parallel_materialize(td / "t1p")
        test_resume_skips_present(td / "t2")
        test_corrupt_shard_rebuilt(td / "t3")
        test_token_cache_reuse(td / "t4")
        test_measure_report_structure(td / "t6")
        test_measure_unknown_target(td / "t7")
        test_measure_deterministic(td / "t8")
        test_checkpoint_resume_keys(td / "t5")
    test_zstd_roundtrip_unit()
    test_no_training_config_change()
    test_decode_parity()
    print("ALL FAST CHECKS PASSED (run with --gate for the overfit-gate check)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
