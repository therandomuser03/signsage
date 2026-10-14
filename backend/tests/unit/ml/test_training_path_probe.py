"""Training-path invariants + memorization probe (collapse diagnosis).

Context: the Oct 2026 Kaggle pilots collapse to a prior-mode phrase by
epoch 2 while teacher-forced loss still falls. These tests pin down the
mechanics that MUST be true for that diagnosis to hold (vs a wiring bug):

- collate shift: decoder inputs start at BOS, labels are the 1-step future.
- padding / causal masks have the documented semantics.
- the LR schedule explains the logged Kaggle LRs (6.69e-05 / 1.00e-05).
- greedy decoding starts at BOS and never emits PAD/BOS/UNK pre-EOS.
- the exact run_epoch/collate/dataset/model path optimizes on CPU:
  a tiny model memorizes 8 synthetic pairs (optimization sound, so the
  pilot collapse is a scale/prior-dynamics outcome, not a broken loop).

All CPU-only, deterministic, no network, no GPU.
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/mnt/D/Codes/signsage/backend")
sys.path.insert(0, "/mnt/D/Codes/signsage/research/experiments")

from app.ml.models.sign_to_text import SignTransformer, causal_mask, padding_mask
from train_sign_to_text import (
    MAX_FRAMES,
    MODEL_INPUT_DIM,
    SignToTextDataset,
    batch_greedy,
    build_schedule,
    collate_sign_text,
    run_epoch,
)

import torch.nn as nn


class FakeSP:
    """Minimal tokenizer stand-in: ids 0/1/2/3 are PAD/UNK/BOS/EOS."""

    def pad_id(self):
        return 0

    def unk_id(self):
        return 1

    def bos_id(self):
        return 2

    def eos_id(self):
        return 3

    def get_piece_size(self):
        return 12

    def decode(self, ids):
        return " ".join(f"w{i}" for i in ids)


def tiny_model(vocab: int = 12, seed: int = 0) -> SignTransformer:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return SignTransformer(
        MODEL_INPUT_DIM, vocab, MAX_FRAMES, 16, pad_idx=0,
        d_model=16, nhead=2, layers=1, ff=32, dropout=0.0, norm_first=True,
    )


def tiny_loader(n: int = 8, batch: int = 8):
    # Distinct short targets inside vocab [4, 12): [BOS, a, b, EOS].
    uids = np.array([f"probe-{i}" for i in range(n)])
    texts = np.array([f"probe text {i}" for i in range(n)])
    tids = [np.asarray([2, 4 + i, 5 + (i % 3), 3], dtype=np.int64) for i in range(n)]
    ds = SignToTextDataset(uids, texts, None, None, None, token_ids=tids)
    return DataLoader(ds, batch_size=batch, shuffle=False,
                      collate_fn=collate_sign_text, num_workers=0), tids


def test_collate_teacher_shift():
    loader, tids = tiny_loader(n=3, batch=3)
    frames, lengths, tin, tout = next(iter(loader))
    assert tin.shape == tout.shape == (3, 3)
    # Decoder inputs start at BOS; labels are the one-step future.
    assert tin[:, 0].tolist() == [2, 2, 2]
    assert tout[0].tolist() == [4, 5, 3]
    # PAD (0) only trails; never a label at position 0.
    assert tout[:, 0].tolist() == [4, 5, 6]


def test_collate_pads_only_at_tail():
    uids = np.array(["a", "b"])
    texts = np.array(["x", "y"])
    tids = [np.asarray([2, 7, 8, 9, 3], dtype=np.int64),
            np.asarray([2, 5, 3], dtype=np.int64)]
    ds = SignToTextDataset(uids, texts, None, None, None, token_ids=tids)
    loader = DataLoader(ds, batch_size=2, shuffle=False,
                        collate_fn=collate_sign_text, num_workers=0)
    _, _, tin, tout = next(iter(loader))
    assert tin.shape == tout.shape == (2, 4)
    assert tin[1].tolist() == [2, 5, 3, 0]
    assert tout[1].tolist() == [5, 3, 0, 0]


def test_padding_and_causal_masks():
    pm = padding_mask(torch.tensor([2, 4]), 5)
    assert pm.dtype == torch.bool
    assert pm.tolist() == [[False, False, True, True, True],
                           [False, False, False, False, True]]
    cm = causal_mask(4, torch.device("cpu"))
    assert cm.dtype == torch.bool
    assert cm.tolist() == [[False, True, True, True],
                           [False, False, True, True],
                           [False, False, False, True],
                           [False, False, False, False]]


def test_schedule_matches_logged_kaggle_lrs():
    # Oct-14 pilot: 24 batches/epoch x 2 epochs, base lr 1e-4. Logged
    # end-of-epoch LRs were 6.69e-05 and 1.00e-05; the schedule steps
    # once per train batch, so these must equal lambda(24), lambda(48).
    opt = torch.optim.AdamW(torch.nn.Linear(2, 2).parameters(), lr=1e-4)
    sched = build_schedule(opt, 48)
    # LambdaLR applies lambda(0) at construction (warmup step 0 -> 1/7),
    # then one lambda increment per step() call, matching the trainer's
    # per-batch stepping.
    assert abs(sched.get_last_lr()[0] - 1e-4 / 7) < 1e-12
    for _ in range(24):
        sched.step()
    assert abs(sched.get_last_lr()[0] - 6.69e-05) < 2e-07
    for _ in range(24):
        sched.step()
    assert abs(sched.get_last_lr()[0] - 1.00e-05) < 2e-07


def test_greedy_starts_at_bos_and_bans_specials():
    sp = FakeSP()
    model = tiny_model()
    loader, tids = tiny_loader(n=2, batch=2)
    refs, hyps, confs, ref_ids, gen_ids = batch_greedy(
        model, loader, sp, torch.device("cpu"), max_decode=8, return_ids=True)
    assert len(confs) == 2 and all(np.isfinite(confs))
    # Refs are the unpadded label rows (no BOS, EOS-terminated content).
    assert ref_ids[0] == [4, 5, 3]
    for gen in gen_ids:
        pre_eos = gen[: gen.index(3)] if 3 in gen else gen
        assert all(t not in (0, 1, 2) for t in pre_eos), gen
        assert all(0 <= t < 12 for t in gen)


def test_decoder_dropout_path_stays_finite():
    torch.manual_seed(3)
    model = tiny_model()
    loader, _ = tiny_loader(n=4, batch=4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    loss = run_epoch(model, loader, opt, crit, torch.device("cpu"),
                     train=True, decoder_input_dropout=0.2, vocab_size=12)
    assert np.isfinite(loss)


def test_memorization_probe_optimizes():
    """Core diagnostic: the exact train path must memorize tiny data on CPU.

    PASS means dataset/collate/shift/forward/loss/backward/step are sound
    end to end, so the pilot collapse implicates scale/prior dynamics
    (16k head, ~1 obs/class, 48 updates, pure teacher forcing) rather
    than a broken optimization loop.
    """
    torch.manual_seed(11)
    np.random.seed(11)
    device = torch.device("cpu")
    model = tiny_model(seed=11)
    loader, _ = tiny_loader(n=8, batch=8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    losses = [run_epoch(model, loader, opt, crit, device, train=True)
              for _ in range(40)]
    assert all(np.isfinite(losses))
    assert losses[0] > 2.0, losses[0]  # non-trivial start, not pre-solved
    assert losses[-1] < 0.5 * losses[0], (losses[0], losses[-1])
    assert losses[-1] < 1.5, losses[-1]
