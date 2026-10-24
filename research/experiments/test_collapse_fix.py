"""CPU-only collapse-diagnostic regression checks; no optimizer steps or downloads.

Run from the repo root: python3 -m pytest -q research/experiments/test_collapse_fix.py
Tiny forward-only models and stubbed orchestration prove mechanics, NOT generalization.
"""
import json
import math
import sys

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, "research/experiments")
sys.path.insert(0, "backend")
import train_sign_to_text as T  # noqa: E402
from app.ml.models.sign_to_text import SignTransformer, causal_mask  # noqa: E402

V, D = 2000, 256


@pytest.fixture(autouse=True)
def forbid_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("These checks must never backpropagate or step an optimizer")

    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _model():
    torch.manual_seed(0)
    return SignTransformer(32, V, 16, 16, pad_idx=0, d_model=D, nhead=4, layers=2, ff=64,
                           dropout=0.0, norm_first=True).eval()


def _init_ce(m):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(4, 16, 16, generator=g)
    tin = torch.randint(4, V, (4, 8), generator=g)
    tout = torch.randint(4, V, (4, 8), generator=g)
    with torch.no_grad():
        lg = m(x, torch.zeros(4, 16, dtype=torch.bool), tin, causal_mask(8, x.device))
    return float(lg.std()), float(nn.CrossEntropyLoss(label_smoothing=0.1)(lg.reshape(-1, V), tout.reshape(-1)))


def test_scaled_init_reduces_initial_logit_scale_not_a_quality_claim():
    legacy, scaled = _model(), _model()
    original = legacy.tok_embed.weight.detach().clone()
    T.init_token_embedding(legacy, "legacy")
    assert torch.equal(legacy.tok_embed.weight, original)
    T.init_token_embedding(scaled, "scaled")
    std_l, ce_l = _init_ce(legacy)
    std_s, ce_s = _init_ce(scaled)
    assert ce_l > 2 * math.log(V), (std_l, ce_l)
    assert ce_s < 1.5 * math.log(V), (std_s, ce_s)
    assert std_s < std_l / 3


def test_scaled_init_stats_pad_row_shapes_and_untied_head():
    m = _model()
    before = {k: v.shape for k, v in m.state_dict().items()}
    head = {k: v.clone() for k, v in m.out.state_dict().items()}
    T.init_token_embedding(m, "scaled")
    w = m.tok_embed.weight.detach()
    assert abs(float(w[1:].std()) - D ** -0.5) < 0.1 * D ** -0.5
    assert float(w[0].abs().max()) == 0.0
    assert m.out.weight.data_ptr() != w.data_ptr()
    assert all(torch.equal(v, head[k]) for k, v in m.out.state_dict().items())
    assert {k: v.shape for k, v in m.state_dict().items()} == before


def test_checkpoint_load_after_init_preserves_every_saved_tensor(tmp_path):
    m = _model()
    T.init_token_embedding(m, "scaled")
    with torch.no_grad():
        m.tok_embed.weight[1].fill_(0.125)
        m.out.bias.fill_(0.25)
    path = tmp_path / "checkpoint.pt"
    T._atomic_torch_save({"state_dict": m.state_dict()}, path)
    restored = _model()
    T.init_token_embedding(restored, "scaled")
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in m.state_dict().items())
    assert restored.tok_embed.weight[0].count_nonzero().item() == 0
    assert restored.out.weight.data_ptr() != restored.tok_embed.weight.data_ptr()


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        T.init_token_embedding(_model(), "bogus")


def test_unigram_baseline_known_values():
    # Targets are 5,6,3 (EOS included), so the NLL approaches ln(3).
    tr = [[2, 5, 6, 3]] * 50
    va = [[2, 5, 6, 3]] * 10
    nll = T.unigram_baseline_nll(tr, va, 100, add_k=(1e-9,))
    assert abs(nll - math.log(3)) < 1e-3
    assert T.unigram_baseline_nll(tr, [[2, 7, 3]], 100) > nll


def test_pose_rolled_loader_swaps_pose_but_not_targets():
    f = torch.arange(4).float().view(4, 1, 1).expand(4, 2, 3).contiguous()
    batch = (f, torch.tensor([1, 2, 3, 4]), torch.arange(8).view(4, 2), torch.arange(8).view(4, 2) + 100)
    out = list(T.PoseRolledLoader([batch]))
    assert len(T.PoseRolledLoader([batch])) == 1
    fr, ln, tin, tout = out[0]
    assert ln.tolist() == [4, 1, 2, 3] and fr[:, 0, 0].tolist() == [3, 0, 1, 2]
    assert torch.equal(tin, batch[2]) and torch.equal(tout, batch[3])
    singleton = tuple(t[:1] for t in batch)
    assert all(torch.equal(a, b) for a, b in zip(next(iter(T.PoseRolledLoader([singleton]))), singleton))


@pytest.mark.parametrize("rolled", [False, True])
def test_token_weighted_nll_includes_eos_ignores_padding_and_not_batch_means(rolled):
    class FixedLogits(nn.Module):
        def forward(self, frames, src_pad, tin, tgt_mask):
            return frames

    # Unequal valid-token counts and a short last batch expose batch-mean bias.
    logits = torch.tensor([[[0., 4., 1., -2.], [1., -1., 3., 0.], [2., 0., -1., 4.]],
                           [[3., 0., -1., 2.], [0., 1., 4., 2.], [0., 0., 0., 0.]],
                           [[4., 0., 1., -3.], [0., 0., 0., 0.], [0., 0., 0., 0.]]])
    targets = torch.tensor([[1, 2, 3], [3, 3, 0], [3, 0, 0]])
    batches = [(lg, torch.full((len(lg),), 3), torch.full_like(tg, 2), tg)
               for lg, tg in zip(logits.split(2), targets.split(2))]
    loader = list(T.PoseRolledLoader(batches)) if rolled else batches
    crit = nn.CrossEntropyLoss(ignore_index=0)
    all_logits = torch.cat([b[0] for b in loader])
    expected = crit(all_logits.reshape(-1, 4), targets.reshape(-1)).item()
    batch_mean = sum(crit(b[0].reshape(-1, 4), b[3].reshape(-1)).item() for b in loader) / len(loader)
    weighted = T.run_epoch(FixedLogits(), loader, None, crit, torch.device("cpu"),
                           train=False, token_weighted=True)
    assert weighted == pytest.approx(expected, abs=1e-6)
    assert abs(weighted - batch_mean) > 0.05
    smooth = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    historical = sum(smooth(b[0].reshape(-1, 4), b[3].reshape(-1)).item() for b in loader) / len(loader)
    assert T.run_epoch(FixedLogits(), loader, None, smooth, torch.device("cpu"),
                       train=False) == pytest.approx(historical)


def test_token_weighted_nll_rejects_incompatible_reduction_and_empty_targets():
    with pytest.raises(ValueError, match="unweighted mean"):
        T.run_epoch(nn.Identity(), [], None, nn.CrossEntropyLoss(reduction="sum"),
                    torch.device("cpu"), train=False, token_weighted=True)
    with pytest.raises(ValueError, match="non-ignored target"):
        T.run_epoch(nn.Identity(), [], None, nn.CrossEntropyLoss(ignore_index=0),
                    torch.device("cpu"), train=False, token_weighted=True)


class _Tokenizer:
    def get_piece_size(self): return 12
    def pad_id(self): return 0
    def unk_id(self): return 1
    def bos_id(self): return 2
    def eos_id(self): return 3
    def encode(self, *args, **kwargs): return [2, 4, 5, 3]


@pytest.fixture
def stubbed_main(monkeypatch):
    """Exercise actual orchestration/guard/saves without dataset access or computation."""
    for name, value in {"LANDMARK_DIM": 16, "MODEL_INPUT_DIM": 32, "MAX_FRAMES": 4,
                        "MAX_TOKENS": 8, "D_MODEL": 16, "NLAYERS": 1, "FF": 32}.items():
        monkeypatch.setattr(T, name, value)
    monkeypatch.setattr(T, "train_tokenizer", lambda *args, **kwargs: _Tokenizer())
    monkeypatch.setattr(T, "_sp_fingerprint", lambda *args: "test-tokenizer")
    calls = {"epochs": [], "decodes": [], "models": [], "collapse": False}

    def run_epoch(model, loader, opt, crit, device, scheduler=None, train=True, **kwargs):
        calls["epochs"].append((train, kwargs.get("token_weighted", False), crit.label_smoothing))
        if train:
            calls["models"].append(model)
            if "before_epoch" in calls:
                calls["before_epoch"](len(calls["models"]))
            return 4.0
        n = len(calls["models"])
        if kwargs.get("token_weighted"):
            return 2.0 / n + (0.2 if isinstance(loader, T.PoseRolledLoader) else 0.0)
        return 3.0 / n

    def batch_greedy(model, loader, sp, device, **kwargs):
        uids = list(loader.dataset.uids)
        refs = [f"hello world {str(u).rsplit('-', 1)[-1]}" for u in uids]
        hyps = ["hello world 0"] * len(refs) if calls["collapse"] else refs.copy()
        calls["decodes"].append((uids, refs, hyps))
        return refs, hyps, [0.5] * len(refs)

    monkeypatch.setattr(T, "run_epoch", run_epoch)
    monkeypatch.setattr(T, "batch_greedy", batch_greedy)
    return calls


def test_collapse_guard_keeps_recovery_and_best_and_writes_forensic_snapshot(tmp_path, monkeypatch, stubbed_main):
    calls = stubbed_main
    calls["collapse"] = True
    monkeypatch.setattr(T, "DIVERSITY_FLOOR", 0.5)  # 1/4 unique fails; epoch 1 remains unchecked.
    out = tmp_path / "run"
    ck = out / "checkpoints"
    previous = {}

    def remember_checkpoints(epoch):
        if epoch == 2:
            for name in ("signsage_transformer_recovery.pt", "signsage_transformer_full.pt"):
                previous[name] = (ck / name).read_bytes()

    calls["before_epoch"] = remember_checkpoints
    with pytest.raises(SystemExit, match="COLLAPSE GATE FIRED"):
        T.main(["--smoke", "--out", str(out), "--epochs", "3", "--batch", "4", "--stages", "train"])
    assert len(calls["models"]) == len(calls["decodes"]) == 2
    assert calls["epochs"] == [(True, False, 0.1), (False, False, 0.1),
                               (False, True, 0.0), (False, True, 0.0)] * 2
    assert (out / "COLLAPSED.txt").is_file()
    for name, contents in previous.items():
        assert (ck / name).read_bytes() == contents
    rec = torch.load(ck / "signsage_transformer_recovery.pt", map_location="cpu", weights_only=False)
    snap = torch.load(ck / "signsage_transformer_collapsed_epoch02.pt", map_location="cpu", weights_only=False)
    assert rec["epoch"] == 1 and snap["epoch"] == 2 and snap["collapsed"] is True
    assert rec["initialization_config"] == snap["initialization_config"] == {"seed": 42, "embed_init": "scaled"}
    assert snap["training_args"]["epochs"] == 3
    hist = json.loads((out / "training_history.json").read_text())
    assert hist[0]["epoch_completed"] and not hist[0]["diversity_checked"]
    assert hist[0]["diversity_pass"] is None
    assert hist[0]["accepted_for_checkpoint"] and hist[0]["checkpoint_written"]
    assert hist[1]["epoch_completed"] and hist[1]["diversity_checked"]
    assert hist[1]["diversity_pass"] is False
    assert not hist[1]["accepted_for_checkpoint"] and not hist[1]["checkpoint_written"]
    assert hist == snap["history"] and hist[:1] == rec["history"]
    for row, (uids, refs, hyps) in zip(hist, calls["decodes"]):
        assert {"val_nll", "val_nll_shuffled_pose", "pose_gap", "embed_init"} <= row.keys()
        assert row["diagnostic_only"] and row["quality_verdict"] is None
        assert row["diagnostic_metrics"] == T.compute_metrics(refs, hyps)
        baseline = T.build_trivial_baseline(refs, [f"hello world {i}" for i in range(8)])
        assert row["dominant_sentence_baseline"] == {"prediction": baseline[0],
                                                     "metrics": T.compute_metrics(refs, baseline)}
        assert row["validation_samples"] == [{"uid": u, "ref": r, "hyp": h, "conf": 0.5}
                                              for u, r, h in zip(uids[:16], refs[:16], hyps[:16])]
    assert not list(out.rglob("*.tmp"))


def test_accepted_epoch_and_resume_preserve_saved_weights_and_init_origin(tmp_path, stubbed_main):
    out = tmp_path / "run"
    args = ["--smoke", "--out", str(out), "--batch", "4", "--stages", "train"]
    assert T.main([*args, "--epochs", "1"]) == 0
    path = out / "checkpoints" / "signsage_transformer_recovery.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    # A different fresh-init choice/seed must not reinitialize loaded weights.
    assert T.main([*args, "--epochs", "2", "--resume", "--seed", "7", "--embed-init", "legacy"]) == 0
    resumed = torch.load(path, map_location="cpu", weights_only=False)
    assert all(torch.equal(v, resumed["state_dict"][k]) for k, v in saved["state_dict"].items())
    assert resumed["initialization_config"] == saved["initialization_config"]
    assert resumed["training_args"]["seed"] == 7 and resumed["training_args"]["embed_init"] == "legacy"
    row = resumed["history"][-1]
    assert row["epoch"] == 2 and row["diversity_checked"] and row["diversity_pass"]
    assert row["accepted_for_checkpoint"] and row["checkpoint_written"]
    assert row["quality_verdict"] is None
    assert json.loads((out / "training_history.json").read_text()) == resumed["history"]


def test_failed_checkpoint_write_is_not_reported_as_written(tmp_path, monkeypatch, stubbed_main):
    def fail_save(*args, **kwargs):
        raise OSError("simulated checkpoint failure")

    monkeypatch.setattr(T, "_atomic_torch_save", fail_save)
    with pytest.raises(OSError, match="simulated checkpoint failure"):
        T.main(["--smoke", "--out", str(tmp_path), "--epochs", "1", "--stages", "train"])
    row = json.loads((tmp_path / "training_history.json").read_text())[-1]
    assert row["epoch_completed"] and row["accepted_for_checkpoint"]
    assert not row["checkpoint_written"]


def test_atomic_save_failure_preserves_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    T._atomic_torch_save({"epoch": 1}, path)
    previous = path.read_bytes()

    def partial_save(obj, fh):
        fh.write(b"partial checkpoint")
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(torch, "save", partial_save)
    with pytest.raises(OSError, match="simulated interrupted write"):
        T._atomic_torch_save({"epoch": 2}, path)
    assert path.read_bytes() == previous


@pytest.mark.parametrize("stage", ["train", "evaluate", "export"])
@pytest.mark.parametrize("evidence", ["flag", "old_history", "guard_history"])
def test_resume_evaluate_export_fail_closed_for_explicitly_collapsed_artifacts(
        tmp_path, stubbed_main, stage, evidence):
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    name = "signsage_transformer_recovery.pt" if stage == "train" else "signsage_transformer_full.pt"
    payload = {"epoch": 2}
    if evidence == "flag":
        payload["collapsed"] = True
    elif evidence == "old_history":
        payload["history"] = [{"epoch": 2, "unique_fraction": 0.01}]
    else:
        payload["history"] = [{"epoch": 2, "diversity_checked": True, "diversity_pass": False}]
    T._atomic_torch_save(payload, ck / name)
    with pytest.raises(SystemExit, match="refusing explicitly collapsed checkpoint"):
        T.main(["--smoke", "--out", str(tmp_path), "--stages", stage, "--resume"])
    assert not stubbed_main["epochs"] and not stubbed_main["decodes"]
    assert not (tmp_path / "signsage_deployment_bundle").exists()


@pytest.mark.parametrize("payload", [
    {"epoch": 1, "history": [{"epoch": 1, "unique_fraction": 0.001}]},
    {"epoch": 1, "history": [{"epoch": 1, "diversity_checked": False, "diversity_pass": None}]},
    {"epoch": 2},  # No evidence is not certification, but legacy format remains loadable.
    {"epoch": 2, "history": [{"epoch": 2, "unique_fraction": 0.5},
                              {"epoch": 3, "unique_fraction": 0.001}]},
])
def test_checkpoint_guard_does_not_mislabel_unchecked_or_earlier_states(payload):
    T.check_checkpoint_not_collapsed(payload, "checkpoint.pt")


@pytest.mark.parametrize("mode", [None, "legacy"])
def test_temporary_benchmark_preserves_rng_model_and_quality_stats_without_steps(monkeypatch, stubbed_main, mode):
    # Zero benchmark iterations: exercise allocation/init/cleanup, never optimization.
    monkeypatch.setattr(T, "BENCH_TRAIN_STEPS", 0)
    sentinel = {"grad_norm_mean": 123.0}
    scaler = object()
    monkeypatch.setattr(T.run_epoch, "last_train_stats", sentinel, raising=False)
    monkeypatch.setattr(T.run_epoch, "_scaler", scaler, raising=False)
    actual = SignTransformer(32, 12, 4, 8, 0, d_model=16, layers=1, ff=32)
    state = {k: v.clone() for k, v in actual.state_dict().items()}
    rng = torch.get_rng_state().clone()
    kwargs = {} if mode is None else {"embed_init": mode}
    result = T.benchmark_training_throughput(torch.device("cpu"), 12, 2, **kwargs)
    assert result["embed_init"] == (mode or "scaled")
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(v, state[k]) for k, v in actual.state_dict().items())
    assert T.run_epoch.last_train_stats is sentinel and T.run_epoch._scaler is scaler
    assert not stubbed_main["epochs"]
