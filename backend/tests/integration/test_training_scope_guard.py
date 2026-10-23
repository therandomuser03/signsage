"""Training-scope awareness: a pilot/smoke bundle must never be servable as a promoted model.

The training script records the scope inside the checkpoint's model_config
(`training_scope`, `is_pilot`, `is_smoke`). These tests tamper with a private copy of the tiny
real fake_bundle to cover every scope, then exercise the adapter, the registry guard and the
`scripts.predict_landmarks` CLI. No training happens here.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.ml.adapters.sign_to_text_adapter import SignToTextAdapter
from app.ml.contracts.inference_contract import DeploymentContract


def _set_scope(bundle, **model_config_updates):
    ckpt = bundle / DeploymentContract.from_bundle(bundle).model.checkpoint_path
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    payload["model_config"].update(model_config_updates)
    torch.save(payload, ckpt)


@pytest.mark.parametrize("updates,expected", [
    ({}, "unknown"),                                              # bundles that predate the key
    ({"training_scope": "full_dataset"}, "full_dataset"),
    ({"is_pilot": True, "training_scope": "pilot_subset"}, "pilot_subset"),
    ({"is_pilot": True}, "pilot_subset"),                         # is_pilot alone is enough
    ({"is_smoke": True, "training_scope": "smoke_synthetic"}, "smoke_synthetic"),
    ({"is_smoke": True, "is_pilot": True}, "smoke_synthetic"),    # smoke wins over pilot
])
def test_adapter_reports_the_recorded_training_scope(fake_bundle, updates, expected):
    _set_scope(fake_bundle, **updates)
    assert SignToTextAdapter.load(fake_bundle, device="cpu").training_scope == expected


@pytest.fixture
def served(monkeypatch):
    """Point the real settings/registry at a bundle with SIGNSAGE_MODEL_PROMOTED=true."""
    from app.core.config import get_settings
    from app.ml.models.registry import get_sign_to_text_pipeline, reset_registry

    def _go(bundle, allow_unpromoted=False):
        monkeypatch.setenv("SIGNSAGE_SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SIGNSAGE_SUPABASE_PUBLISHABLE_KEY", "pub")
        monkeypatch.setenv("SIGNSAGE_MODEL_PROMOTED", "true")
        monkeypatch.setenv("SIGNSAGE_MODEL_BUNDLE_DIR", str(bundle))
        get_settings.cache_clear()
        reset_registry()
        return get_sign_to_text_pipeline(allow_unpromoted=allow_unpromoted)

    yield _go
    reset_registry()
    get_settings.cache_clear()


@pytest.mark.parametrize("scope_updates", [{"is_pilot": True}, {"is_smoke": True}])
def test_promoted_flag_cannot_serve_a_pilot_or_smoke_bundle(fake_bundle, served, scope_updates):
    from app.core.errors import ModelNotPromotedError
    _set_scope(fake_bundle, **scope_updates)
    with pytest.raises(ModelNotPromotedError, match="pipeline validation only"):
        served(fake_bundle)


def test_dev_override_can_still_exercise_a_pilot_bundle(fake_bundle, served):
    _set_scope(fake_bundle, is_pilot=True)
    assert served(fake_bundle, allow_unpromoted=True).training_scope == "pilot_subset"


@pytest.mark.parametrize("updates", [{}, {"training_scope": "full_dataset"}])
def test_promoted_flag_still_serves_full_and_legacy_bundles(fake_bundle, served, updates):
    _set_scope(fake_bundle, **updates)
    assert served(fake_bundle).training_scope in ("unknown", "full_dataset")


# ---------------------------------------------------------------- scripts.predict_landmarks
def _run_cli(capsys, *argv):
    from scripts.predict_landmarks import main
    code = main(list(argv))
    return code, json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_cli_transcribes_a_valid_landmark_file_and_labels_scope(fake_bundle, tmp_path, capsys):
    _set_scope(fake_bundle, is_pilot=True)
    f = tmp_path / "lm.npy"
    np.save(f, np.random.default_rng(0).normal(size=(40, 543, 3)).astype(np.float32))
    code, out = _run_cli(capsys, "--bundle", str(fake_bundle), "--input", str(f), "--device", "cpu")
    assert code == 0 and out["ok"] is True
    assert out["model_scope"] == "pilot_subset" and "PILOT" in out["warning"]
    assert out["frames"] == 40 and out["accepted"] in (True, False) and "reason" in out


def test_cli_rejects_non_canonical_input_with_exit_2(fake_bundle, tmp_path, capsys):
    f = tmp_path / "bad.npy"
    np.save(f, np.zeros((40, 500, 3), np.float32))
    code, out = _run_cli(capsys, "--bundle", str(fake_bundle), "--input", str(f), "--device", "cpu")
    assert code == 2 and out["ok"] is False and out["stage"] == "input"


def test_cli_rejects_a_corrupt_or_unsupported_file_with_exit_2(fake_bundle, tmp_path, capsys):
    corrupt = tmp_path / "x.npy"
    corrupt.write_bytes(b"not numpy")
    assert _run_cli(capsys, "--bundle", str(fake_bundle), "--input", str(corrupt), "--device", "cpu")[0] == 2
    txt = tmp_path / "x.txt"
    txt.write_text("hello")
    assert _run_cli(capsys, "--bundle", str(fake_bundle), "--input", str(txt), "--device", "cpu")[0] == 2


def test_cli_reports_a_missing_bundle_with_exit_1(tmp_path, capsys):
    f = tmp_path / "lm.npy"
    np.save(f, np.zeros((5, 543, 3), np.float32))
    code, out = _run_cli(capsys, "--bundle", str(tmp_path / "nope"), "--input", str(f))
    assert code == 1 and out["stage"] == "load_bundle"
