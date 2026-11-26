"""FDMSEWordAdapter: bundle loading guards, classification contract, and registry dispatch."""
import json

import numpy as np
import pytest
import torch

from app.ml.fdmse_word_adapter import DEFAULT_ASPECT, FDMSEWordAdapter, SignGRU
from app.ml.word_classifier_adapter import WordClassifierAdapter



def _physical_clip(n: int, t0: float = 0.0, t1: float = 1.0):
    """A signing-like clip in physical units (image height = 1), sampled at n uniform times in [t0, t1]."""
    t = np.linspace(t0, t1, n)
    arr = np.zeros((n, 543, 3), np.float32)
    base = np.array([0.9, 0.5])
    arr[:, 11, :2] = base + (-0.25, 0.0)
    arr[:, 12, :2] = base + (0.25, 0.0)
    head = base + (0.0, -0.35)
    arr[:, 0, :2] = head
    arr[:, 2, :2] = head + (-0.04, -0.05)
    arr[:, 5, :2] = head + (0.04, -0.05)
    arr[:, 9, :2] = head + (-0.03, 0.06)
    arr[:, 10, :2] = head + (0.03, 0.06)
    ang = np.linspace(0, 2 * np.pi, 21)
    for slot, (sx, k) in enumerate(((0.3, 1.0), (-0.3, 1.7))):
        wx = base[0] + sx + 0.12 * np.sin(2 * np.pi * k * t)
        wy = base[1] - 0.1 + 0.12 * np.cos(2 * np.pi * t)
        sl = slice(501, 522) if slot == 0 else slice(522, 543)
        arr[:, sl, 0] = wx[:, None] + 0.06 * np.cos(ang)[None]
        arr[:, sl, 1] = wy[:, None] + 0.06 * np.sin(ang)[None]
    return arr


def _to_normalized(phys: np.ndarray, aspect: float, shift=(0.0, 0.0)) -> np.ndarray:
    out = phys.copy()
    nz = out != 0
    out[..., 0] = np.where(nz[..., 0], (out[..., 0] + shift[0]) / aspect, 0.0)
    out[..., 1] = np.where(nz[..., 1], out[..., 1] + shift[1], 0.0)
    return out




def _make_bundle(root, n_classes=5, threshold=0.5, spec="sign_features_v4", face_source="pose", with_threshold=True):
    arch = dict(architecture="SignGRU", n_classes=n_classes, proj=16, hidden=16, layers=2, dropout=0.1)
    torch.manual_seed(0)
    torch.save(SignGRU(n_classes, 122, 16, 16, 2, 0.1).state_dict(), root / "model.pt")
    (root / "model_config.json").write_text(json.dumps(arch))
    (root / "preprocessing.json").write_text(json.dumps(dict(
        feature_spec=spec, feature_spec_detail=dict(name=spec, face_source=face_source, forehead_k=1.2, chin_k=0.6, aspect=1.263158),
        n_frames=60, feat_dim=122, standardize_mean=[0.0] * 122, standardize_std=[1.0] * 122,
        extraction=dict(n_frames=48, face_every_k=4))))
    (root / "labels.json").write_text(json.dumps(dict(classes=[f"word{i}" for i in range(n_classes)])))
    (root / "metrics.json").write_text(json.dumps(dict(confidence_threshold=threshold) if with_threshold else {}))
    return root


@pytest.fixture()
def bundle(tmp_path):
    return _make_bundle(tmp_path)


def test_load_and_classify_contract(bundle):
    ad = FDMSEWordAdapter.load(bundle, device="cpu")
    assert ad.labels == [f"word{i}" for i in range(5)] and ad.confidence_threshold == 0.5
    assert ad.model_version.startswith("fdmse-isl-5w-v4-") and ad.feature_version == "sign_features_v4"
    assert ad.promotion_eligible() is False
    res = ad.classify(_to_normalized(_physical_clip(27), 16 / 9), aspect=16 / 9)
    assert res.status == "ok" and res.label in ad.labels and 0 < res.confidence <= 1
    assert len(res.top3) == 3 and abs(sum(p["probability"] for p in res.top3) - sum(sorted([p["probability"] for p in res.top3]))) < 1e-9
    assert res.top3[0]["probability"] >= res.top3[1]["probability"] >= res.top3[2]["probability"]
    assert res.frames_with_hand == 27 and res.frames_in == 27 and res.feature_ms >= 0 and res.inference_ms >= 0
    assert res.aspect_source == "client" and abs(res.aspect_used - 16 / 9) < 1e-9


def test_aspect_falls_back_to_a_labelled_assumption(bundle):
    ad = FDMSEWordAdapter.load(bundle, device="cpu")
    clip = _to_normalized(_physical_clip(30), 16 / 9)
    for bad in (None, 0.0, -1.0, float("nan"), 99.0):
        r = ad.classify(clip, aspect=bad)
        assert r.aspect_source == "assumed" and abs(r.aspect_used - DEFAULT_ASPECT) < 1e-9


def test_no_hands_short_circuit(bundle):
    ad = FDMSEWordAdapter.load(bundle, device="cpu")
    clip = _to_normalized(_physical_clip(30), 16 / 9)
    clip[:, 501:543] = 0
    r = ad.classify(clip, aspect=16 / 9)
    assert r.status == "no_hands" and r.label is None and r.top3 == []


@pytest.mark.parametrize("bad", [
    np.zeros((10, 100, 3), np.float32), np.zeros((543, 3), np.float32), np.zeros((1, 543, 3), np.float32),
    np.full((5, 543, 3), np.nan, np.float32), np.full((5, 543, 3), np.inf, np.float32),
    np.zeros((5, 543, 3), dtype=complex), np.array([["a"] * 3] * 543 * 5).reshape(5, 543, 3)])
def test_bad_input_raises_value_error(bundle, bad):
    with pytest.raises(ValueError):
        FDMSEWordAdapter.load(bundle, device="cpu").classify(bad)


def test_same_input_same_output_and_batch_independent(bundle):
    ad = FDMSEWordAdapter.load(bundle, device="cpu")
    clip = _to_normalized(_physical_clip(30), 16 / 9)
    a, b = ad.classify(clip, aspect=1.7), ad.classify(clip.copy(), aspect=1.7)
    assert a.top3 == b.top3


def test_motion_gate_helper(bundle):
    ad = FDMSEWordAdapter.load(bundle, device="cpu")
    still = _to_normalized(_physical_clip(30), 16 / 9)
    still[:, 501:543, :2] = still[0:1, 501:543, :2]                                      # frozen hands
    moving = _to_normalized(_physical_clip(30), 16 / 9)
    rs, rm = ad.classify(still, 16 / 9), ad.classify(moving, 16 / 9)
    assert rs.motion_feature < rm.motion_feature
    assert ad.is_moving(rs) is True                                                      # no floor calibrated -> never blocks
    ad.min_motion_feature = (rs.motion_feature + rm.motion_feature) / 2
    assert ad.is_moving(rs) is False and ad.is_moving(rm) is True


@pytest.mark.parametrize("kw,msg", [
    (dict(spec="sign_features_v3"), "unsupported feature spec"),
    (dict(face_source="mesh"), "face_source"),
    (dict(with_threshold=False), "threshold")])
def test_wrong_bundles_are_refused(tmp_path, kw, msg):
    with pytest.raises(ValueError, match=msg):
        FDMSEWordAdapter.load(_make_bundle(tmp_path, **kw), device="cpu")


def test_missing_file_and_label_mismatch_are_refused(tmp_path):
    root = _make_bundle(tmp_path)
    (root / "labels.json").write_text(json.dumps(dict(classes=["only", "two"])))
    with pytest.raises(ValueError, match="disagree"):
        FDMSEWordAdapter.load(root, device="cpu")
    (root / "model.pt").unlink()
    with pytest.raises(FileNotFoundError):
        FDMSEWordAdapter.load(root, device="cpu")


def test_word_classifier_adapter_dispatches_to_fdmse_bundles(bundle):
    ad = WordClassifierAdapter.load(bundle, device="cpu")
    assert isinstance(ad, FDMSEWordAdapter)


def test_motion_floor_is_read_from_metrics(tmp_path):
    root = _make_bundle(tmp_path)
    m = json.loads((root / "metrics.json").read_text())
    m["motion"] = {"min_motion_feature": 0.0123}
    (root / "metrics.json").write_text(json.dumps(m))
    assert FDMSEWordAdapter.load(root, device="cpu").min_motion_feature == 0.0123


def test_adapter_uses_the_robust_recipe_by_default(bundle):
    ad = FDMSEWordAdapter.load(bundle, device="cpu")
    assert ad.robust_hands is True
