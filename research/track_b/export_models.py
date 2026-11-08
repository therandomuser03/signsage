"""Track B step 5: export models.

(a) research/track_b/export_s1/  : cnn-v2 trained on S1 ONLY (the evaluated model).
(b) research/track_b/export_final/ : cnn-v2 trained on S1+S2, labelled FINAL-UNEVALUATED
    (no held-out test exists for it; S2 was already used as the test set for (a)).

For each export:
  - model.pt              (torch state dict)
  - meta.json             (label list, features_v2 spec, normalisation constants,
                           extractor name/version/options, fps note,
                           recommended confidence threshold derived from the S2 table
                           [calibrated on S2], s2_test_metrics for (a))
  - golden_test_vector.npy  (a (T,543,3) input clip)
  - golden_expected_logits.json  (expected logits for that input, from this model)

Also writes a standalone single-file sign_features_v2.py (numpy only, same output as
features_v2.py) plus a unit test proving identical output. Reports CPU latency per clip.

Uses EXACTLY the cnn-v2 setup from REPORT_E (dropout 0.3, 60 epochs, Adam 1e-3, batch 16,
no augmentation). Nothing integrated into the app. Offline, CPU.
"""
import csv, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import features_v2 as fv
from eval_v2 import TinyCNN1D, train_torch, predict_torch

S1_MANIFEST = ROOT / "data" / "s1_converted" / "manifest.csv"
S2_MANIFEST = ROOT / "data" / "s2_converted" / "manifest.csv"
CORE10 = ["hello", "thank you", "please", "yes", "no", "help", "water", "you", "me", "where"]
REST = "rest"
TRAIN_LABELS = CORE10 + [REST]   # 11 classes
SEED = 0
EPOCHS = 60


def load_manifest(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def load_features(manifest_path, wanted_labels):
    rows = list(csv.DictReader(open(manifest_path, encoding="utf-8")))
    X, y, meta = [], [], []
    for r in rows:
        lab = r["label"]
        if wanted_labels is not None and lab not in wanted_labels:
            continue
        arr = np.load(r["npy"]).astype(np.float32)
        f = fv.extract_features(arr, layout="signsage")   # (60, 122)
        assert f.shape == (fv.N_FRAMES, fv.FEAT_DIM), f.shape
        X.append(f); y.append(lab); meta.append(r)
    return np.stack(X), y, meta, rows


def li_map(labels):
    return {w: i for i, w in enumerate(labels)}


def train_on(X, y, labels, n_classes, seed=SEED):
    np.random.seed(seed); torch.manual_seed(seed)
    li = li_map(labels)
    yi = np.array([li[w] for w in y], dtype=np.int64)
    model = TinyCNN1D(n_in=fv.FEAT_DIM, n_classes=n_classes, dropout=0.3)
    train_torch(model, X, yi, epochs=EPOCHS, seed=seed)
    return model, li


def cpu_latency(model, X, n_repeats=3):
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            for i in range(len(X)):
                t0 = time.perf_counter()
                _ = model(torch.from_numpy(X[i:i + 1]).float())
                times.append(time.perf_counter() - t0)
    return float(np.mean(times)) * 1000.0  # ms per clip


def logits_for(model, arr_543):
    """arr_543: (T,543,3) raw landmarks -> logits (11,) numpy."""
    f = fv.extract_features(arr_543, layout="signsage")   # (60,122)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(f[None]).float())
    return logits[0].numpy()


# ---------- standalone sign_features_v2.py ----------
STANDALONE_HEADER = '''"""sign_features_v2.py

Standalone, numpy-only re-implementation of research/track_b/features_v2.py.
Same output: (n_frames, 122) float32 per-frame hand-centric features for a
SignSage (T,543,3) landmark array (face mesh index i at absolute 33+i).

Per-frame vector (122 dims), x,y only:
  left hand  (42): wrist - shoulder_centre / shoulder_width;
                   other 20 landmarks - wrist / hand_size
  right hand (42): same
  left_mask, right_mask (2): 1 if the hand has data after gap-fill
  hand-to-face (32): per hand (L then R), wrist then index tip, -> nose/chin/mouth/forehead,
                     scaled by shoulder width
  wrist velocity (4): frame difference per wrist

Missing-hand handling:
  - hand missing in a frame when all 21 landmarks are zero
  - missing runs <= MAX_GAP frames linearly interpolated (mask stays 1)
  - runs still missing after that keep zero coords + mask=0; every hand-derived
    channel is multiplied by the mask (missing zeros never normalised into shoulder space)
  - output resampled to N_FRAMES after filling
"""
import numpy as np

N_FRAMES = 60
FEAT_DIM = 122
MAX_GAP = 5
EPS = 1e-6

_LHAND = slice(501, 522)
_RHAND = slice(522, 543)
_FACE_NOSE = 33 + 1
_FACE_CHIN = 33 + 152
_FACE_MOUTH = 33 + 13
_FACE_FOREHEAD = 33 + 10
_L_SHOULDER = 11
_R_SHOULDER = 12
_WRIST = 0
_INDEX_TIP = 8
_MIDDLE_TIP = 12
_FACE_POINTS = (_FACE_NOSE, _FACE_CHIN, _FACE_MOUTH, _FACE_FOREHEAD)


def _xy(landmarks, sl):
    return landmarks[:, sl, :2]


def _present(hand_xy):
    return np.any(hand_xy != 0.0, axis=(1, 2))


def _fill_gaps(hand_xy, max_gap=MAX_GAP):
    T = hand_xy.shape[0]
    filled = hand_xy.copy().astype(np.float64)
    present = _present(hand_xy)
    mask = present.copy().astype(np.float64)
    if T == 0:
        return filled, mask
    miss = ~present
    run_start = None
    for t in range(T + 1):
        if t < T and miss[t] and run_start is None:
            run_start = t
        if (t == T or not miss[t]) and run_start is not None:
            s, e = run_start, t - 1
            length = e - s + 1
            if length <= max_gap and s > 0 and e < T - 1:
                a = filled[s - 1]; b = filled[e + 1]
                frac = ((np.arange(s, e + 1, dtype=np.float64) - (s - 1))
                        / ((e + 1) - (s - 1)))
                for k in range(s, e + 1):
                    fk = frac[k - s]
                    filled[k] = a + fk * (b - a)
                mask[s:e + 1] = 1.0
            run_start = None
    return filled, mask


def _shoulder_centre_width(pose):
    lsh = pose[:, _L_SHOULDER, :2]
    rsh = pose[:, _R_SHOULDER, :2]
    centre = (lsh + rsh) / 2.0
    width = np.linalg.norm(lsh - rsh, axis=1, keepdims=True) + EPS
    return centre, width


def _resample(feat, n_frames):
    T = feat.shape[0]
    if T == 0:
        return np.zeros((n_frames, FEAT_DIM), dtype=np.float32)
    if T == n_frames:
        return feat
    if T > n_frames:
        idx = np.linspace(0, T - 1, n_frames).round().astype(int)
        return feat[idx]
    pad = np.repeat(feat[-1:], n_frames - T, axis=0)
    return np.concatenate([feat, pad], axis=0)


def extract_features(landmarks, layout='signsage', n_frames=N_FRAMES):
    if layout != 'signsage':
        raise ValueError("sign_features_v2 supports layout='signsage' only (543-array)")
    if landmarks.ndim != 3 or landmarks.shape[2] < 2:
        raise ValueError("landmarks must be (T, N, >=3)")
    T = landmarks.shape[0]
    pose = landmarks[:, 0:33, :]
    lh = _xy(landmarks, _LHAND)
    rh = _xy(landmarks, _RHAND)
    face = landmarks[:, 33:500, :2]

    centre, width = _shoulder_centre_width(pose)
    l_filled, l_mask = _fill_gaps(lh)
    r_filled, r_mask = _fill_gaps(rh)

    def hand_size(h):
        return np.linalg.norm(h[:, _WRIST, :] - h[:, _MIDDLE_TIP, :],
                              axis=1, keepdims=True) + EPS

    l_size = hand_size(l_filled)
    r_size = hand_size(r_filled)

    def hand_part(hfilled, hmask, hsize):
        wrist = hfilled[:, _WRIST, :]
        others = np.delete(hfilled, _WRIST, axis=1)
        wrs = (wrist - centre) / width
        orw = ((others - wrist[:, None, :]) / hsize.reshape(-1, 1, 1))
        wrs = wrs * hmask[:, None]
        orw = orw * hmask[:, None, None]
        return np.concatenate([wrs.reshape(T, -1), orw.reshape(T, -1)], axis=1)

    lhand = hand_part(l_filled, l_mask, l_size)
    rhand = hand_part(r_filled, r_mask, r_size)

    h2f = []
    for hp, hmask in ((l_filled[:, _WRIST, :], l_mask),
                      (l_filled[:, _INDEX_TIP, :], l_mask),
                      (r_filled[:, _WRIST, :], r_mask),
                      (r_filled[:, _INDEX_TIP, :], r_mask)):
        for fidx in _FACE_POINTS:
            h2f.append(((hp - face[:, fidx, :]) / width) * hmask[:, None])
    h2f = np.concatenate(h2f, axis=1)

    def wrist_vel(hfilled, hmask):
        wrist = hfilled[:, _WRIST, :]
        vel = np.zeros_like(wrist)
        if T > 1:
            vel[1:] = wrist[1:] - wrist[:-1]
        return vel * hmask[:, None]

    l_vel = wrist_vel(l_filled, l_mask)
    r_vel = wrist_vel(r_filled, r_mask)

    feat = np.concatenate([lhand, rhand,
                           l_mask.reshape(T, 1), r_mask.reshape(T, 1),
                           h2f, l_vel, r_vel], axis=1)
    assert feat.shape[1] == FEAT_DIM, feat.shape[1]
    feat = _resample(feat, n_frames)
    return feat.astype(np.float32)
'''

STANDALONE_TEST = '''"""Unit test: sign_features_v2.py matches features_v2.py output exactly.

Run:  python test_sign_features_v2.py
"""
import numpy as np
import features_v2 as fv
import sign_features_v2 as sfv


def _random_landmarks(T=37, seed=3):
    rng = np.random.RandomState(seed)
    h = np.zeros((T, 543, 3), dtype=np.float64)
    # random pose shoulders + face + hands with some missing frames
    h[:, 11, 0] = 10.0 + rng.randn(T) * 0.1
    h[:, 12, 0] = 12.0 + rng.randn(T) * 0.1
    h[:, 11, 1] = 0.5; h[:, 12, 1] = 0.5
    h[:, 34, 0] = 11.0 + rng.randn(T) * 0.05
    h[:, 34, 1] = 0.4 + rng.randn(T) * 0.05
    h[:, 185, 0] = 11.0 + rng.randn(T) * 0.05
    h[:, 185, 1] = 0.7 + rng.randn(T) * 0.05
    # hands: present in a subset of frames
    present = rng.rand(T) > 0.3
    for t in range(T):
        if present[t]:
            h[t, 501, 0] = 5.0 + t * 0.2 + rng.randn() * 0.1
            h[t, 501, 1] = 0.5
            h[t, 509, 0] = h[t, 501, 0] + 1.0
            h[t, 509, 1] = 0.5
            h[t, 522, 0] = 18.0 + rng.randn() * 0.1
            h[t, 522, 1] = 0.5
            h[t, 530, 0] = h[t, 522, 0] + 1.0
            h[t, 530, 1] = 0.5
    return h


def main():
    h = _random_landmarks()
    a = fv.extract_features(h, 'signsage')
    b = sfv.extract_features(h, 'signsage')
    assert a.shape == b.shape == (fv.N_FRAMES, fv.FEAT_DIM), (a.shape, b.shape)
    assert np.allclose(a, b, atol=1e-5), np.abs(a - b).max()
    # all-missing-hand case
    h2 = np.zeros((24, 543, 3))
    h2[:, 11, 0] = 10.0; h2[:, 12, 0] = 12.0
    h2[:, 11, 1] = 0.5; h2[:, 12, 1] = 0.5
    h2[:, 34, 0] = 11.0; h2[:, 34, 1] = 0.4
    h2[:, 185, 0] = 11.0; h2[:, 185, 1] = 0.7
    a2 = fv.extract_features(h2, 'signsage')
    b2 = sfv.extract_features(h2, 'signsage')
    assert np.allclose(a2, b2, atol=1e-5)
    assert np.allclose(a2, 0.0)
    assert np.all(a2[:, 84] == 0.0) and np.all(a2[:, 85] == 0.0)
    print("test_sign_features_v2: sign_features_v2 matches features_v2 exactly. OK")


if __name__ == "__main__":
    main()
'''


def build_meta(export_type, label_list, model, li, s2_metrics, latency_ms, golden_input_path, golden_logits, extractor):
    n_classes = len(label_list)
    return {
        "export_type": export_type,                     # "s1-only" or "s1+s2-final-unevaluated"
        "trained_on": {"sessions": ["S1"]} if export_type == "s1-only" else {"sessions": ["S1", "S2"]},
        "label_list": label_list,
        "n_classes": n_classes,
        "model": "1D-CNN (cnn-v2), dropout 0.3, Conv1d(122->32->64, AdaptiveAvgPool, Linear->n_classes)",
        "hyperparameters": {"epochs": EPOCHS, "lr": 1e-3, "batch": 16, "optimizer": "Adam",
                            "dropout": 0.3, "augmentation": "none", "seed": SEED},
        "features": {
            "spec": "features_v2.py / sign_features_v2.py",
            "dim": fv.FEAT_DIM,
            "frames": fv.N_FRAMES,
            "layout": "signsage",
            "normalisation": ("per-frame shoulder-centre + shoulder-width; per hand: wrist relative to shoulder "
                              "centre / shoulder width; other 20 landmarks relative to wrist / hand size "
                              "(hand_size = ||wrist - middle fingertip||); hand-to-face vectors scaled by shoulder "
                              "width. Missing runs <=5 frames interpolated; longer/leading/trailing missing kept as "
                              "zero coordinates with mask=0 and never normalised into shoulder space."),
            "channels": ["left hand 42 (wrist-rel-shoulder + 20 landmarks-rel-wrist)",
                         "right hand 42 (same)", "left_mask, right_mask",
                         "hand-to-face 32 (L/R x wrist/index-tip x nose/chin/mouth/forehead)",
                         "wrist velocity 4"],
        },
        "extractor": extractor,
        "fps_note": ("Features resampled to 60 frames per clip (uniform subsample if T>=60, edge-pad if T<60). "
                     "Effective FPS varies per clip (S1 ~5-12 fps, S2 ~5-12 fps); the model is invariant to this "
                     "because of resampling. Coordinates are NOT mirrored (only the app preview is CSS-mirrored)."),
        "recommended_confidence_threshold": 0.7,
        "threshold_note": ("Calibrated on the S2 cross-session test (train S1, test S2, cnn-v2, 50 clips, 10 CORE "
                           "words, no rest). S2 max-softmax threshold table: 0.5 -> acc 0.612 coverage 98.0%; "
                           "0.7 -> acc 0.659 coverage 88.0% (recommended operating point); "
                           "0.9 -> acc 0.758 coverage 66.0%; argmax -> acc 0.620 coverage 100%. "
                           "Stated as calibrated on S2 only; not validated on held-out data."),
        "s2_test_metrics": s2_metrics,   # None for export_final (FINAL-UNEVALUATED)
        "cpu_latency_ms_per_clip": latency_ms,
        "golden_test_vector": {
            "input_npy": str(golden_input_path),
            "expected_logits": [float(v) for v in golden_logits],
            "expected_logits_labels": label_list,
            "note": "Run features_v2/sign_features_v2 on golden_test_vector.npy then the model; logits must match expected_logits (within float tolerance).",
        },
    }


def main():
    Xtr, ytr, meta_tr, rows_tr = load_features(S1_MANIFEST, set(TRAIN_LABELS))
    Xte, yte, meta_te, rows_te = load_features(S2_MANIFEST, set(CORE10))
    train_labels = TRAIN_LABELS
    label_list = list(train_labels)   # CORE10 + rest

    # extractor info from a recording JSON
    extractor = json.load(open(rows_tr[0]["json"], encoding="utf-8"))["extractor"]

    # ---- (a) export_s1: train on S1 only ----
    model_s1, li_s1 = train_on(Xtr, ytr, train_labels, len(train_labels), seed=SEED)
    lat_s1 = cpu_latency(model_s1, Xtr)
    # golden test vector: first S1 hello clip
    golden_arr = np.load(rows_tr[0]["npy"]).astype(np.float32)
    golden_logits_s1 = logits_for(model_s1, golden_arr)

    d_s1 = ROOT / "export_s1"
    d_s1.mkdir(parents=True, exist_ok=True)
    torch.save(model_s1.state_dict(), d_s1 / "model.pt")
    np.save(d_s1 / "golden_test_vector.npy", golden_arr)
    with open(d_s1 / "golden_expected_logits.json", "w") as f:
        json.dump([float(v) for v in golden_logits_s1], f)
    s2_metrics_s1 = None
    results_path = ROOT / "results_s1train_s2test.json"
    if results_path.exists():
        _r = json.load(open(results_path))
        _c = _r["cnn-v2"]
        s2_metrics_s1 = dict(acc_mean=float(_c["acc_mean"]), acc_std=float(_c["acc_std"]),
                             macroF1_mean=float(_c["mf1_mean"]), macroF1_std=float(_c["mf1_std"]),
                             pooled_acc=float(_c["pooled_acc"]), pooled_macro_f1=float(_c["pooled_macro_f1"]),
                             per_class={w: dict(precision=float(m["precision"]),
                                                  recall=float(m["recall"]),
                                                  f1=float(m["f1"]))
                                        for w, m in _c["per_class"].items()},
                             top10=[[c, i, j] for c, i, j in _c["top10"]],
                             thresholds=_c["thresholds"],
                             note="from results_s1train_s2test.json (cnn-v2, train S1 test S2, 5 seeds)")
    meta_s1 = build_meta("s1-only", label_list, model_s1, li_s1, s2_metrics_s1, lat_s1,
                         d_s1 / "golden_test_vector.npy", golden_logits_s1, extractor)
    with open(d_s1 / "meta.json", "w") as f:
        json.dump(meta_s1, f, indent=2)
    print(f"export_s1: model.pt + meta.json + golden vector. CPU latency {lat_s1:.2f} ms/clip")

    # ---- (b) export_final: train on S1+S2, FINAL-UNEVALUATED ----
    X_all = np.concatenate([Xtr, Xte], axis=0)
    y_all = ytr + yte          # S2 adds no new labels (no rest)
    model_final, li_final = train_on(X_all, y_all, train_labels, len(train_labels), seed=SEED)
    lat_final = cpu_latency(model_final, X_all)
    golden_logits_final = logits_for(model_final, golden_arr)

    d_final = ROOT / "export_final"
    d_final.mkdir(parents=True, exist_ok=True)
    torch.save(model_final.state_dict(), d_final / "model.pt")
    np.save(d_final / "golden_test_vector.npy", golden_arr)
    with open(d_final / "golden_expected_logits.json", "w") as f:
        json.dump([float(v) for v in golden_logits_final], f)
    meta_final = build_meta("s1+s2-final-unevaluated", label_list, model_final, li_final, None,
                            lat_final, d_final / "golden_test_vector.npy", golden_logits_final, extractor)
    meta_final["note"] = ("FINAL-UNEVALUATED: trained on S1+S2 combined. S2 was already used as the held-out test "
                          "set for export_s1, so this model has NO held-out test set. Do not promote without S2-independent validation.")
    with open(d_final / "meta.json", "w") as f:
        json.dump(meta_final, f, indent=2)
    print(f"export_final: model.pt + meta.json + golden vector (FINAL-UNEVALUATED). CPU latency {lat_final:.2f} ms/clip")

    # ---- standalone sign_features_v2.py + unit test ----
    (ROOT / "sign_features_v2.py").write_text(STANDALONE_HEADER)
    (ROOT / "test_sign_features_v2.py").write_text(STANDALONE_TEST)
    print(f"wrote {ROOT / 'sign_features_v2.py'} and {ROOT / 'test_sign_features_v2.py'}")
    print("All exports PRELIMINARY; export_final is FINAL-UNEVALUATED (no held-out test).")


if __name__ == "__main__":
    main()
