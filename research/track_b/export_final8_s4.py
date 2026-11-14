"""Train cnn-v3 on S1+S2+S3+S4 (8 words + rest), export FINAL-UNEVALUATED model.

Same recipe/hyperparameters as export_final8.py + the 9 S4 external reference
videos (research/track_b/data/s4_converted, convert_videos.py).

Held-out evidence (eval_s4_videos.py): train S1+S2+S4 -> test S3 gives
acc 0.980 ± 0.029 / macroF1 0.979 ± 0.030 over 5 seeds (baseline S1+S2 -> S3:
0.935 ± 0.012 / 0.933 ± 0.013).

Output: research/track_b/export_final8_s4/
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import features_v3 as fv
from eval_v3 import TinyCNN1D, train_model_v3

EIGHT = ["hello", "me", "you", "yes", "no", "help", "water", "where"]
REST = "rest"
LABELS = EIGHT + [REST]          # 9 classes
SEED = 42
EPOCHS = 60
EXPORT_DIR = ROOT / "export_final8_s4"


def load_manifest(p):
    import csv
    return list(csv.DictReader(open(p, encoding="utf-8")))


def load_features(manifest_path, wanted_labels):
    rows = load_manifest(manifest_path)
    X, y, meta = [], [], []
    for r in rows:
        lab = r["label"]
        if wanted_labels is not None and lab not in wanted_labels:
            continue
        arr = np.load(r["npy"]).astype(np.float32)
        f = fv.extract_features(arr, layout="signsage")
        assert f.shape == (fv.N_FRAMES, fv.FEAT_DIM), f.shape
        X.append(f); y.append(lab); meta.append(r)
    return np.stack(X), y, meta, rows


def train_final(X, y, labels, n_classes, seed):
    np.random.seed(seed); torch.manual_seed(seed)
    li = {w: i for i, w in enumerate(labels)}
    yi = np.array([li[w] for w in y], dtype=np.int64)
    model = TinyCNN1D(n_in=fv.FEAT_DIM, n_classes=n_classes, dropout=0.3)
    train_model_v3(model, X, yi, epochs=EPOCHS, seed=seed)
    return model, li


def logits_for(model, arr_543):
    f = fv.extract_features(arr_543, layout="signsage")
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(f[None]).float())
    return logits[0].numpy()


def cpu_latency(model, X, n_repeats=3):
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            for i in range(len(X)):
                t0 = time.perf_counter()
                _ = model(torch.from_numpy(X[i:i + 1]).float())
                times.append(time.perf_counter() - t0)
    return float(np.mean(times)) * 1000.0


def main():
    Xtr, ytr, meta_tr, rows_tr = load_features(ROOT / "data" / "s1_converted" / "manifest.csv", set(LABELS))
    Xs2, ys2, _, rows_s2 = load_features(ROOT / "data" / "s2_converted" / "manifest.csv", set(EIGHT))
    Xs3, ys3, _, rows_s3 = load_features(ROOT / "data" / "s3_converted" / "manifest.csv", set(EIGHT))
    Xs4, ys4, _, rows_s4 = load_features(ROOT / "data" / "s4_converted" / "manifest.csv", set(LABELS))

    X_all = np.concatenate([Xtr, Xs2, Xs3, Xs4], axis=0)
    y_all = ytr + ys2 + ys3 + ys4
    print(f"FINAL train S1+S2+S3+S4 (8 words + rest): {len(X_all)} clips "
          f"(S1={len(Xtr)}, S2={len(Xs2)}, S3={len(Xs3)}, S4={len(Xs4)})")

    model, li = train_final(X_all, y_all, LABELS, len(LABELS), seed=SEED)
    lat = cpu_latency(model, X_all)

    # Golden vector: one S4 reference clip (never in same-session quality view,
    # but final model is FINAL-UNEVALUATED so pipeline verification is the point).
    golden_arr = np.load(rows_s4[0]["npy"]).astype(np.float32)
    golden_logits = logits_for(model, golden_arr)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), EXPORT_DIR / "model.pt")
    np.save(EXPORT_DIR / "golden_test_vector.npy", golden_arr)
    with open(EXPORT_DIR / "golden_expected_logits.json", "w") as f:
        json.dump([float(v) for v in golden_logits], f)

    extractor = json.load(open(rows_s4[0]["json"], encoding="utf-8"))["extractor"]
    meta = {
        "model_version": "word_v3_8w_s4ref_s123",
        "status": "FINAL-UNEVALUATED",
        "note": (
            "Trained on S1+S2+S3 (SignSage captures) plus S4, the external ISL "
            "reference videos (one clip per word, mixed signer(s) and sources). "
            "No held-out data exists for this exact model. Expected quality "
            "estimated by train S1+S2+S4 / test S3 (5 seeds): acc 0.980 +/- 0.029, "
            "macro-F1 0.979 +/- 0.030 (see results_s4_videos.json); baseline "
            "S1+S2 / S3 was 0.935 +/- 0.012."
        ),
        "trained_on": {
            "sessions": ["S1", "S2", "S3", "S4"],
            "s4_note": ("S4 = external reference videos (hello, help, me, no, rest, "
                        "water, where, yes, you), 1 clip/word, converted with the same "
                        "tasks-vision landmarker models as the app."),
        },
        "label_list": LABELS,
        "n_classes": len(LABELS),
        "excluded_words": ["thank you", "please"],
        "exclusion_reason": "Weak on S3 (F1 0.52 and 0.44 in the 10-class model).",
        "signer_note": "adapted to one signer; not signer-independent",
        "model": ("1D-CNN (cnn-v3), dropout 0.3: Conv1d(122->32->64, ReLU, Dropout) "
                  "x2, AdaptiveAvgPool1d, Linear->n_classes. Trained WITH frame-level "
                  "augmentation (hand-frame dropout 0-50%, scale +/-10%, rotation "
                  "+/-10 deg, time-warp +/-15%)."),
        "hyperparameters": {"epochs": EPOCHS, "lr": 1e-3, "batch": 16,
                            "optimizer": "Adam", "dropout": 0.3, "seed": SEED},
        "features": {
            "spec": "features_v3.py / sign_features_v3.py",
            "dim": fv.FEAT_DIM,
            "frames": fv.N_FRAMES,
            "layout": "signsage (pose 0-32 | face 33-500 | hand slots 501-542)",
            "normalisation": ("MediaPipe Left/Right labels IGNORED. Hands assigned to "
                              "two stable physical slots by temporal tracking "
                              "(nearest wrist to previous frame; two hands in one frame "
                              "ordered by image x, then continuity). ACTIVE slot = hand "
                              "with larger total wrist travel (tie: more frames present); "
                              "other = SECONDARY. Per-frame: wrist - shoulder_centre / "
                              "shoulder_width; other 20 landmarks - wrist / hand_size "
                              "(hand_size = ||wrist - middle fingertip||). If the active "
                              "hand's mean wrist x is image-left of the shoulder "
                              "midpoint, all x coords (hands + face) are mirrored about "
                              "the per-frame shoulder midpoint so the active hand is "
                              "right-hand-active. Missing runs <=5 frames interpolated; "
                              "longer/leading/trailing missing kept as zero coords with "
                              "mask=0, never normalised into shoulder space."),
            "channels": ["active hand 42", "secondary hand 42", "active_mask",
                         "secondary_mask", "hand-to-face 32", "wrist velocity 4"],
        },
        "extractor": extractor,
        "fps_note": ("Features resampled to 60 frames per clip (uniform subsample if "
                     "T>=60, edge-pad if T<60). S4 reference videos are 25-30 fps; app "
                     "captures are ~10 fps; the model is invariant to this via resampling. "
                     "Coordinates are NOT mirrored; canonicalisation mirroring is applied "
                     "inside features_v3."),
        "recommended_confidence_threshold": 0.7,
        "threshold_note": ("Threshold 0.7 kept from REPORT_G (calibrated on S2 dev); "
                           "not re-selected from S3. Train S1+S2+S4 / test S3: acc 0.980 "
                           "at 100% coverage; threshold table in results_s4_videos.json."),
        "s3_heldout_metrics": {
            "acc_mean": 0.980, "acc_std": 0.029,
            "macroF1_mean": 0.979, "macroF1_std": 0.030,
            "n_test": 40,
            "note": "from results_s4_videos.json (train S1+S2+S4, test S3, 5 seeds)",
        },
        "cpu_latency_ms_per_clip": lat,
        "golden_test_vector": {
            "input_npy": str(EXPORT_DIR / "golden_test_vector.npy"),
            "expected_logits": [float(v) for v in golden_logits],
            "expected_logits_labels": LABELS,
            "note": "Run features_v3/sign_features_v3 on golden_test_vector.npy then "
                    "the model; logits must match expected_logits (within float tolerance).",
        },
    }
    with open(EXPORT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Verify golden vector reproduces expected logits from the export folder
    model2 = TinyCNN1D(n_in=fv.FEAT_DIM, n_classes=len(LABELS), dropout=0.3)
    model2.load_state_dict(torch.load(EXPORT_DIR / "model.pt"))
    model2.eval()
    arr2 = np.load(EXPORT_DIR / "golden_test_vector.npy")
    f2 = fv.extract_features(arr2, layout="signsage")
    with torch.no_grad():
        logits2 = model2(torch.from_numpy(f2[None]).float())[0].numpy()
    exp2 = np.array(json.load(open(EXPORT_DIR / "golden_expected_logits.json")))
    print(f"Golden vector reproduces expected logits: "
          f"{bool(np.allclose(logits2, exp2, atol=1e-4))} (maxdiff={np.abs(logits2 - exp2).max():.2e})")
    print(f"Predicted label: {LABELS[int(logits2.argmax())]}")
    print(f"CPU latency per clip: {lat:.2f} ms")
    print(f"Wrote export to {EXPORT_DIR}")


if __name__ == "__main__":
    main()
