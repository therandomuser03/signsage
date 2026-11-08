"""Track B step 4: export cnn-v3 models.

(a) research/track_b/export_v3_s1/  : cnn-v3 trained on S1 ONLY (the evaluated model).
(b) research/track_b/export_v3_final/: cnn-v3 trained on S1+S2, labelled
    FINAL-UNEVALUATED (S2 was already used as the dev set for (a), so no held-out
    test exists for this combined model).

For each: model.pt (state dict), meta.json (labels, v3 feature spec, normalisation
constants, extractor name/version/options, fps note, recommended confidence
threshold derived from the S2 dev table, stated as calibrated on S2),
golden_test_vector.npy ((T,543,3)), golden_expected_logits.json.

Also reports CPU latency per clip. Uses EXACTLY the cnn-v3 setup from REPORT_G
(dropout 0.3, 60 epochs, Adam 1e-3, batch 16) WITH training augmentation.
Nothing integrated into the app. Offline, CPU.
"""
import csv, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import features_v3 as fv
from eval_v3 import (TinyCNN1D, train_model_v3, predict_torch,
                     TRAIN_LABELS, S2_WORDS, REST, S1_MANIFEST, S2_MANIFEST)

SEED = 0
EPOCHS = 60


def load_manifest(p):
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


def li_map(labels):
    return {w: i for i, w in enumerate(labels)}


def train_on(X, y, labels, n_classes, seed=SEED):
    np.random.seed(seed); torch.manual_seed(seed)
    li = li_map(labels)
    yi = np.array([li[w] for w in y], dtype=np.int64)
    model = TinyCNN1D(n_in=fv.FEAT_DIM, n_classes=n_classes, dropout=0.3)
    train_model_v3(model, X, yi, epochs=EPOCHS, seed=seed)
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
    return float(np.mean(times)) * 1000.0


def logits_for(model, arr_543):
    f = fv.extract_features(arr_543, layout="signsage")
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(f[None]).float())
    return logits[0].numpy()


META_HEADER = """sign_features_v3 / features_v3 (122 dims x 60 frames, signsage layout)."""


def build_meta(export_type, label_list, model, li, s2_dev_metrics, latency_ms,
               golden_input_path, golden_logits, extractor):
    return {
        "export_type": export_type,   # "s1-only" or "s1+s2-final-unevaluated"
        "trained_on": {"sessions": ["S1"]} if export_type == "s1-only"
                      else {"sessions": ["S1", "S2"]},
        "label_list": label_list,
        "n_classes": len(label_list),
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
                     "T>=60, edge-pad if T<60). Effective FPS varies per clip; the model "
                     "is invariant to this via resampling. Coordinates are NOT mirrored "
                     "(only the app preview is CSS-mirrored); canonicalisation mirroring "
                     "is applied inside features_v3."),
        "recommended_confidence_threshold": 0.7,
        "threshold_note": ("Calibrated on the S2 DEV set (train S1, test S2, cnn-v3, "
                           "50 clips, 10 CORE words, no rest). S2 max-softmax threshold "
                           "table: 0.5 -> acc {t5} coverage {c5}; 0.7 -> acc {t7} "
                           "coverage {c7} (recommended operating point); 0.9 -> acc "
                           "{t9} coverage {c9}; argmax -> acc {ta} coverage 100%. "
                           "Stated as calibrated on S2 only; S2 was used to diagnose the "
                           "handedness problem, so this is NOT a clean held-out test."),
        "s2_dev_metrics": s2_dev_metrics,   # None for export_final
        "cpu_latency_ms_per_clip": latency_ms,
        "golden_test_vector": {
            "input_npy": str(golden_input_path),
            "expected_logits": [float(v) for v in golden_logits],
            "expected_logits_labels": label_list,
            "note": "Run features_v3/sign_features_v3 on golden_test_vector.npy then the model; logits must match expected_logits (within float tolerance).",
        },
    }


def main():
    Xtr, ytr, meta_tr, rows_tr = load_features(S1_MANIFEST, set(TRAIN_LABELS))
    Xte, yte, meta_te, rows_te = load_features(S2_MANIFEST, set(S2_WORDS))
    train_labels = list(TRAIN_LABELS)
    label_list = list(train_labels)
    extractor = json.load(open(rows_tr[0]["json"], encoding="utf-8"))["extractor"]

    # ---- (a) export_v3_s1: train on S1 only ----
    model_s1, li_s1 = train_on(Xtr, ytr, train_labels, len(train_labels), seed=SEED)
    lat_s1 = cpu_latency(model_s1, Xtr)
    golden_arr = np.load(rows_tr[0]["npy"]).astype(np.float32)
    golden_logits_s1 = logits_for(model_s1, golden_arr)

    d_s1 = ROOT / "export_v3_s1"
    d_s1.mkdir(parents=True, exist_ok=True)
    torch.save(model_s1.state_dict(), d_s1 / "model.pt")
    np.save(d_s1 / "golden_test_vector.npy", golden_arr)
    with open(d_s1 / "golden_expected_logits.json", "w") as f:
        json.dump([float(v) for v in golden_logits_s1], f)

    # read S2 dev metrics for the threshold
    results_path = ROOT / "results_v3.json"
    s2_dev_metrics = None
    t5 = c5 = t7 = c7 = t9 = c9 = ta = None
    if results_path.exists():
        r = json.load(open(results_path))
        c = r["results"]["cnn-v3"]
        s2_dev_metrics = dict(acc_mean=float(c["s2_dev_acc_mean"]),
                              acc_std=float(c["s2_dev_acc_std"]),
                              mf1_mean=float(c["s2_dev_mf1_mean"]),
                              mf1_std=float(c["s2_dev_mf1_std"]),
                              pooled_acc=float(c["s2_dev_pooled_acc"]),
                              pooled_mf1=float(c["s2_dev_pooled_mf1"]),
                              per_class={w: dict(precision=float(m["precision"]),
                                                   recall=float(m["recall"]),
                                                   f1=float(m["f1"]))
                                         for w, m in c["s2_per_class"].items()},
                              top10=[[cc, ii, jj] for cc, ii, jj in c["s2_top10"]],
                              thresholds=c["s2_thresholds"],
                              note="from results_v3.json (cnn-v3, train S1 test S2 dev, 5 seeds)")
        thr = {row["threshold"]: row for row in c["s2_thresholds"]}
        for key in (0.5, 0.7, 0.9, "argmax"):
            if key in thr:
                row = thr[key]
                if key == 0.5: t5, c5 = row["accuracy"], row["coverage"]
                if key == 0.7: t7, c7 = row["accuracy"], row["coverage"]
                if key == 0.9: t9, c9 = row["accuracy"], row["coverage"]
                if key == "argmax": ta = row["accuracy"]

    meta_s1 = build_meta("s1-only", label_list, model_s1, li_s1, s2_dev_metrics,
                         lat_s1, d_s1 / "golden_test_vector.npy", golden_logits_s1,
                         extractor)
    meta_s1["recommended_confidence_threshold"] = 0.7
    meta_s1["threshold_note"] = meta_s1["threshold_note"].format(
        t5=(t5 if t5 is not None else 0.0), c5=(c5 if c5 is not None else 0.0),
        t7=(t7 if t7 is not None else 0.0), c7=(c7 if c7 is not None else 0.0),
        t9=(t9 if t9 is not None else 0.0), c9=(c9 if c9 is not None else 0.0),
        ta=(ta if ta is not None else 0.0))
    with open(d_s1 / "meta.json", "w") as f:
        json.dump(meta_s1, f, indent=2)
    print(f"export_v3_s1: model.pt + meta.json + golden vector. CPU latency {lat_s1:.2f} ms/clip")

    # ---- (b) export_v3_final: train on S1+S2, FINAL-UNEVALUATED ----
    X_all = np.concatenate([Xtr, Xte], axis=0)
    y_all = ytr + yte
    model_final, li_final = train_on(X_all, y_all, train_labels, len(train_labels), seed=SEED)
    lat_final = cpu_latency(model_final, X_all)
    golden_logits_final = logits_for(model_final, golden_arr)

    d_final = ROOT / "export_v3_final"
    d_final.mkdir(parents=True, exist_ok=True)
    torch.save(model_final.state_dict(), d_final / "model.pt")
    np.save(d_final / "golden_test_vector.npy", golden_arr)
    with open(d_final / "golden_expected_logits.json", "w") as f:
        json.dump([float(v) for v in golden_logits_final], f)
    meta_final = build_meta("s1+s2-final-unevaluated", label_list, model_final, li_final,
                            None, lat_final, d_final / "golden_test_vector.npy",
                            golden_logits_final, extractor)
    meta_final["recommended_confidence_threshold"] = 0.7
    meta_final["threshold_note"] = (meta_final["threshold_note"]
                                    .format(t5=(t5 if t5 is not None else 0.0),
                                            c5=(c5 if c5 is not None else 0.0),
                                            t7=(t7 if t7 is not None else 0.0),
                                            c7=(c7 if c7 is not None else 0.0),
                                            t9=(t9 if t9 is not None else 0.0),
                                            c9=(c9 if c9 is not None else 0.0),
                                            ta=(ta if ta is not None else 0.0))
                                    + " This combined model is FINAL-UNEVALUATED (no held-out test).")
    meta_final["note"] = ("FINAL-UNEVALUATED: trained on S1+S2 combined. S2 was already "
                          "used as the dev set for export_v3_s1, so this model has NO "
                          "held-out test set. Do not promote without S2-independent validation.")
    with open(d_final / "meta.json", "w") as f:
        json.dump(meta_final, f, indent=2)
    print(f"export_v3_final: model.pt + meta.json + golden vector (FINAL-UNEVALUATED). "
          f"CPU latency {lat_final:.2f} ms/clip")
    print("All exports PRELIMINARY; export_v3_final is FINAL-UNEVALUATED.")


if __name__ == "__main__":
    main()
