"""Evaluate whether adding the S4 reference videos helps the cnn-v3 8+rest model.

Compare (5 seeds each, same recipe/hyperparameters as eval_final8.py):
  baseline:  train S1+S2          -> test S3  (8 words)
  +S4:       train S1+S2+S4video  -> test S3  (8 words; S4 NOT in the test set)

S4 is the external ISL reference videos (one clip per word, mixed signers).
Success criterion (pre-registered): adopt S4 only if +S4 is not worse than
baseline by more than 1 std on S3.
"""
import csv, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import features_v3 as fv
from eval_v3 import TinyCNN1D, train_model_v3, predict_torch
from eval_final8 import word_metrics, threshold_table, train_cnn, predict_cnn, load_manifest

EIGHT = ["hello", "me", "you", "yes", "no", "help", "water", "where"]
REST = "rest"
TRAIN9 = EIGHT + [REST]
TEST8 = EIGHT
N_SEEDS = 5
EPOCHS = 60
OUT = ROOT / "results_s4_videos.json"


def load_features(manifest_path, wanted_labels):
    rows = load_manifest(manifest_path)
    X, y = [], []
    for r in rows:
        lab = r["label"]
        if wanted_labels is not None and lab not in wanted_labels:
            continue
        arr = np.load(r["npy"]).astype(np.float32)
        f = fv.extract_features(arr, layout="signsage")
        assert f.shape == (fv.N_FRAMES, fv.FEAT_DIM), f.shape
        X.append(f); y.append(lab)
    return np.stack(X), y


def run_config(name, Xtr, ytr, Xte, yte):
    per_seed = []
    per_seed_preds = []
    for seed in range(N_SEEDS):
        model, li = train_cnn(Xtr, ytr, TRAIN9, 9, seed)
        pred_te, probs_te = predict_cnn(model, Xte)
        acc, mf1, pc, cm, top, max_probs = word_metrics(yte, pred_te, probs_te, TRAIN9, TEST8)
        per_seed.append(dict(seed=seed, acc=float(acc), macro_f1=float(mf1),
                             per_class=pc, top10=[[c, i, j] for c, i, j in top]))
        per_seed_preds.append((pred_te, probs_te, max_probs))
        print(f"  seed {seed}: acc={acc:.3f} macroF1={mf1:.3f}")
    accs = [r["acc"] for r in per_seed]; mf1s = [r["macro_f1"] for r in per_seed]
    pc_mean = {}
    for w in TEST8:
        fs = [r["per_class"][w] for r in per_seed]
        pc_mean[w] = dict(precision=float(np.mean([f["precision"] for f in fs])),
                          recall=float(np.mean([f["recall"] for f in fs])),
                          f1=float(np.mean([f["f1"] for f in fs])))
    print(f"\n{name}: acc={np.mean(accs):.3f}±{np.std(accs):.3f} "
          f"macroF1={np.mean(mf1s):.3f}±{np.std(mf1s):.3f}")
    print("per-class F1:", ", ".join(f"{w}={m['f1']:.2f}"
          for w, m in sorted(pc_mean.items(), key=lambda kv: kv[1]['f1'])))
    yte0, pred0, probs0, mp0 = None, None, None, None
    # threshold table on seed 0
    model0, li0 = train_cnn(Xtr, ytr, TRAIN9, 9, 0)
    # reuse seed-0 already computed above via re-running (cheap)
    pred0, probs0 = predict_cnn(model0, Xte)
    # recompute max_probs
    import torch.nn.functional as F_  # not needed
    max_probs0 = probs0.max(axis=1)
    thr_rows = threshold_table(max_probs0, pred0, yte, TRAIN9, TEST8)
    return dict(acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
                mf1_mean=float(np.mean(mf1s)), mf1_std=float(np.std(mf1s)),
                per_seed=[dict(seed=r["seed"], acc=float(r["acc"]),
                               macro_f1=float(r["macro_f1"])) for r in per_seed],
                per_class=pc_mean, thresholds=thr_rows, n_test=len(Xte))


def main():
    Xtr_s1, ytr_s1 = load_features(ROOT / "data" / "s1_converted" / "manifest.csv", set(TRAIN9))
    Xtr_s2, ytr_s2 = load_features(ROOT / "data" / "s2_converted" / "manifest.csv", set(TEST8))
    Xte, yte = load_features(ROOT / "data" / "s3_converted" / "manifest.csv", set(TEST8))
    Xtr_s4, ytr_s4 = load_features(ROOT / "data" / "s4_converted" / "manifest.csv", set(TRAIN9))

    X_s1s2 = np.concatenate([Xtr_s1, Xtr_s2], axis=0)
    y_s1s2 = ytr_s1 + ytr_s2
    X_with_s4 = np.concatenate([Xtr_s1, Xtr_s2, Xtr_s4], axis=0)
    y_with_s4 = ytr_s1 + ytr_s2 + ytr_s4

    print(f"S1+S2 train: {len(X_s1s2)} clips | +S4: {len(X_with_s4)} clips "
          f"(S4={len(Xtr_s4)}) | S3 test: {len(Xte)} clips")

    t_all = time.time()
    print("\n===== baseline: train S1+S2 -> test S3 =====")
    base = run_config("S1+S2 -> S3", X_s1s2, y_s1s2, Xte, yte)

    print("\n===== +S4 videos: train S1+S2+S4 -> test S3 =====")
    s4 = run_config("S1+S2+S4 -> S3", X_with_s4, y_with_s4, Xte, yte)

    out = dict(
        baseline_s1s2=base, with_s4_videos=s4,
        s4_clips=[dict(label=r["label"], frames=int(r["frames"]),
                       left_det=1 - float(r["left_miss_frac"]),
                       right_det=1 - float(r["right_miss_frac"]))
                  for r in load_manifest(ROOT / "data" / "s4_converted" / "manifest.csv")],
        note=("S4 = 9 external ISL reference videos (1 clip/word, mixed signers/sources). "
              "Test set S3 is held out from both configs. Adopt S4 only if not worse "
              "than baseline by >1 std."),
    )
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT} (total {time.time() - t_all:.1f}s)")


if __name__ == "__main__":
    main()
