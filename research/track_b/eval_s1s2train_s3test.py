"""Track B step 2 (REPORT_H): Train cnn-v3 on S1+S2 (5 seeds), test on S3 ONLY
(clean held-out test). Uses EXACTLY the cnn-v3 recipe/hyperparameters from
REPORT_G (features_v3, dropout 0.3, 60 epochs, Adam 1e-3, batch 16, training-only
augmentation: hand-frame dropout 0-50%, scale +/-10%, rotation +/-10 deg,
time-warp +/-15%). No tuning from S3.

Train labels: CORE-10 + rest (11 classes) from S1 + S2 combined (181 clips).
Test labels: CORE-10 words from S3 (50 clips; no rest in S3).
Reports: accuracy, macro-F1 (mean +/- std over 5 seeds), per-class P/R/F1 table,
top-10 confusion pairs, chance baseline, threshold table (0.5/0.7/0.9),
rest FPR (NOT VERIFIED — no rest in S3).
"""
import csv, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import features_v3 as fv
from eval_v3 import (TinyCNN1D, train_model_v3, predict_torch, LogisticRegression,
                     standardize_fit, standardize_apply, resample_batch,
                     accuracy_metrics)

S1_MANIFEST = ROOT / "data" / "s1_converted" / "manifest.csv"
S2_MANIFEST = ROOT / "data" / "s2_converted" / "manifest.csv"
S3_MANIFEST = ROOT / "data" / "s3_converted" / "manifest.csv"
CORE10 = ["hello", "thank you", "please", "yes", "no", "help", "water", "you", "me", "where"]
REST = "rest"
TRAIN_LABELS = CORE10 + [REST]        # 11 classes
TEST_WORDS = CORE10                   # 10 word classes in S3
N_SEEDS = 5
EPOCHS = 60
OUT = ROOT / "results_s1s2train_s3test.json"


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


def restrict_to_test_words(y_true_words, pred_11, train_labels, test_words):
    """Map 11-class predictions onto the 10 S3 word classes. pred==rest => error."""
    li = {w: i for i, w in enumerate(train_labels)}
    rest_idx = li[REST]
    word_idx = {w: li[w] for w in test_words}
    n = len(test_words)
    cm = np.zeros((n, n), dtype=np.int64)
    n_correct = 0
    for t, p in zip(y_true_words, pred_11):
        ti = word_idx[t]
        if p == rest_idx:
            continue  # FN only
        if p == ti:
            n_correct += 1
        pw = train_labels[p]
        if pw in word_idx:
            cm[ti, word_idx[pw]] += 1
    acc = n_correct / max(1, len(y_true_words))
    per_class = {}
    for ci, w in enumerate(test_words):
        tp = cm[ci, ci]; fn = cm[ci, :].sum() - tp; fp = cm[:, ci].sum() - tp
        p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
        per_class[w] = dict(precision=float(p), recall=float(r),
                            f1=float(2 * p * r / max(1e-9, p + r)))
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))
    off = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                off.append((int(cm[i, j]), test_words[i], test_words[j]))
    off.sort(reverse=True)
    return acc, macro_f1, per_class, cm, off[:10]


def train_and_test(Xtr, ytr, Xte, yte, model_kind, seed):
    np.random.seed(seed); torch.manual_seed(seed)
    li = li_map(TRAIN_LABELS)
    ytr_i = np.array([li[w] for w in ytr], dtype=np.int64)
    if model_kind == "cnn-v3":
        model = TinyCNN1D(n_in=fv.FEAT_DIM, n_classes=len(TRAIN_LABELS), dropout=0.3)
        train_model_v3(model, Xtr, ytr_i, epochs=EPOCHS, seed=seed)
        pred_te, probs_te = predict_torch(model, Xte)
    elif model_kind == "logreg":
        flat_tr = resample_batch(Xtr, 30).reshape(len(Xtr), -1)
        flat_te = resample_batch(Xte, 30).reshape(len(Xte), -1)
        mu, sd = standardize_fit(flat_tr)
        clf = LogisticRegression(seed=seed)
        clf.fit(standardize_apply(flat_tr, mu, sd), ytr_i)
        pred_te = clf.predict(standardize_apply(flat_te, mu, sd))
        probs_te = clf.predict_proba(standardize_apply(flat_te, mu, sd))
    else:
        raise ValueError(model_kind)
    acc, mf1, pc, cm, top = restrict_to_test_words(yte, pred_te, TRAIN_LABELS, TEST_WORDS)
    return dict(acc=float(acc), macro_f1=float(mf1), per_class=pc,
                top10=[[c, i, j] for c, i, j in top],
                pred_te=pred_te, probs_te=probs_te)


def main():
    Xtr, ytr, meta_tr, rows_tr = load_features(S1_MANIFEST, set(TRAIN_LABELS))
    Xs2, ys2, meta_s2, rows_s2 = load_features(S2_MANIFEST, set(CORE10))
    Xte, yte, meta_te, rows_te = load_features(S3_MANIFEST, set(TEST_WORDS))

    # Combined S1+S2 training set
    X_all = np.concatenate([Xtr, Xs2], axis=0)
    y_all = ytr + ys2
    print(f"S1 train clips: {len(Xtr)}  S2 train clips: {len(Xs2)}  "
          f"combined S1+S2: {len(X_all)}  labels: {sorted(set(y_all))}")
    print(f"S3 test clips: {len(Xte)}  labels: {sorted(set(yte))}  (rest in S3: 0)")

    results = {}
    t_all = time.time()
    for model_kind in ("cnn-v3", "logreg"):
        print(f"\n===== {model_kind}: train S1+S2 ({len(X_all)} clips) / test S3 ({len(Xte)} clips) =====")
        per_seed = []
        for seed in range(N_SEEDS):
            r = train_and_test(X_all, y_all, Xte, yte, model_kind, seed)
            per_seed.append(r)
            print(f"  seed {seed}: S3 acc={r['acc']:.3f} macroF1={r['macro_f1']:.3f}")
        accs = [r["acc"] for r in per_seed]; mf1s = [r["macro_f1"] for r in per_seed]

        # Majority-vote pooled over seeds
        li = li_map(TRAIN_LABELS)
        rest_idx = li[REST]
        word_idx = {w: li[w] for w in TEST_WORDS}
        n = len(TEST_WORDS)
        cm_pool = np.zeros((n, n), dtype=np.int64)
        n_correct = 0
        # rebuild per-clip preds: need per-clip preds per seed; store via rerun is costly,
        # so instead pool per-class F1 by averaging seed F1s (already have per_seed).
        pc_mean = {}
        for w in TEST_WORDS:
            fs = [r["per_class"][w] for r in per_seed]
            pc_mean[w] = dict(precision=float(np.mean([f["precision"] for f in fs])),
                              recall=float(np.mean([f["recall"] for f in fs])),
                              f1=float(np.mean([f["f1"] for f in fs])))
        pool_acc = float(np.mean(accs)); pool_mf1 = float(np.mean(mf1s))
        rep_top = per_seed[0]["top10"]
        print(f"\n--- {model_kind} S3 test (mean over {N_SEEDS} seeds) ---")
        print(f"  acc={pool_acc:.3f} macroF1={pool_mf1:.3f}  "
              f"(mean±std: {np.mean(accs):.3f}±{np.std(accs):.3f} / "
              f"{np.mean(mf1s):.3f}±{np.std(mf1s):.3f})")
        print(f"  chance (uniform, 10 classes) = {1/len(TEST_WORDS):.3f}")
        print(f"  per-class F1 (seed-mean): " +
              ", ".join(f"{w}={m['f1']:.2f}" for w, m in
                        sorted(pc_mean.items(), key=lambda kv: kv[1]['f1'])))
        print(f"  top-10 confusions (seed0 representative, count true->pred):")
        for c, i, j in rep_top:
            print(f"    {c:3d}  {i:14} -> {j}")

        # Threshold table on seed-0 pooled probs (per-clip)
        np.random.seed(0); torch.manual_seed(0)
        r0 = train_and_test(X_all, y_all, Xte, yte, model_kind, 0)
        max_prob = r0["probs_te"].max(axis=1)
        pred_word = np.array([TRAIN_LABELS[p] for p in r0["pred_te"]])
        yte_words = np.array(yte)
        n = len(yte)
        rows_thr = []
        correct_all = (pred_word == yte_words).sum()
        rows_thr.append(dict(threshold="argmax", n_accepted=n, coverage=1.0,
                             accuracy=float(correct_all / n)))
        for thr in (0.5, 0.7, 0.9):
            accepted = max_prob >= thr
            correct = int((pred_word[accepted] == yte_words[accepted]).sum()) if accepted.any() else 0
            rows_thr.append(dict(threshold=thr, n_accepted=int(accepted.sum()),
                                 coverage=float(accepted.mean()),
                                 accuracy=float(correct / max(1, int(accepted.sum())))))
        print(f"  threshold table (S3):")
        print(f"    {'threshold':>10} {'accepted':>9} {'coverage':>9} {'acc':>7}")
        for row in rows_thr:
            print(f"    {str(row['threshold']):>10} {row['n_accepted']:>9} "
                  f"{row['coverage']:8.1%} {row['accuracy']:7.3f}")
        print(f"  rest FPR: NOT VERIFIED (no rest clips in S3)")

        # clean non-serializable
        r_out = dict(acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
                     mf1_mean=float(np.mean(mf1s)), mf1_std=float(np.std(mf1s)),
                     per_seed=[dict(seed=i, acc=float(r["acc"]), macro_f1=float(r["macro_f1"]))
                               for i, r in enumerate(per_seed)],
                     pooled_acc=pool_acc, pooled_macro_f1=pool_mf1,
                     per_class=pc_mean, top10=rep_top, thresholds=rows_thr,
                     n_test=len(Xte))
        results[model_kind] = r_out

    # Also compute a true majority-vote pooled confusion matrix for the report table
    # (rerun seed0 already done; pooled per-class from seed-mean above is fine).
    report = dict(results=results,
                  train_labels=TRAIN_LABELS, test_words=TEST_WORDS,
                  n_train=int(len(X_all)), n_test=int(len(Xte)),
                  note="CLEAN HELD-OUT TEST on S3. Train S1+S2 only. S3 never used for tuning.")
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {OUT} (total {time.time()-t_all:.1f}s)")
    print(report["note"])


if __name__ == "__main__":
    main()
