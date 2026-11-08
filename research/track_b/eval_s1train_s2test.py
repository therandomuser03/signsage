"""Track B step 3: cross-session test — train cnn-v2 (+logreg baseline) on S1 ONLY,
test on S2 ONLY. Uses EXACTLY the cnn-v2 setup/hyperparameters from REPORT_E
(dropout 0.3, 60 epochs, Adam 1e-3, batch 16, no augmentation, same seed handling).

Train: S1 CORE10 + rest (11 classes). Test: S2 labels present in S2 (the 10 CORE
words; no rest in S2). Report mean +/- std over 5 seeds of S2 test metrics.
No tuning based on S2. Offline, CPU. Nothing exported here (step 5 handles export).
"""
import csv, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import features_v2 as fv
from eval_v2 import (TinyCNN1D, train_torch, predict_torch, LogisticRegression,
                     standardize_fit, standardize_apply, resample_batch,
                     accuracy_metrics, threshold_table)

S1_MANIFEST = ROOT / "data" / "s1_converted" / "manifest.csv"
S2_MANIFEST = ROOT / "data" / "s2_converted" / "manifest.csv"
CORE10 = ["hello", "thank you", "please", "yes", "no", "help", "water", "you", "me", "where"]
REST = "rest"
TRAIN_LABELS = CORE10 + [REST]          # 11 classes
N_SEEDS = 5
OUT = ROOT / "results_s1train_s2test.json"


def load_manifest(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def load_features(manifest_path, wanted_labels=None):
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
    X = np.stack(X)
    return X, y, meta, rows


def label_index_map(labels):
    return {w: i for i, w in enumerate(labels)}


def subset_predictions(y_true_words, y_pred_11, probs_11, train_labels, word_labels):
    """Restrict 11-class predictions to the 10 S2 word classes.
    pred==rest counts as an error (FN for the true word; no FP for any word).
    Returns per-seed acc, macroF1, per-class P/R/F1 (dict word->...), confusion 10x10."""
    li = {w: i for i, w in enumerate(word_labels)}
    rest_idx = train_labels.index(REST)
    n = len(word_labels)
    cm = np.zeros((n, n), dtype=np.int64)
    n_correct = 0
    for t, p, pr in zip(y_true_words, y_pred_11, probs_11):
        ti = li[t]
        if p == rest_idx:
            continue  # error: predicted rest on a word clip -> FN only
        if p == ti:
            n_correct += 1
        pi = word_labels.index(train_labels[p])  # p is a word class index in train_labels
        cm[ti, pi] += 1
    acc = n_correct / max(1, len(y_true_words))
    per_class = {}
    for ci, w in enumerate(word_labels):
        tp = cm[ci, ci]
        fn = cm[ci, :].sum() - tp
        fp = cm[:, ci].sum() - tp
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = 2 * p * r / max(1e-9, p + r)
        per_class[w] = dict(precision=float(p), recall=float(r), f1=float(f1))
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))
    return acc, macro_f1, per_class, cm


def top10_confusions(cm, word_labels):
    off = []
    for i in range(len(word_labels)):
        for j in range(len(word_labels)):
            if i != j and cm[i, j] > 0:
                off.append((int(cm[i, j]), word_labels[i], word_labels[j]))
    off.sort(reverse=True)
    return off[:10]


def majority_vote(per_seed_preds):
    """per_seed_preds: list over seeds of list over clips of (true_idx, pred_idx_11, max_prob).
    Returns majority-vote pred_11 per clip and mean max_prob per clip."""
    n_clips = len(per_seed_preds[0])
    final_pred, final_prob, y_true = [], [], []
    for c in range(n_clips):
        votes = defaultdict(int)
        probs = []
        true = None
        for s in range(len(per_seed_preds)):
            t, p, pr = per_seed_preds[s][c]
            true = t
            votes[p] += 1
            probs.append(pr)
        best = max(votes.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        y_true.append(true); final_pred.append(best); final_prob.append(float(np.mean(probs)))
    return y_true, final_pred, final_prob


def train_and_test_s2(Xtr, ytr, train_labels, Xte, yte, word_labels, model_kind, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    li = label_index_map(train_labels)
    ytr_i = np.array([li[w] for w in ytr], dtype=np.int64)
    # map S2 true words to train-label indices (rest not in S2)
    yte_i = np.array([li[w] for w in yte], dtype=np.int64)

    if model_kind == "cnn-v2":
        model = TinyCNN1D(n_in=fv.FEAT_DIM, n_classes=len(train_labels), dropout=0.3)
        train_torch(model, Xtr, ytr_i, seed=seed)
        pred_te, probs_te = predict_torch(model, Xte)   # over 11 classes
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

    acc, mf1, pc, cm = subset_predictions(yte, pred_te, probs_te, train_labels, word_labels)
    top = top10_confusions(cm, word_labels)
    return dict(acc=float(acc), macro_f1=float(mf1), per_class=pc,
                confusions=[[int(v) for v in row] for row in cm], top10=top,
                pred_te=pred_te, probs_te=probs_te, yte_i=yte_i)


def main():
    Xtr, ytr, meta_tr, rows_tr = load_features(S1_MANIFEST, set(TRAIN_LABELS))
    Xte, yte, meta_te, rows_te = load_features(S2_MANIFEST, set(CORE10))  # S2 words only
    train_labels = TRAIN_LABELS
    word_labels = CORE10

    print(f"S1 train clips: {len(Xtr)}  labels={sorted(set(ytr))}")
    print(f"S2 test clips: {len(Xte)}  labels={sorted(set(yte))}  (rest in S2: 0)")

    results = {}
    for model_kind in ("cnn-v2", "logreg"):
        print(f"\n===== {model_kind}: train on S1 ({len(Xtr)} clips), test on S2 ({len(Xte)} clips) =====")
        per_seed = []
        per_seed_preds = []   # for majority-pooling
        t0 = time.time()
        for seed in range(N_SEEDS):
            r = train_and_test_s2(Xtr, ytr, train_labels, Xte, yte, word_labels, model_kind, seed)
            per_seed.append(r)
            # store per-clip (true_idx_11, pred_idx_11, max_prob) for pooling
            li = label_index_map(train_labels)
            yte_i = np.array([li[w] for w in yte], dtype=np.int64)
            clips = [(int(yte_i[i]), int(r["pred_te"][i]), float(r["probs_te"][i].max()))
                     for i in range(len(yte))]
            per_seed_preds.append(clips)
            print(f"  seed {seed}: S2 acc={r['acc']:.3f} macroF1={r['macro_f1']:.3f}")

        accs = [r["acc"] for r in per_seed]
        mf1s = [r["macro_f1"] for r in per_seed]
        # majority-vote pooled over seeds
        y_true_v, y_pred_v, y_prob_v = majority_vote(per_seed_preds)
        li = label_index_map(train_labels)
        rest_idx = li[REST]
        n = len(word_labels)
        cm = np.zeros((n, n), dtype=np.int64)
        n_correct = 0
        for t, p in zip(y_true_v, y_pred_v):
            ti = word_labels.index(train_labels[t])   # t is train-label idx of a word
            if p == rest_idx:
                continue
            if p == ti:
                n_correct += 1
            pi = word_labels.index(train_labels[p])
            cm[ti, pi] += 1
        pool_acc = n_correct / max(1, len(y_true_v))
        pool_pc = {}
        for ci, w in enumerate(word_labels):
            tp = cm[ci, ci]; fn = cm[ci, :].sum() - tp; fp = cm[:, ci].sum() - tp
            p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
            pool_pc[w] = dict(precision=float(p), recall=float(r), f1=float(2 * p * r / max(1e-9, p + r)))
        pool_mf1 = float(np.mean([v["f1"] for v in pool_pc.values()]))
        pool_top = top10_confusions(cm, word_labels)

        # threshold table: pool across seeds using mean max_prob + majority-vote pred
        rest_obs_idx = [i for i, t in enumerate(y_true_v) if train_labels[t] == REST]  # none (no rest in S2)
        rows_thr = []
        for thr in (0.5, 0.7, 0.9):
            accepted = [i for i in range(len(y_true_v)) if y_prob_v[i] >= thr]
            correct = sum(1 for i in accepted if y_pred_v[i] == y_true_v[i])
            rows_thr.append(dict(threshold=thr, n_accepted=len(accepted),
                                 coverage=len(accepted) / len(y_true_v),
                                 accuracy=correct / len(accepted) if accepted else 0.0))
        # argmax row
        correct_all = sum(1 for t, p in zip(y_true_v, y_pred_v) if p == t)
        rows_thr.insert(0, dict(threshold="argmax", n_accepted=len(y_true_v), coverage=1.0,
                                accuracy=correct_all / len(y_true_v)))

        # per-class mean over seeds (report alongside pooled)
        pc_mean = {}
        for w in word_labels:
            fs = [r["per_class"][w] for r in per_seed]
            pc_mean[w] = dict(precision=float(np.mean([f["precision"] for f in fs])),
                              recall=float(np.mean([f["recall"] for f in fs])),
                              f1=float(np.mean([f["f1"] for f in fs])))

        results[model_kind] = dict(
            acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
            mf1_mean=float(np.mean(mf1s)), mf1_std=float(np.std(mf1s)),
            per_seed=[dict(seed=r["seed"] if "seed" in r else i, acc=float(r["acc"]),
                           macro_f1=float(r["macro_f1"])) for i, r in enumerate(per_seed)],
            pooled_acc=pool_acc, pooled_macro_f1=pool_mf1,
            per_class=pool_pc, per_class_seed_mean=pc_mean,
            top10=[[c, i, j] for c, i, j in pool_top],
            thresholds=rows_thr,
            n_test=len(Xte),
        )
        print(f"\n--- {model_kind} S2 test (majority-vote pooled over {N_SEEDS} seeds) ---")
        print(f"  acc={pool_acc:.3f} macroF1={pool_mf1:.3f}  (per-seed mean±std: "
              f"{np.mean(accs):.3f}±{np.std(accs):.3f} / {np.mean(mf1s):.3f}±{np.std(mf1s):.3f})")
        print(f"  chance(uniform, 10 classes)={1/len(word_labels):.3f}")
        print(f"  per-class F1:")
        for w in sorted(pool_pc, key=lambda k: pool_pc[k]["f1"]):
            m = pool_pc[w]
            print(f"    {w:14} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}")
        print(f"  top-10 confusions (count, true->pred):")
        for c, i, j in pool_top:
            print(f"    {c:3d}  {i:14} -> {j}")
        print(f"  threshold table (S2):")
        print(f"    {'threshold':>10} {'accepted':>9} {'coverage':>9} {'acc':>7}")
        for row in rows_thr:
            print(f"    {str(row['threshold']):>10} {row['n_accepted']:>9} {row['coverage']:8.1%} {row['accuracy']:7.3f}")
        print(f"  rest FPR: NOT VERIFIED (no rest clips in S2)")
        print(f"  time: {time.time() - t0:.1f}s")

    # clean: remove non-serializable arrays before saving
    for mk in results:
        results[mk].pop("pred_te", None); results[mk].pop("probs_te", None); results[mk].pop("yte_i", None)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT}")
    print("All numbers PRELIMINARY, OPTIMISTIC — train S1, test S2 (cross-session). No tuning based on S2.")


if __name__ == "__main__":
    main()
