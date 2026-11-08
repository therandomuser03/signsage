"""Track B final vocabulary evaluation (REPORT_I).

Train cnn-v3 on S1+S2 using 8 words + rest, test on S3 restricted to the 8 words,
5 seeds. Also comparison: earlier 10-class model's accuracy on the same 8 words'
S3 clips (counting thank you/please/rest predictions as errors).

Vocabulary: hello, me, you, yes, no, help, water, where + rest.
Excluded (weak on S3): thank you, please. Their clips are dropped completely.

Label: "VOCABULARY CHOSEN AFTER SEEING S3: slightly optimistic".
Threshold kept at 0.7 (not re-selected from S3).
"""
import csv, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import features_v3 as fv
from eval_v3 import TinyCNN1D, train_model_v3, predict_torch

EIGHT = ["hello", "me", "you", "yes", "no", "help", "water", "where"]
REST = "rest"
TRAIN9 = EIGHT + [REST]          # 9 classes
TEST8 = EIGHT                     # 8 classes (no rest in S3)
TRAIN10 = ["hello", "thank you", "please", "yes", "no", "help", "water",
           "you", "me", "where", REST]  # 11 classes
N_SEEDS = 5
EPOCHS = 60
OUT = ROOT / "results_final8.json"


def load_manifest(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


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


def train_cnn(X, y, labels, n_classes, seed):
    np.random.seed(seed); torch.manual_seed(seed)
    li = {w: i for i, w in enumerate(labels)}
    yi = np.array([li[w] for w in y], dtype=np.int64)
    model = TinyCNN1D(n_in=fv.FEAT_DIM, n_classes=n_classes, dropout=0.3)
    train_model_v3(model, X, yi, epochs=EPOCHS, seed=seed)
    return model, li


def predict_cnn(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float())
    probs = torch.softmax(logits, 1).numpy()
    return probs.argmax(1), probs


def word_metrics(y_true_words, pred_9class, probs_9class, train_labels, test_words):
    """Metrics over the 8 test words. pred is index in train_labels space."""
    li = {w: i for i, w in enumerate(train_labels)}
    rest_idx = li[REST]
    word_idx = {w: li[w] for w in test_words}       # word -> train index
    idx_to_pos = {idx: p for p, (w, idx) in enumerate(word_idx.items())}
    n = len(test_words)
    cm = np.zeros((n, n), dtype=np.int64)
    n_correct = 0
    max_probs = []
    for t, p, pr in zip(y_true_words, pred_9class, probs_9class):
        t_pos = list(test_words).index(t)
        if p == rest_idx or p not in idx_to_pos:
            max_probs.append(float(pr.max()))
            continue  # predicted rest/excluded -> error (FN only)
        p_pos = idx_to_pos[p]
        if t_pos == p_pos:
            n_correct += 1
        cm[t_pos, p_pos] += 1
        max_probs.append(float(pr.max()))
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
    return acc, macro_f1, per_class, cm, off[:10], max_probs


def threshold_table(max_probs, pred_9class, y_true_words, train_labels, test_words,
                    thresholds=(0.5, 0.7, 0.9)):
    li = {w: i for i, w in enumerate(train_labels)}
    word_idx = {w: li[w] for w in test_words}
    idx_to_word = {idx: w for w, idx in word_idx.items()}
    n = len(y_true_words)
    rows = []
    correct_all = 0
    for t, p in zip(y_true_words, pred_9class):
        if p in idx_to_word and idx_to_word[p] == t:
            correct_all += 1
    rows.append(dict(threshold="argmax", n_accepted=n, coverage=1.0,
                     accuracy=correct_all / n))
    for thr in thresholds:
        accepted = [i for i in range(n) if max_probs[i] >= thr]
        correct = 0
        for i in accepted:
            p = pred_9class[i]
            if p in idx_to_word and idx_to_word[p] == y_true_words[i]:
                correct += 1
        rows.append(dict(threshold=thr, n_accepted=len(accepted),
                         coverage=len(accepted) / n,
                         accuracy=correct / len(accepted) if accepted else 0.0))
    return rows


def main():
    Xtr_s1, ytr_s1 = load_features(ROOT / "data" / "s1_converted" / "manifest.csv",
                                   set(TRAIN9))
    Xtr_s2, ytr_s2 = load_features(ROOT / "data" / "s2_converted" / "manifest.csv",
                                   set(TEST8))
    Xte, yte = load_features(ROOT / "data" / "s3_converted" / "manifest.csv",
                             set(TEST8))
    Xtr = np.concatenate([Xtr_s1, Xtr_s2], axis=0)
    ytr = ytr_s1 + ytr_s2
    print(f"Train S1+S2 (8 words + rest): {len(Xtr)} clips")
    print(f"Test S3 (8 words): {len(Xte)} clips")

    results = {}
    t_all = time.time()

    # ---- Step 1: 8-class model ----
    print("\n===== Step 1: cnn-v3 8 words + rest, train S1+S2, test S3 =====")
    per_seed = []
    per_seed_preds = []
    for seed in range(N_SEEDS):
        model, li = train_cnn(Xtr, ytr, TRAIN9, 9, seed)
        pred_te, probs_te = predict_cnn(model, Xte)
        acc, mf1, pc, cm, top, max_probs = word_metrics(yte, pred_te, probs_te,
                                                        TRAIN9, TEST8)
        per_seed.append(dict(seed=seed, acc=float(acc), macro_f1=float(mf1),
                             per_class=pc, top10=[[c, i, j] for c, i, j in top]))
        per_seed_preds.append((yte, pred_te, probs_te, max_probs))
        print(f"  seed {seed}: acc={acc:.3f} macroF1={mf1:.3f}")
    accs = [r["acc"] for r in per_seed]; mf1s = [r["macro_f1"] for r in per_seed]
    pc_mean = {}
    for w in TEST8:
        fs = [r["per_class"][w] for r in per_seed]
        pc_mean[w] = dict(precision=float(np.mean([f["precision"] for f in fs])),
                          recall=float(np.mean([f["recall"] for f in fs])),
                          f1=float(np.mean([f["f1"] for f in fs])))
    rep_top = per_seed[0]["top10"]
    print(f"\n8-class: acc={np.mean(accs):.3f}±{np.std(accs):.3f} "
          f"macroF1={np.mean(mf1s):.3f}±{np.std(mf1s):.3f}")
    print("per-class F1:", ", ".join(f"{w}={m['f1']:.2f}"
          for w, m in sorted(pc_mean.items(), key=lambda kv: kv[1]['f1'])))
    print("top-10:", rep_top)
    yte0, pred0, probs0, mp0 = per_seed_preds[0]
    thr_rows = threshold_table(mp0, pred0, yte0, TRAIN9, TEST8)
    for row in thr_rows:
        print(f"  thr {row['threshold']}: accepted={row['n_accepted']} "
              f"cov={row['coverage']:.1%} acc={row['accuracy']:.3f}")

    results["eight_class"] = dict(
        acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
        mf1_mean=float(np.mean(mf1s)), mf1_std=float(np.std(mf1s)),
        per_seed=[dict(seed=r["seed"], acc=float(r["acc"]),
                       macro_f1=float(r["macro_f1"])) for r in per_seed],
        pooled_acc=float(np.mean(accs)), pooled_macro_f1=float(np.mean(mf1s)),
        per_class=pc_mean, top10=rep_top, thresholds=thr_rows,
        n_test=len(Xte),
    )

    # ---- Step 2: 10-class comparison model ----
    print("\n===== Step 2: 10-class comparison on same 8 S3 words =====")
    Xtr10_s1, ytr10_s1 = load_features(ROOT / "data" / "s1_converted" / "manifest.csv",
                                       set(TRAIN10))
    Xtr10_s2, ytr10_s2 = load_features(ROOT / "data" / "s2_converted" / "manifest.csv",
                                       set(TRAIN10))
    Xtr10 = np.concatenate([Xtr10_s1, Xtr10_s2], axis=0)
    ytr10 = ytr10_s1 + ytr10_s2
    print(f"10-class train S1+S2: {len(Xtr10)} clips")
    li10 = {w: i for i, w in enumerate(TRAIN10)}
    word_idx10 = {w: li10[w] for w in TEST8}
    per_seed10 = []
    for seed in range(N_SEEDS):
        model, _ = train_cnn(Xtr10, ytr10, TRAIN10, 11, seed)
        pred_te, _ = predict_cnn(model, Xte)
        n_correct = sum(1 for t, p in zip(yte, pred_te)
                        if p == word_idx10[t])
        acc10 = n_correct / len(yte)
        per_seed10.append(acc10)
        print(f"  seed {seed}: 10-class acc on 8 words={acc10:.3f}")
    print(f"10-class comparison: acc={np.mean(per_seed10):.3f}±{np.std(per_seed10):.3f}")
    results["ten_class_comparison"] = dict(
        acc_mean=float(np.mean(per_seed10)), acc_std=float(np.std(per_seed10)),
        per_seed=per_seed10, n_test=len(Xte),
    )

    results["vocab"] = dict(train8=TEST8, rest=REST,
                            excluded=["thank you", "please"],
                            train9=TRAIN9, test8=TEST8)
    results["n_train"] = int(len(Xtr))
    results["n_test"] = int(len(Xte))
    results["note"] = ("VOCABULARY CHOSEN AFTER SEEING S3: slightly optimistic. "
                       "Threshold kept at 0.7 (not re-selected from S3).")
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT} (total {time.time()-t_all:.1f}s)")
    print(results["note"])


if __name__ == "__main__":
    main()
