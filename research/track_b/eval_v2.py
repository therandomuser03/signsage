"""Track B step 3 (research only, S1): v2 feature models.

Models:
  (a) 1D-CNN with dropout 0.3 on v2 features (122 dims x 60 frames)
  (b) baseline: flattened 30-frame v2 features -> StandardScaler ->
      multinomial logistic regression (numpy, L2)
  (c) old v1 CNN (features.py, 98 dims) for comparison
5-fold CV across takes (take groups never straddle train/test), 5 seeds.
Vocabularies: CORE-10 (10 words + rest), CORE-9 (without "you"), FULL.
Reports: accuracy, macro-F1, per-class P/R/F1 table, top-10 confusions,
chance baseline, max-softmax threshold table (0.5/0.7/0.9), rest FPR.
Nothing exported or integrated into the app. Offline, CPU only.
"""
import csv, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from features import extract_features as extract_v1   # old v1, 98 dims
import features_v2 as fv                               # new v2, 122 dims

MANIFEST = ROOT / "data" / "s1_converted" / "manifest.csv"
CORE10 = ["hello", "thank you", "please", "yes", "no", "help", "water", "you", "me", "where"]
CORE9 = [w for w in CORE10 if w != "you"]             # 9 words (without "you")
REST = "rest"
N_SEEDS = 5
N_FOLDS = 5
EPOCHS = 60
OUT = ROOT / "results_v2.json"
QUICK = int(os.environ.get("QUICK", "0"))   # QUICK=1 => 1 seed, fast smoke test
if QUICK:
    N_SEEDS = 1


# ---------------------------------------------------------------- data
def load_data():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    X, X1, y, meta = [], [], [], []
    for i, r in enumerate(rows):
        arr = np.load(r["npy"]).astype(np.float32)
        f = fv.extract_features(arr, layout="signsage")   # (60, 122) v2
        f1 = extract_v1(arr, layout="signsage")             # (60, 98)  v1
        assert f.shape == (fv.N_FRAMES, fv.FEAT_DIM), f.shape
        assert f1.shape == (60, 98), f1.shape
        X.append(f); X1.append(f1); y.append(r["label"]); meta.append(r)
        if (i + 1) % 50 == 0:
            print(f"  features {i + 1}/{len(rows)}")
    X = np.stack(X); X1 = np.stack(X1)
    labels = sorted(set(y))
    li = {w: i for i, w in enumerate(labels)}
    yi = np.array([li[w] for w in y], dtype=np.int64)
    print(f"features: {len(X)} clips x {X.shape[1]}x{X.shape[2]} v2  "
          f"x {X1.shape[2]} v1  ({len(labels)} labels: {labels})")
    return X, X1, yi, labels, meta


def take_folds(y, labels, meta):
    """fold id per clip: each label's takes split into 5 folds (never mixed)."""
    fold = np.zeros(len(y), dtype=int)
    for w in labels:
        idx = np.where(y == labels.index(w))[0]
        idx = idx[np.argsort([int(meta[i]["take"]) for i in idx])]
        for k, part in enumerate(np.array_split(idx, N_FOLDS)):
            fold[part] = k
    return fold


def subset_idx(y, labels, vocab):
    keep = [w for w in vocab if w in set(labels)]
    idl = {w: i for i, w in enumerate(labels)}
    keep_idx = [idl[w] for w in keep]
    mask = np.isin(y, keep_idx)
    remap = {old: new for new, old in enumerate(keep_idx)}
    y2 = np.array([remap[int(v)] for v in y[mask]], dtype=np.int64)
    return mask, y2, keep


# ---------------------------------------------------------------- metrics
def accuracy_metrics(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    acc = cm.trace() / max(1, cm.sum())
    per_class = {}
    for c in range(n_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = 2 * p * r / max(1e-9, p + r)
        per_class[c] = (p, r, f1)
    macro_f1 = float(np.mean([v[2] for v in per_class.values()]))
    counts = np.bincount(y_true, minlength=n_classes)
    baseline = counts.max() / max(1, len(y_true))
    off = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                off.append((int(cm[i, j]), i, j))
    off.sort(reverse=True)
    return acc, macro_f1, per_class, cm, baseline, off[:10]


def threshold_table(oof, labels, n_classes, thresholds=(0.5, 0.7, 0.9)):
    n = len(oof)
    rest_idx = labels.index(REST)
    rest_obs = [o for o in oof if o["true"] == rest_idx]
    rows = []
    for t in thresholds:
        accepted = [o for o in oof if o["max_prob"] >= t]
        correct = sum(1 for o in accepted if o["pred"] == o["true"])
        rest_as_word = sum(1 for o in rest_obs if o["max_prob"] >= t and o["pred"] != rest_idx)
        rows.append(dict(threshold=t, n_accepted=len(accepted),
                         coverage=len(accepted) / n,
                         accuracy=correct / len(accepted) if accepted else 0.0,
                         rest_clips=len(rest_obs),
                         rest_wrong_as_word=int(rest_as_word),
                         rest_wrong_as_word_rate=(rest_as_word / len(rest_obs) if rest_obs else 0.0)))
    correct = sum(1 for o in oof if o["pred"] == o["true"])
    rest_as_word = sum(1 for o in rest_obs if o["pred"] != rest_idx)
    rows.insert(0, dict(threshold="argmax", n_accepted=n, coverage=1.0,
                        accuracy=correct / n,
                        rest_clips=len(rest_obs),
                        rest_wrong_as_word=int(rest_as_word),
                        rest_wrong_as_word_rate=(rest_as_word / len(rest_obs) if rest_obs else 0.0)))
    return rows


# ---------------------------------------------------------------- models
class TinyCNN1D(nn.Module):
    def __init__(self, n_in, n_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_in, 32, 3, padding=1), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):           # x: (B, 60, n_in) -> (B, n_in, 60)
        x = x.transpose(1, 2)
        x = self.net(x).squeeze(-1)
        return self.head(x)


def train_torch(model, Xtr, ytr, epochs=EPOCHS, lr=1e-3, batch=16, seed=0):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.RandomState(seed)
    N = len(Xtr)
    for ep in range(epochs):
        perm = rng.permutation(N)
        total = 0.0
        for s in range(0, N, batch):
            ib = perm[s:s + batch]
            xb = torch.from_numpy(Xtr[ib]).float()
            yb = torch.from_numpy(ytr[ib]).long()
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(ib)
    model.eval()
    return model


def predict_torch(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float())
    probs = torch.softmax(logits, 1).numpy()
    return probs.argmax(1), probs


# ---------------------------------------------------------------- logistic regression (numpy)
class LogisticRegression:
    """Multinomial logistic regression with L2, trained by gradient descent."""
    def __init__(self, lr=0.05, epochs=2000, l2=1e-3, seed=0, verbose=False):
        self.lr = lr; self.epochs = epochs; self.l2 = l2
        self.seed = seed; self.verbose = verbose
        self.W = None; self.b = None

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        n, d = X.shape
        k = int(y.max()) + 1
        self.W = rng.randn(d, k) * 0.01
        self.b = np.zeros(k)
        Y = np.eye(k)[y]
        for ep in range(self.epochs):
            scores = X @ self.W + self.b
            scores -= scores.max(axis=1, keepdims=True)
            exp = np.exp(scores)
            probs = exp / exp.sum(axis=1, keepdims=True)
            grad = (X.T @ (probs - Y)) / n + self.l2 * self.W
            self.W -= self.lr * grad
            self.b -= self.lr * (probs - Y).mean(axis=0)
            if self.verbose and ep % 500 == 0:
                loss = -np.log(probs[np.arange(n), y] + 1e-12).mean() + 0.5 * self.l2 * (self.W ** 2).sum()
                print(f"    lr-reg epoch {ep} loss={loss:.4f}")
        return self

    def predict_proba(self, X):
        scores = X @ self.W + self.b
        scores -= scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def standardize_fit(Xtr):
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    return mu, sd


def resample_batch(X, n_frames):
    """Resample each clip (B, 60, D) -> (B, n_frames, D) along the frame axis."""
    out = np.stack([fv._resample(c, n_frames) for c in X], axis=0)
    return out


def standardize_apply(X, mu, sd):
    return (X - mu) / sd


# ---------------------------------------------------------------- CV run
def cv_run(X, X1, y, labels, meta, vocab, model_kind, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    mask, y2, keep = subset_idx(y, labels, vocab)
    Xv, yv = X[mask], y2
    X1v = X1[mask]
    folds = take_folds(yv, keep, [meta[i] for i in np.where(mask)[0]])
    n_classes = len(keep)
    fold_metrics, oof = [], []
    meta_v = [meta[i] for i in np.where(mask)[0]]
    for k in range(N_FOLDS):
        tr = np.where(folds != k)[0]; te = np.where(folds == k)[0]
        if model_kind == "cnn-v2":
            model = TinyCNN1D(n_in=fv.FEAT_DIM, n_classes=n_classes, dropout=0.3)
            train_torch(model, Xv[tr], yv[tr], seed=seed)
            pred_te, probs_te = predict_torch(model, Xv[te])
        elif model_kind == "cnn-v1":
            model = TinyCNN1D(n_in=98, n_classes=n_classes, dropout=0.3)
            train_torch(model, X1v[tr], yv[tr], seed=seed)
            pred_te, probs_te = predict_torch(model, X1v[te])
        elif model_kind == "logreg":
            flat_tr = resample_batch(Xv[tr], 30).reshape(len(tr), -1)   # 30*122
            flat_te = resample_batch(Xv[te], 30).reshape(len(te), -1)
            mu, sd = standardize_fit(flat_tr)
            clf = LogisticRegression(seed=seed)
            clf.fit(standardize_apply(flat_tr, mu, sd), yv[tr])
            pred_te = clf.predict(standardize_apply(flat_te, mu, sd))
            probs_te = clf.predict_proba(standardize_apply(flat_te, mu, sd))
        else:
            raise ValueError(model_kind)
        acc, mf1, pc, cm, baseline, off = accuracy_metrics(yv[te], pred_te, n_classes)
        fold_metrics.append(dict(fold=k, n_tr=len(tr), n_te=len(te), acc=float(acc),
                                 macro_f1=float(mf1), baseline=float(baseline),
                                 per_class={keep[c]: dict(precision=float(pc[c][0]),
                                                          recall=float(pc[c][1]),
                                                          f1=float(pc[c][2]))
                                            for c in range(n_classes)},
                                 confusions=[[int(v) for v in row] for row in cm],
                                 labels=list(keep)))
        for j, idx in enumerate(te):
            oof.append(dict(clip=int(np.where(mask)[0][idx]), fold=k, seed=seed,
                            true=int(yv[idx]), pred=int(pred_te[j]),
                            max_prob=float(probs_te[j].max())))
    return dict(model=model_kind, vocab=list(keep), n_classes=n_classes,
                seed=seed, folds=fold_metrics, oof=oof)


def agg(seed_runs):
    out = {}
    for r in seed_runs:
        key = f"{r['model']} {r['vocab_key']}"
        out.setdefault(key, []).append(r)
    res = {}
    for key, runs in out.items():
        accs = [r["cv_mean_acc"] for r in runs]
        mf1s = [r["cv_mean_macro_f1"] for r in runs]
        res[key] = dict(acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
                        mf1_mean=float(np.mean(mf1s)), mf1_std=float(np.std(mf1s)),
                        per_seed=[dict(seed=r["seed"], acc=float(r["cv_mean_acc"]),
                                       macro_f1=float(r["cv_mean_macro_f1"])) for r in runs])
    return res


def pooled(oof_all, labels):
    by_clip = defaultdict(list)
    for o in oof_all:
        by_clip[o["clip"]].append(o)
    y_true, y_pred, probs = [], [], []
    for clip, obs in by_clip.items():
        votes = defaultdict(int)
        for o in obs:
            votes[o["pred"]] += 1
        best = max(votes.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        y_true.append(obs[0]["true"]); y_pred.append(best)
        probs.append(max(o["max_prob"] for o in obs))
    acc, mf1, pc, cm, baseline, off = accuracy_metrics(np.array(y_true), np.array(y_pred), len(labels))
    top = [[int(c), labels[i], labels[j]] for c, i, j in off]
    return dict(acc=float(acc), macro_f1=float(mf1), baseline=float(baseline),
                per_class={labels[c]: dict(precision=float(pc[c][0]), recall=float(pc[c][1]),
                                           f1=float(pc[c][2])) for c in range(len(labels))},
                top10=top, n=int(len(y_true)))


def main():
    X, X1, y, labels, meta = load_data()
    vocabs = {"CORE-10": CORE10 + [REST], "CORE-9": CORE9 + [REST], "FULL": labels}
    all_results, all_oof = [], []
    t_all = time.time()
    for seed in range(N_SEEDS):
        print(f"\n===== SEED {seed} =====")
        for vk, vocab in vocabs.items():
            for model_kind in ("cnn-v2", "logreg", "cnn-v1"):
                r = cv_run(X, X1, y, labels, meta, vocab, model_kind, seed)
                cm_accs = [f["acc"] for f in r["folds"]]
                cm_mf1 = [f["macro_f1"] for f in r["folds"]]
                r["cv_mean_acc"] = float(np.mean(cm_accs))
                r["cv_mean_macro_f1"] = float(np.mean(cm_mf1))
                r["vocab_key"] = vk
                print(f"[{model_kind:8} {vk:7} seed {seed}] "
                      f"CV acc={r['cv_mean_acc']:.3f} macroF1={r['cv_mean_macro_f1']:.3f}")
                all_results.append(r)
                all_oof.extend(r["oof"])
    summary = agg(all_results)

    oof_by_vocab_model = defaultdict(list)
    for r in all_results:
        for o in r["oof"]:
            oof_by_vocab_model[(r["vocab_key"], r["model"])].append(o)

    pooled_res = {}
    for vk in vocabs:
        for model_kind in ("cnn-v2", "logreg", "cnn-v1"):
            key = (vk, model_kind)
            lab_list = vocabs[vk]
            p = pooled(oof_by_vocab_model[key], lab_list)
            p["thresholds"] = threshold_table(oof_by_vocab_model[key], lab_list, len(lab_list))
            pc = p["per_class"]
            order = sorted(pc.items(), key=lambda kv: kv[1]["f1"])
            pooled_res[key] = p
            print(f"\n--- POOLED {model_kind} {vk} (n={p['n']}): acc={p['acc']:.3f} "
                  f"macroF1={p['macro_f1']:.3f} baseline(majority)={p['baseline']:.3f} "
                  f"chance(uniform)={1/len(pc):.3f} ---")
            print(f"{'class':14} {'P':>6} {'R':>6} {'F1':>6}")
            for name, m in order:
                print(f"{name:14} {m['precision']:6.2f} {m['recall']:6.2f} {m['f1']:6.2f}")
            print("Top-10 confusions (count, true -> pred):")
            for c, i, j in p["top10"]:
                print(f"  {c:3d}  {i:14} -> {j:14}")

    # thresholds for best model on CORE-10 (pick best pooled macroF1 cnn-v2)
    print("\n--- max-softmax threshold table (cnn-v2 CORE-10) ---")
    p = pooled_res[("CORE-10", "cnn-v2")]
    print(f"{'threshold':>10} {'accepted':>9} {'coverage':>9} {'acc':>7} "
          f"{'rest_clips':>10} {'rest->word':>11} {'rate':>7}")
    for row in p["thresholds"]:
        print(f"{str(row['threshold']):>10} {row['n_accepted']:>9} {row['coverage']:8.1%} "
              f"{row['accuracy']:7.3f} {row['rest_clips']:>10} "
              f"{row['rest_wrong_as_word']:>11} {row['rest_wrong_as_word_rate']:7.1%}")

    report = dict(summary=summary, pooled={f"{mk} {vk}": pooled_res[(vk, mk)]
                                           for vk in vocabs for mk in ("cnn-v2", "logreg", "cnn-v1")},
                  vocabs=vocabs, n_clips=int(len(X)), labels=labels,
                  note="PRELIMINARY, OPTIMISTIC (same session S1). Not a final result.")
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {OUT} (total {time.time() - t_all:.1f}s)")

    print("\n" + "=" * 70)
    print("SUMMARY: mean +/- std over 5 seeds of 5-fold take CV (S1, v2 features)")
    print("=" * 70)
    for key, s in sorted(summary.items()):
        print(f"{key:20} acc={s['acc_mean']:.3f} +/- {s['acc_std']:.3f}  "
              f"macroF1={s['mf1_mean']:.3f} +/- {s['mf1_std']:.3f}")
    print(report["note"])


if __name__ == "__main__":
    main()
