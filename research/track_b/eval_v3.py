"""Track B step 3: cnn-v3 evaluation (train S1, test S2 dev) + augmentation.

Uses EXACTLY the cnn-v2 hyperparameters from REPORT_E (dropout 0.3, 60 epochs,
Adam 1e-3, batch 16) plus ONE design change (features_v3) and the training-only
augmentation below. No tuning on S2. Offline, CPU.

Augmentation (training batches only, cnn-v3):
  - random hand-frame dropout (mask=0, 0-50% of frames) per hand
  - scale jitter +/-10%
  - rotation +/-10 degrees
  - time-warp +/-15%

Evaluates:
  (a) S1 5-fold CV across takes (cnn-v3 + logreg), CORE-10 + rest.
  (b) Train on S1 only, test on S2 only (DEV SET: S2 was used to diagnose the
      handedness problem, not a clean held-out test). ONE config, 5 seeds.
  (c) Hand-label check AFTER canonicalisation: fraction of clips mirrored per
      label in S1 vs S2.
"""
import csv, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import features_v3 as fv
from eval_v2 import (TinyCNN1D, train_torch, predict_torch, LogisticRegression,
                     standardize_fit, standardize_apply, resample_batch,
                     accuracy_metrics, threshold_table, take_folds, subset_idx)

S1_MANIFEST = ROOT / "data" / "s1_converted" / "manifest.csv"
S2_MANIFEST = ROOT / "data" / "s2_converted" / "manifest.csv"
CORE10 = ["hello", "thank you", "please", "yes", "no", "help", "water", "you", "me", "where"]
REST = "rest"
TRAIN_LABELS = CORE10 + [REST]          # 11 classes (S1 train)
S2_WORDS = CORE10                       # 10 word classes present in S2
N_SEEDS = 5
EPOCHS = 60
OUT = ROOT / "results_v3.json"
QUICK = int(os.environ.get("QUICK", "0"))
if QUICK:
    N_SEEDS = 1


def load_manifest(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def load_features(manifest_path, wanted_labels=None):
    rows = load_manifest(manifest_path)
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


# ---------------------------------------------------------------- augmentation
def augment_v3(x):
    """x: (B, 60, 122). Training-only augmentation, masks left untouched except dropout."""
    B = x.shape[0]
    xp = x.copy()
    for b in range(B):
        # scale jitter +/-10% (all pair dims, not the 2 mask scalars)
        if np.random.rand() < 0.5:
            s = np.random.uniform(0.90, 1.10)
            xp[b, :, 0:84] *= s
            xp[b, :, 86:122] *= s
        # rotation +/-10 degrees (all pair dims)
        if np.random.rand() < 0.5:
            ang = np.deg2rad(np.random.uniform(-10.0, 10.0))
            c, s = np.cos(ang), np.sin(ang)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)
            ah = xp[b, :, 0:84].reshape(60, 42, 2)               # 42 pairs
            xp[b, :, 0:84] = (ah @ R.T).reshape(60, 84)
            sh = xp[b, :, 86:122].reshape(60, 18, 2)             # 18 pairs
            xp[b, :, 86:122] = (sh @ R.T).reshape(60, 36)
        # hand-frame dropout: 0-50% of frames per hand (zero channels + mask)
        for col_slice, mask_col in ((slice(0, 42), 84), (slice(42, 84), 85)):
            n_drop = int(np.random.uniform(0.0, 0.50) * 60)
            if n_drop > 0:
                drop_frames = np.random.choice(60, n_drop, replace=False)
                xp[b, drop_frames, col_slice] = 0.0
                xp[b, drop_frames, mask_col] = 0.0
    return xp


def train_model_v3(model, Xtr, ytr, epochs=EPOCHS, lr=1e-3, batch=16, seed=0):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    rng = np.random.RandomState(seed)
    N = len(Xtr)
    for ep in range(epochs):
        perm = rng.permutation(N)
        total = 0.0
        for s in range(0, N, batch):
            ib = perm[s:s + batch]
            xb = torch.from_numpy(augment_v3(Xtr[ib])).float()
            yb = torch.from_numpy(ytr[ib]).long()
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(ib)
    model.eval()
    return model


# ---------------------------------------------------------------- S2 word restriction
def restrict_to_s2_words(y_true_words, pred_11, probs_11, train_labels, word_labels):
    """Map 11-class predictions onto the 10 S2 word classes. pred==rest => error."""
    li = {w: i for i, w in enumerate(train_labels)}
    rest_idx = li[REST]
    word_idx = {w: li[w] for w in word_labels}
    n = len(word_labels)
    cm = np.zeros((n, n), dtype=np.int64)
    n_correct = 0
    for t, p in zip(y_true_words, pred_11):
        ti = word_idx[t]
        if p == rest_idx:
            continue  # FN only
        if p == ti:
            n_correct += 1
        # p must be a word class
        p_word = train_labels[p]
        if p_word in word_idx:
            cm[ti, word_idx[p_word]] += 1
    acc = n_correct / max(1, len(y_true_words))
    per_class = {}
    for ci, w in enumerate(word_labels):
        tp = cm[ci, ci]; fn = cm[ci, :].sum() - tp; fp = cm[:, ci].sum() - tp
        p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
        per_class[w] = dict(precision=float(p), recall=float(r),
                            f1=float(2 * p * r / max(1e-9, p + r)))
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))
    off = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                off.append((int(cm[i, j]), word_labels[i], word_labels[j]))
    off.sort(reverse=True)
    return acc, macro_f1, per_class, cm, off[:10]


def top10_from_cm(cm, word_labels):
    off = []
    for i in range(len(word_labels)):
        for j in range(len(word_labels)):
            if i != j and cm[i, j] > 0:
                off.append((int(cm[i, j]), word_labels[i], word_labels[j]))
    off.sort(reverse=True)
    return off[:10]


# ---------------------------------------------------------------- CV
def cv_run_v3(X, y, labels, meta, vocab, model_kind, seed):
    np.random.seed(seed); torch.manual_seed(seed)
    keep = [w for w in vocab if w in set(labels)]
    idl = {w: i for i, w in enumerate(labels)}
    keep_idx = [idl[w] for w in keep]
    y_arr = np.array([idl[w] for w in y], dtype=np.int64)
    mask = np.isin(y_arr, keep_idx)
    Xv, yv = X[mask], y_arr[mask]
    remap = {old: new for new, old in enumerate(keep_idx)}
    yv = np.array([remap[int(v)] for v in yv], dtype=np.int64)
    folds = take_folds(yv, keep, [meta[i] for i in np.where(mask)[0]])
    n_classes = len(keep)
    fold_metrics, oof = [], []
    N_FOLDS = 5
    for k in range(N_FOLDS):
        tr = np.where(folds != k)[0]; te = np.where(folds == k)[0]
        if model_kind == "cnn-v3":
            model = TinyCNN1D(n_in=fv.FEAT_DIM, n_classes=n_classes, dropout=0.3)
            train_model_v3(model, Xv[tr], yv[tr], seed=seed)
            pred_te, probs_te = predict_torch(model, Xv[te])
        elif model_kind == "logreg":
            flat_tr = resample_batch(Xv[tr], 30).reshape(len(tr), -1)
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
                                            for c in range(n_classes)}))
        for j, idx in enumerate(te):
            oof.append(dict(clip=int(np.where(mask)[0][idx]), fold=k, seed=seed,
                            true=int(yv[idx]), pred=int(pred_te[j]),
                            max_prob=float(probs_te[j].max())))
    return dict(model=model_kind, vocab=list(keep), n_classes=n_classes, seed=seed,
                folds=fold_metrics, oof=oof)


# ---------------------------------------------------------------- mirror check
def clip_mirrored(arr):
    """True if features_v3 would mirror this clip (active hand image-left of shoulder midpoint)."""
    pose = arr[:, 0:33, :]
    lh = fv._xy(arr, fv._LHAND); rh = fv._xy(arr, fv._RHAND)
    centre, width = fv._shoulder_centre_width(pose)
    phys, wrists = fv._track_hands(lh, rh)
    active = fv._active_secondary(phys, wrists)
    return fv._should_mirror(phys, active, wrists, centre)


def main():
    Xtr, ytr, meta_tr, rows_tr = load_features(S1_MANIFEST, set(TRAIN_LABELS))
    Xte, yte, meta_te, rows_te = load_features(S2_MANIFEST, set(S2_WORDS))
    train_labels = TRAIN_LABELS
    word_labels = S2_WORDS

    print(f"S1 train clips: {len(Xtr)}  labels={sorted(set(ytr))}")
    print(f"S2 dev clips:   {len(Xte)}  labels={sorted(set(yte))}  (rest in S2: 0)")

    results = {}
    t_all = time.time()
    for model_kind in ("cnn-v3", "logreg"):
        print(f"\n===== {model_kind}: S1 5-fold CV (CORE-10 + rest) =====")
        seed_runs = []
        for seed in range(N_SEEDS):
            r = cv_run_v3(Xtr, ytr, train_labels, meta_tr, train_labels, model_kind, seed)
            cm_accs = [f["acc"] for f in r["folds"]]
            cm_mf1 = [f["macro_f1"] for f in r["folds"]]
            r["cv_mean_acc"] = float(np.mean(cm_accs))
            r["cv_mean_macro_f1"] = float(np.mean(cm_mf1))
            seed_runs.append(r)
            print(f"  seed {seed}: CV acc={r['cv_mean_acc']:.3f} macroF1={r['cv_mean_macro_f1']:.3f}")
        accs = [r["cv_mean_acc"] for r in seed_runs]
        mf1s = [r["cv_mean_macro_f1"] for r in seed_runs]
        # pooled per-class over seeds' OOF (majority vote)
        oof_all = [o for r in seed_runs for o in r["oof"]]
        by_clip = defaultdict(list)
        for o in oof_all:
            by_clip[o["clip"]].append(o)
        y_true_v, y_pred_v = [], []
        li = li_map(train_labels)
        for clip, obs in by_clip.items():
            votes = defaultdict(int)
            for o in obs:
                votes[o["pred"]] += 1
            best = max(votes.items(), key=lambda kv: (kv[1], -kv[0]))[0]
            y_true_v.append(obs[0]["true"]); y_pred_v.append(best)
        acc_p, mf1_p, pc_p, cm_p, base_p, off_p = accuracy_metrics(
            np.array(y_true_v), np.array(y_pred_v), len(train_labels))
        pc_order = sorted(pc_p.items(), key=lambda kv: kv[1][2])
        print(f"  POOLED S1 CV: acc={acc_p:.3f} macroF1={mf1_p:.3f} (mean±std over seeds: "
              f"{np.mean(accs):.3f}±{np.std(accs):.3f} / {np.mean(mf1s):.3f}±{np.std(mf1s):.3f})")
        print(f"  per-class F1 (pooled): " +
              ", ".join(f"{keep}={mf:.2f}" for keep, (_, _, mf) in pc_order))

        print(f"\n===== {model_kind}: Train S1 / Test S2 (DEV SET) =====")
        per_seed = []
        for seed in range(N_SEEDS):
            np.random.seed(seed); torch.manual_seed(seed)
            li = li_map(train_labels)
            ytr_i = np.array([li[w] for w in ytr], dtype=np.int64)
            if model_kind == "cnn-v3":
                model = TinyCNN1D(n_in=fv.FEAT_DIM, n_classes=len(train_labels), dropout=0.3)
                train_model_v3(model, Xtr, ytr_i, seed=seed)
                pred_te, probs_te = predict_torch(model, Xte)
            else:
                flat_tr = resample_batch(Xtr, 30).reshape(len(Xtr), -1)
                flat_te = resample_batch(Xte, 30).reshape(len(Xte), -1)
                mu, sd = standardize_fit(flat_tr)
                clf = LogisticRegression(seed=seed)
                clf.fit(standardize_apply(flat_tr, mu, sd), ytr_i)
                pred_te = clf.predict(standardize_apply(flat_te, mu, sd))
                probs_te = clf.predict_proba(standardize_apply(flat_te, mu, sd))
            acc, mf1, pc, cm, top = restrict_to_s2_words(yte, pred_te, probs_te, train_labels, word_labels)
            per_seed.append(dict(seed=seed, acc=float(acc), macro_f1=float(mf1),
                                 per_class=pc, top10=[[c, i, j] for c, i, j in top]))
            print(f"  seed {seed}: S2 acc={acc:.3f} macroF1={mf1:.3f}")
        accs2 = [r["acc"] for r in per_seed]; mf1s2 = [r["macro_f1"] for r in per_seed]
        # pooled majority vote over seeds
        by_clip = defaultdict(list)
        for si, r in enumerate(per_seed):
            pass
        # rebuild per-clip preds for pooling
        li = li_map(train_labels)
        ytr_i = np.array([li[w] for w in ytr], dtype=np.int64)
        # We need per-clip preds; re-run predict per seed is expensive. Instead pool per_seed per-class F1 by averaging.
        pc_mean = {}
        for w in word_labels:
            fs = [r["per_class"][w] for r in per_seed]
            pc_mean[w] = dict(precision=float(np.mean([f["precision"] for f in fs])),
                              recall=float(np.mean([f["recall"] for f in fs])),
                              f1=float(np.mean([f["f1"] for f in fs])))
        pool_acc = float(np.mean(accs2)); pool_mf1 = float(np.mean(mf1s2))
        # top-10: average counts? just take first seed's top as representative + note
        rep_top = per_seed[0]["top10"]
        print(f"  POOLED S2 DEV: acc={pool_acc:.3f} macroF1={pool_mf1:.3f} "
              f"(mean±std over {N_SEEDS} seeds: {np.mean(accs2):.3f}±{np.std(accs2):.3f} / "
              f"{np.mean(mf1s2):.3f}±{np.std(mf1s2):.3f})")
        print(f"  chance (uniform, 10) = {1/len(word_labels):.3f}")
        print(f"  per-class F1 (seed-mean): " +
              ", ".join(f"{w}={m['f1']:.2f}" for w, m in sorted(pc_mean.items(), key=lambda kv: kv[1]['f1'])))
        print(f"  top-10 confusions (seed0 representative, count true->pred):")
        for c, i, j in rep_top:
            print(f"    {c:3d}  {i:14} -> {j}")
        # thresholds on seed0 pooled probs: we need per-clip probs; recompute seed0 probs
        np.random.seed(0); torch.manual_seed(0)
        li = li_map(train_labels)
        ytr_i = np.array([li[w] for w in ytr], dtype=np.int64)
        if model_kind == "cnn-v3":
            model = TinyCNN1D(n_in=fv.FEAT_DIM, n_classes=len(train_labels), dropout=0.3)
            train_model_v3(model, Xtr, ytr_i, seed=0)
            _, probs_te = predict_torch(model, Xte)
        else:
            flat_tr = resample_batch(Xtr, 30).reshape(len(Xtr), -1)
            flat_te = resample_batch(Xte, 30).reshape(len(Xte), -1)
            mu, sd = standardize_fit(flat_tr)
            clf = LogisticRegression(seed=0)
            clf.fit(standardize_apply(flat_tr, mu, sd), ytr_i)
            probs_te = clf.predict_proba(standardize_apply(flat_te, mu, sd))
        max_prob = probs_te.max(axis=1)
        pred_word = np.array([train_labels[p] for p in pred_te])
        yte_words = np.array(yte)
        rows_thr = []
        n = len(yte)
        for thr in (0.5, 0.7, 0.9):
            accepted = max_prob >= thr
            correct = (pred_word[accepted] == yte_words[accepted]).sum() if accepted.any() else 0
            rows_thr.append(dict(threshold=thr, n_accepted=int(accepted.sum()),
                                 coverage=float(accepted.mean()),
                                 accuracy=float(correct / max(1, accepted.sum()))))
        correct_all = (pred_word == yte_words).sum()
        rows_thr.insert(0, dict(threshold="argmax", n_accepted=n, coverage=1.0,
                                accuracy=float(correct_all / n)))
        print(f"  threshold table (S2 dev):")
        print(f"    {'threshold':>10} {'accepted':>9} {'coverage':>9} {'acc':>7}")
        for row in rows_thr:
            print(f"    {str(row['threshold']):>10} {row['n_accepted']:>9} {row['coverage']:8.1%} {row['accuracy']:7.3f}")
        print(f"  rest FPR: NOT VERIFIED (no rest clips in S2)")
        print(f"  comparison: v2 cnn-v2 S2 acc was 0.592; v3 cnn-v3 S2 acc = {pool_acc:.3f}")

        results[model_kind] = dict(
            s1_cv_acc_mean=float(np.mean(accs)), s1_cv_acc_std=float(np.std(accs)),
            s1_cv_mf1_mean=float(np.mean(mf1s)), s1_cv_mf1_std=float(np.std(mf1s)),
            s1_pooled_acc=float(acc_p), s1_pooled_mf1=float(mf1_p),
            s2_dev_acc_mean=float(np.mean(accs2)), s2_dev_acc_std=float(np.std(accs2)),
            s2_dev_mf1_mean=float(np.mean(mf1s2)), s2_dev_mf1_std=float(np.std(mf1s2)),
            s2_dev_pooled_acc=pool_acc, s2_dev_pooled_mf1=pool_mf1,
            s2_per_class=pc_mean, s2_top10=[[c, i, j] for c, i, j in rep_top],
            s2_thresholds=rows_thr,
            n_s1=int(len(Xtr)), n_s2=int(len(Xte)),
        )

    # ---- (c) hand-label check AFTER canonicalisation: fraction mirrored per label
    print(f"\n===== Mirrored-fraction after canonicalisation (S1 vs S2) =====")
    print(f"{'label':14} {'S1 clips':>8} {'S1 mirrored':>12} {'S2 clips':>8} {'S2 mirrored':>12}")
    mirror_rows = {}
    for label in sorted(set([r["label"] for r in rows_tr]) | set([r["label"] for r in rows_te])):
        s1_recs = [r for r in rows_tr if r["label"] == label]
        s2_recs = [r for r in rows_te if r["label"] == label]
        s1_mir = sum(1 for r in s1_recs if clip_mirrored(np.load(r["npy"]).astype(np.float32)))
        s2_mir = sum(1 for r in s2_recs if clip_mirrored(np.load(r["npy"]).astype(np.float32)))
        print(f"{label:14} {len(s1_recs):>8} {s1_mir}/{len(s1_recs)} ({s1_mir/max(1,len(s1_recs)):.2f})  "
              f"{len(s2_recs):>8} {s2_mir}/{len(s2_recs)} ({s2_mir/max(1,len(s2_recs)):.2f})")
        mirror_rows[label] = dict(s1_clips=len(s1_recs), s1_mirrored=s1_mir,
                                  s1_frac=(s1_mir/max(1,len(s1_recs))),
                                  s2_clips=len(s2_recs), s2_mirrored=s2_mir,
                                  s2_frac=(s2_mir/max(1,len(s2_recs))))

    report = dict(results=results, mirror=mirror_rows,
                  vocabs=dict(train=TRAIN_LABELS, s2_words=S2_WORDS),
                  n_s1=int(len(Xtr)), n_s2=int(len(Xte)),
                  note="PRELIMINARY, OPTIMISTIC. S2 is a DEV SET (used to diagnose handedness), not a clean test.")
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {OUT} (total {time.time()-t_all:.1f}s)")
    print(report["note"])


if __name__ == "__main__":
    main()
