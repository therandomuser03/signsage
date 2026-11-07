"""Track B: recorder clips -> 98-dim features -> 5-fold take CV.

1D-CNN primary, tiny GRU as comparison. Two vocabularies:
  CORE  = 10 core words + rest   (11 classes)
  FULL  = all non-excluded labels + rest (29 classes)
5 seeds, reported as mean +/- std over seeds. Offline, CPU only.
Nothing is exported or integrated into the app.
"""
import csv, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import extract_features
from train_and_eval import (
    TinyCNN1D, TinyGRU, train_model, evaluate, accuracy_metrics, augment,
)

MANIFEST = Path("data/s1_converted/manifest.csv")
CORE = ["hello", "thank you", "please", "yes", "no", "help", "water", "you", "me", "where"]
REST = "rest"
N_SEEDS = 5
N_FOLDS = 5
EPOCHS = 60
OUT = Path("results_track_b.json")
MODELS = [("1D-CNN", TinyCNN1D), ("GRU", TinyGRU)]


def load_data():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    X, y, meta = [], [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        arr = np.load(r["npy"]).astype(np.float32)
        feat = extract_features(arr, layout="signsage")  # (60, 98)
        assert feat.shape == (60, 98), feat.shape
        X.append(feat)
        y.append(r["label"])
        meta.append(r)
        if (i + 1) % 50 == 0:
            print(f"  features {i + 1}/{len(rows)}")
    X = np.stack(X)
    labels = sorted(set(y))
    li = {w: i for i, w in enumerate(labels)}
    yi = np.array([li[w] for w in y], dtype=np.int64)
    print(f"features: {len(X)} clips x {X.shape[1]}x{X.shape[2]} in {time.time() - t0:.1f}s, "
          f"{len(labels)} labels: {labels}")
    return X, yi, labels, meta


def take_folds(y, labels, meta):
    """Fold id per clip: all takes of one label are split across folds, never mixed."""
    fold = np.zeros(len(y), dtype=int)
    for w in labels:
        idx = np.where(y == labels.index(w))[0]
        idx = idx[np.argsort([int(meta[i]["take"]) for i in idx])]
        for k, part in enumerate(np.array_split(idx, N_FOLDS)):
            fold[part] = k
    return fold


def subset(y, labels, vocab):
    keep = [w for w in vocab if w in set(labels)]
    idl = {w: i for i, w in enumerate(labels)}
    keep_idx = [idl[w] for w in keep]
    mask = np.isin(y, keep_idx)
    remap = {old: new for new, old in enumerate(keep_idx)}
    y2 = np.array([remap[int(v)] for v in y[mask]], dtype=np.int64)
    return mask, y2, keep


def cv_run(X, y, labels, meta, vocab, model_cls, model_name, seed):
    """One seed: N_FOLDS take-grouped folds. Returns fold metrics + OOF arrays."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    mask, y2, keep = subset(y, labels, vocab)
    Xv, yv = X[mask], y2
    folds = take_folds(yv, keep, [meta[i] for i in np.where(mask)[0]])
    n_classes = len(keep)

    fold_metrics, oof = [], []
    meta_v = [meta[i] for i in np.where(mask)[0]]
    for k in range(N_FOLDS):
        tr = np.where(folds != k)[0]
        te = np.where(folds == k)[0]
        model = model_cls(n_in=98, n_classes=n_classes)
        train_model(model, Xv[tr], yv[tr], epochs=EPOCHS)
        pred_te = evaluate(model, Xv[te])
        acc, mf1, pc, cm, baseline, off = accuracy_metrics(yv[te], pred_te, n_classes)
        probs = None
        with torch.no_grad():
            logits = model(torch.from_numpy(Xv[te]).float())
            probs = torch.softmax(logits, 1).numpy()
        fold_metrics.append(dict(
            fold=k, n_tr=len(tr), n_te=len(te), acc=float(acc), macro_f1=float(mf1),
            baseline=float(baseline),
            per_class={keep[c]: dict(precision=float(pc[c][0]), recall=float(pc[c][1]), f1=float(pc[c][2]))
                       for c in range(n_classes)},
            confusions=[[int(v) for v in row] for row in cm],
            labels=list(keep),
        ))
        for j, idx in enumerate(te):
            oof.append(dict(clip=int(np.where(mask)[0][idx]), fold=k, seed=seed,
                            true=int(yv[idx]), pred=int(pred_te[j]), max_prob=float(probs[j].max())))
    return dict(model=model_name, vocab=list(keep), n_classes=n_classes,
                seed=seed, folds=fold_metrics, oof=oof)


def agg(seed_runs):
    """Mean +/- std over seeds of per-seed mean CV accuracy / macro-F1."""
    out = {}
    for model in ("1D-CNN", "GRU"):
        for vocab_key in ("CORE", "FULL"):
            accs = [r["cv_mean_acc"] for r in seed_runs
                    if r["model"] == model and r["vocab_key"] == vocab_key]
            mf1s = [r["cv_mean_macro_f1"] for r in seed_runs
                    if r["model"] == model and r["vocab_key"] == vocab_key]
            out[f"{model} {vocab_key}"] = dict(
                acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
                mf1_mean=float(np.mean(mf1s)), mf1_std=float(np.std(mf1s)),
                per_seed=[dict(seed=r["seed"], acc=float(r["cv_mean_acc"]),
                               macro_f1=float(r["cv_mean_macro_f1"]))
                          for r in seed_runs
                          if r["model"] == model and r["vocab_key"] == vocab_key],
            )
    return out


def pooled_confusion(all_oof, n_classes):
    """Majority vote over the 5 seeds' OOF predictions -> one prediction per clip."""
    by_clip = defaultdict(list)
    for o in all_oof:
        by_clip[o["clip"]].append(o)
    y_true, y_pred = [], []
    for clip, obs in by_clip.items():
        votes = defaultdict(int)
        for o in obs:
            votes[o["pred"]] += 1
        best = max(votes.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        y_true.append(obs[0]["true"])
        y_pred.append(best)
    acc, mf1, pc, cm, baseline, off = accuracy_metrics(np.array(y_true), np.array(y_pred), n_classes)
    return dict(acc=float(acc), macro_f1=float(mf1), baseline=float(baseline),
                per_class={c: dict(precision=float(pc[c][0]), recall=float(pc[c][1]), f1=float(pc[c][2]))
                           for c in range(n_classes)},
                confusions=[[int(v) for v in row] for row in cm],
                top10=[[int(c), i, j] for c, i, j in off],
                n=int(len(y_true)))


def threshold_table(oof_seed0, labels, n_classes, thresholds=(0.5, 0.7, 0.9)):
    """OOF predictions of a single seed: each clip predicted exactly once."""
    rows = []
    n = len(oof_seed0)
    rest_idx = labels.index(REST)
    rest_obs = [o for o in oof_seed0 if o["true"] == rest_idx]
    for t in thresholds:
        accepted = [o for o in oof_seed0 if o["max_prob"] >= t]
        correct = sum(1 for o in accepted if o["pred"] == o["true"])
        acc = correct / len(accepted) if accepted else 0.0
        cov = len(accepted) / n
        rest_as_word = sum(1 for o in rest_obs if o["max_prob"] >= t and o["pred"] != rest_idx)
        rest_as_word_rate = rest_as_word / len(rest_obs) if rest_obs else 0.0
        rows.append(dict(threshold=t, n_accepted=len(accepted), coverage=float(cov),
                         accuracy=float(acc),
                         rest_clips=len(rest_obs),
                         rest_wrong_as_word=int(rest_as_word),
                         rest_wrong_as_word_rate=float(rest_as_word_rate)))
    # argmax (no reject)
    correct = sum(1 for o in oof_seed0 if o["pred"] == o["true"])
    rest_as_word = sum(1 for o in rest_obs if o["pred"] != rest_idx)
    rows.insert(0, dict(threshold="argmax", n_accepted=n, coverage=1.0,
                        accuracy=float(correct / n),
                        rest_clips=len(rest_obs),
                        rest_wrong_as_word=int(rest_as_word),
                        rest_wrong_as_word_rate=float(rest_as_word / len(rest_obs))))
    return rows


def main():
    X, y, labels, meta = load_data()
    vocabs = {"CORE": CORE + [REST], "FULL": labels}  # labels already contains rest
    all_results, all_oof = [], []
    t_all = time.time()
    for seed in range(N_SEEDS):
        print(f"\n===== SEED {seed} =====")
        for vk, vocab in vocabs.items():
            for model_name, model_cls in MODELS:
                r = cv_run(X, y, labels, meta, vocab, model_cls, model_name, seed)
                cm_accs = [f["acc"] for f in r["folds"]]
                cm_mf1 = [f["macro_f1"] for f in r["folds"]]
                r["cv_mean_acc"] = float(np.mean(cm_accs))
                r["cv_mean_macro_f1"] = float(np.mean(cm_mf1))
                print(f"[{model_name:7} {vk:4} seed {seed}] "
                      f"CV acc={r['cv_mean_acc']:.3f} macroF1={r['cv_mean_macro_f1']:.3f}  "
                      f"per-fold acc={[round(a, 3) for a in cm_accs]}")
                all_results.append({**r, "vocab_key": vk, "seed": seed})
                all_oof.extend(r["oof"])
    summary = agg(all_results)

    # pooled (majority vote over seeds) class tables for the 1D-CNN
    pooled = {}
    # oof records don't carry vocab; rebuild per-vocab pools from all_results
    oof_by_vocab = defaultdict(list)
    for r in all_results:
        for o in r["oof"]:
            oof_by_vocab[r["vocab_key"]].append(o)
    for vk in vocabs:
        lab_list = vocabs[vk]
        pooled[vk] = pooled_confusion(oof_by_vocab[vk], len(lab_list))
        # map indices back to label names
        pc = {lab_list[c]: v for c, v in pooled[vk]["per_class"].items()}
        pooled[vk]["per_class"] = pc
        conf = pooled[vk]["confusions"]
        top = []
        for c, i, j in pooled[vk]["top10"]:
            top.append([c, lab_list[i], lab_list[j]])
        pooled[vk]["top10"] = top

    # threshold analysis on seed 0 OOF, 1D-CNN CORE
    oof_s0_core = [o for r in all_results
                   for o in r["oof"] if r["seed"] == 0 and r["model"] == "1D-CNN" and r["vocab_key"] == "CORE"]
    thresholds = threshold_table(oof_s0_core, CORE + [REST], len(CORE) + 1)

    report = dict(
        summary=summary,
        pooled=pooled,
        thresholds=thresholds,
        vocabs=dict(CORE=CORE + [REST], FULL=labels),
        n_clips=int(len(X)),
        labels=labels,
        note="PRELIMINARY, OPTIMISTIC: 5-fold take CV on S1 only (same session, same position/lighting). "
             "Not a final result.",
    )
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {OUT} (total {time.time() - t_all:.1f}s)")

    # ---------- print tables ----------
    print("\n" + "=" * 70)
    print("SUMMARY: mean +/- std over 5 seeds of 5-fold take CV (S1)")
    print("=" * 70)
    for key, s in summary.items():
        model, vk = key.split(" ")
        print(f"{model:7} {vk:4}  acc={s['acc_mean']:.3f} +/- {s['acc_std']:.3f}  "
              f"macroF1={s['mf1_mean']:.3f} +/- {s['mf1_std']:.3f}  "
              f"per_seed={[round(x['acc'], 3) for x in s['per_seed']]}")

    for vk in ("CORE", "FULL"):
        p = pooled[vk]
        print(f"\n--- POOLED CV class table: 1D-CNN {vk} (majority vote over 5 seeds, n={p['n']}) ---")
        print(f"acc={p['acc']:.3f} macroF1={p['macro_f1']:.3f} baseline(majority)={p['baseline']:.3f} "
              f"chance(uniform)={1 / len(p['per_class']):.3f}")
        order = sorted(p["per_class"].items(), key=lambda kv: kv[1]["f1"])
        print(f"{'class':14} {'P':>6} {'R':>6} {'F1':>6}")
        for name, m in order:
            mark = "  <-- candidate drop (F1<0.80)" if (m["f1"] < 0.80 and name != REST) else ""
            print(f"{name:14} {m['precision']:6.2f} {m['recall']:6.2f} {m['f1']:6.2f}{mark}")
        print(f"Top-10 confusions (count, true -> pred):")
        for c, i, j in p["top10"]:
            print(f"  {c:3d}  {i:14} -> {j:14}")

    print("\n--- Unknown handling (max-softmax threshold), 1D-CNN CORE, seed-0 OOF ---")
    print(f"{'threshold':>10} {'accepted':>9} {'coverage':>9} {'acc':>7} {'rest_clips':>10} "
          f"{'rest->word':>11} {'rate':>7}")
    for row in thresholds:
        print(f"{str(row['threshold']):>10} {row['n_accepted']:>9} {row['coverage']:8.1%} {row['accuracy']:7.3f} "
              f"{row['rest_clips']:>10} {row['rest_wrong_as_word']:>11} {row['rest_wrong_as_word_rate']:7.1%}")

    print("\n" + report["note"])


if __name__ == "__main__":
    main()
