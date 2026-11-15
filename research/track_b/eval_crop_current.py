"""Step 2: evaluate the CURRENT final model (export_final8_s4/model.pt)
on uncropped vs cropped vs jittered 9-class clips.

CAVEAT (printed and stored): S1-S4 are in this model's training data —
these are robustness numbers, NOT a generalisation estimate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import crop_utils as cu
import features_v3 as fv
from eval_v3 import TinyCNN1D

MODEL = ROOT / "export_final8_s4" / "model.pt"
LABELS = cu.LABELS
THRESHOLD = 0.7


def load_model():
    model = TinyCNN1D(n_in=fv.FEAT_DIM, n_classes=len(LABELS), dropout=0.3)
    state = torch.load(MODEL, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model, arr):
    feats = fv.extract_features(arr.astype(np.float32), layout="signsage")
    with torch.no_grad():
        logits = model(torch.from_numpy(feats[None]).float())[0]
    probs = torch.softmax(logits, dim=-1).numpy()
    return int(np.argmax(probs)), float(np.max(probs)), probs


def evaluate(model, rows):
    li = {w: i for i, w in enumerate(LABELS)}
    n = correct = accepted = accepted_correct = 0
    macro_f1_num = {w: [0, 0, 0] for w in LABELS}  # tp, fp, fn
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    maxprobs = []
    for r in rows:
        arr = np.load(r["npy"])
        pred, mp, _ = predict(model, arr)
        true = li[r["label"]]
        n += 1
        maxprobs.append(mp)
        confusion[true, pred] += 1
        correct += int(pred == true)
        acc = mp >= THRESHOLD
        accepted += int(acc)
        accepted_correct += int(acc and pred == true)
        for w, cfg in (("pred", pred), ("true", true)):
            pass
        for i in range(len(LABELS)):
            if true == i and pred == i:
                macro_f1_num[LABELS[i]][0] += 1
            elif true != i and pred == i:
                macro_f1_num[LABELS[i]][1] += 1
            elif true == i and pred != i:
                macro_f1_num[LABELS[i]][2] += 1
    f1s = []
    f1s_present = []
    for w in LABELS:
        tp, fp, fn = macro_f1_num[w]
        p = tp / max(1, tp + fp)
        r_ = tp / max(1, tp + fn)
        f1 = 2 * p * r_ / max(1e-9, p + r_)
        f1s.append(f1)
        if tp + fn > 0:  # class present in this set
            f1s_present.append(f1)
    return {
        "n": n,
        "top1": correct / n,
        "accepted_rate": accepted / n,
        "acc_given_accepted": (accepted_correct / accepted) if accepted else None,
        "macro_f1": float(np.mean(f1s)),
        "macro_f1_present_classes": float(np.mean(f1s_present)) if f1s_present else None,
        "mean_maxprob": float(np.mean(maxprobs)),
        "confusion": confusion.tolist(),
    }


def session_of(r):
    stem = Path(r["npy"]).stem
    for s in ("S1", "S2", "S3", "S4"):
        if f"__{s}__" in stem:
            return s
    return Path(r.get("session", "?")).name


def main():
    model = load_model()
    sources = {
        "uncropped": [
            r for s in ("s1_converted", "s2_converted", "s3_converted", "s4_converted")
            for r in cu.word_rows(cu.load_manifest(cu.DATA / s / "manifest.csv"))
        ],
        "cropped": cu.load_manifest(cu.DATA / "cropped" / "manifest.csv"),
        "jittered": cu.load_manifest(cu.DATA / "cropped_jitter" / "manifest.csv"),
    }
    out = {
        "model": str(MODEL),
        "caveat": (
            "S1-S4 are in this model's training data; this is a robustness test, "
            "NOT a generalisation estimate."
        ),
        "threshold": THRESHOLD,
        "variants": {},
    }
    print("CAVEAT: S1-S4 are in this model's training data — robustness test, NOT a generalisation estimate.")
    print(f"accept = softmax max prob >= {THRESHOLD}; accepted rate reported separately from top-1.\n")
    for variant, rows in sources.items():
        by_session = {}
        for s in ("S1", "S2", "S3", "S4"):
            sr = [r for r in rows if session_of(r) == s]
            if sr:
                by_session[s] = evaluate(model, sr)
        overall = evaluate(model, rows)
        out["variants"][variant] = {"overall": overall, "by_session": by_session}
        print(f"=== {variant} (overall: n={overall['n']}) ===")
        print(f"  top-1 {overall['top1']:.3f} | accepted {overall['accepted_rate']:.3f} | "
              f"acc|accepted {overall['acc_given_accepted'] if overall['acc_given_accepted'] is not None else float('nan'):.3f} | "
              f"macro-F1(all9) {overall['macro_f1']:.3f} | mean maxprob {overall['mean_maxprob']:.3f}")
        for s, m in by_session.items():
            print(f"  {s}: n={m['n']:3d} top-1 {m['top1']:.3f} | accepted {m['accepted_rate']:.3f} | "
                  f"acc|accepted {m['acc_given_accepted'] if m['acc_given_accepted'] is not None else float('nan'):.3f} | macro-F1 {m['macro_f1']:.3f}")
        print()

    # confusion pairs for the cropped set (overall)
    conf = np.array(out["variants"]["cropped"]["overall"]["confusion"])
    pairs = []
    for i in range(9):
        for j in range(9):
            if i != j and conf[i, j] > 0:
                pairs.append((int(conf[i, j]), LABELS[i], LABELS[j]))
    pairs.sort(reverse=True)
    out["cropped_confusion_pairs"] = [
        {"count": c, "true": t, "pred": p} for c, t, p in pairs
    ]
    print("Cropped-set confusion pairs (true -> pred):")
    for c, t, p in pairs:
        print(f"  {c:2d}x  {t} -> {p}")

    with open(ROOT / "results_crop_current.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {ROOT / 'results_crop_current.json'}")


if __name__ == "__main__":
    main()
