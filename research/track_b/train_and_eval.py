"""Step 4: Train 1D-CNN + small GRU on CPU for ISL word classification.

Data: research/openhands_pilot/extracted_poses/*.npy  (122 clips, 6 words,
MediaPipe Holistic (T,75,3)). Features via track_b/features.extract_features
-> (60, 98). Class-stratified split by clip, fixed seed. No signer IDs in the
filenames, so signer leakage cannot be excluded.

Reports: overall accuracy, macro-F1, per-class F1, top-10 confusions,
trivial-baseline accuracy, CPU latency per clip, model file size.
"""
import glob, os, re, time, json
import numpy as np
import torch
import torch.nn as nn
from features import extract_features

POSES_DIR = "/mnt/Drive-D/Codes/signsage/research/openhands_pilot/extracted_poses"
SEED = 42
N_CLASSES_USED = None  # set after loading
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_data():
    X, y, words, files = [], [], [], []
    for fp in sorted(glob.glob(os.path.join(POSES_DIR, "*.npy"))):
        word = os.path.basename(fp).split("__")[0]          # friend__INCLUDE__... -> friend
        lm = np.load(fp).astype(np.float32)                # (T,75,3)
        feat = extract_features(lm, layout="holistic")     # (60,98)
        X.append(feat); y.append(word); words.append(word); files.append(fp)
    X = np.stack(X)                                        # (N,60,98)
    labels = sorted(set(y))
    y = np.array([labels.index(w) for w in y], dtype=np.int64)
    return X, y, labels, files


def stratified_split(X, y, test_frac=0.2, seed=SEED):
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac)))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


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
    # trivial baseline: majority class
    counts = np.bincount(y_true, minlength=n_classes)
    baseline = counts.max() / max(1, len(y_true))
    # top-10 off-diagonal confusions
    off = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                off.append((int(cm[i, j]), i, j))
    off.sort(reverse=True)
    return acc, macro_f1, per_class, cm, baseline, off[:10]


class TinyCNN1D(nn.Module):
    def __init__(self, n_in=98, n_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_in, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):                     # x: (B, 60, 98) -> (B, 98, 60)
        x = x.transpose(1, 2)
        x = self.net(x).squeeze(-1)
        return self.head(x)


class TinyGRU(nn.Module):
    def __init__(self, n_in=98, n_classes=6, hidden=32):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, 1, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):                     # x: (B, 60, 98)
        _, h = self.gru(x)
        return self.head(h.squeeze(0))


def augment(x):                               # x: (B, 60, 98)
    B = x.shape[0]
    xp = x.reshape(B, 60, 49, 2).copy()
    for b in range(B):
        if np.random.rand() < 0.5:
            xp[b] *= np.random.uniform(0.95, 1.05)          # scale
        if np.random.rand() < 0.5:
            xp[b] += np.random.uniform(-0.03, 0.03)         # shift
        if np.random.rand() < 0.5:
            ang = np.deg2rad(np.random.uniform(-8, 8))      # rotation
            c, s = np.cos(ang), np.sin(ang)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)
            xp[b] = xp[b] @ R.T
        if np.random.rand() < 0.5:                          # time warp
            nw = int(60 * np.random.uniform(0.9, 1.1))
            nw = max(2, min(120, nw))
            idx = np.linspace(0, 59, nw).round().astype(int)
            warped = xp[b][idx]                             # (nw,49,2)
            idx2 = np.linspace(0, nw - 1, 60).round().astype(int)
            xp[b] = warped[idx2]
    return xp.reshape(B, 60, 98)


def train_model(model, Xtr, ytr, epochs=60, lr=1e-3, batch=16, verbose=False):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    N = len(Xtr)
    for ep in range(epochs):
        perm = np.random.permutation(N)
        total_loss = 0.0
        for s in range(0, N, batch):
            ib = perm[s:s + batch]
            xb = torch.from_numpy(augment(Xtr[ib])).float()
            yb = torch.from_numpy(ytr[ib]).long()
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(ib)
        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  loss={total_loss / N:.4f}")
    model.eval()


def evaluate(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float())
    pred = logits.argmax(1).numpy()
    return pred


def infer_latency(model, X, n_repeats=5):
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            for i in range(len(X)):
                t0 = time.perf_counter()
                _ = model(torch.from_numpy(X[i:i + 1]).float())
                times.append(time.perf_counter() - t0)
    return float(np.mean(times)) * 1000.0  # ms


def main():
    X, y, labels, files = load_data()
    n_classes = len(labels)
    print(f"Loaded {len(X)} clips, {n_classes} classes: {labels}")
    print(f"Class counts: {dict(zip(*np.unique(y, return_counts=True)))}")

    tr_idx, te_idx = stratified_split(X, y)
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]
    print(f"Train {len(Xtr)}  Test {len(Xte)}  (seed {SEED})")

    results = {}
    for name, Model in [("1D-CNN", TinyCNN1D), ("GRU", TinyGRU)]:
        print(f"\n=== {name} ===")
        model = Model(n_in=98, n_classes=n_classes)
        train_model(model, Xtr, ytr, verbose=True)
        pred_tr = evaluate(model, Xtr)
        pred_te = evaluate(model, Xte)
        acc_tr, mf1_tr, pc_tr, cm_tr, base_tr, off_tr = accuracy_metrics(ytr, pred_tr, n_classes)
        acc_te, mf1_te, pc_te, cm_te, base_te, off_te = accuracy_metrics(yte, pred_te, n_classes)
        latency = infer_latency(model, Xte)
        # model size
        path = f"/tmp/track_b_{name.lower().replace(' ', '_')}.pt"
        torch.save(model.state_dict(), path)
        msize = os.path.getsize(path) / 1e6
        print(f"Train acc={acc_tr:.3f} macroF1={mf1_tr:.3f}  | "
              f"Test acc={acc_te:.3f} macroF1={mf1_te:.3f}  baseline={base_te:.3f}")
        print(f"Per-class F1 (test):")
        for c in range(n_classes):
            print(f"  {labels[c]:12} P={pc_te[c][0]:.2f} R={pc_te[c][1]:.2f} F1={pc_te[c][2]:.2f}")
        print(f"Top confusions (true->pred, count):")
        for cnt, i, j in off_te:
            print(f"  {labels[i]:12} -> {labels[j]:12}  {cnt}")
        print(f"CPU latency/clip: {latency:.2f} ms   model size: {msize:.3f} MB")
        results[name] = dict(
            train_acc=float(acc_tr), train_macro_f1=float(mf1_tr),
            test_acc=float(acc_te), test_macro_f1=float(mf1_te),
            baseline=float(base_te), latency_ms=float(latency), model_size_mb=float(msize),
            per_class={labels[c]: dict(precision=float(pc_te[c][0]), recall=float(pc_te[c][1]), f1=float(pc_te[c][2])) for c in range(n_classes)},
            confusions=[[int(v) for v in row] for row in cm_te],
            labels=labels,
        )

    with open("/mnt/D/Codes/signsage/research/track_b/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nWrote research/track_b/results.json")


if __name__ == "__main__":
    main()
