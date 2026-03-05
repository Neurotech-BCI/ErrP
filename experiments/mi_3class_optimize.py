#!/usr/bin/env python3
"""
3-class optimization: neutral/left/right
Approach: hierarchical two-stage decoder with filter-bank Riemannian features,
robust preprocessing, confidence thresholding, and causal smoothing.
"""
import glob, json, os
import numpy as np
from mne.filter import filter_data
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ─── Feature extractors ────────────────────────────────────────────────────────

def riemann_features(X_tr, X_te, sfreq=300.0, bands=None):
    """Fit filter-bank Riemannian tangent-space features."""
    if bands is None:
        bands = [(8,12),(12,16),(16,20),(20,26),(26,32),(8,30)]
    tr_feats, te_feats = [], []
    for lo, hi in bands:
        Xtr_f = filter_data(X_tr.astype(np.float64), sfreq=sfreq, l_freq=lo, h_freq=hi, verbose="ERROR").astype(np.float32)
        Xte_f = filter_data(X_te.astype(np.float64), sfreq=sfreq, l_freq=lo, h_freq=hi, verbose="ERROR").astype(np.float32)
        cov = Covariances(estimator="oas")
        ts = TangentSpace(metric="riemann")
        tr_feats.append(ts.fit_transform(cov.fit_transform(Xtr_f)))
        te_feats.append(ts.transform(cov.transform(Xte_f)))
    return np.concatenate(tr_feats, axis=1), np.concatenate(te_feats, axis=1)


def preprocess(X, clip_q=0.01):
    """CAR + quantile clip per channel."""
    Y = X - X.mean(axis=1, keepdims=True)  # common average reference
    flat = Y.reshape(Y.shape[0]*Y.shape[1], -1)
    lo = np.quantile(flat, clip_q)
    hi = np.quantile(flat, 1 - clip_q)
    return np.clip(Y, lo, hi).astype(np.float32)


# ─── 3-class approaches ────────────────────────────────────────────────────────

def flat_3class(X_tr, y_tr, X_te, sfreq, C=0.1):
    """Flat multiclass: direct 3-class LR on filter-bank Riemannian features."""
    Ztr, Zte = riemann_features(X_tr, X_te, sfreq)
    sc = StandardScaler().fit(Ztr)
    clf = LogisticRegression(C=C, max_iter=3000, solver="lbfgs",
                             class_weight="balanced", random_state=42,
                             )
    clf.fit(sc.transform(Ztr), y_tr)
    proba = clf.predict_proba(sc.transform(Zte))
    pred = clf.classes_[np.argmax(proba, axis=1)]
    return pred, proba, clf.classes_


def hierarchical_3class(X_tr, y_tr, X_te, sfreq, neutral, C=0.1, gate_thr=0.5):
    """Two-stage: stage1=rest vs active, stage2=left vs right."""
    classes = np.unique(y_tr)
    active_codes = [c for c in classes if c != neutral]

    # Stage 1: rest vs active
    y_gate_tr = (y_tr != neutral).astype(int)
    Ztr, Zte = riemann_features(X_tr, X_te, sfreq)
    sc1 = StandardScaler().fit(Ztr)
    gate = LogisticRegression(C=C, max_iter=3000, solver="lbfgs",
                              class_weight="balanced", random_state=42)
    gate.fit(sc1.transform(Ztr), y_gate_tr)
    gate_proba = gate.predict_proba(sc1.transform(Zte))[:, 1]

    # Stage 2: left vs right (train on active only)
    mask_act = y_tr != neutral
    Ztr2, Zte2 = riemann_features(X_tr[mask_act], X_te, sfreq)
    sc2 = StandardScaler().fit(Ztr2)
    lr = LogisticRegression(C=C, max_iter=3000, solver="lbfgs",
                            class_weight="balanced", random_state=42)
    lr.fit(sc2.transform(Ztr2), y_tr[mask_act])
    lr_proba = lr.predict_proba(sc2.transform(Zte2))
    lr_pred = lr.classes_[np.argmax(lr_proba, axis=1)]
    lr_conf = np.max(lr_proba, axis=1)

    pred = np.where(gate_proba >= gate_thr, lr_pred, neutral)
    conf = np.where(gate_proba >= gate_thr, lr_conf, 1 - gate_proba)
    return pred.astype(int), conf


def smooth(pred, w=3):
    out = pred.copy()
    for i in range(len(pred)):
        window = pred[max(0, i-w+1):i+1]
        vals, cnts = np.unique(window, return_counts=True)
        out[i] = vals[np.argmax(cnts)]
    return out


# ─── Run ───────────────────────────────────────────────────────────────────────

def run_dataset(name, X, y, neutral, sfreq=300.0):
    classes = np.unique(y)
    active = [c for c in classes if c != neutral]
    has_neutral = neutral in classes

    print(f"\n{'='*60}")
    print(f"Dataset: {name}  shape={X.shape}  classes={classes.tolist()}")
    if not has_neutral:
        print("  ⚠ No neutral class found – skipping 3-class eval")
        return None

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    results = {}

    for approach, label in [("flat","Flat 3-class fbank+LR"), ("hier","Hierarchical 2-stage")]:
        bals, f1s = [], []
        for tr_idx, te_idx in cv.split(X, y):
            Xtr, ytr = preprocess(X[tr_idx]), y[tr_idx]
            Xte, yte = preprocess(X[te_idx]), y[te_idx]

            if approach == "flat":
                pred, proba, cls = flat_3class(Xtr, ytr, Xte, sfreq)
            else:
                pred, conf = hierarchical_3class(Xtr, ytr, Xte, sfreq, neutral=neutral)

            pred_smooth = smooth(pred, w=3)
            bals.append(balanced_accuracy_score(yte, pred_smooth))
            f1s.append(f1_score(yte, pred_smooth, average="macro", zero_division=0))

        print(f"  {label}: bal_acc={np.mean(bals):.4f}±{np.std(bals):.4f}  f1={np.mean(f1s):.4f}")
        results[approach] = {"bal_acc": float(np.mean(bals)), "f1": float(np.mean(f1s))}

    # Also run binary left/right for comparison
    mask_lr = np.isin(y, active)
    Xlr, ylr = X[mask_lr], y[mask_lr]
    if len(np.unique(ylr)) == 2:
        bals_bin = []
        for tr_idx, te_idx in cv.split(Xlr, ylr):
            Xtr, ytr = preprocess(Xlr[tr_idx]), ylr[tr_idx]
            Xte, yte = preprocess(Xlr[te_idx]), ylr[te_idx]
            pred, _, _ = flat_3class(Xtr, ytr, Xte, sfreq, C=0.1)
            bals_bin.append(balanced_accuracy_score(yte, pred))
        print(f"  Binary L/R only:    bal_acc={np.mean(bals_bin):.4f}±{np.std(bals_bin):.4f}")
        results["binary_lr"] = float(np.mean(bals_bin))

    return results


def main():
    pairs = []
    for d in sorted(glob.glob("data/drive_dump/BCI/*_data.npy") +
                    glob.glob("data/xavier_2026-03-03/*_windows.npy")):
        l = d.replace("_data.npy","_labels.npy").replace("_windows.npy","_labels.npy")
        if os.path.exists(l):
            pairs.append((d, l))

    all_results = {}
    for d, l in pairs:
        name = os.path.basename(d)
        X = np.load(d).astype(np.float32)
        y = np.load(l).astype(int)
        if X.ndim != 3: continue
        classes = np.unique(y)
        if len(classes) < 3:
            print(f"\n{name}: only {classes} – binary only, skipping 3-class")
            continue
        neutral = int(classes[0])  # assume smallest code = neutral/rest
        r = run_dataset(name, X, y, neutral)
        if r: all_results[name] = r

    out_path = "experiments/results/mi_3class_optimize.json"
    os.makedirs("experiments/results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
