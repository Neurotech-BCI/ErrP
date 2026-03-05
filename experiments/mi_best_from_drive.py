#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, asdict

import numpy as np
from mne.decoding import CSP
from mne.filter import filter_data
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class FilterBankRiemannLR(BaseEstimator, ClassifierMixin):
    def __init__(self, sfreq=300.0, bands=((8, 12), (12, 16), (16, 20), (20, 28), (8, 30)), C=1.0):
        self.sfreq = float(sfreq)
        self.bands = tuple((float(a), float(b)) for a, b in bands)
        self.C = float(C)
        self.band_models_ = None
        self.scaler_ = None
        self.clf_ = None

    def fit(self, X, y):
        feats = []
        self.band_models_ = []
        for lo, hi in self.bands:
            Xf = filter_data(X.astype(np.float64), sfreq=self.sfreq, l_freq=lo, h_freq=hi, verbose="ERROR").astype(np.float32)
            cov = Covariances(estimator="oas")
            ts = TangentSpace(metric="riemann")
            Z = ts.fit_transform(cov.fit_transform(Xf))
            self.band_models_.append((cov, ts, lo, hi))
            feats.append(Z)
        Z = np.concatenate(feats, axis=1)
        self.scaler_ = StandardScaler().fit(Z)
        Zs = self.scaler_.transform(Z)
        self.clf_ = LogisticRegression(C=self.C, max_iter=3000, solver="lbfgs", class_weight="balanced", random_state=42)
        self.clf_.fit(Zs, y)
        self.classes_ = self.clf_.classes_
        return self

    def _feat(self, X):
        feats = []
        for cov, ts, lo, hi in self.band_models_:
            Xf = filter_data(X.astype(np.float64), sfreq=self.sfreq, l_freq=lo, h_freq=hi, verbose="ERROR").astype(np.float32)
            feats.append(ts.transform(cov.transform(Xf)))
        return np.concatenate(feats, axis=1)

    def predict(self, X):
        Z = self._feat(X)
        return self.clf_.predict(self.scaler_.transform(Z))


@dataclass
class Score:
    dataset: str
    model: str
    bal_acc_mean: float
    bal_acc_std: float
    f1_mean: float
    f1_std: float
    n: int
    channels: int
    samples: int


def eval_model(X, y, model, cv):
    bals, f1s = [], []
    for tr, te in cv.split(X, y):
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        bals.append(balanced_accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
    return float(np.mean(bals)), float(np.std(bals)), float(np.mean(f1s)), float(np.std(f1s))


def model_bank(n_ch):
    return {
        "riemann_lr": Pipeline([
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, solver="lbfgs", class_weight="balanced", random_state=42)),
        ]),
        "riemann_lr_C0.1": Pipeline([
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=3000, solver="lbfgs", class_weight="balanced", random_state=42)),
        ]),
        "csp4_lda": Pipeline([
            ("csp", CSP(n_components=min(4, n_ch), reg="ledoit_wolf", log=True, norm_trace=False)),
            ("sc", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]),
        "csp6_lda": Pipeline([
            ("csp", CSP(n_components=min(6, n_ch), reg="ledoit_wolf", log=True, norm_trace=False)),
            ("sc", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]),
        "fbank_riemann": FilterBankRiemannLR(),
        "fbank_riemann_wide": FilterBankRiemannLR(bands=((6,10),(10,14),(14,18),(18,24),(24,32),(8,30))),
    }


def main():
    pairs = []
    for d in sorted(glob.glob("data/drive_dump/BCI/*_data.npy")):
        l = d.replace("_data.npy", "_labels.npy")
        if os.path.exists(l):
            pairs.append((d, l))

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    all_scores: list[Score] = []

    for d, l in pairs:
        X = np.load(d).astype(np.float32)
        y = np.load(l).astype(int)
        if X.ndim != 3 or len(np.unique(y)) != 2:
            continue
        name = os.path.basename(d)
        print(f"\n=== {name}  X={X.shape} classes={np.unique(y).tolist()} ===")
        models = model_bank(X.shape[1])
        best = None
        for mname, model in models.items():
            ba_m, ba_s, f1_m, f1_s = eval_model(X, y, model, cv)
            rec = Score(name, mname, ba_m, ba_s, f1_m, f1_s, X.shape[0], X.shape[1], X.shape[2])
            all_scores.append(rec)
            print(f"{mname:20s} bal_acc={ba_m:.4f}±{ba_s:.4f}  f1={f1_m:.4f}±{f1_s:.4f}")
            if best is None or rec.bal_acc_mean > best.bal_acc_mean:
                best = rec
        print(f"BEST: {best.model} bal_acc={best.bal_acc_mean:.4f}")

    # aggregate by model
    agg = {}
    for r in all_scores:
        agg.setdefault(r.model, []).append(r.bal_acc_mean)
    agg_sorted = sorted(((k, float(np.mean(v)), len(v)) for k, v in agg.items()), key=lambda x: x[1], reverse=True)

    out = {
        "scores": [asdict(s) for s in all_scores],
        "aggregate_bal_acc_mean": [{"model": k, "mean_bal_acc": m, "datasets": n} for k, m, n in agg_sorted],
    }
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/mi_best_from_drive.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Aggregate mean balanced accuracy across datasets ===")
    for k, m, n in agg_sorted:
        print(f"{k:20s} {m:.4f}  (n={n})")


if __name__ == "__main__":
    main()
