#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import numpy as np
from mne.decoding import CSP
from mne.filter import filter_data
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class Result:
    name: str
    balanced_acc: float
    macro_f1: float
    confusion: np.ndarray


def infer_block_ids(labels: np.ndarray, windows_per_block: int) -> np.ndarray:
    labels = np.asarray(labels)
    block_ids = np.zeros_like(labels, dtype=int)
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        block_ids[idx] = np.arange(len(idx)) // windows_per_block
    return block_ids


def grouped_cv_predict(X: np.ndarray, y: np.ndarray, block_ids: np.ndarray, make_model: Callable[[], object]) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    block_ids = np.asarray(block_ids, dtype=int)
    classes = np.unique(y)

    class_blocks = {c: np.unique(block_ids[y == c]) for c in classes}
    n_splits = min(len(v) for v in class_blocks.values())
    if n_splits < 2:
        raise ValueError("Not enough blocks per class for grouped CV")

    y_pred = np.empty_like(y)
    for k in range(n_splits):
        test_mask = np.zeros(len(y), dtype=bool)
        for c in classes:
            test_mask |= (y == c) & (block_ids == class_blocks[c][k])
        train_mask = ~test_mask

        model = make_model()
        model.fit(X[train_mask], y[train_mask])
        y_pred[test_mask] = model.predict(X[test_mask])

    return y_pred


def make_riemann_lr() -> Pipeline:
    return Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("ts", TangentSpace(metric="riemann")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=42)),
    ])


def make_csp_lda() -> Pipeline:
    return Pipeline([
        ("csp", CSP(n_components=6, reg="ledoit_wolf", log=True, norm_trace=False)),
        ("scaler", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])


class FilterBankRiemannLR:
    def __init__(self, sfreq: float, bands: list[tuple[float, float]]):
        self.sfreq = sfreq
        self.bands = bands
        self.scaler_ = StandardScaler()
        self.clf_ = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=42)
        self.band_models_: list[tuple[Covariances, TangentSpace]] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        feats = []
        self.band_models_ = []
        for lo, hi in self.bands:
            Xf = filter_data(X.astype(np.float64), sfreq=self.sfreq, l_freq=lo, h_freq=hi, verbose="ERROR").astype(np.float32)
            cov = Covariances(estimator="oas")
            ts = TangentSpace(metric="riemann")
            z = ts.fit_transform(cov.fit_transform(Xf))
            self.band_models_.append((cov, ts))
            feats.append(z)
        Z = np.concatenate(feats, axis=1)
        Z = self.scaler_.fit_transform(Z)
        self.clf_.fit(Z, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        feats = []
        for (lo, hi), (cov, ts) in zip(self.bands, self.band_models_):
            Xf = filter_data(X.astype(np.float64), sfreq=self.sfreq, l_freq=lo, h_freq=hi, verbose="ERROR").astype(np.float32)
            z = ts.transform(cov.transform(Xf))
            feats.append(z)
        Z = np.concatenate(feats, axis=1)
        Z = self.scaler_.transform(Z)
        return self.clf_.predict(Z)


def evaluate(name: str, X: np.ndarray, y: np.ndarray, block_ids: np.ndarray, maker: Callable[[], object]) -> Result:
    y_pred = grouped_cv_predict(X, y, block_ids, maker)
    classes = np.unique(y)
    return Result(
        name=name,
        balanced_acc=float(balanced_accuracy_score(y, y_pred)),
        macro_f1=float(f1_score(y, y_pred, average="macro")),
        confusion=confusion_matrix(y, y_pred, labels=classes),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--sfreq", type=float, default=300.0)
    ap.add_argument("--windows-per-block", type=int, default=29)
    args = ap.parse_args()

    X = np.load(args.windows)
    y = np.load(args.labels).astype(int)
    block_ids = infer_block_ids(y, windows_per_block=args.windows_per_block)

    print(f"Loaded X={X.shape}, y={y.shape}, classes={np.unique(y).tolist()}")

    models = [
        ("Riemann+LR (baseline)", make_riemann_lr),
        ("CSP+LDA", make_csp_lda),
        ("FilterBank-Riemann+LR", lambda: FilterBankRiemannLR(args.sfreq, [(8, 12), (12, 16), (16, 20), (20, 28), (8, 30)])),
    ]

    print("\n=== 3-class (neutral/left/right) ===")
    for name, maker in models:
        r = evaluate(name, X, y, block_ids, maker)
        print(f"{r.name}: balanced_acc={r.balanced_acc:.4f}, macro_f1={r.macro_f1:.4f}")
        print(r.confusion)

    c = np.unique(y)
    if len(c) >= 3:
        neutral = int(c[0])
        others = [int(v) for v in c if int(v) != neutral]

        # left/right only
        mask_lr = np.isin(y, others)
        X_lr, y_lr, b_lr = X[mask_lr], y[mask_lr], block_ids[mask_lr]
        print("\n=== Binary (command1 vs command2) ===")
        for name, maker in models:
            r = evaluate(name, X_lr, y_lr, b_lr, maker)
            print(f"{r.name}: balanced_acc={r.balanced_acc:.4f}, macro_f1={r.macro_f1:.4f}")

        # active vs rest
        y_ar = (y != neutral).astype(int)
        print("\n=== Binary (active vs neutral) ===")
        for name, maker in models:
            r = evaluate(name, X, y_ar, block_ids, maker)
            print(f"{r.name}: balanced_acc={r.balanced_acc:.4f}, macro_f1={r.macro_f1:.4f}")


if __name__ == "__main__":
    main()
