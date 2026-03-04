from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mne.filter import filter_data, notch_filter
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP

from config import EEGConfig, MentalCommandModelConfig


def filter_window(X: np.ndarray, eeg_cfg: EEGConfig, sfreq: float) -> np.ndarray:
    """Bandpass/notch filter for one or many windows.

    X shape: (n_ch, n_samp) or (n_win, n_ch, n_samp).
    """
    Xf = np.asarray(X, dtype=np.float64, order="C")
    if eeg_cfg.notch is not None:
        Xf = notch_filter(Xf, Fs=sfreq, freqs=[float(eeg_cfg.notch)], verbose="ERROR")
    Xf = filter_data(
        Xf,
        sfreq=float(sfreq),
        l_freq=float(eeg_cfg.l_freq),
        h_freq=float(eeg_cfg.h_freq),
        verbose="ERROR",
    )
    return Xf.astype(np.float32, copy=False)


def split_windows(
    block: np.ndarray,
    sfreq: float,
    window_s: float,
    step_s: float,
) -> np.ndarray:
    """Split one continuous block into fixed windows.

    block shape: (n_ch, n_samples).
    returns shape: (n_windows, n_ch, n_window_samples).
    """
    n_ch, n_samples = block.shape
    w = int(round(float(window_s) * float(sfreq)))
    h = int(round(float(step_s) * float(sfreq)))
    if w <= 0 or h <= 0:
        raise ValueError("window_s and step_s must produce at least one sample")
    if n_samples < w:
        return np.empty((0, n_ch, w), dtype=np.float32)
    starts = np.arange(0, n_samples - w + 1, h, dtype=int)
    out = np.empty((len(starts), n_ch, w), dtype=np.float32)
    for i, s in enumerate(starts):
        out[i] = block[:, s : s + w]
    return out


def make_mental_command_classifier(model_cfg: MentalCommandModelConfig) -> Pipeline:
    backend = str(getattr(model_cfg, "backend", "riemann_lr")).lower()

    if backend == "csp_lda":
        return Pipeline(
            [
                ("csp", CSP(n_components=int(model_cfg.csp_components), reg="ledoit_wolf", log=True, norm_trace=False)),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            ]
        )

    return Pipeline(
        [
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    C=float(model_cfg.C),
                    solver="lbfgs",
                    max_iter=int(model_cfg.max_iter),
                    class_weight=model_cfg.class_weight,
                    random_state=42,
                ),
            ),
        ]
    )


@dataclass
class MCQuality:
    balanced_accuracy: float
    macro_f1: float
    n_samples: int
    n_per_class: dict[str, int]
    per_class_accuracy: dict[str, float]


def evaluate_cv_quality(
    X: np.ndarray,
    y: np.ndarray,
    block_ids: np.ndarray,
    model_cfg: MentalCommandModelConfig,
    cv_splits_max: int,
) -> MCQuality:
    y = np.asarray(y, dtype=int)
    block_ids = np.asarray(block_ids, dtype=int)
    if y.shape[0] != X.shape[0] or block_ids.shape[0] != X.shape[0]:
        raise ValueError("X, y, and block_ids must have the same first dimension")

    classes, counts = np.unique(y, return_counts=True)
    # Build class-local block ids so each fold can hold out exactly one 8s block per class.
    class_block_ids = {}
    for c in classes:
        class_mask = y == int(c)
        uniq = np.unique(block_ids[class_mask])
        class_block_ids[int(c)] = np.sort(uniq)

    min_block_count = min(len(v) for v in class_block_ids.values())
    n_splits = min(int(cv_splits_max), int(min_block_count))
    if n_splits < 2:
        block_counts = {int(c): len(class_block_ids[int(c)]) for c in classes}
        raise ValueError(
            f"Not enough registration blocks for grouped CV: per-class blocks={block_counts}"
        )

    clf = make_mental_command_classifier(model_cfg)
    y_pred = np.empty_like(y)

    # Fold k: hold out one class-specific registration block for each class.
    for k in range(n_splits):
        test_mask = np.zeros(y.shape[0], dtype=bool)
        for c in classes:
            held_out_block = class_block_ids[int(c)][k]
            test_mask |= (y == int(c)) & (block_ids == int(held_out_block))
        train_mask = ~test_mask
        if not np.any(test_mask) or not np.any(train_mask):
            raise ValueError(f"Invalid CV split at fold {k}")

        fold_clf = clone(clf)
        fold_clf.fit(X[train_mask], y[train_mask])
        y_pred[test_mask] = fold_clf.predict(X[test_mask])

    cm = confusion_matrix(y, y_pred, labels=classes)
    denom = np.sum(cm, axis=1)
    per_class_acc = {}
    for i, c in enumerate(classes):
        per_class_acc[str(int(c))] = float(cm[i, i] / denom[i]) if denom[i] > 0 else 0.0

    return MCQuality(
        balanced_accuracy=float(balanced_accuracy_score(y, y_pred)),
        macro_f1=float(f1_score(y, y_pred, average="macro")),
        n_samples=int(len(y)),
        n_per_class={str(int(c)): int(n) for c, n in zip(classes, counts)},
        per_class_accuracy=per_class_acc,
    )


class EMAProbSmoother:
    def __init__(self, alpha: float, n_classes: int):
        self.alpha = float(alpha)
        self.n_classes = int(n_classes)
        self._state: np.ndarray | None = None

    def update(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        if p.shape != (self.n_classes,):
            raise ValueError(f"Expected probs shape ({self.n_classes},), got {p.shape}")
        if self._state is None:
            self._state = p.copy()
        else:
            self._state = (1.0 - self.alpha) * self._state + self.alpha * p
        s = float(np.sum(self._state))
        if s > 0:
            self._state = self._state / s
        return self._state.copy()
