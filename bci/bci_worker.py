# bci_worker.py  –  BCI utility module
#
# Classifier builders, training helpers, epoch filtering, and raw CSV recording.
# Imported by psychopy_task.py (the single-process entry point).
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from mne.decoding import CSP
from mne.filter import filter_data, notch_filter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_score

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from config import EEGConfig, ModelConfig


# ----------------------------------------------------------------
# Classifier builders
# ----------------------------------------------------------------


def _make_classifier_riemann_selectC() -> Pipeline:
    """Calibration-only: selects C via internal CV once."""
    return Pipeline(
        [
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegressionCV(
                    Cs=[0.01, 0.1, 1.0, 10.0],
                    cv=3,
                    solver="liblinear",
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def _make_classifier_riemann_fixedC(C: float) -> Pipeline:
    """Online: fixed C (no CV inside fit)."""
    return Pipeline(
        [
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    C=float(C),
                    solver="liblinear",
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def _make_classifier_csp_selectC(n_csp_components: int) -> Pipeline:
    csp = CSP(n_components=n_csp_components, reg="ledoit_wolf", log=True, norm_trace=False)
    clf = LogisticRegressionCV(
        Cs=[0.01, 0.1, 1.0, 10.0],
        cv=3,
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline([("csp", csp), ("scaler", StandardScaler()), ("clf", clf)])


def _make_classifier_csp_fixedC(n_csp_components: int, C: float) -> Pipeline:
    csp = CSP(n_components=n_csp_components, reg="ledoit_wolf", log=True, norm_trace=False)
    clf = LogisticRegression(
        C=float(C),
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline([("csp", csp), ("scaler", StandardScaler()), ("clf", clf)])


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_cfg: ModelConfig,
    fixed_C: float | None = None,
) -> tuple[float, float, np.ndarray]:
    """Stratified k-fold CV. Returns (mean_acc, std_acc, fold_scores)."""
    classes = np.unique(y)
    if not np.all(np.isin(classes, [1, 2])):
        raise ValueError(f"Expected labels 1/2, got classes={classes}")

    n1 = int(np.sum(y == 1))
    n2 = int(np.sum(y == 2))
    n_splits = min(5, n1, n2)
    if n_splits < 2:
        print("[CV] Not enough data per class for cross-validation")
        return 0.0, 0.0, np.array([])

    if model_cfg.use_riemann:
        eval_clf = (
            _make_classifier_riemann_fixedC(fixed_C)
            if fixed_C is not None
            else _make_classifier_riemann_selectC()
        )
    else:
        eval_clf = (
            _make_classifier_csp_fixedC(model_cfg.n_csp_components, fixed_C)
            if fixed_C is not None
            else _make_classifier_csp_selectC(model_cfg.n_csp_components)
        )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(eval_clf, X, y, cv=cv, scoring="accuracy")
    print(
        f"[CV] {n_splits}-fold: {[f'{s:.3f}' for s in scores]}  "
        f"mean={scores.mean():.3f} +/- {scores.std():.3f}"
    )
    return float(scores.mean()), float(scores.std()), scores


def _extract_best_C(pipeline: Pipeline) -> float:
    clf = pipeline.named_steps.get("clf", None)
    if clf is None or not hasattr(clf, "C_"):
        raise ValueError(
            "Could not extract best C: pipeline does not have LogisticRegressionCV at step 'clf'"
        )
    return float(np.atleast_1d(clf.C_)[0])


def train_initial_classifier(
    X_cal: list[np.ndarray],
    y_cal: list[int],
    model_cfg: ModelConfig,
    left_code: int,
    right_code: int,
) -> tuple[Pipeline, float, float, float, dict[str, int]]:
    """Fit calibration classifier, run CV, return (pipeline, best_C, cv_mean, cv_std, n_per_class)."""
    if len(y_cal) == 0:
        raise ValueError("No calibration epochs collected")

    X_arr = np.stack(X_cal, axis=0)
    y_arr = np.array(y_cal, dtype=int)

    # Select best C via internal CV
    selector = (
        _make_classifier_riemann_selectC()
        if model_cfg.use_riemann
        else _make_classifier_csp_selectC(model_cfg.n_csp_components)
    )
    selector.fit(X_arr, y_arr)
    best_C = _extract_best_C(selector)

    cv_mean, cv_std, _ = run_cv(X_arr, y_arr, model_cfg, fixed_C=best_C)

    # Final classifier with fixed C
    classifier = (
        _make_classifier_riemann_fixedC(best_C)
        if model_cfg.use_riemann
        else _make_classifier_csp_fixedC(model_cfg.n_csp_components, best_C)
    )
    classifier.fit(X_arr, y_arr)

    n_per_class = {
        str(int(left_code)): int(np.sum(y_arr == int(left_code))),
        str(int(right_code)): int(np.sum(y_arr == int(right_code))),
    }
    return classifier, best_C, cv_mean, cv_std, n_per_class


# ----------------------------------------------------------------
# Epoch filtering
# ----------------------------------------------------------------


def filter_epoch(X: np.ndarray, eeg_cfg: EEGConfig, sfreq: float) -> np.ndarray:
    """Apply MI bandpass (and optional notch) to epoch data.

    X shape: (n_channels, n_samples) for a single epoch, or
             (n_epochs, n_channels, n_samples) for a batch.
    """
    Xf = np.asarray(X, dtype=np.float64, order="C")
    if eeg_cfg.notch is not None:
        Xf = notch_filter(Xf, Fs=sfreq, freqs=[float(eeg_cfg.notch)], verbose="ERROR")
    Xf = filter_data(
        Xf, sfreq=sfreq, l_freq=eeg_cfg.l_freq, h_freq=eeg_cfg.h_freq, verbose="ERROR"
    )
    return Xf.astype(np.float32, copy=False)


# ----------------------------------------------------------------
# Raw CSV recorder
# ----------------------------------------------------------------


class RawCSVRecorder:
    """
    Continuously samples StreamLSL and writes *raw* samples to CSV with:
      t, <channel_1>, ..., <channel_N>
    Purely a tap for offline analysis – does not influence model/epoching.
    """

    def __init__(self, filepath: str, ch_names: list[str], winsize_s: float = 0.25):
        self.filepath = filepath
        self.ch_names = ch_names
        self.winsize_s = float(winsize_s)

        self._fh = None
        self._writer = None
        self._last_ts: float | None = None

    def start(self):
        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.filepath, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(["t"] + self.ch_names)
        self._fh.flush()
        self._last_ts = None
        print(f"[RAW] Recording to {self.filepath}")

    def stop(self):
        if self._fh is not None:
            try:
                self._fh.flush()
            except Exception:
                pass
            try:
                self._fh.close()
            except Exception:
                pass
        self._fh = None
        self._writer = None
        print("[RAW] Recording stopped.")

    def is_active(self) -> bool:
        return self._writer is not None

    def update(self, stream, picks: str = "all"):
        """Pull latest samples from *stream* and append to CSV."""
        if not self.is_active():
            return

        data, ts = stream.get_data(winsize=self.winsize_s, picks=picks)
        if data.size == 0 or ts is None or len(ts) == 0:
            return

        ts = np.asarray(ts)
        if self._last_ts is None:
            mask = np.ones_like(ts, dtype=bool)
        else:
            mask = ts > float(self._last_ts)

        if not np.any(mask):
            return

        idx = np.where(mask)[0]
        for j in idx:
            self._writer.writerow([float(ts[j])] + [float(x) for x in data[:, j]])

        self._last_ts = float(ts[idx[-1]])

        try:
            self._fh.flush()
        except Exception:
            pass
