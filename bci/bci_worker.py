# bci_worker.py  –  BCI utility module
#
# Filter-bank classifier pipelines, incremental GMM for online adaptation,
# epoch filtering, and raw CSV recording.
# Imported by psychopy_task.py (the single-process entry point).
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mne.filter import filter_data, notch_filter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import oas

from mental_command_worker import FilterBankTangentSpace
from config import EEGConfig, ModelConfig


# ----------------------------------------------------------------
# Epoch filtering (IIR — safe for short epochs with wide bandpass)
# ----------------------------------------------------------------


def filter_epoch(X: np.ndarray, eeg_cfg: EEGConfig, sfreq: float) -> np.ndarray:
    """Apply wide bandpass (and optional notch) to epoch data.

    Uses IIR (4th-order Butterworth, zero-phase) so that edge effects are
    negligible even on epochs as short as ~2-3 s.

    X shape: (n_channels, n_samples) for a single epoch, or
             (n_epochs, n_channels, n_samples) for a batch.
    """
    Xf = np.asarray(X, dtype=np.float64, order="C")
    orig_shape = Xf.shape
    if Xf.ndim == 3:
        Xf = Xf.reshape(-1, orig_shape[-1])
    if eeg_cfg.notch is not None:
        Xf = notch_filter(Xf, Fs=sfreq, freqs=[float(eeg_cfg.notch)], verbose="ERROR")
    Xf = filter_data(
        Xf,
        sfreq=float(sfreq),
        l_freq=float(eeg_cfg.l_freq),
        h_freq=float(eeg_cfg.h_freq),
        method="iir",
        iir_params=dict(order=4, ftype="butter", output="sos"),
        verbose="ERROR",
    )
    return Xf.reshape(orig_shape).astype(np.float32, copy=False)


# ----------------------------------------------------------------
# Incremental GMM for online co-adaptation
# ----------------------------------------------------------------


class IncrementalGMM:
    """One Gaussian per class with exponentially weighted online updates.

    Fit from calibration data (OAS-shrunk covariance), then call ``update``
    per epoch during the online phase for smooth, continuous adaptation.
    """

    def __init__(self, learning_rate: float = 0.05, reg: float = 1e-6):
        self.learning_rate = float(learning_rate)
        self._reg = float(reg)
        self.classes_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covs_: np.ndarray | None = None
        self.priors_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> IncrementalGMM:
        """Initial fit from calibration features (2-D: n_samples × n_features)."""
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        n_cls = len(self.classes_)
        n_feat = X.shape[1]
        self.means_ = np.zeros((n_cls, n_feat), dtype=np.float64)
        self.covs_ = np.zeros((n_cls, n_feat, n_feat), dtype=np.float64)
        self.priors_ = np.zeros(n_cls, dtype=np.float64)

        for i, c in enumerate(self.classes_):
            Xi = X[y == c]
            self.means_[i] = Xi.mean(axis=0)
            if Xi.shape[0] > 1:
                cov, _ = oas(Xi)
                self.covs_[i] = cov
            else:
                self.covs_[i] = np.eye(n_feat) * self._reg
            self.priors_[i] = len(Xi) / len(y)
        return self

    def update(self, x: np.ndarray, label: int) -> IncrementalGMM:
        """Exponentially weighted mean/covariance update for one new observation."""
        idx = int(np.where(self.classes_ == int(label))[0][0])
        eta = self.learning_rate
        x = np.asarray(x, dtype=np.float64)

        old_mean = self.means_[idx].copy()
        self.means_[idx] = (1.0 - eta) * old_mean + eta * x

        # Correct for mean shift + incorporate new observation.
        mean_diff = old_mean - self.means_[idx]
        obs_diff = x - self.means_[idx]
        self.covs_[idx] = (
            (1.0 - eta) * (self.covs_[idx] + np.outer(mean_diff, mean_diff))
            + eta * np.outer(obs_diff, obs_diff)
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Gaussian posterior probabilities.  X: (n_samples, n_features)."""
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        n_cls = len(self.classes_)
        log_post = np.zeros((n_samples, n_cls), dtype=np.float64)

        for i in range(n_cls):
            diff = X - self.means_[i]
            try:
                L = np.linalg.cholesky(self.covs_[i])
                z = np.linalg.solve(L, diff.T)          # (n_feat, n_samples)
                mahal = np.sum(z ** 2, axis=0)           # (n_samples,)
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(self.covs_[i])
                mahal = np.sum(diff @ inv_cov * diff, axis=1)
                eigvals = np.linalg.eigvalsh(self.covs_[i])
                log_det = np.sum(np.log(np.maximum(eigvals, self._reg)))

            log_post[:, i] = -0.5 * (mahal + log_det) + np.log(self.priors_[i] + 1e-10)

        # Log-sum-exp normalisation.
        log_post -= np.max(log_post, axis=1, keepdims=True)
        probs = np.exp(log_post)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ----------------------------------------------------------------
# Pipeline builders (filter-bank approach, using ModelConfig)
# ----------------------------------------------------------------


def _make_fb_pipeline(model_cfg: ModelConfig, sfreq: float) -> Pipeline:
    """Filter-Bank Riemannian pipeline for CV evaluation."""
    return Pipeline([
        ("fb_ts", FilterBankTangentSpace(
            bands=model_cfg.filter_bank_bands,
            sfreq=sfreq,
        )),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=float(model_cfg.C),
            solver="lbfgs",
            max_iter=int(model_cfg.max_iter),
            class_weight=model_cfg.class_weight,
            random_state=42,
        )),
    ])


# ----------------------------------------------------------------
# Cross-validation helper
# ----------------------------------------------------------------


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    pipeline: Pipeline,
) -> tuple[float, float, np.ndarray]:
    """Stratified k-fold CV. Returns (mean_acc, std_acc, fold_scores)."""
    classes = np.unique(y)
    counts = {int(c): int(np.sum(y == c)) for c in classes}
    n_splits = min(5, *counts.values())
    if n_splits < 2:
        print(f"[CV] Not enough data per class for cross-validation: {counts}")
        return 0.0, 0.0, np.array([])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    print(
        f"[CV] {n_splits}-fold: {[f'{s:.3f}' for s in scores]}  "
        f"mean={scores.mean():.3f} +/- {scores.std():.3f}"
    )
    return float(scores.mean()), float(scores.std()), scores


# ----------------------------------------------------------------
# Calibration trainer
# ----------------------------------------------------------------


@dataclass
class CalibrationResult:
    feature_extractor: Pipeline   # Frozen pipeline[:-1] for online feature extraction.
    gmm: IncrementalGMM           # Online classifier with incremental updates.
    full_pipeline: Pipeline       # Full fitted pipeline (for final CV).
    cv_mean: float
    cv_std: float
    n_per_class: dict[str, int]


def train_initial_classifier(
    X_cal: list[np.ndarray],
    y_cal: list[int],
    model_cfg: ModelConfig,
    sfreq: float,
    left_code: int,
    right_code: int,
) -> CalibrationResult:
    """Fit FB Riemannian pipeline, then initialise IncrementalGMM on features."""
    if len(y_cal) == 0:
        raise ValueError("No calibration epochs collected")

    X_arr = np.stack(X_cal, axis=0)
    y_arr = np.array(y_cal, dtype=int)

    fb_pipe = _make_fb_pipeline(model_cfg, sfreq)

    print("[TRAIN] Evaluating Filter-Bank Riemannian...")
    fb_mean, fb_std, _ = run_cv(X_arr, y_arr, fb_pipe)
    chosen_pipe = fb_pipe
    cv_mean, cv_std = fb_mean, fb_std

    # Fit chosen pipeline on all calibration data.
    chosen_pipe.fit(X_arr, y_arr)

    # Extract frozen feature extractor (all steps except the classifier head).
    feature_extractor = chosen_pipe[:-1]

    # Compute calibration features and fit the online GMM.
    X_features = feature_extractor.transform(X_arr)
    gmm = IncrementalGMM(
        learning_rate=model_cfg.gmm_learning_rate,
        reg=model_cfg.gmm_cov_reg,
    )
    gmm.fit(X_features, y_arr)

    n_per_class = {
        str(int(left_code)): int(np.sum(y_arr == int(left_code))),
        str(int(right_code)): int(np.sum(y_arr == int(right_code))),
    }

    print(
        f"[TRAIN] Selected Filter-Bank Riemannian ({cv_mean:.1%}).  "
        f"GMM fitted on {X_features.shape[1]}-dim features."
    )

    return CalibrationResult(
        feature_extractor=feature_extractor,
        gmm=gmm,
        full_pipeline=chosen_pipe,
        cv_mean=cv_mean,
        cv_std=cv_std,
        n_per_class=n_per_class,
    )


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
