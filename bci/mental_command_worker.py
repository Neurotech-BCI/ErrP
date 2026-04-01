from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import EEGConfig, MentalCommandModelConfig, MentalCommandTaskConfig, StimConfig


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _build_sos_bandpass(sfreq: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    return butter(
        int(order),
        [float(lo), float(hi)],
        btype="bandpass",
        output="sos",
        fs=float(sfreq),
    )


def _build_sos_notch(sfreq: float, freq: float, q: float = 30.0) -> np.ndarray:
    b, a = iirnotch(w0=float(freq), Q=float(q), fs=float(sfreq))
    return tf2sos(b, a)


def _build_wideband_sos(eeg_cfg: EEGConfig, sfreq: float) -> np.ndarray:
    stages = []
    if eeg_cfg.notch is not None:
        stages.append(_build_sos_notch(sfreq, float(eeg_cfg.notch)))
    stages.append(_build_sos_bandpass(sfreq, eeg_cfg.l_freq, eeg_cfg.h_freq, order=4))
    return np.vstack(stages).astype(np.float64, copy=False)


class StreamingIIRFilter:
    """Channel-wise causal SOS filter with persistent per-channel state."""

    def __init__(self, sos: np.ndarray, n_channels: int):
        sos = np.asarray(sos, dtype=np.float64)
        if sos.ndim != 2 or sos.shape[1] != 6:
            raise ValueError(f"Expected SOS shape (n_sections, 6), got {sos.shape}")
        self.sos = sos
        self.n_channels = int(n_channels)
        self._zi_template = sosfilt_zi(self.sos).astype(np.float64, copy=False)
        self.reset()

    @classmethod
    def from_eeg_config(cls, eeg_cfg: EEGConfig, sfreq: float, n_channels: int) -> "StreamingIIRFilter":
        return cls(_build_wideband_sos(eeg_cfg=eeg_cfg, sfreq=sfreq), n_channels=n_channels)

    def reset(self):
        self._zi = np.zeros((self.n_channels,) + self._zi_template.shape, dtype=np.float64)
        self._primed = np.zeros(self.n_channels, dtype=bool)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        X = np.asarray(chunk, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] != self.n_channels:
            raise ValueError(
                f"Expected chunk shape ({self.n_channels}, n_samples), got {X.shape}"
            )
        n_samples = int(X.shape[1])
        if n_samples == 0:
            return np.empty_like(X, dtype=np.float32)

        Y = np.empty_like(X, dtype=np.float64)
        for ch in range(self.n_channels):
            x = X[ch]
            zi = self._zi[ch]
            if not self._primed[ch]:
                zi = self._zi_template * float(x[0])
                self._primed[ch] = True
            y, zf = sosfilt(self.sos, x, zi=zi)
            self._zi[ch] = zf
            Y[ch] = y
        return Y.astype(np.float32, copy=False)


def filter_block(block: np.ndarray, eeg_cfg: EEGConfig, sfreq: float) -> np.ndarray:
    """Causal wideband filtering for one continuous block."""
    X = np.asarray(block, dtype=np.float32, order="C")
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D block (n_ch, n_samples), got {X.shape}")
    filt = StreamingIIRFilter.from_eeg_config(
        eeg_cfg=eeg_cfg, sfreq=float(sfreq), n_channels=int(X.shape[0])
    )
    return filt.process(X)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# EDF-backed MI training data
# ---------------------------------------------------------------------------

@dataclass
class OfflineMIDataset:
    X: np.ndarray
    y: np.ndarray
    sfreq: float
    channel_names: list[str]
    n_files_found: int
    n_files_used: int
    n_trials: int
    n_windows: int


def canonicalize_channel_name(name: str) -> str:
    cleaned = str(name).strip()
    cleaned = cleaned.replace("EEG ", "")
    cleaned = cleaned.replace("EEG_", "")
    cleaned = cleaned.replace("-Pz", "")
    cleaned = cleaned.replace("-PZ", "")
    cleaned = cleaned.replace(" ", "")
    return cleaned.upper()


def resolve_channel_order(
    available_names: list[str] | tuple[str, ...],
    desired_names: list[str] | tuple[str, ...],
) -> tuple[list[str], list[str]]:
    actual_by_key: dict[str, str] = {}
    for name in available_names:
        key = canonicalize_channel_name(name)
        if key not in actual_by_key:
            actual_by_key[key] = str(name)

    resolved: list[str] = []
    missing: list[str] = []
    for desired in desired_names:
        key = canonicalize_channel_name(desired)
        actual = actual_by_key.get(key)
        if actual is None:
            missing.append(str(desired))
        else:
            resolved.append(actual)
    return resolved, missing


def resolve_optional_channel(
    available_names: list[str] | tuple[str, ...],
    desired_name: str | None,
) -> str | None:
    if desired_name is None:
        return None
    resolved, _ = resolve_channel_order(available_names, [desired_name])
    return resolved[0] if resolved else None


def _standardize_edf_channels(
    raw: mne.io.BaseRaw,
    reref_channel: str | None = None,
) -> mne.io.BaseRaw:
    actual_ref = resolve_optional_channel(raw.ch_names, reref_channel)
    if actual_ref is not None:
        raw.set_eeg_reference(ref_channels=[actual_ref], verbose="ERROR")

    rename_map: dict[str, str] = {}
    for ch_name in raw.ch_names:
        new_name = ch_name
        if "-Pz" in new_name:
            new_name = new_name.replace("-Pz", "")
        elif "-PZ" in new_name:
            new_name = new_name.replace("-PZ", "")
        if new_name == "Pz":
            new_name = "EEG Pz"
        if new_name != ch_name:
            rename_map[ch_name] = new_name

    if rename_map:
        raw.rename_channels(rename_map)
    return raw


def rereference_and_select(
    data: np.ndarray,
    channel_names: list[str] | tuple[str, ...],
    target_channel_names: list[str] | tuple[str, ...],
    reref_channel_name: str | None = None,
) -> np.ndarray:
    X = np.asarray(data, dtype=np.float32)
    name_to_idx = {str(name): i for i, name in enumerate(channel_names)}
    target_idx = [name_to_idx[str(name)] for name in target_channel_names]
    X_target = X[target_idx]

    if reref_channel_name is None:
        return X_target.astype(np.float32, copy=False)

    ref_idx = name_to_idx.get(str(reref_channel_name))
    if ref_idx is None:
        return X_target.astype(np.float32, copy=False)

    ref = X[ref_idx:ref_idx + 1]
    return (X_target - ref).astype(np.float32, copy=False)


def _find_stim_channel(raw: mne.io.BaseRaw) -> str:
    candidates = (
        "Trigger",
        "TRG",
        "TRIGGER",
        "trigger",
        "STI",
        "stim",
        "Status",
        "status",
    )
    available = list(raw.ch_names)
    for candidate in candidates:
        if candidate in available:
            return candidate

    by_key = {canonicalize_channel_name(name): name for name in available}
    for candidate in candidates:
        actual = by_key.get(canonicalize_channel_name(candidate))
        if actual is not None:
            return actual

    raise RuntimeError(
        "Could not find a trigger/stim channel in EDF. "
        f"Available channels: {available}"
    )


def load_offline_mi_dataset(
    data_dir: str,
    edf_glob: str,
    eeg_cfg: EEGConfig,
    task_cfg: MentalCommandTaskConfig,
    stim_cfg: StimConfig,
    target_sfreq: float,
    target_channel_names: list[str] | tuple[str, ...],
    reref_channel_name: str | None = None,
) -> OfflineMIDataset:
    data_path = Path(data_dir).expanduser()
    edf_paths = sorted(data_path.rglob(edf_glob))
    if not edf_paths:
        raise FileNotFoundError(
            f"No EDF files matched {edf_glob!r} under {str(data_path)!r}"
        )

    context_n = int(round(float(task_cfg.filter_context_s) * float(target_sfreq)))
    epoch_n = int(round(float(task_cfg.epoch_duration_s) * float(target_sfreq)))
    reject_thresh = eeg_cfg.reject_peak_to_peak

    all_windows: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    n_trials = 0
    n_files_used = 0

    for edf_path in edf_paths:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
        raw = _standardize_edf_channels(raw, reref_channel=reref_channel_name)
        stim_channel = _find_stim_channel(raw)

        events = mne.find_events(
            raw,
            stim_channel=stim_channel,
            min_duration=0.0,
            verbose=False,
        )
        if len(events) == 0:
            continue

        original_sfreq = float(raw.info["sfreq"])
        event_times_s = events[:, 0].astype(np.float64) / original_sfreq

        if abs(original_sfreq - float(target_sfreq)) > 1e-6:
            raw.resample(float(target_sfreq), npad="auto", verbose="ERROR")

        picks, missing = resolve_channel_order(raw.ch_names, target_channel_names)
        if missing:
            raise RuntimeError(
                f"EDF {edf_path} is missing required channels {missing}. "
                f"Available channels: {raw.ch_names}"
            )

        actual_ref = resolve_optional_channel(raw.ch_names, reref_channel_name)
        load_names = list(picks)
        if actual_ref is not None and actual_ref not in load_names:
            load_names.append(actual_ref)

        raw_data = raw.get_data(picks=load_names).astype(np.float32, copy=False)
        file_used = False

        for event_time_s, event_code in zip(event_times_s, events[:, 2]):
            if not stim_cfg.is_lr_code(int(event_code)):
                continue

            start = int(round((float(event_time_s) - float(task_cfg.filter_context_s)) * float(target_sfreq)))
            stop = start + context_n + epoch_n
            if start < 0 or stop > eeg_data.shape[1]:
                continue

            segment = raw_data[:, start:stop]
            segment = rereference_and_select(
                data=segment,
                channel_names=load_names,
                target_channel_names=target_channel_names,
                reref_channel_name=actual_ref,
            )
            filtered_segment = filter_block(segment, eeg_cfg=eeg_cfg, sfreq=float(target_sfreq))
            trial = filtered_segment[:, context_n: context_n + epoch_n]
            windows = split_windows(
                block=trial,
                sfreq=float(target_sfreq),
                window_s=float(task_cfg.train_window_s),
                step_s=float(task_cfg.train_window_step_s),
            )
            if windows.shape[0] == 0:
                continue

            if reject_thresh is not None:
                keep_mask = np.ptp(windows, axis=-1).max(axis=1) <= float(reject_thresh)
                windows = windows[keep_mask]
            if windows.shape[0] == 0:
                continue

            all_windows.append(windows)
            all_labels.append(np.full(windows.shape[0], int(event_code), dtype=int))
            n_trials += 1
            file_used = True

        if file_used:
            n_files_used += 1

    if not all_windows:
        raise RuntimeError(
            "No usable left/right training windows were extracted from the EDF folder. "
            "Check the data path, trigger codes, and channel alignment."
        )

    X = np.concatenate(all_windows, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(all_labels, axis=0).astype(int, copy=False)
    return OfflineMIDataset(
        X=X,
        y=y,
        sfreq=float(target_sfreq),
        channel_names=[str(name) for name in target_channel_names],
        n_files_found=len(edf_paths),
        n_files_used=n_files_used,
        n_trials=n_trials,
        n_windows=int(X.shape[0]),
    )


# ---------------------------------------------------------------------------
# Sub-band helper (shared by both feature extractors)
# ---------------------------------------------------------------------------

def _iir_bandpass(X: np.ndarray, sfreq: float, lo: float, hi: float) -> np.ndarray:
    """Causal IIR bandpass on an (n_windows, n_ch, n_samples) array."""
    X64 = np.asarray(X, dtype=np.float64)
    orig_shape = X64.shape
    if X64.ndim == 3:
        X64 = X64.reshape(-1, orig_shape[-1])
    sos = _build_sos_bandpass(sfreq=float(sfreq), lo=float(lo), hi=float(hi), order=4)
    zi_template = sosfilt_zi(sos).astype(np.float64, copy=False)
    filtered = np.empty_like(X64, dtype=np.float64)
    for i in range(X64.shape[0]):
        x = X64[i]
        zi = zi_template * float(x[0])
        y, _ = sosfilt(sos, x, zi=zi)
        filtered[i] = y
    return filtered.reshape(orig_shape).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

class FilterBankTangentSpace(BaseEstimator, TransformerMixin):
    """Per-band Riemannian covariance → tangent-space projection, concatenated."""

    def __init__(self, bands=((4, 8), (8, 13), (13, 30), (30, 45)),
                 sfreq: float = 300.0, cov_estimator: str = "oas"):
        self.bands = bands
        self.sfreq = sfreq
        self.cov_estimator = cov_estimator

    def fit(self, X, y=None):
        self.cov_estimators_: list[Covariances] = []
        self.ts_estimators_: list[TangentSpace] = []
        for lo, hi in self.bands:
            X_band = _iir_bandpass(X, self.sfreq, lo, hi)
            cov_est = Covariances(estimator=self.cov_estimator)
            covs = cov_est.fit_transform(X_band)
            ts = TangentSpace(metric="riemann")
            ts.fit(covs)
            self.cov_estimators_.append(cov_est)
            self.ts_estimators_.append(ts)
        return self

    def transform(self, X):
        parts = []
        for i, (lo, hi) in enumerate(self.bands):
            X_band = _iir_bandpass(X, self.sfreq, lo, hi)
            covs = self.cov_estimators_[i].transform(X_band)
            parts.append(self.ts_estimators_[i].transform(covs))
        return np.concatenate(parts, axis=1)


class LogBandPowerFeatures(BaseEstimator, TransformerMixin):
    """Log-variance (≈ log band power) per channel per sub-band."""

    def __init__(self, bands=((4, 8), (8, 13), (13, 30), (30, 45)),
                 sfreq: float = 300.0):
        self.bands = bands
        self.sfreq = sfreq

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        parts = []
        for lo, hi in self.bands:
            X_band = _iir_bandpass(X, self.sfreq, lo, hi)
            log_bp = np.log(np.var(X_band, axis=-1) + 1e-10)
            parts.append(log_bp)
        return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# Pipeline constructors
# ---------------------------------------------------------------------------

def _make_lr(model_cfg: MentalCommandModelConfig) -> LogisticRegression:
    return LogisticRegression(
        C=float(model_cfg.C),
        solver="lbfgs",
        multi_class="multinomial",
        max_iter=int(model_cfg.max_iter),
        class_weight=model_cfg.class_weight,
        random_state=42,
    )


def make_fb_riemannian_classifier(
    model_cfg: MentalCommandModelConfig, sfreq: float,
) -> Pipeline:
    return Pipeline([
        ("fb_ts", FilterBankTangentSpace(
            bands=model_cfg.filter_bank_bands,
            sfreq=sfreq,
        )),
        ("scaler", StandardScaler()),
        ("clf", _make_lr(model_cfg)),
    ])


def make_bandpower_classifier(
    model_cfg: MentalCommandModelConfig, sfreq: float,
) -> Pipeline:
    return Pipeline([
        ("bp", LogBandPowerFeatures(
            bands=model_cfg.filter_bank_bands,
            sfreq=sfreq,
        )),
        ("scaler", StandardScaler()),
        ("clf", _make_lr(model_cfg)),
    ])


def make_mi_classifier(model_cfg: MentalCommandModelConfig) -> Pipeline:
    """Single-band Riemannian classifier for left/right motor imagery windows."""
    return Pipeline([
        ("cov", Covariances(estimator=model_cfg.cov_estimator)),
        ("ts", TangentSpace(metric="riemann")),
        ("scaler", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis()),
    ])


# ---------------------------------------------------------------------------
# Cross-validation quality estimate
# ---------------------------------------------------------------------------

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
    pipeline: Pipeline,
) -> MCQuality:
    """LOGO-by-block-triplet CV (leave one neutral/c1/c2 block-group out)."""
    y = np.asarray(y, dtype=int)
    block_ids = np.asarray(block_ids, dtype=int)
    if y.shape[0] != X.shape[0] or block_ids.shape[0] != X.shape[0]:
        raise ValueError("X, y, and block_ids must have the same first dimension")

    classes, counts = np.unique(y, return_counts=True)
    class_block_ids: dict[int, np.ndarray] = {}
    for c in classes:
        class_mask = y == int(c)
        class_block_ids[int(c)] = np.sort(np.unique(block_ids[class_mask]))

    ref_block_ids = class_block_ids[int(classes[0])]
    for c in classes[1:]:
        if not np.array_equal(class_block_ids[int(c)], ref_block_ids):
            block_counts = {int(k): class_block_ids[int(k)].tolist() for k in classes}
            raise ValueError(
                "Per-class block ids do not align for triplet-LOGO CV: "
                f"{block_counts}"
            )

    n_splits = int(len(ref_block_ids))
    if n_splits < 2:
        block_counts = {int(c): len(class_block_ids[int(c)]) for c in classes}
        raise ValueError(
            f"Not enough registration blocks for grouped CV: per-class blocks={block_counts}"
        )

    y_pred = np.full_like(y, fill_value=-1)
    for held_out_block in ref_block_ids:
        test_mask = block_ids == int(held_out_block)
        train_mask = ~test_mask
        if not np.any(test_mask) or not np.any(train_mask):
            raise ValueError(f"Invalid CV split at held-out block {int(held_out_block)}")

        test_classes = np.unique(y[test_mask])
        if len(test_classes) != len(classes):
            raise ValueError(
                f"Held-out block {int(held_out_block)} missing classes: present={test_classes}, expected={classes}"
            )

        fold_clf = clone(pipeline)
        fold_clf.fit(X[train_mask], y[train_mask])
        y_pred[test_mask] = fold_clf.predict(X[test_mask])

    if np.any(y_pred < 0):
        raise RuntimeError("CV prediction vector contains unassigned samples")

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


# ---------------------------------------------------------------------------
# Live-mode smoother
# ---------------------------------------------------------------------------

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
