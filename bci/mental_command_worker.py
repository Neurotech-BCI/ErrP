from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import EEGConfig, MentalCommandModelConfig, MentalCommandTaskConfig, StimConfig, LiveMITaskConfig


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
    """Channel-wise causal SOS filter with persistent state."""

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
        if X.shape[1] == 0:
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
    X = np.asarray(block, dtype=np.float32, order="C")
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D block (n_channels, n_samples), got {X.shape}")
    filt = StreamingIIRFilter.from_eeg_config(
        eeg_cfg=eeg_cfg,
        sfreq=float(sfreq),
        n_channels=int(X.shape[0]),
    )
    return filt.process(X)


def filter_session(block: np.ndarray, eeg_cfg: EEGConfig, sfreq: float) -> np.ndarray:
    """Causally filter a continuous session block from start to finish.

    This is the closest offline analogue to the live stream path, where the
    filter state is carried forward continuously rather than reset at each
    epoch boundary.
    """
    return filter_block(block=block, eeg_cfg=eeg_cfg, sfreq=sfreq)

def prepare_continuous_windows(
    raw_block: np.ndarray,
    eeg_cfg: EEGConfig,
    sfreq: float,
    window_s: float,
    step_s: float,
    reject_peak_to_peak: float | None,
) -> np.ndarray:
    X = np.asarray(raw_block, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected raw_block with shape (n_channels, n_samples), got {X.shape}")
    window_n = int(round(float(window_s) * float(sfreq)))
    if X.shape[1] == 0:
        return np.empty((0, X.shape[0], window_n), dtype=np.float32)

    X_filt = filter_session(block=X, eeg_cfg=eeg_cfg, sfreq=float(sfreq))
    windows = split_windows(
        block=X_filt,
        sfreq=float(sfreq),
        window_s=float(window_s),
        step_s=float(step_s),
    )
    if windows.shape[0] == 0:
        return windows
    if reject_peak_to_peak is not None:
        keep_mask = np.ptp(windows, axis=-1).max(axis=1) <= float(reject_peak_to_peak)
        windows = windows[keep_mask]
    return windows.astype(np.float32, copy=False)




# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def split_windows(
    block: np.ndarray,
    sfreq: float,
    window_s: float,
    step_s: float,
) -> np.ndarray:
    n_channels, n_samples = block.shape
    window_n = int(round(float(window_s) * float(sfreq)))
    step_n = int(round(float(step_s) * float(sfreq)))
    if window_n <= 0 or step_n <= 0:
        raise ValueError("window_s and step_s must produce at least one sample")
    if n_samples < window_n:
        return np.empty((0, n_channels, window_n), dtype=np.float32)
    starts = np.arange(0, n_samples - window_n + 1, step_n, dtype=int)
    windows = np.empty((len(starts), n_channels, window_n), dtype=np.float32)
    for i, start in enumerate(starts):
        windows[i] = block[:, start:start + window_n]
    return windows


# ---------------------------------------------------------------------------
# Filter-bank helper kept for bci_worker compatibility
# ---------------------------------------------------------------------------

def _iir_bandpass_epochs(X: np.ndarray, sfreq: float, lo: float, hi: float) -> np.ndarray:
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


class FilterBankTangentSpace(BaseEstimator, TransformerMixin):
    """Per-band covariance -> tangent space projection, concatenated."""

    def __init__(self, bands=((4, 8), (8, 13), (13, 30), (30, 45)), sfreq: float = 300.0, cov_estimator: str = "oas"):
        self.bands = bands
        self.sfreq = sfreq
        self.cov_estimator = cov_estimator

    def fit(self, X, y=None):
        self.cov_estimators_: list[Covariances] = []
        self.ts_estimators_: list[TangentSpace] = []
        for lo, hi in self.bands:
            X_band = _iir_bandpass_epochs(X, self.sfreq, lo, hi)
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
            X_band = _iir_bandpass_epochs(X, self.sfreq, lo, hi)
            covs = self.cov_estimators_[i].transform(X_band)
            parts.append(self.ts_estimators_[i].transform(covs))
        return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# Offline EDF-backed MI dataset
# ---------------------------------------------------------------------------

@dataclass
class OfflineMIDataset:
    X: np.ndarray
    y: np.ndarray
    session_ids: np.ndarray
    sfreq: float
    channel_names: list[str]
    eeg_units: str
    offline_scale_applied: float
    n_files_found: int
    n_files_used: int
    n_trials: int
    n_windows: int


@dataclass
class LOSOResult:
    session_scores: dict[int, float]
    mean_accuracy: float
    std_accuracy: float


@dataclass
class SharedMIModelResult:
    classifier: Pipeline
    class_index: dict[int, int]
    loso: LOSOResult
    class_counts: dict[int, int]
    loaded_from_cache: bool
    dataset: OfflineMIDataset | None
    cache_path: str


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
        actual = actual_by_key.get(canonicalize_channel_name(desired))
        if actual is None:
            missing.append(str(desired))
        else:
            resolved.append(actual)
    return resolved, missing


def standardize_offline_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    if "EEG LE-Pz" in raw.ch_names:
        raw.set_eeg_reference(ref_channels=["EEG LE-Pz"], verbose="ERROR")

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


def find_stim_channel(raw: mne.io.BaseRaw) -> str:
    candidates = ("Trigger", "TRG", "TRIGGER", "trigger", "STI", "stim", "Status", "status")
    for candidate in candidates:
        if candidate in raw.ch_names:
            return candidate
    raise RuntimeError(f"Could not find trigger channel in EDF. Available channels: {raw.ch_names}")


def make_mi_classifier(model_cfg: MentalCommandModelConfig) -> Pipeline:
    return Pipeline([
        ("cov", Covariances(estimator=model_cfg.cov_estimator)),
        ("ts", TangentSpace(metric="riemann")),
        ("scaler", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis()),
    ])


def load_offline_mi_dataset(
    data_dir: str,
    edf_glob: str,
    eeg_cfg: EEGConfig,
    task_cfg: MentalCommandTaskConfig,
    stim_cfg: StimConfig,
    target_sfreq: float,
    target_channel_names: list[str] | tuple[str, ...],
    calibrateOnParticipant: str
) -> OfflineMIDataset:
    data_path = Path(data_dir).expanduser()
    edf_paths = sorted(data_path.rglob(edf_glob))
    if not edf_paths:
        raise FileNotFoundError(
            f"No EDF files matched {edf_glob!r} under {str(data_path)!r}"
        )

    epoch_n = int(round(float(task_cfg.epoch_duration_s) * float(target_sfreq)))
    reject_thresh = eeg_cfg.reject_peak_to_peak

    all_windows: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_session_ids: list[np.ndarray] = []
    n_trials = 0
    n_files_used = 0

    for session_id, edf_path in enumerate(edf_paths):
        if calibrateOnParticipant not in str(edf_path):
            continue
        print(str(edf_path))
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
        raw = standardize_offline_raw(raw)
        stim_channel = find_stim_channel(raw)

        original_sfreq = float(raw.info["sfreq"])
        events = mne.find_events(raw, stim_channel=stim_channel, min_duration=0.0, verbose=False)
        if len(events) == 0:
            continue
        event_times_s = events[:, 0].astype(np.float64) / original_sfreq

        if abs(original_sfreq - float(target_sfreq)) > 1e-6:
            raw.resample(float(target_sfreq), npad="auto", verbose="ERROR")

        picks, missing = resolve_channel_order(raw.ch_names, target_channel_names)
        if missing:
            raise RuntimeError(
                f"EDF {edf_path} is missing required channels {missing}. "
                f"Available channels: {raw.ch_names}"
            )

        eeg_data = raw.get_data(picks=picks).astype(np.float32, copy=False)
        eeg_data *= float(task_cfg.offline_eeg_scale_to_match_live)
        eeg_data_filt = filter_session(
            block=eeg_data,
            eeg_cfg=eeg_cfg,
            sfreq=float(target_sfreq),
        )
        file_used = False

        for event_time_s, event_code in zip(event_times_s, events[:, 2]):
            if not stim_cfg.is_lr_code(int(event_code)):
                continue

            start = int(round(float(event_time_s) * float(target_sfreq)))
            stop = start + epoch_n
            if start < 0 or stop > eeg_data_filt.shape[1]:
                continue

            epoch = eeg_data_filt[:, start:stop]
            windows = split_windows(
                block=epoch,
                sfreq=float(target_sfreq),
                window_s=float(task_cfg.window_s),
                step_s=float(task_cfg.window_step_s),
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
            all_session_ids.append(np.full(windows.shape[0], int(session_id), dtype=int))
            n_trials += 1
            file_used = True

        if file_used:
            n_files_used += 1

    if not all_windows:
        raise RuntimeError(
            "No usable left/right windows were extracted from the EDF folder. "
            "Check the path, trigger codes, and channel names."
        )

    X = np.concatenate(all_windows, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(all_labels, axis=0).astype(int, copy=False)
    session_ids = np.concatenate(all_session_ids, axis=0).astype(int, copy=False)
    return OfflineMIDataset(
        X=X,
        y=y,
        session_ids=session_ids,
        sfreq=float(target_sfreq),
        channel_names=[str(name) for name in target_channel_names],
        eeg_units=str(task_cfg.live_eeg_units),
        offline_scale_applied=float(task_cfg.offline_eeg_scale_to_match_live),
        n_files_found=len(edf_paths),
        n_files_used=n_files_used,
        n_trials=n_trials,
        n_windows=int(X.shape[0]),
    )


def evaluate_loso_sessions(
    dataset: OfflineMIDataset,
    model_cfg: MentalCommandModelConfig,
    train_only_session_ids: set[int] | None = None,
) -> LOSOResult:
    session_scores: dict[int, float] = {}
    train_only_session_ids = set() if train_only_session_ids is None else {int(x) for x in train_only_session_ids}
    unique_sessions = [int(s) for s in np.unique(dataset.session_ids) if int(s) not in train_only_session_ids]

    for session_id in unique_sessions:
        test_mask = dataset.session_ids == int(session_id)
        train_mask = ~test_mask
        if not np.any(test_mask) or not np.any(train_mask):
            continue

        y_train = dataset.y[train_mask]
        y_test = dataset.y[test_mask]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        clf = make_mi_classifier(model_cfg)
        clf.fit(dataset.X[train_mask], y_train)
        y_pred = clf.predict(dataset.X[test_mask])
        session_scores[int(session_id)] = float(accuracy_score(y_test, y_pred))

    if session_scores:
        scores = np.asarray(list(session_scores.values()), dtype=np.float64)
        mean_accuracy = float(np.mean(scores))
        std_accuracy = float(np.std(scores))
    else:
        mean_accuracy = 0.0
        std_accuracy = 0.0

    return LOSOResult(
        session_scores=session_scores,
        mean_accuracy=mean_accuracy,
        std_accuracy=std_accuracy,
    )


def _edf_manifest(data_dir: str, edf_glob: str, calibrate_on_participant: str) -> list[dict[str, int | str]]:
    root = Path(data_dir).expanduser()
    rows: list[dict[str, int | str]] = []
    if not root.exists():
        return rows
    for p in sorted(root.rglob(edf_glob)):
        if calibrate_on_participant and calibrate_on_participant not in str(p):
            continue
        st = p.stat()
        rows.append(
            {
                "path": str(p.resolve()),
                "size": int(st.st_size),
                "mtime_ns": int(st.st_mtime_ns),
            }
        )
    return rows


def _shared_model_signature(
    *,
    data_dir: str,
    edf_glob: str,
    calibrate_on_participant: str,
    eeg_cfg: EEGConfig,
    task_cfg: MentalCommandTaskConfig,
    stim_cfg: StimConfig,
    model_cfg: MentalCommandModelConfig,
    target_sfreq: float,
    target_channel_names: list[str] | tuple[str, ...],
) -> tuple[str, dict]:
    payload = {
        "schema_version": 1,
        "data_dir": str(Path(data_dir).expanduser().resolve()),
        "edf_glob": str(edf_glob),
        "calibrate_on_participant": str(calibrate_on_participant),
        "target_sfreq": round(float(target_sfreq), 6),
        "target_channel_names": [canonicalize_channel_name(ch) for ch in target_channel_names],
        "eeg_cfg": {
            "picks": [canonicalize_channel_name(ch) for ch in eeg_cfg.picks],
            "reref_channel": None if eeg_cfg.reref_channel is None else canonicalize_channel_name(eeg_cfg.reref_channel),
            "l_freq": float(eeg_cfg.l_freq),
            "h_freq": float(eeg_cfg.h_freq),
            "notch": None if eeg_cfg.notch is None else float(eeg_cfg.notch),
            "reject_peak_to_peak": None if eeg_cfg.reject_peak_to_peak is None else float(eeg_cfg.reject_peak_to_peak),
        },
        "task_cfg": {
            "epoch_duration_s": float(task_cfg.epoch_duration_s),
            "window_s": float(task_cfg.window_s),
            "window_step_s": float(task_cfg.window_step_s),
            "offline_eeg_scale_to_match_live": float(task_cfg.offline_eeg_scale_to_match_live),
            "live_eeg_units": str(task_cfg.live_eeg_units),
        },
        "stim_cfg": {
            "left_code": int(stim_cfg.left_code),
            "right_code": int(stim_cfg.right_code),
        },
        "model_cfg": {
            "cov_estimator": str(model_cfg.cov_estimator),
        },
        "edf_manifest": _edf_manifest(
            data_dir=data_dir,
            edf_glob=edf_glob,
            calibrate_on_participant=calibrate_on_participant,
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def train_or_load_shared_mi_model(
    *,
    cache_name: str,
    data_dir: str,
    edf_glob: str,
    calibrate_on_participant: str,
    eeg_cfg: EEGConfig,
    task_cfg: MentalCommandTaskConfig,
    stim_cfg: StimConfig,
    model_cfg: MentalCommandModelConfig,
    target_sfreq: float,
    target_channel_names: list[str] | tuple[str, ...],
    logger: logging.Logger | None = None,
    force_retrain: bool = False,
) -> SharedMIModelResult:
    cache_dir = Path(__file__).resolve().parent / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_name}.pkl"

    signature, _payload = _shared_model_signature(
        data_dir=data_dir,
        edf_glob=edf_glob,
        calibrate_on_participant=calibrate_on_participant,
        eeg_cfg=eeg_cfg,
        task_cfg=task_cfg,
        stim_cfg=stim_cfg,
        model_cfg=model_cfg,
        target_sfreq=target_sfreq,
        target_channel_names=target_channel_names,
    )

    if not force_retrain and cache_path.exists():
        try:
            with open(cache_path, "rb") as fh:
                cached = pickle.load(fh)
            if cached.get("signature") == signature:
                classifier = cached["classifier"]
                clf_classes = np.asarray(classifier.named_steps["clf"].classes_, dtype=int)
                class_index = {int(c): i for i, c in enumerate(clf_classes)}
                loso = LOSOResult(
                    session_scores={int(k): float(v) for k, v in cached.get("session_scores", {}).items()},
                    mean_accuracy=float(cached.get("loso_mean", 0.0)),
                    std_accuracy=float(cached.get("loso_std", 0.0)),
                )
                class_counts = {int(k): int(v) for k, v in cached.get("class_counts", {}).items()}
                if logger is not None:
                    logger.info("Loaded cached MI model from %s", str(cache_path))
                return SharedMIModelResult(
                    classifier=classifier,
                    class_index=class_index,
                    loso=loso,
                    class_counts=class_counts,
                    loaded_from_cache=True,
                    dataset=None,
                    cache_path=str(cache_path),
                )
        except Exception as exc:
            if logger is not None:
                logger.warning("Failed to load MI cache at %s: %s", str(cache_path), exc)

    dataset = load_offline_mi_dataset(
        data_dir=data_dir,
        edf_glob=edf_glob,
        eeg_cfg=eeg_cfg,
        task_cfg=task_cfg,
        stim_cfg=stim_cfg,
        target_sfreq=float(target_sfreq),
        target_channel_names=target_channel_names,
        calibrateOnParticipant=calibrate_on_participant,
    )
    loso = evaluate_loso_sessions(dataset, model_cfg)
    classifier = make_mi_classifier(model_cfg)
    classifier.fit(dataset.X, dataset.y)

    clf_classes = np.asarray(classifier.named_steps["clf"].classes_, dtype=int)
    class_index = {int(c): i for i, c in enumerate(clf_classes)}
    class_vals, class_n = np.unique(dataset.y, return_counts=True)
    class_counts = {int(c): int(n) for c, n in zip(class_vals, class_n)}

    try:
        payload = {
            "signature": signature,
            "classifier": classifier,
            "session_scores": {int(k): float(v) for k, v in loso.session_scores.items()},
            "loso_mean": float(loso.mean_accuracy),
            "loso_std": float(loso.std_accuracy),
            "class_counts": class_counts,
        }
        with open(cache_path, "wb") as fh:
            pickle.dump(payload, fh)
        if logger is not None:
            logger.info("Saved shared MI model cache to %s", str(cache_path))
    except Exception as exc:
        if logger is not None:
            logger.warning("Failed to save MI cache at %s: %s", str(cache_path), exc)

    return SharedMIModelResult(
        classifier=classifier,
        class_index=class_index,
        loso=loso,
        class_counts=class_counts,
        loaded_from_cache=False,
        dataset=dataset,
        cache_path=str(cache_path),
    )

def load_dataset_for_live_task(
    data_dir: str,
    edf_glob: str,
    eeg_cfg: EEGConfig,
    task_cfg: LiveMITaskConfig,
    stim_cfg: StimConfig,
    target_sfreq: float,
    target_channel_names: list[str] | tuple[str, ...],
) -> OfflineMIDataset:
    data_path = Path(data_dir).expanduser()
    edf_paths = sorted(data_path.rglob(edf_glob))
    if not edf_paths:
        raise FileNotFoundError(
            f"No EDF files matched {edf_glob!r} under {str(data_path)!r}"
        )

    epoch_n = int(round(float(task_cfg.epoch_duration_s) * float(target_sfreq)))
    reject_thresh = eeg_cfg.reject_peak_to_peak

    all_windows: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_session_ids: list[np.ndarray] = []
    n_trials = 0
    n_files_used = 0

    for session_id, edf_path in enumerate(edf_paths):
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
        raw = standardize_offline_raw(raw)
        stim_channel = find_stim_channel(raw)

        original_sfreq = float(raw.info["sfreq"])
        events = mne.find_events(raw, stim_channel=stim_channel, min_duration=0.0, verbose=False)
        if len(events) == 0:
            continue
        event_times_s = events[:, 0].astype(np.float64) / original_sfreq

        if abs(original_sfreq - float(target_sfreq)) > 1e-6:
            raw.resample(float(target_sfreq), npad="auto", verbose="ERROR")

        picks, missing = resolve_channel_order(raw.ch_names, target_channel_names)
        if missing:
            raise RuntimeError(
                f"EDF {edf_path} is missing required channels {missing}. "
                f"Available channels: {raw.ch_names}"
            )

        eeg_data = raw.get_data(picks=picks).astype(np.float32, copy=False)
        eeg_data *= float(task_cfg.offline_eeg_scale_to_match_live)
        eeg_data_filt = filter_session(
            block=eeg_data,
            eeg_cfg=eeg_cfg,
            sfreq=float(target_sfreq),
        )
        file_used = False

        for event_time_s, event_code in zip(event_times_s, events[:, 2]):
            if not stim_cfg.is_lr_code(int(event_code)):
                continue

            start = int(round(float(event_time_s) * float(target_sfreq)))
            stop = start + epoch_n
            if start < 0 or stop > eeg_data_filt.shape[1]:
                continue

            epoch = eeg_data_filt[:, start:stop]
            
            # '''
            windows = split_windows(
                block=epoch,
                sfreq=float(target_sfreq),
                window_s=float(task_cfg.window_s),
                step_s=float(task_cfg.window_step_s),
            )
            # '''

            # windows = np.array([epoch]) # take the whole epoch as one window. alternatively set window length to be the same as the epoch length (3.0s)
            if windows.shape[0] == 0:
                continue

            if reject_thresh is not None:
                keep_mask = np.ptp(windows, axis=-1).max(axis=1) <= float(reject_thresh)
                windows = windows[keep_mask]
            if windows.shape[0] == 0:
                continue

            all_windows.append(windows)
            all_labels.append(np.full(windows.shape[0], int(event_code), dtype=int))
            all_session_ids.append(np.full(windows.shape[0], int(session_id), dtype=int))
            n_trials += 1
            file_used = True

        if file_used:
            n_files_used += 1

    if not all_windows:
        raise RuntimeError(
            "No usable left/right windows were extracted from the EDF folder. "
            "Check the path, trigger codes, and channel names."
        )

    X = np.concatenate(all_windows, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(all_labels, axis=0).astype(int, copy=False)
    session_ids = np.concatenate(all_session_ids, axis=0).astype(int, copy=False)
    return OfflineMIDataset(
        X=X,
        y=y,
        session_ids=session_ids,
        sfreq=float(target_sfreq),
        channel_names=[str(name) for name in target_channel_names],
        eeg_units=str(task_cfg.live_eeg_units),
        offline_scale_applied=float(task_cfg.offline_eeg_scale_to_match_live),
        n_files_found=len(edf_paths),
        n_files_used=n_files_used,
        n_trials=n_trials,
        n_windows=int(X.shape[0]),
    )

def append_windows_to_dataset(
    dataset: OfflineMIDataset,
    windows: np.ndarray,
    labels: np.ndarray,
    session_id: int,
    n_trials_add: int = 0,
) -> OfflineMIDataset:
    X_add = np.asarray(windows, dtype=np.float32)
    y_add = np.asarray(labels, dtype=int)
    if X_add.ndim != 3:
        raise ValueError(f"Expected windows shape (n_windows, n_channels, n_samples), got {X_add.shape}")
    if y_add.ndim != 1 or y_add.shape[0] != X_add.shape[0]:
        raise ValueError(f"Expected labels shape ({X_add.shape[0]},), got {y_add.shape}")

    if X_add.shape[0] == 0:
        return dataset

    session_add = np.full(X_add.shape[0], int(session_id), dtype=int)
    X = np.concatenate((dataset.X, X_add), axis=0).astype(np.float32, copy=False)
    y = np.concatenate((dataset.y, y_add), axis=0).astype(int, copy=False)
    session_ids = np.concatenate((dataset.session_ids, session_add), axis=0).astype(int, copy=False)
    return OfflineMIDataset(
        X=X,
        y=y,
        session_ids=session_ids,
        sfreq=float(dataset.sfreq),
        channel_names=list(dataset.channel_names),
        eeg_units=str(dataset.eeg_units),
        offline_scale_applied=float(dataset.offline_scale_applied),
        n_files_found=int(dataset.n_files_found),
        n_files_used=int(dataset.n_files_used),
        n_trials=int(dataset.n_trials) + int(n_trials_add),
        n_windows=int(X.shape[0]),
    )


# Kalman Filter Stuff
class PostLDA_KalmanSmoother:
    def __init__(self, q=0.005, r=0.1):
        # kalman filters have matrix vars p, q, r, s; x represents state (in this case, the prediction- L/R)
        self.x = 0.0
        self.p = 1.0 # represents cov
        self.q = q
        self.r = r
    
    def step(self, z, r_adapted=False):
        r_current = r_adapted if r_adapted is not None else self.r
        self.p = self.p + self.q # make prediction about p

        k = self.p / (self.p + r_current) # update k
        self.x = self.x + k * (z - self.x)
        self.p = (1 - k) * self.p
        return self.x
    
