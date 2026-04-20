from __future__ import annotations

import numpy as np

from mental_command_worker import canonicalize_channel_name


def select_jaw_channel_indices(ch_names: list[str]) -> list[int]:
    priority = {"FP1", "FP2", "AF3", "AF4", "F7", "F8", "F3", "F4"}
    idxs = [i for i, name in enumerate(ch_names) if canonicalize_channel_name(name) in priority]
    if idxs:
        return idxs
    return list(range(len(ch_names)))


def extract_jaw_features(block: np.ndarray, jaw_idxs: list[int]) -> np.ndarray:
    X = np.asarray(block, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected shape (n_channels, n_samples), got {X.shape}")
    if X.shape[1] < 2:
        return np.zeros(12, dtype=np.float32)

    if jaw_idxs:
        X = X[jaw_idxs]

    X = X - np.mean(X, axis=1, keepdims=True)
    ptp = np.ptp(X, axis=1)
    rms = np.sqrt(np.mean(np.square(X), axis=1))
    mav = np.mean(np.abs(X), axis=1)
    line_len = np.mean(np.abs(np.diff(X, axis=1)), axis=1)

    def _summary(v: np.ndarray) -> list[float]:
        return [float(np.mean(v)), float(np.max(v)), float(np.std(v))]

    features = _summary(ptp) + _summary(rms) + _summary(mav) + _summary(line_len)
    return np.asarray(features, dtype=np.float32)


def split_overlapping_windows(
    block: np.ndarray,
    sfreq: float,
    window_s: float,
    step_s: float,
) -> np.ndarray:
    X = np.asarray(block, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected shape (n_channels, n_samples), got {X.shape}")
    window_n = int(round(float(window_s) * float(sfreq)))
    step_n = int(round(float(step_s) * float(sfreq)))
    if window_n <= 0 or step_n <= 0:
        raise ValueError("window_s and step_s must produce at least one sample")
    if X.shape[1] < window_n:
        return np.empty((0, X.shape[0], window_n), dtype=np.float32)

    starts = np.arange(0, X.shape[1] - window_n + 1, step_n, dtype=int)
    windows = np.empty((len(starts), X.shape[0], window_n), dtype=np.float32)
    for i, start in enumerate(starts):
        windows[i] = X[:, start:start + window_n]
    return windows


def prepare_jaw_calibration_features(
    blocks: list[np.ndarray],
    labels: list[int],
    jaw_idxs: list[int],
    sfreq: float,
    window_s: float,
    step_s: float,
    edge_trim_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(blocks) != len(labels):
        raise ValueError("blocks and labels must have the same length")

    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    trim_n = max(0, int(round(float(edge_trim_s) * float(sfreq))))

    for block, label in zip(blocks, labels):
        X_block = np.asarray(block, dtype=np.float32)
        if X_block.ndim != 2 or X_block.shape[1] == 0:
            continue

        start = trim_n
        stop = X_block.shape[1] - trim_n
        if stop <= start:
            continue
        trimmed = X_block[:, start:stop]
        windows = split_overlapping_windows(
            block=trimmed,
            sfreq=float(sfreq),
            window_s=float(window_s),
            step_s=float(step_s),
        )
        for win in windows:
            X_rows.append(extract_jaw_features(win, jaw_idxs))
            y_rows.append(int(label))

    if not X_rows:
        return np.empty((0, 12), dtype=np.float32), np.empty((0,), dtype=int)
    return np.asarray(X_rows, dtype=np.float32), np.asarray(y_rows, dtype=int)
