from __future__ import annotations

import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from mental_command_worker import canonicalize_channel_name


def select_jaw_channel_indices(ch_names: list[str]) -> list[int]:
    """Pick frontal-priority channels that best capture jaw activity.

    Falls back to all channels when frontal channels are unavailable.
    """
    priority = {"FP1", "FP2", "AF3", "AF4", "F7", "F8", "F3", "F4"}
    idxs = [i for i, name in enumerate(ch_names) if canonicalize_channel_name(name) in priority]
    if idxs:
        return idxs
    return list(range(len(ch_names)))


def extract_jaw_features(block: np.ndarray, jaw_idxs: list[int]) -> np.ndarray:
    """Extract a compact 12D jaw-clench feature vector from EEG samples.

    Expected input shape is (n_channels, n_samples).
    """
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


def build_jaw_clench_classifier(
    *,
    random_state: int = 42,
    class_weight: str | None = "balanced",
    solver: str = "liblinear",
    max_iter: int = 1000,
) -> Pipeline:
    """Build the default jaw-clench binary classifier pipeline."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=int(random_state),
            class_weight=class_weight,
            solver=str(solver),
            max_iter=int(max_iter),
        ),
    )


def normalized_deadband(value: float, deadband: float) -> float:
    """Utility helper for downstream tasks that use jaw outputs in control loops."""
    value = float(np.clip(value, -1.0, 1.0))
    deadband = float(np.clip(deadband, 0.0, 0.99))
    if abs(value) <= deadband:
        return 0.0
    scaled = (abs(value) - deadband) / (1.0 - deadband)
    return float(math.copysign(scaled, value))
