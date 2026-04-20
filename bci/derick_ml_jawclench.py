from __future__ import annotations

import logging
import math
from typing import Any, Callable

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


def train_jaw_clench_classifier(
    feature_rows: list[np.ndarray] | np.ndarray,
    labels: list[int] | np.ndarray,
    *,
    min_total_samples: int = 6,
) -> tuple[Pipeline, float, np.ndarray, np.ndarray]:
    """Train jaw-clench classifier from precomputed feature rows.

    Returns (classifier, train_accuracy, X, y).
    """
    X = np.asarray(feature_rows, dtype=np.float32)
    y = np.asarray(labels, dtype=int)

    if X.ndim != 2:
        raise ValueError(f"Expected feature_rows with shape (n_samples, n_features), got {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"Expected labels with shape (n_samples,), got {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatched sample counts: X has {X.shape[0]}, y has {y.shape[0]}")
    if X.shape[0] < int(min_total_samples) or len(np.unique(y)) < 2:
        raise ValueError(
            "Jaw calibration failed: not enough usable rest/clench samples. "
            "Please rerun and reduce movement/blinks during REST trials."
        )

    clf = build_jaw_clench_classifier()
    clf.fit(X, y)
    train_acc = float(clf.score(X, y))
    return clf, train_acc, X, y


def run_visual_jaw_calibration(
    *,
    cue: Any,
    info: Any,
    status: Any,
    wait_for_space: Callable[[str], None],
    wait_for_seconds: Callable[[float], None],
    collect_stream_block: Callable[[float], np.ndarray],
    jaw_idxs: list[int],
    jaw_window_n: int,
    model_ch_names: list[str],
    logger: logging.Logger,
    n_per_class: int = 5,
    hold_s: float = 1.2,
    prep_s: float = 2.5,
    iti_s: float = 1.5,
    min_total_samples: int = 6,
) -> tuple[Pipeline, float, np.ndarray]:
    """Run the PsychoPy jaw calibration flow and train the classifier.

    The UI objects and timing/data callbacks are task-provided so this stays
    reusable while preserving the exact task behavior.
    """
    original_layout = {
        "cue_pos": cue.pos,
        "info_pos": info.pos,
        "status_pos": status.pos,
        "cue_h": cue.height,
        "info_h": info.height,
        "status_h": status.height,
    }
    cue.pos = (0.0, 0.26)
    info.pos = (0.0, 0.10)
    status.pos = (0.0, -0.06)
    cue.height = 0.065
    info.height = 0.052
    status.height = 0.048

    min_samples = int(jaw_window_n)

    cue.text = "Jaw calibration"
    info.text = "We will collect REST and JAW CLENCH trials to train a pause classifier."
    status.text = "Press SPACE to begin calibration. ESC to quit."
    wait_for_space("Jaw calibration")

    try:
        labels = [0] * int(n_per_class) + [1] * int(n_per_class)
        np.random.default_rng().shuffle(labels)
        X_cal: list[np.ndarray] = []
        y_cal: list[int] = []

        for i, y_label in enumerate(labels, start=1):
            is_clench = bool(y_label == 1)
            trial_name = "JAW CLENCH" if is_clench else "REST"

            cue.text = "Prepare"
            info.text = f"Next trial: {trial_name}"
            status.text = f"Calibration {i}/{len(labels)}"
            wait_for_seconds(float(prep_s))

            cue.text = trial_name
            info.text = (
                "Clench jaw and hold." if is_clench else "Relax face and avoid blinking/movement."
            )
            status.text = f"Hold for {float(hold_s):.1f}s"
            block = collect_stream_block(float(hold_s))

            if block.shape[1] >= min_samples:
                feat_block = block[:, -min_samples:]
                X_cal.append(extract_jaw_features(feat_block, jaw_idxs))
                y_cal.append(int(y_label))
            else:
                logger.warning(
                    "Skipping short calibration block %d: samples=%d, needed=%d",
                    i,
                    int(block.shape[1]),
                    min_samples,
                )

            cue.text = "Relax"
            info.text = "Short break"
            status.text = ""
            wait_for_seconds(float(iti_s))

        jaw_classifier, train_acc, _X_np, y_np = train_jaw_clench_classifier(
            feature_rows=X_cal,
            labels=y_cal,
            min_total_samples=int(min_total_samples),
        )
        logger.info(
            "Jaw calibration complete: samples=%d, rest=%d, clench=%d, train_acc=%.3f, jaw_channels=%s",
            int(len(y_np)),
            int(np.sum(y_np == 0)),
            int(np.sum(y_np == 1)),
            train_acc,
            [model_ch_names[idx] for idx in jaw_idxs],
        )

        cue.text = "Calibration complete"
        info.text = f"Jaw classifier ready (train acc {train_acc:.2f})"
        status.text = "Jaw clench toggles pause/play. Press SPACE to start task."
        wait_for_space("Calibration complete")
        return jaw_classifier, train_acc, y_np
    finally:
        cue.pos = original_layout["cue_pos"]
        info.pos = original_layout["info_pos"]
        status.pos = original_layout["status_pos"]
        cue.height = original_layout["cue_h"]
        info.height = original_layout["info_h"]
        status.height = original_layout["status_h"]


def update_live_jaw_clench_state(
    *,
    jaw_buffer: np.ndarray,
    x_new: np.ndarray,
    keep_n: int,
    jaw_window_n: int,
    jaw_classifier: Pipeline | None,
    jaw_idxs: list[int],
    jaw_prob: float,
    jaw_prev_pred: int,
    jaw_prob_thresh: float,
    jaw_last_toggle_t: float,
    jaw_refractory_s: float,
    now_t: float,
) -> tuple[np.ndarray, float, int, bool]:
    """Update jaw-clench inference state from fresh live samples.

    Returns (updated_buffer, jaw_prob, jaw_prev_pred, should_toggle).
    """
    X_new = np.asarray(x_new, dtype=np.float32)
    if X_new.ndim != 2:
        raise ValueError(f"Expected x_new shape (n_channels, n_samples), got {X_new.shape}")

    updated_buffer = np.concatenate((jaw_buffer, X_new), axis=1)
    max_keep = max(int(keep_n), int(jaw_window_n))
    if updated_buffer.shape[1] > max_keep:
        updated_buffer = updated_buffer[:, -max_keep:]

    updated_prob = float(jaw_prob)
    updated_prev_pred = int(jaw_prev_pred)
    should_toggle = False

    if jaw_classifier is not None and updated_buffer.shape[1] >= int(jaw_window_n):
        jaw_win = updated_buffer[:, -int(jaw_window_n):]
        feat = extract_jaw_features(jaw_win, jaw_idxs).reshape(1, -1)
        updated_prob = float(jaw_classifier.predict_proba(feat)[0, 1])
        jaw_pred = int(updated_prob >= float(jaw_prob_thresh))
        if jaw_pred == 1 and updated_prev_pred == 0 and (float(now_t) - float(jaw_last_toggle_t)) >= float(jaw_refractory_s):
            should_toggle = True
        updated_prev_pred = jaw_pred

    return updated_buffer, updated_prob, updated_prev_pred, should_toggle


def normalized_deadband(value: float, deadband: float) -> float:
    """Utility helper for downstream tasks that use jaw outputs in control loops."""
    value = float(np.clip(value, -1.0, 1.0))
    deadband = float(np.clip(deadband, 0.0, 0.99))
    if abs(value) <= deadband:
        return 0.0
    scaled = (abs(value) - deadband) / (1.0 - deadband)
    return float(math.copysign(scaled, value))
