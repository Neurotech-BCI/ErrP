from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mental_command_worker import canonicalize_channel_name


REST_CLASS_CODE = 0
JAW_CLENCH_CLASS_CODE = 1
RAPID_BLINK_CLASS_CODE = 2


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


def split_overlapping_windows(
    block: np.ndarray,
    sfreq: float,
    window_s: float,
    step_s: float,
) -> np.ndarray:
    """Split a continuous block into overlapping windows."""
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
    """Convert long calibration blocks into windowed jaw feature rows."""
    if len(blocks) != len(labels):
        raise ValueError("blocks and labels must have the same length")

    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    trim_n = max(0, int(round(float(edge_trim_s) * float(sfreq))))
    window_n = int(round(float(window_s) * float(sfreq)))
    if window_n <= 0:
        raise ValueError("window_s must produce at least one sample")

    for block, label in zip(blocks, labels):
        X_block = np.asarray(block, dtype=np.float32)
        if X_block.ndim != 2 or X_block.shape[1] == 0:
            continue

        # Keep edge trimming, but cap it per block so at least one window can remain.
        if X_block.shape[1] > window_n:
            trim_cap = max(0, (int(X_block.shape[1]) - int(window_n)) // 2)
            trim_eff = min(trim_n, trim_cap)
        else:
            trim_eff = 0

        start = int(trim_eff)
        stop = int(X_block.shape[1] - trim_eff)
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


def collect_cue_locked_stream_block(
    *,
    stream: Any,
    sfreq: float,
    n_channels: int,
    duration_s: float,
    cue_offset_s: float = 0.5,
    render_frame: Callable[[float, float], None] | None = None,
    check_abort: Callable[[], None] | None = None,
    query_window_s: float | None = None,
    timeout_s: float | None = None,
    idle_s: float = 0.01,
    logger: logging.Logger | None = None,
    label: str = "calibration block",
) -> np.ndarray:
    """Collect a cue-locked EEG block starting after a short onset delay.

    The cue is assumed to be visually presented by ``render_frame``. This helper:
    1. snapshots the most recent stream timestamp before the cue is drawn,
    2. repeatedly polls a rolling window from the stream,
    3. accumulates only samples newer than the pre-cue timestamp, and
    4. returns the segment beginning ``cue_offset_s`` after cue onset.
    """
    duration_s = max(0.0, float(duration_s))
    cue_offset_s = max(0.0, float(cue_offset_s))
    total_capture_s = cue_offset_s + duration_s
    offset_n = max(0, int(round(cue_offset_s * float(sfreq))))
    duration_n = max(1, int(round(duration_s * float(sfreq))))
    query_s = max(0.50, float(query_window_s or (total_capture_s + 0.50)))
    max_wait_s = float(timeout_s or (total_capture_s + max(2.0, query_s)))
    sleep_s = max(0.0, float(idle_s))

    _data0, ts0 = stream.get_data(winsize=query_s, picks="all")
    ts0_arr = np.asarray(ts0, dtype=np.float64) if ts0 is not None else np.empty((0,), dtype=np.float64)
    anchor_ts = float(ts0_arr[-1]) if ts0_arr.size > 0 else None
    last_ts_local = anchor_ts

    if render_frame is not None:
        render_frame(0.0, total_capture_s)

    chunks: list[np.ndarray] = []
    ts_chunks: list[np.ndarray] = []
    start_wall = time.monotonic()

    while (time.monotonic() - start_wall) < max_wait_s:
        if check_abort is not None:
            check_abort()

        data, ts = stream.get_data(winsize=query_s, picks="all")
        if data.size > 0 and ts is not None and len(ts) > 0:
            ts_arr = np.asarray(ts, dtype=np.float64)
            if last_ts_local is None:
                mask = np.ones_like(ts_arr, dtype=bool)
            else:
                mask = ts_arr > float(last_ts_local)
            if np.any(mask):
                new_chunk = np.asarray(data[:, mask], dtype=np.float32)
                new_ts = ts_arr[mask]
                chunks.append(new_chunk)
                ts_chunks.append(new_ts)
                last_ts_local = float(new_ts[-1])

        elapsed_s = time.monotonic() - start_wall
        if render_frame is not None:
            render_frame(min(elapsed_s, total_capture_s), total_capture_s)

        if anchor_ts is not None and last_ts_local is not None and last_ts_local >= float(anchor_ts) + total_capture_s:
            break
        if sleep_s > 0.0:
            time.sleep(sleep_s)

    if not chunks:
        return np.empty((int(n_channels), 0), dtype=np.float32)

    block_all = np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
    ts_all = np.concatenate(ts_chunks, axis=0).astype(np.float64, copy=False)

    if anchor_ts is not None:
        start_ts = float(anchor_ts) + cue_offset_s
        stop_ts = start_ts + duration_s
        keep = (ts_all >= start_ts) & (ts_all < stop_ts)
        block = block_all[:, keep]
    else:
        block = block_all[:, offset_n : offset_n + duration_n]

    if block.shape[1] < duration_n and block_all.shape[1] >= offset_n + duration_n:
        block = block_all[:, offset_n : offset_n + duration_n]
    elif block.shape[1] > duration_n:
        block = block[:, :duration_n]

    if logger is not None and block.shape[1] < duration_n:
        logger.warning(
            "Cue-locked %s ended short: collected=%d expected=%d samples (duration_s=%.2f cue_offset_s=%.2f).",
            str(label),
            int(block.shape[1]),
            int(duration_n),
            float(duration_s),
            float(cue_offset_s),
        )
    return block.astype(np.float32, copy=False)


def build_jaw_clench_classifier(
    *,
    random_state: int = 42,
    class_weight: str | None = "balanced",
    solver: str = "liblinear",
    max_iter: int = 1000,
) -> Pipeline:
    """Build the default jaw-clench binary classifier pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                random_state=int(random_state),
                class_weight=class_weight,
                solver=str(solver),
                max_iter=int(max_iter),
            ),
        ),
    ])


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


def train_face_event_classifier(
    feature_rows: list[np.ndarray] | np.ndarray,
    labels: list[int] | np.ndarray,
    *,
    min_total_samples: int = 9,
    required_classes: tuple[int, ...] = (REST_CLASS_CODE, JAW_CLENCH_CLASS_CODE, RAPID_BLINK_CLASS_CODE),
) -> tuple[Pipeline, float, np.ndarray, np.ndarray, dict[int, int]]:
    """Train a multi-class face-event classifier (REST/JAW/BLINK)."""
    X = np.asarray(feature_rows, dtype=np.float32)
    y = np.asarray(labels, dtype=int)

    if X.ndim != 2:
        raise ValueError(f"Expected feature_rows with shape (n_samples, n_features), got {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"Expected labels with shape (n_samples,), got {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatched sample counts: X has {X.shape[0]}, y has {y.shape[0]}")
    if X.shape[0] < int(min_total_samples):
        raise ValueError(
            "Face-event calibration failed: not enough usable samples. "
            "Please rerun and reduce extra movement during REST trials."
        )

    present = {int(c) for c in np.unique(y)}
    missing = [int(c) for c in required_classes if int(c) not in present]
    if missing:
        raise ValueError(
            f"Face-event calibration failed: missing required classes {missing}. "
            f"Found classes: {sorted(present)}"
        )

    clf = build_jaw_clench_classifier()
    clf.fit(X, y)
    train_acc = float(clf.score(X, y))
    vals, cnts = np.unique(y, return_counts=True)
    class_counts = {int(v): int(c) for v, c in zip(vals, cnts)}
    return clf, train_acc, X, y, class_counts


def collect_visual_face_event_feature_rows(
    *,
    cue: Any,
    info: Any,
    status: Any,
    wait_for_space: Callable[[str], None],
    wait_for_seconds: Callable[[float], None],
    collect_stream_block: Callable[[float], np.ndarray],
    jaw_idxs: list[int],
    jaw_window_n: int,
    sfreq: float,
    logger: logging.Logger,
    n_per_class: int = 5,
    hold_s: float = 5.0,
    prep_s: float = 2.5,
    iti_s: float = 1.5,
    window_s: float = 0.60,
    step_s: float = 0.10,
    edge_trim_s: float = 0.5,
    include_blinks: bool = False,
    min_total_samples: int = 12,
    cue_offset_s: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Collect special-command calibration rows without fitting a classifier."""
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

    trim_n = max(0, int(round(float(edge_trim_s) * float(sfreq))))
    min_samples = int(jaw_window_n) + 2 * trim_n
    class_map = {
        REST_CLASS_CODE: ("REST", "Relax face and avoid blinking/movement."),
        JAW_CLENCH_CLASS_CODE: ("JAW CLENCH", "Clench jaw and hold."),
    }
    if include_blinks:
        class_map[RAPID_BLINK_CLASS_CODE] = ("RAPID EYE BLINKS", "Blink rapidly and repeatedly.")

    cue.text = "Face-event calibration" if include_blinks else "Jaw calibration"
    info.text = (
        "We will collect REST, JAW CLENCH, and RAPID EYE BLINK blocks."
        if include_blinks
        else "We will collect REST and JAW CLENCH trials."
    )
    status.text = "Press SPACE to begin calibration. ESC to quit."
    wait_for_space(cue.text)

    try:
        labels = [REST_CLASS_CODE] * int(n_per_class) + [JAW_CLENCH_CLASS_CODE] * int(n_per_class)
        if include_blinks:
            labels += [RAPID_BLINK_CLASS_CODE] * int(n_per_class)
        np.random.default_rng().shuffle(labels)

        calib_blocks: list[np.ndarray] = []
        calib_labels: list[int] = []
        collected_block_samples: list[int] = []
        for i, y_label in enumerate(labels, start=1):
            trial_name, instruction = class_map[int(y_label)]
            cue.text = "Prepare"
            info.text = f"Next trial: {trial_name}"
            status.text = f"Calibration {i}/{len(labels)}"
            wait_for_seconds(float(prep_s))

            cue.text = trial_name
            info.text = instruction
            if float(cue_offset_s) > 0.0:
                status.text = f"Capture lasts {float(hold_s):.1f}s"
            else:
                status.text = f"Hold for {float(hold_s):.1f}s"
            block = collect_stream_block(float(hold_s))
            collected_block_samples.append(int(block.shape[1]))
            if block.shape[1] > 0:
                calib_blocks.append(block)
                calib_labels.append(int(y_label))
                if block.shape[1] < min_samples:
                    logger.warning(
                        "Calibration block %d is short (samples=%d, nominal_needed=%d); keeping and relying on adaptive trim/windowing.",
                        i,
                        int(block.shape[1]),
                        min_samples,
                    )
            else:
                logger.warning(
                    "Skipping empty calibration block %d: samples=%d",
                    i,
                    int(block.shape[1]),
                )

            cue.text = "Relax"
            info.text = "Short break"
            status.text = ""
            wait_for_seconds(float(iti_s))

        logger.info(
            "Special-command calibration block samples: sfreq=%.3f hold_s=%.2f window_s=%.2f trim_s=%.2f nominal_needed=%d samples=%s",
            float(sfreq),
            float(hold_s),
            float(window_s),
            float(edge_trim_s),
            int(min_samples),
            collected_block_samples,
        )

        X_cal, y_cal = prepare_jaw_calibration_features(
            blocks=calib_blocks,
            labels=calib_labels,
            jaw_idxs=jaw_idxs,
            sfreq=float(sfreq),
            window_s=float(window_s),
            step_s=float(step_s),
            edge_trim_s=float(edge_trim_s),
        )
        if X_cal.shape[0] < int(min_total_samples):
            raise ValueError(
                "Special-command calibration failed: not enough usable windows. "
                f"Need at least {int(min_total_samples)} windows, got {int(X_cal.shape[0])}. "
                f"Collected block sample counts: {collected_block_samples}. "
                "Please rerun and reduce extra movement during REST trials."
            )
        vals, cnts = np.unique(y_cal, return_counts=True)
        class_counts = {int(v): int(c) for v, c in zip(vals, cnts)}
        return X_cal, y_cal, class_counts
    finally:
        cue.pos = original_layout["cue_pos"]
        info.pos = original_layout["info_pos"]
        status.pos = original_layout["status_pos"]
        cue.height = original_layout["cue_h"]
        info.height = original_layout["info_h"]
        status.height = original_layout["status_h"]


def run_visual_face_event_calibration(
    *,
    cue: Any,
    info: Any,
    status: Any,
    wait_for_space: Callable[[str], None],
    wait_for_seconds: Callable[[float], None],
    collect_stream_block: Callable[[float], np.ndarray],
    jaw_idxs: list[int],
    jaw_window_n: int,
    sfreq: float,
    model_ch_names: list[str],
    logger: logging.Logger,
    n_per_class: int = 5,
    hold_s: float = 5.0,
    prep_s: float = 2.5,
    iti_s: float = 1.5,
    window_s: float = 0.60,
    step_s: float = 0.10,
    edge_trim_s: float = 0.5,
    min_total_samples: int = 18,
    cue_offset_s: float = 0.5,
    ready_cue_text: str = "Calibration complete",
    ready_info_text: str | None = None,
    ready_status_text: str | None = None,
) -> tuple[Pipeline, float, np.ndarray, dict[int, int]]:
    """Run PsychoPy visual calibration for REST/JAW CLENCH/RAPID BLINK."""
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

    trim_n = max(0, int(round(float(edge_trim_s) * float(sfreq))))
    min_samples = int(jaw_window_n) + 2 * trim_n
    class_map = {
        REST_CLASS_CODE: ("REST", "Relax face and avoid blinking/movement."),
        JAW_CLENCH_CLASS_CODE: ("JAW CLENCH", "Clench jaw and hold."),
        RAPID_BLINK_CLASS_CODE: ("RAPID EYE BLINKS", "Blink rapidly and repeatedly."),
    }

    cue.text = "Face-event calibration"
    info.text = "We will collect REST, JAW CLENCH, and RAPID EYE BLINK blocks."
    status.text = "Press SPACE to begin calibration. ESC to quit."
    wait_for_space("Face-event calibration")

    try:
        labels = (
            [REST_CLASS_CODE] * int(n_per_class)
            + [JAW_CLENCH_CLASS_CODE] * int(n_per_class)
            + [RAPID_BLINK_CLASS_CODE] * int(n_per_class)
        )
        np.random.default_rng().shuffle(labels)

        calib_blocks: list[np.ndarray] = []
        calib_labels: list[int] = []

        for i, y_label in enumerate(labels, start=1):
            trial_name, instruction = class_map[int(y_label)]

            cue.text = "Prepare"
            info.text = f"Next trial: {trial_name}"
            status.text = f"Calibration {i}/{len(labels)}"
            wait_for_seconds(float(prep_s))

            cue.text = trial_name
            info.text = instruction
            if float(cue_offset_s) > 0.0:
                status.text = f"Capture lasts {float(hold_s):.1f}s"
            else:
                status.text = f"Hold for {float(hold_s):.1f}s"
            block = collect_stream_block(float(hold_s))

            if block.shape[1] > 0:
                calib_blocks.append(block)
                calib_labels.append(int(y_label))
                if block.shape[1] < min_samples:
                    logger.warning(
                        "Calibration block %d is short (samples=%d, nominal_needed=%d); keeping and relying on adaptive trim/windowing.",
                        i,
                        int(block.shape[1]),
                        min_samples,
                    )
            else:
                logger.warning(
                    "Skipping empty calibration block %d: samples=%d",
                    i,
                    int(block.shape[1]),
                )

            cue.text = "Relax"
            info.text = "Short break"
            status.text = ""
            wait_for_seconds(float(iti_s))

        X_cal, y_cal = prepare_jaw_calibration_features(
            blocks=calib_blocks,
            labels=calib_labels,
            jaw_idxs=jaw_idxs,
            sfreq=float(sfreq),
            window_s=float(window_s),
            step_s=float(step_s),
            edge_trim_s=float(edge_trim_s),
        )

        face_classifier, train_acc, _X_np, y_np, class_counts = train_face_event_classifier(
            feature_rows=X_cal,
            labels=y_cal,
            min_total_samples=int(min_total_samples),
        )
        logger.info(
            "Face-event calibration complete: windows=%d, rest=%d, jaw=%d, blink=%d, train_acc=%.3f, channels=%s, window_s=%.2f, step_s=%.2f, trim_s=%.2f",
            int(len(y_np)),
            int(class_counts.get(REST_CLASS_CODE, 0)),
            int(class_counts.get(JAW_CLENCH_CLASS_CODE, 0)),
            int(class_counts.get(RAPID_BLINK_CLASS_CODE, 0)),
            train_acc,
            [model_ch_names[idx] for idx in jaw_idxs],
            float(window_s),
            float(step_s),
            float(edge_trim_s),
        )

        cue.text = str(ready_cue_text)
        info.text = (
            str(ready_info_text)
            if ready_info_text is not None
            else f"Face-event classifier ready (train acc {train_acc:.2f})"
        )
        status.text = (
            str(ready_status_text)
            if ready_status_text is not None
            else "Rapid eye blinks toggle direction. Jaw clench performs click. Press SPACE to start."
        )
        wait_for_space(str(ready_cue_text))
        return face_classifier, train_acc, y_np, class_counts
    finally:
        cue.pos = original_layout["cue_pos"]
        info.pos = original_layout["info_pos"]
        status.pos = original_layout["status_pos"]
        cue.height = original_layout["cue_h"]
        info.height = original_layout["info_h"]
        status.height = original_layout["status_h"]


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
    sfreq: float,
    model_ch_names: list[str],
    logger: logging.Logger,
    n_per_class: int = 5,
    hold_s: float = 5.0,
    prep_s: float = 2.5,
    iti_s: float = 1.5,
    window_s: float = 0.60,
    step_s: float = 0.10,
    edge_trim_s: float = 0.5,
    min_total_samples: int = 12,
    cue_offset_s: float = 0.5,
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

    trim_n = max(0, int(round(float(edge_trim_s) * float(sfreq))))
    min_samples = int(jaw_window_n) + 2 * trim_n

    cue.text = "Jaw calibration"
    info.text = "We will collect REST and JAW CLENCH trials to train a pause classifier."
    status.text = "Press SPACE to begin calibration. ESC to quit."
    wait_for_space("Jaw calibration")

    try:
        labels = [0] * int(n_per_class) + [1] * int(n_per_class)
        np.random.default_rng().shuffle(labels)
        calib_blocks: list[np.ndarray] = []
        calib_labels: list[int] = []

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
            if float(cue_offset_s) > 0.0:
                status.text = f"Capture lasts {float(hold_s):.1f}s"
            else:
                status.text = f"Hold for {float(hold_s):.1f}s"
            block = collect_stream_block(float(hold_s))

            if block.shape[1] > 0:
                calib_blocks.append(block)
                calib_labels.append(int(y_label))
                if block.shape[1] < min_samples:
                    logger.warning(
                        "Calibration block %d is short (samples=%d, nominal_needed=%d); keeping and relying on adaptive trim/windowing.",
                        i,
                        int(block.shape[1]),
                        min_samples,
                    )
            else:
                logger.warning(
                    "Skipping empty calibration block %d: samples=%d",
                    i,
                    int(block.shape[1]),
                )

            cue.text = "Relax"
            info.text = "Short break"
            status.text = ""
            wait_for_seconds(float(iti_s))

        X_cal, y_cal = prepare_jaw_calibration_features(
            blocks=calib_blocks,
            labels=calib_labels,
            jaw_idxs=jaw_idxs,
            sfreq=float(sfreq),
            window_s=float(window_s),
            step_s=float(step_s),
            edge_trim_s=float(edge_trim_s),
        )

        jaw_classifier, train_acc, _X_np, y_np = train_jaw_clench_classifier(
            feature_rows=X_cal,
            labels=y_cal,
            min_total_samples=int(min_total_samples),
        )
        logger.info(
            "Jaw calibration complete: windows=%d, rest=%d, clench=%d, train_acc=%.3f, jaw_channels=%s, window_s=%.2f, step_s=%.2f, trim_s=%.2f",
            int(len(y_np)),
            int(np.sum(y_np == 0)),
            int(np.sum(y_np == 1)),
            train_acc,
            [model_ch_names[idx] for idx in jaw_idxs],
            float(window_s),
            float(step_s),
            float(edge_trim_s),
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


def update_live_face_event_state(
    *,
    face_buffer: np.ndarray,
    x_new: np.ndarray,
    keep_n: int,
    jaw_window_n: int,
    face_classifier: Pipeline | None,
    class_index: dict[int, int],
    jaw_idxs: list[int],
    rest_prob: float,
    jaw_prob: float,
    blink_prob: float,
    jaw_prev_pred: int,
    blink_prev_pred: int,
    jaw_prob_thresh: float,
    blink_prob_thresh: float,
    jaw_last_event_t: float,
    blink_last_event_t: float,
    jaw_refractory_s: float,
    blink_refractory_s: float,
    now_t: float,
) -> tuple[np.ndarray, float, float, float, int, int, int, bool, bool]:
    """Update multi-class face-event inference state from fresh live samples.

    Returns:
    (buffer, rest_prob, jaw_prob, blink_prob, pred_code,
     jaw_prev_pred, blink_prev_pred, should_click, should_toggle_direction)
    """
    X_new = np.asarray(x_new, dtype=np.float32)
    if X_new.ndim != 2:
        raise ValueError(f"Expected x_new shape (n_channels, n_samples), got {X_new.shape}")

    updated_buffer = np.concatenate((face_buffer, X_new), axis=1)
    max_keep = max(int(keep_n), int(jaw_window_n))
    if updated_buffer.shape[1] > max_keep:
        updated_buffer = updated_buffer[:, -max_keep:]

    updated_rest = float(rest_prob)
    updated_jaw = float(jaw_prob)
    updated_blink = float(blink_prob)
    updated_jaw_prev = int(jaw_prev_pred)
    updated_blink_prev = int(blink_prev_pred)
    pred_code = REST_CLASS_CODE
    should_click = False
    should_toggle_direction = False

    if face_classifier is not None and updated_buffer.shape[1] >= int(jaw_window_n):
        jaw_win = updated_buffer[:, -int(jaw_window_n):]
        feat = extract_jaw_features(jaw_win, jaw_idxs).reshape(1, -1)
        probs = face_classifier.predict_proba(feat)[0]

        idx_rest = class_index.get(REST_CLASS_CODE, -1)
        idx_jaw = class_index.get(JAW_CLENCH_CLASS_CODE, -1)
        idx_blink = class_index.get(RAPID_BLINK_CLASS_CODE, -1)
        if idx_rest < 0 or idx_jaw < 0 or idx_blink < 0:
            raise ValueError(
                "Face classifier does not contain required REST/JAW/BLINK classes. "
                f"Found classes: {sorted(class_index.keys())}"
            )

        updated_rest = float(probs[idx_rest])
        updated_jaw = float(probs[idx_jaw])
        updated_blink = float(probs[idx_blink])

        classes = np.asarray(getattr(face_classifier, "classes_", []), dtype=int)
        if classes.size == 0 and hasattr(face_classifier, "named_steps") and "clf" in face_classifier.named_steps:
            classes = np.asarray(face_classifier.named_steps["clf"].classes_, dtype=int)
        if classes.size == 0:
            raise ValueError("Face classifier has no classes_ attribute after fitting.")

        pred_code = int(classes[int(np.argmax(probs))])
        jaw_pred = int(pred_code == JAW_CLENCH_CLASS_CODE and updated_jaw >= float(jaw_prob_thresh))
        blink_pred = int(pred_code == RAPID_BLINK_CLASS_CODE and updated_blink >= float(blink_prob_thresh))

        if (
            jaw_pred == 1
            and updated_jaw_prev == 0
            and (float(now_t) - float(jaw_last_event_t)) >= float(jaw_refractory_s)
        ):
            should_click = True

        if (
            blink_pred == 1
            and updated_blink_prev == 0
            and (float(now_t) - float(blink_last_event_t)) >= float(blink_refractory_s)
        ):
            should_toggle_direction = True

        updated_jaw_prev = jaw_pred
        updated_blink_prev = blink_pred

    return (
        updated_buffer,
        updated_rest,
        updated_jaw,
        updated_blink,
        pred_code,
        updated_jaw_prev,
        updated_blink_prev,
        should_click,
        should_toggle_direction,
    )


def normalized_deadband(value: float, deadband: float) -> float:
    """Utility helper for downstream tasks that use jaw outputs in control loops."""
    value = float(np.clip(value, -1.0, 1.0))
    deadband = float(np.clip(deadband, 0.0, 0.99))
    if abs(value) <= deadband:
        return 0.0
    scaled = (abs(value) - deadband) / (1.0 - deadband)
    return float(math.copysign(scaled, value))
