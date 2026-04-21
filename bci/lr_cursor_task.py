from __future__ import annotations

import logging
import math
import pickle
from collections import Counter
from datetime import datetime

import numpy as np
from psychopy import core, event, visual

from mne_lsl.stream import StreamLSL

from config import (
    EEGConfig,
    LSLConfig,
    MentalCommandLabelConfig,
    MentalCommandModelConfig,
    MICursorTaskConfig,
    StimConfig,
)
from mental_command_worker import (
    StreamingIIRFilter,
    append_windows_to_dataset,
    canonicalize_channel_name,
    evaluate_loso_sessions,
    load_offline_mi_dataset,
    make_mi_classifier,
    prepare_continuous_windows,
    resolve_channel_order,
)


def _make_task_logger(fname: str) -> logging.Logger:
    logger = logging.getLogger(f"lr_cursor.{fname}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    file_handler = logging.FileHandler(f"{fname}_lr_cursor.log", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _sanitize_participant_name(raw_name: str) -> str:
    cleaned = "_".join(raw_name.strip().lower().split())
    cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch == "_")
    return cleaned.strip("_")


def _build_session_prefix(participant: str) -> str:
    date_prefix = datetime.now().strftime("%m_%d_%y")
    return f"{date_prefix}_{participant}_lr_cursor"


def _prompt_session_prefix() -> str:
    while True:
        raw = input("Enter participant name: ")
        participant = _sanitize_participant_name(raw)
        if participant:
            return _build_session_prefix(participant)
        print("Participant name cannot be empty. Please try again.")


def _wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _apply_deadband(value: float, deadband: float) -> float:
    value = float(np.clip(value, -1.0, 1.0))
    deadband = float(np.clip(deadband, 0.0, 0.99))
    if abs(value) <= deadband:
        return 0.0
    scaled = (abs(value) - deadband) / (1.0 - deadband)
    return float(math.copysign(scaled, value))


def _sample_target(
    rng: np.random.Generator,
    x_limit: float,
    min_distance: float,
) -> np.ndarray:
    for _ in range(1000):
        candidate = np.array(
            [
                rng.uniform(-x_limit, x_limit),
                0,
            ],
            dtype=np.float64,
        )
        if float(np.linalg.norm(candidate)) >= min_distance:
            return candidate
    raise RuntimeError("Failed to sample a valid target location.")


def run_task(fname: str) -> None:
    logger = _make_task_logger(fname)
    lsl_cfg = LSLConfig()
    stim_cfg = StimConfig()
    label_cfg = MentalCommandLabelConfig()
    task_cfg = MICursorTaskConfig()
    model_cfg = MentalCommandModelConfig()
    eeg_cfg = EEGConfig(
        picks=("Pz", "F4", "C4", "P4", "P3", "C3", "F3"),
        l_freq=8.0,
        h_freq=30.0,
        reject_peak_to_peak=150.0,
    )

    rng = np.random.default_rng()

    stream = StreamLSL(
        bufsize=60.0,
        name=lsl_cfg.name,
        stype=lsl_cfg.stype,
        source_id=lsl_cfg.source_id,
    )
    stream.connect(acquisition_delay=0.001, processing_flags="all")
    logger.info("Connected to LSL stream: info=%s", stream.info)

    available = list(stream.info["ch_names"])
    model_ch_names, missing = resolve_channel_order(available, eeg_cfg.picks)
    if len(model_ch_names) < 2:
        event_key = canonicalize_channel_name(lsl_cfg.event_channels)
        model_ch_names = [ch for ch in available if canonicalize_channel_name(ch) != event_key]
    if len(model_ch_names) < 2:
        raise RuntimeError(f"Need >=2 EEG channels, found: {available}")

    stream.pick(model_ch_names)
    sfreq = float(stream.info["sfreq"])
    stream_ch_names = list(stream.info["ch_names"])
    logger.info(
        "Using live EEG channels: sfreq=%.3f, selected=%s, missing_configured=%s",
        sfreq,
        stream_ch_names,
        missing,
    )

    win = visual.Window(
        size=task_cfg.win_size,
        color=(-0.08, -0.08, -0.08),
        units="norm",
        fullscr=task_cfg.fullscreen,
    )

    white = (0.90, 0.90, 0.90)
    target_color = (0.96, 0.78, 0.24)
    cursor_color = (0.38, 0.84, 0.95)
    accent = (0.88, 0.92, 0.96)
    success_color = (0.38, 0.92, 0.56)
    fail_color = (0.92, 0.28, 0.24)

    arena_limit_x = 1.0 - float(task_cfg.arena_margin) - float(task_cfg.cursor_radius)
    arena_limit_y = 1.0 - float(task_cfg.arena_margin) - float(task_cfg.cursor_radius)
    target_limit_x = 1.0 - float(task_cfg.arena_margin) - float(task_cfg.target_radius)
    target_limit_y = 1.0 - float(task_cfg.arena_margin) - float(task_cfg.target_radius)

    title = visual.TextStim(win, text="Motor Imagery LR Cursor Control", pos=(0, 0.90), height=0.055, color=white)
    cue = visual.TextStim(win, text="", pos=(0, 0.76), height=0.055, color=white)
    status = visual.TextStim(win, text="", pos=(0, -0.91), height=0.040, color=(0.84, 0.84, 0.84))
    info = visual.TextStim(win, text="", pos=(0, -0.82), height=0.040, color=accent)
    arena_outline = visual.Rect(
        win,
        width=2.0 - 2.0 * task_cfg.arena_margin,
        height=2.0 - 2.0 * task_cfg.arena_margin,
        pos=(0, 0),
        lineColor=(0.28, 0.28, 0.28),
        fillColor=None,
        lineWidth=1.5,
    )
    target = visual.Circle(
        win,
        radius=task_cfg.target_radius,
        edges=64,
        pos=(0.0, 0.0),
        fillColor=target_color,
        lineColor=white,
        lineWidth=1.5,
    )
    cursor = visual.Circle(
        win,
        radius=task_cfg.cursor_radius,
        edges=64,
        pos=(0.0, 0.0),
        fillColor=cursor_color,
        lineColor=white,
        lineWidth=1.5,
    )
    heading_line = visual.Line(
        win,
        start=(0.0, 0.0),
        end=(0.0, 0.08),
        lineColor=white,
        lineWidth=3.0,
    )
    fail_line = visual.Line(
        win,
        start=(0.0, -target_limit_y),
        end=(0.0, target_limit_y),
        lineColor=fail_color,
        lineWidth=3.0,
    )

    cursor_pos = np.zeros(2, dtype=np.float64)
    heading_rad = math.pi / 2.0
    steering_state = 0.0
    target_pos = np.zeros(2, dtype=np.float64)
    fail_line_x = 0.0

    def _update_cursor_visual() -> None:
        cursor.pos = (float(cursor_pos[0]), float(cursor_pos[1]))
        nose_len = float(task_cfg.cursor_radius) * 2.3
        heading_line.start = (float(cursor_pos[0]), float(cursor_pos[1]))
        heading_line.end = (
            float(cursor_pos[0] + math.cos(heading_rad) * nose_len),
            float(cursor_pos[1] + math.sin(heading_rad) * nose_len),
        )

    def _draw_frame() -> None:
        arena_outline.draw()
        fail_line.draw()
        target.draw()
        heading_line.draw()
        cursor.draw()
        title.draw()
        cue.draw()
        info.draw()
        status.draw()
        win.flip()

    def _reset_trial_state() -> None:
        nonlocal cursor_pos, heading_rad, steering_state
        cursor_pos = np.zeros(2, dtype=np.float64)
        heading_rad = math.pi / 2.0
        steering_state = 0.0
        _update_cursor_visual()

    def _wait_for_seconds(duration_s: float) -> None:
        clock = core.Clock()
        while clock.getTime() < duration_s:
            if "escape" in event.getKeys():
                raise KeyboardInterrupt
            _draw_frame()

    def _collect_stream_block(duration_s: float) -> np.ndarray:
        chunks: list[np.ndarray] = []
        last_ts_local: float | None = None
        clock = core.Clock()
        while clock.getTime() < duration_s:
            if "escape" in event.getKeys():
                raise KeyboardInterrupt
            data, ts = stream.get_data(winsize=min(0.25, duration_s), picks="all")
            if data.size > 0 and ts is not None and len(ts) > 0:
                ts_arr = np.asarray(ts)
                mask = np.ones_like(ts_arr, dtype=bool) if last_ts_local is None else (ts_arr > float(last_ts_local))
                if np.any(mask):
                    chunks.append(np.asarray(data[:, mask], dtype=np.float32))
                    last_ts_local = float(ts_arr[mask][-1])
            _draw_frame()
        if not chunks:
            return np.empty((len(model_ch_names), 0), dtype=np.float32)
        return np.concatenate(chunks, axis=1).astype(np.float32, copy=False)

    classifier = None
    dataset = None
    class_index: dict[int, int] | None = None
    rest_session_id: int | None = None
    online_cal_session_id: int | None = None
    train_only_session_ids: set[int] = set()

    logger.info(
        "Starting offline model preparation: data_dir=%s, edf_glob=%s, window_s=%.3f, step_s=%.3f, "
        "filter_band=[%.1f, %.1f], filter_context_s=%.3f, live_update_interval_s=%.3f",
        task_cfg.data_dir,
        task_cfg.edf_glob,
        task_cfg.window_s,
        task_cfg.window_step_s,
        eeg_cfg.l_freq,
        eeg_cfg.h_freq,
        task_cfg.filter_context_s,
        task_cfg.live_update_interval_s,
    )

    cue.text = "Preparing model from offline EDF sessions..."
    info.text = "Using the same EDF preprocessing and sliding-window decoder as the live MI visualizer."
    status.text = "Please wait..."
    _draw_frame()

    try:
        dataset = load_offline_mi_dataset(
            data_dir=task_cfg.data_dir,
            edf_glob=task_cfg.edf_glob,
            eeg_cfg=eeg_cfg,
            task_cfg=task_cfg,
            stim_cfg=stim_cfg,
            target_sfreq=sfreq,
            target_channel_names=model_ch_names,
            calibrateOnParticipant=task_cfg.calirate_on_participant
        )

        if bool(task_cfg.enable_online_rest_calibration):
            cue.text = "Rest calibration"
            info.text = "Keep both hands relaxed in your motor imagery posture and look at the center."
            status.text = "Press SPACE to begin the REST baseline. ESC to quit."
            _draw_frame()
            while True:
                keys = event.getKeys()
                if "escape" in keys:
                    raise KeyboardInterrupt
                if "space" in keys:
                    break
                _draw_frame()

            cue.text = "REST baseline"
            info.text = f"Relax both hands and keep still for {task_cfg.rest_calibration_duration_s:.0f}s."
            status.text = ""
            _draw_frame()
            rest_block = _collect_stream_block(float(task_cfg.rest_calibration_duration_s))
            rest_windows = prepare_continuous_windows(
                raw_block=rest_block,
                eeg_cfg=eeg_cfg,
                sfreq=sfreq,
                window_s=float(task_cfg.window_s),
                step_s=float(task_cfg.window_step_s),
                reject_peak_to_peak=eeg_cfg.reject_peak_to_peak,
            )
            if rest_windows.shape[0] > 0:
                rest_session_id = -1
                train_only_session_ids.add(rest_session_id)
                rest_labels = np.full(rest_windows.shape[0], int(task_cfg.rest_class_code), dtype=int)
                dataset = append_windows_to_dataset(dataset, rest_windows, rest_labels, rest_session_id, n_trials_add=1)
                logger.info(
                    "Collected online REST calibration: duration_s=%.3f, raw_samples=%d, windows=%d, session_id=%d",
                    float(task_cfg.rest_calibration_duration_s),
                    int(rest_block.shape[1]),
                    int(rest_windows.shape[0]),
                    rest_session_id,
                )
            else:
                logger.warning("REST calibration produced no usable windows after preprocessing/rejection.")

        if bool(task_cfg.enable_online_lr_calibration):
            cue.text = "Live LEFT/RIGHT calibration"
            info.text = "We will collect a few sustained LEFT and RIGHT imagery blocks before the task."
            status.text = "Press SPACE to begin. ESC to quit."
            _draw_frame()
            while True:
                keys = event.getKeys()
                if "escape" in keys:
                    raise KeyboardInterrupt
                if "space" in keys:
                    break
                _draw_frame()

            online_cal_session_id = int(np.max(dataset.session_ids)) + 1
            cal_codes = (
                [int(stim_cfg.left_code)] * int(task_cfg.online_lr_calibration_reps_per_class)
                + [int(stim_cfg.right_code)] * int(task_cfg.online_lr_calibration_reps_per_class)
            )
            rng.shuffle(cal_codes)
            cal_windows: list[np.ndarray] = []
            cal_labels: list[np.ndarray] = []

            for cal_idx, code in enumerate(cal_codes, start=1):
                class_name = label_cfg.left_name if int(code) == int(stim_cfg.left_code) else label_cfg.right_name
                cue.text = "Prepare"
                info.text = f"Get ready for {class_name} motor imagery."
                status.text = f"Calibration block {cal_idx}/{len(cal_codes)}"
                _draw_frame()
                _wait_for_seconds(float(task_cfg.online_lr_calibration_prep_s))

                cue.text = f"SUSTAIN {class_name}"
                info.text = f"Hold {class_name} motor imagery for {task_cfg.online_lr_calibration_hold_s:.0f}s."
                status.text = ""
                _draw_frame()
                cal_block = _collect_stream_block(float(task_cfg.online_lr_calibration_hold_s))
                windows = prepare_continuous_windows(
                    raw_block=cal_block,
                    eeg_cfg=eeg_cfg,
                    sfreq=sfreq,
                    window_s=float(task_cfg.window_s),
                    step_s=float(task_cfg.window_step_s),
                    reject_peak_to_peak=eeg_cfg.reject_peak_to_peak,
                )
                if windows.shape[0] > 0:
                    cal_windows.append(windows)
                    cal_labels.append(np.full(windows.shape[0], int(code), dtype=int))
                logger.info(
                    "Collected live LR calibration block: idx=%d, code=%d, raw_samples=%d, windows=%d",
                    cal_idx,
                    int(code),
                    int(cal_block.shape[1]),
                    int(windows.shape[0]),
                )

                cue.text = ""
                info.text = "Relax"
                status.text = ""
                _draw_frame()
                _wait_for_seconds(float(task_cfg.online_lr_calibration_iti_s))

            if cal_windows:
                X_online = np.concatenate(cal_windows, axis=0).astype(np.float32, copy=False)
                y_online = np.concatenate(cal_labels, axis=0).astype(int, copy=False)
                dataset = append_windows_to_dataset(
                    dataset=dataset,
                    windows=X_online,
                    labels=y_online,
                    session_id=online_cal_session_id,
                    n_trials_add=len(cal_codes),
                )
                logger.info(
                    "Added live LEFT/RIGHT calibration session: session_id=%d, windows=%d, trials=%d",
                    online_cal_session_id,
                    int(X_online.shape[0]),
                    len(cal_codes),
                )
            else:
                logger.warning("Live LEFT/RIGHT calibration produced no usable windows after preprocessing/rejection.")

        classes_present = {int(c) for c in np.unique(dataset.y)}
        expected_classes = {int(stim_cfg.left_code), int(stim_cfg.right_code)}
        if bool(task_cfg.enable_online_rest_calibration) and rest_session_id is not None:
            expected_classes.add(int(task_cfg.rest_class_code))
        missing_classes = expected_classes.difference(classes_present)
        if missing_classes:
            raise RuntimeError(
                f"Training data are missing required classes. Found {sorted(classes_present)}, "
                f"expected at least {sorted(expected_classes)}."
            )

        loso = evaluate_loso_sessions(dataset, model_cfg, train_only_session_ids=train_only_session_ids)
        classifier = make_mi_classifier(model_cfg)
        classifier.fit(dataset.X, dataset.y)
        classifier_classes = np.asarray(classifier.named_steps["clf"].classes_, dtype=int)
        class_index = {int(c): i for i, c in enumerate(classifier_classes)}
        if int(stim_cfg.left_code) not in class_index or int(stim_cfg.right_code) not in class_index:
            raise RuntimeError(
                f"Classifier classes {classifier_classes.tolist()} do not contain the expected "
                f"left/right codes {[int(stim_cfg.left_code), int(stim_cfg.right_code)]}."
            )
        counts = Counter(dataset.y.tolist())
        logger.info(
            "Offline dataset ready: files_used=%d/%d, trials=%d, windows=%d, class_counts=%s, "
            "loso_mean=%.4f, loso_std=%.4f, eeg_units=%s, offline_scale_applied=%.3f, "
            "train_only_session_ids=%s, online_cal_session_id=%s",
            dataset.n_files_used,
            dataset.n_files_found,
            dataset.n_trials,
            dataset.n_windows,
            counts,
            loso.mean_accuracy,
            loso.std_accuracy,
            dataset.eeg_units,
            dataset.offline_scale_applied,
            sorted(train_only_session_ids),
            online_cal_session_id,
        )
        np.save(f"{fname}_lr_cursor_windows.npy", dataset.X)
        np.save(f"{fname}_lr_cursor_labels.npy", dataset.y)
        with open(f"{fname}_lr_cursor_model.pkl", "wb") as fh:
            pickle.dump(classifier, fh)

        session_lines = []
        for session_id, score in sorted(loso.session_scores.items()):
            if online_cal_session_id is not None and int(session_id) == int(online_cal_session_id):
                session_lines.append(f"OnlineCal: {score:.3f}")
            else:
                session_lines.append(f"S{session_id}: {score:.3f}")
        counts = Counter(dataset.y.tolist())
        rest_count_text = ""
        if int(task_cfg.rest_class_code) in counts:
            rest_count_text = f"  {label_cfg.rest_name}: {counts[int(task_cfg.rest_class_code)]}"

        cue.text = "Model ready"
        info.text = (
            f"LOSO mean={loso.mean_accuracy:.3f}  std={loso.std_accuracy:.3f}  "
            f"Window={task_cfg.window_s:.1f}s  Update={task_cfg.live_update_interval_s:.2f}s"
        )
        status.text = (
            f"{label_cfg.left_name}: {counts[int(stim_cfg.left_code)]}  "
            f"{label_cfg.right_name}: {counts[int(stim_cfg.right_code)]}"
            f"{rest_count_text}\n"
            f"{'  '.join(session_lines)}\n"
            "Press SPACE to start a target trial.\n"
            f"Use {label_cfg.left_name}/{label_cfg.right_name} motor imagery to steer the cursor. ESC to quit."
        )
        _reset_trial_state()
        target_pos = _sample_target(
            rng=rng,
            x_limit=target_limit_x,
            min_distance=float(task_cfg.target_min_distance_from_center),
        )
        target.pos = (float(target_pos[0]), float(target_pos[1]))
        _draw_frame()
    except Exception:
        try:
            stream.disconnect()
        except Exception:
            pass
        try:
            win.close()
        except Exception:
            pass
        raise

    pred_clock = core.Clock()
    live_filter = StreamingIIRFilter.from_eeg_config(
        eeg_cfg=eeg_cfg,
        sfreq=sfreq,
        n_channels=len(model_ch_names),
    )
    live_buffer = np.empty((len(model_ch_names), 0), dtype=np.float32)
    keep_n = int(round((task_cfg.window_s + task_cfg.filter_context_s) * sfreq))
    window_n = int(round(task_cfg.window_s * sfreq))
    stream_pull_s = max(0.10, task_cfg.live_update_interval_s * 2.0)
    reject_thresh = eeg_cfg.reject_peak_to_peak
    last_live_ts: float | None = None

    prediction_count = 0
    left_prob = 0.5
    right_prob = 0.5
    rest_prob = 0.0
    raw_command = 0.0
    ema_command = 0.0
    live_note = "warming up"
    latest_pred_code: int | None = None
    bias_offset = float(task_cfg.live_bias_offset) if bool(task_cfg.enable_live_bias_offset) else 0.0

    def _reset_decoder_state() -> None:
        nonlocal last_live_ts, live_buffer, prediction_count
        nonlocal left_prob, right_prob, rest_prob, raw_command, ema_command, live_note, latest_pred_code
        live_filter.reset()
        live_buffer = np.empty((len(model_ch_names), 0), dtype=np.float32)
        last_live_ts = None
        pred_clock.reset()
        prediction_count = 0
        left_prob = 0.5
        right_prob = 0.5
        rest_prob = 0.0
        raw_command = 0.0
        ema_command = 0.0
        latest_pred_code = None
        live_note = "trial reset"

    def _poll_live_decoder(pull_enabled: bool = True) -> None:
        nonlocal last_live_ts, live_buffer, prediction_count
        nonlocal left_prob, right_prob, rest_prob, raw_command, ema_command, live_note, latest_pred_code

        if pull_enabled:
            data, ts = stream.get_data(winsize=stream_pull_s, picks="all")
        else:
            data, ts = np.empty((len(model_ch_names), 0), dtype=np.float32), None
            live_note = "startup hold"
        if data.size > 0 and ts is not None and len(ts) > 0:
            ts_arr = np.asarray(ts)
            mask = np.ones_like(ts_arr, dtype=bool) if last_live_ts is None else (ts_arr > float(last_live_ts))
            if np.any(mask):
                x_new = np.asarray(data[:, mask], dtype=np.float32)
                last_live_ts = float(ts_arr[mask][-1])
                x_new_filt = live_filter.process(x_new)
                live_buffer = np.concatenate((live_buffer, x_new_filt), axis=1)
                if live_buffer.shape[1] > keep_n:
                    live_buffer = live_buffer[:, -keep_n:]

        if pred_clock.getTime() < task_cfg.live_update_interval_s:
            return

        pred_clock.reset()
        if live_buffer.shape[1] < window_n:
            needed_s = max(0.0, (window_n - live_buffer.shape[1]) / sfreq)
            raw_command = 0.0
            ema_command *= 0.95
            live_note = f"warming up ({needed_s:.1f}s)"
            return

        x_win = live_buffer[:, -window_n:]
        max_ptp = float(np.ptp(x_win, axis=-1).max())
        if reject_thresh is not None and max_ptp > float(reject_thresh):
            raw_command = 0.0
            ema_command *= 0.85
            live_note = "artifact reject"
            return

        p_vec = classifier.predict_proba(x_win[np.newaxis, ...])[0]
        left_prob = float(p_vec[class_index[int(stim_cfg.left_code)]])
        right_prob = float(p_vec[class_index[int(stim_cfg.right_code)]])
        rest_prob = (
            float(p_vec[class_index[int(task_cfg.rest_class_code)]])
            if int(task_cfg.rest_class_code) in class_index
            else 0.0
        )
        raw_command = float(np.clip(right_prob - left_prob + bias_offset, -1.0, 1.0))
        if int(task_cfg.rest_class_code) in class_index and rest_prob >= max(left_prob, right_prob):
            latest_pred_code = int(task_cfg.rest_class_code)
        else:
            latest_pred_code = int(stim_cfg.right_code) if raw_command >= 0.0 else int(stim_cfg.left_code)
        alpha = float(np.clip(task_cfg.command_ema_alpha, 0.0, 1.0))
        if prediction_count == 0:
            ema_command = raw_command
        else:
            ema_command = (1.0 - alpha) * ema_command + alpha * raw_command
        prediction_count += 1
        live_note = "tracking"

        if prediction_count % 20 == 0:
            logger.info(
                "Decode %d: left_p=%.4f, right_p=%.4f, rest_p=%.4f, pred_code=%s, raw_command=%.4f, ema_command=%.4f, bias_offset=%.4f",
                prediction_count,
                left_prob,
                right_prob,
                rest_prob,
                latest_pred_code,
                raw_command,
                ema_command,
                bias_offset,
            )

    def _wait_for_space(message: str) -> None:
        cue.text = message
        while True:
            _poll_live_decoder()
            parts = [
                f"{label_cfg.left_name}: {left_prob:.2f}",
                f"{label_cfg.right_name}: {right_prob:.2f}",
            ]
            if int(task_cfg.rest_class_code) in class_index:
                parts.append(f"{label_cfg.rest_name}: {rest_prob:.2f}")
            parts.extend([f"cmd={ema_command:+.2f}", f"bias={bias_offset:+.2f}", live_note])
            info.text = "   ".join(parts)
            status.text = (
                f"Score: {successful_trials}/{completed_trials}   "
                "Press SPACE to start the next target. ESC to stop."
            )
            _draw_frame()
            keys = event.getKeys()
            if "escape" in keys:
                raise KeyboardInterrupt
            if "space" in keys:
                return

    def _warmup_before_trial() -> None:
        warmup_clock = core.Clock()
        # Collect post-reset data during the startup delay so the first decoded
        # window begins after the delay, keeping each trial independent.
        while warmup_clock.getTime() < task_cfg.trial_start_delay_s:
            if "escape" in event.getKeys():
                raise KeyboardInterrupt
            _poll_live_decoder(pull_enabled=True)
            remaining = max(0.0, task_cfg.trial_start_delay_s - warmup_clock.getTime())
            cue.text = f"Trial starting in {remaining:.1f}s"
            parts = [
                f"{label_cfg.left_name}: {left_prob:.2f}",
                f"{label_cfg.right_name}: {right_prob:.2f}",
            ]
            if int(task_cfg.rest_class_code) in class_index:
                parts.append(f"{label_cfg.rest_name}: {rest_prob:.2f}")
            parts.extend([f"cmd={ema_command:+.2f}", f"bias={bias_offset:+.2f}", "startup delay"])
            info.text = "   ".join(parts)
            status.text = (
                f"Score: {successful_trials}/{completed_trials}   "
                f"Trial starts in {remaining:.1f}s"
            )
            _draw_frame()

    trial_results: list[dict[str, float | int | tuple[float, float]]] = []
    completed_trials = 0
    successful_trials = 0
    last_frame_t = core.getTime()

    try:
        while True:
            _reset_trial_state()
            target.fillColor = target_color
            target_pos = _sample_target(
                rng=rng,
                x_limit=target_limit_x,
                min_distance=float(task_cfg.target_min_distance_from_center),
            )
            target.pos = (float(target_pos[0]), float(target_pos[1]))
            fail_line_x = float(-target_pos[0])
            fail_line.start = (fail_line_x, -target_limit_y)
            fail_line.end = (fail_line_x, target_limit_y)
            status.text = (
                f"Score: {successful_trials}/{completed_trials}   "
                f"Completed: {completed_trials}\n"
                "Press SPACE to start the next target. ESC to stop."
            )
            _wait_for_space("Reach the target with left/right motor imagery")

            _reset_trial_state()
            _reset_decoder_state()
            _warmup_before_trial()

            trial_clock = core.Clock()
            trial_pred_start = prediction_count
            path_length = 0.0
            mean_abs_command_sum = 0.0
            mean_raw_command_sum = 0.0
            command_samples = 0
            last_frame_t = core.getTime()

            while True:
                if "escape" in event.getKeys():
                    raise KeyboardInterrupt

                now_t = core.getTime()
                _poll_live_decoder(pull_enabled=True)
                dt = float(np.clip(now_t - last_frame_t, 1e-4, 0.05))
                last_frame_t = now_t

                if latest_pred_code == int(stim_cfg.left_code):
                    proposed_pos = cursor_pos + np.array([-float(task_cfg.forward_speed_norm_s) * dt, 0.0], dtype=np.float64)
                elif latest_pred_code == int(stim_cfg.right_code):
                    proposed_pos = cursor_pos + np.array([float(task_cfg.forward_speed_norm_s) * dt, 0.0], dtype=np.float64)
                else:
                    proposed_pos = cursor_pos.copy()
                clamped_pos = np.array(
                    [
                        np.clip(proposed_pos[0], -arena_limit_x, arena_limit_x),
                        0.0,
                    ],
                    dtype=np.float64,
                )
                path_length += float(np.linalg.norm(clamped_pos - cursor_pos))
                cursor_pos[:] = clamped_pos
                _update_cursor_visual()

                mean_abs_command_sum += abs(raw_command)
                mean_raw_command_sum += raw_command
                command_samples += 1

                distance_to_target = float(np.linalg.norm(cursor_pos - target_pos))
                cue.text = f"Trial {completed_trials + 1}"
                parts = [
                    f"{label_cfg.left_name}: {left_prob:.2f}",
                    f"{label_cfg.right_name}: {right_prob:.2f}",
                ]
                if int(task_cfg.rest_class_code) in class_index:
                    parts.append(f"{label_cfg.rest_name}: {rest_prob:.2f}")
                parts.extend([
                    f"pred={latest_pred_code}",
                    f"raw={raw_command:+.2f}",
                    f"bias={bias_offset:+.2f}",
                ])
                info.text = "   ".join(parts)
                status.text = (
                    f"score={successful_trials}/{completed_trials}   "
                    f"time={trial_clock.getTime():.1f}s   "
                    f"distance={distance_to_target:.2f}   "
                    f"updates={prediction_count - trial_pred_start}   {live_note}"
                )
                _draw_frame()

                hit_target = distance_to_target <= float(task_cfg.cursor_radius + task_cfg.target_radius)
                hit_fail_line = abs(float(cursor_pos[0]) - fail_line_x) <= float(task_cfg.cursor_radius)
                if hit_target or hit_fail_line:
                    completed_trials += 1
                    trial_success = bool(hit_target)
                    if trial_success:
                        successful_trials += 1
                        target.fillColor = success_color
                    else:
                        fail_line.lineColor = fail_color
                    trial_duration = float(trial_clock.getTime())
                    result = {
                        "trial": int(completed_trials),
                        "target_x": float(target_pos[0]),
                        "target_y": float(target_pos[1]),
                        "fail_line_x": float(fail_line_x),
                        "success": int(trial_success),
                        "outcome": "target" if trial_success else "fail_line",
                        "duration_s": trial_duration,
                        "path_length": float(path_length),
                        "mean_abs_command": float(mean_abs_command_sum / max(command_samples, 1)),
                        "mean_raw_command": float(mean_raw_command_sum / max(command_samples, 1)),
                        "prediction_updates": int(prediction_count - trial_pred_start),
                    }
                    trial_results.append(result)
                    logger.info("Trial complete: %s", result)

                    hit_clock = core.Clock()
                    while hit_clock.getTime() < task_cfg.post_hit_pause_s:
                        if "escape" in event.getKeys():
                            raise KeyboardInterrupt
                        _poll_live_decoder()
                        cue.text = "Target reached" if trial_success else "Hit mirrored fail line"
                        info.text = (
                            f"time={trial_duration:.1f}s   path={path_length:.2f}   "
                            f"updates={prediction_count - trial_pred_start}   "
                            f"score={successful_trials}/{completed_trials}"
                        )
                        status.text = (
                            "Success" if trial_success else "Trial lost"
                        ) + "   Press SPACE for the next target after the pause."
                        _draw_frame()
                    fail_line.lineColor = fail_color
                    break

        # Unreachable because the loop exits via KeyboardInterrupt.
    except KeyboardInterrupt:
        logger.info("Session interrupted by user.")
    finally:
        if classifier is not None and trial_results:
            with open(f"{fname}_lr_cursor_trials.pkl", "wb") as fh:
                pickle.dump(trial_results, fh)
            logger.info("Saved %d completed trials to %s_lr_cursor_trials.pkl", len(trial_results), fname)
        try:
            stream.disconnect()
        except Exception:
            pass
        try:
            win.close()
        except Exception:
            pass


if __name__ == "__main__":
    fname = _prompt_session_prefix()
    print(f"[SESSION] Using filename prefix: {fname}")
    run_task(fname=fname)
