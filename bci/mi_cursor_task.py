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
from bci_runtime import (
    apply_runtime_config_overrides,
    resolve_runtime_jaw_classifier,
    resolve_shared_mi_model,
)
from derick_ml_jawclench import (
    run_visual_jaw_calibration,
    select_jaw_channel_indices,
    update_live_jaw_clench_state,
)
from mental_command_worker import (
    StreamingIIRFilter,
    canonicalize_channel_name,
    resolve_channel_order,
)


def _make_task_logger(fname: str) -> logging.Logger:
    logger = logging.getLogger(f"mi_cursor.{fname}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    file_handler = logging.FileHandler(f"{fname}_mi_cursor.log", mode="w")
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
    return f"{date_prefix}_{participant}_mi_cursor"


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
    y_limit: float,
    min_distance: float,
) -> np.ndarray:
    for _ in range(1000):
        candidate = np.array(
            [
                rng.uniform(-x_limit, x_limit),
                rng.uniform(-y_limit, y_limit),
            ],
            dtype=np.float64,
        )
        if float(np.linalg.norm(candidate)) >= min_distance:
            return candidate
    raise RuntimeError("Failed to sample a valid target location.")


def _wrap_position(pos: np.ndarray, x_limit: float, y_limit: float) -> np.ndarray:
    wrapped = np.asarray(pos, dtype=np.float64).copy()
    width = 2.0 * float(x_limit)
    height = 2.0 * float(y_limit)
    if width > 0.0:
        wrapped[0] = ((wrapped[0] + x_limit) % width) - x_limit
    if height > 0.0:
        wrapped[1] = ((wrapped[1] + y_limit) % height) - y_limit
    return wrapped


def run_task(fname: str, max_trials: int | None = None) -> None:
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
    cfgs = apply_runtime_config_overrides(
        "mi_cursor_task",
        lsl_cfg=lsl_cfg,
        stim_cfg=stim_cfg,
        label_cfg=label_cfg,
        task_cfg=task_cfg,
        model_cfg=model_cfg,
        eeg_cfg=eeg_cfg,
    )
    lsl_cfg = cfgs["lsl_cfg"]
    stim_cfg = cfgs["stim_cfg"]
    label_cfg = cfgs["label_cfg"]
    task_cfg = cfgs["task_cfg"]
    model_cfg = cfgs["model_cfg"]
    eeg_cfg = cfgs["eeg_cfg"]

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
    # Correct for non-square displays so circles remain circular in norm units.
    if win.size[1] > 0:
        view_scale_x = float(win.size[1]) / float(win.size[0])
        win.viewScale = (view_scale_x, 1.0)
        logger.info("Applied display aspect correction: viewScale=(%.4f, 1.0000)", view_scale_x)

    white = (0.90, 0.90, 0.90)
    target_color = (0.96, 0.78, 0.24)
    cursor_color = (0.38, 0.84, 0.95)
    accent = (0.88, 0.92, 0.96)
    success_color = (0.38, 0.92, 0.56)

    arena_limit_x = 1.0 - float(task_cfg.arena_margin) - float(task_cfg.cursor_radius)
    arena_limit_y = 1.0 - float(task_cfg.arena_margin) - float(task_cfg.cursor_radius)
    target_limit_x = 1.0 - float(task_cfg.arena_margin) - float(task_cfg.target_radius)
    target_limit_y = 1.0 - float(task_cfg.arena_margin) - float(task_cfg.target_radius)

    title = visual.TextStim(win, text="Motor Imagery Cursor Control", pos=(0, 0.90), height=0.055, color=white)
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

    cursor_pos = np.zeros(2, dtype=np.float64)
    heading_rad = math.pi / 2.0
    steering_state = 0.0
    target_pos = np.zeros(2, dtype=np.float64)

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
    class_index: dict[int, int] | None = None
    jaw_classifier = None
    jaw_idxs = select_jaw_channel_indices(model_ch_names)

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
        shared_model = resolve_shared_mi_model(
            cache_name="mi_shared_lr_model",
            data_dir=task_cfg.data_dir,
            edf_glob=task_cfg.edf_glob,
            calibrate_on_participant=task_cfg.calirate_on_participant,
            eeg_cfg=eeg_cfg,
            task_cfg=task_cfg,
            stim_cfg=stim_cfg,
            model_cfg=model_cfg,
            target_sfreq=float(sfreq),
            target_channel_names=model_ch_names,
            logger=logger,
        )
        classifier = shared_model.classifier
        class_index = shared_model.class_index
        loso = shared_model.loso
        classifier_classes = np.asarray(classifier.named_steps["clf"].classes_, dtype=int)
        if int(stim_cfg.left_code) not in class_index or int(stim_cfg.right_code) not in class_index:
            raise RuntimeError(
                f"Classifier classes {classifier_classes.tolist()} do not contain the expected "
                f"left/right codes {[int(stim_cfg.left_code), int(stim_cfg.right_code)]}."
            )
        counts = Counter(shared_model.class_counts)
        logger.info(
            "Shared MI model ready: class_counts=%s, loso_mean=%.4f, loso_std=%.4f, cache=%s",
            counts,
            loso.mean_accuracy,
            loso.std_accuracy,
            shared_model.cache_path,
        )
        if shared_model.dataset is not None:
            np.save(f"{fname}_mi_cursor_windows.npy", shared_model.dataset.X)
            np.save(f"{fname}_mi_cursor_labels.npy", shared_model.dataset.y)
        with open(f"{fname}_mi_cursor_model.pkl", "wb") as fh:
            pickle.dump(classifier, fh)

        session_lines = []
        for session_id, score in sorted(loso.session_scores.items()):
            session_lines.append(f"S{session_id}: {score:.3f}")
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
            y_limit=target_limit_y,
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

    def _run_jaw_calibration() -> None:
        nonlocal jaw_classifier

        if not bool(task_cfg.enable_jaw_clench_pause):
            return

        runtime_jaw_classifier, runtime_train_acc = resolve_runtime_jaw_classifier(logger=logger, min_total_samples=12)
        if runtime_jaw_classifier is not None:
            jaw_classifier = runtime_jaw_classifier
            logger.info("Using orchestrator-provided jaw calibration (train_acc=%.3f).", float(runtime_train_acc or 0.0))
            return

        jaw_window_n_local = int(round(float(task_cfg.jaw_window_s) * sfreq))
        def _wait_for_space(prompt_text: str) -> None:
            cue.text = prompt_text
            while True:
                keys = event.getKeys()
                if "escape" in keys:
                    raise KeyboardInterrupt
                if "space" in keys:
                    return
                _draw_frame()

        jaw_classifier, _train_acc, _y_np = run_visual_jaw_calibration(
            cue=cue,
            info=info,
            status=status,
            wait_for_space=_wait_for_space,
            wait_for_seconds=_wait_for_seconds,
            collect_stream_block=_collect_stream_block,
            jaw_idxs=jaw_idxs,
            jaw_window_n=jaw_window_n_local,
            sfreq=sfreq,
            model_ch_names=model_ch_names,
            logger=logger,
            n_per_class=int(task_cfg.jaw_calibration_blocks_per_class),
            hold_s=float(task_cfg.jaw_calibration_hold_s),
            prep_s=float(task_cfg.jaw_calibration_prep_s),
            iti_s=float(task_cfg.jaw_calibration_iti_s),
            window_s=float(task_cfg.jaw_window_s),
            step_s=float(task_cfg.jaw_window_step_s),
            edge_trim_s=float(task_cfg.jaw_calibration_trim_s),
            min_total_samples=12,
        )

    _run_jaw_calibration()

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
    jaw_buffer = np.empty((len(model_ch_names), 0), dtype=np.float32)
    jaw_window_n = int(round(float(task_cfg.jaw_window_s) * sfreq))
    jaw_prob = 0.0
    jaw_prev_pred = 0
    jaw_event_pending = False
    jaw_last_toggle_t = -1e9
    jaw_prob_thresh = float(task_cfg.jaw_clench_prob_thresh)
    jaw_refractory_s = float(task_cfg.jaw_clench_refractory_s)

    prediction_count = 0
    left_prob = 0.5
    right_prob = 0.5
    rest_prob = 0.0
    decision_score = 0.0
    raw_command = 0.0
    ema_command = 0.0
    live_note = "warming up"
    bias_offset = float(task_cfg.live_bias_offset) if bool(task_cfg.enable_live_bias_offset) else 0.0
    control_mode = str(getattr(task_cfg, "mi_control_mode", "discrete_sign")).strip().lower()
    if control_mode not in {"discrete_sign", "probability_diff"}:
        raise ValueError(
            f"Unsupported mi_control_mode={task_cfg.mi_control_mode!r}. "
            "Expected 'discrete_sign' or 'probability_diff'."
        )

    def _reset_decoder_state() -> None:
        nonlocal last_live_ts, live_buffer, jaw_buffer, prediction_count
        nonlocal left_prob, right_prob, rest_prob, decision_score, raw_command, ema_command, live_note
        live_filter.reset()
        live_buffer = np.empty((len(model_ch_names), 0), dtype=np.float32)
        jaw_buffer = np.empty((len(model_ch_names), 0), dtype=np.float32)
        last_live_ts = None
        pred_clock.reset()
        prediction_count = 0
        left_prob = 0.5
        right_prob = 0.5
        rest_prob = 0.0
        decision_score = 0.0
        raw_command = 0.0
        ema_command = 0.0
        live_note = "trial reset"

    def _poll_live_decoder(pull_enabled: bool = True) -> None:
        nonlocal last_live_ts, live_buffer, jaw_buffer, prediction_count
        nonlocal left_prob, right_prob, rest_prob, decision_score, raw_command, ema_command, live_note
        nonlocal jaw_prob, jaw_prev_pred, jaw_event_pending, jaw_last_toggle_t

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
                jaw_buffer, jaw_prob, jaw_prev_pred, should_toggle = update_live_jaw_clench_state(
                    jaw_buffer=jaw_buffer,
                    x_new=x_new,
                    keep_n=keep_n,
                    jaw_window_n=jaw_window_n,
                    jaw_classifier=jaw_classifier if bool(task_cfg.enable_jaw_clench_pause) else None,
                    jaw_idxs=jaw_idxs,
                    jaw_prob=jaw_prob,
                    jaw_prev_pred=jaw_prev_pred,
                    jaw_prob_thresh=jaw_prob_thresh,
                    jaw_last_toggle_t=jaw_last_toggle_t,
                    jaw_refractory_s=jaw_refractory_s,
                    now_t=core.getTime(),
                )
                if should_toggle:
                    jaw_last_toggle_t = core.getTime()
                    jaw_event_pending = True
                x_new_filt = live_filter.process(x_new)
                live_buffer = np.concatenate((live_buffer, x_new_filt), axis=1)
                if live_buffer.shape[1] > keep_n:
                    live_buffer = live_buffer[:, -keep_n:]

        if pred_clock.getTime() < task_cfg.live_update_interval_s:
            return

        pred_clock.reset()
        if live_buffer.shape[1] < window_n:
            needed_s = max(0.0, (window_n - live_buffer.shape[1]) / sfreq)
            decision_score = 0.0
            raw_command = 0.0
            ema_command *= 0.95
            live_note = f"warming up ({needed_s:.1f}s)"
            return

        x_win = live_buffer[:, -window_n:]
        max_ptp = float(np.ptp(x_win, axis=-1).max())
        if reject_thresh is not None and max_ptp > float(reject_thresh):
            decision_score = 0.0
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
        decision_score = float(np.clip(right_prob - left_prob + bias_offset, -1.0, 1.0))
        if control_mode == "discrete_sign":
            raw_command = 1.0 if decision_score >= 0.0 else -1.0
        else:
            raw_command = decision_score
        alpha = float(np.clip(task_cfg.command_ema_alpha, 0.0, 1.0))
        if prediction_count == 0:
            ema_command = raw_command
        else:
            ema_command = (1.0 - alpha) * ema_command + alpha * raw_command
        prediction_count += 1
        live_note = "tracking"

        if prediction_count % 20 == 0:
            logger.info(
                "Decode %d: left_p=%.4f, right_p=%.4f, rest_p=%.4f, decision_score=%.4f, raw_command=%.4f, ema_command=%.4f, bias_offset=%.4f, control_mode=%s",
                prediction_count,
                left_prob,
                right_prob,
                rest_prob,
                decision_score,
                raw_command,
                ema_command,
                bias_offset,
                control_mode,
            )

    def _poll_jaw_clench() -> bool:
        nonlocal jaw_event_pending
        if not bool(task_cfg.enable_jaw_clench_pause):
            return False
        if jaw_event_pending:
            jaw_event_pending = False
            return True
        return False

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
            if bool(task_cfg.enable_jaw_clench_pause):
                parts.append(f"jaw={jaw_prob:.2f}")
            parts.extend([
                f"score={decision_score:+.2f}",
                f"cmd={ema_command:+.2f}",
                f"bias={bias_offset:+.2f}",
                f"mode={control_mode}",
                live_note,
            ])
            info.text = "   ".join(parts)
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
            if bool(task_cfg.enable_jaw_clench_pause):
                parts.append(f"jaw={jaw_prob:.2f}")
            parts.extend([
                f"score={decision_score:+.2f}",
                f"cmd={ema_command:+.2f}",
                f"bias={bias_offset:+.2f}",
                f"mode={control_mode}",
                "startup delay",
            ])
            info.text = "   ".join(parts)
            _draw_frame()

    trial_results: list[dict[str, float | int | tuple[float, float]]] = []
    completed_trials = 0
    last_frame_t = core.getTime()

    try:
        while True:
            if max_trials is not None and completed_trials >= int(max_trials):
                logger.info("Reached max_trials=%d; ending task.", int(max_trials))
                break
            _reset_trial_state()
            target.fillColor = target_color
            target_pos = _sample_target(
                rng=rng,
                x_limit=target_limit_x,
                y_limit=target_limit_y,
                min_distance=float(task_cfg.target_min_distance_from_center),
            )
            target.pos = (float(target_pos[0]), float(target_pos[1]))
            status.text = (
                f"Trials completed: {completed_trials}\n"
                "Press SPACE to start the next target. ESC to stop."
            )
            _wait_for_space("Reach the target with left/right motor imagery")

            _reset_trial_state()
            _reset_decoder_state()
            _warmup_before_trial()
            movement_enabled = not bool(task_cfg.enable_jaw_clench_pause)
            if bool(task_cfg.enable_jaw_clench_pause):
                live_note = "paused"

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
                if _poll_jaw_clench():
                    movement_enabled = not movement_enabled
                    _reset_decoder_state()
                    live_note = "resumed" if movement_enabled else "paused"
                    logger.info("Jaw pause toggle -> movement_enabled=%s", movement_enabled)
                dt = float(np.clip(now_t - last_frame_t, 1e-4, 0.05))
                last_frame_t = now_t

                command_drive = _apply_deadband(ema_command, float(task_cfg.command_deadband))
                if task_cfg.steering_time_constant_s <= 1e-6:
                    steering_state = command_drive
                else:
                    blend = float(np.clip(dt / task_cfg.steering_time_constant_s, 0.0, 1.0))
                    steering_state += (command_drive - steering_state) * blend

                if movement_enabled:
                    # Positive command means "right"; in screen coordinates that should rotate clockwise.
                    # Clockwise corresponds to decreasing the heading angle.
                    heading_rad = _wrap_angle(
                        heading_rad - math.radians(task_cfg.max_turn_rate_deg_s) * steering_state * dt
                    )

                    proposed_pos = cursor_pos + np.array(
                        [
                            math.cos(heading_rad) * task_cfg.forward_speed_norm_s * dt,
                            math.sin(heading_rad) * task_cfg.forward_speed_norm_s * dt,
                        ],
                        dtype=np.float64,
                    )
                    wrapped_pos = _wrap_position(
                        proposed_pos,
                        x_limit=arena_limit_x,
                        y_limit=arena_limit_y,
                    )
                    path_length += float(np.linalg.norm(proposed_pos - cursor_pos))
                    cursor_pos[:] = wrapped_pos
                _update_cursor_visual()

                mean_abs_command_sum += abs(command_drive)
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
                if bool(task_cfg.enable_jaw_clench_pause):
                    parts.append(f"jaw={jaw_prob:.2f}")
                parts.extend([
                    f"score={decision_score:+.2f}",
                    f"raw={raw_command:+.2f}",
                    f"ema={ema_command:+.2f}",
                    f"bias={bias_offset:+.2f}",
                    f"steer={steering_state:+.2f}",
                    f"mode={control_mode}",
                    "moving" if movement_enabled else "paused",
                ])
                info.text = "   ".join(parts)
                status.text = (
                    f"time={trial_clock.getTime():.1f}s   "
                    f"distance={distance_to_target:.2f}   "
                    f"updates={prediction_count - trial_pred_start}   {live_note}"
                )
                _draw_frame()

                if distance_to_target <= float(task_cfg.cursor_radius + task_cfg.target_radius):
                    completed_trials += 1
                    target.fillColor = success_color
                    trial_duration = float(trial_clock.getTime())
                    result = {
                        "trial": int(completed_trials),
                        "target_x": float(target_pos[0]),
                        "target_y": float(target_pos[1]),
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
                        cue.text = "Target reached"
                        info.text = (
                            f"time={trial_duration:.1f}s   path={path_length:.2f}   "
                            f"updates={prediction_count - trial_pred_start}"
                        )
                        status.text = "Press SPACE for the next target after the pause."
                        _draw_frame()
                    break

        # Unreachable because the loop exits via KeyboardInterrupt.
    except KeyboardInterrupt:
        logger.info("Session interrupted by user.")
    finally:
        if classifier is not None and trial_results:
            with open(f"{fname}_mi_cursor_trials.pkl", "wb") as fh:
                pickle.dump(trial_results, fh)
            logger.info("Saved %d completed trials to %s_mi_cursor_trials.pkl", len(trial_results), fname)
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
