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
    KnobTaskConfig,
    StimConfig,
)
from bci_runtime import apply_runtime_config_overrides, resolve_shared_mi_model
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
    return f"{date_prefix}_{participant}_knob_task"


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


def _sample_target( # altered to generate a target region for the knob instead
    rng: np.random.Generator,
    min_distance: float,
    radius: float,
    prev_target: float
) -> np.ndarray:
    for _ in range(1000):
        candidate = rng.uniform(-np.pi, np.pi)
        dist = abs((candidate - prev_target + math.pi) % (2 * math.pi) - math.pi)
        if dist >= min_distance:
            return [candidate - radius, candidate + radius]
    raise RuntimeError("Failed to sample a valid target location.")

def create_region_of_death(target_pos: np.ndarray, radius: float):
    target_start = target_pos[0]
    target_end = target_pos[1]
    target_center = (target_start + target_end)/2.0 # all three in radians
    rod_center = 0

    if np.abs(np.abs(target_center) - math.pi) <= radius: # if the target is pi radians away
        rod_center = np.random.choice([-math.pi/2, math.pi/2])
        return [rod_center - radius, rod_center + radius]

    rod_center = _wrap_angle(target_center + np.pi)
    return [rod_center - radius, rod_center + radius]

def create_target_region(start_angle_rad: float, end_angle_rad: float, radius: float, num_points: int = 25):
    vertices = [(0, 0)] # Start at center
    step = (end_angle_rad - start_angle_rad) / (num_points - 1)
    
    for i in range(num_points):
        angle = start_angle_rad + i * step
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        vertices.append((x, y))
        
    return vertices

def run_task(fname: str, max_trials: int | None = None) -> None:
    logger = _make_task_logger(fname)
    lsl_cfg = LSLConfig()
    stim_cfg = StimConfig()
    label_cfg = MentalCommandLabelConfig()
    task_cfg = KnobTaskConfig()
    model_cfg = MentalCommandModelConfig()
    eeg_cfg = EEGConfig(
        l_freq=8.0,
        h_freq=30.0,
        reject_peak_to_peak=150.0,
    )
    cfgs = apply_runtime_config_overrides(
        "knob_task",
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
    if missing:
        raise RuntimeError(
            f"Live stream is missing configured EEG picks {missing}. "
            f"Configured picks: {list(eeg_cfg.picks)}. Available channels: {available}"
        )
    if len(model_ch_names) < 2:
        raise RuntimeError(
            "Need at least 2 configured EEG channels after applying picks. "
            f"Configured picks: {list(eeg_cfg.picks)}. Resolved channels: {model_ch_names}. "
            f"Available channels: {available}"
        )

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
    accent = (0.88, 0.92, 0.96)
    success_color = (0.38, 0.92, 0.56)
    failure_color = (1, -1, -1)

    # draw knob
    knob = visual.Circle(
        win, 
        radius=task_cfg.knob_radius, 
        fillColor=[0, 0, 0], 
        lineColor=[0.5, 0.5, 0.5], 
        lineWidth=3
    )

    title = visual.TextStim(win, text="Motor Imagery Knob Task", pos=(0, 0.90), height=0.055, color=white)
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

    heading_line = visual.Line(
        win,
        start=(0.0, 0.0),
        end=(0.0, 0.08),
        lineColor=white,
        lineWidth=3.0,
    )

    target_pos = _sample_target(
            rng=rng,
            min_distance=float(task_cfg.min_angular_distance),
            radius=float(task_cfg.knob_radius),
            prev_target=0.0
        )

    rod_pos = create_region_of_death(
            target_pos=target_pos,
            radius=float(task_cfg.knob_radius)
        )

    target_region_vertices = create_target_region(target_pos[0], target_pos[1], task_cfg.knob_radius) # change this 100 if necessary
    rod_vertices = create_target_region(rod_pos[0], rod_pos[1], task_cfg.knob_radius)

    target_region = visual.ShapeStim(
        win, 
        vertices=target_region_vertices, 
        fillColor=[0, 0.5, 0],   # Dark green
        lineColor=None,
        opacity=0.6
    )
    
    region_of_death = visual.ShapeStim(
        win, 
        vertices=rod_vertices, 
        fillColor=[1, -1, -1],   # RED
        lineColor=None,
        opacity=0.6
    )

    heading_rad = task_cfg.start_angle
    steering_state = 0.0

    def _update_knob_visual() -> None:
        nose_len = float(task_cfg.knob_radius) * 1.2 # the 1.2 can be altered, I just made it up
        heading_line.start = (0.0, 0.0)
        heading_line.end = (
            float(math.cos(heading_rad) * nose_len),
            float(math.sin(heading_rad) * nose_len),
        )

    def _draw_frame() -> None:
        arena_outline.draw()
        knob.draw()
        target_region.draw()
        region_of_death.draw()
        heading_line.draw()
        title.draw()
        cue.draw()
        info.draw()
        status.draw()
        win.flip()

    def _reset_trial_state() -> None:
        nonlocal heading_rad, steering_state
        heading_rad = task_cfg.start_angle
        steering_state = 0.0
        _update_knob_visual()

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
            f"Use {label_cfg.left_name}/{label_cfg.right_name} motor imagery to turn the knob and reach the target. ESC to quit."
        )
        _reset_trial_state()
        target_pos = _sample_target(
            rng=rng,
            min_distance=float(task_cfg.min_angular_distance),
            radius=float(task_cfg.knob_radius),
            prev_target=np.mean(target_pos)
        )

        rod_pos = create_region_of_death(
            target_pos=target_pos,
            radius=float(task_cfg.knob_radius)
        )

        target_region_vertices = create_target_region(target_pos[0], target_pos[1], task_cfg.knob_radius) # change this 100 if necessary
        rod_vertices = create_target_region(rod_pos[0], rod_pos[1], radius=task_cfg.knob_radius)

        target_region.vertices = target_region_vertices
        region_of_death.vertices = rod_vertices

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
    bias_offset = float(task_cfg.live_bias_offset) if bool(task_cfg.enable_live_bias_offset) else 0.0

    def _poll_live_decoder() -> None:
        nonlocal last_live_ts, live_buffer, prediction_count
        nonlocal left_prob, right_prob, rest_prob, raw_command, ema_command, live_note

        data, ts = stream.get_data(winsize=stream_pull_s, picks="all")
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
        if live_buffer.shape[1] < keep_n:
            needed_s = max(0.0, (keep_n - live_buffer.shape[1]) / sfreq)
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
        alpha = float(np.clip(task_cfg.command_ema_alpha, 0.0, 1.0))
        if prediction_count == 0:
            ema_command = raw_command
        else:
            ema_command = (1.0 - alpha) * ema_command + alpha * raw_command
        prediction_count += 1
        live_note = "tracking"

        if prediction_count % 20 == 0:
            logger.info(
                "Decode %d: left_p=%.4f, right_p=%.4f, rest_p=%.4f, raw_command=%.4f, ema_command=%.4f, bias_offset=%.4f",
                prediction_count,
                left_prob,
                right_prob,
                rest_prob,
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
            _draw_frame()
            keys = event.getKeys()
            if "escape" in keys:
                raise KeyboardInterrupt
            if "space" in keys:
                return

    def _settle_before_trial() -> None:
        settle_clock = core.Clock()
        while settle_clock.getTime() < task_cfg.trial_start_delay_s:
            if "escape" in event.getKeys():
                raise KeyboardInterrupt
            _poll_live_decoder()
            remaining = max(0.0, task_cfg.trial_start_delay_s - settle_clock.getTime())
            cue.text = f"Trial starting in {remaining:.1f}s"
            parts = [
                f"{label_cfg.left_name}: {left_prob:.2f}",
                f"{label_cfg.right_name}: {right_prob:.2f}",
            ]
            if int(task_cfg.rest_class_code) in class_index:
                parts.append(f"{label_cfg.rest_name}: {rest_prob:.2f}")
            parts.extend([f"cmd={ema_command:+.2f}", f"bias={bias_offset:+.2f}", live_note])
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
            # target.fillColor = target_color
            target_pos = _sample_target(
                rng=rng,
                min_distance=float(task_cfg.min_angular_distance),
                radius=task_cfg.knob_radius,
                prev_target=float((target_pos[0] + target_pos[1])/2.0)
            )
            rod_pos = create_region_of_death(
                target_pos=target_pos, radius=task_cfg.knob_radius
            )
            target_region.vertices = create_target_region(target_pos[0], target_pos[1], task_cfg.knob_radius)
            region_of_death.vertices = create_region_of_death(rod_vertices[0], rod_vertices[1], radius=task_cfg.knob_radius)
            status.text = (
                f"Trials completed: {completed_trials}\n"
                "Press SPACE to start the next target. ESC to stop."
            )
            _wait_for_space("Reach the target with left/right motor imagery")

            _reset_trial_state()
            ema_command = 0.0
            raw_command = 0.0
            left_prob = 0.5
            right_prob = 0.5
            rest_prob = 0.0
            live_note = "settling"
            _settle_before_trial()

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

                _poll_live_decoder()

                now_t = core.getTime()
                dt = float(np.clip(now_t - last_frame_t, 1e-4, 0.05))
                last_frame_t = now_t

                command_drive = _apply_deadband(ema_command, float(task_cfg.command_deadband))
                if task_cfg.steering_time_constant_s <= 1e-6:
                    steering_state = command_drive
                else:
                    blend = float(np.clip(dt / task_cfg.steering_time_constant_s, 0.0, 1.0))
                    steering_state += (command_drive - steering_state) * blend

                turn_rate_rad_s = task_cfg.max_turn_rate_deg_s
                heading_rad = _wrap_angle(
                    heading_rad - math.radians(turn_rate_rad_s) * steering_state * dt
                )

                # heading_line.ori = math.degrees(heading_rad)

                _update_knob_visual()

                mean_abs_command_sum += abs(command_drive)
                mean_raw_command_sum += raw_command
                command_samples += 1

                target_center = (target_pos[0] + target_pos[1]) / 2.0
                angular_dist_to_center = abs((heading_rad - target_center + math.pi) % (2 * math.pi) - math.pi)
                target_region_reached = angular_dist_to_center <= task_cfg.knob_radius
                
                rod_center = (rod_pos[0] + rod_pos[1]) / 2.0
                angular_dist_to_rod = abs((heading_rad - rod_center + math.pi) % (2 * math.pi) - math.pi)
                rod_reached = angular_dist_to_rod <= task_cfg.knob_radius

                cue.text = f"Trial {completed_trials + 1}"
                parts = [
                    f"{label_cfg.left_name}: {left_prob:.2f}",
                    f"{label_cfg.right_name}: {right_prob:.2f}",
                ]
                if int(task_cfg.rest_class_code) in class_index:
                    parts.append(f"{label_cfg.rest_name}: {rest_prob:.2f}")
                parts.extend([
                    f"raw={raw_command:+.2f}",
                    f"ema={ema_command:+.2f}",
                    f"bias={bias_offset:+.2f}",
                    f"steer={steering_state:+.2f}",
                ])
                info.text = "   ".join(parts)
                status.text = (
                    f"time={trial_clock.getTime():.1f}s   "
                    f"distance={angular_dist_to_center:.2f}   "
                    f"updates={prediction_count - trial_pred_start}   {live_note}"
                )
                _draw_frame()

                if target_region_reached:
                    completed_trials += 1
                    target_region.fillColor = success_color
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
                elif rod_reached:
                    completed_trials += 1
                    target_region.fillColor = failure_color
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
                    trial_results.append(result) # update this to reflect faild trial
                    logger.info("Trial complete: %s", result)

                    hit_clock = core.Clock()
                    while hit_clock.getTime() < task_cfg.post_hit_pause_s:
                        if "escape" in event.getKeys():
                            raise KeyboardInterrupt
                        _poll_live_decoder()
                        cue.text = "You Have Failed, Andy"
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
