from __future__ import annotations

import argparse
import logging
import math
import pickle
from collections import Counter
from datetime import datetime

import numpy as np
from psychopy import core, event, visual

from mne_lsl.stream import StreamLSL

from config import EEGConfig, LSLConfig, MentalCommandLabelConfig, MentalCommandModelConfig, MICursorTaskConfig, StimConfig
from mental_command_worker import (
    StreamingIIRFilter,
    canonicalize_channel_name,
    evaluate_loso_sessions,
    load_offline_mi_dataset,
    make_mi_classifier,
    resolve_channel_order,
)


def _make_task_logger(fname: str) -> logging.Logger:
    logger = logging.getLogger(f"jaw_pause_cursor.{fname}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    file_handler = logging.FileHandler(f"{fname}_jaw_pause_cursor.log", mode="w")
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
    return f"{date_prefix}_{participant}_jaw_pause_cursor"


def _prompt_session_prefix() -> str:
    while True:
        raw = input("Enter participant name: ")
        participant = _sanitize_participant_name(raw)
        if participant:
            return _build_session_prefix(participant)
        print("Participant name cannot be empty. Please try again.")


def _wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _select_jaw_channel_indices(ch_names: list[str]) -> list[int]:
    priority = {"FP1", "FP2", "AF3", "AF4", "F7", "F8", "F3", "F4"}
    idxs = [i for i, name in enumerate(ch_names) if canonicalize_channel_name(name) in priority]
    if idxs:
        return idxs
    # Fallback: use all available channels when frontal channels are absent.
    return list(range(len(ch_names)))


def run_task(fname: str, debug_mode: bool = False) -> None:
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

    logger.info("Starting jaw-pause cursor task | debug_mode=%s", bool(debug_mode))

    stream: StreamLSL | None = None
    model_ch_names: list[str] = list(eeg_cfg.picks)
    sfreq = 300.0

    if not debug_mode:
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
        logger.info(
            "Using live EEG channels: sfreq=%.3f, selected=%s, missing_configured=%s",
            sfreq,
            list(stream.info["ch_names"]),
            missing,
        )

    logger.info(
        "Preparing offline model: data_dir=%s, edf_glob=%s, window_s=%.3f, step_s=%.3f, sfreq=%.3f",
        task_cfg.data_dir,
        task_cfg.edf_glob,
        task_cfg.window_s,
        task_cfg.window_step_s,
        sfreq,
    )
    dataset = load_offline_mi_dataset(
        data_dir=task_cfg.data_dir,
        edf_glob=task_cfg.edf_glob,
        eeg_cfg=eeg_cfg,
        task_cfg=task_cfg,
        stim_cfg=stim_cfg,
        target_sfreq=float(sfreq),
        target_channel_names=model_ch_names,
        calibrateOnParticipant=task_cfg.calirate_on_participant,
    )
    loso = evaluate_loso_sessions(dataset, model_cfg)
    classifier = make_mi_classifier(model_cfg)
    classifier.fit(dataset.X, dataset.y)
    classifier_classes = np.asarray(classifier.named_steps["clf"].classes_, dtype=int)
    class_index = {int(c): i for i, c in enumerate(classifier_classes)}
    if int(stim_cfg.left_code) not in class_index or int(stim_cfg.right_code) not in class_index:
        raise RuntimeError(
            f"Classifier classes {classifier_classes.tolist()} do not contain expected left/right codes "
            f"{[int(stim_cfg.left_code), int(stim_cfg.right_code)]}."
        )

    counts = Counter(dataset.y.tolist())
    logger.info(
        "Offline dataset ready: files_used=%d/%d, trials=%d, windows=%d, class_counts=%s, loso_mean=%.4f, loso_std=%.4f",
        dataset.n_files_used,
        dataset.n_files_found,
        dataset.n_trials,
        dataset.n_windows,
        counts,
        loso.mean_accuracy,
        loso.std_accuracy,
    )

    np.save(f"{fname}_jaw_pause_cursor_windows.npy", dataset.X)
    np.save(f"{fname}_jaw_pause_cursor_labels.npy", dataset.y)
    with open(f"{fname}_jaw_pause_cursor_model.pkl", "wb") as fh:
        pickle.dump(classifier, fh)

    win = visual.Window(
        size=task_cfg.win_size,
        color=(-0.08, -0.08, -0.08),
        units="norm",
        fullscr=task_cfg.fullscreen,
    )

    white = (0.92, 0.92, 0.92)
    accent = (0.84, 0.90, 0.96)
    paused_color = (0.28, 0.80, 0.44)
    running_color = (0.92, 0.60, 0.22)
    arena_color = (0.28, 0.28, 0.28)

    title = visual.TextStim(win, text="Jaw Pause Rotation Cursor", pos=(0, 0.90), height=0.055, color=white)
    cue = visual.TextStim(win, text="", pos=(0, 0.78), height=0.05, color=white)
    info = visual.TextStim(win, text="", pos=(0, -0.82), height=0.040, color=accent)
    status = visual.TextStim(win, text="", pos=(0, -0.91), height=0.040, color=(0.84, 0.84, 0.84))

    arena_margin = float(task_cfg.arena_margin)
    cursor_radius = max(float(task_cfg.cursor_radius), 0.040)
    arena_limit_x = 1.0 - arena_margin - cursor_radius
    arena_limit_y = 1.0 - arena_margin - cursor_radius
    move_speed_norm_s = float(task_cfg.forward_speed_norm_s)
    arrow_len = cursor_radius * 1.6

    arena_outline = visual.Rect(
        win,
        width=2.0 - 2.0 * arena_margin,
        height=2.0 - 2.0 * arena_margin,
        pos=(0, 0),
        lineColor=arena_color,
        fillColor=None,
        lineWidth=1.5,
    )
    cursor_dot = visual.Circle(
        win,
        radius=cursor_radius,
        edges=64,
        pos=(0.0, 0.0),
        fillColor=paused_color,
        lineColor=white,
        lineWidth=1.5,
    )
    pointer = visual.Line(
        win,
        start=(0.0, 0.0),
        end=(0.0, arrow_len),
        lineColor=white,
        lineWidth=3.0,
    )

    pred_clock = core.Clock()
    last_frame_t = core.getTime()

    live_filter: StreamingIIRFilter | None = None
    live_buffer = np.empty((len(model_ch_names), 0), dtype=np.float32)
    jaw_buffer = np.empty((len(model_ch_names), 0), dtype=np.float32)
    keep_n = int(round((task_cfg.window_s + task_cfg.filter_context_s) * sfreq))
    window_n = int(round(task_cfg.window_s * sfreq))
    jaw_window_n = int(round(0.20 * sfreq))
    stream_pull_s = max(0.08, task_cfg.live_update_interval_s * 2.0)
    reject_thresh = eeg_cfg.reject_peak_to_peak
    last_live_ts: float | None = None

    jaw_idxs = _select_jaw_channel_indices(model_ch_names)
    jaw_clench_ptp_thresh = 220.0
    jaw_refractory_s = 0.70
    jaw_last_toggle_t = -1e9

    prediction_count = 0
    left_prob = 0.5
    right_prob = 0.5
    raw_command = 0.0
    ema_command = 0.0
    latest_pred_code: int | None = None
    live_note = "warming up"

    # Per your spec: rotation is only applied while paused.
    is_paused = True
    rotation_dir = 1.0
    manual_rotation_dir = 1.0
    rotation_speed_rad_s = math.radians(float(task_cfg.max_turn_rate_deg_s) * 0.60)
    heading_rad = math.pi / 2.0
    cursor_pos = np.zeros(2, dtype=np.float64)

    if stream is not None:
        live_filter = StreamingIIRFilter.from_eeg_config(
            eeg_cfg=eeg_cfg,
            sfreq=sfreq,
            n_channels=len(model_ch_names),
        )

    def _update_cursor_visual() -> None:
        cursor_dot.pos = (float(cursor_pos[0]), float(cursor_pos[1]))
        pointer.start = (float(cursor_pos[0]), float(cursor_pos[1]))
        pointer.end = (
            float(cursor_pos[0] + math.cos(heading_rad) * arrow_len),
            float(cursor_pos[1] + math.sin(heading_rad) * arrow_len),
        )
        cursor_dot.fillColor = paused_color if is_paused else running_color

    def _draw_frame() -> None:
        arena_outline.draw()
        pointer.draw()
        cursor_dot.draw()
        title.draw()
        cue.draw()
        info.draw()
        status.draw()
        win.flip()

    def _toggle_pause(reason: str) -> None:
        nonlocal is_paused, jaw_last_toggle_t
        is_paused = not is_paused
        jaw_last_toggle_t = core.getTime()
        logger.info("Pause toggled -> %s (reason=%s)", is_paused, reason)

    def _poll_live() -> None:
        nonlocal last_live_ts, live_buffer, jaw_buffer, prediction_count
        nonlocal left_prob, right_prob, raw_command, ema_command, latest_pred_code, live_note
        nonlocal rotation_dir

        if stream is None or live_filter is None:
            return

        data, ts = stream.get_data(winsize=stream_pull_s, picks="all")
        if data.size > 0 and ts is not None and len(ts) > 0:
            ts_arr = np.asarray(ts)
            mask = np.ones_like(ts_arr, dtype=bool) if last_live_ts is None else (ts_arr > float(last_live_ts))
            if np.any(mask):
                x_new = np.asarray(data[:, mask], dtype=np.float32)
                last_live_ts = float(ts_arr[mask][-1])

                jaw_buffer = np.concatenate((jaw_buffer, x_new), axis=1)
                max_keep = max(keep_n, jaw_window_n)
                if jaw_buffer.shape[1] > max_keep:
                    jaw_buffer = jaw_buffer[:, -max_keep:]

                # Jaw-clench heuristic uses short-window peak-to-peak amplitude.
                if jaw_buffer.shape[1] >= jaw_window_n:
                    jaw_win = jaw_buffer[jaw_idxs, -jaw_window_n:]
                    jaw_ptp = float(np.ptp(jaw_win, axis=-1).max())
                    now_t = core.getTime()
                    if jaw_ptp >= jaw_clench_ptp_thresh and (now_t - jaw_last_toggle_t) >= jaw_refractory_s:
                        _toggle_pause(reason=f"jaw_clench ptp={jaw_ptp:.1f}")

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
            latest_pred_code = None
            live_note = f"warming up ({needed_s:.1f}s)"
            return

        x_win = live_buffer[:, -window_n:]
        max_ptp = float(np.ptp(x_win, axis=-1).max())
        if reject_thresh is not None and max_ptp > float(reject_thresh):
            raw_command = 0.0
            ema_command *= 0.90
            latest_pred_code = None
            live_note = "artifact reject"
            return

        p_vec = classifier.predict_proba(x_win[np.newaxis, ...])[0]
        left_prob = float(p_vec[class_index[int(stim_cfg.left_code)]])
        right_prob = float(p_vec[class_index[int(stim_cfg.right_code)]])
        raw_command = float(np.clip(right_prob - left_prob, -1.0, 1.0))
        alpha = float(np.clip(task_cfg.command_ema_alpha, 0.0, 1.0))
        if prediction_count == 0:
            ema_command = raw_command
        else:
            ema_command = (1.0 - alpha) * ema_command + alpha * raw_command

        if abs(ema_command) < float(task_cfg.command_deadband):
            latest_pred_code = None
        elif ema_command > 0.0:
            latest_pred_code = int(stim_cfg.right_code)
        else:
            latest_pred_code = int(stim_cfg.left_code)

        if not debug_mode and is_paused:
            if latest_pred_code == int(stim_cfg.left_code):
                rotation_dir = -1.0
            elif latest_pred_code == int(stim_cfg.right_code):
                rotation_dir = 1.0

        prediction_count += 1
        live_note = "tracking"

        if prediction_count % 20 == 0:
            logger.info(
                "Decode %d: left_p=%.4f, right_p=%.4f, raw=%.4f, ema=%.4f, pred=%s, paused=%s",
                prediction_count,
                left_prob,
                right_prob,
                raw_command,
                ema_command,
                latest_pred_code,
                is_paused,
            )

    cue.text = "Jaw clench toggles pause/play. Paused: rotate heading. Unpaused: move in set heading. ESC to quit."
    if debug_mode:
        cue.text += " Debug mode: SPACE toggles pause/play, LEFT/RIGHT set rotation."
    _update_cursor_visual()

    try:
        while True:
            keys = event.getKeys()
            if "escape" in keys:
                raise KeyboardInterrupt

            if debug_mode and "space" in keys:
                _toggle_pause(reason="debug_space")

            if debug_mode and is_paused:
                if "left" in keys:
                    manual_rotation_dir = -1.0
                if "right" in keys:
                    manual_rotation_dir = 1.0

            _poll_live()

            now_t = core.getTime()
            dt = float(np.clip(now_t - last_frame_t, 1e-4, 0.05))
            last_frame_t = now_t

            active_dir = manual_rotation_dir if debug_mode else rotation_dir
            # Rotate heading only while paused.
            if is_paused:
                heading_rad = _wrap_angle(heading_rad + active_dir * rotation_speed_rad_s * dt)
            else:
                # Move the cursor in the heading selected while paused.
                cursor_pos[0] += math.cos(heading_rad) * move_speed_norm_s * dt
                cursor_pos[1] += math.sin(heading_rad) * move_speed_norm_s * dt
                # Wrap around screen edges to preserve travel direction.
                if cursor_pos[0] > arena_limit_x:
                    cursor_pos[0] = -arena_limit_x
                elif cursor_pos[0] < -arena_limit_x:
                    cursor_pos[0] = arena_limit_x
                if cursor_pos[1] > arena_limit_y:
                    cursor_pos[1] = -arena_limit_y
                elif cursor_pos[1] < -arena_limit_y:
                    cursor_pos[1] = arena_limit_y

            _update_cursor_visual()

            mode_text = "debug" if debug_mode else "live"
            gate_text = "ON (rotation allowed)" if is_paused else "OFF (rotation frozen)"
            info.text = (
                f"mode={mode_text}   pause_gate={gate_text}   "
                f"{label_cfg.left_name}={left_prob:.2f}   {label_cfg.right_name}={right_prob:.2f}   "
                f"raw={raw_command:+.2f}   ema={ema_command:+.2f}"
            )
            status.text = (
                f"pred={latest_pred_code}   dir={active_dir:+.0f}   updates={prediction_count}   "
                f"jaw_thresh={jaw_clench_ptp_thresh:.0f}uV   {live_note}"
            )
            _draw_frame()

    except KeyboardInterrupt:
        logger.info("Session interrupted by user.")
    finally:
        if stream is not None:
            try:
                stream.disconnect()
            except Exception:
                pass
        try:
            win.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jaw-clench pause cursor rotation task")
    parser.add_argument("--debug", action="store_true", help="Enable keyboard-only debug controls")
    args = parser.parse_args()

    fname = _prompt_session_prefix()
    print(f"[SESSION] Using filename prefix: {fname}")
    run_task(fname=fname, debug_mode=bool(args.debug))
