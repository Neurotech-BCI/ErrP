from __future__ import annotations

import argparse
import logging
import pickle
from collections import Counter
from datetime import datetime

import numpy as np
from mne_lsl.stream import StreamLSL
from psychopy import core, event, visual

from config import (
    EEGConfig,
    LSLConfig,
    MentalCommandLabelConfig,
    MentalCommandModelConfig,
    MICursorTaskConfig,
    StimConfig,
)
from jaw_clench_detector import JawClenchDetector
from mental_command_worker import (
    StreamingIIRFilter,
    canonicalize_channel_name,
    resolve_channel_order,
    train_or_load_shared_mi_model,
)


def _make_task_logger(fname: str) -> logging.Logger:
    logger = logging.getLogger(f"mi_keyboard.{fname}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(f"{fname}_mi_keyboard.log", mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def _sanitize(raw: str) -> str:
    cleaned = "_".join(raw.strip().lower().split())
    cleaned = "".join(c for c in cleaned if c.isalnum() or c == "_")
    return cleaned.strip("_")


def _prompt_prefix() -> str:
    while True:
        raw = input("Enter participant name: ")
        p = _sanitize(raw)
        if p:
            return f"{datetime.now().strftime('%m_%d_%y')}_{p}_mi_keyboard"
        print("Name cannot be empty.")


def _build_virtual_keyboard(win: visual.Window) -> list[dict]:
    rows = [
        ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
        ["Z", "X", "C", "V", "B", "N", "M"],
        ["SPACE", "BKSP", "CLR"],
    ]

    width_scale = {
        "SPACE": 3.6,
        "BKSP": 2.0,
        "CLR": 1.7,
    }

    base_w = 0.125
    key_h = 0.125
    gap = 0.02
    y_top = 0.20
    row_gap = 0.18

    keys: list[dict] = []
    for row_idx, row in enumerate(rows):
        widths = [base_w * width_scale.get(label, 1.0) for label in row]
        total_w = float(np.sum(widths) + max(0, len(widths) - 1) * gap)
        x_left = -total_w / 2.0
        y = y_top - row_idx * row_gap

        for label, width in zip(row, widths):
            x = x_left + width / 2.0
            rect = visual.Rect(
                win,
                width=width,
                height=key_h,
                pos=(x, y),
                fillColor=(-0.35, -0.35, -0.35),
                lineColor=(0.35, 0.35, 0.35),
                lineWidth=1.5,
            )
            text = visual.TextStim(
                win,
                text=label,
                pos=(x, y),
                height=0.045,
                color=(0.93, 0.93, 0.93),
            )
            keys.append({"label": label, "rect": rect, "text": text})
            x_left += width + gap

    return keys


def _apply_key_to_buffer(current_text: str, key_label: str, max_chars: int) -> str:
    if key_label == "SPACE":
        next_text = current_text + " "
    elif key_label == "BKSP":
        next_text = current_text[:-1]
    elif key_label == "CLR":
        next_text = ""
    else:
        next_text = current_text + key_label

    if len(next_text) > int(max_chars):
        next_text = next_text[-int(max_chars):]
    return next_text


def run_task(
    fname: str,
    move_confidence_thresh: float,
    cursor_step_s: float,
    jaw_select_refractory_s: float,
    max_text_chars: int,
) -> None:
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

    logger.info(
        "Starting MI virtual keyboard | move_conf=%.2f cursor_step_s=%.3f jaw_select_refractory_s=%.2f",
        float(move_confidence_thresh),
        float(cursor_step_s),
        float(jaw_select_refractory_s),
    )

    stream = StreamLSL(
        bufsize=60.0,
        name=lsl_cfg.name,
        stype=lsl_cfg.stype,
        source_id=lsl_cfg.source_id,
    )
    stream.connect(acquisition_delay=0.001, processing_flags="all")
    logger.info("Connected to LSL stream: %s", stream.info)

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
        "Using channels: sfreq=%.3f, selected=%s, missing_configured=%s",
        sfreq,
        stream_ch_names,
        missing,
    )

    logger.info(
        "Preparing offline model: data_dir=%s, edf_glob=%s, window_s=%.3f, step_s=%.3f",
        task_cfg.data_dir,
        task_cfg.edf_glob,
        task_cfg.window_s,
        task_cfg.window_step_s,
    )
    shared_model = train_or_load_shared_mi_model(
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

    clf_classes = np.asarray(classifier.named_steps["clf"].classes_, dtype=int)
    if int(stim_cfg.left_code) not in class_index or int(stim_cfg.right_code) not in class_index:
        raise RuntimeError(
            f"Classifier classes {clf_classes.tolist()} do not contain expected left/right codes "
            f"{[int(stim_cfg.left_code), int(stim_cfg.right_code)]}."
        )

    counts = Counter(shared_model.class_counts)
    if shared_model.dataset is not None:
        dataset = shared_model.dataset
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
        np.save(f"{fname}_mi_keyboard_windows.npy", dataset.X)
        np.save(f"{fname}_mi_keyboard_labels.npy", dataset.y)
    else:
        logger.info(
            "Using shared cached MI model: class_counts=%s, loso_mean=%.4f, loso_std=%.4f, cache=%s",
            counts,
            loso.mean_accuracy,
            loso.std_accuracy,
            shared_model.cache_path,
        )

    with open(f"{fname}_mi_keyboard_model.pkl", "wb") as fh:
        pickle.dump(classifier, fh)

    jaw_ch_idx = len(model_ch_names) - 1
    jaw_detector = JawClenchDetector(fs=sfreq)
    jaw_detector.refractory = max(0.60, float(jaw_select_refractory_s))
    jaw_thresh_min = 8.0
    jaw_calibration_duration_s = 5.0

    win = visual.Window(
        size=task_cfg.win_size,
        color=(0.08, 0.08, 0.08),
        units="norm",
        fullscr=task_cfg.fullscreen,
    )

    title = visual.TextStim(win, text="MI Virtual Keyboard", pos=(0, 0.90), height=0.06, color=(0.92, 0.92, 0.92))
    cue = visual.TextStim(win, text="", pos=(0, 0.80), height=0.048, color=(0.92, 0.92, 0.92))
    info = visual.TextStim(win, text="", pos=(0, -0.84), height=0.040, color=(0.72, 0.86, 0.96))
    status = visual.TextStim(win, text="", pos=(0, -0.92), height=0.038, color=(0.78, 0.78, 0.78))

    text_box = visual.Rect(
        win,
        width=1.82,
        height=0.23,
        pos=(0, 0.60),
        fillColor=(-0.20, -0.20, -0.20),
        lineColor=(0.30, 0.30, 0.30),
        lineWidth=1.6,
    )
    typed_text_stim = visual.TextStim(
        win,
        text="",
        pos=(-0.87, 0.60),
        height=0.055,
        color=(0.95, 0.95, 0.95),
        alignText="left",
        anchorHoriz="left",
        wrapWidth=1.72,
    )

    keys = _build_virtual_keyboard(win)
    cursor_index = 0
    typed_text = ""

    def _draw_frame() -> None:
        text_box.draw()
        typed_text_stim.text = typed_text if typed_text else "(jaw clench to select highlighted key)"
        typed_text_stim.draw()

        for i, key in enumerate(keys):
            active = i == cursor_index
            key["rect"].fillColor = (0.20, 0.62, 0.28) if active else (-0.35, -0.35, -0.35)
            key["rect"].lineColor = (0.95, 0.95, 0.95) if active else (0.35, 0.35, 0.35)
            key["rect"].lineWidth = 2.6 if active else 1.5
            key["rect"].draw()
            key["text"].draw()

        title.draw()
        cue.draw()
        info.draw()
        status.draw()
        win.flip()

    def _wait_for_space(prompt_text: str) -> None:
        cue.text = prompt_text
        while True:
            pressed = event.getKeys()
            if "escape" in pressed:
                raise KeyboardInterrupt
            if "space" in pressed:
                return
            _draw_frame()

    cue.text = "Jaw calibration"
    info.text = "Keep your jaw relaxed to measure baseline activity."
    status.text = "Press SPACE to start calibration. ESC to quit."
    _wait_for_space("Jaw calibration")

    calib_raw: list[np.ndarray] = []
    last_cal_ts: float | None = None
    cal_clock = core.Clock()
    while cal_clock.getTime() < jaw_calibration_duration_s:
        if "escape" in event.getKeys():
            raise KeyboardInterrupt
        data, ts = stream.get_data(winsize=0.25, picks="all")
        if data.size > 0 and ts is not None and len(ts) > 0:
            ts_arr = np.asarray(ts)
            mask = np.ones_like(ts_arr, bool) if last_cal_ts is None else (ts_arr > float(last_cal_ts))
            if np.any(mask):
                new_data = np.asarray(data[jaw_ch_idx, mask], dtype=np.float32)
                last_cal_ts = float(ts_arr[mask][-1])
                calib_raw.append(new_data)
        remaining = max(0.0, jaw_calibration_duration_s - cal_clock.getTime())
        cue.text = "Jaw calibration"
        info.text = f"Keep jaw relaxed for {remaining:0.1f}s"
        status.text = ""
        _draw_frame()

    if calib_raw:
        try:
            jaw_signal = np.concatenate(calib_raw).astype(np.float32)
            floor = jaw_detector.calibrate(jaw_signal)
            logger.info("Jaw detector calibrated: floor=%.3f", float(floor))
        except Exception:
            logger.warning("Jaw detector calibration method failed, continuing with adaptive threshold.")

    cue.text = "Ready"
    info.text = "Imagine LEFT/RIGHT to move cursor. Jaw clench to select highlighted key."
    status.text = "Press SPACE to start live keyboard control. ESC to quit."
    _wait_for_space("Ready")

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

    jaw_buffer_raw = np.empty(0, dtype=np.float32)
    jaw_buffer_ts = np.empty(0, dtype=np.float64)

    pred_clock = core.Clock()
    prediction_count = 0
    left_prob = 0.5
    right_prob = 0.5
    raw_command = 0.0
    ema_command = 0.0
    live_note = "warming up"
    latest_pred_code: int | None = None
    bias_offset = float(task_cfg.live_bias_offset) if bool(task_cfg.enable_live_bias_offset) else 0.0

    last_move_t = -1e9
    last_select_t = -1e9

    def _pull_stream_and_update_buffers() -> None:
        nonlocal last_live_ts, live_buffer, jaw_buffer_raw, jaw_buffer_ts

        data, ts = stream.get_data(winsize=stream_pull_s, picks="all")
        if data.size == 0 or ts is None or len(ts) == 0:
            return

        ts_arr = np.asarray(ts)
        mask = np.ones_like(ts_arr, bool) if last_live_ts is None else (ts_arr > float(last_live_ts))
        if not np.any(mask):
            return

        x_new = np.asarray(data[:, mask], dtype=np.float32)
        t_new = ts_arr[mask].astype(np.float64)
        last_live_ts = float(t_new[-1])

        x_new_filt = live_filter.process(x_new)
        live_buffer = np.concatenate((live_buffer, x_new_filt), axis=1)
        if live_buffer.shape[1] > keep_n:
            live_buffer = live_buffer[:, -keep_n:]

        jaw_raw_new = np.asarray(x_new[jaw_ch_idx], dtype=np.float32)
        jaw_buffer_raw = np.concatenate((jaw_buffer_raw, jaw_raw_new))
        jaw_buffer_ts = np.concatenate((jaw_buffer_ts, t_new))

        keep = int(max(sfreq, window_n))
        if jaw_buffer_raw.size > keep:
            jaw_buffer_raw = jaw_buffer_raw[-keep:]
            jaw_buffer_ts = jaw_buffer_ts[-keep:]

    def _poll_jaw_clench() -> bool:
        if jaw_buffer_raw.size < 8 or jaw_buffer_ts.size < 8:
            return False
        _, clenches, _ = jaw_detector.detect(jaw_buffer_raw, jaw_buffer_ts, jaw_thresh_min)
        return len(clenches) > 0

    def _poll_decoder() -> None:
        nonlocal prediction_count, left_prob, right_prob, raw_command, ema_command, live_note, latest_pred_code

        if pred_clock.getTime() < task_cfg.live_update_interval_s:
            return
        pred_clock.reset()

        if live_buffer.shape[1] < window_n:
            needed_s = max(0.0, (window_n - live_buffer.shape[1]) / sfreq)
            live_note = f"warming up ({needed_s:.1f}s)"
            latest_pred_code = None
            return

        x_win = live_buffer[:, -window_n:]
        max_ptp = float(np.ptp(x_win, axis=-1).max())
        if reject_thresh is not None and max_ptp > float(reject_thresh):
            live_note = "artifact reject"
            raw_command = 0.0
            ema_command *= 0.85
            latest_pred_code = None
            return

        p_vec = classifier.predict_proba(x_win[np.newaxis, ...])[0]
        left_prob = float(p_vec[class_index[int(stim_cfg.left_code)]])
        right_prob = float(p_vec[class_index[int(stim_cfg.right_code)]])
        raw_command = float(np.clip(right_prob - left_prob + bias_offset, -1.0, 1.0))

        alpha = float(np.clip(task_cfg.command_ema_alpha, 0.0, 1.0))
        ema_command = raw_command if prediction_count == 0 else (1.0 - alpha) * ema_command + alpha * raw_command

        if abs(ema_command) < float(task_cfg.command_deadband):
            latest_pred_code = None
        elif ema_command > 0.0:
            latest_pred_code = int(stim_cfg.right_code)
        else:
            latest_pred_code = int(stim_cfg.left_code)

        prediction_count += 1
        live_note = "tracking"

    try:
        while True:
            pressed = event.getKeys()
            if "escape" in pressed:
                raise KeyboardInterrupt

            _pull_stream_and_update_buffers()
            _poll_decoder()

            now_t = core.getTime()
            conf = max(left_prob, right_prob)

            move_text = ""
            if latest_pred_code is not None and conf >= float(move_confidence_thresh):
                if (now_t - last_move_t) >= float(cursor_step_s):
                    if latest_pred_code == int(stim_cfg.left_code):
                        cursor_index = (cursor_index - 1) % len(keys)
                        move_text = "cursor LEFT"
                    elif latest_pred_code == int(stim_cfg.right_code):
                        cursor_index = (cursor_index + 1) % len(keys)
                        move_text = "cursor RIGHT"
                    last_move_t = now_t
                    if move_text:
                        logger.info(
                            "%s | key=%s | left=%.3f right=%.3f conf=%.3f",
                            move_text,
                            keys[cursor_index]["label"],
                            left_prob,
                            right_prob,
                            conf,
                        )

            selected_text = ""
            jaw_event = _poll_jaw_clench()
            if jaw_event and (now_t - last_select_t) >= float(jaw_select_refractory_s):
                selected_label = str(keys[cursor_index]["label"])
                typed_text = _apply_key_to_buffer(
                    current_text=typed_text,
                    key_label=selected_label,
                    max_chars=max_text_chars,
                )
                selected_text = f"selected={selected_label}"
                last_select_t = now_t
                logger.info(
                    "Jaw select: key=%s | typed_len=%d",
                    selected_label,
                    len(typed_text),
                )

            cue.text = "LEFT/RIGHT MI moves cursor when confidence threshold is met. Jaw clench selects key."
            info.text = (
                f"{label_cfg.left_name}={left_prob:.2f}   {label_cfg.right_name}={right_prob:.2f}   "
                f"conf={conf:.2f}   raw={raw_command:+.2f}   ema={ema_command:+.2f}"
            )
            status.text = (
                f"key={keys[cursor_index]['label']}   pred={latest_pred_code}   updates={prediction_count}   "
                f"{live_note}   {move_text} {selected_text}"
            )
            _draw_frame()

    except KeyboardInterrupt:
        logger.info("Session interrupted by user.")
    finally:
        try:
            stream.disconnect()
        except Exception:
            pass
        try:
            win.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Left/right MI + jaw clench virtual keyboard")
    parser.add_argument(
        "--move-confidence-thresh",
        type=float,
        default=0.58,
        help="Minimum max(left_prob,right_prob) before cursor moves",
    )
    parser.add_argument(
        "--cursor-step-s",
        type=float,
        default=0.22,
        help="Minimum time between cursor moves when MI confidence is above threshold",
    )
    parser.add_argument(
        "--jaw-select-refractory-s",
        type=float,
        default=0.35,
        help="Minimum time between jaw-based key selections",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=320,
        help="Maximum number of typed characters retained on screen",
    )
    args = parser.parse_args()

    prefix = _prompt_prefix()
    print(f"[SESSION] prefix: {prefix}")
    run_task(
        fname=prefix,
        move_confidence_thresh=float(args.move_confidence_thresh),
        cursor_step_s=float(args.cursor_step_s),
        jaw_select_refractory_s=float(args.jaw_select_refractory_s),
        max_text_chars=int(args.max_text_chars),
    )
