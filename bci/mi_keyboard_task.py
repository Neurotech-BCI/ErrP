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
from bci_runtime import (
    apply_runtime_config_overrides,
    resolve_runtime_face_classifier,
    resolve_shared_mi_model,
)
from derick_ml_jawclench import (
    collect_cue_locked_stream_block,
    JAW_CLENCH_CLASS_CODE,
    RAPID_BLINK_CLASS_CODE,
    REST_CLASS_CODE,
    run_visual_face_event_calibration,
    select_jaw_channel_indices,
    update_live_face_event_state,
)
from mental_command_worker import (
    StreamingIIRFilter,
    canonicalize_channel_name,
    resolve_channel_order,
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


def _build_virtual_keyboard(win: visual.Window) -> tuple[list[dict], list[list[int]]]:
    rows = [
        ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
        ["Z", "X", "C", "V", "B", "N", "M"],
        ["SPACE", "BKSP", "ENTER"],
    ]

    width_scale = {
        "SPACE": 3.6,
        "BKSP": 2.0,
        "ENTER": 1.9,
    }

    base_w = 0.125
    key_h = 0.125
    gap = 0.02
    y_top = 0.20
    row_gap = 0.18

    keys: list[dict] = []
    row_indices: list[list[int]] = []
    for row_idx, row in enumerate(rows):
        widths = [base_w * width_scale.get(label, 1.0) for label in row]
        total_w = float(np.sum(widths) + max(0, len(widths) - 1) * gap)
        x_left = -total_w / 2.0
        y = y_top - row_idx * row_gap
        current_row: list[int] = []

        for col_idx, (label, width) in enumerate(zip(row, widths)):
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
            keys.append({
                "label": label,
                "rect": rect,
                "text": text,
                "row_idx": row_idx,
                "col_idx": col_idx,
            })
            current_row.append(len(keys) - 1)
            x_left += width + gap
        row_indices.append(current_row)

    return keys, row_indices


def _apply_key_to_buffer(current_text: str, key_label: str, max_chars: int) -> str:
    if key_label == "SPACE":
        next_text = current_text + " "
    elif key_label == "BKSP":
        next_text = current_text[:-1]
    elif key_label == "ENTER":
        next_text = current_text
    else:
        next_text = current_text + key_label

    if len(next_text) > int(max_chars):
        next_text = next_text[-int(max_chars):]
    return next_text


def run_task(
    fname: str,
    move_confidence_thresh: float = 0.60,
    cursor_step_s: float = 0.70,
    jaw_select_refractory_s: float = 0.35,
    max_text_chars: int = 320,
    raise_on_escape: bool = False,
) -> str | None:
    logger = _make_task_logger(fname)

    lsl_cfg = LSLConfig()
    stim_cfg = StimConfig()
    label_cfg = MentalCommandLabelConfig()
    task_cfg = MICursorTaskConfig()
    model_cfg = MentalCommandModelConfig()
    eeg_cfg = EEGConfig(
        l_freq=8.0,
        h_freq=30.0,
        reject_peak_to_peak=150.0,
    )
    cfgs = apply_runtime_config_overrides(
        "mi_keyboard_task",
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

    logger.info(
        "Starting MI virtual keyboard | move_conf=%.2f cursor_step_s=%.3f jaw_select_refractory_s=%.2f blink_row_switch=enabled",
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

    jaw_idxs = select_jaw_channel_indices(model_ch_names)
    jaw_window_n = int(round(float(task_cfg.jaw_window_s) * sfreq))
    face_classifier = None
    face_class_index: dict[int, int] | None = None
    rest_prob = 0.0
    jaw_prob = 0.0
    blink_prob = 0.0
    face_pred_code = REST_CLASS_CODE
    jaw_prev_pred = 0
    blink_prev_pred = 0
    jaw_event_pending = False
    blink_event_pending = False
    jaw_last_select_t = -1e9
    blink_last_row_t = -1e9
    jaw_prob_thresh = float(task_cfg.jaw_clench_prob_thresh)
    blink_prob_thresh = float(getattr(task_cfg, "blink_prob_thresh", 0.70))
    jaw_refractory_s = max(float(task_cfg.jaw_clench_refractory_s), float(jaw_select_refractory_s))
    blink_refractory_s = float(getattr(task_cfg, "blink_refractory_s", 0.70))

    logger.info(
        "Face-event model config: channels=%s, jaw_window_s=%.2f, jaw_step_s=%.2f, trim_s=%.2f, jaw_threshold=%.2f, blink_threshold=%.2f, jaw_refractory_s=%.2f, blink_refractory_s=%.2f",
        [model_ch_names[idx] for idx in jaw_idxs],
        float(task_cfg.jaw_window_s),
        float(task_cfg.jaw_window_step_s),
        float(task_cfg.jaw_calibration_trim_s),
        jaw_prob_thresh,
        blink_prob_thresh,
        jaw_refractory_s,
        blink_refractory_s,
    )

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

    keys, key_rows = _build_virtual_keyboard(win)
    cursor_row = 0
    cursor_col = 0
    cursor_index = key_rows[cursor_row][cursor_col]
    typed_text = ""

    def _sync_cursor() -> None:
        nonlocal cursor_index, cursor_col
        row = key_rows[cursor_row]
        cursor_col = cursor_col % len(row)
        cursor_index = row[cursor_col]

    def _move_within_row(delta: int) -> None:
        nonlocal cursor_col
        row = key_rows[cursor_row]
        cursor_col = (cursor_col + int(delta)) % len(row)
        _sync_cursor()

    def _move_down_row() -> None:
        nonlocal cursor_row, cursor_col
        cursor_row = (cursor_row + 1) % len(key_rows)
        cursor_col = min(cursor_col, len(key_rows[cursor_row]) - 1)
        _sync_cursor()

    def _draw_frame() -> None:
        text_box.draw()
        typed_text_stim.text = typed_text if typed_text else "(jaw clench selects highlighted key)"
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

    def _wait_for_seconds(duration_s: float) -> None:
        clock = core.Clock()
        while clock.getTime() < duration_s:
            if "escape" in event.getKeys():
                raise KeyboardInterrupt
            _draw_frame()

    def _collect_stream_block(duration_s: float) -> np.ndarray:
        def _check_abort() -> None:
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

        return collect_cue_locked_stream_block(
            stream=stream,
            sfreq=float(sfreq),
            n_channels=len(model_ch_names),
            duration_s=float(duration_s),
            cue_offset_s=float(task_cfg.special_command_cue_offset_s),
            render_frame=lambda _elapsed_s, _total_s: _draw_frame(),
            check_abort=_check_abort,
            logger=logger,
            label="mi_keyboard face-event calibration block",
        )

    runtime_face_classifier, runtime_train_acc, runtime_counts = resolve_runtime_face_classifier(
        logger=logger,
        min_total_samples=18,
        requested_channel_names=model_ch_names,
    )
    if runtime_face_classifier is not None:
        face_classifier = runtime_face_classifier
        face_train_acc = float(runtime_train_acc or 0.0)
        face_counts = runtime_counts or {}
    else:
        face_classifier, face_train_acc, _face_y, face_counts = run_visual_face_event_calibration(
            cue=cue,
            info=info,
            status=status,
            wait_for_space=_wait_for_space,
            wait_for_seconds=_wait_for_seconds,
            collect_stream_block=_collect_stream_block,
            jaw_idxs=jaw_idxs,
            jaw_window_n=jaw_window_n,
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
            min_total_samples=18,
            cue_offset_s=float(task_cfg.special_command_cue_offset_s),
            ready_info_text=None,
            ready_status_text="Jaw clench selects keys. Rapid eye blinks move down one row. Press SPACE to start.",
        )
    with open(f"{fname}_mi_keyboard_face_event_model.pkl", "wb") as fh:
        pickle.dump(face_classifier, fh)

    face_classes = np.asarray(getattr(face_classifier, "classes_", []), dtype=int)
    if face_classes.size == 0 and hasattr(face_classifier, "named_steps") and "clf" in face_classifier.named_steps:
        face_classes = np.asarray(face_classifier.named_steps["clf"].classes_, dtype=int)
    if face_classes.size == 0:
        raise RuntimeError("Face-event classifier has no classes_ after calibration.")
    face_class_index = {int(c): i for i, c in enumerate(face_classes)}
    for needed in (REST_CLASS_CODE, JAW_CLENCH_CLASS_CODE, RAPID_BLINK_CLASS_CODE):
        if int(needed) not in face_class_index:
            raise RuntimeError(
                "Face-event classifier is missing required classes. "
                f"Found {face_classes.tolist()}, required {[REST_CLASS_CODE, JAW_CLENCH_CLASS_CODE, RAPID_BLINK_CLASS_CODE]}"
            )

    logger.info(
        "Face-event classifier ready: train_acc=%.3f, counts=%s",
        float(face_train_acc),
        face_counts,
    )

    cue.text = "Ready"
    info.text = "LEFT/RIGHT imagery moves within the active row. Rapid eye blinks move down one row. Jaw clench selects the highlighted key."
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

    face_buffer = np.empty((len(model_ch_names), 0), dtype=np.float32)

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
    def _pull_stream_and_update_buffers() -> None:
        nonlocal last_live_ts, live_buffer, face_buffer
        nonlocal rest_prob, jaw_prob, blink_prob, face_pred_code
        nonlocal jaw_prev_pred, blink_prev_pred, jaw_event_pending, blink_event_pending
        nonlocal jaw_last_select_t, blink_last_row_t

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

        face_buffer, rest_prob, jaw_prob, blink_prob, face_pred_code, jaw_prev_pred, blink_prev_pred, should_select, should_row_switch = update_live_face_event_state(
            face_buffer=face_buffer,
            x_new=x_new,
            keep_n=keep_n,
            jaw_window_n=jaw_window_n,
            face_classifier=face_classifier,
            class_index=face_class_index,
            jaw_idxs=jaw_idxs,
            rest_prob=rest_prob,
            jaw_prob=jaw_prob,
            blink_prob=blink_prob,
            jaw_prev_pred=jaw_prev_pred,
            blink_prev_pred=blink_prev_pred,
            jaw_prob_thresh=jaw_prob_thresh,
            blink_prob_thresh=blink_prob_thresh,
            jaw_last_event_t=jaw_last_select_t,
            blink_last_event_t=blink_last_row_t,
            jaw_refractory_s=jaw_refractory_s,
            blink_refractory_s=blink_refractory_s,
            now_t=core.getTime(),
        )
        if should_select:
            jaw_last_select_t = core.getTime()
            jaw_event_pending = True
        if should_row_switch:
            blink_last_row_t = core.getTime()
            blink_event_pending = True

    def _poll_jaw_clench() -> bool:
        nonlocal jaw_event_pending
        if jaw_event_pending:
            jaw_event_pending = False
            return True
        return False

    def _poll_blink() -> bool:
        nonlocal blink_event_pending
        if blink_event_pending:
            blink_event_pending = False
            return True
        return False

    def _poll_decoder() -> None:
        nonlocal prediction_count, left_prob, right_prob, raw_command, ema_command, live_note, latest_pred_code

        if pred_clock.getTime() < task_cfg.live_update_interval_s:
            return False
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

    submitted_text: str | None = None
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
            if _poll_blink():
                _move_down_row()
                move_text = "row DOWN"
                logger.info(
                    "%s | row=%d col=%d key=%s | blink_p=%.3f",
                    move_text,
                    cursor_row,
                    cursor_col,
                    keys[cursor_index]["label"],
                    blink_prob,
                )
            elif latest_pred_code is not None and conf >= float(move_confidence_thresh):
                if (now_t - last_move_t) >= float(cursor_step_s):
                    if latest_pred_code == int(stim_cfg.left_code):
                        _move_within_row(-1)
                        move_text = "cursor LEFT"
                    elif latest_pred_code == int(stim_cfg.right_code):
                        _move_within_row(1)
                        move_text = "cursor RIGHT"
                    last_move_t = now_t
                    if move_text:
                        logger.info(
                            "%s | row=%d col=%d key=%s | left=%.3f right=%.3f conf=%.3f",
                            move_text,
                            cursor_row,
                            cursor_col,
                            keys[cursor_index]["label"],
                            left_prob,
                            right_prob,
                            conf,
                        )

            selected_text = ""
            jaw_event = _poll_jaw_clench()
            if jaw_event:
                selected_label = str(keys[cursor_index]["label"])
                if selected_label == "ENTER":
                    submitted_text = typed_text
                    logger.info(
                        "Jaw select: key=%s | typed_len=%d | jaw_p=%.3f | submitted_text=%r",
                        selected_label,
                        len(typed_text),
                        jaw_prob,
                        submitted_text,
                    )
                    break
                typed_text = _apply_key_to_buffer(
                    current_text=typed_text,
                    key_label=selected_label,
                    max_chars=max_text_chars,
                )
                selected_text = f"selected={selected_label}"
                logger.info(
                    "Jaw select: key=%s | typed_len=%d | jaw_p=%.3f",
                    selected_label,
                    len(typed_text),
                    jaw_prob,
                )

            cue.text = "LEFT/RIGHT MI cycles within row. Rapid eye blink moves down one row. Jaw clench selects key."
            info.text = (
                f"{label_cfg.left_name}={left_prob:.2f}   {label_cfg.right_name}={right_prob:.2f}   "
                f"rest={rest_prob:.2f}   jaw={jaw_prob:.2f}   blink={blink_prob:.2f}"
            )
            status.text = (
                f"{live_note}   {move_text} {selected_text}".strip()
            )
            _draw_frame()

    except KeyboardInterrupt:
        logger.info("Session interrupted by user.")
        if raise_on_escape:
            raise
    finally:
        if submitted_text is not None:
            logger.info("Keyboard submission complete. Ending task with text=%r", submitted_text)
        try:
            stream.disconnect()
        except Exception:
            pass
        try:
            win.close()
        except Exception:
            pass
    return submitted_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Left/right MI virtual keyboard with jaw select and blink row switching"
    )
    parser.add_argument(
        "--move-confidence-thresh",
        type=float,
        default=0.58,
        help="Minimum max(left_prob,right_prob) before cursor moves",
    )
    parser.add_argument(
        "--cursor-step-s",
        type=float,
        default=0.70,
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
