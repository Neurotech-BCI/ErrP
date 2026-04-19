from __future__ import annotations

import logging
import pickle
from collections import Counter
from datetime import datetime

import numpy as np
from psychopy import core, event, visual

from mne_lsl.stream import StreamLSL

from config import EEGConfig, LSLConfig, MentalCommandLabelConfig, MentalCommandModelConfig, MentalCommandTaskConfig, StimConfig
from mental_command_worker import (
    StreamingIIRFilter,
    append_windows_to_dataset,
    canonicalize_channel_name,
    evaluate_loso_sessions,
    load_offline_mi_dataset,
    make_mi_classifier,
    prepare_continuous_windows,
    resolve_channel_order,
    PostLDA_KalmanSmoother
)


def _make_task_logger(fname: str) -> logging.Logger:
    logger = logging.getLogger(f"mi_visualizer.{fname}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    file_handler = logging.FileHandler(f"{fname}_mi_visualizer.log", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _channel_stat_map(channel_names: list[str], values: np.ndarray, precision: int = 6) -> dict[str, float]:
    return {
        str(ch): round(float(val), precision)
        for ch, val in zip(channel_names, np.asarray(values).tolist())
    }


def _top_channel_deviation_summary(
    channel_names: list[str],
    current_values: np.ndarray,
    ref_means: np.ndarray,
    ref_stds: np.ndarray,
    top_k: int = 3,
) -> list[dict[str, float | str]]:
    current_values = np.asarray(current_values, dtype=np.float64)
    ref_means = np.asarray(ref_means, dtype=np.float64)
    ref_stds = np.asarray(ref_stds, dtype=np.float64)
    z = np.abs((current_values - ref_means) / np.maximum(ref_stds, 1e-6))
    top_idx = np.argsort(z)[-int(top_k):][::-1]
    summary: list[dict[str, float | str]] = []
    for idx in top_idx:
        summary.append({
            "channel": str(channel_names[int(idx)]),
            "value": round(float(current_values[int(idx)]), 6),
            "train_mean": round(float(ref_means[int(idx)]), 6),
            "train_std": round(float(ref_stds[int(idx)]), 6),
            "abs_z": round(float(z[int(idx)]), 6),
        })
    return summary

def run_task(fname: str):
    logger = _make_task_logger(fname)
    lsl_cfg = LSLConfig()
    stim_cfg = StimConfig()
    label_cfg = MentalCommandLabelConfig()
    task_cfg = MentalCommandTaskConfig()
    model_cfg = MentalCommandModelConfig()
    eeg_cfg = EEGConfig(
        picks=("Pz", "F4", "C4", "P4", "P3", "C3", "F3"),
        l_freq=8.0,
        h_freq=30.0,
        reject_peak_to_peak=150.0,
    )

    stream = StreamLSL(
        bufsize=max(30.0, task_cfg.live_duration_s + 10.0),
        name=lsl_cfg.name,
        stype=lsl_cfg.stype,
        source_id=lsl_cfg.source_id,
    )
    stream.connect(acquisition_delay=0.001, processing_flags="all")
    print(f"[LSL] Stream info: {stream.info}")
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
    print(f"[LSL] Connected: sfreq={sfreq:.1f} Hz, channels={stream_ch_names}")
    logger.info(
        "Using live EEG channels: sfreq=%.3f, selected=%s, missing_configured=%s",
        sfreq,
        stream_ch_names,
        missing,
    )
    if missing:
        print(f"[LSL] Missing configured channels from live stream: {missing}")

    win = visual.Window(size=(1280, 760), color=(-0.08, -0.08, -0.08), units="norm", fullscr=False)
    title = visual.TextStim(win, text="Motor Imagery Visualizer", pos=(0, 0.78), height=0.06, color=(0.9, 0.9, 0.9))
    cue = visual.TextStim(win, text="", pos=(0, 0.42), height=0.08, color=(0.9, 0.9, 0.9))
    status = visual.TextStim(win, text="", pos=(0, -0.7), height=0.045, color=(0.85, 0.85, 0.85))
    detected = visual.TextStim(win, text="", pos=(0, 0.26), height=0.055, color=(0.95, 0.95, 0.95))

    bar_w = 1.40
    bar_h = 0.16
    bar_y = -0.02
    bar_outline = visual.Rect(
        win,
        width=bar_w,
        height=bar_h,
        pos=(0, bar_y),
        lineColor=(0.8, 0.8, 0.8),
        fillColor=None,
        lineWidth=2,
    )
    center_line = visual.Line(
        win,
        start=(0, bar_y - bar_h / 2),
        end=(0, bar_y + bar_h / 2),
        lineColor=(0.8, 0.8, 0.8),
    )
    left_fill = visual.Rect(
        win,
        width=0.001,
        height=bar_h - 0.01,
        pos=(0, bar_y),
        fillColor=(-0.3, 0.8, 0.95),
        lineColor=None,
    )
    right_fill = visual.Rect(
        win,
        width=0.001,
        height=bar_h - 0.01,
        pos=(0, bar_y),
        fillColor=(0.95, 0.65, -0.2),
        lineColor=None,
    )
    left_lbl = visual.TextStim(win, text=label_cfg.left_name, pos=(-0.48, -0.2), height=0.05, color=(0.8, 0.9, 1.0))
    right_lbl = visual.TextStim(win, text=label_cfg.right_name, pos=(0.48, -0.2), height=0.05, color=(1.0, 0.9, 0.75))

    def update_bar(score: float):
        score = float(np.clip(score, -1.0, 1.0))
        half_width = bar_w / 2.0
        left_width = half_width * max(-score, 0.0)
        right_width = half_width * max(score, 0.0)
        left_fill.width = max(left_width, 0.001)
        left_fill.pos = (-left_width / 2.0, bar_y)
        right_fill.width = max(right_width, 0.001)
        right_fill.pos = (+right_width / 2.0, bar_y)

    def draw_frame():
        title.draw()
        cue.draw()
        detected.draw()
        bar_outline.draw()
        center_line.draw()
        left_fill.draw()
        right_fill.draw()
        left_lbl.draw()
        right_lbl.draw()
        status.draw()
        win.flip()

    def wait_for_space():
        while True:
            draw_frame()
            keys = event.getKeys()
            if "space" in keys:
                return
            if "escape" in keys:
                raise KeyboardInterrupt

    def collect_stream_block(duration_s: float) -> np.ndarray:
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
                    x_new = np.asarray(data[:, mask], dtype=np.float32)
                    last_ts_local = float(ts_arr[mask][-1])
                    chunks.append(x_new)
            draw_frame()
        if not chunks:
            return np.empty((len(model_ch_names), 0), dtype=np.float32)
        return np.concatenate(chunks, axis=1).astype(np.float32, copy=False)

    classifier = None
    class_index = None
    classifier_classes = None
    feature_extractor = None
    train_feature_mean = None
    train_feature_std = None
    train_feature_centroids = None
    train_ch_std_mean = None
    train_ch_std_std = None
    train_ch_ptp_mean = None
    train_ch_ptp_std = None
    dataset = None
    rest_session_id: int | None = None
    online_cal_session_id: int | None = None
    train_only_session_ids: set[int] = set()
    reject_thresh = eeg_cfg.reject_peak_to_peak
    window_n = int(round(task_cfg.window_s * sfreq))

    try:
        logger.info(
            "Starting offline model preparation: data_dir=%s, edf_glob=%s, window_s=%.3f, step_s=%.3f, "
            "filter_band=[%.1f, %.1f], filter_context_s=%.3f, offline_eeg_scale_to_match_live=%.3f, live_units=%s",
            task_cfg.data_dir,
            task_cfg.edf_glob,
            task_cfg.window_s,
            task_cfg.window_step_s,
            eeg_cfg.l_freq,
            eeg_cfg.h_freq,
            task_cfg.filter_context_s,
            task_cfg.offline_eeg_scale_to_match_live,
            task_cfg.live_eeg_units,
        )
        cue.text = "Preparing model from offline EDF sessions..."
        status.text = (
            f"Loading data from {task_cfg.data_dir}\n"
            "Offline EDFs are standardized to the live stream convention: left-ear referenced, standard channel names,\n"
            "then causally filtered as full sessions before epoching/windowing to match live streaming."
        )
        detected.text = ""
        update_bar(0.0)
        draw_frame()

        dataset = load_offline_mi_dataset(
            data_dir=task_cfg.data_dir,
            edf_glob=task_cfg.edf_glob,
            eeg_cfg=eeg_cfg,
            task_cfg=task_cfg,
            stim_cfg=stim_cfg,
            target_sfreq=sfreq,
            target_channel_names=model_ch_names,
            calibrateOnParticipant=""
        )

        if bool(task_cfg.enable_online_rest_calibration):
            cue.text = "Rest calibration"
            status.text = (
                "Sit in your motor imagery position, keep both hands relaxed,\n"
                "and look at the center while we record a short REST baseline.\n\n"
                "Press SPACE to begin. ESC to quit."
            )
            detected.text = ""
            update_bar(0.0)
            wait_for_space()
            cue.text = "REST baseline"
            status.text = (
                f"Relax both hands and keep your gaze centered for {task_cfg.rest_calibration_duration_s:.0f}s."
            )
            draw_frame()
            rest_block = collect_stream_block(float(task_cfg.rest_calibration_duration_s))
            rest_windows = prepare_continuous_windows(
                raw_block=rest_block,
                eeg_cfg=eeg_cfg,
                sfreq=sfreq,
                window_s=float(task_cfg.window_s),
                step_s=float(task_cfg.window_step_s),
                reject_peak_to_peak=reject_thresh,
            )
            if rest_windows.shape[0] > 0:
                rest_session_id = -1
                train_only_session_ids.add(rest_session_id)
                rest_labels = np.full(rest_windows.shape[0], int(task_cfg.rest_class_code), dtype=int)
                dataset = append_windows_to_dataset(
                    dataset=dataset,
                    windows=rest_windows,
                    labels=rest_labels,
                    session_id=rest_session_id,
                    n_trials_add=1,
                )
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
            status.text = (
                "We will collect a few sustained live LEFT and RIGHT imagery blocks.\n"
                "Prepare when cued, then sustain the imagery for the whole block.\n\n"
                "Press SPACE to begin. ESC to quit."
            )
            detected.text = ""
            update_bar(0.0)
            wait_for_space()

            online_cal_session_id = int(np.max(dataset.session_ids)) + 1
            cal_windows: list[np.ndarray] = []
            cal_labels: list[np.ndarray] = []
            cal_codes = (
                [int(stim_cfg.left_code)] * int(task_cfg.online_lr_calibration_reps_per_class)
                + [int(stim_cfg.right_code)] * int(task_cfg.online_lr_calibration_reps_per_class)
            )
            rng = np.random.default_rng()
            rng.shuffle(cal_codes)

            for cal_idx, code in enumerate(cal_codes, start=1):
                class_name = label_cfg.left_name if int(code) == int(stim_cfg.left_code) else label_cfg.right_name
                cue.text = "Prepare"
                status.text = (
                    f"Live calibration {cal_idx}/{len(cal_codes)}\n"
                    f"Get ready for {class_name} motor imagery."
                )
                detected.text = ""
                draw_frame()
                prep_clock = core.Clock()
                while prep_clock.getTime() < float(task_cfg.online_lr_calibration_prep_s):
                    if "escape" in event.getKeys():
                        raise KeyboardInterrupt
                    draw_frame()

                cue.text = f"SUSTAIN {class_name}"
                status.text = f"Hold {class_name} motor imagery for {task_cfg.online_lr_calibration_hold_s:.0f}s."
                draw_frame()
                cal_block = collect_stream_block(float(task_cfg.online_lr_calibration_hold_s))
                windows = prepare_continuous_windows(
                    raw_block=cal_block,
                    eeg_cfg=eeg_cfg,
                    sfreq=sfreq,
                    window_s=float(task_cfg.window_s),
                    step_s=float(task_cfg.window_step_s),
                    reject_peak_to_peak=reject_thresh,
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
                status.text = "Relax"
                draw_frame()
                iti_clock = core.Clock()
                while iti_clock.getTime() < float(task_cfg.online_lr_calibration_iti_s):
                    if "escape" in event.getKeys():
                        raise KeyboardInterrupt
                    draw_frame()

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
        feature_extractor = classifier[:-1]
        class_index = {int(c): i for i, c in enumerate(classifier_classes)}
        if int(stim_cfg.left_code) not in class_index or int(stim_cfg.right_code) not in class_index:
            raise RuntimeError(
                f"Classifier classes {classifier_classes.tolist()} do not contain the expected "
                f"left/right codes {[int(stim_cfg.left_code), int(stim_cfg.right_code)]}."
            )
        counts = Counter(dataset.y.tolist())
        train_features = np.asarray(feature_extractor.transform(dataset.X), dtype=np.float64)
        train_feature_mean = np.mean(train_features, axis=0)
        train_feature_std = np.std(train_features, axis=0)
        train_feature_centroids = {
            int(code): np.mean(train_features[dataset.y == int(code)], axis=0)
            for code in classifier_classes.tolist()
        }
        train_ch_std = np.std(dataset.X, axis=2)
        train_ch_ptp = np.ptp(dataset.X, axis=2)
        train_ch_std_mean = np.mean(train_ch_std, axis=0)
        train_ch_std_std = np.std(train_ch_std, axis=0)
        train_ch_ptp_mean = np.mean(train_ch_ptp, axis=0)
        train_ch_ptp_std = np.std(train_ch_ptp, axis=0)
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
        logger.info(
            "Classifier class order: classes=%s, class_index=%s, left_code=%d -> prob_index=%d, right_code=%d -> prob_index=%d, rest_code=%d -> prob_index=%s",
            classifier_classes.tolist(),
            class_index,
            int(stim_cfg.left_code),
            class_index[int(stim_cfg.left_code)],
            int(stim_cfg.right_code),
            class_index[int(stim_cfg.right_code)],
            int(task_cfg.rest_class_code),
            class_index.get(int(task_cfg.rest_class_code)),
        )
        train_pred = classifier.predict(dataset.X)
        train_acc = float(np.mean(train_pred == dataset.y))
        train_pred_counts = Counter(train_pred.tolist())
        logger.info(
            "Classifier fit sanity check: training_accuracy=%.4f, predicted_counts=%s",
            train_acc,
            train_pred_counts,
        )
        logger.info(
            "Offline channel diagnostics (%s): std_mean_by_channel=%s, ptp_mean_by_channel=%s",
            dataset.eeg_units,
            _channel_stat_map(model_ch_names, train_ch_std_mean),
            _channel_stat_map(model_ch_names, train_ch_ptp_mean),
        )
        for code in classifier_classes.tolist():
            class_mask = dataset.y == int(code)
            if np.any(class_mask):
                class_ch_std_mean = np.mean(train_ch_std[class_mask], axis=0)
                class_ch_ptp_mean = np.mean(train_ch_ptp[class_mask], axis=0)
                logger.info(
                    "Offline channel diagnostics by class (%s): code=%d, std_mean_by_channel=%s, ptp_mean_by_channel=%s",
                    dataset.eeg_units,
                    int(code),
                    _channel_stat_map(model_ch_names, class_ch_std_mean),
                    _channel_stat_map(model_ch_names, class_ch_ptp_mean),
                )
        for code in classifier_classes.tolist():
            class_mask = dataset.y == int(code)
            if np.any(class_mask):
                class_probs = classifier.predict_proba(dataset.X[class_mask])
                mean_prob = np.mean(class_probs, axis=0)
                prob_map = {
                    int(cls): float(mean_prob[idx])
                    for idx, cls in enumerate(classifier_classes.tolist())
                }
                logger.info(
                    "Classifier fit sanity by true class: true_code=%d, n_windows=%d, mean_predicted_probs=%s",
                    int(code),
                    int(np.sum(class_mask)),
                    prob_map,
                )
                centroid_norm = float(np.linalg.norm(train_feature_centroids[int(code)]))
                logger.info(
                    "Offline feature centroid: code=%d, centroid_l2_norm=%.3f",
                    int(code),
                    centroid_norm,
                )
        for session_id, score in sorted(loso.session_scores.items()):
            session_name = f"session={session_id}"
            if online_cal_session_id is not None and int(session_id) == int(online_cal_session_id):
                session_name = "session=online_lr_calibration"
            logger.info("LOSO session result: %s accuracy=%.4f", session_name, score)
        np.save(f"{fname}_mi_visualizer_windows.npy", dataset.X)
        np.save(f"{fname}_mi_visualizer_labels.npy", dataset.y)
        with open(f"{fname}_mi_visualizer_model.pkl", "wb") as fh:
            pickle.dump(classifier, fh)
        with open(f"{fname}_mi_visualizer_diagnostics.pkl", "wb") as fh:
            pickle.dump(
                {
                    "channel_names": list(model_ch_names),
                    "eeg_units": dataset.eeg_units,
                    "offline_scale_applied": dataset.offline_scale_applied,
                    "classifier_classes": classifier_classes.tolist(),
                    "train_feature_mean": train_feature_mean,
                    "train_feature_std": train_feature_std,
                    "train_feature_centroids": train_feature_centroids,
                    "train_ch_std_mean": train_ch_std_mean,
                    "train_ch_std_std": train_ch_std_std,
                    "train_ch_ptp_mean": train_ch_ptp_mean,
                    "train_ch_ptp_std": train_ch_ptp_std,
                },
                fh,
            )

        session_lines = []
        for session_id, score in sorted(loso.session_scores.items()):
            if online_cal_session_id is not None and int(session_id) == int(online_cal_session_id):
                session_lines.append(f"OnlineCal: {score:.3f}")
            else:
                session_lines.append(f"S{session_id}: {score:.3f}")
        session_summary = "  ".join(session_lines) if session_lines else "No valid held-out sessions"
        rest_count_text = ""
        if int(task_cfg.rest_class_code) in counts:
            rest_count_text = f"  {label_cfg.rest_name}: {counts[int(task_cfg.rest_class_code)]}"

        cue.text = "Model ready"
        status.text = (
            f"Files used: {dataset.n_files_used}/{dataset.n_files_found}  "
            f"Trials: {dataset.n_trials}  Windows: {dataset.n_windows}\n"
            f"{label_cfg.left_name}: {counts[int(stim_cfg.left_code)]}  "
            f"{label_cfg.right_name}: {counts[int(stim_cfg.right_code)]}"
            f"{rest_count_text}\n"
            f"LOSO mean={loso.mean_accuracy:.3f} std={loso.std_accuracy:.3f}\n"
            f"{session_summary}\n"
            "Press SPACE to start live feedback. ESC to quit."
        )
        detected.text = (
            f"Window={task_cfg.window_s:.1f}s  "
            f"Step={task_cfg.window_step_s:.2f}s  "
            f"Filter={eeg_cfg.l_freq:.1f}-{eeg_cfg.h_freq:.1f} Hz"
        )
        update_bar(0.0)
        wait_for_space()

        cue.text = "Live motor imagery"
        status.text = (
            f"Imagine {label_cfg.left_name} or {label_cfg.right_name} hand movement to drive the bar.\n"
            "No smoothing is applied. Press ESC to stop."
        )
        detected.text = ""
        update_bar(0.0)

        pred_clock = core.Clock()
        session_clock = core.Clock()
        p_vec = np.full(len(classifier_classes), 1.0 / max(len(classifier_classes), 1), dtype=np.float64)
        prediction_count = 0
        live_note = "warming up"
        last_logged_visualized_state: tuple[float, float, float, int, str] | None = None
        live_pull_count = 0
        total_samples_appended = 0
        samples_since_last_decode = 0
        last_decoded_sample_total = -1
        last_decoded_end_ts: float | None = None
        accepted_feature_zscores: list[float] = []
        accepted_right_probs: list[float] = []
        accepted_left_probs: list[float] = []
        bias_offset = float(task_cfg.live_bias_offset) if bool(task_cfg.enable_live_bias_offset) else 0.0

        live_filter = StreamingIIRFilter.from_eeg_config(
            eeg_cfg=eeg_cfg,
            sfreq=sfreq,
            n_channels=len(model_ch_names),
        )
        live_buffer = np.empty((len(model_ch_names), 0), dtype=np.float32)
        keep_n = int(round((task_cfg.window_s + task_cfg.filter_context_s) * sfreq))
        stream_pull_s = max(0.20, task_cfg.live_update_interval_s * 2.0)
        last_live_ts: float | None = None
        logger.info(
            "Entering live loop: keep_n=%d, window_n=%d, stream_pull_s=%.3f, live_update_interval_s=%.3f",
            keep_n,
            window_n,
            stream_pull_s,
            task_cfg.live_update_interval_s,
        )

        kf = PostLDA_KalmanSmoother(q=0.02, r=0.1)

        while session_clock.getTime() < task_cfg.live_duration_s:
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

            data, ts = stream.get_data(winsize=stream_pull_s, picks="all")
            if data.size > 0 and ts is not None and len(ts) > 0:
                ts = np.asarray(ts)
                mask = np.ones_like(ts, dtype=bool) if last_live_ts is None else (ts > float(last_live_ts))
                if np.any(mask):
                    x_new = np.asarray(data[:, mask], dtype=np.float32)
                    n_new = int(x_new.shape[1])
                    last_live_ts = float(ts[mask][-1])
                    x_new_filt = live_filter.process(x_new)
                    live_buffer = np.concatenate((live_buffer, x_new_filt), axis=1)
                    if live_buffer.shape[1] > keep_n:
                        live_buffer = live_buffer[:, -keep_n:]
                    live_pull_count += 1
                    total_samples_appended += n_new
                    samples_since_last_decode += n_new
                    raw_mean = float(np.mean(x_new))
                    raw_std = float(np.std(x_new))
                    filt_mean = float(np.mean(x_new_filt))
                    filt_std = float(np.std(x_new_filt))
                    start_ts = float(ts[mask][0])
                    logger.info(
                        "Live pull %d: appended=%d samples, total_appended=%d, ts_range=[%.6f, %.6f], "
                        "buffer_samples=%d, raw_mean=%.3f, raw_std=%.3f, filt_mean=%.3f, filt_std=%.3f",
                        live_pull_count,
                        n_new,
                        total_samples_appended,
                        start_ts,
                        last_live_ts,
                        int(live_buffer.shape[1]),
                        raw_mean,
                        raw_std,
                        filt_mean,
                        filt_std,
                    )
                else:
                    logger.warning(
                        "Stream pull returned no strictly new timestamps: last_live_ts=%.6f, received_range=[%.6f, %.6f], "
                        "received_samples=%d",
                        float(last_live_ts) if last_live_ts is not None else -1.0,
                        float(ts[0]),
                        float(ts[-1]),
                        int(ts.size),
                    )
            else:
                logger.warning("Stream pull returned no data: data_size=%d, ts_len=%d", int(data.size), 0 if ts is None else len(ts))

            if pred_clock.getTime() >= task_cfg.live_update_interval_s:
                pred_clock.reset()
                if live_buffer.shape[1] >= keep_n:
                    x_win = live_buffer[:, -window_n:]
                    window_mean = float(np.mean(x_win))
                    window_std = float(np.std(x_win))
                    window_ptp = float(np.ptp(x_win, axis=-1).max())
                    window_ch_std = np.std(x_win, axis=1)
                    window_ch_ptp = np.ptp(x_win, axis=1)
                    is_fresh_window = samples_since_last_decode > 0
                    decode_end_ts = last_live_ts
                    if reject_thresh is None or float(np.ptp(x_win, axis=-1).max()) <= float(reject_thresh):
                        x_feat = np.asarray(feature_extractor.transform(x_win[np.newaxis, ...])[0], dtype=np.float64)
                        p_vec = classifier.predict_proba(x_win[np.newaxis, ...])[0]
                        pred_code = int(classifier.predict(x_win[np.newaxis, ...])[0])
                        prob_map = {
                            int(cls): float(p_vec[idx])
                            for idx, cls in enumerate(classifier_classes.tolist())
                        }
                        feature_abs_z = np.abs((x_feat - train_feature_mean) / np.maximum(train_feature_std, 1e-6))
                        feature_z_l2 = float(np.sqrt(np.mean(feature_abs_z ** 2)))
                        centroid_dist_map = {
                            int(code): float(np.linalg.norm(x_feat - centroid))
                            for code, centroid in train_feature_centroids.items()
                        }
                        ch_std_deviation = _top_channel_deviation_summary(
                            channel_names=model_ch_names,
                            current_values=window_ch_std,
                            ref_means=train_ch_std_mean,
                            ref_stds=train_ch_std_std,
                        )
                        ch_ptp_deviation = _top_channel_deviation_summary(
                            channel_names=model_ch_names,
                            current_values=window_ch_ptp,
                            ref_means=train_ch_ptp_mean,
                            ref_stds=train_ch_ptp_std,
                        )
                        prediction_count += 1
                        live_note = "updating"
                        accepted_feature_zscores.append(feature_z_l2)
                        accepted_left_probs.append(float(prob_map[int(stim_cfg.left_code)]))
                        accepted_right_probs.append(float(prob_map[int(stim_cfg.right_code)]))
                        logger.info(
                            "Decode %d: fresh_window=%s, new_samples_since_last_decode=%d, total_samples=%d, "
                            "window_mean=%.3f %s, window_std=%.3f %s, window_ptp=%.3f %s, end_ts=%s, predicted_code=%d, "
                            "lr_margin_raw=%.6f, lr_margin_biased=%.6f, feature_z_l2=%.3f, probs_by_code=%s, centroid_dists=%s, "
                            "ch_std_top_deviation=%s, ch_ptp_top_deviation=%s",
                            prediction_count,
                            is_fresh_window,
                            samples_since_last_decode,
                            total_samples_appended,
                            window_mean,
                            dataset.eeg_units,
                            window_std,
                            dataset.eeg_units,
                            window_ptp,
                            dataset.eeg_units,
                            "None" if decode_end_ts is None else f"{decode_end_ts:.6f}",
                            pred_code,
                            float(prob_map.get(int(stim_cfg.right_code), 0.0) - prob_map.get(int(stim_cfg.left_code), 0.0)),
                            float(prob_map.get(int(stim_cfg.right_code), 0.0) - prob_map.get(int(stim_cfg.left_code), 0.0) + bias_offset),
                            feature_z_l2,
                            prob_map,
                            centroid_dist_map,
                            ch_std_deviation,
                            ch_ptp_deviation,
                        )
                        if not is_fresh_window or last_decoded_sample_total == total_samples_appended:
                            logger.warning(
                                "Decode %d reused the previous live window: last_decoded_sample_total=%d, total_samples=%d, "
                                "last_decoded_end_ts=%s, current_end_ts=%s",
                                prediction_count,
                                last_decoded_sample_total,
                                total_samples_appended,
                                "None" if last_decoded_end_ts is None else f"{last_decoded_end_ts:.6f}",
                                "None" if decode_end_ts is None else f"{decode_end_ts:.6f}",
                            )
                        last_decoded_sample_total = total_samples_appended
                        last_decoded_end_ts = decode_end_ts
                        samples_since_last_decode = 0
                    else:
                        live_note = "artifact reject"
                        logger.warning(
                            "Decode skipped by artifact reject: new_samples_since_last_decode=%d, total_samples=%d, "
                            "window_ptp=%.3f %s, threshold=%.3f %s, end_ts=%s, ch_ptp_by_channel=%s",
                            samples_since_last_decode,
                            total_samples_appended,
                            window_ptp,
                            dataset.eeg_units,
                            float(reject_thresh),
                            dataset.eeg_units,
                            "None" if decode_end_ts is None else f"{decode_end_ts:.6f}",
                            _channel_stat_map(model_ch_names, window_ch_ptp),
                        )
                else:
                    needed_s = max(0.0, (keep_n - live_buffer.shape[1]) / sfreq)
                    live_note = f"warming up filter ({needed_s:.1f}s)"
                    logger.info(
                        "Decode waiting for warmup: buffer_samples=%d/%d, needed_s=%.3f, total_samples=%d",
                        int(live_buffer.shape[1]),
                        keep_n,
                        needed_s,
                        total_samples_appended,
                    )

            left_p = float(p_vec[class_index[int(stim_cfg.left_code)]])
            right_p = float(p_vec[class_index[int(stim_cfg.right_code)]])
            rest_p = (
                float(p_vec[class_index[int(task_cfg.rest_class_code)]])
                if int(task_cfg.rest_class_code) in class_index
                else 0.0
            )

            signed_score = right_p - left_p + bias_offset

            # run signed score through KF to smooth it
            if task_cfg.activate_kf and len(accepted_feature_zscores) > 0:
                l2_dist = accepted_feature_zscores[-1]
                r_adaptive = 0.1 * (1.0 + (l2_dist / 10.0)**2)
                if task_cfg.make_kf_adaptive:
                    signed_score = kf.step(signed_score, r_adapted=r_adaptive)
                else:
                    signed_score = kf.step(signed_score)

            update_bar(signed_score)
            visualized_state = (
                round(left_p, 6),
                round(right_p, 6),
                round(rest_p, 6),
                round(signed_score, 6),
                int(prediction_count),
                str(live_note),
            )
            if visualized_state != last_logged_visualized_state:
                logger.info(
                    "Visualized state: left_p=%.4f, right_p=%.4f, rest_p=%.4f, signed_score=%.4f, bias_offset=%.4f, prediction_count=%d, note=%s",
                    left_p,
                    right_p,
                    rest_p,
                    signed_score,
                    bias_offset,
                    prediction_count,
                    live_note,
                )
                last_logged_visualized_state = visualized_state
            detected_parts = [
                f"{label_cfg.left_name}: {left_p:.2f}",
                f"{label_cfg.right_name}: {right_p:.2f}",
            ]
            if int(task_cfg.rest_class_code) in class_index:
                detected_parts.append(f"{label_cfg.rest_name}: {rest_p:.2f}")
            detected_parts.extend([
                f"margin={signed_score:+.2f}",
                f"bias={bias_offset:+.2f}",
                f"updates={prediction_count}",
                f"{live_note}",
            ])
            detected.text = "   ".join(detected_parts)
            draw_frame()

        cue.text = "Live session complete"
        status.text = "Press ESC to close."
        detected.text = ""
        while True:
            draw_frame()
            if "escape" in event.getKeys():
                break

    except KeyboardInterrupt:
        logger.info("Session interrupted by user.")
        print("\nSession interrupted.")
    finally:
        if classifier is not None and accepted_feature_zscores:
            logger.info(
                "Live accepted window summary: n=%d, feature_z_l2_mean=%.3f, feature_z_l2_max=%.3f, "
                "left_prob_mean=%.6f, right_prob_mean=%.6f",
                len(accepted_feature_zscores),
                float(np.mean(accepted_feature_zscores)),
                float(np.max(accepted_feature_zscores)),
                float(np.mean(accepted_left_probs)),
                float(np.mean(accepted_right_probs)),
            )
        logger.info("Shutting down live session.")
        try:
            stream.disconnect()
        except Exception:
            pass
        try:
            win.close()
        except Exception:
            pass


def _sanitize_participant_name(raw_name: str) -> str:
    cleaned = "_".join(raw_name.strip().lower().split())
    cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch == "_")
    return cleaned.strip("_")


def _build_session_prefix(participant: str) -> str:
    date_prefix = datetime.now().strftime("%m_%d_%y")
    return f"{date_prefix}_{participant}_mi_visualizer"


def _prompt_session_prefix() -> str:
    while True:
        raw = input("Enter participant name: ")
        participant = _sanitize_participant_name(raw)
        if participant:
            return _build_session_prefix(participant)
        print("Participant name cannot be empty. Please try again.")


if __name__ == "__main__":
    fname = _prompt_session_prefix()
    print(f"[SESSION] Using filename prefix: {fname}")
    run_task(fname=fname)
