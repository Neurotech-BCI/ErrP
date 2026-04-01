from __future__ import annotations

import pickle
from collections import Counter
from datetime import datetime

import numpy as np
from psychopy import core, event, visual

from mne_lsl.stream import StreamLSL

from config import EEGConfig, LSLConfig, MentalCommandLabelConfig, MentalCommandModelConfig, MentalCommandTaskConfig, StimConfig
from mental_command_worker import (
    StreamingIIRFilter,
    canonicalize_channel_name,
    evaluate_loso_sessions,
    load_offline_mi_dataset,
    make_mi_classifier,
    resolve_channel_order,
)


def run_task(fname: str):
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

    classifier = None
    class_index = None
    dataset = None
    reject_thresh = eeg_cfg.reject_peak_to_peak
    window_n = int(round(task_cfg.window_s * sfreq))

    try:
        cue.text = "Preparing model from offline EDF sessions..."
        status.text = (
            f"Loading data from {task_cfg.data_dir}\n"
            "Offline EDFs are standardized to the live stream convention: left-ear referenced, standard channel names,\n"
            "then causally filtered and windowed exactly like the live stream."
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
        )
        classes_present = {int(c) for c in np.unique(dataset.y)}
        expected_classes = {int(stim_cfg.left_code), int(stim_cfg.right_code)}
        if classes_present != expected_classes:
            raise RuntimeError(
                f"Training data must contain both left/right classes. "
                f"Found {sorted(classes_present)}, expected {sorted(expected_classes)}."
            )

        loso = evaluate_loso_sessions(dataset, model_cfg)
        classifier = make_mi_classifier(model_cfg)
        classifier.fit(dataset.X, dataset.y)
        class_index = {int(c): i for i, c in enumerate(classifier.named_steps["clf"].classes_)}

        counts = Counter(dataset.y.tolist())
        np.save(f"{fname}_mi_visualizer_windows.npy", dataset.X)
        np.save(f"{fname}_mi_visualizer_labels.npy", dataset.y)
        with open(f"{fname}_mi_visualizer_model.pkl", "wb") as fh:
            pickle.dump(classifier, fh)

        session_lines = []
        for session_id, score in sorted(loso.session_scores.items()):
            session_lines.append(f"S{session_id}: {score:.3f}")
        session_summary = "  ".join(session_lines) if session_lines else "No valid held-out sessions"

        cue.text = "Model ready"
        status.text = (
            f"Files used: {dataset.n_files_used}/{dataset.n_files_found}  "
            f"Trials: {dataset.n_trials}  Windows: {dataset.n_windows}\n"
            f"{label_cfg.left_name}: {counts[int(stim_cfg.left_code)]}  "
            f"{label_cfg.right_name}: {counts[int(stim_cfg.right_code)]}\n"
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
        p_vec = np.array([0.5, 0.5], dtype=np.float64)
        prediction_count = 0
        live_note = "warming up"

        live_filter = StreamingIIRFilter.from_eeg_config(
            eeg_cfg=eeg_cfg,
            sfreq=sfreq,
            n_channels=len(model_ch_names),
        )
        live_buffer = np.empty((len(model_ch_names), 0), dtype=np.float32)
        keep_n = int(round((task_cfg.window_s + task_cfg.filter_context_s) * sfreq))
        stream_pull_s = max(0.20, task_cfg.live_update_interval_s * 2.0)
        last_live_ts: float | None = None

        while session_clock.getTime() < task_cfg.live_duration_s:
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

            data, ts = stream.get_data(winsize=stream_pull_s, picks="all")
            if data.size > 0 and ts is not None and len(ts) > 0:
                ts = np.asarray(ts)
                mask = np.ones_like(ts, dtype=bool) if last_live_ts is None else (ts > float(last_live_ts))
                if np.any(mask):
                    x_new = np.asarray(data[:, mask], dtype=np.float32)
                    last_live_ts = float(ts[mask][-1])
                    x_new_filt = live_filter.process(x_new)
                    live_buffer = np.concatenate((live_buffer, x_new_filt), axis=1)
                    if live_buffer.shape[1] > keep_n:
                        live_buffer = live_buffer[:, -keep_n:]

            if pred_clock.getTime() >= task_cfg.live_update_interval_s:
                pred_clock.reset()
                if live_buffer.shape[1] >= window_n:
                    x_win = live_buffer[:, -window_n:]
                    if reject_thresh is None or float(np.ptp(x_win, axis=-1).max()) <= float(reject_thresh):
                        p_vec = classifier.predict_proba(x_win[np.newaxis, ...])[0]
                        prediction_count += 1
                        live_note = "updating"
                    else:
                        live_note = "artifact reject"
                else:
                    needed_s = max(0.0, (window_n - live_buffer.shape[1]) / sfreq)
                    live_note = f"warming up ({needed_s:.1f}s)"

            left_p = float(p_vec[class_index[int(stim_cfg.left_code)]])
            right_p = float(p_vec[class_index[int(stim_cfg.right_code)]])
            signed_score = right_p - left_p
            update_bar(signed_score)
            detected.text = (
                f"{label_cfg.left_name}: {left_p:.2f}   "
                f"{label_cfg.right_name}: {right_p:.2f}   "
                f"margin={signed_score:+.2f}   "
                f"updates={prediction_count}   "
                f"{live_note}"
            )
            draw_frame()

        cue.text = "Live session complete"
        status.text = "Press ESC to close."
        detected.text = ""
        while True:
            draw_frame()
            if "escape" in event.getKeys():
                break

    except KeyboardInterrupt:
        print("\nSession interrupted.")
    finally:
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
