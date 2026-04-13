"""Online motor-imagery task.

This script combines the MI task with the live mental command task to create a intermediary hybrid. Training wheels.
"""
from __future__ import annotations

import logging
import pickle
from collections import Counter
from datetime import datetime

import numpy as np

from dataclasses import dataclass
from datetime import datetime
import argparse
import random
import time

import serial
import serial.tools.list_ports
from psychopy import core, event, visual

from mne_lsl.stream import StreamLSL, EpochsStream

from config import EEGConfig, LSLConfig, MentalCommandLabelConfig, MentalCommandModelConfig, LiveMITaskConfig, StimConfig, SerialConfig
from mental_command_worker import (
    StreamingIIRFilter,
    canonicalize_channel_name,
    evaluate_loso_sessions,
    load_dataset_for_live_task,
    make_mi_classifier,
    resolve_channel_order,
    filter_block
)

from bci_worker import (
    RawCSVRecorder
)

# functions from mental_command_task.py

def _make_task_logger(fname: str) -> logging.Logger:
    logger = logging.getLogger(f"live_mi_task.{fname}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    file_handler = logging.FileHandler(f"{fname}_live_mi_task.log", mode="w")
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

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------


class TriggerPort:
    """Send trigger codes to the hardware trigger hub via serial."""

    def __init__(self, port: str, baudrate: int, pulse_width_s: float):
        self.port = port
        self.baudrate = int(baudrate)
        self.pulse_width_s = float(pulse_width_s)
        self.ser: serial.Serial | None = None

    def open(self) -> None:
        self.ser = serial.Serial(self.port, self.baudrate, timeout=0)
        time.sleep(0.05)

    def close(self) -> None:
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def pulse(self, code: int) -> None:
        if self.ser is None:
            return
        code = int(code) & 0xFF
        self.ser.write(bytes([code]))
        self.ser.flush()
        core.wait(self.pulse_width_s)
        self.ser.write(bytes([0]))
        self.ser.flush()


class BalancedBlockScheduler:
    """Generate approximately balanced LEFT/RIGHT codes in shuffled blocks."""

    def __init__(self, block_size: int, left_code: int, right_code: int, seed: int | None = None):
        if block_size < 2:
            raise ValueError("block_size must be >= 2")
        self.block_size = int(block_size)
        self.left_code = int(left_code)
        self.right_code = int(right_code)
        self.rng = random.Random(seed)
        self._bag: list[int] = []

    def _refill(self) -> None:
        n_left = self.block_size // 2
        n_right = self.block_size - n_left
        self._bag = [self.left_code] * n_left + [self.right_code] * n_right
        self.rng.shuffle(self._bag)

    def next_code(self) -> int:
        if not self._bag:
            self._refill()
        return self._bag.pop()


def sanitize_participant_name(raw_name: str) -> str:
    cleaned = "_".join(raw_name.strip().lower().split())
    cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch == "_")
    return cleaned.strip("_")


def build_session_prefix(participant: str) -> str:
    date_prefix = datetime.now().strftime("%m_%d_%y")
    return f"{date_prefix}_{participant}_live_mi_cal"


_ARROW_VERTS_RIGHT = [
    (-0.25, 0.04), (0.05, 0.04), (0.05, 0.10),
    (0.25, 0.0),
    (0.05, -0.10), (0.05, -0.04), (-0.25, -0.04),
]
_ARROW_VERTS_LEFT = [(-x, y) for x, y in _ARROW_VERTS_RIGHT]


def run_task(
        fname: str,
        lsl_cfg = LSLConfig(),
        stim_cfg = StimConfig(),
        label_cfg = MentalCommandLabelConfig(),
        task_cfg = LiveMITaskConfig(),
        model_cfg = MentalCommandModelConfig(),
        eeg_cfg = EEGConfig(
            picks=("Pz", "F4", "C4", "P4", "P3", "C3", "F3"),
            l_freq=8.0,
            h_freq=30.0,
            reject_peak_to_peak=150.0,
        ),
        ser_cfg=SerialConfig()
    ):
    port = ser_cfg.port
    left = int(stim_cfg.left_code)
    right = int(stim_cfg.right_code)
    sfreq = 300.0

    stream = StreamLSL(
        bufsize=max(30.0, eeg_cfg.tmax + 10.0),
        name=lsl_cfg.name,
        stype=lsl_cfg.stype,
        source_id=lsl_cfg.source_id,
    )

    stream.connect(acquisition_delay=0.001, processing_flags="all")
    print(f"[LSL] Stream info: {stream.info}")
    logger = _make_task_logger(fname)
    logger.info("Connected to LSL stream: info=%s", stream.info)

    available = list(stream.info["ch_names"])
    model_ch_names, missing = resolve_channel_order(available, eeg_cfg.picks)
    
    stream.set_channel_types({lsl_cfg.event_channels: "stim"})

    epochs_online = EpochsStream(
        stream,
        bufsize=30,
        event_id={"left": left, "right": right},
        event_channels=lsl_cfg.event_channels,
        tmin=eeg_cfg.tmin,
        tmax=eeg_cfg.tmax,
        baseline=eeg_cfg.baseline,
        reject=None,
    ).connect(acquisition_delay=0.001)

    def code_to_name(code: int) -> str:
        return "LEFT" if int(code) == left else "RIGHT"

    trig = TriggerPort(port=port, baudrate=ser_cfg.baudrate, pulse_width_s=ser_cfg.pulse_width_s)
    trig.open()

    raw_csv_path = f"{fname}_raw.csv"
    raw_recorder = RawCSVRecorder(filepath=raw_csv_path, ch_names=stream.ch_names)
    raw_recorder.start()

    def tick_recorder():
        """Call periodically to flush stream data to CSV."""
        if raw_recorder.is_active():
            raw_recorder.update(stream)

    bg = (-0.1, -0.1, -0.1)
    white = (0.9, 0.9, 0.9)
    lit = (0.9, 0.9, 0.2)

    win = visual.Window(
        size=task_cfg.win_size,
        color=bg,
        units="norm",
        fullscr=task_cfg.fullscreen,
    )

    fixation = visual.TextStim(win, text="+", pos=(0, 0), height=0.16, color=white)
    cue_text = visual.TextStim(win, text="", pos=(0, 0.35), height=0.08, color=white)
    status_text = visual.TextStim(win, text="", pos=(0, -0.45), height=0.05, color=white)
    detected_text = visual.TextStim(win, text="", pos=(0, 0.26), height=0.055, color=(0.95, 0.95, 0.95))
    prep_arrow = visual.ShapeStim(
        win,
        vertices=_ARROW_VERTS_RIGHT,
        pos=(0, 0.15),
        fillColor=lit,
        lineColor=white,
        lineWidth=2,
        opacity=0,
    )

    def draw_scene() -> None:
        fixation.draw()
        prep_arrow.draw()
        cue_text.draw()
        status_text.draw()

    def wait_with_display(duration: float) -> None:
        clock = core.Clock()
        while clock.getTime() < duration:
            tick_recorder()
            draw_scene()
            win.flip()
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

    def wait_for_space() -> None:
        while True:
            tick_recorder()
            draw_scene()
            win.flip()
            keys = event.getKeys()
            if "space" in keys:
                return
            if "escape" in keys:
                raise KeyboardInterrupt
            
    def poll_epoch(timeout_s: float):
        """Wait for a new epoch from EpochsStream while keeping the display alive.

        Returns (epoch_data, event_code) or (None, None) on timeout.
        epoch_data shape: (n_eeg_channels, n_samples).
        """
        clock = core.Clock()
        initial_time = clock.getTime()
        while clock.getTime() < timeout_s:
            tick_recorder()
            draw_scene()
            win.flip()
            if "escape" in event.getKeys():
                raise KeyboardInterrupt
            n_new = epochs_online.n_new_epochs
            if n_new > 0:
                X_new = epochs_online.get_data(n_epochs=n_new, picks="eeg")
                codes = epochs_online.events[-n_new:]
                # Return the most recent left/right epoch
                return X_new[-1], int(codes[-1])
        return None, None
    
    # train on calibration files

    classifier = None
    class_index = None
    classifier_classes = None
    dataset = None
    reject_thresh = eeg_cfg.reject_peak_to_peak
    window_n = int(round(task_cfg.window_s * sfreq))

    try:
        logger.info(
            "Starting offline model preparation: data_dir=%s, edf_glob=%s, window_s=%.3f, step_s=%.3f, "
            "filter_band=[%.1f, %.1f], filter_context_s=%.3f",
            task_cfg.data_dir,
            task_cfg.edf_glob,
            task_cfg.window_s,
            task_cfg.window_step_s,
            eeg_cfg.l_freq,
            eeg_cfg.h_freq,
            task_cfg.filter_context_s,
        )
        cue_text.text = "Preparing model from offline EDF sessions..."
        status_text.text = (
            f"Loading data from {task_cfg.data_dir}\n"
            "Offline EDFs are standardized to the live stream convention: left-ear referenced, standard channel names,\n"
            "then causally filtered as full sessions before epoching/windowing to match live streaming."
        )
        detected_text.text = ""

        dataset = load_dataset_for_live_task(
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
            "loso_mean=%.4f, loso_std=%.4f",
            dataset.n_files_used,
            dataset.n_files_found,
            dataset.n_trials,
            dataset.n_windows,
            counts,
            loso.mean_accuracy,
            loso.std_accuracy,
        )
        logger.info(
            "Classifier class order: classes=%s, class_index=%s, left_code=%d -> prob_index=%d, right_code=%d -> prob_index=%d",
            classifier_classes.tolist(),
            class_index,
            int(stim_cfg.left_code),
            class_index[int(stim_cfg.left_code)],
            int(stim_cfg.right_code),
            class_index[int(stim_cfg.right_code)],
        )
        train_pred = classifier.predict(dataset.X)
        train_acc = float(np.mean(train_pred == dataset.y))
        train_pred_counts = Counter(train_pred.tolist())
        logger.info(
            "Classifier fit sanity check: training_accuracy=%.4f, predicted_counts=%s",
            train_acc,
            train_pred_counts,
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
        for session_id, score in sorted(loso.session_scores.items()):
            logger.info("LOSO session result: session=%d accuracy=%.4f", session_id, score)
        np.save(f"{fname}_live_mi_windows.npy", dataset.X)
        np.save(f"{fname}_live_mi_labels.npy", dataset.y)
        with open(f"{fname}_live_mi_model.pkl", "wb") as fh:
            pickle.dump(classifier, fh)

        session_lines = []
        for session_id, score in sorted(loso.session_scores.items()):
            session_lines.append(f"S{session_id}: {score:.3f}")
        session_summary = "  ".join(session_lines) if session_lines else "No valid held-out sessions"

        cue_text.text = "Model ready"
        status_text.text = (
            f"Files used: {dataset.n_files_used}/{dataset.n_files_found}  "
            f"Trials: {dataset.n_trials}  Windows: {dataset.n_windows}\n"
            f"{label_cfg.left_name}: {counts[int(stim_cfg.left_code)]}  "
            f"{label_cfg.right_name}: {counts[int(stim_cfg.right_code)]}\n"
            f"LOSO mean={loso.mean_accuracy:.3f} std={loso.std_accuracy:.3f}\n"
            f"{session_summary}\n"
            "Press SPACE to start live task. ESC to quit."
        )

        detected_text.text = (
            f"Window={task_cfg.window_s:.1f}s  "
            f"Step={task_cfg.window_step_s:.2f}s  "
            f"Filter={eeg_cfg.l_freq:.1f}-{eeg_cfg.h_freq:.1f} Hz"
        )
        wait_for_space()

        # I skipped the calibration block, if needed it should be added here

        X_live = []
        y_live = []

        scheduler = BalancedBlockScheduler(
            block_size=max(2, task_cfg.max_trials_before_break),
            left_code=left,
            right_code=right,
        )

        print(f"[SESSION] {fname}") 
        print(f"[SERIAL] Sending triggers on {port} @ {ser_cfg.baudrate}")

        cue_text.text = (
            "Brain Controlled Cursor!!!\n\n"
            "Fixate on the center cross.\n"
            "An arrow will cue LEFT or RIGHT preparation.\n"
            "Then execute the cued motor imagery when prompted.\n\n"
            "You will then see the live model output. Try to control it as well as possible.\n"
            "Press SPACE to begin. ESC to quit."
        )
        status_text.text = ""
        fixation.text = "+"
        try:
            wait_for_space()
        except KeyboardInterrupt:
            raw_recorder.stop()
            trig.close()
            try:
                epochs_online.disconnect()
            except Exception:
                pass
            stream.disconnect()
            win.close()
            return
        
        correct_count = 0
        prediction_count = 0

        epoch_filter = StreamingIIRFilter.from_eeg_config(
            eeg_cfg=eeg_cfg,
            sfreq=sfreq,
            n_channels=len(model_ch_names)
        )

        for live_idx in range(task_cfg.n_live_trials):
            y_true = scheduler.next_code()

            # ITI
            prep_arrow.opacity = 0
            cue_text.text = ""
            status_text.text = f"Live Trial {live_idx + 1}/{task_cfg.n_live_trials}"
            fixation.text = "+"
            wait_with_display(task_cfg.iti_duration_s)

            # Prepare phase
            cue_text.text = "Prepare"
            status_text.text = f"Get ready: {code_to_name(y_true)}"
            prep_arrow.vertices = _ARROW_VERTS_LEFT if y_true == left else _ARROW_VERTS_RIGHT
            prep_arrow.opacity = 1
            wait_with_display(task_cfg.prep_duration_s)

            # Execute phase: show cue then pulse trigger
            prep_arrow.opacity = 0
            cue_text.text = f"EXECUTE: {code_to_name(y_true)} MOTOR IMAGERY"
            status_text.text = "Go"
            draw_scene()
            win.flip()
            trig.pulse(y_true)
            wait_with_display(task_cfg.execution_duration_s)

            # Poll EpochsStream for the hardware-triggered epoch
            epoch_poll_timeout = eeg_cfg.tmax # copied from psychopy_task, may need to be changed
            epoch_raw, code = poll_epoch(timeout_s=epoch_poll_timeout)
            if epoch_raw is None:
                print(f"Trial {live_idx + 1}: no epoch received (timeout)")
                continue
            
            if code != y_true:
                print(f"Trial {live_idx + 1}: code mismatch (expected {y_true}, got {code})")

            epoch = epoch_filter.process(epoch_raw)

            if reject_thresh is not None and np.ptp(epoch, axis=-1).max() > reject_thresh:
                print(f"Trial {live_idx + 1}: rejected (artifact)")
                continue
            elif reject_thresh is not None or float(np.ptp(epoch, axis=-1).max()) <= float(reject_thresh):
                p_vec = classifier.predict_proba(epoch[np.newaxis, ...])[0]
                pred_code = int(classifier.predict(epoch[np.newaxis, ...])[0])
                prediction_count += 1

            X_live.append(epoch)
            y_live.append(y_true)

            # show predicted side
            print(f"model confidence in the classes {p_vec}")
            
            if (pred_code == y_true):
                correct_count += 1
            
            prep_arrow.vertices = _ARROW_VERTS_LEFT if pred_code == left else _ARROW_VERTS_RIGHT
            prep_arrow.opacity = 1

            cue_text.text = ""
            status_text.text = (
                    f"Live Trial {live_idx + 1}/{task_cfg.n_live_trials} | "
                    f"Accuracy: {correct_count}/{live_idx + 1}"
                )
            
            wait_with_display(0.8)
            draw_scene()
            win.flip()

            # Breaks
            if (
                (live_idx + 1) % task_cfg.max_trials_before_break == 0
                and (live_idx + 1) < task_cfg.n_live_trials
            ):
                cue_text.text = "Break\n\nPress SPACE to continue"
                status_text.text = f"Completed {live_idx + 1}/{task_cfg.n_live_trials} trials"
                prep_arrow.opacity = 0
                fixation.text = "+"
                wait_for_space()

        cue_text.text = "Session complete\n\nPress ESC to close"
        status_text.text = ""
        prep_arrow.opacity = 0
        fixation.text = "+"
        while "escape" not in event.getKeys():
            draw_scene()
            win.flip()

    except KeyboardInterrupt:
        logger.info("Session interrupted by user.")
        print("\nSession interrupted.")
    finally:
        logger.info("Shutting down live session.")

        # ---- Cleanup ----
        raw_recorder.stop()

        '''
        # Save epoch data + final CV
        if len(y_live) > 0:
            X_save = np.stack(X_live, axis=0)
            y_save = np.array(y_live, dtype=int)
            np.save(f"{fname}_data.npy", X_save)
            np.save(f"{fname}_labels.npy", y_save)
            print(f"[SAVE] {X_save.shape[0]} epochs -> {fname}_data.npy")

            # to be added: either predict on the live dataset using the pre-trained classifier or train on live data as well and predict on live data
        '''
        trig.close()
        for resource in [epochs_online, stream]:
            try:
                resource.disconnect()
            except Exception:
                pass
        try:
            win.close()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline MI calibration trigger task")
    parser.add_argument("--participant", type=str, default=None, help="Participant name for session label")
    parser.add_argument("--port", type=str, default=None, help="Serial port override (e.g., COM6 or /dev/ttyUSB0)")
    parser.add_argument("--trials", type=int, default=None, help="Override number of calibration trials")
    parser.add_argument("--fullscreen", action="store_true", help="Run fullscreen")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    participant = sanitize_participant_name(args.participant or input("Enter participant name: "))
    if not participant:
        raise ValueError("Participant name cannot be empty")

    session_name = build_session_prefix(participant)
    run_task(
        fname=session_name
    )


if __name__ == "__main__":
    main()
