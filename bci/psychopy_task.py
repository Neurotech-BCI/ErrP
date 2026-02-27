# psychopy_task.py  –  Single-process motor-imagery BCI
#
# Connects to the DSI-7 EEG LSL stream, sends hardware triggers via the
# trigger hub, uses EpochsStream for precise hardware-synced epoching,
# and runs calibration + online classification in one synchronous loop.
# The raw CSV captures EEG + TRG so every event marker is preserved.
from __future__ import annotations

import random
import time
import serial

import numpy as np
from psychopy import visual, core, event

from mne_lsl.stream import StreamLSL, EpochsStream

from config import (
    LSLConfig,
    EEGConfig,
    ModelConfig,
    CalibrationConfig,
    SessionConfig,
    StimConfig,
    SerialConfig,
)
from bci_worker import (
    train_initial_classifier,
    run_cv,
    filter_epoch,
    RawCSVRecorder,
)


# ----------------------------------------------------------------
# Serial trigger helper
# ----------------------------------------------------------------


class TriggerPort:
    """Send trigger codes to the hardware trigger hub via serial."""

    def __init__(self, cfg: SerialConfig):
        self.port = cfg.port
        self.baudrate = cfg.baudrate
        self.pulse_width_s = cfg.pulse_width_s
        self.ser = None

    def open(self):
        self.ser = serial.Serial(self.port, self.baudrate, timeout=0)
        time.sleep(0.05)

    def close(self):
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def pulse(self, code: int):
        """Send a single trigger code, then reset to 0 after pulse_width_s."""
        if self.ser is None:
            return
        code = int(code) & 0xFF
        self.ser.write(bytes([code]))
        self.ser.flush()
        core.wait(self.pulse_width_s)
        self.ser.write(bytes([0]))
        self.ser.flush()


# ----------------------------------------------------------------
# Scheduler
# ----------------------------------------------------------------


class BalancedBlockScheduler:
    """Generates approximately balanced LEFT/RIGHT codes in shuffled blocks."""

    def __init__(self, block_size: int, left_code: int, right_code: int, seed: int | None = None):
        if block_size < 2:
            raise ValueError("block_size must be >= 2.")
        self.block_size = block_size
        self.left_code = int(left_code)
        self.right_code = int(right_code)
        self.rng = random.Random(seed)
        self._bag: list[int] = []

    def _refill(self):
        n_left = self.block_size // 2
        n_right = self.block_size - n_left
        self._bag = [self.left_code] * n_left + [self.right_code] * n_right
        self.rng.shuffle(self._bag)

    def next_code(self) -> int:
        if not self._bag:
            self._refill()
        return self._bag.pop()


# ----------------------------------------------------------------
# PsychoPy task
# ----------------------------------------------------------------


def run_task():
    # ---- Configuration ----
    lsl_cfg = LSLConfig()
    eeg_cfg = EEGConfig()
    model_cfg = ModelConfig()
    cal_cfg = CalibrationConfig()
    session_cfg = SessionConfig()
    stim_cfg = StimConfig()
    ser_cfg = SerialConfig()

    LEFT = stim_cfg.left_code
    RIGHT = stim_cfg.right_code
    CORRECT = stim_cfg.correct_code
    ERROR = stim_cfg.error_code
    reject_thresh = eeg_cfg.reject_peak_to_peak

    def code_to_name(code: int) -> str:
        return "LEFT" if int(code) == LEFT else "RIGHT"

    def code_to_side(code: int) -> int:
        return 0 if int(code) == LEFT else 1

    # Timing
    PREP_DURATION = 2.0
    MI_DURATION = eeg_cfg.tmax - eeg_cfg.tmin  # e.g. 2.0 s
    ITI = 3.0
    N_CAL_TRIALS = cal_cfg.n_calibration_trials
    N_LIVE_TRIALS = 60
    EPOCH_POLL_TIMEOUT = eeg_cfg.tmin + 1.5  # seconds to wait for epoch after MI

    # UI geometry & colours
    WIN_SIZE = (1200, 700)
    TARGET_OFFSET_X = 0.45
    TARGET_RADIUS = 0.07
    CURSOR_RADIUS = 0.04
    BG = (-0.1, -0.1, -0.1)
    WHITE = (0.9, 0.9, 0.9)
    DIM = (0.35, 0.35, 0.35)
    LIT = (0.9, 0.9, 0.2)
    CURSOR = (0.2, 0.8, 0.9)

    # ---- Connect to EEG LSL stream ----
    stream_bufsize = max(30.0, eeg_cfg.tmax + 5.0)
    stream = StreamLSL(
        bufsize=stream_bufsize,
        name=lsl_cfg.name,
        stype=lsl_cfg.stype,
        source_id=lsl_cfg.source_id,
    )
    stream.connect(acquisition_delay=0.001, processing_flags="all")
    print(f"[LSL] Stream info: {stream.info}")
    print(f"[LSL] Original channels: {stream.info['ch_names']}")

    # Pick EEG + trigger channel; keep order stable: EEG picks then TRG
    raw_picks = list(eeg_cfg.picks) + [lsl_cfg.event_channels]
    stream.pick(raw_picks)
    stream.set_channel_types({lsl_cfg.event_channels: "stim"})
    sfreq = float(stream.info["sfreq"])
    ch_names = list(stream.info["ch_names"])
    print(f"[LSL] Connected: sfreq={sfreq:.1f} Hz, channels={ch_names}, buf={stream_bufsize}s")

    # ---- EpochsStream for hardware-triggered epoching ----
    epochs_online = EpochsStream(
        stream,
        bufsize=30,
        event_id={"left": LEFT, "right": RIGHT},
        event_channels=lsl_cfg.event_channels,
        tmin=eeg_cfg.tmin,
        tmax=eeg_cfg.tmax,
        baseline=eeg_cfg.baseline,
        reject=None,
    ).connect(acquisition_delay=0.001)

    # ---- Trigger hub ----
    trig = TriggerPort(ser_cfg)
    trig.open()

    # ---- Raw CSV recorder (EEG + TRG) ----
    raw_csv_path = f"{session_cfg.name}{session_cfg.raw_csv_suffix}"
    raw_recorder = RawCSVRecorder(filepath=raw_csv_path, ch_names=ch_names)
    raw_recorder.start()

    def tick_recorder():
        """Call periodically to flush stream data to CSV."""
        if raw_recorder.is_active():
            raw_recorder.update(stream)

    # ---- PsychoPy window & stimuli ----
    win = visual.Window(size=WIN_SIZE, color=BG, units="norm", fullscr=False)
    left_target = visual.Circle(
        win, radius=TARGET_RADIUS, edges=64, pos=(-TARGET_OFFSET_X, 0), fillColor=DIM, lineColor=WHITE,
    )
    right_target = visual.Circle(
        win, radius=TARGET_RADIUS, edges=64, pos=(+TARGET_OFFSET_X, 0), fillColor=DIM, lineColor=WHITE,
    )
    cursor = visual.Circle(win, radius=CURSOR_RADIUS, edges=64, pos=(0, 0), fillColor=CURSOR, lineColor=WHITE)
    cue_text = visual.TextStim(win, text="", pos=(0, 0.35), height=0.08, color=WHITE)
    status_text = visual.TextStim(win, text="", pos=(0, -0.45), height=0.05, color=WHITE)

    def draw_scene():
        left_target.draw()
        right_target.draw()
        cursor.draw()
        cue_text.draw()
        status_text.draw()

    def set_targets(lit_side: int | None):
        left_target.fillColor = LIT if lit_side == 0 else DIM
        right_target.fillColor = LIT if lit_side == 1 else DIM

    def move_cursor_to(side: int, duration: float = 0.5):
        start = cursor.pos
        end = left_target.pos if side == 0 else right_target.pos
        clock = core.Clock()
        while clock.getTime() < duration:
            t = clock.getTime() / duration
            cursor.pos = (start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)
            tick_recorder()
            draw_scene()
            win.flip()
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

    def reset_cursor(duration: float = 0.25):
        start = cursor.pos
        end = (0, 0)
        clock = core.Clock()
        while clock.getTime() < duration:
            t = clock.getTime() / duration
            cursor.pos = (start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)
            tick_recorder()
            draw_scene()
            win.flip()
            if "escape" in event.getKeys():
                raise KeyboardInterrupt
        cursor.pos = end

    def wait_with_display(duration: float):
        """Render loop for a fixed duration, ticking the CSV recorder."""
        clock = core.Clock()
        while clock.getTime() < duration:
            tick_recorder()
            draw_scene()
            win.flip()
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

    def wait_for_space():
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

    # ==============================================================
    # INSTRUCTION SCREEN
    # ==============================================================
    cue_text.text = (
        "Motor Imagery BCI\n\n"
        "Phase 1: Calibration\n"
        f"First {N_CAL_TRIALS} normal trials are used for calibration.\n"
        "No cursor feedback during calibration.\n\n"
        "Phase 2: Online feedback\n"
        "Short MI trials with cursor feedback.\n"
        "Press SPACE to begin. ESC to quit."
    )
    status_text.text = ""
    set_targets(None)
    cursor.pos = (0, 0)
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

    # ==============================================================
    # PHASE 1: CALIBRATION (normal trials, no feedback)
    # ==============================================================
    X_cal: list[np.ndarray] = []
    y_cal: list[int] = []
    X_all: list[np.ndarray] = []
    y_all: list[int] = []

    cal_scheduler = BalancedBlockScheduler(
        block_size=max(2, model_cfg.retrain_every), left_code=LEFT, right_code=RIGHT,
    )

    try:
        for cal_idx in range(N_CAL_TRIALS):
            y_true_code = cal_scheduler.next_code()

            # --- ITI ---
            set_targets(None)
            cue_text.text = ""
            status_text.text = f"Calibration Trial {cal_idx + 1}/{N_CAL_TRIALS}"
            reset_cursor(duration=0.15)
            wait_with_display(ITI)

            # --- Prepare ---
            set_targets(None)
            cue_text.text = f"Prepare: {code_to_name(y_true_code)}"
            status_text.text = "Get ready..."
            wait_with_display(PREP_DURATION)

            # --- MI cue onset: display then trigger ---
            set_targets(code_to_side(y_true_code))
            cue_text.text = f"IMAGINE: {code_to_name(y_true_code)}"
            status_text.text = "Calibration (no feedback)"
            draw_scene()
            win.flip()
            trig.pulse(y_true_code)

            # Wait for MI duration
            wait_with_display(MI_DURATION)

            # Poll EpochsStream for the hardware-triggered epoch
            epoch, code = poll_epoch(timeout_s=EPOCH_POLL_TIMEOUT)
            if epoch is None:
                print(f"[CAL] Trial {cal_idx + 1}: no epoch received (timeout)")
                continue

            if code != y_true_code:
                print(f"[CAL] Trial {cal_idx + 1}: code mismatch (expected {y_true_code}, got {code})")

            epoch = filter_epoch(epoch, eeg_cfg, sfreq)

            if reject_thresh is not None and np.ptp(epoch, axis=-1).max() > reject_thresh:
                print(f"[CAL] Trial {cal_idx + 1}: rejected (artifact)")
                continue

            X_cal.append(epoch)
            y_cal.append(y_true_code)
            X_all.append(epoch)
            y_all.append(y_true_code)

            set_targets(None)
            cue_text.text = ""
            status_text.text = f"Epoch captured ({len(X_cal)}/{N_CAL_TRIALS})"
            draw_scene()
            win.flip()

        # ==============================================================
        # TRAINING
        # ==============================================================
        cue_text.text = "Training classifier on calibration data...\nPlease wait."
        status_text.text = ""
        draw_scene()
        win.flip()

        classifier = None
        best_C: float | None = None

        try:
            classifier, best_C, cv_mean, cv_std, n_per_class = train_initial_classifier(
                X_cal, y_cal, model_cfg, LEFT, RIGHT,
            )
            cue_text.text = (
                f"Calibration Complete!\n\n"
                f"Cross-validated accuracy: {cv_mean:.1%} +/- {cv_std:.1%}\n"
                f"Epochs: {len(y_cal)} "
                f"(L={n_per_class.get(str(LEFT), 0)}, R={n_per_class.get(str(RIGHT), 0)})\n"
                f"Selected C: {best_C}\n\n"
                f"Press SPACE to begin online phase."
            )
        except Exception as exc:
            cue_text.text = f"Training error:\n{exc}\n\nPress SPACE to continue anyway."
            print(f"[TRAIN] Failed: {exc}")

        status_text.text = ""
        draw_scene()
        win.flip()
        wait_for_space()

        if classifier is None:
            cue_text.text = "No classifier available.\nPress ESC to exit."
            status_text.text = ""
            draw_scene()
            win.flip()
            while "escape" not in event.getKeys():
                tick_recorder()
            raise KeyboardInterrupt  # jump to cleanup

        # ==============================================================
        # PHASE 2: ONLINE TRIALS
        # ==============================================================
        X_store: list[np.ndarray] = list(X_cal)
        y_store: list[int] = list(y_cal)
        total_accepted = len(X_cal)
        last_train_at = total_accepted
        correct_count = 0

        scheduler = BalancedBlockScheduler(
            block_size=max(2, model_cfg.retrain_every), left_code=LEFT, right_code=RIGHT,
        )

        for live_idx in range(N_LIVE_TRIALS):
            y_true_code = scheduler.next_code()

            # --- ITI ---
            set_targets(None)
            cue_text.text = ""
            if live_idx > 0:
                status_text.text = (
                    f"Live Trial {live_idx + 1}/{N_LIVE_TRIALS} | "
                    f"Accuracy: {correct_count}/{live_idx}"
                )
            else:
                status_text.text = f"Live Trial 1/{N_LIVE_TRIALS}"
            reset_cursor(duration=0.15)
            wait_with_display(ITI)

            # --- Prepare ---
            set_targets(None)
            cue_text.text = f"Prepare: {code_to_name(y_true_code)}"
            status_text.text = "Get ready..."
            wait_with_display(PREP_DURATION)

            # --- MI cue onset: display then trigger ---
            set_targets(code_to_side(y_true_code))
            cue_text.text = f"IMAGINE: {code_to_name(y_true_code)}"
            status_text.text = "Go!"
            draw_scene()
            win.flip()
            trig.pulse(y_true_code)

            # Wait for MI duration
            wait_with_display(MI_DURATION)

            # Poll EpochsStream for the epoch
            epoch, code = poll_epoch(timeout_s=EPOCH_POLL_TIMEOUT)
            if epoch is None:
                status_text.text = "No epoch received."
                draw_scene()
                win.flip()
                wait_with_display(0.4)
                set_targets(None)
                continue

            if code != y_true_code:
                print(
                    f"[ONLINE] Trial {live_idx + 1}: code mismatch "
                    f"(expected {y_true_code}, got {code})"
                )

            epoch = filter_epoch(epoch, eeg_cfg, sfreq)

            # Artifact rejection
            if reject_thresh is not None and np.ptp(epoch, axis=-1).max() > reject_thresh:
                status_text.text = "Epoch rejected (artifact)."
                draw_scene()
                win.flip()
                wait_with_display(0.4)
                set_targets(None)
                continue

            # Accumulate
            X_all.append(epoch)
            y_all.append(y_true_code)
            X_store.append(epoch)
            y_store.append(y_true_code)
            total_accepted += 1

            # --- Predict ---
            x_i = epoch[np.newaxis, ...]  # (1, n_ch, n_samp)
            proba = classifier.predict_proba(x_i)[0]
            classes = classifier.named_steps["clf"].classes_
            y_pred_code = int(classes[int(np.argmax(proba))])
            conf = float(np.max(proba))

            # --- Feedback: send ErrP marker at cursor movement instant ---
            is_correct = y_pred_code == y_true_code
            if is_correct:
                correct_count += 1
            trig.pulse(CORRECT if is_correct else ERROR)

            cue_text.text = ""
            status_text.text = f"Pred: {code_to_name(y_pred_code)} | conf={conf:.2f}"
            move_cursor_to(code_to_side(y_pred_code), duration=0.5)

            wait_with_display(0.4)
            set_targets(None)

            # --- Sliding window ---
            if model_cfg.use_sliding_window and len(y_store) > model_cfg.window_size_epochs:
                X_store = X_store[-model_cfg.window_size_epochs :]
                y_store = y_store[-model_cfg.window_size_epochs :]

            # --- Online retraining ---
            if (total_accepted - last_train_at) >= model_cfg.retrain_every and len(y_store) >= 4:
                status_text.text = "Retraining..."
                draw_scene()
                win.flip()

                X_train = np.stack(X_store, axis=0)
                y_train = np.array(y_store, dtype=int)
                classifier.fit(X_train, y_train)
                last_train_at = total_accepted
                print(f"[RETRAIN] Fitted on {len(y_store)} epochs (total accepted: {total_accepted})")

        # ==============================================================
        # SESSION COMPLETE
        # ==============================================================
        cue_text.text = (
            f"Session complete.\n"
            f"Online accuracy: {correct_count}/{N_LIVE_TRIALS}\n\n"
            f"Data saved.\n"
            f"Press ESC to close."
        )
        status_text.text = ""
        draw_scene()
        win.flip()
        while "escape" not in event.getKeys():
            tick_recorder()

    except KeyboardInterrupt:
        print("\nSession interrupted.")

    finally:
        # ---- Cleanup ----
        raw_recorder.stop()

        # Save epoch data + final CV
        if len(y_all) > 0:
            X_save = np.stack(X_all, axis=0)
            y_save = np.array(y_all, dtype=int)
            np.save(f"{session_cfg.name}_data.npy", X_save)
            np.save(f"{session_cfg.name}_labels.npy", y_save)
            print(f"[SAVE] {X_save.shape[0]} epochs -> {session_cfg.name}_data.npy")

            cv_mean, cv_std, cv_scores = run_cv(X_save, y_save, model_cfg, fixed_C=best_C)
            if len(cv_scores) > 0:
                print(f"\nFinal CV (fixed C={best_C}): {cv_mean:.3f} +/- {cv_std:.3f}")

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


if __name__ == "__main__":
    try:
        run_task()
    except KeyboardInterrupt:
        pass
