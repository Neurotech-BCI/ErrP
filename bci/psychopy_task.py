# psychopy_task.py
from __future__ import annotations

import json
import random
import time
import serial

import zmq
from psychopy import visual, core, event

from config import (
    ZMQConfig,
    SerialConfig,
    ModelConfig,
    EEGConfig,
    CalibrationConfig,
    StimConfig,
)


# ---------------- Serial trigger helpers ----------------


class TriggerPort:
    def __init__(self, port: str, baudrate: int, pulse_width_s: float):
        self.port = port
        self.baudrate = baudrate
        self.pulse_width_s = pulse_width_s
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


# ---------------- ZMQ helpers ----------------


class BCILink:
    """Bidirectional ZMQ PAIR link to the BCI worker."""

    def __init__(self, addr: str):
        self.ctx = zmq.Context.instance()
        self.pair = self.ctx.socket(zmq.PAIR)
        self.pair.connect(addr)
        self.poller = zmq.Poller()
        self.poller.register(self.pair, zmq.POLLIN)

    def send(self, payload: dict):
        self.pair.send_string(json.dumps(payload))

    def recv(self, timeout_s: float = 10.0) -> dict | None:
        """Blocking receive with timeout. Returns None on timeout."""
        timeout_ms = int(timeout_s * 1000)
        socks = dict(self.poller.poll(timeout_ms))
        if self.pair in socks:
            return json.loads(self.pair.recv_string())
        return None

    def drain(self):
        """Non-blocking: discard any queued messages."""
        while True:
            socks = dict(self.poller.poll(0))
            if self.pair not in socks:
                break
            self.pair.recv_string()

    def close(self):
        try:
            self.pair.close(0)
        except Exception:
            pass


# ---------------- Schedulers ----------------


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


# ---------------- PsychoPy task ----------------


def run_task():
    zmq_cfg = ZMQConfig()
    ser_cfg = SerialConfig()
    model_cfg = ModelConfig()
    eeg_cfg = EEGConfig()
    cal_cfg = CalibrationConfig()
    stim_cfg = StimConfig()

    LEFT = stim_cfg.left_code
    RIGHT = stim_cfg.right_code
    CORRECT = stim_cfg.correct_code
    ERROR = stim_cfg.error_code

    def code_to_name(code: int) -> str:
        return "LEFT" if int(code) == LEFT else "RIGHT"

    def code_to_side(code: int) -> int:
        """Convert stim code -> side index for UI (0=left, 1=right)."""
        return 0 if int(code) == LEFT else 1

    # Timing
    PREP_DURATION = 2.0
    MI_DURATION = eeg_cfg.tmax - eeg_cfg.tmin  # online epoch duration (e.g., 2.0s)
    ITI = 3.0
    N_LIVE_TRIALS = 110
    N_CAL_TRIALS = cal_cfg.n_calibration_trials
    PRED_WAIT_TIMEOUT_S = 5.0
    CAL_TRAIN_TIMEOUT_S = 120.0

    # UI geometry
    WIN_SIZE = (1200, 700)
    TARGET_OFFSET_X = 0.45
    TARGET_RADIUS = 0.07
    CURSOR_RADIUS = 0.04

    # Colors
    BG = (-0.1, -0.1, -0.1)
    WHITE = (0.9, 0.9, 0.9)
    DIM = (0.35, 0.35, 0.35)
    LIT = (0.9, 0.9, 0.2)
    CURSOR = (0.2, 0.8, 0.9)

    # Setup comms
    link = BCILink(zmq_cfg.pair_addr)

    
    trig = TriggerPort(ser_cfg.port, ser_cfg.baudrate, ser_cfg.pulse_width_s)
    trig.open()

    # Setup window & stimuli
    win = visual.Window(size=WIN_SIZE, color=BG, units="norm", fullscr=False)

    left_target = visual.Circle(
        win, radius=TARGET_RADIUS, edges=64, pos=(-TARGET_OFFSET_X, 0), fillColor=DIM, lineColor=WHITE
    )
    right_target = visual.Circle(
        win, radius=TARGET_RADIUS, edges=64, pos=(+TARGET_OFFSET_X, 0), fillColor=DIM, lineColor=WHITE
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
            link.drain()
            draw_scene()
            win.flip()
            if "escape" in event.getKeys():
                raise KeyboardInterrupt
        cursor.pos = end

    def reset_cursor(duration: float = 0.25):
        start = cursor.pos
        end = (0, 0)
        clock = core.Clock()
        while clock.getTime() < duration:
            t = clock.getTime() / duration
            cursor.pos = (start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)
            link.drain()
            draw_scene()
            win.flip()
            if "escape" in event.getKeys():
                raise KeyboardInterrupt
        cursor.pos = end

    def wait_for_space():
        while True:
            draw_scene()
            win.flip()
            keys = event.getKeys()
            if "space" in keys:
                return
            if "escape" in keys:
                raise KeyboardInterrupt

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
    while True:
        draw_scene()
        win.flip()
        keys = event.getKeys()
        if "space" in keys:
            break
        if "escape" in keys:
            win.close()
            trig.close()
            link.close()
            return

    # Session handshake
    link.send({"action": "SESSION_START"})
    ready = link.recv(timeout_s=10.0)
    if ready is None or ready.get("status") != "ready":
        cue_text.text = "Worker not responding.\nIs bci_worker.py running?\n\nPress ESC to quit."
        status_text.text = ""
        draw_scene()
        win.flip()
        while True:
            if "escape" in event.getKeys():
                break
        win.close()
        trig.close()
        link.close()
        return

    # ==============================================================
    # PHASE 1: CALIBRATION (normal trials, no feedback)
    # ==============================================================
    cal_scheduler = BalancedBlockScheduler(block_size=max(2, model_cfg.retrain_every), left_code=LEFT, right_code=RIGHT, seed=None)
    for cal_idx in range(N_CAL_TRIALS):
        y_true_code = cal_scheduler.next_code()
        set_targets(None)
        cue_text.text = ""
        status_text.text = f"Calibration Trial {cal_idx + 1}/{N_CAL_TRIALS}"
        reset_cursor(duration=0.15)

        iti_clock = core.Clock()
        while iti_clock.getTime() < ITI:
            link.drain()
            draw_scene()
            win.flip()
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

        set_targets(None)
        cue_text.text = f"Prepare: {code_to_name(y_true_code)}"
        status_text.text = "Get ready..."
        prep_clock = core.Clock()
        while prep_clock.getTime() < PREP_DURATION:
            link.drain()
            draw_scene()
            win.flip()
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

        set_targets(code_to_side(y_true_code))
        cue_text.text = f"IMAGINE: {code_to_name(y_true_code)}"
        status_text.text = "Calibration (no feedback)"
        trig.pulse(int(y_true_code))

        draw_scene()
        win.flip()
        mi_clock = core.Clock()
        while mi_clock.getTime() < MI_DURATION:
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

        set_targets(None)
        cue_text.text = ""
        status_text.text = "Data captured."
        draw_scene()
        win.flip()
        core.wait(0.2)

    # ==============================================================
    # TRAINING REQUEST
    # ==============================================================
    cue_text.text = "Training classifier on calibration data...\nPlease wait."
    status_text.text = ""
    draw_scene()
    win.flip()

    link.send({"action": "TRAIN", "n_trials": N_CAL_TRIALS})

    # Poll in 100ms chunks for escape-key responsiveness
    deadline = time.time() + CAL_TRAIN_TIMEOUT_S
    cal_result = None
    while time.time() < deadline:
        remaining = max(0.0, deadline - time.time())
        cal_result = link.recv(timeout_s=min(0.1, remaining))
        if cal_result is not None:
            break
        if "escape" in event.getKeys():
            raise KeyboardInterrupt

    if cal_result and cal_result.get("status") == "trained":
        cv_mean = cal_result.get("cv_mean", 0)
        cv_std = cal_result.get("cv_std", 0)
        n_epochs = cal_result.get("n_epochs", 0)
        n_per_class = cal_result.get("n_per_class", {str(LEFT): 0, str(RIGHT): 0})
        best_C = cal_result.get("best_C", None)
        cue_text.text = (
            f"Calibration Complete!\n\n"
            f"Cross-validated accuracy: {cv_mean:.1%} +/- {cv_std:.1%}\n"
            f"Epochs: {n_epochs} "
            f"(L={n_per_class.get(str(LEFT), 0)}, R={n_per_class.get(str(RIGHT), 0)})\n"
            f"Selected C: {best_C}\n\n"
            f"Press SPACE to begin online phase."
        )
    elif cal_result and cal_result.get("status") == "error":
        cue_text.text = f"Calibration training error:\n{cal_result.get('message', 'unknown')}\n\nPress SPACE to continue."
    else:
        cue_text.text = "Calibration training timed out.\nPress SPACE to begin online phase."
    status_text.text = ""
    draw_scene()
    win.flip()
    wait_for_space()

    # Enable live predictions
    link.send({"action": "ONLINE_START"})
    ack = link.recv(timeout_s=5.0)
    if ack is None or ack.get("status") != "ack":
        print("[WARN] ONLINE_START not acknowledged by worker.")

    # ==============================================================
    # PHASE 2: ONLINE TRIALS
    # ==============================================================
    scheduler = BalancedBlockScheduler(block_size=max(2, model_cfg.retrain_every), left_code=LEFT, right_code=RIGHT, seed=None)
    correct_count = 0

    for live_idx in range(N_LIVE_TRIALS):
        y_true_code = scheduler.next_code()  # 1=LEFT, 2=RIGHT

        # --- ITI ---
        set_targets(None)
        cue_text.text = ""
        if live_idx > 0:
            status_text.text = f"Live Trial {live_idx + 1}/{N_LIVE_TRIALS} | Accuracy: {correct_count}/{live_idx}"
        else:
            status_text.text = f"Live Trial 1/{N_LIVE_TRIALS}"
        reset_cursor(duration=0.15)

        iti_clock = core.Clock()
        while iti_clock.getTime() < ITI:
            link.drain()
            draw_scene()
            win.flip()
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

        # --- Prepare ---
        set_targets(None)
        cue_text.text = f"Prepare: {code_to_name(y_true_code)}"
        status_text.text = "Get ready..."
        prep_clock = core.Clock()
        while prep_clock.getTime() < PREP_DURATION:
            link.drain()
            draw_scene()
            win.flip()
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

        # --- MI phase ---
        set_targets(code_to_side(y_true_code))
        cue_text.text = f"IMAGINE: {code_to_name(y_true_code)}"
        status_text.text = "Go!"
        trig.pulse(int(y_true_code))

        draw_scene()
        win.flip()
        mi_clock = core.Clock()
        while mi_clock.getTime() < MI_DURATION:
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

        # --- Wait for prediction (FIFO: next message is always this trial) ---
        payload = link.recv(timeout_s=PRED_WAIT_TIMEOUT_S)

        if payload is None:
            y_pred_code = -1
            conf = 0.0
        elif payload.get("rejected", False):
            y_pred_code = -1
            conf = 0.0
        else:
            y_pred_code = int(payload.get("y_pred_code", -1))
            conf = float(payload.get("conf", 0.0))
            # Alignment check
            y_true_from_eeg = payload.get("y_true_code")
            if y_true_from_eeg is not None and int(y_true_from_eeg) != y_true_code:
                print(f"[WARN] FIFO misalignment: psychopy code={y_true_code}, EEG code={y_true_from_eeg}")

        # --- Feedback ---
        cue_text.text = ""
        if y_pred_code in (LEFT, RIGHT):
            is_correct = (y_pred_code == y_true_code)

            # Send ErrP marker at the instant cursor begins to move.
            trig.pulse(CORRECT if is_correct else ERROR)

            status_text.text = f"Pred: {code_to_name(y_pred_code)} | conf={conf:.2f}"
            move_cursor_to(code_to_side(y_pred_code), duration=0.5)

            if is_correct:
                correct_count += 1
        else:
            status_text.text = "No prediction received."

        hold = core.Clock()
        while hold.getTime() < 0.4:
            link.drain()
            draw_scene()
            win.flip()
            if "escape" in event.getKeys():
                raise KeyboardInterrupt
        set_targets(None)

    # End session
    link.send({"action": "SESSION_STOP"})

    # ==============================================================
    # SESSION COMPLETE
    # ==============================================================
    cue_text.text = (
        f"Session complete.\n"
        f"Online accuracy: {correct_count}/{N_LIVE_TRIALS}\n\n"
        f"Stop the worker (Ctrl+C) to save data\n"
        f"and see final cross-validated accuracy.\n"
        f"Press ESC to close."
    )
    status_text.text = ""
    while True:
        draw_scene()
        win.flip()
        if "escape" in event.getKeys():
            break

    win.close()
    trig.close()
    link.close()


if __name__ == "__main__":
    try:
        run_task()
    except KeyboardInterrupt:
        pass
