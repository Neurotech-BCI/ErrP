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


class PredictionReceiver:
    """ZMQ SUB that handles both PRED and CAL topics."""

    def __init__(self, addr: str, pred_topic: str, cal_topic: str):
        self.pred_topic = pred_topic
        self.cal_topic = cal_topic
        self.ctx = zmq.Context.instance()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(addr)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, pred_topic)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, cal_topic)

        self.poller = zmq.Poller()
        self.poller.register(self.sub, zmq.POLLIN)

        self.last_payload = None  # latest PRED payload
        self.last_cal = None  # latest CAL payload

    def close(self):
        try:
            self.sub.close(0)
        except Exception:
            pass

    def _recv_one(self) -> tuple[str, dict]:
        msg = self.sub.recv_string()
        topic, j = msg.split(" ", 1)
        return topic, json.loads(j)

    def poll_latest(self, timeout_ms: int = 0):
        """Drain queue, keep latest PRED payload. CAL payloads stashed in last_cal."""
        latest_pred = None
        socks = dict(self.poller.poll(timeout_ms))
        while socks.get(self.sub) == zmq.POLLIN:
            topic, payload = self._recv_one()
            if topic == self.pred_topic:
                latest_pred = payload
            elif topic == self.cal_topic:
                self.last_cal = payload
            socks = dict(self.poller.poll(0))
        if latest_pred is not None:
            self.last_payload = latest_pred
        return latest_pred

    def wait_for_cal_status(self, target_status: str, timeout_s: float) -> dict | None:
        """Block until a CAL message with matching status arrives."""
        deadline = time.time() + timeout_s
        self.last_cal = None
        while time.time() < deadline:
            remaining = max(0.0, deadline - time.time())
            timeout_ms = int(min(100, remaining * 1000))
            self.poll_latest(timeout_ms=timeout_ms)
            if self.last_cal is not None and self.last_cal.get("status") == target_status:
                result = self.last_cal
                self.last_cal = None
                return result
            if "escape" in event.getKeys():
                raise KeyboardInterrupt
        return None

    def wait_for_cal_statuses(self, target_statuses: set[str], timeout_s: float) -> dict | None:
        """Block until a CAL message with a status in target_statuses arrives."""
        deadline = time.time() + timeout_s
        self.last_cal = None
        while time.time() < deadline:
            remaining = max(0.0, deadline - time.time())
            timeout_ms = int(min(100, remaining * 1000))
            self.poll_latest(timeout_ms=timeout_ms)
            if self.last_cal is not None and self.last_cal.get("status") in target_statuses:
                result = self.last_cal
                self.last_cal = None
                return result
            if "escape" in event.getKeys():
                raise KeyboardInterrupt
        return None

    def wait_for_trial_id(self, trial_id: int, timeout_s: float) -> dict | None:
        """Block until a PRED payload with matching trial_id arrives."""
        deadline = time.time() + timeout_s

        if self.last_payload is not None and int(self.last_payload.get("trial_id", -1)) == trial_id:
            return self.last_payload

        best = None
        while time.time() < deadline:
            remaining = max(0.0, deadline - time.time())
            timeout_ms = int(min(50, remaining * 1000))
            latest = self.poll_latest(timeout_ms=timeout_ms)
            if latest is not None:
                best = latest
                if int(latest.get("trial_id", -1)) == trial_id:
                    return latest
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

        return best


class ControlSender:
    """ZMQ PUSH socket to send control commands to the worker."""

    def __init__(self, addr: str):
        self.ctx = zmq.Context.instance()
        self.push = self.ctx.socket(zmq.PUSH)
        self.push.connect(addr)

    def send(self, action: str, **kwargs):
        payload = {"action": action, **kwargs}
        self.push.send_string(json.dumps(payload))

    def close(self):
        try:
            self.push.close(0)
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
    N_LIVE_TRIALS = 60
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
    pred_rx = PredictionReceiver(zmq_cfg.pub_addr, zmq_cfg.topic, zmq_cfg.cal_topic)
    ctrl_tx = ControlSender(zmq_cfg.ctrl_addr)

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
            pred_rx.poll_latest(0)
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
            pred_rx.poll_latest(0)
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
            pred_rx.close()
            ctrl_tx.close()
            return

    ctrl_tx.send("SESSION_START", t_task=time.time())

    # ==============================================================
    # PHASE 1: CALIBRATION (normal trials, no feedback)
    # ==============================================================
    trial_id = 0
    cal_scheduler = BalancedBlockScheduler(block_size=max(2, model_cfg.retrain_every), left_code=LEFT, right_code=RIGHT, seed=None)
    for cal_idx in range(N_CAL_TRIALS):
        pred_rx.poll_latest(0)
        pred_rx.last_payload = None

        y_true_code = cal_scheduler.next_code()
        set_targets(None)
        cue_text.text = ""
        status_text.text = f"Calibration Trial {cal_idx + 1}/{N_CAL_TRIALS}"
        reset_cursor(duration=0.15)

        iti_clock = core.Clock()
        while iti_clock.getTime() < ITI:
            pred_rx.poll_latest(0)
            draw_scene()
            win.flip()
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

        set_targets(None)
        cue_text.text = f"Prepare: {code_to_name(y_true_code)}"
        status_text.text = "Get ready..."
        prep_clock = core.Clock()
        while prep_clock.getTime() < PREP_DURATION:
            pred_rx.poll_latest(0)
            draw_scene()
            win.flip()
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

        set_targets(code_to_side(y_true_code))
        cue_text.text = f"IMAGINE: {code_to_name(y_true_code)}"
        status_text.text = "Calibration (no feedback)"
        ctrl_tx.send("TRIAL_START", trial_id=int(trial_id), code=int(y_true_code), t_task=time.time())
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
        trial_id += 1

    cue_text.text = "Training classifier on calibration data...\nPlease wait."
    status_text.text = ""
    draw_scene()
    win.flip()

    cal_result = pred_rx.wait_for_cal_statuses({"trained", "error"}, timeout_s=CAL_TRAIN_TIMEOUT_S)
    if cal_result and cal_result.get("status") == "trained":
        trained_status = cal_result
        cv_mean = trained_status.get("cv_mean", 0)
        cv_std = trained_status.get("cv_std", 0)
        n_epochs = trained_status.get("n_epochs", 0)
        n_per_class = trained_status.get("n_per_class", {str(LEFT): 0, str(RIGHT): 0})
        best_C = trained_status.get("best_C", None)
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

    # Enable live predictions. Raw capture is already active from SESSION_START.
    ctrl_tx.send("ONLINE_START", t_task=time.time())

    # ==============================================================
    # PHASE 2: ONLINE TRIALS
    # ==============================================================
    scheduler = BalancedBlockScheduler(block_size=max(2, model_cfg.retrain_every), left_code=LEFT, right_code=RIGHT, seed=None)
    correct_count = 0

    for live_idx in range(N_LIVE_TRIALS):
        pred_rx.poll_latest(0)
        pred_rx.last_payload = None

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
            pred_rx.poll_latest(0)
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
            pred_rx.poll_latest(0)
            draw_scene()
            win.flip()
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

        # --- MI phase ---
        set_targets(code_to_side(y_true_code))
        cue_text.text = f"IMAGINE: {code_to_name(y_true_code)}"
        status_text.text = "Go!"

        # Handshake FIRST (anti-drift), then trigger (1/2)
        ctrl_tx.send("TRIAL_START", trial_id=int(trial_id), code=int(y_true_code), t_task=time.time())
        trig.pulse(int(y_true_code))

        draw_scene()
        win.flip()
        mi_clock = core.Clock()
        while mi_clock.getTime() < MI_DURATION:
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

        # --- Wait for prediction for THIS trial_id ---
        payload = pred_rx.wait_for_trial_id(trial_id=int(trial_id), timeout_s=PRED_WAIT_TIMEOUT_S)

        if payload is None or int(payload.get("trial_id", -1)) != int(trial_id):
            y_pred_code = -1
            conf = 0.0
        else:
            y_pred_code = int(payload.get("y_pred_code", -1))
            conf = float(payload.get("conf", 0.0))

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
            pred_rx.poll_latest(0)
            draw_scene()
            win.flip()
            if "escape" in event.getKeys():
                raise KeyboardInterrupt
        set_targets(None)
        trial_id += 1

    # End raw capture window (worker will close CSV, but keep running)
    ctrl_tx.send("ONLINE_STOP", t_task=time.time())

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
    pred_rx.close()
    ctrl_tx.close()


if __name__ == "__main__":
    try:
        run_task()
    except KeyboardInterrupt:
        pass
