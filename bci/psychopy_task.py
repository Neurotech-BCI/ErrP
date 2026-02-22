# psychopy_task.py
import json
import random
import time
import serial

import zmq
from psychopy import visual, core, event

from config import (
    ZMQConfig, SerialConfig, ModelConfig, EEGConfig, CalibrationConfig,
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
        self.last_cal = None      # latest CAL payload

    def close(self):
        try:
            self.sub.close(0)
        except Exception:
            pass

    def _recv_one(self) -> tuple[str, dict]:
        """Receive and parse one message. Returns (topic, payload)."""
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

    def wait_for_trial(self, trial_index: int, timeout_s: float):
        """Block until PRED payload with trial_index >= target arrives."""
        deadline = time.time() + timeout_s
        best = None

        if self.last_payload is not None:
            if int(self.last_payload.get("trial_index", -1)) >= trial_index:
                return self.last_payload

        while time.time() < deadline:
            remaining = max(0.0, deadline - time.time())
            timeout_ms = int(min(50, remaining * 1000))
            latest = self.poll_latest(timeout_ms=timeout_ms)
            if latest is not None:
                best = latest
                if int(latest.get("trial_index", -1)) >= trial_index:
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
    """Generates labels in balanced blocks (for online trials)."""

    def __init__(self, block_size: int, seed: int | None = None):
        if block_size % 2 != 0:
            raise ValueError("block_size must be even to balance 0/1.")
        self.block_size = block_size
        self.rng = random.Random(seed)
        self._bag: list[int] = []

    def _refill(self):
        half = self.block_size // 2
        self._bag = [0] * half + [1] * half
        self.rng.shuffle(self._bag)

    def next_label(self) -> int:
        if not self._bag:
            self._refill()
        return self._bag.pop()


class CalibrationScheduler:
    """Balanced L/R calibration schedule with retry on reject."""

    def __init__(self, phases_per_class: int, seed: int | None = None):
        rng = random.Random(seed)
        schedule = [0] * phases_per_class + [1] * phases_per_class
        rng.shuffle(schedule)
        self._queue = list(schedule)

    def has_next(self) -> bool:
        return len(self._queue) > 0

    def peek_next(self) -> int:
        return self._queue[0]

    def accept(self):
        self._queue.pop(0)

    def reject(self):
        pass  # same label stays at front for retry

    @property
    def remaining(self) -> int:
        return len(self._queue)


# ---------------- PsychoPy task ----------------

def run_task():
    zmq_cfg = ZMQConfig()
    ser_cfg = SerialConfig()
    model_cfg = ModelConfig()
    eeg_cfg = EEGConfig()
    cal_cfg = CalibrationConfig()

    # Timing
    PREP_DURATION = 2.0
    MI_DURATION = eeg_cfg.tmax - eeg_cfg.tmin  # online epoch duration
    ITI = 3.0
    N_TRIALS = 80
    PRED_WAIT_TIMEOUT_S = 5.0
    CAL_WAIT_TIMEOUT_S = cal_cfg.mi_duration_s + 5.0

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
        win, radius=TARGET_RADIUS, edges=64,
        pos=(-TARGET_OFFSET_X, 0), fillColor=DIM, lineColor=WHITE,
    )
    right_target = visual.Circle(
        win, radius=TARGET_RADIUS, edges=64,
        pos=(+TARGET_OFFSET_X, 0), fillColor=DIM, lineColor=WHITE,
    )
    cursor = visual.Circle(
        win, radius=CURSOR_RADIUS, edges=64,
        pos=(0, 0), fillColor=CURSOR, lineColor=WHITE,
    )
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
            cursor.pos = (
                start[0] + (end[0] - start[0]) * t,
                start[1] + (end[1] - start[1]) * t,
            )
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
            cursor.pos = (
                start[0] + (end[0] - start[0]) * t,
                start[1] + (end[1] - start[1]) * t,
            )
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
        f"Sustain LEFT/RIGHT imagery for {cal_cfg.mi_duration_s:.0f}s blocks.\n"
        "After each block, press Y to keep or N to retry.\n\n"
        "Phase 2: Online feedback\n"
        "Short MI trials with cursor feedback.\n\n"
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

    # ==============================================================
    # PHASE 1: CALIBRATION
    # ==============================================================
    cal_sched = CalibrationScheduler(cal_cfg.phases_per_class, seed=None)
    total_phases = cal_cfg.phases_per_class * 2
    completed_phases = 0

    while cal_sched.has_next():
        label = cal_sched.peek_next()
        class_name = "LEFT" if label == 0 else "RIGHT"

        # --- Prep ---
        set_targets(None)
        cursor.pos = (0, 0)
        cue_text.text = (
            f"Calibration Block {completed_phases + 1} / {total_phases}\n"
            f"Prepare: {class_name}"
        )
        status_text.text = f"Sustain imagery for {cal_cfg.mi_duration_s:.0f}s when cued."
        prep_clock = core.Clock()
        while prep_clock.getTime() < cal_cfg.prep_duration_s:
            draw_scene()
            win.flip()
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

        # --- MI onset ---
        set_targets(label)
        trig.pulse(label + 1)
        ctrl_tx.send("CAL_START", label=label)

        # --- Countdown ---
        mi_clock = core.Clock()
        while mi_clock.getTime() < cal_cfg.mi_duration_s:
            remaining = cal_cfg.mi_duration_s - mi_clock.getTime()
            cue_text.text = f"IMAGINE: {class_name}"
            status_text.text = f"Time remaining: {remaining:.1f}s"
            draw_scene()
            win.flip()
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

        # --- MI end ---
        set_targets(None)
        cue_text.text = "STOP"
        status_text.text = "Collecting data..."
        draw_scene()
        win.flip()

        # Wait for worker to confirm collection
        cal_status = pred_rx.wait_for_cal_status(
            "collected", timeout_s=CAL_WAIT_TIMEOUT_S
        )
        if cal_status is None:
            cue_text.text = "Worker did not respond.\nPress SPACE to retry."
            status_text.text = ""
            draw_scene()
            win.flip()
            wait_for_space()
            continue

        # --- User accept / reject ---
        cue_text.text = (
            f"Block complete ({class_name})\n"
            f"Did you maintain good imagery?"
        )
        status_text.text = "Y = Keep  /  N = Retry"
        draw_scene()
        win.flip()

        decision = None
        while decision is None:
            keys = event.getKeys()
            if "y" in keys:
                decision = "keep"
            elif "n" in keys:
                decision = "reject"
            elif "escape" in keys:
                raise KeyboardInterrupt
            draw_scene()
            win.flip()

        if decision == "keep":
            ctrl_tx.send("CAL_KEEP")
            cal_sched.accept()
            completed_phases += 1

            kept_status = pred_rx.wait_for_cal_status("kept", timeout_s=5.0)
            if kept_status:
                n_kept = kept_status.get("n_kept", "?")
                n_total = kept_status.get("n_total", "?")
                total_eps = kept_status.get("cal_epochs_so_far", "?")
                cue_text.text = (
                    f"Kept {n_kept}/{n_total} epochs (artifact check)\n"
                    f"Total calibration epochs: {total_eps}"
                )
            else:
                cue_text.text = "Block kept."
            status_text.text = "Press SPACE to continue."
            draw_scene()
            win.flip()
            wait_for_space()
        else:
            ctrl_tx.send("CAL_REJECT")
            cal_sched.reject()
            cue_text.text = f"Block rejected. Will retry {class_name}."
            status_text.text = "Press SPACE to continue."
            draw_scene()
            win.flip()
            wait_for_space()

    # --- Signal calibration complete ---
    ctrl_tx.send("CAL_DONE")

    cue_text.text = "Training classifier on calibration data...\nPlease wait."
    status_text.text = ""
    draw_scene()
    win.flip()

    trained_status = pred_rx.wait_for_cal_status("trained", timeout_s=30.0)
    if trained_status:
        cv_mean = trained_status.get("cv_mean", 0)
        cv_std = trained_status.get("cv_std", 0)
        n_epochs = trained_status.get("n_epochs", 0)
        n_per_class = trained_status.get("n_per_class", [0, 0])
        cue_text.text = (
            f"Calibration Complete!\n\n"
            f"Cross-validated accuracy: {cv_mean:.1%} +/- {cv_std:.1%}\n"
            f"Epochs: {n_epochs} ({n_per_class[0]} left, {n_per_class[1]} right)\n\n"
            f"Press SPACE to begin online phase."
        )
    else:
        cue_text.text = "Calibration complete.\nPress SPACE to begin online phase."
    status_text.text = ""
    draw_scene()
    win.flip()
    wait_for_space()

    # ==============================================================
    # PHASE 2: ONLINE TRIALS
    # ==============================================================
    scheduler = BalancedBlockScheduler(block_size=model_cfg.retrain_every, seed=None)
    correct = 0

    for t_idx in range(N_TRIALS):
        pred_rx.poll_latest(0)
        pred_rx.last_payload = None

        y_true = scheduler.next_label()

        # --- ITI ---
        set_targets(None)
        cue_text.text = ""
        if t_idx > 0:
            status_text.text = (
                f"Trial {t_idx+1}/{N_TRIALS} | Accuracy: {correct}/{t_idx}"
            )
        else:
            status_text.text = f"Trial 1/{N_TRIALS}"
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
        cue_text.text = "Prepare: LEFT" if y_true == 0 else "Prepare: RIGHT"
        status_text.text = "Get ready..."
        prep_clock = core.Clock()
        while prep_clock.getTime() < PREP_DURATION:
            pred_rx.poll_latest(0)
            draw_scene()
            win.flip()
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

        # --- MI phase ---
        set_targets(y_true)
        cue_text.text = "IMAGINE: LEFT" if y_true == 0 else "IMAGINE: RIGHT"
        status_text.text = "Go!"
        trig.pulse(y_true + 1)
        draw_scene()
        win.flip()
        mi_clock = core.Clock()
        while mi_clock.getTime() < MI_DURATION:
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

        # --- Wait for prediction ---
        payload = pred_rx.wait_for_trial(
            trial_index=t_idx, timeout_s=PRED_WAIT_TIMEOUT_S
        )

        if payload is None or int(payload.get("trial_index", -1)) < t_idx:
            y_pred = -1
            conf = 0.0
        else:
            y_pred = int(payload.get("y_pred", -1))
            conf = float(payload.get("conf", 0.0))

        # --- Feedback ---
        cue_text.text = ""
        if y_pred >= 0:
            status_text.text = (
                f"Pred: {'LEFT' if y_pred==0 else 'RIGHT'} | conf={conf:.2f}"
            )
            move_cursor_to(y_pred, duration=0.5)
        else:
            status_text.text = "No prediction received."

        if y_pred == y_true:
            correct += 1

        hold = core.Clock()
        while hold.getTime() < 0.4:
            pred_rx.poll_latest(0)
            draw_scene()
            win.flip()
            if "escape" in event.getKeys():
                raise KeyboardInterrupt
        set_targets(None)

    # ==============================================================
    # SESSION COMPLETE
    # ==============================================================
    cue_text.text = (
        f"Session complete.\n"
        f"Online accuracy: {correct}/{N_TRIALS}\n\n"
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
