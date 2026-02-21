# psychopy_task.py
import json
import random
import time
import serial

import zmq
from psychopy import visual, core, event

from config import ZMQConfig, SerialConfig, ModelConfig, EEGConfig


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
        """
        Send a brief trigger pulse.
        Adjust to your hub protocol if it expects multi-byte frames.
        """
        if self.ser is None:
            return
        code = int(code) & 0xFF
        self.ser.write(bytes([code]))
        self.ser.flush()
        core.wait(self.pulse_width_s)
        self.ser.write(bytes([0]))
        self.ser.flush()


# ---------------- ZMQ SUB (receive predictions) ----------------

class PredictionReceiver:
    def __init__(self, addr: str, topic: str):
        self.addr = addr
        self.topic = topic
        self.ctx = zmq.Context.instance()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(addr)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, topic)

        self.poller = zmq.Poller()
        self.poller.register(self.sub, zmq.POLLIN)

        self.last_payload = None

    def close(self):
        try:
            self.sub.close(0)
        except Exception:
            pass

    def poll_latest(self, timeout_ms: int = 0):
        """
        Non-blocking (or short-block if timeout_ms>0):
        drain queue and keep only the newest payload.
        """
        latest = None
        socks = dict(self.poller.poll(timeout_ms))
        while socks.get(self.sub) == zmq.POLLIN:
            msg = self.sub.recv_string()
            _, j = msg.split(" ", 1)
            latest = json.loads(j)
            socks = dict(self.poller.poll(0))
        if latest is not None:
            self.last_payload = latest
        return latest

    def wait_for_trial(self, trial_index: int, timeout_s: float):
        """
        Block until we receive a payload with payload['trial_index'] >= trial_index,
        or until timeout_s elapses. Returns payload or None.
        """
        deadline = time.time() + timeout_s
        best = None

        # First, if we've already got it, return immediately.
        if self.last_payload is not None:
            if int(self.last_payload.get("trial_index", -1)) >= trial_index:
                return self.last_payload

        # Otherwise, block-poll in small increments and drain
        while time.time() < deadline:
            remaining = max(0.0, deadline - time.time())
            timeout_ms = int(min(50, remaining * 1000))  # 50ms chunks
            latest = self.poll_latest(timeout_ms=timeout_ms)
            if latest is not None:
                best = latest
                if int(latest.get("trial_index", -1)) >= trial_index:
                    return latest

            if "escape" in event.getKeys():
                raise KeyboardInterrupt

        return best


# ---------------- Balanced trial scheduler ----------------

class BalancedBlockScheduler:
    """
    Generates labels in balanced blocks of size block_size.
    For block_size=4 -> two 0s and two 1s, shuffled each block.
    """
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


# ---------------- PsychoPy task ----------------

def run_task():
    zmq_cfg = ZMQConfig()
    ser_cfg = SerialConfig()
    model_cfg = ModelConfig()
    eeg_cfg = EEGConfig()
    # Timing
    PREP_DURATION = 2.0
    MI_DURATION = eeg_cfg.tmax
    ITI = 3.0

    # Use enough trials for your session (can be >= window size, etc.)
    N_TRIALS = 80

    # How long to wait after MI ends for the worker to publish the trial prediction.
    # Should be > typical compute + publish latency.
    PRED_WAIT_TIMEOUT_S = 5.0

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

    # Balanced label scheduler: use retrain_every as the block size (e.g. 4)
    scheduler = BalancedBlockScheduler(block_size=model_cfg.retrain_every, seed=None)

    # Setup comms
    pred_rx = PredictionReceiver(zmq_cfg.pub_addr, zmq_cfg.topic)

    trig = TriggerPort(ser_cfg.port, ser_cfg.baudrate, ser_cfg.pulse_width_s)
    trig.open()

    # Setup window & stimuli
    win = visual.Window(size=WIN_SIZE, color=BG, units="norm", fullscr=False)

    left_target = visual.Circle(win, radius=TARGET_RADIUS, edges=64, pos=(-TARGET_OFFSET_X, 0), fillColor=DIM, lineColor=WHITE)
    right_target = visual.Circle(win, radius=TARGET_RADIUS, edges=64, pos=(+TARGET_OFFSET_X, 0), fillColor=DIM, lineColor=WHITE)
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
            pred_rx.poll_latest(0)  # keep draining
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

    # Instruction screen
    cue_text.text = "Motor Imagery BCI\n\nImagine moving LEFT or RIGHT when cued.\nPress SPACE to start. ESC to quit."
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
            return

    # Trial loop
    correct = 0
    for t_idx in range(N_TRIALS):
        # Drain any old predictions so we don’t accidentally use stale ones
        pred_rx.poll_latest(0)
        pred_rx.last_payload = None

        # Balanced block label
        y_true = scheduler.next_label()

        # --- ITI / reset ---
        set_targets(None)
        cue_text.text = ""
        status_text.text = f"Trial {t_idx+1}/{N_TRIALS} | Accuracy: {correct}/{max(1,t_idx)}"
        reset_cursor(duration=0.15)

        iti_clock = core.Clock()
        while iti_clock.getTime() < ITI:
            pred_rx.poll_latest(0)
            draw_scene()
            win.flip()
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

        # --- Prepare cue ---
        set_targets(None)
        cue_text.text = "Prepare: LEFT" if y_true == 0 else "Prepare: RIGHT"
        status_text.text = "Get ready…"
        prep_clock = core.Clock()
        while prep_clock.getTime() < PREP_DURATION:
            pred_rx.poll_latest(0)
            draw_scene()
            win.flip()
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

        # --- Execute MI phase ---
        set_targets(y_true)
        cue_text.text = "IMAGINE: LEFT" if y_true == 0 else "IMAGINE: RIGHT"
        status_text.text = "Go!"

        # Send trigger exactly at MI onset
        trig.pulse(y_true + 1)
        draw_scene()
        win.flip()
        mi_clock = core.Clock()
        while mi_clock.getTime() < MI_DURATION:
            # We can display *live* updates if any arrive (but final decision is post-wait)
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

        # --- Wait for the prediction for THIS trial index ---
        # The worker increments trial_index per epoch, starting at 0.
        payload = pred_rx.wait_for_trial(trial_index=t_idx, timeout_s=PRED_WAIT_TIMEOUT_S)

        if payload is None or int(payload.get("trial_index", -1)) < t_idx:
            # This should be rare; treat as "no prediction"
            y_pred = random.randint(0, 1)
            conf = 0.5
            trained = False
            status_text.text = "No model response (timeout). Using random feedback."
        else:
            y_pred = int(payload.get("y_pred", random.randint(0, 1)))
            conf = float(payload.get("conf", 0.5))
            trained = bool(payload.get("trained", False))
            status_text.text = f"Pred: {('LEFT' if y_pred==0 else 'RIGHT')} | conf={conf:.2f} | trained={trained}"

        # Move cursor to predicted side
        cue_text.text = ""
        move_cursor_to(y_pred, duration=0.5)

        # Scoring (optional, for display)
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

    # Done
    cue_text.text = f"Done.\nFinal accuracy: {correct}/{N_TRIALS}\nPress ESC."
    status_text.text = ""
    while True:
        draw_scene()
        win.flip()
        if "escape" in event.getKeys():
            break

    win.close()
    trig.close()
    pred_rx.close()


if __name__ == "__main__":
    try:
        run_task()
    except KeyboardInterrupt:
        pass