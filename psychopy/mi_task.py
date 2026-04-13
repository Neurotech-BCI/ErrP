"""Offline motor-imagery calibration task for trigger-only data collection.

This script mirrors the calibration phase from bci/psychopy_task.py but removes
all live EEG/LSL processing. It only presents stimuli and sends LEFT/RIGHT
trigger pulses to the trigger hub.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import argparse
import random
import time

import serial
import serial.tools.list_ports
from psychopy import core, event, visual


# ----------------------------------------------------------------
# Embedded config
# ----------------------------------------------------------------


@dataclass(frozen=True)
class StimConfig:
    left_code: int = 1
    right_code: int = 2


@dataclass(frozen=True)
class TaskConfig:
    n_calibration_trials: int = 50
    max_trials_before_break: int = 20

    prep_duration_s: float = 3.0
    execution_duration_s: float = 3.0
    iti_duration_s: float = 3.0

    fullscreen: bool = False
    win_size: tuple[int, int] = (1200, 700)


@dataclass(frozen=True)
class SerialConfig:
    vid: int = 0x2341
    pid: int = 0x8037
    baudrate: int = 115200
    pulse_width_s: float = 0.01


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


def find_port_by_vid_pid(vid: int, pid: int) -> str:
    for p in serial.tools.list_ports.comports():
        if p.vid == vid and p.pid == pid:
            return p.device
    raise RuntimeError(
        f"No trigger hub found for VID:PID {vid:#06x}:{pid:#06x}. "
        "Connect device or pass --port explicitly."
    )


def sanitize_participant_name(raw_name: str) -> str:
    cleaned = "_".join(raw_name.strip().lower().split())
    cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch == "_")
    return cleaned.strip("_")


def build_session_prefix(participant: str) -> str:
    date_prefix = datetime.now().strftime("%m_%d_%y")
    return f"{date_prefix}_{participant}_offline_cal"


_ARROW_VERTS_RIGHT = [
    (-0.25, 0.04), (0.05, 0.04), (0.05, 0.10),
    (0.25, 0.0),
    (0.05, -0.10), (0.05, -0.04), (-0.25, -0.04),
]
_ARROW_VERTS_LEFT = [(-x, y) for x, y in _ARROW_VERTS_RIGHT]


def run_task(session_name: str, port: str, task_cfg: TaskConfig, stim_cfg: StimConfig, ser_cfg: SerialConfig) -> None:
    left = int(stim_cfg.left_code)
    right = int(stim_cfg.right_code)

    def code_to_name(code: int) -> str:
        return "LEFT" if int(code) == left else "RIGHT"

    trig = TriggerPort(port=port, baudrate=ser_cfg.baudrate, pulse_width_s=ser_cfg.pulse_width_s)
    trig.open()

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
            draw_scene()
            win.flip()
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

    def wait_for_space() -> None:
        while True:
            draw_scene()
            win.flip()
            keys = event.getKeys()
            if "space" in keys:
                return
            if "escape" in keys:
                raise KeyboardInterrupt

    scheduler = BalancedBlockScheduler(
        block_size=max(2, task_cfg.max_trials_before_break),
        left_code=left,
        right_code=right,
    )

    print(f"[SESSION] {session_name}")
    print(f"[SERIAL] Sending triggers on {port} @ {ser_cfg.baudrate}")

    try:
        cue_text.text = (
            "Offline Motor Imagery Calibration\n\n"
            "Fixate on the center cross.\n"
            "An arrow will cue LEFT or RIGHT preparation.\n"
            "Then execute the cued motor imagery when prompted.\n\n"
            "Press SPACE to begin. ESC to quit."
        )
        status_text.text = ""
        fixation.text = "+"
        wait_for_space()

        for cal_idx in range(task_cfg.n_calibration_trials):
            y_true = scheduler.next_code()

            # ITI
            prep_arrow.opacity = 0
            cue_text.text = ""
            status_text.text = f"Calibration Trial {cal_idx + 1}/{task_cfg.n_calibration_trials}"
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

            # Breaks
            if (
                (cal_idx + 1) % task_cfg.max_trials_before_break == 0
                and (cal_idx + 1) < task_cfg.n_calibration_trials
            ):
                cue_text.text = "Break\n\nPress SPACE to continue"
                status_text.text = f"Completed {cal_idx + 1}/{task_cfg.n_calibration_trials} trials"
                prep_arrow.opacity = 0
                fixation.text = "+"
                wait_for_space()

        cue_text.text = "Calibration complete\n\nPress ESC to close"
        status_text.text = ""
        prep_arrow.opacity = 0
        fixation.text = "+"
        while "escape" not in event.getKeys():
            draw_scene()
            win.flip()

    except KeyboardInterrupt:
        print("\n[SESSION] Interrupted by user")

    finally:
        trig.close()
        try:
            win.close()
        except Exception:
            pass
        core.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline MI calibration trigger task")
    parser.add_argument("--participant", type=str, default=None, help="Participant name for session label")
    parser.add_argument("--port", type=str, default=None, help="Serial port override (e.g., COM6 or /dev/ttyUSB0)")
    parser.add_argument("--trials", type=int, default=None, help="Override number of calibration trials")
    parser.add_argument("--fullscreen", action="store_true", help="Run fullscreen")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stim_cfg = StimConfig()
    task_cfg = TaskConfig(
        n_calibration_trials=args.trials if args.trials is not None else TaskConfig.n_calibration_trials,
        fullscreen=bool(args.fullscreen),
    )
    ser_cfg = SerialConfig()

    participant = sanitize_participant_name(args.participant or input("Enter participant name: "))
    if not participant:
        raise ValueError("Participant name cannot be empty")

    session_name = build_session_prefix(participant)
    port = args.port or find_port_by_vid_pid(ser_cfg.vid, ser_cfg.pid)

    run_task(
        session_name=session_name,
        port=port,
        task_cfg=task_cfg,
        stim_cfg=stim_cfg,
        ser_cfg=ser_cfg,
    )


if __name__ == "__main__":
    main()
