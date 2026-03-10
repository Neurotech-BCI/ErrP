"""
Motor Imagery Protocol V2 (PsychoPy)

Implements a leakage-safe, block-structured MI task with explicit Neutral/Left/Right
conditions, kinesthetic imagery instructions, jittered ITI, and per-block accept/reject.

Run:
  python psychopy/mentalCommandTaskV2.py --participant derick

Optional EEG trigger serial:
  python psychopy/mentalCommandTaskV2.py --participant derick --port COM8
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime

# Lazy-loaded PsychoPy modules to avoid import-time failures in non-GUI/test envs.
core = None
event = None
visual = None


def load_psychopy_modules():
    global core, event, visual
    if core is not None and event is not None and visual is not None:
        return

    # Avoid local folder shadowing when launched as:
    #   python psychopy/mentalCommandTaskV2.py
    this_dir = os.path.dirname(os.path.abspath(__file__))
    if this_dir in sys.path:
        sys.path.remove(this_dir)

    from psychopy import core as _core, event as _event, visual as _visual

    core = _core
    event = _event
    visual = _visual

try:
    import serial
except Exception:
    serial = None


@dataclass
class TriggerCodes:
    neutral: int = 10
    left: int = 11
    right: int = 12
    block_start: int = 20
    block_accept: int = 21
    block_reject: int = 22


@dataclass
class ProtocolConfig:
    # Block counts
    neutral_blocks: int = 12
    left_blocks: int = 12
    right_blocks: int = 12

    # Timing (seconds)
    prep_s: float = 1.5
    baseline_s: float = 2.0
    cue_s: float = 4.0
    iti_min_s: float = 1.5
    iti_max_s: float = 2.5

    # Breaks
    break_every_blocks: int = 6
    break_s: int = 45


def sanitize_name(name: str) -> str:
    cleaned = "_".join(name.strip().lower().split())
    cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch == "_")
    return cleaned.strip("_") or "participant"


class TriggerOut:
    def __init__(self, port: str | None, baud: int = 115200):
        self._ser = None
        self.enabled = False
        if not port:
            return
        if serial is None:
            print("[WARN] pyserial not available; running without trigger output.")
            return
        try:
            self._ser = serial.Serial(port=port, baudrate=baud, timeout=0.5)
            self.enabled = True
            print(f"[OK] Trigger serial connected on {port}")
        except Exception as e:
            print(f"[WARN] Could not open serial port {port}: {e}. Continuing without triggers.")

    def send(self, code: int):
        if not self.enabled or self._ser is None:
            return
        try:
            self._ser.write(bytes([int(code) & 0xFF]))
        except Exception:
            pass

    def close(self):
        try:
            if self._ser is not None:
                self._ser.close()
        except Exception:
            pass


class EventLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow([
            "ts_unix",
            "event",
            "participant",
            "session_id",
            "block_id",
            "class_name",
            "class_code",
            "attempt",
            "duration_s",
            "notes",
        ])

    def log(
        self,
        event_name: str,
        participant: str,
        session_id: str,
        block_id: str = "",
        class_name: str = "",
        class_code: int | str = "",
        attempt: int | str = "",
        duration_s: float | str = "",
        notes: str = "",
    ):
        self._writer.writerow([
            f"{time.time():.6f}",
            event_name,
            participant,
            session_id,
            block_id,
            class_name,
            class_code,
            attempt,
            duration_s,
            notes,
        ])
        self._fh.flush()

    def close(self):
        self._fh.close()


def wait_with_escape(seconds: float):
    timer = core.Clock()
    while timer.getTime() < seconds:
        if "escape" in event.getKeys():
            raise KeyboardInterrupt
        core.wait(0.01)


def draw_main(win, title, cue, status):
    title.draw()
    cue.draw()
    status.draw()
    win.flip()


def wait_for_key(valid_keys: set[str]) -> str:
    while True:
        keys = event.getKeys()
        if "escape" in keys:
            raise KeyboardInterrupt
        for k in keys:
            if k in valid_keys:
                return k
        core.wait(0.01)


def run_protocol(participant: str, cfg: ProtocolConfig, trig: TriggerOut, out_csv: str):
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = EventLogger(out_csv)

    win = visual.Window(color=(-0.1, -0.1, -0.1), units="norm", fullscr=True)
    title = visual.TextStim(win, text="Motor Imagery Protocol V2", pos=(0, 0.75), height=0.07, color=(0.9, 0.9, 0.9))
    cue = visual.TextStim(win, text="", pos=(0, 0.15), height=0.09, color=(0.95, 0.95, 0.95), wrapWidth=1.8)
    status = visual.TextStim(win, text="", pos=(0, -0.65), height=0.045, color=(0.8, 0.8, 0.8), wrapWidth=1.8)

    # Balanced class plan with explicit block IDs
    class_plan = (
        [("Neutral", 10, cfg.neutral_blocks)]
        + [("Left", 11, cfg.left_blocks)]
        + [("Right", 12, cfg.right_blocks)]
    )

    # Shuffle per-round by building repeated list of class names (balanced)
    all_blocks = []
    for cls_name, cls_code, n in class_plan:
        for i in range(1, n + 1):
            all_blocks.append((cls_name, cls_code, i))
    random.shuffle(all_blocks)

    instructions = (
        "You will perform KINESTHETIC motor imagery.\n\n"
        "LEFT: feel your left arm/hand movement (do not move).\n"
        "RIGHT: feel your right arm/hand movement (do not move).\n"
        "NEUTRAL: relaxed fixation, no imagery.\n\n"
        "After each block:\n"
        "SPACE = accept block, R = reject/repeat block.\n"
        "ESC = quit.\n\n"
        "Press SPACE to begin."
    )
    cue.text = instructions
    status.text = f"Participant: {participant} | Session: {session_id}"
    draw_main(win, title, cue, status)
    wait_for_key({"space"})

    accepted_count = 0
    pending = list(all_blocks)

    try:
        while pending:
            cls_name, cls_code, logical_idx = pending.pop(0)
            block_id = f"{cls_name.lower()}_{logical_idx:02d}"
            attempt = 0

            while True:
                attempt += 1

                cue.text = f"Upcoming block: {cls_name}\nPrepare..."
                status.text = (
                    f"Block {block_id} | Attempt {attempt}\n"
                    f"Press SPACE to start"
                )
                draw_main(win, title, cue, status)
                wait_for_key({"space"})

                logger.log("block_ready", participant, session_id, block_id, cls_name, cls_code, attempt)

                cue.text = "Baseline fixation"
                status.text = f"Hold still for {cfg.baseline_s:.1f}s"
                draw_main(win, title, cue, status)
                trig.send(TriggerCodes.block_start)
                logger.log("baseline_start", participant, session_id, block_id, cls_name, cls_code, attempt, cfg.baseline_s)
                wait_with_escape(cfg.baseline_s)

                cue.text = f"CUE: {cls_name}"
                status.text = (
                    f"Perform kinesthetic imagery now ({cfg.cue_s:.1f}s)\n"
                    "No overt movement"
                )
                draw_main(win, title, cue, status)
                trig.send(cls_code)
                logger.log("cue_start", participant, session_id, block_id, cls_name, cls_code, attempt, cfg.cue_s)
                wait_with_escape(cfg.cue_s)

                iti = random.uniform(cfg.iti_min_s, cfg.iti_max_s)
                cue.text = "Rest"
                status.text = f"Inter-trial interval: {iti:.2f}s"
                draw_main(win, title, cue, status)
                logger.log("iti_start", participant, session_id, block_id, cls_name, cls_code, attempt, iti)
                wait_with_escape(iti)

                cue.text = f"Review block: {cls_name}"
                status.text = "SPACE = accept, R = reject and repeat"
                draw_main(win, title, cue, status)
                key = wait_for_key({"space", "r"})

                if key == "space":
                    trig.send(TriggerCodes.block_accept)
                    logger.log("block_accept", participant, session_id, block_id, cls_name, cls_code, attempt)
                    accepted_count += 1
                    break

                trig.send(TriggerCodes.block_reject)
                logger.log("block_reject", participant, session_id, block_id, cls_name, cls_code, attempt)

            if accepted_count % cfg.break_every_blocks == 0 and pending:
                for remaining in range(cfg.break_s, 0, -1):
                    cue.text = "Break"
                    status.text = f"Resume in {remaining}s\n(ESC to quit)"
                    draw_main(win, title, cue, status)
                    wait_with_escape(1.0)

        cue.text = "Session complete"
        status.text = "All blocks accepted. Press ESC to exit."
        draw_main(win, title, cue, status)
        logger.log("session_complete", participant, session_id, notes="all_blocks_accepted")

        while True:
            if "escape" in event.getKeys():
                break
            core.wait(0.05)

    except KeyboardInterrupt:
        logger.log("session_aborted", participant, session_id, notes="escape_pressed")
        raise
    finally:
        logger.close()
        win.close()


def parse_args(argv: list[str]):
    p = argparse.ArgumentParser(description="Motor Imagery Protocol V2 task")
    p.add_argument("--participant", required=True, help="Participant name/id")
    p.add_argument("--port", default="", help="Optional serial port for trigger output (e.g., COM8 or /dev/ttyUSB0)")
    p.add_argument("--out-dir", default="data/mi_protocol_v2", help="Output directory for event log csv")
    p.add_argument(
        "--blocks-per-class",
        type=int,
        default=12,
        help="Accepted blocks per class (Neutral/Left/Right). Default: 12",
    )
    p.add_argument("--break-every-blocks", type=int, default=6, help="Break every N accepted blocks. Default: 6")
    p.add_argument("--break-seconds", type=int, default=45, help="Break duration in seconds. Default: 45")
    return p.parse_args(argv)


def main(argv: list[str]):
    args = parse_args(argv)
    participant = sanitize_name(args.participant)

    branch = os.popen("git rev-parse --abbrev-ref HEAD 2>/dev/null").read().strip()
    if branch == "main":
        print("[ERROR] You are on 'main'. Switch branches before running this task.")
        return 2

    timestamp = datetime.now().strftime("%m_%d_%y")
    out_csv = os.path.join(args.out_dir, f"{timestamp}_{participant}_mi_protocol_v2_events.csv")

    cfg = ProtocolConfig(
        neutral_blocks=args.blocks_per_class,
        left_blocks=args.blocks_per_class,
        right_blocks=args.blocks_per_class,
        break_every_blocks=args.break_every_blocks,
        break_s=args.break_seconds,
    )
    trig = TriggerOut(args.port if args.port else None)

    try:
        load_psychopy_modules()
        print(f"[INFO] Branch: {branch or '(unknown)'}")
        print(f"[INFO] Event log: {out_csv}")
        run_protocol(participant=participant, cfg=cfg, trig=trig, out_csv=out_csv)
        return 0
    except ModuleNotFoundError as e:
        print(f"[ERROR] Missing dependency: {e}. Install PsychoPy in this environment.")
        return 3
    except KeyboardInterrupt:
        print("\n[INFO] Task aborted by user.")
        return 130
    finally:
        trig.close()
        if core is not None:
            core.quit()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
