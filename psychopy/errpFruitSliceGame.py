"""Pygame-based ErrP fruit slicing task with optional trigger-hub support.

Task structure:
- One fruit launches from the left or right.
- A central strike zone marks when the participant should act.
- The participant presses LEFT or RIGHT when the fruit is inside the zone.
- Correct in-window responses usually slice the fruit.
- On some correct trials, the sword swings to the opposite side instead.
- Wrong side, bad timing, and no-response misses are marked separately.

The primary EEG event is the onset of visible feedback:
- For swing-based outcomes, triggers are sent when the swing begins.
- For no-response misses, the trigger is sent when the fruit exits the strike
  zone and the miss becomes explicit.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import random
import threading
import time
from typing import Any

import pygame
import serial
import serial.tools.list_ports


SIDE_LEFT = -1
SIDE_RIGHT = 1


@dataclass(frozen=True)
class TriggerCodes:
    correct_left: int = 1
    correct_right: int = 2
    system_error_left: int = 3
    system_error_right: int = 4
    user_wrong_side_left: int = 5
    user_wrong_side_right: int = 6
    user_bad_timing_left: int = 7
    user_bad_timing_right: int = 8
    user_no_response_left: int = 9
    user_no_response_right: int = 10
    cue_fruit_left: int = 11
    cue_fruit_right: int = 12
    block_start: int = 13
    session_end: int = 14


@dataclass(frozen=True)
class SerialConfig:
    port: str | None = None
    baudrate: int = 115200
    pulse_width_s: float = 0.01
    vid: int = 0x2341
    pid: int = 0x8037
    auto_detect: bool = True
    enabled: bool = True


@dataclass(frozen=True)
class TaskConfig:
    trials: int = 250
    break_every: int = 25
    error_prob: float = 0.30
    window_width: int = 1440
    window_height: int = 900
    fullscreen: bool = False
    target_fps: int = 120
    seed: int | None = None
    travel_min_s: float = 2.00
    travel_max_s: float = 2.50
    gap_min_s: float = 0.55
    gap_max_s: float = 1.05
    action_duration_s: float = 0.72
    no_response_duration_s: float = 0.62
    correct_points: int = 100
    system_error_points: int = -150
    user_wrong_side_points: int = -100
    user_bad_timing_points: int = -90
    user_no_response_points: int = -110
    log_dir: str = "."


@dataclass
class TrialPlan:
    trial_index: int
    fruit_side: int
    fruit_kind: str
    travel_duration_s: float
    gap_s: float


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease_in_out(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ease_out_cubic(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return 1.0 - (1.0 - t) ** 3


def side_name(side: int | None) -> str:
    if side == SIDE_LEFT:
        return "left"
    if side == SIDE_RIGHT:
        return "right"
    return ""


def quadratic_bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    u = 1.0 - t
    x = u * u * p0[0] + 2.0 * u * t * p1[0] + t * t * p2[0]
    y = u * u * p0[1] + 2.0 * u * t * p1[1] + t * t * p2[1]
    return x, y


class TriggerHub:
    """Optional serial trigger sender that degrades to simulation mode."""

    def __init__(self, cfg: SerialConfig):
        self.cfg = cfg
        self.ser: serial.Serial | None = None
        self.port_used: str | None = None
        self._lock = threading.Lock()

    def open(self) -> None:
        if not self.cfg.enabled:
            print("[TRIGGER] Disabled; running in simulation mode.")
            return

        port = self.cfg.port
        if port is None and self.cfg.auto_detect:
            for info in serial.tools.list_ports.comports():
                if info.vid == self.cfg.vid and info.pid == self.cfg.pid:
                    port = info.device
                    break

        if port is None:
            print("[TRIGGER] No trigger hub detected; running in simulation mode.")
            return

        try:
            self.ser = serial.Serial(port, self.cfg.baudrate, timeout=0)
            time.sleep(0.05)
        except Exception as exc:
            self.ser = None
            print(f"[TRIGGER] Failed to open {port}: {exc}. Running in simulation mode.")
            return

        self.port_used = port
        print(f"[TRIGGER] Connected on {port} @ {self.cfg.baudrate} baud.")

    def close(self) -> None:
        with self._lock:
            if self.ser is not None:
                try:
                    self.ser.close()
                finally:
                    self.ser = None

    def pulse(self, code: int, wait: bool = False) -> None:
        if self.ser is None:
            return
        code = int(code) & 0xFF
        if wait:
            self._pulse_blocking(code)
            return
        thread = threading.Thread(target=self._pulse_blocking, args=(code,), daemon=True)
        thread.start()

    def _pulse_blocking(self, code: int) -> None:
        with self._lock:
            if self.ser is None:
                return
            self.ser.write(bytes([code]))
            self.ser.flush()
            time.sleep(self.cfg.pulse_width_s)
            self.ser.write(bytes([0]))
            self.ser.flush()


class BalancedSideScheduler:
    def __init__(self, rng: random.Random, block_size: int = 8):
        self.rng = rng
        self.block_size = max(2, int(block_size))
        self._bag: list[int] = []

    def next_side(self) -> int:
        if not self._bag:
            n_left = self.block_size // 2
            n_right = self.block_size - n_left
            self._bag = [SIDE_LEFT] * n_left + [SIDE_RIGHT] * n_right
            self.rng.shuffle(self._bag)
        return self._bag.pop()


class BalancedErrorScheduler:
    """Keep system-error probability stable across eligible responses."""

    def __init__(self, rng: random.Random, error_prob: float, block_size: int = 20):
        self.rng = rng
        self.error_prob = clamp(float(error_prob), 0.0, 1.0)
        self.block_size = max(1, int(block_size))
        self._bag: list[bool] = []

    def next_is_error(self) -> bool:
        if not self._bag:
            n_errors = int(round(self.error_prob * self.block_size))
            n_errors = max(0, min(self.block_size, n_errors))
            self._bag = [True] * n_errors + [False] * (self.block_size - n_errors)
            self.rng.shuffle(self._bag)
        return self._bag.pop()


class SessionLogger:
    def __init__(self, log_dir: Path, session_name: str, config: TaskConfig, triggers: TriggerCodes):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trial_path = self.log_dir / f"{session_name}_trials.csv"
        self.meta_path = self.log_dir / f"{session_name}_config.json"

        self._trial_file = self.trial_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._trial_file,
            fieldnames=[
                "session",
                "trial_index",
                "fruit_side",
                "fruit_kind",
                "travel_duration_s",
                "user_input_side",
                "user_input_rt_ms",
                "input_in_window",
                "event_onset_perf_s",
                "event_onset_monotonic_ns",
                "outcome",
                "trigger_code",
                "trigger_name",
                "executed_side",
                "system_error_applied",
                "fruit_sliced",
                "score_before",
                "score_after",
                "streak_after",
                "gap_s",
            ],
        )
        self._writer.writeheader()

        with self.meta_path.open("w", encoding="utf-8") as meta_file:
            json.dump(
                {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "task_config": asdict(config),
                    "trigger_codes": asdict(triggers),
                },
                meta_file,
                indent=2,
            )

    def write_trial(self, row: dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._trial_file.flush()

    def close(self) -> None:
        self._trial_file.close()


class FruitSliceErrPGame:
    def __init__(self, cfg: TaskConfig, serial_cfg: SerialConfig, session_name: str):
        self.cfg = cfg
        self.session_name = session_name
        self.triggers = TriggerCodes()
        self.rng = random.Random(cfg.seed)
        self.scheduler = BalancedSideScheduler(self.rng)
        self.error_scheduler = BalancedErrorScheduler(self.rng, cfg.error_prob)
        self.trigger_hub = TriggerHub(serial_cfg)
        self.logger = SessionLogger(Path(cfg.log_dir), session_name, cfg, self.triggers)

        self.width = cfg.window_width
        self.height = cfg.window_height
        self.zone_size = int(self.height * 0.15)
        self.zone_gap = int(self.width * 0.12)
        zone_center_y = int(self.height * 0.56)
        left_center_x = int(self.width / 2 - self.zone_gap / 2 - self.zone_size / 2)
        right_center_x = int(self.width / 2 + self.zone_gap / 2 + self.zone_size / 2)
        self.left_zone = pygame.Rect(0, 0, self.zone_size, self.zone_size)
        self.left_zone.center = (left_center_x, zone_center_y)
        self.right_zone = pygame.Rect(0, 0, self.zone_size, self.zone_size)
        self.right_zone.center = (right_center_x, zone_center_y)
        self.zone_union = self.left_zone.union(self.right_zone)
        self.sword_anchor = (self.width / 2, self.height - 150)

        self.fruit_defs = {
            "watermelon": {"fill": (63, 186, 99), "flesh": (245, 92, 96), "highlight": (240, 250, 240), "seed": (34, 34, 34), "radius": 46},
            "orange": {"fill": (255, 168, 46), "flesh": (255, 214, 120), "highlight": (255, 246, 225), "seed": (255, 240, 210), "radius": 38},
            "plum": {"fill": (118, 62, 162), "flesh": (212, 110, 214), "highlight": (240, 220, 255), "seed": (255, 244, 220), "radius": 40},
            "kiwi": {"fill": (128, 84, 46), "flesh": (136, 214, 98), "highlight": (233, 255, 222), "seed": (31, 31, 31), "radius": 42},
        }

        pygame.init()
        pygame.font.init()
        flags = pygame.DOUBLEBUF
        if cfg.fullscreen:
            flags |= pygame.FULLSCREEN
        try:
            self.screen = pygame.display.set_mode((self.width, self.height), flags, vsync=1)
        except TypeError:
            self.screen = pygame.display.set_mode((self.width, self.height), flags)
        pygame.display.set_caption("ErrP Fruit Ninja")
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("arial", 36, bold=True)
        self.font_hud = pygame.font.SysFont("arial", 22, bold=True)
        self.font_small = pygame.font.SysFont("arial", 12, bold=True)
        self.font_body = pygame.font.SysFont("arial", 20)
        self.font_overlay = pygame.font.SysFont("arial", 24)
        self.font_countdown = pygame.font.SysFont("arial", 96, bold=True)

        self.background_surface = self._build_background()
        self.flash_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        self.running = True
        self.closed = False
        self.state = "intro"
        self.state_started_at = time.perf_counter()
        self.phase_started_at = self.state_started_at

        self.score = 0
        self.streak = 0
        self.completed_trials = 0
        self.pending_space = False
        self.pending_input: tuple[int, float, float] | None = None

        self.current_trial: TrialPlan | None = None
        self.current_progress = 0.0
        self.previous_in_window = False
        self.window_enter_s: float | None = None
        self.window_exit_s: float | None = None
        self.fruit_pos = (0.0, 0.0)
        self.fruit_rotation = 0.0

        self.feedback_text = ""
        self.feedback_color = (255, 255, 255)
        self.feedback_until = 0.0
        self.flash_alpha = 0.0
        self.camera_shake = 0.0

        self.action_outcome = ""
        self.action_trigger_code = 0
        self.action_trigger_name = ""
        self.action_user_side: int | None = None
        self.action_user_rt_ms: float | None = None
        self.action_executed_side: int | None = None
        self.action_input_in_window = False
        self.action_system_error = False
        self.action_fruit_sliced = False
        self.action_onset_s = 0.0
        self.action_onset_ns = 0
        self.action_start_progress = 0.0
        self.action_start_elapsed_s = 0.0
        self.action_contact_progress = 0.0
        self.action_contact_pos = (0.0, 0.0)
        self.score_before_action = 0
        self.post_trial_deadline = 0.0
        self.slice_particles: list[dict[str, float]] = []

        self.trigger_hub.open()

    def run(self) -> None:
        try:
            while self.running:
                dt = self.clock.tick(self.cfg.target_fps) / 1000.0
                now = time.perf_counter()
                self._process_events(now)
                self._update(now, dt)
                self._render(now)
                pygame.display.flip()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.running = False
        try:
            self.trigger_hub.pulse(self.triggers.session_end, wait=True)
        except Exception:
            pass
        self.trigger_hub.close()
        self.logger.close()
        pygame.quit()

    def _build_background(self) -> pygame.Surface:
        surface = pygame.Surface((self.width, self.height))
        for y in range(self.height):
            t = y / max(1, self.height - 1)
            r = int(10 + 32 * (1 - t))
            g = int(16 + 28 * (1 - t))
            b = int(24 + 58 * (1 - t))
            pygame.draw.line(surface, (r, g, b), (0, y), (self.width, y))

        for idx in range(6):
            alpha = 90 - idx * 12
            glow = pygame.Surface((520, 220), pygame.SRCALPHA)
            pygame.draw.ellipse(glow, (70, 150, 255, alpha), glow.get_rect())
            surface.blit(glow, (self.width // 2 - 260, 80 + idx * 18))

        for x in range(0, self.width, 120):
            pygame.draw.line(surface, (18, 26, 38), (x, 0), (x - 240, self.height), 1)

        return surface.convert()

    def _process_events(self, now: float) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type != pygame.KEYDOWN:
                continue

            if event.key == pygame.K_ESCAPE:
                self.running = False
                return

            if event.key == pygame.K_SPACE:
                self.pending_space = True
                continue

            if event.key not in (pygame.K_LEFT, pygame.K_RIGHT):
                continue

            if self.state != "active" or self.current_trial is None or self.pending_input is not None:
                continue

            user_side = SIDE_LEFT if event.key == pygame.K_LEFT else SIDE_RIGHT
            rt_ms = (now - self.phase_started_at) * 1000.0
            self.pending_input = (user_side, now, rt_ms)

    def _update(self, now: float, dt: float) -> None:
        self.camera_shake *= 0.88
        self.flash_alpha = max(0.0, self.flash_alpha - dt * 1.7)
        for particle in self.slice_particles:
            particle["x"] += particle["vx"] * dt
            particle["y"] += particle["vy"] * dt
            particle["vy"] += 720.0 * dt
            particle["life"] -= dt
        self.slice_particles = [p for p in self.slice_particles if p["life"] > 0]

        if self.state == "intro":
            if self.pending_space:
                self.pending_space = False
                self._transition("countdown", now)
            return

        if self.state == "countdown":
            if now - self.state_started_at >= 3.4:
                self._begin_next_trial(now)
            return

        if self.state == "active":
            self._update_active(now)
            return

        if self.state == "action":
            self._update_action(now)
            return

        if self.state == "gap":
            if now >= self.post_trial_deadline:
                self._after_gap(now)
            return

        if self.state == "break":
            if self.pending_space:
                self.pending_space = False
                self._begin_next_trial(now)
            return

        if self.state == "complete":
            if self.pending_space:
                self.pending_space = False
                self.running = False

    def _transition(self, state: str, now: float) -> None:
        self.state = state
        self.state_started_at = now

    def _begin_next_trial(self, now: float) -> None:
        if self.completed_trials >= self.cfg.trials:
            self._transition("complete", now)
            return

        fruit_side = self.scheduler.next_side()
        fruit_kind = self.rng.choice(list(self.fruit_defs))
        self.current_trial = TrialPlan(
            trial_index=self.completed_trials + 1,
            fruit_side=fruit_side,
            fruit_kind=fruit_kind,
            travel_duration_s=self.rng.uniform(self.cfg.travel_min_s, self.cfg.travel_max_s),
            gap_s=self.rng.uniform(self.cfg.gap_min_s, self.cfg.gap_max_s),
        )
        self.current_progress = 0.0
        self.previous_in_window = False
        self.window_enter_s = None
        self.window_exit_s = None
        self.pending_input = None
        self.feedback_text = ""
        self.action_outcome = ""
        self.slice_particles.clear()
        self.fruit_pos = self._fruit_position(0.0)
        self.fruit_rotation = 0.0
        self.phase_started_at = now
        self.trigger_hub.pulse(self.triggers.cue_fruit_left if fruit_side == SIDE_LEFT else self.triggers.cue_fruit_right)
        self._transition("active", now)

    def _update_active(self, now: float) -> None:
        assert self.current_trial is not None
        elapsed = now - self.phase_started_at
        self.current_progress = clamp(elapsed / self.current_trial.travel_duration_s, 0.0, 1.0)
        self.fruit_pos = self._fruit_position(self.current_progress)
        self.fruit_rotation = self.current_progress * 480.0 * self.current_trial.fruit_side

        in_window = self._fruit_in_window(self.fruit_pos, self.current_trial.fruit_side)
        if in_window and not self.previous_in_window:
            self.window_enter_s = now
        if not in_window and self.previous_in_window and self.pending_input is None:
            self.window_exit_s = now
            self._begin_action(now, user_side=None, user_rt_ms=None, input_in_window=False, outcome_type="user_no_response")
            return
        self.previous_in_window = in_window

        if self.pending_input is not None:
            user_side, _, rt_ms = self.pending_input
            self.pending_input = None

            if user_side != self.current_trial.fruit_side:
                self._begin_action(now, user_side=user_side, user_rt_ms=rt_ms, input_in_window=in_window, outcome_type="user_wrong_side")
            elif not in_window:
                self._begin_action(now, user_side=user_side, user_rt_ms=rt_ms, input_in_window=False, outcome_type="user_bad_timing")
            else:
                self._begin_action(now, user_side=user_side, user_rt_ms=rt_ms, input_in_window=True, outcome_type="candidate_correct")
            return

        if self.current_progress >= 1.0:
            self._begin_action(now, user_side=None, user_rt_ms=None, input_in_window=False, outcome_type="user_no_response")

    def _begin_action(
        self,
        now: float,
        user_side: int | None,
        user_rt_ms: float | None,
        input_in_window: bool,
        outcome_type: str,
    ) -> None:
        assert self.current_trial is not None

        fruit_side = self.current_trial.fruit_side
        self.action_user_side = user_side
        self.action_user_rt_ms = user_rt_ms
        self.action_input_in_window = input_in_window
        self.action_system_error = False
        self.action_fruit_sliced = False
        self.score_before_action = self.score
        self.action_executed_side = user_side
        self.action_start_progress = self.current_progress
        self.action_start_elapsed_s = self.current_progress * self.current_trial.travel_duration_s
        self.action_contact_progress = self.current_progress
        self.action_contact_pos = self.fruit_pos

        if outcome_type == "candidate_correct":
            if self.error_scheduler.next_is_error():
                self.action_outcome = "system_error"
                self.action_system_error = True
                self.action_executed_side = -fruit_side
                self.action_trigger_code = (
                    self.triggers.system_error_left if self.action_executed_side == SIDE_LEFT else self.triggers.system_error_right
                )
                self.action_trigger_name = "system_error_left" if self.action_executed_side == SIDE_LEFT else "system_error_right"
                self.feedback_text = "SYSTEM SWUNG WRONG WAY"
                self.feedback_color = (255, 103, 143)
                self.score += self.cfg.system_error_points
                self.streak = 0
            else:
                self.action_outcome = "correct"
                self.action_executed_side = fruit_side
                self.action_trigger_code = self.triggers.correct_left if fruit_side == SIDE_LEFT else self.triggers.correct_right
                self.action_trigger_name = "correct_left" if fruit_side == SIDE_LEFT else "correct_right"
                self.action_fruit_sliced = True
                self.feedback_text = "CLEAN SLICE"
                self.feedback_color = (116, 226, 175)
                self.score += self.cfg.correct_points
                self.streak += 1
        elif outcome_type == "user_wrong_side":
            self.action_outcome = "user_wrong_side"
            self.action_trigger_code = self.triggers.user_wrong_side_left if user_side == SIDE_LEFT else self.triggers.user_wrong_side_right
            self.action_trigger_name = "user_wrong_side_left" if user_side == SIDE_LEFT else "user_wrong_side_right"
            self.feedback_text = "WRONG SIDE"
            self.feedback_color = (255, 137, 99)
            self.score += self.cfg.user_wrong_side_points
            self.streak = 0
        elif outcome_type == "user_bad_timing":
            self.action_outcome = "user_bad_timing"
            self.action_trigger_code = self.triggers.user_bad_timing_left if user_side == SIDE_LEFT else self.triggers.user_bad_timing_right
            self.action_trigger_name = "user_bad_timing_left" if user_side == SIDE_LEFT else "user_bad_timing_right"
            self.feedback_text = "BAD TIMING"
            self.feedback_color = (255, 196, 87)
            self.score += self.cfg.user_bad_timing_points
            self.streak = 0
        else:
            self.action_outcome = "user_no_response"
            self.action_executed_side = None
            self.action_trigger_code = self.triggers.user_no_response_left if fruit_side == SIDE_LEFT else self.triggers.user_no_response_right
            self.action_trigger_name = "user_no_response_left" if fruit_side == SIDE_LEFT else "user_no_response_right"
            self.feedback_text = "MISSED FRUIT"
            self.feedback_color = (255, 112, 132)
            self.score += self.cfg.user_no_response_points
            self.streak = 0

        self.action_onset_s = now
        self.action_onset_ns = time.monotonic_ns()
        self.feedback_until = now + 0.95
        self.flash_alpha = 0.20 if self.action_outcome != "correct" else 0.08
        if self.action_outcome != "correct":
            self.camera_shake = max(self.camera_shake, 6.0)
        self.trigger_hub.pulse(self.action_trigger_code)

        if self.action_fruit_sliced:
            self._spawn_slice_particles(self.fruit_pos, self.current_trial.fruit_kind)

        self._transition("action", now)

    def _spawn_slice_particles(self, pos: tuple[float, float], fruit_kind: str) -> None:
        info = self.fruit_defs[fruit_kind]
        for _ in range(18):
            angle = self.rng.uniform(-2.2, -0.9)
            speed = self.rng.uniform(120.0, 340.0)
            self.slice_particles.append(
                {
                    "x": pos[0],
                    "y": pos[1],
                    "vx": math.cos(angle) * speed * self.rng.choice((-1.0, 1.0)),
                    "vy": math.sin(angle) * speed,
                    "life": self.rng.uniform(0.28, 0.52),
                    "r": self.rng.uniform(4.0, 8.0),
                    "color_r": info["flesh"][0],
                    "color_g": info["flesh"][1],
                    "color_b": info["flesh"][2],
                }
            )

    def _update_action(self, now: float) -> None:
        assert self.current_trial is not None
        duration = self.cfg.no_response_duration_s if self.action_outcome == "user_no_response" else self.cfg.action_duration_s
        action_t = clamp((now - self.action_onset_s) / duration, 0.0, 1.0)

        if self.action_fruit_sliced:
            self.current_progress = self.action_start_progress
        else:
            natural_elapsed_s = self.action_start_elapsed_s + (now - self.action_onset_s)
            self.current_progress = clamp(natural_elapsed_s / self.current_trial.travel_duration_s, 0.0, 1.0)
            self.fruit_pos = self._fruit_position(self.current_progress)
            self.fruit_rotation = self.current_progress * 480.0 * self.current_trial.fruit_side

        if action_t >= 1.0:
            self._finalize_trial(now)

    def _finalize_trial(self, now: float) -> None:
        assert self.current_trial is not None
        self.logger.write_trial(
            {
                "session": self.session_name,
                "trial_index": self.current_trial.trial_index,
                "fruit_side": side_name(self.current_trial.fruit_side),
                "fruit_kind": self.current_trial.fruit_kind,
                "travel_duration_s": f"{self.current_trial.travel_duration_s:.4f}",
                "user_input_side": side_name(self.action_user_side),
                "user_input_rt_ms": "" if self.action_user_rt_ms is None else f"{self.action_user_rt_ms:.3f}",
                "input_in_window": int(self.action_input_in_window),
                "event_onset_perf_s": f"{self.action_onset_s:.6f}",
                "event_onset_monotonic_ns": self.action_onset_ns,
                "outcome": self.action_outcome,
                "trigger_code": self.action_trigger_code,
                "trigger_name": self.action_trigger_name,
                "executed_side": side_name(self.action_executed_side),
                "system_error_applied": int(self.action_system_error),
                "fruit_sliced": int(self.action_fruit_sliced),
                "score_before": self.score_before_action,
                "score_after": self.score,
                "streak_after": self.streak,
                "gap_s": f"{self.current_trial.gap_s:.4f}",
            }
        )
        self.completed_trials += 1
        self.post_trial_deadline = now + self.current_trial.gap_s
        self.current_trial = None
        self.pending_input = None
        self.current_progress = 0.0
        self.previous_in_window = False
        self.window_enter_s = None
        self.window_exit_s = None
        self._transition("gap", now)

    def _after_gap(self, now: float) -> None:
        if self.completed_trials >= self.cfg.trials:
            self._transition("complete", now)
            return

        if self.cfg.break_every > 0 and self.completed_trials % self.cfg.break_every == 0:
            self.trigger_hub.pulse(self.triggers.block_start)
            self._transition("break", now)
            return

        self._begin_next_trial(now)

    def _fruit_curve(self, side: int) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        mid = self.left_zone.center if side == SIDE_LEFT else self.right_zone.center
        if side == SIDE_LEFT:
            p0 = (-170.0, self.height * 0.86)
            p2 = (self.width + 180.0, self.height * 0.74)
        else:
            p0 = (self.width + 170.0, self.height * 0.86)
            p2 = (-180.0, self.height * 0.74)
        p1 = (
            2.0 * mid[0] - 0.5 * (p0[0] + p2[0]),
            2.0 * mid[1] - 0.5 * (p0[1] + p2[1]),
        )
        return p0, p1, p2

    def _fruit_position(self, progress: float) -> tuple[float, float]:
        assert self.current_trial is not None
        return quadratic_bezier(*self._fruit_curve(self.current_trial.fruit_side), progress)

    def _fruit_in_window(self, pos: tuple[float, float], fruit_side: int) -> bool:
        zone = self.left_zone if fruit_side == SIDE_LEFT else self.right_zone
        return zone.collidepoint(pos)

    def _draw_text(
        self,
        text: str,
        font: pygame.font.Font,
        color: tuple[int, int, int],
        x: float,
        y: float,
        anchor: str = "topleft",
    ) -> None:
        surface = font.render(text, True, color)
        rect = surface.get_rect()
        if anchor == "topleft":
            rect.topleft = (x, y)
        elif anchor == "topright":
            rect.topright = (x, y)
        elif anchor == "midtop":
            rect.midtop = (x, y)
        elif anchor == "center":
            rect.center = (x, y)
        elif anchor == "midbottom":
            rect.midbottom = (x, y)
        self.screen.blit(surface, rect)

    def _draw_multiline(
        self,
        text: str,
        font: pygame.font.Font,
        color: tuple[int, int, int],
        x: float,
        y: float,
        line_gap: int = 8,
        anchor: str = "midtop",
    ) -> None:
        lines = text.splitlines()
        rendered = [font.render(line, True, color) for line in lines]
        total_h = sum(s.get_height() for s in rendered) + max(0, len(rendered) - 1) * line_gap
        current_y = y
        if anchor == "center":
            current_y = y - total_h / 2
        elif anchor == "midbottom":
            current_y = y - total_h

        for surf in rendered:
            rect = surf.get_rect()
            if anchor in {"midtop", "center", "midbottom"}:
                rect.midtop = (x, current_y)
            else:
                rect.topleft = (x, current_y)
            self.screen.blit(surf, rect)
            current_y += surf.get_height() + line_gap

    def _draw_panel(self, rect: pygame.Rect, fill: tuple[int, int, int, int], border: tuple[int, int, int]) -> None:
        panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(panel, fill, panel.get_rect(), border_radius=20)
        pygame.draw.rect(panel, border, panel.get_rect(), width=2, border_radius=20)
        self.screen.blit(panel, rect)

    def _draw_zone(self, now: float) -> None:
        pulse = 0.5 + 0.5 * math.sin(now * 4.8)
        left_alpha = 90
        right_alpha = 90
        left_fill_alpha = 28
        right_fill_alpha = 28
        if self.current_trial is not None and self.state == "active":
            if self.current_trial.fruit_side == SIDE_LEFT:
                left_alpha = int(180 + 55 * pulse)
                left_fill_alpha = int(48 + 24 * pulse)
            else:
                right_alpha = int(180 + 55 * pulse)
                right_fill_alpha = int(48 + 24 * pulse)

        for zone, fill_color, label, label_color, alpha, fill_alpha in (
            (self.left_zone, (46, 220, 166), "LEFT", (220, 245, 236), left_alpha, left_fill_alpha),
            (self.right_zone, (255, 126, 83), "RIGHT", (255, 232, 226), right_alpha, right_fill_alpha),
        ):
            zone_surf = pygame.Surface((zone.width, zone.height), pygame.SRCALPHA)
            pygame.draw.rect(zone_surf, (10, 26, 44, 160), zone_surf.get_rect(), border_radius=18)
            pygame.draw.rect(zone_surf, (*fill_color, fill_alpha), zone_surf.get_rect(), border_radius=18)
            pygame.draw.rect(zone_surf, (*fill_color, alpha), zone_surf.get_rect(), width=4, border_radius=18)
            self.screen.blit(zone_surf, zone)
            self._draw_text(label, self.font_small, label_color, zone.centerx, zone.top + 12, anchor="midtop")

        connector_y = self.left_zone.centery
        pygame.draw.line(
            self.screen,
            (88, 116, 150),
            (self.left_zone.right + 18, connector_y),
            (self.right_zone.left - 18, connector_y),
            2,
        )
        center_dot_color = (int(120 + 90 * pulse), int(180 + 60 * pulse), 255)
        pygame.draw.circle(self.screen, center_dot_color, (self.width // 2, connector_y), 8)

    def _draw_sword(self, now: float) -> None:
        base = self.sword_anchor
        pommel = (base[0], base[1] + 22)
        pygame.draw.line(self.screen, (145, 82, 46), pommel, (base[0], base[1] - 20), 10)
        pygame.draw.line(self.screen, (224, 210, 156), (base[0] - 28, base[1] - 2), (base[0] + 28, base[1] - 2), 6)

        if self.state != "action" or self.action_executed_side is None:
            blade_len = 210
            rest_angle = -math.pi / 2
            tip = (base[0] + math.cos(rest_angle) * blade_len, base[1] + math.sin(rest_angle) * blade_len)
            pygame.draw.line(self.screen, (210, 225, 238), base, tip, 8)
            return

        swing_t = clamp((now - self.action_onset_s) / 0.26, 0.0, 1.0)
        zone = self.left_zone if self.action_executed_side == SIDE_LEFT else self.right_zone
        target = (zone.centerx, zone.centery + 6)
        blade_len = 230
        rest_angle = -math.pi / 2
        target_angle = math.atan2(target[1] - base[1], target[0] - base[0])
        angle = lerp(rest_angle, target_angle, ease_out_cubic(swing_t))
        tip = (base[0] + math.cos(angle) * blade_len, base[1] + math.sin(angle) * blade_len)

        trail = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for idx in range(5):
            trail_t = clamp(swing_t - idx * 0.08, 0.0, 1.0)
            trail_angle = lerp(rest_angle, target_angle, ease_out_cubic(trail_t))
            trail_tip = (
                base[0] + math.cos(trail_angle) * blade_len,
                base[1] + math.sin(trail_angle) * blade_len,
            )
            alpha = max(0, 150 - idx * 28)
            pygame.draw.line(trail, (170, 228, 255, alpha), base, trail_tip, 18 - idx * 2)
        self.screen.blit(trail, (0, 0))

        pygame.draw.line(self.screen, (234, 245, 252), base, tip, 10)
        pygame.draw.line(self.screen, (128, 194, 255), base, tip, 4)

    def _draw_fruit(
        self,
        pos: tuple[float, float],
        rotation_deg: float,
        fruit_kind: str,
        split: bool = False,
        action_t: float = 0.0,
        split_side: int | None = None,
    ) -> None:
        info = self.fruit_defs[fruit_kind]
        radius = info["radius"]

        if not split:
            shell = pygame.Surface((radius * 2 + 24, radius * 2 + 24), pygame.SRCALPHA)
            center = (shell.get_width() // 2, shell.get_height() // 2)
            pygame.draw.circle(shell, info["fill"], center, radius)
            pygame.draw.circle(shell, info["highlight"], (center[0] - radius // 3, center[1] - radius // 3), max(8, radius // 4))
            for idx in range(6):
                seed_angle = idx * (math.pi / 3.0)
                sx = center[0] + math.cos(seed_angle) * radius * 0.35
                sy = center[1] + math.sin(seed_angle) * radius * 0.25
                pygame.draw.ellipse(shell, info["seed"], pygame.Rect(sx - 3, sy - 6, 6, 12))
            pygame.draw.line(shell, (78, 196, 102), (center[0], center[1] - radius + 6), (center[0] + 14, center[1] - radius - 14), 5)
            rotated = pygame.transform.rotozoom(shell, rotation_deg, 1.0)
            rect = rotated.get_rect(center=(int(pos[0]), int(pos[1])))
            self.screen.blit(rotated, rect)
            return

        split_offset = 84.0 * ease_out_cubic(action_t)
        vertical_lift = 36.0 * ease_out_cubic(action_t)
        travel_follow = 1.0 + 0.2 * action_t
        spin = 140.0 * action_t
        split_side = SIDE_RIGHT if split_side is None else split_side
        for sign in (-1, 1):
            half_surface = pygame.Surface((radius * 2 + 24, radius * 2 + 24), pygame.SRCALPHA)
            center = (half_surface.get_width() // 2, half_surface.get_height() // 2)
            rect = pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2)
            start = math.pi / 2 if sign < 0 else -math.pi / 2
            end = math.pi * 3 / 2 if sign < 0 else math.pi / 2
            pygame.draw.arc(half_surface, info["fill"], rect, start, end, radius)
            points = [center]
            step_count = 16
            for idx in range(step_count + 1):
                ang = lerp(start, end, idx / step_count)
                points.append((center[0] + math.cos(ang) * radius, center[1] + math.sin(ang) * radius))
            pygame.draw.polygon(half_surface, info["flesh"], points)
            pygame.draw.lines(half_surface, info["fill"], False, points[1:], 8)
            for seed_idx in range(4):
                seed_x = center[0] + sign * radius * 0.12 + seed_idx * sign * 4
                seed_y = center[1] - radius * 0.26 + seed_idx * 9
                pygame.draw.ellipse(half_surface, info["seed"], pygame.Rect(seed_x - 2, seed_y - 4, 4, 8))
            rotated = pygame.transform.rotozoom(half_surface, rotation_deg + sign * spin, 1.0)
            lateral_dir = 1 if split_side == SIDE_RIGHT else -1
            half_rect = rotated.get_rect(
                center=(
                    int(pos[0] + sign * split_offset * lateral_dir * travel_follow),
                    int(pos[1] - vertical_lift + sign * 12.0 * action_t),
                )
            )
            self.screen.blit(rotated, half_rect)

    def _render_active_fruit(self) -> None:
        if self.current_trial is None:
            return
        self._draw_fruit(self.fruit_pos, self.fruit_rotation, self.current_trial.fruit_kind)

    def _render_action_fruit(self, now: float) -> None:
        assert self.current_trial is not None
        if self.action_fruit_sliced:
            t = clamp((now - self.action_onset_s) / self.cfg.action_duration_s, 0.0, 1.0)
            self._draw_fruit(
                self.fruit_pos,
                self.fruit_rotation,
                self.current_trial.fruit_kind,
                split=True,
                action_t=t,
                split_side=self.action_executed_side,
            )
            slash_t = clamp((now - self.action_onset_s) / 0.18, 0.0, 1.0)
            slash_alpha = int(170 * (1.0 - slash_t))
            if slash_alpha > 0:
                slash = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                offset = 130.0 * ease_out_cubic(slash_t)
                if self.action_executed_side == SIDE_LEFT:
                    start = (self.action_contact_pos[0] + offset * 0.45, self.action_contact_pos[1] - offset * 0.50)
                    end = (self.action_contact_pos[0] - offset, self.action_contact_pos[1] + offset * 0.65)
                else:
                    start = (self.action_contact_pos[0] - offset * 0.45, self.action_contact_pos[1] - offset * 0.50)
                    end = (self.action_contact_pos[0] + offset, self.action_contact_pos[1] + offset * 0.65)
                pygame.draw.line(slash, (220, 248, 255, slash_alpha), start, end, 12)
                pygame.draw.line(slash, (120, 220, 255, slash_alpha), start, end, 5)
                self.screen.blit(slash, (0, 0))
        else:
            self._draw_fruit(self.fruit_pos, self.fruit_rotation, self.current_trial.fruit_kind)

    def _render(self, now: float) -> None:
        self.screen.blit(self.background_surface, (0, 0))
        self._draw_zone(now)
        self._draw_sword(now)

        if self.state == "active":
            self._render_active_fruit()
        elif self.state == "action" and self.current_trial is not None:
            self._render_action_fruit(now)

        for particle in self.slice_particles:
            pygame.draw.circle(self.screen, (int(particle["color_r"]), int(particle["color_g"]), int(particle["color_b"])), (int(particle["x"]), int(particle["y"])), int(particle["r"]))

        self._draw_hud(now)
        self._draw_overlays(now)

        if self.flash_alpha > 0.0:
            self.flash_surface.fill((255, 184, 118, int(self.flash_alpha * 255)))
            self.screen.blit(self.flash_surface, (0, 0))

    def _draw_hud(self, now: float) -> None:
        trial_number = self.completed_trials if self.state == "complete" else min(self.cfg.trials, self.completed_trials + 1)
        self._draw_text(f"TRIAL {trial_number:03d}/{self.cfg.trials:03d}", self.font_hud, (226, 235, 245), 40, 28)
        self._draw_text("ERRP FRUIT NINJA", self.font_small, (130, 186, 255), 40, 62)
        self._draw_text(f"SCORE {self.score:+05d}", self.font_hud, (240, 244, 248), self.width - 40, 28, anchor="topright")
        self._draw_text(f"STREAK {self.streak:02d}", self.font_small, (113, 230, 176), self.width - 40, 62, anchor="topright")

        if self.current_trial is not None and self.state in {"active", "action"}:
            prompt = "PRESS LEFT" if self.current_trial.fruit_side == SIDE_LEFT else "PRESS RIGHT"
            self._draw_text(prompt, self.font_body, (247, 241, 210), self.width / 2, 82, anchor="midtop")

        if now < self.feedback_until and self.feedback_text:
            self._draw_text(self.feedback_text, self.font_hud, self.feedback_color, self.width / 2, self.height - 116, anchor="midbottom")

    def _draw_overlays(self, now: float) -> None:
        if self.state == "intro":
            rect = pygame.Rect(155, 120, self.width - 310, self.height - 240)
            self._draw_panel(rect, (5, 18, 30, 220), (142, 202, 230))
            body = (
                "A fruit launches from the left or right.\n"
                "Wait until it enters the glowing strike zone.\n"
                "Press LEFT or RIGHT to slice on that side.\n\n"
                "Correct in-window responses usually slice the fruit.\n"
                "On some trials, the sword will swing to the opposite side.\n"
                "That visible wrong swing is the ErrP event of interest.\n\n"
                "Wrong side, bad timing, and missed fruit are marked separately.\n\n"
                "Press SPACE to begin. ESC to quit."
            )
            self._draw_text("ERRP FRUIT NINJA", self.font_title, (233, 241, 246), rect.centerx, rect.top + 54, anchor="midtop")
            self._draw_multiline(body, self.font_body, (199, 214, 225), rect.centerx, rect.top + 132, line_gap=8, anchor="midtop")
            return

        if self.state == "countdown":
            elapsed = now - self.state_started_at
            text = "3" if elapsed < 1.0 else "2" if elapsed < 2.0 else "1" if elapsed < 3.0 else "SLICE"
            self._draw_text(text, self.font_countdown, (255, 240, 187), self.width / 2, self.height / 2 - 10, anchor="center")
            return

        if self.state == "break":
            rect = pygame.Rect(220, 205, self.width - 440, self.height - 410)
            self._draw_panel(rect, (4, 16, 25, 225), (123, 223, 242))
            body = (
                f"Block complete\n\n"
                f"Trials finished: {self.completed_trials}/{self.cfg.trials}\n"
                f"Current score: {self.score:+05d}\n"
                f"Current streak: {self.streak:02d}\n\n"
                "Blink, relax, and reset posture.\n"
                "Press SPACE when ready."
            )
            self._draw_multiline(body, self.font_overlay, (216, 229, 238), rect.centerx, rect.centery, line_gap=10, anchor="center")
            return

        if self.state == "complete":
            rect = pygame.Rect(210, 190, self.width - 420, self.height - 380)
            self._draw_panel(rect, (5, 17, 27, 225), (152, 245, 225))
            body = (
                f"Session complete\n\n"
                f"Trials finished: {self.completed_trials}/{self.cfg.trials}\n"
                f"Final score: {self.score:+05d}\n"
                f"Trial log: {self.logger.trial_path}\n"
                f"Config log: {self.logger.meta_path}\n\n"
                "Press SPACE to close."
            )
            self._draw_multiline(body, self.font_overlay, (231, 245, 239), rect.centerx, rect.centery, line_gap=10, anchor="center")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ErrP fruit slicing task")
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--break-every", type=int, default=20)
    parser.add_argument("--error-prob", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--window-width", type=int, default=1440)
    parser.add_argument("--window-height", type=int, default=900)
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--log-dir", type=str, default=".")
    parser.add_argument("--port", type=str, default=None)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--pulse-width", type=float, default=0.01)
    parser.add_argument("--vid", type=lambda x: int(x, 0), default="0x2341")
    parser.add_argument("--pid", type=lambda x: int(x, 0), default="0x8037")
    parser.add_argument("--no-trigger", action="store_true")
    return parser.parse_args()


def build_session_name() -> str:
    return datetime.now().strftime("errp_fruit_slice_%Y%m%d_%H%M%S")


def main() -> None:
    args = parse_args()
    cfg = TaskConfig(
        trials=args.trials,
        break_every=args.break_every,
        error_prob=args.error_prob,
        window_width=args.window_width,
        window_height=args.window_height,
        fullscreen=args.fullscreen,
        target_fps=args.fps,
        seed=args.seed,
        log_dir=args.log_dir,
    )
    serial_cfg = SerialConfig(
        port=args.port,
        baudrate=args.baudrate,
        pulse_width_s=args.pulse_width,
        vid=args.vid,
        pid=args.pid,
        enabled=not args.no_trigger,
    )
    session_name = build_session_name()

    print(f"[SESSION] {session_name}")
    print(f"[TASK] trials={cfg.trials}, error_prob={cfg.error_prob:.2f}, break_every={cfg.break_every}")
    print("[TRIGGERS]")
    for key, value in asdict(TriggerCodes()).items():
        print(f"  {key}: {value}")

    game = FruitSliceErrPGame(cfg=cfg, serial_cfg=serial_cfg, session_name=session_name)
    game.run()


if __name__ == "__main__":
    main()
