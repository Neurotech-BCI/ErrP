from __future__ import annotations

import logging
import pickle
import random
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pygame
from mne_lsl.stream import StreamLSL

from config import (
    EEGConfig,
    LSLConfig,
    MentalCommandLabelConfig,
    MentalCommandModelConfig,
    MICursorTaskConfig,
    StimConfig,
)
from bci_runtime import apply_runtime_config_overrides, resolve_runtime_jaw_classifier, resolve_shared_mi_model
from derick_ml_jawclench import (
    prepare_jaw_calibration_features,
    select_jaw_channel_indices,
    train_jaw_clench_classifier,
    update_live_jaw_clench_state,
)
from mental_command_worker import (
    StreamingIIRFilter,
    canonicalize_channel_name,
    resolve_channel_order,
)


@dataclass(frozen=True)
class RunnerGameConfig:
    target_fps: int = 120
    lane_move_cooldown_s: float = 0.30
    jump_duration_s: float = 0.65
    jump_height: float = 0.30
    jump_retrigger_cooldown_s: float = 0.50
    move_confidence_thresh: float = 0.58
    base_world_speed: float = 0.42
    cluster_start_depth: float = 0.05
    low_obstacle_depth_offset: float = 0.14
    post_cluster_gap_s: float = 0.60
    open_lane_low_obstacle_prob: float = 0.45
    collision_depth: float = 0.93
    collision_window: float = 0.06
    low_obstacle_jump_clearance: float = 0.10


def _make_task_logger(fname: str) -> logging.Logger:
    logger = logging.getLogger(f"subway_runner.{fname}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(f"{fname}_subway_runner.log", mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def _sanitize(raw: str) -> str:
    cleaned = "_".join(raw.strip().lower().split())
    cleaned = "".join(c for c in cleaned if c.isalnum() or c == "_")
    return cleaned.strip("_")


def _prompt_prefix() -> str:
    while True:
        raw = input("Enter participant name: ")
        p = _sanitize(raw)
        if p:
            return f"{datetime.now().strftime('%m_%d_%y')}_{p}_subway_runner"
        print("Name cannot be empty.")


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class SubwayRunnerGame:
    def __init__(self, fname: str, max_games: int | None = None):
        self.fname = fname
        self.max_games = max_games
        self.logger = _make_task_logger(fname)

        self.lsl_cfg = LSLConfig()
        self.stim_cfg = StimConfig()
        self.label_cfg = MentalCommandLabelConfig()
        self.task_cfg = MICursorTaskConfig()
        self.model_cfg = MentalCommandModelConfig()
        self.eeg_cfg = EEGConfig(
            picks=("Pz", "F4", "C4", "P4", "P3", "C3", "F3"),
            l_freq=8.0,
            h_freq=30.0,
            reject_peak_to_peak=150.0,
        )
        self.game_cfg = RunnerGameConfig()
        cfgs = apply_runtime_config_overrides(
            "subway_runner_task",
            lsl_cfg=self.lsl_cfg,
            stim_cfg=self.stim_cfg,
            label_cfg=self.label_cfg,
            task_cfg=self.task_cfg,
            model_cfg=self.model_cfg,
            eeg_cfg=self.eeg_cfg,
            game_cfg=self.game_cfg,
        )
        self.lsl_cfg = cfgs["lsl_cfg"]
        self.stim_cfg = cfgs["stim_cfg"]
        self.label_cfg = cfgs["label_cfg"]
        self.task_cfg = cfgs["task_cfg"]
        self.model_cfg = cfgs["model_cfg"]
        self.eeg_cfg = cfgs["eeg_cfg"]
        self.game_cfg = cfgs["game_cfg"]

        self.stream = StreamLSL(
            bufsize=60.0,
            name=self.lsl_cfg.name,
            stype=self.lsl_cfg.stype,
            source_id=self.lsl_cfg.source_id,
        )
        self.stream.connect(acquisition_delay=0.001, processing_flags="all")
        self.logger.info("Connected: %s", self.stream.info)

        available = list(self.stream.info["ch_names"])
        self.model_ch_names, missing = resolve_channel_order(available, self.eeg_cfg.picks)
        if len(self.model_ch_names) < 2:
            event_key = canonicalize_channel_name(self.lsl_cfg.event_channels)
            self.model_ch_names = [ch for ch in available if canonicalize_channel_name(ch) != event_key]
        if len(self.model_ch_names) < 2:
            raise RuntimeError(f"Need >=2 EEG channels, found: {available}")

        self.stream.pick(self.model_ch_names)
        self.sfreq = float(self.stream.info["sfreq"])
        self.logger.info(
            "Channels: sfreq=%.1f selected=%s missing=%s",
            self.sfreq,
            list(self.stream.info["ch_names"]),
            missing,
        )

        self._prepare_models()

        pygame.init()
        pygame.font.init()
        flags = pygame.DOUBLEBUF
        if bool(self.task_cfg.fullscreen):
            flags |= pygame.FULLSCREEN
        try:
            self.screen = pygame.display.set_mode(self.task_cfg.win_size, flags, vsync=1)
        except TypeError:
            self.screen = pygame.display.set_mode(self.task_cfg.win_size, flags)
        pygame.display.set_caption("BCI Subway Runner")
        self.clock = pygame.time.Clock()

        self.width, self.height = self.task_cfg.win_size
        self.horizon_y = 150
        self.bottom_y = self.height - 60
        self.road_half_top = 150
        self.road_half_bottom = 530
        self.player_y = self.height - 165

        self.font_title = pygame.font.SysFont("arial", 36, bold=True)
        self.font_hud = pygame.font.SysFont("arial", 22, bold=True)
        self.font_overlay = pygame.font.SysFont("arial", 24)

        self.running = True
        self.score = 0
        self.last_live_ts: float | None = None

        self.live_filter = StreamingIIRFilter.from_eeg_config(
            eeg_cfg=self.eeg_cfg,
            sfreq=self.sfreq,
            n_channels=len(self.model_ch_names),
        )
        self.keep_n = int(round((self.task_cfg.window_s + self.task_cfg.filter_context_s) * self.sfreq))
        self.window_n = int(round(self.task_cfg.window_s * self.sfreq))
        self.stream_pull_s = max(0.10, self.task_cfg.live_update_interval_s * 2.0)

        self.live_buffer = np.empty((len(self.model_ch_names), 0), dtype=np.float32)
        self.jaw_buffer = np.empty((len(self.model_ch_names), 0), dtype=np.float32)

        self.pred_clock = pygame.time.Clock()
        self.pred_elapsed_s = 0.0
        self.prediction_count = 0
        self.left_prob = 0.5
        self.right_prob = 0.5
        self.raw_command = 0.0
        self.ema_command = 0.0
        self.live_note = "warming up"
        self.latest_pred_code: int | None = None
        self.bias_offset = float(self.task_cfg.live_bias_offset) if bool(self.task_cfg.enable_live_bias_offset) else 0.0

        self.jaw_prob = 0.0
        self.jaw_prev_pred = 0
        self.jaw_event_pending = False
        self.jaw_last_event_t = -1e9
        self.jaw_prob_thresh = float(self.task_cfg.jaw_clench_prob_thresh)
        self.jaw_refractory_s = max(
            float(self.task_cfg.jaw_clench_refractory_s),
            float(self.game_cfg.jump_retrigger_cooldown_s),
        )

        self.rng = random.Random(42)

    def _prepare_models(self) -> None:
        self.logger.info(
            "Preparing MI model: data_dir=%s edf_glob=%s window_s=%.2f step_s=%.2f",
            self.task_cfg.data_dir,
            self.task_cfg.edf_glob,
            self.task_cfg.window_s,
            self.task_cfg.window_step_s,
        )
        shared_model = resolve_shared_mi_model(
            cache_name="mi_shared_lr_model",
            data_dir=self.task_cfg.data_dir,
            edf_glob=self.task_cfg.edf_glob,
            calibrate_on_participant=self.task_cfg.calirate_on_participant,
            eeg_cfg=self.eeg_cfg,
            task_cfg=self.task_cfg,
            stim_cfg=self.stim_cfg,
            model_cfg=self.model_cfg,
            target_sfreq=float(self.sfreq),
            target_channel_names=self.model_ch_names,
            logger=self.logger,
        )
        self.classifier = shared_model.classifier
        self.class_index = shared_model.class_index

        classes = np.asarray(self.classifier.named_steps["clf"].classes_, dtype=int)
        if int(self.stim_cfg.left_code) not in self.class_index or int(self.stim_cfg.right_code) not in self.class_index:
            raise RuntimeError(
                f"Classifier classes {classes.tolist()} do not contain left/right codes "
                f"{[int(self.stim_cfg.left_code), int(self.stim_cfg.right_code)]}."
            )

        with open(f"{self.fname}_subway_runner_model.pkl", "wb") as fh:
            pickle.dump(self.classifier, fh)

        self.jaw_idxs = select_jaw_channel_indices(self.model_ch_names)
        self.jaw_window_n = int(round(float(self.task_cfg.jaw_window_s) * self.sfreq))
        self._calibrate_jaw_classifier()

    def _poll_quit(self) -> bool:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return True
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                return True
        return False

    def _draw_wait_screen(self, cue: str, status: str, info: str = "") -> None:
        self.screen.fill((12, 18, 26))
        self._draw_road()
        self._draw_text(cue, self.font_title, (236, 242, 247), self.width // 2, 110, "midtop")
        self._draw_text(status, self.font_overlay, (212, 225, 234), self.width // 2, self.height - 150, "midtop")
        if info:
            self._draw_text(info, self.font_overlay, (148, 209, 236), self.width // 2, self.height - 110, "midtop")
        pygame.display.flip()

    def _wait_for_space(self, cue: str, status: str, info: str = "") -> None:
        while self.running:
            if self._poll_quit():
                raise KeyboardInterrupt
            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_SPACE]:
                return
            self._draw_wait_screen(cue, status, info)
            self.clock.tick(self.game_cfg.target_fps)

    def _collect_stream_block(self, duration_s: float, cue: str, info_text: str) -> np.ndarray:
        chunks: list[np.ndarray] = []
        last_ts_local: float | None = None
        start = pygame.time.get_ticks() / 1000.0

        while (pygame.time.get_ticks() / 1000.0) - start < duration_s:
            if self._poll_quit():
                raise KeyboardInterrupt

            data, ts = self.stream.get_data(winsize=min(0.20, duration_s), picks="all")
            if data.size > 0 and ts is not None and len(ts) > 0:
                ts_arr = np.asarray(ts)
                mask = np.ones_like(ts_arr, dtype=bool) if last_ts_local is None else (ts_arr > float(last_ts_local))
                if np.any(mask):
                    chunks.append(np.asarray(data[:, mask], dtype=np.float32))
                    last_ts_local = float(ts_arr[mask][-1])

            elapsed = (pygame.time.get_ticks() / 1000.0) - start
            self._draw_wait_screen(cue, f"{max(0.0, duration_s - elapsed):0.1f}s", info_text)
            self.clock.tick(self.game_cfg.target_fps)

        if not chunks:
            return np.empty((len(self.model_ch_names), 0), dtype=np.float32)
        return np.concatenate(chunks, axis=1).astype(np.float32, copy=False)

    def _calibrate_jaw_classifier(self) -> None:
        runtime_jaw_classifier, runtime_train_acc = resolve_runtime_jaw_classifier(logger=self.logger, min_total_samples=12)
        if runtime_jaw_classifier is not None:
            self.jaw_classifier = runtime_jaw_classifier
            self.logger.info("Using orchestrator-provided jaw calibration (train_acc=%.3f).", float(runtime_train_acc or 0.0))
            return

        self._wait_for_space(
            "Jaw Calibration",
            "Press SPACE to begin.",
            "Collecting REST and JAW CLENCH blocks for jump detection.",
        )

        n_per_class = int(self.task_cfg.jaw_calibration_blocks_per_class)
        hold_s = float(self.task_cfg.jaw_calibration_hold_s)
        prep_s = float(self.task_cfg.jaw_calibration_prep_s)
        iti_s = float(self.task_cfg.jaw_calibration_iti_s)
        trim_s = float(self.task_cfg.jaw_calibration_trim_s)
        min_samples = self.jaw_window_n + 2 * int(round(trim_s * self.sfreq))

        labels = [0] * n_per_class + [1] * n_per_class
        random.shuffle(labels)
        calib_blocks: list[np.ndarray] = []
        calib_labels: list[int] = []

        for i, y_label in enumerate(labels, start=1):
            is_clench = bool(y_label == 1)
            trial_name = "JAW CLENCH" if is_clench else "REST"

            prep_start = pygame.time.get_ticks() / 1000.0
            while (pygame.time.get_ticks() / 1000.0) - prep_start < prep_s:
                if self._poll_quit():
                    raise KeyboardInterrupt
                remaining = prep_s - ((pygame.time.get_ticks() / 1000.0) - prep_start)
                self._draw_wait_screen("Prepare", f"{remaining:0.1f}s", f"Next: {trial_name} ({i}/{len(labels)})")
                self.clock.tick(self.game_cfg.target_fps)

            info = "Clench jaw and hold." if is_clench else "Relax face and avoid movement."
            block = self._collect_stream_block(hold_s, trial_name, info)
            if block.shape[1] >= min_samples:
                calib_blocks.append(block)
                calib_labels.append(int(y_label))
            else:
                self.logger.warning(
                    "Skipping short jaw calibration block %d: samples=%d needed=%d",
                    i,
                    int(block.shape[1]),
                    min_samples,
                )

            iti_start = pygame.time.get_ticks() / 1000.0
            while (pygame.time.get_ticks() / 1000.0) - iti_start < iti_s:
                if self._poll_quit():
                    raise KeyboardInterrupt
                remaining = iti_s - ((pygame.time.get_ticks() / 1000.0) - iti_start)
                self._draw_wait_screen("Relax", f"{remaining:0.1f}s", "Short break")
                self.clock.tick(self.game_cfg.target_fps)

        X_np, y_np = prepare_jaw_calibration_features(
            blocks=calib_blocks,
            labels=calib_labels,
            jaw_idxs=self.jaw_idxs,
            sfreq=float(self.sfreq),
            window_s=float(self.task_cfg.jaw_window_s),
            step_s=float(self.task_cfg.jaw_window_step_s),
            edge_trim_s=float(self.task_cfg.jaw_calibration_trim_s),
        )
        self.jaw_classifier, train_acc, _X_np, y_np = train_jaw_clench_classifier(
            feature_rows=X_np,
            labels=y_np,
            min_total_samples=12,
        )
        self.logger.info(
            "Jaw calibration ready: windows=%d rest=%d clench=%d train_acc=%.3f",
            int(len(y_np)),
            int(np.sum(y_np == 0)),
            int(np.sum(y_np == 1)),
            float(train_acc),
        )

        self._wait_for_space(
            "Calibration Complete",
            "Press SPACE to start running.",
            f"Jaw train acc {float(train_acc):.2f}",
        )

    def _road_bounds(self, depth: float) -> tuple[float, float, float]:
        d = clamp(depth, 0.0, 1.0)
        y = self.horizon_y + (self.bottom_y - self.horizon_y) * d
        hw = self.road_half_top + (self.road_half_bottom - self.road_half_top) * d
        return self.width / 2, y, hw

    def _lane_screen_x(self, lane: int, depth: float) -> float:
        center, _, hw = self._road_bounds(depth)
        return center + float(lane) * hw * 0.62

    def _draw_road(self) -> None:
        center = self.width / 2
        top_left = (center - self.road_half_top, self.horizon_y)
        top_right = (center + self.road_half_top, self.horizon_y)
        bottom_right = (center + self.road_half_bottom, self.bottom_y)
        bottom_left = (center - self.road_half_bottom, self.bottom_y)

        pygame.draw.polygon(self.screen, (29, 34, 40), [top_left, top_right, bottom_right, bottom_left])
        pygame.draw.lines(self.screen, (84, 97, 107), True, [top_left, top_right, bottom_right, bottom_left], 2)

        pygame.draw.line(
            self.screen,
            (245, 240, 213),
            (self._lane_screen_x(-1, 0.0), self.horizon_y),
            (self._lane_screen_x(-1, 1.0), self.bottom_y),
            2,
        )
        pygame.draw.line(
            self.screen,
            (245, 240, 213),
            (self._lane_screen_x(1, 0.0), self.horizon_y),
            (self._lane_screen_x(1, 1.0), self.bottom_y),
            2,
        )

    def _draw_text(
        self,
        text: str,
        font: pygame.font.Font,
        color: tuple[int, int, int],
        x: float,
        y: float,
        anchor: str = "topleft",
    ) -> None:
        surf = font.render(text, True, color)
        rect = surf.get_rect()
        if anchor == "midtop":
            rect.midtop = (x, y)
        elif anchor == "center":
            rect.center = (x, y)
        elif anchor == "topright":
            rect.topright = (x, y)
        else:
            rect.topleft = (x, y)
        self.screen.blit(surf, rect)

    def _spawn_cluster(self, obstacles: list[dict]) -> None:
        blocked_side = self.rng.choice((-1, 1))
        open_lane = -blocked_side

        start_depth = float(self.game_cfg.cluster_start_depth)
        obstacles.append({"lane": 0, "y": start_depth, "kind": "high", "cleared": False})
        obstacles.append({"lane": int(blocked_side), "y": start_depth, "kind": "high", "cleared": False})

        if self.rng.random() < float(self.game_cfg.open_lane_low_obstacle_prob):
            obstacles.append({
                "lane": int(open_lane),
                "y": start_depth + float(self.game_cfg.low_obstacle_depth_offset),
                "kind": "low",
                "cleared": False,
            })

    def _reset_decoder_state(self) -> None:
        self.live_filter.reset()
        self.live_buffer = np.empty((len(self.model_ch_names), 0), np.float32)
        self.jaw_buffer = np.empty((len(self.model_ch_names), 0), np.float32)
        self.last_live_ts = None
        self.pred_elapsed_s = 0.0
        self.prediction_count = 0
        self.left_prob = 0.5
        self.right_prob = 0.5
        self.raw_command = 0.0
        self.ema_command = 0.0
        self.live_note = "warming up"
        self.latest_pred_code = None
        self.jaw_prob = 0.0
        self.jaw_prev_pred = 0
        self.jaw_event_pending = False
        self.jaw_last_event_t = -1e9

    def _pull_bci(self, now_s: float, dt_s: float) -> None:
        data, ts = self.stream.get_data(winsize=self.stream_pull_s, picks="all")
        if data.size > 0 and ts is not None and len(ts) > 0:
            ts_arr = np.asarray(ts)
            mask = np.ones_like(ts_arr, dtype=bool) if self.last_live_ts is None else (ts_arr > float(self.last_live_ts))
            if np.any(mask):
                x_new = np.asarray(data[:, mask], dtype=np.float32)
                self.last_live_ts = float(ts_arr[mask][-1])

                self.jaw_buffer, self.jaw_prob, self.jaw_prev_pred, should_jump = update_live_jaw_clench_state(
                    jaw_buffer=self.jaw_buffer,
                    x_new=x_new,
                    keep_n=self.keep_n,
                    jaw_window_n=self.jaw_window_n,
                    jaw_classifier=self.jaw_classifier,
                    jaw_idxs=self.jaw_idxs,
                    jaw_prob=self.jaw_prob,
                    jaw_prev_pred=self.jaw_prev_pred,
                    jaw_prob_thresh=self.jaw_prob_thresh,
                    jaw_last_toggle_t=self.jaw_last_event_t,
                    jaw_refractory_s=self.jaw_refractory_s,
                    now_t=now_s,
                )
                if should_jump:
                    self.jaw_last_event_t = now_s
                    self.jaw_event_pending = True

                x_new_filt = self.live_filter.process(x_new)
                self.live_buffer = np.concatenate((self.live_buffer, x_new_filt), axis=1)
                if self.live_buffer.shape[1] > self.keep_n:
                    self.live_buffer = self.live_buffer[:, -self.keep_n:]

        self.pred_elapsed_s += dt_s
        if self.pred_elapsed_s < float(self.task_cfg.live_update_interval_s):
            return

        self.pred_elapsed_s = 0.0
        if self.live_buffer.shape[1] < self.window_n:
            needed_s = max(0.0, (self.window_n - self.live_buffer.shape[1]) / self.sfreq)
            self.live_note = f"warming up ({needed_s:.1f}s)"
            self.latest_pred_code = None
            return

        x_win = self.live_buffer[:, -self.window_n:]
        max_ptp = float(np.ptp(x_win, axis=-1).max())
        if self.eeg_cfg.reject_peak_to_peak is not None and max_ptp > float(self.eeg_cfg.reject_peak_to_peak):
            self.raw_command = 0.0
            self.ema_command *= 0.85
            self.latest_pred_code = None
            self.live_note = "artifact reject"
            return

        p_vec = self.classifier.predict_proba(x_win[np.newaxis, ...])[0]
        self.left_prob = float(p_vec[self.class_index[int(self.stim_cfg.left_code)]])
        self.right_prob = float(p_vec[self.class_index[int(self.stim_cfg.right_code)]])
        self.raw_command = float(np.clip(self.right_prob - self.left_prob + self.bias_offset, -1.0, 1.0))

        alpha = float(np.clip(self.task_cfg.command_ema_alpha, 0.0, 1.0))
        self.ema_command = self.raw_command if self.prediction_count == 0 else ((1.0 - alpha) * self.ema_command + alpha * self.raw_command)

        if abs(self.ema_command) < float(self.task_cfg.command_deadband):
            self.latest_pred_code = None
        elif self.ema_command > 0:
            self.latest_pred_code = int(self.stim_cfg.right_code)
        else:
            self.latest_pred_code = int(self.stim_cfg.left_code)

        self.prediction_count += 1
        self.live_note = "tracking"

    def _poll_jump_event(self) -> bool:
        if self.jaw_event_pending:
            self.jaw_event_pending = False
            return True
        return False

    def _draw_game_frame(
        self,
        lane_idx: int,
        jump_offset: float,
        obstacles: list[dict],
        score: int,
        speed_mult: float,
        status_text: str,
    ) -> None:
        self.screen.fill((10, 15, 22))
        self._draw_road()

        for ob in obstacles:
            y = float(ob["y"])
            lane = int(ob["lane"])
            kind = str(ob["kind"])
            x = self._lane_screen_x(lane, y)
            _, y_px, hw = self._road_bounds(y)
            lane_w = hw * 1.24 / 3.0

            if kind == "high":
                w = max(44.0, lane_w * 0.86)
                h = max(26.0, lane_w * 0.52)
                color = (215, 117, 54)
            else:
                w = max(36.0, lane_w * 0.74)
                h = max(14.0, lane_w * 0.24)
                color = (231, 198, 78)

            rect = pygame.Rect(0, 0, int(w), int(h))
            rect.center = (int(x), int(y_px))
            pygame.draw.rect(self.screen, color, rect, border_radius=8)
            pygame.draw.rect(self.screen, (248, 245, 234), rect, width=2, border_radius=8)

        player_x = self._lane_screen_x(lane_idx, 1.0)
        player_y = self.player_y - jump_offset * 230.0
        pygame.draw.circle(self.screen, (53, 204, 238), (int(player_x), int(player_y)), 28)
        pygame.draw.circle(self.screen, (245, 245, 245), (int(player_x), int(player_y)), 28, width=3)

        self._draw_text("BCI Subway Runner", self.font_title, (236, 242, 247), self.width // 2, 20, "midtop")
        self._draw_text(f"Score {score:05d}", self.font_hud, (237, 242, 246), self.width - 30, 26, "topright")
        self._draw_text(f"Speed {speed_mult:.2f}x", self.font_hud, (168, 204, 222), self.width - 30, 56, "topright")

        bci_text = (
            f"{self.label_cfg.left_name}: {self.left_prob:.2f}   "
            f"{self.label_cfg.right_name}: {self.right_prob:.2f}   "
            f"jaw={self.jaw_prob:.2f}   cmd={self.ema_command:+.2f}"
        )
        self._draw_text(bci_text, self.font_overlay, (148, 209, 236), self.width // 2, self.height - 95, "midtop")
        self._draw_text(status_text, self.font_overlay, (211, 223, 231), self.width // 2, self.height - 62, "midtop")

        pygame.display.flip()

    def _show_game_over(self, score: int, survived_s: float, jumps: int) -> bool:
        while self.running:
            if self._poll_quit():
                return False
            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_SPACE]:
                return True

            self.screen.fill((8, 12, 18))
            self._draw_road()
            self._draw_text("GAME OVER", self.font_title, (248, 124, 124), self.width // 2, self.height // 2 - 130, "center")
            self._draw_text(f"Score: {score}", self.font_overlay, (235, 241, 246), self.width // 2, self.height // 2 - 52, "center")
            self._draw_text(f"Time: {survived_s:.1f}s", self.font_overlay, (235, 241, 246), self.width // 2, self.height // 2 - 22, "center")
            self._draw_text(f"Jumps: {jumps}", self.font_overlay, (235, 241, 246), self.width // 2, self.height // 2 + 8, "center")
            self._draw_text("SPACE to play again   ESC to quit", self.font_overlay, (184, 206, 220), self.width // 2, self.height // 2 + 70, "center")
            pygame.display.flip()
            self.clock.tick(self.game_cfg.target_fps)
        return False

    def run(self) -> None:
        self._wait_for_space(
            "MI Subway Runner",
            "Press SPACE to start. ESC to quit.",
            "MI left/right changes lane. Jaw clench jumps over low obstacles.",
        )

        games_completed = 0
        while self.running:
            if self.max_games is not None and games_completed >= int(self.max_games):
                self.logger.info("Reached max_games=%d; ending task.", int(self.max_games))
                break
            self._reset_decoder_state()

            lane_idx = 0
            last_lane_move_t = -999.0
            jump_active = False
            jump_start_t = -999.0
            last_jump_trigger_t = -999.0
            jump_count = 0

            obstacles: list[dict] = []
            next_cluster_ready_t = game_start + 0.85
            score = 0
            passed = 0
            game_over = False

            game_start = pygame.time.get_ticks() / 1000.0
            last_frame_t = game_start

            while not game_over and self.running:
                if self._poll_quit():
                    raise KeyboardInterrupt

                now_t = pygame.time.get_ticks() / 1000.0
                dt = max(0.0, min(0.05, now_t - last_frame_t))
                last_frame_t = now_t
                elapsed = now_t - game_start

                self._pull_bci(now_s=now_t, dt_s=dt)

                conf = max(self.left_prob, self.right_prob)
                if self.latest_pred_code is not None and conf >= float(self.game_cfg.move_confidence_thresh):
                    if (now_t - last_lane_move_t) >= float(self.game_cfg.lane_move_cooldown_s):
                        if self.latest_pred_code == int(self.stim_cfg.left_code) and lane_idx > -1:
                            lane_idx -= 1
                            last_lane_move_t = now_t
                        elif self.latest_pred_code == int(self.stim_cfg.right_code) and lane_idx < 1:
                            lane_idx += 1
                            last_lane_move_t = now_t

                if self._poll_jump_event() and (now_t - last_jump_trigger_t) >= float(self.game_cfg.jump_retrigger_cooldown_s):
                    if not jump_active:
                        jump_active = True
                        jump_start_t = now_t
                        jump_count += 1
                    last_jump_trigger_t = now_t

                jump_offset = 0.0
                if jump_active:
                    phase = (now_t - jump_start_t) / float(self.game_cfg.jump_duration_s)
                    if phase >= 1.0:
                        jump_active = False
                    else:
                        jump_offset = float(self.game_cfg.jump_height) * float(np.sin(np.pi * phase))

                speed_mult = 1.0 + 0.35 * min(1.5, elapsed / 60.0)
                world_speed = float(self.game_cfg.base_world_speed) * speed_mult

                if (not obstacles) and now_t >= next_cluster_ready_t:
                    self._spawn_cluster(obstacles)
                    next_cluster_ready_t = float("inf")

                remove_count = 0
                for ob in obstacles:
                    ob["y"] = float(ob["y"]) + world_speed * dt
                while obstacles and float(obstacles[0]["y"]) > 1.10:
                    obstacles.pop(0)
                    remove_count += 1
                if remove_count > 0:
                    passed += remove_count
                if not obstacles and next_cluster_ready_t == float("inf"):
                    next_cluster_ready_t = now_t + float(self.game_cfg.post_cluster_gap_s)

                score = int(elapsed * 12.0 + passed * 8)

                for ob in obstacles:
                    if bool(ob.get("cleared", False)):
                        continue
                    if int(ob["lane"]) != lane_idx:
                        continue
                    if abs(float(ob["y"]) - float(self.game_cfg.collision_depth)) > float(self.game_cfg.collision_window):
                        continue

                    kind = str(ob["kind"])
                    if kind == "low":
                        if jump_offset < float(self.game_cfg.low_obstacle_jump_clearance):
                            game_over = True
                            break
                        ob["cleared"] = True
                    else:
                        game_over = True
                        break

                status_text = (
                    f"Lane: {lane_idx + 2}/3   Obstacles: {len(obstacles)}   "
                    f"Predictions: {self.prediction_count}   Jumps: {jump_count}   {self.live_note}"
                )
                self._draw_game_frame(
                    lane_idx=lane_idx,
                    jump_offset=jump_offset,
                    obstacles=obstacles,
                    score=score,
                    speed_mult=speed_mult,
                    status_text=status_text,
                )
                self.clock.tick(self.game_cfg.target_fps)

            if not self.running:
                break

            survived_s = (pygame.time.get_ticks() / 1000.0) - game_start
            self.logger.info(
                "Game over: score=%d survived_s=%.2f jumps=%d predictions=%d",
                int(score),
                float(survived_s),
                int(jump_count),
                int(self.prediction_count),
            )
            games_completed += 1
            if not self._show_game_over(score=int(score), survived_s=float(survived_s), jumps=int(jump_count)):
                break

    def shutdown(self) -> None:
        try:
            self.stream.disconnect()
        except Exception:
            pass
        pygame.quit()


def run_task(fname: str, max_trials: int | None = None) -> None:
    game = SubwayRunnerGame(fname=fname, max_games=max_trials)
    try:
        game.run()
    except KeyboardInterrupt:
        pass
    finally:
        game.shutdown()


if __name__ == "__main__":
    prefix = _prompt_prefix()
    print(f"[SESSION] prefix: {prefix}")
    run_task(fname=prefix)
