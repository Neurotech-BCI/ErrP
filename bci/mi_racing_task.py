from __future__ import annotations

import logging
import pickle
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
from mne_lsl.stream import StreamLSL
from psychopy import core, event, visual

from bci_runtime import apply_runtime_config_overrides, resolve_shared_mi_model
from config import (
    EEGConfig,
    LSLConfig,
    MIRacingTaskConfig,
    MentalCommandLabelConfig,
    MentalCommandModelConfig,
    StimConfig,
)
from mental_command_worker import StreamingIIRFilter, resolve_channel_order


@dataclass
class ObstacleState:
    obstacle_id: int
    lane_idx: int
    y: float


def _make_task_logger(fname: str) -> logging.Logger:
    logger = logging.getLogger(f"mi_racing.{fname}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    file_handler = logging.FileHandler(f"{fname}_mi_racing.log", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _sanitize_participant_name(raw_name: str) -> str:
    cleaned = "_".join(raw_name.strip().lower().split())
    cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch == "_")
    return cleaned.strip("_")


def _build_session_prefix(participant: str) -> str:
    return f"{datetime.now().strftime('%m_%d_%y')}_{participant}_mi_racing"


def _prompt_session_prefix() -> str:
    while True:
        raw = input("Enter participant name: ")
        participant = _sanitize_participant_name(raw)
        if participant:
            return _build_session_prefix(participant)
        print("Participant name cannot be empty. Please try again.")


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def run_task(fname: str, max_trials: int | None = None) -> None:
    logger = _make_task_logger(fname)

    lsl_cfg = LSLConfig()
    stim_cfg = StimConfig()
    label_cfg = MentalCommandLabelConfig()
    task_cfg = MIRacingTaskConfig()
    model_cfg = MentalCommandModelConfig()
    eeg_cfg = EEGConfig(
        l_freq=8.0,
        h_freq=30.0,
        reject_peak_to_peak=150.0,
    )
    cfgs = apply_runtime_config_overrides(
        "mi_racing_task",
        lsl_cfg=lsl_cfg,
        stim_cfg=stim_cfg,
        label_cfg=label_cfg,
        task_cfg=task_cfg,
        model_cfg=model_cfg,
        eeg_cfg=eeg_cfg,
    )
    lsl_cfg = cfgs["lsl_cfg"]
    stim_cfg = cfgs["stim_cfg"]
    label_cfg = cfgs["label_cfg"]
    task_cfg = cfgs["task_cfg"]
    model_cfg = cfgs["model_cfg"]
    eeg_cfg = cfgs["eeg_cfg"]

    rng = np.random.default_rng(task_cfg.seed)

    target_obstacles = int(max_trials) if max_trials is not None else int(task_cfg.trials_to_complete)
    if target_obstacles <= 0:
        raise ValueError("trials_to_complete must be >= 1")

    stream = StreamLSL(
        bufsize=60.0,
        name=lsl_cfg.name,
        stype=lsl_cfg.stype,
        source_id=lsl_cfg.source_id,
    )
    stream.connect(acquisition_delay=0.001, processing_flags="all")
    logger.info("Connected to LSL stream: info=%s", stream.info)

    available = list(stream.info["ch_names"])
    model_ch_names, missing = resolve_channel_order(available, eeg_cfg.picks)
    if missing:
        raise RuntimeError(
            f"Live stream is missing configured EEG picks {missing}. "
            f"Configured picks: {list(eeg_cfg.picks)}. Available channels: {available}"
        )
    if len(model_ch_names) < 2:
        raise RuntimeError(
            "Need at least 2 configured EEG channels after applying picks. "
            f"Configured picks: {list(eeg_cfg.picks)}. Resolved channels: {model_ch_names}. "
            f"Available channels: {available}"
        )

    stream.pick(model_ch_names)
    sfreq = float(stream.info["sfreq"])
    stream_ch_names = list(stream.info["ch_names"])
    logger.info(
        "Using live EEG channels: sfreq=%.3f, selected=%s, missing_configured=%s",
        sfreq,
        stream_ch_names,
        missing,
    )

    win = visual.Window(
        size=task_cfg.win_size,
        color=(-0.12, -0.10, -0.08),
        units="norm",
        fullscr=task_cfg.fullscreen,
    )
    if win.size[1] > 0:
        view_scale_x = float(win.size[1]) / float(win.size[0])
        win.viewScale = (view_scale_x, 1.0)
        logger.info("Applied display aspect correction: viewScale=(%.4f, 1.0000)", view_scale_x)

    title = visual.TextStim(win, text="MI Racing Task", pos=(0, 0.92), height=0.055, color=(0.92, 0.92, 0.92))
    cue = visual.TextStim(win, text="", pos=(0, 0.84), height=0.045, color=(0.90, 0.90, 0.90))
    info = visual.TextStim(win, text="", pos=(0, -0.84), height=0.040, color=(0.82, 0.90, 0.96))
    status = visual.TextStim(win, text="", pos=(0, -0.92), height=0.035, color=(0.84, 0.84, 0.84))
    scoreboard = visual.TextStim(win, text="", pos=(0, 0.73), height=0.040, color=(0.94, 0.94, 0.94))

    road = visual.Rect(
        win,
        width=float(task_cfg.road_width),
        height=float(task_cfg.road_height),
        pos=(0.0, 0.0),
        fillColor=(-0.65, -0.65, -0.65),
        lineColor=(-0.15, -0.15, -0.15),
        lineWidth=2.0,
    )
    lane_divider_1 = visual.Rect(
        win,
        width=float(task_cfg.lane_divider_width),
        height=float(task_cfg.road_height),
        pos=(0.0, 0.0),
        fillColor=(-0.05, -0.05, -0.05),
        lineColor=None,
    )
    lane_divider_2 = visual.Rect(
        win,
        width=float(task_cfg.lane_divider_width),
        height=float(task_cfg.road_height),
        pos=(0.0, 0.0),
        fillColor=(-0.05, -0.05, -0.05),
        lineColor=None,
    )

    car = visual.Rect(
        win,
        width=float(task_cfg.car_width),
        height=float(task_cfg.car_height),
        pos=(0.0, float(task_cfg.car_y)),
        fillColor=(-0.15, 0.55, 0.95),
        lineColor=(0.9, 0.9, 0.9),
        lineWidth=1.6,
    )
    obstacle_stim = visual.Rect(
        win,
        width=float(task_cfg.obstacle_width),
        height=float(task_cfg.obstacle_height),
        pos=(0.0, 0.0),
        fillColor=(0.95, -0.2, -0.2),
        lineColor=(0.96, 0.9, 0.7),
        lineWidth=1.4,
    )

    lane_span = float(task_cfg.road_width) / 3.0
    lane_centers = np.array([-lane_span, 0.0, lane_span], dtype=np.float64)
    lane_divider_1.pos = ((lane_centers[0] + lane_centers[1]) / 2.0, 0.0)
    lane_divider_2.pos = ((lane_centers[1] + lane_centers[2]) / 2.0, 0.0)

    classifier = None
    class_index: dict[int, int] | None = None

    obstacles: list[ObstacleState] = []

    def _draw_world() -> None:
        road.draw()
        lane_divider_1.draw()
        lane_divider_2.draw()
        for obs in obstacles:
            obstacle_stim.pos = (float(lane_centers[obs.lane_idx]), float(obs.y))
            obstacle_stim.draw()
        car.draw()
        title.draw()
        scoreboard.draw()
        cue.draw()
        info.draw()
        status.draw()
        win.flip()

    def _draw_preload_screen(cue_text: str, status_text: str) -> None:
        scoreboard.text = ""
        cue.text = cue_text
        info.text = ""
        status.text = status_text
        _draw_world()

    _draw_preload_screen(
        cue_text="Preparing MI model from offline EDF sessions...",
        status_text="Please wait.",
    )

    try:
        shared_model = resolve_shared_mi_model(
            cache_name="mi_shared_lr_model",
            data_dir=task_cfg.data_dir,
            edf_glob=task_cfg.edf_glob,
            calibrate_on_participant=task_cfg.calirate_on_participant,
            eeg_cfg=eeg_cfg,
            task_cfg=task_cfg,
            stim_cfg=stim_cfg,
            model_cfg=model_cfg,
            target_sfreq=float(sfreq),
            target_channel_names=model_ch_names,
            logger=logger,
        )
        classifier = shared_model.classifier
        class_index = shared_model.class_index
        loso = shared_model.loso
        clf_classes = np.asarray(classifier.named_steps["clf"].classes_, dtype=int)
        if int(stim_cfg.left_code) not in class_index or int(stim_cfg.right_code) not in class_index:
            raise RuntimeError(
                f"Classifier classes {clf_classes.tolist()} do not contain expected left/right codes "
                f"{[int(stim_cfg.left_code), int(stim_cfg.right_code)]}."
            )
        counts = Counter(shared_model.class_counts)
        logger.info(
            "Shared MI model ready: class_counts=%s, loso_mean=%.4f, loso_std=%.4f, cache=%s",
            counts,
            loso.mean_accuracy,
            loso.std_accuracy,
            shared_model.cache_path,
        )

        if shared_model.dataset is not None:
            np.save(f"{fname}_mi_racing_windows.npy", shared_model.dataset.X)
            np.save(f"{fname}_mi_racing_labels.npy", shared_model.dataset.y)

        with open(f"{fname}_mi_racing_model.pkl", "wb") as fh:
            pickle.dump(classifier, fh)

        rest_count_text = ""
        if int(task_cfg.rest_class_code) in counts:
            rest_count_text = f"  {label_cfg.rest_name}: {counts[int(task_cfg.rest_class_code)]}"

        session_lines = [f"S{sid}: {score:.3f}" for sid, score in sorted(loso.session_scores.items())]
        status_text = (
            f"{label_cfg.left_name}: {counts[int(stim_cfg.left_code)]}  "
            f"{label_cfg.right_name}: {counts[int(stim_cfg.right_code)]}"
            f"{rest_count_text}\n"
            f"{'  '.join(session_lines)}"
        )
        _draw_preload_screen(
            cue_text="Model ready",
            status_text=status_text,
        )
    except Exception:
        try:
            stream.disconnect()
        except Exception:
            pass
        try:
            win.close()
        except Exception:
            pass
        raise

    live_filter = StreamingIIRFilter.from_eeg_config(
        eeg_cfg=eeg_cfg,
        sfreq=sfreq,
        n_channels=len(model_ch_names),
    )
    live_buffer = np.empty((len(model_ch_names), 0), dtype=np.float32)
    keep_n = int(round((float(task_cfg.window_s) + float(task_cfg.filter_context_s)) * sfreq))
    window_n = int(round(float(task_cfg.window_s) * sfreq))
    stream_pull_s = max(0.08, float(task_cfg.live_update_interval_s) * float(task_cfg.stream_pull_multiplier))
    reject_thresh = eeg_cfg.reject_peak_to_peak
    last_live_ts: float | None = None

    pred_clock = core.Clock()
    prediction_count = 0
    left_prob = 0.5
    right_prob = 0.5
    rest_prob = 0.0
    raw_command = 0.0
    ema_command = 0.0
    latest_pred_code: int | None = None
    live_note = "warming up"

    obstacle_counter = 0
    spawn_timer = 0.0
    last_spawn_lane: int | None = None

    target_lane_idx = 1
    lane_state = 1.0
    last_switch_t = -1e9

    score = 0
    dodges = 0
    collisions = 0
    resolved_obstacles = 0
    feedback_text = ""
    feedback_color = (0.92, 0.92, 0.92)
    feedback_until = -1e9
    events_log: list[dict[str, object]] = []

    bias_offset = float(task_cfg.live_bias_offset) if bool(task_cfg.enable_live_bias_offset) else 0.0
    use_discrete_mode = str(task_cfg.mi_control_mode).lower() == "discrete_sign"

    def _poll_live_decoder() -> None:
        nonlocal last_live_ts, live_buffer, prediction_count
        nonlocal left_prob, right_prob, rest_prob, raw_command, ema_command, latest_pred_code, live_note

        data, ts = stream.get_data(winsize=stream_pull_s, picks="all")
        if data.size > 0 and ts is not None and len(ts) > 0:
            ts_arr = np.asarray(ts)
            mask = np.ones_like(ts_arr, dtype=bool) if last_live_ts is None else (ts_arr > float(last_live_ts))
            if np.any(mask):
                x_new = np.asarray(data[:, mask], dtype=np.float32)
                last_live_ts = float(ts_arr[mask][-1])
                x_new_filt = live_filter.process(x_new)
                live_buffer = np.concatenate((live_buffer, x_new_filt), axis=1)
                if live_buffer.shape[1] > keep_n:
                    live_buffer = live_buffer[:, -keep_n:]

        if pred_clock.getTime() < float(task_cfg.live_update_interval_s):
            return

        pred_clock.reset()

        if live_buffer.shape[1] < window_n:
            needed_s = max(0.0, (window_n - live_buffer.shape[1]) / sfreq)
            live_note = f"warming up ({needed_s:.1f}s)"
            latest_pred_code = None
            return

        x_win = live_buffer[:, -window_n:]
        max_ptp = float(np.ptp(x_win, axis=-1).max())
        if reject_thresh is not None and max_ptp > float(reject_thresh):
            live_note = "artifact reject"
            raw_command = 0.0
            ema_command *= 0.85
            latest_pred_code = None
            return

        p_vec = classifier.predict_proba(x_win[np.newaxis, ...])[0]
        left_prob = float(p_vec[class_index[int(stim_cfg.left_code)]])
        right_prob = float(p_vec[class_index[int(stim_cfg.right_code)]])
        rest_prob = (
            float(p_vec[class_index[int(task_cfg.rest_class_code)]])
            if int(task_cfg.rest_class_code) in class_index
            else 0.0
        )

        raw_command = float(np.clip(right_prob - left_prob + bias_offset, -1.0, 1.0))

        if bool(task_cfg.respect_rest_class) and int(task_cfg.rest_class_code) in class_index and rest_prob >= max(left_prob, right_prob):
            latest_pred_code = int(task_cfg.rest_class_code)
            command_target = 0.0
        else:
            if raw_command > 0.0:
                latest_pred_code = int(stim_cfg.right_code)
            elif raw_command < 0.0:
                latest_pred_code = int(stim_cfg.left_code)
            else:
                latest_pred_code = None

            if use_discrete_mode:
                if raw_command > 0.0:
                    command_target = 1.0
                elif raw_command < 0.0:
                    command_target = -1.0
                else:
                    command_target = 0.0
            else:
                command_target = raw_command

        alpha = _clamp(float(task_cfg.command_ema_alpha), 0.0, 1.0)
        ema_command = command_target if prediction_count == 0 else (1.0 - alpha) * ema_command + alpha * command_target
        prediction_count += 1
        live_note = "tracking"

    def _intent_from_decoder() -> int:
        confidence = max(left_prob, right_prob)
        if confidence < float(task_cfg.move_confidence_thresh):
            return 0
        if abs(ema_command) < float(task_cfg.command_deadband):
            return 0
        return 1 if ema_command > 0.0 else -1

    def _wait_for_space(prompt_text: str, help_text: str) -> None:
        cue.text = prompt_text
        while True:
            keys = event.getKeys()
            if "escape" in keys:
                raise KeyboardInterrupt
            if "space" in keys:
                return

            _poll_live_decoder()
            scoreboard.text = f"Target obstacles: {target_obstacles}"
            info_parts = [
                f"{label_cfg.left_name}={left_prob:.2f}",
                f"{label_cfg.right_name}={right_prob:.2f}",
                f"{label_cfg.rest_name}={rest_prob:.2f}",
                f"ema={ema_command:+.2f}",
                f"bias={bias_offset:+.2f}",
                live_note,
            ]
            info.text = "   ".join(info_parts)
            status.text = help_text
            _draw_world()

    def _run_countdown() -> None:
        countdown_clock = core.Clock()
        while countdown_clock.getTime() < float(task_cfg.countdown_s):
            keys = event.getKeys()
            if "escape" in keys:
                raise KeyboardInterrupt
            _poll_live_decoder()
            elapsed = countdown_clock.getTime()
            remaining = max(0.0, float(task_cfg.countdown_s) - elapsed)
            count = int(np.ceil(remaining))
            cue.text = "GO" if count <= 0 else str(count)
            scoreboard.text = f"Target obstacles: {target_obstacles}"
            info.text = "Stay relaxed, then apply LEFT/RIGHT imagery to switch lanes."
            status.text = ""
            _draw_world()

    _wait_for_space(
        prompt_text="Model ready",
        help_text=(
            "SPACE to start. ESC to quit. "
            "Use LEFT/RIGHT motor imagery to move one lane at a time and dodge incoming obstacles."
        ),
    )
    _run_countdown()

    loop_clock = core.Clock()
    last_frame_t = loop_clock.getTime()
    completion_reason = "interrupted"

    try:
        while True:
            now_t = loop_clock.getTime()
            dt = _clamp(now_t - last_frame_t, 0.0005, 0.05)
            last_frame_t = now_t

            keys = event.getKeys()
            if "escape" in keys:
                raise KeyboardInterrupt

            keyboard_intent = 0
            if bool(task_cfg.enable_keyboard_override):
                if "left" in keys:
                    keyboard_intent = -1
                elif "right" in keys:
                    keyboard_intent = 1

            _poll_live_decoder()

            mi_intent = _intent_from_decoder()
            net_intent = keyboard_intent if keyboard_intent != 0 else mi_intent

            if net_intent != 0 and (now_t - last_switch_t) >= float(task_cfg.lane_switch_cooldown_s):
                next_lane = int(np.clip(target_lane_idx + int(net_intent), 0, 2))
                if next_lane != target_lane_idx:
                    target_lane_idx = next_lane
                    last_switch_t = now_t

            lane_blend = _clamp(dt * float(task_cfg.lane_transition_rate), 0.0, 1.0)
            lane_state += (float(target_lane_idx) - lane_state) * lane_blend
            lane_state = _clamp(lane_state, 0.0, 2.0)

            car_x = float(np.interp(lane_state, [0.0, 1.0, 2.0], lane_centers.tolist()))
            car.pos = (car_x, float(task_cfg.car_y))
            car_lane_idx = int(np.clip(np.rint(lane_state), 0.0, 2.0))

            current_speed = min(
                float(task_cfg.obstacle_speed_max_norm_s),
                float(task_cfg.obstacle_speed_norm_s) + float(task_cfg.obstacle_speedup_per_obstacle) * float(resolved_obstacles),
            )
            spawn_timer += dt
            if spawn_timer >= float(task_cfg.obstacle_spawn_interval_s) and len(obstacles) < int(task_cfg.max_obstacles_on_screen):
                spawn_timer = 0.0
                if bool(task_cfg.avoid_repeating_spawn_lane) and last_spawn_lane is not None:
                    spawn_options = [idx for idx in (0, 1, 2) if idx != int(last_spawn_lane)]
                else:
                    spawn_options = [0, 1, 2]
                lane_idx = int(rng.choice(spawn_options))
                obstacle_counter += 1
                obstacles.append(
                    ObstacleState(
                        obstacle_id=int(obstacle_counter),
                        lane_idx=lane_idx,
                        y=float(task_cfg.obstacle_spawn_y),
                    )
                )
                last_spawn_lane = lane_idx

            active_obstacles: list[ObstacleState] = []
            for obs in obstacles:
                obs.y -= current_speed * dt
                if obs.y <= float(task_cfg.obstacle_resolution_y):
                    is_collision = int(car_lane_idx) == int(obs.lane_idx)
                    resolved_obstacles += 1
                    event_row = {
                        "obstacle_id": int(obs.obstacle_id),
                        "resolved_time_s": round(float(now_t), 4),
                        "obstacle_lane": int(obs.lane_idx),
                        "car_lane": int(car_lane_idx),
                        "outcome": "collision" if is_collision else "dodge",
                        "left_prob": round(float(left_prob), 4),
                        "right_prob": round(float(right_prob), 4),
                        "rest_prob": round(float(rest_prob), 4),
                        "raw_command": round(float(raw_command), 4),
                        "ema_command": round(float(ema_command), 4),
                    }
                    events_log.append(event_row)

                    if is_collision:
                        collisions += 1
                        score += int(task_cfg.collision_points)
                        feedback_text = "Collision"
                        feedback_color = (0.95, -0.1, -0.1)
                    else:
                        dodges += 1
                        score += int(task_cfg.dodge_points)
                        feedback_text = "Dodge"
                        feedback_color = (-0.10, 0.90, -0.10)
                    feedback_until = now_t + float(task_cfg.feedback_duration_s)
                    continue

                if obs.y >= float(task_cfg.obstacle_despawn_y):
                    active_obstacles.append(obs)
            obstacles = active_obstacles

            scoreboard.text = (
                f"Score {score:+05d}   "
                f"Dodges {dodges:03d}   "
                f"Collisions {collisions:03d}   "
                f"Progress {resolved_obstacles:03d}/{target_obstacles:03d}"
            )
            cue.text = "LEFT/RIGHT imagery to shift lanes"
            info.text = (
                f"{label_cfg.left_name}={left_prob:.2f}   "
                f"{label_cfg.right_name}={right_prob:.2f}   "
                f"{label_cfg.rest_name}={rest_prob:.2f}   "
                f"raw={raw_command:+.2f}   "
                f"ema={ema_command:+.2f}   "
                f"note={live_note}"
            )
            status.text = (
                feedback_text if now_t <= feedback_until else ""
            )
            if now_t <= feedback_until:
                status.color = feedback_color
            else:
                status.color = (0.84, 0.84, 0.84)

            _draw_world()

            if resolved_obstacles >= int(target_obstacles):
                completion_reason = "target_reached"
                break
            if int(task_cfg.max_collisions) > 0 and collisions >= int(task_cfg.max_collisions):
                completion_reason = "collision_limit"
                break

    except KeyboardInterrupt:
        completion_reason = "user_escape"
        logger.info("Session interrupted by user.")

    results_payload = {
        "session": fname,
        "completion_reason": completion_reason,
        "target_obstacles": int(target_obstacles),
        "resolved_obstacles": int(resolved_obstacles),
        "dodges": int(dodges),
        "collisions": int(collisions),
        "score": int(score),
        "prediction_updates": int(prediction_count),
        "config": asdict(task_cfg),
        "events": events_log,
    }

    with open(f"{fname}_mi_racing_results.pkl", "wb") as fh:
        pickle.dump(results_payload, fh)
    logger.info(
        "Saved MI racing results to %s_mi_racing_results.pkl | reason=%s dodges=%d collisions=%d score=%d",
        fname,
        completion_reason,
        dodges,
        collisions,
        score,
    )

    if completion_reason != "user_escape":
        while True:
            keys = event.getKeys()
            if "escape" in keys or "space" in keys:
                break
            scoreboard.text = (
                f"Final score {score:+05d}   Dodges {dodges:03d}   Collisions {collisions:03d}"
            )
            cue.text = "Session complete" if completion_reason == "target_reached" else "Collision limit reached"
            info.text = f"Results saved to {fname}_mi_racing_results.pkl"
            status.text = "Press SPACE to exit task."
            _draw_world()

    try:
        stream.disconnect()
    except Exception:
        pass
    try:
        win.close()
    except Exception:
        pass


if __name__ == "__main__":
    fname = _prompt_session_prefix()
    print(f"[SESSION] prefix: {fname}")
    run_task(fname=fname)
