from __future__ import annotations

import logging
import pickle
import random
from collections import Counter
from datetime import datetime

import numpy as np
from mne_lsl.stream import StreamLSL
from psychopy import core, event, visual

from config import (
    EEGConfig,
    LSLConfig,
    MentalCommandLabelConfig,
    MentalCommandModelConfig,
    MICursorTaskConfig,
    StimConfig,
)
from jaw_clench_detector import JawClenchDetector
from mental_command_worker import (
    StreamingIIRFilter,
    canonicalize_channel_name,
    resolve_channel_order,
    train_or_load_shared_mi_model,
)


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


def run_task(fname: str) -> None:  # noqa: C901
    logger = _make_task_logger(fname)

    lsl_cfg = LSLConfig()
    stim_cfg = StimConfig()
    label_cfg = MentalCommandLabelConfig()
    task_cfg = MICursorTaskConfig()
    model_cfg = MentalCommandModelConfig()
    eeg_cfg = EEGConfig(
        picks=("Pz", "F4", "C4", "P4", "P3", "C3", "F3"),
        l_freq=8.0,
        h_freq=30.0,
        reject_peak_to_peak=150.0,
    )

    # ---------------------------------------------------------------
    # Stream setup
    # ---------------------------------------------------------------
    stream = StreamLSL(
        bufsize=60.0,
        name=lsl_cfg.name,
        stype=lsl_cfg.stype,
        source_id=lsl_cfg.source_id,
    )
    stream.connect(acquisition_delay=0.001, processing_flags="all")
    logger.info("Connected: %s", stream.info)

    available = list(stream.info["ch_names"])
    model_ch_names, missing = resolve_channel_order(available, eeg_cfg.picks)
    if len(model_ch_names) < 2:
        event_key = canonicalize_channel_name(lsl_cfg.event_channels)
        model_ch_names = [ch for ch in available if canonicalize_channel_name(ch) != event_key]
    if len(model_ch_names) < 2:
        raise RuntimeError(f"Need >=2 EEG channels, found: {available}")

    stream.pick(model_ch_names)
    sfreq = float(stream.info["sfreq"])
    stream_ch_names = list(stream.info["ch_names"])
    logger.info("Channels: sfreq=%.1f  selected=%s  missing=%s", sfreq, stream_ch_names, missing)

    jaw_ch_idx = len(model_ch_names) - 1
    jaw_detector = JawClenchDetector(fs=sfreq)
    jaw_detector.refractory = 0.70
    jaw_thresh_min = 8.0
    jaw_calibration_duration_s = 5.0

    # ---------------------------------------------------------------
    # Window and scene
    # ---------------------------------------------------------------
    win = visual.Window(
        size=task_cfg.win_size,
        color=(0.10, 0.10, 0.12),
        units="norm",
        fullscr=task_cfg.fullscreen,
    )

    road = visual.ShapeStim(
        win,
        vertices=[(-0.78, -0.95), (0.78, -0.95), (0.42, 0.95), (-0.42, 0.95)],
        closeShape=True,
        fillColor=(0.18, 0.18, 0.20),
        lineColor=(0.26, 0.26, 0.30),
        lineWidth=2,
    )
    lane_line_l = visual.Line(win, start=(-0.25, -0.95), end=(-0.14, 0.95), lineColor=(0.35, 0.35, 0.40), lineWidth=2)
    lane_line_r = visual.Line(win, start=(0.25, -0.95), end=(0.14, 0.95), lineColor=(0.35, 0.35, 0.40), lineWidth=2)

    # Pre-allocate obstacle rects for speed.
    MAX_OBS = 20
    obs_rects = [
        visual.Rect(win, width=0.20, height=0.10, pos=(0, 0), fillColor=(0.92, 0.42, 0.18), lineColor=None)
        for _ in range(MAX_OBS)
    ]

    player_body = visual.Circle(win, radius=0.055, fillColor=(0.35, 0.85, 0.98), lineColor=(0.92, 0.92, 0.92), lineWidth=2)
    jump_glow = visual.Circle(win, radius=0.075, fillColor=(0.25, 0.45, 0.95), lineColor=None, opacity=0.0)

    txt_title = visual.TextStim(win, text="BCI Subway Runner", pos=(0, 0.93), height=0.055, color=(0.93, 0.93, 0.93))
    txt_score = visual.TextStim(win, text="", pos=(0.74, 0.82), height=0.045, color=(0.95, 0.95, 0.95), anchorHoriz="center")
    txt_speed = visual.TextStim(win, text="", pos=(0.74, 0.72), height=0.040, color=(0.76, 0.76, 0.76), anchorHoriz="center")
    txt_cue = visual.TextStim(win, text="", pos=(0, 0.83), height=0.045, color=(0.93, 0.93, 0.93), anchorHoriz="center")
    txt_info = visual.TextStim(win, text="", pos=(0, -0.78), height=0.035, color=(0.56, 0.82, 1.00), anchorHoriz="center")
    txt_status = visual.TextStim(win, text="", pos=(0, -0.88), height=0.034, color=(0.78, 0.78, 0.78), anchorHoriz="center")

    lane_offsets = (-0.33, 0.0, 0.33)
    player_base_y = -0.74

    def lane_to_x(lane_idx: int, y_pos: float) -> float:
        # Perspective compression: lanes converge near the horizon.
        scale = float(np.interp(y_pos, [-0.95, 0.95], [1.0, 0.44]))
        return lane_offsets[lane_idx] * scale

    def _draw_wait_screen(cue: str, status: str, info: str = "") -> None:
        road.draw()
        lane_line_l.draw()
        lane_line_r.draw()
        txt_title.draw()
        txt_score.text = "Score\n0"
        txt_speed.text = "Speed\n1.00x"
        txt_cue.text = cue
        txt_info.text = info
        txt_status.text = status
        txt_score.draw()
        txt_speed.draw()
        txt_cue.draw()
        txt_info.draw()
        txt_status.draw()
        win.flip()

    def _draw_game_frame(
        *,
        lane_idx: int,
        jump_offset: float,
        obstacles: list[dict],
        score: int,
        speed_mult: float,
        bci_text: str,
        status_text: str,
    ) -> None:
        road.draw()
        lane_line_l.draw()
        lane_line_r.draw()

        # Obstacles
        for i, ob in enumerate(obstacles[:MAX_OBS]):
            y = float(ob["y"])
            lane = int(ob["lane"])
            x = lane_to_x(lane, y)
            depth_scale = float(np.interp(y, [-0.95, 0.95], [1.0, 0.45]))
            w = 0.16 * depth_scale + 0.06
            h = 0.06 * depth_scale + 0.03
            r = obs_rects[i]
            r.pos = (x, y)
            r.width = w
            r.height = h
            r.draw()

        # Player
        player_x = lane_to_x(lane_idx, player_base_y)
        player_y = player_base_y + jump_offset
        jump_glow.pos = (player_x, player_y)
        jump_glow.opacity = 0.35 if jump_offset > 0.01 else 0.0
        jump_glow.draw()
        player_body.pos = (player_x, player_y)
        player_body.draw()

        txt_title.draw()
        txt_score.text = f"Score\n{score:d}"
        txt_speed.text = f"Speed\n{speed_mult:.2f}x"
        txt_cue.text = "LEFT/RIGHT imagery move lanes | Jaw clench jumps"
        txt_info.text = bci_text
        txt_status.text = status_text
        txt_score.draw()
        txt_speed.draw()
        txt_cue.draw()
        txt_info.draw()
        txt_status.draw()
        win.flip()

    # ---------------------------------------------------------------
    # Model preparation (same paradigm as keyboard/tetris)
    # ---------------------------------------------------------------
    _draw_wait_screen("Preparing MI model...", "Loading offline EDF sessions.")

    try:
        shared_model = train_or_load_shared_mi_model(
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
        counts = Counter(shared_model.class_counts)

        clf_classes = np.asarray(classifier.named_steps["clf"].classes_, dtype=int)
        if int(stim_cfg.left_code) not in class_index or int(stim_cfg.right_code) not in class_index:
            raise RuntimeError(
                f"Classifier classes {clf_classes.tolist()} do not contain expected left/right codes "
                f"{[int(stim_cfg.left_code), int(stim_cfg.right_code)]}."
            )

        if shared_model.dataset is not None:
            dataset = shared_model.dataset
            np.save(f"{fname}_subway_runner_windows.npy", dataset.X)
            np.save(f"{fname}_subway_runner_labels.npy", dataset.y)
            logger.info(
                "Offline dataset ready: files_used=%d/%d, trials=%d, windows=%d, class_counts=%s, loso_mean=%.4f, loso_std=%.4f",
                dataset.n_files_used,
                dataset.n_files_found,
                dataset.n_trials,
                dataset.n_windows,
                counts,
                loso.mean_accuracy,
                loso.std_accuracy,
            )
        else:
            logger.info(
                "Using shared cached MI model: class_counts=%s, loso_mean=%.4f, loso_std=%.4f, cache=%s",
                counts,
                loso.mean_accuracy,
                loso.std_accuracy,
                shared_model.cache_path,
            )

        with open(f"{fname}_subway_runner_model.pkl", "wb") as fh:
            pickle.dump(classifier, fh)

        _draw_wait_screen(
            f"Model ready  LOSO={loso.mean_accuracy:.3f}±{loso.std_accuracy:.3f}",
            "Press SPACE to start. ESC to quit.",
            info=f"{label_cfg.left_name}: {counts.get(int(stim_cfg.left_code), 0)}   {label_cfg.right_name}: {counts.get(int(stim_cfg.right_code), 0)}",
        )
        while True:
            keys = event.getKeys()
            if "escape" in keys:
                raise KeyboardInterrupt
            if "space" in keys:
                break
            _draw_wait_screen(
                f"Model ready  LOSO={loso.mean_accuracy:.3f}±{loso.std_accuracy:.3f}",
                "Press SPACE to start. ESC to quit.",
                info=f"{label_cfg.left_name}: {counts.get(int(stim_cfg.left_code), 0)}   {label_cfg.right_name}: {counts.get(int(stim_cfg.right_code), 0)}",
            )

        # Jaw calibration (same paradigm as keyboard/tetris)
        _draw_wait_screen("Jaw calibration", f"Keep jaw relaxed for {jaw_calibration_duration_s:.1f}s")
        calib_raw: list[np.ndarray] = []
        last_cal_ts: float | None = None
        cal_clock = core.Clock()
        while cal_clock.getTime() < jaw_calibration_duration_s:
            if "escape" in event.getKeys():
                raise KeyboardInterrupt
            data, ts = stream.get_data(winsize=0.25, picks="all")
            if data.size > 0 and ts is not None and len(ts) > 0:
                ts_arr = np.asarray(ts)
                mask = np.ones_like(ts_arr, bool) if last_cal_ts is None else (ts_arr > float(last_cal_ts))
                if np.any(mask):
                    new_data = np.asarray(data[jaw_ch_idx, mask], dtype=np.float32)
                    last_cal_ts = float(ts_arr[mask][-1])
                    calib_raw.append(new_data)
            remain = max(0.0, jaw_calibration_duration_s - cal_clock.getTime())
            _draw_wait_screen("Jaw calibration", f"Keep jaw relaxed for {remain:0.1f}s")

        if calib_raw:
            try:
                jaw_signal = np.concatenate(calib_raw).astype(np.float32)
                floor = jaw_detector.calibrate(jaw_signal)
                logger.info("Jaw detector calibrated: floor=%.3f", float(floor))
            except Exception:
                logger.warning("Jaw detector calibration failed, continuing with adaptive baseline.")

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

    # ---------------------------------------------------------------
    # Live decoder state
    # ---------------------------------------------------------------
    live_filter = StreamingIIRFilter.from_eeg_config(
        eeg_cfg=eeg_cfg,
        sfreq=sfreq,
        n_channels=len(model_ch_names),
    )
    live_buffer = np.empty((len(model_ch_names), 0), dtype=np.float32)
    keep_n = int(round((task_cfg.window_s + task_cfg.filter_context_s) * sfreq))
    window_n = int(round(task_cfg.window_s * sfreq))
    stream_pull_s = max(0.10, task_cfg.live_update_interval_s * 2.0)
    reject_thresh = eeg_cfg.reject_peak_to_peak
    last_live_ts: float | None = None

    jaw_buffer_raw = np.empty(0, dtype=np.float32)
    jaw_buffer_ts = np.empty(0, dtype=np.float64)

    pred_clock = core.Clock()
    prediction_count = 0
    left_prob = 0.5
    right_prob = 0.5
    rest_prob = 0.0
    raw_command = 0.0
    ema_command = 0.0
    live_note = "warming up"
    latest_pred_code: int | None = None
    bias_offset = float(task_cfg.live_bias_offset) if bool(task_cfg.enable_live_bias_offset) else 0.0

    def _pull_stream_and_update_buffers() -> None:
        nonlocal last_live_ts, live_buffer, jaw_buffer_raw, jaw_buffer_ts

        data, ts = stream.get_data(winsize=stream_pull_s, picks="all")
        if data.size == 0 or ts is None or len(ts) == 0:
            return

        ts_arr = np.asarray(ts)
        mask = np.ones_like(ts_arr, bool) if last_live_ts is None else (ts_arr > float(last_live_ts))
        if not np.any(mask):
            return

        x_new = np.asarray(data[:, mask], dtype=np.float32)
        t_new = ts_arr[mask].astype(np.float64)
        last_live_ts = float(t_new[-1])

        x_new_filt = live_filter.process(x_new)
        live_buffer = np.concatenate((live_buffer, x_new_filt), axis=1)
        if live_buffer.shape[1] > keep_n:
            live_buffer = live_buffer[:, -keep_n:]

        jaw_raw_new = np.asarray(x_new[jaw_ch_idx], dtype=np.float32)
        jaw_buffer_raw = np.concatenate((jaw_buffer_raw, jaw_raw_new))
        jaw_buffer_ts = np.concatenate((jaw_buffer_ts, t_new))

        keep = int(max(sfreq, window_n))
        if jaw_buffer_raw.size > keep:
            jaw_buffer_raw = jaw_buffer_raw[-keep:]
            jaw_buffer_ts = jaw_buffer_ts[-keep:]

    def _poll_decoder() -> None:
        nonlocal prediction_count, left_prob, right_prob, rest_prob
        nonlocal raw_command, ema_command, live_note, latest_pred_code

        if pred_clock.getTime() < task_cfg.live_update_interval_s:
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
        rest_prob = float(p_vec[class_index[int(task_cfg.rest_class_code)]]) if int(task_cfg.rest_class_code) in class_index else 0.0
        raw_command = float(np.clip(right_prob - left_prob + bias_offset, -1.0, 1.0))

        if int(task_cfg.rest_class_code) in class_index and rest_prob >= max(left_prob, right_prob):
            latest_pred_code = int(task_cfg.rest_class_code)
        else:
            latest_pred_code = int(stim_cfg.right_code) if raw_command >= 0.0 else int(stim_cfg.left_code)

        alpha = float(np.clip(task_cfg.command_ema_alpha, 0.0, 1.0))
        ema_command = raw_command if prediction_count == 0 else (1.0 - alpha) * ema_command + alpha * raw_command
        prediction_count += 1
        live_note = "tracking"

    def _poll_jaw_clench() -> bool:
        if jaw_buffer_raw.size < 8 or jaw_buffer_ts.size < 8:
            return False
        _, clenches, _ = jaw_detector.detect(jaw_buffer_raw, jaw_buffer_ts, jaw_thresh_min)
        return len(clenches) > 0

    # ---------------------------------------------------------------
    # Runner gameplay
    # ---------------------------------------------------------------
    LANE_MOVE_COOLDOWN_S = 0.30
    JUMP_DURATION_S = 0.65
    JUMP_HEIGHT = 0.22
    JUMP_RETRIGGER_COOLDOWN_S = 0.50

    def _show_game_over(score: int, survived_s: float, jumps: int) -> bool:
        go_txt = visual.TextStim(
            win,
            text=(
                "GAME OVER\n\n"
                f"Score: {score}\n"
                f"Time: {survived_s:.1f}s\n"
                f"Jumps: {jumps}\n\n"
                "SPACE to play again   ESC to quit"
            ),
            pos=(0, 0),
            height=0.08,
            color=(0.95, 0.30, 0.30),
            wrapWidth=1.8,
        )
        while True:
            road.draw()
            lane_line_l.draw()
            lane_line_r.draw()
            go_txt.draw()
            win.flip()
            keys = event.getKeys()
            if "escape" in keys:
                return False
            if "space" in keys:
                return True

    session_stats: list[dict] = []
    rng = random.Random(42)

    try:
        while True:
            # Reset per-run decoder state.
            live_filter.reset()
            live_buffer = np.empty((len(model_ch_names), 0), np.float32)
            jaw_buffer_raw = np.empty(0, dtype=np.float32)
            jaw_buffer_ts = np.empty(0, dtype=np.float64)
            jaw_detector.reset_runtime_state()
            last_live_ts = None
            pred_clock.reset()
            prediction_count = 0
            left_prob = right_prob = 0.5
            rest_prob = raw_command = ema_command = 0.0
            latest_pred_code = None
            live_note = "warming up"

            lane_idx = 1
            last_lane_move_t = -999.0
            jump_active = False
            jump_start_t = -999.0
            last_jump_trigger_t = -999.0
            jump_count = 0

            obstacles: list[dict] = []
            spawn_timer = 0.6
            score = 0
            passed = 0
            game_over = False

            game_clock = core.Clock()
            last_frame_t = core.getTime()

            logger.info("New Subway Runner game started.")

            try:
                while not game_over:
                    keys = event.getKeys()
                    if "escape" in keys:
                        raise KeyboardInterrupt

                    now = core.getTime()
                    dt = max(0.0, min(0.050, now - last_frame_t))
                    last_frame_t = now
                    elapsed = game_clock.getTime()

                    _pull_stream_and_update_buffers()
                    _poll_decoder()

                    # Lane movement from MI left/right classification.
                    if now - last_lane_move_t >= LANE_MOVE_COOLDOWN_S:
                        if latest_pred_code == int(stim_cfg.left_code) and lane_idx > 0:
                            lane_idx -= 1
                            last_lane_move_t = now
                            logger.info("Lane move LEFT -> lane=%d", lane_idx)
                        elif latest_pred_code == int(stim_cfg.right_code) and lane_idx < 2:
                            lane_idx += 1
                            last_lane_move_t = now
                            logger.info("Lane move RIGHT -> lane=%d", lane_idx)

                    # Jaw clench jump.
                    if _poll_jaw_clench() and (now - last_jump_trigger_t) >= JUMP_RETRIGGER_COOLDOWN_S:
                        if not jump_active:
                            jump_active = True
                            jump_start_t = now
                            jump_count += 1
                            logger.info("Jaw clench -> jump")
                        last_jump_trigger_t = now

                    jump_offset = 0.0
                    if jump_active:
                        phase = (now - jump_start_t) / JUMP_DURATION_S
                        if phase >= 1.0:
                            jump_active = False
                        else:
                            jump_offset = JUMP_HEIGHT * float(np.sin(np.pi * phase))

                    # Obstacle spawn/update.
                    speed_mult = 1.0 + 0.35 * min(1.5, elapsed / 60.0)
                    world_speed = 0.95 * speed_mult

                    spawn_interval = max(0.34, 0.92 - 0.30 * min(1.0, elapsed / 75.0))
                    spawn_timer -= dt
                    if spawn_timer <= 0.0:
                        lane = rng.randint(0, 2)
                        if obstacles and obstacles[-1]["y"] > 0.35 and int(obstacles[-1]["lane"]) == lane:
                            lane = (lane + rng.choice((1, 2))) % 3
                        obstacles.append({"lane": int(lane), "y": 1.02})
                        spawn_timer = spawn_interval + rng.uniform(-0.12, 0.12)

                    remove_count = 0
                    for ob in obstacles:
                        ob["y"] -= world_speed * dt
                    while obstacles and obstacles[0]["y"] < -1.05:
                        obstacles.pop(0)
                        remove_count += 1

                    if remove_count > 0:
                        passed += remove_count

                    score = int(elapsed * 12.0 + passed * 8)

                    # Collision: same lane and obstacle near player y while not high enough in jump.
                    for ob in obstacles:
                        if int(ob["lane"]) != lane_idx:
                            continue
                        if abs(float(ob["y"]) - player_base_y) > 0.085:
                            continue
                        if jump_offset < 0.10:
                            game_over = True
                            logger.info("Collision detected at t=%.2fs, score=%d", elapsed, score)
                            break

                    bci_parts = [
                        f"{label_cfg.left_name}: {left_prob:.2f}",
                        f"{label_cfg.right_name}: {right_prob:.2f}",
                    ]
                    if int(task_cfg.rest_class_code) in class_index:
                        bci_parts.append(f"{label_cfg.rest_name}: {rest_prob:.2f}")
                    bci_parts.extend([f"cmd={ema_command:+.2f}", f"bias={bias_offset:+.2f}", live_note])

                    status_text = (
                        f"Lane: {lane_idx + 1}/3   Obstacles: {len(obstacles)}   "
                        f"Predictions: {prediction_count}   Jumps: {jump_count}"
                    )

                    _draw_game_frame(
                        lane_idx=lane_idx,
                        jump_offset=jump_offset,
                        obstacles=obstacles,
                        score=score,
                        speed_mult=speed_mult,
                        bci_text="   ".join(bci_parts),
                        status_text=status_text,
                    )

            except KeyboardInterrupt:
                survived = float(game_clock.getTime())
                logger.info("Game interrupted by user.")
                session_stats.append(
                    {
                        "score": int(score),
                        "survived_s": survived,
                        "predictions": int(prediction_count),
                        "jumps": int(jump_count),
                        "passed": int(passed),
                    }
                )
                break

            survived = float(game_clock.getTime())
            session_stats.append(
                {
                    "score": int(score),
                    "survived_s": survived,
                    "predictions": int(prediction_count),
                    "jumps": int(jump_count),
                    "passed": int(passed),
                }
            )
            logger.info("Game over: score=%d, survived_s=%.2f, jumps=%d", int(score), survived, int(jump_count))

            if not _show_game_over(score=int(score), survived_s=survived, jumps=int(jump_count)):
                break

    finally:
        if session_stats:
            with open(f"{fname}_subway_runner_results.pkl", "wb") as fh:
                pickle.dump(session_stats, fh)
            logger.info("Saved %d run(s) to %s_subway_runner_results.pkl", len(session_stats), fname)
        try:
            stream.disconnect()
        except Exception:
            pass
        try:
            win.close()
        except Exception:
            pass


if __name__ == "__main__":
    prefix = _prompt_prefix()
    print(f"[SESSION] prefix: {prefix}")
    run_task(fname=prefix)
