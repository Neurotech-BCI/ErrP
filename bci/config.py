# config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class LSLConfig:
    name: str = "WS-default"
    stype: str = "EEG"
    source_id: str | None = None  # set if needed; otherwise None

    # Trigger / stim channel as it appears in the LSL stream info
    event_channels: str = "TRG"  # Wearable Sensing DSI devices often expose Trigger/TRG


@dataclass(frozen=True)
class StimConfig:
    # IMPORTANT: 0 is treated as "no event" by most pipelines.
    left_code: int = 1
    right_code: int = 2

    # ErrP / feedback markers (sent at cursor movement instant)
    correct_code: int = 3
    error_code: int = 4

    def is_lr_code(self, code: int) -> bool:
        return int(code) in (self.left_code, self.right_code)

    def is_any_code(self, code: int) -> bool:
        return int(code) in (self.left_code, self.right_code, self.correct_code, self.error_code)


@dataclass(frozen=True)
class EEGConfig:
    #picks: tuple[str, ...] = ("Pz", "F4", "C4", "P4", "P3", "C3", "F3")
    picks: tuple[str, ...] = ("Pz", "F4", "C4", "P4", "C3", "F3")
    #picks: tuple[str, ...] = ("C4", "C3")

    # Optional reference channel to apply consistently to offline EDFs and
    # the live stream when available. Keep None to leave the incoming
    # reference unchanged.
    reref_channel: str | None = None

    # Wide bandpass pre-filter (DC removal + anti-aliasing).
    # Sub-band decomposition is handled inside the classifier pipeline.
    l_freq: float = 1.0
    h_freq: float = 45.0
    notch: float | None = None  # set None if not desired

    # Online epoching window (motor imagery)
    tmin: float = 0.5
    tmax: float = 3.5  # online MI window length (seconds)

    # Baseline correction (optional; keep None for pure MI windows)
    baseline: tuple[float | None, float | None] | None = None

    # Artifact rejection: max peak-to-peak amplitude in stream units (µV for DSI).
    # Applied per-window after wide bandpass filtering.
    reject_peak_to_peak: float | None = 150.0


@dataclass(frozen=True)
class CalibrationConfig:
    # Number of initial normal trials to use for calibration (no feedback).
    n_calibration_trials: int = 80
    # Maximum calibration trials in a row before a mandatory break.
    max_trials_before_break: int = 20


@dataclass(frozen=True)
class ModelConfig:
    # Balanced scheduler block size for online trials.
    retrain_every: int = 8

    # LR hyperparameters for calibration CV evaluation.
    C: float = 1.0
    max_iter: int = 1500
    class_weight: str | None = "balanced"

    # Filter-bank sub-band definitions (shared with mental command pipeline).
    filter_bank_bands: tuple[tuple[float, float], ...] = (
        (4.0, 8.0),    # theta
        (8.0, 13.0),   # alpha
        (13.0, 30.0),  # beta
        (30.0, 45.0),  # low-gamma
    )

    # Online GMM adaptation.
    gmm_learning_rate: float = 0.05
    gmm_cov_reg: float = 1e-6

@dataclass(frozen=True)
class SerialConfig:

    def find_port_by_vid_pid(vid: int, pid: int) -> str | None:
        import serial.tools.list_ports
        for p in serial.tools.list_ports.comports():
            if p.vid == vid and p.pid == pid:
                return p.device
        return None

    # Keep trigger-hub discovery lazy so GUI-only tasks can import config.py
    # without requiring the hardware to be connected.
    port: str | None = find_port_by_vid_pid(vid=0x2341, pid=0x8037)
    baudrate: int = 115200
    pulse_width_s: float = 0.01  # send code then reset-to-0 after this


@dataclass(frozen=True)
class SessionConfig:
    # Raw stream capture during full session (calibration + online)
    raw_csv_suffix: str = "_raw.csv"


@dataclass(frozen=True)
class MentalCommandLabelConfig:
    left_name: str = "LEFT"
    right_name: str = "RIGHT"
    rest_name: str = "REST"


@dataclass(frozen=True)
class MentalCommandTaskConfig:
    # Folder of cued MI EDF sessions used to fit the live model at startup.
    data_dir: str = ""
    edf_glob: str = "*.edf"

    # MNE EDF readers typically return EEG in volts, while the live DSI/LSL
    # stream is exposed in microvolts. Scale offline data into the live units
    # before filtering and windowing so artifact thresholds and covariance
    # structure are comparable online vs offline.
    offline_eeg_scale_to_match_live: float = 1e6
    live_eeg_units: str = "uV"

    # Each cued MI execution epoch is 3 seconds long in the offline task.
    epoch_duration_s: float = 3.0

    # Sliding-window decoder definition used for both offline and online data.
    window_s: float = 1.0
    window_step_s: float = 0.50

    # Amount of prior filtered history we require before the first live
    # prediction. Offline sessions are now filtered continuously end-to-end,
    # so this mainly controls live warmup and how much recent context is kept.
    filter_context_s: float = 2.0

    # Continuous live feedback settings.
    live_update_interval_s: float = 0.50

    # If True, collect a short relaxed baseline at startup and add it as a
    # third REST class to the classifier.
    enable_online_rest_calibration: bool = False
    # Internal label used for the optional REST class. This is only for the
    # classifier and is never sent as a hardware trigger.
    rest_class_code: int = 99
    # Duration of the relaxed baseline collection before the live task.
    rest_calibration_duration_s: float = 20.0

    # If True, collect short live LEFT/RIGHT sustained MI blocks at startup
    # and add them as one additional session in LOSO and final fitting.
    enable_online_lr_calibration: bool = False
    # Number of sustained live calibration blocks to collect per class.
    online_lr_calibration_reps_per_class: int = 4
    # Preparation time before each live LEFT/RIGHT calibration block.
    online_lr_calibration_prep_s: float = 2.0
    # Sustained imagery duration for each live LEFT/RIGHT calibration block.
    online_lr_calibration_hold_s: float = 8.0
    # Rest interval between live LEFT/RIGHT calibration blocks.
    online_lr_calibration_iti_s: float = 2.0

    # If True, add a fixed signed offset to the live right-left command.
    enable_live_bias_offset: bool = False
    # Signed command offset added to (p_right - p_left). Positive favors
    # RIGHT, negative favors LEFT.
    live_bias_offset: float = 0.0

    # Visualization runtime.
    live_duration_s: float = 180.0

    # Kalman Filter Stuff
    activate_kf: bool = True
    make_kf_adaptive: bool = True


@dataclass(frozen=True)
class MICursorTaskConfig:
    # Folder of cued MI EDF sessions used to fit the live model at startup.
    data_dir: str = ""
    edf_glob: str = "*.edf"

    # Match offline EDF units to the live stream before preprocessing.
    offline_eeg_scale_to_match_live: float = 1e6
    live_eeg_units: str = "uV"

    # Each cued MI execution epoch is 3 seconds long in the offline task.
    epoch_duration_s: float = 3.0

    # Sliding-window decoder definition used for both offline and online data.
    # Increase `window_s` for stabler but less responsive predictions.
    # Decrease `window_s` for faster but noisier control.
    window_s: float = 1.0
    # Increase `window_step_s` to create fewer, less redundant offline windows.
    # Decrease `window_step_s` to sample training windows more densely.
    window_step_s: float = 0.50

    # Amount of prior filtered history required before the first live prediction.
    filter_context_s: float = 2.0

    # Decoder update cadence.
    # Increase `live_update_interval_s` for fewer, calmer updates.
    # Decrease it for more responsive but potentially twitchier control.
    live_update_interval_s: float = 0.10

    # If True, collect a short relaxed baseline at startup and add it as a
    # third REST class to the classifier.
    enable_online_rest_calibration: bool = False
    # Internal label used for the optional REST class. This is only for the
    # classifier and is never sent as a hardware trigger.
    rest_class_code: int = 99
    # Duration of the relaxed baseline collection before the live task.
    rest_calibration_duration_s: float = 20.0

    # If True, collect short live LEFT/RIGHT sustained MI blocks at startup
    # and add them as one additional session in LOSO and final fitting.
    enable_online_lr_calibration: bool = False
    # Number of sustained live calibration blocks to collect per class.
    online_lr_calibration_reps_per_class: int = 4
    # Preparation time before each live LEFT/RIGHT calibration block.
    online_lr_calibration_prep_s: float = 2.0
    # Sustained imagery duration for each live LEFT/RIGHT calibration block.
    online_lr_calibration_hold_s: float = 8.0
    # Rest interval between live LEFT/RIGHT calibration blocks.
    online_lr_calibration_iti_s: float = 2.0

    # If True, add a fixed signed offset to the live right-left decision score.
    enable_live_bias_offset: bool = True
    # Signed offset added before converting classifier output into a steering
    # command. Positive favors RIGHT, negative favors LEFT.
    live_bias_offset: float = 0.2
    # If True, tasks that opt in may smooth discrete signed left/right
    # decisions (-1 or +1) with the EMA command filter before applying
    # movement. This is currently used by lr_cursor_task and tetris_task.
    enable_discrete_command_ema: bool = True
    # Steering command mapping:
    # - "discrete_sign": use the biased left/right decision and map it to
    #   full-strength signed commands (-1 or +1) before smoothing.
    # - "probability_diff": use the biased analog score (p_right - p_left).
    mi_control_mode: str = "discrete_sign"

    # Window / display settings.
    fullscreen: bool = False
    win_size: tuple[int, int] = (1280, 760)
    # Increase `arena_margin` to shrink the playable area and keep objects farther from the edges.
    # Decrease it to give the cursor more room to move.
    arena_margin: float = 0.06

    # Trial flow.
    # Increase `trial_start_delay_s` to give the decoder more time to settle before motion starts.
    # Decrease it to start trials more quickly.
    trial_start_delay_s: float = 0.75
    # Increase `post_hit_pause_s` for clearer target-hit feedback between trials.
    # Decrease it to speed up the overall task pace.
    post_hit_pause_s: float = 0.40

    # Cursor / target geometry in PsychoPy "norm" units.
    # Increase `cursor_radius` to make the cursor easier to see and collisions easier.
    # Decrease it for more precise but harder control.
    cursor_radius: float = 0.028
    # Increase `target_radius` to make targets easier to hit.
    # Decrease it to make the task more demanding.
    target_radius: float = 0.110
    # Increase `target_min_distance_from_center` to force larger steering movements.
    # Decrease it to allow easier, nearer targets.
    target_min_distance_from_center: float = 0.50

    # Steering control dynamics.
    # Increase `forward_speed_norm_s` to make the task faster and harder.
    # Decrease it to give the user more time to steer.
    forward_speed_norm_s: float = 0.48
    # Increase `max_turn_rate_deg_s` for tighter turns and more agility.
    # Decrease it for gentler, more stable arcs.
    max_turn_rate_deg_s: float = 95.0
    # Increase `command_ema_alpha` for faster but noisier command updates.
    # Decrease it for smoother but laggier control.
    command_ema_alpha: float = 0.28
    # Increase `command_deadband` to suppress small noisy turns.
    # Decrease it to make weak intent move the cursor more readily.
    command_deadband: float = 0.10
    # Increase `steering_time_constant_s` for slower, smoother steering changes.
    # Decrease it for quicker but potentially more abrupt steering.
    steering_time_constant_s: float = 0.50

    # Optional jaw-clench pause/resume control for the steering task.
    # When enabled, the task starts paused and jaw clench toggles movement.
    enable_jaw_clench_pause: bool = True
    # Jaw clench probability threshold for pause/resume toggling.
    jaw_clench_prob_thresh: float = 0.70
    # Minimum time between accepted jaw-clench toggles.
    jaw_clench_refractory_s: float = 0.70
    # Rapid eye-blink probability threshold for tasks that use blink events.
    blink_prob_thresh: float = 0.70
    # Minimum time between accepted rapid eye-blink events.
    blink_refractory_s: float = 0.70
    # Number of long calibration blocks per jaw class (REST and CLENCH).
    jaw_calibration_blocks_per_class: int = 3
    # Preparation time before each jaw calibration trial.
    jaw_calibration_prep_s: float = 2.5
    # Delay after cue onset before the calibration block begins, giving the
    # participant time to initiate the requested face command.
    special_command_cue_offset_s: float = 0.5
    # Hold duration for each jaw calibration block.
    jaw_calibration_hold_s: float = 5.0
    # Trim this much off the start and end of each calibration block before
    # windowing so training better reflects sustained steady-state behavior.
    jaw_calibration_trim_s: float = 0.5
    # Rest interval between jaw calibration blocks.
    jaw_calibration_iti_s: float = 1.5
    # Jaw classifier sliding-window definition shared by training and online use.
    jaw_window_s: float = 0.60
    jaw_window_step_s: float = 0.10

    calirate_on_participant: str = "mi"

@dataclass(frozen=True)
class LiveMITaskConfig:
    # Folder of cued MI EDF sessions used to fit the live model at startup.
    data_dir: str = ""
    edf_glob: str = "*.edf"

    # MNE EDF readers typically return EEG in volts, while the live DSI/LSL
    # stream is exposed in microvolts. Scale offline data into the live units
    # before filtering and windowing so artifact thresholds and covariance
    # structure are comparable online vs offline.
    offline_eeg_scale_to_match_live: float = 1e6
    live_eeg_units: str = "uV"

    # Each cued MI execution epoch is 3 seconds long in the offline task.
    epoch_duration_s: float = 3.0

    # Sliding-window decoder definition used for both offline and online data.
    window_s: float = 3.0
    window_step_s: float = 0.50

    # Amount of prior filtered history we require before the first live
    # prediction. Offline sessions are now filtered continuously end-to-end,
    # so this mainly controls live warmup and how much recent context is kept.
    filter_context_s: float = 2.0

    # Continuous live feedback settings.
    live_update_interval_s: float = 0.50

    # Visualization runtime.
    # live_duration_s: float = 180.0

    n_live_trials: int = 100
    max_trials_before_break: int = 20

    prep_duration_s: float = 3.0
    execution_duration_s: float = 3.0
    iti_duration_s: float = 3.0

    fullscreen: bool = False
    win_size: tuple[int, int] = (1200, 700)

@dataclass(frozen=True)
class KnobTaskConfig:
    # Folder of cued MI EDF sessions used to fit the live model at startup.
    data_dir: str = ""
    edf_glob: str = "*.edf"

    # Match offline EDF units to the live stream before preprocessing.
    offline_eeg_scale_to_match_live: float = 1e6
    live_eeg_units: str = "uV"

    # Each cued MI execution epoch is 3 seconds long in the offline task.
    epoch_duration_s: float = 3.0

    # Sliding-window decoder definition used for both offline and online data.
    # Increase `window_s` for stabler but less responsive predictions.
    # Decrease `window_s` for faster but noisier control.
    window_s: float = 1.0
    # Increase `window_step_s` to create fewer, less redundant offline windows.
    # Decrease `window_step_s` to sample training windows more densely.
    window_step_s: float = 0.50

    # Amount of prior filtered history required before the first live prediction.
    filter_context_s: float = 2.0

    # Decoder update cadence.
    # Increase `live_update_interval_s` for fewer, calmer updates.
    # Decrease it for more responsive but potentially twitchier control.
    live_update_interval_s: float = 0.10

    # If True, collect a short relaxed baseline at startup and add it as a
    # third REST class to the classifier.
    enable_online_rest_calibration: bool = False
    # Internal label used for the optional REST class. This is only for the
    # classifier and is never sent as a hardware trigger.
    rest_class_code: int = 99
    # Duration of the relaxed baseline collection before the live task.
    rest_calibration_duration_s: float = 20.0

    # If True, collect short live LEFT/RIGHT sustained MI blocks at startup
    # and add them as one additional session in LOSO and final fitting.
    enable_online_lr_calibration: bool = False
    # Number of sustained live calibration blocks to collect per class.
    online_lr_calibration_reps_per_class: int = 4
    # Preparation time before each live LEFT/RIGHT calibration block.
    online_lr_calibration_prep_s: float = 2.0
    # Sustained imagery duration for each live LEFT/RIGHT calibration block.
    online_lr_calibration_hold_s: float = 8.0
    # Rest interval between live LEFT/RIGHT calibration blocks.
    online_lr_calibration_iti_s: float = 2.0

    # If True, add a fixed signed offset to the live right-left command.
    enable_live_bias_offset: bool = False
    # Signed command offset added to (p_right - p_left). Positive favors
    # RIGHT, negative favors LEFT.
    live_bias_offset: float = 0.0

    # Window / display settings.
    fullscreen: bool = False
    win_size: tuple[int, int] = (1280, 760)
    # Increase `arena_margin` to shrink the playable area and keep objects farther from the edges.
    # Decrease it to give the cursor more room to move.
    arena_margin: float = 0.06

    # Trial flow.
    # Increase `trial_start_delay_s` to give the decoder more time to settle before motion starts.
    # Decrease it to start trials more quickly.
    trial_start_delay_s: float = 0.75
    # Increase `post_hit_pause_s` for clearer target-hit feedback between trials.
    # Decrease it to speed up the overall task pace.
    post_hit_pause_s: float = 0.40

    # parameters specialized for this task
    knob_radius: float = 0.5
    rotation_speed: float = 0.1 # this is just a multiplier
    min_angular_distance: float = 0.5 # in radians. 0.5 rad is around 28.6 degrees. distances between target region centers in adjacent trials.
    target_region_radius: float = 0.01 # also in radians. target region will be 2*target_region_radius in size
    start_angle: float = 0 # ie start facing up / 12 o'clock

    # Increase `max_turn_rate_deg_s` for tighter turns and more agility.
    # Decrease it for gentler, more stable arcs.
    max_turn_rate_deg_s: float = 65.0
    # Increase `command_ema_alpha` for faster but noisier command updates.
    # Decrease it for smoother but laggier control.
    command_ema_alpha: float = 0.56
    # Increase `command_deadband` to suppress small noisy turns.
    # Decrease it to make weak intent move the cursor more readily.
    command_deadband: float = 0.10
    # Increase `steering_time_constant_s` for slower, smoother steering changes.
    # Decrease it for quicker but potentially more abrupt steering.
    steering_time_constant_s: float = 0.35


@dataclass(frozen=True)
class MIRacingTaskConfig:
    # Folder of cued MI EDF sessions used to fit the live model at startup.
    data_dir: str = ""
    edf_glob: str = "*.edf"

    # Match offline EDF units to the live stream before preprocessing.
    offline_eeg_scale_to_match_live: float = 1e6
    live_eeg_units: str = "uV"

    # Shared-model dataset window definition.
    epoch_duration_s: float = 3.0
    window_s: float = 1.0
    window_step_s: float = 0.50
    filter_context_s: float = 2.0
    live_update_interval_s: float = 0.10

    # Optional REST handling in live decoding.
    rest_class_code: int = 99
    respect_rest_class: bool = True

    # Decoder command shaping.
    mi_control_mode: str = "discrete_sign"
    command_ema_alpha: float = 0.30
    command_deadband: float = 0.18
    move_confidence_thresh: float = 0.56
    lane_switch_cooldown_s: float = 0.24
    lane_transition_rate: float = 11.0
    enable_live_bias_offset: bool = False
    live_bias_offset: float = 0.0

    # Stream pull cadence multiplier relative to `live_update_interval_s`.
    stream_pull_multiplier: float = 2.0

    # Layout / visuals.
    fullscreen: bool = False
    win_size: tuple[int, int] = (1280, 760)
    road_width: float = 1.28
    road_height: float = 1.72
    lane_divider_width: float = 0.012
    car_width: float = 0.17
    car_height: float = 0.12
    car_y: float = -0.74
    obstacle_width: float = 0.20
    obstacle_height: float = 0.12

    # Obstacle flow.
    obstacle_spawn_y: float = 0.98
    obstacle_despawn_y: float = -1.05
    obstacle_resolution_y: float = -0.66
    obstacle_spawn_interval_s: float = 1.10
    obstacle_speed_norm_s: float = 0.75
    obstacle_speedup_per_obstacle: float = 0.002
    obstacle_speed_max_norm_s: float = 1.18
    max_obstacles_on_screen: int = 7
    avoid_repeating_spawn_lane: bool = True

    # Session flow / scoring.
    seed: int | None = None
    countdown_s: float = 3.0
    trials_to_complete: int = 60
    max_collisions: int = 12
    dodge_points: int = 50
    collision_points: int = -90
    feedback_duration_s: float = 0.40

    # Optional keyboard fallback while developing.
    enable_keyboard_override: bool = False

    calirate_on_participant: str = "mi"


@dataclass(frozen=True)
class HingeTaskConfig:
    # Folder of Hinge-style profile directories. Each profile folder should
    # contain picture_1.jpg, picture_2.jpg, picture_3.jpg and
    # profile_metadata.json.
    profiles_dir: str = str(Path(__file__).resolve().parent / "profiles")

    # Folder of cued MI EDF sessions used to fit the full-epoch swipe model.
    data_dir: str = ""
    edf_glob: str = "*.edf"

    # Match offline EDF units to the live stream before preprocessing.
    offline_eeg_scale_to_match_live: float = 1e6
    live_eeg_units: str = "uV"

    # Offline and online swipe decoding both use one full 3 second execution
    # epoch per trial.
    epoch_duration_s: float = 3.0
    window_s: float = 3.0
    window_step_s: float = 3.0
    filter_context_s: float = 2.0

    # Cued trial timing.
    prep_duration_s: float = 3.0
    execute_duration_s: float = 3.0
    outcome_duration_s: float = 1.15
    inter_trial_pause_s: float = 0.35

    # UI / layout.
    fullscreen: bool = False
    win_size: tuple[int, int] = (1280, 760)
    picture_box_width: float = 0.42
    picture_box_height: float = 0.29

    # Optional keyboard follow-up after a right swipe.
    enable_keyboard_on_match: bool = True
    keyboard_move_confidence_thresh: float = 0.60
    keyboard_cursor_step_s: float = 0.70
    keyboard_jaw_select_refractory_s: float = 0.35
    keyboard_max_text_chars: int = 320

    # Full-epoch swipe model should remain unbiased.
    enable_live_bias_offset: bool = False
    live_bias_offset: float = 0.0

    # Shared jaw calibration settings.
    jaw_clench_prob_thresh: float = 0.70
    jaw_clench_refractory_s: float = 0.70
    jaw_calibration_blocks_per_class: int = 3
    jaw_calibration_prep_s: float = 2.5
    special_command_cue_offset_s: float = 0.5
    jaw_calibration_hold_s: float = 5.0
    jaw_calibration_trim_s: float = 0.5
    jaw_calibration_iti_s: float = 1.5
    jaw_window_s: float = 0.60
    jaw_window_step_s: float = 0.10

    calirate_on_participant: str = "mi"


@dataclass(frozen=True)
class MentalCommandModelConfig:
    cov_estimator: str = "oas"


@dataclass(frozen=True)
class SharedBCIConfig:
    participant_name: str = "participant"
    mi_cache_name: str = "mi_shared_lr_model"
    epoch_mi_cache_name: str = "mi_shared_epoch_model"
    jaw_calibration_blocks_per_class: int = 4
    jaw_calibration_prep_s: float = 2.5
    special_command_cue_offset_s: float = 0.5
    jaw_calibration_hold_s: float = 5.0
    jaw_calibration_trim_s: float = 0.5
    jaw_calibration_iti_s: float = 1.5
    jaw_window_s: float = 0.60
    jaw_window_step_s: float = 0.10


@dataclass(frozen=True)
class OrchestratedTaskSpec:
    task_name: str
    repeats: int = 1
    max_trials: int | None = None
    task_kwargs: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BCIOrchestratorConfig:
    shared: SharedBCIConfig = field(default_factory=SharedBCIConfig)
    shared_config_overrides: dict[str, dict[str, object]] = field(default_factory=dict)
    task_config_overrides: dict[str, dict[str, dict[str, object]]] = field(default_factory=dict)
    task_sequence: tuple[OrchestratedTaskSpec, ...] = field(default_factory=tuple)
