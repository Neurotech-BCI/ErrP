# config.py
from __future__ import annotations

from dataclasses import dataclass


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
    picks: tuple[str, ...] = ("Pz", "F4", "C4", "P4", "P3", "C3", "F3")
    #picks: tuple[str, ...] = ("C4", "C3")

    # Real-time filtering (stream-level)
    l_freq: float = 8.0  # MI mu/beta emphasis
    h_freq: float = 30.0
    notch: float | None = None  # set None if not desired

    # Online epoching window (motor imagery)
    tmin: float = 0.5
    tmax: float = 3.5  # online MI window length (seconds)

    # Baseline correction (optional; keep None for pure MI windows)
    baseline: tuple[float | None, float | None] | None = None

    # Artifact rejection: max peak-to-peak amplitude in Volts; None to disable
    reject_peak_to_peak: float | None = None


@dataclass(frozen=True)
class CalibrationConfig:
    # Number of initial normal trials to use for calibration (no feedback).
    n_calibration_trials: int = 100
    # Maximum calibration trials in a row before a mandatory break.
    max_trials_before_break: int = 25


@dataclass(frozen=True)
class ModelConfig:
    # Retrain every N new accepted epochs
    retrain_every: int = 20

    use_riemann: bool = True  # if False, use CSP+LR; if True, use Riemannian geometry approach
    # CSP configuration
    n_csp_components: int = 6

@dataclass(frozen=True)
class SerialConfig:

    def find_port_by_vid_pid(vid: int, pid: int) -> str | None:
        import serial.tools.list_ports
        for p in serial.tools.list_ports.comports():
            if p.vid == vid and p.pid == pid:
                return p.device
        raise RuntimeError(
            "No valid trigger hub found"
        )

    port: str = find_port_by_vid_pid(vid=0x2341, pid=0x8037)
    baudrate: int = 115200
    pulse_width_s: float = 0.01  # send code then reset-to-0 after this


@dataclass(frozen=True)
class SessionConfig:
    # Raw stream capture during full session (calibration + online)
    raw_csv_suffix: str = "_raw.csv"


@dataclass(frozen=True)
class MentalCommandLabelConfig:
    neutral_code: int = 10
    command1_code: int = 11
    command2_code: int = 12

    command1_name: str = "LEFT"
    command2_name: str = "RIGHT"

    def all_codes(self) -> tuple[int, int, int]:
        return (self.neutral_code, self.command1_code, self.command2_code)


@dataclass(frozen=True)
class MentalCommandTaskConfig:
    # Emotiv Mental Command training uses ~8 second training blocks.
    register_duration_s: float = 8.0
    rest_duration_s: float = 3.0
    prep_duration_s: float = 1.5

    # Repetitions per class for registration.
    n_register_neutral: int = 6
    n_register_command: int = 6

    # Sliding-window extraction from registration blocks.
    train_window_s: float = 1.0
    train_window_step_s: float = 0.25

    # Continuous live feedback settings.
    live_update_interval_s: float = 0.10
    live_display_smoothing_alpha: float = 0.25
    min_confidence_to_show: float = 0.20

    # Visualization runtime.
    live_duration_s: float = 180.0


@dataclass(frozen=True)
class MentalCommandModelConfig:
    # Riemannian features + multinomial logistic regression.
    C: float = 1.0
    max_iter: int = 1500
    class_weight: str | None = "balanced"

    # Basic CV quality gate before live mode.
    cv_splits_max: int = 5
    min_per_class_for_cv: int = 4
