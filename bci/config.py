# config.py
from __future__ import annotations

from dataclasses import dataclass
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
    picks: tuple[str, ...] = ("Pz", "F4", "C4", "P4", "P3", "C3", "F3")
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
    left_name: str = "LEFT"
    right_name: str = "RIGHT"


@dataclass(frozen=True)
class MentalCommandTaskConfig:
    # Folder of cued MI EDF sessions used to fit the live model at startup.
    data_dir: str = ""
    edf_glob: str = "*.edf"

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

    # Visualization runtime.
    live_duration_s: float = 180.0


@dataclass(frozen=True)
class MentalCommandModelConfig:
    cov_estimator: str = "oas"
