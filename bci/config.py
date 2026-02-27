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
    # picks: tuple[str, ...] = ("C4", "C3")

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
    n_calibration_trials: int = 40


@dataclass(frozen=True)
class ModelConfig:
    # Retrain every N new accepted epochs
    retrain_every: int = 10

    use_riemann: bool = True  # if False, use CSP+LR; if True, use Riemannian geometry approach
    # CSP configuration
    n_csp_components: int = 6

    # Use sliding window to adapt to nonstationarity (user learning)
    use_sliding_window: bool = True
    window_size_epochs: int = 120  # keep last N epochs for retraining


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
    name: str = "session_1"  # base filename for saved data

    # Raw stream capture during full session (calibration + online)
    raw_csv_suffix: str = "_raw.csv"
