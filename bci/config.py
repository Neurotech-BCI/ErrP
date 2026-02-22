# config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class LSLConfig:
    name: str = "WS-default" 
    stype: str = "EEG"     
    source_id: str | None = None # set if needed; otherwise None

    # Trigger / stim channel as it appears in the LSL stream info
    event_channels: str = "TRG"   # Wearable Sensing DSI devices often expose Trigger/TRG :contentReference[oaicite:4]{index=4}

@dataclass(frozen=True)
class EEGConfig:
    picks: tuple[str, ...] = ('Pz', 'F4', 'C4', 'P4', 'P3', 'C3', 'F3')
    #picks: tuple[str, ...] = ('C4', 'C3')

    # Real-time filtering (stream-level)
    l_freq: float = 8.0     # MI mu/beta emphasis
    h_freq: float = 30.0
    notch: float | None = None  # set None if not desired

    # Epoching window (motor imagery)
    tmin: float = 0.5
    tmax: float = 2.5      # MI window length (seconds)

    # Baseline correction (optional; keep None for pure MI windows)
    baseline: tuple[float | None, float | None] | None = None

    # Artifact rejection: max peak-to-peak amplitude in Volts; None to disable
    reject_peak_to_peak: float | None = 150e-6

@dataclass(frozen=True)
class CalibrationConfig:
    mi_duration_s: float = 10.0      # sustained MI block duration
    phases_per_class: int = 3        # accepted blocks needed per class
    prep_duration_s: float = 2.0     # "Prepare: LEFT/RIGHT" cue duration

@dataclass(frozen=True)
class ModelConfig:
    # Wait for this many epochs before first training
    min_epochs_to_train: int = 20

    # Retrain every N new epochs
    retrain_every: int = 4

    use_riemann: bool = True  # if False, use CSP+LR; if True, use Riemannian geometry approach
    # CSP configuration
    n_csp_components: int = 6

    # Use sliding window to adapt to nonstationarity (user learning)
    use_sliding_window: bool = True
    window_size_epochs: int = 80  # keep last N epochs for retraining

@dataclass(frozen=True)
class ZMQConfig:
    # Worker publishes predictions; PsychoPy subscribes.
    pub_addr: str = "tcp://127.0.0.1:5556"
    ctrl_addr: str = "tcp://127.0.0.1:5557"  # task PUSH -> worker PULL
    topic: str = "PRED"       # online prediction topic
    cal_topic: str = "CAL"    # calibration status topic

@dataclass(frozen=True)
class SerialConfig:
    port: str = "COM6"
    baudrate: int = 115200
    pulse_width_s: float = 0.01  # send code then reset-to-0 after this

@dataclass(frozen=True)
class SessionConfig:
    name: str = "session_1"  # base filename for saved data