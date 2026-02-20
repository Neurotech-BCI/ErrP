# config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class LSLConfig:
    name: str = "WS-default" 
    stype: str = "EEG"     
    source_id: str | None = None # set if needed; otherwise None

    # Trigger / stim channel as it appears in the LSL stream info
    event_channels: str = "Trigger"   # Wearable Sensing DSI devices often expose Trigger/TRG :contentReference[oaicite:4]{index=4}

@dataclass(frozen=True)
class EEGConfig:
    # Channels you want to use (match your renaming convention)
    picks: tuple[str, ...] = ("EEG F4", "EEG C4", "EEG P4", "EEG P3", "EEG C3", "EEG F3", "EEG Pz")

    # Real-time filtering (stream-level)
    l_freq: float = 8.0     # MI mu/beta emphasis
    h_freq: float = 30.0
    notch: float | None = 60.0  # set None if not desired

    # Epoching window (motor imagery)
    tmin: float = 0.0
    tmax: float = 2.5       # MI window length (seconds)

    # Baseline correction (optional; keep None for pure MI windows)
    baseline: tuple[float | None, float | None] | None = None

@dataclass(frozen=True)
class ModelConfig:
    # Wait for this many epochs before first training
    min_epochs_to_train: int = 12

    # Retrain every N new epochs
    retrain_every: int = 4

    # CSP configuration
    n_csp_components: int = 6

    # Use sliding window to adapt to nonstationarity (user learning)
    use_sliding_window: bool = True
    window_size_epochs: int = 80  # keep last N epochs for retraining

@dataclass(frozen=True)
class ZMQConfig:
    # Worker publishes predictions; PsychoPy subscribes.
    pub_addr: str = "tcp://127.0.0.1:5556"
    topic: str = "PRED"  # ZMQ PUB/SUB topic prefix

@dataclass(frozen=True)
class SerialConfig:
    port: str = "COM6"  
    baudrate: int = 115200 
    pulse_width_s: float = 0.01  # send code then reset-to-0 after this