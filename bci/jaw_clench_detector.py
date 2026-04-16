from __future__ import annotations

import time

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

try:
    import pylsl
    _HAS_LSL = True
except ImportError:          # allow offline testing without LSL installed
    _HAS_LSL = False


class JawClenchDetector:
    """Detect jaw-clench bursts in a single EMG/EEG channel."""

    def __init__(self, fs: float) -> None:
        self.fs = float(fs)

        # --- timing ---
        self.refractory   = 0.50   # s  (Tetris: longer than Flappy Bird)
        self.merge_tol    = 0.10   # s  (merge peaks within one chunk boundary)
        self.last_clench_time: float = 0.0
        self.recent_clenches: list[float] = []

        # --- adaptive baseline ---
        self.baseline_window: list[float] = []
        self.baseline_size   = int(5 * self.fs)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bandpass(self, signal: np.ndarray, lowcut: float, highcut: float) -> np.ndarray:
        x = np.asarray(signal, dtype=float)
        if x.size < 20:
            return x
        nyq  = 0.5 * self.fs
        low  = max(lowcut  / nyq, 1e-4)
        high = min(highcut / nyq, 0.9999)
        if low >= high:
            return x
        b, a   = butter(4, [low, high], btype="band")
        padlen = 3 * max(len(a), len(b))
        if x.size <= padlen:
            return x
        return filtfilt(b, a, x)

    def _envelope(self, signal: np.ndarray, window_s: float = 0.05) -> np.ndarray:
        x = np.asarray(signal, dtype=float)
        n = max(int(window_s * self.fs), 1)
        if x.size < n:
            return np.sqrt(np.mean(x ** 2)) * np.ones_like(x)
        kernel = np.ones(n, dtype=float) / n
        power  = np.convolve(x ** 2, kernel, mode="same")
        return np.sqrt(np.maximum(power, 0.0))

    @staticmethod
    def _robust_stats(x: list | np.ndarray) -> tuple[float, float]:
        arr = np.asarray(x, dtype=float)
        if arr.size == 0:
            return 0.0, 1.0
        med   = float(np.median(arr))
        mad   = float(np.median(np.abs(arr - med)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(np.std(arr))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        return med, scale

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        signal: np.ndarray,
        timestamps: np.ndarray,
        thresh_min: float,
    ) -> tuple[np.ndarray, list[float], dict]:
        """
        Parameters
        ----------
        signal     : 1-D array, one EEG/EMG channel
        timestamps : LSL timestamps aligned to *signal*
        thresh_min : hard floor for detection threshold (µV or raw units)

        Returns
        -------
        peaks          : sample indices of detected peaks (np.ndarray)
        new_clenches   : LSL timestamps of newly confirmed clenches
        info           : dict with 'threshold', 'n_peaks', 'peak_heights'
        """
        signal     = np.asarray(signal,     dtype=float)
        timestamps = np.asarray(timestamps, dtype=float)

        if signal.size == 0:
            return np.array([], dtype=int), [], {
                "threshold": 0.0, "n_peaks": 0, "peak_heights": [],
            }

        # 1. Band-pass into the jaw-clench band
        jaw_band = self._bandpass(signal, 20.0, 45.0)

        # 2. RMS envelope
        jaw_env = self._envelope(jaw_band)

        # 3. Update running baseline
        self.baseline_window.extend(jaw_env.tolist())
        if len(self.baseline_window) > self.baseline_size:
            self.baseline_window = self.baseline_window[-self.baseline_size:]

        base_med, base_scale = self._robust_stats(self.baseline_window)

        thresh = max(
            base_med + 1.8 * base_scale,
            float(np.percentile(jaw_env, 88)),
            float(thresh_min),
        )

        # 4. Peak detection
        peaks, props = find_peaks(
            jaw_env,
            height   = thresh,
            width    = (int(0.02 * self.fs), int(0.30 * self.fs)),
            distance = int(0.15 * self.fs),
            prominence = max(thresh * 0.06, 1e-6),
        )

        # 5. Filter by refractory / merge tolerance
        new_clenches: list[float] = []
        now_wall = time.time()

        for p in peaks:
            t = float(timestamps[p])

            if any(abs(t - prev) < self.merge_tol for prev in self.recent_clenches):
                continue

            if now_wall - self.last_clench_time < self.refractory:
                continue

            self.last_clench_time = now_wall
            self.recent_clenches.append(t)
            new_clenches.append(t)

        # 6. Prune stale clench history (keep last 2 s)
        if _HAS_LSL:
            now_lsl = pylsl.local_clock()
            self.recent_clenches = [t for t in self.recent_clenches if now_lsl - t < 2.0]

        info = {
            "threshold":    float(thresh),
            "n_peaks":      int(len(peaks)),
            "peak_heights": (
                props.get("peak_heights", np.array([])).tolist() if len(peaks) else []
            ),
        }
        return peaks, new_clenches, info