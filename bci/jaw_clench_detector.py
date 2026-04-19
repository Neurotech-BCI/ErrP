from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


class JawClenchDetector:
    """Detect jaw-clench bursts in a single EMG/EEG channel."""

    def __init__(self, fs: float) -> None:
        self.fs = float(fs)

        # Detection timing gates (seconds).
        self.refractory = 0.60
        self.merge_tol = 0.10
        self.recent_horizon_s = 3.0

        # Peak constraints.
        self.min_peak_width_s = 0.2
        self.max_peak_width_s = 0.6
        self.min_peak_distance_s = 0.18

        # Robust threshold controls.
        self.threshold_z = 4.0
        self.peak_z = 6.0
        self.calibrated_floor = 0.0

        # Adaptive baseline window (envelope domain).
        self.baseline_window: list[float] = []
        self.baseline_size = int(6 * self.fs)

        # Runtime state in signal timestamp domain.
        self.last_clench_ts: float | None = None
        self.recent_clenches: list[float] = []

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

    def reset_runtime_state(self) -> None:
        self.last_clench_ts = None
        self.recent_clenches.clear()
    """
    def calibrate(self, signal: np.ndarray) -> float:
        
        calibrate a personalized jaw-clench threshold from a recording where the
        user intentionally clenches about 5 times with short rests in between.
        
        x = np.asarray(signal, dtype=float)
        if x.size < 8:
            return self.calibrated_floor

        env = self._envelope(self._bandpass(x, 25.0, 60.0))
        env = np.asarray(env, dtype=float)
        env = env[np.isfinite(env)]
        if env.size < 8:
            return self.calibrated_floor

        min_width = max(1, int(round(self.min_peak_width_s * self.fs)))
        max_width = max(min_width, int(round(self.max_peak_width_s * self.fs)))
        min_dist = max(1, int(round(self.min_peak_distance_s * self.fs)))

        loose_thresh = float(np.percentile(env, 70))
        prominence = max(float(np.std(env)) * 0.25, 1e-6)

        peaks, props = find_peaks(
            env,
            height=loose_thresh,
            width=(min_width, max_width),
            distance=min_dist,
            prominence=prominence,
        )

        peak_heights = np.asarray(props.get("peak_heights", []), dtype=float)
        if peak_heights.size == 0:
            return self.calibrated_floor

        peak_heights = np.sort(peak_heights)[::-1]
        top_peaks = peak_heights[:5] if peak_heights.size >= 5 else peak_heights

        # estimate personalized clench amplitude.
        clench_med = float(np.median(top_peaks))
        clench_p25 = float(np.percentile(top_peaks, 25))

        #set threshold below the typical clench amplitude.
        # use a conservative fraction so weaker clenches still cross threshold.
        floor = max(
            0.40 * clench_med,
            0.55 * clench_p25,
        )

        self.calibrated_floor = float(floor)

        env_med, env_scale = self._robust_stats(env)
        clench_z = (top_peaks - env_med) / max(env_scale, 1e-9)
        if clench_z.size > 0:
            self.peak_z = max(2.5, float(np.percentile(clench_z, 20)) * 0.6)

        return self.calibrated_floor
        """

    def calibrate(self, signal: np.ndarray) -> float:
        x = np.asarray(signal, dtype=float)
        if x.size < 8:
            return self.calibrated_floor

        env = self._envelope(self._bandpass(x, 20.0, 45.0))
        env = env[np.isfinite(env)]
        if env.size < 8:
            return self.calibrated_floor

        quiet_cap = float(np.percentile(env, 35))
        quiet = env[env <= quiet_cap]
        if quiet.size < 8:
            quiet = env

        self.baseline_window = quiet[-self.baseline_size :].tolist()
        base_med, base_scale = self._robust_stats(self.baseline_window)
        floor = max(base_med + 4.0 * base_scale, float(np.percentile(env, 70)) * 0.45)
        self.calibrated_floor = max(self.calibrated_floor, float(floor))
        return self.calibrated_floor

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
        signal = np.asarray(signal, dtype=float)
        timestamps = np.asarray(timestamps, dtype=float)

        n = min(signal.size, timestamps.size)
        if n == 0:
            return np.array([], dtype=int), [], {
                "threshold": 0.0, "n_peaks": 0, "peak_heights": [],
            }
        signal = signal[-n:]
        timestamps = timestamps[-n:]

        if signal.size < 8:
            return np.array([], dtype=int), [], {
                "threshold": 0.0, "n_peaks": 0, "peak_heights": [],
            }

        # 1. Band-pass into the jaw-clench band.
        jaw_band = self._bandpass(signal, 20.0, 45.0)

        # 2. RMS envelope.
        jaw_env = self._envelope(jaw_band)
        jaw_env = np.asarray(jaw_env, dtype=float)
        jaw_env[~np.isfinite(jaw_env)] = 0.0

        # 3. Update baseline from lower-energy samples to avoid contamination.
        if self.baseline_window:
            base_med_prev, base_scale_prev = self._robust_stats(self.baseline_window)
        else:
            base_med_prev, base_scale_prev = self._robust_stats(jaw_env)

        quiet_cap = base_med_prev + 2.0 * max(base_scale_prev, 1e-9)
        quiet = jaw_env[jaw_env <= quiet_cap]
        if quiet.size < max(8, int(0.05 * jaw_env.size)):
            q25 = float(np.percentile(jaw_env, 25))
            quiet = jaw_env[jaw_env <= q25]
        if quiet.size > 0:
            self.baseline_window.extend(quiet.tolist())
            if len(self.baseline_window) > self.baseline_size:
                self.baseline_window = self.baseline_window[-self.baseline_size :]

        base_med, base_scale = self._robust_stats(self.baseline_window)
        adaptive_floor = max(float(thresh_min), self.calibrated_floor)
        thresh = max(
            adaptive_floor,
            base_med + self.threshold_z * base_scale,
        )

        # 4. Peak detection in envelope domain.
        min_width = max(1, int(round(self.min_peak_width_s * self.fs)))
        max_width = max(min_width, int(round(self.max_peak_width_s * self.fs)))
        min_dist = max(1, int(round(self.min_peak_distance_s * self.fs)))
        prominence = max(0.5 * base_scale, 0.10 * thresh, 1e-6)

        peaks, props = find_peaks(
            jaw_env,
            height=thresh,
            width=(min_width, max_width),
            distance=min_dist,
            prominence=prominence,
        )

        # 5. Filter by z-score + refractory / merge tolerance.
        accepted_peak_idx: list[int] = []
        new_clenches: list[float] = []
        peak_heights = props.get("peak_heights", np.array([], dtype=float))

        for i, p in enumerate(peaks):
            t = float(timestamps[p])
            height = float(peak_heights[i]) if i < len(peak_heights) else float(jaw_env[p])
            z = (height - base_med) / max(base_scale, 1e-9)
            if z < self.peak_z:
                continue

            if any(abs(t - prev) < self.merge_tol for prev in self.recent_clenches):
                continue

            if self.last_clench_ts is not None and (t - self.last_clench_ts) < self.refractory:
                continue

            self.last_clench_ts = t
            self.recent_clenches.append(t)
            accepted_peak_idx.append(int(p))
            new_clenches.append(t)

        # 6. Prune stale clench history in signal-time domain.
        now_ts = float(timestamps[-1])
        horizon = max(self.recent_horizon_s, self.refractory * 4.0)
        self.recent_clenches = [t for t in self.recent_clenches if now_ts - t < horizon]

        info = {
            "threshold": float(thresh),
            "baseline_med": float(base_med),
            "baseline_scale": float(base_scale),
            "n_peaks": int(len(peaks)),
            "n_accepted": int(len(accepted_peak_idx)),
            "peak_heights": (
                props.get("peak_heights", np.array([])).tolist() if len(peaks) else []
            ),
        }
        return np.asarray(accepted_peak_idx, dtype=int), new_clenches, info