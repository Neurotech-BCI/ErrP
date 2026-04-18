# jaw_clench_detector.py
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


class JawClenchDetector:
    """
    Threshold-based jaw clench detector for a single 1D signal.

    Designed to match the Tetris integration:
        detector = JawClenchDetector(fs=sfreq)
        detector.refractory = 0.75
        floor = detector.calibrate(jaw_signal)
        peak_idxs, clenches, thresh = detector.detect(signal, timestamps, thresh_min)
        detector.reset_runtime_state()
    """

    def __init__(self, fs: float) -> None:
        self.fs = float(fs)

        # runtime state
        self.recent_clenches: list[float] = []
        self.last_clench_time = -1e9
        self.refractory = 0.75
        self.merge_tol = 0.04

        # calibration / threshold state
        self.calibrated_floor = 8.0

        # peak shape constraints
        self.min_peak_width_s = 0.06
        self.max_peak_width_s = 0.50
        self.min_peak_distance_s = 0.25

    def reset_runtime_state(self) -> None:
        self.recent_clenches = []
        self.last_clench_time = -1e9

    def _bandpass(self, x: np.ndarray, low: float, high: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.size < 16:
            return x

        nyq = 0.5 * self.fs
        low = max(0.1, float(low))
        high = min(float(high), nyq - 1.0)

        if high <= low:
            return x

        b, a = butter(4, [low / nyq, high / nyq], btype="band")
        return filtfilt(b, a, x)

    def _envelope(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.abs(x)

    def calibrate(self, signal: np.ndarray) -> float:
        """
        Calibrate a personalized threshold from a signal where the user
        intentionally clenches about 5 times with short rests in between.
        """
        x = np.asarray(signal, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 8:
            return float(self.calibrated_floor)

        env = self._envelope(self._bandpass(x, 20.0, 45.0))
        env = env[np.isfinite(env)]
        if env.size < 8:
            return float(self.calibrated_floor)

        min_width = max(1, int(round(self.min_peak_width_s * self.fs)))
        max_width = max(min_width, int(round(self.max_peak_width_s * self.fs)))
        min_dist = max(1, int(round(self.min_peak_distance_s * self.fs)))

        loose_thresh = float(np.percentile(env, 70))
        prominence = max(float(np.std(env)) * 0.25, 1e-6)

        peaks, props = find_peaks(
            env,
            height=loose_thresh,
            prominence=prominence,
            width=(min_width, max_width),
            distance=min_dist,
        )

        if peaks.size >= 2:
            peak_heights = np.asarray(props["peak_heights"], dtype=float)
            q25 = float(np.percentile(peak_heights, 25))
            q50 = float(np.percentile(peak_heights, 50))
            floor = max(q25 * 0.85, q50 * 0.60, float(np.percentile(env, 80)))
        else:
            floor = max(
                float(np.percentile(env, 90)),
                float(np.mean(env) + 2.0 * np.std(env)),
            )

        self.calibrated_floor = float(max(1e-6, floor))
        return float(self.calibrated_floor)

    def detect(
        self,
        signal: np.ndarray,
        timestamps: np.ndarray,
        thresh_min: float,
    ) -> tuple[np.ndarray, list[float], float]:
        """
        Detect jaw clench events in the current 1D signal window.

        Returns:
            peak_indices: indices of accepted peaks in this window
            clenches: timestamps of newly detected clench events
            threshold: threshold used for this pass
        """
        x = np.asarray(signal, dtype=float)
        t = np.asarray(timestamps, dtype=float)

        if x.size < 8 or t.size < 8 or x.size != t.size:
            threshold = float(max(thresh_min, self.calibrated_floor))
            return np.array([], dtype=int), [], threshold

        finite_mask = np.isfinite(x) & np.isfinite(t)
        x = x[finite_mask]
        t = t[finite_mask]

        if x.size < 8:
            threshold = float(max(thresh_min, self.calibrated_floor))
            return np.array([], dtype=int), [], threshold

        env = self._envelope(self._bandpass(x, 20.0, 45.0))
        if env.size < 8:
            threshold = float(max(thresh_min, self.calibrated_floor))
            return np.array([], dtype=int), [], threshold

        adaptive = float(np.median(env) + 3.0 * np.std(env))
        threshold = float(max(thresh_min, self.calibrated_floor, adaptive))

        min_width = max(1, int(round(self.min_peak_width_s * self.fs)))
        max_width = max(min_width, int(round(self.max_peak_width_s * self.fs)))
        min_dist = max(1, int(round(self.min_peak_distance_s * self.fs)))
        prominence = max(float(np.std(env)) * 0.20, 1e-6)

        peaks, _ = find_peaks(
            env,
            height=threshold,
            prominence=prominence,
            width=(min_width, max_width),
            distance=min_dist,
        )

        accepted_peaks: list[int] = []
        clenches: list[float] = []

        for peak_idx in peaks:
            peak_time = float(t[int(peak_idx)])

            if (peak_time - self.last_clench_time) < float(self.refractory):
                continue

            if self.recent_clenches and abs(peak_time - self.recent_clenches[-1]) < float(self.merge_tol):
                continue

            self.last_clench_time = peak_time
            self.recent_clenches.append(peak_time)
            accepted_peaks.append(int(peak_idx))
            clenches.append(peak_time)

        if len(self.recent_clenches) > 20:
            self.recent_clenches = self.recent_clenches[-20:]

        return np.asarray(accepted_peaks, dtype=int), clenches, threshold