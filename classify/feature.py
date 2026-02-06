import numpy as np
from mne_features.univariate import (
    compute_samp_entropy,
    compute_spect_entropy,
    compute_hjorth_complexity,
    compute_hjorth_mobility,
    compute_kurtosis,
    compute_skewness,
)
from scipy.signal import welch, periodogram
from collections import defaultdict
from .graph_features import (
    node_strengths_coherence,
    betweenness_centrality_pli,
    clustering_coefficient_plv,
    clustering_coefficient_pli,
)
import itertools
from EntropyHub import PermEn, FuzzEn
import pywt


### Wrapper for extracting temporal features from raw EEG samples in shape (num_channels,time_steps)
### and returning outputs in shape (num_channels, num_features)
class FeatureWrapper:
    def __init__(self, left_ch_idx=None, right_ch_idx=None, lr_pairs=None):
        """
        Contralateral / lateralization support:

        left_ch_idx / right_ch_idx:
            Lists/arrays of indices for "left-ish" and "right-ish" channels.
            Used for group-based lateralization features.

        lr_pairs:
            Optional list of (l_idx, r_idx) symmetric channel pairs.
            Used for pairwise lateralization features.
        """
        self.left_ch_idx = None if left_ch_idx is None else np.asarray(left_ch_idx, dtype=int)
        self.right_ch_idx = None if right_ch_idx is None else np.asarray(right_ch_idx, dtype=int)
        self.lr_pairs = lr_pairs  # list[(l_idx, r_idx)] or None

        self.func_dict = {
            # Existing features
            "spectral_entropy": self.compute_spectral_entropy,
            "sample_entropy": self.compute_sample_entropy,
            "alpha_bandpower": self.compute_alpha_bandpower,
            "beta_bandpower": self.compute_beta_bandpower,
            "theta_bandpower": self.compute_theta_bandpower,
            "delta_bandpower": self.compute_delta_bandpower,
            "hjorth_activity": self.compute_hjorth_activity,
            "hjorth_mobility": self.compute_hjorth_mobility,
            "hjorth_complexity": self.compute_hjorth_complexity,
            "node_strength": self.node_strength_coh,
            "fuzzy_entropy": self.compute_fuzzy_entropy,
            "permutation_entropy": self.compute_permutation_entropy,
            "betweenness_centrality": self.betweenness_pli,
            "clustering_pli": self.clustering_pli,
            "clustering_plv": self.clustering_plv,
            "kurtosis": self.compute_kurtosis,
            "skewness": self.compute_skewness,
            "rms": self.compute_rms,
            "std": self.compute_std,
            "time_max_peak": self.compute_time_max_peak,
            "time_min_peak": self.compute_time_min_peak,
            "max_peak_value": self.compute_value_max_peak,
            "min_peak_value": self.compute_value_min_peak,
            "prominence": self.compute_prominence,
            "mean_frequency": self.compute_mean_frequency,
            "median_frequency": self.compute_median_frequency,
            "power_bandwidth": self.compute_power_bandwidth,
            "fft_max_value": self.compute_fft_max_value,
            "fft_max_frequency": self.compute_fft_max_frequency,
            "peak_location": self.compute_peak_location,  # Welch-PSD peak
            "snr1": self.compute_snr1,
            "snr": self.compute_snr,
            "sinad": self.compute_sinad,
            "wavelet_energy_2_4": self.compute_wavelet_energy_2_4,
            "wavelet_energy_4_8": self.compute_wavelet_energy_4_8,

            # --- Added: MI-relevant mu + log powers ---
            "mu_bandpower": self.compute_mu_bandpower,                 # 8–12 Hz
            "log_mu_bandpower": self.compute_log_mu_bandpower,
            "log_alpha_bandpower": self.compute_log_alpha_bandpower,   # 8–13 Hz
            "log_beta_bandpower": self.compute_log_beta_bandpower,     # 13–30 Hz

            # --- Added: relative power (ERD-like within-epoch normalization) ---
            "rel_mu_power": self.compute_rel_mu_power,                 # mu / broadband(4–40)
            "rel_beta_power": self.compute_rel_beta_power,             # beta / broadband(4–40)
            "rel_alpha_power": self.compute_rel_alpha_power,           # alpha / broadband(4–40)

            # --- Added: contralateral/lateralization features (per-channel outputs) ---
            "mu_lateralization_pairwise": self.compute_mu_lateralization_pairwise,
            "beta_lateralization_pairwise": self.compute_beta_lateralization_pairwise,
            "alpha_lateralization_pairwise": self.compute_alpha_lateralization_pairwise,

            "mu_lateralization_groups": self.compute_mu_lateralization_groups,
            "beta_lateralization_groups": self.compute_beta_lateralization_groups,
            "alpha_lateralization_groups": self.compute_alpha_lateralization_groups,

            "rel_mu_lateralization_groups": self.compute_rel_mu_lateralization_groups,
            "rel_beta_lateralization_groups": self.compute_rel_beta_lateralization_groups,
            "rel_alpha_lateralization_groups": self.compute_rel_alpha_lateralization_groups,
        }

    ### Helpers ###
    def _compute_psd_welch(self, data, fs):
        """Welch PSD for each channel."""
        n_channels, n_times = data.shape
        nperseg = min(n_times, int(fs)) if fs is not None and fs > 0 else n_times
        freqs, psd = welch(data, fs=fs, nperseg=nperseg, axis=1)
        return freqs, psd  # freqs: (n_freqs,), psd: (n_channels, n_freqs)

    def _compute_fft(self, data, fs):
        """RFFT and corresponding frequencies for each channel."""
        n_channels, n_times = data.shape
        freqs = np.fft.rfftfreq(n_times, d=1.0 / fs)
        fft_vals = np.fft.rfft(data, axis=1)  # (n_channels, n_freqs)
        power = np.abs(fft_vals) ** 2
        return freqs, fft_vals, power

    def _safe_log(self, x, eps=1e-12):
        return np.log(np.asarray(x, dtype=float) + eps)

    def _bandpower_all_channels(self, data, fs, band):
        """
        Bandpower via Welch, summed over [band[0], band[1]] (Hz), per channel.
        Returns: (n_channels,)
        """
        n_channels = data.shape[0]
        out = np.zeros(n_channels, dtype=float)
        nperseg = min(data.shape[1], int(fs))
        for ch in range(n_channels):
            freqs, psd = welch(data[ch], fs=fs, nperseg=nperseg)
            idx = (freqs >= band[0]) & (freqs <= band[1])
            out[ch] = float(np.sum(psd[idx]))
        return out

    def _broadband_power(self, data, fs, band=(4.0, 40.0)):
        return self._bandpower_all_channels(data, fs, band=band)

    def _pairwise_lateralization(self, per_ch_power, n_channels):
        """
        For each (l,r) pair: lat = logP(r) - logP(l).
        Returns a per-channel vector:
            out[r] += lat
            out[l] += -lat
        Unpaired channels get 0.
        """
        out = np.zeros(n_channels, dtype=float)
        if not self.lr_pairs:
            return out

        logp = self._safe_log(per_ch_power)
        for (l, r) in self.lr_pairs:
            if l < 0 or r < 0 or l >= n_channels or r >= n_channels:
                continue
            lat = float(logp[r] - logp[l])
            out[r] += lat
            out[l] -= lat
        return out

    def _group_lateralization(self, per_ch_power, n_channels):
        """
        Group lateralization:
            lat = mean(logP(right_group)) - mean(logP(left_group))
        Returns per-channel vector:
            channels in right_group get +lat
            channels in left_group get -lat
            others 0
        """
        out = np.zeros(n_channels, dtype=float)
        if self.left_ch_idx is None or self.right_ch_idx is None:
            return out

        logp = self._safe_log(per_ch_power)
        lidx = self.left_ch_idx[(self.left_ch_idx >= 0) & (self.left_ch_idx < n_channels)]
        ridx = self.right_ch_idx[(self.right_ch_idx >= 0) & (self.right_ch_idx < n_channels)]
        if len(lidx) == 0 or len(ridx) == 0:
            return out

        lat = float(np.mean(logp[ridx]) - np.mean(logp[lidx]))
        out[ridx] = +lat
        out[lidx] = -lat
        return out

    def _compute_wavelet_band_energy(self, data, fs, band, wavelet_name="db4"):
        """
        Approximate wavelet-band energy per channel for a given frequency band.

        band: (f_low, f_high) in Hz, e.g. (2,4) or (4,8)
        Returns: array (n_channels,)
        """
        n_channels, n_times = data.shape
        wavelet = pywt.Wavelet(wavelet_name)
        max_level = pywt.dwt_max_level(n_times, wavelet.dec_len)
        energies = np.zeros(n_channels, dtype=float)

        for ch in range(n_channels):
            coeffs = pywt.wavedec(data[ch], wavelet, level=max_level)
            selected_level = None
            for level in range(1, max_level + 1):
                # detail band: [fs/2^(level+1), fs/2^level]
                f_high_level = fs / (2 ** level)
                f_low_level = fs / (2 ** (level + 1))
                if not (band[1] <= f_low_level or band[0] >= f_high_level):
                    selected_level = level
                    break
            if selected_level is None:
                continue
            detail_coeffs = coeffs[selected_level]
            energies[ch] = np.sum(detail_coeffs ** 2)

        return energies

    ### Existing API features ###
    def compute_skewness(self, data, fs):
        return compute_skewness(data)

    def compute_kurtosis(self, data, fs):
        return compute_kurtosis(data)

    def compute_hjorth_activity(self, data, fs):
        return np.var(data, axis=1)

    def compute_hjorth_mobility(self, data, fs):
        return compute_hjorth_mobility(data)

    def compute_hjorth_complexity(self, data, fs):
        return compute_hjorth_complexity(data)

    def compute_alpha_bandpower(self, data, fs):
        return self._bandpower_all_channels(data, fs, band=(8.0, 13.0))

    def compute_delta_bandpower(self, data, fs):
        return self._bandpower_all_channels(data, fs, band=(0.5, 4.0))

    def compute_beta_bandpower(self, data, fs):
        return self._bandpower_all_channels(data, fs, band=(13.0, 30.0))

    def compute_theta_bandpower(self, data, fs):
        return self._bandpower_all_channels(data, fs, band=(4.0, 8.0))

    def compute_spectral_entropy(self, data, sfreq):
        return compute_spect_entropy(sfreq, data)

    def compute_sample_entropy(self, data, sfreq):
        return compute_samp_entropy(data)

    def compute_permutation_entropy(self, data, sfreq):
        m = 2
        tau = 1
        n_channels = data.shape[0]
        pe = np.zeros(n_channels)
        for ch in range(n_channels):
            pe[ch] = PermEn(data[ch], m=m, tau=tau)[0][-1]
        return pe

    def compute_fuzzy_entropy(self, data, sfreq):
        m = 2
        r = (0.2, 2)
        tau = 1
        n_channels = data.shape[0]
        fe = np.zeros(n_channels)
        for ch in range(n_channels):
            fe[ch] = FuzzEn(data[ch], m=m, r=r, tau=tau)[0][-1]
        return fe

    def node_strength_coh(self, data, fs):
        return node_strengths_coherence(data)

    def betweenness_pli(self, data, fs):
        return betweenness_centrality_pli(data)

    def clustering_pli(self, data, fs):
        return clustering_coefficient_pli(data)

    def clustering_plv(self, data, fs):
        return clustering_coefficient_plv(data)

    def compute_rms(self, data, fs):
        return np.sqrt(np.mean(data ** 2, axis=1))

    def compute_std(self, data, fs):
        return np.std(data, axis=1)

    def compute_value_max_peak(self, data, fs):
        return np.max(data, axis=1)

    def compute_value_min_peak(self, data, fs):
        return np.min(data, axis=1)

    def compute_time_max_peak(self, data, fs):
        idx = np.argmax(data, axis=1)
        return idx.astype(float) / fs

    def compute_time_min_peak(self, data, fs):
        idx = np.argmin(data, axis=1)
        return idx.astype(float) / fs

    def compute_prominence(self, data, fs):
        return np.max(data, axis=1) - np.min(data, axis=1)

    def compute_fft_max_value(self, data, fs):
        _, fft_vals, _ = self._compute_fft(data, fs)
        return np.max(np.abs(fft_vals), axis=1)

    def compute_fft_max_frequency(self, data, fs):
        freqs, fft_vals, _ = self._compute_fft(data, fs)
        idx = np.argmax(np.abs(fft_vals), axis=1)
        return freqs[idx]

    def compute_mean_frequency(self, data, fs):
        freqs, psd = self._compute_psd_welch(data, fs)
        freqs_row = freqs[None, :]
        num = np.sum(psd * freqs_row, axis=1)
        den = np.sum(psd, axis=1)
        den = np.where(den == 0, np.finfo(float).eps, den)
        return num / den

    def compute_median_frequency(self, data, fs):
        freqs, psd = self._compute_psd_welch(data, fs)
        cumsum = np.cumsum(psd, axis=1)
        total = cumsum[:, -1]
        n_channels = data.shape[0]
        med_freq = np.zeros(n_channels, dtype=float)

        for ch in range(n_channels):
            if total[ch] <= 0:
                med_freq[ch] = 0.0
                continue
            half_total = 0.5 * total[ch]
            idx = np.searchsorted(cumsum[ch], half_total)
            if idx == 0:
                med_freq[ch] = freqs[0]
            elif idx >= len(freqs):
                med_freq[ch] = freqs[-1]
            else:
                f1, f2 = freqs[idx - 1], freqs[idx]
                c1, c2 = cumsum[ch, idx - 1], cumsum[ch, idx]
                if c2 == c1:
                    med_freq[ch] = f1
                else:
                    med_freq[ch] = f1 + (half_total - c1) * (f2 - f1) / (c2 - c1)
        return med_freq

    def compute_power_bandwidth(self, data, fs):
        freqs, psd = self._compute_psd_welch(data, fs)
        n_channels = data.shape[0]
        bw = np.zeros(n_channels, dtype=float)

        for ch in range(n_channels):
            p = psd[ch]
            if np.all(p <= 0):
                bw[ch] = 0.0
                continue

            peak_idx = np.argmax(p)
            peak_power = p[peak_idx]
            threshold = peak_power / 2.0  # 3 dB

            left = peak_idx
            while left > 0 and p[left] > threshold:
                left -= 1
            if left == peak_idx:
                f_low = freqs[peak_idx]
            else:
                f_low = np.interp(threshold, [p[left], p[left + 1]], [freqs[left], freqs[left + 1]])

            right = peak_idx
            n_freqs = len(freqs)
            while right < n_freqs - 1 and p[right] > threshold:
                right += 1
            if right == peak_idx:
                f_high = freqs[peak_idx]
            else:
                f_high = np.interp(threshold, [p[right - 1], p[right]], [freqs[right - 1], freqs[right]])

            bw[ch] = max(0.0, f_high - f_low)

        return bw

    def compute_peak_location(self, data, fs):
        freqs, psd = self._compute_psd_welch(data, fs)
        idx = np.argmax(psd, axis=1)
        return freqs[idx]

    def compute_snr1(self, data, fs):
        max_peak = np.max(np.abs(data), axis=1)
        rms = self.compute_rms(data, fs)
        eps = np.finfo(float).eps
        return max_peak / (rms + eps)

    def compute_snr(self, data, fs):
        _, _, power = self._compute_fft(data, fs)
        signal_power = np.max(power, axis=1)
        total_power = np.sum(power, axis=1)
        noise_power = total_power - signal_power
        noise_power = np.where(noise_power <= 0, np.finfo(float).eps, noise_power)
        return signal_power / noise_power

    def compute_sinad(self, data, fs):
        _, _, power = self._compute_fft(data, fs)
        signal_power = np.max(power, axis=1)
        total_power = np.sum(power, axis=1)
        noise_dist_power = total_power - signal_power
        noise_dist_power = np.where(noise_dist_power <= 0, np.finfo(float).eps, noise_dist_power)
        return total_power / noise_dist_power

    def compute_wavelet_energy_2_4(self, data, fs):
        return self._compute_wavelet_band_energy(data, fs, band=(2.0, 4.0))

    def compute_wavelet_energy_4_8(self, data, fs):
        return self._compute_wavelet_band_energy(data, fs, band=(4.0, 8.0))

    # -----------------------
    # Added: MI / contralateral desync features
    # -----------------------
    def compute_mu_bandpower(self, data, fs):
        # Mu band (often 8–12 Hz; narrower than alpha)
        return self._bandpower_all_channels(data, fs, band=(8.0, 12.0))

    def compute_log_mu_bandpower(self, data, fs):
        return self._safe_log(self.compute_mu_bandpower(data, fs))

    def compute_log_alpha_bandpower(self, data, fs):
        return self._safe_log(self.compute_alpha_bandpower(data, fs))

    def compute_log_beta_bandpower(self, data, fs):
        return self._safe_log(self.compute_beta_bandpower(data, fs))

    def compute_rel_mu_power(self, data, fs):
        mu = self.compute_mu_bandpower(data, fs)
        bb = self._broadband_power(data, fs, band=(4.0, 40.0))
        return mu / (bb + 1e-12)

    def compute_rel_beta_power(self, data, fs):
        beta = self.compute_beta_bandpower(data, fs)
        bb = self._broadband_power(data, fs, band=(4.0, 40.0))
        return beta / (bb + 1e-12)

    def compute_rel_alpha_power(self, data, fs):
        alpha = self.compute_alpha_bandpower(data, fs)
        bb = self._broadband_power(data, fs, band=(4.0, 40.0))
        return alpha / (bb + 1e-12)

    # Pairwise lateralization (requires lr_pairs)
    def compute_mu_lateralization_pairwise(self, data, fs):
        mu = self.compute_mu_bandpower(data, fs)
        return self._pairwise_lateralization(mu, data.shape[0])

    def compute_beta_lateralization_pairwise(self, data, fs):
        beta = self.compute_beta_bandpower(data, fs)
        return self._pairwise_lateralization(beta, data.shape[0])

    def compute_alpha_lateralization_pairwise(self, data, fs):
        alpha = self.compute_alpha_bandpower(data, fs)
        return self._pairwise_lateralization(alpha, data.shape[0])

    # Group lateralization (requires left_ch_idx and right_ch_idx)
    def compute_mu_lateralization_groups(self, data, fs):
        mu = self.compute_mu_bandpower(data, fs)
        return self._group_lateralization(mu, data.shape[0])

    def compute_beta_lateralization_groups(self, data, fs):
        beta = self.compute_beta_bandpower(data, fs)
        return self._group_lateralization(beta, data.shape[0])

    def compute_alpha_lateralization_groups(self, data, fs):
        alpha = self.compute_alpha_bandpower(data, fs)
        return self._group_lateralization(alpha, data.shape[0])

    # Relative-power lateralization variants (more robust to session gain differences)
    def compute_rel_mu_lateralization_groups(self, data, fs):
        rel_mu = self.compute_rel_mu_power(data, fs)
        return self._group_lateralization(rel_mu, data.shape[0])

    def compute_rel_beta_lateralization_groups(self, data, fs):
        rel_beta = self.compute_rel_beta_power(data, fs)
        return self._group_lateralization(rel_beta, data.shape[0])

    def compute_rel_alpha_lateralization_groups(self, data, fs):
        rel_alpha = self.compute_rel_alpha_power(data, fs)
        return self._group_lateralization(rel_alpha, data.shape[0])

    # -----------------------
    # Main API
    # -----------------------
    def compute_features(self, data, sfreq, desired_features=("alpha_bandpower",)):
        """
        data: np.ndarray (n_channels, n_times)
        returns: np.ndarray (n_channels, n_features)
        """
        features = []
        for feature in desired_features:
            if feature not in self.func_dict:
                raise KeyError(f"Unknown feature '{feature}'. Available: {list(self.func_dict.keys())}")
            calculated_feature = self.func_dict[feature](data, sfreq)  # (n_channels,)
            features.append(calculated_feature)

        features = np.stack(features, axis=1)  # (n_channels, n_features)
        return features
