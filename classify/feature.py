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

### Wrapper for extracting temporal features from raw EEG samples in shape (num_channels,time_steps) and returning outputs in shape (num_channels, num_features) ###
class FeatureWrapper():
    def __init__(self):
        self.func_dict = {
            'spectral_entropy': self.compute_spectral_entropy,
            'sample_entropy': self.compute_sample_entropy,
            'alpha_bandpower': self.compute_alpha_bandpower,
            'beta_bandpower': self.compute_beta_bandpower,
            'theta_bandpower': self.compute_theta_bandpower,
            'delta_bandpower': self.compute_delta_bandpower,
            'hjorth_activity': self.compute_hjorth_activity,
            'hjorth_mobility': self.compute_hjorth_mobility,
            'hjorth_complexity': self.compute_hjorth_complexity,
            'node_strength': self.node_strength_coh,
            'fuzzy_entropy': self.compute_fuzzy_entropy,
            'permutation_entropy': self.compute_permutation_entropy,
            'betweenness_centrality': self.betweenness_pli,
            'clustering_pli': self.clustering_pli,
            'clustering_plv': self.clustering_plv,
            'kurtosis': self.compute_kurtosis,
            'skewness': self.compute_skewness,

            'rms': self.compute_rms,
            'std': self.compute_std,
            'time_max_peak': self.compute_time_max_peak,
            'time_min_peak': self.compute_time_min_peak,
            'max_peak_value': self.compute_value_max_peak,
            'min_peak_value': self.compute_value_min_peak,
            'prominence': self.compute_prominence,

            'mean_frequency': self.compute_mean_frequency,
            'median_frequency': self.compute_median_frequency,
            'power_bandwidth': self.compute_power_bandwidth,
            'fft_max_value': self.compute_fft_max_value,
            'fft_max_frequency': self.compute_fft_max_frequency,
            'peak_location': self.compute_peak_location,  # Welch-PSD peak

            'snr1': self.compute_snr1,
            'snr': self.compute_snr,
            'sinad': self.compute_sinad,

            'wavelet_energy_2_4': self.compute_wavelet_energy_2_4,
            'wavelet_energy_4_8': self.compute_wavelet_energy_4_8,
        }
        
    ### Helpers ###
    def _compute_psd_welch(self, data, fs):
        """Welch PSD for each channel."""
        n_channels, n_times = data.shape
        # Use at most 1 second worth of data for nperseg (or all if shorter)
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

    def _compute_wavelet_band_energy(self, data, fs, band, wavelet_name='db4'):
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
            # Discrete wavelet decomposition
            coeffs = pywt.wavedec(data[ch], wavelet, level=max_level)
            # coeffs[0] is approximation; coeffs[1:] are D1..Dmax_level
            selected_level = None
            for level in range(1, max_level + 1):
                # Approximate detail band for level "level":
                # [fs/2^(level+1), fs/2^level]
                f_high_level = fs / (2 ** level)
                f_low_level = fs / (2 ** (level + 1))
                # Check if the bands overlap at all
                if not (band[1] <= f_low_level or band[0] >= f_high_level):
                    selected_level = level
                    break
            if selected_level is None:
                # No appropriate level found; leave energy as 0.
                continue
            detail_coeffs = coeffs[selected_level]
            energies[ch] = np.sum(detail_coeffs ** 2)

        return energies
    ### API ###
    def compute_skewness(self,data,fs):
        skewness = compute_skewness(data)
        return skewness
    def compute_kurtosis(self,data,fs):
        kurtosis = compute_kurtosis(data)
        return kurtosis
    def compute_hjorth_activity(self,data,fs):
        activity = np.var(data, axis=1)
        return activity
    def compute_hjorth_mobility(self,data,fs):
        mobility = compute_hjorth_mobility(data)
        return mobility
    def compute_hjorth_complexity(self,data,fs):
        complexity = compute_hjorth_complexity(data)
        return complexity
    def compute_alpha_bandpower(self, data, fs):  
        band=(8, 13)
        n_channels = data.shape[0]
        band_power = np.zeros(n_channels)
    
        for i in range(n_channels):
            freqs, psd = welch(data[i], fs=fs, nperseg=fs) 
            band_idx = np.logical_and(freqs >= band[0], freqs <= band[1])
            band_power[i] = np.sum(psd[band_idx])
    
        return band_power
    def compute_delta_bandpower(self, data, fs):  
        band=(0.5, 4)
        n_channels = data.shape[0]
        band_power = np.zeros(n_channels)
    
        for i in range(n_channels):
            freqs, psd = welch(data[i], fs=fs, nperseg=fs) 
            band_idx = np.logical_and(freqs >= band[0], freqs <= band[1])
            band_power[i] = np.sum(psd[band_idx])
    
        return band_power
    def compute_beta_bandpower(self, data, fs):  
        band=(13, 30)
        n_channels = data.shape[0]
        band_power = np.zeros(n_channels)
    
        for i in range(n_channels):
            freqs, psd = welch(data[i], fs=fs, nperseg=fs) 
            band_idx = np.logical_and(freqs >= band[0], freqs <= band[1])
            band_power[i] = np.sum(psd[band_idx])
    
        return band_power
    def compute_theta_bandpower(self, data, fs):  
        band=(4, 8)
        n_channels = data.shape[0]
        band_power = np.zeros(n_channels)
    
        for i in range(n_channels):
            freqs, psd = welch(data[i], fs=fs, nperseg=fs) 
            band_idx = np.logical_and(freqs >= band[0], freqs <= band[1])
            band_power[i] = np.sum(psd[band_idx])
    
        return band_power

    def compute_spectral_entropy(self,data,sfreq):
        spectral_entropy = compute_spect_entropy(sfreq, data)
        return spectral_entropy
    def compute_sample_entropy(self,data,sfreq):
        sample_entropy = compute_samp_entropy(data)
        return sample_entropy

    def compute_permutation_entropy(self,data,sfreq):
        m = 2
        tau = 1
        n_channels = data.shape[0]
        pe = np.zeros(n_channels)
        for ch in range(n_channels):
            signal = data[ch]
            pe[ch] = PermEn(signal, m=m, tau=tau)[0][-1]
        return pe

    def compute_fuzzy_entropy(self,data, sfreq):
        m = 2 
        r = (0.2,2)
        tau = 1
        n_channels = data.shape[0]
        fe = np.zeros(n_channels)
        for ch in range(n_channels):
            signal = data[ch]
            fe[ch] = FuzzEn(signal, m=m, r=r, tau=tau)[0][-1]
        return fe

    def node_strength_coh(self, data, fs): #input band as a string ie "alpha"
        return node_strengths_coherence(data)
    
    def betweenness_pli(self, data, fs):
        return betweenness_centrality_pli(data)
    
    def clustering_pli(self, data, fs):
        return clustering_coefficient_pli(data)
    
    def clustering_plv(self, data, fs): 
        return clustering_coefficient_plv(data)

    def compute_rms(self, data, fs):
        """
        Root mean square of each channel over time.
        """
        return np.sqrt(np.mean(data ** 2, axis=1))
    def compute_std(self, data, fs):
        """
        Standard deviation of each channel over time.
        """
        return np.std(data, axis=1)
    def compute_value_max_peak(self, data, fs):
        """
        Maximum (positive) peak amplitude per channel.
        """
        return np.max(data, axis=1)

    def compute_value_min_peak(self, data, fs):
        """
        Minimum (negative) peak amplitude per channel.
        """
        return np.min(data, axis=1)


    def compute_time_max_peak(self, data, fs):
        """
        Time (s) at which the maximum peak occurs in each channel.
        """
        idx = np.argmax(data, axis=1)  # sample index of max
        return idx.astype(float) / fs
    def compute_time_min_peak(self, data, fs):
        """
        Time (s) at which the minimum peak occurs in each channel.
        """
        idx = np.argmin(data, axis=1)
        return idx.astype(float) / fs
    def compute_prominence(self, data, fs):
        """
        Prominence = amplitude(max peak) - amplitude(min peak) per channel.
        """
        max_val = np.max(data, axis=1)
        min_val = np.min(data, axis=1)
        return max_val - min_val
    
    def compute_fft_max_value(self, data, fs):
        """
        Maximum magnitude of the FFT per channel (using rFFT).
        """
        _, fft_vals, _ = self._compute_fft(data, fs)
        return np.max(np.abs(fft_vals), axis=1)

    def compute_fft_max_frequency(self, data, fs):
        """
        Frequency (Hz) at which the magnitude of the FFT is maximal per channel.
        """
        freqs, fft_vals, _ = self._compute_fft(data, fs)
        idx = np.argmax(np.abs(fft_vals), axis=1)
        return freqs[idx]

    def compute_mean_frequency(self, data, fs):
        """
        Mean frequency of the power spectrum (Welch) per channel.
        """
        freqs, psd = self._compute_psd_welch(data, fs)
        psd = np.asarray(psd)
        freqs_row = freqs[None, :]  # for broadcasting
        num = np.sum(psd * freqs_row, axis=1)
        den = np.sum(psd, axis=1)
        den = np.where(den == 0, np.finfo(float).eps, den)
        return num / den

    def compute_median_frequency(self, data, fs):
        """
        Median frequency of the power spectrum (Welch) per channel.
        """
        freqs, psd = self._compute_psd_welch(data, fs)
        psd = np.asarray(psd)
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
                # Linear interpolation between (idx-1, idx)
                f1, f2 = freqs[idx - 1], freqs[idx]
                c1, c2 = cumsum[ch, idx - 1], cumsum[ch, idx]
                if c2 == c1:
                    med_freq[ch] = f1
                else:
                    med_freq[ch] = f1 + (half_total - c1) * (f2 - f1) / (c2 - c1)

        return med_freq
    
    def compute_power_bandwidth(self, data, fs):
        """
        Power bandwidth per channel (3 dB bandwidth from peak of Welch PSD).
        """
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
            threshold = peak_power / 2.0  # 3 dB below peak (power)

            # Search left
            left = peak_idx
            while left > 0 and p[left] > threshold:
                left -= 1
            if left == peak_idx:
                f_low = freqs[peak_idx]
            else:
                f_low = np.interp(
                    threshold,
                    [p[left], p[left + 1]],
                    [freqs[left], freqs[left + 1]],
                )

            # Search right
            right = peak_idx
            n_freqs = len(freqs)
            while right < n_freqs - 1 and p[right] > threshold:
                right += 1
            if right == peak_idx:
                f_high = freqs[peak_idx]
            else:
                f_high = np.interp(
                    threshold,
                    [p[right - 1], p[right]],
                    [freqs[right - 1], freqs[right]],
                )

            bw[ch] = max(0.0, f_high - f_low)

        return bw


    def compute_peak_location(self, data, fs):
        """
        Location (frequency in Hz) of the dominant peak of the Welch PSD.
        """
        freqs, psd = self._compute_psd_welch(data, fs)
        idx = np.argmax(psd, axis=1)
        return freqs[idx]

    def compute_snr1(self, data, fs):
        """
        SNR1 = max(|signal|) / RMS(signal) per channel.
        (Paper: 'max peak / RMS'; here applied to the provided window.)
        """
        max_peak = np.max(np.abs(data), axis=1)
        rms = self.compute_rms(data, fs)
        eps = np.finfo(float).eps
        return max_peak / (rms + eps)
    
    def compute_snr(self, data, fs):
        """
        SNR = P_signal / P_noise per channel, based on FFT power spectrum.
        Fundamental = bin with maximal power; noise = sum of remaining bins.
        """
        _, _, power = self._compute_fft(data, fs)
        signal_power = np.max(power, axis=1)
        total_power = np.sum(power, axis=1)
        noise_power = total_power - signal_power
        noise_power = np.where(noise_power <= 0, np.finfo(float).eps, noise_power)
        return signal_power / noise_power

    def compute_sinad(self, data, fs):
        """
        SINAD = (P_total) / (P_noise+distortion) per channel.

        P_total = sum over all FFT bins
        P_signal = power at fundamental (max bin)
        P_noise+distortion = P_total - P_signal
        """
        _, _, power = self._compute_fft(data, fs)
        signal_power = np.max(power, axis=1)
        total_power = np.sum(power, axis=1)
        noise_dist_power = total_power - signal_power
        noise_dist_power = np.where(
            noise_dist_power <= 0, np.finfo(float).eps, noise_dist_power
        )
        return total_power / noise_dist_power

    def compute_wavelet_energy_2_4(self, data, fs):
        """
        Wavelet-based band energy in 2–4 Hz (delta-like) band per channel.
        """
        return self._compute_wavelet_band_energy(data, fs, band=(2.0, 4.0))


    def compute_wavelet_energy_4_8(self, data, fs):
        """
        Wavelet-based band energy in 4–8 Hz (theta-like) band per channel.
        """
        return self._compute_wavelet_band_energy(data, fs, band=(4.0, 8.0))


    
    def compute_features(self, data, sfreq, desired_features = ["alpha_bandpower"]):
        features = []
        for feature in desired_features:
            calculated_feature = (self.func_dict[feature](data, sfreq))
            features.append(calculated_feature)
        features = np.stack(features,axis=1)
        return features
