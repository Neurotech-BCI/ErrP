from pathlib import Path
import mne
import numpy as np
from scipy.io import loadmat
from scipy.stats import kurtosis
from mne.filter import filter_data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from collections import defaultdict
from mne_features.univariate import (
    compute_samp_entropy,
    compute_spect_entropy,
    compute_hjorth_complexity,
    compute_hjorth_mobility,
    compute_kurtosis,
    compute_skewness,
)
from scipy.signal import welch, periodogram
import itertools
from EntropyHub import PermEn, FuzzEn
import pywt


class FeatureWrapper:
    def __init__(self, left_ch_idx=None, right_ch_idx=None, lr_pairs=None):
        self.left_ch_idx = None if left_ch_idx is None else np.asarray(left_ch_idx, dtype=int)
        self.right_ch_idx = None if right_ch_idx is None else np.asarray(right_ch_idx, dtype=int)
        self.lr_pairs = lr_pairs  # list[(l_idx, r_idx)] or None

        self.func_dict = {
            "spectral_entropy": self.compute_spectral_entropy,
            "sample_entropy": self.compute_sample_entropy,
            "alpha_bandpower": self.compute_alpha_bandpower,
            "beta_bandpower": self.compute_beta_bandpower,
            "theta_bandpower": self.compute_theta_bandpower,
            "delta_bandpower": self.compute_delta_bandpower,
            "hjorth_activity": self.compute_hjorth_activity,
            "hjorth_mobility": self.compute_hjorth_mobility,
            "hjorth_complexity": self.compute_hjorth_complexity,
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
            "peak_location": self.compute_peak_location,
            "snr1": self.compute_snr1,
            "snr": self.compute_snr,
            "sinad": self.compute_sinad,
            "wavelet_energy_2_4": self.compute_wavelet_energy_2_4,
            "wavelet_energy_4_8": self.compute_wavelet_energy_4_8,
        }

    def _compute_psd_welch(self, data, fs):
        n_channels, n_times = data.shape
        nperseg = min(n_times, int(fs)) if fs is not None and fs > 0 else n_times
        freqs, psd = welch(data, fs=fs, nperseg=nperseg, axis=1)
        return freqs, psd

    def _compute_fft(self, data, fs):
        n_channels, n_times = data.shape
        freqs = np.fft.rfftfreq(n_times, d=1.0 / fs)
        fft_vals = np.fft.rfft(data, axis=1)
        power = np.abs(fft_vals) ** 2
        return freqs, fft_vals, power

    def _safe_log(self, x, eps=1e-12):
        return np.log(np.asarray(x, dtype=float) + eps)

    def _bandpower_all_channels(self, data, fs, band):
        n_channels = data.shape[0]
        out = np.zeros(n_channels, dtype=float)
        nperseg = min(data.shape[1], int(fs))
        for ch in range(n_channels):
            freqs, psd = welch(data[ch], fs=fs, nperseg=nperseg)
            idx = (freqs >= band[0]) & (freqs <= band[1])
            out[ch] = float(np.sum(psd[idx]))
        return out
    def _compute_wavelet_band_energy(self, data, fs, band, wavelet_name="db4"):
        n_channels, n_times = data.shape
        wavelet = pywt.Wavelet(wavelet_name)
        max_level = pywt.dwt_max_level(n_times, wavelet.dec_len)
        energies = np.zeros(n_channels, dtype=float)

        for ch in range(n_channels):
            coeffs = pywt.wavedec(data[ch], wavelet, level=max_level)
            selected_level = None
            for level in range(1, max_level + 1):
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

    def compute_features(self, data, sfreq, desired_features=("alpha_bandpower",)):
        features = []
        for feature in desired_features:
            if feature not in self.func_dict:
                raise KeyError(f"Unknown feature '{feature}'. Available: {list(self.func_dict.keys())}")
            calculated_feature = self.func_dict[feature](data, sfreq)
            features.append(calculated_feature)

        features = np.stack(features, axis=1)
        return features

def classify_sklearn(X, y, model, normalize=True, cv_splitter=StratifiedKFold(n_splits=5, shuffle=True), 
                     groups=None, return_preds=False, use_smote=False, smote_k_neighbors=5):
    outputs = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    X_reshaped = np.reshape(X, (X.shape[0], -1))
    
    for train, test in cv_splitter.split(X_reshaped, y, groups=groups):
        train_data = X_reshaped[train]
        test_data = X_reshaped[test]
        y_train = y[train]
        y_test = y[test]

        if normalize:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(train_data)
            train_data = scaler.transform(train_data)
            test_data = scaler.transform(test_data)

        if use_smote:
            sm = SMOTE(k_neighbors=smote_k_neighbors)
            train_data, y_train = sm.fit_resample(train_data, y_train)

        model.fit(train_data, y_train)
        y_pred = model.predict(test_data)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        recalls.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

        outputs.extend([(pred, real, index) for pred, real, index in zip(y_pred, y_test, test)])
    
    outputs = sorted(outputs, key=lambda x: x[2])
    outputs = [(pred, real) for pred, real, _ in outputs]

    metrics_dict = {
        'mean_accuracy': np.mean(accuracies),
        'mean_precision': np.mean(precisions),
        'mean_recall': np.mean(recalls),
        'mean_f1': np.mean(f1_scores),
    }

    if return_preds:
        metrics_dict['predictions'] = outputs

    return metrics_dict

def preprocess_eeg(eeg_data: np.ndarray, sfreq: float) -> np.ndarray:
    num_samples, num_channels, num_timesteps = eeg_data.shape
    processed = np.copy(eeg_data)
    ch_names_all = mne.channels.make_standard_montage("standard_1020").ch_names[:num_channels]
    montage = mne.channels.make_standard_montage("standard_1020")

    processed = filter_data(
        processed.reshape(-1, num_timesteps), sfreq,
        l_freq=1.0, h_freq=20.0,
        method='fir', fir_design='firwin', fir_window='hamming',
        phase='zero'
    ).reshape(num_samples, num_channels, num_timesteps)

    kurt_vals = kurtosis(processed, axis=2, fisher=False)
    bad_masks = np.zeros_like(kurt_vals, dtype=bool)

    for i in range(num_samples):
        th = np.mean(kurt_vals[i]) + 3 * np.std(kurt_vals[i])
        bad_masks[i] = kurt_vals[i] > th

    for i in range(num_samples):
        if np.any(bad_masks[i]):
            data = processed[i]
            ch_names = ch_names_all
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
            raw = mne.io.RawArray(data, info)
            raw.set_montage(montage, match_case=False)
            raw.info["bads"] = [ch_names[j] for j, bad in enumerate(bad_masks[i]) if bad]
            raw.interpolate_bads(reset_bads=True)
            processed[i] = raw.get_data()

    processed -= processed.mean(axis=1, keepdims=True)

    return processed

def load_raw_edf(edf_file_path):
  data = mne.io.read_raw_edf(edf_file_path, preload = True)
  # Preprocess with bandpass
  data.filter(l_freq=0.5, h_freq=50.0, picks="eeg")
  # Find events labeled in the Trigger column
  events = mne.find_events(data, stim_channel='Trigger', min_duration=0.0)
  event_id = {f'event_type_{i}': i for i in set(events[:, 2])}
  # Create epochs time locked to the event triggers
  epochs = mne.Epochs(
      data,
      events,
      event_id=event_id,
      tmin=-0.2,      # Start 200 ms before event
      tmax=0.8,       # End 800 ms after event
      baseline=(-0.2, 0),  # Baseline period for correction
      preload=True
  )
  # Convert epochs and event labels to numpy
  X = epochs.get_data()
  y = epochs.events[:, -1]
  return X, y

desired_features=[
        'median_frequency',
        'power_bandwidth',
        'alpha_bandpower',
        "beta_bandpower",
        "theta_bandpower",
        "delta_bandpower",
        'fft_max_value',
        'rms',
        'std',
        'time_max_peak',
        'time_min_peak',
        'max_peak_value',
        'min_peak_value',
        'prominence',
        'mean_frequency',
        'snr1',
        'snr',
        'sinad',
        'peak_location',
        'wavelet_energy_2_4',
        'wavelet_energy_4_8',
]

directory_path = Path('./data_edf')
data_files = [str(file_path.resolve()) for file_path in directory_path.rglob('*.edf')]

for data_file in data_files:
    try:
        X, y = load_raw_edf(data_file)
        
        #The sampling frequency for the edf files is 300, not 512
        cleaned_data = preprocess_eeg(X, sfreq=300.0)
        
        wrapper = FeatureWrapper()
        features = np.stack([wrapper.compute_features(sample, 300.0, desired_features=desired_features) for sample in cleaned_data])
        
        classifier = SVC()
        cv_splitter = StratifiedKFold(n_splits=5,shuffle=True)
        
        #The y values from load_raw_edf are not 0 and 1, so they need to be mapped
        y = np.where(y > 1, 1, 0)

        metrics = classify_sklearn(features, y, classifier, cv_splitter=cv_splitter, return_preds=True, use_smote=True)
        print(f"Results for {data_file}:")
        print(metrics)

    except Exception as e:
        print(f"Error processing {data_file}: {e}")
