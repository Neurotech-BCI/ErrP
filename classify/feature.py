import numpy as np
from mne_features.univariate import compute_samp_entropy, compute_spect_entropy, compute_hjorth_complexity, compute_hjorth_mobility, compute_kurtosis, compute_skewness
from scipy.signal import welch
from collections import defaultdict
from .graph_features import node_strengths_coherence, betweenness_centrality_pli, clustering_coefficient_plv, clustering_coefficient_pli
import itertools
from EntropyHub import PermEn, FuzzEn
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
            'skewness': self.compute_skewness
        }
        self.cache = []
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





    
    def compute_features(self,data,data_idx,sfreq,channel_indices,desired_features = ["alpha_bandpower"]):
        if len(self.cache) <= data_idx:
            self.cache.append({})
        features = []
        
        for feature in desired_features:
            if feature in self.cache[data_idx]:
                features.append(self.cache[data_idx][feature])
            else:
                calculated_feature = (self.func_dict[feature](data,sfreq))
                features.append(calculated_feature)
                self.cache[data_idx][feature] = calculated_feature
        features = np.stack(features,axis=1)
        return features
