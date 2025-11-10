import optuna
from .feature import FeatureWrapper
from .classify import classify_sklearn, classify_torch
import numpy as np
class Optimizer():
    def __init__(self, channels: list, classifier, sfreq: int, model):
        self.feature_wrapper = FeatureWrapper()
        self.features = list(self.feature_wrapper.func_dict.keys())
        self.channels = channels
        self.classifier = classifier
        self.sfreq = sfreq
        self.model = model
    def objective(self, trial, data, labels):
        selected_channels = [ch for ch in self.channels if trial.suggest_categorical(str(ch), [True, False])]

        selected_features = [feat for feat in self.features if trial.suggest_categorical(feat, [True, False])]
        #print(selected_channels)
        #print(selected_features)
        if not selected_channels or not selected_features:
            return 0.0

        #data_selected = data[:, selected_channels, :]
        samples = []
        
        for data_idx, sample in enumerate(data):
            feature_matrix = self.feature_wrapper.compute_features(sample,data_idx,self.sfreq,channel_indices=selected_channels,desired_features=selected_features)
            samples.append(feature_matrix)
        samples = np.array(samples)
        #print(f"Samples shape: {samples.shape}")
        accuracy = self.classifier(samples, labels, self.model)['mean_accuracy']
        return accuracy
    def optimize_hyperparameters(self, data, labels, n_trials=15):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, data, labels), n_trials=n_trials,show_progress_bar=True)
        return study.best_params, study.best_value