import numpy as np
import glob
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from eeg_loader import load_participant_data
from classify.classify import classify_sklearn
from classify.feature import FeatureWrapper
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Define your list of subject files
DATA_PATH = "/Users/Selina/Documents/ErrP/data/*.mat"
SFREQ = 512.0

# Define features to extract (Common ErrP features)
DESIRED_FEATURES = [
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

# Step #1: LOAD DATA
subject_files = sorted(glob.glob(DATA_PATH))
if not subject_files:
    print("No files found. Please check DATA_PATH.")
    exit()

all_participants_X = []
all_participants_y = []

print("Loading Data...")
for filename in subject_files:
    # Load data (try to keep 500, but might return fewer)
    X, y = load_participant_data(filename, n_samples_to_keep=500)

    if X.size > 0:
        all_participants_X.append(X)
        all_participants_y.append(y)

if not all_participants_X:
    print("No valid data loaded.")
    exit()
# Harmonize sample counts among participants
# 1. Find the minimum number of samples across all loaded participants
min_samples = min(x.shape[0] for x in all_participants_X)
print(f"Harmonizing datasets: Truncating all participants to {min_samples} samples.")

# 2. Truncate everyone to this minimum length so np.stack works
all_participants_X = [x[:min_samples] for x in all_participants_X]
all_participants_y = [y[:min_samples] for y in all_participants_y]

# 3. Stack to create 4D Array
X_master = np.stack(all_participants_X, axis=0)
y_master = np.stack(all_participants_y, axis=0)

print(f"\nData Loaded. Master Shapes:")
print(f"X: {X_master.shape} (Part, Samp, Chan, Time)")
print(f"y: {y_master.shape} (Part, Samp)")

# Step #2: FEATURE EXTRACTION
print("\nExtracting Features...")
# We need to transform time-series EEG into feature vectors
# Output shape will be: (num_participants, num_samples, num_channels, num_features)

wrapper = FeatureWrapper()
n_participants, n_samples, n_channels, n_time = X_master.shape

# Pre-allocate array for features
# We calculate n_features by running one dummy sample
dummy_feat = wrapper.compute_features(
    X_master[0, 0, :, :],
    SFREQ,
    desired_features=DESIRED_FEATURES
)
n_features = dummy_feat.shape[1] # Shape is (n_channels, n_features)

X_features = np.zeros((n_participants, n_samples, n_channels, n_features))

for i in range(n_participants):
    print(f"Extracting features for participant {i + 1}...")
    for j in range(n_samples):
        # compute_features takes (n_channels, n_timesteps)
        sample_eeg = X_master[i, j, :, :]
        # We pass data_idx=0 as dummy
        X_features[i, j, :, :] = wrapper.compute_features(
            sample_eeg,
            SFREQ,
            desired_features=DESIRED_FEATURES
        )

print(f"Feature Extraction Complete. Shape: {X_features.shape}")

# Step #3: WITHIN-PARTICIPANT EVALUATION
print("\nCondition 1: Within-Participant Evaluation")
# Train on Subject A, Test on Subject A (5-fold CV)

clf = SVC(kernel='rbf', class_weight='balanced', probability=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []

for i in range(n_participants):
    # Get data for this specific subject
    X_sub = X_features[i]  # (num_samples, num_channels, num_features)
    y_sub = y_master[i]  # (num_samples,)

    # Splitting and evaluation
    metrics = classify_sklearn(
        X_sub,
        y_sub,
        clf,
        cv_splitter=cv,
        normalize=True,
        return_preds=True
    )

    acc = metrics['mean_accuracy']
    accuracies.append(acc)
    print(f"Subject {i + 1} Accuracy: {acc:.4f}")

    # Per-class accuracy
    correct = defaultdict(int)
    total = defaultdict(int)
    for pred, real in metrics["predictions"]:
        total[real] += 1
        if pred == real:
            correct[real] += 1
    accuracy_per_class = {cls: correct[cls] / total[cls] for cls in total}
    for k, v in accuracy_per_class.items():
        print(f"  Accuracy for Class {k}: {(v*100):.1f}%")
    print(f"  Balanced Accuracy: {(np.mean(list(accuracy_per_class.values()))*100):.1f}%")

    # Confusion Matrix
    preds = [p for p, _ in metrics["predictions"]]
    real = [r for _, r in metrics["predictions"]]
    class_names = np.unique(real)
    cm = confusion_matrix(real, preds, labels=class_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    plt.xticks(rotation=0, ha="center", fontsize=8)
    plt.yticks(rotation=0, ha="right", fontsize=8)
    plt.tight_layout()
    plt.show()

print(f"Average Within-Subject Accuracy: {np.mean(accuracies):.4f}")

# Step #4: CROSS-PARTICIPANT EVALUATION
print("\nCondition 2: Cross-Participant Evaluation")
# Train on Subjects A, B, C... Test on Subject Z (Leave-One-Group-Out)

if n_participants < 2:
    print(f"Skipping Cross-Participant evaluation.")
    print(f"Reason: Found only {n_participants} participant. LOGO CV requires at least 2.")
else:
    # Train on Subjects A, B, C... Test on Subject Z (Leave-One-Group-Out)

    # 1. Flatten the arrays
    # From: (n_parts, n_samples, n_ch, n_feat) -> (n_parts*n_samples, n_ch, n_feat)
    X_flat = X_features.reshape(-1, n_channels, n_features)
    y_flat = y_master.reshape(-1)

    # 2. Create Groups Array (Subject IDs)
    # We need an array like [0, 0, ..., 0, 1, 1, ..., 1] matching the flattened data
    groups = np.repeat(np.arange(n_participants), n_samples)

    print(f"Flattened X shape: {X_flat.shape}")
    print(f"Groups shape: {groups.shape}")

    # 3. Define Splitter
    logo = LeaveOneGroupOut()

    # 4. Run Classification
    try:
        metrics_cross = classify_sklearn(
            X_flat,
            y_flat,
            clf,
            cv_splitter=logo,
            groups=groups,
            normalize=True,
            return_preds=True
        )
        print(f"Cross-Subject Accuracy: {metrics_cross['mean_accuracy']:.4f}")

        # Per-class accuracy
        from collections import defaultdict
        correct = defaultdict(int)
        total = defaultdict(int)
        for pred, real in metrics_cross["predictions"]:
            total[real] += 1
            if pred == real:
                correct[real] += 1
        accuracy_per_class = {cls: correct[cls] / total[cls] for cls in total}
        for k, v in accuracy_per_class.items():
            print(f"  Accuracy for Class {k}: {(v*100):.1f}%")
        print(f"  Balanced Accuracy: {(np.mean(list(accuracy_per_class.values()))*100):.1f}%")

        # Confusion Matrix
        preds = [p for p, _ in metrics_cross["predictions"]]
        real = [r for _, r in metrics_cross["predictions"]]
        class_names = np.unique(real)
        cm = confusion_matrix(real, preds, labels=class_names)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(12, 12))
        disp.plot(ax=ax, cmap="Blues", colorbar=True)
        plt.xticks(rotation=0, ha="center", fontsize=8)
        plt.yticks(rotation=0, ha="right", fontsize=8)
        plt.tight_layout()
        plt.show()
    except ValueError as e:
        print(f"Cross-validation failed: {e}")