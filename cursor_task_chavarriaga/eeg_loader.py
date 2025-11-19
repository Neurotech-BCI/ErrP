import mne
import numpy as np
import os
from scipy.io import loadmat

# Define Event and Epoch Parameters; duration of the epoch window: 0.2s before event, 0.8s after event.
PRE_S = 0.2
POST_S = 0.8

# Codes
CORRECT_CODES = {5, 10} 
ERROR_CODES   = {6, 9}
KEEP_CODES    = CORRECT_CODES | ERROR_CODES

# Helper Functions
def map_label(event_code):
    """Maps the dataset's event code to a simple binary label."""
    if event_code in CORRECT_CODES:
        return 0  # Correct
    elif event_code in ERROR_CODES:
        return 1  # Error
    else:
        return -1 # Skip

def _unwrap(x):
    """Unwrap MATLAB cell/1x1 object arrays to the underlying value."""
    # Recursively go into the array until we hit the actual data
    while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x.flat[0]
    return x

def load_participant_data(file_path: str, n_samples_to_keep: int = 500):
    """
    Loads one participant's .mat EEG data containing 10 runs, epochs each run individually using POS and TYP fields,
    applies basic cleaning, concatenates the epochs, and returns the 3D data and 1D labels.

    Args:
        file_path (str): Full path to the .mat file.
        n_samples_to_keep (int): The maximum number of samples to return.

    Returns:
        tuple: (X_data, y_labels)
               X_data: (n_samples, n_channels, n_timesteps)
               y_labels: (n_samples,)
    """
    print(f"\nProcessing file: {os.path.basename(file_path)}")

    mat_data = loadmat(file_path)
    print("Available keys:", list(mat_data.keys()))
    
    run_struct = mat_data['run'].ravel()
    print(f"Found {len(run_struct)} runs.")

    if run_struct.shape[0] != 10:
        raise ValueError(f"Expected 10 runs, but found {run_struct.shape[0]}")
    
    all_X = []
    all_y = []

    for i in range(10):
        print(f"Processing run {i + 1}/10")

        # Get the run structure
        run_i = run_struct[i]

        # Extract Header & SampleRate FIRST
        header = _unwrap(run_i['header'])
        sfreq = _unwrap(header['SampleRate'])

        # Extract EEG Data
        eeg = _unwrap(run_i['eeg'])
        eeg = eeg.T  # Transpose to (Channels, Time) for MNE

        # Check units (Volts vs uV)
        if np.abs(eeg).max() > 1.0:
            eeg = eeg * 1e-6

        # Extract Labels & Events
        try:
            # Unwrap the Label field
            raw_labels = _unwrap(header['Label'])
            # Sometimes labels are further nested, unwrap each one
            ch_names = [str(_unwrap(l)) for l in raw_labels]
        except:
            ch_names = [f"Ch{k + 1}" for k in range(eeg.shape[0])]

        n_data_ch = eeg.shape[0]

        if len(ch_names) != n_data_ch:
            print(f"  Warning: Channel count mismatch (Names: {len(ch_names)}, Data: {n_data_ch}). Adjusting...")

            if len(ch_names) > n_data_ch:
                # We have 65 names but 64 data rows.
                # The extra name is usually a Status/Trigger channel that isn't in the data array.
                # We simply keep the first 64 names.
                ch_names = ch_names[:n_data_ch]
            else:
                # (Rare) We have more data than names. Add dummy names.
                diff = n_data_ch - len(ch_names)
                ch_names.extend([f"Ext{k + 1}" for k in range(diff)])

        event_struct = _unwrap(header['EVENT'])
        pos = _unwrap(event_struct['POS']).flatten()
        typ = _unwrap(event_struct['TYP']).flatten()

        if len(pos) != len(typ):
            print(f"Skipping run {i + 1}: POS/TYP mismatch.")
            continue

        # 5. Create MNE Raw Object
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(eeg, info, verbose=False)

        # 6. Preprocessing
        raw.set_eeg_reference('average', verbose=False)
        raw.filter(0.5, 40., fir_design='firwin', verbose=False)

        # 7. Event Processing
        # Stack events: [sample, 0, type]
        events = np.column_stack((pos, np.zeros_like(pos, dtype=int), typ))

        # Filter for specific codes
        event_mask = np.isin(events[:, 2], list(KEEP_CODES))
        events_filtered = events[event_mask]

        if len(events_filtered) == 0:
            print(f"Run {i + 1}: No valid events, skipping")
            continue

        # 8. Epoching
        tmin = -PRE_S
        tmax = POST_S

        try:
            epochs = mne.Epochs(
                raw,
                events_filtered,
                tmin=tmin,
                tmax=tmax,
                baseline=(-0.2, 0),
                preload=True,
                verbose=False,
                detrend=1
            )

            # Extract data for this run
            X_run = epochs.get_data()
            y_run = np.array([map_label(code) for code in epochs.events[:, 2]])

            if len(X_run) > 0:
                all_X.append(X_run)
                all_y.append(y_run)

        except Exception as e:
            print(f"Epoching failed for run {i + 1}: {e}")
            continue
    
    # Replace the concatenation block in load_participant_data
    if not all_X:
        raise ValueError("No valid epochs found across all runs")

    # Concatenate epoch data
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Sub-select if too many samples
    if X.shape[0] > n_samples_to_keep:
        # Use random choice for representative subsampling
        indices = np.random.choice(X.shape[0], n_samples_to_keep, replace=False)
        X = X[indices]
        y = y[indices]

    print(f"Epoching done. Final X shape: {X.shape}, Final y shape: {y.shape}")

    return X, y

if __name__ == '__main__':
    # Actual file paths [edit]
    TEST_FILE_PATH = "/Users/Selina/Documents/ErrP/data/Subject01_s1.mat"

    # To test, the file must be present at the TEST_FILE_PATH
    if os.path.exists(TEST_FILE_PATH):
        X_test, y_test = load_participant_data(TEST_FILE_PATH)
        print(f"\nLoader Test Successful")
        print(f"X (data) shape: {X_test.shape}")
        print(f"y (labels) shape: {y_test.shape}")
    else:
        print("\nWARNING: Test file not found. Please update TEST_FILE_PATH.")