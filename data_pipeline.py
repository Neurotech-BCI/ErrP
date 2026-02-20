from pathlib import Path
import mne
import numpy as np

directory_path = Path('./data_edf')
data_files = [str(file_path.resolve()) for file_path in directory_path.rglob('*.edf')]
print(data_files)

def load_raw_edf(edf_file_path):
    data = mne.io.read_raw_edf(edf_file_path, preload = True)
    print(data.info)
    print(f"Original channel names: {data.info.ch_names}")

    ### Reference to Left Ear channel, default is Pz ###
    data.set_eeg_reference(ref_channels=['EEG LE-Pz'])

    ### Standardize channel names ###
    rename_dict = {}
    for ch_name in data.ch_names:
        if '-Pz' in ch_name:
            rename_dict[ch_name] = ch_name.replace('-Pz', '')
        if ch_name == "Pz":
            rename_dict[ch_name] = "EEG Pz"
    data.rename_channels(rename_dict)
    print(f"Updated channel names: {data.info.ch_names}")

    ### Preprocess with bandpass ###
    data.filter(l_freq=0.5, h_freq=50.0, picks="eeg")

    ### Find events labeled in the Trigger column ###
    events = mne.find_events(data, stim_channel='Trigger', min_duration=0.0)
    print(f'Found {len(events)} events')
    print(f'Event IDs: {set(events[:, 2])}')
    
    event_id = {f'event_type_{i}': i for i in set(events[:, 2])}

    ### Create epochs time locked to the event triggers ###
    epochs = mne.Epochs(
        data,
        events,
        event_id=event_id,
        tmin=-0.2,      # Start 200 ms before event
        tmax=0.8,       # End 800 ms after event
        baseline=(-0.2, 0),  # Baseline period for correction
        preload=True
    )
    channels = ['EEG F4', 'EEG C4', 'EEG P4', 'EEG P3', 'EEG C3', 'EEG F3', 'EEG Pz']
    epochs = epochs.pick(channels)

    ### Plot epochs ###
    evokeds = [epochs[cond].average() for cond in list(event_id.keys())]
    mne.viz.plot_compare_evokeds(evokeds)

    ### Convert epochs and event labels to numpy ###
    samples = []
    labels = []
    for key in event_id.keys():
        datum = epochs[key].get_data()[:,:,:-1]
        samples.append(datum)
        labels.extend([key]*datum.shape[0])
    X = np.concatenate(samples, axis=0)
    y = np.array(labels)

    return X, y

for data_file in data_files:
    try:
        X, y = load_raw_edf(data_file)
        print(f"Shape of EEG data (num_trials, num_channels, epoch_len): {X.shape}")
        print(f"Shape of labels array (num_trials,): {y.shape}")
        unique_labels, counts = np.unique(y,return_counts=True)
        print("Dataset label counts:")
        print(np.asarray((unique_labels, counts)).T)
    except Exception as e:
        print(f"Error processing {data_file}: {e}")