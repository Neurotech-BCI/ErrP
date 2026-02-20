import mne
import uuid # for creating a name
import time
import csv
import matplotlib.pyplot as plt

from mne_lsl.datasets import sample
from mne_lsl.player import PlayerLSL
from mne_lsl.stream import StreamLSL, EpochsStream
from mne_lsl.lsl import resolve_streams


source_id = "debug-source-id-001" # can also use name if not using simulated data
filepath = r"/Users/chris/Desktop/coding projects/LSL_testing/eeg_data.csv" # where to write your csv file


# Find all streams
streams = resolve_streams(timeout=5.0)
print(f"Found {len(streams)} stream(s):")
for stream in streams:
    print(f"  - {stream.name} ({stream.stype}) @ {stream.sfreq} Hz")

# Open stream
#stream = StreamLSL(bufsize=10, name="WS-default", stype="EEG") for wearable
stream = StreamLSL(bufsize=10, source_id=source_id).connect()
stream.filter(l_freq=1.0, h_freq=40.0, picks="eeg")

print()
print(stream.info)


# Prepare csv
#with open(filepath, "w", newline="") as f:
#  writer = csv.writer(f)
#  writer.writerow(["time"] + stream.ch_names)

plt.ion()
fig,ax = plt.subplots()

try:
    with open(filepath, "w", newline="", buffering=1024*1024) as f:
        writer = csv.writer(f)
        writer.writerow(["time"] + stream.ch_names)

        print(stream.ch_names)
        print(stream.get_channel_types())
        print("Streaming...")

        while True:
            n_new = stream.n_new_samples
            if n_new > 0:
                winsize = n_new / stream.info['sfreq']
                # data is a 2D array of shape (n_channels, n_samples) where each row is a channel and each column is a sample
                # custom winsize to only extract new data samples
                #data, ts = stream.get_data(winsize=winsize, picks="stim") 
                data, ts = stream.get_data(winsize=5) 
                #print(data[0].shape)
                for i in range(data.shape[1]):
                    row = [ts[i]] + data[:, i].tolist()
                    writer.writerow(row)
    
            ax.clear()
            ax.plot(data.T)
            ax.set_title("EEG thing")
            ax.set_xlabel("Samples")
            ax.set_ylabel("Amplitude (V)")
            plt.pause(0.01)

except KeyboardInterrupt:
    print("Stopping acquisition")
finally:
    stream.disconnect()

"""
batch = []
...
if n_new > 0:
    data, ts = stream.get_data(n_new / stream.info["sfreq"])
    for i in range(data.shape[1]):
        batch.append([ts[i]] + data[:, i].tolist())

    if len(batch) >= 5000:
        writer.writerows(batch)
        batch.clear()
"""
