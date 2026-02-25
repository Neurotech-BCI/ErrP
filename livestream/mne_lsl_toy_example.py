import numpy as np
import time
from mne_lsl.stream import StreamLSL
from mne_lsl.lsl import resolve_streams
import matplotlib.pyplot as plt


streams = resolve_streams(timeout=5.0)
print(f"Found {len(streams)} stream(s):")
for stream in streams:
    print(f"{stream.name} ({stream.stype}) at {stream.sfreq} Hz")

stream = StreamLSL(bufsize=10, name="WS-default", stype="EEG")
stream.connect(acquisition_delay=0.1, timeout=5) # try .connect(acquisition_delay=0.1, timeout=5) if running into issues with sync

### Stream info, sampling frequency, channel count ###
info = stream.info
fs = stream.info["sfreq"]
n_channels = len(info["ch_names"])


### initiate plot ###
plt.ion()
fig,ax = plt.subplots()


while True:
      # data is a 2D array of shape (n_channels, n_samples); 8 columns and 10 seconds worth of data by default (3000 samples)
      # ts are timetamps
      # parameters: winsize, timeout
      data, ts = stream.get_data(winsize = 5)
      
      ax.clear()
      ax.plot(data.T)
      ax.set_title("EEG")
      ax.set_xlabel("Samples")
      ax.set_ylabel("Amplitude (V)")
      plt.pause(0.01)
