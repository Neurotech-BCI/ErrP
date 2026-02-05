import serial
import time
import keyboard
import numpy as np
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_byprop, resolve_streams

# Name is set in DSI2LSL, "WS-default" is default name
STREAM_NAME = "WS-default"

streams = resolve_byprop('name', STREAM_NAME, timeout=5)
# keep past 60 seconds, pull 1024 samples at at time
inlet = StreamInlet(streams[0], max_buflen=60, max_chunklen=1024)
info = inlet.info()

# Extract channel count and sampling rate
n_channels = info.channel_count()
srate = info.nominal_srate()

print("Connected to stream:")
print("Name:", info.name())
print("Channels:", n_channels)
print("Sampling rate:", srate)

# Plot Settings
plot_channels = min(8, n_channels)   # plot first 8 channels
window_sec = 5                       # seconds of data in window
buffer_size = int(window_sec * srate)

data_buffer = np.zeros((buffer_size, plot_channels))

# Matplotlib
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))

lines = []
offsets = np.arange(plot_channels) * 100  # vertical offset between channels

for ch in range(plot_channels):
  line, = ax.plot(data_buffer[:, ch] + offsets[ch])
  lines.append(line)

ch = info.desc().child("channels").child("channel")
channel_labels = []

for k in range(plot_channels):
  label = ch.child_value("label")
  channel_labels.append(label if label else f"Ch{k+1}")
  ch = ch.next_sibling()

ax.set_yticks(offsets)
ax.set_yticklabels(channel_labels)

ax.set_ylim(-100, offsets[-1] + 100)
ax.set_xlim(0, buffer_size)
ax.set_title("Live LSL")
ax.set_xlabel("Samples")

# Update loop
try:
  target_fps = 25
  redraw_every = 1.0 / target_fps
  last_draw = 0.0
  while True:
      chunk, timestamps = inlet.pull_chunk(timeout=1.0, max_samples=128)
      if not chunk:
          continue

      chunk = np.asarray(chunk, dtype=np.float32)[:, :plot_channels]
      k = chunk.shape[0]

      if k >= buffer_size:
          data_buffer[:] = chunk[-buffer_size:, :]
      else:
          data_buffer[:-k] = data_buffer[k:]
          data_buffer[-k:] = chunk

      now = time.perf_counter()
      if now - last_draw >= redraw_every:
          for ch in range(plot_channels):
              lines[ch].set_ydata(data_buffer[:, ch] + offsets[ch])

          fig.canvas.draw_idle()
          plt.pause(0.001)
          last_draw = now

      if keyboard.is_pressed('esc'):
          break

except KeyboardInterrupt:
  print("Stopped")
