from __future__ import annotations

import argparse
import ctypes
import logging
import pickle
import platform
from collections import Counter
from ctypes import wintypes
from datetime import datetime

import numpy as np
from psychopy import core, event, visual

from mne_lsl.stream import StreamLSL

from config import (
	EEGConfig,
	LSLConfig,
	MentalCommandLabelConfig,
	MentalCommandModelConfig,
	MICursorTaskConfig,
	StimConfig,
)
from derick_ml_jawclench import (
	run_visual_jaw_calibration,
	select_jaw_channel_indices,
	update_live_jaw_clench_state,
)
from mental_command_worker import (
	StreamingIIRFilter,
	canonicalize_channel_name,
	resolve_channel_order,
	train_or_load_shared_mi_model,
)


class _Point(ctypes.Structure):
	_fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


def _make_task_logger(fname: str) -> logging.Logger:
	logger = logging.getLogger(f"real_cursor.{fname}")
	logger.setLevel(logging.INFO)
	logger.propagate = False
	if logger.handlers:
		return logger

	formatter = logging.Formatter(
		fmt="%(asctime)s | %(levelname)s | %(message)s",
		datefmt="%H:%M:%S",
	)

	file_handler = logging.FileHandler(f"{fname}_real_cursor.log", mode="w")
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)

	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)
	return logger


def _sanitize_participant_name(raw_name: str) -> str:
	cleaned = "_".join(raw_name.strip().lower().split())
	cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch == "_")
	return cleaned.strip("_")


def _build_session_prefix(participant: str) -> str:
	date_prefix = datetime.now().strftime("%m_%d_%y")
	return f"{date_prefix}_{participant}_real_cursor"


def _prompt_session_prefix() -> str:
	while True:
		raw = input("Enter participant name: ")
		participant = _sanitize_participant_name(raw)
		if participant:
			return _build_session_prefix(participant)
		print("Participant name cannot be empty. Please try again.")


def _apply_deadband(value: float, deadband: float) -> float:
	value = float(np.clip(value, -1.0, 1.0))
	deadband = float(np.clip(deadband, 0.0, 0.99))
	if abs(value) <= deadband:
		return 0.0
	scaled = (abs(value) - deadband) / (1.0 - deadband)
	return float(np.copysign(scaled, value))


def _get_windows_cursor_pos() -> tuple[int, int]:
	pt = _Point()
	if not ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
		raise RuntimeError("GetCursorPos failed")
	return int(pt.x), int(pt.y)


def _set_windows_cursor_pos(x: int, y: int) -> None:
	if not ctypes.windll.user32.SetCursorPos(int(x), int(y)):
		raise RuntimeError("SetCursorPos failed")


def _move_windows_cursor_relative(dx: int, dy: int, enabled: bool) -> None:
	if not enabled:
		return
	x, y = _get_windows_cursor_pos()
	_set_windows_cursor_pos(x + int(dx), y + int(dy))


def run_task(fname: str, pixels_per_update: int = 30, dry_run: bool = False) -> None:
	logger = _make_task_logger(fname)
	lsl_cfg = LSLConfig()
	stim_cfg = StimConfig()
	label_cfg = MentalCommandLabelConfig()
	task_cfg = MICursorTaskConfig()
	model_cfg = MentalCommandModelConfig()
	eeg_cfg = EEGConfig(
		picks=("Pz", "F4", "C4", "P4", "P3", "C3", "F3"),
		l_freq=8.0,
		h_freq=30.0,
		reject_peak_to_peak=150.0,
	)

	is_windows = platform.system().lower().startswith("win")
	cursor_enabled = is_windows and not bool(dry_run)
	if not is_windows:
		logger.warning("Non-Windows OS detected; cursor movement will run in dry-run mode.")
	if dry_run:
		logger.info("Dry-run mode enabled; cursor movement calls are disabled.")

	logger.info(
		"Starting real cursor control | pixels_per_update=%d | cursor_enabled=%s",
		int(pixels_per_update),
		bool(cursor_enabled),
	)

	stream = StreamLSL(
		bufsize=60.0,
		name=lsl_cfg.name,
		stype=lsl_cfg.stype,
		source_id=lsl_cfg.source_id,
	)
	stream.connect(acquisition_delay=0.001, processing_flags="all")
	logger.info("Connected to LSL stream: info=%s", stream.info)

	available = list(stream.info["ch_names"])
	model_ch_names, missing = resolve_channel_order(available, eeg_cfg.picks)
	if len(model_ch_names) < 2:
		event_key = canonicalize_channel_name(lsl_cfg.event_channels)
		model_ch_names = [ch for ch in available if canonicalize_channel_name(ch) != event_key]
	if len(model_ch_names) < 2:
		raise RuntimeError(f"Need >=2 EEG channels, found: {available}")

	stream.pick(model_ch_names)
	sfreq = float(stream.info["sfreq"])
	logger.info(
		"Using live EEG channels: sfreq=%.3f, selected=%s, missing_configured=%s",
		sfreq,
		list(stream.info["ch_names"]),
		missing,
	)

	logger.info(
		"Preparing offline LR model exactly like jaw_pause: data_dir=%s, edf_glob=%s, window_s=%.3f, step_s=%.3f",
		task_cfg.data_dir,
		task_cfg.edf_glob,
		task_cfg.window_s,
		task_cfg.window_step_s,
	)
	shared_model = train_or_load_shared_mi_model(
		cache_name="mi_shared_lr_model",
		data_dir=task_cfg.data_dir,
		edf_glob=task_cfg.edf_glob,
		calibrate_on_participant=task_cfg.calirate_on_participant,
		eeg_cfg=eeg_cfg,
		task_cfg=task_cfg,
		stim_cfg=stim_cfg,
		model_cfg=model_cfg,
		target_sfreq=float(sfreq),
		target_channel_names=model_ch_names,
		logger=logger,
	)
	classifier = shared_model.classifier
	class_index = shared_model.class_index
	classifier_classes = np.asarray(classifier.named_steps["clf"].classes_, dtype=int)
	if int(stim_cfg.left_code) not in class_index or int(stim_cfg.right_code) not in class_index:
		raise RuntimeError(
			f"Classifier classes {classifier_classes.tolist()} do not contain expected left/right codes "
			f"{[int(stim_cfg.left_code), int(stim_cfg.right_code)]}."
		)
	counts = Counter(shared_model.class_counts)
	logger.info(
		"LR model ready: class_counts=%s, loso_mean=%.4f, loso_std=%.4f",
		counts,
		shared_model.loso.mean_accuracy,
		shared_model.loso.std_accuracy,
	)

	with open(f"{fname}_real_cursor_lr_model.pkl", "wb") as fh:
		pickle.dump(classifier, fh)

	win = visual.Window(
		size=(980, 620),
		color=(-0.08, -0.08, -0.08),
		units="norm",
		fullscr=False,
	)
	title = visual.TextStim(win, text="Real Cursor Control", pos=(0, 0.88), height=0.06, color=(0.92, 0.92, 0.92))
	cue = visual.TextStim(win, text="", pos=(0, 0.72), height=0.05, color=(0.92, 0.92, 0.92))
	info = visual.TextStim(win, text="", pos=(0, -0.78), height=0.04, color=(0.84, 0.90, 0.96))
	status = visual.TextStim(win, text="", pos=(0, -0.88), height=0.04, color=(0.84, 0.84, 0.84))

	pred_clock = core.Clock()
	live_filter = StreamingIIRFilter.from_eeg_config(
		eeg_cfg=eeg_cfg,
		sfreq=sfreq,
		n_channels=len(model_ch_names),
	)
	live_buffer = np.empty((len(model_ch_names), 0), dtype=np.float32)
	jaw_buffer = np.empty((len(model_ch_names), 0), dtype=np.float32)
	keep_n = int(round((task_cfg.window_s + task_cfg.filter_context_s) * sfreq))
	window_n = int(round(task_cfg.window_s * sfreq))
	jaw_window_n = int(round(0.60 * sfreq))
	stream_pull_s = max(0.08, task_cfg.live_update_interval_s * 2.0)
	reject_thresh = eeg_cfg.reject_peak_to_peak
	last_live_ts: float | None = None

	jaw_idxs = select_jaw_channel_indices(model_ch_names)
	jaw_prob = 0.0
	jaw_prev_pred = 0
	jaw_prob_thresh = 0.70
	jaw_refractory_s = 0.70
	jaw_last_toggle_t = -1e9

	prediction_count = 0
	left_prob = 0.5
	right_prob = 0.5
	raw_command = 0.0
	ema_command = 0.0
	latest_pred_code: int | None = None
	live_note = "warming up"

	axis_mode = "lr"

	def _draw_frame() -> None:
		title.draw()
		cue.draw()
		info.draw()
		status.draw()
		win.flip()

	def _wait_for_seconds(duration_s: float) -> None:
		clock = core.Clock()
		while clock.getTime() < duration_s:
			if "escape" in event.getKeys():
				raise KeyboardInterrupt
			_draw_frame()

	def _wait_for_space(prompt_text: str) -> None:
		cue.text = prompt_text
		while True:
			keys = event.getKeys()
			if "escape" in keys:
				raise KeyboardInterrupt
			if "space" in keys:
				return
			_draw_frame()

	def _collect_stream_block(duration_s: float) -> np.ndarray:
		chunks: list[np.ndarray] = []
		last_ts_local: float | None = None
		clock = core.Clock()
		while clock.getTime() < duration_s:
			if "escape" in event.getKeys():
				raise KeyboardInterrupt
			data, ts = stream.get_data(winsize=min(0.20, duration_s), picks="all")
			if data.size > 0 and ts is not None and len(ts) > 0:
				ts_arr = np.asarray(ts)
				mask = np.ones_like(ts_arr, dtype=bool) if last_ts_local is None else (ts_arr > float(last_ts_local))
				if np.any(mask):
					chunks.append(np.asarray(data[:, mask], dtype=np.float32))
					last_ts_local = float(ts_arr[mask][-1])
			_draw_frame()
		if not chunks:
			return np.empty((len(model_ch_names), 0), dtype=np.float32)
		return np.concatenate(chunks, axis=1).astype(np.float32, copy=False)

	try:
		jaw_classifier, jaw_train_acc, _y_np = run_visual_jaw_calibration(
			cue=cue,
			info=info,
			status=status,
			wait_for_space=_wait_for_space,
			wait_for_seconds=_wait_for_seconds,
			collect_stream_block=_collect_stream_block,
			jaw_idxs=jaw_idxs,
			jaw_window_n=jaw_window_n,
			model_ch_names=model_ch_names,
			logger=logger,
			n_per_class=5,
			hold_s=1.2,
			prep_s=2.5,
			iti_s=1.5,
			min_total_samples=6,
		)
		with open(f"{fname}_real_cursor_jaw_model.pkl", "wb") as fh:
			pickle.dump(jaw_classifier, fh)

		logger.info("Jaw model ready: train_acc=%.3f", jaw_train_acc)

		cue.text = "Live control"
		info.text = (
			"Use LEFT/RIGHT MI to move cursor. Jaw clench toggles axis: LR <-> UD. "
			"Press ESC to stop."
		)
		status.text = "Press SPACE to start."
		_wait_for_space("Live control")

		while True:
			keys = event.getKeys()
			if "escape" in keys:
				raise KeyboardInterrupt

			data, ts = stream.get_data(winsize=stream_pull_s, picks="all")
			if data.size > 0 and ts is not None and len(ts) > 0:
				ts_arr = np.asarray(ts)
				mask = np.ones_like(ts_arr, dtype=bool) if last_live_ts is None else (ts_arr > float(last_live_ts))
				if np.any(mask):
					x_new = np.asarray(data[:, mask], dtype=np.float32)
					last_live_ts = float(ts_arr[mask][-1])

					jaw_buffer, jaw_prob, jaw_prev_pred, should_toggle = update_live_jaw_clench_state(
						jaw_buffer=jaw_buffer,
						x_new=x_new,
						keep_n=keep_n,
						jaw_window_n=jaw_window_n,
						jaw_classifier=jaw_classifier,
						jaw_idxs=jaw_idxs,
						jaw_prob=jaw_prob,
						jaw_prev_pred=jaw_prev_pred,
						jaw_prob_thresh=jaw_prob_thresh,
						jaw_last_toggle_t=jaw_last_toggle_t,
						jaw_refractory_s=jaw_refractory_s,
						now_t=core.getTime(),
					)
					if should_toggle:
						axis_mode = "ud" if axis_mode == "lr" else "lr"
						jaw_last_toggle_t = core.getTime()
						logger.info("Axis toggled to %s (jaw_p=%.2f)", axis_mode, jaw_prob)

					x_new_filt = live_filter.process(x_new)
					live_buffer = np.concatenate((live_buffer, x_new_filt), axis=1)
					if live_buffer.shape[1] > keep_n:
						live_buffer = live_buffer[:, -keep_n:]

			if pred_clock.getTime() < task_cfg.live_update_interval_s:
				status.text = (
					f"axis={axis_mode.upper()}  jaw_p={jaw_prob:.2f}  updates={prediction_count}  {live_note}"
				)
				_draw_frame()
				continue

			pred_clock.reset()
			if live_buffer.shape[1] < window_n:
				needed_s = max(0.0, (window_n - live_buffer.shape[1]) / sfreq)
				raw_command = 0.0
				ema_command *= 0.95
				latest_pred_code = None
				live_note = f"warming up ({needed_s:.1f}s)"
			else:
				x_win = live_buffer[:, -window_n:]
				max_ptp = float(np.ptp(x_win, axis=-1).max())
				if reject_thresh is not None and max_ptp > float(reject_thresh):
					raw_command = 0.0
					ema_command *= 0.90
					latest_pred_code = None
					live_note = "artifact reject"
				else:
					p_vec = classifier.predict_proba(x_win[np.newaxis, ...])[0]
					left_prob = float(p_vec[class_index[int(stim_cfg.left_code)]])
					right_prob = float(p_vec[class_index[int(stim_cfg.right_code)]])
					raw_command = float(np.clip(right_prob - left_prob, -1.0, 1.0))
					alpha = float(np.clip(task_cfg.command_ema_alpha, 0.0, 1.0))
					if prediction_count == 0:
						ema_command = raw_command
					else:
						ema_command = (1.0 - alpha) * ema_command + alpha * raw_command

					if abs(ema_command) < float(task_cfg.command_deadband):
						latest_pred_code = None
					elif ema_command > 0.0:
						latest_pred_code = int(stim_cfg.right_code)
					else:
						latest_pred_code = int(stim_cfg.left_code)

					command_drive = _apply_deadband(ema_command, float(task_cfg.command_deadband))
					step = int(round(command_drive * float(pixels_per_update)))
					if step != 0:
						if axis_mode == "lr":
							dx, dy = int(step), 0
						else:
							# Positive command (RIGHT) maps to UP in UD mode.
							dx, dy = 0, int(-step)
						_move_windows_cursor_relative(dx=dx, dy=dy, enabled=cursor_enabled)

					prediction_count += 1
					live_note = "tracking"

					if prediction_count % 20 == 0:
						logger.info(
							"Decode %d: axis=%s, left_p=%.4f, right_p=%.4f, raw=%.4f, ema=%.4f, jaw_p=%.4f",
							prediction_count,
							axis_mode,
							left_prob,
							right_prob,
							raw_command,
							ema_command,
							jaw_prob,
						)

			cue.text = "Live control"
			info.text = (
				f"axis={axis_mode.upper()}  {label_cfg.left_name}={left_prob:.2f}  {label_cfg.right_name}={right_prob:.2f}  "
				f"raw={raw_command:+.2f}  ema={ema_command:+.2f}"
			)
			status.text = (
				f"pred={latest_pred_code}  jaw_p={jaw_prob:.2f}  updates={prediction_count}  {live_note}"
			)
			_draw_frame()

	except KeyboardInterrupt:
		logger.info("Session interrupted by user.")
	finally:
		try:
			stream.disconnect()
		except Exception:
			pass
		try:
			win.close()
		except Exception:
			pass


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description=(
			"Move Windows cursor with LR MI decoding; jaw clench toggles axis between left/right and up/down."
		)
	)
	parser.add_argument("--pixels-per-update", type=int, default=30, help="Cursor pixels moved per decode update at full command")
	parser.add_argument("--dry-run", action="store_true", help="Run decoding without moving OS cursor")
	args = parser.parse_args()

	fname = _prompt_session_prefix()
	print(f"[SESSION] Using filename prefix: {fname}")
	run_task(fname=fname, pixels_per_update=int(args.pixels_per_update), dry_run=bool(args.dry_run))
