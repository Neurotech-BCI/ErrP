from __future__ import annotations

import logging
import pickle
import random
from collections import Counter
from datetime import datetime
from typing import Optional

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
from mental_command_worker import (
    StreamingIIRFilter,
    append_windows_to_dataset,
    canonicalize_channel_name,
    evaluate_loso_sessions,
    load_offline_mi_dataset,
    make_mi_classifier,
    prepare_continuous_windows,
    resolve_channel_order,
    train_or_load_shared_mi_model,
)
from jaw_clench_detector import JawClenchDetector


# ---------------------------------------------------------------------------
# Tetris geometry constants
# ---------------------------------------------------------------------------

BOARD_COLS = 10
BOARD_ROWS = 20

TETROMINOES: dict[str, list[list[tuple[int, int]]]] = {
    "I": [
        [(0, -1), (0, 0), (0, 1), (0, 2)],
        [(-1, 0), (0, 0), (1, 0), (2, 0)],
        [(0, -1), (0, 0), (0, 1), (0, 2)],
        [(-1, 0), (0, 0), (1, 0), (2, 0)],
    ],
    "O": [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ],
    "T": [
        [(0, -1), (0, 0), (0, 1), (1, 0)],
        [(-1, 0), (0, 0), (1, 0), (0, 1)],
        [(-1, 0), (0, -1), (0, 0), (0, 1)],
        [(-1, 0), (0, -1), (0, 0), (1, 0)],
    ],
    "S": [
        [(0, 0), (0, 1), (1, -1), (1, 0)],
        [(-1, 0), (0, 0), (0, 1), (1, 1)],
        [(0, 0), (0, 1), (1, -1), (1, 0)],
        [(-1, 0), (0, 0), (0, 1), (1, 1)],
    ],
    "Z": [
        [(0, -1), (0, 0), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, -1), (1, 0)],
        [(0, -1), (0, 0), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, -1), (1, 0)],
    ],
    "L": [
        [(0, -1), (0, 0), (0, 1), (1, -1)],
        [(-1, 0), (0, 0), (1, 0), (1, 1)],
        [(-1, 1), (0, -1), (0, 0), (0, 1)],
        [(-1, -1), (-1, 0), (0, 0), (1, 0)],
    ],
    "J": [
        [(0, -1), (0, 0), (0, 1), (1, 1)],
        [(0, 0), (1, -1), (1, 0), (-1, 0)],
        [(-1, -1), (0, -1), (0, 0), (0, 1)],
        [(-1, 0), (-1, 1), (0, 0), (1, 0)],
    ],
}

PIECE_COLORS: dict[str, tuple[float, float, float]] = {
    "I": (0.00, 0.90, 0.90),
    "O": (0.95, 0.85, 0.10),
    "T": (0.70, 0.20, 0.90),
    "S": (0.10, 0.90, 0.20),
    "Z": (0.95, 0.20, 0.20),
    "L": (0.95, 0.55, 0.10),
    "J": (0.10, 0.30, 0.95),
}

BASE_DROP_INTERVAL_S = 1.0
LEVEL_SPEEDUP = 0.08


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _make_task_logger(fname: str) -> logging.Logger:
    logger = logging.getLogger(f"tetris.{fname}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(f"{fname}_tetris.log", mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def _sanitize(raw: str) -> str:
    cleaned = "_".join(raw.strip().lower().split())
    cleaned = "".join(c for c in cleaned if c.isalnum() or c == "_")
    return cleaned.strip("_")


def _prompt_prefix() -> str:
    while True:
        raw = input("Enter participant name: ")
        p = _sanitize(raw)
        if p:
            return f"{datetime.now().strftime('%m_%d_%y')}_{p}_tetris"
        print("Name cannot be empty.")


# ---------------------------------------------------------------------------
# Tetris board logic
# ---------------------------------------------------------------------------

class TetrisBoard:
    def __init__(self, rows: int = BOARD_ROWS, cols: int = BOARD_COLS) -> None:
        self.rows = rows
        self.cols = cols
        self.grid: list[list[Optional[str]]] = [[None] * cols for _ in range(rows)]
        self.score = 0
        self.lines = 0
        self.level = 1
        self.game_over = False

        self._bag: list[str] = []
        self.current_type: str = ""
        self.current_rot: int = 0
        self.current_row: int = 0
        self.current_col: int = 0
        self.next_type: str = ""

        self._spawn_next()
        self._spawn_next()

    def _refill_bag(self) -> None:
        self._bag = list(TETROMINOES.keys())
        random.shuffle(self._bag)

    def _next_piece(self) -> str:
        if not self._bag:
            self._refill_bag()
        return self._bag.pop()

    def _spawn_next(self) -> None:
        self.current_type = self.next_type if self.next_type else self._next_piece()
        self.next_type = self._next_piece()
        self.current_rot = 0
        self.current_row = 0
        self.current_col = self.cols // 2
        if not self._fits(self.current_row, self.current_col, self.current_rot):
            self.game_over = True

    def _cells(self, row: int, col: int, rot: int) -> list[tuple[int, int]]:
        offsets = TETROMINOES[self.current_type][rot]
        return [(row + dr, col + dc) for dr, dc in offsets]

    def _fits(self, row: int, col: int, rot: int) -> bool:
        for r, c in self._cells(row, col, rot):
            if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
                return False
            if self.grid[r][c] is not None:
                return False
        return True

    def move_left(self) -> bool:
        if self._fits(self.current_row, self.current_col - 1, self.current_rot):
            self.current_col -= 1
            return True
        return False

    def move_right(self) -> bool:
        if self._fits(self.current_row, self.current_col + 1, self.current_rot):
            self.current_col += 1
            return True
        return False

    def rotate(self) -> bool:
        new_rot = (self.current_rot + 1) % 4
        for dc in (0, -1, 1, -2, 2):
            if self._fits(self.current_row, self.current_col + dc, new_rot):
                self.current_rot = new_rot
                self.current_col += dc
                return True
        return False

    def drop_one(self) -> bool:
        if self._fits(self.current_row + 1, self.current_col, self.current_rot):
            self.current_row += 1
            return True
        self._lock()
        return False

    def _lock(self) -> None:
        for r, c in self._cells(self.current_row, self.current_col, self.current_rot):
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.grid[r][c] = self.current_type
        cleared = self._clear_lines()
        self.lines += cleared
        self.score += [0, 100, 300, 500, 800][min(cleared, 4)] * self.level
        self.level = self.lines // 10 + 1
        self._spawn_next()

    def _clear_lines(self) -> int:
        new_grid = [row for row in self.grid if any(c is None for c in row)]
        cleared = self.rows - len(new_grid)
        for _ in range(cleared):
            new_grid.insert(0, [None] * self.cols)
        self.grid = new_grid
        return cleared

    @property
    def current_cells(self) -> list[tuple[int, int]]:
        return self._cells(self.current_row, self.current_col, self.current_rot)


# ---------------------------------------------------------------------------
# Main task
# ---------------------------------------------------------------------------

def run_task(fname: str) -> None:  # noqa: C901
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

    stream = StreamLSL(
        bufsize=60.0,
        name=lsl_cfg.name,
        stype=lsl_cfg.stype,
        source_id=lsl_cfg.source_id,
    )
    stream.connect(acquisition_delay=0.001, processing_flags="all")
    logger.info("Connected: %s", stream.info)

    available = list(stream.info["ch_names"])
    model_ch_names, missing = resolve_channel_order(available, eeg_cfg.picks)
    if len(model_ch_names) < 2:
        event_key = canonicalize_channel_name(lsl_cfg.event_channels)
        model_ch_names = [ch for ch in available if canonicalize_channel_name(ch) != event_key]
    if len(model_ch_names) < 2:
        raise RuntimeError(f"Need >=2 EEG channels, found: {available}")

    stream.pick(model_ch_names)
    sfreq = float(stream.info["sfreq"])
    stream_ch_names = list(stream.info["ch_names"])
    logger.info("Channels: sfreq=%.1f  selected=%s  missing=%s", sfreq, stream_ch_names, missing)

    jaw_ch_idx = len(model_ch_names) - 1
    jaw_detector = JawClenchDetector(fs=sfreq)
    jaw_detector.refractory = 0.75
    jaw_thresh_min = 8.0
    jaw_calibration_duration_s = 5.0

    CELL = 0.085
    BOARD_W = BOARD_COLS * CELL
    BOARD_H = BOARD_ROWS * CELL
    BOARD_LEFT = -BOARD_W / 2
    BOARD_TOP = BOARD_H / 2

    # Use a larger window so the instruction area does not overlap the board.
    win = visual.Window(
        size=(1280, 900),
        color=(0.18, 0.18, 0.18),
        units="norm",
        fullscr=False,
    )

    def cell_xy(row: int, col: int) -> tuple[float, float]:
        x = BOARD_LEFT + (col + 0.5) * CELL
        y = BOARD_TOP - (row + 0.5) * CELL
        return float(x), float(y)

    MAX_RECTS = BOARD_ROWS * BOARD_COLS + 4 + 4
    _rects: list[visual.Rect] = []
    for _ in range(MAX_RECTS):
        r = visual.Rect(
            win,
            width=CELL * 0.90,
            height=CELL * 0.90,
            pos=(0, 0),
            fillColor=(0, 0, 0),
            lineColor=None,
        )
        _rects.append(r)

    def _color(name: Optional[str]) -> tuple[float, float, float]:
        if name is None:
            return (-0.80, -0.80, -0.80)
        return PIECE_COLORS[name]

    board_outline = visual.Rect(
        win,
        width=BOARD_W + CELL * 0.12,
        height=BOARD_H + CELL * 0.12,
        pos=(0, 0),
        fillColor=None,
        lineColor=(0.30, 0.30, 0.30),
        lineWidth=2,
    )

    txt_score = visual.TextStim(win, text="", pos=(0.72, 0.70), height=0.055, color=(0.90, 0.90, 0.90), anchorHoriz="center")
    txt_level = visual.TextStim(win, text="", pos=(0.72, 0.55), height=0.045, color=(0.70, 0.70, 0.70), anchorHoriz="center")
    txt_lines = visual.TextStim(win, text="", pos=(0.72, 0.42), height=0.045, color=(0.70, 0.70, 0.70), anchorHoriz="center")
    txt_next = visual.TextStim(win, text="NEXT", pos=(0.72, 0.28), height=0.040, color=(0.60, 0.60, 0.60), anchorHoriz="center")
    txt_bci = visual.TextStim(win, text="", pos=(0, -0.88), height=0.038, color=(0.50, 0.80, 1.00), anchorHoriz="center")
    txt_cue = visual.TextStim(win, text="", pos=(0, 0.95), height=0.050, color=(0.90, 0.90, 0.90), anchorHoriz="center")
    txt_status = visual.TextStim(win, text="", pos=(0, -0.95), height=0.035, color=(0.70, 0.70, 0.70), anchorHoriz="center")

    _next_rects = [
        visual.Rect(win, width=CELL * 0.85, height=CELL * 0.85, pos=(0, 0), fillColor=(0, 0, 0), lineColor=None)
        for _ in range(4)
    ]

    def _draw_board(board: TetrisBoard) -> None:
        rect_idx = 0
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                cell_val = board.grid[row][col]
                r = _rects[rect_idx]
                rect_idx += 1
                r.pos = cell_xy(row, col)
                r.fillColor = _color(cell_val)
                r.draw()
        for row, col in board.current_cells:
            if rect_idx >= len(_rects):
                break
            r = _rects[rect_idx]
            rect_idx += 1
            r.pos = cell_xy(row, col)
            r.fillColor = _color(board.current_type)
            r.draw()

    def _draw_next(board: TetrisBoard) -> None:
        offsets = TETROMINOES[board.next_type][0]
        col_c = BOARD_COLS + 1.5
        row_c = -3.0
        for i, (dr, dc) in enumerate(offsets):
            x = BOARD_LEFT + (col_c + dc) * CELL + CELL * 2.0
            y = BOARD_TOP - (row_c + dr) * CELL - CELL * 8.5
            _next_rects[i].pos = (x, y)
            _next_rects[i].fillColor = _color(board.next_type)
            _next_rects[i].draw()

    EMPTY_CELL_COLOR = (-0.45, -0.45, -0.45)

    def _draw_empty_board_screen(cue: str, status: str = "") -> None:
        board_outline.draw()
        rect_idx = 0
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                r = _rects[rect_idx]
                rect_idx += 1
                r.pos = cell_xy(row, col)
                r.fillColor = EMPTY_CELL_COLOR
                r.draw()

        txt_score.text = "Score\n0"
        txt_level.text = "Level\n1"
        txt_lines.text = "Lines\n0"
        txt_next.text = "NEXT"
        txt_bci.text = ""
        txt_cue.text = cue
        txt_status.text = status

        txt_score.draw()
        txt_level.draw()
        txt_lines.draw()
        txt_next.draw()
        txt_bci.draw()
        txt_cue.draw()
        txt_status.draw()
        win.flip()

    def _draw_frame(board: TetrisBoard) -> None:
        board_outline.draw()
        _draw_board(board)
        _draw_next(board)
        txt_score.draw()
        txt_level.draw()
        txt_lines.draw()
        txt_next.draw()
        txt_bci.draw()
        txt_cue.draw()
        txt_status.draw()
        win.flip()

    classifier = None
    dataset = None
    class_index: dict[int, int] | None = None
    rest_session_id: int | None = None
    online_cal_session_id: int | None = None
    train_only_session_ids: set[int] = set()
    use_shared_model_cache = (
        not bool(task_cfg.enable_online_rest_calibration)
        and not bool(task_cfg.enable_online_lr_calibration)
    )

    _draw_empty_board_screen(
        "Preparing MI model...",
        "Loading offline EDF sessions.",
    )

    try:
        if use_shared_model_cache:
            shared_model = train_or_load_shared_mi_model(
                cache_name="mi_shared_lr_model",
                data_dir=task_cfg.data_dir,
                edf_glob=task_cfg.edf_glob,
                calibrate_on_participant=task_cfg.calirate_on_participant,
                eeg_cfg=eeg_cfg,
                task_cfg=task_cfg,
                stim_cfg=stim_cfg,
                model_cfg=model_cfg,
                target_sfreq=sfreq,
                target_channel_names=model_ch_names,
                logger=logger,
            )
            classifier = shared_model.classifier
            class_index = shared_model.class_index
            loso = shared_model.loso
            counts = Counter(shared_model.class_counts)
            logger.info(
                "Model ready from shared cache: loso_mean=%.4f, loso_std=%.4f, class_counts=%s, cache=%s",
                loso.mean_accuracy,
                loso.std_accuracy,
                counts,
                shared_model.cache_path,
            )
        else:
            dataset = load_offline_mi_dataset(
                data_dir=task_cfg.data_dir,
                edf_glob=task_cfg.edf_glob,
                eeg_cfg=eeg_cfg,
                task_cfg=task_cfg,
                stim_cfg=stim_cfg,
                target_sfreq=sfreq,
                target_channel_names=model_ch_names,
                calibrateOnParticipant=task_cfg.calirate_on_participant,
            )

            def _collect_block_simple(dur: float) -> np.ndarray:
                chunks: list[np.ndarray] = []
                last_ts = None
                clk = core.Clock()
                while clk.getTime() < dur:
                    if "escape" in event.getKeys():
                        raise KeyboardInterrupt
                    data, ts = stream.get_data(winsize=min(0.25, dur), picks="all")
                    if data.size > 0 and ts is not None:
                        ts_arr = np.asarray(ts)
                        mask = np.ones_like(ts_arr, bool) if last_ts is None else (ts_arr > float(last_ts))
                        if np.any(mask):
                            chunks.append(np.asarray(data[:, mask], np.float32))
                            last_ts = float(ts_arr[mask][-1])
                    _draw_empty_board_screen(
                        "REST calibration",
                        f"Relax for {max(0.0, dur - clk.getTime()):0.1f}s",
                    )
                return np.concatenate(chunks, axis=1) if chunks else np.empty((len(model_ch_names), 0), np.float32)

            if bool(task_cfg.enable_online_rest_calibration):
                _draw_empty_board_screen(
                    "REST calibration",
                    "Press SPACE to start. ESC to quit.",
                )
                while True:
                    keys = event.getKeys()
                    if "escape" in keys:
                        raise KeyboardInterrupt
                    if "space" in keys:
                        break
                    win.flip()

                rest_block = _collect_block_simple(float(task_cfg.rest_calibration_duration_s))
                rest_wins = prepare_continuous_windows(
                    raw_block=rest_block,
                    eeg_cfg=eeg_cfg,
                    sfreq=sfreq,
                    window_s=float(task_cfg.window_s),
                    step_s=float(task_cfg.window_step_s),
                    reject_peak_to_peak=eeg_cfg.reject_peak_to_peak,
                )
                if rest_wins.shape[0] > 0:
                    rest_session_id = -1
                    train_only_session_ids.add(rest_session_id)
                    rest_labels = np.full(rest_wins.shape[0], int(task_cfg.rest_class_code), dtype=int)
                    dataset = append_windows_to_dataset(dataset, rest_wins, rest_labels, rest_session_id, n_trials_add=1)

            if bool(task_cfg.enable_online_lr_calibration):
                _draw_empty_board_screen(
                    "Live LR calibration",
                    "Press SPACE to begin. ESC to quit.",
                )
                while True:
                    keys = event.getKeys()
                    if "escape" in keys:
                        raise KeyboardInterrupt
                    if "space" in keys:
                        break
                    win.flip()

                online_cal_session_id = int(np.max(dataset.session_ids)) + 1
                cal_codes = (
                    [int(stim_cfg.left_code)] * int(task_cfg.online_lr_calibration_reps_per_class)
                    + [int(stim_cfg.right_code)] * int(task_cfg.online_lr_calibration_reps_per_class)
                )
                random.shuffle(cal_codes)
                cal_windows_list: list[np.ndarray] = []
                cal_labels_list: list[np.ndarray] = []

                for idx, code in enumerate(cal_codes, 1):
                    cname = label_cfg.left_name if int(code) == int(stim_cfg.left_code) else label_cfg.right_name
                    _draw_empty_board_screen(f"Prepare: {cname}", f"Block {idx}/{len(cal_codes)}")
                    clk = core.Clock()
                    while clk.getTime() < float(task_cfg.online_lr_calibration_prep_s):
                        if "escape" in event.getKeys():
                            raise KeyboardInterrupt
                        win.flip()

                    _draw_empty_board_screen(f"SUSTAIN {cname}", "")
                    blk = _collect_block_simple(float(task_cfg.online_lr_calibration_hold_s))
                    wins = prepare_continuous_windows(
                        raw_block=blk,
                        eeg_cfg=eeg_cfg,
                        sfreq=sfreq,
                        window_s=float(task_cfg.window_s),
                        step_s=float(task_cfg.window_step_s),
                        reject_peak_to_peak=eeg_cfg.reject_peak_to_peak,
                    )
                    if wins.shape[0] > 0:
                        cal_windows_list.append(wins)
                        cal_labels_list.append(np.full(wins.shape[0], int(code), dtype=int))

                    _draw_empty_board_screen("", "Relax")
                    clk2 = core.Clock()
                    while clk2.getTime() < float(task_cfg.online_lr_calibration_iti_s):
                        if "escape" in event.getKeys():
                            raise KeyboardInterrupt
                        win.flip()

                if cal_windows_list:
                    X_on = np.concatenate(cal_windows_list, axis=0)
                    y_on = np.concatenate(cal_labels_list, axis=0)
                    dataset = append_windows_to_dataset(
                        dataset=dataset,
                        windows=X_on,
                        labels=y_on,
                        session_id=online_cal_session_id,
                        n_trials_add=len(cal_codes),
                    )

            loso = evaluate_loso_sessions(dataset, model_cfg, train_only_session_ids=train_only_session_ids)
            classifier = make_mi_classifier(model_cfg)
            classifier.fit(dataset.X, dataset.y)
            counts = Counter(dataset.y.tolist())
            logger.info(
                "Model ready: windows=%d, loso_mean=%.4f, loso_std=%.4f, class_counts=%s",
                dataset.n_windows,
                loso.mean_accuracy,
                loso.std_accuracy,
                counts,
            )

        clf_classes = np.asarray(classifier.named_steps["clf"].classes_, dtype=int)
        class_index = {int(c): i for i, c in enumerate(clf_classes)}
        if int(stim_cfg.left_code) not in class_index or int(stim_cfg.right_code) not in class_index:
            raise RuntimeError(
                f"Classifier classes {clf_classes.tolist()} do not contain expected left/right codes "
                f"{[int(stim_cfg.left_code), int(stim_cfg.right_code)]}."
            )

        with open(f"{fname}_tetris_model.pkl", "wb") as fh:
            pickle.dump(classifier, fh)

        _draw_empty_board_screen(
            f"Model ready  LOSO={loso.mean_accuracy:.3f}±{loso.std_accuracy:.3f}",
            "Press SPACE to start. ESC to quit.",
        )
        while True:
            keys = event.getKeys()
            if "escape" in keys:
                raise KeyboardInterrupt
            if "space" in keys:
                break
            win.flip()

        # Jaw calibration happens immediately after SPACE, before the game appears.
        _draw_empty_board_screen(
            "Jaw calibration",
            f"Keep jaw relaxed for {jaw_calibration_duration_s:.1f}s",
        )
        calib_raw: list[np.ndarray] = []
        last_cal_ts: float | None = None
        cal_clock = core.Clock()
        while cal_clock.getTime() < jaw_calibration_duration_s:
            if "escape" in event.getKeys():
                raise KeyboardInterrupt
            data, ts = stream.get_data(winsize=0.25, picks="all")
            if data.size > 0 and ts is not None and len(ts) > 0:
                ts_arr = np.asarray(ts)
                mask = np.ones_like(ts_arr, bool) if last_cal_ts is None else (ts_arr > float(last_cal_ts))
                if np.any(mask):
                    new_data = np.asarray(data[jaw_ch_idx, mask], dtype=np.float32)
                    new_ts = ts_arr[mask].astype(np.float64)
                    last_cal_ts = float(new_ts[-1])
                    calib_raw.append(new_data)
            remaining = max(0.0, jaw_calibration_duration_s - cal_clock.getTime())
            _draw_empty_board_screen(
                "Jaw calibration",
                f"Keep jaw relaxed for {remaining:0.1f}s",
            )

        if calib_raw:
            try:
                jaw_signal = np.concatenate(calib_raw).astype(np.float32)
                floor = jaw_detector.calibrate(jaw_signal)
                logger.info("Jaw detector calibrated: floor=%.3f", float(floor))
            except Exception:
                logger.warning("Jaw detector calibration failed, continuing with adaptive baseline.")

    except Exception:
        try:
            stream.disconnect()
        except Exception:
            pass
        try:
            win.close()
        except Exception:
            pass
        raise

    # ---------------------------------------------------------------------------
    # Live decoder state
    # ---------------------------------------------------------------------------
    live_filter = StreamingIIRFilter.from_eeg_config(
        eeg_cfg=eeg_cfg,
        sfreq=sfreq,
        n_channels=len(model_ch_names),
    )
    live_buffer = np.empty((len(model_ch_names), 0), dtype=np.float32)
    keep_n = int(round((task_cfg.window_s + task_cfg.filter_context_s) * sfreq))
    window_n = int(round(task_cfg.window_s * sfreq))
    stream_pull_s = max(0.10, task_cfg.live_update_interval_s * 2.0)
    reject_thresh = eeg_cfg.reject_peak_to_peak
    last_live_ts: float | None = None
    jaw_buffer_raw = np.empty(0, dtype=np.float32)
    jaw_buffer_ts = np.empty(0, dtype=np.float64)

    pred_clock = core.Clock()
    prediction_count = 0
    left_prob = 0.5
    right_prob = 0.5
    rest_prob = 0.0
    raw_command = 0.0
    ema_command = 0.0
    live_note = "warming up"
    latest_pred_code: int | None = None
    bias_offset = float(task_cfg.live_bias_offset) if bool(task_cfg.enable_live_bias_offset) else 0.0

    def _pull_stream_and_update_buffers() -> None:
        nonlocal last_live_ts, live_buffer, jaw_buffer_raw, jaw_buffer_ts

        data, ts = stream.get_data(winsize=stream_pull_s, picks="all")
        if data.size == 0 or ts is None or len(ts) == 0:
            return

        ts_arr = np.asarray(ts)
        mask = np.ones_like(ts_arr, bool) if last_live_ts is None else (ts_arr > float(last_live_ts))
        if not np.any(mask):
            return

        x_new = np.asarray(data[:, mask], dtype=np.float32)
        t_new = ts_arr[mask].astype(np.float64)
        last_live_ts = float(t_new[-1])

        x_new_filt = live_filter.process(x_new)
        live_buffer = np.concatenate((live_buffer, x_new_filt), axis=1)
        if live_buffer.shape[1] > keep_n:
            live_buffer = live_buffer[:, -keep_n:]

        jaw_raw_new = np.asarray(x_new[jaw_ch_idx], dtype=np.float32)
        jaw_buffer_raw = np.concatenate((jaw_buffer_raw, jaw_raw_new))
        jaw_buffer_ts = np.concatenate((jaw_buffer_ts, t_new))

        keep = int(max(sfreq, window_n))
        if jaw_buffer_raw.size > keep:
            jaw_buffer_raw = jaw_buffer_raw[-keep:]
            jaw_buffer_ts = jaw_buffer_ts[-keep:]

    def _poll_decoder() -> None:
        nonlocal prediction_count
        nonlocal left_prob, right_prob, rest_prob, raw_command, ema_command, live_note, latest_pred_code

        if pred_clock.getTime() < task_cfg.live_update_interval_s:
            return
        pred_clock.reset()

        if live_buffer.shape[1] < window_n:
            needed_s = max(0.0, (window_n - live_buffer.shape[1]) / sfreq)
            live_note = f"warming up ({needed_s:.1f}s)"
            return

        x_win = live_buffer[:, -window_n:]
        max_ptp = float(np.ptp(x_win, axis=-1).max())
        if reject_thresh is not None and max_ptp > float(reject_thresh):
            live_note = "artifact reject"
            raw_command = 0.0
            ema_command *= 0.85
            return

        p_vec = classifier.predict_proba(x_win[np.newaxis, ...])[0]
        left_prob = float(p_vec[class_index[int(stim_cfg.left_code)]])
        right_prob = float(p_vec[class_index[int(stim_cfg.right_code)]])
        rest_prob = float(p_vec[class_index[int(task_cfg.rest_class_code)]]) if int(task_cfg.rest_class_code) in class_index else 0.0
        raw_command = float(np.clip(right_prob - left_prob + bias_offset, -1.0, 1.0))

        if int(task_cfg.rest_class_code) in class_index and rest_prob >= max(left_prob, right_prob):
            latest_pred_code = int(task_cfg.rest_class_code)
        else:
            latest_pred_code = int(stim_cfg.right_code) if raw_command >= 0.0 else int(stim_cfg.left_code)

        alpha = float(np.clip(task_cfg.command_ema_alpha, 0.0, 1.0))
        ema_command = raw_command if prediction_count == 0 else (1.0 - alpha) * ema_command + alpha * raw_command
        prediction_count += 1
        live_note = "tracking"

    def _poll_jaw_clench() -> bool:
        if jaw_buffer_raw.size < 8 or jaw_buffer_ts.size < 8:
            return False
        _, clenches, _ = jaw_detector.detect(jaw_buffer_raw, jaw_buffer_ts, jaw_thresh_min)
        return len(clenches) > 0

    MOVE_COOLDOWN_S = 0.40
    last_move_t = -999.0

    def _try_move(board: TetrisBoard, now: float) -> bool:
        nonlocal last_move_t
        if now - last_move_t < MOVE_COOLDOWN_S:
            return False
        if latest_pred_code == int(stim_cfg.left_code):
            moved = board.move_left()
        elif latest_pred_code == int(stim_cfg.right_code):
            moved = board.move_right()
        else:
            return False
        if moved:
            last_move_t = now
        return moved

    def _show_game_over(board: TetrisBoard) -> bool:
        go_txt = visual.TextStim(
            win,
            text=(
                f"GAME OVER\n\n"
                f"Score: {board.score}\n"
                f"Lines: {board.lines}  Level: {board.level}\n\n"
                "SPACE to play again   ESC to quit"
            ),
            pos=(0, 0),
            height=0.08,
            color=(0.95, 0.25, 0.25),
            wrapWidth=1.8,
        )
        while True:
            go_txt.draw()
            win.flip()
            keys = event.getKeys()
            if "escape" in keys:
                return False
            if "space" in keys:
                return True

    session_stats: list[dict] = []

    try:
        while True:
            board = TetrisBoard()
            drop_clock = core.Clock()

            live_filter.reset()
            live_buffer = np.empty((len(model_ch_names), 0), np.float32)
            jaw_buffer_raw = np.empty(0, dtype=np.float32)
            jaw_buffer_ts = np.empty(0, dtype=np.float64)
            jaw_detector.reset_runtime_state()
            last_live_ts = None
            pred_clock.reset()
            prediction_count = 0
            left_prob = right_prob = 0.5
            rest_prob = raw_command = ema_command = 0.0
            latest_pred_code = None
            live_note = "warming up"

            logger.info("New Tetris game started.")

            try:
                while not board.game_over:
                    if "escape" in event.getKeys():
                        raise KeyboardInterrupt

                    _pull_stream_and_update_buffers()
                    _poll_decoder()
                    now = core.getTime()

                    if _poll_jaw_clench():
                        board.rotate()
                        logger.info("Jaw clench → rotate  piece=%s", board.current_type)

                    _try_move(board, now)

                    drop_interval = max(0.10, BASE_DROP_INTERVAL_S - LEVEL_SPEEDUP * (board.level - 1))
                    if drop_clock.getTime() >= drop_interval:
                        board.drop_one()
                        drop_clock.reset()

                    txt_score.text = f"Score\n{board.score}"
                    txt_level.text = f"Level\n{board.level}"
                    txt_lines.text = f"Lines\n{board.lines}"
                    parts = [
                        f"{label_cfg.left_name}: {left_prob:.2f}",
                        f"{label_cfg.right_name}: {right_prob:.2f}",
                    ]
                    if int(task_cfg.rest_class_code) in (class_index or {}):
                        parts.append(f"{label_cfg.rest_name}: {rest_prob:.2f}")
                    parts.extend([f"cmd={ema_command:+.2f}", f"bias={bias_offset:+.2f}", live_note])
                    txt_bci.text = "   ".join(parts)
                    txt_cue.text = "LEFT/RIGHT imagery → move  |  Jaw clench → rotate"
                    txt_status.text = f"Predictions: {prediction_count}"

                    _draw_frame(board)

            except KeyboardInterrupt:
                logger.info("Game interrupted by user.")
                session_stats.append({
                    "score": board.score,
                    "lines": board.lines,
                    "level": board.level,
                    "predictions": prediction_count,
                })
                break

            session_stats.append({
                "score": board.score,
                "lines": board.lines,
                "level": board.level,
                "predictions": prediction_count,
            })
            logger.info("Game over: score=%d, lines=%d, level=%d", board.score, board.lines, board.level)

            if not _show_game_over(board):
                break

    finally:
        if session_stats:
            with open(f"{fname}_tetris_results.pkl", "wb") as fh:
                pickle.dump(session_stats, fh)
            logger.info("Saved %d game(s) to %s_tetris_results.pkl", len(session_stats), fname)
        try:
            stream.disconnect()
        except Exception:
            pass
        try:
            win.close()
        except Exception:
            pass


if __name__ == "__main__":
    fname = _prompt_prefix()
    print(f"[SESSION] prefix: {fname}")
    run_task(fname=fname)
