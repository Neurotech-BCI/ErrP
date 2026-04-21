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
from bci_runtime import apply_runtime_config_overrides, resolve_runtime_jaw_classifier, resolve_shared_mi_model
from derick_ml_jawclench import (
    prepare_jaw_calibration_features,
    select_jaw_channel_indices,
    train_jaw_clench_classifier,
    update_live_jaw_clench_state,
)
from mental_command_worker import (
    StreamingIIRFilter,
    canonicalize_channel_name,
    resolve_channel_order,
)


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

def run_task(fname: str, max_trials: int | None = None) -> None:  # noqa: C901
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
    cfgs = apply_runtime_config_overrides(
        "tetris_task",
        lsl_cfg=lsl_cfg,
        stim_cfg=stim_cfg,
        label_cfg=label_cfg,
        task_cfg=task_cfg,
        model_cfg=model_cfg,
        eeg_cfg=eeg_cfg,
    )
    lsl_cfg = cfgs["lsl_cfg"]
    stim_cfg = cfgs["stim_cfg"]
    label_cfg = cfgs["label_cfg"]
    task_cfg = cfgs["task_cfg"]
    model_cfg = cfgs["model_cfg"]
    eeg_cfg = cfgs["eeg_cfg"]

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

    jaw_idxs = select_jaw_channel_indices(model_ch_names)
    jaw_classifier = None
    jaw_prob = 0.0
    jaw_prev_pred = 0
    jaw_event_pending = False
    jaw_prob_thresh = float(task_cfg.jaw_clench_prob_thresh)
    jaw_refractory_s = float(task_cfg.jaw_clench_refractory_s)
    jaw_last_trigger_t = -1e9

    CELL = 0.085
    BOARD_W = BOARD_COLS * CELL
    BOARD_H = BOARD_ROWS * CELL
    BOARD_LEFT = -BOARD_W / 2
    # Shift the board upward slightly to create more room for bottom status text.
    BOARD_TOP = (BOARD_H / 2) + 0.03

    win = visual.Window(
        size=task_cfg.win_size,
        color=(0.18, 0.18, 0.18),
        units="norm",
        fullscr=task_cfg.fullscreen,
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
    txt_bci = visual.TextStim(win, text="", pos=(0, -0.80), height=0.036, color=(0.50, 0.80, 1.00), anchorHoriz="center")
    txt_cue = visual.TextStim(win, text="", pos=(0, 0.90), height=0.048, color=(0.90, 0.90, 0.90), anchorHoriz="center")
    txt_status = visual.TextStim(win, text="", pos=(0, -0.90), height=0.035, color=(0.70, 0.70, 0.70), anchorHoriz="center")

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
    class_index: dict[int, int] | None = None

    _draw_empty_board_screen(
        "Preparing MI model...",
        "Loading offline EDF sessions.",
    )

    try:
        shared_model = resolve_shared_mi_model(
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
            "Model ready from shared model: loso_mean=%.4f, loso_std=%.4f, class_counts=%s, cache=%s",
            loso.mean_accuracy,
            loso.std_accuracy,
            counts,
            shared_model.cache_path,
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
            _draw_empty_board_screen(
                f"Model ready  LOSO={loso.mean_accuracy:.3f}±{loso.std_accuracy:.3f}",
                "Press SPACE to start. ESC to quit.",
            )

        def _run_jaw_calibration() -> None:
            nonlocal jaw_classifier

            runtime_jaw_classifier, runtime_train_acc = resolve_runtime_jaw_classifier(logger=logger, min_total_samples=12)
            if runtime_jaw_classifier is not None:
                jaw_classifier = runtime_jaw_classifier
                logger.info("Using orchestrator-provided jaw calibration (train_acc=%.3f).", float(runtime_train_acc or 0.0))
                return

            n_per_class = int(task_cfg.jaw_calibration_blocks_per_class)
            hold_s = float(task_cfg.jaw_calibration_hold_s)
            prep_s = float(task_cfg.jaw_calibration_prep_s)
            iti_s = float(task_cfg.jaw_calibration_iti_s)
            trim_s = float(task_cfg.jaw_calibration_trim_s)
            jaw_window_n_local = int(round(float(task_cfg.jaw_window_s) * sfreq))
            min_samples = jaw_window_n_local + 2 * int(round(trim_s * sfreq))

            _draw_empty_board_screen(
                "Jaw calibration",
                "We will collect REST and JAW CLENCH trials. Press SPACE to begin.",
            )
            while True:
                keys = event.getKeys()
                if "escape" in keys:
                    raise KeyboardInterrupt
                if "space" in keys:
                    break
                _draw_empty_board_screen(
                    "Jaw calibration",
                    "We will collect REST and JAW CLENCH trials. Press SPACE to begin.",
                )

            labels = [0] * n_per_class + [1] * n_per_class
            random.shuffle(labels)
            calib_blocks: list[np.ndarray] = []
            calib_labels: list[int] = []

            for i, y_label in enumerate(labels, start=1):
                is_clench = bool(y_label == 1)
                trial_name = "JAW CLENCH" if is_clench else "REST"

                clk = core.Clock()
                while clk.getTime() < prep_s:
                    if "escape" in event.getKeys():
                        raise KeyboardInterrupt
                    _draw_empty_board_screen("Prepare", f"Next trial: {trial_name} ({i}/{len(labels)})")

                block_chunks: list[np.ndarray] = []
                last_ts_local: float | None = None
                hold_clock = core.Clock()
                while hold_clock.getTime() < hold_s:
                    if "escape" in event.getKeys():
                        raise KeyboardInterrupt
                    data, ts = stream.get_data(winsize=min(0.20, hold_s), picks="all")
                    if data.size > 0 and ts is not None and len(ts) > 0:
                        ts_arr = np.asarray(ts)
                        mask = np.ones_like(ts_arr, dtype=bool) if last_ts_local is None else (ts_arr > float(last_ts_local))
                        if np.any(mask):
                            block_chunks.append(np.asarray(data[:, mask], dtype=np.float32))
                            last_ts_local = float(ts_arr[mask][-1])
                    _draw_empty_board_screen(
                        trial_name,
                        ("Clench jaw and hold." if is_clench else "Relax face and avoid blinking/movement.")
                        + f"  ({i}/{len(labels)})",
                    )

                block = np.concatenate(block_chunks, axis=1) if block_chunks else np.empty((len(model_ch_names), 0), dtype=np.float32)
                if block.shape[1] >= min_samples:
                    calib_blocks.append(block)
                    calib_labels.append(int(y_label))
                else:
                    logger.warning(
                        "Skipping short jaw calibration block %d: samples=%d, needed=%d",
                        i,
                        int(block.shape[1]),
                        min_samples,
                    )

                iti_clock = core.Clock()
                while iti_clock.getTime() < iti_s:
                    if "escape" in event.getKeys():
                        raise KeyboardInterrupt
                    _draw_empty_board_screen("Relax", "Short break")

            X_np, y_np = prepare_jaw_calibration_features(
                blocks=calib_blocks,
                labels=calib_labels,
                jaw_idxs=jaw_idxs,
                sfreq=float(sfreq),
                window_s=float(task_cfg.jaw_window_s),
                step_s=float(task_cfg.jaw_window_step_s),
                edge_trim_s=trim_s,
            )

            if len(y_np) < 12 or len(set(y_np)) < 2:
                raise RuntimeError(
                    "Jaw calibration failed: not enough usable rest/clench windows. "
                    "Please rerun and reduce movement/blinks during REST trials."
                )

            jaw_classifier, train_acc, _X_np, y_np = train_jaw_clench_classifier(
                feature_rows=X_np,
                labels=y_np,
                min_total_samples=12,
            )
            logger.info(
                "Jaw calibration complete: windows=%d, rest=%d, clench=%d, train_acc=%.3f, jaw_channels=%s, window_s=%.2f, step_s=%.2f, trim_s=%.2f",
                int(len(y_np)),
                int(np.sum(y_np == 0)),
                int(np.sum(y_np == 1)),
                train_acc,
                [model_ch_names[idx] for idx in jaw_idxs],
                float(task_cfg.jaw_window_s),
                float(task_cfg.jaw_window_step_s),
                trim_s,
            )

            _draw_empty_board_screen(
                "Jaw classifier ready",
                f"Train acc {train_acc:.2f}. Jaw clench rotates. Press SPACE to start.",
            )
            while True:
                keys = event.getKeys()
                if "escape" in keys:
                    raise KeyboardInterrupt
                if "space" in keys:
                    return
                _draw_empty_board_screen(
                    "Jaw classifier ready",
                    f"Train acc {train_acc:.2f}. Jaw clench rotates. Press SPACE to start.",
                )

        _run_jaw_calibration()

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
    jaw_window_n = int(round(float(task_cfg.jaw_window_s) * sfreq))
    stream_pull_s = max(0.10, task_cfg.live_update_interval_s * 2.0)
    reject_thresh = eeg_cfg.reject_peak_to_peak
    last_live_ts: float | None = None
    jaw_buffer = np.empty((len(model_ch_names), 0), dtype=np.float32)

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
        nonlocal last_live_ts, live_buffer, jaw_buffer
        nonlocal jaw_prob, jaw_prev_pred, jaw_last_trigger_t, jaw_event_pending

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
            jaw_last_toggle_t=jaw_last_trigger_t,
            jaw_refractory_s=jaw_refractory_s,
            now_t=core.getTime(),
        )
        if should_toggle:
            jaw_last_trigger_t = core.getTime()
            jaw_event_pending = True

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
        nonlocal jaw_event_pending
        if jaw_event_pending:
            jaw_event_pending = False
            return True
        return False

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
            if max_trials is not None and len(session_stats) >= int(max_trials):
                logger.info("Reached max_trials=%d; ending task.", int(max_trials))
                break
            board = TetrisBoard()
            drop_clock = core.Clock()

            live_filter.reset()
            live_buffer = np.empty((len(model_ch_names), 0), np.float32)
            jaw_buffer = np.empty((len(model_ch_names), 0), dtype=np.float32)
            jaw_prob = 0.0
            jaw_prev_pred = 0
            jaw_event_pending = False
            jaw_last_trigger_t = -1e9
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
                    txt_status.text = f"Predictions: {prediction_count}   jaw_p={jaw_prob:.2f}"

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
