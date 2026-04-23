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
from bci_runtime import apply_runtime_config_overrides, resolve_runtime_face_classifier, resolve_shared_mi_model
from derick_ml_jawclench import (
    collect_cue_locked_stream_block,
    JAW_CLENCH_CLASS_CODE,
    RAPID_BLINK_CLASS_CODE,
    REST_CLASS_CODE,
    run_visual_face_event_calibration,
    select_jaw_channel_indices,
    update_live_face_event_state,
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

BASE_DROP_INTERVAL_S = 0.5
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


def _apply_deadband(value: float, deadband: float) -> float:
    value = float(np.clip(value, -1.0, 1.0))
    deadband = float(np.clip(deadband, 0.0, 0.99))
    if abs(value) <= deadband:
        return 0.0
    scaled = (abs(value) - deadband) / (1.0 - deadband)
    return float(np.copysign(scaled, value))


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
        self.pieces_locked = 0

        self._bag: list[str] = []
        self.current_type: str = ""
        self.current_rot: int = 0
        self.current_row: int = 0
        self.current_col: int = 0
        self.next_type: str = ""

        self.next_type = self._next_piece()
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

    def hard_drop(self) -> int:
        steps = 0
        while self._fits(self.current_row + 1, self.current_col, self.current_rot):
            self.current_row += 1
            steps += 1
        self._lock()
        return steps

    def _lock(self) -> None:
        for r, c in self._cells(self.current_row, self.current_col, self.current_rot):
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.grid[r][c] = self.current_type
        self.pieces_locked += 1
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
    if missing:
        raise RuntimeError(
            f"Live stream is missing configured EEG picks {missing}. "
            f"Configured picks: {list(eeg_cfg.picks)}. Available channels: {available}"
        )
    if len(model_ch_names) < 2:
        raise RuntimeError(
            "Need at least 2 configured EEG channels after applying picks. "
            f"Configured picks: {list(eeg_cfg.picks)}. Resolved channels: {model_ch_names}. "
            f"Available channels: {available}"
        )

    stream.pick(model_ch_names)
    sfreq = float(stream.info["sfreq"])
    stream_ch_names = list(stream.info["ch_names"])
    logger.info("Channels: sfreq=%.1f  selected=%s  missing=%s", sfreq, stream_ch_names, missing)

    jaw_idxs = select_jaw_channel_indices(model_ch_names)
    face_classifier = None
    face_class_index: dict[int, int] | None = None
    rest_prob = 0.0
    jaw_prob = 0.0
    blink_prob = 0.0
    face_pred_code = REST_CLASS_CODE
    jaw_prev_pred = 0
    blink_prev_pred = 0
    jaw_event_pending = False
    blink_event_pending = False
    jaw_prob_thresh = float(task_cfg.jaw_clench_prob_thresh)
    blink_prob_thresh = float(getattr(task_cfg, "blink_prob_thresh", 0.70))
    jaw_refractory_s = float(task_cfg.jaw_clench_refractory_s)
    blink_refractory_s = float(getattr(task_cfg, "blink_refractory_s", 0.70))
    jaw_last_trigger_t = -1e9
    blink_last_trigger_t = -1e9

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

        def _draw_calibration_screen() -> None:
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
            txt_score.draw()
            txt_level.draw()
            txt_lines.draw()
            txt_next.draw()
            txt_bci.draw()
            txt_cue.draw()
            txt_status.draw()
            win.flip()

        def _wait_for_space(prompt_text: str) -> None:
            while True:
                keys = event.getKeys()
                if "escape" in keys:
                    raise KeyboardInterrupt
                if "space" in keys:
                    return
                txt_cue.text = prompt_text
                _draw_calibration_screen()

        def _wait_for_seconds(duration_s: float) -> None:
            clock = core.Clock()
            while clock.getTime() < duration_s:
                if "escape" in event.getKeys():
                    raise KeyboardInterrupt
                _draw_calibration_screen()

        def _collect_stream_block(duration_s: float) -> np.ndarray:
            def _check_abort() -> None:
                if "escape" in event.getKeys():
                    raise KeyboardInterrupt

            return collect_cue_locked_stream_block(
                stream=stream,
                sfreq=float(sfreq),
                n_channels=len(model_ch_names),
                duration_s=float(duration_s),
                cue_offset_s=float(task_cfg.special_command_cue_offset_s),
                render_frame=lambda _elapsed_s, _total_s: _draw_calibration_screen(),
                check_abort=_check_abort,
                logger=logger,
                label="tetris face-event calibration block",
            )

        runtime_face_classifier, runtime_train_acc, runtime_counts = resolve_runtime_face_classifier(
            logger=logger,
            min_total_samples=18,
            requested_channel_names=model_ch_names,
        )
        if runtime_face_classifier is not None:
            face_classifier = runtime_face_classifier
            face_train_acc = float(runtime_train_acc or 0.0)
            face_counts = runtime_counts or {}
        else:
            txt_cue.text = "Face-event calibration"
            txt_status.text = "Rapid eye blinks rotate. Jaw clench hard-drops and locks. Press SPACE to begin."
            face_classifier, face_train_acc, _face_y, face_counts = run_visual_face_event_calibration(
                cue=txt_cue,
                info=txt_bci,
                status=txt_status,
                wait_for_space=_wait_for_space,
                wait_for_seconds=_wait_for_seconds,
                collect_stream_block=_collect_stream_block,
                jaw_idxs=jaw_idxs,
                jaw_window_n=int(round(float(task_cfg.jaw_window_s) * sfreq)),
                sfreq=sfreq,
                model_ch_names=model_ch_names,
                logger=logger,
                n_per_class=int(task_cfg.jaw_calibration_blocks_per_class),
                hold_s=float(task_cfg.jaw_calibration_hold_s),
                prep_s=float(task_cfg.jaw_calibration_prep_s),
                iti_s=float(task_cfg.jaw_calibration_iti_s),
                window_s=float(task_cfg.jaw_window_s),
                step_s=float(task_cfg.jaw_window_step_s),
                edge_trim_s=float(task_cfg.jaw_calibration_trim_s),
                min_total_samples=18,
                cue_offset_s=float(task_cfg.special_command_cue_offset_s),
                ready_status_text="Rapid eye blinks rotate. Jaw clench hard-drops and locks. Press SPACE to start.",
            )

        with open(f"{fname}_tetris_face_event_model.pkl", "wb") as fh:
            pickle.dump(face_classifier, fh)

        face_classes = np.asarray(getattr(face_classifier, "classes_", []), dtype=int)
        if face_classes.size == 0 and hasattr(face_classifier, "named_steps") and "clf" in face_classifier.named_steps:
            face_classes = np.asarray(face_classifier.named_steps["clf"].classes_, dtype=int)
        if face_classes.size == 0:
            raise RuntimeError("Face-event classifier has no classes_ after calibration.")
        face_class_index = {int(c): i for i, c in enumerate(face_classes)}
        for needed in (REST_CLASS_CODE, JAW_CLENCH_CLASS_CODE, RAPID_BLINK_CLASS_CODE):
            if int(needed) not in face_class_index:
                raise RuntimeError(
                    "Face-event classifier is missing required classes. "
                    f"Found {face_classes.tolist()}, required {[REST_CLASS_CODE, JAW_CLENCH_CLASS_CODE, RAPID_BLINK_CLASS_CODE]}"
                )

        logger.info(
            "Face-event classifier ready: train_acc=%.3f, counts=%s",
            float(face_train_acc),
            face_counts,
        )
        _draw_empty_board_screen(
            "Ready",
            "LEFT/RIGHT imagery moves. Eye blink rotates. Jaw clench hard-drops and locks. Press SPACE to start.",
        )
        while True:
            keys = event.getKeys()
            if "escape" in keys:
                raise KeyboardInterrupt
            if "space" in keys:
                break
            _draw_empty_board_screen(
                "Ready",
                "LEFT/RIGHT imagery moves. Eye blink rotates. Jaw clench hard-drops and locks. Press SPACE to start.",
            )

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
    face_buffer = np.empty((len(model_ch_names), 0), dtype=np.float32)

    pred_clock = core.Clock()
    prediction_count = 0
    left_prob = 0.5
    right_prob = 0.5
    rest_prob = 0.0
    raw_command = 0.0
    ema_command = 0.0
    live_note = "warming up"
    latest_pred_code: int | None = None
    discrete_command = 0.0
    bias_offset = float(task_cfg.live_bias_offset) if bool(task_cfg.enable_live_bias_offset) else 0.0
    use_discrete_command_ema = bool(getattr(task_cfg, "enable_discrete_command_ema", False))

    def _pull_stream_and_update_buffers() -> None:
        nonlocal last_live_ts, live_buffer, face_buffer
        nonlocal rest_prob, jaw_prob, blink_prob, face_pred_code
        nonlocal jaw_prev_pred, blink_prev_pred
        nonlocal jaw_last_trigger_t, blink_last_trigger_t
        nonlocal jaw_event_pending, blink_event_pending

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

        face_buffer, rest_prob, jaw_prob, blink_prob, face_pred_code, jaw_prev_pred, blink_prev_pred, should_drop, should_rotate = update_live_face_event_state(
            face_buffer=face_buffer,
            x_new=x_new,
            keep_n=keep_n,
            jaw_window_n=jaw_window_n,
            face_classifier=face_classifier,
            class_index=face_class_index,
            jaw_idxs=jaw_idxs,
            rest_prob=rest_prob,
            jaw_prob=jaw_prob,
            blink_prob=blink_prob,
            jaw_prev_pred=jaw_prev_pred,
            blink_prev_pred=blink_prev_pred,
            jaw_prob_thresh=jaw_prob_thresh,
            blink_prob_thresh=blink_prob_thresh,
            jaw_last_event_t=jaw_last_trigger_t,
            blink_last_event_t=blink_last_trigger_t,
            jaw_refractory_s=jaw_refractory_s,
            blink_refractory_s=blink_refractory_s,
            now_t=core.getTime(),
        )
        if should_drop:
            jaw_last_trigger_t = core.getTime()
            jaw_event_pending = True
        if should_rotate:
            blink_last_trigger_t = core.getTime()
            blink_event_pending = True

    def _poll_decoder() -> None:
        nonlocal prediction_count
        nonlocal left_prob, right_prob, rest_prob, raw_command, ema_command, live_note, latest_pred_code, discrete_command

        if pred_clock.getTime() < task_cfg.live_update_interval_s:
            return
        pred_clock.reset()

        if live_buffer.shape[1] < window_n:
            needed_s = max(0.0, (window_n - live_buffer.shape[1]) / sfreq)
            live_note = f"warming up ({needed_s:.1f}s)"
            discrete_command = 0.0
            return

        x_win = live_buffer[:, -window_n:]
        max_ptp = float(np.ptp(x_win, axis=-1).max())
        if reject_thresh is not None and max_ptp > float(reject_thresh):
            live_note = "artifact reject"
            raw_command = 0.0
            ema_command *= 0.85
            discrete_command = 0.0
            return

        p_vec = classifier.predict_proba(x_win[np.newaxis, ...])[0]
        left_prob = float(p_vec[class_index[int(stim_cfg.left_code)]])
        right_prob = float(p_vec[class_index[int(stim_cfg.right_code)]])
        rest_prob = float(p_vec[class_index[int(task_cfg.rest_class_code)]]) if int(task_cfg.rest_class_code) in class_index else 0.0
        raw_command = float(np.clip(right_prob - left_prob + bias_offset, -1.0, 1.0))

        if int(task_cfg.rest_class_code) in class_index and rest_prob >= max(left_prob, right_prob):
            latest_pred_code = int(task_cfg.rest_class_code)
            discrete_command = 0.0
        else:
            latest_pred_code = int(stim_cfg.right_code) if raw_command >= 0.0 else int(stim_cfg.left_code)
            discrete_command = 1.0 if latest_pred_code == int(stim_cfg.right_code) else -1.0

        alpha = float(np.clip(task_cfg.command_ema_alpha, 0.0, 1.0))
        ema_target = discrete_command if use_discrete_command_ema else raw_command
        ema_command = ema_target if prediction_count == 0 else (1.0 - alpha) * ema_command + alpha * ema_target
        prediction_count += 1
        live_note = "tracking"

    def _poll_jaw_clench() -> bool:
        nonlocal jaw_event_pending
        if jaw_event_pending:
            jaw_event_pending = False
            return True
        return False

    def _poll_blink() -> bool:
        nonlocal blink_event_pending
        if blink_event_pending:
            blink_event_pending = False
            return True
        return False

    MOVE_COOLDOWN_S = 0.40
    last_move_t = -999.0

    def _try_move(board: TetrisBoard, now: float) -> bool:
        nonlocal last_move_t
        if now - last_move_t < MOVE_COOLDOWN_S:
            return False
        move_pred_code = latest_pred_code
        if use_discrete_command_ema:
            command_drive = _apply_deadband(ema_command, float(task_cfg.command_deadband))
            if command_drive > 0.0:
                move_pred_code = int(stim_cfg.right_code)
            elif command_drive < 0.0:
                move_pred_code = int(stim_cfg.left_code)
            else:
                move_pred_code = None
        if move_pred_code == int(stim_cfg.left_code):
            moved = board.move_left()
        elif move_pred_code == int(stim_cfg.right_code):
            moved = board.move_right()
        else:
            return False
        if moved:
            last_move_t = now
        return moved

    def _show_game_over(board: TetrisBoard, *, piece_limit_reached: bool = False) -> None:
        body = (
            f"PIECE LIMIT REACHED\n\n"
            f"Pieces placed: {board.pieces_locked}"
        ) if piece_limit_reached else (
            f"GAME OVER\n\n"
            f"Pieces placed: {board.pieces_locked}"
        )
        go_txt = visual.TextStim(
            win,
            text=(body + "\n\nPress ESC to exit task."),
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
                return

    max_pieces = int(max_trials) if max_trials is not None else None
    total_pieces_locked = 0
    session_stats: list[dict] = []

    try:
        while True:
            if max_pieces is not None and total_pieces_locked >= max_pieces:
                logger.info("Reached max_trials=%d piece limit; ending task.", int(max_pieces))
                break
            board = TetrisBoard()
            drop_clock = core.Clock()
            piece_limit_reached = False
            game_start_pieces = total_pieces_locked

            live_filter.reset()
            live_buffer = np.empty((len(model_ch_names), 0), np.float32)
            face_buffer = np.empty((len(model_ch_names), 0), dtype=np.float32)
            rest_prob = 0.0
            jaw_prob = 0.0
            blink_prob = 0.0
            face_pred_code = REST_CLASS_CODE
            jaw_prev_pred = 0
            blink_prev_pred = 0
            jaw_event_pending = False
            blink_event_pending = False
            jaw_last_trigger_t = -1e9
            blink_last_trigger_t = -1e9
            last_live_ts = None
            pred_clock.reset()
            prediction_count = 0
            left_prob = right_prob = 0.5
            raw_command = ema_command = 0.0
            latest_pred_code = None
            discrete_command = 0.0
            live_note = "warming up"

            logger.info("New Tetris game started.")

            try:
                while not board.game_over:
                    if "escape" in event.getKeys():
                        raise KeyboardInterrupt

                    _pull_stream_and_update_buffers()
                    _poll_decoder()
                    now = core.getTime()

                    if _poll_blink():
                        board.rotate()
                        logger.info("Eye blink → rotate  piece=%s blink_p=%.3f", board.current_type, blink_prob)

                    if _poll_jaw_clench():
                        piece_type = board.current_type
                        drop_steps = board.hard_drop()
                        logger.info("Jaw clench → hard drop  piece=%s steps=%d jaw_p=%.3f", piece_type, drop_steps, jaw_prob)

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
                    parts.extend([f"jaw={jaw_prob:.2f}", f"blink={blink_prob:.2f}", live_note])
                    txt_bci.text = "   ".join(parts)
                    txt_cue.text = "LEFT/RIGHT imagery → move  |  Eye blink → rotate  |  Jaw clench → hard drop"
                    session_pieces_locked = game_start_pieces + board.pieces_locked
                    if max_pieces is None:
                        txt_status.text = f"Pieces: {session_pieces_locked}"
                    else:
                        txt_status.text = f"Pieces: {session_pieces_locked}/{max_pieces}"

                    _draw_frame(board)

                    if max_pieces is not None and session_pieces_locked >= max_pieces:
                        piece_limit_reached = True
                        logger.info(
                            "Reached piece limit: session_pieces_locked=%d max_trials=%d",
                            int(session_pieces_locked),
                            int(max_pieces),
                        )
                        break

            except KeyboardInterrupt:
                logger.info("Game interrupted by user.")
                session_stats.append({
                    "score": board.score,
                    "lines": board.lines,
                    "level": board.level,
                    "pieces_locked": board.pieces_locked,
                    "session_pieces_locked": game_start_pieces + board.pieces_locked,
                    "predictions": prediction_count,
                })
                break

            total_pieces_locked = game_start_pieces + board.pieces_locked
            session_stats.append({
                "score": board.score,
                "lines": board.lines,
                "level": board.level,
                "pieces_locked": board.pieces_locked,
                "session_pieces_locked": total_pieces_locked,
                "predictions": prediction_count,
            })
            logger.info(
                "%s: score=%d, lines=%d, level=%d, pieces_locked=%d, session_pieces_locked=%d",
                "Piece limit reached" if piece_limit_reached else "Game over",
                board.score,
                board.lines,
                board.level,
                board.pieces_locked,
                total_pieces_locked,
            )

            _show_game_over(board, piece_limit_reached=piece_limit_reached)
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
