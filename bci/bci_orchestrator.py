from __future__ import annotations

import argparse
import importlib
import inspect
from dataclasses import replace
from datetime import datetime

import numpy as np
from mne_lsl.stream import StreamLSL
from psychopy import core, event, visual

from bci_runtime import BCIOrchestratorRuntime, FaceCalibrationDataset, TaskSequenceItem, set_active_runtime
from config import (
    BCIOrchestratorConfig,
    EEGConfig,
    HingeTaskConfig,
    LSLConfig,
    MentalCommandModelConfig,
    MICursorTaskConfig,
    StimConfig,
)
from derick_ml_jawclench import (
    collect_cue_locked_stream_block,
    collect_visual_face_event_feature_rows,
    select_jaw_channel_indices,
)
from mental_command_worker import canonicalize_channel_name, resolve_channel_order, train_or_load_shared_mi_model
from orchestrator_config import DEFAULT_BCI_ORCHESTRATOR_CONFIG


TASK_MODULES = {
    "hinge_task": "hinge_task",
    "jaw_pause_cursor_task": "jaw_pause_cursor_task",
    "knob_task": "knob_task",
    "lr_cursor_task": "lr_cursor_task",
    "mental_command_task": "mental_command_task",
    "mi_cursor_task": "mi_cursor_task",
    "mi_keyboard_task": "mi_keyboard_task",
    "real_cursor": "real_cursor",
    "subway_runner_task": "subway_runner_task",
    "tetris_task": "tetris_task",
}

JAW_TASKS = {
    "hinge_task",
    "jaw_pause_cursor_task",
    "mi_cursor_task",
    "mi_keyboard_task",
    "real_cursor",
    "subway_runner_task",
    "tetris_task",
}
FACE_TASKS = {"mi_keyboard_task", "real_cursor", "tetris_task"}


def _load_config(config_module: str, config_name: str) -> BCIOrchestratorConfig:
    mod = importlib.import_module(config_module)
    cfg = getattr(mod, config_name)
    if not isinstance(cfg, BCIOrchestratorConfig):
        raise TypeError(f"{config_module}.{config_name} is not a BCIOrchestratorConfig")
    return cfg


def _task_requires_jaw(task_name: str) -> bool:
    return str(task_name) in JAW_TASKS


def _task_requires_face(task_name: str) -> bool:
    return str(task_name) in FACE_TASKS


def _hinge_requires_face(runtime: BCIOrchestratorRuntime) -> bool:
    cfgs = _apply_overrides(
        runtime,
        "hinge_task",
        task_cfg=HingeTaskConfig(),
    )
    return bool(cfgs["task_cfg"].enable_keyboard_on_match)


def _build_task_prefix(participant: str, task_name: str, repeat_idx: int, repeats: int) -> str:
    base = f"{datetime.now().strftime('%m_%d_%y')}_{participant}_{task_name}"
    if repeats > 1:
        return f"{base}_r{repeat_idx + 1}"
    return base


def _apply_overrides(runtime: BCIOrchestratorRuntime, task_name: str, **named_configs):
    shared = runtime.shared_config_overrides
    per_task = runtime.task_config_overrides.get(task_name, {})
    out = {}
    for cfg_name, cfg_value in named_configs.items():
        overrides = {}
        overrides.update(shared.get(cfg_name, {}))
        overrides.update(per_task.get(cfg_name, {}))
        valid = {k: v for k, v in overrides.items() if hasattr(cfg_value, k)}
        out[cfg_name] = replace(cfg_value, **valid) if valid else cfg_value
    return out


def _prepare_stream_signature(
    runtime: BCIOrchestratorRuntime,
    *,
    task_name_for_overrides: str = "orchestrator",
):
    base_cfgs = _apply_overrides(
        runtime,
        str(task_name_for_overrides),
        lsl_cfg=LSLConfig(),
        eeg_cfg=EEGConfig(
            l_freq=8.0,
            h_freq=30.0,
            reject_peak_to_peak=150.0,
        ),
    )
    lsl_cfg = base_cfgs["lsl_cfg"]
    eeg_cfg = base_cfgs["eeg_cfg"]

    stream = StreamLSL(
        bufsize=60.0,
        name=lsl_cfg.name,
        stype=lsl_cfg.stype,
        source_id=lsl_cfg.source_id,
    )
    stream.connect(acquisition_delay=0.001, processing_flags="all")
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
    return stream, sfreq, model_ch_names, eeg_cfg


def _collect_special_calibration(
    runtime: BCIOrchestratorRuntime,
    *,
    include_blinks: bool,
    task_name_for_overrides: str = "orchestrator",
) -> FaceCalibrationDataset:
    stream, sfreq, model_ch_names, _eeg_cfg = _prepare_stream_signature(
        runtime,
        task_name_for_overrides=str(task_name_for_overrides),
    )
    jaw_idxs = select_jaw_channel_indices(model_ch_names)

    win = visual.Window(size=(1200, 700), color=(-0.08, -0.08, -0.08), units="norm", fullscr=False)
    cue = visual.TextStim(win, text="", pos=(0, 0.22), height=0.065, color=(0.92, 0.92, 0.92))
    info = visual.TextStim(win, text="", pos=(0, 0.02), height=0.05, color=(0.84, 0.90, 0.96))
    status = visual.TextStim(win, text="", pos=(0, -0.18), height=0.045, color=(0.84, 0.84, 0.84))

    def _draw():
        cue.draw()
        info.draw()
        status.draw()
        win.flip()

    def _wait_for_space(prompt_text: str) -> None:
        cue.text = prompt_text
        while True:
            keys = event.getKeys()
            if "escape" in keys:
                raise KeyboardInterrupt
            if "space" in keys:
                return
            _draw()

    def _wait_for_seconds(duration_s: float) -> None:
        clk = core.Clock()
        while clk.getTime() < duration_s:
            if "escape" in event.getKeys():
                raise KeyboardInterrupt
            _draw()

    def _collect_stream_block(duration_s: float) -> np.ndarray:
        def _check_abort() -> None:
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

        return collect_cue_locked_stream_block(
            stream=stream,
            sfreq=float(sfreq),
            n_channels=len(model_ch_names),
            duration_s=float(duration_s),
            cue_offset_s=float(base_task_cfg.special_command_cue_offset_s),
            render_frame=lambda _elapsed_s, _total_s: _draw(),
            check_abort=_check_abort,
            logger=importlib.import_module("logging").getLogger("bci_orchestrator.calibration"),
            label="orchestrator special-command calibration block",
        )

    shared_cfg = runtime.shared_config_overrides.get("task_cfg", {})
    base_task_cfg = replace(
        MICursorTaskConfig(),
        **{k: v for k, v in shared_cfg.items() if hasattr(MICursorTaskConfig(), k)},
    )

    try:
        X_cal, y_cal, _class_counts = collect_visual_face_event_feature_rows(
            cue=cue,
            info=info,
            status=status,
            wait_for_space=_wait_for_space,
            wait_for_seconds=_wait_for_seconds,
            collect_stream_block=_collect_stream_block,
            jaw_idxs=jaw_idxs,
            jaw_window_n=int(round(float(base_task_cfg.jaw_window_s) * sfreq)),
            sfreq=sfreq,
            logger=importlib.import_module("logging").getLogger("bci_orchestrator.calibration"),
            n_per_class=int(base_task_cfg.jaw_calibration_blocks_per_class),
            hold_s=float(base_task_cfg.jaw_calibration_hold_s),
            prep_s=float(base_task_cfg.jaw_calibration_prep_s),
            iti_s=float(base_task_cfg.jaw_calibration_iti_s),
            window_s=float(base_task_cfg.jaw_window_s),
            step_s=float(base_task_cfg.jaw_window_step_s),
            edge_trim_s=float(base_task_cfg.jaw_calibration_trim_s),
            include_blinks=bool(include_blinks),
            min_total_samples=18 if include_blinks else 12,
            cue_offset_s=float(base_task_cfg.special_command_cue_offset_s),
        )
    finally:
        try:
            stream.disconnect()
        except Exception:
            pass
        try:
            win.close()
        except Exception:
            pass

    return FaceCalibrationDataset(
        feature_rows=X_cal,
        labels=y_cal,
        channel_names=list(model_ch_names),
        jaw_channel_indices=list(jaw_idxs),
        sfreq=float(sfreq),
        includes_blink=bool(include_blinks),
    )


def _prepare_shared_mi_model(runtime: BCIOrchestratorRuntime) -> None:
    stream, sfreq, model_ch_names, eeg_cfg = _prepare_stream_signature(
        runtime,
        task_name_for_overrides="orchestrator",
    )
    try:
        cfgs = _apply_overrides(
            runtime,
            "orchestrator",
            stim_cfg=StimConfig(),
            model_cfg=MentalCommandModelConfig(),
            task_cfg=MICursorTaskConfig(),
            eeg_cfg=eeg_cfg,
        )
        stim_cfg = cfgs["stim_cfg"]
        model_cfg = cfgs["model_cfg"]
        task_cfg = cfgs["task_cfg"]
        eeg_cfg = cfgs["eeg_cfg"]
        runtime.shared_mi_model = train_or_load_shared_mi_model(
            cache_name=runtime.shared_config_overrides.get("shared", {}).get("mi_cache_name", "mi_shared_lr_model"),
            data_dir=task_cfg.data_dir,
            edf_glob=task_cfg.edf_glob,
            calibrate_on_participant=task_cfg.calirate_on_participant,
            eeg_cfg=eeg_cfg,
            task_cfg=task_cfg,
            stim_cfg=stim_cfg,
            model_cfg=model_cfg,
            target_sfreq=float(sfreq),
            target_channel_names=model_ch_names,
        )
    finally:
        try:
            stream.disconnect()
        except Exception:
            pass


def _prepare_shared_epoch_mi_model(runtime: BCIOrchestratorRuntime) -> None:
    stream, sfreq, model_ch_names, eeg_cfg = _prepare_stream_signature(
        runtime,
        task_name_for_overrides="hinge_task",
    )
    try:
        cfgs = _apply_overrides(
            runtime,
            "hinge_task",
            stim_cfg=StimConfig(),
            model_cfg=MentalCommandModelConfig(),
            task_cfg=HingeTaskConfig(),
            eeg_cfg=eeg_cfg,
        )
        stim_cfg = cfgs["stim_cfg"]
        model_cfg = cfgs["model_cfg"]
        task_cfg = cfgs["task_cfg"]
        eeg_cfg = cfgs["eeg_cfg"]
        runtime.shared_epoch_mi_model = train_or_load_shared_mi_model(
            cache_name=runtime.shared_config_overrides.get("shared", {}).get("epoch_mi_cache_name", "mi_shared_epoch_model"),
            data_dir=task_cfg.data_dir,
            edf_glob=task_cfg.edf_glob,
            calibrate_on_participant=task_cfg.calirate_on_participant,
            eeg_cfg=eeg_cfg,
            task_cfg=task_cfg,
            stim_cfg=stim_cfg,
            model_cfg=model_cfg,
            target_sfreq=float(sfreq),
            target_channel_names=model_ch_names,
        )
    finally:
        try:
            stream.disconnect()
        except Exception:
            pass


def _run_task(item: TaskSequenceItem, participant_name: str) -> None:
    module_name = TASK_MODULES[str(item.task_name)]
    mod = importlib.import_module(module_name)
    task_fn = getattr(mod, "run_task")
    sig = inspect.signature(task_fn)

    for repeat_idx in range(int(item.repeats)):
        fname = _build_task_prefix(participant_name, str(item.task_name), repeat_idx, int(item.repeats))
        kwargs = dict(item.task_kwargs)
        if item.max_trials is not None:
            kwargs.setdefault("max_trials", int(item.max_trials))
        call_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        task_fn(fname=fname, **call_kwargs)


def run_orchestrated_session(cfg: BCIOrchestratorConfig) -> None:
    runtime = BCIOrchestratorRuntime(
        shared_config_overrides={
            **dict(cfg.shared_config_overrides),
            "shared": {
                "mi_cache_name": cfg.shared.mi_cache_name,
                "epoch_mi_cache_name": cfg.shared.epoch_mi_cache_name,
            },
        },
        task_config_overrides=dict(cfg.task_config_overrides),
    )
    set_active_runtime(runtime)
    try:
        _prepare_shared_mi_model(runtime)
        task_names = [item.task_name for item in cfg.task_sequence]
        if "hinge_task" in task_names:
            _prepare_shared_epoch_mi_model(runtime)
        needs_blinks = any(_task_requires_face(name) for name in task_names)
        if "hinge_task" in task_names and _hinge_requires_face(runtime):
            needs_blinks = True
        if any(_task_requires_jaw(name) for name in task_names):
            jaw_calibration_task_name = next(
                (str(name) for name in task_names if _task_requires_jaw(name)),
                "orchestrator",
            )
            runtime.face_calibration = _collect_special_calibration(
                runtime,
                include_blinks=bool(needs_blinks),
                task_name_for_overrides=str(jaw_calibration_task_name),
            )

        for item_cfg in cfg.task_sequence:
            item = TaskSequenceItem(
                task_name=item_cfg.task_name,
                repeats=item_cfg.repeats,
                max_trials=item_cfg.max_trials,
                task_kwargs=dict(item_cfg.task_kwargs),
            )
            runtime.current_task_name = item.task_name
            _run_task(item, cfg.shared.participant_name)
    finally:
        runtime.current_task_name = None
        set_active_runtime(None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a configured sequence of BCI tasks from one entry point.")
    parser.add_argument(
        "--config-module",
        default="orchestrator_config",
        help="Python module containing the orchestrator config object.",
    )
    parser.add_argument(
        "--config-name",
        default="DEFAULT_BCI_ORCHESTRATOR_CONFIG",
        help="Name of the BCIOrchestratorConfig object inside the config module.",
    )
    args = parser.parse_args()
    cfg = _load_config(args.config_module, args.config_name)
    run_orchestrated_session(cfg)


if __name__ == "__main__":
    main()
