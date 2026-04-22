# BCI Orchestration

This folder includes a single orchestration entry point for running a sequence of BCI tasks with shared MI loading and shared special-command calibration.

## Run It

From the repo root:

```bash
python bci/bci_orchestrator.py
```

## Set The Task Sequence

Edit `task_sequence` in [orchestrator_config.py](ErrP/bci/orchestrator_config.py).

Example:

```python
task_sequence=(
    OrchestratedTaskSpec(task_name="lr_cursor_task", repeats=1, max_trials=10),
    OrchestratedTaskSpec(task_name="tetris_task", repeats=1, max_trials=3),
    OrchestratedTaskSpec(task_name="mi_cursor_task", repeats=1, max_trials=5),
)
```

Supported task names:

- `hinge_task`
- `jaw_pause_cursor_task`
- `knob_task`
- `lr_cursor_task`
- `mental_command_task`
- `mi_cursor_task`
- `mi_keyboard_task`
- `real_cursor`
- `subway_runner_task`
- `tetris_task`

Notes:

- For trial/game tasks, `max_trials` limits how many trials or games run before advancing.
- For `mental_command_task`, `mi_keyboard_task`, and `real_cursor`, the task still ends on `ESC`.
- Pressing `ESC` always ends the current task early and moves to the next task.

## Update Parameters

There are two override layers in [orchestrator_config.py](ErrP/bci/orchestrator_config.py):

### `shared_config_overrides`

Use this for settings you want applied across tasks.

Typical example:

```python
shared_config_overrides={
    "task_cfg": {
        "data_dir": "/path/to/edf/root",
        "edf_glob": "*.edf",
        "calirate_on_participant": "andy",
        "live_bias_offset": 0.2,
    },
}
```

### `task_config_overrides`

Use this for one specific task.

Example:

```python
task_config_overrides={
    "mi_cursor_task": {
        "task_cfg": {
            "forward_speed_norm_s": 0.22,
            "enable_jaw_clench_pause": True,
        },
    },
}
```

The override keys should match the config object names used by tasks, most commonly:

- `task_cfg`
- `eeg_cfg`
- `model_cfg`
- `lsl_cfg`
- `stim_cfg`

## Shared Calibration Behavior

The orchestrator prepares shared resources once at startup:

- MI classifier:
  - loaded from cache if available
  - otherwise trained from offline EDF sessions
- Full-epoch MI classifier:
  - runs only if `hinge_task` is in the sequence
  - loaded from cache if available
  - otherwise trained from offline EDF sessions using one full 3 second epoch per trial
- Jaw calibration:
  - runs only if at least one jaw-using task is in the sequence
- Jaw + blink calibration:
  - runs only if `real_cursor` is in the sequence

The special-command calibration data are collected once, then each task fits the classifier it needs from that shared calibration set.

## Output Naming

`participant_name` in [orchestrator_config.py](ErrP/bci/orchestrator_config.py) is used to build the file prefix for logs and saved outputs.
