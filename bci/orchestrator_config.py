from __future__ import annotations

from config import BCIOrchestratorConfig, OrchestratedTaskSpec, SharedBCIConfig


DEFAULT_BCI_ORCHESTRATOR_CONFIG = BCIOrchestratorConfig(
    shared=SharedBCIConfig(
        participant_name="participant",
    ),
    shared_config_overrides={
        # Example:
        # "task_cfg": {
        #     "data_dir": "/path/to/edf/root",
        #     "edf_glob": "*.edf",
        #     "calirate_on_participant": "andy",
        #     "live_bias_offset": 0.2,
        # },
    },
    task_config_overrides={
        # Example:
        # "mi_cursor_task": {
        #     "task_cfg": {
        #         "enable_jaw_clench_pause": True,
        #         "forward_speed_norm_s": 0.22,
        #     },
        # },
    },
    task_sequence=(
        OrchestratedTaskSpec(task_name="lr_cursor_task", repeats=1, max_trials=10),
        OrchestratedTaskSpec(task_name="tetris_task", repeats=1, max_trials=8),
        OrchestratedTaskSpec(task_name="hinge_task", repeats=1, max_trials=1),
        OrchestratedTaskSpec(task_name="mi_cursor_task", repeats=1, max_trials=3),
        
    ),
)
