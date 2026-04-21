from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass, replace
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

from derick_ml_jawclench import (
    JAW_CLENCH_CLASS_CODE,
    RAPID_BLINK_CLASS_CODE,
    REST_CLASS_CODE,
    train_face_event_classifier,
    train_jaw_clench_classifier,
)
from mental_command_worker import SharedMIModelResult, train_or_load_shared_mi_model


@dataclass
class FaceCalibrationDataset:
    feature_rows: np.ndarray
    labels: np.ndarray
    channel_names: list[str]
    jaw_channel_indices: list[int]
    sfreq: float
    includes_blink: bool

    def fit_jaw_classifier(self, *, min_total_samples: int = 12) -> tuple[Pipeline, float]:
        y = np.asarray(self.labels, dtype=int)
        x = np.asarray(self.feature_rows, dtype=np.float32)
        keep = np.isin(y, [REST_CLASS_CODE, JAW_CLENCH_CLASS_CODE])
        clf, train_acc, _x_fit, _y_fit = train_jaw_clench_classifier(
            feature_rows=x[keep],
            labels=y[keep],
            min_total_samples=int(min_total_samples),
        )
        return clf, float(train_acc)

    def fit_face_classifier(self, *, min_total_samples: int = 18) -> tuple[Pipeline, float, dict[int, int]]:
        clf, train_acc, _x_fit, _y_fit, class_counts = train_face_event_classifier(
            feature_rows=self.feature_rows,
            labels=self.labels,
            min_total_samples=int(min_total_samples),
        )
        return clf, float(train_acc), class_counts


@dataclass
class TaskSequenceItem:
    task_name: str
    repeats: int = 1
    max_trials: int | None = None
    task_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class BCIOrchestratorRuntime:
    shared_mi_model: SharedMIModelResult | None = None
    face_calibration: FaceCalibrationDataset | None = None
    shared_config_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    task_config_overrides: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    current_task_name: str | None = None


_ACTIVE_RUNTIME: BCIOrchestratorRuntime | None = None


def set_active_runtime(runtime: BCIOrchestratorRuntime | None) -> None:
    global _ACTIVE_RUNTIME
    _ACTIVE_RUNTIME = runtime


def get_active_runtime() -> BCIOrchestratorRuntime | None:
    return _ACTIVE_RUNTIME


def apply_runtime_config_overrides(task_name: str, **named_configs: Any) -> dict[str, Any]:
    runtime = get_active_runtime()
    if runtime is None:
        return named_configs

    shared = runtime.shared_config_overrides
    per_task = runtime.task_config_overrides.get(str(task_name), {})
    updated: dict[str, Any] = {}
    for cfg_name, cfg_value in named_configs.items():
        merged_overrides: dict[str, Any] = {}
        merged_overrides.update(shared.get(cfg_name, {}))
        merged_overrides.update(per_task.get(cfg_name, {}))
        if merged_overrides and is_dataclass(cfg_value):
            valid = {k: v for k, v in merged_overrides.items() if hasattr(cfg_value, k)}
            updated[cfg_name] = replace(cfg_value, **valid) if valid else cfg_value
        else:
            updated[cfg_name] = cfg_value
    return updated


def resolve_shared_mi_model(
    *,
    cache_name: str,
    data_dir: str,
    edf_glob: str,
    calibrate_on_participant: str,
    eeg_cfg: Any,
    task_cfg: Any,
    stim_cfg: Any,
    model_cfg: Any,
    target_sfreq: float,
    target_channel_names: list[str] | tuple[str, ...],
    logger: Any = None,
) -> SharedMIModelResult:
    runtime = get_active_runtime()
    if runtime is not None and runtime.shared_mi_model is not None:
        if logger is not None:
            logger.info("Using orchestrator-provided shared MI model.")
        return runtime.shared_mi_model

    return train_or_load_shared_mi_model(
        cache_name=str(cache_name),
        data_dir=str(data_dir),
        edf_glob=str(edf_glob),
        calibrate_on_participant=str(calibrate_on_participant),
        eeg_cfg=eeg_cfg,
        task_cfg=task_cfg,
        stim_cfg=stim_cfg,
        model_cfg=model_cfg,
        target_sfreq=float(target_sfreq),
        target_channel_names=target_channel_names,
        logger=logger,
    )


def resolve_runtime_jaw_classifier(*, logger: Any = None, min_total_samples: int = 12) -> tuple[Pipeline | None, float | None]:
    runtime = get_active_runtime()
    if runtime is None or runtime.face_calibration is None:
        return None, None
    clf, train_acc = runtime.face_calibration.fit_jaw_classifier(min_total_samples=int(min_total_samples))
    if logger is not None:
        logger.info(
            "Fitted jaw classifier from orchestrator calibration cache: windows=%d train_acc=%.3f",
            int(runtime.face_calibration.labels.size),
            float(train_acc),
        )
    return clf, float(train_acc)


def resolve_runtime_face_classifier(
    *,
    logger: Any = None,
    min_total_samples: int = 18,
) -> tuple[Pipeline | None, float | None, dict[int, int] | None]:
    runtime = get_active_runtime()
    if runtime is None or runtime.face_calibration is None or not runtime.face_calibration.includes_blink:
        return None, None, None
    clf, train_acc, class_counts = runtime.face_calibration.fit_face_classifier(
        min_total_samples=int(min_total_samples)
    )
    if logger is not None:
        logger.info(
            "Fitted face-event classifier from orchestrator calibration cache: counts=%s train_acc=%.3f",
            class_counts,
            float(train_acc),
        )
    return clf, float(train_acc), class_counts
