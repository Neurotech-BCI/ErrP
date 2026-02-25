# bci_worker.py
from __future__ import annotations

import time
import json
import csv
from pathlib import Path

import numpy as np
import zmq

from mne.decoding import CSP
from mne.filter import filter_data, notch_filter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_score

from mne_lsl.stream import StreamLSL, EpochsStream

from config import (
    LSLConfig,
    EEGConfig,
    ModelConfig,
    ZMQConfig,
    SessionConfig,
    CalibrationConfig,
    StimConfig,
)

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


# ----------------------------------------------------------------
# Classifier builders
# ----------------------------------------------------------------


def _make_classifier_riemann_selectC() -> Pipeline:
    """Calibration-only: selects C via internal CV once."""
    cov = Covariances(estimator="oas")
    ts = TangentSpace(metric="riemann")
    clf = LogisticRegressionCV(
        Cs=[0.01, 0.1, 1.0, 10.0],
        cv=3,
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline(
        [
            ("cov", cov),
            ("ts", ts),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", clf),
        ]
    )


def _make_classifier_riemann_fixedC(C: float) -> Pipeline:
    """Online: fixed C (no CV inside fit)."""
    cov = Covariances(estimator="oas")
    ts = TangentSpace(metric="riemann")
    clf = LogisticRegression(
        C=float(C),
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline(
        [
            ("cov", cov),
            ("ts", ts),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", clf),
        ]
    )


def _make_classifier_csp_selectC(n_csp_components: int) -> Pipeline:
    csp = CSP(n_components=n_csp_components, reg="ledoit_wolf", log=True, norm_trace=False)
    clf = LogisticRegressionCV(
        Cs=[0.01, 0.1, 1.0, 10.0],
        cv=3,
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline([("csp", csp), ("scaler", StandardScaler()), ("clf", clf)])


def _make_classifier_csp_fixedC(n_csp_components: int, C: float) -> Pipeline:
    csp = CSP(n_components=n_csp_components, reg="ledoit_wolf", log=True, norm_trace=False)
    clf = LogisticRegression(
        C=float(C),
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline([("csp", csp), ("scaler", StandardScaler()), ("clf", clf)])


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------


def _send_json(sock: zmq.Socket, payload: dict):
    sock.send_string(json.dumps(payload))


def _poll_recv_json(sock: zmq.Socket, poller: zmq.Poller, timeout_ms: int = 0) -> dict | None:
    socks = dict(poller.poll(timeout_ms))
    if sock in socks:
        return json.loads(sock.recv_string())
    return None


def _run_cv(X: np.ndarray, y: np.ndarray, model_cfg: ModelConfig, fixed_C: float | None = None) -> tuple[float, float, np.ndarray]:
    """Stratified k-fold CV. Returns (mean_acc, std_acc, fold_scores)."""
    classes = np.unique(y)
    if not np.all(np.isin(classes, [1, 2])):
        raise ValueError(f"Expected labels 1/2, got classes={classes}")

    n1 = int(np.sum(y == 1))
    n2 = int(np.sum(y == 2))
    n_splits = min(5, n1, n2)
    if n_splits < 2:
        print("[CV] Not enough data per class for cross-validation")
        return 0.0, 0.0, np.array([])

    if model_cfg.use_riemann:
        eval_clf = _make_classifier_riemann_fixedC(fixed_C) if fixed_C is not None else _make_classifier_riemann_selectC()
    else:
        eval_clf = _make_classifier_csp_fixedC(model_cfg.n_csp_components, fixed_C) if fixed_C is not None else _make_classifier_csp_selectC(model_cfg.n_csp_components)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(eval_clf, X, y, cv=cv, scoring="accuracy")
    print(f"[CV] {n_splits}-fold: {[f'{s:.3f}' for s in scores]}  mean={scores.mean():.3f} +/- {scores.std():.3f}")
    return float(scores.mean()), float(scores.std()), scores


def _extract_best_C(pipeline: Pipeline) -> float:
    clf = pipeline.named_steps.get("clf", None)
    if clf is None or not hasattr(clf, "C_"):
        raise ValueError("Could not extract best C: pipeline does not have LogisticRegressionCV at step 'clf'")
    return float(np.atleast_1d(clf.C_)[0])


def _train_initial_classifier(
    X_cal: list[np.ndarray],
    y_cal: list[int],
    model_cfg: ModelConfig,
    left_code: int,
    right_code: int,
) -> tuple[Pipeline, float, float, float, dict[str, int]]:
    if len(y_cal) == 0:
        raise ValueError("No calibration epochs collected")

    X_cal_arr = np.stack(X_cal, axis=0)
    y_cal_arr = np.array(y_cal, dtype=int)

    selector = _make_classifier_riemann_selectC() if model_cfg.use_riemann else _make_classifier_csp_selectC(model_cfg.n_csp_components)
    selector.fit(X_cal_arr, y_cal_arr)
    best_C = _extract_best_C(selector)

    cv_mean, cv_std, _ = _run_cv(X_cal_arr, y_cal_arr, model_cfg, fixed_C=best_C)

    classifier = _make_classifier_riemann_fixedC(best_C) if model_cfg.use_riemann else _make_classifier_csp_fixedC(model_cfg.n_csp_components, best_C)
    classifier.fit(X_cal_arr, y_cal_arr)

    n_per_class = {
        str(int(left_code)): int(np.sum(y_cal_arr == int(left_code))),
        str(int(right_code)): int(np.sum(y_cal_arr == int(right_code))),
    }
    return classifier, best_C, cv_mean, cv_std, n_per_class


class RawCSVRecorder:
    """
    Continuously samples StreamLSL and writes *raw* samples to CSV with:
      t, <EEG channels...>, <TRG>
    It does NOT influence model/epoching; it's purely a tap for offline analysis.

    Implementation details:
    - We sample overlapping windows (winsize) but de-duplicate using timestamps > last_ts.
    - We write incrementally to avoid holding everything in memory.
    """

    def __init__(self, filepath: str, ch_names: list[str], trigger_ch_name: str, winsize_s: float = 0.25):
        self.filepath = filepath
        self.ch_names = ch_names  # includes EEG + trigger (in the stream pick order)
        self.trigger_ch_name = trigger_ch_name
        self.winsize_s = float(winsize_s)

        self._fh = None
        self._writer = None
        self._last_ts = None  # float timestamp of last written sample

    def start(self):
        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.filepath, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        header = ["t"] + self.ch_names
        self._writer.writerow(header)
        self._fh.flush()
        self._last_ts = None
        print(f"[RAW] Recording to {self.filepath}")

    def stop(self):
        if self._fh is not None:
            try:
                self._fh.flush()
            except Exception:
                pass
            try:
                self._fh.close()
            except Exception:
                pass
        self._fh = None
        self._writer = None
        print("[RAW] Recording stopped.")

    def is_active(self) -> bool:
        return self._writer is not None

    def update_from_stream(self, stream: StreamLSL, picks: list[str]):
        if not self.is_active():
            return

        data, ts = stream.get_data(winsize=self.winsize_s, picks=picks)  # data: (n_ch, n_samp)
        if data.size == 0 or ts is None or len(ts) == 0:
            return

        ts = np.asarray(ts)
        if self._last_ts is None:
            mask = np.ones_like(ts, dtype=bool)
        else:
            mask = ts > float(self._last_ts)

        if not np.any(mask):
            return

        idx = np.where(mask)[0]
        # write sample-wise
        for j in idx:
            row = [float(ts[j])] + [float(x) for x in data[:, j]]
            self._writer.writerow(row)

        self._last_ts = float(ts[idx[-1]])

        # periodic flush to reduce loss on crash
        try:
            self._fh.flush()
        except Exception:
            pass


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------


def main():
    lsl = LSLConfig()
    eeg = EEGConfig()
    model_cfg = ModelConfig()
    zmq_cfg = ZMQConfig()
    session_cfg = SessionConfig()
    cal_cfg = CalibrationConfig()
    stim_cfg = StimConfig()

    LEFT = stim_cfg.left_code
    RIGHT = stim_cfg.right_code

    reject_thresh = eeg.reject_peak_to_peak

    # --- ZMQ PAIR socket ---
    ctx = zmq.Context.instance()
    pair = ctx.socket(zmq.PAIR)
    pair.bind(zmq_cfg.pair_addr)
    poller = zmq.Poller()
    poller.register(pair, zmq.POLLIN)

    # --- Connect to EEG LSL stream ---
    stream_bufsize = max(30.0, eeg.tmax + 5.0)

    stream = StreamLSL(
        bufsize=stream_bufsize,
        name=lsl.name,
        stype=lsl.stype,
        source_id=lsl.source_id,
    )
    stream.connect(acquisition_delay=0.001, processing_flags="all")
    print(f"LSL Stream info: {stream.info}")
    print(f"Original channel names: {stream.info['ch_names']}")

    # Pick EEG + trigger channel. Keep order stable: EEG picks then TRG.
    raw_picks = list(eeg.picks) + [lsl.event_channels]
    stream.pick(raw_picks)
    stream.set_channel_types({lsl.event_channels: "stim"})

    sfreq = float(stream.info["sfreq"])
    print(f"Stream connected: sfreq={sfreq:.2f}, bufsize={stream_bufsize}s")

    epochs_online = EpochsStream(
        stream,
        bufsize=30,
        event_id={"left": LEFT, "right": RIGHT},
        event_channels=lsl.event_channels,
        tmin=eeg.tmin,
        tmax=eeg.tmax,
        baseline=eeg.baseline,
        reject=None,
    ).connect(acquisition_delay=0.001)

    classifier: Pipeline | None = None
    best_C: float | None = None

    X_cal: list[np.ndarray] = []
    y_cal: list[int] = []
    X_store: list[np.ndarray] = []
    y_store: list[int] = []
    X_all: list[np.ndarray] = []
    y_all: list[int] = []
    total_accepted = 0
    last_train_at = 0

    phase = "IDLE"  # IDLE -> CALIBRATING -> TRAINING -> TRAINED -> ONLINE -> STOPPED
    n_cal_target = int(cal_cfg.n_calibration_trials)
    train_deadline: float | None = None

    print(f"BCI worker running. Awaiting SESSION_START from psychopy.")

    raw_recorder: RawCSVRecorder | None = None
    raw_csv_path = f"{session_cfg.name}{session_cfg.raw_csv_suffix}"

    # helper: class label -> proba for RIGHT
    def _proba_for_code(proba: np.ndarray, classes: np.ndarray, code: int) -> float:
        idx = int(np.where(classes == int(code))[0][0])
        return float(proba[idx])

    def _ensure_raw_recorder_started():
        nonlocal raw_recorder
        if raw_recorder is not None and raw_recorder.is_active():
            return
        ch_names = list(stream.info["ch_names"])  # should match raw_picks order after stream.pick()
        raw_recorder = RawCSVRecorder(
            filepath=raw_csv_path,
            ch_names=ch_names,
            trigger_ch_name=lsl.event_channels,
            winsize_s=0.25,
        )
        raw_recorder.start()

    def _filter_mi_epochs(X: np.ndarray) -> np.ndarray:
        """
        Apply MI preprocessing to epoch arrays while leaving StreamLSL raw for CSV tap.
        X shape: (n_epochs, n_channels, n_samples)
        """
        Xf = np.asarray(X, dtype=np.float64, order="C")
        if eeg.notch is not None:
            Xf = notch_filter(
                Xf,
                Fs=sfreq,
                freqs=[float(eeg.notch)],
                picks=np.arange(Xf.shape[1]),
                verbose="ERROR",
            )
        Xf = filter_data(
            Xf,
            sfreq=sfreq,
            l_freq=eeg.l_freq,
            h_freq=eeg.h_freq,
            picks=np.arange(Xf.shape[1]),
            verbose="ERROR",
        )
        return Xf.astype(np.float32, copy=False)

    try:
        while True:
            # ---- (A) Check for command from psychopy ----
            cmd = _poll_recv_json(pair, poller, timeout_ms=0)
            if cmd is not None:
                action = cmd.get("action")

                if action == "SESSION_START":
                    _ensure_raw_recorder_started()
                    _send_json(pair, {"status": "ready"})
                    phase = "CALIBRATING"
                    print("[SESSION] SESSION_START received. Raw capture enabled.")

                elif action == "TRAIN":
                    n_cal_target = int(cmd.get("n_trials", cal_cfg.n_calibration_trials))
                    train_deadline = time.time() + 30.0
                    phase = "TRAINING"
                    print(f"[TRAIN] Received. Waiting for {n_cal_target} accepted epochs (have {len(y_cal)})...")

                    # If we already have enough epochs (accumulated during calibration), train immediately
                    if len(y_cal) >= n_cal_target:
                        try:
                            classifier, best_C, cv_mean, cv_std, n_per_class = _train_initial_classifier(
                                X_cal, y_cal, model_cfg, LEFT, RIGHT,
                            )
                            last_train_at = total_accepted
                            phase = "TRAINED"
                            _send_json(pair, {
                                "status": "trained",
                                "cv_mean": cv_mean,
                                "cv_std": cv_std,
                                "n_epochs": len(y_cal),
                                "n_per_class": n_per_class,
                                "best_C": best_C,
                            })
                            print(f"[CAL] Trained on {len(y_cal)} epochs. CV={cv_mean:.3f}+/-{cv_std:.3f}")
                        except Exception as exc:
                            phase = "TRAINED"
                            _send_json(pair, {"status": "error", "message": str(exc)})
                            print(f"[CAL] Training failed: {exc}")

                elif action == "ONLINE_START":
                    if phase != "TRAINED":
                        print("[ONLINE] ONLINE_START ignored: not trained yet.")
                    else:
                        phase = "ONLINE"
                        _ensure_raw_recorder_started()
                        _send_json(pair, {"status": "ack"})
                        print("[ONLINE] Predictions enabled.")

                elif action == "SESSION_STOP":
                    if raw_recorder is not None and raw_recorder.is_active():
                        raw_recorder.stop()
                    phase = "STOPPED"
                    print("[SESSION] SESSION_STOP received.")

            # ---- (B) Raw CSV tap (independent) ----
            if raw_recorder is not None and raw_recorder.is_active():
                raw_recorder.update_from_stream(stream, picks="all")

            # ---- (C) Pull new epochs from EpochsStream ----
            n_new = epochs_online.n_new_epochs
            if n_new > 0:
                X_new = epochs_online.get_data(n_epochs=n_new, picks="eeg")
                X_new = _filter_mi_epochs(X_new)
                codes_new = epochs_online.events[-n_new:]

                for i in range(n_new):
                    code = int(codes_new[i])
                    if code not in (LEFT, RIGHT):
                        continue

                    epoch_data = X_new[i]

                    # Artifact rejection
                    if reject_thresh is not None:
                        ptp = np.ptp(epoch_data, axis=-1).max()
                        if ptp > reject_thresh:
                            if phase == "ONLINE":
                                _send_json(pair, {"y_pred_code": None, "rejected": True})
                            continue

                    # Accumulate epoch
                    X_all.append(epoch_data)
                    y_all.append(code)
                    X_store.append(epoch_data)
                    y_store.append(code)
                    total_accepted += 1

                    if phase in ("CALIBRATING", "TRAINING"):
                        X_cal.append(epoch_data)
                        y_cal.append(code)

                    # Check if training can now proceed
                    if phase == "TRAINING" and len(y_cal) >= n_cal_target:
                        try:
                            classifier, best_C, cv_mean, cv_std, n_per_class = _train_initial_classifier(
                                X_cal, y_cal, model_cfg, LEFT, RIGHT,
                            )
                            last_train_at = total_accepted
                            phase = "TRAINED"
                            _send_json(pair, {
                                "status": "trained",
                                "cv_mean": cv_mean,
                                "cv_std": cv_std,
                                "n_epochs": len(y_cal),
                                "n_per_class": n_per_class,
                                "best_C": best_C,
                            })
                            print(f"[CAL] Trained on {len(y_cal)} epochs. CV={cv_mean:.3f}+/-{cv_std:.3f}")
                        except Exception as exc:
                            phase = "TRAINED"
                            _send_json(pair, {"status": "error", "message": str(exc)})
                            print(f"[CAL] Training failed: {exc}")

                    # Online prediction
                    if phase == "ONLINE" and classifier is not None:
                        x_i = epoch_data[np.newaxis, ...]
                        proba = classifier.predict_proba(x_i)[0]
                        classes = classifier.named_steps["clf"].classes_
                        y_pred_code = int(classes[int(np.argmax(proba))])
                        conf = float(np.max(proba))
                        p_right = _proba_for_code(proba, classes, RIGHT)

                        _send_json(pair, {
                            "y_pred_code": y_pred_code,
                            "y_true_code": code,
                            "conf": conf,
                            "p_right": p_right,
                        })

            # ---- (D) Training timeout ----
            if phase == "TRAINING" and train_deadline is not None and time.time() > train_deadline:
                _send_json(pair, {
                    "status": "error",
                    "message": f"Timeout: only {len(y_cal)}/{n_cal_target} epochs arrived within deadline",
                })
                phase = "TRAINED"
                train_deadline = None
                print(f"[CAL] Training timeout: {len(y_cal)}/{n_cal_target} epochs.")

            # ---- (E) Sliding window ----
            if model_cfg.use_sliding_window and len(y_store) > model_cfg.window_size_epochs:
                X_store = X_store[-model_cfg.window_size_epochs:]
                y_store = y_store[-model_cfg.window_size_epochs:]

            # ---- (F) Online retraining schedule ----
            if phase == "ONLINE" and classifier is not None:
                if (total_accepted - last_train_at) >= model_cfg.retrain_every and len(y_store) >= 4:
                    X_train = np.stack(X_store, axis=0)
                    y_train = np.array(y_store, dtype=int)
                    classifier.fit(X_train, y_train)
                    last_train_at = total_accepted

            # ---- (G) Idle sleep ----
            if n_new == 0 and cmd is None:
                time.sleep(0.002)

    except KeyboardInterrupt:
        print("\nStopping BCI worker...")

    finally:
        # Stop raw recorder if still active
        try:
            if raw_recorder is not None and raw_recorder.is_active():
                raw_recorder.stop()
        except Exception:
            pass

        # Save epoch data + final CV
        if len(y_all) > 0:
            X_save = np.stack(X_all, axis=0)
            y_save = np.array(y_all, dtype=int)
            np.save(f"{session_cfg.name}_data.npy", X_save)
            np.save(f"{session_cfg.name}_labels.npy", y_save)

            cv_mean, cv_std, cv_scores = _run_cv(X_save, y_save, model_cfg, fixed_C=best_C)
            if len(cv_scores) > 0:
                print(f"\nFinal CV (fixed C): {cv_mean:.3f} +/- {cv_std:.3f} | best_C={best_C}")

        for resource in [epochs_online, stream]:
            try:
                if resource is not None:
                    resource.disconnect()
            except Exception:
                pass
        try:
            pair.close(0)
        except Exception:
            pass
        try:
            ctx.term()
        except Exception:
            pass


if __name__ == "__main__":
    main()
