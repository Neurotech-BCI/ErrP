# bci_worker.py
from __future__ import annotations

import time
import json
import csv
from collections import deque
from dataclasses import dataclass
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


def _encode_msg(topic: str, payload: dict) -> bytes:
    return f"{topic} {json.dumps(payload)}".encode("utf-8")


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


@dataclass
class PendingTrial:
    trial_id: int
    code: int  # 1/2
    t_task: float | None = None
    t_received: float | None = None


@dataclass
class UnmatchedEpoch:
    X: np.ndarray  # (n_channels, n_samples)
    code: int  # 1/2
    t_epoch: float


def _match_trials_to_epochs(
    pending: deque[PendingTrial],
    unmatched: deque[UnmatchedEpoch],
) -> list[tuple[PendingTrial, UnmatchedEpoch]]:
    """
    Match pending TRIAL_START handshakes to epochs by code (1/2).
    Preserve ordering without assuming perfect timing or delivery ordering.
    """
    matches: list[tuple[PendingTrial, UnmatchedEpoch]] = []
    if not pending or not unmatched:
        return matches

    new_pending = deque()
    while pending:
        pt = pending.popleft()
        found_idx = None
        for i, ue in enumerate(unmatched):
            if int(ue.code) == int(pt.code):
                found_idx = i
                break
        if found_idx is None:
            new_pending.append(pt)
            continue
        ue = unmatched[found_idx]
        del unmatched[found_idx]
        matches.append((pt, ue))

    pending.extend(new_pending)
    return matches


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

    # --- ZMQ sockets ---
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.bind(zmq_cfg.pub_addr)

    ctrl = ctx.socket(zmq.PULL)
    ctrl.bind(zmq_cfg.ctrl_addr)

    ctrl_poller = zmq.Poller()
    ctrl_poller.register(ctrl, zmq.POLLIN)
    time.sleep(0.3)

    # --- Connect to EEG LSL stream ---
    stream_bufsize = max(30.0, eeg.tmax + 5.0)

    stream = StreamLSL(
        bufsize=stream_bufsize,
        name=lsl.name,
        stype=lsl.stype,
        source_id=lsl.source_id,
    )
    stream.connect(acquisition_delay=0.001, processing_flags="all")
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
    calibration_trials_target = int(cal_cfg.n_calibration_trials)
    calibration_trials_seen = 0
    calibration_trained = False
    calibration_train_attempted = False

    online_started = False
    print(f"BCI worker running. Awaiting {calibration_trials_target} calibration trials, then ONLINE_START.")

    pending_trials: deque[PendingTrial] = deque()
    unmatched_epochs: deque[UnmatchedEpoch] = deque()

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
            # Always poll control messages (SESSION_START, ONLINE_START/STOP, TRIAL_START)
            socks = dict(ctrl_poller.poll(0))
            if ctrl in socks:
                cmd = json.loads(ctrl.recv_string())
                action = cmd.get("action")

                if action == "SESSION_START":
                    _ensure_raw_recorder_started()
                    print("[SESSION] SESSION_START received. Raw capture enabled.")

                if action == "ONLINE_START":
                    if not calibration_trained:
                        print("[ONLINE] ONLINE_START ignored: calibration model is not trained yet.")
                    elif not online_started:
                        online_started = True
                        _ensure_raw_recorder_started()
                        print("[ONLINE] ONLINE_START received. Predictions enabled.")

                elif action == "ONLINE_STOP":
                    if raw_recorder is not None and raw_recorder.is_active():
                        raw_recorder.stop()

                elif action == "TRIAL_START":
                    trial_id = int(cmd["trial_id"])
                    code = int(cmd["code"])
                    t_task = float(cmd.get("t_task", time.time()))
                    if code in (LEFT, RIGHT):
                        pending_trials.append(PendingTrial(trial_id=trial_id, code=code, t_task=t_task, t_received=time.time()))

            # Update raw recorder (tap) if active. This does NOT affect epoching.
            if raw_recorder is not None and raw_recorder.is_active():
                raw_recorder.update_from_stream(stream, picks="all")

            # Pull new epochs from online EpochsStream (event-locked to 1/2 only)
            n_new = epochs_online.n_new_epochs
            if n_new > 0:
                X_new = epochs_online.get_data(n_epochs=n_new, picks="eeg")
                X_new = _filter_mi_epochs(X_new)
                codes_new = epochs_online.events[-n_new:]
                now = time.time()
                for i in range(n_new):
                    code = int(codes_new[i])
                    if code in (LEFT, RIGHT):
                        unmatched_epochs.append(UnmatchedEpoch(X=X_new[i], code=code, t_epoch=now))

            # Match trial handshakes to epochs (no drift) for both calibration and live phases.
            matches = _match_trials_to_epochs(pending_trials, unmatched_epochs)

            for pt, ue in matches:
                # Store with optional artifact rejection.
                if reject_thresh is not None:
                    ptp = np.ptp(ue.X, axis=-1).max()
                    if ptp > reject_thresh:
                        if not calibration_trained:
                            calibration_trials_seen += 1
                        continue

                is_calibration_trial = not calibration_trained
                if is_calibration_trial:
                    calibration_trials_seen += 1

                if classifier is not None and online_started:
                    x_i = ue.X[np.newaxis, ...]
                    proba = classifier.predict_proba(x_i)[0]
                    classes = classifier.named_steps["clf"].classes_
                    y_pred_code = int(classes[int(np.argmax(proba))])
                    conf = float(np.max(proba))
                    p_right = _proba_for_code(proba, classes, RIGHT)

                    pub.send(
                        _encode_msg(
                            zmq_cfg.topic,
                            dict(
                                trial_id=int(pt.trial_id),
                                y_true_code=int(pt.code),
                                y_pred_code=int(y_pred_code),
                                conf=conf,
                                p_right=p_right,
                                best_C=best_C,
                                n_total_epochs=int(len(y_store)),
                                t_pred=time.time(),
                                t_epoch=float(ue.t_epoch),
                                t_task=float(pt.t_task) if pt.t_task is not None else None,
                                t_ctrl_received=float(pt.t_received) if pt.t_received is not None else None,
                            ),
                        )
                    )

                if is_calibration_trial:
                    X_cal.append(ue.X)
                    y_cal.append(int(pt.code))

                X_store.append(ue.X)
                y_store.append(int(pt.code))  # 1/2
                X_all.append(ue.X)
                y_all.append(int(pt.code))
                total_accepted += 1

                if is_calibration_trial:
                    pub.send(
                        _encode_msg(
                            zmq_cfg.cal_topic,
                            {
                                "status": "progress",
                                "n_done": int(calibration_trials_seen),
                                "n_target": int(calibration_trials_target),
                                "n_accepted": int(len(y_cal)),
                            },
                        )
                    )

                if (not calibration_trained) and (not calibration_train_attempted) and (calibration_trials_seen >= calibration_trials_target):
                    calibration_train_attempted = True
                    try:
                        classifier, best_C, cv_mean, cv_std, n_per_class = _train_initial_classifier(
                            X_cal=X_cal,
                            y_cal=y_cal,
                            model_cfg=model_cfg,
                            left_code=LEFT,
                            right_code=RIGHT,
                        )
                    except Exception as exc:
                        msg = f"Calibration training failed: {exc}"
                        print(f"[CAL] {msg}")
                        pub.send(_encode_msg(zmq_cfg.cal_topic, {"status": "error", "message": msg}))
                    else:
                        calibration_trained = True
                        last_train_at = total_accepted
                        pub.send(
                            _encode_msg(
                                zmq_cfg.cal_topic,
                                {
                                    "status": "trained",
                                    "cv_mean": cv_mean,
                                    "cv_std": cv_std,
                                    "n_epochs": int(len(y_cal)),
                                    "n_per_class": n_per_class,
                                    "best_C": best_C,
                                },
                            )
                        )
                        print(
                            f"[CAL] Trained on {len(y_cal)} accepted calibration epochs "
                            f"from {calibration_trials_seen}/{calibration_trials_target} trials."
                        )

            # Sliding window
            if model_cfg.use_sliding_window and len(y_store) > model_cfg.window_size_epochs:
                X_store = X_store[-model_cfg.window_size_epochs :]
                y_store = y_store[-model_cfg.window_size_epochs :]

            # Retrain schedule (fixed C)
            if calibration_trained and (total_accepted - last_train_at) >= model_cfg.retrain_every and len(y_store) >= 4:
                X_train = np.stack(X_store, axis=0)
                y_train = np.array(y_store, dtype=int)
                if classifier is not None:
                    classifier.fit(X_train, y_train)
                    last_train_at = total_accepted

            if n_new == 0 and not matches:
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
        for sock in [pub, ctrl]:
            try:
                sock.close(0)
            except Exception:
                pass
        try:
            ctx.term()
        except Exception:
            pass


if __name__ == "__main__":
    main()
