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


def _chop_segment_to_epochs(
    segment: np.ndarray,  # (n_channels, n_total_samples)
    sfreq: float,
    epoch_dur_s: float,
) -> np.ndarray:
    """
    Chop a long, event-locked calibration segment into non-overlapping epochs of epoch_dur_s.
    Returns (n_epochs, n_channels, epoch_samples).
    """
    epoch_samples = int(round(epoch_dur_s * sfreq))
    n_total_samples = segment.shape[1]
    n_epochs = n_total_samples // epoch_samples
    if n_epochs <= 0:
        return np.zeros((0, segment.shape[0], epoch_samples), dtype=segment.dtype)
    trimmed = segment[:, : n_epochs * epoch_samples]
    return trimmed.reshape(segment.shape[0], n_epochs, epoch_samples).transpose(1, 0, 2)


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

    epoch_dur_s = eeg.tmax - eeg.tmin
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
    cal_long_tmax = cal_cfg.cal_tmin + cal_cfg.mi_duration_s
    stream_bufsize = max(30.0, cal_long_tmax + 5.0)

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

    if eeg.notch is not None:
        stream.notch_filter(eeg.notch, picks="eeg")
    stream.filter(eeg.l_freq, eeg.h_freq, picks="eeg")

    sfreq = float(stream.info["sfreq"])
    print(f"Stream connected: sfreq={sfreq:.2f}, bufsize={stream_bufsize}s")

    # Create EpochsStreams
    epochs_cal = EpochsStream(
        stream,
        bufsize=30,
        event_id={"left": LEFT, "right": RIGHT},
        event_channels=lsl.event_channels,
        tmin=cal_cfg.cal_tmin,
        tmax=cal_long_tmax,
        baseline=None,
        reject=None,
    ).connect(acquisition_delay=0.001)

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

    X_store: list[np.ndarray] = []
    y_store: list[int] = []
    X_all: list[np.ndarray] = []
    y_all: list[int] = []
    total_accepted = 0
    last_train_at = 0

    pending_segment: np.ndarray | None = None
    pending_code: int | None = None

    def _drain_epochs(es: EpochsStream, max_iter: int = 20):
        for _ in range(max_iter):
            n_new = es.n_new_epochs
            if n_new <= 0:
                break
            _ = es.get_data(n_epochs=n_new, picks="eeg")

    # ==============================================================
    # PHASE 1: CALIBRATION
    # ==============================================================
    print("BCI worker running. Waiting for calibration commands...")

    calibration_done = False
    while not calibration_done:
        socks = dict(ctrl_poller.poll(50))
        if ctrl not in socks:
            _drain_epochs(epochs_cal, max_iter=1)
            continue

        cmd = json.loads(ctrl.recv_string())
        action = cmd.get("action")

        if action == "CAL_START":
            code = int(cmd["code"])  # 1/2
            if code not in (LEFT, RIGHT):
                print(f"[CAL] Ignoring invalid code={code}")
                continue

            print(f"[CAL] CAL_START code={code} long_epoch=[{cal_cfg.cal_tmin},{cal_long_tmax}]")
            _drain_epochs(epochs_cal, max_iter=100)

            deadline = time.time() + (cal_long_tmax + 5.0)
            got = False
            while time.time() < deadline:
                n_new = epochs_cal.n_new_epochs
                if n_new <= 0:
                    time.sleep(0.005)
                    continue

                X_new = epochs_cal.get_data(n_epochs=n_new, picks="eeg")
                ev_new = epochs_cal.events[-n_new:]

                match_idx = None
                for i in range(n_new - 1, -1, -1):
                    if int(ev_new[i]) == int(code):
                        match_idx = i
                        break
                if match_idx is None:
                    continue

                pending_segment = X_new[match_idx]
                pending_code = code
                got = True
                break

            if not got or pending_segment is None:
                print("[CAL] Timeout waiting for long epoch.")
                pub.send(_encode_msg(zmq_cfg.cal_topic, {"status": "error", "message": "Timeout waiting for calibration epoch"}))
                continue

            pub.send(
                _encode_msg(
                    zmq_cfg.cal_topic,
                    {"status": "collected", "code": int(code), "n_samples": int(pending_segment.shape[1]), "sfreq": sfreq},
                )
            )

        elif action == "CAL_KEEP":
            if pending_segment is None or pending_code is None:
                pub.send(_encode_msg(zmq_cfg.cal_topic, {"status": "kept", "n_kept": 0, "n_total": 0, "cal_epochs_so_far": len(y_store)}))
                continue

            block_epochs = _chop_segment_to_epochs(pending_segment, sfreq, eeg.tmax - eeg.tmin)
            n_total = int(block_epochs.shape[0])
            n_kept = 0

            for i in range(n_total):
                ep = block_epochs[i]
                if reject_thresh is not None:
                    ptp = np.ptp(ep, axis=-1).max()
                    if ptp > reject_thresh:
                        continue
                X_store.append(ep)
                y_store.append(int(pending_code))  # 1/2
                X_all.append(ep)
                y_all.append(int(pending_code))
                n_kept += 1

            total_accepted = len(y_all)
            pub.send(_encode_msg(zmq_cfg.cal_topic, {"status": "kept", "n_kept": n_kept, "n_total": n_total, "cal_epochs_so_far": len(y_store)}))

            pending_segment = None
            pending_code = None

        elif action == "CAL_REJECT":
            pub.send(_encode_msg(zmq_cfg.cal_topic, {"status": "rejected"}))
            pending_segment = None
            pending_code = None

        elif action == "CAL_DONE":
            calibration_done = True

    # --- Train (select C once) ---
    if len(y_store) == 0:
        pub.send(_encode_msg(zmq_cfg.cal_topic, {"status": "error", "message": "No calibration epochs collected"}))
        return

    X_cal_arr = np.stack(X_store, axis=0)
    y_cal_arr = np.array(y_store, dtype=int)

    selector = _make_classifier_riemann_selectC() if model_cfg.use_riemann else _make_classifier_csp_selectC(model_cfg.n_csp_components)
    selector.fit(X_cal_arr, y_cal_arr)
    best_C = _extract_best_C(selector)

    cv_mean, cv_std, _ = _run_cv(X_cal_arr, y_cal_arr, model_cfg, fixed_C=best_C)

    classifier = _make_classifier_riemann_fixedC(best_C) if model_cfg.use_riemann else _make_classifier_csp_fixedC(model_cfg.n_csp_components, best_C)
    classifier.fit(X_cal_arr, y_cal_arr)

    n_per_class = {"1": int(np.sum(y_cal_arr == LEFT)), "2": int(np.sum(y_cal_arr == RIGHT))}
    pub.send(
        _encode_msg(
            zmq_cfg.cal_topic,
            {"status": "trained", "cv_mean": cv_mean, "cv_std": cv_std, "n_epochs": int(len(y_cal_arr)), "n_per_class": n_per_class, "best_C": best_C},
        )
    )

    total_accepted = len(y_store)
    last_train_at = total_accepted

    # ==============================================================
    # PHASE 2: ONLINE (with raw CSV tap)
    # ==============================================================
    print("[ONLINE] Waiting for ONLINE_START...")
    online_started = False

    pending_trials: deque[PendingTrial] = deque()
    unmatched_epochs: deque[UnmatchedEpoch] = deque()

    raw_recorder: RawCSVRecorder | None = None
    raw_csv_path = f"{session_cfg.name}{session_cfg.raw_csv_suffix}"

    # helper: class label -> proba for RIGHT
    def _proba_for_code(proba: np.ndarray, classes: np.ndarray, code: int) -> float:
        idx = int(np.where(classes == int(code))[0][0])
        return float(proba[idx])

    try:
        while True:
            # Always poll control messages (ONLINE_START/STOP, TRIAL_START)
            socks = dict(ctrl_poller.poll(0))
            if ctrl in socks:
                cmd = json.loads(ctrl.recv_string())
                action = cmd.get("action")

                if action == "ONLINE_START":
                    if not online_started:
                        online_started = True
                        # start raw recorder (raw stream tap)
                        ch_names = list(stream.info["ch_names"])  # should match raw_picks order after stream.pick()
                        raw_recorder = RawCSVRecorder(
                            filepath=raw_csv_path,
                            ch_names=ch_names,
                            trigger_ch_name=lsl.event_channels,
                            winsize_s=0.25,
                        )
                        raw_recorder.start()
                        print("[ONLINE] ONLINE_START received. Raw capture enabled.")

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

            # If online hasn't started yet, don't run prediction/matching
            if not online_started:
                time.sleep(0.002)
                continue

            # Pull new epochs from online EpochsStream (event-locked to 1/2 only)
            n_new = epochs_online.n_new_epochs
            if n_new > 0:
                X_new = epochs_online.get_data(n_epochs=n_new, picks="eeg")
                codes_new = epochs_online.events[-n_new:]
                now = time.time()
                for i in range(n_new):
                    code = int(codes_new[i])
                    if code in (LEFT, RIGHT):
                        unmatched_epochs.append(UnmatchedEpoch(X=X_new[i], code=code, t_epoch=now))

            # Match trial handshakes to epochs (no drift)
            matches = _match_trials_to_epochs(pending_trials, unmatched_epochs)

            for pt, ue in matches:
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

                # Store with artifact rejection
                if reject_thresh is not None:
                    ptp = np.ptp(ue.X, axis=-1).max()
                    if ptp > reject_thresh:
                        continue

                X_store.append(ue.X)
                y_store.append(int(pt.code))  # 1/2
                X_all.append(ue.X)
                y_all.append(int(pt.code))
                total_accepted += 1

            # Sliding window
            if model_cfg.use_sliding_window and len(y_store) > model_cfg.window_size_epochs:
                X_store = X_store[-model_cfg.window_size_epochs :]
                y_store = y_store[-model_cfg.window_size_epochs :]

            # Retrain schedule (fixed C)
            if (total_accepted - last_train_at) >= model_cfg.retrain_every and len(y_store) >= 4:
                X_train = np.stack(X_store, axis=0)
                y_train = np.array(y_store, dtype=int)
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

        for resource in [epochs_online, epochs_cal, stream]:
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