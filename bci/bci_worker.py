# bci_worker.py
import time
import json
import numpy as np
import zmq

from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_score

from mne_lsl.stream import StreamLSL, EpochsStream

from config import (
    LSLConfig, EEGConfig, ModelConfig, ZMQConfig,
    SessionConfig, CalibrationConfig,
)

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


# ----------------------------------------------------------------
# Classifier builders
# ----------------------------------------------------------------

def _make_classifier_riemann() -> Pipeline:
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
    return Pipeline([
        ("cov", cov),
        ("ts", ts),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", clf),
    ])


def _make_classifier_csp(n_csp_components: int) -> Pipeline:
    csp = CSP(n_components=n_csp_components, reg="ledoit_wolf", log=True, norm_trace=False)
    clf = LogisticRegressionCV(
        Cs=[0.01, 0.1, 1.0, 10.0],
        cv=3,
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline([
        ("csp", csp),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _encode_msg(topic: str, payload: dict) -> bytes:
    return f"{topic} {json.dumps(payload)}".encode("utf-8")


def _chop_segment_to_epochs(
    segment: np.ndarray,   # (n_channels, n_total_samples)
    sfreq: float,
    epoch_dur_s: float,
    mi_duration_s: float,
) -> np.ndarray:
    """Chop a raw segment into non-overlapping epochs.
    Returns (n_epochs, n_channels, epoch_samples)."""
    epoch_samples = int(epoch_dur_s * sfreq)
    n_epochs = int(mi_duration_s / epoch_dur_s)
    return np.stack([
        segment[:, i * epoch_samples : (i + 1) * epoch_samples]
        for i in range(n_epochs)
    ], axis=0)


def _run_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_cfg: ModelConfig,
) -> tuple[float, float, np.ndarray]:
    """Stratified k-fold CV. Returns (mean_acc, std_acc, fold_scores)."""
    n_per_class = np.bincount(y, minlength=2)
    n_splits = min(5, int(n_per_class.min()))
    if n_splits < 2:
        print("[CV] Not enough data per class for cross-validation")
        return 0.0, 0.0, np.array([])
    eval_clf = (
        _make_classifier_riemann() if model_cfg.use_riemann
        else _make_classifier_csp(model_cfg.n_csp_components)
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(eval_clf, X, y, cv=cv, scoring="accuracy")
    print(f"[CV] {n_splits}-fold: {[f'{s:.3f}' for s in scores]}  "
          f"mean={scores.mean():.3f} +/- {scores.std():.3f}")
    return float(scores.mean()), float(scores.std()), scores


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

    epoch_dur_s = eeg.tmax - eeg.tmin  # online epoch duration (2.0s)
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
    stream_bufsize = cal_cfg.mi_duration_s + 2.0
    stream = StreamLSL(
        bufsize=stream_bufsize,
        name=lsl.name,
        stype=lsl.stype,
        source_id=lsl.source_id,
    )
    stream.connect(acquisition_delay=0.001, processing_flags="all")
    print(f"Original channel names: {stream.info['ch_names']}")
    stream.pick(list(eeg.picks) + [lsl.event_channels])
    stream.set_channel_types({lsl.event_channels: 'stim'})

    if eeg.notch is not None:
        stream.notch_filter(eeg.notch, picks="eeg")
    stream.filter(eeg.l_freq, eeg.h_freq, picks="eeg")

    sfreq = stream.info["sfreq"]
    epoch_samples = int(epoch_dur_s * sfreq)
    print(f"Stream connected: sfreq={sfreq}, bufsize={stream_bufsize}s, "
          f"epoch={epoch_dur_s}s ({epoch_samples} samples)")

    # Tracking variables initialised after calibration
    epochs_stream = None
    classifier = None
    X_store: list[np.ndarray] = []
    y_store: list[int] = []
    X_all: list[np.ndarray] = []
    y_all: list[int] = []
    total_accepted = 0
    last_train_at = 0

    # ==============================================================
    # PHASE 1: SUSTAINED-MI CALIBRATION
    # ==============================================================
    X_cal: list[np.ndarray] = []
    y_cal: list[int] = []
    pending_segment: np.ndarray | None = None
    pending_label: int | None = None

    print("BCI worker running. Waiting for calibration commands...")

    calibration_done = False
    while not calibration_done:
        socks = dict(ctrl_poller.poll(50))
        if ctrl not in socks:
            continue

        cmd = json.loads(ctrl.recv_string())
        action = cmd.get("action")

        if action == "CAL_START":
            label = int(cmd["label"])
            print(f"[CAL] MI block: {'LEFT' if label==0 else 'RIGHT'} "
                  f"({cal_cfg.mi_duration_s}s)")

            # Drain stale data from ring buffer
            stream.get_data(picks="eeg")

            # Accumulate MI data
            time.sleep(cal_cfg.mi_duration_s + 0.1)

            data, _ = stream.get_data(
                winsize=cal_cfg.mi_duration_s, picks="eeg"
            )
            target_samples = int(cal_cfg.mi_duration_s * sfreq)
            if data.shape[1] > target_samples:
                data = data[:, :target_samples]

            pending_segment = data
            pending_label = label

            pub.send(_encode_msg(zmq_cfg.cal_topic, {
                "status": "collected",
                "label": label,
                "n_samples": int(data.shape[1]),
            }))
            print(f"[CAL] Collected {data.shape[1]} samples "
                  f"({data.shape[1]/sfreq:.1f}s)")

        elif action == "CAL_KEEP":
            if pending_segment is not None:
                block_epochs = _chop_segment_to_epochs(
                    pending_segment, sfreq, epoch_dur_s,
                    cal_cfg.mi_duration_s,
                )
                n_total = block_epochs.shape[0]
                n_kept = 0
                for i in range(n_total):
                    ep = block_epochs[i]
                    if reject_thresh is not None:
                        ptp = np.ptp(ep, axis=-1).max()
                        if ptp > reject_thresh:
                            print(f"[CAL] Epoch {i} rejected: "
                                  f"ptp={ptp*1e6:.1f} uV")
                            continue
                    X_cal.append(ep)
                    y_cal.append(pending_label)
                    n_kept += 1

                print(f"[CAL] KEEP: {n_kept}/{n_total} epochs passed "
                      f"artifact check (total: {len(y_cal)})")
                pub.send(_encode_msg(zmq_cfg.cal_topic, {
                    "status": "kept",
                    "n_kept": n_kept,
                    "n_total": n_total,
                    "cal_epochs_so_far": len(y_cal),
                }))
            pending_segment = None
            pending_label = None

        elif action == "CAL_REJECT":
            print("[CAL] Block rejected by user")
            pub.send(_encode_msg(zmq_cfg.cal_topic, {
                "status": "rejected",
            }))
            pending_segment = None
            pending_label = None

        elif action == "CAL_DONE":
            calibration_done = True
            print(f"[CAL] Calibration complete: {len(y_cal)} total epochs")

    # --- Post-calibration CV and initial training ---
    X_cal_arr = np.stack(X_cal, axis=0)
    y_cal_arr = np.array(y_cal, dtype=int)

    print(f"[CAL] Calibration data: {X_cal_arr.shape}")
    cv_mean, cv_std, cv_scores = _run_cv(X_cal_arr, y_cal_arr, model_cfg)

    if model_cfg.use_riemann:
        classifier = _make_classifier_riemann()
    else:
        classifier = _make_classifier_csp(model_cfg.n_csp_components)

    classifier.fit(X_cal_arr, y_cal_arr)
    n_per_class = np.bincount(y_cal_arr, minlength=2)
    print(f"[CAL] Classifier trained on {len(y_cal)} epochs "
          f"({n_per_class[0]} left, {n_per_class[1]} right)")

    pub.send(_encode_msg(zmq_cfg.cal_topic, {
        "status": "trained",
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "n_epochs": len(y_cal),
        "n_per_class": n_per_class.tolist(),
    }))

    # Seed stores with calibration data
    X_store = list(X_cal)
    y_store = list(y_cal)
    X_all = list(X_cal)
    y_all = list(y_cal)
    total_accepted = len(y_cal)
    last_train_at = total_accepted

    # ==============================================================
    # PHASE 2: ONLINE PREDICTION
    # ==============================================================
    epochs_stream = EpochsStream(
        stream,
        bufsize=30,
        event_id={"left": 1, "right": 2},
        event_channels=lsl.event_channels,
        tmin=eeg.tmin,
        tmax=eeg.tmax,
        baseline=eeg.baseline,
        reject=None,
    ).connect(acquisition_delay=0.001)

    trial_counter = 0
    print("[ONLINE] EpochsStream created. Entering online phase...")

    try:
        while True:
            n_new = epochs_stream.n_new_epochs
            if n_new == 0:
                time.sleep(0.005)
                continue

            X_new = epochs_stream.get_data(n_epochs=n_new, picks="eeg")
            y_new = epochs_stream.events[-n_new:]

            # Predict BEFORE storing/training
            for i in range(n_new):
                x_i = X_new[i:i+1]
                y_true = int(y_new[i])

                proba = classifier.predict_proba(x_i)[0]
                y_pred = int(np.argmax(proba))
                conf = float(np.max(proba))
                p_right = float(proba[1])

                payload = dict(
                    trial_index=trial_counter,
                    y_true=y_true,
                    y_pred=y_pred,
                    conf=conf,
                    p_right=p_right,
                    trained=True,
                    n_total_epochs=len(y_store),
                    t=time.time(),
                )
                pub.send(_encode_msg(zmq_cfg.topic, payload))
                trial_counter += 1

            # Store with artifact rejection
            for i in range(n_new):
                if reject_thresh is not None:
                    ptp = np.ptp(X_new[i], axis=-1).max()
                    if ptp > reject_thresh:
                        print(f"[REJECT] epoch ptp={ptp*1e6:.1f} uV")
                        continue
                label = int(y_new[i]) - 1
                X_store.append(X_new[i])
                y_store.append(label)
                X_all.append(X_new[i])
                y_all.append(label)
                total_accepted += 1

            # Sliding window
            if model_cfg.use_sliding_window and len(y_store) > model_cfg.window_size_epochs:
                X_store = X_store[-model_cfg.window_size_epochs:]
                y_store = y_store[-model_cfg.window_size_epochs:]

            # Retrain schedule
            if (total_accepted - last_train_at) >= model_cfg.retrain_every:
                X_train = np.stack(X_store, axis=0)
                y_train = np.array(y_store, dtype=int)
                classifier.fit(X_train, y_train)
                last_train_at = total_accepted
                print(f"[TRAIN] retrained on {len(y_store)} epochs")

    except KeyboardInterrupt:
        print("\nStopping BCI worker...")

    finally:
        # --- Save session data and final CV ---
        if len(y_all) > 0:
            X_save = np.stack(X_all, axis=0)
            y_save = np.array(y_all, dtype=int)
            data_path = f"{session_cfg.name}_data.npy"
            labels_path = f"{session_cfg.name}_labels.npy"
            np.save(data_path, X_save)
            np.save(labels_path, y_save)
            n_per_class = np.bincount(y_save, minlength=2)
            print(f"\nSaved {len(y_all)} epochs -> {data_path} "
                  f"{X_save.shape}, {labels_path} {y_save.shape}")

            cv_mean, cv_std, cv_scores = _run_cv(X_save, y_save, model_cfg)
            if len(cv_scores) > 0:
                print(f"\n{'='*45}")
                print(f"  Final Cross-Validated Accuracy")
                print(f"  Folds : {[f'{s:.3f}' for s in cv_scores]}")
                print(f"  Mean  : {cv_mean:.3f} +/- {cv_std:.3f}")
                print(f"  Epochs: {len(y_save)} "
                      f"({n_per_class[0]} left, {n_per_class[1]} right)")
                print(f"{'='*45}")

        for resource in [epochs_stream, stream]:
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
