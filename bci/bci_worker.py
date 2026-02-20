# bci_worker.py
import time
import json
import numpy as np
import zmq

import mne
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from mne_lsl.stream import StreamLSL, EpochsStream  # per MNE-LSL decoding example :contentReference[oaicite:7]{index=7}

from config import LSLConfig, EEGConfig, ModelConfig, ZMQConfig


def _make_classifier(n_csp_components: int) -> Pipeline:
    # CSP expects (n_epochs, n_channels, n_times)
    # and produces (n_epochs, n_components) with log-variance features.
    csp = CSP(n_components=n_csp_components, reg="ledoit_wolf", log=True, norm_trace=False)
    clf = LogisticRegression(
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


def _encode_pred(topic: str, payload: dict) -> bytes:
    msg = f"{topic} {json.dumps(payload)}"
    return msg.encode("utf-8")


def main():
    lsl = LSLConfig()
    eeg = EEGConfig()
    model_cfg = ModelConfig()
    zmq_cfg = ZMQConfig()

    # --- ZMQ PUB (to PsychoPy) ---
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.bind(zmq_cfg.pub_addr)
    time.sleep(0.2)  # allow SUB to connect

    # --- Connect to EEG LSL stream ---
    # StreamLSL identifiers must uniquely identify the stream :contentReference[oaicite:8]{index=8}
    stream = StreamLSL(
        bufsize=10,
        name=lsl.name,
        stype=lsl.stype,
        source_id=lsl.source_id,
    )
    stream.connect(acquisition_delay=0.001, processing_flags="all")
    print(f"Original channel names: {stream.info['ch_names']}")
    stream.pick(list(eeg.picks) + [lsl.event_channels])
    stream.set_channel_types({'TRG': 'stim'})
    # Ensure channel types are correct if needed; often stim channel is already set.
    # Apply stream-level filters (MNE-LSL supports stream filters) :contentReference[oaicite:9]{index=9}
    if eeg.notch is not None:
        try:
            stream.notch_filter(eeg.notch)
        except Exception:
            # Some versions may require specifying picks; keep robust.
            stream.notch_filter(eeg.notch, picks="eeg")

    # Bandpass for MI (mu/beta)
    stream.filter(eeg.l_freq, eeg.h_freq, picks="eeg")


    # --- Epoch extraction around event triggers ---
    # EpochsStream monitors stim channel(s) and extracts epochs on events :contentReference[oaicite:10]{index=10}
    epochs = EpochsStream(
        stream,
        bufsize=30,  # seconds worth of epochs buffer
        event_id={"left":1, "right":2},
        event_channels=lsl.event_channels,
        tmin=eeg.tmin,
        tmax=eeg.tmax,
        baseline=eeg.baseline,
        reject=None,
    ).connect(acquisition_delay=0.001)

    classifier = _make_classifier(model_cfg.n_csp_components)
    is_trained = False

    X_store: list[np.ndarray] = []
    y_store: list[int] = []

    last_seen_total = 0
    trial_counter = 0

    print("BCI worker running.")
    print(f"Listening to LSL stream name={lsl.name!r}, stype={lsl.stype!r}, event_channel={lsl.event_channels!r}")
    print("Waiting for events...")

    try:
        while True:
            # n_new_epochs counts epochs not yet fetched/consumed
            n_new = epochs.n_new_epochs
            if n_new == 0:
                time.sleep(0.005)
                continue
                
            print(f"Got new trigger, n_new={n_new}")
            # Fetch only the new epochs
            X_new = epochs.get_data(n_epochs=n_new)  # shape: (n_new, n_ch, n_times)
            print(f"New data has shape: {X_new.shape}")
            # epochs.events contains MNE-style event array; last column is event code
            y_new = epochs.events[-n_new:]
            # Store
            for i in range(n_new):
                X_store.append(X_new[i])
                y_store.append(int(y_new[i])-1)

            # Sliding window for nonstationarity / user learning
            if model_cfg.use_sliding_window and len(y_store) > model_cfg.window_size_epochs:
                X_store = X_store[-model_cfg.window_size_epochs:]
                y_store = y_store[-model_cfg.window_size_epochs:]

            # Train / retrain schedule
            n_total = len(y_store)
            should_train = (
                (not is_trained and n_total >= model_cfg.min_epochs_to_train) or
                (is_trained and (n_total - last_seen_total) >= model_cfg.retrain_every)
            )

            if should_train:
                X_train = np.stack(X_store, axis=0)
                y_train = np.array(y_store, dtype=int)
                classifier.fit(X_train, y_train)
                is_trained = True
                last_seen_total = n_total
                print(f"[TRAIN] trained on {n_total} epochs")

            # Predict each new epoch immediately (even if untrained -> random)
            for i in range(n_new):
                x_i = X_new[i:i+1]
                y_true = int(y_new[i])

                if is_trained:
                    proba = classifier.predict_proba(x_i)[0]  # [p(class0), p(class1)]
                    y_pred = int(np.argmax(proba))
                    conf = float(np.max(proba))
                    p_right = float(proba[1])
                else:
                    y_pred = int(np.random.randint(0, 2))
                    conf = 0.5
                    p_right = 0.5

                payload = dict(
                    trial_index=trial_counter,
                    y_true=y_true,
                    y_pred=y_pred,
                    conf=conf,
                    p_right=p_right,
                    trained=is_trained,
                    n_total_epochs=n_total,
                    t=time.time(),
                )
                pub.send(_encode_pred(zmq_cfg.topic, payload))
                trial_counter += 1

    except KeyboardInterrupt:
        print("\nStopping BCI worker...")

    finally:
        try:
            epochs.disconnect()
        except Exception:
            pass
        try:
            stream.disconnect()
        except Exception:
            pass
        try:
            pub.close(0)
        except Exception:
            pass
        try:
            ctx.term()
        except Exception:
            pass


if __name__ == "__main__":
    main()
