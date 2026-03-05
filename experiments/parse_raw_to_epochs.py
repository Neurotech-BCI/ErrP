#!/usr/bin/env python3
"""
Parse raw CSV files (t, ch1..chN, TRG columns) into epoched npy arrays.
Trigger codes mark trial onset. We epoch from tmin to tmax after each trigger.
"""
import argparse, os
import numpy as np
import pandas as pd
from mne.filter import filter_data

def parse(csv_path, tmin, tmax, sfreq, left_codes, right_codes, rest_codes=None,
          bandpass=(1.0, 40.0), out_dir=None):
    df = pd.read_csv(csv_path)
    trg_col = [c for c in df.columns if c.upper() in ('TRG', 'TRIGGER', 'STI')]
    eeg_cols = [c for c in df.columns if c not in (['t'] + trg_col)]
    trg = df[trg_col[0]].values if trg_col else np.zeros(len(df))
    eeg = df[eeg_cols].values.T.astype(np.float32)  # (n_ch, n_samples)
    
    # Infer sfreq from timestamps if possible
    if 't' in df.columns:
        dt = np.diff(df['t'].values[:1000])
        dt = dt[dt > 0]
        if len(dt):
            sfreq_inferred = 1.0 / np.median(dt)
            print(f"  Inferred sfreq={sfreq_inferred:.1f} Hz")
            sfreq = sfreq_inferred

    tmin_samp = int(round(tmin * sfreq))
    tmax_samp = int(round(tmax * sfreq))
    n_samp = tmax_samp - tmin_samp

    all_codes = set(left_codes) | set(right_codes) | (set(rest_codes) if rest_codes else set())
    onsets = np.where(np.diff(np.round(trg).astype(int)) > 0)[0] + 1
    
    X, y = [], []
    for onset in onsets:
        code = int(round(trg[onset]))
        if code not in all_codes:
            continue
        start = onset + tmin_samp
        end = onset + tmax_samp
        if start < 0 or end > eeg.shape[1]:
            continue
        epoch = eeg[:, start:end]
        if code in left_codes: label = 1
        elif code in right_codes: label = 2
        elif rest_codes and code in rest_codes: label = 0
        else: continue
        X.append(epoch)
        y.append(label)

    if not X:
        print(f"  No epochs found. Trigger values present: {np.unique(np.round(trg).astype(int))[:20]}")
        return None, None

    X = np.stack(X, axis=0)
    y = np.array(y, dtype=int)
    
    # Bandpass
    X = filter_data(X.astype(np.float64), sfreq=sfreq, l_freq=bandpass[0], h_freq=bandpass[1], verbose="ERROR").astype(np.float32)
    
    uniq, cnt = np.unique(y, return_counts=True)
    print(f"  Epochs: {X.shape}, labels: {dict(zip(map(int,uniq),map(int,cnt)))}")

    if out_dir:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        np.save(os.path.join(out_dir, f"{base}_data.npy"), X)
        np.save(os.path.join(out_dir, f"{base}_labels.npy"), y)
        print(f"  Saved to {out_dir}/{base}_data.npy")

    return X, y


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--tmin", type=float, default=0.5)
    ap.add_argument("--tmax", type=float, default=2.5)
    ap.add_argument("--sfreq", type=float, default=300.0)
    ap.add_argument("--left-codes", type=int, nargs="+", default=[1])
    ap.add_argument("--right-codes", type=int, nargs="+", default=[2])
    ap.add_argument("--rest-codes", type=int, nargs="*", default=None)
    ap.add_argument("--out-dir", default="data/extracted")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Processing {args.csv}")
    parse(args.csv, args.tmin, args.tmax, args.sfreq, args.left_codes, args.right_codes,
          args.rest_codes, out_dir=args.out_dir)
