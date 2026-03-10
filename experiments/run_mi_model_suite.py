#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime


def parse_args():
    ap = argparse.ArgumentParser(description="Run MI model suite on a collected dataset")
    ap.add_argument("--windows", required=True, help="Path to windows .npy (N,C,T)")
    ap.add_argument("--labels", required=True, help="Path to labels .npy (N,)")
    ap.add_argument("--block-ids", default="", help="Optional block ids .npy (N,). If absent, inferred by class order.")
    ap.add_argument("--sfreq", type=float, default=300.0)
    ap.add_argument("--windows-per-block", type=int, default=29)
    ap.add_argument("--out-dir", default="experiments/results")
    ap.add_argument("--tag", default="", help="Optional run tag suffix")
    return ap.parse_args()


def run_cmd(cmd: list[str]):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(args.windows))[0]
    tag = f"_{args.tag}" if args.tag else ""
    os.makedirs(args.out_dir, exist_ok=True)

    bakeoff_json = os.path.join(args.out_dir, f"{base}_bakeoff_{ts}{tag}.json")
    optimize_json = os.path.join(args.out_dir, f"{base}_3class_opt_{ts}{tag}.json")

    bakeoff_cmd = [
        sys.executable,
        "experiments/mi_model_bakeoff.py",
        "--windows",
        args.windows,
        "--labels",
        args.labels,
        "--sfreq",
        str(args.sfreq),
        "--windows-per-block",
        str(args.windows_per_block),
        "--out-json",
        bakeoff_json,
    ]
    if args.block_ids:
        bakeoff_cmd.extend(["--block-ids", args.block_ids])

    optimize_cmd = [
        sys.executable,
        "experiments/mi_3class_optimize.py",
        "--windows",
        args.windows,
        "--labels",
        args.labels,
        "--out-json",
        optimize_json,
    ]

    run_cmd(bakeoff_cmd)
    run_cmd(optimize_cmd)

    print("\n[OK] Model suite finished")
    print(f"- Bakeoff report: {bakeoff_json}")
    print(f"- 3-class optimize report: {optimize_json}")


if __name__ == "__main__":
    main()
