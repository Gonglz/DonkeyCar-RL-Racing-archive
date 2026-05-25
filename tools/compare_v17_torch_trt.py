#!/usr/bin/env python3
"""Compare V17 PyTorch manual actor and TensorRT actor on one deterministic obs."""

import argparse
import importlib.util
import os
import sys

import numpy as np

from v17_trt_runtime import V17TensorRTActor


def load_v17_pilot_module(path):
    spec = importlib.util.spec_from_file_location("v17_pilot_runtime_compare", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/jetson/mycar/models/v17_postpass_hard_gate_final_model.zip")
    parser.add_argument("--engine", default="/home/jetson/mycar/models/v17_actor_fp16.engine")
    parser.add_argument("--metadata", default="/home/jetson/mycar/models/v17_actor_export.json")
    parser.add_argument("--pilot", default="/home/jetson/mycar/v17_pilot.py")
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args()

    mod = load_v17_pilot_module(args.pilot)
    pilot = mod.V17Pilot(
        model_path=args.model,
        obs_size=128,
        domain="ws",
        use_cuda=False,
        warmup_frames=0,
    )
    torch_actor = pilot._load_manual_policy()
    trt_actor = V17TensorRTActor(args.engine, metadata_path=args.metadata)

    obs = {
        "image": np.zeros((6, 128, 128), dtype=np.float32),
        "state": np.zeros((7,), dtype=np.float32),
        "lidar": np.concatenate(
            [np.full((72,), 20.0, dtype=np.float32), np.zeros((72,), dtype=np.float32)]
        ),
        "lidar_meta": np.zeros((2,), dtype=np.float32),
    }

    torch_action = np.asarray(torch_actor.predict_np(obs), dtype=np.float32).reshape(-1)
    trt_action = np.asarray(trt_actor.predict_np(obs), dtype=np.float32).reshape(-1)
    diff = np.abs(torch_action - trt_action)
    print("torch_action:", np.array2string(torch_action, precision=6))
    print("trt_action:", np.array2string(trt_action, precision=6))
    print("abs_diff:", np.array2string(diff, precision=6))
    print("max_abs_diff:", float(diff.max()))
    if float(diff.max()) > float(args.tolerance):
        raise SystemExit("max_abs_diff exceeds tolerance")


if __name__ == "__main__":
    main()
