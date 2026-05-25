#!/usr/bin/env python3
"""Smoke test for the V17 TensorRT actor runtime."""

import argparse
import json

import numpy as np

from v17_trt_runtime import V17TensorRTActor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="/home/jetson/mycar/models/v17_actor_fp16.engine")
    parser.add_argument("--metadata", default="/home/jetson/mycar/models/v17_actor_export.json")
    args = parser.parse_args()

    with open(args.metadata, "r", encoding="utf-8") as f:
        meta = json.load(f)
    shape = meta["shape"]

    actor = V17TensorRTActor(args.engine, metadata_path=args.metadata)
    obs = {
        "image": np.zeros((shape["image_channels"], shape["obs_size"], shape["obs_size"]), dtype=np.float32),
        "state": np.zeros((shape["state_dim"],), dtype=np.float32),
        "lidar": np.concatenate(
            [
                np.full((shape["lidar_dim"] // 2,), 20.0, dtype=np.float32),
                np.zeros((shape["lidar_dim"] // 2,), dtype=np.float32),
            ]
        ),
        "lidar_meta": np.zeros((shape["lidar_meta_dim"],), dtype=np.float32),
    }
    action = actor.predict_np(obs)
    assert action.shape == (3,), action.shape
    assert np.isfinite(action).all(), action
    assert np.max(np.abs(action)) <= 1.0001, action
    print("action:", np.array2string(action, precision=6))
    print("bindings:", actor.binding_summary())


if __name__ == "__main__":
    main()
