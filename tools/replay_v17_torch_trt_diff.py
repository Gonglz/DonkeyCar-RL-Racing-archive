#!/usr/bin/env python3
"""Continuous-rollout PyTorch vs TensorRT diff for V17 actor.

This is stronger than a one-sample smoke test because the PyTorch and TensorRT
actors both keep their recurrent h/c state across the whole generated replay.
"""

import argparse
import importlib.util
import json
import os
import sys
from typing import Dict, List

import numpy as np

from v17_trt_runtime import V17TensorRTActor


def load_module(path: str):
    path = os.path.abspath(os.path.expanduser(path))
    spec = importlib.util.spec_from_file_location("v17_pilot_replay_diff", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def percentile(values: List[float], pct: float):
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * (float(pct) / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def stats(values: List[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    return {
        "mean": None if not vals else float(np.mean(vals)),
        "p95": percentile(vals, 95),
        "max": None if not vals else float(np.max(vals)),
    }


def make_obs(rng, shape: Dict[str, int]) -> Dict[str, np.ndarray]:
    image_channels = int(shape.get("image_channels", 6))
    obs_size = int(shape.get("obs_size", 128))
    state_dim = int(shape.get("state_dim", 7))
    lidar_dim = int(shape.get("lidar_dim", 144))
    lidar_meta_dim = int(shape.get("lidar_meta_dim", 2))
    sectors = lidar_dim // 2

    image = rng.uniform(0.0, 1.0, size=(image_channels, obs_size, obs_size)).astype(np.float32)
    state = rng.uniform(-1.0, 1.0, size=(state_dim,)).astype(np.float32)
    sector_ranges = rng.uniform(0.18, 20.0, size=(sectors,)).astype(np.float32)
    sector_valid = (rng.uniform(0.0, 1.0, size=(sectors,)) > 0.08).astype(np.float32)
    lidar = np.concatenate([sector_ranges, sector_valid]).astype(np.float32)

    lidar_meta = np.zeros((lidar_meta_dim,), dtype=np.float32)
    if lidar_meta_dim >= 1:
        valid_ranges = sector_ranges[sector_valid > 0.5]
        lidar_meta[0] = float(np.min(valid_ranges)) if valid_ranges.size else 20.0
    if lidar_meta_dim >= 2:
        lidar_meta[1] = float(np.mean(sector_valid))

    return {
        "image": image,
        "state": state,
        "lidar": lidar,
        "lidar_meta": lidar_meta,
    }


def tensor_to_numpy(value):
    return value.detach().cpu().numpy()


def write_markdown(path: str, summary: Dict[str, object]) -> None:
    rows = [
        ("samples", summary["samples"]),
        ("seed", summary["seed"]),
        ("action max abs diff", summary["action_abs_diff"]["max"]),
        ("action mean abs diff", summary["action_abs_diff"]["mean"]),
        ("action p95 abs diff", summary["action_abs_diff"]["p95"]),
        ("next_h max abs diff", summary["next_h_abs_diff"]["max"]),
        ("next_h p95 abs diff", summary["next_h_abs_diff"]["p95"]),
        ("next_c max abs diff", summary["next_c_abs_diff"]["max"]),
        ("next_c p95 abs diff", summary["next_c_abs_diff"]["p95"]),
        ("final action abs diff", summary["final_action_abs_diff"]),
        ("action tolerance", summary["action_tolerance"]),
        ("hidden p95 tolerance", summary["hidden_p95_tolerance"]),
        ("nan_or_inf_count", summary["nan_or_inf_count"]),
        ("pass", str(summary["pass"]).lower()),
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# V17 Replay PyTorch vs TensorRT Diff\n\n")
        f.write("| metric | value |\n")
        f.write("|---|---:|\n")
        for key, value in rows:
            if isinstance(value, float):
                text = "%.9g" % value
            else:
                text = str(value)
            f.write("| %s | %s |\n" % (key, text))
        f.write("\n")
        f.write("The replay resets h/c once at the beginning, then compares a continuous rollout.\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/jetson/mycar/models/v17_postpass_hard_gate_final_model.zip")
    parser.add_argument("--engine", default="/home/jetson/mycar/models/v17_actor_fp16.engine")
    parser.add_argument("--metadata", default="/home/jetson/mycar/models/v17_actor_export.json")
    parser.add_argument("--pilot", default="/home/jetson/mycar/v17_pilot.py")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17017)
    parser.add_argument("--domain", default="ws")
    parser.add_argument("--torch-cuda", action="store_true")
    parser.add_argument("--action-tolerance", type=float, default=0.02)
    parser.add_argument("--hidden-p95-tolerance", type=float, default=0.02)
    parser.add_argument("--out-json", default="replay_diff_summary.json")
    parser.add_argument("--out-md", default="replay_diff_summary.md")
    args = parser.parse_args()

    with open(args.metadata, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    shape = metadata.get("shape", {})

    mod = load_module(args.pilot)
    pilot = mod.V17Pilot(
        model_path=args.model,
        obs_size=int(shape.get("obs_size", 128)),
        domain=args.domain,
        use_cuda=bool(args.torch_cuda),
        warmup_frames=0,
    )
    torch_actor = pilot._load_manual_policy()
    torch_actor.reset()
    trt_actor = V17TensorRTActor(args.engine, metadata_path=args.metadata)
    trt_actor.reset()

    rng = np.random.RandomState(args.seed)
    action_diffs = []
    h_diffs = []
    c_diffs = []
    nan_or_inf_count = 0
    first_failure_index = None
    final_action_diff = None

    for idx in range(int(args.samples)):
        obs = make_obs(rng, shape)
        torch_action = np.asarray(torch_actor.predict_np(obs), dtype=np.float32).reshape(-1)
        trt_action = np.asarray(trt_actor.predict_np(obs), dtype=np.float32).reshape(-1)
        torch_h = tensor_to_numpy(torch_actor.h).astype(np.float32)
        torch_c = tensor_to_numpy(torch_actor.c).astype(np.float32)
        trt_h = np.asarray(trt_actor.host["h"], dtype=np.float32)
        trt_c = np.asarray(trt_actor.host["c"], dtype=np.float32)

        values = [torch_action, trt_action, torch_h, torch_c, trt_h, trt_c]
        if any(not np.isfinite(v).all() for v in values):
            nan_or_inf_count += 1
            if first_failure_index is None:
                first_failure_index = idx

        action_diff = np.abs(torch_action - trt_action)
        h_diff = np.abs(torch_h - trt_h)
        c_diff = np.abs(torch_c - trt_c)
        action_diffs.extend(float(x) for x in action_diff.reshape(-1))
        h_diffs.extend(float(x) for x in h_diff.reshape(-1))
        c_diffs.extend(float(x) for x in c_diff.reshape(-1))
        final_action_diff = float(np.max(action_diff))

    action_stats = stats(action_diffs)
    h_stats = stats(h_diffs)
    c_stats = stats(c_diffs)
    passed = (
        nan_or_inf_count == 0
        and float(action_stats["max"] or 0.0) <= float(args.action_tolerance)
        and float(h_stats["p95"] or 0.0) <= float(args.hidden_p95_tolerance)
        and float(c_stats["p95"] or 0.0) <= float(args.hidden_p95_tolerance)
    )
    summary = {
        "samples": int(args.samples),
        "seed": int(args.seed),
        "model": os.path.abspath(os.path.expanduser(args.model)),
        "engine": os.path.abspath(os.path.expanduser(args.engine)),
        "metadata": os.path.abspath(os.path.expanduser(args.metadata)),
        "pilot": os.path.abspath(os.path.expanduser(args.pilot)),
        "torch_cuda": bool(args.torch_cuda),
        "continuous_lstm_rollout": True,
        "action_tolerance": float(args.action_tolerance),
        "hidden_p95_tolerance": float(args.hidden_p95_tolerance),
        "action_abs_diff": action_stats,
        "next_h_abs_diff": h_stats,
        "next_c_abs_diff": c_stats,
        "final_action_abs_diff": final_action_diff,
        "nan_or_inf_count": int(nan_or_inf_count),
        "first_failure_index": first_failure_index,
        "pass": bool(passed),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    write_markdown(args.out_md, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    trt_actor.close()
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
