#!/usr/bin/env python3
"""
Evaluate target-token extraction quality against simulator obstacle truth.

This script measures whether a "LiDAR only handles foreground targets" front-end
is viable before fully switching V17 away from full-scene 36-sector inputs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from module.lidar import CanonicalLidarSpec, TargetTokenBuffer  # noqa: E402


def _iter_jsonl(paths: List[str]) -> Iterator[dict]:
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            files = sorted(path.rglob("*.jsonl"))
        else:
            files = [path]
        for file_path in files:
            if not file_path.is_file() or file_path.suffix.lower() != ".jsonl":
                continue
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def _safe_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return int(default)


def _empty_stats() -> Dict[str, object]:
    return {
        "frames": 0,
        "gt_positive": 0,
        "pred_positive": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "rel_long_err": [],
        "rel_lat_err": [],
        "rel_v_long_err": [],
        "rel_v_lat_err": [],
        "ttc_err": [],
        "confidence_tp": [],
        "age_norm_tp": [],
    }


def _finalize_stats(stats: Dict[str, object]) -> Dict[str, float]:
    frames = max(1, int(stats["frames"]))
    gt_positive = max(1, int(stats["gt_positive"]))
    pred_positive = max(1, int(stats["pred_positive"]))
    tp = int(stats["true_positive"])

    def _mean(values: List[float]) -> float:
        if not values:
            return 0.0
        return float(np.mean(np.asarray(values, dtype=np.float32)))

    return {
        "frames": int(stats["frames"]),
        "gt_positive": int(stats["gt_positive"]),
        "pred_positive": int(stats["pred_positive"]),
        "true_positive": int(stats["true_positive"]),
        "false_positive": int(stats["false_positive"]),
        "false_negative": int(stats["false_negative"]),
        "recall": float(tp / gt_positive),
        "precision": float(tp / pred_positive),
        "positive_rate": float(int(stats["pred_positive"]) / frames),
        "mae_rel_long": _mean(stats["rel_long_err"]),
        "mae_rel_lat": _mean(stats["rel_lat_err"]),
        "mae_rel_v_long": _mean(stats["rel_v_long_err"]),
        "mae_rel_v_lat": _mean(stats["rel_v_lat_err"]),
        "mae_ttc": _mean(stats["ttc_err"]),
        "mean_confidence_tp": _mean(stats["confidence_tp"]),
        "mean_age_norm_tp": _mean(stats["age_norm_tp"]),
    }


def _gt_ttc(rel_long: float, rel_v_long: float, max_ttc_s: float) -> float:
    if rel_long > 0.0 and rel_v_long < -0.05:
        return float(np.clip(rel_long / max(-rel_v_long, 1e-3), 0.0, max_ttc_s))
    return float(max_ttc_s)


def evaluate(paths: List[str], max_range_m: float, near_clip_m: float, control_dt_s: float, conf_thresh: float) -> Dict[str, object]:
    spec = CanonicalLidarSpec(max_range_m=max_range_m, near_clip_m=near_clip_m, invalid_fill_m=max_range_m)
    buffers: Dict[Tuple[str, int], TargetTokenBuffer] = {}
    prev_truth: Dict[Tuple[str, int], Dict[str, float]] = {}
    overall = _empty_stats()
    by_scene: Dict[str, Dict[str, object]] = {}

    for obj in _iter_jsonl(paths):
        scene = str(obj.get("scene_key", obj.get("scene", "all")) or "all")
        ep = _safe_int(obj.get("episode_id", 0), 0)
        key = (scene, ep)
        if key not in buffers:
            buffers[key] = TargetTokenBuffer(spec=spec, control_dt_s=control_dt_s)
            prev_truth[key] = {}
        if scene not in by_scene:
            by_scene[scene] = _empty_stats()

        lidar_range = np.asarray(obj.get("canonical_lidar_range", []), dtype=np.float32).reshape(-1)
        lidar_valid = np.asarray(obj.get("canonical_lidar_valid", []), dtype=np.float32).reshape(-1)
        if lidar_range.size != spec.num_sectors or lidar_valid.size != spec.num_sectors:
            continue

        token, _diag = buffers[key].observe(
            lidar_range=lidar_range,
            lidar_valid=lidar_valid,
            is_new_scan=_safe_float(obj.get("lidar_is_new_scan", 0.0), 0.0),
            steps_since_new_scan=_safe_float(obj.get("lidar_steps_since_new_scan", 0.0), 0.0),
        )

        gt_present = bool(_safe_float(obj.get("obstacle_present", 0.0), 0.0) > 0.5 and _safe_float(obj.get("obstacle_longitudinal", 0.0), 0.0) > -0.4)
        gt_long = _safe_float(obj.get("obstacle_longitudinal", 0.0), 0.0)
        gt_lat = _safe_float(obj.get("obstacle_lateral", 0.0), 0.0)
        pred_present = bool(float(token[0]) > 0.5 and float(token[6]) >= float(conf_thresh) and float(token[1]) > -0.4)

        for stats in (overall, by_scene[scene]):
            stats["frames"] += 1
            if gt_present:
                stats["gt_positive"] += 1
            if pred_present:
                stats["pred_positive"] += 1
            if gt_present and pred_present:
                stats["true_positive"] += 1
                stats["rel_long_err"].append(abs(float(token[1]) - gt_long))
                stats["rel_lat_err"].append(abs(float(token[2]) - gt_lat))
                stats["confidence_tp"].append(float(token[6]))
                stats["age_norm_tp"].append(float(token[7]))
            elif pred_present and not gt_present:
                stats["false_positive"] += 1
            elif gt_present and not pred_present:
                stats["false_negative"] += 1

        t_now = _safe_float(obj.get("elapsed_sec", obj.get("timestamp", 0.0)), 0.0)
        prev = prev_truth[key]
        if gt_present and bool(prev.get("valid", False)):
            dt = max(t_now - float(prev.get("time", t_now - control_dt_s)), control_dt_s)
            gt_v_long = float(np.clip((gt_long - float(prev["gt_long"])) / max(dt, 1e-3), -3.0, 3.0))
            gt_v_lat = float(np.clip((gt_lat - float(prev["gt_lat"])) / max(dt, 1e-3), -3.0, 3.0))
            gt_ttc = _gt_ttc(gt_long, gt_v_long, max_ttc_s=6.0)
            if pred_present:
                for stats in (overall, by_scene[scene]):
                    stats["rel_v_long_err"].append(abs(float(token[3]) - gt_v_long))
                    stats["rel_v_lat_err"].append(abs(float(token[4]) - gt_v_lat))
                    stats["ttc_err"].append(abs(float(token[5]) - gt_ttc))
        prev_truth[key] = {
            "valid": bool(gt_present),
            "gt_long": float(gt_long),
            "gt_lat": float(gt_lat),
            "time": float(t_now),
        }

    return {
        "overall": _finalize_stats(overall),
        "by_scene": {scene: _finalize_stats(stats) for scene, stats in sorted(by_scene.items())},
        "settings": {
            "max_range_m": float(max_range_m),
            "near_clip_m": float(near_clip_m),
            "control_dt_s": float(control_dt_s),
            "conf_thresh": float(conf_thresh),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate target-token extraction against sim obstacle truth")
    parser.add_argument("--sim-paths", nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-range-m", type=float, default=6.0)
    parser.add_argument("--near-clip-m", type=float, default=0.18)
    parser.add_argument("--control-dt-s", type=float, default=0.05)
    parser.add_argument("--conf-thresh", type=float, default=0.25)
    args = parser.parse_args()

    result = evaluate(
        paths=list(args.sim_paths),
        max_range_m=float(args.max_range_m),
        near_clip_m=float(args.near_clip_m),
        control_dt_s=float(args.control_dt_s),
        conf_thresh=float(args.conf_thresh),
    )
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
