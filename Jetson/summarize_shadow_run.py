#!/usr/bin/env python3
"""Summarize a Jetson V17 shadow-mode runtime monitor CSV."""

import argparse
import csv
import glob
import json
import os
import shutil
from typing import Dict, Iterable, List, Optional


def _to_float(value) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        if out != out:
            return None
        return out
    except Exception:
        return None


def _percentile(values: List[float], pct: float) -> Optional[float]:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * (float(pct) / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _mean(values: List[float], keep_negative: bool = False) -> Optional[float]:
    vals = [v for v in values if v is not None and (keep_negative or v >= 0.0)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _max(values: List[float], keep_negative: bool = False) -> Optional[float]:
    vals = [v for v in values if v is not None and (keep_negative or v >= 0.0)]
    if not vals:
        return None
    return max(vals)


def _round_or_none(value: Optional[float], digits: int = 3):
    return None if value is None else round(float(value), digits)


def _expand_csvs(patterns: Iterable[str]) -> List[str]:
    paths: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(os.path.expanduser(pattern)))
        if matches:
            paths.extend(matches)
        elif os.path.exists(os.path.expanduser(pattern)):
            paths.append(os.path.expanduser(pattern))
    return paths


def _read_rows(paths: List[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_csv_path"] = path
                rows.append(row)
    return rows


def summarize(paths: List[str], model_path: Optional[str] = None) -> Dict[str, object]:
    rows = _read_rows(paths)
    if not rows:
        raise ValueError("No rows found in CSV input")

    first = rows[0]
    last = rows[-1]

    def series(name: str, nonnegative: bool = False) -> List[float]:
        vals = []
        for row in rows:
            value = _to_float(row.get(name))
            if value is None:
                continue
            if nonnegative and value < 0:
                continue
            vals.append(value)
        return vals

    latency = series("pilot_inference_latency_ms", nonnegative=True)
    if not latency:
        latency = series("inference_latency_ms", nonnegative=True)
    loop_dt = series("loop_dt_ms", nonnegative=True)
    elapsed = series("elapsed_sec", nonnegative=True)
    lidar_frames = series("lidar_frames", nonnegative=True)
    lidar_valid_points = series("lidar_valid_points", nonnegative=True)
    lidar_points_total = series("lidar_points_total", nonnegative=True)
    lidar_nearest = series("lidar_nearest_min", nonnegative=True)
    lidar_scan_age = series("lidar_scan_age_ms", nonnegative=True)
    safety_inference_timeout_count = series("safety_inference_timeout_count", nonnegative=True)
    safety_lidar_missing_count = series("safety_lidar_missing_count", nonnegative=True)
    safety_lidar_stale_count = series("safety_lidar_stale_count", nonnegative=True)
    safety_rp2040_missing_count = series("safety_rp2040_missing_count", nonnegative=True)
    safety_lidar_age = series("safety_last_lidar_age_ms", nonnegative=True)
    safety_rp2040_age = series("safety_last_rp2040_age_ms", nonnegative=True)

    def any_true(name: str) -> bool:
        for row in rows:
            value = str(row.get(name, "")).strip().lower()
            if value in ("true", "1", "yes"):
                return True
        return False

    def last_text(name: str) -> str:
        for row in reversed(rows):
            value = str(row.get(name, "")).strip()
            if value:
                return value
        return ""

    inferred_duration = None
    if elapsed:
        inferred_duration = max(elapsed) - min(elapsed)

    selected_model_path = model_path or first.get("model_path") or ""
    selected_model_path = os.path.expanduser(selected_model_path)
    model_size = _to_float(first.get("model_size_MB"))
    if selected_model_path and os.path.exists(selected_model_path):
        model_size = os.path.getsize(selected_model_path) / (1024.0 * 1024.0)

    summary = {
        "model_name": first.get("model_name") or (
            os.path.basename(selected_model_path) if selected_model_path else ""
        ),
        "model_path": selected_model_path,
        "model_size_MB": _round_or_none(model_size, 3),
        "backend": first.get("backend") or "PyTorch/SB3 RecurrentPPO",
        "input_resolution": first.get("input_resolution") or "",
        "input_modalities": first.get("input_modalities") or "",
        "policy_chain": first.get("policy_chain") or "",
        "control_mode": first.get("control_mode") or "shadow",
        "run_duration_sec": _round_or_none(inferred_duration, 3),
        "frames_logged": len(rows),
        "csv_files": paths,
        "inference_latency_ms_p50": _round_or_none(_percentile(latency, 50), 3),
        "inference_latency_ms_p95": _round_or_none(_percentile(latency, 95), 3),
        "inference_latency_ms_p99": _round_or_none(_percentile(latency, 99), 3),
        "inference_latency_ms_max": _round_or_none(_max(latency), 3),
        "effective_fps_mean": _round_or_none(_mean(series("effective_fps", True)), 3),
        "loop_dt_ms_p50": _round_or_none(_percentile(loop_dt, 50), 3),
        "loop_dt_ms_p95": _round_or_none(_percentile(loop_dt, 95), 3),
        "loop_dt_ms_p99": _round_or_none(_percentile(loop_dt, 99), 3),
        "loop_dt_ms_max": _round_or_none(_max(loop_dt), 3),
        "cpu_load_mean": _round_or_none(_mean(series("cpu_load_pct", True)), 3),
        "gpu_load_mean": _round_or_none(_mean(series("gpu_load_pct", True)), 3),
        "mem_used_mb_mean": _round_or_none(_mean(series("mem_used_mb", True)), 3),
        "cpu_temp_mean": _round_or_none(_mean(series("cpu_temp", True)), 3),
        "gpu_temp_mean": _round_or_none(_mean(series("gpu_temp", True)), 3),
        "pmic_temp_mean": _round_or_none(_mean(series("pmic_temp", True)), 3),
        "pmic_temp_max": _round_or_none(_max(series("pmic_temp", True), True), 3),
        "power_in_mw_mean": _round_or_none(_mean(series("power_in_mw", True)), 3),
        "lidar_frames_last": _round_or_none(lidar_frames[-1] if lidar_frames else None, 0),
        "lidar_valid_points_mean": _round_or_none(_mean(lidar_valid_points, True), 3),
        "lidar_points_total_mean": _round_or_none(_mean(lidar_points_total, True), 3),
        "lidar_nearest_min_m": _round_or_none(min(lidar_nearest) if lidar_nearest else None, 3),
        "lidar_nearest_mean_m": _round_or_none(_mean(lidar_nearest, True), 3),
        "lidar_scan_age_ms_p50": _round_or_none(_percentile(lidar_scan_age, 50), 3),
        "lidar_scan_age_ms_p95": _round_or_none(_percentile(lidar_scan_age, 95), 3),
        "lidar_scan_age_ms_p99": _round_or_none(_percentile(lidar_scan_age, 99), 3),
        "lidar_scan_age_ms_max": _round_or_none(_max(lidar_scan_age), 3),
        "safety_blocked": any_true("safety_blocked"),
        "safety_block_reason_last": last_text("safety_block_reason"),
        "inference_timeout_count": int(_max(safety_inference_timeout_count) or 0),
        "lidar_missing_count": int(_max(safety_lidar_missing_count) or 0),
        "lidar_stale_count": int(_max(safety_lidar_stale_count) or 0),
        "rp2040_missing_count": int(_max(safety_rp2040_missing_count) or 0),
        "safety_lidar_age_ms_max": _round_or_none(_max(safety_lidar_age), 3),
        "safety_rp2040_age_ms_max": _round_or_none(_max(safety_rp2040_age), 3),
        "first_timestamp": first.get("timestamp"),
        "last_timestamp": last.get("timestamp"),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", required=True,
                        help="CSV path(s) or glob(s), e.g. monitor_logs/run_*.csv")
    parser.add_argument("--model-path", default=None,
                        help="Optional model path override for size/name metadata")
    parser.add_argument("--out", default="jetson_shadow_summary.json",
                        help="Output JSON path")
    parser.add_argument("--log-copy", default=None,
                        help="Optional path to copy the latest CSV as jetson_shadow_log.csv")
    args = parser.parse_args()

    paths = _expand_csvs(args.csv)
    if not paths:
        raise FileNotFoundError("No CSV files matched --csv")
    summary = summarize(paths, model_path=args.model_path)

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"saved: {out}")

    if args.log_copy:
        latest = max(paths, key=lambda p: os.path.getmtime(p))
        dst = os.path.expanduser(args.log_copy)
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.copy2(latest, dst)
        print(f"copied latest csv: {latest} -> {dst}")


if __name__ == "__main__":
    main()
