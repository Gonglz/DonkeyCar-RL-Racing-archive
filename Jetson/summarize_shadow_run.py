#!/usr/bin/env python3
"""Summarize a Jetson V17 deployment runtime monitor CSV."""

import argparse
import csv
import glob
import json
import os
import shutil
import math
import collections
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


def _std(values: List[float], keep_negative: bool = False) -> Optional[float]:
    vals = [v for v in values if v is not None and (keep_negative or v >= 0.0)]
    if not vals:
        return None
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))


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


def _load_notes(paths: List[str], notes_json: Optional[str]) -> Dict[str, object]:
    candidates: List[str] = []
    if notes_json:
        candidates.append(os.path.expanduser(notes_json))
    for path in paths:
        candidates.append(os.path.join(os.path.dirname(path), "run_notes.json"))
    for path in candidates:
        if path and os.path.exists(path):
            with open(path) as f:
                notes = json.load(f)
            notes["_notes_path"] = path
            return notes
    return {}


def _mode_transitions(rows: List[Dict[str, str]], from_modes, to_mode: str) -> int:
    count = 0
    prev = None
    for row in rows:
        mode = (row.get("mode") or "").strip()
        if prev in from_modes and mode == to_mode:
            count += 1
        prev = mode
    return count


def _duration_for_rows(rows: List[Dict[str, str]]) -> Optional[float]:
    elapsed = [
        _to_float(row.get("elapsed_sec")) for row in rows
        if _to_float(row.get("elapsed_sec")) is not None
    ]
    if not elapsed:
        return None
    return max(elapsed) - min(elapsed)


def _match_pct(rows: List[Dict[str, str]]) -> Optional[float]:
    matches = []
    for row in rows:
        pa = _to_float(row.get("pilot_angle"))
        pt = _to_float(row.get("pilot_throttle"))
        fa = _to_float(row.get("final_angle"))
        ft = _to_float(row.get("final_throttle"))
        if None in (pa, pt, fa, ft):
            continue
        matches.append(
            1.0 if abs(pa - fa) <= 0.03 and abs(pt - ft) <= 0.03 else 0.0
        )
    if not matches:
        return None
    return 100.0 * sum(matches) / len(matches)


def summarize(paths: List[str], model_path: Optional[str] = None,
              notes_json: Optional[str] = None) -> Dict[str, object]:
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
    notes = _load_notes(paths, notes_json)

    inferred_duration = None
    if elapsed:
        inferred_duration = max(elapsed) - min(elapsed)

    active_rows = [
        row for row in rows
        if (row.get("mode") or "").strip() in ("local", "local_angle") or
        (row.get("run_pilot") or "").strip().lower() in ("true", "1", "yes")
    ]
    local_rows = [row for row in rows if (row.get("mode") or "").strip() == "local"]
    local_angle_rows = [
        row for row in rows if (row.get("mode") or "").strip() == "local_angle"
    ]
    user_rows = [row for row in rows if (row.get("mode") or "").strip() == "user"]
    mode_counts = dict(collections.Counter((row.get("mode") or "unknown").strip() for row in rows))
    active_duration = _duration_for_rows(active_rows)

    def active_series(name: str, nonnegative: bool = False) -> List[float]:
        vals = []
        for row in active_rows:
            value = _to_float(row.get(name))
            if value is None:
                continue
            if nonnegative and value < 0:
                continue
            vals.append(value)
        return vals

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
        "track_condition": first.get("track_condition") or notes.get("track_condition", ""),
        "run_label": first.get("run_label") or notes.get("run_label", ""),
        "run_duration_sec": _round_or_none(inferred_duration, 3),
        "frames_logged": len(rows),
        "mode_counts": mode_counts,
        "csv_files": paths,
        "inference_latency_ms_p50": _round_or_none(_percentile(latency, 50), 3),
        "inference_latency_ms_p95": _round_or_none(_percentile(latency, 95), 3),
        "effective_fps_mean": _round_or_none(_mean(series("effective_fps", True)), 3),
        "loop_dt_ms_p50": _round_or_none(_percentile(loop_dt, 50), 3),
        "loop_dt_ms_p95": _round_or_none(_percentile(loop_dt, 95), 3),
        "cpu_load_mean": _round_or_none(_mean(series("cpu_load_pct", True)), 3),
        "gpu_load_mean": _round_or_none(_mean(series("gpu_load_pct", True)), 3),
        "mem_used_mb_mean": _round_or_none(_mean(series("mem_used_mb", True)), 3),
        "cpu_temp_mean": _round_or_none(_mean(series("cpu_temp", True)), 3),
        "gpu_temp_mean": _round_or_none(_mean(series("gpu_temp", True)), 3),
        "power_in_mw_mean": _round_or_none(_mean(series("power_in_mw", True)), 3),
        "lidar_frames_last": _round_or_none(lidar_frames[-1] if lidar_frames else None, 0),
        "lidar_valid_points_mean": _round_or_none(_mean(lidar_valid_points, True), 3),
        "lidar_points_total_mean": _round_or_none(_mean(lidar_points_total, True), 3),
        "lidar_nearest_min_m": _round_or_none(min(lidar_nearest) if lidar_nearest else None, 3),
        "lidar_nearest_p50_m": _round_or_none(_percentile(lidar_nearest, 50), 3),
        "lidar_nearest_mean_m": _round_or_none(_mean(lidar_nearest, True), 3),
        "lidar_scan_age_ms_p50": _round_or_none(_percentile(lidar_scan_age, 50), 3),
        "lidar_scan_age_ms_p95": _round_or_none(_percentile(lidar_scan_age, 95), 3),
        "active_duration_sec": _round_or_none(active_duration, 3),
        "active_frames": len(active_rows),
        "full_control_duration_sec": _round_or_none(_duration_for_rows(local_rows), 3),
        "full_control_frames": len(local_rows),
        "steering_only_duration_sec": _round_or_none(_duration_for_rows(local_angle_rows), 3),
        "steering_only_frames": len(local_angle_rows),
        "user_frames": len(user_rows),
        "manual_override_count": _mode_transitions(
            rows, from_modes=("local", "local_angle"), to_mode="user"
        ),
        "active_final_matches_pilot_pct": _round_or_none(_match_pct(active_rows), 3),
        "full_control_final_matches_pilot_pct": _round_or_none(_match_pct(local_rows), 3),
        "steering_only_final_matches_pilot_pct": _round_or_none(_match_pct(local_angle_rows), 3),
        "pilot_angle_mean": _round_or_none(_mean(active_series("pilot_angle"), keep_negative=True), 4),
        "pilot_angle_std": _round_or_none(_std(active_series("pilot_angle"), keep_negative=True), 4),
        "pilot_angle_min": _round_or_none(
            min(active_series("pilot_angle")) if active_series("pilot_angle") else None, 4
        ),
        "pilot_angle_max": _round_or_none(
            max(active_series("pilot_angle")) if active_series("pilot_angle") else None, 4
        ),
        "pilot_throttle_mean": _round_or_none(_mean(active_series("pilot_throttle"), keep_negative=True), 4),
        "pilot_throttle_std": _round_or_none(_std(active_series("pilot_throttle"), keep_negative=True), 4),
        "pilot_throttle_min": _round_or_none(
            min(active_series("pilot_throttle")) if active_series("pilot_throttle") else None, 4
        ),
        "pilot_throttle_max": _round_or_none(
            max(active_series("pilot_throttle")) if active_series("pilot_throttle") else None, 4
        ),
        "run_outcome": notes.get("run_outcome", "unknown"),
        "collision_or_contact": notes.get("collision_or_contact"),
        "stuck_detected": notes.get("stuck_detected"),
        "obstacle_recovery_success": notes.get("obstacle_recovery_success", "unknown"),
        "obstacle_layout": notes.get("obstacle_layout", ""),
        "notes": notes.get("notes", ""),
        "run_notes_file": notes.get("_notes_path", ""),
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
    parser.add_argument("--notes-json", default=None,
                        help="Optional run_notes.json path for outcome labels")
    parser.add_argument("--out", default="jetson_shadow_summary.json",
                        help="Output JSON path")
    parser.add_argument("--log-copy", default=None,
                        help="Optional path to copy the latest CSV as jetson_shadow_log.csv")
    args = parser.parse_args()

    paths = _expand_csvs(args.csv)
    if not paths:
        raise FileNotFoundError("No CSV files matched --csv")
    summary = summarize(paths, model_path=args.model_path,
                        notes_json=args.notes_json)

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
