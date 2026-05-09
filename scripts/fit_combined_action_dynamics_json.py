#!/usr/bin/env python3
"""
Fit a combined action/dynamics sim2real calibration JSON.

This is intentionally low-dimensional and compatible with
module/sim2real_wrapper.py.  It uses:

- action-only Donkey tub roots as a human command envelope prior
- motion Donkey tub roots and/or a 0421 monitor CSV as real response data
- a sim monitor JSONL as the uncalibrated simulator response

The output is a JSON with throttle_gain_ratio, steer_gain_ratio, steer_tau_s,
and throttle_tau_s, plus diagnostics explaining the fit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _clip(value: float, low: float, high: float) -> float:
    return float(max(float(low), min(float(high), float(value))))


def _positive_ratio(num: float, den: float, fallback: float = 1.0) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or den <= 1e-9:
        return float(fallback)
    out = float(num / den)
    return out if math.isfinite(out) and out > 0.0 else float(fallback)


def _stats(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "min": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _read_tub_manifest(root: Path) -> List[Path]:
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing tub manifest: {manifest}")
    lines = manifest.read_text(encoding="utf-8").splitlines()
    if len(lines) < 5:
        raise ValueError(f"manifest too short: {manifest}")
    meta = json.loads(lines[4])
    paths = [root / str(p) for p in meta.get("paths", [])]
    return [p for p in paths if p.is_file() and p.stat().st_size > 0]


def _iter_tub_rows(root: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for catalog_path in _read_tub_manifest(root):
        with catalog_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield str(catalog_path.name), row


def _collect_tub_stats(root: Path, min_dt: float, max_dt: float) -> Dict[str, Any]:
    angles: List[float] = []
    throttles: List[float] = []
    angle_rates: List[float] = []
    throttle_rates: List[float] = []
    dt_vals: List[float] = []
    speed_odom: List[float] = []
    speed_enc: List[float] = []
    speed_delta: List[float] = []
    gyro_z_abs: List[float] = []
    heading_rate_abs: List[float] = []

    total_rows = 0
    rows_with_motion = 0
    prev: Optional[Tuple[str, str, float, float, float]] = None

    for catalog_name, row in _iter_tub_rows(root):
        total_rows += 1
        session_id = str(row.get("_session_id", "unknown"))
        timestamp_s = _safe_float(row.get("_timestamp_ms"), 0.0) / 1000.0
        angle = _clip(_safe_float(row.get("user/angle"), 0.0), -1.0, 1.0)
        throttle = _clip(_safe_float(row.get("user/throttle"), 0.0), -1.0, 1.0)
        angles.append(abs(angle))
        throttles.append(abs(throttle))

        has_motion = "rp2040/speed_odom" in row or "rp2040/delta_x" in row
        if has_motion:
            rows_with_motion += 1
            speed_odom.append(abs(_safe_float(row.get("rp2040/speed_odom"), 0.0)))
            speed_enc.append(abs(_safe_float(row.get("rp2040/speed_enc"), 0.0)))
            gyro_z_abs.append(abs(_safe_float(row.get("rp2040/gyro_z"), 0.0)))
            heading_rate_abs.append(abs(_safe_float(row.get("rp2040/heading_rate_deg"), 0.0)))

        if prev is not None:
            prev_catalog, prev_session, prev_t, prev_angle, prev_throttle = prev
            same_series = (prev_catalog == catalog_name) and (prev_session == session_id)
            dt = timestamp_s - prev_t
            if same_series and min_dt <= dt <= max_dt:
                dt_vals.append(float(dt))
                angle_rates.append(abs(angle - prev_angle) / max(dt, 1e-6))
                throttle_rates.append(abs(throttle - prev_throttle) / max(dt, 1e-6))
                if has_motion:
                    dx = _safe_float(row.get("rp2040/delta_x"), 0.0)
                    dy = _safe_float(row.get("rp2040/delta_y"), 0.0)
                    if abs(dx) > 0.0 or abs(dy) > 0.0:
                        speed_delta.append(float(math.hypot(dx, dy) / max(dt, 1e-6)))
        prev = (catalog_name, session_id, timestamp_s, angle, throttle)

    return {
        "root": str(root),
        "total_rows": int(total_rows),
        "rows_with_motion": int(rows_with_motion),
        "dt_s": _stats(dt_vals),
        "abs_angle": _stats(angles),
        "abs_throttle": _stats(throttles),
        "angle_rate_abs_per_s": _stats(angle_rates),
        "throttle_rate_abs_per_s": _stats(throttle_rates),
        "speed_odom": _stats(speed_odom),
        "speed_enc": _stats(speed_enc),
        "speed_from_delta": _stats(speed_delta),
        "gyro_z_abs": _stats(gyro_z_abs),
        "heading_rate_abs_deg_s": _stats(heading_rate_abs),
    }


def _collect_monitor_csv_stats(path: Path) -> Dict[str, Any]:
    dt_vals: List[float] = []
    speed_proxy: List[float] = []
    angles: List[float] = []
    throttles: List[float] = []
    angle_rates: List[float] = []
    throttle_rates: List[float] = []

    t = 0.0
    last_angle: Optional[float] = None
    last_throttle: Optional[float] = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("recording", "")).strip().lower() not in ("1", "true", "yes"):
                continue
            dt = _safe_float(row.get("loop_dt_ms"), 0.0) / 1000.0
            if dt <= 1e-6 or not math.isfinite(dt):
                continue
            t += dt
            dx = _safe_float(row.get("delta_x"), 0.0)
            dy = _safe_float(row.get("delta_y"), 0.0)
            angle = _clip(_safe_float(row.get("final_angle"), 0.0), -1.0, 1.0)
            throttle = _clip(_safe_float(row.get("final_throttle"), 0.0), -1.0, 1.0)
            dt_vals.append(float(dt))
            speed_proxy.append(float(math.hypot(dx, dy) / max(dt, 1e-6)))
            angles.append(abs(angle))
            throttles.append(abs(throttle))
            if last_angle is not None:
                angle_rates.append(abs(angle - last_angle) / max(dt, 1e-6))
            if last_throttle is not None:
                throttle_rates.append(abs(throttle - last_throttle) / max(dt, 1e-6))
            last_angle = angle
            last_throttle = throttle

    return {
        "path": str(path),
        "dt_s": _stats(dt_vals),
        "speed_proxy": _stats(speed_proxy),
        "abs_angle": _stats(angles),
        "abs_throttle": _stats(throttles),
        "angle_rate_abs_per_s": _stats(angle_rates),
        "throttle_rate_abs_per_s": _stats(throttle_rates),
    }


def _collect_sim_jsonl_stats(path: Path) -> Dict[str, Any]:
    speeds: List[float] = []
    final_angles: List[float] = []
    final_throttles: List[float] = []
    raw_angles: List[float] = []
    raw_throttles: List[float] = []
    dt_vals: List[float] = []
    final_angle_rates: List[float] = []
    final_throttle_rates: List[float] = []
    raw_angle_rates: List[float] = []
    raw_throttle_rates: List[float] = []

    last_t: Optional[float] = None
    last_final_angle: Optional[float] = None
    last_final_throttle: Optional[float] = None
    last_raw_angle: Optional[float] = None
    last_raw_throttle: Optional[float] = None
    sim2real_rows = 0
    total_rows = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            total_rows += 1
            t = _safe_float(row.get("elapsed_sec", row.get("timestamp")), 0.0)
            speed = abs(_safe_float(row.get("speed"), 0.0))
            final_angle = _clip(_safe_float(row.get("final_angle"), 0.0), -1.0, 1.0)
            final_throttle = _clip(_safe_float(row.get("final_throttle"), 0.0), -1.0, 1.0)
            raw_angle = _clip(_safe_float(row.get("pre_sim2real_final_angle", final_angle), 0.0), -1.0, 1.0)
            raw_throttle = _clip(_safe_float(row.get("pre_sim2real_final_throttle", final_throttle), 0.0), -1.0, 1.0)
            if bool(row.get("sim2real_applied", False)):
                sim2real_rows += 1

            speeds.append(speed)
            final_angles.append(abs(final_angle))
            final_throttles.append(abs(final_throttle))
            raw_angles.append(abs(raw_angle))
            raw_throttles.append(abs(raw_throttle))

            if last_t is not None:
                dt = t - last_t
                if 1e-6 < dt < 1.0:
                    dt_vals.append(float(dt))
                    if last_final_angle is not None:
                        final_angle_rates.append(abs(final_angle - last_final_angle) / max(dt, 1e-6))
                    if last_final_throttle is not None:
                        final_throttle_rates.append(abs(final_throttle - last_final_throttle) / max(dt, 1e-6))
                    if last_raw_angle is not None:
                        raw_angle_rates.append(abs(raw_angle - last_raw_angle) / max(dt, 1e-6))
                    if last_raw_throttle is not None:
                        raw_throttle_rates.append(abs(raw_throttle - last_raw_throttle) / max(dt, 1e-6))
            last_t = t
            last_final_angle = final_angle
            last_final_throttle = final_throttle
            last_raw_angle = raw_angle
            last_raw_throttle = raw_throttle

    return {
        "path": str(path),
        "total_rows": int(total_rows),
        "sim2real_applied_rows": int(sim2real_rows),
        "dt_s": _stats(dt_vals),
        "speed": _stats(speeds),
        "abs_final_angle": _stats(final_angles),
        "abs_final_throttle": _stats(final_throttles),
        "abs_raw_angle": _stats(raw_angles),
        "abs_raw_throttle": _stats(raw_throttles),
        "final_angle_rate_abs_per_s": _stats(final_angle_rates),
        "final_throttle_rate_abs_per_s": _stats(final_throttle_rates),
        "raw_angle_rate_abs_per_s": _stats(raw_angle_rates),
        "raw_throttle_rate_abs_per_s": _stats(raw_throttle_rates),
    }


def _merge_stat_max(stats: Sequence[Dict[str, Any]], key_path: Tuple[str, ...], field: str) -> float:
    vals: List[float] = []
    for item in stats:
        cur: Any = item
        for key in key_path:
            cur = cur.get(key, {}) if isinstance(cur, dict) else {}
        if isinstance(cur, dict) and int(cur.get("n", 0) or 0) > 0:
            vals.append(_safe_float(cur.get(field), 0.0))
    vals = [v for v in vals if math.isfinite(v) and v > 0.0]
    return float(max(vals) if vals else 0.0)


def _merge_stat_min(stats: Sequence[Dict[str, Any]], key_path: Tuple[str, ...], field: str) -> float:
    vals: List[float] = []
    for item in stats:
        cur: Any = item
        for key in key_path:
            cur = cur.get(key, {}) if isinstance(cur, dict) else {}
        if isinstance(cur, dict) and int(cur.get("n", 0) or 0) > 0:
            vals.append(_safe_float(cur.get(field), 0.0))
    vals = [v for v in vals if math.isfinite(v)]
    return float(min(vals) if vals else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit combined tub + 0421 sim2real dynamics JSON")
    parser.add_argument("--action-root", action="append", default=[], help="action-only or action tub root; repeatable")
    parser.add_argument("--motion-root", action="append", default=[], help="real motion tub root; repeatable")
    parser.add_argument("--real-monitor-csv", action="append", default=[], help="real 0421 monitor CSV; repeatable")
    parser.add_argument("--sim-jsonl", required=True, help="sim monitor JSONL used as uncalibrated simulator response")
    parser.add_argument("--base-json", default="", help="optional previous JSON; used to preserve stricter lag")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--report-json", default="")
    parser.add_argument("--source", default="combined_action_dynamics_v1")
    parser.add_argument("--min-dt", type=float, default=0.02)
    parser.add_argument("--max-dt", type=float, default=0.20)
    parser.add_argument("--min-throttle-gain", type=float, default=0.05)
    parser.add_argument("--max-throttle-gain", type=float, default=0.60)
    parser.add_argument(
        "--throttle-boost",
        type=float,
        default=1.0,
        help="Optional multiplicative training boost applied after the data fit.",
    )
    parser.add_argument("--min-steer-gain", type=float, default=0.50)
    parser.add_argument("--max-steer-gain", type=float, default=3.00)
    parser.add_argument("--max-tau-s", type=float, default=0.35)
    parser.add_argument("--preserve-base-lag", action="store_true", default=True)
    args = parser.parse_args()

    action_roots = [Path(p).expanduser().resolve() for p in args.action_root]
    motion_roots = [Path(p).expanduser().resolve() for p in args.motion_root]
    monitor_csvs = [Path(p).expanduser().resolve() for p in args.real_monitor_csv]
    sim_jsonl = Path(args.sim_jsonl).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    report_json = Path(args.report_json).expanduser().resolve() if args.report_json else output_json.with_name(output_json.stem + "_report.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    action_stats = [_collect_tub_stats(root, min_dt=args.min_dt, max_dt=args.max_dt) for root in action_roots]
    motion_stats = [_collect_tub_stats(root, min_dt=args.min_dt, max_dt=args.max_dt) for root in motion_roots]
    monitor_stats = [_collect_monitor_csv_stats(path) for path in monitor_csvs]
    sim_stats = _collect_sim_jsonl_stats(sim_jsonl)

    real_speed_p95 = max(
        _merge_stat_max(motion_stats, ("speed_from_delta",), "p95"),
        _merge_stat_max(motion_stats, ("speed_odom",), "p95"),
        _merge_stat_max(monitor_stats, ("speed_proxy",), "p95"),
    )
    real_speed_p99 = max(
        _merge_stat_max(motion_stats, ("speed_from_delta",), "p99"),
        _merge_stat_max(motion_stats, ("speed_odom",), "p99"),
        _merge_stat_max(monitor_stats, ("speed_proxy",), "p99"),
    )
    real_speed_max = max(
        _merge_stat_max(motion_stats, ("speed_from_delta",), "max"),
        _merge_stat_max(motion_stats, ("speed_odom",), "max"),
        _merge_stat_max(monitor_stats, ("speed_proxy",), "max"),
    )
    real_speed_min = min(
        _merge_stat_min(motion_stats, ("speed_from_delta",), "min"),
        _merge_stat_min(motion_stats, ("speed_odom",), "min"),
        _merge_stat_min(monitor_stats, ("speed_proxy",), "min"),
    )
    real_speed_min = max(0.0, real_speed_min)
    sim_speed_min = _safe_float(sim_stats["speed"].get("min"), 0.0)
    sim_speed_p95 = _safe_float(sim_stats["speed"].get("p95"), 0.0)
    sim_speed_p99 = _safe_float(sim_stats["speed"].get("p99"), 0.0)
    sim_speed_max = _safe_float(sim_stats["speed"].get("max"), 0.0)

    speed_gain_p95 = _positive_ratio(real_speed_p95, sim_speed_p95)
    speed_gain_p99 = _positive_ratio(real_speed_p99, sim_speed_p99)
    speed_gain_max = _positive_ratio(real_speed_max, sim_speed_max)
    throttle_gain_ratio = _clip(
        math.sqrt(max(speed_gain_p99, 1e-9) * max(speed_gain_max, 1e-9)),
        args.min_throttle_gain,
        args.max_throttle_gain,
    )
    fitted_throttle_gain_ratio = float(throttle_gain_ratio)
    throttle_gain_ratio = _clip(
        fitted_throttle_gain_ratio * max(0.01, float(args.throttle_boost)),
        args.min_throttle_gain,
        args.max_throttle_gain,
    )

    target_abs_angle_p95 = max(
        _merge_stat_max(action_stats, ("abs_angle",), "p95"),
        _merge_stat_max(motion_stats, ("abs_angle",), "p95"),
        _merge_stat_max(monitor_stats, ("abs_angle",), "p95"),
    )
    target_abs_angle_p95 = _clip(target_abs_angle_p95, 0.05, 1.0)
    sim_raw_angle_p95 = _safe_float(sim_stats["abs_raw_angle"].get("p95"), 0.0)
    if sim_raw_angle_p95 <= 1e-6:
        sim_raw_angle_p95 = _safe_float(sim_stats["abs_final_angle"].get("p95"), 1.0)
    steer_gain_ratio = _clip(
        _positive_ratio(target_abs_angle_p95, sim_raw_angle_p95),
        args.min_steer_gain,
        args.max_steer_gain,
    )

    real_angle_rate_p95 = max(
        _merge_stat_max(monitor_stats, ("angle_rate_abs_per_s",), "p95"),
        1e-6,
    )
    real_throttle_rate_p95 = max(
        _merge_stat_max(monitor_stats, ("throttle_rate_abs_per_s",), "p95"),
        1e-6,
    )
    sim_angle_rate_p95 = max(_safe_float(sim_stats["raw_angle_rate_abs_per_s"].get("p95"), 0.0), 1e-6)
    sim_throttle_rate_p95 = max(_safe_float(sim_stats["raw_throttle_rate_abs_per_s"].get("p95"), 0.0), 1e-6)
    dt_ref = min(
        _merge_stat_max(monitor_stats, ("dt_s",), "p50") or 0.05,
        _safe_float(sim_stats["dt_s"].get("p50"), 0.05) or 0.05,
    )
    steer_rate_ratio = _positive_ratio(sim_angle_rate_p95, real_angle_rate_p95)
    throttle_rate_ratio = _positive_ratio(sim_throttle_rate_p95, real_throttle_rate_p95)
    steer_tau_s = 0.0 if steer_rate_ratio <= 1.0 else _clip(dt_ref * math.log(steer_rate_ratio), 0.0, args.max_tau_s)
    throttle_tau_s = 0.0 if throttle_rate_ratio <= 1.0 else _clip(dt_ref * math.log(throttle_rate_ratio), 0.0, args.max_tau_s)

    base_payload: Dict[str, Any] = {}
    if args.base_json:
        base_path = Path(args.base_json).expanduser().resolve()
        if base_path.is_file():
            base_payload = json.loads(base_path.read_text(encoding="utf-8"))
    if args.preserve_base_lag and base_payload:
        steer_tau_s = max(steer_tau_s, _safe_float(base_payload.get("steer_tau_s"), 0.0))
        throttle_tau_s = max(throttle_tau_s, _safe_float(base_payload.get("throttle_tau_s"), 0.0))
        steer_tau_s = _clip(steer_tau_s, 0.0, args.max_tau_s)
        throttle_tau_s = _clip(throttle_tau_s, 0.0, args.max_tau_s)

    predicted_after_fit = {
        "speed_min": float(sim_speed_min * throttle_gain_ratio),
        "speed_p95": float(sim_speed_p95 * throttle_gain_ratio),
        "speed_p99": float(sim_speed_p99 * throttle_gain_ratio),
        "speed_max": float(sim_speed_max * throttle_gain_ratio),
        "abs_angle_p95": float(min(1.0, sim_raw_angle_p95 * steer_gain_ratio)),
    }

    payload = {
        "source": str(args.source),
        "calibrated_at": datetime.now().isoformat(),
        "action_roots": [str(p) for p in action_roots],
        "motion_roots": [str(p) for p in motion_roots],
        "real_monitor_csvs": [str(p) for p in monitor_csvs],
        "sim_jsonl": str(sim_jsonl),
        "base_json": str(Path(args.base_json).expanduser().resolve()) if args.base_json else None,
        "fit_method": {
            "throttle_gain_ratio": "sqrt((real_speed_p99/sim_speed_p99) * (real_speed_max/sim_speed_max)); speed extrema oriented",
            "steer_gain_ratio": "combined real/action abs(angle)_p95 / sim raw abs(angle)_p95",
            "steer_tau_s": "rate-match sim raw angle p95 to 0421 monitor final_angle p95; preserve base lag if stricter",
            "throttle_tau_s": "rate-match sim raw throttle p95 to 0421 monitor final_throttle p95; preserve base lag if stricter",
        },
        "throttle_boost": float(args.throttle_boost),
        "fitted_throttle_gain_ratio": float(fitted_throttle_gain_ratio),
        "throttle_gain_ratio": float(throttle_gain_ratio),
        "steer_gain_ratio": float(steer_gain_ratio),
        "steer_tau_s": float(steer_tau_s),
        "throttle_tau_s": float(throttle_tau_s),
        "speed_gain_p95_raw": float(speed_gain_p95),
        "speed_gain_p99_raw": float(speed_gain_p99),
        "speed_gain_max_raw": float(speed_gain_max),
        "steer_gain_raw": float(_positive_ratio(target_abs_angle_p95, sim_raw_angle_p95)),
        "steer_rate_ratio_raw": float(steer_rate_ratio),
        "throttle_rate_ratio_raw": float(throttle_rate_ratio),
        "real_speed_target": {
            "min": float(real_speed_min),
            "p95": float(real_speed_p95),
            "p99": float(real_speed_p99),
            "max": float(real_speed_max),
        },
        "sim_speed_source": {
            "min": float(sim_speed_min),
            "p95": float(sim_speed_p95),
            "p99": float(sim_speed_p99),
            "max": float(sim_speed_max),
        },
        "predicted_after_fit": predicted_after_fit,
    }
    report = {
        "calibration": payload,
        "action_stats": action_stats,
        "motion_stats": motion_stats,
        "monitor_stats": monitor_stats,
        "sim_stats": sim_stats,
        "base_payload": base_payload,
    }

    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[saved] output_json={output_json}")
    print(f"[saved] report_json={report_json}")


if __name__ == "__main__":
    main()
