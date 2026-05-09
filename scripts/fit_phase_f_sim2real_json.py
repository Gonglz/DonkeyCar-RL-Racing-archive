#!/usr/bin/env python3
"""
Fit a first-pass low-dimensional sim2real action calibration JSON for Phase F.

This script is intentionally heuristic and low-dimensional. It does not try to
identify full vehicle dynamics. Instead it matches the motion-profile features
used by Gate B:

- speed proxy p50 / p95
- abs(final_angle) p95
- action slew-rate p95 for steering / throttle

The output JSON is compatible with module/sim2real_wrapper.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=np.float32), float(q)))


def _median(values: Sequence[float], default: float = 0.05) -> float:
    if not values:
        return float(default)
    return float(np.median(np.asarray(values, dtype=np.float32)))


def _iter_real_records(csv_path: Path) -> Iterable[Tuple[float, float, float, float]]:
    current_t = 0.0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not _parse_bool(row.get("recording", False)):
                continue
            dt_s = _safe_float(row.get("loop_dt_ms", 0.0), 0.0) / 1000.0
            if dt_s <= 1e-6:
                continue
            dx = _safe_float(row.get("delta_x", 0.0), 0.0)
            dy = _safe_float(row.get("delta_y", 0.0), 0.0)
            speed_proxy = math.sqrt(dx * dx + dy * dy) / dt_s
            final_angle = _safe_float(row.get("final_angle", 0.0), 0.0)
            final_throttle = _safe_float(row.get("final_throttle", 0.0), 0.0)
            current_t += dt_s
            if not math.isfinite(speed_proxy):
                continue
            yield current_t, float(speed_proxy), float(final_angle), float(final_throttle)


def _iter_sim_records(jsonl_path: Path) -> Iterable[Tuple[float, float, float, float]]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            t = _safe_float(row.get("elapsed_sec", row.get("timestamp", 0.0)), 0.0)
            speed = _safe_float(row.get("speed", 0.0), 0.0)
            final_angle = _safe_float(row.get("final_angle", 0.0), 0.0)
            final_throttle = _safe_float(row.get("final_throttle", 0.0), 0.0)
            if not math.isfinite(speed):
                continue
            yield float(t), float(speed), float(final_angle), float(final_throttle)


def _load_motion_series(path: Path, kind: str) -> Dict[str, Any]:
    iterator = _iter_real_records(path) if kind == "real" else _iter_sim_records(path)
    t_vals: List[float] = []
    speed_vals: List[float] = []
    angle_vals: List[float] = []
    throttle_vals: List[float] = []
    angle_rate_vals: List[float] = []
    throttle_rate_vals: List[float] = []
    dt_vals: List[float] = []

    last_t: float | None = None
    last_angle: float | None = None
    last_throttle: float | None = None

    for t, speed, angle, throttle in iterator:
        t_vals.append(float(t))
        speed_vals.append(float(speed))
        angle_vals.append(float(angle))
        throttle_vals.append(float(throttle))
        if last_t is not None:
            dt = float(t - last_t)
            if dt > 1e-6 and math.isfinite(dt):
                dt_vals.append(dt)
                if last_angle is not None:
                    angle_rate_vals.append(abs((float(angle) - float(last_angle)) / dt))
                if last_throttle is not None:
                    throttle_rate_vals.append(abs((float(throttle) - float(last_throttle)) / dt))
        last_t = float(t)
        last_angle = float(angle)
        last_throttle = float(throttle)

    return {
        "usable_rows": int(len(t_vals)),
        "dt_median": _median(dt_vals, default=0.05),
        "speed_proxy_p50": _quantile(speed_vals, 0.50),
        "speed_proxy_p95": _quantile(speed_vals, 0.95),
        "speed_proxy_p99": _quantile(speed_vals, 0.99),
        "speed_proxy_max": float(max(speed_vals) if speed_vals else 0.0),
        "abs_final_angle_p95": _quantile([abs(x) for x in angle_vals], 0.95),
        "abs_final_angle_p99": _quantile([abs(x) for x in angle_vals], 0.99),
        "abs_final_angle_max": float(max((abs(x) for x in angle_vals), default=0.0)),
        "abs_final_throttle_p95": _quantile([abs(x) for x in throttle_vals], 0.95),
        "abs_final_throttle_p99": _quantile([abs(x) for x in throttle_vals], 0.99),
        "abs_final_throttle_max": float(max((abs(x) for x in throttle_vals), default=0.0)),
        "angle_rate_p95": _quantile(angle_rate_vals, 0.95),
        "throttle_rate_p95": _quantile(throttle_rate_vals, 0.95),
    }


def _clip(value: float, low: float, high: float) -> float:
    return float(max(float(low), min(float(high), float(value))))


def _positive_ratio(numerator: float, denominator: float, fallback: float = 1.0) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 1e-6:
        return float(fallback)
    ratio = float(numerator / denominator)
    return float(ratio) if math.isfinite(ratio) and ratio > 0.0 else float(fallback)


def _derive_params(
    *,
    real_stats: Dict[str, Any],
    sim_stats: Dict[str, Any],
    fit_mode: str,
    min_throttle_gain: float,
    max_throttle_gain: float,
    min_steer_gain: float,
    max_steer_gain: float,
    max_tau_s: float,
) -> Dict[str, Any]:
    if str(fit_mode) == "extrema_refine":
        speed_gain_p99 = _positive_ratio(real_stats["speed_proxy_p99"], sim_stats["speed_proxy_p99"], fallback=1.0)
        speed_gain_max = _positive_ratio(real_stats["speed_proxy_max"], sim_stats["speed_proxy_max"], fallback=1.0)
        throttle_gain_ratio = _clip(
            math.sqrt(max(speed_gain_p99, 1e-6) * max(speed_gain_max, 1e-6)),
            min_throttle_gain,
            max_throttle_gain,
        )
        predicted = {
            "speed_proxy_p99": float(sim_stats["speed_proxy_p99"] * throttle_gain_ratio),
            "speed_proxy_max": float(sim_stats["speed_proxy_max"] * throttle_gain_ratio),
            "abs_final_angle_p95": float(sim_stats["abs_final_angle_p95"]),
        }
        return {
            "throttle_gain_ratio": float(throttle_gain_ratio),
            "steer_gain_ratio": 1.0,
            "steer_tau_s": 0.0,
            "throttle_tau_s": 0.0,
            "speed_gain_p99_raw": float(speed_gain_p99),
            "speed_gain_max_raw": float(speed_gain_max),
            "predicted_after_fit": predicted,
        }

    speed_gain_p50 = _positive_ratio(real_stats["speed_proxy_p50"], sim_stats["speed_proxy_p50"], fallback=1.0)
    speed_gain_p95 = _positive_ratio(real_stats["speed_proxy_p95"], sim_stats["speed_proxy_p95"], fallback=1.0)
    throttle_gain_ratio = _clip(
        math.sqrt(max(speed_gain_p50, 1e-6) * max(speed_gain_p95, 1e-6)),
        min_throttle_gain,
        max_throttle_gain,
    )

    raw_steer_gain = _positive_ratio(real_stats["abs_final_angle_p95"], sim_stats["abs_final_angle_p95"], fallback=1.0)
    steer_gain_ratio = _clip(raw_steer_gain, min_steer_gain, max_steer_gain)

    dt_ref = min(
        float(real_stats.get("dt_median", 0.05) or 0.05),
        float(sim_stats.get("dt_median", 0.05) or 0.05),
    )
    angle_rate_ratio = _positive_ratio(sim_stats["angle_rate_p95"], real_stats["angle_rate_p95"], fallback=1.0)
    throttle_rate_ratio = _positive_ratio(sim_stats["throttle_rate_p95"], real_stats["throttle_rate_p95"], fallback=1.0)

    steer_tau_s = 0.0 if angle_rate_ratio <= 1.0 else _clip(dt_ref * math.log(angle_rate_ratio), 0.0, max_tau_s)
    throttle_tau_s = 0.0 if throttle_rate_ratio <= 1.0 else _clip(dt_ref * math.log(throttle_rate_ratio), 0.0, max_tau_s)

    predicted = {
        "speed_proxy_p50": float(sim_stats["speed_proxy_p50"] * throttle_gain_ratio),
        "speed_proxy_p95": float(sim_stats["speed_proxy_p95"] * throttle_gain_ratio),
        "abs_final_angle_p95": float(min(1.0, sim_stats["abs_final_angle_p95"] * steer_gain_ratio)),
    }
    return {
        "throttle_gain_ratio": float(throttle_gain_ratio),
        "steer_gain_ratio": float(steer_gain_ratio),
        "steer_tau_s": float(steer_tau_s),
        "throttle_tau_s": float(throttle_tau_s),
        "speed_gain_p50_raw": float(speed_gain_p50),
        "speed_gain_p95_raw": float(speed_gain_p95),
        "steer_gain_raw": float(raw_steer_gain),
        "angle_rate_ratio_raw": float(angle_rate_ratio),
        "throttle_rate_ratio_raw": float(throttle_rate_ratio),
        "predicted_after_fit": predicted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a first-pass Phase F sim2real action JSON")
    parser.add_argument("--real-csv", type=str, required=True)
    parser.add_argument("--sim-jsonl", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--source", type=str, default="phase_f_motion_fit_v1")
    parser.add_argument("--fit-mode", type=str, default="quantile", choices=("quantile", "extrema_refine"))
    parser.add_argument("--base-json", type=str, default=None)
    parser.add_argument("--compose-with-base", action="store_true", default=False)
    parser.add_argument("--min-throttle-gain", type=float, default=0.05)
    parser.add_argument("--max-throttle-gain", type=float, default=1.00)
    parser.add_argument("--min-steer-gain", type=float, default=0.50)
    parser.add_argument("--max-steer-gain", type=float, default=3.00)
    parser.add_argument("--max-tau-s", type=float, default=0.35)
    args = parser.parse_args()

    real_csv = Path(args.real_csv).expanduser().resolve()
    sim_jsonl = Path(args.sim_jsonl).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    real_stats = _load_motion_series(real_csv, kind="real")
    sim_stats = _load_motion_series(sim_jsonl, kind="sim")
    params = _derive_params(
        real_stats=real_stats,
        sim_stats=sim_stats,
        fit_mode=str(args.fit_mode),
        min_throttle_gain=float(args.min_throttle_gain),
        max_throttle_gain=float(args.max_throttle_gain),
        min_steer_gain=float(args.min_steer_gain),
        max_steer_gain=float(args.max_steer_gain),
        max_tau_s=float(args.max_tau_s),
    )
    base_payload: Dict[str, Any] = {}
    if args.base_json:
        with Path(args.base_json).expanduser().resolve().open("r", encoding="utf-8") as f:
            base_payload = json.load(f)
    if args.compose_with_base and base_payload:
        params["throttle_gain_ratio"] = float(params["throttle_gain_ratio"] * _safe_float(base_payload.get("throttle_gain_ratio", 1.0), 1.0))
        params["steer_gain_ratio"] = float(params["steer_gain_ratio"] * _safe_float(base_payload.get("steer_gain_ratio", 1.0), 1.0))
        params["steer_tau_s"] = float(params["steer_tau_s"] + _safe_float(base_payload.get("steer_tau_s", 0.0), 0.0))
        params["throttle_tau_s"] = float(params["throttle_tau_s"] + _safe_float(base_payload.get("throttle_tau_s", 0.0), 0.0))
        params["throttle_gain_ratio"] = _clip(params["throttle_gain_ratio"], float(args.min_throttle_gain), float(args.max_throttle_gain))
        params["steer_gain_ratio"] = _clip(params["steer_gain_ratio"], float(args.min_steer_gain), float(args.max_steer_gain))
        params["steer_tau_s"] = _clip(params["steer_tau_s"], 0.0, float(args.max_tau_s))
        params["throttle_tau_s"] = _clip(params["throttle_tau_s"], 0.0, float(args.max_tau_s))

    payload = {
        "source": str(args.source),
        "calibrated_at": datetime.now().isoformat(),
        "real_csv": str(real_csv),
        "sim_jsonl": str(sim_jsonl),
        "base_json": None if args.base_json in ("", None) else str(Path(args.base_json).expanduser().resolve()),
        "compose_with_base": bool(args.compose_with_base),
        "real_stats": real_stats,
        "sim_stats": sim_stats,
        "fit_mode": str(args.fit_mode),
        "fit_method": {
            "throttle_gain_ratio": "sqrt((real_speed_p50/sim_speed_p50) * (real_speed_p95/sim_speed_p95)) clipped",
            "steer_gain_ratio": "real_abs_final_angle_p95 / sim_abs_final_angle_p95 clipped",
            "steer_tau_s": "clip(dt_ref * log(sim_angle_rate_p95 / real_angle_rate_p95), 0, max_tau_s)",
            "throttle_tau_s": "clip(dt_ref * log(sim_throttle_rate_p95 / real_throttle_rate_p95), 0, max_tau_s)",
        },
        **params,
    }
    if str(args.fit_mode) == "extrema_refine":
        payload["fit_method"] = {
            "throttle_gain_ratio": "sqrt((real_speed_p99/sim_speed_p99) * (real_speed_max/sim_speed_max)) clipped; intended for compose-with-base refinement",
            "steer_gain_ratio": "1.0 (preserve base steer gain during refinement)",
            "steer_tau_s": "0.0 (preserve base steer lag during refinement)",
            "throttle_tau_s": "0.0 (preserve base throttle lag during refinement)",
        }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
