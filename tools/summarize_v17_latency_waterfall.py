#!/usr/bin/env python3
"""Summarize available V17 final-run latency breakdown fields."""

import argparse
import csv
import glob
import json
import os
import re
from typing import Dict, Iterable, List, Optional


CSV_METRICS = [
    "pilot_preprocess_latency_ms",
    "actor_residual_ms",
    "pilot_inference_latency_ms",
    "loop_dt_ms",
    "lidar_scan_age_ms",
]


def to_float(value) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        if out != out or out < 0:
            return None
        return out
    except Exception:
        return None


def percentile(values: List[float], pct: float) -> Optional[float]:
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


def summarize_values(values: List[float]) -> Dict[str, Optional[float]]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(vals),
        "mean": round(sum(vals) / len(vals), 3),
        "p50": round(percentile(vals, 50), 3),
        "p95": round(percentile(vals, 95), 3),
        "p99": round(percentile(vals, 99), 3),
        "max": round(max(vals), 3),
    }


def expand_paths(patterns: Iterable[str]) -> List[str]:
    out = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        out.extend(matches if matches else [pattern])
    return sorted(dict.fromkeys(out))


def read_csv_metrics(paths: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    values = {name: [] for name in CSV_METRICS}
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total = to_float(row.get("pilot_inference_latency_ms"))
                prep = to_float(row.get("pilot_preprocess_latency_ms"))
                for name in CSV_METRICS:
                    if name == "actor_residual_ms":
                        if total is not None and prep is not None:
                            values[name].append(max(0.0, total - prep))
                    else:
                        val = to_float(row.get(name))
                        if val is not None:
                            values[name].append(val)
    return {name: summarize_values(vals) for name, vals in values.items()}


PART_RE = re.compile(
    r"^\|\s*(?P<part>[^|]+?)\s*\|\s*"
    r"(?P<max>[0-9.]+)\s*\|\s*(?P<min>[0-9.]+)\s*\|\s*"
    r"(?P<avg>[0-9.]+)\s*\|\s*(?P<p50>[0-9.]+)\s*\|\s*"
    r"(?P<p90>[0-9.]+)\s*\|\s*(?P<p99>[0-9.]+)\s*\|\s*"
    r"(?P<p999>[0-9.]+)\s*\|"
)


def parse_part_profile(path: str) -> Dict[str, Dict[str, float]]:
    if not path or not os.path.exists(path):
        return {}
    parts = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = PART_RE.match(line.strip())
            if not m:
                continue
            name = m.group("part").strip()
            parts[name] = {
                "max": float(m.group("max")),
                "min": float(m.group("min")),
                "avg": float(m.group("avg")),
                "p50": float(m.group("p50")),
                "p90": float(m.group("p90")),
                "p95": None,
                "p99": float(m.group("p99")),
                "p999": float(m.group("p999")),
            }
    return parts


def row_value(stats: Dict[str, Optional[float]], key: str) -> str:
    value = stats.get(key)
    return "" if value is None else ("%.3f" % float(value))


def write_markdown(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# V17 Final 20min Latency Waterfall\n\n")
        f.write("This uses fields already present in the final run. Camera capture, semantic internal stages, and obs-build internals were not separately instrumented in that run.\n\n")
        f.write("| module / metric | p50 ms | p95 ms | p99 ms | max ms | count |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for name in CSV_METRICS:
            s = payload["csv_metrics"][name]
            f.write(
                "| `%s` | %s | %s | %s | %s | %s |\n"
                % (name, row_value(s, "p50"), row_value(s, "p95"), row_value(s, "p99"), row_value(s, "max"), s.get("count", 0))
            )
        for part_name in ("V17Pilot", "DeploymentSafetyGate", "DataCollector"):
            part = payload["part_profile"].get(part_name)
            if not part:
                continue
            f.write(
                "| part `%s` | %.3f |  | %.3f | %.3f |  |\n"
                % (part_name, part["p50"], part["p99"], part["max"])
            )
        f.write("\n## Not Separately Instrumented\n\n")
        for item in payload["not_instrumented"]:
            f.write("- %s\n" % item)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--runtime-log", required=True)
    parser.add_argument("--out-json", default="latency_waterfall.json")
    parser.add_argument("--out-md", default="latency_waterfall.md")
    args = parser.parse_args()

    csv_paths = expand_paths(args.csv)
    csv_metrics = read_csv_metrics(csv_paths)
    part_profile = parse_part_profile(args.runtime_log)
    payload = {
        "csv_paths": [os.path.abspath(p) for p in csv_paths],
        "runtime_log": os.path.abspath(args.runtime_log),
        "csv_metrics": csv_metrics,
        "part_profile": part_profile,
        "not_instrumented": [
            "camera capture latency is only available as CSICamera part profile, not per-sample capture time",
            "semantic image preprocess internal stages were not split in this run",
            "obs dict build time was included in pilot_preprocess_latency_ms and not split out",
            "LiDAR receipt age and sectorization age were not split from lidar_scan_age_ms",
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    write_markdown(args.out_md, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
