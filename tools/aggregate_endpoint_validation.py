#!/usr/bin/env python3
"""Aggregate V17 endpoint deployment validation runs.

Input is a run directory containing execution_manifest.jsonl plus per-run
summary.json and run_*.csv files. Output:
  - aggregate_metrics.json
  - aggregate_report.md
"""

import argparse
import csv
import glob
import json
import os
from typing import Dict, Iterable, List, Optional


CSV_METRICS = [
    "pilot_inference_latency_ms",
    "pilot_preprocess_latency_ms",
    "actor_residual_ms",
    "loop_dt_ms",
    "effective_fps",
    "gpu_load_pct",
    "cpu_load_pct",
    "power_in_mw",
    "pmic_temp",
    "lidar_scan_age_ms",
    "safety_inference_timeout_count",
    "safety_lidar_missing_count",
    "safety_lidar_stale_count",
    "safety_rp2040_missing_count",
]


def to_float(value) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        if out != out:
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


def round_or_none(value: Optional[float], digits: int = 3):
    return None if value is None else round(float(value), digits)


def read_manifest(base_dir: str) -> List[Dict[str, object]]:
    path = os.path.join(base_dir, "execution_manifest.jsonl")
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv_rows(paths: Iterable[str]) -> List[Dict[str, str]]:
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


def csv_metrics(csv_rows: List[Dict[str, str]]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    if not csv_rows:
        return out

    for name in CSV_METRICS:
        vals = []
        if name == "actor_residual_ms":
            for row in csv_rows:
                total = to_float(row.get("pilot_inference_latency_ms"))
                prep = to_float(row.get("pilot_preprocess_latency_ms"))
                if total is not None and prep is not None and total >= 0 and prep >= 0:
                    vals.append(max(0.0, total - prep))
        else:
            for row in csv_rows:
                value = to_float(row.get(name))
                if value is not None and value >= 0:
                    vals.append(value)

        out[name] = {
            "mean": round_or_none(sum(vals) / len(vals), 3) if vals else None,
            "p50": round_or_none(percentile(vals, 50), 3),
            "p95": round_or_none(percentile(vals, 95), 3),
            "p99": round_or_none(percentile(vals, 99), 3),
            "max": round_or_none(max(vals), 3) if vals else None,
        }

    modes = sorted(set(str(row.get("mode", "")).strip() for row in csv_rows if row.get("mode")))
    out["rows"] = {"count": len(csv_rows), "modes": modes}
    return out


def load_run(base_dir: str, manifest_row: Dict[str, object]) -> Dict[str, object]:
    run_dir = str(manifest_row.get("run_dir") or os.path.join(base_dir, str(manifest_row.get("name"))))
    summary_path = os.path.join(run_dir, "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
    csv_paths = sorted(glob.glob(os.path.join(run_dir, "run_*.csv")))
    metrics = csv_metrics(read_csv_rows(csv_paths))
    row = dict(manifest_row)
    row.update({
        "run_dir": run_dir,
        "summary": summary,
        "csv_metrics": metrics,
    })
    return row


def delta_pct(before, after):
    if before is None or after is None or float(before) == 0.0:
        return None
    return round((float(after) - float(before)) / float(before) * 100.0, 2)


def backend_ab(runs: List[Dict[str, object]]) -> Dict[str, Dict[str, Dict[str, object]]]:
    pytorch = None
    trt = None
    for run in runs:
        if run.get("category") != "backend_ab":
            continue
        backend = str(run.get("backend", "")).lower()
        if "pytorch" in backend:
            pytorch = run
        elif "tensorrt" in backend:
            trt = run
    if not pytorch or not trt:
        return {}

    out: Dict[str, Dict[str, Dict[str, object]]] = {}
    for name in [
        "pilot_inference_latency_ms",
        "pilot_preprocess_latency_ms",
        "actor_residual_ms",
        "loop_dt_ms",
        "effective_fps",
        "gpu_load_pct",
        "power_in_mw",
    ]:
        out[name] = {}
        for stat in ["p50", "p95", "mean"]:
            pv = pytorch.get("csv_metrics", {}).get(name, {}).get(stat)
            tv = trt.get("csv_metrics", {}).get(name, {}).get(stat)
            out[name][stat] = {
                "pytorch": pv,
                "tensorrt": tv,
                "delta_pct": delta_pct(pv, tv),
            }
    return out


def render_report(base_dir: str, runs: List[Dict[str, object]],
                  comparison: Dict[str, object]) -> str:
    lines = [
        "# V17 endpoint deployment validation aggregate report",
        "",
        "Scope: endpoint deployment stability, safety-blocking evidence, and reproducibility only. This report does not validate obstacle avoidance or active driving quality.",
        "",
        "## Run Matrix",
        "",
        "| run | category | backend | exit | rows | summary | non-takeover | inf p50/p95/p99 | loop p95 | lidar age p95 | safety timeouts |",
        "|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for run in runs:
        metrics = run.get("csv_metrics", {})
        rows = metrics.get("rows", {}).get("count")
        inf = metrics.get("pilot_inference_latency_ms", {})
        loop = metrics.get("loop_dt_ms", {})
        lidar = metrics.get("lidar_scan_age_ms", {})
        timeout = metrics.get("safety_inference_timeout_count", {}).get("max")
        lines.append(
            "| {name} | {category} | {backend} | {exit_code} | {rows} | {summary_exists} | {shadow_non_takeover_log} | {p50}/{p95}/{p99} | {loop_p95} | {lidar_p95} | {timeout} |".format(
                name=run.get("name", ""),
                category=run.get("category", ""),
                backend=run.get("backend", ""),
                exit_code=run.get("exit_code", ""),
                rows="" if rows is None else rows,
                summary_exists=run.get("summary_exists", False),
                shadow_non_takeover_log=run.get("shadow_non_takeover_log", False),
                p50=inf.get("p50"),
                p95=inf.get("p95"),
                p99=inf.get("p99"),
                loop_p95=loop.get("p95"),
                lidar_p95=lidar.get("p95"),
                timeout=timeout,
            )
        )

    if comparison:
        lines += [
            "",
            "## Backend A/B",
            "",
            "| metric | stat | PyTorch | TensorRT | delta % |",
            "|---|---|---:|---:|---:|",
        ]
        for metric, stats in comparison.items():
            for stat, values in stats.items():
                lines.append(
                    "| {metric} | {stat} | {pytorch} | {tensorrt} | {delta_pct} |".format(
                        metric=metric,
                        stat=stat,
                        pytorch=values.get("pytorch"),
                        tensorrt=values.get("tensorrt"),
                        delta_pct=values.get("delta_pct"),
                    )
                )

    lines += [
        "",
        "## Fault Injection Notes",
        "",
        "- Engine/metadata failures should exit before Vehicle.start and write preflight_report.json.",
        "- Sensor gates are deployment-safety evidence only; obstacle avoidance remains out of scope.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_dir", help="Validation run directory")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.expanduser(args.base_dir))
    manifest = read_manifest(base_dir)
    runs = [load_run(base_dir, row) for row in manifest]
    comparison = backend_ab(runs)
    metrics = {
        "base_dir": base_dir,
        "runs": runs,
        "backend_ab_comparison": comparison,
    }

    metrics_path = os.path.join(base_dir, "aggregate_metrics.json")
    report_path = os.path.join(base_dir, "aggregate_report.md")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(report_path, "w") as f:
        f.write(render_report(base_dir, runs, comparison))

    print("wrote:", metrics_path)
    print("wrote:", report_path)


if __name__ == "__main__":
    main()
