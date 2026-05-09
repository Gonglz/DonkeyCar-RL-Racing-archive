#!/usr/bin/env python3
"""
Run a reproducible Phase-F LiDAR parameter sweep.

The default preset focuses on the first-order geometry parameters that are most
likely to reduce sim-real gap before trying stochastic noise.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _run(cmd: List[str], log_path: Path) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(cmd) + "\n\n")
        fh.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=os.environ.copy(),
        )
    return {
        "cmd": cmd,
        "log_path": str(log_path),
        "returncode": int(proc.returncode),
        "duration_sec": float(time.time() - start),
    }


def _find_one(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one match for {pattern} under {path}, found {len(matches)}")
    return matches[0]


def _selection_score(item: Dict[str, Any]) -> float:
    metrics = item["metrics"]
    return float(
        float(metrics["valid_ratio_mae"]) / 0.10
        + float(metrics["wasserstein_median"]) / 0.08
        + float(metrics["wasserstein_p95"]) / 0.20
        + float(metrics["scene_js_divergence"]) / 0.15
    )


def _rank_key(item: Dict[str, Any]) -> tuple[float, float, float, float, float]:
    metrics = item["metrics"]
    return (
        _selection_score(item),
        float(metrics["valid_ratio_mae"]),
        float(metrics["wasserstein_median"]),
        float(metrics["scene_js_divergence"]),
        float(metrics["wasserstein_p95"]),
    )


def _pose_batch1_trials(base: Dict[str, float]) -> List[Dict[str, Any]]:
    offset_y = float(base["offset_y"])
    offset_z = float(base["offset_z"])
    rot_x = float(base["rot_x"])
    return [
        {"name": "baseline", "lidar": {"offset_y": offset_y, "offset_z": offset_z, "rot_x": rot_x}},
        {"name": "offset_y_020", "lidar": {"offset_y": 0.20, "offset_z": offset_z, "rot_x": rot_x}},
        {"name": "offset_y_030", "lidar": {"offset_y": 0.30, "offset_z": offset_z, "rot_x": rot_x}},
        {"name": "offset_y_035", "lidar": {"offset_y": 0.35, "offset_z": offset_z, "rot_x": rot_x}},
        {"name": "offset_z_035", "lidar": {"offset_y": offset_y, "offset_z": 0.35, "rot_x": rot_x}},
        {"name": "offset_z_065", "lidar": {"offset_y": offset_y, "offset_z": 0.65, "rot_x": rot_x}},
        {"name": "offset_z_080", "lidar": {"offset_y": offset_y, "offset_z": 0.80, "rot_x": rot_x}},
        {"name": "rot_x_m6", "lidar": {"offset_y": offset_y, "offset_z": offset_z, "rot_x": -6.0}},
        {"name": "rot_x_m3", "lidar": {"offset_y": offset_y, "offset_z": offset_z, "rot_x": -3.0}},
        {"name": "rot_x_p3", "lidar": {"offset_y": offset_y, "offset_z": offset_z, "rot_x": 3.0}},
        {"name": "rot_x_p6", "lidar": {"offset_y": offset_y, "offset_z": offset_z, "rot_x": 6.0}},
    ]


def _parse_trials(spec: str) -> List[Dict[str, Any]]:
    payload = json.loads(spec)
    if not isinstance(payload, list) or not payload:
        raise ValueError("--trials-json must be a non-empty JSON list")
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"trial #{idx} must be an object")
        name = str(item.get("name", f"trial_{idx:02d}")).strip()
        lidar = dict(item.get("lidar", {}) or {})
        if not name:
            raise ValueError(f"trial #{idx} has empty name")
        out.append({"name": name, "lidar": lidar})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a Phase-F LiDAR sweep")
    ap.add_argument("--manifest", default=str(REPO_ROOT / "scripts" / "v17_formal_readiness_manifest.json"))
    ap.add_argument("--preset", choices=("pose_batch1",), default="pose_batch1")
    ap.add_argument("--trials-json", default="", help="optional explicit trial list; overrides --preset")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--python-bin", default="")
    args = ap.parse_args()

    manifest = _load_json(args.manifest)
    phase_f = dict(manifest.get("phase_f", {}) or {})
    collect_cfg = dict(phase_f.get("collect", {}) or {})
    eval_cfg = dict(phase_f.get("eval", {}) or {})
    thresholds = dict(eval_cfg.get("thresholds", {}) or {})
    sim_path = str(manifest.get("sim_path", "remote"))
    sim_start_delay = float(phase_f.get("sim_start_delay", collect_cfg.get("sim_start_delay", 8.0)))
    python_bin = str(args.python_bin).strip() or str(manifest.get("python_bin", "")).strip() or sys.executable

    base_lidar = {
        "offset_y": float(collect_cfg.get("lidar_offset_y", 0.40)),
        "offset_z": float(collect_cfg.get("lidar_offset_z", 0.5)),
        "rot_x": float(collect_cfg.get("lidar_rot_x", 0.0)),
    }
    if args.trials_json.strip():
        trials = _parse_trials(args.trials_json)
    else:
        trials = _pose_batch1_trials(base_lidar)

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "preset": str(args.preset),
        "frames": int(args.frames),
        "sim_path": sim_path,
        "port": int(collect_cfg.get("port", 9091)),
        "real_paths": list(eval_cfg.get("real_paths", []) or []),
        "base_lidar": base_lidar,
        "trials": trials,
    }
    _write_json(output_root / "run_config.json", run_meta)

    results: List[Dict[str, Any]] = []
    for idx, trial in enumerate(trials, start=1):
        trial_name = str(trial["name"])
        lidar = dict(trial.get("lidar", {}) or {})
        trial_dir = output_root / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{idx}/{len(trials)}] {trial_name} lidar={json.dumps(lidar, ensure_ascii=False)}", flush=True)

        collect_cmd = [
            str(python_bin),
            str(REPO_ROOT / "scripts" / "collect_sim_lidar_monitor.py"),
            "--env-id",
            str(collect_cfg.get("env_id", "donkey-waveshare-v0")),
            "--policy-path",
            str((REPO_ROOT / str(collect_cfg.get("policy_path", ""))).resolve()),
            "--policy-format",
            str(collect_cfg.get("policy_format", "v16")),
            "--curriculum-phase",
            str(collect_cfg.get("curriculum_phase", "warmup")),
            "--sim",
            sim_path,
            "--sim-start-delay",
            str(sim_start_delay),
            "--frames",
            str(int(args.frames)),
            "--max-episode-steps",
            str(int(collect_cfg.get("max_episode_steps", 640))),
            "--port",
            str(int(collect_cfg.get("port", 9091))),
            "--obs-size",
            str(int(collect_cfg.get("obs_size", 128))),
            "--lidar-num-sectors",
            str(int(collect_cfg.get("lidar_num_sectors", 36))),
            "--lidar-max-range-m",
            str(float(collect_cfg.get("lidar_max_range_m", 20.0))),
            "--lidar-near-clip-m",
            str(float(collect_cfg.get("lidar_near_clip_m", 0.18))),
            "--lidar-deg-per-sweep-inc",
            str(float(collect_cfg.get("lidar_deg_per_sweep_inc", 1.0))),
            "--lidar-deg-ang-down",
            str(float(collect_cfg.get("lidar_deg_ang_down", 0.0))),
            "--lidar-deg-ang-delta",
            str(float(collect_cfg.get("lidar_deg_ang_delta", -1.0))),
            "--lidar-num-sweeps-levels",
            str(int(collect_cfg.get("lidar_num_sweeps_levels", 1))),
            "--lidar-noise",
            str(float(collect_cfg.get("lidar_noise", 0.0))),
            "--lidar-offset-x",
            str(float(collect_cfg.get("lidar_offset_x", 0.0))),
            "--lidar-offset-y",
            str(float(lidar.get("offset_y", collect_cfg.get("lidar_offset_y", 0.40)))),
            "--lidar-offset-z",
            str(float(lidar.get("offset_z", collect_cfg.get("lidar_offset_z", 0.5)))),
            "--lidar-rot-x",
            str(float(lidar.get("rot_x", collect_cfg.get("lidar_rot_x", 0.0)))),
            "--seed",
            str(int(manifest.get("seed", 123))),
            "--output-dir",
            str(trial_dir),
        ]
        collect_res = _run(collect_cmd, trial_dir / "collect.log")
        if int(collect_res["returncode"]) != 0:
            results.append(
                {
                    "name": trial_name,
                    "lidar": lidar,
                    "status": "collect_failed",
                    "collect": collect_res,
                    "eval": None,
                    "metrics": {},
                }
            )
            continue

        monitor_dir = trial_dir / "monitor_logs"
        sim_jsonl = _find_one(monitor_dir, "*_lidar_raw.jsonl")
        sim_summary = _find_one(monitor_dir, "*_summary.json")

        eval_json = trial_dir / "lidar_domain_gap_eval.json"
        eval_cmd = [
            str(python_bin),
            str(REPO_ROOT / "scripts" / "eval_lidar_domain_gap.py"),
            "--real-paths",
            *[str(Path(p).expanduser()) for p in list(eval_cfg.get("real_paths", []) or [])],
            "--sim-paths",
            str(sim_jsonl),
            "--num-sectors",
            str(int(eval_cfg.get("num_sectors", 36))),
            "--fov-deg",
            str(float(eval_cfg.get("fov_deg", 180.0))),
            "--max-range-m",
            str(float(eval_cfg.get("max_range_m", 6.0))),
            "--near-clip-m",
            str(float(eval_cfg.get("near_clip_m", 0.18))),
            "--js-bins",
            str(int(eval_cfg.get("js_bins", 24))),
            "--valid-ratio-mae-max",
            str(float(thresholds.get("valid_ratio_mae_max", 0.10))),
            "--wasserstein-median-max",
            str(float(thresholds.get("wasserstein_median_max", 0.08))),
            "--wasserstein-p95-max",
            str(float(thresholds.get("wasserstein_p95_max", 0.20))),
            "--scene-js-divergence-max",
            str(float(thresholds.get("scene_js_divergence_max", 0.15))),
            "--output-json",
            str(eval_json),
        ]
        eval_res = _run(eval_cmd, trial_dir / "eval.log")
        if int(eval_res["returncode"]) != 0:
            results.append(
                {
                    "name": trial_name,
                    "lidar": lidar,
                    "status": "eval_failed",
                    "collect": collect_res,
                    "eval": eval_res,
                    "metrics": {},
                    "artifacts": {
                        "sim_jsonl": str(sim_jsonl),
                        "sim_summary_json": str(sim_summary),
                    },
                }
            )
            continue

        eval_payload = _load_json(eval_json)
        sim_summary_payload = _load_json(sim_summary)
        metrics = dict(eval_payload["overall"]["summary"])
        metrics["selection_score"] = _selection_score({"metrics": metrics})
        metrics["sim_valid_ratio_mean"] = float(sim_summary_payload["valid_ratio"]["mean"])
        metrics["sim_scan_age_ms_mean"] = float(sim_summary_payload["scan_age_ms"]["mean"])
        metrics["sim_new_scan_count"] = int(sim_summary_payload["new_scan_count"])
        results.append(
            {
                "name": trial_name,
                "lidar": lidar,
                "status": "passed" if bool(metrics.get("pass", False)) else "evaluated",
                "collect": collect_res,
                "eval": eval_res,
                "metrics": metrics,
                "artifacts": {
                    "sim_jsonl": str(sim_jsonl),
                    "sim_summary_json": str(sim_summary),
                    "eval_json": str(eval_json),
                },
            }
        )

    successful = [item for item in results if item.get("metrics")]
    successful.sort(key=_rank_key)
    failed = [item for item in results if not item.get("metrics")]
    ordered = successful + failed

    summary = {
        "run_config": run_meta,
        "results": ordered,
        "best": successful[0] if successful else None,
        "completed_trials": len(successful),
        "failed_trials": len(failed),
    }
    _write_json(output_root / "sweep_summary.json", summary)

    if successful:
        best = successful[0]
        print("[best]", json.dumps(
            {
                "name": best["name"],
                "lidar": best["lidar"],
                "metrics": best["metrics"],
            },
            ensure_ascii=False,
            indent=2,
        ))
    print(f"[saved] {output_root / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
