#!/usr/bin/env python3
"""
Run the staged V17/LWM formal-readiness program.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import eval_world_model_readiness as wm_ready  # noqa: E402

DEFAULT_ENV_TRACK_FILES: Dict[str, str] = {
    "donkey-generated-track-v0": "manual_width_generated_track.json",
    "donkey-waveshare-v0": "manual_width_waveshare.json",
    "donkey-roboracingleague-track-v0": "manual_width_roboracingleague_track.json",
}


def _now_iso() -> str:
    return datetime.now().isoformat()


def _load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _rel_repo_path(raw_path: str | Path) -> Path:
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _output_path(output_root: Path, raw_path: str | Path) -> Path:
    path = Path(str(raw_path))
    return (output_root / path).resolve()


def _ensure_empty_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise RuntimeError(f"output directory must be empty for cold-start readiness: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)


def _git_command(args: Sequence[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()
    except Exception:
        return ""


def _run_command(
    cmd: Sequence[str],
    log_path: Path,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(cmd) + "\n\n")
        if env:
            fh.write("# env overrides:\n")
            for key in sorted(env.keys()):
                fh.write(f"{key}={env[key]}\n")
            fh.write("\n")
        fh.flush()
        proc_env = os.environ.copy()
        if env:
            proc_env.update({str(k): str(v) for k, v in env.items()})
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd or REPO_ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=proc_env,
        )
    return {
        "cmd": list(cmd),
        "cwd": str(cwd or REPO_ROOT),
        "log_path": str(log_path),
        "returncode": int(proc.returncode),
        "duration_sec": float(time.time() - start),
    }


def _run_python_json_snippet(
    python_bin: Path,
    code: str,
    argv: Sequence[str],
    output_json: Path,
    log_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cmd = [str(python_bin), "-c", code, *[str(x) for x in argv], str(output_json)]
    result = _run_command(cmd, log_path=log_path, cwd=REPO_ROOT)
    payload: Dict[str, Any] = {}
    if int(result["returncode"]) == 0 and output_json.exists():
        payload = _load_json(output_json)
    return result, payload


def _phase_result(status: str) -> Dict[str, Any]:
    return {
        "status": str(status),
        "started_at": _now_iso(),
        "finished_at": None,
        "artifacts": {},
        "metrics": {},
        "fail_reasons": [],
        "commands": [],
    }


def _finish_phase(phase: Dict[str, Any]) -> Dict[str, Any]:
    phase["finished_at"] = _now_iso()
    return phase


def _relabel_merged_passable_targets(
    npz_path: Path,
    meta_json: Path,
    passable_gap_threshold_m: float,
) -> Dict[str, float]:
    data = np.load(npz_path)
    payload = {key: data[key] for key in data.files}
    if "target_gap" not in payload:
        raise KeyError("target_gap missing from merged dataset")
    gaps = np.asarray(payload["target_gap"], dtype=np.float32)
    if gaps.ndim != 2 or gaps.shape[1] != 2:
        raise ValueError(f"expected target_gap shape (N, 2), got {gaps.shape}")
    passable = (gaps >= float(passable_gap_threshold_m)).astype(np.float32)
    payload["target_passable"] = passable
    np.savez_compressed(npz_path, **payload)

    meta: Dict[str, Any] = {}
    if meta_json.exists():
        try:
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    meta["passable_gap_threshold_m"] = float(passable_gap_threshold_m)
    meta["passable_left_rate"] = float(passable[:, 0].mean())
    meta["passable_right_rate"] = float(passable[:, 1].mean())
    meta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "passable_gap_threshold_m": float(passable_gap_threshold_m),
        "passable_left_rate": float(passable[:, 0].mean()),
        "passable_right_rate": float(passable[:, 1].mean()),
    }


def _read_metrics_events(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                events.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    return events


def _iter_jsonl_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in ("1", "true", "yes", "y", "on")


def _motion_quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(list(values), dtype=np.float32)
    return float(np.quantile(arr, float(q)))


def _compute_real_motion_stats(csv_path: Path) -> Dict[str, Any]:
    speed_proxy: List[float] = []
    abs_final_angle: List[float] = []
    usable_rows = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not _parse_bool(row.get("recording", False)):
                continue
            try:
                dt_s = float(row.get("loop_dt_ms", 0.0)) / 1000.0
                dx = float(row.get("delta_x", 0.0))
                dy = float(row.get("delta_y", 0.0))
                final_angle = float(row.get("final_angle", 0.0))
            except Exception:
                continue
            if not math.isfinite(dt_s) or dt_s <= 1e-6:
                continue
            speed = math.sqrt(dx * dx + dy * dy) / dt_s
            if not math.isfinite(speed):
                continue
            speed_proxy.append(float(speed))
            abs_final_angle.append(abs(float(final_angle)) if math.isfinite(final_angle) else 0.0)
            usable_rows += 1
    return {
        "source": str(csv_path),
        "usable_rows": int(usable_rows),
        "speed_proxy_p50": _motion_quantile(speed_proxy, 0.50),
        "speed_proxy_p95": _motion_quantile(speed_proxy, 0.95),
        "abs_final_angle_p95": _motion_quantile(abs_final_angle, 0.95),
    }


def _compute_sim_motion_stats(jsonl_path: Path) -> Dict[str, Any]:
    speed_vals: List[float] = []
    abs_final_angle: List[float] = []
    abs_final_throttle: List[float] = []
    usable_rows = 0
    for row in _iter_jsonl_records(jsonl_path):
        try:
            speed = float(row.get("speed", 0.0))
            final_angle = float(row.get("final_angle", 0.0))
            final_throttle = float(row.get("final_throttle", 0.0))
        except Exception:
            continue
        if not math.isfinite(speed):
            continue
        speed_vals.append(float(speed))
        abs_final_angle.append(abs(float(final_angle)) if math.isfinite(final_angle) else 0.0)
        abs_final_throttle.append(abs(float(final_throttle)) if math.isfinite(final_throttle) else 0.0)
        usable_rows += 1
    return {
        "source": str(jsonl_path),
        "usable_rows": int(usable_rows),
        "speed_proxy_p50": _motion_quantile(speed_vals, 0.50),
        "speed_proxy_p95": _motion_quantile(speed_vals, 0.95),
        "abs_final_angle_p95": _motion_quantile(abs_final_angle, 0.95),
        "abs_final_throttle_p95": _motion_quantile(abs_final_throttle, 0.95),
    }


def _motion_ratio(sim_value: float, real_value: float) -> float:
    sim_v = max(float(sim_value), 1e-6)
    real_v = max(float(real_value), 1e-6)
    return float(sim_v / real_v)


def _build_motion_report(
    *,
    real_csv_path: Path,
    sim_jsonl_path: Path,
    thresholds: Mapping[str, Any],
) -> Dict[str, Any]:
    real_stats = _compute_real_motion_stats(real_csv_path)
    sim_stats = _compute_sim_motion_stats(sim_jsonl_path)
    speed_ratio_p50 = _motion_ratio(sim_stats["speed_proxy_p50"], real_stats["speed_proxy_p50"])
    speed_ratio_p95 = _motion_ratio(sim_stats["speed_proxy_p95"], real_stats["speed_proxy_p95"])
    angle_gap = abs(float(sim_stats["abs_final_angle_p95"]) - float(real_stats["abs_final_angle_p95"]))
    speed_min = float(thresholds.get("speed_ratio_min", 0.80))
    speed_max = float(thresholds.get("speed_ratio_max", 1.25))
    angle_gap_max = float(thresholds.get("abs_final_angle_p95_gap_max", 0.10))
    passed = bool(
        speed_min <= speed_ratio_p50 <= speed_max
        and speed_min <= speed_ratio_p95 <= speed_max
        and angle_gap <= angle_gap_max
    )
    return {
        "thresholds": {
            "speed_ratio_min": speed_min,
            "speed_ratio_max": speed_max,
            "abs_final_angle_p95_gap_max": angle_gap_max,
        },
        "real": real_stats,
        "sim": sim_stats,
        "derived": {
            "speed_proxy_p50_ratio": speed_ratio_p50,
            "speed_proxy_p95_ratio": speed_ratio_p95,
            "abs_final_angle_p95_gap": angle_gap,
            "motion_score": [
                abs(math.log(max(speed_ratio_p50, 1e-6))),
                abs(math.log(max(speed_ratio_p95, 1e-6))),
                angle_gap,
            ],
        },
        "pass": passed,
    }


def _motion_score_tuple(report: Mapping[str, Any]) -> Tuple[float, float, float]:
    derived = dict(report.get("derived", {}) or {})
    score = list(derived.get("motion_score", []) or [])
    if len(score) != 3:
        return (float("inf"), float("inf"), float("inf"))
    return (float(score[0]), float(score[1]), float(score[2]))


def _motion_pass_with_thresholds(report: Mapping[str, Any], thresholds: Mapping[str, Any]) -> bool:
    derived = dict(report.get("derived", {}) or {})
    speed_ratio_p50 = float(derived.get("speed_proxy_p50_ratio", float("inf")))
    speed_ratio_p95 = float(derived.get("speed_proxy_p95_ratio", float("inf")))
    angle_gap = float(derived.get("abs_final_angle_p95_gap", float("inf")))
    speed_min = float(thresholds.get("speed_ratio_min", 0.80))
    speed_max = float(thresholds.get("speed_ratio_max", 1.25))
    angle_gap_max = float(thresholds.get("abs_final_angle_p95_gap_max", 0.10))
    return bool(
        speed_min <= speed_ratio_p50 <= speed_max
        and speed_min <= speed_ratio_p95 <= speed_max
        and angle_gap <= angle_gap_max
    )


def _derive_real_motion_csv(primary_real_path: str | Path) -> Path:
    path = Path(str(primary_real_path)).expanduser()
    if path.suffix.lower() == ".csv":
        return path
    if path.name.endswith("_lidar_raw.jsonl"):
        candidate = path.with_name(path.name.replace("_lidar_raw.jsonl", ".csv"))
        if candidate.exists():
            return candidate
    csv_candidates = sorted(path.parent.glob("*.csv"))
    if len(csv_candidates) == 1:
        return csv_candidates[0]
    raise FileNotFoundError(f"could not derive a unique real motion CSV from {path}")


def _build_common_manifest_view(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "manifest_id": manifest.get("manifest_id"),
        "launch_target": manifest.get("launch_target"),
        "python_bin": manifest.get("python_bin"),
        "sim_path": manifest.get("sim_path"),
        "seed": manifest.get("seed"),
    }


def _manifest_sim_path(manifest: Mapping[str, Any], fallback: Optional[str] = None) -> str:
    candidate = manifest.get("sim_path")
    if candidate is None:
        candidate = fallback
    return str(candidate or "").strip()


def _track_search_dirs(manifest: Mapping[str, Any]) -> List[Path]:
    phase_a_cfg = dict(manifest.get("phase_a", {}) or {})
    raw_dirs: List[str | Path] = []
    if phase_a_cfg.get("track_dir"):
        raw_dirs.append(str(phase_a_cfg["track_dir"]))
    raw_dirs.extend(list(phase_a_cfg.get("track_search_dirs", []) or []))

    env_override = os.environ.get("MYSIM_TRACK_DIR", "").strip()
    if env_override:
        raw_dirs.append(env_override)

    raw_dirs.extend(
        [
            REPO_ROOT / "track_profiles",
            REPO_ROOT / "track",
            Path("/home/longzhao/track"),
            REPO_ROOT / "module" / "track_data",
        ]
    )

    out: List[Path] = []
    seen: set[str] = set()
    for raw in raw_dirs:
        try:
            path = _rel_repo_path(raw)
        except Exception:
            path = Path(str(raw)).expanduser().resolve()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _materialize_track_dir(
    manifest: Mapping[str, Any],
    output_root: Path,
    env_ids: Sequence[str],
) -> Tuple[Path, Dict[str, Any]]:
    phase_a_cfg = dict(manifest.get("phase_a", {}) or {})
    track_file_map = dict(DEFAULT_ENV_TRACK_FILES)
    track_file_map.update(dict(phase_a_cfg.get("track_files", {}) or {}))
    candidate_dirs = _track_search_dirs(manifest)
    resolved_dir = output_root / "_resolved_track_dir"
    resolved_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "resolved_track_dir": str(resolved_dir),
        "candidate_dirs": [str(path) for path in candidate_dirs],
        "requested_env_ids": [str(x) for x in env_ids],
        "resolved_files": {},
        "missing": [],
    }

    for env_id in env_ids:
        env_id = str(env_id)
        track_file = track_file_map.get(env_id)
        if not track_file:
            summary["missing"].append({"env_id": env_id, "reason": "track_file mapping missing"})
            continue

        dest = resolved_dir / track_file
        found_source: Optional[Path] = None
        for candidate_dir in candidate_dirs:
            source = candidate_dir / track_file
            if source.is_file():
                found_source = source
                break

        if found_source is None:
            summary["missing"].append(
                {
                    "env_id": env_id,
                    "track_file": track_file,
                    "reason": "file not found in candidate_dirs",
                }
            )
            continue

        if not dest.exists():
            try:
                os.symlink(found_source, dest)
                mode = "symlink"
            except OSError:
                shutil.copy2(found_source, dest)
                mode = "copy"
        else:
            mode = "existing"

        summary["resolved_files"][env_id] = {
            "track_file": track_file,
            "source": str(found_source),
            "dest": str(dest),
            "mode": mode,
        }

    if summary["missing"]:
        raise FileNotFoundError(
            "could not materialize a complete track_dir: "
            + json.dumps(summary["missing"], ensure_ascii=False)
        )

    return resolved_dir, summary


def _probe_tcp(host: str, port: int, timeout_s: float = 1.0) -> Tuple[bool, str]:
    try:
        with socket.create_connection((str(host), int(port)), timeout=float(timeout_s)):
            return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _phase_a(
    manifest: Mapping[str, Any],
    output_root: Path,
    report: Dict[str, Any],
) -> Dict[str, Any]:
    phase = _phase_result("passed")
    phase_dir = output_root / "phase_a"
    phase_dir.mkdir(parents=True, exist_ok=True)
    fail_reasons = phase["fail_reasons"]

    python_bin = Path(str(manifest.get("python_bin", sys.executable))).expanduser()
    if not python_bin.exists():
        fail_reasons.append(f"python_bin not found: {python_bin}")

    phase_a_cfg = dict(manifest.get("phase_a", {}) or {})
    required_scripts = [_rel_repo_path(x) for x in phase_a_cfg.get("required_scripts", [])]
    for path in required_scripts:
        if not path.exists():
            fail_reasons.append(f"required script missing: {path}")

    required_paths = {
        key: _rel_repo_path(value)
        for key, value in dict(phase_a_cfg.get("required_paths", {}) or {}).items()
    }
    for key, path in required_paths.items():
        if not path.exists():
            fail_reasons.append(f"required path missing ({key}): {path}")

    phase["artifacts"]["provenance_json"] = str((phase_dir / "provenance.json").resolve())
    phase["metrics"]["git_sha"] = _git_command(["rev-parse", "HEAD"])
    phase["metrics"]["git_dirty"] = bool(_git_command(["status", "--porcelain"]))
    phase["metrics"]["python_bin"] = str(python_bin)
    phase["metrics"]["required_scripts"] = [str(p) for p in required_scripts]
    phase["metrics"]["required_paths"] = {key: str(path) for key, path in required_paths.items()}

    track_dir: Optional[Path] = None
    try:
        track_dir, track_summary = _materialize_track_dir(
            manifest=manifest,
            output_root=output_root,
            env_ids=list(phase_a_cfg.get("env_ids", ["donkey-generated-track-v0", "donkey-waveshare-v0"])),
        )
        phase["metrics"]["resolved_track_dir"] = str(track_dir)
        phase["metrics"]["track_resolution"] = track_summary
        phase["artifacts"]["resolved_track_dir"] = str(track_dir)
        phase["artifacts"]["track_resolution_json"] = str((phase_dir / "track_resolution.json").resolve())
        _write_json(phase_dir / "track_resolution.json", track_summary)
    except Exception as exc:
        fail_reasons.append(f"track_dir resolution failed: {type(exc).__name__}: {exc}")

    preflight_code = r"""
import json
import sys
from pathlib import Path
repo_root = Path.cwd()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from module.track import TrackGeometryManager
from module.utils import load_config
from src import ppo_multitrack_v17 as v17
env_ids = json.loads(sys.argv[1])
track_dir = sys.argv[2]
obs_size = int(sys.argv[3])
lidar_obs_mode = sys.argv[4]
output_json = sys.argv[5]
cfg = load_config(myconfig=v17.DEFAULT_MYCONFIG)
track_geometry = TrackGeometryManager(track_dir=track_dir, env_ids=env_ids, scene_specs=v17.SCENE_SPECS)
v17.run_preflight_tests(track_geometry=track_geometry, obs_size=obs_size, lidar_obs_mode=lidar_obs_mode)
payload = {
    "v17_preflight_passed": True,
    "sim_path": getattr(cfg, "DONKEY_SIM_PATH", None) if cfg is not None else None,
}
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
"""
    phase["metrics"]["v17_preflight_passed"] = False
    phase["metrics"]["sim_path"] = None
    if track_dir is not None and python_bin.exists():
        preflight_result, preflight_payload = _run_python_json_snippet(
            python_bin=python_bin,
            code=preflight_code,
            argv=[
                json.dumps(list(phase_a_cfg.get("env_ids", ["donkey-generated-track-v0", "donkey-waveshare-v0"]))),
                str(track_dir),
                str(int(phase_a_cfg.get("obs_size", 128))),
                str(phase_a_cfg.get("lidar_obs_mode", "full")),
            ],
            output_json=phase_dir / "phase_a_preflight.json",
            log_path=phase_dir / "phase_a_preflight.log",
        )
        phase["commands"].append(preflight_result)
        phase["metrics"]["v17_preflight_passed"] = bool(preflight_payload.get("v17_preflight_passed", False))
        phase["metrics"]["sim_path_from_config"] = preflight_payload.get("sim_path")
        phase["metrics"]["sim_path"] = _manifest_sim_path(manifest, fallback=preflight_payload.get("sim_path"))
        if int(preflight_result["returncode"]) != 0 or not phase["metrics"]["v17_preflight_passed"]:
            fail_reasons.append(f"v17 preflight failed with return code {preflight_result['returncode']}")
    else:
        fail_reasons.append("v17 preflight skipped because track_dir resolution or python discovery failed")

    phase["metrics"]["sim_path"] = _manifest_sim_path(
        manifest,
        fallback=phase["metrics"].get("sim_path"),
    )
    sim_path = str(phase["metrics"].get("sim_path") or "").strip()
    sim_mode = "unknown"
    sim_check_passed = False
    sim_probe_host = str(phase_a_cfg.get("sim_probe_host", "127.0.0.1"))
    sim_probe_port = int(phase_a_cfg.get("sim_probe_port", manifest.get("phase_b", {}).get("port", 9091)))
    sim_probe_timeout_s = float(phase_a_cfg.get("sim_probe_timeout_s", 1.0))
    if sim_path and sim_path not in ("remote", "none"):
        sim_mode = "local_binary"
        sim_check_passed = Path(sim_path).expanduser().exists()
        if not sim_check_passed:
            fail_reasons.append(f"local simulator executable not found: {sim_path}")
    elif sim_path == "remote":
        sim_mode = "remote_tcp"
        sim_check_passed, sim_probe_error = _probe_tcp(
            host=sim_probe_host,
            port=sim_probe_port,
            timeout_s=sim_probe_timeout_s,
        )
        phase["metrics"]["sim_probe_error"] = sim_probe_error
        if not sim_check_passed:
            fail_reasons.append(
                f"remote simulator tcp not reachable at {sim_probe_host}:{sim_probe_port}: {sim_probe_error}"
            )
    else:
        phase["metrics"]["sim_probe_error"] = "sim_path missing or unsupported"
        fail_reasons.append("simulator path could not be resolved from config")
    phase["metrics"]["sim_mode"] = sim_mode
    phase["metrics"]["sim_probe_host"] = sim_probe_host
    phase["metrics"]["sim_probe_port"] = sim_probe_port
    phase["metrics"]["sim_check_passed"] = bool(sim_check_passed)

    real_lidar_paths = [Path(str(p)).expanduser() for p in phase_a_cfg.get("real_lidar_paths", [])]
    real_parse_code = r"""
import json
import sys
from pathlib import Path
repo_root = Path.cwd()
scripts_dir = repo_root / "scripts"
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
from eval_lidar_domain_gap import CanonicalLidarSpec, _extract_real_from_obj, _iter_input_files, _iter_jsonl
real_paths = json.loads(sys.argv[1])
num_sectors = int(sys.argv[2])
fov_deg = float(sys.argv[3])
max_range_m = float(sys.argv[4])
near_clip_m = float(sys.argv[5])
output_json = sys.argv[6]
spec = CanonicalLidarSpec(
    num_sectors=num_sectors,
    fov_deg=fov_deg,
    max_range_m=max_range_m,
    near_clip_m=near_clip_m,
    invalid_fill_m=max_range_m,
)
usable = 0
for raw_path in real_paths:
    for path in _iter_input_files([raw_path]):
        if path.suffix.lower() != ".jsonl":
            continue
        for obj in _iter_jsonl(path):
            if _extract_real_from_obj(obj, spec) is not None:
                usable += 1
                break
        if usable > 0:
            break
    if usable > 0:
        break
with open(output_json, "w", encoding="utf-8") as f:
    json.dump({"usable_real_samples_checked": usable}, f, indent=2, ensure_ascii=False)
"""
    real_parse_result, real_parse_payload = _run_python_json_snippet(
        python_bin=python_bin,
        code=real_parse_code,
        argv=[
            json.dumps([str(p) for p in real_lidar_paths]),
            str(int(phase_a_cfg.get("lidar_num_sectors", 36))),
            str(float(phase_a_cfg.get("lidar_fov_deg", 180.0))),
            str(float(phase_a_cfg.get("lidar_max_range_m", 20.0))),
            str(float(phase_a_cfg.get("lidar_near_clip_m", 0.18))),
        ],
        output_json=phase_dir / "phase_a_real_lidar_parse.json",
        log_path=phase_dir / "phase_a_real_lidar_parse.log",
    )
    phase["commands"].append(real_parse_result)
    usable_real_samples = int(real_parse_payload.get("usable_real_samples_checked", 0) or 0)
    phase["metrics"]["usable_real_lidar_samples_checked"] = usable_real_samples
    phase["metrics"]["real_lidar_paths"] = [str(p) for p in real_lidar_paths]
    if int(real_parse_result["returncode"]) != 0:
        fail_reasons.append(
            f"real LiDAR parse check failed with return code {real_parse_result['returncode']}"
        )
    elif usable_real_samples <= 0:
        fail_reasons.append("real LiDAR logs could not be parsed into canonical samples")

    provenance = {
        "timestamp": _now_iso(),
        "repo_root": str(REPO_ROOT),
        "manifest": _build_common_manifest_view(manifest),
        "git_sha": phase["metrics"].get("git_sha"),
        "git_dirty": phase["metrics"].get("git_dirty"),
        "python_bin": str(python_bin),
        "resolved_track_dir": phase["metrics"].get("resolved_track_dir"),
        "sim_path": phase["metrics"].get("sim_path"),
        "sim_mode": phase["metrics"].get("sim_mode"),
        "sim_check_passed": phase["metrics"].get("sim_check_passed"),
        "sim_probe_host": phase["metrics"].get("sim_probe_host"),
        "sim_probe_port": phase["metrics"].get("sim_probe_port"),
        "required_scripts": [str(p) for p in required_scripts],
        "required_paths": {key: str(path) for key, path in required_paths.items()},
        "real_lidar_paths": phase["metrics"].get("real_lidar_paths", []),
        "usable_real_lidar_samples_checked": phase["metrics"].get("usable_real_lidar_samples_checked", 0),
        "v17_preflight_passed": phase["metrics"].get("v17_preflight_passed", False),
    }
    _write_json(phase_dir / "provenance.json", provenance)

    if fail_reasons:
        phase["status"] = "failed"
    return _finish_phase(phase)


def _phase_b(
    manifest: Mapping[str, Any],
    output_root: Path,
) -> Dict[str, Any]:
    phase = _phase_result("passed")
    phase_cfg = dict(manifest.get("phase_b", {}) or {})
    phase_dir = output_root / str(phase_cfg.get("save_dir", "phase_b_ppo_smoke"))
    phase_dir.mkdir(parents=True, exist_ok=True)
    python_bin = str(Path(str(manifest.get("python_bin", sys.executable))).expanduser())
    script_path = str(_rel_repo_path("src/ppo_multitrack_v17.py"))
    seed = int(manifest.get("seed", 123))
    try:
        resolved_track_dir, track_summary = _materialize_track_dir(
            manifest=manifest,
            output_root=output_root,
            env_ids=list(phase_cfg.get("env_ids", ["donkey-generated-track-v0"])),
        )
    except Exception as exc:
        phase["fail_reasons"].append(f"phase_b track_dir resolution failed: {type(exc).__name__}: {exc}")
        phase["status"] = "failed"
        return _finish_phase(phase)

    cmd = [
        python_bin,
        script_path,
        "--env-ids",
        *[str(x) for x in phase_cfg.get("env_ids", ["donkey-generated-track-v0"])],
        "--sim",
        _manifest_sim_path(manifest, fallback="remote"),
        "--track-dir",
        str(resolved_track_dir),
        "--steps",
        str(int(phase_cfg.get("steps", 256))),
        "--save-dir",
        str(phase_dir),
        "--port",
        str(int(phase_cfg.get("port", 9091))),
        "--obs-size",
        str(int(phase_cfg.get("obs_size", 128))),
        "--ppo-n-steps",
        str(int(phase_cfg.get("ppo_n_steps", 256))),
        "--ppo-batch-size",
        str(int(phase_cfg.get("ppo_batch_size", 64))),
        "--ppo-n-epochs",
        str(int(phase_cfg.get("ppo_n_epochs", 4))),
        "--learning-rate",
        str(float(phase_cfg.get("learning_rate", 8e-5))),
        "--ent-coef",
        str(float(phase_cfg.get("ent_coef", 0.01))),
        "--target-kl",
        str(float(phase_cfg.get("target_kl", 0.01))),
        "--file-metrics-log-freq",
        str(int(phase_cfg.get("file_metrics_log_freq", 50))),
        "--seed",
        str(seed),
        "--exp-tag",
        str(phase_cfg.get("exp_tag", "v17_formal_readiness_smoke")),
    ]
    scene_weights = list(phase_cfg.get("scene_weights", []) or [])
    if scene_weights:
        cmd.extend(["--scene-weights", *[str(x) for x in scene_weights]])

    command_result = _run_command(cmd, log_path=phase_dir / "phase_b_ppo_smoke.log")
    phase["commands"].append(command_result)

    final_model_zip = phase_dir / "final_model.zip"
    final_model_pth = phase_dir / "final_model_policy.pth"
    config_json = phase_dir / "v17_config.json"
    metrics_jsonl = phase_dir / "train_metrics.jsonl"
    phase["artifacts"] = {
        "save_dir": str(phase_dir),
        "resolved_track_dir": str(resolved_track_dir),
        "track_resolution_json": str((phase_dir / "track_resolution.json").resolve()),
        "final_model_zip": str(final_model_zip),
        "final_model_pth": str(final_model_pth),
        "config_json": str(config_json),
        "metrics_jsonl": str(metrics_jsonl),
    }
    _write_json(phase_dir / "track_resolution.json", track_summary)

    if int(command_result["returncode"]) != 0:
        phase["fail_reasons"].append(f"ppo smoke command failed with return code {command_result['returncode']}")

    missing = [str(path) for path in (final_model_zip, final_model_pth, config_json, metrics_jsonl) if not path.exists()]
    if missing:
        phase["fail_reasons"].append(f"ppo smoke missing expected artifacts: {missing}")

    metrics_events = _read_metrics_events(metrics_jsonl)
    non_zero_update = False
    last_end_event: Dict[str, Any] = {}
    for event in metrics_events:
        metrics = dict(event.get("metrics", {}) or {})
        if _safe_float(metrics.get("train/n_updates", 0.0), 0.0) > 0.0:
            non_zero_update = True
        if str(event.get("event")) == "end":
            last_end_event = event
    if not non_zero_update:
        phase["fail_reasons"].append("ppo smoke metrics log does not contain non-zero train/n_updates")

    phase["metrics"] = {
        "metrics_events": int(len(metrics_events)),
        "non_zero_update": bool(non_zero_update),
        "final_callback_num_timesteps": int(last_end_event.get("callback_num_timesteps", 0) or 0),
        "end_metrics": dict(last_end_event.get("metrics", {}) or {}),
    }

    if phase["fail_reasons"]:
        phase["status"] = "failed"
    return _finish_phase(phase)


def _phase_c(
    manifest: Mapping[str, Any],
    output_root: Path,
) -> Dict[str, Any]:
    phase = _phase_result("passed")
    phase_cfg = dict(manifest.get("phase_c", {}) or {})
    phase_dir = output_root / "phase_c_dataset"
    phase_dir.mkdir(parents=True, exist_ok=True)
    python_bin = str(Path(str(manifest.get("python_bin", sys.executable))).expanduser())
    script_path = str(_rel_repo_path("scripts/export_world_model_dataset.py"))
    seed = int(manifest.get("seed", 123))
    base_port = int(phase_cfg.get("base_port", 9092))
    export_sources: List[Dict[str, Any]] = []
    export_env_ids = sorted(
        {
            str(env_id)
            for export_cfg in list(phase_cfg.get("exports", []) or [])
            for env_id in list(export_cfg.get("env_ids", []) or [])
        }
    )
    try:
        resolved_track_dir, track_summary = _materialize_track_dir(
            manifest=manifest,
            output_root=output_root,
            env_ids=export_env_ids,
        )
    except Exception as exc:
        phase["fail_reasons"].append(f"phase_c track_dir resolution failed: {type(exc).__name__}: {exc}")
        phase["status"] = "failed"
        return _finish_phase(phase)
    command_env = {"MYSIM_TRACK_DIR": str(resolved_track_dir)}
    phase["artifacts"]["resolved_track_dir"] = str(resolved_track_dir)
    phase["artifacts"]["track_resolution_json"] = str((phase_dir / "track_resolution.json").resolve())
    _write_json(phase_dir / "track_resolution.json", track_summary)

    for idx, export_cfg in enumerate(list(phase_cfg.get("exports", []) or [])):
        name = str(export_cfg["name"])
        output_npz = _output_path(output_root, export_cfg["output"])
        sim2real_json = str(
            export_cfg.get(
                "sim2real_json",
                phase_cfg.get(
                    "sim2real_json",
                    manifest.get("phase_f", {}).get("collect", {}).get("sim2real_json", ""),
                ),
            )
            or ""
        ).strip()
        cmd = [
            python_bin,
            script_path,
            "--env-ids",
            *[str(x) for x in export_cfg.get("env_ids", [])],
            "--sim",
            str(export_cfg.get("sim", phase_cfg.get("sim", _manifest_sim_path(manifest, fallback="remote")))),
            "--sim-start-delay",
            str(float(export_cfg.get("sim_start_delay", phase_cfg.get("sim_start_delay", 8.0)))),
            "--samples",
            str(int(export_cfg["samples"])),
            "--max-episode-steps",
            str(int(export_cfg.get("max_episode_steps", 640))),
            "--port",
            str(int(export_cfg.get("port", base_port + idx))),
            "--curriculum-phase",
            str(export_cfg.get("curriculum_phase", "lane_pid_full")),
            "--obs-size",
            str(int(phase_cfg.get("obs_size", 128))),
            "--lidar-num-sectors",
            str(int(phase_cfg.get("lidar_num_sectors", 36))),
            "--lidar-max-range-m",
            str(float(phase_cfg.get("lidar_max_range_m", 20.0))),
            "--passable-gap-threshold-m",
            str(float(phase_cfg.get("passable_gap_threshold_m", 0.70))),
            "--output",
            str(output_npz),
            "--seed",
            str(seed),
        ]
        scene_weights = list(export_cfg.get("scene_weights", []) or [])
        if scene_weights:
            cmd.extend(["--scene-weights", *[str(x) for x in scene_weights]])
        if sim2real_json:
            cmd.extend(["--sim2real-json", str(Path(sim2real_json).expanduser())])
        policy_path = _rel_repo_path(export_cfg["policy_path"])
        cmd.extend(
            [
                "--policy-path",
                str(policy_path),
                "--policy-format",
                str(export_cfg.get("policy_format", "v16")),
            ]
        )

        command_result = _run_command(
            cmd,
            log_path=phase_dir / f"{name}.log",
            env=command_env,
        )
        phase["commands"].append(command_result)
        meta_json = output_npz.with_suffix(".json")
        if int(command_result["returncode"]) != 0:
            phase["fail_reasons"].append(f"dataset export {name} failed with return code {command_result['returncode']}")
        if not output_npz.exists() or not meta_json.exists():
            phase["fail_reasons"].append(f"dataset export {name} missing output artifacts")
        export_sources.append({"name": name, "path": str(output_npz), "meta_json": str(meta_json)})

    merged_output_npz = _output_path(output_root, phase_cfg.get("merged_output_npz", "phase_c_dataset/wm_dataset_mix_v1.npz"))
    merged_output_json = _output_path(output_root, phase_cfg.get("merged_output_json", "phase_c_dataset/wm_dataset_mix_v1.json"))
    if not phase["fail_reasons"]:
        try:
            merge_summary = wm_ready.merge_world_model_datasets(
                sources=export_sources,
                output_npz=merged_output_npz,
                output_json=merged_output_json,
            )
            passable_relabel = _relabel_merged_passable_targets(
                npz_path=merged_output_npz,
                meta_json=merged_output_json,
                passable_gap_threshold_m=float(phase_cfg.get("passable_gap_threshold_m", 0.70)),
            )
            dataset_eval = wm_ready.evaluate_dataset_readiness(
                npz_path=merged_output_npz,
                meta_json=merged_output_json,
                manifest=manifest,
            )
            _write_json(phase_dir / "dataset_readiness.json", dataset_eval)
            phase["artifacts"].update(
                {
                    "merged_npz": str(merged_output_npz),
                    "merged_json": str(merged_output_json),
                    "dataset_readiness_json": str((phase_dir / "dataset_readiness.json").resolve()),
                }
            )
            phase["metrics"] = {
                "merge_summary": merge_summary,
                "passable_relabel": passable_relabel,
                "dataset_readiness": dataset_eval.get("metrics", {}),
            }
            if dataset_eval.get("status") != "passed":
                phase["fail_reasons"].extend(list(dataset_eval.get("fail_reasons", [])))
        except Exception as exc:
            phase["fail_reasons"].append(f"dataset merge/readiness failed: {type(exc).__name__}: {exc}")

    if phase["fail_reasons"]:
        phase["status"] = "failed"
    return _finish_phase(phase)


def _phase_d(
    manifest: Mapping[str, Any],
    output_root: Path,
    merged_npz: Path,
) -> Dict[str, Any]:
    phase = _phase_result("passed")
    phase_cfg = dict(manifest.get("phase_d", {}) or {})
    phase_dir = output_root / str(phase_cfg.get("save_dir", "phase_d_world_model_train"))
    phase_dir.mkdir(parents=True, exist_ok=True)
    python_bin = str(Path(str(manifest.get("python_bin", sys.executable))).expanduser())
    script_path = str(_rel_repo_path("scripts/train_world_model_v17.py"))
    seed = int(manifest.get("seed", 123))

    cmd = [
        python_bin,
        script_path,
        "--data",
        str(merged_npz),
        "--save-dir",
        str(phase_dir),
        "--seq-len",
        str(int(phase_cfg.get("seq_len", 4))),
        "--val-ratio",
        str(float(phase_cfg.get("val_ratio", 0.15))),
        "--batch-size",
        str(int(phase_cfg.get("batch_size", 256))),
        "--hidden-dim",
        str(int(phase_cfg.get("hidden_dim", 128))),
        "--epochs-a",
        str(int(phase_cfg.get("epochs_a", 8))),
        "--epochs-b",
        str(int(phase_cfg.get("epochs_b", 6))),
        "--epochs-c",
        str(int(phase_cfg.get("epochs_c", 3))),
        "--lr-a",
        str(float(phase_cfg.get("lr_a", 1e-3))),
        "--lr-b",
        str(float(phase_cfg.get("lr_b", 5e-4))),
        "--lr-c",
        str(float(phase_cfg.get("lr_c", 1e-4))),
        "--seed",
        str(seed),
        "--device",
        str(phase_cfg.get("device", "auto")),
    ]
    command_result = _run_command(cmd, log_path=phase_dir / "phase_d_world_model_train.log")
    phase["commands"].append(command_result)

    summary_json = phase_dir / "train_summary.json"
    phase["artifacts"] = {
        "save_dir": str(phase_dir),
        "train_summary_json": str(summary_json),
        "final_checkpoint": str(phase_dir / "local_world_model_v17_final.pth"),
    }
    if int(command_result["returncode"]) != 0:
        phase["fail_reasons"].append(f"world-model training failed with return code {command_result['returncode']}")
    if not summary_json.exists():
        phase["fail_reasons"].append("world-model training did not produce train_summary.json")

    if not phase["fail_reasons"]:
        try:
            training_eval = wm_ready.evaluate_training_readiness(summary_json=summary_json, manifest=manifest)
            _write_json(phase_dir / "training_readiness.json", training_eval)
            phase["artifacts"]["training_readiness_json"] = str((phase_dir / "training_readiness.json").resolve())
            phase["metrics"] = training_eval.get("metrics", {})
            if training_eval.get("status") != "passed":
                phase["fail_reasons"].extend(list(training_eval.get("fail_reasons", [])))
        except Exception as exc:
            phase["fail_reasons"].append(f"world-model readiness eval failed: {type(exc).__name__}: {exc}")

    if phase["fail_reasons"]:
        phase["status"] = "failed"
    return _finish_phase(phase)


def _locate_single(path_glob: Path) -> Optional[Path]:
    matches = sorted(path_glob.parent.glob(path_glob.name))
    return matches[0] if len(matches) == 1 else None


def _phase_f_collect_run(
    *,
    manifest: Mapping[str, Any],
    collect_cfg: Mapping[str, Any],
    seed: int,
    run_dir: Path,
    policy_path: Optional[Path] = None,
    frames_override: Optional[int] = None,
    phase_log_name: str = "phase_f_collect.log",
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    python_bin = str(Path(str(manifest.get("python_bin", sys.executable))).expanduser())
    collect_cmd = [
        python_bin,
        str(_rel_repo_path("scripts/collect_sim_lidar_monitor.py")),
        "--env-id",
        str(collect_cfg.get("env_id", "donkey-waveshare-v0")),
        "--sim",
        _manifest_sim_path(manifest, fallback="remote"),
        "--sim-start-delay",
        str(float(collect_cfg.get("sim_start_delay", 8.0))),
        "--policy-path",
        str(policy_path or _rel_repo_path(collect_cfg["policy_path"])),
        "--policy-format",
        str(collect_cfg.get("policy_format", "v16")),
        "--curriculum-phase",
        str(collect_cfg.get("curriculum_phase", "warmup")),
        "--frames",
        str(int(frames_override if frames_override is not None else collect_cfg.get("frames", 800))),
        "--max-episode-steps",
        str(int(collect_cfg.get("max_episode_steps", 640))),
        "--port",
        str(int(collect_cfg.get("port", 9095))),
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
        str(float(collect_cfg.get("lidar_offset_y", 0.40))),
        "--lidar-offset-z",
        str(float(collect_cfg.get("lidar_offset_z", 0.5))),
        "--lidar-rot-x",
        str(float(collect_cfg.get("lidar_rot_x", 0.0))),
        "--seed",
        str(seed),
        "--output-dir",
        str(run_dir),
    ]
    sim2real_json = str(collect_cfg.get("sim2real_json", "") or "").strip()
    if sim2real_json:
        collect_cmd.extend(["--sim2real-json", str(Path(sim2real_json).expanduser())])
    command_result = _run_command(collect_cmd, log_path=run_dir / phase_log_name)
    summary_json = _locate_single(run_dir / "monitor_logs" / "*_summary.json")
    sim_jsonl: Optional[Path] = None
    summary_payload: Dict[str, Any] = {}
    if summary_json and summary_json.exists():
        summary_payload = _load_json(summary_json)
        output_jsonl = summary_payload.get("output_jsonl")
        if output_jsonl:
            sim_jsonl = Path(str(output_jsonl)).expanduser()
    return {
        "command": command_result,
        "run_dir": str(run_dir),
        "summary_json": str(summary_json) if summary_json else None,
        "summary_payload": summary_payload,
        "sim_jsonl": str(sim_jsonl) if sim_jsonl else None,
        "policy_path": str(policy_path or _rel_repo_path(collect_cfg["policy_path"])),
        "curriculum_phase": str(collect_cfg.get("curriculum_phase", "warmup")),
        "frames": int(frames_override if frames_override is not None else collect_cfg.get("frames", 800)),
        "collect_cfg": dict(collect_cfg),
    }


def _phase_f_eval_run(
    *,
    python_bin: str,
    eval_cfg: Mapping[str, Any],
    real_paths: Sequence[str],
    sim_jsonl: Path,
    output_json: Path,
    log_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    thresholds = dict(eval_cfg.get("legacy_thresholds", {}) or {})
    range_bands = list(eval_cfg.get("range_bands", ["0,5", "5,max"]) or [])
    eval_cmd = [
        python_bin,
        str(_rel_repo_path("scripts/eval_lidar_domain_gap.py")),
        "--real-paths",
        *[str(Path(str(x)).expanduser()) for x in real_paths],
        "--sim-paths",
        str(sim_jsonl),
        "--num-sectors",
        str(int(eval_cfg.get("num_sectors", 36))),
        "--fov-deg",
        str(float(eval_cfg.get("fov_deg", 180.0))),
        "--compare-max-range-m",
        str(float(eval_cfg.get("compare_max_range_m", eval_cfg.get("max_range_m", 12.0)))),
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
        str(output_json),
    ]
    if range_bands:
        eval_cmd.extend(["--range-bands", *[str(x) for x in range_bands]])
    command_result = _run_command(eval_cmd, log_path=log_path)
    payload: Dict[str, Any] = {}
    if output_json.exists():
        payload = _load_json(output_json)
    return command_result, payload


def _band_summary(eval_payload: Mapping[str, Any], band_name: str) -> Dict[str, Any]:
    return dict(
        (((eval_payload.get("bands", {}) or {}).get(band_name, {}) or {}).get("overall", {}) or {}).get("summary", {})
        or {}
    )


def _phase_f_primary_gate(eval_payload: Mapping[str, Any], thresholds: Mapping[str, Any]) -> Dict[str, Any]:
    near_thresholds = dict(thresholds.get("range_0_5m", {}) or {})
    overall_thresholds = dict(thresholds.get("overall_0_12m", {}) or {})
    near = _band_summary(eval_payload, "range_0_5m")
    far = _band_summary(eval_payload, "range_5_12m")
    overall = _band_summary(eval_payload, "overall_0_12m")
    band_consistency_gap = abs(
        float(near.get("wasserstein_median", float("inf")))
        - float(far.get("wasserstein_median", float("inf")))
    )
    passed = bool(
        float(near.get("valid_ratio_mae", float("inf"))) <= float(near_thresholds.get("valid_ratio_mae_max", 0.10))
        and float(near.get("wasserstein_median", float("inf"))) <= float(near_thresholds.get("wasserstein_median_max", 0.08))
        and float(near.get("wasserstein_p95", float("inf"))) <= float(near_thresholds.get("wasserstein_p95_max", 0.20))
        and float(overall.get("scene_js_divergence", float("inf"))) <= float(overall_thresholds.get("scene_js_divergence_max", 0.15))
    )
    return {
        "pass": passed,
        "band_consistency_gap": band_consistency_gap,
        "near": near,
        "far": far,
        "overall": overall,
    }


def _phase_f_pose_score(eval_payload: Mapping[str, Any], thresholds: Mapping[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    gate = _phase_f_deployment_gate(eval_payload, thresholds)
    metrics = dict(gate.get("metrics", {}) or {})
    if not metrics:
        return None
    return (
        float(metrics.get("front_min_p50_abs_diff", float("inf"))),
        float(metrics.get("valid_ratio_mean_abs_diff", float("inf"))),
        0.5
        * (
            float(metrics.get("left_gap_p50_abs_diff", float("inf")))
            + float(metrics.get("right_gap_p50_abs_diff", float("inf")))
        ),
        float(metrics.get("front_min_wasserstein", float("inf"))),
    )


def _phase_f_feature_alignment(eval_payload: Mapping[str, Any]) -> Dict[str, Any]:
    return dict((((eval_payload.get("feature_alignment", {}) or {}).get("overall", {}) or {}).get("features", {}) or {}))


def _phase_f_deployment_gate(eval_payload: Mapping[str, Any], thresholds: Mapping[str, Any]) -> Dict[str, Any]:
    feature_thresholds = dict(thresholds.get("deployment_feature_alignment", {}) or {})
    features = _phase_f_feature_alignment(eval_payload)
    front_min = dict(features.get("front_min", {}) or {})
    left_gap = dict(features.get("left_gap", {}) or {})
    right_gap = dict(features.get("right_gap", {}) or {})
    valid_ratio = dict(features.get("valid_ratio", {}) or {})
    metrics = {
        "front_min_p50_abs_diff": float(front_min.get("p50_abs_diff", float("inf"))),
        "left_gap_p50_abs_diff": float(left_gap.get("p50_abs_diff", float("inf"))),
        "right_gap_p50_abs_diff": float(right_gap.get("p50_abs_diff", float("inf"))),
        "valid_ratio_mean_abs_diff": float(valid_ratio.get("mean_abs_diff", float("inf"))),
        "front_min_wasserstein": float(front_min.get("wasserstein", float("inf"))),
        "left_gap_wasserstein": float(left_gap.get("wasserstein", float("inf"))),
        "right_gap_wasserstein": float(right_gap.get("wasserstein", float("inf"))),
        "valid_ratio_wasserstein": float(valid_ratio.get("wasserstein", float("inf"))),
    }
    passed = bool(
        metrics["front_min_p50_abs_diff"] <= float(feature_thresholds.get("front_min_p50_abs_diff_max", 1.00))
        and metrics["left_gap_p50_abs_diff"] <= float(feature_thresholds.get("left_gap_p50_abs_diff_max", 1.00))
        and metrics["right_gap_p50_abs_diff"] <= float(feature_thresholds.get("right_gap_p50_abs_diff_max", 1.00))
        and metrics["valid_ratio_mean_abs_diff"] <= float(feature_thresholds.get("valid_ratio_mean_abs_diff_max", 0.50))
    )
    return {
        "pass": passed,
        "thresholds": feature_thresholds,
        "metrics": metrics,
        "features": features,
    }


def _phase_f(
    manifest: Mapping[str, Any],
    output_root: Path,
) -> Dict[str, Any]:
    phase = _phase_result("passed")
    phase_cfg = dict(manifest.get("phase_f", {}) or {})
    collect_cfg = dict(phase_cfg.get("collect", {}) or {})
    eval_cfg = dict(phase_cfg.get("eval", {}) or {})
    phase_dir = output_root / "phase_f_deployment"
    phase_dir.mkdir(parents=True, exist_ok=True)
    python_bin = str(Path(str(manifest.get("python_bin", sys.executable))).expanduser())
    seed = int(manifest.get("seed", 123))
    thresholds = dict(eval_cfg.get("thresholds", {}) or {})
    motion_thresholds = dict(thresholds.get("motion", {}) or {})
    motion_enough_thresholds = dict(thresholds.get("motion_enough", motion_thresholds) or {})
    full_collect_frames = int(collect_cfg.get("frames", 1200))
    motion_precheck_frames = int(phase_cfg.get("motion_precheck_frames", min(400, full_collect_frames)))
    pose_sweep_precheck_frames = int(phase_cfg.get("pose_sweep_precheck_frames", min(300, full_collect_frames)))
    pose_sweep_top_k = max(1, int(phase_cfg.get("pose_sweep_top_k", 3)))

    primary_real_paths = [
        str(Path(str(x)).expanduser()) for x in eval_cfg.get("real_paths_primary", eval_cfg.get("real_paths", []))
    ]
    stress_real_paths = [str(Path(str(x)).expanduser()) for x in eval_cfg.get("real_paths_stress", [])]
    if not primary_real_paths:
        phase["fail_reasons"].append("phase_f eval.real_paths_primary is empty")
        phase["status"] = "failed"
        return _finish_phase(phase)

    real_motion_csv_raw = eval_cfg.get("real_motion_csv_primary")
    if real_motion_csv_raw is None:
        real_motion_csv = _derive_real_motion_csv(primary_real_paths[0])
    else:
        real_motion_csv = Path(str(real_motion_csv_raw)).expanduser()
    if not real_motion_csv.exists():
        phase["fail_reasons"].append(f"real motion CSV not found: {real_motion_csv}")
        phase["status"] = "failed"
        return _finish_phase(phase)

    phase["artifacts"] = {
        "phase_dir": str(phase_dir),
        "real_motion_csv_primary": str(real_motion_csv),
    }

    def _sim_jsonl_from_run(run: Mapping[str, Any]) -> Optional[Path]:
        raw = run.get("sim_jsonl")
        if not raw:
            return None
        path = Path(str(raw)).expanduser()
        return path if path.exists() else None

    baseline_precheck_run = _phase_f_collect_run(
        manifest=manifest,
        collect_cfg=collect_cfg,
        seed=seed,
        run_dir=phase_dir / "baseline_motion_precheck",
        frames_override=motion_precheck_frames,
        phase_log_name="phase_f_collect_motion_precheck.log",
    )
    phase["commands"].append(baseline_precheck_run["command"])
    baseline_precheck_jsonl = _sim_jsonl_from_run(baseline_precheck_run)
    if int(baseline_precheck_run["command"]["returncode"]) != 0 or baseline_precheck_jsonl is None:
        phase["fail_reasons"].append("baseline motion precheck failed or produced no JSONL")
        phase["status"] = "failed"
        return _finish_phase(phase)

    baseline_motion_report = _build_motion_report(
        real_csv_path=real_motion_csv,
        sim_jsonl_path=baseline_precheck_jsonl,
        thresholds=motion_thresholds,
    )
    _write_json(phase_dir / "motion_match_report_baseline.json", baseline_motion_report)

    selected_policy_path = Path(str(baseline_precheck_run["policy_path"])).expanduser()
    selected_precheck_run = baseline_precheck_run
    selected_motion_report = baseline_motion_report
    driver_profile_runs: List[Dict[str, Any]] = [
        {
            "policy_path": str(selected_policy_path),
            "run_dir": str(baseline_precheck_run["run_dir"]),
            "summary_json": baseline_precheck_run.get("summary_json"),
            "sim_jsonl": baseline_precheck_run.get("sim_jsonl"),
            "returncode": int(baseline_precheck_run["command"]["returncode"]),
            "motion_report": baseline_motion_report,
            "source": "baseline_motion_precheck",
        }
    ]

    candidate_paths = [selected_policy_path]
    candidate_paths.extend(Path(str(_rel_repo_path(x))).expanduser() for x in list(phase_cfg.get("driver_profile_candidates", []) or []))
    dedup_candidates: List[Path] = []
    seen_candidates: set[str] = set()
    for path in candidate_paths:
        key = str(path.resolve())
        if key in seen_candidates:
            continue
        seen_candidates.add(key)
        dedup_candidates.append(path)

    for idx, policy_path in enumerate(dedup_candidates):
        if policy_path.resolve() == selected_policy_path.resolve():
            continue
        run = _phase_f_collect_run(
            manifest=manifest,
            collect_cfg=collect_cfg,
            seed=seed,
            run_dir=phase_dir / "driver_profile_sweep" / f"candidate_{idx}",
            policy_path=policy_path,
            frames_override=motion_precheck_frames,
            phase_log_name="phase_f_collect_driver_profile.log",
        )
        phase["commands"].append(run["command"])
        sim_jsonl = _sim_jsonl_from_run(run)
        motion_report: Dict[str, Any] = {}
        if int(run["command"]["returncode"]) == 0 and sim_jsonl is not None:
            motion_report = _build_motion_report(
                real_csv_path=real_motion_csv,
                sim_jsonl_path=sim_jsonl,
                thresholds=motion_thresholds,
            )
            _write_json(Path(run["run_dir"]) / "motion_match_report.json", motion_report)
        driver_profile_runs.append(
            {
                "policy_path": str(policy_path),
                "run_dir": str(run["run_dir"]),
                "summary_json": run.get("summary_json"),
                "sim_jsonl": run.get("sim_jsonl"),
                "returncode": int(run["command"]["returncode"]),
                "motion_report": motion_report,
                "source": "driver_profile_sweep",
            }
        )

    successful_motion_runs = [run for run in driver_profile_runs if dict(run.get("motion_report", {}) or {})]
    if not successful_motion_runs:
        phase["artifacts"]["motion_match_report_json"] = str((phase_dir / "motion_match_report_baseline.json").resolve())
        phase["metrics"] = {
            "motion_confound_pass": False,
            "motion_enough_pass": False,
            "selected_policy_path": None,
            "motion_precheck_frames": motion_precheck_frames,
            "motion_score": list(_motion_score_tuple(baseline_motion_report)),
        }
        phase["fail_reasons"].append("driver-profile sweep produced no usable motion reports")
        phase["status"] = "failed"
        return _finish_phase(phase)

    strict_runs = [
        run for run in successful_motion_runs if bool((run.get("motion_report") or {}).get("pass", False))
    ]
    enough_runs = [
        run
        for run in successful_motion_runs
        if _motion_pass_with_thresholds(run.get("motion_report") or {}, motion_enough_thresholds)
    ]
    candidate_pool = strict_runs or enough_runs or successful_motion_runs
    best_run = min(candidate_pool, key=lambda run: _motion_score_tuple(run.get("motion_report") or {}))

    driver_profile_summary = {
        "motion_precheck_frames": motion_precheck_frames,
        "baseline_policy_path": str(selected_policy_path),
        "selected_policy_path": best_run["policy_path"],
        "runs": [
            {
                "source": run.get("source"),
                "policy_path": run.get("policy_path"),
                "sim_jsonl": run.get("sim_jsonl"),
                "summary_json": run.get("summary_json"),
                "returncode": int(run.get("returncode", 1)),
                "motion_pass_strict": bool((run.get("motion_report") or {}).get("pass", False)),
                "motion_pass_enough": _motion_pass_with_thresholds(run.get("motion_report") or {}, motion_enough_thresholds),
                "motion_score": list(_motion_score_tuple(run.get("motion_report") or {})),
            }
            for run in driver_profile_runs
        ],
    }
    _write_json(phase_dir / "driver_profile_sweep_summary.json", driver_profile_summary)
    phase["artifacts"]["driver_profile_sweep_json"] = str((phase_dir / "driver_profile_sweep_summary.json").resolve())
    selected_policy_path = Path(str(best_run["policy_path"])).expanduser()
    selected_precheck_run = best_run
    selected_motion_report = dict(best_run["motion_report"])

    baseline_run = _phase_f_collect_run(
        manifest=manifest,
        collect_cfg=collect_cfg,
        seed=seed,
        run_dir=phase_dir / "baseline_selected_driver",
        policy_path=selected_policy_path,
        frames_override=full_collect_frames,
        phase_log_name="phase_f_collect_baseline.log",
    )
    phase["commands"].append(baseline_run["command"])
    baseline_sim_jsonl = _sim_jsonl_from_run(baseline_run)
    if int(baseline_run["command"]["returncode"]) != 0 or baseline_sim_jsonl is None:
        phase["fail_reasons"].append("selected-driver baseline collection failed or produced no JSONL")
        phase["status"] = "failed"
        return _finish_phase(phase)

    full_motion_report = _build_motion_report(
        real_csv_path=real_motion_csv,
        sim_jsonl_path=baseline_sim_jsonl,
        thresholds=motion_thresholds,
    )
    full_motion_enough = _motion_pass_with_thresholds(full_motion_report, motion_enough_thresholds)
    _write_json(phase_dir / "motion_match_report.json", full_motion_report)
    phase["artifacts"]["motion_match_report_json"] = str((phase_dir / "motion_match_report.json").resolve())

    primary_eval_json = _output_path(
        output_root,
        eval_cfg.get("output_json_primary", "phase_f_deployment/lidar_domain_gap_eval_primary.json"),
    )
    primary_eval_result, primary_eval_payload = _phase_f_eval_run(
        python_bin=python_bin,
        eval_cfg=eval_cfg,
        real_paths=primary_real_paths,
        sim_jsonl=baseline_sim_jsonl,
        output_json=primary_eval_json,
        log_path=phase_dir / "phase_f_eval_primary.log",
    )
    phase["commands"].append(primary_eval_result)
    if int(primary_eval_result["returncode"]) != 0:
        phase["fail_reasons"].append(f"primary LiDAR domain-gap eval failed with return code {primary_eval_result['returncode']}")
        phase["status"] = "failed"
        return _finish_phase(phase)

    stress_eval_json = _output_path(
        output_root,
        eval_cfg.get("output_json_stress", "phase_f_deployment/lidar_domain_gap_eval_stress.json"),
    )
    if stress_real_paths:
        stress_eval_result, _stress_eval_payload = _phase_f_eval_run(
            python_bin=python_bin,
            eval_cfg=eval_cfg,
            real_paths=stress_real_paths,
            sim_jsonl=baseline_sim_jsonl,
            output_json=stress_eval_json,
            log_path=phase_dir / "phase_f_eval_stress.log",
        )
        phase["commands"].append(stress_eval_result)
        if int(stress_eval_result["returncode"]) != 0:
            phase["fail_reasons"].append(f"stress LiDAR domain-gap eval failed with return code {stress_eval_result['returncode']}")
            phase["status"] = "failed"
            return _finish_phase(phase)

    primary_gate = _phase_f_primary_gate(primary_eval_payload, thresholds)
    deployment_gate = _phase_f_deployment_gate(primary_eval_payload, thresholds)
    phase["artifacts"].update(
        {
            "collect_summary_json": str(baseline_run.get("summary_json")),
            "sim_raw_jsonl": str(baseline_sim_jsonl),
            "primary_eval_json": str(primary_eval_json.resolve()),
            "stress_eval_json": str(stress_eval_json.resolve()) if stress_real_paths else None,
            "selected_policy_path": str(selected_policy_path),
        }
    )
    phase["metrics"] = {
        "motion_confound_pass": bool(full_motion_report.get("pass", False)),
        "motion_enough_pass": bool(full_motion_enough),
        "selected_policy_path": str(selected_policy_path),
        "motion_precheck_frames": motion_precheck_frames,
        "motion_score": list(_motion_score_tuple(full_motion_report)),
        "primary_gate_pass": bool(primary_gate["pass"]),
        "deployment_gate_pass": bool(deployment_gate["pass"]),
        "range_0_5m": primary_gate["near"],
        "range_5_12m": primary_gate["far"],
        "overall_0_12m": primary_gate["overall"],
        "band_consistency_gap": float(primary_gate["band_consistency_gap"]),
        "deployment_feature_alignment": dict(deployment_gate.get("metrics", {}) or {}),
    }
    if bool(full_motion_enough) and bool(deployment_gate["pass"]):
        return _finish_phase(phase)
    if not bool(full_motion_enough):
        phase["fail_reasons"].append("selected-driver full baseline is not motion-enough for deployment-oriented LiDAR pose sweep")
        phase["status"] = "failed"
        return _finish_phase(phase)

    base_pose = {
        "offset_y": float(collect_cfg.get("lidar_offset_y", 0.40)),
        "offset_z": float(collect_cfg.get("lidar_offset_z", 0.50)),
        "rot_x": float(collect_cfg.get("lidar_rot_x", 0.0)),
        "lidar_noise": float(collect_cfg.get("lidar_noise", 0.0)),
        "near_clip": float(collect_cfg.get("lidar_near_clip_m", 0.18)),
    }
    pose_cfg = dict(phase_cfg.get("pose_sweep", {}) or {})
    pose_values = {
        "offset_y": [float(x) for x in pose_cfg.get("offset_y", [0.35, 0.40, 0.45])],
        "offset_z": [float(x) for x in pose_cfg.get("offset_z", [0.45, 0.50, 0.55])],
        "rot_x": [float(x) for x in pose_cfg.get("rot_x", [-2.0, 0.0, 2.0])],
        "lidar_noise": [float(x) for x in pose_cfg.get("lidar_noise", [0.0, 0.02])],
        "near_clip": [float(x) for x in pose_cfg.get("near_clip", [0.18, 0.22])],
    }
    pose_candidates: List[Dict[str, float]] = [base_pose]
    seen_pose_keys = {
        (
            base_pose["offset_y"],
            base_pose["offset_z"],
            base_pose["rot_x"],
            base_pose["lidar_noise"],
            base_pose["near_clip"],
        )
    }
    for offset_y in pose_values["offset_y"]:
        for offset_z in pose_values["offset_z"]:
            for rot_x in pose_values["rot_x"]:
                for lidar_noise in pose_values["lidar_noise"]:
                    for near_clip in pose_values["near_clip"]:
                        key = (offset_y, offset_z, rot_x, lidar_noise, near_clip)
                        if key in seen_pose_keys:
                            continue
                        seen_pose_keys.add(key)
                        pose_candidates.append(
                            {
                                "offset_y": offset_y,
                                "offset_z": offset_z,
                                "rot_x": rot_x,
                                "lidar_noise": lidar_noise,
                                "near_clip": near_clip,
                            }
                        )

    pose_precheck_runs: List[Dict[str, Any]] = [
        {
            "source": "baseline_selected_driver",
            "mode": "full_validation",
            "pose": base_pose,
            "policy_path": str(selected_policy_path),
            "run_dir": str(baseline_run["run_dir"]),
            "summary_json": baseline_run.get("summary_json"),
            "sim_jsonl": baseline_run.get("sim_jsonl"),
            "primary_eval_json": str(primary_eval_json),
            "stress_eval_json": str(stress_eval_json) if stress_real_paths else None,
            "primary_eval_gate": primary_gate,
            "deployment_gate": deployment_gate,
            "pose_score": _phase_f_pose_score(primary_eval_payload, thresholds),
            "returncode": int(baseline_run["command"]["returncode"]),
        }
    ]

    for idx, pose in enumerate(pose_candidates[1:], start=1):
        collect_cfg_pose = dict(collect_cfg)
        collect_cfg_pose["lidar_offset_y"] = float(pose["offset_y"])
        collect_cfg_pose["lidar_offset_z"] = float(pose["offset_z"])
        collect_cfg_pose["lidar_rot_x"] = float(pose["rot_x"])
        collect_cfg_pose["lidar_noise"] = float(pose["lidar_noise"])
        collect_cfg_pose["lidar_near_clip_m"] = float(pose["near_clip"])
        run = _phase_f_collect_run(
            manifest=manifest,
            collect_cfg=collect_cfg_pose,
            seed=seed,
            run_dir=phase_dir / "pose_sweep_precheck" / f"candidate_{idx}",
            policy_path=selected_policy_path,
            frames_override=pose_sweep_precheck_frames,
            phase_log_name="phase_f_collect_pose_precheck.log",
        )
        phase["commands"].append(run["command"])
        sim_jsonl = _sim_jsonl_from_run(run)
        pose_entry: Dict[str, Any] = {
            "source": "pose_sweep_precheck",
            "mode": "precheck",
            "pose": pose,
            "policy_path": str(selected_policy_path),
            "run_dir": str(run["run_dir"]),
            "summary_json": run.get("summary_json"),
            "sim_jsonl": run.get("sim_jsonl"),
            "returncode": int(run["command"]["returncode"]),
            "primary_eval_gate": {"pass": False},
            "deployment_gate": {"pass": False},
            "pose_score": None,
        }
        if int(run["command"]["returncode"]) == 0 and sim_jsonl is not None:
            pose_primary_json = Path(run["run_dir"]) / "lidar_domain_gap_eval_primary_precheck.json"
            pose_primary_result, pose_primary_payload = _phase_f_eval_run(
                python_bin=python_bin,
                eval_cfg=eval_cfg,
                real_paths=primary_real_paths,
                sim_jsonl=sim_jsonl,
                output_json=pose_primary_json,
                log_path=Path(run["run_dir"]) / "phase_f_eval_primary_precheck.log",
            )
            phase["commands"].append(pose_primary_result)
            if int(pose_primary_result["returncode"]) == 0:
                pose_primary_gate = _phase_f_primary_gate(pose_primary_payload, thresholds)
                pose_deployment_gate = _phase_f_deployment_gate(pose_primary_payload, thresholds)
                pose_entry["primary_eval_json"] = str(pose_primary_json)
                pose_entry["primary_eval_gate"] = pose_primary_gate
                pose_entry["deployment_gate"] = pose_deployment_gate
                pose_entry["pose_score"] = _phase_f_pose_score(pose_primary_payload, thresholds)
        pose_precheck_runs.append(pose_entry)

    _write_json(phase_dir / "pose_sweep_precheck_summary.json", {"runs": pose_precheck_runs})
    phase["artifacts"]["pose_sweep_precheck_json"] = str((phase_dir / "pose_sweep_precheck_summary.json").resolve())

    eligible_prechecks = [run for run in pose_precheck_runs if run.get("pose_score") is not None]
    if not eligible_prechecks:
        phase["fail_reasons"].append("pose sweep precheck produced no candidate with usable deployment-oriented feature alignment metrics")
        phase["status"] = "failed"
        return _finish_phase(phase)

    shortlist = sorted(
        eligible_prechecks,
        key=lambda run: tuple(run.get("pose_score") or (float("inf"),) * 4),
    )[:pose_sweep_top_k]
    shortlist_keys = {
        (
            float((run.get("pose") or {}).get("offset_y", float("nan"))),
            float((run.get("pose") or {}).get("offset_z", float("nan"))),
            float((run.get("pose") or {}).get("rot_x", float("nan"))),
            float((run.get("pose") or {}).get("lidar_noise", float("nan"))),
            float((run.get("pose") or {}).get("near_clip", float("nan"))),
        )
        for run in shortlist
    }

    pose_validated_runs: List[Dict[str, Any]] = []
    for run in pose_precheck_runs:
        pose = dict(run.get("pose", {}) or {})
        pose_key = (
            float(pose.get("offset_y", float("nan"))),
            float(pose.get("offset_z", float("nan"))),
            float(pose.get("rot_x", float("nan"))),
            float(pose.get("lidar_noise", float("nan"))),
            float(pose.get("near_clip", float("nan"))),
        )
        if run.get("mode") == "full_validation":
            pose_validated_runs.append(run)
            continue
        if pose_key not in shortlist_keys:
            continue
        collect_cfg_pose = dict(collect_cfg)
        collect_cfg_pose["lidar_offset_y"] = float(pose["offset_y"])
        collect_cfg_pose["lidar_offset_z"] = float(pose["offset_z"])
        collect_cfg_pose["lidar_rot_x"] = float(pose["rot_x"])
        collect_cfg_pose["lidar_noise"] = float(pose["lidar_noise"])
        collect_cfg_pose["lidar_near_clip_m"] = float(pose["near_clip"])
        validated_run = _phase_f_collect_run(
            manifest=manifest,
            collect_cfg=collect_cfg_pose,
            seed=seed,
            run_dir=phase_dir / "pose_sweep_validated" / f"candidate_{len(pose_validated_runs)}",
            policy_path=selected_policy_path,
            frames_override=full_collect_frames,
            phase_log_name="phase_f_collect_pose_validated.log",
        )
        phase["commands"].append(validated_run["command"])
        sim_jsonl = _sim_jsonl_from_run(validated_run)
        validated_entry: Dict[str, Any] = {
            "source": "pose_sweep_validated",
            "mode": "full_validation",
            "pose": pose,
            "policy_path": str(selected_policy_path),
            "run_dir": str(validated_run["run_dir"]),
            "summary_json": validated_run.get("summary_json"),
            "sim_jsonl": validated_run.get("sim_jsonl"),
            "returncode": int(validated_run["command"]["returncode"]),
            "primary_eval_gate": {"pass": False},
            "deployment_gate": {"pass": False},
            "pose_score": None,
        }
        if int(validated_run["command"]["returncode"]) == 0 and sim_jsonl is not None:
            pose_primary_json = Path(validated_run["run_dir"]) / "lidar_domain_gap_eval_primary.json"
            pose_primary_result, pose_primary_payload = _phase_f_eval_run(
                python_bin=python_bin,
                eval_cfg=eval_cfg,
                real_paths=primary_real_paths,
                sim_jsonl=sim_jsonl,
                output_json=pose_primary_json,
                log_path=Path(validated_run["run_dir"]) / "phase_f_eval_primary.log",
            )
            phase["commands"].append(pose_primary_result)
            if int(pose_primary_result["returncode"]) == 0:
                validated_entry["primary_eval_json"] = str(pose_primary_json)
                pose_primary_gate = _phase_f_primary_gate(pose_primary_payload, thresholds)
                pose_deployment_gate = _phase_f_deployment_gate(pose_primary_payload, thresholds)
                validated_entry["primary_eval_gate"] = pose_primary_gate
                validated_entry["deployment_gate"] = pose_deployment_gate
                validated_entry["pose_score"] = _phase_f_pose_score(pose_primary_payload, thresholds)
                if stress_real_paths:
                    pose_stress_json = Path(validated_run["run_dir"]) / "lidar_domain_gap_eval_stress.json"
                    pose_stress_result, _ = _phase_f_eval_run(
                        python_bin=python_bin,
                        eval_cfg=eval_cfg,
                        real_paths=stress_real_paths,
                        sim_jsonl=sim_jsonl,
                        output_json=pose_stress_json,
                        log_path=Path(validated_run["run_dir"]) / "phase_f_eval_stress.log",
                    )
                    phase["commands"].append(pose_stress_result)
                    if int(pose_stress_result["returncode"]) == 0:
                        validated_entry["stress_eval_json"] = str(pose_stress_json)
        pose_validated_runs.append(validated_entry)

    _write_json(
        phase_dir / "pose_sweep_summary.json",
        {
            "precheck_frames": pose_sweep_precheck_frames,
            "full_validation_frames": full_collect_frames,
            "top_k": pose_sweep_top_k,
            "shortlist": [dict(run.get("pose", {}) or {}) for run in shortlist],
            "runs": pose_validated_runs,
        },
    )
    phase["artifacts"]["pose_sweep_summary_json"] = str((phase_dir / "pose_sweep_summary.json").resolve())

    eligible_pose_runs = [run for run in pose_validated_runs if run.get("pose_score") is not None]
    if not eligible_pose_runs:
        phase["fail_reasons"].append("validated pose sweep produced no candidate with usable deployment-oriented feature alignment metrics")
        phase["status"] = "failed"
        return _finish_phase(phase)

    best_pose_run = min(eligible_pose_runs, key=lambda run: tuple(run.get("pose_score") or (float("inf"),) * 4))
    best_pose_gate = dict(best_pose_run.get("primary_eval_gate", {}) or {})
    best_pose_deployment_gate = dict(best_pose_run.get("deployment_gate", {}) or {})
    phase["metrics"].update(
        {
            "pose_sweep_precheck_frames": pose_sweep_precheck_frames,
            "pose_sweep_top_k": pose_sweep_top_k,
            "selected_pose": dict(best_pose_run.get("pose", {}) or {}),
            "selected_pose_score": list(best_pose_run.get("pose_score") or []),
            "selected_pose_gate_pass": bool(best_pose_deployment_gate.get("pass", False)),
        }
    )
    phase["artifacts"]["selected_pose_primary_eval_json"] = best_pose_run.get("primary_eval_json")
    phase["artifacts"]["selected_pose_stress_eval_json"] = best_pose_run.get("stress_eval_json")

    if not bool(best_pose_deployment_gate.get("pass", False)):
        phase["fail_reasons"].append("real-first blocked by deployment-oriented LiDAR feature alignment after driver-profile sweep and pose sweep")
        phase["status"] = "failed"
        return _finish_phase(phase)

    phase["metrics"].update(
        {
            "primary_gate_pass": True,
            "deployment_gate_pass": True,
            "range_0_5m": dict(best_pose_gate.get("near", {}) or {}),
            "range_5_12m": dict(best_pose_gate.get("far", {}) or {}),
            "overall_0_12m": dict(best_pose_gate.get("overall", {}) or {}),
            "band_consistency_gap": float(best_pose_gate.get("band_consistency_gap", float("nan"))),
            "deployment_feature_alignment": dict(best_pose_deployment_gate.get("metrics", {}) or {}),
        }
    )
    phase["artifacts"]["collect_summary_json"] = best_pose_run.get("summary_json")
    phase["artifacts"]["sim_raw_jsonl"] = best_pose_run.get("sim_jsonl")
    return _finish_phase(phase)


def _launch_decision(report: Dict[str, Any]) -> Tuple[bool, bool, bool, str, str]:
    phases = report.get("phases", {})
    formal_train_ready = all(
        phases.get(name, {}).get("status") == "passed"
        for name in ("phase_a", "phase_b", "phase_c", "phase_d")
    )
    deployment_ready = phases.get("phase_f", {}).get("status") == "passed"
    runtime_ready = False
    if formal_train_ready:
        if deployment_ready:
            reason = "formal_train_ready=true; deployment_ready=true; runtime log-only validation still not assessed"
        else:
            reason = "formal_train_ready=true; deployment_ready=false (real-first blocked by LiDAR domain gap)"
    else:
        failing = [name for name in ("phase_a", "phase_b", "phase_c", "phase_d") if phases.get(name, {}).get("status") != "passed"]
        reason = f"formal_train_ready=false due to failed phases: {failing}"
    return (
        formal_train_ready,
        runtime_ready,
        deployment_ready,
        "pass" if formal_train_ready else "fail",
        reason,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run staged V17 formal readiness checks")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str((SCRIPT_DIR / "v17_formal_readiness_manifest.json").resolve()),
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["phase_a", "phase_b", "phase_c", "phase_d", "phase_f"],
        default=["phase_a", "phase_b", "phase_c", "phase_d", "phase_f"],
    )
    args = parser.parse_args()

    manifest = _load_json(args.manifest)
    output_root = Path(args.output_dir).expanduser().resolve()
    _ensure_empty_output_dir(output_root)
    selected_phases = set(args.phases)
    if "phase_a" not in selected_phases:
        raise ValueError("phase_a is required whenever running the readiness runner")
    if "phase_d" in selected_phases and "phase_c" not in selected_phases:
        raise ValueError("phase_d requires phase_c to be selected")

    report: Dict[str, Any] = {
        "manifest_path": str(Path(args.manifest).expanduser().resolve()),
        "manifest_id": manifest.get("manifest_id"),
        "launch_target": manifest.get("launch_target"),
        "started_at": _now_iso(),
        "finished_at": None,
        "repo_root": str(REPO_ROOT),
        "phases": {},
        "formal_train_ready": False,
        "runtime_ready": False,
        "deployment_ready": False,
        "launch_decision": "fail",
        "launch_reason": "",
    }

    resolved_manifest_path = output_root / "readiness_manifest_resolved.json"
    _write_json(resolved_manifest_path, manifest)

    try:
        report["phases"]["phase_a"] = _phase_a(manifest=manifest, output_root=output_root, report=report)
        if report["phases"]["phase_a"]["status"] == "passed":
            if "phase_b" in selected_phases:
                report["phases"]["phase_b"] = _phase_b(manifest=manifest, output_root=output_root)
            else:
                skipped = _phase_result("skipped")
                skipped["fail_reasons"].append("phase not selected")
                report["phases"]["phase_b"] = _finish_phase(skipped)

            if "phase_c" in selected_phases:
                report["phases"]["phase_c"] = _phase_c(manifest=manifest, output_root=output_root)
            else:
                skipped = _phase_result("skipped")
                skipped["fail_reasons"].append("phase not selected")
                report["phases"]["phase_c"] = _finish_phase(skipped)

            merged_npz = _output_path(
                output_root,
                manifest.get("phase_c", {}).get("merged_output_npz", "phase_c_dataset/wm_dataset_mix_v1.npz"),
            )
            if "phase_d" in selected_phases:
                if report["phases"]["phase_c"]["status"] == "passed":
                    report["phases"]["phase_d"] = _phase_d(
                        manifest=manifest,
                        output_root=output_root,
                        merged_npz=merged_npz,
                    )
                else:
                    skipped = _phase_result("skipped")
                    skipped["fail_reasons"].append("skipped because phase_c failed")
                    report["phases"]["phase_d"] = _finish_phase(skipped)
            else:
                skipped = _phase_result("skipped")
                skipped["fail_reasons"].append("phase not selected")
                report["phases"]["phase_d"] = _finish_phase(skipped)

            if "phase_f" in selected_phases:
                report["phases"]["phase_f"] = _phase_f(manifest=manifest, output_root=output_root)
            else:
                skipped = _phase_result("skipped")
                skipped["fail_reasons"].append("phase not selected")
                report["phases"]["phase_f"] = _finish_phase(skipped)
        else:
            for name in ("phase_b", "phase_c", "phase_d", "phase_f"):
                skipped = _phase_result("skipped")
                skipped["fail_reasons"].append("skipped because phase_a failed")
                report["phases"][name] = _finish_phase(skipped)
    except Exception as exc:
        crash_phase = _phase_result("failed")
        crash_phase["fail_reasons"].append(f"runner exception: {type(exc).__name__}: {exc}")
        crash_phase["metrics"]["traceback"] = traceback.format_exc()
        report["phases"]["phase_runner"] = _finish_phase(crash_phase)

    formal_train_ready, runtime_ready, deployment_ready, launch_decision, launch_reason = _launch_decision(report)
    report["formal_train_ready"] = formal_train_ready
    report["runtime_ready"] = runtime_ready
    report["deployment_ready"] = deployment_ready
    report["launch_decision"] = launch_decision
    report["launch_reason"] = launch_reason
    report["finished_at"] = _now_iso()

    report_path = output_root / "readiness_report.json"
    _write_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
