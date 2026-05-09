#!/usr/bin/env python3
"""
Collect one simulator raw-LiDAR monitor log in a real-monitor-like JSONL format.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from sb3_contrib import RecurrentPPO
except Exception:
    RecurrentPPO = None

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import export_world_model_dataset as wm_export  # noqa: E402


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return int(default)


def _find_wrapper_by_class_name(root: Any, class_name: str) -> Optional[Any]:
    current = getattr(root, "active_env", root)
    seen: set[int] = set()
    while current is not None:
        obj_id = id(current)
        if obj_id in seen:
            break
        seen.add(obj_id)
        if current.__class__.__name__ == class_name:
            return current
        current = getattr(current, "env", None)
    return None


def _resolve_logged_final_action(
    *,
    info: Dict[str, Any],
    action: np.ndarray,
    sim2real_wrapper: Optional[Any],
) -> Tuple[float, float, Optional[float], Optional[float], bool]:
    pre_angle = _safe_float(
        info.get("safety/steer_exec", action[0] if np.asarray(action).size > 0 else 0.0),
        0.0,
    )
    pre_throttle = _safe_float(info.get("ctrl/throttle_pi", 0.0), 0.0)
    if sim2real_wrapper is None:
        return pre_angle, pre_throttle, None, None, False
    transformed = getattr(sim2real_wrapper, "last_transformed_action", None)
    try:
        transformed_arr = np.asarray(transformed, dtype=np.float32).reshape(-1)
    except Exception:
        transformed_arr = np.zeros((0,), dtype=np.float32)
    if transformed_arr.size >= 2 and np.isfinite(transformed_arr[0]) and np.isfinite(transformed_arr[1]):
        return float(transformed_arr[0]), float(transformed_arr[1]), pre_angle, pre_throttle, True
    return pre_angle, pre_throttle, None, None, False


def _resolve_steps_since_new_scan(info: Dict[str, Any]) -> int:
    if "lidar_steps_since_new_scan" in info:
        return max(0, _safe_int(info.get("lidar_steps_since_new_scan", 0.0), 0))
    return max(0, _safe_int(_safe_float(info.get("lidar_steps_since_new_scan_norm", 0.0), 0.0) * 4.0, 0))


def _resolve_repeat_count(info: Dict[str, Any]) -> int:
    if "lidar_repeat_count" in info:
        return max(1, _safe_int(info.get("lidar_repeat_count", 1.0), 1))
    return max(1, _safe_int(_safe_float(info.get("lidar_repeat_count_norm", 0.0), 0.0) * 4.0 + 1.0, 1))


def _sanitize_lidar_packet(value: Any) -> list[Dict[str, float]]:
    try:
        items = list(value) if value is not None else []
    except Exception:
        return []
    out: list[Dict[str, float]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            rx = float(item["rx"])
            dist = float(item["d"])
        except Exception:
            continue
        if not np.isfinite(rx) or not np.isfinite(dist):
            continue
        packet_point: Dict[str, float] = {
            "rx": float(rx),
            "d": float(dist),
        }
        ry = item.get("ry")
        try:
            if ry is not None and np.isfinite(float(ry)):
                packet_point["ry"] = float(ry)
        except Exception:
            pass
        out.append(packet_point)
    return out


def _build_lidar_record(
    raw_lidar: np.ndarray,
    lidar_packet: Any,
    info: Dict[str, Any],
    stamp: float,
    control_dt: float,
    near_clip_m: float,
    max_range_m: float,
    frame_idx: int,
) -> Dict[str, Any]:
    raw = np.asarray(raw_lidar, dtype=np.float32).reshape(-1)
    beam_count = int(raw.size)
    packet = _sanitize_lidar_packet(lidar_packet)
    angle_min = -0.5 * np.pi
    angle_max = 0.5 * np.pi
    angle_increment = (angle_max - angle_min) / max(1, beam_count - 1)
    valid_mask = np.isfinite(raw) & (raw >= float(near_clip_m))
    valid_vals = raw[valid_mask]
    packet_valid = np.asarray(
        [float(point["d"]) for point in packet if float(point["d"]) >= float(near_clip_m)],
        dtype=np.float32,
    ).reshape(-1)
    steps_since_new_scan = _resolve_steps_since_new_scan(info)
    repeat_count = _resolve_repeat_count(info)
    packet_rx = np.asarray([float(point["rx"]) for point in packet], dtype=np.float32).reshape(-1)
    source = "sim_packet" if packet else "sim_array"
    return {
        "source": source,
        "angle_min": float(angle_min),
        "angle_max": float(angle_max),
        "angle_increment": float(angle_increment),
        "frame_count": int(frame_idx),
        "intensities": [],
        "nearest_min": (
            float(np.min(packet_valid))
            if packet_valid.size
            else (float(np.min(valid_vals)) if valid_vals.size else float(max_range_m))
        ),
        "points_total": int(packet_rx.size) if packet else int(beam_count),
        "range_max": float(max_range_m),
        "range_min": float(near_clip_m),
        "ranges": raw.astype(float).tolist(),
        "packet": packet,
        "packet_size": int(packet_rx.size),
        "packet_rx_min": float(np.min(packet_rx)) if packet_rx.size else None,
        "packet_rx_max": float(np.max(packet_rx)) if packet_rx.size else None,
        "scan_age_ms": float(steps_since_new_scan * control_dt * 1000.0),
        "stamp": float(stamp),
        "valid_points": int(packet_valid.size) if packet else int(np.sum(valid_mask)),
        "is_new_scan": float(_safe_float(info.get("lidar_is_new_scan", 0.0), 0.0)),
        "steps_since_new_scan": int(steps_since_new_scan),
        "repeat_count": int(repeat_count),
    }


def _init_summary() -> Dict[str, Any]:
    return {
        "frames": 0,
        "episodes": 0,
        "new_scan_count": 0,
        "valid_ratio_values": [],
        "valid_points_values": [],
        "scan_age_ms_values": [],
        "speed_values": [],
    }


def _update_summary(summary: Dict[str, Any], lidar_record: Dict[str, Any], info: Dict[str, Any]) -> None:
    total = max(1, int(lidar_record.get("points_total", 0)))
    valid_points = int(lidar_record.get("valid_points", 0))
    summary["frames"] += 1
    summary["new_scan_count"] += int(float(lidar_record.get("is_new_scan", 0.0)) >= 0.5)
    summary["valid_ratio_values"].append(float(valid_points / total))
    summary["valid_points_values"].append(float(valid_points))
    summary["scan_age_ms_values"].append(float(lidar_record.get("scan_age_ms", 0.0)))
    summary["speed_values"].append(_safe_float(info.get("speed", 0.0), 0.0))


def _finalize_summary(
    summary: Dict[str, Any],
    output_jsonl: str,
    policy_path: Optional[str],
    policy_format: str,
    env_id: str,
    curriculum_phase: Optional[str],
    sim_path: Optional[str],
    lidar_config: Dict[str, Any],
) -> Dict[str, Any]:
    def _stats(values: list[float]) -> Dict[str, float]:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return {"mean": 0.0, "std": 0.0, "median": 0.0, "p95": 0.0}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
        }

    return {
        "timestamp": datetime.now().isoformat(),
        "output_jsonl": str(output_jsonl),
        "policy_path": str(policy_path) if policy_path else None,
        "policy_format": str(policy_format),
        "env_id": str(env_id),
        "curriculum_phase": None if curriculum_phase is None else str(curriculum_phase),
        "sim_path": None if sim_path in ("", None) else str(sim_path),
        "lidar_config": dict(lidar_config),
        "frames": int(summary["frames"]),
        "episodes": int(summary["episodes"]),
        "new_scan_count": int(summary["new_scan_count"]),
        "valid_ratio": _stats(summary["valid_ratio_values"]),
        "valid_points": _stats(summary["valid_points_values"]),
        "scan_age_ms": _stats(summary["scan_age_ms_values"]),
        "speed": _stats(summary["speed_values"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect one sim raw-LiDAR monitor log")
    parser.add_argument("--env-id", type=str, default="donkey-waveshare-v0")
    parser.add_argument("--policy-path", type=str, default="/home/longzhao/mysim_public/models/v16_pid_full_tuned_20260418/best_model_ws.zip")
    parser.add_argument("--policy-format", type=str, choices=("v16", "v17", "random"), default="v16")
    parser.add_argument("--curriculum-phase", type=str, default="warmup")
    parser.add_argument("--sim", type=str, default="remote")
    parser.add_argument("--sim-start-delay", type=float, default=8.0)
    parser.add_argument("--frames", type=int, default=800)
    parser.add_argument("--max-episode-steps", type=int, default=640)
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--obs-size", type=int, default=128)
    parser.add_argument("--lidar-num-sectors", type=int, default=36)
    parser.add_argument("--lidar-fov-deg", type=float, default=180.0)
    parser.add_argument("--lidar-max-range-m", type=float, default=20.0)
    parser.add_argument("--lidar-near-clip-m", type=float, default=0.18)
    parser.add_argument("--lidar-deg-per-sweep-inc", type=float, default=1.0)
    parser.add_argument("--lidar-deg-ang-down", type=float, default=0.0)
    parser.add_argument("--lidar-deg-ang-delta", type=float, default=-1.0)
    parser.add_argument("--lidar-num-sweeps-levels", type=int, default=1)
    parser.add_argument("--lidar-noise", type=float, default=0.0)
    parser.add_argument("--lidar-offset-x", type=float, default=0.0)
    parser.add_argument("--lidar-offset-y", type=float, default=0.40)
    parser.add_argument("--lidar-offset-z", type=float, default=0.5)
    parser.add_argument("--lidar-rot-x", type=float, default=0.0)
    parser.add_argument("--sim2real-json", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=str, default="/home/longzhao/mysim_public/models/lidar_domain_gap_20260421")
    args = parser.parse_args()

    if args.policy_format != "random" and not args.policy_path:
        raise ValueError("--policy-path is required unless --policy-format random is used")
    if args.policy_format != "random" and RecurrentPPO is None:
        raise RuntimeError("sb3_contrib is required to load the rollout policy")
    wm_export.v17._install_sim_wait_timeout_patch(timeout_s=35.0, resend_scene_names_s=3.0)

    lidar_config = {
        "deg_per_sweep_inc": float(args.lidar_deg_per_sweep_inc),
        "deg_ang_down": float(args.lidar_deg_ang_down),
        "deg_ang_delta": float(args.lidar_deg_ang_delta),
        "num_sweeps_levels": int(args.lidar_num_sweeps_levels),
        "max_range": float(args.lidar_max_range_m),
        "noise": float(args.lidar_noise),
        "offset_x": float(args.lidar_offset_x),
        "offset_y": float(args.lidar_offset_y),
        "offset_z": float(args.lidar_offset_z),
        "rot_x": float(args.lidar_rot_x),
    }

    defaults = wm_export._train_defaults()
    env = wm_export._make_env(
        env_ids=[str(args.env_id)],
        scene_weights=[1.0],
        port=int(args.port),
        seed=int(args.seed),
        defaults=defaults,
        curriculum_phase=args.curriculum_phase,
        obs_size=int(args.obs_size),
        max_range_m=float(args.lidar_max_range_m),
        lidar_num_sectors=int(args.lidar_num_sectors),
        lidar_fov_deg=float(args.lidar_fov_deg),
        sim_path=args.sim,
        sim_start_delay=float(args.sim_start_delay),
        image_channel_indices=[0, 1, 2, 3, 4, 5] if args.policy_format == "v16" else [0, 1, 2, 3, 5],
        lidar_config_override=lidar_config,
        sim2real_json=args.sim2real_json,
    )
    sim2real_wrapper = _find_wrapper_by_class_name(env, "Sim2RealActionWrapper")

    model = None
    if args.policy_format != "random":
        model = RecurrentPPO.load(str(args.policy_path))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir).expanduser() / "monitor_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = out_dir / f"run_{ts}_sim_{args.env_id}_lidar_raw.jsonl"
    output_summary = out_dir / f"run_{ts}_sim_{args.env_id}_summary.json"

    lstm_state = None
    episode_start = np.array([True], dtype=bool)
    episode_id = 0
    frame_idx = 0
    start_time = time.time()
    last_scan_stamp = start_time
    summary = _init_summary()
    speed_vmax = float(defaults.get("speed_vmax", 2.2))
    control_dt = float(defaults.get("control_dt", 0.05))

    try:
        with output_jsonl.open("w", encoding="utf-8") as fh:
            while frame_idx < int(args.frames):
                obs = env.reset()
                episode_id += 1
                summary["episodes"] += 1
                done = False
                step_in_episode = 0

                if model is None:
                    action = env.action_space.sample()
                else:
                    warm_info = getattr(env, "_last_info", {}) if hasattr(env, "_last_info") else {}
                    policy_obs = wm_export._build_behavior_obs(
                        policy_format="v17" if args.policy_format == "v17" else "v16",
                        obs=obs,
                        info=dict(warm_info or {}),
                        env=env,
                        speed_vmax=speed_vmax,
                    )
                    action, lstm_state = model.predict(
                        policy_obs,
                        state=lstm_state,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                obs_cur, _reward, done, info_cur = env.step(action)
                episode_start = np.array([bool(done)], dtype=bool)
                if done:
                    lstm_state = None
                    continue

                prev_obs = obs_cur
                prev_info = dict(info_cur)
                while (not done) and step_in_episode < int(args.max_episode_steps) and frame_idx < int(args.frames):
                    if model is None:
                        action = env.action_space.sample()
                    else:
                        policy_obs = wm_export._build_behavior_obs(
                            policy_format="v17" if args.policy_format == "v17" else "v16",
                            obs=prev_obs,
                            info=prev_info,
                            env=env,
                            speed_vmax=speed_vmax,
                        )
                        action, lstm_state = model.predict(
                            policy_obs,
                            state=lstm_state,
                            episode_start=episode_start,
                            deterministic=True,
                        )

                    next_obs, reward, done, next_info = env.step(action)
                    now = time.time()
                    elapsed = now - start_time
                    raw_lidar = np.asarray(next_info.get("lidar", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
                    raw_lidar_packet = _sanitize_lidar_packet(next_info.get("lidar_raw_packet"))
                    if raw_lidar.size == 0 and not raw_lidar_packet:
                        prev_obs = next_obs
                        prev_info = dict(next_info)
                        episode_start = np.array([bool(done)], dtype=bool)
                        step_in_episode += 1
                        if done:
                            lstm_state = None
                        continue

                    is_new_scan = _safe_float(next_info.get("lidar_is_new_scan", 0.0), 0.0) >= 0.5
                    if is_new_scan:
                        last_scan_stamp = now
                    lidar_record = _build_lidar_record(
                        raw_lidar=raw_lidar,
                        lidar_packet=raw_lidar_packet,
                        info=next_info,
                        stamp=last_scan_stamp,
                        control_dt=control_dt,
                        near_clip_m=float(args.lidar_near_clip_m),
                        max_range_m=float(args.lidar_max_range_m),
                        frame_idx=frame_idx,
                    )
                    final_angle, final_throttle, pre_sim2real_angle, pre_sim2real_throttle, sim2real_applied = _resolve_logged_final_action(
                        info=next_info,
                        action=np.asarray(action, dtype=np.float32).reshape(-1),
                        sim2real_wrapper=sim2real_wrapper,
                    )
                    record = {
                        "timestamp": float(now),
                        "elapsed_sec": float(elapsed),
                        "frame": int(frame_idx),
                        "sample_id": int(frame_idx),
                        "episode_id": int(episode_id),
                        "step_in_episode": int(step_in_episode),
                        "scene_key": str(next_info.get("scene_key", "")),
                        "logging_key": str(next_info.get("logging_key", "")),
                        "domain": str(next_info.get("domain", "")),
                        "mode": "pilot",
                        "recording": False,
                        "run_pilot": True,
                        "tub_record_index_est": -1,
                        "tub_records": 0,
                        "user_angle": 0.0,
                        "user_throttle": 0.0,
                        "pilot_angle": _safe_float(action[0] if np.asarray(action).size > 0 else 0.0, 0.0),
                        "pilot_throttle": _safe_float(action[1] if np.asarray(action).size > 1 else 0.0, 0.0),
                        "final_angle": float(final_angle),
                        "final_throttle": float(final_throttle),
                        "pre_sim2real_final_angle": pre_sim2real_angle,
                        "pre_sim2real_final_throttle": pre_sim2real_throttle,
                        "sim2real_applied": bool(sim2real_applied),
                        "reward": _safe_float(reward, 0.0),
                        "done": bool(done),
                        "termination_reason": str(next_info.get("termination_reason", "")),
                        "speed": _safe_float(next_info.get("speed", 0.0), 0.0),
                        "cte": _safe_float(next_info.get("cte", 0.0), 0.0),
                        "obstacle_present": _safe_float(next_info.get("obstacle_present", 0.0), 0.0),
                        "obstacle_longitudinal": _safe_float(next_info.get("obstacle_longitudinal", 0.0), 0.0),
                        "obstacle_lateral": _safe_float(next_info.get("obstacle_lateral", 0.0), 0.0),
                        "obstacle_dist": _safe_float(next_info.get("obstacle_dist", 0.0), 0.0),
                        "obstacle_risk": _safe_float(next_info.get("obstacle_risk", 0.0), 0.0),
                        "lidar_is_new_scan": float(lidar_record["is_new_scan"]),
                        "lidar_steps_since_new_scan": int(lidar_record["steps_since_new_scan"]),
                        "lidar_steps_since_new_scan_norm": _safe_float(next_info.get("lidar_steps_since_new_scan_norm", 0.0), 0.0),
                        "lidar_repeat_count": int(lidar_record["repeat_count"]),
                        "lidar_repeat_count_norm": _safe_float(next_info.get("lidar_repeat_count_norm", 0.0), 0.0),
                        "lidar_scan_age_norm": _safe_float(next_info.get("lidar_scan_age_norm", 0.0), 0.0),
                        "lidar_source": str(lidar_record.get("source", "")),
                        "lidar_packet_size": int(lidar_record.get("packet_size", 0)),
                        "lidar_packet": list(raw_lidar_packet),
                        "canonical_lidar_range": np.asarray(next_info.get("canonical_lidar_range", np.zeros((int(args.lidar_num_sectors),), dtype=np.float32)), dtype=np.float32).astype(float).tolist(),
                        "canonical_lidar_valid": np.asarray(next_info.get("canonical_lidar_valid", np.zeros((int(args.lidar_num_sectors),), dtype=np.float32)), dtype=np.float32).astype(float).tolist(),
                        "target_exist": _safe_float(next_info.get("target_exist", 0.0), 0.0),
                        "target_rel_long": _safe_float(next_info.get("target_rel_long", 0.0), 0.0),
                        "target_rel_lat": _safe_float(next_info.get("target_rel_lat", 0.0), 0.0),
                        "target_rel_v_long": _safe_float(next_info.get("target_rel_v_long", 0.0), 0.0),
                        "target_rel_v_lat": _safe_float(next_info.get("target_rel_v_lat", 0.0), 0.0),
                        "target_ttc": _safe_float(next_info.get("target_ttc", 0.0), 0.0),
                        "target_confidence": _safe_float(next_info.get("target_confidence", 0.0), 0.0),
                        "target_width_proxy": _safe_float(next_info.get("target_width_proxy", 0.0), 0.0),
                        "target_front_min_range": _safe_float(next_info.get("target_front_min_range", 0.0), 0.0),
                        "target_left_gap": _safe_float(next_info.get("target_left_gap", 0.0), 0.0),
                        "target_right_gap": _safe_float(next_info.get("target_right_gap", 0.0), 0.0),
                        "lidar": lidar_record,
                    }
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    _update_summary(summary, lidar_record, next_info)
                    frame_idx += 1

                    prev_obs = next_obs
                    prev_info = dict(next_info)
                    episode_start = np.array([bool(done)], dtype=bool)
                    step_in_episode += 1
                    if done:
                        lstm_state = None
    finally:
        env.close()

    summary_payload = _finalize_summary(
        summary=summary,
        output_jsonl=str(output_jsonl),
        policy_path=(None if args.policy_format == "random" else args.policy_path),
        policy_format=str(args.policy_format),
        env_id=str(args.env_id),
        curriculum_phase=(None if args.curriculum_phase in ("", "none", None) else args.curriculum_phase),
        sim_path=args.sim,
        lidar_config=lidar_config,
    )
    summary_payload["sim2real_json"] = None if args.sim2real_json in ("", None) else str(args.sim2real_json)
    summary_payload["sim2real_wrapper_active"] = bool(sim2real_wrapper is not None)
    with output_summary.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
