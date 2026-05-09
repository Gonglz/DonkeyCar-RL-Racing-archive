#!/usr/bin/env python3
"""Evaluate V17 RecurrentPPO checkpoints on the 1x V17 environment."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from sb3_contrib import RecurrentPPO
except Exception as exc:  # pragma: no cover
    raise ImportError("sb3_contrib is required for V17 evaluation") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
while repo_root_str in sys.path:
    sys.path.remove(repo_root_str)
sys.path.insert(0, repo_root_str)

import src.ppo_multitrack_v17 as p  # noqa: E402


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _mean(items: List[float]) -> Optional[float]:
    return None if not items else float(np.mean(np.asarray(items, dtype=np.float64)))


def _rate(episodes: List[Dict[str, Any]], token: str) -> float:
    if not episodes:
        return float("nan")
    hits = 0
    for ep in episodes:
        reason = str(ep.get("termination_reason", "") or "")
        if token in set(reason.split("+")):
            hits += 1
    return float(hits / max(1, len(episodes)))


def build_env(args: argparse.Namespace) -> DummyVecEnv:
    env_ids = list(args.env_ids or p.DEFAULT_ENV_IDS)
    scene_weights = [1.0, 0.0] if len(env_ids) == 2 else [1.0 / len(env_ids)] * len(env_ids)
    track_dir = p._resolve_track_dir(track_dir=args.track_dir, env_ids=env_ids)
    track_geometry = p.TrackGeometryManager(track_dir=track_dir, env_ids=env_ids, scene_specs=p.SCENE_SPECS)

    if args.sim_loaded_timeout_s > 0:
        p._install_sim_wait_timeout_patch(
            timeout_s=float(args.sim_loaded_timeout_s),
            resend_scene_names_s=float(args.sim_wait_resend_scene_names_s),
        )

    ok, err = p._probe_sim_tcp("127.0.0.1", int(args.port), timeout_s=1.0)
    if not ok:
        raise RuntimeError(f"sim tcp not reachable: 127.0.0.1:{args.port} ({err})")

    cfg = p.load_config(myconfig=p.DEFAULT_MYCONFIG)
    conf = cfg.GYM_CONF.copy() if cfg is not None and hasattr(cfg, "GYM_CONF") else {}
    conf.update(
        {
            "host": "127.0.0.1",
            "port": int(args.port),
            "car_name": "waveshare_v17_eval",
            "racer_name": "V17-LiDAR-Eval",
            "country": "CN",
            "bio": "V17 LiDAR 1x eval",
            "guid": "waveshare-v17-lidar-eval",
            "max_cte": 8.0,
            "lidar_config": {
                "deg_per_sweep_inc": max(1.0, float(args.lidar_fov_deg) / max(1, int(args.lidar_num_sectors) * 5)),
                "deg_ang_down": 0.0,
                "deg_ang_delta": -1.0,
                "num_sweeps_levels": 1,
                "max_range": float(args.lidar_max_range_m),
                "noise": 0.5,
                "offset_x": 0.0,
                "offset_y": 0.40,
                "offset_z": 0.5,
                "rot_x": 0.0,
            },
        }
    )

    reward_overrides = dict(
        p.CURRICULUM_PHASES.get("ws_bootstrap", {}).get("reward_overrides_by_logging_key", {})
    )
    offtrack_leniency_ratio = 0.0 if args.strict_offtrack else 0.25

    def make_env():
        return p.MultiSceneEnvV17(
            env_ids=env_ids,
            conf=conf,
            scene_weights=scene_weights,
            scene_specs=p.SCENE_SPECS,
            track_geometry=track_geometry,
            track_dir=track_dir,
            obs_size=int(args.obs_size),
            augment=False,
            yellow_dropout_prob=0.20,
            dropout_start_step=0,
            dropout_ramp_steps=200000,
            adapter_k_delta=0.08,
            adapter_lambda_bias=0.20,
            adapter_k_bias=0.08,
            adapter_steer_core_decay=0.0,
            adapter_v_nominal=0.58,
            adapter_k_turn=0.35,
            adapter_k_bias_speed=0.0,
            adapter_alpha_speed=0.38,
            adapter_v_min=0.14,
            adapter_v_max=1.0,
            speed_vmax=2.2,
            speed_kp=0.35,
            speed_ki=0.08,
            speed_kff=0.10,
            allow_reverse=False,
            max_throttle=0.30,
            control_dt=0.05,
            total_timesteps=int(args.total_timesteps),
            delta_max=0.20,
            enable_lpf=True,
            beta=0.50,
            steer_delta_delta_max=0.04,
            steer_servo_deadband=0.006,
            w_d=0.04,
            w_dd=0.01,
            w_m=0.03,
            w_sat=0.06,
            w_time=0.01,
            w_center=0.03,
            w_heading=0.015,
            w_speed_ref=0.0,
            speed_ref_vmin=0.35,
            speed_ref_vmax=2.2,
            speed_ref_kappa_ref=0.15,
            lap_reward_scale=1.0,
            progress_reward_scale=48.0,
            survival_reward_scale=0.30,
            collision_penalty_base=8.0,
            offtrack_penalty_base=8.0,
            offtrack_leniency_ratio=offtrack_leniency_ratio,
            offtrack_leniency_mult=2.5,
            adaptive_delta_max=True,
            curve_delta_boost=1.0,
            curve_kappa_ref=0.15,
            steer_intent_boost=0.30,
            hairpin_curve_ratio=0.85,
            hairpin_min_delta_max=0.45,
            hairpin_max_delta_max=0.85,
            w_near_offtrack=0.55,
            near_offtrack_start_ratio=0.45,
            w_near_collision=0.08,
            near_collision_start_ratio=0.82,
            overtake_success_bonus=3.0,
            reward_safe_follow_bonus=0.02,
            reward_prepare_pass_bonus=0.04,
            reward_commit_pass_bonus=0.04,
            reward_post_pass_bonus=0.5,
            reward_post_pass_steps=10,
            curriculum_phase="ws_bootstrap",
            reward_overrides_by_logging_key=reward_overrides,
            terminal_offtrack_progress_scale=0.0,
            bad_episode_guard_min_steps=80,
            bad_episode_guard_reward_floor=-90.0,
            bad_episode_guard_cte_over_in_rate=0.18,
            bad_episode_guard_min_forward_progress=0.18,
            bad_episode_guard_penalty=4.0,
            snapshot_dir="",
            snapshot_max_steps=0,
            min_episodes_per_scene=5,
            max_steps_per_scene=640,
            enable_dynamic_scene_weights=False,
            enable_step_balance_sampling=False,
            obstacle_enabled=False,
            obstacle_count=0,
            obstacle_free_prob=1.0,
            obstacle_modes=["static"],
            ws_obstacle_free_prob=1.0,
            obstacle_spawn_ahead_min_m=3.5,
            obstacle_spawn_ahead_max_m=14.0,
            obstacle_min_agent_planar_dist_m=1.5,
            obstacle_min_agent_arc_dist_m=3.5,
            obstacle_min_separation_world=3.0,
            obstacle_lane_pid_speed_gt=0.85,
            obstacle_lane_pid_speed_ws=0.70,
            obstacle_lane_pid_lookahead_m=0.9,
            obstacle_jitter_amplitude_m=0.10,
            obstacle_jitter_period_s=1.5,
            obstacle_jitter_update_hz=8.0,
            obstacle_nudge_amplitude_m=0.14,
            obstacle_nudge_period_s=1.5,
            obstacle_nudge_update_hz=8.0,
            sim2real_json=str(args.sim2real_json),
            sim2real_throttle_gain_floor=0.25,
            sim2real_throttle_gain_override=None,
            sim2real_steer_gain_floor=None,
            sim2real_steer_gain_override=None,
            sim2real_filter_dt_s=0.05,
            image_channel_indices=[0, 1, 2, 3, 5],
            lidar_num_sectors=int(args.lidar_num_sectors),
            lidar_fov_deg=float(args.lidar_fov_deg),
            lidar_max_range_m=float(args.lidar_max_range_m),
            lidar_near_clip_m=0.18,
            lidar_repeat_min_steps=2,
            lidar_repeat_max_steps=4,
            lidar_obs_mode="full",
            predictive_safety_filter_path=None,
            predictive_safety_filter_mode="log",
            predictive_safety_filter_log_path=None,
        )

    env = DummyVecEnv([make_env])
    p._safe_seed_env(env, args.seed, label="v17_eval_env")
    return env


def evaluate_model(
    model_path: Path,
    vec_env: DummyVecEnv,
    episodes: int,
    deterministic: bool,
    max_steps_per_episode: int,
) -> Dict[str, Any]:
    model = RecurrentPPO.load(str(model_path), env=vec_env)
    obs = vec_env.reset()
    n_envs = int(getattr(vec_env, "num_envs", 1))
    lstm_state = None
    episode_start = np.ones((n_envs,), dtype=bool)
    running_rewards = np.zeros((n_envs,), dtype=np.float32)
    running_lengths = np.zeros((n_envs,), dtype=np.int32)
    episode_rows: List[Dict[str, Any]] = []
    start_time = time.time()

    while len(episode_rows) < int(episodes):
        action, lstm_state = model.predict(
            obs,
            state=lstm_state,
            episode_start=episode_start,
            deterministic=bool(deterministic),
        )
        obs, rewards, dones, infos = vec_env.step(action)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        dones = np.asarray(dones, dtype=bool).reshape(-1)
        running_rewards += rewards
        running_lengths += 1

        for idx in range(n_envs):
            forced_done = False
            if max_steps_per_episode > 0 and running_lengths[idx] >= int(max_steps_per_episode):
                forced_done = True
                dones[idx] = True
            if not dones[idx]:
                continue
            info = dict(infos[idx] or {})
            ep_info = dict(info.get("episode", {}) or {})
            row = {
                "episode": len(episode_rows) + 1,
                "reward": float(ep_info.get("r", running_rewards[idx])),
                "length": int(ep_info.get("l", running_lengths[idx])),
                "forced_done": bool(forced_done),
                "termination_reason": str(info.get("termination_reason", ep_info.get("termination_reason", "")) or ""),
            }
            for key in (
                "ep_speed_mean",
                "ep_speed_max",
                "ep_sim2real_throttle_mean",
                "ep_sim2real_steer_abs_mean",
                "ep_safety_delta_steer_abs_mean",
                "ep_safety_delta_delta_limit_hit_rate",
                "ep_safety_servo_deadband_hold_rate",
                "ep_native_env_done_cte_abs",
                "ep_cte_abs_p90",
                "ep_cte_abs_p99",
            ):
                if key in ep_info:
                    row[key] = _jsonable(ep_info[key])
            episode_rows.append(row)
            running_rewards[idx] = 0.0
            running_lengths[idx] = 0
            if len(episode_rows) >= int(episodes):
                break
        episode_start = dones.astype(bool)

    elapsed = max(1e-6, time.time() - start_time)
    rewards = [float(x["reward"]) for x in episode_rows]
    lengths = [float(x["length"]) for x in episode_rows]
    summary = {
        "model_path": str(model_path),
        "episodes": int(len(episode_rows)),
        "deterministic": bool(deterministic),
        "elapsed_s": float(elapsed),
        "steps_per_s": float(sum(lengths) / elapsed),
        "reward_mean": _mean(rewards),
        "reward_std": float(np.std(np.asarray(rewards, dtype=np.float64))) if rewards else None,
        "length_mean": _mean(lengths),
        "length_min": int(min(lengths)) if lengths else None,
        "length_max": int(max(lengths)) if lengths else None,
        "short_ep_rate": float(sum(1 for x in lengths if x < 15) / max(1, len(lengths))),
        "term_collision_rate": _rate(episode_rows, "collision"),
        "term_offtrack_rate": _rate(episode_rows, "offtrack"),
        "term_stuck_rate": _rate(episode_rows, "stuck"),
        "speed_mean": _mean([float(x["ep_speed_mean"]) for x in episode_rows if "ep_speed_mean" in x]),
        "speed_max_mean": _mean([float(x["ep_speed_max"]) for x in episode_rows if "ep_speed_max" in x]),
        "sim2real_throttle_mean": _mean(
            [float(x["ep_sim2real_throttle_mean"]) for x in episode_rows if "ep_sim2real_throttle_mean" in x]
        ),
        "sim2real_steer_abs_mean": _mean(
            [float(x["ep_sim2real_steer_abs_mean"]) for x in episode_rows if "ep_sim2real_steer_abs_mean" in x]
        ),
        "safety_delta_steer_abs_mean": _mean(
            [float(x["ep_safety_delta_steer_abs_mean"]) for x in episode_rows if "ep_safety_delta_steer_abs_mean" in x]
        ),
        "safety_delta_delta_limit_hit_rate": _mean(
            [float(x["ep_safety_delta_delta_limit_hit_rate"]) for x in episode_rows if "ep_safety_delta_delta_limit_hit_rate" in x]
        ),
        "servo_deadband_hold_rate": _mean(
            [float(x["ep_safety_servo_deadband_hold_rate"]) for x in episode_rows if "ep_safety_servo_deadband_hold_rate" in x]
        ),
    }
    return {"summary": summary, "episodes_detail": episode_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V17 checkpoints on 1x sim.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--sim2real-json", required=True)
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument("--strict-offtrack", action="store_true", default=True)
    parser.add_argument("--lenient-offtrack", dest="strict_offtrack", action="store_false")
    parser.add_argument("--track-dir", default=p.DEFAULT_TRACK_DIR)
    parser.add_argument("--env-ids", nargs="+", default=None)
    parser.add_argument("--obs-size", type=int, default=128)
    parser.add_argument("--lidar-num-sectors", type=int, default=36)
    parser.add_argument("--lidar-fov-deg", type=float, default=180.0)
    parser.add_argument("--lidar-max-range-m", type=float, default=20.0)
    parser.add_argument("--total-timesteps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sim-loaded-timeout-s", type=float, default=35.0)
    parser.add_argument("--sim-wait-resend-scene-names-s", type=float, default=3.0)
    args = parser.parse_args()

    model_paths = [Path(x).expanduser().resolve() for x in args.models]
    for model_path in model_paths:
        if not model_path.exists():
            raise FileNotFoundError(str(model_path))
    labels = list(args.labels or [path.stem for path in model_paths])
    if len(labels) != len(model_paths):
        raise ValueError("--labels length must match --models length")

    args.sim2real_json = Path(args.sim2real_json).expanduser().resolve()
    if not args.sim2real_json.exists():
        raise FileNotFoundError(str(args.sim2real_json))

    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    env = build_env(args)
    results: Dict[str, Any] = {
        "version": "V17_1X_EVAL",
        "timestamp": datetime.now().isoformat(),
        "sim2real_json": str(args.sim2real_json),
        "strict_offtrack": bool(args.strict_offtrack),
        "episodes_requested": int(args.episodes),
        "models": {},
    }
    try:
        for label, model_path in zip(labels, model_paths):
            print(f"[eval] {label}: {model_path}")
            results["models"][label] = evaluate_model(
                model_path=model_path,
                vec_env=env,
                episodes=int(args.episodes),
                deterministic=bool(args.deterministic),
                max_steps_per_episode=int(args.max_steps_per_episode),
            )
            print(json.dumps(results["models"][label]["summary"], ensure_ascii=False, indent=2))
    finally:
        env.close()

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(results), f, ensure_ascii=False, indent=2)
    print(f"[eval] wrote {output_json}")


if __name__ == "__main__":
    main()
