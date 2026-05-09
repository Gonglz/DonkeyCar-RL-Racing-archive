#!/usr/bin/env python3
"""
Export step-level V17 local world-model supervision from sim rollouts.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sb3_contrib import RecurrentPPO
except Exception:
    RecurrentPPO = None

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from module.predictive_safety_filter import PhysState
from module.track import TrackGeometryManager
from module.v17_env import MultiSceneEnvV17
from module.obv import _build_state_v16
from src import ppo_multitrack_v17 as v17


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "none", "null", "nan", "false", "no"):
            return float(default)
        if text in ("true", "yes"):
            return 1.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _train_defaults() -> Dict[str, Any]:
    sig = inspect.signature(v17.train_v17)
    return {
        name: param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect._empty
    }


def _build_conf(
    port: int,
    lidar_num_sectors: int,
    lidar_fov_deg: float,
    lidar_max_range_m: float,
    sim_path: Optional[str] = None,
    sim_start_delay: Optional[float] = None,
    lidar_config_override: Optional[Dict[str, Any]] = None,
    conf_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = v17.load_config(myconfig=v17.DEFAULT_MYCONFIG)
    conf = cfg.GYM_CONF.copy() if cfg is not None and hasattr(cfg, "GYM_CONF") else {}
    lidar_cfg = {
        "deg_per_sweep_inc": max(1.0, float(lidar_fov_deg) / max(1, int(lidar_num_sectors) * 5)),
        "deg_ang_down": 0.0,
        "deg_ang_delta": -1.0,
        "num_sweeps_levels": 1,
        "max_range": float(lidar_max_range_m),
        "noise": 0.5,
        "offset_x": 0.0,
        "offset_y": 0.40,
        "offset_z": 0.5,
        "rot_x": 0.0,
    }
    if lidar_config_override:
        for key, value in dict(lidar_config_override).items():
            if value is None:
                continue
            lidar_cfg[str(key)] = value
    conf.update(
        {
            "host": "127.0.0.1",
            "port": int(port),
            "car_name": "waveshare_v17_wm_export",
            "racer_name": "V17-WM-Export",
            "country": "CN",
            "bio": "V17 world model export",
            "guid": "waveshare-v17-world-model-export",
            "max_cte": 8.0,
            "lidar_config": lidar_cfg,
        }
    )
    if conf_override:
        for key, value in dict(conf_override).items():
            if value is None:
                continue
            conf[str(key)] = value
    sim_path_text = str(sim_path or "").strip()
    if sim_path_text and sim_path_text not in ("remote", "none"):
        conf["exe_path"] = sim_path_text
        conf["start_delay"] = float(max(float(sim_start_delay or 8.0), float(conf.get("start_delay", 0.0) or 0.0)))
    return conf


def _estimate_side_gaps(lidar_flat: np.ndarray, max_range_m: float) -> Tuple[float, float]:
    lidar_flat = np.asarray(lidar_flat, dtype=np.float32).reshape(-1)
    ranges, valid = np.split(lidar_flat, 2)
    left = ranges[:18][valid[:18] > 0.5]
    right = ranges[18:][valid[18:] > 0.5]
    left_gap = float(np.quantile(left, 0.20)) if left.size else float(max_range_m)
    right_gap = float(np.quantile(right, 0.20)) if right.size else float(max_range_m)
    return left_gap, right_gap


def _scene_obstacle_geometry(scene_key: str) -> Tuple[float, float]:
    scene = str(scene_key or "").strip().lower()
    if scene == "waveshare":
        return 0.18, 0.05
    return 0.20, 0.08


def _estimate_track_side_gaps(
    info: Dict[str, Any],
    track_geometry: Optional[TrackGeometryManager],
    max_range_m: float,
) -> Optional[Tuple[float, float]]:
    if track_geometry is None:
        return None
    scene_key = str(info.get("scene_key", "") or "").strip()
    if not scene_key or scene_key not in track_geometry.scenes:
        return None
    if _safe_float(info.get("obstacle_present", 0.0), 0.0) <= 0.5:
        return None

    pos = info.get("pos", (0.0, 0.0, 0.0))
    car = info.get("car", (0.0, 0.0, 0.0))
    try:
        x = float(pos[0])
        z = float(pos[2])
        yaw_deg = float(car[2])
        obstacle_lateral = float(info.get("obstacle_lateral", 0.0) or 0.0)
    except Exception:
        return None

    geo = track_geometry.query(
        scene_key=scene_key,
        x=x,
        z=z,
        yaw_rad=np.deg2rad(yaw_deg),
    )
    idx = int(round(float(geo.get("idx", 0.0)))) % len(track_geometry.scenes[scene_key].width)
    local_width = float(track_geometry.scenes[scene_key].width[idx])
    half_width = max(0.5 * local_width, 1e-3)
    ego_lat = float(geo.get("lat_err", 0.0))
    obstacle_track_lat = ego_lat + obstacle_lateral
    obstacle_radius, safety_margin = _scene_obstacle_geometry(scene_key)

    left_gap = half_width - (obstacle_track_lat + obstacle_radius) - safety_margin
    right_gap = (obstacle_track_lat - obstacle_radius) + half_width - safety_margin
    left_gap = float(np.clip(left_gap, 0.0, max_range_m))
    right_gap = float(np.clip(right_gap, 0.0, max_range_m))
    return left_gap, right_gap


def _collision_flag(info: Dict[str, Any]) -> float:
    candidates = [
        info.get("hit", 0.0),
        info.get("collision", 0.0),
        info.get("collided", 0.0),
        info.get("reward_collision", 0.0),
    ]
    return float(any(_safe_float(x, 0.0) > 0.5 for x in candidates))


def _build_ego8(obs: Dict[str, np.ndarray], info: Dict[str, Any], control_dt: float) -> Tuple[np.ndarray, np.ndarray]:
    gyro = info.get("gyro", (0.0, 0.0, 0.0))
    accel = info.get("accel", (0.0, 0.0, 0.0))
    phys = PhysState.from_raw(
        speed_mps=float(info.get("speed", 0.0) or 0.0),
        gyro_z=float(gyro[1]) if len(gyro) > 1 else 0.0,
        accel_x=float(accel[0]) if len(accel) > 0 else 0.0,
    )
    state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
    prev_steer = float(state[3]) if state.size >= 5 else 0.0
    prev_throttle = float(state[4]) if state.size >= 5 else 0.0
    ego8 = np.array(
        [
            phys.v_long,
            phys.yaw_rate,
            phys.accel_x,
            float(info.get("safety/steer_exec", 0.0) or 0.0),
            float(info.get("ctrl/throttle_pi", 0.0) or 0.0),
            prev_steer,
            prev_throttle,
            float(control_dt / 0.05),
        ],
        dtype=np.float32,
    )
    phys3 = np.array([phys.v_long, phys.yaw_rate, phys.accel_x], dtype=np.float32)
    return ego8, phys3


def _build_async_meta(info: Dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            float(info.get("lidar_scan_age_norm", 0.0) or 0.0),
            float(info.get("lidar_steps_since_new_scan_norm", 0.0) or 0.0),
            float(info.get("lidar_repeat_count_norm", 0.0) or 0.0),
            float(info.get("lidar_is_new_scan", 0.0) or 0.0),
        ],
        dtype=np.float32,
    )


def _resize_semantic_image_nearest(image: np.ndarray, target_size: int) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"expected CHW image, got shape={image.shape}")
    channels, height, width = image.shape
    if height == target_size and width == target_size:
        return image
    ys = np.linspace(0, max(0, height - 1), int(target_size)).round().astype(np.int64)
    xs = np.linspace(0, max(0, width - 1), int(target_size)).round().astype(np.int64)
    return image[:, ys][:, :, xs].reshape(channels, int(target_size), int(target_size))


def _build_camera_tensor(obs: Dict[str, np.ndarray], camera_obs_size: int) -> np.ndarray:
    image = np.asarray(obs["image"], dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"expected CHW image, got shape={image.shape}")
    if image.shape[0] == 6:
        image = image[[0, 1, 2, 3, 5], :, :]
    elif image.shape[0] != 5:
        raise ValueError(f"expected 5ch or 6ch image, got shape={image.shape}")
    image = _resize_semantic_image_nearest(image, target_size=int(camera_obs_size))
    return image.astype(np.float16)


def _build_targets(
    cur_obs: Dict[str, np.ndarray],
    cur_info: Dict[str, Any],
    next_obs: Dict[str, np.ndarray],
    next_info: Dict[str, Any],
    control_dt: float,
    max_range_m: float,
    passable_gap_threshold_m: float,
    gap_label_source: str = "sensor",
    track_geometry: Optional[TrackGeometryManager] = None,
) -> Dict[str, np.ndarray]:
    cur_present = float(cur_info.get("obstacle_present", 0.0) or 0.0) > 0.5
    next_present = float(next_info.get("obstacle_present", 0.0) or 0.0) > 0.5
    cur_long = float(cur_info.get("obstacle_longitudinal", 0.0) or 0.0)
    cur_lat = float(cur_info.get("obstacle_lateral", 0.0) or 0.0)
    next_long = float(next_info.get("obstacle_longitudinal", 0.0) or 0.0)
    next_lat = float(next_info.get("obstacle_lateral", 0.0) or 0.0)

    rel_v_long = 0.0
    rel_v_lat = 0.0
    rel_mask = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if next_present:
        rel_mask[:2] = 1.0
    if cur_present and next_present:
        rel_v_long = float((next_long - cur_long) / max(control_dt, 1e-3))
        rel_v_lat = float((next_lat - cur_lat) / max(control_dt, 1e-3))
        rel_mask[2:] = 1.0

    gap_source = str(gap_label_source or "sensor").strip().lower()
    track_gaps = None
    if gap_source == "track":
        track_gaps = _estimate_track_side_gaps(next_info, track_geometry=track_geometry, max_range_m=max_range_m)
    if track_gaps is None:
        left_gap, right_gap = _estimate_side_gaps(next_obs["lidar"], max_range_m=max_range_m)
    else:
        left_gap, right_gap = track_gaps
    ttc_s = float(next_info.get("reward_debug/obstacle_ttc_s", 999.0) or 999.0)
    if not np.isfinite(ttc_s):
        ttc_s = 5.0
    ttc_s = float(np.clip(ttc_s, 0.0, 5.0))
    lateral_overlap = _safe_float(next_info.get("reward_debug/obstacle_lateral_overlap", 0.0), 0.0)
    near_collision_risk = _safe_float(next_info.get("reward_debug/near_collision_ttc_risk", 0.0), 0.0)
    actual_collision = float(_collision_flag(next_info))
    danger_collision = float(
        (ttc_s < 0.60 and lateral_overlap > 0.20)
        or near_collision_risk >= 0.85
    )
    collision_target = float(max(actual_collision, danger_collision))

    closing_rate = 0.0
    overtake_progress = 0.0
    opp_valid = 0.0
    if cur_present and next_present:
        closing_rate = float((cur_long - next_long) / max(control_dt, 1e-3))
        overtake_progress = float(max(cur_long - next_long, 0.0))
        opp_valid = 1.0

    return {
        "target_rel": np.array([next_long, next_lat, rel_v_long, rel_v_lat], dtype=np.float32),
        "target_rel_mask": rel_mask.astype(np.float32),
        "target_gap": np.array([left_gap, right_gap], dtype=np.float32),
        "target_collision": np.array(collision_target, dtype=np.float32),
        "target_ttc": np.array(ttc_s, dtype=np.float32),
        "target_safety_valid": np.array(1.0, dtype=np.float32),
        "target_passable": np.array(
            [float(left_gap >= passable_gap_threshold_m), float(right_gap >= passable_gap_threshold_m)],
            dtype=np.float32,
        ),
        "target_closing_rate": np.array(closing_rate, dtype=np.float32),
        "target_overtake_progress": np.array(overtake_progress, dtype=np.float32),
        "target_opportunity_valid": np.array(opp_valid, dtype=np.float32),
    }


def _make_env(
    env_ids: List[str],
    scene_weights: List[float],
    port: int,
    seed: int,
    defaults: Dict[str, Any],
    curriculum_phase: Optional[str],
    obs_size: int,
    max_range_m: float,
    lidar_num_sectors: int,
    lidar_fov_deg: float = 180.0,
    sim_path: Optional[str] = None,
    sim_start_delay: Optional[float] = None,
    image_channel_indices: Optional[Sequence[int]] = None,
    lidar_config_override: Optional[Dict[str, Any]] = None,
    conf_override: Optional[Dict[str, Any]] = None,
    sim2real_json: Optional[str] = None,
) -> MultiSceneEnvV17:
    curriculum_values = {
        k: defaults.get(k)
        for k in (
            "scene_weights",
            "obstacle_enabled",
            "obstacle_count",
            "obstacle_free_prob",
            "obstacle_modes",
            "ws_obstacle_free_prob",
            "obstacle_spawn_ahead_min_m",
            "obstacle_spawn_ahead_max_m",
            "obstacle_min_agent_planar_dist_m",
            "obstacle_min_agent_arc_dist_m",
            "obstacle_min_separation_world",
            "obstacle_lateral_choices",
            "obstacle_fixed_progress_ratio",
            "obstacle_fixed_progress_gap",
            "obstacle_fixed_progress_gap_min",
            "obstacle_fixed_progress_gap_max",
            "obstacle_progress_min",
            "obstacle_progress_max",
            "obstacle_fixed_lateral_ratio",
            "gt_obstacle_start_exclusion_half_width_m",
            "ws_obstacle_modes",
            "ws_obstacle_fixed_progress_ratio",
            "ws_obstacle_progress_min",
            "ws_obstacle_progress_max",
            "ws_obstacle_fixed_lateral_ratio",
            "obstacle_randomize_non_lane_pid_yaw",
            "obstacle_lane_pid_speed_gt",
            "obstacle_lane_pid_speed_ws",
            "collision_penalty_base",
            "offtrack_penalty_base",
            "w_near_collision",
            "near_collision_start_ratio",
            "overtake_success_bonus",
        )
    }
    curriculum_phase, _ = v17._apply_curriculum_phase(curriculum_phase, curriculum_values)
    track_dir = v17._resolve_track_dir(track_dir=str(defaults.get("track_dir", v17.DEFAULT_TRACK_DIR)), env_ids=env_ids)
    conf = _build_conf(
        port=port,
        lidar_num_sectors=lidar_num_sectors,
        lidar_fov_deg=lidar_fov_deg,
        lidar_max_range_m=max_range_m,
        sim_path=sim_path,
        sim_start_delay=sim_start_delay,
        lidar_config_override=lidar_config_override,
        conf_override=conf_override,
    )
    track_geometry = TrackGeometryManager(track_dir=track_dir, env_ids=env_ids, scene_specs=v17.SCENE_SPECS)
    random.seed(seed)
    np.random.seed(seed)

    return MultiSceneEnvV17(
        env_ids=env_ids,
        conf=conf,
        scene_weights=scene_weights,
        scene_specs=v17.SCENE_SPECS,
        track_geometry=track_geometry,
        track_dir=track_dir,
        obs_size=int(obs_size),
        augment=False,
        yellow_dropout_prob=float(defaults.get("yellow_dropout_prob", 0.0)),
        dropout_start_step=int(defaults.get("dropout_start_step", 0)),
        dropout_ramp_steps=int(defaults.get("dropout_ramp_steps", 1)),
        adapter_k_delta=float(defaults.get("adapter_k_delta", 0.15)),
        adapter_lambda_bias=float(defaults.get("adapter_lambda_bias", 0.20)),
        adapter_k_bias=float(defaults.get("adapter_k_bias", 0.15)),
        adapter_steer_core_decay=float(defaults.get("adapter_steer_core_decay", 0.0)),
        adapter_v_nominal=float(defaults.get("adapter_v_nominal", 1.4)),
        adapter_k_turn=float(defaults.get("adapter_k_turn", 0.5)),
        adapter_k_bias_speed=float(defaults.get("adapter_k_bias_speed", 0.0)),
        adapter_alpha_speed=float(defaults.get("adapter_alpha_speed", 0.25)),
        adapter_v_min=float(defaults.get("adapter_v_min", 0.6)),
        adapter_v_max=float(defaults.get("adapter_v_max", 1.8)),
        speed_vmax=float(defaults.get("speed_vmax", 2.2)),
        speed_kp=float(defaults.get("speed_kp", 0.35)),
        speed_ki=float(defaults.get("speed_ki", 0.08)),
        speed_kff=float(defaults.get("speed_kff", 0.10)),
        allow_reverse=bool(defaults.get("allow_reverse", False)),
        max_throttle=float(defaults.get("adapter_max_throttle", 0.3)),
        control_dt=float(defaults.get("control_dt", 0.05)),
        total_timesteps=1,
        delta_max=float(defaults.get("delta_max", 0.35)),
        enable_lpf=bool(defaults.get("enable_lpf", True)),
        beta=float(defaults.get("beta", 0.6)),
        w_d=float(defaults.get("w_d", 0.04)),
        w_dd=float(defaults.get("w_dd", 0.01)),
        w_m=float(defaults.get("w_m", 0.0)),
        w_sat=float(defaults.get("w_sat", 0.0)),
        w_time=float(defaults.get("w_time", 0.01)),
        w_center=float(defaults.get("w_center", 0.03)),
        w_heading=float(defaults.get("w_heading", 0.015)),
        w_speed_ref=float(defaults.get("w_speed_ref", 0.0)),
        speed_ref_vmin=float(defaults.get("speed_ref_vmin", 0.35)),
        speed_ref_vmax=float(defaults.get("speed_ref_vmax", 2.2)),
        speed_ref_kappa_ref=float(defaults.get("speed_ref_kappa_ref", 0.15)),
        lap_reward_scale=float(defaults.get("lap_reward_scale", 1.0)),
        progress_reward_scale=float(defaults.get("progress_reward_scale", 48.0)),
        survival_reward_scale=float(defaults.get("survival_reward_scale", 0.30)),
        collision_penalty_base=float(curriculum_values.get("collision_penalty_base", defaults.get("collision_penalty_base", 8.0))),
        offtrack_penalty_base=float(curriculum_values.get("offtrack_penalty_base", defaults.get("offtrack_penalty_base", 5.0))),
        adaptive_delta_max=bool(defaults.get("adaptive_delta_max", True)),
        curve_delta_boost=float(defaults.get("curve_delta_boost", 1.0)),
        curve_kappa_ref=float(defaults.get("curve_kappa_ref", 0.15)),
        steer_intent_boost=float(defaults.get("steer_intent_boost", 0.30)),
        hairpin_curve_ratio=float(defaults.get("hairpin_curve_ratio", 0.85)),
        hairpin_min_delta_max=float(defaults.get("hairpin_min_delta_max", 0.45)),
        hairpin_max_delta_max=float(defaults.get("hairpin_max_delta_max", 0.85)),
        w_near_offtrack=float(defaults.get("w_near_offtrack", 0.55)),
        near_offtrack_start_ratio=float(defaults.get("near_offtrack_start_ratio", 0.45)),
        w_near_collision=float(curriculum_values.get("w_near_collision", defaults.get("w_near_collision", 0.24))),
        near_collision_start_ratio=float(curriculum_values.get("near_collision_start_ratio", defaults.get("near_collision_start_ratio", 0.65))),
        overtake_success_bonus=float(curriculum_values.get("overtake_success_bonus", defaults.get("overtake_success_bonus", 3.0))),
        reward_safe_follow_bonus=float(defaults.get("reward_safe_follow_bonus", 0.02)),
        reward_prepare_pass_bonus=float(defaults.get("reward_prepare_pass_bonus", 0.04)),
        reward_commit_pass_bonus=float(defaults.get("reward_commit_pass_bonus", 0.04)),
        reward_post_pass_bonus=float(defaults.get("reward_post_pass_bonus", 0.5)),
        reward_post_pass_steps=int(defaults.get("reward_post_pass_steps", 10)),
        offtrack_leniency_ratio=float(defaults.get("offtrack_leniency_ratio", 0.25)),
        offtrack_leniency_mult=float(defaults.get("offtrack_leniency_mult", 2.5)),
        snapshot_dir="",
        snapshot_max_steps=0,
        min_episodes_per_scene=1,
        max_steps_per_scene=int(defaults.get("max_steps_per_scene", 640)),
        enable_dynamic_scene_weights=False,
        dynamic_weight_update_episodes=int(defaults.get("dynamic_weight_update_episodes", 24)),
        dynamic_weight_window=int(defaults.get("dynamic_weight_window", 50)),
        dynamic_min_samples_per_scene=int(defaults.get("dynamic_min_samples_per_scene", 6)),
        dynamic_weight_alpha=float(defaults.get("dynamic_weight_alpha", 1.6)),
        dynamic_length_beta=float(defaults.get("dynamic_length_beta", 1.0)),
        dynamic_weight_smoothing=float(defaults.get("dynamic_weight_smoothing", 0.35)),
        dynamic_weight_min=float(defaults.get("dynamic_weight_min", 0.02)),
        dynamic_weight_max=float(defaults.get("dynamic_weight_max", 0.55)),
        dynamic_success_mode=str(defaults.get("dynamic_success_mode", "scene_adaptive")),
        dynamic_success_warmup_episodes=int(defaults.get("dynamic_success_warmup_episodes", 1200)),
        dynamic_success_post_warmup_scale=float(defaults.get("dynamic_success_post_warmup_scale", 0.20)),
        dynamic_success_deficit_mix=float(defaults.get("dynamic_success_deficit_mix", 0.85)),
        enable_step_balance_sampling=False,
        step_balance_sampling_mix=float(defaults.get("step_balance_sampling_mix", 0.3)),
        obstacle_enabled=bool(curriculum_values.get("obstacle_enabled", defaults.get("obstacle_enabled", True))),
        obstacle_count=int(curriculum_values.get("obstacle_count", defaults.get("obstacle_count", 2))),
        obstacle_free_prob=float(curriculum_values.get("obstacle_free_prob", defaults.get("obstacle_free_prob", 0.15))),
        obstacle_modes=curriculum_values.get("obstacle_modes", defaults.get("obstacle_modes")),
        ws_obstacle_free_prob=curriculum_values.get("ws_obstacle_free_prob", defaults.get("ws_obstacle_free_prob")),
        obstacle_spawn_ahead_min_m=float(curriculum_values.get("obstacle_spawn_ahead_min_m", defaults.get("obstacle_spawn_ahead_min_m", 3.5))),
        obstacle_spawn_ahead_max_m=float(curriculum_values.get("obstacle_spawn_ahead_max_m", defaults.get("obstacle_spawn_ahead_max_m", 14.0))),
        obstacle_min_agent_planar_dist_m=float(curriculum_values.get("obstacle_min_agent_planar_dist_m", defaults.get("obstacle_min_agent_planar_dist_m", 1.5))),
        obstacle_min_agent_arc_dist_m=float(curriculum_values.get("obstacle_min_agent_arc_dist_m", defaults.get("obstacle_min_agent_arc_dist_m", 3.5))),
        obstacle_min_separation_world=float(curriculum_values.get("obstacle_min_separation_world", defaults.get("obstacle_min_separation_world", 3.0))),
        obstacle_lateral_choices=curriculum_values.get("obstacle_lateral_choices", defaults.get("obstacle_lateral_choices")),
        obstacle_fixed_progress_ratio=curriculum_values.get("obstacle_fixed_progress_ratio", defaults.get("obstacle_fixed_progress_ratio")),
        obstacle_fixed_progress_gap=curriculum_values.get("obstacle_fixed_progress_gap", defaults.get("obstacle_fixed_progress_gap")),
        obstacle_fixed_progress_gap_min=curriculum_values.get("obstacle_fixed_progress_gap_min", defaults.get("obstacle_fixed_progress_gap_min")),
        obstacle_fixed_progress_gap_max=curriculum_values.get("obstacle_fixed_progress_gap_max", defaults.get("obstacle_fixed_progress_gap_max")),
        obstacle_progress_min=curriculum_values.get("obstacle_progress_min", defaults.get("obstacle_progress_min")),
        obstacle_progress_max=curriculum_values.get("obstacle_progress_max", defaults.get("obstacle_progress_max")),
        obstacle_fixed_lateral_ratio=curriculum_values.get("obstacle_fixed_lateral_ratio", defaults.get("obstacle_fixed_lateral_ratio")),
        gt_obstacle_start_exclusion_half_width_m=curriculum_values.get("gt_obstacle_start_exclusion_half_width_m", defaults.get("gt_obstacle_start_exclusion_half_width_m")),
        ws_obstacle_modes=curriculum_values.get("ws_obstacle_modes", defaults.get("ws_obstacle_modes")),
        ws_obstacle_fixed_progress_ratio=curriculum_values.get("ws_obstacle_fixed_progress_ratio", defaults.get("ws_obstacle_fixed_progress_ratio")),
        ws_obstacle_progress_min=curriculum_values.get("ws_obstacle_progress_min", defaults.get("ws_obstacle_progress_min")),
        ws_obstacle_progress_max=curriculum_values.get("ws_obstacle_progress_max", defaults.get("ws_obstacle_progress_max")),
        ws_obstacle_fixed_lateral_ratio=curriculum_values.get("ws_obstacle_fixed_lateral_ratio", defaults.get("ws_obstacle_fixed_lateral_ratio")),
        obstacle_randomize_non_lane_pid_yaw=bool(curriculum_values.get("obstacle_randomize_non_lane_pid_yaw", defaults.get("obstacle_randomize_non_lane_pid_yaw", True))),
        obstacle_lane_pid_speed_gt=float(curriculum_values.get("obstacle_lane_pid_speed_gt", defaults.get("obstacle_lane_pid_speed_gt", 0.85))),
        obstacle_lane_pid_speed_ws=float(curriculum_values.get("obstacle_lane_pid_speed_ws", defaults.get("obstacle_lane_pid_speed_ws", 0.70))),
        obstacle_lane_pid_lookahead_m=float(defaults.get("obstacle_lane_pid_lookahead_m", 0.9)),
        obstacle_jitter_amplitude_m=float(defaults.get("obstacle_jitter_amplitude_m", 0.10)),
        obstacle_jitter_period_s=float(defaults.get("obstacle_jitter_period_s", 1.5)),
        obstacle_jitter_update_hz=float(defaults.get("obstacle_jitter_update_hz", 8.0)),
        obstacle_nudge_amplitude_m=float(defaults.get("obstacle_nudge_amplitude_m", 0.14)),
        obstacle_nudge_period_s=float(defaults.get("obstacle_nudge_period_s", 1.5)),
        obstacle_nudge_update_hz=float(defaults.get("obstacle_nudge_update_hz", 8.0)),
        obstacle_seed=seed,
        ego_random_spawn=bool(defaults.get("ego_random_spawn", False)),
        ego_spawn_lateral_ratio=float(defaults.get("ego_spawn_lateral_ratio", 0.5)),
        sim2real_json=str(sim2real_json) if sim2real_json else None,
        image_channel_indices=list(image_channel_indices) if image_channel_indices is not None else [0, 1, 2, 3, 5],
        lidar_num_sectors=int(lidar_num_sectors),
        lidar_fov_deg=float(lidar_fov_deg),
        lidar_max_range_m=float(max_range_m),
        lidar_near_clip_m=float(defaults.get("lidar_near_clip_m", 0.18)),
        lidar_repeat_min_steps=int(defaults.get("lidar_repeat_min_steps", 2)),
        lidar_repeat_max_steps=int(defaults.get("lidar_repeat_max_steps", 4)),
    )


def _build_behavior_obs(
    policy_format: str,
    obs: Dict[str, np.ndarray],
    info: Dict[str, Any],
    env: MultiSceneEnvV17,
    speed_vmax: float,
) -> Dict[str, np.ndarray]:
    if policy_format == "v17":
        return obs
    if policy_format != "v16":
        raise ValueError(f"unsupported policy_format={policy_format!r}")
    image = np.asarray(obs["image"], dtype=np.float32)
    if image.shape[0] != 6:
        raise ValueError(f"V16 behavior policy expects 6 image channels, got shape={image.shape}")
    state = _build_state_v16(
        info=info,
        action_safety_wrapper=getattr(env, "action_safety_wrapper", None),
        control_wrapper=getattr(env, "action_adapter_wrapper", None),
        v_max=float(speed_vmax),
    )
    return {
        "image": image,
        "state": np.asarray(state, dtype=np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export V17 world-model dataset")
    parser.add_argument("--env-ids", nargs="+", default=["donkey-generated-track-v0"])
    parser.add_argument("--scene-weights", nargs="+", type=float, default=None)
    parser.add_argument("--policy-path", type=str, default=None, help="Optional RecurrentPPO checkpoint for rollout policy")
    parser.add_argument("--policy-format", type=str, choices=("v17", "v16"), default="v17")
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--max-episode-steps", type=int, default=640)
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--sim", type=str, default="remote")
    parser.add_argument("--sim-start-delay", type=float, default=8.0)
    parser.add_argument("--curriculum-phase", type=str, default=None)
    parser.add_argument("--obs-size", type=int, default=128)
    parser.add_argument("--camera-obs-size", type=int, default=64)
    parser.add_argument("--lidar-num-sectors", type=int, default=36)
    parser.add_argument("--lidar-fov-deg", type=float, default=180.0)
    parser.add_argument("--lidar-max-range-m", type=float, default=20.0)
    parser.add_argument("--lidar-noise", type=float, default=0.5)
    parser.add_argument("--lidar-offset-x", type=float, default=0.0)
    parser.add_argument("--lidar-offset-y", type=float, default=0.40)
    parser.add_argument("--lidar-offset-z", type=float, default=0.5)
    parser.add_argument("--lidar-rot-x", type=float, default=0.0)
    parser.add_argument("--sim2real-json", type=str, default=None)
    parser.add_argument("--passable-gap-threshold-m", type=float, default=0.70)
    parser.add_argument("--gap-label-source", type=str, choices=("sensor", "track"), default="sensor")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    v17._install_sim_wait_timeout_patch(timeout_s=35.0, resend_scene_names_s=3.0)
    defaults = _train_defaults()
    env_ids = list(args.env_ids)
    scene_weights = (
        [float(x) for x in args.scene_weights]
        if args.scene_weights is not None
        else [1.0 / max(1, len(env_ids))] * len(env_ids)
    )
    env = _make_env(
        env_ids=env_ids,
        scene_weights=scene_weights,
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
        image_channel_indices=[0, 1, 2, 3, 4, 5] if args.policy_path and args.policy_format == "v16" else [0, 1, 2, 3, 5],
        lidar_config_override={
            "noise": float(args.lidar_noise),
            "offset_x": float(args.lidar_offset_x),
            "offset_y": float(args.lidar_offset_y),
            "offset_z": float(args.lidar_offset_z),
            "rot_x": float(args.lidar_rot_x),
        },
        sim2real_json=args.sim2real_json,
    )

    model = None
    lstm_state = None
    episode_start = np.array([True], dtype=bool)
    if args.policy_path:
        if RecurrentPPO is None:
            raise RuntimeError("sb3_contrib is required to load a rollout policy")
        model = RecurrentPPO.load(args.policy_path)

    records: Dict[str, List[np.ndarray]] = {
        "camera": [],
        "ego8": [],
        "lidar": [],
        "async_meta": [],
        "target_rel": [],
        "target_rel_mask": [],
        "target_gap": [],
        "target_collision": [],
        "target_ttc": [],
        "target_safety_valid": [],
        "target_passable": [],
        "target_closing_rate": [],
        "target_overtake_progress": [],
        "target_opportunity_valid": [],
        "episode_id": [],
        "step_in_episode": [],
        "scene_id": [],
        "done": [],
    }
    scene_to_id: Dict[str, int] = {}
    scene_counter: Counter[str] = Counter()

    sample_count = 0
    episode_id = 0
    control_dt = float(defaults.get("control_dt", 0.05))
    speed_vmax = float(defaults.get("speed_vmax", 2.2))
    try:
        while sample_count < int(args.samples):
            obs = env.reset()
            episode_id += 1
            done = False
            step_in_episode = 0

            # Warm up one real transition; reset() only carries synthetic zero info.
            if model is None:
                action = env.action_space.sample()
            else:
                warm_info = getattr(env, "_last_info", {}) if hasattr(env, "_last_info") else {}
                policy_obs = _build_behavior_obs(
                    policy_format=str(args.policy_format),
                    obs=obs,
                    info=dict(warm_info or {}),
                    env=env,
                    speed_vmax=speed_vmax,
                )
                action, lstm_state = model.predict(
                    policy_obs,
                    state=lstm_state,
                    episode_start=episode_start,
                    deterministic=bool(args.deterministic),
                )
            obs_cur, _reward, done, info_cur = env.step(action)
            episode_start = np.array([bool(done)], dtype=bool)
            if done:
                lstm_state = None
                continue

            prev_obs = obs_cur
            prev_info = dict(info_cur)
            while (not done) and step_in_episode < int(args.max_episode_steps) and sample_count < int(args.samples):
                if model is None:
                    action = env.action_space.sample()
                else:
                    policy_obs = _build_behavior_obs(
                        policy_format=str(args.policy_format),
                        obs=prev_obs,
                        info=prev_info,
                        env=env,
                        speed_vmax=speed_vmax,
                    )
                    action, lstm_state = model.predict(
                        policy_obs,
                        state=lstm_state,
                        episode_start=episode_start,
                        deterministic=bool(args.deterministic),
                    )
                obs_next, _reward, done, info_next = env.step(action)
                episode_start = np.array([bool(done)], dtype=bool)

                cur_ego8, _ = _build_ego8(prev_obs, prev_info, control_dt=control_dt)
                next_ego8, _ = _build_ego8(obs_next, info_next, control_dt=control_dt)
                del next_ego8
                targets = _build_targets(
                    cur_obs=prev_obs,
                    cur_info=prev_info,
                    next_obs=obs_next,
                    next_info=info_next,
                    control_dt=control_dt,
                    max_range_m=float(args.lidar_max_range_m),
                    passable_gap_threshold_m=float(args.passable_gap_threshold_m),
                    gap_label_source=str(args.gap_label_source),
                    track_geometry=getattr(env, "track_geometry", None),
                )

                scene_key = str(prev_info.get("scene_key", "unknown"))
                if scene_key not in scene_to_id:
                    scene_to_id[scene_key] = len(scene_to_id)
                scene_id = scene_to_id[scene_key]
                scene_counter[scene_key] += 1

                records["camera"].append(_build_camera_tensor(prev_obs, camera_obs_size=int(args.camera_obs_size)))
                records["ego8"].append(cur_ego8.astype(np.float32))
                records["lidar"].append(np.asarray(prev_obs["lidar"], dtype=np.float32))
                records["async_meta"].append(_build_async_meta(prev_info))
                for key in (
                    "target_rel",
                    "target_rel_mask",
                    "target_gap",
                    "target_collision",
                    "target_ttc",
                    "target_safety_valid",
                    "target_passable",
                    "target_closing_rate",
                    "target_overtake_progress",
                    "target_opportunity_valid",
                ):
                    records[key].append(np.asarray(targets[key], dtype=np.float32))
                records["episode_id"].append(np.array(episode_id, dtype=np.int64))
                records["step_in_episode"].append(np.array(step_in_episode, dtype=np.int64))
                records["scene_id"].append(np.array(scene_id, dtype=np.int64))
                records["done"].append(np.array(float(done), dtype=np.float32))

                sample_count += 1
                step_in_episode += 1
                prev_obs = obs_next
                prev_info = dict(info_next)

            if done:
                lstm_state = None
    finally:
        env.close()

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        **{k: np.asarray(v) for k, v in records.items()},
    )
    meta_path = out_path.with_suffix(".json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "output_npz": str(out_path),
                "samples": int(sample_count),
                "env_ids": env_ids,
                "scene_weights": scene_weights,
                "scene_to_id": scene_to_id,
                "scene_counts": dict(scene_counter),
                "policy_path": args.policy_path,
                "policy_format": args.policy_format,
                "curriculum_phase": args.curriculum_phase,
                "sim_path": args.sim,
                "sim_start_delay": float(args.sim_start_delay),
                "camera_obs_size": int(args.camera_obs_size),
                "camera_channels": 5,
                "lidar_num_sectors": int(args.lidar_num_sectors),
                "lidar_fov_deg": float(args.lidar_fov_deg),
                "lidar_max_range_m": float(args.lidar_max_range_m),
                "lidar_noise": float(args.lidar_noise),
                "lidar_offset_x": float(args.lidar_offset_x),
                "lidar_offset_y": float(args.lidar_offset_y),
                "lidar_offset_z": float(args.lidar_offset_z),
                "lidar_rot_x": float(args.lidar_rot_x),
                "sim2real_json": None if args.sim2real_json in ("", None) else str(args.sim2real_json),
                "passable_gap_threshold_m": float(args.passable_gap_threshold_m),
                "gap_label_source": str(args.gap_label_source),
                "seed": int(args.seed),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"saved: {out_path}")
    print(f"meta : {meta_path}")
    print(f"samples={sample_count}")


if __name__ == "__main__":
    main()
