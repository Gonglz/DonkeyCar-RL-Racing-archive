#!/usr/bin/env python3
"""
Export supervised obstacle-context frames for learned V1.

This uses the same runtime truth and visibility logic as the obstacle-context
evaluation script, but stores frame-level samples for offline training.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, Dict, List, Sequence

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

from eval_obstacle_context_compare import _runtime_visible_projection, p as eval_p

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


def _train_defaults() -> Dict[str, Any]:
    import inspect

    sig = inspect.signature(eval_p.train_v16)
    return {
        name: param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect._empty
    }


def _build_conf(port: int) -> Dict[str, Any]:
    cfg = eval_p.load_config(myconfig=eval_p.DEFAULT_MYCONFIG)
    if cfg is not None and hasattr(cfg, "GYM_CONF"):
        conf = cfg.GYM_CONF.copy()
        conf.update(
            {
                "host": "127.0.0.1",
                "port": int(port),
                "car_name": "waveshare_v16_ctx_export",
                "racer_name": "V16-Context-Export",
                "country": "CN",
                "bio": "V16 obstacle context export",
                "guid": "waveshare-v16-context-export",
                "max_cte": 8.0,
            }
        )
        return conf
    return {
        "host": "127.0.0.1",
        "port": int(port),
        "car_name": "waveshare_v16_ctx_export",
        "racer_name": "V16-Context-Export",
        "country": "CN",
        "bio": "V16 obstacle context export",
        "guid": "waveshare-v16-context-export",
        "max_cte": 8.0,
    }


def _curriculum_values(defaults: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "scene_weights",
        "enable_dynamic_scene_weights",
        "enable_step_balance_sampling",
        "obstacle_enabled",
        "obstacle_count",
        "obstacle_free_prob",
        "obstacle_modes",
        "ws_obstacle_free_prob",
        "obstacle_spawn_ahead_min_m",
        "obstacle_spawn_ahead_max_m",
        "obstacle_min_agent_planar_dist_m",
        "obstacle_min_agent_arc_dist_m",
        "obstacle_fixed_progress_ratio",
        "obstacle_fixed_progress_gap",
        "obstacle_fixed_progress_gap_min",
        "obstacle_fixed_progress_gap_max",
        "obstacle_progress_min",
        "obstacle_progress_max",
        "obstacle_lateral_choices",
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
    ]
    return {k: defaults.get(k) for k in keys}


def _sampling_profile_overrides(profile: str) -> Dict[str, Any]:
    name = str(profile or "default").strip().lower()
    if name in ("", "default", "none"):
        return {
            "curriculum_overrides": {},
            "scene_reward_overrides": {},
            "scene_max_cte": {},
        }
    if name == "interaction_single":
        reward_overrides = {
            "safe_follow_bonus_scale": 0.18,
            "post_pass_stability_bonus": 1.2,
            "post_pass_stability_steps": 10,
            "overtake_success_bonus": 3.0,
        }
        return {
            "curriculum_overrides": {
                "obstacle_enabled": True,
                "obstacle_count": 1,
                "obstacle_free_prob": 0.0,
                "ws_obstacle_free_prob": 0.0,
                "obstacle_modes": ["lane_pid"],
                "ws_obstacle_modes": ["lane_pid"],
                "obstacle_progress_min": 0.14,
                "obstacle_progress_max": 0.30,
                "ws_obstacle_progress_min": 0.28,
                "ws_obstacle_progress_max": 0.44,
                "obstacle_fixed_lateral_ratio": 0.5,
                "ws_obstacle_fixed_lateral_ratio": 0.5,
                "obstacle_lane_pid_speed_gt": 0.42,
                "obstacle_lane_pid_speed_ws": 0.30,
                "overtake_success_bonus": 3.0,
            },
            "scene_reward_overrides": {
                "waveshare": dict(reward_overrides),
                "generated_track": dict(reward_overrides),
            },
            "scene_max_cte": {
                "waveshare": 9.0,
                "generated_track": 9.5,
            },
        }
    raise KeyError(f"Unknown sampling_profile={profile}")


def _make_scene_specs_with_overrides(profile_overrides: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    scene_specs = copy.deepcopy(eval_p.SCENE_SPECS)
    reward_override_map = dict(profile_overrides.get("scene_reward_overrides", {}) or {})
    max_cte_map = dict(profile_overrides.get("scene_max_cte", {}) or {})
    for _env_id, spec in scene_specs.items():
        scene_key = str(spec.get("scene_key", "") or "")
        if scene_key in reward_override_map:
            merged = dict(spec.get("reward_overrides", {}) or {})
            merged.update(dict(reward_override_map[scene_key]))
            spec["reward_overrides"] = merged
        if scene_key in max_cte_map:
            spec["max_cte"] = float(max_cte_map[scene_key])
    return scene_specs


def _make_env_compat(scene_env_id: str, phase: str, port: int, seed: int, sampling_profile: str = "default"):
    defaults = _train_defaults()
    curriculum_values = _curriculum_values(defaults)
    _, _ = eval_p._apply_curriculum_phase(phase, curriculum_values)
    profile_overrides = _sampling_profile_overrides(sampling_profile)
    curriculum_values.update(dict(profile_overrides.get("curriculum_overrides", {}) or {}))
    env_ids = [scene_env_id]
    track_dir = eval_p._resolve_track_dir(track_dir=str(defaults.get("track_dir", "")), env_ids=env_ids)
    scene_specs = _make_scene_specs_with_overrides(profile_overrides)
    track_geometry = eval_p.TrackGeometryManager(track_dir=track_dir, env_ids=env_ids, scene_specs=scene_specs)
    conf = _build_conf(port)

    obs_size = int(defaults.get("obs_size", 128))
    snapshot_dir = os.path.join("/tmp", "v16_context_export_snapshots", f"{phase}_{scene_env_id}")
    os.makedirs(snapshot_dir, exist_ok=True)

    return eval_p.MultiSceneEnvV16(
        env_ids=env_ids,
        conf=conf,
        scene_weights=[1.0],
        scene_specs=scene_specs,
        track_geometry=track_geometry,
        track_dir=track_dir,
        obs_size=obs_size,
        augment=bool(defaults.get("augment", True)),
        yellow_dropout_prob=float(defaults.get("yellow_dropout_prob", 0.0)),
        dropout_start_step=int(defaults.get("dropout_start_step", 0)),
        dropout_ramp_steps=int(defaults.get("dropout_ramp_steps", 1)),
        adapter_k_delta=float(defaults.get("adapter_k_delta", 0.9)),
        adapter_lambda_bias=float(defaults.get("adapter_lambda_bias", 0.35)),
        adapter_k_bias=float(defaults.get("adapter_k_bias", 0.18)),
        adapter_steer_core_decay=float(defaults.get("adapter_steer_core_decay", 0.78)),
        adapter_v_nominal=float(defaults.get("adapter_v_nominal", 0.6)),
        adapter_k_turn=float(defaults.get("adapter_k_turn", 0.8)),
        adapter_k_bias_speed=float(defaults.get("adapter_k_bias_speed", 0.35)),
        adapter_alpha_speed=float(defaults.get("adapter_alpha_speed", 0.25)),
        adapter_v_min=float(defaults.get("adapter_v_min", 0.25)),
        adapter_v_max=float(defaults.get("adapter_v_max", 1.2)),
        speed_vmax=float(defaults.get("speed_vmax", 1.0)),
        speed_kp=float(defaults.get("speed_kp", 0.7)),
        speed_ki=float(defaults.get("speed_ki", 0.12)),
        speed_kff=float(defaults.get("speed_kff", 0.3)),
        allow_reverse=bool(defaults.get("allow_reverse", False)),
        max_throttle=float(defaults.get("adapter_max_throttle", 0.6)),
        control_dt=float(defaults.get("control_dt", 0.05)),
        total_timesteps=1,
        delta_max=float(defaults.get("delta_max", 0.2617993877991494)),
        enable_lpf=bool(defaults.get("enable_lpf", True)),
        beta=float(defaults.get("beta", 0.7)),
        w_d=float(defaults.get("w_d", 1.0)),
        w_dd=float(defaults.get("w_dd", 0.0)),
        w_m=float(defaults.get("w_m", 0.0)),
        w_sat=float(defaults.get("w_sat", 0.0)),
        w_time=float(defaults.get("w_time", 0.0)),
        w_center=float(defaults.get("w_center", 0.03)),
        w_heading=float(defaults.get("w_heading", 0.015)),
        w_speed_ref=float(defaults.get("w_speed_ref", 0.0)),
        speed_ref_vmin=float(defaults.get("speed_ref_vmin", 0.3)),
        speed_ref_vmax=float(defaults.get("speed_ref_vmax", 1.0)),
        speed_ref_kappa_ref=float(defaults.get("speed_ref_kappa_ref", 0.45)),
        lap_reward_scale=float(defaults.get("lap_reward_scale", 1.0)),
        progress_reward_scale=float(defaults.get("progress_reward_scale", 48.0)),
        survival_reward_scale=float(defaults.get("survival_reward_scale", 0.3)),
        collision_penalty_base=float(curriculum_values.get("collision_penalty_base", defaults.get("collision_penalty_base", 8.0))),
        offtrack_penalty_base=float(curriculum_values.get("offtrack_penalty_base", defaults.get("offtrack_penalty_base", 5.0))),
        adaptive_delta_max=bool(defaults.get("adaptive_delta_max", True)),
        curve_delta_boost=float(defaults.get("curve_delta_boost", 0.08)),
        curve_kappa_ref=float(defaults.get("curve_kappa_ref", 0.42)),
        steer_intent_boost=float(defaults.get("steer_intent_boost", 1.0)),
        hairpin_curve_ratio=float(defaults.get("hairpin_curve_ratio", 0.85)),
        hairpin_min_delta_max=float(defaults.get("hairpin_min_delta_max", 0.3490658503988659)),
        hairpin_max_delta_max=float(defaults.get("hairpin_max_delta_max", 0.6108652381980153)),
        w_near_offtrack=float(defaults.get("w_near_offtrack", 0.45)),
        near_offtrack_start_ratio=float(defaults.get("near_offtrack_start_ratio", 0.82)),
        w_near_collision=float(curriculum_values.get("w_near_collision", defaults.get("w_near_collision", 0.2))),
        near_collision_start_ratio=float(curriculum_values.get("near_collision_start_ratio", defaults.get("near_collision_start_ratio", 0.66))),
        overtake_success_bonus=float(curriculum_values.get("overtake_success_bonus", defaults.get("overtake_success_bonus", 2.5))),
        offtrack_leniency_ratio=float(defaults.get("offtrack_leniency_ratio", 0.25)),
        offtrack_leniency_mult=float(defaults.get("offtrack_leniency_mult", 2.5)),
        snapshot_dir=snapshot_dir,
        snapshot_max_steps=0,
        min_episodes_per_scene=1,
        max_steps_per_scene=10**9,
        enable_dynamic_scene_weights=False,
        dynamic_weight_update_episodes=int(defaults.get("dynamic_weight_update_episodes", 50)),
        dynamic_weight_window=int(defaults.get("dynamic_weight_window", 50)),
        dynamic_min_samples_per_scene=int(defaults.get("dynamic_min_samples_per_scene", 12)),
        dynamic_weight_alpha=float(defaults.get("dynamic_weight_alpha", 1.6)),
        dynamic_length_beta=float(defaults.get("dynamic_length_beta", 0.35)),
        dynamic_weight_smoothing=float(defaults.get("dynamic_weight_smoothing", 0.25)),
        dynamic_weight_min=float(defaults.get("dynamic_weight_min", 0.2)),
        dynamic_weight_max=float(defaults.get("dynamic_weight_max", 0.8)),
        dynamic_success_mode=str(defaults.get("dynamic_success_mode", "soft_lap")),
        dynamic_success_warmup_episodes=int(defaults.get("dynamic_success_warmup_episodes", 120)),
        dynamic_success_post_warmup_scale=float(defaults.get("dynamic_success_post_warmup_scale", 0.6)),
        dynamic_success_deficit_mix=float(defaults.get("dynamic_success_deficit_mix", 0.35)),
        enable_step_balance_sampling=False,
        step_balance_sampling_mix=float(defaults.get("step_balance_sampling_mix", 0.5)),
        obstacle_enabled=bool(curriculum_values.get("obstacle_enabled", True)),
        obstacle_count=int(curriculum_values.get("obstacle_count", 1)),
        obstacle_free_prob=float(curriculum_values.get("obstacle_free_prob", 0.15)),
        obstacle_modes=curriculum_values.get("obstacle_modes"),
        ws_obstacle_free_prob=curriculum_values.get("ws_obstacle_free_prob"),
        obstacle_spawn_ahead_min_m=float(curriculum_values.get("obstacle_spawn_ahead_min_m", 4.0)),
        obstacle_spawn_ahead_max_m=float(curriculum_values.get("obstacle_spawn_ahead_max_m", 10.0)),
        obstacle_min_agent_planar_dist_m=float(curriculum_values.get("obstacle_min_agent_planar_dist_m", 1.5)),
        obstacle_min_agent_arc_dist_m=float(curriculum_values.get("obstacle_min_agent_arc_dist_m", 3.5)),
        obstacle_min_separation_world=float(defaults.get("obstacle_min_separation_world", 3.0)),
        obstacle_lateral_choices=curriculum_values.get("obstacle_lateral_choices"),
        obstacle_fixed_progress_ratio=curriculum_values.get("obstacle_fixed_progress_ratio"),
        obstacle_fixed_progress_gap=curriculum_values.get("obstacle_fixed_progress_gap"),
        obstacle_fixed_progress_gap_min=curriculum_values.get("obstacle_fixed_progress_gap_min"),
        obstacle_fixed_progress_gap_max=curriculum_values.get("obstacle_fixed_progress_gap_max"),
        obstacle_progress_min=curriculum_values.get("obstacle_progress_min"),
        obstacle_progress_max=curriculum_values.get("obstacle_progress_max"),
        obstacle_fixed_lateral_ratio=curriculum_values.get("obstacle_fixed_lateral_ratio"),
        gt_obstacle_start_exclusion_half_width_m=curriculum_values.get("gt_obstacle_start_exclusion_half_width_m"),
        ws_obstacle_modes=curriculum_values.get("ws_obstacle_modes"),
        ws_obstacle_fixed_progress_ratio=curriculum_values.get("ws_obstacle_fixed_progress_ratio"),
        ws_obstacle_progress_min=curriculum_values.get("ws_obstacle_progress_min"),
        ws_obstacle_progress_max=curriculum_values.get("ws_obstacle_progress_max"),
        ws_obstacle_fixed_lateral_ratio=curriculum_values.get("ws_obstacle_fixed_lateral_ratio"),
        obstacle_randomize_non_lane_pid_yaw=bool(curriculum_values.get("obstacle_randomize_non_lane_pid_yaw", False)),
        obstacle_lane_pid_speed_gt=float(curriculum_values.get("obstacle_lane_pid_speed_gt", 0.45)),
        obstacle_lane_pid_speed_ws=float(curriculum_values.get("obstacle_lane_pid_speed_ws", 0.45)),
        obstacle_lane_pid_lookahead_m=float(defaults.get("obstacle_lane_pid_lookahead_m", 0.9)),
        obstacle_jitter_amplitude_m=float(defaults.get("obstacle_jitter_amplitude_m", 0.10)),
        obstacle_jitter_period_s=float(defaults.get("obstacle_jitter_period_s", 1.5)),
        obstacle_jitter_update_hz=float(defaults.get("obstacle_jitter_update_hz", 8.0)),
        obstacle_nudge_amplitude_m=float(defaults.get("obstacle_nudge_amplitude_m", 0.14)),
        obstacle_nudge_period_s=float(defaults.get("obstacle_nudge_period_s", 1.5)),
        obstacle_nudge_update_hz=float(defaults.get("obstacle_nudge_update_hz", 8.0)),
        obstacle_seed=int(seed),
        ego_random_spawn=bool(defaults.get("ego_random_spawn", False)),
        ego_spawn_lateral_ratio=float(defaults.get("ego_spawn_lateral_ratio", 0.5)),
    )


def _parse_horizons(spec: str) -> List[int]:
    vals: List[int] = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        val = int(tok)
        if val > 0:
            vals.append(val)
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("future horizons must contain at least one positive integer")
    return vals


def _parse_scene_keys(spec: str) -> List[str]:
    vals: List[str] = []
    for tok in str(spec).split(","):
        tok = tok.strip().lower()
        if tok in ("ws", "gt"):
            vals.append(tok)
    vals = list(dict.fromkeys(vals))
    if not vals:
        raise ValueError("scenes must contain at least one of: ws,gt")
    return vals


def _attach_future_targets(part: Dict[str, Any], horizons: Sequence[int]) -> None:
    episode_id = np.asarray(part["episode_id"], dtype=np.int64)
    step_in_episode = np.asarray(part["step_in_episode"], dtype=np.int64)
    target_present = np.asarray(part["target_present"], dtype=np.float32)
    target_longitudinal = np.asarray(part["target_longitudinal"], dtype=np.float32)
    target_lateral = np.asarray(part["target_lateral"], dtype=np.float32)
    target_dist = np.asarray(part["target_dist"], dtype=np.float32)
    overtake_success = np.asarray(part["overtake_success"], dtype=np.float32)
    n = int(episode_id.shape[0])
    idx = np.arange(n, dtype=np.int64)

    for horizon in horizons:
        future_idx = idx + int(horizon)
        valid = future_idx < n
        valid &= episode_id[future_idx.clip(max=max(n - 1, 0))] == episode_id
        valid &= step_in_episode[future_idx.clip(max=max(n - 1, 0))] == (step_in_episode + int(horizon))

        future_present = np.full((n,), np.nan, dtype=np.float32)
        future_longitudinal = np.full((n,), np.nan, dtype=np.float32)
        future_lateral = np.full((n,), np.nan, dtype=np.float32)
        future_dist = np.full((n,), np.nan, dtype=np.float32)
        future_overtake_any = np.zeros((n,), dtype=np.float32)

        if np.any(valid):
            valid_idx = idx[valid]
            valid_future_idx = future_idx[valid]
            future_present[valid] = target_present[valid_future_idx]
            future_longitudinal[valid] = target_longitudinal[valid_future_idx]
            future_lateral[valid] = target_lateral[valid_future_idx]
            future_dist[valid] = target_dist[valid_future_idx]
            for cur_i, nxt_i in zip(valid_idx.tolist(), valid_future_idx.tolist()):
                future_overtake_any[cur_i] = float(np.any(overtake_success[cur_i + 1 : nxt_i + 1] > 0.5))

        part[f"future_valid_h{horizon}"] = valid.astype(np.float32)
        part[f"future_present_h{horizon}"] = future_present
        part[f"future_longitudinal_h{horizon}"] = future_longitudinal
        part[f"future_lateral_h{horizon}"] = future_lateral
        part[f"future_dist_h{horizon}"] = future_dist
        part[f"future_overtake_success_any_h{horizon}"] = future_overtake_any


def _quantize_image_uint8(image_chw: np.ndarray) -> np.ndarray:
    image = np.asarray(image_chw, dtype=np.float32)
    if image.size == 0:
        return np.asarray(image, dtype=np.uint8)
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    if float(np.max(image)) <= 1.5:
        image = image * 255.0
    return np.clip(np.rint(image), 0.0, 255.0).astype(np.uint8)


def _collect_scene_samples(
    model_path: str,
    scene_env_id: str,
    phase: str,
    port: int,
    episodes: int,
    seed: int,
    max_frames: int,
    future_horizons: Sequence[int],
    sampling_profile: str,
) -> Dict[str, Any]:
    def make_env():
        return _make_env_compat(
            scene_env_id=scene_env_id,
            phase=phase,
            port=port,
            seed=seed,
            sampling_profile=sampling_profile,
        )

    vec_env = DummyVecEnv([make_env])
    model = RecurrentPPO.load(model_path, env=vec_env)

    obs = vec_env.reset()
    lstm_state = None
    episode_start = np.ones((1,), dtype=bool)

    images: List[np.ndarray] = []
    state7: List[np.ndarray] = []
    target_present: List[float] = []
    target_scene_present: List[float] = []
    target_longitudinal: List[float] = []
    target_lateral: List[float] = []
    target_dist: List[float] = []
    visible_proj_width_px: List[float] = []
    visible_proj_height_px: List[float] = []
    scene_ids: List[int] = []
    episode_ids: List[int] = []
    step_ids: List[int] = []
    done_flags: List[float] = []
    action_steer: List[float] = []
    action_throttle: List[float] = []
    safe_follow_bonus: List[float] = []
    overtake_success: List[float] = []
    post_pass_stability_bonus: List[float] = []
    near_collision_risk: List[float] = []
    overtake_armed: List[float] = []
    overtake_front_steps: List[float] = []
    overtake_cooldown: List[float] = []
    post_pass_active: List[float] = []
    post_pass_stable_steps: List[float] = []

    scene_key_map = {
        "donkey-waveshare-v0": 0,
        "donkey-generated-track-v0": 1,
    }
    finished_episodes = 0
    current_episode_id = 0
    current_step_id = 0

    while finished_episodes < int(episodes):
        action, lstm_state = model.predict(
            obs,
            state=lstm_state,
            episode_start=episode_start,
            deterministic=True,
        )
        next_obs, rewards, dones, infos = vec_env.step(action)
        del rewards
        info = infos[0]
        obs_dict = obs
        done = bool(np.asarray(dones, dtype=bool).reshape(-1)[0])

        if isinstance(obs_dict, dict) and "image" in obs_dict and "state" in obs_dict:
            image_chw = np.asarray(obs_dict["image"][0], dtype=np.float32)
            state_vec = np.asarray(obs_dict["state"][0], dtype=np.float32).reshape(-1)
            if image_chw.ndim == 3 and state_vec.size >= 7:
                runtime_present = float(info.get("obstacle_present", 0.0) or 0.0)
                runtime_longitudinal = float(info.get("obstacle_longitudinal", 0.0) or 0.0)
                runtime_lateral = float(info.get("obstacle_lateral", 0.0) or 0.0)
                runtime_dist = float(info.get("obstacle_dist", 0.0) or 0.0)
                visible_diag = _runtime_visible_projection(
                    runtime_present=runtime_present,
                    runtime_longitudinal=runtime_longitudinal,
                    runtime_lateral=runtime_lateral,
                    runtime_dist=runtime_dist,
                    image_w=int(image_chw.shape[2]),
                    image_h=int(image_chw.shape[1]),
                )
                visible_present = float(bool(visible_diag["visible"]))

                images.append(_quantize_image_uint8(image_chw))
                state7.append(state_vec[:7].astype(np.float16))
                target_present.append(visible_present)
                target_scene_present.append(float(runtime_present >= 0.5))
                target_longitudinal.append(runtime_longitudinal)
                target_lateral.append(runtime_lateral)
                target_dist.append(runtime_dist)
                visible_proj_width_px.append(float(visible_diag.get("proj_width_px", np.nan)))
                visible_proj_height_px.append(float(visible_diag.get("proj_height_px", np.nan)))
                scene_ids.append(int(scene_key_map.get(scene_env_id, -1)))
                episode_ids.append(int(current_episode_id))
                step_ids.append(int(current_step_id))
                done_flags.append(float(done))

                action_vec = np.asarray(action, dtype=np.float32).reshape(-1)
                action_steer.append(float(action_vec[0]) if action_vec.size > 0 else 0.0)
                action_throttle.append(float(action_vec[1]) if action_vec.size > 1 else 0.0)

                safe_follow_bonus.append(float(info.get("safe_follow_bonus", 0.0) or 0.0))
                overtake_success.append(float(bool(info.get("overtake_success", False))))
                post_pass_stability_bonus.append(float(info.get("post_pass_stability_bonus", 0.0) or 0.0))
                near_collision_risk.append(float(info.get("reward_debug/near_collision_risk", 0.0) or 0.0))
                overtake_armed.append(float(info.get("reward_debug/overtake_armed", 0.0) or 0.0))
                overtake_front_steps.append(float(info.get("reward_debug/overtake_front_steps", 0.0) or 0.0))
                overtake_cooldown.append(float(info.get("reward_debug/overtake_cooldown", 0.0) or 0.0))
                post_pass_active.append(float(info.get("reward_debug/post_pass_active", 0.0) or 0.0))
                post_pass_stable_steps.append(float(info.get("reward_debug/post_pass_stable_steps", 0.0) or 0.0))

                if max_frames > 0 and len(images) >= max_frames:
                    break

        obs = next_obs
        episode_start = np.array([done], dtype=bool)
        if done:
            finished_episodes += 1
            current_episode_id += 1
            current_step_id = 0
        else:
            current_step_id += 1

    vec_env.close()

    part = {
        "image": np.asarray(images, dtype=np.uint8),
        "state7": np.asarray(state7, dtype=np.float16),
        "target_present": np.asarray(target_present, dtype=np.float16),
        "target_scene_present": np.asarray(target_scene_present, dtype=np.float16),
        "target_longitudinal": np.asarray(target_longitudinal, dtype=np.float16),
        "target_lateral": np.asarray(target_lateral, dtype=np.float16),
        "target_dist": np.asarray(target_dist, dtype=np.float16),
        "visible_proj_width_px": np.asarray(visible_proj_width_px, dtype=np.float16),
        "visible_proj_height_px": np.asarray(visible_proj_height_px, dtype=np.float16),
        "scene_id": np.asarray(scene_ids, dtype=np.int64),
        "episode_id": np.asarray(episode_ids, dtype=np.int64),
        "step_in_episode": np.asarray(step_ids, dtype=np.int64),
        "done": np.asarray(done_flags, dtype=np.float16),
        "action_steer": np.asarray(action_steer, dtype=np.float16),
        "action_throttle": np.asarray(action_throttle, dtype=np.float16),
        "safe_follow_bonus": np.asarray(safe_follow_bonus, dtype=np.float16),
        "overtake_success": np.asarray(overtake_success, dtype=np.float16),
        "post_pass_stability_bonus": np.asarray(post_pass_stability_bonus, dtype=np.float16),
        "near_collision_risk": np.asarray(near_collision_risk, dtype=np.float16),
        "reward_debug_overtake_armed": np.asarray(overtake_armed, dtype=np.float16),
        "reward_debug_overtake_front_steps": np.asarray(overtake_front_steps, dtype=np.float16),
        "reward_debug_overtake_cooldown": np.asarray(overtake_cooldown, dtype=np.float16),
        "reward_debug_post_pass_active": np.asarray(post_pass_active, dtype=np.float16),
        "reward_debug_post_pass_stable_steps": np.asarray(post_pass_stable_steps, dtype=np.float16),
        "episodes_collected": int(finished_episodes),
    }
    _attach_future_targets(part, future_horizons)
    return part


def _concat_parts(parts: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    keys = sorted(
        {
            k
            for part in parts
            for k in part.keys()
            if k != "episodes_collected"
        }
    )
    out: Dict[str, np.ndarray] = {}
    for key in keys:
        out[key] = np.concatenate([np.asarray(p[key]) for p in parts], axis=0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--phase", default="lane_pid_intro")
    ap.add_argument("--sampling-profile", default="default")
    ap.add_argument("--scenes", default="ws,gt")
    ap.add_argument("--episodes-per-scene", type=int, default=8)
    ap.add_argument("--port", type=int, default=9091)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-frames-per-scene", type=int, default=0)
    ap.add_argument("--future-horizons", default="5,10,20")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    future_horizons = _parse_horizons(args.future_horizons)
    scene_keys = _parse_scene_keys(args.scenes)

    scene_map = {
        "ws": "donkey-waveshare-v0",
        "gt": "donkey-generated-track-v0",
    }

    parts: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "phase": str(args.phase),
        "sampling_profile": str(args.sampling_profile),
        "scenes": list(scene_keys),
        "episodes_per_scene": int(args.episodes_per_scene),
        "future_horizons": list(future_horizons),
        "scene_stats": {},
    }
    episode_offset = 0

    for scene_key in scene_keys:
        scene_env_id = scene_map[scene_key]
        print(f"[export] phase={args.phase} scene={scene_key} episodes={args.episodes_per_scene}")
        part = _collect_scene_samples(
            model_path=args.checkpoint,
            scene_env_id=scene_env_id,
            phase=args.phase,
            port=int(args.port),
            episodes=int(args.episodes_per_scene),
            seed=int(args.seed),
            max_frames=int(args.max_frames_per_scene),
            future_horizons=future_horizons,
            sampling_profile=str(args.sampling_profile),
        )
        part["episode_id"] = np.asarray(part["episode_id"], dtype=np.int64) + int(episode_offset)
        episode_offset += int(part["episodes_collected"])
        parts.append(part)
        present = np.asarray(part["target_present"], dtype=np.float32)
        overtake = np.asarray(part["overtake_success"], dtype=np.float32)
        follow = np.asarray(part["safe_follow_bonus"], dtype=np.float32)
        post_pass = np.asarray(part["post_pass_stability_bonus"], dtype=np.float32)
        summary["scene_stats"][scene_key] = {
            "frames": int(present.size),
            "episodes_collected": int(part["episodes_collected"]),
            "visible_present_rate": float(np.mean(present)) if present.size > 0 else 0.0,
            "scene_present_rate": float(np.mean(part["target_scene_present"])) if present.size > 0 else 0.0,
            "overtake_success_rate": float(np.mean(overtake > 0.5)) if overtake.size > 0 else 0.0,
            "safe_follow_nonzero_rate": float(np.mean(follow > 0.0)) if follow.size > 0 else 0.0,
            "post_pass_nonzero_rate": float(np.mean(post_pass > 0.0)) if post_pass.size > 0 else 0.0,
        }

    merged = _concat_parts(parts)
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **merged)

    summary["total_frames"] = int(merged["target_present"].shape[0])
    summary["visible_present_rate"] = float(np.mean(merged["target_present"])) if summary["total_frames"] > 0 else 0.0
    summary["scene_present_rate"] = float(np.mean(merged["target_scene_present"])) if summary["total_frames"] > 0 else 0.0

    meta_path = os.path.splitext(output_path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[saved] dataset={output_path}")
    print(f"[saved] meta={meta_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
