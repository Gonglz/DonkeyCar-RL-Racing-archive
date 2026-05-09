#!/usr/bin/env python
import argparse
import cv2
import inspect
import json
import math
import os
import sys
from collections import Counter
from typing import Any, Dict, List

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import ppo_multitrack_v16 as p  # noqa: E402


CHANNEL_NAMES = ("raw_y", "white_prob", "yellow_prob", "edge", "vehicle_prob", "motion")


def _channel_to_uint8(channel: np.ndarray) -> np.ndarray:
    arr = np.asarray(channel, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    return np.uint8(np.round(arr * 255.0))


def _render_semantic_preview(image_chw: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
    image = np.asarray(image_chw, dtype=np.float32)
    ch_count, height, width = image.shape
    tiles: List[np.ndarray] = []
    bbox = info.get("obstacle_cv_bbox", None)
    if bbox is None:
        bbox = info.get("obstacle_learned_bbox", None)
    for idx in range(min(ch_count, 6)):
        gray = _channel_to_uint8(image[idx])
        tile = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if idx == 4 and bbox is not None and len(bbox) == 4:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(tile, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(
            tile,
            CHANNEL_NAMES[idx],
            (4, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    while len(tiles) < 6:
        tiles.append(np.zeros((height, width, 3), dtype=np.uint8))

    top = np.concatenate(tiles[:3], axis=1)
    bottom = np.concatenate(tiles[3:6], axis=1)
    canvas = np.concatenate([top, bottom], axis=0)
    return canvas


def _frame_error_score(
    runtime_scene_has: bool,
    runtime_visible_has: bool,
    cv_has: bool,
    runtime_longitudinal: float,
    runtime_lateral: float,
    runtime_dist: float,
    runtime_risk: float,
    cv_longitudinal: float,
    cv_lateral: float,
    cv_dist: float,
    cv_risk: float,
) -> float:
    if runtime_visible_has and not cv_has:
        return 100.0 + float(runtime_dist)
    if runtime_scene_has and (not runtime_visible_has):
        return 40.0 + abs(float(runtime_longitudinal)) + 0.5 * float(runtime_dist)
    if (not runtime_visible_has) and cv_has:
        return 80.0 + float(cv_dist)
    if runtime_visible_has and cv_has:
        return (
            abs(cv_longitudinal - runtime_longitudinal)
            + abs(cv_lateral - runtime_lateral)
            + abs(cv_dist - runtime_dist)
            + 0.5 * abs(cv_risk - runtime_risk)
        )
    return -1.0


def _runtime_visible_projection(
    runtime_present: float,
    runtime_longitudinal: float,
    runtime_lateral: float,
    runtime_dist: float,
    camera_half_fov_deg: float = 32.0,
    longitudinal_min_m: float = 0.15,
    dist_max_m: float = 8.5,
    image_w: int = 128,
    image_h: int = 128,
    object_width_m: float = 0.50,
    object_height_m: float = 0.28,
    min_projected_width_px: float = 7.0,
    min_projected_height_px: float = 5.0,
    min_horizontal_margin_px: float = 4.0,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "visible": False,
        "angle_deg": float("nan"),
        "proj_center_x_px": float("nan"),
        "proj_width_px": float("nan"),
        "proj_height_px": float("nan"),
        "proj_left_px": float("nan"),
        "proj_right_px": float("nan"),
        "margin_left_px": float("nan"),
        "margin_right_px": float("nan"),
    }
    if float(runtime_present) < 0.5:
        return out
    if not np.isfinite(runtime_dist) or runtime_dist <= 0.1 or runtime_dist > dist_max_m:
        return out
    if not np.isfinite(runtime_longitudinal) or runtime_longitudinal <= longitudinal_min_m:
        return out
    if not np.isfinite(runtime_lateral):
        return out
    angle = math.atan2(float(runtime_lateral), max(float(runtime_longitudinal), 1e-3))
    out["angle_deg"] = math.degrees(angle)
    half_fov_rad = math.radians(float(camera_half_fov_deg))
    if abs(angle) > half_fov_rad * 1.02:
        return out

    img_w = max(int(image_w), 1)
    img_h = max(int(image_h), 1)
    focal_px = (0.5 * float(img_w)) / max(math.tan(half_fov_rad), 1e-6)
    proj_center_x = (float(img_w) * 0.5) + focal_px * math.tan(angle)
    proj_width = focal_px * float(object_width_m) / max(float(runtime_dist), 1e-3)
    proj_height = focal_px * float(object_height_m) / max(float(runtime_dist), 1e-3)
    proj_left = proj_center_x - 0.5 * proj_width
    proj_right = proj_center_x + 0.5 * proj_width
    margin_left = proj_left
    margin_right = float(img_w) - proj_right

    out.update(
        {
            "proj_center_x_px": float(proj_center_x),
            "proj_width_px": float(proj_width),
            "proj_height_px": float(proj_height),
            "proj_left_px": float(proj_left),
            "proj_right_px": float(proj_right),
            "margin_left_px": float(margin_left),
            "margin_right_px": float(margin_right),
        }
    )

    if proj_width < float(min_projected_width_px) or proj_height < float(min_projected_height_px):
        return out
    if margin_left < float(min_horizontal_margin_px) or margin_right < float(min_horizontal_margin_px):
        return out

    out["visible"] = True
    return out


def _runtime_visible_present(
    runtime_present: float,
    runtime_longitudinal: float,
    runtime_lateral: float,
    runtime_dist: float,
    camera_half_fov_deg: float = 32.0,
    longitudinal_min_m: float = 0.15,
    dist_max_m: float = 8.5,
    image_w: int = 128,
    image_h: int = 128,
) -> bool:
    return bool(
        _runtime_visible_projection(
            runtime_present=runtime_present,
            runtime_longitudinal=runtime_longitudinal,
            runtime_lateral=runtime_lateral,
            runtime_dist=runtime_dist,
            camera_half_fov_deg=camera_half_fov_deg,
            longitudinal_min_m=longitudinal_min_m,
            dist_max_m=dist_max_m,
            image_w=image_w,
            image_h=image_h,
        )["visible"]
    )


def _dump_error_frames(
    dump_dir: str,
    phase: str,
    scene_key: str,
    obstacle_context_source: str,
    frame_records: List[Dict[str, Any]],
    top_k: int,
) -> None:
    if not dump_dir or not frame_records or top_k <= 0:
        return

    os.makedirs(dump_dir, exist_ok=True)
    scene_dir = os.path.join(dump_dir, f"{phase}_{scene_key}_{obstacle_context_source}")
    os.makedirs(scene_dir, exist_ok=True)

    ranked = sorted(frame_records, key=lambda x: float(x.get("score", -1.0)), reverse=True)[:top_k]
    summary = []
    for rank, rec in enumerate(ranked, start=1):
        stem = f"{rank:02d}_{rec['kind']}_score_{rec['score']:.2f}"
        preview_path = os.path.join(scene_dir, f"{stem}.png")
        meta_path = os.path.join(scene_dir, f"{stem}.json")
        preview = _render_semantic_preview(rec["image_chw"], rec["info"])
        cv2.imwrite(preview_path, preview)
        meta = {
            "phase": phase,
            "scene_key": scene_key,
            "obstacle_context_source": obstacle_context_source,
            "score": float(rec["score"]),
            "kind": rec["kind"],
            "runtime_scene_present": float(rec["runtime_scene_present"]),
            "runtime_visible_present": float(rec["runtime_visible_present"]),
            "cv_present": float(rec["cv_present"]),
            "runtime_longitudinal": float(rec["runtime_longitudinal"]),
            "runtime_lateral": float(rec["runtime_lateral"]),
            "runtime_dist": float(rec["runtime_dist"]),
            "runtime_risk": float(rec["runtime_risk"]),
            "cv_longitudinal": float(rec["cv_longitudinal"]),
            "cv_lateral": float(rec["cv_lateral"]),
            "cv_dist": float(rec["cv_dist"]),
            "cv_risk": float(rec["cv_risk"]),
            "cv_confidence": float(rec.get("cv_confidence", 0.0)),
            "runtime_visible_diag": rec.get("runtime_visible_diag", {}),
            "cv_bbox": rec["info"].get("obstacle_cv_bbox", rec["info"].get("obstacle_learned_bbox")),
            "termination_reason": rec["info"].get("termination_reason", ""),
            "scene_logging_key": rec["info"].get("logging_key", ""),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        summary.append({"preview": preview_path, "meta": meta_path, "score": meta["score"], "kind": meta["kind"]})

    with open(os.path.join(scene_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def _train_defaults() -> Dict[str, Any]:
    sig = inspect.signature(p.train_v16)
    return {
        name: param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect._empty
    }


def _fallback(value: Any, default: Any) -> Any:
    return default if value is None else value


def _build_conf(port: int) -> Dict[str, Any]:
    cfg = p.load_config(myconfig=p.DEFAULT_MYCONFIG)
    if cfg is not None and hasattr(cfg, "GYM_CONF"):
        conf = cfg.GYM_CONF.copy()
        conf.update(
            {
                "host": "127.0.0.1",
                "port": int(port),
                "car_name": "waveshare_v16_eval",
                "racer_name": "V16-Obstacle-Eval",
                "country": "CN",
                "bio": "V16 obstacle context eval",
                "guid": "waveshare-v16-eval",
                "max_cte": 8.0,
            }
        )
        return conf
    return {
        "host": "127.0.0.1",
        "port": int(port),
        "car_name": "waveshare_v16_eval",
        "racer_name": "V16-Obstacle-Eval",
        "country": "CN",
        "bio": "V16 obstacle context eval",
        "guid": "waveshare-v16-eval",
        "max_cte": 8.0,
    }


def _curriculum_values(defaults: Dict[str, Any], obstacle_context_source: str) -> Dict[str, Any]:
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
        "obstacle_context_source",
        "obstacle_context_checkpoint",
        "obstacle_context_device",
        "obstacle_context_seq_len",
        "obstacle_context_present_threshold",
        "obstacle_context_present_off_threshold",
        "obstacle_context_activation_consecutive",
        "obstacle_context_deactivation_consecutive",
        "collision_penalty_base",
        "offtrack_penalty_base",
        "w_near_collision",
        "near_collision_start_ratio",
        "overtake_success_bonus",
        "safe_follow_bonus_scale",
        "post_pass_stability_bonus",
        "post_pass_stability_steps",
        "overtake_respawn_enabled",
        "overtake_respawn_cooldown_steps",
        "overtake_respawn_max_per_episode",
        "overtake_respawn_progress_min",
        "overtake_respawn_progress_max",
    ]
    out = {k: defaults.get(k) for k in keys}
    out["obstacle_context_source"] = str(obstacle_context_source)
    return out


def _make_env(
    scene_env_id: str,
    phase: str,
    obstacle_context_source: str,
    obstacle_context_checkpoint: str,
    obstacle_context_device: str,
    obstacle_context_seq_len: int,
    obstacle_context_present_threshold: float,
    obstacle_context_present_off_threshold: float,
    obstacle_context_activation_consecutive: int,
    obstacle_context_deactivation_consecutive: int,
    port: int,
    seed: int,
):
    defaults = _train_defaults()
    curriculum_values = _curriculum_values(defaults, obstacle_context_source)
    _, _applied = p._apply_curriculum_phase(phase, curriculum_values)
    env_ids = [scene_env_id]
    track_dir = p._resolve_track_dir(track_dir=str(defaults.get("track_dir", "")), env_ids=env_ids)
    scene_specs = p.SCENE_SPECS
    track_geometry = p.TrackGeometryManager(track_dir=track_dir, env_ids=env_ids, scene_specs=scene_specs)
    conf = _build_conf(port)

    obs_size = int(defaults.get("obs_size", 128))
    snapshot_dir = os.path.join("/tmp", "v16_eval_snapshots", f"{phase}_{scene_env_id}_{obstacle_context_source}")
    os.makedirs(snapshot_dir, exist_ok=True)

    env = p.MultiSceneEnvV16(
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
        obstacle_context_source=str(obstacle_context_source),
        obstacle_context_checkpoint=str(obstacle_context_checkpoint or ""),
        obstacle_context_device=str(obstacle_context_device or "cpu"),
        obstacle_context_seq_len=int(_fallback(obstacle_context_seq_len, defaults.get("obstacle_context_seq_len", 16))),
        obstacle_context_present_threshold=float(
            _fallback(
                obstacle_context_present_threshold,
                defaults.get("obstacle_context_present_threshold", 0.5),
            )
        ),
        obstacle_context_present_off_threshold=(
            None if obstacle_context_present_off_threshold is None else float(obstacle_context_present_off_threshold)
        ),
        obstacle_context_activation_consecutive=int(
            _fallback(
                obstacle_context_activation_consecutive,
                defaults.get("obstacle_context_activation_consecutive", 3),
            )
        ),
        obstacle_context_deactivation_consecutive=int(
            _fallback(
                obstacle_context_deactivation_consecutive,
                defaults.get("obstacle_context_deactivation_consecutive", 2),
            )
        ),
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
    return env


def _evaluate(
    model_path: str,
    scene_env_id: str,
    phase: str,
    obstacle_context_source: str,
    obstacle_context_checkpoint: str,
    obstacle_context_device: str,
    obstacle_context_seq_len: int,
    obstacle_context_present_threshold: float,
    obstacle_context_present_off_threshold: float,
    obstacle_context_activation_consecutive: int,
    obstacle_context_deactivation_consecutive: int,
    port: int,
    episodes: int,
    seed: int,
    dump_error_dir: str = "",
    dump_top_k: int = 0,
):
    def make_env():
        return _make_env(
            scene_env_id=scene_env_id,
            phase=phase,
            obstacle_context_source=obstacle_context_source,
            obstacle_context_checkpoint=obstacle_context_checkpoint,
            obstacle_context_device=obstacle_context_device,
            obstacle_context_seq_len=obstacle_context_seq_len,
            obstacle_context_present_threshold=obstacle_context_present_threshold,
            obstacle_context_present_off_threshold=obstacle_context_present_off_threshold,
            obstacle_context_activation_consecutive=obstacle_context_activation_consecutive,
            obstacle_context_deactivation_consecutive=obstacle_context_deactivation_consecutive,
            port=port,
            seed=seed,
        )

    vec_env = DummyVecEnv([make_env])
    p._safe_seed_env(vec_env, seed, label=f"eval_{phase}_{scene_env_id}_{obstacle_context_source}")
    model = RecurrentPPO.load(model_path, env=vec_env)

    obs = vec_env.reset()
    lstm_state = None
    episode_start = np.ones((1,), dtype=bool)
    running_reward = 0.0
    rows: List[Dict[str, Any]] = []
    error_samples: Dict[str, List[float]] = {
        "longitudinal_abs": [],
        "lateral_abs": [],
        "dist_abs": [],
        "risk_abs": [],
    }
    present_stats = {
        "frames_total": 0,
        "runtime_scene_present_frames": 0,
        "runtime_visible_present_frames": 0,
        "cv_present_frames": 0,
        "visible_present_agree_frames": 0,
        "true_positive_visible_frames": 0,
        "false_positive_visible_frames": 0,
        "false_negative_visible_frames": 0,
        "matched_visible_frames": 0,
        "runtime_out_of_view_frames": 0,
    }
    frame_records: List[Dict[str, Any]] = []

    while len(rows) < int(episodes):
        action, lstm_state = model.predict(
            obs,
            state=lstm_state,
            episode_start=episode_start,
            deterministic=True,
        )
        obs, rewards, dones, infos = vec_env.step(action)
        reward = float(np.array(rewards, dtype=np.float32).reshape(-1)[0])
        done = bool(np.array(dones, dtype=bool).reshape(-1)[0])
        info = infos[0]
        running_reward += reward

        runtime_present = float(info.get("obstacle_present", 0.0) or 0.0)
        runtime_longitudinal = float(info.get("obstacle_longitudinal", 0.0) or 0.0)
        runtime_lateral = float(info.get("obstacle_lateral", 0.0) or 0.0)
        runtime_dist = float(info.get("obstacle_dist", 0.0) or 0.0)
        runtime_risk = float(info.get("obstacle_risk", 0.0) or 0.0)
        pred_prefix = "obstacle_cv" if obstacle_context_source != "learned_v1" else "obstacle_learned"
        cv_present = float(info.get(f"{pred_prefix}_present", 0.0) or 0.0)
        cv_longitudinal = float(info.get(f"{pred_prefix}_longitudinal", 0.0) or 0.0)
        cv_lateral = float(info.get(f"{pred_prefix}_lateral", 0.0) or 0.0)
        cv_dist = float(info.get(f"{pred_prefix}_dist", 0.0) or 0.0)
        cv_risk = float(info.get(f"{pred_prefix}_risk", 0.0) or 0.0)
        cv_confidence = float(info.get(f"{pred_prefix}_confidence", 0.0) or 0.0)

        runtime_scene_has = runtime_present >= 0.5
        obs_image = None
        if isinstance(obs, dict) and "image" in obs:
            try:
                obs_image = np.asarray(obs["image"][0], dtype=np.float32)
            except Exception:
                obs_image = None
        image_h = int(obs_image.shape[1]) if obs_image is not None and obs_image.ndim == 3 else 128
        image_w = int(obs_image.shape[2]) if obs_image is not None and obs_image.ndim == 3 else 128

        runtime_visible_diag = _runtime_visible_projection(
            runtime_present=runtime_present,
            runtime_longitudinal=runtime_longitudinal,
            runtime_lateral=runtime_lateral,
            runtime_dist=runtime_dist,
            image_w=image_w,
            image_h=image_h,
        )
        runtime_visible_has = bool(runtime_visible_diag["visible"])
        cv_has = cv_present >= 0.5
        present_stats["frames_total"] += 1
        present_stats["runtime_scene_present_frames"] += int(runtime_scene_has)
        present_stats["runtime_visible_present_frames"] += int(runtime_visible_has)
        present_stats["cv_present_frames"] += int(cv_has)
        present_stats["visible_present_agree_frames"] += int(runtime_visible_has == cv_has)
        present_stats["true_positive_visible_frames"] += int(runtime_visible_has and cv_has)
        present_stats["false_positive_visible_frames"] += int((not runtime_visible_has) and cv_has)
        present_stats["false_negative_visible_frames"] += int(runtime_visible_has and (not cv_has))
        present_stats["runtime_out_of_view_frames"] += int(runtime_scene_has and (not runtime_visible_has))

        if runtime_visible_has and cv_has:
            present_stats["matched_visible_frames"] += 1
            error_samples["longitudinal_abs"].append(abs(cv_longitudinal - runtime_longitudinal))
            error_samples["lateral_abs"].append(abs(cv_lateral - runtime_lateral))
            error_samples["dist_abs"].append(abs(cv_dist - runtime_dist))
            error_samples["risk_abs"].append(abs(cv_risk - runtime_risk))

        if dump_error_dir and obstacle_context_source in ("cv_v1", "learned_v1"):
            image_chw = obs_image.copy() if obs_image is not None else None
            if image_chw is not None:
                score = _frame_error_score(
                    runtime_scene_has,
                    runtime_visible_has,
                    cv_has,
                    runtime_longitudinal,
                    runtime_lateral,
                    runtime_dist,
                    runtime_risk,
                    cv_longitudinal,
                    cv_lateral,
                    cv_dist,
                    cv_risk,
                )
                if score > 0.0:
                    kind = "matched"
                    if runtime_scene_has and (not runtime_visible_has):
                        kind = "runtime_present_but_not_visible"
                    elif runtime_visible_has and not cv_has:
                        kind = "false_negative_visible"
                    elif (not runtime_visible_has) and cv_has:
                        kind = "false_positive"
                    frame_records.append(
                        {
                            "score": float(score),
                            "kind": kind,
                            "image_chw": image_chw,
                            "info": dict(info),
                            "runtime_scene_present": runtime_present,
                            "runtime_visible_present": float(runtime_visible_has),
                            "cv_present": cv_present,
                            "runtime_longitudinal": runtime_longitudinal,
                            "runtime_lateral": runtime_lateral,
                            "runtime_dist": runtime_dist,
                            "runtime_risk": runtime_risk,
                            "runtime_visible_diag": dict(runtime_visible_diag),
                            "cv_longitudinal": cv_longitudinal,
                            "cv_lateral": cv_lateral,
                            "cv_dist": cv_dist,
                            "cv_risk": cv_risk,
                            "cv_confidence": cv_confidence,
                        }
                    )

        if done:
            ep = info.get("episode", {}) or {}
            row = {
                "scene_key": str(info.get("scene_key", "")),
                "logging_key": str(info.get("logging_key", "")),
                "termination_reason": str(info.get("termination_reason", "")),
                "reward_sum": float(running_reward),
                "ep_r_total": float(info.get("ep_r_total", ep.get("r", running_reward)) or 0.0),
                "ep_len": float(ep.get("l", 0.0) or 0.0),
                "ep_soft_lap_count": float(info.get("ep_soft_lap_count", 0.0) or 0.0),
                "ep_overtake_count": float(info.get("ep_overtake_count", 0.0) or 0.0),
                "ep_r_overtake": float(info.get("ep_r_overtake", 0.0) or 0.0),
                "ep_r_follow": float(info.get("ep_r_follow", 0.0) or 0.0),
                "ep_r_post_pass": float(info.get("ep_r_post_pass", 0.0) or 0.0),
                "ep_term_collision": float(info.get("ep_term_collision", 0.0) or 0.0),
                "ep_term_offtrack": float(info.get("ep_term_offtrack", 0.0) or 0.0),
                "ep_term_env_done": float(info.get("ep_term_env_done", 0.0) or 0.0),
                "ep_term_stuck": float(info.get("ep_term_stuck", 0.0) or 0.0),
                "ep_obstacle_has_lane_pid": float(info.get("ep_obstacle_has_lane_pid", 0.0) or 0.0),
                "ep_obstacle_lane_pid_count": float(info.get("ep_obstacle_lane_pid_count", 0.0) or 0.0),
                "ep_lane_pid_debug_steps": float(info.get("ep_lane_pid_debug_steps", 0.0) or 0.0),
                "ep_native_env_done_likely_cte": float(info.get("ep_native_env_done_likely_cte", 0.0) or 0.0),
                "ep_native_env_done_likely_hit": float(info.get("ep_native_env_done_likely_hit", 0.0) or 0.0),
                "obstacle_context_source": str(info.get("obstacle_context_source", obstacle_context_source)),
                "obstacle_present_runtime": runtime_present,
                "obstacle_present_visible_runtime": float(runtime_visible_has),
                "obstacle_longitudinal_runtime": runtime_longitudinal,
                "obstacle_lateral_runtime": runtime_lateral,
                "obstacle_dist_runtime": runtime_dist,
                "obstacle_risk_runtime": runtime_risk,
                "obstacle_cv_present": cv_present,
                "obstacle_cv_longitudinal": cv_longitudinal,
                "obstacle_cv_lateral": cv_lateral,
                "obstacle_cv_dist": cv_dist,
                "obstacle_cv_risk": cv_risk,
                "obstacle_cv_confidence": cv_confidence,
            }
            rows.append(row)
            running_reward = 0.0
        episode_start = np.array([done], dtype=bool)

    vec_env.close()

    def _series_stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {
                "count": 0.0,
                "mean": float("nan"),
                "p50": float("nan"),
                "p90": float("nan"),
                "p99": float("nan"),
                "max": float("nan"),
            }
        arr = np.array(values, dtype=np.float32)
        return {
            "count": float(arr.size),
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
        }

    frame_total = max(1, int(present_stats["frames_total"]))
    error_summary = {
        "frames_total": int(present_stats["frames_total"]),
        "runtime_scene_present_rate": float(present_stats["runtime_scene_present_frames"] / frame_total),
        "runtime_visible_present_rate": float(present_stats["runtime_visible_present_frames"] / frame_total),
        "cv_present_rate": float(present_stats["cv_present_frames"] / frame_total),
        "visible_present_agreement_rate": float(present_stats["visible_present_agree_frames"] / frame_total),
        "true_positive_visible_rate": float(present_stats["true_positive_visible_frames"] / frame_total),
        "false_positive_visible_rate": float(present_stats["false_positive_visible_frames"] / frame_total),
        "false_negative_visible_rate": float(present_stats["false_negative_visible_frames"] / frame_total),
        "runtime_out_of_view_rate": float(present_stats["runtime_out_of_view_frames"] / frame_total),
        "matched_visible_frames": int(present_stats["matched_visible_frames"]),
        "longitudinal_abs": _series_stats(error_samples["longitudinal_abs"]),
        "lateral_abs": _series_stats(error_samples["lateral_abs"]),
        "dist_abs": _series_stats(error_samples["dist_abs"]),
        "risk_abs": _series_stats(error_samples["risk_abs"]),
    }

    summary: Dict[str, Any] = {
        "phase": phase,
        "scene_env_id": scene_env_id,
        "obstacle_context_source": obstacle_context_source,
        "episodes": len(rows),
        "reward_mean": float(np.mean([r["ep_r_total"] for r in rows])),
        "reward_std": float(np.std([r["ep_r_total"] for r in rows])),
        "soft_lap_mean": float(np.mean([r["ep_soft_lap_count"] for r in rows])),
        "overtake_count_mean": float(np.mean([r["ep_overtake_count"] for r in rows])),
        "overtake_reward_mean": float(np.mean([r["ep_r_overtake"] for r in rows])),
        "follow_reward_mean": float(np.mean([r["ep_r_follow"] for r in rows])),
        "post_pass_reward_mean": float(np.mean([r["ep_r_post_pass"] for r in rows])),
        "collision_rate": float(np.mean([r["ep_term_collision"] for r in rows])),
        "offtrack_rate": float(np.mean([r["ep_term_offtrack"] for r in rows])),
        "env_done_rate": float(np.mean([r["ep_term_env_done"] for r in rows])),
        "stuck_rate": float(np.mean([r["ep_term_stuck"] for r in rows])),
        "success_rate_softlap_ge_2": float(np.mean([1.0 if r["ep_soft_lap_count"] >= 2.0 else 0.0 for r in rows])),
        "termination_reason_counts": dict(Counter(r["termination_reason"] for r in rows)),
        "obstacle_error_summary": error_summary,
        "raw_rows": rows,
    }
    if dump_error_dir and obstacle_context_source in ("cv_v1", "learned_v1"):
        scene_key = rows[0]["scene_key"] if rows else scene_env_id
        _dump_error_frames(
            dump_dir=dump_error_dir,
            phase=phase,
            scene_key=str(scene_key),
            obstacle_context_source=obstacle_context_source,
            frame_records=frame_records,
            top_k=int(dump_top_k),
        )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--phase", default="lane_pid_pass")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--port", type=int, default=9091)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="")
    ap.add_argument("--dump-error-dir", default="")
    ap.add_argument("--dump-top-k", type=int, default=12)
    ap.add_argument("--sources", default="runtime,cv_v1")
    ap.add_argument("--obstacle-context-checkpoint", default="")
    ap.add_argument("--obstacle-context-device", default="cpu")
    ap.add_argument("--obstacle-context-seq-len", type=int, default=16)
    ap.add_argument("--obstacle-context-present-threshold", type=float, default=0.5)
    ap.add_argument("--obstacle-context-present-off-threshold", type=float, default=None)
    ap.add_argument("--obstacle-context-activation-consecutive", type=int, default=3)
    ap.add_argument("--obstacle-context-deactivation-consecutive", type=int, default=2)
    args = ap.parse_args()

    scene_map = {
        "ws": "donkey-waveshare-v0",
        "gt": "donkey-generated-track-v0",
    }
    all_results: Dict[str, Any] = {"checkpoint": os.path.abspath(args.checkpoint), "phase": args.phase, "episodes": int(args.episodes), "results": {}}
    sources = [s.strip() for s in str(args.sources).split(",") if s.strip()]

    for scene_key, env_id in scene_map.items():
        all_results["results"][scene_key] = {}
        for source in sources:
            print(f"[eval] phase={args.phase} scene={scene_key} source={source} episodes={args.episodes}")
            result = _evaluate(
                model_path=args.checkpoint,
                scene_env_id=env_id,
                phase=args.phase,
                obstacle_context_source=source,
                obstacle_context_checkpoint=str(args.obstacle_context_checkpoint),
                obstacle_context_device=str(args.obstacle_context_device),
                obstacle_context_seq_len=int(args.obstacle_context_seq_len),
                obstacle_context_present_threshold=float(args.obstacle_context_present_threshold),
                obstacle_context_present_off_threshold=args.obstacle_context_present_off_threshold,
                obstacle_context_activation_consecutive=int(args.obstacle_context_activation_consecutive),
                obstacle_context_deactivation_consecutive=int(args.obstacle_context_deactivation_consecutive),
                port=args.port,
                episodes=int(args.episodes),
                seed=int(args.seed),
                dump_error_dir=str(args.dump_error_dir),
                dump_top_k=int(args.dump_top_k),
            )
            all_results["results"][scene_key][source] = result
            compact = {k: v for k, v in result.items() if k != "raw_rows"}
            print(json.dumps(compact, ensure_ascii=False, indent=2))

    if args.output:
        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
