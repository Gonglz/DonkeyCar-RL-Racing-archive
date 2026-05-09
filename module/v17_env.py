"""
module/v17_env.py

V17 LiDAR-first environment wiring.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import gym
import numpy as np
from stable_baselines3.common.monitor import Monitor

from .action_adapter import ActionAdapterWrapper
from .control import ActionSafetyWrapper
from .lidar import (
    DEFAULT_CANONICAL_LIDAR_SPEC,
    CanonicalLidarSpec,
    SimAsyncLidarBuffer,
    TargetTokenBuffer,
    flatten_canonical_lidar,
)
from .multi_scene_env import (
    MONITOR_INFO_KEYS,
    MultiSceneEnv,
    MultiSceneEnvV13,
    MultiSceneEnvV16,
    _clear_handler_over,
    _install_custom_episode_over,
    _set_handler_max_cte,
)
from .obv import _build_state_v17
from .reward import DonkeyRewardWrapper


def _domain_id_from_name(domain: str) -> float:
    return 0.0 if str(domain).strip().lower() == "ws" else 1.0


def _ensure_monitor_info_defaults(info: Dict[str, Any]) -> None:
    string_defaults = {
        "domain": str(info.get("domain", "unknown") or "unknown"),
        "scene_key": str(info.get("scene_key", "unknown") or "unknown"),
        "logging_key": str(info.get("logging_key", info.get("scene_key", "unknown")) or "unknown"),
        "termination_reason": str(info.get("termination_reason", "running") or "running"),
    }
    for key, value in string_defaults.items():
        info.setdefault(key, value)
    for key in MONITOR_INFO_KEYS:
        if key in string_defaults:
            continue
        info.setdefault(key, 0.0)


class V17ObsWrapper(gym.Wrapper):
    """
    Dict observation for V17:
      - image:     selected semantic image channels; default 6ch includes rawpink vehicle_prob
      - state:     7D core dynamics/control state
      - lidar:     72D canonical full LiDAR, or 12D target token when
                   lidar_obs_mode="target_token"
      - lidar_meta:2D = [is_new_scan, steps_since_new_scan_norm]
      - domain_id: 1D critic-only routing signal
    """

    def __init__(
        self,
        env,
        scene_key: str,
        logging_key: str,
        domain: str,
        obs_size: int,
        speed_vmax: float,
        control_wrapper: ActionAdapterWrapper,
        action_safety_wrapper: ActionSafetyWrapper,
        sim2real_wrapper=None,
        image_channel_indices: Sequence[int] = (0, 1, 2, 3, 4, 5),
        lidar_spec: CanonicalLidarSpec = DEFAULT_CANONICAL_LIDAR_SPEC,
        lidar_repeat_min_steps: int = 2,
        lidar_repeat_max_steps: int = 4,
        snapshot_dir: Optional[str] = None,
        snapshot_max_steps: int = 0,
        snapshot_preview_tile: int = 160,
        include_domain_id: bool = True,
        lidar_obs_mode: str = "full",
        target_token_control_dt_s: float = 0.05,
    ):
        super().__init__(env)
        self.scene_key = str(scene_key)
        self.logging_key = str(logging_key)
        self.domain = str(domain)
        self.obs_size = int(obs_size)
        self.speed_vmax = float(speed_vmax)
        self.control_wrapper = control_wrapper
        self.action_safety_wrapper = action_safety_wrapper
        self.sim2real_wrapper = sim2real_wrapper
        self.image_channel_indices = tuple(int(x) for x in image_channel_indices)
        self.snapshot_dir = str(snapshot_dir or "").strip()
        self.snapshot_max_steps = int(max(0, snapshot_max_steps))
        self.snapshot_preview_tile = int(max(64, snapshot_preview_tile))
        self.include_domain_id = bool(include_domain_id)
        self.lidar_obs_mode = str(lidar_obs_mode or "full").strip().lower()
        if self.lidar_obs_mode not in ("full", "target_token"):
            raise ValueError(f"unsupported lidar_obs_mode={self.lidar_obs_mode!r}")

        self._snapshot_step = 0
        self._lidar_buffer = SimAsyncLidarBuffer(
            spec=lidar_spec,
            repeat_min_steps=lidar_repeat_min_steps,
            repeat_max_steps=lidar_repeat_max_steps,
        )
        self._target_buffer = TargetTokenBuffer(
            spec=lidar_spec,
            control_dt_s=float(target_token_control_dt_s),
        )
        self._lidar_spec = lidar_spec
        self._last_info: Dict[str, Any] = {}
        self._action_diag_sum: Dict[str, float] = {}
        self._action_diag_count = 0
        self._reset_lidar_episode_diag()
        self._reset_vehicle_prob_episode_diag()

        if self.lidar_obs_mode == "full":
            lidar_low = np.concatenate([
                np.zeros((lidar_spec.num_sectors,), dtype=np.float32),
                np.zeros((lidar_spec.num_sectors,), dtype=np.float32),
            ])
            lidar_high = np.concatenate([
                np.full((lidar_spec.num_sectors,), lidar_spec.max_range_m, dtype=np.float32),
                np.ones((lidar_spec.num_sectors,), dtype=np.float32),
            ])
        else:
            lidar_low = np.array(
                [
                    0.0,
                    -lidar_spec.max_range_m,
                    -lidar_spec.max_range_m,
                    -3.0,
                    -3.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            )
            lidar_high = np.array(
                [
                    1.0,
                    lidar_spec.max_range_m,
                    lidar_spec.max_range_m,
                    3.0,
                    3.0,
                    6.0,
                    1.0,
                    1.0,
                    lidar_spec.max_range_m,
                    lidar_spec.max_range_m,
                    lidar_spec.max_range_m,
                    lidar_spec.max_range_m,
                ],
                dtype=np.float32,
            )
        space_dict: Dict[str, gym.spaces.Space] = {
            "image": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(self.image_channel_indices), self.obs_size, self.obs_size),
                dtype=np.float32,
            ),
            "state": gym.spaces.Box(
                low=np.full((7,), -3.0, dtype=np.float32),
                high=np.full((7,), 3.0, dtype=np.float32),
                dtype=np.float32,
            ),
            "lidar": gym.spaces.Box(
                low=lidar_low,
                high=lidar_high,
                dtype=np.float32,
            ),
            "lidar_meta": gym.spaces.Box(
                low=np.zeros((2,), dtype=np.float32),
                high=np.ones((2,), dtype=np.float32),
                dtype=np.float32,
            ),
        }
        if self.include_domain_id:
            space_dict["domain_id"] = gym.spaces.Box(
                low=np.zeros((1,), dtype=np.float32),
                high=np.ones((1,), dtype=np.float32),
                dtype=np.float32,
            )
        self.observation_space = gym.spaces.Dict(space_dict)

    @staticmethod
    def _safe_action_value(arr: np.ndarray, idx: int, default: float = 0.0) -> float:
        try:
            flat = np.asarray(arr, dtype=np.float32).reshape(-1)
            if idx < flat.size:
                return float(flat[idx])
        except Exception:
            pass
        return float(default)

    def _reset_action_diag(self) -> None:
        self._action_diag_sum = {}
        self._action_diag_min = {}
        self._action_diag_max = {}
        self._action_diag_count = 0

    def _accum_action_diag(self, info: Dict[str, Any]) -> None:
        keys = (
            "speed",
            "action/policy_delta_abs",
            "action/policy_speed_scale_abs",
            "action/policy_line_bias_abs",
            "ctrl/target_steer_abs",
            "ctrl/v_target",
            "ctrl/v_meas",
            "ctrl/throttle_pi",
            "sim2real/raw_steer_abs",
            "sim2real/raw_throttle",
            "sim2real/steer_abs",
            "sim2real/throttle",
            "safety/steer_raw_abs",
            "safety/steer_exec_abs",
            "safety/delta_steer_abs",
            "safety/rate_limit_hit",
            "safety/rate_excess_bounded",
            "safety/delta_delta_steer_abs",
            "safety/delta_delta_limit_hit",
            "safety/delta_delta_excess_bounded",
            "safety/servo_deadband_hold",
            "safety/steer_clip_hit",
            "safety/mismatch_abs",
        )
        self._action_diag_count += 1
        for key in keys:
            try:
                value = float(info.get(key, 0.0) or 0.0)
            except Exception:
                value = 0.0
            self._action_diag_sum[key] = self._action_diag_sum.get(key, 0.0) + value
            self._action_diag_min[key] = min(value, self._action_diag_min.get(key, value))
            self._action_diag_max[key] = max(value, self._action_diag_max.get(key, value))

    def _finalize_action_diag(self, info: Dict[str, Any]) -> None:
        denom = float(max(1, self._action_diag_count))
        mapping = {
            "ep_speed_mean": "speed",
            "ep_policy_delta_abs_mean": "action/policy_delta_abs",
            "ep_policy_speed_scale_abs_mean": "action/policy_speed_scale_abs",
            "ep_policy_line_bias_abs_mean": "action/policy_line_bias_abs",
            "ep_adapter_target_steer_abs_mean": "ctrl/target_steer_abs",
            "ep_adapter_v_target_mean": "ctrl/v_target",
            "ep_adapter_v_meas_mean": "ctrl/v_meas",
            "ep_adapter_throttle_mean": "ctrl/throttle_pi",
            "ep_sim2real_raw_steer_abs_mean": "sim2real/raw_steer_abs",
            "ep_sim2real_raw_throttle_mean": "sim2real/raw_throttle",
            "ep_sim2real_steer_abs_mean": "sim2real/steer_abs",
            "ep_sim2real_throttle_mean": "sim2real/throttle",
            "ep_safety_steer_raw_abs_mean": "safety/steer_raw_abs",
            "ep_safety_steer_exec_abs_mean": "safety/steer_exec_abs",
            "ep_safety_delta_steer_abs_mean": "safety/delta_steer_abs",
            "ep_safety_rate_limit_hit_rate": "safety/rate_limit_hit",
            "ep_safety_rate_excess_bounded_mean": "safety/rate_excess_bounded",
            "ep_safety_delta_delta_steer_abs_mean": "safety/delta_delta_steer_abs",
            "ep_safety_delta_delta_limit_hit_rate": "safety/delta_delta_limit_hit",
            "ep_safety_delta_delta_excess_bounded_mean": "safety/delta_delta_excess_bounded",
            "ep_safety_servo_deadband_hold_rate": "safety/servo_deadband_hold",
            "ep_safety_steer_clip_hit_rate": "safety/steer_clip_hit",
            "ep_safety_mismatch_abs_mean": "safety/mismatch_abs",
        }
        for out_key, src_key in mapping.items():
            info[out_key] = float(self._action_diag_sum.get(src_key, 0.0) / denom)
        max_mapping = {
            "ep_speed_max": "speed",
            "ep_adapter_v_target_max": "ctrl/v_target",
            "ep_adapter_v_meas_max": "ctrl/v_meas",
            "ep_safety_delta_steer_abs_max": "safety/delta_steer_abs",
            "ep_safety_rate_excess_bounded_max": "safety/rate_excess_bounded",
            "ep_safety_delta_delta_steer_abs_max": "safety/delta_delta_steer_abs",
            "ep_safety_delta_delta_excess_bounded_max": "safety/delta_delta_excess_bounded",
            "ep_safety_mismatch_abs_max": "safety/mismatch_abs",
        }
        min_mapping = {
            "ep_speed_min": "speed",
            "ep_adapter_v_target_min": "ctrl/v_target",
            "ep_adapter_v_meas_min": "ctrl/v_meas",
        }
        for out_key, src_key in max_mapping.items():
            info[out_key] = float(self._action_diag_max.get(src_key, 0.0))
        for out_key, src_key in min_mapping.items():
            info[out_key] = float(self._action_diag_min.get(src_key, 0.0))

    def _select_image_channels(self, img_obs: np.ndarray) -> np.ndarray:
        img = np.asarray(img_obs, dtype=np.float32)
        if img.ndim != 3:
            raise ValueError(f"expected CHW image, got shape={img.shape}")
        max_idx = max(self.image_channel_indices)
        if img.shape[0] <= max_idx:
            raise ValueError(f"image has {img.shape[0]} channels, cannot select {self.image_channel_indices}")
        return img[list(self.image_channel_indices), :, :].astype(np.float32)

    def _reset_lidar_episode_diag(self) -> None:
        self._lidar_episode_diag = {
            "steps": 0,
            "valid_frac_sum": 0.0,
            "front_valid_frac_sum": 0.0,
            "new_scan_sum": 0.0,
            "stale_norm_sum": 0.0,
            "min_m_sum": 0.0,
            "p10_m_sum": 0.0,
            "p50_m_sum": 0.0,
            "front_min_m_sum": 0.0,
            "min_m_min": float("inf"),
            "front_min_m_min": float("inf"),
            "near_0p6_steps": 0,
            "near_1p0_steps": 0,
            "all_max_steps": 0,
        }

    def _reset_vehicle_prob_episode_diag(self) -> None:
        self._vehicle_prob_episode_diag = {
            "steps": 0,
            "mean_sum": 0.0,
            "max_sum": 0.0,
            "max_peak": 0.0,
            "hot_frac_sum": 0.0,
            "detected_steps": 0,
            "lidar_cam_gate_steps": 0,
            "lidar_cam_hit_steps": 0,
            "lidar_cam_miss_steps": 0,
            "lidar_cam_valid_frac_sum": 0.0,
            "lidar_cam_min_m_sum": 0.0,
            "lidar_cam_min_m_min": float("inf"),
        }

    def _compute_lidar_camera_visibility_gate(
        self,
        info: Dict[str, Any],
        lidar_range: Optional[np.ndarray],
        lidar_valid: Optional[np.ndarray],
    ) -> Dict[str, float]:
        max_range = float(getattr(self._lidar_spec, "max_range_m", 20.0))
        cam_half_fov_deg = 60.0
        cam_max_m = 3.0

        truth_known = any(
            key in info
            for key in (
                "obstacle_present",
                "obstacle_present_runtime",
                "obstacle_longitudinal",
                "obstacle_longitudinal_runtime",
            )
        )
        obstacle_present = float(
            max(
                float(info.get("obstacle_present", 0.0) or 0.0),
                float(info.get("obstacle_present_runtime", 0.0) or 0.0),
            )
        )
        try:
            obstacle_longitudinal = float(
                info.get("obstacle_longitudinal", info.get("obstacle_longitudinal_runtime", np.nan))
            )
            obstacle_lateral = float(info.get("obstacle_lateral", info.get("obstacle_lateral_runtime", np.nan)))
        except Exception:
            obstacle_longitudinal = float("nan")
            obstacle_lateral = float("nan")
        if "obstacle_planar_distance" in info:
            try:
                obstacle_planar = float(info.get("obstacle_planar_distance", np.nan))
            except Exception:
                obstacle_planar = float("nan")
        elif np.isfinite(obstacle_longitudinal) and np.isfinite(obstacle_lateral):
            obstacle_planar = float(np.hypot(obstacle_longitudinal, obstacle_lateral))
        else:
            obstacle_planar = float("nan")

        if obstacle_longitudinal > 1e-4 and np.isfinite(obstacle_lateral):
            obstacle_angle_deg = abs(float(np.degrees(np.arctan2(obstacle_lateral, obstacle_longitudinal))))
        else:
            obstacle_angle_deg = 180.0
        obstacle_truth_gate = bool(
            obstacle_present > 0.5
            and np.isfinite(obstacle_longitudinal)
            and np.isfinite(obstacle_lateral)
            and np.isfinite(obstacle_planar)
            and obstacle_longitudinal > 0.0
            and obstacle_planar <= cam_max_m
            and obstacle_angle_deg <= cam_half_fov_deg
        )

        if lidar_range is None or lidar_valid is None:
            return {
                "gate": 0.0,
                "min_m": max_range,
                "valid_frac": 0.0,
                "truth_gate": float(obstacle_truth_gate),
                "truth_angle_deg": float(obstacle_angle_deg),
                "truth_planar_m": obstacle_planar if np.isfinite(obstacle_planar) else max_range,
                "truth_known": float(truth_known),
            }

        ranges = np.asarray(lidar_range, dtype=np.float32).reshape(-1)
        valid = np.asarray(lidar_valid, dtype=np.float32).reshape(-1) > 0.5
        n = int(ranges.size)
        if n <= 0 or valid.size != n:
            return {
                "gate": 0.0,
                "min_m": max_range,
                "valid_frac": 0.0,
                "truth_gate": float(obstacle_truth_gate),
                "truth_angle_deg": float(obstacle_angle_deg),
                "truth_planar_m": obstacle_planar if np.isfinite(obstacle_planar) else max_range,
                "truth_known": float(truth_known),
            }

        spec_fov = float(getattr(self._lidar_spec, "fov_deg", 180.0))
        near_clip = float(getattr(self._lidar_spec, "near_clip_m", 0.18))
        edges = np.linspace(0.5 * spec_fov, -0.5 * spec_fov, n + 1, dtype=np.float32)
        centers = 0.5 * (edges[:-1] + edges[1:])

        # Conservative camera-visible proxy: central +/-60 deg and close enough
        # that a fixed-pink obstacle should occupy visible pixels at 128x128.
        cam_mask = np.abs(centers) <= cam_half_fov_deg
        finite = np.isfinite(ranges)
        visible = cam_mask & valid & finite & (ranges >= near_clip) & (ranges <= cam_max_m)
        visible_ranges = ranges[visible]
        if visible_ranges.size:
            min_m = float(np.min(visible_ranges))
            gate = 1.0
        else:
            min_m = max_range
            gate = 0.0
        denom = max(1, int(np.count_nonzero(cam_mask)))
        if truth_known:
            gate = float(gate > 0.5 and obstacle_truth_gate)
        return {
            "gate": float(gate),
            "min_m": float(min_m),
            "valid_frac": float(np.count_nonzero(visible) / denom),
            "truth_gate": float(obstacle_truth_gate),
            "truth_angle_deg": float(obstacle_angle_deg),
            "truth_planar_m": obstacle_planar if np.isfinite(obstacle_planar) else max_range,
            "truth_known": float(truth_known),
        }

    def _update_vehicle_prob_episode_diag(
        self,
        info: Dict[str, Any],
        img_obs: np.ndarray,
        lidar_range: Optional[np.ndarray] = None,
        lidar_valid: Optional[np.ndarray] = None,
    ) -> None:
        img = np.asarray(img_obs, dtype=np.float32)
        if img.ndim != 3 or img.shape[0] <= 4:
            veh = np.zeros((1,), dtype=np.float32)
        else:
            veh = np.clip(img[4].astype(np.float32), 0.0, 1.0).reshape(-1)

        mean_v = float(np.mean(veh)) if veh.size else 0.0
        max_v = float(np.max(veh)) if veh.size else 0.0
        hot_frac = float(np.mean(veh >= 0.10)) if veh.size else 0.0
        # Require both intensity and a small connected-looking area proxy so a
        # single pink pixel does not count as an obstacle observation.
        detected = float(max_v >= 0.15 and hot_frac >= 0.0005)
        in_obs = float(4 in self.image_channel_indices)
        lidar_cam = self._compute_lidar_camera_visibility_gate(info, lidar_range, lidar_valid)
        lidar_cam_gate = float(lidar_cam["gate"])
        lidar_cam_hit = float(lidar_cam_gate > 0.5 and detected > 0.5)
        lidar_cam_miss = float(lidar_cam_gate > 0.5 and detected <= 0.5)

        info["vehicle_prob_mean"] = mean_v
        info["vehicle_prob_max"] = max_v
        info["vehicle_prob_hot_frac"] = hot_frac
        info["vehicle_prob_detected"] = detected
        info["vehicle_prob_channel_in_obs"] = in_obs
        info["vehicle_prob_lidar_cam_gate"] = lidar_cam_gate
        info["vehicle_prob_lidar_cam_hit"] = lidar_cam_hit
        info["vehicle_prob_lidar_cam_miss"] = lidar_cam_miss
        info["vehicle_prob_lidar_cam_min_m"] = float(lidar_cam["min_m"])
        info["vehicle_prob_lidar_cam_valid_frac"] = float(lidar_cam["valid_frac"])
        info["vehicle_prob_obstacle_cam_truth_gate"] = float(lidar_cam["truth_gate"])
        info["vehicle_prob_obstacle_cam_truth_angle_deg"] = float(lidar_cam["truth_angle_deg"])
        info["vehicle_prob_obstacle_cam_truth_planar_m"] = float(lidar_cam["truth_planar_m"])
        info["vehicle_prob_obstacle_cam_truth_known"] = float(lidar_cam["truth_known"])

        diag = self._vehicle_prob_episode_diag
        diag["steps"] += 1
        diag["mean_sum"] += mean_v
        diag["max_sum"] += max_v
        diag["max_peak"] = max(float(diag["max_peak"]), max_v)
        diag["hot_frac_sum"] += hot_frac
        diag["detected_steps"] += int(detected > 0.5)
        diag["lidar_cam_gate_steps"] += int(lidar_cam_gate > 0.5)
        diag["lidar_cam_hit_steps"] += int(lidar_cam_hit > 0.5)
        diag["lidar_cam_miss_steps"] += int(lidar_cam_miss > 0.5)
        diag["lidar_cam_valid_frac_sum"] += float(lidar_cam["valid_frac"])
        if lidar_cam_gate > 0.5:
            diag["lidar_cam_min_m_sum"] += float(lidar_cam["min_m"])
            diag["lidar_cam_min_m_min"] = min(float(diag["lidar_cam_min_m_min"]), float(lidar_cam["min_m"]))

    def _finalize_vehicle_prob_episode_diag(self, info: Dict[str, Any]) -> None:
        diag = self._vehicle_prob_episode_diag
        steps = max(1, int(diag.get("steps", 0)))
        gate_steps = int(diag.get("lidar_cam_gate_steps", 0))
        max_range = float(getattr(self._lidar_spec, "max_range_m", 20.0))
        lidar_cam_min_m_min = float(diag.get("lidar_cam_min_m_min", float("inf")))
        info["ep_vehicle_prob_mean"] = float(diag["mean_sum"] / steps)
        info["ep_vehicle_prob_max_mean"] = float(diag["max_sum"] / steps)
        info["ep_vehicle_prob_max_peak"] = float(diag["max_peak"])
        info["ep_vehicle_prob_hot_frac_mean"] = float(diag["hot_frac_sum"] / steps)
        info["ep_vehicle_prob_detected_rate"] = float(diag["detected_steps"] / steps)
        info["ep_vehicle_prob_channel_in_obs"] = float(4 in self.image_channel_indices)
        info["ep_vehicle_prob_lidar_cam_gate_rate"] = float(gate_steps / steps)
        info["ep_vehicle_prob_lidar_cam_hit_rate"] = float(diag["lidar_cam_hit_steps"] / steps)
        info["ep_vehicle_prob_lidar_cam_miss_rate"] = float(diag["lidar_cam_miss_steps"] / steps)
        info["ep_vehicle_prob_lidar_cam_recall"] = (
            float(diag["lidar_cam_hit_steps"] / gate_steps) if gate_steps > 0 else 0.0
        )
        info["ep_vehicle_prob_lidar_cam_valid_frac_mean"] = float(diag["lidar_cam_valid_frac_sum"] / steps)
        info["ep_vehicle_prob_lidar_cam_min_m_mean"] = (
            float(diag["lidar_cam_min_m_sum"] / gate_steps) if gate_steps > 0 else max_range
        )
        info["ep_vehicle_prob_lidar_cam_min_m_min"] = (
            lidar_cam_min_m_min if np.isfinite(lidar_cam_min_m_min) else max_range
        )

    def _compute_lidar_step_diag(
        self,
        lidar_range: np.ndarray,
        lidar_valid: np.ndarray,
        lidar_diag: Dict[str, Any],
    ) -> Dict[str, float]:
        ranges = np.asarray(lidar_range, dtype=np.float32).reshape(-1)
        valid = np.asarray(lidar_valid, dtype=np.float32).reshape(-1) > 0.5
        max_range = float(getattr(self._lidar_spec, "max_range_m", 20.0))

        valid_ranges = ranges[valid & np.isfinite(ranges)]
        if valid_ranges.size:
            min_m = float(np.min(valid_ranges))
            p10_m = float(np.percentile(valid_ranges, 10))
            p50_m = float(np.percentile(valid_ranges, 50))
        else:
            min_m = max_range
            p10_m = max_range
            p50_m = max_range

        n = int(ranges.size)
        center = n // 2
        half = max(1, min(n // 4, 3))
        front_slice = slice(max(0, center - half), min(n, center + half))
        front_ranges = ranges[front_slice]
        front_valid = valid[front_slice] & np.isfinite(front_ranges)
        front_valid_ranges = front_ranges[front_valid]
        front_min_m = float(np.min(front_valid_ranges)) if front_valid_ranges.size else max_range
        front_denom = max(1, int(front_ranges.size))

        return {
            "valid_frac": float(np.mean(valid)) if n > 0 else 0.0,
            "front_valid_frac": float(np.count_nonzero(front_valid) / front_denom),
            "new_scan": float(lidar_diag.get("is_new_scan", 0.0) or 0.0),
            "stale_norm": float(lidar_diag.get("steps_since_new_scan_norm", 0.0) or 0.0),
            "min_m": min_m,
            "p10_m": p10_m,
            "p50_m": p50_m,
            "front_min_m": front_min_m,
            "near_0p6": float(min_m < 0.6),
            "near_1p0": float(min_m < 1.0),
            "all_max": float(valid_ranges.size == 0 or min_m >= max_range - 1e-4),
        }

    def _update_lidar_episode_diag(
        self,
        info: Dict[str, Any],
        lidar_range: np.ndarray,
        lidar_valid: np.ndarray,
        lidar_diag: Dict[str, Any],
    ) -> None:
        step = self._compute_lidar_step_diag(lidar_range, lidar_valid, lidar_diag)
        info["lidar_valid_frac"] = float(step["valid_frac"])
        info["lidar_front_valid_frac"] = float(step["front_valid_frac"])
        info["lidar_min_m"] = float(step["min_m"])
        info["lidar_p10_m"] = float(step["p10_m"])
        info["lidar_p50_m"] = float(step["p50_m"])
        info["lidar_front_min_m"] = float(step["front_min_m"])
        info["lidar_near_0p6"] = float(step["near_0p6"])
        info["lidar_near_1p0"] = float(step["near_1p0"])
        info["lidar_all_max"] = float(step["all_max"])

        diag = self._lidar_episode_diag
        diag["steps"] += 1
        diag["valid_frac_sum"] += float(step["valid_frac"])
        diag["front_valid_frac_sum"] += float(step["front_valid_frac"])
        diag["new_scan_sum"] += float(step["new_scan"])
        diag["stale_norm_sum"] += float(step["stale_norm"])
        diag["min_m_sum"] += float(step["min_m"])
        diag["p10_m_sum"] += float(step["p10_m"])
        diag["p50_m_sum"] += float(step["p50_m"])
        diag["front_min_m_sum"] += float(step["front_min_m"])
        diag["min_m_min"] = min(float(diag["min_m_min"]), float(step["min_m"]))
        diag["front_min_m_min"] = min(float(diag["front_min_m_min"]), float(step["front_min_m"]))
        diag["near_0p6_steps"] += int(step["near_0p6"] > 0.5)
        diag["near_1p0_steps"] += int(step["near_1p0"] > 0.5)
        diag["all_max_steps"] += int(step["all_max"] > 0.5)

    def _finalize_lidar_episode_diag(self, info: Dict[str, Any]) -> None:
        diag = self._lidar_episode_diag
        steps = max(1, int(diag.get("steps", 0)))
        info["ep_lidar_valid_frac_mean"] = float(diag["valid_frac_sum"] / steps)
        info["ep_lidar_front_valid_frac_mean"] = float(diag["front_valid_frac_sum"] / steps)
        info["ep_lidar_new_scan_rate"] = float(diag["new_scan_sum"] / steps)
        info["ep_lidar_stale_norm_mean"] = float(diag["stale_norm_sum"] / steps)
        info["ep_lidar_min_m_mean"] = float(diag["min_m_sum"] / steps)
        info["ep_lidar_p10_m_mean"] = float(diag["p10_m_sum"] / steps)
        info["ep_lidar_p50_m_mean"] = float(diag["p50_m_sum"] / steps)
        info["ep_lidar_front_min_m_mean"] = float(diag["front_min_m_sum"] / steps)
        min_m_min = float(diag["min_m_min"])
        front_min_m_min = float(diag["front_min_m_min"])
        max_range = float(getattr(self._lidar_spec, "max_range_m", 20.0))
        info["ep_lidar_min_m_min"] = min_m_min if np.isfinite(min_m_min) else max_range
        info["ep_lidar_front_min_m_min"] = (
            front_min_m_min if np.isfinite(front_min_m_min) else max_range
        )
        info["ep_lidar_near_0p6_rate"] = float(diag["near_0p6_steps"] / steps)
        info["ep_lidar_near_1p0_rate"] = float(diag["near_1p0_steps"] / steps)
        info["ep_lidar_all_max_rate"] = float(diag["all_max_steps"] / steps)

    def _build_obs(
        self,
        img_obs: np.ndarray,
        info: Dict[str, Any],
        record_lidar_diag: bool = True,
    ) -> Dict[str, np.ndarray]:
        image = self._select_image_channels(img_obs)
        state = _build_state_v17(
            info=info,
            action_safety_wrapper=self.action_safety_wrapper,
            control_wrapper=self.control_wrapper,
            v_max=self.speed_vmax,
        )
        lidar_range, lidar_valid, lidar_diag = self._lidar_buffer.observe(
            info.get("lidar"),
            raw_lidar_packet=info.get("lidar_raw_packet"),
        )
        canonical_flat = flatten_canonical_lidar(lidar_range, lidar_valid)
        if self.lidar_obs_mode == "full":
            lidar = canonical_flat
        else:
            lidar, target_diag = self._target_buffer.observe(
                lidar_range=lidar_range,
                lidar_valid=lidar_valid,
                is_new_scan=float(lidar_diag["is_new_scan"]),
                steps_since_new_scan=float(lidar_diag["steps_since_new_scan"]),
            )
            info["target_exist"] = float(target_diag["target_exist"])
            info["target_confidence"] = float(target_diag["target_confidence"])
            info["target_age_norm"] = float(target_diag["target_age_norm"])
            info["target_rel_long"] = float(lidar[1])
            info["target_rel_lat"] = float(lidar[2])
            info["target_rel_v_long"] = float(lidar[3])
            info["target_rel_v_lat"] = float(lidar[4])
            info["target_ttc"] = float(lidar[5])
            info["target_width_proxy"] = float(lidar[8])
            info["target_front_min_range"] = float(lidar[9])
            info["target_left_gap"] = float(lidar[10])
            info["target_right_gap"] = float(lidar[11])
        lidar_meta = np.array(
            [
                float(lidar_diag["is_new_scan"]),
                float(lidar_diag["steps_since_new_scan_norm"]),
            ],
            dtype=np.float32,
        )
        info["canonical_lidar_range"] = lidar_range.copy()
        info["canonical_lidar_valid"] = lidar_valid.copy()
        info["canonical_lidar_flat"] = canonical_flat.copy()
        info["lidar_is_new_scan"] = float(lidar_diag["is_new_scan"])
        info["lidar_steps_since_new_scan_norm"] = float(lidar_diag["steps_since_new_scan_norm"])
        info["lidar_steps_since_new_scan"] = float(lidar_diag["steps_since_new_scan"])
        info["lidar_scan_age_norm"] = float(lidar_diag["scan_age_norm"])
        info["lidar_repeat_count_norm"] = float(lidar_diag["repeat_count_norm"])
        info["lidar_repeat_count"] = float(lidar_diag["repeat_count"])
        if record_lidar_diag:
            self._update_lidar_episode_diag(info, lidar_range, lidar_valid, lidar_diag)
            self._update_vehicle_prob_episode_diag(info, img_obs, lidar_range, lidar_valid)
        obs = {
            "image": image,
            "state": state,
            "lidar": lidar,
            "lidar_meta": lidar_meta,
        }
        if self.include_domain_id:
            obs["domain_id"] = np.array([_domain_id_from_name(self.domain)], dtype=np.float32)
        return obs

    def _build_snapshot_preview(self, obs_dict: Dict[str, np.ndarray], meta: Dict[str, Any]) -> np.ndarray:
        image = np.asarray(obs_dict["image"], dtype=np.float32)
        channels = int(image.shape[0])
        cols = min(3, max(1, channels))
        rows = max(1, int(np.ceil(channels / cols)))
        tile = self.snapshot_preview_tile
        header_h = 118
        canvas = np.zeros((rows * tile + header_h, cols * tile, 3), dtype=np.uint8)
        labels = ["raw_y", "edge_line", "guide_line", "sobel", "motion"]
        for idx in range(channels):
            ch = np.clip(image[idx], 0.0, 1.0)
            tile_img = cv2.resize((ch * 255.0).astype(np.uint8), (tile, tile), interpolation=cv2.INTER_NEAREST)
            tile_bgr = cv2.cvtColor(tile_img, cv2.COLOR_GRAY2BGR)
            label = labels[idx] if idx < len(labels) else f"ch{idx}"
            cv2.putText(tile_bgr, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 220, 255), 1, cv2.LINE_AA)
            r = idx // cols
            c = idx % cols
            canvas[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = tile_bgr

        lidar_range = np.asarray(meta.get("lidar_range", np.zeros((36,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        header = canvas[rows * tile:, :]
        lines = [
            f"scene={self.scene_key} domain={self.domain} reward={meta.get('reward', 0.0):.3f} done={int(bool(meta.get('done', False)))}",
            f"speed={meta.get('speed', 0.0):.3f} cte={meta.get('cte', 0.0):.3f} new_scan={int(meta.get('lidar_is_new_scan', 0.0))}",
            f"stale_norm={meta.get('lidar_steps_since_new_scan_norm', 0.0):.3f} obstacle_dist={meta.get('obstacle_dist', 0.0):.3f}",
            f"lidar_min={float(np.min(lidar_range)) if lidar_range.size else 0.0:.3f} lidar_p50={float(np.median(lidar_range)) if lidar_range.size else 0.0:.3f}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(header, line, (10, 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
        return canvas

    def _maybe_save_snapshot(self, obs_dict: Dict[str, np.ndarray], info: Dict[str, Any], reward: float, done: bool) -> None:
        if not self.snapshot_dir or self.snapshot_max_steps <= 0 or self._snapshot_step >= self.snapshot_max_steps:
            return
        scene_dir = os.path.join(self.snapshot_dir, self.scene_key)
        os.makedirs(scene_dir, exist_ok=True)
        idx = self._snapshot_step
        npz_path = os.path.join(scene_dir, f"step_{idx:02d}.npz")
        json_path = os.path.join(scene_dir, f"step_{idx:02d}.json")
        png_path = os.path.join(scene_dir, f"step_{idx:02d}_preview.png")
        np.savez_compressed(npz_path, **obs_dict)
        meta = {
            "scene_key": self.scene_key,
            "logging_key": self.logging_key,
            "domain": self.domain,
            "reward": float(reward),
            "done": bool(done),
            "speed": float(info.get("speed", 0.0) or 0.0),
            "cte": float(info.get("cte", 0.0) or 0.0),
            "lidar_is_new_scan": float(info.get("lidar_is_new_scan", 0.0) or 0.0),
            "lidar_steps_since_new_scan_norm": float(info.get("lidar_steps_since_new_scan_norm", 0.0) or 0.0),
            "lidar_valid_frac": float(info.get("lidar_valid_frac", 0.0) or 0.0),
            "lidar_front_min_m": float(info.get("lidar_front_min_m", 0.0) or 0.0),
            "obstacle_dist": float(info.get("obstacle_dist", 0.0) or 0.0),
            "lidar_range": np.asarray(info.get("canonical_lidar_range", np.zeros((36,), dtype=np.float32)), dtype=np.float32).tolist(),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        cv2.imwrite(png_path, self._build_snapshot_preview(obs_dict, meta))
        self._snapshot_step += 1

    def reset(self, **kwargs):
        img = self.env.reset(**kwargs)
        self._lidar_buffer.reset()
        self._target_buffer.reset()
        self._reset_action_diag()
        self._reset_lidar_episode_diag()
        self._reset_vehicle_prob_episode_diag()
        self._snapshot_step = 0
        self._last_info = {
            "speed": 0.0,
            "gyro": (0.0, 0.0, 0.0),
            "accel": (0.0, 0.0, 0.0),
            "car": (0.0, 0.0, 0.0),
            "pos": (0.0, 0.0, 0.0),
            "cte": 0.0,
            "lidar": np.full((180,), -1.0, dtype=np.float32),
        }
        return self._build_obs(img, self._last_info, record_lidar_diag=False)

    def step(self, action):
        policy_action = np.asarray(action, dtype=np.float32).reshape(-1)
        img, reward, done, info = self.env.step(action)

        if self.control_wrapper is not None:
            self.control_wrapper.consume_info(info)

        obs = self._build_obs(img, info)

        policy_delta = self._safe_action_value(policy_action, 0)
        policy_speed = self._safe_action_value(policy_action, 1)
        policy_bias = self._safe_action_value(policy_action, 2)
        info["action/policy_delta_steer"] = float(policy_delta)
        info["action/policy_speed_scale"] = float(policy_speed)
        info["action/policy_line_bias"] = float(policy_bias)
        info["action/policy_delta_abs"] = abs(float(policy_delta))
        info["action/policy_speed_scale_abs"] = abs(float(policy_speed))
        info["action/policy_line_bias_abs"] = abs(float(policy_bias))

        if self.action_safety_wrapper is not None:
            self.action_safety_wrapper.consume_info(dict(info))
            sdiag = getattr(self.action_safety_wrapper, "diag", {}) or {}
            info["safety/steer_raw"] = float(sdiag.get("steer_raw", 0.0))
            info["safety/steer_exec"] = float(sdiag.get("steer_exec", 0.0))
            info["safety/steer_raw_abs"] = abs(float(sdiag.get("steer_raw", 0.0)))
            info["safety/steer_exec_abs"] = abs(float(sdiag.get("steer_exec", 0.0)))
            info["safety/delta_steer"] = float(sdiag.get("delta_steer", 0.0))
            info["safety/delta_steer_abs"] = abs(float(sdiag.get("delta_steer", 0.0)))
            info["safety/delta_steer_prev"] = float(sdiag.get("delta_steer_prev", 0.0))
            info["safety/rate_limit_hit"] = float(bool(sdiag.get("rate_limit_hit", False)))
            info["safety/rate_excess_raw"] = float(sdiag.get("rate_excess_raw", 0.0))
            info["safety/rate_excess_bounded"] = float(sdiag.get("rate_excess_bounded", 0.0))
            info["safety/delta_delta_steer"] = float(sdiag.get("delta_delta_steer", 0.0))
            info["safety/delta_delta_steer_abs"] = abs(float(sdiag.get("delta_delta_steer", 0.0)))
            info["safety/delta_delta_limit_hit"] = float(bool(sdiag.get("delta_delta_limit_hit", False)))
            info["safety/delta_delta_excess_bounded"] = float(sdiag.get("delta_delta_excess_bounded", 0.0))
            info["safety/servo_deadband_hold"] = float(bool(sdiag.get("servo_deadband_hold", False)))
            info["safety/steer_clip_hit"] = float(bool(sdiag.get("steer_clip_hit", False)))
            info["safety/mismatch"] = float(sdiag.get("mismatch", 0.0))
            info["safety/mismatch_abs"] = abs(float(sdiag.get("mismatch", 0.0)))
            info["safety/effective_delta_max"] = float(sdiag.get("effective_delta_max", 0.0))

        if self.sim2real_wrapper is not None:
            raw = getattr(self.sim2real_wrapper, "last_raw_action", np.zeros((2,), dtype=np.float32))
            transformed = getattr(
                self.sim2real_wrapper,
                "last_transformed_action",
                np.zeros((2,), dtype=np.float32),
            )
            raw_steer = self._safe_action_value(raw, 0)
            raw_throttle = self._safe_action_value(raw, 1)
            out_steer = self._safe_action_value(transformed, 0)
            out_throttle = self._safe_action_value(transformed, 1)
            info["sim2real/raw_steer"] = float(raw_steer)
            info["sim2real/raw_throttle"] = float(raw_throttle)
            info["sim2real/raw_steer_abs"] = abs(float(raw_steer))
            info["sim2real/steer"] = float(out_steer)
            info["sim2real/throttle"] = float(out_throttle)
            info["sim2real/steer_abs"] = abs(float(out_steer))
            info["sim2real/throttle_gain"] = float(getattr(self.sim2real_wrapper, "throttle_gain", 1.0))
            info["sim2real/steer_gain"] = float(getattr(self.sim2real_wrapper, "steer_gain", 1.0))

        if self.control_wrapper is not None:
            d = self.control_wrapper.diag
            info["ctrl/v_target"] = float(d.get("v_target", 0.0))
            info["ctrl/v_meas"] = float(d.get("v_meas", 0.0))
            info["ctrl/v_err"] = float(d.get("v_err", 0.0))
            info["ctrl/throttle_pi"] = float(d.get("throttle_pi", 0.0))
            info["ctrl/target_steer"] = float(d.get("target_steer", 0.0))
            info["ctrl/target_steer_abs"] = abs(float(d.get("target_steer", 0.0)))
            info["ctrl/steer_core"] = float(d.get("steer_core", 0.0))
            info["ctrl/bias_smooth"] = float(d.get("bias_smooth", 0.0))
            info["ctrl/bias_offset"] = float(d.get("bias_offset", 0.0))
            info["ctrl/v_base"] = float(d.get("v_base", 0.0))
            info["ctrl/delta_steer_input"] = float(d.get("delta_steer_input", 0.0))
            info["ctrl/speed_scale_input"] = float(d.get("speed_scale_input", 0.0))
            info["ctrl/line_bias_input"] = float(d.get("line_bias_input", 0.0))
            info["ctrl/safety_filter_triggered"] = float(d.get("safety_filter_triggered", 0.0))
            info["ctrl/safety_filter_trigger_rate"] = float(d.get("safety_filter_trigger_rate", 0.0))
            info["ctrl/safety_filter_throttle_scale"] = float(d.get("safety_filter_throttle_scale", 1.0))
            predictive_safety_filter = getattr(self.control_wrapper, "predictive_safety_filter", None)
            if predictive_safety_filter is not None and self.action_safety_wrapper is not None:
                try:
                    predictive_safety_filter.sync(
                        steer_prev_limited=float(self.action_safety_wrapper.steer_prev_limited),
                        steer_prev_exec=float(self.action_safety_wrapper.steer_prev_exec),
                    )
                except Exception:
                    pass

        info["scene_key"] = self.scene_key
        info["logging_key"] = self.logging_key
        info["domain"] = self.domain if self.domain else "unknown"
        info["domain_id"] = _domain_id_from_name(self.domain)
        self._accum_action_diag(info)
        if done:
            self._finalize_action_diag(info)
            self._finalize_lidar_episode_diag(info)
            self._finalize_vehicle_prob_episode_diag(info)
        _ensure_monitor_info_defaults(info)

        self._maybe_save_snapshot(obs, info, reward, done)
        self._last_info = dict(info)
        return obs, reward, done, info


class MultiSceneEnvV17(MultiSceneEnvV16):
    def __init__(
        self,
        *args,
        image_channel_indices: Sequence[int] = (0, 1, 2, 3, 4, 5),
        lidar_num_sectors: int = 36,
        lidar_fov_deg: float = 180.0,
        lidar_max_range_m: float = 20.0,
        lidar_near_clip_m: float = 0.18,
        lidar_repeat_min_steps: int = 2,
        lidar_repeat_max_steps: int = 4,
        lidar_obs_mode: str = "full",
        predictive_safety_filter_path: Optional[str] = None,
        predictive_safety_filter_mode: str = "log",
        predictive_safety_filter_log_path: Optional[str] = None,
        predictive_safety_yaw_thresh: Optional[float] = None,
        predictive_safety_decel_thresh: Optional[float] = None,
        sim2real_throttle_gain_override: Optional[float] = None,
        sim2real_throttle_gain_floor: Optional[float] = None,
        sim2real_steer_gain_override: Optional[float] = None,
        sim2real_steer_gain_floor: Optional[float] = None,
        sim2real_filter_dt_s: Optional[float] = None,
        steer_delta_delta_max: Optional[float] = None,
        steer_servo_deadband: float = 0.0,
        w_steer_budget: float = 0.0,
        steer_budget_straight: float = 0.58,
        steer_budget_curve: float = 0.88,
        steer_budget_obstacle_relief: float = 0.16,
        w_sign_flip: float = 0.0,
        sign_flip_min_abs_steer: float = 0.20,
        w_micro_wiggle: float = 0.0,
        micro_wiggle_min_abs_steer: float = 0.035,
        micro_wiggle_max_abs_steer: float = 0.22,
        reward_prepare_pass_bonus: float = 0.0,
        reward_commit_pass_bonus: float = 0.0,
        reward_safe_follow_bonus: float = 0.0,
        reward_post_pass_bonus: float = 0.0,
        reward_post_pass_steps: int = 10,
        curriculum_phase: Optional[str] = None,
        reward_overrides_by_logging_key: Optional[Dict[str, Dict[str, Any]]] = None,
        terminal_offtrack_progress_scale: float = 1.0,
        bad_episode_guard_min_steps: int = 0,
        bad_episode_guard_reward_floor: float = -200.0,
        bad_episode_guard_cte_over_in_rate: float = 0.25,
        bad_episode_guard_min_forward_progress: float = 0.25,
        bad_episode_guard_penalty: float = 4.0,
        port_per_scene: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        self.image_channel_indices = tuple(int(x) for x in image_channel_indices)
        self.lidar_spec = CanonicalLidarSpec(
            num_sectors=int(lidar_num_sectors),
            fov_deg=float(lidar_fov_deg),
            max_range_m=float(lidar_max_range_m),
            near_clip_m=float(lidar_near_clip_m),
            invalid_fill_m=float(lidar_max_range_m),
        )
        self.lidar_repeat_min_steps = int(lidar_repeat_min_steps)
        self.lidar_repeat_max_steps = int(lidar_repeat_max_steps)
        self.lidar_obs_mode = str(lidar_obs_mode or "full").strip().lower()
        self.predictive_safety_filter_path = str(predictive_safety_filter_path or "").strip()
        self.predictive_safety_filter_mode = str(predictive_safety_filter_mode or "log").strip().lower()
        self.predictive_safety_filter_log_path = str(predictive_safety_filter_log_path or "").strip()
        self.predictive_safety_yaw_thresh = predictive_safety_yaw_thresh
        self.predictive_safety_decel_thresh = predictive_safety_decel_thresh
        self.sim2real_throttle_gain_override = sim2real_throttle_gain_override
        self.sim2real_throttle_gain_floor = sim2real_throttle_gain_floor
        self.sim2real_steer_gain_override = sim2real_steer_gain_override
        self.sim2real_steer_gain_floor = sim2real_steer_gain_floor
        self.sim2real_filter_dt_s = sim2real_filter_dt_s
        self.steer_delta_delta_max = steer_delta_delta_max
        self.steer_servo_deadband = float(np.clip(steer_servo_deadband, 0.0, 0.2))
        self.w_steer_budget = float(max(0.0, w_steer_budget))
        self.steer_budget_straight = float(np.clip(steer_budget_straight, 0.05, 1.0))
        self.steer_budget_curve = float(np.clip(steer_budget_curve, self.steer_budget_straight, 1.0))
        self.steer_budget_obstacle_relief = float(max(0.0, steer_budget_obstacle_relief))
        self.w_sign_flip = float(max(0.0, w_sign_flip))
        self.sign_flip_min_abs_steer = float(np.clip(sign_flip_min_abs_steer, 0.0, 1.0))
        self.w_micro_wiggle = float(max(0.0, w_micro_wiggle))
        self.micro_wiggle_min_abs_steer = float(np.clip(micro_wiggle_min_abs_steer, 0.0, 1.0))
        self.micro_wiggle_max_abs_steer = float(np.clip(micro_wiggle_max_abs_steer, 0.0, 1.0))
        if self.micro_wiggle_max_abs_steer < self.micro_wiggle_min_abs_steer:
            self.micro_wiggle_min_abs_steer, self.micro_wiggle_max_abs_steer = (
                self.micro_wiggle_max_abs_steer,
                self.micro_wiggle_min_abs_steer,
            )
        self.reward_prepare_pass_bonus = float(max(0.0, reward_prepare_pass_bonus))
        self.reward_commit_pass_bonus = float(max(0.0, reward_commit_pass_bonus))
        self.reward_safe_follow_bonus = float(max(0.0, reward_safe_follow_bonus))
        self.reward_post_pass_bonus = float(max(0.0, reward_post_pass_bonus))
        self.reward_post_pass_steps = int(max(1, reward_post_pass_steps))
        self.curriculum_phase = str(curriculum_phase or "").strip().lower()
        self.reward_overrides_by_logging_key = dict(reward_overrides_by_logging_key or {})
        self.terminal_offtrack_progress_scale = float(np.clip(terminal_offtrack_progress_scale, 0.0, 1.0))
        self.bad_episode_guard_min_steps = int(max(0, bad_episode_guard_min_steps))
        self.bad_episode_guard_reward_floor = float(bad_episode_guard_reward_floor)
        self.bad_episode_guard_cte_over_in_rate = float(np.clip(bad_episode_guard_cte_over_in_rate, 0.0, 1.0))
        self.bad_episode_guard_min_forward_progress = float(max(0.0, bad_episode_guard_min_forward_progress))
        self.bad_episode_guard_penalty = float(max(0.0, bad_episode_guard_penalty))
        self.port_per_scene: Optional[Tuple[int, ...]] = (
            tuple(int(p) for p in port_per_scene) if port_per_scene else None
        )
        # Persistent per-scene env / obstacle-runtime caches for dual-sim mode.
        # Populated lazily in _create_env when port_per_scene is set.
        self._scene_base_envs: Dict[int, Any] = {}
        self._scene_obstacle_runtimes: Dict[int, Any] = {}
        self._dual_sim_neutralize_count = 0
        self._dual_sim_neutralize_fail_count = 0
        super().__init__(*args, **kwargs)
        if self.port_per_scene is not None:
            if len(self.port_per_scene) != len(self.env_ids):
                raise ValueError(
                    f"port_per_scene length ({len(self.port_per_scene)}) "
                    f"must match env_ids ({len(self.env_ids)})"
                )
            print(
                f"   双 sim 模式: ports={list(self.port_per_scene)} (持久 env，无切换重建)"
            )

    @staticmethod
    def _neutralize_base_env(base_env: Any, *, repeats: int = 2, sleep_s: float = 0.02) -> bool:
        """Send a hard neutral/brake command to a persistent inactive DonkeySim env."""
        if base_env is None:
            return False
        try:
            viewer = getattr(base_env, "viewer", None)
            handler = getattr(viewer, "handler", None)
            if handler is not None and hasattr(handler, "send_control"):
                for _ in range(max(1, int(repeats))):
                    handler.send_control(0.0, 0.0, 1.0)
                    if sleep_s > 0:
                        time.sleep(float(sleep_s))
                return True
        except Exception:
            return False
        try:
            viewer = getattr(base_env, "viewer", None)
            if viewer is not None and hasattr(viewer, "take_action"):
                for _ in range(max(1, int(repeats))):
                    viewer.take_action(np.asarray([0.0, 0.0], dtype=np.float32))
                    if sleep_s > 0:
                        time.sleep(float(sleep_s))
                return True
        except Exception:
            return False
        return False

    def _create_env(self, scene_idx: int):
        import gym_donkeycar  # noqa: F401
        from .obv import CanonicalSemanticWrapper
        from .obstacle_runtime import ObstacleRuntimeManager, ScenarioObstacleWrapper
        from .sim2real_wrapper import Sim2RealActionWrapper

        env_id = self.env_ids[scene_idx]
        scene_specs = MultiSceneEnvV13._SCENE_SPECS
        if env_id not in scene_specs:
            raise KeyError(f"V17 unknown env_id: {env_id}")

        spec = scene_specs[env_id]
        level_name = spec["level_name"]
        scene_key = spec["scene_key"]
        logging_key = spec.get("logging_key", scene_key)
        domain = self.scene_domains[scene_idx]

        if self.port_per_scene is not None:
            previous_scene_idx = getattr(self, "active_scene_idx", None)
            previous_base_env = getattr(self, "_base_env", None)
            should_neutralize_previous = (
                previous_base_env is not None
                and previous_scene_idx is not None
                and int(previous_scene_idx) != int(scene_idx)
            )
            # Dual-sim persistent mode: each scene has a dedicated sim/env pinned
            # to its own port; switching is just a reference swap, no reload, no
            # close, no rebuild.
            if scene_idx not in self._scene_base_envs:
                scene_conf = dict(self.conf)
                scene_conf["port"] = int(self.port_per_scene[scene_idx])
                new_env = MultiSceneEnv._make_env_with_retry(
                    env_id, scene_conf, retries=2, retry_wait_s=1.5
                )
                _install_custom_episode_over(new_env)
                self._scene_base_envs[scene_idx] = new_env
                print(
                    f"✅ [scene_idx={scene_idx} port={scene_conf['port']}] "
                    f"已加载持久场景: {level_name}"
                )
            else:
                print(
                    f"🔁 [scene_idx={scene_idx} port={self.port_per_scene[scene_idx]}] "
                    f"切换到持久场景: {level_name}"
                )
            if should_neutralize_previous:
                ok = self._neutralize_base_env(previous_base_env)
                self._dual_sim_neutralize_count += 1
                if not ok:
                    self._dual_sim_neutralize_fail_count += 1
                if self._dual_sim_neutralize_count <= 20 or self._dual_sim_neutralize_count % 50 == 0:
                    prev_port = "?"
                    try:
                        prev_port = str(self.port_per_scene[int(previous_scene_idx)])
                    except Exception:
                        pass
                    status = "ok" if ok else "failed"
                    print(
                        f"🛑 [dual-sim] neutralize inactive scene_idx={previous_scene_idx} "
                        f"port={prev_port}: {status}"
                    )
            self._base_env = self._scene_base_envs[scene_idx]

            cached_runtime = self._scene_obstacle_runtimes.get(scene_idx)
            if cached_runtime is not None:
                # Bind the active runtime reference; downstream block at the
                # bottom of _create_env will call attach_scene which is
                # idempotent within the same scene_key.
                self._obstacle_runtime = cached_runtime
            else:
                # First visit to this scene_idx. Drop any stale runtime
                # reference (held over from a different scene) so the lazy
                # creator below builds a fresh runtime pinned to this scene
                # and caches it. Without this clear, attach_scene would call
                # close() on the stale runtime and rebuild its obstacle fleet
                # against the new base_env every switch — exactly the
                # "starting DonkeyGym env" / fps tank pattern.
                self._obstacle_runtime = None
        else:
            if self._base_env is None:
                self._base_env = MultiSceneEnv._make_env_with_retry(env_id, self.conf, retries=2, retry_wait_s=1.5)
                _install_custom_episode_over(self._base_env)
                print(f"✅ 模拟器已启动，首个场景: {level_name}")
                if self.scene_start_force_reload:
                    try:
                        MultiSceneEnv._force_reload_scene(
                            self._base_env,
                            level_name,
                            preflight=True,
                            timeout_s=self.scene_reload_timeout_s,
                            post_exit_sleep_s=self.scene_reload_post_exit_sleep_s,
                        )
                    except Exception as e:
                        print(f"⚠️  场景预加载失败，将继续当前状态: {type(e).__name__}: {e}")
                else:
                    print(f"↩️ 跳过训练前强制重载: gym.make 已加载目标场景 {level_name}")
            else:
                if self._obstacle_runtime is not None and getattr(self._obstacle_runtime, "scene_key", "") != scene_key:
                    self._obstacle_runtime.close()
                try:
                    MultiSceneEnv._force_reload_scene(
                        self._base_env,
                        level_name,
                        preflight=False,
                        timeout_s=self.scene_reload_timeout_s,
                        post_exit_sleep_s=self.scene_reload_post_exit_sleep_s,
                    )
                except Exception as e:
                    print(f"⚠️  场景切换失败: {type(e).__name__}: {e}")
                    print("🔁 尝试重启模拟器并恢复目标场景...")
                    if self._obstacle_runtime is not None:
                        self._obstacle_runtime.close()
                    try:
                        self._base_env.close()
                    except Exception:
                        pass
                    self._base_env = None
                    self._base_env = MultiSceneEnv._make_env_with_retry(env_id, self.conf, retries=2, retry_wait_s=1.5)
                    _install_custom_episode_over(self._base_env)
                    if self.scene_start_force_reload:
                        MultiSceneEnv._force_reload_scene(
                            self._base_env,
                            level_name,
                            preflight=True,
                            timeout_s=self.scene_reload_timeout_s,
                            post_exit_sleep_s=self.scene_reload_post_exit_sleep_s,
                        )
                    else:
                        print(f"✅ 模拟器已重启并由 gym.make 加载目标场景: {level_name}")

        _scene_max_cte = spec.get("max_cte", self.conf.get("max_cte", 8.0))
        _set_handler_max_cte(self._base_env, _scene_max_cte, logging_key)

        if self._obstacle_runtime is None:
            runtime_conf = dict(self.conf)
            if self.port_per_scene is not None:
                runtime_conf["port"] = int(self.port_per_scene[scene_idx])
                print(
                    f"🚗 [scene_idx={scene_idx} port={runtime_conf['port']}] "
                    f"obstacle runtime 绑定到持久场景: {logging_key}"
                )
            self._obstacle_runtime = ObstacleRuntimeManager(
                track_geometry=self.track_geometry,
                conf=runtime_conf,
                track_dir=self.track_dir,
                config=self._build_obstacle_runtime_config(),
            )
            if self.port_per_scene is not None:
                # Persist this runtime so the next visit to the same scene
                # reuses its obstacle fleet (no respawn / re-handshake).
                self._scene_obstacle_runtimes[scene_idx] = self._obstacle_runtime
        self._obstacle_runtime.attach_scene(
            base_env=self._base_env,
            env_id=env_id,
            scene_key=scene_key,
            logging_key=logging_key,
        )

        env = ScenarioObstacleWrapper(self._base_env, runtime=self._obstacle_runtime)
        env = CanonicalSemanticWrapper(
            env,
            domain=domain,
            obs_size=self.obs_size,
            augment=self.augment,
            dropout_start_step=self.dropout_start_step,
            dropout_ramp_steps=self.dropout_ramp_steps,
            dropout_max_prob=self.yellow_dropout_prob,
        )

        if self.track_geometry is not None and hasattr(self.track_geometry, "scenes") and scene_key in self.track_geometry.scenes:
            geo = self.track_geometry.scenes[scene_key]
            cte_left = float(geo.cte_left)
            cte_right = float(geo.cte_right)
            cte_left_out = float(geo.cte_left_out)
            cte_right_out = float(geo.cte_right_out)
            coord_scale = float(geo.coord_scale)
            cte_half_width = float(geo.cte_half_width)
        else:
            cte_left = 5.0
            cte_right = -5.0
            cte_left_out = 6.5
            cte_right_out = -6.5
            coord_scale = 8.0
            cte_half_width = 4.6

        reward_kwargs = dict(
            total_timesteps=self.total_timesteps,
            action_safety_wrapper=None,
            w_d=self.w_d,
            w_dd=self.w_dd,
            w_m=self.w_m,
            w_sat=self.w_sat,
            w_steer_budget=self.w_steer_budget,
            steer_budget_straight=self.steer_budget_straight,
            steer_budget_curve=self.steer_budget_curve,
            steer_budget_obstacle_relief=self.steer_budget_obstacle_relief,
            w_sign_flip=self.w_sign_flip,
            sign_flip_min_abs_steer=self.sign_flip_min_abs_steer,
            w_micro_wiggle=self.w_micro_wiggle,
            micro_wiggle_min_abs_steer=self.micro_wiggle_min_abs_steer,
            micro_wiggle_max_abs_steer=self.micro_wiggle_max_abs_steer,
            w_time=self.w_time,
            w_center=self.w_center,
            w_heading=self.w_heading,
            w_speed_ref=self.w_speed_ref,
            speed_ref_vmin=self.speed_ref_vmin,
            speed_ref_vmax=self.speed_ref_vmax,
            speed_ref_kappa_ref=self.speed_ref_kappa_ref,
            lap_reward_scale=self.lap_reward_scale,
            progress_reward_scale=self.progress_reward_scale,
            survival_reward_scale=self.survival_reward_scale,
            collision_penalty_base=self.collision_penalty_base,
            offtrack_penalty_base=self.offtrack_penalty_base,
            w_near_offtrack=self.w_near_offtrack,
            near_offtrack_start_ratio=self.near_offtrack_start_ratio,
            w_near_collision=self.w_near_collision,
            near_collision_start_ratio=self.near_collision_start_ratio,
            overtake_success_bonus=self.overtake_success_bonus,
            safe_follow_bonus_scale=self.reward_safe_follow_bonus,
            prepare_pass_bonus_scale=self.reward_prepare_pass_bonus,
            commit_pass_bonus_scale=self.reward_commit_pass_bonus,
            post_pass_stability_bonus=self.reward_post_pass_bonus,
            post_pass_stability_steps=self.reward_post_pass_steps,
            reward_control_dt_s=self.control_dt,
            cte_left=cte_left,
            cte_right=cte_right,
            cte_left_out=cte_left_out,
            cte_right_out=cte_right_out,
            coord_scale=coord_scale,
            offtrack_leniency_ratio=self.offtrack_leniency_ratio,
            offtrack_leniency_mult=self.offtrack_leniency_mult,
            track_geometry=self.track_geometry,
            scene_key=scene_key,
            logging_key=logging_key,
            cte_half_width=cte_half_width,
            reset_env_done_grace_steps=self.reset_env_done_grace_steps,
            reset_collision_grace_steps=self.reset_collision_grace_steps,
            terminal_offtrack_progress_scale=self.terminal_offtrack_progress_scale,
            bad_episode_guard_min_steps=self.bad_episode_guard_min_steps,
            bad_episode_guard_reward_floor=self.bad_episode_guard_reward_floor,
            bad_episode_guard_cte_over_in_rate=self.bad_episode_guard_cte_over_in_rate,
            bad_episode_guard_min_forward_progress=self.bad_episode_guard_min_forward_progress,
            bad_episode_guard_penalty=self.bad_episode_guard_penalty,
        )
        allowed_reward_overrides = {
            "near_offtrack_start_ratio",
            "w_near_offtrack",
            "w_near_collision",
            "near_collision_start_ratio",
            "overtake_success_bonus",
            "safe_follow_bonus_scale",
            "prepare_pass_bonus_scale",
            "commit_pass_bonus_scale",
            "post_pass_stability_bonus",
            "post_pass_stability_steps",
            "wait_window_bonus_scale",
            "wait_window_min_gap_m",
            "wait_window_max_gap_m",
            "wait_window_max_closing_rate",
            "force_pass_penalty_scale",
            "unsafe_close_penalty_scale",
            "obstacle_clearance_penalty_scale",
            "obstacle_clearance_inner_m",
            "obstacle_clearance_outer_m",
            "post_pass_cut_in_penalty_scale",
            "post_pass_watch_longitudinal_m",
            "post_pass_watch_steps",
            "overtake_success_min_progress_ratio",
            "unsafe_close_gap_m",
            "unsafe_close_clearance_m",
            "unsafe_close_longitudinal_m",
            "unsafe_close_ttc_s",
            "lateral_overlap_ref_m",
            "overtake_arm_longitudinal_min_m",
            "overtake_arm_planar_max_m",
            "overtake_pass_longitudinal_threshold_m",
            "overtake_pass_planar_min_m",
            "close_front_planar_max_m",
            "force_pass_planar_max_m",
            "w_d",
            "w_dd",
            "w_m",
            "w_sat",
            "w_steer_budget",
            "steer_budget_straight",
            "steer_budget_curve",
            "steer_budget_obstacle_relief",
            "w_sign_flip",
            "sign_flip_min_abs_steer",
            "w_micro_wiggle",
            "micro_wiggle_min_abs_steer",
            "micro_wiggle_max_abs_steer",
            "w_center",
            "w_heading",
            "w_speed_ref",
            "speed_ref_vmin",
            "speed_ref_vmax",
            "speed_ref_kappa_ref",
            "safe_follow_min_m",
            "safe_follow_max_m",
            "safe_follow_risk_max",
            "safe_follow_speed_min",
            "safe_follow_ttc_min_s",
            "safe_follow_ttc_max_s",
            "collision_penalty_base",
            "offtrack_penalty_base",
            "offtrack_leniency_ratio",
            "offtrack_leniency_mult",
            "offtrack_grace_steps",
            "offtrack_severe_ratio",
            "offtrack_grace_penalty_scale",
            "offtrack_grace_use_leniency",
            "survival_reward_scale",
            "progress_reward_scale",
            "lap_reward_scale",
            "cte_norm_scale",
            "reward_decay_ref_steps",
            "terminal_offtrack_progress_scale",
            "bad_episode_guard_min_steps",
            "bad_episode_guard_reward_floor",
            "bad_episode_guard_cte_over_in_rate",
            "bad_episode_guard_min_forward_progress",
            "bad_episode_guard_penalty",
            "collision_episode_reward_cap",
            "offtrack_episode_reward_cap",
            "stuck_speed_threshold",
            "stuck_progress_threshold",
            "stuck_grace_steps",
            "stuck_low_speed_penalty_start",
            "stuck_low_speed_penalty_scale",
            "stuck_low_speed_penalty_cap",
            "stuck_penalty_base",
            "stuck_penalty_growth",
            "stuck_penalty_cap",
        }
        reward_overrides = dict(spec.get("reward_overrides", {}) or {})
        phase_scene_overrides = {}
        try:
            phase_scene_overrides = dict(
                self.reward_overrides_by_logging_key.get(logging_key)
                or self.reward_overrides_by_logging_key.get(scene_key)
                or {}
            )
        except Exception:
            phase_scene_overrides = {}
        if phase_scene_overrides:
            reward_overrides.update(phase_scene_overrides)
        if reward_overrides:
            applied = {}
            for key, value in reward_overrides.items():
                if key in allowed_reward_overrides:
                    reward_kwargs[key] = value
                    applied[key] = value
                else:
                    print(f"⚠️  [{logging_key}] reward_overrides: unknown key '{key}', ignored")
            if applied:
                print(f"   [{logging_key}] reward_overrides: {applied}")
        reward_wrapper = DonkeyRewardWrapper(env, **reward_kwargs)
        env = reward_wrapper

        action_safety = ActionSafetyWrapper(
            env,
            delta_max=self.delta_max,
            enable_lpf=self.enable_lpf,
            beta=self.beta,
            delta_delta_max=self.steer_delta_delta_max,
            servo_deadband=self.steer_servo_deadband,
            adaptive_delta_max=self.adaptive_delta_max,
            curve_delta_boost=self.curve_delta_boost,
            curve_kappa_ref=self.curve_kappa_ref,
            steer_intent_boost=self.steer_intent_boost,
            hairpin_curve_ratio=self.hairpin_curve_ratio,
            hairpin_min_delta_max=self.hairpin_min_delta_max,
            hairpin_max_delta_max=self.hairpin_max_delta_max,
        )
        env = action_safety
        reward_wrapper.action_safety_wrapper = action_safety

        sim2real_wrapper = None
        if getattr(self, "sim2real_json", None):
            # Keep sim2real shaping on low-level actions, then let ActionSafety
            # remain the final limiter before the simulator sees the command.
            sim2real_wrapper = Sim2RealActionWrapper.from_json(
                env,
                self.sim2real_json,
                throttle_gain_override=self.sim2real_throttle_gain_override,
                throttle_gain_floor=self.sim2real_throttle_gain_floor,
                steer_gain_override=self.sim2real_steer_gain_override,
                steer_gain_floor=self.sim2real_steer_gain_floor,
                filter_dt_s=self.sim2real_filter_dt_s,
            )
            env = sim2real_wrapper

        predictive_safety_filter = None
        if self.predictive_safety_filter_path:
            from .predictive_safety_filter import PredictiveSafetyFilter

            predictive_safety_filter = PredictiveSafetyFilter(
                model_path=self.predictive_safety_filter_path,
                horizon=3,
                mode=self.predictive_safety_filter_mode,
                log_path=self.predictive_safety_filter_log_path,
                yaw_thresh=self.predictive_safety_yaw_thresh,
                decel_thresh=self.predictive_safety_decel_thresh,
            )

        adapter = ActionAdapterWrapper(
            env,
            k_delta=self.adapter_k_delta,
            lambda_bias=self.adapter_lambda_bias,
            k_bias=self.adapter_k_bias,
            steer_core_decay=self.adapter_steer_core_decay,
            v_nominal=self.adapter_v_nominal,
            k_turn=self.adapter_k_turn,
            k_bias_speed=self.adapter_k_bias_speed,
            alpha_speed=self.adapter_alpha_speed,
            v_min=self.adapter_v_min,
            v_max=self.adapter_v_max,
            speed_kp=self.speed_kp,
            speed_ki=self.speed_ki,
            speed_kff=self.speed_kff,
            control_dt=self.control_dt,
            max_throttle=self.max_throttle,
            allow_reverse=self.allow_reverse,
            predictive_safety_filter=predictive_safety_filter,
        )
        env = adapter

        env = V17ObsWrapper(
            env,
            scene_key=scene_key,
            logging_key=logging_key,
            domain=domain,
            obs_size=self.obs_size,
            speed_vmax=self.speed_vmax,
            control_wrapper=adapter,
            action_safety_wrapper=action_safety,
            sim2real_wrapper=sim2real_wrapper,
            image_channel_indices=self.image_channel_indices,
            lidar_spec=self.lidar_spec,
            lidar_repeat_min_steps=self.lidar_repeat_min_steps,
            lidar_repeat_max_steps=self.lidar_repeat_max_steps,
            snapshot_dir=self.snapshot_dir,
            snapshot_max_steps=self.snapshot_max_steps,
            include_domain_id=True,
            lidar_obs_mode=self.lidar_obs_mode,
            target_token_control_dt_s=self.control_dt,
        )

        env = Monitor(env, info_keywords=MONITOR_INFO_KEYS)

        self.action_safety_wrapper = action_safety
        self.action_adapter_wrapper = adapter
        self.reward_wrapper = reward_wrapper
        self.active_env = env
        self.active_scene_idx = scene_idx
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        if self._base_env is not None:
            _clear_handler_over(self._base_env)
        return obs

    def close(self):
        if self.port_per_scene is not None:
            # Tear down every persistent scene env and obstacle runtime that
            # were cached for the dual-sim path; the parent's close() must not
            # double-free the active aliases, so we drop them first.
            for idx, runtime in list(self._scene_obstacle_runtimes.items()):
                try:
                    runtime.close()
                except Exception:
                    pass
            self._scene_obstacle_runtimes.clear()
            for idx, env in list(self._scene_base_envs.items()):
                try:
                    self._neutralize_base_env(env, repeats=1, sleep_s=0.0)
                except Exception:
                    pass
                try:
                    env.close()
                except Exception:
                    pass
            self._scene_base_envs.clear()
            self._obstacle_runtime = None
            self._base_env = None
            return
        super().close()


__all__ = [
    "MultiSceneEnvV17",
    "V17ObsWrapper",
]
