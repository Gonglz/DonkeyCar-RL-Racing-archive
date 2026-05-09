"""
module/obstacle_runtime.py

V16 obstacle runtime:
- manages obstacle fleet lifecycle across scene switches
- injects obstacle-aware info before reward wrapper
- optionally randomizes learner spawn with the same geometry source
"""

from __future__ import annotations

import math
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gym
import numpy as np

from .obstacle import (
    DonkeyObstacleFleet,
    ObstacleSnapshot,
    PoseState,
    TrackTarget,
    _with_target_yaw,
    compute_relative_state,
    pose_from_info,
    resolve_obstacle_fleet_preset,
    sample_track_target,
    spawn_preset_obstacle_fleet,
    telemetry_to_unity_world,
    yaw_deg_to_unity_quaternion,
)
from .track import SceneGeometry, TrackGeometryManager


def _wrap_progress(progress_ratio: float) -> float:
    return float(float(progress_ratio) % 1.0)


def _copy_info(info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not info:
        return {}
    return dict(info)


def _debug_float(value: Any, digits: int = 3) -> Optional[float]:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return round(x, int(digits))


def _extract_viewer(base_env):
    base = base_env
    while hasattr(base, "env"):
        base = base.env
    return getattr(base, "viewer", None)


@dataclass
class ObstacleRuntimeConfig:
    enabled: bool = True
    active_scene_keys: Tuple[str, ...] = ("waveshare", "generated_track")
    obstacle_count: int = 2
    ws_obstacle_count: Optional[int] = None
    obstacle_free_prob: float = 0.15
    obstacle_modes: Tuple[str, ...] = ("static", "jitter")
    ws_obstacle_free_prob: Optional[float] = None
    min_obstacle_separation_world: float = 3.0
    spawn_ahead_min_m: float = 3.5
    spawn_ahead_max_m: float = 14.0
    min_agent_planar_dist_m: float = 1.5
    min_agent_arc_dist_m: float = 3.5
    lateral_choices: Tuple[float, ...] = (0.35, 0.50, 0.65)
    ws_lateral_choices: Optional[Tuple[float, ...]] = None
    fixed_progress_ratio: Optional[float] = None
    fixed_progress_distribution: Optional[Tuple[Tuple[float, float], ...]] = None
    fixed_progress_gap_ratio: Optional[float] = None
    fixed_progress_gap_ratio_min: Optional[float] = None
    fixed_progress_gap_ratio_max: Optional[float] = None
    obstacle_progress_min: Optional[float] = None
    obstacle_progress_max: Optional[float] = None
    fixed_lateral_ratio: Optional[float] = None
    gt_obstacle_start_exclusion_half_width_m: Optional[float] = None
    ws_obstacle_modes: Optional[Tuple[str, ...]] = None
    ws_obstacle_fixed_progress_ratio: Optional[float] = None
    ws_fixed_progress_gap_ratio: Optional[float] = None
    ws_fixed_progress_gap_ratio_min: Optional[float] = None
    ws_fixed_progress_gap_ratio_max: Optional[float] = None
    ws_obstacle_progress_min: Optional[float] = None  # WS障碍progress最小值
    ws_obstacle_progress_max: Optional[float] = None  # WS障碍progress最大值
    ws_obstacle_fixed_lateral_ratio: Optional[float] = None
    randomize_non_lane_pid_yaw: bool = True
    jitter_amplitude_m: float = 0.10
    jitter_period_s: float = 1.5
    jitter_update_hz: float = 8.0
    nudge_amplitude_m: float = 0.14
    nudge_period_s: float = 1.5
    nudge_update_hz: float = 8.0
    lane_pid_speed_gt: float = 0.85
    lane_pid_speed_ws: float = 0.70
    lane_pid_lookahead_m: float = 0.9
    spawn_gap_s: float = 0.0
    placement_timeout_s: float = 1.5
    ego_random_spawn: bool = False
    ego_spawn_lateral_ratio: float = 0.5
    ego_spawn_settle_steps: int = 3
    ego_spawn_settle_sleep_s: float = 0.05
    seed: Optional[int] = None


class ObstacleRuntimeManager:
    """Owns obstacle fleet lifecycle for the active scene."""

    def __init__(
        self,
        track_geometry: TrackGeometryManager,
        conf: Dict[str, Any],
        track_dir: str,
        config: ObstacleRuntimeConfig,
    ) -> None:
        self.track_geometry = track_geometry
        self.conf = dict(conf or {})
        self.track_dir = str(track_dir)
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        self._base_env = None
        self._env_id: str = ""
        self._scene_key: str = ""
        self._logging_key: str = ""
        self._fleet: Optional[DonkeyObstacleFleet] = None
        self._fleet_scene_key: str = ""
        self._episode_index: int = 0
        self._active_this_episode: bool = False
        self._episode_modes_used: Tuple[str, ...] = tuple()
        self._last_agent_info: Dict[str, Any] = {}
        self._last_runtime_error: str = ""
        self._episode_fixed_progress_ratio: Optional[float] = None
        self._episode_target_plan: Tuple[TrackTarget, ...] = tuple()
        self._debug_step_count_this_episode = 0
        self._debug_watch_logged_steps = set()
        self._debug_last_watch_anomaly_t = 0.0

    @property
    def scene_key(self) -> str:
        return self._scene_key

    def close(self) -> None:
        if self._fleet is not None:
            try:
                self._fleet.shutdown()
            except Exception:
                pass
        self._fleet = None
        self._fleet_scene_key = ""
        self._active_this_episode = False
        self._episode_modes_used = tuple()

    def attach_scene(self, base_env, env_id: str, scene_key: str, logging_key: str) -> None:
        scene_changed = bool(self._scene_key) and str(scene_key) != self._scene_key
        if scene_changed:
            self.close()
        self._base_env = base_env
        self._env_id = str(env_id)
        self._scene_key = str(scene_key)
        self._logging_key = str(logging_key)

    def on_episode_reset(self, initial_obs: np.ndarray) -> np.ndarray:
        self._episode_index += 1
        self._active_this_episode = False
        self._episode_modes_used = tuple()
        self._episode_fixed_progress_ratio = None
        self._episode_target_plan = tuple()
        self._debug_step_count_this_episode = 0
        self._debug_watch_logged_steps = set()
        self._debug_last_watch_anomaly_t = 0.0
        self._last_runtime_error = ""
        should_spawn = False
        spawn_decision = "unsupported"
        free_prob: Optional[float] = None

        obs = np.asarray(initial_obs)
        info = self._observe_info_only() or {}

        if self._scene_supports_runtime():
            try:
                if self.config.ego_random_spawn:
                    obs, info = self._randomize_ego_spawn(obs, info)
                distribution_sample = self._sample_scene_fixed_progress_distribution()
                if distribution_sample is None:
                    free_prob = float(self._scene_obstacle_free_prob())
                    should_spawn = self.rng.random() >= free_prob
                    spawn_decision = "random_free_prob"
                else:
                    should_spawn, self._episode_fixed_progress_ratio = distribution_sample
                    spawn_decision = "fixed_progress_distribution"
                if should_spawn:
                    self._ensure_fleet()
                    if self._fleet is not None:
                        self._active_this_episode = bool(
                            self._refresh_obstacle_layout(agent_info=info)
                        )
                    else:
                        self._deactivate_inactive_fleet(reason="no_spawn")
                else:
                    self._deactivate_inactive_fleet(reason=spawn_decision)
            except Exception as exc:
                self._last_runtime_error = f"{type(exc).__name__}: {exc}"
                print(
                    f"⚠️  obstacle runtime reset failed [{self._logging_key or self._scene_key}]: "
                    f"{self._last_runtime_error}"
                )
                self._active_this_episode = False
                self._deactivate_inactive_fleet(reason="reset_exception")
        else:
            self._deactivate_inactive_fleet(reason="unsupported_scene")

        refreshed = self._observe_info_and_obs()
        if refreshed is not None:
            obs, info = refreshed
        self._last_agent_info = _copy_info(info)
        self._log_reset_debug(
            agent_info=info,
            should_spawn=should_spawn,
            spawn_decision=spawn_decision,
            free_prob=free_prob,
        )
        return np.asarray(obs)

    def enrich_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        info = dict(info)
        self._last_agent_info = _copy_info(info)
        info["obstacle_present"] = 0.0
        info["obstacle_count"] = 0.0
        info["obstacle_runtime_active"] = 1.0 if self._active_this_episode else 0.0
        info["obstacle_episode_modes"] = ",".join(self._episode_modes_used)
        info["obstacle_primary_mode"] = ""
        info["obstacle_lane_pid_debug_active"] = 0.0
        info["obstacle_lane_pid_target_speed"] = 0.0
        info["obstacle_lane_pid_speed"] = 0.0
        info["obstacle_lane_pid_speed_error"] = 0.0
        info["obstacle_lane_pid_effective_lookahead"] = 0.0
        info["obstacle_lane_pid_local_forward"] = 0.0
        info["obstacle_lane_pid_local_left"] = 0.0
        info["obstacle_lane_pid_lookahead_distance"] = 0.0
        info["obstacle_lane_pid_lat_err_norm"] = 0.0
        info["obstacle_lane_pid_steer"] = 0.0
        info["obstacle_lane_pid_throttle"] = 0.0
        info["obstacle_lane_pid_reverse_mode"] = 0.0
        mode_counts = self._episode_mode_counts(self._episode_modes_used)
        info["ep_obstacle_has_lane_pid"] = float(mode_counts["lane_pid"] > 0)
        info["ep_obstacle_primary_is_lane_pid"] = 0.0
        info["ep_obstacle_static_count"] = float(mode_counts["static"])
        info["ep_obstacle_jitter_count"] = float(mode_counts["jitter"])
        info["ep_obstacle_nudge_count"] = float(mode_counts["nudge"])
        info["ep_obstacle_lane_pid_count"] = float(mode_counts["lane_pid"])

        if not self._active_this_episode or self._fleet is None:
            return info

        try:
            self._debug_step_count_this_episode += 1
            snapshots = self._fleet.get_snapshots(agent_info=info)
            self._maybe_log_obstacle_watchdog(agent_info=info, snapshots=snapshots)
            primary_idx = self._select_primary_snapshot_index(snapshots)
            primary = None if primary_idx is None else snapshots[primary_idx]
            if primary is None or primary.relative is None:
                return info

            relative = primary.relative
            risk = self._compute_obstacle_risk(relative.longitudinal, relative.lateral, relative.planar_distance)
            info["obstacle_present"] = 1.0
            info["obstacle_count"] = float(len([snap for snap in snapshots if snap.obstacle is not None]))
            info["obstacle_longitudinal"] = float(relative.longitudinal)
            info["obstacle_lateral"] = float(relative.lateral)
            info["obstacle_source"] = "runtime"
            if primary_idx is not None and primary_idx < len(self._episode_modes_used):
                primary_mode = str(self._episode_modes_used[primary_idx])
                info["obstacle_primary_mode"] = primary_mode
                info["ep_obstacle_primary_is_lane_pid"] = float(primary_mode == "lane_pid")
            lane_pid_idx = next(
                (
                    idx for idx, mode in enumerate(self._episode_modes_used)
                    if str(mode).strip().lower() == "lane_pid"
                ),
                None,
            )
            if (
                lane_pid_idx is not None
                and 0 <= int(lane_pid_idx) < len(self._fleet.cars)
            ):
                lane_pid_debug = self._fleet.cars[int(lane_pid_idx)].get_lane_pid_debug()
                info["obstacle_lane_pid_debug_active"] = float(lane_pid_debug.get("active", 0.0) or 0.0)
                info["obstacle_lane_pid_target_speed"] = float(lane_pid_debug.get("target_speed", 0.0) or 0.0)
                info["obstacle_lane_pid_speed"] = float(lane_pid_debug.get("speed", 0.0) or 0.0)
                info["obstacle_lane_pid_speed_error"] = float(lane_pid_debug.get("speed_error", 0.0) or 0.0)
                info["obstacle_lane_pid_effective_lookahead"] = float(
                    lane_pid_debug.get("effective_lookahead", 0.0) or 0.0
                )
                info["obstacle_lane_pid_local_forward"] = float(
                    lane_pid_debug.get("local_forward", 0.0) or 0.0
                )
                info["obstacle_lane_pid_local_left"] = float(lane_pid_debug.get("local_left", 0.0) or 0.0)
                info["obstacle_lane_pid_lookahead_distance"] = float(
                    lane_pid_debug.get("lookahead_distance", 0.0) or 0.0
                )
                info["obstacle_lane_pid_lat_err_norm"] = float(
                    lane_pid_debug.get("lat_err_norm", 0.0) or 0.0
                )
                info["obstacle_lane_pid_steer"] = float(lane_pid_debug.get("steer", 0.0) or 0.0)
                info["obstacle_lane_pid_throttle"] = float(lane_pid_debug.get("throttle", 0.0) or 0.0)
                info["obstacle_lane_pid_reverse_mode"] = float(
                    lane_pid_debug.get("reverse_mode", 0.0) or 0.0
                )
            reward_relevant = bool(float(relative.longitudinal) >= -0.4 or float(relative.planar_distance) <= 1.0)
            if reward_relevant:
                info["obstacle_dist"] = float(relative.planar_distance)
                info["obstacle_risk"] = float(risk)
        except Exception as exc:
            self._last_runtime_error = f"{type(exc).__name__}: {exc}"
            self._log_runtime_debug("snapshot_error", error=self._last_runtime_error)
        return info

    def _scene_supports_runtime(self) -> bool:
        return bool(
            self.config.enabled
            and self._scene_key
            and self._scene_key in set(self.config.active_scene_keys)
        )

    def _scene_alias(self) -> Optional[str]:
        if self._scene_key == "generated_track":
            return "gt"
        if self._scene_key == "waveshare":
            return "ws"
        return None

    def _handler(self):
        viewer = _extract_viewer(self._base_env)
        return getattr(viewer, "handler", None) if viewer is not None else None

    def _viewer(self):
        return _extract_viewer(self._base_env)

    def _observe_info_only(self) -> Optional[Dict[str, Any]]:
        observed = self._observe_info_and_obs()
        if observed is None:
            return None
        _, info = observed
        return info

    def _observe_info_and_obs(self) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        viewer = self._viewer()
        if viewer is None or not hasattr(viewer, "observe"):
            return None
        try:
            obs, _reward, _done, info = viewer.observe()
            return obs, dict(info or {})
        except Exception:
            return None

    def _ensure_fleet(self) -> None:
        if self._fleet is not None and self._fleet_scene_key == self._scene_key:
            return
        alias = self._scene_alias()
        if alias is None:
            return
        self.close()
        active_count = self._active_obstacle_count_for_scene()
        self._fleet = spawn_preset_obstacle_fleet(
            scene=alias,
            host=str(self.conf.get("host", "127.0.0.1")),
            port=int(self.conf.get("port", 9091)),
            track_dir=self.track_dir,
            count=int(max(1, active_count)),
            min_separation_world=float(max(0.0, self.config.min_obstacle_separation_world)),
            seed=int(self.rng.integers(0, 2**31 - 1)),
            hold_brake=True,
            spawn_gap=float(max(0.0, self.config.spawn_gap_s)),
            placement_timeout_s=float(max(0.1, self.config.placement_timeout_s)),
            initial_place=False,
        )
        self._fleet_scene_key = self._scene_key
        self._log_fleet_debug(event="created")
        # 新 client 连入 DonkeySim 时会先在默认起点短暂出现一帧；
        # 这里送去 staging 并等待回读确认，避免 reset 后短暂残留在赛道上。
        if self._scene_key != "waveshare":
            self._park_fleet(reason="fleet_created")
        else:
            self._log_runtime_debug("fleet_created_hidden", reason="fleet_created")

    def _park_car(self, car, idx: int, reason: str = "park") -> Optional[PoseState]:
        preset = self._fleet.preset
        staging_x = float(preset.staging_x_start - int(idx) * preset.staging_x_step)
        world_y = -500.0 if self._scene_key == "waveshare" else car.default_world_y
        request = {
            "idx": int(idx),
            "reason": str(reason),
            "x": _debug_float(staging_x),
            "z": _debug_float(preset.staging_z),
            "yaw": 0.0,
            "world_y": _debug_float(world_y),
        }
        self._log_runtime_debug(
            "park_car_start",
            **request,
            before=self._car_debug_state(idx),
        )
        car.stop_motion(hold_brake=True)
        pose = car.place_pose(
            x=staging_x,
            z=float(preset.staging_z),
            yaw_deg=0.0,
            world_y=world_y,
            hold_brake=True,
            timeout_s=float(max(0.1, self.config.placement_timeout_s)),
        )
        self._log_runtime_debug(
            "park_car_result",
            **request,
            observed=self._pose_debug_payload(pose),
            after=self._car_debug_state(idx),
        )
        return pose

    def _park_fleet(self, reason: str = "park") -> None:
        self._episode_modes_used = tuple()
        if self._fleet is None:
            return
        for idx, car in enumerate(self._fleet.cars):
            try:
                self._park_car(car, idx, reason=reason)
            except Exception as exc:
                self._log_runtime_debug(
                    "park_car_error",
                    idx=int(idx),
                    reason=str(reason),
                    error=f"{type(exc).__name__}: {exc}",
                    after=self._car_debug_state(idx),
                )

    def _shutdown_fleet(self, reason: str = "inactive") -> None:
        if self._fleet is None:
            return
        cars = list(getattr(self._fleet, "cars", []) or [])
        car_count = len(cars)
        self._log_runtime_debug(
            "fleet_shutdown",
            reason=str(reason),
            scene_key=str(self._scene_key),
            car_count=int(car_count),
        )
        if self._scene_key == "waveshare":
            for idx, car in enumerate(cars):
                try:
                    self._park_car(car, idx, reason=f"{reason}_hide_before_shutdown")
                except Exception as exc:
                    self._log_runtime_debug(
                        "park_car_error",
                        idx=int(idx),
                        reason=f"{reason}_hide_before_shutdown",
                        error=f"{type(exc).__name__}: {exc}",
                        after=self._car_debug_state(idx),
                    )
        try:
            self._fleet.shutdown()
        except Exception as exc:
            self._log_runtime_debug(
                "fleet_shutdown_error",
                reason=str(reason),
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            self._fleet = None
            self._fleet_scene_key = ""
            self._episode_modes_used = tuple()
            self._episode_target_plan = tuple()

    def _deactivate_inactive_fleet(self, reason: str = "inactive") -> None:
        self._active_this_episode = False
        if self._scene_key == "waveshare":
            self._shutdown_fleet(reason=reason)
        else:
            self._park_fleet(reason=reason)

    def _randomize_ego_spawn(self, obs: np.ndarray, info: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        target = self._sample_ego_spawn_target()
        if target is None:
            return obs, info
        handler = self._handler()
        viewer = self._viewer()
        if handler is None or viewer is None:
            return obs, info

        world_x, _, world_z = telemetry_to_unity_world(target.x, 0.0, target.z)
        qx, qy, qz, qw = yaw_deg_to_unity_quaternion(target.yaw_deg)
        msg = {
            "msg_type": "set_position",
            "pos_x": str(world_x),
            "pos_y": "0.5",
            "pos_z": str(world_z),
            "Qx": str(qx),
            "Qy": str(qy),
            "Qz": str(qz),
            "Qw": str(qw),
        }
        try:
            handler.send_control(0.0, 0.0, 1.0)
        except Exception:
            pass
        handler.blocking_send(msg)

        last_obs = np.asarray(obs)
        last_info = dict(info)
        settle_steps = max(1, int(self.config.ego_spawn_settle_steps))
        settle_sleep_s = float(max(0.0, self.config.ego_spawn_settle_sleep_s))
        for _ in range(settle_steps):
            if settle_sleep_s > 0.0:
                time.sleep(settle_sleep_s)
            try:
                handler.send_control(0.0, 0.0, 1.0)
            except Exception:
                pass
            observed = self._observe_info_and_obs()
            if observed is not None:
                last_obs, last_info = observed
        return np.asarray(last_obs), dict(last_info)

    def _sample_ego_spawn_target(self) -> Optional[TrackTarget]:
        if self._scene_key not in self.track_geometry.scenes:
            return None
        progress_ratio = float(self.rng.random())
        return sample_track_target(
            track_geometry=self.track_geometry,
            scene_key=self._scene_key,
            progress_ratio=progress_ratio,
            lateral_ratio=float(np.clip(self.config.ego_spawn_lateral_ratio, 0.0, 1.0)),
            obstacle_radius=0.0,
            safety_margin=0.0,
        )

    def _refresh_obstacle_layout(self, agent_info: Dict[str, Any]) -> bool:
        if self._fleet is None:
            return False
        active_count = min(len(self._fleet.cars), self._active_obstacle_count_for_scene())
        active_modes = self._select_episode_modes(
            active_count=active_count,
            active_modes=self._active_obstacle_modes_for_scene(),
        )
        obstacle_radius, safety_margin = self._effective_spawn_clearance()
        targets = list(self._sample_episode_targets(agent_info=agent_info, count=active_count))
        if not targets:
            self._episode_modes_used = tuple()
            self._episode_target_plan = tuple()
            self._log_runtime_debug(
                "layout_no_targets",
                active_count=int(active_count),
                modes=[str(mode) for mode in active_modes],
                agent=self._agent_debug_payload(agent_info),
            )
            self._deactivate_inactive_fleet(reason="layout_no_targets")
            return False
        planned_targets = list(targets[:active_count])
        self._episode_target_plan = tuple(planned_targets)
        self._episode_modes_used = tuple(
            str(mode).strip().lower()
            for mode in active_modes[:active_count]
        )

        for idx, car in enumerate(self._fleet.cars):
            if idx >= active_count:
                try:
                    self._park_car(car, idx, reason="inactive_extra")
                except Exception as exc:
                    self._log_runtime_debug(
                        "park_car_error",
                        idx=int(idx),
                        reason="inactive_extra",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                continue

            target = planned_targets[idx]
            mode = active_modes[idx].strip().lower()
            yaw_override = None
            if mode != "lane_pid" and self.config.randomize_non_lane_pid_yaw:
                yaw_override = self._sample_random_obstacle_yaw_deg()
            if mode == "static":
                if yaw_override is not None:
                    target = _with_target_yaw(target, yaw_override)
                placed_target = self._place_static_target_with_fallback(
                    car=car,
                    idx=idx,
                    target=target,
                    obstacle_radius=obstacle_radius,
                    safety_margin=safety_margin,
                )
                if placed_target is None:
                    self._last_runtime_error = "unstable_static_obstacle_placement"
                    self._episode_modes_used = tuple()
                    self._episode_target_plan = tuple()
                    self._deactivate_inactive_fleet(
                        reason="unstable_static_obstacle_placement"
                    )
                    return False
                target = placed_target
                planned_targets[idx] = placed_target
            elif mode == "jitter":
                car.start_position_jitter(
                    progress_ratio=target.progress_ratio,
                    lateral_ratio=target.lateral_ratio,
                    amplitude_m=self.config.jitter_amplitude_m,
                    period_s=self.config.jitter_period_s,
                    update_hz=self.config.jitter_update_hz,
                    yaw_deg_override=yaw_override,
                    obstacle_radius=obstacle_radius,
                    safety_margin=safety_margin,
                )
            elif mode == "nudge":
                car.start_in_place_nudge(
                    progress_ratio=target.progress_ratio,
                    lateral_ratio=target.lateral_ratio,
                    amplitude_m=self.config.nudge_amplitude_m,
                    period_s=self.config.nudge_period_s,
                    update_hz=self.config.nudge_update_hz,
                    yaw_deg_override=yaw_override,
                    obstacle_radius=obstacle_radius,
                    safety_margin=safety_margin,
                )
            elif mode == "lane_pid":
                placed_target = self._start_lane_pid_with_fallback(
                    car=car,
                    idx=idx,
                    target=target,
                    obstacle_radius=obstacle_radius,
                    safety_margin=safety_margin,
                )
                if placed_target is None:
                    self._last_runtime_error = "unstable_lane_pid_obstacle_placement"
                    self._episode_modes_used = tuple()
                    self._episode_target_plan = tuple()
                    self._deactivate_inactive_fleet(
                        reason="unstable_lane_pid_obstacle_placement"
                    )
                    return False
                target = placed_target
                planned_targets[idx] = placed_target
            else:
                raise ValueError(f"Unsupported obstacle mode: {mode}")
        self._episode_target_plan = tuple(planned_targets)
        self._log_apply_debug(agent_info=agent_info, targets=planned_targets, modes=active_modes[:active_count])
        return True

    def _place_static_target_with_fallback(
        self,
        car,
        idx: int,
        target: TrackTarget,
        obstacle_radius: float,
        safety_margin: float,
    ) -> Optional[TrackTarget]:
        pose = self._place_static_target(car=car, target=target)
        if self._active_target_pose_is_stable(pose, target):
            return target

        self._log_runtime_debug(
            "active_place_unstable",
            idx=int(idx),
            target=self._target_debug_payload(target),
            observed=self._pose_debug_payload(pose),
            target_error=self._target_error_debug(pose, target),
        )
        if self._scene_key != "waveshare":
            return target

        self._reset_static_car_for_fallback(
            car=car,
            idx=idx,
            reason="unstable_static_obstacle_placement",
        )
        retry_pose = self._place_static_target(car=car, target=target)
        retry_stable = self._active_target_pose_is_stable(retry_pose, target)
        self._log_runtime_debug(
            "active_place_retry_after_reset",
            idx=int(idx),
            stable=bool(retry_stable),
            target=self._target_debug_payload(target),
            observed=self._pose_debug_payload(retry_pose),
            target_error=self._target_error_debug(retry_pose, target),
        )
        if retry_stable:
            return target
        if self._active_target_pose_is_obviously_unstable(retry_pose):
            self._reset_static_car_for_fallback(
                car=car,
                idx=idx,
                reason="unstable_static_obstacle_retry",
            )

        for fallback_target in self._ws_static_fallback_targets(
            target=target,
            obstacle_radius=obstacle_radius,
            safety_margin=safety_margin,
        ):
            fallback_pose = self._place_static_target(car=car, target=fallback_target)
            stable = self._active_target_pose_is_stable(fallback_pose, fallback_target)
            self._log_runtime_debug(
                "active_place_fallback",
                idx=int(idx),
                stable=bool(stable),
                from_target=self._target_debug_payload(target),
                target=self._target_debug_payload(fallback_target),
                observed=self._pose_debug_payload(fallback_pose),
                target_error=self._target_error_debug(fallback_pose, fallback_target),
            )
            if stable:
                return fallback_target
            if self._active_target_pose_is_obviously_unstable(fallback_pose):
                self._reset_static_car_for_fallback(
                    car=car,
                    idx=idx,
                    reason="unstable_static_obstacle_fallback",
                )
        return None

    def _start_lane_pid_with_fallback(
        self,
        car,
        idx: int,
        target: TrackTarget,
        obstacle_radius: float,
        safety_margin: float,
    ) -> Optional[TrackTarget]:
        placed_target, pose = self._start_lane_pid_target(
            car=car,
            target=target,
            obstacle_radius=obstacle_radius,
            safety_margin=safety_margin,
        )
        if self._active_target_pose_is_stable(pose, placed_target):
            return placed_target

        self._log_runtime_debug(
            "active_lane_pid_unstable",
            idx=int(idx),
            target=self._target_debug_payload(placed_target),
            observed=self._pose_debug_payload(pose),
            target_error=self._target_error_debug(pose, placed_target),
        )
        if self._scene_key != "waveshare":
            return placed_target

        self._reset_static_car_for_fallback(
            car=car,
            idx=idx,
            reason="unstable_lane_pid_obstacle_placement",
        )
        retry_target, retry_pose = self._start_lane_pid_target(
            car=car,
            target=target,
            obstacle_radius=obstacle_radius,
            safety_margin=safety_margin,
        )
        retry_stable = self._active_target_pose_is_stable(retry_pose, retry_target)
        self._log_runtime_debug(
            "active_lane_pid_retry_after_reset",
            idx=int(idx),
            stable=bool(retry_stable),
            target=self._target_debug_payload(retry_target),
            observed=self._pose_debug_payload(retry_pose),
            target_error=self._target_error_debug(retry_pose, retry_target),
        )
        if retry_stable:
            return retry_target
        if self._active_target_pose_is_obviously_unstable(retry_pose):
            self._reset_static_car_for_fallback(
                car=car,
                idx=idx,
                reason="unstable_lane_pid_obstacle_retry",
            )
        return None

    def _start_lane_pid_target(
        self,
        car,
        target: TrackTarget,
        obstacle_radius: float,
        safety_margin: float,
    ) -> Tuple[TrackTarget, Optional[PoseState]]:
        placed_target = car.start_lane_pid(
            target_speed=self._lane_pid_speed_for_scene(),
            progress_ratio=target.progress_ratio,
            lateral_ratio=target.lateral_ratio,
            lookahead_m=self.config.lane_pid_lookahead_m,
            place_on_start=True,
            obstacle_radius=obstacle_radius,
            safety_margin=safety_margin,
        )
        if not isinstance(placed_target, TrackTarget):
            placed_target = target
        pose = self._observe_active_target_pose(car=car, target=placed_target)
        return placed_target, pose

    def _place_static_target(self, car, target: TrackTarget) -> Optional[PoseState]:
        try:
            car.stop_motion(hold_brake=True)
        except Exception:
            pass
        car.place_explicit_target(
            target=target,
            hold_brake=True,
            timeout_s=self.config.placement_timeout_s,
        )
        car.stop_motion(hold_brake=True)
        return self._observe_active_target_pose(car=car, target=target)

    def _reset_static_car_for_fallback(self, car, idx: int, reason: str) -> None:
        timeout_s = min(1.0, max(0.2, float(self.config.placement_timeout_s)))
        ok = False
        if hasattr(car, "reset_and_wait"):
            try:
                ok = bool(car.reset_and_wait(reason=reason, timeout_s=timeout_s))
            except Exception as exc:
                self._log_runtime_debug(
                    "active_place_reset_error",
                    idx=int(idx),
                    reason=str(reason),
                    error=f"{type(exc).__name__}: {exc}",
                )
        elif hasattr(car, "reset"):
            try:
                car.reset(reason=reason)
                time.sleep(min(0.3, timeout_s))
                ok = True
            except Exception as exc:
                self._log_runtime_debug(
                    "active_place_reset_error",
                    idx=int(idx),
                    reason=str(reason),
                    error=f"{type(exc).__name__}: {exc}",
                )
        try:
            car.stop_motion(hold_brake=True)
        except Exception:
            pass
        self._log_runtime_debug(
            "active_place_reset_before_fallback",
            idx=int(idx),
            reason=str(reason),
            ok=bool(ok),
            after=self._car_debug_state(idx),
        )

    def _observe_active_target_pose(self, car, target: TrackTarget) -> Optional[PoseState]:
        deadline = time.time() + min(0.50, max(0.10, float(self.config.placement_timeout_s) * 0.35))
        last_pose: Optional[PoseState] = None
        first = True
        while first or time.time() < deadline:
            first = False
            try:
                pose = car.get_obstacle_pose()
            except Exception as exc:
                self._log_runtime_debug(
                    "active_place_observe_error",
                    target=self._target_debug_payload(target),
                    error=f"{type(exc).__name__}: {exc}",
                )
                return last_pose
            if pose is not None:
                last_pose = pose
                if self._active_target_pose_is_stable(pose, target):
                    return pose
                if self._active_target_pose_is_obviously_unstable(pose):
                    return pose
            time.sleep(0.02)
        return last_pose

    def _ws_static_fallback_targets(
        self,
        target: TrackTarget,
        obstacle_radius: float,
        safety_margin: float,
    ) -> List[TrackTarget]:
        if self._scene_key != "waveshare":
            return []
        source_lateral = float(target.lateral_ratio)
        lane_choices = self._lane_choices_for_scene()
        preserve_lateral = (
            (len(lane_choices) == 1 and abs(source_lateral - float(lane_choices[0])) <= 1e-3)
            or (
                any(abs(float(choice)) <= 1e-3 for choice in lane_choices)
                and any(abs(float(choice) - 1.0) <= 1e-3 for choice in lane_choices)
                and (abs(source_lateral) <= 1e-3 or abs(source_lateral - 1.0) <= 1e-3)
            )
        )
        if preserve_lateral:
            scene = self.track_geometry.scenes.get(self._scene_key) if self.track_geometry is not None else None
            loop_len = float(getattr(scene, "loop_len", 0.0) or 0.0)
            progress_step = max(0.50 / max(loop_len, 1e-6), 0.01)
            progress_offsets = (
                progress_step,
                -progress_step,
                2.0 * progress_step,
                -2.0 * progress_step,
                3.0 * progress_step,
                -3.0 * progress_step,
            )
            bounds = self._scene_progress_ratio_bounds()
            candidates: List[TrackTarget] = []
            for offset in progress_offsets:
                progress_ratio = _wrap_progress(float(target.progress_ratio) + float(offset))
                if bounds is not None and not (bounds[0] <= progress_ratio <= bounds[1]):
                    continue
                candidate = sample_track_target(
                    track_geometry=self.track_geometry,
                    scene_key=self._scene_key,
                    progress_ratio=progress_ratio,
                    lateral_ratio=source_lateral,
                    obstacle_radius=obstacle_radius,
                    safety_margin=safety_margin,
                )
                candidates.append(_with_target_yaw(candidate, target.yaw_deg))
            return candidates

        if source_lateral >= 0.5:
            fallback_laterals = (0.65, 0.60, 0.55, 0.50, 0.40, 0.35)
        else:
            fallback_laterals = (0.35, 0.40, 0.45, 0.50, 0.60, 0.65)

        candidates: List[TrackTarget] = []
        for lateral in fallback_laterals:
            if abs(float(lateral) - source_lateral) <= 1e-3:
                continue
            candidate = sample_track_target(
                track_geometry=self.track_geometry,
                scene_key=self._scene_key,
                progress_ratio=target.progress_ratio,
                lateral_ratio=float(lateral),
                obstacle_radius=obstacle_radius,
                safety_margin=safety_margin,
            )
            candidates.append(_with_target_yaw(candidate, target.yaw_deg))
        return candidates

    def _active_target_pose_is_stable(
        self,
        pose: Optional[PoseState],
        target: Optional[TrackTarget],
    ) -> bool:
        if pose is None or target is None:
            return False
        if float(pose.y) < -0.5:
            return False
        if abs(float(pose.speed)) > 8.0:
            return False
        target_error = self._target_error_debug(pose, target)
        if isinstance(target_error, dict):
            planar = target_error.get("planar")
            if planar is not None and float(planar) > 0.75:
                return False
        return True

    @staticmethod
    def _active_target_pose_is_obviously_unstable(pose: Optional[PoseState]) -> bool:
        if pose is None:
            return False
        return float(pose.y) < -1.0

    def _log_reset_debug(
        self,
        agent_info: Dict[str, Any],
        should_spawn: bool,
        spawn_decision: str,
        free_prob: Optional[float],
    ) -> None:
        """Emit one compact line per reset so visual sim state can be audited."""
        payload: Dict[str, Any] = {
            "scene": self._logging_key or self._scene_key,
            "scene_key": self._scene_key,
            "episode": int(self._episode_index),
            "runtime_supported": bool(self._scene_supports_runtime()),
            "active": bool(self._active_this_episode),
            "should_spawn": bool(should_spawn),
            "decision": str(spawn_decision),
            "free_prob": _debug_float(free_prob),
            "fixed_progress": _debug_float(self._episode_fixed_progress_ratio, 4),
            "modes": list(self._episode_modes_used),
            "runtime_error": self._last_runtime_error or "",
        }
        agent_pose = pose_from_info(agent_info, self.track_geometry, self._scene_key, None)
        if agent_pose is not None:
            payload["agent"] = {
                "progress": _debug_float(agent_pose.progress_ratio, 4),
                "x": _debug_float(agent_pose.x),
                "z": _debug_float(agent_pose.z),
                "yaw": _debug_float(agent_pose.yaw_deg, 1),
                "cte": _debug_float(agent_pose.cte),
                **self._geometry_point_debug(agent_pose.x, agent_pose.z),
            }
        if self._episode_target_plan and self._active_this_episode:
            payload["planned_targets"] = [
                {
                    "progress": _debug_float(target.progress_ratio, 4),
                    "lateral": _debug_float(target.lateral_ratio, 3),
                    "x": _debug_float(target.x),
                    "z": _debug_float(target.z),
                    "yaw": _debug_float(target.yaw_deg, 1),
                }
                for target in self._episode_target_plan
            ]
        if self._fleet is None:
            print("[obstacle_reset] " + json.dumps(payload, sort_keys=True, ensure_ascii=False))
            return

        try:
            snapshots = self._fleet.get_snapshots(agent_info=agent_info)
        except Exception as exc:
            payload["snapshot_error"] = f"{type(exc).__name__}: {exc}"
            print("[obstacle_reset] " + json.dumps(payload, sort_keys=True, ensure_ascii=False))
            return

        entries: List[Dict[str, Any]] = []
        for idx, snap in enumerate(snapshots):
            target = snap.target
            pose = snap.obstacle
            rel = snap.relative
            entries.append(
                {
                    "idx": int(idx),
                    "client": self._car_debug_state(idx),
                    "mode": (
                        str(self._episode_modes_used[idx])
                        if idx < len(self._episode_modes_used)
                        else "parked"
                    ),
                    "target": None
                    if target is None or not self._active_this_episode
                    else {
                        "progress": _debug_float(target.progress_ratio, 4),
                        "lateral": _debug_float(target.lateral_ratio, 3),
                        "x": _debug_float(target.x),
                        "z": _debug_float(target.z),
                        "yaw": _debug_float(target.yaw_deg, 1),
                        **self._geometry_point_debug(target.x, target.z),
                    },
                    "pose": None
                    if pose is None
                    else {
                        "progress": _debug_float(pose.progress_ratio, 4),
                        "x": _debug_float(pose.x),
                        "z": _debug_float(pose.z),
                        "yaw": _debug_float(pose.yaw_deg, 1),
                        "cte": _debug_float(pose.cte),
                        **self._geometry_point_debug(pose.x, pose.z),
                    },
                    "relative": None
                    if rel is None
                    else {
                        "longitudinal": _debug_float(rel.longitudinal),
                        "lateral": _debug_float(rel.lateral),
                        "planar": _debug_float(rel.planar_distance),
                    },
                    "target_error": self._target_error_debug(pose, target),
                }
            )
        payload["obstacles"] = entries
        payload["last_errors"] = self._fleet.last_errors()
        print("[obstacle_reset] " + json.dumps(payload, sort_keys=True, ensure_ascii=False))

    def _log_fleet_debug(self, event: str) -> None:
        if self._fleet is None:
            return
        payload = {
            "event": str(event),
            "scene": self._logging_key or self._scene_key,
            "scene_key": self._scene_key,
            "fleet_scene_key": self._fleet_scene_key,
            "port": int(self.conf.get("port", 0) or 0),
            "count": int(len(self._fleet.cars)),
            "clients": [self._car_debug_state(idx) for idx in range(len(self._fleet.cars))],
            "last_errors": self._fleet.last_errors(),
        }
        print("[obstacle_fleet] " + json.dumps(payload, sort_keys=True, ensure_ascii=False))

    def _log_apply_debug(
        self,
        agent_info: Dict[str, Any],
        targets: Sequence[TrackTarget],
        modes: Sequence[str],
    ) -> None:
        if self._fleet is None:
            return
        try:
            snapshots = self._fleet.get_snapshots(agent_info=agent_info)
        except Exception as exc:
            print(
                "[obstacle_apply] "
                + json.dumps(
                    {
                        "scene": self._logging_key or self._scene_key,
                        "scene_key": self._scene_key,
                        "episode": int(self._episode_index),
                        "snapshot_error": f"{type(exc).__name__}: {exc}",
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                )
            )
            return
        entries: List[Dict[str, Any]] = []
        for idx, snap in enumerate(snapshots):
            target = targets[idx] if idx < len(targets) else snap.target
            pose = snap.obstacle
            rel = snap.relative
            entries.append(
                {
                    "idx": int(idx),
                    "client": self._car_debug_state(idx),
                    "mode": str(modes[idx]) if idx < len(modes) else "parked",
                    "target": None
                    if target is None
                    else {
                        "progress": _debug_float(target.progress_ratio, 4),
                        "lateral": _debug_float(target.lateral_ratio, 3),
                        "x": _debug_float(target.x),
                        "z": _debug_float(target.z),
                        "yaw": _debug_float(target.yaw_deg, 1),
                        **self._geometry_point_debug(target.x, target.z),
                    },
                    "pose": None
                    if pose is None
                    else {
                        "progress": _debug_float(pose.progress_ratio, 4),
                        "x": _debug_float(pose.x),
                        "z": _debug_float(pose.z),
                        "yaw": _debug_float(pose.yaw_deg, 1),
                        "cte": _debug_float(pose.cte),
                        **self._geometry_point_debug(pose.x, pose.z),
                    },
                    "relative": None
                    if rel is None
                    else {
                        "longitudinal": _debug_float(rel.longitudinal),
                        "lateral": _debug_float(rel.lateral),
                        "planar": _debug_float(rel.planar_distance),
                    },
                    "target_error": self._target_error_debug(pose, target),
                }
            )
        payload = {
            "phase": "post_layout",
            "scene": self._logging_key or self._scene_key,
            "scene_key": self._scene_key,
            "episode": int(self._episode_index),
            "active": True,
            "modes": [str(mode) for mode in modes],
            "obstacles": entries,
            "last_errors": self._fleet.last_errors(),
        }
        print("[obstacle_apply] " + json.dumps(payload, sort_keys=True, ensure_ascii=False))

    @staticmethod
    def _pose_debug_payload(pose: Optional[PoseState]) -> Optional[Dict[str, Any]]:
        if pose is None:
            return None
        return {
            "progress": _debug_float(pose.progress_ratio, 4),
            "x": _debug_float(pose.x),
            "y": _debug_float(pose.y),
            "z": _debug_float(pose.z),
            "yaw": _debug_float(pose.yaw_deg, 1),
            "speed": _debug_float(pose.speed),
            "cte": _debug_float(pose.cte),
            "hit": str(pose.hit),
        }

    def _target_debug_payload(self, target: Optional[TrackTarget]) -> Optional[Dict[str, Any]]:
        if target is None:
            return None
        payload = {
            "progress": _debug_float(target.progress_ratio, 4),
            "lateral": _debug_float(target.lateral_ratio, 3),
            "x": _debug_float(target.x),
            "z": _debug_float(target.z),
            "yaw": _debug_float(target.yaw_deg, 1),
        }
        payload.update(self._geometry_point_debug(target.x, target.z))
        return payload

    def _agent_debug_payload(self, info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pose = pose_from_info(info, self.track_geometry, self._scene_key, None)
        if pose is None:
            return None
        payload = self._pose_debug_payload(pose) or {}
        payload.update(self._geometry_point_debug(pose.x, pose.z))
        return payload

    def _snapshot_debug_entries(
        self,
        snapshots: Sequence[ObstacleSnapshot],
    ) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for idx, snap in enumerate(snapshots):
            pose = snap.obstacle
            target = snap.target
            rel = snap.relative
            entries.append(
                {
                    "idx": int(idx),
                    "mode": (
                        str(self._episode_modes_used[idx])
                        if idx < len(self._episode_modes_used)
                        else "parked"
                    ),
                    "client": self._car_debug_state(idx),
                    "pose": self._pose_debug_payload(pose),
                    "target": None if target is None else {
                        "progress": _debug_float(target.progress_ratio, 4),
                        "lateral": _debug_float(target.lateral_ratio, 3),
                        "x": _debug_float(target.x),
                        "z": _debug_float(target.z),
                        "yaw": _debug_float(target.yaw_deg, 1),
                    },
                    "relative": None if rel is None else {
                        "longitudinal": _debug_float(rel.longitudinal),
                        "lateral": _debug_float(rel.lateral),
                        "planar": _debug_float(rel.planar_distance),
                    },
                    "target_error": self._target_error_debug(pose, target),
                }
            )
        return entries

    def _log_runtime_debug(self, event: str, **fields: Any) -> None:
        payload: Dict[str, Any] = {
            "event": str(event),
            "t": _debug_float(time.time(), 3),
            "scene": self._logging_key or self._scene_key,
            "scene_key": self._scene_key,
            "episode": int(self._episode_index),
            "active": bool(self._active_this_episode),
            "modes": list(self._episode_modes_used),
            "runtime_error": self._last_runtime_error or "",
        }
        payload.update(fields)
        print("[obstacle_runtime] " + json.dumps(payload, sort_keys=True, ensure_ascii=False), flush=True)

    def _maybe_log_obstacle_watchdog(
        self,
        agent_info: Dict[str, Any],
        snapshots: Sequence[ObstacleSnapshot],
    ) -> None:
        if self._scene_key != "waveshare":
            return
        step = int(self._debug_step_count_this_episode)
        watch_steps = {1, 5, 15, 30, 60, 120}
        entries = self._snapshot_debug_entries(snapshots)
        anomaly = False
        for entry in entries:
            mode = str(entry.get("mode", "")).strip().lower()
            target_error = entry.get("target_error")
            pose = entry.get("pose")
            if pose is None:
                anomaly = True
                break
            if isinstance(pose, dict):
                y = pose.get("y")
                speed = pose.get("speed")
                if y is not None and float(y) < -1.0:
                    anomaly = True
                    break
                if speed is not None and float(speed) > 8.0:
                    anomaly = True
                    break
            if mode != "lane_pid" and isinstance(target_error, dict):
                planar = target_error.get("planar")
                if planar is not None and float(planar) > 0.75:
                    anomaly = True
                    break
        now = time.time()
        scheduled = step in watch_steps and step not in self._debug_watch_logged_steps
        if not scheduled and not anomaly:
            return
        if not scheduled and anomaly and now - float(self._debug_last_watch_anomaly_t) < 1.0:
            return
        if scheduled:
            self._debug_watch_logged_steps.add(step)
        if anomaly:
            self._debug_last_watch_anomaly_t = now
        self._log_runtime_debug(
            "watchdog",
            step=step,
            scheduled=bool(scheduled),
            anomaly=bool(anomaly),
            agent=self._agent_debug_payload(agent_info),
            obstacles=entries,
            last_errors=self._fleet.last_errors() if self._fleet is not None else [],
        )

    def _car_debug_state(self, idx: int) -> Dict[str, Any]:
        if self._fleet is None or idx < 0 or idx >= len(self._fleet.cars):
            return {}
        car = self._fleet.cars[int(idx)]
        if hasattr(car, "debug_state"):
            try:
                return dict(car.debug_state())
            except Exception as exc:
                return {"debug_error": f"{type(exc).__name__}: {exc}"}
        return {}

    @staticmethod
    def _target_error_debug(
        pose: Optional[PoseState],
        target: Optional[TrackTarget],
    ) -> Optional[Dict[str, Optional[float]]]:
        if pose is None or target is None:
            return None
        dx = float(pose.x - target.x)
        dz = float(pose.z - target.z)
        yaw_err = (float(pose.yaw_deg) - float(target.yaw_deg) + 180.0) % 360.0 - 180.0
        return {
            "dx": _debug_float(dx),
            "dz": _debug_float(dz),
            "planar": _debug_float(math.hypot(dx, dz)),
            "yaw": _debug_float(yaw_err, 1),
        }

    def _geometry_point_debug(self, x: float, z: float) -> Dict[str, Optional[float]]:
        if self._scene_key not in self.track_geometry.scenes:
            return {}
        try:
            geo = self.track_geometry.query(self._scene_key, x=float(x), z=float(z), yaw_rad=0.0)
            scene = self.track_geometry.scenes[self._scene_key]
            lat_err = float(geo.get("lat_err", 0.0) or 0.0)
            return {
                "geom_lat_err": _debug_float(lat_err),
                "geom_cte": _debug_float(lat_err * float(scene.coord_scale)),
                "geom_norm": _debug_float(geo.get("lat_err_norm")),
            }
        except Exception:
            return {}

    def _select_episode_modes(
        self,
        active_count: int,
        active_modes: Sequence[str],
    ) -> Tuple[str, ...]:
        cleaned_modes = tuple(
            str(mode).strip().lower()
            for mode in active_modes
            if str(mode).strip()
        )
        if active_count <= 0:
            return tuple()
        if not cleaned_modes:
            return ("static",) * int(active_count)
        if len(cleaned_modes) == 1:
            return cleaned_modes * int(active_count)

        if int(active_count) <= len(cleaned_modes):
            chosen = self.rng.choice(
                np.asarray(cleaned_modes, dtype=object),
                size=int(active_count),
                replace=False,
            )
            return tuple(str(mode) for mode in chosen.tolist())

        chosen = list(cleaned_modes)
        extra = self.rng.choice(
            np.asarray(cleaned_modes, dtype=object),
            size=int(active_count) - len(cleaned_modes),
            replace=True,
        )
        chosen.extend(str(mode) for mode in extra.tolist())
        self.rng.shuffle(chosen)
        return tuple(chosen)

    @staticmethod
    def _episode_mode_counts(modes: Sequence[str]) -> Dict[str, int]:
        counts = {
            "static": 0,
            "jitter": 0,
            "nudge": 0,
            "lane_pid": 0,
        }
        for mode in modes:
            cleaned = str(mode).strip().lower()
            if cleaned in counts:
                counts[cleaned] += 1
        return counts

    def _sample_episode_targets(self, agent_info: Dict[str, Any], count: int) -> List[TrackTarget]:
        if self._scene_key not in self.track_geometry.scenes or count <= 0:
            return []

        g = self.track_geometry.scenes[self._scene_key]
        agent_pose = pose_from_info(agent_info, self.track_geometry, self._scene_key, None)
        if agent_pose is None or agent_pose.progress_ratio is None:
            agent_progress = float(self.rng.random())
        else:
            agent_progress = float(agent_pose.progress_ratio)

        lane_choices = list(self._lane_choices_for_scene())
        obstacle_radius, safety_margin = self._effective_spawn_clearance()
        fixed_progress_ratio = self._scene_fixed_progress_ratio()
        fixed_lateral_ratio = self._scene_fixed_lateral_ratio()
        fixed_progress_gap_bounds = self._scene_fixed_progress_gap_bounds()
        fixed_progress_gap_ratio = self._scene_fixed_progress_gap_ratio()
        progress_bounds: Optional[Tuple[float, float]] = None
        sampled_fixed_base = False
        if fixed_progress_ratio is None and fixed_progress_gap_bounds is not None:
            progress_bounds = self._scene_progress_ratio_bounds()
            if progress_bounds is not None:
                fixed_progress_ratio = float(self.rng.uniform(progress_bounds[0], progress_bounds[1]))
                sampled_fixed_base = True
        if fixed_progress_ratio is not None:
            best_targets: List[TrackTarget] = []
            best_score = -1.0
            max_fixed_attempts = 32 if sampled_fixed_base and fixed_progress_gap_bounds is not None else 1
            for attempt in range(max_fixed_attempts):
                attempt_progress = fixed_progress_ratio
                if attempt > 0 and progress_bounds is not None:
                    attempt_progress = float(self.rng.uniform(progress_bounds[0], progress_bounds[1]))
                targets = self._sample_fixed_episode_targets(
                    agent_pose=agent_pose,
                    count=count,
                    g=g,
                    lane_choices=lane_choices,
                    obstacle_radius=obstacle_radius,
                    safety_margin=safety_margin,
                    fixed_progress_ratio=attempt_progress,
                    fixed_lateral_ratio=fixed_lateral_ratio,
                    fixed_progress_gap_bounds=fixed_progress_gap_bounds,
                    fixed_progress_gap_ratio=fixed_progress_gap_ratio,
                )
                score = self._fixed_target_set_score(targets, agent_pose, g)
                if score > best_score:
                    best_targets = targets
                    best_score = score
                if self._fixed_target_set_is_valid(targets, agent_pose, g, fixed_progress_gap_bounds):
                    return targets
            return best_targets

        targets: List[TrackTarget] = []
        max_attempts = max(64, 32 * int(count))
        for _ in range(max_attempts):
            if len(targets) >= int(count):
                break
            target = sample_track_target(
                track_geometry=self.track_geometry,
                scene_key=self._scene_key,
                progress_ratio=self._sample_nonfixed_progress(agent_progress, g),
                lateral_ratio=float(self.rng.choice(lane_choices)),
                obstacle_radius=obstacle_radius,
                safety_margin=safety_margin,
            )
            if not self._target_is_valid(target, agent_pose, targets, g):
                continue
            targets.append(target)

        if len(targets) >= int(count):
            return targets

        while len(targets) < int(count):
            fallback_progress = self._fallback_nonfixed_progress(
                agent_progress=agent_progress,
                g=g,
                target_index=len(targets),
                target_count=int(count),
            )
            targets.append(
                sample_track_target(
                    track_geometry=self.track_geometry,
                    scene_key=self._scene_key,
                    progress_ratio=_wrap_progress(fallback_progress),
                    lateral_ratio=float(lane_choices[len(targets) % len(lane_choices)]),
                    obstacle_radius=obstacle_radius,
                    safety_margin=safety_margin,
                )
            )
        return targets

    def _sample_fixed_episode_targets(
        self,
        agent_pose: Optional[PoseState],
        count: int,
        g: SceneGeometry,
        lane_choices: Sequence[float],
        obstacle_radius: float,
        safety_margin: float,
        fixed_progress_ratio: float,
        fixed_lateral_ratio: Optional[float],
        fixed_progress_gap_bounds: Optional[Tuple[float, float]] = None,
        fixed_progress_gap_ratio: Optional[float] = None,
    ) -> List[TrackTarget]:
        base_progress = _wrap_progress(float(fixed_progress_ratio))
        if fixed_lateral_ratio is not None:
            base_laterals = [
                float(np.clip(fixed_lateral_ratio, 0.0, 1.0))
                for _ in range(int(count))
            ]
        else:
            fixed_choices = list(lane_choices or [0.5])
            start_idx = int(self.rng.integers(0, len(fixed_choices)))
            base_laterals = [
                float(fixed_choices[(start_idx + i) % len(fixed_choices)])
                for i in range(int(count))
            ]

        configured_gap = fixed_progress_gap_ratio
        if fixed_progress_gap_bounds is not None:
            gap_min, gap_max = fixed_progress_gap_bounds
            if abs(gap_max - gap_min) <= 1e-6:
                progress_step = float(gap_min) % 1.0
            else:
                progress_step = float(self.rng.uniform(gap_min, gap_max)) % 1.0
            if progress_step <= 1e-6:
                progress_step = max(float(self.config.min_agent_arc_dist_m), 0.5) / max(float(g.loop_len), 1e-6)
        elif configured_gap is None:
            progress_step = max(float(self.config.min_agent_arc_dist_m), 0.5) / max(float(g.loop_len), 1e-6)
        else:
            progress_step = float(configured_gap) % 1.0
            if progress_step <= 1e-6:
                progress_step = max(float(self.config.min_agent_arc_dist_m), 0.5) / max(float(g.loop_len), 1e-6)
        targets: List[TrackTarget] = []
        for idx in range(int(count)):
            base_candidate_progress = _wrap_progress(base_progress + idx * progress_step)
            candidate_progresses = [
                _wrap_progress(base_candidate_progress + float(offset) * progress_step)
                for offset in (0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0)
            ]
            chosen: Optional[TrackTarget] = None
            spacing_fallback: Optional[TrackTarget] = None
            for progress_ratio in candidate_progresses:
                candidate = sample_track_target(
                    track_geometry=self.track_geometry,
                    scene_key=self._scene_key,
                    progress_ratio=progress_ratio,
                    lateral_ratio=float(base_laterals[idx]),
                    obstacle_radius=obstacle_radius,
                    safety_margin=safety_margin,
                )
                if not self._fixed_progress_spacing_is_valid(
                    candidate=candidate,
                    existing=targets,
                    target_index=idx,
                    target_count=int(count),
                    fixed_progress_gap_bounds=fixed_progress_gap_bounds,
                ):
                    continue
                if spacing_fallback is None:
                    spacing_fallback = candidate
                if self._target_is_valid(candidate, agent_pose, targets, g):
                    chosen = candidate
                    break
            if chosen is None:
                chosen = spacing_fallback or sample_track_target(
                    track_geometry=self.track_geometry,
                    scene_key=self._scene_key,
                    progress_ratio=base_candidate_progress,
                    lateral_ratio=float(base_laterals[idx]),
                    obstacle_radius=obstacle_radius,
                    safety_margin=safety_margin,
                )
            targets.append(chosen)
        return targets

    @staticmethod
    def _directed_progress_gap_ratio(a: float, b: float) -> float:
        return float((float(a) - float(b)) % 1.0)

    @staticmethod
    def _circular_progress_gap_ratio(a: float, b: float) -> float:
        gap = abs(float(a) - float(b)) % 1.0
        return float(min(gap, 1.0 - gap))

    def _fixed_progress_spacing_is_valid(
        self,
        candidate: TrackTarget,
        existing: Sequence[TrackTarget],
        target_index: int,
        target_count: int,
        fixed_progress_gap_bounds: Optional[Tuple[float, float]],
    ) -> bool:
        if fixed_progress_gap_bounds is None or not existing:
            return True

        gap_min, gap_max = fixed_progress_gap_bounds
        gap_min = float(np.clip(gap_min, 0.0, 1.0))
        gap_max = float(np.clip(gap_max, 0.0, 1.0))
        if gap_max < gap_min:
            gap_min, gap_max = gap_max, gap_min

        progress = float(candidate.progress_ratio)
        if int(target_count) == 2 and int(target_index) == 1 and len(existing) == 1:
            directed_gap = self._directed_progress_gap_ratio(progress, existing[0].progress_ratio)
            if directed_gap + 1e-6 < gap_min or directed_gap > gap_max + 1e-6:
                return False

        if int(target_count) > 2:
            # With 3+ cars the configured step creates multiple pairwise gaps;
            # keep the closest circular gap from collapsing during fallback.
            required_min_gap = min(gap_min, max(0.0, (1.0 / float(target_count)) - 0.02))
        else:
            required_min_gap = min(gap_min, 0.5)
        for other in existing:
            if self._circular_progress_gap_ratio(progress, other.progress_ratio) + 1e-6 < required_min_gap:
                return False
        return True

    def _fixed_target_set_is_valid(
        self,
        targets: Sequence[TrackTarget],
        agent_pose: Optional[PoseState],
        g: SceneGeometry,
        fixed_progress_gap_bounds: Optional[Tuple[float, float]],
    ) -> bool:
        if not targets:
            return False
        existing: List[TrackTarget] = []
        target_count = len(targets)
        for idx, target in enumerate(targets):
            if not self._fixed_progress_spacing_is_valid(
                candidate=target,
                existing=existing,
                target_index=idx,
                target_count=target_count,
                fixed_progress_gap_bounds=fixed_progress_gap_bounds,
            ):
                return False
            if not self._target_is_valid(target, agent_pose, existing, g):
                return False
            existing.append(target)
        return True

    def _fixed_target_set_score(
        self,
        targets: Sequence[TrackTarget],
        agent_pose: Optional[PoseState],
        g: SceneGeometry,
    ) -> float:
        if not targets:
            return -1.0

        min_pair_gap = 1.0
        for i, a in enumerate(targets):
            for b in targets[i + 1:]:
                min_pair_gap = min(
                    min_pair_gap,
                    self._circular_progress_gap_ratio(a.progress_ratio, b.progress_ratio),
                )

        min_agent_gap = 1.0
        if agent_pose is not None and agent_pose.progress_ratio is not None:
            for target in targets:
                min_agent_gap = min(
                    min_agent_gap,
                    self._circular_progress_gap_ratio(target.progress_ratio, agent_pose.progress_ratio),
                )

        existing: List[TrackTarget] = []
        valid_count = 0
        for target in targets:
            if self._target_is_valid(target, agent_pose, existing, g):
                valid_count += 1
                existing.append(target)
        return float(valid_count) + float(min_pair_gap) + 0.25 * float(min_agent_gap)

    def _effective_spawn_clearance(self) -> Tuple[float, float]:
        preset = getattr(self._fleet, "preset", None)
        obstacle_radius = float(getattr(preset, "obstacle_radius", 0.20))
        safety_margin = float(getattr(preset, "safety_margin", 0.05))

        # 采样阶段要保证“整台车”的 footprint 更保守地留在赛道内，
        # 不能只保证目标中心点没有越界。
        if self._scene_key == "waveshare":
            # WS 赛道太窄，若按整车 footprint 夹紧会把目标重新压回中线附近；
            # 这里显式允许障碍贴边放置，接受少量车身越界。
            obstacle_radius = 0.0
            safety_margin = 0.0
        else:
            safety_margin = max(safety_margin, 0.10)

        active_modes = tuple(self._active_obstacle_modes_for_scene())
        if (
            self._scene_key != "waveshare"
            and self.config.randomize_non_lane_pid_yaw
            and any(mode != "lane_pid" for mode in active_modes)
        ):
            safety_margin += 0.05

        return obstacle_radius, float(safety_margin)

    def _sample_random_obstacle_yaw_deg(self) -> float:
        return float(self.rng.uniform(0.0, 360.0))

    def _lane_choices_for_scene(self) -> Tuple[float, ...]:
        if self._scene_key == "waveshare":
            if self.config.ws_lateral_choices:
                return tuple(sorted({
                    float(np.clip(x, 0.0, 1.0))
                    for x in self.config.ws_lateral_choices
                }))
            # Match the V16 WS obstacle curriculum: request the geometric
            # edges, while WS sampling disables footprint clearance so the
            # target center is not clipped back toward the middle.
            return (0.0, 1.0)
        configured_choices = tuple(sorted({float(x) for x in self.config.lateral_choices}))
        if configured_choices:
            return configured_choices
        if self._fleet is not None:
            preset = self._fleet.preset
            default_choices = tuple(sorted({float(r) for _p, r in preset.default_layout} | {0.5}))
            if default_choices:
                return default_choices
        return (0.35, 0.50, 0.65)

    def _active_obstacle_count_for_scene(self) -> int:
        if self._scene_key == "waveshare":
            if self.config.ws_obstacle_count is None:
                return 1
            return int(max(1, self.config.ws_obstacle_count))
        return int(max(1, self.config.obstacle_count))

    def _scene_obstacle_free_prob(self) -> float:
        if self._scene_key == "waveshare" and self.config.ws_obstacle_free_prob is not None:
            return float(np.clip(self.config.ws_obstacle_free_prob, 0.0, 1.0))
        return float(np.clip(self.config.obstacle_free_prob, 0.0, 1.0))

    def _scene_fixed_progress_ratio(self) -> Optional[float]:
        if self._episode_fixed_progress_ratio is not None:
            return float(self._episode_fixed_progress_ratio)
        if self._scene_key == "waveshare":
            if self.config.ws_obstacle_fixed_progress_ratio is not None:
                return float(self.config.ws_obstacle_fixed_progress_ratio)
            return None
        return self.config.fixed_progress_ratio

    def _sample_scene_fixed_progress_distribution(self) -> Optional[Tuple[bool, Optional[float]]]:
        if self._scene_key != "generated_track":
            return None
        distribution = self.config.fixed_progress_distribution
        if not distribution:
            return None

        draw = float(self.rng.random())
        cumulative = 0.0
        for probability, progress_ratio in distribution:
            p = float(np.clip(probability, 0.0, 1.0))
            if p <= 0.0:
                continue
            cumulative += p
            if draw < cumulative:
                return True, _wrap_progress(float(progress_ratio))
        return False, None

    def _scene_fixed_progress_gap_bounds(self) -> Optional[Tuple[float, float]]:
        if self._scene_key == "waveshare":
            gap_min = self.config.ws_fixed_progress_gap_ratio_min
            gap_max = self.config.ws_fixed_progress_gap_ratio_max
            if gap_min is None or gap_max is None:
                gap_min = self.config.fixed_progress_gap_ratio_min
                gap_max = self.config.fixed_progress_gap_ratio_max
        else:
            gap_min = self.config.fixed_progress_gap_ratio_min
            gap_max = self.config.fixed_progress_gap_ratio_max
        if gap_min is None or gap_max is None:
            return None
        gap_min = float(np.clip(gap_min, 0.0, 1.0))
        gap_max = float(np.clip(gap_max, 0.0, 1.0))
        if gap_max < gap_min:
            gap_min, gap_max = gap_max, gap_min
        return gap_min, gap_max

    def _scene_fixed_progress_gap_ratio(self) -> Optional[float]:
        if self._scene_key == "waveshare" and self.config.ws_fixed_progress_gap_ratio is not None:
            return float(np.clip(self.config.ws_fixed_progress_gap_ratio, 0.0, 1.0))
        if self.config.fixed_progress_gap_ratio is not None:
            return float(np.clip(self.config.fixed_progress_gap_ratio, 0.0, 1.0))
        return None

    def _scene_progress_ratio_bounds(self) -> Optional[Tuple[float, float]]:
        if self._scene_key == "generated_track":
            min_p = self.config.obstacle_progress_min
            max_p = self.config.obstacle_progress_max
        elif self._scene_key == "waveshare":
            min_p = self.config.ws_obstacle_progress_min
            max_p = self.config.ws_obstacle_progress_max
        else:
            return None
        if min_p is None or max_p is None:
            return None
        min_p = float(np.clip(min_p, 0.0, 1.0))
        max_p = float(np.clip(max_p, 0.0, 1.0))
        if max_p < min_p:
            min_p, max_p = max_p, min_p
        return float(min_p), float(max_p)

    def _scene_fixed_lateral_ratio(self) -> Optional[float]:
        if self._scene_key == "waveshare":
            if self.config.ws_obstacle_fixed_lateral_ratio is None:
                return None
            return float(self.config.ws_obstacle_fixed_lateral_ratio)
        return self.config.fixed_lateral_ratio

    def _active_obstacle_modes_for_scene(self) -> Tuple[str, ...]:
        if self._scene_key == "waveshare":
            if self.config.ws_obstacle_modes:
                return tuple(self.config.ws_obstacle_modes)
            return ("static",)
        return tuple(self.config.obstacle_modes) or ("static",)

    def _sample_ahead_progress(self, agent_progress: float, g: SceneGeometry) -> float:
        if self._scene_key == "waveshare":
            # WS loop is only about 8.3 m. The GT-oriented 4-10 m range
            # puts the obstacle on the opposite side of the oval, so the
            # agent and the human observer do not see it until much later.
            # Place it in the front camera corridor. At 2m+ arc distance on
            # this tiny oval the target is already around the bend and can be
            # invisible even though telemetry says it is nearby.
            ds_min = 1.0
            ds_max = min(1.8, 0.35 * float(g.loop_len))
            ds_max = max(ds_min + 0.35, ds_max)
        else:
            ds_min = float(max(0.2, self.config.spawn_ahead_min_m))
            ds_max = float(max(ds_min + 1e-3, self.config.spawn_ahead_max_m))
        delta_s = float(self.rng.uniform(ds_min, ds_max))
        return _wrap_progress(agent_progress + delta_s / max(float(g.loop_len), 1e-6))

    def _gt_start_exclusion_half_width_ratio(self, g: SceneGeometry) -> Optional[float]:
        if self._scene_key != "generated_track":
            return None
        half_width_m = self.config.gt_obstacle_start_exclusion_half_width_m
        if half_width_m is None:
            return None
        half_width_m = float(max(0.0, half_width_m))
        if half_width_m <= 1e-6:
            return None
        return float(np.clip(half_width_m / max(float(g.loop_len), 1e-6), 0.0, 0.49))

    def _sample_nonfixed_progress(self, agent_progress: float, g: SceneGeometry) -> float:
        progress_bounds = self._scene_progress_ratio_bounds()
        if self._scene_key == "waveshare":
            if progress_bounds is not None:
                min_p, max_p = progress_bounds
                if max_p <= min_p + 1e-6:
                    return float(min_p)
                return float(self.rng.uniform(min_p, max_p))
            return self._sample_ahead_progress(agent_progress, g)
        if progress_bounds is not None:
            min_p, max_p = progress_bounds
            if max_p <= min_p + 1e-6:
                return float(min_p)
            return float(self.rng.uniform(min_p, max_p))
        exclusion_ratio = self._gt_start_exclusion_half_width_ratio(g)
        if exclusion_ratio is not None:
            return float(self.rng.uniform(exclusion_ratio, 1.0 - exclusion_ratio))
        return self._sample_ahead_progress(agent_progress, g)

    def _fallback_nonfixed_progress(
        self,
        agent_progress: float,
        g: SceneGeometry,
        target_index: int,
        target_count: int,
    ) -> float:
        progress_bounds = self._scene_progress_ratio_bounds()
        if self._scene_key == "waveshare" and progress_bounds is None:
            return _wrap_progress(float(agent_progress) + 0.24 + 0.12 * float(target_index))
        if progress_bounds is not None:
            min_p, max_p = progress_bounds
            if max_p <= min_p + 1e-6:
                return float(min_p)
            span = max(1e-6, max_p - min_p)
            frac = (float(target_index) + 0.5) / max(1.0, float(target_count))
            return _wrap_progress(min_p + frac * span)
        exclusion_ratio = self._gt_start_exclusion_half_width_ratio(g)
        if exclusion_ratio is not None:
            span = max(1e-6, 1.0 - 2.0 * exclusion_ratio)
            frac = (float(target_index) + 0.5) / max(1.0, float(target_count))
            return _wrap_progress(exclusion_ratio + frac * span)
        return _wrap_progress(float(agent_progress + 0.18 + 0.12 * target_index))

    def _target_is_valid(
        self,
        target: TrackTarget,
        agent_pose: Optional[PoseState],
        existing: Sequence[TrackTarget],
        g: SceneGeometry,
    ) -> bool:
        if agent_pose is not None and agent_pose.progress_ratio is not None:
            planar = math.hypot(float(target.x - agent_pose.x), float(target.z - agent_pose.z))
            arc_gap = self._progress_gap_m(float(target.progress_ratio), float(agent_pose.progress_ratio), g)
            min_planar = float(self.config.min_agent_planar_dist_m)
            min_arc = float(self.config.min_agent_arc_dist_m)
            if self._scene_key == "waveshare":
                min_planar = min(min_planar, 0.9)
                min_arc = min(min_arc, 1.2)
            if planar < min_planar:
                return False
            if arc_gap < min_arc:
                return False

        min_obs_sep = max(0.1, float(self.config.min_obstacle_separation_world) / 8.0)
        for other in existing:
            planar = math.hypot(float(target.x - other.x), float(target.z - other.z))
            if planar < min_obs_sep:
                return False
        return True

    @staticmethod
    def _progress_gap_m(a: float, b: float, g: SceneGeometry) -> float:
        da = abs(float(a) - float(b)) % 1.0
        da = min(da, 1.0 - da)
        return float(da * float(g.loop_len))

    def _lane_pid_speed_for_scene(self) -> float:
        return (
            float(self.config.lane_pid_speed_ws)
            if self._scene_key == "waveshare"
            else float(self.config.lane_pid_speed_gt)
        )

    @staticmethod
    def _select_primary_snapshot_index(snapshots: Sequence[ObstacleSnapshot]) -> Optional[int]:
        candidates: List[Tuple[float, int]] = []
        fallback: List[Tuple[float, int]] = []
        for idx, snap in enumerate(snapshots):
            rel = snap.relative
            if rel is None:
                continue
            planar = float(rel.planar_distance)
            fallback.append((planar, idx))
            if float(rel.longitudinal) >= -0.4:
                score = planar + 0.15 * abs(float(rel.lateral))
                candidates.append((score, idx))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            return int(candidates[0][1])
        if fallback:
            fallback.sort(key=lambda item: item[0])
            return int(fallback[0][1])
        return None

    @classmethod
    def _select_primary_snapshot(cls, snapshots: Sequence[ObstacleSnapshot]) -> Optional[ObstacleSnapshot]:
        primary_idx = cls._select_primary_snapshot_index(snapshots)
        if primary_idx is None:
            return None
        if primary_idx < 0 or primary_idx >= len(snapshots):
            return None
        return snapshots[primary_idx]

    @staticmethod
    def _compute_obstacle_risk(longitudinal: float, lateral: float, planar_distance: float) -> float:
        if not np.isfinite(planar_distance) or planar_distance <= 0.0:
            return 0.0
        if longitudinal < -0.4:
            return 0.0

        front_gate = float(np.clip((4.0 - max(longitudinal, 0.0)) / 3.5, 0.0, 1.0))
        planar_gate = float(np.clip((5.0 - planar_distance) / 4.5, 0.0, 1.0))
        lateral_gate = float(np.clip(1.0 - abs(lateral) / 1.25, 0.0, 1.0))
        risk = (front_gate ** 2.0) * max(0.20, lateral_gate) * planar_gate
        return float(np.clip(risk, 0.0, 1.0))


class ScenarioObstacleWrapper(gym.Wrapper):
    """Gym wrapper that synchronizes the active learner with obstacle runtime."""

    def __init__(self, env, runtime: ObstacleRuntimeManager):
        super().__init__(env)
        self.runtime = runtime

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self.runtime.on_episode_reset(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info = self.runtime.enrich_info(info)
        return obs, reward, done, info
