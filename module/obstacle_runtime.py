"""
module/obstacle_runtime.py

V16 obstacle runtime:
- manages obstacle fleet lifecycle across scene switches
- injects obstacle-aware info before reward wrapper
- optionally randomizes learner spawn with the same geometry source
"""

from __future__ import annotations

import math
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
    obstacle_free_prob: float = 0.15
    obstacle_modes: Tuple[str, ...] = ("static", "jitter")
    ws_obstacle_free_prob: Optional[float] = None
    min_obstacle_separation_world: float = 3.0
    spawn_ahead_min_m: float = 3.5
    spawn_ahead_max_m: float = 14.0
    min_agent_planar_dist_m: float = 1.5
    min_agent_arc_dist_m: float = 3.5
    lateral_choices: Tuple[float, ...] = (0.35, 0.50, 0.65)
    fixed_progress_ratio: Optional[float] = None
    obstacle_progress_min: Optional[float] = None
    obstacle_progress_max: Optional[float] = None
    fixed_lateral_ratio: Optional[float] = None
    gt_obstacle_start_exclusion_half_width_m: Optional[float] = None
    ws_obstacle_modes: Optional[Tuple[str, ...]] = None
    ws_obstacle_fixed_progress_ratio: Optional[float] = None
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
        self._last_agent_info: Dict[str, Any] = {}
        self._last_runtime_error: str = ""

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

        obs = np.asarray(initial_obs)
        info = self._observe_info_only() or {}

        if self._scene_supports_runtime():
            try:
                if self.config.ego_random_spawn:
                    obs, info = self._randomize_ego_spawn(obs, info)
                self._ensure_fleet()
                if self._fleet is not None and self.rng.random() >= float(self._scene_obstacle_free_prob()):
                    self._refresh_obstacle_layout(agent_info=info)
                    self._active_this_episode = True
                else:
                    self._park_fleet()
            except Exception as exc:
                self._last_runtime_error = f"{type(exc).__name__}: {exc}"
                print(
                    f"⚠️  obstacle runtime reset failed [{self._logging_key or self._scene_key}]: "
                    f"{self._last_runtime_error}"
                )
                self._active_this_episode = False
                self._park_fleet()
        else:
            self._park_fleet()

        refreshed = self._observe_info_and_obs()
        if refreshed is not None:
            obs, info = refreshed
        self._last_agent_info = _copy_info(info)
        return np.asarray(obs)

    def enrich_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        info = dict(info)
        self._last_agent_info = _copy_info(info)
        info["obstacle_present"] = 0.0
        info["obstacle_count"] = 0.0
        info["obstacle_runtime_active"] = 1.0 if self._active_this_episode else 0.0

        if not self._active_this_episode or self._fleet is None:
            return info

        try:
            snapshots = self._fleet.get_snapshots(agent_info=info)
            primary = self._select_primary_snapshot(snapshots)
            if primary is None or primary.relative is None:
                return info

            relative = primary.relative
            risk = self._compute_obstacle_risk(relative.longitudinal, relative.lateral, relative.planar_distance)
            info["obstacle_present"] = 1.0
            info["obstacle_count"] = float(len([snap for snap in snapshots if snap.obstacle is not None]))
            info["obstacle_longitudinal"] = float(relative.longitudinal)
            info["obstacle_lateral"] = float(relative.lateral)
            info["obstacle_source"] = "runtime"
            reward_relevant = bool(float(relative.longitudinal) >= -0.4 or float(relative.planar_distance) <= 1.0)
            if reward_relevant:
                info["obstacle_dist"] = float(relative.planar_distance)
                info["obstacle_risk"] = float(risk)
        except Exception as exc:
            self._last_runtime_error = f"{type(exc).__name__}: {exc}"
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
        # 新 client 连入 DonkeySim 时会先在默认起点短暂出现一帧；
        # 这里立刻送去 staging，避免在正式 episode 放置前留在起点闪烁。
        self._park_fleet()

    def _park_fleet(self) -> None:
        if self._fleet is None:
            return
        preset = self._fleet.preset
        for idx, car in enumerate(self._fleet.cars):
            try:
                staging_x = float(preset.staging_x_start - idx * preset.staging_x_step)
                car.stop_motion(hold_brake=True)
                car.teleport_pose(
                    x=staging_x,
                    z=float(preset.staging_z),
                    yaw_deg=0.0,
                    hold_brake=True,
                )
            except Exception:
                pass

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

    def _refresh_obstacle_layout(self, agent_info: Dict[str, Any]) -> None:
        if self._fleet is None:
            return
        active_count = min(len(self._fleet.cars), self._active_obstacle_count_for_scene())
        active_modes = self._select_episode_modes(
            active_count=active_count,
            active_modes=self._active_obstacle_modes_for_scene(),
        )
        obstacle_radius, safety_margin = self._effective_spawn_clearance()
        targets = self._sample_episode_targets(agent_info=agent_info, count=active_count)
        if not targets:
            self._park_fleet()
            return

        for idx, car in enumerate(self._fleet.cars):
            if idx >= active_count:
                try:
                    car.stop_motion(hold_brake=True)
                    preset = self._fleet.preset
                    staging_x = float(preset.staging_x_start - idx * preset.staging_x_step)
                    car.teleport_pose(
                        x=staging_x,
                        z=float(preset.staging_z),
                        yaw_deg=0.0,
                        hold_brake=True,
                    )
                except Exception:
                    pass
                continue

            target = targets[idx]
            mode = active_modes[idx].strip().lower()
            yaw_override = None
            if mode != "lane_pid" and self.config.randomize_non_lane_pid_yaw:
                yaw_override = self._sample_random_obstacle_yaw_deg()
            if mode == "static":
                car.place_track_target(
                    progress_ratio=target.progress_ratio,
                    lateral_ratio=target.lateral_ratio,
                    yaw_deg_override=yaw_override,
                    obstacle_radius=obstacle_radius,
                    safety_margin=safety_margin,
                    hold_brake=True,
                    timeout_s=self.config.placement_timeout_s,
                )
                car.stop_motion(hold_brake=True)
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
                car.start_lane_pid(
                    target_speed=self._lane_pid_speed_for_scene(),
                    progress_ratio=target.progress_ratio,
                    lateral_ratio=target.lateral_ratio,
                    lookahead_m=self.config.lane_pid_lookahead_m,
                    place_on_start=True,
                    obstacle_radius=obstacle_radius,
                    safety_margin=safety_margin,
                )
            else:
                raise ValueError(f"Unsupported obstacle mode: {mode}")

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
        if fixed_progress_ratio is not None:
            return self._sample_fixed_episode_targets(
                agent_pose=agent_pose,
                count=count,
                g=g,
                lane_choices=lane_choices,
                obstacle_radius=obstacle_radius,
                safety_margin=safety_margin,
                fixed_progress_ratio=fixed_progress_ratio,
                fixed_lateral_ratio=fixed_lateral_ratio,
            )

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

        progress_step = max(float(self.config.min_agent_arc_dist_m), 0.5) / max(float(g.loop_len), 1e-6)
        targets: List[TrackTarget] = []
        for idx in range(int(count)):
            base_candidate_progress = _wrap_progress(base_progress + idx * progress_step)
            candidate_progresses = [
                base_candidate_progress,
                _wrap_progress(base_candidate_progress + progress_step),
                _wrap_progress(base_candidate_progress - progress_step),
                _wrap_progress(base_candidate_progress + 2.0 * progress_step),
                _wrap_progress(base_candidate_progress - 2.0 * progress_step),
            ]
            chosen: Optional[TrackTarget] = None
            for progress_ratio in candidate_progresses:
                candidate = sample_track_target(
                    track_geometry=self.track_geometry,
                    scene_key=self._scene_key,
                    progress_ratio=progress_ratio,
                    lateral_ratio=float(base_laterals[idx]),
                    obstacle_radius=obstacle_radius,
                    safety_margin=safety_margin,
                )
                if self._target_is_valid(candidate, agent_pose, targets, g):
                    chosen = candidate
                    break
            if chosen is None:
                chosen = sample_track_target(
                    track_geometry=self.track_geometry,
                    scene_key=self._scene_key,
                    progress_ratio=base_candidate_progress,
                    lateral_ratio=float(base_laterals[idx]),
                    obstacle_radius=obstacle_radius,
                    safety_margin=safety_margin,
                )
            targets.append(chosen)
        return targets

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
        if self.config.randomize_non_lane_pid_yaw and any(mode != "lane_pid" for mode in active_modes):
            safety_margin += 0.05

        return obstacle_radius, float(safety_margin)

    def _sample_random_obstacle_yaw_deg(self) -> float:
        return float(self.rng.uniform(0.0, 360.0))

    def _lane_choices_for_scene(self) -> Tuple[float, ...]:
        if self._scene_key == "waveshare":
            # WS 障碍固定只在左右两条边线出生；sample_track_target 会再按
            # 障碍半径和安全边距夹紧到“最靠边且合法”的两条线。
            return (0.0, 1.0)
        if self._fleet is not None:
            preset = self._fleet.preset
            default_choices = tuple(sorted({float(r) for _p, r in preset.default_layout} | {0.5}))
            if default_choices:
                return default_choices
        return tuple(sorted({float(x) for x in self.config.lateral_choices}))

    def _active_obstacle_count_for_scene(self) -> int:
        if self._scene_key == "waveshare":
            return 1
        return int(max(1, self.config.obstacle_count))

    def _scene_obstacle_free_prob(self) -> float:
        if self._scene_key == "waveshare" and self.config.ws_obstacle_free_prob is not None:
            return float(np.clip(self.config.ws_obstacle_free_prob, 0.0, 1.0))
        return float(np.clip(self.config.obstacle_free_prob, 0.0, 1.0))

    def _scene_fixed_progress_ratio(self) -> Optional[float]:
        if self._scene_key == "waveshare":
            # 如果设置了范围，从范围内随机选择
            if (self.config.ws_obstacle_progress_min is not None and
                self.config.ws_obstacle_progress_max is not None):
                min_p = float(self.config.ws_obstacle_progress_min)
                max_p = float(self.config.ws_obstacle_progress_max)
                return float(self.rng.uniform(min_p, max_p))
            # 否则使用固定值
            if self.config.ws_obstacle_fixed_progress_ratio is not None:
                return float(self.config.ws_obstacle_fixed_progress_ratio)
        return self.config.fixed_progress_ratio

    def _gt_progress_ratio_bounds(self) -> Optional[Tuple[float, float]]:
        if self._scene_key != "generated_track":
            return None
        min_p = self.config.obstacle_progress_min
        max_p = self.config.obstacle_progress_max
        if min_p is None or max_p is None:
            return None
        min_p = float(np.clip(min_p, 0.0, 1.0))
        max_p = float(np.clip(max_p, 0.0, 1.0))
        if max_p < min_p:
            min_p, max_p = max_p, min_p
        return float(min_p), float(max_p)

    def _scene_fixed_lateral_ratio(self) -> Optional[float]:
        if self._scene_key == "waveshare" and self.config.ws_obstacle_fixed_lateral_ratio is not None:
            return float(self.config.ws_obstacle_fixed_lateral_ratio)
        return self.config.fixed_lateral_ratio

    def _active_obstacle_modes_for_scene(self) -> Tuple[str, ...]:
        if self._scene_key == "waveshare":
            if self.config.ws_obstacle_modes:
                return tuple(self.config.ws_obstacle_modes)
            return ("static",)
        return tuple(self.config.obstacle_modes) or ("static",)

    def _sample_ahead_progress(self, agent_progress: float, g: SceneGeometry) -> float:
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
        progress_bounds = self._gt_progress_ratio_bounds()
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
        progress_bounds = self._gt_progress_ratio_bounds()
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
            if planar < float(self.config.min_agent_planar_dist_m):
                return False
            if arc_gap < float(self.config.min_agent_arc_dist_m):
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
    def _select_primary_snapshot(snapshots: Sequence[ObstacleSnapshot]) -> Optional[ObstacleSnapshot]:
        candidates: List[Tuple[float, ObstacleSnapshot]] = []
        fallback: List[Tuple[float, ObstacleSnapshot]] = []
        for snap in snapshots:
            rel = snap.relative
            if rel is None:
                continue
            planar = float(rel.planar_distance)
            fallback.append((planar, snap))
            if float(rel.longitudinal) >= -0.4:
                score = planar + 0.15 * abs(float(rel.lateral))
                candidates.append((score, snap))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1]
        if fallback:
            fallback.sort(key=lambda item: item[0])
            return fallback[0][1]
        return None

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
