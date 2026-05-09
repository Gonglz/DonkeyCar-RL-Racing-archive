"""
module/obstacle.py

使用第二个 DonkeySim client 生成一辆额外的 donkey 车，作为动态/静态障碍车。

当前 DonkeySim 底层已验证支持两个隐藏消息：
- `{"msg_type": "set_position", "pos_x", "pos_y", "pos_z", "Qx", "Qy", "Qz", "Qw"}`
- `{"msg_type": "node_position", "index": "..."}`

注意坐标系：
- `track.py` / telemetry / `info["pos"]` 使用的是 Python 侧赛道坐标；
- Unity 内部 world 坐标在网络层额外放大了 8 倍；
- 因此从 Python 侧定点放置障碍车时，需要 `x/z * 8` 后再发给 `set_position`。
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .track import MODULE_TRACK_DATA_DIR, SceneGeometry, TrackGeometryManager


def _wrap_pi(x: float) -> float:
    return float((float(x) + math.pi) % (2.0 * math.pi) - math.pi)


def _clip_float(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), float(lo)), float(hi)))


def _wrap_deg(x: float) -> float:
    return float((float(x) + 180.0) % 360.0 - 180.0)


def track_heading_deg_to_telemetry_yaw_deg(track_heading_deg: float) -> float:
    """赛道切线方向(数学坐标系) -> Donkey telemetry yaw。"""
    return _wrap_deg(90.0 - float(track_heading_deg))


def telemetry_yaw_deg_to_track_heading_deg(yaw_deg: float) -> float:
    """Donkey telemetry yaw -> 赛道/几何模块使用的数学朝向角。"""
    return _wrap_deg(90.0 - float(yaw_deg))


def _obstacle_episode_over_disabled(handler) -> None:
    """障碍车 client 不参与 RL episode 终止，避免离屏 staging 触发 reset。"""
    return None


_ENV_TO_SCENE_KEY: Dict[str, str] = {
    "donkey-generated-track-v0": "generated_track",
    "donkey-generated-roads-v0": "generated_track",
    "donkey-waveshare-v0": "waveshare",
    "donkey-warehouse-v0": "warehouse",
    "donkey-mountain-track-v0": "mountain_track",
    "donkey-minimonaco-track-v0": "mini_monaco",
    "donkey-roboracingleague-track-v0": "roboracingleague_track",
    "donkey-avc-sparkfun-v0": "avc_sparkfun",
    "donkey-warren-track-v0": "warren_track",
    "donkey-circuit-launch-track-v0": "circuit_launch",
}

_UNITY_WORLD_SCALE = 8.0
_DEFAULT_WORLD_Y = 0.5
_DEFAULT_TRACK_PROFILE_DIR = MODULE_TRACK_DATA_DIR
# Fixed obstacle color aligned to the real obstacle-car appearance.
_FIXED_OBSTACLE_BODY_RGB: Tuple[int, int, int] = (255, 105, 180)
_DEFAULT_OBSTACLE_BODY_RGBS: Tuple[Tuple[int, int, int], ...] = (
    _FIXED_OBSTACLE_BODY_RGB,
)


@dataclass(frozen=True)
class ObstacleFleetPreset:
    name: str
    env_id: str
    scene_key: str
    track_file: str
    default_layout: Tuple[Tuple[float, float], ...]
    staging_x_start: float
    staging_z: float
    staging_x_step: float
    default_count: int = 2
    min_separation_world: float = 3.0
    obstacle_radius: float = 0.20
    safety_margin: float = 0.08


_OBSTACLE_FLEET_PRESETS: Dict[str, ObstacleFleetPreset] = {
    "gt": ObstacleFleetPreset(
        name="gt",
        env_id="donkey-generated-track-v0",
        scene_key="generated_track",
        track_file="manual_width_generated_track.json",
        default_layout=((0.06, 0.35), (0.12, 0.65)),
        staging_x_start=-24.0,
        staging_z=-20.0,
        staging_x_step=2.0,
    ),
    "ws": ObstacleFleetPreset(
        name="ws",
        env_id="donkey-waveshare-v0",
        scene_key="waveshare",
        track_file="manual_width_waveshare.json",
        default_layout=((0.08, 0.35), (0.20, 0.65)),
        staging_x_start=-10.0,
        staging_z=-6.0,
        staging_x_step=1.2,
        obstacle_radius=0.18,
        safety_margin=0.05,
    ),
}

_OBSTACLE_FLEET_ALIASES: Dict[str, str] = {
    "gt": "gt",
    "generated_track": "gt",
    "donkey-generated-track-v0": "gt",
    "ws": "ws",
    "waveshare": "ws",
    "donkey-waveshare-v0": "ws",
}


def telemetry_to_unity_world(x: float, y: float, z: float, scale: float = _UNITY_WORLD_SCALE) -> Tuple[float, float, float]:
    return float(x) * scale, float(y) * scale, float(z) * scale


def unity_world_to_telemetry(x: float, y: float, z: float, scale: float = _UNITY_WORLD_SCALE) -> Tuple[float, float, float]:
    inv = 1.0 / max(float(scale), 1e-6)
    return float(x) * inv, float(y) * inv, float(z) * inv


def yaw_deg_to_unity_quaternion(yaw_deg: float) -> Tuple[float, float, float, float]:
    half = 0.5 * math.radians(float(yaw_deg))
    return 0.0, float(math.sin(half)), 0.0, float(math.cos(half))


@dataclass
class PoseState:
    x: float
    y: float
    z: float
    yaw_deg: float
    speed: float
    cte: float
    hit: str
    track_idx: Optional[int] = None
    progress_ratio: Optional[float] = None
    lat_err: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RelativeState:
    dx: float
    dy: float
    dz: float
    planar_distance: float
    longitudinal: float
    lateral: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrackTarget:
    scene_key: str
    track_idx: int
    progress_ratio: float
    lateral_ratio: float
    x: float
    y: float
    z: float
    yaw_deg: float
    width: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _with_target_yaw(target: TrackTarget, yaw_deg: float) -> TrackTarget:
    return replace(target, yaw_deg=float(_wrap_deg(yaw_deg)))


@dataclass(frozen=True)
class PositionJitterConfig:
    anchor: TrackTarget
    amplitude_m: float = 0.10
    period_s: float = 1.5
    update_hz: float = 8.0
    start_time: float = 0.0


@dataclass(frozen=True)
class InPlaceNudgeConfig:
    anchor: TrackTarget
    amplitude_m: float = 0.14
    period_s: float = 1.5
    update_hz: float = 8.0
    start_time: float = 0.0


@dataclass(frozen=True)
class LanePIDConfig:
    target_speed: float
    lateral_ratio: float = 0.5
    lookahead_m: float = 0.9
    pure_pursuit_gain: float = 1.0
    lookahead_speed_gain: float = 0.8
    recovery_steer_gain: float = 1.1
    reverse_steer_gain: float = 0.9
    speed_kp: float = 0.90
    speed_ki: float = 0.18
    speed_kd: float = 0.02
    max_throttle: float = 0.32
    min_throttle: float = 0.06
    throttle_steer_damp: float = 0.35


@dataclass
class LanePIDDebugState:
    active: float = 0.0
    target_speed: float = 0.0
    speed: float = 0.0
    speed_error: float = 0.0
    effective_lookahead: float = 0.0
    local_forward: float = 0.0
    local_left: float = 0.0
    lookahead_distance: float = 0.0
    lat_err_norm: float = 0.0
    steer: float = 0.0
    throttle: float = 0.0
    reverse_mode: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "active": float(self.active),
            "target_speed": float(self.target_speed),
            "speed": float(self.speed),
            "speed_error": float(self.speed_error),
            "effective_lookahead": float(self.effective_lookahead),
            "local_forward": float(self.local_forward),
            "local_left": float(self.local_left),
            "lookahead_distance": float(self.lookahead_distance),
            "lat_err_norm": float(self.lat_err_norm),
            "steer": float(self.steer),
            "throttle": float(self.throttle),
            "reverse_mode": float(self.reverse_mode),
        }


@dataclass
class ObstacleSnapshot:
    obstacle: Optional[PoseState]
    target: Optional[TrackTarget]
    agent: Optional[PoseState]
    relative: Optional[RelativeState]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "obstacle": None if self.obstacle is None else self.obstacle.as_dict(),
            "target": None if self.target is None else self.target.as_dict(),
            "agent": None if self.agent is None else self.agent.as_dict(),
            "relative": None if self.relative is None else self.relative.as_dict(),
        }


class _PIDController:
    def __init__(
        self,
        kp: float = 0.0,
        ki: float = 0.0,
        kd: float = 0.0,
        integral_limit: float = 1.0,
        output_limits: Tuple[float, float] = (-1.0, 1.0),
    ):
        self.configure(kp=kp, ki=ki, kd=kd, integral_limit=integral_limit, output_limits=output_limits)
        self.reset()

    def configure(
        self,
        kp: float,
        ki: float,
        kd: float,
        integral_limit: float,
        output_limits: Tuple[float, float],
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.integral_limit = float(max(integral_limit, 0.0))
        lo, hi = output_limits
        self.output_limits = (float(lo), float(hi))

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error: Optional[float] = None
        self._prev_t: Optional[float] = None

    def step(self, error: float, now: float) -> float:
        error = float(error)
        now = float(now)
        dt = 0.0 if self._prev_t is None else max(now - self._prev_t, 1e-3)
        if dt > 0.0:
            self._integral = _clip_float(
                self._integral + error * dt,
                -self.integral_limit,
                self.integral_limit,
            )
            derivative = 0.0 if self._prev_error is None else (error - self._prev_error) / dt
        else:
            derivative = 0.0
        out = self.kp * error + self.ki * self._integral + self.kd * derivative
        self._prev_error = error
        self._prev_t = now
        return _clip_float(out, self.output_limits[0], self.output_limits[1])


class DonkeyObstacleFleet:
    """管理一组静态障碍车（目前仅提供 gt / ws 两种 preset）。"""

    def __init__(
        self,
        preset: ObstacleFleetPreset,
        track_geometry: TrackGeometryManager,
        cars: Sequence["DonkeyObstacleCar"],
        targets: Sequence[TrackTarget],
    ):
        self.preset = preset
        self.track_geometry = track_geometry
        self.cars = list(cars)
        self.targets = list(targets)

    def shutdown(self) -> None:
        for car in reversed(self.cars):
            try:
                car.shutdown()
            except Exception:
                pass

    def get_obstacle_poses(self) -> List[Optional[PoseState]]:
        return [car.get_obstacle_pose() for car in self.cars]

    def obstacle_coordinates(self) -> List[Optional[Tuple[float, float, float]]]:
        return [car.obstacle_coordinates() for car in self.cars]

    def get_snapshots(self, agent_info: Optional[Dict[str, Any]] = None) -> List[ObstacleSnapshot]:
        return [car.get_snapshot(agent_info=agent_info) for car in self.cars]

    def last_errors(self) -> List[Optional[str]]:
        return [car.last_error() for car in self.cars]


def infer_scene_key(env_id: str) -> Optional[str]:
    """根据 gym env_id 推断 scene_key。"""
    return _ENV_TO_SCENE_KEY.get(str(env_id))


def resolve_obstacle_fleet_preset(scene: str) -> ObstacleFleetPreset:
    """仅支持 gt / ws 两类障碍车 preset。"""
    key = _OBSTACLE_FLEET_ALIASES.get(str(scene).strip().lower())
    if key is None or key not in _OBSTACLE_FLEET_PRESETS:
        raise KeyError(f"Unsupported obstacle fleet scene: {scene!r}. Expected one of: gt, ws")
    return _OBSTACLE_FLEET_PRESETS[key]


def build_obstacle_track_geometry(scene: str, track_dir: Optional[str] = None) -> TrackGeometryManager:
    preset = resolve_obstacle_fleet_preset(scene)
    track_dir = str(track_dir or _DEFAULT_TRACK_PROFILE_DIR)
    scene_specs = {
        preset.env_id: {
            "scene_key": preset.scene_key,
            "track_file": preset.track_file,
        }
    }
    return TrackGeometryManager(
        track_dir=track_dir,
        env_ids=[preset.env_id],
        scene_specs=scene_specs,
    )


def default_obstacle_layout(scene: str) -> Tuple[Tuple[float, float], ...]:
    return tuple(resolve_obstacle_fleet_preset(scene).default_layout)


def _copy_info(info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not info:
        return {}
    out = dict(info)
    for key in ("pos", "gyro", "accel", "vel", "car", "lidar"):
        if key in out:
            try:
                out[key] = copy.deepcopy(out[key])
            except Exception:
                pass
    return out


def _extract_pos(info: Dict[str, Any]) -> Tuple[float, float, float]:
    pos = info.get("pos", (0.0, 0.0, 0.0))
    try:
        return float(pos[0]), float(pos[1]), float(pos[2])
    except Exception:
        return 0.0, 0.0, 0.0


def _extract_yaw_deg(info: Dict[str, Any]) -> float:
    car = info.get("car", (0.0, 0.0, 0.0))
    try:
        return float(car[2])
    except Exception:
        return 0.0


def _extract_speed(info: Dict[str, Any]) -> float:
    try:
        return float(info.get("speed", 0.0) or 0.0)
    except Exception:
        return 0.0


def _extract_cte(info: Dict[str, Any]) -> float:
    try:
        return float(info.get("cte", 0.0) or 0.0)
    except Exception:
        return 0.0


def _extract_hit(info: Dict[str, Any]) -> str:
    try:
        return str(info.get("hit", "none") or "none")
    except Exception:
        return "none"


def pose_from_info(
    info: Optional[Dict[str, Any]],
    track_geometry: Optional[TrackGeometryManager] = None,
    scene_key: Optional[str] = None,
    prev_idx: Optional[int] = None,
) -> Optional[PoseState]:
    """从 DonkeySim telemetry info 提取位姿；若提供赛道几何，则附带 track idx / progress。"""
    if not info:
        return None

    x, y, z = _extract_pos(info)
    yaw_deg = _extract_yaw_deg(info)
    speed = _extract_speed(info)
    cte = _extract_cte(info)
    hit = _extract_hit(info)

    track_idx = None
    progress_ratio = None
    lat_err = None
    if track_geometry is not None and scene_key:
        try:
            geo = track_geometry.query(
                scene_key,
                x=x,
                z=z,
                yaw_rad=math.radians(telemetry_yaw_deg_to_track_heading_deg(yaw_deg)),
                prev_idx=prev_idx,
            )
            track_idx = int(geo["idx"])
            lat_err = float(geo["lat_err"])
            g = track_geometry.scenes[scene_key]
            progress_ratio = float(g.cum_len[track_idx] / max(g.loop_len, 1e-6))
        except Exception:
            track_idx = None
            progress_ratio = None
            lat_err = None

    return PoseState(
        x=float(x),
        y=float(y),
        z=float(z),
        yaw_deg=float(yaw_deg),
        speed=float(speed),
        cte=float(cte),
        hit=hit,
        track_idx=track_idx,
        progress_ratio=progress_ratio,
        lat_err=lat_err,
    )


def compute_relative_state(agent: Optional[PoseState], obstacle: Optional[PoseState]) -> Optional[RelativeState]:
    """返回障碍车相对于 agent 的位姿差。"""
    if agent is None or obstacle is None:
        return None

    dx = float(obstacle.x - agent.x)
    dy = float(obstacle.y - agent.y)
    dz = float(obstacle.z - agent.z)
    planar = float(math.hypot(dx, dz))

    yaw_rad = math.radians(telemetry_yaw_deg_to_track_heading_deg(float(agent.yaw_deg)))
    fx = math.cos(yaw_rad)
    fz = math.sin(yaw_rad)
    lx = -math.sin(yaw_rad)
    lz = math.cos(yaw_rad)

    longitudinal = float(dx * fx + dz * fz)
    lateral = float(dx * lx + dz * lz)

    return RelativeState(
        dx=dx,
        dy=dy,
        dz=dz,
        planar_distance=planar,
        longitudinal=longitudinal,
        lateral=lateral,
    )


def _segment_pose_at_progress(g: SceneGeometry, progress_ratio: float) -> Tuple[int, float, np.ndarray, np.ndarray, np.ndarray, float]:
    progress = float(progress_ratio % 1.0)
    s = progress * float(g.loop_len)
    idx = int(np.searchsorted(g.cum_len, s, side="right") - 1)
    idx = int(np.clip(idx, 0, g.center.shape[0] - 1))
    nxt = (idx + 1) % g.center.shape[0]

    seg_s = float(g.cum_len[idx])
    seg_len = float(max(g.seg_len[idx], 1e-6))
    t = float(np.clip((s - seg_s) / seg_len, 0.0, 1.0))

    center = (1.0 - t) * g.center[idx] + t * g.center[nxt]
    left = (1.0 - t) * g.left[idx] + t * g.left[nxt]
    right = (1.0 - t) * g.right[idx] + t * g.right[nxt]
    tangent = (1.0 - t) * g.tangent[idx] + t * g.tangent[nxt]
    tangent_norm = float(max(np.linalg.norm(tangent), 1e-6))
    tangent = tangent / tangent_norm
    width = float(max(np.linalg.norm(left - right), 1e-6))

    return idx, t, center, left, right, width


def sample_track_target(
    track_geometry: TrackGeometryManager,
    scene_key: str,
    progress_ratio: float,
    lateral_ratio: float = 0.5,
    y: float = 0.0,
    obstacle_radius: float = 0.25,
    safety_margin: float = 0.05,
) -> TrackTarget:
    """
    在赛道截面上采样一个目标点。

    `lateral_ratio=0` 为右边界，`1` 为左边界，0.5 为中线附近。
    """
    if scene_key not in track_geometry.scenes:
        raise KeyError("Unknown scene_key for obstacle target: %s" % scene_key)

    g = track_geometry.scenes[scene_key]
    idx, _t, center, left, right, width = _segment_pose_at_progress(g, progress_ratio)

    usable_margin = float(max(obstacle_radius + safety_margin, 0.0))
    margin_ratio = float(np.clip(usable_margin / max(width, 1e-6), 0.0, 0.49))
    lateral = float(np.clip(lateral_ratio, margin_ratio, 1.0 - margin_ratio))

    point = (1.0 - lateral) * right + lateral * left

    tangent = g.tangent[idx]
    track_heading_deg = float(math.degrees(math.atan2(float(tangent[1]), float(tangent[0]))))
    yaw_deg = track_heading_deg_to_telemetry_yaw_deg(track_heading_deg)

    return TrackTarget(
        scene_key=scene_key,
        track_idx=int(idx),
        progress_ratio=float(progress_ratio % 1.0),
        lateral_ratio=float(lateral),
        x=float(point[0]),
        y=float(y),
        z=float(point[1]),
        yaw_deg=yaw_deg,
        width=width,
    )


def sample_random_track_targets(
    track_geometry: TrackGeometryManager,
    scene_key: str,
    count: int = 2,
    y: float = 0.0,
    obstacle_radius: float = 0.25,
    safety_margin: float = 0.05,
    min_separation_world: float = 3.0,
    rng: Optional[np.random.Generator] = None,
    max_attempts: int = 512,
) -> List[TrackTarget]:
    """
    在整条赛道范围内随机采样多个障碍车目标点。

    `min_separation_world` 使用 Unity / sim world 坐标；默认 `3.0` 约等于一个车身长度。
    """
    if int(count) <= 0:
        raise ValueError("count must be positive")

    min_separation = max(float(min_separation_world), 0.0) / max(float(_UNITY_WORLD_SCALE), 1e-6)
    rng = np.random.default_rng() if rng is None else rng

    targets: List[TrackTarget] = []
    attempts = 0
    while len(targets) < int(count) and attempts < int(max_attempts):
        attempts += 1
        candidate = sample_track_target(
            track_geometry=track_geometry,
            scene_key=scene_key,
            progress_ratio=float(rng.random()),
            lateral_ratio=float(rng.random()),
            y=y,
            obstacle_radius=obstacle_radius,
            safety_margin=safety_margin,
        )
        if all(math.hypot(candidate.x - other.x, candidate.z - other.z) >= min_separation for other in targets):
            targets.append(candidate)

    if len(targets) != int(count):
        raise RuntimeError(
            "Failed to sample %d obstacle targets with min separation %.3f world units"
            % (int(count), float(min_separation_world))
        )
    return targets


class DonkeyObstacleCar:
    """
    额外的 DonkeySim client，用一辆可见/可撞的 donkey 车充当障碍车。

    推荐流程：
    1. `spawn()` 连接到同一个 sim；
    2. `set_track_target(...)` 规划赛道内目标位姿；
    3. 每个 agent step 调用 `update(agent_info)` 获取双方位置快照。
    """

    def __init__(
        self,
        env_id: str,
        track_geometry: Optional[TrackGeometryManager] = None,
        scene_key: Optional[str] = None,
        sim_path: str = "remote",
        host: str = "127.0.0.1",
        port: int = 9091,
        conf: Optional[Dict[str, Any]] = None,
        body_style: str = "donkey",
        body_rgb: Tuple[int, int, int] = _FIXED_OBSTACLE_BODY_RGB,
        car_name: str = "obstacle_donkey",
        racer_name: str = "Obstacle",
        country: str = "CN",
        bio: str = "Obstacle donkey car",
        guid: Optional[str] = None,
        max_cte: float = 8.0,
        cruise_throttle: float = 0.22,
        crawl_throttle: float = 0.10,
        slow_distance: float = 1.25,
        stop_distance: float = 0.35,
        approach_distance: float = 2.0,
        k_lat: float = 0.45,
        k_heading: float = 0.90,
        k_target_heading: float = 1.10,
        auto_reset_on_done: bool = True,
        unity_world_scale: float = _UNITY_WORLD_SCALE,
        default_world_y: float = _DEFAULT_WORLD_Y,
        placement_timeout_s: float = 1.0,
    ):
        self.env_id = str(env_id)
        self.track_geometry = track_geometry
        self.scene_key = scene_key or infer_scene_key(self.env_id)

        self.sim_path = str(sim_path)
        self.host = str(host)
        self.port = int(port)
        self.max_cte = float(max_cte)
        self.auto_reset_on_done = bool(auto_reset_on_done)

        self.cruise_throttle = float(max(0.0, cruise_throttle))
        self.crawl_throttle = float(max(0.0, crawl_throttle))
        self.slow_distance = float(max(0.1, slow_distance))
        self.stop_distance = float(max(0.05, stop_distance))
        self.approach_distance = float(max(self.stop_distance, approach_distance))
        self.k_lat = float(k_lat)
        self.k_heading = float(k_heading)
        self.k_target_heading = float(k_target_heading)
        self.unity_world_scale = float(max(unity_world_scale, 1e-6))
        self.default_world_y = float(default_world_y)
        self.placement_timeout_s = float(max(placement_timeout_s, 0.0))

        self.conf = dict(conf or {})
        self.conf.update(
            {
                "host": self.host,
                "port": self.port,
                "body_style": body_style,
                "body_rgb": tuple(int(v) for v in body_rgb),
                "car_name": str(car_name),
                "font_size": max(80, int(self.conf.get("font_size", 60))),
                "racer_name": str(racer_name),
                "country": str(country),
                "bio": str(bio),
                "guid": str(guid or ("obstacle-" + uuid.uuid4().hex[:12])),
                "max_cte": self.max_cte,
                # 障碍车不需要高分辨率图像，缩小带宽占用。
                "cam_resolution": tuple(self.conf.get("cam_resolution", (32, 32, 3))),
            }
        )
        self.conf.setdefault(
            "cam_config",
            {"img_w": 32, "img_h": 32, "img_d": 3},
        )

        self._env = None
        self._thread = None
        self._stop_evt = threading.Event()
        self._reset_evt = threading.Event()
        self._lock = threading.Lock()
        self._last_info: Dict[str, Any] = {}
        self._last_info_t: float = 0.0
        self._last_error: Optional[str] = None
        self._last_set_position_msg: Optional[Dict[str, Any]] = None
        self._target: Optional[TrackTarget] = None
        self._manual_action = np.zeros((2,), dtype=np.float32)
        self._use_autopilot = False
        self._hold_brake = False
        self._last_track_idx: Optional[int] = None
        self._agent_info: Optional[Dict[str, Any]] = None
        self._node_position_evt = threading.Event()
        self._node_position_resp: Optional[Dict[str, Any]] = None
        self._reset_reason: str = ""
        self._jitter_cfg: Optional[PositionJitterConfig] = None
        self._jitter_next_update_t: float = 0.0
        self._nudge_cfg: Optional[InPlaceNudgeConfig] = None
        self._nudge_next_update_t: float = 0.0
        self._lane_pid_cfg: Optional[LanePIDConfig] = None
        self._lane_pid_debug = LanePIDDebugState()
        self._lane_speed_pid = _PIDController(
            output_limits=(0.0, 1.0),
            integral_limit=3.0,
        )
        self._lane_steer_pid = _PIDController(
            output_limits=(-1.0, 1.0),
            integral_limit=2.0,
        )
        self._debug_event_seq = 0
        self._last_debug_error_sig: Optional[str] = None
        self._last_debug_error_t = 0.0

    @staticmethod
    def _import_sim_env():
        try:
            import gym  # type: ignore
            import gym_donkeycar  # noqa: F401  # type: ignore
            return gym
        except ImportError:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "donkeycar"))
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            import gym  # type: ignore
            import gym_donkeycar  # noqa: F401  # type: ignore
            return gym

    def spawn(
        self,
        reset_on_spawn: bool = False,
        hidden_pose: Optional[Tuple[float, float, float]] = None,
        hold_brake: bool = True,
    ) -> None:
        """连接到同一个 DonkeySim server，并创建障碍车 client。"""
        if self._thread is not None and self._thread.is_alive():
            return

        gym = self._import_sim_env()
        conf = dict(self.conf)
        if self.sim_path and self.sim_path not in ("", "remote", "none"):
            conf["exe_path"] = self.sim_path
        else:
            conf.pop("exe_path", None)
        self._env = gym.make(self.env_id, conf=conf)
        if hasattr(self._env, "set_episode_over_fn"):
            self._env.set_episode_over_fn(_obstacle_episode_over_disabled)
        self._install_handler_hooks()
        with self._lock:
            self._manual_action = np.zeros((2,), dtype=np.float32)
            self._use_autopilot = False
            self._hold_brake = bool(hold_brake)
        if hidden_pose is not None:
            hidden_x, hidden_z, hidden_yaw_deg = hidden_pose
            self._teleport_raw(
                x=float(hidden_x),
                z=float(hidden_z),
                yaw_deg=float(hidden_yaw_deg),
                world_y=self.default_world_y,
                hold_brake=hold_brake,
            )
        self._stop_evt.clear()
        if reset_on_spawn:
            with self._lock:
                self._reset_reason = "spawn_reset_on_spawn"
            self._reset_evt.set()
        else:
            with self._lock:
                self._reset_reason = ""
            self._reset_evt.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="donkey-obstacle-car",
            daemon=True,
        )
        self._thread.start()

    def shutdown(self) -> None:
        """停止障碍车 client。"""
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
        self._env = None

    def reset(self, reason: str = "manual") -> None:
        """请求障碍车 reset 到该 client 的默认出生点。"""
        with self._lock:
            self._reset_reason = str(reason)
        self._log_debug_event("reset_requested", reason=str(reason))
        self._reset_evt.set()

    def reset_and_wait(self, reason: str = "manual", timeout_s: float = 1.0) -> bool:
        """请求 reset，并等待后台 client 完成一次 env.reset。"""
        self.reset(reason=reason)
        deadline = time.time() + max(0.0, float(timeout_s))
        while time.time() < deadline:
            if not self._reset_evt.is_set():
                return True
            time.sleep(0.02)
        return not self._reset_evt.is_set()

    def set_manual_action(self, steering: float = 0.0, throttle: float = 0.0) -> None:
        with self._lock:
            self._clear_dynamic_modes_locked()
            self._manual_action = np.array(
                [_clip_float(float(steering), -1.0, 1.0), _clip_float(float(throttle), -1.0, 1.0)],
                dtype=np.float32,
            )
            self._use_autopilot = False
            self._hold_brake = False

    def hold_position(self) -> None:
        with self._lock:
            self._clear_dynamic_modes_locked()
            self._manual_action = np.zeros((2,), dtype=np.float32)
            self._use_autopilot = False
            self._hold_brake = True

    def stop_motion(self, hold_brake: bool = True) -> None:
        with self._lock:
            self._clear_dynamic_modes_locked()
            self._manual_action = np.zeros((2,), dtype=np.float32)
            self._use_autopilot = False
            self._hold_brake = bool(hold_brake)

    def motion_mode(self) -> str:
        with self._lock:
            if self._nudge_cfg is not None:
                return "nudge"
            if self._jitter_cfg is not None:
                return "jitter"
            if self._lane_pid_cfg is not None:
                return "lane_pid"
            if self._use_autopilot:
                return "track_target"
            if self._hold_brake and np.allclose(self._manual_action, 0.0):
                return "hold"
            return "manual"

    def get_lane_pid_debug(self) -> Dict[str, float]:
        with self._lock:
            return self._lane_pid_debug.as_dict()

    def start_position_jitter(
        self,
        progress_ratio: Optional[float] = None,
        lateral_ratio: Optional[float] = None,
        amplitude_m: float = 0.10,
        period_s: float = 1.5,
        update_hz: float = 8.0,
        yaw_deg_override: Optional[float] = None,
        y: float = 0.0,
        obstacle_radius: float = 0.25,
        safety_margin: float = 0.05,
    ) -> TrackTarget:
        """沿赛道纵向在锚点附近前后抖动；位置由 `set_position` 直接更新。"""
        anchor = self._resolve_track_target(
            progress_ratio=progress_ratio,
            lateral_ratio=lateral_ratio,
            y=y,
            obstacle_radius=obstacle_radius,
            safety_margin=safety_margin,
        )
        if yaw_deg_override is not None:
            anchor = _with_target_yaw(anchor, float(yaw_deg_override))
        self.place_pose(
            x=anchor.x,
            z=anchor.z,
            yaw_deg=anchor.yaw_deg,
            world_y=self.default_world_y,
            hold_brake=True,
            timeout_s=self.placement_timeout_s,
        )
        with self._lock:
            self._clear_dynamic_modes_locked()
            self._target = anchor
            self._jitter_cfg = PositionJitterConfig(
                anchor=anchor,
                amplitude_m=float(max(amplitude_m, 0.0)),
                period_s=float(max(period_s, 0.2)),
                update_hz=float(max(update_hz, 1.0)),
                start_time=time.time(),
            )
            self._jitter_next_update_t = 0.0
            self._manual_action = np.zeros((2,), dtype=np.float32)
            self._use_autopilot = False
            self._hold_brake = True
        return anchor

    def start_in_place_nudge(
        self,
        progress_ratio: Optional[float] = None,
        lateral_ratio: Optional[float] = None,
        amplitude_m: float = 0.14,
        period_s: float = 1.5,
        update_hz: float = 8.0,
        yaw_deg_override: Optional[float] = None,
        y: float = 0.0,
        obstacle_radius: float = 0.25,
        safety_margin: float = 0.05,
    ) -> TrackTarget:
        """以当前锚点为中心，沿车头方向小幅前后挪动，保持原朝向不变。"""
        anchor = self._resolve_track_target(
            progress_ratio=progress_ratio,
            lateral_ratio=lateral_ratio,
            y=y,
            obstacle_radius=obstacle_radius,
            safety_margin=safety_margin,
        )
        if yaw_deg_override is not None:
            anchor = _with_target_yaw(anchor, float(yaw_deg_override))
        self.place_pose(
            x=anchor.x,
            z=anchor.z,
            yaw_deg=anchor.yaw_deg,
            world_y=self.default_world_y,
            hold_brake=True,
            timeout_s=self.placement_timeout_s,
        )
        with self._lock:
            self._clear_dynamic_modes_locked()
            self._target = anchor
            self._nudge_cfg = InPlaceNudgeConfig(
                anchor=anchor,
                amplitude_m=float(max(amplitude_m, 0.0)),
                period_s=float(max(period_s, 0.2)),
                update_hz=float(max(update_hz, 1.0)),
                start_time=time.time(),
            )
            self._nudge_next_update_t = 0.0
            self._manual_action = np.zeros((2,), dtype=np.float32)
            self._use_autopilot = False
            self._hold_brake = True
        return anchor

    def start_lane_pid(
        self,
        target_speed: float,
        progress_ratio: Optional[float] = None,
        lateral_ratio: Optional[float] = None,
        lookahead_m: float = 0.9,
        y: float = 0.0,
        obstacle_radius: float = 0.25,
        safety_margin: float = 0.05,
        steer_kp: float = 1.00,
        steer_ki: float = 0.0,
        steer_kd: float = 0.0,
        steer_lat_gain: float = 0.80,
        speed_kp: float = 0.90,
        speed_ki: float = 0.18,
        speed_kd: float = 0.02,
        max_throttle: float = 0.32,
        min_throttle: float = 0.06,
        throttle_steer_damp: float = 0.35,
        place_on_start: bool = True,
    ) -> TrackTarget:
        """用 pure pursuit + speed PID 让障碍车沿指定车道持续绕圈。"""
        anchor = self._resolve_track_target(
            progress_ratio=progress_ratio,
            lateral_ratio=lateral_ratio,
            y=y,
            obstacle_radius=obstacle_radius,
            safety_margin=safety_margin,
        )
        if place_on_start:
            self.place_pose(
                x=anchor.x,
                z=anchor.z,
                yaw_deg=anchor.yaw_deg,
                world_y=self.default_world_y,
                hold_brake=False,
                timeout_s=self.placement_timeout_s,
            )
        with self._lock:
            self._clear_dynamic_modes_locked()
            self._lane_speed_pid.configure(
                kp=float(speed_kp),
                ki=float(speed_ki),
                kd=float(speed_kd),
                integral_limit=3.0,
                output_limits=(0.0, float(max(max_throttle, 0.0))),
            )
            self._lane_speed_pid.reset()
            self._target = anchor
            self._lane_pid_cfg = LanePIDConfig(
                target_speed=float(max(target_speed, 0.0)),
                lateral_ratio=float(anchor.lateral_ratio if lateral_ratio is None else lateral_ratio),
                lookahead_m=float(max(lookahead_m, 0.1)),
                pure_pursuit_gain=float(max(steer_kp, 0.0)),
                lookahead_speed_gain=float(max(steer_lat_gain, 0.0)),
                recovery_steer_gain=float(max(0.6 * max(steer_kp, 0.0) + 0.4, 0.0)),
                reverse_steer_gain=float(max(0.7 * max(steer_kp, 0.0) + 0.2, 0.0)),
                speed_kp=float(speed_kp),
                speed_ki=float(speed_ki),
                speed_kd=float(speed_kd),
                max_throttle=float(max(max_throttle, 0.0)),
                min_throttle=float(_clip_float(min_throttle, 0.0, max(max_throttle, 0.0))),
                throttle_steer_damp=float(_clip_float(throttle_steer_damp, 0.0, 0.95)),
            )
            self._lane_pid_debug = LanePIDDebugState(
                active=0.0,
                target_speed=float(max(target_speed, 0.0)),
                effective_lookahead=float(max(lookahead_m, 0.1)),
            )
            self._manual_action = np.zeros((2,), dtype=np.float32)
            self._use_autopilot = False
            self._hold_brake = False
        return anchor

    def teleport_pose(
        self,
        x: float,
        z: float,
        yaw_deg: float = 0.0,
        world_y: Optional[float] = None,
        hold_brake: bool = True,
    ) -> None:
        """直接发送瞬移消息，不等待位姿回读。适合批量同步放置。"""
        self._teleport_raw(
            x=float(x),
            z=float(z),
            yaw_deg=float(yaw_deg),
            world_y=world_y,
            hold_brake=hold_brake,
        )

    def place_explicit_target(
        self,
        target: TrackTarget,
        hold_brake: bool = True,
        timeout_s: Optional[float] = None,
    ) -> TrackTarget:
        """直接使用已采样好的目标点放置，避免再次按 progress/lateral 重采样。"""
        with self._lock:
            self._clear_dynamic_modes_locked()
            self._target = target
        if self._handler() is not None:
            try:
                placed = self.place_pose(
                    x=target.x,
                    z=target.z,
                    yaw_deg=target.yaw_deg,
                    world_y=self.default_world_y,
                    hold_brake=hold_brake,
                    timeout_s=timeout_s,
                )
                if placed is not None:
                    return target
            except Exception as exc:
                with self._lock:
                    self._last_error = "%s: %s" % (type(exc).__name__, exc)
        with self._lock:
            self._use_autopilot = True
            self._hold_brake = False
        return target

    def set_track_target(
        self,
        progress_ratio: float,
        lateral_ratio: float = 0.5,
        yaw_deg_override: Optional[float] = None,
        y: float = 0.0,
        obstacle_radius: float = 0.25,
        safety_margin: float = 0.05,
        direct_place: bool = True,
        hold_brake: bool = True,
        timeout_s: Optional[float] = None,
    ) -> TrackTarget:
        """规划一个赛道内目标位姿；默认直接放置，失败时可回退到自动驾驶。"""
        if self.track_geometry is None or not self.scene_key:
            raise ValueError("track_geometry and scene_key are required for track targets")

        target = sample_track_target(
            track_geometry=self.track_geometry,
            scene_key=self.scene_key,
            progress_ratio=progress_ratio,
            lateral_ratio=lateral_ratio,
            y=y,
            obstacle_radius=obstacle_radius,
            safety_margin=safety_margin,
        )
        if yaw_deg_override is not None:
            target = _with_target_yaw(target, float(yaw_deg_override))
        with self._lock:
            self._clear_dynamic_modes_locked()
            self._target = target
        if direct_place and self._handler() is not None:
            try:
                placed = self.place_pose(
                    x=target.x,
                    z=target.z,
                    yaw_deg=target.yaw_deg,
                    world_y=self.default_world_y,
                    hold_brake=hold_brake,
                    timeout_s=timeout_s,
                )
                if placed is not None:
                    return target
            except Exception as exc:
                with self._lock:
                    self._last_error = "%s: %s" % (type(exc).__name__, exc)
        with self._lock:
            self._use_autopilot = True
            self._hold_brake = False
        return target

    def place_track_target(
        self,
        progress_ratio: float,
        lateral_ratio: float = 0.5,
        yaw_deg_override: Optional[float] = None,
        y: float = 0.0,
        obstacle_radius: float = 0.25,
        safety_margin: float = 0.05,
        hold_brake: bool = True,
        timeout_s: Optional[float] = None,
    ) -> TrackTarget:
        return self.set_track_target(
            progress_ratio=progress_ratio,
            lateral_ratio=lateral_ratio,
            yaw_deg_override=yaw_deg_override,
            y=y,
            obstacle_radius=obstacle_radius,
            safety_margin=safety_margin,
            direct_place=True,
            hold_brake=hold_brake,
            timeout_s=timeout_s,
        )

    def place_pose(
        self,
        x: float,
        z: float,
        yaw_deg: float = 0.0,
        world_y: Optional[float] = None,
        hold_brake: bool = True,
        timeout_s: Optional[float] = None,
    ) -> Optional[PoseState]:
        """
        按 Python 侧赛道坐标直接放置障碍车。

        说明：
        - `x/z` 使用与 telemetry / track.py 一致的坐标；
        - 发送给 Unity 前会自动乘 `unity_world_scale`（默认 8）。
        """
        handler = self._handler()
        if handler is None:
            raise RuntimeError("Obstacle client is not spawned")

        request = {
            "x": round(float(x), 3),
            "z": round(float(z), 3),
            "yaw": round(float(yaw_deg), 1),
            "world_y": None if world_y is None else round(float(world_y), 3),
            "hold_brake": bool(hold_brake),
            "timeout_s": None if timeout_s is None else round(float(timeout_s), 3),
        }
        self._log_debug_event("place_pose_start", request=request)
        try:
            self._teleport_raw(
                x=float(x),
                z=float(z),
                yaw_deg=float(yaw_deg),
                world_y=world_y,
                hold_brake=hold_brake,
            )
            pose = self._wait_for_pose(
                x=x,
                z=z,
                yaw_deg=yaw_deg,
                timeout_s=self.placement_timeout_s if timeout_s is None else timeout_s,
            )
        except Exception as exc:
            self._log_debug_event(
                "place_pose_error",
                request=request,
                error="%s: %s" % (type(exc).__name__, exc),
            )
            raise

        pos_err = None
        yaw_err = None
        if pose is not None:
            pos_err = math.hypot(float(pose.x - x), float(pose.z - z))
            yaw_err = abs(math.degrees(_wrap_pi(math.radians(float(pose.yaw_deg - yaw_deg)))))
        self._log_debug_event(
            "place_pose_result",
            request=request,
            observed=self._pose_debug_payload(pose),
            pos_err=None if pos_err is None else round(float(pos_err), 3),
            yaw_err=None if yaw_err is None else round(float(yaw_err), 1),
            ok=bool(pos_err is not None and pos_err <= 0.10 and yaw_err is not None and yaw_err <= 10.0),
        )
        return pose

    def query_node_position(self, index: int, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        """查询 Unity car path 节点坐标，同时返回 world 坐标和 telemetry 坐标。"""
        handler = self._handler()
        if handler is None:
            raise RuntimeError("Obstacle client is not spawned")

        self._node_position_evt.clear()
        with self._lock:
            self._node_position_resp = None
        self._send_raw({"msg_type": "node_position", "index": str(int(index))}, blocking=False)
        ok = self._node_position_evt.wait(self.placement_timeout_s if timeout_s is None else timeout_s)
        if not ok:
            raise TimeoutError("Timed out waiting for node_position response")
        with self._lock:
            resp = dict(self._node_position_resp or {})
        world_x = float(resp["pos_x"])
        world_y = float(resp["pos_y"])
        world_z = float(resp["pos_z"])
        x, y, z = unity_world_to_telemetry(world_x, world_y, world_z, self.unity_world_scale)
        resp.update(
            {
                "world_x": world_x,
                "world_y": world_y,
                "world_z": world_z,
                "x": x,
                "y": y,
                "z": z,
            }
        )
        return resp

    def clear_target(self) -> None:
        with self._lock:
            self._clear_dynamic_modes_locked()
            self._target = None
            self._use_autopilot = False
            self._hold_brake = False

    def _clear_dynamic_modes_locked(self) -> None:
        self._nudge_cfg = None
        self._nudge_next_update_t = 0.0
        self._jitter_cfg = None
        self._jitter_next_update_t = 0.0
        self._lane_pid_cfg = None
        self._lane_pid_debug = LanePIDDebugState()
        self._lane_speed_pid.reset()
        self._lane_steer_pid.reset()

    def _current_track_pose(self) -> Optional[PoseState]:
        pose = self.get_obstacle_pose()
        if pose is None or pose.progress_ratio is None:
            return None
        return pose

    def _infer_lateral_ratio(self, pose: Optional[PoseState]) -> float:
        if (
            pose is None
            or pose.progress_ratio is None
            or pose.lat_err is None
            or self.track_geometry is None
            or not self.scene_key
        ):
            return 0.5
        g = self.track_geometry.scenes[self.scene_key]
        _idx, _t, _center, _left, _right, width = _segment_pose_at_progress(g, pose.progress_ratio)
        return _clip_float(0.5 + float(pose.lat_err) / max(width, 1e-6), 0.0, 1.0)

    def _resolve_track_target(
        self,
        progress_ratio: Optional[float],
        lateral_ratio: Optional[float],
        y: float,
        obstacle_radius: float,
        safety_margin: float,
    ) -> TrackTarget:
        if self.track_geometry is None or not self.scene_key:
            raise ValueError("track_geometry and scene_key are required for obstacle motion")

        current_pose = self._current_track_pose()
        with self._lock:
            target = self._target

        active_progress = progress_ratio
        if active_progress is None:
            if target is not None:
                active_progress = float(target.progress_ratio)
            elif current_pose is not None and current_pose.progress_ratio is not None:
                active_progress = float(current_pose.progress_ratio)
            else:
                raise ValueError("progress_ratio is required before obstacle pose/target is available")

        active_lateral = lateral_ratio
        if active_lateral is None:
            if target is not None:
                active_lateral = float(target.lateral_ratio)
            else:
                active_lateral = self._infer_lateral_ratio(current_pose)

        return sample_track_target(
            track_geometry=self.track_geometry,
            scene_key=self.scene_key,
            progress_ratio=float(active_progress),
            lateral_ratio=float(active_lateral),
            y=y,
            obstacle_radius=obstacle_radius,
            safety_margin=safety_margin,
        )

    def _sample_target_with_arc_offset(self, anchor: TrackTarget, delta_s_m: float) -> TrackTarget:
        if self.track_geometry is None or anchor.scene_key not in self.track_geometry.scenes:
            raise ValueError("track_geometry is required for arc-offset obstacle motion")
        g = self.track_geometry.scenes[anchor.scene_key]
        progress_ratio = float(anchor.progress_ratio + float(delta_s_m) / max(float(g.loop_len), 1e-6))
        target = sample_track_target(
            track_geometry=self.track_geometry,
            scene_key=anchor.scene_key,
            progress_ratio=progress_ratio,
            lateral_ratio=float(anchor.lateral_ratio),
            y=anchor.y,
        )
        return _with_target_yaw(target, float(anchor.yaw_deg))

    @staticmethod
    def _sample_target_with_local_offset(anchor: TrackTarget, longitudinal_m: float) -> TrackTarget:
        yaw_rad = math.radians(telemetry_yaw_deg_to_track_heading_deg(float(anchor.yaw_deg)))
        dx = float(math.cos(yaw_rad) * float(longitudinal_m))
        dz = float(math.sin(yaw_rad) * float(longitudinal_m))
        return TrackTarget(
            scene_key=anchor.scene_key,
            track_idx=int(anchor.track_idx),
            progress_ratio=float(anchor.progress_ratio),
            lateral_ratio=float(anchor.lateral_ratio),
            x=float(anchor.x + dx),
            y=float(anchor.y),
            z=float(anchor.z + dz),
            yaw_deg=float(anchor.yaw_deg),
            width=float(anchor.width),
        )

    def obstacle_coordinates(self) -> Optional[Tuple[float, float, float]]:
        pose = self.get_obstacle_pose()
        if pose is None:
            return None
        return pose.x, pose.y, pose.z

    def get_obstacle_pose(self) -> Optional[PoseState]:
        with self._lock:
            info = _copy_info(self._last_info)
            prev_idx = self._last_track_idx
        pose = pose_from_info(info, self.track_geometry, self.scene_key, prev_idx=prev_idx)
        if pose is not None and pose.track_idx is not None:
            with self._lock:
                self._last_track_idx = int(pose.track_idx)
        return pose

    def get_snapshot(self, agent_info: Optional[Dict[str, Any]] = None) -> ObstacleSnapshot:
        if agent_info is not None:
            with self._lock:
                self._agent_info = _copy_info(agent_info)

        obstacle_pose = self.get_obstacle_pose()
        with self._lock:
            agent_info_local = _copy_info(self._agent_info)
            target = self._target
        agent_pose = pose_from_info(agent_info_local, self.track_geometry, self.scene_key, None)
        relative = compute_relative_state(agent_pose, obstacle_pose)
        return ObstacleSnapshot(
            obstacle=obstacle_pose,
            target=target,
            agent=agent_pose,
            relative=relative,
        )

    def update(self, agent_info: Optional[Dict[str, Any]] = None) -> ObstacleSnapshot:
        """
        刷新 agent 位姿缓存并返回最新快照。

        说明：
        - 障碍车的真实推进在后台线程中持续进行；
        - 本方法本身不阻塞 sim，只做信息同步与快照计算。
        """
        return self.get_snapshot(agent_info=agent_info)

    def last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    @staticmethod
    def _pose_debug_payload(pose: Optional[PoseState]) -> Optional[Dict[str, Any]]:
        if pose is None:
            return None
        return {
            "x": round(float(pose.x), 3),
            "y": round(float(pose.y), 3),
            "z": round(float(pose.z), 3),
            "yaw": round(float(pose.yaw_deg), 1),
            "speed": round(float(pose.speed), 3),
            "cte": round(float(pose.cte), 3),
            "hit": str(pose.hit),
            "progress": None if pose.progress_ratio is None else round(float(pose.progress_ratio), 4),
        }

    def _motion_mode_debug_locked(self) -> str:
        if self._nudge_cfg is not None:
            return "nudge"
        if self._jitter_cfg is not None:
            return "jitter"
        if self._lane_pid_cfg is not None:
            return "lane_pid"
        if self._use_autopilot:
            return "track_target"
        if self._hold_brake and np.allclose(self._manual_action, 0.0):
            return "hold"
        return "manual"

    def _log_debug_event(self, event: str, **fields: Any) -> None:
        try:
            with self._lock:
                self._debug_event_seq += 1
                seq = int(self._debug_event_seq)
                last_info_t = float(self._last_info_t)
                last_error = self._last_error
                motion_mode = self._motion_mode_debug_locked()
                target = self._target
                thread_alive = bool(self._thread is not None and self._thread.is_alive())
            payload: Dict[str, Any] = {
                "event": str(event),
                "seq": seq,
                "t": round(float(time.time()), 3),
                "car_name": str(self.conf.get("car_name", "")),
                "racer_name": str(self.conf.get("racer_name", "")),
                "guid": str(self.conf.get("guid", "")),
                "scene_key": self.scene_key,
                "host": self.host,
                "port": int(self.port),
                "motion_mode": motion_mode,
                "thread_alive": thread_alive,
                "handler_ready": self._handler() is not None,
                "last_info_age_s": None if last_info_t <= 0.0 else round(float(max(0.0, time.time() - last_info_t)), 3),
                "last_error": last_error,
                "target": None if target is None else {
                    "progress": round(float(target.progress_ratio), 4),
                    "lateral": round(float(target.lateral_ratio), 3),
                    "x": round(float(target.x), 3),
                    "z": round(float(target.z), 3),
                    "yaw": round(float(target.yaw_deg), 1),
                },
            }
            payload.update(fields)
            print("[obstacle_car] " + json.dumps(payload, sort_keys=True, ensure_ascii=False), flush=True)
        except Exception:
            pass

    def debug_state(self) -> Dict[str, Any]:
        with self._lock:
            last_info_t = float(self._last_info_t)
            last_error = self._last_error
            last_set_position_msg = None if self._last_set_position_msg is None else dict(self._last_set_position_msg)
            target = self._target
            thread_alive = bool(self._thread is not None and self._thread.is_alive())
            motion_mode = self._motion_mode_debug_locked()
        last_info_age_s = None if last_info_t <= 0.0 else max(0.0, time.time() - last_info_t)
        return {
            "car_name": str(self.conf.get("car_name", "")),
            "racer_name": str(self.conf.get("racer_name", "")),
            "guid": str(self.conf.get("guid", "")),
            "env_id": self.env_id,
            "scene_key": self.scene_key,
            "host": self.host,
            "port": int(self.port),
            "body_rgb": list(self.conf.get("body_rgb", ())),
            "font_size": int(self.conf.get("font_size", 0) or 0),
            "thread_alive": thread_alive,
            "handler_ready": self._handler() is not None,
            "motion_mode": motion_mode,
            "last_info_age_s": None if last_info_age_s is None else round(float(last_info_age_s), 3),
            "last_error": last_error,
            "target": None if target is None else {
                "progress": round(float(target.progress_ratio), 4),
                "lateral": round(float(target.lateral_ratio), 3),
                "x": round(float(target.x), 3),
                "z": round(float(target.z), 3),
                "yaw": round(float(target.yaw_deg), 1),
            },
            "last_set_position": last_set_position_msg,
        }

    def _handler(self):
        if self._env is None:
            return None
        viewer = getattr(self._env, "viewer", None)
        return getattr(viewer, "handler", None)

    def _install_handler_hooks(self) -> None:
        handler = self._handler()
        if handler is None:
            return
        try:
            handler.fns["node_position"] = self._on_node_position
        except Exception:
            pass

    def _on_node_position(self, message: Dict[str, Any]) -> None:
        with self._lock:
            self._node_position_resp = dict(message)
        self._node_position_evt.set()

    def _send_raw(self, msg: Dict[str, Any], blocking: bool = True) -> None:
        handler = self._handler()
        if handler is None:
            raise RuntimeError("Obstacle client is not spawned")
        if blocking:
            handler.blocking_send(msg)
        else:
            handler.queue_message(msg)

    def _teleport_raw(
        self,
        x: float,
        z: float,
        yaw_deg: float = 0.0,
        world_y: Optional[float] = None,
        hold_brake: bool = True,
    ) -> None:
        handler = self._handler()
        if handler is None:
            raise RuntimeError("Obstacle client is not spawned")

        if world_y is None:
            pose_now = self.get_obstacle_pose()
            if pose_now is not None:
                _, world_y_now, _ = telemetry_to_unity_world(
                    pose_now.x, pose_now.y, pose_now.z, self.unity_world_scale
                )
                if math.isfinite(float(world_y_now)) and -5.0 <= float(world_y_now) <= 10.0:
                    world_y = world_y_now
                else:
                    world_y = self.default_world_y
            else:
                world_y = self.default_world_y

        world_x, _, world_z = telemetry_to_unity_world(x, 0.0, z, self.unity_world_scale)
        qx, qy, qz, qw = yaw_deg_to_unity_quaternion(yaw_deg)
        msg = {
            "msg_type": "set_position",
            "pos_x": str(world_x),
            "pos_y": str(world_y),
            "pos_z": str(world_z),
            "Qx": str(qx),
            "Qy": str(qy),
            "Qz": str(qz),
            "Qw": str(qw),
        }
        with self._lock:
            self._last_set_position_msg = {
                "x": round(float(x), 3),
                "z": round(float(z), 3),
                "yaw": round(float(yaw_deg), 1),
                "world_x": round(float(world_x), 3),
                "world_y": round(float(world_y), 3),
                "world_z": round(float(world_z), 3),
                "hold_brake": bool(hold_brake),
            }
        if hold_brake:
            try:
                handler.send_control(0.0, 0.0, 1.0)
            except Exception:
                pass
        self._send_raw(msg, blocking=True)
        with self._lock:
            self._manual_action = np.zeros((2,), dtype=np.float32)
            self._use_autopilot = False
            self._hold_brake = bool(hold_brake)

    def _wait_for_pose(
        self,
        x: float,
        z: float,
        yaw_deg: float,
        timeout_s: float,
        pos_tol: float = 0.10,
        yaw_tol_deg: float = 10.0,
    ) -> Optional[PoseState]:
        if timeout_s <= 0.0:
            return self.get_obstacle_pose()
        deadline = time.time() + float(timeout_s)
        last_pose = self.get_obstacle_pose()
        while time.time() < deadline:
            pose = self.get_obstacle_pose()
            if pose is not None:
                last_pose = pose
                pos_err = math.hypot(float(pose.x - x), float(pose.z - z))
                yaw_err = abs(math.degrees(_wrap_pi(math.radians(float(pose.yaw_deg - yaw_deg)))))
                if pos_err <= pos_tol and yaw_err <= yaw_tol_deg:
                    return pose
            time.sleep(0.02)
        return last_pose

    def _run_loop(self) -> None:
        if self._env is None:
            return

        while not self._stop_evt.is_set():
            try:
                if self._reset_evt.is_set():
                    with self._lock:
                        reset_reason = self._reset_reason or "reset_event"
                    self._log_debug_event(
                        "client_reset_start",
                        reason=reset_reason,
                        pose=self._pose_debug_payload(self.get_obstacle_pose()),
                    )
                    self._env.reset()
                    with self._lock:
                        self._last_info = {}
                        self._last_track_idx = None
                        self._reset_reason = ""
                    self._reset_evt.clear()
                    self._log_debug_event("client_reset_done", reason=reset_reason)

                pose = self.get_obstacle_pose()
                now = time.time()
                nudge_target = self._update_nudge_pose(now)
                jitter_target = None if nudge_target is not None else self._update_jitter_pose(now)
                if nudge_target is not None or jitter_target is not None:
                    handler = self._handler()
                    if handler is None:
                        raise RuntimeError("Obstacle client handler is unavailable")
                    handler.send_control(0.0, 0.0, 1.0)
                    _obs, _reward, done, info = self._env.viewer.observe()
                else:
                    action = self._compute_action(pose)
                    with self._lock:
                        hold_brake = bool(
                            self._hold_brake and self._lane_pid_cfg is None and not self._use_autopilot and np.allclose(action, 0.0)
                        )
                    if hold_brake:
                        handler = self._handler()
                        if handler is None:
                            raise RuntimeError("Obstacle client handler is unavailable")
                        handler.send_control(0.0, 0.0, 1.0)
                        _obs, _reward, done, info = self._env.viewer.observe()
                    else:
                        frame_skip = int(max(getattr(self._env, "frame_skip", 1) or 1, 1))
                        for _ in range(frame_skip):
                            self._env.viewer.take_action(action)
                            _obs, _reward, done, info = self._env.viewer.observe()

                with self._lock:
                    self._last_info = _copy_info(info)
                    self._last_info_t = time.time()
                    self._last_error = None

                if done and self.auto_reset_on_done:
                    self._log_debug_event(
                        "auto_reset_on_done",
                        pose=self._pose_debug_payload(pose_from_info(info, self.track_geometry, self.scene_key, self._last_track_idx)),
                        hit=_extract_hit(info),
                        speed=round(float(_extract_speed(info)), 3),
                    )
                    with self._lock:
                        self._reset_reason = "auto_reset_on_done"
                    self._reset_evt.set()
            except Exception as exc:
                error_sig = "%s: %s" % (type(exc).__name__, exc)
                now = time.time()
                should_log = False
                with self._lock:
                    self._last_error = error_sig
                    if (
                        error_sig != self._last_debug_error_sig
                        or now - float(self._last_debug_error_t) >= 2.0
                    ):
                        should_log = True
                        self._last_debug_error_sig = error_sig
                        self._last_debug_error_t = now
                if should_log:
                    self._log_debug_event("run_loop_error", error=error_sig)
                time.sleep(0.1)

    def _compute_action(self, pose: Optional[PoseState]) -> np.ndarray:
        with self._lock:
            manual = self._manual_action.copy()
            target = self._target
            use_autopilot = bool(self._use_autopilot)
            lane_pid_cfg = self._lane_pid_cfg

        if lane_pid_cfg is not None:
            return self._compute_lane_pid_action(pose, lane_pid_cfg)

        if not use_autopilot or target is None:
            return manual

        if pose is None:
            return np.array([0.0, 0.0], dtype=np.float32)

        dx = float(target.x - pose.x)
        dz = float(target.z - pose.z)
        planar_distance = float(math.hypot(dx, dz))
        pose_yaw_rad = math.radians(telemetry_yaw_deg_to_track_heading_deg(float(pose.yaw_deg)))

        if planar_distance <= self.stop_distance:
            return np.array([0.0, 0.0], dtype=np.float32)

        # 靠近目标时直接指向目标点，便于收敛到非中心线位置。
        if planar_distance <= self.approach_distance:
            target_heading = math.atan2(dz, dx)
            heading_to_target = _wrap_pi(target_heading - pose_yaw_rad)
            steer = _clip_float(self.k_target_heading * heading_to_target, -1.0, 1.0)
            if planar_distance > self.stop_distance:
                ratio = float(
                    np.clip(
                        (planar_distance - self.stop_distance) / max(self.approach_distance - self.stop_distance, 1e-6),
                        0.0,
                        1.0,
                    )
                )
                throttle = self.crawl_throttle + (self.cruise_throttle - self.crawl_throttle) * ratio
            else:
                throttle = 0.0
            return np.array([steer, throttle], dtype=np.float32)

        if self.track_geometry is None or not self.scene_key:
            return np.array([0.0, self.crawl_throttle], dtype=np.float32)

        geo = self.track_geometry.query(
            self.scene_key,
            x=pose.x,
            z=pose.z,
            yaw_rad=pose_yaw_rad,
            prev_idx=self._last_track_idx,
        )
        current_idx = int(geo["idx"])
        with self._lock:
            self._last_track_idx = current_idx

        heading_err = math.atan2(float(geo["heading_err_sin"]), float(geo["heading_err_cos"]))
        steer = _clip_float(
            -self.k_lat * float(geo["lat_err_norm"]) - self.k_heading * heading_err,
            -1.0,
            1.0,
        )

        if pose.track_idx is not None:
            g = self.track_geometry.scenes[self.scene_key]
            arc_remaining = self._forward_arc_distance(g, int(pose.track_idx), int(target.track_idx))
        else:
            arc_remaining = planar_distance

        if arc_remaining > self.slow_distance:
            throttle = self.cruise_throttle
        else:
            ratio = float(np.clip(arc_remaining / max(self.slow_distance, 1e-6), 0.0, 1.0))
            throttle = self.crawl_throttle + (self.cruise_throttle - self.crawl_throttle) * ratio

        return np.array([steer, throttle], dtype=np.float32)

    def _update_nudge_pose(self, now: float) -> Optional[TrackTarget]:
        with self._lock:
            cfg = self._nudge_cfg
            next_update_t = self._nudge_next_update_t
        if cfg is None:
            return None
        if now < next_update_t:
            return cfg.anchor

        phase = 2.0 * math.pi * ((float(now) - float(cfg.start_time)) / max(float(cfg.period_s), 1e-6))
        target = self._sample_target_with_local_offset(cfg.anchor, float(cfg.amplitude_m) * math.sin(phase))
        self._teleport_raw(
            x=target.x,
            z=target.z,
            yaw_deg=target.yaw_deg,
            world_y=self.default_world_y,
            hold_brake=True,
        )
        with self._lock:
            self._target = target
            self._nudge_next_update_t = float(now) + 1.0 / max(float(cfg.update_hz), 1e-6)
        return target

    def _update_jitter_pose(self, now: float) -> Optional[TrackTarget]:
        with self._lock:
            cfg = self._jitter_cfg
            next_update_t = self._jitter_next_update_t
        if cfg is None:
            return None
        if now < next_update_t:
            return cfg.anchor

        phase = 2.0 * math.pi * ((float(now) - float(cfg.start_time)) / max(float(cfg.period_s), 1e-6))
        target = self._sample_target_with_arc_offset(cfg.anchor, float(cfg.amplitude_m) * math.sin(phase))
        self._teleport_raw(
            x=target.x,
            z=target.z,
            yaw_deg=target.yaw_deg,
            world_y=self.default_world_y,
            hold_brake=True,
        )
        with self._lock:
            self._target = target
            self._jitter_next_update_t = float(now) + 1.0 / max(float(cfg.update_hz), 1e-6)
        return target

    def _compute_lane_pid_action(self, pose: Optional[PoseState], cfg: LanePIDConfig) -> np.ndarray:
        if pose is None or self.track_geometry is None or not self.scene_key:
            with self._lock:
                self._lane_pid_debug = LanePIDDebugState(
                    active=0.0,
                    target_speed=float(max(cfg.target_speed, 0.0)),
                    effective_lookahead=float(max(cfg.lookahead_m, 0.1)),
                )
            return np.zeros((2,), dtype=np.float32)

        g = self.track_geometry.scenes[self.scene_key]
        pose_yaw_rad = math.radians(telemetry_yaw_deg_to_track_heading_deg(float(pose.yaw_deg)))
        geo = self.track_geometry.query(
            self.scene_key,
            x=pose.x,
            z=pose.z,
            yaw_rad=pose_yaw_rad,
            prev_idx=self._last_track_idx,
        )
        current_idx = int(geo["idx"])
        with self._lock:
            self._last_track_idx = current_idx

        if pose.progress_ratio is not None:
            progress_ratio = float(pose.progress_ratio)
        else:
            progress_ratio = float(g.cum_len[current_idx] / max(g.loop_len, 1e-6))

        effective_lookahead = float(
            max(
                float(cfg.lookahead_m),
                float(cfg.lookahead_m) + float(cfg.lookahead_speed_gain) * max(float(pose.speed), 0.0),
            )
        )
        lookahead_target = sample_track_target(
            track_geometry=self.track_geometry,
            scene_key=self.scene_key,
            progress_ratio=progress_ratio + effective_lookahead / max(float(g.loop_len), 1e-6),
            lateral_ratio=float(cfg.lateral_ratio),
        )
        dx = float(lookahead_target.x - pose.x)
        dz = float(lookahead_target.z - pose.z)
        local_forward = float(dx * math.cos(pose_yaw_rad) + dz * math.sin(pose_yaw_rad))
        local_left = float(dx * (-math.sin(pose_yaw_rad)) + dz * math.cos(pose_yaw_rad))
        lookahead_distance = float(max(math.hypot(dx, dz), 1e-3))

        if local_forward <= 0.05:
            steer = -_clip_float(
                float(cfg.reverse_steer_gain) * math.atan2(local_left, max(abs(local_forward), 1e-3)),
                -1.0,
                1.0,
            )
            throttle_cap = 0.08
        else:
            curvature = float(2.0 * local_left / max(lookahead_distance * lookahead_distance, 1e-3))
            steer = -_clip_float(float(cfg.pure_pursuit_gain) * curvature, -1.0, 1.0)
            throttle_cap = float(cfg.max_throttle)

        target_speed = float(max(cfg.target_speed, 0.0))
        speed_now = float(max(pose.speed, 0.0))
        speed_error = float(target_speed - speed_now)
        if target_speed <= 1e-3:
            throttle = 0.0
        else:
            now = time.time()
            throttle = self._lane_speed_pid.step(speed_error, now)
            if throttle > 0.0:
                throttle = max(float(cfg.min_throttle), throttle)
            alpha = math.atan2(local_left, max(local_forward, 1e-3))
            throttle *= float(np.clip(1.0 - 0.45 * min(abs(alpha) / 0.9, 1.0), 0.25, 1.0))
            throttle *= float(np.clip(1.0 - 0.25 * min(abs(float(geo["lat_err_norm"])) / 1.5, 1.0), 0.35, 1.0))
            throttle *= 1.0 - float(cfg.throttle_steer_damp) * min(abs(float(steer)), 1.0)
            throttle = _clip_float(throttle, 0.0, throttle_cap)

        with self._lock:
            self._target = lookahead_target
            self._lane_pid_debug = LanePIDDebugState(
                active=1.0,
                target_speed=float(target_speed),
                speed=float(speed_now),
                speed_error=float(speed_error),
                effective_lookahead=float(effective_lookahead),
                local_forward=float(local_forward),
                local_left=float(local_left),
                lookahead_distance=float(lookahead_distance),
                lat_err_norm=float(geo.get("lat_err_norm", 0.0) or 0.0),
                steer=float(steer),
                throttle=float(throttle),
                reverse_mode=float(local_forward <= 0.05),
            )
        return np.array([float(steer), float(throttle)], dtype=np.float32)

    @staticmethod
    def _forward_arc_distance(g: SceneGeometry, idx_now: int, idx_target: int) -> float:
        i0 = int(idx_now) % g.center.shape[0]
        i1 = int(idx_target) % g.center.shape[0]
        if i1 >= i0:
            return float(g.cum_len[i1] - g.cum_len[i0])
        return float((g.loop_len - g.cum_len[i0]) + g.cum_len[i1])


def spawn_preset_obstacle_fleet(
    scene: str,
    host: str = "127.0.0.1",
    port: int = 9091,
    track_dir: Optional[str] = None,
    layout: Optional[Sequence[Tuple[float, float]]] = None,
    count: Optional[int] = None,
    min_separation_world: Optional[float] = None,
    seed: Optional[int] = None,
    body_rgbs: Sequence[Tuple[int, int, int]] = _DEFAULT_OBSTACLE_BODY_RGBS,
    hold_brake: bool = True,
    spawn_gap: float = 0.0,
    placement_timeout_s: float = 1.5,
    initial_place: bool = True,
) -> DonkeyObstacleFleet:
    """
    生成一组静态障碍车。

    目前仅支持：
    - `scene="gt"` / `generated_track`
    - `scene="ws"` / `waveshare`

    默认行为：
    - 在赛道范围内随机生成 2 台障碍车
    - 两台初始位置最少相隔 `3.0` 个 sim/world 坐标单位
    - 若显式传入 `layout`，则使用固定布局并忽略随机采样参数
    """
    preset = resolve_obstacle_fleet_preset(scene)
    track_geometry = build_obstacle_track_geometry(preset.name, track_dir=track_dir)
    active_count = preset.default_count if count is None else int(count)
    active_min_separation_world = (
        preset.min_separation_world if min_separation_world is None else float(min_separation_world)
    )

    if layout is not None:
        active_layout = list(layout)
        if not active_layout:
            raise ValueError("Obstacle fleet layout cannot be empty")
        targets: List[TrackTarget] = [
            sample_track_target(
                track_geometry=track_geometry,
                scene_key=preset.scene_key,
                progress_ratio=float(progress_ratio),
                lateral_ratio=float(lateral_ratio),
                obstacle_radius=preset.obstacle_radius,
                safety_margin=preset.safety_margin,
            )
            for progress_ratio, lateral_ratio in active_layout
        ]
    else:
        targets = sample_random_track_targets(
            track_geometry=track_geometry,
            scene_key=preset.scene_key,
            count=active_count,
            obstacle_radius=preset.obstacle_radius,
            safety_margin=preset.safety_margin,
            min_separation_world=active_min_separation_world,
            rng=np.random.default_rng(seed),
        )
    if not targets:
        raise ValueError("Obstacle fleet targets cannot be empty")

    if not body_rgbs:
        raise ValueError("body_rgbs cannot be empty")

    cars: List[DonkeyObstacleCar] = []
    try:
        for i, target in enumerate(targets, start=1):
            color = tuple(int(v) for v in body_rgbs[(i - 1) % len(body_rgbs)])
            car = DonkeyObstacleCar(
                env_id=preset.env_id,
                track_geometry=track_geometry,
                scene_key=preset.scene_key,
                host=host,
                port=int(port),
                # The bare shell sits too low in waveshare and disappears from LiDAR.
                body_style="donkey",
                body_rgb=color,
                car_name=(f"gt obst {i}" if preset.name == "gt" else f"wsobst {i}"),
                racer_name=(f"gt obst {i}" if preset.name == "gt" else f"wsobst {i}"),
                bio=f"{preset.name} obstacle car",
                country="US",
                auto_reset_on_done=False,
                placement_timeout_s=float(placement_timeout_s),
            )
            staging_x = float(preset.staging_x_start - (i - 1) * preset.staging_x_step)
            car.spawn(
                reset_on_spawn=False,
                hidden_pose=(staging_x, float(preset.staging_z), 0.0),
                hold_brake=hold_brake,
            )
            cars.append(car)
            if spawn_gap > 0.0:
                time.sleep(float(spawn_gap))

        if initial_place:
            for car, target in zip(cars, targets):
                car.teleport_pose(
                    x=target.x,
                    z=target.z,
                    yaw_deg=target.yaw_deg,
                    hold_brake=hold_brake,
                )
            time.sleep(0.6)
        return DonkeyObstacleFleet(
            preset=preset,
            track_geometry=track_geometry,
            cars=cars,
            targets=targets,
        )
    except Exception:
        for car in reversed(cars):
            try:
                car.shutdown()
            except Exception:
                pass
        raise


def spawn_gt_obstacles(
    host: str = "127.0.0.1",
    port: int = 9091,
    track_dir: Optional[str] = None,
    layout: Optional[Sequence[Tuple[float, float]]] = None,
    count: Optional[int] = None,
    min_separation_world: Optional[float] = None,
    seed: Optional[int] = None,
) -> DonkeyObstacleFleet:
    return spawn_preset_obstacle_fleet(
        scene="gt",
        host=host,
        port=port,
        track_dir=track_dir,
        layout=layout,
        count=count,
        min_separation_world=min_separation_world,
        seed=seed,
    )


def spawn_ws_obstacles(
    host: str = "127.0.0.1",
    port: int = 9091,
    track_dir: Optional[str] = None,
    layout: Optional[Sequence[Tuple[float, float]]] = None,
    count: Optional[int] = None,
    min_separation_world: Optional[float] = None,
    seed: Optional[int] = None,
) -> DonkeyObstacleFleet:
    return spawn_preset_obstacle_fleet(
        scene="ws",
        host=host,
        port=port,
        track_dir=track_dir,
        layout=layout,
        count=count,
        min_separation_world=min_separation_world,
        seed=seed,
    )
