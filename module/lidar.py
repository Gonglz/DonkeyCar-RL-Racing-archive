"""
module/lidar.py

Canonical LiDAR adapter used by V17.

Design choices
--------------
- Canonical policy input is 36 sectors over a front-facing 180 degree FOV.
- To keep the representation symmetric with 36 sectors, sectors are ordered
  from left to right, not "sector_0 = exact front". With an even number of
  sectors there is no single exact-center bin.
- Each sector stores:
  - range: robust low quantile of valid distances
  - valid: 1 if the sector has any valid beam, else 0
- Invalid sectors use range=max_range and valid=0.

The simulator LiDAR emitted by gym-donkeycar is an angle-indexed array with
invalid entries encoded as negative numbers. For V17 we aggregate contiguous
beam groups into the canonical 36-sector representation and emulate stale-scan
reuse so policy inputs resemble the real asynchronous control loop.

Frozen April 24, 2026 simulator assumptions
-------------------------------------------
- default canonical max range is 20m
- simulator packet `rx ~= 180` corresponds to ego-forward
- simulator packet `d` is converted back to telemetry meters via `/8`
- LiDAR remains a target/safety signal, not a wall-vs-car classifier
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CanonicalLidarSpec:
    num_sectors: int = 36
    fov_deg: float = 180.0
    max_range_m: float = 20.0
    near_clip_m: float = 0.18
    invalid_fill_m: float = 20.0
    robust_quantile: float = 0.20
    sim_packet_front_rx_deg: float = 180.0
    sim_packet_distance_scale_to_m: float = 1.0 / 8.0

    def __post_init__(self) -> None:
        if self.num_sectors <= 0:
            raise ValueError("num_sectors must be positive")
        if self.fov_deg <= 0.0:
            raise ValueError("fov_deg must be positive")


DEFAULT_CANONICAL_LIDAR_SPEC = CanonicalLidarSpec()


@dataclass(frozen=True)
class TargetTokenSpec:
    """
    Near-field target-tracking limits on top of the wider canonical LiDAR range.

    The canonical/safety path now uses a 20m simulator baseline, but the
    primary target token still intentionally focuses on nearer interaction
    structure.
    """
    max_range_m: float = 20.0
    max_target_range_m: float = 5.0
    max_rel_speed_mps: float = 3.0
    max_ttc_s: float = 6.0
    max_width_m: float = 2.5
    max_track_age_steps: int = 8
    max_assoc_distance_m: float = 1.25
    min_cluster_sectors: int = 1
    front_gap_center_width: int = 6


DEFAULT_TARGET_TOKEN_SPEC = TargetTokenSpec()


def _sanitize_raw_lidar(raw_lidar: Optional[Iterable[float]]) -> np.ndarray:
    if raw_lidar is None:
        return np.zeros((0,), dtype=np.float32)
    try:
        arr = np.asarray(list(raw_lidar), dtype=np.float32).reshape(-1)
    except Exception:
        return np.zeros((0,), dtype=np.float32)
    if arr.size == 0:
        return arr
    return arr


def _wrap_angle_deg_pm180(angle_deg: float) -> float:
    return float((float(angle_deg) + 180.0) % 360.0 - 180.0)


def _sim_packet_rx_to_canonical_angle_deg(
    rx_deg: float,
    spec: CanonicalLidarSpec = DEFAULT_CANONICAL_LIDAR_SPEC,
) -> float:
    """
    Convert Unity LiDAR packet `rx` into ego-frame canonical angles.

    Empirically, the simulator emits a full 360-degree horizontal sweep, and
    the ego-forward direction aligns with `rx ~= 180`. This matches
    gym-donkeycar's own packet reconstruction, which indexes points over a full
    360-degree sweep.
    """
    return _wrap_angle_deg_pm180(float(rx_deg) - float(spec.sim_packet_front_rx_deg))


def _sim_packet_distance_to_meters(
    distance_value: float,
    spec: CanonicalLidarSpec = DEFAULT_CANONICAL_LIDAR_SPEC,
) -> float:
    """
    Convert Unity packet distances into telemetry-space meters.

    DonkeySim uses an 8x world scale for vehicle placement relative to Python
    telemetry coordinates; LiDAR packet distances follow that same world scale
    in local testing, so convert them back before canonicalization.
    """
    return float(distance_value) * float(spec.sim_packet_distance_scale_to_m)


def _infer_sim_array_points_per_sweep(total_points: int) -> int:
    """
    Infer the number of horizontal samples in one 360-degree sweep.

    gym-donkeycar reconstructs arrays as `num_sweeps_levels * point_per_sweep`.
    Prefer common sweep sizes; otherwise fall back to treating the whole array
    as one sweep.
    """
    total_points = int(max(total_points, 0))
    if total_points <= 0:
        return 0
    for candidate in (360, 180, 120, 90, 72, 60, 45, 36):
        if total_points % candidate == 0:
            return candidate
    return total_points


def canonical_lidar_from_sim_array(
    raw_lidar: Optional[Iterable[float]],
    spec: CanonicalLidarSpec = DEFAULT_CANONICAL_LIDAR_SPEC,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate simulator LiDAR array into canonical sectors.

    Returns
    -------
    lidar_range : (num_sectors,) float32
    lidar_valid : (num_sectors,) float32 in {0,1}
    """
    arr = _sanitize_raw_lidar(raw_lidar)
    ranges = np.full((spec.num_sectors,), spec.invalid_fill_m, dtype=np.float32)
    valid = np.zeros((spec.num_sectors,), dtype=np.float32)

    if arr.size == 0:
        return ranges, valid

    arr = arr.astype(np.float32, copy=False)
    points_per_sweep = _infer_sim_array_points_per_sweep(int(arr.size))
    if points_per_sweep <= 0:
        return ranges, valid

    try:
        sweep_arr = arr.reshape(-1, points_per_sweep)
    except ValueError:
        sweep_arr = arr.reshape(1, -1)
        points_per_sweep = int(sweep_arr.shape[1])
    deg_per_index = 360.0 / max(float(points_per_sweep), 1.0)

    angles_deg: List[float] = []
    ranges_m: List[float] = []
    for sweep in sweep_arr:
        valid_idx = np.flatnonzero(np.isfinite(sweep) & (sweep >= 0.0))
        if valid_idx.size == 0:
            continue
        for idx in valid_idx.tolist():
            angles_deg.append(
                _sim_packet_rx_to_canonical_angle_deg(
                    float(idx) * deg_per_index,
                    spec=spec,
                )
            )
            ranges_m.append(_sim_packet_distance_to_meters(float(sweep[idx]), spec=spec))

    if not angles_deg:
        return ranges, valid
    return canonical_lidar_from_scan(angles_deg=angles_deg, ranges_m=ranges_m, spec=spec)


def canonical_lidar_from_sim_packet(
    lidar_packet: Optional[Iterable[Dict[str, Any]]],
    spec: CanonicalLidarSpec = DEFAULT_CANONICAL_LIDAR_SPEC,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate Unity LiDAR packet points into canonical sectors.

    DonkeySim telemetry packets expose per-point `rx` and `d`. The simulator
    emits a full 360-degree sweep, while canonical V17 LiDAR only keeps the
    ego-front 180-degree field of view. Convert `rx` into ego-frame angles
    where positive means left and negative means right, then map packet
    distances from Unity world scale into telemetry meters.
    """
    ranges = np.full((spec.num_sectors,), spec.invalid_fill_m, dtype=np.float32)
    valid = np.zeros((spec.num_sectors,), dtype=np.float32)
    if lidar_packet is None:
        return ranges, valid

    try:
        points = list(lidar_packet)
    except Exception:
        return ranges, valid
    if not points:
        return ranges, valid

    angles_deg: List[float] = []
    ranges_m: List[float] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        rx = point.get("rx")
        dist = point.get("d")
        if rx is None or dist is None:
            continue
        try:
            rx_f = float(rx)
            dist_f = float(dist)
        except Exception:
            continue
        angles_deg.append(_sim_packet_rx_to_canonical_angle_deg(rx_f, spec=spec))
        ranges_m.append(_sim_packet_distance_to_meters(dist_f, spec=spec))

    if not angles_deg:
        return ranges, valid
    return canonical_lidar_from_scan(angles_deg=angles_deg, ranges_m=ranges_m, spec=spec)


def canonical_lidar_from_scan(
    angles_deg: Iterable[float],
    ranges_m: Iterable[float],
    spec: CanonicalLidarSpec = DEFAULT_CANONICAL_LIDAR_SPEC,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate raw angle/range beams into canonical sectors.

    The canonical ordering is left -> right over the front 180 degree FOV.
    Angles are expected in degrees in the ego frame, where positive means left.
    """
    ang = np.asarray(list(angles_deg), dtype=np.float32).reshape(-1)
    dist = np.asarray(list(ranges_m), dtype=np.float32).reshape(-1)
    if ang.size != dist.size:
        raise ValueError("angles_deg and ranges_m must have the same length")

    half_fov = 0.5 * float(spec.fov_deg)
    sector_edges = np.linspace(half_fov, -half_fov, spec.num_sectors + 1, dtype=np.float32)
    out_range = np.full((spec.num_sectors,), spec.invalid_fill_m, dtype=np.float32)
    out_valid = np.zeros((spec.num_sectors,), dtype=np.float32)

    for sector_idx in range(spec.num_sectors):
        hi = sector_edges[sector_idx]
        lo = sector_edges[sector_idx + 1]
        if sector_idx == 0:
            mask = (ang <= hi) & (ang >= lo)
        else:
            mask = (ang < hi) & (ang >= lo)
        sector = dist[mask]
        sector_valid = sector[np.isfinite(sector) & (sector >= spec.near_clip_m)]
        if sector_valid.size == 0:
            continue
        clipped = np.clip(sector_valid, spec.near_clip_m, spec.max_range_m)
        out_range[sector_idx] = np.float32(np.quantile(clipped, spec.robust_quantile))
        out_valid[sector_idx] = 1.0
    return out_range, out_valid


def canonical_gap_features(
    lidar_range: np.ndarray,
    lidar_valid: np.ndarray,
    spec: CanonicalLidarSpec = DEFAULT_CANONICAL_LIDAR_SPEC,
) -> Tuple[float, float, float]:
    lidar_range = np.asarray(lidar_range, dtype=np.float32).reshape(-1)
    lidar_valid = np.asarray(lidar_valid, dtype=np.float32).reshape(-1)
    if lidar_range.shape != lidar_valid.shape:
        raise ValueError("lidar_range and lidar_valid must have the same shape")
    if lidar_range.size != spec.num_sectors:
        raise ValueError(f"expected {spec.num_sectors} sectors, got {lidar_range.size}")

    left = lidar_range[: spec.num_sectors // 2][lidar_valid[: spec.num_sectors // 2] > 0.5]
    right = lidar_range[spec.num_sectors // 2 :][lidar_valid[spec.num_sectors // 2 :] > 0.5]
    center_half = max(1, min(spec.num_sectors // 4, 3))
    center = spec.num_sectors // 2
    front = lidar_range[center - center_half : center + center_half][
        lidar_valid[center - center_half : center + center_half] > 0.5
    ]
    left_gap = float(np.quantile(left, 0.20)) if left.size else float(spec.max_range_m)
    right_gap = float(np.quantile(right, 0.20)) if right.size else float(spec.max_range_m)
    front_min = float(np.min(front)) if front.size else float(spec.max_range_m)
    return front_min, left_gap, right_gap


def _sector_center_angles_deg(spec: CanonicalLidarSpec) -> np.ndarray:
    half_fov = 0.5 * float(spec.fov_deg)
    edges = np.linspace(half_fov, -half_fov, spec.num_sectors + 1, dtype=np.float32)
    return 0.5 * (edges[:-1] + edges[1:])


class TargetTokenBuffer:
    """
    Lightweight foreground-target tracker on top of canonical LiDAR.

    Output 12D token layout:
      [exist,
       rel_long, rel_lat,
       rel_v_long, rel_v_lat,
       ttc_s,
       confidence,
       age_norm,
       width_proxy,
       front_min_range,
       left_gap_proxy,
       right_gap_proxy]

    This is intentionally simple: contiguous valid-sector clusters are treated
    as foreground hypotheses and the best front-relevant cluster is tracked over
    time with nearest-neighbor association.
    """

    def __init__(
        self,
        spec: CanonicalLidarSpec = DEFAULT_CANONICAL_LIDAR_SPEC,
        token_spec: TargetTokenSpec = DEFAULT_TARGET_TOKEN_SPEC,
        control_dt_s: float = 0.05,
    ):
        self.spec = spec
        self.token_spec = token_spec
        self.control_dt_s = float(max(1e-3, control_dt_s))
        self._sector_angles_deg = _sector_center_angles_deg(spec)
        self.reset()

    def reset(self) -> None:
        self._prev_track: Optional[Dict[str, float]] = None
        self._cached_token = np.zeros((12,), dtype=np.float32)
        self._steps_since_update = 0
        self._track_age_steps = 0

    def _extract_clusters(self, lidar_range: np.ndarray, lidar_valid: np.ndarray) -> List[Dict[str, float]]:
        valid_mask = (np.asarray(lidar_valid, dtype=np.float32).reshape(-1) > 0.5) & (
            np.asarray(lidar_range, dtype=np.float32).reshape(-1) < float(self.token_spec.max_target_range_m)
        )
        if not np.any(valid_mask):
            return []

        ranges = np.asarray(lidar_range, dtype=np.float32).reshape(-1)
        clusters: List[Dict[str, float]] = []
        start: Optional[int] = None
        for idx, is_valid in enumerate(valid_mask.tolist() + [False]):
            if is_valid and start is None:
                start = idx
                continue
            if is_valid:
                continue
            if start is None:
                continue
            end = idx
            if (end - start) >= int(self.token_spec.min_cluster_sectors):
                sector_idx = np.arange(start, end, dtype=np.int32)
                sector_ranges = ranges[sector_idx]
                angles_deg = self._sector_angles_deg[sector_idx]
                weights = 1.0 / np.clip(sector_ranges, 0.25, float(self.spec.max_range_m))
                mean_angle_deg = float(np.average(angles_deg, weights=weights))
                dist_m = float(np.quantile(np.clip(sector_ranges, self.spec.near_clip_m, self.spec.max_range_m), 0.20))
                angle_rad = np.deg2rad(mean_angle_deg)
                rel_long = float(dist_m * np.cos(angle_rad))
                rel_lat = float(dist_m * np.sin(angle_rad))
                angular_width_rad = np.deg2rad(float(max(1, end - start)) * float(self.spec.fov_deg) / float(self.spec.num_sectors))
                width_proxy = float(np.clip(dist_m * angular_width_rad, 0.0, self.token_spec.max_width_m))
                closeness = float(np.clip(1.0 - dist_m / max(self.token_spec.max_target_range_m, 1e-3), 0.0, 1.0))
                frontness = float(np.clip(np.cos(angle_rad), 0.0, 1.0))
                span_bonus = float(np.clip((end - start) / 3.0, 0.3, 1.0))
                lateral_gate = float(np.clip(1.0 - abs(rel_lat) / 1.8, 0.0, 1.0))
                width_pref = float(np.exp(-((width_proxy - 0.45) / 0.45) ** 2))
                confidence = float(
                    np.clip(
                        0.28 * closeness
                        + 0.28 * frontness
                        + 0.22 * width_pref
                        + 0.12 * lateral_gate
                        + 0.10 * span_bonus,
                        0.0,
                        1.0,
                    )
                )
                score = float(
                    0.22 * closeness
                    + 0.36 * frontness
                    + 0.24 * width_pref
                    + 0.14 * lateral_gate
                    + 0.04 * span_bonus
                )
                clusters.append(
                    {
                        "start": float(start),
                        "end": float(end),
                        "dist_m": float(dist_m),
                        "rel_long": rel_long,
                        "rel_lat": rel_lat,
                        "width_proxy": width_proxy,
                        "confidence": confidence,
                        "score": score,
                    }
                )
            start = None
        return clusters

    def _select_primary_cluster(self, clusters: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if not clusters:
            return None
        candidates = [c for c in clusters if c["rel_long"] > -0.2]
        if not candidates:
            candidates = clusters
        candidates.sort(key=lambda c: (c["score"], c["rel_long"]), reverse=True)
        return dict(candidates[0])

    def _build_no_target_token(self, front_min: float, left_gap: float, right_gap: float) -> np.ndarray:
        token = np.zeros((12,), dtype=np.float32)
        token[5] = float(self.token_spec.max_ttc_s)
        token[9] = float(front_min)
        token[10] = float(left_gap)
        token[11] = float(right_gap)
        return token

    def observe(
        self,
        lidar_range: np.ndarray,
        lidar_valid: np.ndarray,
        is_new_scan: float,
        steps_since_new_scan: float,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        front_min, left_gap, right_gap = canonical_gap_features(lidar_range, lidar_valid, self.spec)
        self._steps_since_update += 1

        if float(is_new_scan) < 0.5:
            cached = self._cached_token.copy()
            cached[9] = float(front_min)
            cached[10] = float(left_gap)
            cached[11] = float(right_gap)
            self._cached_token = cached.copy()
            return cached, {
                "target_exist": float(cached[0]),
                "target_confidence": float(cached[6]),
                "target_age_norm": float(cached[7]),
                "front_min_range": float(front_min),
                "left_gap": float(left_gap),
                "right_gap": float(right_gap),
                "steps_since_new_scan": float(steps_since_new_scan),
            }

        clusters = self._extract_clusters(lidar_range, lidar_valid)
        primary = self._select_primary_cluster(clusters)
        if primary is None:
            self._prev_track = None
            self._track_age_steps = 0
            self._steps_since_update = 0
            token = self._build_no_target_token(front_min, left_gap, right_gap)
            self._cached_token = token.copy()
            return token, {
                "target_exist": 0.0,
                "target_confidence": 0.0,
                "target_age_norm": 0.0,
                "front_min_range": float(front_min),
                "left_gap": float(left_gap),
                "right_gap": float(right_gap),
                "steps_since_new_scan": float(steps_since_new_scan),
            }

        rel_long = float(primary["rel_long"])
        rel_lat = float(primary["rel_lat"])
        rel_v_long = 0.0
        rel_v_lat = 0.0
        confidence = float(primary["confidence"])
        matched = False
        if self._prev_track is not None:
            dx = rel_long - float(self._prev_track["rel_long"])
            dy = rel_lat - float(self._prev_track["rel_lat"])
            assoc_dist = float(np.hypot(dx, dy))
            if assoc_dist <= float(self.token_spec.max_assoc_distance_m):
                dt = float(max(1, self._steps_since_update)) * self.control_dt_s
                rel_v_long = float(np.clip(dx / max(dt, 1e-3), -self.token_spec.max_rel_speed_mps, self.token_spec.max_rel_speed_mps))
                rel_v_lat = float(np.clip(dy / max(dt, 1e-3), -self.token_spec.max_rel_speed_mps, self.token_spec.max_rel_speed_mps))
                self._track_age_steps = min(self._track_age_steps + 1, int(self.token_spec.max_track_age_steps))
                confidence = float(np.clip(confidence + 0.15, 0.0, 1.0))
                matched = True
            else:
                self._track_age_steps = 0
        else:
            self._track_age_steps = 0

        if (not matched) and self._prev_track is None:
            self._track_age_steps = 0

        if rel_v_long < -0.05 and rel_long > 0.0:
            ttc_s = float(np.clip(rel_long / max(-rel_v_long, 1e-3), 0.0, self.token_spec.max_ttc_s))
        else:
            ttc_s = float(self.token_spec.max_ttc_s)

        token = np.array(
            [
                1.0,
                float(np.clip(rel_long, -self.token_spec.max_range_m, self.token_spec.max_range_m)),
                float(np.clip(rel_lat, -self.token_spec.max_range_m, self.token_spec.max_range_m)),
                float(rel_v_long),
                float(rel_v_lat),
                float(ttc_s),
                float(np.clip(confidence, 0.0, 1.0)),
                float(min(self._track_age_steps / max(1, self.token_spec.max_track_age_steps), 1.0)),
                float(np.clip(primary["width_proxy"], 0.0, self.token_spec.max_width_m)),
                float(front_min),
                float(left_gap),
                float(right_gap),
            ],
            dtype=np.float32,
        )
        self._prev_track = {"rel_long": rel_long, "rel_lat": rel_lat}
        self._steps_since_update = 0
        self._cached_token = token.copy()
        return token, {
            "target_exist": 1.0,
            "target_confidence": float(token[6]),
            "target_age_norm": float(token[7]),
            "front_min_range": float(front_min),
            "left_gap": float(left_gap),
            "right_gap": float(right_gap),
            "steps_since_new_scan": float(steps_since_new_scan),
        }


class SimAsyncLidarBuffer:
    """
    Reuse LiDAR scans for 2-4 control steps to emulate asynchronous real-world scans.
    """

    def __init__(
        self,
        spec: CanonicalLidarSpec = DEFAULT_CANONICAL_LIDAR_SPEC,
        repeat_min_steps: int = 2,
        repeat_max_steps: int = 4,
        seed: Optional[int] = None,
    ):
        self.spec = spec
        self.repeat_min_steps = int(max(1, repeat_min_steps))
        self.repeat_max_steps = int(max(self.repeat_min_steps, repeat_max_steps))
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self._cached_range = np.full((self.spec.num_sectors,), self.spec.invalid_fill_m, dtype=np.float32)
        self._cached_valid = np.zeros((self.spec.num_sectors,), dtype=np.float32)
        self._repeat_target = 1
        self._steps_since_new_scan = 0
        self._repeat_count = 0
        self._initialized = False

    def _sample_repeat_target(self) -> int:
        return int(self.rng.integers(self.repeat_min_steps, self.repeat_max_steps + 1))

    def observe(
        self,
        raw_lidar: Optional[Iterable[float]],
        raw_lidar_packet: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        force_refresh = bool(raw_lidar_packet) and (not self._initialized or not np.any(self._cached_valid > 0.5))
        if force_refresh or (not self._initialized) or (self._steps_since_new_scan >= self._repeat_target):
            if raw_lidar_packet:
                self._cached_range, self._cached_valid = canonical_lidar_from_sim_packet(raw_lidar_packet, self.spec)
            else:
                self._cached_range, self._cached_valid = canonical_lidar_from_sim_array(raw_lidar, self.spec)
            self._repeat_target = self._sample_repeat_target()
            self._steps_since_new_scan = 0
            self._repeat_count = 1
            self._initialized = True
            is_new_scan = 1.0
        else:
            self._steps_since_new_scan += 1
            self._repeat_count += 1
            is_new_scan = 0.0

        meta = {
            "is_new_scan": float(is_new_scan),
            "steps_since_new_scan_norm": float(min(self._steps_since_new_scan / 4.0, 1.0)),
            "scan_age_norm": float(min(self._steps_since_new_scan / 4.0, 1.0)),
            "repeat_count_norm": float(min((self._repeat_count - 1) / 4.0, 1.0)),
            "steps_since_new_scan": float(self._steps_since_new_scan),
            "repeat_count": float(self._repeat_count),
        }
        return self._cached_range.copy(), self._cached_valid.copy(), meta


def flatten_canonical_lidar(lidar_range: np.ndarray, lidar_valid: np.ndarray) -> np.ndarray:
    lidar_range = np.asarray(lidar_range, dtype=np.float32).reshape(-1)
    lidar_valid = np.asarray(lidar_valid, dtype=np.float32).reshape(-1)
    if lidar_range.shape != lidar_valid.shape:
        raise ValueError("lidar_range and lidar_valid must have the same shape")
    return np.concatenate([lidar_range, lidar_valid], axis=0).astype(np.float32)


__all__ = [
    "CanonicalLidarSpec",
    "DEFAULT_CANONICAL_LIDAR_SPEC",
    "DEFAULT_TARGET_TOKEN_SPEC",
    "SimAsyncLidarBuffer",
    "TargetTokenBuffer",
    "TargetTokenSpec",
    "canonical_gap_features",
    "canonical_lidar_from_sim_packet",
    "canonical_lidar_from_scan",
    "canonical_lidar_from_sim_array",
    "flatten_canonical_lidar",
]
