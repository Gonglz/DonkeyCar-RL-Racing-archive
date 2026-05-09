#!/usr/bin/env python3
"""
Evaluate sector-level domain gap between real and sim canonical LiDAR.

Supported inputs
----------------
Real side:
- JSONL records containing a nested `lidar` dict from runtime_monitor
- JSONL records containing raw LaserScan-like fields directly
- JSONL / NPZ records containing canonical LiDAR arrays

Sim side:
- JSONL records containing simulator `lidar_packet` / raw packet fields
- JSONL records containing raw simulator `lidar` arrays
- JSONL / NPZ records containing canonical LiDAR arrays

This script turns the V17 Phase-0/Phase-F acceptance criteria into a runnable check:
- per-sector valid-ratio MAE
- per-sector normalized-range Wasserstein median / p95
- scene-level histogram JS divergence
- optional range-band summaries for near/far diagnostics
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from module.lidar import (  # noqa: E402
    CanonicalLidarSpec,
    canonical_gap_features,
    canonical_lidar_from_scan,
    canonical_lidar_from_sim_array,
    canonical_lidar_from_sim_packet,
)


def _iter_input_files(paths: Sequence[str]) -> Iterator[Path]:
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            for sub in sorted(path.rglob("*")):
                if sub.is_file() and sub.suffix.lower() in (".jsonl", ".npz", ".npy"):
                    yield sub
        elif path.is_file():
            yield path


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _get_scene_name(obj: dict, default: str = "all") -> str:
    for key in ("scene_key", "scene", "logging_key", "domain"):
        value = obj.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    lidar_obj = obj.get("lidar")
    if isinstance(lidar_obj, dict):
        for key in ("scene_key", "scene"):
            value = lidar_obj.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return default


def _as_float_array(value: object) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    return arr


def _extract_canonical(obj: dict, spec: CanonicalLidarSpec) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    range_keys = ("canonical_lidar_range", "lidar_range", "range", "ranges_canonical")
    valid_keys = ("canonical_lidar_valid", "lidar_valid", "valid", "valid_canonical")

    lidar_obj = obj.get("lidar")
    payloads: List[dict] = [obj]
    if isinstance(lidar_obj, dict):
        payloads.append(lidar_obj)

    for payload in payloads:
        for rk in range_keys:
            if rk not in payload:
                continue
            r = _as_float_array(payload.get(rk))
            if r is None or r.size != spec.num_sectors:
                continue
            v = None
            for vk in valid_keys:
                v = _as_float_array(payload.get(vk))
                if v is not None:
                    break
            if v is None:
                v = np.ones((spec.num_sectors,), dtype=np.float32)
            if v.size != spec.num_sectors:
                continue
            return r.astype(np.float32), np.clip(v.astype(np.float32), 0.0, 1.0)
    return None


def _extract_real_from_obj(obj: dict, spec: CanonicalLidarSpec) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    canonical = _extract_canonical(obj, spec)
    if canonical is not None:
        return canonical

    payload = obj.get("lidar", obj)
    if not isinstance(payload, dict):
        return None

    ranges = _as_float_array(payload.get("ranges"))
    angle_min = payload.get("angle_min")
    angle_increment = payload.get("angle_increment")
    if ranges is None or angle_min is None or angle_increment is None:
        raw_lidar = _as_float_array(payload.get("lidar"))
        if raw_lidar is not None:
            return canonical_lidar_from_sim_array(raw_lidar, spec)
        return None

    angles_rad = float(angle_min) + np.arange(ranges.size, dtype=np.float32) * float(angle_increment)
    angles_deg = np.degrees(angles_rad)
    return canonical_lidar_from_scan(angles_deg=angles_deg, ranges_m=ranges, spec=spec)


def _extract_sim_packet_from_payload(payload: object) -> Optional[List[dict]]:
    candidates: List[object] = []
    if isinstance(payload, dict):
        candidates.extend(
            [
                payload.get("lidar_packet"),
                payload.get("lidar_raw_packet"),
                payload.get("packet"),
            ]
        )
        lidar_obj = payload.get("lidar")
        if isinstance(lidar_obj, dict):
            candidates.extend(
                [
                    lidar_obj.get("lidar_packet"),
                    lidar_obj.get("lidar_raw_packet"),
                    lidar_obj.get("packet"),
                ]
            )
        info_obj = payload.get("info")
        if isinstance(info_obj, dict):
            candidates.extend(
                [
                    info_obj.get("lidar_packet"),
                    info_obj.get("lidar_raw_packet"),
                ]
            )
    for candidate in candidates:
        try:
            points = list(candidate) if candidate is not None else []
        except Exception:
            continue
        if not points:
            continue
        if any(isinstance(point, dict) and ("rx" in point) and ("d" in point) for point in points):
            return points
    return None


def _extract_sim_from_obj(obj: dict, spec: CanonicalLidarSpec) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    packet = _extract_sim_packet_from_payload(obj)
    if packet is not None:
        return canonical_lidar_from_sim_packet(packet, spec)

    canonical = _extract_canonical(obj, spec)
    if canonical is not None:
        return canonical

    payload = obj.get("lidar", obj)
    if isinstance(payload, dict):
        ranges = _as_float_array(payload.get("ranges"))
        angle_min = payload.get("angle_min")
        angle_increment = payload.get("angle_increment")
        if ranges is not None and angle_min is not None and angle_increment is not None:
            angles_rad = float(angle_min) + np.arange(ranges.size, dtype=np.float32) * float(angle_increment)
            angles_deg = np.degrees(angles_rad)
            return canonical_lidar_from_scan(angles_deg=angles_deg, ranges_m=ranges, spec=spec)
        raw_lidar = _as_float_array(payload.get("raw"))
        if raw_lidar is not None:
            return canonical_lidar_from_sim_array(raw_lidar, spec)

    for key in ("lidar",):
        raw = _as_float_array(obj.get(key))
        if raw is not None:
            return canonical_lidar_from_sim_array(raw, spec)

    info = obj.get("info")
    if isinstance(info, dict):
        raw = _as_float_array(info.get("lidar"))
        if raw is not None:
            return canonical_lidar_from_sim_array(raw, spec)
    return None


def _load_np_like(path: Path, spec: CanonicalLidarSpec, kind: str) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    out: List[Tuple[str, np.ndarray, np.ndarray]] = []

    def _infer_scene_from_path(np_path: Path) -> str:
        for part in reversed(np_path.parts):
            if part in ("waveshare", "generated_track", "roboracingleague_track"):
                return str(part)
        return "all"

    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=True)
        raw = np.asarray(arr, dtype=np.float32).reshape(-1)
        ranges, valid = canonical_lidar_from_sim_array(raw, spec)
        out.append((_infer_scene_from_path(path), ranges, valid))
        return out

    data = np.load(path, allow_pickle=True)
    keys = set(data.files)
    if "canonical_lidar_range" in keys and "canonical_lidar_valid" in keys:
        ranges = np.asarray(data["canonical_lidar_range"], dtype=np.float32).reshape(-1)
        valid = np.asarray(data["canonical_lidar_valid"], dtype=np.float32).reshape(-1)
        out.append((_infer_scene_from_path(path), ranges, valid))
        return out
    if "lidar_range" in keys and "lidar_valid" in keys:
        ranges = np.asarray(data["lidar_range"], dtype=np.float32).reshape(-1)
        valid = np.asarray(data["lidar_valid"], dtype=np.float32).reshape(-1)
        out.append((_infer_scene_from_path(path), ranges, valid))
        return out
    if "lidar" in keys:
        raw = np.asarray(data["lidar"], dtype=np.float32).reshape(-1)
        if raw.size == (2 * spec.num_sectors):
            ranges = raw[: spec.num_sectors].astype(np.float32)
            valid = np.clip(raw[spec.num_sectors :].astype(np.float32), 0.0, 1.0)
            out.append((_infer_scene_from_path(path), ranges, valid))
            return out
        if kind == "real":
            maybe = _extract_real_from_obj({"lidar": raw.tolist()}, spec)
        else:
            maybe = _extract_sim_from_obj({"lidar": raw.tolist()}, spec)
        if maybe is not None:
            out.append((_infer_scene_from_path(path), maybe[0], maybe[1]))
    return out


def _load_samples(paths: Sequence[str], spec: CanonicalLidarSpec, kind: str) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    samples: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for path in _iter_input_files(paths):
        if path.suffix.lower() == ".jsonl":
            for obj in _iter_jsonl(path):
                scene = _get_scene_name(obj, default="all")
                maybe = _extract_real_from_obj(obj, spec) if kind == "real" else _extract_sim_from_obj(obj, spec)
                if maybe is None:
                    continue
                samples.append((scene, maybe[0], maybe[1]))
        elif path.suffix.lower() in (".npz", ".npy"):
            samples.extend(_load_np_like(path, spec, kind=kind))
    return samples


def _wasserstein_1d(a: np.ndarray, b: np.ndarray, num_q: int = 101) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size == 0 and b.size == 0:
        return 0.0
    if a.size == 0 or b.size == 0:
        return 1.0
    q = np.linspace(0.0, 1.0, num_q, dtype=np.float32)
    qa = np.quantile(a, q)
    qb = np.quantile(b, q)
    return float(np.mean(np.abs(qa - qb)))


def _safe_prob_hist(values: np.ndarray, bins: int) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0), density=False)
    hist = hist.astype(np.float64)
    hist += 1e-8
    hist /= np.sum(hist)
    return hist


def _js_divergence(a: np.ndarray, b: np.ndarray) -> float:
    m = 0.5 * (a + b)
    return float(0.5 * np.sum(a * np.log(a / m)) + 0.5 * np.sum(b * np.log(b / m)))


def _kl_divergence(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(a * np.log(a / b)))


def _fmt_range_component(value: float) -> str:
    value = float(value)
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _range_band_name(band_min_m: float, band_max_m: float, overall: bool = False) -> str:
    prefix = "overall" if overall else "range"
    return f"{prefix}_{_fmt_range_component(band_min_m)}_{_fmt_range_component(band_max_m)}m"


def _normalize_band_values(values_m: np.ndarray, band_min_m: float, band_max_m: float) -> np.ndarray:
    width = max(float(band_max_m) - float(band_min_m), 1e-6)
    return np.clip((values_m.astype(np.float32) - float(band_min_m)) / width, 0.0, 1.0)


def _sector_metrics(
    real_ranges: np.ndarray,
    real_valid: np.ndarray,
    sim_ranges: np.ndarray,
    sim_valid: np.ndarray,
    band_min_m: float,
    band_max_m: float,
    js_bins: int,
) -> Dict[str, object]:
    num_sectors = int(real_ranges.shape[1])
    sectors: List[Dict[str, float]] = []
    wd_list: List[float] = []
    js_list: List[float] = []
    valid_mae_list: List[float] = []

    for idx in range(num_sectors):
        real_mask = (
            (real_valid[:, idx] > 0.5)
            & (real_ranges[:, idx] >= float(band_min_m))
            & (real_ranges[:, idx] <= float(band_max_m))
        )
        sim_mask = (
            (sim_valid[:, idx] > 0.5)
            & (sim_ranges[:, idx] >= float(band_min_m))
            & (sim_ranges[:, idx] <= float(band_max_m))
        )
        real_vals_m = real_ranges[:, idx][real_mask].astype(np.float32)
        sim_vals_m = sim_ranges[:, idx][sim_mask].astype(np.float32)
        real_vals = _normalize_band_values(real_vals_m, band_min_m=band_min_m, band_max_m=band_max_m)
        sim_vals = _normalize_band_values(sim_vals_m, band_min_m=band_min_m, band_max_m=band_max_m)

        real_valid_ratio = float(np.mean(real_mask)) if real_mask.size else 0.0
        sim_valid_ratio = float(np.mean(sim_mask)) if sim_mask.size else 0.0
        wd = _wasserstein_1d(real_vals, sim_vals)
        hist_real = _safe_prob_hist(real_vals if real_vals.size else np.array([1.0], dtype=np.float32), bins=js_bins)
        hist_sim = _safe_prob_hist(sim_vals if sim_vals.size else np.array([1.0], dtype=np.float32), bins=js_bins)
        jsd = _js_divergence(hist_real, hist_sim)
        sectors.append(
            {
                "sector": float(idx),
                "valid_ratio_real": real_valid_ratio,
                "valid_ratio_sim": sim_valid_ratio,
                "valid_ratio_abs_diff": abs(real_valid_ratio - sim_valid_ratio),
                "range_mean_real": float(np.mean(real_vals)) if real_vals.size else 1.0,
                "range_mean_sim": float(np.mean(sim_vals)) if sim_vals.size else 1.0,
                "range_std_real": float(np.std(real_vals)) if real_vals.size else 0.0,
                "range_std_sim": float(np.std(sim_vals)) if sim_vals.size else 0.0,
                "range_p10_real": float(np.quantile(real_vals, 0.10)) if real_vals.size else 1.0,
                "range_p10_sim": float(np.quantile(sim_vals, 0.10)) if sim_vals.size else 1.0,
                "range_p50_real": float(np.quantile(real_vals, 0.50)) if real_vals.size else 1.0,
                "range_p50_sim": float(np.quantile(sim_vals, 0.50)) if sim_vals.size else 1.0,
                "range_p90_real": float(np.quantile(real_vals, 0.90)) if real_vals.size else 1.0,
                "range_p90_sim": float(np.quantile(sim_vals, 0.90)) if sim_vals.size else 1.0,
                "range_mean_real_m": float(np.mean(real_vals_m)) if real_vals_m.size else float(band_max_m),
                "range_mean_sim_m": float(np.mean(sim_vals_m)) if sim_vals_m.size else float(band_max_m),
                "range_p10_real_m": float(np.quantile(real_vals_m, 0.10)) if real_vals_m.size else float(band_max_m),
                "range_p10_sim_m": float(np.quantile(sim_vals_m, 0.10)) if sim_vals_m.size else float(band_max_m),
                "range_p50_real_m": float(np.quantile(real_vals_m, 0.50)) if real_vals_m.size else float(band_max_m),
                "range_p50_sim_m": float(np.quantile(sim_vals_m, 0.50)) if sim_vals_m.size else float(band_max_m),
                "range_p90_real_m": float(np.quantile(real_vals_m, 0.90)) if real_vals_m.size else float(band_max_m),
                "range_p90_sim_m": float(np.quantile(sim_vals_m, 0.90)) if sim_vals_m.size else float(band_max_m),
                "wasserstein": wd,
                "js_divergence": jsd,
            }
        )
        wd_list.append(wd)
        js_list.append(jsd)
        valid_mae_list.append(abs(real_valid_ratio - sim_valid_ratio))

    real_all_mask = (real_valid > 0.5) & (real_ranges >= float(band_min_m)) & (real_ranges <= float(band_max_m))
    sim_all_mask = (sim_valid > 0.5) & (sim_ranges >= float(band_min_m)) & (sim_ranges <= float(band_max_m))
    valid_real_all_m = real_ranges[real_all_mask].astype(np.float32)
    valid_sim_all_m = sim_ranges[sim_all_mask].astype(np.float32)
    valid_real_all = _normalize_band_values(valid_real_all_m, band_min_m=band_min_m, band_max_m=band_max_m)
    valid_sim_all = _normalize_band_values(valid_sim_all_m, band_min_m=band_min_m, band_max_m=band_max_m)
    hist_real_all = _safe_prob_hist(valid_real_all if valid_real_all.size else np.array([1.0], dtype=np.float32), bins=js_bins)
    hist_sim_all = _safe_prob_hist(valid_sim_all if valid_sim_all.size else np.array([1.0], dtype=np.float32), bins=js_bins)

    summary = {
        "band_min_m": float(band_min_m),
        "band_max_m": float(band_max_m),
        "valid_ratio_mae": float(np.mean(valid_mae_list)) if valid_mae_list else float("nan"),
        "wasserstein_median": float(np.median(wd_list)) if wd_list else float("nan"),
        "wasserstein_p95": float(np.quantile(wd_list, 0.95)) if wd_list else float("nan"),
        "sector_js_mean": float(np.mean(js_list)) if js_list else float("nan"),
        "scene_js_divergence": _js_divergence(hist_real_all, hist_sim_all),
        "scene_kl_real_to_sim": _kl_divergence(hist_real_all, hist_sim_all),
        "scene_kl_sim_to_real": _kl_divergence(hist_sim_all, hist_real_all),
        "value_count_real": int(valid_real_all_m.size),
        "value_count_sim": int(valid_sim_all_m.size),
    }
    return {"summary": summary, "per_sector": sectors}


def _evaluate_single_band(
    real_samples: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    sim_samples: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    band_min_m: float,
    band_max_m: float,
    js_bins: int,
) -> Dict[str, object]:
    real_grouped = _stack_samples(real_samples)
    sim_grouped = _stack_samples(sim_samples)
    result: Dict[str, object] = {
        "overall": None,
        "by_scene": {},
    }
    overall_real = {
        "ranges": np.stack([s[1] for s in real_samples], axis=0).astype(np.float32),
        "valid": np.stack([s[2] for s in real_samples], axis=0).astype(np.float32),
    }
    overall_sim = {
        "ranges": np.stack([s[1] for s in sim_samples], axis=0).astype(np.float32),
        "valid": np.stack([s[2] for s in sim_samples], axis=0).astype(np.float32),
    }
    overall_metrics = _sector_metrics(
        overall_real["ranges"],
        overall_real["valid"],
        overall_sim["ranges"],
        overall_sim["valid"],
        band_min_m=band_min_m,
        band_max_m=band_max_m,
        js_bins=js_bins,
    )
    result["overall"] = overall_metrics

    common_scenes = sorted(set(real_grouped.keys()) & set(sim_grouped.keys()))
    for scene in common_scenes:
        metrics = _sector_metrics(
            real_grouped[scene]["ranges"],
            real_grouped[scene]["valid"],
            sim_grouped[scene]["ranges"],
            sim_grouped[scene]["valid"],
            band_min_m=band_min_m,
            band_max_m=band_max_m,
            js_bins=js_bins,
        )
        result["by_scene"][scene] = {
            "summary": metrics["summary"],
            "per_sector": metrics["per_sector"],
            "real_count": int(real_grouped[scene]["ranges"].shape[0]),
            "sim_count": int(sim_grouped[scene]["ranges"].shape[0]),
        }
    return result


def _stack_samples(samples: Sequence[Tuple[str, np.ndarray, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
    grouped: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for scene, ranges, valid in samples:
        grouped.setdefault(scene, []).append((ranges, valid))

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for scene, items in grouped.items():
        out[scene] = {
            "ranges": np.stack([x[0] for x in items], axis=0).astype(np.float32),
            "valid": np.stack([x[1] for x in items], axis=0).astype(np.float32),
        }
    return out


def _feature_series(
    samples: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    spec: CanonicalLidarSpec,
) -> Dict[str, Dict[str, np.ndarray]]:
    grouped_raw: Dict[str, Dict[str, List[float]]] = {}
    feature_names = ("front_min", "left_gap", "right_gap", "valid_ratio")
    for scene, ranges, valid in samples:
        front_min, left_gap, right_gap = canonical_gap_features(ranges, valid, spec=spec)
        scene_store = grouped_raw.setdefault(scene, {name: [] for name in feature_names})
        scene_store["front_min"].append(float(front_min))
        scene_store["left_gap"].append(float(left_gap))
        scene_store["right_gap"].append(float(right_gap))
        scene_store["valid_ratio"].append(float(np.mean(np.asarray(valid, dtype=np.float32) > 0.5)))

    grouped: Dict[str, Dict[str, np.ndarray]] = {}
    for scene, payload in grouped_raw.items():
        grouped[scene] = {
            name: np.asarray(values, dtype=np.float32).reshape(-1)
            for name, values in payload.items()
        }
    return grouped


def _feature_metric_summary(
    real_vals: np.ndarray,
    sim_vals: np.ndarray,
    *,
    normalize_max: float,
) -> Dict[str, float]:
    real_vals = np.asarray(real_vals, dtype=np.float32).reshape(-1)
    sim_vals = np.asarray(sim_vals, dtype=np.float32).reshape(-1)
    real_mean = float(np.mean(real_vals)) if real_vals.size else float("nan")
    sim_mean = float(np.mean(sim_vals)) if sim_vals.size else float("nan")
    real_p50 = float(np.quantile(real_vals, 0.50)) if real_vals.size else float("nan")
    sim_p50 = float(np.quantile(sim_vals, 0.50)) if sim_vals.size else float("nan")
    real_p95 = float(np.quantile(real_vals, 0.95)) if real_vals.size else float("nan")
    sim_p95 = float(np.quantile(sim_vals, 0.95)) if sim_vals.size else float("nan")
    scale = max(float(normalize_max), 1e-6)
    real_norm = np.clip(real_vals / scale, 0.0, 1.0)
    sim_norm = np.clip(sim_vals / scale, 0.0, 1.0)
    return {
        "real_count": int(real_vals.size),
        "sim_count": int(sim_vals.size),
        "real_mean": real_mean,
        "sim_mean": sim_mean,
        "mean_abs_diff": abs(real_mean - sim_mean) if math.isfinite(real_mean) and math.isfinite(sim_mean) else float("inf"),
        "real_p50": real_p50,
        "sim_p50": sim_p50,
        "p50_abs_diff": abs(real_p50 - sim_p50) if math.isfinite(real_p50) and math.isfinite(sim_p50) else float("inf"),
        "real_p95": real_p95,
        "sim_p95": sim_p95,
        "p95_abs_diff": abs(real_p95 - sim_p95) if math.isfinite(real_p95) and math.isfinite(sim_p95) else float("inf"),
        "wasserstein": _wasserstein_1d(real_norm, sim_norm),
    }


def _evaluate_feature_alignment(
    real_samples: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    sim_samples: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    spec: CanonicalLidarSpec,
) -> Dict[str, object]:
    real_grouped = _feature_series(real_samples, spec=spec)
    sim_grouped = _feature_series(sim_samples, spec=spec)
    feature_scales = {
        "front_min": float(spec.max_range_m),
        "left_gap": float(spec.max_range_m),
        "right_gap": float(spec.max_range_m),
        "valid_ratio": 1.0,
    }

    def _build_payload(real_payload: Mapping[str, np.ndarray], sim_payload: Mapping[str, np.ndarray]) -> Dict[str, object]:
        return {
            "features": {
                name: _feature_metric_summary(
                    np.asarray(real_payload.get(name, np.zeros((0,), dtype=np.float32)), dtype=np.float32),
                    np.asarray(sim_payload.get(name, np.zeros((0,), dtype=np.float32)), dtype=np.float32),
                    normalize_max=float(feature_scales[name]),
                )
                for name in ("front_min", "left_gap", "right_gap", "valid_ratio")
            }
        }

    overall_real = {
        name: np.concatenate([payload[name] for payload in real_grouped.values()], axis=0)
        if real_grouped
        else np.zeros((0,), dtype=np.float32)
        for name in feature_scales
    }
    overall_sim = {
        name: np.concatenate([payload[name] for payload in sim_grouped.values()], axis=0)
        if sim_grouped
        else np.zeros((0,), dtype=np.float32)
        for name in feature_scales
    }
    result: Dict[str, object] = {
        "overall": _build_payload(overall_real, overall_sim),
        "by_scene": {},
    }
    for scene in sorted(set(real_grouped.keys()) & set(sim_grouped.keys())):
        result["by_scene"][scene] = _build_payload(real_grouped[scene], sim_grouped[scene])
    return result


def _evaluate(
    real_samples: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    sim_samples: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    spec: CanonicalLidarSpec,
    js_bins: int,
    range_bands: Sequence[Tuple[float, float]],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    thresholds = dict(thresholds or {})
    thresholds = {
        "valid_ratio_mae_max": float(thresholds.get("valid_ratio_mae_max", 0.10)),
        "wasserstein_median_max": float(thresholds.get("wasserstein_median_max", 0.08)),
        "wasserstein_p95_max": float(thresholds.get("wasserstein_p95_max", 0.20)),
        "scene_js_divergence_max": float(thresholds.get("scene_js_divergence_max", 0.15)),
    }

    result: Dict[str, object] = {
        "thresholds": thresholds,
        "overall": None,
        "by_scene": {},
        "bands": {},
        "counts": {
            "real_samples": int(len(real_samples)),
            "sim_samples": int(len(sim_samples)),
        },
        "compare_max_range_m": float(spec.max_range_m),
    }
    band_specs: List[Tuple[str, float, float]] = [
        (_range_band_name(0.0, float(spec.max_range_m), overall=True), 0.0, float(spec.max_range_m))
    ]
    seen_band_names = {band_specs[0][0]}
    for band_min_m, band_max_m in range_bands:
        band_name = _range_band_name(band_min_m, band_max_m, overall=False)
        if band_name in seen_band_names:
            continue
        seen_band_names.add(band_name)
        band_specs.append((band_name, float(band_min_m), float(band_max_m)))

    for band_name, band_min_m, band_max_m in band_specs:
        band_result = _evaluate_single_band(
            real_samples=real_samples,
            sim_samples=sim_samples,
            band_min_m=band_min_m,
            band_max_m=band_max_m,
            js_bins=js_bins,
        )
        overall_summary = dict(band_result["overall"]["summary"])
        overall_summary["pass"] = bool(
            overall_summary["valid_ratio_mae"] <= thresholds["valid_ratio_mae_max"]
            and overall_summary["wasserstein_median"] <= thresholds["wasserstein_median_max"]
            and overall_summary["wasserstein_p95"] <= thresholds["wasserstein_p95_max"]
            and overall_summary["scene_js_divergence"] <= thresholds["scene_js_divergence_max"]
        )
        band_result["overall"]["summary"] = overall_summary
        for scene, payload in list(band_result["by_scene"].items()):
            summary = dict(payload["summary"])
            summary["pass"] = bool(
                summary["valid_ratio_mae"] <= thresholds["valid_ratio_mae_max"]
                and summary["wasserstein_median"] <= thresholds["wasserstein_median_max"]
                and summary["wasserstein_p95"] <= thresholds["wasserstein_p95_max"]
                and summary["scene_js_divergence"] <= thresholds["scene_js_divergence_max"]
            )
            payload["summary"] = summary
        result["bands"][band_name] = band_result

    overall_band_name = _range_band_name(0.0, float(spec.max_range_m), overall=True)
    result["overall"] = result["bands"][overall_band_name]["overall"]
    result["by_scene"] = result["bands"][overall_band_name]["by_scene"]
    result["feature_alignment"] = _evaluate_feature_alignment(
        real_samples=real_samples,
        sim_samples=sim_samples,
        spec=spec,
    )
    return result


def _print_summary(result: Dict[str, object]) -> None:
    overall = result["overall"]["summary"]
    print("LiDAR Domain Gap Summary")
    print(f"  real_samples={result['counts']['real_samples']} sim_samples={result['counts']['sim_samples']}")
    print(
        "  overall: "
        f"valid_mae={overall['valid_ratio_mae']:.4f} "
        f"wd_med={overall['wasserstein_median']:.4f} "
        f"wd_p95={overall['wasserstein_p95']:.4f} "
        f"js={overall['scene_js_divergence']:.4f} "
        f"pass={int(bool(overall['pass']))}"
    )
    bands = result.get("bands", {})
    if bands:
        print("  by-band:")
        for band_name, payload in bands.items():
            summary = payload.get("overall", {}).get("summary", {})
            print(
                f"    {band_name}: valid_mae={summary.get('valid_ratio_mae', float('nan')):.4f} "
                f"wd_med={summary.get('wasserstein_median', float('nan')):.4f} "
                f"wd_p95={summary.get('wasserstein_p95', float('nan')):.4f} "
                f"js={summary.get('scene_js_divergence', float('nan')):.4f} "
                f"pass={int(bool(summary.get('pass', False)))}"
            )
    by_scene = result.get("by_scene", {})
    if by_scene:
        print("  per-scene:")
        for scene, payload in by_scene.items():
            summary = payload["summary"]
            print(
                f"    {scene}: valid_mae={summary['valid_ratio_mae']:.4f} "
                f"wd_med={summary['wasserstein_median']:.4f} "
                f"wd_p95={summary['wasserstein_p95']:.4f} "
                f"js={summary['scene_js_divergence']:.4f} "
                f"pass={int(bool(summary['pass']))} "
                f"(real={payload['real_count']} sim={payload['sim_count']})"
            )
    feature_alignment = dict(result.get("feature_alignment", {}) or {})
    feature_overall = dict((feature_alignment.get("overall", {}) or {}).get("features", {}) or {})
    if feature_overall:
        print("  deployment-features:")
        for feature_name in ("front_min", "left_gap", "right_gap", "valid_ratio"):
            payload = dict(feature_overall.get(feature_name, {}) or {})
            print(
                f"    {feature_name}: "
                f"mean_abs_diff={payload.get('mean_abs_diff', float('nan')):.4f} "
                f"p50_abs_diff={payload.get('p50_abs_diff', float('nan')):.4f} "
                f"p95_abs_diff={payload.get('p95_abs_diff', float('nan')):.4f} "
                f"wd={payload.get('wasserstein', float('nan')):.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate canonical LiDAR sim-real domain gap")
    parser.add_argument("--real-paths", nargs="+", required=True)
    parser.add_argument("--sim-paths", nargs="+", required=True)
    parser.add_argument("--num-sectors", type=int, default=36)
    parser.add_argument("--fov-deg", type=float, default=180.0)
    parser.add_argument("--compare-max-range-m", "--max-range-m", dest="max_range_m", type=float, default=6.0)
    parser.add_argument("--near-clip-m", type=float, default=0.18)
    parser.add_argument(
        "--range-bands",
        nargs="*",
        default=["0,5", "5,max"],
        help="band specs as 'lo,hi'; use 'max' for compare_max_range_m",
    )
    parser.add_argument("--js-bins", type=int, default=24)
    parser.add_argument("--valid-ratio-mae-max", type=float, default=0.10)
    parser.add_argument("--wasserstein-median-max", type=float, default=0.08)
    parser.add_argument("--wasserstein-p95-max", type=float, default=0.20)
    parser.add_argument("--scene-js-divergence-max", type=float, default=0.15)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    range_bands: List[Tuple[float, float]] = []
    for raw in list(args.range_bands or []):
        if not str(raw).strip():
            continue
        lo_raw, hi_raw = [part.strip().lower() for part in str(raw).split(",", 1)]
        band_min_m = float(lo_raw)
        band_max_m = float(args.max_range_m) if hi_raw == "max" else float(hi_raw)
        if not (band_min_m >= 0.0 and band_max_m > band_min_m):
            raise ValueError(f"invalid range band: {raw}")
        if band_max_m - float(args.max_range_m) > 1e-6:
            raise ValueError(f"range band exceeds compare_max_range_m: {raw}")
        range_bands.append((band_min_m, band_max_m))

    spec = CanonicalLidarSpec(
        num_sectors=int(args.num_sectors),
        fov_deg=float(args.fov_deg),
        max_range_m=float(args.max_range_m),
        near_clip_m=float(args.near_clip_m),
        invalid_fill_m=float(args.max_range_m),
    )

    real_samples = _load_samples(args.real_paths, spec=spec, kind="real")
    sim_samples = _load_samples(args.sim_paths, spec=spec, kind="sim")
    if not real_samples:
        raise RuntimeError("no usable real LiDAR samples found")
    if not sim_samples:
        raise RuntimeError("no usable sim LiDAR samples found")

    result = _evaluate(
        real_samples,
        sim_samples,
        spec=spec,
        js_bins=int(args.js_bins),
        range_bands=range_bands,
        thresholds={
            "valid_ratio_mae_max": float(args.valid_ratio_mae_max),
            "wasserstein_median_max": float(args.wasserstein_median_max),
            "wasserstein_p95_max": float(args.wasserstein_p95_max),
            "scene_js_divergence_max": float(args.scene_js_divergence_max),
        },
    )
    _print_summary(result)

    if args.output_json:
        out_path = Path(args.output_json).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
