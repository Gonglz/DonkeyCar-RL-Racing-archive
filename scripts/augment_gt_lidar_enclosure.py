#!/usr/bin/env python3
"""
Build a GT canonical-LiDAR enclosure-augmented JSONL using a WS canonical prior.

This is an offline experiment helper for Phase-0 analysis. It does not touch the
training runtime. The goal is to answer:

"If GT keeps its open-track geometry, but we inject a WS-style near-field shell
 into side sectors at the canonical representation level, how much can the
 sim-real domain gap drop?"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from module.lidar import CanonicalLidarSpec  # noqa: E402
from scripts.eval_lidar_domain_gap import (  # noqa: E402
    _extract_sim_from_obj,
    _get_scene_name,
    _iter_input_files,
    _iter_jsonl,
)


def _load_canonical_samples(paths: Sequence[str], spec: CanonicalLidarSpec) -> List[Tuple[dict, np.ndarray, np.ndarray]]:
    samples: List[Tuple[dict, np.ndarray, np.ndarray]] = []
    for path in _iter_input_files(paths):
        if path.suffix.lower() != ".jsonl":
            continue
        for obj in _iter_jsonl(path):
            maybe = _extract_sim_from_obj(obj, spec)
            if maybe is None:
                continue
            samples.append((obj, maybe[0].astype(np.float32), maybe[1].astype(np.float32)))
    return samples


def _fit_sector_prior(samples: Sequence[Tuple[dict, np.ndarray, np.ndarray]], spec: CanonicalLidarSpec) -> Dict[str, np.ndarray]:
    if not samples:
        raise ValueError("no samples available to fit sector prior")
    ranges = np.stack([r for _, r, _ in samples], axis=0)
    valid = np.stack([v for _, _, v in samples], axis=0)
    valid_rate = np.mean(valid, axis=0).astype(np.float32)
    q20 = np.full((spec.num_sectors,), spec.max_range_m, dtype=np.float32)
    q50 = np.full((spec.num_sectors,), spec.max_range_m, dtype=np.float32)
    q80 = np.full((spec.num_sectors,), spec.max_range_m, dtype=np.float32)
    std = np.zeros((spec.num_sectors,), dtype=np.float32)
    for idx in range(spec.num_sectors):
        vals = ranges[:, idx][valid[:, idx] > 0.5]
        if vals.size == 0:
            continue
        q20[idx] = np.float32(np.quantile(vals, 0.20))
        q50[idx] = np.float32(np.quantile(vals, 0.50))
        q80[idx] = np.float32(np.quantile(vals, 0.80))
        std[idx] = np.float32(np.std(vals))
    return {
        "valid_rate": valid_rate,
        "q20": q20,
        "q50": q50,
        "q80": q80,
        "std": std,
    }


def _sector_weight(index: int) -> float:
    # Strong enclosure prior on far-side sectors, weaker in semi-side sectors,
    # no injection in the front-center view.
    if 0 <= index <= 7 or 28 <= index <= 35:
        return 1.0
    if 8 <= index <= 11 or 24 <= index <= 27:
        return 0.5
    return 0.0


def _augment_one(
    lidar_range: np.ndarray,
    lidar_valid: np.ndarray,
    prior: Dict[str, np.ndarray],
    spec: CanonicalLidarSpec,
    rng: np.random.Generator,
    inject_valid_thresh: float,
    far_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_range = np.asarray(lidar_range, dtype=np.float32).copy()
    out_valid = np.asarray(lidar_valid, dtype=np.float32).copy()
    injected = np.zeros_like(out_valid, dtype=np.float32)

    for idx in range(spec.num_sectors):
        weight = _sector_weight(idx)
        if weight <= 0.0:
            continue

        prior_valid = float(prior["valid_rate"][idx])
        if prior_valid < inject_valid_thresh:
            continue

        if rng.random() > weight:
            continue

        # Only inject shell structure into empty or clearly-too-far sectors.
        cur_valid = float(out_valid[idx]) > 0.5
        cur_range = float(out_range[idx])
        prior_target = float(prior["q50"][idx])
        is_far = (not cur_valid) or (cur_range >= max(prior_target * (1.0 + far_ratio), spec.max_range_m * 0.85))
        if not is_far:
            continue

        # Sample around the WS median so the result is not a hard constant wall.
        spread = max(0.02, float(prior["std"][idx]) * 0.35)
        sample = float(rng.normal(loc=prior_target, scale=spread))
        sample = float(np.clip(sample, spec.near_clip_m, min(float(prior["q80"][idx]), spec.max_range_m)))

        out_range[idx] = min(cur_range if cur_valid else spec.max_range_m, sample)
        out_valid[idx] = 1.0
        injected[idx] = 1.0

    return out_range, out_valid, injected


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply WS-style enclosure prior to GT canonical LiDAR")
    parser.add_argument("--ws-paths", nargs="+", required=True)
    parser.add_argument("--gt-paths", nargs="+", required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--num-sectors", type=int, default=36)
    parser.add_argument("--max-range-m", type=float, default=6.0)
    parser.add_argument("--near-clip-m", type=float, default=0.18)
    parser.add_argument("--inject-valid-thresh", type=float, default=0.10)
    parser.add_argument("--far-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    spec = CanonicalLidarSpec(
        num_sectors=int(args.num_sectors),
        fov_deg=180.0,
        max_range_m=float(args.max_range_m),
        near_clip_m=float(args.near_clip_m),
        invalid_fill_m=float(args.max_range_m),
    )

    ws_samples = _load_canonical_samples(args.ws_paths, spec)
    gt_samples = _load_canonical_samples(args.gt_paths, spec)
    if not ws_samples:
        raise RuntimeError("no WS samples loaded")
    if not gt_samples:
        raise RuntimeError("no GT samples loaded")

    prior = _fit_sector_prior(ws_samples, spec)
    rng = np.random.default_rng(int(args.seed))
    output_path = Path(args.output_jsonl).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    injected_counts = np.zeros((spec.num_sectors,), dtype=np.int64)
    total = 0
    with output_path.open("w", encoding="utf-8") as f:
        for obj, lidar_range, lidar_valid in gt_samples:
            aug_range, aug_valid, injected = _augment_one(
                lidar_range=lidar_range,
                lidar_valid=lidar_valid,
                prior=prior,
                spec=spec,
                rng=rng,
                inject_valid_thresh=float(args.inject_valid_thresh),
                far_ratio=float(args.far_ratio),
            )
            injected_counts += injected.astype(np.int64)
            total += 1
            payload = {
                "scene_key": _get_scene_name(obj, default="generated_track"),
                "domain": "gt_shell_aug",
                "canonical_lidar_range": aug_range.astype(float).tolist(),
                "canonical_lidar_valid": aug_valid.astype(float).tolist(),
                "augment_meta": {
                    "source": "ws_enclosure_prior",
                    "inject_valid_thresh": float(args.inject_valid_thresh),
                    "far_ratio": float(args.far_ratio),
                    "injected_sector_count": int(np.sum(injected)),
                },
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    summary = {
        "output_jsonl": str(output_path),
        "ws_samples": int(len(ws_samples)),
        "gt_samples": int(len(gt_samples)),
        "prior_valid_rate_mean": float(np.mean(prior["valid_rate"])),
        "injected_sector_rate": (injected_counts / max(total, 1)).astype(float).tolist(),
        "inject_valid_thresh": float(args.inject_valid_thresh),
        "far_ratio": float(args.far_ratio),
        "seed": int(args.seed),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
