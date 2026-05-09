#!/usr/bin/env python3
"""
Merge obstacle-context datasets while preserving per-episode temporal structure.

Typical usage:
  - primary: interaction-heavy GT dataset
  - supplement: mostly-negative calibration dataset
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {k: np.asarray(data[k]) for k in data.files}


def _split_episodes(data: Dict[str, np.ndarray]) -> List[Dict[str, np.ndarray]]:
    episode_id = np.asarray(data["episode_id"], dtype=np.int64)
    episodes: List[Dict[str, np.ndarray]] = []
    for ep in np.unique(episode_id).tolist():
        mask = episode_id == int(ep)
        episodes.append({k: np.asarray(v[mask]) for k, v in data.items()})
    return episodes


def _episode_stats(ep: Dict[str, np.ndarray]) -> Dict[str, float]:
    target_present = np.asarray(ep["target_present"], dtype=np.float32)
    scene_present = np.asarray(ep.get("target_scene_present", np.zeros_like(target_present)), dtype=np.float32)
    return {
        "frames": float(target_present.shape[0]),
        "visible_present_rate": float(np.mean(target_present)) if target_present.size > 0 else 0.0,
        "scene_present_rate": float(np.mean(scene_present)) if scene_present.size > 0 else 0.0,
    }


def _renumber_and_concat(episodes: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not episodes:
        raise ValueError("no episodes to merge")
    keys = sorted({k for ep in episodes for k in ep.keys()})
    out_parts: Dict[str, List[np.ndarray]] = {k: [] for k in keys}
    next_ep = 0
    for ep in episodes:
        cur = {k: np.asarray(v) for k, v in ep.items()}
        n = int(cur["episode_id"].shape[0])
        cur["episode_id"] = np.full((n,), next_ep, dtype=np.int64)
        cur["step_in_episode"] = np.arange(n, dtype=np.int64)
        for k in keys:
            out_parts[k].append(np.asarray(cur[k]))
        next_ep += 1
    return {k: np.concatenate(v, axis=0) for k, v in out_parts.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True)
    ap.add_argument("--supplement", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-supplement-episodes", type=int, default=0)
    ap.add_argument("--sort-supplement-by", choices=("visible_present_rate", "scene_present_rate", "frames"), default="visible_present_rate")
    ap.add_argument("--ascending", action="store_true")
    args = ap.parse_args()

    primary = _load_npz(args.primary)
    supplement = _load_npz(args.supplement)

    primary_eps = _split_episodes(primary)
    supplement_eps = _split_episodes(supplement)

    supplement_ranked = []
    for ep in supplement_eps:
        stats = _episode_stats(ep)
        supplement_ranked.append((stats, ep))
    supplement_ranked.sort(key=lambda x: float(x[0][args.sort_supplement_by]), reverse=(not args.ascending))

    if int(args.max_supplement_episodes) > 0:
        supplement_ranked = supplement_ranked[: int(args.max_supplement_episodes)]

    merged_eps = list(primary_eps) + [ep for _stats, ep in supplement_ranked]
    merged = _renumber_and_concat(merged_eps)

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **merged)

    target_present = np.asarray(merged["target_present"], dtype=np.float32)
    scene_present = np.asarray(merged.get("target_scene_present", np.zeros_like(target_present)), dtype=np.float32)
    summary: Dict[str, Any] = {
        "primary": os.path.abspath(args.primary),
        "supplement": os.path.abspath(args.supplement),
        "output": output_path,
        "primary_episode_count": int(len(primary_eps)),
        "supplement_episode_count_total": int(len(supplement_eps)),
        "supplement_episode_count_used": int(len(supplement_ranked)),
        "sort_supplement_by": str(args.sort_supplement_by),
        "ascending": bool(args.ascending),
        "frames": int(target_present.shape[0]),
        "episode_count": int(len(merged_eps)),
        "visible_present_rate": float(np.mean(target_present)) if target_present.size > 0 else 0.0,
        "scene_present_rate": float(np.mean(scene_present)) if scene_present.size > 0 else 0.0,
    }
    meta_path = os.path.splitext(output_path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[saved] dataset={output_path}")
    print(f"[saved] meta={meta_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
