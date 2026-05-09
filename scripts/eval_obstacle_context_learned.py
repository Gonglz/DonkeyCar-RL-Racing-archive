#!/usr/bin/env python3
"""
Evaluate learned obstacle-context estimators on sampled sequence datasets.

This script is intentionally lightweight:
- loads an exported `.npz` sequence dataset
- loads a temporal checkpoint (`best_model.pt`)
- evaluates on train/val/all split by episode
- reports overall metrics, interaction-zone metrics, and pre-overtake metrics
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from module.obstacle_context_learned import ObstacleContextTemporalNet
from scripts.train_obstacle_context_temporal import (
    ObstacleContextSequenceDataset,
    _batch_to_device,
    _best_threshold_from_probs,
    _binary_metrics_from_probs,
    _split_dataset_by_episode,
)


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if float(den) != 0.0 else 0.0


def _mask_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    mask: np.ndarray,
) -> Dict[str, float]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    targets = np.asarray(targets, dtype=np.float32).reshape(-1)
    if mask.size == 0 or not np.any(mask):
        out = _binary_metrics_from_probs(np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), threshold)
        out["count"] = 0.0
        return out
    sub = _binary_metrics_from_probs(probs[mask], targets[mask], threshold)
    sub["count"] = float(np.sum(mask))
    return sub


def _compute_geom_mae(
    pred_long: np.ndarray,
    pred_lat: np.ndarray,
    pred_dist: np.ndarray,
    target_present: np.ndarray,
    target_long: np.ndarray,
    target_lat: np.ndarray,
    target_dist: np.ndarray,
    pred_present: np.ndarray | None = None,
) -> Dict[str, float]:
    positive = np.asarray(target_present, dtype=np.float32).reshape(-1) >= 0.5
    if pred_present is not None:
        positive = positive & (np.asarray(pred_present, dtype=bool).reshape(-1))
    count = int(np.sum(positive))
    if count <= 0:
        return {
            "count": 0.0,
            "longitudinal_mae": 0.0,
            "lateral_mae": 0.0,
            "dist_mae": 0.0,
        }
    return {
        "count": float(count),
        "longitudinal_mae": float(np.mean(np.abs(pred_long[positive] - target_long[positive]))),
        "lateral_mae": float(np.mean(np.abs(pred_lat[positive] - target_lat[positive]))),
        "dist_mae": float(np.mean(np.abs(pred_dist[positive] - target_dist[positive]))),
    }


def _contiguous_runs(mask: np.ndarray) -> List[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    runs: List[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = i
        elif (not flag) and start is not None:
            runs.append((int(start), int(i)))
            start = None
    if start is not None:
        runs.append((int(start), int(mask.shape[0])))
    return runs


def _max_consecutive_true(mask: np.ndarray) -> int:
    best = 0
    cur = 0
    for flag in np.asarray(mask, dtype=bool).reshape(-1).tolist():
        if flag:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def _episode_visibility_metrics(
    episode_id: np.ndarray,
    visible_mask: np.ndarray,
    pred_present: np.ndarray,
    stable_steps: int,
) -> Dict[str, float]:
    episode_id = np.asarray(episode_id, dtype=np.int64).reshape(-1)
    visible_mask = np.asarray(visible_mask, dtype=bool).reshape(-1)
    pred_present = np.asarray(pred_present, dtype=bool).reshape(-1)

    delays: List[float] = []
    run_count = 0
    detected_run_count = 0
    stable_run_count = 0
    stable_lengths: List[float] = []

    for ep in np.unique(episode_id).tolist():
        ep_mask = episode_id == int(ep)
        ep_visible = visible_mask[ep_mask]
        ep_pred = pred_present[ep_mask]
        for start, end in _contiguous_runs(ep_visible):
            run_count += 1
            run_pred = ep_pred[start:end]
            detect_idx = np.flatnonzero(run_pred)
            if detect_idx.size > 0:
                detected_run_count += 1
                delays.append(float(int(detect_idx[0])))
            max_stable = _max_consecutive_true(run_pred)
            stable_lengths.append(float(max_stable))
            if max_stable >= int(stable_steps):
                stable_run_count += 1

    return {
        "visible_run_count": float(run_count),
        "visible_run_detected_rate": _safe_div(detected_run_count, run_count),
        "visible_run_stable_rate": _safe_div(stable_run_count, run_count),
        "first_detect_delay_mean_steps": float(np.mean(delays)) if delays else -1.0,
        "first_detect_delay_median_steps": float(np.median(delays)) if delays else -1.0,
        "stable_detect_len_mean_steps": float(np.mean(stable_lengths)) if stable_lengths else 0.0,
        "stable_detect_len_max_steps": float(np.max(stable_lengths)) if stable_lengths else 0.0,
    }


def _pre_overtake_event_metrics(
    future_overtake_mask: np.ndarray,
    pred_present: np.ndarray,
    stable_steps: int,
) -> Dict[str, float]:
    future_overtake_mask = np.asarray(future_overtake_mask, dtype=bool).reshape(-1)
    pred_present = np.asarray(pred_present, dtype=bool).reshape(-1)
    if future_overtake_mask.size == 0:
        return {
            "count": 0.0,
            "frame_trigger_rate": 0.0,
            "stable_trigger_rate": 0.0,
        }
    masked_pred = pred_present[future_overtake_mask]
    stable_hits = 0
    if masked_pred.size >= int(stable_steps) and int(stable_steps) > 1:
        cur = 0
        stable_mask = np.zeros_like(masked_pred, dtype=bool)
        for i, flag in enumerate(masked_pred.tolist()):
            if flag:
                cur += 1
                if cur >= int(stable_steps):
                    stable_mask[i] = True
            else:
                cur = 0
        stable_hits = int(np.sum(stable_mask))
    else:
        stable_hits = int(np.sum(masked_pred))
    count = int(masked_pred.shape[0])
    return {
        "count": float(count),
        "frame_trigger_rate": _safe_div(np.sum(masked_pred), count),
        "stable_trigger_rate": _safe_div(stable_hits, count),
    }


def _collect_predictions(
    model: ObstacleContextTemporalNet,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    out: Dict[str, List[np.ndarray]] = {
        "present_prob": [],
        "pred_longitudinal": [],
        "pred_lateral": [],
        "pred_dist": [],
        "target_present": [],
        "target_longitudinal": [],
        "target_lateral": [],
        "target_dist": [],
    }
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            pred = model(
                image_seq=batch["image_seq"],
                state7_seq=batch["state7_seq"],
                lengths=batch["length"],
            )
            out["present_prob"].append(torch.sigmoid(pred["present_logit"]).cpu().numpy().astype(np.float32))
            out["pred_longitudinal"].append(pred["longitudinal"].cpu().numpy().astype(np.float32))
            out["pred_lateral"].append(pred["lateral"].cpu().numpy().astype(np.float32))
            out["pred_dist"].append(pred["dist"].cpu().numpy().astype(np.float32))
            out["target_present"].append(batch["target_present"].cpu().numpy().astype(np.float32))
            out["target_longitudinal"].append(batch["target_longitudinal"].cpu().numpy().astype(np.float32))
            out["target_lateral"].append(batch["target_lateral"].cpu().numpy().astype(np.float32))
            out["target_dist"].append(batch["target_dist"].cpu().numpy().astype(np.float32))
    return {k: np.concatenate(v, axis=0) if v else np.zeros((0,), dtype=np.float32) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=("all", "train", "val"), default="val")
    ap.add_argument("--val-ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--interaction-dist-max", type=float, default=2.5)
    ap.add_argument("--interaction-longitudinal-max", type=float, default=3.0)
    ap.add_argument("--stable-steps", type=int, default=3)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    seq_len = int(ckpt.get("seq_len", 16))

    dataset = ObstacleContextSequenceDataset(args.dataset, seq_len=seq_len)
    if args.split == "all":
        eval_set = dataset
    else:
        train_set, val_set = _split_dataset_by_episode(dataset, val_ratio=float(args.val_ratio), seed=int(args.seed))
        eval_set = train_set if args.split == "train" else val_set

    loader = DataLoader(eval_set, batch_size=int(args.batch_size), shuffle=False, num_workers=0, drop_last=False)
    device = torch.device(str(args.device))
    model = ObstacleContextTemporalNet().to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    pred = _collect_predictions(model, loader, device)
    probs = pred["present_prob"]
    targets = pred["target_present"]
    target_long = pred["target_longitudinal"]
    target_lat = pred["target_lateral"]
    target_dist = pred["target_dist"]

    default_metrics = _binary_metrics_from_probs(probs, targets, float(args.threshold))
    best_metrics = _best_threshold_from_probs(probs, targets)

    default_pred_present = probs >= float(args.threshold)
    best_pred_present = probs >= float(best_metrics["threshold"])

    geom_on_visible = _compute_geom_mae(
        pred["pred_longitudinal"],
        pred["pred_lateral"],
        pred["pred_dist"],
        targets,
        target_long,
        target_lat,
        target_dist,
        pred_present=None,
    )
    geom_on_tp_default = _compute_geom_mae(
        pred["pred_longitudinal"],
        pred["pred_lateral"],
        pred["pred_dist"],
        targets,
        target_long,
        target_lat,
        target_dist,
        pred_present=default_pred_present,
    )
    geom_on_tp_best = _compute_geom_mae(
        pred["pred_longitudinal"],
        pred["pred_lateral"],
        pred["pred_dist"],
        targets,
        target_long,
        target_lat,
        target_dist,
        pred_present=best_pred_present,
    )

    visible_mask = targets >= 0.5
    interaction_mask = (
        visible_mask
        & (target_dist > 0.0)
        & (target_dist <= float(args.interaction_dist_max))
        & (target_long >= 0.0)
        & (target_long <= float(args.interaction_longitudinal_max))
    )

    extra: Dict[str, Any] = {}
    raw = np.load(args.dataset)
    if args.split == "all":
        raw_episode_id = np.asarray(raw["episode_id"], dtype=np.int64)
        raw_step_in_episode = np.asarray(raw["step_in_episode"], dtype=np.int64)
    else:
        subset_indices = np.asarray(eval_set.indices, dtype=np.int64)  # type: ignore[attr-defined]
        raw_episode_id = np.asarray(raw["episode_id"], dtype=np.int64)[subset_indices]
        raw_step_in_episode = np.asarray(raw["step_in_episode"], dtype=np.int64)[subset_indices]

    extra["visibility_default"] = _episode_visibility_metrics(
        episode_id=raw_episode_id,
        visible_mask=visible_mask,
        pred_present=default_pred_present,
        stable_steps=int(args.stable_steps),
    )
    extra["visibility_best"] = _episode_visibility_metrics(
        episode_id=raw_episode_id,
        visible_mask=visible_mask,
        pred_present=best_pred_present,
        stable_steps=int(args.stable_steps),
    )

    if "future_overtake_success_any_h20" in raw:
        future_overtake = np.asarray(raw["future_overtake_success_any_h20"], dtype=np.float32)
        if args.split != "all":
            future_overtake = future_overtake[subset_indices]
        pre_overtake_mask = future_overtake >= 0.5
        extra["pre_overtake_default"] = _mask_metrics(probs, targets, float(args.threshold), pre_overtake_mask)
        extra["pre_overtake_best"] = _mask_metrics(probs, targets, float(best_metrics["threshold"]), pre_overtake_mask)
        extra["pre_overtake_rate"] = float(np.mean(pre_overtake_mask)) if pre_overtake_mask.size > 0 else 0.0
        extra["pre_overtake_event_default"] = _pre_overtake_event_metrics(
            future_overtake_mask=pre_overtake_mask,
            pred_present=default_pred_present,
            stable_steps=int(args.stable_steps),
        )
        extra["pre_overtake_event_best"] = _pre_overtake_event_metrics(
            future_overtake_mask=pre_overtake_mask,
            pred_present=best_pred_present,
            stable_steps=int(args.stable_steps),
        )

    result = {
        "dataset": os.path.abspath(args.dataset),
        "checkpoint": os.path.abspath(args.checkpoint),
        "split": str(args.split),
        "eval_samples": int(probs.shape[0]),
        "visible_positive_rate": float(np.mean(visible_mask)) if visible_mask.size > 0 else 0.0,
        "default_threshold": float(args.threshold),
        "default_metrics": default_metrics,
        "best_threshold_metrics": best_metrics,
        "geom_on_visible": geom_on_visible,
        "geom_on_tp_default": geom_on_tp_default,
        "geom_on_tp_best": geom_on_tp_best,
        "interaction_default": _mask_metrics(probs, targets, float(args.threshold), interaction_mask),
        "interaction_best": _mask_metrics(probs, targets, float(best_metrics["threshold"]), interaction_mask),
        "interaction_visible_rate": float(np.mean(interaction_mask)) if interaction_mask.size > 0 else 0.0,
        "extra": extra,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(args.out)


if __name__ == "__main__":
    main()
