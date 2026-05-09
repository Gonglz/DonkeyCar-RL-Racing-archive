#!/usr/bin/env python3
"""
Train temporal obstacle-context estimator from sequence-style exported supervision.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import traceback
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from module.obstacle_context_learned import ObstacleContextTemporalNet, obstacle_context_loss


class ObstacleContextSequenceDataset(Dataset):
    def __init__(self, npz_path: str, seq_len: int = 16):
        data = np.load(npz_path)
        self.image = data["image"]
        self.state7 = data["state7"]
        self.target_present = np.asarray(data["target_present"], dtype=np.float32)
        self.target_longitudinal = np.asarray(data["target_longitudinal"], dtype=np.float32)
        self.target_lateral = np.asarray(data["target_lateral"], dtype=np.float32)
        self.target_dist = np.asarray(data["target_dist"], dtype=np.float32)
        self.episode_id = np.asarray(data["episode_id"], dtype=np.int64)
        self.step_in_episode = np.asarray(data["step_in_episode"], dtype=np.int64)
        self.seq_len = int(seq_len)
        if self.image.shape[0] != self.state7.shape[0]:
            raise ValueError("dataset image/state size mismatch")

    def __len__(self) -> int:
        return int(self.image.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep = int(self.episode_id[idx])
        start = idx
        while start > 0 and int(self.episode_id[start - 1]) == ep and (idx - start + 1) < self.seq_len:
            start -= 1
        length = idx - start + 1
        pad = self.seq_len - length
        c, h, w = self.image.shape[1:]
        image_seq = np.zeros((self.seq_len, c, h, w), dtype=np.float32)
        state_seq = np.zeros((self.seq_len, self.state7.shape[1]), dtype=np.float32)
        image_slice = np.asarray(self.image[start : idx + 1], dtype=np.float32)
        if self.image.dtype == np.uint8:
            image_slice = image_slice / 255.0
        image_seq[pad:] = image_slice
        state_seq[pad:] = np.asarray(self.state7[start : idx + 1], dtype=np.float32)
        return {
            "image_seq": torch.from_numpy(image_seq),
            "state7_seq": torch.from_numpy(state_seq),
            "length": torch.tensor(length, dtype=torch.int64),
            "target_present": torch.tensor(self.target_present[idx], dtype=torch.float32),
            "target_longitudinal": torch.tensor(self.target_longitudinal[idx], dtype=torch.float32),
            "target_lateral": torch.tensor(self.target_lateral[idx], dtype=torch.float32),
            "target_dist": torch.tensor(self.target_dist[idx], dtype=torch.float32),
        }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_dataset_by_episode(
    dataset: ObstacleContextSequenceDataset, val_ratio: float, seed: int
) -> Tuple[Subset, Subset]:
    unique_eps = np.unique(dataset.episode_id)
    if unique_eps.size < 2:
        raise ValueError("need at least 2 episodes for train/val split")
    rng = np.random.RandomState(seed)
    episode_positive = {
        int(ep): bool(np.any(dataset.target_present[dataset.episode_id == ep] > 0.5))
        for ep in unique_eps.tolist()
    }
    pos_eps = np.asarray([int(ep) for ep in unique_eps.tolist() if episode_positive[int(ep)]], dtype=np.int64)
    zero_eps = np.asarray([int(ep) for ep in unique_eps.tolist() if not episode_positive[int(ep)]], dtype=np.int64)
    rng.shuffle(pos_eps)
    rng.shuffle(zero_eps)

    val_ep_count = int(round(unique_eps.size * val_ratio))
    val_ep_count = int(np.clip(val_ep_count, 1, max(1, unique_eps.size - 1)))
    val_eps: List[int] = []

    # Prefer at least one positive episode in validation whenever possible,
    # while also keeping at least one positive episode in train.
    if pos_eps.size >= 2 and val_ep_count > 0:
        val_eps.append(int(pos_eps[0]))
    remaining = val_ep_count - len(val_eps)

    if remaining > 0 and zero_eps.size > 0:
        take = min(remaining, zero_eps.size)
        val_eps.extend(zero_eps[:take].tolist())
        remaining = val_ep_count - len(val_eps)

    if remaining > 0:
        remaining_pos = [int(ep) for ep in pos_eps.tolist() if int(ep) not in val_eps]
        val_eps.extend(remaining_pos[:remaining])

    val_eps = set(val_eps[:val_ep_count])
    train_idx = [i for i, ep in enumerate(dataset.episode_id.tolist()) if ep not in val_eps]
    val_idx = [i for i, ep in enumerate(dataset.episode_id.tolist()) if ep in val_eps]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def _batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device=device, non_blocking=True) for k, v in batch.items()}


def _binary_metrics_from_probs(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    pred = probs >= float(threshold)
    positive = targets >= 0.5
    tp = float(np.sum(pred & positive))
    fp = float(np.sum(pred & (~positive)))
    fn = float(np.sum((~pred) & positive))
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = (2.0 * precision * recall) / max(1e-6, precision + recall)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def _best_threshold_from_probs(
    probs: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    if probs.size == 0:
        return _binary_metrics_from_probs(probs, targets, 0.5)
    best = None
    for thr in np.linspace(0.05, 0.95, 19):
        cur = _binary_metrics_from_probs(probs, targets, float(thr))
        if best is None or cur["f1"] > best["f1"] or (
            abs(cur["f1"] - best["f1"]) < 1e-9 and cur["precision"] > best["precision"]
        ):
            best = cur
    return best if best is not None else _binary_metrics_from_probs(probs, targets, 0.5)


def _compute_epoch(
    model: ObstacleContextTemporalNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    present_pos_weight: torch.Tensor,
    present_focal_gamma: float = 0.0,
    max_batches: int | None = None,
    progress_every: int = 0,
    progress_path: str | None = None,
    phase: str = "train",
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)

    stats_acc: Dict[str, float] = {
        "loss_total": 0.0,
        "loss_present": 0.0,
        "loss_longitudinal": 0.0,
        "loss_lateral": 0.0,
        "loss_dist": 0.0,
        "positive_rate": 0.0,
        "present_tp": 0.0,
        "present_fp": 0.0,
        "present_fn": 0.0,
        "geom_mae_longitudinal": 0.0,
        "geom_mae_lateral": 0.0,
        "geom_mae_dist": 0.0,
        "positive_count": 0.0,
        "samples": 0.0,
    }
    present_probs_all: List[np.ndarray] = []
    targets_all: List[np.ndarray] = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = _batch_to_device(batch, device)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        pred = model(
            image_seq=batch["image_seq"],
            state7_seq=batch["state7_seq"],
            lengths=batch["length"],
        )
        loss, loss_stats = obstacle_context_loss(
            pred=pred,
            target_present=batch["target_present"],
            target_longitudinal=batch["target_longitudinal"],
            target_lateral=batch["target_lateral"],
            target_dist=batch["target_dist"],
            present_pos_weight=present_pos_weight,
            present_focal_gamma=float(present_focal_gamma),
        )
        if train_mode:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            present_probs = torch.sigmoid(pred["present_logit"]).detach().cpu().numpy().astype(np.float32)
            target_present = batch["target_present"]
            target_present_np = target_present.detach().cpu().numpy().astype(np.float32)
            pred_present = (torch.from_numpy(present_probs) >= 0.5).float()
            positive = target_present > 0.5
            tp = float(torch.sum((pred_present > 0.5) & positive).detach().cpu().item())
            fp = float(torch.sum((pred_present > 0.5) & (~positive)).detach().cpu().item())
            fn = float(torch.sum((pred_present <= 0.5) & positive).detach().cpu().item())
            pos_count = float(torch.sum(positive).detach().cpu().item())
            if torch.any(positive):
                mae_long = float(torch.mean(torch.abs(pred["longitudinal"][positive] - batch["target_longitudinal"][positive])).detach().cpu().item())
                mae_lat = float(torch.mean(torch.abs(pred["lateral"][positive] - batch["target_lateral"][positive])).detach().cpu().item())
                mae_dist = float(torch.mean(torch.abs(pred["dist"][positive] - batch["target_dist"][positive])).detach().cpu().item())
            else:
                mae_long = 0.0
                mae_lat = 0.0
                mae_dist = 0.0
            present_probs_all.append(present_probs.reshape(-1))
            targets_all.append(target_present_np.reshape(-1))

        batch_size = float(batch["image_seq"].shape[0])
        stats_acc["samples"] += batch_size
        for key in ("loss_total", "loss_present", "loss_longitudinal", "loss_lateral", "loss_dist", "positive_rate"):
            stats_acc[key] += float(loss_stats[key]) * batch_size
        stats_acc["present_tp"] += tp
        stats_acc["present_fp"] += fp
        stats_acc["present_fn"] += fn
        stats_acc["geom_mae_longitudinal"] += mae_long * max(1.0, pos_count)
        stats_acc["geom_mae_lateral"] += mae_lat * max(1.0, pos_count)
        stats_acc["geom_mae_dist"] += mae_dist * max(1.0, pos_count)
        stats_acc["positive_count"] += pos_count

        if progress_every > 0 and ((batch_idx + 1) % int(progress_every) == 0):
            progress_row = {
                "phase": phase,
                "batch": int(batch_idx + 1),
                "samples": float(stats_acc["samples"]),
                "positive_count": float(stats_acc["positive_count"]),
                "present_tp": float(stats_acc["present_tp"]),
                "present_fp": float(stats_acc["present_fp"]),
                "present_fn": float(stats_acc["present_fn"]),
                "loss_total": float(stats_acc["loss_total"] / max(1.0, stats_acc["samples"])),
            }
            if progress_path:
                with open(progress_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(progress_row, ensure_ascii=False) + "\n")
            print(json.dumps(progress_row, ensure_ascii=False), flush=True)

    samples = max(1.0, stats_acc["samples"])
    positive_count = max(1.0, stats_acc["positive_count"])
    precision = stats_acc["present_tp"] / max(1.0, stats_acc["present_tp"] + stats_acc["present_fp"])
    recall = stats_acc["present_tp"] / max(1.0, stats_acc["present_tp"] + stats_acc["present_fn"])
    f1 = (2.0 * precision * recall) / max(1e-6, precision + recall)
    probs_np = np.concatenate(present_probs_all, axis=0) if present_probs_all else np.zeros((0,), dtype=np.float32)
    targets_np = np.concatenate(targets_all, axis=0) if targets_all else np.zeros((0,), dtype=np.float32)
    best_thr = _best_threshold_from_probs(probs_np, targets_np)

    return {
        "loss_total": stats_acc["loss_total"] / samples,
        "loss_present": stats_acc["loss_present"] / samples,
        "loss_longitudinal": stats_acc["loss_longitudinal"] / samples,
        "loss_lateral": stats_acc["loss_lateral"] / samples,
        "loss_dist": stats_acc["loss_dist"] / samples,
        "positive_rate": stats_acc["positive_rate"] / samples,
        "present_precision": precision,
        "present_recall": recall,
        "present_f1": f1,
        "present_best_threshold": float(best_thr["threshold"]),
        "present_best_precision": float(best_thr["precision"]),
        "present_best_recall": float(best_thr["recall"]),
        "present_best_f1": float(best_thr["f1"]),
        "geom_mae_longitudinal": stats_acc["geom_mae_longitudinal"] / positive_count,
        "geom_mae_lateral": stats_acc["geom_mae_lateral"] / positive_count,
        "geom_mae_dist": stats_acc["geom_mae_dist"] / positive_count,
        "positive_count": stats_acc["positive_count"],
        "samples": stats_acc["samples"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--present-focal-gamma", type=float, default=0.0)
    ap.add_argument("--max-train-batches", type=int, default=0)
    ap.add_argument("--max-val-batches", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=0)
    args = ap.parse_args()

    _seed_everything(int(args.seed))
    os.makedirs(args.save_dir, exist_ok=True)
    run_start_path = os.path.join(args.save_dir, "run_start.json")
    with open(run_start_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    try:
        dataset = ObstacleContextSequenceDataset(args.dataset, seq_len=int(args.seq_len))
        train_set, val_set = _split_dataset_by_episode(dataset, val_ratio=float(args.val_ratio), seed=int(args.seed))
        train_loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True, num_workers=0, drop_last=False)
        val_loader = DataLoader(val_set, batch_size=int(args.batch_size), shuffle=False, num_workers=0, drop_last=False)

        device_str = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else str(args.device))
        device = torch.device(device_str)

        model = ObstacleContextTemporalNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))

        target_present = np.asarray(dataset.target_present, dtype=np.float32)
        positive_count = float(np.sum(target_present >= 0.5))
        negative_count = float(target_present.shape[0] - positive_count)
        pos_weight_value = max(1.0, negative_count / max(1.0, positive_count))
        present_pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)

        history: List[Dict[str, Any]] = []
        best_val_f1 = -math.inf
        best_path = os.path.join(args.save_dir, "best_model.pt")
        epoch_metrics_path = os.path.join(args.save_dir, "epoch_metrics.jsonl")
        progress_path = os.path.join(args.save_dir, "progress.jsonl")

        for epoch in range(1, int(args.epochs) + 1):
            train_stats = _compute_epoch(
                model,
                train_loader,
                device,
                optimizer,
                present_pos_weight,
                present_focal_gamma=float(args.present_focal_gamma),
                max_batches=(int(args.max_train_batches) if int(args.max_train_batches) > 0 else None),
                progress_every=int(args.progress_every),
                progress_path=progress_path,
                phase=f"train_e{epoch}",
            )
            with torch.no_grad():
                val_stats = _compute_epoch(
                    model,
                    val_loader,
                    device,
                    None,
                    present_pos_weight,
                    present_focal_gamma=float(args.present_focal_gamma),
                    max_batches=(int(args.max_val_batches) if int(args.max_val_batches) > 0 else None),
                    progress_every=int(args.progress_every),
                    progress_path=progress_path,
                    phase=f"val_e{epoch}",
                )

            row = {"epoch": epoch, "train": train_stats, "val": val_stats}
            history.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            with open(epoch_metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if val_stats["present_f1"] > best_val_f1:
                best_val_f1 = float(val_stats["present_f1"])
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val": val_stats,
                        "train": train_stats,
                        "dataset": os.path.abspath(args.dataset),
                        "seq_len": int(args.seq_len),
                    },
                    best_path,
                )

        final_path = os.path.join(args.save_dir, "final_model.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": int(args.epochs),
                "dataset": os.path.abspath(args.dataset),
                "seq_len": int(args.seq_len),
                "pos_weight": float(pos_weight_value),
            },
            final_path,
        )

        summary = {
            "dataset": os.path.abspath(args.dataset),
            "save_dir": os.path.abspath(args.save_dir),
            "device": device_str,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "seq_len": int(args.seq_len),
            "present_focal_gamma": float(args.present_focal_gamma),
            "dataset_frames": int(len(dataset)),
            "dataset_visible_positive_rate": float(np.mean(target_present)) if target_present.size > 0 else 0.0,
            "present_pos_weight": float(pos_weight_value),
            "best_val_f1": float(best_val_f1),
            "best_model": best_path,
            "final_model": final_path,
            "epoch_metrics": epoch_metrics_path,
            "history": history,
        }
        summary_path = os.path.join(args.save_dir, "train_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"[saved] best={best_path}", flush=True)
        print(f"[saved] final={final_path}", flush=True)
        print(f"[saved] summary={summary_path}", flush=True)
    except Exception:
        error_path = os.path.join(args.save_dir, "run_error.txt")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
