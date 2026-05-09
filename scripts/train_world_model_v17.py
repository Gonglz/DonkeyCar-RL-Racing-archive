#!/usr/bin/env python3
"""
Train the V17 local world model in three stages.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from module.local_world_model_v17 import LocalWorldModelV17, local_world_model_loss


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WorldModelSequenceDataset(Dataset):
    def __init__(self, npz_path: str, seq_len: int = 4):
        data = np.load(npz_path)
        self.camera = np.asarray(data["camera"], dtype=np.float32) if "camera" in data.files else None
        self.ego8 = np.asarray(data["ego8"], dtype=np.float32)
        self.lidar = np.asarray(data["lidar"], dtype=np.float32)
        self.async_meta = np.asarray(data["async_meta"], dtype=np.float32)
        self.target_rel = np.asarray(data["target_rel"], dtype=np.float32)
        self.target_rel_mask = np.asarray(data["target_rel_mask"], dtype=np.float32)
        self.target_gap = np.asarray(data["target_gap"], dtype=np.float32)
        self.target_collision = np.asarray(data["target_collision"], dtype=np.float32)
        self.target_ttc = np.asarray(data["target_ttc"], dtype=np.float32)
        self.target_safety_valid = np.asarray(data["target_safety_valid"], dtype=np.float32)
        self.target_passable = np.asarray(data["target_passable"], dtype=np.float32)
        self.target_closing_rate = np.asarray(data["target_closing_rate"], dtype=np.float32)
        self.target_overtake_progress = np.asarray(data["target_overtake_progress"], dtype=np.float32)
        self.target_opportunity_valid = np.asarray(data["target_opportunity_valid"], dtype=np.float32)
        self.episode_id = np.asarray(data["episode_id"], dtype=np.int64).reshape(-1)
        self.step_in_episode = np.asarray(data["step_in_episode"], dtype=np.int64).reshape(-1)
        self.scene_id = np.asarray(data["scene_id"], dtype=np.int64).reshape(-1)
        self.seq_len = int(max(1, seq_len))
        self.size = int(self.ego8.shape[0])
        self.camera_channels = int(self.camera.shape[1]) if self.camera is not None else 0
        self.camera_shape = tuple(int(x) for x in self.camera.shape[1:]) if self.camera is not None else None
        if not all(arr.shape[0] == self.size for arr in (
            self.lidar, self.async_meta, self.target_rel,
            self.target_gap, self.target_collision, self.target_ttc
        )):
            raise ValueError("dataset arrays have inconsistent first dimension")
        if self.camera is not None and int(self.camera.shape[0]) != self.size:
            raise ValueError("camera array has inconsistent first dimension")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep = int(self.episode_id[idx])
        start = idx
        while start > 0 and int(self.episode_id[start - 1]) == ep and (idx - start + 1) < self.seq_len:
            start -= 1
        length = idx - start + 1

        ego_seq = np.zeros((self.seq_len, self.ego8.shape[1]), dtype=np.float32)
        lidar_seq = np.zeros((self.seq_len, self.lidar.shape[1]), dtype=np.float32)
        async_meta_seq = np.zeros((self.seq_len, self.async_meta.shape[1]), dtype=np.float32)
        camera_seq = None
        if self.camera is not None:
            camera_seq = np.zeros((self.seq_len, *self.camera.shape[1:]), dtype=np.float32)
        ego_slice = self.ego8[start : idx + 1]
        lidar_slice = self.lidar[start : idx + 1]
        async_slice = self.async_meta[start : idx + 1]
        ego_seq[:length] = ego_slice
        lidar_seq[:length] = lidar_slice
        async_meta_seq[:length] = async_slice
        if camera_seq is not None:
            camera_seq[:length] = self.camera[start : idx + 1]

        sample = {
            "ego_seq": torch.from_numpy(ego_seq),
            "lidar_seq": torch.from_numpy(lidar_seq),
            "async_meta_seq": torch.from_numpy(async_meta_seq),
            "length": torch.tensor(length, dtype=torch.int64),
            "target_rel": torch.from_numpy(self.target_rel[idx]),
            "target_rel_mask": torch.from_numpy(self.target_rel_mask[idx]),
            "target_gap": torch.from_numpy(self.target_gap[idx]),
            "target_collision": torch.tensor(self.target_collision[idx], dtype=torch.float32),
            "target_ttc": torch.tensor(self.target_ttc[idx], dtype=torch.float32),
            "target_safety_valid": torch.tensor(self.target_safety_valid[idx], dtype=torch.float32),
            "target_passable": torch.from_numpy(self.target_passable[idx]),
            "target_closing_rate": torch.tensor(self.target_closing_rate[idx], dtype=torch.float32),
            "target_overtake_progress": torch.tensor(self.target_overtake_progress[idx], dtype=torch.float32),
            "target_opportunity_valid": torch.tensor(self.target_opportunity_valid[idx], dtype=torch.float32),
            "scene_id": torch.tensor(self.scene_id[idx], dtype=torch.int64),
        }
        if camera_seq is not None:
            sample["camera_seq"] = torch.from_numpy(camera_seq)
        return sample


def _split_by_episode(dataset: WorldModelSequenceDataset, val_ratio: float, seed: int) -> Tuple[Subset, Subset]:
    unique_eps = np.unique(dataset.episode_id)
    if unique_eps.size < 2:
        raise ValueError("need at least 2 episodes for train/val split")
    rng = np.random.RandomState(seed)
    unique_eps = unique_eps.copy()
    rng.shuffle(unique_eps)
    val_count = int(np.clip(round(unique_eps.size * val_ratio), 1, max(1, unique_eps.size - 1)))
    val_eps = set(unique_eps[:val_count].tolist())
    train_idx = [i for i, ep in enumerate(dataset.episode_id.tolist()) if ep not in val_eps]
    val_idx = [i for i, ep in enumerate(dataset.episode_id.tolist()) if ep in val_eps]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def _batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device=device, non_blocking=True) for k, v in batch.items()}


def _masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    denom = torch.clamp(mask.sum(), min=1.0)
    return (torch.abs(pred - target) * mask).sum() / denom


def _set_stage_trainable(model: LocalWorldModelV17, stage: str) -> None:
    stage = str(stage).lower()
    for p in model.parameters():
        p.requires_grad = True
    if stage == "b":
        # Stage B keeps the interaction-progress and safety/passability branches
        # trainable, while freezing the shared encoder/trunk and geometry heads.
        for module in (
            model.ego_encoder,
            model.lidar_side_encoder,
            model.lidar_post,
            model.step_fusion,
            model.gru,
            model.trunk,
            model.interaction_head.target_rel_head,
            model.safety_head.pre,
            model.safety_head.gap_head,
        ):
            if module is None:
                continue
            for p in module.parameters():
                p.requires_grad = False


def _run_epoch(
    model: LocalWorldModelV17,
    loader: DataLoader,
    device: torch.device,
    stage: str,
    optimizer: torch.optim.Optimizer | None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    acc: Dict[str, float] = {
        "loss_total": 0.0,
        "loss_target": 0.0,
        "loss_gap": 0.0,
        "loss_collision": 0.0,
        "loss_ttc": 0.0,
        "loss_passable": 0.0,
        "loss_closing": 0.0,
        "loss_overtake_gain": 0.0,
        "mae_target_rel": 0.0,
        "mae_gap": 0.0,
        "samples": 0.0,
    }

    for batch in loader:
        batch = _batch_to_device(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        pred = model(
            ego_seq=batch["ego_seq"],
            lidar_seq=batch["lidar_seq"],
            async_meta_seq=batch["async_meta_seq"],
            camera_seq=batch.get("camera_seq"),
            lengths=batch["length"],
        )
        loss, stats = local_world_model_loss(pred=pred, batch=batch, stage=stage)
        if is_train:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            target_mae = _masked_mae(pred["target_rel"], batch["target_rel"], batch["target_rel_mask"])
            gap_mae = torch.mean(torch.abs(pred["gap"] - batch["target_gap"]))
            bsz = float(batch["ego_seq"].shape[0])
            acc["samples"] += bsz
            for key in (
                "loss_total",
                "loss_target",
                "loss_gap",
                "loss_collision",
                "loss_ttc",
                "loss_passable",
                "loss_closing",
                "loss_overtake_gain",
            ):
                acc[key] += float(stats[key]) * bsz
            acc["mae_target_rel"] += float(target_mae.detach().cpu().item()) * bsz
            acc["mae_gap"] += float(gap_mae.detach().cpu().item()) * bsz

    denom = max(1.0, acc["samples"])
    return {k: (v / denom if k != "samples" else v) for k, v in acc.items()}


def _save_checkpoint(path: str, model: LocalWorldModelV17, extra: Dict[str, Any]) -> None:
    payload = {
        "model_state": model.state_dict(),
        "model_kwargs": {
            "ego_dim": model.ego_dim,
            "camera_channels": model.camera_channels,
            "camera_feat_dim": model.camera_feat_dim,
            "lidar_dim": model.lidar_dim,
            "async_meta_dim": model.async_meta_dim,
            "hidden_dim": model.hidden_dim,
        },
    }
    payload.update(extra)
    torch.save(payload, path)


def _load_checkpoint(path: str, device: torch.device) -> LocalWorldModelV17:
    payload = torch.load(path, map_location="cpu")
    model = LocalWorldModelV17(**payload["model_kwargs"])
    model.load_state_dict(payload["model_state"])
    model = model.to(device)
    return model


def _best_geom_baseline(stage_a_best: Dict[str, float], pre_stage_c: Dict[str, float]) -> Dict[str, float]:
    return {
        "mae_target_rel": min(stage_a_best["mae_target_rel"], pre_stage_c["mae_target_rel"]),
        "mae_gap": min(stage_a_best["mae_gap"], pre_stage_c["mae_gap"]),
    }


def _geom_regressed(metrics: Dict[str, float], baseline: Dict[str, float], rel_tol: float = 0.03) -> bool:
    for key, base in baseline.items():
        allow = max(base * (1.0 + rel_tol), base + 1e-6)
        if metrics[key] > allow:
            return True
    return False


def _train_stage(
    stage: str,
    model: LocalWorldModelV17,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_dir: Path,
    epochs: int,
    lr: float,
    stage_c_geom_baseline: Dict[str, float] | None = None,
) -> Tuple[LocalWorldModelV17, Dict[str, Any]]:
    _set_stage_trainable(model, stage=stage)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=float(lr))
    best_val = float("inf")
    best_metrics: Dict[str, float] | None = None
    best_path = save_dir / f"stage_{stage}_best.pth"
    history: List[Dict[str, Any]] = []
    guard_triggered = False
    initial_state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }

    for epoch in range(1, int(epochs) + 1):
        train_metrics = _run_epoch(model, train_loader, device=device, stage=stage, optimizer=optimizer)
        with torch.no_grad():
            val_metrics = _run_epoch(model, val_loader, device=device, stage=stage, optimizer=None)

        epoch_metrics = {"epoch": int(epoch), "train": train_metrics, "val": val_metrics}
        history.append(epoch_metrics)
        print(
            f"[stage {stage}] epoch={epoch} "
            f"train_loss={train_metrics['loss_total']:.4f} "
            f"val_loss={val_metrics['loss_total']:.4f} "
            f"val_mae_target={val_metrics['mae_target_rel']:.4f} "
            f"val_mae_gap={val_metrics['mae_gap']:.4f}"
        )

        if stage == "c" and stage_c_geom_baseline is not None:
            if _geom_regressed(val_metrics, stage_c_geom_baseline, rel_tol=0.03):
                print("[stage c] geometric regression guard triggered; stopping fine-tune")
                guard_triggered = True
                break

        if val_metrics["loss_total"] < best_val:
            best_val = float(val_metrics["loss_total"])
            best_metrics = dict(val_metrics)
            _save_checkpoint(
                str(best_path),
                model,
                extra={
                    "stage": stage,
                    "best_val_loss": best_val,
                    "best_metrics": best_metrics,
                },
            )

    if best_metrics is None:
        if stage == "c" and guard_triggered:
            model.load_state_dict(initial_state)
            fallback_path = save_dir / "stage_c_guard_fallback.pth"
            fallback_metrics = history[-1]["val"] if history else {}
            _save_checkpoint(
                str(fallback_path),
                model,
                extra={
                    "stage": stage,
                    "guard_triggered": True,
                    "fallback_to_stage_start": True,
                    "best_metrics": fallback_metrics,
                },
            )
            return model, {
                "stage": stage,
                "best_path": str(fallback_path),
                "best_val_loss": None,
                "best_metrics": fallback_metrics,
                "history": history,
                "guard_triggered": True,
                "fallback_to_stage_start": True,
            }
        raise RuntimeError(f"stage {stage} produced no checkpoints")
    best_model = _load_checkpoint(str(best_path), device=device)
    return best_model, {
        "stage": stage,
        "best_path": str(best_path),
        "best_val_loss": best_val,
        "best_metrics": best_metrics,
        "history": history,
        "guard_triggered": guard_triggered,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V17 local world model")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--camera-feat-dim", type=int, default=64)
    parser.add_argument("--epochs-a", type=int, default=20)
    parser.add_argument("--epochs-b", type=int, default=12)
    parser.add_argument("--epochs-c", type=int, default=6)
    parser.add_argument("--lr-a", type=float, default=1e-3)
    parser.add_argument("--lr-b", type=float, default=5e-4)
    parser.add_argument("--lr-c", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--stop-after-stage", type=str, choices=("a", "b", "c"), default="c")
    args = parser.parse_args()

    _seed_everything(int(args.seed))
    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    )

    dataset = WorldModelSequenceDataset(args.data, seq_len=int(args.seq_len))
    train_set, val_set = _split_by_episode(dataset, val_ratio=float(args.val_ratio), seed=int(args.seed))
    train_loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=int(args.batch_size), shuffle=False, num_workers=0)

    model = LocalWorldModelV17(
        hidden_dim=int(args.hidden_dim),
        camera_channels=int(dataset.camera_channels),
        camera_feat_dim=int(args.camera_feat_dim),
    ).to(device)
    summary: Dict[str, Any] = {
        "data": args.data,
        "save_dir": str(save_dir),
        "device": str(device),
        "seq_len": int(args.seq_len),
        "train_size": len(train_set),
        "val_size": len(val_set),
        "camera_channels": int(dataset.camera_channels),
        "camera_shape": list(dataset.camera_shape) if dataset.camera_shape is not None else None,
    }

    model_a, stage_a = _train_stage(
        stage="a",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir,
        epochs=int(args.epochs_a),
        lr=float(args.lr_a),
    )
    summary["stage_a"] = stage_a
    if args.stop_after_stage == "a":
        final_path = save_dir / "local_world_model_v17_final.pth"
        _save_checkpoint(str(final_path), model_a, extra={"stage": "final", "summary": summary})
        summary["final_path"] = str(final_path)
        summary_path = save_dir / "train_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"saved final checkpoint: {final_path}")
        print(f"saved summary: {summary_path}")
        return

    model_b, stage_b = _train_stage(
        stage="b",
        model=model_a,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir,
        epochs=int(args.epochs_b),
        lr=float(args.lr_b),
    )
    summary["stage_b"] = stage_b
    if args.stop_after_stage == "b":
        final_path = save_dir / "local_world_model_v17_final.pth"
        _save_checkpoint(str(final_path), model_b, extra={"stage": "final", "summary": summary})
        summary["final_path"] = str(final_path)
        summary_path = save_dir / "train_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"saved final checkpoint: {final_path}")
        print(f"saved summary: {summary_path}")
        return

    with torch.no_grad():
        pre_stage_c_metrics = _run_epoch(model_b, val_loader, device=device, stage="c", optimizer=None)
    geom_baseline = _best_geom_baseline(stage_a["best_metrics"], pre_stage_c_metrics)
    summary["stage_c_geom_baseline"] = geom_baseline
    summary["stage_c_pre_metrics"] = pre_stage_c_metrics

    model_c, stage_c = _train_stage(
        stage="c",
        model=model_b,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir,
        epochs=int(args.epochs_c),
        lr=float(args.lr_c),
        stage_c_geom_baseline=geom_baseline,
    )
    summary["stage_c"] = stage_c

    final_path = save_dir / "local_world_model_v17_final.pth"
    _save_checkpoint(str(final_path), model_c, extra={"stage": "final", "summary": summary})
    summary["final_path"] = str(final_path)

    summary_path = save_dir / "train_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"saved final checkpoint: {final_path}")
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
