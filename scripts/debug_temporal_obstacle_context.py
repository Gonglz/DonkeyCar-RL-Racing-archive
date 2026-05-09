#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from module.obstacle_context_learned import ObstacleContextTemporalNet, obstacle_context_loss
from scripts.train_obstacle_context_temporal import (
    ObstacleContextSequenceDataset,
    _batch_to_device,
    _split_dataset_by_episode,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    result = {"dataset": os.path.abspath(args.dataset)}
    ds = ObstacleContextSequenceDataset(args.dataset, seq_len=int(args.seq_len))
    result["dataset_len"] = int(len(ds))
    item0 = ds[0]
    result["item0"] = {
        "image_seq_shape": list(item0["image_seq"].shape),
        "state7_seq_shape": list(item0["state7_seq"].shape),
        "length": int(item0["length"].item()),
        "target_present": float(item0["target_present"].item()),
    }

    train_set, val_set = _split_dataset_by_episode(ds, val_ratio=0.3, seed=42)
    result["split"] = {"train": int(len(train_set)), "val": int(len(val_set))}

    loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True, num_workers=0, drop_last=False)
    batch = next(iter(loader))
    result["batch_shapes"] = {
        "image_seq": list(batch["image_seq"].shape),
        "state7_seq": list(batch["state7_seq"].shape),
        "length": list(batch["length"].shape),
    }

    device = torch.device("cpu")
    model = ObstacleContextTemporalNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    batch = _batch_to_device(batch, device)
    pred = model(
        image_seq=batch["image_seq"],
        state7_seq=batch["state7_seq"],
        lengths=batch["length"],
    )
    result["pred_shapes"] = {k: list(v.shape) for k, v in pred.items()}

    target_present = ds.target_present
    positive_count = float((target_present >= 0.5).sum())
    negative_count = float(target_present.shape[0] - positive_count)
    pos_weight_value = max(1.0, negative_count / max(1.0, positive_count))
    present_pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)

    loss, loss_stats = obstacle_context_loss(
        pred=pred,
        target_present=batch["target_present"],
        target_longitudinal=batch["target_longitudinal"],
        target_lateral=batch["target_lateral"],
        target_dist=batch["target_dist"],
        present_pos_weight=present_pos_weight,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    result["loss_stats"] = loss_stats
    result["status"] = "ok"

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
