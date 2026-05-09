#!/usr/bin/env python3
"""
Train a real-data ego dynamics alignment model compatible with module/world_model.py.

Input features follow the existing 8D residual-dynamics layout:
  [v_long, yaw_rate, accel_x, steer_exec, throttle,
   prev_steer_exec, prev_throttle, dt_norm]

Targets are the next-step normalized physical state and residual delta for:
  [v_long, yaw_rate, accel_x]

Primary intent:
- train on real tub catalogs (0421)
- optionally evaluate on a separate stress set (0422_1)
- save checkpoints loadable by NeuralPhysicsDynamics.load_checkpoint()
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from module.predictive_safety_filter import PhysState
from module.world_model import NeuralPhysicsDynamics


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _iter_catalog_rows(data_root: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    manifest_path = data_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest_lines = manifest_path.read_text().splitlines()
    if len(manifest_lines) < 5:
        raise ValueError(f"manifest too short: {manifest_path}")
    catalog_meta = json.loads(manifest_lines[4])
    paths = list(catalog_meta.get("paths", []))
    if not paths:
        raise ValueError(f"manifest contains no catalog paths: {manifest_path}")
    for catalog_rel in paths:
        catalog_path = data_root / str(catalog_rel)
        with catalog_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                yield str(catalog_path.name), row


@dataclass
class _RowState:
    catalog_name: str
    session_id: str
    timestamp_ms: float
    steer: float
    throttle: float
    phys: np.ndarray


def _build_phys_from_row(row: Dict[str, Any]) -> np.ndarray:
    speed_mps = _safe_float(row.get("rp2040/speed_odom"), 0.0)
    gyro_z = _safe_float(row.get("rp2040/gyro_z"), 0.0)
    accel_x = _safe_float(row.get("rp2040/accel_x"), 0.0)
    phys = PhysState.from_raw(speed_mps=speed_mps, gyro_z=gyro_z, accel_x=accel_x)
    return np.array([phys.v_long, phys.yaw_rate, phys.accel_x], dtype=np.float32)


def _build_samples_from_root(
    data_root: Path,
    min_dt: float,
    max_dt: float,
) -> Dict[str, np.ndarray]:
    inputs: List[np.ndarray] = []
    target_delta: List[np.ndarray] = []
    target_next: List[np.ndarray] = []
    dt_s_list: List[float] = []
    sample_index: List[int] = []
    session_index: List[int] = []
    catalog_index: List[int] = []
    raw_speed: List[float] = []
    raw_gyro_z: List[float] = []
    raw_accel_x: List[float] = []

    session_map: Dict[str, int] = {}
    catalog_map: Dict[str, int] = {}
    prev_row: _RowState | None = None
    prev_prev_controls: Tuple[float, float] | None = None

    total_rows = 0
    kept_samples = 0
    skipped_bad_dt = 0
    skipped_session_break = 0

    for catalog_name, row in _iter_catalog_rows(data_root):
        total_rows += 1
        session_id = str(row.get("_session_id", "unknown"))
        timestamp_ms = _safe_float(row.get("_timestamp_ms"), float(total_rows))
        steer = float(np.clip(_safe_float(row.get("user/angle"), 0.0), -1.0, 1.0))
        throttle = float(np.clip(_safe_float(row.get("user/throttle"), 0.0), -1.0, 1.0))
        phys = _build_phys_from_row(row)

        cur_row = _RowState(
            catalog_name=str(catalog_name),
            session_id=session_id,
            timestamp_ms=timestamp_ms,
            steer=steer,
            throttle=throttle,
            phys=phys,
        )

        if prev_row is not None:
            same_session = prev_row.session_id == cur_row.session_id
            dt_s = (cur_row.timestamp_ms - prev_row.timestamp_ms) / 1000.0
            if not same_session:
                skipped_session_break += 1
            elif dt_s < float(min_dt) or dt_s > float(max_dt):
                skipped_bad_dt += 1
            else:
                prev_controls = prev_prev_controls if prev_prev_controls is not None else (prev_row.steer, prev_row.throttle)
                x = np.array(
                    [
                        float(prev_row.phys[0]),
                        float(prev_row.phys[1]),
                        float(prev_row.phys[2]),
                        float(prev_row.steer),
                        float(prev_row.throttle),
                        float(prev_controls[0]),
                        float(prev_controls[1]),
                        float(np.clip(dt_s / 0.05, 0.0, 4.0)),
                    ],
                    dtype=np.float32,
                )
                y_next = np.asarray(cur_row.phys, dtype=np.float32)
                y_delta = y_next - np.asarray(prev_row.phys, dtype=np.float32)

                sid = session_map.setdefault(session_id, len(session_map))
                cid = catalog_map.setdefault(cur_row.catalog_name, len(catalog_map))
                inputs.append(x)
                target_next.append(y_next)
                target_delta.append(y_delta)
                dt_s_list.append(float(dt_s))
                sample_index.append(int(kept_samples))
                session_index.append(int(sid))
                catalog_index.append(int(cid))
                raw_speed.append(_safe_float(row.get("rp2040/speed_odom"), 0.0))
                raw_gyro_z.append(_safe_float(row.get("rp2040/gyro_z"), 0.0))
                raw_accel_x.append(_safe_float(row.get("rp2040/accel_x"), 0.0))
                kept_samples += 1

        prev_prev_controls = (cur_row.steer, cur_row.throttle) if prev_row is None else (prev_row.steer, prev_row.throttle)
        prev_row = cur_row

    if not inputs:
        raise ValueError(f"no valid samples built from {data_root}")

    return {
        "x": np.asarray(inputs, dtype=np.float32),
        "target_delta": np.asarray(target_delta, dtype=np.float32),
        "target_next": np.asarray(target_next, dtype=np.float32),
        "dt_s": np.asarray(dt_s_list, dtype=np.float32),
        "sample_index": np.asarray(sample_index, dtype=np.int64),
        "session_index": np.asarray(session_index, dtype=np.int64),
        "catalog_index": np.asarray(catalog_index, dtype=np.int64),
        "raw_speed_odom": np.asarray(raw_speed, dtype=np.float32),
        "raw_gyro_z": np.asarray(raw_gyro_z, dtype=np.float32),
        "raw_accel_x": np.asarray(raw_accel_x, dtype=np.float32),
        "stats_json": {
            "data_root": str(data_root.resolve()),
            "total_rows": int(total_rows),
            "kept_samples": int(kept_samples),
            "skipped_bad_dt": int(skipped_bad_dt),
            "skipped_session_break": int(skipped_session_break),
            "session_count": int(len(session_map)),
            "catalog_count": int(len(catalog_map)),
        },
    }


class EgoDynamicsDataset(Dataset):
    def __init__(self, x: np.ndarray, target_delta: np.ndarray, target_next: np.ndarray):
        if x.shape[0] != target_delta.shape[0] or x.shape[0] != target_next.shape[0]:
            raise ValueError("dataset arrays have inconsistent first dimension")
        self.x = np.asarray(x, dtype=np.float32)
        self.target_delta = np.asarray(target_delta, dtype=np.float32)
        self.target_next = np.asarray(target_next, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.x[idx]),
            "target_delta": torch.from_numpy(self.target_delta[idx]),
            "target_next": torch.from_numpy(self.target_next[idx]),
        }


def _chronological_split(
    x: np.ndarray,
    target_delta: np.ndarray,
    target_next: np.ndarray,
    val_ratio: float,
) -> Tuple[EgoDynamicsDataset, EgoDynamicsDataset]:
    n = int(x.shape[0])
    if n < 10:
        raise ValueError("dataset too small for split")
    val_count = int(np.clip(round(n * float(val_ratio)), 1, max(1, n - 1)))
    split = n - val_count
    return (
        EgoDynamicsDataset(x[:split], target_delta[:split], target_next[:split]),
        EgoDynamicsDataset(x[split:], target_delta[split:], target_next[split:]),
    )


def _batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device=device, non_blocking=True) for k, v in batch.items()}


def _run_epoch(
    model: NeuralPhysicsDynamics,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    total = {
        "loss": 0.0,
        "loss_delta": 0.0,
        "loss_next": 0.0,
        "mae_delta_v": 0.0,
        "mae_delta_yaw_rate": 0.0,
        "mae_delta_accel_x": 0.0,
        "mae_next_v": 0.0,
        "mae_next_yaw_rate": 0.0,
        "mae_next_accel_x": 0.0,
        "samples": 0.0,
    }
    mse = nn.MSELoss()
    for batch in loader:
        batch = _batch_to_device(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        pred_delta, pred_next = model(batch["x"])
        loss_delta = mse(pred_delta, batch["target_delta"])
        loss_next = mse(pred_next, batch["target_next"])
        loss = 0.5 * loss_delta + 0.5 * loss_next
        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        with torch.no_grad():
            bsz = float(batch["x"].shape[0])
            total["samples"] += bsz
            total["loss"] += float(loss.item()) * bsz
            total["loss_delta"] += float(loss_delta.item()) * bsz
            total["loss_next"] += float(loss_next.item()) * bsz
            delta_abs = torch.abs(pred_delta - batch["target_delta"]).mean(dim=0)
            next_abs = torch.abs(pred_next - batch["target_next"]).mean(dim=0)
            total["mae_delta_v"] += float(delta_abs[0].item()) * bsz
            total["mae_delta_yaw_rate"] += float(delta_abs[1].item()) * bsz
            total["mae_delta_accel_x"] += float(delta_abs[2].item()) * bsz
            total["mae_next_v"] += float(next_abs[0].item()) * bsz
            total["mae_next_yaw_rate"] += float(next_abs[1].item()) * bsz
            total["mae_next_accel_x"] += float(next_abs[2].item()) * bsz
    denom = max(1.0, total["samples"])
    return {k: (v / denom if k != "samples" else v) for k, v in total.items()}


def _dataset_metrics(payload: Dict[str, np.ndarray]) -> Dict[str, Any]:
    x = payload["x"]
    tgt_next = payload["target_next"]
    tgt_delta = payload["target_delta"]
    dt_s = payload["dt_s"]
    metrics = dict(payload["stats_json"])
    metrics.update(
        {
            "input_dim": int(x.shape[1]),
            "samples": int(x.shape[0]),
            "dt_s_mean": float(np.mean(dt_s)),
            "dt_s_p50": float(np.percentile(dt_s, 50)),
            "dt_s_p95": float(np.percentile(dt_s, 95)),
            "v_long_next_mean": float(np.mean(tgt_next[:, 0])),
            "yaw_rate_next_mean": float(np.mean(tgt_next[:, 1])),
            "accel_x_next_mean": float(np.mean(tgt_next[:, 2])),
            "delta_v_std": float(np.std(tgt_delta[:, 0])),
            "delta_yaw_rate_std": float(np.std(tgt_delta[:, 1])),
            "delta_accel_x_std": float(np.std(tgt_delta[:, 2])),
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train real-data ego dynamics alignment model")
    parser.add_argument(
        "--train-root",
        action="append",
        default=[],
        help="real data root containing manifest.json and catalog_*.catalog; may be passed multiple times",
    )
    parser.add_argument(
        "--stress-root",
        action="append",
        default=[],
        help="optional stress/test real data root; may be passed multiple times",
    )
    parser.add_argument("--min-dt", type=float, default=0.02)
    parser.add_argument("--max-dt", type=float, default=0.20)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--cache-npz", type=str, default="")
    args = parser.parse_args()

    train_roots = [Path(p).resolve() for p in args.train_root] or [
        Path("/home/longzhao/mysim/data/data_lidar/0421/data").resolve()
    ]
    stress_roots = [Path(p).resolve() for p in args.stress_root] or [
        Path("/home/longzhao/mysim/data/data_lidar/0422_1/data").resolve()
    ]

    _seed_everything(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_payloads = [_build_samples_from_root(root, min_dt=float(args.min_dt), max_dt=float(args.max_dt)) for root in train_roots]
    train_x = np.concatenate([p["x"] for p in train_payloads], axis=0)
    train_delta = np.concatenate([p["target_delta"] for p in train_payloads], axis=0)
    train_next = np.concatenate([p["target_next"] for p in train_payloads], axis=0)
    train_set, val_set = _chronological_split(train_x, train_delta, train_next, val_ratio=float(args.val_ratio))

    stress_eval: List[Tuple[str, EgoDynamicsDataset, Dict[str, Any]]] = []
    for root in stress_roots:
        if not root.exists():
            continue
        payload = _build_samples_from_root(root, min_dt=float(args.min_dt), max_dt=float(args.max_dt))
        stress_eval.append((str(root), EgoDynamicsDataset(payload["x"], payload["target_delta"], payload["target_next"]), _dataset_metrics(payload)))

    if args.cache_npz:
        np.savez_compressed(
            str(Path(args.cache_npz).resolve()),
            train_x=train_x,
            train_target_delta=train_delta,
            train_target_next=train_next,
        )

    train_loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_set, batch_size=int(args.batch_size), shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    stress_loaders = [
        (
            root,
            DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, pin_memory=(device.type == "cuda")),
            meta,
        )
        for root, ds, meta in stress_eval
    ]

    model = NeuralPhysicsDynamics(input_dim=8, hidden_dim=int(args.hidden_dim), dropout=float(args.dropout)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_val = float("inf")
    history: List[Dict[str, Any]] = []
    best_summary: Dict[str, Any] | None = None

    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = _run_epoch(model, train_loader, device=device, optimizer=optimizer)
        with torch.no_grad():
            val_metrics = _run_epoch(model, val_loader, device=device, optimizer=None)
        epoch_summary: Dict[str, Any] = {
            "epoch": int(epoch),
            "train": train_metrics,
            "val": val_metrics,
        }
        if stress_loaders:
            stress_results = {}
            with torch.no_grad():
                for root, loader, _meta in stress_loaders:
                    stress_results[root] = _run_epoch(model, loader, device=device, optimizer=None)
            epoch_summary["stress"] = stress_results
        history.append(epoch_summary)
        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_mae_dv={val_metrics['mae_delta_v']:.5f} "
            f"val_mae_dyaw={val_metrics['mae_delta_yaw_rate']:.5f} "
            f"val_mae_dax={val_metrics['mae_delta_accel_x']:.5f}"
        )
        if val_metrics["loss"] < best_val:
            best_val = float(val_metrics["loss"])
            best_summary = dict(epoch_summary)
            model.save_checkpoint(
                str(output_dir / "best.ckpt"),
                extra={
                    "epoch": int(epoch),
                    "best_val_loss": float(best_val),
                    "train_roots": [str(p) for p in train_roots],
                    "stress_roots": [str(p) for p in stress_roots],
                },
            )

    final_ckpt = output_dir / "last.ckpt"
    model.save_checkpoint(
        str(final_ckpt),
        extra={
            "epoch": int(args.epochs),
            "best_val_loss": float(best_val),
            "train_roots": [str(p) for p in train_roots],
            "stress_roots": [str(p) for p in stress_roots],
        },
    )

    summary = {
        "train_dataset": {
            "roots": [str(p) for p in train_roots],
            "metrics": [_dataset_metrics(p) for p in train_payloads],
            "train_samples": int(len(train_set)),
            "val_samples": int(len(val_set)),
        },
        "stress_dataset": {
            "roots": [root for root, _loader, _meta in stress_loaders],
            "metrics": [meta for _root, _loader, meta in stress_loaders],
        },
        "config": {
            "min_dt": float(args.min_dt),
            "max_dt": float(args.max_dt),
            "val_ratio": float(args.val_ratio),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "seed": int(args.seed),
            "device": str(device),
        },
        "best": best_summary,
        "history": history,
        "artifacts": {
            "best_ckpt": str((output_dir / "best.ckpt").resolve()),
            "last_ckpt": str(final_ckpt.resolve()),
        },
    }
    summary_path = output_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[saved] best_ckpt={(output_dir / 'best.ckpt').resolve()}")
    print(f"[saved] summary={summary_path.resolve()}")


if __name__ == "__main__":
    main()
