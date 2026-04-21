"""
module/world_model_dataset.py

局部动力学世界模型数据管道 — V1

支持两类数据源
--------------
1. CatalogTransitionDataset  —  真实车 DonkeyCar catalog（JSONL）
2. SimTransitionDataset      —  Sim 采集 CSV（由 collect_sim_transitions.py 生成）
3. CombinedTransitionDataset —  两者合并，支持 domain_weight（真实数据加权）

数据格式（每条样本）
--------------------
  x     : float32 tensor, shape (input_dim,)   — 8D 正式版 or 5D baseline
  delta  : float32 tensor, shape (3,)           — 目标残差 [Δv, Δω, Δa]

输入向量约定（8D）
-------------------
  [v_long_t, yaw_rate_t, accel_x_t,      # 当前物理状态（归一化）
   steer_exec_t, throttle_t,              # 当前执行指令
   prev_steer_exec, prev_throttle,        # 上一步执行指令（滞后）
   dt_norm]                              # dt_ms / 50.0

归一化常数（与 obv.py:_build_state_v13 一致）
---------------------------------------------
  V_MAX    = 2.2   m/s
  GYRO_MAX = 4.0   rad/s   (yaw_rate = clip(-gyro_z / GYRO_MAX, -2, 2))
  ACCEL_MAX= 9.8   m/s²
  DT_REF   = 50.0  ms
  DT_MAX   = 200   ms（超过则视为 session 中断，跳过）

注意：真实车 catalog 使用 -rp2040/gyro_z 作为 yaw_rate（取反！）
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ─── 归一化常数 ──────────────────────────────────────────────────
V_MAX     = 2.2
GYRO_MAX  = 4.0
ACCEL_MAX = 9.8
DT_REF    = 50.0   # ms，标准步长
DT_MAX_MS = 200    # ms，超过则跳过（session 中断 / 暂停）


def _norm_v(v: float) -> float:
    return float(np.clip(v / V_MAX, 0.0, 2.0))


def _norm_yaw(gyro_z_raw: float) -> float:
    """真实车 yaw_rate = -gyro_z（取反，与 _build_state_v13 一致）。"""
    return float(np.clip(-gyro_z_raw / GYRO_MAX, -2.0, 2.0))


def _norm_accel(accel_x: float) -> float:
    return float(np.clip(accel_x / ACCEL_MAX, -2.0, 2.0))


def _safe_float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


# ─── Catalog 数据集（真实车）────────────────────────────────────

class CatalogTransitionDataset(Dataset):
    """
    从 DonkeyCar JSONL catalog 目录加载 (x_8d, delta_3d) 训练对。

    目录下须包含 catalog_*.catalog 文件（JSONL，每行一条记录）。
    多个目录可一次传入，合并处理。

    数据录制说明
    ------------
    catalog 为人工驾驶录制（user mode），user/angle 直接是发给
    硬件的方向盘指令，不经过 ActionAdapter 或 ActionSafetyWrapper，
    因此直接用作 steer_exec，无近似误差。

    Parameters
    ----------
    catalog_dirs : list of str
        包含 catalog_*.catalog 文件的目录路径列表。
    input_dim : int
        5（baseline）或 8（正式版）。
    augment_noise : float
        训练时在输入向量上叠加的高斯噪声标准差。0 = 关闭。
    """

    def __init__(
        self,
        catalog_dirs: List[str],
        input_dim: int = 8,
        augment_noise: float = 0.005,
    ):
        assert input_dim in (5, 8), f"input_dim must be 5 or 8, got {input_dim}"
        self.input_dim      = input_dim
        self.augment_noise  = augment_noise
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []

        for d in catalog_dirs:
            self._load_dir(d)

    def _load_dir(self, directory: str) -> None:
        d = Path(directory)
        catalog_files = sorted(
            [p for p in d.glob("catalog_*.catalog") if "manifest" not in p.name],
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if not catalog_files:
            return

        records = []
        for cf in catalog_files:
            with open(cf, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        # 按时间戳排序
        records.sort(key=lambda r: r.get("_timestamp_ms", 0))
        self._build_pairs(records)

    def _build_pairs(self, records: list) -> None:
        prev_rec     = None
        prev_session = None

        for rec in records:
            session = rec.get("_session_id", "")
            ts_ms   = _safe_float(rec.get("_timestamp_ms", 0))

            # session 边界：重置
            if session != prev_session:
                prev_rec     = rec
                prev_session = session
                continue

            # 时间跳变检查
            dt_ms = ts_ms - _safe_float(prev_rec.get("_timestamp_ms", 0))
            if dt_ms <= 0 or dt_ms > DT_MAX_MS:
                prev_rec = rec
                continue

            x, delta = self._make_sample(prev_rec, rec, dt_ms)
            if x is not None:
                self.samples.append((x, delta))

            prev_rec = rec

    def _make_sample(
        self, r_t: dict, r_t1: dict, dt_ms: float
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """从相邻两帧构建 (x, delta)。"""
        # ─ 当前帧物理状态 ─
        v_t   = _norm_v(_safe_float(r_t.get("rp2040/speed_odom")))
        yaw_t = _norm_yaw(_safe_float(r_t.get("rp2040/gyro_z")))
        a_t   = _norm_accel(_safe_float(r_t.get("rp2040/accel_x")))

        # ─ 当前帧执行指令 ─
        steer_t   = float(np.clip(_safe_float(r_t.get("user/angle")),   -1.0, 1.0))
        thr_t     = float(np.clip(_safe_float(r_t.get("user/throttle")),  0.0, 0.3))

        # ─ 下一帧物理状态 ─
        v_t1   = _norm_v(_safe_float(r_t1.get("rp2040/speed_odom")))
        yaw_t1 = _norm_yaw(_safe_float(r_t1.get("rp2040/gyro_z")))
        a_t1   = _norm_accel(_safe_float(r_t1.get("rp2040/accel_x")))

        delta = np.array([v_t1 - v_t, yaw_t1 - yaw_t, a_t1 - a_t], dtype=np.float32)

        if self.input_dim == 5:
            x = np.array([v_t, yaw_t, a_t, steer_t, thr_t], dtype=np.float32)
        else:
            # 前一帧指令（来自 r_t 的前一帧已被处理，这里近似用 r_t 本身的值
            # 作为 prev；真正的 prev 在 _build_pairs 中可追踪，但需要 3 帧窗口）
            # 简化：prev_steer ≈ steer_t（第一步 prev = current，误差小）
            # 若需精确，使用 _build_pairs_3frame 版本
            dt_norm = dt_ms / DT_REF
            x = np.array(
                [v_t, yaw_t, a_t, steer_t, thr_t, steer_t, thr_t, dt_norm],
                dtype=np.float32,
            )

        return x, delta

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, delta = self.samples[idx]
        x = x.copy()
        if self.augment_noise > 0.0 and self.training_mode:
            x += np.random.normal(0, self.augment_noise, x.shape).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(delta)

    # training_mode 由外部设置（Dataset 本身不区分 train/val）
    @property
    def training_mode(self) -> bool:
        return getattr(self, "_training_mode", True)

    def train(self):
        self._training_mode = True
        return self

    def eval(self):
        self._training_mode = False
        return self


class CatalogTransitionDatasetV2(CatalogTransitionDataset):
    """
    精确 8D 版：用 3 帧窗口正确获取 prev_steer_exec / prev_throttle。
    继承 V1 接口，仅覆盖 _build_pairs。
    """

    def _build_pairs(self, records: list) -> None:
        prev_prev_rec = None
        prev_rec      = None
        prev_session  = None

        for rec in records:
            session = rec.get("_session_id", "")
            ts_ms   = _safe_float(rec.get("_timestamp_ms", 0))

            if session != prev_session:
                prev_prev_rec = None
                prev_rec      = rec
                prev_session  = session
                continue

            # 检查相邻帧时间跳变
            dt_ms = ts_ms - _safe_float(prev_rec.get("_timestamp_ms", 0))
            if dt_ms <= 0 or dt_ms > DT_MAX_MS:
                prev_prev_rec = None
                prev_rec      = rec
                continue

            if prev_prev_rec is not None and self.input_dim == 8:
                x, delta = self._make_sample_v2(prev_prev_rec, prev_rec, rec, dt_ms)
                if x is not None:
                    self.samples.append((x, delta))
            elif self.input_dim == 5:
                x, delta = self._make_sample(prev_rec, rec, dt_ms)
                if x is not None:
                    self.samples.append((x, delta))

            prev_prev_rec = prev_rec
            prev_rec      = rec

    def _make_sample_v2(
        self, r_tm1: dict, r_t: dict, r_t1: dict, dt_ms: float
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """精确 3 帧窗口：r_tm1=t-1, r_t=t, r_t1=t+1。"""
        v_t   = _norm_v(_safe_float(r_t.get("rp2040/speed_odom")))
        yaw_t = _norm_yaw(_safe_float(r_t.get("rp2040/gyro_z")))
        a_t   = _norm_accel(_safe_float(r_t.get("rp2040/accel_x")))

        steer_t   = float(np.clip(_safe_float(r_t.get("user/angle")),    -1.0, 1.0))
        thr_t     = float(np.clip(_safe_float(r_t.get("user/throttle")),  0.0, 0.3))
        prev_steer = float(np.clip(_safe_float(r_tm1.get("user/angle")),  -1.0, 1.0))
        prev_thr   = float(np.clip(_safe_float(r_tm1.get("user/throttle")), 0.0, 0.3))

        v_t1   = _norm_v(_safe_float(r_t1.get("rp2040/speed_odom")))
        yaw_t1 = _norm_yaw(_safe_float(r_t1.get("rp2040/gyro_z")))
        a_t1   = _norm_accel(_safe_float(r_t1.get("rp2040/accel_x")))

        delta  = np.array([v_t1 - v_t, yaw_t1 - yaw_t, a_t1 - a_t], dtype=np.float32)
        dt_norm = dt_ms / DT_REF
        x = np.array(
            [v_t, yaw_t, a_t, steer_t, thr_t, prev_steer, prev_thr, dt_norm],
            dtype=np.float32,
        )
        return x, delta


# ─── Sim CSV 数据集 ──────────────────────────────────────────────

class SimTransitionDataset(Dataset):
    """
    从 collect_sim_transitions.py 生成的 CSV 文件加载转移数据。

    CSV 列名（顺序无关，按名称读取）：
      v_t, yaw_t, accel_t, steer_exec_t, throttle_t,
      prev_steer_exec, prev_throttle, dt_ms,
      v_t1, yaw_t1, accel_t1,
      policy_type, episode_id      ← 可选，不参与训练

    Parameters
    ----------
    csv_dirs : list of str
        包含 *.csv 文件的目录列表。
    input_dim : int
        5 或 8。
    augment_noise : float
        训练时输入噪声标准差。
    """

    def __init__(
        self,
        csv_dirs: List[str],
        input_dim: int = 8,
        augment_noise: float = 0.005,
    ):
        assert input_dim in (5, 8)
        self.input_dim     = input_dim
        self.augment_noise = augment_noise
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []

        for d in csv_dirs:
            self._load_dir(d)

    def _load_dir(self, directory: str) -> None:
        for csv_path in sorted(Path(directory).glob("*.csv")):
            self._load_csv(csv_path)

    def _load_csv(self, csv_path: Path) -> None:
        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    v_t    = float(row["v_t"])
                    yaw_t  = float(row["yaw_t"])
                    a_t    = float(row["accel_t"])
                    s_t    = float(row["steer_exec_t"])
                    thr_t  = float(row["throttle_t"])
                    v_t1   = float(row["v_t1"])
                    yaw_t1 = float(row["yaw_t1"])
                    a_t1   = float(row["accel_t1"])

                    delta = np.array(
                        [v_t1 - v_t, yaw_t1 - yaw_t, a_t1 - a_t],
                        dtype=np.float32,
                    )

                    if self.input_dim == 5:
                        x = np.array([v_t, yaw_t, a_t, s_t, thr_t], dtype=np.float32)
                    else:
                        ps_t   = float(row.get("prev_steer_exec", s_t))
                        pthr_t = float(row.get("prev_throttle", thr_t))
                        dt_ms  = float(row.get("dt_ms", 50.0))
                        dt_norm = dt_ms / DT_REF
                        x = np.array(
                            [v_t, yaw_t, a_t, s_t, thr_t, ps_t, pthr_t, dt_norm],
                            dtype=np.float32,
                        )

                    self.samples.append((x, delta))
                except (KeyError, ValueError):
                    continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, delta = self.samples[idx]
        x = x.copy()
        if self.augment_noise > 0.0 and getattr(self, "_training_mode", True):
            x += np.random.normal(0, self.augment_noise, x.shape).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(delta)

    def train(self):
        self._training_mode = True
        return self

    def eval(self):
        self._training_mode = False
        return self


# ─── 合并数据集 ──────────────────────────────────────────────────

class CombinedTransitionDataset(Dataset):
    """
    合并 catalog（真实）和 sim 数据集。

    domain_weight 控制真实数据在损失中的权重倍数（返回额外标志位 is_real）。
    实际加权在 train_world_model.py 的损失函数中实现。

    Parameters
    ----------
    real_dataset : CatalogTransitionDataset or None
    sim_dataset  : SimTransitionDataset or None
    """

    def __init__(
        self,
        real_dataset: Optional[Dataset] = None,
        sim_dataset:  Optional[Dataset] = None,
    ):
        assert real_dataset is not None or sim_dataset is not None
        self.real_ds   = real_dataset
        self.sim_ds    = sim_dataset
        self.real_len  = len(real_dataset) if real_dataset is not None else 0
        self.sim_len   = len(sim_dataset)  if sim_dataset  is not None else 0

    def __len__(self) -> int:
        return self.real_len + self.sim_len

    def __getitem__(self, idx: int):
        """Returns (x, delta, is_real) where is_real=1 for real-car data."""
        if idx < self.real_len:
            x, delta = self.real_ds[idx]
            is_real  = torch.tensor(1, dtype=torch.float32)
        else:
            x, delta = self.sim_ds[idx - self.real_len]
            is_real  = torch.tensor(0, dtype=torch.float32)
        return x, delta, is_real

    def train(self):
        if self.real_ds is not None:
            self.real_ds.train()
        if self.sim_ds is not None:
            self.sim_ds.train()
        return self

    def eval(self):
        if self.real_ds is not None:
            self.real_ds.eval()
        if self.sim_ds is not None:
            self.sim_ds.eval()
        return self


# ─── 工具：时序分割 ──────────────────────────────────────────────

def chronological_split(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
    """
    按时间顺序（非随机）分割数据集，避免时序泄露。

    Returns
    -------
    train_set, val_set, test_set : Subset
    """
    from torch.utils.data import Subset

    n       = len(dataset)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val

    idx = list(range(n))
    return (
        Subset(dataset, idx[:n_train]),
        Subset(dataset, idx[n_train: n_train + n_val]),
        Subset(dataset, idx[n_train + n_val:]),
    )


__all__ = [
    "CatalogTransitionDataset",
    "CatalogTransitionDatasetV2",
    "SimTransitionDataset",
    "CombinedTransitionDataset",
    "chronological_split",
    "V_MAX", "GYRO_MAX", "ACCEL_MAX", "DT_REF", "DT_MAX_MS",
]
