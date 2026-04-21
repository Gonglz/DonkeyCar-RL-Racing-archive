"""
module/world_model.py

自车局部动力学世界模型 — V1（分离式架构，神经物理层）

架构说明
--------
采用"分离式世界模型"：
  - 第一层（解析）：ActionAdapter + ActionSafetyWrapper 内部状态由已知代数方程精确更新，
                    不使用神经网络。
  - 第二层（本文件）：只预测真正不确定的 3 个物理量的残差
                      [Δv_long, Δyaw_rate, Δaccel_x]

模型输入（8 维正式版，5 维 baseline）
--------------------------------------
  [v_long_t,       yaw_rate_t,   accel_x_t,      # 当前物理状态
   steer_exec_t,   throttle_t,                    # 当前执行指令（已过 SafetyWrapper）
   prev_steer_exec, prev_throttle, dt_norm]        # 滞后 + 时间步（正式版补充）

  dt_norm = dt_ms / 50.0（归一化到 50 ms 标准步长）

模型输出（3 维残差）
--------------------
  [Δv_long, Δyaw_rate, Δaccel_x]
  next_state = clip(current + delta, lo, hi)

状态归一化范围（与 _build_state_v13 一致）
------------------------------------------
  v_long:   [0, 2]   = clip(speed / 2.2)
  yaw_rate: [-2, 2]  = clip(-gyro_z / 4.0)
  accel_x:  [-2, 2]  = clip(accel_x / 9.8)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, Optional

# 物理状态维度
PHYS_DIM = 3   # [v_long, yaw_rate, accel_x]

# 输入维度
INPUT_DIM_BASELINE = 5   # [v, ω, a, steer_exec, throttle]
INPUT_DIM_FULL     = 8   # + [prev_steer_exec, prev_throttle, dt_norm]

# 物理状态归一化上下界（顺序：v_long, yaw_rate, accel_x）
STATE_LO = torch.tensor([0.0,  -2.0, -2.0], dtype=torch.float32)
STATE_HI = torch.tensor([2.0,   2.0,  2.0], dtype=torch.float32)


class NeuralPhysicsDynamics(nn.Module):
    """
    局部物理动力学残差 MLP。

    预测 s_{t+1}[:3] = clip(s_t[:3] + f(x_t), lo, hi)
    其中 f 以零为初始化，确保训练早期接近恒等映射。

    Parameters
    ----------
    input_dim : int
        5（baseline sanity check）或 8（正式版，含滞后和时间步）。
    hidden_dim : int
        隐层宽度，默认 128。
    dropout : float
        Dropout 率，推理时自动 eval() 关闭。

    Usage
    -----
    model = NeuralPhysicsDynamics(input_dim=8)
    x = torch.zeros(B, 8)       # [v, ω, a, steer, thr, prev_steer, prev_thr, dt]
    phys_t = x[:, :3]           # 当前物理状态
    delta, s_next = model(x, phys_t)
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM_FULL,
        hidden_dim: int = 128,
        dropout: float = 0.05,
    ):
        super().__init__()
        assert input_dim in (INPUT_DIM_BASELINE, INPUT_DIM_FULL), (
            f"input_dim must be {INPUT_DIM_BASELINE} or {INPUT_DIM_FULL}, got {input_dim}"
        )
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        # 零初始化输出头：训练初期预测接近零残差（恒等映射）
        self.head = nn.Linear(64, PHYS_DIM)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x: torch.Tensor,
        phys_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (..., input_dim)  float32
            完整输入向量。
        phys_t : (..., 3) or None
            当前物理状态 [v, ω, a]。如果为 None，从 x[:3] 取。

        Returns
        -------
        delta  : (..., 3)  预测残差
        s_next : (..., 3)  下一步物理状态（已 clip）
        """
        if phys_t is None:
            phys_t = x[..., :PHYS_DIM]

        h     = self.trunk(x)
        delta = self.head(h)
        s_next = phys_t + delta

        lo = STATE_LO.to(s_next.device, dtype=s_next.dtype)
        hi = STATE_HI.to(s_next.device, dtype=s_next.dtype)
        s_next = torch.clamp(s_next, lo, hi)

        return delta, s_next

    # ─── 保存 / 加载 ────────────────────────────────────────────

    def save_checkpoint(self, path: str, extra: Optional[dict] = None) -> None:
        """保存检查点（含超参）。"""
        ckpt = {
            "model_state": self.state_dict(),
            "model_kwargs": {
                "input_dim":  self.input_dim,
                "hidden_dim": self.hidden_dim,
                "dropout":    0.0,   # 推理时关闭 dropout
            },
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> "NeuralPhysicsDynamics":
        """从检查点加载模型（eval 模式）。"""
        ckpt = torch.load(path, map_location="cpu")
        model = cls(**ckpt["model_kwargs"])
        model.load_state_dict(ckpt["model_state"])
        model = model.to(device)
        model.eval()
        return model


# ─── 工具函数：构建标准输入向量 ─────────────────────────────────

def build_input_5d(
    v: float, yaw: float, accel: float,
    steer_exec: float, throttle: float,
) -> torch.Tensor:
    """构建 5D baseline 输入（单步，返回 shape (1, 5)）。"""
    return torch.tensor(
        [[v, yaw, accel, steer_exec, throttle]], dtype=torch.float32
    )


def build_input_8d(
    v: float, yaw: float, accel: float,
    steer_exec: float, throttle: float,
    prev_steer_exec: float, prev_throttle: float,
    dt_ms: float,
) -> torch.Tensor:
    """构建 8D 正式版输入（单步，返回 shape (1, 8)）。"""
    dt_norm = dt_ms / 50.0
    return torch.tensor(
        [[v, yaw, accel, steer_exec, throttle,
          prev_steer_exec, prev_throttle, dt_norm]],
        dtype=torch.float32,
    )


__all__ = [
    "NeuralPhysicsDynamics",
    "build_input_5d",
    "build_input_8d",
    "INPUT_DIM_BASELINE",
    "INPUT_DIM_FULL",
    "PHYS_DIM",
    "STATE_LO",
    "STATE_HI",
]
