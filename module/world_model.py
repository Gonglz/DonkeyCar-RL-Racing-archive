"""
module/world_model.py

Bootstrap local dynamics model for the V17 predictive safety filter.

This is intentionally a small ego-dynamics residual model:
it predicts only the uncertain next-step deltas for
  [v_long, yaw_rate, accel_x]
from the executed controls and the current normalized physical state.

The longer-term V17 plan still targets a LiDAR-sequence multi-head local world
model. This file provides the first deployable piece needed for the current
log-only safety-filter path.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

PHYS_DIM = 3

INPUT_DIM_BASELINE = 5
INPUT_DIM_FULL = 8

STATE_LO = torch.tensor([0.0, -2.0, -2.0], dtype=torch.float32)
STATE_HI = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)


class NeuralPhysicsDynamics(nn.Module):
    """
    Residual MLP for short-horizon ego dynamics.

    Input layout (full 8D form):
      [v_long, yaw_rate, accel_x, steer_exec, throttle,
       prev_steer_exec, prev_throttle, dt_norm]

    Output:
      delta      (..., 3)
      next_state (..., 3), clipped to the normalized physical limits above
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM_FULL,
        hidden_dim: int = 128,
        dropout: float = 0.05,
    ):
        super().__init__()
        if input_dim not in (INPUT_DIM_BASELINE, INPUT_DIM_FULL):
            raise ValueError(
                f"input_dim must be {INPUT_DIM_BASELINE} or {INPUT_DIM_FULL}, got {input_dim}"
            )
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(max(0.0, dropout))

        self.trunk = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(64, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.head = nn.Linear(64, PHYS_DIM)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x: torch.Tensor,
        phys_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if phys_t is None:
            phys_t = x[..., :PHYS_DIM]

        h = self.trunk(x)
        delta = self.head(h)
        s_next = phys_t + delta

        lo = STATE_LO.to(device=s_next.device, dtype=s_next.dtype)
        hi = STATE_HI.to(device=s_next.device, dtype=s_next.dtype)
        s_next = torch.clamp(s_next, lo, hi)
        return delta, s_next

    def save_checkpoint(self, path: str, extra: Optional[dict] = None) -> None:
        checkpoint = {
            "model_state": self.state_dict(),
            "model_kwargs": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
            },
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> "NeuralPhysicsDynamics":
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(**checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["model_state"])
        model = model.to(device)
        model.eval()
        return model


def build_input_5d(
    v: float,
    yaw: float,
    accel: float,
    steer_exec: float,
    throttle: float,
) -> torch.Tensor:
    return torch.tensor([[v, yaw, accel, steer_exec, throttle]], dtype=torch.float32)


def build_input_8d(
    v: float,
    yaw: float,
    accel: float,
    steer_exec: float,
    throttle: float,
    prev_steer_exec: float,
    prev_throttle: float,
    dt_ms: float,
) -> torch.Tensor:
    dt_norm = float(dt_ms) / 50.0
    return torch.tensor(
        [[v, yaw, accel, steer_exec, throttle, prev_steer_exec, prev_throttle, dt_norm]],
        dtype=torch.float32,
    )


__all__ = [
    "INPUT_DIM_BASELINE",
    "INPUT_DIM_FULL",
    "NeuralPhysicsDynamics",
    "PHYS_DIM",
    "STATE_HI",
    "STATE_LO",
    "build_input_5d",
    "build_input_8d",
]
