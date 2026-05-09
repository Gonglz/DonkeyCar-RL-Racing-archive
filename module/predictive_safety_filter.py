"""
module/predictive_safety_filter.py

Log-first predictive safety filter used by the V17 action chain.

Current scope:
- shadow the ActionSafetyWrapper dynamics
- roll forward an ego-only residual world model for H steps
- record trigger statistics and optional JSONL logs

This is intentionally conservative: it does not rewrite actions yet.
The env wiring keeps it in log-only mode unless a later stage explicitly
enables light intervention.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .world_model import NeuralPhysicsDynamics

V_MAX = 2.2
GYRO_MAX = 4.0
ACCEL_MAX = 9.8


@dataclass
class PhysState:
    v_long: float
    yaw_rate: float
    accel_x: float

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([self.v_long, self.yaw_rate, self.accel_x], dtype=torch.float32)

    @classmethod
    def from_raw(cls, speed_mps: float, gyro_z: float, accel_x: float) -> "PhysState":
        return cls(
            v_long=float(np.clip(float(speed_mps) / V_MAX, 0.0, 2.0)),
            yaw_rate=float(np.clip(-float(gyro_z) / GYRO_MAX, -2.0, 2.0)),
            accel_x=float(np.clip(float(accel_x) / ACCEL_MAX, -2.0, 2.0)),
        )


@dataclass
class _ShadowSafetyState:
    steer_prev_limited: float = 0.0
    steer_prev_exec: float = 0.0
    delta_max: float = 0.5
    beta: float = 0.6

    def step(self, steer_target: float) -> Tuple[float, float]:
        delta = float(steer_target) - self.steer_prev_limited
        if abs(delta) > self.delta_max:
            delta = float(np.clip(delta, -self.delta_max, self.delta_max))
        steer_limited = self.steer_prev_limited + delta
        steer_exec = (1.0 - self.beta) * self.steer_prev_exec + self.beta * steer_limited
        steer_exec = float(np.clip(steer_exec, -1.0, 1.0))
        self.steer_prev_limited = float(steer_limited)
        self.steer_prev_exec = float(steer_exec)
        return self.steer_prev_exec, self.steer_prev_limited

    def copy(self) -> "_ShadowSafetyState":
        return _ShadowSafetyState(
            steer_prev_limited=self.steer_prev_limited,
            steer_prev_exec=self.steer_prev_exec,
            delta_max=self.delta_max,
            beta=self.beta,
        )


class PredictiveSafetyFilter:
    """
    H-step predictive safety filter.

    The filter is currently log-first: `check()` always returns diagnostics,
    and the surrounding control chain decides whether to intervene in later
    project stages.
    """

    def __init__(
        self,
        model_path: str,
        horizon: int = 3,
        delta_max: float = 0.5,
        beta: float = 0.6,
        yaw_thresh: Optional[float] = None,
        decel_thresh: Optional[float] = None,
        mode: str = "log",
        log_path: str = "",
        device: str = "cpu",
        intervene_throttle_scale: float = 0.5,
    ):
        if mode not in ("log", "intervene"):
            raise ValueError(f"mode must be 'log' or 'intervene', got {mode!r}")
        self.horizon = int(max(1, horizon))
        self.mode = str(mode)
        self.yaw_thresh = None if yaw_thresh is None else float(yaw_thresh)
        self.decel_thresh = None if decel_thresh is None else float(decel_thresh)
        self.device = str(device)
        self.intervene_throttle_scale = float(np.clip(intervene_throttle_scale, 0.0, 1.0))

        self.model = NeuralPhysicsDynamics.load_checkpoint(model_path, device=self.device)
        self.model.eval()

        self._shadow = _ShadowSafetyState(delta_max=float(delta_max), beta=float(beta))
        self._total_steps = 0
        self._triggered_steps = 0
        self._episode = 0

        self._log_path = str(log_path or "").strip()
        self._log_file = None
        if self._log_path:
            Path(self._log_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(self._log_path, "a", encoding="utf-8")

    def check(
        self,
        steer_target: float,
        throttle: float,
        phys: PhysState,
        dt_ms: float = 50.0,
    ) -> Tuple[bool, List[Dict[str, float]], Dict[str, float]]:
        self._total_steps += 1

        shadow = self._shadow.copy()
        phys_cur = phys.to_tensor().to(self.device)
        dt_norm = float(dt_ms) / 50.0

        prev_steer_exec = float(shadow.steer_prev_exec)
        prev_throttle = float(throttle)
        predictions: List[Dict[str, float]] = []
        triggered = False
        trigger_dims: List[str] = []

        for h in range(self.horizon):
            steer_exec, _steer_limited = shadow.step(float(steer_target))
            x = torch.tensor(
                [[
                    float(phys_cur[0]),
                    float(phys_cur[1]),
                    float(phys_cur[2]),
                    float(steer_exec),
                    float(throttle),
                    float(prev_steer_exec),
                    float(prev_throttle),
                    float(dt_norm),
                ]],
                dtype=torch.float32,
                device=self.device,
            )

            with torch.no_grad():
                delta, phys_next = self.model(x, phys_cur.unsqueeze(0))

            delta = delta.squeeze(0)
            phys_next = phys_next.squeeze(0)

            step_pred = {
                "h": float(h + 1),
                "v_long": float(phys_next[0]),
                "yaw_rate": float(phys_next[1]),
                "accel_x": float(phys_next[2]),
                "delta_v": float(delta[0]),
                "steer_exec": float(steer_exec),
            }
            predictions.append(step_pred)

            if self.yaw_thresh is not None and abs(float(phys_next[1])) > self.yaw_thresh:
                triggered = True
                if "yaw_rate" not in trigger_dims:
                    trigger_dims.append("yaw_rate")
            if self.decel_thresh is not None and float(delta[0]) < -self.decel_thresh:
                triggered = True
                if "v_long_decel" not in trigger_dims:
                    trigger_dims.append("v_long_decel")

            prev_steer_exec = float(steer_exec)
            prev_throttle = float(throttle)
            phys_cur = phys_next

        if triggered:
            self._triggered_steps += 1

        diag = {
            "triggered": float(bool(triggered)),
            "trigger_rate": float(self._triggered_steps / max(self._total_steps, 1)),
            "total_steps": float(self._total_steps),
            "num_trigger_dims": float(len(trigger_dims)),
            "throttle_scale": float(self.intervene_throttle_scale if triggered else 1.0),
        }

        if triggered or (self._total_steps % 500 == 0):
            self._write_log(
                steer_target=float(steer_target),
                throttle=float(throttle),
                phys=phys,
                predictions=predictions,
                trigger_dims=trigger_dims,
            )
        return triggered, predictions, diag

    def sync(self, steer_prev_limited: float, steer_prev_exec: float) -> None:
        self._shadow.steer_prev_limited = float(steer_prev_limited)
        self._shadow.steer_prev_exec = float(steer_prev_exec)

    def reset(self, episode: Optional[int] = None) -> None:
        self._shadow.steer_prev_limited = 0.0
        self._shadow.steer_prev_exec = 0.0
        self._episode = int(episode) if episode is not None else self._episode + 1

    def stats(self) -> Dict[str, float]:
        return {
            "total_steps": float(self._total_steps),
            "triggered_steps": float(self._triggered_steps),
            "trigger_rate": float(self._triggered_steps / max(self._total_steps, 1)),
            "horizon": float(self.horizon),
        }

    def print_stats(self) -> None:
        s = self.stats()
        print(
            "[SafetyFilter] "
            f"steps={int(s['total_steps'])} "
            f"triggered={int(s['triggered_steps'])} "
            f"rate={s['trigger_rate']:.3%} "
            f"mode={self.mode}"
        )

    def _write_log(
        self,
        steer_target: float,
        throttle: float,
        phys: PhysState,
        predictions: List[Dict[str, float]],
        trigger_dims: List[str],
    ) -> None:
        if self._log_file is None:
            return
        record = {
            "ts": time.time(),
            "episode": int(self._episode),
            "step": int(self._total_steps),
            "mode": self.mode,
            "phys": {
                "v_long": float(phys.v_long),
                "yaw_rate": float(phys.yaw_rate),
                "accel_x": float(phys.accel_x),
            },
            "action": {
                "steer_target": float(steer_target),
                "throttle": float(throttle),
            },
            "predictions": predictions,
            "trigger_dims": list(trigger_dims),
        }
        self._log_file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._log_file.flush()

    def close(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def __del__(self):
        self.close()


__all__ = ["PhysState", "PredictiveSafetyFilter"]
