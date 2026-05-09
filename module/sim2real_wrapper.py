"""
module/sim2real_wrapper.py

Sim2Real 动力学对齐 Wrapper — 训练时使用
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional, Union

import gym
import numpy as np


class Sim2RealActionWrapper(gym.ActionWrapper):
    """对 [steer, throttle] 做缩放和可选一阶滞后。"""

    def __init__(
        self,
        env: gym.Env,
        throttle_gain: float = 1.0,
        steer_gain: float = 1.0,
        steer_tau_s: float = 0.0,
        throttle_tau_s: float = 0.0,
        throttle_gain_override: Optional[float] = None,
        throttle_gain_floor: Optional[float] = None,
        steer_gain_override: Optional[float] = None,
        steer_gain_floor: Optional[float] = None,
        filter_dt_s: Optional[float] = None,
    ):
        super().__init__(env)
        raw_throttle_gain = float(throttle_gain)
        raw_steer_gain = float(steer_gain)
        if throttle_gain_override is not None:
            throttle_gain = float(throttle_gain_override)
        elif throttle_gain_floor is not None:
            throttle_gain = max(float(throttle_gain), float(throttle_gain_floor))
        if steer_gain_override is not None:
            steer_gain = float(steer_gain_override)
        elif steer_gain_floor is not None:
            steer_gain = max(float(steer_gain), float(steer_gain_floor))

        self.throttle_gain = float(max(0.01, throttle_gain))
        self.steer_gain = float(max(0.01, steer_gain))
        self.steer_tau_s = float(max(0.0, steer_tau_s))
        self.throttle_tau_s = float(max(0.0, throttle_tau_s))
        self.raw_throttle_gain = raw_throttle_gain
        self.raw_steer_gain = raw_steer_gain
        self.throttle_gain_override = None if throttle_gain_override is None else float(throttle_gain_override)
        self.throttle_gain_floor = None if throttle_gain_floor is None else float(throttle_gain_floor)
        self.steer_gain_override = None if steer_gain_override is None else float(steer_gain_override)
        self.steer_gain_floor = None if steer_gain_floor is None else float(steer_gain_floor)
        self.filter_dt_s = None if filter_dt_s is None or float(filter_dt_s) <= 0.0 else float(filter_dt_s)

        self._filtered_steer = 0.0
        self._filtered_throttle = 0.0
        self._last_t: Optional[float] = None
        self.last_raw_action = np.zeros((2,), dtype=np.float32)
        self.last_transformed_action = np.zeros((2,), dtype=np.float32)

        filter_dt_label = "wall" if self.filter_dt_s is None else "{:.3f}s".format(self.filter_dt_s)
        print(
            f"[Sim2RealActionWrapper] "
            f"throttle_gain={self.throttle_gain:.4f} (raw={self.raw_throttle_gain:.4f}), "
            f"steer_gain={self.steer_gain:.4f} (raw={self.raw_steer_gain:.4f}), "
            f"steer_tau={self.steer_tau_s:.3f}s, "
            f"throttle_tau={self.throttle_tau_s:.3f}s, "
            f"filter_dt={filter_dt_label}"
        )

    @classmethod
    def from_json(
        cls,
        env: gym.Env,
        json_path: Union[str, Path],
        throttle_gain_override: Optional[float] = None,
        throttle_gain_floor: Optional[float] = None,
        steer_gain_override: Optional[float] = None,
        steer_gain_floor: Optional[float] = None,
        filter_dt_s: Optional[float] = None,
    ) -> "Sim2RealActionWrapper":
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Sim2Real calibration JSON not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)
        inst = cls(
            env,
            throttle_gain=float(params.get("throttle_gain_ratio", 1.0)),
            steer_gain=float(params.get("steer_gain_ratio", 1.0)),
            steer_tau_s=float(params.get("steer_tau_s", 0.0)),
            throttle_tau_s=float(params.get("throttle_tau_s", 0.0)),
            throttle_gain_override=throttle_gain_override,
            throttle_gain_floor=throttle_gain_floor,
            steer_gain_override=steer_gain_override,
            steer_gain_floor=steer_gain_floor,
            filter_dt_s=(
                filter_dt_s
                if filter_dt_s is not None
                else params.get("filter_dt_s", params.get("sim2real_filter_dt_s", None))
            ),
        )
        src = params.get("source", "unknown")
        cal = params.get("calibrated_at", "?")
        print(f"  loaded from {path.name}  (source={src}, date={cal})")
        return inst

    def action(self, action: np.ndarray) -> np.ndarray:
        raw_steer = float(action[0])
        raw_throttle = float(action[1])
        self.last_raw_action[0] = float(raw_steer)
        self.last_raw_action[1] = float(raw_throttle)

        steer = raw_steer * self.steer_gain
        throttle = raw_throttle * self.throttle_gain

        if self.filter_dt_s is not None:
            dt = float(self.filter_dt_s)
        else:
            now = time.monotonic()
            dt = 0.05 if self._last_t is None else max(now - self._last_t, 1e-3)
            self._last_t = now

        if self.steer_tau_s > 1e-4:
            alpha = 1.0 - math.exp(-dt / self.steer_tau_s)
            self._filtered_steer += alpha * (steer - self._filtered_steer)
            steer = self._filtered_steer

        if self.throttle_tau_s > 1e-4:
            alpha_t = 1.0 - math.exp(-dt / self.throttle_tau_s)
            self._filtered_throttle += alpha_t * (throttle - self._filtered_throttle)
            throttle = self._filtered_throttle

        out = action.copy() if isinstance(action, np.ndarray) else np.array(action, dtype=np.float32)
        out[0] = float(np.clip(steer, -1.0, 1.0))
        out[1] = float(np.clip(throttle, -1.0, 1.0))
        self.last_transformed_action[0] = float(out[0])
        self.last_transformed_action[1] = float(out[1])
        return out

    def reset(self, **kwargs):
        self._filtered_steer = 0.0
        self._filtered_throttle = 0.0
        self._last_t = None
        self.last_raw_action[:] = 0.0
        self.last_transformed_action[:] = 0.0
        return self.env.reset(**kwargs)


__all__ = ["Sim2RealActionWrapper"]
