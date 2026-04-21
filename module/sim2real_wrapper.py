"""
module/sim2real_wrapper.py

Sim2Real 动力学对齐 Wrapper — 训练时使用

作用
----
在训练阶段将 Sim 动力学对齐到真实车，使 PPO 学到的策略能更直接部署。

插入位置（MultiSceneEnvV16._create_env 中）：
    base_env (gym DonkeyEnv)
         ↓
    ScenarioObstacleWrapper
         ↓
    Sim2RealActionWrapper   ← 本模块，缩放 steer / throttle
         ↓
    CanonicalSemanticWrapper / DonkeyRewardWrapper / ActionSafetyWrapper / ActionAdapterWrapper ...

参数来源
--------
由 mysim/tools/calibrate_sim2real.py 从 wm_real + wm_sim 自动计算后写入 JSON：

    {
      "throttle_gain_ratio": 0.115,
      "steer_gain_ratio": 0.715,
      "steer_tau_s": 0.0,
      "throttle_tau_s": 0.0,
      "source": "wm_calibration",
      "calibrated_at": "2026-04-19"
    }

标定文件位置
-----------
mysim/models/world_model/dynamics_alignment_wm.json
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
    """
    动力学对齐 gym.ActionWrapper。

    对发给 DonkeySim 的 [steer, throttle] 做缩放 + 可选一阶滞后，
    使 Sim 车的速度和转向响应更接近真实车。

    Parameters
    ----------
    env : gym.Env
        被包裹的环境。
    throttle_gain : float
        油门缩放比例。< 1.0 压制 sim 速度；标定值约 0.115。
    steer_gain : float
        转向缩放比例。< 1.0 减弱 sim 转向响应；标定值约 0.715。
    steer_tau_s : float
        额外转向一阶滞后时间常数（秒）。0.0 = 关闭。
    throttle_tau_s : float
        额外油门一阶滞后时间常数（秒）。0.0 = 关闭。
    """

    def __init__(
        self,
        env: gym.Env,
        throttle_gain: float = 1.0,
        steer_gain: float = 1.0,
        steer_tau_s: float = 0.0,
        throttle_tau_s: float = 0.0,
    ):
        super().__init__(env)
        self.throttle_gain  = float(max(0.01, throttle_gain))
        self.steer_gain     = float(max(0.01, steer_gain))
        self.steer_tau_s    = float(max(0.0, steer_tau_s))
        self.throttle_tau_s = float(max(0.0, throttle_tau_s))

        self._filtered_steer    = 0.0
        self._filtered_throttle = 0.0
        self._last_t: Optional[float] = None

        print(
            f"[Sim2RealActionWrapper] "
            f"throttle_gain={self.throttle_gain:.4f}, "
            f"steer_gain={self.steer_gain:.4f}, "
            f"steer_tau={self.steer_tau_s:.3f}s, "
            f"throttle_tau={self.throttle_tau_s:.3f}s"
        )

    @classmethod
    def from_json(cls, env: gym.Env, json_path: Union[str, Path]) -> "Sim2RealActionWrapper":
        """从标定 JSON 文件加载参数。"""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Sim2Real calibration JSON not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)
        inst = cls(
            env,
            throttle_gain  = float(params.get("throttle_gain_ratio", 1.0)),
            steer_gain     = float(params.get("steer_gain_ratio",    1.0)),
            steer_tau_s    = float(params.get("steer_tau_s",    0.0)),
            throttle_tau_s = float(params.get("throttle_tau_s", 0.0)),
        )
        src = params.get("source", "unknown")
        cal = params.get("calibrated_at", "?")
        print(f"  loaded from {path.name}  (source={src}, date={cal})")
        return inst

    def action(self, action: np.ndarray) -> np.ndarray:
        steer    = float(action[0])
        throttle = float(action[1])

        # 缩放
        steer    *= self.steer_gain
        throttle *= self.throttle_gain

        # 可选一阶滞后
        now = time.monotonic()
        dt  = 0.05 if self._last_t is None else max(now - self._last_t, 1e-3)
        self._last_t = now

        if self.steer_tau_s > 1e-4:
            alpha = 1.0 - math.exp(-dt / self.steer_tau_s)
            self._filtered_steer += alpha * (steer - self._filtered_steer)
            steer = self._filtered_steer

        if self.throttle_tau_s > 1e-4:
            alpha_t = 1.0 - math.exp(-dt / self.throttle_tau_s)
            self._filtered_throttle += alpha_t * (throttle - self._filtered_throttle)
            throttle = self._filtered_throttle

        steer    = float(np.clip(steer,    -1.0, 1.0))
        throttle = float(np.clip(throttle, -1.0, 1.0))

        out = action.copy() if isinstance(action, np.ndarray) else np.array(action, dtype=np.float32)
        out[0] = steer
        out[1] = throttle
        return out

    def reset(self, **kwargs):
        self._filtered_steer    = 0.0
        self._filtered_throttle = 0.0
        self._last_t = None
        return self.env.reset(**kwargs)


__all__ = ["Sim2RealActionWrapper"]
