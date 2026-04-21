"""
module/predictive_safety_filter.py

H 步预测安全滤波器 — V1

架构位置
--------
插在 ActionAdapterWrapper 输出（steer_target, throttle）之后，
ActionSafetyWrapper 之前：

    PPO → ActionAdapterWrapper
                ↓ [steer_target, throttle]
         PredictiveSafetyFilter.check()     ← 本模块
                ↓ 日志模式：原样透传
         ActionSafetyWrapper → DonkeyEnv

工作阶段
--------
Phase 1（log-only，当前默认）：
    只观测、不干预。每步对候选动作做 H=3 步前向预测，记录触发事件。
    基于统计结果（触发率、误报率、与真实事故的相关性）标定阈值。

Phase 2（intervene，阈值确定后启用）：
    触发时修改 steer_target，降低 throttle，保护真实车。

关键依赖
--------
- NeuralPhysicsDynamics（wm_real.pth）：前向预测引擎
- ActionSafetyWrapper 参数：delta_max=0.5, beta=0.6
  必须与实际 wrapper 精确一致，否则影子状态会累积漂移。

使用示例
--------
    from module import PredictiveSafetyFilter, PhysState

    flt = PredictiveSafetyFilter(
        model_path="models/world_model/wm_real.pth",
        horizon=3,
        mode="log",
        log_path="safety_filter_events.jsonl",
    )

    # 每 episode 开始
    flt.reset()

    # 每步
    triggered, preds, diag = flt.check(steer_target, throttle, phys, dt_ms=50.0)

    # env.step 执行后，同步真实 safety wrapper 状态
    flt.sync(safety_wrapper.steer_prev_limited, safety_wrapper.steer_prev_exec)

    # 结束后
    flt.print_stats()
    flt.close()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .world_model import NeuralPhysicsDynamics


# ─── 归一化范围（与 _build_state_v13 保持一致） ────────────────────

V_MAX    = 2.2   # m/s，v_long_norm = speed / V_MAX
GYRO_MAX = 4.0   # rad/s，yaw_rate_norm = -gyro_z / GYRO_MAX
ACCEL_MAX = 9.8  # m/s²，accel_x_norm = accel_x / ACCEL_MAX


# ─── 物理状态数据类 ────────────────────────────────────────────────

@dataclass
class PhysState:
    """
    已归一化的物理状态（与世界模型输入一致）。

    字段
    ----
    v_long    : float  ∈ [0,  2]   speed / 2.2
    yaw_rate  : float  ∈ [-2, 2]   -gyro_z / 4.0
    accel_x   : float  ∈ [-2, 2]   accel_x / 9.8
    """
    v_long: float
    yaw_rate: float
    accel_x: float

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [self.v_long, self.yaw_rate, self.accel_x], dtype=torch.float32
        )

    @classmethod
    def from_raw(cls, speed_mps: float, gyro_z: float, accel_x: float) -> "PhysState":
        """从原始传感器值构建（含符号约定和归一化）。"""
        import numpy as np
        return cls(
            v_long   = float(np.clip(speed_mps / V_MAX,     0.0,  2.0)),
            yaw_rate = float(np.clip(-gyro_z   / GYRO_MAX, -2.0,  2.0)),
            accel_x  = float(np.clip(accel_x   / ACCEL_MAX,-2.0,  2.0)),
        )


# ─── 影子 ActionSafetyWrapper 状态 ──────────────────────────────────

@dataclass
class _ShadowSafetyState:
    """
    精确复现 ActionSafetyWrapper 的内部状态，用于前向预测。
    必须与 control.py::ActionSafetyWrapper 参数完全一致。
    """
    steer_prev_limited: float = 0.0
    steer_prev_exec: float = 0.0
    delta_max: float = 0.5
    beta: float = 0.6

    def step(self, steer_target: float) -> Tuple[float, float]:
        """
        执行一步，返回 (steer_exec, steer_limited)，同时更新内部状态。
        完全对应 ActionSafetyWrapper.action() 逻辑。
        """
        delta = steer_target - self.steer_prev_limited
        if abs(delta) > self.delta_max:
            delta = max(-self.delta_max, min(self.delta_max, delta))
        steer_limited = self.steer_prev_limited + delta
        steer_exec = (1.0 - self.beta) * self.steer_prev_exec + self.beta * steer_limited
        steer_exec = max(-1.0, min(1.0, steer_exec))
        self.steer_prev_limited = steer_limited
        self.steer_prev_exec = steer_exec
        return steer_exec, steer_limited

    def copy(self) -> "_ShadowSafetyState":
        return _ShadowSafetyState(
            steer_prev_limited=self.steer_prev_limited,
            steer_prev_exec=self.steer_prev_exec,
            delta_max=self.delta_max,
            beta=self.beta,
        )


# ─── 主滤波器 ─────────────────────────────────────────────────────

class PredictiveSafetyFilter:
    """
    H 步预测安全滤波器。

    Parameters
    ----------
    model_path : str
        wm_real.pth 的路径。
    horizon : int
        前向预测步数，默认 3。
    delta_max : float
        ActionSafetyWrapper 的速率限制，必须与实际 wrapper 一致（默认 0.5）。
    beta : float
        ActionSafetyWrapper 的 LPF 系数，必须与实际 wrapper 一致（默认 0.6）。
    yaw_thresh : float or None
        |yaw_rate_norm| 超过此值时触发。None = 不设阈值（Phase 1 只统计）。
    decel_thresh : float or None
        单步 Δv_norm < -decel_thresh 触发（急减速预兆）。None = 不设阈值。
    mode : "log" or "intervene"
        "log" = Phase 1，只记录不干预；
        "intervene" = Phase 2，触发时修改动作。
    log_path : str
        触发事件 JSONL 日志路径。空字符串 = 不写日志。
    device : str
        推理设备（默认 "cpu"，目标 < 0.3 ms / 3 次前向）。
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
        log_path: str = "safety_filter_events.jsonl",
        device: str = "cpu",
    ):
        assert mode in ("log", "intervene"), (
            f"mode must be 'log' or 'intervene', got {mode!r}"
        )

        self.horizon     = int(horizon)
        self.mode        = mode
        self.yaw_thresh  = yaw_thresh
        self.decel_thresh = decel_thresh
        self.device      = device

        # 世界模型
        self.model = NeuralPhysicsDynamics.load_checkpoint(model_path, device=device)
        self.model.eval()

        # 影子 safety wrapper 状态
        self._shadow = _ShadowSafetyState(delta_max=delta_max, beta=beta)

        # 统计
        self._total_steps    = 0
        self._triggered_steps = 0
        self._episode        = 0

        # 日志
        self._log_path = log_path
        self._log_file = open(log_path, "a", encoding="utf-8") if log_path else None

        print(
            f"[PredictiveSafetyFilter] mode={mode}, H={horizon}, "
            f"delta_max={delta_max}, beta={beta}"
        )
        print(f"  yaw_thresh={yaw_thresh}, decel_thresh={decel_thresh}")
        print(f"  model : {model_path}")
        if log_path:
            print(f"  log   : {log_path}")

    # ── 主接口 ─────────────────────────────────────────────────────

    def check(
        self,
        steer_target: float,
        throttle: float,
        phys: PhysState,
        dt_ms: float = 50.0,
    ) -> Tuple[bool, List[Dict], Dict]:
        """
        对候选动作做 H 步前向预测。

        Parameters
        ----------
        steer_target : float
            ActionAdapterWrapper 输出的目标转向（未经 safety wrapper 滤波）。
        throttle : float
            ActionAdapterWrapper 输出的油门。
        phys : PhysState
            当前归一化物理状态。
        dt_ms : float
            控制步长（ms），用于 dt_norm 计算。

        Returns
        -------
        triggered : bool
            是否有任何预测步超出阈值。Phase 1 下仅统计，不影响控制。
        predictions : list[dict]
            H 步预测结果，每步包含 {h, v_long, yaw_rate, accel_x, delta_v, steer_exec}。
        diag : dict
            诊断信息：triggered, trigger_dims, trigger_rate, total_steps。
        """
        self._total_steps += 1

        # 用影子状态副本做推演（不污染真实影子状态）
        shadow = self._shadow.copy()
        phys_cur = phys.to_tensor().to(self.device)
        dt_norm = dt_ms / 50.0

        prev_steer_exec = shadow.steer_prev_exec
        prev_throttle   = float(throttle)

        predictions: List[Dict] = []
        triggered    = False
        trigger_dims: List[str] = []

        for h in range(self.horizon):
            # 1. 模拟 ActionSafetyWrapper，得到本步 steer_exec
            steer_exec, _ = shadow.step(float(steer_target))

            # 2. 构建 8D 输入向量
            x = torch.tensor(
                [[
                    float(phys_cur[0]),   # v_long
                    float(phys_cur[1]),   # yaw_rate
                    float(phys_cur[2]),   # accel_x
                    steer_exec,
                    float(throttle),
                    prev_steer_exec,
                    prev_throttle,
                    dt_norm,
                ]],
                dtype=torch.float32,
                device=self.device,
            )

            # 3. 世界模型前向
            with torch.no_grad():
                delta, phys_next = self.model(x, phys_cur.unsqueeze(0))

            phys_next = phys_next.squeeze(0)
            delta     = delta.squeeze(0)

            step_pred = {
                "h":          h + 1,
                "v_long":     float(phys_next[0]),
                "yaw_rate":   float(phys_next[1]),
                "accel_x":    float(phys_next[2]),
                "delta_v":    float(delta[0]),
                "steer_exec": steer_exec,
            }
            predictions.append(step_pred)

            # 4. 阈值检查（None = 只统计，不触发）
            if (self.yaw_thresh is not None
                    and abs(float(phys_next[1])) > self.yaw_thresh):
                triggered = True
                if "yaw_rate" not in trigger_dims:
                    trigger_dims.append("yaw_rate")

            if (self.decel_thresh is not None
                    and float(delta[0]) < -self.decel_thresh):
                triggered = True
                if "v_long_decel" not in trigger_dims:
                    trigger_dims.append("v_long_decel")

            # 5. 更新滚动状态
            prev_steer_exec = steer_exec
            prev_throttle   = float(throttle)
            phys_cur        = phys_next

        if triggered:
            self._triggered_steps += 1

        diag = {
            "triggered":    triggered,
            "trigger_dims": trigger_dims,
            "trigger_rate": self._triggered_steps / max(self._total_steps, 1),
            "total_steps":  self._total_steps,
        }

        # 写日志（触发时 + 每 500 步采样一次背景统计）
        if triggered or (self._total_steps % 500 == 0):
            self._write_log(steer_target, throttle, phys, predictions, trigger_dims)

        return triggered, predictions, diag

    def sync(self, steer_prev_limited: float, steer_prev_exec: float) -> None:
        """
        每步 env.step() 执行后，用真实 ActionSafetyWrapper 的状态同步影子状态。

        调用方式：
            flt.sync(
                safety_wrapper.steer_prev_limited,
                safety_wrapper.steer_prev_exec,
            )

        不调用此方法则影子状态会累积漂移，导致预测不准。
        """
        self._shadow.steer_prev_limited = float(steer_prev_limited)
        self._shadow.steer_prev_exec    = float(steer_prev_exec)

    def reset(self, episode: Optional[int] = None) -> None:
        """
        每个 episode 开始时调用。重置影子状态和 episode 计数。

        Parameters
        ----------
        episode : int or None
            手动指定 episode 编号；None = 自动递增。
        """
        self._shadow.steer_prev_limited = 0.0
        self._shadow.steer_prev_exec    = 0.0
        self._episode = int(episode) if episode is not None else self._episode + 1

    # ── 统计 ──────────────────────────────────────────────────────

    def stats(self) -> Dict:
        """返回当前统计摘要。"""
        return {
            "total_steps":     self._total_steps,
            "triggered_steps": self._triggered_steps,
            "trigger_rate":    self._triggered_steps / max(self._total_steps, 1),
            "mode":            self.mode,
            "horizon":         self.horizon,
            "yaw_thresh":      self.yaw_thresh,
            "decel_thresh":    self.decel_thresh,
        }

    def print_stats(self) -> None:
        s = self.stats()
        print(
            f"[SafetyFilter] steps={s['total_steps']:,}  "
            f"triggered={s['triggered_steps']:,}  "
            f"rate={s['trigger_rate']:.3%}  "
            f"mode={s['mode']}"
        )

    # ── 日志 ──────────────────────────────────────────────────────

    def _write_log(
        self,
        steer_target: float,
        throttle: float,
        phys: PhysState,
        predictions: List[Dict],
        trigger_dims: List[str],
    ) -> None:
        if self._log_file is None:
            return
        record = {
            "ts":          time.time(),
            "episode":     self._episode,
            "step":        self._total_steps,
            "phys":        {"v_long": phys.v_long, "yaw_rate": phys.yaw_rate,
                            "accel_x": phys.accel_x},
            "action":      {"steer_target": steer_target, "throttle": throttle},
            "predictions": predictions,
            "trigger_dims": trigger_dims,
        }
        self._log_file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._log_file.flush()

    def close(self) -> None:
        """关闭日志文件。"""
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def __del__(self):
        self.close()


__all__ = [
    "PredictiveSafetyFilter",
    "PhysState",
]
