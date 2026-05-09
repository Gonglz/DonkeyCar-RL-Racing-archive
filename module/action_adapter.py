"""
module/action_adapter.py

V13 ActionAdapterWrapper: 3D 高层动作 → 2D 低层控制量。

动作空间
--------
actor 输出: [Δsteer, speed_scale, line_bias] ∈ [-1, 1]³

  Δsteer      即时转向增量 → 积累到 steer_core
  speed_scale 相对基础速度的激进/保守系数
  line_bias   持续横向占位意图 → 低通滤波为 bias_smooth

传导链
------
Δsteer      → steer_core (积分器)  ┐
line_bias   → bias_smooth (低通)   ┤→ steer_target = clip(steer_core + k_b·bias_smooth)
speed_scale → v_ref (乘性缩放)     → PI+FF → throttle_cmd

输出: [steer_target, throttle_cmd]
  由下游 ActionSafetyWrapper 对 steer_target 做速率限制后送入 DonkeyEnv。

内部状态（暴露给 obs builder）
------------------------------
  steer_core   ∈ [-1, 1]  基础转向积分器
  bias_smooth  ∈ [-1, 1]  占位意图（低通滤波后）

接口兼容（替代 HighLevelControlWrapper）
-----------------------------------------
  consume_info(info)        更新 last_speed_mps
  last_low_level_action     np.ndarray shape (2,)
  diag                      Dict[str, float]
"""

from __future__ import annotations

from typing import Any, Dict

import gym
import numpy as np

from .utils import _clip_float


class ActionAdapterWrapper(gym.ActionWrapper):
    """V13 3D → 2D action adapter, replacing HighLevelControlWrapper."""

    def __init__(
        self,
        env,
        # ── steering ──
        k_delta: float = 0.10,
        lambda_bias: float = 0.20,
        k_bias: float = 0.10,
        steer_core_decay: float = 0.0,
        # ── speed ──
        v_nominal: float = 1.4,
        k_turn: float = 0.5,
        k_bias_speed: float = 0.0,
        alpha_speed: float = 0.25,
        v_min: float = 0.6,
        v_max: float = 1.8,
        # ── PI+FF speed controller ──
        speed_kp: float = 0.35,
        speed_ki: float = 0.08,
        speed_kff: float = 0.10,
        control_dt: float = 0.05,
        integral_limit: float = 3.0,
        max_throttle: float = 0.3,
        allow_reverse: bool = False,
        predictive_safety_filter=None,
    ):
        super().__init__(env)

        # steering params
        self.k_delta = float(k_delta)
        self.lambda_bias = float(lambda_bias)
        self.k_bias = float(k_bias)
        self.steer_core_decay = float(max(0.0, steer_core_decay))

        # speed params
        self.v_nominal = float(v_nominal)
        self.k_turn = float(k_turn)
        self.k_bias_speed = float(max(0.0, k_bias_speed))
        self.alpha_speed = float(alpha_speed)
        self.v_min = float(v_min)
        self.v_max = float(v_max)

        # PI+FF params
        self.speed_kp = float(speed_kp)
        self.speed_ki = float(speed_ki)
        self.speed_kff = float(speed_kff)
        self.control_dt = float(max(1e-3, control_dt))
        self.integral_limit = float(max(0.1, integral_limit))
        self.max_throttle = float(max(0.05, max_throttle))
        self.allow_reverse = bool(allow_reverse)
        self.predictive_safety_filter = predictive_safety_filter

        # 3D action space
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # internal state
        self.steer_core: float = 0.0
        self.bias_smooth: float = 0.0
        self.i_term: float = 0.0
        self.last_speed_mps: float = 0.0
        self.last_gyro_y: float = 0.0
        self.last_accel_x: float = 0.0
        self.last_low_level_action = np.array([0.0, 0.0], dtype=np.float32)
        self.diag: Dict[str, float] = self._zero_diag()

        print("✅ V13 ActionAdapterWrapper")
        print(
            f"   k_delta={self.k_delta:.3f}, lambda_bias={self.lambda_bias:.3f}, "
            f"k_bias={self.k_bias:.3f}, decay={self.steer_core_decay:.4f}"
        )
        print(
            f"   v_nominal={self.v_nominal:.2f}, k_turn={self.k_turn:.2f}, "
            f"k_bias_speed={self.k_bias_speed:.3f}, alpha={self.alpha_speed:.3f}"
        )
        print(
            f"   PI+FF: kp={self.speed_kp:.3f}, ki={self.speed_ki:.3f}, "
            f"kff={self.speed_kff:.3f}, dt={self.control_dt:.3f}, "
            f"max_thr={self.max_throttle:.2f}"
        )

    # ── helpers ──────────────────────────────────────────────

    def _zero_diag(self) -> Dict[str, float]:
        return {
            "v_target": 0.0,
            "v_meas": 0.0,
            "v_err": 0.0,
            "v_base": 0.0,
            "throttle_pi": 0.0,
            "target_steer": 0.0,
            "steer_core": 0.0,
            "bias_smooth": 0.0,
            "bias_offset": 0.0,
            "delta_steer_input": 0.0,
            "speed_scale_input": 0.0,
            "line_bias_input": 0.0,
            "safety_filter_triggered": 0.0,
            "safety_filter_trigger_rate": 0.0,
            "safety_filter_throttle_scale": 1.0,
        }

    # ── interface: consume_info ──────────────────────────────

    def consume_info(self, info: Dict[str, Any]) -> None:
        """Update speed measurement from env info (called by MultiInputObsWrapper after step)."""
        v = info.get("speed", 0.0)
        try:
            self.last_speed_mps = float(v)
        except Exception:
            self.last_speed_mps = 0.0
        gyro = info.get("gyro", (0.0, 0.0, 0.0))
        accel = info.get("accel", (0.0, 0.0, 0.0))
        try:
            self.last_gyro_y = float(gyro[1])
        except Exception:
            self.last_gyro_y = 0.0
        try:
            self.last_accel_x = float(accel[0])
        except Exception:
            self.last_accel_x = 0.0

    # ── core: action translation ─────────────────────────────

    def action(self, action: np.ndarray) -> np.ndarray:
        delta_steer = _clip_float(float(action[0]), -1.0, 1.0)
        speed_scale = _clip_float(float(action[1]), -1.0, 1.0)
        line_bias = _clip_float(float(action[2]), -1.0, 1.0)

        # 1. steering core integrator (optional decay)
        if self.steer_core_decay > 0.0:
            self.steer_core *= (1.0 - self.steer_core_decay)
        self.steer_core = _clip_float(
            self.steer_core + self.k_delta * delta_steer, -1.0, 1.0
        )

        # 2. line bias low-pass filter
        self.bias_smooth = (
            (1.0 - self.lambda_bias) * self.bias_smooth
            + self.lambda_bias * line_bias
        )
        bias_offset = self.k_bias * self.bias_smooth

        # 3. steering target (no rate limit here — ActionSafetyWrapper handles it)
        steer_target = _clip_float(self.steer_core + bias_offset, -1.0, 1.0)

        # 4. context-aware base speed
        v_base = _clip_float(
            self.v_nominal
            - self.k_turn * abs(steer_target)
            - self.k_bias_speed * abs(self.bias_smooth),
            self.v_min,
            self.v_max,
        )

        # 5. target speed
        v_ref = _clip_float(
            v_base * (1.0 + self.alpha_speed * speed_scale),
            self.v_min,
            self.v_max,
        )

        # 6. PI + feedforward speed controller
        v_meas = self.last_speed_mps
        v_err = v_ref - v_meas

        self.i_term = _clip_float(
            self.i_term + v_err * self.control_dt,
            -self.integral_limit,
            self.integral_limit,
        )

        throttle = (
            self.speed_kff * v_ref
            + self.speed_kp * v_err
            + self.speed_ki * self.i_term
        )

        if self.allow_reverse:
            throttle = _clip_float(throttle, -self.max_throttle, self.max_throttle)
        else:
            throttle = _clip_float(throttle, 0.0, self.max_throttle)

        safety_triggered = 0.0
        safety_trigger_rate = 0.0
        safety_throttle_scale = 1.0
        if self.predictive_safety_filter is not None:
            try:
                from .predictive_safety_filter import PhysState

                phys = PhysState.from_raw(
                    speed_mps=self.last_speed_mps,
                    gyro_z=self.last_gyro_y,
                    accel_x=self.last_accel_x,
                )
                triggered, _preds, diag = self.predictive_safety_filter.check(
                    steer_target=steer_target,
                    throttle=throttle,
                    phys=phys,
                    dt_ms=self.control_dt * 1000.0,
                )
                safety_triggered = float(bool(triggered))
                safety_trigger_rate = float(diag.get("trigger_rate", 0.0) or 0.0)
                safety_throttle_scale = float(diag.get("throttle_scale", 1.0) or 1.0)
                if bool(triggered) and str(getattr(self.predictive_safety_filter, "mode", "log")).lower() == "intervene":
                    throttle = _clip_float(throttle * safety_throttle_scale, 0.0, self.max_throttle)
            except Exception:
                safety_triggered = 0.0
                safety_trigger_rate = 0.0
                safety_throttle_scale = 1.0

        # 7. store output
        low_level = np.array([steer_target, throttle], dtype=np.float32)
        self.last_low_level_action = low_level
        self.diag = {
            "v_target": float(v_ref),
            "v_meas": float(v_meas),
            "v_err": float(v_err),
            "v_base": float(v_base),
            "throttle_pi": float(throttle),
            "target_steer": float(steer_target),
            "steer_core": float(self.steer_core),
            "bias_smooth": float(self.bias_smooth),
            "bias_offset": float(bias_offset),
            "delta_steer_input": float(delta_steer),
            "speed_scale_input": float(speed_scale),
            "line_bias_input": float(line_bias),
            "safety_filter_triggered": float(safety_triggered),
            "safety_filter_trigger_rate": float(safety_trigger_rate),
            "safety_filter_throttle_scale": float(safety_throttle_scale),
        }
        return low_level

    # ── reset ────────────────────────────────────────────────

    def reset(self, **kwargs):
        self.steer_core = 0.0
        self.bias_smooth = 0.0
        self.i_term = 0.0
        self.last_speed_mps = 0.0
        self.last_gyro_y = 0.0
        self.last_accel_x = 0.0
        self.last_low_level_action = np.array([0.0, 0.0], dtype=np.float32)
        self.diag = self._zero_diag()
        if self.predictive_safety_filter is not None:
            try:
                self.predictive_safety_filter.reset()
            except Exception:
                pass
        return self.env.reset(**kwargs)

    def close(self):
        if self.predictive_safety_filter is not None:
            try:
                self.predictive_safety_filter.close()
            except Exception:
                pass
        return self.env.close()


__all__ = ["ActionAdapterWrapper"]
