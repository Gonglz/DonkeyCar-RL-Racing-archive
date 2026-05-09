"""
module/control.py

油门/转向控制链模块（V12/V13）。

该文件是控制相关逻辑的唯一主入口，包含：
- HighLevelControlWrapper
- ActionSafetyWrapper
- ThrottleControlWrapper
- CurvatureAwareThrottleWrapper（兼容保留，不建议在当前简化策略链中启用）

注意：
- 当前训练默认更简化：重点保留转向执行保护，不在主链里启用“曲率油门收紧”。
- 为兼容旧调用，ActionSafetyWrapper 仍保留若干旧参数，但默认不使用自适应策略。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import gym
import numpy as np

from .utils import _clip_float


class HighLevelControlWrapper(gym.ActionWrapper):
    """
    高层速度控制器。

    输入动作（策略输出）：
      action = [target_steer, target_speed_norm]

    输出动作（低层执行）：
      low_level = [steer_cmd, throttle_cmd]

    速度控制律（PI + 前馈）：
      v_tgt = target_speed_norm * speed_vmax
      v_err = v_tgt - v_meas
      i_term = clip(i_term + v_err * dt, -integral_limit, +integral_limit)
      throttle = kff*v_tgt + kp*v_err + ki*i_term
    """

    def __init__(
        self,
        env,
        speed_vmax: float = 2.2,
        speed_kp: float = 0.35,
        speed_ki: float = 0.08,
        speed_kff: float = 0.10,
        control_dt: float = 0.05,
        integral_limit: float = 3.0,
        max_throttle: float = 0.3,
        allow_reverse: bool = False,
        soft_brake_deadband: float = 0.05,
    ):
        super().__init__(env)
        self.speed_vmax = float(max(0.05, speed_vmax))
        self.speed_kp = float(speed_kp)
        self.speed_ki = float(speed_ki)
        self.speed_kff = float(speed_kff)
        self.control_dt = float(max(1e-3, control_dt))
        self.integral_limit = float(max(0.1, integral_limit))
        self.max_throttle = float(max(0.05, max_throttle))
        self.allow_reverse = bool(allow_reverse)
        self.soft_brake_deadband = float(max(0.0, soft_brake_deadband))

        if self.allow_reverse:
            self.action_space = gym.spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = gym.spaces.Box(
                low=np.array([-1.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )

        self.i_term = 0.0
        self.last_speed_mps = 0.0
        self.last_low_level_action = np.array([0.0, 0.0], dtype=np.float32)
        self.diag: Dict[str, float] = {
            "v_target": 0.0,
            "v_meas": 0.0,
            "v_err": 0.0,
            "throttle_pi": 0.0,
            "target_steer": 0.0,
        }

        print("✅ V12 HighLevelControlWrapper")
        print(
            f"   speed_vmax={self.speed_vmax:.2f}, kp={self.speed_kp:.3f}, "
            f"ki={self.speed_ki:.3f}, kff={self.speed_kff:.3f}, dt={self.control_dt:.3f}"
        )
        print(f"   allow_reverse={self.allow_reverse}, max_throttle={self.max_throttle:.2f}")

    def consume_info(self, info: Dict[str, Any]) -> None:
        """每步结束后更新速度测量值，供下一步 PI 使用。"""
        v = info.get("speed", 0.0)
        try:
            self.last_speed_mps = float(v)
        except Exception:
            self.last_speed_mps = 0.0

    def action(self, action: np.ndarray) -> np.ndarray:
        steer_tgt = _clip_float(float(action[0]), -1.0, 1.0)

        if self.allow_reverse:
            v_norm = _clip_float(float(action[1]), -1.0, 1.0)
        else:
            v_norm = _clip_float(float(action[1]), 0.0, 1.0)

        v_tgt = float(v_norm * self.speed_vmax)
        v_meas = float(self.last_speed_mps)
        v_err = float(v_tgt - v_meas)

        self.i_term = _clip_float(
            self.i_term + v_err * self.control_dt,
            -self.integral_limit,
            self.integral_limit,
        )

        thr = (
            self.speed_kff * v_tgt
            + self.speed_kp * v_err
            + self.speed_ki * self.i_term
        )

        # 不允许倒车时：明显超速则不允许继续正油门。
        if (not self.allow_reverse) and (v_err < -self.soft_brake_deadband):
            thr = min(thr, 0.0)

        if self.allow_reverse:
            thr = _clip_float(thr, -self.max_throttle, self.max_throttle)
        else:
            thr = _clip_float(thr, 0.0, self.max_throttle)

        low_level = np.array([steer_tgt, thr], dtype=np.float32)
        self.last_low_level_action = low_level
        self.diag = {
            "v_target": float(v_tgt),
            "v_meas": float(v_meas),
            "v_err": float(v_err),
            "throttle_pi": float(thr),
            "target_steer": float(steer_tgt),
        }
        return low_level

    def reset(self, **kwargs):
        self.i_term = 0.0
        self.last_speed_mps = 0.0
        self.last_low_level_action = np.array([0.0, 0.0], dtype=np.float32)
        self.diag = {
            "v_target": 0.0,
            "v_meas": 0.0,
            "v_err": 0.0,
            "throttle_pi": 0.0,
            "target_steer": 0.0,
        }
        return self.env.reset(**kwargs)


class ActionSafetyWrapper(gym.ActionWrapper):
    """
    转向执行保护（简化版）。

    当前策略：
    - 固定转向速率限制（delta_max）
    - 可选一阶低通滤波（LPF）
    - 不启用曲率/意图自适应放宽（为降低复杂度）

    兼容性：
    - 为保持历史脚本不报错，保留 adaptive/hairpin 相关参数，但不参与决策。
    """

    def __init__(
        self,
        env,
        delta_max: float = 0.5,
        enable_lpf: bool = True,
        beta: float = 0.6,
        delta_delta_max: Optional[float] = None,
        servo_deadband: float = 0.0,
        adaptive_delta_max: bool = False,
        curve_delta_boost: float = 0.0,
        curve_kappa_ref: float = 0.15,
        steer_intent_boost: float = 0.0,
        hairpin_curve_ratio: float = 0.85,
        hairpin_min_delta_max: float = 0.45,
        hairpin_max_delta_max: float = 0.85,
    ):
        super().__init__(env)
        self.delta_max = float(np.clip(delta_max, 0.0, 1.0))
        self.enable_lpf = bool(enable_lpf)
        self.beta = float(np.clip(beta, 0.0, 1.0))
        self.delta_delta_max = (
            None
            if delta_delta_max is None or float(delta_delta_max) <= 0.0
            else float(np.clip(delta_delta_max, 0.0, 1.0))
        )
        self.servo_deadband = float(np.clip(servo_deadband, 0.0, 0.2))
        self.last_kappa_abs = 0.0

        # 兼容旧参数（仅保留字段，不参与决策）
        self.adaptive_delta_max = bool(adaptive_delta_max)
        self.curve_delta_boost = float(max(0.0, curve_delta_boost))
        self.curve_kappa_ref = float(max(1e-6, curve_kappa_ref))
        self.steer_intent_boost = float(max(0.0, steer_intent_boost))
        self.hairpin_curve_ratio = float(np.clip(hairpin_curve_ratio, 0.0, 1.0))
        self.hairpin_min_delta_max = float(np.clip(hairpin_min_delta_max, 0.0, 1.0))
        self.hairpin_max_delta_max = float(np.clip(hairpin_max_delta_max, 0.0, 1.0))

        self.steer_prev_limited = 0.0
        self.steer_prev_exec = 0.0
        self.steer_delta_prev_limited = 0.0
        self.delta_steer_prev = 0.0
        self.diag: Dict[str, Any] = self._zero_diag()

        print(
            f"🛡️  ActionSafetyWrapper(simple): delta_max={self.delta_max:.3f}, "
            f"LPF={'on' if self.enable_lpf else 'off'} (beta={self.beta:.2f}), "
            f"ddelta_max={self.delta_delta_max if self.delta_delta_max is not None else 0.0:.3f}, "
            f"deadband={self.servo_deadband:.3f}, "
            f"adaptive=off"
        )

    def _zero_diag(self) -> Dict[str, Any]:
        return {
            "steer_raw": 0.0,
            "steer_exec": 0.0,
            "delta_steer": 0.0,
            "delta_steer_prev": 0.0,
            "rate_limit_hit": False,
            "rate_excess_raw": 0.0,
            "rate_excess_bounded": 0.0,
            "delta_delta_steer": 0.0,
            "delta_delta_steer_abs": 0.0,
            "delta_delta_limit_hit": False,
            "delta_delta_excess_raw": 0.0,
            "delta_delta_excess_bounded": 0.0,
            "servo_deadband_hold": False,
            "steer_clip_hit": False,
            "mismatch": 0.0,
            "effective_delta_max": float(self.delta_max),
            "kappa_for_limit": 0.0,
            "hairpin_relax_active": False,
        }

    def consume_info(self, info: Dict[str, Any]) -> None:
        # 仅用于日志兼容
        if not isinstance(info, dict):
            return
        kappa = info.get("geo/kappa", info.get("kappa_lookahead", 0.0))
        try:
            self.last_kappa_abs = float(abs(float(kappa)))
        except Exception:
            self.last_kappa_abs = 0.0

    def action(self, action: np.ndarray) -> np.ndarray:
        steer_raw = float(action[0])
        throttle = float(action[1])

        # 固定速率上限，不做自适应放宽。
        effective_delta_max = float(self.delta_max)
        delta = steer_raw - self.steer_prev_limited
        rate_limit_hit = abs(delta) > effective_delta_max
        rate_excess_raw = (
            max(0.0, abs(delta) - effective_delta_max) / max(effective_delta_max, 1e-6)
            if effective_delta_max > 0.0
            else 0.0
        )
        if effective_delta_max > 0.0 and abs(delta) > effective_delta_max:
            delta = np.clip(delta, -effective_delta_max, effective_delta_max)

        delta_delta = float(delta - self.steer_delta_prev_limited)
        delta_delta_limit_hit = False
        delta_delta_excess_raw = 0.0
        if self.delta_delta_max is not None and self.delta_delta_max > 0.0:
            delta_delta_limit_hit = abs(delta_delta) > self.delta_delta_max
            delta_delta_excess_raw = (
                max(0.0, abs(delta_delta) - self.delta_delta_max)
                / max(self.delta_delta_max, 1e-6)
            )
            if delta_delta_limit_hit:
                delta_delta = float(np.clip(delta_delta, -self.delta_delta_max, self.delta_delta_max))
                delta = float(self.steer_delta_prev_limited + delta_delta)

        servo_deadband_hold = False
        if self.servo_deadband > 0.0 and abs(delta) < self.servo_deadband:
            delta = 0.0
            servo_deadband_hold = True
        steer_limited = self.steer_prev_limited + delta

        if self.enable_lpf:
            steer_exec = (1.0 - self.beta) * self.steer_prev_exec + self.beta * steer_limited
        else:
            steer_exec = steer_limited

        steer_exec = float(np.clip(steer_exec, -1.0, 1.0))
        steer_clip_hit = abs(steer_exec) > 0.95

        actual_delta = steer_exec - self.steer_prev_exec
        self.diag = {
            "steer_raw": steer_raw,
            "steer_exec": steer_exec,
            "delta_steer": actual_delta,
            "delta_steer_prev": self.delta_steer_prev,
            "rate_limit_hit": rate_limit_hit,
            "rate_excess_raw": rate_excess_raw,
            "rate_excess_bounded": float(np.tanh(rate_excess_raw)),
            "delta_delta_steer": float(delta - self.steer_delta_prev_limited),
            "delta_delta_steer_abs": abs(float(delta - self.steer_delta_prev_limited)),
            "delta_delta_limit_hit": delta_delta_limit_hit,
            "delta_delta_excess_raw": delta_delta_excess_raw,
            "delta_delta_excess_bounded": float(np.tanh(delta_delta_excess_raw)),
            "servo_deadband_hold": servo_deadband_hold,
            "steer_clip_hit": steer_clip_hit,
            "mismatch": steer_raw - steer_exec,
            "effective_delta_max": effective_delta_max,
            "kappa_for_limit": float(self.last_kappa_abs),
            "hairpin_relax_active": False,
        }
        self.steer_delta_prev_limited = float(delta)
        self.delta_steer_prev = actual_delta
        self.steer_prev_limited = steer_limited
        self.steer_prev_exec = steer_exec
        return np.array([steer_exec, throttle], dtype=np.float32)

    def reset(self, **kwargs):
        self.steer_prev_limited = 0.0
        self.steer_prev_exec = 0.0
        self.steer_delta_prev_limited = 0.0
        self.delta_steer_prev = 0.0
        self.last_kappa_abs = 0.0
        self.diag = self._zero_diag()
        return self.env.reset(**kwargs)


class ThrottleControlWrapper(gym.ActionWrapper):
    """全局油门边界裁剪。"""

    def __init__(self, env, max_throttle: float = 0.3, min_throttle: float = 0.0):
        super().__init__(env)
        self.max_throttle = float(max_throttle)
        self.min_throttle = float(min_throttle)
        if min_throttle > 0:
            print(f"🆕 ThrottleControl: [{min_throttle:.2f}, {max_throttle:.2f}]")
        else:
            print(f"🆕 ThrottleControl: max={max_throttle:.2f}")

    def action(self, action: np.ndarray) -> np.ndarray:
        return np.array(
            [action[0], np.clip(action[1], self.min_throttle, self.max_throttle)],
            dtype=np.float32,
        )


class CurvatureAwareThrottleWrapper(gym.Wrapper):
    """
    曲率感知油门上限（兼容保留）。

    当前简化训练链默认不启用该 wrapper。
    如需启用，可在构建链路中显式插入。
    """

    def __init__(
        self,
        env,
        track_geometry,
        scene_key: str,
        base_throttle_max: float = 0.35,
        min_throttle_max: float = 0.15,
        curvature_scale: float = 0.1,
    ):
        super().__init__(env)
        self.track_geometry = track_geometry
        self.scene_key = scene_key
        self.base_throttle_max = float(base_throttle_max)
        self.min_throttle_max = float(min_throttle_max)
        self.curvature_scale = float(curvature_scale)
        self._prev_track_idx: Optional[int] = None
        self._last_kappa_abs = 0.0

        geom = self.track_geometry.scenes[scene_key]
        n = len(geom.center)
        curvature = np.zeros(n, dtype=np.float64)
        for i in range(n):
            prev_i = (i - 1) % n
            next_i = (i + 1) % n
            v1 = geom.center[i] - geom.center[prev_i]
            v2 = geom.center[next_i] - geom.center[i]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-6 and n2 > 1e-6:
                u1 = v1 / n1
                u2 = v2 / n2
                curvature[i] = abs(u1[0] * u2[1] - u1[1] * u2[0])
        self.curvature = curvature
        self.max_curvature = float(np.max(curvature)) if len(curvature) > 0 else 0.1
        self.kappa_norm_ref = float(max(self.max_curvature, self.curvature_scale, 1e-3))
        self._last_throttle_max_local = float(self.base_throttle_max)

    def _throttle_cap(self, kappa_abs: float) -> float:
        kappa_norm = float(np.clip(kappa_abs / self.kappa_norm_ref, 0.0, 1.0))
        return float(
            np.clip(
                self.base_throttle_max * (1.0 - kappa_norm),
                self.min_throttle_max,
                self.base_throttle_max,
            )
        )

    def _kappa_from_info(self, info: Dict[str, Any]) -> float:
        try:
            pos = info.get("pos", (0.0, 0.0, 0.0))
            x, z = float(pos[0]), float(pos[2])
            car = info.get("car", (0.0, 0.0, 0.0))
            yaw_deg = float(car[2]) if len(car) > 2 else 0.0
            geo = self.track_geometry.query(
                self.scene_key,
                x=x,
                z=z,
                yaw_rad=np.deg2rad(yaw_deg),
                prev_idx=self._prev_track_idx,
            )
            self._prev_track_idx = int(geo.get("idx", 0))
            kappa = abs(float(geo.get("kappa_lookahead", 0.0)))
            if np.isfinite(kappa):
                return kappa
        except Exception:
            pass

        if len(self.curvature) > 0:
            try:
                cur_idx = int(info.get("closest_track_idx", self._prev_track_idx or 0)) % len(self.curvature)
                self._prev_track_idx = cur_idx
                kappa = abs(float(self.curvature[cur_idx]))
                if np.isfinite(kappa):
                    return kappa
            except Exception:
                pass
        return 0.0

    def step(self, action):
        a = np.array(action, dtype=np.float32).copy()
        throttle_raw = float(a[1]) if a.shape[0] > 1 else 0.0
        throttle_cap_now = self._throttle_cap(self._last_kappa_abs)
        if a.shape[0] > 1:
            a[1] = float(np.clip(throttle_raw, 0.0, throttle_cap_now))

        obs, reward, done, info = self.env.step(a)

        kappa_abs = self._kappa_from_info(info)
        self._last_kappa_abs = float(kappa_abs)
        self._last_throttle_max_local = self._throttle_cap(kappa_abs)

        info["throttle_max_local"] = float(self._last_throttle_max_local)
        info["curvature_local"] = float(kappa_abs)
        info["ctrl/throttle_cmd_raw"] = float(throttle_raw)
        info["ctrl/throttle_cmd_exec"] = float(a[1]) if a.shape[0] > 1 else 0.0
        info["ctrl/throttle_clip_hit"] = float((a.shape[0] > 1) and (throttle_raw > float(a[1]) + 1e-6))
        return obs, reward, done, info

    def reset(self, **kwargs):
        self._prev_track_idx = None
        self._last_kappa_abs = 0.0
        self._last_throttle_max_local = float(self.base_throttle_max)
        return self.env.reset(**kwargs)


__all__ = [
    "HighLevelControlWrapper",
    "ActionSafetyWrapper",
    "ThrottleControlWrapper",
    "CurvatureAwareThrottleWrapper",
]
