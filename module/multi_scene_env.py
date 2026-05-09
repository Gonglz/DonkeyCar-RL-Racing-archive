"""
module/multi_scene_env.py

包含：
  - MultiSceneEnv        : V9 多场景切换环境（与 ppo_waveshare_v9.MultiSceneEnv 等价）
  - HighLevelControlWrapper : V12 高层速度控制动作包装器（已迁移到 module/control.py）
  - MultiInputObsWrapper : V12 Dict观测（image + state）
  - MultiSceneEnvV12     : 继承 MultiSceneEnv，_create_env 使用 V12 全链路
"""

import json
import math
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gym
import numpy as np
from stable_baselines3.common.monitor import Monitor

try:
    import torch
except Exception:
    torch = None

from .utils import (
    ENV_DOMAIN_MAP,
    MONITOR_INFO_KEYS,
    _get_domain_for_env,
    _seed_everything,
    _safe_seed_env,
    _clip_float,
    _wrap_pi,
)
from .wrappers import (
    GeneralizationWrapper,
    TransposeWrapper,
    NormalizeWrapper,
    V9YellowLaneWrapper,
    GTResetPerturbWrapper,
    RGBResizeWrapper,
)
from .control import (
    HighLevelControlWrapper,
    ActionSafetyWrapper,
)
from .reward import DonkeyRewardWrapper, ImprovedRewardWrapperV3, V9DomainRewardWrapper


# ============================================================
# 自定义 episode_over 判定（去掉 missed_checkpoint / dq 检查）
# ============================================================
def _donkey_episode_over_no_checkpoint(handler):
    """替换 DonkeyUnitySimHandler.determine_episode_over:
    只保留 CTE 超限 和 碰撞 两种终止条件；
    去掉 missed_checkpoint / dq，避免 sim 内部检查点导致提前截断 episode。
    """
    if math.fabs(handler.cte) > 2 * handler.max_cte:
        pass  # 启动时 CTE 可能瞬间很大，忽略
    elif math.fabs(handler.cte) > handler.max_cte:
        handler.over = True
    elif handler.hit != "none":
        handler.over = True
    # NOTE: missed_checkpoint / dq 不再触发 episode 终止


def _install_custom_episode_over(base_env):
    """在 base_env (DonkeyEnv) 上安装自定义 episode_over 函数。"""
    if hasattr(base_env, "set_episode_over_fn"):
        base_env.set_episode_over_fn(_donkey_episode_over_no_checkpoint)
        print("   [episode_over] 已安装自定义判定（跳过 missed_checkpoint/dq）")
    else:
        print("   [episode_over] ⚠️ base_env 无 set_episode_over_fn，跳过")


def _set_handler_max_cte(base_env, max_cte: float, logging_key: str = ""):
    """per-scene 设置 handler.max_cte，用于缩短出轨后无效步数。"""
    try:
        handler = base_env.viewer.handler
        old_val = getattr(handler, "max_cte", None)
        handler.max_cte = float(max_cte)
        if old_val is not None and abs(old_val - max_cte) > 0.01:
            print(f"   [{logging_key}] max_cte: {old_val:.1f} → {max_cte:.1f}")
    except Exception as e:
        print(f"   [{logging_key}] ⚠️ set max_cte failed: {type(e).__name__}: {e}")


def _clear_handler_over(base_env) -> None:
    """清理底层 handler.over，避免 reset 后残留 done 状态污染下一回合。"""
    try:
        handler = base_env.viewer.handler
        handler.over = False
    except Exception:
        pass


# ============================================================
# 多场景切换环境（V9 逻辑，仅依赖 module imports）
# ============================================================
class MultiSceneEnv(gym.Env):
    """
    多场景交替训练环境（单模拟器 + 场景切换方案）
    V9.3c 兼容版，内部改用 module/ 下的类，不再依赖 ppo_waveshare_v9/v8。
    """

    def __init__(
        self,
        env_ids: List[str],
        conf: Dict[str, Any],
        scene_weights: List[float] = None,
        scene_log_keys: Optional[List[str]] = None,
        target_size=(128, 128),
        enable_dr: bool = False,
        total_timesteps: int = 200000,
        delta_max: float = 0.10,
        enable_lpf: bool = True,
        beta: float = 0.6,
        gt_delta_max: float = 0.18,
        gt_enable_lpf: bool = False,
        gt_beta: float = 0.6,
        w_d: float = 0.25,
        w_dd: float = 0.08,
        w_m: float = 0.08,
        w_sat: float = 0.15,
        gt_w_d: float = 0.08,
        gt_w_dd: float = 0.02,
        gt_w_m: Optional[float] = None,
        gt_w_sat: float = 0.03,
        ws_lap_reward_scale: float = 0.30,
        gt_lap_reward_scale: float = 1.0,
        gt_min_throttle: float = 0.08,
        gt_reset_perturb_steps_lo: int = 0,
        gt_reset_perturb_steps_hi: int = 0,
        gt_reset_perturb_steer: float = 0.25,
        gt_reset_perturb_throttle_lo: float = 0.08,
        gt_reset_perturb_throttle_hi: float = 0.16,
        detector_kwargs: Optional[dict] = None,
        min_episodes_per_scene: int = 20,
        max_steps_per_scene: Optional[int] = 4096,
        enable_dynamic_scene_weights: bool = True,
        dynamic_weight_update_episodes: int = 20,
        dynamic_weight_window: int = 60,
        dynamic_min_samples_per_scene: int = 8,
        dynamic_weight_alpha: float = 2.0,
        dynamic_length_beta: float = 0.5,
        dynamic_gt_prior: float = 1.5,
        dynamic_weight_smoothing: float = 0.35,
        dynamic_weight_min: float = 0.02,
        dynamic_weight_max: float = 0.98,
        dynamic_success_mode: str = "scene_adaptive",
        dynamic_success_warmup_episodes: int = 1200,
        dynamic_success_post_warmup_scale: float = 0.20,
        dynamic_success_deficit_mix: float = 0.85,
        enable_step_balance_sampling: bool = True,
        step_balance_sampling_mix: float = 0.85,
        step_balance_mask: Optional[List[bool]] = None,
        scene_reload_timeout_s: float = 25.0,
        scene_reload_post_exit_sleep_s: float = 1.0,
        scene_start_force_reload: bool = True,
        diff_mode: str = "dr_rgb",
        mask_mode: str = "lane",
        mask_scale: float = 1.0,
    ):
        super().__init__()

        self.env_ids = env_ids
        self.conf = conf
        self.scene_weights = list(scene_weights) if scene_weights is not None else [1.0 / len(env_ids)] * len(env_ids)
        total = sum(self.scene_weights)
        self.scene_weights = [w / total for w in self.scene_weights]
        self.base_scene_weights = list(self.scene_weights)
        self.scene_domains = [_get_domain_for_env(eid) for eid in env_ids]
        if scene_log_keys is not None:
            if len(scene_log_keys) != len(env_ids):
                raise ValueError(
                    f"scene_log_keys length({len(scene_log_keys)}) must match env_ids({len(env_ids)})"
                )
            self.scene_log_keys = [
                (str(k).strip() if str(k).strip() else self._infer_scene_log_key(eid))
                for k, eid in zip(scene_log_keys, env_ids)
            ]
        else:
            self.scene_log_keys = [self._infer_scene_log_key(eid) for eid in env_ids]

        self.target_size = target_size
        self.enable_dr = enable_dr
        self.total_timesteps = total_timesteps
        self.delta_max = delta_max
        self.enable_lpf = enable_lpf
        self.beta = beta
        self.gt_delta_max = gt_delta_max
        self.gt_enable_lpf = gt_enable_lpf
        self.gt_beta = gt_beta
        self.w_d = w_d
        self.w_dd = w_dd
        self.w_m = w_m
        self.w_sat = w_sat
        self.gt_w_d = gt_w_d
        self.gt_w_dd = gt_w_dd
        self.gt_w_m = w_m if gt_w_m is None else gt_w_m
        self.gt_w_sat = gt_w_sat
        self.ws_lap_reward_scale = float(ws_lap_reward_scale)
        self.gt_lap_reward_scale = float(gt_lap_reward_scale)
        self.gt_min_throttle = gt_min_throttle
        self.gt_reset_perturb_steps_lo = int(max(0, gt_reset_perturb_steps_lo))
        self.gt_reset_perturb_steps_hi = int(max(self.gt_reset_perturb_steps_lo, gt_reset_perturb_steps_hi))
        self.gt_reset_perturb_steer = float(max(0.0, gt_reset_perturb_steer))
        self.gt_reset_perturb_throttle_lo = float(max(0.0, gt_reset_perturb_throttle_lo))
        self.gt_reset_perturb_throttle_hi = float(max(self.gt_reset_perturb_throttle_lo, gt_reset_perturb_throttle_hi))
        self.detector_kwargs = detector_kwargs or {}
        self.diff_mode = str(diff_mode)
        self.mask_mode = str(mask_mode)
        self.mask_scale = float(mask_scale)

        self.min_episodes_per_scene = min_episodes_per_scene
        self.max_steps_per_scene = None if (max_steps_per_scene is None or max_steps_per_scene <= 0) else int(max_steps_per_scene)

        self.enable_dynamic_scene_weights = bool(enable_dynamic_scene_weights and len(env_ids) > 1)
        self.dynamic_weight_update_episodes = max(1, int(dynamic_weight_update_episodes))
        self.dynamic_weight_window = max(10, int(dynamic_weight_window))
        self.dynamic_min_samples_per_scene = max(2, int(dynamic_min_samples_per_scene))
        self.dynamic_weight_alpha = float(dynamic_weight_alpha)
        self.dynamic_length_beta = float(dynamic_length_beta)
        self.dynamic_gt_prior = float(dynamic_gt_prior)
        self.dynamic_weight_smoothing = float(np.clip(dynamic_weight_smoothing, 0.0, 1.0))
        self.dynamic_weight_min = float(max(0.0, dynamic_weight_min))
        # per-scene dynamic_weight_max: float → 全局统一, List[float] → 每场景独立上限
        if isinstance(dynamic_weight_max, (list, tuple)):
            self.dynamic_weight_max_per_scene = [float(min(1.0, x)) for x in dynamic_weight_max]
            if len(self.dynamic_weight_max_per_scene) != len(env_ids):
                print(f"⚠️  dynamic_weight_max list length({len(self.dynamic_weight_max_per_scene)}) != env_ids({len(env_ids)}), 回退全局")
                self.dynamic_weight_max_per_scene = [0.55] * len(env_ids)
            self.dynamic_weight_max = float(max(self.dynamic_weight_max_per_scene))  # 兼容日志
        else:
            self.dynamic_weight_max = float(min(1.0, dynamic_weight_max))
            self.dynamic_weight_max_per_scene = [self.dynamic_weight_max] * len(env_ids)
        if self.dynamic_weight_min >= self.dynamic_weight_max:
            self.dynamic_weight_min, self.dynamic_weight_max = 0.02, 0.98
            self.dynamic_weight_max_per_scene = [self.dynamic_weight_max] * len(env_ids)
        self.dynamic_success_mode = str(dynamic_success_mode or "env_done").strip().lower()
        if self.dynamic_success_mode not in ("env_done", "scene_adaptive", "scene_strict"):
            print(f"⚠️  dynamic_success_mode={dynamic_success_mode} 无效，回退为 env_done")
            self.dynamic_success_mode = "env_done"
        self.dynamic_success_warmup_episodes = max(1, int(dynamic_success_warmup_episodes))
        self.dynamic_success_post_warmup_scale = float(np.clip(dynamic_success_post_warmup_scale, 0.0, 1.0))
        self.dynamic_success_deficit_mix = float(np.clip(dynamic_success_deficit_mix, 0.0, 1.0))
        self.enable_step_balance_sampling = bool(enable_step_balance_sampling and len(env_ids) > 1)
        self.step_balance_sampling_mix = float(np.clip(step_balance_sampling_mix, 0.0, 1.0))
        self.scene_reload_timeout_s = float(max(1.0, scene_reload_timeout_s))
        self.scene_reload_post_exit_sleep_s = float(np.clip(scene_reload_post_exit_sleep_s, 0.0, 3.0))
        self.scene_start_force_reload = bool(scene_start_force_reload)
        # step_balance_mask: True=主训练场景(参与步数缺口校正), False=预览/眉熟场景(不参与)
        if step_balance_mask is not None:
            if len(step_balance_mask) != len(env_ids):
                raise ValueError(
                    f"step_balance_mask length({len(step_balance_mask)}) must match env_ids({len(env_ids)})"
                )
            self.step_balance_mask = list(step_balance_mask)
        else:
            self.step_balance_mask = [True] * len(env_ids)
        self.step_balance_min_samples_per_scene = 6
        self._scene_recent_rewards = [deque(maxlen=self.dynamic_weight_window) for _ in env_ids]
        self._scene_recent_lengths = [deque(maxlen=self.dynamic_weight_window) for _ in env_ids]
        # success 定义由 dynamic_success_mode 控制（env_done / scene_adaptive / scene_strict）
        self._scene_recent_success = [deque(maxlen=self.dynamic_weight_window) for _ in env_ids]
        # hard_success 始终使用 env_done/lap 原始规则，供日志与对照分析
        self._scene_recent_hard_success = [deque(maxlen=self.dynamic_weight_window) for _ in env_ids]
        self._scene_best_mean_reward = [-np.inf for _ in env_ids]
        self.completed_episode_count = 0

        # 方案B: 基于累积步数的动态权重（用于补偿短/长episode差异）
        self._total_steps_per_scene = [0] * len(env_ids)  # 累积每个scene的总步数
        self.use_step_based_weights = True  # 启用step-based权重调整

        self.active_env = None
        self.active_scene_idx = 0
        self.episode_count = 0
        self.scene_episode_counts = [0] * len(env_ids)
        self._cur_scene_episodes = 0
        self._cur_scene_steps = 0
        # 场景采样/调权诊断缓存（供 callback 周期记录）
        self._last_sampling_candidates: List[int] = [0] if len(env_ids) > 0 else []
        self._last_sampling_probs: List[float] = [1.0] if len(env_ids) > 0 else []
        self._last_sampling_reason: str = "init"
        self._last_sampling_choice: int = 0
        self._last_sampling_used_step_balance: bool = False
        self._last_dynamic_reward_means: List[float] = [float("nan")] * len(env_ids)
        self._last_dynamic_len_means: List[float] = [float("nan")] * len(env_ids)
        self._last_dynamic_success_means: List[float] = [float("nan")] * len(env_ids)
        self._last_dynamic_hard_success_means: List[float] = [float("nan")] * len(env_ids)
        self._last_dynamic_deficits: List[float] = [float("nan")] * len(env_ids)
        self._last_dynamic_target_weights: List[float] = [float("nan")] * len(env_ids)
        self._last_dynamic_prev_weights: List[float] = list(self.scene_weights)
        self._last_dynamic_update_episode: int = 0

        self.action_safety_wrapper = None
        self.reward_wrapper = None
        self._base_env = None

        self._create_env(0)

        print(f"\n🌍 多场景环境初始化:")
        for i, (eid, w) in enumerate(zip(env_ids, self.scene_weights)):
            domain = self.scene_domains[i]
            print(f"   [{i}] {eid}: 权重={w:.0%}, domain={domain}")
        print(f"   最少连续集数: {self.min_episodes_per_scene} 集/场景才允许切换")
        if self.max_steps_per_scene is None:
            print(f"   最多连续步数: 禁用（仅按集数切换）")
        else:
            print(f"   最多连续步数: {self.max_steps_per_scene} 步/场景（超出后下次 reset 强制换场）")
        print(
            f"   场景切换: timeout={self.scene_reload_timeout_s:.1f}s, "
            f"post_exit_sleep={self.scene_reload_post_exit_sleep_s:.2f}s, "
            f"start_force_reload={int(self.scene_start_force_reload)}"
        )
        if self.enable_dynamic_scene_weights:
            print(
                f"   动态调权: 启用 (每{self.dynamic_weight_update_episodes}集更新, "
                f"window={self.dynamic_weight_window}, α={self.dynamic_weight_alpha}, "
                f"min_samples={self.dynamic_min_samples_per_scene}, lenβ={self.dynamic_length_beta}, "
                f"gt_prior={self.dynamic_gt_prior}, "
                f"w∈[{self.dynamic_weight_min:.2f},{self.dynamic_weight_max:.2f}], "
                f"per_scene_max={[f'{x:.2f}' for x in self.dynamic_weight_max_per_scene]})"
            )
            print(
                f"   success定义: mode={self.dynamic_success_mode}, "
                f"warmup_eps={self.dynamic_success_warmup_episodes}, "
                f"post_scale={self.dynamic_success_post_warmup_scale:.2f}, "
                f"def_mix={self.dynamic_success_deficit_mix:.2f}"
            )
        print(f"   Ablation: diff_mode={self.diff_mode}, mask_mode={self.mask_mode}, mask_scale={self.mask_scale:.2f}")

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            return [None]
        _seed_everything(int(seed))
        if self.active_env is not None:
            _safe_seed_env(self.active_env, int(seed), label="MultiSceneEnv.active_env")
        return [int(seed)]

    @staticmethod
    def _infer_scene_log_key(env_id: str) -> str:
        eid = str(env_id or "").strip().lower()
        if "waveshare" in eid:
            return "ws"
        if "generated-track" in eid:
            return "gt"
        if "warehouse" in eid:
            return "wh"
        if "minimonaco" in eid:
            return "mm"
        if "roboracingleague" in eid:
            return "rrl"
        if "circuit-launch" in eid:
            return "cl"
        if "mountain-track" in eid:
            return "mt"
        if "warren-track" in eid:
            return "wt"
        parts = eid.split("-")
        return parts[1] if len(parts) >= 2 else eid or "unknown"

    def _scene_key_for_idx(self, scene_idx: int) -> str:
        if 0 <= scene_idx < len(self.scene_log_keys):
            return str(self.scene_log_keys[scene_idx])
        if 0 <= scene_idx < len(self.env_ids):
            return self._infer_scene_log_key(self.env_ids[scene_idx])
        return "unknown"

    @staticmethod
    def _force_reload_scene(
        base_env,
        target_level_name: str,
        preflight: bool = False,
        timeout_s: float = 25.0,
        post_exit_sleep_s: float = 1.0,
    ):
        """场景切换公共逻辑——手动逆层找到 viewer、exit_scene、带超时等待加载。"""
        base = base_env
        while hasattr(base, "env"):
            base = base.env

        timeout_s = float(max(1.0, timeout_s))
        post_exit_sleep_s = float(np.clip(post_exit_sleep_s, 0.0, 3.0))

        def _wait_loaded_with_timeout(wait_timeout_s: float, poll_s: float = 0.2):
            deadline = time.time() + wait_timeout_s
            last_requery = 0.0
            last_progress_log = 0.0
            while time.time() < deadline:
                if getattr(base.viewer.handler, "loaded", False):
                    return
                now = time.time()
                if now - last_requery >= 3.0:
                    try:
                        base.viewer.handler.send_get_scene_names()
                    except Exception:
                        pass
                    last_requery = now
                if now - last_progress_log >= 5.0:
                    remain = max(0.0, deadline - now)
                    print(f"⏳ 等待场景加载中: {target_level_name} (剩余超时 {remain:.0f}s)")
                    last_progress_log = now
                time.sleep(poll_s)
            raise TimeoutError(f"场景切换超时（>{timeout_s:.0f}s）: {target_level_name}")

        label = "训练前预处理" if preflight else "切换场景"
        icon = "\u21a9\ufe0f" if preflight else "\U0001f504"
        print(f"{icon} {label}：目标场景 {target_level_name}（复用模拟器进程）")
        base.viewer.handler.SceneToLoad = target_level_name
        base.viewer.handler.loaded = False
        base.viewer.exit_scene()
        if post_exit_sleep_s > 0.0:
            time.sleep(post_exit_sleep_s)
        base.viewer.handler.send_get_scene_names()
        _wait_loaded_with_timeout(wait_timeout_s=timeout_s)
        done_label = "训练前场景重载" if preflight else "场景切换"
        print(f"\u2705 {done_label}完成: {target_level_name}")

    @staticmethod
    def _make_env_with_retry(env_id: str, conf: Dict[str, Any], retries: int = 2, retry_wait_s: float = 1.5):
        """
        Wrap gym.make with bounded retries so transient sim handshake failures
        do not immediately terminate training startup.
        """
        sim_path = str(conf.get("exe_path", "") or "").strip()
        if sim_path and sim_path not in ("remote", "none"):
            # Local simulator startup is materially slower than remote attach.
            retries = max(int(retries), 4)
            retry_wait_s = max(float(retry_wait_s), 3.0)
        retries = int(max(1, retries))
        retry_wait_s = float(max(0.0, retry_wait_s))
        last_err: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                return gym.make(env_id, conf=conf)
            except Exception as e:
                last_err = e
                if attempt >= retries:
                    break
                print(
                    f"⚠️  gym.make failed ({attempt}/{retries}) for {env_id}: "
                    f"{type(e).__name__}: {e} | retry in {retry_wait_s:.1f}s"
                )
                if retry_wait_s > 0.0:
                    time.sleep(retry_wait_s)
        assert last_err is not None
        raise last_err

    def _sample_scene_idx(self, exclude_current: bool = False, reason: str = "") -> int:
        if len(self.env_ids) == 1:
            self._last_sampling_candidates = [0]
            self._last_sampling_probs = [1.0]
            self._last_sampling_reason = "single_scene"
            self._last_sampling_choice = 0
            self._last_sampling_used_step_balance = False
            return 0

        candidates = [i for i in range(len(self.env_ids)) if (not exclude_current or i != self.active_scene_idx)]
        if len(candidates) == 1:
            self._last_sampling_candidates = list(candidates)
            self._last_sampling_probs = [1.0]
            self._last_sampling_reason = str(reason or "single_candidate")
            self._last_sampling_choice = int(candidates[0])
            self._last_sampling_used_step_balance = False
            return candidates[0]

        base = np.array([max(self.scene_weights[i], 0.0) for i in candidates], dtype=np.float64)
        if not np.all(np.isfinite(base)) or base.sum() <= 0:
            base = np.ones(len(candidates), dtype=np.float64)
        base = base / base.sum()

        used_step_balance = False
        final_probs = base.copy()
        if self.enable_step_balance_sampling:
            # 只检查主训练场景 (step_balance_mask=True) 的样本量
            main_indices = [i for i in range(len(self.env_ids)) if self.step_balance_mask[i]]
            counts_ok = all(len(self._scene_recent_lengths[i]) >= self.step_balance_min_samples_per_scene for i in main_indices) if main_indices else False
            if counts_ok:
                recent_steps_all = np.array([float(np.sum(self._scene_recent_lengths[i])) for i in range(len(self.env_ids))], dtype=np.float64)
                total_recent_steps = float(np.sum(recent_steps_all))
                if np.isfinite(total_recent_steps) and total_recent_steps > 1.0:
                    target_steps_all = total_recent_steps * np.array(self.scene_weights, dtype=np.float64)
                    deficits_all = np.maximum(0.0, target_steps_all - recent_steps_all)
                    if float(np.sum(deficits_all)) <= 1e-9:
                        deficits_all = target_steps_all / np.maximum(recent_steps_all, 1.0)
                    # 预览场景 (mask=False) 不参与步数缺口校正，其 deficit 归零
                    for i in range(len(self.env_ids)):
                        if not self.step_balance_mask[i]:
                            deficits_all[i] = 0.0
                    deficit = np.array([max(deficits_all[i], 0.0) for i in candidates], dtype=np.float64)
                    if np.all(np.isfinite(deficit)) and deficit.sum() > 0:
                        deficit = deficit / deficit.sum()
                        final_probs = (1.0 - self.step_balance_sampling_mix) * base + self.step_balance_sampling_mix * deficit
                        final_probs = np.clip(final_probs, 1e-6, None)
                        final_probs = final_probs / final_probs.sum()
                        used_step_balance = True

        choice = int(np.random.choice(candidates, p=final_probs))
        self._last_sampling_candidates = list(candidates)
        self._last_sampling_probs = [float(p) for p in final_probs]
        self._last_sampling_reason = str(reason or "weighted")
        self._last_sampling_choice = int(choice)
        self._last_sampling_used_step_balance = bool(used_step_balance)
        if used_step_balance and reason:
            prob_msg = ", ".join([f"{self.env_ids[i].split('-')[1]}={p:.2f}" for i, p in zip(candidates, final_probs)])
            print(f"🎯 场景采样[{reason}]（步数缺口校正）: {prob_msg} -> {self.env_ids[choice].split('-')[1]}")
        return choice

    def _maybe_update_scene_weights(self):
        if not self.enable_dynamic_scene_weights:
            # 即使禁用了基于成功率的动态权重，仍来应用基于步数的补偿
            if self.use_step_based_weights and self.completed_episode_count > 0 and self.completed_episode_count % max(20, self.dynamic_weight_update_episodes) == 0:
                new_w = np.array(self.scene_weights, dtype=np.float64)
                new_w = self._apply_step_based_weight_compensation(new_w)
                self.scene_weights = new_w.tolist()
            return
        if self.completed_episode_count <= 0 or self.completed_episode_count % self.dynamic_weight_update_episodes != 0:
            return

        # 冻结场景：step_balance_mask=False 的场景权重固定，不参与动态调权
        _frozen = [
            (not self.step_balance_mask[i]) if i < len(self.step_balance_mask) else False
            for i in range(len(self.env_ids))
        ]
        _has_frozen = any(_frozen)
        _frozen_weight_sum = sum(
            self.base_scene_weights[i] for i in range(len(self.env_ids)) if _frozen[i]
        ) if _has_frozen else 0.0

        reward_means = []
        len_means = []
        success_means = []
        hard_success_means = []
        enough_samples = True
        for i in range(len(self.env_ids)):
            rr = self._scene_recent_rewards[i]
            ll = self._scene_recent_lengths[i]
            ss = self._scene_recent_success[i]
            hs = self._scene_recent_hard_success[i]
            # 冻结场景不影响 enough_samples 判断
            if not _frozen[i] and (
                len(rr) < self.dynamic_min_samples_per_scene
                or len(ll) < self.dynamic_min_samples_per_scene
                or len(ss) < self.dynamic_min_samples_per_scene
            ):
                enough_samples = False
            reward_means.append(float(np.mean(rr)) if len(rr) > 0 else np.nan)
            len_means.append(float(np.mean(ll)) if len(ll) > 0 else np.nan)
            success_means.append(float(np.mean(ss)) if len(ss) > 0 else np.nan)
            hard_success_means.append(float(np.mean(hs)) if len(hs) > 0 else np.nan)
        self._last_dynamic_reward_means = [float(x) if np.isfinite(x) else float("nan") for x in reward_means]
        self._last_dynamic_len_means = [float(x) if np.isfinite(x) else float("nan") for x in len_means]
        self._last_dynamic_success_means = [float(x) if np.isfinite(x) else float("nan") for x in success_means]
        self._last_dynamic_hard_success_means = [float(x) if np.isfinite(x) else float("nan") for x in hard_success_means]
        if not enough_samples:
            return

        # success-rate deficits: success 越低，deficit 越高，采样权重越高
        # deficit ∈ [0,1], success=1 -> 0, success=0 -> 1
        succ_deficits = []
        for succ in success_means:
            if np.isnan(succ):
                succ_deficits.append(0.0)
                continue
            succ_deficits.append(float(np.clip(1.0 - succ, 0.0, 1.0)))

        # 额外引入 reward deficit，避免"成功率全 0 时权重无法拉开"。
        reward_deficits = [0.0] * len(self.env_ids)
        valid_reward_idx = [i for i, v in enumerate(reward_means) if np.isfinite(v) and not _frozen[i]]
        if len(valid_reward_idx) >= 2:
            vmin = float(min(reward_means[i] for i in valid_reward_idx))
            vmax = float(max(reward_means[i] for i in valid_reward_idx))
            denom = max(1e-6, vmax - vmin)
            for i in valid_reward_idx:
                reward_deficits[i] = float(np.clip((vmax - reward_means[i]) / denom, 0.0, 1.0))

        mix = self.dynamic_success_deficit_mix
        deficits = [
            float(np.clip(mix * succ_deficits[i] + (1.0 - mix) * reward_deficits[i], 0.0, 1.0))
            for i in range(len(self.env_ids))
        ]
        self._last_dynamic_deficits = [float(x) if np.isfinite(x) else float("nan") for x in deficits]

        valid_lens = [x for x in len_means if not np.isnan(x) and x > 1]
        if not valid_lens:
            return
        ref_len = float(np.median(valid_lens))

        raw = []
        for i in range(len(self.env_ids)):
            if _frozen[i]:
                raw.append(0.0)  # 冻结场景不参与 raw 计算
                continue
            base_w = self.base_scene_weights[i]
            # V12: dynamic_gt_prior=1.0 对所有场景相同；V9 gt 场景可设 > 1.0
            prior = self.dynamic_gt_prior
            deficit_boost = 1.0 + self.dynamic_weight_alpha * deficits[i]
            mean_len = len_means[i] if not np.isnan(len_means[i]) else ref_len
            len_factor = (ref_len / max(mean_len, 1.0)) ** self.dynamic_length_beta if self.dynamic_length_beta != 0 else 1.0
            raw.append(base_w * prior * deficit_boost * len_factor)

        raw = np.array(raw, dtype=np.float64)
        active_sum = sum(raw[i] for i in range(len(self.env_ids)) if not _frozen[i])
        if not np.all(np.isfinite(raw)) or active_sum <= 0:
            return

        # 归一化：active 场景分配 (1 - frozen_sum)，frozen 场景保持 base_weight
        available_mass = max(1e-6, 1.0 - _frozen_weight_sum)
        target = np.zeros(len(self.env_ids), dtype=np.float64)
        for i in range(len(self.env_ids)):
            if _frozen[i]:
                target[i] = self.base_scene_weights[i]
            else:
                target[i] = (raw[i] / active_sum) * available_mass
        # clip 仅对 active 场景（per-scene max）
        _wmax = self.dynamic_weight_max_per_scene
        for i in range(len(self.env_ids)):
            if not _frozen[i]:
                target[i] = float(np.clip(target[i], self.dynamic_weight_min, _wmax[i]))
        # 重新归一化 active 以确保总和精确
        active_target_sum = sum(target[i] for i in range(len(self.env_ids)) if not _frozen[i])
        if active_target_sum > 1e-6:
            for i in range(len(self.env_ids)):
                if not _frozen[i]:
                    target[i] = target[i] / active_target_sum * available_mass

        cur = np.array(self.scene_weights, dtype=np.float64)
        new_w = (1.0 - self.dynamic_weight_smoothing) * cur + self.dynamic_weight_smoothing * target
        # clip + renorm（frozen 场景恢复到 base）
        for i in range(len(self.env_ids)):
            if _frozen[i]:
                new_w[i] = self.base_scene_weights[i]
            else:
                new_w[i] = float(np.clip(new_w[i], self.dynamic_weight_min, _wmax[i]))
        new_w = new_w / new_w.sum()
        # 修复：归一化后可能突破 per-scene weight_max，将溢出部分按比例分配给未饱和场景
        _active_idx = [i for i in range(len(self.env_ids)) if not _frozen[i]]
        for _redistrib in range(5):
            _over = [(i, new_w[i] - _wmax[i]) for i in _active_idx
                     if new_w[i] > _wmax[i] + 1e-6]
            if not _over:
                break
            _excess = sum(e for _, e in _over)
            for i, _ in _over:
                new_w[i] = _wmax[i]
            _under = [i for i in _active_idx if new_w[i] < _wmax[i] - 1e-6]
            if not _under:
                break
            _under_total = sum(new_w[i] for i in _under)
            if _under_total > 1e-6:
                for i in _under:
                    new_w[i] += _excess * (new_w[i] / _under_total)
        self._last_dynamic_prev_weights = [float(x) for x in cur.tolist()]
        self._last_dynamic_target_weights = [float(x) for x in target.tolist()]
        self._last_dynamic_update_episode = int(self.completed_episode_count)

        # 方案B: 基于累积步数进行额外补偿
        if self.use_step_based_weights and len(self.env_ids) > 1:
            new_w = self._apply_step_based_weight_compensation(new_w)

        self.scene_weights = new_w.tolist()

        parts = []
        for i, eid in enumerate(self.env_ids):
            frozen_tag = " [frozen]" if _frozen[i] else ""
            parts.append(
                f"{self._scene_key_for_idx(i)}: succ={success_means[i]:.2f}, rew={reward_means[i]:.1f}, "
                f"sdef={succ_deficits[i]:.2f}, rdef={reward_deficits[i]:.2f}, "
                f"len={len_means[i]:.0f}, def={deficits[i]:.2f}, w={cur[i]:.2f}->{self.scene_weights[i]:.2f}{frozen_tag}"
            )
        print("⚖️  动态调权更新 | " + " | ".join(parts))

    def _scene_success_profile(self, scene_idx: int) -> Dict[str, float]:
        # 不同赛道几何差异较大，软成功阈值需要按场景分档。
        scene_key = self._scene_key_for_idx(scene_idx)
        profiles: Dict[str, Dict[str, float]] = {
            # progress_ratio_ref 使用几何进度累计（约等于"累计前进圈数"），
            # 与 progress_reward_scale 解耦，避免奖励权重改动后 soft-success 漂移。
            "ws": {"len_ref": 70.0, "progress_ratio_ref": 0.25, "cte_out_rate_ref": 0.22, "fail_scale": 0.25},
            "mm": {"len_ref": 95.0, "progress_ratio_ref": 0.05, "cte_out_rate_ref": 0.08, "fail_scale": 0.20},
            "gt": {"len_ref": 75.0, "progress_ratio_ref": 0.07, "cte_out_rate_ref": 0.10, "fail_scale": 0.30},
            "wh": {"len_ref": 70.0, "progress_ratio_ref": 0.06, "cte_out_rate_ref": 0.10, "fail_scale": 0.30},
            "cl": {"len_ref": 80.0, "progress_ratio_ref": 0.06, "cte_out_rate_ref": 0.09, "fail_scale": 0.25},
            "rrl": {"len_ref": 95.0, "progress_ratio_ref": 0.08, "cte_out_rate_ref": 0.09, "fail_scale": 0.25},
        }
        default_profile = {"len_ref": 75.0, "progress_ratio_ref": 0.07, "cte_out_rate_ref": 0.12, "fail_scale": 0.25}
        return profiles.get(scene_key, default_profile)

    def _extract_episode_quality_metrics(self, info: Dict[str, Any]) -> Tuple[float, float, float]:
        ep = info.get("episode", {})
        ep_len = 0.0
        if isinstance(ep, dict):
            ep_len = float(ep.get("l", 0.0) or 0.0)
        if ep_len <= 0.0:
            try:
                ep_len = float(info.get("ep_len", 0.0) or 0.0)
            except Exception:
                ep_len = 0.0

        progress_ratio_forward = float("nan")
        try:
            progress_ratio_forward = float(info.get("ep_progress_ratio_forward_sum", np.nan))
        except Exception:
            progress_ratio_forward = float("nan")
        if not np.isfinite(progress_ratio_forward):
            try:
                progress_sum = float(info.get("ep_r_progress", 0.0) or 0.0)
            except Exception:
                progress_sum = 0.0
            try:
                progress_scale = float(info.get("ep_progress_reward_scale", np.nan))
            except Exception:
                progress_scale = float("nan")
            if (not np.isfinite(progress_scale)) or progress_scale <= 1e-6:
                progress_scale = 55.0
            progress_ratio_forward = max(0.0, progress_sum) / max(1e-6, progress_scale)

        try:
            cte_over_out_rate = float(info.get("ep_cte_over_out_rate", 1.0) or 1.0)
        except Exception:
            cte_over_out_rate = 1.0

        return float(ep_len), float(progress_ratio_forward), float(cte_over_out_rate)

    def _compute_scene_soft_success(self, info: Dict[str, Any], scene_idx: int, reason_tokens: set) -> float:
        profile = self._scene_success_profile(scene_idx)
        ep_len, progress_ratio_forward, cte_over_out_rate = self._extract_episode_quality_metrics(info)

        len_score = float(np.clip(ep_len / max(1e-6, profile["len_ref"]), 0.0, 1.0))
        progress_score = float(np.clip(
            progress_ratio_forward / max(1e-6, profile["progress_ratio_ref"]),
            0.0,
            1.0,
        ))
        lane_score = float(np.clip(1.0 - cte_over_out_rate / max(1e-6, profile["cte_out_rate_ref"]), 0.0, 1.0))

        soft = 0.45 * len_score + 0.35 * progress_score + 0.20 * lane_score
        if ("collision" in reason_tokens) or ("offtrack" in reason_tokens):
            soft *= float(profile.get("fail_scale", 0.25))
        elif "stuck" in reason_tokens:
            soft *= 0.10
        return float(np.clip(soft, 0.0, 1.0))

    def _extract_episode_hard_success_flag(self, info: Dict[str, Any]) -> float:
        hard_success = 0.0
        if "ep_term_env_done" in info:
            try:
                v = float(info.get("ep_term_env_done"))
                if np.isfinite(v):
                    hard_success = float(v > 0.5)
            except Exception:
                pass

        reason = str(info.get("termination_reason", "") or "")
        tokens = {t.strip() for t in reason.split("+") if t.strip()}
        if "env_done" in tokens:
            hard_success = 1.0

        try:
            lap_raw = float(info.get("ep_r_lap_raw", 0.0) or 0.0)
            if np.isfinite(lap_raw) and lap_raw > 0.0:
                hard_success = 1.0
        except Exception:
            pass
        return float(np.clip(hard_success, 0.0, 1.0))

    def _apply_step_based_weight_compensation(self, weights: np.ndarray) -> np.ndarray:
        """
        方案B: 基于累积步数的权重补偿

        原理:
          short_scene_total_steps = 1000 步   (20步/ep × 50 eps)
          long_scene_total_steps = 3000 步    (600步/ep × 5 eps)

          补偿因子 = sqrt(total_steps_max / total_steps_i)
          short: sqrt(3000/1000) = 1.73 → 权重提升
          long:  sqrt(3000/3000) = 1.00 → 权重不变
        """
        total_steps = np.array(self._total_steps_per_scene, dtype=np.float64)

        # 只有当有足够数据时才应用补偿
        if np.all(total_steps > 100):
            max_steps = np.max(total_steps)
            # 平方根衰减：避免过度补偿
            compensation = np.sqrt(max_steps / np.maximum(total_steps, 1.0))
            # 归一化补偿因子到 [0.8, 1.2] 范围
            compensation = compensation / np.mean(compensation)
            compensation = np.clip(compensation, 0.8, 1.2)
            # 应用到权重
            weights = weights * compensation
            # 重新归一化权重
            weights = weights / np.sum(weights)

        return weights

    def _extract_episode_success_flag(self, info: Dict[str, Any], scene_idx: Optional[int] = None) -> float:
        """
        success 定义（用于动态采样）：
        - `env_done`：以 env_done / lap 作为硬成功
        - `scene_adaptive`：硬成功 + 场景自适应软成功分（用于早期探索），
          随训练推进自动衰减软成功占比，后期更接近硬成功。
        - `scene_strict`：只有满足最小长度/进度/车道约束的 env_done 才算成功。
        """
        if scene_idx is None:
            scene_idx = int(self.active_scene_idx)

        hard_success = self._extract_episode_hard_success_flag(info)
        reason = str(info.get("termination_reason", "") or "")
        tokens = {t.strip() for t in reason.split("+") if t.strip()}
        lap_raw = 0.0
        try:
            lap_raw = float(info.get("ep_r_lap_raw", 0.0) or 0.0)
        except Exception:
            pass

        if self.dynamic_success_mode == "env_done":
            return float(np.clip(hard_success, 0.0, 1.0))

        if self.dynamic_success_mode == "scene_strict":
            if ("collision" in tokens) or ("offtrack" in tokens) or ("stuck" in tokens):
                return 0.0
            if np.isfinite(lap_raw) and lap_raw > 0.0:
                return 1.0
            if hard_success <= 0.5:
                return 0.0

            profile = self._scene_success_profile(int(scene_idx))
            ep_len, progress_ratio_forward, cte_over_out_rate = self._extract_episode_quality_metrics(info)

            len_ok = ep_len >= 0.85 * max(1e-6, profile["len_ref"])
            progress_ok = progress_ratio_forward >= 0.85 * max(1e-6, profile["progress_ratio_ref"])
            lane_ok = cte_over_out_rate <= 1.10 * max(1e-6, profile["cte_out_rate_ref"])
            return float(len_ok and progress_ok and lane_ok)

        soft_success = self._compute_scene_soft_success(info, int(scene_idx), tokens)
        # 软成功占比：训练早期=1.0（更宽容），到 warmup 末期衰减到 post_warmup_scale。
        phase = float(np.clip(
            self.completed_episode_count / max(1, self.dynamic_success_warmup_episodes),
            0.0,
            1.0,
        ))
        soft_scale = (1.0 - phase) + phase * self.dynamic_success_post_warmup_scale
        mixed = max(hard_success, soft_success * soft_scale)
        return float(np.clip(mixed, 0.0, 1.0))

    def _create_env(self, scene_idx: int):
        """
        创建指定场景的完整 wrapper 链（V9.3c: 自动匹配域检测策略）
        ★ 场景切换策略（不重启模拟器进程）：
          首次：gym.make 启动模拟器，保存 base_env
          后续：exit_scene → 更新 SceneToLoad → wait_until_loaded
        """
        import gym_donkeycar  # noqa: F401

        env_id = self.env_ids[scene_idx]
        domain = _get_domain_for_env(env_id)
        level_name = {
            "donkey-waveshare-v0": "waveshare",
            "donkey-generated-track-v0": "generated_track",
            "donkey-generated-roads-v0": "generated_road",
        }.get(env_id, "waveshare")

        if self._base_env is None:
            self._base_env = MultiSceneEnv._make_env_with_retry(env_id, self.conf, retries=2, retry_wait_s=1.5)
            print(f"✅ 模拟器已启动，首个场景: {level_name}")
            try:
                MultiSceneEnv._force_reload_scene(self._base_env, level_name, preflight=True)
            except Exception as e:
                print(f"⚠️  训练前场景预退出/重载失败，将继续使用当前状态: {type(e).__name__}: {e}")
        else:
            try:
                MultiSceneEnv._force_reload_scene(self._base_env, level_name, preflight=False)
            except Exception as e:
                print(f"⚠️  场景切换失败（疑似掉线/卡死）: {type(e).__name__}: {e}")
                print("🔁 尝试重启模拟器并恢复当前目标场景...")
                try:
                    self._base_env.close()
                except Exception:
                    pass
                self._base_env = None
                self._base_env = MultiSceneEnv._make_env_with_retry(env_id, self.conf, retries=2, retry_wait_s=1.5)
                print(f"✅ 模拟器已重启，目标场景: {level_name}")
                MultiSceneEnv._force_reload_scene(self._base_env, level_name, preflight=True)

        env = self._base_env

        env = V9YellowLaneWrapper(
            env,
            target_size=self.target_size,
            enable_dr=self.enable_dr,
            dr_prob=0.6,
            domain=domain,
            detector_kwargs=self.detector_kwargs,
            diff_mode=self.diff_mode,
            mask_mode=self.mask_mode,
            mask_scale=self.mask_scale,
        )

        env = GeneralizationWrapper(env, enable_step=600000)
        env = TransposeWrapper(env)
        env = NormalizeWrapper(env)

        self.action_safety_wrapper = ActionSafetyWrapper(
            env,
            delta_max=(self.gt_delta_max if domain == "gt" else self.delta_max),
            enable_lpf=(self.gt_enable_lpf if domain == "gt" else self.enable_lpf),
            beta=(self.gt_beta if domain == "gt" else self.beta),
        )
        env = self.action_safety_wrapper

        min_throttle = self.gt_min_throttle if domain == "gt" else 0.0
        env = ThrottleControlWrapper(env, max_throttle=0.3, min_throttle=min_throttle)

        if domain == "gt" and self.gt_reset_perturb_steps_hi > 0:
            env = GTResetPerturbWrapper(
                env,
                enabled=True,
                steps_lo=self.gt_reset_perturb_steps_lo,
                steps_hi=self.gt_reset_perturb_steps_hi,
                steer_abs_max=self.gt_reset_perturb_steer,
                throttle_lo=self.gt_reset_perturb_throttle_lo,
                throttle_hi=self.gt_reset_perturb_throttle_hi,
            )

        _lap_scale = self.gt_lap_reward_scale if domain == "gt" else self.ws_lap_reward_scale
        self.reward_wrapper = DonkeyRewardWrapper(
            env,
            total_timesteps=self.total_timesteps,
            action_safety_wrapper=self.action_safety_wrapper,
            w_d=(self.gt_w_d if domain == "gt" else self.w_d),
            w_dd=(self.gt_w_dd if domain == "gt" else self.w_dd),
            w_m=(self.gt_w_m if domain == "gt" else self.w_m),
            w_sat=(self.gt_w_sat if domain == "gt" else self.w_sat),
            lap_reward_scale=_lap_scale,
        )
        env = self.reward_wrapper

        env = Monitor(env, info_keywords=MONITOR_INFO_KEYS)

        self.active_env = env
        self.active_scene_idx = scene_idx
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        return env

    def reset(self, **kwargs):
        weighted_scene_count = sum(1 for w in self.scene_weights if float(w) > 1e-9)
        step_budget_hit = (
            self.max_steps_per_scene is not None
            and weighted_scene_count > 1
            and self._cur_scene_steps >= self.max_steps_per_scene
        )

        if self.active_env is None:
            new_scene_idx = self.active_scene_idx
            self._last_sampling_candidates = [int(new_scene_idx)]
            self._last_sampling_probs = [1.0]
            self._last_sampling_reason = "init"
            self._last_sampling_choice = int(new_scene_idx)
            self._last_sampling_used_step_balance = False
        elif (not step_budget_hit) and self._cur_scene_episodes < self.min_episodes_per_scene:
            new_scene_idx = self.active_scene_idx
            self._last_sampling_candidates = [int(new_scene_idx)]
            self._last_sampling_probs = [1.0]
            self._last_sampling_reason = "hold_min_episodes"
            self._last_sampling_choice = int(new_scene_idx)
            self._last_sampling_used_step_balance = False
        else:
            if step_budget_hit and len(self.env_ids) > 1:
                new_scene_idx = self._sample_scene_idx(exclude_current=True, reason="step_budget")
                print(f"⏱️  场景步数预算命中: {self._cur_scene_steps} >= {self.max_steps_per_scene}，强制换场")
            else:
                new_scene_idx = self._sample_scene_idx(exclude_current=False, reason="weighted")

            if new_scene_idx != self.active_scene_idx:
                self._cur_scene_episodes = 0
                self._cur_scene_steps = 0

        if new_scene_idx != self.active_scene_idx or self.active_env is None:
            self._create_env(new_scene_idx)

        self._cur_scene_episodes += 1
        self.episode_count += 1
        self.scene_episode_counts[self.active_scene_idx] += 1

        if self.episode_count % 20 == 0:
            scene_stats = ", ".join([
                f"{eid.split('-')[1]}={cnt}"
                for eid, cnt in zip(self.env_ids, self.scene_episode_counts)
            ])
            cur_name = self.env_ids[self.active_scene_idx].split("-")[1]
            print(
                f"📊 场景统计 (共{self.episode_count}集): {scene_stats} | "
                f"当前场景[{cur_name}]已连续{self._cur_scene_episodes}集/{self._cur_scene_steps}步"
            )

        obs = self.active_env.reset(**kwargs)
        if self._base_env is not None:
            _clear_handler_over(self._base_env)
        return obs

    def step(self, action):
        obs, reward, done, info = self.active_env.step(action)
        self._cur_scene_steps += 1
        if done:
            self.completed_episode_count += 1
            ep = info.get("episode", {})
            if isinstance(ep, dict):
                if "r" in ep:
                    self._scene_recent_rewards[self.active_scene_idx].append(float(ep["r"]))
                if "l" in ep:
                    ep_len = float(ep["l"])
                    self._scene_recent_lengths[self.active_scene_idx].append(ep_len)
                    self._total_steps_per_scene[self.active_scene_idx] += ep_len  # 累积步数
            # 成功率双轨记录：
            # - recent_success: 按 dynamic_success_mode（用于动态采样）
            # - hard_success: 原始 env_done/lap 规则（用于日志对照与误判排查）
            self._scene_recent_success[self.active_scene_idx].append(
                float(self._extract_episode_success_flag(info, scene_idx=self.active_scene_idx))
            )
            self._scene_recent_hard_success[self.active_scene_idx].append(
                float(self._extract_episode_hard_success_flag(info))
            )
            self._maybe_update_scene_weights()
        return obs, reward, done, info

    def render(self, mode="human"):
        return self.active_env.render(mode)

    def close(self):
        if self._base_env is not None:
            try:
                self._base_env.close()
            except Exception:
                pass
            self._base_env = None


# ============================================================
# V12/V13 Dict 观测包装器（image + optional state + optional seg/latent/risk）
# ============================================================
class MultiInputObsWrapper(gym.Wrapper):
    """
    Return Dict observation:
      - rgb:            {"image", "state"}
      - rgb_seg:        {"image", "seg", "state"}
      - rgb_seg_latent: {"image", "seg_latent", "state"}
      - rgb_seg_risk:   {"image", "seg", "risk"}   # no state
    """

    VALID_VISION_MODES = ("rgb", "rgb_seg", "rgb_seg_latent", "rgb_seg_risk")
    _GLOBAL_LATENT_EMA: Optional[np.ndarray] = None
    _SCENE_LATENT_EMA: Dict[str, np.ndarray] = {}
    _SNAPSHOT_STEP_COUNTS: Dict[str, int] = {}

    def __init__(
        self,
        env,
        track_geometry,   # TrackGeometryManager — 避免循环 import，用 Any 隐式
        scene_key: str,
        logging_key: str = None,
        domain: Optional[str] = None,
        obs_size: int = 96,
        image_channels: int = 3,
        include_cte_in_obs: bool = False,
        speed_vmax: float = 2.2,
        control_wrapper: Optional[HighLevelControlWrapper] = None,
        action_safety_wrapper: Optional[ActionSafetyWrapper] = None,
        vision_mode: str = "rgb",
        seg_model_path: Optional[str] = None,
        seg_feature_dim: int = 64,
        latent_align_weight: float = 0.0,
        state_builder=None,    # callable(info, asw, cw) -> np.ndarray; overrides _build_state
        state_dim: Optional[int] = None,  # required (explicit) when state_builder is not None
        snapshot_dir: Optional[str] = None,
        snapshot_max_steps: int = 0,
        snapshot_preview_tile: int = 160,
        obstacle_context_source: str = "runtime",
        obstacle_context_checkpoint: Optional[str] = None,
        obstacle_context_device: str = "cpu",
        obstacle_context_seq_len: int = 16,
        obstacle_context_present_threshold: float = 0.5,
        obstacle_context_present_off_threshold: Optional[float] = None,
        obstacle_context_activation_consecutive: int = 3,
        obstacle_context_deactivation_consecutive: int = 2,
    ):
        super().__init__(env)
        self.track_geometry = track_geometry
        self.scene_key = scene_key
        self.logging_key = logging_key if logging_key else scene_key
        self.domain = str(domain or "").strip().lower()
        self.obs_size = int(obs_size)
        self.image_channels = int(image_channels)
        self.include_cte_in_obs = bool(include_cte_in_obs)
        self.speed_vmax = float(max(0.05, speed_vmax))
        self.control_wrapper = control_wrapper
        self.action_safety_wrapper = action_safety_wrapper

        self.vision_mode = str(vision_mode or "rgb").strip().lower()
        if self.vision_mode not in self.VALID_VISION_MODES:
            raise ValueError(f"invalid vision_mode={vision_mode}, valid={self.VALID_VISION_MODES}")
        self.include_state_in_obs = (self.vision_mode != "rgb_seg_risk")
        self.seg_model_path = str(seg_model_path or "").strip()
        self.seg_feature_dim = int(max(4, seg_feature_dim))
        self.latent_align_weight = float(np.clip(latent_align_weight, 0.0, 1.0))
        self._latent_ema_decay = 0.99
        self.snapshot_dir = str(snapshot_dir or "").strip()
        self.snapshot_max_steps = int(max(0, snapshot_max_steps))
        self.snapshot_preview_tile = int(max(64, snapshot_preview_tile))
        self.obstacle_context_source = str(obstacle_context_source or "runtime").strip().lower()
        self.obstacle_context_checkpoint = str(obstacle_context_checkpoint or "").strip()
        self.obstacle_context_device = str(obstacle_context_device or "cpu").strip()
        self.obstacle_context_seq_len = int(max(1, obstacle_context_seq_len))
        self.obstacle_context_present_threshold = float(
            np.clip(obstacle_context_present_threshold, 0.01, 0.99)
        )
        self.obstacle_context_present_off_threshold = (
            None
            if obstacle_context_present_off_threshold is None
            else float(np.clip(obstacle_context_present_off_threshold, 0.0, 0.99))
        )
        self.obstacle_context_activation_consecutive = int(
            max(1, obstacle_context_activation_consecutive)
        )
        self.obstacle_context_deactivation_consecutive = int(
            max(1, obstacle_context_deactivation_consecutive)
        )
        self._obstacle_context_estimator = None

        self._seg_model = None
        if self.vision_mode != "rgb" and self.seg_model_path:
            if torch is None:
                print("⚠️  seg_model_path 已设置但 torch 不可用，回退到启发式分割。")
            else:
                try:
                    self._seg_model = torch.jit.load(self.seg_model_path, map_location="cpu")
                    self._seg_model.eval()
                    print(f"✅ Seg model loaded: {self.seg_model_path}")
                except Exception as e:
                    print(f"⚠️  seg model load failed: {type(e).__name__}: {e}, 回退到启发式分割。")
                    self._seg_model = None

        if self.obstacle_context_source == "cv_v1":
            from .obstacle_context import CVObstacleContextEstimatorV1

            self._obstacle_context_estimator = CVObstacleContextEstimatorV1()
        elif self.obstacle_context_source == "learned_v1":
            from .obstacle_context import LearnedObstacleContextEstimatorV1

            self._obstacle_context_estimator = LearnedObstacleContextEstimatorV1(
                checkpoint_path=self.obstacle_context_checkpoint,
                device=self.obstacle_context_device,
                seq_len=self.obstacle_context_seq_len,
                present_threshold=self.obstacle_context_present_threshold,
                present_off_threshold=self.obstacle_context_present_off_threshold,
                activation_consecutive=self.obstacle_context_activation_consecutive,
                deactivation_consecutive=self.obstacle_context_deactivation_consecutive,
            )
        elif self.obstacle_context_source != "runtime":
            raise ValueError(
                f"invalid obstacle_context_source={self.obstacle_context_source}, "
                "expected one of ('runtime','cv_v1','learned_v1')"
            )

        self._last_info: Dict[str, Any] = {}
        self._prev_track_idx: Optional[int] = None
        self._state_builder = state_builder
        if state_builder is not None:
            if state_dim is None:
                raise ValueError("state_dim must be provided when state_builder is not None")
            self._state_dim = int(state_dim)
        else:
            self._state_dim = 12 + (1 if self.include_cte_in_obs else 0)

        space_dict: Dict[str, gym.spaces.Space] = {
            "image": gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(self.image_channels, self.obs_size, self.obs_size),
                dtype=np.float32,
            ),
        }
        if self.include_state_in_obs:
            space_dict["state"] = gym.spaces.Box(
                low=np.full((self._state_dim,), -3.0, dtype=np.float32),
                high=np.full((self._state_dim,), 3.0, dtype=np.float32),
                dtype=np.float32,
            )

        if self.vision_mode in ("rgb_seg", "rgb_seg_risk"):
            space_dict["seg"] = gym.spaces.Box(
                low=np.full((1, self.obs_size, self.obs_size), 0.0, dtype=np.float32),
                high=np.full((1, self.obs_size, self.obs_size), 1.0, dtype=np.float32),
                dtype=np.float32,
            )
            if self.vision_mode == "rgb_seg_risk":
                space_dict["risk"] = gym.spaces.Box(
                    low=np.full((1, self.obs_size, self.obs_size), 0.0, dtype=np.float32),
                    high=np.full((1, self.obs_size, self.obs_size), 1.0, dtype=np.float32),
                    dtype=np.float32,
                )
        elif self.vision_mode == "rgb_seg_latent":
            space_dict["seg_latent"] = gym.spaces.Box(
                low=np.full((self.seg_feature_dim,), -3.0, dtype=np.float32),
                high=np.full((self.seg_feature_dim,), 3.0, dtype=np.float32),
                dtype=np.float32,
            )
        self.observation_space = gym.spaces.Dict(space_dict)

        print(
            f"✅ MultiInputObsWrapper: mode={self.vision_mode}, "
            f"seg_model={'on' if self._seg_model is not None else 'heuristic'}"
        )
        if self.obstacle_context_source != "runtime":
            print(
                f"   obstacle_context: source={self.obstacle_context_source}, "
                f"device={self.obstacle_context_device or 'n/a'}"
            )
        if self.snapshot_dir and self.snapshot_max_steps > 0:
            print(
                f"   snapshot: save first {self.snapshot_max_steps} steps/scene -> {self.snapshot_dir}"
            )

    def _snapshot_counter_key(self) -> str:
        return f"{os.path.abspath(self.snapshot_dir)}::{self.scene_key}"

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, (str, bool)) or value is None:
            return value
        if isinstance(value, (int, float, np.integer, np.floating)):
            try:
                v = float(value)
                return int(v) if float(v).is_integer() else v
            except Exception:
                return float(value)
        if isinstance(value, (list, tuple)):
            return [MultiInputObsWrapper._json_safe(v) for v in value]
        if isinstance(value, dict):
            return {str(k): MultiInputObsWrapper._json_safe(v) for k, v in value.items()}
        return str(value)

    def _build_snapshot_preview(self, obs_dict: Dict[str, np.ndarray], meta: Dict[str, Any]) -> np.ndarray:
        image = np.asarray(obs_dict.get("image"), dtype=np.float32)
        channels = int(image.shape[0]) if image.ndim == 3 else 0
        cols = min(3, max(1, channels))
        rows = max(1, int(math.ceil(max(1, channels) / cols)))
        tile = int(self.snapshot_preview_tile)
        header_h = 108
        canvas = np.zeros((rows * tile + header_h, cols * tile, 3), dtype=np.uint8)

        labels = {
            6: ["raw_y", "edge_line", "guide_line", "edge", "vehicle", "motion"],
        }.get(channels, [f"ch{i}" for i in range(channels)])

        for idx in range(channels):
            ch = np.clip(image[idx], 0.0, 1.0)
            tile_img = cv2.resize((ch * 255.0).astype(np.uint8), (tile, tile), interpolation=cv2.INTER_NEAREST)
            tile_bgr = cv2.cvtColor(tile_img, cv2.COLOR_GRAY2BGR)
            cv2.putText(tile_bgr, labels[idx], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 220, 255), 1, cv2.LINE_AA)
            r = idx // cols
            c = idx % cols
            canvas[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = tile_bgr

        header = canvas[rows * tile:, :]
        lines = [
            f"scene={meta.get('scene_key', self.scene_key)} domain={meta.get('domain', self.domain)} snapshot={meta.get('snapshot_index', 0)}",
            f"reward={meta.get('reward', 0.0):.3f} done={int(bool(meta.get('done', False)))} speed={meta.get('speed', 0.0):.3f} cte={meta.get('cte', 0.0):.3f}",
            f"yaw_rate={meta.get('gyro_z', 0.0):.3f} prev_steer={meta.get('state_prev_steer', 0.0):.3f} prev_thr={meta.get('state_prev_throttle', 0.0):.3f}",
            f"steer_core={meta.get('ctrl/steer_core', 0.0):.3f} bias={meta.get('ctrl/bias_smooth', 0.0):.3f} hit={meta.get('hit', 'none')}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(header, line, (10, 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
        return canvas

    def _maybe_save_snapshot(self, obs_dict: Dict[str, np.ndarray], info: Dict[str, Any], reward: float, done: bool) -> None:
        if not self.snapshot_dir or self.snapshot_max_steps <= 0:
            return
        counter_key = self._snapshot_counter_key()
        snapshot_idx = int(self._SNAPSHOT_STEP_COUNTS.get(counter_key, 0))
        if snapshot_idx >= self.snapshot_max_steps:
            return

        scene_dir = os.path.join(self.snapshot_dir, self.scene_key)
        os.makedirs(scene_dir, exist_ok=True)

        npz_path = os.path.join(scene_dir, f"step_{snapshot_idx:02d}.npz")
        json_path = os.path.join(scene_dir, f"step_{snapshot_idx:02d}.json")
        png_path = os.path.join(scene_dir, f"step_{snapshot_idx:02d}_preview.png")

        payload: Dict[str, np.ndarray] = {
            "image": np.asarray(obs_dict.get("image"), dtype=np.float32),
        }
        for key in ("state", "seg", "risk", "seg_latent"):
            if key in obs_dict:
                payload[key] = np.asarray(obs_dict[key], dtype=np.float32)
        np.savez_compressed(npz_path, **payload)

        meta: Dict[str, Any] = {
            "snapshot_index": snapshot_idx,
            "scene_key": self.scene_key,
            "logging_key": self.logging_key,
            "domain": self.domain,
            "reward": float(reward),
            "done": bool(done),
            "speed": float(info.get("speed", 0.0) or 0.0),
            "cte": float(info.get("cte", 0.0) or 0.0),
            "hit": str(info.get("hit", "none")),
            "lap_count": int(info.get("lap_count", 0) or 0),
            "termination_reason": str(info.get("termination_reason", "")),
            "gyro_z": float((info.get("gyro", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0))[2]),
        }
        if "state" in obs_dict:
            state = np.asarray(obs_dict["state"], dtype=np.float32).reshape(-1)
            if state.size >= 5:
                meta["state_prev_steer"] = float(state[3])
                meta["state_prev_throttle"] = float(state[4])
        keep_scalar_keys = {"scene_key", "logging_key", "domain", "mask_coverage"}
        keep_tuple_keys = {"pos", "car", "gyro", "accel", "vel"}
        keep_prefixes = ("ctrl/", "reward_debug/", "geo/", "smooth/", "obstacle_")
        for key, value in info.items():
            if key in keep_scalar_keys or key in keep_tuple_keys or any(key.startswith(prefix) for prefix in keep_prefixes):
                meta[key] = self._json_safe(value)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        preview = self._build_snapshot_preview(obs_dict, meta)
        cv2.imwrite(png_path, preview)

        self._SNAPSHOT_STEP_COUNTS[counter_key] = snapshot_idx + 1

    def _extract_prev_controls(self) -> Tuple[float, float]:
        steer_exec = 0.0
        throttle_exec = 0.0

        if self.action_safety_wrapper is not None:
            try:
                steer_exec = float(self.action_safety_wrapper.diag.get("steer_exec", 0.0))
            except Exception:
                steer_exec = 0.0

        if self.control_wrapper is not None:
            try:
                throttle_exec = float(self.control_wrapper.last_low_level_action[1])
                if self.action_safety_wrapper is None:
                    steer_exec = float(self.control_wrapper.last_low_level_action[0])
            except Exception:
                pass

        return _clip_float(steer_exec, -1.0, 1.0), _clip_float(throttle_exec, -1.0, 1.0)

    def _build_state(self, info: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, float]]:
        speed = float(info.get("speed", 0.0) or 0.0)

        gyro  = info.get("gyro",  (0.0, 0.0, 0.0))
        accel = info.get("accel", (0.0, 0.0, 0.0))
        car   = info.get("car",   (0.0, 0.0, 0.0))
        pos   = info.get("pos",   (0.0, 0.0, 0.0))

        try:
            gyro_z = float(gyro[2])
        except Exception:
            gyro_z = 0.0
        try:
            accel_y = float(accel[1])
        except Exception:
            accel_y = 0.0
        try:
            roll_deg, pitch_deg, yaw_deg = float(car[0]), float(car[1]), float(car[2])
        except Exception:
            roll_deg = pitch_deg = yaw_deg = 0.0
        try:
            x, z = float(pos[0]), float(pos[2])
        except Exception:
            x = z = 0.0

        yaw_rad = math.radians(yaw_deg)
        geo = self.track_geometry.query(
            self.scene_key,
            x=x,
            z=z,
            yaw_rad=yaw_rad,
            prev_idx=self._prev_track_idx,
        )
        self._prev_track_idx = int(geo["idx"])

        prev_steer_exec, prev_throttle_exec = self._extract_prev_controls()

        speed_norm = _clip_float(speed / self.speed_vmax, 0.0, 2.0)
        gyro_z_norm = _clip_float(gyro_z / 4.0, -2.0, 2.0)
        accel_y_norm = _clip_float(accel_y / 9.8, -2.5, 2.5)

        state_list: List[float] = [
            speed_norm,
            gyro_z_norm,
            accel_y_norm,
            math.sin(math.radians(roll_deg)),
            math.sin(math.radians(pitch_deg)),
            prev_steer_exec,
            prev_throttle_exec,
            float(geo["lat_err_norm"]),
            float(geo["heading_err_sin"]),
            float(geo["heading_err_cos"]),
            float(geo["kappa_lookahead"]),
            float(geo["width_norm"]),
        ]

        if self.include_cte_in_obs:
            cte = float(info.get("cte", 0.0) or 0.0)
            state_list.append(_clip_float(cte / 5.0, -3.0, 3.0))

        state_arr = np.asarray(state_list, dtype=np.float32)

        geo_log = {
            "geo/lat_err_norm": float(geo["lat_err_norm"]),
            "geo/heading_err": float(math.atan2(geo["heading_err_sin"], geo["heading_err_cos"])),
            "geo/kappa": float(geo["kappa_lookahead"]),
            "geo/width_norm": float(geo["width_norm"]),
        }
        return state_arr, geo_log

    def _build_seg_map(self, img_chw: np.ndarray) -> np.ndarray:
        """从当前 RGB 观测构建单通道伪分割概率图 [H,W] in [0,1]。"""
        img = np.asarray(img_chw, dtype=np.float32)
        if img.shape != (3, self.obs_size, self.obs_size):
            raise ValueError(f"image obs shape mismatch: got {img.shape}, expected (3,{self.obs_size},{self.obs_size})")
        rgb_u8 = np.uint8(np.clip(img.transpose(1, 2, 0) * 255.0, 0, 255))

        if self._seg_model is not None and torch is not None:
            try:
                with torch.no_grad():
                    x = torch.from_numpy(img[None, ...]).float()
                    y = self._seg_model(x)
                    if isinstance(y, (tuple, list)):
                        y = y[0]
                    if y.ndim == 4:
                        y = y[0]
                    if y.shape[0] > 1:
                        y = torch.softmax(y, dim=0)[1]
                    else:
                        y = torch.sigmoid(y[0])
                    seg = y.detach().cpu().numpy().astype(np.float32)
                    if seg.shape != (self.obs_size, self.obs_size):
                        seg = cv2.resize(seg, (self.obs_size, self.obs_size), interpolation=cv2.INTER_LINEAR)
                    return np.clip(seg, 0.0, 1.0)
            except Exception:
                pass

        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
        yellow = cv2.inRange(hsv, np.array([12, 70, 70], dtype=np.uint8), np.array([45, 255, 255], dtype=np.uint8))
        white = cv2.inRange(hsv, np.array([0, 0, 170], dtype=np.uint8), np.array([180, 60, 255], dtype=np.uint8))
        lane = np.maximum(yellow, white).astype(np.float32) / 255.0

        gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 60, 180).astype(np.float32) / 255.0
        road = np.clip((gray.astype(np.float32) - 35.0) / 220.0, 0.0, 1.0)
        seg = 0.55 * lane + 0.30 * road + 0.15 * edges
        seg = cv2.GaussianBlur(np.clip(seg, 0.0, 1.0), (5, 5), 0)
        return np.clip(seg, 0.0, 1.0).astype(np.float32)

    def _build_risk_map(self, seg_map: np.ndarray) -> np.ndarray:
        """
        从分割概率图生成单通道风险图 [H,W] in [0,1]：
          - 赛道外/低置信区域风险高
          - 距离赛道边界越近，风险越高
        """
        seg = np.clip(np.asarray(seg_map, dtype=np.float32), 0.0, 1.0)
        safe_mask = (seg >= 0.45).astype(np.uint8)
        safe_ratio = float(np.mean(safe_mask))
        if safe_ratio < 0.01:
            return np.clip(1.0 - seg, 0.0, 1.0).astype(np.float32)

        safe_u8 = (safe_mask * 255).astype(np.uint8)
        dist = cv2.distanceTransform(safe_u8, cv2.DIST_L2, 3)
        norm_scale = max(2.0, float(self.obs_size) * 0.12)
        inside_risk = 1.0 - np.clip(dist / norm_scale, 0.0, 1.0)
        outside_risk = 1.0 - safe_mask.astype(np.float32)
        risk = np.maximum(inside_risk.astype(np.float32), outside_risk)
        risk = cv2.GaussianBlur(np.clip(risk, 0.0, 1.0), (5, 5), 0)
        return np.clip(risk, 0.0, 1.0).astype(np.float32)

    def _seg_to_latent(self, seg_map: np.ndarray) -> np.ndarray:
        dim = int(self.seg_feature_dim)
        side = int(round(math.sqrt(dim)))
        if side * side == dim:
            pooled = cv2.resize(seg_map, (side, side), interpolation=cv2.INTER_AREA).reshape(-1)
        else:
            pooled = cv2.resize(seg_map, (dim, 1), interpolation=cv2.INTER_AREA).reshape(-1)
        latent = np.asarray(pooled, dtype=np.float32)
        latent = latent * 2.0 - 1.0
        return np.clip(latent, -3.0, 3.0)

    def _align_latent(self, latent_raw: np.ndarray) -> np.ndarray:
        if self.latent_align_weight <= 1e-6:
            return latent_raw

        scene_ema = MultiInputObsWrapper._SCENE_LATENT_EMA.get(self.scene_key)
        if scene_ema is None or scene_ema.shape != latent_raw.shape:
            scene_ema = latent_raw.copy()
        else:
            scene_ema = self._latent_ema_decay * scene_ema + (1.0 - self._latent_ema_decay) * latent_raw
        MultiInputObsWrapper._SCENE_LATENT_EMA[self.scene_key] = scene_ema

        if (
            MultiInputObsWrapper._GLOBAL_LATENT_EMA is None
            or MultiInputObsWrapper._GLOBAL_LATENT_EMA.shape != latent_raw.shape
        ):
            MultiInputObsWrapper._GLOBAL_LATENT_EMA = latent_raw.copy()
        else:
            MultiInputObsWrapper._GLOBAL_LATENT_EMA = (
                self._latent_ema_decay * MultiInputObsWrapper._GLOBAL_LATENT_EMA
                + (1.0 - self._latent_ema_decay) * latent_raw
            )

        offset = scene_ema - MultiInputObsWrapper._GLOBAL_LATENT_EMA
        aligned = latent_raw - self.latent_align_weight * offset
        return np.clip(aligned, -3.0, 3.0).astype(np.float32)

    def _apply_obstacle_context(self, img: np.ndarray, info: Dict[str, Any]) -> None:
        info["obstacle_context_source"] = self.obstacle_context_source
        if self.obstacle_context_source == "runtime" or self._obstacle_context_estimator is None:
            return

        info["obstacle_present_runtime"] = float(info.get("obstacle_present", 0.0) or 0.0)
        info["obstacle_longitudinal_runtime"] = float(info.get("obstacle_longitudinal", 0.0) or 0.0)
        info["obstacle_lateral_runtime"] = float(info.get("obstacle_lateral", 0.0) or 0.0)
        info["obstacle_dist_runtime"] = float(info.get("obstacle_dist", 0.0) or 0.0)
        info["obstacle_risk_runtime"] = float(info.get("obstacle_risk", 0.0) or 0.0)

        state7 = _build_state_v13(
            info,
            self.action_safety_wrapper,
            self.control_wrapper,
            v_max=self.speed_vmax,
        )
        est = self._obstacle_context_estimator.estimate(img, info=info, state7=state7)
        info.update(est.to_info_dict())
        debug_prefix = "obstacle_cv" if self.obstacle_context_source == "cv_v1" else "obstacle_learned"
        info.update(est.to_prefixed_debug_info(debug_prefix))

    def _obs_dict(self, img_obs: np.ndarray, info: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        img = np.asarray(img_obs, dtype=np.float32)
        if img.shape != (self.image_channels, self.obs_size, self.obs_size):
            raise ValueError(f"image obs shape mismatch: got {img.shape}, expected ({self.image_channels},{self.obs_size},{self.obs_size})")
        self._apply_obstacle_context(img, info)

        if self._state_builder is not None:
            state = self._state_builder(info, self.action_safety_wrapper, self.control_wrapper)
            geo_log: Dict[str, float] = {}
        else:
            state, geo_log = self._build_state(info)

        obs_dict: Dict[str, np.ndarray] = {"image": img}
        if self.include_state_in_obs:
            obs_dict["state"] = state
        if self.vision_mode != "rgb":
            seg = self._build_seg_map(img)
            coverage = float(np.mean(seg))
            geo_log["seg/coverage"] = coverage
            if self.vision_mode in ("rgb_seg", "rgb_seg_risk"):
                obs_dict["seg"] = seg[np.newaxis, ...].astype(np.float32)
                if self.vision_mode == "rgb_seg_risk":
                    risk = self._build_risk_map(seg)
                    obs_dict["risk"] = risk[np.newaxis, ...].astype(np.float32)
                    geo_log["risk/mean"] = float(np.mean(risk))
            else:
                latent_raw = self._seg_to_latent(seg)
                obs_dict["seg_latent"] = self._align_latent(latent_raw)
        return obs_dict, geo_log

    def reset(self, **kwargs):
        img = self.env.reset(**kwargs)
        self._last_info = {
            "speed": 0.0, "gyro": (0.0, 0.0, 0.0), "accel": (0.0, 0.0, 0.0),
            "car": (0.0, 0.0, 0.0), "pos": (0.0, 0.0, 0.0), "cte": 0.0,
        }
        self._prev_track_idx = None
        if self._obstacle_context_estimator is not None:
            self._obstacle_context_estimator.reset()
        obs, _ = self._obs_dict(img, self._last_info)
        return obs

    def step(self, action):
        img, reward, done, info = self.env.step(action)

        if self.control_wrapper is not None:
            self.control_wrapper.consume_info(info)

        obs, geo_log = self._obs_dict(img, info)

        if self.action_safety_wrapper is not None:
            merged_info = dict(info)
            merged_info.update(geo_log)
            self.action_safety_wrapper.consume_info(merged_info)

        if self.control_wrapper is not None:
            d = self.control_wrapper.diag
            info["ctrl/v_target"]    = float(d.get("v_target", 0.0))
            info["ctrl/v_meas"]      = float(d.get("v_meas", 0.0))
            info["ctrl/v_err"]       = float(d.get("v_err", 0.0))
            info["ctrl/throttle_pi"] = float(d.get("throttle_pi", 0.0))
            # V13 ActionAdapter diagnostics (ignored if keys absent → V12 path)
            if "steer_core" in d:
                info["ctrl/steer_core"]   = float(d["steer_core"])
                info["ctrl/bias_smooth"]  = float(d.get("bias_smooth", 0.0))
                info["ctrl/bias_offset"]  = float(d.get("bias_offset", 0.0))
                info["ctrl/v_base"]       = float(d.get("v_base", 0.0))
                info["ctrl/delta_steer_input"] = float(d.get("delta_steer_input", 0.0))
                info["ctrl/speed_scale_input"] = float(d.get("speed_scale_input", 0.0))
                info["ctrl/line_bias_input"]   = float(d.get("line_bias_input", 0.0))

        for k, v in geo_log.items():
            info[k] = float(v)

        # 场景标识：Monitor info_keywords 捕获后进入 ep_info_buffer
        info["scene_key"] = self.scene_key
        info["logging_key"] = self.logging_key
        info["domain"] = self.domain if self.domain else "unknown"
        if "seg/coverage" in geo_log:
            info["mask_coverage"] = float(geo_log["seg/coverage"])

        # V12 填充 MONITOR_INFO_KEYS 的默认值（V9 设置但 V12 不需要的字段）
        info.setdefault("mask_coverage", 0.0)
        info.setdefault("ep_r_survival", 0.0)
        info.setdefault("ep_r_speed", 0.0)
        info.setdefault("ep_r_progress", 0.0)
        info.setdefault("ep_r_cte", 0.0)
        info.setdefault("ep_r_center", 0.0)
        info.setdefault("ep_r_heading", 0.0)
        info.setdefault("ep_r_speed_ref", 0.0)
        info.setdefault("ep_r_time", 0.0)
        info.setdefault("ep_r_collision", 0.0)
        info.setdefault("ep_r_near_offtrack", 0.0)
        info.setdefault("ep_r_near_collision", 0.0)
        info.setdefault("ep_r_lap", 0.0)
        info.setdefault("ep_r_lap_raw", 0.0)
        info.setdefault("ep_soft_lap_count", 0.0)
        info.setdefault("ep_r_smooth", 0.0)
        info.setdefault("ep_r_jerk", 0.0)
        info.setdefault("ep_r_mismatch", 0.0)
        info.setdefault("ep_r_micro_wiggle", 0.0)
        info.setdefault("ep_r_sat", 0.0)
        info.setdefault("ep_r_total", 0.0)
        info.setdefault("ep_term_collision", 0.0)
        info.setdefault("ep_term_stuck", 0.0)
        info.setdefault("ep_term_offtrack", 0.0)
        info.setdefault("ep_term_env_done", 0.0)
        info.setdefault("ep_term_normal", 0.0)
        info.setdefault("ep_lane_pid_debug_steps", 0.0)
        info.setdefault("ep_lane_pid_target_speed_mean", 0.0)
        info.setdefault("ep_lane_pid_speed_mean", 0.0)
        info.setdefault("ep_lane_pid_speed_error_abs_mean", 0.0)
        info.setdefault("ep_lane_pid_effective_lookahead_mean", 0.0)
        info.setdefault("ep_lane_pid_local_forward_mean", 0.0)
        info.setdefault("ep_lane_pid_local_left_abs_mean", 0.0)
        info.setdefault("ep_lane_pid_lat_err_norm_abs_mean", 0.0)
        info.setdefault("ep_lane_pid_steer_abs_mean", 0.0)
        info.setdefault("ep_lane_pid_throttle_mean", 0.0)
        info.setdefault("ep_lane_pid_reverse_rate", 0.0)
        info.setdefault("ep_cte_abs_p50", 0.0)
        info.setdefault("ep_cte_abs_p90", 0.0)
        info.setdefault("ep_cte_abs_p99", 0.0)
        info.setdefault("ep_cte_over_in_rate", 0.0)
        info.setdefault("ep_cte_over_out_rate", 0.0)
        info.setdefault("ep_rate_limit_hit_rate", 0.0)
        info.setdefault("ep_steer_clip_hit_rate", 0.0)
        info.setdefault("ep_throttle_high_penalty_hit_rate", 0.0)
        info.setdefault("ep_offtrack_counter_max", 0.0)
        info.setdefault("ep_stuck_counter_max", 0.0)

        self._maybe_save_snapshot(obs, info, reward, done)
        self._last_info = info
        return obs, reward, done, info


# ============================================================
# V12 Wrapper 链公共工厂（MultiSceneEnvV12._create_env 和 create_v12_env 共用）
# ============================================================
def _build_v12_wrapper_chain(
    base_env,
    scene_key: str,
    logging_key: str,
    domain: Optional[str],
    track_geometry,
    obs_size: int = 96,
    augment: bool = True,
    augment_start_step: int = 600000,
    delta_max: float = 0.35,
    enable_lpf: bool = True,
    beta: float = 0.6,
    max_throttle: float = 0.3,
    w_d: float = 0.25,
    w_dd: float = 0.08,
    w_m: float = 0.08,
    w_sat: float = 0.15,
    w_time: float = 0.0,
    w_center: float = 0.0,
    w_heading: float = 0.0,
    w_speed_ref: float = 0.0,
    speed_ref_vmin: float = 0.35,
    speed_ref_vmax: float = 2.2,
    speed_ref_kappa_ref: float = 0.15,
    lap_reward_scale: float = 1.0,
    progress_reward_scale: float = 30.0,
    survival_reward_scale: float = 0.30,
    collision_penalty_base: float = 12.0,
    offtrack_penalty_base: float = 8.0,
    adaptive_delta_max: bool = True,
    curve_delta_boost: float = 1.0,
    curve_kappa_ref: float = 0.15,
    steer_intent_boost: float = 0.30,
    hairpin_curve_ratio: float = 0.85,
    hairpin_min_delta_max: float = 0.45,
    hairpin_max_delta_max: float = 0.85,
    w_near_offtrack: float = 0.40,
    near_offtrack_start_ratio: float = 0.70,
    w_near_collision: float = 0.35,
    near_collision_start_ratio: float = 0.65,
    total_timesteps: int = 500000,
    speed_vmax: float = 2.2,
    speed_kp: float = 0.35,
    speed_ki: float = 0.08,
    speed_kff: float = 0.10,
    allow_reverse: bool = False,
    control_dt: float = 0.05,
    include_cte_in_obs: bool = False,
    offtrack_leniency_ratio: float = 0.15,
    offtrack_leniency_mult: float = 1.8,
    vision_mode: str = "rgb",
    seg_model_path: Optional[str] = None,
    seg_feature_dim: int = 64,
    latent_align_weight: float = 0.0,
    reset_env_done_grace_steps: int = 0,
    reset_collision_grace_steps: int = 0,
):
    """
    构建 V12 wrapper 链并返回 (env, action_safety, high_level, reward_wrapper)。
    base_env 应是已完成场景加载的原始 DonkeyEnv。
    """
    cte_geometry = track_geometry.scenes[scene_key]

    env = RGBResizeWrapper(
        base_env,
        obs_size=obs_size,
        augment=augment,
        max_steps=total_timesteps,
        augment_start_step=augment_start_step,
    )

    action_safety = ActionSafetyWrapper(
        env,
        delta_max=delta_max,
        enable_lpf=enable_lpf,
        beta=beta,
        adaptive_delta_max=adaptive_delta_max,
        curve_delta_boost=curve_delta_boost,
        curve_kappa_ref=curve_kappa_ref,
        steer_intent_boost=steer_intent_boost,
        hairpin_curve_ratio=hairpin_curve_ratio,
        hairpin_min_delta_max=hairpin_min_delta_max,
        hairpin_max_delta_max=hairpin_max_delta_max,
    )
    env = action_safety

    # 简化控制链：不额外叠加"曲率油门收紧"和"二次油门裁切"。
    # 低层油门约束由 HighLevelControlWrapper 内部限幅负责。

    reward_wrapper = ImprovedRewardWrapperV3(
        env,
        total_timesteps=total_timesteps,
        action_safety_wrapper=action_safety,
        w_d=w_d, w_dd=w_dd, w_m=w_m, w_sat=w_sat,
        w_time=w_time, w_center=w_center, w_heading=w_heading, w_speed_ref=w_speed_ref,
        speed_ref_vmin=speed_ref_vmin, speed_ref_vmax=speed_ref_vmax,
        speed_ref_kappa_ref=speed_ref_kappa_ref,
        lap_reward_scale=lap_reward_scale,
        progress_reward_scale=progress_reward_scale,
        survival_reward_scale=survival_reward_scale,
        collision_penalty_base=collision_penalty_base,
        offtrack_penalty_base=offtrack_penalty_base,
        w_near_offtrack=w_near_offtrack,
        near_offtrack_start_ratio=near_offtrack_start_ratio,
        w_near_collision=w_near_collision,
        near_collision_start_ratio=near_collision_start_ratio,
        cte_left=float(cte_geometry.cte_left),
        cte_right=float(cte_geometry.cte_right),           # 负值，reward 直接使用有符号值
        cte_left_out=float(cte_geometry.cte_left_out),
        cte_right_out=float(cte_geometry.cte_right_out),   # 负值
        coord_scale=float(cte_geometry.coord_scale),
        offtrack_leniency_ratio=offtrack_leniency_ratio,
        offtrack_leniency_mult=offtrack_leniency_mult,
        track_geometry=track_geometry,
        scene_key=scene_key,
        logging_key=logging_key,
        cte_half_width=float(cte_geometry.cte_half_width),
        reset_env_done_grace_steps=reset_env_done_grace_steps,
        reset_collision_grace_steps=reset_collision_grace_steps,
    )
    env = reward_wrapper

    high_level = HighLevelControlWrapper(
        env,
        speed_vmax=speed_vmax, speed_kp=speed_kp, speed_ki=speed_ki, speed_kff=speed_kff,
        control_dt=control_dt, max_throttle=max_throttle, allow_reverse=allow_reverse,
    )
    env = high_level

    env = MultiInputObsWrapper(
        env,
        track_geometry=track_geometry,
        scene_key=scene_key,
        logging_key=logging_key,
        domain=domain,
        obs_size=obs_size,
        include_cte_in_obs=include_cte_in_obs,
        speed_vmax=speed_vmax,
        control_wrapper=high_level,
        action_safety_wrapper=action_safety,
        vision_mode=vision_mode,
        seg_model_path=seg_model_path,
        seg_feature_dim=seg_feature_dim,
        latent_align_weight=latent_align_weight,
    )

    env = Monitor(env, info_keywords=MONITOR_INFO_KEYS)
    return env, action_safety, high_level, reward_wrapper


# ============================================================
# V12 多场景环境（继承 MultiSceneEnv，_create_env 使用 V12 全链路）
# ============================================================
class MultiSceneEnvV12(MultiSceneEnv):
    """
    V12 多场景训练环境。
    复用 V9 的多场景切换 / 动态调权逻辑，_create_env 使用简化的 V12 链路：
      - RGBResizeWrapper：纯 RGB (3,H,W)，无车道检测
      - ActionSafetyWrapper：所有场景统一参数，无 domain 区分
      - DonkeyRewardWrapper：统一奖励权重，scene-specific CTE 边界
      - HighLevelControlWrapper + MultiInputObsWrapper
    """

    # SCENE_SPECS 由 ppo_waveshare_v12.py 初始化后注入（避免循环导入）
    _SCENE_SPECS: Dict[str, Dict[str, str]] = {}

    def __init__(
        self,
        env_ids: List[str],
        conf: Dict[str, Any],
        scene_weights: List[float],
        track_geometry,           # TrackGeometryManager
        scene_specs: Dict[str, Dict[str, str]],
        obs_size: int = 96,
        include_cte_in_obs: bool = False,
        speed_vmax: float = 2.2,
        speed_kp: float = 0.35,
        speed_ki: float = 0.08,
        speed_kff: float = 0.10,
        allow_reverse: bool = False,
        max_throttle: float = 0.3,
        control_dt: float = 0.05,
        augment: bool = True,
        augment_start_step: int = 600000,
        # 统一参数（不区分 domain）
        total_timesteps: int = 500000,
        delta_max: float = 0.35,
        enable_lpf: bool = True,
        beta: float = 0.6,
        w_d: float = 0.25,
        w_dd: float = 0.08,
        w_m: float = 0.08,
        w_sat: float = 0.15,
        w_time: float = 0.0,
        w_center: float = 0.0,
        w_heading: float = 0.0,
        w_speed_ref: float = 0.0,
        speed_ref_vmin: float = 0.35,
        speed_ref_vmax: float = 2.2,
        speed_ref_kappa_ref: float = 0.15,
        offtrack_leniency_ratio: float = 0.15,
        offtrack_leniency_mult: float = 1.8,
        lap_reward_scale: float = 1.0,
        progress_reward_scale: float = 30.0,
        survival_reward_scale: float = 0.30,
        collision_penalty_base: float = 12.0,
        offtrack_penalty_base: float = 8.0,
        vision_mode: str = "rgb",
        seg_model_path: Optional[str] = None,
        seg_feature_dim: int = 64,
        latent_align_weight: float = 0.0,
        snapshot_dir: Optional[str] = None,
        snapshot_max_steps: int = 0,
        adaptive_delta_max: bool = True,
        curve_delta_boost: float = 1.0,
        curve_kappa_ref: float = 0.15,
        steer_intent_boost: float = 0.30,
        hairpin_curve_ratio: float = 0.85,
        hairpin_min_delta_max: float = 0.45,
        hairpin_max_delta_max: float = 0.85,
        w_near_offtrack: float = 0.40,
        near_offtrack_start_ratio: float = 0.70,
        w_near_collision: float = 0.35,
        near_collision_start_ratio: float = 0.65,
        overtake_success_bonus: float = 2.5,
        reset_env_done_grace_steps: int = 0,
        reset_collision_grace_steps: int = 0,
        min_episodes_per_scene: int = 8,
        max_steps_per_scene: Optional[int] = 1024,
        enable_dynamic_scene_weights: bool = True,
        dynamic_weight_update_episodes: int = 36,
        dynamic_weight_window: int = 60,
        dynamic_min_samples_per_scene: int = 8,
        dynamic_weight_alpha: float = 1.5,
        dynamic_length_beta: float = 1.0,
        dynamic_weight_smoothing: float = 0.35,
        dynamic_weight_min: float = 0.02,
        dynamic_weight_max: float = 0.98,
        dynamic_success_mode: str = "scene_adaptive",
        dynamic_success_warmup_episodes: int = 1200,
        dynamic_success_post_warmup_scale: float = 0.20,
        dynamic_success_deficit_mix: float = 0.85,
        enable_step_balance_sampling: bool = True,
        step_balance_sampling_mix: float = 0.85,
        step_balance_mask: Optional[List[bool]] = None,
        scene_reload_timeout_s: float = 25.0,
        scene_reload_post_exit_sleep_s: float = 1.0,
        scene_start_force_reload: bool = True,
    ):
        # 注入 scene_specs，子类 _create_env 会用到
        MultiSceneEnvV12._SCENE_SPECS = scene_specs

        self.track_geometry = track_geometry
        self.obs_size = int(obs_size)
        self.include_cte_in_obs = bool(include_cte_in_obs)
        self.speed_vmax = float(speed_vmax)
        self.speed_kp = float(speed_kp)
        self.speed_ki = float(speed_ki)
        self.speed_kff = float(speed_kff)
        self.allow_reverse = bool(allow_reverse)
        self.max_throttle = float(max_throttle)
        self.control_dt = float(control_dt)
        self.augment = bool(augment)
        self.augment_start_step = int(augment_start_step)
        self.offtrack_leniency_ratio = float(offtrack_leniency_ratio)
        self.offtrack_leniency_mult  = float(offtrack_leniency_mult)
        self.lap_reward_scale = float(lap_reward_scale)
        self.progress_reward_scale = float(progress_reward_scale)
        self.survival_reward_scale = float(survival_reward_scale)
        self.collision_penalty_base = float(collision_penalty_base)
        self.offtrack_penalty_base = float(offtrack_penalty_base)
        self.w_time = float(w_time)
        self.w_center = float(w_center)
        self.w_heading = float(w_heading)
        self.w_speed_ref = float(w_speed_ref)
        self.speed_ref_vmin = float(speed_ref_vmin)
        self.speed_ref_vmax = float(speed_ref_vmax)
        self.speed_ref_kappa_ref = float(speed_ref_kappa_ref)
        self.vision_mode = str(vision_mode or "rgb")
        self.seg_model_path = str(seg_model_path or "")
        self.seg_feature_dim = int(seg_feature_dim)
        self.latent_align_weight = float(latent_align_weight)
        self.snapshot_dir = str(snapshot_dir or "")
        self.snapshot_max_steps = int(max(0, snapshot_max_steps))
        self.adaptive_delta_max = bool(adaptive_delta_max)
        self.curve_delta_boost = float(curve_delta_boost)
        self.curve_kappa_ref = float(curve_kappa_ref)
        self.steer_intent_boost = float(steer_intent_boost)
        self.hairpin_curve_ratio = float(hairpin_curve_ratio)
        self.hairpin_min_delta_max = float(hairpin_min_delta_max)
        self.hairpin_max_delta_max = float(hairpin_max_delta_max)
        self.w_near_offtrack = float(w_near_offtrack)
        self.near_offtrack_start_ratio = float(near_offtrack_start_ratio)
        self.w_near_collision = float(w_near_collision)
        self.near_collision_start_ratio = float(near_collision_start_ratio)
        self.overtake_success_bonus = float(max(0.0, overtake_success_bonus))
        self.reset_env_done_grace_steps = max(0, int(reset_env_done_grace_steps))
        self.reset_collision_grace_steps = max(0, int(reset_collision_grace_steps))
        scene_log_keys = [scene_specs[eid].get("logging_key", MultiSceneEnv._infer_scene_log_key(eid)) for eid in env_ids]

        super().__init__(
            env_ids=env_ids,
            conf=conf,
            scene_weights=scene_weights,
            scene_log_keys=scene_log_keys,
            target_size=(self.obs_size, self.obs_size),
            enable_dr=False,        # V12 不使用 V9YellowLaneWrapper，enable_dr 传 False
            total_timesteps=total_timesteps,
            delta_max=delta_max,
            enable_lpf=enable_lpf,
            beta=beta,
            # gt_* 参数全部与 ws 相同，消除 domain 区分
            gt_delta_max=delta_max,
            gt_enable_lpf=enable_lpf,
            gt_beta=beta,
            w_d=w_d,
            w_dd=w_dd,
            w_m=w_m,
            w_sat=w_sat,
            gt_w_d=w_d,
            gt_w_dd=w_dd,
            gt_w_m=w_m,
            gt_w_sat=w_sat,
            ws_lap_reward_scale=lap_reward_scale,
            gt_lap_reward_scale=lap_reward_scale,
            gt_min_throttle=0.0,
            gt_reset_perturb_steps_lo=0,
            gt_reset_perturb_steps_hi=0,
            detector_kwargs={},
            min_episodes_per_scene=min_episodes_per_scene,
            max_steps_per_scene=max_steps_per_scene,
            enable_dynamic_scene_weights=enable_dynamic_scene_weights,
            dynamic_weight_update_episodes=dynamic_weight_update_episodes,
            dynamic_weight_window=dynamic_weight_window,
            dynamic_min_samples_per_scene=dynamic_min_samples_per_scene,
            dynamic_weight_alpha=dynamic_weight_alpha,
            dynamic_length_beta=dynamic_length_beta,
            dynamic_gt_prior=1.0,   # 所有场景权重先验相同，不区分 domain
            dynamic_weight_smoothing=dynamic_weight_smoothing,
            dynamic_weight_min=dynamic_weight_min,
            dynamic_weight_max=dynamic_weight_max,
            dynamic_success_mode=dynamic_success_mode,
            dynamic_success_warmup_episodes=dynamic_success_warmup_episodes,
            dynamic_success_post_warmup_scale=dynamic_success_post_warmup_scale,
            dynamic_success_deficit_mix=dynamic_success_deficit_mix,
            enable_step_balance_sampling=enable_step_balance_sampling,
            step_balance_sampling_mix=step_balance_sampling_mix,
            step_balance_mask=step_balance_mask,
            scene_reload_timeout_s=scene_reload_timeout_s,
            scene_reload_post_exit_sleep_s=scene_reload_post_exit_sleep_s,
            scene_start_force_reload=scene_start_force_reload,
        )

    def _create_env(self, scene_idx: int):
        import gym_donkeycar  # noqa: F401

        env_id = self.env_ids[scene_idx]
        scene_specs = MultiSceneEnvV12._SCENE_SPECS
        if env_id not in scene_specs:
            raise KeyError(f"V12 unknown env_id: {env_id}")

        spec = scene_specs[env_id]
        level_name = spec["level_name"]
        scene_key = spec["scene_key"]
        logging_key = spec.get("logging_key", scene_key)  # fallback to scene_key if not specified

        # 场景切换（复用静态方法，不再重复内联定义）
        if self._base_env is None:
            self._base_env = MultiSceneEnv._make_env_with_retry(env_id, self.conf, retries=2, retry_wait_s=1.5)
            print(f"✅ 模拟器已启动，首个场景: {level_name}")
            try:
                MultiSceneEnv._force_reload_scene(self._base_env, level_name, preflight=True)
            except Exception as e:
                print(f"⚠️  场景预加载失败，将继续当前状态: {type(e).__name__}: {e}")
        else:
            try:
                MultiSceneEnv._force_reload_scene(self._base_env, level_name, preflight=False)
            except Exception as e:
                print(f"⚠️  场景切换失败: {type(e).__name__}: {e}")
                print("🔁 尝试重启模拟器并恢复目标场景...")
                try:
                    self._base_env.close()
                except Exception:
                    pass
                self._base_env = None
                self._base_env = MultiSceneEnv._make_env_with_retry(env_id, self.conf, retries=2, retry_wait_s=1.5)
                MultiSceneEnv._force_reload_scene(self._base_env, level_name, preflight=True)

        # 复用公共 wrapper 链工厂函数
        env, action_safety, _high_level, reward_wrapper = _build_v12_wrapper_chain(
            self._base_env,
            scene_key=scene_key,
            logging_key=logging_key,
            domain=self.scene_domains[scene_idx],
            track_geometry=self.track_geometry,
            obs_size=self.obs_size,
            augment=self.augment,
            augment_start_step=self.augment_start_step,
            delta_max=self.delta_max,
            enable_lpf=self.enable_lpf,
            beta=self.beta,
            max_throttle=self.max_throttle,
            w_d=self.w_d, w_dd=self.w_dd, w_m=self.w_m, w_sat=self.w_sat,
            w_time=self.w_time, w_center=self.w_center, w_heading=self.w_heading, w_speed_ref=self.w_speed_ref,
            speed_ref_vmin=self.speed_ref_vmin, speed_ref_vmax=self.speed_ref_vmax,
            speed_ref_kappa_ref=self.speed_ref_kappa_ref,
            lap_reward_scale=self.lap_reward_scale,
            progress_reward_scale=self.progress_reward_scale,
            survival_reward_scale=self.survival_reward_scale,
            collision_penalty_base=self.collision_penalty_base,
            offtrack_penalty_base=self.offtrack_penalty_base,
            vision_mode=self.vision_mode,
            seg_model_path=self.seg_model_path,
            seg_feature_dim=self.seg_feature_dim,
            latent_align_weight=self.latent_align_weight,
            adaptive_delta_max=self.adaptive_delta_max,
            curve_delta_boost=self.curve_delta_boost,
            curve_kappa_ref=self.curve_kappa_ref,
            steer_intent_boost=self.steer_intent_boost,
            hairpin_curve_ratio=self.hairpin_curve_ratio,
            hairpin_min_delta_max=self.hairpin_min_delta_max,
            hairpin_max_delta_max=self.hairpin_max_delta_max,
            w_near_offtrack=self.w_near_offtrack,
            near_offtrack_start_ratio=self.near_offtrack_start_ratio,
            w_near_collision=self.w_near_collision,
            near_collision_start_ratio=self.near_collision_start_ratio,
            overtake_success_bonus=self.overtake_success_bonus,
            total_timesteps=self.total_timesteps,
            speed_vmax=self.speed_vmax,
            speed_kp=self.speed_kp, speed_ki=self.speed_ki, speed_kff=self.speed_kff,
            allow_reverse=self.allow_reverse,
            control_dt=self.control_dt,
            include_cte_in_obs=self.include_cte_in_obs,
            offtrack_leniency_ratio=self.offtrack_leniency_ratio,
            offtrack_leniency_mult=self.offtrack_leniency_mult,
            reset_env_done_grace_steps=self.reset_env_done_grace_steps,
            reset_collision_grace_steps=self.reset_collision_grace_steps,
        )
        self.action_safety_wrapper = action_safety
        self.reward_wrapper = reward_wrapper

        self.active_env = env
        self.active_scene_idx = scene_idx
        self.observation_space = env.observation_space
        self.action_space = env.action_space


# V13/V16 状态构建函数实现在 obv.py，此处导入供本模块内部使用
from .obv import _build_state_v13, _build_state_v16  # noqa: F401, E402


# ============================================================
# V13 多场景环境（6ch 语义观测 + 5维纯传感器状态）
# ============================================================
class MultiSceneEnvV13(MultiSceneEnvV12):
    """
    V13 多场景训练环境。
    与 V12 的核心区别：
      - 用 CanonicalSemanticWrapper 替换 RGBResizeWrapper（6ch 语义观测，128×128）
      - 用 _build_state_v13 替换几何状态（5维，无 TrackGeometryManager 依赖）
      - track_geometry 仅用于奖励 CTE 边界；若为 None，使用默认值
    """

    _SCENE_SPECS: Dict[str, Dict[str, str]] = {}

    def __init__(
        self,
        env_ids: List[str],
        conf: Dict[str, Any],
        scene_weights: List[float],
        scene_specs: Dict[str, Dict[str, str]],
        track_geometry=None,  # 可选，若提供则用于奖励 CTE 边界
        obs_size: int = 128,
        augment: bool = False,
        yellow_dropout_prob: float = 0.20,
        dropout_start_step: int = 0,
        dropout_ramp_steps: int = 200_000,
        # ── V13 ActionAdapter params ──
        adapter_k_delta: float = 0.10,
        adapter_lambda_bias: float = 0.20,
        adapter_k_bias: float = 0.10,
        adapter_steer_core_decay: float = 0.0,
        adapter_v_nominal: float = 1.4,
        adapter_k_turn: float = 0.5,
        adapter_k_bias_speed: float = 0.0,
        adapter_alpha_speed: float = 0.25,
        adapter_v_min: float = 0.6,
        adapter_v_max: float = 1.8,
        **kwargs,
    ):
        MultiSceneEnvV13._SCENE_SPECS = scene_specs

        # V13 特有属性（在 super().__init__ 调用 _create_env(0) 之前必须设好）
        self.yellow_dropout_prob = float(yellow_dropout_prob)
        self.dropout_start_step = int(dropout_start_step)
        self.dropout_ramp_steps = int(max(1, dropout_ramp_steps))

        # ActionAdapter params
        self.adapter_k_delta = float(adapter_k_delta)
        self.adapter_lambda_bias = float(adapter_lambda_bias)
        self.adapter_k_bias = float(adapter_k_bias)
        self.adapter_steer_core_decay = float(adapter_steer_core_decay)
        self.adapter_v_nominal = float(adapter_v_nominal)
        self.adapter_k_turn = float(adapter_k_turn)
        self.adapter_k_bias_speed = float(max(0.0, adapter_k_bias_speed))
        self.adapter_alpha_speed = float(adapter_alpha_speed)
        self.adapter_v_min = float(adapter_v_min)
        self.adapter_v_max = float(adapter_v_max)

        super().__init__(
            env_ids=env_ids,
            conf=conf,
            scene_weights=scene_weights,
            track_geometry=track_geometry,
            scene_specs=scene_specs,
            obs_size=obs_size,
            augment=augment,
            **kwargs,
        )

    def _create_env(self, scene_idx: int):
        import gym_donkeycar  # noqa: F401
        from .wrappers import CanonicalSemanticWrapper
        from .action_adapter import ActionAdapterWrapper

        env_id = self.env_ids[scene_idx]
        scene_specs = MultiSceneEnvV13._SCENE_SPECS
        if env_id not in scene_specs:
            raise KeyError(f"V13 unknown env_id: {env_id}")

        spec = scene_specs[env_id]
        level_name = spec["level_name"]
        scene_key = spec["scene_key"]
        logging_key = spec.get("logging_key", scene_key)
        domain = self.scene_domains[scene_idx]  # ws / rrl / gt

        # 场景切换（与 V12 相同逻辑）
        if self._base_env is None:
            self._base_env = MultiSceneEnv._make_env_with_retry(env_id, self.conf, retries=2, retry_wait_s=1.5)
            _install_custom_episode_over(self._base_env)
            print(f"✅ 模拟器已启动，首个场景: {level_name}")
            try:
                MultiSceneEnv._force_reload_scene(self._base_env, level_name, preflight=True)
            except Exception as e:
                print(f"⚠️  场景预加载失败，将继续当前状态: {type(e).__name__}: {e}")
        else:
            try:
                MultiSceneEnv._force_reload_scene(self._base_env, level_name, preflight=False)
            except Exception as e:
                print(f"⚠️  场景切换失败: {type(e).__name__}: {e}")
                print("🔁 尝试重启模拟器并恢复目标场景...")
                try:
                    self._base_env.close()
                except Exception:
                    pass
                self._base_env = None
                self._base_env = MultiSceneEnv._make_env_with_retry(env_id, self.conf, retries=2, retry_wait_s=1.5)
                _install_custom_episode_over(self._base_env)
                MultiSceneEnv._force_reload_scene(self._base_env, level_name, preflight=True)

        # Per-scene max_cte（缩短出轨后死区，加速 episode 重置）
        _scene_max_cte = spec.get("max_cte", self.conf.get("max_cte", 8.0))
        _set_handler_max_cte(self._base_env, _scene_max_cte, logging_key)

        # ── V13 wrapper 链 ──
        # 构造顺序（inner → outer）:
        #   CanonicalSemantic → RewardWrapper → ActionSafety → ActionAdapter → ObsWrapper → Monitor
        # 动作流（outer → inner）:
        #   Adapter(3D→2D) → Safety(rate-limit steer) → Reward(sees executed action) → env

        env = CanonicalSemanticWrapper(
            self._base_env,
            domain=domain,
            obs_size=self.obs_size,
            augment=self.augment,
            dropout_start_step=self.dropout_start_step,
            dropout_ramp_steps=self.dropout_ramp_steps,
            dropout_max_prob=self.yellow_dropout_prob,
        )

        # CTE 边界：若有 track_geometry 则从几何取，否则用默认值
        if self.track_geometry is not None and hasattr(self.track_geometry, "scenes") \
                and scene_key in self.track_geometry.scenes:
            geo = self.track_geometry.scenes[scene_key]
            cte_left       = float(geo.cte_left)
            cte_right      = float(geo.cte_right)
            cte_left_out   = float(geo.cte_left_out)
            cte_right_out  = float(geo.cte_right_out)
            coord_scale    = float(geo.coord_scale)
            cte_half_width = float(geo.cte_half_width)
        else:
            cte_left = 5.0; cte_right = -5.0
            cte_left_out = 6.5; cte_right_out = -6.5
            coord_scale = 8.0; cte_half_width = 4.6

        # Reward（最内层管道，延迟绑定 safety 引用）
        # 构建 reward kwargs，支持 per-scene override
        _reward_kwargs = dict(
            total_timesteps=self.total_timesteps,
            action_safety_wrapper=None,  # 延迟绑定
            w_d=self.w_d, w_dd=self.w_dd, w_m=self.w_m, w_sat=self.w_sat,
            w_time=self.w_time, w_center=self.w_center,
            w_heading=self.w_heading, w_speed_ref=self.w_speed_ref,
            speed_ref_vmin=self.speed_ref_vmin, speed_ref_vmax=self.speed_ref_vmax,
            speed_ref_kappa_ref=self.speed_ref_kappa_ref,
            lap_reward_scale=self.lap_reward_scale,
            progress_reward_scale=self.progress_reward_scale,
            survival_reward_scale=self.survival_reward_scale,
            collision_penalty_base=self.collision_penalty_base,
            offtrack_penalty_base=self.offtrack_penalty_base,
            w_near_offtrack=self.w_near_offtrack,
            near_offtrack_start_ratio=self.near_offtrack_start_ratio,
            w_near_collision=self.w_near_collision,
            near_collision_start_ratio=self.near_collision_start_ratio,
            overtake_success_bonus=self.overtake_success_bonus,
            cte_left=cte_left, cte_right=cte_right,
            cte_left_out=cte_left_out, cte_right_out=cte_right_out,
            coord_scale=coord_scale,
            offtrack_leniency_ratio=self.offtrack_leniency_ratio,
            offtrack_leniency_mult=self.offtrack_leniency_mult,
            track_geometry=self.track_geometry,
            scene_key=scene_key,
            logging_key=logging_key,
            cte_half_width=cte_half_width,
            reset_env_done_grace_steps=self.reset_env_done_grace_steps,
            reset_collision_grace_steps=self.reset_collision_grace_steps,
        )
        # Per-scene reward overrides（白名单合并）
        _ALLOWED_REWARD_OVERRIDES = {
            "near_offtrack_start_ratio", "w_near_offtrack",
            "w_near_collision", "near_collision_start_ratio",
            "overtake_success_bonus",
            "safe_follow_bonus_scale",
            "post_pass_stability_bonus",
            "post_pass_stability_steps",
            "w_center", "w_heading",
            "w_speed_ref", "speed_ref_vmin", "speed_ref_vmax", "speed_ref_kappa_ref",
            "safe_follow_min_m", "safe_follow_max_m", "safe_follow_risk_max",
            "safe_follow_speed_min", "safe_follow_ttc_min_s", "safe_follow_ttc_max_s",
            "wait_window_bonus_scale", "wait_window_min_gap_m", "wait_window_max_gap_m",
            "wait_window_max_closing_rate", "force_pass_penalty_scale",
            "collision_penalty_base", "offtrack_penalty_base",
            "survival_reward_scale", "progress_reward_scale",
            "lap_reward_scale",
            "cte_norm_scale",
            "reward_decay_ref_steps",
        }
        _reward_overrides = spec.get("reward_overrides", {})
        if _reward_overrides:
            _applied = {}
            for _k, _v in _reward_overrides.items():
                if _k in _ALLOWED_REWARD_OVERRIDES:
                    _reward_kwargs[_k] = _v
                    _applied[_k] = _v
                else:
                    print(f"⚠️  [{logging_key}] reward_overrides: unknown key '{_k}', ignored")
            if _applied:
                print(f"   [{logging_key}] reward_overrides: {_applied}")
        reward_wrapper = DonkeyRewardWrapper(env, **_reward_kwargs)
        env = reward_wrapper

        # Safety（中间层，速率限制 steer）
        action_safety = ActionSafetyWrapper(
            env,
            delta_max=self.delta_max,
            enable_lpf=self.enable_lpf,
            beta=self.beta,
            adaptive_delta_max=self.adaptive_delta_max,
            curve_delta_boost=self.curve_delta_boost,
            curve_kappa_ref=self.curve_kappa_ref,
            steer_intent_boost=self.steer_intent_boost,
            hairpin_curve_ratio=self.hairpin_curve_ratio,
            hairpin_min_delta_max=self.hairpin_min_delta_max,
            hairpin_max_delta_max=self.hairpin_max_delta_max,
        )
        env = action_safety

        # 延迟绑定 safety 引用
        reward_wrapper.action_safety_wrapper = action_safety

        # ActionAdapter（最外层 ActionWrapper，3D → 2D）
        adapter = ActionAdapterWrapper(
            env,
            k_delta=self.adapter_k_delta,
            lambda_bias=self.adapter_lambda_bias,
            k_bias=self.adapter_k_bias,
            steer_core_decay=self.adapter_steer_core_decay,
            v_nominal=self.adapter_v_nominal,
            k_turn=self.adapter_k_turn,
            k_bias_speed=self.adapter_k_bias_speed,
            alpha_speed=self.adapter_alpha_speed,
            v_min=self.adapter_v_min,
            v_max=self.adapter_v_max,
            speed_kp=self.speed_kp,
            speed_ki=self.speed_ki,
            speed_kff=self.speed_kff,
            control_dt=self.control_dt,
            max_throttle=self.max_throttle,
            allow_reverse=self.allow_reverse,
        )
        env = adapter

        env = MultiInputObsWrapper(
            env,
            track_geometry=None,      # V13 观测不依赖几何
            scene_key=scene_key,
            logging_key=logging_key,
            domain=domain,
            obs_size=self.obs_size,
            image_channels=6,
            include_cte_in_obs=False,
            speed_vmax=self.speed_vmax,
            control_wrapper=adapter,
            action_safety_wrapper=action_safety,
            state_builder=_build_state_v13,
            state_dim=7,
            snapshot_dir=self.snapshot_dir,
            snapshot_max_steps=self.snapshot_max_steps,
        )

        env = Monitor(env, info_keywords=MONITOR_INFO_KEYS)

        self.action_safety_wrapper = action_safety
        self.action_adapter_wrapper = adapter
        self.reward_wrapper = reward_wrapper
        self.active_env = env
        self.active_scene_idx = scene_idx
        self.observation_space = env.observation_space
        self.action_space = env.action_space


class MultiSceneEnvV16(MultiSceneEnvV13):
    """
    V16 多场景训练环境。
    在 V13 双域语义观测链上，额外插入 obstacle runtime：
      - reset 后按场景几何布置障碍车
      - step 时向 info 注入 obstacle_dist / obstacle_risk / relative pose
      - state 从 7D 扩为 12D（追加 obstacle context）
    """

    def __init__(
        self,
        env_ids: List[str],
        conf: Dict[str, Any],
        scene_weights: List[float],
        scene_specs: Dict[str, Dict[str, str]],
        track_geometry=None,
        track_dir: str = "",
        obstacle_enabled: bool = True,
        obstacle_scene_keys: Optional[List[str]] = None,
        obstacle_count: int = 2,
        ws_obstacle_count: Optional[int] = None,
        obstacle_free_prob: float = 0.15,
        obstacle_modes: Optional[List[str]] = None,
        ws_obstacle_free_prob: Optional[float] = None,
        obstacle_spawn_ahead_min_m: float = 3.5,
        obstacle_spawn_ahead_max_m: float = 14.0,
        obstacle_min_agent_planar_dist_m: float = 1.5,
        obstacle_min_agent_arc_dist_m: float = 3.5,
        obstacle_min_separation_world: float = 3.0,
        obstacle_lateral_choices: Optional[List[float]] = None,
        ws_obstacle_lateral_choices: Optional[List[float]] = None,
        obstacle_fixed_progress_ratio: Optional[float] = None,
        obstacle_fixed_progress_distribution: Optional[List[Tuple[float, float]]] = None,
        obstacle_fixed_progress_gap: Optional[float] = None,
        obstacle_fixed_progress_gap_min: Optional[float] = None,
        obstacle_fixed_progress_gap_max: Optional[float] = None,
        obstacle_progress_min: Optional[float] = None,
        obstacle_progress_max: Optional[float] = None,
        obstacle_fixed_lateral_ratio: Optional[float] = None,
        gt_obstacle_start_exclusion_half_width_m: Optional[float] = None,
        ws_obstacle_modes: Optional[List[str]] = None,
        ws_obstacle_fixed_progress_ratio: Optional[float] = None,
        ws_obstacle_fixed_progress_gap: Optional[float] = None,
        ws_obstacle_fixed_progress_gap_min: Optional[float] = None,
        ws_obstacle_fixed_progress_gap_max: Optional[float] = None,
        ws_obstacle_progress_min: Optional[float] = None,
        ws_obstacle_progress_max: Optional[float] = None,
        ws_obstacle_fixed_lateral_ratio: Optional[float] = None,
        obstacle_randomize_non_lane_pid_yaw: bool = True,
        obstacle_jitter_amplitude_m: float = 0.10,
        obstacle_jitter_period_s: float = 1.5,
        obstacle_jitter_update_hz: float = 8.0,
        obstacle_nudge_amplitude_m: float = 0.14,
        obstacle_nudge_period_s: float = 1.5,
        obstacle_nudge_update_hz: float = 8.0,
        obstacle_lane_pid_speed_gt: float = 0.85,
        obstacle_lane_pid_speed_ws: float = 0.70,
        obstacle_lane_pid_lookahead_m: float = 0.9,
        obstacle_spawn_gap_s: float = 0.0,
        obstacle_placement_timeout_s: float = 1.5,
        obstacle_seed: Optional[int] = None,
        ego_random_spawn: bool = False,
        ego_spawn_lateral_ratio: float = 0.5,
        ego_spawn_settle_steps: int = 3,
        ego_spawn_settle_sleep_s: float = 0.05,
        sim2real_json: Optional[str] = None,
        obstacle_context_source: str = "runtime",
        obstacle_context_checkpoint: Optional[str] = None,
        obstacle_context_device: str = "cpu",
        obstacle_context_seq_len: int = 16,
        obstacle_context_present_threshold: float = 0.5,
        obstacle_context_present_off_threshold: Optional[float] = None,
        obstacle_context_activation_consecutive: int = 3,
        obstacle_context_deactivation_consecutive: int = 2,
        **kwargs,
    ):
        self.sim2real_json = str(sim2real_json) if sim2real_json else None
        self.obstacle_context_source = str(obstacle_context_source or "runtime").strip().lower()
        self.obstacle_context_checkpoint = str(obstacle_context_checkpoint or "").strip()
        self.obstacle_context_device = str(obstacle_context_device or "cpu").strip()
        self.obstacle_context_seq_len = int(max(1, obstacle_context_seq_len))
        self.obstacle_context_present_threshold = float(
            np.clip(obstacle_context_present_threshold, 0.01, 0.99)
        )
        self.obstacle_context_present_off_threshold = (
            None
            if obstacle_context_present_off_threshold is None
            else float(np.clip(obstacle_context_present_off_threshold, 0.0, 0.99))
        )
        self.obstacle_context_activation_consecutive = int(
            max(1, obstacle_context_activation_consecutive)
        )
        self.obstacle_context_deactivation_consecutive = int(
            max(1, obstacle_context_deactivation_consecutive)
        )
        self.track_dir = str(track_dir or "")
        self.obstacle_enabled = bool(obstacle_enabled)
        self.obstacle_scene_keys = tuple(
            str(x) for x in (
                obstacle_scene_keys
                if obstacle_scene_keys is not None
                else ["waveshare", "generated_track"]
            )
        )
        self.obstacle_count = int(max(1, obstacle_count))
        self.ws_obstacle_count = None if ws_obstacle_count is None else int(max(1, ws_obstacle_count))
        self.obstacle_free_prob = float(np.clip(obstacle_free_prob, 0.0, 1.0))
        self.ws_obstacle_free_prob = (
            None
            if ws_obstacle_free_prob is None
            else float(np.clip(ws_obstacle_free_prob, 0.0, 1.0))
        )
        self.obstacle_modes = tuple(
            str(x).strip().lower()
            for x in (obstacle_modes if obstacle_modes is not None else ["static", "jitter"])
            if str(x).strip()
        ) or ("static",)
        self.obstacle_spawn_ahead_min_m = float(max(0.2, obstacle_spawn_ahead_min_m))
        self.obstacle_spawn_ahead_max_m = float(max(self.obstacle_spawn_ahead_min_m + 1e-3, obstacle_spawn_ahead_max_m))
        self.obstacle_min_agent_planar_dist_m = float(max(0.2, obstacle_min_agent_planar_dist_m))
        self.obstacle_min_agent_arc_dist_m = float(max(0.2, obstacle_min_agent_arc_dist_m))
        self.obstacle_min_separation_world = float(max(0.0, obstacle_min_separation_world))
        self.obstacle_lateral_choices = tuple(
            float(np.clip(x, 0.0, 1.0))
            for x in (obstacle_lateral_choices if obstacle_lateral_choices is not None else [0.35, 0.5, 0.65])
        )
        self.ws_obstacle_lateral_choices = (
            None
            if ws_obstacle_lateral_choices is None
            else tuple(
                float(np.clip(x, 0.0, 1.0))
                for x in ws_obstacle_lateral_choices
            )
        )
        self.obstacle_fixed_progress_ratio = (
            None
            if obstacle_fixed_progress_ratio is None
            else float(np.clip(obstacle_fixed_progress_ratio, 0.0, 1.0))
        )
        self.obstacle_fixed_progress_distribution = self._normalize_obstacle_progress_distribution(
            obstacle_fixed_progress_distribution
        )
        self.obstacle_fixed_progress_gap = (
            None
            if obstacle_fixed_progress_gap is None
            else float(np.clip(obstacle_fixed_progress_gap, 0.0, 1.0))
        )
        self.obstacle_fixed_progress_gap_min = (
            None
            if obstacle_fixed_progress_gap_min is None
            else float(np.clip(obstacle_fixed_progress_gap_min, 0.0, 1.0))
        )
        self.obstacle_fixed_progress_gap_max = (
            None
            if obstacle_fixed_progress_gap_max is None
            else float(np.clip(obstacle_fixed_progress_gap_max, 0.0, 1.0))
        )
        if (
            self.obstacle_fixed_progress_gap_min is not None
            and self.obstacle_fixed_progress_gap_max is not None
            and self.obstacle_fixed_progress_gap_max < self.obstacle_fixed_progress_gap_min
        ):
            self.obstacle_fixed_progress_gap_min, self.obstacle_fixed_progress_gap_max = (
                self.obstacle_fixed_progress_gap_max,
                self.obstacle_fixed_progress_gap_min,
            )
        self.obstacle_progress_min = (
            None
            if obstacle_progress_min is None
            else float(np.clip(obstacle_progress_min, 0.0, 1.0))
        )
        self.obstacle_progress_max = (
            None
            if obstacle_progress_max is None
            else float(np.clip(obstacle_progress_max, 0.0, 1.0))
        )
        if (
            self.obstacle_progress_min is not None
            and self.obstacle_progress_max is not None
            and self.obstacle_progress_max < self.obstacle_progress_min
        ):
            self.obstacle_progress_min, self.obstacle_progress_max = (
                self.obstacle_progress_max,
                self.obstacle_progress_min,
            )
        self.obstacle_fixed_lateral_ratio = (
            None
            if obstacle_fixed_lateral_ratio is None
            else float(np.clip(obstacle_fixed_lateral_ratio, 0.0, 1.0))
        )
        self.gt_obstacle_start_exclusion_half_width_m = (
            None
            if gt_obstacle_start_exclusion_half_width_m is None
            else float(max(0.0, gt_obstacle_start_exclusion_half_width_m))
        )
        self.ws_obstacle_modes = (
            None
            if ws_obstacle_modes is None
            else tuple(
                str(x).strip().lower()
                for x in ws_obstacle_modes
                if str(x).strip()
            ) or None
        )
        self.ws_obstacle_fixed_progress_ratio = (
            None
            if ws_obstacle_fixed_progress_ratio is None
            else float(np.clip(ws_obstacle_fixed_progress_ratio, 0.0, 1.0))
        )
        self.ws_obstacle_fixed_progress_gap = (
            None
            if ws_obstacle_fixed_progress_gap is None
            else float(np.clip(ws_obstacle_fixed_progress_gap, 0.0, 1.0))
        )
        self.ws_obstacle_fixed_progress_gap_min = (
            None
            if ws_obstacle_fixed_progress_gap_min is None
            else float(np.clip(ws_obstacle_fixed_progress_gap_min, 0.0, 1.0))
        )
        self.ws_obstacle_fixed_progress_gap_max = (
            None
            if ws_obstacle_fixed_progress_gap_max is None
            else float(np.clip(ws_obstacle_fixed_progress_gap_max, 0.0, 1.0))
        )
        if (
            self.ws_obstacle_fixed_progress_gap_min is not None
            and self.ws_obstacle_fixed_progress_gap_max is not None
            and self.ws_obstacle_fixed_progress_gap_max < self.ws_obstacle_fixed_progress_gap_min
        ):
            self.ws_obstacle_fixed_progress_gap_min, self.ws_obstacle_fixed_progress_gap_max = (
                self.ws_obstacle_fixed_progress_gap_max,
                self.ws_obstacle_fixed_progress_gap_min,
            )
        self.ws_obstacle_progress_min = (
            None
            if ws_obstacle_progress_min is None
            else float(np.clip(ws_obstacle_progress_min, 0.0, 1.0))
        )
        self.ws_obstacle_progress_max = (
            None
            if ws_obstacle_progress_max is None
            else float(np.clip(ws_obstacle_progress_max, 0.0, 1.0))
        )
        self.ws_obstacle_fixed_lateral_ratio = (
            None
            if ws_obstacle_fixed_lateral_ratio is None
            else float(np.clip(ws_obstacle_fixed_lateral_ratio, 0.0, 1.0))
        )
        self.obstacle_randomize_non_lane_pid_yaw = bool(obstacle_randomize_non_lane_pid_yaw)
        self.obstacle_jitter_amplitude_m = float(max(0.0, obstacle_jitter_amplitude_m))
        self.obstacle_jitter_period_s = float(max(0.2, obstacle_jitter_period_s))
        self.obstacle_jitter_update_hz = float(max(1.0, obstacle_jitter_update_hz))
        self.obstacle_nudge_amplitude_m = float(max(0.0, obstacle_nudge_amplitude_m))
        self.obstacle_nudge_period_s = float(max(0.2, obstacle_nudge_period_s))
        self.obstacle_nudge_update_hz = float(max(1.0, obstacle_nudge_update_hz))
        self.obstacle_lane_pid_speed_gt = float(max(0.0, obstacle_lane_pid_speed_gt))
        self.obstacle_lane_pid_speed_ws = float(max(0.0, obstacle_lane_pid_speed_ws))
        self.obstacle_lane_pid_lookahead_m = float(max(0.1, obstacle_lane_pid_lookahead_m))
        self.obstacle_spawn_gap_s = float(max(0.0, obstacle_spawn_gap_s))
        self.obstacle_placement_timeout_s = float(max(0.1, obstacle_placement_timeout_s))
        self.obstacle_seed = obstacle_seed
        self.ego_random_spawn = bool(ego_random_spawn)
        self.ego_spawn_lateral_ratio = float(np.clip(ego_spawn_lateral_ratio, 0.0, 1.0))
        self.ego_spawn_settle_steps = int(max(1, ego_spawn_settle_steps))
        self.ego_spawn_settle_sleep_s = float(max(0.0, ego_spawn_settle_sleep_s))
        self._obstacle_runtime = None

        super().__init__(
            env_ids=env_ids,
            conf=conf,
            scene_weights=scene_weights,
            scene_specs=scene_specs,
            track_geometry=track_geometry,
            **kwargs,
        )

        # 步数预算统计（用于学习窗口补偿）
        self._step_budget_stats = {
            "ws": {
                "episode_count": 0,
                "window_with_obs_steps": 0,
                "window_without_obs_steps": 0,
            },
            "gt": {
                "episode_count": 0,
                "window_with_obs_steps": 0,
                "window_without_obs_steps": 0,
            }
        }
        self._window_episode_threshold = 50
        self._curriculum_phase = "warmup"  # 当前课程阶段，由train_v16更新


    @staticmethod
    def _normalize_obstacle_progress_distribution(
        values: Optional[List[Tuple[float, float]]]
    ) -> Optional[Tuple[Tuple[float, float], ...]]:
        if values is None:
            return None
        normalized: List[Tuple[float, float]] = []
        for item in values:
            if item is None:
                continue
            if len(item) != 2:
                raise ValueError(
                    "obstacle_fixed_progress_distribution entries must be "
                    "(probability, progress_ratio) pairs"
                )
            probability, progress_ratio = item
            probability = float(np.clip(probability, 0.0, 1.0))
            if probability <= 0.0:
                continue
            normalized.append((probability, float(np.clip(progress_ratio, 0.0, 1.0))))
        total = sum(probability for probability, _progress_ratio in normalized)
        if total > 1.0 + 1e-6:
            raise ValueError(
                "obstacle_fixed_progress_distribution probabilities must sum to <= 1.0, "
                f"got {total:.6f}"
            )
        return tuple(normalized) or None


    def _build_obstacle_runtime_config(self):
        from .obstacle_runtime import ObstacleRuntimeConfig

        return ObstacleRuntimeConfig(
            enabled=self.obstacle_enabled,
            active_scene_keys=self.obstacle_scene_keys,
            obstacle_count=self.obstacle_count,
            ws_obstacle_count=self.ws_obstacle_count,
            obstacle_free_prob=self.obstacle_free_prob,
            obstacle_modes=self.obstacle_modes,
            ws_obstacle_free_prob=self.ws_obstacle_free_prob,
            min_obstacle_separation_world=self.obstacle_min_separation_world,
            spawn_ahead_min_m=self.obstacle_spawn_ahead_min_m,
            spawn_ahead_max_m=self.obstacle_spawn_ahead_max_m,
            min_agent_planar_dist_m=self.obstacle_min_agent_planar_dist_m,
            min_agent_arc_dist_m=self.obstacle_min_agent_arc_dist_m,
            lateral_choices=self.obstacle_lateral_choices,
            ws_lateral_choices=self.ws_obstacle_lateral_choices,
            fixed_progress_ratio=self.obstacle_fixed_progress_ratio,
            fixed_progress_distribution=self.obstacle_fixed_progress_distribution,
            fixed_progress_gap_ratio=self.obstacle_fixed_progress_gap,
            fixed_progress_gap_ratio_min=self.obstacle_fixed_progress_gap_min,
            fixed_progress_gap_ratio_max=self.obstacle_fixed_progress_gap_max,
            obstacle_progress_min=self.obstacle_progress_min,
            obstacle_progress_max=self.obstacle_progress_max,
            fixed_lateral_ratio=self.obstacle_fixed_lateral_ratio,
            gt_obstacle_start_exclusion_half_width_m=self.gt_obstacle_start_exclusion_half_width_m,
            ws_obstacle_modes=self.ws_obstacle_modes,
            ws_obstacle_fixed_progress_ratio=self.ws_obstacle_fixed_progress_ratio,
            ws_fixed_progress_gap_ratio=self.ws_obstacle_fixed_progress_gap,
            ws_fixed_progress_gap_ratio_min=self.ws_obstacle_fixed_progress_gap_min,
            ws_fixed_progress_gap_ratio_max=self.ws_obstacle_fixed_progress_gap_max,
            ws_obstacle_progress_min=self.ws_obstacle_progress_min,
            ws_obstacle_progress_max=self.ws_obstacle_progress_max,
            ws_obstacle_fixed_lateral_ratio=self.ws_obstacle_fixed_lateral_ratio,
            randomize_non_lane_pid_yaw=self.obstacle_randomize_non_lane_pid_yaw,
            jitter_amplitude_m=self.obstacle_jitter_amplitude_m,
            jitter_period_s=self.obstacle_jitter_period_s,
            jitter_update_hz=self.obstacle_jitter_update_hz,
            nudge_amplitude_m=self.obstacle_nudge_amplitude_m,
            nudge_period_s=self.obstacle_nudge_period_s,
            nudge_update_hz=self.obstacle_nudge_update_hz,
            lane_pid_speed_gt=self.obstacle_lane_pid_speed_gt,
            lane_pid_speed_ws=self.obstacle_lane_pid_speed_ws,
            lane_pid_lookahead_m=self.obstacle_lane_pid_lookahead_m,
            spawn_gap_s=self.obstacle_spawn_gap_s,
            placement_timeout_s=self.obstacle_placement_timeout_s,
            ego_random_spawn=self.ego_random_spawn,
            ego_spawn_lateral_ratio=self.ego_spawn_lateral_ratio,
            ego_spawn_settle_steps=self.ego_spawn_settle_steps,
            ego_spawn_settle_sleep_s=self.ego_spawn_settle_sleep_s,
            seed=self.obstacle_seed,
        )

    def _create_env(self, scene_idx: int):
        import gym_donkeycar  # noqa: F401
        from .wrappers import CanonicalSemanticWrapper
        from .action_adapter import ActionAdapterWrapper
        from .obstacle_runtime import ObstacleRuntimeManager, ScenarioObstacleWrapper

        env_id = self.env_ids[scene_idx]
        scene_specs = MultiSceneEnvV13._SCENE_SPECS
        if env_id not in scene_specs:
            raise KeyError(f"V16 unknown env_id: {env_id}")

        spec = scene_specs[env_id]
        level_name = spec["level_name"]
        scene_key = spec["scene_key"]
        logging_key = spec.get("logging_key", scene_key)
        domain = self.scene_domains[scene_idx]

        if self._base_env is None:
            self._base_env = MultiSceneEnv._make_env_with_retry(env_id, self.conf, retries=2, retry_wait_s=1.5)
            _install_custom_episode_over(self._base_env)
            print(f"✅ 模拟器已启动，首个场景: {level_name}")
            try:
                MultiSceneEnv._force_reload_scene(self._base_env, level_name, preflight=True)
            except Exception as e:
                print(f"⚠️  场景预加载失败，将继续当前状态: {type(e).__name__}: {e}")
        else:
            if self._obstacle_runtime is not None and getattr(self._obstacle_runtime, "scene_key", "") != scene_key:
                self._obstacle_runtime.close()
            try:
                MultiSceneEnv._force_reload_scene(self._base_env, level_name, preflight=False)
            except Exception as e:
                print(f"⚠️  场景切换失败: {type(e).__name__}: {e}")
                print("🔁 尝试重启模拟器并恢复目标场景...")
                if self._obstacle_runtime is not None:
                    self._obstacle_runtime.close()
                try:
                    self._base_env.close()
                except Exception:
                    pass
                self._base_env = None
                self._base_env = MultiSceneEnv._make_env_with_retry(env_id, self.conf, retries=2, retry_wait_s=1.5)
                _install_custom_episode_over(self._base_env)
                MultiSceneEnv._force_reload_scene(self._base_env, level_name, preflight=True)

        _scene_max_cte = spec.get("max_cte", self.conf.get("max_cte", 8.0))
        _set_handler_max_cte(self._base_env, _scene_max_cte, logging_key)

        if self._obstacle_runtime is None:
            self._obstacle_runtime = ObstacleRuntimeManager(
                track_geometry=self.track_geometry,
                conf=self.conf,
                track_dir=self.track_dir,
                config=self._build_obstacle_runtime_config(),
            )
        self._obstacle_runtime.attach_scene(
            base_env=self._base_env,
            env_id=env_id,
            scene_key=scene_key,
            logging_key=logging_key,
        )

        env = ScenarioObstacleWrapper(self._base_env, runtime=self._obstacle_runtime)
        if getattr(self, "sim2real_json", None):
            from .sim2real_wrapper import Sim2RealActionWrapper
            env = Sim2RealActionWrapper.from_json(env, self.sim2real_json)
        env = CanonicalSemanticWrapper(
            env,
            domain=domain,
            obs_size=self.obs_size,
            augment=self.augment,
            dropout_start_step=self.dropout_start_step,
            dropout_ramp_steps=self.dropout_ramp_steps,
            dropout_max_prob=self.yellow_dropout_prob,
        )

        if self.track_geometry is not None and hasattr(self.track_geometry, "scenes") \
                and scene_key in self.track_geometry.scenes:
            geo = self.track_geometry.scenes[scene_key]
            cte_left = float(geo.cte_left)
            cte_right = float(geo.cte_right)
            cte_left_out = float(geo.cte_left_out)
            cte_right_out = float(geo.cte_right_out)
            coord_scale = float(geo.coord_scale)
            cte_half_width = float(geo.cte_half_width)
        else:
            cte_left = 5.0
            cte_right = -5.0
            cte_left_out = 6.5
            cte_right_out = -6.5
            coord_scale = 8.0
            cte_half_width = 4.6

        _reward_kwargs = dict(
            total_timesteps=self.total_timesteps,
            action_safety_wrapper=None,
            w_d=self.w_d, w_dd=self.w_dd, w_m=self.w_m, w_sat=self.w_sat,
            w_time=self.w_time, w_center=self.w_center,
            w_heading=self.w_heading, w_speed_ref=self.w_speed_ref,
            speed_ref_vmin=self.speed_ref_vmin, speed_ref_vmax=self.speed_ref_vmax,
            speed_ref_kappa_ref=self.speed_ref_kappa_ref,
            lap_reward_scale=self.lap_reward_scale,
            progress_reward_scale=self.progress_reward_scale,
            survival_reward_scale=self.survival_reward_scale,
            collision_penalty_base=self.collision_penalty_base,
            offtrack_penalty_base=self.offtrack_penalty_base,
            w_near_offtrack=self.w_near_offtrack,
            near_offtrack_start_ratio=self.near_offtrack_start_ratio,
            w_near_collision=self.w_near_collision,
            near_collision_start_ratio=self.near_collision_start_ratio,
            overtake_success_bonus=self.overtake_success_bonus,
            cte_left=cte_left, cte_right=cte_right,
            cte_left_out=cte_left_out, cte_right_out=cte_right_out,
            coord_scale=coord_scale,
            offtrack_leniency_ratio=self.offtrack_leniency_ratio,
            offtrack_leniency_mult=self.offtrack_leniency_mult,
            track_geometry=self.track_geometry,
            scene_key=scene_key,
            logging_key=logging_key,
            cte_half_width=cte_half_width,
            reset_env_done_grace_steps=self.reset_env_done_grace_steps,
            reset_collision_grace_steps=self.reset_collision_grace_steps,
        )
        _ALLOWED_REWARD_OVERRIDES = {
            "near_offtrack_start_ratio", "w_near_offtrack",
            "w_near_collision", "near_collision_start_ratio",
            "overtake_success_bonus",
            "safe_follow_bonus_scale",
            "post_pass_stability_bonus",
            "post_pass_stability_steps",
            "w_center", "w_heading",
            "w_speed_ref", "speed_ref_vmin", "speed_ref_vmax", "speed_ref_kappa_ref",
            "safe_follow_min_m", "safe_follow_max_m", "safe_follow_risk_max",
            "safe_follow_speed_min", "safe_follow_ttc_min_s", "safe_follow_ttc_max_s",
            "wait_window_bonus_scale", "wait_window_min_gap_m", "wait_window_max_gap_m",
            "wait_window_max_closing_rate", "force_pass_penalty_scale",
            "collision_penalty_base", "offtrack_penalty_base",
            "survival_reward_scale", "progress_reward_scale",
            "lap_reward_scale",
            "cte_norm_scale",
            "reward_decay_ref_steps",
        }
        _reward_overrides = spec.get("reward_overrides", {})
        if _reward_overrides:
            _applied = {}
            for _k, _v in _reward_overrides.items():
                if _k in _ALLOWED_REWARD_OVERRIDES:
                    _reward_kwargs[_k] = _v
                    _applied[_k] = _v
            if _applied:
                print(f"   [{logging_key}] reward_overrides: {_applied}")
        reward_wrapper = DonkeyRewardWrapper(env, **_reward_kwargs)
        env = reward_wrapper

        action_safety = ActionSafetyWrapper(
            env,
            delta_max=self.delta_max,
            enable_lpf=self.enable_lpf,
            beta=self.beta,
            adaptive_delta_max=self.adaptive_delta_max,
            curve_delta_boost=self.curve_delta_boost,
            curve_kappa_ref=self.curve_kappa_ref,
            steer_intent_boost=self.steer_intent_boost,
            hairpin_curve_ratio=self.hairpin_curve_ratio,
            hairpin_min_delta_max=self.hairpin_min_delta_max,
            hairpin_max_delta_max=self.hairpin_max_delta_max,
        )
        env = action_safety
        reward_wrapper.action_safety_wrapper = action_safety

        adapter = ActionAdapterWrapper(
            env,
            k_delta=self.adapter_k_delta,
            lambda_bias=self.adapter_lambda_bias,
            k_bias=self.adapter_k_bias,
            steer_core_decay=self.adapter_steer_core_decay,
            v_nominal=self.adapter_v_nominal,
            k_turn=self.adapter_k_turn,
            k_bias_speed=self.adapter_k_bias_speed,
            alpha_speed=self.adapter_alpha_speed,
            v_min=self.adapter_v_min,
            v_max=self.adapter_v_max,
            speed_kp=self.speed_kp,
            speed_ki=self.speed_ki,
            speed_kff=self.speed_kff,
            control_dt=self.control_dt,
            max_throttle=self.max_throttle,
            allow_reverse=self.allow_reverse,
        )
        env = adapter

        env = MultiInputObsWrapper(
            env,
            track_geometry=None,
            scene_key=scene_key,
            logging_key=logging_key,
            domain=domain,
            obs_size=self.obs_size,
            image_channels=6,
            include_cte_in_obs=False,
            speed_vmax=self.speed_vmax,
            control_wrapper=adapter,
            action_safety_wrapper=action_safety,
            state_builder=_build_state_v16,
            state_dim=12,
            snapshot_dir=self.snapshot_dir,
            snapshot_max_steps=self.snapshot_max_steps,
            obstacle_context_source=self.obstacle_context_source,
            obstacle_context_checkpoint=self.obstacle_context_checkpoint,
            obstacle_context_device=self.obstacle_context_device,
            obstacle_context_seq_len=self.obstacle_context_seq_len,
            obstacle_context_present_threshold=self.obstacle_context_present_threshold,
            obstacle_context_present_off_threshold=self.obstacle_context_present_off_threshold,
            obstacle_context_activation_consecutive=self.obstacle_context_activation_consecutive,
            obstacle_context_deactivation_consecutive=self.obstacle_context_deactivation_consecutive,
        )

        env = Monitor(env, info_keywords=MONITOR_INFO_KEYS)

        self.action_safety_wrapper = action_safety
        self.action_adapter_wrapper = adapter
        self.reward_wrapper = reward_wrapper
        self.active_env = env
        self.active_scene_idx = scene_idx
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs):
        """重写reset以跟踪step预算统计"""
        obs = super().reset(**kwargs)

        # 从self中获取scene_key信息（MultiSceneEnv的属性）
        if hasattr(self, 'scene_key'):
            scene_key = self.scene_key
        else:
            scene_key = "unknown"

        if hasattr(self, 'logging_key'):
            logging_key = self.logging_key
        else:
            logging_key = scene_key

        # 使用logging_key (ws/gt) 来统计
        if logging_key in self._step_budget_stats:
            self._step_budget_stats[logging_key]["episode_count"] += 1

            # 检查是否达到窗口阈值，需要重置窗口
            if self._step_budget_stats[logging_key]["episode_count"] % self._window_episode_threshold == 0:
                self._step_budget_stats[logging_key]["window_with_obs_steps"] = 0
                self._step_budget_stats[logging_key]["window_without_obs_steps"] = 0

        return obs

    def step(self, action):
        """重写step以跟踪step预算统计"""
        obs, reward, done, info = super().step(action)

        # 检查本步是否有障碍
        scene_key = info.get("scene_key", "unknown")
        logging_key = info.get("logging_key", scene_key)

        # 从info中检测是否有活跃的障碍（通过obstacle_dist > 0或接近碰撞标志）
        has_obstacle = False
        if "obstacle_dist" in info and info["obstacle_dist"] > 0:
            has_obstacle = True
        elif "near_collision" in info and info["near_collision"] > 0:
            has_obstacle = True

        # 统计步数
        if logging_key in self._step_budget_stats:
            if has_obstacle:
                self._step_budget_stats[logging_key]["window_with_obs_steps"] += 1
            else:
                self._step_budget_stats[logging_key]["window_without_obs_steps"] += 1

        return obs, reward, done, info

    def close(self):
        if self._obstacle_runtime is not None:
            self._obstacle_runtime.close()
        super().close()
