"""
module/callbacks.py
所有 SB3 训练回调：PTHExport、Coverage、PerSceneStats、AdaptiveLR、
TrainingMetricsFileLogger、BestModel、ShortEpisodeLogger。
"""

import json
import os
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from stable_baselines3.common.callbacks import BaseCallback

from .utils import _get_domain_for_env


# ============================================================
# checkpoint + .pth 导出
# ============================================================
class PTHExportCallback(BaseCallback):
    """每 save_freq 步自动导出 .zip + .pth。"""

    def __init__(self, save_path: str, save_freq: int = 10000, name_prefix: str = "v12", verbose: int = 0):
        super().__init__(verbose)
        self.save_path   = save_path
        self.save_freq   = int(max(1, save_freq))
        self.name_prefix = name_prefix
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            stem = os.path.join(self.save_path, f"{self.name_prefix}_{self.n_calls}_steps")
            self.model.save(stem)
            pth = stem + "_policy.pth"
            torch.save(self.model.policy.state_dict(), pth)
            if self.verbose > 0:
                print(f"\n💾 Checkpoint: {self.n_calls}步 -> {stem}.zip + {pth}")
        return True


# ============================================================
# Coverage 日志（验证 Coverage Dropout 效果）
# ============================================================
class CoverageLoggingCallback(BaseCallback):
    """周期性记录 mask 通道 coverage 统计到 TensorBoard。"""

    def __init__(self, env, log_freq: int = 500, verbose: int = 0):
        super().__init__(verbose)
        self._env = env
        self.log_freq = int(max(1, log_freq))

    def _find_lane_wrapper(self):
        from .wrappers import V9YellowLaneWrapper
        from .multi_scene_env import MultiSceneEnv

        env = self._env
        while hasattr(env, "envs"):
            env = env.envs[0]
        while env is not None:
            if isinstance(env, V9YellowLaneWrapper):
                return env
            if isinstance(env, MultiSceneEnv):
                inner = getattr(env, "active_env", None)
                while inner is not None:
                    if isinstance(inner, V9YellowLaneWrapper):
                        return inner
                    inner = getattr(inner, "env", None)
                return None
            env = getattr(env, "env", None)
        return None

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            wrapper = self._find_lane_wrapper()
            if wrapper is not None:
                stats = wrapper.get_coverage_stats()
                self.logger.record("coverage/mask_mean", stats["mean"])
                self.logger.record("coverage/mask_std",  stats["std"])
        return True


# ============================================================
# 按场景统计（原 PerDomainStatsCallback，改为按 scene_key 分组）
# ============================================================
class PerSceneStatsCallback(BaseCallback):
    """按 scene_key 汇总 ep_info_buffer，避免混合均值掩盖单场景退化。"""

    REWARD_PART_KEYS = (
        "ep_r_survival", "ep_r_speed", "ep_r_progress", "ep_r_cte", "ep_r_collision",
        "ep_r_near_offtrack", "ep_r_near_collision",
        "ep_r_center", "ep_r_heading", "ep_r_speed_ref", "ep_r_time",
        "ep_r_lap", "ep_r_lap_raw", "ep_r_overtake", "ep_overtake_count",
        "ep_soft_lap_count", "ep_r_smooth", "ep_r_jerk",
        "ep_r_mismatch", "ep_r_sat", "ep_r_total",
    )
    DIAG_EP_KEYS = (
        "ep_cte_abs_p50", "ep_cte_abs_p90", "ep_cte_abs_p99",
        "ep_cte_over_in_rate", "ep_cte_over_out_rate",
        "ep_rate_limit_hit_rate", "ep_steer_clip_hit_rate",
        "ep_throttle_high_penalty_hit_rate",
        "ep_offtrack_counter_max", "ep_stuck_counter_max",
    )
    TERM_BUCKETS = ("collision", "stuck", "offtrack", "env_done", "normal")

    def __init__(self, check_freq: int = 1000, short_episode_threshold: int = 10, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = int(max(1, check_freq))
        self.short_episode_threshold = int(max(1, short_episode_threshold))

    def _collect_scene_stats(self) -> Dict[str, Any]:
        if len(self.model.ep_info_buffer) == 0:
            return {}
        grouped: Dict[str, List[dict]] = {}
        for ep in self.model.ep_info_buffer:
            # 优先用 logging_key（V12），其次 scene_key（V9），向下兼容 domain 字段
            group_key = ep.get("logging_key") or ep.get("scene_key") or ep.get("domain", "unknown")
            grouped.setdefault(group_key, []).append(ep)
        stats: Dict[str, Any] = {}
        for sk, episodes in grouped.items():
            rs   = [float(e["r"]) for e in episodes if "r" in e]
            ls   = [float(e["l"]) for e in episodes if "l" in e]
            covs = [float(e["mask_coverage"]) for e in episodes if "mask_coverage" in e]
            short_eps = [e for e in episodes if float(e.get("l", 9999)) < self.short_episode_threshold]
            reward_parts: Dict[str, float] = {}
            for key in self.REWARD_PART_KEYS:
                vals = [float(e[key]) for e in episodes if key in e]
                if vals:
                    reward_parts[key] = float(np.mean(vals))
            diag_parts: Dict[str, float] = {}
            for key in self.DIAG_EP_KEYS:
                vals = [float(e[key]) for e in episodes if key in e]
                if vals:
                    diag_parts[key] = float(np.mean(vals))
            term_counts = {k: 0 for k in self.TERM_BUCKETS}
            for e in episodes:
                reason_tokens = set(str(e.get("termination_reason", "normal") or "normal").split("+"))
                has_special = False
                for tk in ("collision", "stuck", "offtrack", "env_done"):
                    if tk in reason_tokens:
                        term_counts[tk] += 1
                        has_special = True
                if not has_special:
                    term_counts["normal"] += 1
            denom = max(1, len(episodes))
            term_rates = {f"term_{k}": float(term_counts[k] / denom) for k in self.TERM_BUCKETS}
            stats[sk] = {
                "n":                  len(episodes),
                "mean_reward":        float(np.mean(rs))   if rs   else float("nan"),
                "mean_len":           float(np.mean(ls))   if ls   else float("nan"),
                "mean_cov":           float(np.mean(covs)) if covs else float("nan"),
                "short_episode_rate": len(short_eps) / max(1, len(episodes)),
                "reward_parts":       reward_parts,
                "diag_parts":         diag_parts,
                "term_rates":         term_rates,
            }
        return stats

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True
        stats = self._collect_scene_stats()
        if not stats:
            return True

        all_rewards = []
        for sk, s in stats.items():
            if not np.isnan(s["mean_reward"]):
                self.logger.record(f"scene/{sk}_ep_rew_mean", s["mean_reward"])
                all_rewards.append(s["mean_reward"])
            if not np.isnan(s["mean_len"]):
                self.logger.record(f"scene/{sk}_ep_len_mean", s["mean_len"])
            if not np.isnan(s["mean_cov"]):
                self.logger.record(f"scene/{sk}_mask_cov_mean", s["mean_cov"])
            self.logger.record(f"scene/{sk}_episodes_in_buffer", s["n"])
            self.logger.record(f"scene/{sk}_short_ep_rate", s["short_episode_rate"])
            for key, mean_val in s.get("reward_parts", {}).items():
                self.logger.record(f"scene/{sk}_{key}_mean", mean_val)
            for key, mean_val in s.get("diag_parts", {}).items():
                self.logger.record(f"scene/{sk}_{key}_mean", mean_val)
            for key, rate_val in s.get("term_rates", {}).items():
                self.logger.record(f"scene/{sk}_{key}_rate", rate_val)

        # 跨场景平衡指标
        if len(all_rewards) >= 2:
            self.logger.record("scene/balanced_min_reward", float(min(all_rewards)))

        if self.verbose > 0:
            parts = []
            for sk in sorted(stats.keys()):
                s = stats[sk]
                short_pct = s["short_episode_rate"] * 100
                parts.append(
                    f"{sk}: n={s['n']}, rew={s['mean_reward']:.1f}, len={s['mean_len']:.0f}"
                    + (f", short={short_pct:.0f}%" if short_pct > 5 else "")
                    + (f", cov={s['mean_cov']:.3f}" if not np.isnan(s["mean_cov"]) else "")
                )
            print("📈 按场景统计 | " + " | ".join(parts))
        return True


# 向下兼容别名
PerDomainStatsCallback = PerSceneStatsCallback


# ============================================================
# 场景调度日志（采样概率 / 动态调权）
# ============================================================
class SceneSchedulerLoggingCallback(BaseCallback):
    """记录 MultiSceneEnv 的场景采样与动态调权状态。"""

    def __init__(self, check_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = int(max(1, check_freq))

    @staticmethod
    def _find_multi_scene_env(root_env):
        env = root_env
        while hasattr(env, "envs"):
            env = env.envs[0]
        while env is not None:
            if hasattr(env, "env_ids") and hasattr(env, "scene_weights") and hasattr(env, "active_scene_idx"):
                return env
            env = getattr(env, "env", None)
        return None

    @staticmethod
    def _scene_log_key(ms_env, idx: int) -> str:
        env_id = str(ms_env.env_ids[idx])
        try:
            from .multi_scene_env import MultiSceneEnvV12
            spec = getattr(MultiSceneEnvV12, "_SCENE_SPECS", {}).get(env_id, {})
            if isinstance(spec, dict):
                k = spec.get("logging_key") or spec.get("scene_key")
                if isinstance(k, str) and k:
                    return k
        except Exception:
            pass
        parts = env_id.split("-")
        return parts[1] if len(parts) >= 2 else f"s{idx}"

    @staticmethod
    def _finite_or_none(v: Any) -> Optional[float]:
        try:
            f = float(v)
        except Exception:
            return None
        return f if np.isfinite(f) else None

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        ms_env = self._find_multi_scene_env(getattr(self, "training_env", None))
        if ms_env is None:
            return True

        n = len(getattr(ms_env, "env_ids", []))
        if n <= 0:
            return True

        cur_w = list(getattr(ms_env, "scene_weights", []))
        target_w = list(getattr(ms_env, "_last_dynamic_target_weights", []))
        prev_w = list(getattr(ms_env, "_last_dynamic_prev_weights", []))
        deficits = list(getattr(ms_env, "_last_dynamic_deficits", []))
        rew_means = list(getattr(ms_env, "_last_dynamic_reward_means", []))
        len_means = list(getattr(ms_env, "_last_dynamic_len_means", []))
        succ_means = list(getattr(ms_env, "_last_dynamic_success_means", []))
        hard_succ_means = list(getattr(ms_env, "_last_dynamic_hard_success_means", []))

        cand = [int(i) for i in getattr(ms_env, "_last_sampling_candidates", [])]
        probs = [float(p) for p in getattr(ms_env, "_last_sampling_probs", [])]
        prob_map = {i: p for i, p in zip(cand, probs)}
        selected_idx = int(getattr(ms_env, "_last_sampling_choice", getattr(ms_env, "active_scene_idx", 0)))
        sample_reason = str(getattr(ms_env, "_last_sampling_reason", "unknown") or "unknown")
        sample_reason_keys = (
            "init",
            "single_scene",
            "single_candidate",
            "hold_min_episodes",
            "weighted",
            "step_budget",
            "unknown",
        )

        self.logger.record("scene/active_scene_idx", float(getattr(ms_env, "active_scene_idx", 0)))
        self.logger.record("scene/active_scene_episode_streak", float(getattr(ms_env, "_cur_scene_episodes", 0)))
        self.logger.record("scene/active_scene_step_streak", float(getattr(ms_env, "_cur_scene_steps", 0)))
        self.logger.record("scene/sample_used_step_balance", float(bool(getattr(ms_env, "_last_sampling_used_step_balance", False))))
        self.logger.record("scene/dynamic_update_episode", float(getattr(ms_env, "_last_dynamic_update_episode", 0)))
        for rk in sample_reason_keys:
            self.logger.record(f"scene/sample_reason_{rk}", float(sample_reason == rk))

        for i in range(n):
            sk = self._scene_log_key(ms_env, i)
            self.logger.record(f"scene/{sk}_weight_current", float(cur_w[i]) if i < len(cur_w) else 0.0)
            self.logger.record(f"scene/{sk}_sample_prob_last", float(prob_map.get(i, 0.0)))
            self.logger.record(f"scene/{sk}_sample_selected_last", float(i == selected_idx))
            self.logger.record(f"scene/{sk}_is_active", float(i == int(getattr(ms_env, "active_scene_idx", 0))))

            prev_val = self._finite_or_none(prev_w[i]) if i < len(prev_w) else None
            target_val = self._finite_or_none(target_w[i]) if i < len(target_w) else None
            deficit_val = self._finite_or_none(deficits[i]) if i < len(deficits) else None
            rew_val = self._finite_or_none(rew_means[i]) if i < len(rew_means) else None
            len_val = self._finite_or_none(len_means[i]) if i < len(len_means) else None
            succ_val = self._finite_or_none(succ_means[i]) if i < len(succ_means) else None
            hard_succ_val = self._finite_or_none(hard_succ_means[i]) if i < len(hard_succ_means) else None
            if prev_val is not None:
                self.logger.record(f"scene/{sk}_weight_prev", prev_val)
            if target_val is not None:
                self.logger.record(f"scene/{sk}_weight_target", target_val)
            if deficit_val is not None:
                self.logger.record(f"scene/{sk}_weight_deficit", deficit_val)
            if rew_val is not None:
                self.logger.record(f"scene/{sk}_recent_mean_rew", rew_val)
            if len_val is not None:
                self.logger.record(f"scene/{sk}_recent_mean_len", len_val)
            if succ_val is not None:
                self.logger.record(f"scene/{sk}_recent_success_rate", succ_val)
            if hard_succ_val is not None:
                self.logger.record(f"scene/{sk}_hard_success_rate", hard_succ_val)

        return True


# ============================================================
# 自适应学习率
# ============================================================
class AdaptiveLearningRateCallback(BaseCallback):
    """根据「平衡分数回落」和「approx_kl 过高」自动降低学习率。

    V3 改进:
    - balanced 指标: 各场景平均奖励的 **最小值** (min), 对最差场景退化更敏感
    - 滑动窗口 best: 只和最近 best_window 次检查中的最大值比较, 避免棘轮效应
    - 预热期: 前 warmup_steps 步不触发 LR 衰减, 等训练稳定后再监控
    """

    def __init__(
        self,
        check_freq: int = 1000,
        scene_keys: Optional[Tuple[str, ...]] = None,
        domain_keys: Optional[Tuple[str, ...]] = None,  # 已废弃，优先用 scene_keys
        min_episodes_per_domain: int = 10,
        balanced_drop_threshold: float = 1.0,
        balanced_drop_patience: int = 3,
        high_kl_threshold: float = 0.05,
        high_kl_patience: int = 2,
        decay_factor: float = 0.7,
        min_lr: float = 1e-5,
        cooldown_checks: int = 5,
        warmup_steps: int = 0,
        best_window: int = 50,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.check_freq               = max(1, int(check_freq))
        # scene_keys=None 表示使用缓冲区中所有出现的 scene_key（V12 全场景均衡）
        if scene_keys is not None:
            self.scene_keys: Optional[Tuple[str, ...]] = tuple(scene_keys)
        elif domain_keys is not None:
            self.scene_keys = tuple(domain_keys)  # 向下兼容
        else:
            self.scene_keys = None
        self.domain_keys              = self.scene_keys  # 保留旧属性名
        self.min_episodes_per_domain  = max(1, int(min_episodes_per_domain))
        self.balanced_drop_threshold  = float(max(0.0, balanced_drop_threshold))
        self.balanced_drop_patience   = max(1, int(balanced_drop_patience))
        self.high_kl_threshold        = float(max(0.0, high_kl_threshold))
        self.high_kl_patience         = max(1, int(high_kl_patience))
        self.decay_factor             = float(np.clip(decay_factor, 0.01, 0.999))
        self.min_lr                   = float(max(0.0, min_lr))
        self.cooldown_checks          = max(0, int(cooldown_checks))
        self.warmup_steps             = max(0, int(warmup_steps))
        self.best_window              = max(10, int(best_window))

        self.best_balanced = -np.inf
        self._balanced_history: list = []   # 滑动窗口 history
        self.balanced_drop_streak = 0
        self.high_kl_streak       = 0
        self.cooldown_left        = 0
        self.num_decays           = 0

    def _get_logger_value(self, key: str) -> Optional[float]:
        ntv = getattr(getattr(self, "logger", None), "name_to_value", None)
        if not isinstance(ntv, dict):
            return None
        val = ntv.get(key)
        if val is None:
            return None
        try:
            f = float(val)
        except Exception:
            return None
        return f if np.isfinite(f) else None

    def _compute_balanced(self) -> Optional[float]:
        """计算各场景平均奖励的 **最小值**（关注最差场景）。"""
        if len(self.model.ep_info_buffer) == 0:
            return None
        grouped: Dict[str, list] = {}
        for ep in self.model.ep_info_buffer:
            # 优先用 logging_key（V12），其次 scene_key，兼容旧 domain 字段
            sk = ep.get("logging_key") or ep.get("scene_key") or ep.get("domain", "unknown")
            grouped.setdefault(sk, []).append(ep)
        # 若指定了 scene_keys，只看这些；否则用缓冲区所有场景
        keys_to_check = list(self.scene_keys) if self.scene_keys else list(grouped.keys())
        vals = []
        for k in keys_to_check:
            eps = grouped.get(k, [])
            if len(eps) < self.min_episodes_per_domain:
                return None
            rs = [float(e["r"]) for e in eps if "r" in e]
            if not rs:
                return None
            vals.append(float(np.mean(rs)))
        return float(min(vals)) if vals else None

    def _current_lr(self) -> Optional[float]:
        try:
            pgs = self.model.policy.optimizer.param_groups
            return float(pgs[0]["lr"]) if pgs else None
        except Exception:
            return None

    def _set_lr(self, new_lr: float) -> None:
        self.model.learning_rate = new_lr
        self.model.lr_schedule = lambda _: new_lr
        for pg in self.model.policy.optimizer.param_groups:
            pg["lr"] = new_lr

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        balanced   = self._compute_balanced()
        approx_kl  = self._get_logger_value("train/approx_kl")
        if balanced is not None:
            # 记录最差场景分数（min）
            self.logger.record("train/balanced_min_reward_buffer", balanced)
            # 记录滑动窗口 best
            self._balanced_history.append(balanced)
            if len(self._balanced_history) > self.best_window:
                self._balanced_history = self._balanced_history[-self.best_window:]
            # V3: best 来自滑动窗口，而非全局历史
            window_best = float(max(self._balanced_history))
            self.best_balanced = window_best

        if balanced is not None:
            if balanced >= self.best_balanced:
                self.balanced_drop_streak = 0
            elif (
                self.best_balanced > -np.inf
                and balanced < (self.best_balanced - self.balanced_drop_threshold)
            ):
                self.balanced_drop_streak += 1
            else:
                self.balanced_drop_streak = 0

        if approx_kl is not None and approx_kl > self.high_kl_threshold:
            self.high_kl_streak += 1
        else:
            self.high_kl_streak = 0

        cur_lr = self._current_lr()
        if cur_lr is not None:
            self.logger.record("train/adaptive_lr_current", cur_lr)

        # V3: 预热期内不触发 LR 衰减
        if self.num_timesteps < self.warmup_steps:
            return True

        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            return True

        trigger_reasons = []
        if balanced is not None and self.balanced_drop_streak >= self.balanced_drop_patience:
            trigger_reasons.append(
                f"balanced回落({balanced:.2f}< best={self.best_balanced:.2f}-{self.balanced_drop_threshold:.2f})"
            )
        if self.high_kl_streak >= self.high_kl_patience and approx_kl is not None:
            trigger_reasons.append(f"approx_kl={approx_kl:.4f}>{self.high_kl_threshold:.4f}")

        if not trigger_reasons or cur_lr is None:
            return True

        if cur_lr <= self.min_lr * (1.0 + 1e-6):
            if self.verbose > 0:
                print(f"🧊 自动降LR触发，但已到下限: lr={cur_lr:.2e}")
            self.cooldown_left = self.cooldown_checks
            self.balanced_drop_streak = 0
            self.high_kl_streak = 0
            return True

        new_lr = max(self.min_lr, cur_lr * self.decay_factor)
        if new_lr >= cur_lr:
            return True
        self._set_lr(new_lr)
        self.num_decays += 1
        self.cooldown_left = self.cooldown_checks
        self.balanced_drop_streak = 0
        self.high_kl_streak = 0
        self.logger.record("train/adaptive_lr", new_lr)
        self.logger.record("train/adaptive_lr_num_decays", self.num_decays)
        if self.verbose > 0:
            print(
                f"📉 自动降LR: {cur_lr:.2e} -> {new_lr:.2e} | "
                + " & ".join(trigger_reasons)
            )
        return True


# ============================================================
# 训练指标落盘（JSONL）
# ============================================================
class TrainingMetricsFileLoggerCallback(BaseCallback):
    """将训练关键指标周期性写入 JSONL 文件，断线后离线分析可用。"""

    DEFAULT_PREFIXES = (
        "train/", "domain/", "coverage/", "rollout/", "eval/", "time/",
        "scene/", "short_ep/", "ctrl/", "geo/", "seg/", "smooth/", "reward_debug/",
    )

    def __init__(
        self,
        save_dir: str,
        log_freq: int = 1000,
        filename: str = "train_metrics.jsonl",
        exp_tag: Optional[str] = None,
        prefixes: Tuple[str, ...] = DEFAULT_PREFIXES,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_dir = str(save_dir)
        self.log_freq = max(1, int(log_freq))
        self.filename = str(filename)
        self.exp_tag  = exp_tag
        self.prefixes = tuple(prefixes)
        self.file_path = os.path.join(self.save_dir, self.filename)
        self._fh = None
        self._last_sig = None

    @staticmethod
    def _jsonable(v: Any) -> Any:
        if isinstance(v, (str, bool, int)):
            return v
        if isinstance(v, float):
            return float(v) if np.isfinite(v) else None
        if isinstance(v, (np.floating, np.integer, np.bool_)):
            try:
                fv = v.item()
            except Exception:
                return None
            return fv if (not isinstance(fv, float) or np.isfinite(fv)) else None
        if np.isscalar(v):
            try:
                fv = float(v)
                return fv if np.isfinite(fv) else None
            except Exception:
                return None
        return None

    def _filtered_metrics(self) -> Dict[str, Any]:
        ntv = getattr(getattr(self, "logger", None), "name_to_value", None)
        if not isinstance(ntv, dict):
            return {}
        out: Dict[str, Any] = {}
        for k, v in ntv.items():
            if not isinstance(k, str):
                continue
            if not any(k.startswith(p) for p in self.prefixes):
                continue
            val = self._jsonable(v)
            if val is None and not isinstance(v, bool):
                continue
            out[k] = val
        return out

    def _write(self, event: str = "step", force: bool = False) -> None:
        if self._fh is None:
            return
        metrics = self._filtered_metrics()
        if not metrics and not force:
            return
        sig = (
            metrics.get("time/iterations"),
            metrics.get("time/total_timesteps"),
            int(self.num_timesteps),
        )
        if (not force) and sig == self._last_sig:
            return
        rec = {
            "event":               event,
            "timestamp":           datetime.now().isoformat(),
            "unix_time":           time.time(),
            "callback_num_timesteps": int(self.num_timesteps),
            "exp_tag":             self.exp_tag,
            "metrics":             metrics,
        }
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()
        self._last_sig = sig

    def _on_training_start(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        self._fh = open(self.file_path, "a", encoding="utf-8")
        meta = {
            "event":    "start",
            "timestamp": datetime.now().isoformat(),
            "unix_time": time.time(),
            "exp_tag":   self.exp_tag,
            "pid":       os.getpid(),
            "log_freq":  self.log_freq,
        }
        self._fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
        self._fh.flush()
        if self.verbose > 0:
            print(f"📝 训练指标JSONL: {self.file_path}")

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            self._write(event="step", force=False)
        return True

    def _on_training_end(self) -> None:
        try:
            self._write(event="end", force=True)
            if self._fh is not None:
                self._fh.write(json.dumps({
                    "event": "close",
                    "timestamp": datetime.now().isoformat(),
                    "unix_time": time.time(),
                    "exp_tag": self.exp_tag,
                }, ensure_ascii=False) + "\n")
                self._fh.flush()
        finally:
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception:
                    pass
                self._fh = None


# ============================================================
# 按场景保存最佳模型（原 DomainAwareBestModelCallback）
# ============================================================
class BestModelCallback(BaseCallback):
    """按 scene_key 保存最佳模型（全局 best + 每场景 best + 可选平衡分数 best）。"""

    def __init__(
        self,
        save_path: str,
        check_freq: int = 1000,
        metric_mode: str = "per_scene_min",
        min_episodes_per_scene_for_save: int = 10,
        save_separate_per_scene_best: bool = True,
        scene_keys: Optional[Tuple[str, ...]] = None,
        domain_keys: Optional[Tuple[str, ...]] = None,  # 已废弃，优先用 scene_keys
        save_balanced_from_training_buffer: bool = False,
        verbose: int = 0,
        # 旧参数别名
        min_episodes_per_domain_for_save: Optional[int] = None,
        save_separate_per_domain_best: Optional[bool] = None,
    ):
        super().__init__(verbose)
        self.save_path   = save_path
        self.check_freq  = max(1, int(check_freq))
        self.metric_mode = metric_mode
        # 旧参数兼容
        if min_episodes_per_domain_for_save is not None:
            min_episodes_per_scene_for_save = min_episodes_per_domain_for_save
        if save_separate_per_domain_best is not None:
            save_separate_per_scene_best = save_separate_per_domain_best
        self.min_episodes_per_scene_for_save = max(1, int(min_episodes_per_scene_for_save))
        self.save_separate_per_scene_best    = bool(save_separate_per_scene_best)
        if scene_keys is not None:
            self.scene_keys: Optional[Tuple[str, ...]] = tuple(scene_keys)
        elif domain_keys is not None:
            self.scene_keys = tuple(domain_keys)  # 向下兼容
        else:
            self.scene_keys = None  # None = 动态发现所有场景
        self.save_balanced = bool(save_balanced_from_training_buffer)
        os.makedirs(save_path, exist_ok=True)

        self.best_global_mean_reward = -np.inf
        self.best_per_scene_reward: Dict[str, float] = {}
        self.best_balanced_score = -np.inf

    def _save_pair(self, stem: str) -> Tuple[str, str]:
        zip_path = os.path.join(self.save_path, stem)
        self.model.save(zip_path)
        pth_path = os.path.join(self.save_path, f"{stem}_policy.pth")
        torch.save(self.model.policy.state_dict(), pth_path)
        return zip_path + ".zip", pth_path

    def _split_ep_info(self):
        episodes = list(self.model.ep_info_buffer)
        grouped: Dict[str, List[dict]] = {}
        for ep in episodes:
            sk = ep.get("logging_key") or ep.get("scene_key") or ep.get("domain", "unknown")
            grouped.setdefault(sk, []).append(ep)
        return episodes, grouped

    def _scene_means(self, grouped: Dict[str, List[dict]]):
        means: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for sk, eps in grouped.items():
            rs = [float(e["r"]) for e in eps if "r" in e]
            means[sk]  = float(np.mean(rs)) if rs else float("nan")
            counts[sk] = len(eps)
        return means, counts

    # 旧方法别名
    _domain_means = _scene_means

    def _calc_balanced(self, means: Dict[str, float], counts: Dict[str, int]) -> Optional[float]:
        # 若指定了 scene_keys，只看这些；否则用所有场景
        keys_to_check = list(self.scene_keys) if self.scene_keys else list(counts.keys())
        vals = []
        for k in keys_to_check:
            if counts.get(k, 0) < self.min_episodes_per_scene_for_save:
                return None
            mr = means.get(k, float("nan"))
            if np.isnan(mr):
                return None
            vals.append(mr)
        if not vals:
            return None
        if self.metric_mode == "global_mean":
            return float(np.mean(vals))
        if self.metric_mode == "harmonic_mean":
            if any(v <= 0 for v in vals):
                return float(min(vals))
            return float(len(vals) / sum(1.0 / v for v in vals))
        return float(min(vals))  # per_scene_min

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True
        episodes, grouped = self._split_ep_info()
        if not episodes:
            return True

        global_mean = float(np.mean([ep["r"] for ep in episodes if "r" in ep]))
        means, counts = self._scene_means(grouped)
        balanced = self._calc_balanced(means, counts)

        if self.verbose > 0:
            print(f"\n📊 {self.num_timesteps}步 | 全局平均奖励: {global_mean:.1f}")
            by_scene = ", ".join(
                [f"{sk}: n={counts[sk]} rew={means[sk]:.1f}" for sk in sorted(counts)]
            )
            if by_scene:
                print(f"   按场景: {by_scene}")

        if global_mean > self.best_global_mean_reward:
            self.best_global_mean_reward = global_mean
            self._save_pair("best_model_global")
            self._save_pair("best_model")   # 兼容旧路径
            if self.verbose > 0:
                print(f"⭐ 新全局最佳: {global_mean:.2f}")

        if self.save_separate_per_scene_best:
            for sk, mr in means.items():
                if counts.get(sk, 0) <= 0 or np.isnan(mr):
                    continue
                if mr > self.best_per_scene_reward.get(sk, -np.inf):
                    self.best_per_scene_reward[sk] = mr
                    z, p = self._save_pair(f"best_model_{sk}")
                    if self.verbose > 0:
                        print(f"⭐ 新场景 [{sk}] 最佳: {mr:.2f} -> {z}")

        if self.save_balanced and balanced is not None and balanced > self.best_balanced_score:
            self.best_balanced_score = balanced
            z, p = self._save_pair("best_model_balanced")
            if self.verbose > 0:
                print(f"⭐ 新平衡最佳(缓冲区): {balanced:.2f}")
        return True


# 向下兼容别名
DomainAwareBestModelCallback = BestModelCallback


# ============================================================
# 崩溃检测 + 自动回滚到最佳 checkpoint
# ============================================================
class CrashRecoveryCallback(BaseCallback):
    """
    监控每个场景的 ep_len 滑动平均值，当检测到严重退化时自动回滚模型到最佳 checkpoint。

    检测逻辑:
      1. 维护每个场景的 ep_len 峰值 (peak) 和最近 N 局的滑动平均 (rolling)
      2. 当 rolling < peak * crash_ratio 且 peak 超过 min_peak_len 时判定为崩溃
      3. 回滚到 save_dir 下最近的常规 checkpoint (v13_*_steps.zip)
      4. 回滚后进入冷却期，冷却期间不检测

    回滚行为:
      - 加载 checkpoint 的 policy state_dict（不重建 model，保留 env/buffer）
      - 重置 peak 为当前水平，避免循环回滚
    """

    def __init__(
        self,
        save_dir: str,
        check_freq: int = 2000,
        rolling_window: int = 30,
        crash_ratio: float = 0.25,
        min_peak_len: float = 80.0,
        cooldown_steps: int = 50000,
        min_warmup_steps: int = 30000,
        checkpoint_prefix: str = "v13",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_dir = str(save_dir)
        self.check_freq = max(1, int(check_freq))
        self.rolling_window = max(5, int(rolling_window))
        self.crash_ratio = float(crash_ratio)
        self.min_peak_len = float(min_peak_len)
        self.cooldown_steps = int(cooldown_steps)
        self.min_warmup_steps = int(min_warmup_steps)
        self.checkpoint_prefix = str(checkpoint_prefix)

        # per-scene tracking
        self._scene_ep_lens: Dict[str, deque] = {}   # rolling window of ep_lens
        self._scene_peaks: Dict[str, float] = {}      # historical peak of rolling avg
        self._last_rollback_step: int = 0
        self._rollback_count: int = 0
        self._started_step: Optional[int] = None

    def _on_training_start(self):
        self._started_step = self.num_timesteps

    def _find_latest_checkpoint(self) -> Optional[str]:
        """查找 save_dir 下最新的 v13_*_steps.zip checkpoint。"""
        import re
        pat = re.compile(rf"^{re.escape(self.checkpoint_prefix)}_(\d+)_steps\.zip$")
        best_steps = -1
        best_path: Optional[str] = None
        if not os.path.isdir(self.save_dir):
            return None
        for fn in os.listdir(self.save_dir):
            m = pat.match(fn)
            if m:
                steps = int(m.group(1))
                if steps > best_steps:
                    best_steps = steps
                    best_path = os.path.join(self.save_dir, fn)
        return best_path

    def _rollback(self, crash_scene: str, rolling_avg: float, peak: float):
        """回滚 policy 到最近的 checkpoint。"""
        ckpt = self._find_latest_checkpoint()
        if ckpt is None:
            print(f"🔄 [{crash_scene}] 崩溃检测触发但无可用 checkpoint，跳过回滚")
            return False

        try:
            # 只加载 policy state_dict，不重建整个 model
            from sb3_contrib import RecurrentPPO
            tmp_model = RecurrentPPO.load(ckpt)
            old_state = tmp_model.policy.state_dict()
            self.model.policy.load_state_dict(old_state)
            # 同步 optimizer 的参数引用
            for pg in self.model.policy.optimizer.param_groups:
                pg["params"] = [p for p in self.model.policy.parameters() if p.requires_grad]
            del tmp_model
            self._rollback_count += 1
            self._last_rollback_step = self.num_timesteps
            # 重置所有场景 peak 为当前 rolling（避免回滚后立刻再次触发）
            for sk in self._scene_peaks:
                if sk in self._scene_ep_lens and len(self._scene_ep_lens[sk]) > 0:
                    self._scene_peaks[sk] = float(np.mean(self._scene_ep_lens[sk]))
                else:
                    self._scene_peaks[sk] = 0.0
            print(
                f"🔄 崩溃回滚 #{self._rollback_count} [{crash_scene}]: "
                f"rolling={rolling_avg:.0f} < peak={peak:.0f}×{self.crash_ratio}={peak*self.crash_ratio:.0f} "
                f"→ 恢复 {os.path.basename(ckpt)}"
            )
            return True
        except Exception as e:
            print(f"🔄 [{crash_scene}] 回滚失败: {type(e).__name__}: {e}")
            return False

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        # 预热期不检测
        if self._started_step is not None:
            trained = self.num_timesteps - self._started_step
            if trained < self.min_warmup_steps:
                return True

        # 冷却期不检测
        if self.num_timesteps - self._last_rollback_step < self.cooldown_steps:
            return True

        # 收集 per-scene episode lengths
        if len(self.model.ep_info_buffer) == 0:
            return True

        grouped: Dict[str, List[float]] = {}
        for ep in self.model.ep_info_buffer:
            sk = ep.get("logging_key") or ep.get("scene_key") or ep.get("domain", "unknown")
            if "l" in ep:
                grouped.setdefault(sk, []).append(float(ep["l"]))

        for sk, lens in grouped.items():
            if sk not in self._scene_ep_lens:
                self._scene_ep_lens[sk] = deque(maxlen=self.rolling_window)
                self._scene_peaks[sk] = 0.0

            # 更新 rolling window（用 buffer 中该场景最近的均值作为一个数据点）
            if lens:
                self._scene_ep_lens[sk].append(float(np.mean(lens)))

            if len(self._scene_ep_lens[sk]) < 3:
                continue

            rolling_avg = float(np.mean(self._scene_ep_lens[sk]))

            # 更新 peak
            if rolling_avg > self._scene_peaks[sk]:
                self._scene_peaks[sk] = rolling_avg

            peak = self._scene_peaks[sk]

            # 崩溃检测
            if peak >= self.min_peak_len and rolling_avg < peak * self.crash_ratio:
                if self._rollback(sk, rolling_avg, peak):
                    return True  # 回滚后继续训练

        return True

    def summary(self) -> Dict[str, Any]:
        return {
            "rollback_count": self._rollback_count,
            "scene_peaks": dict(self._scene_peaks),
            "last_rollback_step": self._last_rollback_step,
        }


# ============================================================
# 短命 episode 日志（供后期分析特定地图卡死/即时 done 问题）
# ============================================================
class ShortEpisodeLoggerCallback(BaseCallback):
    """
    记录 episode 步数 < threshold 的早终止事件到 JSONL 文件。
    每条记录包含：timestamp、num_timesteps、scene_key、episode_len、
    episode_reward、termination_reason。
    同时在控制台打印警告，便于实时发现问题地图。
    """

    def __init__(
        self,
        save_dir: str,
        threshold: int = 15,
        filename: str = "short_episodes.jsonl",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_dir  = str(save_dir)
        self.threshold = int(max(1, threshold))
        self.file_path = os.path.join(self.save_dir, filename)
        self._fh: Optional[Any] = None
        self._total_short = 0
        # 按场景统计（供 _on_training_end 摘要）
        self._scene_short_counts: Dict[str, int] = {}

    def _on_training_start(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        self._fh = open(self.file_path, "a", encoding="utf-8")
        self._fh.write(json.dumps({
            "event":     "start",
            "timestamp": datetime.now().isoformat(),
            "threshold": self.threshold,
            "pid":       os.getpid(),
        }, ensure_ascii=False) + "\n")
        self._fh.flush()
        if self.verbose > 0:
            print(f"📋 短命 episode 日志 (< {self.threshold} 步): {self.file_path}")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict):
                continue
            ep = info.get("episode")
            if not isinstance(ep, dict):
                continue
            ep_len = ep.get("l", 9999)
            try:
                ep_len = float(ep_len)
            except Exception:
                continue
            if ep_len >= self.threshold:
                continue

            self._total_short += 1
            # scene_key 优先从 episode dict，其次从 info 顶层
            scene_key = ep.get("scene_key") or info.get("scene_key", "unknown")
            term_reason = ep.get("termination_reason") or info.get("termination_reason", "unknown")
            ep_reward = ep.get("r", float("nan"))

            self._scene_short_counts[scene_key] = self._scene_short_counts.get(scene_key, 0) + 1

            rec = {
                "event":             "short_episode",
                "timestamp":         datetime.now().isoformat(),
                "num_timesteps":     int(self.num_timesteps),
                "episode_len":       ep_len,
                "episode_reward":    float(ep_reward) if np.isfinite(float(ep_reward)) else None,
                "scene_key":         str(scene_key),
                "termination_reason": str(term_reason),
                "total_short":       self._total_short,
            }
            if self._fh is not None:
                try:
                    self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    self._fh.flush()
                except Exception:
                    pass

            if self.verbose > 0:
                print(
                    f"⚠️  短命ep #{self._total_short}: scene={scene_key}, "
                    f"len={ep_len:.0f}<{self.threshold}, "
                    f"rew={ep_reward:.1f}, reason={term_reason} "
                    f"(@{self.num_timesteps}步)"
                )

            # 写入 TensorBoard
            self.logger.record("short_ep/total_count", self._total_short)
            for sk, cnt in self._scene_short_counts.items():
                self.logger.record(f"short_ep/{sk}_count", cnt)

        return True

    def _on_training_end(self) -> None:
        if self._fh is not None:
            try:
                self._fh.write(json.dumps({
                    "event":              "end",
                    "timestamp":          datetime.now().isoformat(),
                    "total_short":        self._total_short,
                    "by_scene":           self._scene_short_counts,
                }, ensure_ascii=False) + "\n")
                self._fh.flush()
            except Exception:
                pass
            finally:
                try:
                    self._fh.close()
                except Exception:
                    pass
                self._fh = None

        if self._total_short > 0 and self.verbose > 0:
            print(f"\n📋 短命 episode 汇总 (总计 {self._total_short} 次):")
            for sk, cnt in sorted(self._scene_short_counts.items(), key=lambda x: -x[1]):
                print(f"   {sk}: {cnt} 次")


# ============================================================
# tqdm 进度条  📊 552000步 | 全局平均奖励: 46.2
# ============================================================
class TqdmProgressCallback(BaseCallback):
    """tqdm 进度条，显示当前步数、全局平均奖励、各场景奖励。"""

    def __init__(self, total_timesteps: int, update_freq: int = 2048, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.update_freq = max(1, update_freq)
        self.pbar = None
        self._last_update = 0

    def _on_training_start(self) -> None:
        try:
            from tqdm import tqdm
        except ImportError:
            print("⚠️  tqdm 未安装，进度条不可用")
            return
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc="📊 训练",
            unit="步",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )

    def _on_step(self) -> bool:
        if self.pbar is None:
            return True
        current = self.num_timesteps
        delta = current - self._last_update
        if delta >= self.update_freq or current >= self.total_timesteps:
            self.pbar.update(delta)
            self._last_update = current

            # ------ 收集全局 + 各场景奖励 ------
            postfix = {}
            if len(self.model.ep_info_buffer) > 0:
                all_rewards = [float(e["r"]) for e in self.model.ep_info_buffer if "r" in e]
                if all_rewards:
                    postfix["全局奖励"] = f"{np.mean(all_rewards):.1f}"

                # 按场景分组
                grouped: Dict[str, list] = {}
                for ep in self.model.ep_info_buffer:
                    sk = ep.get("logging_key") or ep.get("scene_key") or ep.get("domain", "?")
                    grouped.setdefault(sk, []).append(float(ep.get("r", 0)))
                for sk in sorted(grouped.keys()):
                    rs = grouped[sk]
                    postfix[sk] = f"{np.mean(rs):.1f}"

            if postfix:
                self.pbar.set_postfix(postfix, refresh=False)
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.update(self.num_timesteps - self._last_update)
            self.pbar.close()
            self.pbar = None


# ============================================================
# 步数平衡采样：动态调整障碍物概率以补充缺失的障碍物步数
# ============================================================
class ObstacleStepsBalancingCallback(BaseCallback):
    """
    根据步数补充障碍物回合。

    逻辑：
    - 目标障碍物步数 = 总步数 × (1 - obstacle_free_prob)
    - 跟踪：实际带障碍物的步数
    - 补充：如果缺失，动态降低 obstacle_free_prob 来增加后续障碍物出现概率
    """

    def __init__(
        self,
        check_freq: int = 1000,
        target_obstacle_ratio: float = 0.5,  # 50% 的步数应该带障碍物
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.check_freq = max(1, int(check_freq))
        self.target_obstacle_ratio = float(np.clip(target_obstacle_ratio, 0.0, 1.0))

        # Per-scene tracking
        self._scene_total_steps: Dict[str, int] = {}
        self._scene_obstacle_steps: Dict[str, int] = {}
        self._scene_obstacle_free_prob: Dict[str, float] = {}
        self._last_check_step = 0

    def _on_training_start(self) -> None:
        # 初始化场景
        for scene_key in ['ws', 'gt']:
            self._scene_total_steps[scene_key] = 0
            self._scene_obstacle_steps[scene_key] = 0
            # 从env获取初始obstacle_free_prob
            if hasattr(self.model.env, 'envs'):
                # VecEnv
                for env in self.model.env.envs:
                    if hasattr(env, 'obstacle_free_prob'):
                        self._scene_obstacle_free_prob[scene_key] = env.obstacle_free_prob
                        break
            elif hasattr(self.model.env, 'obstacle_free_prob'):
                self._scene_obstacle_free_prob[scene_key] = self.model.env.obstacle_free_prob

    def _on_step(self) -> bool:
        # 从ep_info_buffer中提取信息
        if len(self.model.ep_info_buffer) > 0 and self.num_timesteps - self._last_check_step >= self.check_freq:
            self._last_check_step = self.num_timesteps

            # 处理最近的episodes
            for ep in self.model.ep_info_buffer:
                scene_key = ep.get('logging_key') or 'unknown'
                ep_len = ep.get('l', 0)  # episode length

                if scene_key in self._scene_total_steps:
                    self._scene_total_steps[scene_key] += ep_len

                    # 检查是否有障碍物（通过碰撞或near_collision信息推断）
                    # 简单启发：如果reward中有碰撞或near_collision惩罚，说明有障碍物
                    has_obstacle = 'ep_term_collision' in ep or 'ep_r_near_collision' in ep
                    if has_obstacle:
                        self._scene_obstacle_steps[scene_key] += ep_len

            # 根据缺失程度调整obstacle_free_prob
            self._adjust_obstacle_probs()

        return True

    def _adjust_obstacle_probs(self) -> None:
        """根据缺失的障碍物步数，动态调整obstacle_free_prob"""

        for scene_key in ['ws', 'gt']:
            total = self._scene_total_steps.get(scene_key, 0)
            obstacle = self._scene_obstacle_steps.get(scene_key, 0)

            if total < 100:  # 数据太少，不调整
                continue

            # 计算目标和实际的障碍物步数
            target_obstacle_steps = total * self.target_obstacle_ratio
            actual_ratio = obstacle / total if total > 0 else 0

            # 如果缺失过多，降低obstacle_free_prob来补充
            deficit_ratio = 1.0 - (obstacle / target_obstacle_steps) if target_obstacle_steps > 0 else 0
            deficit_ratio = np.clip(deficit_ratio, 0.0, 1.0)

            # 新的obstacle_free_prob = 旧的 * (1 - deficit_ratio)
            # 这样当缺失多时，障碍物概率会上升
            old_prob = self._scene_obstacle_free_prob.get(scene_key, 0.5)
            new_prob = old_prob * (1.0 - deficit_ratio * 0.1)  # 每次最多调整10%
            new_prob = np.clip(new_prob, 0.0, 0.9)  # 保持在合理范围

            self._scene_obstacle_free_prob[scene_key] = new_prob

            # 应用到environment
            if hasattr(self.model.env, 'envs'):
                for env in self.model.env.envs:
                    if hasattr(env, 'obstacle_free_prob'):
                        if scene_key == 'ws' and hasattr(env, 'ws_obstacle_free_prob'):
                            env.ws_obstacle_free_prob = new_prob
                        elif scene_key == 'gt':
                            env.obstacle_free_prob = new_prob
            elif hasattr(self.model.env, f'{scene_key}_obstacle_free_prob'):
                setattr(self.model.env, f'{scene_key}_obstacle_free_prob', new_prob)

            if deficit_ratio > 0.05 and self.verbose >= 1:
                print(f"[障碍物平衡] {scene_key}: 缺失{deficit_ratio:.1%}, "
                      f"obstacle_free_prob: {old_prob:.3f} → {new_prob:.3f}")


# ============================================================
# Step 预算补偿（基于步数统计的动态调整）
# ============================================================
class StepBudgetCompensationCallback(BaseCallback):
    """
    基于步数窗口统计，自动补偿学习机会不均衡。

    工作原理：
    1. 环境跟踪每个scene的"有障碍"与"无障碍"的步数
    2. 每50个episode评估一次，检查是否达到目标障碍步数比例
    3. 如果缺失，动态降低obstacle_free_prob来补充；如果过度，提高
    """

    def __init__(
        self,
        curriculum_phases: Dict[str, Dict[str, Any]],
        window_episode_count: int = 50,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.curriculum_phases = curriculum_phases
        self.window_episode_count = window_episode_count

        # 跟踪补偿历史（用于日志）
        self.compensation_history = {}  # {scene_key: [(step, stats_dict), ...]}

    def _on_step(self) -> bool:
        """每一步检查是否需要评估和补偿"""

        # 只在 DummyVecEnv 或 SubprocVecEnv 上工作
        if not hasattr(self.model.env, 'envs'):
            return True

        # 获取当前阶段的配置
        if not hasattr(self.model.env, '_curriculum_phase'):
            return True

        phase_name = self.model.env._curriculum_phase
        phase_config = self.curriculum_phases.get(phase_name, {})

        # 获取目标障碍比例
        target_ratios = phase_config.get("obstacle_target_ratios", {})
        if not target_ratios:
            return True

        # 检查所有环境是否有步数统计
        for env in self.model.env.envs:
            if not hasattr(env, '_step_budget_stats'):
                continue

            stats = env._step_budget_stats

            # 检查每个scene是否达到评估窗口
            for scene_key in ['ws', 'gt']:
                if scene_key not in target_ratios or scene_key not in stats:
                    continue

                ep_count = stats[scene_key]['episode_count']
                window_episodes = self.window_episode_count

                # 检查是否达到评估点（但只在有足够数据时）
                if ep_count > 0 and ep_count % window_episodes == 0 and ep_count % (window_episodes + 1) != 0:
                    self._evaluate_and_compensate(env, scene_key, phase_config, target_ratios[scene_key])

        return True

    def _evaluate_and_compensate(
        self,
        env,
        scene_key: str,
        phase_config: Dict[str, Any],
        target_ratio: float,
    ) -> None:
        """评估窗口内的步数分布，计算缺陷，调整障碍概率"""

        stats = env._step_budget_stats[scene_key]
        with_obs_steps = stats["window_with_obs_steps"]
        without_obs_steps = stats["window_without_obs_steps"]
        total_steps = with_obs_steps + without_obs_steps

        if total_steps < 10:  # 数据太少，不调整
            return

        # 计算实际比例和缺陷
        actual_ratio = with_obs_steps / total_steps if total_steps > 0 else 0
        target_steps = int(total_steps * target_ratio)
        deficit = target_steps - with_obs_steps  # 负数表示过度

        # 只在缺陷显著时调整
        if abs(deficit) < total_steps * 0.05:  # 缺陷 < 5% 不调整
            return

        # 获取当前的obstacle_free_prob
        if scene_key == 'ws' and hasattr(env, 'ws_obstacle_free_prob'):
            current_free_prob = env.ws_obstacle_free_prob or 0.5
            is_ws = True
        else:
            current_free_prob = env.obstacle_free_prob
            is_ws = False

        max_compensation_ratio = phase_config.get("max_compensation_ratio", 0.25)

        if deficit > 0:
            # 缺少有障碍步数 → 降低 obstacle_free_prob
            reduction = min(deficit / total_steps, max_compensation_ratio)
            new_free_prob = max(0.0, current_free_prob - reduction)
            direction = "↓"
        else:
            # 过度有障碍 → 提高 obstacle_free_prob
            addition = min(-deficit / total_steps, max_compensation_ratio)
            new_free_prob = min(1.0, current_free_prob + addition)
            direction = "↑"

        # 应用调整
        if is_ws:
            env.ws_obstacle_free_prob = new_free_prob
        else:
            env.obstacle_free_prob = new_free_prob

        # 记录到历史
        if scene_key not in self.compensation_history:
            self.compensation_history[scene_key] = []

        comp_record = {
            "step": self.num_timesteps,
            "episode_count": stats["episode_count"],
            "with_obs_steps": with_obs_steps,
            "without_obs_steps": without_obs_steps,
            "target_ratio": target_ratio,
            "actual_ratio": actual_ratio,
            "deficit": deficit,
            "old_free_prob": current_free_prob,
            "new_free_prob": new_free_prob,
        }
        self.compensation_history[scene_key].append(comp_record)

        # 打印日志
        if self.verbose >= 1:
            log_msg = (
                f"\n📊 [Step预算补偿] {scene_key.upper()} @ 步{self.num_timesteps}\n"
                f"   目标障碍率: {target_ratio:.0%} | 实际: {actual_ratio:.0%} | "
                f"缺陷: {deficit:+d}步\n"
                f"   obstacle_free_prob: {current_free_prob:.3f} {direction} {new_free_prob:.3f} "
                f"(±{abs(new_free_prob - current_free_prob):.3f})"
            )
            print(log_msg)
