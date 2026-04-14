"""
module/utils.py
通用工具函数、常量，供 V12 所有子模块共享。
"""

import os
import re
import sys
import json
import math
import random
import importlib.util
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ============================================================
# 配置加载
# ============================================================
def load_config(myconfig: Optional[str] = None):
    """从 Python 文件加载大写变量CONFIG对象（若 donkeycar 可用则复用，否则自建）。"""
    try:
        import donkeycar as dk
        try:
            from donkeycar.utils import load_config as _dc_load
            return _dc_load(myconfig)
        except (ImportError, AttributeError):
            pass
    except ImportError:
        pass

    if myconfig is None:
        return None

    spec = importlib.util.spec_from_file_location("myconfig", myconfig)
    if spec is None:
        return None

    cfg_module = importlib.util.module_from_spec(spec)
    sys.modules["myconfig"] = cfg_module
    try:
        spec.loader.exec_module(cfg_module)
    except Exception:
        return None

    class Config:
        pass

    cfg = Config()
    for attr in dir(cfg_module):
        if attr.isupper():
            setattr(cfg, attr, getattr(cfg_module, attr))
    return cfg


# ============================================================
# 场景域映射（9 maps）
# ============================================================
ENV_DOMAIN_MAP: Dict[str, str] = {
    "donkey-waveshare-v0": "ws",
    "donkey-generated-track-v0": "gt",
    "donkey-generated-roads-v0": "gt",
    "donkey-warehouse-v0": "gt",
    "donkey-mountain-track-v0": "gt",
    "donkey-minimonaco-track-v0": "gt",
    "donkey-roboracingleague-track-v0": "rrl",
    "donkey-avc-sparkfun-v0": "gt",
    "donkey-warren-track-v0": "gt",
    "donkey-circuit-launch-track-v0": "gt",
}


def _get_domain_for_env(env_id: str) -> str:
    """根据 env_id 返回域标识（ws/gt，默认 ws）。"""
    return ENV_DOMAIN_MAP.get(env_id, "ws")


# ============================================================
# Monitor 记录的 episode 末尾 info 字段
# ============================================================
MONITOR_INFO_KEYS: Tuple[str, ...] = (
    "domain",
    "scene_key",           # V12: 场景标识符（用于按场景统计，替代 domain）
    "logging_key",         # V12: 日志记录用的缩短场景名称（防止 TensorBoard 键过长）
    "termination_reason",  # 终止原因（normal/collision/stuck/persistent_offtrack）
    "mask_coverage",
    "ep_r_survival",
    "ep_r_speed",
    "ep_r_progress",
    "ep_r_cte",
    "ep_r_center",
    "ep_r_heading",
    "ep_r_speed_ref",
    "ep_r_time",
    "ep_r_collision",
    "ep_r_near_offtrack",
    "ep_r_near_collision",
    "ep_r_lap",
    "ep_r_lap_raw",
    "ep_r_overtake",
    "ep_overtake_count",
    "ep_soft_lap_count",
    "ep_r_smooth",
    "ep_r_jerk",
    "ep_r_mismatch",
    "ep_r_sat",
    "ep_r_total",
    # 终止原因 one-hot
    "ep_term_collision",
    "ep_term_stuck",
    "ep_term_offtrack",
    "ep_term_env_done",
    "ep_term_normal",
    # episode 级几何/惩罚诊断
    "ep_cte_abs_p50",
    "ep_cte_abs_p90",
    "ep_cte_abs_p99",
    "ep_cte_over_in_rate",
    "ep_cte_over_out_rate",
    "ep_rate_limit_hit_rate",
    "ep_steer_clip_hit_rate",
    "ep_throttle_high_penalty_hit_rate",
    "ep_offtrack_counter_max",
    "ep_stuck_counter_max",
)


# ============================================================
# 随机种子
# ============================================================
def _seed_everything(seed: Optional[int]) -> None:
    """统一 Python / NumPy / PyTorch 随机性。"""
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _safe_seed_env(env, seed: Optional[int], label: str = "env") -> None:
    """best-effort 给 VecEnv/Gym env 设置 seed，失败不阻塞训练。"""
    if seed is None:
        return
    try:
        if hasattr(env, "seed"):
            env.seed(int(seed))
    except Exception as e:
        print(f"⚠️  {label}.seed({seed}) 失败: {type(e).__name__}: {e}")
    try:
        if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
            env.action_space.seed(int(seed))
    except Exception:
        pass
    try:
        if hasattr(env, "observation_space") and hasattr(env.observation_space, "seed"):
            env.observation_space.seed(int(seed))
    except Exception:
        pass


# ============================================================
# RecurrentPPO LSTM 评估
# ============================================================
def _evaluate_recurrent_policy_on_vec_env(
    model,
    vec_env,
    n_eval_episodes: int = 5,
    deterministic: bool = True,
) -> Tuple[float, float]:
    """
    显式传递 LSTM state / episode_start 的评估循环。
    兼容 sb3_contrib.RecurrentPPO。
    """
    n_envs = int(getattr(vec_env, "num_envs", 1))
    obs = vec_env.reset()
    lstm_state = None
    episode_start = np.ones((n_envs,), dtype=bool)

    episode_rewards: List[float] = []
    running_rewards = np.zeros((n_envs,), dtype=np.float32)

    while len(episode_rewards) < int(n_eval_episodes):
        action, lstm_state = model.predict(
            obs,
            state=lstm_state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        obs, rewards, dones, infos = vec_env.step(action)
        rewards = np.array(rewards, dtype=np.float32).reshape(-1)
        dones = np.array(dones, dtype=bool).reshape(-1)
        running_rewards += rewards

        for i in range(n_envs):
            if dones[i]:
                episode_rewards.append(float(running_rewards[i]))
                running_rewards[i] = 0.0
                if len(episode_rewards) >= int(n_eval_episodes):
                    break

        episode_start = dones.astype(bool)

    if not episode_rewards:
        return float("nan"), float("nan")
    return float(np.mean(episode_rewards)), float(np.std(episode_rewards))


# ============================================================
# Checkpoint 查找
# ============================================================
def _find_latest_checkpoint(save_dir: str, name_prefix: str = "v12") -> Optional[str]:
    """查找 `{name_prefix}_N_steps.zip` 中步数最大的文件。"""
    if not os.path.isdir(save_dir):
        return None
    pat = re.compile(rf"^{re.escape(name_prefix)}_(\d+)_steps\.zip$")
    best_steps = -1
    best_path: Optional[str] = None
    for fn in os.listdir(save_dir):
        m = pat.match(fn)
        if not m:
            continue
        try:
            steps = int(m.group(1))
        except Exception:
            continue
        if steps > best_steps:
            best_steps = steps
            best_path = os.path.join(save_dir, fn)
    return best_path


# ============================================================
# Float 工具
# ============================================================
def _wrap_pi(x: float) -> float:
    while x > math.pi:
        x -= 2.0 * math.pi
    while x < -math.pi:
        x += 2.0 * math.pi
    return x


def _clip_float(x: float, lo: float, hi: float) -> float:
    return float(np.clip(float(x), float(lo), float(hi)))


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Best-effort float conversion with fallback."""
    try:
        return float(v)
    except Exception:
        return float(default)


def _apply_global_seeds(seed: Optional[int]) -> int:
    """Apply global RNG seeds and return normalized integer seed."""
    s = int(_safe_float(seed, 42))
    _seed_everything(s)
    return s


def _write_json(path: str, payload: Any) -> None:
    """Write JSON to path (mkdir parent), ignore failures with warning."""
    if not path:
        return
    try:
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"🧾 JSON写入: {abs_path}")
    except Exception as e:
        print(f"⚠️ 写入JSON失败({path}): {e}")
