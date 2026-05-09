#!/usr/bin/env python3
"""
DonkeyCar PPO V16
- 2-domain (ws/gt) recurrent PPO
- 6-channel semantic observation + 12D state
- obstacle-aware runtime for dual-domain avoidance training
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
_repo_root = str(REPO_ROOT)
while _repo_root in sys.path:
    sys.path.remove(_repo_root)
sys.path.insert(0, _repo_root)

import gym
import gym_donkeycar
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from sb3_contrib import RecurrentPPO
except Exception:
    RecurrentPPO = None

from module.actor import FiLMFeatureExtractor
from module.callbacks import (
    AdaptiveLearningRateCallback,
    BestModelCallback,
    CrashRecoveryCallback,
    PerSceneStatsCallback,
    PTHExportCallback,
    SceneSchedulerLoggingCallback,
    ShortEpisodeLoggerCallback,
    StepBudgetCompensationCallback,
    TqdmProgressCallback,
    TrainingMetricsFileLoggerCallback,
)
from module.multi_scene_env import MultiInputObsWrapper, MultiSceneEnvV16, _build_state_v16
from module.track import TrackGeometryManager
from module.utils import (
    _find_latest_checkpoint,
    _safe_seed_env,
    _seed_everything,
    load_config,
)


DEFAULT_TRACK_DIR = os.environ.get("MYSIM_TRACK_DIR", str(REPO_ROOT / "track_profiles"))
DEFAULT_MYCONFIG = os.environ.get("MYSIM_MYCONFIG", str(REPO_ROOT / "myconfig.py"))
WS_FINISH_OBSTACLE_PROGRESS_RATIO_V16 = 0.5  # 改为对面(0.08太近，learner一出生就撞)

# WS 在正式避障阶段统一复用同一套“静态 + 边缘 + 中后段 progress”放置逻辑，
# 避免 avoid/pid 阶段各自维护一份后逐渐分叉。
WS_SHARED_AVOID_PLACEMENT_V16: Dict[str, Any] = {
    "ws_obstacle_modes": ["static"],
    "ws_obstacle_progress_min": 0.20,
    "ws_obstacle_progress_max": 0.80,
    "ws_obstacle_fixed_lateral_ratio": None,
}

WS_REWARD_OVERRIDES_V16: Dict[str, float] = {
    # WS 赛道更窄，避障时经常需要主动吃一点边界/出界代价。
    # 这里让 WS 更偏向“先避障、再回正”，同时抑制
    # “高进度奖励 + 短回合碰撞也不亏”的激进策略：
    # - 降低 near/offtrack 与中心线约束，允许避障时多吃一点边线代价
    # - 保留前进驱动，但更早、更明确地惩罚贴车/碰撞风险
    "cte_norm_scale": 0.50,
    "w_near_offtrack": 0.28,
    "near_offtrack_start_ratio": 0.68,
    "offtrack_penalty_base": 3.0,
    "collision_penalty_base": 18.0,
    "w_near_collision": 0.50,
    "near_collision_start_ratio": 0.58,
    "safe_follow_bonus_scale": 0.035,
    "w_center": 0.020,
    "w_heading": 0.012,
    "progress_reward_scale": 50.0,
    "survival_reward_scale": 0.34,
}

SCENE_SPECS: Dict[str, Dict[str, Any]] = {
    "donkey-waveshare-v0": {
        "scene_key": "waveshare",
        "logging_key": "ws",
        "level_name": "waveshare",
        "track_file": "manual_width_waveshare.json",
        "domain": "ws",
        "reward_overrides": dict(WS_REWARD_OVERRIDES_V16),
    },
    "donkey-generated-track-v0": {
        "scene_key": "generated_track",
        "logging_key": "gt",
        "level_name": "generated_track",
        "track_file": "manual_width_generated_track.json",
        "domain": "gt",
    },
}

DEFAULT_ENV_IDS: List[str] = [
    "donkey-waveshare-v0",
    "donkey-generated-track-v0",
]

CURRICULUM_PHASES: Dict[str, Dict[str, Any]] = {
    # 合并版 warmup：
    # - GT 保持固定半程单静态障碍
    # - WS 先从“终点处固定单静态障碍”起步
    "warmup": {
        "scene_weights": [0.5, 0.5],
        "enable_dynamic_scene_weights": False,
        "enable_step_balance_sampling": True,
        "obstacle_enabled": True,
        "obstacle_count": 1,
        "obstacle_free_prob": 0.75,
        "obstacle_modes": ["static"],
        "ws_obstacle_free_prob": 1.0,  # warmup禁用WS障碍，集中学基础驾驶
        "obstacle_fixed_progress_ratio": 0.50,
        "obstacle_fixed_lateral_ratio": 0.50,
        "ws_obstacle_modes": ["static"],
        "ws_obstacle_fixed_progress_ratio": WS_FINISH_OBSTACLE_PROGRESS_RATIO_V16,
        "ws_obstacle_fixed_lateral_ratio": None,  # 随机边缘：最大离边
        "obstacle_lateral_choices": [0.0, 1.0],  # 赛道几何边缘（最激进，自动被安全距离约束）
        "obstacle_randomize_non_lane_pid_yaw": False,  # 车头沿赛道progress切向（不随机）
        "obstacle_spawn_ahead_min_m": 6.0,
        "obstacle_spawn_ahead_max_m": 12.0,
        "obstacle_min_agent_planar_dist_m": 2.0,
        "obstacle_min_agent_arc_dist_m": 5.5,
        "collision_penalty_base": 6.0,
        "offtrack_penalty_base": 5.0,
        "w_near_collision": 0.10,
        "near_collision_start_ratio": 0.80,
    },
    # Phase 1A:
    # - GT: 固定单静态障碍热身
    # - WS: 单静态障碍在 70%-90% progress处
    "warmup_a": {
        "scene_weights": [0.5, 0.5],
        "enable_dynamic_scene_weights": False,
        "enable_step_balance_sampling": True,
        "obstacle_enabled": True,
        "obstacle_count": 1,
        "obstacle_free_prob": 0.50,
        "obstacle_modes": ["static"],
        "ws_obstacle_free_prob": 0.50,
        "obstacle_fixed_progress_ratio": 0.50,
        "obstacle_fixed_lateral_ratio": 0.50,
        "ws_obstacle_modes": ["static"],
        "ws_obstacle_progress_min": 0.15,  # WS障碍范围：15%-85%
        "ws_obstacle_progress_max": 0.85,
        "ws_obstacle_fixed_lateral_ratio": None,   # 修复：随机边缘放置，WS太窄无法绕开中央障碍
        "obstacle_lateral_choices": [0.0, 1.0],  # GT用边缘障碍
        "obstacle_randomize_non_lane_pid_yaw": False,  # 车头沿赛道方向
        "obstacle_spawn_ahead_min_m": 6.0,
        "obstacle_spawn_ahead_max_m": 12.0,
        "obstacle_min_agent_planar_dist_m": 2.0,
        "obstacle_min_agent_arc_dist_m": 5.5,
        "collision_penalty_base": 6.0,
        "offtrack_penalty_base": 5.0,
        "w_near_collision": 0.08,
        "near_collision_start_ratio": 0.82,
    },
    # Phase 1B:
    # - GT: 固定单静态障碍继续热身
    # - WS: 单障碍随机位置，沿赛道前后晃动
    "warmup_b": {
        "scene_weights": [0.5, 0.5],
        "enable_dynamic_scene_weights": False,
        "enable_step_balance_sampling": True,
        "obstacle_enabled": True,
        "obstacle_count": 1,
        "obstacle_free_prob": 0.70,
        "obstacle_modes": ["static"],
        "obstacle_fixed_progress_ratio": 0.50,
        "obstacle_fixed_lateral_ratio": 0.50,
        "ws_obstacle_modes": ["jitter"],
        "ws_obstacle_fixed_progress_ratio": None,
        "ws_obstacle_fixed_lateral_ratio": None,
        "obstacle_randomize_non_lane_pid_yaw": False,
        "obstacle_spawn_ahead_min_m": 6.0,
        "obstacle_spawn_ahead_max_m": 12.0,
        "obstacle_min_agent_planar_dist_m": 2.0,
        "obstacle_min_agent_arc_dist_m": 5.5,
        "collision_penalty_base": 6.0,
        "offtrack_penalty_base": 5.0,
        "w_near_collision": 0.12,
        "near_collision_start_ratio": 0.78,
    },
    # Phase 2: 进入正式静态避障
    "avoid_static": {
        "scene_weights": [0.5, 0.5],
        "enable_dynamic_scene_weights": False,  # ✅ 禁用动态权重 - 确保WS获得公平的采样机会
        "enable_step_balance_sampling": True,
        "obstacle_enabled": True,
        "obstacle_count": 2,  # GT 使用两个障碍物；WS 仍由运行时限制为单障碍
        "obstacle_free_prob": 0.20,
        "obstacle_modes": ["static"],
        "obstacle_progress_min": 0.20,  # GT障碍范围：20%-80%
        "obstacle_progress_max": 0.80,
        "ws_obstacle_free_prob": 0.20,
        **WS_SHARED_AVOID_PLACEMENT_V16,
        "obstacle_lateral_choices": [0.0, 1.0],  # GT用边缘障碍
        "obstacle_randomize_non_lane_pid_yaw": False,  # 车头沿赛道progress切向（不随机）
        "obstacle_spawn_ahead_min_m": 5.0,
        "obstacle_spawn_ahead_max_m": 12.0,
        "obstacle_min_agent_planar_dist_m": 1.8,
        "obstacle_min_agent_arc_dist_m": 4.5,
        "collision_penalty_base": 7.0,
        "offtrack_penalty_base": 5.0,
        "w_near_collision": 0.16,
        "near_collision_start_ratio": 0.72,
        # 步数预算补偿配置
        "obstacle_target_ratios": {"ws": 0.80, "gt": 0.80},  # 统一：80% 有障碍 / 20% 无障碍
        "window_episode_count": 50,  # 每50个episode评估一次
        "max_compensation_ratio": 0.25,  # 单次最多调整25%
    },
    # Phase 3: GT 混入轻微动态障碍；WS 保持静态边缘障碍
    # 修复：WS赛道仅0.55m宽，障碍半径0.25m，lateral=0.5时物理上无法绕行。
    # 改为static+边缘放置（lateral_ratio=None随机边缘），并放宽free_prob和惩罚参数。
    "avoid_mixed": {
        "scene_weights": [0.5, 0.5],
        "enable_dynamic_scene_weights": True,
        "enable_step_balance_sampling": True,
        "obstacle_enabled": True,
        "obstacle_count": 2,  # GT 使用两个障碍物；WS 仍由运行时限制为单障碍
        "obstacle_free_prob": 0.20,
        "obstacle_modes": ["static", "jitter", "nudge"],
        "obstacle_progress_min": 0.20,  # GT障碍范围：20%-80%
        "obstacle_progress_max": 0.80,
        "ws_obstacle_free_prob": 0.20,          # 统一：20% 无障碍比例
        **WS_SHARED_AVOID_PLACEMENT_V16,
        "obstacle_randomize_non_lane_pid_yaw": False,  # 车头沿赛道切向，不随机转向
        "obstacle_spawn_ahead_min_m": 4.5,
        "obstacle_spawn_ahead_max_m": 12.0,
        "obstacle_min_agent_planar_dist_m": 1.6,
        "obstacle_min_agent_arc_dist_m": 4.0,
        "collision_penalty_base": 7.0,           # 从8.0降回7.0：WS窄道已足够难
        "offtrack_penalty_base": 5.0,
        "w_near_collision": 0.20,
        "near_collision_start_ratio": 0.72,      # 从0.68放宽回0.72：给更多接近余量
        # 步数预算补偿配置
        "obstacle_target_ratios": {"ws": 0.80, "gt": 0.80},  # 统一：80% 有障碍 / 20% 无障碍
        "window_episode_count": 50,
        "max_compensation_ratio": 0.25,
    },
    # Phase 4: 引入 lane-pid 动态车，先保守速度
    "lane_pid_intro": {
        "scene_weights": [0.5, 0.5],
        "enable_dynamic_scene_weights": True,
        "enable_step_balance_sampling": True,
        "obstacle_enabled": True,
        "obstacle_count": 1,  # PID阶段GT仅保留1台lane_pid障碍；WS仍由运行时限制为单障碍
        "obstacle_free_prob": 0.20,
        "obstacle_modes": ["lane_pid"],
        "obstacle_progress_min": 0.20,  # GT障碍范围：20%-80%
        "obstacle_progress_max": 0.80,
        "ws_obstacle_free_prob": 0.20,
        **WS_SHARED_AVOID_PLACEMENT_V16,
        "obstacle_randomize_non_lane_pid_yaw": False,  # 车头沿赛道切向，不随机转向
        "obstacle_spawn_ahead_min_m": 4.0,
        "obstacle_spawn_ahead_max_m": 11.0,
        "obstacle_min_agent_planar_dist_m": 1.5,
        "obstacle_min_agent_arc_dist_m": 3.8,
        "obstacle_lane_pid_speed_gt": 0.30,
        "obstacle_lane_pid_speed_ws": 0.30,
        "collision_penalty_base": 8.0,
        "offtrack_penalty_base": 5.0,
        "w_near_collision": 0.24,
        "near_collision_start_ratio": 0.62,
        # 步数预算补偿配置
        "obstacle_target_ratios": {"ws": 0.80, "gt": 0.80},  # 统一：80% 有障碍 / 20% 无障碍
        "window_episode_count": 50,
        "max_compensation_ratio": 0.25,
    },
    # Phase 5: 完整障碍课程
    "lane_pid_full": {
        "scene_weights": [0.5, 0.5],
        "enable_dynamic_scene_weights": True,
        "enable_step_balance_sampling": True,
        "obstacle_enabled": True,
        "obstacle_count": 2,  # GT: 双 lane_pid 车；WS 仍由运行时限制为单障碍
        "obstacle_free_prob": 0.10,
        "obstacle_modes": ["lane_pid"],
        "obstacle_fixed_progress_ratio": None,
        "obstacle_fixed_progress_gap": None,
        "obstacle_fixed_progress_gap_min": 0.40,  # GT: 第二台与第一台间隔 40%-50% progress
        "obstacle_fixed_progress_gap_max": 0.50,
        "obstacle_fixed_lateral_ratio": 0.50,    # GT: 双车都走赛道中线
        "obstacle_progress_min": 0.15,  # GT 第一台范围：15%-40%
        "obstacle_progress_max": 0.40,
        "ws_obstacle_free_prob": 0.10,
        **WS_SHARED_AVOID_PLACEMENT_V16,
        "obstacle_randomize_non_lane_pid_yaw": False,  # 车头沿赛道切向，不随机转向
        "obstacle_spawn_ahead_min_m": 4.0,
        "obstacle_spawn_ahead_max_m": 10.0,
        "obstacle_min_agent_planar_dist_m": 1.5,
        "obstacle_min_agent_arc_dist_m": 3.5,
        "obstacle_lane_pid_speed_gt": 0.70,
        "obstacle_lane_pid_speed_ws": 0.70,
        "collision_penalty_base": 8.0,
        "offtrack_penalty_base": 5.0,
        "w_near_collision": 0.24,
        "near_collision_start_ratio": 0.60,
        # 步数预算补偿配置
        "obstacle_target_ratios": {"ws": 0.90, "gt": 0.90},  # 90% 有障碍 / 10% 无障碍
        "window_episode_count": 50,
        "max_compensation_ratio": 0.25,
    },
}

CURRICULUM_PHASE_ALIASES: Dict[str, str] = {
    "stage1": "warmup",
    "stage1a": "warmup_a",
    "stage1b": "warmup_b",
    "stage2": "avoid_static",
    "stage3": "avoid_mixed",
    "stage4": "lane_pid_intro",
    "stage5": "lane_pid_full",
}

AUTO_CURRICULUM_STAGES: Tuple[Dict[str, Any], ...] = (
    {
        "stage_name": "warmup",
        "phase": "warmup",
        "required_logging_keys": ["ws", "gt"],
        "recent_episodes": 10,
        "min_success_episodes": 2,
        "min_soft_laps": 1.0,
        "max_collision_rate_by_key": {"ws": 0.75},
        "min_stage_timesteps": 900_000,
        "max_stage_timesteps": 1_500_000,
    },
    {
        "stage_name": "warmup_a",
        "phase": "warmup_a",
        "required_logging_keys": ["ws", "gt"],
        "recent_episodes": 10,
        "min_success_episodes": 2,
        "min_soft_laps": 1.5,
        "max_collision_rate_by_key": {"ws": 0.65},
        "min_stage_timesteps": 300_000,
        "max_stage_timesteps": 900_000,
    },
    {
        "stage_name": "avoid_static",
        "phase": "avoid_static",
        "required_logging_keys": ["ws", "gt"],
        "recent_episodes": 10,
        "min_success_episodes": 2,
        "min_soft_laps": 2.0,
        "max_collision_rate_by_key": {"ws": 0.55},
        "min_stage_timesteps": 300_000,
        "max_stage_timesteps": 1_500_000,
    },
    {
        "stage_name": "avoid_mixed",
        "phase": "avoid_mixed",
        "required_logging_keys": ["ws", "gt"],
        "recent_episodes": 10,
        "min_success_episodes": 2,
        "min_soft_laps": 2.0,
        "max_collision_rate_by_key": {"ws": 0.55, "gt": 0.60},
        "min_stage_timesteps": 300_000,
        "max_stage_timesteps": 1_500_000,
    },
    {
        "stage_name": "lane_pid_intro",
        "phase": "lane_pid_intro",
        "required_logging_keys": ["ws", "gt"],
        "recent_episodes": 10,
        "min_success_episodes": 2,
        "min_soft_laps": 2.0,
        "max_collision_rate_by_key": {"ws": 0.55, "gt": 0.50},
        "min_stage_timesteps": 300_000,
        "max_stage_timesteps": 1_500_000,
    },
    {
        "stage_name": "lane_pid_full",
        "phase": "lane_pid_full",
    },
)

TRAIN_V16_DEFAULTS: Dict[str, Any] = {
    "scene_weights": None,
    "enable_dynamic_scene_weights": True,
    "enable_step_balance_sampling": True,
    "obstacle_enabled": True,
    "obstacle_count": 2,
    "obstacle_free_prob": 0.15,
    "obstacle_modes": None,
    "ws_obstacle_free_prob": None,
    "obstacle_spawn_ahead_min_m": 3.5,
    "obstacle_spawn_ahead_max_m": 14.0,
    "obstacle_min_agent_planar_dist_m": 1.5,
    "obstacle_min_agent_arc_dist_m": 3.5,
    "obstacle_fixed_progress_ratio": None,
    "obstacle_fixed_progress_gap": None,
    "obstacle_fixed_progress_gap_min": None,
    "obstacle_fixed_progress_gap_max": None,
    "obstacle_progress_min": None,
    "obstacle_progress_max": None,
    "obstacle_fixed_lateral_ratio": None,
    "gt_obstacle_start_exclusion_half_width_m": None,
    "ws_obstacle_modes": None,
    "ws_obstacle_fixed_progress_ratio": None,
    "ws_obstacle_fixed_lateral_ratio": None,
    "obstacle_randomize_non_lane_pid_yaw": True,
    "obstacle_lane_pid_speed_gt": 0.85,
    "obstacle_lane_pid_speed_ws": 0.70,
    "collision_penalty_base": 8.0,
    "offtrack_penalty_base": 5.0,
    "w_near_collision": 0.24,
    "near_collision_start_ratio": 0.65,
    "overtake_success_bonus": 2.5,
}


def _clone_curriculum_value(value: Any) -> Any:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def _resolve_curriculum_phase(curriculum_phase: Optional[str]) -> Optional[str]:
    name = str(curriculum_phase or "").strip().lower()
    if not name or name in ("none", "off", "default"):
        return None
    name = CURRICULUM_PHASE_ALIASES.get(name, name)
    if name not in CURRICULUM_PHASES:
        available = ", ".join(sorted(CURRICULUM_PHASES.keys()))
        raise KeyError(f"Unknown curriculum_phase={curriculum_phase}. Available: {available}")
    return name


def _apply_curriculum_phase(
    curriculum_phase: Optional[str],
    values: Dict[str, Any],
) -> Tuple[Optional[str], Dict[str, Any]]:
    resolved = _resolve_curriculum_phase(curriculum_phase)
    if resolved is None:
        return None, {}

    applied: Dict[str, Any] = {}
    phase_overrides = CURRICULUM_PHASES[resolved]
    for key, target_value in phase_overrides.items():
        current_value = values.get(key)
        default_value = TRAIN_V16_DEFAULTS.get(key, None)
        if current_value == default_value:
            cloned = _clone_curriculum_value(target_value)
            values[key] = cloned
            applied[key] = _clone_curriculum_value(cloned)
    return resolved, applied


class CurriculumWindowAdvanceCallback(BaseCallback):
    """Advance a curriculum stage when recent per-scene windows satisfy all gates."""

    def __init__(
        self,
        stage_name: str,
        required_logging_keys: List[str],
        min_stage_timesteps: int,
        recent_episodes: int = 20,
        min_success_episodes: int = 12,
        min_soft_laps: float = 2.0,
        min_episode_len: Optional[int] = None,
        min_progress_ratio_forward_sum: Optional[float] = None,
        min_reward: Optional[float] = None,
        max_episode_speed_mean: Optional[float] = None,
        max_episode_speed_max: Optional[float] = None,
        require_no_stuck_for_success: bool = True,
        max_collision_rate_by_key: Optional[Dict[str, float]] = None,
        max_stage_timesteps: Optional[int] = None,
        save_dir: Optional[str] = None,
        exp_tag: Optional[str] = None,
        filename: str = "curriculum_window.jsonl",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.stage_name = str(stage_name)
        self.required_logging_keys = [str(x) for x in required_logging_keys]
        self.min_stage_timesteps = max(0, int(min_stage_timesteps))
        self.recent_episodes = max(1, int(recent_episodes))
        self.min_success_episodes = max(1, int(min_success_episodes))
        self.min_soft_laps = float(max(0.0, min_soft_laps))
        self.min_episode_len = (
            None if min_episode_len is None else max(1, int(min_episode_len))
        )
        self.min_progress_ratio_forward_sum = (
            None
            if min_progress_ratio_forward_sum is None
            else max(0.0, float(min_progress_ratio_forward_sum))
        )
        self.min_reward = None if min_reward is None else float(min_reward)
        self.max_episode_speed_mean = (
            None if max_episode_speed_mean is None else max(0.0, float(max_episode_speed_mean))
        )
        self.max_episode_speed_max = (
            None if max_episode_speed_max is None else max(0.0, float(max_episode_speed_max))
        )
        self.require_no_stuck_for_success = bool(require_no_stuck_for_success)
        self.max_collision_rate_by_key: Dict[str, float] = {}
        for key, limit in dict(max_collision_rate_by_key or {}).items():
            key_s = str(key)
            if key_s not in self.required_logging_keys:
                continue
            try:
                limit_f = float(limit)
            except Exception:
                continue
            if np.isfinite(limit_f):
                self.max_collision_rate_by_key[key_s] = float(np.clip(limit_f, 0.0, 1.0))
        self.max_stage_timesteps = (
            None if max_stage_timesteps is None else max(1, int(max_stage_timesteps))
        )
        self.save_dir = (None if not save_dir else str(save_dir))
        self.exp_tag = exp_tag
        self.filename = str(filename)
        self.recent_soft_laps: Dict[str, deque] = {
            key: deque(maxlen=self.recent_episodes) for key in self.required_logging_keys
        }
        self.recent_gate_success: Dict[str, deque] = {
            key: deque(maxlen=self.recent_episodes) for key in self.required_logging_keys
        }
        self.recent_term_collisions: Dict[str, deque] = {
            key: deque(maxlen=self.recent_episodes) for key in self.required_logging_keys
        }
        self.triggered = False
        self.stop_reason = ""
        self.stop_num_timesteps = 0
        self.stop_stage_timesteps = 0
        self._start_num_timesteps = 0
        self._fh = None

    def _recover_resume_state(
        self,
        current_num_timesteps: int,
    ) -> Tuple[Optional[int], Dict[str, deque], Dict[str, deque], Dict[str, deque]]:
        if not self.save_dir:
            return None, {}, {}, {}
        log_path = os.path.join(self.save_dir, self.filename)
        if not os.path.isfile(log_path):
            return None, {}, {}, {}

        recovered_soft_laps: Dict[str, deque] = {
            key: deque(maxlen=self.recent_episodes) for key in self.required_logging_keys
        }
        recovered_gate_success: Dict[str, deque] = {
            key: deque(maxlen=self.recent_episodes) for key in self.required_logging_keys
        }
        recovered_term_collisions: Dict[str, deque] = {
            key: deque(maxlen=self.recent_episodes) for key in self.required_logging_keys
        }
        stage_start_num_timesteps: Optional[int] = None

        try:
            with open(log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue

                    if str(rec.get("stage_name", "")) != self.stage_name:
                        continue
                    if self.exp_tag is not None and str(rec.get("exp_tag", "")) != str(self.exp_tag):
                        continue

                    try:
                        rec_num_timesteps = int(rec.get("num_timesteps", 0) or 0)
                        rec_stage_timesteps = int(rec.get("stage_timesteps", 0) or 0)
                    except Exception:
                        continue
                    if rec_num_timesteps > current_num_timesteps:
                        continue

                    rec_stage_start_num_timesteps = max(
                        0,
                        int(rec_num_timesteps) - max(0, int(rec_stage_timesteps)),
                    )
                    event = str(rec.get("event", ""))
                    if event == "stop":
                        stage_start_num_timesteps = None
                        recovered_soft_laps = {
                            key: deque(maxlen=self.recent_episodes)
                            for key in self.required_logging_keys
                        }
                        recovered_gate_success = {
                            key: deque(maxlen=self.recent_episodes)
                            for key in self.required_logging_keys
                        }
                        recovered_term_collisions = {
                            key: deque(maxlen=self.recent_episodes)
                            for key in self.required_logging_keys
                        }
                        continue

                    if stage_start_num_timesteps is None:
                        stage_start_num_timesteps = int(rec_stage_start_num_timesteps)

                    if event != "episode":
                        continue
                    logging_key = str(
                        rec.get("logging_key")
                        or rec.get("scene_key")
                        or rec.get("domain")
                        or ""
                    )
                    if logging_key not in recovered_soft_laps:
                        continue
                    try:
                        soft_laps = float(rec.get("soft_laps", 0.0) or 0.0)
                    except Exception:
                        soft_laps = 0.0
                    if not np.isfinite(soft_laps):
                        soft_laps = 0.0
                    try:
                        term_collision = float(rec.get("ep_term_collision", 0.0) or 0.0)
                    except Exception:
                        term_collision = 0.0
                    if not np.isfinite(term_collision):
                        term_collision = 0.0
                    try:
                        gate_success = float(rec.get("gate_success", np.nan))
                    except Exception:
                        gate_success = np.nan
                    if not np.isfinite(gate_success):
                        gate_success, _ = self._record_gate_success(
                            rec,
                            soft_laps=soft_laps,
                            term_collision=term_collision,
                        )
                    recovered_soft_laps[logging_key].append(max(0.0, float(soft_laps)))
                    recovered_gate_success[logging_key].append(
                        1.0 if float(gate_success) >= 0.5 else 0.0
                    )
                    recovered_term_collisions[logging_key].append(
                        1.0 if float(term_collision) >= 0.5 else 0.0
                    )
        except Exception:
            return None, {}, {}, {}

        return (
            stage_start_num_timesteps,
            recovered_soft_laps,
            recovered_gate_success,
            recovered_term_collisions,
        )

    def _window_snapshot(self) -> Dict[str, Any]:
        return {
            key: {
                "success_count": int(self._window_success_count(key)),
                "window_size": int(len(self.recent_soft_laps.get(key, []))),
                "values": [float(x) for x in self.recent_soft_laps.get(key, [])],
                "gate_success_values": [
                    float(x) for x in self.recent_gate_success.get(key, [])
                ],
                "term_collision_rate": self._window_collision_rate(key),
            }
            for key in self.required_logging_keys
        }

    def _write_event(self, event: str, payload: Dict[str, Any]) -> None:
        if self._fh is None:
            return
        rec = {
            "event": str(event),
            "timestamp": datetime.now().isoformat(),
            "unix_time": time.time(),
            "exp_tag": self.exp_tag,
            "stage_name": self.stage_name,
            "num_timesteps": int(self.num_timesteps),
            "stage_timesteps": int(self._stage_timesteps()),
        }
        rec.update(payload)
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    def _on_training_start(self) -> None:
        current_num_timesteps = int(getattr(self.model, "num_timesteps", 0))
        self._start_num_timesteps = int(current_num_timesteps)
        (
            recovered_stage_start,
            recovered_soft_laps,
            recovered_gate_success,
            recovered_term_collisions,
        ) = (
            self._recover_resume_state(current_num_timesteps)
        )
        if recovered_stage_start is not None and int(recovered_stage_start) <= int(current_num_timesteps):
            self._start_num_timesteps = int(recovered_stage_start)
            for key in self.required_logging_keys:
                self.recent_soft_laps[key].clear()
                self.recent_soft_laps[key].extend(list(recovered_soft_laps.get(key, [])))
                self.recent_gate_success[key].clear()
                self.recent_gate_success[key].extend(
                    list(recovered_gate_success.get(key, []))
                )
                self.recent_term_collisions[key].clear()
                self.recent_term_collisions[key].extend(
                    list(recovered_term_collisions.get(key, []))
                )
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            self._fh = open(os.path.join(self.save_dir, self.filename), "a", encoding="utf-8")
            self._write_event(
                "start",
                {
                    "required_logging_keys": list(self.required_logging_keys),
                    "min_stage_timesteps": int(self.min_stage_timesteps),
                    "max_stage_timesteps": (
                        None if self.max_stage_timesteps is None else int(self.max_stage_timesteps)
                    ),
                    "recent_episodes": int(self.recent_episodes),
                    "min_success_episodes": int(self.min_success_episodes),
                    "min_soft_laps": float(self.min_soft_laps),
                    "min_episode_len": self.min_episode_len,
                    "min_progress_ratio_forward_sum": self.min_progress_ratio_forward_sum,
                    "min_reward": self.min_reward,
                    "max_episode_speed_mean": self.max_episode_speed_mean,
                    "max_episode_speed_max": self.max_episode_speed_max,
                    "require_no_stuck_for_success": bool(self.require_no_stuck_for_success),
                    "max_collision_rate_by_key": dict(self.max_collision_rate_by_key),
                    "recovered_stage_start_num_timesteps": int(self._start_num_timesteps),
                    "recovered_stage_timesteps": int(self._stage_timesteps()),
                    "recovered_window_sizes": {
                        key: int(len(self.recent_soft_laps.get(key, [])))
                        for key in self.required_logging_keys
                    },
                },
            )
        if self.verbose > 0:
            joined = "+".join(self.required_logging_keys)
            max_txt = (
                f", fallback@{self.max_stage_timesteps} stage steps"
                if self.max_stage_timesteps is not None else ""
            )
            collision_txt = ""
            if self.max_collision_rate_by_key:
                collision_parts = [
                    f"{key} term_collision_rate <= {limit:.2f}"
                    for key, limit in self.max_collision_rate_by_key.items()
                ]
                collision_txt = ", " + ", ".join(collision_parts)
            perf_parts = []
            if self.min_episode_len is not None:
                perf_parts.append(f"len>={self.min_episode_len}")
            if self.min_progress_ratio_forward_sum is not None:
                perf_parts.append(f"progress>={self.min_progress_ratio_forward_sum:.3f}")
            if self.min_reward is not None:
                perf_parts.append(f"reward>={self.min_reward:.2f}")
            if self.max_episode_speed_mean is not None:
                perf_parts.append(f"speed_mean<={self.max_episode_speed_mean:.3f}")
            if self.max_episode_speed_max is not None:
                perf_parts.append(f"speed_max<={self.max_episode_speed_max:.3f}")
            performance_txt = ""
            if perf_parts:
                performance_txt = " or performance(" + ", ".join(perf_parts) + ")"
            inherit_txt = ""
            if self._start_num_timesteps < int(current_num_timesteps):
                inherit_txt = (
                    f", resumed stage_steps={self._stage_timesteps()}"
                    f" (stage_start_total={self._start_num_timesteps})"
                )
            print(
                f"🎓 阶段晋级门控[{self.stage_name}]: "
                f"after {self.min_stage_timesteps} stage steps, "
                f"{joined} recent {self.recent_episodes} eps need "
                f">= {self.min_success_episodes} eps with soft_lap >= {self.min_soft_laps:.1f}"
                f"{performance_txt}{collision_txt}{max_txt}{inherit_txt}"
            )

    def _stage_timesteps(self) -> int:
        return max(0, int(self.num_timesteps) - int(self._start_num_timesteps))

    @staticmethod
    def _extract_soft_laps(info: Dict[str, Any]) -> float:
        if "ep_soft_lap_count" in info:
            try:
                soft_laps = float(info.get("ep_soft_lap_count", 0.0) or 0.0)
            except Exception:
                soft_laps = np.nan
            if np.isfinite(soft_laps):
                return max(0.0, soft_laps)

        try:
            lap_raw = float(info.get("ep_r_lap_raw", 0.0) or 0.0)
        except Exception:
            lap_raw = 0.0
        if not np.isfinite(lap_raw):
            lap_raw = 0.0
        return max(0.0, lap_raw / 6.0)

    def _window_success_count(self, logging_key: str) -> int:
        dq = self.recent_gate_success.get(logging_key)
        if dq is None:
            return 0
        return int(sum(float(v) >= 0.5 for v in dq))

    @staticmethod
    def _finite_float(value: Any, default: float = 0.0) -> float:
        try:
            out = float(value)
        except Exception:
            return float(default)
        if not np.isfinite(out):
            return float(default)
        return float(out)

    @staticmethod
    def _finite_int(value: Any, default: int = 0) -> int:
        try:
            out = int(value)
        except Exception:
            return int(default)
        return int(out)

    def _record_gate_success(
        self,
        record: Dict[str, Any],
        soft_laps: float,
        term_collision: float,
    ) -> Tuple[float, str]:
        if float(term_collision) >= 0.5:
            return 0.0, "collision"

        term_stuck = self._finite_float(record.get("ep_term_stuck", 0.0), 0.0)
        if self.require_no_stuck_for_success and term_stuck >= 0.5:
            return 0.0, "stuck"

        if float(soft_laps) + 1e-9 >= self.min_soft_laps:
            return 1.0, "soft_lap"

        has_performance_gate = any(
            value is not None
            for value in (
                self.min_episode_len,
                self.min_progress_ratio_forward_sum,
                self.min_reward,
                self.max_episode_speed_mean,
                self.max_episode_speed_max,
            )
        )
        if not has_performance_gate:
            return 0.0, "soft_lap_below_threshold"

        episode_len = self._finite_int(record.get("episode_len", 0), 0)
        if self.min_episode_len is not None and episode_len < self.min_episode_len:
            return 0.0, "episode_len_below_threshold"

        progress = self._finite_float(
            record.get("ep_progress_ratio_forward_sum", 0.0),
            0.0,
        )
        if (
            self.min_progress_ratio_forward_sum is not None
            and progress + 1e-9 < self.min_progress_ratio_forward_sum
        ):
            return 0.0, "progress_below_threshold"

        reward = self._finite_float(record.get("episode_reward", 0.0), 0.0)
        if self.min_reward is not None and reward + 1e-9 < self.min_reward:
            return 0.0, "reward_below_threshold"

        speed_mean = self._finite_float(record.get("ep_speed_mean", 0.0), 0.0)
        if (
            self.max_episode_speed_mean is not None
            and speed_mean > self.max_episode_speed_mean + 1e-9
        ):
            return 0.0, "speed_mean_above_real_envelope"

        speed_max = self._finite_float(record.get("ep_speed_max", 0.0), 0.0)
        if (
            self.max_episode_speed_max is not None
            and speed_max > self.max_episode_speed_max + 1e-9
        ):
            return 0.0, "speed_max_above_real_envelope"

        return 1.0, "performance"

    @staticmethod
    def _extract_term_collision(info: Dict[str, Any]) -> float:
        if "ep_term_collision" in info:
            try:
                collision = float(info.get("ep_term_collision", 0.0) or 0.0)
            except Exception:
                collision = np.nan
            if np.isfinite(collision):
                return 1.0 if collision >= 0.5 else 0.0

        reason = str(info.get("termination_reason", "") or "").strip().lower()
        tokens = {tok for tok in reason.split("+") if tok}
        return 1.0 if "collision" in tokens else 0.0

    def _window_collision_rate(self, logging_key: str) -> Optional[float]:
        dq = self.recent_term_collisions.get(logging_key)
        if dq is None or len(dq) < 1:
            return None
        return float(sum(float(v) for v in dq) / max(1, len(dq)))

    def _collision_windows_ready(self) -> bool:
        for key, limit in self.max_collision_rate_by_key.items():
            dq = self.recent_term_collisions.get(key)
            if dq is None or len(dq) < self.recent_episodes:
                return False
            rate = self._window_collision_rate(key)
            if rate is None or rate > float(limit) + 1e-9:
                return False
        return True

    def _all_recent_windows_ready(self) -> bool:
        for key in self.required_logging_keys:
            dq = self.recent_soft_laps.get(key)
            if dq is None or len(dq) < self.recent_episodes:
                return False
            if self._window_success_count(key) < self.min_success_episodes:
                return False
        return self._collision_windows_ready()

    def summary(self) -> Dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "triggered": bool(self.triggered),
            "stop_reason": str(self.stop_reason),
            "min_stage_timesteps": int(self.min_stage_timesteps),
            "max_stage_timesteps": (
                None if self.max_stage_timesteps is None else int(self.max_stage_timesteps)
            ),
            "recent_episodes": int(self.recent_episodes),
            "min_success_episodes": int(self.min_success_episodes),
            "min_soft_laps": float(self.min_soft_laps),
            "min_episode_len": self.min_episode_len,
            "min_progress_ratio_forward_sum": self.min_progress_ratio_forward_sum,
            "min_reward": self.min_reward,
            "max_episode_speed_mean": self.max_episode_speed_mean,
            "max_episode_speed_max": self.max_episode_speed_max,
            "require_no_stuck_for_success": bool(self.require_no_stuck_for_success),
            "max_collision_rate_by_key": dict(self.max_collision_rate_by_key),
            "stop_num_timesteps": int(self.stop_num_timesteps),
            "stop_stage_timesteps": int(self.stop_stage_timesteps),
            "window_success_counts": {
                key: int(self._window_success_count(key))
                for key in self.required_logging_keys
            },
            "window_term_collision_rates": {
                key: self._window_collision_rate(key)
                for key in self.required_logging_keys
            },
            "recent_soft_laps": {
                key: [float(x) for x in self.recent_soft_laps.get(key, [])]
                for key in self.required_logging_keys
            },
            "recent_gate_success": {
                key: [float(x) for x in self.recent_gate_success.get(key, [])]
                for key in self.required_logging_keys
            },
            "recent_term_collisions": {
                key: [float(x) for x in self.recent_term_collisions.get(key, [])]
                for key in self.required_logging_keys
            },
        }

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict):
                continue
            if "episode" not in info:
                continue
            logging_key = str(
                info.get("logging_key")
                or info.get("scene_key")
                or info.get("domain")
                or ""
            )
            if logging_key not in self.recent_soft_laps:
                continue
            soft_laps = self._extract_soft_laps(info)
            term_collision = self._extract_term_collision(info)
            ep = info.get("episode", {})
            gate_record = {
                "episode_reward": ep.get("r", 0.0),
                "episode_len": ep.get("l", 0),
                "ep_progress_ratio_forward_sum": info.get("ep_progress_ratio_forward_sum", 0.0),
                "ep_term_stuck": info.get("ep_term_stuck", 0.0),
                "ep_speed_mean": info.get("ep_speed_mean", 0.0),
                "ep_speed_max": info.get("ep_speed_max", 0.0),
            }
            gate_success, gate_success_reason = self._record_gate_success(
                gate_record,
                soft_laps=soft_laps,
                term_collision=term_collision,
            )
            self.recent_soft_laps[logging_key].append(soft_laps)
            self.recent_gate_success[logging_key].append(gate_success)
            self.recent_term_collisions[logging_key].append(term_collision)
            episode_payload = {
                "logging_key": logging_key,
                "scene_key": str(info.get("scene_key", logging_key)),
                "domain": str(info.get("domain", "")),
                "episode_reward": (
                    None if "r" not in ep else float(ep.get("r", 0.0))
                ),
                "episode_len": (
                    None if "l" not in ep else int(ep.get("l", 0))
                ),
                "soft_laps": float(soft_laps),
                "gate_success": float(gate_success),
                "gate_success_reason": str(gate_success_reason),
                "termination_reason": str(info.get("termination_reason", "")),
                "ep_term_collision": float(info.get("ep_term_collision", 0.0) or 0.0),
                "ep_term_offtrack": float(info.get("ep_term_offtrack", 0.0) or 0.0),
                "ep_term_stuck": float(info.get("ep_term_stuck", 0.0) or 0.0),
                "ep_r_progress": float(info.get("ep_r_progress", 0.0) or 0.0),
                "ep_r_near_collision": float(info.get("ep_r_near_collision", 0.0) or 0.0),
                "ep_r_near_offtrack": float(info.get("ep_r_near_offtrack", 0.0) or 0.0),
                "ep_r_collision": float(info.get("ep_r_collision", 0.0) or 0.0),
                "ep_r_total": float(info.get("ep_r_total", 0.0) or 0.0),
                "obstacle_episode_modes": str(info.get("obstacle_episode_modes", "")),
                "obstacle_primary_mode": str(info.get("obstacle_primary_mode", "")),
                "ep_obstacle_has_lane_pid": float(info.get("ep_obstacle_has_lane_pid", 0.0) or 0.0),
                "ep_obstacle_primary_is_lane_pid": float(
                    info.get("ep_obstacle_primary_is_lane_pid", 0.0) or 0.0
                ),
                "ep_obstacle_static_count": float(info.get("ep_obstacle_static_count", 0.0) or 0.0),
                "ep_obstacle_jitter_count": float(info.get("ep_obstacle_jitter_count", 0.0) or 0.0),
                "ep_obstacle_nudge_count": float(info.get("ep_obstacle_nudge_count", 0.0) or 0.0),
                "ep_obstacle_lane_pid_count": float(info.get("ep_obstacle_lane_pid_count", 0.0) or 0.0),
                "ep_lane_pid_debug_steps": float(info.get("ep_lane_pid_debug_steps", 0.0) or 0.0),
                "ep_lane_pid_target_speed_mean": float(
                    info.get("ep_lane_pid_target_speed_mean", 0.0) or 0.0
                ),
                "ep_lane_pid_speed_mean": float(info.get("ep_lane_pid_speed_mean", 0.0) or 0.0),
                "ep_lane_pid_speed_error_abs_mean": float(
                    info.get("ep_lane_pid_speed_error_abs_mean", 0.0) or 0.0
                ),
                "ep_lane_pid_effective_lookahead_mean": float(
                    info.get("ep_lane_pid_effective_lookahead_mean", 0.0) or 0.0
                ),
                "ep_lane_pid_local_forward_mean": float(
                    info.get("ep_lane_pid_local_forward_mean", 0.0) or 0.0
                ),
                "ep_lane_pid_local_left_abs_mean": float(
                    info.get("ep_lane_pid_local_left_abs_mean", 0.0) or 0.0
                ),
                "ep_lane_pid_lat_err_norm_abs_mean": float(
                    info.get("ep_lane_pid_lat_err_norm_abs_mean", 0.0) or 0.0
                ),
                "ep_lane_pid_steer_abs_mean": float(
                    info.get("ep_lane_pid_steer_abs_mean", 0.0) or 0.0
                ),
                "ep_lane_pid_throttle_mean": float(
                    info.get("ep_lane_pid_throttle_mean", 0.0) or 0.0
                ),
                "ep_lane_pid_reverse_rate": float(
                    info.get("ep_lane_pid_reverse_rate", 0.0) or 0.0
                ),
                "ep_cte_abs_p90": float(info.get("ep_cte_abs_p90", 0.0) or 0.0),
                "ep_progress_ratio_forward_sum": float(
                    info.get("ep_progress_ratio_forward_sum", 0.0) or 0.0
                ),
                "ep_speed_mean": float(info.get("ep_speed_mean", 0.0) or 0.0),
                "ep_speed_min": float(info.get("ep_speed_min", 0.0) or 0.0),
                "ep_speed_max": float(info.get("ep_speed_max", 0.0) or 0.0),
                "window": self._window_snapshot(),
            }
            for key in (
                "ep_r_wait_window",
                "ep_r_force_pass",
                "ep_r_unsafe_close",
                "ep_r_obstacle_clearance",
                "ep_r_overtake",
                "ep_r_post_pass",
                "ep_r_post_pass_cut_in",
                "ep_overtake_count",
                "ep_post_pass_stability_count",
                "ep_pass_window_valid_rate",
                "ep_invalid_window_close_rate",
                "ep_unsafe_close_rate",
                "ep_obstacle_clearance_band_rate",
                "ep_obstacle_clearance_critical_rate",
                "ep_obstacle_clearance_risk_mean",
                "ep_obstacle_planar_distance_min",
                "ep_post_pass_watch_rate",
                "ep_post_pass_cut_in_rate",
                "ep_post_pass_cut_in_risk_mean",
                "ep_post_pass_planar_distance_min",
                "ep_post_pass_terminal_collision",
                "ep_overtake_success_ready_rate",
                "ep_overtake_passed_longitudinal_rate",
                "ep_overtake_success_clearance_ok_rate",
                "ep_overtake_success_candidate_rate",
                "ep_overtake_success_blocked_clearance_rate",
                "ep_overtake_success_blocked_progress_rate",
                "ep_overtake_success_blocked_safety_rate",
                "ep_overtake_success_grant_count",
            ):
                if key in info:
                    episode_payload[key] = float(info.get(key, 0.0) or 0.0)
            self._write_event("episode", episode_payload)

        stage_timesteps = self._stage_timesteps()
        if stage_timesteps >= self.min_stage_timesteps and self._all_recent_windows_ready():
            self.triggered = True
            self.stop_reason = "window_gates_ready"
            self.stop_num_timesteps = int(self.num_timesteps)
            self.stop_stage_timesteps = int(stage_timesteps)
            self._write_event("stop", {"stop_reason": self.stop_reason, "summary": self.summary()})
            if self.verbose > 0:
                print(
                    f"🎯 阶段晋级[{self.stage_name}]: "
                    f"total_steps={self.stop_num_timesteps}, "
                    f"stage_steps={self.stop_stage_timesteps}, "
                    f"success_counts={self.summary()['window_success_counts']}"
                )
            return False

        if self.max_stage_timesteps is not None and stage_timesteps >= self.max_stage_timesteps:
            self.triggered = False
            self.stop_reason = "max_stage_timesteps_reached"
            self.stop_num_timesteps = int(self.num_timesteps)
            self.stop_stage_timesteps = int(stage_timesteps)
            self._write_event("stop", {"stop_reason": self.stop_reason, "summary": self.summary()})
            if self.verbose > 0:
                print(
                    f"⏱️  阶段硬兜底[{self.stage_name}]命中: "
                    f"stage_steps={self.stop_stage_timesteps}"
                )
            return False

        return True

    def _on_training_end(self) -> None:
        try:
            self._write_event("end", {"summary": self.summary()})
        finally:
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception:
                    pass
                self._fh = None


def _resolve_track_dir(track_dir: Optional[str], env_ids: List[str]) -> str:
    raw_candidates: List[Path] = []
    if track_dir:
        raw_candidates.append(Path(track_dir).expanduser())

    env_override = os.environ.get("MYSIM_TRACK_DIR", "").strip()
    if env_override:
        raw_candidates.append(Path(env_override).expanduser())

    raw_candidates.extend([
        REPO_ROOT / "track_profiles",
        REPO_ROOT / "track",
        Path("/home/longzhao/track"),
        REPO_ROOT / "module" / "track_data",
    ])

    requested_abs = os.path.abspath(os.path.expanduser(str(track_dir))) if track_dir else ""
    required_files = [SCENE_SPECS[eid]["track_file"] for eid in env_ids]
    seen = set()
    candidate_summaries: List[str] = []

    for candidate in raw_candidates:
        candidate_abs = os.path.abspath(os.path.expanduser(str(candidate)))
        if candidate_abs in seen:
            continue
        seen.add(candidate_abs)

        candidate_path = Path(candidate_abs)
        if not candidate_path.is_dir():
            candidate_summaries.append(f"{candidate_abs} (missing directory)")
            continue

        missing = [fn for fn in required_files if not (candidate_path / fn).is_file()]
        if not missing:
            if requested_abs and candidate_abs != requested_abs:
                print(f"ℹ️  track_dir 回退到: {candidate_abs}")
            elif not requested_abs:
                print(f"ℹ️  auto track_dir: {candidate_abs}")
            return candidate_abs

        candidate_summaries.append(f"{candidate_abs} (missing: {', '.join(missing)})")

    raise FileNotFoundError(
        "No usable track_dir found for env_ids="
        f"{env_ids}. Tried: {'; '.join(candidate_summaries)}"
    )


def _probe_sim_tcp(host: str, port: int, timeout_s: float = 1.0) -> Tuple[bool, str]:
    try:
        with socket.create_connection((str(host), int(port)), timeout=float(timeout_s)):
            return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _install_sim_wait_timeout_patch(timeout_s: float = 35.0, resend_scene_names_s: float = 3.0) -> bool:
    try:
        from gym_donkeycar.envs.donkey_sim import DonkeyUnitySimContoller
    except Exception as e:
        print(f"⚠️  cannot patch sim wait timeout: {type(e).__name__}: {e}")
        return False

    timeout_s = float(max(5.0, timeout_s))
    resend_scene_names_s = float(max(0.0, resend_scene_names_s))
    sim_logger = logging.getLogger("gym_donkeycar.envs.donkey_sim")

    def _wait_until_loaded_with_timeout(self) -> None:
        time.sleep(0.1)
        start_t = time.time()
        last_resend_t = 0.0
        while not self.handler.loaded:
            elapsed = time.time() - start_t
            sim_logger.warning("waiting for sim to start..")
            if elapsed >= timeout_s:
                host = getattr(self, "address", ("localhost", -1))[0]
                port = getattr(self, "address", ("localhost", -1))[1]
                raise TimeoutError(
                    f"sim load handshake timeout after {timeout_s:.0f}s for {host}:{port}. "
                    f"Please restart DonkeySim and retry."
                )
            if resend_scene_names_s > 0.0 and (elapsed - last_resend_t) >= resend_scene_names_s:
                try:
                    if hasattr(self, "handler") and self.handler is not None:
                        self.handler.send_get_scene_names()
                except Exception:
                    pass
                last_resend_t = elapsed
            time.sleep(1.0)
        sim_logger.info("sim started!")

    DonkeyUnitySimContoller.wait_until_loaded = _wait_until_loaded_with_timeout
    return True


def run_offline_track_checks(track_geometry: TrackGeometryManager) -> None:
    summaries = track_geometry.scene_summary()
    if len(summaries) < 1:
        raise RuntimeError("No scene geometry loaded")
    for scene, summary in summaries.items():
        if int(summary["points"]) <= 100:
            raise AssertionError(f"scene={scene} has too few points: {summary['points']}")
    for scene in track_geometry.list_scenes():
        g = track_geometry.scenes[scene]
        n = g.center.shape[0]
        for idx in [0, n // 4, n // 2, (3 * n) // 4]:
            x, z = g.center[idx]
            out = track_geometry.query(scene, x=float(x), z=float(z), yaw_rad=0.0, prev_idx=idx)
            vals = [
                out["lat_err_norm"],
                out["heading_err_sin"],
                out["heading_err_cos"],
                out["kappa_lookahead"],
                out["width_norm"],
            ]
            if not all(np.isfinite(vals)):
                raise AssertionError(f"Non-finite geometry output at scene={scene}, idx={idx}")


def run_v16_contract_tests(obs_size: int = 128) -> None:
    from module.action_adapter import ActionAdapterWrapper
    from module.control import ActionSafetyWrapper
    from module.reward import DonkeyRewardWrapper
    from module.wrappers import CanonicalSemanticWrapper

    class DummyBaseEnv(gym.Env):
        def __init__(self, obs_size_: int):
            self.obs_size = int(obs_size_)
            self.observation_space = gym.spaces.Box(
                low=0, high=255,
                shape=(self.obs_size, self.obs_size, 3),
                dtype=np.uint8,
            )
            self.action_space = gym.spaces.Box(
                low=np.array([-1.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
            self._step_count = 0

        def reset(self):
            self._step_count = 0
            return np.random.randint(0, 255, (self.obs_size, self.obs_size, 3), dtype=np.uint8)

        def step(self, action):
            self._step_count += 1
            obs = np.random.randint(0, 255, (self.obs_size, self.obs_size, 3), dtype=np.uint8)
            info = {
                "speed": 0.8,
                "gyro": (0.0, 0.2, 0.0),
                "accel": (0.1, 0.5, 9.7),
                "car": (0.0, 0.0, 30.0),
                "pos": (0.0, 0.0, 0.0),
                "cte": 0.1,
                "obstacle_present": 1.0,
                "obstacle_longitudinal": 2.2,
                "obstacle_lateral": -0.4,
                "obstacle_dist": 2.4,
                "obstacle_risk": 0.55,
            }
            done = self._step_count >= 8
            return obs, 0.0, done, info

    print("🧪 V16 Contract Tests")
    base = DummyBaseEnv(obs_size)
    semantic = CanonicalSemanticWrapper(base, domain="ws", obs_size=obs_size, augment=False)
    reward_wrapper = DonkeyRewardWrapper(
        semantic,
        total_timesteps=100000,
        action_safety_wrapper=None,
        progress_reward_scale=48.0,
        survival_reward_scale=0.30,
        collision_penalty_base=8.0,
        offtrack_penalty_base=5.0,
        w_near_collision=0.20,
        near_collision_start_ratio=0.65,
    )
    action_safety = ActionSafetyWrapper(reward_wrapper, delta_max=0.35)
    reward_wrapper.action_safety_wrapper = action_safety
    adapter = ActionAdapterWrapper(action_safety, max_throttle=0.3)
    env = MultiInputObsWrapper(
        adapter,
        track_geometry=None,
        scene_key="waveshare",
        logging_key="ws",
        domain="ws",
        obs_size=obs_size,
        image_channels=6,
        include_cte_in_obs=False,
        speed_vmax=2.2,
        control_wrapper=adapter,
        action_safety_wrapper=action_safety,
        state_builder=_build_state_v16,
        state_dim=12,
    )

    obs = env.reset()
    assert obs["image"].shape == (6, obs_size, obs_size)
    assert obs["state"].shape == (12,)
    obs2, _r, _d, info = env.step(np.array([0.3, -0.2, 0.7], dtype=np.float32))
    assert obs2["state"].shape == (12,)
    assert float(obs2["state"][7]) >= 0.0
    assert "ctrl/steer_core" in info
    print("  ✅ V16 contract tests passed")


def run_preflight_tests(track_geometry: TrackGeometryManager, obs_size: int = 128) -> None:
    print("\n🔍 Running preflight checks...")
    run_offline_track_checks(track_geometry)
    run_v16_contract_tests(obs_size=obs_size)
    print("✅ All preflight checks passed\n")


def train_v16(
    env_ids: Optional[List[str]] = None,
    scene_weights: Optional[List[float]] = None,
    track_dir: str = DEFAULT_TRACK_DIR,
    sim_path: str = "remote",
    total_timesteps: int = 2_000_000,
    save_dir: str = "models/v16_multi_scene_obstacle",
    port: int = 9091,
    obs_size: int = 128,
    augment: bool = False,
    yellow_dropout_prob: float = 0.20,
    dropout_start_step: int = 0,
    dropout_ramp_steps: int = 200_000,
    scene_start_snapshot_steps: int = 10,
    lstm_hidden_size: int = 256,
    lstm_layers: int = 1,
    adapter_k_delta: float = 0.15,
    adapter_lambda_bias: float = 0.20,
    adapter_k_bias: float = 0.15,
    adapter_steer_core_decay: float = 0.0,
    adapter_v_nominal: float = 1.4,
    adapter_k_turn: float = 0.5,
    adapter_k_bias_speed: float = 0.0,
    adapter_alpha_speed: float = 0.25,
    adapter_v_min: float = 0.6,
    adapter_v_max: float = 1.8,
    adapter_max_throttle: float = 0.3,
    speed_vmax: float = 2.2,
    speed_kp: float = 0.35,
    speed_ki: float = 0.08,
    speed_kff: float = 0.10,
    allow_reverse: bool = False,
    control_dt: float = 0.05,
    learning_rate: float = 8e-5,
    ent_coef: float = 0.01,
    ppo_n_steps: int = 4096,
    ppo_batch_size: int = 256,
    ppo_n_epochs: int = 4,
    ppo_clip_range: float = 0.2,
    target_kl: Optional[float] = 0.01,
    min_episodes_per_scene: int = 5,
    max_steps_per_scene: int = 640,
    enable_dynamic_scene_weights: bool = True,
    dynamic_weight_update_episodes: int = 24,
    dynamic_weight_window: int = 50,
    dynamic_min_samples_per_scene: int = 6,
    dynamic_weight_alpha: float = 1.6,
    dynamic_length_beta: float = 1.0,
    dynamic_weight_smoothing: float = 0.35,
    dynamic_weight_min: float = 0.02,
    dynamic_weight_max: float = 0.55,
    dynamic_success_mode: str = "scene_adaptive",
    dynamic_success_warmup_episodes: int = 1200,
    dynamic_success_post_warmup_scale: float = 0.20,
    dynamic_success_deficit_mix: float = 0.85,
    enable_step_balance_sampling: bool = True,
    step_balance_sampling_mix: float = 0.3,
    delta_max: float = 0.35,
    enable_lpf: bool = True,
    beta: float = 0.6,
    w_d: float = 0.04,
    w_dd: float = 0.01,
    w_m: float = 0.0,
    w_sat: float = 0.0,
    w_time: float = 0.01,
    w_center: float = 0.03,
    w_heading: float = 0.015,
    w_speed_ref: float = 0.0,
    speed_ref_vmin: float = 0.35,
    speed_ref_vmax: float = 2.2,
    speed_ref_kappa_ref: float = 0.15,
    lap_reward_scale: float = 1.0,
    progress_reward_scale: float = 48.0,
    survival_reward_scale: float = 0.30,
    collision_penalty_base: float = 8.0,
    offtrack_penalty_base: float = 5.0,
    adaptive_delta_max: bool = True,
    curve_delta_boost: float = 1.0,
    curve_kappa_ref: float = 0.15,
    steer_intent_boost: float = 0.30,
    hairpin_curve_ratio: float = 0.85,
    hairpin_min_delta_max: float = 0.45,
    hairpin_max_delta_max: float = 0.85,
    w_near_offtrack: float = 0.55,
    near_offtrack_start_ratio: float = 0.45,
    w_near_collision: float = 0.24,
    near_collision_start_ratio: float = 0.65,
    overtake_success_bonus: float = 2.5,
    offtrack_leniency_ratio: float = 0.25,
    offtrack_leniency_mult: float = 2.5,
    obstacle_enabled: bool = True,
    obstacle_count: int = 2,
    obstacle_free_prob: float = 0.15,
    obstacle_modes: Optional[List[str]] = None,
    ws_obstacle_free_prob: Optional[float] = None,
    obstacle_spawn_ahead_min_m: float = 3.5,
    obstacle_spawn_ahead_max_m: float = 14.0,
    obstacle_min_agent_planar_dist_m: float = 1.5,
    obstacle_min_agent_arc_dist_m: float = 3.5,
    obstacle_min_separation_world: float = 3.0,
    obstacle_lateral_choices: Optional[List[float]] = None,
    obstacle_fixed_progress_ratio: Optional[float] = None,
    obstacle_fixed_progress_gap: Optional[float] = None,
    obstacle_fixed_progress_gap_min: Optional[float] = None,
    obstacle_fixed_progress_gap_max: Optional[float] = None,
    obstacle_progress_min: Optional[float] = None,
    obstacle_progress_max: Optional[float] = None,
    obstacle_fixed_lateral_ratio: Optional[float] = None,
    gt_obstacle_start_exclusion_half_width_m: Optional[float] = None,
    ws_obstacle_modes: Optional[List[str]] = None,
    ws_obstacle_fixed_progress_ratio: Optional[float] = None,
    ws_obstacle_progress_min: Optional[float] = None,
    ws_obstacle_progress_max: Optional[float] = None,
    ws_obstacle_fixed_lateral_ratio: Optional[float] = None,
    obstacle_randomize_non_lane_pid_yaw: bool = True,
    obstacle_lane_pid_speed_gt: float = 0.85,
    obstacle_lane_pid_speed_ws: float = 0.70,
    obstacle_lane_pid_lookahead_m: float = 0.9,
    obstacle_jitter_amplitude_m: float = 0.10,
    obstacle_jitter_period_s: float = 1.5,
    obstacle_jitter_update_hz: float = 8.0,
    obstacle_nudge_amplitude_m: float = 0.14,
    obstacle_nudge_period_s: float = 1.5,
    obstacle_nudge_update_hz: float = 8.0,
    obstacle_seed: Optional[int] = None,
    ego_random_spawn: bool = False,
    ego_spawn_lateral_ratio: float = 0.5,
    sim_loaded_timeout_s: float = 35.0,
    sim_wait_resend_scene_names_s: float = 3.0,
    curriculum_phase: Optional[str] = None,
    seed: Optional[int] = None,
    exp_tag: Optional[str] = None,
    resume_latest: bool = False,
    resume_path: Optional[str] = None,
    run_preflight_checks: bool = True,
    enable_file_metrics_log: bool = True,
    file_metrics_log_freq: int = 500,
    file_metrics_log_name: str = "train_metrics.jsonl",
    enable_auto_lr_decay: bool = True,
    auto_lr_check_freq: int = 1000,
    auto_lr_decay_factor: float = 0.92,
    auto_lr_min: float = 5e-5,
    auto_lr_high_kl: float = 0.05,
    auto_lr_high_kl_patience: int = 3,
    auto_lr_balanced_drop: float = 8.0,
    auto_lr_balanced_patience: int = 12,
    auto_lr_cooldown_checks: int = 15,
    auto_lr_warmup_steps: int = 250000,
    auto_lr_best_window: int = 50,
    sim2real_json: Optional[str] = None,
    extra_callbacks: Optional[List[BaseCallback]] = None,
    extra_run_metadata: Optional[Dict[str, Any]] = None,
    config_filename: str = "v16_config.json",
):
    if RecurrentPPO is None:
        raise ImportError("sb3_contrib not available, please install sb3-contrib==1.8.0")

    env_ids = list(DEFAULT_ENV_IDS if env_ids is None else env_ids)
    for env_id in env_ids:
        if env_id not in SCENE_SPECS:
            raise KeyError(f"Unsupported env_id for V16: {env_id}")

    curriculum_values = {
        "scene_weights": scene_weights,
        "enable_dynamic_scene_weights": enable_dynamic_scene_weights,
        "enable_step_balance_sampling": enable_step_balance_sampling,
        "obstacle_enabled": obstacle_enabled,
        "obstacle_count": obstacle_count,
        "obstacle_free_prob": obstacle_free_prob,
        "obstacle_modes": obstacle_modes,
        "ws_obstacle_free_prob": ws_obstacle_free_prob,
        "obstacle_spawn_ahead_min_m": obstacle_spawn_ahead_min_m,
        "obstacle_spawn_ahead_max_m": obstacle_spawn_ahead_max_m,
        "obstacle_min_agent_planar_dist_m": obstacle_min_agent_planar_dist_m,
        "obstacle_min_agent_arc_dist_m": obstacle_min_agent_arc_dist_m,
        "obstacle_fixed_progress_ratio": obstacle_fixed_progress_ratio,
        "obstacle_fixed_progress_gap": obstacle_fixed_progress_gap,
        "obstacle_fixed_progress_gap_min": obstacle_fixed_progress_gap_min,
        "obstacle_fixed_progress_gap_max": obstacle_fixed_progress_gap_max,
        "obstacle_progress_min": obstacle_progress_min,
        "obstacle_progress_max": obstacle_progress_max,
        "obstacle_fixed_lateral_ratio": obstacle_fixed_lateral_ratio,
        "gt_obstacle_start_exclusion_half_width_m": gt_obstacle_start_exclusion_half_width_m,
        "ws_obstacle_modes": ws_obstacle_modes,
        "ws_obstacle_fixed_progress_ratio": ws_obstacle_fixed_progress_ratio,
        "ws_obstacle_progress_min": ws_obstacle_progress_min,
        "ws_obstacle_progress_max": ws_obstacle_progress_max,
        "ws_obstacle_fixed_lateral_ratio": ws_obstacle_fixed_lateral_ratio,
        "obstacle_randomize_non_lane_pid_yaw": obstacle_randomize_non_lane_pid_yaw,
        "obstacle_lane_pid_speed_gt": obstacle_lane_pid_speed_gt,
        "obstacle_lane_pid_speed_ws": obstacle_lane_pid_speed_ws,
        "collision_penalty_base": collision_penalty_base,
        "offtrack_penalty_base": offtrack_penalty_base,
        "w_near_collision": w_near_collision,
        "near_collision_start_ratio": near_collision_start_ratio,
        "overtake_success_bonus": overtake_success_bonus,
    }
    curriculum_phase, curriculum_applied = _apply_curriculum_phase(curriculum_phase, curriculum_values)
    scene_weights = curriculum_values["scene_weights"]
    enable_dynamic_scene_weights = curriculum_values["enable_dynamic_scene_weights"]
    enable_step_balance_sampling = curriculum_values["enable_step_balance_sampling"]
    obstacle_enabled = curriculum_values["obstacle_enabled"]
    obstacle_count = curriculum_values["obstacle_count"]
    obstacle_free_prob = curriculum_values["obstacle_free_prob"]
    obstacle_modes = curriculum_values["obstacle_modes"]
    ws_obstacle_free_prob = curriculum_values["ws_obstacle_free_prob"]
    obstacle_spawn_ahead_min_m = curriculum_values["obstacle_spawn_ahead_min_m"]
    obstacle_spawn_ahead_max_m = curriculum_values["obstacle_spawn_ahead_max_m"]
    obstacle_min_agent_planar_dist_m = curriculum_values["obstacle_min_agent_planar_dist_m"]
    obstacle_min_agent_arc_dist_m = curriculum_values["obstacle_min_agent_arc_dist_m"]
    obstacle_fixed_progress_ratio = curriculum_values["obstacle_fixed_progress_ratio"]
    obstacle_fixed_progress_gap = curriculum_values["obstacle_fixed_progress_gap"]
    obstacle_fixed_progress_gap_min = curriculum_values.get("obstacle_fixed_progress_gap_min", None)
    obstacle_fixed_progress_gap_max = curriculum_values.get("obstacle_fixed_progress_gap_max", None)
    obstacle_progress_min = curriculum_values.get("obstacle_progress_min", None)
    obstacle_progress_max = curriculum_values.get("obstacle_progress_max", None)
    obstacle_fixed_lateral_ratio = curriculum_values["obstacle_fixed_lateral_ratio"]
    gt_obstacle_start_exclusion_half_width_m = curriculum_values.get(
        "gt_obstacle_start_exclusion_half_width_m",
        None,
    )
    ws_obstacle_modes = curriculum_values["ws_obstacle_modes"]
    ws_obstacle_fixed_progress_ratio = curriculum_values["ws_obstacle_fixed_progress_ratio"]
    ws_obstacle_progress_min = curriculum_values.get("ws_obstacle_progress_min", None)
    ws_obstacle_progress_max = curriculum_values.get("ws_obstacle_progress_max", None)
    ws_obstacle_fixed_lateral_ratio = curriculum_values["ws_obstacle_fixed_lateral_ratio"]
    obstacle_randomize_non_lane_pid_yaw = curriculum_values["obstacle_randomize_non_lane_pid_yaw"]
    obstacle_lane_pid_speed_gt = curriculum_values["obstacle_lane_pid_speed_gt"]
    obstacle_lane_pid_speed_ws = curriculum_values["obstacle_lane_pid_speed_ws"]
    collision_penalty_base = curriculum_values["collision_penalty_base"]
    offtrack_penalty_base = curriculum_values["offtrack_penalty_base"]
    w_near_collision = curriculum_values["w_near_collision"]
    near_collision_start_ratio = curriculum_values["near_collision_start_ratio"]
    overtake_success_bonus = curriculum_values["overtake_success_bonus"]

    if scene_weights is None:
        scene_weights = [1.0 / len(env_ids)] * len(env_ids)
    else:
        total_w = float(sum(scene_weights))
        if len(scene_weights) != len(env_ids) or total_w <= 0:
            raise ValueError("scene_weights length/sum invalid")
        scene_weights = [float(w) / total_w for w in scene_weights]

    _seed_everything(seed)
    track_dir = _resolve_track_dir(track_dir=track_dir, env_ids=env_ids)
    os.makedirs(save_dir, exist_ok=True)
    snapshot_dir = os.path.join(save_dir, "scene_start_snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    print("\n" + "=" * 76)
    print("🚀 DonkeyCar PPO V16 - Dual-Domain Obstacle Training")
    print("=" * 76)
    print(f"maps: {env_ids}")
    print(f"obs: Dict(image=6x{obs_size}x{obs_size}, state=12)")
    print(f"track_dir: {track_dir}")
    if curriculum_phase is not None:
        if curriculum_applied:
            print(f"curriculum_phase: {curriculum_phase} ({', '.join(sorted(curriculum_applied.keys()))})")
        else:
            print(f"curriculum_phase: {curriculum_phase} (explicit settings kept all defaults from being overridden)")
    print(f"obstacle: enabled={obstacle_enabled}, count={obstacle_count}, modes={obstacle_modes or ['static', 'jitter']}")

    track_geometry = TrackGeometryManager(track_dir=track_dir, env_ids=env_ids, scene_specs=SCENE_SPECS)
    if run_preflight_checks:
        run_preflight_tests(track_geometry=track_geometry, obs_size=obs_size)

    _launch_sim = bool(sim_path and sim_path not in ("", "remote", "none"))
    sim_host = "127.0.0.1"
    sim_port = int(port)

    if sim_loaded_timeout_s > 0:
        _install_sim_wait_timeout_patch(
            timeout_s=float(sim_loaded_timeout_s),
            resend_scene_names_s=float(sim_wait_resend_scene_names_s),
        )
    if not _launch_sim:
        ok, err = _probe_sim_tcp(sim_host, sim_port, timeout_s=1.0)
        if ok:
            print(f"✅ sim tcp reachable: {sim_host}:{sim_port}")
        else:
            print(f"⚠️  sim tcp not reachable: {sim_host}:{sim_port} ({err})")

    cfg = load_config(myconfig=DEFAULT_MYCONFIG)
    if cfg is not None and hasattr(cfg, "GYM_CONF"):
        conf = cfg.GYM_CONF.copy()
        conf.update(
            {
                "port": sim_port,
                "car_name": "waveshare_v16",
                "racer_name": "V16-Obstacle",
                "country": "CN",
                "bio": "V16 dual-domain obstacle PPO",
                "guid": "waveshare-v16-obstacle",
                "max_cte": 8.0,
            }
        )
        if _launch_sim:
            conf["exe_path"] = sim_path
    else:
        conf = {
            "host": sim_host,
            "port": sim_port,
            "car_name": "waveshare_v16",
            "racer_name": "V16-Obstacle",
            "country": "CN",
            "bio": "V16 dual-domain obstacle PPO",
            "guid": "waveshare-v16-obstacle",
            "max_cte": 8.0,
        }
        if _launch_sim:
            conf["exe_path"] = sim_path

    from gym import spaces

    dummy_obs_space = spaces.Dict(
        {
            "image": spaces.Box(
                low=np.full((6, obs_size, obs_size), 0.0, dtype=np.float32),
                high=np.full((6, obs_size, obs_size), 1.0, dtype=np.float32),
                dtype=np.float32,
            ),
            "state": spaces.Box(
                low=np.full((12,), -3.0, dtype=np.float32),
                high=np.full((12,), 3.0, dtype=np.float32),
                dtype=np.float32,
            ),
        }
    )
    dummy_act_space = spaces.Box(
        low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        dtype=np.float32,
    )

    class DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = dummy_obs_space
            self.action_space = dummy_act_space

        def reset(self):
            return {
                "image": np.zeros((6, obs_size, obs_size), dtype=np.float32),
                "state": np.zeros((12,), dtype=np.float32),
            }

        def step(self, action):
            return self.reset(), 0.0, False, {}

    dummy_vec_env = DummyVecEnv([lambda: DummyEnv()])
    _safe_seed_env(dummy_vec_env, seed, label="dummy_v16_env")

    policy_kwargs = dict(
        features_extractor_class=FiLMFeatureExtractor,
        features_extractor_kwargs=dict(image_feat_dim=128, state_feat_dim=32),
        lstm_hidden_size=int(lstm_hidden_size),
        n_lstm_layers=int(lstm_layers),
        shared_lstm=False,
        enable_critic_lstm=True,
    )

    resume_ckpt_path = resume_path
    if resume_ckpt_path is None and resume_latest:
        resume_ckpt_path = _find_latest_checkpoint(save_dir, name_prefix="v16")
        if resume_ckpt_path is None:
            raise FileNotFoundError(f"No v16 checkpoint found in {save_dir}")

    if resume_ckpt_path is not None:
        print(f"🔄 Resume from: {resume_ckpt_path}")
        model = RecurrentPPO.load(resume_ckpt_path, env=dummy_vec_env)
        model.learning_rate = float(learning_rate)
        model.lr_schedule = lambda _p, _v=float(learning_rate): _v
        for pg in model.policy.optimizer.param_groups:
            pg["lr"] = float(learning_rate)
        model.ent_coef = float(ent_coef)
        model.n_steps = int(ppo_n_steps)
        model.batch_size = int(ppo_batch_size)
        model.n_epochs = int(ppo_n_epochs)
        model.clip_range = lambda _p, _v=float(ppo_clip_range): _v
        model.target_kl = (None if target_kl is None else float(target_kl))
    else:
        model = RecurrentPPO(
            "MultiInputLstmPolicy",
            dummy_vec_env,
            policy_kwargs=policy_kwargs,
            learning_rate=float(learning_rate),
            n_steps=int(ppo_n_steps),
            batch_size=int(ppo_batch_size),
            n_epochs=int(ppo_n_epochs),
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=float(ppo_clip_range),
            clip_range_vf=None,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=target_kl,
            ent_coef=float(ent_coef),
            verbose=1,
            tensorboard_log=(os.path.join(save_dir, "tensorboard", exp_tag) if exp_tag else os.path.join(save_dir, "tensorboard")),
            seed=(None if seed is None else int(seed)),
        )
    model_start_timesteps = int(getattr(model, "num_timesteps", 0))
    dummy_vec_env.close()

    def make_env():
        return MultiSceneEnvV16(
            env_ids=env_ids,
            conf=conf,
            scene_weights=scene_weights,
            scene_specs=SCENE_SPECS,
            track_geometry=track_geometry,
            track_dir=track_dir,
            obs_size=obs_size,
            augment=augment,
            yellow_dropout_prob=yellow_dropout_prob,
            dropout_start_step=dropout_start_step,
            dropout_ramp_steps=dropout_ramp_steps,
            adapter_k_delta=adapter_k_delta,
            adapter_lambda_bias=adapter_lambda_bias,
            adapter_k_bias=adapter_k_bias,
            adapter_steer_core_decay=adapter_steer_core_decay,
            adapter_v_nominal=adapter_v_nominal,
            adapter_k_turn=adapter_k_turn,
            adapter_k_bias_speed=adapter_k_bias_speed,
            adapter_alpha_speed=adapter_alpha_speed,
            adapter_v_min=adapter_v_min,
            adapter_v_max=adapter_v_max,
            speed_vmax=speed_vmax,
            speed_kp=speed_kp,
            speed_ki=speed_ki,
            speed_kff=speed_kff,
            allow_reverse=allow_reverse,
            max_throttle=adapter_max_throttle,
            control_dt=control_dt,
            total_timesteps=total_timesteps,
            delta_max=delta_max,
            enable_lpf=enable_lpf,
            beta=beta,
            w_d=w_d,
            w_dd=w_dd,
            w_m=w_m,
            w_sat=w_sat,
            w_time=w_time,
            w_center=w_center,
            w_heading=w_heading,
            w_speed_ref=w_speed_ref,
            speed_ref_vmin=speed_ref_vmin,
            speed_ref_vmax=speed_ref_vmax,
            speed_ref_kappa_ref=speed_ref_kappa_ref,
            lap_reward_scale=lap_reward_scale,
            progress_reward_scale=progress_reward_scale,
            survival_reward_scale=survival_reward_scale,
            collision_penalty_base=collision_penalty_base,
            offtrack_penalty_base=offtrack_penalty_base,
            adaptive_delta_max=adaptive_delta_max,
            curve_delta_boost=curve_delta_boost,
            curve_kappa_ref=curve_kappa_ref,
            steer_intent_boost=steer_intent_boost,
            hairpin_curve_ratio=hairpin_curve_ratio,
            hairpin_min_delta_max=hairpin_min_delta_max,
            hairpin_max_delta_max=hairpin_max_delta_max,
            w_near_offtrack=w_near_offtrack,
            near_offtrack_start_ratio=near_offtrack_start_ratio,
            w_near_collision=w_near_collision,
            near_collision_start_ratio=near_collision_start_ratio,
            overtake_success_bonus=overtake_success_bonus,
            offtrack_leniency_ratio=offtrack_leniency_ratio,
            offtrack_leniency_mult=offtrack_leniency_mult,
            snapshot_dir=snapshot_dir,
            snapshot_max_steps=scene_start_snapshot_steps,
            min_episodes_per_scene=min_episodes_per_scene,
            max_steps_per_scene=max_steps_per_scene,
            enable_dynamic_scene_weights=enable_dynamic_scene_weights,
            dynamic_weight_update_episodes=dynamic_weight_update_episodes,
            dynamic_weight_window=dynamic_weight_window,
            dynamic_min_samples_per_scene=dynamic_min_samples_per_scene,
            dynamic_weight_alpha=dynamic_weight_alpha,
            dynamic_length_beta=dynamic_length_beta,
            dynamic_weight_smoothing=dynamic_weight_smoothing,
            dynamic_weight_min=dynamic_weight_min,
            dynamic_weight_max=dynamic_weight_max,
            dynamic_success_mode=dynamic_success_mode,
            dynamic_success_warmup_episodes=dynamic_success_warmup_episodes,
            dynamic_success_post_warmup_scale=dynamic_success_post_warmup_scale,
            dynamic_success_deficit_mix=dynamic_success_deficit_mix,
            enable_step_balance_sampling=enable_step_balance_sampling,
            step_balance_sampling_mix=step_balance_sampling_mix,
            obstacle_enabled=obstacle_enabled,
            obstacle_count=obstacle_count,
            obstacle_free_prob=obstacle_free_prob,
            obstacle_modes=obstacle_modes,
            ws_obstacle_free_prob=ws_obstacle_free_prob,
            obstacle_spawn_ahead_min_m=obstacle_spawn_ahead_min_m,
            obstacle_spawn_ahead_max_m=obstacle_spawn_ahead_max_m,
            obstacle_min_agent_planar_dist_m=obstacle_min_agent_planar_dist_m,
            obstacle_min_agent_arc_dist_m=obstacle_min_agent_arc_dist_m,
            obstacle_min_separation_world=obstacle_min_separation_world,
            obstacle_lateral_choices=obstacle_lateral_choices,
            obstacle_fixed_progress_ratio=obstacle_fixed_progress_ratio,
            obstacle_fixed_progress_gap=obstacle_fixed_progress_gap,
            obstacle_fixed_progress_gap_min=obstacle_fixed_progress_gap_min,
            obstacle_fixed_progress_gap_max=obstacle_fixed_progress_gap_max,
            obstacle_progress_min=obstacle_progress_min,
            obstacle_progress_max=obstacle_progress_max,
            obstacle_fixed_lateral_ratio=obstacle_fixed_lateral_ratio,
            gt_obstacle_start_exclusion_half_width_m=gt_obstacle_start_exclusion_half_width_m,
            ws_obstacle_modes=ws_obstacle_modes,
            ws_obstacle_fixed_progress_ratio=ws_obstacle_fixed_progress_ratio,
            ws_obstacle_progress_min=ws_obstacle_progress_min,
            ws_obstacle_progress_max=ws_obstacle_progress_max,
            ws_obstacle_fixed_lateral_ratio=ws_obstacle_fixed_lateral_ratio,
            obstacle_randomize_non_lane_pid_yaw=obstacle_randomize_non_lane_pid_yaw,
            obstacle_lane_pid_speed_gt=obstacle_lane_pid_speed_gt,
            obstacle_lane_pid_speed_ws=obstacle_lane_pid_speed_ws,
            obstacle_lane_pid_lookahead_m=obstacle_lane_pid_lookahead_m,
            obstacle_jitter_amplitude_m=obstacle_jitter_amplitude_m,
            obstacle_jitter_period_s=obstacle_jitter_period_s,
            obstacle_jitter_update_hz=obstacle_jitter_update_hz,
            obstacle_nudge_amplitude_m=obstacle_nudge_amplitude_m,
            obstacle_nudge_period_s=obstacle_nudge_period_s,
            obstacle_nudge_update_hz=obstacle_nudge_update_hz,
            obstacle_seed=obstacle_seed,
            ego_random_spawn=ego_random_spawn,
            ego_spawn_lateral_ratio=ego_spawn_lateral_ratio,
            sim2real_json=sim2real_json,
        )

    env = DummyVecEnv([make_env])
    _safe_seed_env(env, seed, label="v16_train_env")

    # 设置环境的当前课程阶段（用于step预算补偿callback）
    if curriculum_phase is not None:
        for envs_list in env.envs:
            if hasattr(envs_list, '_curriculum_phase'):
                envs_list._curriculum_phase = curriculum_phase

    model.set_env(env)

    callbacks: List[BaseCallback] = [
        PTHExportCallback(save_path=save_dir, save_freq=20000, name_prefix="v16", verbose=1),
        BestModelCallback(
            save_path=save_dir,
            check_freq=1000,
            metric_mode="per_scene_min",
            min_episodes_per_scene_for_save=10,
            save_separate_per_scene_best=True,
            scene_keys=None,
            save_balanced_from_training_buffer=False,
            verbose=1,
        ),
        PerSceneStatsCallback(check_freq=1000, short_episode_threshold=15, verbose=1),
        SceneSchedulerLoggingCallback(check_freq=1000, verbose=0),
        ShortEpisodeLoggerCallback(save_dir=save_dir, threshold=15, verbose=1),
        TqdmProgressCallback(total_timesteps=total_timesteps, update_freq=2048),
        CrashRecoveryCallback(
            save_dir=save_dir,
            check_freq=2000,
            rolling_window=30,
            crash_ratio=0.25,
            min_peak_len=80.0,
            cooldown_steps=50000,
            min_warmup_steps=30000,
            verbose=1,
        ),
        StepBudgetCompensationCallback(
            curriculum_phases=CURRICULUM_PHASES,
            window_episode_count=50,
            verbose=1,
        ),
    ]

    if enable_auto_lr_decay:
        callbacks.append(
            AdaptiveLearningRateCallback(
                check_freq=auto_lr_check_freq,
                scene_keys=None,
                min_episodes_per_domain=10,
                balanced_drop_threshold=auto_lr_balanced_drop,
                balanced_drop_patience=auto_lr_balanced_patience,
                high_kl_threshold=auto_lr_high_kl,
                high_kl_patience=auto_lr_high_kl_patience,
                decay_factor=auto_lr_decay_factor,
                min_lr=auto_lr_min,
                cooldown_checks=auto_lr_cooldown_checks,
                warmup_steps=auto_lr_warmup_steps,
                best_window=auto_lr_best_window,
                verbose=1,
            )
        )
    if enable_file_metrics_log:
        callbacks.append(
            TrainingMetricsFileLoggerCallback(
                save_dir=save_dir,
                log_freq=file_metrics_log_freq,
                filename=file_metrics_log_name,
                exp_tag=exp_tag,
                verbose=1,
            )
        )
    if extra_callbacks:
        callbacks.extend(list(extra_callbacks))

    print("\n" + "=" * 76)
    print("🚦 Start V16 training")
    print("=" * 76)
    start_time = time.time()
    interrupted = False
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=False,
            reset_num_timesteps=(resume_ckpt_path is None),
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\n⚠️  Training interrupted by user")

    elapsed = time.time() - start_time
    model_end_timesteps = int(getattr(model, "num_timesteps", model_start_timesteps))
    trained_timesteps = max(0, model_end_timesteps - model_start_timesteps)

    final_model_path = os.path.join(save_dir, "final_model")
    model.save(final_model_path)
    final_pth_path = os.path.join(save_dir, "final_model_policy.pth")
    torch.save(model.policy.state_dict(), final_pth_path)

    callback_summaries: Dict[str, Any] = {}
    callback_name_counts: Dict[str, int] = {}
    for cb in callbacks:
        summary_fn = getattr(cb, "summary", None)
        if not callable(summary_fn):
            continue
        try:
            summary_value = summary_fn()
        except Exception as e:
            summary_value = {"summary_error": f"{type(e).__name__}: {e}"}
        base_name = getattr(cb, "stage_name", None) or cb.__class__.__name__
        base_name = str(base_name)
        idx = callback_name_counts.get(base_name, 0)
        callback_name_counts[base_name] = idx + 1
        key = base_name if idx == 0 else f"{base_name}_{idx + 1}"
        callback_summaries[key] = summary_value

    config = {
        "version": "V16",
        "timestamp": datetime.now().isoformat(),
        "curriculum": {
            "phase": curriculum_phase,
            "applied_overrides": curriculum_applied,
        },
        "env_ids": env_ids,
        "scene_weights": scene_weights,
        "track_dir": track_dir,
        "track_geometry_summary": track_geometry.scene_summary(),
        "observation": {
            "image_shape": [6, obs_size, obs_size],
            "state_dim": 12,
            "state_features": [
                "v_long_norm",
                "yaw_rate_norm",
                "accel_x_norm",
                "prev_steer_exec",
                "prev_throttle_exec",
                "steer_core",
                "bias_smooth",
                "obstacle_present",
                "obstacle_longitudinal_norm",
                "obstacle_lateral_norm",
                "obstacle_dist_norm",
                "obstacle_risk",
            ],
        },
        "obstacle_runtime": {
            "enabled": obstacle_enabled,
            "count": obstacle_count,
            "free_prob": obstacle_free_prob,
            "modes": list(obstacle_modes or ["static", "jitter"]),
            "ws_free_prob": ws_obstacle_free_prob,
            "spawn_ahead_min_m": obstacle_spawn_ahead_min_m,
            "spawn_ahead_max_m": obstacle_spawn_ahead_max_m,
            "min_agent_planar_dist_m": obstacle_min_agent_planar_dist_m,
            "min_agent_arc_dist_m": obstacle_min_agent_arc_dist_m,
            "min_separation_world": obstacle_min_separation_world,
            "fixed_progress_ratio": obstacle_fixed_progress_ratio,
            "fixed_progress_gap": obstacle_fixed_progress_gap,
            "fixed_progress_gap_min": obstacle_fixed_progress_gap_min,
            "fixed_progress_gap_max": obstacle_fixed_progress_gap_max,
            "progress_min": obstacle_progress_min,
            "progress_max": obstacle_progress_max,
            "fixed_lateral_ratio": obstacle_fixed_lateral_ratio,
            "gt_start_exclusion_half_width_m": gt_obstacle_start_exclusion_half_width_m,
            "ws_modes": list(ws_obstacle_modes) if ws_obstacle_modes else None,
            "ws_fixed_progress_ratio": ws_obstacle_fixed_progress_ratio,
            "ws_progress_min": ws_obstacle_progress_min,
            "ws_progress_max": ws_obstacle_progress_max,
            "ws_fixed_lateral_ratio": ws_obstacle_fixed_lateral_ratio,
            "randomize_non_lane_pid_yaw": obstacle_randomize_non_lane_pid_yaw,
            "lane_pid_speed_gt": obstacle_lane_pid_speed_gt,
            "lane_pid_speed_ws": obstacle_lane_pid_speed_ws,
            "lane_pid_lookahead_m": obstacle_lane_pid_lookahead_m,
            "ego_random_spawn": ego_random_spawn,
            "ego_spawn_lateral_ratio": ego_spawn_lateral_ratio,
        },
        "ppo": {
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "n_steps": ppo_n_steps,
            "batch_size": ppo_batch_size,
            "n_epochs": ppo_n_epochs,
            "clip_range": ppo_clip_range,
            "target_kl": target_kl,
            "lstm_hidden_size": lstm_hidden_size,
            "lstm_layers": lstm_layers,
        },
        "reward": {
            "progress_reward_scale": progress_reward_scale,
            "survival_reward_scale": survival_reward_scale,
            "w_near_collision": w_near_collision,
            "near_collision_start_ratio": near_collision_start_ratio,
            "overtake_success_bonus": overtake_success_bonus,
            "w_center": w_center,
            "w_heading": w_heading,
        },
        "seed": seed,
        "exp_tag": exp_tag,
        "interrupted": bool(interrupted),
        "training_time_hours": elapsed / 3600.0,
        "trained_timesteps": int(trained_timesteps),
        "num_timesteps_total": int(model_end_timesteps),
        "final_model_zip": final_model_path + ".zip",
        "final_model_pth": final_pth_path,
        "callback_summaries": callback_summaries,
        "extra_run_metadata": dict(extra_run_metadata or {}),
    }
    config_path = os.path.join(save_dir, str(config_filename))
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 76)
    print("✅ V16 training finished")
    print("=" * 76)
    print(f"elapsed: {elapsed / 3600.0:.2f} h")
    print(f"model: {final_model_path}.zip")
    print(f"policy: {final_pth_path}")
    print(f"config: {config_path}")

    env.close()
    return {
        "trained_timesteps": int(trained_timesteps),
        "num_timesteps_total": int(model_end_timesteps),
        "final_model_zip": final_model_path + ".zip",
        "final_model_pth": final_pth_path,
        "config_path": config_path,
        "interrupted": bool(interrupted),
        "curriculum_phase": curriculum_phase,
        "curriculum_applied": dict(curriculum_applied),
        "callback_summaries": callback_summaries,
        "extra_run_metadata": dict(extra_run_metadata or {}),
    }


def train_v16_auto_curriculum(
    total_timesteps: int = 2_000_000,
    save_dir: str = "models/v16_multi_scene_obstacle",
    exp_tag: Optional[str] = None,
    resume_latest: bool = False,
    resume_path: Optional[str] = None,
    auto_curriculum_start_stage: Optional[str] = None,
    **train_kwargs: Any,
) -> Dict[str, Any]:
    total_requested_timesteps = max(0, int(total_timesteps))
    remaining_timesteps = int(total_requested_timesteps)
    os.makedirs(save_dir, exist_ok=True)

    # 计算跳过的起始 stage 索引
    start_stage_idx = 0
    if auto_curriculum_start_stage:
        # 支持 stage 名称或 alias（如 "avoid_mixed" / "stage3"）
        resolved_start = CURRICULUM_PHASE_ALIASES.get(
            str(auto_curriculum_start_stage).strip().lower(),
            str(auto_curriculum_start_stage).strip().lower(),
        )
        for _i, _s in enumerate(AUTO_CURRICULUM_STAGES):
            if _s["stage_name"] == resolved_start or _s["phase"] == resolved_start:
                start_stage_idx = _i
                break
        else:
            available = [s["stage_name"] for s in AUTO_CURRICULUM_STAGES]
            raise ValueError(
                f"Unknown auto-curriculum start stage: '{auto_curriculum_start_stage}'. "
                f"Available: {available}"
            )

    print("\n" + "=" * 76)
    print("🎓 Start V16 Auto Curriculum")
    print("=" * 76)
    print(f"save_dir: {save_dir}")
    print(f"requested_steps: {total_requested_timesteps}")
    if start_stage_idx > 0:
        skipped = [AUTO_CURRICULUM_STAGES[i]["stage_name"] for i in range(start_stage_idx)]
        print(f"⏭️  Skipping stages: {skipped} → starting at '{AUTO_CURRICULUM_STAGES[start_stage_idx]['stage_name']}'")

    stage_results: List[Dict[str, Any]] = []
    stage_resume_latest = bool(resume_latest)
    stage_resume_path = resume_path

    for stage_idx, stage_def in enumerate(AUTO_CURRICULUM_STAGES):
        if stage_idx < start_stage_idx:
            continue
        if remaining_timesteps <= 0:
            break

        stage_name = str(stage_def["stage_name"])
        stage_phase = str(stage_def["phase"])
        is_final_stage = stage_idx == (len(AUTO_CURRICULUM_STAGES) - 1)
        stage_max_timesteps = int(stage_def.get("max_stage_timesteps", remaining_timesteps))
        stage_budget_timesteps = (
            int(remaining_timesteps)
            if is_final_stage else int(min(remaining_timesteps, stage_max_timesteps))
        )
        if stage_budget_timesteps <= 0:
            break

        stage_exp_tag = f"{exp_tag}_{stage_name}" if exp_tag else stage_name
        stage_callbacks: List[BaseCallback] = []
        gate_callback: Optional[CurriculumWindowAdvanceCallback] = None
        if not is_final_stage:
            gate_callback = CurriculumWindowAdvanceCallback(
                stage_name=stage_name,
                required_logging_keys=list(stage_def["required_logging_keys"]),
                min_stage_timesteps=int(stage_def["min_stage_timesteps"]),
                recent_episodes=int(stage_def.get("recent_episodes", 20)),
                min_success_episodes=int(stage_def.get("min_success_episodes", 12)),
                min_soft_laps=float(stage_def.get("min_soft_laps", 2.0)),
                max_collision_rate_by_key=stage_def.get("max_collision_rate_by_key"),
                max_stage_timesteps=int(stage_def["max_stage_timesteps"]),
                save_dir=save_dir,
                exp_tag=stage_exp_tag,
                filename="curriculum_window.jsonl",
                verbose=1,
            )
            stage_callbacks.append(gate_callback)

        stage_metadata = {
            "auto_curriculum": {
                "enabled": True,
                "stage_name": stage_name,
                "stage_index": int(stage_idx),
                "stage_phase": stage_phase,
                "stage_budget_timesteps": int(stage_budget_timesteps),
                "stage_max_timesteps": (
                    None if is_final_stage else int(stage_max_timesteps)
                ),
                "remaining_before_stage": int(remaining_timesteps),
                "total_requested_timesteps": int(total_requested_timesteps),
            }
        }

        print("\n" + "-" * 76)
        print(
            f"🪜 Auto stage {stage_idx + 1}/{len(AUTO_CURRICULUM_STAGES)}: "
            f"{stage_name} (phase={stage_phase}, budget={stage_budget_timesteps})"
        )
        print("-" * 76)

        stage_train_kwargs = dict(train_kwargs)
        if stage_idx > 0:
            stage_train_kwargs["run_preflight_checks"] = False

        stage_result = train_v16(
            total_timesteps=stage_budget_timesteps,
            save_dir=save_dir,
            exp_tag=stage_exp_tag,
            curriculum_phase=stage_phase,
            resume_latest=stage_resume_latest,
            resume_path=stage_resume_path,
            extra_callbacks=stage_callbacks,
            extra_run_metadata=stage_metadata,
            config_filename=f"v16_config_{stage_name}.json",
            **stage_train_kwargs,
        )

        trained_timesteps = int(stage_result.get("trained_timesteps", 0) or 0)
        remaining_timesteps = max(0, int(remaining_timesteps) - trained_timesteps)
        gate_summary = gate_callback.summary() if gate_callback is not None else None

        stage_results.append(
            {
                "stage_name": stage_name,
                "phase": stage_phase,
                "stage_index": int(stage_idx),
                "budget_timesteps": int(stage_budget_timesteps),
                "trained_timesteps": int(trained_timesteps),
                "remaining_after_stage": int(remaining_timesteps),
                "interrupted": bool(stage_result.get("interrupted", False)),
                "config_path": stage_result.get("config_path"),
                "final_model_zip": stage_result.get("final_model_zip"),
                "final_model_pth": stage_result.get("final_model_pth"),
                "callback_summaries": stage_result.get("callback_summaries", {}),
                "gate_summary": gate_summary,
            }
        )

        if bool(stage_result.get("interrupted", False)):
            print(f"⚠️  auto curriculum stopped early at stage={stage_name} due to interrupt")
            break

        next_resume_path = stage_result.get("final_model_zip")
        if next_resume_path:
            stage_resume_latest = False
            stage_resume_path = str(next_resume_path)
        else:
            stage_resume_latest = True
            stage_resume_path = None

    total_trained_timesteps = int(sum(stage["trained_timesteps"] for stage in stage_results))
    auto_summary = {
        "version": "V16_AUTO_CURRICULUM",
        "timestamp": datetime.now().isoformat(),
        "save_dir": save_dir,
        "exp_tag": exp_tag,
        "total_requested_timesteps": int(total_requested_timesteps),
        "total_trained_timesteps": int(total_trained_timesteps),
        "remaining_timesteps": int(max(0, total_requested_timesteps - total_trained_timesteps)),
        "stages": stage_results,
    }
    auto_summary_path = os.path.join(save_dir, "v16_auto_curriculum_summary.json")
    with open(auto_summary_path, "w", encoding="utf-8") as f:
        json.dump(auto_summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 76)
    print("✅ V16 auto curriculum finished")
    print("=" * 76)
    print(f"summary: {auto_summary_path}")
    print(f"trained: {total_trained_timesteps}/{total_requested_timesteps}")

    auto_summary["summary_path"] = auto_summary_path
    return auto_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V16: dual-domain recurrent PPO with obstacle runtime",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-ids", nargs="+", default=None)
    parser.add_argument("--scene-weights", nargs="+", type=float, default=None)
    parser.add_argument("--track-dir", type=str, default=DEFAULT_TRACK_DIR)
    parser.add_argument("--sim", type=str, default="remote")
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--save-dir", type=str, default="models/v16_multi_scene_obstacle")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument(
        "--curriculum-phase",
        type=str,
        default=None,
        choices=["none"] + sorted(CURRICULUM_PHASES.keys()) + sorted(CURRICULUM_PHASE_ALIASES.keys()),
        help="Apply a built-in training curriculum phase; explicit non-default flags still take precedence.",
    )
    parser.add_argument(
        "--auto-curriculum",
        action="store_true",
        default=False,
        help="Run the built-in window-gated multi-stage curriculum automatically.",
    )
    parser.add_argument(
        "--auto-curriculum-start-stage",
        type=str,
        default=None,
        help=(
            "Skip earlier stages and start auto-curriculum from this stage. "
            "Accepts stage name (e.g. 'avoid_mixed') or alias (e.g. 'stage3'). "
            "Requires --auto-curriculum and --resume-path to a checkpoint."
        ),
    )
    parser.add_argument("--obs-size", type=int, default=128)
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--yellow-dropout-prob", type=float, default=0.20)
    parser.add_argument("--lstm-hidden-size", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--ppo-n-steps", type=int, default=4096)
    parser.add_argument("--ppo-batch-size", type=int, default=256)
    parser.add_argument("--ppo-n-epochs", type=int, default=4)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--w-near-collision", type=float, default=0.24)
    parser.add_argument("--overtake-success-bonus", type=float, default=2.5)
    parser.add_argument("--w-center", type=float, default=0.03)
    parser.add_argument("--w-heading", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--exp-tag", type=str, default=None)
    parser.add_argument("--resume-latest", action="store_true", default=False)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--disable-preflight-checks", action="store_true", default=False)
    parser.add_argument("--disable-file-metrics-log", action="store_true", default=False)
    parser.add_argument("--file-metrics-log-freq", type=int, default=500)
    parser.add_argument("--file-metrics-log-name", type=str, default="train_metrics.jsonl")
    parser.add_argument("--disable-auto-lr-decay", action="store_true", default=False)

    parser.add_argument("--disable-obstacles", action="store_true", default=False)
    parser.add_argument("--obstacle-count", type=int, default=2)
    parser.add_argument("--obstacle-free-prob", type=float, default=0.15)
    parser.add_argument("--obstacle-modes", nargs="+", default=None)
    parser.add_argument("--obstacle-spawn-ahead-min-m", type=float, default=3.5)
    parser.add_argument("--obstacle-spawn-ahead-max-m", type=float, default=14.0)
    parser.add_argument("--obstacle-min-agent-planar-dist-m", type=float, default=1.5)
    parser.add_argument("--obstacle-min-agent-arc-dist-m", type=float, default=3.5)
    parser.add_argument("--obstacle-min-separation-world", type=float, default=3.0)
    parser.add_argument("--obstacle-fixed-progress-ratio", type=float, default=None)
    parser.add_argument("--obstacle-fixed-progress-gap", type=float, default=None)
    parser.add_argument("--obstacle-progress-min", type=float, default=None)
    parser.add_argument("--obstacle-progress-max", type=float, default=None)
    parser.add_argument("--obstacle-fixed-lateral-ratio", type=float, default=None)
    parser.add_argument("--disable-obstacle-random-yaw", action="store_true", default=False)
    parser.add_argument("--obstacle-lane-pid-speed-gt", type=float, default=0.85)
    parser.add_argument("--obstacle-lane-pid-speed-ws", type=float, default=0.70)
    parser.add_argument("--obstacle-lane-pid-lookahead-m", type=float, default=0.9)
    parser.add_argument("--obstacle-jitter-amplitude-m", type=float, default=0.10)
    parser.add_argument("--obstacle-jitter-period-s", type=float, default=1.5)
    parser.add_argument("--obstacle-jitter-update-hz", type=float, default=8.0)
    parser.add_argument("--obstacle-nudge-amplitude-m", type=float, default=0.14)
    parser.add_argument("--obstacle-nudge-period-s", type=float, default=1.5)
    parser.add_argument("--obstacle-nudge-update-hz", type=float, default=8.0)
    parser.add_argument("--obstacle-seed", type=int, default=None)
    parser.add_argument("--ego-random-spawn", action="store_true", default=False)
    parser.add_argument("--ego-spawn-lateral-ratio", type=float, default=0.5)
    parser.add_argument(
        "--sim2real-json", type=str, default=None,
        help="Path to dynamics_alignment_wm.json for sim2real throttle/steer scaling",
    )

    args = parser.parse_args()

    manual_curriculum_phase = args.curriculum_phase
    if manual_curriculum_phase == "none":
        manual_curriculum_phase = None
    if args.auto_curriculum and manual_curriculum_phase is not None:
        raise ValueError("--auto-curriculum cannot be combined with --curriculum-phase")

    common_kwargs = dict(
        env_ids=args.env_ids,
        scene_weights=args.scene_weights,
        track_dir=args.track_dir,
        sim_path=args.sim,
        total_timesteps=args.steps,
        save_dir=args.save_dir,
        port=args.port,
        obs_size=args.obs_size,
        augment=args.augment,
        yellow_dropout_prob=args.yellow_dropout_prob,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_layers=args.lstm_layers,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        ppo_n_steps=args.ppo_n_steps,
        ppo_batch_size=args.ppo_batch_size,
        ppo_n_epochs=args.ppo_n_epochs,
        target_kl=args.target_kl,
        w_near_collision=args.w_near_collision,
        overtake_success_bonus=args.overtake_success_bonus,
        w_center=args.w_center,
        w_heading=args.w_heading,
        obstacle_enabled=(not args.disable_obstacles),
        obstacle_count=args.obstacle_count,
        obstacle_free_prob=args.obstacle_free_prob,
        obstacle_modes=args.obstacle_modes,
        obstacle_spawn_ahead_min_m=args.obstacle_spawn_ahead_min_m,
        obstacle_spawn_ahead_max_m=args.obstacle_spawn_ahead_max_m,
        obstacle_min_agent_planar_dist_m=args.obstacle_min_agent_planar_dist_m,
        obstacle_min_agent_arc_dist_m=args.obstacle_min_agent_arc_dist_m,
        obstacle_min_separation_world=args.obstacle_min_separation_world,
        obstacle_fixed_progress_ratio=args.obstacle_fixed_progress_ratio,
        obstacle_fixed_progress_gap=args.obstacle_fixed_progress_gap,
        obstacle_progress_min=args.obstacle_progress_min,
        obstacle_progress_max=args.obstacle_progress_max,
        obstacle_fixed_lateral_ratio=args.obstacle_fixed_lateral_ratio,
        obstacle_randomize_non_lane_pid_yaw=(not args.disable_obstacle_random_yaw),
        obstacle_lane_pid_speed_gt=args.obstacle_lane_pid_speed_gt,
        obstacle_lane_pid_speed_ws=args.obstacle_lane_pid_speed_ws,
        obstacle_lane_pid_lookahead_m=args.obstacle_lane_pid_lookahead_m,
        obstacle_jitter_amplitude_m=args.obstacle_jitter_amplitude_m,
        obstacle_jitter_period_s=args.obstacle_jitter_period_s,
        obstacle_jitter_update_hz=args.obstacle_jitter_update_hz,
        obstacle_nudge_amplitude_m=args.obstacle_nudge_amplitude_m,
        obstacle_nudge_period_s=args.obstacle_nudge_period_s,
        obstacle_nudge_update_hz=args.obstacle_nudge_update_hz,
        obstacle_seed=args.obstacle_seed,
        ego_random_spawn=args.ego_random_spawn,
        ego_spawn_lateral_ratio=args.ego_spawn_lateral_ratio,
        sim2real_json=args.sim2real_json,
        seed=args.seed,
        exp_tag=args.exp_tag,
        resume_latest=args.resume_latest,
        resume_path=args.resume_path,
        run_preflight_checks=(not args.disable_preflight_checks),
        enable_file_metrics_log=(not args.disable_file_metrics_log),
        file_metrics_log_freq=args.file_metrics_log_freq,
        file_metrics_log_name=args.file_metrics_log_name,
        enable_auto_lr_decay=(not args.disable_auto_lr_decay),
    )
    if args.auto_curriculum:
        train_v16_auto_curriculum(
            auto_curriculum_start_stage=args.auto_curriculum_start_stage,
            **common_kwargs,
        )
    else:
        train_v16(curriculum_phase=manual_curriculum_phase, **common_kwargs)


if __name__ == "__main__":
    main()
