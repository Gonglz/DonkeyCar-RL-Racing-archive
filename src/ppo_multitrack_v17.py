#!/usr/bin/env python3
"""
DonkeyCar PPO V17

- Dual-domain recurrent PPO
- 5ch semantic image + 7D core state + canonical LiDAR/target token + 2D lidar meta
- LiDAR-first policy with optional critic-only dual value heads
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
_repo_root = str(REPO_ROOT)
while _repo_root in sys.path:
    sys.path.remove(_repo_root)
sys.path.insert(0, _repo_root)

import gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from sb3_contrib import RecurrentPPO
    from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer, RecurrentRolloutBuffer
    from sb3_contrib.common.recurrent.type_aliases import RNNStates
except Exception:
    RecurrentPPO = None
    RecurrentDictRolloutBuffer = None
    RecurrentRolloutBuffer = None
    RNNStates = None

from module.action_adapter import ActionAdapterWrapper
from module.callbacks import (
    AdaptiveLearningRateCallback,
    ActionDiagnosticsCallback,
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
from module.control import ActionSafetyWrapper
from module.lidar import CanonicalLidarSpec
from module.reward import DonkeyRewardWrapper
from module.track import TrackGeometryManager
from module.v17_callbacks import CriticCalibrationCallback
from module.v17_env import MultiSceneEnvV17, V17ObsWrapper
from module.v17_policy import LiDARFiLMFeatureExtractor, V17RecurrentMultiInputPolicy
from module.utils import _find_latest_checkpoint, _safe_seed_env, _seed_everything, load_config
from module.obv import CanonicalSemanticWrapper
from src.ppo_multitrack_v16 import (
    AUTO_CURRICULUM_STAGES,
    CURRICULUM_PHASES,
    CURRICULUM_PHASE_ALIASES,
    CurriculumWindowAdvanceCallback,
    DEFAULT_ENV_IDS,
    DEFAULT_MYCONFIG,
    DEFAULT_TRACK_DIR,
    SCENE_SPECS,
    TRAIN_V16_DEFAULTS,
    _install_sim_wait_timeout_patch,
    _probe_sim_tcp,
    _resolve_track_dir,
    run_offline_track_checks,
)

_V16_AUTO_CURRICULUM_STAGES = AUTO_CURRICULUM_STAGES
_V16_CURRICULUM_PHASES = CURRICULUM_PHASES
_V16_CURRICULUM_PHASE_ALIASES = CURRICULUM_PHASE_ALIASES
_V17_WS_EDGE_OBSTACLE_LATERALS = [0.0, 1.0]

SCENE_SPECS = {env_id: dict(spec) for env_id, spec in SCENE_SPECS.items()}
if "donkey-waveshare-v0" in SCENE_SPECS:
    _ws_spec = dict(SCENE_SPECS["donkey-waveshare-v0"])
    _ws_reward_overrides = dict(_ws_spec.get("reward_overrides", {}) or {})
    _ws_reward_overrides.update(
        {
            # V17 PID logs showed WS can still make close passes profitable:
            # 2-lap collision episodes kept large positive reward. Push the
            # model toward wider clearance while keeping offtrack cheap enough
            # to use the line when that is safer than brushing the obstacle.
            "w_near_offtrack": 0.22,
            "near_offtrack_start_ratio": 0.74,
            "collision_penalty_base": 72.0,
            "w_near_collision": 1.20,
            "near_collision_start_ratio": 0.58,
            "safe_follow_bonus_scale": 0.08,
            "progress_reward_scale": 46.0,
            "survival_reward_scale": 0.30,
        }
    )
    _ws_spec["reward_overrides"] = _ws_reward_overrides
    SCENE_SPECS["donkey-waveshare-v0"] = _ws_spec

CURRICULUM_PHASES = {name: dict(values) for name, values in _V16_CURRICULUM_PHASES.items()}
CURRICULUM_PHASES["warmup"]["obstacle_fixed_progress_ratio"] = None
CURRICULUM_PHASES["warmup"]["obstacle_fixed_progress_distribution"] = None
CURRICULUM_PHASES["warmup"].update(
    {
        "obstacle_spawn_ahead_min_m": 2.0,
        "obstacle_spawn_ahead_max_m": 6.0,
        "w_d": 0.03,
        "w_dd": 0.005,
        "w_sat": 0.04,
        "w_steer_budget": 0.0,
        "w_sign_flip": 0.0,
        "w_micro_wiggle": 0.0,
        "ws_obstacle_lateral_choices": _V17_WS_EDGE_OBSTACLE_LATERALS,
    }
)
CURRICULUM_PHASES["warmup_a"].update(
    {
        # Keep V16's WS no-obstacle coverage in early training; otherwise a
        # fresh policy sees WS obstacles before it has recovered basic driving.
        "ws_obstacle_free_prob": 0.50,
        "obstacle_fixed_progress_ratio": None,
        "obstacle_fixed_progress_distribution": None,
        "obstacle_spawn_ahead_min_m": 2.0,
        "obstacle_spawn_ahead_max_m": 6.0,
        "ws_obstacle_fixed_progress_ratio": None,
        "ws_obstacle_lateral_choices": _V17_WS_EDGE_OBSTACLE_LATERALS,
        # Current WS warmup can exploit large alternating steering to survive.
        # Keep this as reward shaping so the policy learns smoother actions;
        # do not add a snake-specific hard output clamp at this stage.
        "steer_delta_delta_max": None,
        "w_d": 0.05,
        "w_dd": 0.015,
        "w_sat": 0.08,
        "w_steer_budget": 0.0,
        "w_sign_flip": 0.0,
        "sign_flip_min_abs_steer": 0.18,
        "w_micro_wiggle": 0.0,
        "micro_wiggle_min_abs_steer": 0.035,
        "micro_wiggle_max_abs_steer": 0.22,
    }
)
CURRICULUM_PHASE_ALIASES = dict(_V16_CURRICULUM_PHASE_ALIASES)
AUTO_CURRICULUM_STAGES = tuple(_V16_AUTO_CURRICULUM_STAGES)

# V17 的障碍阶段日志显示 steer_clip_hit 在 avoid_static 后开始明显升高；
# PID full 直接用 0.70m/s 又让追车相对速度过小。这里把 PID 拆成三档，
# 但不收紧转向执行层，只用奖励项惩罚长期大舵角/限幅冲突。
_V17_AVOID_REWARD_SHAPING = {
    "w_d": 0.05,
    "w_dd": 0.015,
    "w_m": 0.015,
    "w_sat": 0.08,
    "w_steer_budget": 0.0,
    "w_sign_flip": 0.0,
    "w_micro_wiggle": 0.0,
    "collision_penalty_base": 12.0,
    "offtrack_penalty_base": 10.0,
    "terminal_offtrack_progress_scale": 0.0,
}
_V17_PID_REWARD_SHAPING = {
    "w_d": 0.06,
    "w_dd": 0.020,
    "w_m": 0.030,
    "w_sat": 0.10,
    "w_steer_budget": 0.0,
    "w_sign_flip": 0.0,
    "w_micro_wiggle": 0.0,
    "micro_wiggle_min_abs_steer": 0.035,
    "micro_wiggle_max_abs_steer": 0.22,
}
_V17_OVERTAKE_LATERAL_CLEARANCE_M = 0.30
_V17_OVERTAKE_CAPSULE_LONGITUDINAL_M = 0.50
_V17_OBSTACLE_CLEARANCE_INNER_M = 0.30
_V17_OBSTACLE_CLEARANCE_OUTER_M = 0.60
_V17_POST_PASS_WATCH_LONGITUDINAL_M = 1.20
_V17_POST_PASS_WATCH_STEPS = 18
_V17_OVERTAKE_SUCCESS_MIN_PROGRESS_RATIO = 1e-6
_V17_WS_FOLLOW_MIN_M = 0.80
_V17_WS_FOLLOW_MAX_M = 1.80
_V17_WS_WAIT_MAX_M = 2.20
_V17_WS_UNSAFE_GAP_M = 0.55
_V17_WS_PLANAR_MAX_M = 2.60
_V17_WS_ARM_LONGITUDINAL_M = 0.45
_V17_GT_FOLLOW_MIN_M = 1.00
_V17_GT_FOLLOW_MAX_M = 1.40
_V17_GT_WAIT_MAX_M = 3.00
_V17_GT_UNSAFE_GAP_M = 0.90
_V17_GT_PLANAR_MAX_M = 5.00
_V17_GT_ARM_LONGITUDINAL_M = 0.70
_V17_PID_OVERTAKE_INTRO = {
    "w_micro_wiggle": 0.0,
    "collision_penalty_base": 58.0,
    "offtrack_penalty_base": 12.0,
    "terminal_offtrack_progress_scale": 0.0,
    "w_near_collision": 1.28,
    "near_collision_start_ratio": 0.62,
    "overtake_success_bonus": 0.50,
    "reward_safe_follow_bonus": 0.42,
    "reward_prepare_pass_bonus": 0.0,
    "reward_commit_pass_bonus": 0.0,
    "reward_post_pass_bonus": 0.0,
    "reward_overrides_by_logging_key": {
        "ws": {
            "progress_reward_scale": 38.0,
            "survival_reward_scale": 0.20,
            "w_speed_ref": 0.14,
            "speed_ref_vmin": 0.24,
            "speed_ref_vmax": 0.62,
            "collision_episode_reward_cap": 56.0,
            "offtrack_episode_reward_cap": 70.0,
            "offtrack_leniency_ratio": 0.14,
            "offtrack_leniency_mult": 1.8,
            "safe_follow_bonus_scale": 0.30,
            "safe_follow_min_m": _V17_WS_FOLLOW_MIN_M,
            "safe_follow_max_m": _V17_WS_FOLLOW_MAX_M,
            "safe_follow_risk_max": 0.36,
            "safe_follow_ttc_min_s": 1.8,
            "safe_follow_ttc_max_s": 5.2,
            "wait_window_bonus_scale": 0.35,
            "wait_window_min_gap_m": _V17_WS_FOLLOW_MIN_M,
            "wait_window_max_gap_m": _V17_WS_WAIT_MAX_M,
            "wait_window_max_closing_rate": 0.12,
            "force_pass_penalty_scale": 1.18,
            "unsafe_close_penalty_scale": 0.95,
            "obstacle_clearance_penalty_scale": 0.90,
            "obstacle_clearance_inner_m": _V17_OBSTACLE_CLEARANCE_INNER_M,
            "obstacle_clearance_outer_m": _V17_OBSTACLE_CLEARANCE_OUTER_M,
            "post_pass_cut_in_penalty_scale": 1.10,
            "post_pass_watch_longitudinal_m": _V17_POST_PASS_WATCH_LONGITUDINAL_M,
            "post_pass_watch_steps": _V17_POST_PASS_WATCH_STEPS,
            "overtake_success_min_progress_ratio": _V17_OVERTAKE_SUCCESS_MIN_PROGRESS_RATIO,
            "unsafe_close_gap_m": _V17_WS_UNSAFE_GAP_M,
            "unsafe_close_clearance_m": _V17_OVERTAKE_LATERAL_CLEARANCE_M,
            "unsafe_close_longitudinal_m": _V17_OVERTAKE_CAPSULE_LONGITUDINAL_M,
            "lateral_overlap_ref_m": _V17_OVERTAKE_LATERAL_CLEARANCE_M,
            "unsafe_close_ttc_s": 2.2,
            "overtake_arm_longitudinal_min_m": _V17_WS_ARM_LONGITUDINAL_M,
            "overtake_arm_planar_max_m": _V17_WS_PLANAR_MAX_M,
            "overtake_pass_longitudinal_threshold_m": -_V17_WS_ARM_LONGITUDINAL_M,
            "overtake_pass_planar_min_m": _V17_OBSTACLE_CLEARANCE_OUTER_M,
            "close_front_planar_max_m": _V17_WS_PLANAR_MAX_M,
            "force_pass_planar_max_m": _V17_WS_PLANAR_MAX_M,
        },
        "gt": {
            "progress_reward_scale": 44.0,
            "survival_reward_scale": 0.20,
            "w_speed_ref": 0.07,
            "speed_ref_vmin": 0.28,
            "speed_ref_vmax": 0.72,
            "collision_episode_reward_cap": 82.0,
            "offtrack_episode_reward_cap": 75.0,
            "offtrack_leniency_ratio": 0.14,
            "offtrack_leniency_mult": 1.8,
            "safe_follow_bonus_scale": 0.25,
            "safe_follow_min_m": _V17_GT_FOLLOW_MIN_M,
            "safe_follow_max_m": _V17_GT_FOLLOW_MAX_M,
            "safe_follow_risk_max": 0.38,
            "safe_follow_ttc_min_s": 1.6,
            "safe_follow_ttc_max_s": 5.0,
            "wait_window_bonus_scale": 0.30,
            "wait_window_min_gap_m": _V17_GT_FOLLOW_MIN_M,
            "wait_window_max_gap_m": _V17_GT_WAIT_MAX_M,
            "wait_window_max_closing_rate": 0.14,
            "force_pass_penalty_scale": 0.86,
            "unsafe_close_penalty_scale": 0.75,
            "obstacle_clearance_penalty_scale": 0.70,
            "obstacle_clearance_inner_m": _V17_OBSTACLE_CLEARANCE_INNER_M,
            "obstacle_clearance_outer_m": _V17_OBSTACLE_CLEARANCE_OUTER_M,
            "post_pass_cut_in_penalty_scale": 0.85,
            "post_pass_watch_longitudinal_m": _V17_POST_PASS_WATCH_LONGITUDINAL_M,
            "post_pass_watch_steps": _V17_POST_PASS_WATCH_STEPS,
            "overtake_success_min_progress_ratio": _V17_OVERTAKE_SUCCESS_MIN_PROGRESS_RATIO,
            "unsafe_close_gap_m": _V17_GT_UNSAFE_GAP_M,
            "unsafe_close_clearance_m": _V17_OVERTAKE_LATERAL_CLEARANCE_M,
            "unsafe_close_longitudinal_m": _V17_OVERTAKE_CAPSULE_LONGITUDINAL_M,
            "lateral_overlap_ref_m": _V17_OVERTAKE_LATERAL_CLEARANCE_M,
            "unsafe_close_ttc_s": 2.2,
            "overtake_arm_longitudinal_min_m": _V17_GT_ARM_LONGITUDINAL_M,
            "overtake_arm_planar_max_m": _V17_GT_PLANAR_MAX_M,
            "overtake_pass_longitudinal_threshold_m": -_V17_GT_ARM_LONGITUDINAL_M,
            "overtake_pass_planar_min_m": 0.70,
            "close_front_planar_max_m": _V17_GT_PLANAR_MAX_M,
            "force_pass_planar_max_m": _V17_GT_PLANAR_MAX_M,
        },
    },
}
_V17_PID_OVERTAKE_MID = {
    "w_d": 0.09,
    "w_dd": 0.035,
    "w_m": 0.050,
    "w_sat": 0.15,
    "w_steer_budget": 0.0,
    "w_sign_flip": 0.0,
    "w_micro_wiggle": 0.0,
    "collision_penalty_base": 58.0,
    "offtrack_penalty_base": 18.0,
    "terminal_offtrack_progress_scale": 0.0,
    "w_near_collision": 0.92,
    "near_collision_start_ratio": 0.60,
    "overtake_success_bonus": 0.80,
    "reward_safe_follow_bonus": 0.28,
    "reward_prepare_pass_bonus": 0.0,
    "reward_commit_pass_bonus": 0.0,
    "reward_post_pass_bonus": 0.0,
    "reward_overrides_by_logging_key": {
        "ws": {
            "progress_reward_scale": 36.0,
            "survival_reward_scale": 0.20,
            "w_speed_ref": 0.09,
            "speed_ref_vmin": 0.32,
            "speed_ref_vmax": 0.85,
            "collision_episode_reward_cap": 62.0,
            "offtrack_episode_reward_cap": 105.0,
            "offtrack_leniency_ratio": 0.10,
            "offtrack_leniency_mult": 1.7,
            "safe_follow_bonus_scale": 0.42,
            "safe_follow_min_m": _V17_WS_FOLLOW_MIN_M,
            "safe_follow_max_m": _V17_WS_FOLLOW_MAX_M,
            "safe_follow_risk_max": 0.36,
            "safe_follow_ttc_min_s": 1.5,
            "safe_follow_ttc_max_s": 4.4,
            "wait_window_bonus_scale": 0.54,
            "wait_window_min_gap_m": _V17_WS_FOLLOW_MIN_M,
            "wait_window_max_gap_m": _V17_WS_WAIT_MAX_M,
            "wait_window_max_closing_rate": 0.16,
            "force_pass_penalty_scale": 0.95,
            "unsafe_close_penalty_scale": 1.15,
            "obstacle_clearance_penalty_scale": 0.80,
            "obstacle_clearance_inner_m": _V17_OBSTACLE_CLEARANCE_INNER_M,
            "obstacle_clearance_outer_m": _V17_OBSTACLE_CLEARANCE_OUTER_M,
            "post_pass_cut_in_penalty_scale": 1.00,
            "post_pass_watch_longitudinal_m": _V17_POST_PASS_WATCH_LONGITUDINAL_M,
            "post_pass_watch_steps": _V17_POST_PASS_WATCH_STEPS,
            "overtake_success_min_progress_ratio": _V17_OVERTAKE_SUCCESS_MIN_PROGRESS_RATIO,
            "unsafe_close_gap_m": _V17_WS_UNSAFE_GAP_M,
            "unsafe_close_clearance_m": _V17_OVERTAKE_LATERAL_CLEARANCE_M,
            "unsafe_close_longitudinal_m": _V17_OVERTAKE_CAPSULE_LONGITUDINAL_M,
            "lateral_overlap_ref_m": _V17_OVERTAKE_LATERAL_CLEARANCE_M,
            "unsafe_close_ttc_s": 2.2,
            "overtake_arm_longitudinal_min_m": _V17_WS_ARM_LONGITUDINAL_M,
            "overtake_arm_planar_max_m": _V17_WS_PLANAR_MAX_M,
            "overtake_pass_longitudinal_threshold_m": -_V17_WS_ARM_LONGITUDINAL_M,
            "overtake_pass_planar_min_m": _V17_OBSTACLE_CLEARANCE_OUTER_M,
            "close_front_planar_max_m": _V17_WS_PLANAR_MAX_M,
            "force_pass_planar_max_m": _V17_WS_PLANAR_MAX_M,
        },
        "gt": {
            "progress_reward_scale": 40.0,
            "survival_reward_scale": 0.24,
            "w_speed_ref": 0.05,
            "speed_ref_vmin": 0.34,
            "speed_ref_vmax": 0.95,
            "collision_episode_reward_cap": 82.0,
            "offtrack_episode_reward_cap": 90.0,
            "offtrack_leniency_ratio": 0.10,
            "offtrack_leniency_mult": 1.7,
            "safe_follow_bonus_scale": 0.36,
            "safe_follow_min_m": _V17_GT_FOLLOW_MIN_M,
            "safe_follow_max_m": _V17_GT_FOLLOW_MAX_M,
            "safe_follow_risk_max": 0.38,
            "safe_follow_ttc_min_s": 1.4,
            "safe_follow_ttc_max_s": 4.2,
            "wait_window_bonus_scale": 0.46,
            "wait_window_min_gap_m": _V17_GT_FOLLOW_MIN_M,
            "wait_window_max_gap_m": _V17_GT_WAIT_MAX_M,
            "wait_window_max_closing_rate": 0.18,
            "force_pass_penalty_scale": 0.80,
            "unsafe_close_penalty_scale": 0.95,
            "obstacle_clearance_penalty_scale": 0.65,
            "obstacle_clearance_inner_m": _V17_OBSTACLE_CLEARANCE_INNER_M,
            "obstacle_clearance_outer_m": _V17_OBSTACLE_CLEARANCE_OUTER_M,
            "post_pass_cut_in_penalty_scale": 0.80,
            "post_pass_watch_longitudinal_m": _V17_POST_PASS_WATCH_LONGITUDINAL_M,
            "post_pass_watch_steps": _V17_POST_PASS_WATCH_STEPS,
            "overtake_success_min_progress_ratio": _V17_OVERTAKE_SUCCESS_MIN_PROGRESS_RATIO,
            "unsafe_close_gap_m": _V17_GT_UNSAFE_GAP_M,
            "unsafe_close_clearance_m": _V17_OVERTAKE_LATERAL_CLEARANCE_M,
            "unsafe_close_longitudinal_m": _V17_OVERTAKE_CAPSULE_LONGITUDINAL_M,
            "lateral_overlap_ref_m": _V17_OVERTAKE_LATERAL_CLEARANCE_M,
            "unsafe_close_ttc_s": 2.2,
            "overtake_arm_longitudinal_min_m": _V17_GT_ARM_LONGITUDINAL_M,
            "overtake_arm_planar_max_m": _V17_GT_PLANAR_MAX_M,
            "overtake_pass_longitudinal_threshold_m": -_V17_GT_ARM_LONGITUDINAL_M,
            "overtake_pass_planar_min_m": 0.70,
            "close_front_planar_max_m": _V17_GT_PLANAR_MAX_M,
            "force_pass_planar_max_m": _V17_GT_PLANAR_MAX_M,
        },
    },
}
_V17_PID_OVERTAKE_FULL = {
    "w_d": 0.12,
    "w_dd": 0.050,
    "w_m": 0.060,
    "w_sat": 0.20,
    "w_steer_budget": 0.0,
    "w_sign_flip": 0.0,
    "w_micro_wiggle": 0.0,
    "collision_penalty_base": 76.0,
    "offtrack_penalty_base": 20.0,
    "terminal_offtrack_progress_scale": 0.0,
    "w_near_collision": 1.00,
    "near_collision_start_ratio": 0.58,
    "overtake_success_bonus": 1.20,
    "reward_safe_follow_bonus": 0.33,
    "reward_prepare_pass_bonus": 0.0,
    "reward_commit_pass_bonus": 0.0,
    "reward_post_pass_bonus": 0.0,
    "reward_overrides_by_logging_key": {
        "ws": {
            "collision_penalty_base": 72.0,
            "progress_reward_scale": 40.0,
            "survival_reward_scale": 0.20,
            "w_speed_ref": 0.08,
            "speed_ref_vmin": 0.32,
            "speed_ref_vmax": 1.00,
            "collision_episode_reward_cap": 58.0,
            "offtrack_penalty_base": 4.0,
            "offtrack_episode_reward_cap": 90.0,
            "offtrack_leniency_ratio": 0.10,
            "offtrack_leniency_mult": 1.7,
            "safe_follow_bonus_scale": 0.44,
            "safe_follow_min_m": _V17_WS_FOLLOW_MIN_M,
            "safe_follow_max_m": _V17_WS_FOLLOW_MAX_M,
            "safe_follow_risk_max": 0.36,
            "safe_follow_ttc_min_s": 1.5,
            "safe_follow_ttc_max_s": 4.2,
            "wait_window_bonus_scale": 0.56,
            "wait_window_min_gap_m": _V17_WS_FOLLOW_MIN_M,
            "wait_window_max_gap_m": _V17_WS_WAIT_MAX_M,
            "wait_window_max_closing_rate": 0.18,
            "force_pass_penalty_scale": 1.10,
            "unsafe_close_penalty_scale": 1.35,
            "obstacle_clearance_penalty_scale": 0.85,
            "obstacle_clearance_inner_m": _V17_OBSTACLE_CLEARANCE_INNER_M,
            "obstacle_clearance_outer_m": _V17_OBSTACLE_CLEARANCE_OUTER_M,
            "post_pass_cut_in_penalty_scale": 1.05,
            "post_pass_watch_longitudinal_m": _V17_POST_PASS_WATCH_LONGITUDINAL_M,
            "post_pass_watch_steps": _V17_POST_PASS_WATCH_STEPS,
            "overtake_success_min_progress_ratio": _V17_OVERTAKE_SUCCESS_MIN_PROGRESS_RATIO,
            "unsafe_close_gap_m": 0.75,
            "unsafe_close_clearance_m": _V17_OVERTAKE_LATERAL_CLEARANCE_M,
            "unsafe_close_longitudinal_m": _V17_OVERTAKE_CAPSULE_LONGITUDINAL_M,
            "lateral_overlap_ref_m": _V17_OVERTAKE_LATERAL_CLEARANCE_M,
            "unsafe_close_ttc_s": 2.2,
            "overtake_arm_longitudinal_min_m": _V17_WS_ARM_LONGITUDINAL_M,
            "overtake_arm_planar_max_m": _V17_WS_PLANAR_MAX_M,
            "overtake_pass_longitudinal_threshold_m": -_V17_WS_ARM_LONGITUDINAL_M,
            "overtake_pass_planar_min_m": _V17_OBSTACLE_CLEARANCE_OUTER_M,
            "close_front_planar_max_m": _V17_WS_PLANAR_MAX_M,
            "force_pass_planar_max_m": _V17_WS_PLANAR_MAX_M,
        },
        "gt": {
            "collision_penalty_base": 78.0,
            "collision_episode_reward_cap": 76.0,
            "offtrack_penalty_base": 20.0,
            "offtrack_episode_reward_cap": 85.0,
            "offtrack_leniency_ratio": 0.10,
            "offtrack_leniency_mult": 1.7,
            "w_near_offtrack": 0.36,
            "near_offtrack_start_ratio": 0.60,
            "progress_reward_scale": 47.0,
            "w_speed_ref": 0.03,
            "speed_ref_vmin": 0.34,
            "speed_ref_vmax": 1.10,
            "safe_follow_bonus_scale": 0.38,
            "safe_follow_min_m": _V17_GT_FOLLOW_MIN_M,
            "safe_follow_max_m": _V17_GT_FOLLOW_MAX_M,
            "safe_follow_risk_max": 0.38,
            "safe_follow_ttc_min_s": 1.4,
            "safe_follow_ttc_max_s": 4.2,
            "wait_window_bonus_scale": 0.48,
            "wait_window_min_gap_m": _V17_GT_FOLLOW_MIN_M,
            "wait_window_max_gap_m": _V17_GT_WAIT_MAX_M,
            "wait_window_max_closing_rate": 0.20,
            "force_pass_penalty_scale": 0.95,
            "unsafe_close_penalty_scale": 1.15,
            "obstacle_clearance_penalty_scale": 0.70,
            "obstacle_clearance_inner_m": _V17_OBSTACLE_CLEARANCE_INNER_M,
            "obstacle_clearance_outer_m": _V17_OBSTACLE_CLEARANCE_OUTER_M,
            "post_pass_cut_in_penalty_scale": 0.85,
            "post_pass_watch_longitudinal_m": _V17_POST_PASS_WATCH_LONGITUDINAL_M,
            "post_pass_watch_steps": _V17_POST_PASS_WATCH_STEPS,
            "overtake_success_min_progress_ratio": _V17_OVERTAKE_SUCCESS_MIN_PROGRESS_RATIO,
            "unsafe_close_gap_m": _V17_GT_UNSAFE_GAP_M,
            "unsafe_close_clearance_m": _V17_OVERTAKE_LATERAL_CLEARANCE_M,
            "unsafe_close_longitudinal_m": _V17_OVERTAKE_CAPSULE_LONGITUDINAL_M,
            "lateral_overlap_ref_m": _V17_OVERTAKE_LATERAL_CLEARANCE_M,
            "unsafe_close_ttc_s": 2.2,
            "overtake_arm_longitudinal_min_m": _V17_GT_ARM_LONGITUDINAL_M,
            "overtake_arm_planar_max_m": _V17_GT_PLANAR_MAX_M,
            "overtake_pass_longitudinal_threshold_m": -_V17_GT_ARM_LONGITUDINAL_M,
            "overtake_pass_planar_min_m": 0.70,
            "close_front_planar_max_m": _V17_GT_PLANAR_MAX_M,
            "force_pass_planar_max_m": _V17_GT_PLANAR_MAX_M,
        },
    },
}

CURRICULUM_PHASES["avoid_static"].update(
    {
        **_V17_AVOID_REWARD_SHAPING,
        "obstacle_free_prob": 0.10,
        "ws_obstacle_free_prob": 0.10,
        "ws_obstacle_lateral_choices": _V17_WS_EDGE_OBSTACLE_LATERALS,
        "ws_obstacle_progress_min": 0.40,
        "ws_obstacle_progress_max": 0.60,
        "obstacle_target_ratios": {"ws": 0.90, "gt": 0.90},
        "reward_overrides_by_logging_key": {
            "ws": {
                "obstacle_clearance_penalty_scale": 1.00,
                "obstacle_clearance_inner_m": _V17_OBSTACLE_CLEARANCE_INNER_M,
                "obstacle_clearance_outer_m": _V17_OBSTACLE_CLEARANCE_OUTER_M,
            },
            "gt": {
                "obstacle_clearance_penalty_scale": 0.80,
                "obstacle_clearance_inner_m": _V17_OBSTACLE_CLEARANCE_INNER_M,
                "obstacle_clearance_outer_m": _V17_OBSTACLE_CLEARANCE_OUTER_M,
            },
        },
    }
)
CURRICULUM_PHASES["avoid_mixed"].update(
    {
        **_V17_AVOID_REWARD_SHAPING,
        "obstacle_free_prob": 0.10,
        "ws_obstacle_free_prob": 0.10,
        "ws_obstacle_lateral_choices": _V17_WS_EDGE_OBSTACLE_LATERALS,
        "ws_obstacle_progress_min": 0.40,
        "ws_obstacle_progress_max": 0.60,
        "obstacle_target_ratios": {"ws": 0.90, "gt": 0.90},
        "reward_overrides_by_logging_key": {
            "ws": {
                "obstacle_clearance_penalty_scale": 1.10,
                "obstacle_clearance_inner_m": _V17_OBSTACLE_CLEARANCE_INNER_M,
                "obstacle_clearance_outer_m": _V17_OBSTACLE_CLEARANCE_OUTER_M,
            },
            "gt": {
                "obstacle_clearance_penalty_scale": 0.90,
                "obstacle_clearance_inner_m": _V17_OBSTACLE_CLEARANCE_INNER_M,
                "obstacle_clearance_outer_m": _V17_OBSTACLE_CLEARANCE_OUTER_M,
            },
        },
    }
)
CURRICULUM_PHASES["lane_pid_intro"].update(
    {
        **_V17_PID_REWARD_SHAPING,
        **_V17_PID_OVERTAKE_INTRO,
        "obstacle_count": 1,
        "obstacle_free_prob": 0.05,
        "ws_obstacle_modes": ["lane_pid"],
        "ws_obstacle_free_prob": 0.05,
        "ws_obstacle_lateral_choices": _V17_WS_EDGE_OBSTACLE_LATERALS,
        "obstacle_lane_pid_speed_gt": 0.45,
        "obstacle_lane_pid_speed_ws": 0.45,
        "obstacle_spawn_ahead_min_m": 3.5,
        "obstacle_spawn_ahead_max_m": 9.0,
        "obstacle_progress_min": 0.10,
        "obstacle_progress_max": 0.20,
        "ws_obstacle_progress_min": 0.40,
        "ws_obstacle_progress_max": 0.60,
        "obstacle_target_ratios": {"ws": 0.95, "gt": 0.95},
        "max_compensation_ratio": 0.10,
    }
)
CURRICULUM_PHASES["lane_pid_mid"] = dict(CURRICULUM_PHASES["lane_pid_intro"])
CURRICULUM_PHASES["lane_pid_mid"].update(
    {
        **_V17_PID_OVERTAKE_MID,
        "obstacle_free_prob": 0.05,
        "ws_obstacle_modes": ["lane_pid"],
        "ws_obstacle_free_prob": 0.05,
        "ws_obstacle_lateral_choices": _V17_WS_EDGE_OBSTACLE_LATERALS,
        "obstacle_lane_pid_speed_gt": 0.50,
        "obstacle_lane_pid_speed_ws": 0.50,
        "obstacle_spawn_ahead_min_m": 3.5,
        "obstacle_spawn_ahead_max_m": 9.5,
        "obstacle_progress_min": 0.10,
        "obstacle_progress_max": 0.20,
        "ws_obstacle_progress_min": 0.35,
        "ws_obstacle_progress_max": 0.70,
        "obstacle_target_ratios": {"ws": 0.95, "gt": 0.95},
        "max_compensation_ratio": 0.10,
    }
)
CURRICULUM_PHASES["lane_pid_full"].update(
    {
        **_V17_PID_REWARD_SHAPING,
        **_V17_PID_OVERTAKE_FULL,
        "obstacle_count": 2,
        "ws_obstacle_count": 1,
        "ws_obstacle_modes": ["lane_pid"],
        "obstacle_free_prob": 0.05,
        "ws_obstacle_free_prob": 0.05,
        "ws_obstacle_lateral_choices": _V17_WS_EDGE_OBSTACLE_LATERALS,
        "obstacle_lane_pid_speed_gt": 0.55,
        "obstacle_lane_pid_speed_ws": 0.55,
        "obstacle_spawn_ahead_min_m": 3.5,
        "obstacle_spawn_ahead_max_m": 8.5,
        "obstacle_progress_min": 0.10,
        "obstacle_progress_max": 0.20,
        "ws_obstacle_progress_min": 0.25,
        "ws_obstacle_progress_max": 0.85,
        "ws_obstacle_fixed_progress_gap_min": 0.38,
        "ws_obstacle_fixed_progress_gap_max": 0.68,
        "obstacle_target_ratios": {"ws": 0.95, "gt": 0.95},
        "max_compensation_ratio": 0.10,
    }
)
_V17_LANE_PID_MID_STAGE = {
    "stage_name": "lane_pid_mid",
    "phase": "lane_pid_mid",
    "required_logging_keys": ["ws", "gt"],
    "recent_episodes": 10,
    "min_success_episodes": 2,
    "min_soft_laps": 2.0,
    "max_collision_rate_by_key": {"ws": 0.55, "gt": 0.55},
    "min_stage_timesteps": 300_000,
    "max_stage_timesteps": 1_500_000,
}
_V17_LANE_PID_FULL_STAGE = {
    "stage_name": "lane_pid_full",
    "phase": "lane_pid_full",
    "required_logging_keys": ["ws", "gt"],
    "recent_episodes": 16,
    "min_success_episodes": 3,
    "min_soft_laps": 2.0,
    "min_episode_len": 220,
    "max_collision_rate_by_key": {"ws": 0.60, "gt": 0.65},
    "min_stage_timesteps": 40_000,
    "max_stage_timesteps": 160_000,
}
_auto_stages_with_pid_mid = []
for _stage in AUTO_CURRICULUM_STAGES:
    if _stage.get("stage_name") == "lane_pid_full":
        _auto_stages_with_pid_mid.append(_V17_LANE_PID_FULL_STAGE)
        continue
    _auto_stages_with_pid_mid.append(_stage)
    if _stage.get("stage_name") == "lane_pid_intro":
        _auto_stages_with_pid_mid.append(_V17_LANE_PID_MID_STAGE)
AUTO_CURRICULUM_STAGES = tuple(_auto_stages_with_pid_mid)
_V17_STAGE_GATE_OVERRIDES = {
    # Gates still require recent WS/GT success windows; these hard bounds only
    # keep stages from missing a good window or burning time after clear failure.
    "warmup": {
        "min_stage_timesteps": 400_000,
        "max_stage_timesteps": 1_100_000,
        "max_collision_rate_by_key": {"ws": 0.75, "gt": 0.75},
    },
    "warmup_a": {
        "min_stage_timesteps": 120_000,
        "max_stage_timesteps": 600_000,
        "max_collision_rate_by_key": {"ws": 0.65, "gt": 0.65},
    },
    "avoid_static": {
        "min_stage_timesteps": 150_000,
        "max_stage_timesteps": 800_000,
        "max_collision_rate_by_key": {"ws": 0.55, "gt": 0.60},
        "max_obstacle_clearance_critical_rate_by_key": {"ws": 0.0, "gt": 0.0},
        "max_obstacle_clearance_band_rate_by_key": {"ws": 0.05, "gt": 0.05},
    },
    "avoid_mixed": {
        "min_stage_timesteps": 180_000,
        "max_stage_timesteps": 800_000,
        "max_collision_rate_by_key": {"ws": 0.55, "gt": 0.60},
        "max_obstacle_clearance_critical_rate_by_key": {"ws": 0.0, "gt": 0.0},
        "max_obstacle_clearance_band_rate_by_key": {"ws": 0.05, "gt": 0.05},
    },
    "lane_pid_intro": {
        "min_stage_timesteps": 180_000,
        "max_stage_timesteps": 520_000,
        "min_success_episodes": 3,
        "required_obstacle_mode_for_gate": "lane_pid",
        "max_collision_rate_by_key": {"ws": 0.25, "gt": 0.35},
        "max_obstacle_clearance_critical_rate_by_key": {"ws": 0.0, "gt": 0.0},
        "max_obstacle_clearance_band_rate_by_key": {"ws": 0.05, "gt": 0.05},
    },
    "lane_pid_mid": {
        "min_stage_timesteps": 220_000,
        "max_stage_timesteps": 560_000,
        "min_success_episodes": 3,
        "required_obstacle_mode_for_gate": "lane_pid",
        "max_collision_rate_by_key": {"ws": 0.18, "gt": 0.25},
        "max_obstacle_clearance_critical_rate_by_key": {"ws": 0.0, "gt": 0.0},
        "max_obstacle_clearance_band_rate_by_key": {"ws": 0.05, "gt": 0.05},
    },
    "lane_pid_full": {
        "min_stage_timesteps": 240_000,
        "max_stage_timesteps": 620_000,
        "min_success_episodes": 3,
        "required_obstacle_mode_for_gate": "lane_pid",
        "max_collision_rate_by_key": {"ws": 0.20, "gt": 0.30},
        "max_obstacle_clearance_critical_rate_by_key": {"ws": 0.0, "gt": 0.0},
        "max_obstacle_clearance_band_rate_by_key": {"ws": 0.05, "gt": 0.05},
    },
}
AUTO_CURRICULUM_STAGES = tuple(
    {
        **stage,
        **_V17_STAGE_GATE_OVERRIDES.get(str(stage.get("stage_name", "")), {}),
    }
    for stage in AUTO_CURRICULUM_STAGES
)
CURRICULUM_PHASE_ALIASES.update(
    {
        "pid_intro": "lane_pid_intro",
        "pid_mid": "lane_pid_mid",
        "stage4b": "lane_pid_mid",
        "pid_full": "lane_pid_full",
    }
)

TRAIN_V17_CURRICULUM_DEFAULTS: Dict[str, Any] = dict(TRAIN_V16_DEFAULTS)
TRAIN_V17_CURRICULUM_DEFAULTS.update(
    {
        "reward_overrides_by_logging_key": None,
        "learning_rate": 8e-5,
        "ppo_n_steps": 4096,
        "reward_safe_follow_bonus": 0.02,
        "adapter_k_delta": 0.15,
        "adapter_k_bias": 0.15,
        "adapter_v_nominal": 1.15,
        "adapter_k_turn": 0.55,
        "adapter_alpha_speed": 0.55,
        "adapter_v_min": 0.35,
        "adapter_v_max": 2.00,
        "adapter_max_throttle": 0.80,
        "obstacle_lateral_choices": None,
        "ws_obstacle_lateral_choices": None,
        "delta_max": 0.35,
        "beta": 0.6,
        "steer_delta_delta_max": None,
        "steer_servo_deadband": 0.0,
        "w_d": 0.04,
        "w_dd": 0.01,
        "w_m": 0.0,
        "w_sat": 0.0,
        "w_steer_budget": 0.0,
        "steer_budget_straight": 0.58,
        "steer_budget_curve": 0.88,
        "steer_budget_obstacle_relief": 0.16,
        "w_sign_flip": 0.0,
        "sign_flip_min_abs_steer": 0.20,
        "w_micro_wiggle": 0.0,
        "micro_wiggle_min_abs_steer": 0.035,
        "micro_wiggle_max_abs_steer": 0.22,
        "collision_penalty_base": 8.0,
        "w_near_collision": 0.24,
        "near_collision_start_ratio": 0.65,
        "overtake_success_bonus": 3.0,
        "reward_prepare_pass_bonus": 0.0,
        "reward_commit_pass_bonus": 0.0,
        "reward_post_pass_bonus": 0.0,
        "reward_post_pass_steps": 10,
        "terminal_offtrack_progress_scale": 1.0,
        "sim2real_steer_gain_override": None,
        "sim2real_throttle_gain_override": None,
        "obstacle_fixed_progress_distribution": None,
    }
)


def _clone_curriculum_value(value: Any) -> Any:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {k: _clone_curriculum_value(v) for k, v in value.items()}
    return value


def _resolve_learn_total_timesteps(
    requested_timesteps: int,
    model_start_timesteps: int,
    resume_ckpt_path: Optional[str],
) -> int:
    requested = max(0, int(requested_timesteps))
    if resume_ckpt_path is None:
        return requested
    return max(0, int(model_start_timesteps)) + requested


def _resolve_curriculum_phase(curriculum_phase: Optional[str]) -> Optional[str]:
    name = str(curriculum_phase or "").strip().lower()
    if not name or name in ("none", "off", "default"):
        return None
    name = CURRICULUM_PHASE_ALIASES.get(name, name)
    if name not in CURRICULUM_PHASES:
        available = ", ".join(sorted(CURRICULUM_PHASES.keys()))
        raise KeyError(f"Unknown curriculum_phase={curriculum_phase}. Available: {available}")
    return name


def _rebuild_recurrent_rollout_state(
    model: Any,
    *,
    n_steps: int,
    batch_size: Optional[int],
    reason: str,
) -> None:
    """Keep RecurrentPPO buffers consistent after resume-time hyperparameter edits."""
    if RecurrentDictRolloutBuffer is None or RecurrentRolloutBuffer is None or RNNStates is None:
        raise RuntimeError("sb3_contrib recurrent buffer classes are unavailable")

    n_steps = int(n_steps)
    if n_steps <= 0:
        raise ValueError(f"ppo_n_steps must be positive, got {n_steps}")

    n_envs = int(getattr(model, "n_envs", 1))
    rollout_size = n_steps * n_envs
    if batch_size is not None:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"ppo_batch_size must be positive, got {batch_size}")
        if batch_size > rollout_size:
            raise ValueError(
                f"ppo_batch_size ({batch_size}) must be <= n_steps * n_envs ({rollout_size})"
            )
        model.batch_size = batch_size

    lstm = getattr(getattr(model, "policy", None), "lstm_actor", None)
    if lstm is None:
        raise RuntimeError("Cannot rebuild recurrent rollout state: model.policy.lstm_actor is missing")

    old_buffer_size = getattr(getattr(model, "rollout_buffer", None), "buffer_size", None)
    model.n_steps = n_steps

    single_hidden_state_shape = (int(lstm.num_layers), n_envs, int(lstm.hidden_size))

    def _zero_state() -> torch.Tensor:
        return torch.zeros(single_hidden_state_shape, device=model.device)

    model._last_lstm_states = RNNStates(
        (_zero_state(), _zero_state()),
        (_zero_state(), _zero_state()),
    )

    hidden_state_buffer_shape = (
        n_steps,
        int(lstm.num_layers),
        n_envs,
        int(lstm.hidden_size),
    )
    buffer_cls = (
        RecurrentDictRolloutBuffer
        if isinstance(model.observation_space, gym.spaces.Dict)
        else RecurrentRolloutBuffer
    )
    model.rollout_buffer = buffer_cls(
        n_steps,
        model.observation_space,
        model.action_space,
        hidden_state_buffer_shape,
        model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=n_envs,
    )
    print(
        "🔁 Rebuilt recurrent rollout state "
        f"({reason}): buffer_size {old_buffer_size} -> {n_steps}, "
        f"batch_size={model.batch_size}, hidden_state_shape={hidden_state_buffer_shape}"
    )


def _policy_pth_for_checkpoint(checkpoint_path: str) -> str:
    path = str(checkpoint_path)
    if path.endswith(".zip"):
        return path[:-4] + "_policy.pth"
    return path + "_policy.pth"


def _apply_curriculum_phase(
    curriculum_phase: Optional[str],
    values: Dict[str, Any],
    explicit_keys: Optional[Set[str]] = None,
) -> Any:
    resolved = _resolve_curriculum_phase(curriculum_phase)
    if resolved is None:
        return None, {}

    explicit_keys = set(explicit_keys or set())
    applied: Dict[str, Any] = {}
    skipped_explicit: List[str] = []
    phase_overrides = CURRICULUM_PHASES[resolved]
    for key, target_value in phase_overrides.items():
        if key in explicit_keys:
            skipped_explicit.append(key)
            continue
        current_value = values.get(key)
        default_value = TRAIN_V17_CURRICULUM_DEFAULTS.get(key, None)
        if current_value == default_value:
            cloned = _clone_curriculum_value(target_value)
            values[key] = cloned
            applied[key] = _clone_curriculum_value(cloned)
    if skipped_explicit:
        print(
            "🧩 curriculum explicit CLI keeps: "
            f"phase={resolved}, keys={sorted(skipped_explicit)}"
        )
    return resolved, applied


CURRICULUM_CLI_FLAG_TO_KEY: Dict[str, str] = {
    "--scene-weights": "scene_weights",
    "--learning-rate": "learning_rate",
    "--ppo-n-steps": "ppo_n_steps",
    "--disable-step-balance-sampling": "enable_step_balance_sampling",
    "--adapter-k-delta": "adapter_k_delta",
    "--adapter-k-bias": "adapter_k_bias",
    "--adapter-v-nominal": "adapter_v_nominal",
    "--adapter-k-turn": "adapter_k_turn",
    "--adapter-alpha-speed": "adapter_alpha_speed",
    "--adapter-v-min": "adapter_v_min",
    "--adapter-v-max": "adapter_v_max",
    "--adapter-max-throttle": "adapter_max_throttle",
    "--delta-max": "delta_max",
    "--beta": "beta",
    "--steer-delta-delta-max": "steer_delta_delta_max",
    "--steer-servo-deadband": "steer_servo_deadband",
    "--w-d": "w_d",
    "--w-dd": "w_dd",
    "--w-m": "w_m",
    "--w-sat": "w_sat",
    "--w-steer-budget": "w_steer_budget",
    "--steer-budget-straight": "steer_budget_straight",
    "--steer-budget-curve": "steer_budget_curve",
    "--steer-budget-obstacle-relief": "steer_budget_obstacle_relief",
    "--w-sign-flip": "w_sign_flip",
    "--sign-flip-min-abs-steer": "sign_flip_min_abs_steer",
    "--w-micro-wiggle": "w_micro_wiggle",
    "--micro-wiggle-min-abs-steer": "micro_wiggle_min_abs_steer",
    "--micro-wiggle-max-abs-steer": "micro_wiggle_max_abs_steer",
    "--sim2real-steer-gain-override": "sim2real_steer_gain_override",
    "--sim2real-throttle-gain-override": "sim2real_throttle_gain_override",
    "--disable-obstacles": "obstacle_enabled",
    "--obstacle-count": "obstacle_count",
    "--ws-obstacle-count": "ws_obstacle_count",
    "--obstacle-free-prob": "obstacle_free_prob",
    "--obstacle-modes": "obstacle_modes",
    "--ws-obstacle-free-prob": "ws_obstacle_free_prob",
    "--obstacle-lateral-choices": "obstacle_lateral_choices",
    "--ws-obstacle-lateral-choices": "ws_obstacle_lateral_choices",
    "--obstacle-spawn-ahead-min-m": "obstacle_spawn_ahead_min_m",
    "--obstacle-spawn-ahead-max-m": "obstacle_spawn_ahead_max_m",
    "--obstacle-min-agent-planar-dist-m": "obstacle_min_agent_planar_dist_m",
    "--obstacle-min-agent-arc-dist-m": "obstacle_min_agent_arc_dist_m",
    "--obstacle-fixed-progress-ratio": "obstacle_fixed_progress_ratio",
    "--obstacle-fixed-progress-gap": "obstacle_fixed_progress_gap",
    "--obstacle-fixed-progress-gap-min": "obstacle_fixed_progress_gap_min",
    "--obstacle-fixed-progress-gap-max": "obstacle_fixed_progress_gap_max",
    "--obstacle-progress-min": "obstacle_progress_min",
    "--obstacle-progress-max": "obstacle_progress_max",
    "--obstacle-fixed-lateral-ratio": "obstacle_fixed_lateral_ratio",
    "--gt-obstacle-start-exclusion-half-width-m": "gt_obstacle_start_exclusion_half_width_m",
    "--ws-obstacle-modes": "ws_obstacle_modes",
    "--ws-obstacle-fixed-progress-ratio": "ws_obstacle_fixed_progress_ratio",
    "--ws-obstacle-fixed-progress-gap": "ws_obstacle_fixed_progress_gap",
    "--ws-obstacle-fixed-progress-gap-min": "ws_obstacle_fixed_progress_gap_min",
    "--ws-obstacle-fixed-progress-gap-max": "ws_obstacle_fixed_progress_gap_max",
    "--ws-obstacle-progress-min": "ws_obstacle_progress_min",
    "--ws-obstacle-progress-max": "ws_obstacle_progress_max",
    "--ws-obstacle-fixed-lateral-ratio": "ws_obstacle_fixed_lateral_ratio",
    "--disable-obstacle-random-yaw": "obstacle_randomize_non_lane_pid_yaw",
    "--obstacle-lane-pid-speed-gt": "obstacle_lane_pid_speed_gt",
    "--obstacle-lane-pid-speed-ws": "obstacle_lane_pid_speed_ws",
    "--collision-penalty-base": "collision_penalty_base",
    "--offtrack-penalty-base": "offtrack_penalty_base",
    "--w-near-collision": "w_near_collision",
    "--overtake-success-bonus": "overtake_success_bonus",
    "--reward-prepare-pass-bonus": "reward_prepare_pass_bonus",
    "--reward-commit-pass-bonus": "reward_commit_pass_bonus",
    "--reward-post-pass-bonus": "reward_post_pass_bonus",
    "--reward-post-pass-steps": "reward_post_pass_steps",
    "--terminal-offtrack-progress-scale": "terminal_offtrack_progress_scale",
}


def _explicit_curriculum_keys_from_argv(argv: List[str]) -> Set[str]:
    keys: Set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        flag = token.split("=", 1)[0]
        key = CURRICULUM_CLI_FLAG_TO_KEY.get(flag)
        if key:
            keys.add(key)
    return keys

V17_DRIVER_PROFILE_DEFAULTS: Dict[str, float] = {
    # Real-aligned speed profile: lower median speed than V16-era defaults,
    # while keeping enough high-end throttle authority for the policy to choose speed.
    "adapter_v_nominal": 1.15,
    "adapter_k_turn": 0.55,
    "adapter_alpha_speed": 0.55,
    "adapter_v_min": 0.35,
    "adapter_v_max": 2.00,
    "adapter_max_throttle": 0.80,
}


def run_v17_contract_tests(
    obs_size: int = 128,
    lidar_obs_mode: str = "full",
    lidar_num_sectors: int = 36,
    lidar_fov_deg: float = 180.0,
    lidar_max_range_m: float = 20.0,
    lidar_near_clip_m: float = 0.18,
) -> None:
    class DummyBaseEnv(gym.Env):
        def __init__(self, obs_size_: int):
            self.obs_size = int(obs_size_)
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8
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
            lidar = np.linspace(0.4, 4.0, 180, dtype=np.float32)
            info = {
                "speed": 0.8,
                "gyro": (0.0, 0.2, 0.0),
                "accel": (0.1, 0.0, 9.7),
                "car": (0.0, 0.0, 30.0),
                "pos": (0.0, 0.0, 0.0),
                "cte": 0.1,
                "obstacle_present": 1.0,
                "obstacle_longitudinal": 2.2,
                "obstacle_lateral": -0.3,
                "obstacle_dist": 2.25,
                "obstacle_risk": 0.4,
                "lidar": lidar,
            }
            done = self._step_count >= 8
            return obs, 0.0, done, info

    print(
        f"🧪 V17 Contract Tests [{lidar_obs_mode}, "
        f"sectors={lidar_num_sectors}, fov={lidar_fov_deg}]"
    )
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
        w_near_collision=0.24,
        near_collision_start_ratio=0.65,
        safe_follow_bonus_scale=0.02,
        prepare_pass_bonus_scale=0.02,
        commit_pass_bonus_scale=0.02,
        reward_control_dt_s=0.05,
    )
    action_safety = ActionSafetyWrapper(reward_wrapper, delta_max=0.35)
    reward_wrapper.action_safety_wrapper = action_safety
    adapter = ActionAdapterWrapper(
        action_safety,
        v_nominal=V17_DRIVER_PROFILE_DEFAULTS["adapter_v_nominal"],
        k_turn=V17_DRIVER_PROFILE_DEFAULTS["adapter_k_turn"],
        alpha_speed=V17_DRIVER_PROFILE_DEFAULTS["adapter_alpha_speed"],
        v_min=V17_DRIVER_PROFILE_DEFAULTS["adapter_v_min"],
        v_max=V17_DRIVER_PROFILE_DEFAULTS["adapter_v_max"],
        max_throttle=V17_DRIVER_PROFILE_DEFAULTS["adapter_max_throttle"],
    )
    env = V17ObsWrapper(
        adapter,
        scene_key="waveshare",
        logging_key="ws",
        domain="ws",
        obs_size=obs_size,
        speed_vmax=2.2,
        control_wrapper=adapter,
        action_safety_wrapper=action_safety,
        lidar_obs_mode=lidar_obs_mode,
        lidar_spec=CanonicalLidarSpec(
            num_sectors=int(lidar_num_sectors),
            fov_deg=float(lidar_fov_deg),
            max_range_m=float(lidar_max_range_m),
            near_clip_m=float(lidar_near_clip_m),
            invalid_fill_m=float(lidar_max_range_m),
        ),
    )
    obs = env.reset()
    expected_image_channels = len(getattr(env, "image_channel_indices", (0, 1, 2, 3, 4, 5)))
    assert obs["image"].shape == (expected_image_channels, obs_size, obs_size)
    assert obs["state"].shape == (7,)
    expected_lidar_dim = (
        2 * int(lidar_num_sectors)
        if str(lidar_obs_mode).strip().lower() == "full"
        else 12
    )
    assert obs["lidar"].shape == (expected_lidar_dim,)
    assert obs["lidar_meta"].shape == (2,)
    assert obs["domain_id"].shape == (1,)
    obs2, _r, _d, info = env.step(np.array([0.2, -0.1, 0.5], dtype=np.float32))
    assert obs2["image"].shape == (expected_image_channels, obs_size, obs_size)
    assert obs2["lidar"].shape == (expected_lidar_dim,)
    assert float(info["lidar_steps_since_new_scan_norm"]) >= 0.0
    if expected_lidar_dim == 12:
        assert "target_exist" in info
        assert "target_rel_long" in info
    extractor = LiDARFiLMFeatureExtractor(env.observation_space)
    torch_obs = {
        key: torch.as_tensor(np.asarray(value)[None, ...], dtype=torch.float32)
        for key, value in obs2.items()
    }
    feat = extractor(torch_obs)
    assert feat.shape[0] == 1
    print("  ✅ V17 contract tests passed")


def run_preflight_tests(
    track_geometry: TrackGeometryManager,
    obs_size: int = 128,
    lidar_obs_mode: str = "full",
    lidar_num_sectors: int = 36,
    lidar_fov_deg: float = 180.0,
    lidar_max_range_m: float = 20.0,
    lidar_near_clip_m: float = 0.18,
) -> None:
    print("\n🔍 Running V17 preflight checks...")
    run_offline_track_checks(track_geometry)
    run_v17_contract_tests(
        obs_size=obs_size,
        lidar_obs_mode=lidar_obs_mode,
        lidar_num_sectors=lidar_num_sectors,
        lidar_fov_deg=lidar_fov_deg,
        lidar_max_range_m=lidar_max_range_m,
        lidar_near_clip_m=lidar_near_clip_m,
    )
    print("✅ All V17 preflight checks passed\n")


def train_v17(
    env_ids: Optional[List[str]] = None,
    scene_weights: Optional[List[float]] = None,
    track_dir: str = DEFAULT_TRACK_DIR,
    sim_path: str = "remote",
    total_timesteps: int = 2_000_000,
    save_dir: str = "models/v17_multi_scene_lidar",
    port: int = 9091,
    obs_size: int = 128,
    augment: bool = False,
    yellow_dropout_prob: float = 0.20,
    dropout_start_step: int = 0,
    dropout_ramp_steps: int = 200_000,
    scene_start_snapshot_steps: int = 10,
    lstm_hidden_size: int = 256,
    lstm_layers: int = 2,
    adapter_k_delta: float = 0.15,
    adapter_lambda_bias: float = 0.20,
    adapter_k_bias: float = 0.15,
    adapter_steer_core_decay: float = 0.0,
    adapter_v_nominal: float = V17_DRIVER_PROFILE_DEFAULTS["adapter_v_nominal"],
    adapter_k_turn: float = V17_DRIVER_PROFILE_DEFAULTS["adapter_k_turn"],
    adapter_k_bias_speed: float = 0.0,
    adapter_alpha_speed: float = V17_DRIVER_PROFILE_DEFAULTS["adapter_alpha_speed"],
    adapter_v_min: float = V17_DRIVER_PROFILE_DEFAULTS["adapter_v_min"],
    adapter_v_max: float = V17_DRIVER_PROFILE_DEFAULTS["adapter_v_max"],
    adapter_max_throttle: float = V17_DRIVER_PROFILE_DEFAULTS["adapter_max_throttle"],
    speed_vmax: float = 2.2,
    speed_kp: float = 0.35,
    speed_ki: float = 0.08,
    speed_kff: float = 0.10,
    allow_reverse: bool = False,
    control_dt: float = 0.05,
    sim_timescale: float = 1.0,
    learning_rate: float = 8e-5,
    ent_coef: float = 0.01,
    ppo_n_steps: int = 4096,
    ppo_batch_size: int = 256,
    ppo_n_epochs: int = 4,
    ppo_clip_range: float = 0.2,
    target_kl: Optional[float] = 0.02,
    min_episodes_per_scene: int = 5,
    max_steps_per_scene: int = 2048,
    scene_reload_timeout_s: float = 2.0,
    scene_reload_post_exit_sleep_s: float = 0.3,
    scene_start_force_reload: bool = False,
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
    steer_delta_delta_max: Optional[float] = None,
    steer_servo_deadband: float = 0.0,
    w_d: float = 0.04,
    w_dd: float = 0.01,
    w_m: float = 0.0,
    w_sat: float = 0.0,
    w_steer_budget: float = 0.0,
    steer_budget_straight: float = 0.58,
    steer_budget_curve: float = 0.88,
    steer_budget_obstacle_relief: float = 0.16,
    w_sign_flip: float = 0.0,
    sign_flip_min_abs_steer: float = 0.20,
    w_micro_wiggle: float = 0.0,
    micro_wiggle_min_abs_steer: float = 0.035,
    micro_wiggle_max_abs_steer: float = 0.22,
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
    overtake_success_bonus: float = 3.0,
    reward_safe_follow_bonus: float = 0.02,
    reward_prepare_pass_bonus: float = 0.04,
    reward_commit_pass_bonus: float = 0.04,
    reward_post_pass_bonus: float = 0.5,
    reward_post_pass_steps: int = 10,
    reward_overrides_by_logging_key: Optional[Dict[str, Dict[str, Any]]] = None,
    terminal_offtrack_progress_scale: float = 1.0,
    bad_episode_guard_min_steps: int = 320,
    bad_episode_guard_reward_floor: float = -160.0,
    bad_episode_guard_cte_over_in_rate: float = 0.25,
    bad_episode_guard_min_forward_progress: float = 0.35,
    bad_episode_guard_penalty: float = 4.0,
    offtrack_leniency_ratio: float = 0.25,
    offtrack_leniency_mult: float = 2.5,
    obstacle_enabled: bool = True,
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
    image_channel_indices: Optional[List[int]] = None,
    disable_lidar_meta: bool = False,
    lidar_encoder_mode: str = "side_separated",
    lidar_obs_mode: str = "full",
    lidar_num_sectors: int = 36,
    lidar_fov_deg: float = 180.0,
    lidar_max_range_m: float = 20.0,
    lidar_near_clip_m: float = 0.18,
    lidar_repeat_min_steps: int = 2,
    lidar_repeat_max_steps: int = 4,
    predictive_safety_filter_path: Optional[str] = None,
    predictive_safety_filter_mode: str = "log",
    predictive_safety_filter_log_path: Optional[str] = None,
    predictive_safety_yaw_thresh: Optional[float] = None,
    predictive_safety_decel_thresh: Optional[float] = None,
    port_per_scene: Optional[List[int]] = None,
    critic_calibration_freq: int = 50_000,
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
    auto_lr_min: float = 2e-5,
    auto_lr_high_kl: float = 0.05,
    auto_lr_high_kl_patience: int = 3,
    auto_lr_balanced_drop: float = 8.0,
    auto_lr_balanced_patience: int = 12,
    auto_lr_cooldown_checks: int = 15,
    auto_lr_warmup_steps: int = 250000,
    auto_lr_best_window: int = 50,
    sim2real_json: Optional[str] = None,
    sim2real_throttle_gain_floor: Optional[float] = 0.25,
    sim2real_throttle_gain_override: Optional[float] = None,
    sim2real_steer_gain_floor: Optional[float] = None,
    sim2real_steer_gain_override: Optional[float] = None,
    extra_callbacks: Optional[List[BaseCallback]] = None,
    extra_run_metadata: Optional[Dict[str, Any]] = None,
    config_filename: str = "v17_config.json",
    explicit_cli_keys: Optional[Set[str]] = None,
    best_model_name_prefix: str = "best_model",
    checkpoint_keep_last: int = 5,
) -> Dict[str, Any]:
    if RecurrentPPO is None:
        raise ImportError("sb3_contrib not available, please install sb3-contrib==1.8.0")

    env_ids = list(DEFAULT_ENV_IDS if env_ids is None else env_ids)
    for env_id in env_ids:
        if env_id not in SCENE_SPECS:
            raise KeyError(f"Unsupported env_id for V17: {env_id}")
    if port_per_scene is not None:
        port_per_scene = [int(p) for p in port_per_scene]
        if len(port_per_scene) != len(env_ids):
            raise ValueError(
                f"--ports length ({len(port_per_scene)}) must match --env-ids ({len(env_ids)})"
            )
        if len(set(port_per_scene)) != len(port_per_scene):
            raise ValueError(f"--ports must be one dedicated port per scene, got {port_per_scene}")

    curriculum_values = {
        "scene_weights": scene_weights,
        "enable_dynamic_scene_weights": enable_dynamic_scene_weights,
        "enable_step_balance_sampling": enable_step_balance_sampling,
        "learning_rate": learning_rate,
        "ppo_n_steps": ppo_n_steps,
        "adapter_k_delta": adapter_k_delta,
        "adapter_k_bias": adapter_k_bias,
        "adapter_v_nominal": adapter_v_nominal,
        "adapter_k_turn": adapter_k_turn,
        "adapter_alpha_speed": adapter_alpha_speed,
        "adapter_v_min": adapter_v_min,
        "adapter_v_max": adapter_v_max,
        "adapter_max_throttle": adapter_max_throttle,
        "delta_max": delta_max,
        "beta": beta,
        "steer_delta_delta_max": steer_delta_delta_max,
        "steer_servo_deadband": steer_servo_deadband,
        "w_d": w_d,
        "w_dd": w_dd,
        "w_m": w_m,
        "w_sat": w_sat,
        "w_steer_budget": w_steer_budget,
        "steer_budget_straight": steer_budget_straight,
        "steer_budget_curve": steer_budget_curve,
        "steer_budget_obstacle_relief": steer_budget_obstacle_relief,
        "w_sign_flip": w_sign_flip,
        "sign_flip_min_abs_steer": sign_flip_min_abs_steer,
        "w_micro_wiggle": w_micro_wiggle,
        "micro_wiggle_min_abs_steer": micro_wiggle_min_abs_steer,
        "micro_wiggle_max_abs_steer": micro_wiggle_max_abs_steer,
        "sim2real_steer_gain_override": sim2real_steer_gain_override,
        "sim2real_throttle_gain_override": sim2real_throttle_gain_override,
        "obstacle_enabled": obstacle_enabled,
        "obstacle_count": obstacle_count,
        "ws_obstacle_count": ws_obstacle_count,
        "obstacle_free_prob": obstacle_free_prob,
        "obstacle_modes": obstacle_modes,
        "ws_obstacle_free_prob": ws_obstacle_free_prob,
        "obstacle_spawn_ahead_min_m": obstacle_spawn_ahead_min_m,
        "obstacle_spawn_ahead_max_m": obstacle_spawn_ahead_max_m,
        "obstacle_min_agent_planar_dist_m": obstacle_min_agent_planar_dist_m,
        "obstacle_min_agent_arc_dist_m": obstacle_min_agent_arc_dist_m,
        "obstacle_lateral_choices": obstacle_lateral_choices,
        "ws_obstacle_lateral_choices": ws_obstacle_lateral_choices,
        "obstacle_fixed_progress_ratio": obstacle_fixed_progress_ratio,
        "obstacle_fixed_progress_distribution": obstacle_fixed_progress_distribution,
        "obstacle_fixed_progress_gap": obstacle_fixed_progress_gap,
        "obstacle_fixed_progress_gap_min": obstacle_fixed_progress_gap_min,
        "obstacle_fixed_progress_gap_max": obstacle_fixed_progress_gap_max,
        "obstacle_progress_min": obstacle_progress_min,
        "obstacle_progress_max": obstacle_progress_max,
        "obstacle_fixed_lateral_ratio": obstacle_fixed_lateral_ratio,
        "gt_obstacle_start_exclusion_half_width_m": gt_obstacle_start_exclusion_half_width_m,
        "ws_obstacle_modes": ws_obstacle_modes,
        "ws_obstacle_fixed_progress_ratio": ws_obstacle_fixed_progress_ratio,
        "ws_obstacle_fixed_progress_gap": ws_obstacle_fixed_progress_gap,
        "ws_obstacle_fixed_progress_gap_min": ws_obstacle_fixed_progress_gap_min,
        "ws_obstacle_fixed_progress_gap_max": ws_obstacle_fixed_progress_gap_max,
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
        "reward_prepare_pass_bonus": reward_prepare_pass_bonus,
        "reward_commit_pass_bonus": reward_commit_pass_bonus,
        "reward_post_pass_bonus": reward_post_pass_bonus,
        "reward_post_pass_steps": reward_post_pass_steps,
        "reward_overrides_by_logging_key": reward_overrides_by_logging_key,
        "terminal_offtrack_progress_scale": terminal_offtrack_progress_scale,
    }
    curriculum_phase, curriculum_applied = _apply_curriculum_phase(
        curriculum_phase,
        curriculum_values,
        explicit_keys=explicit_cli_keys,
    )
    scene_weights = curriculum_values["scene_weights"]
    enable_dynamic_scene_weights = curriculum_values["enable_dynamic_scene_weights"]
    enable_step_balance_sampling = curriculum_values["enable_step_balance_sampling"]
    learning_rate = curriculum_values["learning_rate"]
    ppo_n_steps = curriculum_values["ppo_n_steps"]
    adapter_k_delta = curriculum_values["adapter_k_delta"]
    adapter_k_bias = curriculum_values["adapter_k_bias"]
    adapter_v_nominal = curriculum_values["adapter_v_nominal"]
    adapter_k_turn = curriculum_values["adapter_k_turn"]
    adapter_alpha_speed = curriculum_values["adapter_alpha_speed"]
    adapter_v_min = curriculum_values["adapter_v_min"]
    adapter_v_max = curriculum_values["adapter_v_max"]
    adapter_max_throttle = curriculum_values["adapter_max_throttle"]
    delta_max = curriculum_values["delta_max"]
    beta = curriculum_values["beta"]
    steer_delta_delta_max = curriculum_values["steer_delta_delta_max"]
    steer_servo_deadband = curriculum_values["steer_servo_deadband"]
    w_d = curriculum_values["w_d"]
    w_dd = curriculum_values["w_dd"]
    w_m = curriculum_values["w_m"]
    w_sat = curriculum_values["w_sat"]
    w_steer_budget = curriculum_values["w_steer_budget"]
    steer_budget_straight = curriculum_values["steer_budget_straight"]
    steer_budget_curve = curriculum_values["steer_budget_curve"]
    steer_budget_obstacle_relief = curriculum_values["steer_budget_obstacle_relief"]
    w_sign_flip = curriculum_values["w_sign_flip"]
    sign_flip_min_abs_steer = curriculum_values["sign_flip_min_abs_steer"]
    w_micro_wiggle = curriculum_values["w_micro_wiggle"]
    micro_wiggle_min_abs_steer = curriculum_values["micro_wiggle_min_abs_steer"]
    micro_wiggle_max_abs_steer = curriculum_values["micro_wiggle_max_abs_steer"]
    sim2real_steer_gain_override = curriculum_values["sim2real_steer_gain_override"]
    sim2real_throttle_gain_override = curriculum_values["sim2real_throttle_gain_override"]
    obstacle_enabled = curriculum_values["obstacle_enabled"]
    obstacle_count = curriculum_values["obstacle_count"]
    ws_obstacle_count = curriculum_values["ws_obstacle_count"]
    obstacle_free_prob = curriculum_values["obstacle_free_prob"]
    obstacle_modes = curriculum_values["obstacle_modes"]
    ws_obstacle_free_prob = curriculum_values["ws_obstacle_free_prob"]
    obstacle_spawn_ahead_min_m = curriculum_values["obstacle_spawn_ahead_min_m"]
    obstacle_spawn_ahead_max_m = curriculum_values["obstacle_spawn_ahead_max_m"]
    obstacle_min_agent_planar_dist_m = curriculum_values["obstacle_min_agent_planar_dist_m"]
    obstacle_min_agent_arc_dist_m = curriculum_values["obstacle_min_agent_arc_dist_m"]
    obstacle_lateral_choices = curriculum_values["obstacle_lateral_choices"]
    ws_obstacle_lateral_choices = curriculum_values["ws_obstacle_lateral_choices"]
    obstacle_fixed_progress_ratio = curriculum_values["obstacle_fixed_progress_ratio"]
    obstacle_fixed_progress_distribution = curriculum_values["obstacle_fixed_progress_distribution"]
    obstacle_fixed_progress_gap = curriculum_values["obstacle_fixed_progress_gap"]
    obstacle_fixed_progress_gap_min = curriculum_values["obstacle_fixed_progress_gap_min"]
    obstacle_fixed_progress_gap_max = curriculum_values["obstacle_fixed_progress_gap_max"]
    obstacle_progress_min = curriculum_values["obstacle_progress_min"]
    obstacle_progress_max = curriculum_values["obstacle_progress_max"]
    obstacle_fixed_lateral_ratio = curriculum_values["obstacle_fixed_lateral_ratio"]
    gt_obstacle_start_exclusion_half_width_m = curriculum_values["gt_obstacle_start_exclusion_half_width_m"]
    ws_obstacle_modes = curriculum_values["ws_obstacle_modes"]
    ws_obstacle_fixed_progress_ratio = curriculum_values["ws_obstacle_fixed_progress_ratio"]
    ws_obstacle_fixed_progress_gap = curriculum_values["ws_obstacle_fixed_progress_gap"]
    ws_obstacle_fixed_progress_gap_min = curriculum_values["ws_obstacle_fixed_progress_gap_min"]
    ws_obstacle_fixed_progress_gap_max = curriculum_values["ws_obstacle_fixed_progress_gap_max"]
    ws_obstacle_progress_min = curriculum_values["ws_obstacle_progress_min"]
    ws_obstacle_progress_max = curriculum_values["ws_obstacle_progress_max"]
    ws_obstacle_fixed_lateral_ratio = curriculum_values["ws_obstacle_fixed_lateral_ratio"]
    obstacle_randomize_non_lane_pid_yaw = curriculum_values["obstacle_randomize_non_lane_pid_yaw"]
    obstacle_lane_pid_speed_gt = curriculum_values["obstacle_lane_pid_speed_gt"]
    obstacle_lane_pid_speed_ws = curriculum_values["obstacle_lane_pid_speed_ws"]
    collision_penalty_base = curriculum_values["collision_penalty_base"]
    offtrack_penalty_base = curriculum_values["offtrack_penalty_base"]
    w_near_collision = curriculum_values["w_near_collision"]
    near_collision_start_ratio = curriculum_values["near_collision_start_ratio"]
    overtake_success_bonus = curriculum_values["overtake_success_bonus"]
    reward_prepare_pass_bonus = curriculum_values["reward_prepare_pass_bonus"]
    reward_commit_pass_bonus = curriculum_values["reward_commit_pass_bonus"]
    reward_post_pass_bonus = curriculum_values["reward_post_pass_bonus"]
    reward_post_pass_steps = curriculum_values["reward_post_pass_steps"]
    reward_overrides_by_logging_key = curriculum_values["reward_overrides_by_logging_key"]
    terminal_offtrack_progress_scale = curriculum_values["terminal_offtrack_progress_scale"]

    sim_timescale = float(max(1e-3, sim_timescale))
    base_timescale_params = {
        "control_dt": float(control_dt),
        "adapter_k_delta": float(adapter_k_delta),
        "adapter_lambda_bias": float(adapter_lambda_bias),
        "delta_max": float(delta_max),
        "beta": float(beta),
        "steer_delta_delta_max": (
            None if steer_delta_delta_max is None else float(steer_delta_delta_max)
        ),
        "lidar_repeat_min_steps": int(lidar_repeat_min_steps),
        "lidar_repeat_max_steps": int(lidar_repeat_max_steps),
    }
    if abs(sim_timescale - 1.0) > 1e-6:
        # DonkeySim timescale changes how far the car advances between Python
        # control decisions.  Keeping the 1x controller envelope is safer than
        # scaling per-step steering/throttle changes up, which makes random PPO
        # actions jump the servo and immediately leave the track.
        lidar_repeat_min_steps = max(1, int(round(float(lidar_repeat_min_steps) / sim_timescale)))
        lidar_repeat_max_steps = max(
            lidar_repeat_min_steps,
            int(round(float(lidar_repeat_max_steps) / sim_timescale)),
        )
        print(
            "⏱️  timescale scaling: "
            f"sim_timescale={sim_timescale:.3f}, "
            f"controller_envelope=1x, "
            f"control_dt={control_dt:.3f}, "
            f"k_delta={adapter_k_delta:.3f}, "
            f"lambda_bias={adapter_lambda_bias:.3f}, "
            f"delta_max={delta_max:.3f}, "
            f"ddelta={steer_delta_delta_max}, "
            f"beta={beta:.3f}, "
            f"lidar_repeat={base_timescale_params['lidar_repeat_min_steps']}-{base_timescale_params['lidar_repeat_max_steps']}"
            f"->{lidar_repeat_min_steps}-{lidar_repeat_max_steps}"
        )
    sim2real_filter_dt_s = float(control_dt)

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
    print("🚀 DonkeyCar PPO V17 - LiDAR-First Training")
    print("=" * 76)
    print(f"maps: {env_ids}")
    lidar_obs_dim = (
        2 * int(lidar_num_sectors)
        if str(lidar_obs_mode).strip().lower() == "full"
        else 12
    )
    active_image_channels = list(image_channel_indices or [0, 1, 2, 3, 4, 5])
    print(f"obs: Dict(image={len(active_image_channels)}x{obs_size}x{obs_size}, state=7, lidar={lidar_obs_dim}, lidar_meta=2, domain_id=1)")
    print(f"track_dir: {track_dir}")
    print(f"image_channels: {active_image_channels}")
    print(f"vehicle_prob_channel: {'on' if 4 in active_image_channels else 'off'} (ch4=rawpink, fixed pink RGB=(255,105,180))")
    print(
        f"lidar: mode={lidar_obs_mode}, sectors={lidar_num_sectors}, "
        f"fov={lidar_fov_deg}, max_range={lidar_max_range_m}, "
        f"repeat={lidar_repeat_min_steps}-{lidar_repeat_max_steps}"
    )
    if curriculum_phase is not None:
        if curriculum_applied:
            print(f"curriculum_phase: {curriculum_phase} ({', '.join(sorted(curriculum_applied.keys()))})")
        else:
            print(f"curriculum_phase: {curriculum_phase} (explicit settings kept all defaults from being overridden)")

    track_geometry = TrackGeometryManager(track_dir=track_dir, env_ids=env_ids, scene_specs=SCENE_SPECS)
    if run_preflight_checks:
        run_preflight_tests(
            track_geometry=track_geometry,
            obs_size=obs_size,
            lidar_obs_mode=lidar_obs_mode,
            lidar_num_sectors=lidar_num_sectors,
            lidar_fov_deg=lidar_fov_deg,
            lidar_max_range_m=lidar_max_range_m,
            lidar_near_clip_m=lidar_near_clip_m,
        )

    _launch_sim = bool(sim_path and sim_path not in ("", "remote", "none"))
    sim_host = "127.0.0.1"
    sim_port = int(port)
    if sim_loaded_timeout_s > 0:
        _install_sim_wait_timeout_patch(
            timeout_s=float(sim_loaded_timeout_s),
            resend_scene_names_s=float(sim_wait_resend_scene_names_s),
        )
    if not _launch_sim:
        ports_to_probe = list(port_per_scene) if port_per_scene is not None else [sim_port]
        for probe_port in sorted(set(int(p) for p in ports_to_probe)):
            ok, err = _probe_sim_tcp(sim_host, probe_port, timeout_s=1.0)
            if ok:
                print(f"✅ sim tcp reachable: {sim_host}:{probe_port}")
            else:
                print(f"⚠️  sim tcp not reachable: {sim_host}:{probe_port} ({err})")

    cfg = load_config(myconfig=DEFAULT_MYCONFIG)
    if cfg is not None and hasattr(cfg, "GYM_CONF"):
        conf = cfg.GYM_CONF.copy()
    else:
        conf = {}
    conf.update(
        {
            "host": sim_host,
            "port": sim_port,
            "car_name": "waveshare_v17",
            "racer_name": "V17-LiDAR",
            "country": "CN",
            "bio": "V17 LiDAR-first PPO",
            "guid": "waveshare-v17-lidar",
            "max_cte": 8.0,
            "lidar_config": {
                "deg_per_sweep_inc": max(1.0, float(lidar_fov_deg) / max(1, lidar_num_sectors * 5)),
                "deg_ang_down": 0.0,
                "deg_ang_delta": -1.0,
                "num_sweeps_levels": 1,
                "max_range": float(lidar_max_range_m),
                "noise": 0.5,
                "offset_x": 0.0,
                "offset_y": 0.40,
                "offset_z": 0.5,
                "rot_x": 0.0,
            },
        }
    )
    if _launch_sim:
        conf["exe_path"] = sim_path

    from gym import spaces

    image_channel_indices = list(image_channel_indices or [0, 1, 2, 3, 4, 5])
    image_channel_count = len(image_channel_indices)
    lidar_obs_mode_norm = str(lidar_obs_mode).strip().lower()
    if lidar_obs_mode_norm == "full":
        lidar_low = np.concatenate([
            np.zeros((lidar_num_sectors,), dtype=np.float32),
            np.zeros((lidar_num_sectors,), dtype=np.float32),
        ])
        lidar_high = np.concatenate([
            np.full((lidar_num_sectors,), lidar_max_range_m, dtype=np.float32),
            np.ones((lidar_num_sectors,), dtype=np.float32),
        ])
        dummy_lidar_obs = np.concatenate([
            np.full((lidar_num_sectors,), lidar_max_range_m, dtype=np.float32),
            np.zeros((lidar_num_sectors,), dtype=np.float32),
        ])
    elif lidar_obs_mode_norm == "target_token":
        lidar_low = np.array(
            [0.0, -lidar_max_range_m, -lidar_max_range_m, -3.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        lidar_high = np.array(
            [1.0, lidar_max_range_m, lidar_max_range_m, 3.0, 3.0, 6.0, 1.0, 1.0, lidar_max_range_m, lidar_max_range_m, lidar_max_range_m, lidar_max_range_m],
            dtype=np.float32,
        )
        dummy_lidar_obs = np.zeros((12,), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported lidar_obs_mode={lidar_obs_mode!r}")
    lidar_obs_dim = int(lidar_low.shape[0])

    dummy_obs_space = spaces.Dict(
        {
            "image": spaces.Box(
                low=np.full((image_channel_count, obs_size, obs_size), 0.0, dtype=np.float32),
                high=np.full((image_channel_count, obs_size, obs_size), 1.0, dtype=np.float32),
                dtype=np.float32,
            ),
            "state": spaces.Box(
                low=np.full((7,), -3.0, dtype=np.float32),
                high=np.full((7,), 3.0, dtype=np.float32),
                dtype=np.float32,
            ),
            "lidar": spaces.Box(
                low=lidar_low,
                high=lidar_high,
                dtype=np.float32,
            ),
            "lidar_meta": spaces.Box(
                low=np.zeros((2,), dtype=np.float32),
                high=np.ones((2,), dtype=np.float32),
                dtype=np.float32,
            ),
            "domain_id": spaces.Box(
                low=np.zeros((1,), dtype=np.float32),
                high=np.ones((1,), dtype=np.float32),
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
                "image": np.zeros((image_channel_count, obs_size, obs_size), dtype=np.float32),
                "state": np.zeros((7,), dtype=np.float32),
                "lidar": dummy_lidar_obs.copy(),
                "lidar_meta": np.zeros((2,), dtype=np.float32),
                "domain_id": np.zeros((1,), dtype=np.float32),
            }

        def step(self, action):
            return self.reset(), 0.0, False, {}

    dummy_vec_env = DummyVecEnv([lambda: DummyEnv()])
    _safe_seed_env(dummy_vec_env, seed, label="dummy_v17_env")

    tensorboard_log_path = (
        os.path.join(save_dir, "tensorboard", exp_tag)
        if exp_tag
        else os.path.join(save_dir, "tensorboard")
    )
    policy_kwargs = dict(
        features_extractor_class=LiDARFiLMFeatureExtractor,
        features_extractor_kwargs=dict(
            image_feat_dim=128,
            state_feat_dim=32,
            lidar_feat_dim=96,
            disable_lidar_meta=bool(disable_lidar_meta),
            lidar_encoder_mode=str(lidar_encoder_mode),
        ),
        lstm_hidden_size=int(lstm_hidden_size),
        n_lstm_layers=int(lstm_layers),
        shared_lstm=False,
        enable_critic_lstm=True,
        normalize_images=False,
    )

    def _new_recurrent_model() -> Any:
        return RecurrentPPO(
            V17RecurrentMultiInputPolicy,
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
            tensorboard_log=tensorboard_log_path,
            seed=(None if seed is None else int(seed)),
        )

    resume_ckpt_path = resume_path
    if resume_ckpt_path is None and resume_latest:
        resume_ckpt_path = _find_latest_checkpoint(save_dir, name_prefix="v17")
        if resume_ckpt_path is None:
            raise FileNotFoundError(f"No v17 checkpoint found in {save_dir}")

    if resume_ckpt_path is not None:
        print(f"🔄 Resume from: {resume_ckpt_path}")
        try:
            model = RecurrentPPO.load(resume_ckpt_path, env=dummy_vec_env)
        except ValueError as e:
            msg = str(e)
            if "parameter group" not in msg or "optimizer" not in msg:
                raise
            print(
                "⚠️  Full resume failed because optimizer parameter groups do not match; "
                "falling back to policy-only resume with a fresh optimizer."
            )
            data, params, _pytorch_vars = load_from_zip_file(resume_ckpt_path, device="cpu")
            model = _new_recurrent_model()
            policy_pth_path = _policy_pth_for_checkpoint(str(resume_ckpt_path))
            if os.path.exists(policy_pth_path):
                policy_state = torch.load(policy_pth_path, map_location=model.device)
                policy_source = policy_pth_path
            else:
                policy_state = params.get("policy")
                policy_source = str(resume_ckpt_path) + "::policy"
            if policy_state is None:
                raise RuntimeError(f"Could not find policy weights for resume checkpoint {resume_ckpt_path}") from e
            incompatible = model.policy.load_state_dict(policy_state, strict=False)
            missing = list(getattr(incompatible, "missing_keys", []) or [])
            unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
            if missing or unexpected:
                print(
                    "⚠️  Policy-only resume loaded with non-strict keys: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            saved_num_timesteps = int((data or {}).get("num_timesteps") or 0)
            if saved_num_timesteps > 0:
                model.num_timesteps = saved_num_timesteps
                if hasattr(model, "_num_timesteps_at_start"):
                    model._num_timesteps_at_start = saved_num_timesteps
            print(
                "✅ Policy-only resume loaded: "
                f"source={policy_source}, num_timesteps={int(getattr(model, 'num_timesteps', 0))}"
            )
        model.tensorboard_log = tensorboard_log_path
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
        print(
            "🔧 Resume LR override: "
            f"lr_schedule=constant, learning_rate={float(learning_rate):.6g}, "
            f"target_kl={model.target_kl}, "
            f"n_steps={model.n_steps}, batch_size={model.batch_size}, n_epochs={model.n_epochs}"
        )
    else:
        model = _new_recurrent_model()
    model_start_timesteps = int(getattr(model, "num_timesteps", 0))
    dummy_vec_env.close()

    def make_env():
        return MultiSceneEnvV17(
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
            steer_delta_delta_max=steer_delta_delta_max,
            steer_servo_deadband=steer_servo_deadband,
            w_d=w_d,
            w_dd=w_dd,
            w_m=w_m,
            w_sat=w_sat,
            w_steer_budget=w_steer_budget,
            steer_budget_straight=steer_budget_straight,
            steer_budget_curve=steer_budget_curve,
            steer_budget_obstacle_relief=steer_budget_obstacle_relief,
            w_sign_flip=w_sign_flip,
            sign_flip_min_abs_steer=sign_flip_min_abs_steer,
            w_micro_wiggle=w_micro_wiggle,
            micro_wiggle_min_abs_steer=micro_wiggle_min_abs_steer,
            micro_wiggle_max_abs_steer=micro_wiggle_max_abs_steer,
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
            reward_safe_follow_bonus=reward_safe_follow_bonus,
            reward_prepare_pass_bonus=reward_prepare_pass_bonus,
            reward_commit_pass_bonus=reward_commit_pass_bonus,
            reward_post_pass_bonus=reward_post_pass_bonus,
            reward_post_pass_steps=reward_post_pass_steps,
            curriculum_phase=curriculum_phase,
            reward_overrides_by_logging_key=reward_overrides_by_logging_key,
            terminal_offtrack_progress_scale=terminal_offtrack_progress_scale,
            bad_episode_guard_min_steps=bad_episode_guard_min_steps,
            bad_episode_guard_reward_floor=bad_episode_guard_reward_floor,
            bad_episode_guard_cte_over_in_rate=bad_episode_guard_cte_over_in_rate,
            bad_episode_guard_min_forward_progress=bad_episode_guard_min_forward_progress,
            bad_episode_guard_penalty=bad_episode_guard_penalty,
            offtrack_leniency_ratio=offtrack_leniency_ratio,
            offtrack_leniency_mult=offtrack_leniency_mult,
            snapshot_dir=snapshot_dir,
            snapshot_max_steps=scene_start_snapshot_steps,
            min_episodes_per_scene=min_episodes_per_scene,
            max_steps_per_scene=max_steps_per_scene,
            scene_reload_timeout_s=scene_reload_timeout_s,
            scene_reload_post_exit_sleep_s=scene_reload_post_exit_sleep_s,
            scene_start_force_reload=scene_start_force_reload,
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
            ws_obstacle_count=ws_obstacle_count,
            obstacle_free_prob=obstacle_free_prob,
            obstacle_modes=obstacle_modes,
            ws_obstacle_free_prob=ws_obstacle_free_prob,
            obstacle_spawn_ahead_min_m=obstacle_spawn_ahead_min_m,
            obstacle_spawn_ahead_max_m=obstacle_spawn_ahead_max_m,
            obstacle_min_agent_planar_dist_m=obstacle_min_agent_planar_dist_m,
            obstacle_min_agent_arc_dist_m=obstacle_min_agent_arc_dist_m,
            obstacle_min_separation_world=obstacle_min_separation_world,
            obstacle_lateral_choices=obstacle_lateral_choices,
            ws_obstacle_lateral_choices=ws_obstacle_lateral_choices,
            obstacle_fixed_progress_ratio=obstacle_fixed_progress_ratio,
            obstacle_fixed_progress_distribution=obstacle_fixed_progress_distribution,
            obstacle_fixed_progress_gap=obstacle_fixed_progress_gap,
            obstacle_fixed_progress_gap_min=obstacle_fixed_progress_gap_min,
            obstacle_fixed_progress_gap_max=obstacle_fixed_progress_gap_max,
            obstacle_progress_min=obstacle_progress_min,
            obstacle_progress_max=obstacle_progress_max,
            obstacle_fixed_lateral_ratio=obstacle_fixed_lateral_ratio,
            gt_obstacle_start_exclusion_half_width_m=gt_obstacle_start_exclusion_half_width_m,
            ws_obstacle_modes=ws_obstacle_modes,
            ws_obstacle_fixed_progress_ratio=ws_obstacle_fixed_progress_ratio,
            ws_obstacle_fixed_progress_gap=ws_obstacle_fixed_progress_gap,
            ws_obstacle_fixed_progress_gap_min=ws_obstacle_fixed_progress_gap_min,
            ws_obstacle_fixed_progress_gap_max=ws_obstacle_fixed_progress_gap_max,
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
            image_channel_indices=(image_channel_indices or [0, 1, 2, 3, 4, 5]),
            lidar_num_sectors=lidar_num_sectors,
            lidar_fov_deg=lidar_fov_deg,
            lidar_max_range_m=lidar_max_range_m,
            lidar_near_clip_m=lidar_near_clip_m,
            lidar_repeat_min_steps=lidar_repeat_min_steps,
            lidar_repeat_max_steps=lidar_repeat_max_steps,
            lidar_obs_mode=lidar_obs_mode,
            sim2real_throttle_gain_floor=sim2real_throttle_gain_floor,
            sim2real_throttle_gain_override=sim2real_throttle_gain_override,
            sim2real_steer_gain_floor=sim2real_steer_gain_floor,
            sim2real_steer_gain_override=sim2real_steer_gain_override,
            sim2real_filter_dt_s=sim2real_filter_dt_s,
            predictive_safety_filter_path=predictive_safety_filter_path,
            predictive_safety_filter_mode=predictive_safety_filter_mode,
            predictive_safety_filter_log_path=predictive_safety_filter_log_path,
            predictive_safety_yaw_thresh=predictive_safety_yaw_thresh,
            predictive_safety_decel_thresh=predictive_safety_decel_thresh,
            port_per_scene=port_per_scene,
        )

    env = DummyVecEnv([make_env])
    _safe_seed_env(env, seed, label="v17_train_env")
    if curriculum_phase is not None:
        for envs_list in env.envs:
            if hasattr(envs_list, "_curriculum_phase"):
                envs_list._curriculum_phase = curriculum_phase
    model.set_env(env)
    if resume_ckpt_path is not None:
        _rebuild_recurrent_rollout_state(
            model,
            n_steps=int(ppo_n_steps),
            batch_size=int(ppo_batch_size),
            reason="resume",
        )

    callbacks: List[BaseCallback] = [
        PTHExportCallback(
            save_path=save_dir,
            save_freq=20000,
            name_prefix="v17",
            keep_last=checkpoint_keep_last,
            verbose=1,
        ),
        BestModelCallback(
            save_path=save_dir,
            check_freq=1000,
            name_prefix=str(best_model_name_prefix),
            metric_mode="per_scene_min",
            min_episodes_per_scene_for_save=10,
            save_separate_per_scene_best=True,
            scene_keys=None,
            save_balanced_from_training_buffer=False,
            verbose=1,
        ),
        PerSceneStatsCallback(check_freq=1000, short_episode_threshold=15, verbose=1),
        ActionDiagnosticsCallback(check_freq=1000, window=1000, verbose=0),
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
            checkpoint_prefix="v17",
            verbose=1,
        ),
        StepBudgetCompensationCallback(
            curriculum_phases=CURRICULUM_PHASES,
            window_episode_count=50,
            verbose=1,
        ),
        CriticCalibrationCallback(
            eval_freq_timesteps=int(critic_calibration_freq),
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
    print("🚦 Start V17 training")
    print("=" * 76)
    start_time = time.time()
    interrupted = False
    learn_total_timesteps = _resolve_learn_total_timesteps(
        requested_timesteps=total_timesteps,
        model_start_timesteps=model_start_timesteps,
        resume_ckpt_path=resume_ckpt_path,
    )
    try:
        model.learn(
            total_timesteps=learn_total_timesteps,
            callback=callbacks,
            progress_bar=False,
            reset_num_timesteps=(resume_ckpt_path is None),
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        error_path = os.path.join(save_dir, "training_error.json")
        error_payload = {
            "timestamp": datetime.now().isoformat(),
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
            "model_num_timesteps": int(getattr(model, "num_timesteps", model_start_timesteps)),
            "model_start_timesteps": int(model_start_timesteps),
        }
        try:
            with open(error_path, "w", encoding="utf-8") as f:
                json.dump(error_payload, f, indent=2, ensure_ascii=False)
            crash_model_path = os.path.join(save_dir, "crash_model")
            model.save(crash_model_path)
            torch.save(model.policy.state_dict(), os.path.join(save_dir, "crash_model_policy.pth"))
            print(f"\n⚠️  Training failed; error saved to {error_path}")
            print(f"⚠️  Crash checkpoint saved to {crash_model_path}.zip")
        except Exception as save_exc:
            print(f"\n⚠️  Training failed and crash save failed: {type(save_exc).__name__}: {save_exc}")
        raise

    elapsed = time.time() - start_time
    model_end_timesteps = int(getattr(model, "num_timesteps", model_start_timesteps))
    trained_timesteps = max(0, model_end_timesteps - model_start_timesteps)

    final_model_path = os.path.join(save_dir, "final_model")
    model.save(final_model_path)
    final_pth_path = os.path.join(save_dir, "final_model_policy.pth")
    torch.save(model.policy.state_dict(), final_pth_path)
    final_learning_rate = float(model.policy.optimizer.param_groups[0]["lr"])

    callback_summaries: Dict[str, Any] = {}
    for cb in callbacks:
        summary_fn = getattr(cb, "summary", None)
        if callable(summary_fn):
            try:
                callback_summaries[cb.__class__.__name__] = summary_fn()
            except Exception as e:
                callback_summaries[cb.__class__.__name__] = {"summary_error": f"{type(e).__name__}: {e}"}

    config = {
        "version": "V17",
        "timestamp": datetime.now().isoformat(),
        "curriculum": {
            "phase": curriculum_phase,
            "applied_overrides": curriculum_applied,
            "explicit_cli_keys": sorted(explicit_cli_keys or []),
        },
        "env_ids": env_ids,
        "scene_weights": scene_weights,
        "track_dir": track_dir,
        "sim": {
            "path": sim_path,
            "launch_sim": bool(_launch_sim),
            "host": sim_host,
            "port": sim_port,
            "ports": list(port_per_scene) if port_per_scene is not None else None,
            "port_per_scene": (
                [
                    {
                        "scene_idx": int(idx),
                        "env_id": str(env_id),
                        "scene_key": str(SCENE_SPECS[env_id]["scene_key"]),
                        "logging_key": str(SCENE_SPECS[env_id].get("logging_key", SCENE_SPECS[env_id]["scene_key"])),
                        "port": int(port_per_scene[idx]),
                    }
                    for idx, env_id in enumerate(env_ids)
                ]
                if port_per_scene is not None
                else None
            ),
        },
        "track_geometry_summary": track_geometry.scene_summary(),
        "observation": {
            "image_shape": [image_channel_count, obs_size, obs_size],
            "image_channels": list(image_channel_indices),
            "state_dim": 7,
            "lidar_dim": lidar_obs_dim,
            "lidar_meta_dim": 2,
            "disable_lidar_meta": bool(disable_lidar_meta),
            "lidar_encoder_mode": str(lidar_encoder_mode),
            "lidar_obs_mode": str(lidar_obs_mode),
        },
        "lidar": {
            "num_sectors": lidar_num_sectors,
            "fov_deg": lidar_fov_deg,
            "max_range_m": lidar_max_range_m,
            "near_clip_m": lidar_near_clip_m,
            "repeat_min_steps": lidar_repeat_min_steps,
            "repeat_max_steps": lidar_repeat_max_steps,
        },
        "sim2real": {
            "json": sim2real_json,
            "throttle_gain_floor": sim2real_throttle_gain_floor,
            "throttle_gain_override": sim2real_throttle_gain_override,
            "steer_gain_floor": sim2real_steer_gain_floor,
            "steer_gain_override": sim2real_steer_gain_override,
            "filter_dt_s": sim2real_filter_dt_s,
        },
        "timescale": {
            "sim_timescale": sim_timescale,
            "base": base_timescale_params,
            "scaled": {
                "control_dt": control_dt,
                "adapter_k_delta": adapter_k_delta,
                "adapter_lambda_bias": adapter_lambda_bias,
                "delta_max": delta_max,
                "beta": beta,
                "steer_delta_delta_max": steer_delta_delta_max,
                "lidar_repeat_min_steps": lidar_repeat_min_steps,
                "lidar_repeat_max_steps": lidar_repeat_max_steps,
            },
        },
        "action_adapter": {
            "k_delta": adapter_k_delta,
            "lambda_bias": adapter_lambda_bias,
            "k_bias": adapter_k_bias,
            "steer_core_decay": adapter_steer_core_decay,
            "v_nominal": adapter_v_nominal,
            "k_turn": adapter_k_turn,
            "k_bias_speed": adapter_k_bias_speed,
            "alpha_speed": adapter_alpha_speed,
            "v_min": adapter_v_min,
            "v_max": adapter_v_max,
            "max_throttle": adapter_max_throttle,
            "speed_kp": speed_kp,
            "speed_ki": speed_ki,
            "speed_kff": speed_kff,
            "control_dt": control_dt,
            "allow_reverse": allow_reverse,
        },
        "steering_safety": {
            "delta_max": delta_max,
            "enable_lpf": enable_lpf,
            "beta": beta,
            "delta_delta_max": steer_delta_delta_max,
            "servo_deadband": steer_servo_deadband,
        },
        "ppo": {
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "n_steps": ppo_n_steps,
            "batch_size": ppo_batch_size,
            "n_epochs": ppo_n_epochs,
            "clip_range": ppo_clip_range,
            "target_kl": target_kl,
            "final_learning_rate": final_learning_rate,
            "lstm_hidden_size": lstm_hidden_size,
            "lstm_layers": lstm_layers,
            "checkpoint_keep_last": checkpoint_keep_last,
            "best_model_name_prefix": str(best_model_name_prefix),
        },
        "reward": {
            "progress_reward_scale": progress_reward_scale,
            "survival_reward_scale": survival_reward_scale,
            "collision_penalty_base": collision_penalty_base,
            "offtrack_penalty_base": offtrack_penalty_base,
            "w_d": w_d,
            "w_dd": w_dd,
            "w_m": w_m,
            "w_sat": w_sat,
            "w_steer_budget": w_steer_budget,
            "steer_budget_straight": steer_budget_straight,
            "steer_budget_curve": steer_budget_curve,
            "steer_budget_obstacle_relief": steer_budget_obstacle_relief,
            "w_sign_flip": w_sign_flip,
            "sign_flip_min_abs_steer": sign_flip_min_abs_steer,
            "w_micro_wiggle": w_micro_wiggle,
            "micro_wiggle_min_abs_steer": micro_wiggle_min_abs_steer,
            "micro_wiggle_max_abs_steer": micro_wiggle_max_abs_steer,
            "w_speed_ref": w_speed_ref,
            "speed_ref_vmin": speed_ref_vmin,
            "speed_ref_vmax": speed_ref_vmax,
            "speed_ref_kappa_ref": speed_ref_kappa_ref,
            "w_near_collision": w_near_collision,
            "near_collision_start_ratio": near_collision_start_ratio,
            "overtake_success_bonus": overtake_success_bonus,
            "reward_safe_follow_bonus": reward_safe_follow_bonus,
            "reward_prepare_pass_bonus": reward_prepare_pass_bonus,
            "reward_commit_pass_bonus": reward_commit_pass_bonus,
            "reward_post_pass_bonus": reward_post_pass_bonus,
            "reward_overrides_by_logging_key": reward_overrides_by_logging_key,
            "terminal_offtrack_progress_scale": terminal_offtrack_progress_scale,
            "bad_episode_guard_min_steps": bad_episode_guard_min_steps,
            "bad_episode_guard_reward_floor": bad_episode_guard_reward_floor,
            "bad_episode_guard_cte_over_in_rate": bad_episode_guard_cte_over_in_rate,
            "bad_episode_guard_min_forward_progress": bad_episode_guard_min_forward_progress,
            "bad_episode_guard_penalty": bad_episode_guard_penalty,
        },
        "obstacle_runtime": {
            "enabled": obstacle_enabled,
            "count": obstacle_count,
            "ws_count": ws_obstacle_count,
            "free_prob": obstacle_free_prob,
            "modes": list(obstacle_modes or ["static", "jitter"]),
            "ws_free_prob": ws_obstacle_free_prob,
            "spawn_ahead_min_m": obstacle_spawn_ahead_min_m,
            "spawn_ahead_max_m": obstacle_spawn_ahead_max_m,
            "min_agent_planar_dist_m": obstacle_min_agent_planar_dist_m,
            "min_agent_arc_dist_m": obstacle_min_agent_arc_dist_m,
            "min_separation_world": obstacle_min_separation_world,
            "lateral_choices": obstacle_lateral_choices,
            "ws_lateral_choices": ws_obstacle_lateral_choices,
            "fixed_progress_ratio": obstacle_fixed_progress_ratio,
            "fixed_progress_distribution": obstacle_fixed_progress_distribution,
            "fixed_progress_gap": obstacle_fixed_progress_gap,
            "fixed_progress_gap_min": obstacle_fixed_progress_gap_min,
            "fixed_progress_gap_max": obstacle_fixed_progress_gap_max,
            "progress_min": obstacle_progress_min,
            "progress_max": obstacle_progress_max,
            "fixed_lateral_ratio": obstacle_fixed_lateral_ratio,
            "gt_start_exclusion_half_width_m": gt_obstacle_start_exclusion_half_width_m,
            "ws_modes": list(ws_obstacle_modes) if ws_obstacle_modes else None,
            "ws_fixed_progress_ratio": ws_obstacle_fixed_progress_ratio,
            "ws_fixed_progress_gap": ws_obstacle_fixed_progress_gap,
            "ws_fixed_progress_gap_min": ws_obstacle_fixed_progress_gap_min,
            "ws_fixed_progress_gap_max": ws_obstacle_fixed_progress_gap_max,
            "ws_progress_min": ws_obstacle_progress_min,
            "ws_progress_max": ws_obstacle_progress_max,
            "ws_fixed_lateral_ratio": ws_obstacle_fixed_lateral_ratio,
            "randomize_non_lane_pid_yaw": obstacle_randomize_non_lane_pid_yaw,
            "lane_pid_speed_gt": obstacle_lane_pid_speed_gt,
            "lane_pid_speed_ws": obstacle_lane_pid_speed_ws,
            "lane_pid_lookahead_m": obstacle_lane_pid_lookahead_m,
        },
        "predictive_safety_filter": {
            "path": predictive_safety_filter_path,
            "mode": predictive_safety_filter_mode,
            "log_path": predictive_safety_filter_log_path,
            "yaw_thresh": predictive_safety_yaw_thresh,
            "decel_thresh": predictive_safety_decel_thresh,
        },
        "seed": seed,
        "exp_tag": exp_tag,
        "interrupted": bool(interrupted),
        "training_time_hours": elapsed / 3600.0,
        "trained_timesteps": int(trained_timesteps),
        "num_timesteps_total": int(model_end_timesteps),
        "final_model_zip": final_model_path + ".zip",
        "final_model_pth": final_pth_path,
        "final_learning_rate": final_learning_rate,
        "callback_summaries": callback_summaries,
        "extra_run_metadata": dict(extra_run_metadata or {}),
    }
    config_path = os.path.join(save_dir, str(config_filename))
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 76)
    print("✅ V17 training finished")
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
        "final_learning_rate": final_learning_rate,
        "config_path": config_path,
        "interrupted": bool(interrupted),
        "curriculum_phase": curriculum_phase,
        "curriculum_applied": dict(curriculum_applied),
        "callback_summaries": callback_summaries,
        "extra_run_metadata": dict(extra_run_metadata or {}),
    }


def train_v17_auto_curriculum(
    total_timesteps: int = 2_000_000,
    save_dir: str = "models/v17_multi_scene_lidar",
    exp_tag: Optional[str] = None,
    resume_latest: bool = False,
    resume_path: Optional[str] = None,
    auto_curriculum_start_stage: Optional[str] = None,
    auto_curriculum_no_hard_min_gate: bool = False,
    auto_curriculum_require_gate_success: bool = False,
    **train_kwargs: Any,
) -> Dict[str, Any]:
    total_requested_timesteps = max(0, int(total_timesteps))
    remaining_timesteps = int(total_requested_timesteps)
    os.makedirs(save_dir, exist_ok=True)

    start_stage_idx = 0
    if auto_curriculum_start_stage:
        resolved_start = CURRICULUM_PHASE_ALIASES.get(
            str(auto_curriculum_start_stage).strip().lower(),
            str(auto_curriculum_start_stage).strip().lower(),
        )
        for _i, _stage in enumerate(AUTO_CURRICULUM_STAGES):
            if _stage["stage_name"] == resolved_start or _stage["phase"] == resolved_start:
                start_stage_idx = _i
                break
        else:
            available = [stage["stage_name"] for stage in AUTO_CURRICULUM_STAGES]
            raise ValueError(
                f"Unknown auto-curriculum start stage: '{auto_curriculum_start_stage}'. "
                f"Available: {available}"
            )

    print("\n" + "=" * 76)
    print("🎓 Start V17 Auto Curriculum")
    print("=" * 76)
    print(f"save_dir: {save_dir}")
    print(f"requested_steps: {total_requested_timesteps}")
    if auto_curriculum_no_hard_min_gate:
        print("gate_mode: no hard min-stage step gate; advance as soon as windows pass")
    if auto_curriculum_require_gate_success:
        print("gate_mode: max-stage fallback will stop instead of promoting without success")
    if start_stage_idx > 0:
        skipped = [AUTO_CURRICULUM_STAGES[i]["stage_name"] for i in range(start_stage_idx)]
        print(
            "⏭️  Skipping stages: "
            f"{skipped} → starting at '{AUTO_CURRICULUM_STAGES[start_stage_idx]['stage_name']}'"
        )

    stage_results: List[Dict[str, Any]] = []
    stage_resume_latest = bool(resume_latest)
    stage_resume_path = resume_path
    carried_learning_rate: Optional[float] = None

    for stage_idx, stage_def in enumerate(AUTO_CURRICULUM_STAGES):
        if stage_idx < start_stage_idx:
            continue
        if remaining_timesteps <= 0:
            break

        stage_name = str(stage_def["stage_name"])
        stage_phase = str(stage_def["phase"])
        is_final_stage = stage_idx == (len(AUTO_CURRICULUM_STAGES) - 1)
        stage_max_timesteps = int(stage_def.get("max_stage_timesteps", remaining_timesteps))
        min_stage_timesteps = 0 if auto_curriculum_no_hard_min_gate else int(
            stage_def.get("min_stage_timesteps", 0)
        )
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
                min_stage_timesteps=int(min_stage_timesteps),
                recent_episodes=int(stage_def.get("recent_episodes", 20)),
                min_success_episodes=int(stage_def.get("min_success_episodes", 12)),
                min_soft_laps=float(stage_def.get("min_soft_laps", 2.0)),
                min_episode_len=stage_def.get("min_episode_len"),
                min_progress_ratio_forward_sum=stage_def.get("min_progress_ratio_forward_sum"),
                min_reward=stage_def.get("min_reward"),
                max_episode_speed_mean=stage_def.get("max_episode_speed_mean"),
                max_episode_speed_max=stage_def.get("max_episode_speed_max"),
                required_obstacle_mode=stage_def.get("required_obstacle_mode_for_gate"),
                max_collision_rate_by_key=stage_def.get("max_collision_rate_by_key"),
                max_obstacle_clearance_critical_rate_by_key=stage_def.get(
                    "max_obstacle_clearance_critical_rate_by_key"
                ),
                max_obstacle_clearance_band_rate_by_key=stage_def.get(
                    "max_obstacle_clearance_band_rate_by_key"
                ),
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
                "stage_min_timesteps": (
                    None if is_final_stage else int(min_stage_timesteps)
                ),
                "require_gate_success": bool(auto_curriculum_require_gate_success),
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
        stage_overrides = dict(stage_def.get("train_overrides", {}) or {})
        if stage_overrides:
            stage_train_kwargs.update(stage_overrides)
            print(f"   stage_train_overrides: {stage_overrides}")
        if carried_learning_rate is not None and "learning_rate" not in stage_overrides:
            stage_train_kwargs["learning_rate"] = float(carried_learning_rate)
            print(f"   carry_learning_rate: {carried_learning_rate:.6g}")
        stage_train_kwargs["best_model_name_prefix"] = f"best_model_{stage_name}"
        if stage_idx > 0:
            stage_train_kwargs["run_preflight_checks"] = False

        stage_result = train_v17(
            total_timesteps=stage_budget_timesteps,
            save_dir=save_dir,
            exp_tag=stage_exp_tag,
            curriculum_phase=stage_phase,
            resume_latest=stage_resume_latest,
            resume_path=stage_resume_path,
            extra_callbacks=stage_callbacks,
            extra_run_metadata=stage_metadata,
            config_filename=f"v17_config_{stage_name}.json",
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
                "final_learning_rate": stage_result.get("final_learning_rate"),
                "callback_summaries": stage_result.get("callback_summaries", {}),
                "gate_summary": gate_summary,
            }
        )
        if stage_result.get("final_learning_rate") is not None:
            carried_learning_rate = float(stage_result["final_learning_rate"])

        if bool(stage_result.get("interrupted", False)):
            print(f"⚠️  auto curriculum stopped early at stage={stage_name} due to interrupt")
            break

        if (
            auto_curriculum_require_gate_success
            and gate_callback is not None
            and not bool(gate_callback.triggered)
        ):
            reason = gate_summary.get("stop_reason", "") if isinstance(gate_summary, dict) else ""
            print(
                f"🛑 auto curriculum stopped at stage={stage_name}: "
                f"gate not satisfied (reason={reason or 'unknown'})"
            )
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
        "version": "V17_AUTO_CURRICULUM",
        "timestamp": datetime.now().isoformat(),
        "save_dir": save_dir,
        "exp_tag": exp_tag,
        "total_requested_timesteps": int(total_requested_timesteps),
        "total_trained_timesteps": int(total_trained_timesteps),
        "remaining_timesteps": int(max(0, total_requested_timesteps - total_trained_timesteps)),
        "stages": stage_results,
    }
    auto_summary_path = os.path.join(save_dir, "v17_auto_curriculum_summary.json")
    with open(auto_summary_path, "w", encoding="utf-8") as f:
        json.dump(auto_summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 76)
    print("✅ V17 auto curriculum finished")
    print("=" * 76)
    print(f"summary: {auto_summary_path}")
    print(f"trained: {total_trained_timesteps}/{total_requested_timesteps}")

    auto_summary["summary_path"] = auto_summary_path
    return auto_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V17: LiDAR-first recurrent PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-ids", nargs="+", default=None)
    parser.add_argument("--scene-weights", nargs="+", type=float, default=None)
    parser.add_argument("--track-dir", type=str, default=DEFAULT_TRACK_DIR)
    parser.add_argument("--sim", type=str, default="remote")
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--save-dir", type=str, default="models/v17_multi_scene_lidar")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument(
        "--ports",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Per-scene sim ports for dual-sim persistent mode. "
            "Length must match --env-ids order (default ws,gt → 9091 9093). "
            "When set, scenes get dedicated sims and switching is just a "
            "reference swap (no close/rebuild)."
        ),
    )
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
            "Use with --resume-path or --resume-latest when continuing an existing run."
        ),
    )
    parser.add_argument(
        "--auto-curriculum-no-hard-min-gate",
        action="store_true",
        default=False,
        help=(
            "Set non-final stage min_stage_timesteps to 0 so a stage can advance "
            "as soon as its recent-episode performance window passes."
        ),
    )
    parser.add_argument(
        "--auto-curriculum-require-gate-success",
        action="store_true",
        default=False,
        help=(
            "Do not advance to the next stage on max_stage_timesteps fallback; "
            "stop auto-curriculum instead if the performance gate never passes."
        ),
    )
    parser.add_argument("--obs-size", type=int, default=128)
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--yellow-dropout-prob", type=float, default=0.20)
    parser.add_argument("--disable-lidar-meta", action="store_true", default=False)
    parser.add_argument(
        "--lidar-encoder-mode",
        type=str,
        default="side_separated",
        choices=["side_separated", "pooled"],
    )
    parser.add_argument("--lidar-obs-mode", type=str, default="full", choices=["full", "target_token"])
    parser.add_argument("--lstm-hidden-size", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--ppo-n-steps", type=int, default=4096)
    parser.add_argument("--ppo-batch-size", type=int, default=256)
    parser.add_argument("--ppo-n-epochs", type=int, default=4)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--min-episodes-per-scene", type=int, default=5)
    parser.add_argument("--max-steps-per-scene", type=int, default=2048)
    parser.add_argument(
        "--scene-reload-timeout-s",
        type=float,
        default=2.0,
        help="Timeout for direct in-sim scene reload before rebuilding the env connection.",
    )
    parser.add_argument(
        "--scene-reload-post-exit-sleep-s",
        type=float,
        default=0.3,
        help="Small wait after exit_scene before requesting scene names.",
    )
    parser.add_argument(
        "--scene-start-force-reload",
        action="store_true",
        default=False,
        help="Force an extra exit/load cycle after gym.make starts a scene.",
    )
    parser.add_argument("--delta-max", type=float, default=0.35)
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--steer-delta-delta-max", type=float, default=None)
    parser.add_argument("--steer-servo-deadband", type=float, default=0.0)
    parser.add_argument("--adapter-k-delta", type=float, default=0.15)
    parser.add_argument("--adapter-lambda-bias", type=float, default=0.20)
    parser.add_argument("--adapter-k-bias", type=float, default=0.15)
    parser.add_argument("--adapter-steer-core-decay", type=float, default=0.0)
    parser.add_argument("--adapter-v-nominal", type=float, default=V17_DRIVER_PROFILE_DEFAULTS["adapter_v_nominal"])
    parser.add_argument("--adapter-k-turn", type=float, default=V17_DRIVER_PROFILE_DEFAULTS["adapter_k_turn"])
    parser.add_argument("--adapter-alpha-speed", type=float, default=V17_DRIVER_PROFILE_DEFAULTS["adapter_alpha_speed"])
    parser.add_argument("--adapter-v-min", type=float, default=V17_DRIVER_PROFILE_DEFAULTS["adapter_v_min"])
    parser.add_argument("--adapter-v-max", type=float, default=V17_DRIVER_PROFILE_DEFAULTS["adapter_v_max"])
    parser.add_argument("--adapter-max-throttle", type=float, default=V17_DRIVER_PROFILE_DEFAULTS["adapter_max_throttle"])
    parser.add_argument("--w-d", type=float, default=0.04)
    parser.add_argument("--w-dd", type=float, default=0.01)
    parser.add_argument("--w-m", type=float, default=0.0)
    parser.add_argument("--w-sat", type=float, default=0.0)
    parser.add_argument("--w-steer-budget", type=float, default=0.0)
    parser.add_argument("--steer-budget-straight", type=float, default=0.58)
    parser.add_argument("--steer-budget-curve", type=float, default=0.88)
    parser.add_argument("--steer-budget-obstacle-relief", type=float, default=0.16)
    parser.add_argument("--w-sign-flip", type=float, default=0.0)
    parser.add_argument("--sign-flip-min-abs-steer", type=float, default=0.20)
    parser.add_argument("--w-micro-wiggle", type=float, default=0.0)
    parser.add_argument("--micro-wiggle-min-abs-steer", type=float, default=0.035)
    parser.add_argument("--micro-wiggle-max-abs-steer", type=float, default=0.22)
    parser.add_argument("--w-near-collision", type=float, default=0.24)
    parser.add_argument("--collision-penalty-base", type=float, default=8.0)
    parser.add_argument("--offtrack-penalty-base", type=float, default=5.0)
    parser.add_argument("--overtake-success-bonus", type=float, default=3.0)
    parser.add_argument("--w-speed-ref", type=float, default=0.0)
    parser.add_argument("--speed-ref-vmin", type=float, default=0.35)
    parser.add_argument("--speed-ref-vmax", type=float, default=2.2)
    parser.add_argument("--speed-ref-kappa-ref", type=float, default=0.15)
    parser.add_argument("--reward-safe-follow-bonus", type=float, default=0.02)
    parser.add_argument("--reward-prepare-pass-bonus", type=float, default=0.04)
    parser.add_argument("--reward-commit-pass-bonus", type=float, default=0.04)
    parser.add_argument("--reward-post-pass-bonus", type=float, default=0.5)
    parser.add_argument("--terminal-offtrack-progress-scale", type=float, default=1.0)
    parser.add_argument("--bad-episode-guard-min-steps", type=int, default=320)
    parser.add_argument("--bad-episode-guard-reward-floor", type=float, default=-160.0)
    parser.add_argument("--bad-episode-guard-cte-over-in-rate", type=float, default=0.25)
    parser.add_argument("--bad-episode-guard-min-forward-progress", type=float, default=0.35)
    parser.add_argument("--bad-episode-guard-penalty", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--exp-tag", type=str, default=None)
    parser.add_argument("--resume-latest", action="store_true", default=False)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--disable-preflight-checks", action="store_true", default=False)
    parser.add_argument("--disable-file-metrics-log", action="store_true", default=False)
    parser.add_argument("--file-metrics-log-freq", type=int, default=500)
    parser.add_argument("--file-metrics-log-name", type=str, default="train_metrics.jsonl")
    parser.add_argument("--disable-auto-lr-decay", action="store_true", default=False)
    parser.add_argument("--auto-lr-min", type=float, default=2e-5)
    parser.add_argument("--disable-step-balance-sampling", action="store_true", default=False)
    parser.add_argument("--checkpoint-keep-last", type=int, default=5)
    parser.add_argument("--image-channel-indices", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--lidar-num-sectors", type=int, default=36)
    parser.add_argument("--lidar-fov-deg", type=float, default=180.0)
    parser.add_argument("--lidar-max-range-m", type=float, default=20.0)
    parser.add_argument("--lidar-near-clip-m", type=float, default=0.18)
    parser.add_argument("--lidar-repeat-min-steps", type=int, default=2)
    parser.add_argument("--lidar-repeat-max-steps", type=int, default=4)
    parser.add_argument("--predictive-safety-filter-path", type=str, default=None)
    parser.add_argument("--predictive-safety-filter-mode", type=str, default="log", choices=["log", "intervene"])
    parser.add_argument("--predictive-safety-filter-log-path", type=str, default=None)
    parser.add_argument("--predictive-safety-yaw-thresh", type=float, default=None)
    parser.add_argument("--predictive-safety-decel-thresh", type=float, default=None)
    parser.add_argument("--critic-calibration-freq", type=int, default=50_000)
    parser.add_argument("--disable-obstacles", action="store_true", default=False)
    parser.add_argument("--obstacle-count", type=int, default=2)
    parser.add_argument("--ws-obstacle-count", type=int, default=None)
    parser.add_argument("--obstacle-free-prob", type=float, default=0.15)
    parser.add_argument("--obstacle-modes", nargs="+", default=None)
    parser.add_argument("--obstacle-spawn-ahead-min-m", type=float, default=3.5)
    parser.add_argument("--obstacle-spawn-ahead-max-m", type=float, default=14.0)
    parser.add_argument("--obstacle-min-agent-planar-dist-m", type=float, default=1.5)
    parser.add_argument("--obstacle-min-agent-arc-dist-m", type=float, default=3.5)
    parser.add_argument("--obstacle-min-separation-world", type=float, default=3.0)
    parser.add_argument("--obstacle-lateral-choices", nargs="+", type=float, default=None)
    parser.add_argument("--ws-obstacle-lateral-choices", nargs="+", type=float, default=None)
    parser.add_argument("--obstacle-fixed-progress-ratio", type=float, default=None)
    parser.add_argument("--obstacle-fixed-progress-gap", type=float, default=None)
    parser.add_argument("--obstacle-fixed-progress-gap-min", type=float, default=None)
    parser.add_argument("--obstacle-fixed-progress-gap-max", type=float, default=None)
    parser.add_argument("--obstacle-progress-min", type=float, default=None)
    parser.add_argument("--obstacle-progress-max", type=float, default=None)
    parser.add_argument("--obstacle-fixed-lateral-ratio", type=float, default=None)
    parser.add_argument("--gt-obstacle-start-exclusion-half-width-m", type=float, default=None)
    parser.add_argument("--ws-obstacle-free-prob", type=float, default=None)
    parser.add_argument("--ws-obstacle-modes", nargs="+", default=None)
    parser.add_argument("--ws-obstacle-fixed-progress-ratio", type=float, default=None)
    parser.add_argument("--ws-obstacle-fixed-progress-gap", type=float, default=None)
    parser.add_argument("--ws-obstacle-fixed-progress-gap-min", type=float, default=None)
    parser.add_argument("--ws-obstacle-fixed-progress-gap-max", type=float, default=None)
    parser.add_argument("--ws-obstacle-progress-min", type=float, default=None)
    parser.add_argument("--ws-obstacle-progress-max", type=float, default=None)
    parser.add_argument("--ws-obstacle-fixed-lateral-ratio", type=float, default=None)
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
    parser.add_argument("--sim2real-json", type=str, default=None)
    parser.add_argument(
        "--sim2real-throttle-gain-floor",
        type=float,
        default=0.25,
        help="Training-side floor for throttle_gain loaded from --sim2real-json; use 0 to keep JSON value",
    )
    parser.add_argument("--sim2real-throttle-gain-override", type=float, default=None)
    parser.add_argument("--sim2real-steer-gain-floor", type=float, default=None)
    parser.add_argument("--sim2real-steer-gain-override", type=float, default=None)
    parser.add_argument(
        "--sim-timescale",
        type=float,
        default=1.0,
        help="Simulator timescale used for dt-aware control/safety/sim2real scaling.",
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
        disable_lidar_meta=args.disable_lidar_meta,
        lidar_encoder_mode=args.lidar_encoder_mode,
        lidar_obs_mode=args.lidar_obs_mode,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_layers=args.lstm_layers,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        ppo_n_steps=args.ppo_n_steps,
        ppo_batch_size=args.ppo_batch_size,
        ppo_n_epochs=args.ppo_n_epochs,
        target_kl=args.target_kl,
        min_episodes_per_scene=args.min_episodes_per_scene,
        max_steps_per_scene=args.max_steps_per_scene,
        scene_reload_timeout_s=args.scene_reload_timeout_s,
        scene_reload_post_exit_sleep_s=args.scene_reload_post_exit_sleep_s,
        scene_start_force_reload=args.scene_start_force_reload,
        delta_max=args.delta_max,
        beta=args.beta,
        steer_delta_delta_max=args.steer_delta_delta_max,
        steer_servo_deadband=args.steer_servo_deadband,
        w_d=args.w_d,
        w_dd=args.w_dd,
        w_m=args.w_m,
        w_sat=args.w_sat,
        w_steer_budget=args.w_steer_budget,
        steer_budget_straight=args.steer_budget_straight,
        steer_budget_curve=args.steer_budget_curve,
        steer_budget_obstacle_relief=args.steer_budget_obstacle_relief,
        w_sign_flip=args.w_sign_flip,
        sign_flip_min_abs_steer=args.sign_flip_min_abs_steer,
        w_micro_wiggle=args.w_micro_wiggle,
        micro_wiggle_min_abs_steer=args.micro_wiggle_min_abs_steer,
        micro_wiggle_max_abs_steer=args.micro_wiggle_max_abs_steer,
        w_near_collision=args.w_near_collision,
        collision_penalty_base=args.collision_penalty_base,
        offtrack_penalty_base=args.offtrack_penalty_base,
        overtake_success_bonus=args.overtake_success_bonus,
        w_speed_ref=args.w_speed_ref,
        speed_ref_vmin=args.speed_ref_vmin,
        speed_ref_vmax=args.speed_ref_vmax,
        speed_ref_kappa_ref=args.speed_ref_kappa_ref,
        reward_safe_follow_bonus=args.reward_safe_follow_bonus,
        reward_prepare_pass_bonus=args.reward_prepare_pass_bonus,
        reward_commit_pass_bonus=args.reward_commit_pass_bonus,
        reward_post_pass_bonus=args.reward_post_pass_bonus,
        terminal_offtrack_progress_scale=args.terminal_offtrack_progress_scale,
        bad_episode_guard_min_steps=args.bad_episode_guard_min_steps,
        bad_episode_guard_reward_floor=args.bad_episode_guard_reward_floor,
        bad_episode_guard_cte_over_in_rate=args.bad_episode_guard_cte_over_in_rate,
        bad_episode_guard_min_forward_progress=args.bad_episode_guard_min_forward_progress,
        bad_episode_guard_penalty=args.bad_episode_guard_penalty,
        adapter_k_delta=args.adapter_k_delta,
        adapter_lambda_bias=args.adapter_lambda_bias,
        adapter_k_bias=args.adapter_k_bias,
        adapter_steer_core_decay=args.adapter_steer_core_decay,
        adapter_v_nominal=args.adapter_v_nominal,
        adapter_k_turn=args.adapter_k_turn,
        adapter_alpha_speed=args.adapter_alpha_speed,
        adapter_v_min=args.adapter_v_min,
        adapter_v_max=args.adapter_v_max,
        adapter_max_throttle=args.adapter_max_throttle,
        obstacle_enabled=(not args.disable_obstacles),
        obstacle_count=args.obstacle_count,
        ws_obstacle_count=args.ws_obstacle_count,
        obstacle_free_prob=args.obstacle_free_prob,
        obstacle_modes=args.obstacle_modes,
        obstacle_spawn_ahead_min_m=args.obstacle_spawn_ahead_min_m,
        obstacle_spawn_ahead_max_m=args.obstacle_spawn_ahead_max_m,
        obstacle_min_agent_planar_dist_m=args.obstacle_min_agent_planar_dist_m,
        obstacle_min_agent_arc_dist_m=args.obstacle_min_agent_arc_dist_m,
        obstacle_min_separation_world=args.obstacle_min_separation_world,
        obstacle_lateral_choices=args.obstacle_lateral_choices,
        ws_obstacle_lateral_choices=args.ws_obstacle_lateral_choices,
        obstacle_fixed_progress_ratio=args.obstacle_fixed_progress_ratio,
        obstacle_fixed_progress_gap=args.obstacle_fixed_progress_gap,
        obstacle_fixed_progress_gap_min=args.obstacle_fixed_progress_gap_min,
        obstacle_fixed_progress_gap_max=args.obstacle_fixed_progress_gap_max,
        obstacle_progress_min=args.obstacle_progress_min,
        obstacle_progress_max=args.obstacle_progress_max,
        obstacle_fixed_lateral_ratio=args.obstacle_fixed_lateral_ratio,
        gt_obstacle_start_exclusion_half_width_m=args.gt_obstacle_start_exclusion_half_width_m,
        ws_obstacle_free_prob=args.ws_obstacle_free_prob,
        ws_obstacle_modes=args.ws_obstacle_modes,
        ws_obstacle_fixed_progress_ratio=args.ws_obstacle_fixed_progress_ratio,
        ws_obstacle_fixed_progress_gap=args.ws_obstacle_fixed_progress_gap,
        ws_obstacle_fixed_progress_gap_min=args.ws_obstacle_fixed_progress_gap_min,
        ws_obstacle_fixed_progress_gap_max=args.ws_obstacle_fixed_progress_gap_max,
        ws_obstacle_progress_min=args.ws_obstacle_progress_min,
        ws_obstacle_progress_max=args.ws_obstacle_progress_max,
        ws_obstacle_fixed_lateral_ratio=args.ws_obstacle_fixed_lateral_ratio,
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
        sim_timescale=args.sim_timescale,
        sim2real_throttle_gain_floor=args.sim2real_throttle_gain_floor,
        sim2real_throttle_gain_override=args.sim2real_throttle_gain_override,
        sim2real_steer_gain_floor=args.sim2real_steer_gain_floor,
        sim2real_steer_gain_override=args.sim2real_steer_gain_override,
        image_channel_indices=args.image_channel_indices,
        lidar_num_sectors=args.lidar_num_sectors,
        lidar_fov_deg=args.lidar_fov_deg,
        lidar_max_range_m=args.lidar_max_range_m,
        lidar_near_clip_m=args.lidar_near_clip_m,
        lidar_repeat_min_steps=args.lidar_repeat_min_steps,
        lidar_repeat_max_steps=args.lidar_repeat_max_steps,
        predictive_safety_filter_path=args.predictive_safety_filter_path,
        predictive_safety_filter_mode=args.predictive_safety_filter_mode,
        predictive_safety_filter_log_path=args.predictive_safety_filter_log_path,
        predictive_safety_yaw_thresh=args.predictive_safety_yaw_thresh,
        predictive_safety_decel_thresh=args.predictive_safety_decel_thresh,
        port_per_scene=args.ports,
        critic_calibration_freq=args.critic_calibration_freq,
        seed=args.seed,
        exp_tag=args.exp_tag,
        resume_latest=args.resume_latest,
        resume_path=args.resume_path,
        run_preflight_checks=(not args.disable_preflight_checks),
        enable_file_metrics_log=(not args.disable_file_metrics_log),
        file_metrics_log_freq=args.file_metrics_log_freq,
        file_metrics_log_name=args.file_metrics_log_name,
        enable_auto_lr_decay=(not args.disable_auto_lr_decay),
        auto_lr_min=args.auto_lr_min,
        enable_step_balance_sampling=(not args.disable_step_balance_sampling),
        checkpoint_keep_last=args.checkpoint_keep_last,
    )
    if args.auto_curriculum:
        train_v17_auto_curriculum(
            auto_curriculum_start_stage=args.auto_curriculum_start_stage,
            auto_curriculum_no_hard_min_gate=args.auto_curriculum_no_hard_min_gate,
            auto_curriculum_require_gate_success=args.auto_curriculum_require_gate_success,
            **common_kwargs,
        )
    else:
        explicit_cli_keys = _explicit_curriculum_keys_from_argv(sys.argv[1:])
        train_v17(
            curriculum_phase=manual_curriculum_phase,
            explicit_cli_keys=explicit_cli_keys,
            **common_kwargs,
        )


if __name__ == "__main__":
    main()
