#!/usr/bin/env python3
"""
DonkeyCar PPO V16 — Multi-Simulator Parallel Training
======================================================
与 ppo_multitrack_v16.py 功能完全相同，额外支持 --ports 参数，
同时连接多个模拟器实例（SubprocVecEnv），理论提速 N_envs 倍。

不修改任何原有文件，所有逻辑通过 import 复用 ppo_multitrack_v16。

用法示例:
  # 先启动3个模拟器
  bash ~/bin/start_donkey_vnc_multi.sh

  # 使用3个模拟器训练（端口9093/9095/9097）
  python src/ppo_multitrack_v16_multisim.py \
      --ports 9093 9095 9097 \
      --auto-curriculum \
      --steps 2000000

  # 仍然支持单端口模式（等价于原脚本）
  python src/ppo_multitrack_v16_multisim.py --port 9091 --steps 2000000

  # 若未显式指定 --auto-curriculum / --curriculum-phase，
  # 则使用默认平坦配置直接训练（obstacle_count=2, obstacle_free_prob=0.15）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── sys.path 与原脚本保持一致 ──────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
_repo_root = str(REPO_ROOT)
while _repo_root in sys.path:
    sys.path.remove(_repo_root)
sys.path.insert(0, _repo_root)

# ── 第三方 ────────────────────────────────────────────────────────────────────
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

try:
    from sb3_contrib import RecurrentPPO
except Exception:
    RecurrentPPO = None

# ── 从原模块直接 import（不修改原文件）──────────────────────────────────────────
# 定义在 ppo_multitrack_v16.py 里的常量与帮助函数
import ppo_multitrack_v16 as _v16
from ppo_multitrack_v16 import (
    # 常量
    SCENE_SPECS,
    DEFAULT_ENV_IDS,
    DEFAULT_TRACK_DIR,
    DEFAULT_MYCONFIG,
    WS_REWARD_OVERRIDES_V16,
    WS_FINISH_OBSTACLE_PROGRESS_RATIO_V16,
    CURRICULUM_PHASES,
    CURRICULUM_PHASE_ALIASES,
    AUTO_CURRICULUM_STAGES,
    # 帮助函数（定义在 v16 文件中）
    _apply_curriculum_phase,
    _resolve_track_dir,
    run_preflight_tests,
    _install_sim_wait_timeout_patch,
    _probe_sim_tcp,
    # Callback 类（v16 中定义或 re-export）
    CurriculumWindowAdvanceCallback,
)

# 从各自原始模块 import
import gym
import gym_donkeycar  # noqa: F401
from module.actor import FiLMFeatureExtractor
from module.callbacks import (
    AdaptiveLearningRateCallback,
    BestModelCallback,
    CrashRecoveryCallback,
    PerSceneStatsCallback,
    PTHExportCallback,
    SceneSchedulerLoggingCallback,
    ShortEpisodeLoggerCallback,
    TqdmProgressCallback,
    TrainingMetricsFileLoggerCallback,
)
from module.multi_scene_env import MultiSceneEnvV16
from module.track import TrackGeometryManager
from module.utils import (
    _find_latest_checkpoint,
    _safe_seed_env,
    _seed_everything,
    load_config,
)

# ── WS 障碍位置修正（直接修改原模块全局变量）─────────────────────────────────
# WS 是环形赛道（loop_len≈8.3m），progress∈[0,1) 首尾相连。
# 原始 WS_FINISH_OBSTACLE_PROGRESS_RATIO_V16 = 0.08：
#   min_arc = min(0.08, 0.92) × 8.3 = 0.664m  → 障碍在起点前 0.664m，即时碰撞
#
# 正确做法：progress=0.5（圆弧对面），min_arc = 0.5×8.3 = 4.15m > 3.0m ✓
# 车需要跑半圈（4.15m）才能遇到障碍，validity check 通过，fallback 不触发。
#
# 关键：_apply_curriculum_phase() 定义在 ppo_multitrack_v16.py 里，
# 读的是该模块自己的 CURRICULUM_PHASES 全局变量。
# 在本模块 deepcopy 只改了本地副本，不影响 v16 模块内部的 _apply_curriculum_phase。
# 必须直接修改 _v16.CURRICULUM_PHASES 才能生效。
_v16.CURRICULUM_PHASES["warmup"]["ws_obstacle_fixed_progress_ratio"] = 0.5
# warmup 阶段原 obstacle_min_agent_arc_dist_m=5.5m > WS 半圈=4.15m → fallback 必然触发
# 改为 3.0m；配合 progress=0.5 时 min_arc=4.15m > 3.0m，validity 必然通过
_v16.CURRICULUM_PHASES["warmup"]["obstacle_min_agent_arc_dist_m"] = 3.0
# warmup 阶段禁用 WS 障碍：
#   1. WS 是环形小赛道（8.3m），障碍 TCP 操作（place_pose 1.5s 超时）拖慢 SubprocVecEnv 的所有 env
#   2. teleport_pose 失败时悄悄 pass → 障碍车残留在赛道 → "撞到锥桶不动" 视觉问题
#   3. warmup 课程本就是练习基础驾驶，WS 障碍属于后期阶段
_v16.CURRICULUM_PHASES["warmup"]["ws_obstacle_free_prob"] = 1.0


# ══════════════════════════════════════════════════════════════════════════════
# train_v16_multisim
# ══════════════════════════════════════════════════════════════════════════════

def train_v16_multisim(
    # ── 与 train_v16 完全相同的参数列表 ──────────────────────────────────────
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
    offtrack_leniency_ratio: float = 0.25,
    offtrack_leniency_mult: float = 2.5,
    reset_env_done_grace_steps: int = 3,
    reset_collision_grace_steps: int = 2,
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
    obstacle_progress_min: Optional[float] = None,
    obstacle_progress_max: Optional[float] = None,
    obstacle_fixed_lateral_ratio: Optional[float] = None,
    ws_obstacle_modes: Optional[List[str]] = None,
    ws_obstacle_fixed_progress_ratio: Optional[float] = None,
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
    vec_env_start_method: str = "spawn",
    extra_callbacks: Optional[List[BaseCallback]] = None,
    extra_run_metadata: Optional[Dict[str, Any]] = None,
    config_filename: str = "v16_config.json",
    # ── 新增：多端口支持 ───────────────────────────────────────────────────────
    ports: Optional[List[int]] = None,
):
    """
    与 train_v16() 完全相同，额外支持 ports 参数。
    当 ports 包含多个端口时，使用 SubprocVecEnv 并行连接多个模拟器实例。
    """
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
        "obstacle_progress_min": obstacle_progress_min,
        "obstacle_progress_max": obstacle_progress_max,
        "obstacle_fixed_lateral_ratio": obstacle_fixed_lateral_ratio,
        "ws_obstacle_modes": ws_obstacle_modes,
        "ws_obstacle_fixed_progress_ratio": ws_obstacle_fixed_progress_ratio,
        "ws_obstacle_fixed_lateral_ratio": ws_obstacle_fixed_lateral_ratio,
        "obstacle_randomize_non_lane_pid_yaw": obstacle_randomize_non_lane_pid_yaw,
        "obstacle_lane_pid_speed_gt": obstacle_lane_pid_speed_gt,
        "obstacle_lane_pid_speed_ws": obstacle_lane_pid_speed_ws,
        "collision_penalty_base": collision_penalty_base,
        "offtrack_penalty_base": offtrack_penalty_base,
        "w_near_collision": w_near_collision,
        "near_collision_start_ratio": near_collision_start_ratio,
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
    obstacle_progress_min = curriculum_values.get("obstacle_progress_min", None)
    obstacle_progress_max = curriculum_values.get("obstacle_progress_max", None)
    obstacle_fixed_lateral_ratio = curriculum_values["obstacle_fixed_lateral_ratio"]
    ws_obstacle_modes = curriculum_values["ws_obstacle_modes"]
    ws_obstacle_fixed_progress_ratio = curriculum_values["ws_obstacle_fixed_progress_ratio"]
    ws_obstacle_fixed_lateral_ratio = curriculum_values["ws_obstacle_fixed_lateral_ratio"]
    obstacle_randomize_non_lane_pid_yaw = curriculum_values["obstacle_randomize_non_lane_pid_yaw"]
    obstacle_lane_pid_speed_gt = curriculum_values["obstacle_lane_pid_speed_gt"]
    obstacle_lane_pid_speed_ws = curriculum_values["obstacle_lane_pid_speed_ws"]
    collision_penalty_base = curriculum_values["collision_penalty_base"]
    offtrack_penalty_base = curriculum_values["offtrack_penalty_base"]
    w_near_collision = curriculum_values["w_near_collision"]
    near_collision_start_ratio = curriculum_values["near_collision_start_ratio"]

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

    # ── 解析端口列表（新增逻辑）────────────────────────────────────────────────
    _launch_sim = bool(sim_path and sim_path not in ("", "remote", "none"))
    sim_host = "127.0.0.1"
    sim_port = int(port)
    _ports: List[int] = list(ports) if ports else [sim_port]
    n_envs = len(_ports)

    print("\n" + "=" * 76)
    print("🚀 DonkeyCar PPO V16 - Multi-Sim Parallel Training")
    print("=" * 76)
    print(f"maps: {env_ids}")
    print(f"obs: Dict(image=6x{obs_size}x{obs_size}, state=12)")
    print(f"track_dir: {track_dir}")
    print(f"n_envs: {n_envs}  ports: {_ports}")
    auto_stage_enabled = bool((extra_run_metadata or {}).get("auto_curriculum", {}).get("enabled"))
    if curriculum_phase is not None:
        if curriculum_applied:
            print(f"curriculum_phase: {curriculum_phase} ({', '.join(sorted(curriculum_applied.keys()))})")
        else:
            print(f"curriculum_phase: {curriculum_phase} (explicit settings kept all defaults from being overridden)")
    elif not auto_stage_enabled:
        print(
            "⚠️  No curriculum selected: using flat obstacle defaults "
            f"(count={obstacle_count}, free_prob={obstacle_free_prob}, modes={obstacle_modes or ['static', 'jitter']})."
        )
        print("   Add --auto-curriculum to follow the staged WS/GT obstacle curriculum.")
    print(f"obstacle: enabled={obstacle_enabled}, count={obstacle_count}, modes={obstacle_modes or ['static', 'jitter']}")
    if reset_env_done_grace_steps > 0 or reset_collision_grace_steps > 0:
        print(
            "reset safeguards: "
            f"env_done_grace_steps={int(reset_env_done_grace_steps)}, "
            f"collision_grace_steps={int(reset_collision_grace_steps)}"
        )

    track_geometry = TrackGeometryManager(track_dir=track_dir, env_ids=env_ids, scene_specs=SCENE_SPECS)
    if run_preflight_checks:
        run_preflight_tests(track_geometry=track_geometry, obs_size=obs_size)

    if sim_loaded_timeout_s > 0:
        _install_sim_wait_timeout_patch(
            timeout_s=float(sim_loaded_timeout_s),
            resend_scene_names_s=float(sim_wait_resend_scene_names_s),
        )

    # ── TCP 探测（探测所有端口）──────────────────────────────────────────────
    if not _launch_sim:
        for p in _ports:
            ok, err = _probe_sim_tcp(sim_host, p, timeout_s=1.0)
            if ok:
                print(f"✅ sim tcp reachable: {sim_host}:{p}")
            else:
                print(f"⚠️  sim tcp not reachable: {sim_host}:{p} ({err})")

    # ── conf 基础（以第一个端口为基础；各 env 的 conf 在工厂函数中覆盖端口）──
    cfg = load_config(myconfig=DEFAULT_MYCONFIG)
    if cfg is not None and hasattr(cfg, "GYM_CONF"):
        conf = cfg.GYM_CONF.copy()
        conf.update(
            {
                "port": _ports[0],
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
            "port": _ports[0],
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

    # n_envs 份 DummyEnv，确保 model 初始化时 n_envs 与后续 SubprocVecEnv 一致
    dummy_vec_env = DummyVecEnv([lambda: DummyEnv()] * n_envs)
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

    # ── env 工厂函数（核心改动：每个 env 使用独立端口）────────────────────────
    _repo_root_str = str(REPO_ROOT)  # 捕获进 closure，spawn 子进程重建 sys.path 用

    def make_env_factory(env_port: int):
        """为指定端口创建 env 工厂闭包。conf 浅拷贝后覆盖 port。"""
        env_conf = dict(conf)
        env_conf["port"] = env_port

        def _make():
            # spawn 模式下子进程 sys.path 不继承父进程动态修改，手动补充
            import sys as _sys
            if _repo_root_str not in _sys.path:
                _sys.path.insert(0, _repo_root_str)
            import gym_donkeycar  # noqa: F401 — 确保 gym 注册
            from module.multi_scene_env import MultiSceneEnvV16 as _MSE
            return _MSE(
                env_ids=env_ids,
                conf=env_conf,
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
                offtrack_leniency_ratio=offtrack_leniency_ratio,
                offtrack_leniency_mult=offtrack_leniency_mult,
                reset_env_done_grace_steps=reset_env_done_grace_steps,
                reset_collision_grace_steps=reset_collision_grace_steps,
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
                obstacle_progress_min=obstacle_progress_min,
                obstacle_progress_max=obstacle_progress_max,
                obstacle_fixed_lateral_ratio=obstacle_fixed_lateral_ratio,
                ws_obstacle_modes=ws_obstacle_modes,
                ws_obstacle_fixed_progress_ratio=ws_obstacle_fixed_progress_ratio,
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
            )
        return _make

    env_fns = [make_env_factory(p) for p in _ports]

    # ── VecEnv 创建（核心改动）────────────────────────────────────────────────
    if n_envs > 1:
        vec_env_start_method = str(vec_env_start_method or "spawn").strip().lower()
        if vec_env_start_method not in {"fork", "forkserver", "spawn"}:
            raise ValueError(f"Unsupported vec_env_start_method: {vec_env_start_method}")
        print(
            f"🚀 Using SubprocVecEnv with {n_envs} envs on ports {_ports} "
            f"(start_method={vec_env_start_method})"
        )
        env = SubprocVecEnv(env_fns, start_method=vec_env_start_method)
    else:
        env = DummyVecEnv(env_fns)
    _safe_seed_env(env, seed, label="v16_train_env")
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
    print("🚦 Start V16 Multi-Sim training")
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
        "version": "V16_MULTISIM",
        "timestamp": datetime.now().isoformat(),
        "n_envs": n_envs,
        "ports": _ports,
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
            "progress_min": obstacle_progress_min,
            "progress_max": obstacle_progress_max,
            "fixed_lateral_ratio": obstacle_fixed_lateral_ratio,
            "ws_modes": list(ws_obstacle_modes) if ws_obstacle_modes else None,
            "ws_fixed_progress_ratio": ws_obstacle_fixed_progress_ratio,
            "ws_fixed_lateral_ratio": ws_obstacle_fixed_lateral_ratio,
            "randomize_non_lane_pid_yaw": obstacle_randomize_non_lane_pid_yaw,
            "lane_pid_speed_gt": obstacle_lane_pid_speed_gt,
            "lane_pid_speed_ws": obstacle_lane_pid_speed_ws,
            "lane_pid_lookahead_m": obstacle_lane_pid_lookahead_m,
            "ego_random_spawn": ego_random_spawn,
            "ego_spawn_lateral_ratio": ego_spawn_lateral_ratio,
        },
        "runtime_safeguards": {
            "vec_env_start_method": vec_env_start_method,
            "reset_env_done_grace_steps": reset_env_done_grace_steps,
            "reset_collision_grace_steps": reset_collision_grace_steps,
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
    print("✅ V16 Multi-Sim training finished")
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


# ══════════════════════════════════════════════════════════════════════════════
# Auto-curriculum wrapper（与 train_v16_auto_curriculum 等价，调用 multisim 版本）
# ══════════════════════════════════════════════════════════════════════════════

def train_v16_multisim_auto_curriculum(
    total_timesteps: int = 2_000_000,
    save_dir: str = "models/v16_multi_scene_obstacle",
    exp_tag: Optional[str] = None,
    resume_latest: bool = False,
    resume_path: Optional[str] = None,
    **train_kwargs: Any,
) -> Dict[str, Any]:
    total_requested_timesteps = max(0, int(total_timesteps))
    remaining_timesteps = int(total_requested_timesteps)
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 76)
    print("🎓 Start V16 Multi-Sim Auto Curriculum")
    print("=" * 76)
    print(f"save_dir: {save_dir}")
    print(f"requested_steps: {total_requested_timesteps}")

    stage_results: List[Dict[str, Any]] = []
    stage_resume_latest = bool(resume_latest)
    stage_resume_path = resume_path

    for stage_idx, stage_def in enumerate(AUTO_CURRICULUM_STAGES):
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

        # 调用 multisim 版本而非原 train_v16
        stage_result = train_v16_multisim(
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
        "version": "V16_MULTISIM_AUTO_CURRICULUM",
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
    print("✅ V16 Multi-Sim auto curriculum finished")
    print("=" * 76)
    print(f"summary: {auto_summary_path}")
    print(f"trained: {total_trained_timesteps}/{total_requested_timesteps}")

    auto_summary["summary_path"] = auto_summary_path
    return auto_summary


# ══════════════════════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="V16 Multi-Sim: dual-domain recurrent PPO with parallel simulators",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-ids", nargs="+", default=None)
    parser.add_argument("--scene-weights", nargs="+", type=float, default=None)
    parser.add_argument("--track-dir", type=str, default=DEFAULT_TRACK_DIR)
    parser.add_argument("--sim", type=str, default="remote")
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--save-dir", type=str, default="models/v16_multi_scene_obstacle")
    parser.add_argument("--port", type=int, default=9091,
                        help="单端口模式端口号（未指定 --ports 时使用）")
    parser.add_argument(
        "--ports", nargs="+", type=int, default=None,
        help="多模拟器端口列表，例如 --ports 9093 9095 9097。指定后忽略 --port。"
    )
    parser.add_argument(
        "--curriculum-phase",
        type=str,
        default=None,
        choices=["none"] + sorted(CURRICULUM_PHASES.keys()) + sorted(CURRICULUM_PHASE_ALIASES.keys()),
        help="Apply a built-in training curriculum phase.",
    )
    parser.add_argument(
        "--auto-curriculum",
        action="store_true",
        default=False,
        help="Run the built-in window-gated multi-stage curriculum automatically.",
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
    parser.add_argument("--obstacle-modes", nargs="+", default=["static", "jitter"])
    parser.add_argument("--obstacle-spawn-ahead-min-m", type=float, default=3.5)
    parser.add_argument("--obstacle-spawn-ahead-max-m", type=float, default=14.0)
    parser.add_argument("--obstacle-min-agent-planar-dist-m", type=float, default=1.5)
    parser.add_argument("--obstacle-min-agent-arc-dist-m", type=float, default=3.5)
    parser.add_argument("--obstacle-min-separation-world", type=float, default=3.0)
    parser.add_argument("--obstacle-fixed-progress-ratio", type=float, default=None)
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

    args = parser.parse_args()

    manual_curriculum_phase = args.curriculum_phase
    if manual_curriculum_phase == "none":
        manual_curriculum_phase = None
    if args.auto_curriculum and manual_curriculum_phase is not None:
        raise ValueError("--auto-curriculum cannot be combined with --curriculum-phase")
    if (not args.auto_curriculum) and manual_curriculum_phase is None:
        print(
            "⚠️  Neither --auto-curriculum nor --curriculum-phase was provided. "
            "This run will start from the flat default obstacle setup."
        )
        print(
            "   If you want the staged curriculum, use: "
            "python src/ppo_multitrack_v16_multisim.py --ports ... --auto-curriculum --steps ..."
        )

    # 多端口优先；未指定 --ports 则退化为单端口
    resolved_ports = args.ports if args.ports else None

    common_kwargs = dict(
        env_ids=args.env_ids,
        scene_weights=args.scene_weights,
        track_dir=args.track_dir,
        sim_path=args.sim,
        total_timesteps=args.steps,
        save_dir=args.save_dir,
        port=args.port,
        ports=resolved_ports,
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
        train_v16_multisim_auto_curriculum(**common_kwargs)
    else:
        train_v16_multisim(curriculum_phase=manual_curriculum_phase, **common_kwargs)


if __name__ == "__main__":
    main()
