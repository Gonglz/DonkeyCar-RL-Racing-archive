"""
module/reward.py
DonkeyRewardWrapper：DonkeyCar 统一奖励包装器。

所有奖励和惩罚逻辑均在本文件中定义，不依赖外部版本脚本。
详细说明见 docs/reward.md。
"""

import math
from collections import deque
from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np


# ============================================================
# 统一奖励包装器
# ============================================================
class DonkeyRewardWrapper(gym.Wrapper):
    """
    DonkeyCar 统一奖励包装器。

    奖励构成：
      survival      = survival_reward_scale * speed_gate * 1[progress>0]
      speed         = 0.25 * ontrack * speed_gate * center_factor * 1[progress>0]
      progress      = progress_reward_scale * signed_progress_ratio（按赛道弧长）
      cte           = 超界惩罚 | 在界奖励（× speed_gate_cte）
      lap           = 完圈奖励
      center        = -w_center * |lat_err_norm|
      heading       = -w_heading * |heading_err| / pi
      speed_ref     = -w_speed_ref * ((v-v_ref(kappa))/v_ref_max)^2
      time          = -w_time
      near_offtrack = 出界前线性惩罚（cte 接近 out 边界）
      near_collision= 碰撞前风险线性惩罚（obstacle/heading/speed/control）
      safe_follow   = 安全跟车区间的小额正奖励
      overtake      = 成功绕过并超过障碍车后的 bonus
      post_pass     = 超车后保持稳定若干步的确认奖励
      collision     = 碰撞/stuck/offtrack 终止惩罚
      smooth        = -w_d  * |Δsteer_exec|
      jerk          = -w_dd * |jerk|
      mismatch      = -w_m  * |steer_raw - steer_exec|
      sat           = -w_sat * tanh(rate_excess)
      steer_budget  = -w_steer_budget * excess(|steer_exec| over curvature/risk budget)^2
      sign_flip     = -w_sign_flip when steer_exec changes sign abruptly
      micro_wiggle  = -w_micro_wiggle for low-amplitude steering chatter

    CTE 边界符号约定：cte_left > 0（左侧），cte_right < 0（右侧），与 lat_err 方向一致。
    step() 使用 lat_err_cte = lat_err * coord_scale 作为有符号横偏距与边界比较。
    详细参数说明见 docs/reward.md。
    """

    def __init__(
        self,
        env,
        total_timesteps: int = 200000,
        action_safety_wrapper=None,
        w_d: float = 0.0,
        w_dd: float = 0.0,
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
        w_time: float = 0.0,
        w_center: float = 0.0,
        w_heading: float = 0.0,
        w_speed_ref: float = 0.0,
        speed_ref_vmin: float = 0.35,
        speed_ref_vmax: float = 2.2,
        speed_ref_kappa_ref: float = 0.15,
        lap_reward_scale: float = 1.0,
        progress_reward_scale: float = 80.0,
        progress_curve_boost: float = 0.35,
        progress_kappa_ref: float = 0.15,
        progress_center_gate_min: float = 0.10,
        progress_center_gate_power: float = 1.0,
        smooth_curve_relief: float = 0.5,
        throttle_penalty_threshold: float = 1.0,
        throttle_penalty_amount: float = 0.0,
        survival_reward_scale: float = 0.2,
        collision_penalty_base: float = 8.0,
        offtrack_penalty_base: float = 6.0,
        w_near_offtrack: float = 0.40,
        near_offtrack_start_ratio: float = 0.45,
        w_near_collision: float = 0.35,
        near_collision_start_ratio: float = 0.65,
        overtake_success_bonus: float = 2.5,
        safe_follow_bonus_scale: float = 0.0,
        prepare_pass_bonus_scale: float = 0.0,
        commit_pass_bonus_scale: float = 0.0,
        safe_follow_min_m: float = 1.6,
        safe_follow_max_m: float = 4.5,
        safe_follow_risk_max: float = 0.35,
        safe_follow_speed_min: float = 0.22,
        safe_follow_ttc_min_s: float = 1.0,
        safe_follow_ttc_max_s: float = 2.4,
        ttc_penalty_start_s: float = 1.6,
        ttc_penalty_full_s: float = 0.85,
        lateral_overlap_ref_m: float = 0.30,
        reward_control_dt_s: float = 0.05,
        post_pass_stability_bonus: float = 0.0,
        post_pass_stability_steps: int = 12,
        cte_left: float = 5.0,
        cte_right: float = -5.0,
        cte_left_out: Optional[float] = None,
        cte_right_out: Optional[float] = None,
        coord_scale: float = 8.0,
        offtrack_leniency_ratio: float = 0.25,
        offtrack_leniency_mult: float = 2.5,
        offtrack_grace_steps: int = 3,
        offtrack_severe_ratio: float = 2.0,
        offtrack_grace_penalty_scale: float = 0.0,
        offtrack_grace_use_leniency: bool = True,
        track_geometry=None,
        scene_key: str = "",
        logging_key: str = "",
        cte_half_width: float = 4.6,
        cte_norm_scale: Optional[float] = None,
        reward_decay_ref_steps: int = 0,
        terminal_offtrack_progress_scale: float = 1.0,
        bad_episode_guard_min_steps: int = 0,
        bad_episode_guard_reward_floor: float = -200.0,
        bad_episode_guard_cte_over_in_rate: float = 0.25,
        bad_episode_guard_min_forward_progress: float = 0.25,
        bad_episode_guard_penalty: float = 4.0,
        collision_episode_reward_cap: Optional[float] = None,
        offtrack_episode_reward_cap: Optional[float] = None,
        wait_window_bonus_scale: float = 0.0,
        wait_window_min_gap_m: float = 1.2,
        wait_window_max_gap_m: float = 6.0,
        wait_window_max_closing_rate: float = 0.30,
        force_pass_penalty_scale: float = 0.0,
        unsafe_close_penalty_scale: float = 0.0,
        obstacle_clearance_penalty_scale: float = 0.0,
        obstacle_clearance_inner_m: float = 0.30,
        obstacle_clearance_outer_m: float = 0.60,
        post_pass_cut_in_penalty_scale: float = 0.0,
        post_pass_watch_longitudinal_m: float = 1.20,
        post_pass_watch_steps: int = 18,
        overtake_success_min_progress_ratio: float = 1e-6,
        unsafe_close_gap_m: float = 2.6,
        unsafe_close_clearance_m: float = 0.30,
        unsafe_close_longitudinal_m: float = 0.50,
        unsafe_close_ttc_s: float = 2.2,
        overtake_arm_longitudinal_min_m: float = 0.8,
        overtake_arm_planar_max_m: float = 8.0,
        overtake_pass_longitudinal_threshold_m: float = -0.8,
        overtake_pass_planar_min_m: float = 0.9,
        close_front_planar_max_m: float = 7.0,
        force_pass_planar_max_m: float = 7.0,
        stuck_speed_threshold: float = 0.10,
        stuck_progress_threshold: float = 4e-4,
        stuck_grace_steps: int = 30,
        stuck_low_speed_penalty_start: int = 10,
        stuck_low_speed_penalty_scale: float = 0.04,
        stuck_low_speed_penalty_cap: float = 0.8,
        stuck_penalty_base: float = 10.0,
        stuck_penalty_growth: float = 0.5,
        stuck_penalty_cap: float = 14.0,
        enable_step_diagnostics: bool = False,
        step_diagnostics_first_steps: int = 3,
        step_diagnostics_every_episodes: int = 0,
        reset_env_done_grace_steps: int = 0,
        reset_collision_grace_steps: int = 0,
    ):
        super().__init__(env)
        self.total_timesteps = total_timesteps
        self.current_step = 0
        self.action_safety_wrapper = action_safety_wrapper
        # 轨迹几何：用于计算 lat_err_cte = lat_err * coord_scale（有符号横偏距，与 CTE 表同单位）
        self._track_geometry = track_geometry
        self._scene_key = scene_key
        self._logging_key = str(logging_key or scene_key or "")
        self._prev_track_idx = None
        self.coord_scale = float(max(coord_scale, 1e-3))
        self.enable_step_diagnostics = bool(enable_step_diagnostics)
        self.step_diagnostics_first_steps = max(1, int(step_diagnostics_first_steps))
        self.step_diagnostics_every_episodes = max(0, int(step_diagnostics_every_episodes))
        self.reset_env_done_grace_steps = max(0, int(reset_env_done_grace_steps))
        self.reset_collision_grace_steps = max(0, int(reset_collision_grace_steps))
        self._episode_index = 0

        # reward decay: 长 episode 每步奖励递减，抑制总回报线性膨胀
        self.reward_decay_ref_steps = max(0, int(reward_decay_ref_steps))
        self.terminal_offtrack_progress_scale = float(np.clip(terminal_offtrack_progress_scale, 0.0, 1.0))
        self.bad_episode_guard_min_steps = max(0, int(bad_episode_guard_min_steps))
        self.bad_episode_guard_reward_floor = float(bad_episode_guard_reward_floor)
        self.bad_episode_guard_cte_over_in_rate = float(np.clip(bad_episode_guard_cte_over_in_rate, 0.0, 1.0))
        self.bad_episode_guard_min_forward_progress = float(max(0.0, bad_episode_guard_min_forward_progress))
        self.bad_episode_guard_penalty = float(max(0.0, bad_episode_guard_penalty))
        self.stuck_speed_threshold = float(max(0.0, stuck_speed_threshold))
        self.stuck_progress_threshold = float(max(0.0, stuck_progress_threshold))
        self.stuck_grace_steps = int(max(1, stuck_grace_steps))
        self.stuck_low_speed_penalty_start = int(max(0, stuck_low_speed_penalty_start))
        self.stuck_low_speed_penalty_scale = float(max(0.0, stuck_low_speed_penalty_scale))
        self.stuck_low_speed_penalty_cap = float(max(0.0, stuck_low_speed_penalty_cap))
        self.stuck_penalty_base = float(max(0.0, stuck_penalty_base))
        self.stuck_penalty_growth = float(max(0.0, stuck_penalty_growth))
        self.stuck_penalty_cap = float(max(self.stuck_penalty_base, stuck_penalty_cap))

        # offtrack done 阈值课程：前 leniency_ratio 比例步数内从 mult 倍线性收缩到 1.0 倍
        # 注意：CTE 惩罚(cte_term)始终基于真实边界，不受此影响
        self._leniency_steps = int(total_timesteps * float(np.clip(offtrack_leniency_ratio, 0.0, 0.5)))
        self._leniency_mult  = float(max(1.0, offtrack_leniency_mult))
        self.offtrack_grace_steps = int(max(1, offtrack_grace_steps))
        self.offtrack_severe_ratio = float(max(1.0, offtrack_severe_ratio))
        self.offtrack_grace_penalty_scale = float(max(0.0, offtrack_grace_penalty_scale))
        self.offtrack_grace_use_leniency = bool(offtrack_grace_use_leniency)

        self.w_d   = float(w_d)
        self.w_dd  = float(w_dd)
        self.w_m   = float(w_m)
        self.w_sat = float(w_sat)
        self.w_steer_budget = float(max(0.0, w_steer_budget))
        self.steer_budget_straight = float(np.clip(steer_budget_straight, 0.05, 1.0))
        self.steer_budget_curve = float(np.clip(steer_budget_curve, self.steer_budget_straight, 1.0))
        self.steer_budget_obstacle_relief = float(max(0.0, steer_budget_obstacle_relief))
        self.w_sign_flip = float(max(0.0, w_sign_flip))
        self.sign_flip_min_abs_steer = float(np.clip(sign_flip_min_abs_steer, 0.0, 1.0))
        self.w_micro_wiggle = float(max(0.0, w_micro_wiggle))
        self.micro_wiggle_min_abs_steer = float(np.clip(micro_wiggle_min_abs_steer, 0.0, 1.0))
        self.micro_wiggle_max_abs_steer = float(np.clip(micro_wiggle_max_abs_steer, 0.0, 1.0))
        if self.micro_wiggle_max_abs_steer < self.micro_wiggle_min_abs_steer:
            self.micro_wiggle_min_abs_steer, self.micro_wiggle_max_abs_steer = (
                self.micro_wiggle_max_abs_steer,
                self.micro_wiggle_min_abs_steer,
            )
        self.w_time = float(max(0.0, w_time))
        self.w_center = float(max(0.0, w_center))
        self.w_heading = float(max(0.0, w_heading))
        self.w_speed_ref = float(max(0.0, w_speed_ref))
        self.speed_ref_vmin = float(max(0.0, speed_ref_vmin))
        self.speed_ref_vmax = float(max(self.speed_ref_vmin + 1e-3, speed_ref_vmax))
        self.speed_ref_kappa_ref = float(max(1e-6, speed_ref_kappa_ref))
        self.lap_reward_scale = float(max(0.0, lap_reward_scale))
        self.progress_reward_scale = float(progress_reward_scale)
        self.progress_curve_boost = float(max(0.0, progress_curve_boost))
        self.progress_kappa_ref = float(max(1e-6, progress_kappa_ref))
        self.progress_center_gate_min = float(np.clip(progress_center_gate_min, 0.0, 1.0))
        self.progress_center_gate_power = float(max(0.1, progress_center_gate_power))
        self.smooth_curve_relief = float(np.clip(smooth_curve_relief, 0.0, 0.9))
        self.throttle_penalty_threshold = float(np.clip(throttle_penalty_threshold, 0.0, 1.0))
        self.throttle_penalty_amount = float(max(0.0, throttle_penalty_amount))
        self.survival_reward_scale = float(max(0.0, survival_reward_scale))
        self.collision_penalty_base = float(max(0.0, collision_penalty_base))
        self.offtrack_penalty_base = float(max(0.0, offtrack_penalty_base))
        self.w_near_offtrack = float(max(0.0, w_near_offtrack))
        self.near_offtrack_start_ratio = float(np.clip(near_offtrack_start_ratio, 0.0, 0.98))
        self.w_near_collision = float(max(0.0, w_near_collision))
        self.near_collision_start_ratio = float(np.clip(near_collision_start_ratio, 0.0, 0.98))
        self.overtake_success_bonus = float(max(0.0, overtake_success_bonus))
        self.safe_follow_bonus_scale = float(max(0.0, safe_follow_bonus_scale))
        self.prepare_pass_bonus_scale = float(max(0.0, prepare_pass_bonus_scale))
        self.commit_pass_bonus_scale = float(max(0.0, commit_pass_bonus_scale))
        self.safe_follow_min_m = float(max(0.1, safe_follow_min_m))
        self.safe_follow_max_m = float(max(self.safe_follow_min_m + 0.1, safe_follow_max_m))
        self.safe_follow_risk_max = float(np.clip(safe_follow_risk_max, 0.05, 1.0))
        self.safe_follow_speed_min = float(max(0.0, safe_follow_speed_min))
        self.safe_follow_ttc_min_s = float(max(0.1, safe_follow_ttc_min_s))
        self.safe_follow_ttc_max_s = float(max(self.safe_follow_ttc_min_s + 0.05, safe_follow_ttc_max_s))
        self.ttc_penalty_start_s = float(max(0.1, ttc_penalty_start_s))
        self.ttc_penalty_full_s = float(max(0.05, min(self.ttc_penalty_start_s, ttc_penalty_full_s)))
        self.lateral_overlap_ref_m = float(max(0.05, lateral_overlap_ref_m))
        self.reward_control_dt_s = float(max(1e-3, reward_control_dt_s))
        self.post_pass_stability_bonus = float(max(0.0, post_pass_stability_bonus))
        self.post_pass_stability_steps = int(max(1, post_pass_stability_steps))
        self.collision_episode_reward_cap = (
            None
            if collision_episode_reward_cap is None
            else float(collision_episode_reward_cap)
        )
        self.offtrack_episode_reward_cap = (
            None
            if offtrack_episode_reward_cap is None
            else float(offtrack_episode_reward_cap)
        )
        self.wait_window_bonus_scale = float(max(0.0, wait_window_bonus_scale))
        self.wait_window_min_gap_m = float(max(0.1, wait_window_min_gap_m))
        self.wait_window_max_gap_m = float(max(self.wait_window_min_gap_m + 0.1, wait_window_max_gap_m))
        self.wait_window_max_closing_rate = float(max(0.02, wait_window_max_closing_rate))
        self.force_pass_penalty_scale = float(max(0.0, force_pass_penalty_scale))
        self.unsafe_close_penalty_scale = float(max(0.0, unsafe_close_penalty_scale))
        self.obstacle_clearance_penalty_scale = float(max(0.0, obstacle_clearance_penalty_scale))
        self.obstacle_clearance_inner_m = float(max(0.05, obstacle_clearance_inner_m))
        self.obstacle_clearance_outer_m = float(max(self.obstacle_clearance_inner_m + 0.05, obstacle_clearance_outer_m))
        self.post_pass_cut_in_penalty_scale = float(max(0.0, post_pass_cut_in_penalty_scale))
        self.post_pass_watch_longitudinal_m = float(max(self.obstacle_clearance_outer_m, post_pass_watch_longitudinal_m))
        self.post_pass_watch_steps = int(max(1, post_pass_watch_steps))
        self.overtake_success_min_progress_ratio = float(max(0.0, overtake_success_min_progress_ratio))
        self.unsafe_close_gap_m = float(max(0.2, unsafe_close_gap_m))
        self.unsafe_close_clearance_m = float(max(0.05, unsafe_close_clearance_m))
        self.unsafe_close_longitudinal_m = float(max(self.unsafe_close_clearance_m, unsafe_close_longitudinal_m))
        self.unsafe_close_ttc_s = float(max(0.1, unsafe_close_ttc_s))
        # 风险触发后较快拉满惩罚；上一轮显示贴车碰撞仍能拿到高回报。
        self.near_penalty_ramp_steps = 6
        self._near_offtrack_ramp_step = 0
        self._near_collision_ramp_step = 0
        self.overtake_arm_longitudinal_min_m = float(max(0.05, overtake_arm_longitudinal_min_m))
        self.overtake_arm_planar_max_m = float(max(0.2, overtake_arm_planar_max_m))
        self.overtake_pass_longitudinal_threshold_m = float(min(-0.05, overtake_pass_longitudinal_threshold_m))
        self.overtake_pass_planar_min_m = float(max(0.05, overtake_pass_planar_min_m))
        self.close_front_planar_max_m = float(max(0.2, close_front_planar_max_m))
        self.force_pass_planar_max_m = float(max(0.2, force_pass_planar_max_m))
        self.overtake_min_front_steps = 4
        self.overtake_rearm_cooldown_steps = 12

        self.smooth_stats: deque = deque(maxlen=1000)

        # CTE 边界（非对称左右）
        # cte_left/right = left_in_max_sim：车辆确认仍在赛道的最大 CTE（用于奖励梯度）
        # cte_left_out/right_out = left_out_first_sim：首次确认出界的 CTE（用于 done 判定）
        self.cte_left      = float(max(cte_left,  0.1))
        self.cte_right     = float(min(cte_right, -0.1))   # 负值：右侧边界
        self.cte_left_out  = float(max(cte_left_out,  self.cte_left))  if cte_left_out  is not None else self.cte_left  * 1.1
        self.cte_right_out = float(min(cte_right_out, self.cte_right)) if cte_right_out is not None else self.cte_right * 1.1

        # CTE 奖励归一化：使各赛道 CTE 奖惩「比例」一致
        # 核心思想：cte_abs / cte_boundary 已经是 [0,1] 的比例量，
        #   但 cte_boundary 本身决定了「在边界内能获得的正奖励积分总量」。
        #   窄赛道 boundary 小 → 正奖励区间窄 → 每步正 CTE 奖励低；
        #   宽赛道 boundary 大 → 正奖励区间宽 → 每步正 CTE 奖励高。
        #   归一化目标：让窄赛道保持居中的难度与宽赛道一致。
        #   当前再做一次截断，避免窄赛道（如 ws）惩罚过轻、宽赛道惩罚过重。
        CTE_REF_HALF_WIDTH = 4.6
        self.cte_half_width = float(max(cte_half_width, 0.5))
        if cte_norm_scale is not None:
            self.cte_norm_scale = float(np.clip(cte_norm_scale, 0.1, 2.0))
        else:
            self.cte_norm_scale = float(np.clip(self.cte_half_width / CTE_REF_HALF_WIDTH, 0.75, 1.15))

        self.stuck_counter = 0
        self.offtrack_counter = 0
        self.episode_stats: Dict[str, Any] = self._zero_episode_stats()
        self.prev_lap_count = 0
        self._episode_overtake_count = 0
        self._overtake_front_steps = 0
        self._overtake_window_steps = 0
        self._overtake_window_seen = False
        self._overtake_armed = False
        self._overtake_cooldown_steps_left = 0
        self._overtake_last_longitudinal: Optional[float] = None
        self._prev_obstacle_longitudinal: Optional[float] = None
        self._prev_obstacle_lateral: Optional[float] = None
        self._prev_near_collision_risk: float = 0.0
        self._episode_post_pass_count = 0
        self._post_pass_active = False
        self._post_pass_stable_steps = 0
        self._post_pass_watch_steps_left = 0

        _side_method = "lat_err * coord_scale（精确）" if track_geometry and scene_key else "-sim_cte（近似 fallback）"
        print(
            f"DonkeyRewardWrapper: w_d={w_d}, w_dd={w_dd}, w_m={w_m}, w_sat={w_sat}, "
            f"w_steer_budget={self.w_steer_budget:.3f}, w_sign_flip={self.w_sign_flip:.3f}, "
            f"w_time={self.w_time:.3f}, w_center={self.w_center:.3f}, "
            f"w_heading={self.w_heading:.3f}, w_speed_ref={self.w_speed_ref:.3f}, "
            f"w_near_offtrack={self.w_near_offtrack:.3f}, w_near_collision={self.w_near_collision:.3f}, "
            f"survival_scale={self.survival_reward_scale:.3f}, "
            f"term(collision={self.collision_penalty_base:.2f}, offtrack={self.offtrack_penalty_base:.2f}), "
            f"lap_scale={self.lap_reward_scale:.2f}, prog_scale={self.progress_reward_scale:.2f}, "
            f"prog_curve_boost={self.progress_curve_boost:.2f}, "
            f"prog_gate(min={self.progress_center_gate_min:.2f}, p={self.progress_center_gate_power:.2f})"
        )
        if self.w_steer_budget > 0.0:
            print(
                f"   steer_budget: straight={self.steer_budget_straight:.2f}, "
                f"curve={self.steer_budget_curve:.2f}, "
                f"obstacle_relief={self.steer_budget_obstacle_relief:.2f}"
            )
        if self.w_sign_flip > 0.0:
            print(
                f"   sign_flip_penalty: min_abs_steer={self.sign_flip_min_abs_steer:.2f}"
            )
        if self.w_micro_wiggle > 0.0:
            print(
                f"   micro_wiggle_penalty: band={self.micro_wiggle_min_abs_steer:.3f}"
                f"-{self.micro_wiggle_max_abs_steer:.3f}"
            )
        print(
            f"   speed_ref: vmin={self.speed_ref_vmin:.2f}, vmax={self.speed_ref_vmax:.2f}, "
            f"kappa_ref={self.speed_ref_kappa_ref:.3f}"
        )
        print(f"   CTE in-边界: left=+{self.cte_left:.3f}, right={self.cte_right:.3f}  out-边界: left=+{self.cte_left_out:.3f}, right={self.cte_right_out:.3f}  coord_scale={self.coord_scale:.1f}")
        print(f"   CTE 归一化: half_width={self.cte_half_width:.2f}, norm_scale={self.cte_norm_scale:.3f} (ref={CTE_REF_HALF_WIDTH})")
        print(f"   左右侧判断: {_side_method}")
        print(
            f"   offtrack done 课程: 前 {self._leniency_steps:,} 步 done阈值 "
            f"{self._leniency_mult:.1f}x → 1.0x, grace={self.offtrack_grace_steps}步, "
            f"severe={self.offtrack_severe_ratio:.2f}x, "
            f"grace_penalty_scale={self.offtrack_grace_penalty_scale:.2f}, "
            f"counter_boundary={'lenient' if self.offtrack_grace_use_leniency else 'true'}  "
            "(CTE惩罚始终生效)"
        )
        if self.overtake_success_bonus > 0.0:
            print(
                f"   overtake_bonus: +{self.overtake_success_bonus:.2f} "
                f"(front>={self.overtake_arm_longitudinal_min_m:.1f}m for {self.overtake_min_front_steps} steps, "
                f"behind<={self.overtake_pass_longitudinal_threshold_m:.1f}m)"
            )
        if self.safe_follow_bonus_scale > 0.0:
            print(
                f"   safe_follow_bonus: scale={self.safe_follow_bonus_scale:.3f} "
                f"(gap={self.safe_follow_min_m:.1f}-{self.safe_follow_max_m:.1f}m, "
                f"risk<={self.safe_follow_risk_max:.2f})"
            )
        if self.wait_window_bonus_scale > 0.0 or self.force_pass_penalty_scale > 0.0:
            print(
                f"   wait_window: bonus_scale={self.wait_window_bonus_scale:.3f} "
                f"(gap={self.wait_window_min_gap_m:.1f}-{self.wait_window_max_gap_m:.1f}m, "
                f"closing<={self.wait_window_max_closing_rate:.2f}), "
                f"force_pass_penalty_scale={self.force_pass_penalty_scale:.3f}"
            )
        if self.unsafe_close_penalty_scale > 0.0:
            print(
                f"   unsafe_close_penalty: scale={self.unsafe_close_penalty_scale:.3f} "
                f"(gap<={self.unsafe_close_gap_m:.1f}m, "
                f"clearance<{self.unsafe_close_clearance_m:.2f}m, "
                f"capsule_long={self.unsafe_close_longitudinal_m:.2f}m, "
                f"ttc<{self.unsafe_close_ttc_s:.1f}s)"
            )
        if self.obstacle_clearance_penalty_scale > 0.0:
            print(
                f"   obstacle_clearance_penalty: scale={self.obstacle_clearance_penalty_scale:.3f} "
                f"(risk 0 at {self.obstacle_clearance_outer_m:.2f}m, "
                f"risk 1 at {self.obstacle_clearance_inner_m:.2f}m)"
            )
        if self.post_pass_cut_in_penalty_scale > 0.0:
            print(
                f"   post_pass_cut_in_penalty: scale={self.post_pass_cut_in_penalty_scale:.3f} "
                f"(watch={self.post_pass_watch_steps} steps, "
                f"behind<={self.post_pass_watch_longitudinal_m:.2f}m, "
                f"clearance>={self.obstacle_clearance_outer_m:.2f}m)"
            )
        if self.post_pass_stability_bonus > 0.0:
            print(
                f"   post_pass_bonus: +{self.post_pass_stability_bonus:.2f} "
                f"(stable_steps>={self.post_pass_stability_steps})"
            )
        if self.reward_decay_ref_steps > 0:
            print(f"   reward_decay: ref_steps={self.reward_decay_ref_steps} (超过后每步奖励按 ref/step 衰减)")
        if self.terminal_offtrack_progress_scale < 1.0:
            print(
                f"   terminal_offtrack_progress_scale={self.terminal_offtrack_progress_scale:.2f} "
                "(offtrack终止步progress打折)"
            )
        if self.bad_episode_guard_min_steps > 0:
            print(
                f"   bad_episode_guard: min_steps={self.bad_episode_guard_min_steps}, "
                f"reward_floor={self.bad_episode_guard_reward_floor:.1f}, "
                f"cte_over_in_rate>={self.bad_episode_guard_cte_over_in_rate:.2f}, "
                f"min_forward={self.bad_episode_guard_min_forward_progress:.2f}, "
                f"penalty={self.bad_episode_guard_penalty:.1f}"
            )
        if self.reset_env_done_grace_steps > 0 or self.reset_collision_grace_steps > 0:
            print(
                f"   reset保护: env_done前{self.reset_env_done_grace_steps}步忽略, "
                f"collision前{self.reset_collision_grace_steps}步忽略"
            )

        # 奖励分项累计（每个 episode 重置）—— 供 Monitor → PerSceneStatsCallback 使用
        self._reward_parts_episode: Dict[str, float] = self._zero_reward_parts()
        # 额外诊断统计（每个 episode 重置）—— 供日志分析
        self._episode_diag: Dict[str, Any] = self._zero_episode_diag()

    @staticmethod
    def _extract_obstacle_risk(info: Dict[str, Any]) -> float:
        """
        从 info 中提取“障碍物接近风险”[0,1]。

        优先级：
        1) 直接风险字段（已归一化）
        2) 距离字段（2.0 距离单位内按指数曲线映射）
        3) lidar（若存在）按最近有效距离估计风险
        4) 未提供则返回 -1（表示无可用障碍物信号）
        """
        # 直接风险字段（值越大越危险）
        risk_keys = (
            "obstacle_risk",
            "collision_risk",
            "hit_risk",
            "near_hit_risk",
            "risk_collision",
        )
        for k in risk_keys:
            if k in info:
                try:
                    v = float(info.get(k, 0.0))
                    if np.isfinite(v):
                        info["obstacle_risk_source"] = f"direct:{k}"
                        return float(np.clip(v, 0.0, 1.0))
                except Exception:
                    pass

        # 距离字段（值越小越危险）
        # 风险区间定义：
        # - d >= 4.0: 视为安全（风险0）
        # - d <= 0.5: 视为高危（风险1）
        # - 中间按指数曲线上升，保证远处风险较小、近处陡增。
        dist_keys = (
            "obstacle_dist",
            "obstacle_distance",
            "nearest_obstacle_dist",
            "closest_obstacle_dist",
            "front_obstacle_dist",
            "wall_dist",
            "distance_to_obstacle",
        )
        d_risk_start = 4.0
        d_risk_full = 0.5
        risk_exp = 4.0

        def _distance_to_exp_risk(d: float) -> float:
            if d >= d_risk_start:
                return 0.0
            if d <= d_risk_full:
                return 1.0
            x = (d_risk_start - d) / max(1e-6, (d_risk_start - d_risk_full))
            num = math.exp(risk_exp * x) - 1.0
            den = math.exp(risk_exp) - 1.0
            return float(np.clip(num / max(den, 1e-6), 0.0, 1.0))

        for k in dist_keys:
            if k in info:
                try:
                    d = float(info.get(k, np.inf))
                    if np.isfinite(d):
                        risk = _distance_to_exp_risk(d)
                        info["obstacle_risk_source"] = f"distance:{k}"
                        info.setdefault("obstacle_dist", float(d))
                        return risk
                except Exception:
                    pass

        # lidar 回退：使用最近有效距离的 5% 分位，降低单点噪声影响。
        lidar = info.get("lidar", None)
        if lidar is not None:
            try:
                arr = np.asarray(lidar, dtype=np.float32).reshape(-1)
                valid = arr[np.isfinite(arr) & (arr > 0.0)]
                if valid.size > 0:
                    d_lidar = float(np.percentile(valid, 5))
                    risk = _distance_to_exp_risk(d_lidar)
                    info["obstacle_risk_source"] = "lidar:p5"
                    info["obstacle_dist"] = float(d_lidar)
                    return risk
            except Exception:
                pass

        info["obstacle_risk_source"] = "none"
        return -1.0

    @staticmethod
    def _zero_reward_parts() -> Dict[str, float]:
        return {
            "survival": 0.0, "speed": 0.0, "cte": 0.0, "collision": 0.0,
            "near_offtrack": 0.0, "near_collision": 0.0,
            "progress": 0.0, "lap": 0.0, "lap_raw": 0.0, "overtake": 0.0,
            "follow": 0.0, "post_pass": 0.0, "smooth": 0.0, "jerk": 0.0,
            "mismatch": 0.0, "steer_budget": 0.0, "sign_flip": 0.0,
            "micro_wiggle": 0.0,
            "center": 0.0, "heading": 0.0, "speed_ref": 0.0, "time": 0.0,
            "sat": 0.0, "stuck": 0.0, "bad_guard": 0.0,
            "wait_window": 0.0, "force_pass": 0.0, "unsafe_close": 0.0,
            "obstacle_clearance": 0.0, "post_pass_cut_in": 0.0,
            "collision_cap": 0.0, "offtrack_cap": 0.0, "total": 0.0,
        }

    @staticmethod
    def _zero_episode_diag() -> Dict[str, Any]:
        return {
            "steps_total": 0,
            "cte_abs_samples": [],
            "progress_ratio_signed_sum": 0.0,
            "progress_ratio_forward_sum": 0.0,
            "lane_pid_debug_steps": 0,
            "lane_pid_target_speed_sum": 0.0,
            "lane_pid_speed_sum": 0.0,
            "lane_pid_speed_error_abs_sum": 0.0,
            "lane_pid_effective_lookahead_sum": 0.0,
            "lane_pid_local_forward_sum": 0.0,
            "lane_pid_local_left_abs_sum": 0.0,
            "lane_pid_lat_err_norm_abs_sum": 0.0,
            "lane_pid_steer_abs_sum": 0.0,
            "lane_pid_throttle_sum": 0.0,
            "lane_pid_reverse_steps": 0,
            "steps_cte_over_in": 0,
            "steps_cte_over_out": 0,
            "steps_rate_limit_hit": 0,
            "steps_steer_clip_hit": 0,
            "steps_throttle_high_penalty_hit": 0,
            "steps_pass_window_valid": 0,
            "steps_invalid_window_close": 0,
            "steps_unsafe_close": 0,
            "steps_obstacle_clearance_band": 0,
            "steps_obstacle_clearance_critical": 0,
            "obstacle_clearance_risk_sum": 0.0,
            "obstacle_planar_distance_min": float("inf"),
            "steps_post_pass_watch": 0,
            "steps_post_pass_cut_in": 0,
            "post_pass_cut_in_risk_sum": 0.0,
            "post_pass_planar_distance_min": float("inf"),
            "post_pass_terminal_collision": 0,
            "steps_overtake_success_ready": 0,
            "steps_overtake_passed_longitudinal": 0,
            "steps_overtake_success_clearance_ok": 0,
            "steps_overtake_success_candidate": 0,
            "steps_overtake_success_blocked_clearance": 0,
            "steps_overtake_success_blocked_progress": 0,
            "steps_overtake_success_blocked_safety": 0,
            "overtake_success_grant_count": 0,
            "offtrack_counter_max": 0,
            "stuck_counter_max": 0,
            "bad_episode_guard_triggered": 0,
            "bad_episode_guard_step": 0,
            "terminal_offtrack_progress_discount": 0.0,
            "collision_episode_reward_cap_penalty": 0.0,
            "offtrack_episode_reward_cap_penalty": 0.0,
        }

    @staticmethod
    def _signed_arc_ratio(g, idx_prev: int, idx_now: int) -> float:
        """返回基于赛道弧长的有向进度比例，前进为正，后退为负。"""
        n = int(g.center.shape[0])
        i0 = int(idx_prev) % n
        i1 = int(idx_now) % n

        if i1 >= i0:
            ds_fwd = float(g.cum_len[i1] - g.cum_len[i0])
        else:
            ds_fwd = float((g.loop_len - g.cum_len[i0]) + g.cum_len[i1])

        if i0 >= i1:
            ds_back = float(g.cum_len[i0] - g.cum_len[i1])
        else:
            ds_back = float((g.loop_len - g.cum_len[i1]) + g.cum_len[i0])

        ds_signed = ds_fwd if ds_fwd <= ds_back else -ds_back
        if not np.isfinite(ds_signed) or g.loop_len <= 1e-6:
            return 0.0

        # 屏蔽异常跳变（reset/定位抖动）
        max_reasonable = max(3.0, 0.03 * float(g.loop_len))
        if abs(ds_signed) > max_reasonable:
            return 0.0

        return float(ds_signed / float(g.loop_len))

    def _zero_episode_stats(self) -> Dict[str, Any]:
        return {
            "steps": 0,
            "max_speed": 0.0,
            "collision": False,
            "total_reward": 0.0,
            "cte_violations": 0,
            "overtake_count": 0,
            "post_pass_stability_count": 0,
        }

    @staticmethod
    def _extract_obstacle_relative_state(info: Dict[str, Any]) -> Tuple[float, float, float, float]:
        try:
            present = float(info.get("obstacle_present", 0.0) or 0.0)
        except Exception:
            present = 0.0
        try:
            longitudinal = float(info.get("obstacle_longitudinal", np.nan))
        except Exception:
            longitudinal = float("nan")
        try:
            lateral = float(info.get("obstacle_lateral", np.nan))
        except Exception:
            lateral = float("nan")
        try:
            planar_distance = float(info.get("obstacle_dist", np.nan))
        except Exception:
            planar_distance = float("nan")
        if (not np.isfinite(planar_distance)) and np.isfinite(longitudinal) and np.isfinite(lateral):
            planar_distance = float(math.hypot(longitudinal, lateral))
        return float(present), float(longitudinal), float(lateral), float(planar_distance)

    def _estimate_obstacle_ttc_and_overlap(
        self,
        obstacle_available: bool,
        obstacle_longitudinal: float,
        obstacle_lateral: float,
    ) -> Tuple[float, float, float]:
        if not obstacle_available or not np.isfinite(obstacle_longitudinal):
            return float("inf"), 0.0, 0.0

        closing_rate = 0.0
        if self._prev_obstacle_longitudinal is not None and np.isfinite(self._prev_obstacle_longitudinal):
            closing_rate = (self._prev_obstacle_longitudinal - obstacle_longitudinal) / max(self.reward_control_dt_s, 1e-3)

        if obstacle_longitudinal <= 0.0 or closing_rate <= 1e-4:
            ttc_s = float("inf")
        else:
            ttc_s = float(obstacle_longitudinal / max(closing_rate, 1e-4))

        lateral_overlap = float(np.clip(
            1.0 - abs(obstacle_lateral) / max(self.lateral_overlap_ref_m, 1e-3),
            0.0,
            1.0,
        ))
        return ttc_s, lateral_overlap, float(max(0.0, closing_rate))

    @staticmethod
    def _capsule_zone_risk(
        obstacle_longitudinal: float,
        obstacle_lateral: float,
        lateral_radius_m: float,
        longitudinal_radius_m: float,
    ) -> float:
        if not (np.isfinite(obstacle_longitudinal) and np.isfinite(obstacle_lateral)):
            return 0.0
        lateral_radius = float(max(lateral_radius_m, 1e-6))
        longitudinal_radius = float(max(longitudinal_radius_m, lateral_radius))
        segment_half = float(max(0.0, longitudinal_radius - lateral_radius))
        dx = max(0.0, abs(float(obstacle_longitudinal)) - segment_half)
        dy = abs(float(obstacle_lateral))
        dist_to_capsule_axis = math.hypot(dx, dy)
        return float(np.clip(
            (lateral_radius - dist_to_capsule_axis) / max(lateral_radius, 1e-6),
            0.0,
            1.0,
        ))

    @staticmethod
    def _linear_distance_band_risk(distance_m: float, inner_m: float, outer_m: float) -> float:
        if not np.isfinite(distance_m):
            return 0.0
        inner = float(max(0.0, inner_m))
        outer = float(max(inner + 1e-6, outer_m))
        return float(np.clip((outer - float(distance_m)) / max(outer - inner, 1e-6), 0.0, 1.0))

    def _unwrap_base_env(self):
        base = self.env
        depth = 0
        while hasattr(base, "env") and depth < 32:
            base = base.env
            depth += 1
        return base

    def _clear_base_handler_over(self) -> None:
        try:
            handler = self._get_base_handler()
            if handler is not None:
                handler.over = False
        except Exception:
            pass

    def _get_base_handler(self):
        try:
            base = self._unwrap_base_env()
            viewer = getattr(base, "viewer", None)
            handler = getattr(viewer, "handler", None)
            return handler
        except Exception:
            return None

    def _get_obstacle_runtime(self):
        try:
            wrapped = self.env
            depth = 0
            while wrapped is not None and depth < 32:
                runtime = getattr(wrapped, "runtime", None)
                if runtime is not None:
                    return runtime
                wrapped = getattr(wrapped, "env", None)
                depth += 1
        except Exception:
            return None
        return None

    def _request_overtake_respawn(self, info: Dict[str, Any]) -> bool:
        runtime = self._get_obstacle_runtime()
        if runtime is None:
            return False
        try:
            return bool(runtime.request_overtake_respawn(agent_info=info))
        except Exception:
            return False

    @staticmethod
    def _drop_env_done_from_reasons(term_reasons):
        cleaned = []
        for reason in term_reasons:
            tokens = [tok for tok in str(reason).split("+") if tok and tok != "env_done"]
            if tokens:
                cleaned.append("+".join(tokens))
        return cleaned

    def reset(self, **kwargs):
        self.episode_stats = self._zero_episode_stats()
        self.prev_lap_count = 0
        self._soft_lap_progress = 0.0   # 累计前进弧长比例，≥1.0 计一圈
        self._soft_lap_count = 0        # 软件检测的圈数
        self.stuck_counter  = 0
        self.offtrack_counter = 0
        self._prev_track_idx = None
        self._near_offtrack_ramp_step = 0
        self._near_collision_ramp_step = 0
        self._episode_overtake_count = 0
        self._overtake_front_steps = 0
        self._overtake_window_steps = 0
        self._overtake_window_seen = False
        self._overtake_armed = False
        self._overtake_cooldown_steps_left = 0
        self._overtake_last_longitudinal = None
        self._prev_obstacle_longitudinal = None
        self._prev_obstacle_lateral = None
        self._prev_near_collision_risk = 0.0
        self._episode_post_pass_count = 0
        self._post_pass_active = False
        self._post_pass_stable_steps = 0
        self._post_pass_watch_steps_left = 0
        self._reward_parts_episode = self._zero_reward_parts()
        self._episode_diag = self._zero_episode_diag()
        self._episode_index += 1
        
        obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.current_step += 1
        
        # 记录环境级的 done 状态（用于诊断）
        env_done_before_processing = done
        term_reasons = []
        prev_reason = str(info.get("termination_reason", "") or "").strip()
        if prev_reason and prev_reason != "normal":
            term_reasons.append(prev_reason)
        episode_step = int(self.episode_stats["steps"]) + 1
        reset_env_done_grace_active = (
            self.reset_env_done_grace_steps > 0
            and episode_step <= self.reset_env_done_grace_steps
        )
        reset_collision_grace_active = (
            self.reset_collision_grace_steps > 0
            and episode_step <= self.reset_collision_grace_steps
        )
        env_done_masked = False
        collision_masked = False
        native_cte_promoted_to_offtrack = False
        if env_done_before_processing and reset_env_done_grace_active:
            done = False
            env_done_masked = True
            self._clear_base_handler_over()

        cte_signed = float(info.get("cte", 0))
        cte_abs    = abs(cte_signed)
        speed      = float(info.get("speed", 0) or 0)
        hit        = info.get("hit", "none")
        handler = self._get_base_handler()
        if handler is not None:
            try:
                native_env_cte = float(getattr(handler, "cte", cte_signed) or 0.0)
            except Exception:
                native_env_cte = float(info.get("native_env_cte", cte_signed) or 0.0)
            try:
                native_env_max_cte = float(getattr(handler, "max_cte", 0.0) or 0.0)
            except Exception:
                native_env_max_cte = float(info.get("native_env_max_cte", 0.0) or 0.0)
            try:
                native_env_over = float(bool(getattr(handler, "over", False)))
            except Exception:
                native_env_over = float(info.get("native_env_over", 0.0) or 0.0)
            try:
                native_env_hit = str(getattr(handler, "hit", hit) or "none")
            except Exception:
                native_env_hit = str(info.get("native_env_hit", hit) or "none")
        else:
            native_env_cte = float(info.get("native_env_cte", cte_signed) or 0.0)
            native_env_max_cte = float(info.get("native_env_max_cte", 0.0) or 0.0)
            native_env_over = float(info.get("native_env_over", 0.0) or 0.0)
            native_env_hit = str(info.get("native_env_hit", hit) or "none")
        native_env_cte_abs = abs(native_env_cte)
        native_env_done_likely_hit = float(
            env_done_before_processing and native_env_hit not in ("", "none")
        )
        native_env_done_likely_cte = float(
            env_done_before_processing
            and native_env_max_cte > 0.0
            and native_env_cte_abs > native_env_max_cte
            and native_env_done_likely_hit < 0.5
        )
        native_env_done_reason = ""
        if env_done_before_processing:
            if native_env_done_likely_hit >= 0.5:
                native_env_done_reason = "native_hit"
            elif native_env_done_likely_cte >= 0.5:
                native_env_done_reason = "native_cte_exceed"
            else:
                native_env_done_reason = "native_unknown"
        if env_done_before_processing and native_env_done_likely_cte >= 0.5 and native_env_done_likely_hit < 0.5:
            done = False
            env_done_masked = True
            native_cte_promoted_to_offtrack = True
            term_reasons = self._drop_env_done_from_reasons(term_reasons)
            self._clear_base_handler_over()
        lap_count  = int(info.get("lap_count", 0) or 0)
        if "ctrl/throttle_cmd_exec" in info:
            throttle_cmd = float(info.get("ctrl/throttle_cmd_exec", 0.0) or 0.0)
        else:
            throttle_cmd = float(action[1]) if len(action) > 1 else 0.0
        prev_track_idx = self._prev_track_idx
        curr_track_idx = None
        kappa_abs = 0.0

        self.episode_stats["max_speed"] = max(self.episode_stats["max_speed"], speed)

        # 用轨迹几何计算有符号横偏距 lat_err_cte（与 _SCENE_CTE_TABLE 同单位）
        # lat_err_cte > 0 = 赛道左侧，lat_err_cte < 0 = 赛道右侧
        # 无几何时用 -cte_signed 近似（lat_err * coord_scale ≈ -sim_cte）
        lat_err_cte = -cte_signed   # fallback
        lat_err_norm = 0.0
        heading_err_abs = 0.0
        if self._track_geometry is not None and self._scene_key:
            try:
                pos = info.get("pos", (0.0, 0.0, 0.0))
                x, z = float(pos[0]), float(pos[2])
                car = info.get("car", (0.0, 0.0, 0.0))
                yaw_deg = float(car[2]) if len(car) >= 3 else 0.0
                yaw_rad = math.radians(yaw_deg)
                geo = self._track_geometry.query(
                    self._scene_key, x=x, z=z, yaw_rad=yaw_rad,
                    prev_idx=self._prev_track_idx,
                )
                curr_track_idx = int(geo["idx"])
                self._prev_track_idx = curr_track_idx
                lat_err = geo["lat_err"]
                lat_err_cte = lat_err * self.coord_scale  # 有符号横偏距，与 CTE 表同单位
                lat_err_norm = float(geo.get("lat_err_norm", 0.0))
                heading_err_abs = abs(float(math.atan2(
                    float(geo.get("heading_err_sin", 0.0)),
                    float(geo.get("heading_err_cos", 1.0)),
                )))
                kappa_abs = abs(float(geo.get("kappa_lookahead", 0.0)))
            except Exception:
                pass

        # 有符号横偏距的幅值（用于边界比较）
        lat_err_cte_abs = abs(lat_err_cte)
        is_left_side = (lat_err_cte >= 0)  # 用于日志和边界选择

        # 非对称 CTE 边界（有符号选边，幅值比较）
        cte_boundary     = abs(self.cte_left     if is_left_side else self.cte_right)
        cte_out_boundary = abs(self.cte_left_out if is_left_side else self.cte_right_out)
        # ontrack：边界点仍视为在界内；超过边界才记 over_in / over_out
        ontrack = float(lat_err_cte_abs <= cte_boundary)
        cte_over_in = float(lat_err_cte_abs > cte_boundary)
        cte_over_out = float(lat_err_cte_abs > cte_out_boundary)

        self._episode_diag["steps_total"] += 1
        self._episode_diag["cte_abs_samples"].append(float(lat_err_cte_abs))
        self._episode_diag["steps_cte_over_in"] += int(cte_over_in > 0.5)
        self._episode_diag["steps_cte_over_out"] += int(cte_over_out > 0.5)
        lane_pid_debug_active = float(info.get("obstacle_lane_pid_debug_active", 0.0) or 0.0)
        if lane_pid_debug_active > 0.5:
            self._episode_diag["lane_pid_debug_steps"] += 1
            self._episode_diag["lane_pid_target_speed_sum"] += float(
                info.get("obstacle_lane_pid_target_speed", 0.0) or 0.0
            )
            self._episode_diag["lane_pid_speed_sum"] += float(
                info.get("obstacle_lane_pid_speed", 0.0) or 0.0
            )
            self._episode_diag["lane_pid_speed_error_abs_sum"] += abs(float(
                info.get("obstacle_lane_pid_speed_error", 0.0) or 0.0
            ))
            self._episode_diag["lane_pid_effective_lookahead_sum"] += float(
                info.get("obstacle_lane_pid_effective_lookahead", 0.0) or 0.0
            )
            self._episode_diag["lane_pid_local_forward_sum"] += float(
                info.get("obstacle_lane_pid_local_forward", 0.0) or 0.0
            )
            self._episode_diag["lane_pid_local_left_abs_sum"] += abs(float(
                info.get("obstacle_lane_pid_local_left", 0.0) or 0.0
            ))
            self._episode_diag["lane_pid_lat_err_norm_abs_sum"] += abs(float(
                info.get("obstacle_lane_pid_lat_err_norm", 0.0) or 0.0
            ))
            self._episode_diag["lane_pid_steer_abs_sum"] += abs(float(
                info.get("obstacle_lane_pid_steer", 0.0) or 0.0
            ))
            self._episode_diag["lane_pid_throttle_sum"] += float(
                info.get("obstacle_lane_pid_throttle", 0.0) or 0.0
            )
            self._episode_diag["lane_pid_reverse_steps"] += int(
                float(info.get("obstacle_lane_pid_reverse_mode", 0.0) or 0.0) > 0.5
            )

        speed_gate = float(np.clip(speed / 0.5, 0.0, 1.0))
        v_normalized = float(np.clip(speed, 0.0, 4.0) / 4.0)
        cte_norm = lat_err_cte_abs / max(1e-6, cte_boundary)
        center_factor = float(np.clip(1.0 - cte_norm * cte_norm, 0.0, 1.0))

        # Progress reward（按赛道几何弧长计算有向进度，前进正、后退负）
        progress_reward = 0.0
        progress_reward_raw = 0.0
        progress_ratio = 0.0
        progress_ratio_unclipped = 0.0  # 未裁切的弧长比例，用于软件圈数检测
        progress_center_gate = 1.0
        progress_forward_gain = 1.0
        if (
            self._track_geometry is not None
            and self._scene_key
            and prev_track_idx is not None
            and curr_track_idx is not None
        ):
            try:
                g = self._track_geometry.scenes[self._scene_key]
                progress_ratio_unclipped = self._signed_arc_ratio(g, int(prev_track_idx), int(curr_track_idx))
                progress_ratio = float(np.clip(progress_ratio_unclipped, -0.02, 0.02))
                curve_ratio = float(np.clip(kappa_abs / self.progress_kappa_ref, 0.0, 1.0))
                if progress_ratio > 0.0:
                    # 修复：正向进度奖励与横向控制耦合，贴边时不再用高进度“抵消”CTE/碰撞风险
                    progress_center_gate = float(max(
                        self.progress_center_gate_min,
                        center_factor ** self.progress_center_gate_power,
                    ))
                    # 弯道增益也受中心线因子调制，避免“贴边+高曲率”被额外鼓励
                    progress_forward_gain = float(1.0 + self.progress_curve_boost * curve_ratio * center_factor)
                    progress_reward_raw = ontrack * self.progress_reward_scale * progress_ratio * progress_forward_gain
                    progress_reward = progress_reward_raw * progress_center_gate
                else:
                    # 负向进度保持全额惩罚，不做门控
                    progress_reward = ontrack * self.progress_reward_scale * progress_ratio
                    progress_reward_raw = progress_reward
            except Exception:
                progress_ratio = 0.0
                progress_reward_raw = 0.0
                progress_reward = 0.0
        # 记录几何进度累计（与 progress_reward_scale 解耦，供动态采样软成功使用）
        self._episode_diag["progress_ratio_signed_sum"] += float(progress_ratio)
        self._episode_diag["progress_ratio_forward_sum"] += float(max(0.0, progress_ratio))

        # ── 软件圈数检测（解决 WS 等赛道无 starting-line trigger 的问题）──
        # 使用 unclipped 弧长比例累加，当净前进距离 ≥ 1.0 圈时计一圈
        self._soft_lap_progress += float(progress_ratio_unclipped)
        # 防止长时间倒退造成巨大赤字
        self._soft_lap_progress = max(-0.5, self._soft_lap_progress)
        while self._soft_lap_progress >= 1.0:
            self._soft_lap_count += 1
            self._soft_lap_progress -= 1.0

        # Survival / speed：都要求“真的在前进”，避免学成慢速保命。
        alive_forward_gate = float(progress_ratio > 1e-6)
        survival_reward = self.survival_reward_scale * speed_gate * alive_forward_gate
        speed_reward = 0.25 * ontrack * speed_gate * center_factor * alive_forward_gate

        # 稠密项：最短路径 / 姿态 / 曲率目标速度 / 时间惩罚
        center_penalty = -self.w_center * abs(float(lat_err_norm))
        heading_penalty = -self.w_heading * (float(heading_err_abs) / math.pi)
        curve_ratio_speed = float(np.clip(kappa_abs / self.speed_ref_kappa_ref, 0.0, 1.0))
        v_ref = float(
            self.speed_ref_vmax
            - (self.speed_ref_vmax - self.speed_ref_vmin) * curve_ratio_speed
        )
        speed_err_norm = float((speed - v_ref) / max(self.speed_ref_vmax, 1e-6))
        speed_ref_penalty = -self.w_speed_ref * (speed_err_norm * speed_err_norm)
        time_penalty = -self.w_time

        # ★ CTE reward（BUG FIXED：使用 cte_abs + cte_boundary）
        # V3 归一化: 乘以 cte_norm_scale 使各赛道 CTE 奖惩量级一致
        #   norm_scale = cte_half_width / REF → 窄赛道 <1（缩小惩罚），宽赛道 >1（放大惩罚）
        if lat_err_cte_abs > cte_boundary:
            # 出界量（有符号）= lat_err_cte - 边界（正=左出界，负=右出界）
            # 分母用 cte_half_width（平均宽度）而非侧向 boundary，
            # 防止非对称赛道（wt/gt/wh 右侧窄）惩罚因分母小而爆炸
            exceed_ratio = (lat_err_cte_abs - cte_boundary) / max(1e-6, self.cte_half_width)
            # clip exceed_ratio 防止出界后惩罚无限累积
            exceed_ratio = min(exceed_ratio, 2.0)
            cte_term = -(1.0 + 4.0 * exceed_ratio) * self.cte_norm_scale
            self.episode_stats["cte_violations"] += 1
        else:
            cte_base = 0.3 * (1.0 - lat_err_cte_abs / max(1e-6, cte_boundary))
            speed_gate_cte = float(np.clip(speed / 0.3, 0.0, 1.0))
            cte_term = cte_base * speed_gate_cte * self.cte_norm_scale

        # Terminal penalty（仅记录真正的终止惩罚；near_* 单独累计）
        terminal_penalty = 0.0
        if hit != "none":
            if reset_collision_grace_active:
                collision_masked = True
                if env_done_before_processing:
                    done = False
                    env_done_masked = True
                self._clear_base_handler_over()
            else:
                terminal_penalty = -self.collision_penalty_base
                self.episode_stats["collision"] = True
                done = True
                term_reasons.append("collision")

        # Lap reward（合并 sim 计圈和软件计圈，取较大值）
        effective_lap_count = max(lap_count, self._soft_lap_count)
        lap_reward = 0.0
        lap_reward_raw = 0.0
        # WS无障碍episode用更短的圈数上限：
        # 有障碍时ep_len≈200步，无障碍时ep_len≈1300步，步数严重倾斜（有障碍仅占18%步数）
        # 无障碍WS限制3圈done，让两类episode步数接近，提升有障碍的学习信号比例
        obstacle_active = float(info.get("obstacle_runtime_active", 1.0))
        if self._scene_key == "waveshare" and obstacle_active < 0.5:
            MAX_LAPS_FOR_REWARD = 3  # WS无障碍：3圈done
        else:
            MAX_LAPS_FOR_REWARD = 5  # WS有障碍 / GT：5圈done
        if effective_lap_count > self.prev_lap_count and effective_lap_count <= MAX_LAPS_FOR_REWARD:
            # 每步最多按 1 圈计奖，避免计数抖动造成奖励尖峰
            laps_completed_raw = effective_lap_count - self.prev_lap_count
            laps_completed = int(max(0, min(laps_completed_raw, 1)))
            lap_reward_raw = 6.0 * laps_completed
            lap_reward = lap_reward_raw * self.lap_reward_scale
            self.prev_lap_count = effective_lap_count
            lap_source = "sim" if lap_count >= self._soft_lap_count else "soft"
            print(
                f"\n🎉 [{self._logging_key}] 完成第 {effective_lap_count} 圈 ({lap_source})! "
                f"奖励 +{lap_reward:.1f} (raw={lap_reward_raw:.1f}, "
                f"scale={self.lap_reward_scale:.2f}, "
                f"sim_lap={lap_count}, soft_lap={self._soft_lap_count})"
            )
        elif effective_lap_count > MAX_LAPS_FOR_REWARD:
            # 超过上限，不再给奖励，但更新prev_lap_count避免重复计数
            self.prev_lap_count = effective_lap_count

        # ★ 圈数上限到达后立即终止 episode
        if effective_lap_count >= MAX_LAPS_FOR_REWARD:
            done = True
            term_reasons.append("max_laps_reached")

        # Stuck 检测：必须同时满足低速和几乎没有几何进度。
        # 只看 speed 会误杀 real-aligned 的慢速过弯；当前 speed proxy 的 real p50 约 0.06。
        stuck_penalty = 0.0
        stuck_low_speed = bool(speed < self.stuck_speed_threshold)
        stuck_low_progress = bool(abs(float(progress_ratio)) <= self.stuck_progress_threshold)
        if ontrack and stuck_low_speed and stuck_low_progress:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        if (
            self.stuck_low_speed_penalty_scale > 0.0
            and self.stuck_counter > self.stuck_low_speed_penalty_start
        ):
            low_speed_steps = self.stuck_counter - self.stuck_low_speed_penalty_start
            stuck_penalty -= min(
                self.stuck_low_speed_penalty_cap,
                self.stuck_low_speed_penalty_scale * low_speed_steps,
            )
        if self.stuck_counter > self.stuck_grace_steps:
            done = True
            terminal_stuck_penalty = min(
                self.stuck_penalty_cap,
                self.stuck_penalty_base
                + self.stuck_penalty_growth * (self.stuck_counter - self.stuck_grace_steps),
            )
            stuck_penalty -= terminal_stuck_penalty
            term_reasons.append("stuck")

        # 出界课程：done 触发阈值前期放宽，后期收紧；CTE 惩罚(cte_term)始终基于真实边界
        if self._leniency_steps > 0 and self.current_step < self._leniency_steps:
            _progress = self.current_step / self._leniency_steps          # 0 → 1
            _leniency = self._leniency_mult * (1.0 - _progress) + _progress  # mult → 1.0
        else:
            _leniency = 1.0
        _effective_out = cte_out_boundary * _leniency
        _offtrack_counter_out = _effective_out if self.offtrack_grace_use_leniency else cte_out_boundary
        _offtrack_severe_out = cte_out_boundary * self.offtrack_severe_ratio

        if native_cte_promoted_to_offtrack:
            self.offtrack_counter = max(self.offtrack_counter, min(3, self.offtrack_grace_steps))
        elif lat_err_cte_abs > _offtrack_counter_out:
            self.offtrack_counter += 1
        else:
            self.offtrack_counter = 0

        offtrack_severe_exceed = bool(lat_err_cte_abs > _offtrack_severe_out)

        # Offtrack done：轻微出界先给恢复窗口；严重出界仍硬终止，避免跑出赛道太远污染 value。
        if self.offtrack_counter >= self.offtrack_grace_steps or offtrack_severe_exceed:
            terminal_penalty -= self.offtrack_penalty_base
            done = True
            term_reasons.append("offtrack")

        # 线性“预惩罚”：在真正出界/碰撞前就开始惩罚，降低最后一刻急打方向的策略偏好。
        # 注意：风险评估使用“真实 out 边界”，不跟随 done 阈值 leniency 放宽。
        cte_out_ratio_done = float(np.clip(lat_err_cte_abs / max(_offtrack_counter_out, 1e-6), 0.0, 2.0))
        cte_out_ratio_risk = float(np.clip(lat_err_cte_abs / max(cte_out_boundary, 1e-6), 0.0, 2.0))

        near_offtrack_ratio = float(np.clip(
            (cte_out_ratio_risk - self.near_offtrack_start_ratio)
            / max(1e-6, 1.0 - self.near_offtrack_start_ratio),
            0.0,
            1.0,
        ))
        if near_offtrack_ratio > 1e-6:
            self._near_offtrack_ramp_step = min(
                self.near_penalty_ramp_steps,
                self._near_offtrack_ramp_step + 1,
            )
        else:
            self._near_offtrack_ramp_step = 0
        near_offtrack_ramp_scale = float(self._near_offtrack_ramp_step / max(1, self.near_penalty_ramp_steps))
        near_offtrack_penalty = -self.w_near_offtrack * near_offtrack_ratio * near_offtrack_ramp_scale
        offtrack_grace_penalty = 0.0
        if self.offtrack_grace_penalty_scale > 0.0 and self.offtrack_counter > 0:
            grace_ratio = float(np.clip(
                self.offtrack_counter / max(1, self.offtrack_grace_steps),
                0.0,
                1.0,
            ))
            offtrack_grace_penalty = -self.offtrack_grace_penalty_scale * grace_ratio
        near_offtrack_total_penalty = near_offtrack_penalty + offtrack_grace_penalty

        heading_risk = float(np.clip(heading_err_abs / (0.70 * math.pi), 0.0, 1.0))
        speed_risk = float(np.clip(speed / max(self.speed_ref_vmax, 1e-6), 0.0, 1.0))
        near_collision_ratio = float(np.clip(
            (cte_out_ratio_risk - self.near_collision_start_ratio)
            / max(1e-6, 1.0 - self.near_collision_start_ratio),
            0.0,
            1.0,
        ))
        control_risk = 0.0
        if self.action_safety_wrapper is not None:
            try:
                diag = self.action_safety_wrapper.diag
                control_risk = float(max(
                    float(diag.get("rate_excess_bounded", 0.0)),
                    float(diag.get("steer_clip_hit", 0.0)),
                ))
            except Exception:
                control_risk = 0.0
        obstacle_present, obstacle_longitudinal, obstacle_lateral, obstacle_planar_distance = (
            self._extract_obstacle_relative_state(info)
        )
        obstacle_available = bool(obstacle_present > 0.5 and np.isfinite(obstacle_longitudinal))
        obstacle_planar_ok = bool(np.isfinite(obstacle_planar_distance) and obstacle_planar_distance > 0.0)
        obstacle_ttc_s, obstacle_lateral_overlap, obstacle_closing_rate = (
            self._estimate_obstacle_ttc_and_overlap(
                obstacle_available=obstacle_available,
                obstacle_longitudinal=float(obstacle_longitudinal),
                obstacle_lateral=float(obstacle_lateral),
            )
        )
        if np.isfinite(obstacle_ttc_s):
            if obstacle_ttc_s <= self.ttc_penalty_full_s:
                ttc_risk = 1.0
            elif obstacle_ttc_s >= self.ttc_penalty_start_s:
                ttc_risk = 0.0
            else:
                ttc_risk = float(
                    (self.ttc_penalty_start_s - obstacle_ttc_s)
                    / max(1e-6, self.ttc_penalty_start_s - self.ttc_penalty_full_s)
                )
        else:
            ttc_risk = 0.0
        obstacle_risk = self._extract_obstacle_risk(info)
        has_obstacle_signal = float(obstacle_risk >= 0.0)
        if obstacle_risk < 0.0:
            obstacle_risk = 0.0
        capsule_lateral_m = max(
            self.lateral_overlap_ref_m,
            self.unsafe_close_clearance_m if self.unsafe_close_penalty_scale > 0.0 else 0.0,
            0.05,
        )
        capsule_longitudinal_m = max(self.unsafe_close_longitudinal_m, capsule_lateral_m)
        capsule_zone_risk = (
            self._capsule_zone_risk(
                obstacle_longitudinal=float(obstacle_longitudinal),
                obstacle_lateral=float(obstacle_lateral),
                lateral_radius_m=capsule_lateral_m,
                longitudinal_radius_m=capsule_longitudinal_m,
            )
            if obstacle_available
            else 0.0
        )
        obstacle_distance_risk = (
            self._linear_distance_band_risk(
                float(obstacle_planar_distance),
                self.obstacle_clearance_inner_m,
                self.obstacle_clearance_outer_m,
            )
            if obstacle_available and obstacle_planar_ok
            else 0.0
        )
        if obstacle_available and obstacle_planar_ok:
            self._episode_diag["obstacle_planar_distance_min"] = min(
                float(self._episode_diag["obstacle_planar_distance_min"]),
                float(obstacle_planar_distance),
            )
        self._episode_diag["steps_obstacle_clearance_band"] += int(obstacle_distance_risk > 1e-6)
        self._episode_diag["steps_obstacle_clearance_critical"] += int(
            obstacle_available
            and obstacle_planar_ok
            and float(obstacle_planar_distance) <= self.obstacle_clearance_inner_m
        )
        self._episode_diag["obstacle_clearance_risk_sum"] += float(obstacle_distance_risk)
        side_clearance_risk = capsule_zone_risk
        proxy_collision_risk = float(np.clip(
            0.30 * near_collision_ratio + 0.30 * heading_risk + 0.20 * speed_risk + 0.20 * control_risk,
            0.0,
            1.0,
        ))
        # 优先使用障碍物信号；无信号时回退到代理风险。
        if has_obstacle_signal > 0.5:
            near_collision_risk_raw = float(np.clip(
                0.50 * obstacle_risk
                + 0.25 * proxy_collision_risk
                + 0.20 * ttc_risk
                + 0.05 * obstacle_lateral_overlap,
                0.0,
                1.0,
            ))
        else:
            near_collision_risk_raw = float(np.clip(
                0.45 * proxy_collision_risk
                + 0.45 * ttc_risk
                + 0.10 * obstacle_lateral_overlap,
                0.0,
                1.0,
            ))
        near_collision_risk_raw = float(max(
            near_collision_risk_raw,
            side_clearance_risk,
            obstacle_distance_risk,
        ))

        near_collision_trigger = float(max(
            0.50 * near_collision_ratio,
            heading_risk,
            control_risk,
            obstacle_risk,
            ttc_risk,
            obstacle_lateral_overlap,
            side_clearance_risk,
            obstacle_distance_risk,
        ))
        if near_collision_trigger > 1e-6:
            self._near_collision_ramp_step = min(
                self.near_penalty_ramp_steps,
                self._near_collision_ramp_step + 1,
            )
        else:
            self._near_collision_ramp_step = 0

        near_collision_ramp_scale = float(self._near_collision_ramp_step / max(1, self.near_penalty_ramp_steps))
        near_collision_risk = near_collision_risk_raw * near_collision_ramp_scale
        near_collision_penalty = -self.w_near_collision * near_collision_risk

        info["reward_debug/offtrack_leniency"]  = _leniency
        info["reward_debug/effective_cte_out"]  = _effective_out
        info["reward_debug/offtrack_counter_cte_out"] = float(_offtrack_counter_out)
        info["reward_debug/offtrack_severe_cte_out"] = float(_offtrack_severe_out)
        info["reward_debug/cte_out_ratio"] = float(cte_out_ratio_risk)
        info["reward_debug/cte_out_ratio_done"] = float(cte_out_ratio_done)
        info["reward_debug/near_offtrack_ratio"] = float(near_offtrack_ratio)
        info["reward_debug/near_offtrack_ramp_scale"] = float(near_offtrack_ramp_scale)
        info["reward_debug/near_collision_ramp_scale"] = float(near_collision_ramp_scale)
        info["reward_debug/near_offtrack_ramp_step"] = float(self._near_offtrack_ramp_step)
        info["reward_debug/near_collision_ramp_step"] = float(self._near_collision_ramp_step)
        info["reward_debug/r_near_offtrack"] = float(near_offtrack_penalty)
        info["reward_debug/r_offtrack_grace"] = float(offtrack_grace_penalty)
        info["reward_debug/offtrack_grace_steps"] = float(self.offtrack_grace_steps)
        info["reward_debug/offtrack_severe_exceed"] = float(offtrack_severe_exceed)
        info["reward_debug/near_collision_proxy_risk"] = float(proxy_collision_risk)
        info["reward_debug/near_collision_obstacle_risk"] = float(obstacle_risk)
        info["reward_debug/near_collision_has_obstacle_signal"] = float(has_obstacle_signal)
        info["reward_debug/obstacle_dist"] = float(info.get("obstacle_dist", np.nan))
        info["reward_debug/obstacle_risk_source"] = str(info.get("obstacle_risk_source", "none"))
        info["reward_debug/obstacle_ttc_s"] = float(obstacle_ttc_s if np.isfinite(obstacle_ttc_s) else 999.0)
        info["reward_debug/obstacle_lateral_overlap"] = float(obstacle_lateral_overlap)
        info["reward_debug/obstacle_side_clearance_risk"] = float(side_clearance_risk)
        info["reward_debug/obstacle_capsule_risk"] = float(capsule_zone_risk)
        info["reward_debug/obstacle_capsule_lateral_m"] = float(capsule_lateral_m)
        info["reward_debug/obstacle_capsule_longitudinal_m"] = float(capsule_longitudinal_m)
        info["reward_debug/obstacle_distance_band_risk"] = float(obstacle_distance_risk)
        info["reward_debug/obstacle_clearance_inner_m"] = float(self.obstacle_clearance_inner_m)
        info["reward_debug/obstacle_clearance_outer_m"] = float(self.obstacle_clearance_outer_m)
        info["reward_debug/obstacle_closing_rate"] = float(obstacle_closing_rate)
        info["reward_debug/near_collision_ttc_risk"] = float(ttc_risk)
        info["reward_debug/near_collision_risk"] = float(near_collision_risk)
        info["reward_debug/r_near_collision"] = float(near_collision_penalty)

        safe_follow_bonus = 0.0
        wait_window_bonus = 0.0
        force_pass_penalty = 0.0
        unsafe_close_penalty = 0.0
        obstacle_clearance_penalty = 0.0
        post_pass_cut_in_penalty = 0.0
        wait_window_gate = False
        force_pass_risk = 0.0
        unsafe_close_risk = 0.0
        unsafe_close_active = False
        post_pass_cut_in_risk = 0.0
        post_pass_cut_in_active = False
        post_pass_watch_active = False
        post_pass_clearance_ok = True
        prepare_pass_bonus = 0.0
        commit_pass_bonus = 0.0
        overtake_bonus = 0.0
        post_pass_bonus = 0.0
        overtake_success = False
        if self.safe_follow_bonus_scale > 0.0 and obstacle_available:
            longitudinal_gap = float(obstacle_longitudinal)
            in_follow_band = bool(self.safe_follow_min_m <= longitudinal_gap <= self.safe_follow_max_m)
            in_ttc_band = bool(
                np.isfinite(obstacle_ttc_s)
                and self.safe_follow_ttc_min_s <= obstacle_ttc_s <= self.safe_follow_ttc_max_s
            )
            follow_safe_state = bool(
                (not done)
                and (in_follow_band or in_ttc_band)
                and obstacle_risk <= self.safe_follow_risk_max
                and near_collision_risk <= self.safe_follow_risk_max
                and cte_over_out <= 0.5
                and speed >= self.safe_follow_speed_min
                and ("collision" not in term_reasons)
                and ("offtrack" not in term_reasons)
                and ("stuck" not in term_reasons)
            )
            if follow_safe_state:
                if in_ttc_band:
                    follow_mid = 0.5 * (self.safe_follow_ttc_min_s + self.safe_follow_ttc_max_s)
                    follow_half = max(1e-6, 0.5 * (self.safe_follow_ttc_max_s - self.safe_follow_ttc_min_s))
                    gap_score = float(np.clip(1.0 - abs(obstacle_ttc_s - follow_mid) / follow_half, 0.0, 1.0))
                else:
                    follow_mid = 0.5 * (self.safe_follow_min_m + self.safe_follow_max_m)
                    follow_half = max(1e-6, 0.5 * (self.safe_follow_max_m - self.safe_follow_min_m))
                    gap_score = float(np.clip(1.0 - abs(longitudinal_gap - follow_mid) / follow_half, 0.0, 1.0))
                risk_score = float(np.clip(
                    1.0 - max(obstacle_risk, near_collision_risk) / max(self.safe_follow_risk_max, 1e-6),
                    0.0,
                    1.0,
                ))
                safe_follow_bonus = float(self.safe_follow_bonus_scale * gap_score * risk_score)

        if self.wait_window_bonus_scale > 0.0 and obstacle_available:
            longitudinal_gap = float(obstacle_longitudinal)
            window_risk_max = max(0.45, self.safe_follow_risk_max)
            closing_rate = max(0.0, float(obstacle_closing_rate))
            wait_window_gate = bool(
                (not done)
                and self.wait_window_min_gap_m <= longitudinal_gap <= self.wait_window_max_gap_m
                and (not obstacle_planar_ok or obstacle_planar_distance <= self.overtake_arm_planar_max_m)
                and cte_over_out <= 0.45
                and near_offtrack_ratio <= 0.55
                and max(obstacle_risk, near_collision_risk, side_clearance_risk) <= window_risk_max
                and closing_rate <= self.wait_window_max_closing_rate
                and speed >= self.safe_follow_speed_min
                and ("collision" not in term_reasons)
                and ("offtrack" not in term_reasons)
                and ("stuck" not in term_reasons)
            )
            if wait_window_gate:
                window_mid = 0.5 * (self.wait_window_min_gap_m + self.wait_window_max_gap_m)
                window_half = max(1e-6, 0.5 * (self.wait_window_max_gap_m - self.wait_window_min_gap_m))
                gap_score = float(np.clip(1.0 - abs(longitudinal_gap - window_mid) / window_half, 0.0, 1.0))
                closing_score = float(np.clip(
                    1.0 - closing_rate / max(self.wait_window_max_closing_rate, 1e-6),
                    0.0,
                    1.0,
                ))
                risk_score = float(np.clip(
                    1.0 - max(obstacle_risk, near_collision_risk, side_clearance_risk) / max(window_risk_max, 1e-6),
                    0.0,
                    1.0,
                ))
                wait_window_bonus = float(
                    self.wait_window_bonus_scale * gap_score * closing_score * risk_score
                )

        pass_window_valid = False
        pass_window_risk = 0.0
        pass_lateral_clear_m = max(
            self.lateral_overlap_ref_m,
            self.unsafe_close_clearance_m if self.unsafe_close_penalty_scale > 0.0 else 0.0,
            0.05,
        )
        invalid_window_close_risk = 0.0
        if obstacle_available:
            longitudinal_gap = float(obstacle_longitudinal)
            closing_rate = max(0.0, float(obstacle_closing_rate))
            obstacle_lateral_abs = abs(float(obstacle_lateral)) if np.isfinite(obstacle_lateral) else 0.0
            pass_front_min_m = max(0.25, self.overtake_arm_longitudinal_min_m * 0.75)
            pass_front_max_m = max(
                self.wait_window_max_gap_m,
                self.overtake_arm_longitudinal_min_m + self.unsafe_close_longitudinal_m,
            )
            pass_window_front = bool(
                pass_front_min_m <= longitudinal_gap <= pass_front_max_m
                and (not obstacle_planar_ok or obstacle_planar_distance <= self.overtake_arm_planar_max_m)
            )
            pass_window_risk = float(max(
                obstacle_risk,
                near_collision_risk,
                side_clearance_risk,
                obstacle_distance_risk,
                0.50 * near_offtrack_ratio,
            ))
            pass_window_valid = bool(
                (not done)
                and pass_window_front
                and obstacle_lateral_abs >= pass_lateral_clear_m
                and cte_over_out <= 0.35
                and near_offtrack_ratio <= 0.42
                and pass_window_risk <= self.safe_follow_risk_max
                and closing_rate <= max(0.20, self.wait_window_max_closing_rate * 1.25)
                and ("collision" not in term_reasons)
                and ("offtrack" not in term_reasons)
                and ("stuck" not in term_reasons)
            )
            if pass_window_valid:
                self._overtake_window_steps += 1
                if self._overtake_window_steps >= self.overtake_min_front_steps:
                    self._overtake_window_seen = True
            elif pass_window_front:
                self._overtake_window_steps = 0

            close_front_no_window = bool(
                0.0 <= longitudinal_gap <= self.wait_window_max_gap_m
                and (not obstacle_planar_ok or obstacle_planar_distance <= self.close_front_planar_max_m)
                and not pass_window_valid
                and not wait_window_gate
            )
            if close_front_no_window:
                too_close_risk = 0.0
                if longitudinal_gap < self.safe_follow_min_m and obstacle_lateral_abs < pass_lateral_clear_m:
                    too_close_risk = float(np.clip(
                        (self.safe_follow_min_m - longitudinal_gap) / max(self.safe_follow_min_m, 1e-6),
                        0.0,
                        1.0,
                    ))
                invalid_window_close_risk = float(max(
                    too_close_risk,
                    capsule_zone_risk,
                    obstacle_distance_risk,
                    near_collision_risk,
                    0.65 * near_offtrack_ratio,
                ))
        elif not self._overtake_armed:
            self._overtake_window_steps = 0
            self._overtake_window_seen = False

        self._episode_diag["steps_pass_window_valid"] += int(pass_window_valid)
        self._episode_diag["steps_invalid_window_close"] += int(invalid_window_close_risk > 1e-6)

        if self.unsafe_close_penalty_scale > 0.0 and obstacle_available:
            longitudinal_gap = float(obstacle_longitudinal)
            closing_rate = max(0.0, float(obstacle_closing_rate))
            obstacle_lateral_abs = (
                abs(float(obstacle_lateral)) if np.isfinite(obstacle_lateral) else float("inf")
            )
            front_close_gate = bool(0.0 <= longitudinal_gap <= self.unsafe_close_gap_m)
            front_close_without_clearance = bool(
                front_close_gate and obstacle_lateral_abs < pass_lateral_clear_m
            )
            close_without_window = bool(
                (capsule_zone_risk > 1e-6 or front_close_without_clearance)
                and (not obstacle_planar_ok or obstacle_planar_distance <= self.close_front_planar_max_m)
                and not pass_window_valid
                and not wait_window_gate
                and (not done)
                and speed > self.safe_follow_speed_min
                and ("collision" not in term_reasons)
                and ("offtrack" not in term_reasons)
                and ("stuck" not in term_reasons)
            )
            if close_without_window:
                gap_risk = float(np.clip(
                    (self.unsafe_close_gap_m - longitudinal_gap) / max(self.unsafe_close_gap_m, 1e-6),
                    0.0,
                    1.0,
                )) if front_close_without_clearance else 0.0
                clearance_risk = 0.0
                if np.isfinite(obstacle_lateral_abs):
                    clearance_risk = float(np.clip(
                        (self.unsafe_close_clearance_m - obstacle_lateral_abs)
                        / max(self.unsafe_close_clearance_m, 1e-6),
                        0.0,
                        1.0,
                    ))
                ttc_close_risk = 0.0
                if np.isfinite(obstacle_ttc_s) and obstacle_ttc_s > 0.0:
                    ttc_close_risk = float(np.clip(
                        (self.unsafe_close_ttc_s - obstacle_ttc_s)
                        / max(self.unsafe_close_ttc_s, 1e-6),
                        0.0,
                        1.0,
                    ))
                closing_risk = float(np.clip(
                    (closing_rate - self.wait_window_max_closing_rate)
                    / max(self.wait_window_max_closing_rate * 2.0, 1e-6),
                    0.0,
                    1.0,
                ))
                unsafe_close_risk = float(max(
                    capsule_zone_risk,
                    gap_risk,
                    clearance_risk,
                    ttc_close_risk,
                    closing_risk,
                    near_collision_risk,
                    invalid_window_close_risk,
                    obstacle_distance_risk,
                ))
                if unsafe_close_risk > 1e-6:
                    unsafe_close_active = True
                    unsafe_close_penalty = float(-self.unsafe_close_penalty_scale * unsafe_close_risk)
        self._episode_diag["steps_unsafe_close"] += int(unsafe_close_active)

        if (
            self.obstacle_clearance_penalty_scale > 0.0
            and obstacle_available
            and obstacle_distance_risk > 1e-6
            and (not done)
            and ("collision" not in term_reasons)
            and ("offtrack" not in term_reasons)
            and ("stuck" not in term_reasons)
        ):
            obstacle_clearance_penalty = float(
                -self.obstacle_clearance_penalty_scale * obstacle_distance_risk
            )

        if self.force_pass_penalty_scale > 0.0 and obstacle_available:
            longitudinal_gap = float(obstacle_longitudinal)
            close_front_obstacle = bool(
                -capsule_longitudinal_m <= longitudinal_gap <= self.wait_window_max_gap_m
                and (not obstacle_planar_ok or obstacle_planar_distance <= self.force_pass_planar_max_m)
            )
            closing_rate = max(0.0, float(obstacle_closing_rate))
            cte_push_risk = float(np.clip((cte_over_out - 0.32) / max(1e-6, 0.75 - 0.32), 0.0, 1.0))
            offtrack_push_risk = float(np.clip((near_offtrack_ratio - 0.35) / max(1e-6, 1.0 - 0.35), 0.0, 1.0))
            closing_push_risk = float(np.clip(
                (closing_rate - self.wait_window_max_closing_rate)
                / max(self.wait_window_max_closing_rate * 2.0, 1e-6),
                0.0,
                1.0,
            ))
            force_pass_risk = float(max(
                cte_push_risk,
                offtrack_push_risk,
                side_clearance_risk,
                obstacle_distance_risk,
                near_collision_risk,
                closing_push_risk,
                invalid_window_close_risk,
                unsafe_close_risk,
            ))
            if (
                (not done)
                and close_front_obstacle
                and speed > self.safe_follow_speed_min
                and force_pass_risk > 1e-6
                and ("collision" not in term_reasons)
                and ("offtrack" not in term_reasons)
                and ("stuck" not in term_reasons)
            ):
                force_pass_penalty = float(-self.force_pass_penalty_scale * force_pass_risk)
        if self._overtake_cooldown_steps_left > 0:
            self._overtake_cooldown_steps_left -= 1

        if (
            self.prepare_pass_bonus_scale > 0.0
            and obstacle_available
            and np.isfinite(obstacle_lateral)
            and np.isfinite(obstacle_longitudinal)
            and obstacle_longitudinal > 0.0
        ):
            prev_lateral_abs = abs(self._prev_obstacle_lateral) if self._prev_obstacle_lateral is not None else None
            curr_lateral_abs = abs(float(obstacle_lateral))
            clearance_gain = 0.0 if prev_lateral_abs is None else max(0.0, curr_lateral_abs - prev_lateral_abs)
            prepare_gate = bool(
                (not done)
                and speed > 0.2
                and (pass_window_valid or wait_window_gate)
                and np.isfinite(obstacle_ttc_s)
                and 0.8 <= obstacle_ttc_s <= 2.5
                and near_collision_risk <= self.safe_follow_risk_max
                and near_offtrack_ratio <= 0.42
                and cte_over_out <= 0.45
            )
            if prepare_gate and clearance_gain > 1e-4:
                prepare_pass_bonus = float(
                    self.prepare_pass_bonus_scale * np.clip(clearance_gain / max(self.lateral_overlap_ref_m, 1e-3), 0.0, 1.0)
                )

        encounter_front = bool(
            obstacle_available
            and obstacle_longitudinal >= self.overtake_arm_longitudinal_min_m
            and (
                (not obstacle_planar_ok)
                or obstacle_planar_distance <= self.overtake_arm_planar_max_m
            )
        )
        if encounter_front:
            self._overtake_front_steps += 1
            if (
                self._overtake_front_steps >= self.overtake_min_front_steps
                and self._overtake_cooldown_steps_left <= 0
            ):
                self._overtake_armed = True
        elif not self._overtake_armed:
            self._overtake_front_steps = 0

        safe_overtake_state = bool(
            (not done)
            and (cte_over_out <= 0.5)
            and (speed > 0.2)
            and (obstacle_risk <= self.safe_follow_risk_max)
            and (near_collision_risk <= self.safe_follow_risk_max)
            and (side_clearance_risk <= self.safe_follow_risk_max)
            and (obstacle_distance_risk <= self.safe_follow_risk_max)
            and ("collision" not in term_reasons)
            and ("offtrack" not in term_reasons)
            and ("stuck" not in term_reasons)
        )
        if (
            self.commit_pass_bonus_scale > 0.0
            and self._overtake_armed
            and pass_window_valid
            and safe_overtake_state
            and obstacle_available
            and self._prev_obstacle_longitudinal is not None
            and np.isfinite(self._prev_obstacle_longitudinal)
        ):
            longitudinal_improve = max(0.0, self._prev_obstacle_longitudinal - float(obstacle_longitudinal))
            risk_not_worse = float(max(0.0, self._prev_near_collision_risk - near_collision_risk))
            if longitudinal_improve > 1e-4:
                commit_pass_bonus = float(
                    self.commit_pass_bonus_scale
                    * np.clip(longitudinal_improve / 0.5, 0.0, 1.0)
                    * np.clip(0.5 + risk_not_worse / 0.2, 0.0, 1.0)
                )
        overtake_success_clearance_min_m = float(max(
            self.overtake_pass_planar_min_m,
            self.obstacle_clearance_outer_m,
        ))
        overtake_armed_before = bool(self._overtake_armed)
        overtake_window_seen_before = bool(self._overtake_window_seen)
        passed_longitudinal_ok = bool(
            obstacle_available
            and obstacle_longitudinal <= self.overtake_pass_longitudinal_threshold_m
        )
        passed_clearance_ok = bool(
            (not obstacle_planar_ok)
            or obstacle_planar_distance >= overtake_success_clearance_min_m
        )
        passed_to_back = bool(passed_longitudinal_ok and passed_clearance_ok)
        crossed_from_front = bool(
            self._overtake_last_longitudinal is not None
            and self._overtake_last_longitudinal >= 0.0
        )
        overtake_front_ready = bool(
            crossed_from_front or self._overtake_front_steps >= self.overtake_min_front_steps
        )
        post_pass_crossed_now = bool(
            obstacle_available
            and crossed_from_front
            and obstacle_longitudinal <= 0.0
        )
        if post_pass_crossed_now:
            self._post_pass_watch_steps_left = self.post_pass_watch_steps
        overtake_progress_ok = bool(progress_ratio > self.overtake_success_min_progress_ratio)
        overtake_success_ready = bool(
            self.overtake_success_bonus > 0.0
            and overtake_armed_before
            and overtake_window_seen_before
            and overtake_front_ready
        )
        overtake_success_candidate = bool(overtake_success_ready and passed_to_back)
        overtake_success_blocked_clearance = bool(
            overtake_success_ready
            and passed_longitudinal_ok
            and not passed_clearance_ok
        )
        overtake_success_blocked_progress = bool(
            overtake_success_candidate
            and safe_overtake_state
            and not overtake_progress_ok
        )
        overtake_success_blocked_safety = bool(
            overtake_success_candidate
            and not safe_overtake_state
        )
        self._episode_diag["steps_overtake_success_ready"] += int(overtake_success_ready)
        self._episode_diag["steps_overtake_passed_longitudinal"] += int(passed_longitudinal_ok)
        self._episode_diag["steps_overtake_success_clearance_ok"] += int(
            overtake_success_ready and passed_clearance_ok
        )
        self._episode_diag["steps_overtake_success_candidate"] += int(overtake_success_candidate)
        self._episode_diag["steps_overtake_success_blocked_clearance"] += int(
            overtake_success_blocked_clearance
        )
        self._episode_diag["steps_overtake_success_blocked_progress"] += int(
            overtake_success_blocked_progress
        )
        self._episode_diag["steps_overtake_success_blocked_safety"] += int(
            overtake_success_blocked_safety
        )
        if (
            self.overtake_success_bonus > 0.0
            and overtake_armed_before
            and overtake_window_seen_before
            and safe_overtake_state
            and overtake_progress_ok
            and passed_to_back
            and overtake_front_ready
        ):
            overtake_bonus = float(self.overtake_success_bonus)
            overtake_success = True
            self._episode_diag["overtake_success_grant_count"] += 1
            self._episode_overtake_count += 1
            self.episode_stats["overtake_count"] = int(self._episode_overtake_count)
            self._overtake_armed = False
            self._overtake_front_steps = 0
            self._overtake_window_steps = 0
            self._overtake_window_seen = False
            self._overtake_cooldown_steps_left = self.overtake_rearm_cooldown_steps
            self._post_pass_active = bool(self.post_pass_stability_bonus > 0.0)
            self._post_pass_stable_steps = 0
            info["overtake_respawn_requested"] = float(self._request_overtake_respawn(info))
        elif (
            overtake_armed_before
            and passed_to_back
            and overtake_front_ready
        ):
            # A pass that never had a safe window should not keep the arm state
            # alive or earn delayed stability credit.
            self._overtake_armed = False
            self._overtake_front_steps = 0
            self._overtake_window_steps = 0
            self._overtake_window_seen = False
            self._overtake_cooldown_steps_left = self.overtake_rearm_cooldown_steps
            self._post_pass_active = False
            self._post_pass_stable_steps = 0
        elif done and (("collision" in term_reasons) or ("offtrack" in term_reasons) or ("stuck" in term_reasons)):
            self._overtake_armed = False
            self._overtake_front_steps = 0
            self._overtake_window_steps = 0
            self._overtake_window_seen = False
            self._post_pass_active = False
            self._post_pass_stable_steps = 0

        if self._post_pass_active:
            post_pass_stable_state = bool(
                (not done)
                and (cte_over_out <= 0.5)
                and (speed > 0.2)
                and (near_collision_risk <= self.safe_follow_risk_max)
                and (side_clearance_risk <= self.safe_follow_risk_max)
                and ("collision" not in term_reasons)
                and ("offtrack" not in term_reasons)
                and ("stuck" not in term_reasons)
                and ((not obstacle_available) or obstacle_longitudinal <= -0.3)
            )
            if post_pass_stable_state:
                self._post_pass_stable_steps += 1
                if self._post_pass_stable_steps >= self.post_pass_stability_steps:
                    post_pass_bonus = float(self.post_pass_stability_bonus)
                    self._episode_post_pass_count += 1
                    self.episode_stats["post_pass_stability_count"] = int(self._episode_post_pass_count)
                    self._post_pass_active = False
                    self._post_pass_stable_steps = 0
            else:
                self._post_pass_active = False
                self._post_pass_stable_steps = 0

        post_pass_watch_active = bool(self._post_pass_watch_steps_left > 0)
        post_pass_behind_near = bool(
            obstacle_available
            and obstacle_longitudinal <= 0.0
            and obstacle_longitudinal >= -self.post_pass_watch_longitudinal_m
            and (
                (not obstacle_planar_ok)
                or obstacle_planar_distance <= self.post_pass_watch_longitudinal_m
            )
        )
        post_pass_clearance_ok = bool(
            (not post_pass_behind_near)
            or (
                obstacle_planar_ok
                and obstacle_planar_distance >= self.obstacle_clearance_outer_m
            )
        )
        if post_pass_watch_active:
            self._episode_diag["steps_post_pass_watch"] += 1
            if post_pass_behind_near and obstacle_planar_ok:
                self._episode_diag["post_pass_planar_distance_min"] = min(
                    float(self._episode_diag["post_pass_planar_distance_min"]),
                    float(obstacle_planar_distance),
                )
            if post_pass_behind_near:
                progress_risk = 0.0 if overtake_progress_ok else 0.35
                post_pass_cut_in_risk = float(max(
                    obstacle_distance_risk,
                    capsule_zone_risk,
                    progress_risk,
                ))
                post_pass_cut_in_active = bool(
                    post_pass_cut_in_risk > 1e-6
                    and (not post_pass_clearance_ok or not overtake_progress_ok)
                )
                if post_pass_cut_in_active:
                    self._episode_diag["steps_post_pass_cut_in"] += 1
                    self._episode_diag["post_pass_cut_in_risk_sum"] += float(post_pass_cut_in_risk)
                    if "collision" in term_reasons:
                        self._episode_diag["post_pass_terminal_collision"] = 1
                    if (
                        self.post_pass_cut_in_penalty_scale > 0.0
                        and (not done)
                        and ("collision" not in term_reasons)
                        and ("offtrack" not in term_reasons)
                        and ("stuck" not in term_reasons)
                    ):
                        post_pass_cut_in_penalty = float(
                            -self.post_pass_cut_in_penalty_scale * post_pass_cut_in_risk
                        )
            self._post_pass_watch_steps_left = max(0, self._post_pass_watch_steps_left - 1)

        self._overtake_last_longitudinal = (
            float(obstacle_longitudinal) if obstacle_available else None
        )
        info["safe_follow_bonus"] = float(safe_follow_bonus)
        info["wait_window_bonus"] = float(wait_window_bonus)
        info["force_pass_penalty"] = float(force_pass_penalty)
        info["unsafe_close_penalty"] = float(unsafe_close_penalty)
        info["obstacle_clearance_penalty"] = float(obstacle_clearance_penalty)
        info["post_pass_cut_in_penalty"] = float(post_pass_cut_in_penalty)
        info["prepare_pass_bonus"] = float(prepare_pass_bonus)
        info["commit_pass_bonus"] = float(commit_pass_bonus)
        info["overtake_success"] = bool(overtake_success)
        info["overtake_bonus"] = float(overtake_bonus)
        info["overtake_count"] = int(self._episode_overtake_count)
        info.setdefault("overtake_respawn_requested", 0.0)
        info["post_pass_stability_bonus"] = float(post_pass_bonus)
        info["post_pass_stability_count"] = int(self._episode_post_pass_count)
        info["reward_debug/overtake_armed"] = float(self._overtake_armed)
        info["reward_debug/overtake_front_steps"] = float(self._overtake_front_steps)
        info["reward_debug/overtake_window_steps"] = float(self._overtake_window_steps)
        info["reward_debug/overtake_window_seen"] = float(self._overtake_window_seen)
        info["reward_debug/overtake_cooldown"] = float(self._overtake_cooldown_steps_left)
        info["reward_debug/post_pass_active"] = float(self._post_pass_active)
        info["reward_debug/post_pass_stable_steps"] = float(self._post_pass_stable_steps)
        info["reward_debug/r_safe_follow"] = float(safe_follow_bonus)
        info["reward_debug/r_wait_window"] = float(wait_window_bonus)
        info["reward_debug/r_force_pass"] = float(force_pass_penalty)
        info["reward_debug/r_unsafe_close"] = float(unsafe_close_penalty)
        info["reward_debug/r_obstacle_clearance"] = float(obstacle_clearance_penalty)
        info["reward_debug/r_post_pass_cut_in"] = float(post_pass_cut_in_penalty)
        info["reward_debug/wait_window_gate"] = float(wait_window_gate)
        info["reward_debug/force_pass_risk"] = float(force_pass_risk)
        info["reward_debug/unsafe_close_risk"] = float(unsafe_close_risk)
        info["reward_debug/unsafe_close_active"] = float(unsafe_close_active)
        info["reward_debug/post_pass_watch_active"] = float(post_pass_watch_active)
        info["reward_debug/post_pass_behind_near"] = float(post_pass_behind_near)
        info["reward_debug/post_pass_clearance_ok"] = float(post_pass_clearance_ok)
        info["reward_debug/post_pass_cut_in_active"] = float(post_pass_cut_in_active)
        info["reward_debug/post_pass_cut_in_risk"] = float(post_pass_cut_in_risk)
        info["reward_debug/post_pass_watch_steps_left"] = float(self._post_pass_watch_steps_left)
        info["reward_debug/overtake_armed_before"] = float(overtake_armed_before)
        info["reward_debug/overtake_window_seen_before"] = float(overtake_window_seen_before)
        info["reward_debug/overtake_front_ready"] = float(overtake_front_ready)
        info["reward_debug/overtake_success_ready"] = float(overtake_success_ready)
        info["reward_debug/overtake_passed_longitudinal_ok"] = float(passed_longitudinal_ok)
        info["reward_debug/overtake_passed_clearance_ok"] = float(passed_clearance_ok)
        info["reward_debug/overtake_success_candidate"] = float(overtake_success_candidate)
        info["reward_debug/overtake_success_blocked_clearance"] = float(overtake_success_blocked_clearance)
        info["reward_debug/overtake_success_blocked_progress"] = float(overtake_success_blocked_progress)
        info["reward_debug/overtake_success_blocked_safety"] = float(overtake_success_blocked_safety)
        info["reward_debug/overtake_success_granted"] = float(overtake_success)
        info["reward_debug/overtake_success_clearance_min_m"] = float(overtake_success_clearance_min_m)
        info["reward_debug/overtake_progress_ok"] = float(overtake_progress_ok)
        info["reward_debug/overtake_success_min_progress_ratio"] = float(self.overtake_success_min_progress_ratio)
        info["reward_debug/unsafe_close_gap_m"] = float(self.unsafe_close_gap_m)
        info["reward_debug/unsafe_close_clearance_m"] = float(self.unsafe_close_clearance_m)
        info["reward_debug/unsafe_close_longitudinal_m"] = float(self.unsafe_close_longitudinal_m)
        info["reward_debug/unsafe_close_ttc_s"] = float(self.unsafe_close_ttc_s)
        info["reward_debug/pass_window_valid"] = float(pass_window_valid)
        info["reward_debug/pass_window_risk"] = float(pass_window_risk)
        info["reward_debug/pass_lateral_clear_m"] = float(pass_lateral_clear_m)
        info["reward_debug/invalid_window_close_risk"] = float(invalid_window_close_risk)
        info["reward_debug/r_prepare_pass"] = float(prepare_pass_bonus)
        info["reward_debug/r_commit_pass"] = float(commit_pass_bonus)
        info["reward_debug/overtake_obstacle_longitudinal"] = float(obstacle_longitudinal)
        info["reward_debug/overtake_obstacle_planar_distance"] = float(obstacle_planar_distance)
        info["reward_debug/r_overtake"] = float(overtake_bonus)
        info["reward_debug/r_post_pass"] = float(post_pass_bonus)

        # 调试日志
        info["reward_debug/survival"]         = survival_reward
        info["reward_debug/speed_gate"]       = speed_gate
        info["reward_debug/alive_forward_gate"] = alive_forward_gate
        info["reward_debug/center_factor"]    = center_factor
        info["reward_debug/stuck_counter"]    = self.stuck_counter
        info["reward_debug/cte_boundary"]     = cte_boundary
        info["reward_debug/cte_out_boundary"] = cte_out_boundary
        info["reward_debug/offtrack_counter"] = self.offtrack_counter
        info["reward_debug/lat_err_cte"]      = float(lat_err_cte)
        info["reward_debug/cte_abs"]          = float(lat_err_cte_abs)
        info["reward_debug/cte_over_in"]      = float(cte_over_in)
        info["reward_debug/cte_over_out"]     = float(cte_over_out)
        info["reward_debug/reset_env_done_grace_active"] = float(reset_env_done_grace_active)
        info["reward_debug/reset_collision_grace_active"] = float(reset_collision_grace_active)
        info["reward_debug/reset_env_done_masked"] = float(env_done_masked)
        info["reward_debug/reset_collision_masked"] = float(collision_masked)
        self._episode_diag["offtrack_counter_max"] = max(
            int(self._episode_diag["offtrack_counter_max"]),
            int(self.offtrack_counter),
        )
        self._episode_diag["stuck_counter_max"] = max(
            int(self._episode_diag["stuck_counter_max"]),
            int(self.stuck_counter),
        )

        # 平滑惩罚
        smooth_penalty = 0.0
        jerk_penalty   = 0.0
        mismatch_penalty = 0.0
        sat_penalty    = 0.0
        steer_budget_penalty = 0.0
        sign_flip_penalty = 0.0
        micro_wiggle_penalty = 0.0
        steer_budget = 1.0
        steer_over_budget = 0.0
        steer_sign_flip = 0.0
        micro_wiggle_signal = 0.0
        rate_limit_hit = 0.0
        steer_clip_hit = 0.0
        delta_delta_limit_hit = 0.0
        curve_ratio_for_penalty = float(np.clip(kappa_abs / self.progress_kappa_ref, 0.0, 1.0))
        curve_penalty_scale = float(max(0.35, 1.0 - self.smooth_curve_relief * curve_ratio_for_penalty))
        if self.action_safety_wrapper is not None:
            diag = self.action_safety_wrapper.diag
            abs_delta             = abs(diag["delta_steer"])
            abs_jerk              = abs(diag["delta_steer"] - diag["delta_steer_prev"])
            abs_mismatch          = abs(diag["mismatch"])
            steer_exec            = float(diag.get("steer_exec", 0.0))
            prev_steer_exec       = float(steer_exec - diag["delta_steer"])
            rate_excess_bounded   = float(diag["rate_excess_bounded"])
            delta_delta_excess_bounded = float(diag.get("delta_delta_excess_bounded", 0.0))
            steer_clip_bounded = float(
                np.tanh(
                    max(0.0, abs(float(diag.get("steer_exec", 0.0))) - 0.85) / 0.15
                )
            )
            rate_limit_hit        = float(diag["rate_limit_hit"])
            delta_delta_limit_hit = float(diag.get("delta_delta_limit_hit", False))
            steer_clip_hit        = float(diag["steer_clip_hit"])

            # 发夹弯等高曲率段适度降低平滑惩罚，避免策略“怕转向”
            smooth_penalty = -self.w_d   * abs_delta * curve_penalty_scale
            jerk_penalty   = -self.w_dd  * abs_jerk * curve_penalty_scale
            mismatch_penalty = -self.w_m * abs_mismatch * curve_penalty_scale
            sat_penalty    = -self.w_sat * max(
                rate_excess_bounded,
                delta_delta_excess_bounded,
                0.5 * steer_clip_bounded,
            )
            steer_budget = float(
                self.steer_budget_straight
                + (self.steer_budget_curve - self.steer_budget_straight) * curve_ratio_for_penalty
                + self.steer_budget_obstacle_relief * near_collision_risk
            )
            steer_budget = float(np.clip(steer_budget, 0.05, 0.98))
            steer_over_budget = float(max(0.0, abs(steer_exec) - steer_budget))
            if self.w_steer_budget > 0.0 and steer_over_budget > 0.0:
                # 高速直道超预算大舵角最伤舵机也最容易蛇形；弯道/近障碍预算已放宽。
                speed_penalty_scale = 0.50 + 0.50 * speed_gate
                steer_budget_penalty = -self.w_steer_budget * (steer_over_budget ** 2) * speed_penalty_scale
            if (
                self.w_sign_flip > 0.0
                and abs(steer_exec) >= self.sign_flip_min_abs_steer
                and abs(prev_steer_exec) >= self.sign_flip_min_abs_steer
                and steer_exec * prev_steer_exec < 0.0
            ):
                steer_sign_flip = 1.0
                # 曲率越大、障碍越近，越可能是合理避让，降低但不完全取消惩罚。
                sign_flip_scale = float(
                    max(0.20, 1.0 - 0.55 * curve_ratio_for_penalty - 0.55 * near_collision_risk)
                )
                sign_flip_penalty = -self.w_sign_flip * sign_flip_scale
            if self.w_micro_wiggle > 0.0:
                steer_pair_abs = max(abs(steer_exec), abs(prev_steer_exec))
                in_micro_band = (
                    steer_pair_abs >= self.micro_wiggle_min_abs_steer
                    and steer_pair_abs <= self.micro_wiggle_max_abs_steer
                )
                if in_micro_band:
                    micro_cross = float(steer_exec * prev_steer_exec < 0.0)
                    micro_delta_gate = float(np.clip(abs_delta / 0.08, 0.0, 1.0))
                    micro_jerk_gate = float(np.clip((abs_jerk - 0.025) / 0.12, 0.0, 1.0))
                    micro_wiggle_signal = float(max(micro_cross, 0.5 * micro_delta_gate * micro_jerk_gate))
                    if micro_wiggle_signal > 0.0:
                        # 弯道和近障碍允许必要换线；直道低风险的小幅来回拧头才重点惩罚。
                        micro_relief = float(
                            max(0.15, 1.0 - 0.70 * curve_ratio_for_penalty - 0.85 * near_collision_risk)
                        )
                        micro_wiggle_penalty = -self.w_micro_wiggle * micro_wiggle_signal * micro_relief
            self._episode_diag["steps_rate_limit_hit"] += int(rate_limit_hit > 0.5)
            self._episode_diag["steps_steer_clip_hit"] += int(steer_clip_hit > 0.5)

            self.smooth_stats.append({
                "abs_delta":            abs_delta,
                "abs_jerk":             abs_jerk,
                "abs_mismatch":         abs_mismatch,
                "steer_budget":         steer_budget,
                "steer_over_budget":    steer_over_budget,
                "steer_sign_flip":      steer_sign_flip,
                "micro_wiggle_signal":  micro_wiggle_signal,
                "rate_limit_hit":       rate_limit_hit,
                "rate_excess_raw":      float(diag["rate_excess_raw"]),
                "rate_excess_bounded":  rate_excess_bounded,
                "delta_delta_limit_hit": delta_delta_limit_hit,
                "delta_delta_excess_bounded": delta_delta_excess_bounded,
                "steer_clip_hit":       steer_clip_hit,
            })
            info["smooth/abs_delta_steer"]      = abs_delta
            info["smooth/rate_limit_hit"]        = rate_limit_hit
            info["smooth/rate_excess_raw"]       = float(diag["rate_excess_raw"])
            info["smooth/rate_excess_bounded"]   = rate_excess_bounded
            info["smooth/delta_delta_limit_hit"] = delta_delta_limit_hit
            info["smooth/delta_delta_excess_bounded"] = delta_delta_excess_bounded
            info["smooth/servo_deadband_hold"]   = float(diag.get("servo_deadband_hold", False))
            info["smooth/steer_clip_hit"]        = steer_clip_hit
            info["smooth/abs_mismatch"]          = abs_mismatch
            info["smooth/abs_jerk"]              = abs_jerk
            info["smooth/steer_budget"]          = steer_budget
            info["smooth/steer_over_budget"]     = steer_over_budget
            info["smooth/steer_sign_flip"]       = steer_sign_flip
            info["smooth/micro_wiggle_signal"]   = micro_wiggle_signal
            info["smooth/hairpin_relax_active"]  = float(diag.get("hairpin_relax_active", 0.0))

        info["reward_debug/progress_ratio"] = float(progress_ratio)
        info["reward_debug/progress_reward_raw"] = float(progress_reward_raw)
        info["reward_debug/progress_reward"] = float(progress_reward)
        info["reward_debug/progress_center_gate"] = float(progress_center_gate)
        info["reward_debug/progress_forward_gain"] = float(progress_forward_gain)
        info["reward_debug/progress_curve_ratio"] = float(curve_ratio_for_penalty)
        info["reward_debug/curve_penalty_scale"] = float(curve_penalty_scale)
        info["reward_debug/lat_err_norm"] = float(lat_err_norm)
        info["reward_debug/heading_err_abs"] = float(heading_err_abs)
        info["reward_debug/v_ref"] = float(v_ref)
        info["reward_debug/speed_ref_err_norm"] = float(speed_err_norm)
        info["reward_debug/r_center"] = float(center_penalty)
        info["reward_debug/r_heading"] = float(heading_penalty)
        info["reward_debug/r_speed_ref"] = float(speed_ref_penalty)
        info["reward_debug/r_time"] = float(time_penalty)
        info["reward_debug/r_steer_budget"] = float(steer_budget_penalty)
        info["reward_debug/r_sign_flip"] = float(sign_flip_penalty)
        info["reward_debug/r_micro_wiggle"] = float(micro_wiggle_penalty)
        info["reward_debug/r_stuck"] = float(stuck_penalty)
        info["reward_debug/stuck_speed_threshold"] = float(self.stuck_speed_threshold)
        info["reward_debug/stuck_progress_threshold"] = float(self.stuck_progress_threshold)
        info["reward_debug/stuck_low_speed"] = float(stuck_low_speed)
        info["reward_debug/stuck_low_progress"] = float(stuck_low_progress)
        info["reward_debug/stuck_grace_steps"] = float(self.stuck_grace_steps)

        throttle_high_penalty = 0.0
        if throttle_cmd > self.throttle_penalty_threshold:
            speed_norm_for_penalty = float(np.clip(speed / 4.0, 0.0, 2.0))
            # 安全驾驶惩罚：油门过高时按速度加重（速度越大，扣分越多）
            throttle_high_penalty = -self.throttle_penalty_amount * (1.0 + speed_norm_for_penalty)
        else:
            speed_norm_for_penalty = float(np.clip(speed / 4.0, 0.0, 2.0))
        throttle_high_penalty_hit = float(throttle_high_penalty < 0.0)
        self._episode_diag["steps_throttle_high_penalty_hit"] += int(throttle_high_penalty_hit > 0.5)
        info["reward_debug/throttle_cmd"] = float(throttle_cmd)
        info["reward_debug/speed_norm_for_throttle_penalty"] = float(speed_norm_for_penalty)
        info["reward_debug/throttle_high_penalty"] = float(throttle_high_penalty)
        info["reward_debug/throttle_high_penalty_hit"] = throttle_high_penalty_hit

        terminal_offtrack_progress_discount = 0.0
        if "offtrack" in term_reasons and self.terminal_offtrack_progress_scale < 1.0:
            progress_before_discount = float(progress_reward)
            progress_reward *= self.terminal_offtrack_progress_scale
            progress_reward_raw *= self.terminal_offtrack_progress_scale
            terminal_offtrack_progress_discount = progress_before_discount - float(progress_reward)
        self._episode_diag["terminal_offtrack_progress_discount"] += float(
            terminal_offtrack_progress_discount
        )

        total_reward = (
            survival_reward + speed_reward + progress_reward + cte_term +
            center_penalty + heading_penalty + speed_ref_penalty + time_penalty +
            terminal_penalty + near_offtrack_total_penalty + near_collision_penalty + lap_reward +
            safe_follow_bonus + wait_window_bonus + force_pass_penalty + unsafe_close_penalty +
            obstacle_clearance_penalty + post_pass_cut_in_penalty +
            prepare_pass_bonus + commit_pass_bonus + overtake_bonus + post_pass_bonus +
            smooth_penalty + jerk_penalty + mismatch_penalty + sat_penalty +
            steer_budget_penalty + sign_flip_penalty + micro_wiggle_penalty +
            throttle_high_penalty + stuck_penalty
        )

        bad_guard_triggered = False
        bad_guard_reason = ""
        bad_guard_penalty = 0.0
        if (
            self.bad_episode_guard_min_steps > 0
            and (not done)
            and episode_step >= self.bad_episode_guard_min_steps
        ):
            diag_steps_for_guard = max(1, int(self._episode_diag["steps_total"]))
            cte_over_in_rate_for_guard = float(
                self._episode_diag["steps_cte_over_in"] / diag_steps_for_guard
            )
            forward_progress_for_guard = float(self._episode_diag["progress_ratio_forward_sum"])
            projected_episode_reward = float(self._reward_parts_episode["total"] + total_reward)
            poor_return = projected_episode_reward <= self.bad_episode_guard_reward_floor
            poor_geometry = (
                cte_over_in_rate_for_guard >= self.bad_episode_guard_cte_over_in_rate
                and forward_progress_for_guard <= self.bad_episode_guard_min_forward_progress
            )
            if poor_return or poor_geometry:
                bad_guard_triggered = True
                bad_guard_reason = "reward_floor" if poor_return else "poor_geometry"
                done = True
                if "bad_episode_guard" not in term_reasons:
                    term_reasons.append("bad_episode_guard")
                if self.bad_episode_guard_penalty > 0.0:
                    bad_guard_penalty = -self.bad_episode_guard_penalty
                    total_reward += bad_guard_penalty
        if bad_guard_triggered:
            self._episode_diag["bad_episode_guard_triggered"] = 1
            self._episode_diag["bad_episode_guard_step"] = int(episode_step)

        # reward decay: 超过 ref_steps 后按 ref/step 衰减每步奖励
        ep_steps = self.episode_stats["steps"] + 1   # 当前步（从1开始）
        if self.reward_decay_ref_steps > 0 and ep_steps > self.reward_decay_ref_steps:
            total_reward /= (ep_steps / self.reward_decay_ref_steps)

        collision_cap_penalty = 0.0
        if (
            done
            and "collision" in term_reasons
            and self.collision_episode_reward_cap is not None
        ):
            projected_episode_reward = float(self._reward_parts_episode["total"] + total_reward)
            if projected_episode_reward > self.collision_episode_reward_cap:
                collision_cap_penalty = float(self.collision_episode_reward_cap - projected_episode_reward)
                total_reward += collision_cap_penalty
                self._episode_diag["collision_episode_reward_cap_penalty"] += float(collision_cap_penalty)

        offtrack_cap_penalty = 0.0
        if (
            done
            and "offtrack" in term_reasons
            and self.offtrack_episode_reward_cap is not None
        ):
            projected_episode_reward = float(self._reward_parts_episode["total"] + total_reward)
            if projected_episode_reward > self.offtrack_episode_reward_cap:
                offtrack_cap_penalty = float(self.offtrack_episode_reward_cap - projected_episode_reward)
                total_reward += offtrack_cap_penalty
                self._episode_diag["offtrack_episode_reward_cap_penalty"] += float(offtrack_cap_penalty)

        info["reward_debug/terminal_offtrack_progress_discount"] = float(
            terminal_offtrack_progress_discount
        )
        info["reward_debug/collision_episode_reward_cap"] = float(
            self.collision_episode_reward_cap
            if self.collision_episode_reward_cap is not None
            else 1e9
        )
        info["reward_debug/collision_episode_reward_cap_penalty"] = float(collision_cap_penalty)
        info["reward_debug/offtrack_episode_reward_cap"] = float(
            self.offtrack_episode_reward_cap
            if self.offtrack_episode_reward_cap is not None
            else 1e9
        )
        info["reward_debug/offtrack_episode_reward_cap_penalty"] = float(offtrack_cap_penalty)
        info["reward_debug/bad_episode_guard_triggered"] = float(bad_guard_triggered)
        info["reward_debug/bad_episode_guard_reason"] = str(bad_guard_reason)
        info["reward_debug/bad_episode_guard_step"] = float(
            self._episode_diag["bad_episode_guard_step"]
        )

        self.episode_stats["total_reward"] += total_reward
        self.episode_stats["steps"] += 1

        # ── 奖励分项累计（供 ep_info_buffer → PerSceneStatsCallback 消费）──
        self._reward_parts_episode["survival"]  += survival_reward
        self._reward_parts_episode["speed"]     += speed_reward
        self._reward_parts_episode["progress"]  += progress_reward
        self._reward_parts_episode["cte"]       += cte_term
        self._reward_parts_episode["center"]    += center_penalty
        self._reward_parts_episode["heading"]   += heading_penalty
        self._reward_parts_episode["speed_ref"] += speed_ref_penalty
        self._reward_parts_episode["time"]      += time_penalty
        self._reward_parts_episode["collision"] += terminal_penalty
        self._reward_parts_episode["near_offtrack"] += near_offtrack_total_penalty
        self._reward_parts_episode["near_collision"] += near_collision_penalty
        self._reward_parts_episode["lap"]       += lap_reward
        self._reward_parts_episode["lap_raw"]   += lap_reward_raw
        self._reward_parts_episode["follow"]    += safe_follow_bonus + wait_window_bonus + prepare_pass_bonus
        self._reward_parts_episode["wait_window"] += wait_window_bonus
        self._reward_parts_episode["force_pass"] += force_pass_penalty
        self._reward_parts_episode["unsafe_close"] += unsafe_close_penalty
        self._reward_parts_episode["obstacle_clearance"] += obstacle_clearance_penalty
        self._reward_parts_episode["overtake"]  += overtake_bonus + commit_pass_bonus
        self._reward_parts_episode["post_pass"] += post_pass_bonus
        self._reward_parts_episode["post_pass_cut_in"] += post_pass_cut_in_penalty
        self._reward_parts_episode["smooth"]    += smooth_penalty
        self._reward_parts_episode["jerk"]      += jerk_penalty
        self._reward_parts_episode["mismatch"]  += mismatch_penalty
        self._reward_parts_episode["steer_budget"] += steer_budget_penalty
        self._reward_parts_episode["sign_flip"] += sign_flip_penalty
        self._reward_parts_episode["micro_wiggle"] += micro_wiggle_penalty
        self._reward_parts_episode["sat"]       += sat_penalty
        self._reward_parts_episode["stuck"]     += stuck_penalty
        self._reward_parts_episode["bad_guard"] += bad_guard_penalty
        self._reward_parts_episode["collision_cap"] += collision_cap_penalty
        self._reward_parts_episode["offtrack_cap"] += offtrack_cap_penalty
        self._reward_parts_episode["total"]     += total_reward

        self._prev_obstacle_longitudinal = float(obstacle_longitudinal) if obstacle_available else None
        self._prev_obstacle_lateral = float(obstacle_lateral) if obstacle_available and np.isfinite(obstacle_lateral) else None
        self._prev_near_collision_risk = float(near_collision_risk)

        if done:
            info["ep_r_survival"]  = self._reward_parts_episode["survival"]
            info["ep_r_speed"]     = self._reward_parts_episode["speed"]
            info["ep_r_progress"]  = self._reward_parts_episode["progress"]
            info["ep_r_cte"]       = self._reward_parts_episode["cte"]
            info["ep_r_center"]    = self._reward_parts_episode["center"]
            info["ep_r_heading"]   = self._reward_parts_episode["heading"]
            info["ep_r_speed_ref"] = self._reward_parts_episode["speed_ref"]
            info["ep_r_time"]      = self._reward_parts_episode["time"]
            info["ep_r_collision"] = self._reward_parts_episode["collision"]
            info["ep_r_near_offtrack"] = self._reward_parts_episode["near_offtrack"]
            info["ep_r_near_collision"] = self._reward_parts_episode["near_collision"]
            info["ep_r_lap"]       = self._reward_parts_episode["lap"]
            info["ep_r_lap_raw"]   = self._reward_parts_episode["lap_raw"]
            info["ep_r_follow"]    = self._reward_parts_episode["follow"]
            info["ep_r_wait_window"] = self._reward_parts_episode["wait_window"]
            info["ep_r_force_pass"] = self._reward_parts_episode["force_pass"]
            info["ep_r_unsafe_close"] = self._reward_parts_episode["unsafe_close"]
            info["ep_r_obstacle_clearance"] = self._reward_parts_episode["obstacle_clearance"]
            info["ep_r_overtake"]  = self._reward_parts_episode["overtake"]
            info["ep_r_post_pass"] = self._reward_parts_episode["post_pass"]
            info["ep_r_post_pass_cut_in"] = self._reward_parts_episode["post_pass_cut_in"]
            info["ep_overtake_count"] = int(self._episode_overtake_count)
            info["ep_post_pass_stability_count"] = int(self._episode_post_pass_count)
            info["ep_soft_lap_count"] = self._soft_lap_count
            info["ep_r_smooth"]    = self._reward_parts_episode["smooth"]
            info["ep_r_jerk"]      = self._reward_parts_episode["jerk"]
            info["ep_r_mismatch"]  = self._reward_parts_episode["mismatch"]
            info["ep_r_steer_budget"] = self._reward_parts_episode["steer_budget"]
            info["ep_r_sign_flip"] = self._reward_parts_episode["sign_flip"]
            info["ep_r_micro_wiggle"] = self._reward_parts_episode["micro_wiggle"]
            info["ep_r_sat"]       = self._reward_parts_episode["sat"]
            info["ep_r_stuck"]     = self._reward_parts_episode["stuck"]
            info["ep_r_bad_guard"] = self._reward_parts_episode["bad_guard"]
            info["ep_r_collision_cap"] = self._reward_parts_episode["collision_cap"]
            info["ep_r_offtrack_cap"] = self._reward_parts_episode["offtrack_cap"]
            info["ep_r_total"]     = self._reward_parts_episode["total"]
            diag_steps = max(1, int(self._episode_diag["steps_total"]))
            cte_samples = np.asarray(self._episode_diag["cte_abs_samples"], dtype=np.float64)
            if cte_samples.size > 0:
                info["ep_cte_abs_p50"] = float(np.percentile(cte_samples, 50))
                info["ep_cte_abs_p90"] = float(np.percentile(cte_samples, 90))
                info["ep_cte_abs_p99"] = float(np.percentile(cte_samples, 99))
            else:
                info["ep_cte_abs_p50"] = 0.0
                info["ep_cte_abs_p90"] = 0.0
                info["ep_cte_abs_p99"] = 0.0
            info["ep_cte_over_in_rate"] = float(self._episode_diag["steps_cte_over_in"] / diag_steps)
            info["ep_cte_over_out_rate"] = float(self._episode_diag["steps_cte_over_out"] / diag_steps)
            info["ep_rate_limit_hit_rate"] = float(self._episode_diag["steps_rate_limit_hit"] / diag_steps)
            info["ep_steer_clip_hit_rate"] = float(self._episode_diag["steps_steer_clip_hit"] / diag_steps)
            info["ep_throttle_high_penalty_hit_rate"] = float(
                self._episode_diag["steps_throttle_high_penalty_hit"] / diag_steps
            )
            info["ep_pass_window_valid_rate"] = float(
                self._episode_diag["steps_pass_window_valid"] / diag_steps
            )
            info["ep_invalid_window_close_rate"] = float(
                self._episode_diag["steps_invalid_window_close"] / diag_steps
            )
            info["ep_unsafe_close_rate"] = float(
                self._episode_diag["steps_unsafe_close"] / diag_steps
            )
            info["ep_obstacle_clearance_band_rate"] = float(
                self._episode_diag["steps_obstacle_clearance_band"] / diag_steps
            )
            info["ep_obstacle_clearance_critical_rate"] = float(
                self._episode_diag["steps_obstacle_clearance_critical"] / diag_steps
            )
            info["ep_obstacle_clearance_risk_mean"] = float(
                self._episode_diag["obstacle_clearance_risk_sum"] / diag_steps
            )
            min_obstacle_dist = float(self._episode_diag["obstacle_planar_distance_min"])
            info["ep_obstacle_planar_distance_min"] = (
                min_obstacle_dist if np.isfinite(min_obstacle_dist) else 0.0
            )
            info["ep_post_pass_watch_rate"] = float(
                self._episode_diag["steps_post_pass_watch"] / diag_steps
            )
            info["ep_post_pass_cut_in_rate"] = float(
                self._episode_diag["steps_post_pass_cut_in"] / diag_steps
            )
            info["ep_post_pass_cut_in_risk_mean"] = float(
                self._episode_diag["post_pass_cut_in_risk_sum"] / diag_steps
            )
            min_post_pass_dist = float(self._episode_diag["post_pass_planar_distance_min"])
            info["ep_post_pass_planar_distance_min"] = (
                min_post_pass_dist if np.isfinite(min_post_pass_dist) else 0.0
            )
            info["ep_post_pass_terminal_collision"] = float(
                self._episode_diag["post_pass_terminal_collision"]
            )
            info["ep_overtake_success_ready_rate"] = float(
                self._episode_diag["steps_overtake_success_ready"] / diag_steps
            )
            info["ep_overtake_passed_longitudinal_rate"] = float(
                self._episode_diag["steps_overtake_passed_longitudinal"] / diag_steps
            )
            info["ep_overtake_success_clearance_ok_rate"] = float(
                self._episode_diag["steps_overtake_success_clearance_ok"] / diag_steps
            )
            info["ep_overtake_success_candidate_rate"] = float(
                self._episode_diag["steps_overtake_success_candidate"] / diag_steps
            )
            info["ep_overtake_success_blocked_clearance_rate"] = float(
                self._episode_diag["steps_overtake_success_blocked_clearance"] / diag_steps
            )
            info["ep_overtake_success_blocked_progress_rate"] = float(
                self._episode_diag["steps_overtake_success_blocked_progress"] / diag_steps
            )
            info["ep_overtake_success_blocked_safety_rate"] = float(
                self._episode_diag["steps_overtake_success_blocked_safety"] / diag_steps
            )
            info["ep_overtake_success_grant_count"] = float(
                self._episode_diag["overtake_success_grant_count"]
            )
            info["ep_offtrack_counter_max"] = float(self._episode_diag["offtrack_counter_max"])
            info["ep_stuck_counter_max"] = float(self._episode_diag["stuck_counter_max"])
            info["ep_bad_episode_guard_triggered"] = float(
                self._episode_diag["bad_episode_guard_triggered"]
            )
            info["ep_bad_episode_guard_step"] = float(
                self._episode_diag["bad_episode_guard_step"]
            )
            info["ep_terminal_offtrack_progress_discount"] = float(
                self._episode_diag["terminal_offtrack_progress_discount"]
            )
            info["ep_collision_episode_reward_cap_penalty"] = float(
                self._episode_diag["collision_episode_reward_cap_penalty"]
            )
            info["ep_offtrack_episode_reward_cap_penalty"] = float(
                self._episode_diag["offtrack_episode_reward_cap_penalty"]
            )
            info["ep_progress_ratio_signed_sum"] = float(self._episode_diag["progress_ratio_signed_sum"])
            info["ep_progress_ratio_forward_sum"] = float(self._episode_diag["progress_ratio_forward_sum"])
            info["ep_progress_reward_scale"] = float(self.progress_reward_scale)
            lane_pid_steps = int(self._episode_diag["lane_pid_debug_steps"])
            info["ep_lane_pid_debug_steps"] = float(lane_pid_steps)
            if lane_pid_steps > 0:
                lane_pid_steps_f = float(lane_pid_steps)
                info["ep_lane_pid_target_speed_mean"] = float(
                    self._episode_diag["lane_pid_target_speed_sum"] / lane_pid_steps_f
                )
                info["ep_lane_pid_speed_mean"] = float(
                    self._episode_diag["lane_pid_speed_sum"] / lane_pid_steps_f
                )
                info["ep_lane_pid_speed_error_abs_mean"] = float(
                    self._episode_diag["lane_pid_speed_error_abs_sum"] / lane_pid_steps_f
                )
                info["ep_lane_pid_effective_lookahead_mean"] = float(
                    self._episode_diag["lane_pid_effective_lookahead_sum"] / lane_pid_steps_f
                )
                info["ep_lane_pid_local_forward_mean"] = float(
                    self._episode_diag["lane_pid_local_forward_sum"] / lane_pid_steps_f
                )
                info["ep_lane_pid_local_left_abs_mean"] = float(
                    self._episode_diag["lane_pid_local_left_abs_sum"] / lane_pid_steps_f
                )
                info["ep_lane_pid_lat_err_norm_abs_mean"] = float(
                    self._episode_diag["lane_pid_lat_err_norm_abs_sum"] / lane_pid_steps_f
                )
                info["ep_lane_pid_steer_abs_mean"] = float(
                    self._episode_diag["lane_pid_steer_abs_sum"] / lane_pid_steps_f
                )
                info["ep_lane_pid_throttle_mean"] = float(
                    self._episode_diag["lane_pid_throttle_sum"] / lane_pid_steps_f
                )
                info["ep_lane_pid_reverse_rate"] = float(
                    self._episode_diag["lane_pid_reverse_steps"] / lane_pid_steps_f
                )
            else:
                info["ep_lane_pid_target_speed_mean"] = 0.0
                info["ep_lane_pid_speed_mean"] = 0.0
                info["ep_lane_pid_speed_error_abs_mean"] = 0.0
                info["ep_lane_pid_effective_lookahead_mean"] = 0.0
                info["ep_lane_pid_local_forward_mean"] = 0.0
                info["ep_lane_pid_local_left_abs_mean"] = 0.0
                info["ep_lane_pid_lat_err_norm_abs_mean"] = 0.0
                info["ep_lane_pid_steer_abs_mean"] = 0.0
                info["ep_lane_pid_throttle_mean"] = 0.0
                info["ep_lane_pid_reverse_rate"] = 0.0

        # 可选：前若干步诊断日志（默认关闭，避免训练日志污染）
        _diag_episode_hit = (
            self.step_diagnostics_every_episodes <= 0
            or (self._episode_index % self.step_diagnostics_every_episodes == 0)
        )
        if (
            self.enable_step_diagnostics
            and _diag_episode_hit
            and self.episode_stats["steps"] <= self.step_diagnostics_first_steps
        ):
            side_str = "L" if is_left_side else "R"
            reason_preview = (
                prev_reason
                if prev_reason
                else ("env_done" if env_done_before_processing else "normal")
            )
            print(
                f"🔍 [{self._logging_key}] ep={self._episode_index} step={self.episode_stats['steps']}: "
                f"lat_err_cte={lat_err_cte:.3f} side={side_str} "
                f"(in={cte_boundary:.2f}, out={cte_out_boundary:.2f}), "
                f"speed={speed:.2f}, hit={hit}, done={done}, "
                f"env_done={env_done_before_processing}, reason={reason_preview}"
            )

        info["reward_debug/native_env_done_cte_abs"] = float(native_env_cte_abs)
        info["reward_debug/native_env_done_max_cte"] = float(native_env_max_cte)
        info["reward_debug/native_env_done_likely_cte"] = float(native_env_done_likely_cte)
        info["reward_debug/native_env_done_likely_hit"] = float(native_env_done_likely_hit)
        info["reward_debug/native_env_done_handler_over"] = float(native_env_over)
        info["reward_debug/native_env_done_reason"] = str(native_env_done_reason)

        if term_reasons:
            dedup = []
            for r in term_reasons:
                if r and r not in dedup:
                    dedup.append(r)
            info["termination_reason"] = "+".join(dedup)
        else:
            if env_done_before_processing and (not env_done_masked):
                info.setdefault("termination_reason", "env_done")
            else:
                info.setdefault("termination_reason", "normal")
        if done:
            reason_tokens = set(str(info.get("termination_reason", "normal")).split("+"))
            info["ep_term_collision"] = float("collision" in reason_tokens)
            info["ep_term_stuck"] = float("stuck" in reason_tokens)
            info["ep_term_offtrack"] = float("offtrack" in reason_tokens)
            info["ep_term_bad_episode_guard"] = float("bad_episode_guard" in reason_tokens)
            info["ep_term_env_done"] = float("env_done" in reason_tokens)
            info["ep_term_normal"] = float(
                ("normal" in reason_tokens)
                and ("collision" not in reason_tokens)
                and ("stuck" not in reason_tokens)
                and ("offtrack" not in reason_tokens)
                and ("bad_episode_guard" not in reason_tokens)
                and ("env_done" not in reason_tokens)
            )
            info["ep_native_env_done_cte_abs"] = float(native_env_cte_abs)
            info["ep_native_env_done_max_cte"] = float(native_env_max_cte)
            info["ep_native_env_done_likely_cte"] = float(native_env_done_likely_cte)
            info["ep_native_env_done_likely_hit"] = float(native_env_done_likely_hit)
            info["ep_native_env_done_handler_over"] = float(native_env_over)
            info["ep_native_env_done_reason"] = str(native_env_done_reason)
        return obs, total_reward, done, info



# ---------------------------------------------------------------------------
# 向后兼容别名（供旧脚本 ppo_waveshare_v8/v9/test 等直接导入）
# ---------------------------------------------------------------------------
ImprovedRewardWrapperV3 = DonkeyRewardWrapper
V9DomainRewardWrapper   = DonkeyRewardWrapper
