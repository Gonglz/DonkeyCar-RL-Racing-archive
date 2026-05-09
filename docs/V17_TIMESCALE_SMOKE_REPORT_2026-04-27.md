# V17 Timescale Smoke Report - 2026-04-27

Purpose: compare DonkeySim timescale settings with the same V17 WS bootstrap smoke protocol.

Common protocol:
- Command family: `src/ppo_multitrack_v17.py --auto-curriculum --steps 1024`
- Sim2real JSON: `models/sim2real_phasef_20260427/phase_f_current_mid_throttle_fixed.json`
- Stage: `ws_bootstrap`
- `ppo_n_steps=1024`
- `adapter_v_nominal=0.58`
- `adapter_v_max=1.00`
- `learning_rate=1.5e-4`
- `throttle_gain_ratio=0.313185`
- `steer_gain_ratio=2.158203`
- `steer_tau_s=0.337859`
- `throttle_tau_s=0.129834`

## 10x Smoke

Note: this run was originally labelled as 1x during collection. The simulator was later confirmed by the operator to have been at 10x timescale.

Run directory:

`models/v17_timescale1_smoke_record_20260427_200828`

Result summary:

| Metric | Value |
| --- | ---: |
| requested steps | 1024 |
| completed steps | 1024 |
| elapsed | 535 s |
| effective fps | 1.9 steps/s |
| short episodes | 220 |
| short episode len mean | 4.69 |
| short episode len min/max | 4 / 6 |
| short episode reward mean | -14.46 |
| ws ep len mean | 4.71 |
| ws short episode rate | 1.00 |
| ws offtrack terminal rate | 1.00 |
| ws collision terminal rate | 0.29 |
| short offtrack count | 218 |
| short collision count | 62 |
| terminal CTE abs mean | 7.76 |
| terminal CTE abs max | 16.91 |
| ws speed mean | 0.232 |
| ws speed max mean | 0.535 |
| sim2real throttle mean | 0.0483 |
| adapter target speed mean | 0.547 |

Interpretation:

The 10x run is not behaviorally healthy under this current sim state. It completes as a process, but the vehicle terminates almost immediately.

## 1x Smoke

Run directory:

`models/v17_timescale1_smoke_record_20260427_202441_true1x`

Result summary:

| Metric | Value |
| --- | ---: |
| requested steps | 1024 |
| completed steps | 1024 |
| elapsed | 66 s |
| effective fps | 15.5 steps/s |
| short episodes | 0 |
| ws ep len mean | 102.22 |
| ws ep reward mean | -1.86 |
| ws short episode rate | 0.00 |
| ws offtrack terminal rate | 1.00 |
| ws collision terminal rate | 0.00 |
| ws CTE abs p50 | 0.243 |
| ws CTE abs p90 | 1.965 |
| ws CTE abs p99 | 3.103 |
| ws speed mean | 0.277 |
| ws speed max mean | 0.429 |
| sim2real throttle mean | 0.0570 |
| adapter target speed mean | 0.493 |

Interpretation:

The corrected 1x run is process-healthy and behaviorally much healthier than the 10x run. Episodes are still mostly offtrack-terminal in this early random-policy smoke, but they last about 102 steps instead of about 5, with no short episodes and no collision terminals in the recorded episode window.

## 5x Smoke

Run directory:

`models/v17_timescale5_smoke_record_20260427_203311`

Result summary:

| Metric | Value |
| --- | ---: |
| requested steps | 1024 |
| completed steps | 1024 |
| elapsed | 327 s |
| effective fps | 3.1 steps/s |
| short episodes | 112 |
| short episode len mean | 8.55 |
| short episode len min/max | 7 / 11 |
| short episode reward mean | -10.13 |
| ws ep len mean | 8.82 |
| ws ep reward mean | -10.08 |
| ws short episode rate | 0.98 |
| ws offtrack terminal rate | 0.99 |
| ws collision terminal rate | 0.16 |
| short offtrack count | 111 |
| short collision count | 17 |
| terminal CTE abs mean | 5.04 |
| terminal CTE abs max | 8.14 |
| ws CTE abs p50 | 0.135 |
| ws CTE abs p90 | 3.370 |
| ws CTE abs p99 | 4.665 |
| ws speed mean | 0.271 |
| ws speed max mean | 0.469 |
| sim2real throttle mean | 0.0479 |
| adapter target speed mean | 0.548 |

Interpretation:

The 5x run is better than 10x but still behaviorally unhealthy. Episode length collapses from about 102 steps at 1x to about 9 steps at 5x, and 98% of episodes are short. This timescale is not suitable for current training settings.

## 底层代码原因分析

结论：高倍 timescale 效果不好，不是 PPO 参数本身突然失效，而是当前控制链没有真正跟着模拟物理时间一起提频。timescale 提高后，DonkeySim 在两次 Python policy 决策之间前进得更远，但 V17 的 action adapter、sim2real 滤波、舵机保护、奖励终止窗口仍主要按 1x 的 step 语义工作，所以闭环控制来不及纠偏。

关键代码点：

1. `src/ppo_multitrack_v17.py` 里 `--sim-timescale` 只做了有限补偿。代码在 timescale 不等于 1 时保留 `controller_envelope=1x`，只把 `lidar_repeat_min_steps/max_steps` 除以 timescale，最低压到 `1-1`。也就是说，5x/10x 并没有把 policy 决策频率变成 5 倍或 10 倍，只是减少了 LiDAR 复用步数。

2. `module/action_adapter.py` 里转向是按 step 积分的：`steer_core += k_delta * delta_steer`；速度 PI 也用固定 `control_dt=0.05` 更新积分项。高 timescale 下一个 Python step 覆盖更多模拟物理时间，但这里仍按 1x 控制周期计算，等效上控制器变慢。

3. `module/sim2real_wrapper.py` 里一阶滞后固定用 `filter_dt_s=0.05` 算 `alpha = 1 - exp(-dt/tau)`。5x/10x 下模拟物理时间已经前进更多，但滤波器仍只前进 0.05 秒，导致 steer/throttle 执行动态相对模拟物理时间偏慢。

4. `module/control.py` 的 `ActionSafetyWrapper` 用 `delta_max` / `steer_delta_delta_max` 做每 step 限速，不是按模拟物理秒限速。高 timescale 下单位物理时间内可用的纠偏次数减少；如果反过来把 `delta_max` 放大，10x scaled run 已验证动作链会太激进，随机 PPO 动作直接把车打出赛道。

5. `module/lidar.py` 的 `SimAsyncLidarBuffer` 最低只能把 LiDAR repeat 降到 1。5x/10x 时即使每个 Python step 都刷新 LiDAR，两个观测之间车辆也已经移动了更多物理距离，传感器新鲜度无法再靠 repeat 修正。

6. `module/reward.py` 的 offtrack grace、near-offtrack ramp、stuck/bad-episode guard 都是按 step 计数。高 timescale 时 CTE 可能一两个 step 就跨过边界甚至严重出界，奖励中的预警区和恢复窗口来不及发挥作用，rollout 很快变成 `offtrack/collision` 短 episode。

因此，5x/10x 的核心问题是控制频率与模拟物理时间尺度不匹配。当前代码只是记录并部分补偿 timescale，没有在单个 env step 内做动作子步进，也没有把 adapter、sim2real 滤波、safety、reward 的时间窗口统一重标定。3x 勉强能跑，是因为失配还没有完全超过闭环容忍度；但 3x-only checkpoint 切回 1x 后仍明显退化，说明它学到的是 3x 节奏下的策略，不是最终可部署的 1x/real-aligned 策略。

## Pending

Add matching rows for:

- 15x timescale smoke
