# V17 Timescale 与大规模训练决策报告 - 2026-04-27/28

本文记录 V17 在 1x、3x、5x、10x 模拟器 timescale 下的训练表现、回到 1x 后的策略退化情况，以及当前大规模训练的选择。

结论先写在前面：

- 大规模训练当前选择 1x。
- 3x 可以训练，但 3x-only 的 checkpoint 回到 1x 后明显退化，不能直接作为可部署策略。
- 5x 和 10x 在当前 Python 控制频率下不可用，短 episode 会快速爆炸。
- 后续所有 Markdown 报告默认使用中文。

## 标定与配置

1x 对齐 JSON：

- `models/sim2real_phasef_20260427/phase_f_timescale_1x_aligned_v1.json`
- 便捷链接：`models/sim2real_phasef_20260427/phase_f_current_1x_aligned.json`

关键参数：

| 参数 | 数值 |
| --- | ---: |
| `filter_dt_s` | 0.05 |
| `throttle_gain_ratio` | 0.313185 |
| `steer_gain_ratio` | 2.158203 |
| `steer_tau_s` | 0.337859 |
| `throttle_tau_s` | 0.129834 |
| `adapter_k_delta` | 0.08 |
| `adapter_k_bias` | 0.08 |
| `delta_max` | 0.20 |
| `steer_delta_delta_max` | 0.04 |
| `beta` | 0.50 |

这套配置的原则是：速度上下限跟 real 对齐，转向动力学更严格，重点限制舵机高频跳变。

## 1x 基准训练

5k smoke：

- Run：`models/v17_timescale1_aligned_smoke5k_20260427_210027`

| 指标 | 数值 |
| --- | ---: |
| 训练步数 | 5120 |
| 耗时 | 0.10 h |
| fps | 14.7 |
| `ws_ep_len_mean` | 103.90 |
| `ws_ep_rew_mean` | -5.96 |
| `ws_short_ep_rate` | 0.0208 |
| `ws_term_collision_rate` | 0.00 |
| `ws_term_stuck_rate` | 0.00 |
| `speed_mean` | 0.266 |
| `speed_max_mean` | 0.419 |
| `sim2real_throttle_mean` | 0.0553 |
| `sim2real_steer_abs_mean` | 0.477 |

50k baseline：

- Run：`models/v17_timescale1_aligned_train50k_20260427_210705`

| 指标 | 数值 |
| --- | ---: |
| 目标步数 | 50000 |
| 实际步数 | 50176 |
| 耗时 | 0.94 h |
| best checkpoint | `best_model_ws.zip` |
| best step | 44000 |
| best `ws_ep_rew_mean` | -6.16 |
| best `ws_ep_len_mean` | 121.04 |
| final `ws_ep_rew_mean` | -10.15 |
| final `ws_ep_len_mean` | 115.27 |
| final `ws_short_ep_rate` | 0.00 |
| final `ws_term_collision_rate` | 0.01 |
| final `ws_term_stuck_rate` | 0.00 |
| final `speed_mean` | 0.228 |
| final `speed_max_mean` | 0.332 |
| final `explained_variance` | 0.773 |

判断：1x 是当前唯一经过回测确认没有动力学退化的主训练基准。

## 10x 测试

10x 做了两次：

| Run | 步数 | 短 episode | 平均短 episode 长度 | 终止原因 | 结果 |
| --- | ---: | ---: | ---: | --- | --- |
| `models/v17_timescale10_scaled_train50k_20260427_221919` | 61 | 13 | 4.46 | 10 offtrack, 3 collision+offtrack | 失败 |
| `models/v17_timescale10_control1x_smoke1k_20260427_222221` | 70 | 13 | 5.08 | 10 offtrack, 3 collision+offtrack | 失败 |

第一次 10x 把控制包络也放大了，例如 `delta_max=1.0`、`adapter_k_delta=0.8`、`beta=0.999`，结果动作链太激进，随机 PPO 动作直接把车打出赛道。

第二次 10x 改为保留 1x 控制包络，只降低 LiDAR repeat，但仍然很快失败。当前判断是：10x 下模拟器在两次 Python policy 决策之间前进太远，agent 来不及形成有效闭环。

结论：10x 目前不可用于训练。

## 5x 测试

- JSON：`models/sim2real_phasef_20260427/phase_f_timescale_5x_control1x_v1.json`
- Run：`models/v17_timescale5_control1x_smoke1k_20260427_222935`

| 指标 | 数值 |
| --- | ---: |
| 目标步数 | 1024 |
| 实际停止步数 | 102 |
| 短 episode | 11 |
| 平均短 episode 长度 | 9.00 |
| 短 episode 长度范围 | 8 / 10 |
| offtrack 短 episode | 10 |
| collision+offtrack 短 episode | 1 |
| terminal CTE abs mean | 4.80 |
| terminal CTE abs max | 5.61 |

结论：5x 比 10x 好，但仍然不可训练。

## 3x Smoke

3x 使用保守配置：

- JSON：`models/sim2real_phasef_20260427/phase_f_timescale_3x_control1x_v1.json`
- 便捷链接：`models/sim2real_phasef_20260427/phase_f_current_3x_control1x.json`
- 1k gate：`models/v17_timescale3_control1x_smoke1k_20260427_223225`
- 5k validation：`models/v17_timescale3_control1x_smoke5k_20260427_223531`

| 指标 | 1k gate | 5k validation |
| --- | ---: | ---: |
| 目标步数 | 1024 | 5000 |
| 实际步数 | 1024 | 5120 |
| early stop | 否 | 否 |
| `ws_ep_len_mean` | 28.88 | 30.47 |
| 折算 1x 物理步数 | 86.6 | 91.4 |
| `ws_ep_rew_mean` | -7.45 | -6.65 |
| `ws_short_ep_rate` | 0.00 | 0.01 |
| 短 episode | 0 | 1 |
| `ws_term_collision_rate` | 0.0588 | 0.01 |
| `ws_term_offtrack_rate` | 1.00 | 0.99 |
| `ws_term_stuck_rate` | 0.00 | 0.00 |
| `speed_mean` | 0.279 | 0.284 |
| `speed_max_mean` | 0.417 | 0.419 |
| `sim2real_throttle_mean` | 0.0466 | 0.0469 |
| `sim2real_steer_abs_mean` | 0.255 | 0.253 |
| `safety_delta_steer_abs_mean` | 0.0203 | 0.0203 |
| `safety_delta_delta_limit_hit_rate` | 0.0012 | 0.0025 |
| `servo_deadband_hold_rate` | 0.194 | 0.182 |
| final `explained_variance` | n/a | 0.464 |

结论：3x 是目前唯一能跑通的加速 timescale，但它不是 3 倍 wall-clock 加速。1x 约 14.7 fps，3x 约 8 fps，折算物理时间吞吐约 `3 * 8 / 14.7 ~= 1.6x`。

## 3x 50k 训练与 1x 回测

3x 50k：

- Run：`models/v17_timescale3_control1x_train50k_20260427_230008`

| 指标 | 数值 |
| --- | ---: |
| 目标步数 | 50000 |
| 实际步数 | 50176 |
| 耗时 | 1.76 h |
| 短 episode | 18 |
| rollback | 0 |
| best checkpoint | `best_model_ws.zip` |
| best step | 49000 |
| best `ws_ep_rew_mean` | -4.43 |
| best `ws_ep_len_mean` | 39.09 |
| final `ws_ep_rew_mean` | -4.82 |
| final `ws_ep_len_mean` | 39.98 |
| final `ws_short_ep_rate` | 0.00 |
| final `ws_term_collision_rate` | 0.03 |
| final `ws_term_offtrack_rate` | 0.98 |
| final `ws_term_stuck_rate` | 0.00 |

切回 1x 后做严格评估：

- Eval JSON：`models/v17_timescale3_control1x_train50k_20260427_230008/eval_1x_compare_baseline_vs_3x_best_20260428_1.json`
- 每个模型 30 个 deterministic episode
- strict offtrack
- 使用 1x aligned sim2real JSON

| 指标 | 1x baseline best | 3x train best |
| --- | ---: | ---: |
| reward mean | -10.43 | -22.95 |
| reward std | 0.74 | 0.45 |
| length mean | 108.67 | 94.43 |
| length min/max | 107 / 110 | 92 / 97 |
| short episode rate | 0.00 | 0.00 |
| collision rate | 0.00 | 0.00 |
| offtrack rate | 1.00 | 1.00 |
| stuck rate | 0.00 | 0.00 |
| speed mean | 0.237 | 0.164 |
| speed max mean | 0.320 | 0.220 |
| sim2real throttle mean | 0.0435 | 0.0354 |
| sim2real steer abs mean | 0.588 | 0.878 |
| safety delta steer abs mean | 0.0092 | 0.0106 |
| safety delta-delta limit hit rate | 0.0000 | 0.0063 |
| servo deadband hold rate | 0.280 | 0.798 |

结论：3x-only 模型切回 1x 后明显退化。它没有崩成短 episode，也没有 stuck，但速度更慢、转向更大、舵机保护触发更多、reward 明显更差。

## 大规模训练决策

当前大规模训练选择 1x，原因如下：

| 对比项 | 1x 50k | 3x 50k |
| --- | ---: | ---: |
| 训练耗时 | 0.94 h | 1.76 h |
| PPO steps/s | 约 14.7 | 约 7.9 |
| 同样 50k wall-clock | 更快 | 更慢 |
| 1x 回测 reward | -10.43 | -22.95 |
| 1x 回测 length | 108.67 | 94.43 |
| 1x 回测 speed mean | 0.237 | 0.164 |
| 1x 回测 steer abs | 0.588 | 0.878 |

3x 的唯一优势是物理时间覆盖量更大，但这个优势没有转化成 1x 可用策略。对于最终要对齐 real 的策略，1x 更可靠。

## 当前全量训练

已经启动 1x 全量 auto curriculum：

- tmux session：`v17_full_1x_20260428`
- PID：`4013846`
- Run：`models/v17_1x_full_autocurriculum_from1xbest_20260428_010300`
- 日志：`models/v17_1x_full_autocurriculum_from1xbest_20260428_010300/train.log`
- 总步数：`2,000,000`
- 起点 checkpoint：`models/v17_timescale1_aligned_train50k_20260427_210705/best_model_ws.zip`
- sim2real JSON：`models/sim2real_phasef_20260427/phase_f_current_1x_aligned.json`
- sim timescale：`1`
- 第一阶段：`ws_bootstrap`
- 第一阶段预算：`300,000`

启动时已确认：

- `filter_dt=0.050s`
- `adapter_k_delta=0.08`
- `adapter_k_bias=0.08`
- `adapter_v_nominal=0.58`
- `adapter_v_max=1.0`
- `delta_max=0.20`
- `steer_delta_delta_max=0.04`
- `beta=0.50`
- `lidar_repeat=2-4`

## 操作建议

1. 大规模训练继续用 1x。
2. 3x 只作为可选 warm start 实验，不作为最终训练主线。
3. 如果以后继续探索 3x，必须做 `3x 预训练 -> 1x finetune -> 1x eval gate`，不能直接使用 3x checkpoint。
4. 5x/10x 暂停，除非改成更高频控制闭环或在单个 env step 内做动作子步进。
5. 所有可部署候选模型都必须在 1x 严格环境上验收。
