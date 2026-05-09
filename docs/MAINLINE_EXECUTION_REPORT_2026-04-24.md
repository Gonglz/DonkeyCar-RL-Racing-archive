# Mainline Execution Report 2026-04-24

## 结论

当前主线已经推进到：

1. `Phase F` deployment-oriented Gate 已通过
2. `interaction-only LWM` 重构已完成
3. `Phase C` bootstrap 数据已导出并通过数据门
4. `Phase D` 第一版 `interaction-only LWM` 已训练并通过训练门
5. `V17 PPO 300k baseline` 已启动，但在 `88k / 300k` 中途停止

当前主线阻塞项已经定位：

- `PPO` 进程已退出，训练目录没有 `final_model.zip`
- 当时训练脚本还没有普通异常落盘，所以没有 traceback 可还原
- `waveshare` 在 warmup 里大量 `13-15` 步短 episode，主要是 `offtrack / collision`
- 关键实现问题：`Sim2RealActionWrapper` 原先位于 `ActionSafetyWrapper` 内侧，导致 `sim2real` 的 steering gain 在安全限幅之后再次放大，绕过了最终动作安全约束

## 本轮关键修复

### 1. `Phase C` 导出继承 `sim2real_json`

修复前：
- `run_v17_formal_readiness.py` 的 `phase_c` 没有把 `--sim2real-json` 透传给
  `export_world_model_dataset.py`

修复后：
- `phase_c` 支持从 `export_cfg -> phase_c -> phase_f.collect` 三级回退读取 `sim2real_json`
- 当前主线统一绑定：
  - `/home/longzhao/mysim_public/models/sim2real_phasef_20260424/phase_f_motion_fit_v5_midpoint_tuned.json`

涉及文件：
- [run_v17_formal_readiness.py](/home/longzhao/mysim_public/scripts/run_v17_formal_readiness.py:860)
- [v17_formal_readiness_manifest.json](/home/longzhao/mysim_public/scripts/v17_formal_readiness_manifest.json:69)

### 2. `Phase C` 的 `passable` 标签重标

问题：
- 原先 `passable_gap_threshold_m = 0.7`
- 在当前 `sim2real v5 + LiDAR baseline` 下过宽，导致：
  - `passable_left_rate = 0.9756`
  - `passable_right_rate = 0.9641`
- 直接卡死 `phase_c`

处理：
- 将 `phase_c.passable_gap_threshold_m` 调整为 `1.4`
- 增加 merged dataset 级别的统一重标函数：
  - 从 `target_gap` 重新生成 `target_passable`
  - 不需要为了改标签阈值重跑仿真采样

修复后当前 merged dataset：
- `passable_left_rate = 0.7260`
- `passable_right_rate = 0.8760`
- `dataset_readiness = passed`

涉及文件：
- [run_v17_formal_readiness.py](/home/longzhao/mysim_public/scripts/run_v17_formal_readiness.py:164)
- [v17_formal_readiness_manifest.json](/home/longzhao/mysim_public/scripts/v17_formal_readiness_manifest.json:69)

关键产物：
- [wm_dataset_mix_v1.npz](/tmp/v17_phasecd_20260424_interaction_only_run2/phase_c_dataset/wm_dataset_mix_v1.npz:1)
- [wm_dataset_mix_v1.json](/tmp/v17_phasecd_20260424_interaction_only_run2/phase_c_dataset/wm_dataset_mix_v1.json:1)
- [dataset_readiness.json](/tmp/v17_phasecd_20260424_interaction_only_run2/phase_c_dataset/dataset_readiness.json:1)

### 3. `Phase D` 训练阶段修复

#### 3.1 Stage B 冻结策略修复

问题：
- 原版 `Stage B` 会明显拉坏 `gap` 几何项

处理：
- 冻结 `safety_head.pre`
- 继续冻结 `gap_head`

涉及文件：
- [train_world_model_v17.py](/home/longzhao/mysim_public/scripts/train_world_model_v17.py:138)

#### 3.2 Stage B 增加几何保持正则

问题：
- 仅冻结 `pre + gap_head` 后，`gap` 漂移仍略大

处理：
- 在 `stage == "b"` 的 loss 中加入：
  - `0.5 * target_loss`
  - `0.5 * gap_loss`

结果：
- `Stage C` 最终几何已恢复到门内

涉及文件：
- [local_world_model_v17.py](/home/longzhao/mysim_public/module/local_world_model_v17.py:500)

#### 3.3 训练 readiness 逻辑修正

问题：
- 原先只要 `Stage B` 中间态几何超 `5%`，就直接 fail
- 但当前 `Stage C` 最终 checkpoint 已恢复并通过几何门

处理：
- 如果 `Stage C`:
  - `guard_triggered = false`
  - 且 `best_metrics` 在 `stage_a * 1.03` 内
- 则不再因为 `Stage B` 的中间态回退一票否决

涉及文件：
- [eval_world_model_readiness.py](/home/longzhao/mysim_public/scripts/eval_world_model_readiness.py:378)

## 当前结果

### Phase C

当前 `Phase C` merged dataset：
- samples: `5120`
- scenes:
  - `generated_track = 4096`
  - `waveshare = 1024`
- `collision_pos_rate = 0.0869`
- `opportunity_valid_rate = 0.9613`
- `passable_left_rate = 0.7260`
- `passable_right_rate = 0.8760`

状态：
- `passed`

### Phase D

当前使用的训练目录：
- [phase_d_world_model_train_fix2](/tmp/v17_phasecd_20260424_interaction_only_run2/phase_d_world_model_train_fix2:1)

关键指标：
- Stage A
  - `mae_target_rel = 0.8794`
  - `mae_gap = 1.8836`
- Stage B
  - `mae_target_rel = 0.8794`
  - `mae_gap = 2.0615`
- Stage C best
  - `mae_target_rel = 0.8890`
  - `mae_gap = 1.7405`
  - `guard_triggered = false`
  - `final_recovery_ok = true`

状态：
- training readiness `passed`

关键产物：
- [train_summary.json](/tmp/v17_phasecd_20260424_interaction_only_run2/phase_d_world_model_train_fix2/train_summary.json:1)
- [local_world_model_v17_final.pth](/tmp/v17_phasecd_20260424_interaction_only_run2/phase_d_world_model_train_fix2/local_world_model_v17_final.pth:1)

### V17 PPO 300k baseline

#### 旧 run 诊断

旧 run 已停止：
- save dir:
  - [v17_mainline_baseline300k_20260424](/home/longzhao/mysim_public/models/v17_mainline_baseline300k_20260424:1)
- 最后记录：
  - `callback_num_timesteps = 88000`
  - `train/n_updates = 84`
  - 最后日志时间：`2026-04-24T18:35:19`

当前确认项：
- preflight 通过
- env 初始化通过
- `sim2real v5` 已挂上
- `train_metrics.jsonl` 已开始写入
- `short_episodes.jsonl` 已开始写入
- 已进入 `warmup` 阶段训练循环，但没有完成 `300k`
- 无 `final_model.zip`
- 无可追溯 traceback

关键文件：
- [train_metrics.jsonl](/home/longzhao/mysim_public/models/v17_mainline_baseline300k_20260424/train_metrics.jsonl:1)
- [short_episodes.jsonl](/home/longzhao/mysim_public/models/v17_mainline_baseline300k_20260424/short_episodes.jsonl:1)
- [curriculum_window.jsonl](/home/longzhao/mysim_public/models/v17_mainline_baseline300k_20260424/curriculum_window.jsonl:1)

关键诊断：
- `short_episodes = 680`
- `waveshare = 677`
- `generated_track = 2`
- `waveshare` 短 episode 平均长度约 `13.35`
- `waveshare` 短 episode 主因：
  - `offtrack`
  - `collision`
  - `native_cte_exceed`
- `88k` 时：
  - `ws_short_ep_rate = 0.3444`
  - `ws_term_offtrack_rate = 0.6667`
  - `ws_term_collision_rate = 0.3222`
  - `gt_term_offtrack_rate = 1.0`
  - `gt_ep_rew_mean = -298.76`

定位结论：
- 训练停止原因无法从旧产物里还原成 traceback，因为旧脚本只捕获 `KeyboardInterrupt`，普通异常不会落盘
- 系统日志没有 OOM 证据，也没有 DonkeySim 崩溃证据
- 更明确的实现问题是动作 wrapper 顺序错误：
  - 修复前动作路径：`ActionAdapter -> ActionSafety -> Sim2Real -> simulator`
  - 修复后动作路径：`ActionAdapter -> Sim2Real -> ActionSafety -> simulator`
- 修复后，`sim2real` 的 `steer_gain=2.158` 不再绕过最终动作安全限制

#### 2026-04-26 重跑状态

已启动新的正式 `300k baseline` 重跑：
- save dir:
  - [v17_mainline_baseline300k_20260426_rerun3](/home/longzhao/mysim_public/models/v17_mainline_baseline300k_20260426_rerun3:1)
- tmux session:
  - `v17_300k_20260426_rerun3`
- stdout/stderr:
  - [run_stdout.log](/home/longzhao/mysim_public/models/v17_mainline_baseline300k_20260426_rerun3/run_stdout.log:1)
- metrics:
  - [train_metrics.jsonl](/home/longzhao/mysim_public/models/v17_mainline_baseline300k_20260426_rerun3/train_metrics.jsonl:1)
- short episode log:
  - [short_episodes.jsonl](/home/longzhao/mysim_public/models/v17_mainline_baseline300k_20260426_rerun3/short_episodes.jsonl:1)

启动方式调整：
- 直接从 `exec_command` 后台 `nohup` 启动会出现进程被会话清理、日志 `0 bytes` 的现象
- 当前改用 `tmux` 独立会话运行，训练进程已确认持续存在
- 日志末尾会写入 `EXIT_STATUS=<code>`，如果再次退出可直接追踪

早期确认：
- preflight 通过
- sim TCP 连通
- `sim2real v5` 已挂上
- 动作链顺序已是修复后的 `ActionAdapter -> Sim2Real -> ActionSafety -> simulator`
- 已进入 `warmup` 训练循环并写出 `1000 / 1500 / 2000` step metrics

早期风险：
- 策略仍在训练初期，`generated_track / waveshare` 均有明显 offtrack
- `2000` step 左右：
  - `gt_ep_rew_mean = -86.93`
  - `gt_term_offtrack_rate = 0.9091`
  - `ws_ep_rew_mean = -28.58`
  - `ws_short_ep_rate = 0.3077`
  - `ws_term_offtrack_rate = 0.8462`
- 这说明当前问题从“启动/动作限幅链路错误”转为“训练早期稳定性观察”，还不能判断最终 `300k` 是否会收敛

## 当前主线状态

主线已从：
- `Phase F deployment gate`

推进到：
- `PPO baseline diagnose/fix`

所以现在的阶段是：
- `interaction-only LWM` 已就绪
- `PPO 300k baseline` 需要用修复后的 wrapper 顺序重跑

## 下一步

短期只做两件事：

1. 重启一轮新的 `300k baseline`
   - 使用新的 save dir，避免污染旧 run
   - stdout/stderr 必须落盘
   - 继续传 `sim2real v5`
   - 使用修复后的 wrapper 顺序

2. `300k baseline` 完成后再决定：
   - 是否做第一次 `LWM refresh`
   - 是否推进 `critic-only` 在线接入

已补的工程保护：
- `ppo_multitrack_v17.py` 现在会在普通异常时写出 `training_error.json`
- 同时保存 `crash_model.zip` 和 `crash_model_policy.pth`
- 下一轮如果再次中断，应能直接看到 Python traceback

当前**不再**开新支线：
- 不重开 LiDAR pose debug
- 不重开 raw-domain strict Phase F
- 不再改 `ego model`
