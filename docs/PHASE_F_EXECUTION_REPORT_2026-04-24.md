# Phase F 执行报告（2026-04-24）

## 1. 本轮目标

按当前主线计划，先完成 `Phase F` 的 LiDAR sim2real 对齐门禁，再决定是否推进：

- `interaction-only LWM` 重构
- bootstrap 数据重导出
- 第一版 `interaction LWM` 训练
- `V17 PPO 300k` baseline

本轮执行重点是把 `Phase F` 从旧版“整体单指标”流程，落成新版：

- 近场优先分段门禁
- `motion-confound` 先判定
- motion 不通过时先做 `driver-profile sweep`
- 只有 motion 通过后才允许 `pose sweep`

## 2. 本轮已实现的代码改动

### 2.1 `Phase F` 评估脚本

已修改：

- [eval_lidar_domain_gap.py](/home/longzhao/mysim_public/scripts/eval_lidar_domain_gap.py:1)

当前已支持：

- `collect max_range = 20m`
- `eval compare_max_range = 12m`
- 分段输出：
  - `bands.overall_0_12m`
  - `bands.range_0_5m`
  - `bands.range_5_12m`

### 2.2 `Phase F` 正式执行器

已修改：

- [run_v17_formal_readiness.py](/home/longzhao/mysim_public/scripts/run_v17_formal_readiness.py:1)

当前执行逻辑已经变成：

1. `baseline motion precheck`
2. 如果 motion 不通过，进入 `driver-profile sweep`
3. 只有选出的 driver 通过 motion gate，才进入 full baseline eval
4. 如果 primary gate 不过，再进入 `pose sweep`

另外加入了执行层优化：

- `motion_precheck_frames`
- `pose_sweep_precheck_frames`
- `pose_sweep_top_k`

也就是先用短采样判断，再决定是否进入 full validation。

### 2.3 `Phase F` manifest

已修改：

- [v17_formal_readiness_manifest.json](/home/longzhao/mysim_public/scripts/v17_formal_readiness_manifest.json:1)

当前关键参数已固定为：

- collect:
  - `offset_y = 0.40`
  - `offset_z = 0.50`
  - `rot_x = 0.0`
  - `lidar_max_range_m = 20.0`
  - `near_clip = 0.18`
- eval:
  - `compare_max_range_m = 12.0`
  - `range_bands = ["0,5", "5,max"]`
- real:
  - `0421` 作为 `primary`
  - `0422_1` 作为 `stress`
- precheck:
  - `motion_precheck_frames = 200`
  - `pose_sweep_precheck_frames = 200`
  - `pose_sweep_top_k = 3`

### 2.4 文档同步

已同步到：

- [LWM_PPO_COTRAINING_2026-04-22.md](/home/longzhao/mysim_public/docs/LWM_PPO_COTRAINING_2026-04-22.md:1)
- [LIDAR_DEBUG_RECORD_2026-04-24.md](/home/longzhao/mysim_public/docs/LIDAR_DEBUG_RECORD_2026-04-24.md:1)

## 3. 正式执行结果

正式运行目录：

- [/tmp/v17_phasef_20260424_run4/readiness_report.json](/tmp/v17_phasef_20260424_run4/readiness_report.json:1)

本轮实际执行到了：

- `phase_a`: 通过
- `phase_f`: 失败

失败原因不是 `pose`，而是更前面的：

- `driver-profile mismatch unresolved`

也就是说，本轮在 `motion-confound -> driver-profile sweep` 这一层就已经停住，**没有进入 pose sweep**。

## 4. Motion 结果

### 4.1 baseline

`baseline policy = best_model_ws.zip`

结果：

- `speed_proxy_p50_ratio ≈ 10.69`
- `speed_proxy_p95_ratio ≈ 6.15`
- `abs_final_angle_p95_gap ≈ 0.537`
- `motion_pass = false`

对应文件：

- [motion_match_report_baseline.json](/tmp/v17_phasef_20260424_run4/phase_f_deployment/motion_match_report_baseline.json:1)

### 4.2 driver-profile sweep

执行的候选：

- `best_model_ws.zip`
- `v16_200000_steps.zip`
- `v16_240000_steps.zip`
- `v16_280000_steps.zip`

汇总文件：

- [driver_profile_sweep_summary.json](/tmp/v17_phasef_20260424_run4/phase_f_deployment/driver_profile_sweep_summary.json:1)

结果如下：

1. `best_model_ws.zip`
- `motion_pass = false`
- `motion_score ≈ [2.369, 1.816, 0.537]`

2. `v16_200000_steps.zip`
- `motion_pass = false`
- `motion_score ≈ [2.332, 1.839, 0.427]`

3. `v16_240000_steps.zip`
- `motion_pass = false`
- `motion_score ≈ [2.363, 1.849, 0.565]`

4. `v16_280000_steps.zip`
- `motion_pass = false`
- `motion_score ≈ [2.366, 1.926, 0.634]`

其中相对最好的候选是：

- `v16_200000_steps.zip`

但它仍然远远不满足当前 motion gate：

- `speed_ratio` 目标区间是 `[0.80, 1.25]`
- 实际仍然在 `10x / 6x` 量级

## 5. 当前结论

### 5.1 已确认的事实

- 当前新版 `Phase F` 工具链已经可用。
- 分段 eval、motion gate、driver-profile sweep 都已经实际跑通。
- 当前冻结 LiDAR pose 还没有机会进入 `pose sweep`，因为更早的 motion gate 已经失败。
- 失败主因不是 LiDAR 几何本身先暴露出来，而是：
  - **sim 轨迹剖面和 `0421` real 正常集差异太大**

### 5.2 当前主线状态

按计划，`Phase F` 不通过就不能推进：

- `interaction-only LWM`
- bootstrap 重导出
- 第一版 `interaction LWM`
- `V17 PPO 300k`

因此当前主线仍然**停在 Gate B**。

## 6. 我已经自己解决掉的卡点

本轮不是只做文档改动，执行中我还处理了两个实际卡点：

### 6.1 执行时间过长

原来 `motion precheck = 400` 太慢，正式执行会在确认明显 motion mismatch 前浪费很多时间。

已调整为：

- `motion_precheck_frames = 200`
- `pose_sweep_precheck_frames = 200`

### 6.2 driver candidate 路径错误

原 manifest 里有不存在的 checkpoint：

- `v16_400000_steps.zip`
- `v16_800000_steps.zip`

执行时实际报成：

- `...v16_400000_steps.zip.zip`

已修成真实存在的 checkpoint：

- `v16_200000_steps.zip`
- `v16_240000_steps.zip`
- `v16_280000_steps.zip`

## 7. 现在的真正阻塞点

当前不是工具链阻塞，也不是执行器死锁。

当前阻塞点很明确：

- **在现有这组 `V16` warmup driver profile` 下，sim 的运动剖面无法匹配 `0421` real 正常集**

这意味着下一步如果继续推进，优先级应该是：

1. 扩大 `driver-profile` 候选集合
2. 或者重新定义一个更接近 `0421` 的 sim driving profile

而不是现在就去做：

- `pose sweep`
- `interaction-only LWM`
- `PPO baseline`

## 8. 建议的审核点

你审核时重点看这三件事就够了：

1. 是否接受当前 Gate B 的停机结论：
   - `driver-profile mismatch unresolved`
2. 是否接受当前 `motion gate` 标准继续保持不变：
   - `speed_proxy_p50_ratio/p95_ratio`
   - `abs_final_angle_p95_gap`
3. 下一轮是否允许我扩大 `driver-profile` 搜索范围：
   - 不再局限于 `v16_pid_overtake_course_20260420` 这一个目录

## 9. 本轮最终判断

本轮已经按计划推进到可判定状态，且遇到的中途问题都已自行解决：

- 工具链已落地
- 正式 Phase F 已执行
- motion gate 与 driver sweep 已跑完
- 当前主线阻塞原因已明确

所以这轮可以交你审核，不需要我继续盲目往下跑。
