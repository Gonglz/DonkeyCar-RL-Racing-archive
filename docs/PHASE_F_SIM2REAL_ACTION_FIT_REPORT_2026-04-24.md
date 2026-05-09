# Phase F Sim2Real Action Fit Report (2026-04-24)

## 背景

本轮目标不是继续盲扫 LiDAR pose，而是先验证：

- 能否基于 `0421` real 正常集和当前 sim monitor log
- 拟合一个第一版 `sim2real_json`
- 接入 `Phase F collect`
- 重新跑 Gate B，观察 `motion-confound` 是否能被解除

当前 Gate B 的阻塞点此前已经确认是：

- 不是 LiDAR packet 解释
- 不是 LiDAR pose 首先失配
- 而是 `sim driver profile` 和 `0421` 的 motion profile 差异过大

## 本轮实际改动

### 1. 新增第一版 sim2real 拟合脚本

新增：

- [fit_phase_f_sim2real_json.py](/home/longzhao/mysim_public/scripts/fit_phase_f_sim2real_json.py:1)

功能：

- 输入：
  - `0421` real csv
  - 一份 sim monitor jsonl
- 输出兼容 [sim2real_wrapper.py](/home/longzhao/mysim_public/module/sim2real_wrapper.py:1) 的低维动作校正 JSON：
  - `throttle_gain_ratio`
  - `steer_gain_ratio`
  - `steer_tau_s`
  - `throttle_tau_s`

拟合原则：

- `throttle_gain_ratio`
  - 基于 `speed_proxy_p50 / p95` 的几何平均缩放
- `steer_gain_ratio`
  - 基于 `abs(final_angle)_p95` 的幅值比
- `steer_tau_s / throttle_tau_s`
  - 基于命令变化率 `p95` 的对数型一阶滞后估计

### 2. 把 sim2real_json 接进 Phase F collect

改动：

- [export_world_model_dataset.py](/home/longzhao/mysim_public/scripts/export_world_model_dataset.py:327)
  - `_make_env(...)` 增加 `sim2real_json`
- [collect_sim_lidar_monitor.py](/home/longzhao/mysim_public/scripts/collect_sim_lidar_monitor.py:1)
  - 新增 `--sim2real-json`
  - 通过 wrapper 链查找 `Sim2RealActionWrapper`
  - 日志里 `final_angle / final_throttle` 改成优先记录 **sim2real 后的实际送入动作**
  - 同时保留：
    - `pre_sim2real_final_angle`
    - `pre_sim2real_final_throttle`
    - `sim2real_applied`
- [sim2real_wrapper.py](/home/longzhao/mysim_public/module/sim2real_wrapper.py:1)
  - 记录：
    - `last_raw_action`
    - `last_transformed_action`
- [run_v17_formal_readiness.py](/home/longzhao/mysim_public/scripts/run_v17_formal_readiness.py:1020)
  - `_phase_f_collect_run(...)` 读取 manifest 里的 `sim2real_json`
  - 自动传给 `collect_sim_lidar_monitor.py`
- [v17_formal_readiness_manifest.json](/home/longzhao/mysim_public/scripts/v17_formal_readiness_manifest.json:190)
  - `phase_f.collect.sim2real_json` 已接入

## 拟合产物

### 第一版

- [phase_f_motion_fit_v1.json](/home/longzhao/mysim_public/models/sim2real_phasef_20260424/phase_f_motion_fit_v1.json:1)

来源：

- real:
  - `/home/longzhao/mysim/data/data_lidar/0421/monitor_logs/run_20260421_005659.csv`
- sim:
  - `/tmp/v17_phasef_20260424_run4/phase_f_deployment/baseline_motion_precheck/monitor_logs/run_20260424_052638_sim_donkey-waveshare-v0_lidar_raw.jsonl`

核心参数：

- `throttle_gain_ratio = 0.12335`
- `steer_gain_ratio = 2.15820`
- `steer_tau_s = 0.17273`
- `throttle_tau_s = 0.12766`

### 第二版

- [phase_f_motion_fit_v2.json](/home/longzhao/mysim_public/models/sim2real_phasef_20260424/phase_f_motion_fit_v2.json:1)

来源：

- 以 `v1` 为 base
- 用 `run5` 中最优候选的 sim log 做一次 compose refit

核心参数：

- `throttle_gain_ratio = 0.05032`
- `steer_gain_ratio = 2.15820`
- `steer_tau_s = 0.33786`
- `throttle_tau_s = 0.12766`

## Gate B 重跑结果

### Run 5: 第一版 sim2real_json

结果目录：

- [/tmp/v17_phasef_20260424_run5/readiness_report.json](/tmp/v17_phasef_20260424_run5/readiness_report.json:1)

关键结论：

- `abs_final_angle_p95_gap` 被直接拉到 `0.0`
- 但速度仍明显偏快
- Gate B 仍卡在 `driver-profile mismatch unresolved`
- 仍未进入 pose sweep

baseline 结果：

- [motion_match_report_baseline.json](/tmp/v17_phasef_20260424_run5/phase_f_deployment/motion_match_report_baseline.json:1)

指标变化：

- 旧 baseline（未加 sim2real）：
  - `speed_proxy_p50_ratio ≈ 10.69`
  - `speed_proxy_p95_ratio ≈ 6.15`
  - `abs_final_angle_p95_gap ≈ 0.537`
- `v1` baseline：
  - `speed_proxy_p50_ratio ≈ 3.80`
  - `speed_proxy_p95_ratio ≈ 1.58`
  - `abs_final_angle_p95_gap = 0.0`

解释：

- 这说明 `v1` 已经把**转向幅值**对齐问题解决了
- 也显著压低了速度
- 但还没把速度压进 gate 区间

### Run 6: 第二版 sim2real_json

结果目录：

- [/tmp/v17_phasef_20260424_run6/readiness_report.json](/tmp/v17_phasef_20260424_run6/readiness_report.json:1)

关键结论：

- `abs_final_angle_p95_gap` 继续保持 `0.0`
- `speed_proxy_p50_ratio` 继续下降
- 但 `speed_proxy_p95_ratio` 掉到了 `< 0.8`
- Gate B 仍卡在 `driver-profile mismatch unresolved`
- 仍未进入 pose sweep

baseline 结果：

- [motion_match_report_baseline.json](/tmp/v17_phasef_20260424_run6/phase_f_deployment/motion_match_report_baseline.json:1)

baseline 指标：

- `speed_proxy_p50_ratio ≈ 2.00`
- `speed_proxy_p95_ratio ≈ 0.659`
- `abs_final_angle_p95_gap = 0.0`

driver sweep 摘要：

- [driver_profile_sweep_summary.json](/tmp/v17_phasef_20260424_run6/phase_f_deployment/driver_profile_sweep_summary.json:1)

四个候选都未通过 motion gate。

其中最优的是：

- `v16_240000_steps.zip`

近似结果：

- `speed_proxy_p50_ratio ≈ 1.94`
- `speed_proxy_p95_ratio ≈ 0.66`
- `abs_final_angle_p95_gap = 0.0`

## 结论

这两轮结果说明了一件更具体的事：

- **低维全局动作校正层是有效的**
  - 它能明显修正 steering 幅值
  - 也能明显压低整体速度
- 但**当前 Gate B 的 motion gate 需要同时对齐 `p50` 和 `p95`**
  - `v1` 还能说明“整体太快”
  - `v2` 已经说明“不是单纯整体过快，而是分布形状不对”

更直白地说：

- real `0421` 的速度分布更宽
- 当前 `warmup + V16 + gain/tau wrapper` 的 sim 速度分布更窄
- 单个 `throttle_gain + tau` 很难同时把：
  - `p50` 拉低到正常
  - 又保住 `p95` 不掉到过低

因此当前阻塞点已经从：

- `pose mismatch`

进一步收敛成：

- **driver speed-profile shape mismatch**

## 当前判断

截至这一步，可以明确说：

1. `sim2real_json` 这条线不是无效投入，已经证明有收益。
2. 但只靠当前这层 `gain + first-order lag`，不足以让 Gate B 通过。
3. 所以当前不适合继续推进：
   - pose sweep
   - interaction-only LWM
   - PPO baseline

## 建议的下一步

优先级从高到低：

1. 扩展 `sim2real` 校正自由度，不再只用 `gain/tau`
   - 进入 `ActionAdapter` 级别：
     - `adapter_v_nominal`
     - `max_throttle`
     - 可能再加 `v_min / v_max`
   - 也就是从“动作校正层”升级成“driver shaping layer”
2. 如果不想动 adapter 参数，就要调整 Gate B 的 motion profile 采样策略
   - 当前 `warmup` 驱动本身分布太窄
   - 需要一个更接近 `0421` 的 matched driver，而不只是换早期 checkpoint
3. 在这两条都不做的前提下，不建议继续扫 LiDAR pose
   - 因为 motion gate 还没过，pose 结论会被 confound

## 最终状态

- `Phase F` 主线仍然**停在 Gate B**
- 当前最新阻塞不是 LiDAR 几何本身，而是：
  - **sim driver 的速度分布形状仍然和 `0421` 不匹配**
- 两轮 `sim2real_json` 已验证：
  - 可改善
  - 但不足以单独解决问题
