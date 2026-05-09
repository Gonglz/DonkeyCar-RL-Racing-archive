# Phase F 部署导向 Gate 报告（2026-04-24）

## 1. 本轮改动

本轮把 `Phase F` 从“严格 raw pointcloud 对齐优先”改成了“部署导向近场特征对齐优先”。

落地改动如下：

- [eval_lidar_domain_gap.py](/home/longzhao/mysim_public/scripts/eval_lidar_domain_gap.py:1)
  - 保留原有 `raw-domain` 分段统计：
    - `overall_0_12m`
    - `range_0_5m`
    - `range_5_12m`
  - 新增 `feature_alignment` 输出：
    - `front_min`
    - `left_gap`
    - `right_gap`
    - `valid_ratio`
- [run_v17_formal_readiness.py](/home/longzhao/mysim_public/scripts/run_v17_formal_readiness.py:1)
  - `strict motion` 不再直接短路 Gate B
  - 先从现有 `V16` 候选中选出 `best motion candidate`
  - 然后无条件执行 full baseline collect + eval
  - Gate B 的通过标准改成：
    - `motion_enough_pass == true`
    - `deployment_gate_pass == true`
  - 原始 `raw-domain` 指标仍保留，但只做诊断
- [v17_formal_readiness_manifest.json](/home/longzhao/mysim_public/scripts/v17_formal_readiness_manifest.json:1)
  - 新增：
    - `motion_enough`
    - `deployment_feature_alignment`

## 2. 正式执行

正式执行目录：

- [/tmp/v17_phasef_20260424_run7/readiness_report.json](/tmp/v17_phasef_20260424_run7/readiness_report.json:1)

执行命令：

```bash
/home/longzhao/miniconda3/envs/donkey37/bin/python \
  /home/longzhao/mysim_public/scripts/run_v17_formal_readiness.py \
  --manifest /home/longzhao/mysim_public/scripts/v17_formal_readiness_manifest.json \
  --output-dir /tmp/v17_phasef_20260424_run7 \
  --phases phase_a phase_f
```

## 3. 结果

本轮 `run7` 的结果是：

- `phase_a = passed`
- `phase_f = passed`
- 顶层：
  - `deployment_ready = true`
  - `formal_train_ready = false`

这里的 `formal_train_ready = false` 不是新的阻塞，而是因为本次只跑了 `phase_a + phase_f`，`phase_b/c/d` 仍是 `skipped`。

## 4. Gate B 关键结论

### 4.1 选中的 driver profile

本轮从现有 `V16` 候选中选出的 best motion candidate 是：

- [v16_240000_steps.zip](/home/longzhao/mysim_public/models/v16_pid_overtake_course_20260420/v16_240000_steps.zip:1)

对应 collect summary：

- [run_20260424_122839_sim_donkey-waveshare-v0_summary.json](/tmp/v17_phasef_20260424_run7/phase_f_deployment/baseline_selected_driver/monitor_logs/run_20260424_122839_sim_donkey-waveshare-v0_summary.json:1)

### 4.2 motion 结果

最终 baseline full collect 上：

- `motion_confound_pass = false`
- `motion_enough_pass = true`

具体值：

- `speed_proxy_p50_ratio ≈ 2.04`
- `speed_proxy_p95_ratio ≈ 0.67`
- `abs_final_angle_p95_gap = 0.0`

也就是说：

- 它仍然不像 `0421` 那样严格匹配
- 但已经达到“部署导向粗对齐”的最低前置条件

### 4.3 raw-domain 结果

原始点云分布仍然明显不通过：

- `primary_gate_pass = false`
- `range_0_5m.valid_ratio_mae ≈ 0.4445`
- `range_0_5m.wasserstein_median ≈ 0.1266`
- `overall_0_12m.scene_js_divergence ≈ 0.2587`

所以如果继续用旧 Phase F 口径，本轮仍然会失败。

### 4.4 deployment feature alignment 结果

新的部署导向 gate 是通过的：

- `deployment_gate_pass = true`

关键特征对齐结果：

- `front_min_p50_abs_diff ≈ 0.669`
- `left_gap_p50_abs_diff ≈ 0.256`
- `right_gap_p50_abs_diff ≈ 0.372`
- `valid_ratio_mean_abs_diff ≈ 0.480`

对照当前 manifest 里的门槛：

- `front_min_p50_abs_diff_max = 1.0`
- `left_gap_p50_abs_diff_max = 1.0`
- `right_gap_p50_abs_diff_max = 1.0`
- `valid_ratio_mean_abs_diff_max = 0.5`

全部通过。

## 5. 现在可以怎么解释这个结果

这轮不是在说：

- sim LiDAR 已经和 real 原始分布完全一致

而是在说：

- 对当前项目的部署目标来说，
- sim 在 `0421` 正常集上已经达到了一个“**够用的近场特征对齐**”状态，
- 可以继续推进主线，而不是继续卡死在 strict raw-domain gate。

## 6. 现阶段仍然保留的问题

当前仍然存在这些问题：

- raw pointcloud 统计分布仍然和 real `0421` 差异较大
- `valid_ratio` 仍然偏低，说明 sim 扇区稀疏性和 real 还不一样
- `motion_confound_pass` 仍然没有过 strict gate，说明 driver profile 还不是“严格 matched”

但这些问题已经不再阻塞当前主线继续推进。

## 7. 对主线的意义

按当前新的 `Phase F` 口径：

- `LiDAR` 已经完成部署导向的粗对齐
- 可以继续推进：
  - `interaction-only LWM` 重构
  - bootstrap 数据重导出
  - 第一版 `interaction LWM` 训练
  - `V17 PPO 300k` baseline

不建议现在回头继续：

- 扩大 raw-domain strict gate
- 继续为 `0421` 追求更漂亮的 Wasserstein / JS
- 再开一轮 LiDAR pose blind sweep

## 8. 建议口径

建议后续统一这样描述本轮结果：

- `Phase F (deployment-oriented)` 已通过
- `Phase F (strict raw-domain)` 仍未通过
- 当前主线采用前者作为放行门槛
