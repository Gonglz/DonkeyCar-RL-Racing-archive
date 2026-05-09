# V17 Pre-Long-Train Preparation

Date: 2026-04-21
Timezone: America/New_York

## Goal

在启动正式 `V17` 长训前，先把以下准备项固定下来：

- 关键代码与当前 smoke 产物可回滚
- `world-model` 数据导出链可稳定运行
- `Stage A/B/C` 训练链可稳定运行
- 正式长训入口与配置可复现

## Rollback Backup

备份目录：

- `/home/longzhao/mysim_public/backups/v17_prelongtrain_20260421_070420`

内容：

- `code/`
  - `module/` 下 V17 相关核心文件
  - `scripts/` 下 LiDAR / world-model 相关脚本
  - `src/ppo_multitrack_v17.py`
  - 当前相关文档
- `artifacts/`
  - `models/v17_smoke_20260421_d`
  - `models/world_model_v17_smoke_20260421_a`
- `SHA256SUMS.txt`

回滚方式：

1. 从备份目录把目标文件 `rsync -a` 回工作树。
2. 如需回滚 smoke / world-model 产物，直接从 `artifacts/` 复制回 `models/`。
3. 需要精确校验时，对照 `SHA256SUMS.txt`。

## World-Model Data Collection Policy

结论：

- `Stage B/C` 数据不应主要依赖 random rollout。
- 主数据应由训练好的 `V16` policy 驱动，确保能进入真实障碍交互区。
- random rollout 只适合作为补充尾部分布，不适合作为主采样源。

当前 exporter 已支持：

- `--policy-format v17`
- `--policy-format v16`

其中 `v16` 模式下：

- 行为策略使用 `6ch image + 12D state`
- 采样环境仍为 `V17` canonical LiDAR 环境
- 导出仍保留 `ego8 + lidar + async_meta`

## Gap / Passable Labels

当前版本不再只用 LiDAR 侧分位数构造 `gap/passable`。

优先使用：

- 赛道局部宽度
- ego 当前赛道横向位置
- runtime obstacle 相对横向位置

构造：

- `left_gap`
- `right_gap`
- `passable_left`
- `passable_right`

若几何信息不可用，才回退到 LiDAR 侧估计。

## Collision Label

当前 `target_collision` 不再只认“真实碰撞”。

同时纳入 near-collision surrogate：

- 真实 collision flag
- 或 `TTC < 0.60s` 且 `lateral_overlap > 0.20`
- 或 `near_collision_ttc_risk >= 0.85`

目的：

- 避免 `collision` 头在 obstacle-rich rollout 中仍退化成全 `0`
- 让 safety head 更接近 `danger window` 定义

## LiDAR Sim-Real Validation

脚本已具备：

- `scripts/eval_lidar_domain_gap.py`

当前状态：

- 已找到真实 LiDAR monitor logs：
  - `/home/longzhao/mysim/data/data_lidar/0421/monitor_logs/run_20260421_005659_lidar_raw.jsonl`
  - `/home/longzhao/mysim/data/data_lidar/0422_1/monitor_logs/run_20260422_165617_lidar_raw.jsonl`
- 已确认 real LiDAR 采集脚本来源：
  - GitHub：`https://github.com/Gonglz/DonkeyCar-RL-Racing/blob/main/Jetson/runtime_monitor.py`
  - 当前仓库同步副本：`/home/longzhao/mysim_public/Jetson/runtime_monitor.py`
  - 文件 `sha256`：`20ad094ee8a2999f187311a6d3e8c4d924b883e3a709169aea735a92261ec081`
- 该文件可被 `eval_lidar_domain_gap.py` 直接解析，格式包含：
  - `lidar.angle_min`
  - `lidar.angle_increment`
  - `lidar.ranges`
  - `lidar.scan_age_ms`
- 实测 real 侧基础统计：
  - `samples = 5545`
  - `valid_ratio_mean ≈ 0.991`
  - `mean_valid_range_m ≈ 2.327`
- 因此 real 侧数据缺失问题已解除
- 因此 real 侧“数据 + 采集脚本来源”问题都已解除

当前 blocker：

- 已补充 sim raw-lidar monitor log：
  - `scripts/collect_sim_lidar_monitor.py`
  - 正式采集输出：
    - `/home/longzhao/mysim_public/models/lidar_domain_gap_20260421/monitor_logs/run_20260421_165110_sim_donkey-waveshare-v0_lidar_raw.jsonl`
    - `/home/longzhao/mysim_public/models/lidar_domain_gap_20260421/monitor_logs/run_20260421_165110_sim_donkey-waveshare-v0_summary.json`
- 采集使用 `V16` 已训练 policy 驱动 `V17` 环境，避免 random policy 采不到障碍或赛道侧边
- 本次正式 sim LiDAR 配置：
  - `deg_per_sweep_inc = 1.0`
  - `deg_ang_down = 0.0`
  - `deg_ang_delta = -1.0`
  - `num_sweeps_levels = 1`
  - `max_range = 6.0`
  - `noise = 0.0`
  - `offset_x = 0.0`
  - `offset_y = 0.25`
  - `offset_z = 0.5`
  - `rot_x = 0.0`
- 采集统计：
  - `frames = 800`
  - `episodes = 91`
  - `new_scan_count = 195`
  - `valid_ratio_mean ≈ 0.292`
  - `scan_age_ms mean ≈ 81.9`
- 一个重要发现：
  - 代码当前默认的 sim LiDAR `offset_y = 1.14` 会导致 raw LiDAR 基本全 `-1`
  - 只有把传感器高度降到 `offset_y = 0.25` 后，sim raw-LiDAR 才开始稳定产生有效点
- 因此 Phase 0 现在不再是“缺 sim raw-lidar log”，而是“已有正式 sim log，但正式验收未通过”

当前 smoke 记录：

- 输出：
  - `/home/longzhao/mysim_public/models/lidar_domain_gap_smoke_20260421.json`
- 该 smoke 只是验证脚本通路，不代表真实 domain-gap 结论
- 结果显示：
  - `real_samples = 5545`
  - `sim_samples = 10`
  - `valid_ratio_mae ≈ 0.991`
  - `wasserstein_median = 1.0`
  - `scene_js_divergence ≈ 0.646`
  - `pass = false`
- 失败主因不是已经证明 sim-real gap 极大，而是 sim 侧样本无效

正式验收结果：

- 输出：
  - `/home/longzhao/mysim_public/models/lidar_domain_gap_20260421/formal_eval_20260421_ws_offset025_max6.json`
- 输入：
  - real：
    - `/home/longzhao/mysim/data/data_lidar/0421/monitor_logs/run_20260421_005659_lidar_raw.jsonl`
    - `/home/longzhao/mysim/data/data_lidar/0422_1/monitor_logs/run_20260422_165617_lidar_raw.jsonl`
  - sim：
    - `/home/longzhao/mysim_public/models/lidar_domain_gap_20260421/monitor_logs/run_20260421_165110_sim_donkey-waveshare-v0_lidar_raw.jsonl`
- 阈值：
  - `valid_ratio_mae <= 0.10`
  - `wasserstein_median <= 0.08`
  - `wasserstein_p95 <= 0.20`
  - `scene_js_divergence <= 0.15`
- 实测：
  - `valid_ratio_mae = 0.7332`
  - `wasserstein_median = 0.2698`
  - `wasserstein_p95 = 1.0000`
  - `scene_js_divergence = 0.3088`
  - `pass = false`
- 当前结论：
  - 正式 sim-real LiDAR gap 验收已经跑完，结果为**不通过**
  - 这次失败已经不是“sim 样本无效”，而是**在可用 sim raw-LiDAR 条件下，sector 级统计仍明显偏离 real**
  - 下一步优先级应转为：
    - 为 sim LiDAR 增加更接近 real 的随机化和 blind-zone / dropout 建模
    - 重新审视 V17 / export 里默认 `offset_y = 1.14` 的配置
    - 收集更多 matched-scene real logs，而不是直接假设当前 waveshare sim 可代表 real 房间场景

## Formal Prep Runs

### Data Export

正式准备阶段保留的可用导出：

- `GT` 主集
  - `/home/longzhao/mysim_public/models/world_model_v17_prelongtrain_20260421/formal_dataset/gt_lane_pid_full_v16gt_main.npz`
  - `3072` samples
  - `21` episodes
  - `collision_pos_rate ≈ 0.154`
  - `passable_left_rate ≈ 0.752`
  - `passable_right_rate ≈ 0.236`
- `GT` 补充集
  - `/home/longzhao/mysim_public/models/world_model_v17_prelongtrain_20260421/pilot_gt_lane_pid_full_v16gt_collisionfix.npz`
  - `1024` samples
  - `collision_pos_rate ≈ 0.102`
- `WS` 辅集
  - `/home/longzhao/mysim_public/models/world_model_v17_prelongtrain_20260421/pilot_ws_lane_pid_full_v16ws_collisionfix.npz`
  - `1024` samples
  - `collision_pos_rate ≈ 0.187`

合并后的正式训练集：

- `/home/longzhao/mysim_public/models/world_model_v17_prelongtrain_20260421/formal_dataset/wm_dataset_mix_v1.npz`
- `/home/longzhao/mysim_public/models/world_model_v17_prelongtrain_20260421/formal_dataset/wm_dataset_mix_v1.json`

合并统计：

- 总样本数：`5120`
- 总 episodes：`33`
- `scene_counts`
  - `generated_track = 4096`
  - `waveshare = 1024`
- `collision_pos_rate ≈ 0.150`
- `opportunity_valid_rate ≈ 0.862`
- `passable_left_rate ≈ 0.650`
- `passable_right_rate ≈ 0.338`

### World-Model Training

正式训练结果目录：

- `/home/longzhao/mysim_public/models/world_model_v17_prelongtrain_20260421/formal_training/wm_mix_v1_run2`

关键产物：

- `stage_a_best.pth`
- `stage_b_best.pth`
- `stage_c_guard_fallback.pth`
- `local_world_model_v17_final.pth`
- `train_summary.json`

训练规模：

- `train_size = 4448`
- `val_size = 672`
- `seq_len = 4`
- `epochs_a / epochs_b / epochs_c = 8 / 6 / 3`

关键结果：

- `Stage A best`
  - `val_loss ≈ 2.975`
  - `mae_ego ≈ 0.034`
  - `mae_target_rel ≈ 2.028`
  - `mae_gap ≈ 1.740`
- `Stage B best`
  - `val_loss ≈ 5.179`
  - 几何指标保持不变，说明 trunk / geometry heads 冻结生效
- `Stage C`
  - 第 1 个 epoch 触发几何回退 guard
  - 自动回退到 `Stage B` 起点
  - 已保存 `stage_c_guard_fallback.pth`
  - 最终 `local_world_model_v17_final.pth` 为 guard 后的稳定版本

### V17 PPO Readiness

主训练链 smoke 已完成：

- `/home/longzhao/mysim_public/models/v17_smoke_20260421_d`

关键产物：

- `final_model.zip`
- `final_model_policy.pth`
- `v17_config.json`
- `train_metrics.jsonl`

这说明 `V17 PPO` 已通过：

- simulator connect
- collect rollouts
- PPO update
- checkpoint save

## Recommended Formal Long-Train Command

`V17` 现在已支持自动课程调度。

对“从零开始”的正式长训，推荐直接走自动课程，而不是手动固定在 `lane_pid_full`。

推荐命令：

```bash
/home/longzhao/miniconda3/envs/donkey37/bin/python \
  /home/longzhao/mysim_public/src/ppo_multitrack_v17.py \
  --env-ids donkey-generated-track-v0 donkey-waveshare-v0 \
  --steps 2000000 \
  --save-dir /home/longzhao/mysim_public/models/v17_formal_20260421 \
  --auto-curriculum \
  --critic-calibration-freq 50000 \
  --file-metrics-log-freq 500 \
  --exp-tag v17_formal_20260421
```

如需更保守地先做正式预跑，可先把：

- `--steps` 降到 `300000`
- 其余参数不变

如需从中间阶段续训，例如直接从 `avoid_static` 开始：

```bash
/home/longzhao/miniconda3/envs/donkey37/bin/python \
  /home/longzhao/mysim_public/src/ppo_multitrack_v17.py \
  --env-ids donkey-generated-track-v0 donkey-waveshare-v0 \
  --steps 1200000 \
  --save-dir /home/longzhao/mysim_public/models/v17_formal_20260421 \
  --auto-curriculum \
  --auto-curriculum-start-stage avoid_static \
  --resume-latest \
  --critic-calibration-freq 50000 \
  --file-metrics-log-freq 500 \
  --exp-tag v17_formal_20260421_resume
```

自动课程会在 `save_dir` 下额外写出：

- `curriculum_window.jsonl`
- `v17_auto_curriculum_summary.json`

## Remaining Blockers

### 已解决

- V17 PPO 主训练链可运行
- world-model 导出链可运行
- world-model `Stage A/B/C` 训练链可运行
- `Stage C` 几何回退 guard 可优雅回退，不再直接失败
- 关键代码和当前产物已备份，可回滚

### 仍未解决

- 缺少文档中提到的真实 LiDAR monitor logs
- 因此 `scripts/eval_lidar_domain_gap.py` 还不能完成正式 sim-real 验收

结论：

- **sim 侧正式长训准备已完成**
- **real-first / deployment readiness 仍缺一块 LiDAR sim-real 验收**
