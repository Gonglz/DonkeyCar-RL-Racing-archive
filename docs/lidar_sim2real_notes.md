# LiDAR Sim2Real 笔记

## 1. 范围

本文总结当前 LiDAR 数据链路、已采集的真实车数据、sim 与 real 的 LiDAR 差异，以及训练和部署时推荐采用的规范化处理路径。

核心目标不是直接使用原始 LiDAR，而是让 sim 和 real 在进入策略或推理之前，先产生同一种 LiDAR 表达。

## 2. 当前状态

### 2.1 真实车 LiDAR

- Jetson 侧硬件链路：
  - ROS `rplidar_ros`
  - topic: `/scan`
  - message type: `sensor_msgs/LaserScan`
- 运行时采集链路：
  - Jetson 侧权威脚本：`Jetson/runtime_monitor.py`
  - GitHub 来源：`https://github.com/Gonglz/DonkeyCar-RL-Racing/blob/main/Jetson/runtime_monitor.py`
  - 当前仓库同步副本：`/home/longzhao/mysim_public/Jetson/runtime_monitor.py`
  - 一份 CSV 记录每个主循环的元信息
  - 一份 JSONL sidecar 记录原始 LiDAR 快照
- 说明：
  - 本机另有一份历史副本：`/home/longzhao/monitor/runtime_monitor.py`
  - 该历史副本不是当前应作为数据来源基准的版本
  - 后续 real LiDAR 追加采集、字段对齐、schema 追踪统一以 `Jetson/runtime_monitor.py` 为准
- 原始快照格式：
  - 顶层是样本元信息
  - 内部嵌套 `lidar` 对象，字段包括：
    - `frame_count`
    - `scan_age_ms`
    - `valid_points`
    - `points_total`
    - `nearest_min`
    - `angle_min`
    - `angle_max`
    - `angle_increment`
    - `range_min`
    - `range_max`
    - `ranges`
    - `intensities`

### 2.2 当前真实车数据集

本地副本路径：

- `data/data_lidar/0421/data`
- `data/data_lidar/0421/monitor_logs/run_20260421_005659.csv`
- `data/data_lidar/0421/monitor_logs/run_20260421_005659_lidar_raw.jsonl`
- `data/data_lidar/0422_1/data`
- `data/data_lidar/0422_1/monitor_logs/run_20260422_165617.csv`
- `data/data_lidar/0422_1/monitor_logs/run_20260422_165617_lidar_raw.jsonl`

这次采集的关键事实：

- tub 记录数：`3927`
- monitor CSV 行数：`5604`
- raw LiDAR JSONL 行数：`5545`
- 与 tub 索引对齐的 recording + LiDAR 样本数：`3927`
- 对齐 tub 索引范围：`0..3926`
- recording 期间唯一 LiDAR scan 数：`1517`
- 每个 scan 被复用的中位次数：约 `3`
- `scan_age_ms` 中位数：约 `228.8`
- LiDAR 频率：约 `7 Hz`
- 相机 / 控制主循环频率：约 `16.6 FPS`

解释：

- 真实系统是异步的
- 一帧 LiDAR 会被多个图像 / 控制循环复用
- 训练代码不能假设 `1 张图像 == 1 帧新 LiDAR`

### 2.3 Sim LiDAR

当前 sim 中的 LiDAR 是由 Unity 扫描点重构成的固定长度数组。

重要特性：

- 数组长度由 `deg_per_sweep_inc` 决定
- 例如：
  - `deg_per_sweep_inc = 2.0`
  - `num_sweeps_levels = 1`
  - 输出长度 = `180`
- 无效值通常表示为 `-1`

这与 ROS `LaserScan` 本质不同。后者是按角度索引的原始扫描，包含 `inf`、`nan`、近场伪点，以及真实传感器的时间抖动。

## 3. 为什么不能直接用 Raw LiDAR

如果训练时使用 sim raw LiDAR、部署时直接喂 real raw `/scan`，通常不稳定。

主要原因：

- 无效值语义不同
- 角度参数化不同
- 噪声分布不同
- 真实车存在自反射和近场假点
- 真实车存在 scan 延迟和跨控制步复用
- 真实设备存在盲区和掉点

正确做法是定义一套统一的规范化 LiDAR 表达，让 sim 和 real 都先经过这一层。

## 4. 推荐的规范化 LiDAR 表达

### 4.1 核心表示

推荐第一版采用：

- `num_sectors = 36`
- `sector_0 = front`
- 所有 sector 使用固定旋转约定
- `max_range = 3.5 m` 或 `4.0 m`
- 额外一条有效性向量
- 额外一个 scan-age 标量

建议输出张量：

- `lidar_range`：shape `(36,)`
- `lidar_valid`：shape `(36,)`
- `lidar_age`：shape `(1,)`

### 4.2 Sector 语义

每个 sector 存的应该是稳健的距离统计，而不是某个随机 raw 点。

推荐统计：

- sector 内最小有效距离，或
- 较低分位数，例如 `p20`

直接取最小值更简单，而且从安全角度更保守。

### 4.3 无效值处理

不要把无效 sector 直接喂成 `0`。

推荐规则：

- `lidar_valid[i]` 保持在 `{0, 1}`
- 无效时令 `lidar_range[i] = max_range`

这样可以避免模型把“没有回波”误学成“前方极近障碍”。

### 4.4 近场过滤

真实 LiDAR 已经观察到传感器附近和车体结构带来的近场伪点。

推荐第一版规则：

- 丢弃 `< 0.18 m` 的点

该阈值应保持可配置。

## 5. Real `/scan` -> 规范化 LiDAR

### 5.1 输入

对单帧 `LaserScan`，输入包括：

- `ranges`
- `angle_min`
- `angle_increment`
- `range_min`
- `range_max`

### 5.2 处理流程

对每一根 beam：

1. 计算 beam 角度：

```text
theta_i = angle_min + i * angle_increment
```

2. 剔除无效值：

- `nan`
- `inf`
- `< near_clip`
- `> max_range`

3. 将 beam 角度映射到规范化 sector 索引。

4. 对该 sector 内的有效 beam 做聚合。

5. 输出：

- `lidar_range[sector]`
- `lidar_valid[sector]`

### 5.3 对齐规则

规范化适配器必须明确：

- 车头朝向
- 顺时针 / 逆时针索引方向
- 角度是按 `[-pi, pi)` 还是 `[0, 2pi)` 包裹

这个约定必须在以下场景保持完全一致：

- real 离线预处理
- real 在线推理
- sim 在线训练

## 6. Sim LiDAR -> 规范化 LiDAR

### 6.1 输入

sim 通常提供固定长度数组，特点包括：

- 角度间隔规则
- 无效值为 `-1`

### 6.2 处理流程

对 sim：

1. 把 `-1` 转成无效值
2. 依据相同的 sector 定义进行映射
3. 裁剪到同一个 `max_range`
4. 输出相同的：
   - `lidar_range`
   - `lidar_valid`
   - `lidar_age`

对 sim 来说，`lidar_age` 可以是：

- 直接为零
- 或人为加入延迟，用来模拟真实系统

### 6.3 2026-04-21 正式 sim 采集发现

为了做正式 sim-real gap 验收，本地新增了：

- `scripts/collect_sim_lidar_monitor.py`

该脚本会用一个已训练好的 `V16` policy 驱动 `V17` 环境，导出和 real monitor 类似的 sim raw-LiDAR JSONL。

这次有两个关键发现：

1. 当前代码默认的 sim LiDAR `offset_y = 1.14` 会导致 Unity handler 的 raw LiDAR 基本全是 `-1`。
2. 把传感器高度降到 `offset_y = 0.25` 后，sim raw-LiDAR 才开始稳定产生有效点。

本次正式 sim 采集配置为：

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

正式 sim log：

- `/home/longzhao/mysim_public/models/lidar_domain_gap_20260421/monitor_logs/run_20260421_165110_sim_donkey-waveshare-v0_lidar_raw.jsonl`

对应采集摘要：

- `frames = 800`
- `valid_ratio_mean ≈ 0.292`
- `scan_age_ms mean ≈ 81.9`

## 7. 规范化之后仍然存在的 Sim2Real Gap

规范化是必要条件，但不是充分条件。

sim 还需要进一步“做脏”，更像真实 LiDAR：

- 随机 beam dropout
- 距离抖动
- 偶发整扇区缺失
- scan 延迟
- 一个 scan 复用 `2-4` 个控制步
- 盲区 mask

否则 sim LiDAR 仍然会比真实 LiDAR 更干净、更同步。

### 7.1 当前正式验收结果

使用：

- real：
  - `/home/longzhao/mysim/data/data_lidar/0421/monitor_logs/run_20260421_005659_lidar_raw.jsonl`
  - `/home/longzhao/mysim/data/data_lidar/0422_1/monitor_logs/run_20260422_165617_lidar_raw.jsonl`
- sim：
  - `/home/longzhao/mysim_public/models/lidar_domain_gap_20260421/monitor_logs/run_20260421_165110_sim_donkey-waveshare-v0_lidar_raw.jsonl`
- 评估脚本：
  - `scripts/eval_lidar_domain_gap.py`

正式输出：

- `/home/longzhao/mysim_public/models/lidar_domain_gap_20260421/formal_eval_20260421_ws_offset025_max6.json`

阈值：

- `valid_ratio_mae <= 0.10`
- `wasserstein_median <= 0.08`
- `wasserstein_p95 <= 0.20`
- `scene_js_divergence <= 0.15`

实测结果：

- `valid_ratio_mae = 0.7332`
- `wasserstein_median = 0.2698`
- `wasserstein_p95 = 1.0000`
- `scene_js_divergence = 0.3088`
- `pass = false`

这说明：

- sim raw-LiDAR 现在已经能正确采到，不再是“sim 侧样本无效”
- 但 canonical 之后的 sector 级统计仍然明显偏离 real
- 当前 sim LiDAR 还不能直接拿来做 encoder 迁移或 real-first 验收签字

### 7.2 `GT` 近场包围增强实验

考虑到：

- `WS` 是封闭空间
- `GT` 更开放
- 策略最终吃的是 canonical LiDAR，而不是 raw scan

这里额外做了一次离线 `policy-faithful` 实验：

- 用 `WS` 的 canonical LiDAR 拟合 sector prior
- 在 `GT` 的 canonical LiDAR 上，只对侧向 / 斜前侧 sector 注入 `WS-style enclosure shell`
- 不修改 raw sim LiDAR，不改训练主链，只看 gap 能否下降

实验脚本：

- `scripts/augment_gt_lidar_enclosure.py`

输入：

- `WS` raw sim log：
  - `/home/longzhao/mysim_public/models/lidar_domain_gap_20260421/monitor_logs/run_20260421_165110_sim_donkey-waveshare-v0_lidar_raw.jsonl`
- `GT` raw sim log：
  - `/home/longzhao/mysim_public/models/lidar_domain_gap_20260421/monitor_logs/run_20260421_171744_sim_donkey-generated-track-v0_lidar_raw.jsonl`

最佳实验配置：

- `inject_valid_thresh = 0.05`
- `far_ratio = 0.20`

增强后输出：

- `/home/longzhao/mysim_public/models/lidar_domain_gap_20260421/gt_shell_aug_a.jsonl`
- `/home/longzhao/mysim_public/models/lidar_domain_gap_20260421/gt_shell_aug_a.summary.json`

增强后 formal gap：

- `/home/longzhao/mysim_public/models/lidar_domain_gap_20260421/formal_eval_20260421_gt_shell_aug_a.json`

对比结果：

- 原始 `GT -> real`
  - `valid_ratio_mae = 0.7176`
  - `wasserstein_median = 0.4621`
  - `scene_js_divergence = 0.4492`
- 增强后 `GT_shell_aug -> real`
  - `valid_ratio_mae = 0.4982`
  - `wasserstein_median = 0.2283`
  - `scene_js_divergence = 0.1677`

解释：

- 近场包围增强确实有效
- 最大改善来自 `left_far / left_mid` 角带
- `right_mid / right_far` 仍然较差，所以还不能直接通过 formal gate

这说明：

- `WS` 作为 LiDAR 对齐主域是合理的
- `GT` 不适合直接当 raw-LiDAR 对齐域
- 但 `GT` 可以在 canonical 层注入 `WS-style enclosure prior`，明显缩小和 real 的统计差距

### 7.3 2026-04-21 `GT` 地图调整后复测

在上一轮实验基础上，又对 `donkey-generated-track-v0` 的地图做了一次调整，并保持和旧实验完全一致的 LiDAR 配置重新采样：

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

新 raw sim log：

- `/home/longzhao/mysim_public/models/lidar_domain_gap_20260421_rerun_gtmap/monitor_logs/run_20260421_174843_sim_donkey-generated-track-v0_lidar_raw.jsonl`

对应采集摘要：

- `/home/longzhao/mysim_public/models/lidar_domain_gap_20260421_rerun_gtmap/monitor_logs/run_20260421_174843_sim_donkey-generated-track-v0_summary.json`

新的 formal gap：

- `/home/longzhao/mysim_public/models/lidar_domain_gap_20260421_rerun_gtmap/formal_eval_gt_rerun.json`

关键结果：

- 新 `GT -> real`
  - `valid_ratio_mae = 0.7134`
  - `wasserstein_median = 1.0000`
  - `wasserstein_p95 = 1.0000`
  - `scene_js_divergence = 0.4509`
- 旧 `GT -> real`
  - `valid_ratio_mae = 0.7176`
  - `wasserstein_median = 0.4621`
  - `wasserstein_p95 = 1.0000`
  - `scene_js_divergence = 0.4492`

解释：

- 原始 `GT` 的有效覆盖率没有明显改善
- `valid_ratio_mae` 只是从 `0.7176` 小幅降到 `0.7134`
- 但 `wasserstein_median` 从 `0.4621` 恶化到了 `1.0000`
- `scene_js_divergence` 也基本不变

这说明：

- 这次 `GT` 地图调整没有把 raw canonical LiDAR 分布实质性拉近 real
- 改动更像是把一部分原本“偶尔有近场返回”的 sector 重新推回了“无返回 / 极远返回”
- 恶化最明显的角带是 `left_mid` 和 `center`

具体表现为：

- 多个中左和中前 sector 的 `valid_ratio_sim` 从小于 `0.1` 的低频有效点，进一步降成了 `0`
- 对应 sector 的 `wasserstein` 直接跳到 `1.0`
- 右侧少数 sector 有改善，但不足以抵消左侧和中前区的退化

代表性恶化 sector：

- `sector 8, 9, 10, 11, 12, 13, 19`

因此这轮地图修改的结论是：

- 对 `GT` 的 LiDAR sim-real 对齐没有带来净收益
- 如果继续改 `GT`，方向不应只是“更封闭”或“更多墙”
- 更重要的是在 `left_mid / center / right_mid` 这些策略最敏感的角带里，补回稳定近场结构，而不是把少量有效返回变成彻底无返回

另外，也用这份新 `GT` raw log 重新测试了上一轮最好的 `GT shell augmenter`：

- 增强后输出：
  - `/home/longzhao/mysim_public/models/lidar_domain_gap_20260421_rerun_gtmap/gt_shell_aug_rerun_a.jsonl`
- 增强后 formal gap：
  - `/home/longzhao/mysim_public/models/lidar_domain_gap_20260421_rerun_gtmap/formal_eval_gt_shell_aug_rerun_a.json`

结果：

- 新 `GT_shell_aug -> real`
  - `valid_ratio_mae = 0.4940`
  - `wasserstein_median = 0.4830`
  - `wasserstein_p95 = 1.0000`
  - `scene_js_divergence = 0.1644`
- 旧 `GT_shell_aug -> real`
  - `valid_ratio_mae = 0.4982`
  - `wasserstein_median = 0.2283`
  - `wasserstein_p95 = 1.0000`
  - `scene_js_divergence = 0.1677`

解释：

- 新图上的 shell prior 依然能显著压低 `valid_ratio_mae` 和 `scene_js_divergence`
- 但 `wasserstein_median` 从 `0.2283` 回升到 `0.4830`
- 也就是说，增强后的“覆盖率”更像 real 了，但距离分布形状反而比旧图差

当前更合理的判断是：

- `GT` 继续作为开放场景行为多样性来源
- `WS` 继续作为 LiDAR 对齐主域
- `GT` 上如果要继续走 `policy-faithful` 路线，下一步应优先做 `scene-aware / right-biased enclosure prior` 或 `ROI-conditioned shell prior`

### 7.4 2026-04-21 `target-token` 可行性实验

为了验证“LiDAR 不再负责 full-scene 几何，而只负责前景目标 token”这条路线，代码里新增了一个最小实验版本：

- `module/lidar.py`
  - 新增 `TargetTokenBuffer`
  - 从 canonical LiDAR 提取 `12D` target token：
    - `exist`
    - `rel_long`
    - `rel_lat`
    - `rel_v_long`
    - `rel_v_lat`
    - `ttc`
    - `confidence`
    - `age_norm`
    - `width_proxy`
    - `front_min_range`
    - `left_gap_proxy`
    - `right_gap_proxy`
- `module/v17_env.py`
  - 新增 `lidar_obs_mode = full | target_token`
- `module/v17_policy.py`
  - `72D full lidar` 继续走 side-separated encoder
  - `12D target token` 改走小 MLP encoder
- `src/ppo_multitrack_v17.py`
  - 新增 CLI: `--lidar-obs-mode {full,target_token}`
  - contract test 已覆盖 `target_token`

另外新增评估脚本：

- `scripts/eval_target_token_feasibility.py`

它用 sim monitor log 里的 obstacle runtime truth 评估 target token 前端：

- detection recall / precision
- `rel_long / rel_lat` MAE
- `rel_v_long / rel_v_lat` MAE
- `TTC` MAE

本轮采样使用：

- `WS`：
  - `/home/longzhao/mysim_public/models/target_token_feasibility_20260421/monitor_logs/run_20260421_184507_sim_donkey-waveshare-v0_lidar_raw.jsonl`
- `GT`：
  - `/home/longzhao/mysim_public/models/target_token_feasibility_20260421/monitor_logs/run_20260421_184655_sim_donkey-generated-track-v0_lidar_raw.jsonl`

首轮评估结果：

- `/home/longzhao/mysim_public/models/target_token_feasibility_20260421/target_token_eval_ws_gt.json`

整体：

- `recall = 0.887`
- `precision = 0.532`
- `mae_rel_long = 1.121 m`
- `mae_rel_lat = 2.582 m`
- `mae_rel_v_long = 2.466 m/s`
- `mae_rel_v_lat = 2.028 m/s`
- `mae_ttc = 2.610 s`

分场景：

- `generated_track`
  - `recall = 0.951`
  - `precision = 0.503`
  - `mae_rel_long = 1.415 m`
  - `mae_rel_lat = 3.474 m`
- `waveshare`
  - `recall = 0.829`
  - `precision = 0.566`
  - `mae_rel_long = 0.812 m`
  - `mae_rel_lat = 1.646 m`

之后又做了一次更偏“前方窄目标”的 scorer 调整，结果在：

- `/home/longzhao/mysim_public/models/target_token_feasibility_20260421/target_token_eval_ws_gt_v2.json`

结果没有实质改善：

- `recall = 0.887`
- `precision = 0.532`
- `mae_rel_long = 1.124 m`
- `mae_rel_lat = 2.715 m`
- `mae_ttc = 2.575 s`

这轮实验的结论很明确：

- 从工程实现角度看，`target-token` 路线是可行的
  - `target_token` 模式 env / policy / PPO smoke 都已跑通
- 但从当前“只靠 canonical LiDAR 聚类”的前景提取质量看，还不够直接替代 full lidar
  - recall 很高，说明“看到东西”不难
  - precision 偏低，说明会把大量背景结构误当目标
  - `rel_lat / rel_v / TTC` 误差偏大，说明简单 cluster tracker 还不够区分“前景目标”和“侧墙/包围结构”

因此更合理的下一步不是直接切到“纯 LiDAR target-token 部署”，而是：

- `image ROI / enclosure prior` 帮助 target gating
- `scene-aware / right-biased shell prior`
- 或直接走 `teacher target-token` 路线，先验证 PPO 是否真的受益，再决定是否继续把前端感知做强

## 8. LiDAR 在训练中的用法

目前有三种有价值的层级。

### 8.1 只保留障碍摘要

把规范化 LiDAR 压成：

- `obstacle_present`
- `obstacle_longitudinal`
- `obstacle_lateral`
- `obstacle_dist`
- `obstacle_risk`

优点：

- 与当前 PPO V16 的接入成本最低

缺点：

- 丢掉了绝大部分 LiDAR 几何信息

### 8.2 把规范化 LiDAR 作为策略输入

输入改成：

- `image`
- `state`
- `lidar_range`
- `lidar_valid`
- `lidar_age`

优点：

- 保留障碍物几何结构
- 表达能力和工程复杂度比较平衡

缺点：

- 需要多模态特征提取器

### 8.3 局部占据图 / BEV

把 LiDAR 投影成局部占据栅格，再用于训练。

优点：

- 空间表达能力最强

缺点：

- 预处理更重
- 模型与部署成本更高

## 9. LiDAR 在部署中的用法

部署时的运行链应为：

```text
/scan
-> canonical_lidar_adapter
-> 策略输入和/或安全层
-> 动作输出
```

不能训练时让模型吃 canonical LiDAR，部署时却直接把 raw `/scan` 喂进去。

## 10. 近期推荐方向

推荐的实用路径：

1. 持续保存 raw LiDAR 快照。
2. 为当前数据集构建离线 canonical LiDAR 生成器。
3. 为 sim 构建同样的 canonical adapter。
4. 训练时只使用 canonical 表达。
5. 在真实车运行时加入在线 canonical 化。

推荐第一版 canonical 规格：

- sectors：`36`
- `sector_0 = front`
- `max_range = 3.5`
- `near_clip = 0.18`
- 输出：
  - `lidar_range[36]`
  - `lidar_valid[36]`
  - `lidar_age[1]`

## 11. 相关文件

- `data/data_lidar/0421/monitor_logs/run_20260421_005659.csv`
- `data/data_lidar/0421/monitor_logs/run_20260421_005659_lidar_raw.jsonl`
- `data/data_lidar/0422_1/monitor_logs/run_20260422_165617.csv`
- `data/data_lidar/0422_1/monitor_logs/run_20260422_165617_lidar_raw.jsonl`
- `DonkeyCar-RL-Racing/src/ppo_multitrack_v16.py`
- `DonkeyCar-RL-Racing/module/obv.py`
- `DonkeyCar-RL-Racing/module/multi_scene_env.py`
- `DonkeyCar-RL-Racing/module/actor.py`
- `mysim/module/world_model.py`
- `mysim/module/world_model_dataset.py`
- `mysim/module/predictive_safety_filter.py`

## 12. 决策总结

这里最关键的设计决策是：

- raw LiDAR 要存
- 训练和部署都吃 canonical LiDAR
- 策略代码不能直接绑定 raw ROS scan 格式

这样可以保证数据可复用、sim 与 real 可对齐，并且以后就算修改原始传感器解析逻辑，也不必整体重训。
