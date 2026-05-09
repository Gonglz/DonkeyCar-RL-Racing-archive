# 当前 LWM 实现说明

本文档描述仓库里当前的 `LWM` 实现，指的是 `V17 local world model`，即 [module/local_world_model_v17.py](/home/longzhao/mysim_public/module/local_world_model_v17.py:1) 对应的模型与其数据、训练、用途和当前限制。

这份文档只讲“现在代码里已经实现了什么”，不讲更远期的理想方案。更偏规划性的背景可参考 [local_world_model_plan.md](/home/longzhao/mysim_public/docs/local_world_model_plan.md:1)。

## 1. 先说结论

当前 `LWM` 是一个：

- 局部的
- 短时序的
- LiDAR 驱动的
- 多头监督预测模型

它的主要目标不是替代 `PPO`，也不是生成整张未来场景，而是预测“当前局部交互会怎么发展”，例如：

- 自车短时动力学怎么变
- 前方目标车相对位置怎么变
- 左右空隙有多大、能不能过
- 未来短时间内碰撞/TTC风险怎样
- 当前动作是否让超车机会变好

当前实现里，`LWM` 需要单独离线训练，不和 `PPO` 联合在线训练。

## 2. 它不是哪个 world model

仓库里现在实际上有两套“world model”相关实现。

### 2.1 新的 LWM

- 文件: [module/local_world_model_v17.py](/home/longzhao/mysim_public/module/local_world_model_v17.py:1)
- 特点:
  - 输入是短时序
  - 读 `ego8 + lidar72 + async_meta4`
  - 有多个输出头
  - 目标是局部交互理解和机会/风险评估

### 2.2 旧的 bootstrap world model

- 文件: [module/world_model.py](/home/longzhao/mysim_public/module/world_model.py:1)
- 特点:
  - 只预测 ego 物理状态残差
  - 输入是 `8D` 控制+物理状态
  - 是当前在线 `predictive safety filter` 使用的模型

代码里也写得很明确：

- 新 LWM 在 [local_world_model_v17.py](/home/longzhao/mysim_public/module/local_world_model_v17.py:4) 被定义为 `offline training` 用的局部交互模型
- 旧模型在 [world_model.py](/home/longzhao/mysim_public/module/world_model.py:4) 被定义为当前 safety filter 的 `bootstrap local dynamics model`

所以，当前仓库中的“LWM”并没有替换掉在线 safety filter 使用的旧模型。

## 3. 当前 LWM 在做什么

### 3.1 输入

`LocalWorldModelV17` 的输入定义在 [local_world_model_v17.py](/home/longzhao/mysim_public/module/local_world_model_v17.py:91)：

- `ego_seq`: `(B, T, 8)`
- `camera_seq`: `(B, T, C, H, W)`，可选的降采样语义图像张量
- `lidar_seq`: `(B, T, 72)`
- `async_meta_seq`: `(B, T, 4)`
- `lengths`: `(B,)`

其中：

- `ego8` 在 [export_world_model_dataset.py](/home/longzhao/mysim_public/scripts/export_world_model_dataset.py:167) 构造，包含：
  - `v_long`
  - `yaw_rate`
  - `accel_x`
  - `steer_exec`
  - `throttle`
  - `prev_steer`
  - `prev_throttle`
  - `dt_norm`
- `lidar72` 是规范化后的 LiDAR 扁平向量
- `camera` 当前由 [export_world_model_dataset.py](/home/longzhao/mysim_public/scripts/export_world_model_dataset.py:226) 降采样构造，默认是 `5 x 64 x 64` 的语义图像
- `async_meta4` 在 [export_world_model_dataset.py](/home/longzhao/mysim_public/scripts/export_world_model_dataset.py:195) 构造，包含：
  - `lidar_scan_age_norm`
  - `lidar_steps_since_new_scan_norm`
  - `lidar_repeat_count_norm`
  - `lidar_is_new_scan`

### 3.2 输出头

输出头现在按职责拆成 3 组，定义在 [local_world_model_v17.py](/home/longzhao/mysim_public/module/local_world_model_v17.py:62) 到 [local_world_model_v17.py](/home/longzhao/mysim_public/module/local_world_model_v17.py:143)：

- `ego head`
  - `ego_delta`: 自车物理状态增量，3维
- `interaction head`
  - `target_rel`: 前方关键目标相对状态，4维
  - `closing_rate`: 逼近速度
  - `overtake_progress`: 超车进展代理
- `safety/passability head`
  - `gap`: 左右 gap，2维
  - `collision_logit`: 碰撞风险 logit
  - `ttc_proxy`: TTC 代理值
  - `passable_logits`: 左右可通行性 logits，2维

换句话说，当前 LWM 不试图回答“未来整张图长什么样”，而是回答“未来短时间内局部交互会不会更危险、哪边能过、是否更接近超车”。

## 4. 模型结构怎么实现的

整体结构在 [local_world_model_v17.py](/home/longzhao/mysim_public/module/local_world_model_v17.py:148) 到 [local_world_model_v17.py](/home/longzhao/mysim_public/module/local_world_model_v17.py:331)。

### 4.1 ego 分支

`ego_seq` 和 `async_meta_seq` 先拼接，再通过一个 MLP 编码：

- 见 [local_world_model_v17.py](/home/longzhao/mysim_public/module/local_world_model_v17.py:97)
- 输出单步 `64D` 特征

### 4.2 LiDAR 分支

LiDAR 编码方式是“左右分侧 + 共享编码器”：

- 先把 `72D` 拆成 `range` 和 `valid`
- 再拆成左 18 扇区和右 18 扇区
- 左右各走同一个 `SharedSideLidarEncoder`

对应代码：

- 编码器定义: [local_world_model_v17.py](/home/longzhao/mysim_public/module/local_world_model_v17.py:34)
- LiDAR 拆分与编码: [local_world_model_v17.py](/home/longzhao/mysim_public/module/local_world_model_v17.py:137)

这个设计的含义是：

- 保留左右两侧的结构信息
- 不把整圈 LiDAR 直接拍平成一个大 MLP
- 让模型更容易学“左边能不能过、右边能不能过”这类局部对称问题

### 4.3 Camera 分支

当前已经加了一个轻量 camera encoder：

- 输入是降采样后的语义图像序列
- 每帧用一个小 CNN 编成 `camera_feat`
- 然后和 ego/LiDAR 特征一起进入 `step_fusion`

这一步的目标不是让 LWM 变成“图像世界模型”，而是给 `gap/passable` 这类局部可通行语义留出相机输入通道。

### 4.4 时序融合

单步 ego 特征、LiDAR 特征以及可选 camera 特征拼接后，先过 `step_fusion`，再送进单层 `GRU`：

- `step_fusion`: [local_world_model_v17.py](/home/longzhao/mysim_public/module/local_world_model_v17.py:113)
- `GRU`: [local_world_model_v17.py](/home/longzhao/mysim_public/module/local_world_model_v17.py:119)

最后取最后时刻的隐状态 `hidden[-1]` 作为整段短序列的表征，再送进 trunk 和多个 head。

也就是说，当前 LWM 不是逐帧独立预测，而是明确利用了过去几步的局部时序信息。

## 5. 训练数据怎么来的

数据导出入口是 [scripts/export_world_model_dataset.py](/home/longzhao/mysim_public/scripts/export_world_model_dataset.py:1)。

### 5.1 数据来源

导出脚本会在仿真里跑 rollout，然后把每一步整理成监督样本。

rollout 来源可以是：

- 随机动作
- `V17` policy
- `V16` policy

对应 CLI 在 [export_world_model_dataset.py](/home/longzhao/mysim_public/scripts/export_world_model_dataset.py:498)。

关键参数：

- `--policy-path`
- `--policy-format {v17,v16}`
- `--samples`
- `--curriculum-phase`
- `--output`

### 5.2 样本内容

导出时会保存：

- 输入:
  - `ego8`
  - `lidar`
  - `async_meta`
- 监督目标:
  - `target_ego_delta`
  - `target_rel`
  - `target_rel_mask`
  - `target_gap`
  - `target_collision`
  - `target_ttc`
  - `target_safety_valid`
  - `target_passable`
  - `target_closing_rate`
  - `target_overtake_progress`
  - `target_opportunity_valid`
- 索引:
  - `episode_id`
  - `step_in_episode`
  - `scene_id`
  - `done`

对应保存逻辑见 [export_world_model_dataset.py](/home/longzhao/mysim_public/scripts/export_world_model_dataset.py:545) 到 [export_world_model_dataset.py](/home/longzhao/mysim_public/scripts/export_world_model_dataset.py:684)。

### 5.3 target 是怎么构造的

目标构造逻辑在 [export_world_model_dataset.py](/home/longzhao/mysim_public/scripts/export_world_model_dataset.py:207)。

核心思想：

- `target_rel`: 用相邻时刻的障碍物相对位置和相对速度构造
- `target_gap`: 当前默认使用传感器侧局部空域估计左右 gap；若显式指定 `gap_label_source=track`，才使用赛道几何 + 障碍物横向位置的标签路径
- `target_collision`: 不只用真实碰撞，还融合 `TTC < 0.60s 且 lateral_overlap > 0.20` 或 `near_collision_ttc_risk >= 0.85`
- `target_passable`: 根据左右 gap 是否超过阈值生成二分类标签；默认与传感器侧 `target_gap` 同源
- `target_closing_rate` / `target_overtake_progress`: 用连续两帧相对纵向位置变化构造

这说明当前 LWM 不是无监督的 latent world model，而是一个监督式的局部任务预测器。

## 6. 训练流程怎么实现的

训练入口是 [scripts/train_world_model_v17.py](/home/longzhao/mysim_public/scripts/train_world_model_v17.py:346)。

### 6.1 数据组织

`WorldModelSequenceDataset` 在 [train_world_model_v17.py](/home/longzhao/mysim_public/scripts/train_world_model_v17.py:35) 定义：

- 读取 `.npz`
- 按 `episode_id` 回溯最近 `seq_len` 帧
- 不够长的序列前面补零
- 返回 `length` 供 `pack_padded_sequence` 使用

所以当前训练是“按 episode 切分，再从 episode 中抽短时序窗口”的方式。

### 6.2 训练阶段

当前训练分三阶段。

#### Stage A

只训练几何/动力学相关头：

- `ego_loss`
- `target_loss`
- `gap_loss`

见 [local_world_model_v17.py](/home/longzhao/mysim_public/module/local_world_model_v17.py:230)。

目的：

- 先把基础几何和局部运动关系学稳

#### Stage B

冻结 trunk 和几何头，只训练安全/机会相关头：

- 冻结逻辑: [train_world_model_v17.py](/home/longzhao/mysim_public/scripts/train_world_model_v17.py:127)
- 只优化:
  - `collision_loss`
  - `ttc_loss`
  - `passable_loss`
  - `closing_loss`
  - `overtake_gain_loss`

见 [local_world_model_v17.py](/home/longzhao/mysim_public/module/local_world_model_v17.py:232)。

目的：

- 不破坏基础几何表征
- 单独把风险和超车机会头调起来

#### Stage C

联合微调所有头：

- 损失定义见 [local_world_model_v17.py](/home/longzhao/mysim_public/module/local_world_model_v17.py:234)

但 Stage C 有几何回退保护：

- 回退基线: [train_world_model_v17.py](/home/longzhao/mysim_public/scripts/train_world_model_v17.py:235)
- 回退判定: [train_world_model_v17.py](/home/longzhao/mysim_public/scripts/train_world_model_v17.py:243)
- 触发 guard 后回滚到 stage start: [train_world_model_v17.py](/home/longzhao/mysim_public/scripts/train_world_model_v17.py:291)

这个 guard 的意图是：

- 如果安全/机会头继续变好，但几何头明显退化，就宁可回滚，不接受联合微调结果

## 7. 当前正式训练结果

当前仓库里已有一轮正式训练结果：

- summary: [train_summary.json](/home/longzhao/mysim_public/models/world_model_v17_prelongtrain_20260421/formal_training/wm_mix_v1_run2/train_summary.json)
- 产物目录: `/home/longzhao/mysim_public/models/world_model_v17_prelongtrain_20260421/formal_training/wm_mix_v1_run2`

从这份 summary 可以看出：

- `Stage A`
  - `best_val_loss ≈ 2.975`
  - `mae_ego ≈ 0.034`
  - `mae_target_rel ≈ 2.028`
  - `mae_gap ≈ 1.740`
- `Stage B`
  - `best_val_loss ≈ 5.179`
  - 几何 MAE 基本保持不变
  - `collision/ttc/passable/closing` 等风险/机会头相对变好
- `Stage C`
  - 触发了几何回退 guard
  - 最终保存的是 `stage_c_guard_fallback.pth`
  - `local_world_model_v17_final.pth` 是 guard 后的稳定版本，不是完全成功联合微调后的版本

所以当前更准确的判断是：

- `LWM` 已经能稳定训练
- `Stage A/B` 可用
- `Stage C` 还不够稳，需要 guard 保底

## 8. 它现在有什么用

### 8.1 已经有用的地方

当前 LWM 已经能作为一个离线局部交互预测器，用来：

- 验证局部交互信号是否可学
- 预测 gap / passable / collision surrogate / closing rate
- 作为未来在线 safety / overtaking assessor 的候选模型
- 为 sim2real 做结构化局部表征准备

### 8.2 还没真正落地的地方

当前它还没有真正作为在线 `PPO` 控制链的一部分工作。

原因很直接：

- 在线 `predictive safety filter` 现在用的是旧的 [world_model.py](/home/longzhao/mysim_public/module/world_model.py:1)
- 见 [predictive_safety_filter.py](/home/longzhao/mysim_public/module/predictive_safety_filter.py:27) 和 [predictive_safety_filter.py](/home/longzhao/mysim_public/module/predictive_safety_filter.py:110)
- `v17` 环境里接 filter 的位置在 [v17_env.py](/home/longzhao/mysim_public/module/v17_env.py:518)
- 动作链消费 filter 的位置在 [action_adapter.py](/home/longzhao/mysim_public/module/action_adapter.py:233)

也就是说，当前“新 LWM”还主要是：

- 已训练
- 可评估
- 可作为未来模块接入

但还不是：

- 当前 PPO 在线训练时实际在用的安全预测模型

## 9. 它和 PPO 的关系

当前关系应该理解成：

1. `PPO` 在线和环境交互，产生 rollout
2. rollout 被导出成 LWM 数据集
3. `LWM` 在这些数据上离线监督训练
4. 训练好的 LWM 以后可以被接回在线安全过滤/机会评估链

所以当前不是“world model + PPO 联合优化”，而是“PPO 提供数据，LWM 单独训练”。

这也是为什么：

- 现在可以单独训练 LWM
- 但不能说当前 `PPO` 已经享受到新 LWM 的在线收益

## 10. 当前实现的边界和限制

### 10.1 不是完整场景生成模型

它不会：

- 重建图像
- 生成完整未来轨迹分布
- 做地图级长期规划
- 替代策略网络直接出控制

这点也和规划文档一致，见 [local_world_model_plan.md](/home/longzhao/mysim_public/docs/local_world_model_plan.md:12)。

### 10.2 当前在线集成还没完成

虽然 `PPO` 已经提供了 `predictive_safety_filter` 接口，但现在接进去的仍是旧的 ego-only 模型，不是新 LWM。

### 10.3 当前 safety filter 还是 log-first

即便启用在线 `predictive safety filter`，它也默认偏保守：

- `mode` 默认是 `log`
- `yaw_thresh` / `decel_thresh` 默认是 `None`

所以它默认更像“前向预测 + 记日志”，不是强干预器。

### 10.4 Stage C 仍不稳定

当前联合微调容易让几何能力退化，所以必须靠 guard 保护。

## 11. 如果你要把它真正用起来，下一步通常是什么

最自然的下一步不是“把 LWM 直接替代 PPO”，而是：

1. 继续积累高质量 rollout 数据
2. 让 `Stage B/C` 更稳
3. 写一个适配层，把 `LocalWorldModelV17` 接到当前安全过滤接口
4. 先跑 `log-only`
5. 再做轻量 `intervene`

也就是说，当前 LWM 更像一个：

- 已经成型的离线局部预测器
- 尚未完全产品化的在线安全/机会评估模块

## 12. 一句话总结

当前 `LWM` 是一个基于短时序 `ego8 + lidar72 + async_meta4` 的多头局部交互预测模型，已经能离线学习自车动力学、目标相对状态、gap、碰撞/TTC 和超车机会，但它目前仍需要单独训练，且还没有真正替代当前在线控制链里使用的旧 ego-only world model。
