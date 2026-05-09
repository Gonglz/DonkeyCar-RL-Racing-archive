# 局部世界模型计划（超车导向）

## 1. 目标

构建一个面向超车任务的局部世界模型，同时预测：

- 自车短时域动力学
- 对手车相对状态的短时变化
- 左右可通行缺口的演化
- 当前动作对碰撞风险和超车机会的影响

这不是完整的 latent generative world model，也不是全局地图级规划器。
它是一个 **局部交互世界模型**，服务于以下目标：

- 超车策略学习
- 近场交互预测
- 短时域安全过滤
- sim2real 迁移

V1 明确只覆盖 `0.3~0.8 s` 的短时预测。
更准确地说，它是一个 **risk / gap / opportunity assessor**，不是独立决策核心。

## 2. 为什么要改成超车导向

如果最终任务是超车，那么“避障世界模型”不够用。

避障导向的局部模型只关心：

- 前方是否有障碍
- 当前动作是否危险

但超车任务真正需要回答的是：

- 前方目标是静态障碍还是正在运动的对手车
- 对手相对速度是接近还是远离
- 左边还是右边存在可通行缺口
- 当前动作会打开超车窗口还是把自己送进碰撞
- 未来短时窗口 `0.3~0.8 s` 内超车机会是在打开还是关闭

因此，世界模型的重点不应只是 `obstacle risk`，而应是：

- `opponent relative motion`
- `gap evolution`
- `overtake opportunity`

## 3. 推荐的模型边界

### 3.1 V1 需要覆盖的内容

- `0.3~0.8 s` 短时域 ego dynamics 预测
- 最近前向对手的相对状态预测
- 左右缺口大小预测
- 碰撞概率 / TTC 预测
- 当前动作是否有利于超车的代理预测
- 给安全过滤或机会评分提供依据，但不直接接管控制

### 3.2 V1 暂时不做的内容

- 全图像重建
- image-conditioned latent world model
- 全场景多目标完整轨迹生成
- 地图级长期行为规划
- Dreamer 风格 latent rollout
- 大幅改写 `steer/throttle` 的在线控制接管

V1 的目标不是做“完整世界模拟器”，而是做一个能直接服务超车决策的局部预测器。
它首先服务于离线预测验证，之后才可能进入 `log-only` 车上验证。

## 4. 超车任务下的建模对象

建议把局部世界模型的核心对象明确成三类：

### 4.1 自车状态

- `v_long`
- `yaw_rate`
- `accel_x`
- 已执行转向 / 油门

### 4.2 最近关键对手

V1 不做多目标完整建模，先聚焦一个“最相关对手”：

- 优先选择前向锥区内风险最大或最近的目标
- 输出其相对状态：
  - `rel_longitudinal`
  - `rel_lateral`
  - `rel_speed_longitudinal`
  - 可选 `rel_heading`

### 4.3 缺口与可通行性

比单独的障碍距离更重要的是：

- `left_gap`
- `right_gap`
- `passable_left`
- `passable_right`

超车决策的本质是“选哪边过”和“什么时候能过”，因此 gap 是核心目标。

## 5. 推荐的 V1 输入定义

在时刻 `t`，建议输入由三部分组成。

### 5.1 Ego 动力学与控制输入

- `v_long_t`
- `yaw_rate_t`
- `accel_x_t`
- `steer_exec_t`
- `throttle_t`
- `prev_steer_exec_t`
- `prev_throttle_t`
- `dt_norm`

共 `8D`。

### 5.2 LiDAR 规范化输入

推荐：

- `lidar_range_t[36]`
- `lidar_valid_t[36]`
- `lidar_age_t`

共 `73D`。

### 5.3 时间上下文

超车依赖相对运动，单帧 LiDAR 不足以区分：

- 静态障碍
- 正在追上的对手
- 正在远离的目标

因此 V1 就应该带短时间窗，而不是只看单帧。

推荐两种实现二选一：

1. 堆叠最近 `K=3` 帧 canonical LiDAR
2. 使用小型 GRU 编码最近 `K=3~5` 步

如果采用堆叠，LiDAR 输入会变成：

- `3 x (36 + 36 + 1)`

如果采用 GRU，则单步输入维度不变，但训练时按时间窗组织样本。

## 6. 推荐的 V1 输出定义

V1 建议采用多头输出，而不是直接预测完整下一帧 LiDAR。
这里的 `Δ` 表示落在 `0.3~0.8 s` 内的短时预测间隔。

### 6.1 Ego dynamics head

输出：

- `delta_v_long`
- `delta_yaw_rate`
- `delta_accel_x`

### 6.2 Opponent state head

输出最近关键对手在 `t+Δ` 的相对状态：

- `next_rel_longitudinal`
- `next_rel_lateral`
- `next_rel_speed_longitudinal`
- 可选 `next_rel_speed_lateral`

### 6.3 Gap head

输出：

- `gap_left_t+Δ`
- `gap_right_t+Δ`

### 6.4 Safety head

输出：

- `collision_prob`
- `ttc_proxy`

### 6.5 Overtake opportunity head

输出一个或多个超车相关代理量：

- `passable_left`
- `passable_right`
- `closing_rate`
- `overtake_progress_gain`

这里的关键不是“世界长什么样”，而是“当前动作会不会让超车更容易完成”。
V1 不直接预测“最终是否超车成功”，因为这个标签在真实车上太依赖长时策略和上下文。

## 7. 为什么不建议 V1 直接预测 next LiDAR

理论上可以让模型直接预测：

- `next_lidar_range[36]`
- 或局部 `BEV occupancy`

但对当前任务，第一步不建议这么做。

原因：

- 标签维度更高
- 时序误差更容易累积
- 训练更难
- 真实车标注更脏
- 对 PPO / safety filter 的直接收益不如摘要量明显

对超车任务来说，先预测“关键对手 + 左右缺口 + 风险”更划算。

## 8. 标签设计

### 8.1 自车标签

沿用现有 ego world model 目标，但都定义在 `t -> t+Δ` 的短时窗口上：

- `delta_v_long`
- `delta_yaw_rate`
- `delta_accel_x`

### 8.2 对手标签

#### sim 侧

sim 里优先使用真值相对位姿监督。

你当前代码中已经有对手注入链路，能构造：

- `obstacle_longitudinal`
- `obstacle_lateral`
- `obstacle_dist`
- `obstacle_risk`

但对超车来说，建议补成：

- `rel_longitudinal`
- `rel_lateral`
- `rel_speed_longitudinal`

也就是从“障碍摘要”升级成“对手相对状态摘要”。

#### real 侧

real 侧不具备真值对手位姿，建议先用 LiDAR 时序跟踪产生伪标签。

这里必须把异步问题当成主风险，而不是数据清洗细节。当前 real 记录里：

- LiDAR 频率约 `7 Hz`
- 相机 / 控制主循环约 `16.6 FPS`
- `scan_age_ms` 中位数约 `228.8 ms`
- 同一个 scan 被复用的中位次数约 `3`

因此 V1 训练样本必须：

- 按 monitor 时间戳和 scan identity 对齐
- 显式保留 `lidar_age_t`
- 禁止把“当前图像 / 当前动作 / 当前 LiDAR”当成严格同步样本

V1 不做全多目标跟踪，先做：

- 前向扇区中最相关目标
- 最近前向目标的短时跨帧匹配
- 从相邻帧差分估计 `rel_speed_longitudinal`

### 8.3 缺口标签

gap 标签建议直接从 canonical LiDAR 提取：

- `left_gap`
- `right_gap`

定义应与车宽和安全冗余相关，而不是简单使用最小距离。

推荐：

- 在左前 / 右前角度窗口内，计算可通行宽度代理
- 或计算“离最近障碍的横向余量”

### 8.4 超车代理标签

真实车上很难直接监督“最终会不会超车成功”，所以 V1 可以先用代理标签：

- `passable_left/right`
- `closing_rate`
- `overtake_progress_gain`

其中：

- `closing_rate > 0` 表示正在追近
- 某侧 `gap` 足够且风险不高，说明该侧更可能可超

## 9. 数据计划

### 9.1 真实车数据

当前已经具备：

- tub 图像与控制
- RP2040 自车状态
- monitor CSV
- raw LiDAR JSONL

必须显式承认 real 是异步数据源。任何默认同频同步的 dataset 构造，都会直接污染 `closing_rate`、gap 演化和碰撞代理标签。

需要新增的预处理：

1. 对齐 tub 与 monitor 行
2. raw `/scan` -> canonical LiDAR
3. 跟踪 scan 复用关系并保留 `lidar_age`
4. canonical LiDAR -> gap / 对手伪标签
5. 基于真实时间戳构造时间窗样本
6. 构造 `0.3~0.8 s` 短时 transition 对

### 9.2 sim 数据

sim 数据建议同时导出：

- ego dynamics
- canonical LiDAR
- 对手真值相对状态
- 超车相关代理量

建议在 sim 中显式覆盖：

- 追近对手
- 并行跑位
- 左右不同超车窗口
- 静态障碍干扰
- lane-pid 对手车

这样训练出来的模型才是真正超车导向，而不是障碍导向。

## 10. 推荐的模型结构

### 10.1 推荐的 V1

建议使用多分支网络：

- ego branch：处理 8D dynamics/control
- lidar temporal branch：处理最近 `K` 帧 canonical LiDAR
- fusion trunk：融合 ego 与 lidar 时序信息
- 多个输出 head：ego / opponent / gap / safety / overtake

示意：

```text
ego(8) -> MLP(64)
lidar_seq(K x 73) -> GRU(128)
concat -> MLP(128 -> 128)
 -> ego head (3)
 -> opponent head (3~4)
 -> gap head (2)
 -> safety head (2)
 -> overtake head (3~4)
```

### 10.2 为什么建议 GRU

超车是时序任务。

单帧 LiDAR 很难判断：

- 对手是不是在减速
- 当前 closing rate 是正是负
- gap 是在变宽还是变窄

因此 V1 就建议加入轻量时间编码器，而不是纯单帧 MLP。

### 10.3 V1 不推荐

- 直接上 Dreamer / RSSM
- 直接做 full next-lidar prediction
- 直接做 image + lidar 的完整 latent world model

这些方案太重，不适合当前任务的第一步。

## 11. 训练损失

建议总损失：

```text
L =
  w_ego * L_ego
  + w_opp * L_opponent
  + w_gap * L_gap
  + w_safe * L_safety
  + w_ot * L_overtake
```

推荐：

- `L_ego`：weighted MSE
- `L_opponent`：masked MSE
- `L_gap`：MSE
- `L_safety`：BCE 或 focal loss
- `L_overtake`：BCE / regression，取决于标签定义

重要点：

- 当没有可靠目标时，不强行监督对手状态
- 使用 mask 控制无目标样本

## 12. 与 PPO 和部署的关系

### 12.1 MVP 期间的原则

不要一上来让 world model 替代 PPO。

推荐顺序：

1. PPO 先吃 canonical LiDAR
2. 单独训练超车导向 local world model
3. 先离线证明它对 risk / gap / opportunity 有预测价值
4. MVP 内不让它接管控制，也不把它直接喂回 PPO

### 12.2 MVP 之后的运行时接入

Phase 3 之后如果上车验证，首版只能是 `log-only`：

```text
policy action
-> action adapter
-> overtaking-aware local world model rollout
-> risk / gap / opportunity evaluation
-> log-only score dump
```

只有在 `log-only` 证明稳定后，才允许非常轻的动作修正：

- throttle 缩放
- lane bias 微调
- 短时 veto 明显危险动作

不允许直接输出全新的 `steer/throttle`，否则会变成两套控制器打架。

### 12.3 对 PPO 的后续融合方式（非 MVP）

后续可以把世界模型输出作为额外特征给策略：

- `pred_next_rel_longitudinal`
- `pred_next_rel_speed_longitudinal`
- `pred_left_gap`
- `pred_right_gap`
- `pred_collision_prob`
- `pred_overtake_gain`

这些量更贴近超车任务本身。

## 13. MVP 清单（只保留 Phase 0 到 Phase 2）

### Phase 0：LiDAR 规范化基础层

- [ ] 固化 canonical 规格：`36` sectors、`sector_0 = front`、`max_range`、`near_clip`、`lidar_range + lidar_valid + lidar_age`
- [ ] 提供 real raw `/scan` -> canonical 的离线生成器
- [ ] 提供 sim lidar -> canonical 的同规格适配器
- [ ] 在训练表中保存时间戳、scan identity、`lidar_age`
- [ ] 构造按真实时间对齐的 `K=3~5` 步时间窗样本

退出条件：

- sim / real 产生统一的 canonical LiDAR 张量定义
- 训练代码不再假设 `1 张图像 == 1 帧新 LiDAR`
- 能从真实日志稳定重建 scan 复用和 `lidar_age`

### Phase 1：超车标签层

- [ ] 最近关键对手提取逻辑
- [ ] left / right gap 提取逻辑
- [ ] `closing_rate` 估计
- [ ] `passable_left/right` 标签逻辑
- [ ] 无可靠目标时的 mask 逻辑
- [ ] sim 真值切片与 real 伪标签切片的人工抽检脚本

退出条件：

- 能稳定产生 `rel_longitudinal / rel_lateral / rel_speed_longitudinal`
- 能稳定产生 `left_gap / right_gap / passable_left / passable_right`
- 关键标签在代表性片段上通过人工抽检，不依赖最终“是否超车成功”标签

### Phase 2：超车导向局部世界模型

- [ ] 扩展后的 dataset：`ego(8D) + canonical_lidar_seq + opponent/gap/safety/overtake labels`
- [ ] 轻量 GRU 版本 world model
- [ ] 离线训练脚本
- [ ] 离线评估脚本
- [ ] 至少一组 trivial baseline 对比

退出条件：

- ego 预测稳定
- 对手相对状态预测有意义
- gap 预测误差可控
- `passable_left/right` 和 `collision_prob` 对关键片段有可解释的变化
- 在离线日志里，near-collision 之前风险分数应提前上升
- 在离线日志里，可超窗口出现前 `passable_left/right` 应提前变化

### 不属于 MVP 的内容

- Phase 3：车上 `log-only` 验证
- Phase 4：在线 safety / bias 与 PPO 融合

## 14. 验证指标

建议跟踪：

- ego 单步 RMSE
  - `v_long`
  - `yaw_rate`
  - `accel_x`
- 对手状态误差
  - `rel_longitudinal`
  - `rel_lateral`
  - `rel_speed_longitudinal`
- gap 误差
  - `left_gap`
  - `right_gap`
- 安全指标
  - collision AUC
  - TTC error
- 超车指标
  - `passable_left/right` 分类精度
  - `closing_rate` 误差
  - near-collision 前风险分数的提前量
  - 成功窗口与错失窗口上的 `overtake_progress_gain` 排序能力

## 15. 主要风险

### 15.1 真实车标签不干净

real 侧对手状态是 LiDAR 伪标签，不是地面真值。
而且它还是异步伪标签，不是同步真值：LiDAR 约 `7 Hz`，控制主循环约 `16.6 FPS`，`scan_age_ms` 中位数约 `228.8 ms`。

缓解方式：

- 先把 horizon 限制在 `0.3~0.8 s`
- 严格按时间戳和 scan 复用关系构造样本
- 保留 `lidar_age`
- 只先跟踪最近关键对手
- 使用时间平滑
- 在 sim 上先训出结构，再用 real 微调
- 只做代理量监督，不做“最终超车成功率”监督

### 15.2 单目标摘要可能不够

在复杂交互中，左右可能同时存在多个目标。

缓解方式：

- V1 先做关键目标
- V2 再扩成 top-K object slots

### 15.3 sim / real 运动模式差异

sim 中对手车行为可能比真实车规则很多。

缓解方式：

- 加入 lane-pid、jitter、nudge、多种速度差配置
- 增强对手行为随机性

## 16. 推荐的第一版实现

最佳第一版不是“通用障碍世界模型”，而是：

- `36` 维 canonical LiDAR
- 带短时间窗
- `0.3~0.8 s` 短时预测
- 面向最近关键对手
- 预测左右 gap、risk 与超车代理量
- 先离线证明预测价值

这才与最终“超车”任务一致。

## 17. 需要重点扩展的文件

MVP 高概率会优先改这些位置：

- `mysim/module/world_model.py`
- `mysim/module/world_model_dataset.py`
- `DonkeyCar-RL-Racing/module/multi_scene_env.py`
- `DonkeyCar-RL-Racing/src/ppo_multitrack_v16.py`

`mysim/module/predictive_safety_filter.py` 和策略侧融合代码属于 MVP 之后。

## 18. 最终建议

如果最终任务是超车，那么世界模型不应停留在“障碍风险预测”。

第一版最值得做的是：

- short horizon
- local
- temporal
- ego + opponent + gap
- overtaking-aware

也就是：

**不是一个酷炫的完整 world model，而是一个保守、局部、短时、任务导向的交互预测器。**
