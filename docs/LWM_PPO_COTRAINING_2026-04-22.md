# LWM + PPO 协同训练路线图（April 22, 2026）

## 1. 定位

这份文档描述的是一条 **长期可行、分阶段推进** 的 `V17` 路线，而不是“当前仓库已经完整闭环的训练主线”。

当前判断是：

- `LWM + PPO` 这条线作为长期方向是可行的
- 但当前仓库里真正已经落地的是 `Phase 0-2`
- `Phase 3+` 仍然需要新增 `PPO <- lwm_summary` 在线接口后才能实施

一句话总结：

- `PPO` 继续做主控制器
- `interaction LWM` 做局部交互预测器
- `real ego dynamics model` 独立存在，不并入当前主 `LWM`
- 训练主线先按“离线 interaction LWM + 在线 PPO”推进
- 等在线摘要接口补齐后，再进入 `critic-only -> actor+critic -> alternating` 闭环


## 2. 当前结论

这条线要解决的问题不是“做一个更大的 world model”，而是：

- 让 `V17` 更快学会对手车交互
- 让 `V17` 更快形成超车策略
- 降低 `PPO` 从原始观测硬学局部交互动力学的样本复杂度

当前最重要的边界条件有三个：

1. 离线 `LWM` 导出和训练链路已经存在，可以继续推进。
2. `PPO <- lwm_summary` 在线接入还没有现成接口，不能写成已经具备的能力。
3. `Phase F` 的 strict raw-domain LiDAR sim-real realism 还未通过，但 deployment-oriented 粗对齐 gate 已通过，因此当前可以继续推进主线，但仍不能把 LiDAR 默认描述成 `gap/passable` 的唯一主语义来源。

因此，当前推荐表述必须改成：

- `camera` 主导 `gap/passable`
- `LiDAR` 主导 `target/safety`
- `real LiDAR` 先作为参考集和 stress 集，不直接进入 round1 主监督训练

### 2.1 当前冻结的 Sim LiDAR 基线

此前一轮集中 debug 后，当前仓库应冻结在下面这套 sim LiDAR 基线，不再回退到更早的低位/短量程假设：

- 安装位姿：`offset_y = 0.40`
- 安装位姿：`offset_z = 0.50`
- 安装位姿：`rot_x = 0.0`
- canonical / sim 最大量程：`max_range = 20.0m`
- packet 解释：`rx ~= 180 deg` 视为 ego 正前
- packet 距离缩放：`d / 8 -> telemetry meters`

当前这条 LiDAR 线的结论也应冻结为：

- 已经不再是“基本看不到障碍车”
- 当前主要剩余问题是 `primary target` 选择不稳，部分帧会被近侧结构带偏
- 因此 round1 不做 `wall vs car` 分类，只保留 `target/safety` 职责
- `camera` 继续承担 `gap/passable`

另外，当前 `Phase F` 的口径应明确拆开：

- `strict raw-domain Phase F`：仍未通过
- `deployment-oriented Phase F`：已通过，可放行当前主线

对应执行报告见：

- [PHASE_F_DEPLOYMENT_GATE_REPORT_2026-04-24.md](./PHASE_F_DEPLOYMENT_GATE_REPORT_2026-04-24.md)

### 2.2 当前推荐执行架构

当前最推荐的执行形式不是：

- `PPO + 完整 world model` 联合训练

而是：

- `PPO` 在线训练，继续做唯一主控制器
- `interaction LWM` 离线并行训练，作为局部交互预测器
- `real ego dynamics model` 用 real 数据独立训练
- 第一轮数据由成熟 `V16` 模型做 bootstrap rollout 提供
- 后续再逐步切到 `V17 PPO` 自己产 rollout 数据
- `interaction LWM` 周期性刷新，但当前不做 end-to-end joint train

当前更建议把主线拆成两个预测模块：

- 一个 `interaction-first local predictor`
- 一个独立的 `real ego dynamics model`

而不是：

- 一个追求统一 latent、imagined rollout、联合反传的大一统 world model

因此当前项目主线应固定为：

1. `V16` 先提供高质量 rollout
2. 第一版 `interaction LWM` 先离线训出来
3. `real ego dynamics model` 并行开始训练
4. `V17 PPO` 开始在线训练
5. 周期性导出新数据并刷新 `interaction LWM`
6. 在线接回时优先 `critic-only`


## 3. 已实现基线

这一节只记录仓库里已经具备的基线能力。

### 3.1 PPO 在线基线

当前 `V17` 在线主观测仍然是：

- `image`
- `state`
- `lidar`
- `lidar_meta`

这条主观测链继续服务当前 `PPO`。

### 3.2 离线 Interaction LWM 输入

当前 `interaction LWM` 输入是短时序：

- `ego_seq`
- `camera_seq`
- `lidar_seq`
- `async_meta_seq`

其中：

- `ego_seq` 作为 ego 上下文输入，帮助交互预测对齐自车状态
- `camera_seq` 负责局部可通行语义
- `lidar_seq` 负责局部目标和障碍几何
- `async_meta_seq` 负责处理 LiDAR 异步重复和 stale-scan 问题

保留 `async_meta_seq` 的原因不是抽象上的“也许有用”，而是当前 real 数据已经明确显示：

- `0421` 中每帧 LiDAR 平均被主循环复用约 `2.61x`
- `0422_1` 中每帧 LiDAR 平均被主循环复用约 `2.65x`

因此异步重复是当前系统的真实约束，不应删除。

### 3.3 离线 Interaction LWM 输出

当前主线只保留两组输出头：

#### Interaction head

- `target_rel`
- `closing_rate`
- `overtake_progress`

#### Safety / passability head

- `gap`
- `collision_logit`
- `ttc_proxy`
- `passable_logits`

### 3.4 独立 Real Ego Dynamics Model

当前推荐把 `ego dynamics` 从主 `LWM` 里拆出去，单独做成 real-first 模型。

原因：

- `ego dynamics` 更适合直接吃 real 数据
- `interaction` 更适合继续吃 sim runtime truth
- 二者硬塞进一个统一 `LWM` 会增加数据源和目标定义复杂度

因此当前推荐形式是：

- `interaction LWM`：预测 `target_rel / closing_rate / overtake_progress / gap / collision / ttc / passable`
- `real ego dynamics model`：独立预测自车短时动力学，不并入当前主线 `lwm_summary`

### 3.5 当前推荐语义分工

当前更合理的语义分工不是“LiDAR 包办一切”，而是：

- `camera` 主要负责 `gap/passable`
- `LiDAR` 主要负责 `target_rel / collision / TTC / enclosure / closing_rate`

也就是说：

- 保留 “LiDAR 继续主导 interaction”
- 但不再把 LiDAR 写成完整 `passability` 语义的唯一来源
- 当前 round1 不实现 `wall vs car` 区分，只做几何目标与安全摘要

关于这个调整，可参考 [CAMERA_FREESPACE_LIDAR_TOKENS_2026-04-22.md](./CAMERA_FREESPACE_LIDAR_TOKENS_2026-04-22.md)。


## 4. 为什么不是现在就做完整闭环

如果只训练 `PPO`：

- 它要自己从原始 `image + state + lidar + lidar_meta` 里学出局部交互动力学
- 可以学，但慢，而且对样本敏感

如果只训练 `interaction LWM`：

- 它能预测局部局势
- 但不会直接产出控制动作

如果现在就让 `PPO + interaction LWM` 完全 joint train：

- credit assignment 会很乱
- 训练不稳定
- 很难区分是控制器还是预测器出了问题

如果现在把 `Phase 3+` 写成“已经可直接落地”：

- 会掩盖一个关键事实：当前代码里还没有真正的 `lwm_summary` 在线接入接口
- 会让实现者误以为 `critic-only` 只需要改几行配置

所以当前更合理的文档口径是：

- `Phase 0-2` 是当前可落地子集
- `Phase 3+` 是明确设计好的下一阶段计划


## 5. 分阶段路线

### 5.1 总体原则

训练不采用“一次训完”，而采用分阶段路线：

1. 先用成熟 `V16` 做 bootstrap rollout
2. 导出第一版离线 `interaction LWM` 数据
3. 训练第一版 `interaction-first LWM`
4. 并行训练 `real ego dynamics model`
5. 启动 `V17 PPO` 在线训练
6. 周期性导出新 rollout 并离线刷新 `interaction LWM`
7. 补齐在线 `lwm_summary` 接口
8. 再做 `critic-only`
9. 只有当 `critic-only` 明确有效后，再进入 `actor+critic`
10. 最后再考虑 alternating 闭环

这条路线强调：

- `PPO` 在线、`interaction LWM` 离线并行
- `real ego dynamics model` 作为独立支线并行推进
- 先做可验证的最小闭环
- 再做更强的协同训练

### 5.2 Phase 0: `V16` Bootstrap 采样预热

目标：

- 用成熟 `V16` 模型先产生第一轮高质量交互数据

要求：

- 优先覆盖静态障碍、动态对手、lane-pid 对手、超车场景
- 不要求这一步就是最终 `V17 policy`
- 关键是先把高质量 `interaction` 数据拿到

输出产物：

- `v16 bootstrap rollout`
- 第一版 `interaction LWM` 导出数据
- 数据分布统计

### 5.3 Phase 1: 导出第一版离线 Interaction LWM 数据

使用 `Phase 0` 的 `V16` bootstrap rollout 导出：

- `camera`
- `ego8`
- `lidar`
- `async_meta`
- 所有 `target_*`

数据应覆盖：

- `generated_track`
- `waveshare`
- 静态障碍
- 动态对手
- 跟车
- 试探性超车
- 失败超车

当前 round1 主数据仍以仿真带真值数据为主，不把 real LiDAR 直接并入主监督训练。

这一阶段的关键不是“先训练新的采样器”，而是：

- 先复用现有成熟 `V16` 模型
- 通过 mixed rollout 拿到第一轮 `interaction-first` 数据
- 只有当第一轮数据里稳定跟车/交互片段明显不足时，再考虑补一个专用 follow-only 采样模型

### 5.4 Phase 2: 训练第一版离线 Interaction LWM

训练顺序继续采用 staged training。

#### Stage A

先学基础交互几何：

- `target_rel`
- `gap`

目标：

- 学稳局部运动和几何结构

#### Stage B

再学交互机会和风险：

- `collision_logit`
- `ttc_proxy`
- `passable_logits`
- `closing_rate`
- `overtake_progress`

这里的关键是：

- `camera` 分支主要在这一阶段开始发挥价值
- 它应主要帮助 `gap/passable`

当前 round1 的实际优先级应明确为：

1. `interaction head`
2. `safety/passability head`

也就是说：

- 当前真正最值钱的是 `interaction-first local predictor`

#### Stage C

再做联合微调，但保留 guard，防止几何退化。

输出产物：

- `stage_a_best.pth`
- `stage_b_best.pth`
- `local_world_model_v17_round1_final.pth`
- `train_summary_round1.json`

#### 并行支线：训练独立 Real Ego Dynamics Model

这条支线不并入当前主 `interaction LWM`，而是单独训练。

推荐输入：

- `speed_odom`
- `heading_rate_deg` 或 `gyro_z`
- `accel_x`
- `user/angle`
- `user/throttle`
- `prev_user_angle`
- `prev_user_throttle`
- `dt`

推荐目标：

- `delta_v_long`
- `delta_yaw_rate`
- `delta_accel_x`

当前推荐数据划分：

- `0421` 作为主训练/验证集
- `0422_1` 作为 stress/test 集

### 5.5 Phase 3: 补齐在线 `lwm_summary` 接口

这一阶段是 `Phase 3+` 的前置依赖。

在没有这一层之前，不进入 `critic-only` 主实验。

这一阶段需要新增：

- `obs["lwm_summary"]`
- `obs["lwm_valid"]`

第一版 `lwm_summary` 固定为 `12D`：

- `pred_target_rel` 4 维
- `pred_gap` 2 维
- `pred_passable` 2 维
- `pred_ttc` 1 维
- `pred_collision` 1 维
- `pred_closing_rate` 1 维
- `pred_overtake_progress` 1 维

第一版不把 `real ego dynamics model` 的输出接回 `PPO`。

在线侧需要的最小机制是：

- 在 env 或 runner 侧为每个并行环境维护 `seq_len=4` 的 ring buffer
- 缓存 `ego / lidar / async_meta / camera`
- 使用冻结的 `interaction LWM` 做在线前向
- 把预测结果压成低维 `lwm_summary`

冷启动和异常时统一采用保守回退：

- `seq_len` 不足时：`lwm_summary = 0`，`lwm_valid = 0`
- `LWM` 推理失败时：`lwm_summary = 0`，`lwm_valid = 0`
- 输入缺帧或 buffer 失效时：`lwm_summary = 0`，`lwm_valid = 0`

这一阶段还要明确一个实现事实：

- `critic-only` 不是“把 summary 直接塞给现有 critic 就完了”
- policy 侧需要新增 critic 专用 summary encoder，或 critic MLP 后拼接路径

也就是说，`critic-only` 本身就是一个明确的代码增量，不是纯配置切换。

### 5.6 Phase 4: PPO + Interaction LWM（critic-only）

只有在 `Phase 3` 接口补齐后，才进入这一步。

接法：

- actor 仍然只看原始观测
- critic 看原始观测 + `lwm_summary`

目标：

- 验证 `interaction LWM` 是否真的提高 value 学习质量
- 验证它是否能提升超车相关学习速度

观察指标至少包括：

- value loss 是否更稳
- 超车相关 reward 是否更快上涨
- 首次成功超车环境步数是否下降

输出产物：

- `ppo_v17_lwm_round1_critic_only.zip`

### 5.7 Phase 5: PPO + Interaction LWM（actor + critic）

只有在 `critic-only` 明确有效后，才进入这一步。

接法：

- actor 看原始观测 + `lwm_summary`
- critic 也看原始观测 + `lwm_summary`

训练时：

- `interaction LWM` 冻结
- `PPO` 学习如何使用预测摘要

风险：

- 对 `interaction LWM` 稳定性要求更高
- 更容易把预测误差直接传到动作分布

输出产物：

- `ppo_v17_lwm_round1_actor_critic.zip`

### 5.8 Phase 6: 进入 alternating 闭环

只有当前面两步已经证明 `interaction LWM summary` 有收益时，再进入真正的闭环：

1. 新 PPO rollout
2. 刷新 `interaction LWM`
3. 新 `interaction LWM` 接回 PPO

直到：

- 超车学习速度不再明显提升
- 或 `interaction LWM summary` 对 `PPO` 的边际收益趋近于零


## 6. 当前不建议做的事

当前不建议：

- 从零开始完全 joint train `PPO + interaction LWM`
- 把 `real ego dynamics` 强行并回当前 `interaction LWM`
- 把完整 `LWM latent` 大向量直接拼进 `PPO`
- 一开始就做 imagined rollout 再回传给 `PPO`
- 让 `LWM` 直接输出动作
- 让 `PPO` 兼做交互预测监督头
- 把 real LiDAR 直接混入 round1 主监督训练

原因不是这些方向永远不值得做，而是：

- 太重
- 太不稳
- 诊断困难
- 当前还没有必要越过最小闭环


## 7. Real 数据定位

当前 real 数据不作为 round1 `interaction LWM` 主监督训练集，但应作为 `real ego dynamics model` 的主数据源，同时继续承担设计约束、验收数据和 stress 数据角色。

### 7.1 `0421` 的角色

`0421` 是当前推荐的正常参考集。

主要用途：

- canonical LiDAR 分布参考
- async reuse 现象确认
- `near_clip` 阈值 sanity check
- sim-real gap 的正常工况参考

### 7.2 `0422_1` 的角色

`0422_1` 是当前推荐的 stress 集。

已知特点：

- 主要是开阔场地 LiDAR 采样
- 存在较多超量程
- 近场异常和空值率明显高于 `0421`
- 当前只有 `1` 个 catalog，tub 记录约 `349` 条，规模也明显偏小
- 不适合承担 round1 主监督职责

主要用途：

- 超量程鲁棒性检查
- 近场伪点鲁棒性检查
- 空值率和开阔场地分布 stress check

### 7.3 当前为什么不把 real 数据并入 Interaction LWM 主训练

当前 real LiDAR 数据没有与 `interaction LWM` 主任务严格对齐的 `target_*` 真值：

- 没有可靠的 `target_rel`
- 没有可靠的 `target_collision`
- 没有可靠的 `target_ttc`
- 没有可靠的 `target_passable`
- 没有可靠的 `target_overtake_progress`

因此当前不建议：

- 用规则伪标签直接混入 round1 主训练
- 在没有额外标注或弱监督设计前，把 real 数据当作和 sim 同等级的监督样本

但这不影响另一条高 ROI 支线：

- 用 real 数据单独训练 `real ego dynamics model`

后续如果继续采集 real 数据，可以单列一条支线：

- real adaptation
- weak-label / pseudo-label
- representation alignment

但这条支线不属于当前 round1 主线的必要前提。


## 8. Gate 与验收顺序

在进入 `Phase 3` 之前，当前推荐新增三个 gate。

### 8.1 Gate A: dataset / training readiness

继续复用现有 readiness 体系，至少检查：

- 数据键完整
- scene 覆盖完整
- `collision/passable/opportunity` 分布不过偏
- `Stage A/B/C` 指标达标

当前离线 `LWM` 自身指标至少看：

- `mae_target_rel`
- `mae_gap`
- `loss_passable`
- `loss_ttc`
- `loss_closing`

独立 `real ego dynamics model` 则单独看：

- `mae_delta_v`
- `mae_delta_yaw_rate`
- `mae_delta_accel_x`

并继续用几何 guard 检查：

- `Stage B` 对 `Stage A` 的几何退化率
- `Stage C` 对前序最优几何指标的回退幅度

### 8.2 Gate B: Phase F LiDAR realism gate

`Phase F` 现在要分成两种口径：

- `strict raw-domain Phase F`：仍然没有通过
- `deployment-oriented Phase F`：已经通过

这意味着当前项目应按下面的方式理解 Gate B：

- LiDAR 仍然不能被当作“已经足够真实、可承载全部 passability 语义”的输入
- `camera 主导 passability，LiDAR 主导 target/safety` 的语义分工必须保留
- 但当前主线已经不需要继续被 strict raw-domain 指标阻塞

因此，当前不建议把文档写成：

- LiDAR 单独承担完整局部通行语义

当前 `Phase F` 的正式执行口径应固定为：

- `0421` 是唯一硬门禁参考集
- `0422_1` 只做 stress 报告，不参与 pass/fail
- sim 收集仍使用冻结 LiDAR 基线：
  - `offset_y=0.40`
  - `offset_z=0.50`
  - `rot_x=0.0`
  - `collect max_range=20.0`
- 正式 eval 使用 `compare_max_range=12.0`
- strict raw-domain 继续输出：
  - `0–5m` 近场带的 `valid_ratio_mae / wasserstein_median / wasserstein_p95`
  - `overall_0_12m.scene_js_divergence`
- deployment-oriented gate 改成：
  - `motion_enough`
  - `front_min / left_gap / right_gap / valid_ratio` 的近场特征对齐

此外，`Phase F` 现在不是“直接 baseline -> pose sweep”的两步，而是三步：

1. baseline rerun
2. motion-profile calibration
3. pose sweep

其中：

- 如果 motion-confound 失败，不直接做 pose sweep
- 先扫现有 `V16` driver profile
- 只有达到 `motion_enough` 后，才允许进入 pose sweep

当前最新执行结论见：

- [PHASE_F_DEPLOYMENT_GATE_REPORT_2026-04-24.md](./PHASE_F_DEPLOYMENT_GATE_REPORT_2026-04-24.md)

相关现状可参考 [PHASE_F_CURRENT_ISSUES_2026-04-22.md](./PHASE_F_CURRENT_ISSUES_2026-04-22.md) 和 [LIDAR_DEBUG_RECORD_2026-04-24.md](./LIDAR_DEBUG_RECORD_2026-04-24.md)。

### 8.3 Gate C: `lwm_summary` online interface gate

在开始 `critic-only` 之前，必须先通过在线接口 gate。

至少要完成：

- 单环境 smoke 测试
- 多环境 rollout 测试
- reset / done 后 ring buffer 正确清空
- stale scan、冷启动、LWM 失效时 fallback 正确
- `lwm_valid` 可以稳定表达“当前摘要是否可信”


## 9. 实验矩阵

实验顺序固定为：

1. `PPO baseline`
   - 无 `interaction LWM`

2. `PPO + interaction LWM (critic-only)`
   - actor 不看 `interaction LWM`
   - critic 看 `lwm_summary`

3. `PPO + interaction LWM (actor+critic)`
   - actor / critic 都看 `lwm_summary`

4. `PPO + interaction LWM (no camera branch)`
   - 用于验证 `camera` 是否真的帮助 `gap/passable`

5. `PPO + interaction LWM (camera branch on)`
   - 验证 `camera` 辅助的实际收益

`real ego dynamics model` 不在这组在线实验矩阵里，它作为独立支线单独训练和评估。

顺序要求是：

- 不跳过 `baseline`
- 不跳过 `critic-only`
- `actor+critic` 不作为 round1 默认


## 10. 验收指标

### 10.1 离线 Interaction LWM 指标

至少看：

- `mae_target_rel`
- `mae_gap`
- `loss_passable`
- `loss_ttc`
- `loss_closing`

### 10.2 独立 Real Ego Dynamics 指标

至少看：

- `mae_delta_v`
- `mae_delta_yaw_rate`
- `mae_delta_accel_x`

### 10.3 在线接入 smoke 指标

至少看：

- buffer 未满时 `lwm_valid == 0`
- reset 后历史状态被正确清空
- scan 重复时 `async_meta` 仍能驱动稳定前向
- `LWM` 失效时 summary 正确归零
- 多环境并行时不同 env 的 `lwm_summary` 不串线

### 10.4 PPO 提升指标

真正必须看的不是 `LWM` loss 有多漂亮，而是加入 `LWM summary` 后：

- 超车成功率是否提升
- 首次成功超车所需环境步数是否下降
- close interaction reward 收敛速度是否提升
- near-collision 率是否下降
- 跟车到超车的转化率是否提升

最核心的问题只有一个：

- 加入 `LWM summary` 后，`PPO` 是否更快学会超车


## 11. 当前推荐结论

当前推荐主线不是“已经完整落地的统一 `LWM + PPO` 联训体系”，而是：

1. 保留 `PPO` 作为当前主控制器。
2. 保留 `interaction LWM` 作为离线训练的局部交互预测器。
3. 单独训练 `real ego dynamics model`，不把它并入当前主 `LWM`。
4. 先把 `Phase 0-2` 做稳，形成可靠的离线 `interaction LWM` 基线。
5. 再补齐 `obs["lwm_summary"] + obs["lwm_valid"]` 在线接口。
6. 先做 `critic-only`，确认有收益后再做 `actor+critic`。
7. 只有当前面都验证成立后，才进入真正的 alternating 闭环。
8. `camera` 主导 `gap/passable`，`LiDAR` 主导 `target/safety`。
9. `0421` 作为正常参考集，同时作为 `real ego dynamics model` 主训练源；`0422_1` 作为 stress 集，不进入 round1 `interaction LWM` 主监督训练。

一句话总结：

**interaction LWM 不是 PPO 的替代品；在当前阶段，它首先是离线训练的局部交互预测器，之后才是 PPO 的交互学习加速器。与此同时，real ego dynamics model 作为独立支线推进，不再强行并入统一 LWM。**
