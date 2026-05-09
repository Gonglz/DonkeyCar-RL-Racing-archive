# LiDAR Debug 纪要（2026-04-24）

## 1. 目的

这份文档用于记录本轮 `sim LiDAR -> V17/LWM` 调试中已经确认的事实、已经落地的修复、当前冻结基线，以及仍然没有解决的边界。

目标不是复述所有试验过程，而是回答下面四个问题：

- 这轮到底修好了什么
- 当前默认应该用哪套 LiDAR 配置
- 还有哪些问题没解决
- 这些结论会如何影响 `V17 / LWM / PPO` 主流程


## 2. 本轮最终结论

一句话总结：

- `sim LiDAR` 这条线已经从“基本看不到障碍车”推进到了“可以用于 `target/safety`，但 `primary target` 选择还不够稳”

当前项目级结论应固定为：

- `camera` 主导 `gap/passable`
- `LiDAR` 主导 `target/safety`
- round1 不做 `wall vs car` 分类
- 当前残留问题不是“原始回波完全不可用”，而是“目标关联和主目标选择不稳定”


## 3. 已确认的关键事实

### 3.1 Real LiDAR 量程与数据角色

当前 real 端 monitor 日志显示：

- `0421` 名义量程为 `12.0m`
- `0422_1` 名义量程也为 `12.0m`

但两批数据的角色不同：

- `0421` 是正常参考集
- `0422_1` 是 stress 集

当前更合理的使用方式是：

- `0421` 用于正常分布参考、async reuse 现象确认、`near_clip` sanity check
- `0422_1` 用于超量程、近场异常、空值率 stress check
- 两者都不直接进入 round1 主监督训练


### 3.2 之前“看不到车”的结论不完全正确

最初排查时，曾一度得到“sim 里 LiDAR 基本识别不到第二台车/障碍车”的结论。

后续更深入排查后确认：

- Unity 侧并不是完全没有障碍车回波
- Python 侧对 `lidar_raw_packet` 的解释存在问题
- 修正 packet 解释后，LiDAR 对障碍车/对手车的观测已经不是全盲

因此当前正确表述应为：

- 不是“完全看不到”
- 而是“能看到，但在部分场景中主目标会被近侧结构带偏”


### 3.3 当前不需要做 `wall vs car` 分类

从本轮排查看，当前主线不需要把问题定义成“LiDAR 必须区分墙和车”。

原因是：

- 对 `safety` 来说，墙和车本质上都是要避开的占据物
- 对 `interaction` 来说，真正重要的是“前景主目标 vs 背景结构”
- 在当前单层、36-sector、近场 token 的表示下，强做 `wall vs car` 分类收益不高

因此当前策略是：

- 不做 `wall vs car`
- 只做 `safety token + primary target token`


## 4. 这轮真正修好的东西

### 4.1 Packet 路径解释修正

当前 `module/lidar.py` 已经固定为下面这套解释：

- 使用 Unity `lidar_raw_packet`
- 将 packet 视为完整 `360 deg` 水平扫描
- 以 `rx ~= 180 deg` 作为 ego 正前
- 将 packet 距离按 `d / 8` 换回 telemetry meters

这一步是本轮最关键修复之一。

修正后：

- 侧向目标已经可以进入 canonical LiDAR / target token
- 前向目标在真实 `v16 + gt + avoid_mixed + 双障碍` 轨迹里也已出现有效命中


### 4.2 主链路默认 LiDAR 配置统一

当前主链路已统一到这套 sim LiDAR 基线：

- `offset_y = 0.40`
- `offset_z = 0.50`
- `rot_x = 0.0`
- `max_range = 20.0m`

这套基线已经在以下路径中统一：

- `src/ppo_multitrack_v17.py`
- `module/v17_env.py`
- `scripts/export_world_model_dataset.py`
- `scripts/collect_sim_lidar_monitor.py`
- `scripts/run_v17_formal_readiness.py`
- `scripts/v17_formal_readiness_manifest.json`
- `module/lidar.py`

说明：

- 早期排查里曾出现 `0.25 / 0.65 / 6m` 一类中间实验结论
- 那些结果现在应视为历史调试阶段结果，不再是当前项目级默认基线


### 4.3 `module/lidar.py` 内部默认 spec 已对齐

当前内部默认值已经对齐为：

- `CanonicalLidarSpec.max_range_m = 20.0`
- `CanonicalLidarSpec.invalid_fill_m = 20.0`
- `TargetTokenSpec.max_range_m = 20.0`

同时保留：

- `TargetTokenSpec.max_target_range_m = 5.0`

这表示：

- canonical/safety 路径按 `20m` 工作
- primary target 仍然只聚焦更近的交互目标


### 4.4 文档口径已同步

目前几份关键文档已经对齐到新结论：

- `LWM_PPO_COTRAINING_2026-04-22.md`
- `CAMERA_FREESPACE_LIDAR_TOKENS_2026-04-22.md`
- `PHASE_F_CURRENT_ISSUES_2026-04-22.md`

当前文档层面已明确：

- `camera -> gap/passable`
- `LiDAR -> target/safety`
- 不做 `wall vs car`
- `PHASE_F_CURRENT_ISSUES_2026-04-22.md` 主要保留为 April 22 历史记录，不再当成当前默认配置说明


## 5. 试验中确认过、但最终没有成为主结论的事项

### 5.1 单纯调高/调低 LiDAR 位姿不是根因

这轮排查里做过多组高度、前后位置、俯仰、量程测试。

中间阶段曾经出现过以下判断：

- “LiDAR 太低”
- “LiDAR 太高”
- “是不是量程太短”
- “是不是障碍车 collider 太低”

最终收敛后的结论是：

- 这些都只解释了局部现象
- 真正影响主线的是 packet 解释、默认基线统一、以及主目标选择逻辑


### 5.2 Unity collider 补丁不是决定性突破

本轮还做过 Unity 侧尝试：

- 调整车辆 `hitBox/body` 的 collider 高度
- 重启 DonkeySim 后复测

结果表明：

- Unity collider 调整本身不是本轮决定性突破
- 关键突破仍然是 Python 侧 packet 解释修正

这意味着：

- 当前项目仍然可以继续保留 Unity patch 工具
- 但主线不应把希望寄托在“继续微调 collider 就会彻底解决”


## 6. 当前仍未解决的问题

### 6.1 `primary target` 选择还不够稳

这是当前 LiDAR 线最主要残留问题。

表现是：

- 在一部分 `frontish` 场景里，LiDAR 已经能打到障碍车
- 但在另一部分帧里，token 会被近侧围栏、边界、近结构带偏

因此当前问题的正确描述是：

- 不是“LiDAR 看不到障碍”
- 而是“主目标选择不稳定”


### 6.2 `Phase F` sim-real realism 仍未通过

尽管 packet 路径和当前默认基线已经修正，本轮仍不能宣称 `Phase F` 已通过。

当前仍需承认：

- sim LiDAR 与 real LiDAR 的分布差异仍存在
- 某些 sector 的有效率和距离分布仍不匹配
- sim-real realism 还没有达到 deployment-ready

因此当前仍应坚持：

- `camera` 不可退出 `gap/passable`
- `LiDAR` 不承担完整 passability 语义

当前正式 `Phase F` 口径应同步为：

- `0421` 是唯一硬门禁参考集
- `0422_1` 只做 stress 报告
- sim 收集保持 `max_range=20.0`
- 正式 eval 改为 `compare_max_range=12.0`
- 正式门禁优先看 `0–5m` 近场带，而不是单一整体 Wasserstein
- 在进入 pose sweep 之前，先做一次 `motion-profile calibration`
- 如果 motion-confound 失败，先扫现有 `V16` driver profile，而不是直接做 pose sweep


### 6.3 在线 `lwm_summary` 接口仍未补齐

这不是 LiDAR 本体问题，但会直接限制 `LWM + PPO` 进入下一阶段。

当前仍然缺：

- `obs["lwm_summary"]`
- `obs["lwm_valid"]`
- env/runner 侧 `seq_len=4` ring buffer
- `critic-only` 的专用接入路径

所以当前 LiDAR 线虽然已经可用，但还不能等价于“LWM 闭环已经 ready”。


## 7. 当前对 LWM / PPO 主流程的影响

### 7.1 对离线 LWM

当前 LiDAR 已经足够支撑第一版离线 LWM：

- `lidar_seq`
- `async_meta_seq`
- `target/safety` 相关监督

前提是：

- 继续保持 `camera -> gap/passable`
- `LiDAR -> target/safety`


### 7.2 对 PPO 主流程

当前 LiDAR 可以继续作为 `V17` 在线主观测的一部分。

但当前推荐的使用方式是：

- 用于几何安全
- 用于主目标 token
- 不扩展成“墙/车语义识别器”


### 7.3 对 round1 范围

round1 当前建议固定为：

- 允许：`LiDAR target/safety`
- 允许：离线 `LWM`
- 允许：`PPO baseline`
- 不做：`wall vs car`
- 不做：real LiDAR 主监督混训
- 不做：把 LiDAR 单独当成完整 `gap/passable` 来源


## 8. 当前冻结基线

除非后续有新的正式验证结论，否则当前项目级 LiDAR 基线固定为：

- 安装位姿：`offset_y = 0.40`
- 安装位姿：`offset_z = 0.50`
- 安装位姿：`rot_x = 0.0`
- 最大量程：`20.0m`
- packet 解释：完整 `360 deg`
- ego 正前：`rx ~= 180 deg`
- 距离换算：`d / 8 -> telemetry meters`

当前职责划分固定为：

- `camera -> gap/passable`
- `LiDAR -> target/safety`

当前暂不做：

- `wall vs car`
- `dynamic vs static`


## 9. 下一步最值得做的事

如果继续推进 LiDAR 主线，优先级应是：

1. 收紧 `TargetTokenBuffer._select_primary_cluster()` 的主目标选择逻辑
2. 重跑 `v16 + gt + avoid_mixed + 双障碍` 的审计验证
3. 用同一套基线重新导出 round1 的 LWM 数据
4. 再评估 `critic-only` 在线接口准备度

不建议当前继续投入时间的方向：

- `wall vs car` 分类
- 再做大范围位姿扫描
- 把 real LiDAR 直接混入主监督
- 继续把 LiDAR 逼成完整 passability 引擎


## 10. 结论

本轮 LiDAR 调试的真实产出不是“彻底解决了一切”，而是：

- 修掉了 packet 解释中的关键错误
- 把主链路默认配置统一到了可复现的基线
- 把项目口径稳定到了 `camera -> gap/passable`、`LiDAR -> target/safety`
- 把问题从“看不到障碍车”收敛成了“主目标选择还需继续稳定化”

因此当前 LiDAR 线已经可以进入 round1 主流程，但应按“几何安全 + 主目标跟踪”来用，而不是按“完整语义感知”来用。
