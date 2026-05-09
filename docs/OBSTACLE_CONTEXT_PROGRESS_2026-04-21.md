# Obstacle Context Progress Report

Date: 2026-04-21
Timezone: America/New_York
Branch: `feature/obstacle-context-temporal-sampling`

## 1. 目标

当前工作的目标是把 `V16` 里原先依赖 sim/runtime 真值注入的 obstacle context：

- `obstacle_present`
- `obstacle_longitudinal`
- `obstacle_lateral`
- `obstacle_dist`
- `obstacle_risk`

逐步替换成可部署路径：

1. 先验证 `CV` 是否能直接支撑。
2. 如果纯 `CV` 不稳，转成 `learned obstacle-context estimator`。
3. 最终保持 PPO 输入接口不变，仍然输出同一组 5 维 context。

## 2. 本轮代码工作

### 2.1 颜色与障碍车外观

- 将障碍车默认颜色固定为 `pink rgb(255,105,180)`。
- 更新了基于颜色的 detector，使其从旧的“绿色车”假设切换到围绕粉色做检测。

相关文件：

- `module/obstacle.py`
- `module/green_vehicle_detect.py`
- `module/obv.py`
- `docs/V16_CURRICULUM.md`

### 2.2 learned obstacle-context 路线

新增或持续维护的核心文件：

- `module/obstacle_context_learned.py`
- `scripts/export_obstacle_context_dataset.py`
- `scripts/train_obstacle_context_estimator.py`
- `scripts/train_obstacle_context_temporal.py`
- `scripts/eval_obstacle_context_learned.py`

功能包括：

- 从 sim 采样导出监督数据集。
- 支持单帧 estimator。
- 支持 temporal estimator（GRU）。
- 支持 `pink + temporal + focal` 训练配置。
- 支持 held-out sampled evaluation。

### 2.3 online 接线与 gating

新增/修改：

- `module/obstacle_context.py`
- `module/multi_scene_env.py`
- `scripts/eval_obstacle_context_compare.py`
- `scripts/sweep_obstacle_context_online_gating.py`

主要能力：

- 在环境里接入 `learned_v1` obstacle context source。
- 支持 online hysteresis / gating：
  - `present_threshold`
  - `present_off_threshold`
  - `activation_consecutive`
  - `deactivation_consecutive`
- 支持 runtime vs learned 的在线 smoke A/B。

### 2.4 数据混合与诊断

新增：

- `scripts/mix_obstacle_context_datasets.py`
- `scripts/run_train_obstacle_context_temporal_diag.sh`

作用：

- 合并主训练集与补充 hard-negative 数据。
- 给训练脚本增加最小诊断能力，避免训练 silent fail 时无法定位。

## 3. 纯 CV 路线的结论

### 3.1 做过的方向

尝试过的纯 `CV` 方向包括：

- 基于颜色的 vehicle detector。
- `veh_prob` 上游过滤。
- `edge / motion / road / objectness` 等规则式补救。
- `visible_present` 标签重定义。
- 大量错帧 dump 和人工检查。

### 3.2 结论

纯 `CV` 路线已经验证得比较充分，结论如下：

1. `WS` 的近车目标在某些条件下还能被规则法救回来。
2. `GT` 的远小车在规则法下越来越接近上限。
3. 继续堆规则的边际收益已经明显下降。
4. 纯 `CV` 适合作为：
   - baseline
   - `veh_prob` 辅助先验
   - debug 工具
5. 纯 `CV` 不适合作为后续主线解决方案。

## 4. sim 采样数据链

### 4.1 数据来源

当前 learned estimator 的训练数据全部来自 sim 采样，不是人工标注图片。

输入：

- `6ch image`
- `7d self/base state`

标签：

- `target_present`
- `target_longitudinal`
- `target_lateral`
- `target_dist`

标签来自 sim/runtime 相对几何真值。

### 4.2 可见性标签

没有直接把“场景里有障碍”当作正样本，而是额外用了 `visible_present` 逻辑：

- 只把当前视角下按几何条件应该可见的障碍作为检测正样本。

这一步是必要的。否则大量“场景存在但图像不可见”的帧会污染训练和评测。

## 5. learned temporal 路线进展

### 5.1 单帧到时序

先做过单帧 estimator，随后切到 temporal estimator。

结论：

- 单帧版只能证明链路能通。
- 真正可用必须上时序。
- 当前超车问题更像“持续若干秒的交互过程”，不适合只做单帧判断。

### 5.2 网络结构调整

已做的结构调整：

- 不再过早把图像全局池化到 `1x1`。
- 改为先保留 `4x4` 空间特征，再做融合。
- temporal 主干使用 GRU。

结论：

- 结构调整本身不是当前第一瓶颈。
- 时序是必要条件。
- 但单靠改 backbone 不能解决 online 可用性问题。

## 6. 关键训练与评测结果

### 6.1 粉色对齐后的 temporal + focal 路线

核心模型目录：

- `models/v16_pid_overtake_course_20260420/obstacle_context_seq_interaction_gt_8ep_u8_pink_temporal_focal_8ep_monitor`

对应 held-out sampled eval：

- `models/v16_pid_overtake_course_20260420/obstacle_context_seq_interaction_gt_8ep_u8_pink_temporal_focal_8ep_monitor_eval_val.json`

关键结果：

- 默认阈值 `0.5`
  - `precision = 0.1116`
  - `recall = 1.0`
  - `f1 = 0.2008`
- 交互区：
  - `interaction_default f1 = 1.0`
- 可见窗口：
  - `visible_run_detected_rate = 1.0`
  - `visible_run_stable_rate = 1.0`
  - `first_detect_delay_mean_steps = 0`
- 超车前窗口：
  - `frame_trigger_rate = 1.0`
  - `stable_trigger_rate ≈ 0.988`

结论：

- 模型已经学到“交互区有东西”的信号。
- 当前主要问题不是看不到，而是全局误报偏高。

### 6.2 online smoke：ungated learned

文件：

- `models/v16_pid_overtake_course_20260420/online_smoke_runtime_vs_learned_v1_intro_1ep.json`

结论：

- 技术上接线成功。
- 但 online 上会长期误报 `present=1`。
- 策略显著变坏，不可用。

### 6.3 online smoke：gated learned

文件：

- `models/v16_pid_overtake_course_20260420/online_smoke_runtime_vs_learned_v1_intro_1ep_gated065.json`

gating 配置：

- `present_threshold = 0.65`
- `present_off_threshold = 0.50`
- `activation_consecutive = 3`
- `deactivation_consecutive = 2`

结论：

- 在线误报问题被明显压下去。
- 但策略表现仍然远差于 runtime truth。
- 说明 gating 只能压 hallucination，不能解决 estimator 本体精度不足的问题。

### 6.4 online gating sweep

文件：

- `models/v16_pid_overtake_course_20260420/online_gating_sweep_small.json`

结论：

- 找到了相对更稳的 operating point。
- 但本质问题没有变：只靠 gating 无法把 learned_v1 直接变成控制可用输入。

## 7. hard-negative 方向测试结果

### 7.1 负样本补采

补采集：

- `models/v16_pid_overtake_course_20260420/obstacle_context_seq_negmix_wsgt_4ep_u8_pink.json`

混合集：

- `models/v16_pid_overtake_course_20260420/obstacle_context_seq_interaction_gt_8ep_plus_negmix_8ep.json`

### 7.2 bounded hard-negative probe

训练目录：

- `models/v16_pid_overtake_course_20260420/obstacle_context_seq_interaction_gt_8ep_plus_negmix_bounded64`

对应 held-out eval：

- `models/v16_pid_overtake_course_20260420/obstacle_context_seq_interaction_gt_8ep_plus_negmix_bounded64_eval_val.json`

### 7.3 与基线对比

和 `8ep_monitor` 基线相比：

- 默认阈值 `precision`: `0.1116 -> 0.1027`
- 默认阈值 `recall`: `1.0 -> 0.9`
- 默认阈值 `f1`: `0.2008 -> 0.1844`
- 交互区 `f1`: `1.0 -> 1.0`
- 可见窗口首次检测延迟：`0.0 -> 1.3`
- 超车前稳定触发率：基本不变

结论：

- 这版 hard-negative 补采 + 重训没有带来收益。
- 当前问题不主要是“负样本不够”。
- 继续原样堆 hard-negative 不值得。

## 8. 当前总体判断

### 8.1 已经确认成立的点

1. sim 采样 + runtime 真值监督这条数据链是通的。
2. `pink + temporal + focal` 比早期版本明显更合理。
3. 模型内部已经学到交互区信号。
4. online gating 是有效的，但只能解决一部分 hallucination。

### 8.2 当前还没解决的问题

1. 默认阈值下的全局分数分离仍然不够理想。
2. online 上即使压掉误报，策略仍然不能直接吃 learned context。
3. hard-negative 方向没有带来预期收益。

### 8.3 当前不建议做的事

- 不建议直接把当前 learned_v1 接回 PPO 正式训练。
- 不建议继续原样追加 hard-negative 数据。
- 不建议再投入大量时间深调纯 `CV` 规则。

## 9. 建议的下一步

当前最合理的下一步是：

1. 在当前 temporal backbone 上加 short-horizon future head：
   - `future_longitudinal h5/h10/h20`
   - `future_lateral h5/h10/h20`
   - `future_dist h5/h10/h20`

2. 用同一套 sampled eval 再看：
   - 默认阈值指标
   - 可见窗口首次检测延迟
   - 超车前窗口稳定触发率
   - 交互区 precision/recall

原因：

- 当前模型已经能在交互区感知到“有东西”。
- 下一个缺的更像是“这是不是即将影响超车的那个前车”，而不是再做全局当前帧二分类。
- 这个问题更接近 short-horizon interaction prediction，而不是继续堆负样本。

## 10. 当前结论

当前 obstacle-context 主线结论如下：

- 纯 `CV`：保留为 baseline 和先验，不再作为主线。
- learned temporal：是正确方向。
- `pink + temporal + focal`：已经证明有效。
- online gating：有效，但不够。
- hard-negative：本轮验证无增益。
- 下一个合理方向：`future head / short-horizon prediction`。
