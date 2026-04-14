# Step 预算补偿机制 - 实现完成

## ✅ 实现清单

### 1. 环境端修改 (`module/multi_scene_env.py`)

**添加到 `MultiSceneEnvV16.__init__`：**
```python
# 步数预算统计（用于学习窗口补偿）
self._step_budget_stats = {
    "ws": {"episode_count": 0, "window_with_obs_steps": 0, "window_without_obs_steps": 0},
    "gt": {"episode_count": 0, "window_with_obs_steps": 0, "window_without_obs_steps": 0}
}
self._window_episode_threshold = 50
self._curriculum_phase = "warmup"  # 当前课程阶段
```

**新增 `reset()` override：**
- 跟踪每个scene的episode计数
- 每50个episode重置窗口统计

**新增 `step()` override：**
- 检测本步是否有活跃障碍（通过 `obstacle_dist > 0` 或 `near_collision > 0`）
- 统计"有障碍"和"无障碍"的步数

---

### 2. Callback 实现 (`module/callbacks.py`)

**新增 `StepBudgetCompensationCallback` 类：**

工作流程：
```
每一步检查
  ↓
检查是否达到评估窗口 (N=50个episode)
  ↓
计算实际 vs 目标的障碍步数比例
  ↓
如果缺陷 > 5%：
  ├─ 缺少有障碍 → 降低 obstacle_free_prob
  └─ 过度有障碍 → 提高 obstacle_free_prob
  ↓
应用调整，记录历史
```

核心算法：
```python
deficit = target_steps - actual_steps

if deficit > 0:
    # 缺少有障碍步数
    reduction = min(deficit / total_steps, max_compensation_ratio)
    new_free_prob = current_free_prob - reduction
else:
    # 过度有障碍
    addition = min(-deficit / total_steps, max_compensation_ratio)
    new_free_prob = current_free_prob + addition
```

---

### 3. 课程配置更新 (`src/ppo_multitrack_v16.py`)

**各阶段添加的配置：**

```python
CURRICULUM_PHASES = {
    "avoid_static": {
        # ... 原有配置 ...
        "obstacle_target_ratios": {"ws": 0.50, "gt": 0.60},
        "window_episode_count": 50,
        "max_compensation_ratio": 0.25,
    },
    "avoid_mixed": {
        "obstacle_target_ratios": {"ws": 0.65, "gt": 0.75},
        "window_episode_count": 50,
        "max_compensation_ratio": 0.25,
    },
    "lane_pid_intro": {
        "obstacle_target_ratios": {"ws": 0.70, "gt": 0.80},
        "window_episode_count": 50,
        "max_compensation_ratio": 0.25,
    },
    "lane_pid_full": {
        "obstacle_target_ratios": {"ws": 0.75, "gt": 0.85},
        "window_episode_count": 50,
        "max_compensation_ratio": 0.25,
    },
}
```

**目标设置说明：**

| 阶段 | WS目标 | GT目标 | 说明 |
|------|--------|--------|------|
| avoid_static | 50% | 60% | 开始接触障碍，WS专注避障 |
| avoid_mixed | 65% | 75% | 提升难度，GT接触多种障碍 |
| lane_pid_intro | 70% | 80% | 自动车难度，高频障碍 |
| lane_pid_full | 75% | 85% | 最终难度，几乎全是挑战 |

---

### 4. 训练脚本集成 (`src/ppo_multitrack_v16.py`)

**Import：**
```python
from module.callbacks import (
    ...
    StepBudgetCompensationCallback,
    ...
)
```

**环境初始化后设置 curriculum phase：**
```python
if curriculum_phase is not None:
    for envs_list in env.envs:
        if hasattr(envs_list, '_curriculum_phase'):
            envs_list._curriculum_phase = curriculum_phase
```

**Callback 注册：**
```python
callbacks.append(
    StepBudgetCompensationCallback(
        curriculum_phases=CURRICULUM_PHASES,
        window_episode_count=50,
        verbose=1,
    )
)
```

---

## 🔍 工作原理详解

### 场景1：WS 缺少有障碍步数

```
目标: 50% 有障碍 (WS at avoid_static)

窗口统计 (50 episodes):
  有障碍步数: 200
  无障碍步数: 300
  总步数: 500
  
分析:
  目标步数 = 500 × 50% = 250
  实际步数 = 200
  缺陷 = 250 - 200 = +50 步
  缺陷比例 = 50 / 500 = 10% > 5% → 触发调整
  
调整:
  reduction = min(50 / 500, 0.25) = 0.10
  new_free_prob = 0.50 - 0.10 = 0.40
  
效果:
  下一窗口期望 = 500 × 0.40 = 200 无障碍步
                500 × 0.60 = 300 有障碍步 ✓ (达到目标!)
```

### 场景2：GT 过度有障碍

```
目标: 60% 有障碍 (GT at avoid_static)

窗口统计:
  有障碍步数: 380
  无障碍步数: 120
  总步数: 500
  
分析:
  目标步数 = 500 × 60% = 300
  实际步数 = 380
  缺陷 = 300 - 380 = -80 步 (过度)
  缺陷比例 = 80 / 500 = 16% > 5% → 触发调整
  
调整:
  addition = min(80 / 500, 0.25) = 0.16
  new_free_prob = 0.40 + 0.16 = 0.56
  
效果:
  下一窗口期望 = 500 × 0.56 = 280 无障碍步
                500 × 0.44 = 220 有障碍步 (略低，但接近60%)
```

---

## 📊 日志输出示例

```
📊 [Step预算补偿] WS @ 步 150000
   目标障碍率: 50% | 实际: 40% | 缺陷: +50步
   obstacle_free_prob: 0.500 ↓ 0.400 (±0.100)

📊 [Step预算补偿] GT @ 步 150500
   目标障碍率: 60% | 实际: 76% | 缺陷: -80步
   obstacle_free_prob: 0.400 ↑ 0.560 (±0.160)
```

---

## 🎯 预期效果

### Before (无补偿)
```
GT 在 avoid_static:
  - 前50个episode: 80% 无障碍 (太宽松)
  - 中期: 逐渐调整到 ~50% 有障碍
  - 后期: 才接近目标 60%
  
问题: 学习不均衡，前期浪费了学习机会
```

### After (有补偿)
```
GT 在 avoid_static:
  - 第1窗口: 自动从 obstacle_free_prob 0.4 → 0.28
  - 第2窗口: 微调到 0.35 (达到 ~60%)
  - 之后: 持续维持 ±5% 的偏差
  
优势: 快速收敛到目标，避免浪费步数
```

---

## 🔧 调优参数

### 1. `obstacle_target_ratios`
```python
# 较激进: 更高的有障碍比例
"obstacle_target_ratios": {"ws": 0.70, "gt": 0.80}

# 较保守: 更多无障碍环节
"obstacle_target_ratios": {"ws": 0.30, "gt": 0.40}
```

### 2. `max_compensation_ratio`
```python
# 激进调整: 单次最多改变 ±30%
"max_compensation_ratio": 0.30  # 容易过度调整

# 保守调整: 单次最多改变 ±15%
"max_compensation_ratio": 0.15  # 收敛较慢但稳定
```

### 3. `window_episode_count`
```python
# 频繁评估: 每 30 个 episode
"window_episode_count": 30  # 快速响应，但可能波动

# 稀疏评估: 每 100 个 episode
"window_episode_count": 100  # 稳定，但响应慢
```

---

## 🚨 注意事项

### 1. 障碍检测依赖
目前通过 `obstacle_dist > 0` 或 `near_collision > 0` 检测。
如果这些字段不准确，统计会有误。

### 2. 多环境情况
DummyVecEnv 只有一个并行环境，所以统计简单。
SubprocVecEnv 的情况需要在各环境中分别统计。

### 3. 自动课程交互
StepBudgetCompensationCallback 和 ObstacleStepsBalancingCallback 并行运行。
如果两者目标冲突，可能会相互抵消。建议仅用其中一个。

---

## 🧪 测试清单

```
□ 启动训练: python src/ppo_multitrack_v16.py --curriculum avoid_static --steps 100000
□ 监控日志中的"Step预算补偿"输出
□ 验证 obstacle_free_prob 是否有在调整
□ 确认调整方向正确 (缺陷→降低free_prob, 过度→提高free_prob)
□ 运行 1M+ 步长期测试，验证收敛稳定性
□ 对比 WS/GT 的学习曲线是否更均衡
```

