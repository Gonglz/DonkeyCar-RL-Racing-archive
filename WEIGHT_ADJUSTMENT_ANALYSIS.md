# 动态权重调整机制分析 - Episode vs Step

## 📊 当前机制

### 1. 更新频率（按EPISODE）
```
V16: 每 36 个 episode 更新一次      (multi_scene_env.py 行 1745)
V13: 每 20 个 episode 更新一次      (multi_scene_env.py 行 143)
```

### 2. 统计基础（按EPISODE）
```python
# 行 258-259: 最近 N 个 episodes 的统计
_scene_recent_rewards = [deque(maxlen=50) for _ in env_ids]  # 最近50个episodes的奖励
_scene_recent_lengths = [deque(maxlen=50) for _ in env_ids]  # 最近50个episodes的步数

# 行 942-944: 每个episode结束时追加数据
_scene_recent_rewards[active_scene_idx].append(float(ep["r"]))
_scene_recent_lengths[active_scene_idx].append(float(ep["l"]))
```

### 3. 权重计算（混合EPISODE统计+STEP因子）
```python
# 行 515-518: 计算平均值（按episode）
reward_means[i] = mean(_scene_recent_rewards[i])  # 平均奖励/episode
len_means[i] = mean(_scene_recent_lengths[i])    # 平均长度/episode

# 行 567: 长度因子（考虑step）
len_factor = (ref_len / mean_len_ep) ^ beta
# 例如:
#   GT: 平均600步/episode → factor=1.0
#   WS: 平均20步/episode  → factor=(600/20)^1.0=30
```

---

## ❌ 问题分析

### 问题1: Episode计数偏差
```
权重更新周期 = 36 episodes

当WS在720步时完成36个episode：
  36 episodes × 20步/episode = 720步

同期GT只完成：
  720步 ÷ 500步/episode ≈ 1.4 episodes

结果: WS需要36个episode才能更新一次权重
     GT根本还没累积到下一次权重更新的threshold

→ GT持续获得采样优势！
```

### 问题2: 样本积累太慢
```
权重更新需要:
  dynamic_min_samples_per_scene >= 6 (默认值)

WS如果成功率为10%，每10个episode才有1个成功，则:
  需要 6 个成功 episode × 10 = 60 个 episodes
  = 60 × 20步 = 1200步数据才能积累好样本

这期间GT已经完成 1200/500 = 2.4个完整训练周期
```

### 问题3: Episode级统计不公平
```
假设权重更新时的统计:

WS 最近50个episode:
  - 40个碰撞episode (10步, 奖励1)
  - 10个正常episode (50步, 奖励8)
  → 平均: (40×1 + 10×8) / 50 = 2.4 奖励/episode

GT 最近50个episode:
  - 所有都是完整episode (600步, 奖励500+)
  → 平均: 500+ 奖励/episode

权重调整:
  WS deficit ∝ (max_reward - 2.4) / (max - min)  → 很大
  GT deficit ∝ (max_reward - 500) / (max - min)  → 很小

→ WS理论上会被提升，但新权重下一次更新要等36个episode后
```

---

## ✅ 已应用的修复

### 方案: 禁用动态权重
```python
"avoid_static": {
    "enable_dynamic_scene_weights": False,  # ✅ 固定权重
    "scene_weights": [0.5, 0.5],  # 平分
}
```

**优点:**
- ✅ 完全消除episode计数差异影响
- ✅ WS和GT获得公平的采样机会 (1:1)
- ✅ 简单明确，易于调试

**缺点:**
- ❌ GT如果真的学得很好，不能自动让步给WS
- ❌ 无法自适应

---

## 💡 其他可能的方案

### 方案A: 启用Step级补偿（更激进）
理想情况：权重更新应该按**步数**而不是**episode数**来触发

```python
# 修改权重更新触发条件（伪代码）
if total_steps_since_last_update >= some_step_threshold:  # 而不是按episode
    _update_weights()
```

**优点:** 完全公平的step分配
**缺点:** 需要重写MultiSceneEnv的权重逻辑

### 方案B: 提高采样整体频率
```python
"dynamic_weight_update_episodes": 10,  # 从36改为10
"dynamic_min_samples_per_scene": 3,    # 从6改为3
```

**优点:** WS获得更频繁的权重调整机会
**缺点:** 权重波动可能更剧烈

### 方案C: 改进长度补偿因子
```python
"dynamic_length_beta": 2.0,  # 从1.0改为2.0
```
加强长度补偿 → WS相对权重 = (600/20)^2 = 900（而不是30）

**优点:** 短episode的场景获得指数级别的权重提升
**缺点:** 可能过度补偿

---

## 📋 当前配置总结

| 参数 | V16值 | 说明 |
|------|-------|------|
| 更新频率 | 36 episodes | 太稀疏，导致WS权重更新不及时 |
| 统计窗口 | 50 episodes | 合理 |
| 最小样本数 | 6 episodes | 对短episode场景偏严格 |
| 成功权重mix | 0.85 | 85%看成功率，15%看奖励 |
| 长度因子beta | 1.0 | 线性补偿（可加强为2.0） |
| 动态启用 | avoid_static关闭 | ✅ 已修复 |

---

## 🎯 建议

**当前状态（已修复）:**
- ✅ avoid_static 已禁用动态权重 → WS能获得公平采样
- ✅ 其他阶段保持动态权重（但需监控）

**如果未来WS学到一定程度，想开启动态权重:**
1. 降低更新频率: `dynamic_weight_update_episodes: 36 → 15`
2. 降低样本要求: `dynamic_min_samples_per_scene: 6 → 3`
3. 加强长度补偿: `dynamic_length_beta: 1.0 → 2.0`

这样的组合能更快地让WS获得权重提升，同时保持稳定性。
