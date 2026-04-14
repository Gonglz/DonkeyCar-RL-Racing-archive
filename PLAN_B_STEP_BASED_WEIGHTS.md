# 方案 B: 基于累积步数的动态权重补偿

## 📋 实现完成清单

### ✅ 已应用的改动

#### 1. 环境状态追踪 (`MultiSceneEnvV16.__init__`)
```python
# 新增追踪累积步数
self._total_steps_per_scene = [0] * len(env_ids)
self.use_step_based_weights = True
```

#### 2. 步数累积 (在 episode 结束时)
```python
if "l" in ep:
    ep_len = float(ep["l"])
    self._scene_recent_lengths[self.active_scene_idx].append(ep_len)
    self._total_steps_per_scene[self.active_scene_idx] += ep_len  # ← 新增
```

#### 3. 权重补偿方法
新增方法 `_apply_step_based_weight_compensation()`：
- 计算累积步数比例
- 应用平方根衰减：`compensation = sqrt(max_steps / scene_steps)`
- 权重范围限制在 `[0.8, 1.2]` 以避免过度调整
- 最终归一化

#### 4. 集成到权重更新流程
- 即使禁用了基于**成功率**的动态权重，仍然应用基于**步数**的补偿
- 每 20 个 episode 评估一次

---

## 🎯 工作原理

### 场景：WS vs GT

```
初始状态（假设训练 100 个 episode）:
  WS: 小碰撞，平均 20 步/episode  → 总计 2000 步
  GT: 长圈数，平均 600 步/episode → 总计 3000 步

传统动态权重问题:
  ❌ 计算的是 episode 数量: WS [100 eps], GT [100 eps]
  ❌ 但 WS 贡献的步数少 (2000 < 3000)
  ❌ 结果：权重平分，但采样"质量"不均（GT占用60%的计算预算）

方案B补偿:
  ✅ 检测到 WS 总步数 (2000) < GT 总步数 (3000)
  ✅ 计算补偿:
       WS: sqrt(3000/2000) = 1.22 → 权重 × 1.22
       GT: sqrt(3000/3000) = 1.00 → 权重 × 1.00
  ✅ 归一化后: WS [0.55], GT [0.45]
  ✅ WS 获得更多采样机会！
```

### 与 Step 预算补偿的关系

两个机制**正交**，可同时使用：

```
┌─ Level 1: 场景采样权重 ─────────────────────┐
│  WS [0.55] ← Step-based补偿  GT [0.45]     │
│                                             │
│  ┌─ Level 2: 障碍/无障碍比例补偿 ─────┐    │
│  │ WS的55%采样:                       │    │
│  │  ├─ 50% 有障碍 (70-90% progress) │    │
│  │  └─ 50% 无障碍                    │    │
│  │ GT的45%采样:                       │    │
│  │  ├─ 60% 有障碍 (随机+边缘)       │    │
│  │  └─ 40% 无障碍                    │    │
│  └────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

---

## 📊 效果预期

### Before (禁用权重 + 固定 50-50)
```
采样分配: WS [0.50], GT [0.50] (固定)
问题: 如果WS努力学习但仍短episode，GT仍占50%采样
```

### After (禁用权重 + Step-based补偿)
```
采样分配: WS [0.55-0.65], GT [0.35-0.45] (自适应)
优势: WS短episode时自动获得补偿，鼓励继续尝试
```

---

## 🔧 参数调整

### 补偿强度

在 `_apply_step_based_weight_compensation()` 中：

```python
# 当前设置（保守）:
compensation = np.clip(compensation, 0.8, 1.2)
# 范围 [0.8, 1.2] 意味着最多调整 ±20%

# 激进方案:
compensation = np.clip(compensation, 0.6, 1.4)
# 范围 [0.6, 1.4] 最多调整 ±40%
```

### 更新频率

```python
# 当前: 每 20 个 episode 更新
max(20, self.dynamic_weight_update_episodes)

# 改为 10 个 episode 更新（反应更快）:
max(10, self.dynamic_weight_update_episodes)
```

---

## ✅ 验证清单

跑训练时观察：

- [ ] 日志是否显示权重在调整？
- [ ] WS权重是否 > GT权重？
- [ ] WS是否逐步提升性能（reward/laps）？
- [ ] GT性能是否维持稳定（不下降）？
- [ ] 权重调整是否平滑（不剧烈波动）？

---

## 🚀 启动训练

现在可以启动测试：

```bash
cd /home/longzhao/mysim_public

python src/ppo_multitrack_v16.py \
  --resume-path models/v16_single_test/v16_2000000_steps.zip \
  --curriculum avoid_static \
  --steps 100000 \
  --exp-tag "plan_b_step_based_weights"
```

预期：
- ✅ avoid_static 禁用成功率权重，启用 step-based 补偿
- ✅ WS 采样权重自动提升
- ✅ WS 性能逐步改善

---

## 📝 后续优化

如果方案B仍不够：

1. **增加 WS 基础奖励**
   - 提高 `w_center`, `survival_reward_scale`
   - 给少量无障碍回合的基础驾驶奖励

2. **更激进的补偿**
   - 调整 `compensation = np.clip(..., 0.6, 1.4)`
   - 让 WS 权重能达到 70%

3. **减少 GT 采样**
   - 改动 base_scene_weights: `[0.6, 0.4]` 而不是 `[0.5, 0.5]`
   - 然后让 step-based 补偿在这个基础上调整
