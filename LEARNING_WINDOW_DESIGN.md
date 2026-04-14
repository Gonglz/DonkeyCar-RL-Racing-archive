# 学习窗口补偿机制设计

## 问题描述

当前训练中，某些 scene（如 GT）可能出现：
- **大量步数在无障碍环境中消耗** → 学不到避障
- **有障碍步数不足** → 避障技能不足

例如：
```
GT avoid_static 阶段目标: 50% 有障碍
实际结果:
  - 第1批100个episode: 400步无障碍, 100步有障碍 (80% 无障碍，不足目标)
  - 第2批100个episode: 300步无障碍, 200步有障碍 (60% 无障碍，仍不足)
  - 累积deficit: (200 - 100) + (200 - 200) = 100步缺失
  
解决: 后续自动调整 obstacle_free_prob，强制补充缺失的100步有障碍
```

---

## 设计方案：步数预算跟踪与动态补偿

### 核心思路

1. **定义目标**: 每个阶段的每个 scene 有"目标障碍步数比例"
2. **跟踪实际**: 每个评估窗口（如100个episode）内的实际步数分布
3. **计算缺陷**: 与目标对比，得出"缺失步数"
4. **动态补偿**: 在后续窗口中调整 `obstacle_free_prob` 来补充缺失步数

### 具体实现

#### 1. 配置层（CURRICULUM_PHASES 中添加）

```python
CURRICULUM_PHASES = {
    "avoid_static": {
        # ... 现有配置 ...
        
        # 新增：目标步数分配
        "obstacle_target_ratios": {
            "ws": 0.50,  # WS 目标 50% 步数有障碍
            "gt": 0.60,  # GT 目标 60% 步数有障碍
        },
        
        # 新增：评估窗口大小（每N个episode评估一次）
        "window_episode_count": 50,  # 每50个episode评估一次
        
        # 新增：最大补偿力度
        "max_compensation_ratio": 0.3,  # 单次补偿最多调整 30% 的 obstacle_free_prob
    },
    
    "avoid_mixed": {
        # ...
        "obstacle_target_ratios": {
            "ws": 0.65,  # WS 目标 65% 有障碍
            "gt": 0.75,  # GT 目标 75% 有障碍
        },
        "window_episode_count": 50,
        "max_compensation_ratio": 0.3,
    },
}
```

#### 2. Callback 实现

新增 `StepBudgetCompensationCallback` 类：

```python
class StepBudgetCompensationCallback(BaseCallback):
    """
    跟踪每个scene的步数消耗，动态调整obstacle_free_prob以补偿不足
    """
    
    def __init__(self, window_episode_count=50):
        super().__init__()
        self.window_episode_count = window_episode_count
        
        # 跟踪每个scene的步数消耗
        self.scene_step_counts = {}  # {scene_key: {"with_obs": X, "without_obs": Y}}
        self.episode_counts = {}     # {scene_key: N}
        self.last_eval_step = {}     # {scene_key: last_step_evaluated}
        
        # 记录调整历史
        self.compensation_history = {}  # {scene_key: [(step, deficit, new_free_prob), ...]}
    
    def _on_step(self) -> bool:
        """每一步都检查是否需要评估和补偿"""
        
        # 从 env 中提取 scene 信息和步数统计
        info = self.model.env.get_attr("_scene_step_stats")  # 需要环境提供
        
        if not info:
            return True
        
        for scene_key, stats in info.items():
            episode_count = stats.get("episode_count", 0)
            
            # 检查是否达到评估窗口
            last_eval = self.last_eval_step.get(scene_key, 0)
            if episode_count - last_eval >= self.window_episode_count:
                self._evaluate_and_compensate(scene_key, stats)
                self.last_eval_step[scene_key] = episode_count
        
        return True
    
    def _evaluate_and_compensate(self, scene_key: str, stats: dict):
        """评估窗口内的步数分布，计算缺陷，调整障碍概率"""
        
        # 从配置获取目标比例
        target_ratio = self.curriculum_config.get("obstacle_target_ratios", {}).get(scene_key, 0.5)
        window_episodes = self.window_episode_count
        
        # 提取窗口内的步数统计
        with_obs_steps = stats.get("window_with_obs_steps", 0)
        without_obs_steps = stats.get("window_without_obs_steps", 0)
        total_steps = with_obs_steps + without_obs_steps
        
        if total_steps == 0:
            return
        
        # 计算实际比例和缺陷
        actual_ratio = with_obs_steps / total_steps
        target_steps = int(total_steps * target_ratio)
        deficit = target_steps - with_obs_steps  # 缺失的步数
        
        # 只在缺陷显著时调整
        if abs(deficit) < total_steps * 0.05:  # 缺陷 < 5% 不调整
            return
        
        # 计算补偿调整
        current_free_prob = self.current_free_prob[scene_key]
        
        if deficit > 0:
            # 缺少有障碍步数 → 降低 obstacle_free_prob
            # 需要增加的有障碍步数 = deficit
            # deficit = total_steps * (free_prob_old - free_prob_new)
            # free_prob_new = free_prob_old - deficit / total_steps
            
            reduction = min(deficit / total_steps, self.max_compensation_ratio)
            new_free_prob = max(0.0, current_free_prob - reduction)
            
            log_msg = f"[{scene_key}] 缺少 {deficit} 步有障碍 (目标{target_ratio:.0%}, 实际{actual_ratio:.0%}) → " \
                      f"obstacle_free_prob: {current_free_prob:.2f} → {new_free_prob:.2f}"
        
        else:
            # 过度有障碍 → 提高 obstacle_free_prob
            addition = min(-deficit / total_steps, self.max_compensation_ratio)
            new_free_prob = min(1.0, current_free_prob + addition)
            
            log_msg = f"[{scene_key}] 过度有障碍 {-deficit} 步 (目标{target_ratio:.0%}, 实际{actual_ratio:.0%}) → " \
                      f"obstacle_free_prob: {current_free_prob:.2f} → {new_free_prob:.2f}"
        
        # 应用调整
        self.current_free_prob[scene_key] = new_free_prob
        self.compensation_history.setdefault(scene_key, []).append({
            "step": self.num_timesteps,
            "window_episodes": window_episodes,
            "with_obs_steps": with_obs_steps,
            "without_obs_steps": without_obs_steps,
            "target_ratio": target_ratio,
            "actual_ratio": actual_ratio,
            "deficit": deficit,
            "old_free_prob": current_free_prob,
            "new_free_prob": new_free_prob,
        })
        
        logger.info(log_msg)
```

#### 3. 环境端改动

在 `MultiSceneEnvV16` 中添加窗口统计：

```python
class MultiSceneEnvV16(gym.Env):
    def __init__(self, ...):
        # ...
        
        # 新增：步数统计
        self._scene_step_stats = {
            "ws": {
                "episode_count": 0,
                "window_with_obs_steps": 0,  # 窗口内有障碍的步数
                "window_without_obs_steps": 0,  # 窗口内无障碍的步数
                "total_steps": 0,
            },
            "gt": {
                "episode_count": 0,
                "window_with_obs_steps": 0,
                "window_without_obs_steps": 0,
                "total_steps": 0,
            }
        }
        self._window_episode_threshold = 50  # 每50个episode重置窗口
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        
        # 统计步数
        scene_key = info.get("scene")  # ws 或 gt
        has_obstacle = info.get("episode_obstacle_active", False)
        
        if scene_key in self._scene_step_stats:
            if has_obstacle:
                self._scene_step_stats[scene_key]["window_with_obs_steps"] += 1
            else:
                self._scene_step_stats[scene_key]["window_without_obs_steps"] += 1
        
        return obs, reward, done, info
    
    def reset(self):
        # 每次 reset，检查是否达到窗口阈值
        obs, info = super().reset()
        
        scene_key = info.get("scene")
        if scene_key:
            self._scene_step_stats[scene_key]["episode_count"] += 1
            
            # 达到窗口大小时重置
            if self._scene_step_stats[scene_key]["episode_count"] % self._window_episode_threshold == 0:
                self._scene_step_stats[scene_key]["window_with_obs_steps"] = 0
                self._scene_step_stats[scene_key]["window_without_obs_steps"] = 0
        
        return obs, info
    
    def get_attr(self, name):
        """供callback调用获取统计信息"""
        if name == "_scene_step_stats":
            return self._scene_step_stats
        return super().get_attr(name)
```

---

## 使用方式

### 1. 注册 Callback

```python
# 在 ppo_multitrack_v16.py 的 train_v16() 中
callbacks = [
    # ... 其他 callbacks ...
    StepBudgetCompensationCallback(
        window_episode_count=50,
        curriculum_phases=CURRICULUM_PHASES,
    ),
]
```

### 2. 配置示例

```python
CURRICULUM_PHASES = {
    "avoid_static": {
        "scene_weights": [0.5, 0.5],
        # ... 现有配置 ...
        
        # 新增补偿配置
        "obstacle_target_ratios": {
            "ws": 0.50,  # WS 目标50%步数有障碍
            "gt": 0.60,  # GT 目标60%步数有障碍
        },
        "window_episode_count": 50,      # 每50个episode评估
        "max_compensation_ratio": 0.25,  # 单次最多调整25%
    },
}
```

### 3. 日志输出示例

```
[gt] 缺少 234 步有障碍 (目标60%, 实际42%) → obstacle_free_prob: 0.40 → 0.28
[ws] 过度有障碍 89 步 (目标50%, 实际58%) → obstacle_free_prob: 0.50 → 0.58
```

---

## 工作流程示意

```
Timeline:
--------

Episode 1-50:
├─ GT: 300步无障碍, 150步有障碍 (实际33%, 目标60%)
│  Deficit: 150步 - (150 × 1.0) = 0 ... 等等
│  计算: target_steps = 450 × 60% = 270
│       actual = 150
│       deficit = 120 步缺失
├─ WS: 250步无障碍, 250步有障碍 (实际50%, 目标50%)
│  Deficit: 0 (正好)

评估触发 → 调整:
├─ GT: obstacle_free_prob 0.40 → 0.33 (降低自由度，强制更多障碍)
├─ WS: obstacle_free_prob 0.50 → 0.50 (保持)

Episode 51-100:
├─ GT: 250步无障碍, 350步有障碍 (实际58%, 目标60%)
│  Deficit: -20步 (略微过度，可接受)
├─ WS: 240步无障碍, 260步有障碍 (实际52%, 目标50%)
│  Deficit: +10步 (轻微过度)

评估触发 → 调整:
├─ GT: obstacle_free_prob 0.33 → 0.35 (略提高)
├─ WS: obstacle_free_prob 0.50 → 0.52 (略提高)

继续循环...
```

---

## 优势

✅ **自动均衡**: 无需手动调整，系统自动补偿  
✅ **公平竞争**: GT 和 WS 都获得平衡的学习机会  
✅ **可调节**: 通过 `max_compensation_ratio` 控制激进程度  
✅ **可追溯**: 记录所有调整历史，便于调试  
✅ **稳定**: 缓慢渐进的调整，避免剧烈波动  

---

## 可选增强

### 1. 场景级目标
```python
# 按阶段和场景微调目标
"obstacle_target_ratios": {
    "ws": 0.50,
    "gt": 0.60,
    "gt_secondary": 0.40,  # 某个特殊场景
}
```

### 2. 动态调整灵敏度
```python
# 根据实际缺陷大小调整敏感度
if abs(deficit_ratio) > 0.20:
    compensation = aggressive  # 缺陷>20%, 大幅调整
elif abs(deficit_ratio) > 0.10:
    compensation = moderate    # 缺陷>10%, 适度调整
else:
    compensation = conservative # 缺陷<10%, 保守
```

### 3. 滑动窗口
```python
# 改用滑动窗口而非固定窗口
# 更平滑的补偿效果
```

---

## 实现路线

| 阶段 | 任务 | 难度 |
|------|------|------|
| 1 | 在 MultiSceneEnvV16 中添加步数统计 | ⭐ |
| 2 | 实现 StepBudgetCompensationCallback | ⭐⭐ |
| 3 | 在 CURRICULUM_PHASES 中添加配置字段 | ⭐ |
| 4 | 测试和调优参数 | ⭐⭐⭐ |
| 5 | 可视化补偿过程 (可选) | ⭐⭐ |

