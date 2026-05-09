# V17 训练问题与处理记录

记录日期: 2026-05-08
关联版本: `src/ppo_multitrack_v17.py`, `src/ppo_multitrack_v18.py`
当前主线: V17 双 sim, 端口 `9091/9093`

---

## 1. 结论摘要

本轮对话里最终选择继续使用 V17 双窗口训练，而不是 V18 四窗口并行。

V18 四 worker 方案已经验证能连接并运行，但实测吞吐更差:

| 方案 | 配置 | 实测速度 |
|---|---|---:|
| V17 | 2 个 DonkeySim, `9091/9093` | 约 `12.3-14.1 steps/s` |
| V18 | 4 个 DonkeySim, `9091/9093/9095/9097` | 约 `5.9 steps/s` |

训练主线已从 `warmup` 自动晋级到 `warmup_a`。截至最近一次检查:

- `warmup` 在 `total_steps=460032` 晋级, 成功窗口为 `success_counts={'ws': 3, 'gt': 2}`。
- 当前在 `warmup_a`, 修复障碍 park 后已从 `final_model.zip` 续跑, 门控恢复到约 `142k stage steps`。
- 最近 `ws` 窗口已经比较稳定, `gt` 仍需要更多成功 episode；暂不手动强跳 `avoid_static`。
- 建议继续让 auto curriculum 自己判断, 不手动强跳 `avoid_static`。

---

## 2. 课程设置问题

### 2.1 问题: 前置课程过多, 影响主线

原先课程里有 `ws_bootstrap` 和 dense/PID 相关阶段。当前目标是重新从基础 warmup 开始训练, 不让过多前置阶段或 dense 阶段干扰。

处理:

- Auto curriculum 起点改为 `warmup`。
- 去掉 `ws_bootstrap`。
- 去掉 `lane_pid_dense` / dense 阶段。
- PID 阶段保留为 `lane_pid_intro -> lane_pid_mid -> lane_pid_full`。

验证:

- `tests/test_v17_curriculum.py::test_auto_curriculum_starts_at_warmup_without_removed_phases`
- 当前阶段顺序:
  - `warmup`
  - `warmup_a`
  - `avoid_static`
  - `avoid_mixed`
  - `lane_pid_intro`
  - `lane_pid_mid`
  - `lane_pid_full`

### 2.2 问题: avoid / PID 阶段障碍覆盖率需要明确

本轮要求:

- `avoid_static` / `avoid_mixed`: free prob 设为 `0.10`, 即 90% episode 有障碍。
- PID 阶段: free prob 设为 `0.05`, 即 95% episode 有障碍。

当前配置:

| 阶段 | `obstacle_free_prob` | `ws_obstacle_free_prob` | 模式 |
|---|---:|---:|---|
| `warmup` | `0.75` | `1.00` | static |
| `warmup_a` | `0.50` | `0.50` | static |
| `avoid_static` | `0.10` | `0.10` | static |
| `avoid_mixed` | `0.10` | `0.10` | static/jitter/nudge |
| `lane_pid_intro` | `0.05` | `0.05` | lane_pid |
| `lane_pid_mid` | `0.05` | `0.05` | lane_pid |
| `lane_pid_full` | `0.05` | `0.05` | lane_pid |

补充: `avoid_static` / `avoid_mixed` 也有安全距离惩罚, 但主要是
`obstacle_clearance_penalty_scale`, 不是 PID/overtake 阶段更重的
`unsafe_close_penalty_scale`。V17 当前设定:

| 阶段 | 赛道 | `obstacle_clearance_penalty_scale` | 距离区间 |
|---|---|---:|---|
| `avoid_static` | WS | `0.45` | `0.30m-0.60m` |
| `avoid_static` | GT | `0.40` | `0.30m-0.60m` |
| `avoid_mixed` | WS | `0.55` | `0.30m-0.60m` |
| `avoid_mixed` | GT | `0.45` | `0.30m-0.60m` |

含义: 障碍距离低于 `0.60m` 开始出现 clearance risk, 接近 `0.30m`
时惩罚达到最大。avoid 阶段同时保留更高的碰撞/出界 terminal 惩罚,
用于先学习“不要贴障碍、不要撞”, 之后 PID/overtake 阶段再引入更强的
跟车过近和超车通过约束。

验证:

- `tests/test_v17_curriculum.py::test_v17_free_prob_schedule`

---

## 3. GT 障碍固定进度与 reset 几何不对齐

### 3.1 现象

在 warmup 早期, GT 障碍位置存在风险: 固定 progress 与车辆 reset 起点/track 几何不对齐时, 障碍可能不是出现在合理前方窗口, 导致模型还没学会基础循迹时就被不合理障碍分布干扰。

### 3.2 处理

对 `warmup` 和 `warmup_a`:

- 清空 `obstacle_fixed_progress_ratio`。
- 清空 `obstacle_fixed_progress_distribution`。
- 用相对 ego 的 ahead sampling:
  - `obstacle_spawn_ahead_min_m = 2.0`
  - `obstacle_spawn_ahead_max_m = 6.0`
- WS 也清空 `ws_obstacle_fixed_progress_ratio`。

这样 warmup 障碍从“固定赛道 progress”改为“基于当前 reset pose 的前方采样”, 避免 GT 固定 progress 与 reset pose 不对齐。

### 3.3 验证

测试覆盖:

- `tests/test_v17_curriculum.py::test_gt_warmup_obstacle_samples_in_front_of_reset_pose`

该测试用 GT reset 点构造 agent pose, 调用 runtime 采样障碍, 验证:

- obstacle 在车前方: `relative.longitudinal > 0.5`
- 距离合理: `relative.planar_distance < 6.5`

---

## 4. V18 四 sim 加速实验

### 4.1 目标

尝试 4 个 DonkeySim 窗口, 2 个场景各 2 个 replica:

- `ws@9091`
- `gt@9093`
- `ws@9095`
- `gt@9097`

V18 文件:

- `src/ppo_multitrack_v18.py`
- `tests/test_v18_multisim.py`

### 4.2 实施

V18 使用 `SubprocVecEnv`:

- 多 worker 时 `vec_env_mode = "subproc"`。
- 每个 worker 是单 scene 的 `MultiSceneEnvV17`。
- 每个 worker 绑定一个固定端口, 避免 worker 内部切场景。

验证:

- `python -m unittest tests.test_v18_multisim`
- `python -m unittest tests.test_v18_multisim tests.test_v17_curriculum`

### 4.3 结果

速度测试目录:

`models/v18_multisim_4worker_speedtest_20260508_062420`

V18 实测:

- `924 steps`
- `0.0436958 hours`
- 约 `5.87 steps/s`

V17 对照:

- 整体约 `12.33 steps/s`
- 最近窗口约 `14.06 steps/s`

结论:

V18 四窗口不是加速方向。瓶颈不在 PPO 模型计算, 而在:

- 多个 Unity/DonkeySim 渲染进程消耗 CPU。
- X11/窗口系统负载。
- `SubprocVecEnv` 同步等待最慢 worker。
- warmup 随机策略带来大量 reset / obstacle 日志, 同步开销更明显。

处理:

- 停止 V18 测试。
- 回到 V17 双 sim 主线。
- 停掉额外 `9095/9097` sim, 只保留 `9091/9093`。

---

## 5. V17 主线重启与恢复

### 5.1 原因

V18 测试前, 为释放 `9091/9093`, 停掉了旧 V17 训练进程。之后需要恢复 V17 主线。

### 5.2 处理

从旧 V17 warmup 最佳模型恢复, 不是随机初始化:

`models/v17_rawpink_36072_warmup_autocurr_20260508_055453/best_model_warmup_global.zip`

新训练目录:

`models/v17_rawpink_36072_warmup_autocurr_resume_20260508_063207`

启动命令要点:

```bash
python -u src/ppo_multitrack_v17.py \
  --auto-curriculum \
  --auto-curriculum-require-gate-success \
  --steps 3000000 \
  --save-dir models/v17_rawpink_36072_warmup_autocurr_resume_20260508_063207 \
  --sim remote \
  --port 9091 \
  --ports 9091 9093 \
  --lidar-num-sectors 72 \
  --lidar-fov-deg 360.0 \
  --image-channel-indices 0 1 2 3 4 5 \
  --ppo-n-steps 4096 \
  --ppo-batch-size 256 \
  --ppo-n-epochs 4 \
  --resume-path models/v17_rawpink_36072_warmup_autocurr_20260508_055453/best_model_warmup_global.zip
```

运行在 tmux:

`train-v17-resume-063207`

---

## 6. 加速排查

### 6.1 现象

用户希望训练加速。V18 四窗口没有加速后, 继续排查当前 V17 的系统瓶颈。

### 6.2 证据

当时的系统负载显示:

- GPU 利用率很低, 约 `7-19%`。
- 两个 DonkeySim 消耗主要 CPU:
  - `9093` 约 `190%`
  - `9091` 约 `136%`
- Python 训练进程约 `90-97%`
- `x11vnc` 约 `28%`
- 两个 `fluxbox` 异常空转:
  - 一个约 `100%`
  - 一个约 `70%`

结论:

瓶颈主要是 DonkeySim / X11 / CPU, 不是 GPU 或模型规模。

### 6.3 处理

尝试停掉:

- `x11vnc`
- 异常高 CPU 的 `fluxbox`

结果:

- `fluxbox` 空转被清除。
- `x11vnc` 被停掉后, 用户本地看不到画面。这是错误处理, 后续已恢复。
- 清理后训练速度没有明显提升到预期, 最近区间仍大约 `13-16 steps/s`。

后续更合理的加速方向:

- 不增加 sim 数量。
- 可以考虑重启 DonkeySim 到更低分辨率, 如 `640x480` 或 `512x384`。
- 可以给 obstacle reset/apply JSON debug 加开关或限频, 减少 stdout/I/O。
- 不优先扩大模型。当前瓶颈不是模型容量或 GPU 计算。

---

## 7. VNC 恢复

### 7.1 问题

停掉 `x11vnc` 后, 本地电脑无法继续查看 sim 画面。

### 7.2 恢复命令

用原参数恢复:

```bash
x11vnc \
  -display :1 \
  -rfbauth /home/longzhao/.vnc/passwd \
  -rfbport 5901 \
  -listen 127.0.0.1 \
  -localhost \
  -shared \
  -forever \
  -noxdamage \
  -o /home/longzhao/x11vnc.log \
  -bg
```

恢复后状态:

- `x11vnc` 监听 `127.0.0.1:5901`
- VNC desktop: `localhost:1`
- V17 训练进程未受影响。
- DonkeySim `9091/9093` 仍正常。

注意:

以后不要直接停 `x11vnc`, 除非确认不需要远程画面。若要省 CPU, 优先处理异常 `fluxbox` 或降低 sim 分辨率。

---

## 8. 当前训练状态与 avoid 判断

### 8.1 当前状态

截至最近一次检查:

- 总步数约 `603k`
- 当前阶段 `warmup_a`
- `warmup_a` stage steps 已恢复为约 `142k`
- `warmup_a` 最小晋级步数为 `120k`, 重启后没有被重置

`warmup` 已自动晋级:

```text
阶段晋级[warmup]: total_steps=460032, stage_steps=450032, success_counts={'ws': 3, 'gt': 2}
```

当前 `warmup_a` 最近窗口:

| 场景 | 最近 10 集 soft lap 总数 | gate success | collision rate |
|---|---:|---:|---:|
| WS | 已多次达到 `2-5` soft lap | 约 `4/10` | `0.0` 附近 |
| GT | 偶发 `3` soft lap, 但不够稳定 | 约 `1/10` | 低于门槛 |

这说明画面上看到“WS/GT 都能跑几圈”是成立的, 但 auto curriculum
按最近窗口判断, `gt` 还没有稳定到足够晋级。

### 8.2 是否手动进入 avoid

结论: 暂时不建议手动强跳。

原因:

- `warmup_a` 已超过 `120k stage steps`, 但最近 `gt` gate_success 仍不够稳定。
- 虽然已有 soft lap, 但窗口门控要求 WS/GT 都稳定达标。
- `avoid_static` 会把 free prob 压到 `0.10`, 即 90% episode 有障碍。
- 太早进入 avoid 可能打乱刚恢复的基础循迹。

建议:

- 继续观察 `gt` 最近 10 集窗口是否补到至少 2 个 gate success。
- 若当前趋势保持, auto curriculum 会自动进入 `avoid_static`。
- 如果 `gt` 长时间卡在 1/10, 再考虑调 `warmup_a` gate 或奖励。

---

## 9. 常用监控命令

查看训练进程:

```bash
pgrep -af 'src/ppo_multitrack_v17\.py'
```

查看端口:

```bash
ss -ltnp | rg ':(9091|9093|5901)\b'
```

查看训练日志:

```bash
tail -f models/v17_rawpink_36072_warmup_autocurr_resume_20260508_063207/stdout.log
```

查看课程窗口:

```bash
tail -f models/v17_rawpink_36072_warmup_autocurr_resume_20260508_063207/curriculum_window.jsonl
```

查看错误:

```bash
rg -n "Traceback|Training failed|RuntimeError|Crash checkpoint|auto curriculum stopped" \
  models/v17_rawpink_36072_warmup_autocurr_resume_20260508_063207/stdout.log
```

---

## 10. 后续决策点

1. `warmup_a` 当前窗口
   - 重点看 `gt` 最近 10 集 gate_success 是否从约 `1/10` 提升到至少 `2/10`。
   - 如果 WS/GT 同时满足窗口门控, 让系统自动进入 `avoid_static`。

2. `warmup_a` 持续卡住
   - 如果 `gt` 长时间无法补足成功窗口, 再讨论是否降低 gate、延长 warmup_a、或微调 offtrack/collision 奖励。

3. 进入 `avoid_static` 后
   - 重点观察 `steer_clip_hit`, collision rate, obstacle clearance, pass window valid rate。
   - 如果基础循迹明显退化, 不要直接进 `avoid_mixed`。

---

## 11. WS 障碍车闪现/消失问题

### 11.1 现象

WS 画面里有时看到障碍车闪一下就消失。日志里对应表现是 free episode
已经把 `last_set_position` 写成 park 点 `(-10, -6)`, 但即时 pose 回读仍停留在
上一条赛道目标附近, 下一次 reset 才真正回到 park 点。

### 11.2 根因

`ObstacleRuntime._park_fleet()` 使用非等待的 `teleport_pose()`, reset 与
DonkeySim pose 回读之间存在短暂竞态。active episode 后紧接 free episode 时,
障碍车可能还保留上一帧赛道位置, 看起来就是“出生/闪现后消失”。

### 11.3 修复

新增 `_park_car()` 统一 park 路径:

- 先 `stop_motion(hold_brake=True)`。
- 再使用 `place_pose(..., hold_brake=True, timeout_s=placement_timeout_s)`。
- `_park_fleet()` 和多余车辆回收都走同一个等待确认路径。

相关文件:

- `module/obstacle_runtime.py`
- `tests/test_v17_curriculum.py`

验证:

```bash
/home/longzhao/miniconda3/envs/donkey37/bin/python -m unittest \
  tests.test_v17_curriculum tests.test_v18_multisim

/home/longzhao/miniconda3/envs/donkey37/bin/python -m py_compile \
  module/obstacle_runtime.py tests/test_v17_curriculum.py
```

结果均通过。重启后第一条新 `obstacle_reset` 已显示 free/park 回读为
`x=-10.0, z=-6.0`, 不再是上一条赛道目标位置。

### 11.4 续跑方式

旧训练先用 `SIGINT` 停止, 生成/更新:

`models/v17_rawpink_36072_warmup_autocurr_resume_20260508_063207/final_model.zip`

随后在新 tmux 会话续跑:

`train-v17-parkfix-175745`

续跑命令关键点:

- `--resume-path .../final_model.zip`
- `--auto-curriculum-start-stage warmup_a`
- 保持原 `save_dir` 和原 `exp_tag`, 让 `curriculum_window.jsonl` 恢复 `warmup_a` 的阶段步数和窗口。

启动日志确认:

```text
resumed stage_steps=142024 (stage_start_total=460032)
```

因此这次重启没有把 `warmup_a` 的硬门槛重新从 0 开始计算。
