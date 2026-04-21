# DonkeySim 障碍车说明

## 概述

当前障碍车逻辑集中在 `module/obstacle.py`，测试入口是 `src/spawn_gt_obstacles.py`。

实现方式不是在场景里生成静态 mesh，而是启动额外的 DonkeySim client，让它以 `donkey` 车体进入同一个场景，因此障碍车是可见、可撞、可读位姿的。

当前只维护两类 preset：

- `gt` -> `generated_track`
- `ws` -> `waveshare`

当前默认障碍车车身颜色固定为 `pink`，RGB 为 `(255, 105, 180)`。
这个值来自当前真实相机采样下的颜色分离测试：相对蓝线、黄线和白色路面最稳。

## 底层约定

### 1. 定点放置

底层使用 DonkeySim 已验证可用的隐藏消息：

- `set_position`
- `node_position`

`set_position` 允许直接把障碍车放到指定位置和朝向。

### 2. 坐标缩放

Python 侧 `track.py`、telemetry、`info["pos"]` 使用同一套赛道坐标。  
Unity 网络层内部 world 坐标比这套坐标大 `8` 倍，所以：

- 发送 `set_position` 时：`world = telemetry * 8`
- 从 `node_position` / Unity world 读回时：`telemetry = world / 8`

### 3. 朝向换算

赛道切线朝向和 Donkey telemetry yaw 不是同一个定义。

当前统一使用：

```python
telemetry_yaw = 90.0 - track_heading_deg
```

对应接口在 `module/obstacle.py`：

- `track_heading_deg_to_telemetry_yaw_deg()`
- `telemetry_yaw_deg_to_track_heading_deg()`

`sample_track_target()` 现在返回的 `yaw_deg` 已经是可直接下发给 DonkeySim 的 telemetry yaw。

## 赛道内采样

障碍车目标位姿使用赛道内参数 `(progress_ratio, lateral_ratio)` 采样：

- `progress_ratio`：沿赛道弧长位置，范围 `[0, 1)`
- `lateral_ratio`：横向位置，`0` 靠右边界，`1` 靠左边界，`0.5` 近似中线

主要接口：

- `sample_track_target(...)`
- `sample_random_track_targets(...)`

默认随机生成规则：

- 默认生成 `2` 台
- 两台初始位置最少相隔 `3.0` 个 sim/world 坐标单位

## 主要接口

### 1. Fleet 入口

```python
from module.obstacle import spawn_preset_obstacle_fleet

fleet = spawn_preset_obstacle_fleet(scene="gt", host="127.0.0.1", port=9091)
```

可用快捷入口：

- `spawn_gt_obstacles(...)`
- `spawn_ws_obstacles(...)`

### 2. 单车接口

`DonkeyObstacleCar` 支持：

- `spawn()`
- `teleport_pose(...)`
- `place_pose(...)`
- `set_track_target(...)`
- `start_position_jitter(...)`
- `start_in_place_nudge(...)`
- `start_lane_pid(...)`
- `stop_motion()`
- `motion_mode()`

### 3. 位姿读取

单车：

- `get_obstacle_pose()`
- `obstacle_coordinates()`
- `update(agent_info=info)`
- `get_snapshot(agent_info=info)`

车队：

- `fleet.get_obstacle_poses()`
- `fleet.obstacle_coordinates()`
- `fleet.get_snapshots(agent_info=info)`
- `fleet.last_errors()`

`update()` / `get_snapshot()` 返回 `ObstacleSnapshot`，包含：

- `obstacle`: 障碍车位姿 `PoseState`
- `target`: 目标赛道点 `TrackTarget`
- `agent`: agent 位姿 `PoseState`
- `relative`: 障碍车相对 agent 的 `RelativeState`

## 动态模式

### 1. `static`

只放置，不运动，默认刹住。

### 2. `jitter`

沿赛道前后小幅抖动，位置直接通过 `set_position` 更新。  
建议只用很小幅度，否则视觉上不再像“原地缓动”的障碍车。

建议范围：

- `amplitude_m = 0.04 ~ 0.10`
- `period_s = 1.2 ~ 2.0`

### 3. `nudge`

以当前锚点为中心，沿车头前向和后向做小幅挪动，朝向保持不变。  
这个模式更适合模拟“原地前后挪一点”的障碍车。

### 4. `lane-pid`

接口名仍叫 `lane-pid` / `start_lane_pid()`，但当前内部实现已经换成：

- 横向：`pure pursuit`
- 纵向：`speed PID`

这台车会先按赛道切线方向放正，再沿指定 `lateral_ratio` 对应的车道位置持续绕圈。

建议初始参数：

- `target_speed = 0.60 ~ 0.75`
- `lookahead_m = 0.9 ~ 1.1`

## CLI 用法

兼容脚本：

```bash
/home/longzhao/miniconda3/envs/donkey37/bin/python /home/longzhao/Car/mysim/src/spawn_gt_obstacles.py --scene gt
```

常用参数：

- `--scene gt|ws`
- `--mode static|jitter|nudge|lane-pid`
- `--layout progress:lateral,...`
- `--count`
- `--min-separation-world`
- `--seed`
- `--speed`
- `--lookahead-m`
- `--jitter-amplitude`
- `--jitter-period`
- `--jitter-update-hz`
- `--duration`

示例：

```bash
# GT 随机两台静态障碍车
/home/longzhao/miniconda3/envs/donkey37/bin/python /home/longzhao/Car/mysim/src/spawn_gt_obstacles.py --scene gt

# WS 两台小幅 jitter
/home/longzhao/miniconda3/envs/donkey37/bin/python /home/longzhao/Car/mysim/src/spawn_gt_obstacles.py --scene ws --mode jitter --jitter-amplitude 0.06

# GT 一台 nudge
/home/longzhao/miniconda3/envs/donkey37/bin/python /home/longzhao/Car/mysim/src/spawn_gt_obstacles.py --scene gt --mode nudge --count 1 --jitter-amplitude 0.06

# GT 一台 pure pursuit + speed PID
/home/longzhao/miniconda3/envs/donkey37/bin/python /home/longzhao/Car/mysim/src/spawn_gt_obstacles.py --scene gt --mode lane-pid --count 1 --speed 0.68 --lookahead-m 1.0
```

## Python 示例

```python
from module.obstacle import spawn_preset_obstacle_fleet

fleet = spawn_preset_obstacle_fleet(
    scene="gt",
    host="127.0.0.1",
    port=9091,
    layout=[(0.55, 0.82), (0.08, 0.50)],
)

car1, car2 = fleet.cars

car1.start_in_place_nudge(
    progress_ratio=0.55,
    lateral_ratio=0.82,
    amplitude_m=0.06,
    period_s=1.8,
)

car2.start_lane_pid(
    target_speed=0.68,
    progress_ratio=0.08,
    lateral_ratio=0.50,
    lookahead_m=1.0,
)

poses = fleet.get_obstacle_poses()
snapshots = fleet.get_snapshots(agent_info=info)
```

## 已验证行为

- GT / WS 都支持赛道内随机两车生成
- 初始位置最小间距约束按 sim/world 坐标生效
- `jitter` 可用于小幅前后抖动
- `nudge` 可用于原位前后小幅挪动
- `lane-pid` 当前实际为 `pure pursuit + speed PID`
- 修正 yaw 映射后，巡线车放置时车头前向与赛道 progress 前向一致

## 备注

- 当前脚本名仍叫 `spawn_gt_obstacles.py`，但实际上同时支持 `gt` 和 `ws`
- 当前 CLI 里的 `lane-pid` 名称是兼容旧接口，控制器实现已经不是传统横向 PID
