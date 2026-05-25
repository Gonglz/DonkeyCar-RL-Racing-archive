# V17 LiDAR Stale Gate 与 PMIC Thermal 核查结果

日期：2026-05-24
Jetson：`jetson@192.168.1.176`
验证目录：`/home/jetson/mycar/monitor_logs/v17_lidar_stale_pmic_validation_20260524_2325`

## 1. 口径

本轮只补齐 P2 缺口：

- 单独验证 LiDAR stale scan age gate。
- 核查 PMIC 固定 100C 的来源。

未跑 active，不评价模型避障效果。

## 2. LiDAR stale 注入方法

新增工具：

`/home/jetson/mycar/tools/publish_stale_lidar_scan.py`

该脚本使用 ROS Melodic Python2 发布 `/stale_scan`，消息类型为 `sensor_msgs/LaserScan`，但 `header.stamp` 固定为 `rospy.Time(1, 0)`。因此 runtime 收到 scan 后会看到极大的 `scan_age_ms`。

执行方式：

- 临时启动 `roscore`。
- 启动 stale publisher：`/usr/bin/python tools/publish_stale_lidar_scan.py /stale_scan 180 10`
- 运行 V17 shadow startup gate：

```bash
python runtime_monitor.py drive \
  --model /home/jetson/mycar/models/v17_postpass_hard_gate_final_model.zip \
  --type v17 \
  --js \
  --control-mode shadow \
  --shadow-duration 5 \
  --log-dir /home/jetson/mycar/monitor_logs/v17_lidar_stale_pmic_validation_20260524_2325/fault_lidar_stale_require_lidar_rerun \
  --run-label fault_lidar_stale_require_lidar_rerun \
  --track-condition lidar_stale_pmic_validation \
  --lidar-topic /stale_scan \
  --no-start-lidar-driver \
  --require-lidar \
  --max-lidar-age-ms 350 \
  --shadow-engine /home/jetson/mycar/models/v17_actor_fp16.engine \
  --shadow-engine-metadata /home/jetson/mycar/models/v17_actor_export.json
```

## 3. LiDAR stale 结果

run：

`fault_lidar_stale_require_lidar_rerun`

结果：

- exit：2
- Vehicle loop：未启动
- CSV/summary：无，符合 startup gate 阻断预期
- `preflight_report.json`：已生成

关键日志：

```text
Startup safety gate failed: require_lidar startup check failed:
connected=True, frames=45, age_ms=1779681389560.4214,
last_update=1.0, valid_points=360, points_total=360, parse_errors=0
```

判断：

- runtime 已实际收到 stale LaserScan。
- scan 有效点数为 360，解析错误为 0，说明不是 topic 缺失或解析失败。
- `last_update=1.0` 导致 `scan_age_ms` 远大于 `max_lidar_age_ms=350`。
- `require_lidar + max_lidar_age_ms` 能在 Vehicle loop 启动前 fail-fast。

P2 LiDAR stale gate 已闭环。

## 4. PMIC Thermal Zone 核查

sysfs snapshot：

```text
/sys/devices/virtual/thermal/thermal_zone0 type=AO-therm temp=31000
/sys/devices/virtual/thermal/thermal_zone1 type=CPU-therm temp=23000
/sys/devices/virtual/thermal/thermal_zone2 type=GPU-therm temp=21000
/sys/devices/virtual/thermal/thermal_zone3 type=PLL-therm temp=19500
/sys/devices/virtual/thermal/thermal_zone4 type=PMIC-Die temp=100000
/sys/devices/virtual/thermal/thermal_zone5 type=thermal-fan-est temp=21500
/sys/devices/virtual/thermal/thermal_zone6 type=iwlwifi temp=30000
```

`tegrastats` 同步确认：

```text
PLL@19C CPU@23.5C iwlwifi@28C PMIC@100C GPU@20.5C AO@31.5C thermal@22C
```

trip point：

```text
thermal_zone4 type=PMIC-Die
trip_point_0_temp=120000
trip_point_0_type=active
mode=enabled
policy=step_wise
```

判断：

- PMIC=100C 不是 `runtime_monitor.py` 映射错误；系统 sysfs 和 `tegrastats` 都这样报。
- 10min shadow 第二轮中 CPU/GPU/AO/PLL/Fan 均低且稳定，FPS 和 latency 没有热降频/热失控形态。
- 当前应把 PMIC 固定 100C 记录为 Jetson PMIC sensor/driver 上报异常或板级 PMIC 读数风险，不能直接作为 V17 部署热失控证据。
- 如果后续要把热安全做严，需要单独核查板型、JetPack/内核驱动、PMIC thermal driver，或使用外部温度计/电源侧监控交叉验证。

## 5. 清理状态

临时进程已停止：

- stale publisher pid `26755`
- stale publisher rerun pid `29108`
- temporary roscore pid `26297`

没有遗留 `runtime_monitor.py drive` 或 stale publisher 进程。

## 6. 结论

本轮补齐了上轮剩余的 P2 关键缺口：

- LiDAR stale gate 已实测通过。
- PMIC=100C 来源已定位到 Jetson 系统 thermal zone / `tegrastats`，不是 runtime monitor 口径错误。

当前端侧部署安全证据链已覆盖：

- engine/metadata preflight fail-fast。
- LiDAR disabled fail-fast。
- LiDAR stale fail-fast。
- RP2040 missing fail-fast。
- inference timeout 计数。
- shadow 不接管。
- 10min TensorRT shadow 稳定性。
