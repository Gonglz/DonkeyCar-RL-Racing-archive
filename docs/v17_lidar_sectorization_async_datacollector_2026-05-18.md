# V17 LiDAR sectorization 与 DataCollector 异步化复盘

日期：2026-05-18

## 范围

本轮只改主链路里的 LiDAR 特征构建和 DataCollector 抖动，不改 V17 图像前处理。V17 仍使用单摄像头 + LiDAR + 自车传感器输入，TensorRT FP16 actor engine 路径保持不变。

## 已完成改动

### 1. LiDAR sectorization 前移

改动文件：

- `/home/jetson/mycar/runtime_monitor.py`
- `/home/jetson/mycar/v17_pilot.py`

原来 `V17Pilot._build_lidar_obs()` 每帧从约 1147 个 raw LiDAR ranges 构建 72 sector range + 72 valid mask，微基准 p50 约 43.8ms。现在 ROS LiDAR helper 收到 scan 后直接预计算：

- `sector_ranges`: 72 维
- `sector_valid`: 72 维

`V17Pilot` 优先使用这两组预计算数组，raw `ranges` 路径保留为 fallback。sectorization 保持旧逻辑语义：

- 20% quantile 聚合，不改成 min；
- finite 且 `>= range_min` 即有效；
- `> range_max` 的点裁剪为 `range_max`；
- sector 边界规则与旧 fallback 保持一致。

### 2. DataCollector 写盘异步化

新增 `AsyncLogWriter`，CSV 和 LiDAR raw JSONL 写入放到后台线程，主 loop 只做非阻塞入队。队列满时丢弃 debug sample，避免阻塞车辆控制 loop。

### 3. Jetson telemetry 后台缓存

第一次 shadow 后发现 DataCollector 仍有 p99 约 471ms 长尾。拆分确认 `_read_thermal_zones()` 非缓存读 p50 约 301ms、p99 约 576ms，主要来自系统/电源/thermal telemetry 读取，而不是 CSV flush。

新增 `AsyncTelemetryCache`：

- 后台线程每 1s 刷新 Jetson 温度、CPU/GPU 负载、内存、电源、WiFi 等数据；
- DataCollector 主 loop 只读取最近一次快照；
- Jetson 微基准中缓存读取 p50 约 0.09ms、p99 约 0.34ms。

## 验证

本地：

- `py -m py_compile runtime_monitor.remote.py v17_pilot.remote.py test_runtime_optimizations.py`
- `py -m unittest -v test_runtime_optimizations.py`
- 结果：6 tests OK

Jetson：

- `python -m py_compile runtime_monitor.py v17_pilot.py test_runtime_optimizations.py`
- `python -m unittest -v test_runtime_optimizations.py`
- LiDAR helper 脚本 `py_compile` OK
- 结果：6 tests OK

新增测试覆盖：

- sectorized LiDAR 输出维度与 valid mask；
- sector 边界兼容旧 fallback；
- `range_max` 裁剪兼容旧 fallback；
- `V17Pilot` 优先使用预计算 sector arrays；
- async log writer 写 CSV/JSONL；
- async telemetry cache 非阻塞读取。

## 60 秒 shadow A/B 结果

baseline 是本地已有 TensorRT 60 秒 shadow：

`/home/jetson/mycar/monitor_logs/v17_trt_benchmark_20260518_040300/runtime_shadow_tensorrt_60s`

本轮最终结果：

`/home/jetson/mycar/monitor_logs/v17_lidar_async_telemetry_smoke_20260518_053731/runtime_shadow_tensorrt_60s`

| 指标 | baseline TensorRT 60s | 本轮最终 60s | 变化 |
| --- | ---: | ---: | ---: |
| V17 latency p50 | 168.570 ms | 87.153 ms | -48.3% |
| V17 latency p95 | 253.958 ms | 124.753 ms | -50.9% |
| effective FPS mean | 4.270 | 10.641 | +149.2% |
| vehicle loop p50 | 192.050 ms | 87.800 ms | -54.3% |
| vehicle loop p95 | 696.175 ms | 130.800 ms | -81.2% |
| LiDAR scan age p50 | 241.200 ms | 244.300 ms | +1.3% |
| LiDAR scan age p95 | 292.510 ms | 325.250 ms | +11.2% |
| GPU load mean | 13.684% | 10.596% | -22.6% |
| power in mean | 3093 mW | 3712 mW | +20.0% |

Part profile 对比：

| Part | baseline p50 / p90 / p99 / max | 本轮最终 p50 / p90 / p99 / max |
| --- | --- | --- |
| V17Pilot | 165.47 / 239.70 / 279.69 / 311.08 ms | 86.76 / 103.37 / 181.05 / 211.80 ms |
| DataCollector | 0.04 / 403.27 / 537.42 / 566.49 ms | 0.04 / 3.14 / 5.67 / 8.69 ms |

## 结论

这轮优化是有效的：核心收益来自把 LiDAR sectorization 从 V17Pilot 主前处理路径前移到 LiDAR ROS helper，并把 DataCollector 的慢写盘和慢 telemetry 读取移出 vehicle loop。最终 TensorRT shadow 下，vehicle loop p95 从约 696ms 降到约 131ms，DataCollector p99 从约 537ms 降到约 5.7ms。

剩余主要瓶颈已经回到 V17 图像语义前处理和 TensorRT actor 调用本身。当前最终 run 中 V17Pilot p50 约 86.8ms、p90 约 103.4ms，但 p99 仍有约 181ms；后续若继续压延时，应单独做图像前处理轻量化/缓存策略、TensorRT I/O buffer 复用、摄像头帧率与 loop rate 对齐，而不是继续优先抠 DataCollector。

## 注意事项

- 本轮没有改图像前处理语义，所以模型输入 domain 逻辑不变。
- LiDAR raw ranges 仍保留在日志里，便于事后复盘；如果正式比赛/高速部署追求极限稳定，可以增加开关禁用 raw JSONL 或降低 raw 记录频率。
- CPU load 和 power 在短 run 中上升，可能来自更高 effective FPS、后台 telemetry 线程和 run 间状态差异；建议用 180s A/B 再确认功耗趋势。
- LiDAR scan age p95 没有改善，且略升到约 325ms；这说明 LiDAR 数据源/ROS bridge 更新频率仍是独立瓶颈，当前改动主要减少主 loop 消耗，不会让 LiDAR 本身更快。
