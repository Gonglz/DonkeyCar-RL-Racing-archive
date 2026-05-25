# V17 优化后 PyTorch vs TensorRT A/B

日期：2026-05-18

目的：在 LiDAR sectorization、DataCollector 异步写入、Jetson telemetry cache 都已启用之后，重新量化 PyTorch 后端和 TensorRT 后端的单独差异。

## 实验口径

输出目录：

`/home/jetson/mycar/monitor_logs/v17_optimized_backend_ab_20260518_060718`

两组 run 都使用当前优化后的主链路：

`单 CSI 摄像头 + 360 度 LiDAR /scan + RP2040 自车传感器 + V17 shadow + DataCollector`

对比后端：

- PyTorch：`PyTorch/manual SB3 RecurrentPPO actor`，设备为 PyTorch CUDA。
- TensorRT：`TensorRT FP16 actor`，engine 为 `/home/jetson/mycar/models/v17_actor_fp16.engine`。

注意：这不是 CPU PyTorch vs TensorRT，而是 PyTorch CUDA actor vs TensorRT FP16 actor。

## Summary 对比

| 指标 | 优化后 PyTorch 60s | 优化后 TensorRT 60s | 变化 |
| --- | ---: | ---: | ---: |
| V17 latency p50 | 106.325 ms | 94.603 ms | -11.0% |
| V17 latency p95 | 202.382 ms | 199.075 ms | -1.6% |
| effective FPS mean | 7.964 | 9.476 | +19.0% |
| loop dt p50 | 107.100 ms | 95.500 ms | -10.8% |
| loop dt p95 | 203.020 ms | 199.820 ms | -1.6% |
| LiDAR scan age p50 | 256.500 ms | 248.000 ms | -3.3% |
| LiDAR scan age p95 | 327.700 ms | 326.800 ms | -0.3% |
| CPU load mean | 58.070% | 57.233% | -1.4% |
| GPU load mean | 13.719% | 12.398% | -9.6% |
| power in mean | 3693.981 mW | 3667.676 mW | -0.7% |
| frames logged | 109 | 109 | same |

## Preprocess / Actor residual 拆分

`actor_residual = pilot_inference_latency_ms - pilot_preprocess_latency_ms`

| 指标 | PyTorch p50 | TensorRT p50 | 变化 |
| --- | ---: | ---: | ---: |
| total latency | 106.325 ms | 94.603 ms | -11.0% |
| preprocess | 79.995 ms | 81.413 ms | +1.8% |
| actor residual | 24.325 ms | 12.474 ms | -48.7% |
| loop dt | 107.100 ms | 95.500 ms | -10.8% |

| 指标 | PyTorch p95 | TensorRT p95 | 变化 |
| --- | ---: | ---: | ---: |
| total latency | 202.382 ms | 199.075 ms | -1.6% |
| preprocess | 178.629 ms | 187.059 ms | +4.7% |
| actor residual | 39.913 ms | 17.215 ms | -56.9% |
| loop dt | 203.020 ms | 199.820 ms | -1.6% |

## Vehicle part profile

| Part | PyTorch p50 / p90 / p99 / max | TensorRT p50 / p90 / p99 / max |
| --- | --- | --- |
| V17Pilot | 105.13 / 132.06 / 221.70 / 1275.26 ms | 89.29 / 184.94 / 215.12 / 229.96 ms |
| DataCollector | 0.04 / 3.29 / 5.42 / 10.93 ms | 0.04 / 3.19 / 6.78 / 13.06 ms |

PyTorch run 有一个 1275ms 的启动/长尾 outlier；p50/p95 仍然可用，但后续正式结论建议用 180 秒 A/B 降低短 run 偶然性。

## 结论

LiDAR 和 DataCollector 优化之后，TensorRT/CUDA 的模型后端收益已经能被单独看出来：

- actor residual p50 从 24.325ms 降到 12.474ms，约 -48.7%。
- actor residual p95 从 39.913ms 降到 17.215ms，约 -56.9%。
- 端到端 V17 latency p50 从 106.325ms 降到 94.603ms，约 -11.0%。
- effective FPS mean 从 7.964 提升到 9.476，约 +19.0%。

但端到端 p95 只改善约 1.6%。原因是现在长尾主要来自视觉语义前处理和实时系统抖动，而不是 actor 后端。当前最有价值的下一步不是继续折腾 ONNX/TensorRT，而是拆视觉前处理阶段耗时，并做不改变语义的实现层优化。

## 注意事项

- TensorRT run 日志中出现一次 ROS Python2 订阅线程关闭时的 `socket.close()` 异常，但 run 完整结束，LiDAR frames 正常记录；这更像 ROS helper 的 shutdown/连接边界噪声，不影响本次 summary 生成。
- LiDAR scan age 仍在 250-330ms 区间，说明 LiDAR 数据源/ROS bridge freshness 仍未被 TensorRT 影响。
- 这组 60 秒 A/B 已足够证明 TensorRT actor 后端收益；若要写最终部署报告，建议补一组 180 秒同口径 A/B。
