# V17 Jetson 端侧部署工程成果报告

日期：2026-05-18
整理更新：2026-05-24

## 0. 范围边界

本报告只复盘已经完成、并且有 Jetson 本机数据支撑的端侧部署工程成果：

- PyTorch/SB3 V17 actor 到 ONNX 的导出。
- TensorRT FP16 engine 构建、正确性校验和 runtime 集成。
- 不依赖 PyCUDA 的 CUDA runtime buffer/stream 封装。
- LiDAR sectorization 前移。
- DataCollector 异步写入。
- Jetson telemetry 后台缓存。
- runtime monitor / shadow A/B / summary 数据链路。

视觉前端、视觉语义前处理和 `CanonicalSemanticWrapper` 优化不计入本轮部署工程成果。它仍然是当前端到端长尾的重要解释因素，但细节已经拆分到：

`docs/v17_vision_frontend_separate_analysis_2026-05-24.md`

模型策略效果也不计入本轮端侧部署成果。实车 active 已确认会出现乱跑/乱爬，原因按当前判断归为训练侧策略质量不足，包括奖励设计、数据分布、泛化和闭环策略本身；后续不再用“是否避障成功、是否跑通赛道、是否 obstacle recovery 成功”评价端侧部署链路。

后续待审核实验计划见：

`docs/v17_post_deployment_experiment_plan_2026-05-24.md`

## 1. 当前部署主链路

V17 当前端侧主链路是：

`单 CSI 摄像头 + 360 度 LiDAR /scan + RP2040 自车传感器 -> V17 RecurrentPPO actor -> action adapter / safety -> DonkeyCar 输出`

部署模型：

`/home/jetson/mycar/models/v17_postpass_hard_gate_final_model.zip`

实际 actor 输入输出形状：

| 输入/输出 | 形状 |
| --- | --- |
| `image` | `(1, 6, 128, 128)` |
| `state` | `(1, 7)` |
| `lidar` | `(1, 144)` |
| `lidar_meta` | `(1, 2)` |
| LSTM `h/c` | `(2, 1, 256)` |
| `action` | `(1, 3)` |

`lidar=144` 表示 72 个 sector range + 72 个 valid mask。当前推理使用 360 度 LiDAR，不是 180 度；最新 shadow CSV 中 `angle_min=-3.141593`、`angle_max=3.141593`，span 约 360 度，`points_total=1147`。

## 2. Jetson 部署环境

| 项目 | 当前结果 |
| --- | --- |
| 机器 | `jetson@192.168.1.176` |
| Python 环境 | `/home/jetson/env` |
| 系统 | Jetson L4T R32.5.2 / Ubuntu 18.04.6 |
| CUDA | 10.2 |
| TensorRT | 7.1.3.0，Python `tensorrt` 可 import |
| ONNX Runtime | venv 中未安装；当前部署不依赖 ONNX Runtime |
| OpenCV | 4.1.1 |
| OpenCV CUDA | 不可用；视觉前端 GPU 化另行评估 |
| torch/cv2 兼容 | import 时需要 `LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1` |

实际运行入口使用：

```bash
. /home/jetson/env/bin/activate
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 \
PYTHONPATH=/home/jetson/mycar/tools \
python runtime_monitor.py drive ...
```

`LD_PRELOAD` 是兼容性处理，不是性能优化。

## 3. 已完成工程成果

### 3.1 ONNX 导出

ONNX 在本项目中是 PyTorch/SB3 actor 到 TensorRT 的中间格式，不是实际加速器。导出范围只包含实时驾驶需要的 deterministic actor：

`image/state/lidar/lidar_meta + h/c -> action + next_h/next_c`

没有导出 critic/value head、SB3 训练封装、optimizer 或训练时采样逻辑。

主要脚本：

- `/home/jetson/mycar/tools/export_v17_actor_onnx.py`
- 本地副本：`export_v17_actor_onnx.py`

关键实现点：

- `V17ActorONNX` 复刻 actor 的 feature extractor、LSTM actor、policy net、action net。
- `infer_shape()` 从 `policy.pth` 自动推断真实 shape，避免训练默认参数误导部署。
- `FixedLayerNorm1D` 用固定 axis 的 LayerNorm 表达兼容 TensorRT 7。
- 避免 `chunk(dim=-1)` 这类 TensorRT 7 不友好的 ONNX 表达。
- 固定 batch=1、固定输入尺寸、opset 11。

输出 artifacts：

| 文件 | 大小 | 说明 |
| --- | ---: | --- |
| `/home/jetson/mycar/models/v17_actor.onnx` | 7,144,791 bytes | actor ONNX graph |
| `/home/jetson/mycar/models/v17_actor_export.json` | 485 bytes | 输入输出 shape metadata |

### 3.2 TensorRT FP16 engine

构建命令：

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=/home/jetson/mycar/models/v17_actor.onnx \
  --explicitBatch \
  --workspace=512 \
  --fp16 \
  --saveEngine=/home/jetson/mycar/models/v17_actor_fp16.engine \
  --verbose
```

输出 engine：

| 文件 | 大小 |
| --- | ---: |
| `/home/jetson/mycar/models/v17_actor_fp16.engine` | 4,102,829 bytes |

model-only `trtexec --loadEngine` 结果：

| 指标 | 数值 |
| --- | ---: |
| GPU compute mean | 约 3.72 ms |
| Host latency mean | 约 3.98 ms |
| Throughput | 约 245.7 qps |

这个数字只代表 actor network 本体，不包含相机、视觉语义前处理、LiDAR、DataCollector 或 DonkeyCar vehicle loop。

### 3.3 CUDA runtime / PyCUDA 决策

当前没有安装 PyCUDA，也没有必要为了当前部署路径单独安装 PyCUDA。TensorRT 执行 engine 时已经使用 CUDA kernels；部署侧真正需要的是稳定地管理 TensorRT binding buffer、host/device copy、stream 和 LSTM state。

主要文件：

- `/home/jetson/mycar/tools/v17_trt_runtime.py`
- 本地副本：`v17_trt_runtime.py`

关键实现：

- `_CudaRuntime` 用 `ctypes.CDLL` 加载 `libcudart.so`。
- 使用 `cudaMalloc` 为每个 binding 分配 device buffer。
- 使用 `cudaMemcpyAsync` 做 H2D/D2H。
- 创建 CUDA stream。
- `V17TensorRTActor.predict_np()` 维护 persistent LSTM `h/c`。

当前收益是环境风险更低、buffer 分配稳定、无需在 Jetson Nano 上额外编译 PyCUDA。后续 pinned host memory 或更细的 copy 优化可以评估，但不是本轮最大收益来源。

### 3.4 TensorRT 集成

TensorRT 已接入 V17Pilot 和 runtime_monitor：

- `/home/jetson/mycar/v17_pilot.py`
- `/home/jetson/mycar/runtime_monitor.py`

本地副本：

- `v17_pilot.remote.py`
- `runtime_monitor.remote.py`

新增能力：

- `v17_pilot.py --engine`
- `v17_pilot.py --engine-metadata`
- `runtime_monitor.py --shadow-engine`
- `runtime_monitor.py --shadow-engine-metadata`

在 `runtime_monitor.py drive --control-mode shadow` 下，TensorRT 只用于 shadow pilot 推理，actuator path 仍保持 user/manual，不主动接管车辆。

### 3.5 正确性验证

TensorRT runtime smoke：

```bash
PYTHONPATH=/home/jetson/mycar/tools \
python tools/check_v17_trt_runtime.py
```

PyTorch vs TensorRT deterministic action 对齐：

```bash
PYTHONPATH=/home/jetson/mycar/tools \
python tools/compare_v17_torch_trt.py --tolerance 0.02
```

结果：

| 项目 | 数值 |
| --- | ---: |
| PyTorch action | `[-0.029122, -0.249469, -0.026623]` |
| TensorRT action | `[-0.029739, -0.249512, -0.026794]` |
| max abs diff | `0.000617` |
| tolerance | `0.02` |

结论：TensorRT FP16 actor 与 PyTorch actor 输出误差远低于当前部署阈值。

## 4. 性能证据链

### 4.1 初始 TensorRT runtime shadow A/B

输出目录：

`/home/jetson/mycar/monitor_logs/v17_trt_benchmark_20260518_040300`

口径：

`单 CSI 摄像头 + RP2040 自车传感器 + LiDAR /scan + V17Pilot + DataCollector`

| 指标 | PyTorch shadow 60s | TensorRT shadow 60s | 变化 |
| --- | ---: | ---: | ---: |
| inference p50 | 180.689 ms | 168.570 ms | -6.71% |
| inference p95 | 294.797 ms | 253.958 ms | -13.85% |
| effective FPS mean | 4.028 | 4.270 | +6.01% |
| CPU load mean | 51.224% | 46.525% | -9.17% |
| GPU load mean | 29.881% | 13.684% | -54.21% |
| power in mean | 3307.576 mW | 3093.224 mW | -6.48% |
| loop dt p50 | 189.300 ms | 192.050 ms | 基本持平 |
| loop dt p95 | 705.870 ms | 696.175 ms | -1.37% |
| LiDAR scan age p50 | 235.300 ms | 241.200 ms | 基本持平 |

这说明 TensorRT actor 有端侧收益，但当时 full runtime 仍被 LiDAR 构建、DataCollector 和 loop 抖动限制。

### 4.2 180 秒 TensorRT runtime shadow A/B

输出目录：

`/home/jetson/mycar/monitor_logs/v17_trt_benchmark_180s_20260518_042116`

| 指标 | PyTorch 180s | TensorRT 180s | 变化 |
| --- | ---: | ---: | ---: |
| inference p50 | 178.159 ms | 168.947 ms | -5.17% |
| inference p95 | 264.161 ms | 249.953 ms | -5.38% |
| effective FPS mean | 3.975 | 4.274 | +7.52% |
| CPU load mean | 49.107% | 47.493% | -3.29% |
| GPU load mean | 28.938% | 20.764% | -28.25% |
| power in mean | 3274.272 mW | 3166.666 mW | -3.29% |
| loop dt p50 | 183.000 ms | 177.500 ms | -3.01% |
| loop dt p95 | 694.640 ms | 723.170 ms | +4.11% |
| LiDAR scan age p50 | 235.300 ms | 234.900 ms | 基本持平 |

180 秒结果确认：TensorRT actor 收益存在，但无法单独解决完整 vehicle loop 的 p95 抖动。

### 4.3 LiDAR sectorization + DataCollector + telemetry 优化

本轮优化只改主链路里的 LiDAR 特征构建和 DataCollector/telemetry 抖动，不改 V17 图像前处理。

#### LiDAR sectorization 前移

旧路径中，`V17Pilot._build_lidar_obs()` 每帧从约 1147 个 raw LiDAR ranges 构建 72 sector range + 72 valid mask，微基准 p50 约 43.8 ms。

现在 ROS LiDAR helper 收到 scan 后预计算：

- `sector_ranges`: 72 维。
- `sector_valid`: 72 维。

`V17Pilot` 优先使用预计算数组，raw `ranges` 路径保留为 fallback。语义保持旧逻辑：

- 20% quantile 聚合。
- finite 且 `>= range_min` 即有效。
- `> range_max` 的点裁剪为 `range_max`。
- sector 边界规则与旧 fallback 保持一致。

#### DataCollector 异步写入

新增 `AsyncLogWriter`，CSV 和 LiDAR raw JSONL 写入放到后台线程。主 loop 只做非阻塞入队，队列满时丢弃 debug sample，避免阻塞车辆控制 loop。

#### Jetson telemetry cache

第一次异步写盘后，DataCollector p99 仍有约 471 ms 长尾。进一步拆分发现 `_read_thermal_zones()` 非缓存读 p50 约 301 ms、p99 约 576 ms，主要来自系统/电源/thermal telemetry。

新增 `AsyncTelemetryCache`：

- 后台线程每 1 秒刷新 Jetson 温度、CPU/GPU、内存、电源、WiFi 等数据。
- DataCollector 主 loop 只读取最近一次快照。
- Jetson 微基准中缓存读取 p50 约 0.09 ms、p99 约 0.34 ms。

验证：

- 本地 `py_compile` OK。
- 本地 `py -m unittest -v test_runtime_optimizations.py`：6 tests OK。
- Jetson `python -m unittest -v test_runtime_optimizations.py`：6 tests OK。

新增测试覆盖：

- sectorized LiDAR 输出维度与 valid mask。
- sector 边界兼容旧 fallback。
- `range_max` 裁剪兼容旧 fallback。
- `V17Pilot` 优先使用预计算 sector arrays。
- async log writer 写 CSV/JSONL。
- async telemetry cache 非阻塞读取。

### 4.4 最终 TensorRT shadow 效果

baseline：

`/home/jetson/mycar/monitor_logs/v17_trt_benchmark_20260518_040300/runtime_shadow_tensorrt_60s`

最终优化后：

`/home/jetson/mycar/monitor_logs/v17_lidar_async_telemetry_smoke_20260518_053731/runtime_shadow_tensorrt_60s`

| 指标 | TensorRT baseline | 最终优化后 | 变化 |
| --- | ---: | ---: | ---: |
| V17 latency p50 | 168.570 ms | 87.153 ms | -48.3% |
| V17 latency p95 | 253.958 ms | 124.753 ms | -50.9% |
| effective FPS mean | 4.270 | 10.641 | +149.2% |
| vehicle loop p50 | 192.050 ms | 87.800 ms | -54.3% |
| vehicle loop p95 | 696.175 ms | 130.800 ms | -81.2% |
| DataCollector p99 | 537.42 ms | 5.67 ms | -98.9% |
| LiDAR age p50 | 241.200 ms | 244.300 ms | 基本持平 |
| LiDAR age p95 | 292.510 ms | 325.250 ms | 未改善 |

解释：

- p50 和 FPS 的大提升主要来自 LiDAR sectorization 前移。
- p95 的大提升主要来自 DataCollector 写盘异步化和 telemetry cache。
- LiDAR scan age 没有改善，说明这轮降低的是主 loop 计算和日志阻塞，不会让 LiDAR 驱动/ROS bridge 本身更快。

### 4.5 优化后 PyTorch vs TensorRT A/B

输出目录：

`/home/jetson/mycar/monitor_logs/v17_optimized_backend_ab_20260518_060718`

目的：在 LiDAR sectorization、DataCollector 异步写入、Jetson telemetry cache 都已启用后，重新量化 TensorRT 后端的单独收益。

| 指标 | 优化后 PyTorch 60s | 优化后 TensorRT 60s | 变化 |
| --- | ---: | ---: | ---: |
| V17 latency p50 | 106.325 ms | 94.603 ms | -11.0% |
| V17 latency p95 | 202.382 ms | 199.075 ms | -1.6% |
| effective FPS mean | 7.964 | 9.476 | +19.0% |
| loop dt p50 | 107.100 ms | 95.500 ms | -10.8% |
| loop dt p95 | 203.020 ms | 199.820 ms | -1.6% |
| GPU load mean | 13.719% | 12.398% | -9.6% |
| power in mean | 3693.981 mW | 3667.676 mW | -0.7% |

拆分 `actor_residual = pilot_inference_latency_ms - pilot_preprocess_latency_ms`：

| 指标 | PyTorch p50 | TensorRT p50 | 变化 |
| --- | ---: | ---: | ---: |
| total latency | 106.325 ms | 94.603 ms | -11.0% |
| preprocess | 79.995 ms | 81.413 ms | +1.8% |
| actor residual | 24.325 ms | 12.474 ms | -48.7% |

| 指标 | PyTorch p95 | TensorRT p95 | 变化 |
| --- | ---: | ---: | ---: |
| total latency | 202.382 ms | 199.075 ms | -1.6% |
| preprocess | 178.629 ms | 187.059 ms | +4.7% |
| actor residual | 39.913 ms | 17.215 ms | -56.9% |

结论：

- LiDAR/DataCollector 优化后，TensorRT actor 后端收益已经能被单独看出来：actor residual p50 约减半，p95 下降约 57%。
- 端到端 p50/FPS 也能看到 TensorRT 收益。
- 端到端 p95 只改善约 1.6%，长尾仍由视觉前端和实时系统抖动主导；视觉细节不在本报告展开。

## 5. 工程成熟度判断

当前工作已经超过“能跑 demo”的层级，具备端侧部署工程成果的基本证据链：

- 有明确的目标硬件和软件栈记录。
- 有模型真实 shape、actor-only 导出和 TensorRT engine artifact。
- 有 TensorRT 与 PyTorch action diff 正确性校验。
- 有 shadow 模式，不直接接管车辆。
- 有 runtime monitor CSV/JSONL、summary、part profile、功耗/GPU/CPU 指标。
- 有 LiDAR/DataCollector/telemetry 优化前后 A/B。
- 有单独量化的 TensorRT 后端收益。

本报告的工程成熟度只评价端侧部署能力，不评价策略是否学会避障。active 乱跑是已知模型效果限制，不能用来否定 ONNX/TensorRT、runtime monitor、LiDAR/DataCollector 或安全观测链路本身。

但这还不是实车长期运行的 production-ready 状态。还缺少：

- 多轮 180s/300s/10min shadow 重复实验，降低短 run 偶然性。
- 部署安全阻断矩阵：engine missing、LiDAR disconnected/stale、sensor missing、inference timeout、logging/telemetry failure。
- LiDAR scan header age 与 receipt age 拆分。
- DataCollector raw logging 开关 A/B。
- 推理超时、engine 加载失败、LiDAR stale 等阻断或 fallback 策略的实测验证。

这些已整理到后续实验计划中。

## 6. 风险和注意事项

1. TensorRT engine 与 Jetson 软件栈强绑定。当前 engine 面向 Jetson Nano / CUDA 10.2 / TensorRT 7.1.3；换 JetPack、TensorRT 或设备后应重新 build。

2. LSTM `h/c` 是 actor 状态的一部分。episode reset、人工接管、active/shadow 切换时要明确 reset 策略。

3. ONNX Runtime 未安装，也不是当前部署路径要求。当前链路是 PyTorch/SB3 -> ONNX -> TensorRT engine。

4. PyCUDA 不是当前优先项。当前 TensorRT runtime 已直接用 CUDA runtime API；单独安装 PyCUDA不会自动提升 Numpy/OpenCV/Python 代码性能。

5. 视觉前端仍会影响端到端 p95，但它不计入本轮部署工程成果。所有视觉前端优化应走独立 profile、golden image、action diff 和 shadow A/B。

6. active 乱跑是模型训练/策略质量限制，不纳入端侧部署验收。后续如需 active，只允许作为硬件接管链路 smoke 或安全阻断 smoke，不能作为模型避障效果验收。

## 7. 结论

本轮端侧部署成果可以概括为：

> V17 actor 已完成 Jetson 上的 ONNX/TensorRT FP16 部署链路，TensorRT 输出与 PyTorch 对齐；在 LiDAR sectorization、DataCollector 异步化和 telemetry cache 后，完整 shadow 链路的 V17 latency p50 从 168.570 ms 降到 87.153 ms，vehicle loop p95 从 696.175 ms 降到 130.800 ms，DataCollector p99 从 537.42 ms 降到 5.67 ms。优化后 PyTorch vs TensorRT A/B 进一步证明 actor residual p50/p95 约减半，但端到端 p95 仍受视觉前端和实时系统抖动影响。

因此，本报告把端侧部署成果、视觉前端工作、模型策略效果三者明确分离：部署成果已经可展示、可复盘；视觉前端作为独立后续方向继续评估；模型避障失败作为训练侧限制记录，不作为端侧部署成果的验收指标。
