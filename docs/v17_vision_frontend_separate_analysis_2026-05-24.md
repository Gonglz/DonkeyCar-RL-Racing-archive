# V17 视觉前端独立分析

日期：2026-05-24

## 1. 定位

本文档把 V17 视觉前端从端侧部署工程成果中单独拆出。原因是：当前已完成的部署成果主要集中在 ONNX/TensorRT、CUDA runtime、LiDAR sectorization、DataCollector 异步化、telemetry cache 和 runtime monitor；视觉语义前处理尚未被重写或系统优化，不能把它作为本轮部署优化成果。

视觉前端仍然重要。它解释了为什么 TensorRT actor 已经很快，但完整 runtime 的 p95 长尾仍没有完全降下来。

## 2. 当前视觉前端职责

V17 actor 的 `image` 输入是 `(1, 6, 128, 128)`，不是普通 RGB 单图。部署侧通过 `V17SemanticPreprocessor` 构建 6 通道 semantic image：

- 入口在 `/home/jetson/mycar/v17_pilot.py`。
- 本地副本中对应 `v17_pilot.remote.py` 的 `V17SemanticPreprocessor`。
- 默认优先使用 `CanonicalSemanticWrapper(domain=ws)`。
- `runtime_monitor.py` 默认 `--shadow-domain ws`。

当前链路可以概括为：

`cam/image_array -> V17SemanticPreprocessor -> CanonicalSemanticWrapper(domain=ws) -> 6x128x128 image tensor -> V17 actor`

训练侧区分 WS/GT/RRL domain；实车部署侧当前主链路使用 `ws` domain。视觉前端输出必须保持训练-部署语义契约，不应在没有 golden image / action diff 验证的情况下改通道含义。

## 3. 为什么不算本轮部署成果

本轮部署成果已经完成并验证的是：

- V17 actor ONNX 导出。
- TensorRT FP16 engine 构建。
- CUDA runtime 封装。
- TensorRT/PyTorch action 对齐。
- runtime monitor 中 TensorRT shadow A/B。
- LiDAR sectorization 前移。
- DataCollector 和 telemetry 去阻塞。

视觉前端当前只是被测量、被识别为残余瓶颈；还没有完成以下动作：

- 没有新增细粒度视觉阶段 profile 字段。
- 没有改写 `CanonicalSemanticWrapper` 或 `WS2NewTrack` 实现。
- 没有做 `copy/astype`、buffer 复用、OpenCV threading、C++/CUDA 小样的 A/B。
- 没有 golden image 回归和 action diff 回归。

所以，视觉前端应作为“独立后续优化方向”，而不是混入已完成部署成果。

## 4. 已观测延时数据

已有数据能说明视觉前端是剩余端到端长尾的重要因素，但不能证明它已经被优化。

### 4.1 早期 180s A/B 拆分

来自 `/home/jetson/mycar/monitor_logs/v17_trt_benchmark_180s_20260518_042116`：

| 指标 | PyTorch p50 | TensorRT p50 | 说明 |
| --- | ---: | ---: | --- |
| Total V17Pilot inference | 178.159 ms | 168.947 ms | runtime monitor 记录的总推理口径 |
| V17 preprocess | 153.070 ms | 155.082 ms | 主导项 |
| Actor residual | 23.589 ms | 12.612 ms | TensorRT 主要改善这里 |
| Loop dt | 183.000 ms | 177.500 ms | 完整 vehicle loop 周期 |
| LiDAR age | 235.300 ms | 234.900 ms | 不受 TensorRT 影响 |

同期 isolated microprofile：

| Component | p50 | p95 | 说明 |
| --- | ---: | ---: | --- |
| Official WS image preprocessor | 34.916 ms | 37.361 ms | `CanonicalSemanticWrapper(domain=ws)` |
| LiDAR feature build from 1147 ranges | 43.773 ms | 48.858 ms | 优化前 `_build_lidar_obs` |
| TensorRT actor only | 7.787 ms | 11.812 ms | `_predict_action` |
| Full preprocess without live sensor I/O | 79.328 ms | 84.940 ms | image + state + LiDAR + obs dict |

### 4.2 LiDAR/DataCollector 优化后 A/B

来自 `/home/jetson/mycar/monitor_logs/v17_optimized_backend_ab_20260518_060718`：

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

解释：

- TensorRT actor residual 已经明显下降。
- `preprocess` 在 p50/p95 中仍占主要比例。
- 视觉前端不是唯一 preprocess 成本；其中还包括 sensor snapshot、LiDAR obs/meta、obs dict 构建、数据 copy 和 runtime 竞争。因此下一步必须先做细粒度 profile，而不是直接重写视觉逻辑。

## 5. 当前不能直接用 `cv2.cuda`

Jetson 当前 OpenCV 结果：

| 检查项 | 结果 |
| --- | --- |
| OpenCV version | 4.1.1 |
| `cv2.cuda.getCudaEnabledDeviceCount()` | 0 |
| `has cv2.cuda.resize` | False |
| `has cv2.cuda.cvtColor` | False |
| `has cv2.cuda.GpuMat` | False |

因此当前环境不能简单把 resize、cvtColor、Sobel、morphology 等切到 `cv2.cuda`。这不是 CUDA 硬件不可用，而是当前 OpenCV Python build 没有启用对应 CUDA API。

## 6. 可行优化路线

### 6.1 先做分段 profile

在 `V17Pilot.run()` 和 `V17SemanticPreprocessor.__call__()` 周围记录细粒度耗时，建议字段：

- `image_preprocess_total_ms`
- `image_resize_ms`
- `image_color_convert_ms`
- `semantic_wrapper_ms`
- `semantic_channel_stack_ms`
- `sensor_snapshot_ms`
- `lidar_snapshot_ms`
- `lidar_feature_ms`
- `obs_dict_build_ms`
- `actor_ms`
- `adapter_safety_ms`

目的：明确 p95 长尾到底来自视觉前端、LiDAR/sensor snapshot、TensorRT I/O，还是 Python runtime 竞争。

### 6.2 CPU 实现层优化

在不改变 6 通道语义输出的前提下，优先评估：

- 减少重复 `astype(np.float32)`。
- 减少 HWC/CHW 反复 copy。
- 预分配输出 buffer。
- 缓存固定 kernel、floor map、函数引用。
- 共享 BGR/RGB/HSV 转换结果。
- 避免每帧 import 或重复构造 wrapper。
- 检查 OpenCV threading 设置对 Jetson Nano CPU 竞争的影响。

这些改动必须满足：

- golden image 输出差异在阈值内。
- PyTorch/TensorRT action diff 不超过已接受阈值。
- shadow A/B 不恶化 p95。

### 6.3 OpenCV threading sweep

建议只作为实验，不直接改默认配置：

- `cv2.setNumThreads(1)`
- `cv2.setNumThreads(2)`
- `cv2.setNumThreads(4)` 或默认值

观测指标：

- image preprocess p50/p95/p99。
- total V17 latency p50/p95。
- loop dt p50/p95。
- CPU load。
- TensorRT actor residual。

Jetson Nano 上 OpenCV 多线程可能和 DonkeyCar loop、ROS helper、TensorRT/PyTorch 竞争 CPU；线程数不是越高越好。

### 6.4 GPU 可行性小样

GPU 化不是当前第一优先级，但可以在 profile 指向明确慢点之后做小样：

1. 自编 OpenCV CUDA build。
   - 风险：Jetson Nano 编译和 ABI 环境复杂。
   - 收益：可用 `cv2.cuda` API。

2. GStreamer/NVMM 前处理。
   - 适合 resize、颜色转换等相机前处理。
   - 不适合直接覆盖复杂 semantic wrapper。

3. TensorRT plugin 或 custom CUDA kernel。
   - 只适合稳定、纯 pixelwise 或局部窗口操作。
   - 当前视觉语义含 morphology、形状过滤、时序逻辑，不应整体搬 GPU。

4. C++ OpenCV 扩展。
   - 可能比 Python/Numpy 循环更现实。
   - 仍需 golden image/action diff。

## 7. 独立验收标准

视觉前端优化不能只看速度，必须同时满足：

- golden image 回归：同一批真实 Jetson 帧输入，新旧 6 通道输出差异可解释。
- action diff 回归：同一 obs 下 PyTorch/TensorRT action 差异不超过阈值。
- shadow A/B：至少 180s，记录 p50/p95/p99、FPS、loop dt、CPU/GPU、功耗。
- 不改变 `domain=ws` 的部署默认语义。
- 不把视觉优化结果和 TensorRT actor 优化结果混在一个口径里。

## 8. 结论

视觉前端现在是 V17 端到端延时的重要残余瓶颈，但不是本轮已经完成的部署工程成果。下一步应先补分段 profile 和回归基线，再做低风险 CPU 实现层优化；只有 profile 证明某个固定阶段稳定占大头，才值得做 GPU/CUDA/C++ 小样。
