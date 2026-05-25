# V17 ONNX/TensorRT Jetson 复盘与本机数据对比

日期：2026-05-18

目标：复盘已完成的 V17 ONNX/TensorRT 加速工作，并基于 Jetson 本机 `/home/jetson/mycar` 现有文档与日志判断当前性能优化效果。

## 1. Jetson 环境结论

已确认目标机器 `jetson@192.168.1.176` 可以使用 TensorRT：

| 项目 | 结果 |
|---|---|
| 主机 | `nano-4gb-jp451` |
| 系统 | Jetson L4T R32.5.2 / Ubuntu 18.04.6 |
| CUDA | 10.2 |
| TensorRT | 7.1.3.0，Python `tensorrt` 可 import |
| ONNX Runtime | venv 中未安装，不作为当前部署路径 |
| Python 环境 | `/home/jetson/env` |
| 注意事项 | import `torch/cv2` 时需要 `LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1` |

结论：这台 Jetson 支持 TensorRT 部署；ONNX 主要作为 PyTorch 到 TensorRT 的中间格式，不需要依赖 ONNX Runtime 才能完成当前加速链路。

## 2. 已完成任务复盘

### 2.1 训练代码与模型结构确认

已检查训练分支 `Gonglz/DonkeyCar-RL-Racing` 的 V17 相关代码，关键文件为：

- `src/ppo_multitrack_v17.py`
- `module/v17_policy.py`
- `module/v17_env.py`

V17 不是普通 CNN，而是 `RecurrentPPO + LiDARFiLMFeatureExtractor`。实际部署模型位于：

`/home/jetson/mycar/models/v17_postpass_hard_gate_final_model.zip`

已确认模型真实输入/状态形状：

| 输入/输出 | 形状 |
|---|---|
| `image` | `(1, 6, 128, 128)` |
| `state` | `(1, 7)` |
| `lidar` | `(1, 144)` |
| `lidar_meta` | `(1, 2)` |
| LSTM `h/c` | `(2, 1, 256)` |
| `action` | `(1, 3)` |

关键注意点：训练代码默认值容易误导，当前实际模型使用 `lidar_dim=144`，不是 72 或 36。

### 2.2 ONNX 导出

已编写并部署：

`/home/jetson/mycar/tools/export_v17_actor_onnx.py`

导出范围是 deterministic actor path：

`image/state/lidar/lidar_meta + h/c -> action + next_h/next_c`

没有导出 critic、SB3 训练封装和 optimizer。这样更适合部署，因为自动驾驶实时推理只需要 actor。

导出时做过两类 TensorRT 7 兼容修正：

- 用固定轴的 LayerNorm 替代 PyTorch 动态导出的 LayerNorm 表达。
- 避免 `chunk(dim=-1)` 产生 TensorRT 7 不兼容的负轴 Gather。

Jetson 上生成的文件：

| 文件 | 大小 | 说明 |
|---|---:|---|
| `/home/jetson/mycar/models/v17_actor.onnx` | 6.9M | ONNX actor graph |
| `/home/jetson/mycar/models/v17_actor_export.json` | 485B | 输入输出和形状元数据 |

`v17_actor_export.json` 记录：

```json
{
  "inputs": ["image", "state", "lidar", "lidar_meta", "h", "c"],
  "outputs": ["action", "next_h", "next_c"],
  "opset": 11,
  "shape": {
    "image_channels": 6,
    "obs_size": 128,
    "state_dim": 7,
    "lidar_dim": 144,
    "lidar_meta_dim": 2,
    "lstm_layers": 2,
    "lstm_hidden_size": 256
  }
}
```

### 2.3 TensorRT engine 构建

已通过 `/usr/src/tensorrt/bin/trtexec` 构建 FP16 engine：

`/home/jetson/mycar/models/v17_actor_fp16.engine`

文件大小：4.0M。

本次会话中 `trtexec --loadEngine` 验证结果：

| 指标 | 结果 |
|---|---:|
| GPU compute mean | 约 3.72 ms |
| Host latency mean | 约 3.98 ms |
| Throughput | 约 245.7 qps |

解释：这是模型本体的 TensorRT 推理耗时，不包含相机读取、图像预处理、LiDAR 组帧、DonkeyCar vehicle loop、日志写入等开销。

### 2.4 TensorRT Runtime 与 V17Pilot 集成

已部署：

- `/home/jetson/mycar/tools/v17_trt_runtime.py`
- `/home/jetson/mycar/tools/check_v17_trt_runtime.py`
- `/home/jetson/mycar/tools/compare_v17_torch_trt.py`

Runtime 没有安装 PyCUDA，而是用 `ctypes + libcudart` 调用 CUDA runtime API：

- `cudaMalloc`
- `cudaMemcpyAsync`
- stream 创建/同步
- TensorRT `execute_async_v2`

这避免了在 Jetson Nano 上额外编译 PyCUDA 的风险。

已修改：

`/home/jetson/mycar/v17_pilot.py`

新增可选参数：

- `--engine /home/jetson/mycar/models/v17_actor_fp16.engine`
- `--engine-metadata /home/jetson/mycar/models/v17_actor_export.json`

备份文件：

`/home/jetson/mycar/v17_pilot.py.bak_before_trt_runtime`

后续已补充修改：

`/home/jetson/mycar/runtime_monitor.py`

新增可选参数：

- `--shadow-engine`
- `--shadow-engine-metadata`

用途：让 V17 shadow/active runtime monitor 可以把 TensorRT engine 透传给 `V17Pilot`。当前仅用于 shadow A/B 验证；没有改变控制逻辑。

备份文件：

`/home/jetson/mycar/runtime_monitor.py.bak_before_trt_engine_args`

### 2.5 正确性验证

TensorRT runtime smoke：

```bash
PYTHONPATH=/home/jetson/mycar/tools python tools/check_v17_trt_runtime.py
```

结果：通过。输出 action 约为：

```text
[-0.029739 -0.249512 -0.026794]
```

PyTorch actor 与 TensorRT actor 对齐：

```bash
PYTHONPATH=/home/jetson/mycar/tools python tools/compare_v17_torch_trt.py --tolerance 0.02
```

结果：通过。

| 项目 | 数值 |
|---|---:|
| PyTorch action | `[-0.029122, -0.249469, -0.026623]` |
| TensorRT action | `[-0.029739, -0.249512, -0.026794]` |
| max abs diff | `0.000617` |
| tolerance | `0.02` |

结论：当前 TensorRT engine 与 PyTorch actor 输出足够一致，精度误差远低于部署阈值。

## 3. Jetson 本机已有文档/数据对比

### 3.1 `/home/jetson/mycar/README_2cam.md`

该文档是双 CSI 摄像头探索记录，不是当前 V17 主链路任务。

当前 V17 模型部署主链路是：

`单摄像头 + LiDAR + 自车状态传感器 -> V17 RecurrentPPO actor -> 控制输出`

因此 `README_2cam.md` 不作为 TensorRT/ONNX 性能判断依据。它最多说明 Jetson 摄像头链路可能存在固定 I/O 和预处理开销，但不能拿来代表 V17 的正式 runtime 延迟。

### 3.2 `/home/jetson/mycar/module/README.md`

该文档是模块职责说明，覆盖图像转换、环境、wrapper、V14 模块等。

与 TensorRT/ONNX 的关系：

- 没有记录 V17 ONNX/TensorRT 部署链路。
- 没有模型推理延迟对比。
- 可作为代码结构背景，不适合作为性能结论依据。

### 3.3 `/home/jetson/mycar/bench_logs*` 与 `/home/jetson/mycar/dynamics_data/hardware`

本机存在 18 个可解析的 `native_bench_*.json`。最近几条字段显示：

| 项目 | 典型值 |
|---|---|
| `profile` | `steering_sine` |
| `rate_hz` | `20.0` |
| `total_duration_sec` | `7.8` |
| `phases` | 3 |
| phase keys | `angle`, `duration_sec`, `group`, `name`, `throttle` |

判断：

- 这些是底盘/舵机/油门硬件 bench，不是 V17 模型推理 bench。
- 可以用来理解车辆控制回路和硬件测试节奏，但不能直接说明 TensorRT/ONNX 提升了多少推理性能。

### 3.4 `/home/jetson/mycar/monitor_logs`

这是当前最有价值的 Jetson 本机性能数据。已有 summary 记录了 V17 在单摄像头、LiDAR、自车传感器链路里的运行时延迟。历史 summary 都是 PyTorch 后端；本次已新增一组 60 秒 PyTorch/TensorRT shadow A/B 记录。

| Run | Backend | Mode | Frames | FPS mean | 推理 p50 | 推理 p95 |
|---|---|---|---:|---:|---:|---:|
| `v17_shadow_smoke` | PyTorch/SB3 RecurrentPPO | shadow | 313 | 11.110 | 73.819 ms | 89.274 ms |
| `v17_shadow_180s` | PyTorch/SB3 RecurrentPPO | shadow | 1881 | 10.866 | 76.908 ms | 90.877 ms |
| `v17_shadow_180s_fixed` | PyTorch/manual SB3 actor | shadow | 1059 | 10.899 | 68.735 ms | 80.616 ms |
| `v17_shadow_lidar_smoke` | PyTorch/manual SB3 actor | shadow | 150 | 4.206 | 169.618 ms | 206.020 ms |
| `v17_shadow_180s_lidar_fixed` | PyTorch/manual SB3 actor | shadow | 931 | 4.842 | 164.949 ms | 202.536 ms |
| `v17_active_obstacle_recovery_20260510_155148` | PyTorch/manual SB3 actor | active | 561 | 3.356 | 235.789 ms | 400.650 ms |

判断：

- 历史 monitor 数据显示 PyTorch V17 在真实 runtime monitor 链路中 p50 约 69-77ms；带 LiDAR 后 p50 约 165-170ms。
- active obstacle run 延迟更高，包含接管、控制、障碍恢复和现场运行因素，不适合直接和 shadow smoke 混比。
- 目前缺少同一套 `runtime_monitor.py + summarize_shadow_run.py` 生成的 TensorRT summary，这是下一步最应该补的实验。

## 4. 本次 TensorRT/ONNX 性能效果判断

### 4.1 ONNX 的作用

ONNX 本身不是加速器。它在当前链路中的作用是：

`PyTorch/SB3 actor -> ONNX -> TensorRT FP16 engine`

真正带来部署加速的是 TensorRT engine：

- 图优化
- kernel fusion
- FP16 执行
- 减少 PyTorch/SB3 Python 层路径
- 固定 batch/shape 后减少动态框架开销

### 4.2 模型本体效果

TensorRT engine 单独跑模型本体约 3.7-4.0ms。这说明 actor network 本身已经可以很快。

但目前没有同口径 PyTorch model-only loop benchmark JSON。因此严格说：

- 可以确认 TensorRT model-only 已经达到毫秒级。
- 还不能从本机归档数据中给出“PyTorch model-only vs TensorRT model-only”的精确倍数。

### 4.3 `v17_pilot.py --frames 300` smoke 效果

本次会话用 `v17_pilot.py --frames 300` 做过 PyTorch 与 TensorRT smoke 对照。该测试不启动完整 DonkeyCar runtime monitor，但会走 V17Pilot 的单帧预处理和 actor 推理路径。

| Backend | p50 | p95 | 备注 |
|---|---:|---:|---|
| PyTorch/manual SB3 actor | 40.246 ms | 45.486 ms | startup 会扭曲 avg |
| TensorRT FP16 actor | 34.346 ms | 44.982 ms | 同一 pilot smoke 路径 |

折算：

| 指标 | 改善 |
|---|---:|
| p50 延迟降低 | 14.66% |
| p95 延迟降低 | 1.11% |

解释：

- TensorRT 把 actor 本体压到了约 4ms。
- 但端到端 `v17_pilot.py` 还包含图像/状态/LiDAR 预处理，300 帧 smoke 中预处理约 22-23ms。
- 因此端到端提升被预处理吃掉一部分。

### 4.4 `runtime_monitor.py` 60 秒 shadow A/B 效果

本次已在 Jetson 上用正式 runtime monitor 主链路跑 60 秒 shadow A/B：

`单 CSI 摄像头 + RP2040 自车传感器 + LiDAR /scan + V17Pilot + DataCollector`

输出目录：

`/home/jetson/mycar/monitor_logs/v17_trt_benchmark_20260518_040300`

| 指标 | PyTorch shadow 60s | TensorRT shadow 60s | 变化 |
|---|---:|---:|---:|
| backend | PyTorch/manual SB3 actor | TensorRT FP16 actor | - |
| frames logged | 99 | 98 | - |
| inference p50 | 180.689 ms | 168.570 ms | 降 6.71% |
| inference p95 | 294.797 ms | 253.958 ms | 降 13.85% |
| effective FPS mean | 4.028 | 4.270 | 升 6.01% |
| CPU load mean | 51.224% | 46.525% | 降 9.17% |
| GPU load mean | 29.881% | 13.684% | 降 54.21% |
| power in mean | 3307.576 mW | 3093.224 mW | 降 6.48% |
| loop dt p50 | 189.300 ms | 192.050 ms | 基本持平 |
| loop dt p95 | 705.870 ms | 696.175 ms | 降 1.37% |
| LiDAR scan age p50 | 235.300 ms | 241.200 ms | 基本持平 |

解释：

- TensorRT 在正式 runtime shadow 里仍然有效，p50/p95 和资源占用都有改善。
- 改善幅度小于 model-only，是因为正式链路中 `V17Pilot` 的前处理、LiDAR scan age、DataCollector 和 vehicle loop 抖动占比很高。
- 两次 run 的 LiDAR scan age p50 都在约 235-241ms，说明 LiDAR/传感器链路是当前延迟观测里的重要因素。
- `loop_dt_ms_p50` 基本持平，说明 TensorRT 只优化 actor 后端，无法单独解决整车循环节拍。

### 4.5 与 Jetson 旧 monitor 数据怎么对比

严谨结论：

- 历史 Jetson monitor 数据证明 PyTorch runtime 链路偏慢，特别是带 LiDAR 时明显变慢。
- 本次 TensorRT 数据证明 engine 本体很快，且 smoke 端到端 p50 有 14.66% 改善。
- 新增 60 秒 shadow A/B 后，已经有同格式 TensorRT monitor summary；正式 runtime 里 TensorRT p50 降 6.71%、p95 降 13.85%、平均 FPS 升 6.01%。
- 不能把 `trtexec` 的 4ms 和旧 monitor 的 69-170ms 直接做倍数对比，因为二者口径不同：`trtexec` 是模型本体，monitor 是完整单摄像头/LiDAR/自车传感器 runtime 链路。

当前最可信的说法是：

> TensorRT 已经显著降低 V17 actor 本体推理开销；在 `v17_pilot.py` smoke 里 p50 降 14.66%，在正式 60 秒 shadow runtime 链路里 p50 降 6.71%、p95 降 13.85%、GPU load 降 54.21%。下一步优化重点应转向 V17 前处理、LiDAR scan age 和 runtime loop 抖动。

## 5. 主要风险和注意事项

1. TensorRT engine 与 Jetson 软件栈强绑定。
   当前 engine 面向 Jetson Nano / CUDA 10.2 / TensorRT 7.1.3。换 JetPack、TensorRT 版本或设备后建议重新 build。

2. LSTM 状态必须持续维护。
   `h/c -> next_h/next_c` 是 actor 的一部分。episode reset、人工接管或场景重置时要明确是否 reset hidden state。

3. `runtime_monitor.py` 还没有暴露 engine 参数。
   `v17_pilot.py` 已支持 `--engine`，但历史 monitor 入口目前还没有 `--shadow-engine`/`--active-engine` 参数。要得到同格式 TensorRT summary，需要补这个集成。

4. 预处理是下一阶段瓶颈。
   TensorRT 解决的是 actor network；相机、语义通道构建、LiDAR 特征、Python glue 仍然会限制端到端 FPS。

5. PyCUDA 不应作为第一优先级。
   当前 runtime 已经直接调用 CUDA runtime。单纯安装 PyCUDA通常不会自动提升性能，除非后续把预处理或自定义 kernel 搬到 GPU 上。

## 6. 建议的下一步

1. 扩展 60 秒 shadow A/B 到 180 秒。
   60 秒结果已经证明链路可跑；正式报告建议跑 180 秒，降低短 run 抖动影响。

2. 单独测预处理耗时。
   如果 TensorRT 后端 p50 仍高，优先优化图像/语义/LiDAR 预处理，而不是继续折腾 PyCUDA。

3. 分离 LiDAR scan age 与模型推理。
   当前 LiDAR scan age p50 约 235-241ms，需要确认是雷达频率、ROS bridge、特征构建还是日志采样造成的观测延迟。

4. 做 active 前继续跑 shadow。
   同一路况、同一模型、同一 duration，先跑 PyTorch shadow，再跑 TensorRT shadow，对比 p50/p95/FPS/CPU/GPU/温度。
