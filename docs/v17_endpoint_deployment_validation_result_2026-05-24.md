# V17 端侧部署稳定性与安全阻断执行结果

日期：2026-05-24
执行主机：Jetson `jetson@192.168.1.176`
执行目录：`/home/jetson/mycar/monitor_logs/v17_endpoint_deploy_validation_20260524_193325`

最终冻结版报告见：

`docs/v17_endpoint_deployment_final_frozen_report_2026-05-25.md`

P0 safety gate 后续实现与验证结果见：

`docs/v17_p0_safety_gate_implementation_result_2026-05-24.md`

## 1. 口径

本次执行只验证端侧部署链路：稳定性、低延时趋势、shadow 不接管、日志可追溯、故障可观测和安全阻断证据。
不验证模型是否会避障，不验证 active 跑圈，不用实车乱跑结果评价 ONNX/TensorRT 部署质量。

使用环境：

- 项目目录：`/home/jetson/mycar`
- Python 环境：`. /home/jetson/env/bin/activate`
- Runtime：`LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 python runtime_monitor.py drive`
- 模型：`/home/jetson/mycar/models/v17_postpass_hard_gate_final_model.zip`
- TensorRT engine：`/home/jetson/mycar/models/v17_actor_fp16.engine`
- TensorRT metadata：`/home/jetson/mycar/models/v17_actor_export.json`

本次执行未修改 runtime 代码，全部实验为 shadow 或故障注入 smoke。

## 2. 执行矩阵

| 实验 | 类别 | 后端 | 时长 | exit | CSV | summary | shadow 不接管 |
|---|---|---:|---:|---:|---:|---|---|
| `trt_shadow_180s_run1` | stability_180s | TensorRT | 180s | 0 | 1 | True | True |
| `trt_shadow_180s_run2` | stability_180s | TensorRT | 180s | 0 | 1 | True | True |
| `trt_shadow_180s_run3` | stability_180s | TensorRT | 180s | 0 | 1 | True | True |
| `trt_shadow_10min_run1` | stability_10min | TensorRT | 600s | 0 | 1 | True | True |
| `pytorch_shadow_ab_180s` | backend_ab | PyTorch | 180s | 0 | 1 | True | True |
| `trt_shadow_ab_180s` | backend_ab | TensorRT | 180s | 0 | 1 | True | True |
| `fault_engine_missing` | fault_injection | mixed | 5s | 124 | 0 | False | False |
| `fault_metadata_missing` | fault_injection | mixed | 5s | 124 | 0 | False | False |
| `fault_lidar_disabled_shadow` | fault_injection | mixed | 30s | 0 | 1 | True | True |
| `fault_sensor_missing_shadow` | fault_injection | mixed | 30s | 0 | 1 | True | True |

`fault_engine_missing` 和 `fault_metadata_missing` 的 exit 124 是外层 `timeout` 清理结果，不是 runtime 干净退出。这是本次最重要的安全工程缺口之一。

## 3. 稳定性结果

TensorRT shadow 180s x3 和 10min x1 均正常结束，均生成 CSV、summary 和 runtime log。所有 shadow run 日志都有 `V17 shadow pilot injected; actuator path remains user/manual`，说明 V17 pilot 注入后没有接管 actuator。

| run | V17 latency p50/p95/p99 ms | loop p95 ms | LiDAR age p95 ms | DataCollector p99 ms | FPS mean |
|---|---:|---:|---:|---:|---:|
| `trt_shadow_180s_run1` | 194.674 / 238.282 / 260.407 | 242.900 | 327.560 | 7.410 | 5.237 |
| `trt_shadow_180s_run2` | 217.706 / 267.284 / 294.433 | 271.920 | 330.520 | 8.250 | 5.101 |
| `trt_shadow_180s_run3` | 213.478 / 254.785 / 274.144 | 260.845 | 329.015 | 7.510 | 4.439 |
| `trt_shadow_10min_run1` | 218.625 / 257.805 / 274.420 | 262.750 | 328.610 | 8.690 | 4.358 |

判断：

- shadow 稳定性通过：3 轮 180s 和 1 轮 10min 均无崩溃。
- DataCollector 异步化效果保持稳定：p99 为 7.41 到 8.69 ms，低于 20 ms 通过线。
- LiDAR scan age p95 约 325 到 331 ms，仍是需要持续监控的实时性指标，但本轮没有造成 runtime 崩溃。
- 10min 运行中 CPU/GPU 温度低，但 PMIC 日志长期显示 100C，应作为 Jetson 电源/传感器读数风险继续记录，不能简单忽略。

## 4. PyTorch vs TensorRT A/B

本轮 A/B 使用完整 runtime shadow，不是 model-only microbench。指标口径为完整 V17 pilot latency、preprocess latency、actor residual、vehicle loop 和系统资源。

| 指标 | 统计 | PyTorch | TensorRT | 变化 |
|---|---|---:|---:|---:|
| `pilot_inference_latency_ms` | p50 | 233.468 | 214.006 | -8.34% |
| `pilot_inference_latency_ms` | p95 | 270.476 | 259.794 | -3.95% |
| `pilot_inference_latency_ms` | mean | 229.556 | 199.060 | -13.28% |
| `actor_residual_ms` | p50 | 25.856 | 12.479 | -51.74% |
| `actor_residual_ms` | p95 | 40.988 | 17.046 | -58.41% |
| `actor_residual_ms` | mean | 27.921 | 12.420 | -55.52% |
| `loop_dt_ms` | p50 | 238.750 | 218.800 | -8.36% |
| `loop_dt_ms` | p95 | 277.315 | 265.920 | -4.11% |
| `effective_fps` | mean | 4.199 | 4.867 | +15.91% |
| `gpu_load_pct` | mean | 8.365 | 7.093 | -15.21% |
| `power_in_mw` | mean | 3594.317 | 3584.308 | -0.28% |

判断：

- TensorRT 对 actor 计算本体有效：`actor_residual_ms` p50/p95 约下降 52% 到 58%。
- 完整 runtime 也有收益，但幅度较小：V17 latency mean 下降 13.28%，FPS mean 提升 15.91%，loop p95 只下降 4.11%。
- p95 未出现数量级改善，原因仍是 actor 只占完整 V17 pipeline 的一部分，前处理、camera、LiDAR age、vehicle loop jitter 会限制端到端收益。

## 5. 故障注入结果

### 5.1 Engine / metadata 缺失

结果：

- `fault_engine_missing` 明确抛出 `FileNotFoundError: /home/jetson/mycar/models/NO_SUCH_ENGINE.engine`。
- `fault_metadata_missing` 明确抛出 `FileNotFoundError: /home/jetson/mycar/models/NO_SUCH_METADATA.json`。
- 两个 run 均未生成 CSV/summary，说明没有进入可用 shadow 记录阶段。

缺口：

- runtime 没有干净退出，外层 `timeout` 最终以 exit 124 清理。
- 这只能算“故障可观测”，不能算“安全阻断实现完整”。

后续必须补：

- runtime 启动前做 engine/metadata preflight。
- 缺失或 shape mismatch 时直接 fail-fast，返回明确非 0 exit code。
- active 模式下必须在车辆 loop 启动前阻断，不能靠异常堆栈和 timeout。

### 5.2 LiDAR disabled

结果：

- `fault_lidar_disabled_shadow` 正常结束，生成 CSV/summary。
- runtime log 显示 `LiDAR: 关闭` 和 `LiDAR ROS: 未连接`。
- summary 中 LiDAR points、nearest、scan age 为 null。
- shadow 不接管日志存在。

判断：

- LiDAR 不可用状态已经可观测。
- 这不等于 active 安全阻断已完成。当前证据只说明 shadow 记录能跑完，active gate 仍需显式 `require_lidar` 或 stale-age 阻断逻辑。

### 5.3 Sensor missing

结果：

- `fault_sensor_missing_shadow` 启动参数使用 `/dev/NO_SUCH_RP2040`，日志先显示 RP2040 串口打开失败。
- 同一个 run 后续又出现 `[RP2040] connected to /dev/ttyACM0`，说明 DonkeyCar vehicle loop 内部仍按 `myconfig.py` 默认串口连接了 RP2040。

判断：

- 这个实验暴露出配置路径不一致：`runtime_monitor.py --serial-port` 影响 monitor 注入的串口 reader，但 `manage.py` 的 `RP2040SensorPart` 仍使用 `cfg.RP2040_SERIAL_PORT`。
- 因此本轮不能把 sensor missing 算作完整通过，只能算“发现了 sensor fault injection 设计缺口”。

后续必须补：

- 增加统一的 sensor source 配置，避免 monitor 和 vehicle loop 各自连接串口。
- active 前增加 `require_rp2040` 或关键状态 freshness gate。
- 用临时 myconfig 或物理断开重跑 sensor missing，确认 vehicle loop 不会绕过故障注入。

## 6. 验收结论

| 维度 | 结论 | 依据 |
|---|---|---|
| TensorRT shadow 稳定性 | 通过 | 180s x3 和 10min x1 均 exit 0，均有 CSV/summary，均为 shadow non-takeover |
| DataCollector 不阻塞主 loop | 通过 | DataCollector p99 7.41 到 8.69 ms，低于 20 ms 通过线 |
| TensorRT 性能收益 | 通过，但端到端收益有限 | actor residual p95 下降 58.41%，full runtime FPS mean 提升 15.91%，loop p95 下降 4.11% |
| shadow 安全口径 | 通过 | 所有有效 shadow run 均保持 user/manual，不接管 actuator |
| engine/metadata 缺失安全处理 | 部分通过 | FileNotFoundError 可观测，但需要 timeout 清理，缺少干净 fail-fast |
| LiDAR 缺失处理 | 部分通过 | shadow 可记录 LiDAR missing/null，但 active blocking 未验证 |
| sensor 缺失处理 | 未通过完整阻断验证 | fault run 被默认 `/dev/ttyACM0` 连接路径绕过，需要重新设计注入方式 |
| 模型避障效果 | 不评价 | active 乱跑属于训练/策略质量问题，不纳入端侧部署验收 |

最终口径：

> V17 已能在 Jetson 上以 ONNX/TensorRT actor 路径完成稳定 shadow 部署验证，并具备较完整的日志和复现证据。TensorRT 对 actor 本体加速明显，但端到端收益被前处理、LiDAR age 和 vehicle loop jitter 稀释。安全阻断方面，engine/metadata 和 LiDAR/sensor 故障已经可观测，但 active 前的干净 fail-fast、require-lidar、require-rp2040、inference-timeout gate 仍需补齐后才能宣称安全阻断完整。

## 7. 原始证据位置

- 汇总报告：`/home/jetson/mycar/monitor_logs/v17_endpoint_deploy_validation_20260524_193325/aggregate_report.md`
- 汇总 JSON：`/home/jetson/mycar/monitor_logs/v17_endpoint_deploy_validation_20260524_193325/aggregate_metrics.json`
- 执行报告：`/home/jetson/mycar/monitor_logs/v17_endpoint_deploy_validation_20260524_193325/execution_report.md`
- 执行上下文：`/home/jetson/mycar/monitor_logs/v17_endpoint_deploy_validation_20260524_193325/run_context.txt`
- runner：`/home/jetson/mycar/monitor_logs/v17_endpoint_deploy_validation_20260524_193325/run_endpoint_validation.sh`
- 每个 run 的 `command.txt`、`runtime.log`、`summary.json` 位于各自子目录。

## 8. 后续状态

2026-05-24 已完成 P0 safety gate 实现和 P1 主要复验。结果见：

`docs/v17_p0_safety_gate_implementation_result_2026-05-24.md`

2026-05-24 已追加完成 LiDAR stale gate 与 PMIC thermal zone 核查。结果见：

`docs/v17_lidar_stale_pmic_validation_2026-05-24.md`

仍未完成的后续项：

- 如要进一步提高证据强度，可做 20min 或多轮 10min TensorRT shadow 复现实验。
- 如要把 PMIC 热风险完全闭环，需要板级/外部温度测量或 JetPack/内核 PMIC driver 核查。
