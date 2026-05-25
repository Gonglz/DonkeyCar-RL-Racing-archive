# V17 端侧部署最终冻结报告

冻结日期：2026-05-25
Jetson：`jetson@192.168.1.176`
最终 20min 复现实验目录：`/home/jetson/mycar/monitor_logs/v17_final_20min_shadow_20260525_000752/trt_shadow_20min_final`

## 1. 最终结论

从端侧部署角度，V17 已达到可交付状态。

交付口径：

> V17 模型已在 Jetson 上完成 ONNX/TensorRT actor 部署、runtime shadow 复现、安全 preflight/gate、日志与 summary 可追溯验证。该结论只评价端侧部署工程链路，不评价模型避障效果、active 闭环效果或赛道通过能力。

不能写成：

> V17 已具备实车避障能力。

## 2. 已完成工程能力

端侧部署链路已覆盖：

- ONNX/TensorRT FP16 actor 路径。
- TensorRT engine / metadata preflight fail-fast。
- TensorRT metadata shape 与 engine binding 检查。
- `--serial-port` 与 `cfg.RP2040_SERIAL_PORT` 统一，避免双串口来源。
- `require_lidar`、`require_rp2040`、`max_lidar_age_ms`、`max_rp2040_age_ms`、`max_inference_ms` gate。
- LiDAR disabled fail-fast。
- LiDAR stale fail-fast。
- RP2040 missing fail-fast。
- inference timeout 计数与 summary 字段。
- DataCollector 异步写入。
- telemetry cache。
- shadow non-takeover：shadow 模式只记录 V17 输出，不接管 actuator。
- 180s x3、10min x2、20min x1 TensorRT shadow 证据。
- P0 hardening 证据：1000-sample continuous LSTM replay diff、final 20min latency waterfall、runtime LiDAR freeze/drop 注入、active safety mock、reproducibility manifest。

## 3. 最终 20min TensorRT Shadow 复现实验

运行命令见：

`/home/jetson/mycar/monitor_logs/v17_final_20min_shadow_20260525_000752/trt_shadow_20min_final/command.txt`

运行上下文：

- control mode：`shadow`
- backend：TensorRT FP16 actor
- planned duration：1200s
- actual duration：1199.55s
- exit code：0
- model：`/home/jetson/mycar/models/v17_postpass_hard_gate_final_model.zip`
- engine：`/home/jetson/mycar/models/v17_actor_fp16.engine`
- metadata：`/home/jetson/mycar/models/v17_actor_export.json`
- CSV：`run_20260525_000813.csv`
- summary：`summary.json`

### 3.1 20min 关键指标

| 指标 | 数值 |
|---|---:|
| duration | 1199.55 s |
| frames logged | 1985 |
| effective FPS mean | 5.030 |
| V17 latency p50 | 237.113 ms |
| V17 latency p95 | 281.205 ms |
| V17 latency p99 | 303.734 ms |
| V17 latency max | 555.380 ms |
| loop dt p50 | 242.100 ms |
| loop dt p95 | 286.900 ms |
| loop dt p99 | 309.448 ms |
| loop dt max | 559.100 ms |
| LiDAR scan age p50 | 274.600 ms |
| LiDAR scan age p95 | 351.220 ms |
| LiDAR scan age p99 | 454.944 ms |
| LiDAR scan age max | 606.800 ms |
| CPU load mean | 77.810% |
| GPU load mean | 7.772% |
| power in mean | 4172.058 mW |
| PMIC mean/max | 100.0 / 100.0 C |

### 3.2 Safety counters

| 字段 | 数值 |
|---|---:|
| `safety_blocked` | false |
| `inference_timeout_count` | 0 |
| `lidar_missing_count` | 0 |
| `lidar_stale_count` | 0 |
| `rp2040_missing_count` | 0 |

说明：20min final shadow 未启用 `--max-lidar-age-ms`，因此 `lidar_stale_count=0` 是预期结果。LiDAR stale gate 已在独立实验中通过固定旧 timestamp 的 `/stale_scan` 验证。

### 3.3 Part profile

| Part | max | min | avg | p50 | p90 | p99 | p999 |
|---|---:|---:|---:|---:|---:|---:|---:|
| V17Pilot | 555.53 | 85.15 | 199.97 | 223.09 | 265.52 | 297.13 | 357.17 |
| DeploymentSafetyGate | 2.28 | 0.06 | 0.07 | 0.06 | 0.09 | 0.15 | 0.30 |
| DataCollector | 23.09 | 0.04 | 1.53 | 0.04 | 4.50 | 9.87 | 15.38 |

DataCollector p99 为 9.87ms，仍低于 20ms 目标线。

## 4. TensorRT 性能结论

优化后 PyTorch vs TensorRT A/B 已证明 TensorRT actor 本体收益：

| 指标 | PyTorch | TensorRT | 变化 |
|---|---:|---:|---:|
| actor residual p50 | 25.856 ms | 12.479 ms | -51.74% |
| actor residual p95 | 40.988 ms | 17.046 ms | -58.41% |
| full V17 latency mean | 229.556 ms | 199.060 ms | -13.28% |
| effective FPS mean | 4.199 | 4.867 | +15.91% |
| loop dt p95 | 277.315 ms | 265.920 ms | -4.11% |

判断：

- TensorRT 对 actor 本体加速明确。
- 端到端收益被视觉前处理、LiDAR scan age、camera/runtime jitter 稀释。
- 这不影响端侧部署成果成立，但限制了 full runtime p95 的改善幅度。

## 5. 安全阻断验证结论

已验证：

| 场景 | 结果 |
|---|---|
| engine missing | Vehicle loop 前 exit 2，写 `preflight_report.json` |
| metadata missing | Vehicle loop 前 exit 2，写 `preflight_report.json` |
| RP2040 missing + `require_rp2040` | Vehicle loop 前 exit 2，无 `/dev/ttyACM0` 旁路 |
| LiDAR disabled + `require_lidar` | Vehicle loop 前 exit 2 |
| LiDAR stale + `max_lidar_age_ms=350` | 收到 stale scan 后 Vehicle loop 前 exit 2 |
| inference timeout counter | shadow 中可计数，active 中可触发安全输出/停 loop |
| shadow non-takeover | 多轮 shadow 均保持 user/manual，不接管 actuator |

## 6. PMIC 100C 结论

PMIC 100C 不是 runtime monitor 映射错误：

```text
/sys/devices/virtual/thermal/thermal_zone4 type=PMIC-Die temp=100000
tegrastats: PMIC@100C
```

20min final shadow 中 CPU/GPU/AO/PLL/Fan 温度低且稳定，FPS 和 runtime 没有热失控形态。因此本报告将 PMIC 100C 记录为 Jetson 系统 PMIC sensor/driver 上报异常或板级读数风险，不能直接作为 V17 部署热失控证据。

## 7. 已知边界

- 不验证模型避障效果。
- 不验证 active closed-loop。
- 不使用 active 乱跑结果评价端侧部署。
- 20min final shadow 中出现 5 次 `[LiDAR ROS] AttributeError: 'NoneType' object has no attribute 'close'`，属于 rospy TCP close 日志异常；LiDAR CSV 数据持续有效，`lidar_missing_count=0`。后续可做日志降噪或桥接进程健壮性优化。
- 20min final shadow 的 LiDAR age p95 为 351.22ms，接近 active 默认 gate `350ms`。这说明 active gate 会偏保守；如果后续真的做 active smoke，应先决定是优化 LiDAR age，还是把 active 阈值调到更符合实测抖动的值。

## 8. 冻结状态

端侧部署工程成果冻结为以下交付物：

- 本报告：`docs/v17_endpoint_deployment_final_frozen_report_2026-05-25.md`
- 端侧部署完整记录：`docs/v17_endpoint_deployment_complete_record_2026-05-25.md`
- P0 safety gate 报告：`docs/v17_p0_safety_gate_implementation_result_2026-05-24.md`
- P0 证据补强报告：`docs/v17_endpoint_deployment_p0_hardening_result_2026-05-25.md`
- LiDAR stale / PMIC 核查报告：`docs/v17_lidar_stale_pmic_validation_2026-05-24.md`
- 部署验证执行结果：`docs/v17_endpoint_deployment_validation_result_2026-05-24.md`
- 视觉前端独立分析：`docs/v17_vision_frontend_separate_analysis_2026-05-24.md`

最终状态：

> V17 端侧部署工程链路完成并冻结。后续工作应进入训练侧模型效果改进、视觉前端独立优化，或更高强度长稳 shadow 复现；不应继续用 active 避障失败否定本轮端侧部署成果。
