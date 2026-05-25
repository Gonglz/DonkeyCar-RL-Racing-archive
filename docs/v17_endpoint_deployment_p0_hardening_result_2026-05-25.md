# V17 端侧部署 P0 证据补强结果

日期：2026-05-25
Jetson：`jetson@192.168.1.176`
实验目录：`/home/jetson/mycar/monitor_logs/v17_p0_hardening_20260525_011504`
manifest source commit：`168cd319149353f010c8d66679276fbf50e013db`

## 1. 结论

本轮补齐了 4 个面试/答辩容易被追问的证据：

- 连续 LSTM replay 级 PyTorch vs TensorRT diff：通过。
- final 20min latency waterfall：已生成，明确 actor residual 已很小，主要耗时在 preprocess / LiDAR age / loop。
- 运行中 LiDAR freeze/drop 故障注入：shadow 中可计数、不接管、不阻断；active safety 用 mock 验证安全输出。
- final 20min reproducibility manifest：已生成，包含 commit、branch、模型/ONNX/engine/metadata SHA256、环境版本和运行命令。

仍保持原验收边界：不跑真实 active 上路，不评价模型避障效果。

## 2. Replay Diff

输出目录：

`/home/jetson/mycar/monitor_logs/v17_p0_hardening_20260525_011504/replay_diff_1000`

测试方式：

- 固定 seed：`17017`
- samples：`1000`
- PyTorch manual actor 和 TensorRT actor 只在开头 reset。
- 连续 rollout，逐帧比较 action、next_h、next_c。
- PyTorch 参考使用 CPU float32；TensorRT 使用 FP16 engine。

| 指标 | 数值 |
|---|---:|
| samples | 1000 |
| action max abs diff | 0.004552633 |
| action mean abs diff | 0.000430575 |
| action p95 abs diff | 0.001630769 |
| next_h max abs diff | 0.005768955 |
| next_h p95 abs diff | 0.000851244 |
| next_c max abs diff | 0.066115379 |
| next_c p95 abs diff | 0.001974382 |
| final action abs diff | 0.000272512 |
| NaN/Inf count | 0 |
| tolerance | 0.02 |
| pass | true |

判断：

- action diff 远低于 0.02。
- hidden state p95 远低于 0.02。
- `next_c` 存在单点 max 0.066，但 p95 只有 0.00197，没有长期漂移或 action 放大。
- 这比单样本 smoke 更强，可以作为连续 LSTM 状态一致性的回归测试。

## 3. Final 20min Latency Waterfall

输出目录：

`/home/jetson/mycar/monitor_logs/v17_p0_hardening_20260525_011504/latency_waterfall_final_20min`

输入：

`/home/jetson/mycar/monitor_logs/v17_final_20min_shadow_20260525_000752/trt_shadow_20min_final`

| 模块 / 指标 | p50 ms | p95 ms | p99 ms | max ms |
|---|---:|---:|---:|---:|
| `pilot_preprocess_latency_ms` | 223.702 | 266.509 | 288.793 | 538.753 |
| `actor_residual_ms` | 13.131 | 20.911 | 32.147 | 38.004 |
| `pilot_inference_latency_ms` | 237.113 | 281.205 | 303.734 | 555.380 |
| `loop_dt_ms` | 242.100 | 286.900 | 309.448 | 559.100 |
| `lidar_scan_age_ms` | 274.600 | 351.220 | 454.944 | 606.800 |
| `V17Pilot` part | 223.090 | n/a | 297.130 | 555.530 |
| `DeploymentSafetyGate` part | 0.060 | n/a | 0.150 | 2.280 |
| `DataCollector` part | 0.040 | n/a | 9.870 | 23.090 |

当前没有单独 instrument 的字段：

- camera capture per-sample latency。
- semantic preprocess 内部分段。
- obs dict build。
- LiDAR receipt age / sectorization age。

判断：

- TensorRT actor residual p95 只有 20.911ms。
- final 20min V17 p95 281.205ms 的主因不是 TensorRT actor 本体，而是 preprocess、LiDAR age 和 loop jitter。
- DataCollector p99 9.870ms，仍低于 20ms 目标线。
- Safety gate part p99 0.150ms，开销可以忽略。

## 4. Runtime LiDAR Freeze / Drop 故障注入

工具：

`/home/jetson/mycar/tools/publish_lidar_fault_sequence.py`

测试方法：

- 使用唯一 topic，避免旧 latched scan 污染启动检查。
- publisher normal window 设置为 25s，用于覆盖 TensorRT preflight 和 vehicle loop 启动时间。
- runtime 参数包含 `--require-lidar --max-lidar-age-ms 350`。
- shadow 模式只计数，不阻断，不接管 actuator。

### 4.1 Freeze

目录：

`/home/jetson/mycar/monitor_logs/v17_p0_hardening_20260525_011504/runtime_lidar_freeze_shadow_35s_rerun`

| 指标 | 数值 |
|---|---:|
| exit | 0 |
| run duration | 34.26s |
| frames logged | 59 |
| safety_blocked | false |
| lidar_stale_count | 207 |
| lidar_missing_count | 0 |
| lidar_scan_age_ms p50 | 36100.0 |
| lidar_scan_age_ms p95 | 51586.73 |
| lidar_scan_age_ms max | 53143.7 |

关键日志：

```text
DeploymentSafetyGate: lidar_stale:18882.3ms>350.0ms
```

### 4.2 Drop

目录：

`/home/jetson/mycar/monitor_logs/v17_p0_hardening_20260525_011504/runtime_lidar_drop_shadow_35s_rerun2`

| 指标 | 数值 |
|---|---:|
| exit | 0 |
| run duration | 33.98s |
| frames logged | 61 |
| safety_blocked | false |
| lidar_stale_count | 235 |
| lidar_missing_count | 0 |
| lidar_scan_age_ms p50 | 48264.8 |
| lidar_scan_age_ms p95 | 63233.8 |
| lidar_scan_age_ms max | 65021.4 |

关键日志：

```text
DeploymentSafetyGate: lidar_stale:31039.4ms>350.0ms
```

说明：

- Drop 模式下 LiDAR bridge 保留最后一帧数据，因此 runtime 表现为 stale，而不是 missing。
- 这仍然覆盖运行中 LiDAR 失效：scan age 持续增长，runtime safety monitor 正常计数。
- shadow 中 `safety_blocked=false` 是预期结果；active 才阻断。

## 5. Active Safety Mock

输出目录：

`/home/jetson/mycar/monitor_logs/v17_p0_hardening_20260525_011504/active_safety_gate_mock`

该测试不启动车辆、不接管 actuator，只直接 mock `DeploymentSafetyGate` 的输入。

| case | safe angle | safe throttle | blocked | vehicle.on | reason | pass |
|---|---:|---:|---:|---:|---|---:|
| lidar_stale | 0.0 | 0.0 | true | false | `lidar_stale:1200.0ms>350.0ms` | true |
| inference_timeout | 0.0 | 0.0 | true | false | `inference_timeout:999.0ms>350.0ms` | true |
| rp2040_stale | 0.0 | 0.0 | true | false | `rp2040_stale:2500.3ms>1000.0ms` | true |

判断：active safety gate 的阻断动作已用 mock 验证：安全输出为 0，`vehicle.on=false`。

## 6. Reproducibility Manifest

final 20min run 已补：

`/home/jetson/mycar/monitor_logs/v17_final_20min_shadow_20260525_000752/trt_shadow_20min_final/repro_manifest.json`

同步副本：

`/home/jetson/mycar/monitor_logs/v17_p0_hardening_20260525_011504/manifest_final_20min_retrofit/repro_manifest.json`

记录内容：

| 项目 | 值 |
|---|---|
| branch | `codex/v17-endpoint-deployment` |
| commit | `168cd319149353f010c8d66679276fbf50e013db` |
| model SHA256 | `6dded04f69bcb827939e1a06b55b5f376faf4948162d96b3dcb28e0eb4b96a5d` |
| ONNX SHA256 | `d4680f9c433abb95987c62f256b1b4ba0eebfb9d3ef7b0f4fbe3f10447bbf9f3` |
| engine SHA256 | `6176a98be9757f8ecd7d98a58322b0f50e20f6191a5618d8c5c3df610aae1de6` |
| metadata SHA256 | `a4e04786ecc65ea93b4a464c53698242b178d1866e9d565a045fdfc756b52b86` |
| CUDA | 10.2.89 |
| TensorRT | 7.1.3.0 |
| PyTorch | 1.10.0 |
| OpenCV | 4.1.1 |
| DonkeyCar | 4.3.6.3 |
| Jetson L4T | R32.5.2 |

已额外校验：manifest 中 model、ONNX、engine、metadata 的 SHA256 与 Jetson 本机重新计算结果一致。

## 7. 注意事项

- runtime stale/drop 的第一次尝试使用了复用 topic `/fault_scan`，读到了旧 stale scan，startup gate 直接 fail-fast。正式有效结果使用唯一 topic 和非 latch publisher 重跑。
- replay diff 使用固定 seed 合法 shape/range obs，不声称覆盖真实视觉分布；它验证的是 actor 数值一致性和 LSTM 连续状态一致性。
- final 20min waterfall 只使用已有字段，不补造 camera/semantic 内部分段。

## 8. P1 Backlog 状态

- 60min shadow burn-in。
- Async queue depth / dropped sample summary：已在 `docs/v17_endpoint_deployment_p1_evidence_result_2026-05-25.md` 完成。
- Engine/metadata mismatch negative tests：已在 `docs/v17_endpoint_deployment_p1_evidence_result_2026-05-25.md` 完成。
- Shadow non-takeover CSV 显式字段：已在 `docs/v17_endpoint_deployment_p1_evidence_result_2026-05-25.md` 完成。
- LiDAR header age / receipt age / sectorization age 拆分。
