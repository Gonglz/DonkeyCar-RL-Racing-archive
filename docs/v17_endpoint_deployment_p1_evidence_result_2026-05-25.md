# V17 端侧部署 P1 证据补强结果

日期：2026-05-25
执行目录：`/home/jetson/mycar/monitor_logs/v17_p1_evidence_20260525_021003`

## 1. 结论

P1 补强完成三类证据：

- Async DataCollector 没有把阻塞简单转移成后台队列堆积。
- Shadow 模式 CSV 已显式证明 V17 output 只走 `shadow_only`，实际 actuator source 仍是 `user/manual`。
- TensorRT engine / metadata mismatch 已做 negative test，全部在 vehicle loop 启动前 fail-fast。

这组证据不改变 P0 端侧部署结论；它补强的是工程答辩中容易被追问的“异步是否藏问题”和“shadow 是否真的不接管”。

## 2. 代码与字段

新增或扩展：

| 文件 | 改动 |
|---|---|
| `Jetson/runtime_monitor.py` | `AsyncLogWriter` 记录 queue depth、dropped records、written records；DataCollector 记录 process RSS；CSV 增加 shadow actuator route 字段 |
| `Jetson/summarize_shadow_run.py` | 合并 CSV 与 `*_async_writer_stats.json` sidecar，输出 queue/RSS/non-takeover summary |
| `tools/aggregate_endpoint_validation.py` | aggregate report 增加 queue/drop、RSS、CSV non-takeover 指标 |
| `tools/check_v17_preflight_negative_cases.py` | 构造 metadata mismatch 副本并验证 preflight fail-fast |

新增 CSV 字段：

| 字段 | 含义 |
|---|---|
| `actual_actuator_source` | shadow 中为 `user/manual` |
| `v17_output_route` | shadow 中为 `shadow_only` |
| `shadow_action_angle/throttle` | V17 pilot 输出 |
| `vehicle_action_angle/throttle` | DonkeyCar 最终 actuator action |
| `shadow_non_takeover` | shadow 行级不接管判定 |
| `async_queue_depth` / `async_queue_max_depth` | Async writer queue 当前/历史最大深度 |
| `async_writer_backlog` / `async_writer_max_backlog` | writer backlog 当前/历史最大值 |
| `async_writer_dropped_records` | 累计丢弃日志样本 |
| `async_writer_records_written` | 已写 CSV 行数 |
| `async_writer_raw_records_written` | 已写 LiDAR raw JSONL 行数 |
| `process_rss_mb` | runtime process RSS |

每个 runtime CSV 旁边会生成 sidecar：

```text
run_<timestamp>_async_writer_stats.json
```

该 sidecar 记录 run 结束后的最终 queue/RSS/writer 计数，避免只依赖 CSV 某一帧的近似值。

## 3. 180s TensorRT Shadow 结果

运行目录：

`/home/jetson/mycar/monitor_logs/v17_p1_evidence_20260525_021003/trt_shadow_queue_metrics_180s`

运行口径：

| 项目 | 记录 |
|---|---|
| control mode | `shadow` |
| backend | TensorRT FP16 actor |
| duration | 180s |
| exit code | 0 |
| summary | `summary.json` |
| CSV rows | 301 |

Async queue / RSS：

| 指标 | 数值 |
|---|---:|
| `async_queue_depth_max` | 1 |
| `async_writer_backlog_final` | 0 |
| `async_writer_dropped_records` | 0 |
| `async_writer_records_written` | 301 |
| `async_writer_raw_records_written` | 301 |
| `process_rss_mb_start` | 2360.789 |
| `process_rss_mb_end` | 2312.117 |
| `process_rss_mb_max` | 2363.199 |

判断：

- queue 最大深度只有 1，run 结束 backlog 为 0。
- dropped records 为 0。
- RSS 没有增长趋势；end 低于 start，max 只比 start 高约 2.41MB。
- 这说明本轮 180s shadow 中，DataCollector 异步写入没有形成后台堆积。

Shadow non-takeover：

| 指标 | 数值 |
|---|---|
| `shadow_non_takeover_csv` | true |
| `shadow_non_takeover_rows` | 301 |
| `shadow_non_takeover_failures` | 0 |
| `actual_actuator_sources` | `['user/manual']` |
| `v17_output_routes` | `['shadow_only']` |

判断：CSV 已显式证明 301 行 shadow 样本中，V17 action 只作为 shadow output 记录，实际 actuator source 保持 `user/manual`。

延时背景：

| 指标 | 数值 |
|---|---:|
| `inference_latency_ms_p95` | 272.238 |
| `loop_dt_ms_p95` | 276.700 |
| `lidar_scan_age_ms_p95` | 349.000 |

这组延时只作为 P1 evidence run 的背景，不替代 final 20min 结论。

## 4. Engine / Metadata Mismatch Negative Tests

运行目录：

`/home/jetson/mycar/monitor_logs/v17_p1_evidence_20260525_021003/preflight_negative_cases`

所有 case 使用临时 `metadata_bad.json`，不修改真实 metadata。

| case | exit | entered vehicle loop | 预期错误 | pass |
|---|---:|---:|---|---:|
| `metadata_lidar_dim_72` | 2 | false | `shape mismatch for lidar_dim` | true |
| `metadata_lstm_hidden_128` | 2 | false | `binding shape mismatch` | true |
| `metadata_missing_lidar_meta` | 2 | false | `missing inputs` | true |
| `metadata_bad_next_h_output` | 2 | false | `missing outputs` | true |

实际错误摘要：

| case | error |
|---|---|
| `metadata_lidar_dim_72` | `TensorRT metadata shape mismatch for lidar_dim: expected 144, got 72` |
| `metadata_lstm_hidden_128` | `TensorRT binding shape mismatch for h: expected (2, 1, 128), got (2, 1, 256)` |
| `metadata_missing_lidar_meta` | `TensorRT metadata missing inputs ['lidar_meta']` |
| `metadata_bad_next_h_output` | `TensorRT metadata missing outputs ['next_h']` |

判断：preflight 不只覆盖 engine/metadata missing，也覆盖 metadata shape、binding shape、input/output name mismatch。

## 5. 验证命令

本地：

```bash
py -3 -m py_compile Jetson/runtime_monitor.py Jetson/summarize_shadow_run.py tools/aggregate_endpoint_validation.py tools/check_v17_preflight_negative_cases.py
bash -n tools/run_v17_20min_final_shadow.sh
```

Jetson：

```bash
python3 -m py_compile runtime_monitor.py tools/summarize_shadow_run.py tools/aggregate_endpoint_validation.py tools/check_v17_preflight_negative_cases.py
bash -n tools/run_v17_20min_final_shadow.sh
```

Jetson 实验：

```bash
python runtime_monitor.py drive \
  --model /home/jetson/mycar/models/v17_postpass_hard_gate_final_model.zip \
  --type v17 \
  --js \
  --control-mode shadow \
  --shadow-duration 180 \
  --log-dir /home/jetson/mycar/monitor_logs/v17_p1_evidence_20260525_021003/trt_shadow_queue_metrics_180s \
  --run-label trt_shadow_queue_metrics_180s \
  --track-condition p1_async_shadow_non_takeover \
  --shadow-engine /home/jetson/mycar/models/v17_actor_fp16.engine \
  --shadow-engine-metadata /home/jetson/mycar/models/v17_actor_export.json \
  --force-recording
```

```bash
python tools/check_v17_preflight_negative_cases.py \
  --runtime-monitor /home/jetson/mycar/runtime_monitor.py \
  --cwd /home/jetson/mycar \
  --model /home/jetson/mycar/models/v17_postpass_hard_gate_final_model.zip \
  --engine /home/jetson/mycar/models/v17_actor_fp16.engine \
  --metadata /home/jetson/mycar/models/v17_actor_export.json \
  --out-dir /home/jetson/mycar/monitor_logs/v17_p1_evidence_20260525_021003/preflight_negative_cases
```

## 6. 边界

- P1 不是新的性能优化主线，不声称 TensorRT 或模型效果因此提升。
- P1 不做 60min shadow。
- P1 不做真实 active 上路。
- P1 只补强端侧部署工程证据：queue 健康、shadow non-takeover、preflight mismatch fail-fast。
