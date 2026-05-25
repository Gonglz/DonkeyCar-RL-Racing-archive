# V17 P0 Safety Gate Implementation Result

日期：2026-05-24
Jetson：`jetson@192.168.1.176`
验证目录：`/home/jetson/mycar/monitor_logs/v17_p0_safety_gate_validation_20260524_2305`

最终冻结版报告见：

`docs/v17_endpoint_deployment_final_frozen_report_2026-05-25.md`

LiDAR stale gate 与 PMIC thermal zone 追加验证见：

`docs/v17_lidar_stale_pmic_validation_2026-05-24.md`

## 1. 本轮实现范围

本轮只做端侧部署安全工程和可复现工具，不验证模型避障效果，不跑 active 接管。

已改 Jetson 文件：

- `/home/jetson/mycar/runtime_monitor.py`
- `/home/jetson/mycar/tools/summarize_shadow_run.py`
- `/home/jetson/mycar/tools/aggregate_endpoint_validation.py`
- `/home/jetson/mycar/tools/run_v17_10min_post_gate.sh`

远端备份：

- `/home/jetson/mycar/runtime_monitor.py.bak_20260524_230448`
- `/home/jetson/mycar/tools/summarize_shadow_run.py.bak_20260524_230448`

## 2. P0 变更

### 2.1 Engine / metadata preflight

`runtime_monitor.py` 在 `drive()` 调用前新增部署 preflight：

- 检查 V17 model 文件存在。
- 检查 TensorRT engine 文件存在。
- 检查 TensorRT metadata 文件存在并可解析。
- 检查 metadata inputs/outputs 完整。
- 检查 metadata shape：`image_channels=6`、`obs_size=128`、`state_dim=7`、`lidar_dim=144`、`lidar_meta_dim=2`、LSTM shape 有效。
- 反序列化 TensorRT engine，检查 binding 名称和 shape 与 metadata 一致。
- 将结果写入 `preflight_report.json`。
- 失败时在 Vehicle loop 启动前 exit 2。

这解决了上一轮 engine/metadata 缺失需要外层 timeout 清理的问题。

### 2.2 Startup safety gate

新增 CLI：

- `--require-lidar` / `--no-require-lidar`
- `--require-rp2040` / `--no-require-rp2040`
- `--max-lidar-age-ms`
- `--max-rp2040-age-ms`
- `--max-inference-ms`

默认策略：

- `active` 默认 `require_lidar=True`、`require_rp2040=True`、`max_lidar_age_ms=350`、`max_rp2040_age_ms=1000`、`max_inference_ms=350`。
- `shadow` 默认不强制 require，避免影响故障注入和纯观测实验。

startup gate 在 `install_monitor()` 阶段执行，早于 `manage.drive()` 和 Vehicle loop：

- `require_lidar` 且 `--disable-lidar`：直接 fail-fast。
- `require_rp2040` 且串口不可用/无帧：直接 fail-fast。
- 失败写入 `preflight_report.json`，exit 2，不打印 traceback。

### 2.3 Runtime safety monitor

新增 `DeploymentSafetyGate` Part：

- 监控 `pilot/inference_latency_ms`。
- 监控 LiDAR missing/stale。
- 监控 RP2040 missing/stale。
- 在 shadow 中只计数、不接管、不阻断。
- 在 active 中如触发 violation，则输出安全角度/油门 0，并停止 vehicle loop。

新增 CSV/summary 字段：

- `safety_blocked`
- `safety_block_reason`
- `safety_inference_timeout_count`
- `safety_lidar_missing_count`
- `safety_lidar_stale_count`
- `safety_rp2040_missing_count`
- `safety_last_lidar_age_ms`
- `safety_last_rp2040_age_ms`

### 2.4 串口配置统一

`runtime_monitor.py --serial-port` 现在会同步覆盖：

`cfg.RP2040_SERIAL_PORT = args.serial_port`

因此 `manage.py` 内部的 `RP2040SensorPart` 不再绕过 CLI 指定的串口路径。上一轮 sensor missing 中 `/dev/NO_SUCH_RP2040` 被默认 `/dev/ttyACM0` 绕过的问题已被修掉。

## 3. P1/P2 工具变更

### 3.1 Summary 增强

`tools/summarize_shadow_run.py` 新增：

- `inference_latency_ms_p99`
- `inference_latency_ms_max`
- `loop_dt_ms_p99`
- `loop_dt_ms_max`
- `lidar_scan_age_ms_p99`
- `lidar_scan_age_ms_max`
- `pmic_temp_mean`
- `pmic_temp_max`
- safety gate counters

### 3.2 Aggregate 脚本固定到 tools

新增：

`/home/jetson/mycar/tools/aggregate_endpoint_validation.py`

已验证它能重新生成上一轮完整矩阵：

- `/home/jetson/mycar/monitor_logs/v17_endpoint_deploy_validation_20260524_193325/aggregate_metrics.json`
- `/home/jetson/mycar/monitor_logs/v17_endpoint_deploy_validation_20260524_193325/aggregate_report.md`

## 4. 验证矩阵

| run | 目的 | exit | 结果 |
|---|---|---:|---|
| `fault_engine_missing_preflight` | engine 缺失 fail-fast | 2 | 通过，Vehicle loop 前失败 |
| `fault_metadata_missing_preflight` | metadata 缺失 fail-fast | 2 | 通过，Vehicle loop 前失败 |
| `fault_sensor_missing_require_rp2040_clean` | RP2040 缺失 startup gate | 2 | 通过，无 traceback，无 `/dev/ttyACM0` 旁路 |
| `fault_lidar_disabled_require_lidar` | LiDAR disabled + require gate | 2 | 通过，Vehicle loop 前失败 |
| `trt_shadow_30s_post_gate` | 新 runtime 正常 shadow | 0 | 通过，CSV/summary 生成，shadow 不接管 |
| `trt_shadow_timeout_counter_20s` | inference timeout 计数 | 0 | 通过，`inference_timeout_count=104`，shadow 不阻断 |
| `trt_shadow_10min_post_gate_run2` | 第二轮 10min 稳定性/PMIC 复核 | 0 | 通过，599.35s，无 safety block |

## 5. 关键数据

### 5.1 30s 正常 shadow

`trt_shadow_30s_post_gate`：

- exit：0
- shadow non-takeover：通过
- `inference_latency_ms_p95`：267.221 ms
- `inference_timeout_count`：0
- `safety_blocked`：false
- `pmic_temp_mean/max`：100.0 / 100.0

### 5.2 Timeout counter shadow

`trt_shadow_timeout_counter_20s` 使用 `--max-inference-ms 1` 做人工低阈值测试：

- exit：0
- `inference_timeout_count`：104
- `safety_blocked`：false
- `safety_block_reason_last`：`inference_timeout:97.0ms>1.0ms`

判断：shadow 只记录 timeout，不接管、不停 loop；active 才会触发安全输出和停 loop。

### 5.3 10min TensorRT shadow 第二轮

`trt_shadow_10min_post_gate_run2`：

- exit：0
- duration：599.35 s
- frames logged：973
- effective FPS mean：5.036
- V17 latency p50/p95/p99/max：209.379 / 249.673 / 270.420 / 314.397 ms
- loop dt p95/p99/max：254.840 / 276.972 / 648.700 ms
- LiDAR scan age p95/p99/max：325.820 / 335.624 / 497.100 ms
- `inference_timeout_count`：0
- `lidar_missing_count`：0
- `rp2040_missing_count`：0
- `safety_blocked`：false
- PMIC mean/max：100.0 / 100.0

PMIC 判断：第二轮 10min 中 PMIC 仍固定 100C，但 CPU/GPU/AO/PLL/Fan 温度低且稳定，FPS 和 runtime 没有热失控形态。因此当前更像 PMIC telemetry 读取/映射异常，不能作为真实热失控证据；后续应单独核查 Jetson thermal zone 映射。

## 6. 结论

P0 已完成：

- engine/metadata 缺失不再拖到 Vehicle loop 内才报错。
- active 默认 safety gate 已接入。
- shadow 可显式打开 require gate 做故障注入。
- sensor missing 不再被 `manage.py` 默认 `/dev/ttyACM0` 路径绕过。
- inference timeout、LiDAR/RP2040 missing/stale 已有 CSV 和 summary 计数字段。

P1 已完成主要验证：

- sensor missing 已重跑并通过干净 fail-fast。
- inference timeout 计数已验证。
- 10min TensorRT shadow 第二轮已完成，PMIC 固定 100C 的现象被复现。

P2 已补齐关键验证：

- aggregate report 生成脚本已固定到 `tools/` 并验证可运行。
- LiDAR stale 注入已单独做：`/stale_scan` 使用固定旧 timestamp，`require_lidar + max_lidar_age_ms=350` 在 Vehicle loop 前 exit 2。
- PMIC=100C 已核查：sysfs `PMIC-Die temp=100000` 且 `tegrastats` 也显示 `PMIC@100C`，不是 runtime monitor 映射错误；当前记录为 Jetson PMIC sensor/driver 上报异常或板级读数风险。
