# V17 端侧部署稳定性与安全阻断实验计划

日期：2026-05-24
口径更新：2026-05-24

执行状态：2026-05-24 已执行一轮 Jetson shadow 稳定性与故障注入验证。结果见：

`docs/v17_endpoint_deployment_validation_result_2026-05-24.md`

## 1. 实验原则

本计划只验证端侧部署链路，不验证模型避障效果。

已知实车 active 会乱跑/乱爬，原因归为模型训练和策略质量不足。后续不再做 obstacle recovery 成功率、active 跑圈、避障通过率等验收。端侧部署只回答这些问题：

- TensorRT/ONNX runtime 能否稳定运行。
- shadow 是否不接管车辆。
- 日志、summary、part profile 是否可复现。
- LiDAR、DataCollector、telemetry 是否不阻塞主 loop。
- 关键故障是否能被记录并阻断 active 或触发安全输出。

视觉前端优化也不放在本计划内；视觉细节见：

`docs/v17_vision_frontend_separate_analysis_2026-05-24.md`

## 2. 稳定性复现实验矩阵

### 实验 A：TensorRT shadow 180s x3

目的：验证优化后的 TensorRT 部署链路在短中时长内稳定、可复现。

建议口径：

- control mode：`shadow`
- backend：TensorRT FP16 actor
- duration：180s
- repetitions：3
- actuator：manual/user，不由 V17 接管
- model：`/home/jetson/mycar/models/v17_postpass_hard_gate_final_model.zip`
- engine：`/home/jetson/mycar/models/v17_actor_fp16.engine`
- metadata：`/home/jetson/mycar/models/v17_actor_export.json`

每轮必须记录：

- 命令。
- 开始/结束时间。
- 模型路径。
- engine 路径。
- log dir。
- summary JSON。
- run notes。

通过标准：

- 3 轮都无崩溃。
- 每轮都有 CSV、summary JSON、part profile 或等价 profile 输出。
- V17 latency、loop dt、DataCollector p99、LiDAR age、CPU/GPU、power 均可追溯。
- DataCollector p99 <= 20ms。
- p95 若受视觉或系统抖动影响，只记录原因，不强行归咎 TensorRT。

### 实验 B：TensorRT shadow 10min x1

目的：验证长时间运行中的温度、功耗、内存、loop jitter 和日志线程稳定性。

建议口径：

- control mode：`shadow`
- backend：TensorRT FP16 actor
- duration：600s
- actuator：manual/user
- 强制记录：开启
- LiDAR：开启

主要指标：

- V17 latency p50/p95/p99。
- actor residual p50/p95/p99。
- loop dt p50/p95/p99。
- DataCollector p99/max。
- LiDAR scan age p50/p95/p99。
- CPU/GPU load。
- power in mean/max。
- 温度曲线。
- dropped log sample 计数，如果已有该字段。

通过标准：

- 10min run 无崩溃。
- 日志线程和 telemetry 线程没有拖死主 loop。
- DataCollector p99 <= 20ms；若超过，必须能定位到日志、telemetry、raw LiDAR 或磁盘写入。
- power/temperature 不出现持续不可控上升。

### 实验 C：优化后 PyTorch vs TensorRT A/B

目的：复现 TensorRT actor 后端收益，不把模型效果混入性能结论。

建议口径：

- control mode：`shadow`
- PyTorch 后端：不传 `--shadow-engine`
- TensorRT 后端：传 `--shadow-engine` 和 `--shadow-engine-metadata`
- duration：180s x1 或 60s x3
- 顺序：PyTorch、TensorRT 交替，避免温度和电源状态偏置。

主要指标：

- `actor_residual = pilot_inference_latency_ms - pilot_preprocess_latency_ms`
- total V17 latency。
- effective FPS。
- loop dt。
- CPU/GPU/power。

通过标准：

- TensorRT actor residual 相对 PyTorch 明确下降。
- full runtime p50/FPS 有可复现改善。
- p95 若仍受视觉/系统抖动影响，只记录原因，不作为 TensorRT 失败。
- 不把 model-only、pilot smoke、runtime shadow、active run 混成一个倍数结论。

## 3. 安全阻断与故障注入矩阵

### 实验 D：shadow non-takeover gate

目的：证明 shadow 模式不会接管 actuator。

通过标准：

- runtime 输出显示 `V17 shadow pilot injected; actuator path remains user/manual`。
- shadow run 中 actuator 仍由 user/manual 控制。
- `ShadowModeGuard` 强制 user mode 的日志可见，或在 summary/run notes 中确认。

### 实验 E：engine missing / metadata missing

目的：验证 TensorRT artifact 缺失不会导致静默错误。

建议场景：

- 传入不存在的 engine 路径。
- 传入不存在的 metadata 路径。
- 传入 engine 和 metadata shape 不匹配的组合，如果已有安全样本。

通过标准：

- shadow/active 初始化失败必须显式报错。
- active 不允许在 engine 缺失或 metadata 不匹配时继续接管。
- 错误日志记录 engine path 和 metadata path。

### 实验 F：LiDAR disconnected / stale

目的：验证 LiDAR 断开、卡住或 stale 时能被记录并阻断 active 或触发安全输出。

建议场景：

- `--disable-lidar` shadow。
- LiDAR topic 不可用。
- LiDAR driver 启动超时。
- scan age p95 > 350ms。

通过标准：

- summary 中能看到 LiDAR connected、points_total、scan age 或缺失状态。
- active 不应在 LiDAR required 但 unavailable/stale 时进入正式接管。
- 如果当前 runtime 只能记录不能阻断，需要在报告中标记为安全工程缺口，不能包装成已完成能力。

### 实验 G：sensor missing

目的：验证 RP2040 或关键自车状态缺失时的部署行为。

建议场景：

- 串口设备不存在。
- RP2040 无数据。
- speed/yaw/accel 长时间为默认值。

通过标准：

- 缺失状态被记录到 run notes 或 summary。
- active 不应在自车状态不可用时作为正式部署通过。
- 如果当前只能 fallback 到默认值，需要明确标为风险。

### 实验 H：inference timeout / slow loop

目的：验证推理超时或 loop 抖动不会被误判为模型效果问题。

建议场景：

- 用 shadow 长 run 捕捉自然 outlier。
- 若后续允许轻量代码，可增加 inference timeout 计数。

通过标准：

- summary 中记录 V17 latency p95/p99/max。
- loop dt p95/p99/max 可追溯。
- 超时原因被归类为 runtime/jitter/vision/LiDAR，而不是模型避障效果。

## 4. 不再执行的模型效果验收

以下内容从端侧部署验收中删除：

- obstacle recovery 成功率。
- active 跑圈。
- active 避障通过率。
- “是否乱跑”作为部署链路成败指标。
- 用一次 active 实车行为判断 TensorRT/ONNX 部署质量。

如后续必须做 active，只允许作为硬件接管链路 smoke：

- 极短时长。
- 极低速。
- 人工随时接管。
- 只验证接管/停止/记录链路，不评价模型策略效果。

## 5. 输出物与结论口径

每个实验输出目录：

- `/home/jetson/mycar/monitor_logs/<experiment_name>_<timestamp>/`
- `run_*.csv`
- `summary.json`
- `part_profile.json` 或等价 profile
- `run_notes.json`
- 原始命令记录

每个结论必须说明口径：

- model-only：只说明 TensorRT engine 本体。
- pilot smoke：只说明 `v17_pilot.py` 单进程路径。
- runtime shadow：说明完整 DonkeyCar runtime monitor 但不接管车辆。
- active smoke：只说明硬件接管链路，不说明模型避障效果。

最终端侧部署结论只允许写：

> V17 模型已能在 Jetson 上通过 ONNX/TensorRT 链路稳定运行，并具备可观测、可复现、可安全阻断的部署证据链。

不能写：

> V17 已具备实车避障能力。
