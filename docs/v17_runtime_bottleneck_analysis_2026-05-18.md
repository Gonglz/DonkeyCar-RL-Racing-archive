# V17 Runtime Bottleneck Analysis

Date: 2026-05-18

Jetson output directory:

`/home/jetson/mycar/monitor_logs/v17_trt_benchmark_180s_20260518_042116`

Scope:

`single CSI camera + LiDAR /scan + RP2040 ego sensors + V17 shadow`

This analysis does not use the exploratory dual-camera path.

## 1. 180s Shadow A/B Result

| Metric | PyTorch 180s | TensorRT 180s | Change |
|---|---:|---:|---:|
| Backend | PyTorch/manual SB3 actor | TensorRT FP16 actor | - |
| Duration | 179.80 s | 179.34 s | - |
| Logged samples | 305 | 299 | - |
| Inference p50 | 178.159 ms | 168.947 ms | -5.17% |
| Inference p95 | 264.161 ms | 249.953 ms | -5.38% |
| Effective FPS mean | 3.975 | 4.274 | +7.52% |
| CPU load mean | 49.107% | 47.493% | -3.29% |
| GPU load mean | 28.938% | 20.764% | -28.25% |
| Power in mean | 3274.272 mW | 3166.666 mW | -3.29% |
| Loop dt p50 | 183.000 ms | 177.500 ms | -3.01% |
| Loop dt p95 | 694.640 ms | 723.170 ms | +4.11% worse |
| LiDAR scan age p50 | 235.300 ms | 234.900 ms | flat |
| LiDAR scan age p95 | 297.920 ms | 292.920 ms | -1.68% |

Takeaway: TensorRT helps, but it only removes a small part of the full runtime latency. The dominant cost is now before or around the actor: semantic image preprocessing, LiDAR feature construction/transport, and logging/loop jitter.

## 2. Runtime Latency Decomposition

From runtime CSV:

| Metric | PyTorch p50 | TensorRT p50 | Notes |
|---|---:|---:|---|
| Total V17Pilot inference | 178.159 ms | 168.947 ms | What monitor calls inference |
| V17 preprocess | 153.070 ms | 155.082 ms | Dominant path |
| Actor residual | 23.589 ms | 12.612 ms | Approx `inference - preprocess` |
| Loop dt | 183.000 ms | 177.500 ms | Vehicle loop effective period |
| LiDAR age | 235.300 ms | 234.900 ms | Not improved by TensorRT |

Correlations:

| Correlation | PyTorch | TensorRT | Interpretation |
|---|---:|---:|---|
| inference vs preprocess | 0.987 | 0.990 | Total latency is driven by preprocess |
| inference vs LiDAR age | -0.052 | 0.003 | LiDAR age is mostly independent of actor latency |
| loop dt vs inference | 0.453 | 0.188 | V17Pilot contributes to loop time, but p95 jitter has other causes |

Actor optimization space after TensorRT is small: TensorRT reduced actor residual p50 from about 23.6ms to 12.6ms. Even if actor became free, runtime p50 would still be around 155ms unless preprocess is fixed.

## 3. Component Microprofile

Using a real LiDAR sample from the 180s TensorRT JSONL and a deterministic synthetic image:

| Component | p50 | p95 | Notes |
|---|---:|---:|---|
| Official WS image preprocessor | 34.916 ms | 37.361 ms | `CanonicalSemanticWrapper(domain=ws)` |
| LiDAR feature build from 1147 ranges | 43.773 ms | 48.858 ms | `_build_lidar_obs`, 72 sectors |
| TensorRT actor only | 7.787 ms | 11.812 ms | `_predict_action` |
| Full preprocess without live sensor I/O | 79.328 ms | 84.940 ms | image + state + LiDAR + obs dict |

Runtime preprocess p50 is about 155ms, while isolated preprocess is about 80ms. The missing 70-80ms likely comes from real runtime contention and live data movement: `lidar_reader.get_data()` copies full ranges, the ROS bridge JSON-decodes full scans, DataCollector also copies/writes full ranges, and all of this competes on the same Nano CPU.

## 4. Bottleneck Ranking And Optimization Space

### 1. V17 Preprocess: Very High Priority

Current:

- TensorRT runtime preprocess p50: 155.082ms.
- Isolated image preprocessor p50: 34.916ms.
- Isolated LiDAR feature build p50: 43.773ms.
- Isolated full preprocess p50: 79.328ms.

Likely optimization space:

- Conservative: save 40-60ms p50.
- Aggressive: save 70-100ms p50 if LiDAR sectorization and live data copies are removed from the main loop.

Concrete changes:

1. Precompute LiDAR angle-to-sector mapping once.
2. Replace per-sector masks and `np.quantile` loops with vectorized binning, or compute 72-sector range/valid features in the ROS helper.
3. Avoid copying full `ranges` lists twice per loop.
4. Add detailed timing fields inside V17Pilot:
   - `image_preprocess_ms`
   - `sensor_snapshot_ms`
   - `lidar_snapshot_ms`
   - `lidar_feature_ms`
   - `actor_ms`
   - `adapter_safety_ms`

Expected result after first pass:

- Runtime V17Pilot p50 could plausibly move from about 169ms toward 90-120ms.
- With a stronger LiDAR bridge refactor, p50 below 80-100ms is realistic.

### 2. LiDAR Scan Age And Bridge: High Priority

Current:

- LiDAR frame rate estimate: 8.1-8.5Hz.
- Scan age p50: about 235ms.
- Scan age p95: about 293-298ms.
- TensorRT does not change this.

Root cause candidates:

1. ROS helper uses `rospy.wait_for_message()` in a loop rather than a persistent subscriber callback.
2. Helper serializes every scan as JSON with about 1147 ranges.
3. Python3 side parses JSON and stores full lists.
4. `get_data()` copies `ranges` and `intensities` each call.
5. `scan_age_ms` uses LaserScan header stamp; that may include sensor/driver latency and not just Python receipt age.

Likely optimization space:

- Measured scan age could drop from about 235ms to below 100-150ms with a streaming subscriber and receipt timestamp.
- Main-loop CPU overhead can drop if the bridge sends already-sectorized 144-dim V17 features instead of 1147 raw ranges.

Concrete changes:

1. Replace `wait_for_message()` helper with `rospy.Subscriber` callback.
2. Track both `scan_header_age_ms` and `scan_receipt_age_ms`.
3. Move sectorization into the ROS helper and emit:
   - `lidar_ranges_72`
   - `lidar_valid_72`
   - `nearest_min`
   - metadata
4. Keep full raw ranges optional, not always copied into the control path.

Expected result:

- Latency perception improves more than TensorRT can deliver now.
- Main-loop p50 could save another 30-50ms if full-range processing leaves `V17Pilot.run()`.

### 3. DataCollector: High Priority For Jitter

Current TensorRT profile:

| Part | avg | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|
| DataCollector | 48.64ms | 0.04ms | 29.67ms | 555.63ms | 574.16ms |

The p50 is tiny because most frames skip logging. The p99 is huge because logging samples do heavy synchronous work:

- image mean/std over a 1280x720 frame,
- CSV `writerow`,
- `csv_file.flush()` every row,
- raw LiDAR JSONL with full 1147 ranges,
- `lidar_raw_file.flush()` every row.

Likely optimization space:

- p99 loop spikes can drop by 300-500ms.
- Average DataCollector part time can drop from about 49ms toward low single digits.
- FPS p95 stability should improve more than p50 inference.

Concrete changes:

1. Move DataCollector file writes to an async queue/thread.
2. Flush every N rows or every few seconds, not every sample.
3. Make full raw LiDAR JSONL optional for benchmark/debug only.
4. Downsample image stats or compute them on a small ROI/thumbnail.
5. Separate control-path logging from debug artifact logging.

Expected result:

- Loop p95 should improve substantially.
- This will not change V17Pilot p50 much, but it should make real driving less jittery.

### 4. Vehicle Loop: Medium Priority After Above

Current:

- Target vehicle rate: 20Hz.
- Actual effective FPS: about 4.0-4.3.
- Loop dt p50: 177-183ms.
- Loop dt p95: about 695-723ms.

Root cause:

- The loop is blocked by synchronous V17Pilot work and occasional DataCollector stalls.
- Camera and RP2040 parts are not the bottleneck.
- TensorRT actor alone cannot make the loop hit 20Hz.

Likely optimization space:

- After preprocess and DataCollector fixes, 8-12Hz looks realistic.
- 15Hz+ may require an async policy part that consumes the latest sensor snapshot and returns the latest completed action.

Concrete changes:

1. First make V17 preprocess and logging faster.
2. Then consider threaded/asynchronous V17Pilot with careful LSTM-state ownership.
3. Keep shadow mode validation before active.

Expected result:

- Lower loop p50 after preprocess fixes.
- Lower loop p95 after DataCollector async logging.

## 5. Practical Priority Order

1. Add fine-grained timing fields to `V17Pilot.run()`.
2. Optimize LiDAR sectorization and avoid full-range copies in the main loop.
3. Make DataCollector asynchronous and stop flushing every row.
4. Re-run 180s TensorRT shadow.
5. Only then consider async V17Pilot or deeper image preprocessor simplification.

The highest-leverage work is not more TensorRT/PyCUDA. The engine is already fast enough. The next big wins are CPU-side preprocessing, LiDAR transport/feature generation, and synchronous logging jitter.
