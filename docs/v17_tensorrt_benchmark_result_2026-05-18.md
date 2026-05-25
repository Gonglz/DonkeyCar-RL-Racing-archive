# V17 TensorRT Benchmark Result

Date: 2026-05-18

Jetson output directory:

`/home/jetson/mycar/monitor_logs/v17_trt_benchmark_20260518_040300`

## Scope

This benchmark uses the V17 main deployment path:

`single CSI camera + LiDAR /scan + RP2040 ego sensors + V17Pilot`

The exploratory dual-camera note is not used as a performance baseline.

## Correctness

TensorRT runtime smoke exited successfully with expected bindings:

- `image`: `(1, 6, 128, 128)`
- `state`: `(1, 7)`
- `lidar`: `(1, 144)`
- `lidar_meta`: `(1, 2)`
- `h/c`: `(2, 1, 256)`
- `action`: `(1, 3)`

PyTorch vs TensorRT deterministic action comparison:

| Item | Value |
|---|---:|
| max abs diff | `0.0006172265857458115` |
| tolerance | `0.02` |

## Pilot Smoke A/B

Command path: `python v17_pilot.py --frames 300`

| Metric | PyTorch | TensorRT | Change |
|---|---:|---:|---:|
| p50 latency | 40.246 ms | 34.346 ms | -14.66% |
| p95 latency | 45.486 ms | 44.982 ms | -1.11% |
| avg latency | 82.900 ms | 36.900 ms | startup-sensitive |
| avg preprocess | 22.400 ms | 23.200 ms | roughly same |

## Runtime Shadow A/B

Command path: `python runtime_monitor.py drive --control-mode shadow --shadow-duration 60`

| Metric | PyTorch shadow 60s | TensorRT shadow 60s | Change |
|---|---:|---:|---:|
| frames logged | 99 | 98 | - |
| inference p50 | 180.689 ms | 168.570 ms | -6.71% |
| inference p95 | 294.797 ms | 253.958 ms | -13.85% |
| effective FPS mean | 4.028 | 4.270 | +6.01% |
| CPU load mean | 51.224% | 46.525% | -9.17% |
| GPU load mean | 29.881% | 13.684% | -54.21% |
| power in mean | 3307.576 mW | 3093.224 mW | -6.48% |
| loop dt p50 | 189.300 ms | 192.050 ms | about flat |
| loop dt p95 | 705.870 ms | 696.175 ms | -1.37% |
| LiDAR scan age p50 | 235.300 ms | 241.200 ms | about flat |

## Interpretation

TensorRT helps, but the full runtime is now dominated by preprocessing, LiDAR scan age, DataCollector cost, and vehicle-loop jitter. The engine itself is fast, but the vehicle runtime path only sees part of that improvement.

Best next step: run the same shadow A/B for 180 seconds, then profile V17 preprocessing and LiDAR feature construction separately.
