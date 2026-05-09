# Phase F Current Issues (April 22, 2026)

> Update on April 24, 2026:
> this note is now historical context for the April 22 packet-path and pose sweeps.
> The later simulator debug freeze is:
> `offset_y = 0.40`, `offset_z = 0.50`, `rot_x = 0.0`, `max_range = 20.0m`,
> with packet interpretation fixed to full `360 deg`, ego-forward at `rx ~= 180 deg`,
> and packet distance scaling `d / 8 -> telemetry meters`.
> The remaining blocker is no longer the old parse-path issue; it is target-selection stability
> plus the broader sim-real realism gap.

## 1. Scope

This note summarizes the current blocker for `Phase F` in the V17/LWM readiness program.

`Phase F` means:

- collect simulator LiDAR monitor logs
- compare simulator canonical LiDAR against real canonical LiDAR
- decide whether the current sim LiDAR setup is good enough for `deployment_ready`

This document reflects the latest state after:

- fixing the simulator raw-packet canonicalization path
- updating `Phase F` collection/eval to use the same `raw_packet -> canonical` path as runtime
- sweeping several LiDAR installation poses
- rerunning formal `Phase F`

## 2. April 22 Status

Best verified `Phase F` pose in the April 22 sweep:

- `offset_y = 0.25`
- `offset_z = 0.65`
- `rot_x = 0.0`

Latest formal rerun output:

- report: `/tmp/v17_readiness_phase_f_y025_20260422/readiness_report.json`
- eval: `/tmp/v17_readiness_phase_f_y025_20260422/phase_f_deployment/lidar_domain_gap_eval.json`

Current overall metrics:

- `valid_ratio_mae = 0.5730`
- `wasserstein_median = 0.1810`
- `wasserstein_p95 = 0.5493`
- `scene_js_divergence = 0.1727`

Current gate thresholds:

- `valid_ratio_mae <= 0.10`
- `wasserstein_median <= 0.08`
- `wasserstein_p95 <= 0.20`
- `scene_js_divergence <= 0.15`

Conclusion:

- `Phase F` is still `failed`
- `deployment_ready = false`

## 3. What Was Fixed Already

The previous "right side dead sectors" issue was a real code bug, not just a pose problem.

What was fixed:

- runtime canonicalization now prefers Unity `lidar_raw_packet`
- `Phase F` collection now writes `lidar_packet` into JSONL
- `Phase F` eval now prefers `lidar_packet -> canonical` instead of the old reconstructed sparse array

Relevant files:

- `module/lidar.py`
- `module/v17_env.py`
- `gym_donkeycar/envs/donkey_sim.py`
- `scripts/collect_sim_lidar_monitor.py`
- `scripts/eval_lidar_domain_gap.py`

This fix improved `Phase F` materially, but did not make it pass.

Packet-path formal rerun before the latest pose change:

- report: `/tmp/v17_readiness_phase_f_packetpath_20260422/readiness_report.json`
- eval: `/tmp/v17_readiness_phase_f_packetpath_20260422/phase_f_deployment/lidar_domain_gap_eval.json`

Metrics at that stage:

- `valid_ratio_mae = 0.6265`
- `wasserstein_median = 0.2770`
- `wasserstein_p95 = 0.6298`
- `scene_js_divergence = 0.2375`

Then moving from `offset_y=0.20` to `offset_y=0.25` improved the metrics again.

## 4. What The Current Problem Actually Is

The current blocker is no longer "wrong parsing path".

The current blocker is:

- simulator LiDAR visibility topology still does not match real LiDAR
- some sectors miss echoes that real LiDAR sees almost all the time
- some sectors do see echoes, but their range distribution is still wrong

This is a sim-LiDAR realism problem, not just a single parser bug.

## 5. Where It Fails

### 5.1 Coverage mismatch

The largest remaining problem is sector valid-ratio mismatch.

Sectors with `valid_ratio_abs_diff > 0.5` in the current best run:

- `0..11`
- `18..28`
- `33..35`

These correspond to the following canonical angle ranges:

- `0..11`: left/front-left, about `+87.5 deg` down to `+32.5 deg`
- `18..28`: center-right to right/front-right, about `-2.5 deg` down to `-52.5 deg`
- `33..35`: extreme right edge, about `-77.5 deg` down to `-87.5 deg`

Interpretation:

- real LiDAR sees returns in these sectors almost always
- sim LiDAR still drops too many of them

### 5.2 Extreme right edge is still the worst region

Worst sectors in the latest best run:

- `sector 35`, center `-87.5 deg`
- `sector 34`, center `-82.5 deg`
- `sector 33`, center `-77.5 deg`

For these sectors:

- real valid ratio is about `0.992 ~ 0.993`
- sim valid ratio is only about `0.037 ~ 0.089`
- Wasserstein is about `0.536 ~ 0.650`

This is still a structural failure region, not noise-level mismatch.

### 5.3 Some sectors have correct existence but wrong distance distribution

Example sectors:

- `12..15` (`+27.5 deg` to `+12.5 deg`)

These sectors have very small valid-ratio difference, but still large Wasserstein / JS divergence.

Typical pattern:

- real `range_p50` is around `0.37 ~ 0.42`
- sim `range_p50` is around `0.085 ~ 0.09`

Interpretation:

- sim is seeing something much closer than real in those directions
- this looks like near-field self-hit, over-aggressive close return, or missing blind-zone modeling

## 6. What Is Not The Main Problem

### 6.1 It is not mainly a "LiDAR too low" problem

This hypothesis was tested directly.

High-mount sweep output:

- `/tmp/v17_lidar_phase_f_highmount_sweep_20260422/sweep_summary.json`

Tested points included:

- `y=0.40,z=0.30`
- `y=0.50,z=0.30`
- `y=0.60,z=0.30`
- `y=0.70,z=0.30`
- `y=0.80,z=0.30`
- `y=0.60,z=0.50`
- `y=0.70,z=0.50`

Result:

- all of them were much worse than the low baseline
- many runs collapsed to `wasserstein_median = 1.0`

So the current evidence says:

- "raise LiDAR to a clearly roof-like position" is not the fix

### 6.2 DonkeySim default pose is also bad for this task

Default gym-donkeycar LiDAR pose:

- `offset_y = 0.5`
- `offset_z = 0.5`
- `rot_x = 0.0`

Quick comparison output:

- `/tmp/v17_lidar_phase_f_defaultpos_sweep_20260422/sweep_summary.json`

Default-pose result:

- `valid_ratio_mae = 0.9553`
- `wasserstein_median = 1.0`
- `wasserstein_p95 = 1.0`
- `scene_js_divergence = 0.6171`

Interpretation:

- the simulator default pose is not a suitable sim2real reference for this project

## 7. Best Pose Found So Far

Mid-height sweep output:

- `/tmp/v17_lidar_phase_f_midheight_sweep_20260422/sweep_summary.json`

Best point from that sweep:

- `offset_y = 0.25`
- `offset_z = 0.65`
- `rot_x = 0.0`

200-frame sweep result:

- `valid_ratio_mae = 0.5624`
- `wasserstein_median = 0.1710`
- `wasserstein_p95 = 0.5842`
- `scene_js_divergence = 0.1667`

800-frame formal rerun result:

- `valid_ratio_mae = 0.5730`
- `wasserstein_median = 0.1810`
- `wasserstein_p95 = 0.5493`
- `scene_js_divergence = 0.1727`

This is the best verified pose so far, but still not enough to pass.

## 8. Root-Cause Hypothesis

Current evidence supports this diagnosis:

1. The remaining problem is primarily about missing or wrong visibility structure.
2. The simulator still does not model the real sensor's blocked azimuth regions.
3. Some sectors likely need explicit blind-zone handling rather than pure pose tuning.
4. Some sectors still have unrealistic near returns and need stronger close-range filtering or dropout.

In short:

- pose tuning helped
- parser/path fixes helped
- but the remaining gap now looks like `blind-zone + near-hit + dropout realism`

## 9. Recommended Next Steps

Priority order:

1. Add configurable `azimuth mask / blind-zone mask` before canonical aggregation.
2. Revisit `near_clip` now that the packet path is stable.
3. Add `beam/sector dropout` to mimic real missing-return behavior.
4. Only after the above, consider another limited pose sweep.

Things that are currently not worth prioritizing:

- pushing LiDAR much higher to an obvious roof position
- using gym-donkeycar default pose as the baseline
- more broad pose sweeps without blind-zone modeling

## 10. Operational Baseline At That Time

In the April 22 sweep context, the recommended working baseline was:

- keep `Phase F` on packet-path collection/evaluation
- keep LiDAR pose at `offset_y = 0.25`, `offset_z = 0.65`, `rot_x = 0.0`
- treat `Phase F` as still blocked on sim sensor realism

That is no longer the current project-wide frozen sim baseline.
The current frozen simulator baseline is the April 24 debug result noted at the top
of this document.
