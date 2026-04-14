# V16 Curriculum

`src/ppo_multitrack_v16.py` is the main dual-domain obstacle training entrypoint.

## Current Auto Curriculum

Stages:

1. `warmup`
2. `avoid_static`
3. `avoid_mixed`
4. `lane_pid_intro`
5. `lane_pid_full`

Promotion rules:

- `warmup`:
  - minimum stage steps: `1_100_000`
  - window: recent `10` episodes per domain
  - success rule: at least `2` episodes with `ep_soft_lap_count >= 1.0`
  - hard fallback: `1_500_000` stage steps
- `avoid_static` / `avoid_mixed` / `lane_pid_intro`:
  - minimum stage steps: `300_000`
  - window: recent `10` episodes per domain
  - success rule: at least `2` episodes with `ep_soft_lap_count >= 2.0`
  - hard fallback: `1_500_000` stage steps
- `lane_pid_full`:
  - final stage, consumes remaining timesteps

The gate uses `ep_soft_lap_count`, not simulator `lap_time` / `lap_count`.

## Domain Design

`WS`:

- always one obstacle car
- always `static`
- reward overrides are more avoidance-friendly because the map is narrower

`GT`:

- obstacle curriculum is active
- later stages can use `static`, `jitter`, `nudge`, `lane_pid`

Warmup placement:

- fixed obstacle position
- `progress_ratio = 0.5`
- `lateral_ratio = 0.5`
- no random yaw for non-`lane_pid` obstacle cars

Obstacle color:

- pure green `(0, 255, 0)`
- aligned with `mysim/module/green_vehicle_detect.py`

## Logs

Main files:

- `models/<run>/train_metrics.jsonl`
  - periodic training metrics snapshot
  - default frequency: every `500` callback steps
- `models/<run>/curriculum_window.jsonl`
  - one record for each completed episode seen by the auto curriculum gate
  - includes `scene_key`, `logging_key`, `soft_laps`, `episode_len`, `episode_reward`,
    termination flags, reward slices, and the current window contents
  - also records `start`, `stop`, and `end` events for each stage
- `models/<run>/v16_config_<stage>.json`
  - effective config for each stage
- `models/<run>/v16_auto_curriculum_summary.json`
  - full stage-by-stage summary across the whole run

Suggested workflow:

1. Watch `train_metrics.jsonl` for broad stability trends.
2. Read `curriculum_window.jsonl` when a stage upgrades too early or too late.
3. Compare `v16_config_<stage>.json` between runs when tuning thresholds.

## Recommended Command

```bash
cd /home/longzhao/mysim_public
/home/longzhao/miniconda3/envs/donkey37/bin/python src/ppo_multitrack_v16.py \
  --auto-curriculum \
  --sim remote \
  --port 9091 \
  --track-dir /home/longzhao/track \
  --steps 6000000 \
  --save-dir models/v16_auto_curriculum \
  --exp-tag v16_auto \
  --file-metrics-log-freq 250
```

If you want a lighter log file, raise `--file-metrics-log-freq` back to `500` or `1000`.
