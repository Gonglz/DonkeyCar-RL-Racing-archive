# V16 `lane_pid` Stage Override Bug

## Summary

In the `v16_sim2real_auto` run, auto curriculum stage promotion did reach
`lane_pid_intro` and then `lane_pid_full`, but the GT obstacle mode set did not
switch to include `lane_pid`.

The result was:

- stage labels and training tags showed `lane_pid_*`
- `waveshare` still behaved as expected for WS (`static` only)
- `generated_track` never actually received `lane_pid` obstacle modes
- visual inspection in sim looked like training was still stuck in an avoid stage

This was a real bug, not just an observation error.

## Affected Run

- save dir: `models/v16_sim2real_auto`
- script: `src/ppo_multitrack_v16.py`
- branch: `feature/sim2real-pink-obstacle`

## Symptoms

Observed in sim:

- GT obstacle cars never entered `lane_pid`
- behavior looked like `avoid_mixed` instead of `lane_pid_intro` / `lane_pid_full`

Observed in artifacts:

- `curriculum_window.jsonl` showed stage promotion into `lane_pid_intro`
- `train_metrics.jsonl` contained many `sim2real_auto_lane_pid_intro` and
  `sim2real_auto_lane_pid_full` records
- saved config for `lane_pid_intro` still recorded GT obstacle modes as
  `["static", "jitter"]`

## Evidence

### 1. Stage gate really advanced

From `models/v16_sim2real_auto/curriculum_window.jsonl`:

- `avoid_mixed` stopped at `2026-04-21T08:53:53.405786`
- `lane_pid_intro` started at `2026-04-21T08:54:02.615380`
- `lane_pid_intro` stopped at `2026-04-21T17:46:19.927363`

This proves the curriculum gate itself was not stuck in `avoid_mixed`.

### 2. Training tags also moved into lane-pid stages

From `models/v16_sim2real_auto/train_metrics.jsonl`:

- `sim2real_auto_avoid_mixed`: 843 step records
- `sim2real_auto_lane_pid_intro`: 757 step records
- `sim2real_auto_lane_pid_full`: 830 step records

So the run was not merely misnamed at one timestamp. It spent substantial time
inside both `lane_pid` stages.

### 3. Saved stage config did not apply GT lane-pid modes

From `models/v16_sim2real_auto/v16_config_lane_pid_intro.json`:

- `curriculum.phase` was `"lane_pid_intro"`
- but `obstacle_runtime.modes` was still `["static", "jitter"]`
- expected value was `["static", "lane_pid"]`

This is the key contradiction: the stage name changed, but the GT obstacle mode
set did not.

### 4. Earlier stage config shows the same stale mode set

From `models/v16_sim2real_auto/v16_config_avoid_mixed.json`:

- `curriculum.phase` was `"avoid_mixed"`
- `obstacle_runtime.modes` was still `["static", "jitter"]`
- expected value for `avoid_mixed` was `["static", "jitter", "nudge"]`

So the problem was broader than just `lane_pid_intro`: curriculum stage-specific
`obstacle_modes` were not being applied at all.

## Root Cause

The bug came from an interaction between:

- `_apply_curriculum_phase()` in `src/ppo_multitrack_v16.py`
- `TRAIN_V16_DEFAULTS["obstacle_modes"] = None`
- CLI parser default for `--obstacle-modes`

The curriculum override logic only applies a stage value when the current value
matches the training default:

```python
if current_value == default_value:
    values[key] = cloned
```

But the CLI parser had:

```python
parser.add_argument("--obstacle-modes", nargs="+", default=["static", "jitter"])
```

That meant `obstacle_modes` was always passed in as `["static", "jitter"]`,
even when the user did not explicitly set it.

So for every curriculum stage:

- `current_value` was `["static", "jitter"]`
- `default_value` was `None`
- the override was skipped

As a result, stage-specific obstacle mode sets such as:

- `["static", "jitter", "nudge"]`
- `["static", "lane_pid"]`
- `["static", "jitter", "nudge", "lane_pid"]`

never replaced the CLI default.

## Fix

Changed the CLI default for `--obstacle-modes` from a concrete list to `None`:

- `src/ppo_multitrack_v16.py`
- `src/ppo_multitrack_v16_multisim.py`

New behavior:

```python
parser.add_argument("--obstacle-modes", nargs="+", default=None)
```

This lets curriculum stages own `obstacle_modes` unless the user explicitly
passes `--obstacle-modes ...`.

## Impact

### Existing run

The already-running `v16_sim2real_auto` job was affected for its whole
`lane_pid_intro` and `lane_pid_full` phases. It cannot be retroactively fixed by
changing the code after the fact.

### Future runs

New runs launched with the patched scripts will correctly apply:

- `avoid_mixed` -> GT can include `nudge`
- `lane_pid_intro` -> GT can include `lane_pid`
- `lane_pid_full` -> GT can include `lane_pid` plus the full mixed mode set

## Recommended Verification

For the next run:

1. start auto curriculum from a checkpoint before `lane_pid_intro`, or restart
   the run cleanly
2. inspect `v16_config_lane_pid_intro.json`
3. confirm `obstacle_runtime.modes` includes `lane_pid`
4. optionally add episode-level mode logging if visual confirmation in sim is
   still ambiguous

## Files Changed

- `docs/V16_LANE_PID_STAGE_OVERRIDE_BUG.md`
- `src/ppo_multitrack_v16.py`
- `src/ppo_multitrack_v16_multisim.py`
