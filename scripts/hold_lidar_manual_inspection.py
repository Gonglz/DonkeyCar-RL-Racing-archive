#!/usr/bin/env python3
"""
Hold a V17 env in a manual-inspection state with the current best LiDAR pose.

This is intended for VNC/manual debugging:
- connect to an already running remote DonkeySim
- place the ego car in waveshare
- apply the chosen LiDAR pose
- keep sending no-op actions so the scene stays alive
- print the task-relevant "bad" sectors to inspect manually
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import export_world_model_dataset as wm_export  # noqa: E402


PROBLEM_SECTORS = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
HARD_DEAD_SECTORS = [22, 23, 24, 25, 26, 30, 31]
SELF_HIT_SUSPECT_SECTORS = [27, 28, 29]


def _fmt(arr: np.ndarray, idxs: list[int]) -> str:
    vals = [float(arr[i]) for i in idxs if 0 <= i < int(arr.shape[0])]
    return "[" + ", ".join(f"{v:.3f}" for v in vals) + "]"


def main() -> None:
    ap = argparse.ArgumentParser(description="Hold waveshare for manual LiDAR inspection")
    ap.add_argument("--env-id", default="donkey-waveshare-v0")
    ap.add_argument("--sim", default="remote")
    ap.add_argument("--port", type=int, default=9091)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--curriculum-phase", default="warmup")
    ap.add_argument("--obs-size", type=int, default=128)
    ap.add_argument("--sleep-s", type=float, default=0.20)
    ap.add_argument("--status-every", type=int, default=25)
    ap.add_argument("--hold-after-first-step", action="store_true", default=True)
    ap.add_argument("--body-style", default="donkey")
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--warmup-steer", type=float, default=0.0)
    ap.add_argument("--warmup-throttle", type=float, default=0.0)
    ap.add_argument("--with-obstacle", action="store_true")
    ap.add_argument("--obstacle-mode", default="static")
    ap.add_argument("--obstacle-fixed-progress-ratio", type=float, default=None)
    ap.add_argument("--obstacle-fixed-lateral-ratio", type=float, default=0.10)
    ap.add_argument("--obstacle-progress-min", type=float, default=0.28)
    ap.add_argument("--obstacle-progress-max", type=float, default=0.34)
    ap.add_argument("--ws-obstacle-fixed-progress-ratio", type=float, default=None)
    ap.add_argument("--ws-obstacle-fixed-lateral-ratio", type=float, default=0.10)
    ap.add_argument("--ws-obstacle-progress-min", type=float, default=0.28)
    ap.add_argument("--ws-obstacle-progress-max", type=float, default=0.34)
    ap.add_argument("--lidar-offset-y", type=float, default=0.40)
    ap.add_argument("--lidar-offset-z", type=float, default=0.5)
    ap.add_argument("--lidar-rot-x", type=float, default=0.0)
    ap.add_argument("--lidar-num-sectors", type=int, default=36)
    ap.add_argument("--lidar-fov-deg", type=float, default=180.0)
    ap.add_argument("--lidar-max-range-m", type=float, default=20.0)
    args = ap.parse_args()

    wm_export.v17._install_sim_wait_timeout_patch(timeout_s=35.0, resend_scene_names_s=3.0)
    defaults = wm_export._train_defaults()
    env_id = str(args.env_id).strip() or "donkey-waveshare-v0"
    is_waveshare = env_id == "donkey-waveshare-v0"
    if bool(args.with_obstacle):
        obstacle_mode = str(args.obstacle_mode or "static").strip().lower() or "static"
        obstacle_defaults = {
            "obstacle_enabled": True,
            "obstacle_count": 1,
            "obstacle_free_prob": 0.0,
            "obstacle_modes": [obstacle_mode],
            "obstacle_fixed_progress_ratio": (
                None if args.obstacle_fixed_progress_ratio is None else float(args.obstacle_fixed_progress_ratio)
            ),
            "obstacle_fixed_lateral_ratio": float(args.obstacle_fixed_lateral_ratio),
            "obstacle_progress_min": float(args.obstacle_progress_min),
            "obstacle_progress_max": float(args.obstacle_progress_max),
            "ws_obstacle_free_prob": 0.0,
            "ws_obstacle_modes": [obstacle_mode],
            "ws_obstacle_fixed_progress_ratio": (
                None if args.ws_obstacle_fixed_progress_ratio is None else float(args.ws_obstacle_fixed_progress_ratio)
            ),
            "ws_obstacle_fixed_lateral_ratio": float(args.ws_obstacle_fixed_lateral_ratio),
            "ws_obstacle_progress_min": float(args.ws_obstacle_progress_min),
            "ws_obstacle_progress_max": float(args.ws_obstacle_progress_max),
        }
        defaults.update(obstacle_defaults)

    lidar_cfg = {
        "deg_per_sweep_inc": max(1.0, float(args.lidar_fov_deg) / max(1, int(args.lidar_num_sectors) * 5)),
        "deg_ang_down": 0.0,
        "deg_ang_delta": -1.0,
        "num_sweeps_levels": 1,
        "max_range": float(args.lidar_max_range_m),
        "noise": 0.0,
        "offset_x": 0.0,
        "offset_y": float(args.lidar_offset_y),
        "offset_z": float(args.lidar_offset_z),
        "rot_x": float(args.lidar_rot_x),
    }

    env = wm_export._make_env(
        env_ids=[env_id],
        scene_weights=[1.0],
        port=int(args.port),
        seed=int(args.seed),
        defaults=defaults,
        curriculum_phase=str(args.curriculum_phase),
        obs_size=int(args.obs_size),
        max_range_m=float(args.lidar_max_range_m),
        lidar_num_sectors=int(args.lidar_num_sectors),
        lidar_fov_deg=float(args.lidar_fov_deg),
        sim_path=str(args.sim),
        sim_start_delay=10.0,
        image_channel_indices=[0, 1, 2, 3, 4, 5],
        lidar_config_override=lidar_cfg,
        conf_override={
            "body_style": str(args.body_style).strip() or "donkey",
        },
    )

    try:
        obs = env.reset()
        time.sleep(1.0)
        action_shape = tuple(int(x) for x in getattr(env.action_space, "shape", ()) or ())
        if not action_shape:
            raise RuntimeError(f"unexpected action_space shape: {getattr(env.action_space, 'shape', None)}")
        zero_action = np.zeros(action_shape, dtype=np.float32)
        warmup_action = np.zeros(action_shape, dtype=np.float32)
        warmup_action[0] = np.float32(args.warmup_steer)
        if warmup_action.shape[0] >= 2:
            warmup_action[1] = np.float32(args.warmup_throttle)
        print("Manual LiDAR Inspection")
        print(f"  sim={args.sim} port={args.port} env_id={env_id}")
        print(f"  body_style={str(args.body_style).strip() or 'donkey'}")
        print(f"  lidar_config={json.dumps(lidar_cfg, ensure_ascii=False)}")
        print(f"  canonical_max_range_m={float(args.lidar_max_range_m):.3f}")
        print(
            f"  warmup_steps={int(args.warmup_steps)} "
            f"warmup_steer={float(args.warmup_steer):.3f} "
            f"warmup_throttle={float(args.warmup_throttle):.3f}"
        )
        if bool(args.with_obstacle):
            if is_waveshare:
                print(
                    f"  obstacle=on mode={str(args.obstacle_mode).strip().lower()} "
                    f"ws_lateral_ratio={float(args.ws_obstacle_fixed_lateral_ratio):.3f} "
                    f"ws_progress=[{float(args.ws_obstacle_progress_min):.3f}, {float(args.ws_obstacle_progress_max):.3f}]"
                )
            else:
                print(
                    f"  obstacle=on mode={str(args.obstacle_mode).strip().lower()} "
                    f"gt_fixed_progress={args.obstacle_fixed_progress_ratio!r} "
                    f"gt_lateral_ratio={float(args.obstacle_fixed_lateral_ratio):.3f} "
                    f"gt_progress=[{float(args.obstacle_progress_min):.3f}, {float(args.obstacle_progress_max):.3f}]"
                )
        else:
            print("  obstacle=off")
        print(f"  inspect right-front/right-side sectors={PROBLEM_SECTORS}")
        print(f"  hard-dead sectors={HARD_DEAD_SECTORS}")
        print(f"  self-hit suspect sectors={SELF_HIT_SUSPECT_SECTORS}")
        print("  canonical sectors are ordered left -> right over the front 180 FOV")
        print("  focus on the ego car's right-front/right-side enclosure and self-occlusion")
        print("  Ctrl-C to stop")

        step = 0
        while True:
            action = warmup_action if step < int(args.warmup_steps) else zero_action
            obs, reward, done, info = env.step(action)
            step += 1

            lidar_flat = np.asarray(obs["lidar"], dtype=np.float32).reshape(-1)
            half = lidar_flat.shape[0] // 2
            lidar_range = lidar_flat[:half]
            lidar_valid = lidar_flat[half:]

            if step == 1 or (step % int(args.status_every) == 0):
                print(
                    f"[step {step}] "
                    f"scene={info.get('scene_key','')} "
                    f"speed={float(info.get('speed', 0.0) or 0.0):.3f} "
                    f"obstacle_present={float(info.get('obstacle_present', 0.0) or 0.0):.0f} "
                    f"obs_long={float(info.get('obstacle_longitudinal', 0.0) or 0.0):.3f} "
                    f"obs_lat={float(info.get('obstacle_lateral', 0.0) or 0.0):.3f} "
                    f"valid={_fmt(lidar_valid, PROBLEM_SECTORS)} "
                    f"range={_fmt(lidar_range, PROBLEM_SECTORS)}"
                )

            hold_after_step = max(1, int(args.warmup_steps) + 1)
            if bool(args.hold_after_first_step) and step >= hold_after_step:
                print("[hold] scene is now static for manual inspection")
                while True:
                    time.sleep(1.0)

            if done:
                print(f"[reset] done=True reason={info.get('done_reason', '')}")
                obs = env.reset()
            time.sleep(float(args.sleep_s))
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
