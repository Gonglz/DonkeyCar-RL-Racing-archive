import math
import unittest
from pathlib import Path

import numpy as np

from module.obstacle import (
    DonkeyObstacleCar,
    ObstacleSnapshot,
    PoseState,
    compute_relative_state,
    pose_from_info,
    sample_track_target,
)
from module.obstacle_runtime import ObstacleRuntimeConfig, ObstacleRuntimeManager
from module.track import TrackGeometryManager
from src import ppo_multitrack_v17 as v17


class _FakePreset:
    staging_x_start = -10.0
    staging_x_step = 1.2
    staging_z = -6.0


class _FakeObstacleCar:
    default_world_y = 0.5

    def __init__(self):
        self.stop_motion_calls = []
        self.place_pose_calls = []
        self.teleport_pose_calls = []

    def stop_motion(self, hold_brake=True):
        self.stop_motion_calls.append({"hold_brake": hold_brake})

    def place_pose(self, **kwargs):
        self.place_pose_calls.append(dict(kwargs))

    def teleport_pose(self, **kwargs):
        self.teleport_pose_calls.append(dict(kwargs))


class _FallbackObstacleCar(_FakeObstacleCar):
    def __init__(self):
        super().__init__()
        self.targets = []
        self._target = None
        self.reset_calls = []

    def place_explicit_target(self, target, **kwargs):
        self.targets.append(target)
        self._target = target
        return target

    def reset(self, reason="manual"):
        self.reset_calls.append(reason)

    def get_obstacle_pose(self):
        if self._target is None:
            return None
        unstable = len(self.targets) == 1
        return PoseState(
            x=float(self._target.x),
            y=-2.0 if unstable else 0.08,
            z=float(self._target.z),
            yaw_deg=float(self._target.yaw_deg),
            speed=50.0 if unstable or not self.reset_calls else 0.0,
            cte=0.0,
            hit="none",
            progress_ratio=float(self._target.progress_ratio),
        )

    def debug_state(self):
        return {}


class _UnstableLanePIDObstacleCar(_FakeObstacleCar):
    def __init__(self, target):
        super().__init__()
        self.target = target
        self.start_lane_pid_calls = []
        self.reset_calls = []

    def start_lane_pid(self, **kwargs):
        self.start_lane_pid_calls.append(dict(kwargs))
        return self.target

    def reset(self, reason="manual"):
        self.reset_calls.append(reason)

    def get_obstacle_pose(self):
        return PoseState(
            x=float(self.target.x),
            y=-3.0,
            z=float(self.target.z),
            yaw_deg=float(self.target.yaw_deg),
            speed=195.0,
            cte=0.0,
            hit="none",
            progress_ratio=float(self.target.progress_ratio),
        )

    def debug_state(self):
        return {}


class _TargetErrorLanePIDObstacleCar(_UnstableLanePIDObstacleCar):
    def get_obstacle_pose(self):
        return PoseState(
            x=float(self.target.x + 1.0),
            y=0.08,
            z=float(self.target.z),
            yaw_deg=float(self.target.yaw_deg),
            speed=0.0,
            cte=0.0,
            hit="none",
            progress_ratio=float(self.target.progress_ratio),
        )


class _FakeFleet:
    preset = _FakePreset()

    def __init__(self, cars):
        self.cars = cars
        self.shutdown_calls = 0

    def last_errors(self):
        return [None for _ in self.cars]

    def shutdown(self):
        self.shutdown_calls += 1


class V17CurriculumTests(unittest.TestCase):
    def test_auto_curriculum_starts_at_warmup_without_removed_phases(self):
        stage_names = [stage["stage_name"] for stage in v17.AUTO_CURRICULUM_STAGES]

        self.assertEqual(stage_names[0], "warmup")
        self.assertNotIn("ws_bootstrap", stage_names)
        self.assertNotIn("lane_pid_dense", stage_names)
        self.assertNotIn("ws_bootstrap", v17.CURRICULUM_PHASES)
        self.assertNotIn("lane_pid_dense", v17.CURRICULUM_PHASES)

    def test_v17_free_prob_schedule(self):
        for phase_name in ("avoid_static", "avoid_mixed"):
            phase = v17.CURRICULUM_PHASES[phase_name]
            self.assertEqual(phase["obstacle_free_prob"], 0.10)
            self.assertEqual(phase["ws_obstacle_free_prob"], 0.10)
            self.assertEqual(phase["obstacle_target_ratios"], {"ws": 0.90, "gt": 0.90})

        for phase_name in ("lane_pid_intro", "lane_pid_mid", "lane_pid_full"):
            phase = v17.CURRICULUM_PHASES[phase_name]
            self.assertEqual(phase["obstacle_free_prob"], 0.05)
            self.assertEqual(phase["ws_obstacle_free_prob"], 0.05)
            self.assertEqual(phase["obstacle_target_ratios"], {"ws": 0.95, "gt": 0.95})

    def test_avoid_static_uses_stronger_clearance_penalties_and_gate_limits(self):
        phase = v17.CURRICULUM_PHASES["avoid_static"]
        reward_overrides = phase["reward_overrides_by_logging_key"]

        self.assertEqual(reward_overrides["ws"]["obstacle_clearance_penalty_scale"], 1.0)
        self.assertEqual(reward_overrides["gt"]["obstacle_clearance_penalty_scale"], 0.8)

        stage = next(
            stage for stage in v17.AUTO_CURRICULUM_STAGES
            if stage["stage_name"] == "avoid_static"
        )
        self.assertEqual(
            stage["max_obstacle_clearance_critical_rate_by_key"],
            {"ws": 0.0, "gt": 0.0},
        )
        self.assertEqual(
            stage["max_obstacle_clearance_band_rate_by_key"],
            {"ws": 0.05, "gt": 0.05},
        )

    def test_lane_pid_stages_keep_clearance_gate_limits(self):
        for stage_name in ("lane_pid_intro", "lane_pid_mid", "lane_pid_full"):
            stage = next(
                stage for stage in v17.AUTO_CURRICULUM_STAGES
                if stage["stage_name"] == stage_name
            )
            self.assertEqual(
                stage["max_obstacle_clearance_critical_rate_by_key"],
                {"ws": 0.0, "gt": 0.0},
                stage_name,
            )
            self.assertEqual(
                stage["max_obstacle_clearance_band_rate_by_key"],
                {"ws": 0.05, "gt": 0.05},
                stage_name,
            )

    def test_lane_pid_stages_use_gradual_speed_schedule(self):
        expected = {
            "lane_pid_intro": {
                "obstacle_speed": 0.45,
                "ws_speed_ref_vmax": 0.62,
                "ws_w_speed_ref": 0.14,
                "gt_speed_ref_vmax": 0.72,
                "gt_w_speed_ref": 0.07,
            },
            "lane_pid_mid": {
                "obstacle_speed": 0.50,
                "ws_speed_ref_vmax": 0.85,
                "ws_w_speed_ref": 0.09,
                "gt_speed_ref_vmax": 0.95,
                "gt_w_speed_ref": 0.05,
            },
            "lane_pid_full": {
                "obstacle_speed": 0.55,
                "ws_speed_ref_vmax": 1.00,
                "ws_w_speed_ref": 0.08,
                "gt_speed_ref_vmax": 1.10,
                "gt_w_speed_ref": 0.03,
            },
        }

        for stage_name, values in expected.items():
            phase = v17.CURRICULUM_PHASES[stage_name]
            reward_overrides = phase["reward_overrides_by_logging_key"]
            self.assertEqual(phase["obstacle_lane_pid_speed_ws"], values["obstacle_speed"], stage_name)
            self.assertEqual(phase["obstacle_lane_pid_speed_gt"], values["obstacle_speed"], stage_name)
            self.assertEqual(reward_overrides["ws"]["speed_ref_vmax"], values["ws_speed_ref_vmax"], stage_name)
            self.assertEqual(reward_overrides["ws"]["w_speed_ref"], values["ws_w_speed_ref"], stage_name)
            self.assertEqual(reward_overrides["gt"]["speed_ref_vmax"], values["gt_speed_ref_vmax"], stage_name)
            self.assertEqual(reward_overrides["gt"]["w_speed_ref"], values["gt_w_speed_ref"], stage_name)

    def test_lane_pid_stages_require_lane_pid_obstacle_for_gate(self):
        for stage_name in ("lane_pid_intro", "lane_pid_mid", "lane_pid_full"):
            stage = next(
                stage for stage in v17.AUTO_CURRICULUM_STAGES
                if stage["stage_name"] == stage_name
            )
            self.assertEqual(stage["required_obstacle_mode_for_gate"], "lane_pid", stage_name)

    def test_lane_pid_gate_rejects_soft_lap_without_lane_pid_obstacle(self):
        callback = v17.CurriculumWindowAdvanceCallback(
            stage_name="lane_pid_intro",
            required_logging_keys=["ws"],
            min_stage_timesteps=0,
            recent_episodes=10,
            min_success_episodes=1,
            min_soft_laps=2.0,
            required_obstacle_mode="lane_pid",
        )

        gate_success, reason = callback._record_gate_success(
            {
                "logging_key": "ws",
                "episode_reward": 100.0,
                "episode_len": 500,
                "ep_obstacle_has_lane_pid": 0.0,
            },
            soft_laps=3.0,
            term_collision=0.0,
        )
        self.assertEqual((gate_success, reason), (0.0, "required_lane_pid_obstacle_missing"))

        gate_success, reason = callback._record_gate_success(
            {
                "logging_key": "ws",
                "episode_reward": 100.0,
                "episode_len": 500,
                "ep_obstacle_has_lane_pid": 1.0,
            },
            soft_laps=3.0,
            term_collision=0.0,
        )
        self.assertEqual((gate_success, reason), (1.0, "soft_lap"))

    def test_clearance_gate_rejects_soft_lap_with_close_obstacle(self):
        callback = v17.CurriculumWindowAdvanceCallback(
            stage_name="avoid_static",
            required_logging_keys=["ws"],
            min_stage_timesteps=0,
            recent_episodes=10,
            min_success_episodes=1,
            min_soft_laps=2.0,
            max_obstacle_clearance_critical_rate_by_key={"ws": 0.0},
            max_obstacle_clearance_band_rate_by_key={"ws": 0.05},
        )

        gate_success, reason = callback._record_gate_success(
            {
                "logging_key": "ws",
                "episode_reward": 100.0,
                "episode_len": 500,
                "ep_obstacle_clearance_critical_rate": 0.01,
                "ep_obstacle_clearance_band_rate": 0.04,
            },
            soft_laps=2.0,
            term_collision=0.0,
        )
        self.assertEqual((gate_success, reason), (0.0, "clearance_critical_above_threshold"))

        gate_success, reason = callback._record_gate_success(
            {
                "logging_key": "ws",
                "episode_reward": 100.0,
                "episode_len": 500,
                "ep_obstacle_clearance_critical_rate": 0.0,
                "ep_obstacle_clearance_band_rate": 0.08,
            },
            soft_laps=2.0,
            term_collision=0.0,
        )
        self.assertEqual((gate_success, reason), (0.0, "clearance_band_above_threshold"))

        gate_success, reason = callback._record_gate_success(
            {
                "logging_key": "ws",
                "episode_reward": 100.0,
                "episode_len": 500,
                "ep_obstacle_clearance_critical_rate": 0.0,
                "ep_obstacle_clearance_band_rate": 0.03,
            },
            soft_laps=2.0,
            term_collision=0.0,
        )
        self.assertEqual((gate_success, reason), (1.0, "soft_lap"))

    def test_resume_learn_target_adds_stage_budget_to_existing_timesteps(self):
        self.assertEqual(
            v17._resolve_learn_total_timesteps(
                requested_timesteps=800_000,
                model_start_timesteps=754_661,
                resume_ckpt_path="/tmp/model.zip",
            ),
            1_554_661,
        )
        self.assertEqual(
            v17._resolve_learn_total_timesteps(
                requested_timesteps=800_000,
                model_start_timesteps=754_661,
                resume_ckpt_path=None,
            ),
            800_000,
        )

    def test_ws_obstacle_laterals_use_edge_targets(self):
        checked_phases = (
            "warmup",
            "warmup_a",
            "avoid_static",
            "avoid_mixed",
            "lane_pid_intro",
            "lane_pid_mid",
            "lane_pid_full",
        )
        for phase_name in checked_phases:
            choices = v17.CURRICULUM_PHASES[phase_name].get("ws_obstacle_lateral_choices")
            self.assertEqual(choices, [0.0, 1.0], phase_name)

        manager = ObstacleRuntimeManager(
            track_geometry=None,
            conf={},
            track_dir="",
            config=ObstacleRuntimeConfig(),
        )
        manager.attach_scene(None, "donkey-waveshare-v0", "waveshare", "ws")

        self.assertEqual(manager._lane_choices_for_scene(), (0.0, 1.0))

    def test_ws_edge_target_sampling_ignores_spawn_clearance(self):
        track_dir = str(Path("module/track_data").resolve())
        track_geometry = TrackGeometryManager(
            track_dir=track_dir,
            env_ids=v17.DEFAULT_ENV_IDS,
            scene_specs=v17.SCENE_SPECS,
        )
        manager = ObstacleRuntimeManager(
            track_geometry=track_geometry,
            conf={},
            track_dir=track_dir,
            config=ObstacleRuntimeConfig(
                ws_lateral_choices=(0.0, 1.0),
                ws_obstacle_progress_min=0.40,
                ws_obstacle_progress_max=0.40,
                min_agent_planar_dist_m=0.0,
                min_agent_arc_dist_m=0.0,
                seed=7,
            ),
        )
        manager.attach_scene(None, "donkey-waveshare-v0", "waveshare", "ws")

        obstacle_radius, safety_margin = manager._effective_spawn_clearance()
        targets = manager._sample_episode_targets(
            {
                "pos": (0.0, 0.0, -0.963),
                "car": (0.0, 0.0, 90.0),
                "speed": 0.0,
                "cte": 0.0,
                "hit": "none",
            },
            count=8,
        )

        self.assertEqual((obstacle_radius, safety_margin), (0.0, 0.0))
        self.assertTrue(targets)
        self.assertTrue(all(target.lateral_ratio in (0.0, 1.0) for target in targets))

    def test_ws_lane_pid_lookahead_preserves_start_clearance_for_edge_lateral(self):
        track_dir = str(Path("module/track_data").resolve())
        track_geometry = TrackGeometryManager(
            track_dir=track_dir,
            env_ids=v17.DEFAULT_ENV_IDS,
            scene_specs=v17.SCENE_SPECS,
        )
        car = DonkeyObstacleCar(
            env_id="donkey-waveshare-v0",
            track_geometry=track_geometry,
            scene_key="waveshare",
            placement_timeout_s=0.1,
        )
        anchor = car.start_lane_pid(
            target_speed=0.5,
            progress_ratio=0.40,
            lateral_ratio=0.0,
            obstacle_radius=0.0,
            safety_margin=0.0,
            place_on_start=False,
        )
        pose = PoseState(
            x=anchor.x,
            y=0.0,
            z=anchor.z,
            yaw_deg=anchor.yaw_deg,
            speed=0.0,
            cte=0.0,
            hit="none",
            progress_ratio=anchor.progress_ratio,
        )

        car._compute_lane_pid_action(pose, car._lane_pid_cfg)
        g = track_geometry.scenes["waveshare"]
        lookahead_progress = anchor.progress_ratio + car._lane_pid_cfg.lookahead_m / g.loop_len
        expected_edge_target = sample_track_target(
            track_geometry=track_geometry,
            scene_key="waveshare",
            progress_ratio=lookahead_progress,
            lateral_ratio=0.0,
            obstacle_radius=0.0,
            safety_margin=0.0,
        )
        default_clipped_target = sample_track_target(
            track_geometry=track_geometry,
            scene_key="waveshare",
            progress_ratio=lookahead_progress,
            lateral_ratio=0.0,
        )
        edge_error = math.hypot(
            float(car._target.x - expected_edge_target.x),
            float(car._target.z - expected_edge_target.z),
        )
        clipped_error = math.hypot(
            float(car._target.x - default_clipped_target.x),
            float(car._target.z - default_clipped_target.z),
        )

        self.assertEqual(anchor.lateral_ratio, 0.0)
        self.assertEqual(car._target.lateral_ratio, 0.0)
        self.assertLess(edge_error, 1e-6)
        self.assertGreater(clipped_error, 0.05)

    def test_ws_lane_pid_watchdog_ignores_lookahead_target_error(self):
        track_dir = str(Path("module/track_data").resolve())
        track_geometry = TrackGeometryManager(
            track_dir=track_dir,
            env_ids=v17.DEFAULT_ENV_IDS,
            scene_specs=v17.SCENE_SPECS,
        )
        manager = ObstacleRuntimeManager(
            track_geometry=track_geometry,
            conf={},
            track_dir=track_dir,
            config=ObstacleRuntimeConfig(),
        )
        manager.attach_scene(None, "donkey-waveshare-v0", "waveshare", "ws")
        manager._episode_index = 1
        manager._active_this_episode = True
        manager._episode_modes_used = ("lane_pid",)
        manager._debug_step_count_this_episode = 1
        manager._fleet = _FakeFleet([_FakeObstacleCar()])

        anchor = sample_track_target(
            track_geometry=track_geometry,
            scene_key="waveshare",
            progress_ratio=0.40,
            lateral_ratio=0.0,
            obstacle_radius=0.0,
            safety_margin=0.0,
        )
        g = track_geometry.scenes["waveshare"]
        lookahead = sample_track_target(
            track_geometry=track_geometry,
            scene_key="waveshare",
            progress_ratio=anchor.progress_ratio + 1.2 / g.loop_len,
            lateral_ratio=anchor.lateral_ratio,
            obstacle_radius=0.0,
            safety_margin=0.0,
        )
        pose = PoseState(
            x=anchor.x,
            y=0.07,
            z=anchor.z,
            yaw_deg=anchor.yaw_deg,
            speed=0.55,
            cte=0.0,
            hit="none",
            progress_ratio=anchor.progress_ratio,
        )
        self.assertGreater(
            ObstacleRuntimeManager._target_error_debug(pose, lookahead)["planar"],
            0.75,
        )
        logs = []
        manager._log_runtime_debug = lambda event, **fields: logs.append({"event": event, **fields})

        manager._maybe_log_obstacle_watchdog(
            agent_info={},
            snapshots=[
                ObstacleSnapshot(
                    obstacle=pose,
                    target=lookahead,
                    agent=None,
                    relative=None,
                )
            ],
        )

        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["event"], "watchdog")
        self.assertFalse(logs[0]["anomaly"])

    def test_ws_lane_pid_watchdog_keeps_pose_anomaly_checks(self):
        track_dir = str(Path("module/track_data").resolve())
        track_geometry = TrackGeometryManager(
            track_dir=track_dir,
            env_ids=v17.DEFAULT_ENV_IDS,
            scene_specs=v17.SCENE_SPECS,
        )
        manager = ObstacleRuntimeManager(
            track_geometry=track_geometry,
            conf={},
            track_dir=track_dir,
            config=ObstacleRuntimeConfig(),
        )
        manager.attach_scene(None, "donkey-waveshare-v0", "waveshare", "ws")
        manager._episode_index = 1
        manager._active_this_episode = True
        manager._episode_modes_used = ("lane_pid",)
        manager._fleet = _FakeFleet([_FakeObstacleCar()])

        target = sample_track_target(
            track_geometry=track_geometry,
            scene_key="waveshare",
            progress_ratio=0.40,
            lateral_ratio=0.0,
            obstacle_radius=0.0,
            safety_margin=0.0,
        )
        cases = [
            None,
            PoseState(
                x=target.x,
                y=-1.01,
                z=target.z,
                yaw_deg=target.yaw_deg,
                speed=0.55,
                cte=0.0,
                hit="none",
                progress_ratio=target.progress_ratio,
            ),
            PoseState(
                x=target.x,
                y=0.07,
                z=target.z,
                yaw_deg=target.yaw_deg,
                speed=8.01,
                cte=0.0,
                hit="none",
                progress_ratio=target.progress_ratio,
            ),
        ]
        for idx, pose in enumerate(cases):
            with self.subTest(idx=idx):
                manager._debug_step_count_this_episode = 2 + idx
                manager._debug_last_watch_anomaly_t = 0.0
                logs = []
                manager._log_runtime_debug = lambda event, **fields: logs.append({"event": event, **fields})

                manager._maybe_log_obstacle_watchdog(
                    agent_info={},
                    snapshots=[
                        ObstacleSnapshot(
                            obstacle=pose,
                            target=target,
                            agent=None,
                            relative=None,
                        )
                    ],
                )

                self.assertEqual(len(logs), 1)
                self.assertEqual(logs[0]["event"], "watchdog")
                self.assertTrue(logs[0]["anomaly"])

    def test_ws_static_layout_retries_edge_after_reset_when_pose_is_unstable(self):
        track_dir = str(Path("module/track_data").resolve())
        track_geometry = TrackGeometryManager(
            track_dir=track_dir,
            env_ids=v17.DEFAULT_ENV_IDS,
            scene_specs=v17.SCENE_SPECS,
        )
        manager = ObstacleRuntimeManager(
            track_geometry=track_geometry,
            conf={},
            track_dir=track_dir,
            config=ObstacleRuntimeConfig(
                ws_obstacle_count=1,
                ws_obstacle_modes=("static",),
                ws_lateral_choices=(0.75,),
                placement_timeout_s=0.1,
                seed=11,
            ),
        )
        manager.attach_scene(None, "donkey-waveshare-v0", "waveshare", "ws")
        car = _FallbackObstacleCar()
        manager._fleet = _FakeFleet([car])

        edge_target = sample_track_target(
            track_geometry=track_geometry,
            scene_key="waveshare",
            progress_ratio=0.27,
            lateral_ratio=0.75,
            obstacle_radius=0.0,
            safety_margin=0.0,
        )
        manager._sample_episode_targets = lambda agent_info, count: [edge_target]

        active = manager._refresh_obstacle_layout(
            agent_info={
                "pos": (0.0, 0.0, 0.0),
                "car": (0.0, 0.0, 0.0),
                "speed": 0.0,
                "cte": 0.0,
                "hit": "none",
            }
        )

        self.assertTrue(active)
        self.assertTrue(car.reset_calls)
        self.assertGreaterEqual(len(car.targets), 2)
        self.assertEqual(car.targets[-1].lateral_ratio, edge_target.lateral_ratio)
        self.assertEqual(manager._episode_target_plan[0].lateral_ratio, car.targets[-1].lateral_ratio)

    def test_ws_lane_pid_layout_parks_when_initial_pose_is_unstable(self):
        track_dir = str(Path("module/track_data").resolve())
        track_geometry = TrackGeometryManager(
            track_dir=track_dir,
            env_ids=v17.DEFAULT_ENV_IDS,
            scene_specs=v17.SCENE_SPECS,
        )
        manager = ObstacleRuntimeManager(
            track_geometry=track_geometry,
            conf={},
            track_dir=track_dir,
            config=ObstacleRuntimeConfig(
                ws_obstacle_count=1,
                ws_obstacle_modes=("lane_pid",),
                ws_lateral_choices=(0.0,),
                placement_timeout_s=0.1,
                seed=11,
            ),
        )
        manager.attach_scene(None, "donkey-waveshare-v0", "waveshare", "ws")
        edge_target = sample_track_target(
            track_geometry=track_geometry,
            scene_key="waveshare",
            progress_ratio=0.27,
            lateral_ratio=0.0,
            obstacle_radius=0.0,
            safety_margin=0.0,
        )
        car = _UnstableLanePIDObstacleCar(edge_target)
        manager._fleet = _FakeFleet([car])
        manager._sample_episode_targets = lambda agent_info, count: [edge_target]

        active = manager._refresh_obstacle_layout(
            agent_info={
                "pos": (0.0, 0.0, 0.0),
                "car": (0.0, 0.0, 0.0),
                "speed": 0.0,
                "cte": 0.0,
                "hit": "none",
            }
        )

        self.assertFalse(active)
        self.assertGreaterEqual(len(car.start_lane_pid_calls), 2)
        self.assertTrue(car.reset_calls)
        self.assertEqual(manager._fleet, None)
        self.assertEqual(len(car.place_pose_calls), 1)
        self.assertEqual(car.place_pose_calls[0]["world_y"], -500.0)
        for call in car.start_lane_pid_calls:
            self.assertEqual(call["lateral_ratio"], 0.0)
            self.assertEqual(call["obstacle_radius"], 0.0)
            self.assertEqual(call["safety_margin"], 0.0)
        self.assertEqual(manager._episode_modes_used, tuple())
        self.assertEqual(manager._episode_target_plan, tuple())

    def test_ws_lane_pid_layout_rejects_target_error_only_initial_pose(self):
        track_dir = str(Path("module/track_data").resolve())
        track_geometry = TrackGeometryManager(
            track_dir=track_dir,
            env_ids=v17.DEFAULT_ENV_IDS,
            scene_specs=v17.SCENE_SPECS,
        )
        manager = ObstacleRuntimeManager(
            track_geometry=track_geometry,
            conf={},
            track_dir=track_dir,
            config=ObstacleRuntimeConfig(
                ws_obstacle_count=1,
                ws_obstacle_modes=("lane_pid",),
                ws_lateral_choices=(1.0,),
                placement_timeout_s=0.1,
                seed=12,
            ),
        )
        manager.attach_scene(None, "donkey-waveshare-v0", "waveshare", "ws")
        edge_target = sample_track_target(
            track_geometry=track_geometry,
            scene_key="waveshare",
            progress_ratio=0.31,
            lateral_ratio=1.0,
            obstacle_radius=0.0,
            safety_margin=0.0,
        )
        car = _TargetErrorLanePIDObstacleCar(edge_target)
        manager._fleet = _FakeFleet([car])
        manager._sample_episode_targets = lambda agent_info, count: [edge_target]

        active = manager._refresh_obstacle_layout(
            agent_info={
                "pos": (0.0, 0.0, 0.0),
                "car": (0.0, 0.0, 0.0),
                "speed": 0.0,
                "cte": 0.0,
                "hit": "none",
            }
        )

        self.assertFalse(active)
        self.assertGreaterEqual(len(car.start_lane_pid_calls), 2)
        self.assertTrue(car.reset_calls)
        self.assertEqual(manager._fleet, None)
        for call in car.start_lane_pid_calls:
            self.assertEqual(call["lateral_ratio"], 1.0)
            self.assertEqual(call["obstacle_radius"], 0.0)
            self.assertEqual(call["safety_margin"], 0.0)

    def test_ws_edge_fallback_does_not_change_lateral(self):
        track_dir = str(Path("module/track_data").resolve())
        track_geometry = TrackGeometryManager(
            track_dir=track_dir,
            env_ids=v17.DEFAULT_ENV_IDS,
            scene_specs=v17.SCENE_SPECS,
        )
        manager = ObstacleRuntimeManager(
            track_geometry=track_geometry,
            conf={},
            track_dir=track_dir,
            config=ObstacleRuntimeConfig(ws_lateral_choices=(0.0, 1.0)),
        )
        manager.attach_scene(None, "donkey-waveshare-v0", "waveshare", "ws")
        target = sample_track_target(
            track_geometry=track_geometry,
            scene_key="waveshare",
            progress_ratio=0.61,
            lateral_ratio=0.0,
            obstacle_radius=0.0,
            safety_margin=0.0,
        )

        fallbacks = manager._ws_static_fallback_targets(
            target,
            obstacle_radius=0.0,
            safety_margin=0.0,
        )

        self.assertTrue(fallbacks)
        self.assertTrue(all(abs(candidate.lateral_ratio - 0.0) <= 1e-6 for candidate in fallbacks))

    def test_ws_free_episode_does_not_create_or_park_fleet(self):
        manager = ObstacleRuntimeManager(
            track_geometry=None,
            conf={},
            track_dir="",
            config=ObstacleRuntimeConfig(
                enabled=True,
                active_scene_keys=("waveshare",),
                ws_obstacle_free_prob=1.0,
                obstacle_free_prob=1.0,
                seed=3,
            ),
        )
        manager.attach_scene(None, "donkey-waveshare-v0", "waveshare", "ws")
        manager._observe_info_only = lambda: {
            "pos": (0.0, 0.0, -0.963),
            "car": (0.0, 0.0, 90.0),
            "speed": 0.0,
            "cte": 0.0,
            "hit": "none",
        }
        manager._observe_info_and_obs = lambda: (
            np.zeros((1,), dtype=np.float32),
            {
                "pos": (0.0, 0.0, -0.963),
                "car": (0.0, 0.0, 90.0),
                "speed": 0.0,
                "cte": 0.0,
                "hit": "none",
            },
        )
        ensure_calls = []
        park_calls = []
        manager._ensure_fleet = lambda: ensure_calls.append(True)
        manager._park_fleet = lambda reason="park": park_calls.append(reason)
        manager._log_reset_debug = lambda **kwargs: None

        manager.on_episode_reset(np.zeros((1,), dtype=np.float32))

        self.assertEqual(ensure_calls, [])
        self.assertEqual(park_calls, [])
        self.assertIsNone(manager._fleet)

    def test_ws_deactivation_shutdowns_existing_fleet_instead_of_parking(self):
        manager = ObstacleRuntimeManager(
            track_geometry=None,
            conf={},
            track_dir="",
            config=ObstacleRuntimeConfig(),
        )
        manager.attach_scene(None, "donkey-waveshare-v0", "waveshare", "ws")
        fleet = _FakeFleet([_FakeObstacleCar()])
        manager._fleet = fleet
        manager._fleet_scene_key = "waveshare"
        manager._park_fleet = lambda reason="park": (_ for _ in ()).throw(
            AssertionError("WS inactive deactivation should not park off-map")
        )

        manager._deactivate_inactive_fleet(reason="random_free_prob")

        self.assertEqual(fleet.shutdown_calls, 1)
        self.assertEqual(len(fleet.cars[0].place_pose_calls), 1)
        self.assertEqual(fleet.cars[0].place_pose_calls[0]["world_y"], -500.0)
        self.assertIsNone(manager._fleet)
        self.assertEqual(manager._fleet_scene_key, "")

    def test_gt_warmup_obstacle_samples_in_front_of_reset_pose(self):
        for phase_name in ("warmup", "warmup_a"):
            phase = v17.CURRICULUM_PHASES[phase_name]
            self.assertIsNone(phase.get("obstacle_fixed_progress_ratio"))
            self.assertIsNone(phase.get("obstacle_fixed_progress_distribution"))
            self.assertGreaterEqual(phase["obstacle_spawn_ahead_min_m"], 1.0)
            self.assertLessEqual(phase["obstacle_spawn_ahead_max_m"], 6.0)

        phase = v17.CURRICULUM_PHASES["warmup_a"]
        track_dir = str(Path("module/track_data").resolve())
        track_geometry = TrackGeometryManager(
            track_dir=track_dir,
            env_ids=v17.DEFAULT_ENV_IDS,
            scene_specs=v17.SCENE_SPECS,
        )
        config = ObstacleRuntimeConfig(
            obstacle_count=int(phase["obstacle_count"]),
            obstacle_free_prob=0.0,
            obstacle_modes=tuple(phase["obstacle_modes"]),
            spawn_ahead_min_m=float(phase["obstacle_spawn_ahead_min_m"]),
            spawn_ahead_max_m=float(phase["obstacle_spawn_ahead_max_m"]),
            min_agent_planar_dist_m=float(phase["obstacle_min_agent_planar_dist_m"]),
            min_agent_arc_dist_m=float(phase["obstacle_min_agent_arc_dist_m"]),
            lateral_choices=tuple(phase["obstacle_lateral_choices"]),
            fixed_progress_ratio=phase.get("obstacle_fixed_progress_ratio"),
            fixed_lateral_ratio=phase.get("obstacle_fixed_lateral_ratio"),
            seed=7,
        )
        manager = ObstacleRuntimeManager(track_geometry, {}, track_dir, config)
        manager.attach_scene(None, "donkey-generated-track-v0", "generated_track", "gt")

        agent_info = {
            "pos": (6.211, 0.0, 5.980),
            "car": (0.0, 0.0, 0.0),
            "speed": 0.0,
            "cte": 0.001,
            "hit": "none",
        }
        target = manager._sample_episode_targets(agent_info, 1)[0]
        agent_pose = pose_from_info(agent_info, track_geometry, "generated_track", None)
        obstacle_pose = pose_from_info(
            {
                "pos": (target.x, target.y, target.z),
                "car": (0.0, 0.0, target.yaw_deg),
                "speed": 0.0,
                "cte": 0.0,
                "hit": "none",
            },
            track_geometry,
            "generated_track",
            None,
        )
        relative = compute_relative_state(agent_pose, obstacle_pose)

        self.assertGreater(relative.longitudinal, 0.5)
        self.assertLess(relative.planar_distance, 6.5)

    def test_obstacle_park_waits_for_pose_confirmation(self):
        manager = ObstacleRuntimeManager(
            track_geometry=None,
            conf={},
            track_dir="",
            config=ObstacleRuntimeConfig(placement_timeout_s=1.5),
        )
        car = _FakeObstacleCar()
        manager._fleet = _FakeFleet([car])

        manager._park_fleet()

        self.assertEqual(
            car.place_pose_calls,
            [
                {
                    "x": -10.0,
                    "z": -6.0,
                    "yaw_deg": 0.0,
                    "world_y": 0.5,
                    "hold_brake": True,
                    "timeout_s": 1.5,
                }
            ],
        )
        self.assertEqual(car.teleport_pose_calls, [])
        self.assertEqual(car.stop_motion_calls, [{"hold_brake": True}])


if __name__ == "__main__":
    unittest.main()
