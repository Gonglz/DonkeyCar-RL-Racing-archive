import unittest
from pathlib import Path

from module.obstacle import PoseState, compute_relative_state, pose_from_info, sample_track_target
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


class _FakeFleet:
    preset = _FakePreset()

    def __init__(self, cars):
        self.cars = cars


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
