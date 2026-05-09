import unittest

from src import ppo_multitrack_v18 as v18


class V18MultiSimTests(unittest.TestCase):
    def test_default_ports_create_two_balanced_scene_replicas(self):
        assignments = v18.build_v18_worker_assignments(
            env_ids=v18.DEFAULT_ENV_IDS,
            ports=v18.DEFAULT_V18_PORTS,
        )

        self.assertEqual([a["port"] for a in assignments], [9091, 9093, 9095, 9097])
        self.assertEqual(
            [a["env_id"] for a in assignments],
            [
                "donkey-waveshare-v0",
                "donkey-generated-track-v0",
                "donkey-waveshare-v0",
                "donkey-generated-track-v0",
            ],
        )
        self.assertEqual([a["logging_key"] for a in assignments], ["ws", "gt", "ws", "gt"])
        self.assertEqual([a["scene_weight"] for a in assignments], [1.0, 1.0, 1.0, 1.0])

    def test_worker_ports_must_be_unique_and_balanced_by_scene(self):
        with self.assertRaises(ValueError):
            v18.build_v18_worker_assignments(
                env_ids=v18.DEFAULT_ENV_IDS,
                ports=[9091, 9091, 9095, 9097],
            )

        with self.assertRaises(ValueError):
            v18.build_v18_worker_assignments(
                env_ids=v18.DEFAULT_ENV_IDS,
                ports=[9091, 9093, 9095],
            )

    def test_parallel_vec_env_is_selected_for_multiple_workers(self):
        assignments = v18.build_v18_worker_assignments(
            env_ids=v18.DEFAULT_ENV_IDS,
            ports=v18.DEFAULT_V18_PORTS,
        )

        self.assertEqual(v18._v18_vec_env_mode(assignments), "subproc")
        self.assertEqual(v18._v18_vec_env_mode(assignments[:1]), "dummy")


if __name__ == "__main__":
    unittest.main()
