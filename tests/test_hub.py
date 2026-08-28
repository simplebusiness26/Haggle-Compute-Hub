import unittest

import hub


class ComputeHubTests(unittest.TestCase):
    def setUp(self):
        self.catalog = {
            "families": {
                "model_lab": {"default_compute": "gpu", "default_value": 75},
                "dataset": {"default_compute": "cpu", "default_value": 65},
            },
            "priorities": {"low": 0.75, "normal": 1.0, "high": 1.5, "urgent": 2.0},
        }

    def test_high_value_short_job_scores_above_long_low_value_job(self):
        a = {
            "priority": "high",
            "value": 90,
            "estimated_minutes": 15,
            "deadline": None,
        }
        b = {
            "priority": "normal",
            "value": 40,
            "estimated_minutes": 120,
            "deadline": None,
        }
        self.assertGreater(hub.scheduler_score(a, self.catalog), hub.scheduler_score(b, self.catalog))

    def test_validate_rejects_arbitrary_kernel_path(self):
        with self.assertRaises(ValueError):
            hub.validate_job(
                {
                    "title": "Bad path",
                    "family": "model_lab",
                    "compute": "gpu",
                    "kernel_path": "../../tmp",
                    "estimated_minutes": 10,
                },
                "test-001",
                self.catalog,
            )

    def test_cpu_job_needs_no_gpu_budget(self):
        job = {"compute": "cpu", "estimated_minutes": 500, "priority": "normal"}
        resources = {
            "configured_gpu_hours_per_week": 30,
            "estimated_gpu_hours_used": 30,
            "reserve_fraction": 0.2,
            "resources": {"cpu": {"enabled": True}},
        }
        ok, _ = hub.can_run(job, resources)
        self.assertTrue(ok)

    def test_gpu_job_held_when_budget_exhausted(self):
        job = {"compute": "gpu", "estimated_minutes": 60, "priority": "normal"}
        resources = {
            "configured_gpu_hours_per_week": 30,
            "estimated_gpu_hours_used": 25,
            "reserve_fraction": 0.2,
            "resources": {"gpu": {"enabled": True}},
        }
        ok, _ = hub.can_run(job, resources)
        self.assertFalse(ok)

    def test_urgent_can_use_reserve_but_not_exceed_total(self):
        resources = {
            "configured_gpu_hours_per_week": 30,
            "estimated_gpu_hours_used": 25,
            "reserve_fraction": 0.2,
            "resources": {"gpu": {"enabled": True}},
        }
        ok, _ = hub.can_run({"compute": "gpu", "estimated_minutes": 180, "priority": "urgent"}, resources)
        self.assertTrue(ok)
        ok, _ = hub.can_run({"compute": "gpu", "estimated_minutes": 360, "priority": "urgent"}, resources)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
