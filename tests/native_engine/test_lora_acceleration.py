import unittest

from h3serve.native_engine.planner import (
    H3LoraAccelerationScheduler,
    JointAccelerationError,
    JointWorkloadContext,
    LORA_NO_FORECAST_SCHEDULER_ID,
    verify_joint_plan_certificate,
)


class LoraAccelerationSchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scheduler = H3LoraAccelerationScheduler()
        self.workload = JointWorkloadContext(
            packed_tokens=25_000,
            condition_count=2,
            service_family="first_last",
            model_variant="lora",
        )

    def test_identity_is_explicit_and_forecast_is_structurally_disabled(self):
        self.assertEqual(
            self.scheduler.scheduler_id,
            LORA_NO_FORECAST_SCHEDULER_ID,
        )
        for steps in range(4, 11):
            plan = self.scheduler.plan(
                steps, 100, workload=self.workload
            )
            self.assertEqual(plan.actual_step_indices, tuple(range(steps)))
            self.assertEqual(plan.forecast_step_indices, ())
            self.assertFalse(plan.forecast_allowed)
            self.assertEqual(len(plan.physical_action_schedule()), steps * 50)
            self.assertTrue(verify_joint_plan_certificate(plan).valid)

    def test_acceleration_only_reallocates_attention_compute(self):
        plans = [
            self.scheduler.plan(8, value, workload=self.workload)
            for value in (0, 50, 100)
        ]
        self.assertFalse(plans[0].uses_sparse_attention)
        self.assertTrue(plans[1].uses_sparse_attention)
        self.assertTrue(plans[2].uses_sparse_attention)
        self.assertGreater(
            plans[0].estimated_compute_units,
            plans[1].estimated_compute_units,
        )
        self.assertGreaterEqual(
            plans[1].estimated_compute_units,
            plans[2].estimated_compute_units,
        )

    def test_rejects_base_workload_and_non_lora_step_count(self):
        with self.assertRaises(JointAccelerationError):
            self.scheduler.plan(
                8,
                50,
                workload=JointWorkloadContext(
                    packed_tokens=25_000,
                    model_variant="base",
                ),
            )
        for steps in (3, 11):
            with self.assertRaises(JointAccelerationError):
                self.scheduler.plan(steps, 50, workload=self.workload)


if __name__ == "__main__":
    unittest.main()
