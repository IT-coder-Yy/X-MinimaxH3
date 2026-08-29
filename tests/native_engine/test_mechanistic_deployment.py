import json
from pathlib import Path
import tempfile
import unittest

from h3serve.native_engine.planner import (
    H3MechanisticAdmission,
    H3MechanisticParetoRuntimeSelector,
    MECHANISTIC_ADMISSION_SCHEMA,
    MECHANISTIC_DEPLOYMENT_POLICY_ID,
    V19PlanningError,
    V19WorkloadContext,
    load_h3_mechanistic_deployment_config,
)


class H3MechanisticDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Deliberately synthetic: production must load a Human-reviewed
        # workload-scoped admission artifact instead of this permissive gate.
        cls.admission = H3MechanisticAdmission(
            calibration_id="unit_test_only",
            maximum_modeled_risk=100.0,
            evidence_ids=("synthetic_unit_test",),
        )
        cls.selector = H3MechanisticParetoRuntimeSelector(
            admission=cls.admission,
            calibrated_video_token_interval=(34_040, 34_040),
            maximum_runtime_promotions=1,
        )
        cls.workload = V19WorkloadContext(
            model_variant="base",
            service_family="first_last",
            packed_tokens=34_871,
            condition_count=2,
            width=1280,
            height=736,
            frames=124,
            steps=20,
            sampler="res_multistep",
            scheduler="simple",
        )
        cls.selection = cls.selector.select(
            workload=cls.workload,
            acceleration=50.0,
        )

    def test_public_dial_compiles_one_joint_mechanistic_plan(self) -> None:
        summary = self.selection.summary
        self.assertEqual(summary["policy_id"], MECHANISTIC_DEPLOYMENT_POLICY_ID)
        self.assertEqual(summary["acceleration"], 50.0)
        self.assertEqual(summary["video_tokens"], 34_040)
        self.assertFalse(summary["prompt_semantics_used"])
        self.assertFalse(summary["historical_schedule_used"])
        self.assertTrue(summary["accelerated"])
        self.assertTrue(self.selection.attention_action_schedule)
        self.assertEqual(
            self.selection.actual_step_indices,
            tuple(sorted(set(self.selection.actual_step_indices))),
        )
        self.assertTrue(summary["certificate"]["exact_for_declared_lagrangian_problem"])

    def test_runtime_recovery_is_paid_from_the_public_compute_budget(self) -> None:
        summary = self.selection.summary
        feedback = summary["runtime_feedback"]
        self.assertIsNotNone(self.selection.runtime_controller)
        self.assertEqual(feedback["mode"], "mechanistic_risk_reserve")
        self.assertEqual(feedback["max_runtime_promotions"], 1)
        self.assertGreater(feedback["recovery_reserve_ms"], 0.0)
        self.assertLessEqual(
            summary["reserved_total_cost_ms"],
            summary["public_target_cost_ms"] + 1.0e-8,
        )
        self.assertFalse(feedback["adds_teacher_evaluations"])

    def test_outside_admission_geometry_fails_closed_to_dense(self) -> None:
        workload = V19WorkloadContext(
            model_variant="base",
            service_family="first_last",
            packed_tokens=18_000,
            condition_count=2,
            width=640,
            height=736,
            frames=124,
            steps=20,
        )
        selection = self.selector.select(workload=workload, acceleration=100.0)
        self.assertEqual(selection.actual_step_indices, tuple(range(20)))
        self.assertFalse(selection.attention_action_schedule)
        self.assertIsNone(selection.runtime_controller)
        self.assertFalse(selection.summary["accelerated"])
        self.assertEqual(
            selection.summary["reason"],
            "outside_mechanistic_admission_scope_dense_fallback",
        )

    def test_invalid_h3_geometry_is_rejected_before_planning(self) -> None:
        workload = V19WorkloadContext(
            model_variant="base",
            service_family="first_last",
            packed_tokens=34_871,
            condition_count=2,
            width=1280,
            height=736,
            frames=125,
            steps=20,
        )
        with self.assertRaises(V19PlanningError):
            self.selector.select(workload=workload, acceleration=50.0)

    def test_admission_artifact_contains_a_boundary_not_a_schedule(self) -> None:
        document = {
            "schema_version": MECHANISTIC_ADMISSION_SCHEMA,
            "status": "release",
            "calibration_id": "reviewed_holdout_v1",
            "maximum_modeled_risk": 1.25,
            "evidence_ids": ["calibration_review"],
            "held_out_evidence_ids": ["held_out_review"],
            "calibrated_video_token_interval": [34_040, 98_440],
            "maximum_runtime_promotions": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "admission.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            loaded = load_h3_mechanistic_deployment_config(path)
            self.assertEqual(loaded.status, "release")
            self.assertEqual(
                loaded.calibrated_video_token_interval,
                (34_040, 98_440),
            )
            self.assertEqual(len(loaded.source_sha256), 64)
            document["actual_step_indices"] = [0, 1, 4, 8, 19]
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "forbidden/unknown"):
                load_h3_mechanistic_deployment_config(path)


if __name__ == "__main__":
    unittest.main()
