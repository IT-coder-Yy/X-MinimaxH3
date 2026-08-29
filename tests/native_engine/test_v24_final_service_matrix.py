from __future__ import annotations

import unittest

from h3serve.contract import GenerationSpec
from h3serve.native_engine.planner import (
    H3LoraAccelerationScheduler,
    JointWorkloadContext,
    V24FinalParetoRuntimeSelector,
    V19WorkloadContext,
)


class V24FinalServiceMatrixTests(unittest.TestCase):
    def test_all_four_public_routes_accept_the_1080p15_boundary(self) -> None:
        expected = {
            "original": ("first_last", "base"),
            "lora": ("first_last", "lora"),
            "reference": ("reference", "base"),
            "reference_lora": ("reference", "lora"),
        }
        for engine, identity in expected.items():
            with self.subTest(engine=engine):
                spec = GenerationSpec.from_mapping({
                    "prompt": "route-boundary contract fixture",
                    "engine": engine,
                    "resolution": "1080p",
                    "aspect_ratio": "16:9",
                    "duration_seconds": 15,
                })
                self.assertEqual((spec.service_family, spec.model_variant), identity)
                self.assertEqual(
                    (spec.width, spec.height, spec.frames),
                    (1920, 1088, 362),
                )

    def test_ref2va_base_static_media_uses_the_same_continuous_surface(self) -> None:
        workload = V19WorkloadContext(
            model_variant="base",
            service_family="reference",
            packed_tokens=245_000,
            condition_count=12,
            reference_images=9,
            reference_audio=3,
            reference_videos=0,
            width=1920,
            height=1088,
            frames=362,
            steps=20,
            actual_step_indices=tuple(range(20)),
            sampler="res_multistep",
            scheduler="simple",
        )
        selected = V24FinalParetoRuntimeSelector().select(
            workload=workload,
            acceleration=75,
        )
        self.assertTrue(selected.summary["accelerated"])
        self.assertEqual(selected.summary["prompt_semantics_used"], False)
        self.assertIn("reference_layout_guard", selected.summary["safety_guards"])
        self.assertIn("reference_media_guard", selected.summary["safety_guards"])
        self.assertNotEqual(
            selected.summary["reason"],
            "packed_token_envelope_exceeded_dense_fallback",
        )

    def test_both_lora_service_families_keep_the_distilled_trajectory(self) -> None:
        scheduler = H3LoraAccelerationScheduler()
        for family in ("first_last", "reference"):
            with self.subTest(family=family):
                plan = scheduler.plan(
                    10,
                    75,
                    workload=JointWorkloadContext(
                        packed_tokens=245_000,
                        condition_count=12 if family == "reference" else 0,
                        service_family=family,
                        model_variant="lora",
                    ),
                )
                self.assertEqual(plan.actual_step_indices, tuple(range(10)))
                self.assertEqual(plan.forecast_step_indices, ())
                self.assertEqual(len(plan.physical_action_schedule()), 500)


if __name__ == "__main__":
    unittest.main()
