from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest

from h3serve.native_engine.planner import (
    ROUND188_REVIEWED_20_STEP_ACTUALS,
    V19ExperimentalLongRuntimeSelector,
    V19WorkloadContext,
    classify_v19_long_runtime_workload,
)


EXPECTED_BATCH07_EXECUTION_DIGEST = (
    "495c2c8ff75f76aed16b1fd81a41f3e050df5bad38456dd68d0806c3e9c7cbad"
)


def _workload(
    *,
    width: int = 1920,
    height: int = 1088,
    frames: int = 362,
    packed_tokens: int = 219_890,
    steps: int = 20,
) -> V19WorkloadContext:
    return V19WorkloadContext(
        model_variant="base",
        service_family="first_last",
        packed_tokens=packed_tokens,
        condition_count=0,
        width=width,
        height=height,
        frames=frames,
        steps=steps,
        actual_step_indices=tuple(range(steps)),
    )


class _RecordingReleaseSelector:
    def __init__(self) -> None:
        self.calls = []

    def select(
        self,
        *,
        workload,
        acceleration,
        required_actual_step_indices=(),
    ):
        self.calls.append((workload, acceleration, required_actual_step_indices))
        return SimpleNamespace(
            actual_step_indices=tuple(range(int(workload.steps))),
            attention_action_schedule=(),
            summary={"reason": "certified_release_delegate"},
        )


class V19LongRuntimeTests(unittest.TestCase):
    def test_all_measured_long_video_geometries_are_admitted(self) -> None:
        cases = (
            (_workload(), "experimental_1080p15_base_no_reference_v1"),
            (
                _workload(
                    width=1280,
                    height=736,
                    packed_tokens=100_141,
                ),
                "experimental_720p15_base_no_reference_v1",
            ),
            (
                _workload(
                    width=1280,
                    height=736,
                    frames=243,
                    packed_tokens=67_535,
                ),
                "experimental_720p10_base_no_reference_v1",
            ),
        )
        for workload, envelope_id in cases:
            with self.subTest(envelope_id=envelope_id):
                admission = classify_v19_long_runtime_workload(
                    workload,
                    acceleration=75.0,
                )
                self.assertTrue(admission.admitted)
                self.assertEqual(admission.envelope_id, envelope_id)

    def test_selection_replays_exact_batch07_physical_execution(self) -> None:
        selection = V19ExperimentalLongRuntimeSelector().select(
            workload=_workload(),
            acceleration=75.0,
        )
        self.assertEqual(
            selection.actual_step_indices,
            ROUND188_REVIEWED_20_STEP_ACTUALS,
        )
        self.assertEqual(len(selection.attention_action_schedule), 624)
        self.assertEqual(
            selection.summary["execution_digest"],
            EXPECTED_BATCH07_EXECUTION_DIGEST,
        )
        self.assertTrue(selection.summary["proposal_eligible"])
        self.assertFalse(selection.summary["release_eligible"])
        self.assertEqual(
            selection.summary["technique_mix"]["actual_attention_cells"],
            {"sparse_topk_0.0625": 420, "sparse_topk_0.1": 180},
        )
        self.assertEqual(
            selection.summary["technique_mix"][
                "forecast_anchor_attention_cells"
            ],
            {"sparse_topk_0.0625": 24},
        )

    def test_720p10_envelope_selects_the_measured_v012_execution(self) -> None:
        selection = V19ExperimentalLongRuntimeSelector().select(
            workload=_workload(
                width=1280,
                height=736,
                frames=243,
                packed_tokens=67_535,
            ),
            acceleration=75.0,
        )
        self.assertEqual(
            selection.actual_step_indices,
            ROUND188_REVIEWED_20_STEP_ACTUALS,
        )
        self.assertEqual(
            selection.summary["execution_digest"],
            EXPECTED_BATCH07_EXECUTION_DIGEST,
        )
        self.assertEqual(
            selection.summary["envelope_id"],
            "experimental_720p10_base_no_reference_v1",
        )
        self.assertIn(
            "batch09_v012_speedup_1p250012",
            selection.summary["evidence_ids"],
        )
        self.assertFalse(selection.summary["release_eligible"])

    def test_prompt_content_cannot_change_the_route(self) -> None:
        # Prompt text/seed are not selector inputs.  Normal exact-token length
        # variation inside the measured envelope preserves one physical plan.
        selector = V19ExperimentalLongRuntimeSelector()
        short_text = selector.select(
            workload=_workload(packed_tokens=218_200),
            acceleration=75.0,
        )
        long_text = selector.select(
            workload=_workload(packed_tokens=220_300),
            acceleration=75.0,
        )
        self.assertEqual(
            short_text.attention_action_schedule,
            long_text.attention_action_schedule,
        )
        self.assertEqual(
            short_text.summary["execution_digest"],
            long_text.summary["execution_digest"],
        )

    def test_strength_above_75_is_clamped_to_human_evidence(self) -> None:
        selector = V19ExperimentalLongRuntimeSelector()
        strength75 = selector.select(workload=_workload(), acceleration=75.0)
        strength95 = selector.select(workload=_workload(), acceleration=95.0)
        self.assertEqual(
            strength75.attention_action_schedule,
            strength95.attention_action_schedule,
        )
        self.assertFalse(
            strength75.summary["acceleration_clamped_to_human_evidence"]
        )
        self.assertTrue(
            strength95.summary["acceleration_clamped_to_human_evidence"]
        )
        self.assertEqual(strength95.summary["effective_acceleration"], 75.0)

    def test_short_and_unmeasured_requests_delegate_without_mutation(self) -> None:
        release = _RecordingReleaseSelector()
        selector = V19ExperimentalLongRuntimeSelector(release)
        workloads = (
            _workload(frames=124, packed_tokens=70_000),
            _workload(frames=243, packed_tokens=75_000),
            _workload(steps=18, packed_tokens=219_890),
            replace(
                _workload(),
                condition_count=1,
                reference_images=1,
                packed_tokens=221_930,
            ),
        )
        for workload in workloads:
            with self.subTest(workload=workload):
                selected = selector.select(
                    workload=workload,
                    acceleration=75.0,
                )
                self.assertEqual(selected.summary["reason"], "certified_release_delegate")
        self.assertEqual(len(release.calls), len(workloads))

    def test_below_floor_and_unreviewed_preview_anchor_fail_closed(self) -> None:
        no_release = V19ExperimentalLongRuntimeSelector()
        below = no_release.select(workload=_workload(), acceleration=50.0)
        self.assertEqual(below.actual_step_indices, tuple(range(20)))
        self.assertEqual(below.attention_action_schedule, ())
        self.assertFalse(below.summary["accelerated"])

        preview = no_release.select(
            workload=_workload(),
            acceleration=75.0,
            required_actual_step_indices=(15,),
        )
        self.assertEqual(preview.actual_step_indices, tuple(range(20)))
        self.assertEqual(preview.attention_action_schedule, ())
        self.assertIn("preview_anchor_unmeasured", preview.summary["reason"])


if __name__ == "__main__":
    unittest.main()
