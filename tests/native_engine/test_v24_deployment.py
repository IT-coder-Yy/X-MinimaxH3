from __future__ import annotations

import unittest

from h3serve.native_engine.planner import (
    V24_HUMAN_ANCHORS,
    V24ParetoRuntimeSelector,
    V19WorkloadContext,
    blueprint_from_runtime_schedule,
    v19_blueprint_execution_digest,
)


_SHAPES = (
    (1280, 736, 124, 34_780),
    (1280, 736, 243, 67_535),
    (1280, 736, 362, 100_141),
    (1920, 1088, 362, 219_890),
)


def _workload(
    width: int,
    height: int,
    frames: int,
    packed_tokens: int,
    *,
    steps: int = 20,
    service_family: str = "first_last",
    condition_count: int = 0,
    reference_images: int = 0,
    reference_audio: int = 0,
    reference_videos: int = 0,
) -> V19WorkloadContext:
    return V19WorkloadContext(
        model_variant="base",
        service_family=service_family,
        packed_tokens=packed_tokens,
        condition_count=condition_count,
        reference_images=reference_images,
        reference_audio=reference_audio,
        reference_videos=reference_videos,
        width=width,
        height=height,
        frames=frames,
        steps=steps,
        actual_step_indices=tuple(range(steps)),
        sampler="res_multistep",
        scheduler="simple",
    )


def _rank(action: str) -> int:
    return {
        "sparse_topk_0.0625": 0,
        "sparse_topk_0.1": 1,
        "sparse_topk_0.25": 2,
        "sparse_topk_0.5": 3,
        "dense": 4,
    }[action.rsplit(":", 1)[-1]]


class V24ParetoRuntimeSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.selector = V24ParetoRuntimeSelector()

    def test_strength_100_reproduces_every_reviewed_physical_blueprint(self) -> None:
        for anchor, shape in zip(V24_HUMAN_ANCHORS, _SHAPES):
            with self.subTest(anchor=anchor.anchor_id):
                selected = self.selector.select(
                    workload=_workload(*shape), acceleration=100
                )
                blueprint = blueprint_from_runtime_schedule(
                    candidate_id="v24_digest_replay",
                    total_steps=20,
                    actual_step_indices=selected.actual_step_indices,
                    attention_action_schedule=selected.attention_action_schedule,
                    source="v24_test",
                )
                self.assertEqual(
                    v19_blueprint_execution_digest(blueprint),
                    anchor.source_execution_digest,
                )
                self.assertEqual(
                    selected.summary["execution_digest"],
                    anchor.source_execution_digest,
                )
                self.assertTrue(
                    selected.summary["calibration_surface"]["direct_human_anchor"]
                )

    def test_acceleration_chain_is_compute_monotone_and_never_more_approximate(self) -> None:
        for shape in _SHAPES:
            previous = self.selector.select(
                workload=_workload(*shape), acceleration=0
            )
            previous_cost = float(previous.summary["estimated_compute_ratio"])
            for acceleration in range(1, 101):
                selected = self.selector.select(
                    workload=_workload(*shape), acceleration=acceleration
                )
                cost = float(selected.summary["estimated_compute_ratio"])
                self.assertLessEqual(cost, previous_cost + 1.0e-12)
                self.assertTrue(
                    set(selected.actual_step_indices).issubset(
                        previous.actual_step_indices
                    )
                )
                previous_actions = {
                    (step, layer): action
                    for step, layer, action in previous.attention_action_schedule
                }
                selected_actions = {
                    (step, layer): action
                    for step, layer, action in selected.attention_action_schedule
                }
                if previous_actions:
                    for step in selected.actual_step_indices:
                        for layer in range(50):
                            self.assertLessEqual(
                                _rank(selected_actions[(step, layer)]),
                                _rank(previous_actions[(step, layer)]),
                            )
                self.assertLessEqual(selected.summary["maximum_forecast_run"], 3)
                previous = selected
                previous_cost = cost

    def test_zero_is_the_unique_dense_endpoint(self) -> None:
        selected = self.selector.select(
            workload=_workload(*_SHAPES[2]), acceleration=0
        )
        self.assertEqual(selected.actual_step_indices, tuple(range(20)))
        self.assertEqual(selected.attention_action_schedule, ())
        self.assertEqual(selected.summary["estimated_compute_ratio"], 1.0)
        self.assertIsNone(
            selected.summary["runtime_feedback"]["policy_id"]
        )

    def test_one_joint_chain_couples_trajectory_and_attention_for_all_strengths(self) -> None:
        selected = [
            self.selector.select(
                workload=_workload(*_SHAPES[1]), acceleration=value
            )
            for value in (25, 50, 75, 100)
        ]
        optimizers = [item.summary["optimizer"] for item in selected]
        self.assertEqual(
            {item["chain_digest"] for item in optimizers},
            {optimizers[0]["chain_digest"]},
        )
        self.assertGreater(
            optimizers[0]["chain_upgrade_counts"]["attention_fidelity"], 0
        )
        self.assertGreater(
            optimizers[0]["chain_upgrade_counts"]["forecast_to_actual"], 0
        )
        self.assertEqual(
            selected[-1].summary["runtime_feedback"],
            {
                "policy_id": "v24_request_local_forecast_debt_v1",
                "mode": "observe_only",
                "signal": (
                    "request-local secant-tail error "
                    "at already-computed Actual corrections"
                ),
                "adds_teacher_evaluations": False,
                "max_runtime_promotions": 0,
            },
        )

    def test_arbitrary_step_counts_compile_complete_physical_schedules(self) -> None:
        for steps in (5, 8, 12, 17, 20, 24, 30):
            with self.subTest(steps=steps):
                selected = self.selector.select(
                    workload=_workload(*_SHAPES[1], steps=steps),
                    acceleration=83,
                )
                actual = set(selected.actual_step_indices)
                cells = {
                    (step, layer)
                    for step, layer, _action in selected.attention_action_schedule
                }
                expected = {
                    (step, layer)
                    for step in actual
                    for layer in range(50)
                } | {
                    (step, layer)
                    for step in range(steps)
                    if step not in actual
                    for layer in range(3)
                }
                self.assertEqual(cells, expected)
                self.assertLessEqual(selected.summary["maximum_forecast_run"], 3)

    def test_prompt_length_inside_normal_prefix_band_does_not_change_schedule(self) -> None:
        first = self.selector.select(
            workload=_workload(1280, 736, 243, 67_000),
            acceleration=75,
        )
        second = self.selector.select(
            workload=_workload(1280, 736, 243, 69_000),
            acceleration=75,
        )
        self.assertEqual(first.actual_step_indices, second.actual_step_indices)
        self.assertEqual(
            first.attention_action_schedule,
            second.attention_action_schedule,
        )
        self.assertFalse(first.summary["prompt_semantics_used"])

    def test_reference_video_guard_disables_forecast_without_disabling_attention(self) -> None:
        selected = self.selector.select(
            workload=_workload(
                1280,
                736,
                243,
                110_000,
                service_family="reference",
                condition_count=1,
                reference_videos=1,
            ),
            acceleration=100,
        )
        self.assertEqual(selected.actual_step_indices, tuple(range(20)))
        self.assertIn(
            "reference_video_no_forecast_guard", selected.summary["safety_guards"]
        )
        self.assertIn(
            "block_sparse_attention",
            selected.summary["technique_mix"]["coupled_techniques"],
        )

    def test_required_preview_step_is_promoted_to_dense_actual(self) -> None:
        selected = self.selector.select(
            workload=_workload(*_SHAPES[1]),
            acceleration=100,
            required_actual_step_indices=(14,),
        )
        self.assertIn(14, selected.actual_step_indices)
        actions = {
            (step, layer): action
            for step, layer, action in selected.attention_action_schedule
        }
        self.assertEqual(
            {actions[(14, layer)] for layer in range(50)}, {"dense"}
        )
        self.assertFalse(
            selected.summary["calibration_surface"]["direct_human_anchor"]
        )

    def test_unsupported_packed_envelope_fails_closed_to_dense(self) -> None:
        selected = self.selector.select(
            workload=_workload(2560, 1440, 362, 410_000),
            acceleration=100,
        )
        self.assertEqual(selected.actual_step_indices, tuple(range(20)))
        self.assertEqual(selected.attention_action_schedule, ())
        self.assertEqual(
            selected.summary["reason"],
            "packed_token_envelope_exceeded_dense_fallback",
        )

    def test_2k_shape_uses_xlong_curve_but_discloses_evidence_extrapolation(self) -> None:
        selected = self.selector.select(
            workload=_workload(2560, 1440, 362, 386_923),
            acceleration=95,
        )
        self.assertTrue(selected.summary["accelerated"])
        self.assertIn(
            "xlong_anchor_shape_extrapolation",
            selected.summary["safety_guards"],
        )
        surface = selected.summary["calibration_surface"]
        self.assertFalse(surface["direct_human_anchor"])
        self.assertTrue(surface["shape_extrapolated_beyond_human_evidence"])
        self.assertEqual(surface["upper_anchor_id"], "xlong_218k_v012_v018_exact")

    def test_only_direct_medium_anchor_requests_byte_exact_helper_stack(self) -> None:
        medium = self.selector.select(
            workload=_workload(*_SHAPES[1]), acceleration=100
        )
        medium_partial = self.selector.select(
            workload=_workload(*_SHAPES[1]), acceleration=99
        )
        long = self.selector.select(
            workload=_workload(*_SHAPES[2]), acceleration=100
        )
        xlong = self.selector.select(
            workload=_workload(*_SHAPES[3]), acceleration=100
        )
        self.assertEqual(
            medium.summary["execution_profile_hint"],
            "v22_medium_byte_exact_helpers",
        )
        self.assertIsNone(medium_partial.summary["execution_profile_hint"])
        self.assertIsNone(long.summary["execution_profile_hint"])
        self.assertEqual(
            xlong.summary["execution_profile_hint"],
            "v18_xlong_byte_exact_helpers",
        )


if __name__ == "__main__":
    unittest.main()
