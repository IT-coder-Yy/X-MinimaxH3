from __future__ import annotations

import json
from pathlib import Path
import unittest

from h3serve.native_engine.planner import (
    V24_FINAL_DEFAULT_CANDIDATE,
    V24_FINAL_POLICY_ID,
    V24_FINAL_RELEASE_CANDIDATES,
    V24FinalParetoRuntimeSelector,
    V24ResearchParetoRuntimeSelector,
    V19PlanningError,
    V19WorkloadContext,
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


class V24FinalParetoRuntimeSelectorTests(unittest.TestCase):
    def test_2k_runtime_surface_discloses_xlong_evidence_extrapolation(self) -> None:
        selected = V24FinalParetoRuntimeSelector().select(
            workload=_workload(2560, 1440, 362, 386_482),
            acceleration=95,
        )
        self.assertTrue(selected.summary["accelerated"])
        self.assertIn(
            "xlong_anchor_shape_extrapolation",
            selected.summary["safety_guards"],
        )
        surface = selected.summary["calibration_surface"]
        self.assertFalse(surface["direct_human_quality_knee"])
        self.assertTrue(surface["shape_extrapolated_beyond_human_evidence"])

    def test_release_curve_summary_replays_the_selector(self) -> None:
        root = Path(__file__).resolve().parents[2]
        document = json.loads((
            root / "tests/fixtures/v24_curve_summary_c02.json"
        ).read_text(encoding="utf-8"))
        selector = V24ResearchParetoRuntimeSelector(
            candidate_id=document["candidate_id"]
        )
        for workload in document["workloads"]:
            width, height, frames = workload["geometry"]
            context = _workload(
                width,
                height,
                frames,
                workload["packed_tokens"],
            )
            for point in workload["points"]:
                with self.subTest(
                    workload=workload["name"],
                    acceleration=point["acceleration"],
                ):
                    selected = selector.select(
                        workload=context,
                        acceleration=point["acceleration"],
                    )
                    self.assertEqual(
                        len(selected.actual_step_indices), point["actual"]
                    )
                    self.assertEqual(
                        20 - len(selected.actual_step_indices), point["forecast"]
                    )
                    self.assertAlmostEqual(
                        selected.summary["estimated_compute_units"],
                        point["compute_units"],
                        places=4,
                    )
                    self.assertEqual(
                        selected.summary.get("maximum_forecast_run", 0),
                        point["max_forecast_run"],
                    )

    def test_default_is_the_human_selected_c02_release_anchor(self) -> None:
        selector = V24FinalParetoRuntimeSelector()
        self.assertEqual(selector.candidate.candidate_id, V24_FINAL_DEFAULT_CANDIDATE)
        self.assertEqual(selector.candidate.candidate_id, "v24_final_c02_round2_trajectory_u7p00")
        self.assertEqual(selector.candidate.human_status, "accepted_release_default")
        self.assertEqual(selector.policy_id, V24_FINAL_POLICY_ID)

    def test_acceleration_35_reaches_a_material_low_control_saving(self) -> None:
        selected = V24FinalParetoRuntimeSelector().select(
            workload=_workload(*_SHAPES[2]), acceleration=35
        )
        self.assertEqual(len(selected.actual_step_indices), 13)
        self.assertLessEqual(
            selected.summary["estimated_compute_units"],
            14.27 / 1.10,
        )
        actions = {
            action.rsplit(":", 1)[-1]
            for _step, _layer, action in selected.attention_action_schedule
        }
        self.assertIn("sparse_topk_0.5", actions)

    def test_production_selector_cannot_switch_calibration(self) -> None:
        with self.assertRaises(TypeError):
            V24FinalParetoRuntimeSelector(candidate_id="v009")

        for alias in ("stable", "v009", "c01", "c02", "c03"):
            with self.subTest(alias=alias):
                with self.assertRaises(V19PlanningError):
                    V24ResearchParetoRuntimeSelector(candidate_id=alias)

    def test_unknown_candidate_fails_closed(self) -> None:
        with self.assertRaises(V19PlanningError):
            V24ResearchParetoRuntimeSelector(candidate_id="not-a-candidate")

    def test_acceleration_75_replays_shared_knees_and_final_c02_long_knee(self) -> None:
        expected = (
            "c48c8b25ea97641100d07cc826ee336ebc8ecc77435ad9f80082aa91aca3abc9",
            "f279d44e88e798c1c273329268e36af020795a1bb2e6b54459da65a103488cae",
            "46105dcb98bb3375df161647627303f95ce0f2c333e66ec6bd8412c6a150ff4e",
            "495c2c8ff75f76aed16b1fd81a41f3e050df5bad38456dd68d0806c3e9c7cbad",
        )
        selector = V24FinalParetoRuntimeSelector()
        for shape, digest in zip(_SHAPES, expected):
            with self.subTest(shape=shape):
                selected = selector.select(
                    workload=_workload(*shape), acceleration=75
                )
                self.assertEqual(selected.summary["execution_digest"], digest)
                self.assertEqual(
                    selected.summary["reason"], "direct_human_quality_knee"
                )

    def test_each_long_candidate_is_an_exact_75_knot(self) -> None:
        for candidate_id, candidate in V24_FINAL_RELEASE_CANDIDATES.items():
            with self.subTest(candidate=candidate_id):
                selected = V24ResearchParetoRuntimeSelector(
                    candidate_id=candidate_id
                ).select(workload=_workload(*_SHAPES[2]), acceleration=75)
                self.assertEqual(
                    selected.summary["execution_digest"],
                    candidate.long_anchor.source_execution_digest,
                )

    def test_curve_is_nested_and_compute_monotone(self) -> None:
        selector = V24FinalParetoRuntimeSelector()
        previous = selector.select(
            workload=_workload(*_SHAPES[2]), acceleration=0
        )
        for acceleration in range(1, 101):
            current = selector.select(
                workload=_workload(*_SHAPES[2]),
                acceleration=acceleration,
            )
            self.assertLessEqual(
                current.summary["estimated_compute_units"],
                previous.summary["estimated_compute_units"] + 1.0e-12,
            )
            self.assertTrue(
                set(current.actual_step_indices).issubset(
                    previous.actual_step_indices
                )
            )
            before = {
                (step, layer): action
                for step, layer, action in previous.attention_action_schedule
            }
            after = {
                (step, layer): action
                for step, layer, action in current.attention_action_schedule
            }
            if before:
                for step in current.actual_step_indices:
                    for layer in range(50):
                        self.assertLessEqual(
                            _rank(after[(step, layer)]),
                            _rank(before[(step, layer)]),
                        )
            self.assertLessEqual(current.summary["maximum_forecast_run"], 4)
            previous = current

    def test_100_is_faster_than_the_quality_knee(self) -> None:
        selector = V24FinalParetoRuntimeSelector()
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                knee = selector.select(
                    workload=_workload(*shape), acceleration=75
                )
                fast = selector.select(
                    workload=_workload(*shape), acceleration=100
                )
                self.assertLess(
                    fast.summary["estimated_compute_units"],
                    knee.summary["estimated_compute_units"],
                )
                self.assertEqual(
                    fast.summary["curve_segment"],
                    "human_quality_knee_to_aggressive_endpoint",
                )
        xlong_knee = selector.select(
            workload=_workload(*_SHAPES[3]), acceleration=75
        )
        xlong_fast = selector.select(
            workload=_workload(*_SHAPES[3]), acceleration=100
        )
        self.assertLess(
            len(xlong_fast.actual_step_indices),
            len(xlong_knee.actual_step_indices),
        )

    def test_prompt_length_does_not_change_the_physical_schedule(self) -> None:
        selector = V24FinalParetoRuntimeSelector()
        short = selector.select(
            workload=_workload(1280, 736, 243, 67_000),
            acceleration=82,
        )
        long = selector.select(
            workload=_workload(1280, 736, 243, 69_000),
            acceleration=82,
        )
        self.assertEqual(short.actual_step_indices, long.actual_step_indices)
        self.assertEqual(
            short.attention_action_schedule,
            long.attention_action_schedule,
        )
        self.assertFalse(short.summary["prompt_semantics_used"])

    def test_reference_media_continuously_shifts_to_safer_compute(self) -> None:
        selector = V24FinalParetoRuntimeSelector()
        base = selector.select(
            workload=_workload(*_SHAPES[1]), acceleration=90
        )
        reference = selector.select(
            workload=_workload(
                1280,
                736,
                243,
                72_000,
                service_family="reference",
                condition_count=2,
                reference_images=1,
                reference_audio=1,
            ),
            acceleration=90,
        )
        self.assertGreaterEqual(
            reference.summary["estimated_compute_units"],
            base.summary["estimated_compute_units"],
        )
        self.assertIn("reference_layout_guard", reference.summary["safety_guards"])
        self.assertIn("reference_media_guard", reference.summary["safety_guards"])

    def test_reference_video_keeps_every_dit_evaluation(self) -> None:
        selected = V24FinalParetoRuntimeSelector().select(
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
            "reference_video_no_forecast_guard",
            selected.summary["safety_guards"],
        )

    def test_arbitrary_step_counts_compile_complete_schedules(self) -> None:
        selector = V24FinalParetoRuntimeSelector()
        for steps in (4, 5, 8, 12, 17, 20, 24, 30):
            with self.subTest(steps=steps):
                selected = selector.select(
                    workload=_workload(*_SHAPES[1], steps=steps),
                    acceleration=88,
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


if __name__ == "__main__":
    unittest.main()
