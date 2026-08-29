from __future__ import annotations

from collections import Counter
import unittest

from h3serve.native_engine.planner.v19_long_horizon import (
    build_v19_long_horizon_round188_replay,
)
from h3serve.native_engine.planner.v19_long_quality import (
    V19LongQualityShieldSpec,
    build_v19_long_quality_shield,
    propose_v19_long_quality_frontier,
)
from h3serve.native_engine.planner.v19_candidates import (
    v19_blueprint_execution_digest,
)
from h3serve.native_engine.planner.v19_planner import V19PlanningError


def _source():
    return build_v19_long_horizon_round188_replay(
        candidate_id="source_round188", total_steps=20, acceleration=75.0
    )


class V19LongQualityShieldTests(unittest.TestCase):
    def test_preserves_trajectory_and_strengthens_cells(self) -> None:
        for core_action in ("sparse_topk_0.25", "sparse_topk_0.5"):
            with self.subTest(core_action=core_action):
                result = build_v19_long_quality_shield(
                    _source(),
                    candidate_id=f"quality_{core_action}",
                    spec=V19LongQualityShieldSpec(core_action=core_action),
                )
                self.assertEqual(
                    result.actual_step_indices,
                    (0, 1, 2, 3, 4, 6, 8, 11, 14, 17, 18, 19),
                )
                self.assertEqual(
                    result.terminal_actual_step_indices, (17, 18, 19)
                )
                self.assertEqual(
                    result.source_action_cell_counts,
                    (("sparse_topk_0.0625", 420), ("sparse_topk_0.1", 180)),
                )
                counts = Counter(dict(result.candidate_action_cell_counts))
                if core_action == "sparse_topk_0.25":
                    self.assertEqual(counts["sparse_topk_0.25"], 204)
                else:
                    self.assertEqual(counts["sparse_topk_0.5"], 72)
                    self.assertEqual(counts["sparse_topk_0.25"], 132)
                self.assertEqual(counts["sparse_topk_0.1"], 81)
                self.assertEqual(counts["sparse_topk_0.0625"], 315)
                self.assertEqual(sum(counts.values()), 600)
                self.assertEqual(result.core_upgraded_cells, 72)
                self.assertEqual(result.terminal_upgraded_cells, 132)
                self.assertTrue(result.constraint_report.proposal_eligible)
                self.assertFalse(result.constraint_report.release_eligible)

    def test_rejects_invalid_spec(self) -> None:
        with self.assertRaises(V19PlanningError):
            V19LongQualityShieldSpec(core_layers=(45, 39))
        with self.assertRaises(V19PlanningError):
            V19LongQualityShieldSpec(core_action="sparse_topk_0.2")
        with self.assertRaises(V19PlanningError):
            V19LongQualityShieldSpec(terminal_actual_count=0)

    def test_measured_frontier_is_speed_monotonic_and_digest_sealed(self) -> None:
        expected = (
            (
                75.0,
                "quality_core_0.5_terminal_0.25",
                "94ee124158bc394e0b640a9d68b013421b47ef1cf59312ec23552941ec142e55",
                165.68732891899708,
            ),
            (
                85.0,
                "balanced_core_0.25_terminal_0.25",
                "40cca9d011625da9b94d57b736972baa8041b0eb203919bfca819e298b38453e",
                163.857086789998,
            ),
            (
                95.0,
                "fast_round188",
                "495c2c8ff75f76aed16b1fd81a41f3e050df5bad38456dd68d0806c3e9c7cbad",
                152.123896590012,
            ),
        )
        previous_seconds = float("inf")
        for acceleration, point, digest, seconds in expected:
            with self.subTest(acceleration=acceleration):
                selection = propose_v19_long_quality_frontier(
                    total_steps=20,
                    acceleration=acceleration,
                    width=1280,
                    height=736,
                    latent_frames=72,
                    spatial_tokens_per_frame=920,
                    packed_tokens=67_535,
                )
                self.assertEqual(selection.operating_point, point)
                self.assertEqual(
                    v19_blueprint_execution_digest(selection.blueprint), digest
                )
                self.assertEqual(selection.observed_e2e_seconds, seconds)
                self.assertLess(selection.observed_e2e_seconds, previous_seconds)
                self.assertTrue(selection.proposal_eligible)
                self.assertFalse(selection.release_eligible)
                previous_seconds = selection.observed_e2e_seconds

    def test_measured_frontier_uses_exact_geometry_envelopes(self) -> None:
        selection = propose_v19_long_quality_frontier(
            total_steps=20,
            acceleration=85.0,
            width=1280,
            height=736,
            latent_frames=107,
            spatial_tokens_per_frame=920,
            packed_tokens=100_141,
        )
        self.assertEqual(
            selection.envelope_id, "v19_long_quality_720p15_base_no_reference_v1"
        )
        self.assertEqual(selection.observed_denoise_seconds, 222.71847620399785)

        invalid = (
            {"total_steps": 15},
            {"acceleration": 74.0},
            {"width": 1920},
            {"packed_tokens": 103_001},
            {"reference_images": 1},
        )
        base = {
            "total_steps": 20,
            "acceleration": 85.0,
            "width": 1280,
            "height": 736,
            "latent_frames": 107,
            "spatial_tokens_per_frame": 920,
            "packed_tokens": 100_141,
        }
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(V19PlanningError):
                propose_v19_long_quality_frontier(**(base | override))


if __name__ == "__main__":
    unittest.main()
