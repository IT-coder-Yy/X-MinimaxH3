from __future__ import annotations

from collections import Counter
import unittest

from h3serve.native_engine.planner.v19_long_horizon import (
    build_v19_long_horizon_round188_replay,
)
from h3serve.native_engine.planner.v19_long_quality import (
    V19LongQualityShieldSpec,
    build_v19_long_quality_shield,
)
from h3serve.native_engine.planner.v19_long_temporal_stability import (
    V19LongTemporalStabilitySpec,
    build_v19_long_temporal_stability_shield,
)
from h3serve.native_engine.planner.v19_planner import V19PlanningError


def _v014b():
    source = build_v19_long_horizon_round188_replay(
        candidate_id="source_v012",
        total_steps=20,
        acceleration=75.0,
    )
    return build_v19_long_quality_shield(
        source,
        candidate_id="source_v014b",
        spec=V19LongQualityShieldSpec(core_action="sparse_topk_0.5"),
    ).blueprint


class V19LongTemporalStabilityTests(unittest.TestCase):
    def test_recovery_shield_only_strengthens_post_long_forecast_actuals(self):
        result = build_v19_long_temporal_stability_shield(
            _v014b(), candidate_id="v019a_recovery025"
        )
        self.assertEqual(
            result.actual_step_indices,
            (0, 1, 2, 3, 4, 6, 8, 11, 14, 17, 18, 19),
        )
        self.assertEqual(
            result.forecast_runs,
            ((5,), (7,), (9, 10), (12, 13), (15, 16)),
        )
        self.assertEqual(result.recovery_actual_step_indices, (11, 14, 17))
        self.assertEqual(result.recovery_upgraded_cells, 88)
        self.assertEqual(result.structural_upgraded_cells, 0)
        self.assertEqual(
            Counter(dict(result.candidate_action_cell_counts)),
            Counter({
                "sparse_topk_0.0625": 245,
                "sparse_topk_0.1": 63,
                "sparse_topk_0.25": 220,
                "sparse_topk_0.5": 72,
            }),
        )
        self.assertTrue(result.constraint_report.proposal_eligible)
        self.assertFalse(result.constraint_report.release_eligible)

    def test_mid_structural_rail_is_an_independent_superset(self):
        result = build_v19_long_temporal_stability_shield(
            _v014b(),
            candidate_id="v019b_recovery025_mid22_24",
            spec=V19LongTemporalStabilitySpec(structural_layers=(22, 23, 24)),
        )
        self.assertEqual(result.recovery_upgraded_cells, 88)
        self.assertEqual(result.structural_upgraded_cells, 21)
        self.assertEqual(
            Counter(dict(result.candidate_action_cell_counts)),
            Counter({
                "sparse_topk_0.0625": 224,
                "sparse_topk_0.1": 63,
                "sparse_topk_0.25": 241,
                "sparse_topk_0.5": 72,
            }),
        )

    def test_invalid_specs_fail_closed(self):
        with self.assertRaises(V19PlanningError):
            V19LongTemporalStabilitySpec(minimum_forecast_run=0)
        with self.assertRaises(V19PlanningError):
            V19LongTemporalStabilitySpec(structural_layers=(24, 22))
        with self.assertRaises(V19PlanningError):
            V19LongTemporalStabilitySpec(recovery_action="sparse_topk_0.2")


if __name__ == "__main__":
    unittest.main()
