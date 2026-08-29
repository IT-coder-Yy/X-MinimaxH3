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
from h3serve.native_engine.planner.v19_long_temporal_consolidation import (
    V19LongTemporalConsolidationSpec,
    build_v19_long_temporal_consolidation,
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


class V19LongTemporalConsolidationTests(unittest.TestCase):
    def test_consolidates_to_reviewed_v009_temporal_anchors(self):
        result = build_v19_long_temporal_consolidation(
            _v014b(), candidate_id="v020a_consolidated_refresh"
        )
        self.assertEqual(
            result.actual_step_indices,
            (0, 1, 2, 3, 4, 8, 12, 15, 18, 19),
        )
        self.assertEqual(
            result.forecast_runs,
            ((5, 6, 7), (9, 10, 11), (13, 14), (16, 17)),
        )
        self.assertEqual(
            dict(result.cloned_from_source_actual),
            {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 8: 8, 12: 11,
             15: 14, 18: 18, 19: 19},
        )
        self.assertEqual(result.recovery_actual_step_indices, ())
        self.assertEqual(result.recovery_upgraded_cells, 0)
        self.assertEqual(
            Counter(dict(result.candidate_action_cell_counts)),
            Counter({
                "sparse_topk_0.0625": 280,
                "sparse_topk_0.1": 72,
                "sparse_topk_0.25": 88,
                "sparse_topk_0.5": 60,
            }),
        )
        self.assertTrue(result.constraint_report.proposal_eligible)
        self.assertFalse(result.constraint_report.release_eligible)

    def test_strong_recovery_isolated_to_post_forecast_actuals(self):
        result = build_v19_long_temporal_consolidation(
            _v014b(),
            candidate_id="v020b_consolidated_refresh_recovery025",
            spec=V19LongTemporalConsolidationSpec(
                recovery_action="sparse_topk_0.25"
            ),
        )
        self.assertEqual(
            result.recovery_actual_step_indices, (8, 12, 15, 18)
        )
        self.assertEqual(result.recovery_upgraded_cells, 132)
        self.assertEqual(
            Counter(dict(result.candidate_action_cell_counts)),
            Counter({
                "sparse_topk_0.0625": 175,
                "sparse_topk_0.1": 45,
                "sparse_topk_0.25": 220,
                "sparse_topk_0.5": 60,
            }),
        )

    def test_invalid_or_uncalibrated_trajectory_fails_closed(self):
        with self.assertRaises(V19PlanningError):
            V19LongTemporalConsolidationSpec(
                target_actual_step_indices=(0, 2, 1, 19)
            )
        with self.assertRaises(V19PlanningError):
            build_v19_long_temporal_consolidation(
                _v014b(),
                candidate_id="too_few",
                spec=V19LongTemporalConsolidationSpec(
                    target_actual_step_indices=(0, 1, 2, 3, 4, 18, 19),
                    maximum_forecast_run=14,
                ),
            )


if __name__ == "__main__":
    unittest.main()
