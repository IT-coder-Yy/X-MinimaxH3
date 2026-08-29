from __future__ import annotations

from collections import Counter
import unittest

from h3serve.native_engine.planner.v19_long_boundary_replay import (
    V19LongBoundaryReplaySpec,
    build_v19_long_boundary_replay,
)
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
from h3serve.native_engine.planner.v19_runtime_bridge import (
    ROUND229_FORECAST_ANCHOR,
    blueprint_from_runtime_schedule,
)


ACTUALS = (0, 1, 2, 3, 4, 8, 12, 15, 18, 19)


def _source():
    round188 = build_v19_long_horizon_round188_replay(
        candidate_id="round188", total_steps=20, acceleration=75.0
    )
    v014b = build_v19_long_quality_shield(
        round188,
        candidate_id="v014b",
        spec=V19LongQualityShieldSpec(core_action="sparse_topk_0.5"),
    ).blueprint
    return build_v19_long_temporal_consolidation(
        v014b,
        candidate_id="v020b",
        spec=V19LongTemporalConsolidationSpec(
            recovery_action="sparse_topk_0.25"
        ),
    ).blueprint


def _dense_donor():
    actual = set(ACTUALS)
    schedule = []
    for step in range(20):
        if step in actual:
            schedule.extend((step, layer, "dense") for layer in range(50))
        else:
            schedule.extend(
                (step, layer, ROUND229_FORECAST_ANCHOR)
                for layer in range(3)
            )
    return blueprint_from_runtime_schedule(
        candidate_id="dense_donor",
        total_steps=20,
        actual_step_indices=ACTUALS,
        attention_action_schedule=schedule,
    )


class V19LongBoundaryReplayTests(unittest.TestCase):
    def test_replays_only_selected_donor_actual_cells(self):
        result = build_v19_long_boundary_replay(
            _source(), _dense_donor(), candidate_id="terminal_dense"
        )
        self.assertEqual(result.actual_step_indices, ACTUALS)
        self.assertEqual(result.replay_actual_step_indices, (18, 19))
        self.assertEqual(result.replay_layer_indices, tuple(range(50)))
        self.assertEqual(result.replayed_cells, 100)
        self.assertEqual(result.physically_changed_cells, 100)
        self.assertEqual(
            Counter(dict(result.candidate_action_cell_counts)),
            Counter({
                "dense": 100,
                "sparse_topk_0.0625": 175,
                "sparse_topk_0.1": 45,
                "sparse_topk_0.25": 132,
                "sparse_topk_0.5": 48,
            }),
        )
        self.assertTrue(result.constraint_report.proposal_eligible)
        self.assertFalse(result.constraint_report.release_eligible)

    def test_partial_opening_spine_replays_only_selected_layers(self):
        layers = tuple((*range(30, 44), 45))
        result = build_v19_long_boundary_replay(
            _source(),
            _dense_donor(),
            candidate_id="opening_spine_dense",
            spec=V19LongBoundaryReplaySpec(
                replay_actual_step_indices=(0,),
                replay_layer_indices=layers,
            ),
        )
        self.assertEqual(result.replay_layer_indices, layers)
        self.assertEqual(result.replayed_cells, 15)
        self.assertEqual(result.physically_changed_cells, 15)
        self.assertEqual(
            Counter(dict(result.candidate_action_cell_counts)),
            Counter({
                "dense": 15,
                "sparse_topk_0.0625": 175,
                "sparse_topk_0.1": 36,
                "sparse_topk_0.25": 220,
                "sparse_topk_0.5": 54,
            }),
        )

    def test_non_actual_replay_step_fails_closed(self):
        with self.assertRaises(V19PlanningError):
            build_v19_long_boundary_replay(
                _source(),
                _dense_donor(),
                candidate_id="bad",
                spec=V19LongBoundaryReplaySpec(
                    replay_actual_step_indices=(5,)
                ),
            )
        with self.assertRaises(V19PlanningError):
            V19LongBoundaryReplaySpec(replay_layer_indices=(45, 30))


if __name__ == "__main__":
    unittest.main()
