from __future__ import annotations

import unittest

from h3serve.native_engine.planner import (
    ROUND188_REVIEWED_20_STEP_ACTUALS,
    build_v19_long_horizon_round188_replay,
    evaluate_v19_human_constraints,
    runtime_schedule_from_blueprint,
    v19_long_horizon_actual_steps,
    v19_long_horizon_screening_policy,
)


class V19LongHorizonTests(unittest.TestCase):
    def test_acceleration_mapping_is_nested_and_stops_at_reviewed_floor(self) -> None:
        previous = set(range(20))
        for acceleration in (0, 10, 25, 40, 50, 60, 75, 95, 100):
            actual = v19_long_horizon_actual_steps(20, acceleration)
            current = set(actual)
            self.assertTrue(current.issubset(previous))
            previous = current
        self.assertEqual(
            v19_long_horizon_actual_steps(20, 75),
            ROUND188_REVIEWED_20_STEP_ACTUALS,
        )
        self.assertEqual(
            v19_long_horizon_actual_steps(20, 100),
            ROUND188_REVIEWED_20_STEP_ACTUALS,
        )

    def test_round188_replay_has_exact_trajectory_and_causal_head_rail(self) -> None:
        blueprint = build_v19_long_horizon_round188_replay(
            candidate_id="long_replay",
            total_steps=20,
            acceleration=75,
        )
        runtime = {
            (step, layer): action
            for step, layer, action in runtime_schedule_from_blueprint(blueprint)
        }
        for step in ROUND188_REVIEWED_20_STEP_ACTUALS:
            self.assertEqual(
                runtime[(step, 29)],
                "frontier:sparse_topk_0.0625",
            )
            self.assertEqual(
                runtime[(step, 30)],
                "frontier:sparse_topk_0.1",
            )
            self.assertEqual(
                runtime[(step, 45)],
                "frontier:sparse_topk_0.1",
            )
            self.assertEqual(
                runtime[(step, 46)],
                "frontier:sparse_topk_0.0625",
            )
        report = evaluate_v19_human_constraints(
            blueprint,
            v19_long_horizon_screening_policy(20),
        )
        self.assertTrue(report.proposal_eligible)
        self.assertFalse(report.release_eligible)
        self.assertEqual(report.actual_step_fraction, 0.60)
        self.assertEqual(max(map(len, report.forecast_runs)), 2)


if __name__ == "__main__":
    unittest.main()
