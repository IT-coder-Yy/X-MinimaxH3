from __future__ import annotations

import unittest

from h3serve.native_engine.planner import (
    ROUND229_FORECAST_ANCHOR,
    V19ActionUse,
    V19PlanningError,
    blueprint_from_runtime_schedule,
    runtime_schedule_from_blueprint,
)
from scripts.derive_v19_trajectory_candidate import derive_candidate


ACTUAL = (0, 1, 2, 3, 4, 8, 12, 15, 18, 19)


def _source():
    actual = set(ACTUAL)
    schedule = []
    for step in range(20):
        if step in actual:
            schedule.extend(
                (step, layer, "forecastfrontier:sparse_topk_0.0625")
                for layer in range(50)
            )
        else:
            schedule.extend(
                (step, layer, ROUND229_FORECAST_ANCHOR)
                for layer in range(3)
            )
    return blueprint_from_runtime_schedule(
        candidate_id="source",
        total_steps=20,
        actual_step_indices=ACTUAL,
        attention_action_schedule=tuple(schedule),
    )


class V19TrajectoryCandidateTests(unittest.TestCase):
    def test_tail_actual_can_become_bounded_forecast_with_correction_rail(self) -> None:
        candidate, summary = derive_candidate(
            _source(),
            candidate_id="tail_forecast",
            remove_actual_steps=(18,),
            correction_causal_topk=0.25,
        )
        actual = tuple(sorted({
            step
            for use in candidate.action_uses
            if isinstance(use, V19ActionUse)
            for step in use.step_indices
        }))
        self.assertEqual(actual, (0, 1, 2, 3, 4, 8, 12, 15, 19))
        self.assertEqual(summary["extended_forecast_runs"], [[16, 17, 18]])
        runtime = {
            (step, layer): action
            for step, layer, action in runtime_schedule_from_blueprint(candidate)
        }
        for layer in (*range(30, 44), 45):
            self.assertEqual(
                runtime[(19, layer)],
                "forecastfrontier:sparse_topk_0.25",
            )

    def test_forecast_run_limit_fails_closed(self) -> None:
        with self.assertRaises(V19PlanningError):
            derive_candidate(
                _source(),
                candidate_id="too_long",
                remove_actual_steps=(15,),
                maximum_forecast_run=3,
            )


if __name__ == "__main__":
    unittest.main()
