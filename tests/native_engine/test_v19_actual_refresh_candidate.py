from __future__ import annotations

import unittest

from h3serve.native_engine.planner import (
    ROUND229_FORECAST_ANCHOR,
    V19PlanningError,
    blueprint_from_runtime_schedule,
    runtime_schedule_from_blueprint,
)
from scripts.derive_v19_actual_refresh_candidate import (
    derive_actual_refresh_candidate,
)


ACTUAL = (0, 1, 2, 3, 4, 8, 12, 15, 18, 19)


def _source():
    actual = set(ACTUAL)
    schedule = []
    for step in range(20):
        if step in actual:
            action = (
                "dense"
                if step == 18
                else "forecastfrontier:sparse_topk_0.25"
            )
            schedule.extend((step, layer, action) for layer in range(50))
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


class V19ActualRefreshCandidateTests(unittest.TestCase):
    def test_terminal_refresh_splits_forecast_run_and_clones_next_correction(self) -> None:
        candidate, summary = derive_actual_refresh_candidate(
            _source(),
            candidate_id="terminal_refresh",
            add_actual_steps=(17,),
        )
        self.assertEqual(
            summary["actual_step_indices"],
            [0, 1, 2, 3, 4, 8, 12, 15, 17, 18, 19],
        )
        self.assertEqual(summary["forecast_runs"][-1], [16])
        self.assertEqual(
            summary["cloned_attention_from_source_actual"], {"17": 18}
        )
        runtime = {
            (step, layer): action
            for step, layer, action in runtime_schedule_from_blueprint(candidate)
        }
        self.assertTrue(all(runtime[(17, layer)] == "dense" for layer in range(50)))
        self.assertEqual(runtime[(16, 0)], ROUND229_FORECAST_ANCHOR)
        self.assertTrue(summary["constraint_report"]["proposal_eligible"])
        self.assertFalse(summary["constraint_report"]["release_eligible"])

    def test_existing_actual_step_cannot_be_added_again(self) -> None:
        with self.assertRaises(V19PlanningError):
            derive_actual_refresh_candidate(
                _source(),
                candidate_id="invalid",
                add_actual_steps=(18,),
            )


if __name__ == "__main__":
    unittest.main()
