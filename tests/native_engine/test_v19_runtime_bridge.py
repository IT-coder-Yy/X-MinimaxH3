from __future__ import annotations

import unittest

from h3serve.native_engine.planner import (
    V19ActionUse,
    V19ForecastUse,
    V19PlanningError,
    blueprint_from_runtime_schedule,
    runtime_schedule_from_blueprint,
)


class V19RuntimeBridgeTests(unittest.TestCase):
    def _schedule(self):
        actual = (0, 1, 4)
        cells = []
        for step in actual:
            for layer in range(50):
                action = (
                    "dense"
                    if layer >= 45
                    else "forecastfrontier:sparse_topk_0.1"
                )
                cells.append((step, layer, action))
        for step in (2, 3):
            for layer in range(3):
                cells.append((step, layer, "forecastfrontier:sparse_topk_0.0625"))
        return tuple(sorted(cells))

    def test_round_trip_preserves_every_physical_cell(self) -> None:
        blueprint = blueprint_from_runtime_schedule(
            candidate_id="round_trip",
            total_steps=5,
            actual_step_indices=(0, 1, 4),
            attention_action_schedule=self._schedule(),
        )
        self.assertEqual(runtime_schedule_from_blueprint(blueprint), self._schedule())
        self.assertEqual(
            sum(isinstance(use, V19ForecastUse) for use in blueprint.action_uses),
            1,
        )
        self.assertGreater(
            sum(isinstance(use, V19ActionUse) for use in blueprint.action_uses),
            1,
        )
        self.assertEqual(blueprint.maximum_debt.consecutive_forecasts, 2)

    def test_missing_actual_cell_fails_closed(self) -> None:
        with self.assertRaises(V19PlanningError):
            blueprint_from_runtime_schedule(
                candidate_id="missing",
                total_steps=5,
                actual_step_indices=(0, 1, 4),
                attention_action_schedule=self._schedule()[1:],
            )


if __name__ == "__main__":
    unittest.main()
