from __future__ import annotations

import unittest

import torch

from h3serve.native_engine.forecast import (
    CalibrationForecastController,
    CurvatureForecastController,
    QualityConstrainedForecastFactory,
    QualityConstrainedForecastPolicy,
    TargetLayout,
)


class QualityConstrainedForecastTests(unittest.TestCase):
    def _constant_tail_policy(self) -> QualityConstrainedForecastPolicy:
        policy = QualityConstrainedForecastPolicy(step_count=20, sentinel_rows=128)
        layout = TargetLayout(
            start=0,
            stop=6,
            audio_rows=2,
            video_rows=4,
            latent_t=1,
            grid_h=2,
            grid_w=2,
        )
        tail = torch.full((6, 4), 0.25, dtype=torch.float32)
        for step in range(20):
            anchor = torch.full((6, 4), float(step), dtype=torch.float32)
            policy.observe(
                step_index=step,
                anchor_residual=anchor,
                tail_residual_host=tail,
                layout=layout,
            )
        return policy

    def test_policy_finds_global_minimum_when_tail_is_stationary(self) -> None:
        policy = self._constant_tail_policy()
        self.assertEqual(policy.finalize(), (0, 1, 2, 18, 19))
        self.assertEqual(policy.selected_mode, "curvature")
        self.assertEqual(policy.export()["calibration_points"], 20)

    def test_factory_changes_from_calibration_to_selected_controller(self) -> None:
        factory = QualityConstrainedForecastFactory(step_count=20, sentinel_rows=128)
        first = factory(segment_cache=None)
        self.assertIsInstance(first, CalibrationForecastController)
        source = self._constant_tail_policy()
        factory.policy.points = source.points
        first.export()
        second = factory(segment_cache=None)
        self.assertIsInstance(second, CurvatureForecastController)
        self.assertEqual(tuple(sorted(second.actual_steps)), (0, 1, 2, 18, 19))


if __name__ == "__main__":
    unittest.main()
