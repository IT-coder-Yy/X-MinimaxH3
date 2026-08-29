from __future__ import annotations

import unittest

import torch

from h3serve.native_engine.forecast import (
    ForecastErrorDebtController,
    TargetLayout,
    V24_FORECAST_FEEDBACK_POLICY_ID,
    _TailHistory,
)


class V24ForecastFeedbackTests(unittest.TestCase):
    def test_actual_corrections_measure_request_local_secant_debt(self) -> None:
        controller = ForecastErrorDebtController(actual_steps=(0, 1, 2, 5))
        layout = TargetLayout(
            start=0,
            stop=8,
            audio_rows=2,
            video_rows=6,
            latent_t=1,
            grid_h=2,
            grid_w=3,
        )
        zeros = torch.zeros(8, 4)
        ones = torch.ones(8, 4)
        controller.history = [
            _TailHistory(0, zeros, zeros),
            _TailHistory(1, ones, ones),
        ]
        step2_anchor = torch.full((8, 4), 1.1)
        step2_tail = torch.full((8, 4), 1.111)
        controller._on_actual_observation(
            step_index=2,
            anchor_residual=step2_anchor,
            tail_residual_host=step2_tail,
            layout=layout,
        )
        self.assertTrue(controller.feedback_records[-1]["opening_baseline"])

        controller.history = [
            _TailHistory(1, ones, ones),
            _TailHistory(2, step2_anchor, step2_tail),
        ]
        controller._on_actual_observation(
            step_index=5,
            anchor_residual=torch.full((8, 4), 1.2),
            tail_residual_host=torch.full((8, 4), 10.0),
            layout=layout,
        )
        record = controller.feedback_records[-1]
        self.assertEqual(record["horizon"], 3)
        self.assertGreater(record["risk_ratio"], 1.0)
        self.assertGreater(record["forecast_debt"], 0.0)

    def test_observe_only_never_changes_the_requested_trajectory(self) -> None:
        controller = ForecastErrorDebtController(
            actual_steps=(0, 1, 4),
            recovery_enabled=False,
            max_runtime_promotions=0,
        )
        controller.history = [
            _TailHistory(0, torch.zeros(8, 4), torch.zeros(8, 4)),
            _TailHistory(1, torch.ones(8, 4), torch.ones(8, 4)),
        ]
        controller._forecast_debt = 100.0

        self.assertTrue(controller.should_forecast(2, requested_actual=False))
        report = controller.export()
        self.assertEqual(report["feedback_mode"], "observe_only")
        self.assertEqual(
            report["feedback_policy_id"], V24_FORECAST_FEEDBACK_POLICY_ID
        )
        self.assertFalse(report["adds_teacher_evaluations"])
        self.assertEqual(report["runtime_promotions"], [])

    def test_continuous_debt_can_spend_only_the_bounded_promotion_budget(self) -> None:
        controller = ForecastErrorDebtController(
            actual_steps=(0, 1, 4),
            recovery_enabled=True,
            max_runtime_promotions=1,
        )
        controller.history = [
            _TailHistory(0, torch.zeros(8, 4), torch.zeros(8, 4)),
            _TailHistory(1, torch.ones(8, 4), torch.ones(8, 4)),
        ]
        controller._forecast_debt = 1.25

        self.assertFalse(controller.should_forecast(2, requested_actual=False))
        self.assertAlmostEqual(controller._forecast_debt, 0.25)
        self.assertTrue(controller.should_forecast(3, requested_actual=False))
        self.assertEqual(controller.export()["runtime_promotions"], [2])

    def test_invalid_recovery_contract_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "promotion budget"):
            ForecastErrorDebtController(
                actual_steps=(0, 1),
                recovery_enabled=True,
                max_runtime_promotions=0,
            )


if __name__ == "__main__":
    unittest.main()
