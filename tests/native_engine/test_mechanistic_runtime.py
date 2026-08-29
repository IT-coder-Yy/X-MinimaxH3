from __future__ import annotations

import unittest

import torch

from h3serve.native_engine.forecast import (
    ForecastErrorDebtController,
    _TailHistory,
)

from h3serve.native_engine.planner.mechanistic_control import (
    H3MechanisticControlModel,
    H3MechanisticWorkload,
    MechanisticControlError,
)
from h3serve.native_engine.planner.mechanistic_runtime import (
    H3MechanisticRuntimeController,
    MECHANISTIC_RUNTIME_POLICY_ID,
)


class H3MechanisticRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = H3MechanisticControlModel(H3MechanisticWorkload(
            total_steps=20,
            packed_tokens=34_871,
            video_tokens=34_040,
        ))
        cls.plan = cls.model.plan_for_cost_budget(maximum_cost_ms=60_000.0)

    def controller(self, **overrides) -> H3MechanisticRuntimeController:
        values = {
            "model": self.model,
            "plan": self.plan,
            "recovery_reserve_ms": 9_000.0,
            "maximum_promotions": 2,
            "risk_limit": self.plan.modeled_risk.total * 1.10,
        }
        values.update(overrides)
        return H3MechanisticRuntimeController(**values)

    def test_static_projection_exactly_replays_offline_risk(self) -> None:
        controller = self.controller()
        replay = controller._project_risk(
            after_step=-1,
            future_promotions=frozenset(),
        )
        self.assertAlmostEqual(replay.total, self.plan.modeled_risk.total)

    def test_null_observation_spends_no_recovery_compute(self) -> None:
        decision = self.controller().observe_actual(
            step_index=2,
            audio_risk_ratio=1.0,
            video_risk_ratio=1.0,
        )
        self.assertTrue(decision.admitted)
        self.assertEqual(decision.selected_future_promotions, ())
        self.assertEqual(decision.projected_extra_cost_ms, 0.0)
        self.assertEqual(decision.policy_id, MECHANISTIC_RUNTIME_POLICY_ID)
        self.assertFalse(decision.to_dict()["historical_schedule_used"])

    def test_continuous_uncertainty_selects_minimum_cost_admitted_recovery(self) -> None:
        decision = self.controller().observe_actual(
            step_index=2,
            audio_risk_ratio=1.2,
            video_risk_ratio=1.2,
        )
        self.assertTrue(decision.admitted)
        self.assertEqual(decision.selected_future_promotions, (6,))
        self.assertLessEqual(
            decision.projected_extra_cost_ms,
            decision.recovery_reserve_ms,
        )
        self.assertGreater(decision.candidate_subsets_evaluated, 1)

    def test_observed_forecast_run_is_charged_as_incurred_risk(self) -> None:
        controller = self.controller(
            recovery_reserve_ms=0.0,
            maximum_promotions=0,
            risk_limit=self.plan.modeled_risk.total * 10.0,
        )
        baseline = controller._project_risk(
            after_step=7,
            future_promotions=frozenset(),
        )
        # In this plan step 6 is Forecast and step 7 is its Actual correction.
        decision = controller.observe_actual(
            step_index=7,
            audio_risk_ratio=1.5,
            video_risk_ratio=1.5,
        )
        self.assertGreater(decision.projected_risk.total, baseline.total)
        self.assertIn("6", controller.export()["realized_forecast_beliefs"])

    def test_extreme_uncertainty_fails_visibly_when_reserve_is_insufficient(self) -> None:
        decision = self.controller().observe_actual(
            step_index=2,
            audio_risk_ratio=3.0,
            video_risk_ratio=3.0,
        )
        self.assertFalse(decision.admitted)
        self.assertEqual(len(decision.selected_future_promotions), 2)
        self.assertLessEqual(
            decision.projected_extra_cost_ms,
            decision.recovery_reserve_ms,
        )

    def test_running_upper_belief_cannot_relax_after_anomaly(self) -> None:
        controller = self.controller()
        first = controller.observe_actual(
            step_index=2,
            audio_risk_ratio=1.3,
            video_risk_ratio=1.4,
        )
        second = controller.observe_actual(
            step_index=4,
            audio_risk_ratio=0.8,
            video_risk_ratio=0.9,
        )
        self.assertEqual(second.belief.audio_scale, first.belief.audio_scale)
        self.assertEqual(second.belief.video_scale, first.belief.video_scale)

    def test_observations_must_follow_trajectory_order(self) -> None:
        controller = self.controller()
        controller.observe_actual(
            step_index=2,
            audio_risk_ratio=1.0,
            video_risk_ratio=1.0,
        )
        with self.assertRaisesRegex(
            MechanisticControlError, "strictly increasing"
        ):
            controller.observe_actual(
                step_index=2,
                audio_risk_ratio=1.0,
                video_risk_ratio=1.0,
            )

    def test_checkpoint_round_trip_preserves_runtime_control_state(self) -> None:
        controller = self.controller()
        expected = controller.observe_actual(
            step_index=2,
            audio_risk_ratio=1.2,
            video_risk_ratio=1.2,
        )
        state = controller.checkpoint_state()
        restored = self.controller()
        restored.restore_checkpoint_state(state)

        self.assertEqual(restored.belief, expected.belief)
        self.assertEqual(
            restored.selected_future_promotions,
            expected.selected_future_promotions,
        )
        self.assertEqual(restored.checkpoint_state(), state)

    def test_forecast_executor_spends_only_optimizer_selected_promotion(self) -> None:
        runtime = self.controller()
        decision = runtime.observe_actual(
            step_index=2,
            audio_risk_ratio=1.2,
            video_risk_ratio=1.2,
        )
        self.assertEqual(decision.selected_future_promotions, (6,))
        forecast = ForecastErrorDebtController(
            actual_steps=self.plan.actual_step_indices,
            risk_reserve_controller=runtime,
        )
        forecast.history = [
            _TailHistory(0, torch.zeros(2, 2), torch.zeros(2, 2)),
            _TailHistory(2, torch.ones(2, 2), torch.ones(2, 2)),
        ]

        self.assertFalse(forecast.should_forecast(6, requested_actual=False))
        self.assertTrue(forecast.should_forecast(8, requested_actual=False))
        self.assertEqual(forecast.export()["runtime_promotions"], [6])

        state = forecast.checkpoint_state()
        restored_runtime = self.controller()
        restored = ForecastErrorDebtController(
            actual_steps=self.plan.actual_step_indices,
            risk_reserve_controller=restored_runtime,
        )
        restored.restore_checkpoint_state(state)
        restored_state = restored.checkpoint_state()
        self.assertEqual(
            restored_state["forecast_feedback"], state["forecast_feedback"]
        )
        self.assertEqual(len(restored.history), len(forecast.history))
        for left, right in zip(restored.history, forecast.history, strict=True):
            self.assertEqual(left.step_index, right.step_index)
            self.assertTrue(torch.equal(
                left.anchor_residual_sample.cpu(),
                right.anchor_residual_sample.cpu(),
            ))
            self.assertTrue(torch.equal(
                left.tail_residual_host,
                right.tail_residual_host,
            ))


if __name__ == "__main__":
    unittest.main()
