import math
import unittest

from h3serve.native_engine.planner.mechanistic_control import (
    H3MechanisticAdmission,
    H3MechanisticControlModel,
    H3MechanisticWorkload,
    MECHANISTIC_CONTROL_POLICY_ID,
    verify_mechanistic_pareto_order,
)


class H3MechanisticControlTests(unittest.TestCase):
    @staticmethod
    def workload(**overrides) -> H3MechanisticWorkload:
        values = {
            "total_steps": 20,
            "packed_tokens": 34_871,
            "video_tokens": 34_040,
        }
        values.update(overrides)
        return H3MechanisticWorkload(**values)

    def test_dense_endpoint_emerges_from_zero_risk_price(self) -> None:
        model = H3MechanisticControlModel(self.workload())
        plan = model.solve_lagrangian(0.0)

        self.assertEqual(plan.policy_id, MECHANISTIC_CONTROL_POLICY_ID)
        self.assertEqual(plan.actual_step_indices, tuple(range(20)))
        self.assertEqual(plan.forecast_step_indices, ())
        self.assertEqual(plan.modeled_risk.total, 0.0)
        self.assertEqual(
            {action for choice in plan.choices for action in choice.attention_actions},
            {"dense"},
        )
        self.assertFalse(plan.certificate.historical_schedule_used)
        self.assertTrue(plan.certificate.exact_for_declared_lagrangian_problem)
        self.assertTrue(model.verify(plan).valid)

    def test_compute_price_jointly_changes_trajectory_and_attention(self) -> None:
        model = H3MechanisticControlModel(self.workload())
        dense = model.solve_lagrangian(0.0)
        balanced = model.solve_lagrangian(1.0e-4)
        fast = model.solve_lagrangian(None)

        self.assertLess(balanced.predicted_cost_ms, dense.predicted_cost_ms)
        self.assertLess(fast.predicted_cost_ms, balanced.predicted_cost_ms)
        self.assertGreater(balanced.modeled_risk.total, dense.modeled_risk.total)
        self.assertGreater(fast.modeled_risk.total, balanced.modeled_risk.total)
        self.assertTrue(balanced.forecast_step_indices)
        self.assertIn(
            "sparse_topk_0.5",
            {action for row in balanced.choices for action in row.attention_actions},
        )
        self.assertTrue(model.verify(balanced).valid)

    def test_forecast_attention_coupling_is_continuous_not_a_layer_floor(self) -> None:
        model = H3MechanisticControlModel(self.workload())
        clean = model._actual_local_action(
            step=8,
            layer=37,
            action="sparse_topk_0.25",
            prior_forecast_run=0,
        )
        after_one = model._actual_local_action(
            step=8,
            layer=37,
            action="sparse_topk_0.25",
            prior_forecast_run=1,
        )
        after_four = model._actual_local_action(
            step=8,
            layer=37,
            action="sparse_topk_0.25",
            prior_forecast_run=4,
        )

        self.assertEqual(clean.risk.interaction_energy, 0.0)
        self.assertGreater(after_one.risk.interaction_energy, 0.0)
        self.assertGreater(
            after_four.risk.interaction_energy,
            after_one.risk.interaction_energy,
        )
        self.assertEqual(clean.name, after_one.name)
        self.assertEqual(after_one.name, after_four.name)

    def test_forecast_horizon_increases_local_error_at_fixed_phase(self) -> None:
        model = H3MechanisticControlModel(self.workload())
        for modality in ("audio", "video"):
            one = model._forecast_response(modality=modality, step=10, horizon=1)
            four = model._forecast_response(modality=modality, step=10, horizon=4)
            self.assertGreater(four.mean, one.mean)
            self.assertGreater(four.upper, four.mean)

    def test_impulse_identified_propagation_gain_protects_early_phase(self) -> None:
        model = H3MechanisticControlModel(self.workload())
        for mechanism in ("single_forecast", "single_attention"):
            for modality in ("audio", "video"):
                gains = tuple(
                    model._propagation_gain(
                        mechanism=mechanism,
                        modality=modality,
                        step=step,
                    ).upper
                    for step in (2, 10, 18)
                )
                self.assertGreater(gains[0], gains[1])
                self.assertGreater(gains[1], gains[2])

    def test_same_speed_solver_uses_phase_gain_without_schedule_anchor(self) -> None:
        model = H3MechanisticControlModel(self.workload())
        plan = model.plan_for_cost_budget(maximum_cost_ms=45_500.0)

        # The opening causal spine now emerges from the measured downstream
        # gain surface: no V009/V022 Actual-step table enters the solver.
        self.assertEqual(plan.actual_step_indices[:3], (0, 1, 2))
        self.assertLessEqual(plan.maximum_forecast_run, 2)
        self.assertFalse(plan.certificate.historical_schedule_used)

    def test_required_actual_is_a_capability_constraint(self) -> None:
        model = H3MechanisticControlModel(self.workload(
            required_actual_steps=(7, 11, 19),
        ))
        plan = model.solve_lagrangian(None)
        self.assertTrue({7, 11, 19}.issubset(plan.actual_step_indices))
        self.assertEqual(plan.actual_step_indices[:2], (0, 1))
        self.assertEqual(plan.actual_step_indices[-1], 19)

    def test_out_of_calibration_shape_inflates_epistemic_risk(self) -> None:
        in_range = H3MechanisticControlModel(self.workload())
        out_of_range = H3MechanisticControlModel(self.workload(
            packed_tokens=218_280,
            video_tokens=216_200,
        ))
        inside = in_range._forecast_response(
            modality="video", step=8, horizon=4
        )
        outside = out_of_range._forecast_response(
            modality="video", step=8, horizon=4
        )
        self.assertGreater(outside.upper / outside.mean, inside.upper / inside.mean)

    def test_acceleration_selects_a_monotone_supported_pareto_sequence(self) -> None:
        model = H3MechanisticControlModel(self.workload())
        admission = H3MechanisticAdmission(
            calibration_id="test_mechanism_risk_boundary",
            maximum_modeled_risk=5.0,
            evidence_ids=("synthetic-test-boundary",),
        )
        plans = tuple(
            model.plan_for_acceleration(
                acceleration=value,
                admission=admission,
            )
            for value in (0.0, 25.0, 50.0, 75.0, 100.0)
        )

        verification = verify_mechanistic_pareto_order(plans)
        self.assertTrue(verification.valid, verification.reasons)
        self.assertEqual(plans[0].modeled_risk.total, 0.0)
        self.assertLessEqual(plans[-1].modeled_risk.total, 5.0)
        self.assertLess(plans[-1].predicted_cost_ms, plans[0].predicted_cost_ms)
        self.assertAlmostEqual(
            plans[2].target_cost_ms,
            0.5 * (
                plans[0].predicted_cost_ms + plans[-1].predicted_cost_ms
            ),
        )
        for plan in plans:
            self.assertTrue(model.verify(plan).valid)
            self.assertEqual(plan.admission_id, admission.calibration_id)

    def test_same_speed_budget_is_solved_without_historical_schedule_input(self) -> None:
        model = H3MechanisticControlModel(self.workload())
        plan = model.plan_for_cost_budget(maximum_cost_ms=45_500.0)
        self.assertLessEqual(plan.predicted_cost_ms, 45_500.0)
        self.assertIsNone(plan.admission_id)
        self.assertIsNone(plan.acceleration)
        self.assertFalse(plan.certificate.historical_schedule_used)
        self.assertTrue(model.verify(plan).valid)

    def test_runtime_schedule_is_complete_and_implementation_bound(self) -> None:
        model = H3MechanisticControlModel(self.workload())
        plan = model.solve_lagrangian(1.0e-4)
        schedule = plan.runtime_action_schedule()
        expected = len(plan.actual_step_indices) * 50 + len(
            plan.forecast_step_indices
        ) * 3
        self.assertEqual(len(schedule), expected)
        self.assertTrue(any(
            action.startswith("round215:sparse_topk_")
            for _step, _layer, action in schedule
        ))
        self.assertTrue(any(
            action == "forecastfrontier:sparse_topk_0.0625"
            for _step, _layer, action in schedule
        ))

    def test_external_schedule_projection_never_hides_kernel_mismatch(self) -> None:
        model = H3MechanisticControlModel(self.workload())
        optimized = model.solve_lagrangian(1.0e-4)
        frontier_schedule = tuple(
            (step, layer, action.replace("round215:", "frontier:"))
            for step, layer, action in optimized.runtime_action_schedule()
        )
        evaluation = model.evaluate_external_schedule(
            evaluation_id="historical-frontier-proxy",
            actual_step_indices=optimized.actual_step_indices,
            attention_action_schedule=frontier_schedule,
        )
        self.assertFalse(evaluation.calibration_implementation_match)
        self.assertIn("frontier", evaluation.source_implementation_ids)
        self.assertTrue(math.isclose(
            evaluation.modeled_risk.total,
            optimized.modeled_risk.total,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ))

        matched = model.evaluate_external_schedule(
            evaluation_id="round215-replay",
            actual_step_indices=optimized.actual_step_indices,
            attention_action_schedule=optimized.runtime_action_schedule(),
        )
        self.assertTrue(matched.calibration_implementation_match)
        self.assertEqual(matched.schedule_digest, evaluation.schedule_digest)

    def test_cost_only_plan_is_still_finite_and_replayable(self) -> None:
        model = H3MechanisticControlModel(self.workload())
        plan = model.solve_lagrangian(None)
        self.assertTrue(math.isfinite(plan.predicted_cost_ms))
        self.assertGreater(plan.modeled_risk.total, 0.0)
        self.assertEqual(plan.actual_step_indices[-1], 19)
        self.assertTrue(model.verify(plan).valid)


if __name__ == "__main__":
    unittest.main()
