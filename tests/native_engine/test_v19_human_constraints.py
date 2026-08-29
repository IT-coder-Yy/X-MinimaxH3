from __future__ import annotations

from dataclasses import replace
import unittest

from h3serve.native_engine.planner import (
    ROUND229_FORECAST_ANCHOR,
    V19HumanConstraintPolicy,
    blueprint_from_runtime_schedule,
    evaluate_v19_human_constraints,
    v19_long_horizon_screening_policy,
    v19_round02_av_motion_screening_policy,
    v19_blueprint_execution_digest,
)


ACTUAL = (0, 1, 2, 4, 5)


def _blueprint(
    *,
    action: str = "forecastfrontier:sparse_topk_0.25",
    actual_steps: tuple[int, ...] = ACTUAL,
):
    actual = set(actual_steps)
    schedule = []
    for step in range(6):
        if step in actual:
            schedule.extend((step, layer, action) for layer in range(50))
        else:
            schedule.extend(
                (step, layer, ROUND229_FORECAST_ANCHOR) for layer in range(3)
            )
    return blueprint_from_runtime_schedule(
        candidate_id="screened",
        total_steps=6,
        actual_step_indices=actual_steps,
        attention_action_schedule=tuple(schedule),
    )


def _policy(**changes):
    base = V19HumanConstraintPolicy(
        policy_id="test",
        minimum_actual_keep_ratio=0.25,
        maximum_forecast_run=1,
        required_human_gates=("motion_causality", "mouth_clarity"),
    )
    return replace(base, **changes)


class V19HumanConstraintTests(unittest.TestCase):
    def test_round02_policy_seals_reviewed_failures_and_quality_gates(self) -> None:
        policy = v19_round02_av_motion_screening_policy()
        self.assertEqual(policy.minimum_actual_keep_ratio, 0.25)
        self.assertEqual(policy.maximum_forecast_run, 3)
        self.assertEqual(len(policy.rejected_execution_digests), 2)
        self.assertEqual(
            policy.required_human_gates,
            (
                "normal_motion_causality",
                "speaking_mouth_clarity",
                "speech_pacing_and_dialogue_fit",
            ),
        )

    def test_static_pass_still_requires_human_outcome_gates(self) -> None:
        report = evaluate_v19_human_constraints(_blueprint(), _policy())
        self.assertTrue(report.proposal_eligible)
        self.assertFalse(report.release_eligible)
        self.assertEqual(report.minimum_observed_keep_ratio, 0.25)
        self.assertEqual(report.forecast_runs, ((3,),))
        self.assertEqual(
            report.unevaluated_human_gates,
            ("motion_causality", "mouth_clarity"),
        )

    def test_low_density_action_fails_noncompensating_floor(self) -> None:
        report = evaluate_v19_human_constraints(
            _blueprint(action="forecastfrontier:sparse_topk_0.1"),
            _policy(),
        )
        self.assertFalse(report.proposal_eligible)
        self.assertIn(
            "actual_keep_ratio_below_human_search_floor:0.1<0.25",
            report.rejection_reasons,
        )

    def test_exact_human_rejected_execution_identity_is_blocked(self) -> None:
        blueprint = _blueprint()
        report = evaluate_v19_human_constraints(
            blueprint,
            _policy(rejected_execution_digests=(
                v19_blueprint_execution_digest(blueprint),
            )),
        )
        self.assertFalse(report.proposal_eligible)
        self.assertIn(
            "execution_digest_rejected_by_human_review",
            report.rejection_reasons,
        )

    def test_forecast_run_limit_is_a_hard_constraint(self) -> None:
        report = evaluate_v19_human_constraints(
            _blueprint(actual_steps=(0, 1, 2, 5)),
            _policy(maximum_forecast_run=1),
        )
        self.assertFalse(report.proposal_eligible)
        self.assertIn(
            "forecast_run_exceeds_human_search_limit:2>1",
            report.rejection_reasons,
        )

    def test_long_horizon_policy_separates_short_clip_search_floor(self) -> None:
        policy = v19_long_horizon_screening_policy(20)
        self.assertEqual(policy.minimum_actual_steps, 12)
        self.assertEqual(policy.minimum_actual_fraction, 0.60)
        self.assertEqual(policy.maximum_forecast_run, 2)
        self.assertEqual(
            policy.required_actual_step_indices,
            (0, 1, 2, 3, 4, 17, 18, 19),
        )
        self.assertEqual(policy.minimum_layer_keep_ratios[29], 0.0625)
        self.assertEqual(policy.minimum_layer_keep_ratios[30], 0.10)
        self.assertEqual(policy.minimum_layer_keep_ratios[45], 0.10)
        self.assertEqual(policy.minimum_layer_keep_ratios[46], 0.0625)


if __name__ == "__main__":
    unittest.main()
