from __future__ import annotations

import unittest

from h3serve.native_engine.planner import (
    V19PlanningError,
    V19WorkloadContext,
    V24CurveProfile,
    V24HumanReview,
    V24ParetoRuntimeSelector,
    V24ReviewCandidate,
    V24StrategyVector,
    fit_v24_preference_posterior,
    v24_strategy_features,
)


def _workload() -> V19WorkloadContext:
    return V19WorkloadContext(
        model_variant="base",
        service_family="first_last",
        packed_tokens=67_535,
        condition_count=0,
        width=1280,
        height=736,
        frames=243,
        steps=20,
        actual_step_indices=tuple(range(20)),
        sampler="res_multistep",
        scheduler="simple",
    )


class V24PreferenceLearningTests(unittest.TestCase):
    def test_vector_expands_dense_and_forecast_physics(self) -> None:
        dense = V24ParetoRuntimeSelector().select(
            workload=_workload(), acceleration=0
        )
        dense_vector = V24StrategyVector.from_selection(dense, total_steps=20)
        self.assertEqual(dense_vector.actual_step_indices, tuple(range(20)))
        self.assertEqual(set(dense_vector.attention_ranks), {4})

        endpoint = V24ParetoRuntimeSelector().select(
            workload=_workload(), acceleration=100
        )
        vector = V24StrategyVector.from_selection(endpoint, total_steps=20)
        self.assertEqual(len(vector.attention_ranks), 1000)
        for step in vector.forecast_step_indices:
            row = vector.attention_ranks[step * 50:(step + 1) * 50]
            self.assertNotIn(-1, row[:3])
            self.assertEqual(set(row[3:]), {-1})
        self.assertEqual(len(vector.digest), 64)

    def test_curve_profiles_redistribute_one_matched_compute_budget(self) -> None:
        profiles = (
            V24CurveProfile(profile_id="attention", forecast_risk_scale=1.0),
            V24CurveProfile(profile_id="balanced", forecast_risk_scale=15.0),
            V24CurveProfile(profile_id="trajectory", forecast_risk_scale=25.0),
        )
        selections = tuple(
            V24ParetoRuntimeSelector(curve=profile).select(
                workload=_workload(), acceleration=75
            )
            for profile in profiles
        )
        ratios = tuple(
            float(row.summary["estimated_compute_ratio"]) for row in selections
        )
        self.assertLess(max(ratios) - min(ratios), 0.001)
        self.assertEqual(tuple(len(row.actual_step_indices) for row in selections), (10, 11, 12))
        self.assertEqual(
            len({
                V24StrategyVector.from_selection(row, total_steps=20).digest
                for row in selections
            }),
            3,
        )

    def test_invalid_curve_is_fail_closed(self) -> None:
        with self.assertRaises(V19PlanningError):
            V24CurveProfile(profile_id="bad", opening_decay=0.0)

    def test_rank_score_and_issue_labels_update_posterior(self) -> None:
        selections = tuple(
            V24ParetoRuntimeSelector(curve=profile).select(
                workload=_workload(), acceleration=75
            )
            for profile in (
                V24CurveProfile(profile_id="a", forecast_risk_scale=1.0),
                V24CurveProfile(profile_id="b", forecast_risk_scale=25.0),
            )
        )
        candidates = []
        for identifier, profile_id, selection in zip(
            ("a", "b"), ("a", "b"), selections
        ):
            vector = V24StrategyVector.from_selection(selection, total_steps=20)
            feature_map = v24_strategy_features(vector)
            candidates.append(V24ReviewCandidate(
                candidate_id=identifier,
                comparison_group="same",
                strategy_digest=vector.digest,
                features=tuple(feature_map.values()),
                video_path=f"{identifier}.mp4",
                curve_profile_id=profile_id,
                acceleration=75.0,
                workload_id="test",
            ))
        review = V24HumanReview(
            batch_id="r1",
            candidates=tuple(candidates),
            rankings={"same": (("b",), ("a",))},
            overall_scores={"a": 70.0, "b": 90.0},
            issues={
                "a": {
                    "contact_causality": "present",
                    "object_geometry_consistency": "present",
                },
                "b": {
                    "contact_causality": "absent",
                    "object_geometry_consistency": "absent",
                },
            },
            notes={"a": "motion error", "b": "normal"},
        )
        posterior = fit_v24_preference_posterior((review,))
        self.assertEqual(posterior.comparison_count, 1)
        self.assertGreater(
            posterior.preference_probability(
                candidates[1].model_features,
                candidates[0].model_features,
            ),
            0.5,
        )
        self.assertEqual(
            posterior.issue_heads["contact_causality"]["reported_count"], 2
        )
        self.assertEqual(
            posterior.issue_heads["object_geometry_consistency"]["reported_count"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
