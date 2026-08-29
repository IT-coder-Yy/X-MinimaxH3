"""Human preference posterior for the V24 strategy-curve learning loop.

Forty videos cannot identify thousands of physical cells independently.  The
learner therefore consumes fixed, interpretable projections of the complete
strategy vector and performs Bayesian pairwise logistic regression.  Its
posterior is used only to propose the next review batch; deployment remains
frozen until a Human-approved curve passes the release gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .v19_contracts import V19HumanRiskVector
from .v19_planner import V19PlanningError
from .v24_strategy import V24_STRATEGY_FEATURE_NAMES


V24_HUMAN_REVIEW_SCHEMA = "h3_v24_human_preference_review_v1"
V24_PREFERENCE_POSTERIOR_SCHEMA = "h3_v24_preference_posterior_v1"
V24_ISSUE_DIMENSIONS = (
    "prompt_adherence",
    "contact_causality",
    "trajectory_continuity",
    "temporal_clarity",
    "identity_binding",
    "object_geometry_consistency",
    "object_count_consistency",
    "audio_integrity",
    "anomaly",
)
V24_ISSUE_STATES = ("present", "absent", "not_reported")
V24_CONTEXT_INTERACTION_FEATURE_NAMES = (
    "spatial_x_forecast_fraction",
    "spatial_x_attention_approximation",
    "spatial_x_attention_causal",
    "spatial_x_attention_bridge",
    "spatial_x_attention_boundary",
    "horizon_x_forecast_fraction",
    "horizon_x_forecast_run",
    "horizon_x_attention_boundary",
    "horizon_x_attention_approximation",
)
V24_PREFERENCE_FEATURE_NAMES = (
    *V24_STRATEGY_FEATURE_NAMES,
    *V24_CONTEXT_INTERACTION_FEATURE_NAMES,
)


@dataclass(frozen=True, slots=True)
class V24ReviewCandidate:
    candidate_id: str
    comparison_group: str
    strategy_digest: str
    features: tuple[float, ...]
    video_path: str
    curve_profile_id: str
    acceleration: float
    workload_id: str
    latency_seconds: float | None = None
    width: int = 1280
    height: int = 736
    frames: int = 243
    packed_tokens: int = 67_535

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.comparison_group:
            raise V19PlanningError("V24 review candidate identity is incomplete")
        if len(self.strategy_digest) != 64:
            raise V19PlanningError("V24 review candidate lacks a strategy digest")
        if len(self.features) != len(V24_STRATEGY_FEATURE_NAMES) or any(
            not math.isfinite(value) for value in self.features
        ):
            raise V19PlanningError("V24 review candidate features are invalid")
        if not 0.0 <= self.acceleration <= 100.0:
            raise V19PlanningError("V24 review acceleration lies outside [0,100]")
        if self.latency_seconds is not None and self.latency_seconds <= 0.0:
            raise V19PlanningError("V24 review latency must be positive")
        if min(self.width, self.height, self.frames, self.packed_tokens) <= 0:
            raise V19PlanningError("V24 review workload context must be positive")

    @property
    def model_features(self) -> tuple[float, ...]:
        """Strategy features plus compact, centered workload interactions.

        Pure workload main effects would cancel in every within-workload duel.
        Only interactions are included, allowing the posterior to represent a
        strategy that changes behavior with spatial load or sequence horizon
        without learning prompt-specific identifiers.
        """

        lookup = dict(zip(V24_STRATEGY_FEATURE_NAMES, self.features))
        spatial = math.log(
            (self.width * self.height) / float(1280 * 736)
        )
        horizon = math.log(self.frames / 243.0)
        boundary = (
            lookup["attention_opening_exposure"]
            + lookup["attention_terminal_exposure"]
        )
        interactions = (
            spatial * lookup["forecast_fraction"],
            spatial * lookup["attention_approximation_mean"],
            spatial * lookup["attention_causal_30_43"],
            spatial * lookup["attention_bridge_44_45"],
            spatial * boundary,
            horizon * lookup["forecast_fraction"],
            horizon * lookup["forecast_run_quadratic"],
            horizon * boundary,
            horizon * lookup["attention_approximation_mean"],
        )
        return (*self.features, *interactions)


@dataclass(frozen=True, slots=True)
class V24HumanReview:
    batch_id: str
    candidates: tuple[V24ReviewCandidate, ...]
    rankings: Mapping[str, tuple[tuple[str, ...], ...]]
    overall_scores: Mapping[str, float]
    issues: Mapping[str, Mapping[str, str]]
    notes: Mapping[str, str]
    reviewer: str = "Human"
    schema_version: str = V24_HUMAN_REVIEW_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V24_HUMAN_REVIEW_SCHEMA or not self.batch_id:
            raise V19PlanningError("invalid V24 Human review schema")
        identifiers = tuple(row.candidate_id for row in self.candidates)
        if len(set(identifiers)) != len(identifiers):
            raise V19PlanningError("duplicate V24 review candidate")
        known = set(identifiers)
        groups = {row.comparison_group for row in self.candidates}
        for group, tiers in self.rankings.items():
            if group not in groups:
                raise V19PlanningError("V24 ranking names an unknown group")
            flattened = [candidate for tier in tiers for candidate in tier]
            if len(set(flattened)) != len(flattened) or not set(flattened) <= known:
                raise V19PlanningError("V24 ranking contains invalid candidates")
            if any(
                next(row for row in self.candidates if row.candidate_id == candidate)
                .comparison_group != group
                for candidate in flattened
            ):
                raise V19PlanningError("V24 ranking crosses comparison groups")
        for candidate, score in self.overall_scores.items():
            if candidate not in known or not math.isfinite(score) or not 0.0 <= score <= 100.0:
                raise V19PlanningError("V24 overall score must lie inside [0,100]")
        for candidate, dimensions in self.issues.items():
            if candidate not in known or any(
                dimension not in V24_ISSUE_DIMENSIONS or state not in V24_ISSUE_STATES
                for dimension, state in dimensions.items()
            ):
                raise V19PlanningError("invalid V24 issue annotation")
        if not set(self.notes) <= known:
            raise V19PlanningError("V24 notes name an unknown candidate")

    def pairwise_preferences(self) -> tuple[tuple[str, str, float], ...]:
        """Compile strict ranking tiers and score gaps into weighted duels."""

        result: dict[tuple[str, str], float] = {}
        for tiers in self.rankings.values():
            for better_index, better in enumerate(tiers):
                for worse in tiers[better_index + 1:]:
                    for winner in better:
                        for loser in worse:
                            result[(winner, loser)] = max(
                                1.0, result.get((winner, loser), 0.0)
                            )
        by_group: dict[str, list[V24ReviewCandidate]] = {}
        for candidate in self.candidates:
            by_group.setdefault(candidate.comparison_group, []).append(candidate)
        for rows in by_group.values():
            for left_index, left in enumerate(rows):
                if left.candidate_id not in self.overall_scores:
                    continue
                for right in rows[left_index + 1:]:
                    if right.candidate_id not in self.overall_scores:
                        continue
                    delta = (
                        self.overall_scores[left.candidate_id]
                        - self.overall_scores[right.candidate_id]
                    )
                    if abs(delta) < 2.0:
                        continue
                    winner, loser = (
                        (left.candidate_id, right.candidate_id)
                        if delta > 0.0
                        else (right.candidate_id, left.candidate_id)
                    )
                    weight = min(0.75, abs(delta) / 25.0)
                    result[(winner, loser)] = max(
                        weight, result.get((winner, loser), 0.0)
                    )
        return tuple(
            (winner, loser, weight)
            for (winner, loser), weight in sorted(result.items())
        )


@dataclass(frozen=True, slots=True)
class V24PreferencePosterior:
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    comparison_count: int
    review_batch_ids: tuple[str, ...]
    issue_heads: Mapping[str, Mapping[str, object]]
    schema_version: str = V24_PREFERENCE_POSTERIOR_SCHEMA

    def __post_init__(self) -> None:
        width = len(self.feature_names)
        if len(self.mean) != width or len(self.covariance) != width or any(
            len(row) != width for row in self.covariance
        ):
            raise V19PlanningError("invalid V24 posterior dimensions")

    def utility(self, features: Iterable[float]) -> tuple[float, float]:
        vector = np.asarray(tuple(features), dtype=np.float64)
        mean = np.asarray(self.mean, dtype=np.float64)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if vector.shape != mean.shape:
            raise V19PlanningError("posterior feature vector has the wrong shape")
        variance = max(0.0, float(vector @ covariance @ vector))
        return float(vector @ mean), math.sqrt(variance)

    def preference_probability(
        self,
        left: Iterable[float],
        right: Iterable[float],
    ) -> float:
        delta = np.asarray(tuple(left), dtype=np.float64) - np.asarray(
            tuple(right), dtype=np.float64
        )
        location = float(delta @ np.asarray(self.mean, dtype=np.float64))
        # Logistic-normal moment approximation keeps acquisition deterministic.
        variance = max(
            0.0,
            float(delta @ np.asarray(self.covariance) @ delta),
        )
        scaled = location / math.sqrt(1.0 + math.pi * variance / 8.0)
        return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, scaled))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feature_names": list(self.feature_names),
            "mean": list(self.mean),
            "covariance": [list(row) for row in self.covariance],
            "comparison_count": self.comparison_count,
            "review_batch_ids": list(self.review_batch_ids),
            "issue_heads": self.issue_heads,
        }


def _laplace_logistic(
    rows: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    prior_mean: np.ndarray,
    prior_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    width = prior_mean.shape[0]
    precision = np.eye(width, dtype=np.float64) / prior_variance
    beta = prior_mean.copy()
    if not len(rows):
        return beta, np.eye(width, dtype=np.float64) * prior_variance
    for _iteration in range(80):
        logits = np.clip(rows @ beta, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = rows.T @ (weights * (probabilities - labels))
        gradient += precision @ (beta - prior_mean)
        curvature = weights * probabilities * (1.0 - probabilities)
        hessian = rows.T @ (curvature[:, None] * rows) + precision
        update = np.linalg.solve(hessian, gradient)
        beta -= update
        if float(np.max(np.abs(update))) < 1.0e-8:
            break
    logits = np.clip(rows @ beta, -40.0, 40.0)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    curvature = weights * probabilities * (1.0 - probabilities)
    hessian = rows.T @ (curvature[:, None] * rows) + precision
    covariance = np.linalg.inv(hessian)
    return beta, covariance


def fit_v24_preference_posterior(
    reviews: Iterable[V24HumanReview],
    *,
    prior_variance: float = 4.0,
) -> V24PreferencePosterior:
    reviews = tuple(reviews)
    feature_names = tuple(V24_PREFERENCE_FEATURE_NAMES)
    feature_width = len(feature_names)
    # More exact compute is a weak positive prior; every approximation exposure
    # is weakly negative.  Forty Human videos, not this prior, determine scale.
    prior_mean = np.asarray((
        0.5,
        *(-0.25 for _ in range(len(V24_STRATEGY_FEATURE_NAMES) - 1)),
        *(0.0 for _ in V24_CONTEXT_INTERACTION_FEATURE_NAMES),
    ), dtype=np.float64)
    candidate_features: dict[str, np.ndarray] = {}
    comparisons: list[np.ndarray] = []
    comparison_weights: list[float] = []
    for review in reviews:
        for candidate in review.candidates:
            previous = candidate_features.get(candidate.candidate_id)
            features = np.asarray(candidate.model_features, dtype=np.float64)
            if previous is not None and not np.array_equal(previous, features):
                raise V19PlanningError("candidate id changed strategy features")
            candidate_features[candidate.candidate_id] = features
        for winner, loser, weight in review.pairwise_preferences():
            comparisons.append(
                candidate_features[winner] - candidate_features[loser]
            )
            comparison_weights.append(weight)
    matrix = (
        np.stack(comparisons)
        if comparisons
        else np.empty((0, feature_width), dtype=np.float64)
    )
    labels = np.ones(len(comparisons), dtype=np.float64)
    weights = np.asarray(comparison_weights, dtype=np.float64)
    mean, covariance = _laplace_logistic(
        matrix,
        labels,
        weights,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
    )

    issue_heads: dict[str, Mapping[str, object]] = {}
    for dimension in V24_ISSUE_DIMENSIONS:
        issue_rows: list[np.ndarray] = []
        issue_labels: list[float] = []
        for review in reviews:
            lookup = {row.candidate_id: row for row in review.candidates}
            for candidate_id, states in review.issues.items():
                state = states.get(dimension, "not_reported")
                if state == "not_reported":
                    continue
                issue_rows.append(np.concatenate((
                    np.ones(1, dtype=np.float64),
                    np.asarray(lookup[candidate_id].model_features, dtype=np.float64),
                )))
                issue_labels.append(float(state == "present"))
        issue_width = feature_width + 1
        issue_matrix = (
            np.stack(issue_rows)
            if issue_rows
            else np.empty((0, issue_width), dtype=np.float64)
        )
        issue_mean, issue_covariance = _laplace_logistic(
            issue_matrix,
            np.asarray(issue_labels, dtype=np.float64),
            np.ones(len(issue_labels), dtype=np.float64),
            prior_mean=np.zeros(issue_width, dtype=np.float64),
            prior_variance=prior_variance,
        )
        issue_heads[dimension] = {
            "reported_count": len(issue_labels),
            "mean": issue_mean.tolist(),
            "covariance": issue_covariance.tolist(),
        }
    return V24PreferencePosterior(
        feature_names=feature_names,
        mean=tuple(float(value) for value in mean),
        covariance=tuple(tuple(float(value) for value in row) for row in covariance),
        comparison_count=len(comparisons),
        review_batch_ids=tuple(review.batch_id for review in reviews),
        issue_heads=issue_heads,
    )


def load_v24_human_review(path: str | Path) -> V24HumanReview:
    source = Path(path)
    document = json.loads(source.read_text(encoding="utf-8"))
    candidates = tuple(V24ReviewCandidate(
        candidate_id=str(row["candidate_id"]),
        comparison_group=str(row["comparison_group"]),
        strategy_digest=str(row["strategy_digest"]),
        features=tuple(float(row["features"][name]) for name in V24_STRATEGY_FEATURE_NAMES),
        video_path=str(row["video_path"]),
        curve_profile_id=str(row["curve_profile_id"]),
        acceleration=float(row["acceleration"]),
        workload_id=str(row["workload_id"]),
        latency_seconds=(
            None if row.get("latency_seconds") is None else float(row["latency_seconds"])
        ),
        width=int((row.get("workload_context") or {}).get("width", 1280)),
        height=int((row.get("workload_context") or {}).get("height", 736)),
        frames=int((row.get("workload_context") or {}).get("frames", 243)),
        packed_tokens=int(
            (row.get("workload_context") or {}).get("packed_tokens", 67_535)
        ),
    ) for row in document["candidates"])
    feedback = document.get("human_feedback", {})
    rankings = {
        str(group): tuple(tuple(str(value) for value in tier) for tier in tiers)
        for group, tiers in feedback.get("rankings", {}).items()
    }
    return V24HumanReview(
        batch_id=str(document["batch_id"]),
        candidates=candidates,
        rankings=rankings,
        overall_scores={
            str(key): float(value)
            for key, value in feedback.get("overall_scores", {}).items()
        },
        issues={
            str(candidate): {
                str(dimension): str(state)
                for dimension, state in dimensions.items()
            }
            for candidate, dimensions in feedback.get("issues", {}).items()
        },
        notes={
            str(key): str(value)
            for key, value in feedback.get("notes", {}).items()
        },
        reviewer=str(feedback.get("reviewer", "Human")),
        schema_version=str(document["schema_version"]),
    )


__all__ = [
    "V24_HUMAN_REVIEW_SCHEMA",
    "V24_CONTEXT_INTERACTION_FEATURE_NAMES",
    "V24_ISSUE_DIMENSIONS",
    "V24_PREFERENCE_FEATURE_NAMES",
    "V24_PREFERENCE_POSTERIOR_SCHEMA",
    "V24HumanReview",
    "V24PreferencePosterior",
    "V24ReviewCandidate",
    "fit_v24_preference_posterior",
    "load_v24_human_review",
]
