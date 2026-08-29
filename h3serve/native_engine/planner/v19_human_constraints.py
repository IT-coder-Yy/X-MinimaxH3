"""Non-compensating Human-feedback constraints for V19 proposal screening.

Numerical Attention error is useful for allocating compute, but the V19
round-02 review demonstrated that more aggregate Attention compute can still
produce worse mouth articulation and speech pacing.  This module keeps those
outcome-level facts separate from numerical proxy scores.  It can reject an
already failed execution identity, low-density schedules outside the current
Human search floor, and trajectories with overly long Forecast runs.  Passing
these static checks only makes a blueprint eligible for an E2E experiment;
it never certifies the unevaluated Human gates.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

from .v19_candidates import V19CandidateBlueprint, v19_blueprint_execution_digest
from .v19_planner import V19ActionUse, V19ForecastUse, V19PlanningError


V19_HUMAN_CONSTRAINT_SCHEMA = "h3_v19_human_constraint_policy_v1"
V19_ROUND02_AV_MOTION_POLICY = "h3_v19_round02_av_motion_screening_v1"
V19_ROUND02_REJECTED_EXECUTION_DIGESTS = tuple(sorted((
    "16d034d86a78b0f435df463e99026b34073d93ffd381ff2638e2e91ba47a9eec",
    "f69d793247c6262f917c508d1940d46f9beaaa4c5dcc24bedb41346e5c6b836e",
)))
V19_ROUND02_REQUIRED_HUMAN_GATES = tuple(sorted((
    "normal_motion_causality",
    "speaking_mouth_clarity",
    "speech_pacing_and_dialogue_fit",
)))
V19_LONG_HORIZON_POLICY = "h3_v19_long_horizon_round188_screening_v1"
V19_LONG_HORIZON_REQUIRED_HUMAN_GATES = tuple(sorted((
    "identity_and_geometry_continuity",
    "long_horizon_motion_causality",
    "speaking_mouth_clarity",
    "speech_pacing_and_audio_integrity",
    "temporal_clarity_and_flicker",
)))


def _valid_sha256(value: str) -> bool:
    if len(value) != 64 or value == "0" * 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _keep_ratio(canonical_action: str) -> float:
    if canonical_action == "dense":
        return 1.0
    try:
        value = float(canonical_action.removeprefix("sparse_topk_"))
    except ValueError as error:
        raise V19PlanningError(
            f"cannot decode V19 keep ratio: {canonical_action}"
        ) from error
    if not 0.0 < value <= 1.0:
        raise V19PlanningError(
            f"V19 keep ratio lies outside (0, 1]: {canonical_action}"
        )
    return value


@dataclass(frozen=True, slots=True)
class V19HumanConstraintPolicy:
    """A versioned search policy derived from explicit Human outcomes.

    ``required_human_gates`` are deliberately not assigned proxy values.  They
    remain unevaluated until the exact generated artifact is reviewed.
    """

    policy_id: str
    minimum_actual_keep_ratio: float
    maximum_forecast_run: int
    rejected_execution_digests: tuple[str, ...] = ()
    required_human_gates: tuple[str, ...] = ()
    minimum_actual_steps: int = 0
    minimum_actual_fraction: float = 0.0
    required_actual_step_indices: tuple[int, ...] = ()
    minimum_layer_keep_ratios: tuple[float, ...] = ()
    schema_version: str = V19_HUMAN_CONSTRAINT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V19_HUMAN_CONSTRAINT_SCHEMA:
            raise V19PlanningError("unsupported V19 Human constraint schema")
        if not self.policy_id:
            raise V19PlanningError("V19 Human constraint policy id cannot be empty")
        if (
            not math.isfinite(self.minimum_actual_keep_ratio)
            or not 0.0 < self.minimum_actual_keep_ratio <= 1.0
        ):
            raise V19PlanningError(
                "V19 minimum actual keep ratio must lie inside (0, 1]"
            )
        if self.maximum_forecast_run <= 0:
            raise V19PlanningError("V19 maximum Forecast run must be positive")
        if self.minimum_actual_steps < 0:
            raise V19PlanningError("V19 minimum Actual count cannot be negative")
        if (
            not math.isfinite(self.minimum_actual_fraction)
            or not 0.0 <= self.minimum_actual_fraction <= 1.0
        ):
            raise V19PlanningError(
                "V19 minimum Actual fraction must lie inside [0, 1]"
            )
        if (
            tuple(sorted(set(self.required_actual_step_indices)))
            != self.required_actual_step_indices
            or any(step < 0 for step in self.required_actual_step_indices)
        ):
            raise V19PlanningError(
                "V19 required Actual steps must be sorted, unique and non-negative"
            )
        if self.minimum_layer_keep_ratios and (
            len(self.minimum_layer_keep_ratios) != 50
            or any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in self.minimum_layer_keep_ratios
            )
        ):
            raise V19PlanningError(
                "V19 layer keep floors require 50 finite values inside [0, 1]"
            )
        if tuple(sorted(set(self.rejected_execution_digests))) != (
            self.rejected_execution_digests
        ) or any(
            not _valid_sha256(value)
            for value in self.rejected_execution_digests
        ):
            raise V19PlanningError(
                "V19 rejected execution digests must be sorted unique SHA256 values"
            )
        if (
            tuple(sorted(set(self.required_human_gates)))
            != self.required_human_gates
            or any(not value for value in self.required_human_gates)
        ):
            raise V19PlanningError(
                "V19 required Human gates must be sorted, unique and non-empty"
            )


@dataclass(frozen=True, slots=True)
class V19HumanConstraintReport:
    policy_id: str
    candidate_id: str
    execution_digest: str
    proposal_eligible: bool
    release_eligible: bool
    rejection_reasons: tuple[str, ...]
    actual_step_indices: tuple[int, ...]
    forecast_runs: tuple[tuple[int, ...], ...]
    minimum_observed_keep_ratio: float
    action_cell_counts: tuple[tuple[str, int], ...]
    unevaluated_human_gates: tuple[str, ...]
    total_steps: int
    actual_step_fraction: float


def v19_round02_av_motion_screening_policy() -> V19HumanConstraintPolicy:
    """Return the frozen search floor from the v008/v009/v010 Human review.

    This is deliberately a proposal-screening policy, not a release policy.
    The source review was one prompt/seed and therefore cannot certify the
    retained schedules outside an actual E2E review case.
    """

    return V19HumanConstraintPolicy(
        policy_id=V19_ROUND02_AV_MOTION_POLICY,
        minimum_actual_keep_ratio=0.25,
        maximum_forecast_run=3,
        rejected_execution_digests=V19_ROUND02_REJECTED_EXECUTION_DIGESTS,
        required_human_gates=V19_ROUND02_REQUIRED_HUMAN_GATES,
    )


def v19_long_horizon_screening_policy(
    total_steps: int = 20,
) -> V19HumanConstraintPolicy:
    """Return the evidence-backed proposal floor for long-video schedules.

    Round188 is the fastest 720p15 trajectory in the checked-in evidence with
    a positive Human review: 12/8 at 20 sampling positions, no Forecast run
    longer than two, a five-step Actual opening and three Actual terminal
    positions.  V13's 6/14 trajectory was rejected for unformed geometry and
    identity drift.  Layers 30--43 and 45 retain the reviewed differentiated
    8--10% causal head rail.  These facts are only a search floor; a new
    runtime, prompt or artifact still requires fresh Human review.
    """

    if total_steps < 4:
        raise V19PlanningError(
            "V19 long-horizon screening requires at least four sampling steps"
        )
    opening_count = max(1, int(math.ceil(total_steps * 0.25)))
    terminal_count = min(3, total_steps - opening_count)
    required_actual = tuple(sorted(set((
        *range(opening_count),
        *range(total_steps - terminal_count, total_steps),
    ))))
    layer_floors = tuple(
        0.10 if layer in {*range(30, 44), 45} else 0.0625
        for layer in range(50)
    )
    return V19HumanConstraintPolicy(
        policy_id=V19_LONG_HORIZON_POLICY,
        minimum_actual_keep_ratio=0.0625,
        maximum_forecast_run=2,
        required_human_gates=V19_LONG_HORIZON_REQUIRED_HUMAN_GATES,
        minimum_actual_steps=int(math.ceil(total_steps * 0.60)),
        minimum_actual_fraction=0.60,
        required_actual_step_indices=required_actual,
        minimum_layer_keep_ratios=layer_floors,
    )


def evaluate_v19_human_constraints(
    blueprint: V19CandidateBlueprint,
    policy: V19HumanConstraintPolicy,
) -> V19HumanConstraintReport:
    """Screen one exact schedule without pretending to score Human quality."""

    digest = v19_blueprint_execution_digest(blueprint)
    actual_uses = tuple(
        use for use in blueprint.action_uses if isinstance(use, V19ActionUse)
    )
    forecast_uses = tuple(
        use for use in blueprint.action_uses if isinstance(use, V19ForecastUse)
    )
    actual_steps = tuple(sorted({
        step for use in actual_uses for step in use.step_indices
    }))
    forecast_runs = tuple(sorted(
        (
            tuple(use.composite_key.forecast_step_indices)
            for use in forecast_uses
        ),
        key=lambda run: run[0],
    ))
    keep_ratios = tuple(
        _keep_ratio(use.canonical_action)
        for use in actual_uses
    )
    minimum_observed = min(keep_ratios, default=1.0)
    counts = Counter()
    for use in actual_uses:
        counts[use.canonical_action] += (
            len(use.step_indices) * (use.layer_stop - use.layer_start)
        )

    covered_steps = set(actual_steps)
    for run in forecast_runs:
        covered_steps.update(run)
    total_steps = 1 + max(covered_steps, default=-1)
    actual_fraction = (
        0.0 if total_steps == 0 else len(actual_steps) / total_steps
    )

    reasons: list[str] = []
    if digest in set(policy.rejected_execution_digests):
        reasons.append("execution_digest_rejected_by_human_review")
    if minimum_observed < policy.minimum_actual_keep_ratio:
        reasons.append(
            "actual_keep_ratio_below_human_search_floor:"
            f"{minimum_observed:g}<{policy.minimum_actual_keep_ratio:g}"
        )
    longest_run = max((len(run) for run in forecast_runs), default=0)
    if longest_run > policy.maximum_forecast_run:
        reasons.append(
            "forecast_run_exceeds_human_search_limit:"
            f"{longest_run}>{policy.maximum_forecast_run}"
        )
    if len(actual_steps) < policy.minimum_actual_steps:
        reasons.append(
            "actual_count_below_human_search_floor:"
            f"{len(actual_steps)}<{policy.minimum_actual_steps}"
        )
    if actual_fraction + 1e-12 < policy.minimum_actual_fraction:
        reasons.append(
            "actual_fraction_below_human_search_floor:"
            f"{actual_fraction:g}<{policy.minimum_actual_fraction:g}"
        )
    missing_required = tuple(
        step for step in policy.required_actual_step_indices
        if step not in set(actual_steps)
    )
    if missing_required:
        reasons.append(
            "required_actual_steps_missing:"
            + ",".join(str(step) for step in missing_required)
        )
    if policy.minimum_layer_keep_ratios:
        violated_layers: set[int] = set()
        for use in actual_uses:
            keep = _keep_ratio(use.canonical_action)
            for layer in range(use.layer_start, use.layer_stop):
                if keep + 1e-12 < policy.minimum_layer_keep_ratios[layer]:
                    violated_layers.add(layer)
        if violated_layers:
            reasons.append(
                "actual_layer_keep_ratio_below_human_search_floor:"
                + ",".join(str(layer) for layer in sorted(violated_layers))
            )

    proposal_eligible = not reasons
    # Static proposal checks can never satisfy outcome-level Human gates.
    release_eligible = proposal_eligible and not policy.required_human_gates
    return V19HumanConstraintReport(
        policy_id=policy.policy_id,
        candidate_id=blueprint.candidate_id,
        execution_digest=digest,
        proposal_eligible=proposal_eligible,
        release_eligible=release_eligible,
        rejection_reasons=tuple(reasons),
        actual_step_indices=actual_steps,
        forecast_runs=forecast_runs,
        minimum_observed_keep_ratio=minimum_observed,
        action_cell_counts=tuple(sorted(counts.items())),
        unevaluated_human_gates=policy.required_human_gates,
        total_steps=total_steps,
        actual_step_fraction=actual_fraction,
    )


def require_v19_human_proposal_eligible(
    blueprint: V19CandidateBlueprint,
    policy: V19HumanConstraintPolicy,
) -> V19HumanConstraintReport:
    report = evaluate_v19_human_constraints(blueprint, policy)
    if not report.proposal_eligible:
        raise V19PlanningError(
            "V19 proposal violates Human constraints: "
            + "; ".join(report.rejection_reasons)
        )
    return report


__all__ = [
    "V19_HUMAN_CONSTRAINT_SCHEMA",
    "V19_ROUND02_AV_MOTION_POLICY",
    "V19_ROUND02_REJECTED_EXECUTION_DIGESTS",
    "V19_ROUND02_REQUIRED_HUMAN_GATES",
    "V19_LONG_HORIZON_POLICY",
    "V19_LONG_HORIZON_REQUIRED_HUMAN_GATES",
    "V19HumanConstraintPolicy",
    "V19HumanConstraintReport",
    "evaluate_v19_human_constraints",
    "require_v19_human_proposal_eligible",
    "v19_round02_av_motion_screening_policy",
    "v19_long_horizon_screening_policy",
]
