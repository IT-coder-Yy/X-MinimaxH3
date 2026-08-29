"""Consolidate long-video Actual refreshes onto stable temporal anchors.

Human review found that the v009 20-step trajectory has cleaner peripheral
temporal behaviour than the more frequent Round188 refresh trajectory, even
though v009 uses longer Forecast runs.  This module isolates that trajectory
mechanism: it re-anchors an existing low-cost Attention rail without reading
prompt text, seed, reference content or scene category.

The result is an experimental proposal.  Reusing a reviewed trajectory does
not transfer Human approval to a new Attention allocation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .v19_candidates import V19CandidateBlueprint, v19_blueprint_execution_digest
from .v19_human_constraints import (
    V19_LONG_HORIZON_REQUIRED_HUMAN_GATES,
    V19HumanConstraintPolicy,
    V19HumanConstraintReport,
    require_v19_human_proposal_eligible,
)
from .v19_long_horizon import ROUND188_CAUSAL_LAYERS
from .v19_planner import V19ActionUse, V19PlanningError
from .v19_runtime_bridge import (
    ROUND229_FORECAST_ANCHOR,
    blueprint_from_runtime_schedule,
    contiguous_forecast_runs,
    runtime_schedule_from_blueprint,
)


V19_LONG_TEMPORAL_CONSOLIDATION_SOURCE = (
    "v19_long_temporal_consolidation_v1"
)
V19_LONG_TEMPORAL_CONSOLIDATION_POLICY = (
    "h3_v19_long_temporal_consolidation_screening_v1"
)
V19_LONG_TEMPORAL_CONSOLIDATED_20_STEP_ACTUALS = (
    0, 1, 2, 3, 4, 8, 12, 15, 18, 19,
)
_CANONICAL_KEEP = {
    "sparse_topk_0.0625": 0.0625,
    "sparse_topk_0.1": 0.1,
    "sparse_topk_0.25": 0.25,
    "sparse_topk_0.5": 0.5,
    "dense": 1.0,
}


def _actual_steps(blueprint: V19CandidateBlueprint) -> tuple[int, ...]:
    return tuple(sorted({
        step
        for use in blueprint.action_uses
        if isinstance(use, V19ActionUse)
        for step in use.step_indices
    }))


def _canonical(runtime_action: str) -> str:
    return runtime_action.rsplit(":", 1)[-1]


def _runtime_action(canonical: str) -> str:
    return "dense" if canonical == "dense" else f"frontier:{canonical}"


def _cell_counts(
    schedule: tuple[tuple[int, int, str], ...],
    actual_steps: tuple[int, ...],
) -> tuple[tuple[str, int], ...]:
    actual = set(actual_steps)
    return tuple(sorted(Counter(
        _canonical(action)
        for step, _layer, action in schedule
        if step in actual
    ).items()))


def v19_long_temporal_consolidation_screening_policy(
    total_steps: int,
) -> V19HumanConstraintPolicy:
    """Return a proposal-only floor bounded by the reviewed v009 trajectory.

    v009 authorizes testing a 10/10 trajectory with Forecast runs of at most
    three at 20 steps.  The lower Attention rail is new, so every audiovisual
    gate remains unevaluated and ``release_eligible`` stays false.
    """

    if total_steps != 20:
        raise V19PlanningError(
            "temporal consolidation is currently calibrated only for 20 steps"
        )
    layer_floors = tuple(
        0.10 if layer in set(ROUND188_CAUSAL_LAYERS) else 0.0625
        for layer in range(50)
    )
    return V19HumanConstraintPolicy(
        policy_id=V19_LONG_TEMPORAL_CONSOLIDATION_POLICY,
        minimum_actual_keep_ratio=0.0625,
        maximum_forecast_run=3,
        required_human_gates=V19_LONG_HORIZON_REQUIRED_HUMAN_GATES,
        minimum_actual_steps=10,
        minimum_actual_fraction=0.50,
        required_actual_step_indices=(0, 1, 2, 3, 4, 18, 19),
        minimum_layer_keep_ratios=layer_floors,
    )


@dataclass(frozen=True, slots=True)
class V19LongTemporalConsolidationSpec:
    """A content-independent target trajectory for one total-step envelope."""

    target_actual_step_indices: tuple[int, ...] = (
        V19_LONG_TEMPORAL_CONSOLIDATED_20_STEP_ACTUALS
    )
    maximum_forecast_run: int = 3
    recovery_action: str | None = None
    minimum_recovery_forecast_run: int = 2

    def __post_init__(self) -> None:
        if (
            not self.target_actual_step_indices
            or tuple(sorted(set(self.target_actual_step_indices)))
            != self.target_actual_step_indices
            or any(step < 0 for step in self.target_actual_step_indices)
        ):
            raise V19PlanningError(
                "target Actual steps must be sorted, unique and non-negative"
            )
        if self.maximum_forecast_run <= 0:
            raise V19PlanningError("maximum Forecast run must be positive")
        if self.recovery_action not in (None, *_CANONICAL_KEEP):
            raise V19PlanningError("unsupported consolidated recovery action")
        if self.minimum_recovery_forecast_run <= 0:
            raise V19PlanningError(
                "minimum recovery Forecast run must be positive"
            )


@dataclass(frozen=True, slots=True)
class V19LongTemporalConsolidationResult:
    blueprint: V19CandidateBlueprint
    source_execution_digest: str
    source_actual_step_indices: tuple[int, ...]
    actual_step_indices: tuple[int, ...]
    forecast_runs: tuple[tuple[int, ...], ...]
    cloned_from_source_actual: tuple[tuple[int, int], ...]
    recovery_actual_step_indices: tuple[int, ...]
    recovery_upgraded_cells: int
    source_action_cell_counts: tuple[tuple[str, int], ...]
    candidate_action_cell_counts: tuple[tuple[str, int], ...]
    constraint_report: V19HumanConstraintReport


def build_v19_long_temporal_consolidation(
    source: V19CandidateBlueprint,
    *,
    candidate_id: str,
    spec: V19LongTemporalConsolidationSpec = (
        V19LongTemporalConsolidationSpec()
    ),
) -> V19LongTemporalConsolidationResult:
    """Re-anchor a source Attention rail onto fewer temporal corrections."""

    if not candidate_id:
        raise V19PlanningError("temporal consolidation candidate id cannot be empty")
    source_schedule = runtime_schedule_from_blueprint(source)
    total_steps = max(step for step, _layer, _action in source_schedule) + 1
    source_actual = _actual_steps(source)
    target_actual = spec.target_actual_step_indices
    if target_actual[0] != 0 or target_actual[-1] != total_steps - 1:
        raise V19PlanningError(
            "target trajectory must retain first and final Actual steps"
        )
    if any(step >= total_steps for step in target_actual):
        raise V19PlanningError("target Actual step lies outside the trajectory")
    runs = contiguous_forecast_runs(
        total_steps=total_steps,
        actual_step_indices=target_actual,
    )
    longest = max((len(run) for run in runs), default=0)
    if longest > spec.maximum_forecast_run:
        raise V19PlanningError(
            f"consolidated Forecast run {longest} exceeds limit "
            f"{spec.maximum_forecast_run}"
        )

    source_cells = {
        (step, layer): action
        for step, layer, action in source_schedule
        if step in set(source_actual)
    }
    expected_source = {
        (step, layer) for step in source_actual for layer in range(50)
    }
    if set(source_cells) != expected_source:
        raise V19PlanningError(
            "source blueprint does not cover every source Actual cell"
        )

    # The nearest correction phase transfers only the source Attention rail.
    # Ties prefer the preceding source Actual to avoid borrowing future-phase
    # terminal protection into an earlier denoising phase.
    clone_map = {
        target: min(
            source_actual,
            key=lambda source_step: (
                abs(source_step - target),
                source_step > target,
                source_step,
            ),
        )
        for target in target_actual
    }
    schedule: dict[tuple[int, int], str] = {}
    for target, source_step in clone_map.items():
        for layer in range(50):
            schedule[(target, layer)] = source_cells[(source_step, layer)]
    for run in runs:
        for step in run:
            for layer in range(3):
                schedule[(step, layer)] = ROUND229_FORECAST_ANCHOR

    recovery_steps: tuple[int, ...] = ()
    recovery_upgraded = 0
    if spec.recovery_action is not None:
        recovery_steps = tuple(
            run[-1] + 1
            for run in runs
            if len(run) >= spec.minimum_recovery_forecast_run
        )
        for step in recovery_steps:
            for layer in range(50):
                cell = (step, layer)
                current = _canonical(schedule[cell])
                if (
                    _CANONICAL_KEEP[spec.recovery_action]
                    <= _CANONICAL_KEEP[current]
                ):
                    continue
                schedule[cell] = _runtime_action(spec.recovery_action)
                recovery_upgraded += 1

    candidate = blueprint_from_runtime_schedule(
        candidate_id=candidate_id,
        total_steps=total_steps,
        actual_step_indices=target_actual,
        attention_action_schedule=tuple(
            (step, layer, action)
            for (step, layer), action in sorted(schedule.items())
        ),
        source=V19_LONG_TEMPORAL_CONSOLIDATION_SOURCE,
    )
    report = require_v19_human_proposal_eligible(
        candidate,
        v19_long_temporal_consolidation_screening_policy(total_steps),
    )
    candidate_schedule = runtime_schedule_from_blueprint(candidate)
    return V19LongTemporalConsolidationResult(
        blueprint=candidate,
        source_execution_digest=v19_blueprint_execution_digest(source),
        source_actual_step_indices=source_actual,
        actual_step_indices=target_actual,
        forecast_runs=runs,
        cloned_from_source_actual=tuple(sorted(clone_map.items())),
        recovery_actual_step_indices=recovery_steps,
        recovery_upgraded_cells=recovery_upgraded,
        source_action_cell_counts=_cell_counts(source_schedule, source_actual),
        candidate_action_cell_counts=_cell_counts(
            candidate_schedule, target_actual
        ),
        constraint_report=report,
    )


__all__ = [
    "V19_LONG_TEMPORAL_CONSOLIDATED_20_STEP_ACTUALS",
    "V19_LONG_TEMPORAL_CONSOLIDATION_POLICY",
    "V19_LONG_TEMPORAL_CONSOLIDATION_SOURCE",
    "V19LongTemporalConsolidationResult",
    "V19LongTemporalConsolidationSpec",
    "build_v19_long_temporal_consolidation",
    "v19_long_temporal_consolidation_screening_policy",
]
