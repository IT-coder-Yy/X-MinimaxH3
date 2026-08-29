"""Task-independent temporal recovery shields for long V19 trajectories.

The Human long-video review found that v014b preserves the main subject but
still exhibits subtle peripheral/background jitter, deformation and transient
bright spots.  The matched v009 quality anchor spends substantially more
Attention on the Actual evaluation following each long Forecast run.  This
module encodes that mechanism without inspecting prompt text, seed, scene
category or spatial saliency.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .v19_candidates import V19CandidateBlueprint, v19_blueprint_execution_digest
from .v19_human_constraints import (
    V19HumanConstraintReport,
    require_v19_human_proposal_eligible,
    v19_long_horizon_screening_policy,
)
from .v19_planner import V19ActionUse, V19PlanningError
from .v19_runtime_bridge import (
    blueprint_from_runtime_schedule,
    runtime_schedule_from_blueprint,
)


V19_LONG_TEMPORAL_STABILITY_SOURCE = "v19_long_temporal_stability_v1"
_CANONICAL_KEEP = {
    "sparse_topk_0.0625": 0.0625,
    "sparse_topk_0.1": 0.1,
    "sparse_topk_0.25": 0.25,
    "sparse_topk_0.5": 0.5,
    "dense": 1.0,
}


def _validate_action(action: str) -> str:
    if action not in _CANONICAL_KEEP:
        raise V19PlanningError(f"unsupported temporal stability action: {action}")
    return action


def _canonical(runtime_action: str) -> str:
    return runtime_action.rsplit(":", 1)[-1]


def _runtime_action(canonical: str) -> str:
    return "dense" if canonical == "dense" else f"frontier:{canonical}"


def _actual_steps(blueprint: V19CandidateBlueprint) -> tuple[int, ...]:
    return tuple(sorted({
        step
        for use in blueprint.action_uses
        if isinstance(use, V19ActionUse)
        for step in use.step_indices
    }))


def _forecast_runs(
    total_steps: int,
    actual_steps: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    actual = set(actual_steps)
    runs: list[tuple[int, ...]] = []
    current: list[int] = []
    for step in range(total_steps):
        if step in actual:
            if current:
                runs.append(tuple(current))
                current = []
        else:
            current.append(step)
    if current:
        runs.append(tuple(current))
    return tuple(runs)


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


@dataclass(frozen=True, slots=True)
class V19LongTemporalStabilitySpec:
    """One content-independent recovery allocation.

    ``minimum_forecast_run`` locates Actual corrections solely from the fixed
    trajectory. ``structural_layers`` is an optional second diagnostic rail;
    it defaults empty so the first experiment isolates recovery timing.
    """

    minimum_forecast_run: int = 2
    recovery_action: str = "sparse_topk_0.25"
    structural_layers: tuple[int, ...] = ()
    structural_action: str = "sparse_topk_0.25"

    def __post_init__(self) -> None:
        if self.minimum_forecast_run <= 0:
            raise V19PlanningError("minimum Forecast run must be positive")
        _validate_action(self.recovery_action)
        _validate_action(self.structural_action)
        if (
            tuple(sorted(set(self.structural_layers))) != self.structural_layers
            or any(layer < 0 or layer >= 50 for layer in self.structural_layers)
        ):
            raise V19PlanningError(
                "structural layers must be sorted unique values in [0, 50)"
            )


@dataclass(frozen=True, slots=True)
class V19LongTemporalStabilityResult:
    blueprint: V19CandidateBlueprint
    source_execution_digest: str
    actual_step_indices: tuple[int, ...]
    forecast_runs: tuple[tuple[int, ...], ...]
    recovery_actual_step_indices: tuple[int, ...]
    source_action_cell_counts: tuple[tuple[str, int], ...]
    candidate_action_cell_counts: tuple[tuple[str, int], ...]
    recovery_upgraded_cells: int
    structural_upgraded_cells: int
    constraint_report: V19HumanConstraintReport


def build_v19_long_temporal_stability_shield(
    source: V19CandidateBlueprint,
    *,
    candidate_id: str,
    spec: V19LongTemporalStabilitySpec = V19LongTemporalStabilitySpec(),
) -> V19LongTemporalStabilityResult:
    """Strengthen post-Forecast corrections without changing the trajectory."""

    if not candidate_id:
        raise V19PlanningError("temporal stability candidate id cannot be empty")
    source_schedule = runtime_schedule_from_blueprint(source)
    total_steps = max(step for step, _layer, _action in source_schedule) + 1
    actual_steps = _actual_steps(source)
    actual = set(actual_steps)
    forecast_runs = _forecast_runs(total_steps, actual_steps)
    recovery_steps = tuple(
        run[-1] + 1
        for run in forecast_runs
        if len(run) >= spec.minimum_forecast_run and run[-1] + 1 in actual
    )
    if not recovery_steps:
        raise V19PlanningError(
            "trajectory has no post-Forecast Actual matching the recovery spec"
        )

    schedule = {
        (step, layer): action for step, layer, action in source_schedule
    }
    recovery = set(recovery_steps)
    structural = set(spec.structural_layers)
    recovery_upgraded = 0
    structural_upgraded = 0
    for step in actual_steps:
        for layer in range(50):
            cell = (step, layer)
            current = _canonical(schedule[cell])
            desired: str | None = None
            reason: str | None = None
            if step in recovery:
                desired = spec.recovery_action
                reason = "recovery"
            if layer in structural and (
                desired is None
                or _CANONICAL_KEEP[spec.structural_action]
                > _CANONICAL_KEEP[desired]
            ):
                desired = spec.structural_action
                reason = "structural"
            if desired is None or (
                _CANONICAL_KEEP[desired] <= _CANONICAL_KEEP[current]
            ):
                continue
            schedule[cell] = _runtime_action(desired)
            if reason == "recovery":
                recovery_upgraded += 1
            else:
                structural_upgraded += 1

    candidate = blueprint_from_runtime_schedule(
        candidate_id=candidate_id,
        total_steps=total_steps,
        actual_step_indices=actual_steps,
        attention_action_schedule=tuple(
            (step, layer, action)
            for (step, layer), action in sorted(schedule.items())
        ),
        source=V19_LONG_TEMPORAL_STABILITY_SOURCE,
    )
    if _actual_steps(candidate) != actual_steps:
        raise V19PlanningError("temporal shield changed the Actual trajectory")
    report = require_v19_human_proposal_eligible(
        candidate,
        v19_long_horizon_screening_policy(total_steps),
    )
    return V19LongTemporalStabilityResult(
        blueprint=candidate,
        source_execution_digest=v19_blueprint_execution_digest(source),
        actual_step_indices=actual_steps,
        forecast_runs=forecast_runs,
        recovery_actual_step_indices=recovery_steps,
        source_action_cell_counts=_cell_counts(source_schedule, actual_steps),
        candidate_action_cell_counts=_cell_counts(
            runtime_schedule_from_blueprint(candidate), actual_steps
        ),
        recovery_upgraded_cells=recovery_upgraded,
        structural_upgraded_cells=structural_upgraded,
        constraint_report=report,
    )


__all__ = [
    "V19_LONG_TEMPORAL_STABILITY_SOURCE",
    "V19LongTemporalStabilityResult",
    "V19LongTemporalStabilitySpec",
    "build_v19_long_temporal_stability_shield",
]
