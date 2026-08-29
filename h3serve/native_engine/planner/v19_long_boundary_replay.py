"""Replay a minimal set of Human-stable Attention boundary cells.

This module transfers exact physical Attention actions from a donor blueprint
onto selected Actual steps of a lower-cost long-video trajectory.  Selection
is fixed by sampling position and never inspects prompt semantics, seed,
reference content or scene motion.  It is intended for mechanism isolation:
Human approval of the donor does not automatically approve the hybrid.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .v19_candidates import V19CandidateBlueprint, v19_blueprint_execution_digest
from .v19_human_constraints import (
    V19HumanConstraintReport,
    require_v19_human_proposal_eligible,
)
from .v19_long_temporal_consolidation import (
    v19_long_temporal_consolidation_screening_policy,
)
from .v19_planner import V19ActionUse, V19PlanningError
from .v19_runtime_bridge import (
    blueprint_from_runtime_schedule,
    runtime_schedule_from_blueprint,
)


V19_LONG_BOUNDARY_REPLAY_SOURCE = "v19_long_boundary_replay_v1"


def _actual_steps(blueprint: V19CandidateBlueprint) -> tuple[int, ...]:
    return tuple(sorted({
        step
        for use in blueprint.action_uses
        if isinstance(use, V19ActionUse)
        for step in use.step_indices
    }))


def _canonical(runtime_action: str) -> str:
    return runtime_action.rsplit(":", 1)[-1]


def _counts(
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
class V19LongBoundaryReplaySpec:
    replay_actual_step_indices: tuple[int, ...] = (18, 19)
    replay_layer_indices: tuple[int, ...] = tuple(range(50))

    def __post_init__(self) -> None:
        if (
            not self.replay_actual_step_indices
            or tuple(sorted(set(self.replay_actual_step_indices)))
            != self.replay_actual_step_indices
            or any(step < 0 for step in self.replay_actual_step_indices)
        ):
            raise V19PlanningError(
                "boundary replay steps must be sorted, unique and non-negative"
            )
        if (
            not self.replay_layer_indices
            or tuple(sorted(set(self.replay_layer_indices)))
            != self.replay_layer_indices
            or any(layer < 0 or layer >= 50 for layer in self.replay_layer_indices)
        ):
            raise V19PlanningError(
                "boundary replay layers must be sorted unique values in [0, 50)"
            )


@dataclass(frozen=True, slots=True)
class V19LongBoundaryReplayResult:
    blueprint: V19CandidateBlueprint
    source_execution_digest: str
    donor_execution_digest: str
    actual_step_indices: tuple[int, ...]
    replay_actual_step_indices: tuple[int, ...]
    replay_layer_indices: tuple[int, ...]
    replayed_cells: int
    physically_changed_cells: int
    source_action_cell_counts: tuple[tuple[str, int], ...]
    candidate_action_cell_counts: tuple[tuple[str, int], ...]
    constraint_report: V19HumanConstraintReport


def build_v19_long_boundary_replay(
    source: V19CandidateBlueprint,
    donor: V19CandidateBlueprint,
    *,
    candidate_id: str,
    spec: V19LongBoundaryReplaySpec = V19LongBoundaryReplaySpec(),
) -> V19LongBoundaryReplayResult:
    """Replace selected source Actual cells with exact donor actions."""

    if not candidate_id:
        raise V19PlanningError("boundary replay candidate id cannot be empty")
    source_schedule = runtime_schedule_from_blueprint(source)
    donor_schedule = runtime_schedule_from_blueprint(donor)
    total_steps = max(step for step, _layer, _action in source_schedule) + 1
    donor_total_steps = max(step for step, _layer, _action in donor_schedule) + 1
    if donor_total_steps != total_steps:
        raise V19PlanningError("boundary replay requires equal total steps")
    source_actual = _actual_steps(source)
    donor_actual = set(_actual_steps(donor))
    replay = spec.replay_actual_step_indices
    replay_layers = spec.replay_layer_indices
    if any(step not in set(source_actual) for step in replay):
        raise V19PlanningError("boundary replay step must be a source Actual")
    if any(step not in donor_actual for step in replay):
        raise V19PlanningError("boundary replay step must be a donor Actual")

    source_cells = {
        (step, layer): action
        for step, layer, action in source_schedule
    }
    donor_cells = {
        (step, layer): action
        for step, layer, action in donor_schedule
    }
    changed = 0
    for step in replay:
        for layer in replay_layers:
            cell = (step, layer)
            try:
                donor_action = donor_cells[cell]
            except KeyError as error:
                raise V19PlanningError(
                    "donor blueprint omits a replay Attention cell"
                ) from error
            if source_cells[cell] != donor_action:
                changed += 1
            source_cells[cell] = donor_action

    candidate = blueprint_from_runtime_schedule(
        candidate_id=candidate_id,
        total_steps=total_steps,
        actual_step_indices=source_actual,
        attention_action_schedule=tuple(
            (step, layer, action)
            for (step, layer), action in sorted(source_cells.items())
        ),
        source=V19_LONG_BOUNDARY_REPLAY_SOURCE,
    )
    report = require_v19_human_proposal_eligible(
        candidate,
        v19_long_temporal_consolidation_screening_policy(total_steps),
    )
    candidate_schedule = runtime_schedule_from_blueprint(candidate)
    return V19LongBoundaryReplayResult(
        blueprint=candidate,
        source_execution_digest=v19_blueprint_execution_digest(source),
        donor_execution_digest=v19_blueprint_execution_digest(donor),
        actual_step_indices=source_actual,
        replay_actual_step_indices=replay,
        replay_layer_indices=replay_layers,
        replayed_cells=len(replay) * len(replay_layers),
        physically_changed_cells=changed,
        source_action_cell_counts=_counts(source_schedule, source_actual),
        candidate_action_cell_counts=_counts(
            candidate_schedule, source_actual
        ),
        constraint_report=report,
    )


__all__ = [
    "V19_LONG_BOUNDARY_REPLAY_SOURCE",
    "V19LongBoundaryReplayResult",
    "V19LongBoundaryReplaySpec",
    "build_v19_long_boundary_replay",
]
