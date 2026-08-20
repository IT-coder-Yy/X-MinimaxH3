"""Fail-closed translation from a certified V19 frontier to hot-session state."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .v19_candidates import V19CandidateBlueprint
from .v19_contracts import V19_POLICY_ID
from .v19_frontier import V19AccelerationDecision, V19ReleaseFrontierCatalog
from .v19_planner import V19ActionUse, V19ForecastUse, V19PlanningError, V19WorkloadContext
from .v19_runtime_bridge import runtime_schedule_from_blueprint


V19_RUNTIME_SELECTION_SCHEMA = "h3_v19_runtime_selection_v1"


@dataclass(frozen=True, slots=True)
class V19RuntimeSelection:
    decision: V19AccelerationDecision
    actual_step_indices: tuple[int, ...]
    attention_action_schedule: tuple[tuple[int, int, str], ...]
    summary: dict[str, object]
    schema_version: str = V19_RUNTIME_SELECTION_SCHEMA

    def __post_init__(self) -> None:
        if not self.actual_step_indices:
            raise V19PlanningError("V19 runtime selection has no actual steps")
        if tuple(sorted(set(self.actual_step_indices))) != self.actual_step_indices:
            raise V19PlanningError("V19 actual steps must be sorted and unique")
        if self.decision.accelerated != bool(self.attention_action_schedule):
            raise V19PlanningError("V19 runtime selection is inconsistent")


def _compile_candidate(
    decision: V19AccelerationDecision,
    *,
    total_steps: int,
) -> tuple[tuple[int, ...], tuple[tuple[int, int, str], ...]]:
    candidate = decision.candidate
    if candidate is None:
        return tuple(range(total_steps)), ()
    actual = {
        step
        for use in candidate.action_uses
        if isinstance(use, V19ActionUse)
        for step in use.step_indices
    }
    forecast = {
        step
        for use in candidate.action_uses
        if isinstance(use, V19ForecastUse)
        for step in use.step_indices
    }
    if actual & forecast or actual | forecast != set(range(total_steps)):
        raise V19PlanningError(
            "V19 certified candidate does not partition the sigma trajectory"
        )
    actual_steps = tuple(sorted(actual))
    blueprint = V19CandidateBlueprint(
        candidate_id=candidate.candidate_id,
        action_uses=candidate.action_uses,
        terminal_debt=candidate.terminal_debt,
        maximum_debt=candidate.maximum_debt,
        source=candidate.source,
    )
    schedule = runtime_schedule_from_blueprint(blueprint)
    actual_cells = {
        (step, layer)
        for step, layer, _action in schedule
        if step in actual
    }
    forecast_cells = {
        (step, layer)
        for step, layer, _action in schedule
        if step in forecast
    }
    expected_actual = {
        (step, layer) for step in actual for layer in range(50)
    }
    expected_forecast = {
        (step, layer) for step in forecast for layer in range(3)
    }
    if actual_cells != expected_actual or forecast_cells != expected_forecast:
        raise V19PlanningError(
            "V19 runtime schedule does not cover its certified physical cells"
        )
    return actual_steps, schedule


class V19RuntimeSelector:
    """Select and compile one complete plan after exact token accounting."""

    def __init__(
        self,
        catalog: V19ReleaseFrontierCatalog,
        *,
        runtime_digest: str,
    ) -> None:
        if len(runtime_digest) != 64:
            raise V19PlanningError("V19 runtime selector requires a SHA256 identity")
        self.catalog = catalog
        self.runtime_digest = runtime_digest

    def select(
        self,
        *,
        workload: V19WorkloadContext,
        acceleration: float,
        required_actual_step_indices: tuple[int, ...] = (),
    ) -> V19RuntimeSelection:
        if workload.steps is None:
            raise V19PlanningError("V19 runtime selection requires total steps")
        decision = self.catalog.select(
            workload=workload,
            runtime_digest=self.runtime_digest,
            acceleration=acceleration,
            required_actual_step_indices=required_actual_step_indices,
        )
        actual, schedule = _compile_candidate(
            decision,
            total_steps=int(workload.steps),
        )
        summary: dict[str, object] = {
            "schema_version": V19_RUNTIME_SELECTION_SCHEMA,
            "policy_id": V19_POLICY_ID,
            "accelerated": decision.accelerated,
            "reason": decision.reason,
            "acceleration": decision.acceleration,
            "generation_request_digest": decision.generation_request_digest,
            "runtime_digest": decision.runtime_digest,
            "actual_step_indices": list(actual),
            "forecast_steps": int(workload.steps) - len(actual),
            "required_actual_step_indices": list(required_actual_step_indices),
        }
        if decision.candidate is not None:
            assert decision.certificate is not None
            summary.update({
                "candidate_id": decision.candidate.candidate_id,
                "candidate_digest": decision.candidate.digest,
                "execution_digest": decision.candidate.execution_digest,
                "envelope_digest": decision.certificate.envelope_digest,
                "certificate_digest": decision.certificate.certificate_digest,
                "cost_p90_ms": decision.candidate.predicted_cost_p90_ms,
                "risk_ucb": asdict(decision.candidate.risk_ucb),
            })
        return V19RuntimeSelection(
            decision=decision,
            actual_step_indices=actual,
            attention_action_schedule=schedule,
            summary=summary,
        )


__all__ = [
    "V19_RUNTIME_SELECTION_SCHEMA",
    "V19RuntimeSelection",
    "V19RuntimeSelector",
]
