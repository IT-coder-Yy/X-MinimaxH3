"""Materialize V19 schedule blueprints from exact physical evidence.

The optimizer is allowed to propose action schedules, but it is not allowed to
invent their latency, VRAM or Human risk.  This module is the only bridge from
an abstract blueprint to a candidate plan: costs come from registry-bound
physical catalogs and missing Human evidence becomes an explicit worst-case
risk vector rather than an optimistic zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

from .action_registry import ActionRegistry
from .v19_calibration import V19CalibrationCatalog, V19CalibrationError
from .v19_contracts import V19HumanRiskVector, V19TrajectoryDebt
from .v19_forecast_calibration import (
    V19ForecastCalibrationCatalog,
    V19ForecastCompositeKey,
)
from .v19_planner import (
    V19ActionUse,
    V19CandidatePlan,
    V19ForecastUse,
    V19PlanningError,
    V19PlanningRequest,
    validate_candidate_actions,
)
from .v19_risk_calibration import (
    V19RiskCalibrationCatalog,
    V19RiskCalibrationError,
)
from .v19_schedule_calibration import V19ScheduleCostCatalog


V19_CANDIDATE_BLUEPRINT_SCHEMA = "h3_v19_candidate_blueprint_v1"


@dataclass(frozen=True, slots=True)
class V19CandidateBlueprint:
    candidate_id: str
    action_uses: tuple[V19ActionUse | V19ForecastUse, ...]
    terminal_debt: V19TrajectoryDebt = V19TrajectoryDebt()
    maximum_debt: V19TrajectoryDebt = V19TrajectoryDebt()
    source: str = "v19_optimizer"

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.action_uses:
            raise V19PlanningError("V19 candidate blueprint cannot be empty")
        if not self.terminal_debt.within(self.maximum_debt):
            raise V19PlanningError("blueprint terminal debt exceeds its trajectory maximum")


def v19_blueprint_execution_digest(blueprint: V19CandidateBlueprint) -> str:
    """Return the same schedule identity used by E2E and Human artifacts."""

    return V19CandidatePlan(
        candidate_id=blueprint.candidate_id,
        action_uses=blueprint.action_uses,
        predicted_cost_p50_ms=0.0,
        predicted_cost_p90_ms=0.0,
        predicted_peak_vram_gib=0.0,
        risk_ucb=V19HumanRiskVector(*(1.0 for _ in range(7))),
        terminal_debt=blueprint.terminal_debt,
        maximum_debt=blueprint.maximum_debt,
        source=blueprint.source,
    ).execution_digest


def save_v19_candidate_blueprint(
    path: str | Path,
    blueprint: V19CandidateBlueprint,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "schema_version": V19_CANDIDATE_BLUEPRINT_SCHEMA,
        "execution_digest": v19_blueprint_execution_digest(blueprint),
        "blueprint": asdict(blueprint),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def load_v19_candidate_blueprint(path: str | Path) -> V19CandidateBlueprint:
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
        if document["schema_version"] != V19_CANDIDATE_BLUEPRINT_SCHEMA:
            raise V19PlanningError("unsupported V19 candidate blueprint schema")
        payload = document["blueprint"]
        uses: list[V19ActionUse | V19ForecastUse] = []
        for row in payload["action_uses"]:
            if "composite_key" in row:
                key = dict(row["composite_key"])
                key["forecast_step_indices"] = tuple(key["forecast_step_indices"])
                uses.append(V19ForecastUse(
                    action_id=str(row["action_id"]),
                    canonical_action=str(row.get("canonical_action", "forecast")),
                    composite_key=V19ForecastCompositeKey(**key),
                ))
            else:
                uses.append(V19ActionUse(
                    action_id=str(row["action_id"]),
                    canonical_action=str(row["canonical_action"]),
                    step_indices=tuple(int(value) for value in row["step_indices"]),
                    layer_start=int(row["layer_start"]),
                    layer_stop=int(row["layer_stop"]),
                ))
        blueprint = V19CandidateBlueprint(
            candidate_id=str(payload["candidate_id"]),
            action_uses=tuple(uses),
            terminal_debt=V19TrajectoryDebt(**payload["terminal_debt"]),
            maximum_debt=V19TrajectoryDebt(**payload["maximum_debt"]),
            source=str(payload.get("source", "v19_optimizer")),
        )
        expected_digest = str(document["execution_digest"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, V19PlanningError):
            raise
        raise V19PlanningError(f"invalid V19 candidate blueprint: {source}") from error
    if v19_blueprint_execution_digest(blueprint) != expected_digest:
        raise V19PlanningError("V19 candidate blueprint execution digest mismatch")
    return blueprint


@dataclass(frozen=True, slots=True)
class V19MaterializedCandidate:
    candidate: V19CandidatePlan
    end_to_end_cost_calibrated: bool
    human_risk_calibrated: bool


class V19CandidateFactory:
    """Exact repricing and risk binding for optimizer-produced blueprints."""

    def __init__(
        self,
        registry: ActionRegistry,
        attention_catalog: V19CalibrationCatalog,
        forecast_catalog: V19ForecastCalibrationCatalog | None = None,
        schedule_cost_catalog: V19ScheduleCostCatalog | None = None,
        risk_catalog: V19RiskCalibrationCatalog | None = None,
    ) -> None:
        self.registry = registry
        self.attention_catalog = attention_catalog
        self.forecast_catalog = forecast_catalog
        self.schedule_cost_catalog = schedule_cost_catalog
        self.risk_catalog = risk_catalog

    @staticmethod
    def _validate_complete_schedule(
        request: V19PlanningRequest,
        blueprint: V19CandidateBlueprint,
    ) -> None:
        workload = request.workload.to_calibration_workload()
        attention = tuple(
            use for use in blueprint.action_uses if isinstance(use, V19ActionUse)
        )
        forecasts = tuple(
            use for use in blueprint.action_uses if isinstance(use, V19ForecastUse)
        )
        expected_attention = {
            (step, layer)
            for step in workload.actual_step_indices
            for layer in range(50)
        }
        actual_attention = [
            (step, layer)
            for use in attention
            for step in use.step_indices
            for layer in range(use.layer_start, use.layer_stop)
        ]
        if (
            len(set(actual_attention)) != len(actual_attention)
            or set(actual_attention) != expected_attention
        ):
            raise V19PlanningError(
                "V19 blueprint must cover every actual-step Attention cell once"
            )
        expected_forecast = set(range(workload.steps)) - set(
            workload.actual_step_indices
        )
        actual_forecast = [
            step for use in forecasts for step in use.step_indices
        ]
        if (
            len(set(actual_forecast)) != len(actual_forecast)
            or set(actual_forecast) != expected_forecast
        ):
            raise V19PlanningError(
                "V19 blueprint must cover every forecast step once"
            )

    def materialize(
        self,
        request: V19PlanningRequest,
        blueprint: V19CandidateBlueprint,
        *,
        require_end_to_end_cost: bool = False,
        require_human_risk: bool = False,
    ) -> V19MaterializedCandidate:
        if request.runtime is None:
            raise V19PlanningError("candidate materialization requires a runtime fingerprint")
        validate_candidate_actions(
            self.registry,
            request.workload,
            V19CandidatePlan(
                candidate_id=blueprint.candidate_id,
                action_uses=blueprint.action_uses,
                predicted_cost_p50_ms=0.0,
                predicted_cost_p90_ms=0.0,
                predicted_peak_vram_gib=0.0,
                risk_ucb=V19HumanRiskVector(*(1.0 for _ in range(7))),
                terminal_debt=blueprint.terminal_debt,
                maximum_debt=blueprint.maximum_debt,
                source=blueprint.source,
            ),
            permit_evidence_bound_actions=True,
        )
        self._validate_complete_schedule(request, blueprint)
        workload = request.workload.to_calibration_workload()
        attention_uses = tuple(
            use for use in blueprint.action_uses if isinstance(use, V19ActionUse)
        )
        forecast_uses = tuple(
            use for use in blueprint.action_uses if isinstance(use, V19ForecastUse)
        )
        try:
            attention_cost = self.attention_catalog.estimate_schedule(
                attention_uses,
                workload=workload,
                runtime=request.runtime,
            )
            p50_ms = attention_cost.p50_ms
            p90_ms = attention_cost.p90_ms
            peak_vram_gib = attention_cost.peak_vram_gib
            evidence_ids = set(attention_cost.evidence_ids)
            if forecast_uses:
                if self.forecast_catalog is None:
                    raise V19PlanningError(
                        "forecast blueprint requires exact forecast calibration"
                    )
                for use in forecast_uses:
                    cost = self.forecast_catalog.estimate(
                        action_id=use.action_id,
                        key=use.composite_key,
                        workload=workload,
                        runtime=request.runtime,
                    )
                    p50_ms += cost.p50_ms
                    p90_ms += cost.p90_ms
                    peak_vram_gib = max(peak_vram_gib, cost.peak_vram_gib)
                    evidence_ids.add(cost.evidence_id)
        except V19CalibrationError as error:
            raise V19PlanningError(str(error)) from error

        conservative_risk = V19HumanRiskVector(*(1.0 for _ in range(7)))
        candidate = V19CandidatePlan(
            candidate_id=blueprint.candidate_id,
            action_uses=blueprint.action_uses,
            predicted_cost_p50_ms=p50_ms,
            predicted_cost_p90_ms=p90_ms,
            predicted_peak_vram_gib=peak_vram_gib,
            risk_ucb=conservative_risk,
            terminal_debt=blueprint.terminal_debt,
            maximum_debt=blueprint.maximum_debt,
            evidence_ids=tuple(sorted(evidence_ids)),
            source=blueprint.source,
        )
        end_to_end_calibrated = False
        if self.schedule_cost_catalog is not None:
            try:
                schedule_cost = self.schedule_cost_catalog.estimate(
                    execution_digest=candidate.execution_digest,
                    workload=workload,
                    runtime=request.runtime,
                    action_ids=(use.action_id for use in blueprint.action_uses),
                )
            except V19CalibrationError:
                if require_end_to_end_cost:
                    raise
            else:
                candidate = replace(
                    candidate,
                    predicted_cost_p50_ms=schedule_cost.p50_ms,
                    predicted_cost_p90_ms=schedule_cost.p90_ms,
                    predicted_peak_vram_gib=schedule_cost.peak_vram_gib,
                    evidence_ids=tuple(sorted((
                        *candidate.evidence_ids,
                        schedule_cost.evidence_id,
                    ))),
                )
                end_to_end_calibrated = True
        elif require_end_to_end_cost:
            raise V19PlanningError(
                "candidate lacks an end-to-end schedule calibration catalog"
            )
        risk_calibrated = False
        if self.risk_catalog is not None:
            try:
                risk_ucb, risk_id = self.risk_catalog.estimate(
                    execution_digest=candidate.execution_digest,
                    workload=workload,
                    runtime=request.runtime,
                    action_ids=(use.action_id for use in blueprint.action_uses),
                )
            except V19RiskCalibrationError:
                if require_human_risk:
                    raise
            else:
                candidate = replace(
                    candidate,
                    risk_ucb=risk_ucb,
                    evidence_ids=tuple(sorted((*candidate.evidence_ids, risk_id))),
                )
                risk_calibrated = True
        elif require_human_risk:
            raise V19PlanningError("candidate lacks a Human-risk calibration catalog")
        return V19MaterializedCandidate(
            candidate=candidate,
            end_to_end_cost_calibrated=end_to_end_calibrated,
            human_risk_calibrated=risk_calibrated,
        )


__all__ = [
    "V19_CANDIDATE_BLUEPRINT_SCHEMA",
    "V19CandidateBlueprint",
    "V19CandidateFactory",
    "V19MaterializedCandidate",
    "load_v19_candidate_blueprint",
    "save_v19_candidate_blueprint",
    "v19_blueprint_execution_digest",
]
