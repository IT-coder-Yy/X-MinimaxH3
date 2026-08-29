"""Strict Pareto and certificate primitives for the V19 H3 planner.

This is the first V19 planning layer.  It intentionally accepts only actions
whose exact physical implementation is planner-eligible in Action Registry.
It does not yet invent Human risk values or extrapolate missing calibration.
Those artifacts are prerequisites supplied by later V19 calibration stages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from typing import Iterable

from .action_registry import ActionKind, ActionRegistry, ActionRegistryError, EvidenceStatus
from .v19_calibration import (
    V19CalibrationCatalog,
    V19CalibrationError,
    V19CalibrationWorkload,
    V19RuntimeFingerprint,
)
from .v19_contracts import (
    V19_CONTRACT_SCHEMA,
    V19HumanRiskVector,
    V19ParetoObjectiveVector,
    V19TrajectoryDebt,
    V19_INPUT_CAPABILITY,
)
from .v19_forecast_calibration import (
    V19ForecastCalibrationCatalog,
    V19ForecastCompositeKey,
)
from .v19_risk_calibration import (
    V19RiskCalibrationCatalog,
    V19RiskCalibrationError,
)
from .v19_schedule_calibration import V19ScheduleCostCatalog


V19_PLAN_CERTIFICATE_SCHEMA = "h3_v19_plan_certificate_v1"


class V19PlanningError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class V19WorkloadContext:
    model_variant: str
    service_family: str
    packed_tokens: int
    condition_count: int
    reference_images: int = 0
    reference_audio: int = 0
    reference_videos: int = 0
    device_arch: str = "sm89"
    width: int | None = None
    height: int | None = None
    frames: int | None = None
    steps: int | None = None
    actual_step_indices: tuple[int, ...] = ()
    sampler: str = "res_multistep"
    scheduler: str = "simple"

    def __post_init__(self) -> None:
        if self.packed_tokens <= 0 or self.condition_count < 0:
            raise V19PlanningError("invalid V19 workload dimensions")
        if not V19_INPUT_CAPABILITY.accepts(
            service_family=self.service_family,
            model_variant=self.model_variant,
            reference_images=self.reference_images,
            reference_audio=self.reference_audio,
            reference_videos=self.reference_videos,
        ):
            raise V19PlanningError("request exceeds the Dense input capability contract")
        optional_geometry = (self.width, self.height, self.frames, self.steps)
        if any(value is not None and value <= 0 for value in optional_geometry):
            raise V19PlanningError("V19 workload geometry must be positive")
        if self.actual_step_indices and (
            tuple(sorted(set(self.actual_step_indices))) != self.actual_step_indices
            or self.actual_step_indices[0] < 0
            or self.steps is None
            or self.actual_step_indices[-1] >= self.steps
        ):
            raise V19PlanningError("V19 actual step schedule is invalid")

    @property
    def digest(self) -> str:
        return _sha256(asdict(self))

    def to_calibration_workload(self) -> V19CalibrationWorkload:
        if any(value is None for value in (
            self.width, self.height, self.frames, self.steps
        )) or not self.actual_step_indices:
            raise V19PlanningError(
                "strict physical planning requires exact geometry and actual steps"
            )
        return V19CalibrationWorkload(
            model_variant=self.model_variant,
            service_family=self.service_family,
            width=int(self.width),
            height=int(self.height),
            frames=int(self.frames),
            packed_tokens=self.packed_tokens,
            condition_count=self.condition_count,
            steps=int(self.steps),
            actual_step_indices=self.actual_step_indices,
            device_arch=self.device_arch,
            sampler=self.sampler,
            scheduler=self.scheduler,
        )


@dataclass(frozen=True, slots=True)
class V19ActionUse:
    action_id: str
    canonical_action: str
    step_indices: tuple[int, ...]
    layer_start: int = 0
    layer_stop: int = 50

    def __post_init__(self) -> None:
        if not self.action_id or not self.canonical_action:
            raise V19PlanningError("action use requires stable action identifiers")
        if not self.step_indices:
            raise V19PlanningError("action use requires at least one denoising step")
        if tuple(sorted(set(self.step_indices))) != self.step_indices:
            raise V19PlanningError("action-use steps must be sorted and unique")
        if self.step_indices[0] < 0:
            raise V19PlanningError("action-use step cannot be negative")
        if not 0 <= self.layer_start < self.layer_stop <= 50:
            raise V19PlanningError("action-use layer range is invalid")


@dataclass(frozen=True, slots=True)
class V19ForecastUse:
    action_id: str
    composite_key: V19ForecastCompositeKey
    canonical_action: str = "forecast"

    def __post_init__(self) -> None:
        if not self.action_id or self.canonical_action != "forecast":
            raise V19PlanningError("invalid V19 forecast action use")

    @property
    def step_indices(self) -> tuple[int, ...]:
        return self.composite_key.forecast_step_indices


@dataclass(frozen=True, slots=True)
class V19CandidatePlan:
    candidate_id: str
    action_uses: tuple[V19ActionUse | V19ForecastUse, ...]
    predicted_cost_p50_ms: float
    predicted_cost_p90_ms: float
    risk_ucb: V19HumanRiskVector
    predicted_peak_vram_gib: float = 0.0
    terminal_debt: V19TrajectoryDebt = V19TrajectoryDebt()
    maximum_debt: V19TrajectoryDebt = V19TrajectoryDebt()
    evidence_ids: tuple[str, ...] = ()
    source: str = "v19_optimizer"

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.action_uses:
            raise V19PlanningError("candidate requires an id and action schedule")
        if (
            not math.isfinite(self.predicted_cost_p50_ms)
            or not math.isfinite(self.predicted_cost_p90_ms)
            or self.predicted_cost_p50_ms < 0.0
            or self.predicted_cost_p90_ms < self.predicted_cost_p50_ms
        ):
            raise V19PlanningError("candidate p50/p90 cost envelope is invalid")
        if (
            not math.isfinite(self.predicted_peak_vram_gib)
            or self.predicted_peak_vram_gib < 0.0
        ):
            raise V19PlanningError("candidate peak VRAM must be finite and non-negative")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise V19PlanningError("candidate evidence ids must be unique")
        if not self.terminal_debt.within(self.maximum_debt):
            raise V19PlanningError("terminal debt cannot exceed trajectory maximum debt")

    @property
    def digest(self) -> str:
        document = asdict(self)
        return _sha256(document)

    @property
    def execution_digest(self) -> str:
        """Identity of the executed schedule, independent of claimed metrics."""

        return _sha256({
            "action_uses": [asdict(use) for use in self.action_uses],
            "terminal_debt": asdict(self.terminal_debt),
            "maximum_debt": asdict(self.maximum_debt),
        })

    @property
    def pareto_objective(self) -> V19ParetoObjectiveVector:
        return V19ParetoObjectiveVector(
            cost_p90_ms=self.predicted_cost_p90_ms,
            peak_vram_gib=self.predicted_peak_vram_gib,
            human_risk=self.risk_ucb,
            terminal_debt=self.terminal_debt,
            maximum_debt=self.maximum_debt,
        )


@dataclass(frozen=True, slots=True)
class V19PlanningRequest:
    workload: V19WorkloadContext
    maximum_cost_p90_ms: float
    risk_limits: V19HumanRiskVector
    runtime: V19RuntimeFingerprint | None = None
    maximum_peak_vram_gib: float = 24.0
    debt_limits: V19TrajectoryDebt = V19TrajectoryDebt(
        consecutive_forecasts=20,
        forecast_debt=1.0e9,
        sparse_mass_deficit=1.0e9,
        audio_debt=1.0e9,
    )

    def __post_init__(self) -> None:
        if not math.isfinite(self.maximum_cost_p90_ms) or self.maximum_cost_p90_ms <= 0:
            raise V19PlanningError("V19 budget must be finite and positive")
        if (
            not math.isfinite(self.maximum_peak_vram_gib)
            or self.maximum_peak_vram_gib <= 0.0
        ):
            raise V19PlanningError("V19 peak-VRAM budget must be finite and positive")


@dataclass(frozen=True, slots=True)
class V19PlanCertificate:
    registry_digest: str
    workload_digest: str
    runtime_digest: str
    candidate_digest: str
    maximum_cost_p90_ms: float
    selected_cost_p90_ms: float
    maximum_peak_vram_gib: float
    selected_peak_vram_gib: float
    selected_risk_ucb: V19HumanRiskVector
    debt_limits: V19TrajectoryDebt
    selected_maximum_debt: V19TrajectoryDebt
    action_ids: tuple[str, ...]
    certificate_digest: str
    schema_version: str = V19_PLAN_CERTIFICATE_SCHEMA
    contract_schema: str = V19_CONTRACT_SCHEMA

    @staticmethod
    def issue(
        registry: ActionRegistry,
        request: V19PlanningRequest,
        candidate: V19CandidatePlan,
    ) -> "V19PlanCertificate":
        base = V19PlanCertificate(
            registry_digest=registry.digest,
            workload_digest=request.workload.digest,
            runtime_digest=(
                request.runtime.digest if request.runtime is not None else "unbound"
            ),
            candidate_digest=candidate.digest,
            maximum_cost_p90_ms=request.maximum_cost_p90_ms,
            selected_cost_p90_ms=candidate.predicted_cost_p90_ms,
            maximum_peak_vram_gib=request.maximum_peak_vram_gib,
            selected_peak_vram_gib=candidate.predicted_peak_vram_gib,
            selected_risk_ucb=candidate.risk_ucb,
            debt_limits=request.debt_limits,
            selected_maximum_debt=candidate.maximum_debt,
            action_ids=tuple(sorted({use.action_id for use in candidate.action_uses})),
            certificate_digest="",
        )
        return replace(base, certificate_digest=_certificate_digest(base))


@dataclass(frozen=True, slots=True)
class V19PlanVerification:
    valid: bool
    reasons: tuple[str, ...]


def _sha256(document: object) -> str:
    encoded = json.dumps(
        document, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _certificate_digest(certificate: V19PlanCertificate) -> str:
    document = asdict(certificate)
    document["certificate_digest"] = ""
    return _sha256(document)


def validate_candidate_actions(
    registry: ActionRegistry,
    workload: V19WorkloadContext,
    candidate: V19CandidatePlan,
    *,
    permit_evidence_bound_actions: bool = False,
) -> None:
    eligible = {
        action.action_id: action
        for action in registry.planner_actions_for(
            model_variant=workload.model_variant,
            service_family=workload.service_family,
            packed_tokens=workload.packed_tokens,
            condition_count=workload.condition_count,
            device_arch=workload.device_arch,
        )
    }
    for use in candidate.action_uses:
        try:
            action = registry.resolve(use.action_id)
        except ActionRegistryError as error:
            raise V19PlanningError(str(error)) from error
        if use.action_id not in eligible and not (
            permit_evidence_bound_actions
            and action.evidence_status is not EvidenceStatus.REJECTED
            and action.calibration_ids
        ):
            raise V19PlanningError(
                f"action is not calibrated and eligible for this workload: {use.action_id}"
            )
        if use.canonical_action not in action.canonical_actions:
            raise V19PlanningError(
                f"canonical action {use.canonical_action} is not implemented by {use.action_id}"
            )
        if isinstance(use, V19ForecastUse):
            if action.kind is not ActionKind.FORECAST_COMPOSITE:
                raise V19PlanningError("forecast use names a non-forecast registry action")
            try:
                anchor = registry.resolve(use.composite_key.anchor_action_id)
            except ActionRegistryError as error:
                raise V19PlanningError(str(error)) from error
            if (
                anchor.kind is not ActionKind.SPARSE_ATTENTION
                or use.composite_key.anchor_canonical_action
                not in anchor.canonical_actions
            ):
                raise V19PlanningError("forecast anchor action identity is invalid")


def _dominates(left: V19CandidatePlan, right: V19CandidatePlan) -> bool:
    return left.pareto_objective.dominates(right.pareto_objective)


class V19ParetoPlanner:
    """Finite, auditable frontier; no scalar quality score is manufactured."""

    def __init__(
        self,
        registry: ActionRegistry,
        calibration_catalog: V19CalibrationCatalog | None = None,
        forecast_calibration_catalog: V19ForecastCalibrationCatalog | None = None,
        risk_calibration_catalog: V19RiskCalibrationCatalog | None = None,
        schedule_cost_catalog: V19ScheduleCostCatalog | None = None,
        *,
        require_physical_calibration: bool = True,
        require_complete_schedule: bool = True,
        require_human_risk_calibration: bool = False,
        require_end_to_end_cost_calibration: bool = False,
    ) -> None:
        self.registry = registry
        self.calibration_catalog = calibration_catalog
        self.forecast_calibration_catalog = forecast_calibration_catalog
        self.risk_calibration_catalog = risk_calibration_catalog
        self.schedule_cost_catalog = schedule_cost_catalog
        self.require_physical_calibration = require_physical_calibration
        self.require_complete_schedule = require_complete_schedule
        self.require_human_risk_calibration = require_human_risk_calibration
        self.require_end_to_end_cost_calibration = (
            require_end_to_end_cost_calibration
        )

    def _validate_physical_cost(
        self,
        request: V19PlanningRequest,
        candidate: V19CandidatePlan,
    ) -> None:
        if not self.require_physical_calibration:
            return
        if self.calibration_catalog is None or request.runtime is None:
            raise V19PlanningError(
                "V19 physical calibration catalog and runtime fingerprint are required"
            )
        try:
            attention_uses = tuple(
                use for use in candidate.action_uses if isinstance(use, V19ActionUse)
            )
            forecast_uses = tuple(
                use for use in candidate.action_uses if isinstance(use, V19ForecastUse)
            )
            workload = request.workload.to_calibration_workload()
            if self.require_complete_schedule:
                expected_attention = {
                    (step, layer)
                    for step in workload.actual_step_indices
                    for layer in range(50)
                }
                actual_attention = {
                    (step, layer)
                    for use in attention_uses
                    for step in use.step_indices
                    for layer in range(use.layer_start, use.layer_stop)
                }
                if actual_attention != expected_attention:
                    raise V19PlanningError(
                        "candidate does not cover every actual-step Attention cell exactly"
                    )
                expected_forecast = set(range(workload.steps)) - set(
                    workload.actual_step_indices
                )
                actual_forecast = [
                    step for use in forecast_uses for step in use.step_indices
                ]
                if (
                    len(set(actual_forecast)) != len(actual_forecast)
                    or set(actual_forecast) != expected_forecast
                ):
                    raise V19PlanningError(
                        "candidate forecast composites do not exactly cover forecast steps"
                    )
            estimate = self.calibration_catalog.estimate_schedule(
                attention_uses,
                workload=request.workload.to_calibration_workload(),
                runtime=request.runtime,
            )
            p50_ms = estimate.p50_ms
            p90_ms = estimate.p90_ms
            peak_vram_gib = estimate.peak_vram_gib
            evidence_ids = set(estimate.evidence_ids)
            if forecast_uses:
                if self.forecast_calibration_catalog is None:
                    raise V19PlanningError(
                        "V19 forecast calibration catalog is required"
                    )
                for use in forecast_uses:
                    forecast_cost = self.forecast_calibration_catalog.estimate(
                        action_id=use.action_id,
                        key=use.composite_key,
                        workload=workload,
                        runtime=request.runtime,
                    )
                    p50_ms += forecast_cost.p50_ms
                    p90_ms += forecast_cost.p90_ms
                    peak_vram_gib = max(
                        peak_vram_gib,
                        forecast_cost.peak_vram_gib,
                    )
                    evidence_ids.add(forecast_cost.evidence_id)
            if self.require_end_to_end_cost_calibration:
                if self.schedule_cost_catalog is None:
                    raise V19PlanningError(
                        "V19 end-to-end schedule calibration catalog is required"
                    )
                schedule_cost = self.schedule_cost_catalog.estimate(
                    execution_digest=candidate.execution_digest,
                    workload=workload,
                    runtime=request.runtime,
                    action_ids=(use.action_id for use in candidate.action_uses),
                )
                p50_ms = schedule_cost.p50_ms
                p90_ms = schedule_cost.p90_ms
                peak_vram_gib = schedule_cost.peak_vram_gib
                evidence_ids.add(schedule_cost.evidence_id)
        except V19CalibrationError as error:
            raise V19PlanningError(str(error)) from error
        if not math.isclose(
            candidate.predicted_cost_p50_ms,
            p50_ms,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ) or not math.isclose(
            candidate.predicted_cost_p90_ms,
            p90_ms,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise V19PlanningError(
                "candidate cost is not the registry-bound physical calibration cost"
            )
        if not math.isclose(
            candidate.predicted_peak_vram_gib,
            peak_vram_gib,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise V19PlanningError(
                "candidate peak VRAM is not the registry-bound physical calibration peak"
            )
        if not evidence_ids.issubset(candidate.evidence_ids):
            raise V19PlanningError("candidate omits its physical calibration evidence")

    def _validate_human_risk(
        self,
        request: V19PlanningRequest,
        candidate: V19CandidatePlan,
    ) -> None:
        if not self.require_human_risk_calibration:
            return
        if self.risk_calibration_catalog is None or request.runtime is None:
            raise V19PlanningError(
                "V19 Human-risk catalog and runtime fingerprint are required"
            )
        try:
            risk_ucb, evidence_id = self.risk_calibration_catalog.estimate(
                execution_digest=candidate.execution_digest,
                workload=request.workload.to_calibration_workload(),
                runtime=request.runtime,
                action_ids=(use.action_id for use in candidate.action_uses),
            )
        except V19RiskCalibrationError as error:
            raise V19PlanningError(str(error)) from error
        if candidate.risk_ucb != risk_ucb:
            raise V19PlanningError(
                "candidate Human risk is not the plan-bound calibrated UCB"
            )
        if evidence_id not in candidate.evidence_ids:
            raise V19PlanningError("candidate omits its Human-risk evidence")

    def feasible_frontier(
        self,
        request: V19PlanningRequest,
        candidates: Iterable[V19CandidatePlan],
    ) -> tuple[V19CandidatePlan, ...]:
        feasible: list[V19CandidatePlan] = []
        for candidate in candidates:
            validate_candidate_actions(
                self.registry,
                request.workload,
                candidate,
                permit_evidence_bound_actions=self.require_physical_calibration,
            )
            self._validate_physical_cost(request, candidate)
            self._validate_human_risk(request, candidate)
            if candidate.predicted_cost_p90_ms > request.maximum_cost_p90_ms:
                continue
            if candidate.predicted_peak_vram_gib > request.maximum_peak_vram_gib:
                continue
            if not candidate.risk_ucb.within(request.risk_limits):
                continue
            if not candidate.maximum_debt.within(request.debt_limits):
                continue
            feasible.append(candidate)
        frontier = [
            candidate
            for candidate in feasible
            if not any(
                other.candidate_id != candidate.candidate_id
                and _dominates(other, candidate)
                for other in feasible
            )
        ]
        return tuple(
            sorted(
                frontier,
                key=lambda row: (row.predicted_cost_p90_ms, row.risk_ucb.as_tuple(), row.candidate_id),
            )
        )

    def certify(
        self,
        request: V19PlanningRequest,
        candidate: V19CandidatePlan,
    ) -> V19PlanCertificate:
        validate_candidate_actions(
            self.registry,
            request.workload,
            candidate,
            permit_evidence_bound_actions=self.require_physical_calibration,
        )
        self._validate_physical_cost(request, candidate)
        self._validate_human_risk(request, candidate)
        if candidate.predicted_cost_p90_ms > request.maximum_cost_p90_ms:
            raise V19PlanningError("candidate exceeds the p90 compute budget")
        if candidate.predicted_peak_vram_gib > request.maximum_peak_vram_gib:
            raise V19PlanningError("candidate exceeds the peak-VRAM budget")
        if not candidate.risk_ucb.within(request.risk_limits):
            raise V19PlanningError("candidate violates a non-compensating Human risk limit")
        if not candidate.maximum_debt.within(request.debt_limits):
            raise V19PlanningError("candidate violates a non-compensating trajectory-debt limit")
        return V19PlanCertificate.issue(self.registry, request, candidate)


def verify_v19_plan_certificate(
    registry: ActionRegistry,
    request: V19PlanningRequest,
    candidate: V19CandidatePlan,
    certificate: V19PlanCertificate,
) -> V19PlanVerification:
    reasons: list[str] = []
    if certificate.schema_version != V19_PLAN_CERTIFICATE_SCHEMA:
        reasons.append("certificate schema mismatch")
    if certificate.contract_schema != V19_CONTRACT_SCHEMA:
        reasons.append("V19 contract schema mismatch")
    if certificate.registry_digest != registry.digest:
        reasons.append("registry digest mismatch")
    if certificate.workload_digest != request.workload.digest:
        reasons.append("workload digest mismatch")
    expected_runtime_digest = (
        request.runtime.digest if request.runtime is not None else "unbound"
    )
    if certificate.runtime_digest != expected_runtime_digest:
        reasons.append("runtime digest mismatch")
    if certificate.candidate_digest != candidate.digest:
        reasons.append("candidate digest mismatch")
    if certificate.maximum_cost_p90_ms != request.maximum_cost_p90_ms:
        reasons.append("budget mismatch")
    if certificate.selected_cost_p90_ms != candidate.predicted_cost_p90_ms:
        reasons.append("selected cost mismatch")
    if certificate.maximum_peak_vram_gib != request.maximum_peak_vram_gib:
        reasons.append("peak-VRAM budget mismatch")
    if certificate.selected_peak_vram_gib != candidate.predicted_peak_vram_gib:
        reasons.append("selected peak-VRAM mismatch")
    if certificate.selected_risk_ucb != candidate.risk_ucb:
        reasons.append("selected Human risk mismatch")
    if certificate.debt_limits != request.debt_limits:
        reasons.append("trajectory-debt limits mismatch")
    if certificate.selected_maximum_debt != candidate.maximum_debt:
        reasons.append("selected maximum trajectory debt mismatch")
    if certificate.action_ids != tuple(
        sorted({use.action_id for use in candidate.action_uses})
    ):
        reasons.append("action identity mismatch")
    if certificate.certificate_digest != _certificate_digest(certificate):
        reasons.append("certificate digest mismatch")
    if candidate.predicted_cost_p90_ms > request.maximum_cost_p90_ms:
        reasons.append("candidate exceeds the certified p90 compute budget")
    if candidate.predicted_peak_vram_gib > request.maximum_peak_vram_gib:
        reasons.append("candidate exceeds the certified peak-VRAM budget")
    if not candidate.risk_ucb.within(request.risk_limits):
        reasons.append("candidate exceeds a certified Human-risk limit")
    if not candidate.maximum_debt.within(request.debt_limits):
        reasons.append("candidate exceeds a certified trajectory-debt limit")
    try:
        validate_candidate_actions(
            registry,
            request.workload,
            candidate,
            permit_evidence_bound_actions=True,
        )
    except V19PlanningError as error:
        reasons.append(str(error))
    return V19PlanVerification(valid=not reasons, reasons=tuple(reasons))


__all__ = [
    "V19_PLAN_CERTIFICATE_SCHEMA",
    "V19ActionUse",
    "V19CandidatePlan",
    "V19ForecastUse",
    "V19ParetoPlanner",
    "V19PlanCertificate",
    "V19PlanVerification",
    "V19PlanningError",
    "V19PlanningRequest",
    "V19WorkloadContext",
    "validate_candidate_actions",
    "verify_v19_plan_certificate",
]
