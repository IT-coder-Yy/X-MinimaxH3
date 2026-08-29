"""Opt-in deployment bridge for the mechanism-driven H3 controller.

This selector intentionally has no built-in Human risk threshold.  A caller
must provide a workload-scoped admission artifact; until long-video/Ref holdout
evidence exists, unsupported token geometry fails closed to Dense.  This keeps
the implementation testable without silently turning the first identification
curve into a release promise.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .mechanistic_control import (
    H3MechanisticAdmission,
    H3MechanisticControlModel,
    H3MechanisticPlan,
    H3MechanisticWorkload,
    MechanisticControlError,
)
from .mechanistic_runtime import H3MechanisticRuntimeController
from .v19_planner import V19PlanningError, V19WorkloadContext


MECHANISTIC_DEPLOYMENT_SCHEMA = "h3_mechanistic_deployment_selection_v1"
MECHANISTIC_DEPLOYMENT_POLICY_ID = "h3_mechanistic_pareto_deployment_v1"
MECHANISTIC_ADMISSION_SCHEMA = "h3_mechanistic_deployment_admission_v1"


@dataclass(frozen=True, slots=True)
class H3MechanisticDeploymentConfig:
    """Schedule-free Human admission artifact for one evidence envelope."""

    admission: H3MechanisticAdmission
    calibrated_video_token_interval: tuple[int, int]
    maximum_runtime_promotions: int
    status: str
    source: Path
    source_sha256: str


def load_h3_mechanistic_deployment_config(
    path: str | Path,
) -> H3MechanisticDeploymentConfig:
    """Load a strict admission boundary; physical schedules are forbidden."""

    source = Path(path).expanduser().resolve()
    try:
        raw = source.read_bytes()
        document = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MechanisticControlError(
            f"cannot load mechanistic deployment admission: {source}"
        ) from error
    if not isinstance(document, dict):
        raise MechanisticControlError("mechanistic admission must be a JSON object")
    allowed = {
        "schema_version",
        "status",
        "calibration_id",
        "maximum_modeled_risk",
        "evidence_ids",
        "held_out_evidence_ids",
        "calibrated_video_token_interval",
        "maximum_runtime_promotions",
    }
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise MechanisticControlError(
            "mechanistic admission contains forbidden/unknown fields: "
            + ", ".join(unknown)
        )
    if document.get("schema_version") != MECHANISTIC_ADMISSION_SCHEMA:
        raise MechanisticControlError("unexpected mechanistic admission schema")
    status = str(document.get("status", ""))
    if status not in ("experimental", "release"):
        raise MechanisticControlError(
            "mechanistic admission status must be experimental or release"
        )
    evidence = document.get("evidence_ids")
    held_out = document.get("held_out_evidence_ids", [])
    interval = document.get("calibrated_video_token_interval")
    if (
        not isinstance(evidence, list)
        or not all(isinstance(value, str) and value for value in evidence)
        or len(set(evidence)) != len(evidence)
        or not isinstance(held_out, list)
        or not all(isinstance(value, str) and value for value in held_out)
        or len(set(held_out)) != len(held_out)
    ):
        raise MechanisticControlError("mechanistic admission evidence ids are invalid")
    if status == "release" and not held_out:
        raise MechanisticControlError(
            "release admission requires held-out Human evidence"
        )
    calibration_id = document.get("calibration_id")
    maximum_risk = document.get("maximum_modeled_risk")
    if not isinstance(calibration_id, str) or not calibration_id:
        raise MechanisticControlError(
            "mechanistic admission calibration id is invalid"
        )
    if not isinstance(maximum_risk, (int, float)) or isinstance(
        maximum_risk, bool
    ):
        raise MechanisticControlError(
            "mechanistic maximum modeled risk must be numeric"
        )
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in interval
        )
    ):
        raise MechanisticControlError(
            "mechanistic calibrated video-token interval is invalid"
        )
    promotions = document.get("maximum_runtime_promotions", 0)
    if not isinstance(promotions, int):
        raise MechanisticControlError(
            "mechanistic maximum runtime promotions must be an integer"
        )
    admission = H3MechanisticAdmission(
        calibration_id=calibration_id,
        maximum_modeled_risk=float(maximum_risk),
        evidence_ids=tuple(evidence),
        held_out_evidence_ids=tuple(held_out),
    )
    lower, upper = interval
    # Reuse selector validation so loader and direct construction cannot drift.
    H3MechanisticParetoRuntimeSelector(
        admission=admission,
        calibrated_video_token_interval=(lower, upper),
        maximum_runtime_promotions=promotions,
    )
    return H3MechanisticDeploymentConfig(
        admission=admission,
        calibrated_video_token_interval=(lower, upper),
        maximum_runtime_promotions=promotions,
        status=status,
        source=source,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class H3MechanisticDeploymentSelection:
    actual_step_indices: tuple[int, ...]
    attention_action_schedule: tuple[tuple[int, int, str], ...]
    summary: dict[str, Any]
    runtime_controller: H3MechanisticRuntimeController | None = None


class H3MechanisticParetoRuntimeSelector:
    """Compile the public dial after exact H3 tokenisation."""

    policy_id = MECHANISTIC_DEPLOYMENT_POLICY_ID

    def __init__(
        self,
        *,
        admission: H3MechanisticAdmission,
        calibrated_video_token_interval: tuple[int, int],
        maximum_runtime_promotions: int = 1,
    ) -> None:
        lower, upper = (int(value) for value in calibrated_video_token_interval)
        if lower <= 0 or upper < lower:
            raise MechanisticControlError(
                "deployment video-token interval is invalid"
            )
        if not 0 <= int(maximum_runtime_promotions) <= 2:
            raise MechanisticControlError(
                "deployment supports zero to two runtime promotions"
            )
        self.admission = admission
        self.calibrated_video_token_interval = (lower, upper)
        self.maximum_runtime_promotions = int(maximum_runtime_promotions)

    @staticmethod
    def _video_tokens(workload: V19WorkloadContext) -> int:
        if (
            workload.width is None
            or workload.height is None
            or workload.frames is None
        ):
            raise V19PlanningError("mechanistic deployment requires exact geometry")
        if workload.width % 32 or workload.height % 32:
            raise V19PlanningError("H3 geometry must be divisible by 32")
        if workload.frames < 5 or (workload.frames - 5) % 17:
            raise V19PlanningError("H3 frames must satisfy 17*n+5")
        latent_frames = ((workload.frames - 5) // 17) * 5 + 2
        return latent_frames * (workload.width // 32) * (workload.height // 32)

    @staticmethod
    def _promotion_increment_ms(model: H3MechanisticControlModel) -> float:
        actions = ("sparse_topk_0.0625",) * 3 + ("dense",) * 47
        actual = model._physical.non_attention_ms + sum(
            model._layer_cost_ms(layer, action)
            for layer, action in enumerate(actions)
        )
        return max(0.0, actual - model._physical.forecast_ms)

    def _allocate_reserve(
        self,
        *,
        model: H3MechanisticControlModel,
        public_plan: H3MechanisticPlan,
    ) -> tuple[H3MechanisticPlan, float, int]:
        if (
            self.maximum_runtime_promotions == 0
            or not public_plan.forecast_step_indices
            or public_plan.target_cost_ms is None
        ):
            return public_plan, 0.0, 0
        promotion_cost = self._promotion_increment_ms(model)
        target = float(public_plan.target_cost_ms)
        # A Forecast promotion is indivisible.  Enumerating the at-most-two
        # discrete reserve counts is exact and avoids turning a floating-point
        # value infinitesimally below one promotion into zero usable reserve.
        for promotion_count in range(
            self.maximum_runtime_promotions,
            0,
            -1,
        ):
            reserve = promotion_cost * promotion_count
            try:
                candidate = model.plan_for_cost_budget(
                    maximum_cost_ms=target - reserve
                )
            except MechanisticControlError:
                continue
            feasible = (
                candidate.modeled_risk.total
                <= self.admission.maximum_modeled_risk + 1.0e-12
                and candidate.predicted_cost_ms + reserve <= target + 1.0e-8
            )
            if feasible:
                return candidate, reserve, promotion_count
        return public_plan, 0.0, 0

    @staticmethod
    def _summary(
        *,
        workload: H3MechanisticWorkload,
        plan: H3MechanisticPlan,
        acceleration: float,
        reason: str,
        reserve_ms: float,
        runtime_promotions: int,
        admission: H3MechanisticAdmission,
        public_target_cost_ms: float,
    ) -> dict[str, Any]:
        action_counts: dict[str, int] = {}
        for choice in plan.choices:
            for action in choice.attention_actions:
                action_counts[action] = action_counts.get(action, 0) + 1
        return {
            "schema_version": MECHANISTIC_DEPLOYMENT_SCHEMA,
            "policy_id": MECHANISTIC_DEPLOYMENT_POLICY_ID,
            "accelerated": bool(
                plan.forecast_step_indices or set(action_counts) != {"dense"}
            ),
            "reason": reason,
            "acceleration": acceleration,
            "public_dial_semantics": (
                "fraction_of_human_admitted_compute_saving_interval"
            ),
            "prompt_semantics_used": False,
            "historical_schedule_used": False,
            "packed_tokens": workload.packed_tokens,
            "video_tokens": workload.video_tokens,
            "actual_step_indices": list(plan.actual_step_indices),
            "forecast_step_indices": list(plan.forecast_step_indices),
            "maximum_forecast_run": plan.maximum_forecast_run,
            "predicted_cost_ms": plan.predicted_cost_ms,
            # The creator dial pays for both the sealed offline path and the
            # request-local correction reserve.  Reporting only the former
            # would make recovery appear to be free compute.
            "offline_plan_target_cost_ms": plan.target_cost_ms,
            "public_target_cost_ms": public_target_cost_ms,
            "reserved_total_cost_ms": plan.predicted_cost_ms + reserve_ms,
            "modeled_risk": plan.modeled_risk.to_dict(),
            "admission": {
                "calibration_id": admission.calibration_id,
                "maximum_modeled_risk": admission.maximum_modeled_risk,
                "evidence_ids": list(admission.evidence_ids),
                "held_out_evidence_ids": list(admission.held_out_evidence_ids),
            },
            "certificate": plan.to_dict()["certificate"],
            "technique_mix": {
                "actual_dit_evaluations": len(plan.actual_step_indices),
                "forecast_evaluations": len(plan.forecast_step_indices),
                "actual_attention_cells": dict(sorted(action_counts.items())),
                "coupled_techniques": [
                    "measured_attention_actions",
                    "directional_forecast",
                    "phase_propagation_risk",
                    "request_local_risk_reserve",
                ],
            },
            "runtime_feedback": {
                "policy_id": (
                    None
                    if runtime_promotions == 0
                    else "h3_request_local_risk_reserve_mpc_v1"
                ),
                "mode": (
                    "observe_only_no_discrete_reserve"
                    if runtime_promotions == 0
                    else "mechanistic_risk_reserve"
                ),
                "adds_teacher_evaluations": False,
                "recovery_reserve_ms": reserve_ms,
                "max_runtime_promotions": runtime_promotions,
            },
        }

    def select(
        self,
        *,
        workload: V19WorkloadContext,
        acceleration: float,
        required_actual_step_indices: tuple[int, ...] = (),
    ) -> H3MechanisticDeploymentSelection:
        if workload.steps is None:
            raise V19PlanningError("mechanistic deployment requires total steps")
        try:
            acceleration = round(float(acceleration), 1)
        except (TypeError, ValueError) as error:
            raise V19PlanningError("mechanistic acceleration must be numeric") from error
        if not math.isfinite(acceleration) or not 0.0 <= acceleration <= 100.0:
            raise V19PlanningError("mechanistic acceleration lies outside [0, 100]")
        video_tokens = self._video_tokens(workload)
        if (
            workload.model_variant != "base"
            or workload.device_arch != "sm89"
            or workload.sampler != "res_multistep"
            or workload.scheduler != "simple"
            or not (
                self.calibrated_video_token_interval[0]
                <= video_tokens
                <= self.calibrated_video_token_interval[1]
            )
        ):
            # Unsupported identities remain model-capable via exact Dense.
            acceleration = 0.0
            reason = "outside_mechanistic_admission_scope_dense_fallback"
        else:
            reason = "mechanistic_joint_pareto_optimization"
        model_workload = H3MechanisticWorkload(
            total_steps=int(workload.steps),
            packed_tokens=workload.packed_tokens,
            video_tokens=video_tokens,
            condition_count=workload.condition_count,
            allow_forecast=True,
            required_actual_steps=required_actual_step_indices,
            service_family=workload.service_family,
            model_variant="base",
        )
        model = H3MechanisticControlModel(model_workload)
        public_plan = model.plan_for_acceleration(
            acceleration=acceleration,
            admission=self.admission,
        )
        if public_plan.target_cost_ms is None:
            raise AssertionError("public mechanistic plan has no compute target")
        public_target_cost_ms = float(public_plan.target_cost_ms)
        plan, reserve_ms, runtime_promotions = self._allocate_reserve(
            model=model,
            public_plan=public_plan,
        )
        runtime_controller = (
            None
            if runtime_promotions == 0 or not plan.forecast_step_indices
            else H3MechanisticRuntimeController(
                model=model,
                plan=plan,
                recovery_reserve_ms=reserve_ms,
                maximum_promotions=runtime_promotions,
                risk_limit=self.admission.maximum_modeled_risk,
            )
        )
        summary = self._summary(
            workload=model_workload,
            plan=plan,
            acceleration=acceleration,
            reason=reason,
            reserve_ms=reserve_ms,
            runtime_promotions=runtime_promotions,
            admission=self.admission,
            public_target_cost_ms=public_target_cost_ms,
        )
        return H3MechanisticDeploymentSelection(
            actual_step_indices=plan.actual_step_indices,
            attention_action_schedule=(
                () if acceleration == 0.0 else plan.runtime_action_schedule()
            ),
            summary=summary,
            runtime_controller=runtime_controller,
        )


__all__ = [
    "H3MechanisticDeploymentConfig",
    "H3MechanisticDeploymentSelection",
    "H3MechanisticParetoRuntimeSelector",
    "MECHANISTIC_ADMISSION_SCHEMA",
    "MECHANISTIC_DEPLOYMENT_POLICY_ID",
    "MECHANISTIC_DEPLOYMENT_SCHEMA",
    "load_h3_mechanistic_deployment_config",
]
