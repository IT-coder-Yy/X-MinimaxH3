"""Experimental, fail-closed runtime routing for long V19 workloads.

The certified V19 release catalog remains the only production authority.  This
module is a deliberately separate bridge for exact schedules that already have
end-to-end measurements but are still waiting for continuous Human review.
It never inspects prompt text or media semantics: admission depends only on the
public ``steps + acceleration`` controls, exact packed-layout resource facts,
and measured output geometry.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from typing import Any

from .v19_candidates import v19_blueprint_execution_digest
from .v19_human_constraints import (
    evaluate_v19_human_constraints,
    v19_long_horizon_screening_policy,
)
from .v19_long_horizon import build_v19_long_horizon_round188_replay
from .v19_planner import V19PlanningError, V19WorkloadContext
from .v19_runtime_bridge import runtime_schedule_from_blueprint


V19_LONG_RUNTIME_SCHEMA = "h3_v19_experimental_long_runtime_selection_v1"
V19_LONG_RUNTIME_POLICY = "h3_v19_long_15s_round188_experimental_v1"
V19_LONG_RUNTIME_CANDIDATE = "v012_long_round188_replay_12a8f"
V19_LONG_RUNTIME_ACCELERATION_FLOOR = 75.0

# These are measured research envelopes, not extrapolated product promises.
# Packed-token ranges tolerate normal text-token variation while remaining
# close to the two completed 15.084 s E2E workloads.  Reference inputs remain
# outside the envelope until their additional packed rows receive an E2E run.
V19_LONG_RUNTIME_ENVELOPES = (
    {
        "envelope_id": "experimental_720p10_base_no_reference_v1",
        "width": 1280,
        "height": 736,
        "frames": 243,
        "steps": 20,
        "min_packed_tokens": 65_000,
        "max_packed_tokens": 70_000,
        "evidence_ids": (
            "batch09_720p10_radio_console_67535_tokens",
            "batch09_v012_e2e_152p123897_seconds",
            "batch09_v012_speedup_1p250012",
        ),
    },
    {
        "envelope_id": "experimental_720p15_base_no_reference_v1",
        "width": 1280,
        "height": 736,
        "frames": 362,
        "steps": 20,
        "min_packed_tokens": 98_000,
        "max_packed_tokens": 103_000,
        "evidence_ids": (
            "batch05_720p15_workshop_100141_tokens",
            "batch06_720p15_motorcycle_100038_tokens",
        ),
    },
    {
        "envelope_id": "experimental_1080p15_base_no_reference_v1",
        "width": 1920,
        "height": 1088,
        "frames": 362,
        "steps": 20,
        "min_packed_tokens": 218_000,
        "max_packed_tokens": 220_500,
        "evidence_ids": (
            "batch07_1080p15_lighthouse_219890_tokens",
            "batch07_v012_e2e_731p638489_seconds",
            "batch07_v012_execution_495c2c8ff75f",
        ),
    },
)


@dataclass(frozen=True, slots=True)
class V19LongRuntimeAdmission:
    admitted: bool
    reason: str
    envelope_id: str | None = None
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class V19LongRuntimeSelection:
    """Hot-session-compatible selection without a release certificate."""

    actual_step_indices: tuple[int, ...]
    attention_action_schedule: tuple[tuple[int, int, str], ...]
    summary: dict[str, object]
    schema_version: str = V19_LONG_RUNTIME_SCHEMA

    def __post_init__(self) -> None:
        if not self.actual_step_indices:
            raise V19PlanningError("experimental long selection has no Actual step")
        if tuple(sorted(set(self.actual_step_indices))) != self.actual_step_indices:
            raise V19PlanningError(
                "experimental long Actual steps must be sorted and unique"
            )


def classify_v19_long_runtime_workload(
    workload: V19WorkloadContext,
    *,
    acceleration: float,
) -> V19LongRuntimeAdmission:
    """Admit only measured long-video resource envelopes.

    Prompt wording, seed and scene category are intentionally absent from the
    workload contract.  Reference counts are resource dimensions rather than
    semantic categories; they fail closed because the present 1080p run leaves
    little VRAM headroom and no reference-bearing 15-second run exists yet.
    """

    if not math.isfinite(acceleration) or not 0.0 <= acceleration <= 100.0:
        raise V19PlanningError("V19 acceleration must lie inside [0, 100]")
    if acceleration < V19_LONG_RUNTIME_ACCELERATION_FLOOR:
        return V19LongRuntimeAdmission(False, "below_measured_acceleration_floor")
    if workload.model_variant != "base":
        return V19LongRuntimeAdmission(False, "model_variant_unmeasured")
    if workload.service_family != "first_last":
        return V19LongRuntimeAdmission(False, "service_family_unmeasured")
    if workload.device_arch != "sm89":
        return V19LongRuntimeAdmission(False, "device_arch_unmeasured")
    if (
        workload.condition_count != 0
        or workload.reference_images != 0
        or workload.reference_audio != 0
        or workload.reference_videos != 0
    ):
        return V19LongRuntimeAdmission(False, "reference_profile_unmeasured")

    for envelope in V19_LONG_RUNTIME_ENVELOPES:
        if (
            workload.width == envelope["width"]
            and workload.height == envelope["height"]
            and workload.frames == envelope["frames"]
            and workload.steps == envelope["steps"]
            and envelope["min_packed_tokens"]
            <= workload.packed_tokens
            <= envelope["max_packed_tokens"]
        ):
            return V19LongRuntimeAdmission(
                True,
                "measured_experimental_long_envelope",
                envelope_id=str(envelope["envelope_id"]),
                evidence_ids=tuple(envelope["evidence_ids"]),
            )
    return V19LongRuntimeAdmission(False, "geometry_or_token_envelope_unmeasured")


def _technique_mix(
    *,
    total_steps: int,
    actual_steps: tuple[int, ...],
    schedule: tuple[tuple[int, int, str], ...],
) -> dict[str, object]:
    actual = set(actual_steps)
    actual_actions = Counter(
        action.rsplit(":", 1)[-1]
        for step, _layer, action in schedule
        if step in actual
    )
    forecast_actions = Counter(
        action.rsplit(":", 1)[-1]
        for step, _layer, action in schedule
        if step not in actual
    )
    return {
        "actual_dit_evaluations": len(actual_steps),
        "forecast_evaluations": total_steps - len(actual_steps),
        "actual_attention_cells": dict(sorted(actual_actions.items())),
        "forecast_anchor_attention_cells": dict(sorted(forecast_actions.items())),
        "coupled_techniques": [
            "exact_runtime",
            "directional_forecast",
            "block_sparse_attention",
            "pageable_chunked_forecast_history_when_required",
        ],
    }


def _dense_fallback(
    workload: V19WorkloadContext,
    *,
    acceleration: float,
    reason: str,
) -> V19LongRuntimeSelection:
    assert workload.steps is not None
    actual = tuple(range(int(workload.steps)))
    return V19LongRuntimeSelection(
        actual_step_indices=actual,
        attention_action_schedule=(),
        summary={
            "schema_version": V19_LONG_RUNTIME_SCHEMA,
            "policy_id": V19_LONG_RUNTIME_POLICY,
            "experimental": True,
            "release_eligible": False,
            "accelerated": False,
            "reason": reason,
            "acceleration": acceleration,
            "actual_step_indices": list(actual),
            "forecast_steps": 0,
            "technique_mix": {
                "actual_dit_evaluations": len(actual),
                "forecast_evaluations": 0,
                "actual_attention_cells": {"dense": len(actual) * 50},
                "forecast_anchor_attention_cells": {},
                "coupled_techniques": ["exact_runtime"],
            },
        },
    )


class V19ExperimentalLongRuntimeSelector:
    """Overlay measured long-video proposals on a certified release selector.

    The wrapper is constructed only after an explicit deployment-side opt-in.
    Non-admitted requests are delegated unchanged to the release selector.  If
    no release catalog exists, they execute the complete Dense trajectory.
    """

    def __init__(self, release_selector: Any | None = None) -> None:
        self.release_selector = release_selector

    def _delegate_or_dense(
        self,
        *,
        workload: V19WorkloadContext,
        acceleration: float,
        required_actual_step_indices: tuple[int, ...],
        reason: str,
    ) -> Any:
        if self.release_selector is not None:
            return self.release_selector.select(
                workload=workload,
                acceleration=acceleration,
                required_actual_step_indices=required_actual_step_indices,
            )
        return _dense_fallback(
            workload,
            acceleration=acceleration,
            reason=f"experimental_long_{reason}_dense_fallback",
        )

    def select(
        self,
        *,
        workload: V19WorkloadContext,
        acceleration: float,
        required_actual_step_indices: tuple[int, ...] = (),
    ) -> Any:
        if workload.steps is None:
            raise V19PlanningError(
                "experimental long runtime selection requires total steps"
            )
        admission = classify_v19_long_runtime_workload(
            workload,
            acceleration=acceleration,
        )
        if not admission.admitted:
            return self._delegate_or_dense(
                workload=workload,
                acceleration=acceleration,
                required_actual_step_indices=required_actual_step_indices,
                reason=admission.reason,
            )

        # The physical endpoint is clamped at the measured 75-strength quality
        # floor.  Acceleration 76--100 must not silently become more aggressive
        # before another long-video Human result expands that evidence floor.
        effective_acceleration = V19_LONG_RUNTIME_ACCELERATION_FLOOR
        blueprint = build_v19_long_horizon_round188_replay(
            candidate_id=V19_LONG_RUNTIME_CANDIDATE,
            total_steps=int(workload.steps),
            acceleration=effective_acceleration,
        )
        report = evaluate_v19_human_constraints(
            blueprint,
            v19_long_horizon_screening_policy(int(workload.steps)),
        )
        actual = report.actual_step_indices
        if any(step not in set(actual) for step in required_actual_step_indices):
            return self._delegate_or_dense(
                workload=workload,
                acceleration=acceleration,
                required_actual_step_indices=required_actual_step_indices,
                reason="preview_anchor_unmeasured",
            )
        schedule = runtime_schedule_from_blueprint(blueprint)
        summary: dict[str, object] = {
            "schema_version": V19_LONG_RUNTIME_SCHEMA,
            "policy_id": V19_LONG_RUNTIME_POLICY,
            "experimental": True,
            "proposal_eligible": report.proposal_eligible,
            "release_eligible": report.release_eligible,
            "accelerated": True,
            "reason": admission.reason,
            "acceleration": acceleration,
            "effective_acceleration": effective_acceleration,
            "acceleration_clamped_to_human_evidence": (
                acceleration > effective_acceleration
            ),
            "generation_request_digest": workload.digest,
            "envelope_id": admission.envelope_id,
            "evidence_ids": list(admission.evidence_ids),
            "candidate_id": blueprint.candidate_id,
            "execution_digest": v19_blueprint_execution_digest(blueprint),
            "actual_step_indices": list(actual),
            "forecast_steps": int(workload.steps) - len(actual),
            "required_actual_step_indices": list(required_actual_step_indices),
            "constraint_report": asdict(report),
            "technique_mix": _technique_mix(
                total_steps=int(workload.steps),
                actual_steps=actual,
                schedule=schedule,
            ),
        }
        return V19LongRuntimeSelection(
            actual_step_indices=actual,
            attention_action_schedule=schedule,
            summary=summary,
        )


__all__ = [
    "V19ExperimentalLongRuntimeSelector",
    "V19LongRuntimeAdmission",
    "V19LongRuntimeSelection",
    "V19_LONG_RUNTIME_ACCELERATION_FLOOR",
    "V19_LONG_RUNTIME_CANDIDATE",
    "V19_LONG_RUNTIME_ENVELOPES",
    "V19_LONG_RUNTIME_POLICY",
    "V19_LONG_RUNTIME_SCHEMA",
    "classify_v19_long_runtime_workload",
]
