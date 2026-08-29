"""Geometry-independent quality shields for long-horizon V19 schedules.

The public contract remains ``sampling_steps + acceleration``.  This module
does not inspect prompt text, seed, reference content or scene category.  It
raises Attention compute only at two structural risk surfaces already exposed
by the V19 evidence: a fixed set of late DiT layers and the final Actual
corrections.  It never removes an Actual evaluation or downgrades an existing
Attention cell.

The builder produces experimental blueprints.  Static Human constraints can
make one eligible for an end-to-end experiment, but only continuous Human
review can promote the resulting execution digest.
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
from .v19_long_horizon import build_v19_long_horizon_round188_replay
from .v19_planner import V19ActionUse, V19PlanningError
from .v19_runtime_bridge import (
    blueprint_from_runtime_schedule,
    runtime_schedule_from_blueprint,
)


V19_LONG_QUALITY_SHIELD_SOURCE = "v19_long_quality_shield_v1"
V19_LONG_QUALITY_CORE_LAYERS = tuple((*range(39, 44), 45))
V19_LONG_QUALITY_FRONTIER_POLICY = "h3_v19_long_quality_frontier_proposal_v1"
V19_LONG_QUALITY_ACCELERATION_FLOOR = 75.0
V19_LONG_QUALITY_BALANCED_THRESHOLD = 82.5
V19_LONG_QUALITY_FAST_THRESHOLD = 92.5
V19_LONG_QUALITY_V012_CANDIDATE = "v012_long_round188_replay_12a8f"
V19_LONG_QUALITY_V014A_CANDIDATE = (
    "v014a_long_quality_shield_core025_terminal025"
)
V19_LONG_QUALITY_V014B_CANDIDATE = (
    "v014b_long_quality_shield_core050_terminal025"
)
_CANONICAL_KEEP = {
    "sparse_topk_0.0625": 0.0625,
    "sparse_topk_0.1": 0.1,
    "sparse_topk_0.25": 0.25,
    "sparse_topk_0.5": 0.5,
    "dense": 1.0,
}


def _validate_action(action: str) -> str:
    if action not in _CANONICAL_KEEP:
        raise V19PlanningError(f"unsupported long quality action: {action}")
    return action


def _canonical(runtime_action: str) -> str:
    return runtime_action.rsplit(":", 1)[-1]


def _runtime_action(canonical: str) -> str:
    return "dense" if canonical == "dense" else f"frontier:{canonical}"


@dataclass(frozen=True, slots=True)
class V19LongQualityShieldSpec:
    """One task-content-independent compute reallocation policy."""

    core_layers: tuple[int, ...] = V19_LONG_QUALITY_CORE_LAYERS
    core_action: str = "sparse_topk_0.25"
    terminal_actual_count: int = 3
    terminal_action: str = "sparse_topk_0.25"

    def __post_init__(self) -> None:
        if (
            not self.core_layers
            or tuple(sorted(set(self.core_layers))) != self.core_layers
            or any(layer < 0 or layer >= 50 for layer in self.core_layers)
        ):
            raise V19PlanningError(
                "long quality core layers must be sorted unique values in [0, 50)"
            )
        _validate_action(self.core_action)
        _validate_action(self.terminal_action)
        if self.terminal_actual_count <= 0:
            raise V19PlanningError(
                "long quality terminal Actual count must be positive"
            )


@dataclass(frozen=True, slots=True)
class V19LongQualityShieldResult:
    blueprint: V19CandidateBlueprint
    source_execution_digest: str
    actual_step_indices: tuple[int, ...]
    terminal_actual_step_indices: tuple[int, ...]
    source_action_cell_counts: tuple[tuple[str, int], ...]
    candidate_action_cell_counts: tuple[tuple[str, int], ...]
    core_upgraded_cells: int
    terminal_upgraded_cells: int
    constraint_report: V19HumanConstraintReport


@dataclass(frozen=True, slots=True)
class V19LongQualityCostEnvelope:
    """One measured geometry envelope; no cross-envelope interpolation."""

    envelope_id: str
    width: int
    height: int
    latent_frames: int
    spatial_tokens_per_frame: int
    minimum_packed_tokens: int
    maximum_packed_tokens: int
    observed_e2e_seconds: tuple[tuple[str, float], ...]
    observed_denoise_seconds: tuple[tuple[str, float], ...]
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.envelope_id
            or self.width <= 0
            or self.height <= 0
            or self.latent_frames <= 0
            or self.spatial_tokens_per_frame <= 0
            or self.minimum_packed_tokens <= 0
            or self.maximum_packed_tokens < self.minimum_packed_tokens
        ):
            raise V19PlanningError("invalid long quality cost envelope")
        expected = {
            V19_LONG_QUALITY_V012_CANDIDATE,
            V19_LONG_QUALITY_V014A_CANDIDATE,
            V19_LONG_QUALITY_V014B_CANDIDATE,
        }
        if (
            {name for name, _value in self.observed_e2e_seconds} != expected
            or {name for name, _value in self.observed_denoise_seconds}
            != expected
            or any(value <= 0.0 for _name, value in self.observed_e2e_seconds)
            or any(value <= 0.0 for _name, value in self.observed_denoise_seconds)
            or not self.evidence_ids
        ):
            raise V19PlanningError(
                "long quality envelope must cover all positive operating points"
            )

    def e2e_seconds(self, candidate_id: str) -> float:
        return dict(self.observed_e2e_seconds)[candidate_id]

    def denoise_seconds(self, candidate_id: str) -> float:
        return dict(self.observed_denoise_seconds)[candidate_id]


V19_LONG_QUALITY_COST_ENVELOPES = (
    V19LongQualityCostEnvelope(
        envelope_id="v19_long_quality_720p10_base_no_reference_v1",
        width=1280,
        height=736,
        latent_frames=72,
        spatial_tokens_per_frame=920,
        minimum_packed_tokens=65_000,
        maximum_packed_tokens=70_000,
        observed_e2e_seconds=(
            (V19_LONG_QUALITY_V012_CANDIDATE, 152.123896590012),
            (V19_LONG_QUALITY_V014A_CANDIDATE, 163.857086789998),
            (V19_LONG_QUALITY_V014B_CANDIDATE, 165.68732891899708),
        ),
        observed_denoise_seconds=(
            (V19_LONG_QUALITY_V012_CANDIDATE, 126.34053195100569),
            (V19_LONG_QUALITY_V014A_CANDIDATE, 135.5917176750081),
            (V19_LONG_QUALITY_V014B_CANDIDATE, 139.82180255799904),
        ),
        evidence_ids=(
            "batch09_v012_67535_tokens",
            "batch11_v014a_67535_tokens",
            "batch11_v014b_67535_tokens",
        ),
    ),
    V19LongQualityCostEnvelope(
        envelope_id="v19_long_quality_720p15_base_no_reference_v1",
        width=1280,
        height=736,
        latent_frames=107,
        spatial_tokens_per_frame=920,
        minimum_packed_tokens=98_000,
        maximum_packed_tokens=103_000,
        observed_e2e_seconds=(
            (V19_LONG_QUALITY_V012_CANDIDATE, 240.791668742997),
            (V19_LONG_QUALITY_V014A_CANDIDATE, 263.4175397930085),
            (V19_LONG_QUALITY_V014B_CANDIDATE, 271.137938519998),
        ),
        observed_denoise_seconds=(
            (V19_LONG_QUALITY_V012_CANDIDATE, 202.5463885000063),
            (V19_LONG_QUALITY_V014A_CANDIDATE, 222.71847620399785),
            (V19_LONG_QUALITY_V014B_CANDIDATE, 232.9728840089956),
        ),
        evidence_ids=(
            "batch05_v012_100141_tokens",
            "batch12_v014a_100141_tokens",
            "batch12_v014b_100141_tokens",
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class V19LongQualityFrontierSelection:
    """A Human-pending proposal point selected by the public acceleration dial."""

    blueprint: V19CandidateBlueprint
    operating_point: str
    acceleration: float
    envelope_id: str
    observed_e2e_seconds: float
    observed_denoise_seconds: float
    evidence_ids: tuple[str, ...]
    proposal_eligible: bool
    release_eligible: bool = False
    policy_id: str = V19_LONG_QUALITY_FRONTIER_POLICY


def _actual_steps(blueprint: V19CandidateBlueprint) -> tuple[int, ...]:
    return tuple(sorted({
        step
        for use in blueprint.action_uses
        if isinstance(use, V19ActionUse)
        for step in use.step_indices
    }))


def _cell_counts(
    schedule: tuple[tuple[int, int, str], ...],
    actual_steps: tuple[int, ...],
) -> tuple[tuple[str, int], ...]:
    actual = set(actual_steps)
    counts = Counter(
        _canonical(action)
        for step, _layer, action in schedule
        if step in actual
    )
    return tuple(sorted(counts.items()))


def build_v19_long_quality_shield(
    source: V19CandidateBlueprint,
    *,
    candidate_id: str,
    spec: V19LongQualityShieldSpec = V19LongQualityShieldSpec(),
) -> V19LongQualityShieldResult:
    """Raise structural Attention budgets without changing the trajectory."""

    if not candidate_id:
        raise V19PlanningError("long quality candidate id cannot be empty")
    actual_steps = _actual_steps(source)
    if len(actual_steps) < spec.terminal_actual_count:
        raise V19PlanningError(
            "long quality source has fewer Actual steps than the terminal shield"
        )
    source_schedule = runtime_schedule_from_blueprint(source)
    schedule = {
        (step, layer): action for step, layer, action in source_schedule
    }
    terminal_steps = actual_steps[-spec.terminal_actual_count :]
    terminal = set(terminal_steps)
    core = set(spec.core_layers)
    core_upgraded = 0
    terminal_upgraded = 0
    for step in actual_steps:
        for layer in range(50):
            cell = (step, layer)
            current_runtime = schedule[cell]
            current = _canonical(current_runtime)
            desired = None
            reason = None
            if layer in core:
                desired = spec.core_action
                reason = "core"
            elif step in terminal:
                desired = spec.terminal_action
                reason = "terminal"
            if desired is None or _CANONICAL_KEEP[desired] <= _CANONICAL_KEEP[current]:
                continue
            schedule[cell] = _runtime_action(desired)
            if reason == "core":
                core_upgraded += 1
            else:
                terminal_upgraded += 1

    candidate = blueprint_from_runtime_schedule(
        candidate_id=candidate_id,
        total_steps=max(step for step, _layer, _action in source_schedule) + 1,
        actual_step_indices=actual_steps,
        attention_action_schedule=tuple(
            (step, layer, action)
            for (step, layer), action in sorted(schedule.items())
        ),
        source=V19_LONG_QUALITY_SHIELD_SOURCE,
    )
    if _actual_steps(candidate) != actual_steps:
        raise V19PlanningError("long quality shield changed the Actual trajectory")
    report = require_v19_human_proposal_eligible(
        candidate,
        v19_long_horizon_screening_policy(
            max(step for step, _layer, _action in source_schedule) + 1
        ),
    )
    return V19LongQualityShieldResult(
        blueprint=candidate,
        source_execution_digest=v19_blueprint_execution_digest(source),
        actual_step_indices=actual_steps,
        terminal_actual_step_indices=terminal_steps,
        source_action_cell_counts=_cell_counts(source_schedule, actual_steps),
        candidate_action_cell_counts=_cell_counts(
            runtime_schedule_from_blueprint(candidate), actual_steps
        ),
        core_upgraded_cells=core_upgraded,
        terminal_upgraded_cells=terminal_upgraded,
        constraint_report=report,
    )


def propose_v19_long_quality_frontier(
    *,
    total_steps: int,
    acceleration: float,
    width: int,
    height: int,
    latent_frames: int,
    spatial_tokens_per_frame: int,
    packed_tokens: int,
    condition_count: int = 0,
    reference_images: int = 0,
    reference_audio: int = 0,
    reference_videos: int = 0,
) -> V19LongQualityFrontierSelection:
    """Compile one measured long-quality proposal without semantic routing.

    This is deliberately not wired into the release selector.  Batch11/12
    provide physical costs and static screening only; continuous Human review
    is still pending.  Unsupported workloads raise and must be delegated to a
    certified selector or Dense fallback by the caller.
    """

    if total_steps != 20:
        raise V19PlanningError(
            "long quality frontier currently requires the measured 20 steps"
        )
    if not (
        V19_LONG_QUALITY_ACCELERATION_FLOOR <= acceleration <= 100.0
    ):
        raise V19PlanningError(
            "long quality acceleration lies outside the measured [75, 100] range"
        )
    if any(
        value != 0
        for value in (
            condition_count,
            reference_images,
            reference_audio,
            reference_videos,
        )
    ):
        raise V19PlanningError(
            "long quality reference-bearing workload is not yet measured"
        )
    envelope = next(
        (
            row
            for row in V19_LONG_QUALITY_COST_ENVELOPES
            if row.width == width
            and row.height == height
            and row.latent_frames == latent_frames
            and row.spatial_tokens_per_frame == spatial_tokens_per_frame
            and row.minimum_packed_tokens <= packed_tokens <= row.maximum_packed_tokens
        ),
        None,
    )
    if envelope is None:
        raise V19PlanningError(
            "long quality geometry or packed-token envelope is unmeasured"
        )
    source = build_v19_long_horizon_round188_replay(
        candidate_id=V19_LONG_QUALITY_V012_CANDIDATE,
        total_steps=total_steps,
        acceleration=V19_LONG_QUALITY_ACCELERATION_FLOOR,
    )
    if acceleration < V19_LONG_QUALITY_BALANCED_THRESHOLD:
        operating_point = "quality_core_0.5_terminal_0.25"
        result = build_v19_long_quality_shield(
            source,
            candidate_id=V19_LONG_QUALITY_V014B_CANDIDATE,
            spec=V19LongQualityShieldSpec(core_action="sparse_topk_0.5"),
        )
        blueprint = result.blueprint
        proposal_eligible = result.constraint_report.proposal_eligible
    elif acceleration < V19_LONG_QUALITY_FAST_THRESHOLD:
        operating_point = "balanced_core_0.25_terminal_0.25"
        result = build_v19_long_quality_shield(
            source,
            candidate_id=V19_LONG_QUALITY_V014A_CANDIDATE,
            spec=V19LongQualityShieldSpec(core_action="sparse_topk_0.25"),
        )
        blueprint = result.blueprint
        proposal_eligible = result.constraint_report.proposal_eligible
    else:
        operating_point = "fast_round188"
        blueprint = source
        proposal_eligible = require_v19_human_proposal_eligible(
            source, v19_long_horizon_screening_policy(total_steps)
        ).proposal_eligible
    return V19LongQualityFrontierSelection(
        blueprint=blueprint,
        operating_point=operating_point,
        acceleration=float(acceleration),
        envelope_id=envelope.envelope_id,
        observed_e2e_seconds=envelope.e2e_seconds(blueprint.candidate_id),
        observed_denoise_seconds=envelope.denoise_seconds(blueprint.candidate_id),
        evidence_ids=envelope.evidence_ids,
        proposal_eligible=proposal_eligible,
    )


__all__ = [
    "V19_LONG_QUALITY_ACCELERATION_FLOOR",
    "V19_LONG_QUALITY_BALANCED_THRESHOLD",
    "V19_LONG_QUALITY_CORE_LAYERS",
    "V19_LONG_QUALITY_COST_ENVELOPES",
    "V19_LONG_QUALITY_FAST_THRESHOLD",
    "V19_LONG_QUALITY_FRONTIER_POLICY",
    "V19_LONG_QUALITY_SHIELD_SOURCE",
    "V19LongQualityCostEnvelope",
    "V19LongQualityFrontierSelection",
    "V19LongQualityShieldResult",
    "V19LongQualityShieldSpec",
    "build_v19_long_quality_shield",
    "propose_v19_long_quality_frontier",
]
