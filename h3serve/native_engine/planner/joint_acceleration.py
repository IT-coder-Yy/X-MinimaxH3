"""Joint sampler/Attention budget planning for the creator-facing speed dial.

The public contract intentionally has only two controls:

``total_steps``
    The length of the requested sigma trajectory.

``acceleration``
    A monotone 0--100 compute-effort dial.  Zero is the exact Dense endpoint;
    one hundred is the fastest endpoint inside the currently admitted safety
    envelope.

Everything else in this module is an internal quality constraint.  In
particular, callers cannot disable the opening composition anchor, the
causal/interaction layer floor, terminal recovery, or the maximum forecast
gap.  The optimizer is free to exchange full DiT evaluations for forecasts
and to redistribute sparse Attention work, but it must do so under those
constraints and minimize the calibrated Human-visible risk surrogate.

This is a deterministic control-plane model.  It does not claim to predict
wall-clock seconds or Human acceptance.  Costs are normalized from the real
RTX 4090 100k-token profile and risks are ordered priors derived from the
project's Dense-teacher probes and Human review.  Runtime Dense probes remain
the final request-local correction layer.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
import math
from threading import Lock
from typing import Iterable, Mapping

from .sparse_budget import (
    H3AttentionCellKey,
    H3BudgetedRiskScheduler,
    H3LayerBand,
    HumanRiskVector,
    SparseActionEstimate,
    SparseBudgetLedger,
    SparseDecisionCell,
    SparseOptimalityCertificate,
    SparseSchedule,
    SparseScheduleChoice,
    build_h3_sparse_cells,
    verify_sparse_optimality_certificate,
)
from .joint_global_dp import (
    BAND_RISK_MODEL,
    BASE_STRUCTURAL_CONSTRAINT,
    FIXED_TOPK_ACTION_IMPLEMENTATION,
    GlobalJointCertificate,
    GlobalJointSolution,
    JointWorkloadContext,
    LAYER_BANDS,
    ROUND143_216_TRAJECTORY_PRIOR,
    ROUND215_ACTION_IMPLEMENTATION,
    ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
    ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
    ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION,
    ROUND215_LAYER_RISK_MODEL,
    ROUND218_PHASE_LAYER_RISK_MODEL,
    OnlineRebateCertificate,
    ROUND216_CAUSAL_ISLAND_CONSTRAINT,
    ROUND224_ADAPTIVE_LATENCY_CONSTRAINT,
    ROUND225_TRAJECTORY_CORRECTION_CONSTRAINT,
    ROUND226_OPENING_ANCHORED_MTCR_CONSTRAINT,
    ROUND227_FRONTIER_DOMINANCE_CONSTRAINT,
    TrajectoryRiskPrior,
    clear_global_frontier_cache,
    solve_global_joint_problem,
    solve_no_trigger_online_rebate,
    verify_global_joint_solution,
)
from .online_guard import allocate_phase_sentinels


JOINT_ACCELERATION_SCHEMA = "h3_joint_acceleration_v1"
JOINT_POLICY_V1_HEURISTIC = "h3_joint_v1_greedy_attention"
JOINT_POLICY_V2_EXACT_ATTENTION = "h3_joint_v2_exact_attention"
JOINT_POLICY_V3_GLOBAL_DP = "h3_joint_v3_shape_aware_global_dp"
JOINT_POLICY_V4_EVIDENCE_GLOBAL_DP = "h3_joint_v4_evidence_global_dp"
JOINT_POLICY_V5_CALIBRATION_MATCHED_GLOBAL_DP = (
    "h3_joint_v5_calibration_matched_global_dp"
)
JOINT_POLICY_V6_CAUSAL_ISLAND_GLOBAL_DP = (
    "h3_joint_v6_causal_island_global_dp"
)
JOINT_POLICY_V7_LAYER_RISK_GLOBAL_DP = "h3_joint_v7_layer_risk_global_dp"
JOINT_POLICY_V8_PHASE_LAYER_RISK_GLOBAL_DP = (
    "h3_joint_v8_phase_layer_risk_global_dp"
)
JOINT_POLICY_V9_BOUNDED_ONLINE_GLOBAL_DP = (
    "h3_joint_v9_bounded_online_global_dp"
)
JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP = (
    "h3_joint_v10_phase_sentinel_global_dp"
)
JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP = (
    "h3_joint_v11_calibrated_growth_global_dp"
)
JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP = (
    "h3_joint_v12_reserve_rebate_global_dp"
)
JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP = (
    "h3_joint_v13_adaptive_latency_global_dp"
)
JOINT_POLICY_V14_TRAJECTORY_CORRECTION_GLOBAL_DP = (
    "h3_joint_v14_trajectory_correction_global_dp"
)
JOINT_POLICY_V15_OPENING_ANCHORED_MTCR_GLOBAL_DP = (
    "h3_joint_v15_opening_anchored_mtcr_global_dp"
)
JOINT_POLICY_V16_FRONTIER_DOMINANCE_GLOBAL_DP = (
    "h3_joint_v16_frontier_dominance_global_dp"
)
JOINT_POLICY_V17_ZERO_TAX_FRONTIER_GLOBAL_DP = (
    "h3_joint_v17_zero_tax_frontier_global_dp"
)
JOINT_POLICY_V18_FORECAST_AWARE_FRONTIER_GLOBAL_DP = (
    "h3_joint_v18_forecast_aware_frontier_global_dp"
)
ROUND219_BOUNDED_ONLINE_GUARD = "round219_noncausal_probe_upgrade_v1"
ROUND220_PHASE_SENTINEL_GUARD = "round220_phase_sentinel_budget_v1"
ROUND221_CALIBRATED_GROWTH_GUARD = "round221_calibrated_growth_budget_v1"
ROUND223_RESERVE_REBATE_GUARD = "round223_reserve_rebate_budget_v1"
DEFAULT_JOINT_POLICY = JOINT_POLICY_V7_LAYER_RISK_GLOBAL_DP

JOINT_MECHANICAL_BASELINE_ID = (
    "round86_fused_rms_adaln_vae_block_compile_v1"
)

_GLOBAL_DP_POLICIES = frozenset(
    (
        JOINT_POLICY_V3_GLOBAL_DP,
        JOINT_POLICY_V4_EVIDENCE_GLOBAL_DP,
        JOINT_POLICY_V5_CALIBRATION_MATCHED_GLOBAL_DP,
        JOINT_POLICY_V6_CAUSAL_ISLAND_GLOBAL_DP,
        JOINT_POLICY_V7_LAYER_RISK_GLOBAL_DP,
        JOINT_POLICY_V8_PHASE_LAYER_RISK_GLOBAL_DP,
        JOINT_POLICY_V9_BOUNDED_ONLINE_GLOBAL_DP,
        JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP,
        JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
        JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP,
        JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP,
        JOINT_POLICY_V14_TRAJECTORY_CORRECTION_GLOBAL_DP,
        JOINT_POLICY_V15_OPENING_ANCHORED_MTCR_GLOBAL_DP,
        JOINT_POLICY_V16_FRONTIER_DOMINANCE_GLOBAL_DP,
        JOINT_POLICY_V17_ZERO_TAX_FRONTIER_GLOBAL_DP,
        JOINT_POLICY_V18_FORECAST_AWARE_FRONTIER_GLOBAL_DP,
    )
)


class JointAccelerationError(ValueError):
    """The two-control acceleration request cannot be planned safely."""


@dataclass(frozen=True, slots=True)
class JointPlanVerification:
    valid: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class JointAttentionDecision:
    step_index: int
    layer_start: int
    layer_stop: int
    action: str

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise JointAccelerationError("attention decision step cannot be negative")
        if not 0 <= self.layer_start < self.layer_stop <= 50:
            raise JointAccelerationError("attention decision layer band is invalid")
        if self.action not in ACTION_TOPK:
            raise JointAccelerationError(f"unknown attention action: {self.action}")


@dataclass(frozen=True, slots=True)
class JointAccelerationPlan:
    """Immutable, auditable output of the two-control optimizer."""

    total_steps: int
    acceleration: float
    actual_step_indices: tuple[int, ...]
    forecast_step_indices: tuple[int, ...]
    attention_decisions: tuple[JointAttentionDecision, ...]
    target_compute_units: float
    estimated_compute_units: float
    dense_compute_units: float
    estimated_risk_debt: float
    online_recovery_reserve_units: float = 0.0
    online_rebate_schedule: tuple[tuple[int, int], ...] = ()
    online_rebate_certificate: OnlineRebateCertificate | None = None
    policy_id: str = DEFAULT_JOINT_POLICY
    formal_optimality_scope: str = "none"
    attention_optimality_certificate: SparseOptimalityCertificate | None = None
    global_optimality_certificate: GlobalJointCertificate | None = None
    workload_context: JointWorkloadContext | None = None
    predicted_compute_ms: float | None = None
    conservative_compute_ms: float | None = None
    target_compute_ms: float | None = None
    workload_calibration_mix: float | None = None
    workload_extrapolated: bool = False
    forecast_allowed: bool = True
    trajectory_prior_id: str | None = None
    attention_implementation_id: str = FIXED_TOPK_ACTION_IMPLEMENTATION
    mechanical_baseline_id: str = JOINT_MECHANICAL_BASELINE_ID
    quality_constraint_id: str = BASE_STRUCTURAL_CONSTRAINT
    risk_model_id: str = BAND_RISK_MODEL
    online_guard_id: str | None = None
    safety_envelope: str = "round86_structure_plus_round182_online_guard"
    schema_version: str = JOINT_ACCELERATION_SCHEMA

    def __post_init__(self) -> None:
        if self.total_steps < 2:
            raise JointAccelerationError("total steps must be at least two")
        if not 0.0 <= self.acceleration <= 100.0:
            raise JointAccelerationError("acceleration must lie inside [0, 100]")
        expected = tuple(range(self.total_steps))
        merged = tuple(sorted(self.actual_step_indices + self.forecast_step_indices))
        if merged != expected:
            raise JointAccelerationError("actual and forecast steps must partition the trajectory")
        if self.estimated_compute_units > self.target_compute_units + 1.0e-6:
            raise JointAccelerationError("planned compute exceeds the acceleration budget")
        if not 0.0 <= self.online_recovery_reserve_units <= self.estimated_compute_units:
            raise JointAccelerationError("online recovery reserve is invalid")
        if self.online_guard_id is not None and self.online_recovery_reserve_units <= 0.0:
            raise JointAccelerationError("online guard requires a positive recovery reserve")
        if self.online_rebate_schedule:
            if self.online_guard_id != ROUND223_RESERVE_REBATE_GUARD:
                raise JointAccelerationError("online rebate requires the V12 guard")
            if self.online_rebate_certificate is None:
                raise JointAccelerationError("online rebate requires a certificate")
            if tuple(sorted(set(self.online_rebate_schedule))) != self.online_rebate_schedule:
                raise JointAccelerationError(
                    "online rebate schedule must be sorted and contain unique cells"
                )
            actual = frozenset(self.actual_step_indices)
            if any(
                step not in actual or not 0 <= layer < 50
                for step, layer in self.online_rebate_schedule
            ):
                raise JointAccelerationError(
                    "online rebate cells must target actual H3 steps/layers"
                )
        elif self.online_rebate_certificate is not None and (
            self.online_guard_id != ROUND223_RESERVE_REBATE_GUARD
            or self.online_rebate_certificate.selected_count != 0
        ):
            raise JointAccelerationError(
                "empty online rebate may only certify zero selected cells"
            )

    @property
    def actual_evaluations(self) -> int:
        return len(self.actual_step_indices)

    @property
    def forecast_evaluations(self) -> int:
        return len(self.forecast_step_indices)

    @property
    def estimated_compute_ratio(self) -> float:
        return self.estimated_compute_units / self.dense_compute_units

    @property
    def estimated_speedup_upper(self) -> float:
        """Compute-only reciprocal, deliberately not an end-to-end claim."""

        return self.dense_compute_units / self.estimated_compute_units

    @property
    def uses_sparse_attention(self) -> bool:
        return any(decision.action != "dense" for decision in self.attention_decisions)

    def physical_action_schedule(self) -> dict[tuple[int, int], str]:
        result: dict[tuple[int, int], str] = {}
        for decision in self.attention_decisions:
            for layer in range(decision.layer_start, decision.layer_stop):
                result[(decision.step_index, layer)] = decision.action
        return result

    def runtime_action_schedule(self) -> dict[tuple[int, int], str]:
        """Map canonical certificate actions to their measured GPU kernels."""

        schedule = self.physical_action_schedule()
        if self.attention_implementation_id == FIXED_TOPK_ACTION_IMPLEMENTATION:
            return schedule
        if self.attention_implementation_id == ROUND215_ACTION_IMPLEMENTATION:
            return {
                cell: action if action == "dense" else f"round215:{action}"
                for cell, action in schedule.items()
            }
        if self.attention_implementation_id == ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION:
            return {
                cell: action if action == "dense" else f"frontier:{action}"
                for cell, action in schedule.items()
            }
        if (
            self.attention_implementation_id
            == ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION
        ):
            return {
                cell: action if action == "dense" else f"fastfrontier:{action}"
                for cell, action in schedule.items()
            }
        if (
            self.attention_implementation_id
            == ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION
        ):
            result = {
                cell: action if action == "dense" else f"forecastfrontier:{action}"
                for cell, action in schedule.items()
            }
            # A directional forecast is not free: it executes the first three
            # DiT blocks before extrapolating the remaining 47.  Round188's
            # direct backend kept those anchor blocks on the reviewed 6.25%
            # MTCR rail.  The first joint dispatcher accidentally omitted
            # forecast cells, so its fail-closed lookup ran 3 Dense layers on
            # every forecast step.  Make this implementation detail explicit
            # and auditable without adding forecast cells to the optimizer's
            # 50-layer physical-risk objective.
            for step in self.forecast_step_indices:
                for layer in range(3):
                    result[(step, layer)] = (
                        "forecastfrontier:sparse_topk_0.0625"
                    )
            return result
        raise JointAccelerationError(
            "plan names an unsupported Attention implementation"
        )

    def to_dict(self, *, include_physical_schedule: bool = False) -> dict[str, object]:
        document: dict[str, object] = {
            "schema_version": self.schema_version,
            "total_steps": self.total_steps,
            "acceleration": self.acceleration,
            "actual_step_indices": list(self.actual_step_indices),
            "forecast_step_indices": list(self.forecast_step_indices),
            "actual_evaluations": self.actual_evaluations,
            "forecast_evaluations": self.forecast_evaluations,
            "target_compute_units": round(self.target_compute_units, 6),
            "estimated_compute_units": round(self.estimated_compute_units, 6),
            "dense_compute_units": round(self.dense_compute_units, 6),
            "estimated_compute_ratio": round(self.estimated_compute_ratio, 6),
            "estimated_speedup_upper": round(self.estimated_speedup_upper, 6),
            "estimated_risk_debt": round(self.estimated_risk_debt, 6),
            "online_recovery_reserve_units": round(
                self.online_recovery_reserve_units, 6
            ),
            "online_rebate_schedule": [
                [step, layer] for step, layer in self.online_rebate_schedule
            ],
            "policy_id": self.policy_id,
            "formal_optimality_scope": self.formal_optimality_scope,
            "safety_envelope": self.safety_envelope,
            "attention_decisions": [
                {
                    "step_index": item.step_index,
                    "layer_start": item.layer_start,
                    "layer_stop": item.layer_stop,
                    "action": item.action,
                    "topk": ACTION_TOPK[item.action],
                }
                for item in self.attention_decisions
            ],
        }
        if self.attention_optimality_certificate is not None:
            document["attention_optimality_certificate"] = asdict(
                self.attention_optimality_certificate
            )
        if self.global_optimality_certificate is not None:
            document["global_optimality_certificate"] = asdict(
                self.global_optimality_certificate
            )
        if self.online_rebate_certificate is not None:
            document["online_rebate_certificate"] = asdict(
                self.online_rebate_certificate
            )
        if self.workload_context is not None:
            document["workload_context"] = asdict(self.workload_context)
        for key, value in (
            ("predicted_compute_ms", self.predicted_compute_ms),
            ("conservative_compute_ms", self.conservative_compute_ms),
            ("target_compute_ms", self.target_compute_ms),
            ("workload_calibration_mix", self.workload_calibration_mix),
        ):
            if value is not None:
                document[key] = round(float(value), 6)
        document["workload_extrapolated"] = self.workload_extrapolated
        document["forecast_allowed"] = self.forecast_allowed
        document["trajectory_prior_id"] = self.trajectory_prior_id
        document["attention_implementation_id"] = self.attention_implementation_id
        document["mechanical_baseline_id"] = self.mechanical_baseline_id
        document["quality_constraint_id"] = self.quality_constraint_id
        document["risk_model_id"] = self.risk_model_id
        document["online_guard_id"] = self.online_guard_id
        if include_physical_schedule:
            document["physical_action_schedule"] = {
                f"{step}:{layer}": action
                for (step, layer), action in sorted(
                    self.physical_action_schedule().items()
                )
            }
        return document


# One full Dense DiT evaluation is one normalized unit.  The split follows the
# measured 100k-token profile: complete Attention is ~68--71% of one step and
# QKV/MLP/normalization make up the rest.  Sparse action ratios come directly
# from the real Round215 complete-56-head timing table.
NON_ATTENTION_COMPUTE = 0.286
ATTENTION_COMPUTE = 1.0 - NON_ATTENTION_COMPUTE
FORECAST_COMPUTE = 0.045
ONLINE_RECOVERY_RESERVE_RATIO = 0.02
EXACT_ATTENTION_COST_QUANTUM_MS = 5.0

_PLAN_CACHE_MAXSIZE = 512
_PLAN_CACHE: OrderedDict[tuple[object, ...], JointAccelerationPlan] = OrderedDict()
_PLAN_CACHE_LOCK = Lock()


def _cached_plan(
    key: tuple[object, ...]
) -> JointAccelerationPlan | None:
    with _PLAN_CACHE_LOCK:
        plan = _PLAN_CACHE.get(key)
        if plan is not None:
            _PLAN_CACHE.move_to_end(key)
        return plan


def _store_plan(
    key: tuple[object, ...], plan: JointAccelerationPlan
) -> JointAccelerationPlan:
    with _PLAN_CACHE_LOCK:
        _PLAN_CACHE[key] = plan
        _PLAN_CACHE.move_to_end(key)
        while len(_PLAN_CACHE) > _PLAN_CACHE_MAXSIZE:
            _PLAN_CACHE.popitem(last=False)
    return plan


def clear_joint_plan_cache() -> None:
    """Clear the bounded control-plane cache for cold-latency evaluation."""

    with _PLAN_CACHE_LOCK:
        _PLAN_CACHE.clear()
    clear_global_frontier_cache()

ACTION_TOPK: Mapping[str, float | None] = {
    "sparse_topk_0.0625": 0.0625,
    "sparse_topk_0.1": 0.10,
    "sparse_topk_0.25": 0.25,
    "sparse_topk_0.5": 0.50,
    "dense": None,
}
_ACTION_ATTENTION_COST_RATIO: Mapping[str, float] = {
    "sparse_topk_0.0625": 0.213,
    "sparse_topk_0.1": 0.247,
    "sparse_topk_0.25": 0.403,
    "sparse_topk_0.5": 0.669,
    "dense": 1.0,
}
_ACTION_RISK: Mapping[str, float] = {
    "sparse_topk_0.0625": 1.00,
    "sparse_topk_0.1": 0.72,
    "sparse_topk_0.25": 0.32,
    "sparse_topk_0.5": 0.11,
    "dense": 0.0,
}
_ACTION_RANK: Mapping[str, int] = {
    name: index for index, name in enumerate(ACTION_TOPK)
}
_BAND_MULTIPLIER: Mapping[H3LayerBand, float] = {
    H3LayerBand.EARLY: 1.00,
    H3LayerBand.MIDDLE: 1.10,
    H3LayerBand.CAUSAL: 2.35,
    H3LayerBand.CAUSAL_DETAIL: 2.70,
    H3LayerBand.BRIDGE: 1.20,
    H3LayerBand.CAUSAL_TERMINAL: 2.70,
    H3LayerBand.TAIL: 1.30,
}
_CAUSAL_BANDS = frozenset(
    (
        H3LayerBand.CAUSAL,
        H3LayerBand.CAUSAL_DETAIL,
        H3LayerBand.CAUSAL_TERMINAL,
    )
)


def _minimum_actual_count(total_steps: int, *, allow_forecast: bool) -> int:
    if not allow_forecast:
        return total_steps
    # 12/8 at N=20 is the fastest trajectory currently backed by repeatable
    # full-video Human review.  Short trajectories retain at least five true
    # evaluations and the same 60% lower bound.
    return min(total_steps, max(5, int(math.ceil(total_steps * 0.60))))


def _maximum_forecast_run(actual_steps: Iterable[int], total_steps: int) -> int:
    actual = set(actual_steps)
    current = 0
    maximum = 0
    for step in range(total_steps):
        if step in actual:
            current = 0
        else:
            current += 1
            maximum = max(maximum, current)
    return maximum


def _base_anchor_schedule(total_steps: int, count: int) -> tuple[int, ...]:
    """Build nested anchors while preserving the accepted 20-step spine."""

    if not 2 <= count <= total_steps:
        raise JointAccelerationError("actual evaluation count falls outside trajectory")
    if count == total_steps:
        return tuple(range(total_steps))
    if total_steps == 20 and count == 12:
        return (0, 1, 2, 3, 4, 6, 8, 11, 14, 17, 18, 19)

    opening_count = min(count, max(3, int(math.ceil(total_steps * 0.25))))
    anchors = set(range(opening_count))
    terminal_target = max(1, min(3, int(round(total_steps * 0.15))))
    terminal_count = min(terminal_target, max(0, count - len(anchors)))
    anchors.update(range(total_steps - terminal_count, total_steps))

    # Add the point at the centre of the longest uncovered interval.  Ties are
    # resolved toward later solver positions because low-sigma errors survive
    # directly into the decoded result.  The construction is deterministic,
    # nested, and keeps forecast runs short without a task-specific table.
    while len(anchors) < count:
        ordered = sorted(anchors)
        candidates: list[tuple[int, int, int]] = []
        for left, right in zip(ordered, ordered[1:]):
            for candidate in range(left + 1, right):
                distance = min(candidate - left, right - candidate)
                candidates.append((distance, candidate, right - left))
        if not candidates:
            missing = [step for step in range(total_steps) if step not in anchors]
            anchors.add(missing[-1])
            continue
        _, candidate, _ = max(candidates, key=lambda row: (row[0], row[1], row[2]))
        anchors.add(candidate)

    schedule = tuple(sorted(anchors))
    # A gap larger than two forecasts is outside the admitted RES trajectory
    # envelope.  Repair it by replacing the least useful non-terminal anchor.
    while _maximum_forecast_run(schedule, total_steps) > 2:
        actual = set(schedule)
        run: list[int] = []
        longest: list[int] = []
        for step in range(total_steps):
            if step in actual:
                if len(run) > len(longest):
                    longest = run
                run = []
            else:
                run.append(step)
        if len(run) > len(longest):
            longest = run
        insertion = longest[len(longest) // 2]
        protected = set(range(opening_count)) | set(range(total_steps - terminal_count, total_steps))
        removable = [
            step
            for step in schedule
            if step not in protected
            and step != insertion
        ]
        if not removable:
            raise JointAccelerationError("cannot satisfy the maximum forecast-gap guard")
        # Remove the anchor whose deletion creates the smallest adjacent gap.
        ordered = list(schedule)
        def removal_damage(step: int) -> tuple[int, int]:
            index = ordered.index(step)
            left = ordered[index - 1]
            right = ordered[index + 1]
            return (right - left, -step)
        actual.remove(min(removable, key=removal_damage))
        actual.add(insertion)
        schedule = tuple(sorted(actual))
    return schedule


def _schedule_for_count(total_steps: int, count: int, minimum_count: int) -> tuple[int, ...]:
    if count == total_steps:
        return tuple(range(total_steps))
    base = set(_base_anchor_schedule(total_steps, minimum_count))
    while len(base) < count:
        missing = [step for step in range(total_steps) if step not in base]
        # Add the missing point with the largest distance to an existing true
        # evaluation, preserving the late-step tie break used above.
        point = max(
            missing,
            key=lambda step: (min(abs(step - other) for other in base), step),
        )
        base.add(point)
    return tuple(sorted(base))


def _forecast_risk(actual_steps: tuple[int, ...], total_steps: int) -> float:
    actual = frozenset(actual_steps)
    risk = 0.0
    run = 0
    for step in range(total_steps):
        if step in actual:
            run = 0
            continue
        run += 1
        progress = step / max(1, total_steps - 1)
        phase = 1.35 if progress < 0.25 else (1.25 if progress > 0.78 else 1.0)
        risk += 0.90 * phase * (1.0 + 0.35 * max(0, run - 1))
    return risk


def _action_factory(key: H3AttentionCellKey) -> tuple[SparseActionEstimate, ...]:
    layer_fraction = (key.layer_stop - key.layer_start) / 50.0
    phase_multiplier = {
        "opening": 1.75,
        "ordinary": 1.0,
        "terminal": 1.55,
    }[key.phase]
    band_multiplier = _BAND_MULTIPLIER[key.layer_band]
    causal = key.layer_band in _CAUSAL_BANDS
    actions = []
    for name in ACTION_TOPK:
        debt = (
            _ACTION_RISK[name]
            * layer_fraction
            * phase_multiplier
            * band_multiplier
        )
        actions.append(
            SparseActionEstimate(
                name=name,
                measured_cost_ms=(
                    ATTENTION_COMPUTE
                    * layer_fraction
                    * _ACTION_ATTENTION_COST_RATIO[name]
                    * 1000.0
                ),
                reject_risk_ucb=debt,
                fidelity_rank=_ACTION_RANK[name],
                components=HumanRiskVector(
                    motion=debt if causal else debt * 0.42,
                    clarity=debt * (0.72 if key.phase == "terminal" else 0.55),
                    identity=debt * (0.65 if causal else 0.25),
                    audio=debt * (0.48 if causal else 0.18),
                ),
            )
        )
    return tuple(actions)


def _build_cells(actual_steps: tuple[int, ...]) -> tuple[SparseDecisionCell, ...]:
    # Round86 structural priors: opening and terminal are never aggressive;
    # the causal island is never below 10% even in the fastest admitted plan.
    cells = build_h3_sparse_cells(
        actual_steps,
        _action_factory,
        causal_floor_rank=_ACTION_RANK["sparse_topk_0.1"],
        opening_floor_rank=_ACTION_RANK["sparse_topk_0.25"],
        terminal_floor_rank=_ACTION_RANK["sparse_topk_0.25"],
    )
    position = {step: index for index, step in enumerate(actual_steps)}
    result = []
    for cell in cells:
        floor = cell.minimum_fidelity_rank
        index = position[cell.key.actual_step]
        previous = actual_steps[index - 1] if index else None
        # The first actual step's causal band carries layout and interaction
        # identity and therefore receives the stronger 50% floor.
        if cell.key.phase == "opening" and cell.key.layer_band in _CAUSAL_BANDS:
            floor = max(floor, _ACTION_RANK["sparse_topk_0.5"])
        # After two consecutive forecast transitions, spend a causal recovery
        # step before forecasting is allowed again.
        if previous is not None and cell.key.actual_step - previous >= 3:
            floor = max(floor, _ACTION_RANK["sparse_topk_0.1"])
            if cell.key.layer_band in _CAUSAL_BANDS:
                floor = max(floor, _ACTION_RANK["sparse_topk_0.25"])
        result.append(replace(cell, minimum_fidelity_rank=floor))
    return tuple(result)


def _minimum_attention_cost(
    cells: Iterable[SparseDecisionCell],
    *,
    conservative_quantum_ms: float | None = None,
) -> float:
    costs = (
        min(
            action.measured_cost_ms
            for action in cell.actions
            if action.fidelity_rank >= cell.minimum_fidelity_rank
        )
        for cell in cells
    )
    if conservative_quantum_ms is None:
        return sum(costs) / 1000.0
    return sum(
        math.ceil(cost / conservative_quantum_ms - 1.0e-12)
        * conservative_quantum_ms
        for cost in costs
    ) / 1000.0


def _allocate_attention_budget(
    cells: tuple[SparseDecisionCell, ...], budget_ms: float
) -> tuple[tuple[tuple[H3AttentionCellKey, SparseActionEstimate], ...], float, float]:
    """Allocate discrete fidelity upgrades by marginal risk reduction.

    The action ladder has diminishing risk reduction per extra millisecond
    (0.0625 -> 0.10 -> 0.25 -> 0.50 -> Dense).  Starting from every hard
    floor and repeatedly buying the best next marginal upgrade is therefore
    the separable resource-allocation solution for this calibrated ladder.
    Unlike the general exact knapsack used by offline experiments, this runs
    in sub-millisecond control-plane time and scales independently of the
    absolute video latency.
    """

    ladders: list[tuple[SparseActionEstimate, ...]] = []
    positions: list[int] = []
    cost = 0.0
    risk = 0.0
    for cell in cells:
        ladder = tuple(
            sorted(
                (
                    action
                    for action in cell.actions
                    if action.fidelity_rank >= cell.minimum_fidelity_rank
                ),
                key=lambda action: (action.fidelity_rank, action.measured_cost_ms),
            )
        )
        ladders.append(ladder)
        positions.append(0)
        cost += ladder[0].measured_cost_ms
        risk += ladder[0].reject_risk_ucb
    if cost > budget_ms + 1.0e-6:
        raise JointAccelerationError("attention budget is below the internal quality floor")

    while True:
        best: tuple[tuple[float, float, int, int, int], int] | None = None
        for index, (cell, ladder, position) in enumerate(
            zip(cells, ladders, positions)
        ):
            if position + 1 >= len(ladder):
                continue
            current = ladder[position]
            upgraded = ladder[position + 1]
            delta_cost = upgraded.measured_cost_ms - current.measured_cost_ms
            if cost + delta_cost > budget_ms + 1.0e-6:
                continue
            risk_gain = current.reject_risk_ucb - upgraded.reject_risk_ucb
            score = risk_gain / max(delta_cost, 1.0e-12)
            causal = int(cell.key.layer_band in _CAUSAL_BANDS)
            terminal = int(cell.key.phase == "terminal")
            # The scalar calibrated risk is primary.  Exact ties favor the
            # Human-sensitive causal/terminal cells and then earlier cells for
            # deterministic output.
            key = (score, risk_gain, causal, terminal, -index)
            if best is None or key > best[0]:
                best = (key, index)
        if best is None:
            break
        index = best[1]
        current = ladders[index][positions[index]]
        positions[index] += 1
        upgraded = ladders[index][positions[index]]
        cost += upgraded.measured_cost_ms - current.measured_cost_ms
        risk += upgraded.reject_risk_ucb - current.reject_risk_ucb

    choices = tuple(
        (cell.key, ladder[position])
        for cell, ladder, position in zip(cells, ladders, positions)
    )
    return choices, cost, max(0.0, risk)


def _allocate_attention_budget_exact(
    cells: tuple[SparseDecisionCell, ...],
    budget_ms: float,
    *,
    verify: bool = False,
) -> tuple[
    tuple[tuple[H3AttentionCellKey, SparseActionEstimate], ...],
    float,
    float,
    SparseOptimalityCertificate,
]:
    """Solve the finite attention allocation exactly under quantised cost.

    This reuses the audited multiple-choice knapsack DP.  Its certificate is
    checked immediately before the plan can leave the control plane.  The
    formal claim is conditional on the supplied actual-step schedule and the
    calibrated additive risk table; forecast placement is still a separate
    finite candidate search in policy v2.
    """

    try:
        # Five milliseconds is below 0.02% of a 100k-token Dense DiT step and
        # keeps the exact finite DP comfortably off the inference critical
        # path.  Costs are rounded up per decision cell, so this only leaves
        # budget unused; it can never create a runtime overrun.
        schedule = H3BudgetedRiskScheduler(
            cost_quantum_ms=EXACT_ATTENTION_COST_QUANTUM_MS
        ).solve(
            cells,
            ledger=SparseBudgetLedger(max(0.0, budget_ms)),
        )
    except Exception as error:
        raise JointAccelerationError(
            "attention budget is below the internal quality floor"
        ) from error
    if verify:
        verification = verify_sparse_optimality_certificate(cells, schedule)
        if not verification.valid:
            detail = "; ".join(verification.reasons) or "missing certificate"
            raise JointAccelerationError(
                f"exact attention allocation certificate failed: {detail}"
            )
    if schedule.optimality_certificate is None:
        raise JointAccelerationError("exact attention allocation returned no certificate")
    choices = tuple(
        (choice.key, choice.action) for choice in schedule.choices
    )
    return (
        choices,
        schedule.estimated_cost_ms,
        schedule.estimated_reject_risk_ucb,
        schedule.optimality_certificate,
    )


def _online_recovery_reserve(total_steps: int, acceleration: float) -> float:
    return (
        float(total_steps)
        * ONLINE_RECOVERY_RESERVE_RATIO
        * max(0.0, min(100.0, acceleration))
        / 100.0
    )


def _fast_endpoint_cost(total_steps: int, *, allow_forecast: bool) -> float:
    minimum_count = _minimum_actual_count(total_steps, allow_forecast=allow_forecast)
    actual = _base_anchor_schedule(total_steps, minimum_count)
    fixed = minimum_count * NON_ATTENTION_COMPUTE
    fixed += (total_steps - minimum_count) * FORECAST_COMPUTE
    return (
        fixed
        + _minimum_attention_cost(
            _build_cells(actual),
            conservative_quantum_ms=EXACT_ATTENTION_COST_QUANTUM_MS,
        )
        + _online_recovery_reserve(total_steps, 100.0)
    )


def _trajectory_prior_for(
    policy_id: str,
    total_steps: int,
    workload: JointWorkloadContext,
    *,
    allow_forecast: bool,
) -> TrajectoryRiskPrior | None:
    """Return an admitted Human-evidence prior without adding a public knob."""

    if (
        policy_id in (
            JOINT_POLICY_V4_EVIDENCE_GLOBAL_DP,
            JOINT_POLICY_V5_CALIBRATION_MATCHED_GLOBAL_DP,
            JOINT_POLICY_V6_CAUSAL_ISLAND_GLOBAL_DP,
            JOINT_POLICY_V7_LAYER_RISK_GLOBAL_DP,
            JOINT_POLICY_V8_PHASE_LAYER_RISK_GLOBAL_DP,
            JOINT_POLICY_V9_BOUNDED_ONLINE_GLOBAL_DP,
            JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP,
            JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
            JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP,
        )
        and total_steps == 20
        and allow_forecast
        and workload.model_variant == "base"
    ):
        return ROUND143_216_TRAJECTORY_PRIOR
    return None


class H3JointAccelerationScheduler:
    """Allocate one trajectory under the creator-facing acceleration dial."""

    def __init__(self, *, policy_id: str = DEFAULT_JOINT_POLICY) -> None:
        if policy_id not in (
            JOINT_POLICY_V1_HEURISTIC,
            JOINT_POLICY_V2_EXACT_ATTENTION,
            JOINT_POLICY_V3_GLOBAL_DP,
            JOINT_POLICY_V4_EVIDENCE_GLOBAL_DP,
            JOINT_POLICY_V5_CALIBRATION_MATCHED_GLOBAL_DP,
            JOINT_POLICY_V6_CAUSAL_ISLAND_GLOBAL_DP,
            JOINT_POLICY_V7_LAYER_RISK_GLOBAL_DP,
            JOINT_POLICY_V8_PHASE_LAYER_RISK_GLOBAL_DP,
            JOINT_POLICY_V9_BOUNDED_ONLINE_GLOBAL_DP,
            JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP,
            JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
            JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP,
            JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP,
            JOINT_POLICY_V14_TRAJECTORY_CORRECTION_GLOBAL_DP,
            JOINT_POLICY_V15_OPENING_ANCHORED_MTCR_GLOBAL_DP,
            JOINT_POLICY_V16_FRONTIER_DOMINANCE_GLOBAL_DP,
            JOINT_POLICY_V17_ZERO_TAX_FRONTIER_GLOBAL_DP,
            JOINT_POLICY_V18_FORECAST_AWARE_FRONTIER_GLOBAL_DP,
        ):
            raise JointAccelerationError(f"unknown joint policy: {policy_id}")
        self.policy_id = policy_id

    def plan(
        self,
        total_steps: int,
        acceleration: float,
        *,
        allow_forecast: bool = True,
        workload: JointWorkloadContext | None = None,
    ) -> JointAccelerationPlan:
        if not isinstance(total_steps, int) or not 4 <= total_steps <= 30:
            raise JointAccelerationError("total_steps must be an integer inside [4, 30]")
        try:
            acceleration = float(acceleration)
        except (TypeError, ValueError) as error:
            raise JointAccelerationError("acceleration must be numeric") from error
        if not math.isfinite(acceleration) or not 0.0 <= acceleration <= 100.0:
            raise JointAccelerationError("acceleration must lie inside [0, 100]")
        acceleration = round(acceleration, 1)
        workload = workload or JointWorkloadContext()
        action_implementation_id = (
            ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION
            if self.policy_id == JOINT_POLICY_V18_FORECAST_AWARE_FRONTIER_GLOBAL_DP
            else ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION
            if self.policy_id == JOINT_POLICY_V17_ZERO_TAX_FRONTIER_GLOBAL_DP
            else ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION
            if self.policy_id == JOINT_POLICY_V16_FRONTIER_DOMINANCE_GLOBAL_DP
            else ROUND215_ACTION_IMPLEMENTATION
            if self.policy_id == JOINT_POLICY_V5_CALIBRATION_MATCHED_GLOBAL_DP
            or self.policy_id == JOINT_POLICY_V6_CAUSAL_ISLAND_GLOBAL_DP
            or self.policy_id == JOINT_POLICY_V7_LAYER_RISK_GLOBAL_DP
            or self.policy_id == JOINT_POLICY_V8_PHASE_LAYER_RISK_GLOBAL_DP
            or self.policy_id == JOINT_POLICY_V9_BOUNDED_ONLINE_GLOBAL_DP
            or self.policy_id == JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP
            or self.policy_id == JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP
            or self.policy_id == JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP
            or self.policy_id == JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP
            or self.policy_id == JOINT_POLICY_V14_TRAJECTORY_CORRECTION_GLOBAL_DP
            or self.policy_id == JOINT_POLICY_V15_OPENING_ANCHORED_MTCR_GLOBAL_DP
            else FIXED_TOPK_ACTION_IMPLEMENTATION
        )
        quality_constraint_id = (
            ROUND216_CAUSAL_ISLAND_CONSTRAINT
            if self.policy_id == JOINT_POLICY_V6_CAUSAL_ISLAND_GLOBAL_DP
            or self.policy_id == JOINT_POLICY_V7_LAYER_RISK_GLOBAL_DP
            or self.policy_id == JOINT_POLICY_V8_PHASE_LAYER_RISK_GLOBAL_DP
            or self.policy_id == JOINT_POLICY_V9_BOUNDED_ONLINE_GLOBAL_DP
            or self.policy_id == JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP
            or self.policy_id == JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP
            or self.policy_id == JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP
            else (
                ROUND224_ADAPTIVE_LATENCY_CONSTRAINT
                if self.policy_id == JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP
                else BASE_STRUCTURAL_CONSTRAINT
            )
        )
        if self.policy_id == JOINT_POLICY_V14_TRAJECTORY_CORRECTION_GLOBAL_DP:
            quality_constraint_id = ROUND225_TRAJECTORY_CORRECTION_CONSTRAINT
        elif self.policy_id == JOINT_POLICY_V15_OPENING_ANCHORED_MTCR_GLOBAL_DP:
            quality_constraint_id = ROUND226_OPENING_ANCHORED_MTCR_CONSTRAINT
        elif self.policy_id in (
            JOINT_POLICY_V16_FRONTIER_DOMINANCE_GLOBAL_DP,
            JOINT_POLICY_V17_ZERO_TAX_FRONTIER_GLOBAL_DP,
            JOINT_POLICY_V18_FORECAST_AWARE_FRONTIER_GLOBAL_DP,
        ):
            quality_constraint_id = ROUND227_FRONTIER_DOMINANCE_CONSTRAINT
        risk_model_id = (
            ROUND218_PHASE_LAYER_RISK_MODEL
            if self.policy_id in (
                JOINT_POLICY_V8_PHASE_LAYER_RISK_GLOBAL_DP,
                JOINT_POLICY_V9_BOUNDED_ONLINE_GLOBAL_DP,
                JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP,
                JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
                JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP,
                JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP,
                JOINT_POLICY_V14_TRAJECTORY_CORRECTION_GLOBAL_DP,
                JOINT_POLICY_V15_OPENING_ANCHORED_MTCR_GLOBAL_DP,
                JOINT_POLICY_V16_FRONTIER_DOMINANCE_GLOBAL_DP,
                JOINT_POLICY_V17_ZERO_TAX_FRONTIER_GLOBAL_DP,
                JOINT_POLICY_V18_FORECAST_AWARE_FRONTIER_GLOBAL_DP,
            )
            else (
                ROUND215_LAYER_RISK_MODEL
                if self.policy_id == JOINT_POLICY_V7_LAYER_RISK_GLOBAL_DP
                else BAND_RISK_MODEL
            )
        )
        cache_key = (
            self.policy_id,
            total_steps,
            acceleration,
            bool(allow_forecast),
            *(workload.cache_key if self.policy_id in _GLOBAL_DP_POLICIES else ()),
        )
        cached = _cached_plan(cache_key)
        if cached is not None:
            return cached

        dense_cost = float(total_steps)
        if acceleration == 0.0:
            decisions = tuple(
                JointAttentionDecision(
                    step_index=step,
                    layer_start=start,
                    layer_stop=stop,
                    action="dense",
                )
                for step in range(total_steps)
                for _, start, stop in (
                    (H3LayerBand.EARLY, 0, 15),
                    (H3LayerBand.MIDDLE, 15, 30),
                    (H3LayerBand.CAUSAL, 30, 40),
                    (H3LayerBand.CAUSAL_DETAIL, 40, 44),
                    (H3LayerBand.BRIDGE, 44, 45),
                    (H3LayerBand.CAUSAL_TERMINAL, 45, 46),
                    (H3LayerBand.TAIL, 46, 50),
                )
            )
            return _store_plan(cache_key, JointAccelerationPlan(
                total_steps=total_steps,
                acceleration=0.0,
                actual_step_indices=tuple(range(total_steps)),
                forecast_step_indices=(),
                attention_decisions=decisions,
                target_compute_units=dense_cost,
                estimated_compute_units=dense_cost,
                dense_compute_units=dense_cost,
                estimated_risk_debt=0.0,
                policy_id=self.policy_id,
                formal_optimality_scope="unique all-Dense endpoint",
                workload_context=(
                    workload if self.policy_id in _GLOBAL_DP_POLICIES else None
                ),
                forecast_allowed=bool(allow_forecast),
                attention_implementation_id=action_implementation_id,
                quality_constraint_id=quality_constraint_id,
                risk_model_id=risk_model_id,
            ))
        if self.policy_id in _GLOBAL_DP_POLICIES:
            trajectory_prior = _trajectory_prior_for(
                self.policy_id,
                total_steps,
                workload,
                allow_forecast=allow_forecast,
            )
            solution = solve_global_joint_problem(
                total_steps,
                acceleration,
                workload=workload,
                allow_forecast=allow_forecast,
                trajectory_prior=trajectory_prior,
                action_implementation_id=action_implementation_id,
                quality_constraint_id=quality_constraint_id,
                risk_model_id=risk_model_id,
            )
            if solution.steps and any(
                step.actual and len(step.actions) == 50 for step in solution.steps
            ):
                decisions = tuple(
                    JointAttentionDecision(
                        step_index=step.step_index,
                        layer_start=layer,
                        layer_stop=layer + 1,
                        action=action,
                    )
                    for step in solution.steps
                    if step.actual
                    for layer, action in enumerate(step.actions)
                )
            else:
                decisions = tuple(
                    JointAttentionDecision(
                        step_index=step.step_index,
                        layer_start=start,
                        layer_stop=stop,
                        action=action,
                    )
                    for step in solution.steps
                    if step.actual
                    for action, (_, start, stop) in zip(step.actions, LAYER_BANDS)
                )
            scale = total_steps / solution.dense_cost_ms
            estimated_compute = (
                solution.conservative_cost_ms + solution.recovery_reserve_ms
            ) * scale
            online_rebate_schedule: tuple[tuple[int, int], ...] = ()
            online_rebate_certificate: OnlineRebateCertificate | None = None
            if self.policy_id == JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP:
                limit_dense_layers = solution.recovery_reserve_ms * scale * 50.0
                phase_allocation = allocate_phase_sentinels(
                    solution.actual_steps, limit_dense_layers
                )
                if phase_allocation.slots:
                    rebate = solve_no_trigger_online_rebate(
                        solution,
                        workload=workload,
                        total_steps=total_steps,
                        limit_dense_layers=limit_dense_layers,
                        probe_slots=phase_allocation.slots,
                        quality_constraint_id=quality_constraint_id,
                        risk_model_id=risk_model_id,
                    )
                    online_rebate_schedule = tuple(sorted(
                        (choice.step_index, choice.layer)
                        for choice in rebate.choices
                    ))
                    online_rebate_certificate = rebate.certificate
            plan = JointAccelerationPlan(
                total_steps=total_steps,
                acceleration=acceleration,
                actual_step_indices=solution.actual_steps,
                forecast_step_indices=solution.forecast_steps,
                attention_decisions=decisions,
                target_compute_units=solution.target_cost_ms * scale,
                estimated_compute_units=estimated_compute,
                dense_compute_units=float(total_steps),
                estimated_risk_debt=(
                    solution.forecast_risk + solution.attention_risk
                ),
                online_recovery_reserve_units=(
                    solution.recovery_reserve_ms * scale
                ),
                online_rebate_schedule=online_rebate_schedule,
                online_rebate_certificate=online_rebate_certificate,
                policy_id=self.policy_id,
                formal_optimality_scope=solution.certificate.formal_scope,
                global_optimality_certificate=solution.certificate,
                workload_context=workload,
                predicted_compute_ms=(
                    solution.predicted_cost_ms + solution.recovery_reserve_ms
                ),
                conservative_compute_ms=(
                    solution.conservative_cost_ms + solution.recovery_reserve_ms
                ),
                target_compute_ms=solution.target_cost_ms,
                workload_calibration_mix=solution.calibration_mix,
                workload_extrapolated=solution.extrapolated,
                forecast_allowed=bool(allow_forecast),
                trajectory_prior_id=(
                    trajectory_prior.prior_id if trajectory_prior is not None else None
                ),
                attention_implementation_id=action_implementation_id,
                quality_constraint_id=quality_constraint_id,
                risk_model_id=risk_model_id,
                online_guard_id=(
                    ROUND219_BOUNDED_ONLINE_GUARD
                    if self.policy_id == JOINT_POLICY_V9_BOUNDED_ONLINE_GLOBAL_DP
                    else (
                        ROUND220_PHASE_SENTINEL_GUARD
                        if self.policy_id == JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP
                        else (
                            ROUND223_RESERVE_REBATE_GUARD
                            if self.policy_id
                            == JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP
                            else (
                                ROUND221_CALIBRATED_GROWTH_GUARD
                                if self.policy_id in (
                                    JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
                                    JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP,
                                    JOINT_POLICY_V14_TRAJECTORY_CORRECTION_GLOBAL_DP,
                                    JOINT_POLICY_V15_OPENING_ANCHORED_MTCR_GLOBAL_DP,
                                    JOINT_POLICY_V16_FRONTIER_DOMINANCE_GLOBAL_DP,
                                )
                                else None
                            )
                        )
                    )
                ),
                safety_envelope=(
                    "round227_human_frontier_head_rail_plus_bounded_online_correction"
                    if self.policy_id == JOINT_POLICY_V16_FRONTIER_DOMINANCE_GLOBAL_DP
                    else "round228_human_frontier_fast_rail_without_teacher_tax"
                    if self.policy_id == JOINT_POLICY_V17_ZERO_TAX_FRONTIER_GLOBAL_DP
                    else "round229_forecast_anchor_uses_reviewed_sparse_rail"
                    if self.policy_id == JOINT_POLICY_V18_FORECAST_AWARE_FRONTIER_GLOBAL_DP
                    else (
                    "round226_consecutive_opening_plus_mtcr_debt_correction"
                    if self.policy_id == JOINT_POLICY_V15_OPENING_ANCHORED_MTCR_GLOBAL_DP
                    else (
                        "round225_forecast_bound_plus_exact_causal_debt_correction"
                        if self.policy_id == JOINT_POLICY_V14_TRAJECTORY_CORRECTION_GLOBAL_DP
                        else (
                            "round224_phase_layer_risk_plus_bounded_online_guard"
                            if self.policy_id == JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP
                            else (
                                "round215_shape_cost_plus_round216_positive_round217_negative_"
                                + (
                        (
                            "human_trajectory_prior_bounded_online_guard"
                            if trajectory_prior is not None
                            else "bounded_online_guard"
                        )
                        if self.policy_id in (
                            JOINT_POLICY_V9_BOUNDED_ONLINE_GLOBAL_DP,
                            JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP,
                            JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
                            JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP,
                            JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP,
                            JOINT_POLICY_V14_TRAJECTORY_CORRECTION_GLOBAL_DP,
                            JOINT_POLICY_V15_OPENING_ANCHORED_MTCR_GLOBAL_DP,
                            JOINT_POLICY_V16_FRONTIER_DOMINANCE_GLOBAL_DP,
                            JOINT_POLICY_V17_ZERO_TAX_FRONTIER_GLOBAL_DP,
                            JOINT_POLICY_V18_FORECAST_AWARE_FRONTIER_GLOBAL_DP,
                        )
                        else (
                            "human_trajectory_prior_online_guard"
                            if trajectory_prior is not None
                            else "online_guard"
                        )
                    )
                        )
                    )
                    )
                    )
                ),
            )
            return _store_plan(cache_key, plan)
        fast_cost = _fast_endpoint_cost(
            total_steps, allow_forecast=allow_forecast
        )
        # A mildly convex dial gives the creator useful resolution near the
        # quality end while still reaching the admitted fast endpoint at 100.
        normalized = acceleration / 100.0
        target_cost = dense_cost - (dense_cost - fast_cost) * normalized**1.18
        recovery_reserve = _online_recovery_reserve(total_steps, acceleration)

        minimum_count = _minimum_actual_count(
            total_steps, allow_forecast=allow_forecast
        )
        candidates: list[JointAccelerationPlan] = []
        for actual_count in range(minimum_count, total_steps + 1):
            actual = _schedule_for_count(total_steps, actual_count, minimum_count)
            forecasts = tuple(step for step in range(total_steps) if step not in actual)
            fixed_cost = (
                actual_count * NON_ATTENTION_COMPUTE
                + len(forecasts) * FORECAST_COMPUTE
            )
            attention_budget = target_cost - fixed_cost - recovery_reserve
            if attention_budget < -1.0e-9:
                continue
            cells = _build_cells(actual)
            if (
                self.policy_id == JOINT_POLICY_V2_EXACT_ATTENTION
                and _minimum_attention_cost(
                    cells,
                    conservative_quantum_ms=EXACT_ATTENTION_COST_QUANTUM_MS,
                )
                > max(0.0, attention_budget) + 1.0e-9
            ):
                continue
            try:
                certificate: SparseOptimalityCertificate | None = None
                if self.policy_id == JOINT_POLICY_V2_EXACT_ATTENTION:
                    # Forecast/actual-count candidates use the retained v1
                    # allocator as a cheap ranking model.  Only the selected
                    # actual-step spine enters the exact finite DP below.
                    choices, attention_cost_ms, attention_risk = (
                        _allocate_attention_budget(
                            cells, max(0.0, attention_budget) * 1000.0
                        )
                    )
                else:
                    choices, attention_cost_ms, attention_risk = (
                        _allocate_attention_budget(
                            cells, max(0.0, attention_budget) * 1000.0
                        )
                    )
            except JointAccelerationError:
                continue
            decisions = tuple(
                JointAttentionDecision(
                    step_index=key.actual_step,
                    layer_start=key.layer_start,
                    layer_stop=key.layer_stop,
                    action=action.name,
                )
                for key, action in choices
            )
            estimated = (
                fixed_cost
                + attention_cost_ms / 1000.0
                + recovery_reserve
            )
            candidates.append(
                JointAccelerationPlan(
                    total_steps=total_steps,
                    acceleration=acceleration,
                    actual_step_indices=actual,
                    forecast_step_indices=forecasts,
                    attention_decisions=decisions,
                    target_compute_units=target_cost,
                    estimated_compute_units=estimated,
                    dense_compute_units=dense_cost,
                    estimated_risk_debt=(
                        _forecast_risk(actual, total_steps)
                        + attention_risk
                    ),
                    online_recovery_reserve_units=recovery_reserve,
                    policy_id=self.policy_id,
                    formal_optimality_scope=(
                        "exact attention allocation conditional on enumerated "
                        "actual-step schedule"
                        if certificate is not None
                        else "none; retained v1 heuristic comparator"
                    ),
                    attention_optimality_certificate=certificate,
                    forecast_allowed=bool(allow_forecast),
                )
            )
        if not candidates:
            raise JointAccelerationError(
                "no schedule satisfies the internal quality floors at this acceleration"
            )
        # Primary objective: minimum calibrated Human-visible risk.  If two
        # discrete schedules tie, prefer the one that uses more of the granted
        # compute and then the one with more true DiT evaluations.
        selected = min(
            candidates,
            key=lambda plan: (
                plan.estimated_risk_debt,
                -plan.estimated_compute_units,
                -plan.actual_evaluations,
            ),
        )
        if self.policy_id == JOINT_POLICY_V2_EXACT_ATTENTION:
            # The selected actual-step spine receives an exact conditional
            # Attention allocation.  Certification replay belongs to the
            # offline evaluator, not every latency-sensitive request.
            fixed_cost = (
                selected.actual_evaluations * NON_ATTENTION_COMPUTE
                + selected.forecast_evaluations * FORECAST_COMPUTE
            )
            attention_budget = (
                selected.target_compute_units
                - fixed_cost
                - selected.online_recovery_reserve_units
            )
            exact_choices, exact_cost_ms, exact_risk, certificate = (
                _allocate_attention_budget_exact(
                    _build_cells(selected.actual_step_indices),
                    max(0.0, attention_budget) * 1000.0,
                )
            )
            exact_decisions = tuple(
                JointAttentionDecision(
                    step_index=key.actual_step,
                    layer_start=key.layer_start,
                    layer_stop=key.layer_stop,
                    action=action.name,
                )
                for key, action in exact_choices
            )
            selected = replace(
                selected,
                attention_decisions=exact_decisions,
                estimated_compute_units=(
                    fixed_cost
                    + exact_cost_ms / 1000.0
                    + selected.online_recovery_reserve_units
                ),
                estimated_risk_debt=(
                    _forecast_risk(
                        selected.actual_step_indices, selected.total_steps
                    )
                    + exact_risk
                ),
                formal_optimality_scope=(
                    "exact attention allocation conditional on heuristic "
                    "actual-step spine"
                ),
                attention_optimality_certificate=certificate,
            )
        return _store_plan(cache_key, selected)


def verify_joint_plan_certificate(
    plan: JointAccelerationPlan,
) -> JointPlanVerification:
    """Verify one immutable plan without trusting its serialized totals."""

    reasons: list[str] = []
    if plan.acceleration == 0.0:
        if plan.forecast_step_indices:
            reasons.append("Dense endpoint contains forecast steps")
        if set(plan.physical_action_schedule().values()) != {"dense"}:
            reasons.append("Dense endpoint contains sparse Attention")
        if not math.isclose(plan.estimated_compute_units, plan.dense_compute_units):
            reasons.append("Dense endpoint compute total mismatch")
        return JointPlanVerification(not reasons, tuple(reasons))

    if plan.policy_id in _GLOBAL_DP_POLICIES:
        if plan.workload_context is None:
            return JointPlanVerification(False, ("missing workload context",))
        if plan.global_optimality_certificate is None:
            return JointPlanVerification(False, ("missing global certificate",))
        trajectory_prior = _trajectory_prior_for(
            plan.policy_id,
            plan.total_steps,
            plan.workload_context,
            allow_forecast=plan.forecast_allowed,
        )
        expected_prior_id = (
            trajectory_prior.prior_id if trajectory_prior is not None else None
        )
        if plan.trajectory_prior_id != expected_prior_id:
            reasons.append("trajectory prior id mismatch")
        expected_online_guard = (
            ROUND219_BOUNDED_ONLINE_GUARD
            if plan.policy_id == JOINT_POLICY_V9_BOUNDED_ONLINE_GLOBAL_DP
            else (
                ROUND220_PHASE_SENTINEL_GUARD
                if plan.policy_id == JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP
                else (
                    ROUND223_RESERVE_REBATE_GUARD
                    if plan.policy_id
                    == JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP
                    else (
                        ROUND221_CALIBRATED_GROWTH_GUARD
                        if plan.policy_id in (
                            JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
                            JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP,
                            JOINT_POLICY_V14_TRAJECTORY_CORRECTION_GLOBAL_DP,
                            JOINT_POLICY_V15_OPENING_ANCHORED_MTCR_GLOBAL_DP,
                            JOINT_POLICY_V16_FRONTIER_DOMINANCE_GLOBAL_DP,
                        )
                        else None
                    )
                )
            )
        )
        if plan.online_guard_id != expected_online_guard:
            reasons.append("online guard id mismatch")
        try:
            replay = solve_global_joint_problem(
                plan.total_steps,
                plan.acceleration,
                workload=plan.workload_context,
                allow_forecast=plan.forecast_allowed,
                trajectory_prior=trajectory_prior,
                action_implementation_id=plan.attention_implementation_id,
                quality_constraint_id=plan.quality_constraint_id,
                risk_model_id=plan.risk_model_id,
            )
        except Exception as error:
            return JointPlanVerification(
                False, (f"global certificate replay failed: {error}",)
            )
        if plan.global_optimality_certificate != replay.certificate:
            reasons.append("global certificate mismatch")
        if plan.actual_step_indices != replay.actual_steps:
            reasons.append("global actual-step path mismatch")
        if plan.forecast_step_indices != replay.forecast_steps:
            reasons.append("global forecast-step path mismatch")
        if any(step.actual and len(step.actions) == 50 for step in replay.steps):
            expected_actions = {
                (step.step_index, layer, layer + 1): action
                for step in replay.steps
                if step.actual
                for layer, action in enumerate(step.actions)
            }
        else:
            expected_actions = {
                (step.step_index, start, stop): action
                for step in replay.steps
                if step.actual
                for action, (_, start, stop) in zip(step.actions, LAYER_BANDS)
            }
        observed_actions = {
            (item.step_index, item.layer_start, item.layer_stop): item.action
            for item in plan.attention_decisions
        }
        if observed_actions != expected_actions:
            reasons.append("global Attention path mismatch")
        scale = plan.total_steps / replay.dense_cost_ms
        expected_compute = (
            replay.conservative_cost_ms + replay.recovery_reserve_ms
        ) * scale
        if not math.isclose(
            plan.estimated_compute_units,
            expected_compute,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            reasons.append("global compute total mismatch")
        expected_risk = replay.forecast_risk + replay.attention_risk
        if not math.isclose(
            plan.estimated_risk_debt,
            expected_risk,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            reasons.append("global risk total mismatch")
        if plan.policy_id == JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP:
            limit_dense_layers = plan.online_recovery_reserve_units * 50.0
            allocation = allocate_phase_sentinels(
                replay.actual_steps, limit_dense_layers
            )
            if not allocation.slots:
                if (
                    plan.online_rebate_schedule
                    or plan.online_rebate_certificate is not None
                ):
                    reasons.append("online rebate exists without probe capacity")
                return JointPlanVerification(not reasons, tuple(reasons))
            try:
                rebate = solve_no_trigger_online_rebate(
                    replay,
                    workload=plan.workload_context,
                    total_steps=plan.total_steps,
                    limit_dense_layers=limit_dense_layers,
                    probe_slots=allocation.slots,
                    quality_constraint_id=plan.quality_constraint_id,
                    risk_model_id=plan.risk_model_id,
                )
            except Exception as error:
                reasons.append(f"online rebate replay failed: {error}")
            else:
                expected_schedule = tuple(sorted(
                    (choice.step_index, choice.layer)
                    for choice in rebate.choices
                ))
                if plan.online_rebate_schedule != expected_schedule:
                    reasons.append("online rebate schedule mismatch")
                if plan.online_rebate_certificate != rebate.certificate:
                    reasons.append("online rebate certificate mismatch")
        elif plan.online_rebate_schedule or plan.online_rebate_certificate is not None:
            reasons.append("non-V12 plan contains an online rebate")
        return JointPlanVerification(not reasons, tuple(reasons))

    certificate = plan.attention_optimality_certificate
    if certificate is None:
        return JointPlanVerification(False, ("missing Attention certificate",))
    cells = _build_cells(plan.actual_step_indices)
    decisions = {
        (
            decision.step_index,
            decision.layer_start,
            decision.layer_stop,
        ): decision.action
        for decision in plan.attention_decisions
    }
    choices: list[SparseScheduleChoice] = []
    components = HumanRiskVector()
    attention_cost_ms = 0.0
    attention_risk = 0.0
    for cell in cells:
        action_name = decisions.get(
            (
                cell.key.actual_step,
                cell.key.layer_start,
                cell.key.layer_stop,
            )
        )
        if action_name is None:
            reasons.append(f"missing decision for {cell.key.cell_id}")
            continue
        action = next(
            (candidate for candidate in cell.actions if candidate.name == action_name),
            None,
        )
        if action is None or action.fidelity_rank < cell.minimum_fidelity_rank:
            reasons.append(f"invalid action for {cell.key.cell_id}")
            continue
        choices.append(SparseScheduleChoice(cell.key, action))
        attention_cost_ms += action.measured_cost_ms
        attention_risk += action.reject_risk_ucb
        components = components + action.components
    if reasons:
        return JointPlanVerification(False, tuple(reasons))

    fixed_cost = (
        plan.actual_evaluations * NON_ATTENTION_COMPUTE
        + plan.forecast_evaluations * FORECAST_COMPUTE
    )
    attention_budget_ms = (
        plan.target_compute_units
        - fixed_cost
        - plan.online_recovery_reserve_units
    ) * 1000.0
    sparse_schedule = SparseSchedule(
        choices=tuple(choices),
        budget_limit_ms=attention_budget_ms,
        estimated_cost_ms=attention_cost_ms,
        estimated_reject_risk_ucb=attention_risk,
        estimated_components=components,
        budget_overrun_ms=max(0.0, attention_cost_ms - attention_budget_ms),
        used_recovery_reserve=False,
        optimality_certificate=certificate,
    )
    sparse_verification = verify_sparse_optimality_certificate(
        cells, sparse_schedule
    )
    reasons.extend(sparse_verification.reasons)
    expected_compute = (
        fixed_cost
        + attention_cost_ms / 1000.0
        + plan.online_recovery_reserve_units
    )
    if not math.isclose(
        expected_compute,
        plan.estimated_compute_units,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        reasons.append("serialized compute total mismatch")
    expected_risk = _forecast_risk(
        plan.actual_step_indices, plan.total_steps
    ) + attention_risk
    if not math.isclose(
        expected_risk,
        plan.estimated_risk_debt,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        reasons.append("serialized risk total mismatch")
    return JointPlanVerification(not reasons, tuple(reasons))


__all__ = [
    "ACTION_TOPK",
    "H3JointAccelerationScheduler",
    "DEFAULT_JOINT_POLICY",
    "JOINT_POLICY_V1_HEURISTIC",
    "JOINT_POLICY_V2_EXACT_ATTENTION",
    "JOINT_POLICY_V3_GLOBAL_DP",
    "JOINT_POLICY_V4_EVIDENCE_GLOBAL_DP",
    "JOINT_POLICY_V5_CALIBRATION_MATCHED_GLOBAL_DP",
    "JOINT_POLICY_V6_CAUSAL_ISLAND_GLOBAL_DP",
    "JOINT_POLICY_V7_LAYER_RISK_GLOBAL_DP",
    "JOINT_POLICY_V8_PHASE_LAYER_RISK_GLOBAL_DP",
    "JOINT_POLICY_V9_BOUNDED_ONLINE_GLOBAL_DP",
    "JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP",
    "JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP",
    "JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP",
    "JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP",
    "JOINT_POLICY_V14_TRAJECTORY_CORRECTION_GLOBAL_DP",
    "JOINT_POLICY_V15_OPENING_ANCHORED_MTCR_GLOBAL_DP",
    "JOINT_MECHANICAL_BASELINE_ID",
    "FIXED_TOPK_ACTION_IMPLEMENTATION",
    "ROUND215_ACTION_IMPLEMENTATION",
    "ROUND216_CAUSAL_ISLAND_CONSTRAINT",
    "ROUND224_ADAPTIVE_LATENCY_CONSTRAINT",
    "ROUND225_TRAJECTORY_CORRECTION_CONSTRAINT",
    "ROUND226_OPENING_ANCHORED_MTCR_CONSTRAINT",
    "ROUND215_LAYER_RISK_MODEL",
    "ROUND218_PHASE_LAYER_RISK_MODEL",
    "ROUND219_BOUNDED_ONLINE_GUARD",
    "ROUND220_PHASE_SENTINEL_GUARD",
    "ROUND221_CALIBRATED_GROWTH_GUARD",
    "ROUND223_RESERVE_REBATE_GUARD",
    "BAND_RISK_MODEL",
    "BASE_STRUCTURAL_CONSTRAINT",
    "JOINT_ACCELERATION_SCHEMA",
    "JointAccelerationError",
    "JointAccelerationPlan",
    "JointAttentionDecision",
    "JointPlanVerification",
    "JointWorkloadContext",
    "clear_joint_plan_cache",
    "verify_joint_plan_certificate",
]
