"""Final two-control Pareto surface for all H3 Base service families.

The public API remains ``(sampling_steps, acceleration)``.  Internally the
surface has one Human-constrained quality knee at acceleration 75:

* ``0`` is exact Dense execution;
* ``75`` replays/interpolates reviewed workload anchors;
* ``100`` is an explicitly aggressive Attention endpoint.

This is one nested, resource-conditioned strategy curve.  The Human-selected
C02 720p15 calibration anchor is immutable in the production selector.
Historical C01/C03/V009 snapshots remain available only through the explicitly
named research compiler so that old experiments stay reproducible.  Neither
compiler inspects prompt words, scene labels, seeds or generated pixels.  FL2VA
and Ref2VA therefore share the same mechanism, while reference-media pressure
continuously shifts the request toward the safer side of the curve.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math

from .v19_planner import V19PlanningError, V19WorkloadContext
from .v19_runtime_bridge import ROUND229_FORECAST_ANCHOR
from .v24_deployment import (
    V24CurveProfile,
    V24DeploymentSelection,
    V24HumanAnchor,
    V24_HUMAN_ANCHORS,
    V24_HUMAN_EVIDENCE_MAX_PACKED_TOKENS,
    V24_MAX_PACKED_TOKENS,
    _ACTION_RISK,
    _ATTENTION_COMPUTE,
    _ATTENTION_COST_RATIO,
    _FORECAST_COMPUTE,
    _NON_ATTENTION_COMPUTE,
    _RANK_TO_CANONICAL,
    _anchor_bracket,
    _build_upgrade_chain,
    _compute_units,
    _context_progress,
    _dense_selection,
    _endpoint_actual_steps,
    _execution_digest,
    _layer_risk_weight,
    _longest_forecast_run,
    _phase_risk_weight,
    _runtime_action,
    _solve_nested_budget,
    _state_compute_units,
    _trajectory_risk,
    _video_tokens,
)


V24_FINAL_SCHEMA = "h3_pareto_v24_final_runtime_selection_v5"
V24_FINAL_POLICY_ID = "h3_pareto_v24_human_knee_continuous_release_v5"
V24_FINAL_QUALITY_KNEE = 75.0
V24_FINAL_HISTORICAL_V009 = "v24_final_stable_v009"
V24_FINAL_DEFAULT_CANDIDATE = "v24_final_c02_round2_trajectory_u7p00"
_V24_FINAL_AGGRESSIVE_OPTIMIZER = "v24_nested_attention_downgrade_v1"

V24_FINAL_HISTORICAL_CURVE_PROFILE = V24CurveProfile(
    profile_id="v24_final_contact_peripheral_shield_v1",
    budget_exponent=1.12,
    forecast_risk_scale=25.0,
    forecast_run_coupling=0.22,
    opening_amplitude=1.80,
    terminal_amplitude=1.50,
    causal_layer_amplitude=1.20,
    bridge_layer_amplitude=0.70,
)

# Human feedback on the product console found acceleration 35 too close to the
# Dense endpoint in wall time.  Lowering only the Dense-to-knee response
# exponent moves compute onto the same already-reviewed nested path sooner;
# it does not change the Human-selected acceleration-75 knot or any 75--100
# aggressive schedule.  At 720p15/20 steps, 35 now compiles 12.9038 modeled
# compute units instead of 14.2700 (1.106x less DiT work), using 13 Actual
# evaluations with a small top-k 0.5 allocation rather than 14 fully Dense
# Actual evaluations.
V24_FINAL_CURVE_PROFILE = V24CurveProfile(
    profile_id="v24_final_contact_peripheral_shield_v2_low_control_response",
    budget_exponent=0.95,
    forecast_risk_scale=25.0,
    forecast_run_coupling=0.22,
    opening_amplitude=1.80,
    terminal_amplitude=1.50,
    causal_layer_amplitude=1.20,
    bridge_layer_amplitude=0.70,
)


@dataclass(frozen=True, slots=True)
class V24FinalReleaseCandidate:
    """One versioned 720p15 knot on the otherwise shared final surface."""

    candidate_id: str
    long_anchor: V24HumanAnchor
    estimated_compute_units: float
    human_status: str
    role: str


def _anchor(
    *,
    anchor_id: str,
    rows: tuple[tuple[int, str], ...],
    execution_digest: str,
    strategy_digest: str,
    evidence: tuple[str, ...],
) -> V24HumanAnchor:
    return V24HumanAnchor(
        anchor_id=anchor_id,
        video_tokens=98_440,
        rows=rows,
        source_execution_digest=execution_digest,
        artifact_sha256=(strategy_digest,),
        evidence=evidence,
    )


_C01 = _anchor(
    anchor_id="long_98k_final_c01_v014b_shield",
    rows=(
        (0, "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqhhhhhhhhhhhhhhhqq"),
        (1, "tttttttttttttttttttttqqqqqqqqqqqqqqqqqqhhhhhhhqqqq"),
        (2, "ttttttttttttttttttttttttttqqqqqqqqqqqqqhhhhhqhqqqq"),
        (3, "ttttttttttttttttttttttttttttqqqqqqqqqqqhhhhhqhqqqq"),
        (4, "ttttttttttttttttttttttttttttttqqqqqqqqqhhhhhqhqqqt"),
        (6, "tttttttttttttttttttttttttttttttttqqqqqqhhhhhqhqttt"),
        (8, "tttttttttttttttttttttttttttttttttttqqqqhhhhhqhtttt"),
        (11, "tttttttttttttttttttttttttttttttttttttqqhhhhhqhtttt"),
        (14, "ttttttttttttttttttttttttttttttttttqqqqqhhhhhqhqttt"),
        (17, "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqhhhhhqhqqqq"),
        (18, "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqhhhhhqhqqqq"),
        (19, "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqhhhhhhhhhhhqqqq"),
    ),
    execution_digest="3fe39984bc60fa9a471020cd63bd789281bc96601d9a966c82b59a4665ccb0e3",
    strategy_digest="1a8859f0b2645d95c03340475a02f7e381163739553606c524e279f9eca28c2c",
    evidence=(
        "V014b trajectory plus contact/peripheral fidelity shield",
        "Human review passed; retained as a historical mechanism control",
    ),
)

_C02 = _anchor(
    anchor_id="long_98k_final_c02_round2_trajectory",
    rows=(
        (0, "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqhhhhhhhhhhhhhhhhhhhhq"),
        (1, "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqhhhhhhhhhhhhhhhhqq"),
        (2, "tttttttttttttttttqqqqqqqqqqqqqqqqqqqqqqqqhhhhhqqqq"),
        (3, "tttttttttttttttttttttttttqqqqqqqqqqqqqqqqqqqqqqqqq"),
        (4, "tttttttttttttttttttttttttttqqqqqqqqqqqqqqqqqqqqqqq"),
        (6, "tttttttttttttttttttttttttttttqqqqqqqqqqqqqqqqqqqqt"),
        (8, "tttttttttttttttttttttttttttttttqqqqqqqqqqqqqqqqqtt"),
        (10, "tttttttttttttttttttttttttttttttqqqqqqqqqqqqqqqqqtt"),
        (12, "tttttttttttttttttttttttttttttttqqqqqqqqqqqqqqqqqtt"),
        (15, "tttttttttttttttttttttttttttttqqqqqqqqqqqqqqqqqqqqt"),
        (18, "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqhhhhhhhhhhhqqqq"),
        (19, "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqhhhhhhhhhhhhhhhhhqq"),
    ),
    execution_digest="46105dcb98bb3375df161647627303f95ce0f2c333e66ec6bd8412c6a150ff4e",
    strategy_digest="6f1023db0b4689fb88204ee414778511dffff276d4ff066311040276e31dc847",
    evidence=(
        "Round2 correction trajectory had no reported background anomaly",
        "Human review selected this calibration for the production surface",
    ),
)

_C03 = _anchor(
    anchor_id="long_98k_final_c03_round2_trajectory_shield",
    rows=(
        (0, "qqqqqqqqqqqqqqqqqqqqqqqqqqqhhhhhhhhhhhhhhhDDDhhhhh"),
        (1, "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqhhhhhhhhhhhhhhhhhhhq"),
        (2, "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqhhhhhhhhhhhhhhhhqq"),
        (3, "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqhhhhhhhhhhqqqq"),
        (4, "tttttttttttttttttttttttqqqqqqqqqqqqqqqqqqqqqqqqqqq"),
        (6, "tttttttttttttttttttttttttttqqqqqqqqqqqqqqqqqqqqqqq"),
        (8, "ttttttttttttttttttttttttttttqqqqqqqqqqqqqqqqqqqqqq"),
        (10, "tttttttttttttttttttttttttttttqqqqqqqqqqqqqqqqqqqqt"),
        (12, "tttttttttttttttttttttttttttttqqqqqqqqqqqqqqqqqqqqq"),
        (15, "ttttttttttttttttttttttttttqqqqqqqqqqqqqqqqqqqqqqqq"),
        (18, "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqhhhhhhhhhhhhhhhhqq"),
        (19, "qqqqqqqqqqqqqqqqqqqqqqqqqqqqhhhhhhhhhhhhhhhhhhhhhh"),
    ),
    execution_digest="973c47f7c374088e99aa10e2b368c0664609542ac1198ca6035afd5ad498620a",
    strategy_digest="81ec8a8274c0449a9e0949499cf7699973c35198629e39cd207cc0c46eb09ef1",
    evidence=(
        "Round2 correction trajectory with extra peripheral fidelity",
        "Human review found weaker edge flicker but worse key-handoff motion",
    ),
)

_V009 = next(
    anchor for anchor in V24_HUMAN_ANCHORS
    if anchor.anchor_id == "long_98k_v009_stable"
)

V24_FINAL_RELEASE_CANDIDATES = {
    V24_FINAL_HISTORICAL_V009: V24FinalReleaseCandidate(
        candidate_id=V24_FINAL_HISTORICAL_V009,
        long_anchor=_V009,
        estimated_compute_units=6.483,
        human_status="human_accepted_historical_baseline",
        role="historical V009 long-horizon comparison anchor",
    ),
    "v24_final_c01_v014b_shield_u7p00": V24FinalReleaseCandidate(
        candidate_id="v24_final_c01_v014b_shield_u7p00",
        long_anchor=_C01,
        estimated_compute_units=6.998402640000005,
        human_status="human_review_passed_alternative",
        role="reviewed V014b trajectory mechanism alternative",
    ),
    "v24_final_c02_round2_trajectory_u7p00": V24FinalReleaseCandidate(
        candidate_id="v24_final_c02_round2_trajectory_u7p00",
        long_anchor=_C02,
        estimated_compute_units=6.999088080000013,
        human_status="accepted_release_default",
        role=(
            "final Human-selected Round2 trajectory: contact/motion priority "
            "with the best reviewed speed tradeoff"
        ),
    ),
    "v24_final_c03_round2_trajectory_u7p30": V24FinalReleaseCandidate(
        candidate_id="v24_final_c03_round2_trajectory_u7p30",
        long_anchor=_C03,
        estimated_compute_units=7.298611080000014,
        human_status="human_review_passed_edge_shield_tradeoff",
        role=(
            "reviewed stronger edge shield; retained as an offline research "
            "alternative because handoff motion regressed"
        ),
    ),
}

def _resolve_v24_research_candidate(value: str) -> V24FinalReleaseCandidate:
    """Resolve an exact historical id for offline experiment replay only."""

    candidate_id = value.strip()
    try:
        return V24_FINAL_RELEASE_CANDIDATES[candidate_id]
    except KeyError as error:
        choices = ", ".join(sorted(V24_FINAL_RELEASE_CANDIDATES))
        raise V19PlanningError(
            f"unknown V24 final candidate {candidate_id!r}; choose one of {choices}"
        ) from error


def _surface(candidate: V24FinalReleaseCandidate) -> tuple[V24HumanAnchor, ...]:
    return (
        V24_HUMAN_ANCHORS[0],
        V24_HUMAN_ANCHORS[1],
        candidate.long_anchor,
        V24_HUMAN_ANCHORS[3],
    )


def _surface_digest(anchors: tuple[V24HumanAnchor, ...]) -> str:
    return hashlib.sha256(json.dumps(
        [anchor.digest for anchor in anchors],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _bracket(
    anchors: tuple[V24HumanAnchor, ...],
    video_tokens: int,
) -> tuple[V24HumanAnchor, V24HumanAnchor, float]:
    if anchors == V24_HUMAN_ANCHORS:
        return _anchor_bracket(video_tokens)
    if video_tokens <= anchors[0].video_tokens:
        return anchors[0], anchors[0], 0.0
    if video_tokens >= anchors[-1].video_tokens:
        return anchors[-1], anchors[-1], 0.0
    for lower, upper in zip(anchors, anchors[1:]):
        if lower.video_tokens <= video_tokens <= upper.video_tokens:
            if video_tokens == lower.video_tokens:
                return lower, lower, 0.0
            if video_tokens == upper.video_tokens:
                return upper, upper, 0.0
            mix = (
                math.log(video_tokens) - math.log(lower.video_tokens)
            ) / (
                math.log(upper.video_tokens) - math.log(lower.video_tokens)
            )
            return lower, upper, min(1.0, max(0.0, mix))
    raise AssertionError("unreachable V24 final anchor bracket")


@dataclass(frozen=True, slots=True)
class _Downgrade:
    step: int
    layer: int
    from_rank: int
    to_rank: int
    saved_compute: float
    added_risk: float

    @property
    def penalty(self) -> float:
        return self.added_risk / max(self.saved_compute, 1.0e-12)

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "layer": self.layer,
            "from_rank": self.from_rank,
            "to_rank": self.to_rank,
            "saved_compute": self.saved_compute,
            "added_risk": self.added_risk,
        }


@lru_cache(maxsize=256)
def _build_downgrade_chain(
    frozen_state: tuple[tuple[int, int, int, str | None], ...],
    *,
    total_steps: int,
    curve: V24CurveProfile,
) -> tuple[_Downgrade, ...]:
    ranks = {
        (step, layer): rank
        for step, layer, rank, _prefix in frozen_state
    }
    operations: list[_Downgrade] = []
    while any(rank > 0 for rank in ranks.values()):
        choices: list[_Downgrade] = []
        for (step, layer), rank in ranks.items():
            if rank <= 0:
                continue
            next_rank = rank - 1
            saved = (
                _ATTENTION_COMPUTE / 50.0
                * (
                    _ATTENTION_COST_RATIO[_RANK_TO_CANONICAL[rank]]
                    - _ATTENTION_COST_RATIO[_RANK_TO_CANONICAL[next_rank]]
                )
            )
            added_risk = (
                _phase_risk_weight(step, total_steps, curve)
                * _layer_risk_weight(layer, curve)
                * (_ACTION_RISK[next_rank] - _ACTION_RISK[rank])
            )
            choices.append(_Downgrade(
                step=step,
                layer=layer,
                from_rank=rank,
                to_rank=next_rank,
                saved_compute=saved,
                added_risk=added_risk,
            ))
        selected = min(
            choices,
            key=lambda item: (
                item.penalty,
                item.added_risk,
                -item.saved_compute,
                item.step,
                item.layer,
            ),
        )
        operations.append(selected)
        ranks[(selected.step, selected.layer)] = selected.to_rank
    return tuple(operations)


def _solve_aggressive_attention(
    *,
    total_steps: int,
    actual: tuple[int, ...],
    knee_state: dict[tuple[int, int], tuple[int, str | None]],
    progress: float,
    curve: V24CurveProfile,
) -> tuple[
    dict[tuple[int, int], tuple[int, str | None]],
    dict[str, object],
]:
    state = dict(knee_state)
    ranks = {cell: rank for cell, (rank, _prefix) in state.items()}
    knee_cost = _state_compute_units(
        total_steps=total_steps,
        actual=set(actual),
        ranks=ranks,
    )
    floor_ranks = {cell: 0 for cell in ranks}
    floor_cost = _state_compute_units(
        total_steps=total_steps,
        actual=set(actual),
        ranks=floor_ranks,
    )
    target = knee_cost - (knee_cost - floor_cost) * progress ** 1.08
    chain = _build_downgrade_chain(
        tuple(
            (step, layer, rank, prefix)
            for (step, layer), (rank, prefix) in sorted(state.items())
        ),
        total_steps=total_steps,
        curve=curve,
    )
    current = knee_cost
    applied: list[_Downgrade] = []
    for operation in chain:
        proposed = current - operation.saved_compute
        if proposed < target - 1.0e-12:
            break
        _rank, prefix = state[(operation.step, operation.layer)]
        state[(operation.step, operation.layer)] = (
            operation.to_rank,
            prefix,
        )
        current = proposed
        applied.append(operation)
    chain_digest = hashlib.sha256(json.dumps(
        [operation.to_dict() for operation in chain],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return state, {
        "optimizer_id": _V24_FINAL_AGGRESSIVE_OPTIMIZER,
        "objective": (
            "minimize modeled Human-risk increase per saved measured compute"
        ),
        "formal_scope": (
            "single nested per-cell Attention downgrade chain beyond the "
            "Human quality knee; Actual/Forecast placement remains protected"
        ),
        "global_human_optimality_claimed": False,
        "chain_digest": chain_digest,
        "chain_length": len(chain),
        "applied_downgrades": len(applied),
        "target_compute_units": target,
        "achieved_compute_units": current,
        "knee_compute_units": knee_cost,
        "aggressive_floor_compute_units": floor_cost,
        "next_downgrade": (
            None
            if len(applied) == len(chain)
            else chain[len(applied)].to_dict()
        ),
    }


def _trajectory_demotion_chain(
    actual: tuple[int, ...],
    *,
    total_steps: int,
    mandatory: tuple[int, ...],
    curve: V24CurveProfile,
) -> tuple[int, ...]:
    """Find a nested aggressive-only correction-removal chain.

    The quality knee and most of the aggressive segment preserve its complete
    correction trajectory.  Only the top 20% of the public aggressive range
    may remove up to 20% of knee Actual steps.  Four opening evaluations, two
    terminal evaluations and request-mandatory steps remain immutable.
    """

    state = set(actual)
    minimum_count = max(2, int(math.ceil(len(actual) * 0.80)))
    protected = (
        set(range(min(total_steps, max(2, int(math.ceil(total_steps * 0.20))))))
        | set(range(max(0, total_steps - 2), total_steps))
        | set(mandatory)
    )
    chain: list[int] = []
    while len(state) > minimum_count:
        current_risk = _trajectory_risk(state, total_steps, curve)
        choices: list[tuple[float, float, int]] = []
        for step in state - protected:
            proposed = set(state)
            proposed.remove(step)
            if len(_longest_forecast_run(proposed, total_steps)) > 4:
                continue
            added_risk = (
                _trajectory_risk(proposed, total_steps, curve) - current_risk
            )
            choices.append((
                added_risk,
                _phase_risk_weight(step, total_steps, curve),
                step,
            ))
        if not choices:
            break
        _risk, _phase, selected = min(choices)
        state.remove(selected)
        chain.append(selected)
    return tuple(chain)


class _V24ParetoSurfaceCompiler:
    """Shared compiler used by production and explicit offline replay."""

    policy_id = V24_FINAL_POLICY_ID

    def __init__(
        self,
        *,
        candidate: V24FinalReleaseCandidate,
        curve: V24CurveProfile = V24_FINAL_CURVE_PROFILE,
    ) -> None:
        self.candidate = candidate
        self.curve = curve
        self.anchors = _surface(self.candidate)
        self.human_surface_digest = _surface_digest(self.anchors)

    def _dense(
        self,
        workload: V19WorkloadContext,
        *,
        acceleration: float,
        reason: str,
        video_tokens: int | None,
    ) -> V24DeploymentSelection:
        legacy = _dense_selection(
            workload,
            acceleration=acceleration,
            reason=reason,
            video_tokens=video_tokens,
        )
        summary = dict(legacy.summary)
        summary.update({
            "schema_version": V24_FINAL_SCHEMA,
            "policy_id": V24_FINAL_POLICY_ID,
            "human_surface_digest": self.human_surface_digest,
            "quality_knee_acceleration": V24_FINAL_QUALITY_KNEE,
            "prompt_semantics_used": False,
            "public_controls": ["sampling_steps", "acceleration"],
            "release_candidate": {
                "candidate_id": self.candidate.candidate_id,
                "human_status": self.candidate.human_status,
                "role": self.candidate.role,
            },
        })
        return V24DeploymentSelection(
            actual_step_indices=legacy.actual_step_indices,
            attention_action_schedule=legacy.attention_action_schedule,
            summary=summary,
            schema_version=V24_FINAL_SCHEMA,
        )

    def select(
        self,
        *,
        workload: V19WorkloadContext,
        acceleration: float,
        required_actual_step_indices: tuple[int, ...] = (),
    ) -> V24DeploymentSelection:
        if workload.steps is None:
            raise V19PlanningError("V24 final selection requires total steps")
        total_steps = int(workload.steps)
        if not 4 <= total_steps <= 30:
            raise V19PlanningError("V24 final steps must lie inside [4, 30]")
        try:
            acceleration = float(acceleration)
        except (TypeError, ValueError) as error:
            raise V19PlanningError("V24 final acceleration must be numeric") from error
        if not math.isfinite(acceleration) or not 0.0 <= acceleration <= 100.0:
            raise V19PlanningError(
                "V24 final acceleration must lie inside [0, 100]"
            )
        acceleration = round(acceleration, 1)
        required = tuple(sorted(set(
            int(step) for step in required_actual_step_indices
        )))
        if required != required_actual_step_indices or any(
            step < 0 or step >= total_steps for step in required
        ):
            raise V19PlanningError("V24 final required Actual steps are invalid")
        try:
            video_tokens = _video_tokens(workload)
        except V19PlanningError:
            return self._dense(
                workload,
                acceleration=acceleration,
                reason="geometry_unavailable_dense_fallback",
                video_tokens=None,
            )
        if acceleration == 0.0:
            return self._dense(
                workload,
                acceleration=acceleration,
                reason="zero_acceleration_dense_endpoint",
                video_tokens=video_tokens,
            )
        if (
            workload.model_variant != "base"
            or workload.device_arch != "sm89"
            or workload.sampler != "res_multistep"
            or workload.scheduler != "simple"
        ):
            return self._dense(
                workload,
                acceleration=acceleration,
                reason="unsupported_runtime_identity_dense_fallback",
                video_tokens=video_tokens,
            )
        if workload.packed_tokens > V24_MAX_PACKED_TOKENS:
            return self._dense(
                workload,
                acceleration=acceleration,
                reason="packed_token_envelope_exceeded_dense_fallback",
                video_tokens=video_tokens,
            )

        lower, upper, mix = _bracket(self.anchors, video_tokens)
        endpoint = _endpoint_actual_steps(
            total_steps=total_steps,
            lower=lower,
            upper=upper,
            mix=mix,
        )
        requested_progress = acceleration / 100.0
        trajectory_progress, attention_progress, guards = _context_progress(
            workload,
            video_tokens=video_tokens,
            requested_progress=requested_progress,
        )
        evidence_extrapolated = (
            workload.packed_tokens > V24_HUMAN_EVIDENCE_MAX_PACKED_TOKENS
        )
        if evidence_extrapolated:
            guards = (*guards, "xlong_anchor_shape_extrapolation")
        context_factor = (
            1.0
            if requested_progress <= 1.0e-12
            else attention_progress / requested_progress
        )
        effective_acceleration = min(
            100.0,
            max(0.0, acceleration * context_factor),
        )
        mandatory_actual = set(required)
        if workload.reference_videos:
            mandatory_actual.update(range(total_steps))
        mandatory = tuple(sorted(mandatory_actual))
        forecast_risk_multiplier = self.curve.forecast_risk_scale * (
            1.0
            + 3.0 * max(
                0.0,
                1.0 - trajectory_progress / max(attention_progress, 1.0e-12),
            )
        )

        operations, initial_rows, initial_actual, chain_digest = (
            _build_upgrade_chain(
                total_steps=total_steps,
                lower=lower,
                upper=upper,
                mix=round(mix, 12),
                endpoint=endpoint,
                mandatory_actual=mandatory,
                forecast_risk_multiplier=round(
                    forecast_risk_multiplier, 12
                ),
                curve=self.curve,
            )
        )
        knee_cost = _state_compute_units(
            total_steps=total_steps,
            actual=set(initial_actual),
            ranks={
                (step, layer): rank
                for step, layer, rank, _prefix in initial_rows
            },
        )
        if effective_acceleration <= V24_FINAL_QUALITY_KNEE:
            segment = "dense_to_human_quality_knee"
            segment_progress = (
                effective_acceleration / V24_FINAL_QUALITY_KNEE
            )
            target_compute_units = (
                float(total_steps)
                - (float(total_steps) - knee_cost)
                * segment_progress ** self.curve.budget_exponent
            )
            actual, action_state, optimizer = _solve_nested_budget(
                total_steps=total_steps,
                lower=lower,
                upper=upper,
                mix=mix,
                endpoint=endpoint,
                mandatory_actual=mandatory,
                target_compute_units=target_compute_units,
                forecast_risk_multiplier=forecast_risk_multiplier,
                curve=self.curve,
            )
        else:
            segment = "human_quality_knee_to_aggressive_endpoint"
            segment_progress = (
                (effective_acceleration - V24_FINAL_QUALITY_KNEE)
                / (100.0 - V24_FINAL_QUALITY_KNEE)
            )
            actual, knee_state, knee_optimizer = _solve_nested_budget(
                total_steps=total_steps,
                lower=lower,
                upper=upper,
                mix=mix,
                endpoint=endpoint,
                mandatory_actual=mandatory,
                target_compute_units=knee_cost,
                forecast_risk_multiplier=forecast_risk_multiplier,
                curve=self.curve,
            )
            action_state, optimizer = _solve_aggressive_attention(
                total_steps=total_steps,
                actual=actual,
                knee_state=knee_state,
                progress=segment_progress,
                curve=self.curve,
            )
            optimizer["quality_knee_optimizer"] = {
                "chain_digest": knee_optimizer["chain_digest"],
                "achieved_compute_units": knee_optimizer[
                    "achieved_compute_units"
                ],
            }
            # Forecast placement is the failure-sensitive technique.  Keep it
            # fixed through 80% of the aggressive segment, then admit a small
            # nested correction-removal chain only near the explicit 100 end.
            demotion_chain = _trajectory_demotion_chain(
                actual,
                total_steps=total_steps,
                mandatory=mandatory,
                curve=self.curve,
            )
            demotion_progress = max(
                0.0,
                min(1.0, (segment_progress - 0.80) / 0.20),
            )
            demotion_count = min(
                len(demotion_chain),
                int(math.floor(
                    len(demotion_chain) * demotion_progress + 1.0e-12
                )),
            )
            applied_demotions = demotion_chain[:demotion_count]
            actual_set_after = set(actual)
            demotion_audit: list[dict[str, object]] = []
            for step in applied_demotions:
                cell_cost = _NON_ATTENTION_COMPUTE + (
                    _ATTENTION_COMPUTE / 50.0
                    * sum(
                        _ATTENTION_COST_RATIO[
                            _RANK_TO_CANONICAL[action_state[(step, layer)][0]]
                        ]
                        for layer in range(50)
                    )
                )
                demotion_audit.append({
                    "step": step,
                    "saved_compute": cell_cost - _FORECAST_COMPUTE,
                })
                actual_set_after.remove(step)
                for layer in range(50):
                    action_state.pop((step, layer))
            actual = tuple(sorted(actual_set_after))
            post_demotion_cost = _state_compute_units(
                total_steps=total_steps,
                actual=actual_set_after,
                ranks={
                    cell: rank
                    for cell, (rank, _prefix) in action_state.items()
                },
            )
            optimizer["pre_trajectory_compute_units"] = optimizer[
                "achieved_compute_units"
            ]
            optimizer["achieved_compute_units"] = post_demotion_cost
            optimizer["trajectory_demotion"] = {
                "activation_fraction_of_aggressive_segment": 0.80,
                "progress": demotion_progress,
                "chain": list(demotion_chain),
                "applied": demotion_audit,
                "maximum_forecast_run": 4,
                "opening_actual_fraction_protected": 0.20,
                "terminal_actual_steps_protected": 2,
            }

        endpoint_set = set(endpoint)
        actual_set = set(actual)
        schedule: list[tuple[int, int, str]] = []
        for step in range(total_steps):
            if step not in actual_set:
                schedule.extend(
                    (step, layer, ROUND229_FORECAST_ANCHOR)
                    for layer in range(3)
                )
                continue
            for layer in range(50):
                rank, prefix = (
                    action_state[(step, layer)]
                    if step in endpoint_set
                    else (4, None)
                )
                schedule.append((
                    step,
                    layer,
                    _runtime_action(rank, prefix),
                ))
        physical = tuple(schedule)
        all_dense = len(actual) == total_steps and all(
            action == "dense" for _step, _layer, action in physical
        )
        if all_dense:
            physical = ()
        execution_digest = _execution_digest(
            total_steps=total_steps,
            actual=actual,
            schedule=physical,
        )
        compute_units = _compute_units(
            total_steps=total_steps,
            actual=actual,
            schedule=physical,
        )
        achieved = float(optimizer["achieved_compute_units"])
        if not math.isclose(
            compute_units,
            achieved,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ):
            raise V19PlanningError(
                "V24 final optimizer and physical schedule disagree"
            )
        actual_actions = Counter(
            action.rsplit(":", 1)[-1]
            for step, _layer, action in physical
            if step in actual_set
        )
        forecast_actions = Counter(
            action.rsplit(":", 1)[-1]
            for step, _layer, action in physical
            if step not in actual_set
        )
        direct_knee = (
            total_steps == 20
            and math.isclose(
                effective_acceleration,
                V24_FINAL_QUALITY_KNEE,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and not guards
            and lower.anchor_id == upper.anchor_id
            and not required
        )
        execution_hint = (
            "v22_medium_byte_exact_helpers"
            if direct_knee and lower.anchor_id == "medium_66k_v022_causal"
            else "v18_xlong_byte_exact_helpers"
            if direct_knee and lower.anchor_id == "xlong_218k_v012_v018_exact"
            else None
        )
        coupled = ["exact_runtime"]
        if len(actual) < total_steps:
            coupled.append("directional_forecast")
        if any(action != "dense" for action in actual_actions):
            coupled.append("block_sparse_attention")
        if execution_hint:
            coupled.append("byte_exact_execution_helpers")
        return V24DeploymentSelection(
            actual_step_indices=actual,
            attention_action_schedule=physical,
            summary={
                "schema_version": V24_FINAL_SCHEMA,
                "policy_id": V24_FINAL_POLICY_ID,
                "human_surface_digest": self.human_surface_digest,
                "generation_request_digest": workload.digest,
                "execution_digest": execution_digest,
                "accelerated": not all_dense,
                "reason": (
                    "direct_human_quality_knee"
                    if direct_knee
                    else "continuous_human_knee_cost_risk_optimization"
                ),
                "acceleration": acceleration,
                "effective_acceleration": effective_acceleration,
                "quality_knee_acceleration": V24_FINAL_QUALITY_KNEE,
                "curve_segment": segment,
                "segment_progress": segment_progress,
                "requested_progress": requested_progress,
                "trajectory_progress": trajectory_progress,
                "attention_progress": attention_progress,
                "curve_profile": self.curve.to_dict(),
                "prompt_semantics_used": False,
                "public_controls": ["sampling_steps", "acceleration"],
                "video_tokens": video_tokens,
                "packed_tokens": workload.packed_tokens,
                "release_candidate": {
                    "candidate_id": self.candidate.candidate_id,
                    "human_status": self.candidate.human_status,
                    "role": self.candidate.role,
                },
                "calibration_surface": {
                    "lower_anchor_id": lower.anchor_id,
                    "upper_anchor_id": upper.anchor_id,
                    "mix": mix,
                    "lower_anchor_digest": lower.digest,
                    "upper_anchor_digest": upper.digest,
                    "source_execution_digests": list(dict.fromkeys((
                        lower.source_execution_digest,
                        upper.source_execution_digest,
                    ))),
                    "direct_human_quality_knee": direct_knee,
                    "human_evidence_max_packed_tokens": (
                        V24_HUMAN_EVIDENCE_MAX_PACKED_TOKENS
                    ),
                    "shape_extrapolated_beyond_human_evidence": (
                        evidence_extrapolated
                    ),
                },
                "safety_guards": list(guards),
                "required_actual_step_indices": list(required),
                "mandatory_actual_step_indices": list(mandatory),
                "knee_actual_step_indices": list(endpoint),
                "actual_step_indices": list(actual),
                "forecast_steps": total_steps - len(actual),
                "maximum_forecast_run": len(
                    _longest_forecast_run(actual_set, total_steps)
                ),
                "estimated_compute_units": compute_units,
                "estimated_compute_ratio": compute_units / total_steps,
                "optimizer": optimizer,
                "quality_knee_chain_digest": chain_digest,
                "quality_knee_upgrade_count": len(operations),
                "execution_profile_hint": execution_hint,
                "runtime_feedback": {
                    "policy_id": (
                        "v24_request_local_forecast_debt_v1"
                        if len(actual) < total_steps
                        else None
                    ),
                    "mode": (
                        "observe_only"
                        if len(actual) < total_steps
                        else "disabled_dense_trajectory"
                    ),
                    "adds_teacher_evaluations": False,
                    "max_runtime_promotions": 0,
                },
                "technique_mix": {
                    "actual_dit_evaluations": len(actual),
                    "forecast_evaluations": total_steps - len(actual),
                    "actual_attention_cells": dict(
                        sorted(actual_actions.items())
                    ),
                    "forecast_anchor_attention_cells": dict(
                        sorted(forecast_actions.items())
                    ),
                    "coupled_techniques": coupled,
                },
            },
        )


class V24FinalParetoRuntimeSelector(_V24ParetoSurfaceCompiler):
    """Compile the single immutable C02 production Pareto surface."""

    def __init__(
        self,
        *,
        curve: V24CurveProfile = V24_FINAL_CURVE_PROFILE,
    ) -> None:
        super().__init__(
            candidate=V24_FINAL_RELEASE_CANDIDATES[
                V24_FINAL_DEFAULT_CANDIDATE
            ],
            curve=curve,
        )


class V24ResearchParetoRuntimeSelector(_V24ParetoSurfaceCompiler):
    """Replay an exact historical calibration outside the product service."""

    def __init__(
        self,
        *,
        candidate_id: str,
        curve: V24CurveProfile = V24_FINAL_HISTORICAL_CURVE_PROFILE,
    ) -> None:
        super().__init__(
            candidate=_resolve_v24_research_candidate(candidate_id),
            curve=curve,
        )


__all__ = [
    "V24_FINAL_CURVE_PROFILE",
    "V24_FINAL_HISTORICAL_CURVE_PROFILE",
    "V24_FINAL_DEFAULT_CANDIDATE",
    "V24_FINAL_HISTORICAL_V009",
    "V24_FINAL_POLICY_ID",
    "V24_FINAL_QUALITY_KNEE",
    "V24_FINAL_RELEASE_CANDIDATES",
    "V24_FINAL_SCHEMA",
    "V24FinalParetoRuntimeSelector",
    "V24FinalReleaseCandidate",
    "V24ResearchParetoRuntimeSelector",
]
