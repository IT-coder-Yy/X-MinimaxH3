"""Shape-aware exact finite DP for joint H3 acceleration planning.

This module is deliberately a control-plane optimizer, not a claim that a
local tensor metric is equivalent to Human video judgement.  It solves the
finite problem it is given exactly:

* choose an actual or forecast transition at each sigma position;
* for every actual transition choose one action for each of seven H3 layer
  bands;
* obey opening/terminal anchors, the 60% actual-evaluation floor and the
  maximum two-forecast run;
* stay below a conservatively quantised, shape-specific RTX 4090 budget; and
* minimise an additive Human-risk surrogate seeded by Dense-teacher and
  Human positive/negative evidence.

The two calibration endpoints are compact copies of the real Round215
teacher-side measurements.  Their source SHA256 values are retained in the
certificate so a release can be audited without depending on ``runtime/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Mapping

from .sparse_budget import H3LayerBand, HumanRiskVector


GLOBAL_JOINT_CERTIFICATE_SCHEMA = "h3_global_joint_dp_certificate_v3"
ONLINE_REBATE_CERTIFICATE_SCHEMA = "h3_online_rebate_certificate_v1"

# A calibrated cost is evidence only for the exact kernel/policy family which
# produced it.  Keep this identity in the certificate instead of silently
# treating every implementation of the same nominal TopK fraction as equal.
FIXED_TOPK_ACTION_IMPLEMENTATION = "fixed_topk_v1"
ROUND215_ACTION_IMPLEMENTATION = "interaction_hybrid_round215_v1"
ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION = "round188_head_rail_mtcr_v1"
ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION = "round228_fast_head_rail_mtcr_v1"
ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION = (
    "round229_forecast_aware_head_rail_mtcr_v1"
)
BASE_STRUCTURAL_CONSTRAINT = "round86_structural_floors_v1"
ROUND216_CAUSAL_ISLAND_CONSTRAINT = "round216_human_causal_island_v1"
ROUND224_ADAPTIVE_LATENCY_CONSTRAINT = (
    "round224_adaptive_latency_frontier_v1"
)
ROUND225_TRAJECTORY_CORRECTION_CONSTRAINT = (
    "round225_trajectory_correction_frontier_v1"
)
ROUND226_OPENING_ANCHORED_MTCR_CONSTRAINT = (
    "round226_opening_anchored_mtcr_frontier_v1"
)
ROUND227_FRONTIER_DOMINANCE_CONSTRAINT = (
    "round227_frontier_dominance_frontier_v1"
)
BAND_RISK_MODEL = "round215_band_upper_risk_v1"
ROUND215_LAYER_RISK_MODEL = "round215_per_layer_dense_disagreement_v1"
ROUND218_PHASE_LAYER_RISK_MODEL = "round218_phase_layer_dense_disagreement_v1"

ACTION_NAMES = (
    "sparse_topk_0.0625",
    "sparse_topk_0.1",
    "sparse_topk_0.25",
    "sparse_topk_0.5",
    "dense",
)
ACTION_RANK = {name: index for index, name in enumerate(ACTION_NAMES)}

LAYER_BANDS: tuple[tuple[H3LayerBand, int, int], ...] = (
    (H3LayerBand.EARLY, 0, 15),
    (H3LayerBand.MIDDLE, 15, 30),
    (H3LayerBand.CAUSAL, 30, 40),
    (H3LayerBand.CAUSAL_DETAIL, 40, 44),
    (H3LayerBand.BRIDGE, 44, 45),
    (H3LayerBand.CAUSAL_TERMINAL, 45, 46),
    (H3LayerBand.TAIL, 46, 50),
)

_CAUSAL_BANDS = frozenset(
    (H3LayerBand.CAUSAL, H3LayerBand.CAUSAL_DETAIL, H3LayerBand.CAUSAL_TERMINAL)
)
_BAND_RISK_MULTIPLIER: Mapping[H3LayerBand, float] = {
    H3LayerBand.EARLY: 1.00,
    H3LayerBand.MIDDLE: 1.10,
    H3LayerBand.CAUSAL: 2.20,
    H3LayerBand.CAUSAL_DETAIL: 2.50,
    H3LayerBand.BRIDGE: 1.20,
    H3LayerBand.CAUSAL_TERMINAL: 2.50,
    H3LayerBand.TAIL: 1.30,
}
_PHASE_RISK_MULTIPLIER = {"opening": 1.60, "ordinary": 1.00, "terminal": 1.40}


@dataclass(frozen=True, slots=True)
class _MeasuredAction:
    cost_ms: float
    dense_error_upper: float


@dataclass(frozen=True, slots=True)
class _CalibrationEndpoint:
    name: str
    packed_tokens: int
    source_sha256: str
    forecast_ms: float
    actions: Mapping[H3LayerBand, Mapping[str, _MeasuredAction]]


def _endpoint(
    name: str,
    packed_tokens: int,
    source_sha256: str,
    forecast_ms: float,
    rows: Mapping[str, tuple[tuple[float, float], ...]],
) -> _CalibrationEndpoint:
    actions: dict[H3LayerBand, dict[str, _MeasuredAction]] = {}
    for band, _, _ in LAYER_BANDS:
        values = rows[band.value]
        actions[band] = {
            action: _MeasuredAction(*measurement)
            for action, measurement in zip(ACTION_NAMES, values)
        }
    return _CalibrationEndpoint(
        name=name,
        packed_tokens=packed_tokens,
        source_sha256=source_sha256,
        forecast_ms=forecast_ms,
        actions=actions,
    )


# Each tuple is (complete-band wall ms, observed Dense relative-RMS upper).
_SHORT = _endpoint(
    "round215_34871",
    34_871,
    "636c75f09825dcc146590cedf85a0bbf92a80281e8973dad690dfd113f5a3a29",
    540.0,
    {
        "layers_00_14": ((416.024, .214719), (441.145, .171548), (593.002, .091554), (853.904, .042555), (1100.145, 0.0)),
        "layers_15_29": ((414.566, .244826), (445.037, .203217), (603.456, .114961), (878.005, .052918), (1093.875, 0.0)),
        "layers_30_39": ((269.684, .282185), (284.776, .242696), (388.969, .152823), (579.548, .077099), (734.956, 0.0)),
        "layers_40_43": ((105.126, .359542), (111.952, .308489), (151.793, .193891), (225.300, .097626), (291.332, 0.0)),
        "layer_44": ((26.083, .254915), (27.603, .209623), (37.258, .115921), (55.059, .054766), (73.473, 0.0)),
        "layer_45": ((26.153, .247639), (27.747, .200642), (37.546, .121001), (54.943, .075074), (72.742, 0.0)),
        "layers_46_49": ((105.328, .183848), (112.208, .148081), (150.935, .102218), (220.545, .070245), (291.576, 0.0)),
    },
)

_LONG = _endpoint(
    "round215_100163",
    100_163,
    "1004807c8494994b41ce1c4d38506091ee2877003a3b28630dcf809e2551b96d",
    2_650.0,
    {
        "layers_00_14": ((1957.801, .169071), (2267.419, .133687), (3650.674, .069247), (5964.293, .031715), (8942.715, 0.0)),
        "layers_15_29": ((1939.215, .211137), (2253.314, .171550), (3676.161, .093897), (6084.513, .042784), (8940.779, 0.0)),
        "layers_30_39": ((1233.207, .261093), (1441.121, .219905), (2374.212, .133317), (4006.578, .065807), (5977.687, 0.0)),
        "layers_40_43": ((487.938, .323994), (563.492, .272101), (934.194, .166252), (1568.398, .083032), (2385.080, 0.0)),
        "layer_44": ((120.537, .215586), (142.332, .172114), (228.825, .093263), (383.786, .044235), (595.744, 0.0)),
        "layer_45": ((119.746, .211506), (141.490, .168610), (230.474, .108841), (384.403, .078251), (592.700, 0.0)),
        "layers_46_49": ((494.576, .162054), (568.587, .138427), (933.032, .103875), (1545.736, .076050), (2380.143, 0.0)),
    },
)


@dataclass(frozen=True, slots=True)
class JointWorkloadContext:
    """Internal request context; none of these fields is a creator control."""

    packed_tokens: int = _LONG.packed_tokens
    condition_count: int = 0
    service_family: str = "first_last"
    model_variant: str = "base"

    def __post_init__(self) -> None:
        if self.packed_tokens <= 0:
            raise ValueError("packed_tokens must be positive")
        if self.condition_count < 0:
            raise ValueError("condition_count cannot be negative")
        if self.service_family not in ("first_last", "reference"):
            raise ValueError("unsupported service family")
        if self.model_variant not in ("base", "lora"):
            raise ValueError("unsupported model variant")

    @property
    def cache_key(self) -> tuple[int, int, str, str]:
        # Cost interpolation is smooth; 64-token buckets avoid cache churn
        # from harmless prompt-token differences.
        return (
            int(round(self.packed_tokens / 64.0) * 64),
            self.condition_count,
            self.service_family,
            self.model_variant,
        )


@dataclass(frozen=True, slots=True)
class TrajectoryRiskPrior:
    """Evidence support for actual-step placement, never a public control."""

    prior_id: str
    supported_actual_steps: tuple[int, ...]
    unsupported_forecast_risk: float

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.supported_actual_steps))) != self.supported_actual_steps:
            raise ValueError("trajectory prior steps must be sorted and unique")
        if self.unsupported_forecast_risk < 0.0:
            raise ValueError("trajectory prior risk cannot be negative")


ROUND143_216_TRAJECTORY_PRIOR = TrajectoryRiskPrior(
    prior_id="round143_round216_human_positive_20step_12a8f_v1",
    supported_actual_steps=(0, 1, 2, 3, 4, 6, 8, 11, 14, 17, 18, 19),
    # This is epistemic debt, not a fitted probability.  One unit is similar
    # to one ordinary forecast transition in the original surrogate and is
    # enough to prefer a Human-supported path when compute/risk otherwise tie.
    unsupported_forecast_risk=1.0,
)


@dataclass(frozen=True, slots=True)
class InterpolatedAction:
    name: str
    cost_ms: float
    risk: float
    components: HumanRiskVector


@dataclass(frozen=True, slots=True)
class _BandBundle:
    conservative_units: int
    actual_cost_ms: float
    risk: float
    components: HumanRiskVector
    actions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GlobalStepChoice:
    step_index: int
    actual: bool
    actions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GlobalJointCertificate:
    schema_version: str
    solver: str
    objective: str
    formal_scope: str
    cost_quantum_ms: float
    budget_units: int
    minimum_actual_count: int
    state_count: int
    conservative_cost_units: int
    optimum_risk: float
    choice_sha256: str
    model_sha256: str
    action_implementation_id: str
    quality_constraint_id: str
    risk_model_id: str


@dataclass(frozen=True, slots=True)
class GlobalJointSolution:
    steps: tuple[GlobalStepChoice, ...]
    predicted_cost_ms: float
    conservative_cost_ms: float
    target_cost_ms: float
    dense_cost_ms: float
    fastest_cost_ms: float
    recovery_reserve_ms: float
    forecast_risk: float
    attention_risk: float
    components: HumanRiskVector
    calibration_mix: float
    extrapolated: bool
    certificate: GlobalJointCertificate

    @property
    def actual_steps(self) -> tuple[int, ...]:
        return tuple(step.step_index for step in self.steps if step.actual)

    @property
    def forecast_steps(self) -> tuple[int, ...]:
        return tuple(step.step_index for step in self.steps if not step.actual)


@dataclass(frozen=True, slots=True)
class GlobalJointVerification:
    valid: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OnlineRebateChoice:
    """One conditional Dense upgrade bought from an unused online reserve."""

    step_index: int
    layer: int
    from_action: str
    surrogate_risk_reduction: float


@dataclass(frozen=True, slots=True)
class OnlineRebateCertificate:
    """Replayable optimum certificate for the no-trigger contingency branch."""

    schema_version: str
    solver: str
    objective: str
    maximum_choices: int
    candidate_count: int
    selected_count: int
    selected_risk_reduction: float
    choice_sha256: str
    model_sha256: str


@dataclass(frozen=True, slots=True)
class OnlineRebateSolution:
    choices: tuple[OnlineRebateChoice, ...]
    certificate: OnlineRebateCertificate


@dataclass(frozen=True, slots=True)
class _InterpolatedProfile:
    packed_tokens: int
    mix: float
    extrapolated: bool
    quantum_ms: float
    dense_attention_ms: float
    dense_step_ms: float
    non_attention_ms: float
    forecast_ms: float
    actions: Mapping[H3LayerBand, Mapping[str, _MeasuredAction]]
    layer_errors: Mapping[str, tuple[float, ...]]
    temporal_progress_anchors: tuple[float, ...]
    temporal_layer_errors: tuple[Mapping[str, tuple[float, ...]], ...]
    temporal_risk_scales: tuple[float, ...]
    model_sha256: str


def _lerp(left: float, right: float, mix: float) -> float:
    return left + (right - left) * mix


def _load_layer_risk_evidence() -> dict[str, object]:
    source = Path(__file__).with_name("evidence") / "round215_layer_risk_v1.json"
    document = json.loads(source.read_text(encoding="utf-8"))
    if document.get("schema_version") != "h3_round215_layer_risk_v1":
        raise RuntimeError("unexpected Round215 layer-risk evidence schema")
    if document.get("action_implementation_id") != ROUND215_ACTION_IMPLEMENTATION:
        raise RuntimeError("Round215 layer-risk implementation identity mismatch")
    endpoints = document.get("endpoints")
    if not isinstance(endpoints, dict):
        raise RuntimeError("Round215 layer-risk endpoints are missing")
    for label, endpoint, expected in (
        ("short", endpoints.get("short"), _SHORT),
        ("long", endpoints.get("long"), _LONG),
    ):
        if not isinstance(endpoint, dict):
            raise RuntimeError(f"Round215 {label} layer-risk endpoint is missing")
        if endpoint.get("packed_tokens") != expected.packed_tokens:
            raise RuntimeError(f"Round215 {label} packed-token mismatch")
        if endpoint.get("source_sha256") != expected.source_sha256:
            raise RuntimeError(f"Round215 {label} source digest mismatch")
        actions = endpoint.get("actions")
        if not isinstance(actions, dict) or set(actions) != set(ACTION_NAMES[:-1]):
            raise RuntimeError(f"Round215 {label} layer-risk actions are incomplete")
        if any(not isinstance(values, list) or len(values) != 50 for values in actions.values()):
            raise RuntimeError(f"Round215 {label} layer-risk rows must cover 50 layers")
    return document


_LAYER_RISK_EVIDENCE = _load_layer_risk_evidence()


def _load_phase_layer_risk_evidence() -> dict[str, object]:
    source = Path(__file__).with_name("evidence") / "round218_phase_layer_risk_v1.json"
    document = json.loads(source.read_text(encoding="utf-8"))
    if document.get("schema_version") != "h3_phase_layer_risk_v1":
        raise RuntimeError("unexpected phase-layer risk evidence schema")
    if document.get("action_implementation_id") != ROUND215_ACTION_IMPLEMENTATION:
        raise RuntimeError("phase-layer risk implementation identity mismatch")
    if document.get("actions") != list(ACTION_NAMES[:-1]):
        raise RuntimeError("phase-layer action set mismatch")
    endpoints = document.get("endpoints")
    if not isinstance(endpoints, dict):
        raise RuntimeError("phase-layer risk endpoints are missing")
    expected_progress: tuple[float, ...] | None = None
    for label, expected in (("short", _SHORT), ("long", _LONG)):
        endpoint = endpoints.get(label)
        if not isinstance(endpoint, dict):
            raise RuntimeError(f"phase-layer {label} endpoint is missing")
        if endpoint.get("packed_tokens") != expected.packed_tokens:
            raise RuntimeError(f"phase-layer {label} packed-token mismatch")
        sources = endpoint.get("sources")
        if not isinstance(sources, list) or not sources:
            raise RuntimeError(f"phase-layer {label} provenance is missing")
        if any(
            not isinstance(row, dict)
            or not isinstance(row.get("sha256"), str)
            or len(row["sha256"]) != 64
            for row in sources
        ):
            raise RuntimeError(f"phase-layer {label} provenance is malformed")
        anchors = endpoint.get("anchors")
        if not isinstance(anchors, list) or len(anchors) != 5:
            raise RuntimeError(f"phase-layer {label} must contain five anchors")
        progress = tuple(float(row["trajectory_progress"]) for row in anchors)
        if tuple(sorted(set(progress))) != progress:
            raise RuntimeError(f"phase-layer {label} progress must be sorted and unique")
        if expected_progress is None:
            expected_progress = progress
        elif progress != expected_progress:
            raise RuntimeError("phase-layer endpoint progress anchors differ")
        for anchor in anchors:
            actions = anchor.get("actions")
            if not isinstance(actions, dict) or set(actions) != set(ACTION_NAMES[:-1]):
                raise RuntimeError(f"phase-layer {label} anchor actions are incomplete")
            if any(
                not isinstance(values, list) or len(values) != 50
                for values in actions.values()
            ):
                raise RuntimeError(f"phase-layer {label} anchors must cover 50 layers")
    return document


_PHASE_LAYER_RISK_EVIDENCE = _load_phase_layer_risk_evidence()


def _profile(
    context: JointWorkloadContext,
    risk_model_id: str = BAND_RISK_MODEL,
) -> _InterpolatedProfile:
    raw_mix = (context.packed_tokens - _SHORT.packed_tokens) / (
        _LONG.packed_tokens - _SHORT.packed_tokens
    )
    mix = max(0.0, min(1.0, raw_mix))
    extrapolated = not 0.0 <= raw_mix <= 1.0
    actions: dict[H3LayerBand, dict[str, _MeasuredAction]] = {}
    for band, _, _ in LAYER_BANDS:
        actions[band] = {}
        for name in ACTION_NAMES:
            left = _SHORT.actions[band][name]
            right = _LONG.actions[band][name]
            actions[band][name] = _MeasuredAction(
                _lerp(left.cost_ms, right.cost_ms, mix),
                _lerp(left.dense_error_upper, right.dense_error_upper, mix),
            )
    # Outside the measured interval cost grows conservatively.  It is still
    # marked OOD, so the runtime can fail closed or reserve extra recovery.
    if context.packed_tokens < _SHORT.packed_tokens:
        scale = max(0.15, context.packed_tokens / _SHORT.packed_tokens) ** 1.55
    elif context.packed_tokens > _LONG.packed_tokens:
        scale = (context.packed_tokens / _LONG.packed_tokens) ** 1.80
    else:
        scale = 1.0
    if not math.isclose(scale, 1.0):
        actions = {
            band: {
                name: _MeasuredAction(value.cost_ms * scale, value.dense_error_upper)
                for name, value in by_name.items()
            }
            for band, by_name in actions.items()
        }
    evidence_endpoints = _LAYER_RISK_EVIDENCE["endpoints"]
    short_errors = evidence_endpoints["short"]["actions"]
    long_errors = evidence_endpoints["long"]["actions"]
    layer_errors = {
        name: tuple(
            _lerp(float(left), float(right), mix)
            for left, right in zip(short_errors[name], long_errors[name])
        )
        for name in ACTION_NAMES[:-1]
    }
    layer_errors["dense"] = (0.0,) * 50
    phase_endpoints = _PHASE_LAYER_RISK_EVIDENCE["endpoints"]
    short_anchors = phase_endpoints["short"]["anchors"]
    long_anchors = phase_endpoints["long"]["anchors"]
    temporal_progress_anchors = tuple(
        float(anchor["trajectory_progress"]) for anchor in short_anchors
    )
    temporal_layer_errors: list[dict[str, tuple[float, ...]]] = []
    for short_anchor, long_anchor in zip(short_anchors, long_anchors):
        temporal_layer_errors.append({
            name: tuple(
                _lerp(float(left), float(right), mix)
                for left, right in zip(
                    short_anchor["actions"][name],
                    long_anchor["actions"][name],
                )
            )
            for name in ACTION_NAMES[:-1]
        })
        temporal_layer_errors[-1]["dense"] = (0.0,) * 50
    baseline_values = tuple(
        value
        for name in ACTION_NAMES[:-1]
        for value in layer_errors[name]
    )
    baseline_square_sum = sum(value * value for value in baseline_values)
    temporal_risk_scales = tuple(
        sum(
            baseline * measured
            for baseline, measured in zip(
                baseline_values,
                (
                    value
                    for name in ACTION_NAMES[:-1]
                    for value in anchor[name]
                ),
            )
        )
        / baseline_square_sum
        for anchor in temporal_layer_errors
    )
    dense_attention_ms = sum(actions[band]["dense"].cost_ms for band, _, _ in LAYER_BANDS)
    dense_step_ms = dense_attention_ms / 0.714
    non_attention_ms = dense_step_ms - dense_attention_ms
    forecast_ms = _lerp(_SHORT.forecast_ms, _LONG.forecast_ms, mix) * scale
    # Global trajectory DP has many more states than the conditional v2
    # Attention knapsack.  The quantum is at most 2.5% of one Dense step at
    # either endpoint; every transition rounds upward, so coarsening can leave
    # compute unused but can never create a runtime budget overrun.
    quantum_ms = 200.0 if mix < 0.35 else (500.0 if mix < 0.70 else 1_000.0)
    model_document = {
        "short_sha256": _SHORT.source_sha256,
        "long_sha256": _LONG.source_sha256,
        "packed_tokens": context.packed_tokens,
        "mix": mix,
        "scale": scale,
        "quantum_ms": quantum_ms,
        "condition_count": context.condition_count,
        "service_family": context.service_family,
        "model_variant": context.model_variant,
        "layer_risk_evidence_sha256": hashlib.sha256(
            json.dumps(
                _LAYER_RISK_EVIDENCE,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }
    if risk_model_id == ROUND218_PHASE_LAYER_RISK_MODEL:
        model_document["phase_layer_risk_evidence_sha256"] = hashlib.sha256(
            json.dumps(
                _PHASE_LAYER_RISK_EVIDENCE,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    model_sha256 = hashlib.sha256(
        json.dumps(model_document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return _InterpolatedProfile(
        packed_tokens=context.packed_tokens,
        mix=mix,
        extrapolated=extrapolated,
        quantum_ms=quantum_ms,
        dense_attention_ms=dense_attention_ms,
        dense_step_ms=dense_step_ms,
        non_attention_ms=non_attention_ms,
        forecast_ms=forecast_ms,
        actions=actions,
        layer_errors=layer_errors,
        temporal_progress_anchors=temporal_progress_anchors,
        temporal_layer_errors=tuple(temporal_layer_errors),
        temporal_risk_scales=temporal_risk_scales,
        model_sha256=model_sha256,
    )


def _ceil_units(cost_ms: float, quantum_ms: float) -> int:
    return int(math.ceil(cost_ms / quantum_ms - 1.0e-12))


def _phase(step: int, total_steps: int) -> str:
    if step == 0:
        return "opening"
    if step >= total_steps - 3:
        return "terminal"
    return "ordinary"


def _temporal_anchor_index(
    profile: _InterpolatedProfile,
    step: int,
    total_steps: int,
) -> int:
    """Map an arbitrary trajectory to one of five measured risk regimes.

    The raw evidence contains five normalized trajectory anchors.  Using the
    nearest anchor keeps the finite optimizer exact for a declared piecewise
    constant risk model and bounds cold planning to five physical-layer
    frontiers instead of rebuilding one frontier for every possible step.
    """

    progress = step / max(1, total_steps - 1)
    return min(
        range(len(profile.temporal_progress_anchors)),
        key=lambda index: (
            abs(profile.temporal_progress_anchors[index] - progress),
            index,
        ),
    )


def _temporal_risk_scale(
    profile: _InterpolatedProfile,
    step: int,
    total_steps: int,
) -> float:
    return profile.temporal_risk_scales[
        _temporal_anchor_index(profile, step, total_steps)
    ]


def _scale_components(
    components: HumanRiskVector,
    scale: float,
) -> HumanRiskVector:
    return HumanRiskVector(
        motion=components.motion * scale,
        clarity=components.clarity * scale,
        identity=components.identity * scale,
        audio=components.audio * scale,
    )


def _minimum_rank(
    band: H3LayerBand,
    phase: str,
    recovery: bool,
    quality_constraint_id: str,
) -> int:
    if quality_constraint_id == ROUND216_CAUSAL_ISLAND_CONSTRAINT:
        # Human accepted Round143/216 and rejected the relaxed Round217.  The
        # common non-compensating invariant is an exact opening plus an exact
        # interaction/causal island at layers 30--43 and 45.  Remaining bands
        # stay optimizable, so this is a constraint rather than a frozen plan.
        if phase == "opening" or band in _CAUSAL_BANDS:
            return ACTION_RANK["dense"]
        return (
            ACTION_RANK["sparse_topk_0.25"]
            if phase == "terminal"
            else ACTION_RANK["sparse_topk_0.0625"]
        )
    if quality_constraint_id == ROUND224_ADAPTIVE_LATENCY_CONSTRAINT:
        # V13 deliberately does not freeze a Human-reviewed layer schedule.
        # These are only catastrophic-failure rails: modestly protect the
        # opening/terminal state and raise causal bands after a long forecast
        # run.  The phase/layer risk model remains responsible for allocating
        # every additional unit of compute under the request budget.
        floor = ACTION_RANK["sparse_topk_0.0625"]
        if phase == "opening":
            floor = ACTION_RANK[
                "sparse_topk_0.25" if band in _CAUSAL_BANDS else "sparse_topk_0.1"
            ]
        elif phase == "terminal":
            floor = ACTION_RANK["sparse_topk_0.1"]
        if recovery and band in _CAUSAL_BANDS:
            floor = max(floor, ACTION_RANK["sparse_topk_0.1"])
        return floor
    if quality_constraint_id == ROUND225_TRAJECTORY_CORRECTION_CONSTRAINT:
        # Round224 proved that a locally low-risk sparse action cannot repair
        # a latent trajectory after a long forecast run: 6 actual / 14
        # forecast evaluations plus an all-sparse correction path failed the
        # Human continuous-play review.  V14 therefore couples trajectory and
        # layer fidelity.  A real evaluation following two forecasts must
        # execute the Human-supported causal island exactly; the remaining
        # bands are still selected by the measured risk/cost frontier.
        floor = ACTION_RANK["sparse_topk_0.0625"]
        if phase == "opening":
            floor = ACTION_RANK[
                "sparse_topk_0.5" if band in _CAUSAL_BANDS else "sparse_topk_0.25"
            ]
        elif phase == "terminal":
            floor = ACTION_RANK["sparse_topk_0.25"]
        if recovery and band in _CAUSAL_BANDS:
            floor = ACTION_RANK["dense"]
        return floor
    if quality_constraint_id == ROUND226_OPENING_ANCHORED_MTCR_CONSTRAINT:
        # Human review of V14 found that the door begins moving before hand
        # contact even though post-debt causal layers were exact.  The missing
        # invariant is therefore moved earlier in the trajectory: establish
        # composition and interaction for the first quarter with consecutive
        # real DiT evaluations.  Later debt is corrected by the already
        # validated temporal-correspondence sparse operator instead of paying
        # for a complete 15-layer Dense causal island on every recovery step.
        floor = ACTION_RANK[
            "sparse_topk_0.1" if band in _CAUSAL_BANDS
            else "sparse_topk_0.0625"
        ]
        # The first five steps are all genuine DiT evaluations, so charging a
        # second, high-TopK opening premium repeats the expensive V14 mistake.
        # MTCR already preserves aligned/remote video keys inside these sparse
        # actions.  Spend extra quota only where forecast debt is actually
        # repaid and at the terminal state.
        if phase == "terminal":
            floor = max(
                floor,
                ACTION_RANK[
                    "sparse_topk_0.25"
                    if band in _CAUSAL_BANDS
                    else "sparse_topk_0.1"
                ],
            )
        if recovery and band in _CAUSAL_BANDS:
            floor = max(floor, ACTION_RANK["sparse_topk_0.25"])
        return floor
    if quality_constraint_id == ROUND227_FRONTIER_DOMINANCE_CONSTRAINT:
        # Round188 is the fastest Human-reviewed long-sequence comparator that
        # remains usable.  Its important primitive is not a uniform Top-K
        # scalar: non-sensitive layers use the 6.25% floor while the measured
        # high-risk heads in layers 30--43/45 receive the 8--10% head rail.
        # The action implementation carries that per-head meaning; this floor
        # only states which cells may not fall below the reviewed rail.
        return ACTION_RANK[
            "sparse_topk_0.1"
            if band in _CAUSAL_BANDS
            else "sparse_topk_0.0625"
        ]
    floor = ACTION_RANK["sparse_topk_0.1"] if band in _CAUSAL_BANDS else 0
    if phase == "opening":
        floor = max(floor, ACTION_RANK["sparse_topk_0.25"])
        if band in _CAUSAL_BANDS:
            floor = max(floor, ACTION_RANK["sparse_topk_0.5"])
    if phase == "terminal":
        floor = max(floor, ACTION_RANK["sparse_topk_0.25"])
    if recovery and band in _CAUSAL_BANDS:
        floor = max(floor, ACTION_RANK["sparse_topk_0.25"])
    return floor


def _action(
    profile: _InterpolatedProfile,
    band: H3LayerBand,
    layer_count: int,
    name: str,
    phase: str,
) -> InterpolatedAction:
    measured = profile.actions[band][name]
    base = measured.dense_error_upper * layer_count
    risk = base * _BAND_RISK_MULTIPLIER[band] * _PHASE_RISK_MULTIPLIER[phase]
    causal = band in _CAUSAL_BANDS
    return InterpolatedAction(
        name=name,
        cost_ms=measured.cost_ms,
        risk=risk,
        components=HumanRiskVector(
            motion=risk if causal else risk * .45,
            clarity=base,
            identity=risk * (.60 if causal else .25),
            audio=risk * (.45 if phase == "terminal" else .20),
        ),
    )


def _band_for_layer(layer: int) -> tuple[H3LayerBand, int]:
    for band, start, stop in LAYER_BANDS:
        if start <= layer < stop:
            return band, stop - start
    raise ValueError(f"H3 layer lies outside [0, 50): {layer}")


def _layer_action(
    profile: _InterpolatedProfile,
    layer: int,
    name: str,
    phase: str,
    *,
    risk_model_id: str = ROUND215_LAYER_RISK_MODEL,
    temporal_anchor_index: int | None = None,
) -> InterpolatedAction:
    band, layer_count = _band_for_layer(layer)
    measured = profile.actions[band][name]
    # V8 factorises the five measured trajectory tables into the stable
    # per-layer step-3 ranking here and a request-step scale in the trajectory
    # DP.  This retains measured temporal magnitude while sharing one exact
    # physical-layer frontier across the trajectory.
    base = profile.layer_errors[name][layer]
    risk = base * _BAND_RISK_MULTIPLIER[band] * _PHASE_RISK_MULTIPLIER[phase]
    causal = band in _CAUSAL_BANDS
    return InterpolatedAction(
        name=name,
        # Round215 v1 has robust median timing at band granularity and exact
        # Dense disagreement at layer granularity.  Split the measured band
        # cost uniformly rather than pretending a single noisy layer timing
        # is a stable wall-clock model.
        cost_ms=measured.cost_ms / layer_count,
        risk=risk,
        components=HumanRiskVector(
            motion=risk if causal else risk * .45,
            clarity=base,
            identity=risk * (.60 if causal else .25),
            audio=risk * (.45 if phase == "terminal" else .20),
        ),
    )


def _prune_real_cost_bundles(states: list[_BandBundle]) -> list[_BandBundle]:
    """Exact Pareto pruning before any cost quantisation.

    A lower-real-cost state may only be discarded when it is no worse on the
    lexicographic objective.  This preserves candidates that the former
    incremental coarse buckets could erase before later layers were added.
    """

    ordered = sorted(
        states,
        key=lambda state: (
            state.actual_cost_ms,
            state.risk,
            state.components.worst_component,
            state.actions,
        ),
    )
    result: list[_BandBundle] = []
    best_risk = math.inf
    best_worst = math.inf
    for state in ordered:
        if state.risk < best_risk - 1.0e-12 or (
            math.isclose(state.risk, best_risk, rel_tol=0.0, abs_tol=1.0e-12)
            and state.components.worst_component < best_worst - 1.0e-12
        ):
            result.append(state)
            best_risk = state.risk
            best_worst = state.components.worst_component
    return result


def _layer_bundle_frontier(
    profile: _InterpolatedProfile,
    phase: str,
    recovery: bool,
    quality_constraint_id: str,
    *,
    risk_model_id: str = ROUND215_LAYER_RISK_MODEL,
    temporal_anchor_index: int | None = None,
) -> tuple[_BandBundle, ...]:
    """Solve all 50 physical H3 layer actions before trajectory placement."""

    states = [_BandBundle(0, 0.0, 0.0, HumanRiskVector(), ())]
    for layer in range(50):
        band, _ = _band_for_layer(layer)
        floor = _minimum_rank(band, phase, recovery, quality_constraint_id)
        candidates = tuple(
            _layer_action(
                profile,
                layer,
                name,
                phase,
                risk_model_id=risk_model_id,
                temporal_anchor_index=temporal_anchor_index,
            )
            for name in ACTION_NAMES
            if ACTION_RANK[name] >= floor
        )
        expanded: list[_BandBundle] = []
        for state in states:
            for action in candidates:
                expanded.append(_BandBundle(
                    0,
                    state.actual_cost_ms + action.cost_ms,
                    state.risk + action.risk,
                    state.components + action.components,
                    state.actions + (action.name,),
                ))
        states = _prune_real_cost_bundles(expanded)

    by_units: dict[int, _BandBundle] = {}
    for state in states:
        units = _ceil_units(state.actual_cost_ms, profile.quantum_ms)
        proposed = _BandBundle(
            units,
            state.actual_cost_ms,
            state.risk,
            state.components,
            state.actions,
        )
        incumbent = by_units.get(units)
        if incumbent is None or (
            proposed.risk,
            proposed.components.worst_component,
            -proposed.actual_cost_ms,
            proposed.actions,
        ) < (
            incumbent.risk,
            incumbent.components.worst_component,
            -incumbent.actual_cost_ms,
            incumbent.actions,
        ):
            by_units[units] = proposed
    frontier: list[_BandBundle] = []
    best_risk = math.inf
    for units, state in sorted(by_units.items()):
        if state.risk < best_risk - 1.0e-12:
            frontier.append(state)
            best_risk = state.risk
    return tuple(frontier)


def _bundle_frontier(
    profile: _InterpolatedProfile,
    phase: str,
    recovery: bool,
    quality_constraint_id: str,
) -> tuple[_BandBundle, ...]:
    candidate_bands: list[tuple[InterpolatedAction, ...]] = []
    for band, start, stop in LAYER_BANDS:
        floor = _minimum_rank(
            band, phase, recovery, quality_constraint_id
        )
        candidate_bands.append(tuple(
            _action(profile, band, stop - start, name, phase)
            for name in ACTION_NAMES
            if ACTION_RANK[name] >= floor
        ))

    # There are at most 5**7=78,125 complete bundles.  Enumerating that finite
    # lattice before quantisation is cheap and, unlike incremental bucket
    # merging, cannot discard a lower-real-cost partial path that later lands
    # in a cheaper final bucket.  This is the exact action subproblem claimed
    # by the certificate.
    by_units: dict[int, _BandBundle] = {}
    for choices in itertools.product(*candidate_bands):
        actual_cost_ms = sum(choice.cost_ms for choice in choices)
        units = _ceil_units(actual_cost_ms, profile.quantum_ms)
        components = HumanRiskVector()
        for choice in choices:
            components += choice.components
        proposed = _BandBundle(
            units,
            actual_cost_ms,
            sum(choice.risk for choice in choices),
            components,
            tuple(choice.name for choice in choices),
        )
        incumbent = by_units.get(units)
        if incumbent is None or (
            proposed.risk,
            proposed.components.worst_component,
            -proposed.actual_cost_ms,
            proposed.actions,
        ) < (
            incumbent.risk,
            incumbent.components.worst_component,
            -incumbent.actual_cost_ms,
            incumbent.actions,
        ):
            by_units[units] = proposed

    frontier: dict[int, _BandBundle] = {}
    best_risk = math.inf
    for units, state in sorted(by_units.items()):
        if state.risk < best_risk - 1.0e-12:
            frontier[units] = state
            best_risk = state.risk
    return tuple(frontier[units] for units in sorted(frontier))


@dataclass(frozen=True, slots=True)
class _TrajectoryState:
    conservative_units: int
    actual_cost_ms: float
    risk: float
    forecast_risk: float
    attention_risk: float
    components: HumanRiskVector
    actual_count: int
    forecast_run: int
    path: tuple[GlobalStepChoice, ...]


def _required_actual(
    step: int,
    total_steps: int,
    quality_constraint_id: str,
) -> bool:
    if quality_constraint_id in (
        ROUND226_OPENING_ANCHORED_MTCR_CONSTRAINT,
        ROUND227_FRONTIER_DOMINANCE_CONSTRAINT,
    ):
        opening_count = max(3, int(math.ceil(total_steps * .25)))
        terminal_count = max(2, int(math.ceil(total_steps * .10)))
        return step < opening_count or step >= total_steps - terminal_count
    if quality_constraint_id in (
        ROUND224_ADAPTIVE_LATENCY_CONSTRAINT,
        ROUND225_TRAJECTORY_CORRECTION_CONSTRAINT,
    ):
        opening_count = max(2, int(math.ceil(total_steps * .10)))
        terminal_count = max(2, int(math.ceil(total_steps * .10)))
        return step < opening_count or step >= total_steps - terminal_count
    opening_count = max(3, int(math.ceil(total_steps * .25)))
    return step < opening_count or step >= total_steps - 3


def _minimum_actual_count(
    total_steps: int,
    allow_forecast: bool,
    quality_constraint_id: str,
) -> int:
    if quality_constraint_id == ROUND224_ADAPTIVE_LATENCY_CONSTRAINT:
        return (
            total_steps
            if not allow_forecast
            else min(total_steps, max(5, int(math.ceil(total_steps * .30))))
        )
    if quality_constraint_id == ROUND225_TRAJECTORY_CORRECTION_CONSTRAINT:
        return (
            total_steps
            if not allow_forecast
            else min(total_steps, max(6, int(math.ceil(total_steps * .40))))
        )
    if quality_constraint_id == ROUND226_OPENING_ANCHORED_MTCR_CONSTRAINT:
        return (
            total_steps
            if not allow_forecast
            else min(total_steps, max(7, int(math.ceil(total_steps * .45))))
        )
    if quality_constraint_id == ROUND227_FRONTIER_DOMINANCE_CONSTRAINT:
        return (
            total_steps
            if not allow_forecast
            else min(total_steps, max(8, int(math.ceil(total_steps * .50))))
        )
    return (
        total_steps
        if not allow_forecast
        else min(total_steps, max(5, int(math.ceil(total_steps * .60))))
    )


def _maximum_forecast_run(quality_constraint_id: str) -> int:
    if quality_constraint_id == ROUND224_ADAPTIVE_LATENCY_CONSTRAINT:
        return 5
    if quality_constraint_id == ROUND226_OPENING_ANCHORED_MTCR_CONSTRAINT:
        return 4
    if quality_constraint_id == ROUND227_FRONTIER_DOMINANCE_CONSTRAINT:
        return 3
    return 2


def _forecast_increment(step: int, total_steps: int, run: int) -> float:
    progress = step / max(1, total_steps - 1)
    phase = 1.35 if progress < .25 else (1.25 if progress > .78 else 1.0)
    return .90 * phase * (1.0 + .35 * max(0, run - 1))


def _prune(states: dict[tuple[int, int, int], _TrajectoryState]) -> dict[tuple[int, int, int], _TrajectoryState]:
    grouped: dict[tuple[int, int], list[_TrajectoryState]] = {}
    for state in states.values():
        grouped.setdefault((state.actual_count, state.forecast_run), []).append(state)
    result: dict[tuple[int, int, int], _TrajectoryState] = {}
    for (actual_count, run), rows in grouped.items():
        best_risk = math.inf
        for state in sorted(rows, key=lambda item: item.conservative_units):
            if state.risk < best_risk - 1.0e-12:
                result[(actual_count, run, state.conservative_units)] = state
                best_risk = state.risk
    return result


def _minimum_path_cost_units(
    total_steps: int,
    profile: _InterpolatedProfile,
    allow_forecast: bool,
    frontiers: Mapping[tuple[int, bool], tuple[_BandBundle, ...]],
    quality_constraint_id: str,
) -> int:
    minimum_actual = _minimum_actual_count(
        total_steps, allow_forecast, quality_constraint_id
    )
    maximum_forecast_run = _maximum_forecast_run(quality_constraint_id)
    states: dict[tuple[int, int], int] = {(0, 0): 0}
    forecast_units = _ceil_units(profile.forecast_ms, profile.quantum_ms)
    for step in range(total_steps):
        next_states: dict[tuple[int, int], int] = {}
        required = (
            _required_actual(step, total_steps, quality_constraint_id)
            or not allow_forecast
        )
        phase = _phase(step, total_steps)
        for (actual_count, run), cost in states.items():
            if not required and run < maximum_forecast_run:
                key = (actual_count, run + 1)
                next_states[key] = min(next_states.get(key, 10**18), cost + forecast_units)
            recovery = run >= 2
            cheapest = frontiers[(step, recovery)][0]
            key = (actual_count + 1, 0)
            actual_units = _ceil_units(profile.non_attention_ms, profile.quantum_ms)
            proposed = cost + actual_units + cheapest.conservative_units
            next_states[key] = min(next_states.get(key, 10**18), proposed)
        states = next_states
    feasible = [cost for (count, _), cost in states.items() if count >= minimum_actual]
    if not feasible:
        raise ValueError("joint trajectory has no feasible minimum-cost path")
    return min(feasible)


@lru_cache(maxsize=32)
def _cached_profile_frontiers(
    workload: JointWorkloadContext,
    quality_constraint_id: str,
    risk_model_id: str,
    total_steps: int,
) -> tuple[
    _InterpolatedProfile,
    Mapping[tuple[int, bool], tuple[_BandBundle, ...]],
]:
    """Reuse the acceleration-independent finite action lattice per shape."""

    profile = _profile(workload, risk_model_id)
    frontier_builder = (
        _layer_bundle_frontier
        if risk_model_id in (
            ROUND215_LAYER_RISK_MODEL,
            ROUND218_PHASE_LAYER_RISK_MODEL,
        )
        else _bundle_frontier
    )
    # Recovery only changes a frontier when it raises at least one physical
    # layer floor.  The causal-island policy deliberately fixes the causal
    # layers to Dense, so its recovery=True/False lattices are identical.
    # Building both used to repeat the exact 50-layer Pareto expansion and
    # made the cold planner miss its latency gate.  Reuse only *identical*
    # finite action spaces; phase remains in the key because it changes risk.
    unique: dict[tuple[object, ...], tuple[_BandBundle, ...]] = {}
    frontiers: dict[tuple[int, bool], tuple[_BandBundle, ...]] = {}
    for step in range(total_steps):
        phase = _phase(step, total_steps)
        # V8's measured temporal term is applied after selecting a bundle;
        # the layer/action frontier itself uses the shared step-3 ranking.
        temporal_anchor = None
        for recovery in (False, True):
            floor_signature = tuple(
                _minimum_rank(
                    _band_for_layer(layer)[0],
                    phase,
                    recovery,
                    quality_constraint_id,
                )
                for layer in range(50)
            )
            identity = (phase, floor_signature, temporal_anchor)
            frontier = unique.get(identity)
            if frontier is None:
                if risk_model_id in (
                    ROUND215_LAYER_RISK_MODEL,
                    ROUND218_PHASE_LAYER_RISK_MODEL,
                ):
                    frontier = frontier_builder(
                        profile,
                        phase,
                        recovery,
                        quality_constraint_id,
                        risk_model_id=risk_model_id,
                        temporal_anchor_index=temporal_anchor,
                    )
                else:
                    frontier = frontier_builder(
                        profile, phase, recovery, quality_constraint_id
                    )
                unique[identity] = frontier
            frontiers[(step, recovery)] = frontier
    return profile, frontiers


def clear_global_frontier_cache() -> None:
    _cached_profile_frontiers.cache_clear()


def _choice_digest(steps: tuple[GlobalStepChoice, ...]) -> str:
    document = [
        {"step": item.step_index, "actual": item.actual, "actions": item.actions}
        for item in steps
    ]
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _path_key(steps: tuple[GlobalStepChoice, ...]) -> tuple[tuple[object, ...], ...]:
    return tuple((item.step_index, item.actual, *item.actions) for item in steps)


def solve_global_joint_problem(
    total_steps: int,
    acceleration: float,
    *,
    workload: JointWorkloadContext,
    allow_forecast: bool = True,
    trajectory_prior: TrajectoryRiskPrior | None = None,
    action_implementation_id: str = FIXED_TOPK_ACTION_IMPLEMENTATION,
    quality_constraint_id: str = BASE_STRUCTURAL_CONSTRAINT,
    risk_model_id: str = BAND_RISK_MODEL,
) -> GlobalJointSolution:
    """Solve the full finite actual/forecast/layer-band problem exactly."""

    if not 4 <= total_steps <= 30:
        raise ValueError("total_steps must be inside [4, 30]")
    if not math.isfinite(acceleration) or not 0.0 < acceleration <= 100.0:
        raise ValueError("global DP acceleration must be inside (0, 100]")
    if action_implementation_id not in (
        FIXED_TOPK_ACTION_IMPLEMENTATION,
        ROUND215_ACTION_IMPLEMENTATION,
        ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
        ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
        ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION,
    ):
        raise ValueError("unsupported Attention action implementation")
    if quality_constraint_id not in (
        BASE_STRUCTURAL_CONSTRAINT,
        ROUND216_CAUSAL_ISLAND_CONSTRAINT,
        ROUND224_ADAPTIVE_LATENCY_CONSTRAINT,
        ROUND225_TRAJECTORY_CORRECTION_CONSTRAINT,
        ROUND226_OPENING_ANCHORED_MTCR_CONSTRAINT,
        ROUND227_FRONTIER_DOMINANCE_CONSTRAINT,
    ):
        raise ValueError("unsupported joint quality constraint")
    if risk_model_id not in (
        BAND_RISK_MODEL,
        ROUND215_LAYER_RISK_MODEL,
        ROUND218_PHASE_LAYER_RISK_MODEL,
    ):
        raise ValueError("unsupported joint risk model")
    profile, frontiers = _cached_profile_frontiers(
        workload, quality_constraint_id, risk_model_id, total_steps
    )
    if trajectory_prior is not None and any(
        step >= total_steps for step in trajectory_prior.supported_actual_steps
    ):
        raise ValueError("trajectory prior lies outside the requested trajectory")
    minimum_actual = _minimum_actual_count(
        total_steps, allow_forecast, quality_constraint_id
    )
    maximum_forecast_run = _maximum_forecast_run(quality_constraint_id)
    fastest_units = _minimum_path_cost_units(
        total_steps,
        profile,
        allow_forecast,
        frontiers,
        quality_constraint_id,
    )
    dense_units_per_step = _ceil_units(profile.dense_step_ms, profile.quantum_ms)
    dense_units = total_steps * dense_units_per_step
    normalized = acceleration / 100.0
    admitted_fast_units = fastest_units
    if quality_constraint_id == ROUND224_ADAPTIVE_LATENCY_CONSTRAINT:
        # The creator-facing 100 endpoint is an outcome contract, not the
        # mathematically cheapest path.  Reserve roughly 21% of full-Dense
        # DiT compute at the measured shape; on the 100k/20-step workload this
        # is the calibrated envelope for an approximately 200-second E2E
        # candidate after decode/mux.  The DP remains free to exchange actual
        # steps, forecasts and per-layer actions inside this budget.
        admitted_fast_units = max(
            fastest_units,
            int(math.ceil(dense_units * .21)),
        )
    elif quality_constraint_id in (
        ROUND225_TRAJECTORY_CORRECTION_CONSTRAINT,
        ROUND226_OPENING_ANCHORED_MTCR_CONSTRAINT,
        ROUND227_FRONTIER_DOMINANCE_CONSTRAINT,
    ):
        # Do not prescribe a historical trajectory.  The endpoint is instead
        # the cheapest path satisfying the 40% actual floor, two-forecast
        # bound and exact post-debt causal correction.  This makes the added
        # latency an auditable consequence of the Round224 failure rather
        # than an arbitrary back-off percentage.
        admitted_fast_units = fastest_units
    target_units = int(math.floor(
        dense_units - (dense_units - admitted_fast_units) * normalized**1.18
    ))
    # Two percent of Dense-equivalent compute is withheld from the initial
    # plan for request-local causal recovery.  The offline optimum is exact
    # for the remaining ledger, and the reserve is never silently spent.
    reserve_units = int(math.ceil(dense_units * .02 * normalized))
    budget_units = max(fastest_units, target_units - reserve_units)

    initial = _TrajectoryState(
        conservative_units=0,
        actual_cost_ms=0.0,
        risk=0.0,
        forecast_risk=0.0,
        attention_risk=0.0,
        components=HumanRiskVector(),
        actual_count=0,
        forecast_run=0,
        path=(),
    )
    states: dict[tuple[int, int, int], _TrajectoryState] = {(0, 0, 0): initial}
    forecast_units = _ceil_units(profile.forecast_ms, profile.quantum_ms)
    non_attention_units = _ceil_units(profile.non_attention_ms, profile.quantum_ms)
    state_count = 1
    for step in range(total_steps):
        proposed: dict[tuple[int, int, int], _TrajectoryState] = {}
        required = (
            _required_actual(step, total_steps, quality_constraint_id)
            or not allow_forecast
        )
        phase = _phase(step, total_steps)
        attention_risk_scale = (
            _temporal_risk_scale(profile, step, total_steps)
            if risk_model_id == ROUND218_PHASE_LAYER_RISK_MODEL
            else 1.0
        )
        step_frontiers = {
            recovery: tuple(
                _BandBundle(
                    conservative_units=bundle.conservative_units,
                    actual_cost_ms=bundle.actual_cost_ms,
                    risk=bundle.risk * attention_risk_scale,
                    components=_scale_components(
                        bundle.components, attention_risk_scale
                    ),
                    actions=bundle.actions,
                )
                for bundle in frontiers[(step, recovery)]
            )
            for recovery in (False, True)
        }
        remaining = total_steps - step - 1
        for state in states.values():
            if not required and state.forecast_run < maximum_forecast_run:
                cost_units = state.conservative_units + forecast_units
                if cost_units <= budget_units and state.actual_count + remaining >= minimum_actual:
                    debt = _forecast_increment(
                        step, total_steps, state.forecast_run + 1
                    )
                    if (
                        trajectory_prior is not None
                        and step in trajectory_prior.supported_actual_steps
                    ):
                        debt += trajectory_prior.unsupported_forecast_risk
                    candidate = _TrajectoryState(
                        cost_units,
                        state.actual_cost_ms + profile.forecast_ms,
                        state.risk + debt,
                        state.forecast_risk + debt,
                        state.attention_risk,
                        state.components,
                        state.actual_count,
                        state.forecast_run + 1,
                        state.path + (GlobalStepChoice(step, False),),
                    )
                    key = (candidate.actual_count, candidate.forecast_run, cost_units)
                    incumbent = proposed.get(key)
                    if incumbent is None or candidate.risk < incumbent.risk - 1.0e-12:
                        proposed[key] = candidate
            recovery = state.forecast_run >= 2
            for bundle in step_frontiers[recovery]:
                cost_units = (
                    state.conservative_units
                    + non_attention_units
                    + bundle.conservative_units
                )
                if cost_units > budget_units:
                    break
                candidate = _TrajectoryState(
                    cost_units,
                    state.actual_cost_ms + profile.non_attention_ms + bundle.actual_cost_ms,
                    state.risk + bundle.risk,
                    state.forecast_risk,
                    state.attention_risk + bundle.risk,
                    state.components + bundle.components,
                    state.actual_count + 1,
                    0,
                    state.path + (GlobalStepChoice(step, True, bundle.actions),),
                )
                key = (candidate.actual_count, 0, cost_units)
                incumbent = proposed.get(key)
                if incumbent is None or (
                    candidate.risk,
                    candidate.components.worst_component,
                ) < (
                    incumbent.risk,
                    incumbent.components.worst_component,
                ):
                    proposed[key] = candidate
        states = _prune(proposed)
        state_count += len(states)
        if not states:
            raise ValueError(f"joint DP became infeasible at step {step}")
    feasible = [state for state in states.values() if state.actual_count >= minimum_actual]
    if not feasible:
        raise ValueError("joint DP found no path satisfying the actual-step floor")
    optimum = min(
        feasible,
        key=lambda state: (
            state.risk,
            state.components.worst_component,
            -state.conservative_units,
            -state.actual_count,
        ),
    )
    certificate_model_sha = hashlib.sha256(
        json.dumps(
            {
                "shape_model_sha256": profile.model_sha256,
                "action_implementation_id": action_implementation_id,
                "quality_constraint_id": quality_constraint_id,
                "maximum_forecast_run": maximum_forecast_run,
                "risk_model_id": risk_model_id,
                "total_steps": total_steps,
                "trajectory_prior": (
                    None
                    if trajectory_prior is None
                    else {
                        "prior_id": trajectory_prior.prior_id,
                        "supported_actual_steps": trajectory_prior.supported_actual_steps,
                        "unsupported_forecast_risk": trajectory_prior.unsupported_forecast_risk,
                    }
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    certificate = GlobalJointCertificate(
        schema_version=GLOBAL_JOINT_CERTIFICATE_SCHEMA,
        solver="exact Pareto-pruned finite dynamic program",
        objective="min additive calibrated risk; tie min worst component; tie max compute",
        formal_scope=(
            "finite actual/forecast placements + "
            + (
                "phase-binned 50-layer actions"
                if risk_model_id == ROUND218_PHASE_LAYER_RISK_MODEL
                else (
                    "50-layer actions"
                    if risk_model_id == ROUND215_LAYER_RISK_MODEL
                    else "seven-band actions"
                )
            )
            + " + hard anchors "
            "+ additive calibrated risk + conservative quantised shape budget"
        ),
        cost_quantum_ms=profile.quantum_ms,
        budget_units=budget_units,
        minimum_actual_count=minimum_actual,
        state_count=state_count,
        conservative_cost_units=optimum.conservative_units,
        optimum_risk=optimum.risk,
        choice_sha256=_choice_digest(optimum.path),
        model_sha256=certificate_model_sha,
        action_implementation_id=action_implementation_id,
        quality_constraint_id=quality_constraint_id,
        risk_model_id=risk_model_id,
    )
    return GlobalJointSolution(
        steps=optimum.path,
        predicted_cost_ms=optimum.actual_cost_ms,
        conservative_cost_ms=optimum.conservative_units * profile.quantum_ms,
        target_cost_ms=(budget_units + reserve_units) * profile.quantum_ms,
        dense_cost_ms=dense_units * profile.quantum_ms,
        fastest_cost_ms=fastest_units * profile.quantum_ms,
        recovery_reserve_ms=reserve_units * profile.quantum_ms,
        forecast_risk=optimum.forecast_risk,
        attention_risk=optimum.attention_risk,
        components=optimum.components,
        calibration_mix=profile.mix,
        extrapolated=profile.extrapolated,
        certificate=certificate,
    )


def verify_global_joint_solution(
    solution: GlobalJointSolution,
    total_steps: int,
    acceleration: float,
    *,
    workload: JointWorkloadContext,
    allow_forecast: bool = True,
    trajectory_prior: TrajectoryRiskPrior | None = None,
    action_implementation_id: str = FIXED_TOPK_ACTION_IMPLEMENTATION,
    quality_constraint_id: str = BASE_STRUCTURAL_CONSTRAINT,
    risk_model_id: str = BAND_RISK_MODEL,
) -> GlobalJointVerification:
    """Replay the complete finite DP and compare the optimum certificate."""

    reasons: list[str] = []
    if solution.certificate.schema_version != GLOBAL_JOINT_CERTIFICATE_SCHEMA:
        reasons.append("unsupported global certificate schema")
    try:
        replay = solve_global_joint_problem(
            total_steps,
            acceleration,
            workload=workload,
            allow_forecast=allow_forecast,
            trajectory_prior=trajectory_prior,
            action_implementation_id=action_implementation_id,
            quality_constraint_id=quality_constraint_id,
            risk_model_id=risk_model_id,
        )
    except Exception as error:  # pragma: no cover - verifier defensive boundary
        return GlobalJointVerification(False, (f"global DP replay failed: {error}",))
    expected = replay.certificate
    for field in (
        "cost_quantum_ms",
        "budget_units",
        "minimum_actual_count",
        "conservative_cost_units",
        "choice_sha256",
        "model_sha256",
        "action_implementation_id",
        "quality_constraint_id",
        "risk_model_id",
    ):
        if getattr(solution.certificate, field) != getattr(expected, field):
            reasons.append(f"global certificate {field} mismatch")
    if not math.isclose(
        solution.certificate.optimum_risk,
        expected.optimum_risk,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        reasons.append("global certificate optimum risk mismatch")
    if _choice_digest(solution.steps) != solution.certificate.choice_sha256:
        reasons.append("global solution choice digest mismatch")
    return GlobalJointVerification(not reasons, tuple(reasons))


def solve_no_trigger_online_rebate(
    solution: GlobalJointSolution,
    *,
    workload: JointWorkloadContext,
    total_steps: int,
    limit_dense_layers: float,
    probe_slots: tuple[tuple[int, int], ...],
    quality_constraint_id: str,
    risk_model_id: str,
) -> OnlineRebateSolution:
    """Spend an otherwise dead reserve on the best remaining Dense cells.

    This is a *conditional* optimum: every scheduled probe has completed and
    no guard fired, so observation and emergency work already have priority.
    Each runtime upgrade is conservatively charged as one full Dense-layer
    equivalent.  With equal charges the finite optimum is exactly the top-K
    remaining sparse cells by calibrated risk reduction.  The result does not
    claim that the additive tensor-risk surrogate equals Human preference.
    """

    if risk_model_id != ROUND218_PHASE_LAYER_RISK_MODEL:
        raise ValueError("online rebate requires the phase-layer risk model")
    if not math.isfinite(limit_dense_layers) or limit_dense_layers < 0.0:
        raise ValueError("online rebate limit must be finite and non-negative")
    if not probe_slots:
        raise ValueError("online rebate requires a completed probe schedule")
    raw_slots = tuple((int(step), int(layer)) for step, layer in probe_slots)
    if len(set(raw_slots)) != len(raw_slots):
        raise ValueError("online rebate probe slots must be unique")
    normalized_slots = tuple(sorted(raw_slots))
    last_probe_step = max(step for step, _ in normalized_slots)
    maximum_choices = max(
        0,
        int(math.floor(limit_dense_layers - len(normalized_slots) + 1.0e-9)),
    )
    profile, _ = _cached_profile_frontiers(
        workload, quality_constraint_id, risk_model_id, total_steps
    )
    candidates: list[OnlineRebateChoice] = []
    model_rows: list[dict[str, object]] = []
    for step in solution.steps:
        if not step.actual or step.step_index <= last_probe_step:
            continue
        if len(step.actions) != 50:
            raise ValueError("online rebate requires physical 50-layer actions")
        phase = _phase(step.step_index, total_steps)
        temporal_scale = _temporal_risk_scale(
            profile, step.step_index, total_steps
        )
        for layer, action_name in enumerate(step.actions):
            if action_name == "dense":
                continue
            action = _layer_action(
                profile,
                layer,
                action_name,
                phase,
                risk_model_id=risk_model_id,
            )
            risk_reduction = action.risk * temporal_scale
            choice = OnlineRebateChoice(
                step_index=step.step_index,
                layer=layer,
                from_action=action_name,
                surrogate_risk_reduction=risk_reduction,
            )
            candidates.append(choice)
            model_rows.append(
                {
                    "step": choice.step_index,
                    "layer": choice.layer,
                    "from_action": choice.from_action,
                    "surrogate_risk_reduction": round(risk_reduction, 15),
                }
            )
    ranked = sorted(
        candidates,
        key=lambda item: (
            -item.surrogate_risk_reduction,
            item.step_index,
            item.layer,
            item.from_action,
        ),
    )
    selected = tuple(ranked[:maximum_choices])
    choice_document = [
        {
            "step": item.step_index,
            "layer": item.layer,
            "from_action": item.from_action,
        }
        for item in selected
    ]
    choice_sha = hashlib.sha256(
        json.dumps(
            choice_document, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    model_sha = hashlib.sha256(
        json.dumps(
            {
                "global_model_sha256": solution.certificate.model_sha256,
                "global_choice_sha256": solution.certificate.choice_sha256,
                "workload": {
                    "packed_tokens": workload.packed_tokens,
                    "condition_count": workload.condition_count,
                    "service_family": workload.service_family,
                    "model_variant": workload.model_variant,
                },
                "limit_dense_layers": round(limit_dense_layers, 15),
                "probe_slots": normalized_slots,
                "quality_constraint_id": quality_constraint_id,
                "risk_model_id": risk_model_id,
                "candidates": model_rows,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    selected_risk = sum(item.surrogate_risk_reduction for item in selected)
    return OnlineRebateSolution(
        choices=selected,
        certificate=OnlineRebateCertificate(
            schema_version=ONLINE_REBATE_CERTIFICATE_SCHEMA,
            solver="exact equal-charge top-k conditional allocation",
            objective="max additive calibrated risk reduction after no trigger",
            maximum_choices=maximum_choices,
            candidate_count=len(candidates),
            selected_count=len(selected),
            selected_risk_reduction=selected_risk,
            choice_sha256=choice_sha,
            model_sha256=model_sha,
        ),
    )


__all__ = [
    "ACTION_NAMES",
    "GLOBAL_JOINT_CERTIFICATE_SCHEMA",
    "ONLINE_REBATE_CERTIFICATE_SCHEMA",
    "GlobalJointCertificate",
    "GlobalJointSolution",
    "GlobalJointVerification",
    "OnlineRebateCertificate",
    "OnlineRebateChoice",
    "OnlineRebateSolution",
    "FIXED_TOPK_ACTION_IMPLEMENTATION",
    "BASE_STRUCTURAL_CONSTRAINT",
    "BAND_RISK_MODEL",
    "GlobalStepChoice",
    "JointWorkloadContext",
    "LAYER_BANDS",
    "ROUND143_216_TRAJECTORY_PRIOR",
    "ROUND215_ACTION_IMPLEMENTATION",
    "ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION",
    "ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION",
    "ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION",
    "ROUND216_CAUSAL_ISLAND_CONSTRAINT",
    "ROUND224_ADAPTIVE_LATENCY_CONSTRAINT",
    "ROUND225_TRAJECTORY_CORRECTION_CONSTRAINT",
    "ROUND226_OPENING_ANCHORED_MTCR_CONSTRAINT",
    "ROUND227_FRONTIER_DOMINANCE_CONSTRAINT",
    "ROUND215_LAYER_RISK_MODEL",
    "ROUND218_PHASE_LAYER_RISK_MODEL",
    "TrajectoryRiskPrior",
    "clear_global_frontier_cache",
    "solve_global_joint_problem",
    "solve_no_trigger_online_rebate",
    "verify_global_joint_solution",
]
