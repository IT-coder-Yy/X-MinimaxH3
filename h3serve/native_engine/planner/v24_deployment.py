"""Human-calibrated Pareto deployment policy for the H3 Base model.

The creator-facing contract remains exactly two controls: the requested sigma
trajectory length and one acceleration value in ``[0, 100]``.  V24 turns those
controls, exact generated-video geometry and packed resource counts into one
complete Actual/Forecast and per-layer Attention schedule.

The reviewed V19 candidates below are calibration observations, not runtime
"versions" selected by the user.  Between Dense and the calibrated endpoint,
V24 constructs a nested partial-order chain:

* every less accelerated point contains every Actual step of the faster point;
* every shared Attention cell is at least as faithful as at the faster point;
* no prompt words, scene labels, seed or media content are inspected.

Consequently, acceleration 100 means the fastest point currently admitted by
the Human evidence surface, rather than an unreviewed extrapolation beyond it.
The surface can be expanded later by adding reviewed observations without
changing the public API.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math

from .v19_planner import V19PlanningError, V19WorkloadContext
from .v19_candidates import v19_blueprint_execution_digest
from .v19_runtime_bridge import (
    ROUND229_FORECAST_ANCHOR,
    blueprint_from_runtime_schedule,
)


V24_DEPLOYMENT_SCHEMA = "h3_pareto_v24_runtime_selection_v2"
V24_DEPLOYMENT_POLICY_ID = "h3_pareto_v24_human_calibrated_deployment_v2"
# Human review currently reaches the 1080p15/xlong neighborhood.  The same
# fractional action surface can be evaluated beyond it because geometry enters
# through continuous video-token cost and the largest anchor is the explicit
# boundary condition.  Keep the evidence and operational limits separate:
# the latter admits the measured 2K15 packed shape, while telemetry must never
# misrepresent that extrapolation as direct Human calibration.
V24_HUMAN_EVIDENCE_MAX_PACKED_TOKENS = 250_000
V24_MAX_PACKED_TOKENS = 400_000

# Runtime action character encoding.  Upper-case sparse actions use the
# Round229 forecast-aware physical rail, lower-case actions use the reviewed
# Round188 frontier rail, and D is exact Dense Attention.
_CODE_TO_CANONICAL = {
    "S": "sparse_topk_0.0625",
    "T": "sparse_topk_0.1",
    "Q": "sparse_topk_0.25",
    "H": "sparse_topk_0.5",
    "s": "sparse_topk_0.0625",
    "t": "sparse_topk_0.1",
    "q": "sparse_topk_0.25",
    "h": "sparse_topk_0.5",
    "D": "dense",
}
_CANONICAL_RANK = {
    "sparse_topk_0.0625": 0,
    "sparse_topk_0.1": 1,
    "sparse_topk_0.25": 2,
    "sparse_topk_0.5": 3,
    "dense": 4,
}
_RANK_TO_CANONICAL = {rank: name for name, rank in _CANONICAL_RANK.items()}
_ATTENTION_COST_RATIO = {
    "sparse_topk_0.0625": 0.213,
    "sparse_topk_0.1": 0.247,
    "sparse_topk_0.25": 0.403,
    "sparse_topk_0.5": 0.669,
    "dense": 1.0,
}
_NON_ATTENTION_COMPUTE = 0.286
_ATTENTION_COMPUTE = 1.0 - _NON_ATTENTION_COMPUTE
_FORECAST_COMPUTE = 0.045
_ACTION_RISK = {
    0: 1.00,
    1: 0.72,
    2: 0.41,
    3: 0.17,
    4: 0.00,
}
_V24_OPTIMIZER_ID = "v24_nested_marginal_risk_waterfill_v1"
_V24_FORECAST_FEEDBACK_POLICY_ID = "v24_request_local_forecast_debt_v1"
V24_CURVE_PROFILE_SCHEMA = "h3_v24_curve_profile_v1"


@dataclass(frozen=True, slots=True)
class V24CurveProfile:
    """Low-dimensional parameters which generate one high-dimensional path.

    The physical action alphabet is deliberately absent from this object: a
    profile may only redistribute the already admitted Actual/Forecast and
    Attention actions.  Human feedback updates these smooth coefficients,
    never a prompt-specific schedule table.
    """

    profile_id: str = "v24_curve_control_v1"
    budget_exponent: float = 1.12
    forecast_risk_scale: float = 1.0
    forecast_run_coupling: float = 0.22
    opening_amplitude: float = 1.10
    opening_decay: float = 0.16
    terminal_amplitude: float = 0.82
    terminal_decay: float = 0.13
    causal_layer_amplitude: float = 0.72
    causal_layer_center: float = 37.0
    causal_layer_width: float = 6.5
    bridge_layer_amplitude: float = 0.34
    bridge_layer_center: float = 45.0
    bridge_layer_width: float = 2.8

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise V19PlanningError("V24 curve profile id cannot be empty")
        values = tuple(
            float(value)
            for name, value in asdict(self).items()
            if name != "profile_id"
        )
        if any(not math.isfinite(value) for value in values):
            raise V19PlanningError("V24 curve parameters must be finite")
        if self.budget_exponent <= 0.0 or self.forecast_risk_scale <= 0.0:
            raise V19PlanningError("V24 curve budget/risk scales must be positive")
        if self.forecast_run_coupling < 0.0:
            raise V19PlanningError("V24 Forecast-run coupling cannot be negative")
        if min(
            self.opening_amplitude,
            self.terminal_amplitude,
            self.causal_layer_amplitude,
            self.bridge_layer_amplitude,
        ) < 0.0:
            raise V19PlanningError("V24 curve amplitudes cannot be negative")
        if min(
            self.opening_decay,
            self.terminal_decay,
            self.causal_layer_width,
            self.bridge_layer_width,
        ) <= 0.0:
            raise V19PlanningError("V24 curve widths must be positive")
        if not 0.0 <= self.causal_layer_center <= 49.0 or not (
            0.0 <= self.bridge_layer_center <= 49.0
        ):
            raise V19PlanningError("V24 curve layer centers must lie inside H3")

    @property
    def parameter_digest(self) -> str:
        payload = asdict(self)
        payload.pop("profile_id")
        return hashlib.sha256(json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": V24_CURVE_PROFILE_SCHEMA,
            **asdict(self),
            "parameter_digest": self.parameter_digest,
        }


V24_DEFAULT_CURVE_PROFILE = V24CurveProfile()


@dataclass(frozen=True, slots=True)
class V24HumanAnchor:
    """One reviewed observation used to calibrate the deployment surface."""

    anchor_id: str
    video_tokens: int
    rows: tuple[tuple[int, str], ...]
    source_execution_digest: str
    artifact_sha256: tuple[str, ...]
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.video_tokens <= 0 or not self.anchor_id:
            raise V19PlanningError("V24 Human anchor identity is invalid")
        steps = tuple(step for step, _row in self.rows)
        if tuple(sorted(set(steps))) != steps:
            raise V19PlanningError("V24 Human anchor steps must be sorted and unique")
        if not steps or steps[0] != 0 or steps[-1] != 19:
            raise V19PlanningError("V24 Human anchor must protect both endpoints")
        for _step, row in self.rows:
            if len(row) != 50 or any(code not in _CODE_TO_CANONICAL for code in row):
                raise V19PlanningError("V24 Human anchor contains an invalid layer row")
        digests = (self.source_execution_digest, *self.artifact_sha256)
        if any(len(value) != 64 for value in digests):
            raise V19PlanningError("V24 Human anchor requires SHA256 provenance")

    @property
    def actual_steps(self) -> tuple[int, ...]:
        return tuple(step for step, _row in self.rows)

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(
            {
                "anchor_id": self.anchor_id,
                "video_tokens": self.video_tokens,
                "rows": self.rows,
                "source_execution_digest": self.source_execution_digest,
                "artifact_sha256": self.artifact_sha256,
                "evidence": self.evidence,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()


_SHORT_V007_ROWS = (
    (0, "HQDDDQQQDQQQQDQDQHDQQQDHQHHHHHDDDDDDDDDDDDDDDDDDHD"),
    (1, "HQTQQQQQTQQQTTQTTHTQQQTQQHQQQQDDDDDDDDDDDDDDQDQDHD"),
    (2, "QTQQQQQTQTQQTDQQTQTQQQTQQHQTQQDDDDDDDDDDDDDDQDQHQD"),
    (3, "QQQQTTQTTTQTTTQTTQQQQQQQQHQQQQDDDDDDDDDDDDDDQDQDQD"),
    (4, "QQQQQQHTQTQTQTQQQQQTQQTQQHQQQQDDDDDDDDDDDDDDQDQHQS"),
    (8, "QQQQQQQTTTTTQTTQQQQQQQTQQHTTQSDDDDDDDDDDDDDDQDQHQD"),
    (12, "HQTQQQHQTTTQTTTTQQQQHTQQQHQTQQDDDDDDDDDDDDDDQDQQQT"),
    (15, "QQTTQQQQTTTTTTTTQQQTQTQQQHQTQSDDDDDDDDDDDDDDQDQQQS"),
    (18, "QSQQQHQQTQQQQQQQQQQQQTQQQHQQQQDDDDDDDDDDDDDDQDHHQD"),
    (19, "QSQQQHHQTQTQQQQQQQQQQTHQQHQQQQDDDDDDDDDDDDDDQDHHHD"),
)

_MEDIUM_V022_ROWS = (
    (0, "ssssssssssssssssssssssssssssssDDDDDDDDDDDDDDsDssss"),
    (1, "ssssssssssssssssssssssssssssssttttttttthhhhhshssss"),
    (2, "ssssssssssssssssssssssssssssssttttttttthhhhhshssss"),
    (3, "ssssssssssssssssssssssssssssssttttttttthhhhhshssss"),
    (4, "ssssssssssssssssssssssssssssssttttttttthhhhhshssss"),
    (8, "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqhhhhhqhqqqq"),
    (12, "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqhhhhhqhqqqq"),
    (15, "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqhhhhhqhqqqq"),
    (18, "HDQQHHDQQQQQQQQDQQQQDDDDDHQDQDDDDDDDDDDDDDDDDDDHHD"),
    (19, "HDQQQHHQQQQQQQQQQQQQDDDDDHQDHHDDDDDDDDDDDDDDDDDDHD"),
)

_LONG_V009_ROWS = (
    (0, "HHDHDQQQDQDQQDQDQHQQQQDDQHHQHQDDDDDDDDDDDDDDDDDDDD"),
    (1, "HQHQQQQQQQQQQDQQQHQQQQQQQHQQQQHHHHDDDHDDDDDDQDQDHD"),
    (2, "HQHQQQQQQQQQQDQDQQQQQQQQQHQQQQHDDHDDDHDDDDDDQDQDQD"),
    (3, "HQHQQHQQQQQQQDQQQQQQQQQQQHQQQQHHHHDDDHDDDDDDQDQDQD"),
    (4, "HQQQQQHQQQQQQDQQQQQQQQQQQHQQQQHDDHDDDHHDDDDDQDQDQD"),
    (8, "QQQQQQQQQQQQQDQQQQQQQQQQQHQQQQHHHHHHDHHDDDDDQDQQQQ"),
    (12, "QQQQQQHQQQQQQQQQQQQQQQQQQHQQQQHHHHHHHHHDDDDDQDQQQQ"),
    (15, "QQQQQQQQQQQQQQQQQQQQQQQQQHQQQQHHHHDHDHHDDDDDQDQQQQ"),
    (18, "HDQQHHDQQQQQQQQDQQQQDDDDDHQDQDDDDDDDDDDDDDDDDDDHHD"),
    (19, "HDQQQHHQQQQQQQQQQQQQDDDDDHQDHHDDDDDDDDDDDDDDDDDDHD"),
)

_XL_V012_ROWS = (
    (0, "ssssssssssssssssssssssssssssssttttttttttttttstssss"),
    (1, "ssssssssssssssssssssssssssssssttttttttttttttstssss"),
    (2, "ssssssssssssssssssssssssssssssttttttttttttttstssss"),
    (3, "ssssssssssssssssssssssssssssssttttttttttttttstssss"),
    (4, "ssssssssssssssssssssssssssssssttttttttttttttstssss"),
    (6, "ssssssssssssssssssssssssssssssttttttttttttttstssss"),
    (8, "ssssssssssssssssssssssssssssssttttttttttttttstssss"),
    (11, "ssssssssssssssssssssssssssssssttttttttttttttstssss"),
    (14, "ssssssssssssssssssssssssssssssttttttttttttttstssss"),
    (17, "ssssssssssssssssssssssssssssssttttttttttttttstssss"),
    (18, "ssssssssssssssssssssssssssssssttttttttttttttstssss"),
    (19, "ssssssssssssssssssssssssssssssttttttttttttttstssss"),
)


V24_HUMAN_ANCHORS = (
    V24HumanAnchor(
        anchor_id="short_34k_v007_quality",
        video_tokens=34_040,
        rows=_SHORT_V007_ROWS,
        source_execution_digest="c48c8b25ea97641100d07cc826ee336ebc8ecc77435ad9f80082aa91aca3abc9",
        artifact_sha256=(
            "1ce3b2fed494db988693bd4cb8ee1a6a922f8717af4880ec311cde9bf0b4886a",
        ),
        evidence=(
            "Human: v007 quality significantly exceeds the legacy acceleration-75 comparator",
            "Human: retain as the 720p5 single-scene quality frontier",
        ),
    ),
    V24HumanAnchor(
        anchor_id="medium_66k_v022_causal",
        video_tokens=66_240,
        rows=_MEDIUM_V022_ROWS,
        source_execution_digest="f279d44e88e798c1c273329268e36af020795a1bb2e6b54459da65a103488cae",
        artifact_sha256=(
            "6703b105fa5a443c1ee0755542cca7d66a85ee5b2011b88299f4ac9a697f1eb1",
        ),
        evidence=(
            "Human: V22效果很好",
            "720p10 opening causal spine and terminal replay quality gate passed",
        ),
    ),
    V24HumanAnchor(
        anchor_id="long_98k_v009_stable",
        video_tokens=98_440,
        rows=_LONG_V009_ROWS,
        source_execution_digest="4c7bf37f80390dae9c6eb69985813c9211e994d5d3a70fd969a0797b6634396a",
        artifact_sha256=(
            "1173bd8f611ae1265e690ad5e20f2165f4b2863d34477941ca2022731882010e",
            "15b182db3095d32aabd99373f9f52e45cd5d81ad516ab1d3631d0cb4d896dd5b",
        ),
        evidence=(
            "Human: v009 is the 720p15 quality anchor",
            "Human: no v014-family peripheral jitter, deformation or bright spots",
        ),
    ),
    V24HumanAnchor(
        anchor_id="xlong_218k_v012_v018_exact",
        video_tokens=218_280,
        rows=_XL_V012_ROWS,
        source_execution_digest="495c2c8ff75f76aed16b1fd81a41f3e050df5bad38456dd68d0806c3e9c7cbad",
        artifact_sha256=(
            "3affb34e3900ebee63896120b769b0bce806fbe6e7c5f7366c745bfadf8528b5",
        ),
        evidence=(
            "Human: 1080p15 v013 output family is best",
            "v018 exact execution is byte-identical and 1.028x faster than v013",
        ),
    ),
)

V24_HUMAN_SURFACE_DIGEST = hashlib.sha256(json.dumps(
    [anchor.digest for anchor in V24_HUMAN_ANCHORS],
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class V24DeploymentSelection:
    actual_step_indices: tuple[int, ...]
    attention_action_schedule: tuple[tuple[int, int, str], ...]
    summary: dict[str, object]
    schema_version: str = V24_DEPLOYMENT_SCHEMA

    def __post_init__(self) -> None:
        if not self.actual_step_indices:
            raise V19PlanningError("V24 selection requires an Actual step")
        if tuple(sorted(set(self.actual_step_indices))) != self.actual_step_indices:
            raise V19PlanningError("V24 Actual steps must be sorted and unique")
        if tuple(sorted(set(self.attention_action_schedule))) != (
            self.attention_action_schedule
        ):
            raise V19PlanningError("V24 Attention schedule must be sorted and unique")


def _video_tokens(workload: V19WorkloadContext) -> int:
    if workload.width is None or workload.height is None or workload.frames is None:
        raise V19PlanningError("V24 requires exact output geometry")
    if workload.width % 32 or workload.height % 32:
        raise V19PlanningError("V24 geometry must be divisible by 32")
    if workload.frames < 5 or (workload.frames - 5) % 17:
        raise V19PlanningError("V24 frames must satisfy the H3 17*n+5 grid")
    latent_frames = ((workload.frames - 5) // 17) * 5 + 2
    return latent_frames * (workload.height // 32) * (workload.width // 32)


def _anchor_bracket(video_tokens: int) -> tuple[V24HumanAnchor, V24HumanAnchor, float]:
    if video_tokens <= V24_HUMAN_ANCHORS[0].video_tokens:
        anchor = V24_HUMAN_ANCHORS[0]
        return anchor, anchor, 0.0
    if video_tokens >= V24_HUMAN_ANCHORS[-1].video_tokens:
        anchor = V24_HUMAN_ANCHORS[-1]
        return anchor, anchor, 0.0
    for lower, upper in zip(V24_HUMAN_ANCHORS, V24_HUMAN_ANCHORS[1:]):
        if lower.video_tokens <= video_tokens <= upper.video_tokens:
            if video_tokens == lower.video_tokens:
                return lower, lower, 0.0
            if video_tokens == upper.video_tokens:
                return upper, upper, 0.0
            # Attention scaling is closer to logarithmic than linear across
            # these packed-video sizes.  Interpolate only the calibration
            # surface; runtime action choices remain discrete and auditable.
            mix = (
                math.log(video_tokens) - math.log(lower.video_tokens)
            ) / (
                math.log(upper.video_tokens) - math.log(lower.video_tokens)
            )
            return lower, upper, min(1.0, max(0.0, mix))
    raise AssertionError("unreachable V24 anchor bracket")


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _select_bounded_steps(
    *,
    total_steps: int,
    count: int,
    score,
) -> tuple[int, ...]:
    """Select a scored Actual set while structurally bounding forecast gaps."""

    opening_count = min(count, max(1, int(math.ceil(total_steps * 0.25))))
    selected = set(range(opening_count))
    if len(selected) < count:
        selected.add(total_steps - 1)
    if (
        len(selected) < count
        and total_steps > 2
        and count - len(selected) >= 2
    ):
        selected.add(total_steps - 2)

    # Split every initially uncovered interval into segments no wider than
    # four solver positions (three forecasts).  Computing all required split
    # points together avoids a greedy midpoint consuming a slot needed by the
    # opposite half of a large interval.
    ordered = sorted(selected)
    structural: list[int] = []
    for left, right in zip(ordered, ordered[1:]):
        distance = right - left
        segments = int(math.ceil(distance / 4.0))
        for index in range(1, segments):
            structural.append(_round_half_up(
                left + index * distance / segments
            ))
    for insertion in structural:
        selected.add(insertion)
    if len(selected) > count:
        raise V19PlanningError("V24 Actual count cannot satisfy forecast bound")

    for step in sorted(range(total_steps), key=lambda value: (score(value), value)):
        if len(selected) >= count:
            break
        selected.add(step)
    if len(selected) != count:
        raise V19PlanningError("V24 failed to materialize its Actual count")
    return tuple(sorted(selected))


def _scaled_anchor_steps(anchor: V24HumanAnchor, total_steps: int) -> tuple[int, ...]:
    if total_steps == 20:
        return anchor.actual_steps
    count = min(
        total_steps,
        max(2, _round_half_up(len(anchor.actual_steps) / 20.0 * total_steps)),
    )
    def score(step: int) -> float:
        position = step / max(1, total_steps - 1)
        return min(
            abs(position - source / 19.0) for source in anchor.actual_steps
        )

    return _select_bounded_steps(
        total_steps=total_steps,
        count=count,
        score=score,
    )


def _longest_forecast_run(actual: set[int], total_steps: int) -> tuple[int, ...]:
    longest: tuple[int, ...] = ()
    current: list[int] = []
    for step in range(total_steps):
        if step in actual:
            if len(current) > len(longest):
                longest = tuple(current)
            current = []
        else:
            current.append(step)
    if len(current) > len(longest):
        longest = tuple(current)
    return longest


def _endpoint_actual_steps(
    *,
    total_steps: int,
    lower: V24HumanAnchor,
    upper: V24HumanAnchor,
    mix: float,
) -> tuple[int, ...]:
    if lower.anchor_id == upper.anchor_id:
        return _scaled_anchor_steps(lower, total_steps)
    lower_steps = _scaled_anchor_steps(lower, total_steps)
    upper_steps = _scaled_anchor_steps(upper, total_steps)
    count = _round_half_up((1.0 - mix) * len(lower_steps) + mix * len(upper_steps))
    def distance(step: int, anchors: tuple[int, ...]) -> float:
        position = step / max(1, total_steps - 1)
        return min(
            abs(position - source / max(1, total_steps - 1))
            for source in anchors
        )

    def score(step: int) -> float:
        return (
            (1.0 - mix) * distance(step, lower_steps)
            + mix * distance(step, upper_steps)
        )

    return _select_bounded_steps(
        total_steps=total_steps,
        count=count,
        score=score,
    )


def _nearest_row(anchor: V24HumanAnchor, step: int, total_steps: int) -> str:
    position = step / max(1, total_steps - 1)
    source_step, row = min(
        anchor.rows,
        key=lambda item: (abs(position - item[0] / 19.0), item[0]),
    )
    del source_step
    return row


def _decode_code(code: str) -> tuple[int, str | None]:
    canonical = _CODE_TO_CANONICAL[code]
    if canonical == "dense":
        return 4, None
    return (
        _CANONICAL_RANK[canonical],
        "forecastfrontier" if code.isupper() else "frontier",
    )


def _endpoint_cell(
    *,
    step: int,
    layer: int,
    total_steps: int,
    lower: V24HumanAnchor,
    upper: V24HumanAnchor,
    mix: float,
) -> tuple[int, str | None]:
    lower_code = _nearest_row(lower, step, total_steps)[layer]
    if lower.anchor_id == upper.anchor_id:
        return _decode_code(lower_code)
    upper_code = _nearest_row(upper, step, total_steps)[layer]
    lower_rank, lower_prefix = _decode_code(lower_code)
    upper_rank, upper_prefix = _decode_code(upper_code)
    rank = _round_half_up((1.0 - mix) * lower_rank + mix * upper_rank)
    if rank == 4:
        return rank, None
    preferred = lower_prefix if mix < 0.5 else upper_prefix
    return rank, preferred or lower_prefix or upper_prefix or "forecastfrontier"


def _context_progress(
    workload: V19WorkloadContext,
    *,
    video_tokens: int,
    requested_progress: float,
) -> tuple[float, float, tuple[str, ...]]:
    """Apply resource-only guards without inspecting conditioning semantics."""

    guard = 0.0
    reasons: list[str] = []
    non_video_ratio = max(0, workload.packed_tokens - video_tokens) / video_tokens
    if non_video_ratio > 0.05:
        guard += min(0.15, (non_video_ratio - 0.05) * 0.5)
        reasons.append("extended_packed_prefix")
    if workload.condition_count:
        guard += min(0.18, 0.03 * workload.condition_count)
        reasons.append("conditioning_rows")
    if workload.service_family == "reference":
        guard += 0.10
        reasons.append("reference_layout_guard")
    if workload.reference_images or workload.reference_audio:
        guard += min(
            0.10,
            0.02 * workload.reference_images + 0.03 * workload.reference_audio,
        )
        reasons.append("reference_media_guard")
    guard = min(0.45, guard)
    trajectory = requested_progress * (1.0 - guard)
    attention = requested_progress * math.sqrt(1.0 - guard)
    if workload.reference_videos:
        # A reference video can dominate the prefix length and has no matched
        # long-horizon Human anchor.  Keep every DiT evaluation and allow only
        # a conservative part of the Attention path.
        trajectory = 0.0
        attention *= 0.5
        reasons.append("reference_video_no_forecast_guard")
    return (
        min(1.0, max(0.0, trajectory)),
        min(1.0, max(0.0, attention)),
        tuple(dict.fromkeys(reasons)),
    )


def _runtime_action(rank: int, prefix: str | None) -> str:
    canonical = _RANK_TO_CANONICAL[rank]
    if canonical == "dense":
        return "dense"
    return f"{prefix or 'forecastfrontier'}:{canonical}"


def _compute_units(
    *,
    total_steps: int,
    actual: tuple[int, ...],
    schedule: tuple[tuple[int, int, str], ...],
) -> float:
    actual_set = set(actual)
    actions = {
        (step, layer): action.rsplit(":", 1)[-1]
        for step, layer, action in schedule
        if step in actual_set
    }
    if not schedule:
        return float(total_steps)
    attention = sum(_ATTENTION_COST_RATIO[action] for action in actions.values())
    return (
        len(actual) * _NON_ATTENTION_COMPUTE
        + _ATTENTION_COMPUTE * attention / 50.0
        + (total_steps - len(actual)) * _FORECAST_COMPUTE
    )


@dataclass(frozen=True, slots=True)
class _UpgradeOperation:
    """One quality-increasing move on the single nested deployment path."""

    kind: str
    step: int
    layer: int
    from_rank: int
    to_rank: int
    cost_delta: float
    risk_reduction: float

    @property
    def utility(self) -> float:
        return self.risk_reduction / max(self.cost_delta, 1.0e-12)

    def audit_row(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "step": self.step,
            "layer": self.layer,
            "from_rank": self.from_rank,
            "to_rank": self.to_rank,
            "cost_delta": round(self.cost_delta, 15),
            "risk_reduction": round(self.risk_reduction, 15),
        }


def _phase_risk_weight(
    step: int,
    total_steps: int,
    curve: V24CurveProfile = V24_DEFAULT_CURVE_PROFILE,
) -> float:
    position = step / max(1, total_steps - 1)
    opening = math.exp(-position / curve.opening_decay)
    terminal = math.exp(-(1.0 - position) / curve.terminal_decay)
    return (
        1.0
        + curve.opening_amplitude * opening
        + curve.terminal_amplitude * terminal
    )


def _layer_risk_weight(
    layer: int,
    curve: V24CurveProfile = V24_DEFAULT_CURVE_PROFILE,
) -> float:
    # Smooth peaks replace named layer-band switches.  Human failures cluster
    # around interaction/causal structure and the final bridge, while every
    # layer retains non-zero value.
    causal = math.exp(
        -0.5
        * ((layer - curve.causal_layer_center) / curve.causal_layer_width) ** 2
    )
    bridge = math.exp(
        -0.5
        * ((layer - curve.bridge_layer_center) / curve.bridge_layer_width) ** 2
    )
    return (
        1.0
        + curve.causal_layer_amplitude * causal
        + curve.bridge_layer_amplitude * bridge
    )


def _attention_risk(
    ranks: dict[tuple[int, int], int],
    *,
    total_steps: int,
    curve: V24CurveProfile = V24_DEFAULT_CURVE_PROFILE,
) -> float:
    return sum(
        _phase_risk_weight(step, total_steps, curve)
        * _layer_risk_weight(layer, curve)
        * _ACTION_RISK[rank]
        for (step, layer), rank in ranks.items()
    )


def _trajectory_risk(
    actual: set[int],
    total_steps: int,
    curve: V24CurveProfile = V24_DEFAULT_CURVE_PROFILE,
) -> float:
    risk = 0.0
    run: list[int] = []

    def add_run(indices: list[int]) -> float:
        if not indices:
            return 0.0
        # Consecutive forecasts compound history error smoothly; there is no
        # scene label or hand-authored schedule case in this term.
        coupling = 1.0 + curve.forecast_run_coupling * (len(indices) - 1) ** 2
        return coupling * sum(
            _phase_risk_weight(step, total_steps, curve) for step in indices
        )

    for step in range(total_steps):
        if step in actual:
            risk += add_run(run)
            run = []
        else:
            run.append(step)
    return risk + add_run(run)


def _state_compute_units(
    *,
    total_steps: int,
    actual: set[int],
    ranks: dict[tuple[int, int], int],
) -> float:
    attention = sum(
        _ATTENTION_COST_RATIO[_RANK_TO_CANONICAL[rank]]
        for rank in ranks.values()
    )
    # Steps promoted from Forecast are deliberately Dense and therefore do not
    # appear in ``ranks``.
    promoted = len(actual) - len({step for step, _layer in ranks})
    return (
        len(actual) * _NON_ATTENTION_COMPUTE
        + _ATTENTION_COMPUTE * (attention / 50.0 + promoted)
        + (total_steps - len(actual)) * _FORECAST_COMPUTE
    )


@lru_cache(maxsize=512)
def _build_upgrade_chain(
    *,
    total_steps: int,
    lower: V24HumanAnchor,
    upper: V24HumanAnchor,
    mix: float,
    endpoint: tuple[int, ...],
    mandatory_actual: tuple[int, ...],
    forecast_risk_multiplier: float,
    curve: V24CurveProfile = V24_DEFAULT_CURVE_PROFILE,
) -> tuple[
    tuple[_UpgradeOperation, ...],
    tuple[tuple[int, int, int, str | None], ...],
    tuple[int, ...],
    str,
]:
    """Build one coupled, nested Pareto path from the evidence floor to Dense.

    Attention upgrades and Forecast-to-Actual promotions compete in the same
    marginal risk-reduction / measured-compute ledger.  The resulting order is
    independent of the requested acceleration, so every slower point is a
    strict quality-compute superset of every faster point.
    """

    ranks: dict[tuple[int, int], int] = {}
    prefixes: dict[tuple[int, int], str | None] = {}
    for step in endpoint:
        for layer in range(50):
            rank, prefix = _endpoint_cell(
                step=step,
                layer=layer,
                total_steps=total_steps,
                lower=lower,
                upper=upper,
                mix=mix,
            )
            ranks[(step, layer)] = rank
            prefixes[(step, layer)] = prefix
    initial_rows = tuple(
        (step, layer, rank, prefixes[(step, layer)])
        for (step, layer), rank in sorted(ranks.items())
    )
    actual = set(endpoint) | set(mandatory_actual)
    initial_actual = tuple(sorted(actual))
    operations: list[_UpgradeOperation] = []

    while len(actual) < total_steps or any(rank < 4 for rank in ranks.values()):
        candidates: list[_UpgradeOperation] = []
        for (step, layer), rank in ranks.items():
            if rank >= 4:
                continue
            next_rank = rank + 1
            candidates.append(_UpgradeOperation(
                kind="attention_fidelity",
                step=step,
                layer=layer,
                from_rank=rank,
                to_rank=next_rank,
                cost_delta=(
                    _ATTENTION_COMPUTE
                    / 50.0
                    * (
                        _ATTENTION_COST_RATIO[_RANK_TO_CANONICAL[next_rank]]
                        - _ATTENTION_COST_RATIO[_RANK_TO_CANONICAL[rank]]
                    )
                ),
                risk_reduction=(
                    _phase_risk_weight(step, total_steps, curve)
                    * _layer_risk_weight(layer, curve)
                    * (_ACTION_RISK[rank] - _ACTION_RISK[next_rank])
                ),
            ))
        current_trajectory_risk = _trajectory_risk(actual, total_steps, curve)
        for step in range(total_steps):
            if step in actual:
                continue
            promoted = set(actual)
            promoted.add(step)
            candidates.append(_UpgradeOperation(
                kind="forecast_to_actual",
                step=step,
                layer=-1,
                from_rank=-1,
                to_rank=4,
                cost_delta=1.0 - _FORECAST_COMPUTE,
                risk_reduction=(
                    current_trajectory_risk
                    - _trajectory_risk(promoted, total_steps, curve)
                ) * forecast_risk_multiplier,
            ))
        if not candidates:
            break
        selected = max(
            candidates,
            key=lambda item: (
                item.utility,
                item.risk_reduction,
                -item.cost_delta,
                item.kind == "forecast_to_actual",
                -item.step,
                -item.layer,
            ),
        )
        operations.append(selected)
        if selected.kind == "forecast_to_actual":
            actual.add(selected.step)
        else:
            ranks[(selected.step, selected.layer)] = selected.to_rank

    frozen_operations = tuple(operations)
    chain_digest = hashlib.sha256(json.dumps(
        {
            "curve_parameter_digest": curve.parameter_digest,
            "operations": [
                operation.audit_row() for operation in frozen_operations
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return frozen_operations, initial_rows, initial_actual, chain_digest


def _solve_nested_budget(
    *,
    total_steps: int,
    lower: V24HumanAnchor,
    upper: V24HumanAnchor,
    mix: float,
    endpoint: tuple[int, ...],
    mandatory_actual: tuple[int, ...],
    target_compute_units: float,
    forecast_risk_multiplier: float,
    curve: V24CurveProfile = V24_DEFAULT_CURVE_PROFILE,
) -> tuple[
    tuple[int, ...],
    dict[tuple[int, int], tuple[int, str | None]],
    dict[str, object],
]:
    operations, initial_rows, initial_actual, chain_digest = _build_upgrade_chain(
        total_steps=total_steps,
        lower=lower,
        upper=upper,
        mix=round(mix, 12),
        endpoint=endpoint,
        mandatory_actual=mandatory_actual,
        forecast_risk_multiplier=round(forecast_risk_multiplier, 12),
        curve=curve,
    )
    actual = set(initial_actual)
    state = {
        (step, layer): (rank, prefix)
        for step, layer, rank, prefix in initial_rows
    }

    current_cost = _state_compute_units(
        total_steps=total_steps,
        actual=actual,
        ranks={cell: rank for cell, (rank, _prefix) in state.items()},
    )

    applied: list[_UpgradeOperation] = []
    for operation in operations:
        proposed_cost = current_cost + operation.cost_delta
        if proposed_cost > target_compute_units + 1.0e-12:
            break
        if operation.kind == "forecast_to_actual":
            actual.add(operation.step)
        else:
            _rank, prefix = state[(operation.step, operation.layer)]
            state[(operation.step, operation.layer)] = (
                operation.to_rank,
                prefix if operation.to_rank < 4 else None,
            )
        applied.append(operation)
        current_cost = proposed_cost
    ranks = {cell: rank for cell, (rank, _prefix) in state.items()}
    total_risk = (
        _attention_risk(ranks, total_steps=total_steps, curve=curve)
        + forecast_risk_multiplier * _trajectory_risk(
            actual, total_steps, curve
        )
    )
    chain_counts = Counter(operation.kind for operation in operations)
    applied_counts = Counter(operation.kind for operation in applied)
    return tuple(sorted(actual)), state, {
        "optimizer_id": _V24_OPTIMIZER_ID,
        "curve_profile": curve.to_dict(),
        "objective": "maximize modeled Human-risk reduction per measured compute unit",
        "formal_scope": (
            "deterministic nested marginal allocation over Forecast-to-Actual "
            "and per-layer Attention upgrades; Human quality is an external gate"
        ),
        "global_human_optimality_claimed": False,
        "path_invariant": "single nested evidence-floor-to-Dense upgrade chain",
        "chain_digest": chain_digest,
        "chain_length": len(operations),
        "chain_upgrade_counts": dict(sorted(chain_counts.items())),
        "applied_upgrades": len(applied),
        "applied_upgrade_counts": dict(sorted(applied_counts.items())),
        "target_compute_units": target_compute_units,
        "achieved_compute_units": current_cost,
        "modeled_remaining_risk": total_risk,
        "next_upgrade": (
            None
            if len(applied) == len(operations)
            else operations[len(applied)].audit_row()
        ),
    }


def _execution_digest(
    *,
    total_steps: int,
    actual: tuple[int, ...],
    schedule: tuple[tuple[int, int, str], ...],
) -> str:
    """Reuse the historical physical-schedule identity for audit continuity."""

    complete_schedule = schedule or tuple(
        (step, layer, "dense")
        for step in actual
        for layer in range(50)
    )
    blueprint = blueprint_from_runtime_schedule(
        candidate_id="v24_runtime_selection",
        total_steps=total_steps,
        actual_step_indices=actual,
        attention_action_schedule=complete_schedule,
        source="v24_deployment",
    )
    return v19_blueprint_execution_digest(blueprint)


def _dense_selection(
    workload: V19WorkloadContext,
    *,
    acceleration: float,
    reason: str,
    video_tokens: int | None,
) -> V24DeploymentSelection:
    assert workload.steps is not None
    actual = tuple(range(int(workload.steps)))
    execution_digest = _execution_digest(
        total_steps=int(workload.steps),
        actual=actual,
        schedule=(),
    )
    return V24DeploymentSelection(
        actual_step_indices=actual,
        attention_action_schedule=(),
        summary={
            "schema_version": V24_DEPLOYMENT_SCHEMA,
            "policy_id": V24_DEPLOYMENT_POLICY_ID,
            "human_surface_digest": V24_HUMAN_SURFACE_DIGEST,
            "generation_request_digest": workload.digest,
            "execution_digest": execution_digest,
            "accelerated": False,
            "reason": reason,
            "acceleration": acceleration,
            "prompt_semantics_used": False,
            "video_tokens": video_tokens,
            "packed_tokens": workload.packed_tokens,
            "actual_step_indices": list(actual),
            "forecast_steps": 0,
            "estimated_compute_units": float(workload.steps),
            "estimated_compute_ratio": 1.0,
            "technique_mix": {
                "actual_dit_evaluations": len(actual),
                "forecast_evaluations": 0,
                "actual_attention_cells": {"dense": len(actual) * 50},
                "forecast_anchor_attention_cells": {},
                "coupled_techniques": ["exact_runtime"],
            },
            "runtime_feedback": {
                "policy_id": None,
                "mode": "disabled_dense_trajectory",
                "adds_teacher_evaluations": False,
                "max_runtime_promotions": 0,
            },
        },
    )


class V24ParetoRuntimeSelector:
    """Compile one deterministic deployment plan after exact tokenisation."""

    policy_id = V24_DEPLOYMENT_POLICY_ID

    def __init__(
        self,
        *,
        curve: V24CurveProfile = V24_DEFAULT_CURVE_PROFILE,
    ) -> None:
        if not isinstance(curve, V24CurveProfile):
            raise V19PlanningError("V24 selector requires a curve profile")
        self.curve = curve

    def select(
        self,
        *,
        workload: V19WorkloadContext,
        acceleration: float,
        required_actual_step_indices: tuple[int, ...] = (),
    ) -> V24DeploymentSelection:
        if workload.steps is None:
            raise V19PlanningError("V24 selection requires total steps")
        total_steps = int(workload.steps)
        if not 4 <= total_steps <= 30:
            raise V19PlanningError("V24 total steps must lie inside [4, 30]")
        try:
            acceleration = float(acceleration)
        except (TypeError, ValueError) as error:
            raise V19PlanningError("V24 acceleration must be numeric") from error
        if not math.isfinite(acceleration) or not 0.0 <= acceleration <= 100.0:
            raise V19PlanningError("V24 acceleration must lie inside [0, 100]")
        acceleration = round(acceleration, 1)
        required = tuple(sorted(set(int(step) for step in required_actual_step_indices)))
        if required != required_actual_step_indices or any(
            step < 0 or step >= total_steps for step in required
        ):
            raise V19PlanningError("V24 required Actual steps are invalid")
        try:
            video_tokens = _video_tokens(workload)
        except V19PlanningError:
            return _dense_selection(
                workload,
                acceleration=acceleration,
                reason="geometry_unavailable_dense_fallback",
                video_tokens=None,
            )
        if acceleration == 0.0:
            return _dense_selection(
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
            return _dense_selection(
                workload,
                acceleration=acceleration,
                reason="unsupported_runtime_identity_dense_fallback",
                video_tokens=video_tokens,
            )
        if workload.packed_tokens > V24_MAX_PACKED_TOKENS:
            return _dense_selection(
                workload,
                acceleration=acceleration,
                reason="packed_token_envelope_exceeded_dense_fallback",
                video_tokens=video_tokens,
            )

        lower, upper, mix = _anchor_bracket(video_tokens)
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

        # Human-reviewed anchors define the fastest admitted evidence floor;
        # they are not runtime branches.  Everything between that floor and
        # Dense is selected by one joint continuous-budget optimizer.
        mandatory_actual = set(required)
        if workload.reference_videos:
            mandatory_actual.update(range(total_steps))
        mandatory = tuple(sorted(mandatory_actual))

        # Resource pressure changes the relative price of trajectory error and
        # Attention approximation continuously.  The ratio is independent of
        # the requested acceleration, preserving one nested path for every
        # point on a fixed workload surface.
        forecast_risk_multiplier = (
            self.curve.forecast_risk_scale
            if attention_progress <= 1.0e-12
            else self.curve.forecast_risk_scale * (
                1.0 + 3.0 * max(
                    0.0,
                    1.0 - trajectory_progress / attention_progress,
                )
            )
        )
        _operations, initial_rows, initial_actual, _chain_digest = _build_upgrade_chain(
            total_steps=total_steps,
            lower=lower,
            upper=upper,
            mix=round(mix, 12),
            endpoint=endpoint,
            mandatory_actual=mandatory,
            forecast_risk_multiplier=round(forecast_risk_multiplier, 12),
            curve=self.curve,
        )
        floor_cost = _state_compute_units(
            total_steps=total_steps,
            actual=set(initial_actual),
            ranks={
                (step, layer): rank
                for step, layer, rank, _prefix in initial_rows
            },
        )
        dense_cost = float(total_steps)
        budget_progress = attention_progress
        target_compute_units = max(
            floor_cost,
            min(
                dense_cost,
                dense_cost
                - (dense_cost - floor_cost)
                * budget_progress ** self.curve.budget_exponent,
            ),
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
                schedule.append((step, layer, _runtime_action(rank, prefix)))
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
        if not math.isclose(
            compute_units,
            float(optimizer["achieved_compute_units"]),
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ):
            raise V19PlanningError(
                "V24 optimizer and compiled physical schedule disagree"
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
        exact_anchor = (
            total_steps == 20
            and acceleration == 100.0
            and not guards
            and lower.anchor_id == upper.anchor_id
            and not required
        )
        execution_hint = (
            "v22_medium_byte_exact_helpers"
            if exact_anchor and lower.anchor_id == "medium_66k_v022_causal"
            else "v18_xlong_byte_exact_helpers"
            if video_tokens >= 200_000
            else None
        )
        coupled = ["exact_runtime"]
        if len(actual) < total_steps:
            coupled.append("directional_forecast")
        if any(action != "dense" for action in actual_actions):
            coupled.append("block_sparse_attention")
        if execution_hint is not None:
            coupled.append("byte_exact_execution_helpers")
        return V24DeploymentSelection(
            actual_step_indices=actual,
            attention_action_schedule=physical,
            summary={
                "schema_version": V24_DEPLOYMENT_SCHEMA,
                "policy_id": V24_DEPLOYMENT_POLICY_ID,
                "human_surface_digest": V24_HUMAN_SURFACE_DIGEST,
                "generation_request_digest": workload.digest,
                "execution_digest": execution_digest,
                "accelerated": not all_dense,
                "reason": (
                    "human_anchor_endpoint"
                    if exact_anchor
                    else "continuous_joint_cost_risk_optimization"
                ),
                "acceleration": acceleration,
                "requested_progress": requested_progress,
                "trajectory_progress": trajectory_progress,
                "attention_progress": attention_progress,
                "compute_budget_progress": budget_progress,
                "curve_profile": self.curve.to_dict(),
                "prompt_semantics_used": False,
                "video_tokens": video_tokens,
                "packed_tokens": workload.packed_tokens,
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
                    "artifact_sha256": list(dict.fromkeys((
                        *lower.artifact_sha256,
                        *upper.artifact_sha256,
                    ))),
                    "direct_human_anchor": exact_anchor,
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
                "endpoint_actual_step_indices": list(endpoint),
                "actual_step_indices": list(actual),
                "forecast_steps": total_steps - len(actual),
                "maximum_forecast_run": len(
                    _longest_forecast_run(actual_set, total_steps)
                ),
                "estimated_compute_units": compute_units,
                "estimated_compute_ratio": compute_units / total_steps,
                "optimizer": optimizer,
                "execution_profile_hint": execution_hint,
                "runtime_feedback": {
                    "policy_id": (
                        _V24_FORECAST_FEEDBACK_POLICY_ID
                        if len(actual) < total_steps
                        else None
                    ),
                    "mode": (
                        "observe_only"
                        if len(actual) < total_steps
                        else "disabled_dense_trajectory"
                    ),
                    "signal": (
                        "request-local secant-tail error "
                        "at already-computed Actual corrections"
                    ),
                    "adds_teacher_evaluations": False,
                    "max_runtime_promotions": 0,
                },
                "technique_mix": {
                    "actual_dit_evaluations": len(actual),
                    "forecast_evaluations": total_steps - len(actual),
                    "actual_attention_cells": dict(sorted(actual_actions.items())),
                    "forecast_anchor_attention_cells": dict(
                        sorted(forecast_actions.items())
                    ),
                    "coupled_techniques": coupled,
                },
            },
        )


__all__ = [
    "V24_CURVE_PROFILE_SCHEMA",
    "V24_DEFAULT_CURVE_PROFILE",
    "V24_DEPLOYMENT_POLICY_ID",
    "V24_DEPLOYMENT_SCHEMA",
    "V24_HUMAN_ANCHORS",
    "V24_HUMAN_SURFACE_DIGEST",
    "V24_MAX_PACKED_TOKENS",
    "V24_HUMAN_EVIDENCE_MAX_PACKED_TOKENS",
    "V24DeploymentSelection",
    "V24CurveProfile",
    "V24HumanAnchor",
    "V24ParetoRuntimeSelector",
]
