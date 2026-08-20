"""Pure, auditable allocation rules for bounded online Attention guards.

The runtime verifier lives in :mod:`h3serve.native_engine.model.kernels`, but
the question "how much may observation spend?" is a planner property.  Keeping
that rule here lets the evaluator replay it without importing CUDA/Torch code
and prevents a runtime-only heuristic from silently consuming the recovery
reserve certified by the scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path


ROUND220_PROBE_LAYERS = (24, 44, 4)
ROUND220_PHASE_TARGETS = (0.15, 0.45, 0.75)
ROUND220_MAX_PROBES = 9
ROUND220_HIGH_RESERVE_REPAIR_FLOOR = 5.0
ROUND221_CALIBRATION_SCHEMA = "h3_round221_probe_growth_calibration_v1"
ROUND221_CALIBRATION_FILE = (
    Path(__file__).with_name("evidence")
    / "round221_probe_growth_calibration_v1.json"
)


@dataclass(frozen=True, slots=True)
class PhaseSentinelAllocation:
    """A finite observation schedule plus its conservative repair balance."""

    slots: tuple[tuple[int, int], ...]
    limit_dense_layers: float
    observation_dense_layers: float
    remaining_dense_layers: float
    required_remaining_dense_layers: float

    @property
    def budget_respected(self) -> bool:
        return (
            self.observation_dense_layers <= self.limit_dense_layers + 1.0e-9
            and self.remaining_dense_layers + 1.0e-9
            >= self.required_remaining_dense_layers
        )


@dataclass(frozen=True, slots=True)
class ProbeGrowthCalibration:
    """Validated, digest-addressed null envelope for phase-probe growth."""

    source_policy_id: str
    source_online_guard_id: str
    probe_domain: str
    task_count: int
    observed_max_task_score: float
    safety_margin: float
    runtime_growth_threshold: float
    marginal_coverage_lower_bound: float
    task_scores: tuple[tuple[str, float], ...]
    evidence_sha256: str


@dataclass(frozen=True, slots=True)
class PhaseGrowthObservation:
    """One request-relative observation; a trigger only authorizes an upgrade."""

    baseline_relative_rms: float
    growth_ratio: float
    triggered: bool


class CalibratedPhaseGrowthGuard:
    """Track later/first phase growth independently for each probed layer.

    Absolute sampled error changes materially with shape.  This guard instead
    treats the first scheduled observation of each layer as a request-local
    baseline.  State is reset by *object identity* when a new request ledger
    arrives, so hot-session reuse cannot leak one scene's scale into another.
    """

    def __init__(self, threshold: float) -> None:
        if not math.isfinite(threshold) or threshold <= 1.0:
            raise ValueError("phase-growth threshold must be finite and above one")
        self.threshold = float(threshold)
        self._request_token: object | None = None
        self._baselines: dict[int, float] = {}

    def begin_request(self, request_token: object) -> bool:
        if request_token is None:
            raise ValueError("phase-growth request token cannot be None")
        if request_token is self._request_token:
            return False
        self._request_token = request_token
        self._baselines.clear()
        return True

    def has_request(self, request_token: object) -> bool:
        """Return whether ``request_token`` owns the current detector state."""

        return request_token is not None and request_token is self._request_token

    def observe(self, layer: int, relative_rms: float) -> PhaseGrowthObservation:
        if self._request_token is None:
            raise RuntimeError("begin_request must be called before observe")
        if layer < 0:
            raise ValueError("phase-growth layer cannot be negative")
        if not math.isfinite(relative_rms) or relative_rms < 0.0:
            raise ValueError("phase-growth RMS must be finite and non-negative")
        baseline = self._baselines.setdefault(int(layer), float(relative_rms))
        denominator = max(baseline, 1.0e-12)
        ratio = float(relative_rms) / denominator
        return PhaseGrowthObservation(
            baseline_relative_rms=baseline,
            growth_ratio=ratio,
            triggered=ratio > self.threshold,
        )

    def checkpoint_state(self, request_token: object) -> dict[str, object]:
        """Return the request-local detector state for exact sampler resume."""

        if request_token is not self._request_token:
            raise ValueError("phase-growth checkpoint token is not the active request")
        return {
            "schema_version": "h3_phase_growth_checkpoint_v1",
            "threshold": self.threshold,
            "baselines": {
                str(layer): value for layer, value in sorted(self._baselines.items())
            },
        }

    def restore_checkpoint_state(
        self,
        request_token: object,
        state: object,
    ) -> None:
        """Restore one detector state and bind it to the new request ledger."""

        if request_token is None:
            raise ValueError("phase-growth restore token cannot be None")
        if not isinstance(state, dict):
            raise ValueError("phase-growth checkpoint state must be an object")
        if state.get("schema_version") != "h3_phase_growth_checkpoint_v1":
            raise ValueError("unexpected phase-growth checkpoint schema")
        recorded_threshold = float(state.get("threshold", math.nan))
        if not math.isclose(
            recorded_threshold,
            self.threshold,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("phase-growth checkpoint threshold mismatch")
        raw_baselines = state.get("baselines")
        if not isinstance(raw_baselines, dict):
            raise ValueError("phase-growth checkpoint baselines are missing")
        baselines: dict[int, float] = {}
        for raw_layer, raw_value in raw_baselines.items():
            try:
                layer = int(raw_layer)
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid phase-growth checkpoint baseline") from exc
            if str(layer) != str(raw_layer) or layer < 0:
                raise ValueError("invalid phase-growth checkpoint layer")
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("invalid phase-growth checkpoint value")
            baselines[layer] = value
        self._request_token = request_token
        self._baselines = baselines


def load_probe_growth_calibration(
    path: Path = ROUND221_CALIBRATION_FILE,
) -> ProbeGrowthCalibration:
    """Load and fail closed on a malformed or internally inconsistent file."""

    payload = path.read_bytes()
    document = json.loads(payload)
    if document.get("schema_version") != ROUND221_CALIBRATION_SCHEMA:
        raise ValueError("unexpected phase-growth calibration schema")
    records = document.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("phase-growth calibration records are missing")
    scores: list[tuple[str, float]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("phase-growth calibration record must be an object")
        record_id = str(record.get("id", ""))
        score = float(record.get("task_score", math.nan))
        if not record_id or not math.isfinite(score) or score <= 0.0:
            raise ValueError("invalid phase-growth calibration task score")
        scores.append((record_id, score))
    if len({record_id for record_id, _ in scores}) != len(scores):
        raise ValueError("duplicate phase-growth calibration task id")
    task_count = int(document.get("calibration_tasks", -1))
    if task_count != len(scores):
        raise ValueError("phase-growth calibration task count mismatch")
    observed_max = float(document.get("observed_max_task_score", math.nan))
    safety_margin = float(document.get("multiplicative_safety_margin", math.nan))
    threshold = float(document.get("runtime_growth_threshold", math.nan))
    coverage = float(
        document.get("exchangeable_next_task_marginal_coverage_lower_bound", math.nan)
    )
    if not math.isclose(observed_max, max(score for _, score in scores), abs_tol=1e-12):
        raise ValueError("phase-growth observed maximum mismatch")
    if not math.isfinite(safety_margin) or safety_margin < 1.0:
        raise ValueError("phase-growth safety margin must be at least one")
    if not math.isclose(threshold, observed_max * safety_margin, abs_tol=1e-12):
        raise ValueError("phase-growth runtime threshold mismatch")
    expected_coverage = task_count / (task_count + 1.0)
    if not math.isclose(coverage, expected_coverage, abs_tol=1e-12):
        raise ValueError("phase-growth order-statistic coverage mismatch")
    return ProbeGrowthCalibration(
        source_policy_id=str(document.get("source_policy_id", "")),
        source_online_guard_id=str(document.get("source_online_guard_id", "")),
        probe_domain=str(document.get("probe_domain", "")),
        task_count=task_count,
        observed_max_task_score=observed_max,
        safety_margin=safety_margin,
        runtime_growth_threshold=threshold,
        marginal_coverage_lower_bound=coverage,
        task_scores=tuple(scores),
        evidence_sha256=hashlib.sha256(payload).hexdigest(),
    )


ROUND221_PROBE_GROWTH_CALIBRATION = load_probe_growth_calibration()
ROUND221_RUNTIME_GROWTH_THRESHOLD = (
    ROUND221_PROBE_GROWTH_CALIBRATION.runtime_growth_threshold
)


def allocate_phase_sentinels(
    actual_steps: tuple[int, ...],
    limit_dense_layers: float,
) -> PhaseSentinelAllocation:
    """Allocate at most three phase × three layer probes, leaving repair room.

    A sampled verifier is conservatively charged as one complete Dense
    Attention layer.  Reserves of six or more layer-equivalents always retain
    five for intervention.  Smaller reserves receive only one observation and
    retain every remaining unit; this is a structural guarantee, not a claim
    that the remainder is sufficient for every possible repair horizon.
    """

    if not math.isfinite(limit_dense_layers) or limit_dense_layers < 0.0:
        raise ValueError("online Dense-layer limit must be finite and non-negative")
    if not actual_steps or limit_dense_layers < 1.0:
        return PhaseSentinelAllocation(
            slots=(),
            limit_dense_layers=float(limit_dense_layers),
            observation_dense_layers=0.0,
            remaining_dense_layers=float(limit_dense_layers),
            required_remaining_dense_layers=0.0,
        )
    canonical_steps = tuple(int(step) for step in actual_steps)
    if canonical_steps != tuple(sorted(set(canonical_steps))) or canonical_steps[0] < 0:
        raise ValueError("actual steps must be sorted, unique, and non-negative")

    last_step = max(canonical_steps[-1], 1)
    phase_steps: list[int] = []
    for fraction in ROUND220_PHASE_TARGETS:
        target = fraction * last_step
        selected = min(
            canonical_steps,
            key=lambda step: (abs(step - target), step),
        )
        if selected not in phase_steps:
            phase_steps.append(selected)
    middle_first = tuple(
        phase_steps[index]
        for index in (1, 2, 0)
        if index < len(phase_steps)
    )

    if limit_dense_layers < 6.0:
        maximum_probes = 1
        required_remaining = max(0.0, limit_dense_layers - 1.0)
    else:
        maximum_probes = min(
            ROUND220_MAX_PROBES,
            max(
                1,
                int(math.floor(
                    limit_dense_layers - ROUND220_HIGH_RESERVE_REPAIR_FLOOR
                )),
            ),
        )
        required_remaining = ROUND220_HIGH_RESERVE_REPAIR_FLOOR
    candidates = tuple(
        (step, layer)
        for layer in ROUND220_PROBE_LAYERS
        for step in middle_first
    )
    slots = candidates[:maximum_probes]
    observation = float(len(slots))
    remaining = max(0.0, float(limit_dense_layers) - observation)
    return PhaseSentinelAllocation(
        slots=slots,
        limit_dense_layers=float(limit_dense_layers),
        observation_dense_layers=observation,
        remaining_dense_layers=remaining,
        required_remaining_dense_layers=required_remaining,
    )


__all__ = [
    "CalibratedPhaseGrowthGuard",
    "PhaseSentinelAllocation",
    "PhaseGrowthObservation",
    "ProbeGrowthCalibration",
    "ROUND220_HIGH_RESERVE_REPAIR_FLOOR",
    "ROUND220_MAX_PROBES",
    "ROUND220_PHASE_TARGETS",
    "ROUND220_PROBE_LAYERS",
    "ROUND221_CALIBRATION_FILE",
    "ROUND221_CALIBRATION_SCHEMA",
    "ROUND221_PROBE_GROWTH_CALIBRATION",
    "ROUND221_RUNTIME_GROWTH_THRESHOLD",
    "allocate_phase_sentinels",
    "load_probe_growth_calibration",
]
