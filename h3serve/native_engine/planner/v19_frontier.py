"""Release-facing V19 frontier over complete, evidence-bound trajectories.

The cell optimizer answers a deliberately narrow question: how should a fixed
actual-step trajectory spend its Attention budget?  The creator-facing speed
dial has a different job.  It must choose between *complete* trajectories
(Dense/actual steps, sparse actions and forecast composites together) without
pretending that a tensor-distance proxy is Human quality.

This module therefore admits only candidates which already have both repeated
end-to-end timing and plan-bound Human-risk calibration.  Candidates from
different actual/forecast schedules share a generation-request identity which
excludes ``actual_step_indices``; that schedule is an optimizer decision, not
part of the user's request.  Out-of-distribution requests fail closed to the
ordinary Dense service path instead of losing any model capability.

One public scalar can only represent a totally ordered quality/speed chain.
If two plans exchange risk between dimensions (for example better audio but
worse contact causality), V19 refuses to hide that trade-off in a weighted
score.  Such candidates remain research evidence but cannot both back the
same one-dimensional release dial.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math

from .action_registry import ActionRegistry, ActionRegistryError
from .v19_candidates import V19MaterializedCandidate
from .v19_contracts import V19HumanRiskVector
from .v19_planner import (
    V19ActionUse,
    V19CandidatePlan,
    V19ForecastUse,
    V19PlanCertificate,
    V19PlanningError,
    V19PlanningRequest,
    V19WorkloadContext,
    verify_v19_plan_certificate,
)


V19_RELEASE_FRONTIER_SCHEMA = "h3_v19_release_frontier_v1"
V19_ENVELOPE_CERTIFICATE_SCHEMA = "h3_v19_envelope_certificate_v1"


def _sha256(document: object) -> str:
    return hashlib.sha256(json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class V19GenerationRequestKey:
    """Exact request identity used for telemetry, not evidence extrapolation."""

    model_variant: str
    service_family: str
    packed_tokens: int
    condition_count: int
    reference_images: int
    reference_audio: int
    reference_videos: int
    device_arch: str
    width: int
    height: int
    frames: int
    steps: int
    sampler: str
    scheduler: str

    @classmethod
    def from_workload(
        cls, workload: V19WorkloadContext
    ) -> "V19GenerationRequestKey":
        if any(value is None for value in (
            workload.width,
            workload.height,
            workload.frames,
            workload.steps,
        )):
            raise V19PlanningError(
                "V19 release selection requires exact generation geometry"
            )
        return cls(
            model_variant=workload.model_variant,
            service_family=workload.service_family,
            packed_tokens=workload.packed_tokens,
            condition_count=workload.condition_count,
            reference_images=workload.reference_images,
            reference_audio=workload.reference_audio,
            reference_videos=workload.reference_videos,
            device_arch=workload.device_arch,
            width=int(workload.width),
            height=int(workload.height),
            frames=int(workload.frames),
            steps=int(workload.steps),
            sampler=workload.sampler,
            scheduler=workload.scheduler,
        )

    @property
    def digest(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True, slots=True, order=True)
class V19ReferenceProfile:
    """One explicitly calibrated packed-layout/input combination."""

    condition_count: int = 0
    images: int = 0
    audio: int = 0
    videos: int = 0

    def __post_init__(self) -> None:
        if min(self.condition_count, self.images, self.audio, self.videos) < 0:
            raise V19PlanningError("V19 reference counts cannot be negative")

    @classmethod
    def from_workload(cls, workload: V19WorkloadContext) -> "V19ReferenceProfile":
        return cls(
            condition_count=workload.condition_count,
            images=workload.reference_images,
            audio=workload.reference_audio,
            videos=workload.reference_videos,
        )


@dataclass(frozen=True, slots=True)
class V19WorkloadEnvelope:
    """A closed release scope supported by exact endpoint evidence.

    Prompt semantics are represented by the Human review cases in the source
    risk artifacts.  ``packed_tokens`` is only the numerical sequence-length
    axis; it may be interpolated inside an explicitly observed closed interval
    but is never silently extrapolated.  Reference combinations and condition
    counts are enumerated because their packed layouts are not interchangeable.
    """

    envelope_id: str
    model_variant: str
    service_family: str
    device_arch: str
    width: int
    height: int
    frames: int
    steps: int
    sampler: str
    scheduler: str
    min_packed_tokens: int
    max_packed_tokens: int
    reference_profiles: tuple[V19ReferenceProfile, ...]

    def __post_init__(self) -> None:
        if not self.envelope_id or self.envelope_id.strip() != self.envelope_id:
            raise V19PlanningError("V19 envelope requires a stable id")
        if any(value <= 0 for value in (
            self.width,
            self.height,
            self.frames,
            self.steps,
            self.min_packed_tokens,
            self.max_packed_tokens,
        )):
            raise V19PlanningError("V19 envelope dimensions must be positive")
        if self.min_packed_tokens > self.max_packed_tokens:
            raise V19PlanningError("V19 packed-token envelope is inverted")
        if (
            not self.reference_profiles
            or tuple(sorted(set(self.reference_profiles))) != self.reference_profiles
        ):
            raise V19PlanningError(
                "V19 envelope reference profiles must be sorted and unique"
            )

    @property
    def digest(self) -> str:
        return _sha256(asdict(self))

    def contains(self, workload: V19WorkloadContext) -> bool:
        geometry = (
            workload.model_variant,
            workload.service_family,
            workload.device_arch,
            workload.width,
            workload.height,
            workload.frames,
            workload.steps,
            workload.sampler,
            workload.scheduler,
        )
        expected = (
            self.model_variant,
            self.service_family,
            self.device_arch,
            self.width,
            self.height,
            self.frames,
            self.steps,
            self.sampler,
            self.scheduler,
        )
        return (
            geometry == expected
            and self.min_packed_tokens
            <= workload.packed_tokens
            <= self.max_packed_tokens
            and V19ReferenceProfile.from_workload(workload)
            in self.reference_profiles
        )

    def is_subset_of(self, other: "V19WorkloadEnvelope") -> bool:
        same_geometry = (
            self.model_variant,
            self.service_family,
            self.device_arch,
            self.width,
            self.height,
            self.frames,
            self.steps,
            self.sampler,
            self.scheduler,
        ) == (
            other.model_variant,
            other.service_family,
            other.device_arch,
            other.width,
            other.height,
            other.frames,
            other.steps,
            other.sampler,
            other.scheduler,
        )
        return (
            same_geometry
            and self.min_packed_tokens >= other.min_packed_tokens
            and self.max_packed_tokens <= other.max_packed_tokens
            and set(self.reference_profiles).issubset(other.reference_profiles)
        )


@dataclass(frozen=True, slots=True)
class V19CertifiedFrontierEntry:
    """One exact-workload source certificate used to build an envelope."""

    planning_request: V19PlanningRequest
    materialized: V19MaterializedCandidate
    certificate: V19PlanCertificate

    @property
    def candidate(self) -> V19CandidatePlan:
        return self.materialized.candidate

    @property
    def generation_key(self) -> V19GenerationRequestKey:
        return V19GenerationRequestKey.from_workload(
            self.planning_request.workload
        )

    @property
    def runtime_digest(self) -> str:
        runtime = self.planning_request.runtime
        if runtime is None:
            raise V19PlanningError(
                "V19 release candidate is not bound to a physical runtime"
            )
        return runtime.digest


@dataclass(frozen=True, slots=True)
class V19EnvelopeCertificate:
    registry_digest: str
    envelope_digest: str
    runtime_digest: str
    execution_digest: str
    candidate_digest: str
    source_certificate_digests: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    certificate_digest: str
    schema_version: str = V19_ENVELOPE_CERTIFICATE_SCHEMA

    @classmethod
    def issue(
        cls,
        registry: ActionRegistry,
        envelope: V19WorkloadEnvelope,
        candidate: V19CandidatePlan,
        sources: tuple[V19CertifiedFrontierEntry, ...],
    ) -> "V19EnvelopeCertificate":
        if not sources:
            raise V19PlanningError("V19 envelope certificate requires sources")
        if len({
            source.certificate.certificate_digest for source in sources
        }) != len(sources):
            raise V19PlanningError("V19 envelope contains duplicate sources")
        runtime_digests = {source.runtime_digest for source in sources}
        if len(runtime_digests) != 1:
            raise V19PlanningError("V19 envelope sources span physical runtimes")
        base = cls(
            registry_digest=registry.digest,
            envelope_digest=envelope.digest,
            runtime_digest=next(iter(runtime_digests)),
            execution_digest=candidate.execution_digest,
            candidate_digest=candidate.digest,
            source_certificate_digests=tuple(sorted(
                source.certificate.certificate_digest for source in sources
            )),
            evidence_ids=tuple(sorted(candidate.evidence_ids)),
            certificate_digest="",
        )
        return replace(base, certificate_digest=_envelope_certificate_digest(base))


def _envelope_certificate_digest(certificate: V19EnvelopeCertificate) -> str:
    document = asdict(certificate)
    document["certificate_digest"] = ""
    return _sha256(document)


@dataclass(frozen=True, slots=True)
class V19CertifiedEnvelopeEntry:
    envelope: V19WorkloadEnvelope
    materialized: V19MaterializedCandidate
    sources: tuple[V19CertifiedFrontierEntry, ...]
    certificate: V19EnvelopeCertificate

    @property
    def candidate(self) -> V19CandidatePlan:
        return self.materialized.candidate

    @property
    def runtime_digest(self) -> str:
        return self.certificate.runtime_digest


def _validate_exact_source(
    registry: ActionRegistry,
    source: V19CertifiedFrontierEntry,
) -> None:
    if not source.materialized.end_to_end_cost_calibrated:
        raise V19PlanningError(
            "V19 envelope source requires repeated end-to-end timing"
        )
    if not source.materialized.human_risk_calibrated:
        raise V19PlanningError(
            "V19 envelope source requires plan-bound Human risk"
        )
    verification = verify_v19_plan_certificate(
        registry,
        source.planning_request,
        source.candidate,
        source.certificate,
    )
    if not verification.valid:
        raise V19PlanningError(
            "invalid V19 source certificate: " + "; ".join(verification.reasons)
        )
    workload = source.planning_request.workload
    if workload.steps is None or not workload.actual_step_indices:
        raise V19PlanningError("V19 exact source lacks a complete step trajectory")
    expected_attention = {
        (step, layer)
        for step in workload.actual_step_indices
        for layer in range(50)
    }
    actual_attention = [
        (step, layer)
        for use in source.candidate.action_uses
        if isinstance(use, V19ActionUse)
        for step in use.step_indices
        for layer in range(use.layer_start, use.layer_stop)
    ]
    if (
        len(set(actual_attention)) != len(actual_attention)
        or set(actual_attention) != expected_attention
    ):
        raise V19PlanningError(
            "V19 exact source does not cover every actual Attention cell once"
        )
    expected_forecast = set(range(workload.steps)) - set(
        workload.actual_step_indices
    )
    actual_forecast = [
        step
        for use in source.candidate.action_uses
        if isinstance(use, V19ForecastUse)
        for step in use.step_indices
    ]
    if (
        len(set(actual_forecast)) != len(actual_forecast)
        or set(actual_forecast) != expected_forecast
    ):
        raise V19PlanningError(
            "V19 exact source does not cover every forecast step once"
        )


def build_v19_certified_envelope_entry(
    registry: ActionRegistry,
    *,
    envelope: V19WorkloadEnvelope,
    sources: tuple[V19CertifiedFrontierEntry, ...],
) -> V19CertifiedEnvelopeEntry:
    """Aggregate exact sources conservatively into one release envelope."""

    if not sources:
        raise V19PlanningError("V19 envelope cannot be built without sources")
    for source in sources:
        _validate_exact_source(registry, source)
        if not envelope.contains(source.planning_request.workload):
            raise V19PlanningError("V19 envelope does not contain a source workload")
    executions = {source.candidate.execution_digest for source in sources}
    runtimes = {source.runtime_digest for source in sources}
    if len(executions) != 1:
        raise V19PlanningError("V19 envelope sources use different schedules")
    if len(runtimes) != 1:
        raise V19PlanningError("V19 envelope sources use different runtimes")
    observed_tokens = {
        source.planning_request.workload.packed_tokens for source in sources
    }
    if envelope.min_packed_tokens not in observed_tokens:
        raise V19PlanningError("V19 envelope minimum token endpoint is unobserved")
    if envelope.max_packed_tokens not in observed_tokens:
        raise V19PlanningError("V19 envelope maximum token endpoint is unobserved")
    observed_references = {
        V19ReferenceProfile.from_workload(source.planning_request.workload)
        for source in sources
    }
    if not set(envelope.reference_profiles).issubset(observed_references):
        raise V19PlanningError("V19 envelope contains an unobserved reference profile")
    template = sources[0].candidate
    for use in template.action_uses:
        try:
            action = registry.resolve(use.action_id)
        except ActionRegistryError as error:
            raise V19PlanningError(str(error)) from error
        for profile in envelope.reference_profiles:
            for packed_tokens in (
                envelope.min_packed_tokens,
                envelope.max_packed_tokens,
            ):
                if not action.envelope.contains(
                    model_variant=envelope.model_variant,
                    service_family=envelope.service_family,
                    packed_tokens=packed_tokens,
                    condition_count=profile.condition_count,
                    device_arch=envelope.device_arch,
                ):
                    raise V19PlanningError(
                        "V19 release envelope exceeds an action workload envelope: "
                        + use.action_id
                    )
    risks = tuple(source.candidate.risk_ucb.as_tuple() for source in sources)
    evidence_ids = tuple(sorted({
        evidence_id
        for source in sources
        for evidence_id in source.candidate.evidence_ids
    }))
    candidate = replace(
        template,
        candidate_id=(
            f"{template.candidate_id}:envelope:{envelope.digest[:12]}"
        ),
        predicted_cost_p50_ms=max(
            source.candidate.predicted_cost_p50_ms for source in sources
        ),
        predicted_cost_p90_ms=max(
            source.candidate.predicted_cost_p90_ms for source in sources
        ),
        predicted_peak_vram_gib=max(
            source.candidate.predicted_peak_vram_gib for source in sources
        ),
        risk_ucb=V19HumanRiskVector(*(
            max(values[index] for values in risks) for index in range(7)
        )),
        evidence_ids=evidence_ids,
        source="v19_certified_workload_envelope",
    )
    materialized = V19MaterializedCandidate(
        candidate=candidate,
        end_to_end_cost_calibrated=True,
        human_risk_calibrated=True,
    )
    certificate = V19EnvelopeCertificate.issue(
        registry,
        envelope,
        candidate,
        sources,
    )
    return V19CertifiedEnvelopeEntry(
        envelope=envelope,
        materialized=materialized,
        sources=sources,
        certificate=certificate,
    )


def verify_v19_envelope_entry(
    registry: ActionRegistry,
    entry: V19CertifiedEnvelopeEntry,
) -> tuple[str, ...]:
    reasons: list[str] = []
    for source in entry.sources:
        try:
            _validate_exact_source(registry, source)
        except V19PlanningError as error:
            reasons.append(str(error))
    try:
        rebuilt = build_v19_certified_envelope_entry(
            registry,
            envelope=entry.envelope,
            sources=entry.sources,
        )
    except V19PlanningError as error:
        reasons.append(str(error))
        return tuple(reasons)
    if rebuilt.materialized != entry.materialized:
        reasons.append("V19 envelope conservative aggregate mismatch")
    if rebuilt.certificate != entry.certificate:
        reasons.append("V19 envelope certificate mismatch")
    if entry.certificate.certificate_digest != _envelope_certificate_digest(
        entry.certificate
    ):
        reasons.append("V19 envelope certificate digest mismatch")
    return tuple(reasons)


@dataclass(frozen=True, slots=True)
class V19AccelerationDecision:
    accelerated: bool
    reason: str
    generation_request_digest: str
    runtime_digest: str
    acceleration: float
    target_cost_p90_ms: float | None = None
    candidate: V19CandidatePlan | None = None
    certificate: V19EnvelopeCertificate | None = None
    schema_version: str = V19_RELEASE_FRONTIER_SCHEMA

    def __post_init__(self) -> None:
        if not 0.0 <= self.acceleration <= 100.0:
            raise V19PlanningError("V19 acceleration must lie in [0, 100]")
        if self.accelerated != (self.candidate is not None):
            raise V19PlanningError("V19 acceleration decision is inconsistent")
        if self.accelerated and self.certificate is None:
            raise V19PlanningError("accelerated V19 decision lacks a certificate")


def _dominates(left: V19CandidatePlan, right: V19CandidatePlan) -> bool:
    left_values = (left.predicted_cost_p90_ms, *left.risk_ucb.as_tuple())
    right_values = (right.predicted_cost_p90_ms, *right.risk_ucb.as_tuple())
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def _risk_not_better(
    faster: V19HumanRiskVector,
    slower: V19HumanRiskVector,
) -> bool:
    """A faster point may be equal or riskier, never silently incomparable."""

    return all(
        fast >= slow
        for fast, slow in zip(faster.as_tuple(), slower.as_tuple())
    )


def _candidate_actual_steps(candidate: V19CandidatePlan) -> frozenset[int]:
    return frozenset(
        step
        for use in candidate.action_uses
        if isinstance(use, V19ActionUse)
        for step in use.step_indices
    )


class V19ReleaseFrontierCatalog:
    """Strict one-dial selector over complete V19 execution schedules."""

    def __init__(self, registry: ActionRegistry) -> None:
        self.registry = registry
        self._entries: dict[
            tuple[str, str], dict[str, V19CertifiedEnvelopeEntry]
        ] = {}
        self._envelopes: dict[str, V19WorkloadEnvelope] = {}

    def add(self, entry: V19CertifiedEnvelopeEntry) -> None:
        reasons = verify_v19_envelope_entry(self.registry, entry)
        if reasons:
            raise V19PlanningError(
                "invalid V19 release envelope: " + "; ".join(reasons)
            )
        key = (entry.envelope.digest, entry.runtime_digest)
        bucket = self._entries.setdefault(key, {})
        self._envelopes[entry.envelope.digest] = entry.envelope
        execution_digest = entry.candidate.execution_digest
        if execution_digest in bucket:
            raise V19PlanningError(
                "duplicate V19 execution schedule in release frontier"
            )
        bucket[execution_digest] = entry
        try:
            # Reject a bundle-time multi-dimensional trade-off rather than
            # discovering it as a creator-facing request failure.
            self._one_dial_chain(tuple(bucket.values()))
        except V19PlanningError:
            del bucket[execution_digest]
            if not bucket:
                del self._entries[key]
            raise

    @staticmethod
    def _pareto(
        entries: tuple[V19CertifiedEnvelopeEntry, ...],
    ) -> tuple[V19CertifiedEnvelopeEntry, ...]:
        return tuple(
            entry
            for entry in entries
            if not any(
                other.candidate.execution_digest
                != entry.candidate.execution_digest
                and _dominates(other.candidate, entry.candidate)
                for other in entries
            )
        )

    @classmethod
    def _one_dial_chain(
        cls,
        entries: tuple[V19CertifiedEnvelopeEntry, ...],
    ) -> tuple[V19CertifiedEnvelopeEntry, ...]:
        frontier = sorted(
            cls._pareto(entries),
            key=lambda row: (
                -row.candidate.predicted_cost_p90_ms,
                row.candidate.execution_digest,
            ),
        )
        for slower, faster in zip(frontier, frontier[1:]):
            if not _risk_not_better(
                faster.candidate.risk_ucb,
                slower.candidate.risk_ucb,
            ):
                raise V19PlanningError(
                    "V19 certified frontier is multi-dimensional and cannot "
                    "be hidden behind one acceleration dial"
                )
        return tuple(frontier)

    def select(
        self,
        *,
        workload: V19WorkloadContext,
        runtime_digest: str,
        acceleration: float,
        required_actual_step_indices: tuple[int, ...] = (),
    ) -> V19AccelerationDecision:
        if not math.isfinite(acceleration) or not 0.0 <= acceleration <= 100.0:
            raise V19PlanningError("V19 acceleration must lie in [0, 100]")
        if (
            tuple(sorted(set(required_actual_step_indices)))
            != required_actual_step_indices
            or any(
                step < 0
                or workload.steps is None
                or step >= workload.steps
                for step in required_actual_step_indices
            )
        ):
            raise V19PlanningError(
                "V19 required actual steps must be sorted, unique and in range"
            )
        generation_key = V19GenerationRequestKey.from_workload(workload)
        if acceleration == 0.0:
            return V19AccelerationDecision(
                accelerated=False,
                reason="zero_acceleration_dense_fallback",
                generation_request_digest=generation_key.digest,
                runtime_digest=runtime_digest,
                acceleration=acceleration,
            )
        matching = tuple(
            envelope
            for digest, envelope in self._envelopes.items()
            if (digest, runtime_digest) in self._entries
            and envelope.contains(workload)
        )
        if not matching:
            return V19AccelerationDecision(
                accelerated=False,
                reason="uncalibrated_or_ood_dense_fallback",
                generation_request_digest=generation_key.digest,
                runtime_digest=runtime_digest,
                acceleration=acceleration,
            )
        most_specific = tuple(
            envelope
            for envelope in matching
            if not any(
                other.digest != envelope.digest
                and other.is_subset_of(envelope)
                for other in matching
            )
        )
        if len(most_specific) != 1:
            return V19AccelerationDecision(
                accelerated=False,
                reason="ambiguous_calibrated_envelope_dense_fallback",
                generation_request_digest=generation_key.digest,
                runtime_digest=runtime_digest,
                acceleration=acceleration,
            )
        envelope = most_specific[0]
        bucket = self._entries[(envelope.digest, runtime_digest)]
        required = frozenset(required_actual_step_indices)
        compatible = tuple(
            entry
            for entry in bucket.values()
            if required.issubset(_candidate_actual_steps(entry.candidate))
        )
        if not compatible:
            return V19AccelerationDecision(
                accelerated=False,
                reason="preview_anchor_uncalibrated_dense_fallback",
                generation_request_digest=generation_key.digest,
                runtime_digest=runtime_digest,
                acceleration=acceleration,
            )
        chain = self._one_dial_chain(compatible)
        slowest = chain[0].candidate.predicted_cost_p90_ms
        fastest = chain[-1].candidate.predicted_cost_p90_ms
        target = slowest - acceleration / 100.0 * (slowest - fastest)
        eligible = tuple(
            entry
            for entry in chain
            if entry.candidate.predicted_cost_p90_ms <= target + 1.0e-9
        )
        selected = max(
            eligible or (chain[-1],),
            key=lambda row: row.candidate.predicted_cost_p90_ms,
        )
        return V19AccelerationDecision(
            accelerated=True,
            reason="certified_frontier_selection",
            generation_request_digest=generation_key.digest,
            runtime_digest=runtime_digest,
            acceleration=acceleration,
            target_cost_p90_ms=target,
            candidate=selected.candidate,
            certificate=selected.certificate,
        )


__all__ = [
    "V19_ENVELOPE_CERTIFICATE_SCHEMA",
    "V19_RELEASE_FRONTIER_SCHEMA",
    "V19AccelerationDecision",
    "V19CertifiedEnvelopeEntry",
    "V19CertifiedFrontierEntry",
    "V19EnvelopeCertificate",
    "V19GenerationRequestKey",
    "V19ReferenceProfile",
    "V19ReleaseFrontierCatalog",
    "V19WorkloadEnvelope",
    "build_v19_certified_envelope_entry",
    "verify_v19_envelope_entry",
]
