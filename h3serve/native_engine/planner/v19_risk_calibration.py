"""Plan-bound Human risk calibration for V19.

V19 does not manufacture a scalar quality score from Dense-relative RMS.  A
release candidate is instead bound to complete, mechanism-level Human reviews
of the exact execution schedule.  Each quality dimension receives its own
conservative rejection-probability upper bound; dimensions never compensate
for one another.

The artifact is deliberately strict:

* the registry, physical action ids, workload, runtime and execution schedule
  are part of the identity;
* raw per-case labels are retained and sealed;
* shared Dense/candidate failures and unattributed failures cannot be used as
  acceleration-risk samples;
* every dimension needs repeated observations before an artifact is planner
  ready.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

from .action_registry import ActionRegistry, ActionRegistryError
from .v19_calibration import (
    V19CalibrationError,
    V19CalibrationWorkload,
    V19RuntimeFingerprint,
    V19SourceRecord,
)
from .v19_contracts import V19HumanRiskVector
from .v19_evidence import Attribution, V19DimensionLabels


V19_RISK_CALIBRATION_SCHEMA = "h3_v19_plan_human_risk_calibration_v2"
_DIMENSIONS = (
    "prompt_adherence",
    "contact_causality",
    "trajectory_continuity",
    "temporal_clarity",
    "identity_binding",
    "audio_integrity",
    "anomaly",
)


class V19RiskCalibrationError(V19CalibrationError):
    """A Human-risk artifact cannot safely support planning."""


def _sha256(document: object) -> str:
    encoded = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def wilson_upper_bound(rejects: int, total: int, *, z_score: float) -> float:
    """One-sided Wilson upper confidence bound for a rejection probability."""

    if total <= 0 or rejects < 0 or rejects > total:
        raise V19RiskCalibrationError("invalid Human review counts")
    if not math.isfinite(z_score) or z_score <= 0.0:
        raise V19RiskCalibrationError("risk z-score must be finite and positive")
    p_hat = rejects / total
    z2 = z_score * z_score
    denominator = 1.0 + z2 / total
    centre = p_hat + z2 / (2.0 * total)
    radius = z_score * math.sqrt(
        p_hat * (1.0 - p_hat) / total + z2 / (4.0 * total * total)
    )
    return min(1.0, (centre + radius) / denominator)


@dataclass(frozen=True, slots=True)
class V19RiskReview:
    """One independent Human review case for an exact candidate schedule."""

    case_id: str
    mechanism: str
    attribution: Attribution
    dimensions: V19DimensionLabels
    candidate_artifact_sha256: str
    comparator_artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.case_id or not self.mechanism:
            raise V19RiskCalibrationError("Human risk review identity cannot be empty")
        if self.attribution not in ("candidate_positive", "candidate_regression"):
            raise V19RiskCalibrationError(
                "shared or unattributed failures cannot calibrate acceleration risk"
            )
        labels = self.dimensions.as_tuple()
        if all(label == "not_reported" for label in labels):
            raise V19RiskCalibrationError("Human risk review reports no dimensions")
        if self.attribution == "candidate_positive" and "reject" in labels:
            raise V19RiskCalibrationError(
                "a candidate-positive review cannot contain a rejected dimension"
            )
        if self.attribution == "candidate_regression" and "reject" not in labels:
            raise V19RiskCalibrationError(
                "a candidate regression must identify at least one rejected dimension"
            )
        for name in (
            "candidate_artifact_sha256",
            "comparator_artifact_sha256",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if len(value) != 64:
                raise V19RiskCalibrationError(
                    f"{name} must be a SHA256 digest"
                )
            try:
                int(value, 16)
            except ValueError as error:
                raise V19RiskCalibrationError(
                    f"{name} is not hexadecimal"
                ) from error
            if value == "0" * 64:
                raise V19RiskCalibrationError(
                    f"{name} cannot be an unbound placeholder"
                )


@dataclass(frozen=True, slots=True)
class V19RiskDimensionSummary:
    accepted: int
    rejected: int
    not_reported: int
    upper_bound: float

    def __post_init__(self) -> None:
        if min(self.accepted, self.rejected, self.not_reported) < 0:
            raise V19RiskCalibrationError("Human review counts cannot be negative")
        if not math.isfinite(self.upper_bound) or not 0.0 <= self.upper_bound <= 1.0:
            raise V19RiskCalibrationError("Human risk bound must lie in [0, 1]")

    @property
    def reported(self) -> int:
        return self.accepted + self.rejected


@dataclass(frozen=True, slots=True)
class V19PlanRiskBinding:
    registry_digest: str
    execution_digest: str
    risk_model_id: str
    payload_sha256: str

    def __post_init__(self) -> None:
        if not self.risk_model_id:
            raise V19RiskCalibrationError("risk model id cannot be empty")
        for name in ("registry_digest", "execution_digest", "payload_sha256"):
            value = getattr(self, name)
            if len(value) != 64:
                raise V19RiskCalibrationError(f"{name} must be a SHA256 digest")
            try:
                int(value, 16)
            except ValueError as error:
                raise V19RiskCalibrationError(f"{name} is not hexadecimal") from error


@dataclass(frozen=True, slots=True)
class V19PlanRiskCalibration:
    binding: V19PlanRiskBinding
    action_ids: tuple[str, ...]
    workload: V19CalibrationWorkload
    runtime: V19RuntimeFingerprint
    reviews: tuple[V19RiskReview, ...]
    dimension_summaries: tuple[V19RiskDimensionSummary, ...]
    sources: tuple[V19SourceRecord, ...]
    complete: bool
    minimum_reported_cases: int = 3
    # 2.45 is approximately a Bonferroni-adjusted one-sided 95% family-wise
    # bound for seven independently inspected dimensions.
    z_score: float = 2.45
    schema_version: str = V19_RISK_CALIBRATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V19_RISK_CALIBRATION_SCHEMA:
            raise V19RiskCalibrationError("unsupported V19 risk calibration schema")
        if not self.action_ids or tuple(sorted(set(self.action_ids))) != self.action_ids:
            raise V19RiskCalibrationError("risk action ids must be sorted and unique")
        if self.minimum_reported_cases < 3:
            raise V19RiskCalibrationError(
                "V19 Human risk calibration requires at least three cases per dimension"
            )
        if len(self.dimension_summaries) != len(_DIMENSIONS):
            raise V19RiskCalibrationError("risk calibration dimension count mismatch")
        if not self.reviews or not self.sources:
            raise V19RiskCalibrationError("risk calibration lacks evidence")
        if len({row.case_id for row in self.reviews}) != len(self.reviews):
            raise V19RiskCalibrationError("Human risk case ids must be unique")
        if len({row.source_id for row in self.sources}) != len(self.sources):
            raise V19RiskCalibrationError("risk source ids must be unique")
        expected = _summaries_from_reviews(self.reviews, z_score=self.z_score)
        if self.dimension_summaries != expected:
            raise V19RiskCalibrationError(
                "stored Human risk summaries do not match raw reviews"
            )
        if self.binding.payload_sha256 != self.payload_sha256:
            raise V19RiskCalibrationError("risk payload digest does not match binding")

    def _payload_document(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "binding": {
                "registry_digest": self.binding.registry_digest,
                "execution_digest": self.binding.execution_digest,
                "risk_model_id": self.binding.risk_model_id,
            },
            "action_ids": list(self.action_ids),
            "workload": asdict(self.workload),
            "runtime": asdict(self.runtime),
            "reviews": [
                {
                    "case_id": row.case_id,
                    "mechanism": row.mechanism,
                    "attribution": row.attribution,
                    "dimensions": asdict(row.dimensions),
                    "candidate_artifact_sha256": (
                        row.candidate_artifact_sha256
                    ),
                    "comparator_artifact_sha256": (
                        row.comparator_artifact_sha256
                    ),
                }
                for row in self.reviews
            ],
            "dimension_summaries": [asdict(row) for row in self.dimension_summaries],
            "sources": [asdict(row) for row in self.sources],
            "complete": self.complete,
            "minimum_reported_cases": self.minimum_reported_cases,
            "z_score": self.z_score,
        }

    @property
    def payload_sha256(self) -> str:
        return _sha256(self._payload_document())

    @property
    def risk_ucb(self) -> V19HumanRiskVector:
        return V19HumanRiskVector(
            *(row.upper_bound for row in self.dimension_summaries)
        )

    @property
    def planner_ready(self) -> bool:
        return self.complete and all(
            row.reported >= self.minimum_reported_cases
            for row in self.dimension_summaries
        )

    def require_planner_ready(self) -> None:
        if not self.complete:
            raise V19RiskCalibrationError("risk calibration is explicitly incomplete")
        deficient = tuple(
            dimension
            for dimension, row in zip(_DIMENSIONS, self.dimension_summaries)
            if row.reported < self.minimum_reported_cases
        )
        if deficient:
            raise V19RiskCalibrationError(
                "risk calibration lacks repeated Human labels for: "
                + ", ".join(deficient)
            )

    def to_dict(self) -> dict[str, object]:
        document = self._payload_document()
        document["binding"] = asdict(self.binding)
        document["risk_ucb"] = asdict(self.risk_ucb)
        document["planner_ready"] = self.planner_ready
        return document


def _summaries_from_reviews(
    reviews: Iterable[V19RiskReview],
    *,
    z_score: float,
) -> tuple[V19RiskDimensionSummary, ...]:
    rows = tuple(reviews)
    summaries: list[V19RiskDimensionSummary] = []
    for index, _dimension in enumerate(_DIMENSIONS):
        labels = tuple(row.dimensions.as_tuple()[index] for row in rows)
        accepted = labels.count("accept")
        rejected = labels.count("reject")
        not_reported = labels.count("not_reported")
        reported = accepted + rejected
        upper = (
            wilson_upper_bound(rejected, reported, z_score=z_score)
            if reported
            else 1.0
        )
        summaries.append(V19RiskDimensionSummary(
            accepted=accepted,
            rejected=rejected,
            not_reported=not_reported,
            upper_bound=upper,
        ))
    return tuple(summaries)


def create_v19_plan_risk_calibration(
    *,
    registry: ActionRegistry,
    execution_digest: str,
    risk_model_id: str,
    action_ids: Iterable[str],
    workload: V19CalibrationWorkload,
    runtime: V19RuntimeFingerprint,
    reviews: Iterable[V19RiskReview],
    sources: Iterable[V19SourceRecord],
    complete: bool,
    minimum_reported_cases: int = 3,
    z_score: float = 2.45,
) -> V19PlanRiskCalibration:
    stable_actions = tuple(sorted(set(action_ids)))
    if not stable_actions:
        raise V19RiskCalibrationError("risk calibration requires physical actions")
    for action_id in stable_actions:
        try:
            registry.resolve(action_id)
        except ActionRegistryError as error:
            raise V19RiskCalibrationError(str(error)) from error
    placeholder = V19PlanRiskBinding(
        registry_digest=registry.digest,
        execution_digest=execution_digest,
        risk_model_id=risk_model_id,
        payload_sha256="0" * 64,
    )
    review_rows = tuple(reviews)
    summary_rows = _summaries_from_reviews(review_rows, z_score=z_score)
    provisional = object.__new__(V19PlanRiskCalibration)
    for name, value in (
        ("binding", placeholder),
        ("action_ids", stable_actions),
        ("workload", workload),
        ("runtime", runtime),
        ("reviews", review_rows),
        ("dimension_summaries", summary_rows),
        ("sources", tuple(sources)),
        ("complete", bool(complete)),
        ("minimum_reported_cases", minimum_reported_cases),
        ("z_score", z_score),
        ("schema_version", V19_RISK_CALIBRATION_SCHEMA),
    ):
        object.__setattr__(provisional, name, value)
    binding = replace(placeholder, payload_sha256=provisional.payload_sha256)
    return V19PlanRiskCalibration(
        binding=binding,
        action_ids=stable_actions,
        workload=workload,
        runtime=runtime,
        reviews=review_rows,
        dimension_summaries=summary_rows,
        sources=provisional.sources,
        complete=bool(complete),
        minimum_reported_cases=minimum_reported_cases,
        z_score=z_score,
    )


def _risk_review_from_dict(document: Mapping[str, object]) -> V19RiskReview:
    return V19RiskReview(
        case_id=str(document["case_id"]),
        mechanism=str(document["mechanism"]),
        attribution=str(document["attribution"]),
        dimensions=V19DimensionLabels(**document["dimensions"]),
        candidate_artifact_sha256=str(
            document["candidate_artifact_sha256"]
        ),
        comparator_artifact_sha256=(
            None
            if document.get("comparator_artifact_sha256") is None
            else str(document["comparator_artifact_sha256"])
        ),
    )


def load_v19_plan_risk_calibration(
    path: str | Path,
    *,
    registry: ActionRegistry,
    expected_workload: V19CalibrationWorkload | None = None,
    expected_runtime: V19RuntimeFingerprint | None = None,
    require_planner_ready: bool = True,
) -> V19PlanRiskCalibration:
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
        workload_document = dict(document["workload"])
        workload_document["actual_step_indices"] = tuple(
            workload_document["actual_step_indices"]
        )
        artifact = V19PlanRiskCalibration(
            binding=V19PlanRiskBinding(**document["binding"]),
            action_ids=tuple(document["action_ids"]),
            workload=V19CalibrationWorkload(**workload_document),
            runtime=V19RuntimeFingerprint(**document["runtime"]),
            reviews=tuple(
                _risk_review_from_dict(row) for row in document["reviews"]
            ),
            dimension_summaries=tuple(
                V19RiskDimensionSummary(**row)
                for row in document["dimension_summaries"]
            ),
            sources=tuple(V19SourceRecord(**row) for row in document["sources"]),
            complete=bool(document["complete"]),
            minimum_reported_cases=int(document.get("minimum_reported_cases", 3)),
            z_score=float(document.get("z_score", 2.45)),
            schema_version=str(document["schema_version"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, V19RiskCalibrationError):
            raise
        raise V19RiskCalibrationError(
            f"invalid V19 Human risk artifact: {source}"
        ) from error
    if artifact.binding.registry_digest != registry.digest:
        raise V19RiskCalibrationError("risk evidence was produced for another registry")
    for action_id in artifact.action_ids:
        try:
            registry.resolve(action_id)
        except ActionRegistryError as error:
            raise V19RiskCalibrationError(str(error)) from error
    if expected_workload is not None and artifact.workload != expected_workload:
        raise V19RiskCalibrationError("risk workload bucket mismatch")
    if expected_runtime is not None and artifact.runtime != expected_runtime:
        raise V19RiskCalibrationError("risk runtime/build fingerprint mismatch")
    if require_planner_ready:
        artifact.require_planner_ready()
    return artifact


class V19RiskCalibrationCatalog:
    """Exact plan/workload/runtime Human-risk lookup; never interpolates."""

    def __init__(self, registry: ActionRegistry) -> None:
        self.registry = registry
        self._artifacts: dict[
            tuple[str, str, str], V19PlanRiskCalibration
        ] = {}

    def add(self, artifact: V19PlanRiskCalibration) -> None:
        if artifact.binding.registry_digest != self.registry.digest:
            raise V19RiskCalibrationError("risk registry digest mismatch")
        artifact.require_planner_ready()
        key = (
            artifact.binding.execution_digest,
            artifact.workload.digest,
            artifact.runtime.digest,
        )
        if key in self._artifacts:
            raise V19RiskCalibrationError("duplicate exact plan risk calibration")
        self._artifacts[key] = artifact

    def estimate(
        self,
        *,
        execution_digest: str,
        workload: V19CalibrationWorkload,
        runtime: V19RuntimeFingerprint,
        action_ids: Iterable[str],
    ) -> tuple[V19HumanRiskVector, str]:
        key = (execution_digest, workload.digest, runtime.digest)
        artifact = self._artifacts.get(key)
        if artifact is None:
            raise V19RiskCalibrationError(
                "no exact execution/workload/runtime Human-risk calibration"
            )
        if artifact.action_ids != tuple(sorted(set(action_ids))):
            raise V19RiskCalibrationError("Human-risk action identities mismatch")
        return artifact.risk_ucb, artifact.binding.risk_model_id


__all__ = [
    "V19_RISK_CALIBRATION_SCHEMA",
    "V19PlanRiskBinding",
    "V19PlanRiskCalibration",
    "V19RiskCalibrationCatalog",
    "V19RiskCalibrationError",
    "V19RiskDimensionSummary",
    "V19RiskReview",
    "create_v19_plan_risk_calibration",
    "load_v19_plan_risk_calibration",
    "wilson_upper_bound",
]
