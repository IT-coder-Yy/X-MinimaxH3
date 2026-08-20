"""Exact end-to-end hot-session calibration for one V19 execution schedule.

Per-cell Attention probes are counterfactual search evidence.  They cannot
certify creator-visible latency because one request also executes MLP/Linear,
solver updates, text conditioning, VAE decode and mux.  A release candidate is
therefore timed repeatedly as the *complete* execution digest before V19 may
make an end-to-end p50/p90 claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

from .action_registry import ActionRegistry
from .v19_calibration import (
    V19CalibrationError,
    V19CalibrationWorkload,
    V19RuntimeFingerprint,
    V19SourceRecord,
    conservative_quantile,
)


V19_SCHEDULE_CALIBRATION_SCHEMA = "h3_v19_end_to_end_schedule_calibration_v1"


def _sha256(document: object) -> str:
    return hashlib.sha256(json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class V19ScheduleCostBinding:
    registry_digest: str
    execution_digest: str
    workload_digest: str
    runtime_digest: str
    action_ids: tuple[str, ...]
    evidence_id: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.action_ids))) != self.action_ids:
            raise V19CalibrationError("schedule action ids must be sorted and unique")
        for name in (
            "registry_digest", "execution_digest", "workload_digest",
            "runtime_digest", "evidence_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64:
                raise V19CalibrationError(f"invalid schedule binding digest: {name}")
            try:
                int(value, 16)
            except ValueError as error:
                raise V19CalibrationError(
                    f"non-hexadecimal schedule binding digest: {name}"
                ) from error
        if not self.action_ids or not self.evidence_id:
            raise V19CalibrationError("schedule binding identity cannot be empty")


@dataclass(frozen=True, slots=True)
class V19ScheduleCostCalibration:
    calibration_id: str
    binding: V19ScheduleCostBinding
    workload: V19CalibrationWorkload
    runtime: V19RuntimeFingerprint
    total_samples_ms: tuple[float, ...]
    denoise_samples_ms: tuple[float, ...]
    peak_vram_gib_samples: tuple[float, ...]
    sources: tuple[V19SourceRecord, ...]
    complete: bool
    minimum_samples: int = 3
    schema_version: str = V19_SCHEDULE_CALIBRATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V19_SCHEDULE_CALIBRATION_SCHEMA:
            raise V19CalibrationError("unsupported schedule calibration schema")
        if self.calibration_id != self.binding.evidence_id:
            raise V19CalibrationError("schedule calibration identity disagrees")
        if self.minimum_samples < 3:
            raise V19CalibrationError("schedule p90 requires at least three samples")
        for name in (
            "total_samples_ms", "denoise_samples_ms", "peak_vram_gib_samples"
        ):
            values = tuple(float(value) for value in getattr(self, name))
            if not values or any(
                not math.isfinite(value) or value < 0.0 for value in values
            ):
                raise V19CalibrationError(f"invalid schedule {name}")
            object.__setattr__(self, name, values)
        if not self.sources:
            raise V19CalibrationError("schedule calibration lacks provenance")
        if self.binding.workload_digest != self.workload.digest:
            raise V19CalibrationError("schedule workload digest mismatch")
        if self.binding.runtime_digest != self.runtime.digest:
            raise V19CalibrationError("schedule runtime digest mismatch")
        if self.binding.evidence_sha256 != self.payload_sha256:
            raise V19CalibrationError("schedule payload digest mismatch")

    def _payload_document(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "calibration_id": self.calibration_id,
            "binding": {
                key: value
                for key, value in asdict(self.binding).items()
                if key != "evidence_sha256"
            },
            "workload": asdict(self.workload),
            "runtime": asdict(self.runtime),
            "total_samples_ms": self.total_samples_ms,
            "denoise_samples_ms": self.denoise_samples_ms,
            "peak_vram_gib_samples": self.peak_vram_gib_samples,
            "sources": [asdict(source) for source in self.sources],
            "complete": self.complete,
            "minimum_samples": self.minimum_samples,
        }

    @property
    def payload_sha256(self) -> str:
        return _sha256(self._payload_document())

    @property
    def p50_ms(self) -> float:
        return conservative_quantile(self.total_samples_ms, 0.50)

    @property
    def p90_ms(self) -> float:
        return conservative_quantile(self.total_samples_ms, 0.90)

    @property
    def denoise_p50_ms(self) -> float:
        return conservative_quantile(self.denoise_samples_ms, 0.50)

    @property
    def denoise_p90_ms(self) -> float:
        return conservative_quantile(self.denoise_samples_ms, 0.90)

    @property
    def peak_vram_gib(self) -> float:
        return max(self.peak_vram_gib_samples)

    @property
    def planner_ready(self) -> bool:
        return self.complete and all(
            len(values) >= self.minimum_samples
            for values in (
                self.total_samples_ms,
                self.denoise_samples_ms,
                self.peak_vram_gib_samples,
            )
        )

    def require_planner_ready(self) -> None:
        if not self.planner_ready:
            raise V19CalibrationError(
                "schedule calibration lacks complete repeated E2E samples"
            )

    def to_dict(self) -> dict[str, object]:
        document = self._payload_document()
        document["binding"] = asdict(self.binding)
        document.update({
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "denoise_p50_ms": self.denoise_p50_ms,
            "denoise_p90_ms": self.denoise_p90_ms,
            "peak_vram_gib": self.peak_vram_gib,
            "planner_ready": self.planner_ready,
        })
        return document


def create_v19_schedule_cost_calibration(
    *,
    registry: ActionRegistry,
    calibration_id: str,
    execution_digest: str,
    action_ids: Iterable[str],
    workload: V19CalibrationWorkload,
    runtime: V19RuntimeFingerprint,
    total_samples_ms: Iterable[float],
    denoise_samples_ms: Iterable[float],
    peak_vram_gib_samples: Iterable[float],
    sources: Iterable[V19SourceRecord],
    complete: bool,
    minimum_samples: int = 3,
) -> V19ScheduleCostCalibration:
    action_ids = tuple(sorted(set(action_ids)))
    for action_id in action_ids:
        registry.resolve(action_id)
    placeholder = V19ScheduleCostBinding(
        registry_digest=registry.digest,
        execution_digest=execution_digest,
        workload_digest=workload.digest,
        runtime_digest=runtime.digest,
        action_ids=action_ids,
        evidence_id=calibration_id,
        evidence_sha256="0" * 64,
    )
    provisional = object.__new__(V19ScheduleCostCalibration)
    for name, value in (
        ("calibration_id", calibration_id),
        ("binding", placeholder),
        ("workload", workload),
        ("runtime", runtime),
        ("total_samples_ms", tuple(total_samples_ms)),
        ("denoise_samples_ms", tuple(denoise_samples_ms)),
        ("peak_vram_gib_samples", tuple(peak_vram_gib_samples)),
        ("sources", tuple(sources)),
        ("complete", bool(complete)),
        ("minimum_samples", int(minimum_samples)),
        ("schema_version", V19_SCHEDULE_CALIBRATION_SCHEMA),
    ):
        object.__setattr__(provisional, name, value)
    binding = replace(placeholder, evidence_sha256=provisional.payload_sha256)
    return V19ScheduleCostCalibration(
        calibration_id=calibration_id,
        binding=binding,
        workload=workload,
        runtime=runtime,
        total_samples_ms=provisional.total_samples_ms,
        denoise_samples_ms=provisional.denoise_samples_ms,
        peak_vram_gib_samples=provisional.peak_vram_gib_samples,
        sources=provisional.sources,
        complete=bool(complete),
        minimum_samples=minimum_samples,
    )


def load_v19_schedule_cost_calibration(
    path: Path,
    *,
    registry: ActionRegistry,
    require_planner_ready: bool = True,
) -> V19ScheduleCostCalibration:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        workload_document = dict(document["workload"])
        workload_document["actual_step_indices"] = tuple(
            workload_document["actual_step_indices"]
        )
        binding_document = dict(document["binding"])
        binding_document["action_ids"] = tuple(binding_document["action_ids"])
        artifact = V19ScheduleCostCalibration(
            calibration_id=str(document["calibration_id"]),
            binding=V19ScheduleCostBinding(**binding_document),
            workload=V19CalibrationWorkload(**workload_document),
            runtime=V19RuntimeFingerprint(**document["runtime"]),
            total_samples_ms=tuple(document["total_samples_ms"]),
            denoise_samples_ms=tuple(document["denoise_samples_ms"]),
            peak_vram_gib_samples=tuple(document["peak_vram_gib_samples"]),
            sources=tuple(V19SourceRecord(**row) for row in document["sources"]),
            complete=bool(document["complete"]),
            minimum_samples=int(document.get("minimum_samples", 3)),
            schema_version=str(document["schema_version"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, V19CalibrationError):
            raise
        raise V19CalibrationError(f"invalid schedule calibration: {path}") from error
    if artifact.binding.registry_digest != registry.digest:
        raise V19CalibrationError("schedule registry digest mismatch")
    for action_id in artifact.binding.action_ids:
        registry.resolve(action_id)
    if require_planner_ready:
        artifact.require_planner_ready()
    return artifact


@dataclass(frozen=True, slots=True)
class V19CalibratedEndToEndCost:
    p50_ms: float
    p90_ms: float
    denoise_p50_ms: float
    denoise_p90_ms: float
    peak_vram_gib: float
    evidence_id: str


class V19ScheduleCostCatalog:
    def __init__(self, registry: ActionRegistry) -> None:
        self.registry = registry
        self._artifacts: dict[
            tuple[str, str, str], V19ScheduleCostCalibration
        ] = {}

    def add(self, artifact: V19ScheduleCostCalibration) -> None:
        artifact.require_planner_ready()
        if artifact.binding.registry_digest != self.registry.digest:
            raise V19CalibrationError("schedule registry digest mismatch")
        key = (
            artifact.binding.execution_digest,
            artifact.workload.digest,
            artifact.runtime.digest,
        )
        if key in self._artifacts:
            raise V19CalibrationError("duplicate exact schedule calibration")
        self._artifacts[key] = artifact

    def estimate(
        self,
        *,
        execution_digest: str,
        workload: V19CalibrationWorkload,
        runtime: V19RuntimeFingerprint,
        action_ids: Iterable[str],
    ) -> V19CalibratedEndToEndCost:
        try:
            artifact = self._artifacts[(
                execution_digest, workload.digest, runtime.digest
            )]
        except KeyError as error:
            raise V19CalibrationError(
                "execution schedule has no exact repeated end-to-end calibration"
            ) from error
        if tuple(sorted(set(action_ids))) != artifact.binding.action_ids:
            raise V19CalibrationError("schedule action identities disagree")
        return V19CalibratedEndToEndCost(
            p50_ms=artifact.p50_ms,
            p90_ms=artifact.p90_ms,
            denoise_p50_ms=artifact.denoise_p50_ms,
            denoise_p90_ms=artifact.denoise_p90_ms,
            peak_vram_gib=artifact.peak_vram_gib,
            evidence_id=artifact.calibration_id,
        )


__all__ = [
    "V19_SCHEDULE_CALIBRATION_SCHEMA",
    "V19CalibratedEndToEndCost",
    "V19ScheduleCostBinding",
    "V19ScheduleCostCalibration",
    "V19ScheduleCostCatalog",
    "create_v19_schedule_cost_calibration",
    "load_v19_schedule_cost_calibration",
]
