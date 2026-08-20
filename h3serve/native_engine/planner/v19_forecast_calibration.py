"""First-class calibration contract for V19 forecast composite actions.

The calibrated latency scope is deliberately *forecast steps only*.  The
preceding/following actual steps remain part of the composite identity because
they define the legal trajectory and correction policy, but their Attention
costs are priced independently by :mod:`v19_calibration`.  Including the
following correction step here would silently double-charge every forecast
run in a complete V19 plan.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

from .action_registry import ActionEvidenceBinding, ActionKind, ActionRegistry
from .v19_calibration import (
    V19CalibrationError,
    V19CalibrationWorkload,
    V19RuntimeFingerprint,
    V19SourceRecord,
    conservative_quantile,
)


V19_FORECAST_CALIBRATION_SCHEMA = "h3_v19_forecast_composite_calibration_v2"
V19_FORECAST_COST_SCOPE = "forecast_steps_only"


def _sha256(document: object) -> str:
    encoded = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class V19ForecastCompositeKey:
    forecast_step_indices: tuple[int, ...]
    preceding_actual_step: int
    following_actual_step: int
    anchor_depth: int
    anchor_action_id: str
    anchor_canonical_action: str
    extrapolator_id: str
    correction_id: str

    def __post_init__(self) -> None:
        if (
            not self.forecast_step_indices
            or tuple(range(
                self.forecast_step_indices[0], self.forecast_step_indices[-1] + 1
            )) != self.forecast_step_indices
        ):
            raise V19CalibrationError("forecast composite must be one contiguous run")
        if self.forecast_step_indices[0] != self.preceding_actual_step + 1:
            raise V19CalibrationError("forecast run does not follow its actual anchor")
        if self.following_actual_step != self.forecast_step_indices[-1] + 1:
            raise V19CalibrationError("forecast run lacks an immediate actual correction")
        if self.anchor_depth <= 0 or self.anchor_depth > 50:
            raise V19CalibrationError("forecast anchor depth is invalid")
        if not all((
            self.anchor_action_id,
            self.anchor_canonical_action,
            self.extrapolator_id,
            self.correction_id,
        )):
            raise V19CalibrationError("forecast composite identity cannot be empty")

    @property
    def digest(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class V19ForecastCompositeMeasurement:
    key: V19ForecastCompositeKey
    warm_samples_ms: tuple[float, ...]
    initialization_samples_ms: tuple[float, ...] = ()
    peak_vram_gib_samples: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "warm_samples_ms",
            "initialization_samples_ms",
            "peak_vram_gib_samples",
        ):
            values = tuple(float(value) for value in getattr(self, name))
            if (
                (name == "warm_samples_ms" and not values)
                or any(not math.isfinite(value) or value < 0.0 for value in values)
            ):
                raise V19CalibrationError(f"invalid forecast {name}")
            object.__setattr__(self, name, values)

    @property
    def p50_ms(self) -> float:
        return conservative_quantile(self.warm_samples_ms, 0.50)

    @property
    def p90_ms(self) -> float:
        return conservative_quantile(self.warm_samples_ms, 0.90)

    @property
    def peak_vram_gib(self) -> float | None:
        return max(self.peak_vram_gib_samples, default=None)

    def to_dict(self) -> dict[str, object]:
        return {
            "key": asdict(self.key),
            "key_digest": self.key.digest,
            "warm_samples_ms": self.warm_samples_ms,
            "initialization_samples_ms": self.initialization_samples_ms,
            "peak_vram_gib_samples": self.peak_vram_gib_samples,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "peak_vram_gib": self.peak_vram_gib,
        }


@dataclass(frozen=True, slots=True)
class V19ForecastCompositeCalibration:
    calibration_id: str
    binding: ActionEvidenceBinding
    workload: V19CalibrationWorkload
    runtime: V19RuntimeFingerprint
    measurements: tuple[V19ForecastCompositeMeasurement, ...]
    sources: tuple[V19SourceRecord, ...]
    complete: bool
    minimum_warm_samples: int = 3
    cost_scope: str = V19_FORECAST_COST_SCOPE
    schema_version: str = V19_FORECAST_CALIBRATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V19_FORECAST_CALIBRATION_SCHEMA:
            raise V19CalibrationError("unsupported forecast calibration schema")
        if self.cost_scope != V19_FORECAST_COST_SCOPE:
            raise V19CalibrationError(
                "forecast calibration must exclude actual/correction-step costs"
            )
        if self.calibration_id != self.binding.evidence_id:
            raise V19CalibrationError("forecast calibration binding disagrees")
        if self.minimum_warm_samples < 3:
            raise V19CalibrationError("forecast p90 requires at least three samples")
        if not self.measurements or not self.sources:
            raise V19CalibrationError("forecast calibration lacks evidence")
        keys = tuple(row.key.digest for row in self.measurements)
        if len(set(keys)) != len(keys):
            raise V19CalibrationError("duplicate forecast composite key")
        actual = set(self.workload.actual_step_indices)
        for row in self.measurements:
            if (
                row.key.preceding_actual_step not in actual
                or row.key.following_actual_step not in actual
                or actual.intersection(row.key.forecast_step_indices)
            ):
                raise V19CalibrationError("forecast key contradicts workload trajectory")
        if self.binding.evidence_sha256 != self.payload_sha256:
            raise V19CalibrationError("forecast payload digest does not match binding")

    def _payload_document(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "calibration_id": self.calibration_id,
            "binding": {
                "action_id": self.binding.action_id,
                "implementation_id": self.binding.implementation_id,
                "registry_digest": self.binding.registry_digest,
                "evidence_id": self.binding.evidence_id,
            },
            "workload": asdict(self.workload),
            "runtime": asdict(self.runtime),
            "measurements": [row.to_dict() for row in self.measurements],
            "sources": [asdict(row) for row in self.sources],
            "complete": self.complete,
            "minimum_warm_samples": self.minimum_warm_samples,
            "cost_scope": self.cost_scope,
        }

    @property
    def payload_sha256(self) -> str:
        return _sha256(self._payload_document())

    @property
    def planner_ready(self) -> bool:
        return self.complete and all(
            len(row.warm_samples_ms) >= self.minimum_warm_samples
            and len(row.peak_vram_gib_samples) >= self.minimum_warm_samples
            for row in self.measurements
        )

    def require_planner_ready(self) -> None:
        if not self.planner_ready:
            raise V19CalibrationError(
                "forecast calibration lacks complete repeated latency/VRAM samples"
            )

    def to_dict(self) -> dict[str, object]:
        document = self._payload_document()
        document["binding"] = asdict(self.binding)
        document["planner_ready"] = self.planner_ready
        return document


def create_v19_forecast_calibration(
    *,
    registry: ActionRegistry,
    action_id: str,
    calibration_id: str,
    workload: V19CalibrationWorkload,
    runtime: V19RuntimeFingerprint,
    measurements: Iterable[V19ForecastCompositeMeasurement],
    sources: Iterable[V19SourceRecord],
    complete: bool,
    minimum_warm_samples: int = 3,
) -> V19ForecastCompositeCalibration:
    action = registry.resolve(action_id)
    if action.kind is not ActionKind.FORECAST_COMPOSITE:
        raise V19CalibrationError("forecast artifact is bound to a non-forecast action")
    if calibration_id not in action.calibration_ids:
        raise V19CalibrationError("forecast calibration id is not registered")
    placeholder = ActionEvidenceBinding(
        action_id=action.action_id,
        implementation_id=action.implementation_id,
        registry_digest=registry.digest,
        evidence_id=calibration_id,
        evidence_sha256="0" * 64,
    )
    provisional = object.__new__(V19ForecastCompositeCalibration)
    for name, value in (
        ("calibration_id", calibration_id),
        ("binding", placeholder),
        ("workload", workload),
        ("runtime", runtime),
        ("measurements", tuple(measurements)),
        ("sources", tuple(sources)),
        ("complete", bool(complete)),
        ("minimum_warm_samples", minimum_warm_samples),
        ("cost_scope", V19_FORECAST_COST_SCOPE),
        ("schema_version", V19_FORECAST_CALIBRATION_SCHEMA),
    ):
        object.__setattr__(provisional, name, value)
    binding = replace(placeholder, evidence_sha256=provisional.payload_sha256)
    return V19ForecastCompositeCalibration(
        calibration_id=calibration_id,
        binding=binding,
        workload=workload,
        runtime=runtime,
        measurements=provisional.measurements,
        sources=provisional.sources,
        complete=bool(complete),
        minimum_warm_samples=minimum_warm_samples,
        cost_scope=V19_FORECAST_COST_SCOPE,
    )


def _measurement_from_dict(
    document: Mapping[str, object],
) -> V19ForecastCompositeMeasurement:
    key_document = dict(document["key"])
    key_document["forecast_step_indices"] = tuple(
        key_document["forecast_step_indices"]
    )
    row = V19ForecastCompositeMeasurement(
        key=V19ForecastCompositeKey(**key_document),
        warm_samples_ms=tuple(float(value) for value in document["warm_samples_ms"]),
        initialization_samples_ms=tuple(
            float(value) for value in document.get("initialization_samples_ms", ())
        ),
        peak_vram_gib_samples=tuple(
            float(value) for value in document.get("peak_vram_gib_samples", ())
        ),
    )
    if document.get("key_digest", row.key.digest) != row.key.digest:
        raise V19CalibrationError("forecast key digest mismatch")
    for name, expected in (("p50_ms", row.p50_ms), ("p90_ms", row.p90_ms)):
        if name in document and not math.isclose(
            float(document[name]), expected, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise V19CalibrationError(f"stored forecast {name} is not reproducible")
    return row


def load_v19_forecast_calibration(
    path: Path,
    *,
    registry: ActionRegistry,
    expected_workload: V19CalibrationWorkload | None = None,
    expected_runtime: V19RuntimeFingerprint | None = None,
    require_planner_ready: bool = True,
) -> V19ForecastCompositeCalibration:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        workload_document = dict(document["workload"])
        workload_document["actual_step_indices"] = tuple(
            workload_document["actual_step_indices"]
        )
        artifact = V19ForecastCompositeCalibration(
            calibration_id=str(document["calibration_id"]),
            binding=ActionEvidenceBinding(**document["binding"]),
            workload=V19CalibrationWorkload(**workload_document),
            runtime=V19RuntimeFingerprint(**document["runtime"]),
            measurements=tuple(
                _measurement_from_dict(row) for row in document["measurements"]
            ),
            sources=tuple(V19SourceRecord(**row) for row in document["sources"]),
            complete=bool(document["complete"]),
            minimum_warm_samples=int(document.get("minimum_warm_samples", 3)),
            cost_scope=str(document["cost_scope"]),
            schema_version=str(document["schema_version"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, V19CalibrationError):
            raise
        raise V19CalibrationError(f"invalid forecast calibration: {path}") from error
    registry.verify_evidence_binding(artifact.binding)
    if expected_workload is not None and artifact.workload != expected_workload:
        raise V19CalibrationError("forecast workload bucket mismatch")
    if expected_runtime is not None and artifact.runtime != expected_runtime:
        raise V19CalibrationError("forecast runtime/build fingerprint mismatch")
    if require_planner_ready:
        artifact.require_planner_ready()
    return artifact


@dataclass(frozen=True, slots=True)
class V19ForecastCalibratedCost:
    p50_ms: float
    p90_ms: float
    evidence_id: str
    peak_vram_gib: float


class V19ForecastCalibrationCatalog:
    def __init__(self, registry: ActionRegistry) -> None:
        self.registry = registry
        self._rows: dict[
            tuple[str, str, str, str],
            tuple[V19ForecastCompositeCalibration, V19ForecastCompositeMeasurement],
        ] = {}

    def add(self, artifact: V19ForecastCompositeCalibration) -> None:
        self.registry.verify_evidence_binding(artifact.binding)
        artifact.require_planner_ready()
        for row in artifact.measurements:
            key = (
                artifact.binding.action_id,
                artifact.workload.digest,
                artifact.runtime.digest,
                row.key.digest,
            )
            if key in self._rows:
                raise V19CalibrationError("duplicate calibrated forecast composite")
            self._rows[key] = (artifact, row)

    def estimate(
        self,
        *,
        action_id: str,
        key: V19ForecastCompositeKey,
        workload: V19CalibrationWorkload,
        runtime: V19RuntimeFingerprint,
    ) -> V19ForecastCalibratedCost:
        try:
            artifact, row = self._rows[(
                action_id, workload.digest, runtime.digest, key.digest
            )]
        except KeyError as error:
            raise V19CalibrationError(
                "forecast composite has no exact physical calibration"
            ) from error
        assert row.peak_vram_gib is not None
        return V19ForecastCalibratedCost(
            p50_ms=row.p50_ms,
            p90_ms=row.p90_ms,
            evidence_id=artifact.calibration_id,
            peak_vram_gib=row.peak_vram_gib,
        )


__all__ = [
    "V19_FORECAST_CALIBRATION_SCHEMA",
    "V19_FORECAST_COST_SCOPE",
    "V19ForecastCalibratedCost",
    "V19ForecastCalibrationCatalog",
    "V19ForecastCompositeCalibration",
    "V19ForecastCompositeKey",
    "V19ForecastCompositeMeasurement",
    "create_v19_forecast_calibration",
    "load_v19_forecast_calibration",
]
