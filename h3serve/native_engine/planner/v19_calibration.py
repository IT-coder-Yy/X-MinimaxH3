"""Strict, registry-bound physical calibration artifacts for V19.

V19 never treats a friendly Top-K label as a measured action.  Every timing
sample is bound to the physical executor implementation, exact request shape,
and kernel/source fingerprint that produced it.  Single-run historical probes
remain useful evidence, but they cannot supply a p90 service-time promise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Iterable, Mapping, Protocol

from .action_registry import (
    ActionEvidenceBinding,
    ActionRegistry,
    ActionRegistryError,
)


V19_CALIBRATION_SCHEMA = "h3_v19_physical_action_calibration_v2"


class V19CalibrationError(ValueError):
    """The artifact cannot safely support V19 planning."""


def _sha256(document: object) -> str:
    encoded = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_files(paths: Iterable[Path], *, root: Path) -> str:
    """Hash file identities and bytes, not mutable timestamps."""

    digest = hashlib.sha256()
    for path in sorted((value.resolve() for value in paths), key=str):
        if not path.is_file():
            raise V19CalibrationError(f"runtime fingerprint input is missing: {path}")
        try:
            identity = path.relative_to(root.resolve()).as_posix()
        except ValueError:
            identity = str(path)
        digest.update(identity.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def capture_v19_runtime_fingerprint(
    *,
    serve_root: Path,
    sparge_build_dir: Path,
    kernel_runtime,
) -> "V19RuntimeFingerprint":
    """Capture the same physical identity at calibration and service startup."""

    try:
        import torch

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (ImportError, OSError, subprocess.SubprocessError) as error:
        raise V19CalibrationError("cannot fingerprint the SM89 runtime") from error
    versions = tuple(row.strip() for row in result.stdout.splitlines() if row.strip())
    if len(versions) != 1:
        raise V19CalibrationError("V19 calibration requires exactly one NVIDIA GPU")
    sparge_package = sparge_build_dir.resolve() / "spas_sage_attn"
    action_paths = (
        serve_root / "h3serve/native_engine/model/kernels.py",
        sparge_package / "core.py",
        sparge_package / "_qattn.cpython-310-x86_64-linux-gnu.so",
        sparge_package / "_fused.cpython-310-x86_64-linux-gnu.so",
    )
    planner_paths = (
        serve_root / "h3serve/native_engine/planner/joint_acceleration.py",
        serve_root / "h3serve/native_engine/planner/joint_global_dp.py",
        serve_root / "h3serve/native_engine/planner/action_registry.py",
    )
    capability = tuple(kernel_runtime.cuda_capability)
    return V19RuntimeFingerprint(
        gpu_name=torch.cuda.get_device_name(0),
        device_arch=f"sm{capability[0]}{capability[1]}",
        torch_version=str(torch.__version__),
        cuda_runtime=str(torch.version.cuda),
        driver_version=versions[0],
        quant_backend=str(kernel_runtime.quant_backend),
        comfy_kitchen_cuda_sha256=kernel_runtime.comfy_kitchen_cuda_sha256,
        sageattention_sm89_sha256=kernel_runtime.sageattention_sm89_sha256,
        action_source_sha256=_sha256_files(action_paths, root=serve_root),
        planner_source_sha256=_sha256_files(planner_paths, root=serve_root),
    )


def _finite_nonnegative(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) or value < 0.0 for value in result):
        raise V19CalibrationError(f"{name} must be finite, non-negative and non-empty")
    return result


def conservative_quantile(values: Iterable[float], quantile: float) -> float:
    """Return a deterministic nearest-rank quantile (conservative for p90)."""

    ordered = tuple(sorted(_finite_nonnegative(values, name="quantile samples")))
    if not 0.0 < quantile <= 1.0:
        raise V19CalibrationError("quantile must lie in (0, 1]")
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


@dataclass(frozen=True, slots=True)
class V19CalibrationWorkload:
    model_variant: str
    service_family: str
    width: int
    height: int
    frames: int
    packed_tokens: int
    condition_count: int
    steps: int
    actual_step_indices: tuple[int, ...]
    device_arch: str = "sm89"
    sampler: str = "res_multistep"
    scheduler: str = "simple"

    def __post_init__(self) -> None:
        if self.model_variant not in ("base", "lora"):
            raise V19CalibrationError("unsupported calibration model variant")
        if self.service_family not in ("first_last", "reference"):
            raise V19CalibrationError("unsupported calibration service family")
        if any(value <= 0 for value in (
            self.width, self.height, self.frames, self.packed_tokens, self.steps
        )):
            raise V19CalibrationError("calibration dimensions must be positive")
        if self.condition_count < 0:
            raise V19CalibrationError("condition count cannot be negative")
        if (
            not self.actual_step_indices
            or tuple(sorted(set(self.actual_step_indices))) != self.actual_step_indices
            or self.actual_step_indices[0] < 0
            or self.actual_step_indices[-1] >= self.steps
        ):
            raise V19CalibrationError("actual step indices are invalid")
        if self.device_arch != "sm89":
            raise V19CalibrationError("V19 release calibration currently targets SM89")

    @property
    def digest(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class V19RuntimeFingerprint:
    gpu_name: str
    device_arch: str
    torch_version: str
    cuda_runtime: str
    driver_version: str
    quant_backend: str
    comfy_kitchen_cuda_sha256: str
    sageattention_sm89_sha256: str
    action_source_sha256: str
    planner_source_sha256: str

    def __post_init__(self) -> None:
        if self.device_arch != "sm89":
            raise V19CalibrationError("runtime fingerprint must describe SM89")
        if not all((
            self.gpu_name,
            self.torch_version,
            self.cuda_runtime,
            self.driver_version,
            self.quant_backend,
        )):
            raise V19CalibrationError("runtime fingerprint has an empty identity field")
        for name in (
            "comfy_kitchen_cuda_sha256",
            "sageattention_sm89_sha256",
            "action_source_sha256",
            "planner_source_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64:
                raise V19CalibrationError(f"{name} must be a SHA256 digest")
            try:
                int(value, 16)
            except ValueError as error:
                raise V19CalibrationError(f"{name} is not hexadecimal") from error

    @property
    def digest(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class V19SourceRecord:
    source_id: str
    relative_path: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.source_id or not self.relative_path:
            raise V19CalibrationError("source record identity cannot be empty")
        if Path(self.relative_path).is_absolute() or ".." in Path(self.relative_path).parts:
            raise V19CalibrationError("source record path must be safe and relative")
        if len(self.sha256) != 64:
            raise V19CalibrationError("source record requires SHA256")
        try:
            int(self.sha256, 16)
        except ValueError as error:
            raise V19CalibrationError("source record digest is not hexadecimal") from error


@dataclass(frozen=True, slots=True)
class V19NumericalErrorSample:
    """One Dense-relative numerical observation for a physical action cell.

    These diagnostics are not Human risk.  A single prompt/seed observation is
    retained for mechanism analysis but is never promoted to a quality UCB.
    """

    mean_cosine: float
    min_cosine: float
    global_relative_rms: float
    mean_head_relative_rms: float
    max_head_relative_rms: float
    max_relative_l1: float

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or not -1.0 <= value <= 1.0
            for value in (self.mean_cosine, self.min_cosine)
        ):
            raise V19CalibrationError("Dense-relative cosine must lie in [-1, 1]")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (
                self.global_relative_rms,
                self.mean_head_relative_rms,
                self.max_head_relative_rms,
                self.max_relative_l1,
            )
        ):
            raise V19CalibrationError("Dense-relative errors must be non-negative")
        if self.min_cosine > self.mean_cosine:
            raise V19CalibrationError("minimum cosine cannot exceed mean cosine")


@dataclass(frozen=True, slots=True)
class V19TimingMeasurement:
    canonical_action: str
    step_index: int
    layer_start: int
    layer_stop: int
    warm_samples_ms: tuple[float, ...]
    initialization_samples_ms: tuple[float, ...] = ()
    peak_vram_gib_samples: tuple[float, ...] = ()
    numerical_error_samples: tuple[V19NumericalErrorSample, ...] = ()

    def __post_init__(self) -> None:
        if not self.canonical_action:
            raise V19CalibrationError("measurement action cannot be empty")
        if self.step_index < 0 or not 0 <= self.layer_start < self.layer_stop <= 50:
            raise V19CalibrationError("measurement cell is invalid")
        object.__setattr__(
            self,
            "warm_samples_ms",
            _finite_nonnegative(self.warm_samples_ms, name="warm samples"),
        )
        for name in ("initialization_samples_ms", "peak_vram_gib_samples"):
            values = tuple(float(value) for value in getattr(self, name))
            if any(not math.isfinite(value) or value < 0.0 for value in values):
                raise V19CalibrationError(f"{name} contains an invalid value")
            object.__setattr__(self, name, values)
        object.__setattr__(
            self,
            "numerical_error_samples",
            tuple(self.numerical_error_samples),
        )

    @property
    def p50_ms(self) -> float:
        return conservative_quantile(self.warm_samples_ms, 0.50)

    @property
    def p90_ms(self) -> float:
        return conservative_quantile(self.warm_samples_ms, 0.90)

    @property
    def peak_vram_gib(self) -> float | None:
        return max(self.peak_vram_gib_samples, default=None)

    @property
    def numerical_error_repeated(self) -> bool:
        return len(self.numerical_error_samples) >= 3

    def to_dict(self) -> dict[str, object]:
        document = asdict(self)
        document.update({
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "peak_vram_gib": self.peak_vram_gib,
            "numerical_error_repeated": self.numerical_error_repeated,
        })
        return document


@dataclass(frozen=True, slots=True)
class V19ActionCalibration:
    calibration_id: str
    binding: ActionEvidenceBinding
    workload: V19CalibrationWorkload
    runtime: V19RuntimeFingerprint
    measurements: tuple[V19TimingMeasurement, ...]
    sources: tuple[V19SourceRecord, ...]
    timing_scope: str
    complete: bool
    minimum_warm_samples: int = 3
    schema_version: str = V19_CALIBRATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V19_CALIBRATION_SCHEMA:
            raise V19CalibrationError("unsupported V19 calibration schema")
        if self.calibration_id != self.binding.evidence_id:
            raise V19CalibrationError("calibration id and evidence binding disagree")
        if self.workload.device_arch != self.runtime.device_arch:
            raise V19CalibrationError("workload and runtime architecture disagree")
        if self.timing_scope not in (
            "attention_layer_call",
            "dit_step",
            "end_to_end",
            "forecast_composite",
        ):
            raise V19CalibrationError("unsupported timing scope")
        if self.minimum_warm_samples < 3:
            raise V19CalibrationError("V19 p90 calibration requires at least three warm samples")
        if not self.measurements or not self.sources:
            raise V19CalibrationError("calibration requires measurements and provenance")
        cells = tuple(
            (row.canonical_action, row.step_index, row.layer_start, row.layer_stop)
            for row in self.measurements
        )
        if len(set(cells)) != len(cells):
            raise V19CalibrationError("calibration contains duplicate measurement cells")
        if len({row.source_id for row in self.sources}) != len(self.sources):
            raise V19CalibrationError("calibration source ids must be unique")
        if self.binding.evidence_sha256 != self.payload_sha256:
            raise V19CalibrationError("calibration payload digest does not match binding")

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
            "timing_scope": self.timing_scope,
            "complete": self.complete,
            "minimum_warm_samples": self.minimum_warm_samples,
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
        if not self.complete:
            raise V19CalibrationError("calibration is explicitly incomplete")
        deficient = [
            (row.canonical_action, row.step_index, row.layer_start, row.layer_stop)
            for row in self.measurements
            if len(row.warm_samples_ms) < self.minimum_warm_samples
        ]
        if deficient:
            raise V19CalibrationError(
                "calibration lacks repeated warm samples for p90: "
                + ", ".join(map(str, deficient[:4]))
            )
        deficient_peak = [
            (row.canonical_action, row.step_index, row.layer_start, row.layer_stop)
            for row in self.measurements
            if len(row.peak_vram_gib_samples) < self.minimum_warm_samples
        ]
        if deficient_peak:
            raise V19CalibrationError(
                "calibration lacks repeated peak-VRAM samples: "
                + ", ".join(map(str, deficient_peak[:4]))
            )

    def to_dict(self) -> dict[str, object]:
        document = self._payload_document()
        document["binding"] = asdict(self.binding)
        document["planner_ready"] = self.planner_ready
        return document


def create_v19_action_calibration(
    *,
    registry: ActionRegistry,
    action_id: str,
    calibration_id: str,
    workload: V19CalibrationWorkload,
    runtime: V19RuntimeFingerprint,
    measurements: Iterable[V19TimingMeasurement],
    sources: Iterable[V19SourceRecord],
    timing_scope: str,
    complete: bool,
    minimum_warm_samples: int = 3,
) -> V19ActionCalibration:
    action = registry.resolve(action_id)
    if calibration_id not in action.calibration_ids:
        raise V19CalibrationError("calibration id is not registered for this action")
    placeholder = ActionEvidenceBinding(
        action_id=action.action_id,
        implementation_id=action.implementation_id,
        registry_digest=registry.digest,
        evidence_id=calibration_id,
        evidence_sha256="0" * 64,
    )
    # Build the digest from the immutable payload, then seal the binding.
    provisional = object.__new__(V19ActionCalibration)
    for name, value in (
        ("calibration_id", calibration_id),
        ("binding", placeholder),
        ("workload", workload),
        ("runtime", runtime),
        ("measurements", tuple(measurements)),
        ("sources", tuple(sources)),
        ("timing_scope", timing_scope),
        ("complete", bool(complete)),
        ("minimum_warm_samples", minimum_warm_samples),
        ("schema_version", V19_CALIBRATION_SCHEMA),
    ):
        object.__setattr__(provisional, name, value)
    sealed = replace(
        placeholder,
        evidence_sha256=provisional.payload_sha256,
    )
    return V19ActionCalibration(
        calibration_id=calibration_id,
        binding=sealed,
        workload=workload,
        runtime=runtime,
        measurements=provisional.measurements,
        sources=provisional.sources,
        timing_scope=timing_scope,
        complete=bool(complete),
        minimum_warm_samples=minimum_warm_samples,
    )


def _measurement_from_dict(document: Mapping[str, object]) -> V19TimingMeasurement:
    row = V19TimingMeasurement(
        canonical_action=str(document["canonical_action"]),
        step_index=int(document["step_index"]),
        layer_start=int(document["layer_start"]),
        layer_stop=int(document["layer_stop"]),
        warm_samples_ms=tuple(float(value) for value in document["warm_samples_ms"]),
        initialization_samples_ms=tuple(
            float(value) for value in document.get("initialization_samples_ms", ())
        ),
        peak_vram_gib_samples=tuple(
            float(value) for value in document.get("peak_vram_gib_samples", ())
        ),
        numerical_error_samples=tuple(
            V19NumericalErrorSample(**row)
            for row in document.get("numerical_error_samples", ())
        ),
    )
    for key, expected in (("p50_ms", row.p50_ms), ("p90_ms", row.p90_ms)):
        if key in document and not math.isclose(
            float(document[key]), expected, rel_tol=0.0, abs_tol=1e-9
        ):
            raise V19CalibrationError(f"stored {key} does not match raw samples")
    return row


def load_v19_action_calibration(
    path: Path,
    *,
    registry: ActionRegistry,
    expected_workload: V19CalibrationWorkload | None = None,
    expected_runtime: V19RuntimeFingerprint | None = None,
    require_planner_ready: bool = True,
) -> V19ActionCalibration:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        binding = ActionEvidenceBinding(**document["binding"])
        workload_document = dict(document["workload"])
        workload_document["actual_step_indices"] = tuple(
            workload_document["actual_step_indices"]
        )
        artifact = V19ActionCalibration(
            calibration_id=str(document["calibration_id"]),
            binding=binding,
            workload=V19CalibrationWorkload(**workload_document),
            runtime=V19RuntimeFingerprint(**document["runtime"]),
            measurements=tuple(
                _measurement_from_dict(row) for row in document["measurements"]
            ),
            sources=tuple(V19SourceRecord(**row) for row in document["sources"]),
            timing_scope=str(document["timing_scope"]),
            complete=bool(document["complete"]),
            minimum_warm_samples=int(document.get("minimum_warm_samples", 3)),
            schema_version=str(document["schema_version"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, V19CalibrationError):
            raise
        raise V19CalibrationError(f"invalid V19 calibration artifact: {path}") from error
    try:
        registry.verify_evidence_binding(artifact.binding)
    except ActionRegistryError as error:
        raise V19CalibrationError(str(error)) from error
    action = registry.resolve(artifact.binding.action_id)
    invalid_actions = sorted(
        {row.canonical_action for row in artifact.measurements}
        - set(action.canonical_actions)
    )
    if invalid_actions:
        raise V19CalibrationError(
            f"artifact measures actions not provided by executor: {invalid_actions}"
        )
    if expected_workload is not None and artifact.workload != expected_workload:
        raise V19CalibrationError("calibration workload bucket mismatch")
    if expected_runtime is not None and artifact.runtime != expected_runtime:
        raise V19CalibrationError("calibration runtime/build fingerprint mismatch")
    if require_planner_ready:
        artifact.require_planner_ready()
    return artifact


class V19ActionUseLike(Protocol):
    action_id: str
    canonical_action: str
    step_indices: tuple[int, ...]
    layer_start: int
    layer_stop: int


@dataclass(frozen=True, slots=True)
class V19CalibratedScheduleCost:
    p50_ms: float
    p90_ms: float
    peak_vram_gib: float
    evidence_ids: tuple[str, ...]
    cell_count: int


@dataclass(frozen=True, slots=True)
class V19CalibratedCellEvidence:
    action_id: str
    canonical_action: str
    step_index: int
    layer_index: int
    p50_ms: float
    p90_ms: float
    peak_vram_gib: float
    numerical_error_samples: tuple[V19NumericalErrorSample, ...]
    evidence_id: str


class V19CalibrationCatalog:
    """Exact lookup only; V19 does not interpolate missing physical cells."""

    def __init__(self, registry: ActionRegistry) -> None:
        self.registry = registry
        self._artifacts: dict[
            tuple[str, str, str], V19ActionCalibration
        ] = {}
        self._measurements: dict[
            tuple[str, str, str, str, int, int], V19TimingMeasurement
        ] = {}

    def add(self, artifact: V19ActionCalibration) -> None:
        try:
            self.registry.verify_evidence_binding(artifact.binding)
        except ActionRegistryError as error:
            raise V19CalibrationError(str(error)) from error
        artifact.require_planner_ready()
        artifact_key = (
            artifact.binding.action_id,
            artifact.workload.digest,
            artifact.runtime.digest,
        )
        if artifact_key in self._artifacts:
            raise V19CalibrationError("duplicate action/workload/runtime calibration")
        staged: dict[
            tuple[str, str, str, str, int, int], V19TimingMeasurement
        ] = {}
        for row in artifact.measurements:
            for layer in range(row.layer_start, row.layer_stop):
                key = (
                    artifact.binding.action_id,
                    artifact.workload.digest,
                    artifact.runtime.digest,
                    row.canonical_action,
                    row.step_index,
                    layer,
                )
                if key in self._measurements or key in staged:
                    raise V19CalibrationError("overlapping physical calibration cells")
                staged[key] = row
        self._artifacts[artifact_key] = artifact
        self._measurements.update(staged)

    def estimate_schedule(
        self,
        action_uses: Iterable[V19ActionUseLike],
        *,
        workload: V19CalibrationWorkload,
        runtime: V19RuntimeFingerprint,
    ) -> V19CalibratedScheduleCost:
        p50 = 0.0
        p90 = 0.0
        evidence_ids: set[str] = set()
        scheduled_cells: set[tuple[int, int]] = set()
        peak_vram_gib = 0.0
        count = 0
        for use in action_uses:
            artifact_key = (use.action_id, workload.digest, runtime.digest)
            artifact = self._artifacts.get(artifact_key)
            if artifact is None:
                raise V19CalibrationError(
                    "no exact action/workload/runtime calibration for "
                    f"{use.action_id}"
                )
            for step_index in use.step_indices:
                if step_index not in workload.actual_step_indices:
                    raise V19CalibrationError(
                        f"scheduled step is not an actual DiT step: {step_index}"
                    )
                for layer in range(use.layer_start, use.layer_stop):
                    cell = (step_index, layer)
                    if cell in scheduled_cells:
                        raise V19CalibrationError(
                            f"multiple actions occupy V19 cell {cell}"
                        )
                    key = (
                        use.action_id,
                        workload.digest,
                        runtime.digest,
                        use.canonical_action,
                        step_index,
                        layer,
                    )
                    measurement = self._measurements.get(key)
                    if measurement is None:
                        raise V19CalibrationError(
                            "uncalibrated V19 physical cell: "
                            f"action={use.action_id}, canonical={use.canonical_action}, "
                            f"step={step_index}, layer={layer}"
                        )
                    scheduled_cells.add(cell)
                    p50 += measurement.p50_ms
                    # Sum of cell p90s is deliberately conservative.  V19 may
                    # later replace this with an end-to-end joint distribution,
                    # but never with a falsely precise independence estimate.
                    p90 += measurement.p90_ms
                    peak_vram_gib = max(
                        peak_vram_gib,
                        measurement.peak_vram_gib or 0.0,
                    )
                    evidence_ids.add(artifact.calibration_id)
                    count += 1
        if count == 0:
            raise V19CalibrationError("empty V19 schedule")
        return V19CalibratedScheduleCost(
            p50_ms=p50,
            p90_ms=p90,
            peak_vram_gib=peak_vram_gib,
            evidence_ids=tuple(sorted(evidence_ids)),
            cell_count=count,
        )

    def lookup_cell(
        self,
        *,
        action_id: str,
        canonical_action: str,
        step_index: int,
        layer_index: int,
        workload: V19CalibrationWorkload,
        runtime: V19RuntimeFingerprint,
    ) -> V19CalibratedCellEvidence:
        """Return one exact physical cell, including diagnostic numerical error."""

        if step_index not in workload.actual_step_indices:
            raise V19CalibrationError("cell lookup step is not an actual DiT step")
        if not 0 <= layer_index < 50:
            raise V19CalibrationError("cell lookup layer is invalid")
        artifact_key = (action_id, workload.digest, runtime.digest)
        artifact = self._artifacts.get(artifact_key)
        if artifact is None:
            raise V19CalibrationError(
                f"no exact action/workload/runtime calibration for {action_id}"
            )
        key = (
            action_id,
            workload.digest,
            runtime.digest,
            canonical_action,
            step_index,
            layer_index,
        )
        measurement = self._measurements.get(key)
        if measurement is None:
            raise V19CalibrationError("uncalibrated V19 physical cell")
        return V19CalibratedCellEvidence(
            action_id=action_id,
            canonical_action=canonical_action,
            step_index=step_index,
            layer_index=layer_index,
            p50_ms=measurement.p50_ms,
            p90_ms=measurement.p90_ms,
            peak_vram_gib=measurement.peak_vram_gib or 0.0,
            numerical_error_samples=measurement.numerical_error_samples,
            evidence_id=artifact.calibration_id,
        )


__all__ = [
    "V19_CALIBRATION_SCHEMA",
    "V19ActionCalibration",
    "V19ActionUseLike",
    "V19CalibratedScheduleCost",
    "V19CalibratedCellEvidence",
    "V19CalibrationCatalog",
    "V19CalibrationError",
    "V19CalibrationWorkload",
    "V19RuntimeFingerprint",
    "V19NumericalErrorSample",
    "V19SourceRecord",
    "V19TimingMeasurement",
    "conservative_quantile",
    "capture_v19_runtime_fingerprint",
    "create_v19_action_calibration",
    "load_v19_action_calibration",
    "sha256_file",
]
