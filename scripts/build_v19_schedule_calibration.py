#!/usr/bin/env python3
"""Build one exact repeated end-to-end V19 schedule calibration."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


SERVE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVE_ROOT))

from h3serve.native_engine.planner import (  # noqa: E402
    FIXED_TOPK_ACTION_IMPLEMENTATION,
    ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
    ROUND215_ACTION_IMPLEMENTATION,
    ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
    ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION,
    V19CalibrationError,
    V19CalibrationWorkload,
    V19CandidatePlan,
    V19HumanRiskVector,
    V19RuntimeFingerprint,
    V19SourceRecord,
    blueprint_from_runtime_schedule,
    build_v19_bootstrap_registry,
    create_v19_schedule_cost_calibration,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-report", type=Path, action="append", required=True)
    parser.add_argument(
        "--service-family", choices=("first_last", "reference"), required=True
    )
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blueprint-output", type=Path)
    parser.add_argument("--minimum-samples", type=int, default=3)
    return parser.parse_args()


def _registry():
    return build_v19_bootstrap_registry(implementation_ids={
        "fixed_topk": FIXED_TOPK_ACTION_IMPLEMENTATION,
        "round215": ROUND215_ACTION_IMPLEMENTATION,
        "round188": ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
        "round228": ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
        "round229": ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION,
    })


def _model_variant(engine: str) -> str:
    if engine in ("original", "reference"):
        return "base"
    if engine in ("lora", "reference-lora"):
        return "lora"
    raise V19CalibrationError(f"unknown benchmark engine: {engine}")


def _source(path: Path, index: int) -> V19SourceRecord:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(SERVE_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise V19CalibrationError(
            f"schedule provenance is outside the release: {resolved}"
        ) from error
    return V19SourceRecord(
        source_id=f"complete_schedule_session_{index}",
        relative_path=relative,
        sha256=sha256_file(resolved),
    )


def _execution_digest(blueprint) -> str:
    return V19CandidatePlan(
        candidate_id=blueprint.candidate_id,
        action_uses=blueprint.action_uses,
        predicted_cost_p50_ms=0.0,
        predicted_cost_p90_ms=0.0,
        predicted_peak_vram_gib=0.0,
        risk_ucb=V19HumanRiskVector(*(1.0 for _ in range(7))),
        terminal_debt=blueprint.terminal_debt,
        maximum_debt=blueprint.maximum_debt,
        source=blueprint.source,
    ).execution_digest


def main() -> int:
    args = parse_args()
    if args.minimum_samples < 3:
        raise SystemExit("--minimum-samples must be at least three")
    registry = _registry()
    workload = None
    runtime = None
    blueprint = None
    execution_digest = None
    total_samples_ms: list[float] = []
    denoise_samples_ms: list[float] = []
    peak_samples: list[float] = []
    sources: list[V19SourceRecord] = []

    for report_index, report_path in enumerate(args.session_report, start=1):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        fingerprint = report.get("v19_runtime_fingerprint")
        contract = report.get("contract")
        requests = report.get("requests")
        if (
            not isinstance(fingerprint, dict)
            or not isinstance(contract, dict)
            or not isinstance(requests, list)
            or not requests
        ):
            raise V19CalibrationError("invalid complete-schedule session report")
        current_runtime = V19RuntimeFingerprint(**fingerprint)
        for index, request in enumerate(requests, start=1):
            if request.get("status") == "failed" or "total_seconds" not in request:
                raise V19CalibrationError("schedule calibration includes a failed request")
            execution = request.get("execution_profile")
            phases = request.get("phases")
            if not isinstance(execution, dict) or not isinstance(phases, dict):
                raise V19CalibrationError("schedule request lacks runtime telemetry")
            actual_steps = tuple(int(v) for v in request["actual_step_indices"])
            current_workload = V19CalibrationWorkload(
                model_variant=_model_variant(str(report["engine"])),
                service_family=args.service_family,
                width=int(request["width"]),
                height=int(request["height"]),
                frames=int(request["frames"]),
                packed_tokens=int(execution["packed_tokens"]),
                condition_count=int(execution["condition_count"]),
                steps=int(request["steps"]),
                actual_step_indices=actual_steps,
                device_arch=str(fingerprint["device_arch"]),
                sampler=str(contract["sampler"]),
                scheduler=str(contract["scheduler"]),
            )
            current_blueprint = blueprint_from_runtime_schedule(
                candidate_id=args.calibration_id,
                total_steps=int(request["steps"]),
                actual_step_indices=actual_steps,
                attention_action_schedule=tuple(
                    tuple(cell) for cell in request["attention_action_schedule"]
                ),
                source="v19_exact_session_import",
            )
            current_digest = _execution_digest(current_blueprint)
            if workload is None:
                workload = current_workload
                runtime = current_runtime
                blueprint = current_blueprint
                execution_digest = current_digest
            elif (
                current_workload != workload
                or current_runtime != runtime
                or current_digest != execution_digest
                or current_blueprint.action_uses != blueprint.action_uses
            ):
                raise V19CalibrationError(
                    "complete-schedule repeats do not execute one exact plan/build"
                )
            total_samples_ms.append(1000.0 * float(request["total_seconds"]))
            denoise_samples_ms.append(1000.0 * float(phases["denoise"]))
            peak_samples.append(float(request["peak_allocated_gib"]))
        sources.append(_source(report_path, report_index))

    assert workload is not None and runtime is not None and blueprint is not None
    assert execution_digest is not None
    artifact = create_v19_schedule_cost_calibration(
        registry=registry,
        calibration_id=args.calibration_id,
        execution_digest=execution_digest,
        action_ids=(use.action_id for use in blueprint.action_uses),
        workload=workload,
        runtime=runtime,
        total_samples_ms=total_samples_ms,
        denoise_samples_ms=denoise_samples_ms,
        peak_vram_gib_samples=peak_samples,
        sources=sources,
        complete=True,
        minimum_samples=args.minimum_samples,
    )
    artifact.require_planner_ready()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.blueprint_output is not None:
        args.blueprint_output.parent.mkdir(parents=True, exist_ok=True)
        args.blueprint_output.write_text(json.dumps({
            "schema_version": "h3_v19_candidate_blueprint_v1",
            "execution_digest": execution_digest,
            "blueprint": asdict(blueprint),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "execution_digest": execution_digest,
        "workload_digest": workload.digest,
        "runtime_digest": runtime.digest,
        "samples": len(total_samples_ms),
        "p50_ms": artifact.p50_ms,
        "p90_ms": artifact.p90_ms,
        "denoise_p50_ms": artifact.denoise_p50_ms,
        "denoise_p90_ms": artifact.denoise_p90_ms,
        "peak_vram_gib": artifact.peak_vram_gib,
        "planner_ready": artifact.planner_ready,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
