#!/usr/bin/env python3
"""Seal repeated Round229 forecast runs into one strict V19 artifact.

The source reports must come from the same warm Native session/build and must
retain the exact request action schedule.  Only forecast-step wall time is
charged here.  Actual anchor/correction Attention cells are deliberately
priced by the physical action catalog, so this builder rejects reports that
cannot prove the three forecast anchor blocks used the registered Round229
action.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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
    V19ForecastCompositeKey,
    V19ForecastCompositeMeasurement,
    V19RuntimeFingerprint,
    V19SourceRecord,
    build_v19_bootstrap_registry,
    create_v19_forecast_calibration,
    sha256_file,
)


FORECAST_ACTION_ID = "h3.forecast.directional.anchor3.round229.v1"
FORECAST_CALIBRATION_ID = "v19_round229_forecast_composite_cost_v1"
ATTENTION_ACTION_ID = "h3.attention.mtcr_head_rail.round229.v1"
ANCHOR_RUNTIME_ACTION = "forecastfrontier:sparse_topk_0.0625"
ANCHOR_CANONICAL_ACTION = "sparse_topk_0.0625"
EXTRAPOLATOR_ID = "native_depth3_local_directional_v1"
CORRECTION_ID = "next_actual_full_stack_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-report", type=Path, action="append", required=True)
    parser.add_argument(
        "--service-family", choices=("first_last", "reference"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-warm-samples", type=int, default=3)
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


def _relative_source(path: Path, index: int) -> V19SourceRecord:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(SERVE_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise V19CalibrationError(
            f"forecast provenance is outside the release: {resolved}"
        ) from error
    return V19SourceRecord(
        source_id=f"forecast_session_{index}",
        relative_path=relative,
        sha256=sha256_file(resolved),
    )


def _forecast_runs(actual_steps: tuple[int, ...], steps: int) -> tuple[tuple[int, ...], ...]:
    actual = set(actual_steps)
    runs: list[tuple[int, ...]] = []
    current: list[int] = []
    for step in range(steps):
        if step in actual:
            if current:
                runs.append(tuple(current))
                current = []
        else:
            current.append(step)
    if current:
        runs.append(tuple(current))
    if any(run[0] == 0 or run[-1] == steps - 1 for run in runs):
        raise V19CalibrationError(
            "every forecast run requires preceding and following actual steps"
        )
    return tuple(runs)


def _require_forecast_profile(request: dict, actual_steps: tuple[int, ...]) -> None:
    profile = request.get("forecast_profile")
    if not isinstance(profile, dict) or profile.get("mode") != (
        "native_depth3_local_directional"
    ):
        raise V19CalibrationError("request did not execute the registered forecast mode")
    records = profile.get("records")
    if not isinstance(records, list) or len(records) != int(request["steps"]):
        raise V19CalibrationError("forecast telemetry does not cover every solver step")
    expected_actual = set(actual_steps)
    for step, record in enumerate(records):
        if int(record.get("step_index", -1)) != step:
            raise V19CalibrationError("forecast telemetry step order is malformed")
        expected_mode = "actual" if step in expected_actual else "forecast"
        if record.get("mode") != expected_mode or int(record.get("anchor_depth", 0)) != 3:
            raise V19CalibrationError("forecast telemetry contradicts the solver schedule")


def _require_anchor_schedule(
    request: dict,
    runs: tuple[tuple[int, ...], ...],
) -> None:
    raw = request.get("attention_action_schedule")
    if not isinstance(raw, list):
        raise V19CalibrationError(
            "session report lacks the exact request Attention action schedule"
        )
    schedule = {
        (int(step), int(layer)): str(action)
        for step, layer, action in raw
    }
    forecast_steps = {step for run in runs for step in run}
    for step in forecast_steps:
        for layer in range(3):
            if schedule.get((step, layer)) != ANCHOR_RUNTIME_ACTION:
                raise V19CalibrationError(
                    "forecast anchor did not execute the registered Round229 rail"
                )
        if any((step, layer) in schedule for layer in range(3, 50)):
            raise V19CalibrationError(
                "forecast schedule unexpectedly contains a full-stack action"
            )


def _require_round229_execution(request: dict) -> None:
    """Accept either the legacy policy summary or a sealed V19 blueprint.

    A sealed blueprint does not carry the old scheduler's implementation-id
    field.  Its exact action table is nevertheless persisted in the request,
    and every Round229 runtime action name is bound to the fingerprinted
    ``forecastfrontier`` executor.  Requiring both the sealed execution digest
    and that complete table is stronger than trusting a free-form summary.
    """

    execution = request.get("execution_profile")
    if not isinstance(execution, dict):
        raise V19CalibrationError("forecast request lacks execution profile")
    joint = execution.get("joint_acceleration")
    if not isinstance(joint, dict):
        raise V19CalibrationError(
            "forecast report lacks a sealed scheduler execution identity"
        )
    if (
        joint.get("attention_implementation_id")
        == ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION
    ):
        return
    digest = str(joint.get("execution_digest", ""))
    if (
        joint.get("schema_version") != "h3_v19_sealed_blueprint_execution_v1"
        or len(digest) != 64
    ):
        raise V19CalibrationError(
            "forecast report is not bound to the Round229 physical action"
        )
    try:
        int(digest, 16)
    except ValueError as error:
        raise V19CalibrationError(
            "sealed V19 execution digest is malformed"
        ) from error
    schedule = request.get("attention_action_schedule")
    if not isinstance(schedule, list) or not schedule:
        raise V19CalibrationError(
            "sealed V19 execution lacks its exact action table"
        )
    if any(
        not str(row[2]).startswith("forecastfrontier:")
        for row in schedule
    ):
        raise V19CalibrationError(
            "sealed V19 execution contains a non-Round229 Attention action"
        )


def main() -> int:
    args = parse_args()
    if args.minimum_warm_samples < 3:
        raise SystemExit("--minimum-warm-samples must be at least three")
    registry = _registry()
    samples: dict[V19ForecastCompositeKey, list[float]] = defaultdict(list)
    peak_samples: dict[V19ForecastCompositeKey, list[float]] = defaultdict(list)
    workload: V19CalibrationWorkload | None = None
    runtime: V19RuntimeFingerprint | None = None
    sources: list[V19SourceRecord] = []

    for report_index, report_path in enumerate(args.session_report, start=1):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        fingerprint = report.get("v19_runtime_fingerprint")
        if not isinstance(fingerprint, dict):
            raise V19CalibrationError("forecast report lacks a V19 runtime fingerprint")
        current_runtime = V19RuntimeFingerprint(**fingerprint)
        contract = report.get("contract")
        requests = report.get("requests")
        if not isinstance(contract, dict) or not isinstance(requests, list):
            raise V19CalibrationError("forecast source is not a benchmark session report")
        if not requests:
            raise V19CalibrationError("forecast session contains no successful requests")
        for request in requests:
            if request.get("status") == "failed" or "step_seconds" not in request:
                raise V19CalibrationError("forecast calibration contains a failed request")
            steps = int(request["steps"])
            actual_steps = tuple(int(value) for value in request["actual_step_indices"])
            step_seconds = tuple(float(value) for value in request["step_seconds"])
            if len(step_seconds) != steps:
                raise V19CalibrationError("step timing count does not match solver steps")
            execution = request.get("execution_profile")
            _require_round229_execution(request)
            assert isinstance(execution, dict)
            current_workload = V19CalibrationWorkload(
                model_variant=_model_variant(str(report["engine"])),
                service_family=args.service_family,
                width=int(request["width"]),
                height=int(request["height"]),
                frames=int(request["frames"]),
                packed_tokens=int(execution["packed_tokens"]),
                condition_count=int(execution["condition_count"]),
                steps=steps,
                actual_step_indices=actual_steps,
                device_arch=str(fingerprint["device_arch"]),
                sampler=str(contract["sampler"]),
                scheduler=str(contract["scheduler"]),
            )
            if workload is None:
                workload = current_workload
                runtime = current_runtime
            elif current_workload != workload:
                raise V19CalibrationError(
                    "forecast repeats do not share one exact workload bucket"
                )
            elif current_runtime != runtime:
                raise V19CalibrationError(
                    "forecast repeats do not share one runtime/build fingerprint"
                )
            runs = _forecast_runs(actual_steps, steps)
            _require_forecast_profile(request, actual_steps)
            _require_anchor_schedule(request, runs)
            peak = float(request["peak_allocated_gib"])
            for run in runs:
                key = V19ForecastCompositeKey(
                    forecast_step_indices=run,
                    preceding_actual_step=run[0] - 1,
                    following_actual_step=run[-1] + 1,
                    anchor_depth=3,
                    anchor_action_id=ATTENTION_ACTION_ID,
                    anchor_canonical_action=ANCHOR_CANONICAL_ACTION,
                    extrapolator_id=EXTRAPOLATOR_ID,
                    correction_id=CORRECTION_ID,
                )
                # Exclusive scope: the following actual correction is not in
                # this sum and will be charged through the Attention catalog.
                samples[key].append(1000.0 * sum(step_seconds[step] for step in run))
                # Whole-request peak is conservative for the composite and is
                # preferable to an unmeasured allocation-only estimate.
                peak_samples[key].append(peak)
        sources.append(_relative_source(report_path, report_index))

    assert workload is not None and runtime is not None
    measurements = tuple(
        V19ForecastCompositeMeasurement(
            key=key,
            warm_samples_ms=tuple(values),
            peak_vram_gib_samples=tuple(peak_samples[key]),
        )
        for key, values in sorted(samples.items(), key=lambda item: item[0].forecast_step_indices)
    )
    artifact = create_v19_forecast_calibration(
        registry=registry,
        action_id=FORECAST_ACTION_ID,
        calibration_id=FORECAST_CALIBRATION_ID,
        workload=workload,
        runtime=runtime,
        measurements=measurements,
        sources=sources,
        complete=True,
        minimum_warm_samples=args.minimum_warm_samples,
    )
    artifact.require_planner_ready()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "workload": asdict(workload),
        "runtime_digest": runtime.digest,
        "forecast_runs": [list(row.key.forecast_step_indices) for row in measurements],
        "p50_ms": [row.p50_ms for row in measurements],
        "p90_ms": [row.p90_ms for row in measurements],
        "planner_ready": artifact.planner_ready,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
