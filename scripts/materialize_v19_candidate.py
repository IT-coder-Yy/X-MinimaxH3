#!/usr/bin/env python3
"""Bind one V19 blueprint to exact physical and optional Human evidence."""

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
    V19CalibrationCatalog,
    V19CandidateFactory,
    V19ForecastCalibrationCatalog,
    V19PlanningRequest,
    V19RiskCalibrationCatalog,
    V19ScheduleCostCatalog,
    V19WorkloadContext,
    V19HumanRiskVector,
    build_v19_bootstrap_registry,
    load_v19_action_calibration,
    load_v19_candidate_blueprint,
    load_v19_forecast_calibration,
    load_v19_plan_risk_calibration,
    load_v19_schedule_cost_calibration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-calibration", type=Path, action="append", required=True)
    parser.add_argument("--forecast-calibration", type=Path)
    parser.add_argument("--schedule-calibration", type=Path, required=True)
    parser.add_argument("--risk-calibration", type=Path)
    parser.add_argument("--blueprint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _registry():
    return build_v19_bootstrap_registry(implementation_ids={
        "fixed_topk": FIXED_TOPK_ACTION_IMPLEMENTATION,
        "round215": ROUND215_ACTION_IMPLEMENTATION,
        "round188": ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
        "round228": ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
        "round229": ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION,
    })


def main() -> int:
    args = parse_args()
    registry = _registry()
    attention_catalog = V19CalibrationCatalog(registry)
    workload = None
    runtime = None
    for path in args.attention_calibration:
        artifact = load_v19_action_calibration(
            path,
            registry=registry,
            expected_workload=workload,
            expected_runtime=runtime,
        )
        workload = artifact.workload
        runtime = artifact.runtime
        attention_catalog.add(artifact)
    assert workload is not None and runtime is not None
    forecast_catalog = None
    if args.forecast_calibration is not None:
        forecast_catalog = V19ForecastCalibrationCatalog(registry)
        forecast_catalog.add(load_v19_forecast_calibration(
            args.forecast_calibration,
            registry=registry,
            expected_workload=workload,
            expected_runtime=runtime,
        ))
    schedule_catalog = V19ScheduleCostCatalog(registry)
    schedule_catalog.add(load_v19_schedule_cost_calibration(
        args.schedule_calibration, registry=registry
    ))
    risk_catalog = None
    if args.risk_calibration is not None:
        risk_catalog = V19RiskCalibrationCatalog(registry)
        risk_catalog.add(load_v19_plan_risk_calibration(
            args.risk_calibration, registry=registry
        ))
    blueprint = load_v19_candidate_blueprint(args.blueprint)
    request = V19PlanningRequest(
        workload=V19WorkloadContext(
            model_variant=workload.model_variant,
            service_family=workload.service_family,
            packed_tokens=workload.packed_tokens,
            condition_count=workload.condition_count,
            width=workload.width,
            height=workload.height,
            frames=workload.frames,
            steps=workload.steps,
            actual_step_indices=workload.actual_step_indices,
            device_arch=workload.device_arch,
            sampler=workload.sampler,
            scheduler=workload.scheduler,
        ),
        maximum_cost_p90_ms=1.0e12,
        maximum_peak_vram_gib=24.0,
        runtime=runtime,
        risk_limits=V19HumanRiskVector(*(1.0 for _ in range(7))),
    )
    result = V19CandidateFactory(
        registry,
        attention_catalog,
        forecast_catalog=forecast_catalog,
        schedule_cost_catalog=schedule_catalog,
        risk_catalog=risk_catalog,
    ).materialize(
        request,
        blueprint,
        require_end_to_end_cost=True,
        require_human_risk=args.risk_calibration is not None,
    )
    document = {
        "schema_version": "h3_v19_materialized_candidate_v1",
        "candidate": asdict(result.candidate),
        "end_to_end_cost_calibrated": result.end_to_end_cost_calibrated,
        "human_risk_calibrated": result.human_risk_calibrated,
        "release_certifiable": (
            result.end_to_end_cost_calibrated and result.human_risk_calibrated
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "candidate_id": result.candidate.candidate_id,
        "cost_p50_ms": result.candidate.predicted_cost_p50_ms,
        "cost_p90_ms": result.candidate.predicted_cost_p90_ms,
        "peak_vram_gib": result.candidate.predicted_peak_vram_gib,
        "human_risk_calibrated": result.human_risk_calibrated,
        "release_certifiable": document["release_certifiable"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
