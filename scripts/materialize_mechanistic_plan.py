#!/usr/bin/env python3
"""Materialize one schedule-independent mechanistic H3 plan for GPU A/B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SERVE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVE_ROOT))

from h3serve.native_engine.planner import (  # noqa: E402
    H3MechanisticControlModel,
    H3MechanisticWorkload,
    blueprint_from_runtime_schedule,
    save_v19_candidate_blueprint,
    v19_blueprint_execution_digest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output-blueprint", type=Path, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--packed-tokens", type=int, required=True)
    parser.add_argument("--video-tokens", type=int, required=True)
    parser.add_argument("--condition-count", type=int, default=0)
    parser.add_argument(
        "--service-family", choices=("first_last", "reference"), default="first_last"
    )
    parser.add_argument("--maximum-cost-ms", type=float, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workload = H3MechanisticWorkload(
        total_steps=args.steps,
        packed_tokens=args.packed_tokens,
        video_tokens=args.video_tokens,
        condition_count=args.condition_count,
        service_family=args.service_family,
    )
    model = H3MechanisticControlModel(workload)
    plan = model.plan_for_cost_budget(maximum_cost_ms=args.maximum_cost_ms)
    verification = model.verify(plan)
    if not verification.valid:
        raise RuntimeError("mechanistic plan failed replay: " + "; ".join(
            verification.reasons
        ))
    blueprint = blueprint_from_runtime_schedule(
        candidate_id=args.candidate_id,
        total_steps=workload.total_steps,
        actual_step_indices=plan.actual_step_indices,
        attention_action_schedule=plan.runtime_action_schedule(),
        source="h3_mechanistic_same_speed_ab_v1",
    )
    save_v19_candidate_blueprint(args.output_blueprint, blueprint)
    document = {
        "schema_version": "h3_mechanistic_materialized_plan_v1",
        "claim_scope": (
            "same-speed exploratory mechanism A/B; no Human admission and "
            "not a release creator-dial endpoint"
        ),
        "candidate_id": args.candidate_id,
        "execution_digest": v19_blueprint_execution_digest(blueprint),
        "blueprint": str(args.output_blueprint.resolve()),
        "plan": plan.to_dict(),
        "verification": {
            "valid": verification.valid,
            "reasons": list(verification.reasons),
        },
        "identification_readiness": "720p5_single_seed_phase_shape_only",
        "historical_schedule_used": False,
    }
    target = args.output_plan.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "blueprint": str(args.output_blueprint.resolve()),
        "plan": str(target),
        "predicted_cost_ms": plan.predicted_cost_ms,
        "modeled_risk": plan.modeled_risk.total,
        "actual_step_indices": list(plan.actual_step_indices),
        "maximum_forecast_run": plan.maximum_forecast_run,
        "execution_digest": document["execution_digest"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
