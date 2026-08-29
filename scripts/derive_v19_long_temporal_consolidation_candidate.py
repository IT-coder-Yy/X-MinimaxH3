#!/usr/bin/env python3
"""Derive a task-independent long-video temporal consolidation candidate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


SERVE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVE_ROOT))

from h3serve.native_engine.planner.v19_candidates import (  # noqa: E402
    load_v19_candidate_blueprint,
    save_v19_candidate_blueprint,
    v19_blueprint_execution_digest,
)
from h3serve.native_engine.planner.v19_long_temporal_consolidation import (  # noqa: E402
    V19LongTemporalConsolidationSpec,
    build_v19_long_temporal_consolidation,
)


def _steps(value: str) -> tuple[int, ...]:
    try:
        result = tuple(sorted({int(part.strip()) for part in value.split(",")}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("steps must be comma-separated integers") from error
    if not result or any(step < 0 for step in result):
        raise argparse.ArgumentTypeError("steps must be non-negative")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-blueprint", type=Path, required=True)
    parser.add_argument("--output-blueprint", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument(
        "--target-actual-steps",
        type=_steps,
        default="0,1,2,3,4,8,12,15,18,19",
    )
    parser.add_argument("--maximum-forecast-run", type=int, default=3)
    parser.add_argument(
        "--recovery-action",
        choices=("sparse_topk_0.1", "sparse_topk_0.25", "sparse_topk_0.5", "dense"),
    )
    parser.add_argument("--minimum-recovery-forecast-run", type=int, default=2)
    parser.add_argument("--derivation", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = load_v19_candidate_blueprint(args.source_blueprint)
    spec = V19LongTemporalConsolidationSpec(
        target_actual_step_indices=args.target_actual_steps,
        maximum_forecast_run=args.maximum_forecast_run,
        recovery_action=args.recovery_action,
        minimum_recovery_forecast_run=args.minimum_recovery_forecast_run,
    )
    result = build_v19_long_temporal_consolidation(
        source,
        candidate_id=args.candidate_id,
        spec=spec,
    )
    save_v19_candidate_blueprint(args.output_blueprint, result.blueprint)
    derivation = args.derivation or args.output_blueprint.with_name("derivation.json")
    derivation.parent.mkdir(parents=True, exist_ok=True)
    derivation.write_text(json.dumps({
        "schema_version": "h3_v19_long_temporal_consolidation_derivation_v1",
        "warning": "Mechanism-derived proposal; Human continuous playback is required.",
        "prompt_semantics_used": False,
        "model_weights_changed": False,
        "source_blueprint": str(args.source_blueprint.resolve()),
        "source_execution_digest": result.source_execution_digest,
        "candidate_execution_digest": v19_blueprint_execution_digest(result.blueprint),
        "spec": asdict(spec),
        "source_actual_step_indices": result.source_actual_step_indices,
        "actual_step_indices": result.actual_step_indices,
        "forecast_runs": result.forecast_runs,
        "cloned_from_source_actual": dict(result.cloned_from_source_actual),
        "recovery_actual_step_indices": result.recovery_actual_step_indices,
        "recovery_upgraded_cells": result.recovery_upgraded_cells,
        "source_action_cell_counts": dict(result.source_action_cell_counts),
        "candidate_action_cell_counts": dict(result.candidate_action_cell_counts),
        "constraint_report": asdict(result.constraint_report),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "blueprint": str(args.output_blueprint.resolve()),
        "derivation": str(derivation.resolve()),
        "execution_digest": v19_blueprint_execution_digest(result.blueprint),
        "actual_step_indices": result.actual_step_indices,
        "forecast_runs": result.forecast_runs,
        "recovery_actual_step_indices": result.recovery_actual_step_indices,
        "recovery_upgraded_cells": result.recovery_upgraded_cells,
        "candidate_action_cell_counts": dict(result.candidate_action_cell_counts),
        "proposal_eligible": result.constraint_report.proposal_eligible,
        "release_eligible": result.constraint_report.release_eligible,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
