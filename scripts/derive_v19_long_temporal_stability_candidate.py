#!/usr/bin/env python3
"""Derive one long-video post-Forecast temporal recovery candidate."""

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
from h3serve.native_engine.planner.v19_long_temporal_stability import (  # noqa: E402
    V19LongTemporalStabilitySpec,
    build_v19_long_temporal_stability_shield,
)


def _layers(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    layers: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            left, right = (int(item) for item in part.split("-", 1))
            if right < left:
                raise argparse.ArgumentTypeError("layer ranges must be ascending")
            layers.update(range(left, right + 1))
        elif part:
            layers.add(int(part))
    result = tuple(sorted(layers))
    if any(layer < 0 or layer >= 50 for layer in result):
        raise argparse.ArgumentTypeError("layers must lie inside [0, 50)")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-blueprint", type=Path, required=True)
    parser.add_argument("--output-blueprint", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--minimum-forecast-run", type=int, default=2)
    parser.add_argument(
        "--recovery-action",
        choices=("sparse_topk_0.1", "sparse_topk_0.25", "sparse_topk_0.5", "dense"),
        default="sparse_topk_0.25",
    )
    parser.add_argument("--structural-layers", type=_layers, default=())
    parser.add_argument(
        "--structural-action",
        choices=("sparse_topk_0.1", "sparse_topk_0.25", "sparse_topk_0.5", "dense"),
        default="sparse_topk_0.25",
    )
    parser.add_argument("--derivation", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = load_v19_candidate_blueprint(args.source_blueprint)
    spec = V19LongTemporalStabilitySpec(
        minimum_forecast_run=args.minimum_forecast_run,
        recovery_action=args.recovery_action,
        structural_layers=args.structural_layers,
        structural_action=args.structural_action,
    )
    result = build_v19_long_temporal_stability_shield(
        source,
        candidate_id=args.candidate_id,
        spec=spec,
    )
    save_v19_candidate_blueprint(args.output_blueprint, result.blueprint)
    derivation = args.derivation or args.output_blueprint.with_name("derivation.json")
    derivation.parent.mkdir(parents=True, exist_ok=True)
    derivation.write_text(json.dumps({
        "schema_version": "h3_v19_long_temporal_stability_derivation_v1",
        "warning": "Mechanism-derived proposal; Human continuous playback is required.",
        "prompt_semantics_used": False,
        "model_weights_changed": False,
        "source_blueprint": str(args.source_blueprint.resolve()),
        "source_execution_digest": result.source_execution_digest,
        "candidate_execution_digest": v19_blueprint_execution_digest(result.blueprint),
        "spec": asdict(spec),
        "actual_step_indices": result.actual_step_indices,
        "forecast_runs": result.forecast_runs,
        "recovery_actual_step_indices": result.recovery_actual_step_indices,
        "source_action_cell_counts": dict(result.source_action_cell_counts),
        "candidate_action_cell_counts": dict(result.candidate_action_cell_counts),
        "recovery_upgraded_cells": result.recovery_upgraded_cells,
        "structural_upgraded_cells": result.structural_upgraded_cells,
        "constraint_report": asdict(result.constraint_report),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "blueprint": str(args.output_blueprint.resolve()),
        "derivation": str(derivation.resolve()),
        "execution_digest": v19_blueprint_execution_digest(result.blueprint),
        "recovery_actual_step_indices": result.recovery_actual_step_indices,
        "candidate_action_cell_counts": dict(result.candidate_action_cell_counts),
        "proposal_eligible": result.constraint_report.proposal_eligible,
        "release_eligible": result.constraint_report.release_eligible,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
