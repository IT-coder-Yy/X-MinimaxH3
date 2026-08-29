#!/usr/bin/env python3
"""Derive a hybrid with exact donor Attention at fixed Actual boundaries."""

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
from h3serve.native_engine.planner.v19_long_boundary_replay import (  # noqa: E402
    V19LongBoundaryReplaySpec,
    build_v19_long_boundary_replay,
)


def _steps(value: str) -> tuple[int, ...]:
    try:
        result = tuple(sorted({int(part.strip()) for part in value.split(",")}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("steps must be comma-separated integers") from error
    if not result or any(step < 0 for step in result):
        raise argparse.ArgumentTypeError("steps must be non-negative")
    return result


def _layers(value: str) -> tuple[int, ...]:
    layers: set[int] = set()
    try:
        for part in value.split(","):
            part = part.strip()
            if "-" in part:
                left, right = (int(item) for item in part.split("-", 1))
                if right < left:
                    raise argparse.ArgumentTypeError(
                        "layer ranges must be ascending"
                    )
                layers.update(range(left, right + 1))
            elif part:
                layers.add(int(part))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "layers must be comma-separated integers or ranges"
        ) from error
    result = tuple(sorted(layers))
    if not result or any(layer < 0 or layer >= 50 for layer in result):
        raise argparse.ArgumentTypeError("layers must lie inside [0, 50)")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-blueprint", type=Path, required=True)
    parser.add_argument("--donor-blueprint", type=Path, required=True)
    parser.add_argument("--output-blueprint", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--replay-actual-steps", type=_steps, required=True)
    parser.add_argument(
        "--replay-layers", type=_layers, default=tuple(range(50))
    )
    parser.add_argument("--derivation", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = load_v19_candidate_blueprint(args.source_blueprint)
    donor = load_v19_candidate_blueprint(args.donor_blueprint)
    spec = V19LongBoundaryReplaySpec(
        replay_actual_step_indices=args.replay_actual_steps,
        replay_layer_indices=args.replay_layers,
    )
    result = build_v19_long_boundary_replay(
        source,
        donor,
        candidate_id=args.candidate_id,
        spec=spec,
    )
    save_v19_candidate_blueprint(args.output_blueprint, result.blueprint)
    derivation = args.derivation or args.output_blueprint.with_name("derivation.json")
    derivation.parent.mkdir(parents=True, exist_ok=True)
    derivation.write_text(json.dumps({
        "schema_version": "h3_v19_long_boundary_replay_derivation_v1",
        "warning": "A reviewed donor does not transfer Human approval to this hybrid.",
        "prompt_semantics_used": False,
        "model_weights_changed": False,
        "source_blueprint": str(args.source_blueprint.resolve()),
        "donor_blueprint": str(args.donor_blueprint.resolve()),
        "source_execution_digest": result.source_execution_digest,
        "donor_execution_digest": result.donor_execution_digest,
        "candidate_execution_digest": v19_blueprint_execution_digest(result.blueprint),
        "spec": asdict(spec),
        "actual_step_indices": result.actual_step_indices,
        "replayed_cells": result.replayed_cells,
        "physically_changed_cells": result.physically_changed_cells,
        "source_action_cell_counts": dict(result.source_action_cell_counts),
        "candidate_action_cell_counts": dict(result.candidate_action_cell_counts),
        "constraint_report": asdict(result.constraint_report),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "blueprint": str(args.output_blueprint.resolve()),
        "derivation": str(derivation.resolve()),
        "execution_digest": v19_blueprint_execution_digest(result.blueprint),
        "replay_actual_step_indices": result.replay_actual_step_indices,
        "replay_layer_indices": result.replay_layer_indices,
        "physically_changed_cells": result.physically_changed_cells,
        "candidate_action_cell_counts": dict(result.candidate_action_cell_counts),
        "proposal_eligible": result.constraint_report.proposal_eligible,
        "release_eligible": result.constraint_report.release_eligible,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
