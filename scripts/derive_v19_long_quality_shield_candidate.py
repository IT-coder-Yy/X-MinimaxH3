#!/usr/bin/env python3
"""Derive and seal one task-independent long-video quality-shield proposal."""

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
from h3serve.native_engine.planner.v19_long_quality import (  # noqa: E402
    V19LongQualityShieldSpec,
    build_v19_long_quality_shield,
)


def _layers(value: str) -> tuple[int, ...]:
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = (int(item) for item in part.split("-", 1))
            if right < left:
                raise argparse.ArgumentTypeError("layer ranges must be ascending")
            result.update(range(left, right + 1))
        else:
            result.add(int(part))
    layers = tuple(sorted(result))
    if not layers or any(layer < 0 or layer >= 50 for layer in layers):
        raise argparse.ArgumentTypeError("layers must lie inside [0, 50)")
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-blueprint", type=Path, required=True)
    parser.add_argument("--output-blueprint", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--core-layers", type=_layers, default=_layers("39-43,45"))
    parser.add_argument(
        "--core-action",
        choices=(
            "sparse_topk_0.1",
            "sparse_topk_0.25",
            "sparse_topk_0.5",
            "dense",
        ),
        default="sparse_topk_0.25",
    )
    parser.add_argument("--terminal-actual-count", type=int, default=3)
    parser.add_argument(
        "--terminal-action",
        choices=("sparse_topk_0.1", "sparse_topk_0.25", "sparse_topk_0.5", "dense"),
        default="sparse_topk_0.25",
    )
    parser.add_argument("--derivation", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = load_v19_candidate_blueprint(args.source_blueprint)
    spec = V19LongQualityShieldSpec(
        core_layers=args.core_layers,
        core_action=args.core_action,
        terminal_actual_count=args.terminal_actual_count,
        terminal_action=args.terminal_action,
    )
    result = build_v19_long_quality_shield(
        source,
        candidate_id=args.candidate_id,
        spec=spec,
    )
    save_v19_candidate_blueprint(args.output_blueprint, result.blueprint)
    derivation = args.derivation or args.output_blueprint.with_name("derivation.json")
    derivation.parent.mkdir(parents=True, exist_ok=True)
    derivation.write_text(
        json.dumps(
            {
                "schema_version": "h3_v19_long_quality_shield_derivation_v1",
                "warning": (
                    "Static proposal eligibility is not a Human quality claim."
                ),
                "prompt_semantics_used": False,
                "model_weights_changed": False,
                "source_blueprint": str(args.source_blueprint.resolve()),
                "source_execution_digest": result.source_execution_digest,
                "candidate_execution_digest": v19_blueprint_execution_digest(
                    result.blueprint
                ),
                "spec": asdict(spec),
                "actual_step_indices": result.actual_step_indices,
                "terminal_actual_step_indices": (
                    result.terminal_actual_step_indices
                ),
                "source_action_cell_counts": dict(
                    result.source_action_cell_counts
                ),
                "candidate_action_cell_counts": dict(
                    result.candidate_action_cell_counts
                ),
                "core_upgraded_cells": result.core_upgraded_cells,
                "terminal_upgraded_cells": result.terminal_upgraded_cells,
                "constraint_report": asdict(result.constraint_report),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "blueprint": str(args.output_blueprint.resolve()),
                "derivation": str(derivation.resolve()),
                "candidate_id": result.blueprint.candidate_id,
                "execution_digest": v19_blueprint_execution_digest(
                    result.blueprint
                ),
                "candidate_action_cell_counts": dict(
                    result.candidate_action_cell_counts
                ),
                "proposal_eligible": result.constraint_report.proposal_eligible,
                "release_eligible": result.constraint_report.release_eligible,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
