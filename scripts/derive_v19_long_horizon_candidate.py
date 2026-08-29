#!/usr/bin/env python3
"""Materialize an evidence-bounded long-video V19 replay candidate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


SERVE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVE_ROOT))

from h3serve.native_engine.planner import (  # noqa: E402
    build_v19_long_horizon_round188_replay,
    evaluate_v19_human_constraints,
    runtime_schedule_from_blueprint,
    save_v19_candidate_blueprint,
    v19_blueprint_execution_digest,
    v19_long_horizon_screening_policy,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-blueprint", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--total-steps", type=int, default=20)
    parser.add_argument("--acceleration", type=float, default=75.0)
    parser.add_argument("--provenance-report", type=Path)
    parser.add_argument("--provenance-video", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    blueprint = build_v19_long_horizon_round188_replay(
        candidate_id=args.candidate_id,
        total_steps=args.total_steps,
        acceleration=args.acceleration,
    )
    policy = v19_long_horizon_screening_policy(args.total_steps)
    report = evaluate_v19_human_constraints(blueprint, policy)
    runtime = runtime_schedule_from_blueprint(blueprint)
    action_counts: dict[str, int] = {}
    for _step, _layer, action in runtime:
        action_counts[action] = action_counts.get(action, 0) + 1

    provenance = []
    for role, path in (
        ("round188_hot_session_report", args.provenance_report),
        ("round188_human_reviewed_video", args.provenance_video),
    ):
        if path is None:
            continue
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        provenance.append({
            "role": role,
            "path": str(resolved),
            "sha256": _sha256(resolved),
            "size_bytes": resolved.stat().st_size,
        })

    summary = {
        "schema_version": "h3_v19_long_horizon_candidate_derivation_v1",
        "warning": (
            "The historical Round188 mechanism has positive 720p15 Human "
            "evidence, but this exact CUDA/PyTorch runtime artifact remains "
            "unevaluated and cannot inherit release eligibility."
        ),
        "candidate_id": blueprint.candidate_id,
        "execution_digest": v19_blueprint_execution_digest(blueprint),
        "user_controls": {
            "sampling_steps": args.total_steps,
            "acceleration": args.acceleration,
        },
        "actual_step_indices": list(report.actual_step_indices),
        "forecast_runs": [list(run) for run in report.forecast_runs],
        "action_cell_counts": action_counts,
        "constraint_policy": asdict(policy),
        "constraint_report": asdict(report),
        "historical_evidence": provenance,
        "new_runtime_end_to_end_cost_calibrated": False,
        "new_runtime_human_quality_calibrated": False,
    }
    save_v19_candidate_blueprint(args.output_blueprint, blueprint)
    summary_path = args.summary or args.output_blueprint.with_name(
        "derivation.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "blueprint": str(args.output_blueprint.resolve()),
        "summary": str(summary_path.resolve()),
        "execution_digest": summary["execution_digest"],
        "actual_step_indices": summary["actual_step_indices"],
        "forecast_runs": summary["forecast_runs"],
        "proposal_eligible": report.proposal_eligible,
        "release_eligible": report.release_eligible,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
