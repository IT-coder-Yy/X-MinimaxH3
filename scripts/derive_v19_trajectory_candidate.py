#!/usr/bin/env python3
"""Derive an explicitly experimental V19 Actual/Forecast trajectory.

This tool does not claim that a newly extended Forecast run is calibrated or
Human-safe. It preserves the physical Attention actions of a source blueprint,
rebuilds the required depth-3 Forecast anchors, optionally strengthens the
causal layers of the immediate correction step, and seals the resulting
execution digest for a real A/B run.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


SERVE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVE_ROOT))

from h3serve.native_engine.planner import (  # noqa: E402
    ROUND229_FORECAST_ANCHOR,
    V19ActionUse,
    V19CandidateBlueprint,
    V19PlanningError,
    blueprint_from_runtime_schedule,
    contiguous_forecast_runs,
    load_v19_candidate_blueprint,
    runtime_schedule_from_blueprint,
    save_v19_candidate_blueprint,
    v19_blueprint_execution_digest,
)


CAUSAL_LAYERS = tuple((*range(30, 44), 45))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-blueprint", type=Path, required=True)
    parser.add_argument("--output-blueprint", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument(
        "--remove-actual-step",
        type=int,
        action="append",
        required=True,
        help="repeat to turn one or more source Actual steps into Forecast steps",
    )
    parser.add_argument(
        "--maximum-forecast-run",
        type=int,
        default=3,
        help="fail closed if the derived trajectory exceeds this run length",
    )
    parser.add_argument(
        "--correction-causal-topk",
        choices=("0.0625", "0.1", "0.25", "0.5"),
        help=(
            "minimum Round229 Top-K for layers 30--43/45 on the Actual step "
            "immediately following every newly extended Forecast run"
        ),
    )
    return parser.parse_args()


def _actual_steps(blueprint: V19CandidateBlueprint) -> tuple[int, ...]:
    return tuple(sorted({
        step
        for use in blueprint.action_uses
        if isinstance(use, V19ActionUse)
        for step in use.step_indices
    }))


def _total_steps(blueprint: V19CandidateBlueprint) -> int:
    runtime = runtime_schedule_from_blueprint(blueprint)
    return 1 + max(step for step, _layer, _action in runtime)


def _topk(action: str) -> float:
    if action == "dense":
        return 1.0
    try:
        return float(action.rsplit("sparse_topk_", 1)[1])
    except (IndexError, ValueError) as error:
        raise V19PlanningError(f"cannot read runtime Top-K action: {action}") from error


def derive_candidate(
    source: V19CandidateBlueprint,
    *,
    candidate_id: str,
    remove_actual_steps: tuple[int, ...],
    maximum_forecast_run: int = 3,
    correction_causal_topk: float | None = None,
) -> tuple[V19CandidateBlueprint, dict[str, object]]:
    total_steps = _total_steps(source)
    source_actual = _actual_steps(source)
    remove = tuple(sorted(set(int(step) for step in remove_actual_steps)))
    if not remove or any(step not in source_actual for step in remove):
        raise V19PlanningError("removed steps must be source Actual steps")
    derived_actual = tuple(step for step in source_actual if step not in remove)
    if not derived_actual or derived_actual[0] != 0 or derived_actual[-1] != total_steps - 1:
        raise V19PlanningError("derived trajectory must retain first and final Actual steps")
    if len(derived_actual) < 3 or derived_actual[:3] != (0, 1, 2):
        raise V19PlanningError("derived trajectory must retain three opening Actual anchors")
    if maximum_forecast_run <= 0:
        raise V19PlanningError("maximum Forecast run must be positive")

    source_runs = contiguous_forecast_runs(
        total_steps=total_steps,
        actual_step_indices=source_actual,
    )
    derived_runs = contiguous_forecast_runs(
        total_steps=total_steps,
        actual_step_indices=derived_actual,
    )
    longest = max(map(len, derived_runs), default=0)
    if longest > maximum_forecast_run:
        raise V19PlanningError(
            f"derived Forecast run {longest} exceeds limit {maximum_forecast_run}"
        )

    source_runtime = runtime_schedule_from_blueprint(source)
    schedule = {
        (step, layer): action
        for step, layer, action in source_runtime
        if step in set(derived_actual)
    }
    expected_actual = {
        (step, layer) for step in derived_actual for layer in range(50)
    }
    if set(schedule) != expected_actual:
        raise V19PlanningError("source blueprint does not cover every retained Actual cell")

    source_run_sets = tuple(set(run) for run in source_runs)
    extended_runs = tuple(
        run
        for run in derived_runs
        if not any(set(run).issubset(source_run) for source_run in source_run_sets)
    )
    correction_steps = tuple(run[-1] + 1 for run in extended_runs)
    if correction_causal_topk is not None:
        canonical = f"forecastfrontier:sparse_topk_{correction_causal_topk:g}"
        for step in correction_steps:
            for layer in CAUSAL_LAYERS:
                current = schedule[(step, layer)]
                if _topk(current) < correction_causal_topk:
                    schedule[(step, layer)] = canonical

    for run in derived_runs:
        for step in run:
            for layer in range(3):
                schedule[(step, layer)] = ROUND229_FORECAST_ANCHOR

    candidate = blueprint_from_runtime_schedule(
        candidate_id=candidate_id,
        total_steps=total_steps,
        actual_step_indices=derived_actual,
        attention_action_schedule=tuple(
            (step, layer, action)
            for (step, layer), action in sorted(schedule.items())
        ),
        source="v19_experimental_trajectory_derivation_v1",
    )
    counts = Counter(
        action
        for step, _layer, action in runtime_schedule_from_blueprint(candidate)
        if step in set(derived_actual)
    )
    summary = {
        "schema_version": "h3_v19_experimental_trajectory_derivation_v1",
        "warning": (
            "Extended Forecast runs are research candidates without matched "
            "Human-risk or exact composite-cost calibration."
        ),
        "source_candidate_id": source.candidate_id,
        "source_execution_digest": v19_blueprint_execution_digest(source),
        "candidate_id": candidate.candidate_id,
        "execution_digest": v19_blueprint_execution_digest(candidate),
        "total_steps": total_steps,
        "source_actual_step_indices": list(source_actual),
        "actual_step_indices": list(derived_actual),
        "removed_actual_step_indices": list(remove),
        "forecast_runs": [list(run) for run in derived_runs],
        "maximum_forecast_run": longest,
        "extended_forecast_runs": [list(run) for run in extended_runs],
        "correction_steps": list(correction_steps),
        "correction_causal_topk": correction_causal_topk,
        "actual_attention_action_counts": dict(sorted(counts.items())),
        "human_risk_calibrated": False,
        "forecast_composite_cost_calibrated": False,
    }
    return candidate, summary


def main() -> int:
    args = parse_args()
    source = load_v19_candidate_blueprint(args.source_blueprint)
    candidate, summary = derive_candidate(
        source,
        candidate_id=args.candidate_id,
        remove_actual_steps=tuple(args.remove_actual_step),
        maximum_forecast_run=args.maximum_forecast_run,
        correction_causal_topk=(
            None
            if args.correction_causal_topk is None
            else float(args.correction_causal_topk)
        ),
    )
    save_v19_candidate_blueprint(args.output_blueprint, candidate)
    summary_path = args.summary or args.output_blueprint.with_name(
        f"{args.output_blueprint.stem}_derivation.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "blueprint": str(args.output_blueprint.resolve()),
        "summary": str(summary_path.resolve()),
        "execution_digest": summary["execution_digest"],
        "actual_step_indices": summary["actual_step_indices"],
        "forecast_runs": summary["forecast_runs"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
