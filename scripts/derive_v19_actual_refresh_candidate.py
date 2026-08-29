#!/usr/bin/env python3
"""Add one or more bounded Actual refreshes to a V19 trajectory.

The operation preserves every existing Actual Attention cell and replaces a
selected Forecast position with a full 50-layer Actual evaluation.  The new
cell actions are cloned from the next source Actual correction step so the A/B
isolates trajectory refresh rather than inventing a second Attention policy.
The result remains an experimental proposal until exact cost measurement and
Human audiovisual review are complete.
"""

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
    ROUND229_FORECAST_ANCHOR,
    V19ActionUse,
    V19CandidateBlueprint,
    V19HumanConstraintPolicy,
    V19PlanningError,
    blueprint_from_runtime_schedule,
    contiguous_forecast_runs,
    evaluate_v19_human_constraints,
    load_v19_candidate_blueprint,
    runtime_schedule_from_blueprint,
    save_v19_candidate_blueprint,
    v19_round02_av_motion_screening_policy,
    v19_blueprint_execution_digest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-blueprint", type=Path, required=True)
    parser.add_argument("--output-blueprint", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument(
        "--add-actual-step",
        type=int,
        action="append",
        required=True,
        help="repeat to convert one or more source Forecast positions to Actual",
    )
    parser.add_argument(
        "--maximum-forecast-run",
        type=int,
        default=3,
        help="fail closed if the refreshed trajectory exceeds this run length",
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


def derive_actual_refresh_candidate(
    source: V19CandidateBlueprint,
    *,
    candidate_id: str,
    add_actual_steps: tuple[int, ...],
    maximum_forecast_run: int = 3,
    constraint_policy: V19HumanConstraintPolicy | None = None,
) -> tuple[V19CandidateBlueprint, dict[str, object]]:
    total_steps = _total_steps(source)
    source_actual = _actual_steps(source)
    source_actual_set = set(source_actual)
    add = tuple(sorted(set(int(step) for step in add_actual_steps)))
    if not add:
        raise V19PlanningError("Actual refresh derivation requires a new step")
    if any(step < 0 or step >= total_steps for step in add):
        raise V19PlanningError("added Actual step lies outside the trajectory")
    if any(step in source_actual_set for step in add):
        raise V19PlanningError("added Actual steps must be source Forecast steps")
    if maximum_forecast_run <= 0:
        raise V19PlanningError("maximum Forecast run must be positive")

    derived_actual = tuple(sorted((*source_actual, *add)))
    derived_runs = contiguous_forecast_runs(
        total_steps=total_steps,
        actual_step_indices=derived_actual,
    )
    longest = max((len(run) for run in derived_runs), default=0)
    if longest > maximum_forecast_run:
        raise V19PlanningError(
            f"refreshed Forecast run {longest} exceeds limit {maximum_forecast_run}"
        )

    source_runtime = {
        (step, layer): action
        for step, layer, action in runtime_schedule_from_blueprint(source)
    }
    schedule = {
        (step, layer): source_runtime[(step, layer)]
        for step in source_actual
        for layer in range(50)
    }
    clone_steps: dict[int, int] = {}
    for step in add:
        following = next(
            (actual for actual in source_actual if actual > step),
            None,
        )
        if following is None:
            raise V19PlanningError(
                "Actual refresh requires a following source correction step"
            )
        clone_steps[step] = following
        for layer in range(50):
            schedule[(step, layer)] = source_runtime[(following, layer)]

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
        source="v19_experimental_actual_refresh_derivation_v1",
    )
    policy = constraint_policy or v19_round02_av_motion_screening_policy()
    constraint_report = evaluate_v19_human_constraints(candidate, policy)
    if not constraint_report.proposal_eligible:
        raise V19PlanningError(
            "refreshed candidate violates Human screening constraints: "
            + "; ".join(constraint_report.rejection_reasons)
        )
    summary = {
        "schema_version": "h3_v19_experimental_actual_refresh_derivation_v1",
        "warning": (
            "This trajectory mechanism has no matched physical cost or Human "
            "risk calibration. Static constraints only permit an E2E A/B."
        ),
        "source_candidate_id": source.candidate_id,
        "source_execution_digest": v19_blueprint_execution_digest(source),
        "candidate_id": candidate.candidate_id,
        "execution_digest": v19_blueprint_execution_digest(candidate),
        "total_steps": total_steps,
        "source_actual_step_indices": list(source_actual),
        "actual_step_indices": list(derived_actual),
        "added_actual_step_indices": list(add),
        "cloned_attention_from_source_actual": {
            str(step): clone_steps[step] for step in add
        },
        "forecast_runs": [list(run) for run in derived_runs],
        "maximum_forecast_run": longest,
        "constraint_policy": asdict(policy),
        "constraint_report": asdict(constraint_report),
        "human_risk_calibrated": False,
        "forecast_composite_cost_calibrated": False,
        "end_to_end_cost_calibrated": False,
    }
    return candidate, summary


def main() -> int:
    args = parse_args()
    source = load_v19_candidate_blueprint(args.source_blueprint)
    candidate, summary = derive_actual_refresh_candidate(
        source,
        candidate_id=args.candidate_id,
        add_actual_steps=tuple(args.add_actual_step),
        maximum_forecast_run=args.maximum_forecast_run,
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
        "release_eligible": summary["constraint_report"]["release_eligible"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
