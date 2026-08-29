#!/usr/bin/env python3
"""Build schedule-controlled probes below the current V24 Human anchor.

The script is an experiment generator, not a deployment policy.  Starting
from one reviewed physical blueprint, it repeatedly chooses the smallest
modeled-risk increase per unit of compute saved across two coupled moves:

* downgrade one actual-step Attention cell by one measured fidelity rank; or
* demote one Actual evaluation to directional Forecast while preserving the
  physical history, terminal correction and a bounded Forecast run.

No prompt text, scene label or output resolution participates in the search.
The same generated blueprints can therefore be replayed at 480p and 720p.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from h3serve.native_engine.planner import (
    V19ActionUse,
    V24CurveProfile,
    blueprint_from_runtime_schedule,
    load_v19_candidate_blueprint,
    runtime_schedule_from_blueprint,
    save_v19_candidate_blueprint,
    v19_blueprint_execution_digest,
)
from h3serve.native_engine.planner.v24_deployment import (
    _ACTION_RISK,
    _ATTENTION_COMPUTE,
    _ATTENTION_COST_RATIO,
    _CANONICAL_RANK,
    _FORECAST_COMPUTE,
    _NON_ATTENTION_COMPUTE,
    _RANK_TO_CANONICAL,
    _layer_risk_weight,
    _phase_risk_weight,
    _trajectory_risk,
)


SCHEMA = "h3_v24_aggressive_endpoint_probe_v1"


def _longest_forecast_run(actual: set[int], total_steps: int) -> int:
    run = maximum = 0
    for step in range(total_steps):
        run = 0 if step in actual else run + 1
        maximum = max(maximum, run)
    return maximum


def _cost(
    *,
    actual: set[int],
    ranks: dict[tuple[int, int], int],
    total_steps: int,
) -> float:
    attention = sum(
        _ATTENTION_COST_RATIO[_RANK_TO_CANONICAL[rank]]
        for rank in ranks.values()
    )
    return (
        len(actual) * _NON_ATTENTION_COMPUTE
        + _ATTENTION_COMPUTE * attention / 50.0
        + (total_steps - len(actual)) * _FORECAST_COMPUTE
    )


def _attention_risk_delta(
    step: int,
    layer: int,
    rank: int,
    *,
    curve: V24CurveProfile,
) -> float:
    return (
        _phase_risk_weight(step, 20, curve)
        * _layer_risk_weight(layer, curve)
        * (_ACTION_RISK[rank - 1] - _ACTION_RISK[rank])
    )


def _load_curve_profile(path: Path | None) -> V24CurveProfile:
    if path is None:
        return V24CurveProfile(profile_id="v24_aggressive_mechanistic_prior")
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = document.get("curve_profile", document)
    if not isinstance(payload, dict):
        raise ValueError("curve profile JSON must contain an object")
    allowed = {
        name
        for name in V24CurveProfile.__dataclass_fields__
    }
    values = {
        key: value
        for key, value in payload.items()
        if key in allowed
    }
    values["profile_id"] = str(
        values.get("profile_id", path.stem)
    )
    return V24CurveProfile(**values)


def _decode_source(
    source: Path,
) -> tuple[
    set[int],
    dict[tuple[int, int], int],
    dict[tuple[int, int], str | None],
]:
    blueprint = load_v19_candidate_blueprint(source)
    actual = {
        step
        for use in blueprint.action_uses
        if isinstance(use, V19ActionUse)
        for step in use.step_indices
    }
    runtime = runtime_schedule_from_blueprint(blueprint)
    ranks: dict[tuple[int, int], int] = {}
    prefixes: dict[tuple[int, int], str | None] = {}
    for step, layer, action in runtime:
        if step not in actual:
            continue
        if ":" in action:
            prefix, canonical = action.rsplit(":", 1)
        else:
            prefix, canonical = None, action
        ranks[(step, layer)] = _CANONICAL_RANK[canonical]
        prefixes[(step, layer)] = prefix
    expected = {(step, layer) for step in actual for layer in range(50)}
    if set(ranks) != expected:
        raise ValueError("source blueprint does not cover every Actual cell")
    return actual, ranks, prefixes


def _solve(
    *,
    source: Path,
    target_fraction: float,
    forecast_risk_multiplier: float,
    maximum_forecast_run: int,
    mandatory_actual_steps: tuple[int, ...],
    curve: V24CurveProfile,
) -> tuple[
    tuple[int, ...],
    tuple[tuple[int, int, str], ...],
    dict[str, object],
]:
    total_steps = 20
    actual, ranks, prefixes = _decode_source(source)
    original_cost = _cost(actual=actual, ranks=ranks, total_steps=total_steps)
    target_cost = original_cost * target_fraction
    mandatory = set(mandatory_actual_steps)
    operations: list[dict[str, object]] = []

    while _cost(actual=actual, ranks=ranks, total_steps=total_steps) > target_cost:
        candidates: list[tuple[float, float, float, str, int, int]] = []
        for (step, layer), rank in ranks.items():
            if rank <= 0:
                continue
            current = _ATTENTION_COST_RATIO[_RANK_TO_CANONICAL[rank]]
            proposed = _ATTENTION_COST_RATIO[_RANK_TO_CANONICAL[rank - 1]]
            saving = _ATTENTION_COMPUTE / 50.0 * (current - proposed)
            risk = _attention_risk_delta(
                step,
                layer,
                rank,
                curve=curve,
            )
            candidates.append((risk / saving, risk, saving, "attention", step, layer))

        before_trajectory = _trajectory_risk(actual, total_steps, curve)
        for step in sorted(actual - mandatory):
            proposed_actual = set(actual)
            proposed_actual.remove(step)
            if _longest_forecast_run(proposed_actual, total_steps) > maximum_forecast_run:
                continue
            proposed_ranks = {
                cell: rank for cell, rank in ranks.items() if cell[0] != step
            }
            saving = _cost(
                actual=actual, ranks=ranks, total_steps=total_steps
            ) - _cost(
                actual=proposed_actual,
                ranks=proposed_ranks,
                total_steps=total_steps,
            )
            # Do not credit a Forecast for removing an approximate Attention
            # row.  Demotion risk is the additional coherent trajectory debt;
            # this prevents unlike proxy units from making Forecast look like
            # a numerical improvement merely because the full row disappeared.
            risk = forecast_risk_multiplier * max(
                0.0,
                _trajectory_risk(proposed_actual, total_steps, curve)
                - before_trajectory,
            )
            candidates.append((risk / saving, risk, saving, "forecast", step, -1))

        if not candidates:
            break
        utility, risk, saving, kind, step, layer = min(
            candidates,
            key=lambda row: (row[0], row[1], -row[2], row[3], row[4], row[5]),
        )
        before = _cost(actual=actual, ranks=ranks, total_steps=total_steps)
        if kind == "forecast":
            actual.remove(step)
            for cell in tuple(ranks):
                if cell[0] == step:
                    ranks.pop(cell)
                    prefixes.pop(cell)
        else:
            ranks[(step, layer)] -= 1
            if prefixes[(step, layer)] is None:
                prefixes[(step, layer)] = "frontier"
        after = _cost(actual=actual, ranks=ranks, total_steps=total_steps)
        operations.append({
            "index": len(operations),
            "kind": kind,
            "step": step,
            "layer": layer,
            "modeled_risk_increase": risk,
            "compute_saving": saving,
            "risk_per_compute_saved": utility,
            "cost_before": before,
            "cost_after": after,
        })

    runtime: list[tuple[int, int, str]] = []
    for step in range(total_steps):
        if step not in actual:
            runtime.extend(
                (step, layer, "forecastfrontier:sparse_topk_0.0625")
                for layer in range(3)
            )
            continue
        for layer in range(50):
            rank = ranks[(step, layer)]
            canonical = _RANK_TO_CANONICAL[rank]
            prefix = prefixes[(step, layer)]
            action = canonical if canonical == "dense" else f"{prefix or 'frontier'}:{canonical}"
            runtime.append((step, layer, action))
    achieved = _cost(actual=actual, ranks=ranks, total_steps=total_steps)
    actions = Counter(_RANK_TO_CANONICAL[rank] for rank in ranks.values())
    summary = {
        "schema_version": SCHEMA,
        "source_blueprint": str(source.resolve()),
        "target_fraction_of_anchor_compute": target_fraction,
        "forecast_risk_multiplier": forecast_risk_multiplier,
        "curve_profile": curve.to_dict(),
        "maximum_forecast_run_constraint": maximum_forecast_run,
        "mandatory_actual_step_indices": sorted(mandatory),
        "original_compute_units": original_cost,
        "target_compute_units": target_cost,
        "achieved_compute_units": achieved,
        "predicted_speedup_over_anchor": original_cost / achieved,
        "actual_step_indices": sorted(actual),
        "forecast_step_indices": [
            step for step in range(total_steps) if step not in actual
        ],
        "maximum_forecast_run": _longest_forecast_run(actual, total_steps),
        "actual_attention_cells": dict(sorted(actions.items())),
        "operations": operations,
        "prompt_semantics_used": False,
        "resolution_used": False,
        "historical_schedule_role": "human_quality_anchor_only",
        "claim_scope": "aggressive endpoint probe; Human review required",
    }
    return tuple(sorted(actual)), tuple(runtime), summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-blueprint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--fractions",
        default="0.86,0.72,0.60",
        help="comma-separated target fractions of the Human anchor compute",
    )
    parser.add_argument("--forecast-risk-multiplier", type=float, default=10.0)
    parser.add_argument("--maximum-forecast-run", type=int, default=3)
    parser.add_argument("--mandatory-actual", default="0,1,19")
    parser.add_argument("--candidate-prefix", default="v24_aggressive")
    parser.add_argument(
        "--curve-profile-json",
        type=Path,
        help=(
            "optional V24 curve-profile JSON or proposal JSON containing "
            "curve_profile; this transfers Human-learned phase/layer risk"
        ),
    )
    args = parser.parse_args()
    curve = _load_curve_profile(args.curve_profile_json)
    fractions = tuple(float(value) for value in args.fractions.split(","))
    mandatory_actual = tuple(
        sorted({int(value) for value in args.mandatory_actual.split(",")})
    )
    if any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in fractions):
        parser.error("fractions must be finite values inside (0,1)")
    if any(step < 0 or step >= 20 for step in mandatory_actual):
        parser.error("mandatory Actual steps must lie inside [0,20)")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "purpose": (
            "V24 aggressive endpoint search below the V22 Human quality knee; "
            "same physical plans replay at 480p10 and 720p10"
        ),
        "candidates": [
            {
                "name": "control_v022_human_quality_knee",
                "blueprint": str(args.source_blueprint.resolve()),
            }
        ],
    }
    for index, fraction in enumerate(fractions, start=1):
        candidate_id = (
            f"{args.candidate_prefix}_p{index:02d}_f{fraction:.2f}"
            .replace(".", "p")
        )
        actual, runtime, summary = _solve(
            source=args.source_blueprint,
            target_fraction=fraction,
            forecast_risk_multiplier=args.forecast_risk_multiplier,
            maximum_forecast_run=args.maximum_forecast_run,
            mandatory_actual_steps=mandatory_actual,
            curve=curve,
        )
        blueprint = blueprint_from_runtime_schedule(
            candidate_id=candidate_id,
            total_steps=20,
            actual_step_indices=actual,
            attention_action_schedule=runtime,
            source="v24_aggressive_joint_degradation_probe",
        )
        candidate_root = args.output_root / candidate_id
        candidate_root.mkdir(parents=True, exist_ok=True)
        blueprint_path = candidate_root / "blueprint.json"
        save_v19_candidate_blueprint(blueprint_path, blueprint)
        summary["candidate_id"] = candidate_id
        summary["execution_digest"] = v19_blueprint_execution_digest(blueprint)
        summary["blueprint"] = str(blueprint_path.resolve())
        (candidate_root / "plan.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest["candidates"].append({
            "name": candidate_id,
            "blueprint": str(blueprint_path.resolve()),
        })
        print(json.dumps({
            "candidate_id": candidate_id,
            "predicted_speedup_over_anchor": summary["predicted_speedup_over_anchor"],
            "actual_step_indices": summary["actual_step_indices"],
            "maximum_forecast_run": summary["maximum_forecast_run"],
            "actual_attention_cells": summary["actual_attention_cells"],
        }, ensure_ascii=False))
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"manifest: {manifest_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
