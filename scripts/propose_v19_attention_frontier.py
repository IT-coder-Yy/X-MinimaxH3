#!/usr/bin/env python3
"""Generate digest-sealed V19 proposals from exact physical cell evidence.

The resulting files are research candidates, not release plans: their
Dense-relative numerical proxy is explicitly marked as *not* Human risk.
They become certifiable only after complete-request timing and Human review.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys


SERVE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVE_ROOT))

from h3serve.native_engine.planner import (  # noqa: E402
    DENSE_ACTION_ID,
    FIXED_TOPK_ACTION_IMPLEMENTATION,
    ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
    ROUND215_ACTION_IMPLEMENTATION,
    ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
    ROUND229_ACTION_ID,
    ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION,
    V19ActionUse,
    V19BudgetedCellOptimizer,
    V19BudgetedProposalRequest,
    V19CalibrationCatalog,
    V19CellAction,
    V19ForecastCalibrationCatalog,
    V19PlanningError,
    build_v19_bootstrap_registry,
    couple_v19_proposal,
    evaluate_v19_human_constraints,
    load_v19_action_calibration,
    load_v19_candidate_blueprint,
    load_v19_forecast_calibration,
    save_v19_candidate_blueprint,
    sha256_file,
    v19_av_clarity_importance_profile,
    v19_blueprint_execution_digest,
    v19_coupled_numerical_frontier,
    v19_numerical_proposal_frontier,
    v19_round02_av_motion_screening_policy,
    v19_structural_causal_importance_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-calibration", type=Path, required=True)
    parser.add_argument("--sparse-calibration", type=Path, required=True)
    parser.add_argument(
        "--forecast-calibration",
        type=Path,
        help=(
            "optional exact Forecast-composite calibration; when provided the "
            "reported numerical frontier is priced across the complete coupled "
            "trajectory instead of Attention alone"
        ),
    )
    parser.add_argument("--comparator-blueprint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--budget-ratios",
        default="0.80,0.85,0.90,0.95,1.00",
        help="comma-separated fractions of comparator Attention p90",
    )
    parser.add_argument(
        "--auto-budget-points",
        type=int,
        default=0,
        help=(
            "replace --budget-ratios with this many evenly spaced physical "
            "budgets between all-6.25%% sparse and all-Dense evidence"
        ),
    )
    parser.add_argument("--cost-quantum-ms", type=float, default=5.0)
    parser.add_argument("--maximum-cell-proxy", type=float, default=float("inf"))
    parser.add_argument(
        "--importance-profile",
        choices=("uniform", "structural-causal-v1", "av-clarity-v1", "all"),
        default="uniform",
        help=(
            "proposal-only cell importance; structural-causal-v1 protects "
            "opening/terminal and known H3 causal-interaction regions; "
            "av-clarity-v1 adds Human-feedback-derived keep floors while "
            "remaining an uncertified research prior"
        ),
    )
    parser.add_argument(
        "--candidate-prefix",
        default="v19_attention_frontier",
        help="stable candidate id prefix for an independent research frontier",
    )
    parser.add_argument(
        "--allow-replacing-comparator-dense",
        action="store_true",
        help="research ablation only; default freezes reviewed Dense causal rails",
    )
    parser.add_argument(
        "--human-constraint-policy",
        choices=("round02-av-motion-v1", "none"),
        default="round02-av-motion-v1",
        help=(
            "non-compensating proposal screen; the default rejects the exact "
            "v008/v010 schedules, sub-0.25 Actual actions and Forecast runs "
            "longer than three, while leaving AV/motion gates unevaluated"
        ),
    )
    return parser.parse_args()


def _registry():
    return build_v19_bootstrap_registry(implementation_ids={
        "fixed_topk": FIXED_TOPK_ACTION_IMPLEMENTATION,
        "round215": ROUND215_ACTION_IMPLEMENTATION,
        "round188": ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
        "round228": ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
        "round229": ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION,
    })


def _ratios(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(sorted(set(float(value.strip()) for value in raw.split(","))))
    except ValueError as error:
        raise SystemExit("--budget-ratios must contain numbers") from error
    if not values or any(not 0.0 < value <= 4.0 for value in values):
        raise SystemExit("--budget-ratios must lie inside (0,4]")
    return values


def _automatic_ratios(
    *,
    points: int,
    catalog: V19CalibrationCatalog,
    comparator_attention_p90_ms: float,
    workload,
    runtime,
) -> tuple[float, ...]:
    if points < 2:
        raise SystemExit("--auto-budget-points must be zero or at least two")
    lower = catalog.estimate_schedule(
        (V19ActionUse(
            action_id=ROUND229_ACTION_ID,
            canonical_action="sparse_topk_0.0625",
            step_indices=workload.actual_step_indices,
        ),),
        workload=workload,
        runtime=runtime,
    ).p90_ms / comparator_attention_p90_ms
    upper = catalog.estimate_schedule(
        (V19ActionUse(
            action_id=DENSE_ACTION_ID,
            canonical_action="dense",
            step_indices=workload.actual_step_indices,
        ),),
        workload=workload,
        runtime=runtime,
    ).p90_ms / comparator_attention_p90_ms
    return tuple(
        lower + (upper - lower) * index / (points - 1)
        for index in range(points)
    )


def main() -> int:
    args = parse_args()
    registry = _registry()
    dense = load_v19_action_calibration(
        args.dense_calibration, registry=registry
    )
    sparse = load_v19_action_calibration(
        args.sparse_calibration,
        registry=registry,
        expected_workload=dense.workload,
        expected_runtime=dense.runtime,
    )
    catalog = V19CalibrationCatalog(registry)
    catalog.add(dense)
    catalog.add(sparse)
    forecast_catalog = None
    if args.forecast_calibration is not None:
        forecast = load_v19_forecast_calibration(
            args.forecast_calibration,
            registry=registry,
            expected_workload=dense.workload,
            expected_runtime=dense.runtime,
        )
        forecast_catalog = V19ForecastCalibrationCatalog(registry)
        forecast_catalog.add(forecast)
    comparator = load_v19_candidate_blueprint(args.comparator_blueprint)
    comparator_attention = catalog.estimate_schedule(
        tuple(
            use for use in comparator.action_uses if isinstance(use, V19ActionUse)
        ),
        workload=dense.workload,
        runtime=dense.runtime,
    )
    actions = (
        V19CellAction(DENSE_ACTION_ID, "dense"),
        V19CellAction(ROUND229_ACTION_ID, "sparse_topk_0.5"),
        V19CellAction(ROUND229_ACTION_ID, "sparse_topk_0.25"),
        V19CellAction(ROUND229_ACTION_ID, "sparse_topk_0.1"),
        V19CellAction(ROUND229_ACTION_ID, "sparse_topk_0.0625"),
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    input_root = args.output_root / "inputs"
    input_root.mkdir(parents=True, exist_ok=True)
    input_snapshots = {}
    input_sources = [
        ("dense_calibration", args.dense_calibration),
        ("sparse_calibration", args.sparse_calibration),
        ("comparator_blueprint", args.comparator_blueprint),
    ]
    if args.forecast_calibration is not None:
        input_sources.append((
            "forecast_calibration", args.forecast_calibration
        ))
    for role, source in input_sources:
        source = source.resolve()
        target = input_root / f"{role}{source.suffix or '.json'}"
        if source != target.resolve():
            shutil.copy2(source, target)
        input_snapshots[role] = {
            "source": str(source),
            "snapshot": target.relative_to(args.output_root).as_posix(),
            "sha256": sha256_file(target),
        }
    proposals: list[dict[str, object]] = []
    proposal_objects = {}
    coupled_objects = {}
    digests: set[str] = set()
    optimizer = V19BudgetedCellOptimizer(catalog)
    importance_profiles = {
        "uniform": None,
        "structural-causal-v1": v19_structural_causal_importance_profile(
            dense.workload.steps
        ),
        "av-clarity-v1": v19_av_clarity_importance_profile(
            dense.workload.steps
        ),
    }
    human_constraint_policy = (
        None
        if args.human_constraint_policy == "none"
        else v19_round02_av_motion_screening_policy()
    )
    selected_profiles = (
        tuple(importance_profiles.items())
        if args.importance_profile == "all"
        else ((args.importance_profile, importance_profiles[args.importance_profile]),)
    )
    ratios = (
        _automatic_ratios(
            points=args.auto_budget_points,
            catalog=catalog,
            comparator_attention_p90_ms=comparator_attention.p90_ms,
            workload=dense.workload,
            runtime=dense.runtime,
        )
        if args.auto_budget_points
        else _ratios(args.budget_ratios)
    )
    for profile_name, cell_importance in selected_profiles:
        profile_slug = profile_name.replace("-v1", "").replace("-", "_")
        for ratio in ratios:
            prefix = (
                f"{args.candidate_prefix}_{profile_slug}"
                if args.importance_profile == "all"
                else args.candidate_prefix
            )
            candidate_id = f"{prefix}_r{ratio:.4f}".replace(".", "p")
            try:
                proposal = optimizer.optimize(V19BudgetedProposalRequest(
                    candidate_id=candidate_id,
                    comparator=comparator,
                    workload=dense.workload,
                    runtime=dense.runtime,
                    maximum_attention_p90_ms=comparator_attention.p90_ms * ratio,
                    actions=actions,
                    cost_quantum_ms=args.cost_quantum_ms,
                    maximum_cell_proxy=args.maximum_cell_proxy,
                    protect_comparator_dense=(
                        not args.allow_replacing_comparator_dense
                    ),
                    cell_importance=cell_importance,
                ))
            except V19PlanningError as error:
                proposals.append({
                    "importance_profile": profile_name,
                    "budget_ratio": ratio,
                    "feasible": False,
                    "physical_feasible": False,
                    "human_constraint_eligible": False,
                    "release_eligible": False,
                    "reason": str(error),
                })
                continue
            execution_digest = v19_blueprint_execution_digest(proposal.blueprint)
            human_constraint_report = (
                None
                if human_constraint_policy is None
                else evaluate_v19_human_constraints(
                    proposal.blueprint, human_constraint_policy
                )
            )
            if (
                human_constraint_report is not None
                and not human_constraint_report.proposal_eligible
            ):
                proposals.append({
                    "importance_profile": profile_name,
                    "budget_ratio": ratio,
                    "feasible": False,
                    "physical_feasible": True,
                    "human_constraint_eligible": False,
                    "release_eligible": False,
                    "reason": "violates non-compensating Human constraints",
                    "human_constraint_report": asdict(human_constraint_report),
                })
                continue
            if execution_digest in digests:
                continue
            digests.add(execution_digest)
            proposal_objects[execution_digest] = proposal
            coupled = None
            if forecast_catalog is not None:
                coupled = couple_v19_proposal(
                    proposal,
                    workload=dense.workload,
                    runtime=dense.runtime,
                    forecast_catalog=forecast_catalog,
                )
                coupled_objects[execution_digest] = coupled
            path = args.output_root / f"{candidate_id}_blueprint.json"
            save_v19_candidate_blueprint(path, proposal.blueprint)
            row = {
                "importance_profile": profile_name,
                "budget_ratio": ratio,
                "feasible": True,
                "physical_feasible": True,
                "human_constraint_eligible": True,
                "release_eligible": (
                    False
                    if human_constraint_report is None
                    else human_constraint_report.release_eligible
                ),
                "blueprint": path.name,
                "execution_digest": execution_digest,
                "human_constraint_report": (
                    None
                    if human_constraint_report is None
                    else asdict(human_constraint_report)
                ),
                **{
                    key: value
                    for key, value in asdict(proposal).items()
                    if key != "blueprint"
                },
            }
            if coupled is not None:
                row["coupled_physical"] = {
                    "schema_version": coupled.schema_version,
                    "forecast_p50_ms": coupled.forecast_p50_ms,
                    "forecast_p90_ms": coupled.forecast_p90_ms,
                    "physical_p50_ms": coupled.physical_p50_ms,
                    "physical_p90_ms": coupled.physical_p90_ms,
                    "peak_vram_gib": coupled.peak_vram_gib,
                    "evidence_ids": list(coupled.evidence_ids),
                }
            proposals.append(row)
    if coupled_objects:
        numerical_frontier = {
            v19_blueprint_execution_digest(proposal.blueprint)
            for proposal in v19_coupled_numerical_frontier(
                coupled_objects.values()
            )
        }
    else:
        numerical_frontier = {
            v19_blueprint_execution_digest(proposal.blueprint)
            for proposal in v19_numerical_proposal_frontier(
                proposal_objects.values()
            )
        }
    feasible = tuple(row for row in proposals if bool(row.get("feasible")))
    for row in feasible:
        row["numerical_pareto"] = row["execution_digest"] in numerical_frontier
    summary = {
        "schema_version": "h3_v19_attention_proposal_frontier_v3",
        "warning": (
            "The non-compensating Dense-relative numerical Pareto vector is not "
            "Human quality; every proposal requires exact E2E timing and Human "
            "mechanism review."
        ),
        "comparator": {
            "candidate_id": comparator.candidate_id,
            "execution_digest": v19_blueprint_execution_digest(comparator),
            "attention_p50_ms": comparator_attention.p50_ms,
            "attention_p90_ms": comparator_attention.p90_ms,
        },
        "workload_digest": dense.workload.digest,
        "runtime_digest": dense.runtime.digest,
        "complete_coupled_physical_pricing": bool(coupled_objects),
        "human_constraint_policy": (
            None
            if human_constraint_policy is None
            else asdict(human_constraint_policy)
        ),
        "input_snapshots": input_snapshots,
        "importance_profile": (
            selected_profiles[0][0]
            if len(selected_profiles) == 1
            else "multiple_non_compensating_profiles"
        ),
        "importance_profiles": [name for name, _profile in selected_profiles],
        "budget_ratios": list(ratios),
        "numerical_objective_dimensions": [
            (
                "physical_p90_ms"
                if coupled_objects
                else "attention_p90_ms"
            ),
            (
                "peak_vram_gib"
                if coupled_objects
                else "attention_peak_vram_gib"
            ),
            "one_minus_mean_cosine_sum",
            "one_minus_min_cosine_sum",
            "global_relative_rms_sum",
            "mean_head_relative_rms_sum",
            "max_head_relative_rms_sum",
            "max_relative_l1_sum",
            "one_minus_mean_cosine_max",
            "one_minus_min_cosine_max",
            "global_relative_rms_max",
            "mean_head_relative_rms_max",
            "max_head_relative_rms_max",
            "max_relative_l1_max",
            "terminal_consecutive_forecasts",
            "terminal_forecast_debt",
            "terminal_sparse_mass_deficit",
            "terminal_audio_debt",
            "maximum_consecutive_forecasts",
            "maximum_forecast_debt",
            "maximum_sparse_mass_deficit",
            "maximum_audio_debt",
        ],
        "numerical_pareto_count": sum(
            bool(row.get("numerical_pareto")) for row in feasible
        ),
        "proposals": proposals,
    }
    summary_path = args.output_root / "frontier_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(summary_path.resolve()),
        "comparator_attention_p90_ms": comparator_attention.p90_ms,
        "unique_proposals": sum(bool(row.get("feasible")) for row in proposals),
        "numerical_pareto_proposals": sum(
            bool(row.get("numerical_pareto")) for row in proposals
        ),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
