#!/usr/bin/env python3
"""Propose Round 2 by contextual preference-BO over the V24 curve family.

The same five curve profiles are compiled for 720p5 and 720p15.  Two are
Human-evidence anchors (Round-1 p04 and p02); three are selected from a bounded
deterministic candidate pool by posterior UCB plus batch diversity.  Every
workload group is cost-matched to the p04 control before blueprints are sealed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np


SERVE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVE_ROOT))

from h3serve.native_engine.planner import (  # noqa: E402
    V19WorkloadContext,
    V24CurveProfile,
    V24ParetoRuntimeSelector,
    V24ReviewCandidate,
    V24StrategyVector,
    blueprint_from_runtime_schedule,
    fit_v24_preference_posterior,
    load_v24_human_review,
    save_v19_candidate_blueprint,
    v19_blueprint_execution_digest,
    v24_strategy_features,
)
from h3serve.native_engine.planner.v24_strategy import (  # noqa: E402
    V24_STRATEGY_FEATURE_NAMES,
)


BATCH_ID = "v24_curve_preference_r02"
SCHEMA = "h3_v24_preference_batch_proposal_v1"
RNG_SEED = 24_020_825


P02 = V24CurveProfile(
    profile_id="p02_trajectory_anchor",
    forecast_risk_scale=25.0,
)
P04 = V24CurveProfile(
    profile_id="p04_boundary_anchor",
    forecast_risk_scale=25.0,
    opening_amplitude=1.80,
    terminal_amplitude=1.50,
)


WORKLOADS = (
    {
        "workload_id": "720p5_castle_gate_seed82303",
        "comparison_group": "r02_720p5",
        "width": 1280,
        "height": 736,
        "frames": 124,
        "packed_tokens": 34_871,
        "scenario_manifest": "benchmarks/720p5_seed82303.json",
    },
    {
        "workload_id": "720p15_workshop_seed82332",
        "comparison_group": "r02_720p15",
        "width": 1280,
        "height": 736,
        "frames": 362,
        "packed_tokens": 100_141,
        "scenario_manifest": "benchmarks/720p15_seed82332_long_workshop.json",
    },
)


def _workload(row: dict[str, object]) -> V19WorkloadContext:
    return V19WorkloadContext(
        model_variant="base",
        service_family="first_last",
        packed_tokens=int(row["packed_tokens"]),
        condition_count=0,
        width=int(row["width"]),
        height=int(row["height"]),
        frames=int(row["frames"]),
        steps=20,
        actual_step_indices=tuple(range(20)),
        sampler="res_multistep",
        scheduler="simple",
    )


def _cost_matched(
    workload: V19WorkloadContext,
    profile: V24CurveProfile,
    target: float,
) -> tuple[float, Any]:
    selector = V24ParetoRuntimeSelector(curve=profile)
    best: tuple[float, float, Any] | None = None

    def inspect(acceleration: float) -> float:
        nonlocal best
        selection = selector.select(workload=workload, acceleration=acceleration)
        ratio = float(selection.summary["estimated_compute_ratio"])
        error = abs(ratio - target)
        row = (error, acceleration, selection)
        if best is None or row[:2] < best[:2]:
            best = row
        return ratio

    # The nested schedule is monotonic in acceleration.  Bisection finds both
    # sides of any discrete upgrade jump without evaluating 501 slider points.
    lower = 50.0
    upper = 100.0
    inspect(lower)
    inspect(upper)
    for _iteration in range(28):
        middle = 0.5 * (lower + upper)
        ratio = inspect(middle)
        if ratio > target:
            lower = middle
        else:
            upper = middle
    assert best is not None
    return round(best[1], 6), best[2]


def _candidate(
    *,
    profile: V24CurveProfile,
    workload_row: dict[str, object],
    selection: Any,
    acceleration: float,
) -> V24ReviewCandidate:
    vector = V24StrategyVector.from_selection(selection, total_steps=20)
    feature_map = v24_strategy_features(vector)
    return V24ReviewCandidate(
        candidate_id=f"pool_{profile.profile_id}_{workload_row['comparison_group']}",
        comparison_group=str(workload_row["comparison_group"]),
        strategy_digest=vector.digest,
        features=tuple(
            float(feature_map[name]) for name in V24_STRATEGY_FEATURE_NAMES
        ),
        video_path="PENDING_GENERATION",
        curve_profile_id=profile.profile_id,
        acceleration=acceleration,
        workload_id=str(workload_row["workload_id"]),
        width=int(workload_row["width"]),
        height=int(workload_row["height"]),
        frames=int(workload_row["frames"]),
        packed_tokens=int(workload_row["packed_tokens"]),
    )


def _profile_pool(count: int = 96) -> tuple[V24CurveProfile, ...]:
    rng = random.Random(RNG_SEED)
    result = [P04, P02]
    for index in range(count):
        result.append(V24CurveProfile(
            profile_id=f"pool_{index:03d}",
            forecast_risk_scale=rng.uniform(21.0, 29.0),
            forecast_run_coupling=rng.uniform(0.17, 0.31),
            opening_amplitude=rng.uniform(1.45, 2.35),
            terminal_amplitude=rng.uniform(1.15, 2.00),
            causal_layer_amplitude=rng.uniform(0.45, 1.00),
            bridge_layer_amplitude=rng.uniform(0.20, 0.55),
        ))
    return tuple(result)


def _select_profiles(
    rows: list[dict[str, Any]],
    *,
    posterior: Any,
    count: int = 5,
) -> list[dict[str, Any]]:
    # One profile is represented by its concatenated 5s/15s model features.
    matrix = np.asarray([
        np.concatenate([
            np.asarray(candidate.model_features, dtype=np.float64)
            for candidate in row["review_candidates"]
        ])
        for row in rows
    ])
    scale = np.std(matrix, axis=0)
    scale[scale < 1.0e-8] = 1.0
    standardized = (matrix - np.mean(matrix, axis=0)) / scale
    for row in rows:
        utilities = [
            posterior.utility(candidate.model_features)
            for candidate in row["review_candidates"]
        ]
        row["posterior_mean"] = float(np.mean([value[0] for value in utilities]))
        row["posterior_std"] = float(np.mean([value[1] for value in utilities]))
        row["ucb"] = row["posterior_mean"] + 0.35 * row["posterior_std"]

    selected = [
        next(index for index, row in enumerate(rows) if row["profile"] == P04),
        next(index for index, row in enumerate(rows) if row["profile"] == P02),
    ]
    while len(selected) < count:
        choices: list[tuple[float, int]] = []
        for index, row in enumerate(rows):
            if index in selected:
                continue
            distance = min(
                float(np.linalg.norm(standardized[index] - standardized[other]))
                / math.sqrt(standardized.shape[1])
                for other in selected
            )
            choices.append((float(row["ucb"]) + 0.30 * distance, index))
        selected.append(max(choices)[1])
    return [rows[index] for index in selected]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round01-review", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--control-acceleration", type=float, default=75.0)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    posterior = fit_v24_preference_posterior((
        load_v24_human_review(args.round01_review.resolve()),
    ))
    workloads = tuple(_workload(row) for row in WORKLOADS)
    targets = tuple(
        float(V24ParetoRuntimeSelector(curve=P04).select(
            workload=workload,
            acceleration=args.control_acceleration,
        ).summary["estimated_compute_ratio"])
        for workload in workloads
    )

    pool: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for profile in _profile_pool():
        selections = []
        candidates = []
        accelerations = []
        cost_errors = []
        for workload, workload_row, target in zip(workloads, WORKLOADS, targets):
            acceleration, selection = _cost_matched(workload, profile, target)
            error = abs(
                float(selection.summary["estimated_compute_ratio"]) - target
            )
            accelerations.append(acceleration)
            cost_errors.append(error)
            selections.append(selection)
            candidates.append(_candidate(
                profile=profile,
                workload_row=workload_row,
                selection=selection,
                acceleration=acceleration,
            ))
        if profile not in {P04, P02} and max(cost_errors) > 0.0015:
            continue
        signature = tuple(candidate.strategy_digest for candidate in candidates)
        if signature in seen:
            continue
        seen.add(signature)
        pool.append({
            "profile": profile,
            "selections": tuple(selections),
            "review_candidates": tuple(candidates),
            "accelerations": tuple(accelerations),
            "cost_errors": tuple(cost_errors),
        })
    selected = _select_profiles(pool, posterior=posterior)

    review_candidates: list[dict[str, Any]] = []
    run_manifests: list[dict[str, Any]] = []
    selected_profile_rows = []
    for selected_index, row in enumerate(selected):
        original = row["profile"]
        profile = V24CurveProfile(
            **{
                **{
                    key: value
                    for key, value in original.to_dict().items()
                    if key not in {"schema_version", "parameter_digest", "profile_id"}
                },
                "profile_id": (
                    f"r02_{selected_index:02d}_"
                    + (
                        "boundary_anchor"
                        if original == P04
                        else "trajectory_anchor"
                        if original == P02
                        else "bo_ucb_diverse"
                    )
                ),
            }
        )
        row["sealed_profile"] = profile
        selected_profile_rows.append({
            "profile": profile.to_dict(),
            "posterior_mean": row["posterior_mean"],
            "posterior_std": row["posterior_std"],
            "ucb": row["ucb"],
            "selection_role": profile.profile_id.split("_", 3)[-1],
            "maximum_compute_ratio_error": max(row["cost_errors"]),
        })

    for workload_index, (workload_row, target) in enumerate(zip(WORKLOADS, targets)):
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "purpose": "Contextual preference-BO Round 2 cost-matched proposals",
            "candidates": [],
        }
        for row in selected:
            profile = row["sealed_profile"]
            selection = row["selections"][workload_index]
            acceleration = row["accelerations"][workload_index]
            candidate_id = (
                f"{BATCH_ID}_{workload_row['comparison_group']}_{profile.profile_id}"
            )
            vector = V24StrategyVector.from_selection(selection, total_steps=20)
            feature_map = v24_strategy_features(vector)
            candidate_root = root / "candidates" / candidate_id
            candidate_root.mkdir(parents=True, exist_ok=True)
            blueprint = blueprint_from_runtime_schedule(
                candidate_id=candidate_id,
                total_steps=20,
                actual_step_indices=selection.actual_step_indices,
                attention_action_schedule=selection.attention_action_schedule,
                source="v24_contextual_preference_bayesian_optimization_r02",
            )
            blueprint_path = candidate_root / "blueprint.json"
            save_v19_candidate_blueprint(blueprint_path, blueprint)
            (candidate_root / "strategy.json").write_text(
                json.dumps(vector.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            proposal = {
                "schema_version": SCHEMA,
                "batch_id": BATCH_ID,
                "candidate_id": candidate_id,
                "comparison_group": workload_row["comparison_group"],
                "workload_id": workload_row["workload_id"],
                "curve_profile": profile.to_dict(),
                "requested_acceleration": acceleration,
                "control_acceleration": args.control_acceleration,
                "target_compute_ratio": target,
                "achieved_compute_ratio": selection.summary["estimated_compute_ratio"],
                "strategy_digest": vector.digest,
                "execution_digest": v19_blueprint_execution_digest(blueprint),
                "actual_step_indices": list(selection.actual_step_indices),
                "maximum_forecast_run": selection.summary["maximum_forecast_run"],
                "technique_mix": selection.summary["technique_mix"],
                "features": feature_map,
                "blueprint": str(blueprint_path),
                "claim_scope": "Human preference proposal; not deployment admitted",
            }
            (candidate_root / "proposal.json").write_text(
                json.dumps(proposal, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            manifest["candidates"].append({
                "name": candidate_id,
                "blueprint": str(blueprint_path),
            })
            review_candidates.append({
                "candidate_id": candidate_id,
                "comparison_group": workload_row["comparison_group"],
                "strategy_digest": vector.digest,
                "execution_digest": proposal["execution_digest"],
                "features": feature_map,
                "video_path": "PENDING_GENERATION",
                "curve_profile_id": profile.profile_id,
                "acceleration": acceleration,
                "workload_id": workload_row["workload_id"],
                "latency_seconds": None,
                "workload_context": {
                    key: workload_row[key]
                    for key in ("width", "height", "frames", "packed_tokens")
                },
            })
        manifest_path = root / f"{workload_row['comparison_group']}_blueprints.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        run_manifests.append({
            "comparison_group": workload_row["comparison_group"],
            "blueprint_manifest": str(manifest_path),
            "scenario_manifest": str(SERVE_ROOT / workload_row["scenario_manifest"]),
            "target_compute_ratio": target,
        })

    review = {
        "schema_version": "h3_v24_human_preference_review_v1",
        "batch_id": BATCH_ID,
        "purpose": "Test Round-1 preferred curves across 720p5 and 720p15 contexts.",
        "candidate_count": len(review_candidates),
        "acquisition": {
            "method": "posterior_ucb_plus_standardized_batch_diversity",
            "posterior_source": str(args.round01_review.resolve()),
            "pool_seed": RNG_SEED,
            "pool_unique_strategy_pairs": len(pool),
            "selected_profiles": selected_profile_rows,
        },
        "candidates": review_candidates,
        "run_manifests": run_manifests,
        "human_feedback": {
            "reviewer": "Human",
            "rankings": {str(row["comparison_group"]): [] for row in WORKLOADS},
            "overall_scores": {},
            "issues": {},
            "notes": {},
        },
    }
    review_path = root / "human_review.json"
    review_path.write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "batch_id": BATCH_ID,
        "review": str(review_path),
        "pool_unique_strategy_pairs": len(pool),
        "selected_profiles": selected_profile_rows,
        "run_manifests": run_manifests,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
