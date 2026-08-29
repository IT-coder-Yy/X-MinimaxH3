#!/usr/bin/env python3
"""Build the first cost-matched V24 strategy-curve preference batch.

This is an offline proposal compiler.  It freezes the current physical action
alphabet and changes only smooth curve parameters.  Every workload group is
matched to the control's estimated compute ratio before blueprints are sealed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SERVE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVE_ROOT))

from h3serve.native_engine.planner import (  # noqa: E402
    V19WorkloadContext,
    V24CurveProfile,
    V24_ISSUE_DIMENSIONS,
    V24ParetoRuntimeSelector,
    V24StrategyVector,
    blueprint_from_runtime_schedule,
    save_v19_candidate_blueprint,
    v19_blueprint_execution_digest,
    v24_strategy_features,
)


SCHEMA = "h3_v24_preference_batch_proposal_v1"
BATCH_ID = "v24_curve_preference_r01"


PROFILES = (
    V24CurveProfile(
        profile_id="p00_attention_heavy_control",
        forecast_risk_scale=1.0,
    ),
    V24CurveProfile(
        profile_id="p01_balanced_11actual",
        forecast_risk_scale=15.0,
    ),
    V24CurveProfile(
        profile_id="p02_trajectory_12actual",
        forecast_risk_scale=25.0,
    ),
    V24CurveProfile(
        profile_id="p03_causal_bridge_12actual",
        forecast_risk_scale=25.0,
        causal_layer_amplitude=1.35,
        bridge_layer_amplitude=0.75,
    ),
    V24CurveProfile(
        profile_id="p04_boundary_12actual",
        forecast_risk_scale=25.0,
        opening_amplitude=1.80,
        terminal_amplitude=1.50,
    ),
)


WORKLOADS = (
    {
        "workload_id": "480p10_radio_console_seed82341",
        "comparison_group": "r01_480p10",
        "width": 864,
        "height": 480,
        "frames": 243,
        "packed_tokens": 30_455,
        "scenario_manifest": "benchmarks/480p10_seed82341_radio_console_curve_r01.json",
    },
    {
        "workload_id": "720p10_radio_console_seed82341",
        "comparison_group": "r01_720p10",
        "width": 1280,
        "height": 736,
        "frames": 243,
        "packed_tokens": 67_535,
        "scenario_manifest": "benchmarks/720p10_seed82341_radio_console_quality20.json",
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


def _cost_matched_selection(
    workload: V19WorkloadContext,
    profile: V24CurveProfile,
    *,
    target_ratio: float,
) -> tuple[float, object]:
    selector = V24ParetoRuntimeSelector(curve=profile)
    best: tuple[float, float, object] | None = None
    for index in range(500, 1001):
        acceleration = index / 10.0
        selection = selector.select(
            workload=workload,
            acceleration=acceleration,
        )
        ratio = float(selection.summary["estimated_compute_ratio"])
        row = (abs(ratio - target_ratio), acceleration, selection)
        if best is None or row[:2] < best[:2]:
            best = row
    assert best is not None
    return best[1], best[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--control-acceleration", type=float, default=75.0)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    review_candidates: list[dict[str, object]] = []
    run_manifests: list[dict[str, object]] = []
    for workload_row in WORKLOADS:
        workload = _workload(workload_row)
        control = V24ParetoRuntimeSelector(curve=PROFILES[0]).select(
            workload=workload,
            acceleration=args.control_acceleration,
        )
        target_ratio = float(control.summary["estimated_compute_ratio"])
        manifest = {
            "schema_version": 1,
            "purpose": (
                "Cost-matched preference comparison over one frozen V24 "
                "physical action alphabet"
            ),
            "candidates": [],
        }
        for profile in PROFILES:
            acceleration, selection = _cost_matched_selection(
                workload,
                profile,
                target_ratio=target_ratio,
            )
            candidate_id = (
                f"{BATCH_ID}_{workload_row['comparison_group']}_{profile.profile_id}"
            )
            vector = V24StrategyVector.from_selection(selection, total_steps=20)
            features = v24_strategy_features(vector)
            candidate_root = root / "candidates" / candidate_id
            candidate_root.mkdir(parents=True, exist_ok=True)
            blueprint = blueprint_from_runtime_schedule(
                candidate_id=candidate_id,
                total_steps=20,
                actual_step_indices=selection.actual_step_indices,
                attention_action_schedule=selection.attention_action_schedule,
                source="v24_curve_preference_bayesian_optimization_r01",
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
                "target_compute_ratio": target_ratio,
                "achieved_compute_ratio": selection.summary["estimated_compute_ratio"],
                "strategy_digest": vector.digest,
                "execution_digest": v19_blueprint_execution_digest(blueprint),
                "actual_step_indices": list(selection.actual_step_indices),
                "maximum_forecast_run": selection.summary["maximum_forecast_run"],
                "technique_mix": selection.summary["technique_mix"],
                "features": features,
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
                "features": features,
                "video_path": "PENDING_GENERATION",
                "curve_profile_id": profile.profile_id,
                "acceleration": acceleration,
                "workload_id": workload_row["workload_id"],
                "latency_seconds": None,
                "workload_context": {
                    "width": workload_row["width"],
                    "height": workload_row["height"],
                    "frames": workload_row["frames"],
                    "packed_tokens": workload_row["packed_tokens"],
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
            "target_compute_ratio": target_ratio,
        })

    review = {
        "schema_version": "h3_v24_human_preference_review_v1",
        "batch_id": BATCH_ID,
        "purpose": (
            "Identify the quality-optimal exchange rate between full DiT "
            "trajectory evaluations and per-layer Attention fidelity."
        ),
        "candidate_count": len(review_candidates),
        "candidates": review_candidates,
        "run_manifests": run_manifests,
        "human_feedback": {
            "reviewer": "Human",
            "instructions": {
                "ranking": "Rank each comparison_group from best to worst; ties share one tier.",
                "overall_scores": "Score each video from 0 to 100.",
                "issues": (
                    "For each inspected dimension use present, absent or not_reported: "
                    + "/".join(V24_ISSUE_DIMENSIONS)
                    + "."
                ),
                "notes": "Describe concrete visible or audible failure mechanisms.",
            },
            "rankings": {
                str(row["comparison_group"]): [] for row in WORKLOADS
            },
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
        "candidate_count": len(review_candidates),
        "review": str(review_path),
        "run_manifests": run_manifests,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
