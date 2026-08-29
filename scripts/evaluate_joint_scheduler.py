#!/usr/bin/env python3
"""Run the H3 joint-scheduler control-plane evaluation contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from h3serve.native_engine.planner.evaluation import evaluate_joint_scheduler
from h3serve.native_engine.planner.joint_acceleration import (
    DEFAULT_JOINT_POLICY,
    JOINT_POLICY_V1_HEURISTIC,
    JOINT_POLICY_V2_EXACT_ATTENTION,
    JOINT_POLICY_V3_GLOBAL_DP,
    JOINT_POLICY_V4_EVIDENCE_GLOBAL_DP,
    JOINT_POLICY_V5_CALIBRATION_MATCHED_GLOBAL_DP,
    JOINT_POLICY_V6_CAUSAL_ISLAND_GLOBAL_DP,
    JOINT_POLICY_V7_LAYER_RISK_GLOBAL_DP,
    JOINT_POLICY_V8_PHASE_LAYER_RISK_GLOBAL_DP,
    JOINT_POLICY_V9_BOUNDED_ONLINE_GLOBAL_DP,
    JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP,
    JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
    JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP,
    JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=(
            JOINT_POLICY_V1_HEURISTIC,
            JOINT_POLICY_V2_EXACT_ATTENTION,
            JOINT_POLICY_V3_GLOBAL_DP,
            JOINT_POLICY_V4_EVIDENCE_GLOBAL_DP,
            JOINT_POLICY_V5_CALIBRATION_MATCHED_GLOBAL_DP,
            JOINT_POLICY_V6_CAUSAL_ISLAND_GLOBAL_DP,
            JOINT_POLICY_V7_LAYER_RISK_GLOBAL_DP,
            JOINT_POLICY_V8_PHASE_LAYER_RISK_GLOBAL_DP,
            JOINT_POLICY_V9_BOUNDED_ONLINE_GLOBAL_DP,
            JOINT_POLICY_V10_PHASE_SENTINEL_GLOBAL_DP,
            JOINT_POLICY_V11_CALIBRATED_GROWTH_GLOBAL_DP,
            JOINT_POLICY_V12_RESERVE_REBATE_GLOBAL_DP,
            JOINT_POLICY_V13_ADAPTIVE_LATENCY_GLOBAL_DP,
        ),
        default=DEFAULT_JOINT_POLICY,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_joint_scheduler(policy_id=args.policy)
    document = report.to_dict()
    rendered = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(args.output)
    print(rendered)
    return 1 if report.hard_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
