#!/usr/bin/env python3
"""Seal matched Human reviews into an exact V19 plan-risk artifact.

The command is intentionally plan-bound.  A review naming a friendly policy
or a nominal Top-K value is insufficient: the candidate blueprint, repeated
end-to-end schedule artifact, runtime fingerprint and physical action ids must
all agree.  Incomplete artifacts are useful for tracking coverage but remain
ineligible for the release frontier.
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
    FIXED_TOPK_ACTION_IMPLEMENTATION,
    ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
    ROUND215_ACTION_IMPLEMENTATION,
    ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
    ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION,
    V19RiskCalibrationError,
    V19RiskReview,
    V19SourceRecord,
    build_v19_bootstrap_registry,
    create_v19_plan_risk_calibration,
    load_v19_candidate_blueprint,
    load_v19_human_evidence,
    load_v19_schedule_cost_calibration,
    sha256_file,
    v19_blueprint_execution_digest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-evidence", type=Path, action="append", required=True)
    parser.add_argument("--blueprint", type=Path, required=True)
    parser.add_argument("--schedule-calibration", type=Path, required=True)
    parser.add_argument("--risk-model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-reported-cases", type=int, default=3)
    parser.add_argument("--z-score", type=float, default=2.45)
    parser.add_argument(
        "--complete",
        action="store_true",
        help="claim complete coverage; fails unless every dimension has enough labels",
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


def _source(path: Path, index: int) -> V19SourceRecord:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(SERVE_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise V19RiskCalibrationError(
            f"Human evidence is outside the release: {resolved}"
        ) from error
    return V19SourceRecord(
        source_id=f"human_review_source_{index}",
        relative_path=relative,
        sha256=sha256_file(resolved),
    )


def main() -> int:
    args = parse_args()
    registry = _registry()
    blueprint = load_v19_candidate_blueprint(args.blueprint)
    execution_digest = v19_blueprint_execution_digest(blueprint)
    schedule = load_v19_schedule_cost_calibration(
        args.schedule_calibration,
        registry=registry,
    )
    if schedule.binding.execution_digest != execution_digest:
        raise V19RiskCalibrationError(
            "Human-risk blueprint does not match the E2E execution schedule"
        )
    action_ids = tuple(sorted({use.action_id for use in blueprint.action_uses}))
    if schedule.binding.action_ids != action_ids:
        raise V19RiskCalibrationError(
            "Human-risk blueprint action ids do not match the E2E schedule"
        )

    reviews: list[V19RiskReview] = []
    sources: list[V19SourceRecord] = []
    case_ids: set[str] = set()
    for index, path in enumerate(args.human_evidence, start=1):
        evidence = load_v19_human_evidence(
            path,
            serve_root=SERVE_ROOT,
            require_artifacts=True,
        )
        sources.append(_source(path, index))
        for record in evidence.records:
            if record.candidate_id != blueprint.candidate_id:
                continue
            if record.attribution not in (
                "candidate_positive", "candidate_regression"
            ):
                continue
            if record.evidence_id in case_ids:
                raise V19RiskCalibrationError(
                    f"duplicate Human review case: {record.evidence_id}"
                )
            case_ids.add(record.evidence_id)
            reviews.append(V19RiskReview(
                case_id=record.evidence_id,
                mechanism=record.mechanism,
                attribution=record.attribution,
                dimensions=record.dimensions,
                candidate_artifact_sha256=record.artifact_sha256,
                comparator_artifact_sha256=(
                    record.comparator_artifact_sha256
                ),
            ))
    if not reviews:
        raise V19RiskCalibrationError(
            "no attributable Human reviews match the exact candidate id"
        )

    artifact = create_v19_plan_risk_calibration(
        registry=registry,
        execution_digest=execution_digest,
        risk_model_id=args.risk_model_id,
        action_ids=action_ids,
        workload=schedule.workload,
        runtime=schedule.runtime,
        reviews=reviews,
        sources=sources,
        complete=args.complete,
        minimum_reported_cases=args.minimum_reported_cases,
        z_score=args.z_score,
    )
    if args.complete:
        artifact.require_planner_ready()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "candidate_id": blueprint.candidate_id,
        "execution_digest": execution_digest,
        "review_cases": len(reviews),
        "complete": artifact.complete,
        "planner_ready": artifact.planner_ready,
        "risk_ucb": artifact.risk_ucb.as_tuple(),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
