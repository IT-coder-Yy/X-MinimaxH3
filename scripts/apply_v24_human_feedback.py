#!/usr/bin/env python3
"""Validate and apply one Human feedback source to a V24 review document."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SERVE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVE_ROOT))

from h3serve.native_engine.planner import (  # noqa: E402
    fit_v24_preference_posterior,
    load_v24_human_review,
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument(
        "--history-review",
        type=Path,
        action="append",
        default=[],
        help="previous Human-review JSON to include in the cumulative posterior",
    )
    args = parser.parse_args()

    review_path = args.review.resolve()
    feedback_path = args.feedback.resolve()
    posterior_path = args.posterior.resolve()
    review = _load(review_path)
    feedback = _load(feedback_path)
    if feedback.get("batch_id") != review.get("batch_id"):
        raise ValueError("feedback and review batch identifiers do not match")
    known = {str(row["candidate_id"]) for row in review["candidates"]}
    named = {
        str(value)
        for tiers in feedback.get("rankings", {}).values()
        for tier in tiers
        for value in tier
    }
    named.update(str(value) for value in feedback.get("overall_scores", {}))
    named.update(str(value) for value in feedback.get("issues", {}))
    named.update(str(value) for value in feedback.get("notes", {}))
    if not named <= known:
        raise ValueError(f"feedback names unknown candidates: {sorted(named - known)}")

    target = review.setdefault("human_feedback", {})
    for key in ("reviewer", "rankings", "overall_scores", "issues", "notes"):
        target[key] = feedback.get(key, {} if key != "reviewer" else "Human")
    target["source"] = str(feedback_path)
    target["interpretation"] = feedback.get("interpretation")
    review_path.write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    validated = load_v24_human_review(review_path)
    history = tuple(
        load_v24_human_review(path.resolve())
        for path in args.history_review
    )
    batch_ids = tuple(row.batch_id for row in (*history, validated))
    if len(set(batch_ids)) != len(batch_ids):
        raise ValueError("history reviews must name distinct batch identifiers")
    posterior = fit_v24_preference_posterior((*history, validated))
    posterior_path.parent.mkdir(parents=True, exist_ok=True)
    posterior_path.write_text(
        json.dumps(posterior.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "review": str(review_path),
        "feedback": str(feedback_path),
        "posterior": str(posterior_path),
        "comparison_count": posterior.comparison_count,
        "review_batch_ids": list(posterior.review_batch_ids),
        "issue_report_counts": {
            name: row["reported_count"] for name, row in posterior.issue_heads.items()
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
