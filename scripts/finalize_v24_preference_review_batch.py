#!/usr/bin/env python3
"""Bind a V24 Human-review proposal to completed hot-session evidence.

The proposal is created before generation.  This finalizer accepts one or more
hot-session reports, verifies candidate and execution digests, resolves the
relocated project-tree videos by basename, and writes measured runtime fields
back to the review document.  Existing Human feedback is preserved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SUMMARY_SCHEMA = "h3_v24_preference_execution_summary_v1"


def _load(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return document


def _candidate_record(request: dict[str, Any]) -> tuple[str, str]:
    joint = (request.get("execution_profile") or {}).get("joint_acceleration")
    if not isinstance(joint, dict):
        raise ValueError("runtime request lacks sealed joint-acceleration evidence")
    candidate_id = str(joint.get("candidate_id") or "")
    execution_digest = str(joint.get("execution_digest") or "")
    if not candidate_id or len(execution_digest) != 64:
        raise ValueError("runtime request has incomplete candidate provenance")
    return candidate_id, execution_digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--session", type=Path, action="append", required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    review_path = args.review.resolve()
    review = _load(review_path)
    proposed = {
        str(row["candidate_id"]): row for row in review.get("candidates", [])
    }
    if len(proposed) != int(review.get("candidate_count", -1)):
        raise ValueError("review candidate count or identities are inconsistent")

    runtime: dict[str, dict[str, Any]] = {}
    for raw_session in args.session:
        session_path = raw_session.resolve()
        session = _load(session_path)
        for request in session.get("requests", []):
            if not isinstance(request, dict):
                raise ValueError(f"invalid request record in {session_path}")
            candidate_id, execution_digest = _candidate_record(request)
            if candidate_id in runtime:
                raise ValueError(f"duplicate runtime evidence for {candidate_id}")
            if candidate_id not in proposed:
                raise ValueError(f"runtime evidence names unknown {candidate_id}")
            if execution_digest != str(proposed[candidate_id]["execution_digest"]):
                raise ValueError(f"execution digest mismatch for {candidate_id}")
            if request.get("status") != "complete":
                raise ValueError(f"candidate did not complete: {candidate_id}")
            original_output = Path(str(request.get("output") or ""))
            project_output = session_path.parent / original_output.name
            if not project_output.is_file():
                raise ValueError(f"relocated video does not exist: {project_output}")
            phases = request.get("phases") or {}
            runtime[candidate_id] = {
                "candidate_id": candidate_id,
                "comparison_group": proposed[candidate_id]["comparison_group"],
                "execution_digest": execution_digest,
                "video_path": str(project_output),
                "total_seconds": float(request["total_seconds"]),
                "denoise_seconds": float(phases["denoise"]),
                "video_decode_seconds": float(phases["video_decode"]),
                "actual_steps": int(request["actual_steps"]),
                "forecast_steps": int(request["forecast_steps"]),
                "width": int(request["width"]),
                "height": int(request["height"]),
                "frames": int(request["frames"]),
                "packed_tokens": int(request["execution_profile"]["packed_tokens"]),
                "peak_allocated_gib": float(request["peak_allocated_gib"]),
                "peak_reserved_gib": float(request["peak_reserved_gib"]),
                "runtime_report": str(session_path),
            }

    missing = sorted(set(proposed) - set(runtime))
    if missing:
        raise ValueError(f"missing runtime evidence for: {', '.join(missing)}")

    for candidate_id, row in proposed.items():
        measured = runtime[candidate_id]
        row["video_path"] = measured["video_path"]
        row["latency_seconds"] = measured["total_seconds"]
        row["workload_context"] = {
            key: measured[key]
            for key in ("width", "height", "frames", "packed_tokens")
        }
        row["runtime_measurements"] = {
            key: value
            for key, value in measured.items()
            if key not in {"candidate_id", "comparison_group", "video_path"}
        }

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "batch_id": review["batch_id"],
        "candidate_count": len(runtime),
        "all_execution_digests_verified": True,
        "candidates": [runtime[candidate_id] for candidate_id in proposed],
    }
    summary_path = (
        args.summary.resolve()
        if args.summary is not None
        else review_path.with_name("execution_summary.json")
    )
    review_path.write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "review": str(review_path),
        "summary": str(summary_path),
        "candidate_count": len(runtime),
        "all_execution_digests_verified": True,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
