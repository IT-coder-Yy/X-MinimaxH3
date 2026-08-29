#!/usr/bin/env python3
"""Validate and compact multi-phase H3 Attention calibration evidence.

The raw layer-calibration reports intentionally retain head-level diagnostics
and are too large/noisy to serve as a release-time planner model.  This tool
checks that both workload endpoints used the same measured Attention action,
merges split calibration runs, and emits:

* a compact step x layer x action Dense-disagreement table; and
* a reproducible analysis of temporal magnitude and layer-rank stability.

No Human-quality conclusion is inferred from tensor disagreement.  The output
is only a risk-model input for the finite scheduler, whose hard causal floors
and Human review gates remain separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Iterable, Mapping


SCHEMA_VERSION = "h3_phase_layer_risk_v1"
ACTION_IMPLEMENTATION_ID = "interaction_hybrid_round215_v1"
ACTION_NAMES = (
    "sparse_topk_0.0625",
    "sparse_topk_0.1",
    "sparse_topk_0.25",
    "sparse_topk_0.5",
)
EXPECTED_STEPS = (1, 3, 8, 14, 18)
REFERENCE_TOTAL_STEPS = 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("complete") is not True:
        raise ValueError(f"calibration is incomplete: {path}")
    contract = document.get("contract")
    if not isinstance(contract, dict):
        raise ValueError(f"calibration contract is missing: {path}")
    expected_contract = {
        "engine": "original",
        "heads": 56,
        "head_dim": 128,
        "full_head_budgets": [0.0625, 0.1, 0.25, 0.5],
        "temporal_correspondence_radius": 1,
        "temporal_spatial_block_radius": 1,
        "temporal_global_anchor_stride": 8,
        "temporal_global_spatial_block_radius": 0,
        "sparse_selection_mode": "interaction_hybrid",
        "timing_scope": "one complete 56-head call",
        "dense_result_returned": True,
        "weights_modified": False,
    }
    mismatches = {
        key: (contract.get(key), value)
        for key, value in expected_contract.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise ValueError(f"calibration contract mismatch in {path}: {mismatches}")
    return document


def _extract_steps(
    documents: Iterable[tuple[Path, Mapping[str, object]]],
) -> tuple[int, dict[int, dict[str, tuple[float, ...]]], list[dict[str, str]]]:
    packed_tokens: int | None = None
    merged: dict[int, dict[str, tuple[float, ...]]] = {}
    sources: list[dict[str, str]] = []
    for path, document in documents:
        contract = document["contract"]
        assert isinstance(contract, Mapping)
        current_tokens = int(contract["sequence_tokens"])
        if packed_tokens is None:
            packed_tokens = current_tokens
        elif current_tokens != packed_tokens:
            raise ValueError("one endpoint cannot merge different token lengths")
        sources.append({"path": str(path), "sha256": _sha256(path)})
        rows = document.get("steps")
        if not isinstance(rows, list):
            raise ValueError(f"step rows are missing: {path}")
        for row in rows:
            if not isinstance(row, Mapping) or row.get("complete") is not True:
                raise ValueError(f"incomplete step row: {path}")
            step_index = int(row["step_index"])
            layers = row.get("layers")
            if not isinstance(layers, list) or len(layers) != 50:
                raise ValueError(f"step {step_index} must contain 50 layers")
            if [int(item["layer"]) for item in layers] != list(range(50)):
                raise ValueError(f"step {step_index} layer order is not 0..49")
            actions: dict[str, list[float]] = {name: [] for name in ACTION_NAMES}
            for layer in layers:
                candidates = layer.get("candidates")
                if not isinstance(candidates, list):
                    raise ValueError(f"step {step_index} candidates are missing")
                by_name = {str(item["name"]): item for item in candidates}
                if set(by_name) != set(ACTION_NAMES):
                    raise ValueError(
                        f"step {step_index} layer {layer['layer']} action mismatch"
                    )
                for name in ACTION_NAMES:
                    value = float(by_name[name]["global_relative_rms"])
                    if not math.isfinite(value) or value < 0.0:
                        raise ValueError("Dense disagreement must be finite and nonnegative")
                    actions[name].append(value)
            compact = {name: tuple(values) for name, values in actions.items()}
            previous = merged.get(step_index)
            if previous is not None and previous != compact:
                raise ValueError(f"conflicting duplicate step {step_index}")
            merged[step_index] = compact
    if packed_tokens is None:
        raise ValueError("endpoint has no calibration documents")
    if tuple(sorted(merged)) != EXPECTED_STEPS:
        raise ValueError(
            f"endpoint anchors must be {EXPECTED_STEPS}, got {tuple(sorted(merged))}"
        )
    return packed_tokens, merged, sources


def _ranks(values: Iterable[float]) -> tuple[float, ...]:
    values = tuple(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + stop - 1) / 2.0
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return tuple(ranks)


def _pearson(left: Iterable[float], right: Iterable[float]) -> float:
    left = tuple(left)
    right = tuple(right)
    if len(left) != len(right) or not left:
        raise ValueError("correlation inputs must have equal nonzero length")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_norm = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_norm = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0 if left == right else 0.0
    return numerator / (left_norm * right_norm)


def _analysis(
    endpoints: Mapping[str, Mapping[int, Mapping[str, tuple[float, ...]]]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for label, steps in endpoints.items():
        per_action: dict[str, object] = {}
        for name in ACTION_NAMES:
            means = {
                str(step): statistics.fmean(steps[step][name])
                for step in EXPECTED_STEPS
            }
            rank_correlations: dict[str, float] = {}
            pearson_correlations: dict[str, float] = {}
            for left, right in zip(EXPECTED_STEPS, EXPECTED_STEPS[1:]):
                key = f"{left}:{right}"
                rank_correlations[key] = round(
                    _pearson(_ranks(steps[left][name]), _ranks(steps[right][name])),
                    8,
                )
                pearson_correlations[key] = round(
                    _pearson(steps[left][name], steps[right][name]), 8
                )
            per_action[name] = {
                "mean_global_relative_rms": {
                    key: round(value, 10) for key, value in means.items()
                },
                "terminal_over_opening": round(
                    means[str(EXPECTED_STEPS[-1])] / means[str(EXPECTED_STEPS[0])],
                    8,
                ),
                "adjacent_spearman": rank_correlations,
                "adjacent_pearson": pearson_correlations,
            }
        baseline = tuple(
            value
            for name in ACTION_NAMES
            for value in steps[3][name]
        )
        baseline_square_sum = sum(value * value for value in baseline)
        separable: dict[str, object] = {}
        for step in EXPECTED_STEPS:
            measured = tuple(
                value
                for name in ACTION_NAMES
                for value in steps[step][name]
            )
            scale = sum(
                left * right for left, right in zip(baseline, measured)
            ) / baseline_square_sum
            normalized_residual = math.sqrt(
                sum(
                    (right - scale * left) ** 2
                    for left, right in zip(baseline, measured)
                )
                / sum(value * value for value in measured)
            )
            separable[str(step)] = {
                "least_squares_scale_from_step3": round(scale, 10),
                "normalized_residual_rms": round(normalized_residual, 10),
            }
        result[label] = {
            "by_action": per_action,
            "separable_temporal_model": separable,
        }
    return result


def build_evidence(
    short_paths: tuple[Path, ...],
    long_paths: tuple[Path, ...],
) -> dict[str, object]:
    short_tokens, short_steps, short_sources = _extract_steps(
        (path, _load(path)) for path in short_paths
    )
    long_tokens, long_steps, long_sources = _extract_steps(
        (path, _load(path)) for path in long_paths
    )
    if short_tokens >= long_tokens:
        raise ValueError("short endpoint must contain fewer packed tokens")

    def endpoint_document(
        packed_tokens: int,
        steps: Mapping[int, Mapping[str, tuple[float, ...]]],
        sources: list[dict[str, str]],
    ) -> dict[str, object]:
        return {
            "packed_tokens": packed_tokens,
            "sources": sources,
            "anchors": [
                {
                    "step_index": step,
                    "trajectory_progress": round(
                        step / (REFERENCE_TOTAL_STEPS - 1), 12
                    ),
                    "actions": {
                        name: [round(value, 10) for value in steps[step][name]]
                        for name in ACTION_NAMES
                    },
                }
                for step in EXPECTED_STEPS
            ],
        }

    evidence = {
        "schema_version": SCHEMA_VERSION,
        "metric": "global_relative_rms_to_dense",
        "reference_total_steps": REFERENCE_TOTAL_STEPS,
        "temporal_interpolation": "piecewise_linear_in_normalized_step_index",
        "shape_interpolation": "piecewise_linear_in_packed_tokens",
        "action_implementation_id": ACTION_IMPLEMENTATION_ID,
        "actions": list(ACTION_NAMES),
        "endpoints": {
            "short": endpoint_document(short_tokens, short_steps, short_sources),
            "long": endpoint_document(long_tokens, long_steps, long_sources),
        },
        "analysis": _analysis({"short": short_steps, "long": long_steps}),
        "limitations": [
            "Dense tensor disagreement is a scheduler risk surrogate, not Human acceptance.",
            "Five trajectory anchors are linearly interpolated; unmeasured prompts remain epistemically uncertain.",
            "Hard causal, opening and terminal safety constraints are maintained separately.",
        ],
    }
    return evidence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--short", type=Path, action="append", required=True)
    parser.add_argument("--long", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    document = build_evidence(tuple(args.short), tuple(args.long))
    payload = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
