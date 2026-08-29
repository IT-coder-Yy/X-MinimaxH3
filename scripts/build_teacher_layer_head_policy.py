#!/usr/bin/env python3
"""Convert a dense-teacher layer/head probe into a bounded causal policy.

The builder identifies the robust high-error layer band and protects only the
upper error quartile of heads within each selected layer.  Within that band,
head budgets are assigned continuously from that head's robust relative-L1
score.  This makes the policy reproducible from evidence instead of a list of
hand-picked H3 knobs.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("probe", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _budgets(
    errors: list[float],
    base: list[float],
    ceiling: float,
) -> list[float]:
    # Only the upper error quartile receives extra compute.  Protecting every
    # above-median head recreated most of the whole-layer dense-island cost.
    # P75/P95 keeps the repair focused on the measured causal tail.
    threshold = _quantile(errors, 0.75)
    high = _quantile(errors, 0.95)
    scale = max(high - threshold, 1e-8)
    result = []
    for error, floor in zip(errors, base):
        # The square-root response protects moderately risky causal heads
        # without forcing the entire layer close to dense attention.
        risk = min(1.0, max(0.0, (error - threshold) / scale)) ** 0.5
        result.append(round(floor + risk * (ceiling - floor), 6))
    return result


def main() -> int:
    args = _args()
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    steps = probe["steps"]
    layer_heads = {layer: [[] for _ in range(56)] for layer in range(50)}
    layer_scores = {layer: [] for layer in range(50)}
    for step in steps:
        for row in step["layers"]:
            layer = int(row["layer"])
            layer_scores[layer].append(float(row["mean_relative_l1"]))
            for head, error in enumerate(row["head_relative_l1"]):
                layer_heads[layer][head].append(float(error))

    aggregate = {
        layer: statistics.mean(values) for layer, values in layer_scores.items()
    }
    layer_values = list(aggregate.values())
    layer_median = statistics.median(layer_values)
    layer_mad = statistics.median(
        abs(value - layer_median) for value in layer_values
    )
    layer_threshold = layer_median + 1.5 * layer_mad
    selected = tuple(
        layer
        for layer, score in sorted(
            aggregate.items(), key=lambda item: item[1], reverse=True
        )
        if score >= layer_threshold
    )
    default_floor = [0.070 if value >= 0.10 else 0.065 for value in probe["contract"]["head_budgets"]]
    recovery_floor = [0.125 if value >= 0.10 else 0.100 for value in probe["contract"]["head_budgets"]]
    anchor_floor = [0.180 if value >= 0.10 else 0.150 for value in probe["contract"]["head_budgets"]]

    phases = {"default": {}, "anchor": {}, "recovery": {}}
    evidence = {}
    for layer in selected:
        errors = [statistics.mean(values) for values in layer_heads[layer]]
        phases["default"][str(layer)] = _budgets(errors, default_floor, 1.0)
        phases["anchor"][str(layer)] = _budgets(errors, anchor_floor, 1.0)
        phases["recovery"][str(layer)] = _budgets(errors, recovery_floor, 1.0)
        evidence[str(layer)] = {
            "mean_relative_l1": aggregate[layer],
            "head_median_relative_l1": statistics.median(errors),
            "head_p90_relative_l1": _quantile(errors, 0.90),
        }

    document = {
        "schema_version": 1,
        "method": "dense_teacher_top5_continuous_head_rebate",
        "source_probe": str(args.probe.resolve()),
        "selected_layers": list(selected),
        "default_steps": [1, 2, 3, 4, 6, 8, 11, 14],
        "anchor_steps": [0],
        "recovery_steps": [17, 18, 19],
        "phase_layer_head_topks": phases,
        "evidence": evidence,
        "constraints": {
            "requested_actual_steps_unchanged": True,
            "weights_unchanged": True,
            "layer_error_threshold": layer_threshold,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
