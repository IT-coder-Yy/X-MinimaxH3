#!/usr/bin/env python3
"""Falsify fixed ConvRot-group pruning on one real H3 FC2 boundary.

The input is an opt-in ``quantized_fc2`` research capture.  Groups are ranked
on one deterministic row sample and evaluated on a disjoint sample.  This is
an oracle study only: it never modifies the generation runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-rows", type=int, default=1024)
    parser.add_argument("--evaluation-rows", type=int, default=1024)
    return parser.parse_args()


def evenly_spaced(start: int, stop: int, count: int) -> torch.Tensor:
    if stop <= start:
        raise ValueError("empty row interval")
    count = min(count, stop - start)
    return torch.linspace(start, stop - 1, count).round().long().unique()


def main() -> None:
    args = parse_args()
    document = torch.load(args.capture, map_location="cpu", weights_only=False)
    if document.get("kind") != "quantized_fc2":
        raise SystemExit("capture is not a quantized_fc2 boundary")
    qx = document["qx_video"]
    x_scale = document["x_scale_video"].reshape(-1)
    weight = document["qweight"]
    weight_scale = document["weight_scale"].reshape(-1)
    group = int(document["convrot_group_size"])
    if qx.shape[1] % group or weight.shape[1] != qx.shape[1]:
        raise SystemExit("captured FC2 operands do not align to ConvRot groups")

    split = qx.shape[0] // 2
    calibration_indices = evenly_spaced(0, split, args.calibration_rows)
    evaluation_indices = evenly_spaced(split, qx.shape[0], args.evaluation_rows)
    calibration = qx.index_select(0, calibration_indices).float()
    groups = qx.shape[1] // group
    calibration_rms = calibration.square().view(-1, groups, group).mean((0, 2)).sqrt()
    weight_rms = weight.float().square().view(weight.shape[0], groups, group).mean((0, 2)).sqrt()
    proxy = calibration_rms * weight_rms
    order = torch.argsort(proxy, descending=True)
    del calibration

    eval_q = qx.index_select(0, evaluation_indices).cuda()
    eval_x_scale = x_scale.index_select(0, evaluation_indices).cuda().float()
    gpu_weight = weight.cuda()
    gpu_weight_scale = weight_scale.cuda().float()
    reference_i32 = torch._int_mm(eval_q, gpu_weight.t())
    reference = (
        reference_i32.float()
        * eval_x_scale[:, None]
        * gpu_weight_scale[None, :]
    )
    reference_norm = reference.norm().clamp_min(1e-12)
    rows: list[dict[str, object]] = []
    for keep_fraction in (0.50, 0.625, 0.75, 0.875, 0.9375):
        keep_groups = max(1, round(groups * keep_fraction))
        selected_groups = order[:keep_groups].sort().values
        selected_columns = (
            selected_groups[:, None] * group
            + torch.arange(group, dtype=torch.long)[None, :]
        ).reshape(-1)
        q_selected = eval_q.index_select(1, selected_columns.cuda()).contiguous()
        w_selected = gpu_weight.index_select(1, selected_columns.cuda()).contiguous()
        candidate_i32 = torch._int_mm(q_selected, w_selected.t())
        candidate = (
            candidate_i32.float()
            * eval_x_scale[:, None]
            * gpu_weight_scale[None, :]
        )
        difference = candidate - reference
        row_cosine = F.cosine_similarity(candidate, reference, dim=1, eps=1e-12)
        rows.append(
            {
                "keep_fraction": keep_groups / groups,
                "kept_groups": keep_groups,
                "total_groups": groups,
                "relative_l2": float(difference.norm() / reference_norm),
                "mean_absolute_error": float(difference.abs().mean()),
                "mean_row_cosine": float(row_cosine.mean()),
                "p01_row_cosine": float(torch.quantile(row_cosine, 0.01)),
                "minimum_row_cosine": float(row_cosine.min()),
            }
        )
        del q_selected, w_selected, candidate_i32, candidate, difference

    report = {
        "schema_version": 1,
        "method": "fixed_convrot_group_pruning_oracle",
        "capture": str(args.capture.resolve()),
        "step_index": int(document["step_index"]),
        "layer": int(document["layer"]),
        "shape": {
            "video_rows": int(qx.shape[0]),
            "input_features": int(qx.shape[1]),
            "output_features": int(weight.shape[0]),
            "convrot_group_size": group,
        },
        "calibration_rows": int(calibration_indices.numel()),
        "evaluation_rows": int(evaluation_indices.numel()),
        "ranking": "activation_rms_times_weight_rms",
        "candidates": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
