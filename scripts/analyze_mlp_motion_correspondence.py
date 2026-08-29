#!/usr/bin/env python3
"""Evaluate confidence-gated temporal reuse on one real H3 MLP capture.

This is an offline falsification tool.  Odd latent frames may borrow an exact
MLP residual from a locally matched token in an adjacent even frame.  Rows
that do not meet the requested hidden-state confidence remain exact.  The
script reports quality/error curves; it does not alter the serving runtime.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--sketch-width", type=int, default=96)
    parser.add_argument("--batch-rows", type=int, default=512)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument(
        "--thresholds",
        default="0.90,0.93,0.95,0.97,0.98,0.99,0.995",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def local_candidates(
    frame: int,
    *,
    frames: int,
    rows: int,
    columns: int,
    radius: int,
    device: torch.device,
) -> torch.Tensor:
    anchors = []
    for candidate_frame in (frame - 1, frame + 1):
        if 0 <= candidate_frame < frames and candidate_frame % 2 == 0:
            anchors.append(candidate_frame)
    positions = []
    for row in range(rows):
        for column in range(columns):
            choices = []
            for anchor in anchors:
                for dr in range(-radius, radius + 1):
                    rr = min(rows - 1, max(0, row + dr))
                    for dc in range(-radius, radius + 1):
                        cc = min(columns - 1, max(0, column + dc))
                        choices.append(anchor * rows * columns + rr * columns + cc)
            positions.append(choices)
    return torch.tensor(positions, dtype=torch.long, device=device)


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if args.radius < 0 or args.sketch_width <= 0 or args.batch_rows <= 0:
        raise SystemExit("radius must be non-negative; widths must be positive")
    thresholds = tuple(float(item) for item in args.thresholds.split(","))
    if not thresholds or any(not 0.0 <= item <= 1.0 for item in thresholds):
        raise SystemExit("thresholds must lie inside [0,1]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    capture = torch.load(args.capture, map_location="cpu", weights_only=True)
    hidden = capture["hidden_video"].to(device)
    delta = capture["delta_video"].to(device)
    frames = int(capture["latent_frames"])
    frame_tokens = int(capture["frame_tokens"])
    # H3's 1280x736 patch grid is 40x23.  Factor generically while preferring
    # the wide layout used by this research workload.
    factors = [value for value in range(1, int(frame_tokens**0.5) + 1) if frame_tokens % value == 0]
    rows = max(factors)
    columns = frame_tokens // rows
    if rows > columns:
        rows, columns = columns, rows
    if rows * columns != frame_tokens:
        raise RuntimeError("could not factor the video frame token grid")

    channel_indices = torch.linspace(
        0,
        hidden.shape[1] - 1,
        min(args.sketch_width, hidden.shape[1]),
        device=device,
    ).round().long().unique()
    sketch = F.normalize(hidden.index_select(1, channel_indices).float(), dim=1)
    target_indices_parts: list[torch.Tensor] = []
    match_indices_parts: list[torch.Tensor] = []
    confidence_parts: list[torch.Tensor] = []
    margin_parts: list[torch.Tensor] = []
    oracle_score_parts: list[torch.Tensor] = []
    oracle_match_parts: list[torch.Tensor] = []
    normalized_delta = F.normalize(delta.float(), dim=1) if args.oracle else None
    torch.cuda.synchronize() if device.type == "cuda" else None
    started = time.perf_counter()
    for frame in range(1, frames, 2):
        targets = torch.arange(
            frame * frame_tokens,
            (frame + 1) * frame_tokens,
            dtype=torch.long,
            device=device,
        )
        candidates = local_candidates(
            frame,
            frames=frames,
            rows=rows,
            columns=columns,
            radius=args.radius,
            device=device,
        )
        for start in range(0, frame_tokens, args.batch_rows):
            stop = min(start + args.batch_rows, frame_tokens)
            local_targets = targets[start:stop]
            candidate_rows = candidates[start:stop]
            query = sketch.index_select(0, local_targets)
            keys = sketch.index_select(0, candidate_rows.flatten()).view(
                candidate_rows.shape[0], candidate_rows.shape[1], -1
            )
            scores = torch.einsum("bd,bkd->bk", query, keys)
            best_two = torch.topk(scores, min(2, scores.shape[1]), dim=1)
            best_pos = best_two.indices[:, 0]
            best = best_two.values[:, 0]
            second = best_two.values[:, 1] if scores.shape[1] > 1 else torch.zeros_like(best)
            match = candidate_rows.gather(1, best_pos.unsqueeze(1)).squeeze(1)
            target_indices_parts.append(local_targets)
            match_indices_parts.append(match)
            confidence_parts.append(best)
            margin_parts.append(best - second)
            if normalized_delta is not None:
                delta_query = normalized_delta.index_select(0, local_targets)
                delta_keys = normalized_delta.index_select(
                    0, candidate_rows.flatten()
                ).view(candidate_rows.shape[0], candidate_rows.shape[1], -1)
                delta_scores = torch.einsum("bd,bkd->bk", delta_query, delta_keys)
                oracle_score, oracle_pos = delta_scores.max(dim=1)
                oracle_score_parts.append(oracle_score)
                oracle_match_parts.append(
                    candidate_rows.gather(1, oracle_pos.unsqueeze(1)).squeeze(1)
                )
    if device.type == "cuda":
        torch.cuda.synchronize()
    match_seconds = time.perf_counter() - started
    targets = torch.cat(target_indices_parts)
    matches = torch.cat(match_indices_parts)
    confidence = torch.cat(confidence_parts)
    margin = torch.cat(margin_parts)
    truth = delta.index_select(0, targets).float()
    estimate = delta.index_select(0, matches).float()
    row_cosine = F.cosine_similarity(estimate, truth, dim=1)
    row_relative_l2 = (
        (estimate - truth).square().sum(1)
        / truth.square().sum(1).clamp_min(1.0e-12)
    ).sqrt()
    total_delta_energy = delta.float().square().sum()
    oracle = None
    if oracle_score_parts:
        oracle_scores = torch.cat(oracle_score_parts)
        oracle_matches = torch.cat(oracle_match_parts)
        oracle_estimate = delta.index_select(0, oracle_matches).float()
        oracle_relative_l2 = (
            (oracle_estimate - truth).square().sum(1)
            / truth.square().sum(1).clamp_min(1.0e-12)
        ).sqrt()
        oracle = {
            "delta_cosine_mean": float(oracle_scores.mean()),
            "delta_cosine_p05": float(torch.quantile(oracle_scores, 0.05)),
            "delta_cosine_p50": float(torch.quantile(oracle_scores, 0.50)),
            "delta_cosine_p95": float(torch.quantile(oracle_scores, 0.95)),
            "rows_cosine_ge_0_95": float((oracle_scores >= 0.95).float().mean()),
            "rows_cosine_ge_0_99": float((oracle_scores >= 0.99).float().mean()),
            "relative_l2_mean": float(oracle_relative_l2.mean()),
        }
    curves = []
    for threshold in thresholds:
        # Margin prevents a superficially high cosine score from accepting an
        # ambiguous match in repetitive backgrounds.
        selected = (confidence >= threshold) & (margin >= 0.002)
        count = int(selected.sum())
        error_energy = (
            (estimate[selected] - truth[selected]).square().sum()
            if count
            else torch.tensor(0.0, device=device)
        )
        curves.append(
            {
                "threshold": threshold,
                "reused_target_fraction": count / max(1, int(targets.numel())),
                "reused_all_video_fraction": count / (frames * frame_tokens),
                "hybrid_global_relative_l2": float(
                    torch.sqrt(error_energy / total_delta_energy.clamp_min(1.0e-12))
                ),
                "selected_delta_cosine_mean": (
                    float(row_cosine[selected].mean()) if count else None
                ),
                "selected_delta_cosine_p05": (
                    float(torch.quantile(row_cosine[selected], 0.05)) if count else None
                ),
                "selected_delta_relative_l2_mean": (
                    float(row_relative_l2[selected].mean()) if count else None
                ),
            }
        )
    report = {
        "capture": str(args.capture.resolve()),
        "capture_metadata": {
            key: value
            for key, value in capture.items()
            if not torch.is_tensor(value)
        },
        "shape": {
            "frames": frames,
            "rows": rows,
            "columns": columns,
            "hidden": int(hidden.shape[1]),
            "target_rows": int(targets.numel()),
        },
        "matching": {
            "adjacent_even_frames": True,
            "spatial_radius": args.radius,
            "sketch_channels": int(channel_indices.numel()),
            "seconds": match_seconds,
            "hidden_confidence_mean": float(confidence.mean()),
            "hidden_confidence_p05": float(torch.quantile(confidence, 0.05)),
            "hidden_confidence_p50": float(torch.quantile(confidence, 0.50)),
            "hidden_confidence_p95": float(torch.quantile(confidence, 0.95)),
            "confidence_delta_cosine_correlation": float(
                torch.corrcoef(torch.stack((confidence, row_cosine)))[0, 1]
            ),
        },
        "local_delta_oracle": oracle,
        "curves": curves,
        "weights_modified": False,
        "runtime_modified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
