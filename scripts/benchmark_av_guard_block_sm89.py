#!/usr/bin/env python3
"""Benchmark one real H3 block with protected AV-prefix refresh."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
    )
    parser.add_argument(
        "--sparge-build-dir",
        type=Path,
        default=Path("runtime/extensions/sparge-sm89-py310-torch28-cu12"),
    )
    parser.add_argument("--block", type=int, default=20)
    parser.add_argument("--topk", type=float, default=0.50)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument(
        "--active-video-ratio",
        type=float,
        default=0.0,
        help="also benchmark selected same-coordinate video-row refresh",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runtime/calibration/workload_routing_round24/av_guard_real_block.json"),
    )
    return parser.parse_args()


def measure(operation, reset, *, warmup: int, repeat: int) -> tuple[list[float], int]:
    for _ in range(warmup):
        reset()
        operation()
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples: list[float] = []
    for _ in range(repeat):
        reset()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    return samples, int(torch.cuda.max_memory_allocated())


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this benchmark requires one RTX 4090 / SM89 GPU")
    if not 0 <= args.block < 50:
        raise SystemExit("--block must lie inside [0, 50)")
    if not 0.0 <= args.active_video_ratio <= 1.0:
        raise SystemExit("--active-video-ratio must lie inside [0, 1]")
    sys.path.insert(0, str(args.sparge_build_dir.resolve()))

    from h3serve.native_engine.model import (
        SafeTensorSource,
        assemble_pruned_block,
        build_fl2va_layout,
        comfy_kitchen_int8_kernel,
        make_split_modality_protected_sparge_attention_sm89,
        refresh_protected_modalities,
        refresh_selected_video_tiles,
    )
    from h3serve.native_engine.model.dit import FullH3DiT
    from h3serve.native_engine.model.kernels import (
        attention_layer,
        attention_protected_prefix,
    )
    from h3serve.native_engine.model.layers import rope_frequencies, rope_rotation_table
    from h3serve.native_engine.model.lora import AdaLNCurveRows, interpolate_curve
    from h3serve.native_engine.sm89_policy import configure_sm89_runtime

    configure_sm89_runtime(quant_backend="cuda", smoke_test=True)
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    layout = build_fl2va_layout(
        text_length=517,
        latent_frames=107,
        latent_height=46,
        latent_width=80,
        audio_frames=603,
    )
    if layout.sequence_length != 100_163:
        raise RuntimeError(f"unexpected probe length: {layout.sequence_length}")
    protected = layout.segment("video", last=True).start
    sigma = torch.tensor([0.5], device=device)
    unique_timesteps, segments, _ = FullH3DiT._timestep_plan(
        sigma,
        layout,
        sigma_shift_video=5.0,
        sigma_shift_audio=2.0,
        visual_condition_timestep=0.999,
        audio_condition_timestep=1.0,
        text_token_tags=None,
        device=device,
    )
    attention = make_split_modality_protected_sparge_attention_sm89(args.topk)
    with SafeTensorSource(str(args.checkpoint)) as source:
        block = assemble_pruned_block(
            args.block,
            source,
            device=device,
            compute_dtype=torch.bfloat16,
            int8_kernel=comfy_kitchen_int8_kernel,
            attention_backend=attention,
        )
        block.eval().requires_grad_(False)
        curve_rows = AdaLNCurveRows(
            compressed=interpolate_curve(
                source.tensor("adaln_t_table").to(device), unique_timesteps
            )
        )
        frequencies = rope_rotation_table(
            rope_frequencies(
                layout.position_ids.to(device), source.tensor("rope.inv_freq").to(device)
            ),
            torch.bfloat16,
        )

    base = torch.randn(
        layout.sequence_length, 5376, device=device, dtype=torch.bfloat16
    )
    working = torch.empty_like(base)
    active_video_indices = None
    if args.active_video_ratio > 0.0:
        video_tokens = layout.sequence_length - protected
        query_block = 128
        video_blocks = (video_tokens + query_block - 1) // query_block
        active_blocks = max(1, round(video_blocks * args.active_video_ratio))
        selected_blocks = torch.linspace(
            0,
            video_blocks - 1,
            active_blocks,
            device=device,
            dtype=torch.float32,
        ).round().long().unique(sorted=True)
        relative = torch.cat(
            tuple(
                torch.arange(
                    int(block_index) * query_block,
                    min((int(block_index) + 1) * query_block, video_tokens),
                    device=device,
                    dtype=torch.long,
                )
                for block_index in selected_blocks
            )
        )
        active_video_indices = relative + protected

    def reset() -> None:
        working.copy_(base)

    def dense() -> torch.Tensor:
        with attention_protected_prefix(protected), attention_layer(args.block):
            return block(
                working,
                timestep_rows=curve_rows,
                modulation_segments=segments,
                frequencies=frequencies,
                mlp_chunk_tokens=None,
            )

    def guarded() -> torch.Tensor:
        with attention_protected_prefix(protected), attention_layer(args.block):
            return refresh_protected_modalities(
                block,
                working,
                protected_tokens=protected,
                timestep_rows=curve_rows,
                modulation_segments=segments,
                frequencies=frequencies,
            )

    def selected_guarded() -> torch.Tensor:
        if active_video_indices is None:
            raise RuntimeError("selected benchmark requires active video rows")
        with attention_protected_prefix(protected), attention_layer(args.block):
            return refresh_selected_video_tiles(
                block,
                working,
                protected_tokens=protected,
                active_video_indices=active_video_indices,
                timestep_rows=curve_rows,
                modulation_segments=segments,
                frequencies=frequencies,
            )

    reset()
    dense_result = dense().clone()
    reset()
    dense_repeat_result = dense().clone()
    reset()
    guarded_result = guarded().clone()
    prefix_delta = (dense_result[:protected].float() - guarded_result[:protected].float()).abs()
    dense_repeat_delta = (
        dense_result[:protected].float() - dense_repeat_result[:protected].float()
    ).abs()
    prefix_reference = dense_result[:protected].float()
    prefix_candidate = guarded_result[:protected].float()
    prefix_max_abs = float(prefix_delta.max().cpu())
    prefix_mean_abs = float(prefix_delta.mean().cpu())
    prefix_rmse = float(prefix_delta.square().mean().sqrt().cpu())
    prefix_cosine = float(
        torch.nn.functional.cosine_similarity(
            prefix_reference.reshape(1, -1),
            prefix_candidate.reshape(1, -1),
        ).cpu()
    )
    dense_repeat_mean_abs = float(dense_repeat_delta.mean().cpu())
    dense_repeat_max_abs = float(dense_repeat_delta.max().cpu())
    dense_repeat_cosine = float(
        torch.nn.functional.cosine_similarity(
            prefix_reference.reshape(1, -1),
            dense_repeat_result[:protected].float().reshape(1, -1),
        ).cpu()
    )
    video_unchanged = torch.equal(guarded_result[protected:], base[protected:])
    del dense_result, dense_repeat_result, guarded_result, prefix_delta, dense_repeat_delta
    torch.cuda.empty_cache()

    dense_samples, dense_peak = measure(
        dense, reset, warmup=args.warmup, repeat=args.repeat
    )
    guarded_samples, guarded_peak = measure(
        guarded, reset, warmup=args.warmup, repeat=args.repeat
    )
    dense_ms = statistics.median(dense_samples)
    guarded_ms = statistics.median(guarded_samples)
    report = {
        "shape": {
            "tokens": layout.sequence_length,
            "protected_tokens": protected,
            "video_tokens": layout.sequence_length - protected,
            "hidden": 5376,
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "block": args.block,
        "attention": {"backend": "split-modality-sparge", "topk": args.topk},
        "candidate": {
            "mechanism": "full-kv-protected-query-and-mlp-refresh",
            "video_rows_unchanged": bool(video_unchanged),
            "prefix_vs_full_block": {
                "max_abs": prefix_max_abs,
                "mean_abs": prefix_mean_abs,
                "rmse": prefix_rmse,
                "cosine": prefix_cosine,
            },
        },
        "dense": {
            "samples_ms": dense_samples,
            "median_ms": dense_ms,
            "peak_allocated_gib": dense_peak / (1024**3),
            "repeat_variance": {
                "max_abs": dense_repeat_max_abs,
                "mean_abs": dense_repeat_mean_abs,
                "cosine": dense_repeat_cosine,
            },
        },
        "av_guard": {
            "samples_ms": guarded_samples,
            "median_ms": guarded_ms,
            "peak_allocated_gib": guarded_peak / (1024**3),
        },
        "speedup": dense_ms / guarded_ms,
        "note": "prefix differences include the accepted dense-prefix attention kernel's FP8/INT8 reduction-order variance",
    }
    if active_video_indices is not None:
        reset()
        dense_for_selected = dense().clone()
        reset()
        selected_result = selected_guarded().clone()
        selected_delta = (
            dense_for_selected.index_select(0, active_video_indices).float()
            - selected_result.index_select(0, active_video_indices).float()
        ).abs()
        selected_samples, selected_peak = measure(
            selected_guarded, reset, warmup=args.warmup, repeat=args.repeat
        )
        selected_ms = statistics.median(selected_samples)
        report["selected_video"] = {
            "requested_ratio": args.active_video_ratio,
            "active_tokens": int(active_video_indices.numel()),
            "active_ratio": float(active_video_indices.numel()) / float(
                layout.sequence_length - protected
            ),
            "samples_ms": selected_samples,
            "median_ms": selected_ms,
            "peak_allocated_gib": selected_peak / (1024**3),
            "speedup": dense_ms / selected_ms,
            "active_vs_full_block": {
                "max_abs": float(selected_delta.max().cpu()),
                "mean_abs": float(selected_delta.mean().cpu()),
            },
        }
        del dense_for_selected, selected_result, selected_delta
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    del block, base, working, frequencies, curve_rows
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
