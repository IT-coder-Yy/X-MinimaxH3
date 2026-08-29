#!/usr/bin/env python3
"""Benchmark same-frame spatial Query lattice refresh on one real H3 block.

Unlike the rejected frame-interleave experiment, every latent frame contributes
active queries and no hidden state or residual is copied between frames.  Full
K/V, text/condition/audio prefix rows and the accepted MTCR attention policy
remain available.  The probe measures whether this is a large enough second
lever before wiring it into full denoising.
"""

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
        default=Path(
            "models/diffusion_models/"
            "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
        ),
    )
    parser.add_argument(
        "--sparge-build-dir",
        type=Path,
        default=Path("runtime/extensions/sparge-sm89-py310-torch28-cu12"),
    )
    parser.add_argument("--block", type=int, default=20)
    parser.add_argument("--step", type=int, default=11)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runtime/calibration/workload_routing_round76/"
            "spatial_query_lattice_real_block.json"
        ),
    )
    return parser.parse_args()


def measure(operation, reset, *, warmup: int, repeat: int):
    for _ in range(warmup):
        reset()
        operation()
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
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


def error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.float()
    cand = candidate.float()
    delta = ref - cand
    return {
        "mean_abs": float(delta.abs().mean().cpu()),
        "rmse": float(delta.square().mean().sqrt().cpu()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                ref.reshape(1, -1), cand.reshape(1, -1)
            ).cpu()
        ),
    }


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this benchmark requires one RTX 4090 / SM89 GPU")
    if not 0 <= args.block < 50 or not 0 <= args.step < 20:
        raise SystemExit("block/step lies outside H3")
    if args.stride < 2 or not 0 <= args.phase < args.stride:
        raise SystemExit("invalid lattice stride/phase")
    sys.path.insert(0, str(args.sparge_build_dir.resolve()))

    from h3serve.native_engine.model import (
        SafeTensorSource,
        SplitModalityProtectedSpargeAttentionBackend,
        assemble_pruned_block,
        build_fl2va_layout,
        comfy_kitchen_int8_kernel,
    )
    from h3serve.native_engine.model.dit import FullH3DiT
    from h3serve.native_engine.model.kernels import (
        attention_layer,
        attention_protected_prefix,
        attention_step,
        attention_video_layout,
    )
    from h3serve.native_engine.model.layers import rope_frequencies, rope_rotation_table
    from h3serve.native_engine.model.lora import AdaLNCurveRows, interpolate_curve
    from h3serve.native_engine.model.modality_refresh import refresh_selected_video_tiles
    from h3serve.native_engine.sm89_policy import configure_sm89_runtime

    configure_sm89_runtime(quant_backend="cuda", smoke_test=True)
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
    frame_tokens = 920
    spatial_h, spatial_w = 23, 40
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
    attention = SplitModalityProtectedSpargeAttentionBackend(
        0.10,
        experimental_minimum_topk=0.0625,
        temporal_correspondence_radius=1,
        temporal_spatial_block_radius=1,
        temporal_global_anchor_stride=8,
    )
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

    # Sparge scores one 128-row Query block at a time.  Selecting individual
    # checkerboard tokens and then repacking them changes the block statistics
    # even for active rows.  Keep original 128-row groups intact and rotate
    # the chosen groups across layers.  Each 920-token frame intersects seven
    # or eight groups, so every frame remains represented in every phase.
    video_tokens = 107 * frame_tokens
    query_blocks = (video_tokens + 127) // 128
    active_parts = []
    for query_block in range(query_blocks):
        if query_block % args.stride != args.phase:
            continue
        start = query_block * 128
        stop = min(start + 128, video_tokens)
        active_parts.append(
            torch.arange(protected + start, protected + stop, device=device)
        )
    active_video = torch.cat(active_parts).contiguous()
    base = torch.randn(
        layout.sequence_length, 5376, device=device, dtype=torch.bfloat16
    )
    working = torch.empty_like(base)

    def reset() -> None:
        working.copy_(base)

    def dense() -> torch.Tensor:
        with (
            attention_protected_prefix(protected),
            attention_video_layout(107, frame_tokens),
            attention_step(args.step, 20),
            attention_layer(args.block),
        ):
            return block(
                working,
                timestep_rows=curve_rows,
                modulation_segments=segments,
                frequencies=frequencies,
                mlp_chunk_tokens=None,
            )

    def lattice() -> torch.Tensor:
        with (
            attention_protected_prefix(protected),
            attention_video_layout(107, frame_tokens),
            attention_step(args.step, 20),
            attention_layer(args.block),
        ):
            return refresh_selected_video_tiles(
                block,
                working,
                protected_tokens=protected,
                active_video_indices=active_video,
                timestep_rows=curve_rows,
                modulation_segments=segments,
                frequencies=frequencies,
            )

    reset()
    dense_reference = dense().clone()
    reset()
    lattice_reference = lattice().clone()
    dense_samples, dense_peak = measure(
        dense, reset, warmup=args.warmup, repeat=args.repeat
    )
    lattice_samples, lattice_peak = measure(
        lattice, reset, warmup=args.warmup, repeat=args.repeat
    )
    selected = torch.cat(
        (torch.arange(protected, device=device), active_video), dim=0
    )
    dense_ms = statistics.median(dense_samples)
    lattice_ms = statistics.median(lattice_samples)
    report = {
        "shape": {
            "tokens": layout.sequence_length,
            "protected_tokens": protected,
            "latent_frames": 107,
            "spatial_grid": [spatial_h, spatial_w],
        },
        "candidate": {
            "mechanism": "same-frame-spatial-query-lattice",
            "stride": args.stride,
            "phase": args.phase,
            "query_block_rows": 128,
            "block_aligned": True,
            "active_video_tokens": int(active_video.numel()),
            "active_video_fraction": float(active_video.numel() / (107 * frame_tokens)),
            "cross_frame_interpolation": False,
            "full_kv": True,
        },
        "dense": {
            "samples_ms": dense_samples,
            "median_ms": dense_ms,
            "peak_allocated_gib": dense_peak / 1024**3,
        },
        "lattice": {
            "samples_ms": lattice_samples,
            "median_ms": lattice_ms,
            "peak_allocated_gib": lattice_peak / 1024**3,
        },
        "speedup": dense_ms / lattice_ms,
        "full_output_error": error(dense_reference, lattice_reference),
        "selected_output_error": error(
            dense_reference.index_select(0, selected),
            lattice_reference.index_select(0, selected),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
