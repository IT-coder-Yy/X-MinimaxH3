#!/usr/bin/env python3
"""Measure one real H3 block at the 720p/15 s packed-token shape.

This isolates the repeated DiT body from model loading, VAE decode, muxing and
the sampler.  Both arms use the same INT8 ConvRot checkpoint and Split0.50
attention backend; the only changed mechanism is generated-video frame
interleaving inside the transformer block.
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
        default=Path("models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
    )
    parser.add_argument(
        "--sparge-build-dir",
        type=Path,
        default=Path("runtime/extensions/sparge-sm89-py310-torch28-cu12"),
    )
    parser.add_argument("--block", type=int, default=20)
    parser.add_argument("--topk", type=float, default=0.50)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument("--mlp-chunk-tokens", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runtime/calibration/workload_routing_round16/frame_interleave_real_block.json"),
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
    if args.stride <= 1:
        raise SystemExit("--stride must be greater than one")
    sys.path.insert(0, str(args.sparge_build_dir.resolve()))

    from h3serve.native_engine.model import (
        FrameInterleaveConfig,
        FrameInterleavePlan,
        SafeTensorSource,
        assemble_pruned_block,
        build_fl2va_layout,
        comfy_kitchen_int8_kernel,
        frame_interleave_plan,
        make_split_modality_protected_sparge_attention_sm89,
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
    # 362 output frames map to 107 H3 video-latent frames; 736x1280 maps to
    # 46x80 latent pixels and 920 packed tokens per latent frame.  Text length
    # 517 reproduces the observed 100,163-token production request exactly.
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
        layout.sequence_length,
        5376,
        device=device,
        dtype=torch.bfloat16,
    )
    working = torch.empty_like(base)
    candidate_plan = FrameInterleavePlan(
        layout,
        FrameInterleaveConfig(
            stride=args.stride,
            layer_start=args.block,
            layer_stop=args.block + 1,
        ),
        device,
    )

    def reset() -> None:
        working.copy_(base)

    def run(plan) -> torch.Tensor:
        with (
            torch.inference_mode(),
            attention_protected_prefix(protected),
            attention_layer(args.block),
            frame_interleave_plan(plan),
        ):
            return block(
                working,
                timestep_rows=curve_rows,
                modulation_segments=segments,
                frequencies=frequencies,
                mlp_chunk_tokens=args.mlp_chunk_tokens,
            )

    dense_samples, dense_peak = measure(
        lambda: run(None), reset, warmup=args.warmup, repeat=args.repeat
    )
    candidate_samples, candidate_peak = measure(
        lambda: run(candidate_plan), reset, warmup=args.warmup, repeat=args.repeat
    )
    dense_ms = statistics.median(dense_samples)
    candidate_ms = statistics.median(candidate_samples)
    layer_plan = candidate_plan.for_layer(args.block)
    assert layer_plan is not None
    report = {
        "shape": {
            "tokens": layout.sequence_length,
            "protected_tokens": protected,
            "latent_frames": 107,
            "tokens_per_frame": 920,
            "hidden": 5376,
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "block": args.block,
        "attention": {"backend": "split-modality-sparge", "topk": args.topk},
        "candidate": {
            "mechanism": "protected-av-frame-interleave",
            "stride": args.stride,
            "selected_tokens": int(layer_plan.selected_indices.numel()),
            "mlp_chunk_tokens": args.mlp_chunk_tokens,
        },
        "dense": {
            "samples_ms": dense_samples,
            "median_ms": dense_ms,
            "peak_allocated_gib": dense_peak / (1024**3),
        },
        "frame_interleave": {
            "samples_ms": candidate_samples,
            "median_ms": candidate_ms,
            "peak_allocated_gib": candidate_peak / (1024**3),
        },
        "speedup": dense_ms / candidate_ms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    del block, base, working, frequencies, curve_rows
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
