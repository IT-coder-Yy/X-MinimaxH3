#!/usr/bin/env python3
"""Run one production-shaped H3 Video-VAE long decode on RTX 4090."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from h3serve.native_engine.adapters.real_vae import (
    decode_native_video,
    load_native_video_vae,
    select_uint8_postprocess_frame_chunk,
)
from h3serve.native_engine.adapters.vae_compile import (
    enable_transformer_block_compile,
    prewarm_transformer_block_compile,
    transformer_block_compile,
)


def _sha256_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(memoryview(value.numpy()).cast("B")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1088)
    parser.add_argument("--frames", type=int, default=362)
    parser.add_argument("--tile-size", type=int, default=288)
    parser.add_argument("--temporal-host-chunk-frames", type=int)
    parser.add_argument("--enforce-vram-gib", type=float)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--minimax-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this benchmark requires one RTX 4090 / SM89 GPU")
    if args.temporal_host_chunk_frames is not None and args.temporal_host_chunk_frames <= 0:
        raise SystemExit("--temporal-host-chunk-frames must be positive")
    if args.enforce_vram_gib is not None:
        physical_bytes = torch.cuda.get_device_properties(0).total_memory
        cap_bytes = int(args.enforce_vram_gib * 1024**3)
        if cap_bytes <= 0 or cap_bytes > physical_bytes:
            raise SystemExit("--enforce-vram-gib must lie inside physical VRAM")
        torch.cuda.set_per_process_memory_fraction(cap_bytes / physical_bytes, 0)
    if (
        args.width % 32
        or args.height % 32
        or args.frames < 5
        or (args.frames - 5) % 17
    ):
        raise SystemExit("invalid H3 output geometry")

    checkpoint = args.model_root / "vae/minimax_h3_video_vae_fp16.safetensors"
    load_started = time.perf_counter()
    model, mean, std = load_native_video_vae(
        args.minimax_source,
        checkpoint,
        device="cuda:0",
        tile_size=args.tile_size,
        tile_batch_size=1,
        compile_feed_forward=True,
    )
    enable_transformer_block_compile(model)
    prewarm_transformer_block_compile(model)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    latent_frames = ((args.frames - 5) // 17) * 5 + 2
    generator = torch.Generator("cpu").manual_seed(82303)
    normalized = torch.randn(
        (1, 24, latent_frames, args.height // 16, args.width // 16),
        generator=generator,
        dtype=torch.float32,
    ).cuda()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    started = time.perf_counter()
    with transformer_block_compile(True):
        output = decode_native_video(
            model,
            normalized,
            mean,
            std,
            args.frames,
            output_dtype="uint8",
            temporal_host_chunk_frames=args.temporal_host_chunk_frames,
        )
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    output_sha256 = _sha256_tensor(output)
    shape = (1, 3, args.frames, args.height, args.width)
    report = {
        "schema_version": "h3_video_vae_long_decode_sm89_v1",
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "geometry": [args.width, args.height, args.frames],
        "latent_shape": list(normalized.shape),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_weights_changed": False,
        "tile_size": args.tile_size,
        "tile_batch_size": 1,
        "temporal_host_chunk_frames": args.temporal_host_chunk_frames,
        "enforced_vram_gib": args.enforce_vram_gib,
        "transformer_block_compile": True,
        "uint8_postprocess_frame_chunk": select_uint8_postprocess_frame_chunk(shape),
        "load_and_compile_seconds": load_seconds,
        "decode_seconds": decode_seconds,
        "baseline_allocated_gib": baseline / 1024**3,
        "peak_allocated_gib": peak / 1024**3,
        "incremental_peak_gib": (peak - baseline) / 1024**3,
        "output_shape": list(output.shape),
        "output_bytes": output.numel() * output.element_size(),
        "output_sha256": output_sha256,
        "physical_24gib_fit": peak < 24 * 1024**3,
        "configured_budget_fit": (
            True
            if args.enforce_vram_gib is None
            else peak < args.enforce_vram_gib * 1024**3
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["physical_24gib_fit"]:
        raise SystemExit("long Video-VAE decode exceeded 24 GiB allocated")
    if not report["configured_budget_fit"]:
        raise SystemExit("long Video-VAE decode exceeded configured VRAM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
