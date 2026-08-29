#!/usr/bin/env python3
"""Profile the real H3 Video-VAE decode tail on one RTX 4090."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from h3serve.native_engine.adapters.real_vae import (
    decode_native_video,
    load_native_video_vae,
)
from h3serve.native_engine.adapters.vae_compile import (
    enable_transformer_block_compile,
    prewarm_feed_forward_compile,
    prewarm_transformer_block_compile,
)


def parse_args() -> argparse.Namespace:
    serve_root = Path(__file__).resolve().parents[1]
    main_root = serve_root.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=736)
    parser.add_argument("--frames", type=int, default=362)
    parser.add_argument("--tile-size", type=int, default=288)
    parser.add_argument(
        "--compile-region",
        choices=("feed-forward", "transformer-block"),
        default="feed-forward",
    )
    parser.add_argument("--model-root", type=Path, default=serve_root / "models")
    parser.add_argument("--minimax-source", type=Path, default=main_root / "MiniMax-H3")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=serve_root / "runtime/profile/video_vae_sm89",
    )
    args = parser.parse_args()
    if args.width % 32 or args.height % 32:
        parser.error("width and height must be divisible by 32")
    if args.frames < 5 or (args.frames - 5) % 17:
        parser.error("frames must satisfy 17*n+5")
    return args


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this profile requires one RTX 4090 / SM89 GPU")
    checkpoint = args.model_root / "vae/minimax_h3_video_vae_fp16.safetensors"
    model, mean, std = load_native_video_vae(
        args.minimax_source,
        checkpoint,
        device="cuda:0",
        tile_size=args.tile_size,
        tile_batch_size=1,
        compile_feed_forward=args.compile_region == "feed-forward",
    )
    if args.compile_region == "transformer-block":
        compiled_modules = enable_transformer_block_compile(model)
        prewarm_transformer_block_compile(model)
    else:
        compiled_modules = 0
        prewarm_feed_forward_compile(model)
    latent_frames = ((args.frames - 5) // 17) * 5 + 2
    generator = torch.Generator("cpu").manual_seed(4090)
    normalized = torch.randn(
        (1, 24, latent_frames, args.height // 16, args.width // 16),
        generator=generator,
        dtype=torch.float32,
    ).cuda()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        with torch.profiler.record_function("h3_video_vae_decode"):
            decoded = decode_native_video(
                model, normalized, mean, std, args.frames
            )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    table = profiler.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=80
    )
    (args.output_dir / "kernels.txt").write_text(table, encoding="utf-8")
    report = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(),
        "geometry": [args.width, args.height, args.frames],
        "latent_shape": list(normalized.shape),
        "tile_size": args.tile_size,
        "compile_region": args.compile_region,
        "compiled_modules": compiled_modules,
        "elapsed_seconds": elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "output_shape": list(decoded.shape),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    print(table, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
