#!/usr/bin/env python3
"""Scan bounded Video-VAE tile batches on one RTX 4090 with real weights."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from h3serve.native_engine.adapters.real_vae import (
    decode_native_video,
    load_native_video_vae,
)
from h3serve.native_engine.adapters.vae_tiling import configure_vae_tile_batching
from h3serve.native_engine.adapters.vae_block_streaming import (
    stream_video_vae_decoder_tail,
)


def parse_case(value: str) -> tuple[int, int, int]:
    try:
        # Accept WIDTHxHEIGHTxFRAMES while keeping the parser error concise.
        parts = value.lower().split("x")
        if len(parts) != 3:
            raise ValueError
        width, height, frames = (int(part) for part in parts)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "case must be WIDTHxHEIGHTxFRAMES"
        ) from error
    if width % 32 or height % 32 or frames < 5 or (frames - 5) % 17:
        raise argparse.ArgumentTypeError(
            "canvas must be divisible by 32 and frames must satisfy 17*n+5"
        )
    return width, height, frames


def parse_args() -> argparse.Namespace:
    serve_root = Path(__file__).resolve().parents[1]
    main_root = serve_root.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        required=True,
        help="repeatable WIDTHxHEIGHTxFRAMES geometry",
    )
    parser.add_argument("--tile-size", type=int, default=288)
    parser.add_argument(
        "--allocator-limit-gib",
        type=float,
        help="optional hard Torch CUDA allocator ceiling for capacity gates",
    )
    parser.add_argument(
        "--temporal-host-chunk-frames",
        type=int,
        help="stream exact uint8 output to host in bounded temporal chunks",
    )
    parser.add_argument("--stream-decoder-tail-blocks", type=int, default=0)
    parser.add_argument("--batch-sizes", default="1,2,3,4")
    parser.add_argument("--seed", type=int, default=82311)
    parser.add_argument("--model-root", type=Path, default=serve_root / "models")
    parser.add_argument("--minimax-source", type=Path, default=main_root / "MiniMax-H3")
    parser.add_argument(
        "--output",
        type=Path,
        default=serve_root / "runtime/calibration/vae_tile_batch_scan.json",
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        help="optionally persist the first deterministic strided output sample",
    )
    args = parser.parse_args()
    try:
        args.batch_sizes = tuple(
            int(item.strip()) for item in args.batch_sizes.split(",")
        )
    except ValueError:
        parser.error("batch sizes must be comma-separated integers")
    if not args.batch_sizes:
        parser.error("batch size scan cannot be empty")
    if any(value <= 0 for value in args.batch_sizes):
        parser.error("batch sizes must be positive")
    return args


def release(*values) -> None:
    del values
    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this benchmark requires an RTX 4090 / SM89 GPU")
    if args.allocator_limit_gib is not None:
        if args.allocator_limit_gib <= 0:
            raise SystemExit("allocator limit must be positive")
        total = torch.cuda.get_device_properties(0).total_memory
        fraction = args.allocator_limit_gib * 1024**3 / total
        if not 0.0 < fraction <= 1.0:
            raise SystemExit("allocator limit exceeds physical VRAM")
        torch.cuda.memory.set_per_process_memory_fraction(fraction, 0)
    checkpoint = args.model_root / "vae/minimax_h3_video_vae_fp16.safetensors"
    model, mean, std = load_native_video_vae(
        args.minimax_source,
        checkpoint,
        device="cuda:0",
        tile_size=args.tile_size,
        tile_batch_size=1,
    )
    report = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(),
        "checkpoint": str(checkpoint.resolve()),
        "tile_size": args.tile_size,
        "allocator_limit_gib": args.allocator_limit_gib,
        "temporal_host_chunk_frames": args.temporal_host_chunk_frames,
        "stream_decoder_tail_blocks": args.stream_decoder_tail_blocks,
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for width, height, frames in args.case:
        latent_frames = ((frames - 5) // 17) * 5 + 2
        generator = torch.Generator("cpu").manual_seed(args.seed)
        normalized = torch.randn(
            (1, 24, latent_frames, height // 16, width // 16),
            generator=generator,
            dtype=torch.float32,
        ).cuda()
        y_tiles = len(model.split_tiles(height, True)[0])
        x_tiles = len(model.split_tiles(width, True)[0])
        case = {
            "width": width,
            "height": height,
            "frames": frames,
            "latent_shape": list(normalized.shape),
            "spatial_tiles": y_tiles * x_tiles,
            "runs": [],
        }
        reference = None
        for batch_size in args.batch_sizes:
            configure_vae_tile_batching(model, batch_size)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            try:
                with stream_video_vae_decoder_tail(
                    model,
                    enabled=args.stream_decoder_tail_blocks > 0,
                    block_count=max(1, args.stream_decoder_tail_blocks),
                ):
                    decoded = decode_native_video(
                        model,
                        normalized,
                        mean,
                        std,
                        frames,
                        output_dtype=(
                            "uint8"
                            if args.temporal_host_chunk_frames is not None
                            else "float32"
                        ),
                        temporal_host_chunk_frames=args.temporal_host_chunk_frames,
                    )
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                # A deterministic strided sample catches numerical drift while
                # avoiding a multi-gigabyte duplicate of a 720p/15s tensor.
                sample = decoded[
                    :, :, :: max(1, frames // 12), ::16, ::16
                ].clone()
                if (
                    args.sample_output is not None
                    and not args.sample_output.exists()
                ):
                    args.sample_output.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(sample, args.sample_output)
                if reference is None:
                    reference = sample
                    max_abs = 0.0
                    mean_abs = 0.0
                else:
                    delta = (sample - reference).abs()
                    max_abs = float(delta.max())
                    mean_abs = float(delta.mean())
                item = {
                    "batch_size": batch_size,
                    "status": "success",
                    "seconds": seconds,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                    "sample_max_abs_vs_batch1": max_abs,
                    "sample_mean_abs_vs_batch1": mean_abs,
                }
                del decoded, sample
            except torch.cuda.OutOfMemoryError as error:
                item = {
                    "batch_size": batch_size,
                    "status": "oom",
                    "error": str(error),
                    "seconds": time.perf_counter() - started,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                }
            case["runs"].append(item)
            args.output.write_text(json.dumps(report | {"cases": report["cases"] + [case]}, indent=2), encoding="utf-8")
            print(json.dumps(item, ensure_ascii=False), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        report["cases"].append(case)
        del normalized, reference
        release()
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
