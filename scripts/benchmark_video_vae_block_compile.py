#!/usr/bin/env python3
"""Same-process eager/compiled H3 Video-VAE A/B on one RTX 4090."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from h3serve.native_engine.adapters.real_vae import (
    decode_native_video,
    load_native_video_vae,
)
from h3serve.native_engine.adapters.vae_compile import (
    enable_transformer_block_compile,
    prewarm_transformer_block_compile,
    transformer_block_compile,
)


def parse_case(value: str) -> tuple[int, int, int]:
    try:
        width, height, frames = (int(part) for part in value.lower().split("x"))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("case must be WIDTHxHEIGHTxFRAMES") from error
    if width % 32 or height % 32 or frames < 5 or (frames - 5) % 17:
        raise argparse.ArgumentTypeError("invalid H3 canvas or frame count")
    return width, height, frames


def parse_args() -> argparse.Namespace:
    serve_root = Path(__file__).resolve().parents[1]
    main_root = serve_root.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=parse_case, required=True)
    parser.add_argument("--tile-size", type=int, default=288)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument("--model-root", type=Path, default=serve_root / "models")
    parser.add_argument("--minimax-source", type=Path, default=main_root / "MiniMax-H3")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def timed_decode(model, normalized, mean, std, frames):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = decode_native_video(model, normalized, mean, std, frames)
    torch.cuda.synchronize()
    return (
        output,
        time.perf_counter() - started,
        torch.cuda.max_memory_allocated() / 1024**3,
    )


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this benchmark requires one RTX 4090 / SM89 GPU")
    width, height, frames = args.case
    checkpoint = args.model_root / "vae/minimax_h3_video_vae_fp16.safetensors"
    model, mean, std = load_native_video_vae(
        args.minimax_source,
        checkpoint,
        device="cuda:0",
        tile_size=args.tile_size,
        tile_batch_size=1,
        compile_feed_forward=False,
    )
    latent_frames = ((frames - 5) // 17) * 5 + 2
    generator = torch.Generator("cpu").manual_seed(args.seed)
    normalized = torch.randn(
        (1, 24, latent_frames, height // 16, width // 16),
        generator=generator,
        dtype=torch.float32,
    ).cuda()

    eager_output, _, _ = timed_decode(model, normalized, mean, std, frames)
    eager_times = []
    eager_peaks = []
    for _ in range(args.repeat):
        eager_output, seconds, peak = timed_decode(
            model, normalized, mean, std, frames
        )
        eager_times.append(seconds)
        eager_peaks.append(peak)

    compiled_modules = enable_transformer_block_compile(model)
    prewarm_transformer_block_compile(model)
    compiled_times = []
    compiled_peaks = []
    compiled_output = None
    with transformer_block_compile(True):
        for _ in range(args.repeat):
            compiled_output, seconds, peak = timed_decode(
                model, normalized, mean, std, frames
            )
            compiled_times.append(seconds)
            compiled_peaks.append(peak)
    assert compiled_output is not None
    delta = (compiled_output.float() - eager_output.float()).abs()
    eager_median = statistics.median(eager_times)
    compiled_median = statistics.median(compiled_times)
    report = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(),
        "case": [width, height, frames],
        "tile_size": args.tile_size,
        "compiled_modules": compiled_modules,
        "eager_times_seconds": eager_times,
        "compiled_times_seconds": compiled_times,
        "eager_median_seconds": eager_median,
        "compiled_median_seconds": compiled_median,
        "speedup": eager_median / compiled_median,
        "eager_peak_allocated_gib": max(eager_peaks),
        "compiled_peak_allocated_gib": max(compiled_peaks),
        "output_max_abs": float(delta.max()),
        "output_mean_abs": float(delta.mean()),
        "output_equal": bool(torch.equal(compiled_output, eager_output)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
