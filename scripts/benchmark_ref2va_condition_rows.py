#!/usr/bin/env python3
"""Measure the exact Ref2VA immutable-condition-row cache in isolation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from h3serve.native_engine.model.config import H3CoreConfig
from h3serve.native_engine.model.dit import FullH3DiT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.steps, args.warmup, args.repeats) <= 0:
        parser.error("steps, warmup and repeats must be positive")
    return args


def timed(operation) -> float:
    torch.cuda.synchronize()
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    operation()
    finished.record()
    finished.synchronize()
    return float(started.elapsed_time(finished))


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    model = FullH3DiT.__new__(FullH3DiT)
    nn.Module.__init__(model)
    model.config = H3CoreConfig()

    # ref-test images are normalized to 1152x768, hence 72x48 VAE latents.
    video_latents = tuple(
        torch.randn((1, 24, 1, 48, 72), device=device, dtype=torch.float32)
        for _ in range(3)
    )
    # 5.12s and 2.24s at H3's 40 latent frames/s.
    audio_latents = (
        torch.randn((1, 32, 2, 205), device=device, dtype=torch.float32),
        torch.randn((1, 32, 2, 90), device=device, dtype=torch.float32),
    )
    video_rows = 3 * 24 * 36
    audio_rows = 2 * (205 + 90)

    def rebuild_once():
        model._condition_rows(
            video_latents, device=device, augmentation=0.999,
            seed=82416, expected_rows=video_rows,
        )
        model._condition_audio_rows(
            audio_latents, device=device, augmentation=1.0,
            seed=82417, expected_rows=audio_rows,
        )

    for _ in range(args.warmup):
        rebuild_once()
    dense = [timed(lambda: [rebuild_once() for _ in range(args.steps)]) for _ in range(args.repeats)]
    cached = [timed(rebuild_once) for _ in range(args.repeats)]
    dense.sort()
    cached.sort()
    dense_median = dense[len(dense) // 2]
    cached_median = cached[len(cached) // 2]
    report = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(device),
        "video_latent_shapes": [list(value.shape) for value in video_latents],
        "audio_latent_shapes": [list(value.shape) for value in audio_latents],
        "steps": args.steps,
        "repeats": args.repeats,
        "rebuild_every_step_median_ms": dense_median,
        "build_once_median_ms": cached_median,
        "saved_per_request_ms": dense_median - cached_median,
        "micro_path_speedup": dense_median / cached_median,
        "note": "This is only the condition-row subpath, not end-to-end speedup.",
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
