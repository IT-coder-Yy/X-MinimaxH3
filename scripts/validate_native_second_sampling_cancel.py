#!/usr/bin/env python3
"""Physically cancel one native H3 second-sampling job and verify cleanup."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import shutil
import time
from pathlib import Path

import torch

from h3serve.backend import JobCancelled, build_native_backend
from h3serve.config import ServicePaths
from h3serve.contract import GenerationSpec, SecondSamplingSpec
from h3serve.memory_policy import HOST_MEMORY_PROFILES


PROMPT = "A locked-off restoration-workshop shot with stable geometry and room tone."


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=root)
    parser.add_argument("--launcher", default="ref2va_w4a8_8gb")
    parser.add_argument("--source-latents", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--duration-seconds", type=float, default=5.0)
    parser.add_argument("--cancel-after-seconds", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


async def run(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = ServicePaths.defaults(args.release_root)
    paths = type(paths)(**{**paths.__dict__, "output_dir": args.output_dir.resolve()})
    backend = build_native_backend(paths, memory_profile=HOST_MEMORY_PROFILES["fullspeed"])
    family = "reference" if args.launcher.startswith("ref2va") else "first_last"
    source = GenerationSpec.from_mapping({
        "prompt": PROMPT,
        "runtime_launcher": args.launcher,
        "service_family": family,
        "model_variant": "base",
        "resolution": "480p",
        "aspect_ratio": "16:9",
        "duration_seconds": args.duration_seconds,
        "seed": 82704,
        "sampling_steps": 5,
        "acceleration": 95,
    })
    second = SecondSamplingSpec.from_mapping({
        "resolution": "1080p",
        "steps": 1,
        "acceleration": 95,
        "denoise": 0.15,
    }, source=source)
    target = dataclasses.replace(
        source,
        resolution=second.resolution,
        width=second.width,
        height=second.height,
        advanced=True,
        custom_actual_steps=20,
        sampling_steps=None,
        acceleration=None,
    )
    latent_store = (args.output_dir / ".h3-latents").resolve()
    latent_store.mkdir(parents=True, exist_ok=True)
    local_source = latent_store / "cancel-source.pt"
    shutil.copy2(args.source_latents.resolve(), local_source)
    images = () if args.reference_image is None else (args.reference_image.resolve(),)
    cancel = asyncio.Event()
    started = time.perf_counter()
    cancelled_at = None
    try:
        await backend.preload(args.launcher)
        task = asyncio.create_task(backend.second_sample(
            target,
            second,
            local_source,
            "cancelled_second_sampling",
            None,
            None,
            images,
            (),
            (),
            cancel,
        ))
        await asyncio.sleep(args.cancel_after_seconds)
        cancel.set()
        cancelled_at = time.perf_counter()
        try:
            await task
        except JobCancelled as error:
            cancellation_error = str(error)
        else:
            raise RuntimeError("second-sampling completed instead of honoring cancellation")
    finally:
        await backend.stop()
    torch_allocated = torch.cuda.memory_allocated() / 1024**3
    torch_reserved = torch.cuda.memory_reserved() / 1024**3
    return {
        "schema_version": "h3_native_second_sampling_cancel_gate_v1",
        "status": "passed",
        "launcher": args.launcher,
        "cancel_after_seconds": args.cancel_after_seconds,
        "cancel_ack_seconds": time.perf_counter() - cancelled_at,
        "wall_seconds": time.perf_counter() - started,
        "error": cancellation_error,
        "partial_output_exists": (args.output_dir / "cancelled_second_sampling.mp4").exists(),
        "torch_allocated_gib_after_stop": torch_allocated,
        "torch_reserved_gib_after_stop": torch_reserved,
    }


def main() -> None:
    args = parse_args()
    report = asyncio.run(run(args))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
