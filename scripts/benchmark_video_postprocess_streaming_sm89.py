#!/usr/bin/env python3
"""Measure exact long-video uint8 postprocess streaming on RTX 4090.

This isolates the checkpoint-independent pixel transform after Video-VAE
decode.  It compares the former whole-video FP32 implementation with the new
geometry-routed temporal stream using one identical synthetic decoded tensor.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import torch

from h3serve.native_engine.adapters.real_vae import (
    postprocess_native_video,
    select_uint8_postprocess_frame_chunk,
)


def _legacy_uint8(decoded: torch.Tensor) -> torch.Tensor:
    pixel_mean = decoded.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
    pixel_std = decoded.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
    pixels = (decoded.float() * pixel_std + pixel_mean).clamp_(0, 1)
    return torch.round(pixels.mul_(255.0)).to(torch.uint8).cpu()


def _sha256_tensor(value: torch.Tensor) -> str:
    if value.device.type != "cpu" or not value.is_contiguous():
        value = value.contiguous().cpu()
    return hashlib.sha256(memoryview(value.numpy()).cast("B")).hexdigest()


def _run(name: str, operation, decoded: torch.Tensor) -> tuple[dict, str | None]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    started = time.perf_counter()
    try:
        output = operation(decoded)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        digest = _sha256_tensor(output)
        row = {
            "name": name,
            "status": "complete",
            "seconds": seconds,
            "baseline_allocated_gib": baseline / 1024**3,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "incremental_peak_gib": (
                torch.cuda.max_memory_allocated() - baseline
            ) / 1024**3,
            "output_bytes": output.numel() * output.element_size(),
            "output_sha256": digest,
        }
        del output
        return row, digest
    except torch.cuda.OutOfMemoryError as error:
        return ({
            "name": name,
            "status": "oom",
            "seconds": time.perf_counter() - started,
            "baseline_allocated_gib": baseline / 1024**3,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "incremental_peak_gib": (
                torch.cuda.max_memory_allocated() - baseline
            ) / 1024**3,
            "error": str(error),
        }, None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1088)
    parser.add_argument("--frames", type=int, default=362)
    parser.add_argument("--seed", type=int, default=133)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this benchmark requires one RTX 4090 / SM89 GPU")
    if min(args.width, args.height, args.frames) <= 0:
        raise SystemExit("video geometry must be positive")

    shape = (1, 3, args.frames, args.height, args.width)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    decoded = torch.empty(shape, device="cuda:0", dtype=torch.float16)
    decoded.uniform_(-4.0, 4.0, generator=generator)
    runs = []
    legacy, legacy_digest = _run("whole_video_fp32", _legacy_uint8, decoded)
    runs.append(legacy)
    streamed, streamed_digest = _run(
        "temporal_streaming_exact",
        lambda value: postprocess_native_video(value, output_dtype="uint8"),
        decoded,
    )
    runs.append(streamed)
    byte_exact = (
        legacy_digest is not None
        and streamed_digest is not None
        and legacy_digest == streamed_digest
    )
    report = {
        "schema_version": "h3_video_uint8_postprocess_streaming_benchmark_v1",
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "geometry_ncthw": list(shape),
        "whole_fp32_working_tensor_gib": (
            torch.tensor(shape).prod().item() * 4 / 1024**3
        ),
        "selected_frame_chunk": select_uint8_postprocess_frame_chunk(shape),
        "model_weights_involved": False,
        "runs": runs,
        "byte_exact": byte_exact,
    }
    if legacy["status"] == "complete" and streamed["status"] == "complete":
        report["peak_reduction_gib"] = (
            legacy["peak_allocated_gib"] - streamed["peak_allocated_gib"]
        )
        report["speedup"] = legacy["seconds"] / streamed["seconds"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    del decoded
    if legacy["status"] == "complete" and not byte_exact:
        raise SystemExit("temporal uint8 streaming changed output bytes")
    if streamed["status"] != "complete":
        raise SystemExit("temporal uint8 streaming did not complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
