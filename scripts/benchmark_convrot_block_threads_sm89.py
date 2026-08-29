#!/usr/bin/env python3
"""Sweep exact ConvRot row-INT8 block sizes on H3's real SM89 widths."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
import sys
import types
from pathlib import Path

import torch

from h3serve.native_engine.sm89_policy import configure_sm89_runtime


SHAPES = (
    ("hidden_5376", 8192, 5376, 0),
    ("attention_7168", 8192, 7168, 0),
    ("swiglu_14336", 4096, 14336, 2),
)
BLOCK_THREADS = (512, 640, 704, 768, 896, 1024)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=21)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            root
            / "runtime/calibration/tiered_backend_20260828/kernels"
            / "convrot_block_threads_sm89_r1.json"
        ),
    )
    args = parser.parse_args()
    if args.warmup < 1 or args.repeat < 5:
        parser.error("warmup must be positive and repeat must be at least five")
    return args


def load_candidate(path: Path):
    package_name = "h3_convrot_candidate"
    package = types.ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(f"{package_name}._C", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate extension: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def capsule(tensor: torch.Tensor):
    return tensor.__dlpack__(stream=-1)


def invoke(
    extension,
    value: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    act_code: int,
    block_threads: int,
) -> None:
    extension.quantize_int8_rowwise_convrot64_config(
        capsule(value),
        capsule(qdata),
        capsule(scale),
        256,
        False,
        act_code,
        block_threads,
        0,
        torch.cuda.current_stream(value.device).cuda_stream,
    )


def elapsed_ms(operation) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    operation()
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop))


def main() -> int:
    args = parse_args()
    runtime = configure_sm89_runtime(quant_backend="cuda", smoke_test=True)
    import comfy_kitchen.backends.cuda as release_cuda

    candidate = load_candidate(args.candidate.resolve())
    if not hasattr(candidate, "quantize_int8_rowwise_convrot64_config"):
        raise RuntimeError("candidate extension lacks the block-size sweep entry point")

    generator = torch.Generator(device="cuda:0").manual_seed(4090)
    report: dict[str, object] = {
        "schema_version": "h3_convrot_block_threads_sm89_v1",
        "warning": "Executor-only kernel sweep; no end-to-end latency claim.",
        "runtime": {
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "release_sha256": runtime.comfy_kitchen_cuda_sha256,
            "candidate": str(args.candidate.resolve()),
        },
        "warmup": args.warmup,
        "repeat": args.repeat,
        "runs": [],
    }

    for name, rows, width, act_code in SHAPES:
        raw_width = width * (2 if act_code == 2 else 1)
        value = torch.randn(
            (rows, raw_width),
            device="cuda:0",
            dtype=torch.bfloat16,
            generator=generator,
        )
        reference_q = torch.empty((rows, width), device="cuda:0", dtype=torch.int8)
        reference_scale = torch.empty((rows, 1), device="cuda:0", dtype=torch.float32)
        release_cuda._C.quantize_int8_rowwise_convrot64(
            capsule(value),
            capsule(reference_q),
            capsule(reference_scale),
            256,
            False,
            act_code,
            0,
            torch.cuda.current_stream().cuda_stream,
        )
        torch.cuda.synchronize()

        qdata = torch.empty_like(reference_q)
        scale = torch.empty_like(reference_scale)
        for block_threads in BLOCK_THREADS:
            for _ in range(args.warmup):
                invoke(candidate, value, qdata, scale, act_code, block_threads)
        torch.cuda.synchronize()

        samples: dict[int, list[float]] = {value: [] for value in BLOCK_THREADS}
        order = list(BLOCK_THREADS)
        rng = random.Random(4090 + width)
        for _ in range(args.repeat):
            rng.shuffle(order)
            for block_threads in order:
                samples[block_threads].append(
                    elapsed_ms(
                        lambda block_threads=block_threads: invoke(
                            candidate,
                            value,
                            qdata,
                            scale,
                            act_code,
                            block_threads,
                        )
                    )
                )

        candidates = []
        for block_threads in BLOCK_THREADS:
            invoke(candidate, value, qdata, scale, act_code, block_threads)
            torch.cuda.synchronize()
            candidate_entry = {
                "block_threads": block_threads,
                "median_ms": statistics.median(samples[block_threads]),
                "min_ms": min(samples[block_threads]),
                "max_ms": max(samples[block_threads]),
                "q_exact_vs_release": bool(torch.equal(qdata, reference_q)),
                "scale_exact_vs_release": bool(torch.equal(scale, reference_scale)),
                "scale_max_abs_vs_release": float(
                    scale.sub(reference_scale).abs().max()
                ),
            }
            candidates.append(candidate_entry)
        reference = next(
            item for item in candidates if item["block_threads"] == 1024
        )
        for item in candidates:
            item["speedup_vs_1024"] = (
                reference["median_ms"] / item["median_ms"]
            )
        fastest = min(candidates, key=lambda item: item["median_ms"])
        shape_result = {
            "name": name,
            "rows": rows,
            "width": width,
            "raw_width": raw_width,
            "act_code": act_code,
            "fastest_block_threads": fastest["block_threads"],
            "fastest_median_ms": fastest["median_ms"],
            "speedup_vs_1024": fastest["speedup_vs_1024"],
            "candidates": candidates,
        }
        report["runs"].append(shape_result)
        print(json.dumps(shape_result, ensure_ascii=False), flush=True)
        del value, reference_q, reference_scale, qdata, scale
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
