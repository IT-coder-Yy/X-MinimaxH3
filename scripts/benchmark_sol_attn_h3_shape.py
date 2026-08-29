#!/usr/bin/env python3
"""Benchmark the experimental H3 Sol-Attn kernel at a real packed shape.

This is a screening benchmark, not an end-to-end quality claim.  It loads the
standalone Triton kernels from the research snapshot without importing its
ComfyUI wrapper, compares them with the production SM89 dense kernel, and
records latency, peak memory and numerical error on a deterministic tensor.
"""

from __future__ import annotations

import argparse
import importlib
import json
import site
import statistics
import sys
import time
import types
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=100_163)
    parser.add_argument("--protected-tokens", type=int, default=1_723)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--tau", type=float, action="append", default=[])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument(
        "--sol-source",
        type=Path,
        default=Path("../../DIT-knowledge/sources/projects/solattn_h3"),
    )
    parser.add_argument(
        "--sparge-build-dir",
        type=Path,
        default=Path("runtime/extensions/sparge-sm89-py310-torch28-cu12"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_sol_kernel(source: Path):
    """Load only the standalone kernels; the upstream package entry imports ComfyUI."""
    source = source.resolve()
    if not (source / "_int8_fwd.py").is_file():
        raise FileNotFoundError(f"Sol-Attn source is incomplete: {source}")
    package_name = "_h3_sol_attn_screening"
    package = types.ModuleType(package_name)
    package.__path__ = [str(source)]
    sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}._int8_fwd").sol_attn_int8


def timed(operation, *, warmup: int, repeat: int) -> tuple[torch.Tensor, list[float]]:
    output = None
    for _ in range(warmup):
        output = operation()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        output = operation()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    assert output is not None
    return output, samples


def error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    left = reference.float().reshape(-1)
    right = candidate.float().reshape(-1)
    delta = left - right
    return {
        "cosine": float(torch.nn.functional.cosine_similarity(left, right, dim=0)),
        "rmse": float(delta.square().mean().sqrt()),
        "mean_abs": float(delta.abs().mean()),
        "max_abs": float(delta.abs().max()),
    }


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this benchmark requires one RTX 4090 / SM89 GPU")
    if not 0 < args.protected_tokens < args.tokens:
        raise SystemExit("protected token count must lie inside the packed sequence")
    if args.warmup < 0 or args.repeat < 1:
        raise SystemExit("warmup must be non-negative and repeat must be positive")

    taus = tuple(args.tau or (0.6, 0.8, 1.0, 1.2))
    site.addsitedir(str(args.sparge_build_dir.resolve()))
    sys.path.insert(0, str(args.sparge_build_dir.resolve()))
    from sageattention import sageattn_qk_int8_pv_fp8_cuda

    sol_attn = load_sol_kernel(args.sol_source)
    torch.manual_seed(args.seed)
    shape = (1, args.tokens, args.heads, args.head_dim)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    protected_blocks = (args.protected_tokens + 63) // 64

    def dense():
        return sageattn_qk_int8_pv_fp8_cuda(
            query,
            key,
            value,
            tensor_layout="NHD",
            is_causal=False,
            qk_quant_gran="per_warp",
            pv_accum_dtype="fp32+fp16",
        )

    torch.cuda.reset_peak_memory_stats()
    compile_started = time.perf_counter()
    dense_reference, dense_samples = timed(dense, warmup=args.warmup, repeat=args.repeat)
    dense_compile_wall = time.perf_counter() - compile_started
    dense_peak = torch.cuda.max_memory_allocated()
    dense_ms = statistics.median(dense_samples)

    rows = []
    for tau in taus:
        torch.cuda.reset_peak_memory_stats()

        def sparse(current_tau=tau):
            return sol_attn(
                query,
                key,
                value,
                tau=current_tau,
                sink_blocks=(0, protected_blocks),
                int8_pv=True,
            )

        compile_started = time.perf_counter()
        candidate, samples = timed(sparse, warmup=args.warmup, repeat=args.repeat)
        compile_wall = time.perf_counter() - compile_started
        median_ms = statistics.median(samples)
        rows.append(
            {
                "tau": tau,
                "median_ms": median_ms,
                "samples_ms": samples,
                "speedup_vs_dense": dense_ms / median_ms,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "screening_error_vs_dense": error(dense_reference, candidate),
                "compile_plus_measure_wall_seconds": compile_wall,
            }
        )
        del candidate
        torch.cuda.empty_cache()

    document = {
        "scope": "synthetic-kernel-screening-not-video-quality",
        "device": torch.cuda.get_device_name(),
        "shape": list(shape),
        "protected_tokens": args.protected_tokens,
        "protected_blocks": protected_blocks,
        "dense": {
            "median_ms": dense_ms,
            "samples_ms": dense_samples,
            "peak_allocated_gib": dense_peak / 2**30,
            "compile_plus_measure_wall_seconds": dense_compile_wall,
        },
        "sol_attn": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps(document, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
