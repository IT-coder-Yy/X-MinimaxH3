#!/usr/bin/env python3
"""Screen SageAttention2 SM89 PV accumulators at an H3 packed shape.

This is a kernel filter.  The accepted Native backend remains
``fp32+fp16`` until an alternative passes full original-weight and LoRA video
gates.  Random-tensor error metrics are diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=100_000)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def measure(operation, warmup: int, repeat: int):
    output = None
    for _ in range(warmup):
        output = operation()
    torch.cuda.synchronize()
    samples = []
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


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this benchmark requires one SM89 GPU")
    if args.warmup < 1 or args.repeat < 3:
        raise SystemExit("use at least one warmup and three measured repeats")

    from sageattention import sageattn_qk_int8_pv_fp8_cuda

    torch.manual_seed(args.seed)
    shape = (1, args.sequence, args.heads, args.head_dim)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16).clamp_(-4, 4)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16).clamp_(-4, 4)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16).clamp_(-4, 4)

    results = []
    reference = None
    reference_ms = None
    for accumulator in ("fp32+fp16", "fp32+fp32", "fp32"):
        operation = lambda accumulator=accumulator: sageattn_qk_int8_pv_fp8_cuda(
            query,
            key,
            value,
            tensor_layout="NHD",
            is_causal=False,
            qk_quant_gran="per_thread",
            pv_accum_dtype=accumulator,
            smooth_k=True,
        )
        output, samples = measure(operation, args.warmup, args.repeat)
        median_ms = statistics.median(samples)
        if reference is None:
            reference = output.detach().clone()
            reference_ms = median_ms
        delta = output.float() - reference.float()
        reference_float = reference.float()
        results.append(
            {
                "pv_accum_dtype": accumulator,
                "samples_ms": samples,
                "median_ms": median_ms,
                "speedup_vs_fp32_plus_fp16": reference_ms / median_ms,
                "max_abs_vs_fp32_plus_fp16": float(delta.abs().max()),
                "nrmse_vs_fp32_plus_fp16": float(
                    torch.sqrt(torch.mean(delta.square()))
                    / torch.sqrt(torch.mean(reference_float.square())).clamp_min(1e-12)
                ),
                "cosine_vs_fp32_plus_fp16": float(
                    torch.nn.functional.cosine_similarity(
                        output.float().flatten(), reference_float.flatten(), dim=0
                    )
                ),
            }
        )
        del output, delta, reference_float
        torch.cuda.empty_cache()

    document = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "shape": list(shape),
        "dtype": "bfloat16",
        "qk_quant_gran": "per_thread",
        "smooth_k": True,
        "reference": "fp32+fp16 (accepted SageAttention2++ path)",
        "warmup": args.warmup,
        "repeat": args.repeat,
        "results": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps(document, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
