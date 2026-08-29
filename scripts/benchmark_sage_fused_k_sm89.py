#!/usr/bin/env python3
"""Compare accepted SageAttention with the fused smooth-K quantizer on SM89.

This is a real-shape kernel filter.  It never changes the production backend;
full Ref2VA generation remains mandatory before accepting a candidate.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from h3serve.native_engine.model import (
    sage_attention_sm89,
    sage_attention_sm89_fused_k_quant,
)
from h3serve.native_engine.sm89_policy import configure_sm89_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence", type=int, action="append", default=[],
        help="packed-token length; repeat for multiple Ref2VA shapes",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--seed", type=int, default=82416)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeat < 3:
        parser.error("use at least one warmup and three measured repeats")
    if any(value <= 0 for value in args.sequence):
        parser.error("sequence lengths must be positive")
    if not args.sequence:
        args.sequence = [27_392, 59_897]
    return args


def measure(function, *, warmup: int, repeat: int):
    for _ in range(warmup):
        output = function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    return output, samples


def main() -> int:
    args = parse_args()
    runtime = configure_sm89_runtime(quant_backend="cuda", smoke_test=False)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    rows = []
    for sequence in args.sequence:
        shape = (sequence, 56, 128)
        query = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        key = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        value = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        reference, reference_samples = measure(
            lambda: sage_attention_sm89(query, key, value),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        candidate, candidate_samples = measure(
            lambda: sage_attention_sm89_fused_k_quant(query, key, value),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        difference = candidate.float() - reference.float()
        reference_median = statistics.median(reference_samples)
        candidate_median = statistics.median(candidate_samples)
        rows.append({
            "sequence": sequence,
            "shape": list(shape),
            "reference_samples_ms": reference_samples,
            "candidate_samples_ms": candidate_samples,
            "reference_median_ms": reference_median,
            "candidate_median_ms": candidate_median,
            "speedup": reference_median / candidate_median,
            "elementwise_equal": bool(torch.equal(reference, candidate)),
            "max_abs": float(difference.abs().max()),
            "mean_abs": float(difference.abs().mean()),
        })
        del query, key, value, reference, candidate, difference
        torch.cuda.empty_cache()
    document = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "runtime": runtime.to_dict(),
        "candidate": "fused smooth-K subtraction and per-thread INT8 quantization",
        "warmup": args.warmup,
        "repeat": args.repeat,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
