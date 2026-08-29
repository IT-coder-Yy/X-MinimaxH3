#!/usr/bin/env python3
"""Bitwise and timing screen for one-pass H3 segmented AdaLN modulation."""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from backends.original.kernels.fused_norm_rope import (
    native_rms_one_pass_adaln,
    native_rms_two_pass_adaln,
    segmented_rms_adaln,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=34_519)
    parser.add_argument("--hidden", type=int, default=5_376)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--seed", type=int, default=4090)
    return parser.parse_args()


def measure(function, warmup: int, repeat: int):
    for _ in range(warmup):
        output = function()
    torch.cuda.synchronize()
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(repeat):
        start.record()
        output = function()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    return output, samples


def main() -> int:
    args = parse_args()
    if torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this benchmark requires SM89")
    if args.rows < 3 or args.hidden <= 0 or args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("invalid benchmark dimensions or repeat count")

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16
    value = torch.randn(
        (args.rows, args.hidden), device="cuda", dtype=dtype
    ).clamp_(-4, 4)
    norm = torch.nn.RMSNorm(args.hidden, eps=1e-6, device="cuda", dtype=dtype)
    norm.weight.data.uniform_(0.9, 1.1)
    shift = torch.randn((3, args.hidden), device="cuda", dtype=dtype).mul_(0.1)
    scale = torch.randn((3, args.hidden), device="cuda", dtype=dtype).mul_(0.1)
    stop0 = max(1, min(65, args.rows - 2))
    stop1 = max(stop0 + 1, min(stop0 + 414, args.rows - 1))
    segments = ((0, stop0, 0), (stop0, stop1, 1), (stop1, args.rows, 2))

    two, two_samples = measure(
        lambda: native_rms_two_pass_adaln(value, norm, shift, scale, segments),
        args.warmup,
        args.repeat,
    )
    one, one_samples = measure(
        lambda: native_rms_one_pass_adaln(value, norm, shift, scale, segments),
        args.warmup,
        args.repeat,
    )
    fused, fused_samples = measure(
        lambda: segmented_rms_adaln(value, norm, shift, scale, segments),
        args.warmup,
        args.repeat,
    )
    two_median = statistics.median(two_samples)
    one_median = statistics.median(one_samples)
    mismatch_count = int(torch.count_nonzero(two != one))
    max_abs = float((two.float() - one.float()).abs().max())
    fused_difference = (one.float() - fused.float()).abs()
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "shape": [args.rows, args.hidden],
                "dtype": str(dtype),
                "two_pass_ms": two_median,
                "one_pass_ms": one_median,
                "speedup": two_median / one_median,
                "bitwise_equal": mismatch_count == 0,
                "mismatch_count": mismatch_count,
                "max_abs": max_abs,
                "two_pass_samples_ms": two_samples,
                "one_pass_samples_ms": one_samples,
                "fused_rms_adaln_ms": statistics.median(fused_samples),
                "fused_speedup_vs_one_pass": one_median
                / statistics.median(fused_samples),
                "fused_max_abs_vs_one_pass": float(fused_difference.max()),
                "fused_mean_abs_vs_one_pass": float(fused_difference.mean()),
                "fused_mismatch_fraction_vs_one_pass": float(
                    (one != fused).float().mean()
                ),
                "fused_samples_ms": fused_samples,
            },
            indent=2,
        )
    )
    return 0 if mismatch_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
