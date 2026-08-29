#!/usr/bin/env python3
"""Screen SageAttention SM89 variants at an H3 720p packed shape.

This is a kernel candidate filter, not a generation-quality benchmark.  The
accepted production backend remains unchanged until a candidate passes a full
same-request video run and the visual-first gate.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class Variant:
    name: str
    qk_quant_gran: str
    smooth_k: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=34_519)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def elapsed_ms(function) -> tuple[torch.Tensor, list[float]]:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    output = None
    for _ in range(2):
        output = function()
    torch.cuda.synchronize()
    for _ in range(7):
        start.record()
        output = function()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    assert output is not None
    return output, samples


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this benchmark requires one SM89 GPU")
    if args.warmup != 2 or args.repeat != 7:
        raise SystemExit("the audited benchmark currently requires --warmup 2 --repeat 7")

    from sageattention import sageattn_qk_int8_pv_fp8_cuda

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    shape = (1, args.sequence, args.heads, args.head_dim)
    # RMS-normalized H3 Q/K and projected V are well represented by bounded
    # unit-scale inputs for speed screening.  Quality is never accepted here.
    query = torch.randn(shape, device=device, dtype=torch.bfloat16).clamp_(-4, 4)
    key = torch.randn(shape, device=device, dtype=torch.bfloat16).clamp_(-4, 4)
    value = torch.randn(shape, device=device, dtype=torch.bfloat16).clamp_(-4, 4)

    variants = (
        Variant("thread_smooth", "per_thread", True),
        Variant("warp_smooth", "per_warp", True),
        Variant("thread_unsmoothed", "per_thread", False),
        Variant("warp_unsmoothed", "per_warp", False),
    )
    results: list[dict[str, object]] = []
    reference = None
    reference_ms = None
    for variant in variants:
        function = lambda variant=variant: sageattn_qk_int8_pv_fp8_cuda(
            query,
            key,
            value,
            tensor_layout="NHD",
            is_causal=False,
            qk_quant_gran=variant.qk_quant_gran,
            pv_accum_dtype="fp32+fp16",
            smooth_k=variant.smooth_k,
        )
        output, samples = elapsed_ms(function)
        median_ms = statistics.median(samples)
        if reference is None:
            reference = output.detach().clone()
            reference_ms = median_ms
        difference = (output.float() - reference.float()).flatten()
        reference_flat = reference.float().flatten()
        nrmse = float(
            torch.sqrt(torch.mean(difference.square()))
            / torch.sqrt(torch.mean(reference_flat.square())).clamp_min(1e-12)
        )
        cosine = float(
            torch.nn.functional.cosine_similarity(
                output.float().flatten(), reference_flat, dim=0
            )
        )
        results.append(
            {
                "name": variant.name,
                "qk_quant_gran": variant.qk_quant_gran,
                "smooth_k": variant.smooth_k,
                "samples_ms": samples,
                "median_ms": median_ms,
                "speedup_vs_thread_smooth": reference_ms / median_ms,
                "max_abs_vs_thread_smooth": float(difference.abs().max()),
                "nrmse_vs_thread_smooth": nrmse,
                "cosine_vs_thread_smooth": cosine,
            }
        )
        del output, difference, reference_flat
        torch.cuda.empty_cache()

    document = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "shape": list(shape),
        "dtype": "bfloat16",
        "pv_accum_dtype": "fp32+fp16",
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
