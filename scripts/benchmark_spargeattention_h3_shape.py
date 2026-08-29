#!/usr/bin/env python3
"""Compare isolated SpargeAttention candidates with the accepted H3 dense path."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=34_519)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    return parser.parse_args()


def measure(function, warmup: int, repeat: int):
    for _ in range(warmup):
        result = function()
    torch.cuda.synchronize()
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(repeat):
        start.record()
        result = function()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    return result, samples


def main() -> int:
    args = parse_args()
    build_dir = os.environ.get("H3_SPARGE_BUILD_DIR", "").strip()
    if not build_dir:
        raise SystemExit("H3_SPARGE_BUILD_DIR must identify an isolated build")
    sys.path.insert(0, build_dir)
    if torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this benchmark requires SM89")

    from sageattention import sageattn_qk_int8_pv_fp8_cuda
    from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda

    torch.manual_seed(args.seed)
    shape = (1, args.sequence, args.heads, args.head_dim)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16).clamp_(-4, 4)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16).clamp_(-4, 4)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16).clamp_(-4, 4)

    dense, dense_samples = measure(
        lambda: sageattn_qk_int8_pv_fp8_cuda(
            q,
            k,
            v,
            tensor_layout="NHD",
            is_causal=False,
            qk_quant_gran="per_thread",
            pv_accum_dtype="fp32+fp16",
            smooth_k=True,
        ),
        args.warmup,
        args.repeat,
    )
    dense_median = statistics.median(dense_samples)
    results = [
        {
            "name": "dense_sage",
            "median_ms": dense_median,
            "samples_ms": dense_samples,
            "speedup": 1.0,
            "sparsity": 0.0,
            "nrmse": 0.0,
            "cosine": 1.0,
        }
    ]
    dense_f32 = dense.float()
    dense_rms = torch.sqrt(torch.mean(dense_f32.square())).clamp_min(1e-12)
    for topk in (0.9, 0.75, 0.65, 0.6, 0.5):
        def sparse_call(topk=topk):
            return spas_sage2_attn_meansim_topk_cuda(
                q,
                k,
                v,
                topk=topk,
                is_causal=False,
                tensor_layout="NHD",
                return_sparsity=True,
            )

        (output, sparsity), samples = measure(
            sparse_call, args.warmup, args.repeat
        )
        delta = output.float() - dense_f32
        median_ms = statistics.median(samples)
        results.append(
            {
                "name": f"sparge_topk_{topk}",
                "topk": topk,
                "median_ms": median_ms,
                "samples_ms": samples,
                "speedup": dense_median / median_ms,
                "sparsity": float(sparsity),
                "max_abs": float(delta.abs().max()),
                "nrmse": float(torch.sqrt(torch.mean(delta.square())) / dense_rms),
                "cosine": float(
                    torch.nn.functional.cosine_similarity(
                        output.float().flatten(), dense_f32.flatten(), dim=0
                    )
                ),
            }
        )
        del output, delta
        torch.cuda.empty_cache()

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "build_dir": build_dir,
                "shape": list(shape),
                "results": results,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
