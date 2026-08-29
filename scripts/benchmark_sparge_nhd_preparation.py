#!/usr/bin/env python3
"""Compare upstream HND and native-H3 NHD Sparge preparation on SM89."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=99_709)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=9)
    parser.add_argument(
        "--sparge-build-dir",
        type=Path,
        default=Path(
            "runtime/extensions/sparge-sm89-py310-torch213-cu133"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runtime/calibration/tiered_backend_20260828/kernels/"
            "sparge_nhd_preparation_99k_r1.json"
        ),
    )
    return parser.parse_args()


def timed(operation, *, warmup: int, repeat: int) -> list[float]:
    for _ in range(warmup):
        values = operation()
        del values
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        values = operation()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
        del values
    return samples


def hnd_compatible_mean_headwise(nhd: torch.Tensor) -> torch.Tensor:
    """Bounded-memory HND mean candidate for an NHD-resident key tensor."""

    pieces = []
    for head in range(int(nhd.shape[2])):
        materialized = (
            nhd[:, :, head : head + 1, :]
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        pieces.append(materialized.mean(dim=2, keepdim=True))
    return torch.cat(pieces, dim=1)


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this benchmark requires RTX 4090 / SM89")
    build_dir = args.sparge_build_dir.resolve()
    if not build_dir.is_dir():
        raise SystemExit(f"missing Sparge build: {build_dir}")
    sys.path.insert(0, str(build_dir))
    from spas_sage_attn import core as sparge_core
    from spas_sage_attn.utils import (
        get_pool_sim_triton_simmean_fuse_quant,
        hyperparameter_check,
    )
    from h3serve.native_engine.model.sparge_nhd import pool_sim_quant_nhd

    device = torch.device("cuda")
    torch.manual_seed(20260828)
    nhd = torch.randn(
        (1, args.tokens, args.heads, args.head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    hnd = nhd.permute(0, 2, 1, 3).contiguous()
    mean_hnd = hnd.mean(dim=2, keepdim=True)
    mean_nhd = nhd.mean(dim=1, keepdim=True)
    # A permuted view is allocation-free.  Check whether asking PyTorch to
    # reduce the logical HND token dimension also preserves the exact
    # reduction tree used by the materialized HND reference at this shape.
    mean_strided_hnd = nhd.permute(0, 2, 1, 3).mean(dim=2, keepdim=True)
    mean_headwise_hnd = hnd_compatible_mean_headwise(nhd)
    threshold = hyperparameter_check(-0.1, args.heads, device)

    reference = get_pool_sim_triton_simmean_fuse_quant(
        hnd, mean_hnd, 64, threshold
    )
    candidate = pool_sim_quant_nhd(nhd, mean_nhd, 64, threshold)
    torch.cuda.synchronize()
    pool_hnd, similar_hnd, quant_hnd, scale_hnd = reference
    pool_nhd, similar_nhd, quant_nhd, scale_nhd = candidate
    equivalence = {
        "mean": torch.equal(mean_hnd, mean_nhd.permute(0, 2, 1, 3)),
        "mean_strided_hnd_view": torch.equal(mean_hnd, mean_strided_hnd),
        "mean_materialized_headwise": torch.equal(mean_hnd, mean_headwise_hnd),
        "pool": torch.equal(pool_hnd, pool_nhd),
        "similar": torch.equal(similar_hnd, similar_nhd),
        "quant": torch.equal(
            quant_hnd, quant_nhd.permute(0, 2, 1, 3)
        ),
        "scale": torch.equal(scale_hnd, scale_nhd),
    }
    del reference, candidate

    mean_hnd_samples = timed(
        lambda: hnd.mean(dim=2, keepdim=True),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    mean_headwise_samples = timed(
        lambda: hnd_compatible_mean_headwise(nhd),
        warmup=args.warmup,
        repeat=args.repeat,
    )

    hnd_samples = timed(
        lambda: get_pool_sim_triton_simmean_fuse_quant(
            hnd, mean_hnd, 64, threshold
        ),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    nhd_samples = timed(
        lambda: pool_sim_quant_nhd(nhd, mean_nhd, 64, threshold),
        warmup=args.warmup,
        repeat=args.repeat,
    )

    value_nhd = nhd.to(torch.float16)
    value_hnd = hnd.to(torch.float16)
    padded = (args.tokens + 127) // 128 * 128

    def prepare_value_hnd():
        transposed = torch.empty(
            (1, args.heads, args.head_dim, padded),
            device=device,
            dtype=torch.float16,
        )
        sparge_core.fused.transpose_pad_permute_cuda(value_hnd, transposed, 1)
        fp8 = torch.empty_like(transposed, dtype=torch.float8_e4m3fn)
        scale = torch.empty(
            (1, args.heads, args.head_dim),
            device=device,
            dtype=torch.float32,
        )
        sparge_core.fused.scale_fuse_quant_cuda(
            transposed, fp8, scale, args.tokens, 2.25, 1
        )
        return fp8, scale

    def prepare_value_nhd():
        transposed = torch.empty(
            (1, args.head_dim, args.heads, padded),
            device=device,
            dtype=torch.float16,
        )
        sparge_core.fused.transpose_pad_permute_cuda(value_nhd, transposed, 0)
        fp8 = torch.empty_like(transposed, dtype=torch.float8_e4m3fn)
        scale = torch.empty(
            (1, args.heads, args.head_dim),
            device=device,
            dtype=torch.float32,
        )
        sparge_core.fused.scale_fuse_quant_cuda(
            transposed, fp8, scale, args.tokens, 2.25, 0
        )
        return fp8, scale

    v_hnd = prepare_value_hnd()
    v_nhd = prepare_value_nhd()
    torch.cuda.synchronize()
    value_equivalence = {
        "fp8": torch.equal(v_hnd[0], v_nhd[0].permute(0, 2, 1, 3)),
        "scale": torch.equal(v_hnd[1], v_nhd[1]),
    }
    del v_hnd, v_nhd
    value_hnd_samples = timed(
        prepare_value_hnd, warmup=args.warmup, repeat=args.repeat
    )
    value_nhd_samples = timed(
        prepare_value_nhd, warmup=args.warmup, repeat=args.repeat
    )

    report = {
        "schema_version": "h3_sparge_nhd_preparation_probe_v1",
        "shape": {
            "tokens": args.tokens,
            "heads": args.heads,
            "head_dim": args.head_dim,
        },
        "equivalence": equivalence,
        "mean": {
            "hnd_ms_samples": mean_hnd_samples,
            "headwise_nhd_ms_samples": mean_headwise_samples,
            "hnd_ms": statistics.median(mean_hnd_samples),
            "headwise_nhd_ms": statistics.median(mean_headwise_samples),
        },
        "key": {
            "hnd_ms_samples": hnd_samples,
            "nhd_ms_samples": nhd_samples,
            "hnd_ms": statistics.median(hnd_samples),
            "nhd_ms": statistics.median(nhd_samples),
            "speedup": statistics.median(hnd_samples)
            / statistics.median(nhd_samples),
        },
        "value_equivalence": value_equivalence,
        "value": {
            "hnd_ms_samples": value_hnd_samples,
            "nhd_ms_samples": value_nhd_samples,
            "hnd_ms": statistics.median(value_hnd_samples),
            "nhd_ms": statistics.median(value_nhd_samples),
            "speedup": statistics.median(value_hnd_samples)
            / statistics.median(value_nhd_samples),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
