#!/usr/bin/env python3
"""Screen a dialogue-preserving Sparge block mask on real H3 tensor shapes."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparge-build-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=67_368)
    parser.add_argument("--protected-tokens", type=int, default=900)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--topk", type=float, default=0.50)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--seed", type=int, default=4090)
    return parser.parse_args()


def measure(operation, warmup: int, repeat: int):
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
    return output, samples


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.sparge_build_dir.resolve()))
    if torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this benchmark requires one SM89 GPU")
    if not 0 < args.protected_tokens < args.tokens:
        raise SystemExit("protected tokens must be inside the packed sequence")
    if not 0.5 <= args.topk <= 1.0:
        raise SystemExit("topk must be between 0.5 and 1.0")

    from sageattention import sageattn_qk_int8_pv_fp8_cuda
    from einops import rearrange
    from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda
    from spas_sage_attn import core as sparge_core
    from spas_sage_attn.utils import (
        block_map_lut_triton,
        get_block_map_meansim_fuse_quant,
        hyperparameter_check,
    )

    torch.manual_seed(args.seed)
    shape = (1, args.tokens, args.heads, args.head_dim)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    def dense():
        return sageattn_qk_int8_pv_fp8_cuda(
            query,
            key,
            value,
            tensor_layout="NHD",
            is_causal=False,
            qk_quant_gran="per_thread",
            pv_accum_dtype="fp32+fp16",
        )

    def sparse():
        return spas_sage2_attn_meansim_topk_cuda(
            query,
            key,
            value,
            topk=args.topk,
            tensor_layout="NHD",
            is_causal=False,
            return_sparsity=False,
        )

    def protected_sparse():
        # This mirrors Sparge's fused top-k wrapper through block-map creation,
        # then widens the map before LUT conversion.  Keeping its fused Q/K
        # quantization is important: a public-map pass followed by the generic
        # block-sparse wrapper quantizes twice and is slower than dense.
        q, k, v = map(
            lambda tensor: rearrange(tensor, "... L H D -> ... H L D"),
            (query, key, value),
        )
        q, k = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16)
        v = v.contiguous().to(torch.float16)
        key_mean = k.mean(dim=-2, keepdim=True)
        (
            block_map,
            q_int8,
            q_scale,
            k_int8,
            k_scale,
        ) = get_block_map_meansim_fuse_quant(
            q,
            k,
            key_mean,
            BLKQ=128,
            BLKK=64,
            simthreshd1=-0.1,
            cdfthreshd=None,
            topk=args.topk,
            is_causal=False,
        )
        protected_q_blocks = (args.protected_tokens + 127) // 128
        protected_k_blocks = (args.protected_tokens + 63) // 64
        # Text/audio queries attend every key; every query retains all
        # text/audio keys. Only the dominant video-to-video region is sparse.
        block_map[:, :, :protected_q_blocks, :] = True
        block_map[:, :, :, :protected_k_blocks] = True
        lut, valid_block_num = block_map_lut_triton(block_map.contiguous())
        scale = 1.0 / (args.head_dim**0.5)
        pv_threshold = hyperparameter_check(50, args.heads, q.device)
        output = torch.empty_like(q)
        sparge_core.qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(
            q_int8,
            k_int8,
            v,
            output,
            lut,
            valid_block_num,
            pv_threshold,
            q_scale,
            k_scale,
            1,
            0,
            1,
            scale,
            0,
        )
        sparsity = 1 - valid_block_num.float().sum() / valid_block_num.numel() / lut.shape[-1]
        return rearrange(output, "... H L D -> ... L H D"), float(sparsity)

    dense_output, dense_samples = measure(dense, args.warmup, args.repeat)
    sparse_output, sparse_samples = measure(sparse, args.warmup, args.repeat)
    protected_result, protected_samples = measure(
        protected_sparse, args.warmup, args.repeat
    )
    protected_output, protected_sparsity = protected_result
    dense_ms = statistics.median(dense_samples)
    protected_slice = slice(0, args.protected_tokens)

    def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
        return float(
            torch.nn.functional.cosine_similarity(
                left.float().flatten(), right.float().flatten(), dim=0
            )
        )

    result = {
        "device": torch.cuda.get_device_name(),
        "shape": list(shape),
        "protected_tokens": args.protected_tokens,
        "topk": args.topk,
        "dense_ms": dense_ms,
        "sparse_ms": statistics.median(sparse_samples),
        "protected_sparse_ms": statistics.median(protected_samples),
        "sparse_speedup_vs_dense": dense_ms / statistics.median(sparse_samples),
        "protected_speedup_vs_dense": dense_ms
        / statistics.median(protected_samples),
        "protected_reported_sparsity": protected_sparsity,
        "sparse_cosine_vs_dense": cosine(dense_output, sparse_output),
        "protected_cosine_vs_dense": cosine(dense_output, protected_output),
        "sparse_protected_query_cosine": cosine(
            dense_output[:, protected_slice], sparse_output[:, protected_slice]
        ),
        "protected_query_cosine": cosine(
            dense_output[:, protected_slice], protected_output[:, protected_slice]
        ),
        "sparse_video_query_cosine": cosine(
            dense_output[:, args.protected_tokens :],
            sparse_output[:, args.protected_tokens :],
        ),
        "protected_video_query_cosine": cosine(
            dense_output[:, args.protected_tokens :],
            protected_output[:, args.protected_tokens :],
        ),
        "dense_samples_ms": dense_samples,
        "sparse_samples_ms": sparse_samples,
        "protected_samples_ms": protected_samples,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
