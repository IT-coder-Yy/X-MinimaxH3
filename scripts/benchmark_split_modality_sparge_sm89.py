#!/usr/bin/env python3
"""Screen split dense-prefix / sparse-video attention for packed H3 AV."""

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
    parser.add_argument("--tokens", type=int, default=100_000)
    parser.add_argument("--protected-tokens", type=int, default=1_560)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--topk", type=float, default=0.50)
    parser.add_argument(
        "--selection-mode",
        choices=("fixed_topk", "budget_adaptive"),
        default="fixed_topk",
    )
    parser.add_argument("--adaptive-safety-margin", type=float, default=0.65)
    parser.add_argument(
        "--head-topks",
        help="optional comma-separated per-head budgets; overrides --topk",
    )
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
    sys.path.insert(0, str(args.sparge_build_dir.resolve()))
    if torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this benchmark requires one SM89 GPU")
    if not 0 < args.protected_tokens < args.tokens:
        raise SystemExit("protected tokens must be inside the packed sequence")
    if not 0.0625 <= args.topk <= 1.0:
        raise SystemExit("topk must be between 0.0625 and 1.0")
    if not 0.0 <= args.adaptive_safety_margin <= 1.0:
        raise SystemExit("adaptive safety margin must be between 0 and 1")
    if args.head_topks:
        try:
            head_topks = tuple(float(value) for value in args.head_topks.split(","))
        except ValueError as error:
            raise SystemExit("head topks must be comma-separated numbers") from error
        if len(head_topks) != args.heads or any(
            not 0.0625 <= value <= 1.0 for value in head_topks
        ):
            raise SystemExit("head topks must contain one [0.0625, 1.0] value per head")
    else:
        head_topks = None

    from einops import rearrange
    from sageattention import sageattn_qk_int8_pv_fp8_cuda
    from sageattention import sm89_compile
    from sageattention.triton.quant_per_thread import per_thread_int8
    from spas_sage_attn import core as sparge_core
    from spas_sage_attn.utils import (
        block_map_lut_triton,
        get_block_map_meansim_fuse_quant,
        get_quant,
        hyperparameter_check,
    )
    from h3serve.native_engine.model.kernels import (
        SplitModalityProtectedSpargeAttentionBackend,
        attention_protected_prefix,
    )

    torch.manual_seed(args.seed)
    shape = (1, args.tokens, args.heads, args.head_dim)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    def dense(q=query):
        return sageattn_qk_int8_pv_fp8_cuda(
            q,
            key,
            value,
            tensor_layout="NHD",
            is_causal=False,
            qk_quant_gran="per_thread",
            pv_accum_dtype="fp32+fp16",
        )

    def quantize_value():
        v = rearrange(value, "B L H D -> B H L D").contiguous().to(torch.float16)
        batch, heads, kv_len, head_dim = v.shape
        padded_len = (kv_len + 127) // 128 * 128
        transposed = torch.empty(
            (batch, heads, head_dim, padded_len), dtype=v.dtype, device=v.device
        )
        sparge_core.fused.transpose_pad_permute_cuda(v, transposed, 1)
        v_fp8 = torch.empty_like(transposed, dtype=torch.float8_e4m3fn)
        v_scale = torch.empty(
            (batch, heads, head_dim), dtype=torch.float32, device=v.device
        )
        sparge_core.fused.scale_fuse_quant_cuda(
            transposed, v_fp8, v_scale, kv_len, 2.25, 1
        )
        return v_fp8, v_scale

    def dense_prefix_shared_v(v_fp8, v_scale):
        q = rearrange(
            query[:, : args.protected_tokens], "B L H D -> B H L D"
        ).contiguous()
        k = rearrange(key, "B L H D -> B H L D").contiguous()
        key_mean = k.mean(dim=2, keepdim=True)
        q_int8, q_scale, k_int8, k_scale = per_thread_int8(
            q,
            k,
            key_mean,
            tensor_layout="HND",
            BLKQ=128,
            WARPQ=32,
            BLKK=64,
            WARPK=64,
        )
        output = torch.empty_like(q)
        sm89_compile.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
            q_int8,
            k_int8,
            v_fp8,
            output,
            q_scale,
            k_scale,
            v_scale,
            1,
            0,
            3,
            1.0 / (args.head_dim**0.5),
            0,
        )
        return rearrange(output, "B H L D -> B L H D")

    def sparse_video(v_fp8=None, v_scale=None, *, return_k=False):
        video_query = query[:, args.protected_tokens :]
        q, k, v = map(
            lambda tensor: rearrange(tensor, "B L H D -> B H L D"),
            (video_query, key, value),
        )
        q = q.contiguous().to(torch.bfloat16)
        k = k.contiguous().to(torch.bfloat16)
        v = v.contiguous().to(torch.float16)
        key_mean = k.mean(dim=-2, keepdim=True)
        block_map, q_int8, q_scale, k_int8, k_scale = (
            get_block_map_meansim_fuse_quant(
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
        )
        protected_k_blocks = (args.protected_tokens + 63) // 64
        block_map[:, :, :, :protected_k_blocks] = True
        lut, valid_block_num = block_map_lut_triton(block_map.contiguous())

        batch, heads, kv_len, head_dim = v.shape
        if v_fp8 is None or v_scale is None:
            v_fp8, v_scale = quantize_value()
        output = torch.empty_like(q)
        pv_threshold = hyperparameter_check(50, heads, q.device)
        sparge_core.qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
            q_int8,
            k_int8,
            v_fp8,
            output,
            lut,
            valid_block_num,
            pv_threshold,
            q_scale,
            k_scale,
            v_scale,
            1,
            0,
            1,
            1.0 / (head_dim**0.5),
            0,
        )
        output = rearrange(output, "B H L D -> B L H D")
        if return_k:
            return output, k_int8, k_scale
        return output

    def sparse_prefix_shared_kv(k_int8, k_scale, v_fp8, v_scale):
        q = rearrange(
            query[:, : args.protected_tokens], "B L H D -> B H L D"
        ).contiguous().to(torch.bfloat16)
        q_int8, q_scale = get_quant(q, None, 128)
        batch, heads, query_len, head_dim = q.shape
        key_blocks = (args.tokens + 63) // 64
        query_blocks = (query_len + 127) // 128
        full_map = torch.ones(
            (batch, heads, query_blocks, key_blocks),
            dtype=torch.bool,
            device=q.device,
        )
        lut, valid_block_num = block_map_lut_triton(full_map)
        output = torch.empty_like(q)
        pv_threshold = hyperparameter_check(50, heads, q.device)
        sparge_core.qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
            q_int8,
            k_int8,
            v_fp8,
            output,
            lut,
            valid_block_num,
            pv_threshold,
            q_scale,
            k_scale,
            v_scale,
            1,
            0,
            1,
            1.0 / (head_dim**0.5),
            0,
        )
        return rearrange(output, "B H L D -> B L H D")

    def split():
        prefix = dense(query[:, : args.protected_tokens])
        video = sparse_video()
        return torch.cat((prefix, video), dim=1)

    def split_shared_v():
        v_fp8, v_scale = quantize_value()
        prefix = dense_prefix_shared_v(v_fp8, v_scale)
        video = sparse_video(v_fp8, v_scale)
        return torch.cat((prefix, video), dim=1)

    def split_shared_kv():
        v_fp8, v_scale = quantize_value()
        video, k_int8, k_scale = sparse_video(
            v_fp8, v_scale, return_k=True
        )
        prefix = sparse_prefix_shared_kv(
            k_int8, k_scale, v_fp8, v_scale
        )
        return torch.cat((prefix, video), dim=1)

    production_backend = SplitModalityProtectedSpargeAttentionBackend(
        head_topks or args.topk,
        experimental_minimum_topk=0.0625,
        selection_mode=args.selection_mode,
        adaptive_safety_margin=args.adaptive_safety_margin,
    )

    def production_split():
        with attention_protected_prefix(args.protected_tokens):
            output = production_backend(
                query.squeeze(0), key.squeeze(0), value.squeeze(0)
            )
        return output.unsqueeze(0)

    dense_output, dense_samples = measure(dense, args.warmup, args.repeat)
    split_output, split_samples = measure(split, args.warmup, args.repeat)
    shared_output, shared_samples = measure(
        split_shared_v, args.warmup, args.repeat
    )
    shared_kv_output, shared_kv_samples = measure(
        split_shared_kv, args.warmup, args.repeat
    )
    production_output, production_samples = measure(
        production_split, args.warmup, args.repeat
    )
    dense_ms = statistics.median(dense_samples)
    split_ms = statistics.median(split_samples)
    shared_ms = statistics.median(shared_samples)
    shared_kv_ms = statistics.median(shared_kv_samples)
    production_ms = statistics.median(production_samples)
    prefix = slice(0, args.protected_tokens)
    video = slice(args.protected_tokens, None)

    def cosine(left, right):
        return float(
            torch.nn.functional.cosine_similarity(
                left.float().flatten(), right.float().flatten(), dim=0
            )
        )

    document = {
        "device": torch.cuda.get_device_name(),
        "shape": list(shape),
        "protected_tokens": args.protected_tokens,
        "topk": args.topk,
        "selection_mode": args.selection_mode,
        "adaptive_safety_margin": args.adaptive_safety_margin,
        "head_topks": None if head_topks is None else list(head_topks),
        "dense_ms": dense_ms,
        "split_ms": split_ms,
        "split_speedup_vs_dense": dense_ms / split_ms,
        "shared_v_ms": shared_ms,
        "shared_v_speedup_vs_dense": dense_ms / shared_ms,
        "shared_v_speedup_vs_split": split_ms / shared_ms,
        "shared_v_cosine_vs_split": cosine(shared_output, split_output),
        "shared_v_max_abs_vs_split": float(
            (shared_output.float() - split_output.float()).abs().max()
        ),
        "shared_kv_ms": shared_kv_ms,
        "shared_kv_speedup_vs_dense": dense_ms / shared_kv_ms,
        "shared_kv_speedup_vs_split": split_ms / shared_kv_ms,
        "shared_kv_full_cosine_vs_dense": cosine(shared_kv_output, dense_output),
        "shared_kv_protected_cosine_vs_dense": cosine(
            shared_kv_output[:, prefix], dense_output[:, prefix]
        ),
        "shared_kv_video_cosine_vs_dense": cosine(
            shared_kv_output[:, video], dense_output[:, video]
        ),
        "production_ms": production_ms,
        "production_speedup_vs_dense": dense_ms / production_ms,
        "production_cosine_vs_split": cosine(production_output, split_output),
        "production_max_abs_vs_split": float(
            (production_output.float() - split_output.float()).abs().max()
        ),
        "production_full_cosine_vs_dense": cosine(
            production_output, dense_output
        ),
        "production_video_cosine_vs_dense": cosine(
            production_output[:, video], dense_output[:, video]
        ),
        "full_cosine_vs_dense": cosine(split_output, dense_output),
        "protected_query_cosine_vs_dense": cosine(
            split_output[:, prefix], dense_output[:, prefix]
        ),
        "video_query_cosine_vs_dense": cosine(
            split_output[:, video], dense_output[:, video]
        ),
        "dense_samples_ms": dense_samples,
        "split_samples_ms": split_samples,
        "shared_v_samples_ms": shared_samples,
        "shared_kv_samples_ms": shared_kv_samples,
        "production_samples_ms": production_samples,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps(document, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
