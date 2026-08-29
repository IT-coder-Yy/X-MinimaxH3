#!/usr/bin/env python3
"""Profile mask construction and temporal-mask reuse at the real H3 shape.

This benchmark isolates the production Split0.50 attention path used by the
long-sequence H3 backend.  It compares a freshly selected Sparge block map with
reusing a previously selected map while still requantizing the current Q/K.
The cached-map path is only an algorithmic probe; it is not enabled by the
release planner.
"""

from __future__ import annotations

import argparse
import json
import site
import statistics
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sparge-build-dir",
        type=Path,
        default=Path("runtime/extensions/sparge-sm89-py310-torch28-cu12"),
    )
    parser.add_argument("--tokens", type=int, default=100_163)
    parser.add_argument("--protected-tokens", type=int, default=1_723)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--topk", type=float, default=0.10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runtime/calibration/v19_next_mask_reuse/"
            "split_mask_reuse_real_shape.json"
        ),
    )
    return parser.parse_args()


def timed(operation, *, warmup: int, repeat: int) -> dict[str, object]:
    for _ in range(warmup):
        operation()
        torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    return {"samples_ms": samples, "median_ms": statistics.median(samples)}


def error_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    delta = (reference.float() - candidate.float()).abs()
    return {
        "mean_abs": float(delta.mean().cpu()),
        "max_abs": float(delta.max().cpu()),
        "rmse": float(delta.square().mean().sqrt().cpu()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                reference.float().reshape(1, -1),
                candidate.float().reshape(1, -1),
            ).cpu()
        ),
    }


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this benchmark requires one RTX 4090 / SM89 GPU")
    if not 0 < args.protected_tokens < args.tokens:
        raise SystemExit("protected token count must lie inside the sequence")
    if not 0.0625 <= args.topk <= 1.0:
        raise SystemExit("topk must lie inside [0.0625, 1.0]")

    site.addsitedir(str(args.sparge_build_dir.resolve()))
    sys.path.insert(0, str(args.sparge_build_dir.resolve()))

    from einops import rearrange
    from spas_sage_attn import core as sparge_core
    from spas_sage_attn.utils import (
        block_map_lut_triton,
        get_block_map_meansim_fuse_quant,
        get_vanilla_qk_quant,
        hyperparameter_check,
    )

    from h3serve.native_engine.model import (
        make_split_modality_protected_sparge_attention_sm89,
    )
    from h3serve.native_engine.sm89_policy import configure_sm89_runtime

    configure_sm89_runtime(quant_backend="cuda", smoke_test=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    shape = (args.tokens, args.heads, args.head_dim)
    query = torch.randn(shape, device=device, dtype=dtype)
    key = torch.randn(shape, device=device, dtype=dtype)
    value = torch.randn(shape, device=device, dtype=dtype)
    backend = make_split_modality_protected_sparge_attention_sm89(
        args.topk,
        experimental_minimum_topk=min(0.5, args.topk),
    )

    # Prepare shared K/V exactly as production does.  The experiment isolates
    # the video-query branch, which dominates at the 100k-token shape.
    k, v_fp8, v_scale, heads, _kv_len, head_dim = backend._prepare_kv(key, value)
    video_query = query[args.protected_tokens :]
    q = rearrange(video_query, "L H D -> 1 H L D").contiguous().to(dtype)
    key_mean = k.mean(dim=-2, keepdim=True)
    topk = backend._head_topk(heads, device)
    protected_k_blocks = (args.protected_tokens + 63) // 64
    pv_threshold = hyperparameter_check(50, heads, device)

    def fresh_map_quant():
        return get_block_map_meansim_fuse_quant(
            q,
            k,
            key_mean,
            BLKQ=128,
            BLKK=64,
            simthreshd1=-0.1,
            cdfthreshd=None,
            topk=topk,
            is_causal=False,
        )

    block_map, q_int8, q_scale, k_int8, k_scale = fresh_map_quant()
    block_map[:, :, :, :protected_k_blocks] = True
    lut, valid_block_num = block_map_lut_triton(block_map.contiguous())
    output = torch.empty_like(q)

    def sparse_core(
        q8=q_int8,
        qs=q_scale,
        k8=k_int8,
        ks=k_scale,
        target=output,
    ) -> None:
        sparge_core.qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
            q8,
            k8,
            v_fp8,
            target,
            lut,
            valid_block_num,
            pv_threshold,
            qs,
            ks,
            v_scale,
            1,
            0,
            1,
            1.0 / (head_dim**0.5),
            0,
        )

    def full_fresh() -> torch.Tensor:
        fresh_map, fresh_q8, fresh_qs, fresh_k8, fresh_ks = fresh_map_quant()
        fresh_map[:, :, :, :protected_k_blocks] = True
        fresh_lut, fresh_valid = block_map_lut_triton(fresh_map.contiguous())
        target = torch.empty_like(q)
        sparge_core.qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
            fresh_q8,
            fresh_k8,
            v_fp8,
            target,
            fresh_lut,
            fresh_valid,
            pv_threshold,
            fresh_qs,
            fresh_ks,
            v_scale,
            1,
            0,
            1,
            1.0 / (head_dim**0.5),
            0,
        )
        return target

    def quant_only():
        return get_vanilla_qk_quant(q, k, key_mean, 128, 64)

    def reused_map() -> torch.Tensor:
        rq8, rqs, rk8, rks = quant_only()
        target = torch.empty_like(q)
        sparge_core.qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
            rq8,
            rk8,
            v_fp8,
            target,
            lut,
            valid_block_num,
            pv_threshold,
            rqs,
            rks,
            v_scale,
            1,
            0,
            1,
            1.0 / (head_dim**0.5),
            0,
        )
        return target

    fresh_reference = full_fresh()
    reused_reference = reused_map()
    torch.cuda.synchronize()
    stages = {
        "prepare_kv": timed(
            lambda: backend._prepare_kv(key, value),
            warmup=args.warmup,
            repeat=args.repeat,
        ),
        "fresh_map_and_qk_quant": timed(
            fresh_map_quant, warmup=args.warmup, repeat=args.repeat
        ),
        "qk_quant_only": timed(quant_only, warmup=args.warmup, repeat=args.repeat),
        "lut_from_cached_map": timed(
            lambda: block_map_lut_triton(block_map.contiguous()),
            warmup=args.warmup,
            repeat=args.repeat,
        ),
        "sparse_core_cached_inputs": timed(
            sparse_core, warmup=args.warmup, repeat=args.repeat
        ),
        "fresh_map_full_video_attention": timed(
            full_fresh, warmup=args.warmup, repeat=args.repeat
        ),
        "reused_map_full_video_attention": timed(
            reused_map, warmup=args.warmup, repeat=args.repeat
        ),
    }
    fresh_ms = float(stages["fresh_map_full_video_attention"]["median_ms"])
    reuse_ms = float(stages["reused_map_full_video_attention"]["median_ms"])
    report = {
        "shape": {
            "tokens": args.tokens,
            "protected_tokens": args.protected_tokens,
            "video_tokens": args.tokens - args.protected_tokens,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "topk": args.topk,
        },
        "stages": stages,
        "cached_map_same_input_error": error_stats(
            fresh_reference, reused_reference
        ),
        "reused_map_speedup_video_branch": fresh_ms / reuse_ms,
        "selection_overhead_fraction": max(0.0, (fresh_ms - reuse_ms) / fresh_ms),
        "planner_status": "research_only_not_registered",
        "boundary": (
            "Same-input equivalence only. Cross-step map stability must be "
            "measured on real denoise trajectories before end-to-end use."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
