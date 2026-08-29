#!/usr/bin/env python3
"""Locate large-row numerical boundaries in H3's real INT8 QKV projection.

The native 1080p/15s layout produces more than 2**32 BF16 output elements in
one fused QKV call.  This probe compares that whole projection against the
same row-local operator evaluated in bounded chunks and records the first
row at which they disagree.  It is a kernel-correctness probe, not a video
quality claim.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "models/diffusion_models/"
            "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
        ),
    )
    parser.add_argument("--block", type=int, default=20)
    parser.add_argument("--tokens", type=int, default=220_003)
    parser.add_argument("--hidden-width", type=int, default=5_376)
    parser.add_argument("--comparison-chunk-tokens", type=int, default=8_192)
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument(
        "--sparge-build-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "H3_NATIVE_SPARGE_BUILD_DIR",
                "runtime/extensions/sparge-sm89-py310-torch213-cu133",
            )
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runtime/calibration/long_1080p15_20260824/"
            "qkv_projection_boundary.json"
        ),
    )
    return parser.parse_args()


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if args.tokens <= 0 or args.comparison_chunk_tokens <= 0:
        raise SystemExit("token counts must be positive")
    if not 0 <= args.block < 50:
        raise SystemExit("--block must lie inside [0, 50)")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this probe requires one RTX 4090 / SM89 GPU")
    build_dir = args.sparge_build_dir.resolve()
    if not build_dir.is_dir():
        raise SystemExit(f"missing Sparge build: {build_dir}")
    sys.path.insert(0, str(build_dir))

    from h3serve.native_engine.model import (
        SafeTensorSource,
        apply_qknorm_rope,
        assemble_pruned_block,
        comfy_kitchen_int8_kernel,
        sage_attention_sm89,
    )
    from h3serve.native_engine.sm89_policy import configure_sm89_runtime

    configure_sm89_runtime(quant_backend="cuda", smoke_test=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    with SafeTensorSource(str(args.checkpoint)) as source:
        block = assemble_pruned_block(
            args.block,
            source,
            device=device,
            compute_dtype=torch.bfloat16,
            int8_kernel=comfy_kitchen_int8_kernel,
            attention_backend=sage_attention_sm89,
        )
    block.eval().requires_grad_(False)
    projection = block.attention.qkv_proj
    output_width = int(projection.out_features)
    hidden = torch.empty(
        args.tokens,
        args.hidden_width,
        device=device,
        dtype=torch.bfloat16,
    ).normal_().clamp_(-4, 4)
    torch.cuda.synchronize()

    baseline_allocated = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    start_event.record()
    whole = projection(hidden)
    stop_event.record()
    stop_event.synchronize()
    whole_seconds = time.perf_counter() - started
    whole_cuda_ms = float(start_event.elapsed_time(stop_event))
    whole_checksum = float(
        whole[:: max(1, args.tokens // 32)].float().mean().cpu()
    )

    chunk_reports: list[dict[str, object]] = []
    first_mismatch_row: int | None = None
    last_mismatch_row: int | None = None
    mismatch_rows = 0
    all_exact = True
    max_abs = 0.0
    weighted_abs_sum = 0.0
    compared_elements = 0
    chunk_started = time.perf_counter()
    for start in range(0, args.tokens, args.comparison_chunk_tokens):
        stop = min(args.tokens, start + args.comparison_chunk_tokens)
        candidate = projection(hidden[start:stop])
        reference = whole[start:stop]
        exact = bool(torch.equal(candidate, reference))
        report: dict[str, object] = {
            "start": start,
            "stop": stop,
            "exact_equal": exact,
        }
        if not exact:
            all_exact = False
            unequal_rows = torch.nonzero(
                torch.ne(candidate, reference).any(dim=1), as_tuple=False
            ).flatten()
            local_first = int(unequal_rows[0].item())
            local_last = int(unequal_rows[-1].item())
            bad_rows = int(unequal_rows.numel())
            first_mismatch_row = (
                start + local_first
                if first_mismatch_row is None
                else first_mismatch_row
            )
            last_mismatch_row = start + local_last
            mismatch_rows += bad_rows
            delta = candidate.sub(reference)
            chunk_max = float(delta.abs().max().item())
            chunk_mean = float(delta.float().abs().mean().item())
            elements = int(delta.numel())
            max_abs = max(max_abs, chunk_max)
            weighted_abs_sum += chunk_mean * elements
            compared_elements += elements
            report.update(
                {
                    "mismatch_rows": bad_rows,
                    "first_mismatch_row": start + local_first,
                    "last_mismatch_row": start + local_last,
                    "max_abs": chunk_max,
                    "mean_abs": chunk_mean,
                }
            )
            del unequal_rows, delta
        chunk_reports.append(report)
        del candidate, reference
    torch.cuda.synchronize()
    chunk_seconds = time.perf_counter() - chunk_started

    # The fused QK RMSNorm+RoPE kernel sees strided views into the very large
    # QKV allocation.  Compare that call independently from Attention.  An
    # identity rotation table still exercises the exact fused indexing and
    # BF16 store boundary without tying this kernel diagnostic to one prompt
    # or reference-media layout.
    inner_width = output_width // 3
    query, key, _value = whole.split(inner_width, dim=-1)
    head_dim = int(block.attention.head_dim)
    heads = int(block.attention.num_heads)
    query = query.view(args.tokens, heads, head_dim)
    key = key.view(args.tokens, heads, head_dim)
    identity_rotation = torch.eye(
        2, device=device, dtype=torch.bfloat16
    ).reshape(1, 1, 1, 1, 2, 2).expand(
        1, args.tokens, 1, 48, 2, 2
    ).contiguous()
    qk_started = time.perf_counter()
    query, key = apply_qknorm_rope(
        query,
        key,
        q_weight=block.attention.q_norm.weight,
        k_weight=block.attention.k_norm.weight,
        frequencies=identity_rotation,
        eps=block.attention.q_norm.eps,
    )
    torch.cuda.synchronize()
    qk_whole_seconds = time.perf_counter() - qk_started
    qk_all_exact = True
    qk_first_mismatch_row: int | None = None
    qk_last_mismatch_row: int | None = None
    qk_mismatch_rows = 0
    qk_max_abs = 0.0
    qk_chunks: list[dict[str, object]] = []
    qk_chunk_started = time.perf_counter()
    for start in range(0, args.tokens, args.comparison_chunk_tokens):
        stop = min(args.tokens, start + args.comparison_chunk_tokens)
        chunk_qkv = projection(hidden[start:stop])
        q_chunk, k_chunk, _ = chunk_qkv.split(inner_width, dim=-1)
        chunk_shape = (stop - start, heads, head_dim)
        q_chunk, k_chunk = apply_qknorm_rope(
            q_chunk.view(chunk_shape),
            k_chunk.view(chunk_shape),
            q_weight=block.attention.q_norm.weight,
            k_weight=block.attention.k_norm.weight,
            frequencies=identity_rotation[:, start:stop],
            eps=block.attention.q_norm.eps,
        )
        q_reference = query[start:stop]
        k_reference = key[start:stop]
        exact = bool(
            torch.equal(q_chunk, q_reference)
            and torch.equal(k_chunk, k_reference)
        )
        chunk_report: dict[str, object] = {
            "start": start,
            "stop": stop,
            "exact_equal": exact,
        }
        if not exact:
            qk_all_exact = False
            unequal_rows = torch.nonzero(
                torch.ne(q_chunk, q_reference).any(dim=(1, 2))
                | torch.ne(k_chunk, k_reference).any(dim=(1, 2)),
                as_tuple=False,
            ).flatten()
            local_first = int(unequal_rows[0].item())
            local_last = int(unequal_rows[-1].item())
            bad_rows = int(unequal_rows.numel())
            qk_first_mismatch_row = (
                start + local_first
                if qk_first_mismatch_row is None
                else qk_first_mismatch_row
            )
            qk_last_mismatch_row = start + local_last
            qk_mismatch_rows += bad_rows
            q_max = float(q_chunk.sub(q_reference).abs().max().item())
            k_max = float(k_chunk.sub(k_reference).abs().max().item())
            qk_max_abs = max(qk_max_abs, q_max, k_max)
            chunk_report.update(
                {
                    "mismatch_rows": bad_rows,
                    "first_mismatch_row": start + local_first,
                    "last_mismatch_row": start + local_last,
                    "q_max_abs": q_max,
                    "k_max_abs": k_max,
                }
            )
            del unequal_rows
        qk_chunks.append(chunk_report)
        del chunk_qkv, q_chunk, k_chunk, q_reference, k_reference
    torch.cuda.synchronize()
    qk_chunk_seconds = time.perf_counter() - qk_chunk_started

    # The output projection is also row-local.  Use normalized Query rows as
    # a representative contiguous Attention result, then compare one whole
    # INT8 projection with the same projection evaluated in row chunks.
    attended = query.reshape(args.tokens, inner_width).contiguous()
    del whole, query, key, _value, identity_rotation, hidden
    torch.cuda.empty_cache()
    out_started = time.perf_counter()
    whole_out = block.attention.out_proj(attended)
    torch.cuda.synchronize()
    out_whole_seconds = time.perf_counter() - out_started
    out_all_exact = True
    out_first_mismatch_row: int | None = None
    out_last_mismatch_row: int | None = None
    out_mismatch_rows = 0
    out_max_abs = 0.0
    out_chunks: list[dict[str, object]] = []
    out_chunk_started = time.perf_counter()
    for start in range(0, args.tokens, args.comparison_chunk_tokens):
        stop = min(args.tokens, start + args.comparison_chunk_tokens)
        candidate = block.attention.out_proj(attended[start:stop])
        reference = whole_out[start:stop]
        exact = bool(torch.equal(candidate, reference))
        chunk_report = {"start": start, "stop": stop, "exact_equal": exact}
        if not exact:
            out_all_exact = False
            unequal_rows = torch.nonzero(
                torch.ne(candidate, reference).any(dim=1), as_tuple=False
            ).flatten()
            local_first = int(unequal_rows[0].item())
            local_last = int(unequal_rows[-1].item())
            bad_rows = int(unequal_rows.numel())
            out_first_mismatch_row = (
                start + local_first
                if out_first_mismatch_row is None
                else out_first_mismatch_row
            )
            out_last_mismatch_row = start + local_last
            out_mismatch_rows += bad_rows
            chunk_max = float(candidate.sub(reference).abs().max().item())
            out_max_abs = max(out_max_abs, chunk_max)
            chunk_report.update(
                {
                    "mismatch_rows": bad_rows,
                    "first_mismatch_row": start + local_first,
                    "last_mismatch_row": start + local_last,
                    "max_abs": chunk_max,
                }
            )
            del unequal_rows
        out_chunks.append(chunk_report)
        del candidate, reference
    torch.cuda.synchronize()
    out_chunk_seconds = time.perf_counter() - out_chunk_started

    element_limit_floor_row = (2**32) // output_width
    report = {
        "schema_version": "h3_long_qkv_projection_boundary_v1",
        "warning": "Synthetic hidden-state projection probe; no video-quality claim.",
        "runtime": {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "checkpoint": str(args.checkpoint.resolve()),
            "sparge_build_dir": str(build_dir),
        },
        "shape": {
            "tokens": args.tokens,
            "hidden_width": args.hidden_width,
            "output_width": output_width,
            "output_elements": args.tokens * output_width,
            "output_gib": args.tokens * output_width * 2 / GIB,
            "uint32_element_limit": 2**32,
            "uint32_limit_floor_rows": element_limit_floor_row,
            "first_row_exceeding_uint32_elements": element_limit_floor_row + 1,
            "comparison_chunk_tokens": args.comparison_chunk_tokens,
        },
        "whole_projection": {
            "cuda_ms": whole_cuda_ms,
            "wall_seconds": whole_seconds,
            "checksum_mean_sample": whole_checksum,
        },
        "chunked_comparison": {
            "wall_seconds": chunk_seconds,
            "all_exact": all_exact,
            "mismatch_rows": mismatch_rows,
            "first_mismatch_row": first_mismatch_row,
            "last_mismatch_row": last_mismatch_row,
            "max_abs": max_abs,
            "mean_abs_over_mismatched_chunks": (
                weighted_abs_sum / compared_elements if compared_elements else 0.0
            ),
            "chunks": chunk_reports,
        },
        "qknorm_rope_comparison": {
            "whole_wall_seconds": qk_whole_seconds,
            "chunked_wall_seconds": qk_chunk_seconds,
            "all_exact": qk_all_exact,
            "mismatch_rows": qk_mismatch_rows,
            "first_mismatch_row": qk_first_mismatch_row,
            "last_mismatch_row": qk_last_mismatch_row,
            "max_abs": qk_max_abs,
            "chunks": qk_chunks,
        },
        "out_projection_comparison": {
            "whole_wall_seconds": out_whole_seconds,
            "chunked_wall_seconds": out_chunk_seconds,
            "all_exact": out_all_exact,
            "mismatch_rows": out_mismatch_rows,
            "first_mismatch_row": out_first_mismatch_row,
            "last_mismatch_row": out_last_mismatch_row,
            "max_abs": out_max_abs,
            "chunks": out_chunks,
        },
        "memory": {
            "baseline_allocated_gib": baseline_allocated / GIB,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / GIB,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / GIB,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
