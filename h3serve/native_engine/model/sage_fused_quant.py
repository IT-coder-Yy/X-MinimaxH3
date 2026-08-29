"""SM89 SageAttention preparation kernels with reference BF16 boundaries."""

from __future__ import annotations

from typing import Any


_KERNEL: Any | None = None


def _kernel():
    """Build lazily so the public Native graph remains CPU importable."""

    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def quant_key_sub_mean_per_thread_int8_kernel(
        input_ptr,
        mean_ptr,
        output_ptr,
        scale_ptr,
        length,
        stride_iz,
        stride_ih,
        stride_in,
        stride_mz,
        stride_mh,
        stride_oz,
        stride_oh,
        stride_on,
        stride_sz,
        stride_sh,
        channels: tl.constexpr,
        block: tl.constexpr,
    ):
        block_id = tl.program_id(0) // 4
        thread_id = tl.program_id(0) % 4
        head = tl.program_id(1)
        batch = tl.program_id(2)
        rows0 = block_id * block + tl.arange(0, block // 8) * 8 + thread_id * 2
        rows1 = rows0 + 1
        columns = tl.arange(0, channels)
        mean = tl.load(
            mean_ptr + batch * stride_mz + head * stride_mh + columns
        )
        pointers0 = (
            input_ptr
            + batch * stride_iz
            + head * stride_ih
            + rows0[:, None] * stride_in
            + columns[None, :]
        )
        pointers1 = (
            input_ptr
            + batch * stride_iz
            + head * stride_ih
            + rows1[:, None] * stride_in
            + columns[None, :]
        )
        valid0 = rows0[:, None] < length
        valid1 = rows1[:, None] < length
        value0 = tl.load(pointers0, mask=valid0)
        value1 = tl.load(pointers1, mask=valid1)
        # Reference path is ``centered = k - k.mean(...)`` where both K and
        # mean are BF16 and the centered tensor is materialized as BF16 before
        # the upstream quantizer casts to FP32.  Preserve that store boundary
        # explicitly; an FP32 subtract here is observably different.
        centered0 = (value0 - mean[None, :]).to(tl.bfloat16).to(tl.float32)
        centered1 = (value1 - mean[None, :]).to(tl.bfloat16).to(tl.float32)
        # Upstream first materializes the valid ``K - mean(K)`` tensor and
        # only then launches a masked quantizer.  A masked load in this fused
        # kernel yields zero for an out-of-range row; subtracting a non-zero
        # mean from that padding would incorrectly let ``-mean`` determine
        # the final ragged block's scale.  Restore the upstream boundary by
        # zeroing invalid centered rows before the reduction.  This matters
        # for every H3 packed length that is not divisible by 64.
        centered0 = tl.where(valid0, centered0, 0.0)
        centered1 = tl.where(valid1, centered1, 0.0)
        scale = (
            tl.maximum(tl.max(tl.abs(centered0)), tl.max(tl.abs(centered1)))
            / 127.0
            + 0.0000001
        )
        quantized0 = centered0 / scale
        quantized1 = centered1 / scale
        quantized0 += 0.5 * tl.where(quantized0 >= 0, 1, -1)
        quantized1 += 0.5 * tl.where(quantized1 >= 0, 1, -1)
        quantized0 = quantized0.to(tl.int8)
        quantized1 = quantized1.to(tl.int8)
        output0 = (
            output_ptr
            + batch * stride_oz
            + head * stride_oh
            + rows0[:, None] * stride_on
            + columns[None, :]
        )
        output1 = (
            output_ptr
            + batch * stride_oz
            + head * stride_oh
            + rows1[:, None] * stride_on
            + columns[None, :]
        )
        tl.store(output0, quantized0, mask=valid0)
        tl.store(output1, quantized1, mask=valid1)
        tl.store(
            scale_ptr
            + batch * stride_sz
            + head * stride_sh
            + block_id * 4
            + thread_id,
            scale,
        )

    _KERNEL = quant_key_sub_mean_per_thread_int8_kernel
    return _KERNEL


def quantize_key_sub_mean_per_thread_int8(
    key,
    key_mean,
    output,
    scale,
) -> None:
    """Write exact upstream per-thread K quantization without centered K."""

    if key.ndim != 4 or key.shape != output.shape:
        raise ValueError("key/output must be matching [batch,tokens,heads,channels]")
    if key_mean.shape != (key.shape[0], 1, key.shape[2], key.shape[3]):
        raise ValueError("key_mean must be [batch,1,heads,channels]")
    if key.shape[-1] != 128:
        raise ValueError("H3 fused K quantization requires head dimension 128")
    batch, tokens, heads, channels = (int(value) for value in key.shape)
    expected_scale = (batch, heads, (tokens + 63) // 64 * 4)
    if tuple(scale.shape) != expected_scale:
        raise ValueError(f"K scale shape {tuple(scale.shape)} != {expected_scale}")
    grid = ((tokens + 63) // 64 * 4, heads, batch)
    _kernel()[grid](
        key,
        key_mean,
        output,
        scale,
        tokens,
        key.stride(0),
        key.stride(2),
        key.stride(1),
        key_mean.stride(0),
        key_mean.stride(2),
        output.stride(0),
        output.stride(2),
        output.stride(1),
        scale.stride(0),
        scale.stride(1),
        channels=channels,
        block=64,
    )


def quantize_qk_sub_mean_per_thread_int8_hnd(
    query,
    key,
    key_mean,
):
    """Match Sage ``per_thread_int8`` in HND layout without centered BF16 K."""

    import torch
    from sageattention.triton.quant_per_thread import (
        quant_query_per_thread_int8_kernel,
    )

    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("fused HND Q/K quantization requires rank-4 tensors")
    if query.shape[0] != key.shape[0] or query.shape[1] != key.shape[1]:
        raise ValueError("fused HND Q/K batch and head counts must match")
    if query.shape[-1] != 128 or key.shape[-1] != 128:
        raise ValueError("fused HND Q/K quantization requires head dimension 128")
    if key_mean.shape != (key.shape[0], key.shape[1], 1, key.shape[3]):
        raise ValueError("fused HND key mean must be [batch,head,1,channels]")
    batch, heads, query_tokens, channels = query.shape
    key_tokens = int(key.shape[2])
    query_int8 = torch.empty_like(query, dtype=torch.int8)
    key_int8 = torch.empty_like(key, dtype=torch.int8)
    query_scale = torch.empty(
        (batch, heads, (query_tokens + 127) // 128 * 32),
        device=query.device,
        dtype=torch.float32,
    )
    key_scale = torch.empty(
        (batch, heads, (key_tokens + 63) // 64 * 4),
        device=key.device,
        dtype=torch.float32,
    )
    query_grid = ((query_tokens + 127) // 128 * 32, heads, batch)
    quant_query_per_thread_int8_kernel[query_grid](
        query,
        query_int8,
        query_scale,
        query_tokens,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query_int8.stride(0),
        query_int8.stride(1),
        query_int8.stride(2),
        query_scale.stride(0),
        query_scale.stride(1),
        C=channels,
        BLK=32,
    )
    key_grid = ((key_tokens + 63) // 64 * 4, heads, batch)
    _kernel()[key_grid](
        key,
        key_mean,
        key_int8,
        key_scale,
        key_tokens,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key_mean.stride(0),
        key_mean.stride(1),
        key_int8.stride(0),
        key_int8.stride(1),
        key_int8.stride(2),
        key_scale.stride(0),
        key_scale.stride(1),
        channels=channels,
        block=64,
    )
    return query_int8, query_scale, key_int8, key_scale


def quantize_qk_sub_mean_per_thread_int8_nhd(
    query,
    key,
    key_mean,
):
    """Match Sage ``per_thread_int8`` in NHD without centered BF16 K."""

    import torch
    from sageattention.triton.quant_per_thread import (
        quant_query_per_thread_int8_kernel,
    )

    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("fused NHD Q/K quantization requires rank-4 tensors")
    if query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2]:
        raise ValueError("fused NHD Q/K batch and head counts must match")
    if query.shape[-1] != 128 or key.shape[-1] != 128:
        raise ValueError("fused NHD Q/K quantization requires head dimension 128")
    if key_mean.shape != (key.shape[0], 1, key.shape[2], key.shape[3]):
        raise ValueError("fused NHD key mean must be [batch,1,head,channels]")
    batch, query_tokens, heads, channels = query.shape
    key_tokens = int(key.shape[1])
    query_int8 = torch.empty_like(query, dtype=torch.int8)
    key_int8 = torch.empty_like(key, dtype=torch.int8)
    query_scale = torch.empty(
        (batch, heads, (query_tokens + 127) // 128 * 32),
        device=query.device,
        dtype=torch.float32,
    )
    key_scale = torch.empty(
        (batch, heads, (key_tokens + 63) // 64 * 4),
        device=key.device,
        dtype=torch.float32,
    )
    query_grid = ((query_tokens + 127) // 128 * 32, heads, batch)
    quant_query_per_thread_int8_kernel[query_grid](
        query,
        query_int8,
        query_scale,
        query_tokens,
        query.stride(0),
        query.stride(2),
        query.stride(1),
        query_int8.stride(0),
        query_int8.stride(2),
        query_int8.stride(1),
        query_scale.stride(0),
        query_scale.stride(1),
        C=channels,
        BLK=32,
    )
    quantize_key_sub_mean_per_thread_int8(
        key, key_mean, key_int8, key_scale
    )
    return query_int8, query_scale, key_int8, key_scale


__all__ = [
    "quantize_key_sub_mean_per_thread_int8",
    "quantize_qk_sub_mean_per_thread_int8_hnd",
    "quantize_qk_sub_mean_per_thread_int8_nhd",
]
