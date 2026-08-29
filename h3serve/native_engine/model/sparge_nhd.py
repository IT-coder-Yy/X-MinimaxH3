"""NHD-native Sparge preparation for the batch-one SM89 H3 path.

The upstream Sparge Triton helper hard-codes contiguous HND addressing.  H3's
INT8 projection naturally emits contiguous ``[tokens, heads, dim]`` rows, so
materializing HND first is pure layout traffic.  This kernel preserves the
upstream per-block arithmetic and HND metadata shapes while reading and
writing Q/K in native NHD storage.
"""

from __future__ import annotations

from typing import Any


_POOL_SIM_QUANT_NHD: Any | None = None

# Direct NHD and materialized HND use the same PyTorch reduction tree through
# the largest conditioned 720p15 shape physically checked so far.  PyTorch
# switches reduction geometry at the 1080p15/220k-token bucket, where reducing
# NHD directly changes a few BF16 mean values.  Above the proven boundary,
# materialize only one head (about 54 MiB at 1080p15), preserving the reference
# HND reduction without recreating the multi-GiB full HND tensor.
_NATIVE_MEAN_EXACT_MAX_TOKENS = 101_695


def hnd_compatible_key_mean_nhd(value):
    """Return ``HND.contiguous().mean(tokens)`` bytes from NHD-resident K."""

    import torch

    if value.ndim != 4 or value.shape[0] != 1 or not value.is_contiguous():
        raise ValueError("NHD key mean requires contiguous batch-one [B,N,H,D]")
    if int(value.shape[1]) <= _NATIVE_MEAN_EXACT_MAX_TOKENS:
        return value.mean(dim=1, keepdim=True)
    means = []
    for head in range(int(value.shape[2])):
        materialized_head = (
            value[:, :, head : head + 1, :]
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        means.append(materialized_head.mean(dim=2, keepdim=True))
        del materialized_head
    return (
        torch.cat(means, dim=1)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def _pool_sim_quant_nhd_kernel():
    global _POOL_SIM_QUANT_NHD
    if _POOL_SIM_QUANT_NHD is not None:
        return _POOL_SIM_QUANT_NHD

    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
        x_ptr,
        mean_ptr,
        pool_ptr,
        similar_ptr,
        quant_ptr,
        scale_ptr,
        threshold_ptr,
        stride_xb,
        stride_xn,
        stride_xh,
        stride_mb,
        stride_mh,
        stride_pb,
        stride_ph,
        stride_pn,
        stride_qb,
        stride_qn,
        stride_qh,
        stride_sb,
        stride_sh,
        length: tl.constexpr,
        channels: tl.constexpr,
        block: tl.constexpr,
        fuse_mean: tl.constexpr,
    ):
        batch = tl.program_id(0)
        head = tl.program_id(1)
        block_id = tl.program_id(2)
        rows = block_id * block + tl.arange(0, block)
        columns = tl.arange(0, channels)
        row_mask = rows[:, None] < length
        pointers = (
            x_ptr
            + batch * stride_xb
            + rows[:, None] * stride_xn
            + head * stride_xh
            + columns[None, :]
        )
        value = tl.load(pointers, mask=row_mask)
        if fuse_mean:
            mean = tl.load(
                mean_ptr
                + batch * stride_mb
                + head * stride_mh
                + columns
            )
            value -= mean[None, :]
            value = tl.where(row_mask, value, 0.0)

        valid_rows = tl.minimum(block, length - block_id * block)
        value_fp32 = value.to(tl.float32)
        pooled = tl.sum(value_fp32, axis=0) / valid_rows
        norm = tl.sqrt(
            tl.sum(value_fp32 * value_fp32, axis=1, keep_dims=True)
        )
        normalized = (value / norm).to(tl.float16)
        gram = tl.dot(normalized, tl.trans(normalized))
        similarity = (
            tl.sum(gram).to(tl.float32) / (valid_rows * valid_rows)
        ) > tl.load(threshold_ptr + head)

        pool_out = (
            pool_ptr
            + batch * stride_pb
            + head * stride_ph
            + block_id * stride_pn
            + columns
        )
        tl.store(pool_out, pooled)
        total_blocks = tl.num_programs(2)
        tl.store(
            similar_ptr
            + batch * tl.num_programs(1) * total_blocks
            + head * total_blocks
            + block_id,
            similarity,
        )

        scale = tl.max(tl.abs(value_fp32)) / 127.0 + 0.0000001
        quantized = value_fp32 / scale
        quantized += 0.5 * tl.where(quantized >= 0, 1, -1)
        quantized = quantized.to(tl.int8)
        quant_out = (
            quant_ptr
            + batch * stride_qb
            + rows[:, None] * stride_qn
            + head * stride_qh
            + columns[None, :]
        )
        tl.store(quant_out, quantized, mask=row_mask)
        tl.store(
            scale_ptr
            + batch * stride_sb
            + head * stride_sh
            + block_id,
            scale,
        )

    _POOL_SIM_QUANT_NHD = kernel
    return kernel


def pool_sim_quant_nhd(value, mean, block_size: int, threshold):
    """Return upstream-compatible pool/sim/INT8 metadata from contiguous NHD."""

    import torch

    if value.ndim != 4 or not value.is_contiguous():
        raise ValueError("NHD Sparge input must be contiguous [B,N,H,D]")
    if value.dtype != torch.bfloat16:
        raise ValueError("NHD Sparge input must be BF16")
    if value.shape[-1] != 128:
        raise ValueError("NHD Sparge preparation requires head dimension 128")
    if block_size not in (64, 128):
        raise ValueError("NHD Sparge block size must be 64 or 128")
    batch, tokens, heads, channels = (int(item) for item in value.shape)
    if mean is not None and tuple(mean.shape) != (
        batch,
        1,
        heads,
        channels,
    ):
        raise ValueError("NHD key mean must be [B,1,H,D]")
    if tuple(threshold.shape) != (heads,) or threshold.device != value.device:
        raise ValueError("NHD Sparge threshold must contain one value per head")

    blocks = (tokens + block_size - 1) // block_size
    pool = torch.empty(
        (batch, heads, blocks, channels),
        device=value.device,
        dtype=value.dtype,
    )
    similar = torch.empty(
        (batch, heads, blocks), device=value.device, dtype=torch.bool
    )
    quantized = torch.empty_like(value, dtype=torch.int8)
    scale = torch.empty(
        (batch, heads, blocks), device=value.device, dtype=torch.float32
    )
    mean_arg = value if mean is None else mean
    _pool_sim_quant_nhd_kernel()[(batch, heads, blocks)](
        value,
        mean_arg,
        pool,
        similar,
        quantized,
        scale,
        threshold,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        mean_arg.stride(0),
        mean_arg.stride(2),
        pool.stride(0),
        pool.stride(1),
        pool.stride(2),
        quantized.stride(0),
        quantized.stride(1),
        quantized.stride(2),
        scale.stride(0),
        scale.stride(1),
        length=tokens,
        channels=channels,
        block=block_size,
        fuse_mean=mean is not None,
    )
    return pool, similar, quantized, scale


__all__ = ["hnd_compatible_key_mean_nhd", "pool_sim_quant_nhd"]
