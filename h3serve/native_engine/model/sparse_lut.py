"""Long-sequence sparse block-map compaction kernels.

SpargeAttention's reference Triton converter assigns one program to each
query/head row and scans all KV blocks serially.  H3's 1080p15 shape has 3,438
KV blocks, so that conversion becomes a measurable part of every real DiT
block.  The kernels below retain the exact full-stride, delta-encoded LUT ABI
while parallelizing the scan inside each row.

Triton is imported lazily so the public model package remains CPU importable.
"""

from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _parallel_lut_kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def block_map_to_absolute_lut(
        block_map,
        lut,
        valid_block_num,
        key_blocks: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        selected = tl.load(
            block_map + row * key_blocks + offsets,
            mask=offsets < key_blocks,
            other=0,
        ).to(tl.int32)
        rank = tl.cumsum(selected, axis=0) - 1
        tl.store(
            lut + row * key_blocks + rank,
            offsets,
            mask=(offsets < key_blocks) & (selected != 0),
        )
        tl.store(valid_block_num + row, tl.sum(selected, axis=0))

    @triton.jit
    def absolute_lut_to_delta(
        absolute_lut,
        delta_lut,
        valid_block_num,
        key_blocks: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        ranks = tl.arange(0, BLOCK)
        count = tl.load(valid_block_num + row)
        current = tl.load(
            absolute_lut + row * key_blocks + ranks,
            mask=ranks < count,
            other=0,
        )
        previous = tl.load(
            absolute_lut + row * key_blocks + ranks - 1,
            mask=(ranks > 0) & (ranks < count),
            other=0,
        )
        tl.store(
            delta_lut + row * key_blocks + ranks,
            current - previous,
            mask=ranks < count,
        )

    return triton, block_map_to_absolute_lut, absolute_lut_to_delta


@lru_cache(maxsize=1)
def _partial_topk_fill_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def fill_partial_topk(
        block_map,
        selected_count,
        selected_indices,
        key_blocks: tl.constexpr,
        selected_width: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        count = tl.load(selected_count + row)
        active = (offsets < selected_width) & (offsets < count)
        indices = tl.load(
            selected_indices + row * selected_width + offsets,
            mask=active,
            other=0,
        )
        tl.store(
            block_map + row * key_blocks + indices,
            1,
            mask=active,
        )

    return triton, fill_partial_topk


def fill_block_map_partial_topk(
    block_map: torch.Tensor,
    selected_count: torch.Tensor,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    """Set compact Top-K indices without Sparge's full-K stride assumption."""

    if block_map.ndim != 4 or not block_map.is_contiguous():
        raise ValueError("partial Top-K fill requires a contiguous rank-4 map")
    if block_map.dtype != torch.bool or not block_map.is_cuda:
        raise ValueError("partial Top-K fill requires a CUDA boolean map")
    if selected_count.shape != block_map.shape[:-1]:
        raise ValueError("partial Top-K counts do not match block-map rows")
    if selected_indices.shape[:-1] != block_map.shape[:-1]:
        raise ValueError("partial Top-K indices do not match block-map rows")
    if not selected_count.is_contiguous() or not selected_indices.is_contiguous():
        raise ValueError("partial Top-K metadata must be contiguous")
    width = int(selected_indices.shape[-1])
    if width <= 0:
        raise ValueError("partial Top-K indices cannot be empty")
    triton, kernel = _partial_topk_fill_kernel()
    block = triton.next_power_of_2(width)
    rows = block_map.numel() // block_map.shape[-1]
    kernel[(rows,)](
        block_map,
        selected_count,
        selected_indices,
        key_blocks=block_map.shape[-1],
        selected_width=width,
        BLOCK=block,
        num_warps=4,
    )
    return block_map


def parallel_block_map_lut(
    block_map: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Sparge-compatible delta LUT and valid counts.

    The unused tail of each full-stride LUT row is intentionally unspecified;
    Sparge reads only ``valid_block_num`` entries.  This avoids a 200MB zero
    fill at the 1080p15 chunk shape without changing any consumed value.
    """

    if block_map.ndim != 4 or not block_map.is_contiguous():
        raise ValueError("parallel sparse LUT requires a contiguous rank-4 map")
    if block_map.dtype != torch.bool or not block_map.is_cuda:
        raise ValueError("parallel sparse LUT requires a CUDA boolean map")
    batch, heads, query_blocks, key_blocks = block_map.shape
    if key_blocks <= 0:
        raise ValueError("parallel sparse LUT requires at least one KV block")
    triton, compact, delta = _parallel_lut_kernels()
    block = triton.next_power_of_2(key_blocks)
    if block > 8192:
        raise ValueError("parallel sparse LUT supports at most 8192 KV blocks")
    rows = batch * heads * query_blocks
    absolute_lut = torch.empty(
        block_map.shape, dtype=torch.int32, device=block_map.device
    )
    lut = torch.empty(block_map.shape, dtype=torch.int32, device=block_map.device)
    valid = torch.empty(
        (batch, heads, query_blocks),
        dtype=torch.int32,
        device=block_map.device,
    )
    compact[(rows,)](
        block_map,
        absolute_lut,
        valid,
        key_blocks=key_blocks,
        BLOCK=block,
        num_warps=8,
    )
    delta[(rows,)](
        absolute_lut,
        lut,
        valid,
        key_blocks=key_blocks,
        BLOCK=block,
        num_warps=8,
    )
    del absolute_lut
    return lut, valid


__all__ = ["fill_block_map_partial_topk", "parallel_block_map_lut"]
