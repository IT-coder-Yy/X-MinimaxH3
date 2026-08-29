"""Exact slab-wise Sage FP8-V writer for memory-bounded Attention."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_PERMUTATION = (0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15)


@triton.jit
def _write_sage_fp8_hnd_full_kernel(
    source,
    target,
    absmax,
    rows: tl.constexpr,
    padded_rows: tl.constexpr,
    head_dim: tl.constexpr,
    row_block: tl.constexpr,
    dim_block: tl.constexpr,
):
    """Transpose, Sage-permute and FP8-quantize one HND tile."""

    head = tl.program_id(0)
    output_row = tl.program_id(1) * row_block + tl.arange(0, row_block)
    dimension = tl.program_id(2) * dim_block + tl.arange(0, dim_block)
    within = output_row % 16
    group = output_row - within
    source_within = tl.where(
        within < 2,
        within,
        tl.where(
            within < 4,
            within + 6,
            tl.where(
                within < 6,
                within - 2,
                tl.where(
                    within < 8,
                    within + 4,
                    tl.where(
                        within < 10,
                        within - 4,
                        tl.where(
                            within < 12,
                            within + 2,
                            tl.where(within < 14, within - 6, within),
                        ),
                    ),
                ),
            ),
        ),
    )
    source_row = group + source_within
    source_pointer = (
        source
        + head * rows * head_dim
        + source_row[:, None] * head_dim
        + dimension[None, :]
    )
    source_mask = (source_row[:, None] < rows) & (
        dimension[None, :] < head_dim
    )
    value = tl.load(source_pointer, mask=source_mask, other=0.0).to(tl.float32)
    maximum = tl.load(
        absmax + head * head_dim + dimension,
        mask=dimension < head_dim,
        other=1.0,
    )
    scaled = value * (2.25 / maximum[None, :])
    converted = tl.inline_asm_elementwise(
        """{
        .reg .b16 lo;
        .reg .b16 hi;
        cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;
        cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;
        mov.b32 $0, {lo, hi};
        }""",
        "=r,f,f,f,f",
        [scaled],
        dtype=tl.float8e4nv,
        is_pure=True,
        pack=4,
    )
    target_pointer = (
        target
        + head * head_dim * padded_rows
        + dimension[:, None] * padded_rows
        + output_row[None, :]
    )
    target_mask = (dimension[:, None] < head_dim) & (
        output_row[None, :] < padded_rows
    )
    tl.store(target_pointer, tl.trans(converted), mask=target_mask)


@triton.jit
def _write_sage_fp8_kernel(
    source,
    target,
    absmax,
    rows: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    padded_rows: tl.constexpr,
    target_tokens: tl.constexpr,
    target_start: tl.constexpr,
    source_layout_hnd: tl.constexpr,
    layout_hnd: tl.constexpr,
    division_mode: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    total = padded_rows * heads * head_dim
    mask = offsets < total
    dimension = offsets % head_dim
    head = (offsets // head_dim) % heads
    output_row = offsets // (heads * head_dim)
    within = output_row % 16
    group = output_row - within
    # Output rows contain input rows in Sage's FP8 MMA permutation.
    source_within = tl.where(
        within < 2,
        within,
        tl.where(
            within < 4,
            within + 6,
            tl.where(
                within < 6,
                within - 2,
                tl.where(
                    within < 8,
                    within + 4,
                    tl.where(
                        within < 10,
                        within - 4,
                        tl.where(
                            within < 12,
                            within + 2,
                            tl.where(within < 14, within - 6, within),
                        ),
                    ),
                ),
            ),
        ),
    )
    source_row = group + source_within
    if source_layout_hnd:
        source_offset = (head * rows + source_row) * head_dim + dimension
    else:
        source_offset = (source_row * heads + head) * head_dim + dimension
    value = tl.load(source + source_offset, mask=mask & (source_row < rows), other=0.0)
    maximum = tl.load(absmax + head * head_dim + dimension)
    if division_mode == 0:
        reciprocal = tl.inline_asm_elementwise(
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            [maximum],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        scaled = value.to(tl.float32) * (2.25 * reciprocal)
    elif division_mode == 1:
        numerator = tl.full(maximum.shape, 2.25, tl.float32)
        reciprocal_scale = tl.inline_asm_elementwise(
            "div.approx.ftz.f32 $0, $1, $2;",
            "=f,f,f",
            [numerator, maximum],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        scaled = value.to(tl.float32) * reciprocal_scale
    else:
        scaled = value.to(tl.float32) * (2.25 / maximum)
    global_row = target_start + output_row
    if layout_hnd:
        target_offset = (head * head_dim + dimension) * target_tokens + global_row
    else:
        target_offset = (dimension * heads + head) * target_tokens + global_row
    converted = tl.inline_asm_elementwise(
        """{
        .reg .b16 lo;
        .reg .b16 hi;
        cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;
        cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;
        mov.b32 $0, {lo, hi};
        }""",
        "=r,f,f,f,f",
        [scaled],
        dtype=tl.float8e4nv,
        is_pure=True,
        pack=4,
    )
    tl.store(
        target + target_offset,
        converted,
        mask=mask,
    )


def write_sage_fp8_slab(
    target: torch.Tensor,
    source: torch.Tensor,
    value_absmax: torch.Tensor,
    *,
    start: int,
    layout: str,
    source_layout: str = "NHD",
    division_mode: int = 0,
    pad_to_target: bool = False,
) -> None:
    """Write one contiguous NHD or HND source slab into Sage storage."""

    if source_layout == "NHD":
        rows, heads, head_dim = map(int, source.shape)
    elif source_layout == "HND":
        heads, rows, head_dim = map(int, source.shape)
    else:
        raise ValueError("FP8 source layout must be NHD or HND")
    if not source.is_contiguous():
        raise ValueError("FP8 source slab must be contiguous")
    target_tokens = int(target.shape[-1])
    if pad_to_target:
        if start != 0 or target_tokens < rows or target_tokens % 16:
            raise ValueError("full FP8 padding requires an aligned whole target")
        padded_rows = target_tokens
    else:
        padded_rows = (rows + 15) // 16 * 16
    if start % 16 or start + padded_rows > target_tokens:
        raise ValueError("FP8 slab does not fit its aligned target")
    grid = (triton.cdiv(padded_rows * heads * head_dim, 256),)
    _write_sage_fp8_kernel[grid](
        source,
        target,
        value_absmax,
        rows=rows,
        heads=heads,
        head_dim=head_dim,
        padded_rows=padded_rows,
        target_tokens=target_tokens,
        target_start=int(start),
        source_layout_hnd=source_layout == "HND",
        layout_hnd=layout == "HND",
        division_mode=int(division_mode),
        block=256,
    )


def prepare_sage_fp8_hnd_direct(
    value_hnd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize contiguous HND V directly into Sage's permuted FP8 ABI.

    Sage's established helper first materializes another full FP16
    ``[B,H,D,N]`` tensor and then scans it again for per-channel scaling and
    FP8 conversion.  Min/max is order-independent for finite FP16 values, so
    reducing the original HND tensor and writing the final permuted bytes in
    one pass preserves the consumed representation while removing that large
    intermediate allocation and transpose.
    """

    if (
        value_hnd.ndim != 4
        or value_hnd.shape[0] != 1
        or value_hnd.shape[-1] != 128
        or not value_hnd.is_contiguous()
    ):
        raise ValueError("direct Sage FP8 preparation requires contiguous [1,H,N,128]")
    if value_hnd.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("direct Sage FP8 preparation requires FP16 or BF16")
    _, heads, rows, head_dim = map(int, value_hnd.shape)
    minimum, maximum = torch.aminmax(value_hnd[0], dim=1)
    value_absmax = torch.maximum(minimum.abs(), maximum.abs()).float()
    padded_rows = (rows + 127) // 128 * 128
    value_fp8 = torch.empty(
        (1, heads, head_dim, padded_rows),
        device=value_hnd.device,
        dtype=torch.float8_e4m3fn,
    )
    row_block = 64
    dim_block = 32
    _write_sage_fp8_hnd_full_kernel[
        (heads, triton.cdiv(padded_rows, row_block), head_dim // dim_block)
    ](
        value_hnd[0],
        value_fp8,
        value_absmax,
        rows=rows,
        padded_rows=padded_rows,
        head_dim=head_dim,
        row_block=row_block,
        dim_block=dim_block,
        num_warps=8,
    )
    value_scale = (value_absmax / 2.25).reshape(1, heads, head_dim)
    return value_fp8, value_scale


__all__ = ["prepare_sage_fp8_hnd_direct", "write_sage_fp8_slab"]
