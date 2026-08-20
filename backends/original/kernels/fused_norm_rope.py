"""High-precision-intent Triton fusions for MiniMax H3 on SM89."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from comfy_kitchen._rope_utils import check_rope_inplace, detect_rms_rope_bnhd


@triton.jit
def _segmented_rms_adaln_kernel(
    x_ptr, weight_ptr, scale_ptr, shift_ptr, out_ptr,
    row_start, d,
    stride_x_row, stride_out_row,
    eps,
    block_d: tl.constexpr,
    norm_dtype: tl.constexpr,
):
    row = row_start + tl.program_id(0)
    x_base = x_ptr + row * stride_x_row
    out_base = out_ptr + row * stride_out_row
    sumsq = tl.zeros([block_d], dtype=tl.float32)
    for off in range(0, d, block_d):
        cols = off + tl.arange(0, block_d)
        mask = cols < d
        value = tl.load(x_base + cols, mask=mask, other=0.0).to(tl.float32)
        sumsq += value * value
    inv_rms = tl.rsqrt(tl.sum(sumsq) / d + eps)
    for off in range(0, d, block_d):
        cols = off + tl.arange(0, block_d)
        mask = cols < d
        value = tl.load(x_base + cols, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        # Preserve the three BF16 materialization boundaries of the reference:
        # RMSNorm(weight), in-place multiply, then in-place add.
        normalized = (value * inv_rms * weight).to(norm_dtype).to(tl.float32)
        one_plus_scale = (1.0 + scale).to(norm_dtype).to(tl.float32)
        modulated = (normalized * one_plus_scale).to(norm_dtype).to(tl.float32)
        result = (modulated + shift.to(norm_dtype).to(tl.float32)).to(norm_dtype)
        tl.store(out_base + cols, result, mask=mask)


@triton.jit
def _partial_rms_rope_kernel(
    x_ptr, freqs_ptr, scale_ptr, out_ptr,
    head_dim, rot_dim, freqs_batch, freqs_seq,
    stride_x_batch, stride_x_head, stride_x_seq, stride_x_dim,
    stride_out_batch, stride_out_head, stride_out_seq, stride_out_dim,
    stride_freqs_batch, stride_freqs_seq, stride_freqs_dim,
    stride_freqs_rot, stride_freqs_pair,
    epsilon,
    compute_dtype: tl.constexpr,
    norm_dtype: tl.constexpr,
    block_full: tl.constexpr,
    block_pairs: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)
    x_base = x_ptr + batch_idx * stride_x_batch + head_idx * stride_x_head + seq_idx * stride_x_seq
    out_base = out_ptr + batch_idx * stride_out_batch + head_idx * stride_out_head + seq_idx * stride_out_seq

    full_offsets = tl.arange(0, block_full)
    full_mask = full_offsets < head_dim
    x_full = tl.load(x_base + full_offsets * stride_x_dim, mask=full_mask, other=0.0).to(tl.float32)
    scale_full = tl.load(scale_ptr + full_offsets, mask=full_mask, other=0.0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x_full * x_full) / head_dim + epsilon)
    normalized_full = (x_full * inv_rms * scale_full).to(norm_dtype)

    # The suffix is RMS-normalized but intentionally not rotated.
    suffix_mask = (full_offsets >= rot_dim) & full_mask
    tl.store(out_base + full_offsets * stride_out_dim, normalized_full, mask=suffix_mask)

    pair = tl.arange(0, block_pairs)
    pair_count = rot_dim // 2
    pair_mask = pair < pair_count
    dim0 = pair
    dim1 = pair + pair_count
    x0 = tl.load(x_base + dim0 * stride_x_dim, mask=pair_mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_base + dim1 * stride_x_dim, mask=pair_mask, other=0.0).to(tl.float32)
    s0 = tl.load(scale_ptr + dim0, mask=pair_mask, other=0.0).to(tl.float32)
    s1 = tl.load(scale_ptr + dim1, mask=pair_mask, other=0.0).to(tl.float32)
    x0 = (x0 * inv_rms * s0).to(norm_dtype).to(compute_dtype)
    x1 = (x1 * inv_rms * s1).to(norm_dtype).to(compute_dtype)

    freq_b = tl.where(freqs_batch == 1, 0, batch_idx)
    freq_s = tl.where(freqs_seq == 1, 0, seq_idx)
    freq_base = freqs_ptr + freq_b * stride_freqs_batch + freq_s * stride_freqs_seq + pair * stride_freqs_dim
    f00 = tl.load(freq_base, mask=pair_mask, other=0.0).to(compute_dtype)
    f01 = tl.load(freq_base + stride_freqs_pair, mask=pair_mask, other=0.0).to(compute_dtype)
    f10 = tl.load(freq_base + stride_freqs_rot, mask=pair_mask, other=0.0).to(compute_dtype)
    f11 = tl.load(freq_base + stride_freqs_rot + stride_freqs_pair, mask=pair_mask, other=0.0).to(compute_dtype)

    # Match eager BF16 multiply materialization before the final add.
    p00 = (f00 * x0).to(norm_dtype).to(compute_dtype)
    p01 = (f01 * x1).to(norm_dtype).to(compute_dtype)
    p10 = (f10 * x0).to(norm_dtype).to(compute_dtype)
    p11 = (f11 * x1).to(norm_dtype).to(compute_dtype)
    out0 = (p00 + p01).to(norm_dtype)
    out1 = (p10 + p11).to(norm_dtype)
    tl.store(out_base + dim0 * stride_out_dim, out0, mask=pair_mask)
    tl.store(out_base + dim1 * stride_out_dim, out1, mask=pair_mask)


@triton.jit
def _segmented_adaln_kernel(
    h_ptr, scale_ptr, shift_ptr,
    row_start, d, stride_h_row,
    block_d: tl.constexpr,
    value_dtype: tl.constexpr,
):
    """Modulate an already native-RMS-normalized tensor in place."""
    row = row_start + tl.program_id(0)
    base = h_ptr + row * stride_h_row
    for off in range(0, d, block_d):
        cols = off + tl.arange(0, block_d)
        mask = cols < d
        value = tl.load(base + cols, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        one_plus_scale = (1.0 + scale).to(value_dtype).to(tl.float32)
        value = (value * one_plus_scale).to(value_dtype).to(tl.float32)
        value = (value + shift.to(value_dtype).to(tl.float32)).to(value_dtype)
        tl.store(base + cols, value, mask=mask)


@triton.jit
def _three_segment_scale_kernel(
    h_ptr, scale_ptr,
    rows, d, stop0, stop1, row0, row1, row2,
    stride_h_row, stride_scale_row,
    block_d: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    cols = chunk * block_d + tl.arange(0, block_d)
    mask = (row < rows) & (cols < d)
    mod_row = tl.where(row < stop0, row0, tl.where(row < stop1, row1, row2))
    value = tl.load(h_ptr + row * stride_h_row + cols, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + mod_row * stride_scale_row + cols, mask=mask, other=0.0)
    # A separate store is an intentional BF16/FP16 rounding barrier.
    tl.store(h_ptr + row * stride_h_row + cols, value * (1.0 + scale), mask=mask)


@triton.jit
def _three_segment_shift_kernel(
    h_ptr, shift_ptr,
    rows, d, stop0, stop1, row0, row1, row2,
    stride_h_row, stride_shift_row,
    block_d: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    cols = chunk * block_d + tl.arange(0, block_d)
    mask = (row < rows) & (cols < d)
    mod_row = tl.where(row < stop0, row0, tl.where(row < stop1, row1, row2))
    value = tl.load(h_ptr + row * stride_h_row + cols, mask=mask, other=0.0)
    shift = tl.load(shift_ptr + mod_row * stride_shift_row + cols, mask=mask, other=0.0)
    tl.store(h_ptr + row * stride_h_row + cols, value + shift, mask=mask)


@triton.jit
def _three_segment_adaln_kernel(
    h_ptr, scale_ptr, shift_ptr,
    rows, d, stop0, stop1, row0, row1, row2,
    stride_h_row, stride_scale_row, stride_shift_row,
    block_d: tl.constexpr,
    value_dtype: tl.constexpr,
):
    """One traversal while retaining eager's BF16/FP16 multiply rounding."""
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    cols = chunk * block_d + tl.arange(0, block_d)
    mask = (row < rows) & (cols < d)
    mod_row = tl.where(row < stop0, row0, tl.where(row < stop1, row1, row2))
    value = tl.load(
        h_ptr + row * stride_h_row + cols, mask=mask, other=0.0
    )
    scale = tl.load(
        scale_ptr + mod_row * stride_scale_row + cols,
        mask=mask,
        other=0.0,
    )
    shift = tl.load(
        shift_ptr + mod_row * stride_shift_row + cols,
        mask=mask,
        other=0.0,
    )
    # Materialize exactly the same value-dtype rounding boundary as the
    # separate scale kernel before applying the shift.
    # A real global-memory round trip is required for bitwise parity with the
    # accepted two-kernel implementation; a register dtype cast is optimized
    # differently by Triton on SM89.  This still removes one host launch.
    address = h_ptr + row * stride_h_row + cols
    tl.store(address, value * (1.0 + scale), mask=mask)
    tl.debug_barrier()
    modulated = tl.load(address, mask=mask, other=0.0)
    tl.store(address, modulated + shift, mask=mask)


@triton.jit
def _partial_rope_from_normalized_kernel(
    normalized_ptr, freqs_ptr, out_ptr,
    head_dim, rot_dim, freqs_batch, freqs_seq,
    stride_x_batch, stride_x_head, stride_x_seq, stride_x_dim,
    stride_out_batch, stride_out_head, stride_out_seq, stride_out_dim,
    stride_freqs_batch, stride_freqs_seq, stride_freqs_dim,
    stride_freqs_rot, stride_freqs_pair,
    compute_dtype: tl.constexpr,
    value_dtype: tl.constexpr,
    block_full: tl.constexpr,
    block_pairs: tl.constexpr,
):
    """Rotate a native RMSNorm result while writing directly to packed Q/K."""
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)
    x_base = normalized_ptr + batch_idx * stride_x_batch + head_idx * stride_x_head + seq_idx * stride_x_seq
    out_base = out_ptr + batch_idx * stride_out_batch + head_idx * stride_out_head + seq_idx * stride_out_seq

    full_offsets = tl.arange(0, block_full)
    suffix_mask = (full_offsets >= rot_dim) & (full_offsets < head_dim)
    suffix = tl.load(x_base + full_offsets * stride_x_dim, mask=suffix_mask, other=0.0)
    tl.store(out_base + full_offsets * stride_out_dim, suffix, mask=suffix_mask)

    pair = tl.arange(0, block_pairs)
    pair_count = rot_dim // 2
    pair_mask = pair < pair_count
    dim0 = pair
    dim1 = pair + pair_count
    x0 = tl.load(x_base + dim0 * stride_x_dim, mask=pair_mask, other=0.0).to(compute_dtype)
    x1 = tl.load(x_base + dim1 * stride_x_dim, mask=pair_mask, other=0.0).to(compute_dtype)
    freq_b = tl.where(freqs_batch == 1, 0, batch_idx)
    freq_s = tl.where(freqs_seq == 1, 0, seq_idx)
    freq_base = freqs_ptr + freq_b * stride_freqs_batch + freq_s * stride_freqs_seq + pair * stride_freqs_dim
    f00 = tl.load(freq_base, mask=pair_mask, other=0.0).to(compute_dtype)
    f01 = tl.load(freq_base + stride_freqs_pair, mask=pair_mask, other=0.0).to(compute_dtype)
    f10 = tl.load(freq_base + stride_freqs_rot, mask=pair_mask, other=0.0).to(compute_dtype)
    f11 = tl.load(freq_base + stride_freqs_rot + stride_freqs_pair, mask=pair_mask, other=0.0).to(compute_dtype)
    p00 = (f00 * x0).to(value_dtype).to(compute_dtype)
    p01 = (f01 * x1).to(value_dtype).to(compute_dtype)
    p10 = (f10 * x0).to(value_dtype).to(compute_dtype)
    p11 = (f11 * x1).to(value_dtype).to(compute_dtype)
    tl.store(out_base + dim0 * stride_out_dim, (p00 + p01).to(value_dtype), mask=pair_mask)
    tl.store(out_base + dim1 * stride_out_dim, (p10 + p11).to(value_dtype), mask=pair_mask)


@triton.jit
def _partial_rope_products_kernel(
    normalized_ptr, freqs_ptr, products_ptr,
    rot_dim, freqs_batch, freqs_seq,
    stride_x_batch, stride_x_head, stride_x_seq, stride_x_dim,
    stride_freqs_batch, stride_freqs_seq, stride_freqs_dim,
    stride_freqs_rot, stride_freqs_pair,
    compute_dtype: tl.constexpr,
    value_dtype: tl.constexpr,
    block_pairs: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)
    pair = tl.arange(0, block_pairs)
    pair_count = rot_dim // 2
    mask = pair < pair_count
    x_base = normalized_ptr + batch_idx * stride_x_batch + head_idx * stride_x_head + seq_idx * stride_x_seq
    x0 = tl.load(x_base + pair * stride_x_dim, mask=mask, other=0.0).to(compute_dtype)
    x1 = tl.load(x_base + (pair + pair_count) * stride_x_dim, mask=mask, other=0.0).to(compute_dtype)
    freq_b = tl.where(freqs_batch == 1, 0, batch_idx)
    freq_s = tl.where(freqs_seq == 1, 0, seq_idx)
    freq_base = freqs_ptr + freq_b * stride_freqs_batch + freq_s * stride_freqs_seq + pair * stride_freqs_dim
    f00 = tl.load(freq_base, mask=mask, other=0.0).to(compute_dtype)
    f01 = tl.load(freq_base + stride_freqs_pair, mask=mask, other=0.0).to(compute_dtype)
    f10 = tl.load(freq_base + stride_freqs_rot, mask=mask, other=0.0).to(compute_dtype)
    f11 = tl.load(freq_base + stride_freqs_rot + stride_freqs_pair, mask=mask, other=0.0).to(compute_dtype)
    linear = ((batch_idx * tl.num_programs(1) + head_idx) * tl.num_programs(2) + seq_idx) * pair_count * 4
    # Stores reproduce eager's four BF16 multiplication temporaries.
    tl.store(products_ptr + linear + pair * 4 + 0, (f00 * x0).to(value_dtype), mask=mask)
    tl.store(products_ptr + linear + pair * 4 + 1, (f01 * x1).to(value_dtype), mask=mask)
    tl.store(products_ptr + linear + pair * 4 + 2, (f10 * x0).to(value_dtype), mask=mask)
    tl.store(products_ptr + linear + pair * 4 + 3, (f11 * x1).to(value_dtype), mask=mask)


@triton.jit
def _partial_rope_sum_copy_kernel(
    normalized_ptr, products_ptr, out_ptr,
    head_dim, rot_dim,
    stride_x_batch, stride_x_head, stride_x_seq, stride_x_dim,
    stride_out_batch, stride_out_head, stride_out_seq, stride_out_dim,
    compute_dtype: tl.constexpr,
    value_dtype: tl.constexpr,
    block_full: tl.constexpr,
    block_pairs: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)
    x_base = normalized_ptr + batch_idx * stride_x_batch + head_idx * stride_x_head + seq_idx * stride_x_seq
    out_base = out_ptr + batch_idx * stride_out_batch + head_idx * stride_out_head + seq_idx * stride_out_seq
    full_offsets = tl.arange(0, block_full)
    suffix_mask = (full_offsets >= rot_dim) & (full_offsets < head_dim)
    suffix = tl.load(x_base + full_offsets * stride_x_dim, mask=suffix_mask, other=0.0)
    tl.store(out_base + full_offsets * stride_out_dim, suffix, mask=suffix_mask)
    pair = tl.arange(0, block_pairs)
    pair_count = rot_dim // 2
    mask = pair < pair_count
    linear = ((batch_idx * tl.num_programs(1) + head_idx) * tl.num_programs(2) + seq_idx) * pair_count * 4
    p00 = tl.load(products_ptr + linear + pair * 4 + 0, mask=mask, other=0.0).to(compute_dtype)
    p01 = tl.load(products_ptr + linear + pair * 4 + 1, mask=mask, other=0.0).to(compute_dtype)
    p10 = tl.load(products_ptr + linear + pair * 4 + 2, mask=mask, other=0.0).to(compute_dtype)
    p11 = tl.load(products_ptr + linear + pair * 4 + 3, mask=mask, other=0.0).to(compute_dtype)
    tl.store(out_base + pair * stride_out_dim, (p00 + p01).to(value_dtype), mask=mask)
    tl.store(out_base + (pair + pair_count) * stride_out_dim, (p10 + p11).to(value_dtype), mask=mask)


def segmented_rms_adaln(x, norm, shift, scale, segments):
    if x.ndim != 2 or x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("production segmented RMS-AdaLN expects a 2D FP16/BF16 CUDA tensor")
    if norm.weight.device == x.device and norm.weight.dtype == x.dtype:
        weight = norm.weight.contiguous()
    else:
        import comfy.model_management
        weight = comfy.model_management.cast_to(norm.weight, device=x.device, dtype=x.dtype).contiguous()
    out = torch.empty_like(x)
    d = x.shape[-1]
    dtype_map = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}
    previous = 0
    for start, stop, row in segments:
        if start != previous or stop <= start:
            raise ValueError("production modulation segments must form one ordered contiguous partition")
        segment_scale = scale[row].to(device=x.device, dtype=x.dtype).contiguous()
        segment_shift = shift[row].to(device=x.device, dtype=x.dtype).contiguous()
        _segmented_rms_adaln_kernel[(stop - start,)](
            x, weight, segment_scale, segment_shift, out,
            start, d, x.stride(0), out.stride(0), norm.eps,
            block_d=min(triton.next_power_of_2(d), 4096),
            norm_dtype=dtype_map[x.dtype],
        )
        previous = stop
    if previous != x.shape[0]:
        raise ValueError("production modulation segments do not cover the input")
    return out


def native_rms_segmented_adaln(x, norm, shift, scale, segments):
    """High-precision path: preserve native RMSNorm, fuse only scale/shift."""
    h = norm(x)
    if h.ndim != 2 or h.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("production segmented AdaLN expects a 2D FP16/BF16 CUDA tensor")
    d = h.shape[-1]
    dtype_map = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}
    previous = 0
    for start, stop, row in segments:
        if start != previous or stop <= start:
            raise ValueError("production modulation segments must form one ordered contiguous partition")
        segment_scale = scale[row].to(device=h.device, dtype=h.dtype).contiguous()
        segment_shift = shift[row].to(device=h.device, dtype=h.dtype).contiguous()
        _segmented_adaln_kernel[(stop - start,)](
            h, segment_scale, segment_shift,
            start, d, h.stride(0),
            block_d=min(triton.next_power_of_2(d), 4096),
            value_dtype=dtype_map[h.dtype],
        )
        previous = stop
    if previous != h.shape[0]:
        raise ValueError("production modulation segments do not cover the input")
    return h


def native_rms_two_pass_adaln(x, norm, shift, scale, segments):
    """Native RMSNorm plus two rounding-preserving full-sequence modulation passes."""
    if len(segments) != 3:
        # Non-T2V layouts keep the exact upstream implementation until separately validated.
        h = norm(x)
        for start, stop, row in segments:
            h[start:stop].mul_(1.0 + scale[row].to(h.dtype)).add_(shift[row].to(h.dtype))
        return h
    (start0, stop0, row0), (start1, stop1, row1), (start2, stop2, row2) = segments
    if start0 != 0 or start1 != stop0 or start2 != stop1 or stop2 != x.shape[0]:
        raise ValueError("production modulation segments must form one ordered contiguous partition")
    h = norm(x)
    scale_local = scale.to(device=h.device, dtype=h.dtype).contiguous()
    shift_local = shift.to(device=h.device, dtype=h.dtype).contiguous()
    block_d = 256
    grid = (h.shape[0], triton.cdiv(h.shape[1], block_d))
    _three_segment_scale_kernel[grid](
        h, scale_local, h.shape[0], h.shape[1], stop0, stop1, row0, row1, row2,
        h.stride(0), scale_local.stride(0), block_d=block_d)
    _three_segment_shift_kernel[grid](
        h, shift_local, h.shape[0], h.shape[1], stop0, stop1, row0, row1, row2,
        h.stride(0), shift_local.stride(0), block_d=block_d)
    return h


def native_rms_one_pass_adaln(x, norm, shift, scale, segments):
    """Native RMSNorm plus one bitwise-screened segmented modulation pass."""
    if len(segments) != 3:
        return native_rms_two_pass_adaln(x, norm, shift, scale, segments)
    (start0, stop0, row0), (start1, stop1, row1), (start2, stop2, row2) = segments
    if start0 != 0 or start1 != stop0 or start2 != stop1 or stop2 != x.shape[0]:
        raise ValueError("production modulation segments must form one ordered contiguous partition")
    h = norm(x)
    scale_local = scale.to(device=h.device, dtype=h.dtype).contiguous()
    shift_local = shift.to(device=h.device, dtype=h.dtype).contiguous()
    block_d = 256
    grid = (h.shape[0], triton.cdiv(h.shape[1], block_d))
    dtype_map = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}
    _three_segment_adaln_kernel[grid](
        h,
        scale_local,
        shift_local,
        h.shape[0],
        h.shape[1],
        stop0,
        stop1,
        row0,
        row1,
        row2,
        h.stride(0),
        scale_local.stride(0),
        shift_local.stride(0),
        block_d=block_d,
        value_dtype=dtype_map[h.dtype],
    )
    return h


def _partial_one(x, freqs, scale, epsilon, rot_dim):
    bnhd = detect_rms_rope_bnhd(x, freqs, rot_dim=rot_dim)
    if bnhd is None:
        raise ValueError("production partial RMS-RoPE frequencies are not broadcastable to input")
    batch, dim1, dim2, head_dim = x.shape
    if bnhd:
        heads, sequence = dim2, dim1
        stride_head, stride_seq = x.stride(2), x.stride(1)
        freq_sequence_stride = freqs.stride(1)
        out_head_stride, out_seq_stride = x.stride(2), x.stride(1)
    else:
        heads, sequence = dim1, dim2
        stride_head, stride_seq = x.stride(1), x.stride(2)
        freq_sequence_stride = freqs.stride(2)
        out_head_stride, out_seq_stride = x.stride(1), x.stride(2)
    dtype_map = {torch.float32: tl.float32, torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}
    _partial_rms_rope_kernel[(batch, heads, sequence)](
        x, freqs, scale.contiguous(), x,
        head_dim, rot_dim, freqs.shape[0], freqs.shape[1 if bnhd else 2],
        x.stride(0), stride_head, stride_seq, x.stride(3),
        x.stride(0), out_head_stride, out_seq_stride, x.stride(3),
        freqs.stride(0), freq_sequence_stride, freqs.stride(3), freqs.stride(4), freqs.stride(5),
        epsilon,
        compute_dtype=dtype_map.get(freqs.dtype, tl.float32),
        norm_dtype=dtype_map.get(x.dtype, tl.float32),
        block_full=triton.next_power_of_2(head_dim),
        block_pairs=triton.next_power_of_2(rot_dim // 2),
    )
    return x


def make_partial_rms_rope(original):
    def partial_rms_rope(q, k, freqs, q_scale, k_scale=None, epsilon=1e-6, rot_dim=0):
        if k_scale is None:
            k_scale = q_scale
        target = (
            q.is_cuda and k.is_cuda and q.ndim == k.ndim == 4
            and q.shape[-1] == k.shape[-1] == 128 and rot_dim == 96
            and q.dtype in (torch.float16, torch.bfloat16) and k.dtype == q.dtype
        )
        if not target:
            return original(q, k, freqs, q_scale, k_scale, epsilon, rot_dim=rot_dim)
        check_rope_inplace(q, k, readonly=(freqs, q_scale, k_scale))
        _partial_one(q, freqs, q_scale, epsilon, rot_dim)
        _partial_one(k, freqs, k_scale, epsilon, rot_dim)
        return q, k
    partial_rms_rope.__name__ = "partial_rms_rope_split_half_"
    partial_rms_rope._h3_partial_rope = True
    return partial_rms_rope


def _partial_rotate_from_normalized(normalized, out, freqs, rot_dim):
    bnhd = detect_rms_rope_bnhd(normalized, freqs, rot_dim=rot_dim)
    if bnhd is None:
        raise ValueError("production partial RoPE frequencies are not broadcastable to input")
    batch, dim1, dim2, head_dim = normalized.shape
    if bnhd:
        heads, sequence = dim2, dim1
        x_head_stride, x_seq_stride = normalized.stride(2), normalized.stride(1)
        out_head_stride, out_seq_stride = out.stride(2), out.stride(1)
        freq_sequence_stride = freqs.stride(1)
    else:
        heads, sequence = dim1, dim2
        x_head_stride, x_seq_stride = normalized.stride(1), normalized.stride(2)
        out_head_stride, out_seq_stride = out.stride(1), out.stride(2)
        freq_sequence_stride = freqs.stride(2)
    dtype_map = {torch.float32: tl.float32, torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}
    _partial_rope_from_normalized_kernel[(batch, heads, sequence)](
        normalized, freqs, out,
        head_dim, rot_dim, freqs.shape[0], freqs.shape[1 if bnhd else 2],
        normalized.stride(0), x_head_stride, x_seq_stride, normalized.stride(3),
        out.stride(0), out_head_stride, out_seq_stride, out.stride(3),
        freqs.stride(0), freq_sequence_stride, freqs.stride(3), freqs.stride(4), freqs.stride(5),
        compute_dtype=dtype_map.get(freqs.dtype, tl.float32),
        value_dtype=dtype_map.get(normalized.dtype, tl.float32),
        block_full=triton.next_power_of_2(head_dim),
        block_pairs=triton.next_power_of_2(rot_dim // 2),
    )


def _partial_rotate_exact(normalized, out, freqs, rot_dim):
    bnhd = detect_rms_rope_bnhd(normalized, freqs, rot_dim=rot_dim)
    if bnhd is None:
        raise ValueError("production partial RoPE frequencies are not broadcastable to input")
    batch, dim1, dim2, head_dim = normalized.shape
    if bnhd:
        heads, sequence = dim2, dim1
        x_head_stride, x_seq_stride = normalized.stride(2), normalized.stride(1)
        out_head_stride, out_seq_stride = out.stride(2), out.stride(1)
        freq_sequence_stride = freqs.stride(1)
    else:
        heads, sequence = dim1, dim2
        x_head_stride, x_seq_stride = normalized.stride(1), normalized.stride(2)
        out_head_stride, out_seq_stride = out.stride(1), out.stride(2)
        freq_sequence_stride = freqs.stride(2)
    dtype_map = {torch.float32: tl.float32, torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}
    pair_count = rot_dim // 2
    products = torch.empty(
        (batch, heads, sequence, pair_count, 4), device=normalized.device, dtype=normalized.dtype)
    grid = (batch, heads, sequence)
    _partial_rope_products_kernel[grid](
        normalized, freqs, products,
        rot_dim, freqs.shape[0], freqs.shape[1 if bnhd else 2],
        normalized.stride(0), x_head_stride, x_seq_stride, normalized.stride(3),
        freqs.stride(0), freq_sequence_stride, freqs.stride(3), freqs.stride(4), freqs.stride(5),
        compute_dtype=dtype_map.get(freqs.dtype, tl.float32),
        value_dtype=dtype_map.get(normalized.dtype, tl.float32),
        block_pairs=triton.next_power_of_2(pair_count))
    _partial_rope_sum_copy_kernel[grid](
        normalized, products, out, head_dim, rot_dim,
        normalized.stride(0), x_head_stride, x_seq_stride, normalized.stride(3),
        out.stride(0), out_head_stride, out_seq_stride, out.stride(3),
        compute_dtype=dtype_map.get(freqs.dtype, tl.float32),
        value_dtype=dtype_map.get(normalized.dtype, tl.float32),
        block_full=triton.next_power_of_2(head_dim),
        block_pairs=triton.next_power_of_2(pair_count))


def make_native_rms_partial_rope(original):
    """High-precision path: native RMSNorm followed by fused partial RoPE/copy."""
    def partial_rms_rope(q, k, freqs, q_scale, k_scale=None, epsilon=1e-6, rot_dim=0):
        if k_scale is None:
            k_scale = q_scale
        target = (
            q.is_cuda and k.is_cuda and q.ndim == k.ndim == 4
            and q.shape[-1] == k.shape[-1] == 128 and rot_dim == 96
            and q.dtype in (torch.float16, torch.bfloat16) and k.dtype == q.dtype
        )
        if not target:
            return original(q, k, freqs, q_scale, k_scale, epsilon, rot_dim=rot_dim)
        check_rope_inplace(q, k, readonly=(freqs, q_scale, k_scale))
        q_normalized = torch.nn.functional.rms_norm(q, (q.shape[-1],), weight=q_scale, eps=epsilon)
        k_normalized = torch.nn.functional.rms_norm(k, (k.shape[-1],), weight=k_scale, eps=epsilon)
        _partial_rotate_exact(q_normalized, q, freqs, rot_dim)
        _partial_rotate_exact(k_normalized, k, freqs, rot_dim)
        return q, k
    partial_rms_rope.__name__ = "native_rms_partial_rope_split_half_"
    partial_rms_rope._h3_partial_rope = True
    return partial_rms_rope


def install(
    minimax_module, *, enable_rms_adaln: bool, enable_partial_rope: bool,
    aggressive_fused_rms: bool = False,
) -> dict[str, bool]:
    if enable_partial_rope:
        ck = minimax_module.comfy.quant_ops.ck
        factory = make_partial_rms_rope if aggressive_fused_rms else make_native_rms_partial_rope
        ck.rms_rope_split_half_ = factory(ck.rms_rope_split_half_)

    if enable_rms_adaln:
        def dit_forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
            fused_norm = segmented_rms_adaln if aggressive_fused_rms else native_rms_two_pass_adaln
            h = fused_norm(x, self.norm1, shift_msa, scale_msa, mod_segments)
            x = minimax_module._mod_gate(
                x, gate_msa,
                self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options),
                mod_segments,
            )
            h = fused_norm(x, self.norm2, shift_mlp, scale_mlp, mod_segments)
            return minimax_module._mod_gate(x, gate_mlp, self.mlp(h), mod_segments)

        dit_forward.__name__ = "dit_forward_segmented_rms_adaln"
        dit_forward._h3_rms_adaln = True
        minimax_module.DiTBlock.forward = dit_forward

    return {
        "segmented_rms_adaln": enable_rms_adaln,
        "partial_rms_rope": enable_partial_rope,
        "aggressive_fused_rms": aggressive_fused_rms,
    }
