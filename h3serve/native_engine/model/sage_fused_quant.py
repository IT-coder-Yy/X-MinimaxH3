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
        value0 = tl.load(pointers0, mask=rows0[:, None] < length)
        value1 = tl.load(pointers1, mask=rows1[:, None] < length)
        # Reference path is ``centered = k - k.mean(...)`` where both K and
        # mean are BF16 and the centered tensor is materialized as BF16 before
        # the upstream quantizer casts to FP32.  Preserve that store boundary
        # explicitly; an FP32 subtract here is observably different.
        centered0 = (value0 - mean[None, :]).to(tl.bfloat16).to(tl.float32)
        centered1 = (value1 - mean[None, :]).to(tl.bfloat16).to(tl.float32)
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
        tl.store(output0, quantized0, mask=rows0[:, None] < length)
        tl.store(output1, quantized1, mask=rows1[:, None] < length)
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


__all__ = ["quantize_key_sub_mean_per_thread_int8"]
