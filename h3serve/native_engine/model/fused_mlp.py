"""RTX 4090 fused INT8 H3 MLP execution.

This is the standalone-native adaptation of the project-owned V6.6 kernel.
It preserves the accepted inference boundary:

``fc1 INT8 -> BF16 -> SwiGLU -> ConvRot -> row INT8 -> fc2 INT8 ->
BF16 -> gated residual``.

The down projection, dequantization, AdaLN gate, and residual update share one
Triton kernel.  Unlike the research implementation, production uses a fixed
SM89 launch configuration so a cold process does not run Triton autotuning.
Unsupported devices and LoRA-wrapped linears explicitly fall back to the
transparent PyTorch/module path in :mod:`layers`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .layers import ModulationSegment, SwiGLUMLP


_ROW_MAP_CACHE: dict[tuple[object, ...], torch.Tensor] = {}


def _enabled() -> bool:
    return os.environ.get("H3_NATIVE_DISABLE_FUSED_MLP", "0") != "1"


def _segment_row_map(
    length: int,
    segments: tuple["ModulationSegment", ...],
    device: torch.device,
) -> torch.Tensor:
    normalized = tuple((int(a), int(b), int(row)) for a, b, row in segments)
    key = (device.type, device.index, int(length), normalized)
    cached = _ROW_MAP_CACHE.get(key)
    if cached is not None:
        return cached
    host = torch.empty(length, dtype=torch.int32, device="cpu")
    previous = 0
    for start, stop, row in normalized:
        if start != previous or stop <= start:
            raise ValueError("modulation segments must be an ordered partition")
        host[start:stop] = row
        previous = stop
    if previous != length:
        raise ValueError("modulation segments do not cover the packed sequence")
    cached = host.to(device=device)
    _ROW_MAP_CACHE[key] = cached
    return cached


def _load_triton_kernel():
    # Keep CPU-only service imports independent of Triton.
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
        a_ptr,
        b_ptr,
        residual_ptr,
        a_scale_ptr,
        b_scale_ptr,
        bias_ptr,
        gate_ptr,
        gate_row_ptr,
        m,
        n: tl.constexpr,
        k: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_rm: tl.constexpr,
        stride_rn: tl.constexpr,
        stride_gm: tl.constexpr,
        stride_gn: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
        group_m: tl.constexpr,
        has_bias: tl.constexpr,
        per_channel_scale: tl.constexpr,
        output_dtype_code: tl.constexpr,
    ):
        pid = tl.program_id(0)
        grid_m = tl.cdiv(m, block_m)
        grid_n = tl.cdiv(n, block_n)
        group_width = group_m * grid_n
        group_id = pid // group_width
        first_m = group_id * group_m
        actual_group_m = tl.minimum(grid_m - first_m, group_m)
        pid_m = first_m + (pid % actual_group_m)
        pid_n = (pid % group_width) // actual_group_m

        offs_m = (pid_m * block_m + tl.arange(0, block_m)) % m
        offs_n = (pid_n * block_n + tl.arange(0, block_n)) % n
        offs_k = tl.arange(0, block_k)
        a = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        accumulator = tl.zeros((block_m, block_n), dtype=tl.int32)
        for k_index in range(0, tl.cdiv(k, block_k)):
            av = tl.load(a, mask=offs_k[None, :] < k - k_index * block_k, other=0)
            bv = tl.load(b, mask=offs_k[:, None] < k - k_index * block_k, other=0)
            accumulator += tl.dot(av, bv)
            a += block_k * stride_ak
            b += block_k * stride_bk

        a_scale = tl.load(a_scale_ptr + offs_m)
        if per_channel_scale:
            b_scale = tl.load(b_scale_ptr + offs_n)
        else:
            b_scale = tl.load(b_scale_ptr)
        projected = accumulator.to(tl.float32) * a_scale[:, None] * b_scale
        if has_bias:
            projected += tl.load(bias_ptr + offs_n)[None, :]

        # Match the accepted boundary: projection rounds to the activation
        # dtype before addcmul consumes it.
        if output_dtype_code == 1:
            projected = projected.to(tl.float16).to(tl.float32)
        else:
            projected = projected.to(tl.bfloat16).to(tl.float32)
        row_mask = offs_m < m
        gate_row = tl.load(gate_row_ptr + offs_m, mask=row_mask, other=0)
        gate = tl.load(
            gate_ptr + gate_row[:, None] * stride_gm + offs_n[None, :] * stride_gn
        )
        residual_addresses = (
            residual_ptr
            + offs_m[:, None] * stride_rm
            + offs_n[None, :] * stride_rn
        )
        valid = row_mask[:, None] & (offs_n[None, :] < n)
        residual = tl.load(residual_addresses, mask=valid, other=0.0).to(tl.float32)
        result = residual + projected * gate.to(tl.float32)
        tl.store(residual_addresses, result, mask=valid)

    return triton, kernel


_TRITON = None
_GATED_RESIDUAL_KERNEL = None


def _kernel_objects():
    global _TRITON, _GATED_RESIDUAL_KERNEL
    if _GATED_RESIDUAL_KERNEL is None:
        _TRITON, _GATED_RESIDUAL_KERNEL = _load_triton_kernel()
    return _TRITON, _GATED_RESIDUAL_KERNEL


def _fused_down_gated_residual(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    gate_rows: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> None:
    from backends.original.kernels.convrot_nvrtc import (
        fused_swiglu_convrot_row_quant,
    )

    qx, x_scale = fused_swiglu_convrot_row_quant(hidden.reshape(-1, hidden.shape[-1]))
    rows, inner = qx.shape
    features = qweight.shape[0]
    residual_2d = residual.reshape(rows, features)
    if weight_scale.numel() not in (1, features):
        raise ValueError("fc2 scale must be scalar or per output channel")
    per_channel = weight_scale.numel() != 1
    scale = (
        weight_scale.reshape(features).contiguous()
        if per_channel
        else weight_scale.reshape(1)
    )
    bias_ptr = hidden if bias is None else bias
    triton, kernel = _kernel_objects()
    block_m, block_n, block_k, group_m = 128, 256, 64, 8
    grid = (
        triton.cdiv(rows, block_m) * triton.cdiv(features, block_n),
    )
    kernel[grid](
        qx,
        qweight,
        residual_2d,
        x_scale,
        scale,
        bias_ptr,
        gate,
        gate_rows,
        m=rows,
        n=features,
        k=inner,
        stride_am=qx.stride(0),
        stride_ak=qx.stride(1),
        stride_bk=qweight.stride(1),
        stride_bn=qweight.stride(0),
        stride_rm=residual_2d.stride(0),
        stride_rn=residual_2d.stride(1),
        stride_gm=gate.stride(0),
        stride_gn=gate.stride(1),
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        group_m=group_m,
        has_bias=bias is not None,
        per_channel_scale=per_channel,
        output_dtype_code=1 if residual.dtype == torch.float16 else 2,
        num_stages=3,
        num_warps=8,
    )


def try_fused_gated_mlp(
    residual: torch.Tensor,
    hidden: torch.Tensor,
    mlp: "SwiGLUMLP",
    gate: torch.Tensor,
    segments: tuple["ModulationSegment", ...],
    *,
    chunk_tokens: int | None = None,
) -> bool:
    """Update ``residual`` in place and report whether the fast path ran."""

    if not (
        _enabled()
        and hidden.is_cuda
        and hidden.ndim == 2
        and hidden.dtype in (torch.float16, torch.bfloat16)
        and torch.cuda.get_device_capability(hidden.device) == (8, 9)
    ):
        return False
    # Base and Larry paths are handled separately; silently dropping a LoRA
    # contribution is forbidden.
    from .quantization import ConvRotInt8Linear
    from .lora import RuntimeLoRALinear

    fc1, fc2 = mlp.fc1, mlp.fc2
    lora_path = isinstance(fc1, RuntimeLoRALinear) and isinstance(
        fc2, RuntimeLoRALinear
    )
    base_fc1 = fc1.base if lora_path else fc1
    base_fc2 = fc2.base if lora_path else fc2
    if not isinstance(base_fc1, ConvRotInt8Linear) or not isinstance(
        base_fc2, ConvRotInt8Linear
    ):
        return False
    if (
        base_fc1.kernel is None
        or base_fc2.kernel is None
        or not base_fc2.spec.convrot
        or base_fc2.spec.convrot_groupsize != 256
    ):
        return False
    row_map = _segment_row_map(hidden.shape[0], segments, hidden.device)
    if chunk_tokens is None:
        # Backward-compatible process default. Production routing passes an
        # explicit immutable request-level value and never mutates env vars.
        chunk_env = (
            "H3_NATIVE_LORA_MLP_CHUNK_TOKENS"
            if lora_path
            else "H3_NATIVE_MLP_CHUNK_TOKENS"
        )
        chunk_default = "4096" if lora_path else "8192"
        chunk_tokens = int(os.environ.get(chunk_env, chunk_default))
    if chunk_tokens <= 0:
        raise ValueError("MLP chunk_tokens must be positive")
    for start in range(0, hidden.shape[0], chunk_tokens):
        stop = min(start + chunk_tokens, hidden.shape[0])
        raw = fc1(hidden[start:stop])
        from .research_capture import (
            capture_kind,
            capture_quantized_fc2_chunk,
            capture_target,
        )
        from .kernels import current_attention_layer

        capture = capture_target(current_attention_layer())
        if capture is not None and capture_kind() == "quantized_fc2":
            from comfy_kitchen.backends.cuda import (
                quantize_int8_rowwise_convrot64,
            )

            captured_qx, captured_scale = quantize_int8_rowwise_convrot64(
                raw, 256, input_act="swiglu"
            )
            capture_quantized_fc2_chunk(
                capture,
                qx=captured_qx,
                x_scale=captured_scale,
                qweight=base_fc2.qweight,
                weight_scale=base_fc2.scale,
                chunk_start=start,
                chunk_stop=stop,
                protected_tokens=segments[-1][0],
            )
            del captured_qx, captured_scale
        if (
            not lora_path
            and os.environ.get("H3_NATIVE_EXPERIMENTAL_MLP_EPILOGUE", "0") == "1"
        ):
            _fused_down_gated_residual(
                raw,
                residual[start:stop],
                gate,
                row_map[start:stop],
                base_fc2.qweight,
                base_fc2.scale,
                base_fc2.bias,
            )
            continue

        # Current Comfy-Kitchen's fused input-activation kernel is faster on
        # the installed CUDA 12.6/SM89 stack than the older project epilogue,
        # and exactly matches the optimized Comfy execution boundary.  The
        # release package vendors this low-level kernel library, not ComfyUI.
        from comfy_kitchen import int8_linear

        projected = int8_linear(
            raw,
            base_fc2.qweight,
            base_fc2.scale,
            base_fc2.bias,
            hidden.dtype,
            convrot=True,
            convrot_groupsize=256,
            input_act="swiglu",
        )
        if lora_path:
            # The distilled fc2 LoRA consumes the exact BF16 SwiGLU
            # activation.  Its rank-64 update is accumulated into the fresh
            # base result by addmm, while the INT8 base retains the faster
            # fused input-activation path above.
            gate_value, up_value = raw.chunk(2, dim=-1)
            activated = torch.nn.functional.silu(gate_value).mul_(up_value)
            projected = fc2.update.apply(activated, projected)
        chunk_residual = residual[start:stop]
        for segment_start, segment_stop, row in segments:
            overlap_start = max(start, segment_start)
            overlap_stop = min(stop, segment_stop)
            if overlap_start >= overlap_stop:
                continue
            local_start = overlap_start - start
            local_stop = overlap_stop - start
            chunk_residual[local_start:local_stop].addcmul_(
                projected[local_start:local_stop], gate[row].to(hidden.dtype)
            )
    return True


__all__ = ["try_fused_gated_mlp"]
