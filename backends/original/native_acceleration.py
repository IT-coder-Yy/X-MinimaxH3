"""production: correct fused SwiGLU + ConvRot + row-INT8 preprocessing."""

from __future__ import annotations

import torch

try:
    # Standalone release-package import.
    from .kernels.convrot_nvrtc import fused_swiglu_convrot_row_quant, warmup_module
except ImportError:
    # Compatibility with the historical launcher that places this directory
    # directly on ``sys.path``.
    from kernels.convrot_nvrtc import fused_swiglu_convrot_row_quant, warmup_module


def _matmul_from_quantized(
    x: torch.Tensor,
    qx: torch.Tensor,
    x_scale: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    from comfy_kitchen.backends.triton import quantization as ck_triton
    import triton

    rows, inner = qx.shape
    out_features = qweight.shape[0]
    output = torch.empty((rows, out_features), device=x.device, dtype=out_dtype)
    if not isinstance(weight_scale, torch.Tensor):
        weight_scale = torch.tensor([weight_scale], device=x.device, dtype=torch.float32)
    per_channel = weight_scale.numel() != 1
    weight_scale = (
        weight_scale.reshape(out_features).contiguous()
        if per_channel else weight_scale.reshape(1)
    )
    has_bias = bias is not None
    bias_ptr = bias if has_bias else x

    def grid(meta):
        return (
            triton.cdiv(rows, meta["block_m"])
            * triton.cdiv(out_features, meta["block_n"]),
        )

    kernel = (
        ck_triton._int8_matmul_dequant_per_row_kernel
        if per_channel else ck_triton._int8_matmul_dequant_kernel
    )
    kernel[grid](
        a_ptr=qx, b_ptr=qweight, c_ptr=output,
        a_scale_ptr=x_scale, b_scale_ptr=weight_scale, bias_ptr=bias_ptr,
        m=rows, n=out_features, k=inner,
        stride_am=qx.stride(0), stride_ak=qx.stride(1),
        stride_bk=qweight.stride(1), stride_bn=qweight.stride(0),
        stride_cm=output.stride(0), stride_cn=output.stride(1),
        has_bias=has_bias,
    )
    return output


def int8_linear_fused_swiglu_convrot(
    x: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    original = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    qx, x_scale = fused_swiglu_convrot_row_quant(x_2d)
    output = _matmul_from_quantized(
        x_2d, qx, x_scale, qweight, weight_scale, bias, out_dtype
    )
    return output.reshape(*original[:-1], output.shape[-1])


def resident_weight_chunked_gated_mlp(
    x, h, mlp, gate, segments, chunk_tokens, comfy_ops, ck, linear_kernels,
):
    fc1, fc2 = mlp.fc1, mlp.fc2
    for linear in (fc1, fc2):
        if getattr(linear, "pre_quant_scale", None) is not None:
            return None
    weight1, bias1 = comfy_ops.cast_bias_weight(
        fc1, h, offloadable=False, compute_dtype=h.dtype, want_requant=True
    )
    weight2, bias2 = comfy_ops.cast_bias_weight(
        fc2, h, offloadable=False, compute_dtype=h.dtype, want_requant=True
    )
    info1 = linear_kernels._plain_int8_weight(weight1)
    info2 = linear_kernels._plain_int8_weight(weight2)
    if info1 is None or info2 is None:
        del weight1, weight2
        return None
    qdata2, scale2, params2 = info2
    if not getattr(params2, "convrot", False) or getattr(params2, "convrot_groupsize", 256) != 256:
        del weight1, weight2
        return None

    segment_index = 0
    for start in range(0, h.shape[0], chunk_tokens):
        stop = min(start + chunk_tokens, h.shape[0])
        hidden = linear_kernels._int8_linear(ck, h[start:stop], info1, bias1, fc1)
        other = int8_linear_fused_swiglu_convrot(
            hidden, qdata2, scale2, bias2, h.dtype
        )
        while segment_index < len(segments) and segments[segment_index][1] <= start:
            segment_index += 1
        scan = segment_index
        while scan < len(segments):
            seg_start, seg_stop, row = segments[scan]
            if seg_start >= stop:
                break
            overlap_start = max(seg_start, start)
            overlap_stop = min(seg_stop, stop)
            if overlap_start < overlap_stop:
                x[overlap_start:overlap_stop].addcmul_(
                    other[overlap_start - start:overlap_stop - start],
                    gate[row].to(x.dtype),
                )
            scan += 1
        del hidden, other
    del weight1, weight2
    return x


def install(minimax_module, norm_kernels, mlp_kernels, linear_kernels, *, chunk_tokens: int):
    warmup_module()
    state = linear_kernels.install(
        minimax_module, norm_kernels, mlp_kernels,
        chunk_tokens=chunk_tokens,
        enable_chunked_mlp=True,
        enable_resident_mlp_weights=True,
    )
    comfy_ops = minimax_module.comfy.ops
    ck = minimax_module.comfy.quant_ops.ck

    def dit_forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        fused_norm = norm_kernels.native_rms_two_pass_adaln
        h = fused_norm(x, self.norm1, shift_msa, scale_msa, mod_segments)
        attention_output = self.attn(
            h, rope_freqs=rope_freqs, transformer_options=transformer_options
        )
        x = minimax_module._mod_gate(x, gate_msa, attention_output, mod_segments)
        del attention_output
        h = fused_norm(x, self.norm2, shift_mlp, scale_mlp, mod_segments)
        result = resident_weight_chunked_gated_mlp(
            x, h, self.mlp, gate_mlp, mod_segments, chunk_tokens,
            comfy_ops, ck, linear_kernels,
        )
        if result is not None:
            return result
        return mlp_kernels.chunked_gated_mlp(
            x, h, self.mlp, gate_mlp, mod_segments, chunk_tokens
        )

    dit_forward.__name__ = "production_dit_forward"
    minimax_module.DiTBlock.forward = dit_forward
    return {
        **state,
        "native_acceleration": "sm89_triton",
        "fused_swiglu_convrot_row_quant": True,
        "convrot_math_present": True,
        "nvrtc_runtime": "cuda12.6-driver-ptx",
    }
