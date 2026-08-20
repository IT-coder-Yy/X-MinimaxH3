"""Production INT8 hooks, initially anchored to the validated bounded-memory mathematical path."""

from __future__ import annotations

import torch


def _plain_int8_weight(weight):
    from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout

    if not isinstance(weight, QuantizedTensor) or weight._layout_cls != "TensorWiseINT8Layout":
        return None
    return (*TensorWiseINT8Layout.get_plain_tensors(weight), weight._params)


def _int8_linear(ck, x, weight_info, bias, linear, *, input_act=None):
    qdata, scale, params = weight_info
    return ck.int8_linear(
        x,
        qdata,
        scale,
        bias,
        x.dtype,
        convrot=getattr(params, "convrot", False),
        convrot_groupsize=getattr(params, "convrot_groupsize", 256),
        input_act=input_act,
    )


def resident_weight_chunked_gated_mlp(
    x: torch.Tensor,
    h: torch.Tensor,
    mlp,
    gate: torch.Tensor,
    segments,
    chunk_tokens: int,
    comfy_ops,
    ck,
) -> torch.Tensor:
    """Load both MLP weights once, then reuse them across all token chunks.

    bounded-memory calls the module for every chunk.  Under ComfyUI async offload that moves
    fc1/fc2 from host memory for every call (about 285 GB per 720p DiT step).
    Keeping the same quantized weights alive for the enclosing block removes the
    repeated transfers while invoking the identical comfy-kitchen INT8 kernels.
    """
    fc1, fc2 = mlp.fc1, mlp.fc2
    for linear in (fc1, fc2):
        if getattr(linear, "pre_quant_scale", None) is not None:
            return None

    weight1, bias1 = comfy_ops.cast_bias_weight(
        fc1, h, offloadable=False, compute_dtype=h.dtype, want_requant=True)
    weight2, bias2 = comfy_ops.cast_bias_weight(
        fc2, h, offloadable=False, compute_dtype=h.dtype, want_requant=True)
    info1 = _plain_int8_weight(weight1)
    info2 = _plain_int8_weight(weight2)
    if info1 is None or info2 is None:
        del weight1, weight2
        return None

    segment_index = 0
    for chunk_start in range(0, h.shape[0], chunk_tokens):
        chunk_stop = min(chunk_start + chunk_tokens, h.shape[0])
        hidden = _int8_linear(ck, h[chunk_start:chunk_stop], info1, bias1, fc1)
        other = _int8_linear(ck, hidden, info2, bias2, fc2, input_act="swiglu")

        while segment_index < len(segments) and segments[segment_index][1] <= chunk_start:
            segment_index += 1
        scan = segment_index
        while scan < len(segments):
            seg_start, seg_stop, row = segments[scan]
            if seg_start >= chunk_stop:
                break
            overlap_start = max(seg_start, chunk_start)
            overlap_stop = min(seg_stop, chunk_stop)
            if overlap_start < overlap_stop:
                x[overlap_start:overlap_stop].addcmul_(
                    other[overlap_start - chunk_start:overlap_stop - chunk_start],
                    gate[row].to(x.dtype),
                )
            scan += 1
        del hidden, other
    del weight1, weight2
    return x


def install(
    minimax_module,
    norm_kernels,
    mlp_kernels,
    *,
    chunk_tokens: int,
    enable_chunked_mlp: bool = True,
    enable_resident_mlp_weights: bool = True,
) -> dict[str, object]:
    """Install the bounded-memory exact-memory implementation as production's controlled baseline."""
    state = mlp_kernels.install(
        minimax_module,
        norm_kernels,
        chunk_tokens=chunk_tokens,
        enable_chunked_mlp=enable_chunked_mlp,
    )
    if enable_chunked_mlp and enable_resident_mlp_weights:
        comfy_ops = minimax_module.comfy.ops
        ck = minimax_module.comfy.quant_ops.ck

        def dit_forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
            fused_norm = norm_kernels.native_rms_two_pass_adaln
            h = fused_norm(x, self.norm1, shift_msa, scale_msa, mod_segments)
            attention_output = self.attn(
                h, rope_freqs=rope_freqs, transformer_options=transformer_options)
            x = minimax_module._mod_gate(x, gate_msa, attention_output, mod_segments)
            del attention_output
            h = fused_norm(x, self.norm2, shift_mlp, scale_mlp, mod_segments)
            result = resident_weight_chunked_gated_mlp(
                x, h, self.mlp, gate_mlp, mod_segments, chunk_tokens, comfy_ops, ck)
            if result is not None:
                return result
            return mlp_kernels.chunked_gated_mlp(
                x, h, self.mlp, gate_mlp, mod_segments, chunk_tokens)

        dit_forward.__name__ = "dit_forward_resident_weight_chunked_mlp"
        dit_forward._h3_resident_mlp_weights = True
        minimax_module.DiTBlock.forward = dit_forward
    return {
        **state,
        "core_math_exact": True,
        "resident_mlp_weights": enable_resident_mlp_weights,
    }
