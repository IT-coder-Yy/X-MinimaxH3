"""Shape-adaptive exact MLP chunk routing."""

from __future__ import annotations


def select_chunk_tokens(
    sequence_rows: int,
    *,
    short_chunk_tokens: int = 8192,
    long_chunk_tokens: int = 4096,
    long_sequence_threshold: int = 20000,
) -> int:
    """Select the measured 4090 tile without changing per-token MLP math."""
    if int(sequence_rows) <= int(long_sequence_threshold):
        return int(short_chunk_tokens)
    return int(long_chunk_tokens)


def install_adaptive_mlp(
    minimax_module,
    norm_kernels,
    mlp_kernels,
    linear_kernels,
    *,
    short_chunk_tokens=8192,
    long_chunk_tokens=4096,
    long_sequence_threshold=20000,
):
    import native_acceleration

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
        chunk_tokens = select_chunk_tokens(
            h.shape[0],
            short_chunk_tokens=short_chunk_tokens,
            long_chunk_tokens=long_chunk_tokens,
            long_sequence_threshold=long_sequence_threshold,
        )
        result = native_acceleration.resident_weight_chunked_gated_mlp(
            x, h, self.mlp, gate_mlp, mod_segments, chunk_tokens,
            comfy_ops, ck, linear_kernels,
        )
        if result is not None:
            return result
        return mlp_kernels.chunked_gated_mlp(
            x, h, self.mlp, gate_mlp, mod_segments, chunk_tokens
        )

    dit_forward.__name__ = "shape_adaptive_exact_mlp"
    minimax_module.DiTBlock.forward = dit_forward
    return {
        "adaptive_mlp_chunks": True,
        "short_chunk_tokens": int(short_chunk_tokens),
        "long_chunk_tokens": int(long_chunk_tokens),
        "long_sequence_threshold": int(long_sequence_threshold),
    }
