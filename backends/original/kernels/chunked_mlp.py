"""bounded-memory exact-memory hooks for MiniMax H3 on RTX 4090.

bounded-memory keeps the released production attention, RMSNorm/AdaLN and partial-RoPE paths.  It
adds bounded-memory MLP evaluation and segmented dispatch for long RoPE
sequences.  Both transformations preserve the per-token mathematical graph.
"""

from __future__ import annotations

import torch


def install_long_sequence_rope_dispatch(
    norm_kernels,
    *,
    sequence_chunk_tokens: int = 32768,
) -> dict[str, object]:
    """Split only sequences that exceed CUDA's grid.z launch limit.

    The production exact partial-RoPE kernel uses ``(batch, heads, sequence)`` as its
    launch grid.  CUDA limits grid.z to 65,535, while a 1280x736x362 H3 job has
    roughly 99k sequence entries.  Sequence rows are independent, so slicing
    them and invoking the unchanged production kernel is numerically equivalent.
    """
    original = norm_kernels._partial_rotate_exact
    if getattr(original, "_h3_long_sequence_dispatch", False):
        return {
            "long_sequence_rope": True,
            "rope_sequence_chunk_tokens": original._h3_sequence_chunk_tokens,
        }

    def partial_rotate_long_sequence_exact(normalized, out, freqs, rot_dim):
        bnhd = norm_kernels.detect_rms_rope_bnhd(normalized, freqs, rot_dim=rot_dim)
        if bnhd is None:
            return original(normalized, out, freqs, rot_dim)

        sequence_dim = 1 if bnhd else 2
        sequence = normalized.shape[sequence_dim]
        if sequence <= 65535:
            return original(normalized, out, freqs, rot_dim)

        for start in range(0, sequence, sequence_chunk_tokens):
            stop = min(start + sequence_chunk_tokens, sequence)
            if bnhd:
                normalized_chunk = normalized[:, start:stop, :, :]
                out_chunk = out[:, start:stop, :, :]
                freqs_chunk = freqs if freqs.shape[1] == 1 else freqs[:, start:stop, ...]
            else:
                normalized_chunk = normalized[:, :, start:stop, :]
                out_chunk = out[:, :, start:stop, :]
                freqs_chunk = freqs if freqs.shape[2] == 1 else freqs[:, :, start:stop, ...]
            original(normalized_chunk, out_chunk, freqs_chunk, rot_dim)

    partial_rotate_long_sequence_exact.__name__ = "partial_rotate_long_sequence_exact"
    partial_rotate_long_sequence_exact._h3_long_sequence_dispatch = True
    partial_rotate_long_sequence_exact._h3_sequence_chunk_tokens = sequence_chunk_tokens
    norm_kernels._partial_rotate_exact = partial_rotate_long_sequence_exact
    return {
        "long_sequence_rope": True,
        "rope_sequence_chunk_tokens": sequence_chunk_tokens,
    }


def chunked_gated_mlp(
    x: torch.Tensor,
    h: torch.Tensor,
    mlp,
    gate: torch.Tensor,
    segments,
    chunk_tokens: int,
) -> torch.Tensor:
    """Evaluate MLP in bounded sequence chunks and consume each result at once."""
    if chunk_tokens <= 0 or h.shape[0] <= chunk_tokens:
        other = mlp(h)
        for start, stop, row in segments:
            x[start:stop].addcmul_(other[start:stop], gate[row].to(x.dtype))
        return x

    sequence = h.shape[0]
    segment_index = 0
    for chunk_start in range(0, sequence, chunk_tokens):
        chunk_stop = min(chunk_start + chunk_tokens, sequence)
        other = mlp(h[chunk_start:chunk_stop])

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
        del other
    return x


def install(
    minimax_module,
    norm_kernels,
    *,
    chunk_tokens: int,
    enable_chunked_mlp: bool = True,
) -> dict[str, object]:
    """Install production-exact kernels plus the bounded-memory bounded-memory DiT forward."""
    rope_state = install_long_sequence_rope_dispatch(norm_kernels)
    norm_state = norm_kernels.install(
        minimax_module,
        enable_rms_adaln=True,
        enable_partial_rope=True,
        aggressive_fused_rms=False,
    )

    if enable_chunked_mlp:
        def dit_forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
            fused_norm = norm_kernels.native_rms_two_pass_adaln
            h = fused_norm(x, self.norm1, shift_msa, scale_msa, mod_segments)
            attention_output = self.attn(
                h,
                rope_freqs=rope_freqs,
                transformer_options=transformer_options,
            )
            x = minimax_module._mod_gate(x, gate_msa, attention_output, mod_segments)
            del attention_output
            h = fused_norm(x, self.norm2, shift_mlp, scale_mlp, mod_segments)
            return chunked_gated_mlp(
                x,
                h,
                self.mlp,
                gate_mlp,
                mod_segments,
                chunk_tokens,
            )

        dit_forward.__name__ = "dit_forward_exact_chunked_mlp"
        dit_forward._h3_chunked_mlp = True
        dit_forward._h3_chunk_tokens = chunk_tokens
        minimax_module.DiTBlock.forward = dit_forward

    return {
        **norm_state,
        **rope_state,
        "chunked_mlp": enable_chunked_mlp,
        "mlp_chunk_tokens": chunk_tokens,
    }
