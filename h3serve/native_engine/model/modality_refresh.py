"""Asymmetric protected-modality refresh for packed H3 transformer blocks.

H3 packs a small text/condition/audio prefix before a dominant generated-video
suffix.  For one transformer block, prefix queries need the full key/value
sequence but do not require evaluating video queries.  The MLP is row-local.
This helper therefore refreshes the sensitive prefix while leaving generated
video rows untouched.  A later cache policy may combine it with a predicted
same-coordinate video residual.

The prefix result is a real block computation for the supplied input state.
Across multiple consecutive blocks, however, leaving video rows unchanged
alters the K/V state seen by later blocks; callers must treat a multi-block
use as an approximation and gate/refresh it accordingly.
"""

from __future__ import annotations

import torch

from .layers import (
    H3TransformerBlock,
    ModulationSegment,
    _gated_residual,
    apply_qknorm_rope,
    split_adaln,
)


def _prefix_segments(
    segments: tuple[ModulationSegment, ...], protected_tokens: int
) -> tuple[ModulationSegment, ...]:
    selected: list[ModulationSegment] = []
    for start, stop, row in segments:
        if start >= protected_tokens:
            break
        if stop > protected_tokens:
            raise ValueError("protected boundary must align with a modality segment")
        selected.append((start, stop, row))
    if not selected or selected[-1][1] != protected_tokens:
        raise ValueError("modulation segments do not cover the protected prefix")
    return tuple(selected)


@torch.inference_mode()
def refresh_protected_modalities(
    block: H3TransformerBlock,
    value: torch.Tensor,
    *,
    protected_tokens: int,
    timestep_rows: torch.Tensor,
    modulation_segments: tuple[ModulationSegment, ...],
    frequencies: torch.Tensor,
) -> torch.Tensor:
    """Refresh text/condition/audio rows and preserve generated-video rows.

    QKV is projected for every packed row because protected queries still
    attend to generated-video keys and values.  Attention output projection,
    the second normalization and the MLP are evaluated only for the prefix.
    The input tensor is updated in place and returned.
    """

    if value.ndim != 2:
        raise ValueError("packed H3 activation must be rank two")
    if not 0 < protected_tokens < value.shape[0]:
        raise ValueError("protected token count must split prefix and video")
    prefix_segments = _prefix_segments(modulation_segments, protected_tokens)
    params = split_adaln(
        block.adaln_projector(timestep_rows), hidden_size=block.hidden_size
    )
    shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = params

    from .kernels import rms_adaln

    hidden = rms_adaln(
        value, block.norm1, shift_a, scale_a, modulation_segments
    )
    attention = block.attention
    qkv = attention.qkv_proj(hidden)
    expected = 3 * attention.inner_width
    if qkv.shape[-1] != expected:
        raise ValueError(f"fused QKV output width {qkv.shape[-1]} != {expected}")
    query, key, val = qkv.split(attention.inner_width, dim=-1)
    shape = (value.shape[0], attention.num_heads, attention.head_dim)
    query, key, val = query.view(shape), key.view(shape), val.view(shape)
    if frequencies.shape[-1]:
        query, key = apply_qknorm_rope(
            query,
            key,
            q_weight=attention.q_norm.weight,
            k_weight=attention.k_norm.weight,
            frequencies=frequencies,
            eps=attention.q_norm.eps,
        )
    else:
        query, key = attention.q_norm(query), attention.k_norm(key)
    protected_attention = getattr(attention.backend, "protected_queries", None)
    attended = (
        protected_attention(query[:protected_tokens], key, val)
        if callable(protected_attention)
        else attention.backend(query[:protected_tokens], key, val)
    ).reshape(
        protected_tokens, attention.inner_width
    )
    prefix = value[:protected_tokens]
    _gated_residual(
        prefix,
        attention.out_proj(attended),
        gate_a,
        prefix_segments,
    )
    del hidden, qkv, query, key, val, attended

    prefix_hidden = rms_adaln(
        prefix, block.norm2, shift_m, scale_m, prefix_segments
    )
    _gated_residual(
        prefix,
        block.mlp(prefix_hidden),
        gate_m,
        prefix_segments,
    )
    return value


@torch.inference_mode()
def refresh_selected_video_tiles(
    block: H3TransformerBlock,
    value: torch.Tensor,
    *,
    protected_tokens: int,
    active_video_indices: torch.Tensor,
    timestep_rows: torch.Tensor,
    modulation_segments: tuple[ModulationSegment, ...],
    frequencies: torch.Tensor,
) -> torch.Tensor:
    """Refresh the protected prefix plus selected same-coordinate video rows.

    The full packed state still supplies K/V. Selected video indices are
    absolute packed-row indices, sorted and unique, and are written back at
    their original coordinates. No row is replaced by another frame or tile.
    """

    if value.ndim != 2 or active_video_indices.ndim != 1:
        raise ValueError("selected refresh expects packed value and 1-D indices")
    if active_video_indices.dtype != torch.long:
        raise ValueError("selected video indices must use torch.long")
    if active_video_indices.numel() <= 0:
        return refresh_protected_modalities(
            block,
            value,
            protected_tokens=protected_tokens,
            timestep_rows=timestep_rows,
            modulation_segments=modulation_segments,
            frequencies=frequencies,
        )
    if int(active_video_indices[0]) < protected_tokens or int(
        active_video_indices[-1]
    ) >= value.shape[0]:
        raise ValueError("selected video indices lie outside the video suffix")
    if not bool(torch.all(active_video_indices[1:] > active_video_indices[:-1])):
        raise ValueError("selected video indices must be sorted and unique")

    prefix_segments = _prefix_segments(modulation_segments, protected_tokens)
    video_segments = [
        segment for segment in modulation_segments if segment[0] == protected_tokens
    ]
    if len(video_segments) != 1 or video_segments[0][1] != value.shape[0]:
        raise ValueError("selected refresh requires one final generated-video segment")
    video_row = video_segments[0][2]
    active_count = int(active_video_indices.numel())
    active_segments = prefix_segments + (
        (protected_tokens, protected_tokens + active_count, video_row),
    )

    params = split_adaln(
        block.adaln_projector(timestep_rows), hidden_size=block.hidden_size
    )
    shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = params
    from .kernels import rms_adaln

    hidden = rms_adaln(
        value, block.norm1, shift_a, scale_a, modulation_segments
    )
    attention = block.attention
    qkv = attention.qkv_proj(hidden)
    expected = 3 * attention.inner_width
    if qkv.shape[-1] != expected:
        raise ValueError(f"fused QKV output width {qkv.shape[-1]} != {expected}")
    query, key, val = qkv.split(attention.inner_width, dim=-1)
    shape = (value.shape[0], attention.num_heads, attention.head_dim)
    query, key, val = query.view(shape), key.view(shape), val.view(shape)
    if frequencies.shape[-1]:
        query, key = apply_qknorm_rope(
            query,
            key,
            q_weight=attention.q_norm.weight,
            k_weight=attention.k_norm.weight,
            frequencies=frequencies,
            eps=attention.q_norm.eps,
        )
    else:
        query, key = attention.q_norm(query), attention.k_norm(key)
    selected_attention = getattr(attention.backend, "selected_queries", None)
    selected_query = query.index_select(0, active_video_indices)
    if callable(selected_attention):
        attended = selected_attention(
            query[:protected_tokens],
            selected_query,
            key,
            val,
            protected_tokens=protected_tokens,
            video_query_indices=active_video_indices - protected_tokens,
        )
    else:
        combined_query = torch.cat((query[:protected_tokens], selected_query), dim=0)
        attended = attention.backend(combined_query, key, val)
    attended = attended.reshape(protected_tokens + active_count, attention.inner_width)
    active_value = torch.cat(
        (value[:protected_tokens], value.index_select(0, active_video_indices)), dim=0
    )
    _gated_residual(
        active_value,
        attention.out_proj(attended),
        gate_a,
        active_segments,
    )
    del hidden, qkv, query, key, val, selected_query, attended

    active_hidden = rms_adaln(
        active_value, block.norm2, shift_m, scale_m, active_segments
    )
    _gated_residual(
        active_value,
        block.mlp(active_hidden),
        gate_m,
        active_segments,
    )
    value[:protected_tokens].copy_(active_value[:protected_tokens])
    value.index_copy_(0, active_video_indices, active_value[protected_tokens:])
    return value


__all__ = ["refresh_protected_modalities", "refresh_selected_video_tiles"]
