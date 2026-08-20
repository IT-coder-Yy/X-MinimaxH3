"""Lightweight single-GPU H3 transformer layers.

The fused-QKV graph, indexed AdaLN formulation and partial 3-D RoPE are adapted
from SGLang's Apache-2.0 MiniMax H3 implementation, rewritten here using only
PyTorch primitives and injected linear/attention backends. Modified for a
batch-one, no-TP, no-distributed serving target.
"""

from __future__ import annotations

import math
from typing import Callable, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F

AttentionBackend = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
ModulationSegment: TypeAlias = tuple[int, int, int]


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-5, *, device=None, dtype=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # Match the mature H3 reference path.  Besides using PyTorch's fused
        # reduction, this avoids materializing two full FP32 copies of the
        # packed [tokens, hidden] activation in every transformer block.
        return F.rms_norm(
            value,
            (self.weight.shape[0],),
            weight=self.weight.to(dtype=value.dtype),
            eps=self.eps,
        )


def rope_frequencies(
    position_ids: torch.Tensor, inv_freq: torch.Tensor
) -> torch.Tensor:
    if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
        raise ValueError("position_ids must be [sequence,3]")
    per_axis = position_ids.float().unsqueeze(-1) * inv_freq.float().reshape(1, 1, -1)
    half = torch.cat(per_axis.unbind(dim=1), dim=-1)
    return torch.cat((half, half), dim=-1)


def rope_rotation_table(angles: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Pack split-half angles once for all transformer blocks."""

    half = angles.shape[-1] // 2
    active = angles[:, :half]
    cosine, sine = torch.cos(active), torch.sin(active)
    return torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(
        1, angles.shape[0], 1, half, 2, 2
    ).to(dtype)


def apply_qknorm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    frequencies: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference order: per-head RMSNorm, then split-half partial RoPE."""

    if frequencies.ndim == 6:
        rotate_width = int(frequencies.shape[-3]) * 2
        if query.is_cuda and query.dtype in (torch.float16, torch.bfloat16):
            from comfy_kitchen import rms_rope_split_half_

            query_4d, key_4d = query.unsqueeze(0), key.unsqueeze(0)
            rms_rope_split_half_(
                query_4d,
                key_4d,
                frequencies,
                q_weight.to(device=query.device, dtype=query.dtype),
                k_weight.to(device=key.device, dtype=key.dtype),
                epsilon=eps,
                rot_dim=rotate_width,
            )
            return query_4d[0], key_4d[0]

        query = F.rms_norm(
            query, (query.shape[-1],), weight=q_weight.to(query.dtype), eps=eps
        )
        key = F.rms_norm(
            key, (key.shape[-1],), weight=k_weight.to(key.dtype), eps=eps
        )

        def rotate_table(value: torch.Tensor) -> torch.Tensor:
            half_width = rotate_width // 2
            first = value[..., :half_width]
            second = value[..., half_width:rotate_width]
            table = frequencies[0, :, 0].to(value.dtype)
            out_first = (
                first * table[:, None, :, 0, 0]
                + second * table[:, None, :, 0, 1]
            )
            out_second = (
                first * table[:, None, :, 1, 0]
                + second * table[:, None, :, 1, 1]
            )
            return torch.cat(
                (out_first, out_second, value[..., rotate_width:]), dim=-1
            )

        return rotate_table(query), rotate_table(key)

    def normalize(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            value,
            (value.shape[-1],),
            weight=weight.to(dtype=value.dtype),
            eps=eps,
        )

    query, key = normalize(query, q_weight), normalize(key, k_weight)
    rotate_width = int(frequencies.shape[-1])
    half = rotate_width // 2
    cosine = torch.cos(frequencies[:, :half]).to(query.dtype)
    sine = torch.sin(frequencies[:, :half]).to(query.dtype)
    cosine = torch.cat((cosine, cosine), dim=-1).unsqueeze(1)
    sine = torch.cat((sine, sine), dim=-1).unsqueeze(1)

    def rotate(value: torch.Tensor) -> torch.Tensor:
        active, passthrough = value[..., :rotate_width], value[..., rotate_width:]
        first, second = active.chunk(2, dim=-1)
        perpendicular = torch.cat((-second, first), dim=-1)
        active = active * cosine + perpendicular * sine
        return torch.cat((active, passthrough), dim=-1)

    return rotate(query), rotate(key)


def torch_sdpa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    q = query.transpose(0, 1).unsqueeze(0)
    k = key.transpose(0, 1).unsqueeze(0)
    v = value.transpose(0, 1).unsqueeze(0)
    output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    return output.squeeze(0).transpose(0, 1).contiguous()


class FusedQKVAttention(nn.Module):
    def __init__(
        self,
        qkv_proj: nn.Module,
        out_proj: nn.Module,
        *,
        num_heads: int,
        head_dim: int,
        qk_eps: float = 1e-5,
        backend: AttentionBackend = torch_sdpa,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.qkv_proj = qkv_proj
        self.out_proj = out_proj
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_width = num_heads * head_dim
        self.q_norm = RMSNorm(head_dim, qk_eps, device=device, dtype=dtype)
        self.k_norm = RMSNorm(head_dim, qk_eps, device=device, dtype=dtype)
        self.backend = backend

    def forward(self, value: torch.Tensor, frequencies: torch.Tensor | None) -> torch.Tensor:
        qkv = self.qkv_proj(value)
        expected = 3 * self.inner_width
        if qkv.shape[-1] != expected:
            raise ValueError(f"fused QKV output width {qkv.shape[-1]} != {expected}")
        query, key, val = qkv.split(self.inner_width, dim=-1)
        shape = (value.shape[0], self.num_heads, self.head_dim)
        query, key, val = query.view(shape), key.view(shape), val.view(shape)
        if frequencies is None:
            zero = value.new_zeros((value.shape[0], 0), dtype=torch.float32)
            frequencies = zero
        if frequencies.shape[-1]:
            query, key = apply_qknorm_rope(
                query,
                key,
                q_weight=self.q_norm.weight,
                k_weight=self.k_norm.weight,
                frequencies=frequencies,
                eps=self.q_norm.eps,
            )
        else:
            query = self.q_norm(query)
            key = self.k_norm(key)
        attended = self.backend(query, key, val).reshape(value.shape[0], self.inner_width)
        return self.out_proj(attended)


class SwiGLUMLP(nn.Module):
    def __init__(self, fc1: nn.Module, fc2: nn.Module) -> None:
        super().__init__()
        self.fc1, self.fc2 = fc1, fc2

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate, up = self.fc1(value).chunk(2, dim=-1)
        return self.fc2(F.silu(gate).mul_(up))


def split_adaln(output: torch.Tensor, *, hidden_size: int) -> tuple[torch.Tensor, ...]:
    if output.shape[-1] != 18 * hidden_size:
        raise ValueError("block AdaLN projection must produce 18 * hidden_size")
    return output.view(-1, 6 * hidden_size).chunk(6, dim=-1)


def _scale_shift(
    value: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    segments: tuple[ModulationSegment, ...],
) -> torch.Tensor:
    """Apply AdaLN modulation without expanding rows to every packed token.

    H3 has only a handful of contiguous modality/timestep runs.  The former
    implementation used ``index_select`` to create full [tokens, hidden]
    shift and scale tensors twice per block.  At 480p this moved hundreds of
    MiB for each call.  Segment broadcasts are the operation order used by the
    accepted Comfy implementation and preserve the BF16 materialization
    boundaries (multiply, then add).
    """

    previous = 0
    for start, stop, row in segments:
        if start != previous or stop <= start:
            raise ValueError("modulation segments must be an ordered partition")
        target = value[start:stop]
        target.mul_(1.0 + scale[row].to(value.dtype)).add_(shift[row].to(value.dtype))
        previous = stop
    if previous != value.shape[0]:
        raise ValueError("modulation segments do not cover the packed sequence")
    return value


def _gated_residual(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    segments: tuple[ModulationSegment, ...],
) -> torch.Tensor:
    previous = 0
    for start, stop, row in segments:
        if start != previous or stop <= start:
            raise ValueError("modulation segments must be an ordered partition")
        residual[start:stop].addcmul_(
            update[start:stop], gate[row].to(update.dtype)
        )
        previous = stop
    if previous != residual.shape[0]:
        raise ValueError("modulation segments do not cover the packed sequence")
    return residual


class H3TransformerBlock(nn.Module):
    """One full H3 block; AdaLN projector accepts unique timestep rows."""

    def __init__(
        self,
        attention: FusedQKVAttention,
        mlp: SwiGLUMLP,
        adaln_projector: nn.Module,
        *,
        hidden_size: int,
        norm_eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.attention = attention
        self.mlp = mlp
        self.adaln_projector = adaln_projector
        self.hidden_size = hidden_size
        self.norm1 = RMSNorm(hidden_size, norm_eps, device=device, dtype=dtype)
        self.norm2 = RMSNorm(hidden_size, norm_eps, device=device, dtype=dtype)

    def forward(
        self,
        value: torch.Tensor,
        *,
        timestep_rows: torch.Tensor,
        modulation_segments: tuple[ModulationSegment, ...],
        frequencies: torch.Tensor,
        mlp_chunk_tokens: int | None = None,
    ) -> torch.Tensor:
        from .frame_interleave import current_frame_interleave_layer
        from .kernels import current_attention_layer
        from .spatial_query_lattice import current_spatial_query_lattice_layer

        layer = current_attention_layer()
        lattice_plan = current_spatial_query_lattice_layer(layer)
        if lattice_plan is not None:
            from .modality_refresh import refresh_selected_video_tiles

            # The final generated-video run is the only suffix after the
            # protected text/condition/audio prefix.
            protected_tokens = modulation_segments[-1][0]
            active_before = value.index_select(
                0, lattice_plan.active_video_indices
            )
            value = refresh_selected_video_tiles(
                self,
                value,
                protected_tokens=protected_tokens,
                active_video_indices=lattice_plan.active_video_indices,
                timestep_rows=timestep_rows,
                modulation_segments=modulation_segments,
                frequencies=frequencies,
            )
            return lattice_plan.reconstruct_inactive_(value, active_before)

        frame_plan = current_frame_interleave_layer(layer)
        if frame_plan is not None:
            selected_input = value.index_select(0, frame_plan.selected_indices)
            selected_output = self._forward_complete(
                selected_input.clone(),
                timestep_rows=timestep_rows,
                modulation_segments=frame_plan.selected_modulation_segments(
                    modulation_segments
                ),
                frequencies=frame_plan.selected_frequencies(frequencies),
                mlp_chunk_tokens=mlp_chunk_tokens,
            )
            return frame_plan.reconstruct_(value, selected_input, selected_output)
        return self._forward_complete(
            value,
            timestep_rows=timestep_rows,
            modulation_segments=modulation_segments,
            frequencies=frequencies,
            mlp_chunk_tokens=mlp_chunk_tokens,
        )

    def _forward_complete(
        self,
        value: torch.Tensor,
        *,
        timestep_rows: torch.Tensor,
        modulation_segments: tuple[ModulationSegment, ...],
        frequencies: torch.Tensor,
        mlp_chunk_tokens: int | None,
    ) -> torch.Tensor:
        params = split_adaln(
            self.adaln_projector(timestep_rows), hidden_size=self.hidden_size
        )
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = params
        from .kernels import current_attention_layer, rms_adaln

        hidden = rms_adaln(
            value, self.norm1, shift_a, scale_a, modulation_segments
        )
        value = _gated_residual(
            value,
            self.attention(hidden, frequencies),
            gate_a,
            modulation_segments,
        )
        hidden = rms_adaln(
            value, self.norm2, shift_m, scale_m, modulation_segments
        )
        from .fused_mlp import try_fused_gated_mlp
        from .research_capture import (
            capture_kind,
            capture_target,
            persist_mlp_capture,
            record_video_mlp_gate,
        )

        record_video_mlp_gate(
            gate_m,
            row=modulation_segments[-1][2],
            layer=current_attention_layer(),
        )

        from .kernels import current_attention_step
        from .mlp_gate_budget import skip_video_mlp

        step_context = current_attention_step()
        video_modulation_row = modulation_segments[-1][2]
        if skip_video_mlp(
            layer=current_attention_layer(),
            step=None if step_context is None else step_context[0],
            step_count=None if step_context is None else step_context[1],
            gate=gate_m,
            row=video_modulation_row,
        ):
            # The generated-video segment is the final packed run.  Preserve
            # every conditioning/audio row exactly; only its local video MLP
            # residual becomes identity at the gate-selected coordinates.
            protected = modulation_segments[-1][0]
            prefix_segments = tuple(
                segment for segment in modulation_segments if segment[1] <= protected
            )
            if not prefix_segments or prefix_segments[-1][1] != protected:
                raise ValueError("video MLP budget requires a complete protected prefix")
            prefix_value = value[:protected]
            prefix_hidden = hidden[:protected]
            if not try_fused_gated_mlp(
                prefix_value,
                prefix_hidden,
                self.mlp,
                gate_m,
                prefix_segments,
                chunk_tokens=mlp_chunk_tokens,
            ):
                _gated_residual(
                    prefix_value,
                    self.mlp(prefix_hidden),
                    gate_m,
                    prefix_segments,
                )
            return value

        capture = capture_target(current_attention_layer())
        capture_protected = modulation_segments[-1][0]
        capture_delta = capture is not None and capture_kind() == "delta"
        capture_before = value[capture_protected:].clone() if capture_delta else None

        # Preserve the complete attention/motion path while routing only the
        # row-local MLP.  Prefix modalities remain exact, and omitted video
        # residuals are reconstructed strictly inside one frame/spatial row.
        from .mlp_spatial_lattice import current_mlp_spatial_lattice_layer

        mlp_plan = current_mlp_spatial_lattice_layer(
            current_attention_layer()
        )
        if mlp_plan is not None:
            protected = mlp_plan.protected_tokens
            from .mlp_spatial_lattice import current_mlp_spatial_lattice_config

            mlp_config = current_mlp_spatial_lattice_config()
            detail_fraction = 0.0 if mlp_config is None else mlp_config.detail_fraction
            detail_positions, detail_indices = mlp_plan.select_detail_positions(
                hidden, detail_fraction
            )
            prefix_indices = torch.arange(
                protected, dtype=torch.long, device=value.device
            )
            selected_indices = torch.cat(
                (prefix_indices, mlp_plan.active_video_indices, detail_indices), dim=0
            )
            selected_hidden = hidden.index_select(0, selected_indices)
            selected_value = value.index_select(0, selected_indices)
            selected_before = selected_value.clone()
            prefix_segments = tuple(
                segment for segment in modulation_segments if segment[1] <= protected
            )
            video_segments = [
                segment for segment in modulation_segments if segment[0] == protected
            ]
            if len(video_segments) != 1:
                raise ValueError("MLP spatial lattice requires one generated-video segment")
            video_row = video_segments[0][2]
            selected_segments = prefix_segments + (
                (protected, selected_value.shape[0], video_row),
            )
            if not try_fused_gated_mlp(
                selected_value,
                selected_hidden,
                self.mlp,
                gate_m,
                selected_segments,
                chunk_tokens=mlp_chunk_tokens,
            ):
                selected_value = _gated_residual(
                    selected_value,
                    self.mlp(selected_hidden),
                    gate_m,
                    selected_segments,
                )
            value[:protected].copy_(selected_value[:protected])
            selected_delta = selected_value[protected:].sub_(selected_before[protected:])
            active_count = int(mlp_plan.active_video_indices.numel())
            mlp_plan.reconstruct_(
                value,
                selected_delta[:active_count],
                detail_positions=detail_positions,
                detail_delta=selected_delta[active_count:],
            )
            return value

        if try_fused_gated_mlp(
            value,
            hidden,
            self.mlp,
            gate_m,
            modulation_segments,
            chunk_tokens=mlp_chunk_tokens,
        ):
            if capture is not None:
                persist_mlp_capture(
                    capture,
                    hidden_video=hidden[capture_protected:],
                    delta_video=(
                        value[capture_protected:] - capture_before
                        if capture_before is not None
                        else None
                    ),
                    protected_tokens=capture_protected,
                )
            return value
        value = _gated_residual(
            value, self.mlp(hidden), gate_m, modulation_segments
        )
        if capture is not None:
            persist_mlp_capture(
                capture,
                hidden_video=hidden[capture_protected:],
                delta_video=(
                    value[capture_protected:] - capture_before
                    if capture_before is not None
                    else None
                ),
                protected_tokens=capture_protected,
            )
        return value


def time_shift_sigma(sigma: torch.Tensor, from_shift: float, to_shift: float) -> torch.Tensor:
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def time_shift_slope(sigma: torch.Tensor, from_shift: float, to_shift: float) -> torch.Tensor:
    """Derivative of the target shifted schedule with respect to the source."""

    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return (
        to_shift
        * (1.0 + (from_shift - 1.0) * base).square()
        / (from_shift * (1.0 + (to_shift - 1.0) * base).square())
    )


def timestep_embedding(timestep: torch.Tensor, width: int = 256) -> torch.Tensor:
    half = width // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32, device=timestep.device)
        / half
    )
    angles = timestep.float().reshape(-1, 1) * frequencies.reshape(1, -1)
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
