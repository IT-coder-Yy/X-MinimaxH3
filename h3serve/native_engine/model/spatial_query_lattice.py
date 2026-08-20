"""Rotating, block-aligned Query routing for long H3 video sequences.

SQLR keeps the packed prefix and the complete K/V memory at every layer.  It
only selects which generated-video Query rows receive a transformer update in
one layer.  Selection follows the native 128-row Sparge Query blocks so
repacking does not change the backend's pooling or top-k semantics.  The phase
rotates across layers (and sampler steps), therefore no spatial region is
permanently removed and every latent frame remains represented.

Unlike the rejected frame-interleave prototype, SQLR never copies or
interpolates hidden states across frames or spatial coordinates.
"""

from __future__ import annotations

import contextlib
import contextvars
import bisect
from dataclasses import dataclass

import torch

from .packed import PackedLayout


@dataclass(frozen=True, slots=True)
class SpatialQueryLatticeConfig:
    stride: int = 1
    layer_start: int = 0
    layer_stop: int = 50
    dense_layers: tuple[int, ...] = ()
    phase_offset: int = 0
    query_block_rows: int = 128

    def __post_init__(self) -> None:
        if self.stride <= 0:
            raise ValueError("spatial Query lattice stride must be positive")
        if not 0 <= self.layer_start <= self.layer_stop <= 50:
            raise ValueError("spatial Query lattice layer range must be inside [0, 50]")
        if tuple(sorted(set(self.dense_layers))) != self.dense_layers:
            raise ValueError("spatial Query lattice dense layers must be sorted and unique")
        if any(layer < 0 or layer >= 50 for layer in self.dense_layers):
            raise ValueError("spatial Query lattice dense layer is outside [0, 50)")
        if self.query_block_rows <= 0:
            raise ValueError("spatial Query lattice block rows must be positive")

    def sparse_at(self, layer: int) -> bool:
        return (
            self.stride > 1
            and self.layer_start <= layer < self.layer_stop
            and layer not in self.dense_layers
        )


@dataclass(frozen=True, slots=True)
class SpatialQueryLatticeLayerPlan:
    active_video_indices: torch.Tensor
    inactive_video_indices: torch.Tensor
    left_active_positions: torch.Tensor
    right_active_positions: torch.Tensor
    right_weights: torch.Tensor

    def reconstruct_inactive_(
        self,
        value: torch.Tensor,
        active_before: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct this layer's missing residual inside each latent frame.

        No hidden state or residual crosses a frame boundary.  Exact active
        rows have already been written into ``value`` by the real H3 block.
        """

        active_after = value.index_select(0, self.active_video_indices)
        active_delta = active_after.sub(active_before)
        chunk_rows = 8192
        for start in range(0, int(self.inactive_video_indices.numel()), chunk_rows):
            stop = min(
                start + chunk_rows,
                int(self.inactive_video_indices.numel()),
            )
            left = active_delta.index_select(
                0, self.left_active_positions[start:stop]
            )
            right = active_delta.index_select(
                0, self.right_active_positions[start:stop]
            )
            weights = self.right_weights[start:stop].to(
                dtype=value.dtype
            ).unsqueeze(1)
            update = torch.lerp(left, right, weights)
            target_indices = self.inactive_video_indices[start:stop]
            target = value.index_select(0, target_indices)
            target.add_(update)
            value.index_copy_(0, target_indices, target)
        return value


class SpatialQueryLatticePlan:
    def __init__(
        self,
        layout: PackedLayout,
        config: SpatialQueryLatticeConfig,
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device
        video = layout.segment("video", last=True)
        self.protected_tokens = int(video.start)
        self.video_tokens = int(video.length)
        self.latent_frames = int(layout.signature[1])
        if self.latent_frames <= 0 or self.video_tokens % self.latent_frames:
            raise ValueError("spatial Query lattice requires complete latent frames")
        self.frame_tokens = self.video_tokens // self.latent_frames
        self._phases: dict[int, SpatialQueryLatticeLayerPlan] = {}

    def for_layer(self, layer: int) -> SpatialQueryLatticeLayerPlan | None:
        if not self.config.sparse_at(layer):
            return None
        phase = (
            layer - self.config.layer_start + self.config.phase_offset
        ) % self.config.stride
        cached = self._phases.get(phase)
        if cached is not None:
            return cached

        rows = self.config.query_block_rows
        selected: list[torch.Tensor] = []
        query_blocks = (self.video_tokens + rows - 1) // rows
        for query_block in range(query_blocks):
            if query_block % self.config.stride != phase:
                continue
            start = query_block * rows
            stop = min(start + rows, self.video_tokens)
            selected.append(
                torch.arange(
                    self.protected_tokens + start,
                    self.protected_tokens + stop,
                    dtype=torch.long,
                    device=self.device,
                )
            )
        if not selected:
            raise ValueError("spatial Query lattice selected no generated-video rows")
        active = torch.cat(selected)
        active_local = (active - self.protected_tokens).tolist()
        active_lookup = {row: position for position, row in enumerate(active_local)}
        inactive: list[int] = []
        left_positions: list[int] = []
        right_positions: list[int] = []
        right_weights: list[float] = []
        # Interpolation is constrained to one latent frame.  This preserves
        # temporal identity and avoids the cross-frame ghosting seen in the
        # rejected frame-interleave route.
        for frame in range(self.latent_frames):
            frame_start = frame * self.frame_tokens
            frame_stop = frame_start + self.frame_tokens
            frame_active = [
                row for row in active_local if frame_start <= row < frame_stop
            ]
            if not frame_active:
                raise ValueError(
                    "spatial Query lattice left one latent frame without an exact rail"
                )
            for row in range(frame_start, frame_stop):
                if row in active_lookup:
                    continue
                insertion = bisect.bisect_left(frame_active, row)
                if insertion == 0:
                    left = right = frame_active[0]
                elif insertion == len(frame_active):
                    left = right = frame_active[-1]
                else:
                    left, right = frame_active[insertion - 1], frame_active[insertion]
                inactive.append(self.protected_tokens + row)
                left_positions.append(active_lookup[left])
                right_positions.append(active_lookup[right])
                right_weights.append(
                    0.0 if left == right else float(row - left) / float(right - left)
                )
        cached = SpatialQueryLatticeLayerPlan(
            active_video_indices=active,
            inactive_video_indices=torch.tensor(
                inactive, dtype=torch.long, device=self.device
            ),
            left_active_positions=torch.tensor(
                left_positions, dtype=torch.long, device=self.device
            ),
            right_active_positions=torch.tensor(
                right_positions, dtype=torch.long, device=self.device
            ),
            right_weights=torch.tensor(
                right_weights, dtype=torch.float32, device=self.device
            ),
        )
        self._phases[phase] = cached
        return cached


_CONFIG: contextvars.ContextVar[SpatialQueryLatticeConfig | None] = (
    contextvars.ContextVar("h3_spatial_query_lattice_config", default=None)
)
_PLAN: contextvars.ContextVar[SpatialQueryLatticePlan | None] = (
    contextvars.ContextVar("h3_spatial_query_lattice_plan", default=None)
)


@contextlib.contextmanager
def spatial_query_lattice_config(config: SpatialQueryLatticeConfig | None):
    token = _CONFIG.set(config)
    try:
        yield
    finally:
        _CONFIG.reset(token)


def current_spatial_query_lattice_config() -> SpatialQueryLatticeConfig | None:
    return _CONFIG.get()


@contextlib.contextmanager
def spatial_query_lattice_plan(plan: SpatialQueryLatticePlan | None):
    token = _PLAN.set(plan)
    try:
        yield
    finally:
        _PLAN.reset(token)


def current_spatial_query_lattice_layer(
    layer: int | None,
) -> SpatialQueryLatticeLayerPlan | None:
    plan = _PLAN.get()
    if plan is None or layer is None:
        return None
    return plan.for_layer(layer)


__all__ = [
    "SpatialQueryLatticeConfig",
    "SpatialQueryLatticePlan",
    "SpatialQueryLatticeLayerPlan",
    "current_spatial_query_lattice_config",
    "current_spatial_query_lattice_layer",
    "spatial_query_lattice_config",
    "spatial_query_lattice_plan",
]
