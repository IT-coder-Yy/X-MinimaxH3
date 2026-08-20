"""Motion-preserving spatial routing for the row-local H3 MLP.

The attention path remains untouched: every Query still observes the complete
packed sequence through the accepted MTCR backend.  Only the row-local MLP
update is evaluated on a rotating set of video columns.  Missing MLP residuals
are reconstructed from the nearest exact neighbours in the *same frame and
same spatial row*, so neither identity nor motion state is copied across time.

This is an opt-in research mechanism.  ``stride=1`` is the exact reference.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass

import torch

from .packed import PackedLayout


@dataclass(frozen=True, slots=True)
class MLPSpatialLatticeConfig:
    stride: int = 1
    layer_start: int = 0
    layer_stop: int = 50
    dense_layers: tuple[int, ...] = ()
    phase_offset: int = 0
    detail_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.stride <= 0:
            raise ValueError("MLP spatial lattice stride must be positive")
        if not 0 <= self.layer_start <= self.layer_stop <= 50:
            raise ValueError("MLP spatial lattice layer range must lie inside [0, 50]")
        if tuple(sorted(set(self.dense_layers))) != self.dense_layers:
            raise ValueError("MLP spatial lattice dense layers must be sorted and unique")
        if any(layer < 0 or layer >= 50 for layer in self.dense_layers):
            raise ValueError("MLP spatial lattice dense layer is outside [0, 50)")
        if not 0.0 <= self.detail_fraction < 1.0:
            raise ValueError("MLP spatial lattice detail fraction must lie inside [0, 1)")

    def sparse_at(self, layer: int) -> bool:
        return (
            self.stride > 1
            and self.layer_start <= layer < self.layer_stop
            and layer not in self.dense_layers
        )


@dataclass(frozen=True, slots=True)
class MLPSpatialLatticeLayerPlan:
    protected_tokens: int
    active_video_indices: torch.Tensor
    inactive_video_indices: torch.Tensor
    left_active_positions: torch.Tensor
    right_active_positions: torch.Tensor
    right_weights: torch.Tensor

    def select_detail_positions(
        self, hidden: torch.Tensor, fraction: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select omitted rows whose hidden state is least locally predictable.

        The score is a normalized reconstruction error against the same-row
        spatial lattice.  It is measured before the MLP, costs no model
        evaluation, and protects edges/subjects rather than assigning a fixed
        global keep rate to every scene.
        """

        if fraction <= 0.0 or self.inactive_video_indices.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=hidden.device)
            return empty, empty
        # A fixed, evenly spaced channel sketch preserves the edge/novelty
        # ranking while avoiding a second full-width activation pass.  This
        # selector is only a routing signal; exact MLP evaluation still uses
        # every hidden channel for all selected rows.
        channel_step = max(1, hidden.shape[1] // 128)
        sketch = hidden[:, ::channel_step]
        if sketch.shape[1] > 128:
            sketch = sketch[:, :128]
        active_hidden = sketch.index_select(0, self.active_video_indices)
        scores: list[torch.Tensor] = []
        chunk_rows = 2048
        for start in range(0, int(self.inactive_video_indices.numel()), chunk_rows):
            stop = min(start + chunk_rows, int(self.inactive_video_indices.numel()))
            left = active_hidden.index_select(
                0, self.left_active_positions[start:stop]
            )
            right = active_hidden.index_select(
                0, self.right_active_positions[start:stop]
            )
            weight = self.right_weights[start:stop].to(hidden.dtype).unsqueeze(1)
            estimate = torch.lerp(left, right, weight)
            target = sketch.index_select(0, self.inactive_video_indices[start:stop])
            numerator = (target.float() - estimate.float()).square().mean(dim=1)
            denominator = target.float().square().mean(dim=1).clamp_min_(1.0e-8)
            scores.append(numerator.div_(denominator))
        all_scores = torch.cat(scores)
        count = max(1, int(round(all_scores.numel() * fraction)))
        positions = torch.topk(all_scores, count, sorted=False).indices
        positions = torch.sort(positions).values
        return positions, self.inactive_video_indices.index_select(0, positions)

    def reconstruct_(
        self,
        value: torch.Tensor,
        active_delta: torch.Tensor,
        *,
        detail_positions: torch.Tensor | None = None,
        detail_delta: torch.Tensor | None = None,
    ) -> None:
        """Write exact active updates and same-row interpolated inactive updates."""

        value.index_add_(0, self.active_video_indices, active_delta)
        keep = None
        if detail_positions is not None and detail_positions.numel():
            if detail_delta is None or detail_delta.shape[0] != detail_positions.numel():
                raise ValueError("MLP detail positions and exact deltas must align")
            detail_indices = self.inactive_video_indices.index_select(
                0, detail_positions
            )
            value.index_add_(0, detail_indices, detail_delta)
            keep = torch.ones(
                self.inactive_video_indices.numel(),
                dtype=torch.bool,
                device=value.device,
            )
            keep[detail_positions] = False
        chunk_rows = 8192
        for start in range(0, int(self.inactive_video_indices.numel()), chunk_rows):
            stop = min(start + chunk_rows, int(self.inactive_video_indices.numel()))
            left = active_delta.index_select(0, self.left_active_positions[start:stop])
            right = active_delta.index_select(0, self.right_active_positions[start:stop])
            weight = self.right_weights[start:stop].to(value.dtype).unsqueeze(1)
            update = torch.lerp(left, right, weight)
            target = self.inactive_video_indices[start:stop]
            if keep is not None:
                selected = keep[start:stop]
                target = target[selected]
                update = update[selected]
            value.index_add_(0, target, update)


class MLPSpatialLatticePlan:
    def __init__(
        self,
        layout: PackedLayout,
        config: MLPSpatialLatticeConfig,
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device
        video = layout.segment("video", last=True)
        self.protected_tokens = int(video.start)
        self.latent_frames = int(layout.signature[1])
        latent_height = int(layout.signature[2])
        latent_width = int(layout.signature[3])
        if latent_height % 2 or latent_width % 2:
            raise ValueError("MLP spatial lattice requires even latent geometry")
        self.rows = latent_height // 2
        self.columns = latent_width // 2
        self.frame_tokens = self.rows * self.columns
        if video.length != self.latent_frames * self.frame_tokens:
            raise ValueError("MLP spatial lattice does not match packed video geometry")
        self._phases: dict[int, MLPSpatialLatticeLayerPlan] = {}

    def for_layer(self, layer: int) -> MLPSpatialLatticeLayerPlan | None:
        if not self.config.sparse_at(layer):
            return None
        phase = (
            layer - self.config.layer_start + self.config.phase_offset
        ) % self.config.stride
        cached = self._phases.get(phase)
        if cached is not None:
            return cached

        active_local: list[int] = []
        inactive_local: list[int] = []
        left_positions: list[int] = []
        right_positions: list[int] = []
        right_weights: list[float] = []
        active_lookup: dict[int, int] = {}

        # Select complete spatial columns.  Unlike a flattened token lattice,
        # interpolation never crosses a row or temporal boundary.
        for frame in range(self.latent_frames):
            for row in range(self.rows):
                line_start = (frame * self.rows + row) * self.columns
                line_active = [
                    line_start + column
                    for column in range(self.columns)
                    if column % self.config.stride == phase
                ]
                if not line_active:
                    # Very narrow research shapes retain one exact rail.
                    line_active = [line_start + min(phase, self.columns - 1)]
                for local in line_active:
                    active_lookup[local] = len(active_local)
                    active_local.append(local)
                for column in range(self.columns):
                    local = line_start + column
                    if local in active_lookup:
                        continue
                    left_candidates = [item for item in line_active if item < local]
                    right_candidates = [item for item in line_active if item > local]
                    left = left_candidates[-1] if left_candidates else line_active[0]
                    right = right_candidates[0] if right_candidates else line_active[-1]
                    inactive_local.append(local)
                    left_positions.append(active_lookup[left])
                    right_positions.append(active_lookup[right])
                    right_weights.append(
                        0.0 if left == right else float(local - left) / float(right - left)
                    )

        offset = self.protected_tokens
        cached = MLPSpatialLatticeLayerPlan(
            protected_tokens=offset,
            active_video_indices=torch.tensor(
                [offset + item for item in active_local], dtype=torch.long, device=self.device
            ),
            inactive_video_indices=torch.tensor(
                [offset + item for item in inactive_local], dtype=torch.long, device=self.device
            ),
            left_active_positions=torch.tensor(
                left_positions, dtype=torch.long, device=self.device
            ),
            right_active_positions=torch.tensor(
                right_positions, dtype=torch.long, device=self.device
            ),
            right_weights=torch.tensor(right_weights, dtype=torch.float32, device=self.device),
        )
        self._phases[phase] = cached
        return cached


_CONFIG: contextvars.ContextVar[MLPSpatialLatticeConfig | None] = contextvars.ContextVar(
    "h3_mlp_spatial_lattice_config", default=None
)
_PLAN: contextvars.ContextVar[MLPSpatialLatticePlan | None] = contextvars.ContextVar(
    "h3_mlp_spatial_lattice_plan", default=None
)


@contextlib.contextmanager
def mlp_spatial_lattice_config(config: MLPSpatialLatticeConfig | None):
    token = _CONFIG.set(config)
    try:
        yield
    finally:
        _CONFIG.reset(token)


def current_mlp_spatial_lattice_config() -> MLPSpatialLatticeConfig | None:
    return _CONFIG.get()


@contextlib.contextmanager
def mlp_spatial_lattice_plan(plan: MLPSpatialLatticePlan | None):
    token = _PLAN.set(plan)
    try:
        yield
    finally:
        _PLAN.reset(token)


def current_mlp_spatial_lattice_layer(
    layer: int | None,
) -> MLPSpatialLatticeLayerPlan | None:
    plan = _PLAN.get()
    if plan is None or layer is None:
        return None
    return plan.for_layer(layer)


__all__ = [
    "MLPSpatialLatticeConfig",
    "MLPSpatialLatticePlan",
    "current_mlp_spatial_lattice_config",
    "current_mlp_spatial_lattice_layer",
    "mlp_spatial_lattice_config",
    "mlp_spatial_lattice_plan",
]
