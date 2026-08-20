"""Training-free frame-interleaved execution for packed H3 AV tokens.

The policy keeps every text, visual-condition and audio token active.  Only
the final generated-video segment is sampled along its latent-frame axis.
Selected frame slices are processed by the real transformer block and the
remaining slices are reconstructed by local linear interpolation.  Rotating
the anchor phase across layers ensures that every latent frame is refreshed.

This module owns only structural routing.  It does not change weights,
sampler steps or the audio/video latent clocks.
"""

from __future__ import annotations

import bisect
import contextlib
import contextvars
from dataclasses import dataclass

import torch

from .packed import PackedLayout


@dataclass(frozen=True, slots=True)
class FrameInterleaveConfig:
    """One request-step policy; ``stride=1`` means full computation."""

    stride: int = 1
    layer_start: int = 0
    layer_stop: int = 50
    dense_layers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.stride <= 0:
            raise ValueError("frame interleave stride must be positive")
        if not 0 <= self.layer_start <= self.layer_stop <= 50:
            raise ValueError("frame interleave layer range must lie inside [0, 50]")
        if tuple(sorted(set(self.dense_layers))) != self.dense_layers:
            raise ValueError("dense frame-interleave layers must be sorted and unique")
        if any(layer < 0 or layer >= 50 for layer in self.dense_layers):
            raise ValueError("dense frame-interleave layer falls outside [0, 50)")

    def sparse_at(self, layer: int) -> bool:
        return (
            self.stride > 1
            and self.layer_start <= layer < self.layer_stop
            and layer not in self.dense_layers
        )


@dataclass(frozen=True, slots=True)
class FrameInterleaveLayerPlan:
    """Device tensors needed by one rotating anchor phase."""

    selected_indices: torch.Tensor
    anchor_frames: torch.Tensor
    left_anchor_positions: torch.Tensor
    right_anchor_positions: torch.Tensor
    right_weights: torch.Tensor
    protected_tokens: int
    latent_frames: int
    frame_tokens: int

    def selected_modulation_segments(
        self,
        segments: tuple[tuple[int, int, int], ...],
    ) -> tuple[tuple[int, int, int], ...]:
        """Map full packed AdaLN runs to ``[prefix | selected video]``."""

        result: list[tuple[int, int, int]] = []
        for start, stop, row in segments:
            if stop <= self.protected_tokens:
                result.append((start, stop, row))
                continue
            if start != self.protected_tokens:
                raise ValueError(
                    "frame interleave requires the generated video to be the final run"
                )
            result.append(
                (
                    self.protected_tokens,
                    int(self.selected_indices.numel()),
                    row,
                )
            )
        if not result or result[-1][1] != int(self.selected_indices.numel()):
            raise ValueError("selected modulation runs do not cover selected tokens")
        return tuple(result)

    def selected_frequencies(self, frequencies: torch.Tensor) -> torch.Tensor:
        full_tokens = self.protected_tokens + self.latent_frames * self.frame_tokens
        if frequencies.ndim != 6 or frequencies.shape[1] != full_tokens:
            raise ValueError("unexpected packed RoPE table for frame interleave")
        return frequencies.index_select(1, self.selected_indices)

    def reconstruct_(
        self,
        full_value: torch.Tensor,
        selected_input: torch.Tensor,
        selected_output: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter protected rows and interpolate the block's video update.

        Interpolating absolute hidden states destroys the unselected frame's
        spatial/motion identity.  The transformer is residual, so retain each
        frame's incoming state and approximate only this block's update.
        """

        expected_selected = self.protected_tokens + int(
            self.anchor_frames.numel()
        ) * self.frame_tokens
        if selected_input.shape != selected_output.shape:
            raise ValueError("frame-interleave input/output shapes do not match")
        if selected_output.shape[0] != expected_selected:
            raise ValueError("selected frame-interleave output has the wrong length")
        if full_value.shape[0] != self.protected_tokens + self.latent_frames * self.frame_tokens:
            raise ValueError("full frame-interleave residual has the wrong length")

        full_value[: self.protected_tokens].copy_(
            selected_output[: self.protected_tokens]
        )
        # Reuse selected_output as the delta buffer after the exact protected
        # rows have been published. This avoids another ~0.5 GiB allocation.
        selected_output[self.protected_tokens :].sub_(
            selected_input[self.protected_tokens :]
        )
        anchor_updates = selected_output[self.protected_tokens :].view(
            int(self.anchor_frames.numel()), self.frame_tokens, selected_output.shape[-1]
        )
        # Reconstruct in small temporal tiles.  Expanding left/right anchors for
        # every 720p frame at once creates three roughly 1 GiB temporaries and
        # defeats the purpose of reducing the block working set on a 4090.
        # Eight frames keeps the temporary below ~230 MiB at H3's 920
        # tokens/frame and 5376 hidden width while retaining vectorized kernels.
        target = full_value[self.protected_tokens :].view(
            self.latent_frames, self.frame_tokens, full_value.shape[-1]
        )
        chunk_frames = 8
        for start in range(0, self.latent_frames, chunk_frames):
            stop = min(start + chunk_frames, self.latent_frames)
            left = anchor_updates.index_select(
                0, self.left_anchor_positions[start:stop]
            )
            right = anchor_updates.index_select(
                0, self.right_anchor_positions[start:stop]
            )
            weights = self.right_weights[start:stop].to(
                dtype=selected_output.dtype
            ).view(-1, 1, 1)
            target[start:stop].add_(torch.lerp(left, right, weights))
        return full_value


class FrameInterleavePlan:
    """Lazily materialized layer plans for one packed request layout."""

    def __init__(
        self,
        layout: PackedLayout,
        config: FrameInterleaveConfig,
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device
        video = layout.segment("video", last=True)
        self.protected_tokens = int(video.start)
        self.latent_frames = int(layout.signature[1])
        if self.latent_frames <= 1 or video.length % self.latent_frames:
            raise ValueError("target video rows do not form complete latent frames")
        self.frame_tokens = video.length // self.latent_frames
        self._phases: dict[int, FrameInterleaveLayerPlan] = {}

    def for_layer(self, layer: int) -> FrameInterleaveLayerPlan | None:
        if not self.config.sparse_at(layer):
            return None
        phase = (layer - self.config.layer_start) % self.config.stride
        cached = self._phases.get(phase)
        if cached is not None:
            return cached

        anchors = sorted(
            {0, self.latent_frames - 1}
            | {
                frame
                for frame in range(self.latent_frames)
                if frame % self.config.stride == phase
            }
        )
        anchor_lookup = {frame: index for index, frame in enumerate(anchors)}
        left_positions: list[int] = []
        right_positions: list[int] = []
        right_weights: list[float] = []
        for frame in range(self.latent_frames):
            insertion = bisect.bisect_left(anchors, frame)
            if insertion < len(anchors) and anchors[insertion] == frame:
                left = right = anchors[insertion]
            elif insertion == 0:
                left = right = anchors[0]
            elif insertion == len(anchors):
                left = right = anchors[-1]
            else:
                left, right = anchors[insertion - 1], anchors[insertion]
            left_positions.append(anchor_lookup[left])
            right_positions.append(anchor_lookup[right])
            right_weights.append(
                0.0 if left == right else float(frame - left) / float(right - left)
            )

        selected = list(range(self.protected_tokens))
        for frame in anchors:
            start = self.protected_tokens + frame * self.frame_tokens
            selected.extend(range(start, start + self.frame_tokens))
        cached = FrameInterleaveLayerPlan(
            selected_indices=torch.tensor(
                selected, dtype=torch.long, device=self.device
            ),
            anchor_frames=torch.tensor(
                anchors, dtype=torch.long, device=self.device
            ),
            left_anchor_positions=torch.tensor(
                left_positions, dtype=torch.long, device=self.device
            ),
            right_anchor_positions=torch.tensor(
                right_positions, dtype=torch.long, device=self.device
            ),
            right_weights=torch.tensor(
                right_weights, dtype=torch.float32, device=self.device
            ),
            protected_tokens=self.protected_tokens,
            latent_frames=self.latent_frames,
            frame_tokens=self.frame_tokens,
        )
        self._phases[phase] = cached
        return cached


_CONFIG: contextvars.ContextVar[FrameInterleaveConfig | None] = (
    contextvars.ContextVar("h3_frame_interleave_config", default=None)
)
_PLAN: contextvars.ContextVar[FrameInterleavePlan | None] = contextvars.ContextVar(
    "h3_frame_interleave_plan", default=None
)


@contextlib.contextmanager
def frame_interleave_config(config: FrameInterleaveConfig | None):
    token = _CONFIG.set(config)
    try:
        yield
    finally:
        _CONFIG.reset(token)


def current_frame_interleave_config() -> FrameInterleaveConfig | None:
    return _CONFIG.get()


@contextlib.contextmanager
def frame_interleave_plan(plan: FrameInterleavePlan | None):
    token = _PLAN.set(plan)
    try:
        yield
    finally:
        _PLAN.reset(token)


def current_frame_interleave_layer(layer: int | None) -> FrameInterleaveLayerPlan | None:
    plan = _PLAN.get()
    if plan is None or layer is None:
        return None
    return plan.for_layer(layer)


__all__ = [
    "FrameInterleaveConfig",
    "FrameInterleaveLayerPlan",
    "FrameInterleavePlan",
    "current_frame_interleave_config",
    "current_frame_interleave_layer",
    "frame_interleave_config",
    "frame_interleave_plan",
]
