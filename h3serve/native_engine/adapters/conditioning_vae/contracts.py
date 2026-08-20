"""Narrow values exchanged by H3 conditioning and VAE adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


KeyframeRole = Literal["first", "last"]


@dataclass(frozen=True, slots=True)
class PreparedKeyframe:
    """One RGB keyframe on the already-resolved target canvas."""

    role: KeyframeRole
    semantic_frame_index: int
    resolved_frame_index: int
    image: Any

    @property
    def width(self) -> int:
        return int(self.image.size[0])

    @property
    def height(self) -> int:
        return int(self.image.size[1])


@dataclass(frozen=True, slots=True)
class KeyframeCondition:
    """One normalized and patchified video-VAE anchor."""

    role: KeyframeRole
    semantic_frame_index: int
    resolved_frame_index: int
    latent: Any
    rows: Any
    latent_height: int
    latent_width: int


@dataclass(frozen=True, slots=True)
class FrameConditioning:
    """Ordered FL2VA anchors consumed by the packed DiT adapter."""

    rows: Any
    keyframes: tuple[KeyframeCondition, ...]
    semantic_frame_indices: tuple[int, ...]
    resolved_frame_indices: tuple[int, ...]
    frame_count: int


@dataclass(frozen=True, slots=True)
class ReferenceConditioning:
    """Ordered Ref2VA image/video latents and their native geometry."""

    latents: tuple[Any, ...]
    latent_shapes: tuple[tuple[int, int, int], ...]
    kinds: tuple[Literal["image", "video"], ...]
    media: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class PreparedReferenceVideo:
    """One decoded 24-fps Ref2VA clip; embedded audio is intentionally absent."""

    frames: Any
    qwen_frames: Any
    qwen_block_timestamps: tuple[float, ...]
    source_fps: float
    source_duration_seconds: float


@dataclass(frozen=True, slots=True)
class PreparedReferenceAudio:
    """One decoded stereo Ref2VA reference clip at the H3 VAE sample rate."""

    waveform: Any
    sample_rate: int
    source_duration_seconds: float


# Compatibility name for downstream imports from the image-only release.
ReferenceImageConditioning = ReferenceConditioning


@dataclass(frozen=True, slots=True)
class TextConditioning:
    """Layer-50 Qwen3-VL representation and aligned modality tags."""

    hidden_states: Any
    token_tags: Any
    input_ids: Any
    keyframes: tuple[PreparedKeyframe, ...]

    @property
    def text_length(self) -> int:
        return int(self.input_ids.shape[0])
