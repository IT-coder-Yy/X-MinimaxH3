from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TargetLayout:
    audio_rows: int
    video_rows: int
    latent_t: int
    grid_h: int
    grid_w: int


def target_layout(layout: Any) -> TargetLayout:
    """Resolve the packed MiniMax H3 [audio | video] target layout."""
    segments = tuple(getattr(layout, "segments"))
    audio = [(int(a), int(b)) for a, b, kind in segments if kind == "audio"]
    video = [(int(a), int(b)) for a, b, kind in segments if kind == "video"]
    if len(audio) != 1 or len(video) != 1:
        raise RuntimeError("expected one target audio and one target video segment")
    aa, ab = audio[0]
    va, vb = video[0]
    if ab != va:
        raise RuntimeError("expected contiguous [audio | video] target rows")
    signature = tuple(int(value) for value in getattr(layout, "signature"))
    if len(signature) != 5:
        raise RuntimeError("invalid MiniMax H3 layout signature")
    _, latent_t, latent_h, latent_w, _ = signature
    info = TargetLayout(ab - aa, vb - va, latent_t, latent_h // 2, latent_w // 2)
    if info.video_rows != info.latent_t * info.grid_h * info.grid_w:
        raise RuntimeError("video target rows do not match the packed H3 grid")
    return info
