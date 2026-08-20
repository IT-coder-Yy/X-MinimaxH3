"""Measured H3 Video-VAE tile selection from exact upstream tile geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VaeTileDecision:
    tile_size: int
    tile_rows: int
    tile_columns: int
    decoded_tile_pixels: int


def _axis_tile_plan(
    length: int,
    tile_size: int,
    *,
    overlap_min: int,
) -> tuple[int, int]:
    """Return tile count and summed decoded extent for one spatial axis."""

    if length <= 0 or tile_size <= 0 or overlap_min < 0:
        raise ValueError("VAE tile geometry must be positive")
    if tile_size >= length:
        return 1, length
    count = math.ceil(length / tile_size)
    while tile_size * count - overlap_min * (count - 1) < length:
        count += 1
    return count, count * tile_size


def select_vae_tile(
    *,
    width: int,
    height: int,
    candidates: tuple[int, ...] = (256, 288),
    overlap_min: int = 64,
    vae_ratio: int = 16,
) -> VaeTileDecision:
    """Select the least-work tile among visually accepted candidates.

    ``decoded_tile_pixels`` exactly counts the spatial area submitted to the
    VAE decoder across the upstream overlap grid.  It explains the measured
    360p/480p/720p crossover and intentionally excludes rejected tile 384.
    """

    if width <= 0 or height <= 0:
        raise ValueError("VAE canvas must be positive")
    if not candidates:
        raise ValueError("at least one measured VAE tile candidate is required")
    decisions: list[VaeTileDecision] = []
    for tile_size in candidates:
        if tile_size < 128 or tile_size % vae_ratio:
            raise ValueError("VAE tile candidates must be >=128 and VAE-ratio aligned")
        tile_rows, decoded_height = _axis_tile_plan(
            height, tile_size, overlap_min=overlap_min
        )
        tile_columns, decoded_width = _axis_tile_plan(
            width, tile_size, overlap_min=overlap_min
        )
        decisions.append(
            VaeTileDecision(
                tile_size=tile_size,
                tile_rows=tile_rows,
                tile_columns=tile_columns,
                decoded_tile_pixels=decoded_height * decoded_width,
            )
        )
    return min(
        decisions,
        key=lambda decision: (decision.decoded_tile_pixels, decision.tile_size),
    )


__all__ = ["VaeTileDecision", "select_vae_tile"]
