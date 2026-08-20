"""Bounded single-GPU batching for the official H3 Video-VAE tile loop.

The upstream implementation offers either one tile per decoder invocation or
all spatial tiles in one batch.  Neither policy is a good fit for a 24 GiB
card across the full 360p--720p workload envelope.  This adapter retains the
upstream tile split, order and blending math while limiting each decoder batch
to a measured number of tiles.
"""

from __future__ import annotations

from types import MethodType
from typing import Any


def install_bounded_tile_batching(model: Any, tile_batch_size: int = 1) -> Any:
    """Install a project-owned bounded replacement for ``_run_tile_tasks``.

    A size of one preserves the upstream sequential path.  Sizes above one
    activate ``stack_tiling`` and concatenate at most that many same-shaped
    spatial tiles along the existing sample-batch dimension.  The returned
    tile list is in exactly the same order expected by upstream blending.
    """

    if isinstance(tile_batch_size, bool) or tile_batch_size <= 0:
        raise ValueError("VAE tile batch size must be a positive integer")
    if not callable(getattr(model, "_run_tile_tasks", None)):
        raise TypeError("the H3 Video-VAE does not expose _run_tile_tasks")

    def _run_bounded_tile_tasks(
        self,
        tiles,
        tile_indices,
        forward_fn,
        stack_tiling,
        cls_agg=None,
    ):
        if not stack_tiling or not tile_indices:
            tasks = []
            for index in tile_indices:
                tasks.append(forward_fn(tiles[index]))
                if cls_agg is not None:
                    cls_agg.collect()
            return tasks

        limit = int(self.native_tile_batch_size)
        tasks = []
        for start in range(0, len(tile_indices), limit):
            chunk = tile_indices[start : start + limit]
            sample_batch_size = tiles[chunk[0]].shape[0]
            tile_batch = __import__("torch").cat(
                [tiles[index] for index in chunk], dim=0
            )
            output_batch = forward_fn(tile_batch)
            tasks.extend(
                output_batch.unflatten(
                    0, (len(chunk), sample_batch_size)
                ).unbind(dim=0)
            )
            if cls_agg is not None:
                cls_agg.collect_stacked(len(chunk), sample_batch_size)
        return tasks

    model._run_tile_tasks = MethodType(_run_bounded_tile_tasks, model)
    model.native_tile_batch_size = 1
    configure_vae_tile_batching(model, tile_batch_size)
    return model


def configure_vae_tile_batching(model: Any, tile_batch_size: int) -> None:
    """Change the bounded batch for one request without rebuilding the VAE."""

    if isinstance(tile_batch_size, bool) or tile_batch_size <= 0:
        raise ValueError("VAE tile batch size must be a positive integer")
    if not hasattr(model, "native_tile_batch_size"):
        raise TypeError("bounded VAE tile batching has not been installed")
    model.native_tile_batch_size = int(tile_batch_size)
    model.stack_tiling = tile_batch_size > 1


__all__ = ["configure_vae_tile_batching", "install_bounded_tile_batching"]
