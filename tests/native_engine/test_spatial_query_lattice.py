from __future__ import annotations

import unittest

import torch

from h3serve.native_engine.model.packed import build_fl2va_layout
from h3serve.native_engine.model.spatial_query_lattice import (
    SpatialQueryLatticeConfig,
    SpatialQueryLatticePlan,
)


class SpatialQueryLatticePlanTest(unittest.TestCase):
    def setUp(self) -> None:
        # 23x40 is the 720p packed spatial grid used by the long-video target.
        self.layout = build_fl2va_layout(
            text_length=3,
            latent_frames=7,
            latent_height=46,
            latent_width=80,
            audio_frames=2,
        )

    def test_rotates_native_query_blocks_without_repacking_them(self) -> None:
        plan = SpatialQueryLatticePlan(
            self.layout,
            SpatialQueryLatticeConfig(stride=2, query_block_rows=128),
            torch.device("cpu"),
        )
        even = plan.for_layer(0)
        odd = plan.for_layer(1)
        assert even is not None and odd is not None
        even_local = even.active_video_indices - plan.protected_tokens
        odd_local = odd.active_video_indices - plan.protected_tokens
        self.assertTrue(torch.all(even_local[:-1] < even_local[1:]))
        self.assertTrue(torch.all(odd_local[:-1] < odd_local[1:]))
        self.assertEqual(set(even_local.tolist()) & set(odd_local.tolist()), set())
        self.assertEqual(
            set(even_local.tolist()) | set(odd_local.tolist()),
            set(range(plan.video_tokens)),
        )
        # Every non-terminal selected run begins at the original 128-row
        # boundary. Concatenating the runs therefore preserves Sparge pooling.
        starts = [
            int(even_local[index])
            for index in range(even_local.numel())
            if index == 0 or int(even_local[index - 1]) + 1 != int(even_local[index])
        ]
        self.assertTrue(all(start % 128 == 0 for start in starts))

    def test_phase_rotates_across_step_offset_and_honors_dense_layers(self) -> None:
        base = SpatialQueryLatticePlan(
            self.layout,
            SpatialQueryLatticeConfig(stride=2, dense_layers=(3,)),
            torch.device("cpu"),
        )
        shifted = SpatialQueryLatticePlan(
            self.layout,
            SpatialQueryLatticeConfig(stride=2, phase_offset=1),
            torch.device("cpu"),
        )
        self.assertTrue(
            torch.equal(
                base.for_layer(0).active_video_indices,
                shifted.for_layer(1).active_video_indices,
            )
        )
        self.assertIsNone(base.for_layer(3))

    def test_reconstruction_updates_every_inactive_row_without_crossing_frames(self) -> None:
        plan = SpatialQueryLatticePlan(
            self.layout,
            SpatialQueryLatticeConfig(stride=2),
            torch.device("cpu"),
        )
        layer = plan.for_layer(0)
        assert layer is not None
        value = torch.zeros(self.layout.sequence_length, 1)
        before = value.index_select(0, layer.active_video_indices)
        # Give each frame's exact rails a unique residual. If reconstruction
        # crossed frames, at least one inactive row would receive another ID.
        for frame in range(plan.latent_frames):
            start = plan.protected_tokens + frame * plan.frame_tokens
            stop = start + plan.frame_tokens
            mask = (layer.active_video_indices >= start) & (
                layer.active_video_indices < stop
            )
            value.index_fill_(
                0, layer.active_video_indices[mask], float(frame + 1)
            )
        layer.reconstruct_inactive_(value, before)
        video = value[plan.protected_tokens :].view(plan.latent_frames, -1)
        for frame in range(plan.latent_frames):
            self.assertTrue(torch.allclose(video[frame], torch.full_like(video[frame], frame + 1)))

    def test_invalid_ranges_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            SpatialQueryLatticeConfig(stride=0)
        with self.assertRaises(ValueError):
            SpatialQueryLatticeConfig(layer_start=10, layer_stop=9)


if __name__ == "__main__":
    unittest.main()
