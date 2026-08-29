import unittest

import torch

from h3serve.native_engine.model.mlp_spatial_lattice import (
    MLPSpatialLatticeConfig,
    MLPSpatialLatticePlan,
)
from h3serve.native_engine.model.packed import build_fl2va_layout


class MLPSpatialLatticeTests(unittest.TestCase):
    def setUp(self):
        self.layout = build_fl2va_layout(
            text_length=4,
            latent_frames=3,
            latent_height=8,
            latent_width=12,
            audio_frames=2,
        )

    def test_rotating_columns_partition_each_frame(self):
        plan = MLPSpatialLatticePlan(
            self.layout,
            MLPSpatialLatticeConfig(stride=2),
            torch.device("cpu"),
        )
        layer = plan.for_layer(0)
        self.assertIsNotNone(layer)
        assert layer is not None
        combined = torch.cat(
            (layer.active_video_indices, layer.inactive_video_indices)
        )
        self.assertEqual(combined.unique().numel(), 72)

    def test_detail_rows_override_interpolated_updates(self):
        plan = MLPSpatialLatticePlan(
            self.layout,
            MLPSpatialLatticeConfig(stride=3, detail_fraction=0.2),
            torch.device("cpu"),
        )
        layer = plan.for_layer(0)
        assert layer is not None
        hidden = torch.randn(self.layout.sequence_length, 256)
        positions, indices = layer.select_detail_positions(hidden, 0.2)
        value = torch.zeros_like(hidden)
        active_delta = torch.ones(layer.active_video_indices.numel(), 256)
        detail_delta = torch.full((indices.numel(), 256), 2.0)
        layer.reconstruct_(
            value,
            active_delta,
            detail_positions=positions,
            detail_delta=detail_delta,
        )
        self.assertTrue(torch.all(value[indices] == 2.0))

    def test_stride_one_is_disabled(self):
        plan = MLPSpatialLatticePlan(
            self.layout,
            MLPSpatialLatticeConfig(stride=1),
            torch.device("cpu"),
        )
        self.assertIsNone(plan.for_layer(12))


if __name__ == "__main__":
    unittest.main()
