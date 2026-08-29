from __future__ import annotations

import unittest

import torch

from h3serve.native_engine.latent_upscaler import (
    H3LatentResizer3D,
    upscale_h3_video_latent,
)


class LearnedH3LatentUpscalerTests(unittest.TestCase):
    def test_lightweight_graph_preserves_time_and_changes_only_spatial_shape(self) -> None:
        model = H3LatentResizer3D(
            channels=32,
            in_blocks=1,
            out_blocks=1,
            temporal_every=1,
            temporal_kernel=3,
        ).eval()
        source = torch.randn(1, 24, 2, 3, 4)
        result = upscale_h3_video_latent(
            model,
            source,
            target_height=6,
            target_width=8,
            temporal_chunk_frames=24,
        )
        self.assertEqual(tuple(result.shape), (1, 24, 2, 6, 8))
        self.assertEqual(result.dtype, source.dtype)
        self.assertTrue(torch.isfinite(result).all())

    def test_image_shaped_or_downscale_inputs_fail_closed(self) -> None:
        model = H3LatentResizer3D(
            channels=32,
            in_blocks=1,
            out_blocks=1,
            temporal_every=1,
            temporal_kernel=3,
        ).eval()
        with self.assertRaisesRegex(ValueError, "B,24,T,H,W"):
            upscale_h3_video_latent(
                model,
                torch.randn(1, 24, 4, 4),
                target_height=8,
                target_width=8,
            )
        with self.assertRaisesRegex(ValueError, "only supports upscaling"):
            upscale_h3_video_latent(
                model,
                torch.randn(1, 24, 2, 8, 8),
                target_height=4,
                target_width=4,
            )


if __name__ == "__main__":
    unittest.main()
