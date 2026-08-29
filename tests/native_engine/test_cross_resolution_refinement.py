from __future__ import annotations

import unittest

import torch

from h3serve.native_engine.hot_session import (
    resize_refinement_video_latent_spatial,
    spatial_highpass_noise,
)


class CrossResolutionRefinementTests(unittest.TestCase):
    def test_resize_changes_only_spatial_axes(self) -> None:
        latent = torch.stack(
            tuple(torch.full((1, 2, 2), float(index)) for index in range(3)),
            dim=1,
        ).unsqueeze(0)
        resized = resize_refinement_video_latent_spatial(
            latent,
            target_height=4,
            target_width=6,
        )
        self.assertEqual(tuple(resized.shape), (1, 1, 3, 4, 6))
        for index in range(3):
            expected = torch.full((4, 6), float(index))
            torch.testing.assert_close(resized[0, 0, index], expected)

    def test_same_geometry_preserves_tensor_identity(self) -> None:
        latent = torch.randn(1, 2, 3, 4, 5)
        resized = resize_refinement_video_latent_spatial(
            latent,
            target_height=4,
            target_width=5,
        )
        self.assertIs(resized, latent)

    def test_invalid_rank_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "B,C,T,H,W"):
            resize_refinement_video_latent_spatial(
                torch.zeros(1, 2, 3, 4),
                target_height=4,
                target_width=6,
            )

    def test_highpass_has_unit_detail_variance(self) -> None:
        noise = torch.randn(1, 2, 3, 8, 10)
        highpass = spatial_highpass_noise(
            noise,
            low_height=4,
            low_width=5,
        )
        self.assertEqual(tuple(highpass.shape), tuple(noise.shape))
        variance = highpass.square().mean(dim=(-2, -1))
        torch.testing.assert_close(variance, torch.ones_like(variance))


if __name__ == "__main__":
    unittest.main()
