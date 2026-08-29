import unittest

import torch

from h3serve.native_engine.terminal_latent_guard import (
    stabilize_terminal_video_latent_,
)


def _stable_latent() -> torch.Tensor:
    generator = torch.Generator("cpu").manual_seed(7)
    base = torch.randn((1, 4, 17, 20, 12), generator=generator)
    # Matching temporal phases should have similar spatial energy while still
    # carrying different content.
    for index in range(5, base.shape[2]):
        base[:, :, index].mul_(0.98 + 0.01 * (index % 5))
    return base


class TerminalLatentGuardTests(unittest.TestCase):
    def test_normal_latent_is_bit_identical(self):
        value = _stable_latent()
        before = value.clone()
        report = stabilize_terminal_video_latent_(value)
        self.assertFalse(report["triggered"])
        self.assertTrue(torch.equal(value, before))

    def test_localized_terminal_bottom_collapse_is_repaired(self):
        value = _stable_latent()
        split = round(value.shape[-2] * 0.52)
        value[:, :, -1, split:].mul_(0.35)
        before = value.clone()
        report = stabilize_terminal_video_latent_(value)
        self.assertTrue(report["triggered"])
        self.assertEqual(report["repair"], "adaptive_same_phase_linear_feather")
        self.assertGreater(report["repair_strength"], 0.0)
        self.assertLessEqual(report["repair_strength"], 1.0)
        self.assertTrue(torch.equal(value[:, :, -1, :split], before[:, :, -1, :split]))
        self.assertGreater(
            value[:, :, -1, -2:].std(unbiased=False),
            before[:, :, -1, -2:].std(unbiased=False),
        )

    def test_persistent_flat_bottom_is_not_treated_as_terminal_failure(self):
        value = _stable_latent()
        split = round(value.shape[-2] * 0.52)
        value[..., split:, :].mul_(0.35)
        before = value.clone()
        report = stabilize_terminal_video_latent_(value)
        self.assertFalse(report["triggered"])
        self.assertTrue(torch.equal(value, before))

    def test_global_terminal_collapse_fails_closed(self):
        value = _stable_latent()
        value[:, :, -1].mul_(0.35)
        before = value.clone()
        report = stabilize_terminal_video_latent_(value)
        self.assertFalse(report["triggered"])
        self.assertIn(report["reason"], {"ratio_not_outlier", "top_region_not_stable"})
        self.assertTrue(torch.equal(value, before))


if __name__ == "__main__":
    unittest.main()
