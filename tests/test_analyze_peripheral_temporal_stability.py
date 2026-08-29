from __future__ import annotations

import unittest

import numpy as np

from scripts.analyze_peripheral_temporal_stability import analyze_frames


class PeripheralTemporalStabilityTests(unittest.TestCase):
    def test_border_flash_is_detected_above_smooth_luminance_change(self):
        frames = np.stack([
            np.full((40, 64), 80 + 2 * index, dtype=np.uint8)
            for index in range(9)
        ])
        smooth = analyze_frames(frames)
        flashed = frames.copy()
        flashed[4, 2:10, 2:12] = 255
        observed = analyze_frames(flashed)
        self.assertGreater(
            observed["positive_luminance_impulse_p99"]["max"],
            smooth["positive_luminance_impulse_p99"]["max"],
        )
        self.assertIn(4, observed["bright_outlier_frame_indices"])

    def test_invalid_contract_fails_closed(self):
        with self.assertRaises(ValueError):
            analyze_frames(np.zeros((2, 8, 8), dtype=np.uint8))
        with self.assertRaises(ValueError):
            analyze_frames(np.zeros((3, 8, 8), dtype=np.uint8), border_fraction=0.01)


if __name__ == "__main__":
    unittest.main()
