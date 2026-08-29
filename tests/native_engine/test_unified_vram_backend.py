from __future__ import annotations

import unittest

from scripts.validate_unified_vram_backend import build_report


class UnifiedVramProductEnvelopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_report()

    def test_all_required_16_and_24_gib_cases_fit(self) -> None:
        self.assertEqual(self.report["summary"]["required_failed"], 0)
        self.assertGreaterEqual(self.report["summary"]["required_cases"], 200)

    def test_16gib_1080p15_uses_compact_full_context(self) -> None:
        matches = [
            row for row in self.report["rows"]
            if row["required"]
            and row["profile"] == "int8_16gb"
            and row["workload"] == "second_sampling"
            and row["resolution"] == "1080p"
            and row["requested_duration_seconds"] == 15.0
        ]
        self.assertTrue(matches)
        self.assertTrue(all(row["fits_budget"] for row in matches))
        self.assertTrue(any(row["compact_kv"] for row in matches))
        self.assertTrue(any(row["vae_output_strategy"] == "host_temporal_exact" for row in matches))

    def test_24gib_2k15_uses_throughput_temporal_windows(self) -> None:
        matches = [
            row for row in self.report["rows"]
            if row["required"]
            and row["profile"] == "int8_24gb"
            and row["workload"] == "second_sampling"
            and row["resolution"] == "2k"
            and row["requested_duration_seconds"] == 15.0
        ]
        self.assertTrue(matches)
        self.assertTrue(all(row["fits_budget"] for row in matches))
        self.assertTrue(any(row["temporal_pieces"] >= 2 for row in matches))
        self.assertTrue(all(row["predicted_peak_gib"] <= 23.25 for row in matches))

    def test_8gib_1080p15_second_sampling_uses_temporal_windows(self) -> None:
        matches = [
            row for row in self.report["rows"]
            if row["required"]
            and row["profile"] == "w4a8_8gb"
            and row["workload"] == "second_sampling"
            and row["resolution"] == "1080p"
            and row["requested_duration_seconds"] == 15.0
        ]
        self.assertTrue(matches)
        self.assertTrue(all(row["fits_budget"] for row in matches))
        self.assertTrue(all(row["temporal_pieces"] >= 2 for row in matches))
        self.assertTrue(all(row["predicted_peak_gib"] <= 6.5 for row in matches))


if __name__ == "__main__":
    unittest.main()
