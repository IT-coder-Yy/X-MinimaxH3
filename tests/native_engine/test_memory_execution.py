from __future__ import annotations

import unittest

from h3serve.native_engine.planner import (
    H3WorkloadAnalyzer,
    select_memory_execution,
)


class MemoryExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = H3WorkloadAnalyzer()

    def features(self, width: int, height: int, frames: int):
        return self.analyzer.analyze(
            width=width,
            height=height,
            frames=frames,
            text_tokens=512,
            condition_count=0,
            engine="original",
            actual_evaluations=9,
            forecast_evaluations=11,
        )

    def test_24gib_720p15_uses_measured_exact_streaming_winner(self) -> None:
        decision = select_memory_execution(
            self.features(1280, 736, 362),
            requested_mode="auto",
            device_budget_bytes=23 * 1024**3,
        )
        self.assertEqual(decision.selected_scheme, "exact_streaming")
        self.assertEqual(decision.query_chunk_tokens, 49_152)
        self.assertFalse(decision.compact_kv)
        self.assertEqual(decision.mlp_chunk_tokens, 4096)
        self.assertEqual(decision.resource_profile, "int8_24gb")
        self.assertEqual(decision.vae_spatial_tile, 288)
        self.assertIsNone(decision.vae_temporal_tile)
        self.assertTrue(decision.fits_budget)

    def test_16gib_720p15_uses_same_fast_exact_graph(self) -> None:
        decision = select_memory_execution(
            self.features(1280, 736, 362),
            requested_mode="auto",
            # Simulate the 16GB backend on a physical 24GB development GPU;
            # the explicit backend must still enforce its own admission cap.
            device_budget_bytes=23 * 1024**3,
            resource_profile="int8_16gb",
        )
        self.assertEqual(decision.selected_scheme, "exact_streaming")
        self.assertEqual(decision.query_chunk_tokens, 49_152)
        self.assertEqual(decision.mlp_chunk_tokens, 4096)
        self.assertFalse(decision.compact_kv)
        self.assertEqual(decision.resource_profile, "int8_16gb")
        self.assertLessEqual(decision.device_budget_bytes, int(15.25 * 1024**3))
        self.assertTrue(decision.fits_budget)

    def test_legacy_modes_cannot_change_the_unified_plan(self) -> None:
        decisions = [
            select_memory_execution(
                self.features(864, 480, 124),
                requested_mode=mode,
                device_budget_bytes=23 * 1024**3,
            )
            for mode in ("auto", "performance", "low_vram")
        ]
        signatures = {
            (
                item.selected_scheme,
                item.query_chunk_tokens,
                item.compact_kv,
                item.block_buffer_count,
                item.mlp_chunk_tokens,
                item.vae_spatial_tile,
            )
            for item in decisions
        }
        self.assertEqual(len(signatures), 1)
        self.assertEqual(decisions[0].selected_scheme, "exact_streaming")
        self.assertFalse(decisions[0].telemetry()["legacy_mode_ignored"])
        self.assertTrue(decisions[1].telemetry()["legacy_mode_ignored"])

    def test_compact_streaming_admits_estimated_2k15_on_24gib(self) -> None:
        decision = select_memory_execution(
            self.features(2560, 1440, 362),
            requested_mode="auto",
            device_budget_bytes=23 * 1024**3,
        )
        self.assertEqual(decision.selected_scheme, "compact_streaming")
        self.assertTrue(decision.compact_kv)
        self.assertEqual(decision.query_chunk_tokens, 8192)
        self.assertTrue(decision.fits_budget)
        receipt = decision.telemetry()
        self.assertEqual(
            receipt["numerical_contract"], "compact_quantized_full_context"
        )
        self.assertFalse(receipt["bit_exact"])
        self.assertTrue(receipt["weights_steps_schedule_unchanged"])

    def test_480p15_uses_single_buffer_to_cross_8gib_boundary(self) -> None:
        decision = select_memory_execution(
            self.features(864, 480, 362),
            requested_mode="auto",
            device_budget_bytes=7 * 1024**3,
        )
        self.assertEqual(decision.selected_scheme, "compact_streaming")
        self.assertTrue(decision.compact_kv)
        self.assertEqual(decision.block_buffer_count, 1)
        self.assertTrue(decision.fits_budget)

    def test_w4a8_720p15_uses_physically_anchored_single_buffer_route(self) -> None:
        decision = select_memory_execution(
            self.features(1280, 736, 362),
            requested_mode="auto",
            device_budget_bytes=int(7.25 * 1024**3),
            weight_tier="w4a8",
        )
        self.assertEqual(decision.selected_scheme, "compact_streaming")
        self.assertTrue(decision.compact_kv)
        self.assertEqual(decision.query_chunk_tokens, 4096)
        self.assertEqual(decision.block_buffer_count, 1)
        self.assertEqual(decision.projection_chunk_tokens, 4096)
        self.assertEqual(decision.mlp_chunk_tokens, 2048)
        self.assertTrue(decision.fits_budget)
        receipt = decision.telemetry()
        self.assertEqual(receipt["resource_profile"], "w4a8_8gb")
        self.assertEqual(receipt["base_weight_quantization"], "grouped_w4a8_convrot")
        self.assertEqual(
            receipt["numerical_contract"],
            "w4a8_compact_quantized_full_context",
        )
        self.assertFalse(receipt["bit_exact"])

    def test_1080p15_keeps_copy_compute_overlap_on_16gib_boundary(self) -> None:
        decision = select_memory_execution(
            self.features(1920, 1088, 362),
            requested_mode="auto",
            device_budget_bytes=15 * 1024**3,
        )
        self.assertEqual(decision.selected_scheme, "compact_streaming")
        self.assertTrue(decision.compact_kv)
        self.assertEqual(decision.block_buffer_count, 2)
        self.assertIsNotNone(decision.vae_temporal_tile)
        self.assertEqual(
            decision.telemetry()["vae_output_strategy"], "host_temporal_exact"
        )
        self.assertTrue(decision.fits_budget)

    def test_1080p15_exact_host_vae_model_matches_physical_gate(self) -> None:
        from h3serve.native_engine.planner import (
            estimate_vae_host_streaming_peak_bytes,
            estimate_vae_materialized_peak_bytes,
        )

        features = self.features(1920, 1088, 362)
        host_peak = estimate_vae_host_streaming_peak_bytes(features) / 1024**3
        materialized_peak = (
            estimate_vae_materialized_peak_bytes(features) / 1024**3
        )
        self.assertGreaterEqual(host_peak, 8.50)
        self.assertLess(host_peak, 9.0)
        self.assertGreaterEqual(materialized_peak, 16.95)
        self.assertLess(materialized_peak, 17.5)

    def test_compact_peak_model_tracks_real_block_plus_service_envelope(self) -> None:
        from h3serve.native_engine.planner import (
            estimate_compact_streaming_peak_bytes,
        )

        measured = (
            ((864, 480, 362), 2.848 + 4.05),
            ((1920, 1088, 362), 10.625 + 4.05),
            ((2560, 1440, 362), 18.308 + 4.05),
        )
        for geometry, measured_gib in measured:
            with self.subTest(geometry=geometry):
                estimated = estimate_compact_streaming_peak_bytes(
                    self.features(*geometry), query_chunk_tokens=8192
                ) / (1024**3)
                self.assertLess(abs(estimated - measured_gib), 0.35)


if __name__ == "__main__":
    unittest.main()
