from __future__ import annotations

import unittest

from h3serve.native_engine.planner import (
    CalibratedProfile,
    ExecutionPlan,
    H3WorkloadAnalyzer,
    LatencyModel,
    MemoryModel,
    NoFeasibleProfile,
    RTX4090Planner,
    review_combined_profiles_2026_08_12,
    review_fused_rms_profiles_2026_08_12,
    review_sparse_profiles_2026_08_12,
    select_vae_tile,
    validated_lora_profiles_2026_08_11,
    validated_original_profiles_2026_08_11,
    validated_profiles_for_engine,
)
from h3serve.native_engine.runtime import OffloadMode


class WorkloadAnalyzerTests(unittest.TestCase):
    def test_ref2va_full_public_condition_count_is_supported(self) -> None:
        features = H3WorkloadAnalyzer().analyze(
            width=1280,
            height=736,
            frames=362,
            text_tokens=512,
            condition_count=15,
            condition_tokens_override=12_000,
            engine="reference",
            actual_evaluations=5,
        )
        self.assertEqual(features.condition_count, 15)
        self.assertEqual(features.condition_tokens, 12_000)

    def setUp(self) -> None:
        self.analyzer = H3WorkloadAnalyzer()

    def test_exact_360p5_token_contract(self) -> None:
        features = self.analyzer.analyze(
            width=640,
            height=352,
            frames=124,
            text_tokens=1024,
            condition_count=0,
            engine="original",
            actual_evaluations=9,
            forecast_evaluations=11,
        )
        self.assertEqual(features.spatial_tokens, 220)
        self.assertEqual(features.latent_frames, 37)
        self.assertEqual(features.video_tokens, 8140)
        self.assertEqual(features.audio_tokens, 414)
        self.assertEqual(features.packed_tokens, 9578)

    def test_ref2va_dense_9_11_routes_through_original_family(self) -> None:
        features = self.analyzer.analyze(
            width=736, height=736, frames=124, text_tokens=1024,
            condition_count=5, condition_tokens_override=5200,
            engine="reference", actual_evaluations=9, forecast_evaluations=11,
        )
        decision = RTX4090Planner(
            validated_profiles_for_engine("reference")
        ).select(features, free_device_bytes=24 * 1024**3)
        self.assertEqual(decision.plan.offload_mode, OffloadMode.BLOCK)
        self.assertIn("original911", decision.profile_id)

    def test_ref2va_lora6_routes_through_lora_family(self) -> None:
        features = self.analyzer.analyze(
            width=736, height=736, frames=124, text_tokens=1024,
            condition_count=5, condition_tokens_override=5200,
            engine="reference", actual_evaluations=6, forecast_evaluations=0,
        )
        decision = RTX4090Planner(
            validated_profiles_for_engine("reference_lora")
        ).select(features, free_device_bytes=24 * 1024**3)
        self.assertEqual(decision.plan.offload_mode, OffloadMode.BLOCK)
        self.assertIn("lora6", decision.profile_id)

    def test_each_keyframe_adds_one_spatial_grid(self) -> None:
        base = self.analyzer.analyze(
            width=864,
            height=480,
            frames=73,
            text_tokens=512,
            condition_count=0,
            engine="original",
            actual_evaluations=9,
            forecast_evaluations=11,
        )
        anchored = self.analyzer.analyze(
            width=864,
            height=480,
            frames=73,
            text_tokens=512,
            condition_count=2,
            engine="original",
            actual_evaluations=9,
            forecast_evaluations=11,
        )
        self.assertEqual(anchored.packed_tokens - base.packed_tokens, 2 * 405)

    def test_rejects_invalid_h3_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "17.n.5"):
            self.analyzer.analyze(
                width=640,
                height=352,
                frames=125,
                text_tokens=10,
                condition_count=0,
                engine="original",
                actual_evaluations=9,
                forecast_evaluations=11,
            )


class PlannerTests(unittest.TestCase):
    def test_vae_tile_work_metric_matches_measured_crossovers(self) -> None:
        cases = {
            (640, 352): (256, 2, 3, 393_216),
            (864, 480): (288, 2, 4, 663_552),
            (640, 480): (288, 2, 3, 497_664),
            (1280, 736): (288, 3, 6, 1_492_992),
        }
        for (width, height), expected in cases.items():
            with self.subTest(width=width, height=height):
                decision = select_vae_tile(width=width, height=height)
                self.assertEqual(
                    (
                        decision.tile_size,
                        decision.tile_rows,
                        decision.tile_columns,
                        decision.decoded_tile_pixels,
                    ),
                    expected,
                )

    def test_conditioned_memory_floor_is_stage_peak_not_per_anchor(self) -> None:
        analyzer = H3WorkloadAnalyzer()
        common = dict(
            width=640,
            height=352,
            frames=124,
            text_tokens=68,
            engine="lora",
            actual_evaluations=6,
            forecast_evaluations=0,
        )
        unconditioned = analyzer.analyze(condition_count=0, **common)
        first_last = analyzer.analyze(condition_count=2, **common)
        model = MemoryModel(
            base_bytes=1024,
            conditioned_min_bytes=10 * 1024**3,
        )
        self.assertEqual(model.predict(unconditioned), 1024)
        self.assertEqual(model.predict(first_last), 10 * 1024**3)

    def test_execution_plan_rejects_unsupported_video_vae_tiles(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            ExecutionPlan(
                OffloadMode.BLOCK,
                8192,
                vae_spatial_tile=(280, 280),
            )
        with self.assertRaisesRegex(ValueError, "square tiles"):
            ExecutionPlan(
                OffloadMode.BLOCK,
                8192,
                vae_spatial_tile=(288, 256),
            )
        with self.assertRaisesRegex(ValueError, "dense_qk_quant_gran"):
            ExecutionPlan(
                OffloadMode.BLOCK,
                8192,
                dense_qk_quant_gran="invalid",  # type: ignore[arg-type]
            )
        for invalid in (0, 127, 129):
            with self.assertRaisesRegex(ValueError, "Query chunks"):
                ExecutionPlan(
                    OffloadMode.BLOCK,
                    8192,
                    long_sequence_query_chunk_tokens=invalid,
                )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            ExecutionPlan(
                OffloadMode.BLOCK,
                8192,
                long_sequence_query_chunk_tokens=4096,
                long_sequence_projection_chunk_tokens=8192,
            )
        with self.assertRaisesRegex(ValueError, "split QKV"):
            ExecutionPlan(
                OffloadMode.BLOCK,
                8192,
                long_sequence_split_qkv_outputs=True,
            )
        for field_name in (
            "long_sequence_exact_helper_stack",
            "long_sequence_single_qknorm_rope",
            "long_sequence_parallel_sparse_lut",
            "long_sequence_partial_sparse_topk",
            "long_sequence_fused_prefix_k_quant",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(ValueError, "helper experiments"):
                    ExecutionPlan(
                        OffloadMode.BLOCK,
                        8192,
                        **{field_name: True},
                    )

    def setUp(self) -> None:
        self.features = H3WorkloadAnalyzer().analyze(
            width=640,
            height=352,
            frames=124,
            text_tokens=1024,
            condition_count=0,
            engine="original",
            actual_evaluations=9,
            forecast_evaluations=11,
        )
        self.fast = CalibratedProfile(
            profile_id="resident-8192",
            supported_engines=("original",),
            plan=ExecutionPlan(OffloadMode.MODEL, 8192),
            latency=LatencyModel(intercept_seconds=15.0),
            memory=MemoryModel(base_bytes=20 * 1024**3),
            evidence_status="validated",
            max_packed_tokens=20_000,
        )
        self.safe = CalibratedProfile(
            profile_id="block-safe",
            supported_engines=("original", "lora"),
            plan=ExecutionPlan(OffloadMode.BLOCK, 4096),
            latency=LatencyModel(intercept_seconds=25.0),
            memory=MemoryModel(base_bytes=10 * 1024**3),
            evidence_status="validated",
        )

    def test_selects_fastest_validated_profile_that_fits(self) -> None:
        planner = RTX4090Planner((self.fast, self.safe), reserve_bytes=512 * 1024**2)
        decision = planner.select(
            self.features,
            free_device_bytes=24 * 1024**3,
            cached_shape_keys=(self.features.shape_key,),
        )
        self.assertEqual(decision.profile_id, "resident-8192")
        self.assertTrue(decision.shape_cache_hit)

    def test_low_free_memory_falls_back_without_changing_quality(self) -> None:
        planner = RTX4090Planner((self.fast, self.safe), reserve_bytes=1024**3)
        decision = planner.select(self.features, free_device_bytes=16 * 1024**3)
        self.assertEqual(decision.profile_id, "block-safe")
        self.assertFalse(hasattr(decision.plan, "actual_evaluations"))

    def test_experimental_profile_is_fail_closed(self) -> None:
        experimental = CalibratedProfile(
            profile_id="guess",
            supported_engines=("original",),
            plan=ExecutionPlan(OffloadMode.MODEL, 8192),
            latency=LatencyModel(intercept_seconds=1.0),
            memory=MemoryModel(base_bytes=1024),
        )
        planner = RTX4090Planner((experimental,))
        with self.assertRaises(NoFeasibleProfile):
            planner.select(self.features, free_device_bytes=24 * 1024**3)

    def test_checked_in_table_routes_product_envelope_to_block_and_adaptive_tile(self) -> None:
        analyzer = H3WorkloadAnalyzer()
        planner = RTX4090Planner(validated_original_profiles_2026_08_11())
        common = dict(
            text_tokens=256,
            condition_count=0,
            engine="original",
            actual_evaluations=9,
            forecast_evaluations=11,
        )
        small = analyzer.analyze(width=640, height=352, frames=124, **common)
        short = analyzer.analyze(width=864, height=480, frames=124, **common)
        large = analyzer.analyze(width=1280, height=736, frames=362, **common)
        small_plan = planner.select(small, free_device_bytes=24 * 1024**3).plan
        short_plan = planner.select(short, free_device_bytes=24 * 1024**3).plan
        large_plan = planner.select(large, free_device_bytes=24 * 1024**3).plan
        self.assertEqual(small_plan.offload_mode, OffloadMode.BLOCK)
        self.assertEqual(short_plan.offload_mode, OffloadMode.BLOCK)
        self.assertEqual(large_plan.offload_mode, OffloadMode.BLOCK)
        self.assertEqual(small_plan.vae_spatial_tile, (256, 256))
        self.assertEqual(short_plan.vae_spatial_tile, (288, 288))
        self.assertEqual(large_plan.vae_spatial_tile, (288, 288))

    def test_checked_in_table_rejects_beyond_validated_720p15_boundary(self) -> None:
        analyzer = H3WorkloadAnalyzer()
        planner = RTX4090Planner(validated_original_profiles_2026_08_11())
        beyond = analyzer.analyze(
            width=1280,
            height=736,
            frames=379,
            text_tokens=256,
            condition_count=0,
            engine="original",
            actual_evaluations=9,
            forecast_evaluations=11,
        )
        with self.assertRaises(NoFeasibleProfile):
            planner.select(beyond, free_device_bytes=24 * 1024**3)

    def test_checked_in_lora_table_routes_measured_product_envelope_to_block(self) -> None:
        analyzer = H3WorkloadAnalyzer()
        planner = RTX4090Planner(validated_lora_profiles_2026_08_11())
        common = dict(
            text_tokens=68,
            condition_count=0,
            engine="lora",
            actual_evaluations=6,
            forecast_evaluations=0,
        )
        for width, height, frames in (
            (640, 352, 124),
            (864, 480, 124),
            (1280, 736, 124),
            (1280, 736, 362),
        ):
            features = analyzer.analyze(
                width=width,
                height=height,
                frames=frames,
                **common,
            )
            decision = planner.select(
                features,
                free_device_bytes=24 * 1024**3,
            )
            self.assertEqual(decision.plan.offload_mode, OffloadMode.BLOCK)
            expected_tile = (256, 256) if (width, height) == (640, 352) else (288, 288)
            self.assertEqual(decision.plan.vae_spatial_tile, expected_tile)

    def test_checked_in_lora_table_rejects_uncalibrated_step_presets(self) -> None:
        features = H3WorkloadAnalyzer().analyze(
            width=864,
            height=480,
            frames=124,
            text_tokens=68,
            condition_count=0,
            engine="lora",
            actual_evaluations=4,
            forecast_evaluations=0,
        )
        planner = RTX4090Planner(validated_lora_profiles_2026_08_11())
        with self.assertRaises(NoFeasibleProfile):
            planner.select(features, free_device_bytes=24 * 1024**3)

    def test_checked_in_lora_table_accounts_for_condition_vae_peak(self) -> None:
        features = H3WorkloadAnalyzer().analyze(
            width=640,
            height=352,
            frames=124,
            text_tokens=234,
            condition_count=2,
            engine="lora",
            actual_evaluations=6,
            forecast_evaluations=0,
        )
        planner = RTX4090Planner(validated_lora_profiles_2026_08_11())
        decision = planner.select(features, free_device_bytes=24 * 1024**3)
        self.assertGreaterEqual(
            decision.predicted_peak_bytes,
            int(round(10.25 * 1024**3)),
        )

    def test_review_sparse_profiles_fail_closed_outside_measured_envelope(self) -> None:
        analyzer = H3WorkloadAnalyzer()
        planner = RTX4090Planner(
            validated_lora_profiles_2026_08_11()
            + review_sparse_profiles_2026_08_12(),
            allow_experimental=True,
        )
        common = dict(
            engine="lora",
            actual_evaluations=6,
            forecast_evaluations=0,
            condition_count=0,
        )
        accepted_720p10 = analyzer.analyze(
            width=1280,
            height=736,
            frames=243,
            text_tokens=318,
            **common,
        )
        rejected_480p15 = analyzer.analyze(
            width=864,
            height=480,
            frames=362,
            text_tokens=353,
            **common,
        )
        accepted = planner.select(
            accepted_720p10, free_device_bytes=24 * 1024**3
        )
        rejected = planner.select(
            rejected_480p15, free_device_bytes=24 * 1024**3
        )
        self.assertEqual(accepted.plan.attention_topk, 0.50)
        self.assertIsNone(rejected.plan.attention_topk)

    def test_review_720p15_combo_is_exact_shape_and_stays_experimental(self) -> None:
        analyzer = H3WorkloadAnalyzer()
        profiles = (
            validated_lora_profiles_2026_08_11()
            + review_sparse_profiles_2026_08_12()
            + review_combined_profiles_2026_08_12()
        )
        common = dict(
            width=1280,
            height=736,
            frames=362,
            condition_count=0,
            engine="lora",
            actual_evaluations=6,
            forecast_evaluations=0,
        )
        measured = analyzer.analyze(text_tokens=354, **common)
        neighboring_prompt = analyzer.analyze(text_tokens=353, **common)

        default = RTX4090Planner(profiles).select(
            measured, free_device_bytes=24 * 1024**3
        )
        reviewed = RTX4090Planner(
            profiles, allow_experimental=True
        ).select(measured, free_device_bytes=24 * 1024**3)
        neighboring = RTX4090Planner(
            profiles, allow_experimental=True
        ).select(neighboring_prompt, free_device_bytes=24 * 1024**3)

        self.assertFalse(default.plan.fused_rms_adaln)
        self.assertEqual(
            reviewed.profile_id,
            "sm89_lora6_sparse050_rms_720landscape_15s_review",
        )
        self.assertTrue(reviewed.plan.fused_rms_adaln)
        self.assertEqual(reviewed.plan.dense_qk_quant_gran, "per_warp")
        self.assertFalse(neighboring.plan.fused_rms_adaln)

    def test_fused_rms_review_routes_only_three_measured_dense_shapes(self) -> None:
        analyzer = H3WorkloadAnalyzer()
        profiles = (
            validated_lora_profiles_2026_08_11()
            + review_fused_rms_profiles_2026_08_12()
        )
        planner = RTX4090Planner(profiles, allow_experimental=True)
        common = dict(
            condition_count=0,
            engine="lora",
            actual_evaluations=6,
            forecast_evaluations=0,
        )
        measured_480p5 = analyzer.analyze(
            width=864,
            height=480,
            frames=124,
            text_tokens=215,
            **common,
        )
        neighboring_prompt = analyzer.analyze(
            width=864,
            height=480,
            frames=124,
            text_tokens=214,
            **common,
        )
        measured = planner.select(
            measured_480p5, free_device_bytes=24 * 1024**3
        )
        neighboring = planner.select(
            neighboring_prompt, free_device_bytes=24 * 1024**3
        )
        self.assertTrue(measured.plan.fused_rms_adaln)
        self.assertFalse(neighboring.plan.fused_rms_adaln)

    def test_fused_rms_review_never_routes_original_weight_engine(self) -> None:
        analyzer = H3WorkloadAnalyzer()
        planner = RTX4090Planner(
            validated_original_profiles_2026_08_11()
            + review_fused_rms_profiles_2026_08_12(),
            allow_experimental=True,
        )
        original_480p5 = analyzer.analyze(
            width=864,
            height=480,
            frames=124,
            text_tokens=215,
            condition_count=0,
            engine="original",
            actual_evaluations=9,
            forecast_evaluations=11,
        )

        selected = planner.select(
            original_480p5, free_device_bytes=24 * 1024**3
        )

        self.assertFalse(selected.plan.fused_rms_adaln)

    def test_review_sparse_profiles_only_accept_measured_condition_modes(self) -> None:
        analyzer = H3WorkloadAnalyzer()
        planner = RTX4090Planner(
            validated_lora_profiles_2026_08_11()
            + review_sparse_profiles_2026_08_12(),
            allow_experimental=True,
        )
        common = dict(
            width=1280,
            height=736,
            frames=243,
            engine="lora",
            actual_evaluations=6,
            forecast_evaluations=0,
        )
        first_last = analyzer.analyze(
            text_tokens=2049, condition_count=2, **common
        )
        first_only = analyzer.analyze(
            text_tokens=1184, condition_count=1, **common
        )
        self.assertEqual(
            planner.select(first_last, free_device_bytes=24 * 1024**3)
            .plan.attention_topk,
            0.50,
        )
        self.assertIsNone(
            planner.select(first_only, free_device_bytes=24 * 1024**3)
            .plan.attention_topk
        )

    def test_original_sparse_review_profile_is_exactly_balanced_720p10(self) -> None:
        analyzer = H3WorkloadAnalyzer()
        planner = RTX4090Planner(
            validated_original_profiles_2026_08_11()
            + review_sparse_profiles_2026_08_12(),
            allow_experimental=True,
        )
        accepted = analyzer.analyze(
            width=1280,
            height=736,
            frames=243,
            text_tokens=318,
            condition_count=0,
            engine="original",
            actual_evaluations=9,
            forecast_evaluations=11,
        )
        self.assertEqual(
            planner.select(accepted, free_device_bytes=24 * 1024**3)
            .plan.attention_topk,
            0.75,
        )


if __name__ == "__main__":
    unittest.main()
