from __future__ import annotations

import unittest

import torch

from h3serve.native_engine.segment_cache import (
    CoordinateAlignedSegmentCache,
    SegmentResidualCacheConfig,
)


class _LinearStepStack:
    def __init__(self, count: int = 6) -> None:
        self.blocks = [object()] * count

    def run_range(self, value, *, start, stop, step_scale, **_kwargs):
        value.add_(float(stop - start) * float(step_scale))
        return value

    def run_protected_range(
        self,
        value,
        *,
        start,
        stop,
        protected_tokens,
        active_video_indices=None,
        active_video_layer_start=0,
        active_video_layer_stop=50,
        step_scale,
        **_kwargs,
    ):
        value[:protected_tokens].add_(float(stop - start) * float(step_scale))
        if active_video_indices is not None:
            active_layers = max(
                0,
                min(stop, active_video_layer_stop)
                - max(start, active_video_layer_start),
            )
            value.index_add_(
                0,
                active_video_indices,
                torch.full(
                    (active_video_indices.numel(), value.shape[1]),
                    float(active_layers) * float(step_scale),
                    dtype=value.dtype,
                ),
            )
        return value


class CoordinateAlignedSegmentCacheTest(unittest.TestCase):
    def test_two_dense_observations_predict_same_coordinate_linear_update(self) -> None:
        cache = CoordinateAlignedSegmentCache(
            SegmentResidualCacheConfig(
                layer_start=2,
                layer_stop=4,
                reuse_steps=(2,),
                transfer_chunk_rows=2,
            )
        )
        stack = _LinearStepStack()
        outputs = []
        for step, scale in enumerate((1.0, 2.0, 3.0)):
            value = torch.zeros((5, 3), dtype=torch.float32)
            outputs.append(
                cache.run_actual_tail(
                    stack,
                    value,
                    step_index=step,
                    prefix_stop=1,
                    protected_tokens=0,
                    block_kwargs={"step_scale": scale},
                )
            )
        self.assertTrue(torch.all(outputs[0] == 5.0))
        self.assertTrue(torch.all(outputs[1] == 10.0))
        self.assertTrue(torch.all(outputs[2] == 15.0))
        self.assertEqual(cache.records[-1]["mode"], "predicted")
        self.assertEqual(cache.records[-1]["alpha"], 2.0)

    def test_reuse_without_two_observations_fails_closed_to_dense(self) -> None:
        cache = CoordinateAlignedSegmentCache(
            SegmentResidualCacheConfig(1, 3, (0,))
        )
        result = cache.run_actual_tail(
            _LinearStepStack(4),
            torch.zeros((2, 2)),
            step_index=0,
            prefix_stop=1,
            protected_tokens=0,
            block_kwargs={"step_scale": 2.0},
        )
        self.assertTrue(torch.all(result == 6.0))
        self.assertEqual(cache.records[-1]["mode"], "dense")

    def test_directional_trust_extends_safe_range_for_aligned_trajectory(self) -> None:
        cache = CoordinateAlignedSegmentCache(
            SegmentResidualCacheConfig(
                layer_start=2,
                layer_stop=4,
                reuse_steps=(4,),
                transfer_chunk_rows=2,
                directional_trust=True,
                directional_max_extra=0.35,
                directional_min_cosine=0.25,
            )
        )
        stack = _LinearStepStack()
        for step, scale in ((0, 1.0), (1, 2.0), (4, 5.0)):
            cache.run_actual_tail(
                stack,
                torch.zeros((5, 3), dtype=torch.float32),
                step_index=step,
                prefix_stop=1,
                protected_tokens=0,
                block_kwargs={"step_scale": scale},
            )
        record = cache.records[-1]
        self.assertEqual(record["mode"], "predicted")
        self.assertAlmostEqual(record["directional_cosine"], 1.0, places=5)
        self.assertAlmostEqual(record["alpha"], 1.35, places=5)
        self.assertEqual(record["directional_extended_safe_range"], 1.0)

    def test_directional_trust_preserves_valid_fixed_safe_range(self) -> None:
        cache = CoordinateAlignedSegmentCache(
            SegmentResidualCacheConfig(
                layer_start=2,
                layer_stop=4,
                reuse_steps=(2,),
                directional_trust=True,
            )
        )
        stack = _LinearStepStack()
        for step, scale in enumerate((1.0, 2.0, 3.0)):
            cache.run_actual_tail(
                stack,
                torch.zeros((5, 3), dtype=torch.float32),
                step_index=step,
                prefix_stop=1,
                protected_tokens=0,
                block_kwargs={"step_scale": scale},
            )
        record = cache.records[-1]
        self.assertEqual(record["mode"], "predicted")
        self.assertAlmostEqual(record["alpha"], 2.0, places=5)
        self.assertNotIn("directional_cosine", record)

    def test_directional_trust_fails_closed_on_reversed_live_direction(self) -> None:
        cache = CoordinateAlignedSegmentCache(
            SegmentResidualCacheConfig(
                layer_start=2,
                layer_stop=4,
                reuse_steps=(4,),
                directional_trust=True,
                directional_min_cosine=0.25,
            )
        )
        stack = _LinearStepStack()
        for step, scale in enumerate((1.0, 2.0)):
            cache.run_actual_tail(
                stack,
                torch.zeros((5, 3), dtype=torch.float32),
                step_index=step,
                prefix_stop=1,
                protected_tokens=0,
                block_kwargs={"step_scale": scale},
            )
        cache.run_actual_tail(
            stack,
            torch.zeros((5, 3), dtype=torch.float32),
            step_index=4,
            prefix_stop=1,
            protected_tokens=0,
            block_kwargs={"step_scale": -1.0},
        )
        self.assertEqual(cache.records[-1]["mode"], "dense")
        self.assertIn("outside", cache.records[-1]["fallback_reason"])

    def test_protected_refresh_updates_prefix_and_predicts_only_video(self) -> None:
        cache = CoordinateAlignedSegmentCache(
            SegmentResidualCacheConfig(
                layer_start=2,
                layer_stop=4,
                reuse_steps=(2,),
                transfer_chunk_rows=2,
                protected_refresh=True,
            )
        )
        stack = _LinearStepStack()
        outputs = []
        for step, scale in enumerate((1.0, 2.0, 3.0)):
            outputs.append(
                cache.run_actual_tail(
                    stack,
                    torch.zeros((5, 1), dtype=torch.float32),
                    step_index=step,
                    prefix_stop=1,
                    protected_tokens=2,
                    block_kwargs={"step_scale": scale},
                )
            )
        # Prefix: dense blocks [1,2) + protected [2,4) + dense [4,6).
        self.assertTrue(torch.all(outputs[-1][:2] == 15.0))
        # Video: dense blocks [1,2) + predicted segment residual + [4,6).
        self.assertTrue(torch.all(outputs[-1][2:] == 15.0))
        self.assertTrue(cache.records[-1]["protected_refresh"])
        self.assertEqual(cache.records[-1]["protected_tokens"], 2)

    def test_active_video_router_selects_aligned_query_blocks(self) -> None:
        cache = CoordinateAlignedSegmentCache(
            SegmentResidualCacheConfig(
                layer_start=2,
                layer_stop=4,
                reuse_steps=(2,),
                transfer_chunk_rows=2,
                protected_refresh=True,
                active_video_ratio=0.25,
                active_query_block=2,
            )
        )
        stack = _LinearStepStack()
        for step, scale in enumerate((1.0, 2.0, 3.0)):
            result = cache.run_actual_tail(
                stack,
                torch.zeros((8, 3), dtype=torch.float32),
                step_index=step,
                prefix_stop=1,
                protected_tokens=2,
                video_shape=(1, 2, 3),
                block_kwargs={"step_scale": scale},
            )
        self.assertTrue(torch.all(result == 15.0))
        record = cache.records[-1]
        self.assertEqual(record["mode"], "predicted")
        self.assertEqual(record["active_video_tokens"], 2.0)
        self.assertAlmostEqual(record["active_video_ratio"], 1.0 / 3.0)

    def test_active_video_refresh_can_be_limited_to_a_sensitive_layer_group(self) -> None:
        cache = CoordinateAlignedSegmentCache(
            SegmentResidualCacheConfig(
                layer_start=1,
                layer_stop=5,
                reuse_steps=(2,),
                protected_refresh=True,
                active_video_ratio=0.5,
                active_query_block=2,
                active_layer_start=3,
                active_layer_stop=5,
            )
        )
        stack = _LinearStepStack()
        for step, scale in enumerate((1.0, 2.0, 3.0)):
            result = cache.run_actual_tail(
                stack,
                torch.zeros((6, 1), dtype=torch.float32),
                step_index=step,
                prefix_stop=0,
                protected_tokens=2,
                video_shape=(1, 1, 4),
                block_kwargs={"step_scale": scale},
            )
        # Prefix refreshes all four cached layers plus the exact tail. Selected
        # video rows refresh two cached layers; the remainder use the predicted
        # four-layer residual, and every row then executes the exact tail.
        self.assertTrue(torch.all(result[:2] == 18.0))
        self.assertEqual(torch.unique(result[2:]).tolist(), [12.0, 18.0])
        exported = cache.export()
        self.assertEqual(exported["active_layer_start"], 3)
        self.assertEqual(exported["active_layer_stop"], 5)

    def test_sequential_layer_groups_compose_updates_in_layer_order(self) -> None:
        cache = CoordinateAlignedSegmentCache(
            SegmentResidualCacheConfig(
                layer_start=1,
                layer_stop=5,
                reuse_steps=(2,),
                protected_refresh=True,
                active_video_ratio=0.5,
                active_query_block=2,
                active_layer_start=3,
                active_layer_stop=5,
                sequential_layer_groups=True,
            )
        )
        stack = _LinearStepStack()
        for step, scale in enumerate((1.0, 2.0, 3.0)):
            result = cache.run_actual_tail(
                stack,
                torch.zeros((6, 1), dtype=torch.float32),
                step_index=step,
                prefix_stop=0,
                protected_tokens=2,
                video_shape=(1, 1, 4),
                block_kwargs={"step_scale": scale},
            )
        # The early predicted group is applied before the selected true
        # refresh. Both selected and unselected video rows therefore retain
        # all four cached layers instead of replacing the early-layer update.
        self.assertTrue(torch.all(result == 18.0))
        record = cache.records[-1]
        self.assertTrue(record["sequential_layer_groups"])
        self.assertEqual(
            [(item["layer_start"], item["layer_stop"]) for item in record["layer_group_records"]],
            [(1, 3), (3, 5)],
        )
        self.assertTrue(record["layer_group_records"][1]["active_video_refresh"])
        self.assertEqual(cache.export()["layer_groups"], [[1, 3], [3, 5]])

    def test_sequential_low_confidence_group_uses_zero_order_hold(self) -> None:
        cache = CoordinateAlignedSegmentCache(
            SegmentResidualCacheConfig(
                layer_start=1,
                layer_stop=5,
                reuse_steps=(4,),
                directional_trust=True,
                directional_min_cosine=0.25,
                protected_refresh=True,
                active_video_ratio=0.5,
                active_query_block=2,
                active_layer_start=3,
                active_layer_stop=5,
                sequential_layer_groups=True,
                sequential_conservative_hold=True,
            )
        )
        stack = _LinearStepStack()
        for step, scale in ((0, 1.0), (1, 2.0), (4, -1.0)):
            result = cache.run_actual_tail(
                stack,
                torch.zeros((6, 1), dtype=torch.float32),
                step_index=step,
                prefix_stop=0,
                protected_tokens=2,
                video_shape=(1, 1, 4),
                block_kwargs={"step_scale": scale},
            )
        record = cache.records[-1]
        self.assertEqual(record["mode"], "predicted")
        self.assertTrue(
            any(
                item.get("directional_conservative_hold") == 1.0
                for item in record["layer_group_records"]
            )
        )
        self.assertTrue(torch.isfinite(result).all())

    def test_dynamic_budget_uses_minimum_blocks_covering_innovation(self) -> None:
        cache = CoordinateAlignedSegmentCache(
            SegmentResidualCacheConfig(
                layer_start=1,
                layer_stop=3,
                reuse_steps=(2,),
                protected_refresh=True,
                active_video_ratio=0.5,
                dynamic_video_budget=True,
                active_video_min_ratio=0.25,
                innovation_risk_coverage=0.8,
                innovation_max_relative=4.0,
                active_query_block=2,
                active_layer_start=2,
                active_layer_stop=3,
            )
        )
        stack = _LinearStepStack()
        cache.run_actual_tail(
            stack,
            torch.zeros((10, 3), dtype=torch.float32),
            step_index=0,
            prefix_stop=0,
            protected_tokens=2,
            video_shape=(1, 1, 8),
            block_kwargs={"step_scale": 1.0},
        )
        cache.run_actual_tail(
            stack,
            torch.ones((10, 3), dtype=torch.float32),
            step_index=1,
            prefix_stop=0,
            protected_tokens=2,
            video_shape=(1, 1, 8),
            block_kwargs={"step_scale": 2.0},
        )
        current = torch.full((10, 3), 2.0, dtype=torch.float32)
        current[2:4].add_(1.0)
        cache.run_actual_tail(
            stack,
            current,
            step_index=2,
            prefix_stop=0,
            protected_tokens=2,
            video_shape=(1, 1, 8),
            block_kwargs={"step_scale": 3.0},
        )
        record = cache.records[-1]
        self.assertEqual(record["mode"], "predicted")
        self.assertEqual(record["active_video_tokens"], 2.0)
        self.assertAlmostEqual(record["active_video_ratio"], 0.25)
        self.assertGreaterEqual(
            record["innovation_risk_coverage_achieved"], 0.8
        )

    def test_dynamic_budget_fails_closed_on_large_global_innovation(self) -> None:
        cache = CoordinateAlignedSegmentCache(
            SegmentResidualCacheConfig(
                layer_start=1,
                layer_stop=3,
                reuse_steps=(2,),
                protected_refresh=True,
                active_video_ratio=0.5,
                dynamic_video_budget=True,
                active_video_min_ratio=0.25,
                innovation_max_relative=0.5,
                active_query_block=2,
            )
        )
        stack = _LinearStepStack()
        for step, value in enumerate((0.0, 1.0, 12.0)):
            cache.run_actual_tail(
                stack,
                torch.full((10, 3), value, dtype=torch.float32),
                step_index=step,
                prefix_stop=0,
                protected_tokens=2,
                video_shape=(1, 1, 8),
                block_kwargs={"step_scale": float(step + 1)},
            )
        self.assertEqual(cache.records[-1]["mode"], "dense")
        self.assertIn("outside", cache.records[-1]["fallback_reason"])

    def test_contract_rejects_malformed_ranges_and_steps(self) -> None:
        with self.assertRaises(ValueError):
            SegmentResidualCacheConfig(4, 4, ())
        with self.assertRaises(ValueError):
            SegmentResidualCacheConfig(2, 4, (3, 2))
        with self.assertRaises(ValueError):
            SegmentResidualCacheConfig(
                2, 4, (2,), directional_trust=True, directional_max_extra=1.1
            )
        with self.assertRaises(ValueError):
            SegmentResidualCacheConfig(
                2,
                4,
                (2,),
                protected_refresh=True,
                active_video_ratio=0.25,
                active_layer_start=1,
                active_layer_stop=3,
            )
        with self.assertRaises(ValueError):
            SegmentResidualCacheConfig(
                2,
                4,
                (2,),
                protected_refresh=True,
                sequential_layer_groups=True,
            )
        with self.assertRaises(ValueError):
            SegmentResidualCacheConfig(
                2,
                4,
                (2,),
                sequential_conservative_hold=True,
            )


if __name__ == "__main__":
    unittest.main()
