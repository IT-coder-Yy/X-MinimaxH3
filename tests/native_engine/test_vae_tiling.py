from __future__ import annotations

import unittest

import torch

from h3serve.native_engine.adapters.vae_tiling import (
    configure_vae_tile_batching,
    install_bounded_tile_batching,
)


class FakeVAE:
    def __init__(self) -> None:
        self.stack_tiling = False
        self.batch_sizes: list[int] = []

    def _run_tile_tasks(self, tiles, tile_indices, forward_fn, stack_tiling, cls_agg=None):
        raise AssertionError("upstream implementation should be replaced")


class BoundedTileBatchingTests(unittest.TestCase):
    def test_bounded_batches_preserve_tile_order_and_values(self) -> None:
        model = install_bounded_tile_batching(FakeVAE(), 2)
        tiles = [torch.full((1, 1), float(index)) for index in range(5)]

        def forward(value):
            model.batch_sizes.append(value.shape[0])
            return value * 3 + 1

        outputs = model._run_tile_tasks(
            tiles, [0, 1, 2, 3, 4], forward, model.stack_tiling
        )
        self.assertEqual(model.batch_sizes, [2, 2, 1])
        self.assertEqual([value.item() for value in outputs], [1, 4, 7, 10, 13])

    def test_size_one_preserves_sequential_execution(self) -> None:
        model = install_bounded_tile_batching(FakeVAE(), 1)
        tiles = [torch.full((1, 1), float(index)) for index in range(3)]
        outputs = model._run_tile_tasks(
            tiles,
            [0, 1, 2],
            lambda value: model.batch_sizes.append(value.shape[0]) or value,
            model.stack_tiling,
        )
        self.assertEqual(model.batch_sizes, [1, 1, 1])
        self.assertEqual([value.item() for value in outputs], [0, 1, 2])

    def test_request_level_reconfiguration_does_not_reinstall(self) -> None:
        model = install_bounded_tile_batching(FakeVAE(), 1)
        configure_vae_tile_batching(model, 3)
        self.assertTrue(model.stack_tiling)
        self.assertEqual(model.native_tile_batch_size, 3)
        configure_vae_tile_batching(model, 1)
        self.assertFalse(model.stack_tiling)

    def test_invalid_size_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            install_bounded_tile_batching(FakeVAE(), 0)


if __name__ == "__main__":
    unittest.main()
