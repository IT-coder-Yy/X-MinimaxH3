from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from h3serve.native_engine.adapters.vae_compile import (
    enable_feed_forward_compile,
    enable_transformer_block_compile,
    transformer_block_compile,
    transformer_block_compile_ready,
)


class FeedForward(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self._compile_forward_enabled = False
        self._compiled_forward = None

    def _forward_impl(self, value):
        return value * self.weight


class FakeVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ff = FeedForward()
        self.other = torch.nn.Linear(1, 1)


class TransformerBlock(torch.nn.Module):
    def forward(self, value):
        return value + 1


class FakeTransformerVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = TransformerBlock()

    def forward(self, value):
        return self.block(value)


class VAERegionalCompileTests(unittest.TestCase):
    def test_only_feed_forward_region_is_compiled_without_cudagraph_mode(self) -> None:
        model = FakeVAE()
        sentinel = object()
        with patch("torch.compile", return_value=sentinel) as compiler:
            count = enable_feed_forward_compile(model)
        self.assertEqual(count, 1)
        self.assertIs(model.ff._compiled_forward, sentinel)
        self.assertTrue(model.ff._compile_forward_enabled)
        compiler.assert_called_once()
        self.assertEqual(compiler.call_args.kwargs["backend"], "inductor")
        self.assertFalse(compiler.call_args.kwargs["fullgraph"])
        self.assertNotIn("mode", compiler.call_args.kwargs)

    def test_transformer_compile_is_request_scoped_and_restores_default(self) -> None:
        model = FakeTransformerVAE()

        def fake_compile(eager, **_kwargs):
            return lambda value: eager(value) + 10

        with patch("torch.compile", side_effect=fake_compile) as compiler:
            self.assertEqual(enable_transformer_block_compile(model), 1)
        self.assertTrue(transformer_block_compile_ready(model))
        self.assertEqual(model(torch.tensor(1)).item(), 2)
        with transformer_block_compile(True):
            self.assertEqual(model(torch.tensor(1)).item(), 12)
        self.assertEqual(model(torch.tensor(1)).item(), 2)
        self.assertFalse(compiler.call_args.kwargs["dynamic"])
        self.assertEqual(
            compiler.call_args.kwargs["options"], {"triton.cudagraphs": False}
        )

    def test_transformer_compile_scope_resets_after_failure(self) -> None:
        model = FakeTransformerVAE()
        with patch("torch.compile", side_effect=lambda eager, **_: eager):
            enable_transformer_block_compile(model)
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with transformer_block_compile(True):
                raise RuntimeError("stop")
        self.assertEqual(model(torch.tensor(1)).item(), 2)


if __name__ == "__main__":
    unittest.main()
