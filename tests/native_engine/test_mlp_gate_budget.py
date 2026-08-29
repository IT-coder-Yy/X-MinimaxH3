from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import torch

from h3serve.native_engine.model import mlp_gate_budget


class MlpGateBudgetTests(unittest.TestCase):
    def tearDown(self) -> None:
        mlp_gate_budget.configured_budget.cache_clear()
        mlp_gate_budget.adaptive_gate_config.cache_clear()

    def test_default_is_exact(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.tearDown()
            self.assertFalse(
                mlp_gate_budget.skip_video_mlp(
                    layer=4,
                    step=3,
                    step_count=20,
                    gate=torch.full((2, 4), 0.1),
                    row=1,
                )
            )

    def test_adaptive_gate_protects_trajectory_ends(self) -> None:
        environment = {
            "H3_NATIVE_EXPERIMENTAL_VIDEO_MLP_GATE_MAX_RMS": "0.2",
            "H3_NATIVE_EXPERIMENTAL_VIDEO_MLP_GATE_HEAD_STEPS": "1",
            "H3_NATIVE_EXPERIMENTAL_VIDEO_MLP_GATE_TAIL_STEPS": "3",
        }
        gate = torch.tensor([[1.0, 1.0], [0.1, 0.1]])
        with patch.dict(os.environ, environment, clear=True):
            self.tearDown()
            self.assertFalse(
                mlp_gate_budget.skip_video_mlp(
                    layer=4, step=0, step_count=20, gate=gate, row=1
                )
            )
            self.assertTrue(
                mlp_gate_budget.skip_video_mlp(
                    layer=4, step=3, step_count=20, gate=gate, row=1
                )
            )
            self.assertFalse(
                mlp_gate_budget.skip_video_mlp(
                    layer=4, step=17, step_count=20, gate=gate, row=1
                )
            )

    def test_adaptive_gate_uses_video_row(self) -> None:
        environment = {"H3_NATIVE_EXPERIMENTAL_VIDEO_MLP_GATE_MAX_RMS": "0.2"}
        with patch.dict(os.environ, environment, clear=True):
            self.tearDown()
            gate = torch.tensor([[0.1, 0.1], [0.3, 0.3]])
            self.assertFalse(
                mlp_gate_budget.skip_video_mlp(
                    layer=4, step=3, step_count=20, gate=gate, row=1
                )
            )

    def test_explicit_budget_remains_available_for_reproduction(self) -> None:
        environment = {
            "H3_NATIVE_EXPERIMENTAL_VIDEO_MLP_SKIP_LAYERS": "4,5",
            "H3_NATIVE_EXPERIMENTAL_VIDEO_MLP_SKIP_STEPS": "3",
            "H3_NATIVE_EXPERIMENTAL_VIDEO_MLP_GATE_MAX_RMS": "0.01",
        }
        with patch.dict(os.environ, environment, clear=True):
            self.tearDown()
            self.assertTrue(mlp_gate_budget.skip_video_mlp(layer=4, step=3))
            self.assertFalse(mlp_gate_budget.skip_video_mlp(layer=6, step=3))


if __name__ == "__main__":
    unittest.main()
