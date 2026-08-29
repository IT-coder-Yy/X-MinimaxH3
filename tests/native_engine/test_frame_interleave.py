from __future__ import annotations

import unittest

import torch

from h3serve.native_engine.model.frame_interleave import (
    FrameInterleaveConfig,
    FrameInterleavePlan,
)
from h3serve.native_engine.model.packed import build_fl2va_layout


class FrameInterleavePlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = build_fl2va_layout(
            text_length=3,
            latent_frames=5,
            latent_height=4,
            latent_width=4,
            audio_frames=2,
        )

    def test_rotates_complete_frames_and_keeps_prefix(self) -> None:
        plan = FrameInterleavePlan(
            self.layout,
            FrameInterleaveConfig(stride=2),
            torch.device("cpu"),
        )
        even = plan.for_layer(0)
        odd = plan.for_layer(1)
        assert even is not None and odd is not None
        self.assertEqual(even.anchor_frames.tolist(), [0, 2, 4])
        self.assertEqual(odd.anchor_frames.tolist(), [0, 1, 3, 4])
        self.assertTrue(
            torch.equal(
                even.selected_indices[: even.protected_tokens],
                torch.arange(even.protected_tokens),
            )
        )
        self.assertEqual(
            int(even.selected_indices.numel()),
            even.protected_tokens + 3 * even.frame_tokens,
        )

    def test_linear_update_reconstruction_preserves_each_frame_state(self) -> None:
        plan = FrameInterleavePlan(
            self.layout,
            FrameInterleaveConfig(stride=2),
            torch.device("cpu"),
        )
        layer = plan.for_layer(0)
        assert layer is not None
        full = torch.arange(
            self.layout.sequence_length * 2, dtype=torch.float32
        ).view(self.layout.sequence_length, 2)
        original = full.clone()
        selected_input = full.index_select(0, layer.selected_indices)
        selected_output = selected_input.clone()
        selected_output[: layer.protected_tokens] = 7.0
        for anchor_position, frame in enumerate(layer.anchor_frames.tolist()):
            start = layer.protected_tokens + anchor_position * layer.frame_tokens
            selected_output[start : start + layer.frame_tokens].add_(float(frame))
        layer.reconstruct_(full, selected_input, selected_output)
        self.assertTrue(torch.all(full[: layer.protected_tokens] == 7.0))
        video = full[layer.protected_tokens :].view(5, layer.frame_tokens, 2)
        original_video = original[layer.protected_tokens :].view(
            5, layer.frame_tokens, 2
        )
        for frame in range(5):
            self.assertTrue(
                torch.allclose(
                    video[frame], original_video[frame] + float(frame)
                )
            )

    def test_dense_layers_and_range_fail_closed(self) -> None:
        plan = FrameInterleavePlan(
            self.layout,
            FrameInterleaveConfig(
                stride=3,
                layer_start=2,
                layer_stop=8,
                dense_layers=(4,),
            ),
            torch.device("cpu"),
        )
        self.assertIsNone(plan.for_layer(1))
        self.assertIsNotNone(plan.for_layer(2))
        self.assertIsNone(plan.for_layer(4))
        self.assertIsNone(plan.for_layer(8))


if __name__ == "__main__":
    unittest.main()
