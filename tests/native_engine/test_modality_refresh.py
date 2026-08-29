from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from h3serve.native_engine.model.layers import (
    FusedQKVAttention,
    H3TransformerBlock,
    SwiGLUMLP,
    torch_sdpa,
)
from h3serve.native_engine.model.modality_refresh import (
    refresh_protected_modalities,
    refresh_selected_video_tiles,
)


class ProtectedModalityRefreshTest(unittest.TestCase):
    @staticmethod
    def _block() -> H3TransformerBlock:
        torch.manual_seed(7)
        attention = FusedQKVAttention(
            nn.Linear(8, 24),
            nn.Linear(8, 8),
            num_heads=2,
            head_dim=4,
            backend=torch_sdpa,
        )
        mlp = SwiGLUMLP(nn.Linear(8, 32), nn.Linear(16, 8))
        block = H3TransformerBlock(
            attention,
            mlp,
            nn.Linear(4, 18 * 8),
            hidden_size=8,
        ).eval()
        with torch.no_grad():
            block.norm1.weight.fill_(1.0)
            block.norm2.weight.fill_(1.0)
            block.attention.q_norm.weight.fill_(1.0)
            block.attention.k_norm.weight.fill_(1.0)
        return block

    def test_prefix_matches_full_block_and_video_is_unchanged(self) -> None:
        block = self._block()
        value = torch.randn(6, 8)
        rows = torch.randn(2, 4)
        segments = ((0, 2, 0), (2, 3, 1), (3, 6, 0))
        frequencies = torch.zeros(6, 0)
        with torch.inference_mode():
            dense = block(
                value.clone(),
                timestep_rows=rows,
                modulation_segments=segments,
                frequencies=frequencies,
            )
            guarded = refresh_protected_modalities(
                block,
                value.clone(),
                protected_tokens=3,
                timestep_rows=rows,
                modulation_segments=segments,
                frequencies=frequencies,
            )
        torch.testing.assert_close(guarded[:3], dense[:3])
        self.assertTrue(torch.equal(guarded[3:], value[3:]))

    def test_boundary_must_align_with_modality_segments(self) -> None:
        with self.assertRaises(ValueError):
            refresh_protected_modalities(
                self._block(),
                torch.randn(6, 8),
                protected_tokens=2,
                timestep_rows=torch.randn(2, 4),
                modulation_segments=((0, 3, 0), (3, 6, 1)),
                frequencies=torch.zeros(6, 0),
            )

    def test_selected_video_rows_match_full_block_and_other_rows_stay_put(self) -> None:
        block = self._block()
        value = torch.randn(6, 8)
        rows = torch.randn(2, 4)
        segments = ((0, 2, 0), (2, 3, 1), (3, 6, 0))
        frequencies = torch.zeros(6, 0)
        active = torch.tensor([4], dtype=torch.long)
        with torch.inference_mode():
            dense = block(
                value.clone(),
                timestep_rows=rows,
                modulation_segments=segments,
                frequencies=frequencies,
            )
            selected = refresh_selected_video_tiles(
                block,
                value.clone(),
                protected_tokens=3,
                active_video_indices=active,
                timestep_rows=rows,
                modulation_segments=segments,
                frequencies=frequencies,
            )
        torch.testing.assert_close(selected[:3], dense[:3])
        torch.testing.assert_close(selected[4], dense[4])
        self.assertTrue(torch.equal(selected[3], value[3]))
        self.assertTrue(torch.equal(selected[5], value[5]))


if __name__ == "__main__":
    unittest.main()
