from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from h3serve.native_engine.model.dit import H3FinalLayer


class _StaticAdaLN(nn.Module):
    def __init__(self, params: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("params", params)

    def forward(self, _curve_rows) -> torch.Tensor:
        return self.params


class _TrackingNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.max_rows = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.max_rows = max(self.max_rows, int(value.shape[0]))
        return value


class FinalProjectionChunkingTests(unittest.TestCase):
    def _layer(self, norm: nn.Module) -> H3FinalLayer:
        torch.manual_seed(17)
        hidden = 8
        params = torch.randn(2, hidden * 2, dtype=torch.float32)
        return H3FinalLayer(
            norm,
            _StaticAdaLN(params),
            nn.Linear(hidden, 5, dtype=torch.float32),
            nn.Linear(hidden, 3, dtype=torch.float32),
            hidden_size=hidden,
        )

    def test_chunked_projection_matches_established_path(self) -> None:
        torch.manual_seed(23)
        value = torch.randn(19, 8, dtype=torch.bfloat16)
        layer = self._layer(nn.Identity())
        kwargs = dict(
            curve_rows=object(),
            video_segment=slice(2, 16),
            audio_segment=slice(16, 19),
            video_timestep_row=0,
            audio_timestep_row=1,
        )
        expected = layer(value, **kwargs)
        actual = layer(value, final_projection_chunk_tokens=4, **kwargs)
        torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)

    def test_chunking_limits_pre_fp32_working_set(self) -> None:
        norm = _TrackingNorm()
        layer = self._layer(norm)
        value = torch.randn(25, 8, dtype=torch.bfloat16)
        layer(
            value,
            curve_rows=object(),
            video_segment=slice(0, 21),
            audio_segment=slice(21, 25),
            video_timestep_row=0,
            audio_timestep_row=1,
            final_projection_chunk_tokens=5,
        )
        self.assertEqual(norm.max_rows, 5)

    def test_non_positive_chunk_is_rejected(self) -> None:
        layer = self._layer(nn.Identity())
        with self.assertRaisesRegex(ValueError, "must be positive"):
            layer(
                torch.randn(4, 8, dtype=torch.bfloat16),
                curve_rows=object(),
                video_segment=slice(0, 3),
                audio_segment=slice(3, 4),
                video_timestep_row=0,
                audio_timestep_row=1,
                final_projection_chunk_tokens=0,
            )


if __name__ == "__main__":
    unittest.main()
