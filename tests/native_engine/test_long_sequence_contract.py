from __future__ import annotations

import unittest

import torch

from h3serve.native_engine.model.long_sequence_contract import (
    resolve_physical_long_sequence_contract,
)
from h3serve.native_engine.model.layers import FusedQKVAttention
from h3serve.native_engine.model.kernels import (
    attention_protected_prefix,
    attention_video_layout,
    long_sequence_query_chunking,
)


class PhysicalLongSequenceContractTests(unittest.TestCase):
    def test_hnd_backend_retains_released_hnd_optimizations(self) -> None:
        class HNDBackend:
            long_sequence_kv_layout = "HND"

        contract = resolve_physical_long_sequence_contract(
            HNDBackend(),
            compact_kv=False,
            direct_nhd_kv_requested=False,
            fused_qknorm_hnd_requested=True,
            direct_hnd_fp8_value_requested=True,
        )

        self.assertEqual(contract.kv_layout, "HND")
        self.assertTrue(contract.fused_qknorm_hnd_layout)
        self.assertTrue(contract.direct_hnd_fp8_value)

    def test_nhd_backend_falls_back_without_rejecting_request(self) -> None:
        class NHDBackend:
            long_sequence_kv_layout = "NHD"

        contract = resolve_physical_long_sequence_contract(
            NHDBackend(),
            compact_kv=False,
            direct_nhd_kv_requested=False,
            fused_qknorm_hnd_requested=True,
            direct_hnd_fp8_value_requested=True,
        )

        self.assertEqual(contract.kv_layout, "NHD")
        self.assertFalse(contract.fused_qknorm_hnd_layout)
        self.assertFalse(contract.direct_hnd_fp8_value)

    def test_direct_nhd_override_disables_hnd_only_features(self) -> None:
        class FlexibleBackend:
            long_sequence_kv_layout = "HND"
            supports_direct_nhd_kv = True

        contract = resolve_physical_long_sequence_contract(
            FlexibleBackend(),
            compact_kv=False,
            direct_nhd_kv_requested=True,
            fused_qknorm_hnd_requested=True,
            direct_hnd_fp8_value_requested=True,
        )

        self.assertEqual(contract.kv_layout, "NHD")
        self.assertFalse(contract.fused_qknorm_hnd_layout)
        self.assertFalse(contract.direct_hnd_fp8_value)

    def test_invalid_backend_layout_fails_closed(self) -> None:
        class InvalidBackend:
            long_sequence_kv_layout = "HDN"

        with self.assertRaisesRegex(ValueError, "unsupported"):
            resolve_physical_long_sequence_contract(
                InvalidBackend(),
                compact_kv=False,
                direct_nhd_kv_requested=False,
                fused_qknorm_hnd_requested=False,
                direct_hnd_fp8_value_requested=False,
            )

    def test_nhd_streaming_executes_when_request_enables_hnd_fast_path(self) -> None:
        """Regression for the real 2K second-sampling layout failure."""

        class SliceLinear(torch.nn.Linear):
            def forward_output_slice(self, value, start, stop):
                bias = None if self.bias is None else self.bias[start:stop]
                return torch.nn.functional.linear(
                    value, self.weight[start:stop], bias
                )

        class NHDPreparedBackend:
            long_sequence_kv_layout = "NHD"

            def resolve_long_sequence_backend(self, query_tokens):
                return self if query_tokens >= 128 else None

            @staticmethod
            def prepare_long_sequence_values(value_nhd):
                batch, tokens, heads, head_dim = value_nhd.shape
                assert batch == 1
                return value_nhd, torch.ones(1), heads, tokens, head_dim

            @staticmethod
            def prepare_long_sequence_keys(key_nhd, value_fp8, value_scale):
                del value_fp8, value_scale
                return key_nhd

            @staticmethod
            def long_sequence_prefix_queries(query, prepared):
                del prepared
                return query

            @staticmethod
            def long_sequence_video_queries(
                query,
                prepared,
                *,
                protected_tokens,
                query_token_indices,
            ):
                del prepared, protected_tokens, query_token_indices
                return query

        torch.manual_seed(20260828)
        attention = FusedQKVAttention(
            SliceLinear(8, 24, bias=False),
            torch.nn.Linear(8, 8, bias=False),
            num_heads=2,
            head_dim=4,
            backend=NHDPreparedBackend(),
            dtype=torch.float32,
        )
        attention.q_norm.weight.data.fill_(1.0)
        attention.k_norm.weight.data.fill_(1.0)
        hidden = torch.randn(384, 8)
        residual = torch.randn_like(hidden)
        gate = torch.ones(2, 8)
        segments = ((0, 128, 0), (128, 384, 1))

        with (
            attention_protected_prefix(128),
            attention_video_layout(2, 128),
            long_sequence_query_chunking(
                128,
                split_qkv_outputs=True,
                single_qknorm_rope=True,
                fused_query_projection=True,
                fused_qknorm_hnd_layout=True,
                direct_hnd_fp8_value=True,
            ),
        ):
            self.assertTrue(
                attention.stream_gated_residual_(
                    residual,
                    hidden,
                    None,
                    gate,
                    segments,
                    query_chunk_tokens=128,
                )
            )


if __name__ == "__main__":
    unittest.main()
