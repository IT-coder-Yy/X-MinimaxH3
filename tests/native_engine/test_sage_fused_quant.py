from __future__ import annotations

import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class SageFusedQuantTests(unittest.TestCase):
    def setUp(self) -> None:
        if torch.cuda.get_device_capability() != (8, 9):
            self.skipTest("the release kernel is validated on SM89")

    @staticmethod
    def _inputs(layout: str):
        torch.manual_seed(82736)
        query_tokens = 131
        key_tokens = 197
        heads = 4
        query_hnd = torch.randn(
            (1, heads, query_tokens, 128),
            device="cuda",
            dtype=torch.bfloat16,
        )
        # A non-zero mean exposes invalid tail rows that were accidentally
        # centered before masking in the old fused implementation.
        key_hnd = (
            torch.randn(
                (1, heads, key_tokens, 128),
                device="cuda",
                dtype=torch.float32,
            )
            * 0.125
            + 5.0
        ).to(torch.bfloat16)
        if layout == "HND":
            return query_hnd, key_hnd
        return (
            query_hnd.permute(0, 2, 1, 3).contiguous(),
            key_hnd.permute(0, 2, 1, 3).contiguous(),
        )

    def test_hnd_ragged_nonzero_mean_matches_sage_byte_for_byte(self) -> None:
        from sageattention.triton.quant_per_thread import per_thread_int8

        from h3serve.native_engine.model.sage_fused_quant import (
            quantize_qk_sub_mean_per_thread_int8_hnd,
        )

        query, key = self._inputs("HND")
        key_mean = key.mean(dim=2, keepdim=True)
        reference = per_thread_int8(
            query,
            key,
            key_mean,
            tensor_layout="HND",
            BLKQ=128,
            WARPQ=32,
            BLKK=64,
            WARPK=64,
        )
        candidate = quantize_qk_sub_mean_per_thread_int8_hnd(
            query, key, key_mean
        )
        torch.cuda.synchronize()
        for reference_tensor, candidate_tensor in zip(reference, candidate):
            self.assertTrue(torch.equal(reference_tensor, candidate_tensor))

    def test_nhd_ragged_nonzero_mean_matches_sage_byte_for_byte(self) -> None:
        from sageattention.triton.quant_per_thread import per_thread_int8

        from h3serve.native_engine.model.sage_fused_quant import (
            quantize_qk_sub_mean_per_thread_int8_nhd,
        )

        query, key = self._inputs("NHD")
        key_mean = key.mean(dim=1, keepdim=True)
        reference = per_thread_int8(
            query,
            key,
            key_mean,
            tensor_layout="NHD",
            BLKQ=128,
            WARPQ=32,
            BLKK=64,
            WARPK=64,
        )
        candidate = quantize_qk_sub_mean_per_thread_int8_nhd(
            query, key, key_mean
        )
        torch.cuda.synchronize()
        for reference_tensor, candidate_tensor in zip(reference, candidate):
            self.assertTrue(torch.equal(reference_tensor, candidate_tensor))


if __name__ == "__main__":
    unittest.main()
