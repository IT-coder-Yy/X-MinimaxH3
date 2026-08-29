from __future__ import annotations

import unittest

import torch


class DirectHNDFP8ValueTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_sm89_direct_writer_is_byte_exact_to_sparge_abi(self) -> None:
        if torch.cuda.get_device_capability() != (8, 9):
            self.skipTest("direct HND FP8 V is calibrated for SM89")

        from h3serve.native_engine.sm89_policy import configure_sm89_runtime

        configure_sm89_runtime(quant_backend="cuda", smoke_test=True)
        from spas_sage_attn import core as sparge_core

        from h3serve.native_engine.model.compact_fp8 import (
            prepare_sage_fp8_hnd_direct,
        )

        torch.manual_seed(20260828)
        value_hnd = torch.randn(
            (1, 4, 997, 128),
            device="cuda:0",
            dtype=torch.float16,
        )
        padded = (value_hnd.shape[2] + 127) // 128 * 128
        transposed = torch.empty(
            (1, 4, 128, padded),
            device=value_hnd.device,
            dtype=value_hnd.dtype,
        )
        sparge_core.fused.transpose_pad_permute_cuda(
            value_hnd, transposed, 1
        )
        reference_fp8 = torch.empty_like(
            transposed, dtype=torch.float8_e4m3fn
        )
        reference_scale = torch.empty(
            (1, 4, 128),
            device=value_hnd.device,
            dtype=torch.float32,
        )
        sparge_core.fused.scale_fuse_quant_cuda(
            transposed,
            reference_fp8,
            reference_scale,
            value_hnd.shape[2],
            2.25,
            1,
        )

        candidate_fp8, candidate_scale = prepare_sage_fp8_hnd_direct(
            value_hnd
        )
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(
                candidate_fp8.view(torch.uint8),
                reference_fp8.view(torch.uint8),
            )
        )
        self.assertTrue(torch.equal(candidate_scale, reference_scale))


if __name__ == "__main__":
    unittest.main()
