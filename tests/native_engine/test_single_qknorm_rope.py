from __future__ import annotations

import hashlib
import unittest

import torch

from h3serve.native_engine.model.kernels import (
    current_long_sequence_direct_hnd_fp8_value,
    current_long_sequence_exact_helper_stack,
    current_long_sequence_fused_prefix_k_quant,
    current_long_sequence_parallel_sparse_lut,
    current_long_sequence_partial_sparse_topk,
    current_long_sequence_single_qknorm_rope,
    long_sequence_query_chunking,
)
from h3serve.native_engine.model.single_qknorm_rope import (
    try_apply_single_qknorm_rope_,
    try_apply_single_qknorm_rope_to_hnd,
)


class SingleQKNormRopeTests(unittest.TestCase):
    def test_full_output_hash_covers_every_tensor_byte_in_chunks(self) -> None:
        from scripts.profile_h3_1080p15_real_block_memory import tensor_sha256

        value = torch.arange(35, dtype=torch.float32).reshape(7, 5).to(
            torch.bfloat16
        )
        expected = hashlib.sha256(
            value.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        self.assertEqual(tensor_sha256(value, chunk_rows=3), expected)

    def test_request_scope_does_not_leak(self) -> None:
        self.assertFalse(current_long_sequence_single_qknorm_rope())
        self.assertFalse(current_long_sequence_exact_helper_stack())
        self.assertFalse(current_long_sequence_parallel_sparse_lut())
        self.assertFalse(current_long_sequence_partial_sparse_topk())
        self.assertFalse(current_long_sequence_fused_prefix_k_quant())
        self.assertFalse(current_long_sequence_direct_hnd_fp8_value())
        with long_sequence_query_chunking(
            128,
            split_qkv_outputs=True,
            single_qknorm_rope=True,
            exact_helper_stack=True,
            parallel_sparse_lut=True,
            partial_sparse_topk=True,
            fused_prefix_k_quant=True,
            direct_hnd_fp8_value=True,
        ):
            self.assertTrue(current_long_sequence_single_qknorm_rope())
            self.assertTrue(current_long_sequence_exact_helper_stack())
            self.assertTrue(current_long_sequence_parallel_sparse_lut())
            self.assertTrue(current_long_sequence_partial_sparse_topk())
            self.assertTrue(current_long_sequence_fused_prefix_k_quant())
            self.assertTrue(current_long_sequence_direct_hnd_fp8_value())
        self.assertFalse(current_long_sequence_single_qknorm_rope())
        self.assertFalse(current_long_sequence_exact_helper_stack())
        self.assertFalse(current_long_sequence_parallel_sparse_lut())
        self.assertFalse(current_long_sequence_partial_sparse_topk())
        self.assertFalse(current_long_sequence_fused_prefix_k_quant())
        self.assertFalse(current_long_sequence_direct_hnd_fp8_value())

    def test_cpu_fails_closed_without_mutation(self) -> None:
        value = torch.randn(3, 2, 128, dtype=torch.bfloat16)
        original = value.clone()
        applied = try_apply_single_qknorm_rope_(
            value,
            weight=torch.randn(128, dtype=torch.bfloat16),
            frequencies=torch.randn(1, 3, 1, 48, 2, 2, dtype=torch.bfloat16),
            eps=1.0e-5,
        )
        self.assertFalse(applied)
        self.assertTrue(torch.equal(value, original))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_sm89_single_partial_kernel_is_bit_exact_to_paired_kernel(self) -> None:
        if torch.cuda.get_device_capability() != (8, 9):
            self.skipTest("exact physical contract is calibrated for SM89")
        from h3serve.native_engine.sm89_policy import configure_sm89_runtime

        # Runtime policy must select and hash-check the release-owned module
        # before the test imports any Comfy-Kitchen symbol. Importing the
        # ambient site-package first would correctly trigger the production
        # fail-closed guard and make this test depend on discovery order.
        configure_sm89_runtime(quant_backend="cuda", smoke_test=True)
        from comfy_kitchen import rms_rope_split_half_
        torch.manual_seed(89)
        tokens, heads, head_dim, rotate_width = 257, 4, 128, 96
        source = torch.randn(
            tokens,
            heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        reference = source.clone()
        candidate = source.clone()
        scratch = torch.randn_like(reference).unsqueeze(0)
        weight = torch.randn(
            head_dim, device="cuda", dtype=torch.bfloat16
        )
        frequencies = torch.randn(
            1,
            tokens,
            1,
            rotate_width // 2,
            2,
            2,
            device="cuda",
            dtype=torch.bfloat16,
        )
        rms_rope_split_half_(
            reference.unsqueeze(0),
            scratch,
            frequencies,
            weight,
            weight,
            epsilon=1.0e-5,
            rot_dim=rotate_width,
        )
        self.assertTrue(
            try_apply_single_qknorm_rope_(
                candidate,
                weight=weight,
                frequencies=frequencies,
                eps=1.0e-5,
            )
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(candidate, reference))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_sm89_single_partial_kernel_writes_exact_hnd_backing(self) -> None:
        if torch.cuda.get_device_capability() != (8, 9):
            self.skipTest("exact physical contract is calibrated for SM89")
        from h3serve.native_engine.sm89_policy import configure_sm89_runtime

        configure_sm89_runtime(quant_backend="cuda", smoke_test=True)
        from comfy_kitchen import rms_rope_split_half_

        torch.manual_seed(8913)
        tokens, heads, head_dim, rotate_width = 257, 4, 128, 96
        source = torch.randn(
            tokens,
            heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        reference = source.clone()
        scratch = torch.randn_like(reference).unsqueeze(0)
        weight = torch.randn(
            head_dim, device="cuda", dtype=torch.bfloat16
        )
        frequencies = torch.randn(
            1,
            tokens,
            1,
            rotate_width // 2,
            2,
            2,
            device="cuda",
            dtype=torch.bfloat16,
        )
        rms_rope_split_half_(
            reference.unsqueeze(0),
            scratch,
            frequencies,
            weight,
            weight,
            epsilon=1.0e-5,
            rot_dim=rotate_width,
        )
        # Use a larger backing sequence so the destination head stride also
        # covers the K-slab case inside a whole-video HND allocation.
        candidate_hnd = torch.empty(
            heads,
            tokens + 13,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        candidate_view = candidate_hnd[:, :tokens].permute(1, 0, 2)
        self.assertTrue(
            try_apply_single_qknorm_rope_to_hnd(
                source.clone(),
                candidate_view,
                weight=weight,
                frequencies=frequencies,
                eps=1.0e-5,
            )
        )
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(candidate_hnd[:, :tokens], reference.permute(1, 0, 2))
        )


if __name__ == "__main__":
    unittest.main()
