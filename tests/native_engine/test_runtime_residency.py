from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from h3serve.native_engine.runtime import ImmutablePinnedModuleResidency
from h3serve.native_engine.runtime.pinned_pool import pack_pinned_tensors


class _SharedWeightModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared = nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4))
        self.weight = shared
        self.alias = shared
        self.register_buffer("scale", torch.tensor([2.0]))


class ImmutablePinnedModuleResidencyTest(unittest.TestCase):
    def test_cpu_round_trip_preserves_values_and_aliases(self) -> None:
        module = _SharedWeightModule()
        residency = ImmutablePinnedModuleResidency(
            "tiny", module, pin_host_weights=False
        )
        residency.prepare_host()

        self.assertTrue(residency.prepared)
        self.assertEqual(residency.host_bytes, 52)
        self.assertIs(module.weight, module.alias)
        torch.testing.assert_close(
            module.weight,
            torch.arange(12, dtype=torch.float32).reshape(3, 4),
        )
        residency.move_to("cpu", non_blocking=True)
        self.assertIs(module.weight, module.alias)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_evict_rebinds_original_host_storage_without_d2h(self) -> None:
        module = _SharedWeightModule()
        residency = ImmutablePinnedModuleResidency(
            "tiny", module, pin_host_weights=True
        )
        residency.prepare_host()
        host_pointer = module.weight.data_ptr()
        self.assertTrue(residency.host_is_pinned)

        residency.move_to("cuda:0", non_blocking=True)
        torch.cuda.synchronize()
        self.assertEqual(module.weight.device.type, "cuda")
        self.assertIs(module.weight, module.alias)

        # Mutating the disposable device copy must not mutate the immutable
        # authoritative host master used by inference phase transitions.
        module.weight.data.zero_()
        residency.move_to("cpu", non_blocking=False)
        self.assertEqual(module.weight.device.type, "cpu")
        self.assertEqual(module.weight.data_ptr(), host_pointer)
        self.assertGreater(float(module.weight.detach().sum()), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_pinned_pool_preserves_values_strides_and_async_copy(self) -> None:
        contiguous = torch.arange(60, dtype=torch.float32).reshape(5, 12)
        strided = torch.arange(48, dtype=torch.float16).reshape(6, 8).t()
        packed = pack_pinned_tensors((contiguous, strided), slab_bytes=4096)

        self.assertEqual(len(packed.slabs), 2)  # one slab per dtype
        self.assertTrue(all(tensor.is_pinned() for tensor in packed.tensors))
        self.assertEqual(packed.tensors[1].stride(), strided.stride())
        torch.testing.assert_close(packed.tensors[0], contiguous)
        torch.testing.assert_close(packed.tensors[1], strided)
        copied = packed.tensors[0].to("cuda:0", non_blocking=True)
        torch.cuda.synchronize()
        torch.testing.assert_close(copied.cpu(), contiguous)


if __name__ == "__main__":
    unittest.main()
