import os
import unittest
from unittest.mock import patch

import h3serve.native_engine.model.kernels as kernels


class ReferenceRMSIslandTests(unittest.TestCase):
    def setUp(self) -> None:
        kernels._reference_rms_island_values.cache_clear()

    def tearDown(self) -> None:
        kernels._reference_rms_island_values.cache_clear()

    def test_requires_matching_step_and_layer(self) -> None:
        environment = {
            "H3_NATIVE_EXPERIMENTAL_REFERENCE_RMS_STEPS": "0,1,2,3,4",
            "H3_NATIVE_EXPERIMENTAL_REFERENCE_RMS_LAYERS": "30,31,32",
        }
        with patch.dict(os.environ, environment, clear=False):
            with kernels.attention_step(2, 20), kernels.attention_layer(31):
                self.assertTrue(kernels._reference_rms_island_active())
            with kernels.attention_step(8, 20), kernels.attention_layer(31):
                self.assertFalse(kernels._reference_rms_island_active())
            with kernels.attention_step(2, 20), kernels.attention_layer(12):
                self.assertFalse(kernels._reference_rms_island_active())

    def test_empty_policy_keeps_global_fused_choice(self) -> None:
        with patch.dict(
            os.environ,
            {
                "H3_NATIVE_EXPERIMENTAL_REFERENCE_RMS_STEPS": "",
                "H3_NATIVE_EXPERIMENTAL_REFERENCE_RMS_LAYERS": "",
            },
            clear=False,
        ):
            with kernels.attention_step(2, 20), kernels.attention_layer(31):
                self.assertFalse(kernels._reference_rms_island_active())


if __name__ == "__main__":
    unittest.main()
