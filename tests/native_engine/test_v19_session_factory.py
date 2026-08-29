from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from h3serve.native_engine.session_factory import (
    NativeSessionFactory,
    NativeSessionPaths,
)


def _factory(root: Path) -> NativeSessionFactory:
    return NativeSessionFactory(NativeSessionPaths(
        model_root=root / "models",
        minimax_source=root / "minimax",
        lightx_source=root / "lightx",
        turbo_curve=root / "curve.safetensors",
        output_root=root / "outputs",
    ))


class V19SessionFactoryTests(unittest.TestCase):
    def test_lora_checkpoint_must_be_an_installed_safetensors_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            installed = root / "models/loras/nested/release.safetensors"
            installed.parent.mkdir(parents=True)
            installed.write_bytes(b"header-only-test")
            factory = _factory(root)
            factory.set_lora_checkpoint(installed)
            self.assertEqual(factory._lora_checkpoint, installed.absolute())
            outside = root / "outside.safetensors"
            outside.write_bytes(b"not-installed")
            with self.assertRaisesRegex(ValueError, "models/loras"):
                factory.set_lora_checkpoint(outside)

    def test_explicit_mechanistic_admission_overrides_v24(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            admission = root / "mechanistic-admission.json"
            admission.write_text("{}", encoding="utf-8")
            with patch.dict(os.environ, {
                "H3_NATIVE_ENABLE_SPARSE": "1",
                "H3_NATIVE_PARETO_V24": "auto",
                "H3_NATIVE_MECHANISTIC_ADMISSION": str(admission),
            }, clear=True):
                factory = _factory(root)
                self.assertTrue(factory.mechanistic_deployment_enabled)
                self.assertFalse(factory.v24_release_enabled)
                self.assertTrue(factory.v19_scheduler_enabled)

    def test_missing_mechanistic_admission_is_not_silently_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing-mechanistic-admission.json"
            with patch.dict(os.environ, {
                "H3_NATIVE_ENABLE_SPARSE": "1",
                "H3_NATIVE_PARETO_V24": "auto",
                "H3_NATIVE_MECHANISTIC_ADMISSION": str(missing),
            }, clear=True):
                factory = _factory(root)
                self.assertFalse(factory.mechanistic_deployment_enabled)
                self.assertFalse(factory.v24_release_enabled)
                result = factory.preflight("first_last")
                self.assertFalse(result["checks"]["mechanistic_admission"])

    def test_v24_is_the_default_selector_when_release_sparse_runtime_is_on(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {
                "H3_NATIVE_ENABLE_SPARSE": "1",
                "H3_NATIVE_PARETO_V24": "auto",
                "H3_NATIVE_V19_EXPERIMENTAL_LONG_HORIZON": "0",
            }, clear=True):
                factory = _factory(root)
                self.assertTrue(factory.v24_release_enabled)
                self.assertTrue(factory.v19_scheduler_enabled)

    def test_v24_has_an_explicit_rollback_switch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {
                "H3_NATIVE_ENABLE_SPARSE": "1",
                "H3_NATIVE_PARETO_V24": "0",
                "H3_NATIVE_V19_EXPERIMENTAL_LONG_HORIZON": "0",
            }, clear=True):
                factory = _factory(root)
                self.assertFalse(factory.v24_release_enabled)
                self.assertFalse(factory.v19_scheduler_enabled)

    def test_explicit_missing_bundle_is_a_preflight_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing-v19.json"
            with patch.dict(os.environ, {
                "H3_NATIVE_V19_RELEASE_BUNDLE": str(missing),
                "H3_NATIVE_ENABLE_SPARSE": "0",
            }):
                result = _factory(root).preflight("first_last")
        self.assertFalse(result["ready"])
        self.assertFalse(result["checks"]["v19_release_bundle"])

    def test_explicit_sparse_disable_prevents_bundle_activation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "v19.json"
            bundle.write_text("{}", encoding="utf-8")
            with patch.dict(os.environ, {
                "H3_NATIVE_V19_RELEASE_BUNDLE": str(bundle),
                "H3_NATIVE_ENABLE_SPARSE": "0",
            }):
                self.assertFalse(_factory(root).v19_release_enabled)

    def test_experimental_long_overlay_does_not_claim_release_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing-v19.json"
            with patch.dict(os.environ, {
                "H3_NATIVE_V19_RELEASE_BUNDLE": str(missing),
                "H3_NATIVE_V19_EXPERIMENTAL_LONG_HORIZON": "1",
            }):
                factory = _factory(root)
                self.assertTrue(factory.v19_scheduler_enabled)
                self.assertFalse(factory.v19_release_enabled)

    def test_experimental_long_overlay_is_disabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing-v19.json"
            with patch.dict(os.environ, {
                "H3_NATIVE_V19_RELEASE_BUNDLE": str(missing),
                "H3_NATIVE_V19_EXPERIMENTAL_LONG_HORIZON": "0",
            }):
                self.assertFalse(_factory(root).v19_scheduler_enabled)


if __name__ == "__main__":
    unittest.main()
