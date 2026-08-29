from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from h3serve.native_engine.local_checkpoint_cache import (
    default_cache_root,
    drop_file_page_cache,
    materialize_local_checkpoint,
    materialize_qwen_layer_cache,
)


class LocalCheckpointCacheTest(unittest.TestCase):
    def test_materializes_and_reuses_byte_identical_file(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / "source" / "weights.safetensors"
            source.parent.mkdir()
            source.write_bytes(b"packed-qwen" * 1024)
            import hashlib
            expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            cache = root / "cache"
            with patch.dict(os.environ, {"H3_SERVE_LOCALIZE_QWEN": "1"}):
                target = materialize_local_checkpoint(
                    source, cache_root=cache, reserve_bytes=0,
                    expected_sha256=expected_hash,
                )
                self.assertEqual(target.read_bytes(), source.read_bytes())
                first_mtime = target.stat().st_mtime_ns
                reused = materialize_local_checkpoint(
                    source, cache_root=cache, reserve_bytes=0,
                    expected_sha256=expected_hash,
                )
            self.assertEqual(reused, target)
            self.assertEqual(reused.stat().st_mtime_ns, first_mtime)
            metadata = json.loads(
                target.with_suffix(target.suffix + ".source.json").read_text()
            )
            self.assertEqual(metadata["source_size"], source.stat().st_size)
            self.assertEqual(metadata["sha256"], expected_hash)

    def test_default_cache_is_per_user_and_configurable(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            with patch.dict(os.environ, {"H3_SERVE_LOCAL_MODEL_CACHE": name}):
                self.assertEqual(default_cache_root(), Path(name))

    def test_bad_hash_never_publishes_cache(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / "weights.safetensors"
            source.write_bytes(b"weights")
            cache = root / "cache"
            with patch.dict(os.environ, {"H3_SERVE_LOCALIZE_QWEN": "1"}):
                selected = materialize_local_checkpoint(
                    source, cache_root=cache, reserve_bytes=0,
                    expected_sha256="0" * 64,
                )
            self.assertEqual(selected, source.resolve())
            self.assertFalse((cache / source.name).exists())

    def test_disabled_localization_returns_source(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            source = Path(name) / "weights.safetensors"
            source.write_bytes(b"weights")
            with patch.dict(os.environ, {"H3_SERVE_LOCALIZE_QWEN": "0"}):
                self.assertEqual(
                    materialize_local_checkpoint(source, cache_root=Path(name) / "cache"),
                    source.resolve(),
                )

    def test_page_cache_release_is_best_effort(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "file"
            path.write_bytes(b"data")
            drop_file_page_cache(path)
            drop_file_page_cache(path.with_name("missing"))

    def test_qwen_layer_cache_is_execution_ordered_and_reusable(self) -> None:
        try:
            import torch
            from safetensors import safe_open
            from safetensors.torch import save_file
        except ImportError as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / "qwen.safetensors"
            expected = {
                "model.layers.0.self_attn.q_proj.weight": torch.arange(8),
                "model.layers.0.mlp.down_proj.weight": torch.arange(3),
                "model.layers.1.self_attn.q_proj.weight": torch.arange(8) + 10,
                "model.layers.1.mlp.down_proj.weight": torch.arange(3) + 10,
                "model.embed_tokens.weight": torch.arange(2),
            }
            save_file(expected, source)
            cache = root / "cache"
            selected = materialize_qwen_layer_cache(
                source, cache_root=cache, layers=2, reserve_bytes=0,
            )
            self.assertIsNotNone(selected)
            for index in range(2):
                shard = selected / f"layer-{index:02d}.safetensors"
                with safe_open(shard, framework="pt", device="cpu") as handle:
                    self.assertTrue(all(
                        key.startswith(f"model.layers.{index}.")
                        for key in handle.keys()
                    ))
                    for key in handle.keys():
                        torch.testing.assert_close(handle.get_tensor(key), expected[key])
            reused = materialize_qwen_layer_cache(
                source, cache_root=cache, layers=2, reserve_bytes=0,
            )
            self.assertEqual(reused, selected)


if __name__ == "__main__":
    unittest.main()
