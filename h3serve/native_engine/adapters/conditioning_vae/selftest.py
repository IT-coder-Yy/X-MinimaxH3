"""CPU-only unit and real-header checks for conditioning/VAE adapters."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from .preprocess import cover_crop_plan, prepare_keyframes


class CanvasTests(unittest.TestCase):
    def test_cover_crop_plan_is_centered_and_covering(self) -> None:
        plan = cover_crop_plan(400, 200, 100, 100)
        self.assertEqual(plan["resized_size"], (200, 100))
        self.assertEqual(plan["crop_box"], (50, 0, 150, 100))

    def test_roles_not_list_position_choose_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = Image.new("RGB", (4, 2))
            # Red left half, blue right half.  Stretch keeps both; square
            # cover/crop keeps the centered two columns.
            for x in range(4):
                color = (255, 0, 0) if x < 2 else (0, 0, 255)
                for y in range(2):
                    source.putpixel((x, y), color)
            image_path = directory / "key.png"
            source.save(image_path)
            last_only = SimpleNamespace(
                width=2,
                height=2,
                num_frames=22,
                first_frame=None,
                last_frame=image_path,
            )
            item = prepare_keyframes(last_only)[0]
            self.assertEqual(item.role, "last")
            self.assertEqual(item.semantic_frame_index, -1)
            self.assertEqual(item.resolved_frame_index, 21)
            self.assertEqual(item.image.size, (2, 2))
            self.assertEqual(item.image.getpixel((0, 0)), (255, 0, 0))
            self.assertEqual(item.image.getpixel((1, 0)), (0, 0, 255))

    def test_first_last_order_and_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            first = directory / "first.png"
            last = directory / "last.png"
            Image.new("RGB", (3, 5), (10, 20, 30)).save(first)
            Image.new("RGBA", (5, 3), (40, 50, 60, 20)).save(last)
            request = SimpleNamespace(
                width=32,
                height=64,
                num_frames=362,
                first_frame=first,
                last_frame=last,
            )
            items = prepare_keyframes(request)
            self.assertEqual(tuple(item.role for item in items), ("first", "last"))
            self.assertEqual(
                tuple(item.resolved_frame_index for item in items), (0, 361)
            )
            self.assertTrue(all(item.image.mode == "RGB" for item in items))
            self.assertTrue(all(item.image.size == (32, 64) for item in items))


class RealHeaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import safetensors  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("safetensors is not installed")
        cls.model_root = Path(__file__).resolve().parents[4] / "models"
        if not cls.model_root.is_dir():
            raise unittest.SkipTest("release model root is unavailable")

    def test_real_checkpoint_headers(self) -> None:
        from .audit import audit_checkpoint

        cases = (
            (
                "text",
                "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                "comfy_nvfp4_awq_single_file_v1",
            ),
            (
                "video_vae",
                "vae/minimax_h3_video_vae_fp16.safetensors",
                "sglang_fused_h3_video_vae_v1",
            ),
            (
                "audio_vae",
                "vae/minimax_h3_audio_vae_fp32.safetensors",
                "h3_audio_raw_weight_v1",
            ),
        )
        for kind, relative, layout in cases:
            with self.subTest(kind=kind):
                report = audit_checkpoint(self.model_root / relative, kind)
                self.assertEqual(report.layout, layout)
                self.assertEqual(report.errors, ())


class TensorAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("torch is not installed")
        cls.torch = torch

    def test_audio_stereo_flatten_denormalize_and_loudness(self) -> None:
        from .audio_vae import H3AudioVAEAdapter

        torch = self.torch

        class FakeAudio:
            sample_rate = 32000

            def decode(self, latents):
                self.seen = latents
                return SimpleNamespace(sample=latents[:, :1].repeat(1, 1, 3))

        model = FakeAudio()
        adapter = H3AudioVAEAdapter(
            model,
            latents_mean=[1.0] * 32,
            latents_std=[2.0] * 32,
        )
        audio, rate = adapter.decode(torch.zeros(1, 32, 2, 4))
        self.assertEqual(tuple(model.seen.shape), (2, 32, 4))
        self.assertTrue(torch.all(model.seen == 1.0))
        self.assertEqual(tuple(audio.shape), (1, 2, 12))
        self.assertEqual(rate, 32000)

    def test_keyframe_patch_shape(self) -> None:
        from .video_vae import patchify_keyframe

        latent = self.torch.arange(24 * 1 * 4 * 6).reshape(1, 24, 1, 4, 6)
        rows = patchify_keyframe(latent)
        self.assertEqual(tuple(rows.shape), (6, 96))
        self.assertEqual(rows.dtype, self.torch.float32)


if __name__ == "__main__":
    unittest.main()
