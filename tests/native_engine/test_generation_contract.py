from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path, PurePosixPath

from h3serve.app import JobRecord, JobService
from h3serve.native_engine.hot_session import (
    HotSessionRequest,
    NativeT2AVHotSession,
)
from h3serve.contract import (
    FPS,
    LORA_PRESETS,
    ORIGINAL_PRESETS,
    GenerationSpec,
    resolve_frames,
    resolve_geometry,
)


SERVE_ROOT = Path(__file__).resolve().parents[2]


def latent_shape(spec: GenerationSpec) -> dict[str, tuple[int, ...]]:
    video_t = 2 if spec.frames <= 5 else ((spec.frames - 5) // 17) * 5 + 2
    audio_t = round(spec.actual_duration_seconds * 40)
    return {
        "video": (1, 24, video_t, spec.height // 16, spec.width // 16),
        "audio": (1, 32, 2, audio_t),
    }


class GenerationPlanningContractTest(unittest.TestCase):
    def spec(self, **overrides) -> GenerationSpec:
        request = {
            "prompt": "Contract fixture.",
            "engine": "original",
            "quality": "balanced",
            "resolution": "480p",
            "aspect_ratio": "16:9",
            "duration_seconds": 5,
            "seed": 4404,
        }
        request.update(overrides)
        return GenerationSpec.from_mapping(request)

    def test_geometry_is_the_resolved_canvas_not_the_marketing_label(self) -> None:
        self.assertEqual(resolve_geometry("360p", "16:9"), (640, 352))
        self.assertEqual(resolve_geometry("480p", "16:9"), (864, 480))
        self.assertEqual(resolve_geometry("720p", "16:9"), (1280, 736))
        self.assertEqual(resolve_geometry("1080p", "16:9"), (1920, 1088))
        self.assertEqual(resolve_geometry("2k", "16:9"), (2560, 1440))
        self.assertEqual(resolve_geometry("480p", "9:16"), (480, 864))
        for resolution in ("360p", "480p", "720p", "1080p", "2k"):
            for ratio in ("1:1", "4:3", "3:4", "16:9", "9:16"):
                width, height = resolve_geometry(resolution, ratio)
                self.assertEqual(width % 32, 0)
                self.assertEqual(height % 32, 0)

    def test_fixed_360p_preview_is_admitted_for_1080p_checkpoint(self) -> None:
        request = HotSessionRequest(
            prompt="fixed checkpoint preview",
            seed=1,
            width=1920,
            height=1088,
            frames=362,
            fps=24,
            steps=5,
            output_path=Path("unused.mp4"),
            preview_step_index=0,
            preview_output_path=Path("unused.preview.mp4"),
            preview_decode_mode="fast_finish",
            preview_branch_steps=4,
            preview_branch_spatial_scale=352 / 1088,
        )

        request.validate()

    def test_time_and_latent_grids_are_fully_resolved_before_engine_entry(self) -> None:
        five = self.spec(duration_seconds=5)
        self.assertEqual(resolve_frames(5), (124, 124 / FPS))
        self.assertEqual(latent_shape(five), {
            "video": (1, 24, 37, 30, 54),
            "audio": (1, 32, 2, 207),
        })
        fifteen = self.spec(duration_seconds=15)
        self.assertEqual(resolve_frames(15), (362, 362 / FPS))
        self.assertEqual(latent_shape(fifteen), {
            "video": (1, 24, 107, 30, 54),
            "audio": (1, 32, 2, 603),
        })
        for seconds in (1, 2.5, 3, 5, 10, 15):
            spec = self.spec(duration_seconds=seconds)
            self.assertEqual((spec.frames - 5) % 17, 0)
            self.assertTrue(math.isclose(
                spec.actual_duration_seconds, spec.frames / FPS
            ))

    def test_presets_keep_exact_algorithms_not_only_step_counts(self) -> None:
        self.assertEqual(
            ORIGINAL_PRESETS["fast"]["actual_step_indices"],
            [0, 1, 2, 3, 4, 8, 13, 19],
        )
        self.assertEqual(ORIGINAL_PRESETS["balanced"]["actual_steps"], 9)
        self.assertEqual(ORIGINAL_PRESETS["balanced"]["forecast_steps"], 11)
        self.assertEqual(ORIGINAL_PRESETS["quality"]["actual_steps"], 12)
        self.assertEqual(ORIGINAL_PRESETS["ultra"]["actual_steps"], 20)
        self.assertEqual(ORIGINAL_PRESETS["ultra"]["forecast_steps"], 0)
        self.assertEqual(
            {name: preset["steps"] for name, preset in LORA_PRESETS.items()},
            {"fast": 4, "balanced": 5, "quality": 6, "ultra": 8},
        )
        self.assertTrue(all(preset["strength"] == 1.0 for preset in LORA_PRESETS.values()))

    def test_request_round_trip_preserves_seed_and_resolved_plan(self) -> None:
        original = self.spec(
            engine="lora", quality="quality", duration_seconds=15,
            resolution="720p", aspect_ratio="9:16", seed=2**64 - 1,
        )
        document = json.loads(json.dumps(
            original.to_dict(include_execution=True), ensure_ascii=False
        ))
        restored = GenerationSpec.from_mapping(document)
        self.assertEqual(restored, original)
        self.assertEqual(document["execution"]["steps"], 6)


class ArtifactPathContractTest(unittest.TestCase):
    def test_uploaded_media_cache_identity_follows_content_not_job_path(self) -> None:
        with tempfile.TemporaryDirectory(prefix="h3-content-cache-") as directory:
            root = Path(directory)
            first = root / "job-a" / "first.png"
            second = root / "job-b" / "first.png"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"same uploaded image bytes")
            second.write_bytes(first.read_bytes())
            self.assertNotEqual(first.resolve(), second.resolve())
            self.assertEqual(
                NativeT2AVHotSession._file_content_digest(first),
                NativeT2AVHotSession._file_content_digest(second),
            )

    def test_manifest_has_one_safe_pinned_artifact_per_required_role(self) -> None:
        manifest = json.loads(
            (SERVE_ROOT / "models/manifest.json").read_text(encoding="utf-8")
        )
        roles = {artifact["role"] for artifact in manifest["artifacts"]}
        required_roles = {
            "diffusion_model", "reference_diffusion_model",
            "diffusion_model_w4a8", "reference_diffusion_model_w4a8",
            "text_encoder", "video_vae", "audio_vae", "turbo_lora",
            "latent_upscaler",
        }
        self.assertTrue(required_roles.issubset(roles))
        self.assertEqual(len(roles), len(manifest["artifacts"]))
        seen_paths = set()
        for artifact in manifest["artifacts"]:
            install = PurePosixPath(artifact["install_path"])
            self.assertFalse(install.is_absolute())
            self.assertNotIn("..", install.parts)
            self.assertNotIn(str(install), seen_paths)
            seen_paths.add(str(install))
            self.assertGreater(artifact["bytes"], 0)
            self.assertRegex(artifact["sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(artifact["revision"], r"^[0-9a-f]{40}$")

    def test_persisted_job_does_not_recompute_duration_or_seed(self) -> None:
        class NoopBackend:
            key = None

        with tempfile.TemporaryDirectory(prefix="h3-persist-contract-") as directory:
            data = Path(directory)
            service = JobService(data, NoopBackend())
            job = JobRecord(
                id="round-trip",
                spec=GenerationSpec.from_mapping({
                    "prompt": "fifteen seconds",
                    "duration_seconds": 15,
                    "seed": 18446744073709551615,
                }),
            )
            service.jobs[job.id] = job
            service.persist(job)
            restored = JobService(data, NoopBackend()).jobs[job.id]
            self.assertEqual(restored.spec.frames, 362)
            self.assertEqual(restored.spec.requested_duration_seconds, 15)
            self.assertEqual(restored.spec.seed, 2**64 - 1)


class AcceptancePolicyContractTest(unittest.TestCase):
    def test_contract_makes_visual_gate_prior_and_ssim_non_blocking(self) -> None:
        contract = (SERVE_ROOT / "docs/COMFY_MIGRATION_CONTRACT.md").read_text(
            encoding="utf-8"
        )
        visual = contract.index("先**通过多帧视觉门控")
        numeric = contract.index("通过视觉门控后才记录")
        self.assertLess(visual, numeric)
        self.assertIn("SSIM 只作同 seed 数值差异诊断", contract)
        self.assertIn("不得**作为质量硬门", contract)


if __name__ == "__main__":
    unittest.main()
