from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
import unittest
import io
import wave
from pathlib import Path
import aiohttp
import av
import numpy as np
from aiohttp.test_utils import AioHTTPTestCase
from PIL import Image

from h3serve.app import (
    JobRecord, JobService, create_app,
    _load_persisted_mimo_key, _mimo_key_path,
)
from h3serve.backend import CheckpointResult, GenerationResult, JobCancelled
from h3serve.config import ServicePaths
from h3serve.contract import GenerationSpec, SecondSamplingSpec
from h3serve.prompt_enhancer import EnhancementRequest
from h3serve.memory_policy import HOST_MEMORY_PROFILES


class FakeBackend:
    def __init__(self, video_path: Path) -> None:
        self.video_path = video_path
        self.key: str | None = None
        self.preloaded: str | None = None
        self.warm_state = {"status": "cold", "engine": None}
        self.reference_images: tuple[Path, ...] = ()
        self.reference_videos: tuple[Path, ...] = ()
        self.reference_audios: tuple[Path, ...] = ()
        self.last_spec = None
        self.lora_checkpoint: Path | None = None
        self.fail_lora_checkpoint: str | None = None

    async def preload(self, engine: str) -> None:
        self.preloaded = engine
        if (
            self.lora_checkpoint is not None
            and self.lora_checkpoint.name == self.fail_lora_checkpoint
        ):
            self.warm_state = {"status": "failed", "engine": engine}
            return
        self.warm_state = {
            "status": "ready", "engine": engine,
            "lora_checkpoint": (
                self.lora_checkpoint.name if self.lora_checkpoint else None
            ),
        }

    def configure_lora_checkpoint(self, checkpoint: Path) -> None:
        self.lora_checkpoint = Path(checkpoint)

    async def generate(
        self, spec, _job_id: str, _first_frame: Path | None,
        _last_frame: Path | None, _reference_images: tuple[Path, ...],
        _reference_videos: tuple[Path, ...], _reference_audios: tuple[Path, ...], cancel_event: asyncio.Event,
        progress_callback=None, **preview_callbacks,
    ) -> GenerationResult:
        self.last_spec = spec
        self.reference_images = _reference_images
        self.reference_videos = _reference_videos
        self.reference_audios = _reference_audios
        self.key = (
            "reference:native-sm89" if spec.engine == "reference" else
            "original:balanced" if spec.engine == "original" else "turbo:shared"
        )
        if cancel_event.is_set():
            raise JobCancelled("cancelled")
        if progress_callback is not None:
            progress_callback({"percent": 50, "stage": "denoise", "detail": "1/2"})
        checkpoint_path = preview_callbacks.get("checkpoint_path")
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"formal-checkpoint")
            preview_path = None
            if spec.checkpoint_preview:
                preview_path = self.video_path.with_name("checkpoint-preview.mp4")
                preview_path.write_bytes(b"checkpoint-preview")
            return CheckpointResult(
                runtime_key=self.key,
                elapsed_seconds=0.5,
                checkpoint_path=checkpoint_path,
                preview_path=preview_path,
                completed_steps=int(spec.checkpoint_step),
                total_steps=20 if spec.model_variant == "base" else int(spec.preset["steps"]),
            )
        preview_ready = preview_callbacks.get("preview_ready_callback")
        if preview_ready is not None:
            preview_path = self.video_path.with_name("preview.mp4")
            preview_path.write_bytes(b"preview-video")
            preview_ready({"output_path": str(preview_path)})
            wait_decision = preview_callbacks.get("preview_decision_wait")
            if wait_decision is not None:
                decision = await asyncio.to_thread(wait_decision)
                if decision != "continue":
                    raise JobCancelled("preview discarded")
        latent_path = self.video_path.with_suffix(".pt")
        latent_path.write_bytes(b"clean-h3-av-latent")
        return GenerationResult(
            runtime_key=self.key,
            elapsed_seconds=1.25,
            output_path=self.video_path,
            final_latents_path=latent_path,
        )

    async def second_sample(
        self, spec, second_sampling: SecondSamplingSpec, source_latents_path,
        job_id, _first_frame, _last_frame, _reference_images,
        _reference_videos, _reference_audios, cancel_event,
        progress_callback=None,
    ) -> GenerationResult:
        if cancel_event.is_set():
            raise JobCancelled("cancelled")
        self.last_spec = spec
        self.last_second_sampling = second_sampling
        output = self.video_path.with_name(f"{job_id}.mp4")
        output.write_bytes(b"h3-second-sampled-video")
        latent_path = output.with_suffix(".pt")
        latent_path.write_bytes(b"second-pass-clean-latent")
        if progress_callback:
            progress_callback({
                "percent": 70, "stage": "second_sampling", "detail": "1/1",
            })
        return GenerationResult(
            runtime_key="original:native-sm89",
            elapsed_seconds=2.0,
            output_path=output,
            inference_plan={"ultimate_upscale": {"full_canvas": True}},
            final_latents_path=latent_path,
        )

    async def stop(self) -> None:
        self.key = None
        self.preloaded = None
        self.warm_state = {"status": "cold", "engine": None}

    def preflight(self, _engine: str) -> dict:
        return {"ready": True, "checks": {"fake": True}}


class FakePromptEnhancer:
    def __init__(self) -> None:
        self.api_key: str | None = None
        self.request: EnhancementRequest | None = None
        self.images = ()
        self.videos = ()
        self.audios = ()

    async def enhance(self, *, api_key, request, images=(), videos=(), audios=()):
        self.api_key = api_key
        self.request = request
        self.images = images
        self.videos = videos
        self.audios = audios
        return {
            "shots": list(request.shots),
            "soundtrack": {
                "overall_soundscape": "海风与自行车链条声",
                "non_diegetic_music": "N/A",
            },
        }


class FakeUpscaler:
    def __init__(self):
        self.stop_calls = 0

    def status(self):
        return {"ready": True, "implementation": "fake", "missing": []}

    async def upscale(
        self, source, *, target_width, target_height, cancel_event,
        progress_callback=None,
    ):
        from h3serve.upscaler import UpscaleResult

        if progress_callback:
            progress_callback({
                "percent": 50, "stage": "upscaling", "detail": "fake upscale"
            })
        output = source.with_name("upscaled.mp4")
        output.write_bytes(source.read_bytes() + b"-upscaled")
        return UpscaleResult(
            output, 2.5, target_width, target_height, 9000.0, 9800.0
        )

    async def stop(self):
        self.stop_calls += 1


class ApiTest(AioHTTPTestCase):
    async def get_application(self):
        self.temporary = Path(tempfile.mkdtemp(prefix="h3serve-test-"))
        self.video = self.temporary / "result.mp4"
        self.video.write_bytes(b"test-video")
        serve_dir = Path(__file__).resolve().parents[1]
        paths = ServicePaths.defaults(self.temporary, data_dir=self.temporary / "data")
        self.prompt_enhancer = FakePromptEnhancer()
        return create_app(
            paths=paths,
            serve_dir=serve_dir,
            api_key="secret",
            backend=FakeBackend(self.video),
            prompt_enhancer=self.prompt_enhancer,
        )

    async def asyncTearDown(self) -> None:
        await super().asyncTearDown()
        shutil.rmtree(self.temporary, ignore_errors=True)

    async def test_auth_contract_queue_and_video(self) -> None:
        response = await self.client.get("/api/v1/options")
        self.assertEqual(response.status, 401)

        headers = {"X-API-Key": "secret"}
        response = await self.client.get("/api/v1/options", headers=headers)
        self.assertEqual(response.status, 200)
        options = await response.json()
        self.assertEqual(options["deployment_mode"], "fixed_engine")
        self.assertEqual(options["current_engine"], "first_last")
        self.assertEqual(set(options["engines"]), {"original", "lora"})
        self.assertEqual(options["defaults"]["quality"], "balanced")
        self.assertIn("1080p", options["resolutions"])
        self.assertNotIn("2k", options["resolutions"])
        self.assertIn(
            "1440p", options["advanced_limits"]["second_sampling"]["levels"]
        )
        self.assertEqual(options["duration"]["max_by_resolution"]["1080p"], 15)
        self.assertEqual(options["duration"]["max_by_preset"]["1080p"]["4:3"], 15.0)
        self.assertIn("1440p", options["advanced_limits"]["upscaler"]["levels"])
        self.assertFalse(options["advanced_limits"]["sparse_attention_available"])
        self.assertEqual(
            options["advanced_limits"]["acceleration"]["scheduler"],
            "h3_int8_frozen_round229",
        )
        self.assertEqual(
            options["advanced_limits"]["acceleration"]["scheduler_by_variant"],
            {
                "base": "h3_int8_frozen_round229",
                "lora": "h3_lora_v1_no_forecast_round229",
            },
        )
        self.assertEqual(self.app["job_service"].backend.preloaded, "fl2va_int8_24gb")

        health = await (await self.client.get("/healthz")).json()
        self.assertEqual(health["warm_state"]["status"], "ready")

        response = await self.client.post("/api/v1/generations", headers=headers, json={
            "prompt": "A short stable scene.",
            "resolution": "480p",
            "aspect_ratio": "16:9",
            "duration_seconds": 5,
            "seed": 4404,
        })
        self.assertEqual(response.status, 202)
        job_id = (await response.json())["id"]

        job = None
        for _ in range(50):
            response = await self.client.get(f"/api/v1/jobs/{job_id}", headers=headers)
            job = await response.json()
            if job["status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertIsNotNone(job)
        self.assertEqual(job["status"], "succeeded")
        self.assertEqual(job["request"]["engine"], "original")
        self.assertEqual(job["request"]["quality"], "balanced")
        self.assertEqual(job["progress"]["percent"], 100.0)
        self.assertEqual(job["progress"]["stage"], "completed")
        self.assertEqual(job["progress"]["estimated_remaining_seconds"], 0.0)
        self.assertEqual(job["elapsed_seconds"], 1.25)
        self.assertNotIn("execution", job["request"])
        self.assertNotIn("runtime_key", job)

        response = await self.client.get(f"/api/v1/jobs/{job_id}/video", headers=headers)
        self.assertEqual(response.status, 200)
        self.assertEqual(await response.read(), b"test-video")

    async def test_settings_can_discover_and_switch_native_h3_lora(self) -> None:
        from safetensors.numpy import save_file

        lora_root = self.temporary / "models" / "loras"
        lora_root.mkdir(parents=True)
        compatible = lora_root / "release-v2.safetensors"
        save_file({
            "blocks.0.attn.to_q.lora_A.weight": np.zeros((1, 1), dtype=np.float16),
            "blocks.0.attn.to_q.lora_B.weight": np.zeros((1, 1), dtype=np.float16),
        }, compatible, metadata={"base_model": "MiniMax-H3"})
        save_file({
            "diffusion_model.other.weight": np.zeros((1, 1), dtype=np.float16),
        }, lora_root / "foreign.safetensors")
        lightx = lora_root / "minimax_h3_fl2v_turbo_4step_v1.1_768p_bf16.safetensors"
        lightx_state = {}
        for index in range(312):
            prefix = f"synthetic.{index}"
            lightx_state[f"{prefix}.lora_A.default.weight"] = np.zeros(
                (1, 1), dtype=np.float16
            )
            lightx_state[f"{prefix}.lora_B.default.weight"] = np.zeros(
                (1, 1), dtype=np.float16
            )
        save_file(
            lightx_state,
            lightx,
            metadata={"key_format": "minimax-h3-diffusers", "alpha": "128"},
        )

        headers = {"X-API-Key": "secret"}
        response = await self.client.get("/api/v1/settings/lora", headers=headers)
        self.assertEqual(response.status, 200)
        catalog = await response.json()
        by_id = {item["id"]: item for item in catalog["available"]}
        self.assertTrue(by_id["release-v2.safetensors"]["compatible"])
        self.assertEqual(by_id["release-v2.safetensors"]["pair_count"], 1)
        self.assertFalse(by_id["foreign.safetensors"]["compatible"])
        lightx_item = by_id[lightx.name]
        self.assertTrue(lightx_item["compatible"])
        self.assertEqual(lightx_item["pair_count"], 312)
        self.assertEqual(
            lightx_item["profile"]["profile_id"],
            "lightx2v_fl2v_4step_v1_1_768p",
        )
        self.assertEqual(lightx_item["profile"]["default_steps"], 4)

        response = await self.client.put(
            "/api/v1/settings/lora", headers=headers,
            json={"checkpoint": "release-v2.safetensors"},
        )
        self.assertEqual(response.status, 200)
        changed = await response.json()
        self.assertTrue(changed["changed"])
        self.assertEqual(changed["selected"], "release-v2.safetensors")
        self.assertEqual(changed["loaded"], "release-v2.safetensors")
        self.assertEqual(
            json.loads(
                (self.temporary / "data/settings/lora.json").read_text(
                    encoding="utf-8"
                )
            )["checkpoint"],
            "release-v2.safetensors",
        )

        response = await self.client.put(
            "/api/v1/settings/lora", headers=headers,
            json={"checkpoint": "foreign.safetensors"},
        )
        self.assertEqual(response.status, 400)

        failing = lora_root / "broken-build.safetensors"
        save_file({
            "blocks.0.attn.to_q.lora_A.weight": np.zeros((1, 1), dtype=np.float16),
            "blocks.0.attn.to_q.lora_B.weight": np.zeros((1, 1), dtype=np.float16),
        }, failing, metadata={"base_model": "MiniMax-H3"})
        backend = self.app["job_service"].backend
        backend.fail_lora_checkpoint = failing.name
        response = await self.client.put(
            "/api/v1/settings/lora", headers=headers,
            json={"checkpoint": failing.name},
        )
        self.assertEqual(response.status, 500)
        self.assertEqual(backend.lora_checkpoint.name, "release-v2.safetensors")
        self.assertEqual(backend.warm_state["status"], "ready")
        persisted = json.loads(
            (self.temporary / "data/settings/lora.json").read_text(encoding="utf-8")
        )
        self.assertEqual(persisted["checkpoint"], "release-v2.safetensors")

    async def test_completed_card_can_queue_native_h3_second_sampling(self) -> None:
        headers = {"X-API-Key": "secret"}
        response = await self.client.post(
            "/api/v1/generations", headers=headers, json={
                "prompt": "low resolution selectable card",
                "resolution": "480p", "aspect_ratio": "16:9",
                "duration_seconds": 15,
                "model_variant": "lora",
            },
        )
        source_id = (await response.json())["id"]
        for _ in range(80):
            source = await (
                await self.client.get(
                    f"/api/v1/jobs/{source_id}", headers=headers
                )
            ).json()
            if source["status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(source["status"], "succeeded")
        self.assertEqual(source["request"]["model_variant"], "lora")
        self.assertTrue(source["second_sampling_available"])

        response = await self.client.post(
            f"/api/v1/jobs/{source_id}/second-sampling",
            headers=headers,
            json={
                "resolution": "1080p", "steps": 1,
                "acceleration": 75, "strength": "enhance",
                "memory_mode": "auto",
                "temporal_window_frames": 119,
            },
        )
        self.assertEqual(response.status, 202, await response.text())
        child_id = (await response.json())["id"]
        for _ in range(80):
            child = await (
                await self.client.get(
                    f"/api/v1/jobs/{child_id}", headers=headers
                )
            ).json()
            if child["status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(child["status"], "succeeded")
        self.assertEqual(child["second_sampling"]["source_job_id"], source_id)
        self.assertEqual(child["request"]["model_variant"], "base")
        self.assertEqual(child["second_sampling"]["model_variant"], "base")
        self.assertEqual(child["second_sampling"]["strength"], "enhance")
        self.assertEqual(child["second_sampling"]["denoise"], 0.25)
        self.assertEqual(
            child["second_sampling"]["temporal_window_frames"], 119
        )
        self.assertEqual(
            (child["second_sampling"]["width"], child["second_sampling"]["height"]),
            (1920, 1088),
        )
        self.assertTrue(child["inference_plan"]["ultimate_upscale"]["full_canvas"])
        video = await self.client.get(
            f"/api/v1/jobs/{child_id}/video", headers=headers
        )
        self.assertEqual(await video.read(), b"h3-second-sampled-video")

    async def test_latent_cache_clear_keeps_history_and_videos(self) -> None:
        service = self.app["job_service"]
        latent_root = service.output_root / ".h3-latents"
        latent_root.mkdir(parents=True, exist_ok=True)
        latent = latent_root / "cache-test.pt"
        latent.write_bytes(b"reproducible-clean-av-latent")
        checkpoint = service.data_dir / "checkpoints" / "cache-test.pt"
        checkpoint.write_bytes(b"formal-checkpoint-tensor")
        spec = GenerationSpec.from_mapping({
            "prompt": "cache cleanup card", "resolution": "480p",
        })
        job = JobRecord(
            id="cache-test", spec=spec, status="succeeded",
            output_path=self.video, final_latents_path=latent,
            checkpoint_path=checkpoint, checkpoint_retained=True,
        )
        service.jobs[job.id] = job
        service.persist(job)

        response = await self.client.delete(
            "/api/v1/cache/latents", headers={"X-API-Key": "secret"}
        )
        self.assertEqual(response.status, 200, await response.text())
        result = await response.json()
        self.assertEqual(result["removed_files"], 2)
        self.assertEqual(result["removed_checkpoint_files"], 1)
        self.assertFalse(latent.exists())
        self.assertFalse(checkpoint.exists())
        self.assertTrue(self.video.exists())
        self.assertIn(job.id, service.jobs)
        self.assertIsNone(service.jobs[job.id].final_latents_path)
        self.assertIsNone(service.jobs[job.id].checkpoint_path)
        self.assertFalse(service.serialize(service.jobs[job.id])["second_sampling_available"])

    async def test_w4a8_completed_card_exposes_second_sampling_entry(self) -> None:
        service = self.app["job_service"]
        latent_root = service.output_root / ".h3-latents"
        latent_root.mkdir(parents=True, exist_ok=True)
        latent = latent_root / "w4a8-source.pt"
        latent.write_bytes(b"retained-w4a8-clean-av-latent")
        spec = GenerationSpec.from_mapping({
            "prompt": "8GB selectable card",
            "runtime_launcher": "fl2va_w4a8_8gb",
            "resolution": "480p",
        })
        job = JobRecord(
            id="w4a8-source",
            spec=spec,
            status="succeeded",
            output_path=self.video,
            final_latents_path=latent,
        )
        self.assertTrue(service.serialize(job)["second_sampling_available"])

    async def test_public_second_sampling_uses_1440p_name(self) -> None:
        service = self.app["job_service"]
        source = GenerationSpec.from_mapping({
            "prompt": "public resolution name",
            "resolution": "480p",
        })
        second = SecondSamplingSpec.from_mapping({
            "resolution": "1440p",
            "steps": 1,
        }, source=source)
        job = JobRecord(
            id="public-1440p",
            spec=source,
            status="queued",
            second_sampling=second,
        )
        public = service.serialize(job)
        self.assertEqual(second.resolution, "2k")
        self.assertEqual(public["second_sampling"]["resolution"], "1440p")

    async def test_generation_limits_are_saved_and_drive_options_and_validation(self) -> None:
        headers = {"X-API-Key": "secret"}
        current = await (
            await self.client.get(
                "/api/v1/settings/generation-limits", headers=headers
            )
        ).json()
        limits = current["preset_limits"]
        limits["720p"]["1:1"] = 9
        limits["720p"]["16:9"] = 12
        limits["1080p"]["16:9"] = 10
        response = await self.client.put(
            "/api/v1/settings/generation-limits",
            headers=headers,
            json={"preset_limits": limits},
        )
        self.assertEqual(response.status, 200)
        policy = await response.json()
        self.assertEqual(policy["preset_limits"]["720p"]["1:1"], 9)
        self.assertEqual(policy["preset_limits"]["720p"]["16:9"], 12)
        self.assertEqual(policy["preset_limits"]["1080p"]["16:9"], 10)
        self.assertIn("detected_vram_gib", policy)

        options = await (
            await self.client.get("/api/v1/options", headers=headers)
        ).json()
        self.assertEqual(options["duration"]["max_by_preset"]["720p"]["1:1"], 9)
        self.assertEqual(options["duration"]["max_by_preset"]["720p"]["16:9"], 12)
        self.assertEqual(options["duration"]["max_by_preset"]["1080p"]["16:9"], 10)
        rejected = await self.client.post(
            "/api/v1/generations",
            headers=headers,
            json={
                "prompt": "too long for the configured preset ceiling",
                "resolution": "1080p",
                "aspect_ratio": "16:9",
                "duration_seconds": 10.5,
            },
        )
        self.assertEqual(rejected.status, 400)
        self.assertIn("configured server limit", await rejected.text())
        self.assertTrue(
            (self.temporary / "data/settings/generation_limits.json").is_file()
        )

    async def test_fl2va_generation_forwards_one_raw_prompt_without_enhancement(self) -> None:
        prompt = "  自由文本第一段。\n\noverall_soundscape: 保持这一行原样。  "
        headers = {"X-API-Key": "secret"}
        response = await self.client.post(
            "/api/v1/generations", headers=headers,
            json={"prompt": prompt, "duration_seconds": 5},
        )
        self.assertEqual(response.status, 202, await response.text())
        job_id = (await response.json())["id"]
        for _ in range(50):
            state = await (
                await self.client.get(f"/api/v1/jobs/{job_id}", headers=headers)
            ).json()
            if state["status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(self.app["job_service"].backend.last_spec.prompt, prompt)
        self.assertIsNone(self.prompt_enhancer.request)

    async def test_1080p_duration_uses_the_native_pixel_frame_budget(self) -> None:
        headers = {"X-API-Key": "secret"}
        rejected = await self.client.post(
            "/api/v1/generations", headers=headers, json={
                "prompt": "too long at 1080p", "resolution": "1080p",
                "aspect_ratio": "16:9", "duration_seconds": 15.5,
            },
        )
        rejected_body = await rejected.text()
        self.assertEqual(rejected.status, 400, rejected_body)
        self.assertIn("at most 15.000 seconds", rejected_body)

        accepted = await self.client.post(
            "/api/v1/generations", headers=headers, json={
                "prompt": "validated 1080p", "resolution": "1080p",
                "aspect_ratio": "16:9", "duration_seconds": 15,
            },
        )
        self.assertEqual(accepted.status, 202, await accepted.text())
        request = (await accepted.json())["request"]
        self.assertEqual((request["width"], request["height"]), (1920, 1088))
        self.assertEqual(request["frames"], 362)

        accepted_four_three = await self.client.post(
            "/api/v1/generations", headers=headers, json={
                "prompt": "validated longer 1080p 4:3", "resolution": "1080p",
                "aspect_ratio": "4:3", "duration_seconds": 15,
            },
        )
        self.assertEqual(accepted_four_three.status, 202, await accepted_four_three.text())
        request = (await accepted_four_three.json())["request"]
        self.assertEqual((request["width"], request["height"], request["frames"]),
                         (1440, 1088, 362))

        accepted_square = await self.client.post(
            "/api/v1/generations", headers=headers, json={
                "prompt": "validated longer 1080p square", "resolution": "1080p",
                "aspect_ratio": "1:1", "duration_seconds": 15,
            },
        )
        self.assertEqual(accepted_square.status, 202, await accepted_square.text())

    async def test_2k_first_pass_is_rejected_and_reserved_for_second_sampling(self) -> None:
        response = await self.client.post(
            "/api/v1/generations",
            headers={"X-API-Key": "secret"},
            json={
                "prompt": "experimental full-context 2k route",
                "resolution": "2k",
                "aspect_ratio": "16:9",
                "duration_seconds": 15,
                "memory_mode": "auto",
            },
        )
        self.assertEqual(response.status, 400, await response.text())
        self.assertIn("1080p", await response.text())

    async def test_request_can_hot_switch_variant_inside_fixed_family(self) -> None:
        response = await self.client.post(
            "/api/v1/generations",
            headers={"X-API-Key": "secret"},
            json={"prompt": "hot lora", "model_variant": "lora"},
        )
        self.assertEqual(response.status, 202, await response.text())
        submitted = await response.json()
        self.assertEqual(submitted["request"]["engine"], "lora")

    async def test_preset_request_accepts_direct_steps_and_acceleration(self) -> None:
        response = await self.client.post(
            "/api/v1/generations",
            headers={"X-API-Key": "secret"},
            json={
                "prompt": "preset geometry with direct execution controls",
                "resolution": "480p",
                "aspect_ratio": "16:9",
                "duration_seconds": 5,
                "sampling_steps": 15,
                "acceleration": 60,
            },
        )
        self.assertEqual(response.status, 202, await response.text())
        request = (await response.json())["request"]
        self.assertFalse(request["advanced"])
        self.assertEqual(request["sampling_steps"], 15)
        self.assertEqual(request["acceleration"], 60.0)

    async def test_scheduled_lora_checkpoint_releases_worker_and_can_resume(self) -> None:
        headers = {"X-API-Key": "secret"}
        response = await self.client.post(
            "/api/v1/generations", headers=headers, json={
                "prompt": "scheduled LoRA checkpoint task",
                "service_family": "first_last",
                "model_variant": "lora",
                "mode": "advanced",
                "width": 864,
                "height": 480,
                "duration_seconds": 5,
                "sampling_steps": 8,
                "acceleration": 50,
                "execution_mode": "checkpoint",
                "checkpoint_step": 3,
                "checkpoint_retain": True,
                "checkpoint_preview": False,
            },
        )
        self.assertEqual(response.status, 202, await response.text())
        job_id = (await response.json())["id"]
        for _ in range(100):
            state = await (
                await self.client.get(f"/api/v1/jobs/{job_id}", headers=headers)
            ).json()
            if state["status"] == "checkpointed":
                break
            await asyncio.sleep(0.01)
        self.assertEqual(state["status"], "checkpointed")
        self.assertEqual(state["checkpoint"]["completed_steps"], 3)
        self.assertEqual(state["checkpoint"]["total_steps"], 8)
        self.assertTrue(state["checkpoint"]["resume_available"])
        self.assertNotIn("preview", state)
        self.assertEqual(state["request"]["model_variant"], "lora")
        self.assertEqual(state["request"]["sampling_steps"], 8)
        self.assertEqual(state["request"]["acceleration"], 50.0)

        # A second job is not blocked by the stopped checkpoint task.
        second = await self.client.post(
            "/api/v1/generations", headers=headers, json={"prompt": "next job"}
        )
        second_id = (await second.json())["id"]
        for _ in range(100):
            second_state = await (
                await self.client.get(f"/api/v1/jobs/{second_id}", headers=headers)
            ).json()
            if second_state["status"] == "succeeded":
                break
            await asyncio.sleep(0.01)
        self.assertEqual(second_state["status"], "succeeded")

        resumed = await self.client.post(
            f"/api/v1/jobs/{job_id}/resume", headers=headers
        )
        self.assertEqual(resumed.status, 202, await resumed.text())
        for _ in range(100):
            state = await (
                await self.client.get(f"/api/v1/jobs/{job_id}", headers=headers)
            ).json()
            if state["status"] == "succeeded":
                break
            await asyncio.sleep(0.01)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(state["generation_elapsed_seconds"], 1.75)

    async def test_failed_resume_with_retained_checkpoint_can_be_retried(self) -> None:
        headers = {"X-API-Key": "secret"}
        response = await self.client.post(
            "/api/v1/generations", headers=headers, json={
                "prompt": "retry retained formal checkpoint",
                "execution_mode": "checkpoint",
                "checkpoint_step": 3,
                "checkpoint_retain": True,
            },
        )
        job_id = (await response.json())["id"]
        for _ in range(100):
            state = await (
                await self.client.get(f"/api/v1/jobs/{job_id}", headers=headers)
            ).json()
            if state["status"] == "checkpointed":
                break
            await asyncio.sleep(0.01)
        service = self.app["job_service"]
        job = service.jobs[job_id]
        job.status = "failed"
        job.error = "simulated resume failure"
        service.persist(job)
        failed_state = await (
            await self.client.get(f"/api/v1/jobs/{job_id}", headers=headers)
        ).json()
        self.assertTrue(failed_state["checkpoint"]["resume_available"])

        retried = await self.client.post(
            f"/api/v1/jobs/{job_id}/resume", headers=headers
        )
        self.assertEqual(retried.status, 202, await retried.text())

    async def test_fork_preview_is_replaced_by_second_sampling(self) -> None:
        headers = {"X-API-Key": "secret"}
        response = await self.client.post(
            "/api/v1/generations", headers=headers, json={
                "prompt": "preview branch",
                "preview_mode": "pause",
                "preview_branch_steps": 2,
            },
        )
        self.assertEqual(response.status, 400)
        self.assertIn("native H3 second sampling", await response.text())

    async def test_checkpoint_preview_is_generated_and_resumable(self) -> None:
        headers = {"X-API-Key": "secret"}
        response = await self.client.post(
            "/api/v1/generations", headers=headers, json={
                "prompt": "retired checkpoint preview",
                "execution_mode": "checkpoint",
                "checkpoint_step": 3,
                "checkpoint_preview": True,
            },
        )
        self.assertEqual(response.status, 202, await response.text())
        job_id = (await response.json())["id"]
        for _ in range(100):
            state = await (
                await self.client.get(
                    f"/api/v1/jobs/{job_id}", headers=headers
                )
            ).json()
            if state["status"] == "checkpointed":
                break
            await asyncio.sleep(0.01)
        self.assertEqual(state["status"], "checkpointed")
        self.assertTrue(state["preview"]["ready"])
        self.assertTrue(state["checkpoint"]["resume_available"])

    async def test_checkpoint_preview_defaults_on_for_legacy_console(self) -> None:
        headers = {"X-API-Key": "secret"}
        response = await self.client.post(
            "/api/v1/generations", headers=headers, json={
                "prompt": "legacy console checkpoint",
                "execution_mode": "checkpoint",
                "checkpoint_step": 3,
            },
        )
        self.assertEqual(response.status, 202, await response.text())
        job_id = (await response.json())["id"]
        for _ in range(100):
            state = await (
                await self.client.get(
                    f"/api/v1/jobs/{job_id}", headers=headers
                )
            ).json()
            if state["status"] == "checkpointed":
                break
            await asyncio.sleep(0.01)
        self.assertEqual(state["status"], "checkpointed")
        self.assertTrue(state["request"]["checkpoint_preview"])
        self.assertTrue(state["preview"]["ready"])

    async def test_stale_same_origin_console_false_still_gets_preview(self) -> None:
        origin = str(self.client.make_url("/")).rstrip("/")
        headers = {"X-API-Key": "secret", "Origin": origin}
        response = await self.client.post(
            "/api/v1/generations", headers=headers, json={
                "prompt": "stale console checkpoint",
                "execution_mode": "checkpoint",
                "checkpoint_step": 3,
                "checkpoint_preview": False,
            },
        )
        self.assertEqual(response.status, 202, await response.text())
        job_id = (await response.json())["id"]
        for _ in range(100):
            state = await (
                await self.client.get(
                    f"/api/v1/jobs/{job_id}",
                    headers={"X-API-Key": "secret"},
                )
            ).json()
            if state["status"] == "checkpointed":
                break
            await asyncio.sleep(0.01)
        self.assertEqual(state["status"], "checkpointed")
        self.assertTrue(state["request"]["checkpoint_preview"])
        self.assertTrue(state["preview"]["ready"])

    async def test_multipart_console_without_origin_cannot_disable_preview(self) -> None:
        form = aiohttp.FormData(default_to_multipart=True)
        for name, value in {
            "prompt": "embedded browser checkpoint",
            "execution_mode": "checkpoint",
            "checkpoint_step": "3",
            "checkpoint_preview": "false",
            "checkpoint_preview_steps": "4",
            "checkpoint_preview_resolution": "360p",
        }.items():
            form.add_field(name, value)
        response = await self.client.post(
            "/api/v1/generations",
            headers={"X-API-Key": "secret"},
            data=form,
        )
        self.assertEqual(response.status, 202, await response.text())
        job_id = (await response.json())["id"]
        for _ in range(100):
            state = await (
                await self.client.get(
                    f"/api/v1/jobs/{job_id}",
                    headers={"X-API-Key": "secret"},
                )
            ).json()
            if state["status"] == "checkpointed":
                break
            await asyncio.sleep(0.01)
        self.assertEqual(state["status"], "checkpointed")
        self.assertTrue(state["request"]["checkpoint_preview"])
        self.assertEqual(state["request"]["checkpoint_preview_steps"], 4)
        self.assertEqual(
            state["request"]["checkpoint_preview_resolution"], "360p"
        )
        self.assertTrue(state["preview"]["ready"])

    async def test_checkpoint_preview_global_settings_round_trip(self) -> None:
        headers = {"X-API-Key": "secret"}
        response = await self.client.put(
            "/api/v1/settings/checkpoint-preview",
            headers=headers,
            json={"steps": 6, "resolution": "480p"},
        )
        self.assertEqual(response.status, 200, await response.text())
        self.assertEqual(
            await response.json(),
            {
                "steps": 6,
                "resolution": "480p",
                "step_range": {"min": 1, "max": 8},
                "resolutions": ["360p", "480p", "720p"],
            },
        )
        response = await self.client.post(
            "/api/v1/generations",
            headers=headers,
            json={
                "prompt": "global checkpoint preview defaults",
                "execution_mode": "checkpoint",
                "checkpoint_step": 3,
                "checkpoint_preview": True,
            },
        )
        self.assertEqual(response.status, 202, await response.text())
        self.assertEqual(
            self.app["job_service"].backend.last_spec.checkpoint_preview_steps,
            6,
        )
        self.assertEqual(
            self.app["job_service"].backend.last_spec.checkpoint_preview_resolution,
            "480p",
        )

    async def test_legacy_upscale_request_points_to_native_second_sampling(self) -> None:
        response = await self.client.post(
            "/api/v1/generations", headers={"X-API-Key": "secret"}, json={
                "prompt": "upscaled scene", "seed": 7,
                "upscale_enabled": True, "upscale_mode": "basic",
                "upscale_resolution": "1080p",
            },
        )
        self.assertEqual(response.status, 400)
        self.assertIn("native H3 second sampling", await response.text())

    async def test_default_service_does_not_preload_retired_flashvsr(self) -> None:
        status = self.app["job_service"].upscaler.status()
        self.assertFalse(status["ready"])
        self.assertEqual(status["resident_state"], "removed")
        self.assertIn("second-sampling", status["replacement"])

    async def test_mimo_enhancement_key_is_ephemeral_request_metadata(self) -> None:
        storyboard = {
            "shots": [{
                "id": "shot-1", "duration_seconds": 5,
                "prompt": "女孩沿海岸骑自行车，只有海风和链条声。",
            }],
            "bgm_enabled": False,
            "bgm_style": "",
        }
        form = aiohttp.FormData()
        form.add_field("storyboard", json.dumps(storyboard, ensure_ascii=False))
        image_buffer = io.BytesIO()
        Image.new("RGB", (16, 16), (30, 80, 120)).save(image_buffer, format="PNG")
        form.add_field(
            "reference_image_1", image_buffer.getvalue(),
            filename="visual-reference.png", content_type="image/png",
        )
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16_000)
            writer.writeframes(b"\x00\x00" * 1_600)
        form.add_field(
            "reference_audio_1", audio_buffer.getvalue(),
            filename="voice.wav", content_type="audio/wav",
        )
        response = await self.client.post(
            "/studio/prompt-enhancements",
            headers={"X-API-Key": "secret", "X-MiMo-API-Key": "mimo-secret"},
            data=form,
        )
        self.assertEqual(response.status, 200)
        result = await response.json()
        self.assertEqual(result["soundtrack"]["non_diegetic_music"], "N/A")
        self.assertEqual(self.prompt_enhancer.api_key, "mimo-secret")
        self.assertEqual(self.prompt_enhancer.request.condition_mode, "T2VA")
        self.assertEqual(self.prompt_enhancer.images[0][0], "<Picture 1>")
        self.assertEqual(self.prompt_enhancer.audios[0][0], "<Audio 1>")
        self.assertEqual(self.prompt_enhancer.audios[0][1], "audio/wav")
        self.assertEqual(len(self.app["job_service"].jobs), 0)

    async def test_prompt_enhancement_is_not_a_public_api_route(self) -> None:
        response = await self.client.post(
            "/api/v1/prompt-enhancements",
            headers={"X-API-Key": "secret"},
            data={"storyboard": "{}"},
        )
        self.assertEqual(response.status, 404)

    async def test_console_mimo_key_is_shared_in_memory_with_api_clients(self) -> None:
        response = await self.client.put(
            "/api/v1/settings/mimo-key",
            headers={"X-API-Key": "secret"},
            json={"api_key": "console-mimo-secret"},
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(await response.json(), {"configured": True})

        status = await self.client.get(
            "/api/v1/settings/mimo-key", headers={"X-API-Key": "secret"}
        )
        self.assertEqual(await status.json(), {"configured": True})

        storyboard = {
            "shots": [{"id": "shot-1", "duration_seconds": 5,
                       "prompt": "女孩沿海岸骑车，保留海风声。"}],
            "bgm_enabled": False,
            "bgm_style": "",
        }
        response = await self.client.post(
            "/studio/prompt-enhancements",
            headers={"X-API-Key": "secret"},
            data={"storyboard": json.dumps(storyboard, ensure_ascii=False)},
        )
        self.assertEqual(response.status, 200, await response.text())
        self.assertEqual(self.prompt_enhancer.api_key, "console-mimo-secret")

        cleared = await self.client.put(
            "/api/v1/settings/mimo-key",
            headers={"X-API-Key": "secret"},
            json={"api_key": ""},
        )
        self.assertEqual(await cleared.json(), {"configured": False})

    async def test_console_mimo_key_persists_privately_and_can_be_cleared(self) -> None:
        response = await self.client.put(
            "/api/v1/settings/mimo-key",
            headers={"X-API-Key": "secret"},
            json={"api_key": "persistent-console-secret"},
        )
        self.assertEqual(response.status, 200)
        key_path = _mimo_key_path(self.temporary / "data")
        self.assertEqual(_load_persisted_mimo_key(self.temporary / "data"),
                         "persistent-console-secret")
        self.assertEqual(key_path.stat().st_mode & 0o777, 0o600)

        response = await self.client.put(
            "/api/v1/settings/mimo-key",
            headers={"X-API-Key": "secret"},
            json={"api_key": ""},
        )
        self.assertEqual(response.status, 200)
        self.assertFalse(key_path.exists())

    async def test_resource_snapshot_exposes_host_and_gpu_contract(self) -> None:
        response = await self.client.get(
            "/api/v1/resources", headers={"X-API-Key": "secret"}
        )
        self.assertEqual(response.status, 200)
        document = await response.json()
        self.assertIn("cpu", document)
        self.assertIn("memory", document)
        self.assertIn("gpu", document)
        self.assertIn("queue", document)
        self.assertGreater(document["memory"]["total_gib"], 0)

    async def test_queue_reorder_and_record_delete_contract(self) -> None:
        headers = {"X-API-Key": "secret"}
        # Stop the worker so both jobs remain reorderable.
        service = self.app["job_service"]
        service.worker_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await service.worker_task
        service.worker_task = None

        ids = []
        for seed in (1, 2):
            response = await self.client.post(
                "/api/v1/generations", headers=headers,
                json={"prompt": f"queued {seed}", "seed": seed},
            )
            self.assertEqual(response.status, 202)
            ids.append((await response.json())["id"])
        response = await self.client.put(
            "/api/v1/jobs/order", headers={**headers, "Content-Type": "application/json"},
            json={"job_ids": list(reversed(ids))},
        )
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["job_ids"], list(reversed(ids)))

        response = await self.client.delete(
            f"/api/v1/jobs/{ids[0]}/record", headers=headers
        )
        self.assertEqual(response.status, 200)
        response = await self.client.get(f"/api/v1/jobs/{ids[0]}", headers=headers)
        self.assertEqual(response.status, 404)

    async def test_record_delete_retains_untrusted_legacy_output(self) -> None:
        """A legacy result outside output/ must not make its card undeletable."""

        headers = {"X-API-Key": "secret"}
        service = self.app["job_service"]
        legacy_output = self.temporary / "runtime" / "old-smoke" / "result.mp4"
        legacy_output.parent.mkdir(parents=True)
        legacy_output.write_bytes(b"legacy-result")
        job = JobRecord(
            id="legacy-output-job",
            spec=GenerationSpec.from_mapping({"prompt": "legacy", "seed": 7}),
            status="succeeded",
            output_path=legacy_output,
        )
        service.jobs[job.id] = job
        service.cancel_events[job.id] = asyncio.Event()
        service.persist(job)

        response = await self.client.delete(
            f"/api/v1/jobs/{job.id}/record", headers=headers
        )
        self.assertEqual(response.status, 200)
        document = await response.json()
        self.assertTrue(document["deleted"])
        self.assertFalse(document["output_deleted"])
        self.assertTrue(document["output_retained"])
        self.assertTrue(legacy_output.is_file())
        self.assertNotIn(job.id, service.jobs)
        self.assertFalse((service.data_dir / "jobs" / f"{job.id}.json").exists())

    async def test_persisted_fifteen_second_job_round_trips(self) -> None:
        data = self.temporary / "roundtrip"
        backend = FakeBackend(self.video)
        service = JobService(data, backend)
        original = JobRecord(
            id="roundtrip-job",
            spec=GenerationSpec.from_mapping({
                "prompt": "fifteen seconds", "duration_seconds": 15, "seed": 1
            }),
            inference_plan={
                "policy_id": "h3_v19_human_aligned_budgeted_adaptive_inference",
                "accelerated": True,
                "execution_digest": "a" * 64,
            },
        )
        service.jobs[original.id] = original
        service.persist(original)
        restored = JobService(data, backend).jobs[original.id]
        self.assertEqual(restored.spec.requested_duration_seconds, 15)
        self.assertEqual(restored.spec.frames, 362)
        self.assertEqual(restored.inference_plan, original.inference_plan)
        self.assertEqual(
            service.serialize(original)["inference_plan"],
            original.inference_plan,
        )

    async def test_eta_counts_down_and_includes_jobs_ahead(self) -> None:
        service = self.app["job_service"]
        if service.worker_task is not None:
            service.worker_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await service.worker_task
            service.worker_task = None

        spec = GenerationSpec.from_mapping({"prompt": "eta", "seed": 1})
        running = JobRecord(
            id="eta-running", spec=spec, status="running",
            estimated_total_seconds=40.0, started_at=time.time() - 10.0,
        )
        first = JobRecord(
            id="eta-first", spec=spec, estimated_total_seconds=20.0,
            estimated_remaining_seconds=20.0,
        )
        second = JobRecord(
            id="eta-second", spec=spec, estimated_total_seconds=25.0,
            estimated_remaining_seconds=25.0,
        )
        service.jobs.update({job.id: job for job in (running, first, second)})
        service.pending[:] = [first.id, second.id]

        running_progress = service.serialize(running)["progress"]
        second_progress = service.serialize(second)["progress"]
        self.assertAlmostEqual(running_progress["estimated_remaining_seconds"], 30, delta=1)
        self.assertAlmostEqual(second_progress["estimated_queue_seconds"], 50, delta=1)
        self.assertAlmostEqual(second_progress["estimated_completion_seconds"], 75, delta=1)

    async def test_eta_history_matches_execution_parameters(self) -> None:
        service = self.app["job_service"]
        sparse = GenerationSpec.from_mapping({
            "prompt": "sparse history", "advanced": True, "width": 864,
            "height": 480, "frames": 124, "actual_steps": 9,
            "attention_keep_ratio": 0.75, "sparse_scope": "middle_only", "seed": 1,
        })
        dense = GenerationSpec.from_mapping({
            "prompt": "dense request", "advanced": True, "width": 864,
            "height": 480, "frames": 124, "actual_steps": 9,
            "attention_keep_ratio": 1.0, "sparse_scope": "middle_only", "seed": 2,
        })
        service.jobs["sparse-history"] = JobRecord(
            id="sparse-history", spec=sparse, status="succeeded", elapsed_seconds=9.0,
        )
        self.assertNotEqual(service._estimate_total(dense, "text"), 9.0)
        self.assertEqual(service._estimate_total(sparse, "text"), 9.0)

    async def test_starting_backend_eta_waits_for_model_preload(self) -> None:
        spec = GenerationSpec.from_mapping({"prompt": "wait for preload", "seed": 3})
        job = JobRecord(
            id="preload-eta", spec=spec, status="starting_backend",
            estimated_total_seconds=35.0, estimated_remaining_seconds=35.0,
            started_at=None,
        )
        progress = self.app["job_service"].serialize(job)["progress"]
        self.assertIsNone(progress["estimated_remaining_seconds"])
        self.assertIsNone(progress["estimated_completion_seconds"])


class UnifiedConsoleApiTest(AioHTTPTestCase):
    async def get_application(self):
        self.temporary = Path(tempfile.mkdtemp(prefix="h3serve-unified-test-"))
        self.video = self.temporary / "unified.mp4"
        self.video.write_bytes(b"unified-video")
        paths = ServicePaths.defaults(self.temporary, data_dir=self.temporary / "data")
        self.backend = FakeBackend(self.video)
        return create_app(
            paths=paths, serve_dir=Path(__file__).resolve().parents[1],
            backend=self.backend, fixed_engine=None, preload=False,
        )

    async def asyncTearDown(self) -> None:
        await super().asyncTearDown()
        shutil.rmtree(self.temporary, ignore_errors=True)

    async def test_enter_exit_and_switch_engine(self) -> None:
        options = await (await self.client.get("/api/v1/options")).json()
        self.assertEqual(options["deployment_mode"], "unified_console")
        self.assertIsNone(options["current_engine"])
        response = await self.client.post(
            "/api/v1/generations", json={"prompt": "must select first"}
        )
        self.assertEqual(response.status, 400)

        response = await self.client.put("/api/v1/engine", json={"engine": "lora"})
        self.assertEqual(response.status, 200, await response.text())
        self.assertEqual(self.backend.preloaded, "fl2va_int8_24gb")
        options = await (await self.client.get("/api/v1/options")).json()
        self.assertEqual(options["current_engine"], "first_last")
        self.assertEqual(options["current_model_variant"], "lora")
        self.assertEqual(options["defaults"]["quality"], "quality")

        response = await self.client.delete("/api/v1/engine")
        self.assertEqual(response.status, 200, await response.text())
        self.assertIsNone(self.backend.preloaded)
        options = await (await self.client.get("/api/v1/options")).json()
        self.assertIsNone(options["current_engine"])

        response = await self.client.put("/api/v1/engine", json={"engine": "reference"})
        self.assertEqual(response.status, 200, await response.text())
        self.assertEqual(self.backend.preloaded, "ref2va_int8_24gb")

        response = await self.client.delete("/api/v1/engine")
        self.assertEqual(response.status, 200, await response.text())
        response = await self.client.put("/api/v1/engine", json={"engine": "reference_lora"})
        self.assertEqual(response.status, 200, await response.text())
        self.assertEqual(self.backend.preloaded, "ref2va_int8_24gb")

        response = await self.client.delete("/api/v1/engine")
        self.assertEqual(response.status, 200, await response.text())
        response = await self.client.put(
            "/api/v1/engine",
            json={"launcher": "fl2va_w4a8_8gb", "model_variant": "lora"},
        )
        self.assertEqual(response.status, 200, await response.text())
        self.assertEqual(self.backend.preloaded, "fl2va_w4a8_8gb")
        options = await (await self.client.get("/api/v1/options")).json()
        self.assertEqual(options["current_launcher"], "fl2va_w4a8_8gb")
        self.assertEqual(options["active_weight_tier"], "w4a8")
        self.assertEqual(options["resolutions"], ["360p", "480p", "720p"])
        self.assertTrue(
            options["advanced_limits"]["second_sampling"]["available"]
        )
        self.assertEqual(
            options["advanced_limits"]["second_sampling"]["levels"],
            ["720p", "1080p"],
        )

        response = await self.client.delete("/api/v1/engine")
        self.assertEqual(response.status, 200, await response.text())
        response = await self.client.put(
            "/api/v1/engine",
            json={"launcher": "ref2va_int8_16gb"},
        )
        self.assertEqual(response.status, 200, await response.text())
        options = await (await self.client.get("/api/v1/options")).json()
        self.assertEqual(options["active_vram_profile"], "16gb")
        self.assertEqual(
            options["resolutions"], ["360p", "480p", "720p", "1080p"]
        )
        self.assertEqual(
            options["advanced_limits"]["second_sampling"]["levels"],
            ["720p", "1080p", "1440p"],
        )

    async def test_16gb_completed_card_can_queue_1440p_second_sampling(self) -> None:
        response = await self.client.put(
            "/api/v1/engine", json={"launcher": "ref2va_int8_16gb"}
        )
        self.assertEqual(response.status, 200, await response.text())
        service = self.app["job_service"]
        latent = service.output_root / ".h3-latents" / "ref16-source.pt"
        latent.parent.mkdir(parents=True, exist_ok=True)
        latent.write_bytes(b"ref16-clean-av-latent")
        source = JobRecord(
            id="ref16-source",
            spec=GenerationSpec.from_mapping({
                "prompt": "16GB 1440P second-sampling release boundary",
                "runtime_launcher": "ref2va_int8_16gb",
                "service_family": "reference",
                "resolution": "480p",
                "duration_seconds": 15,
            }),
            status="succeeded",
            output_path=self.video,
            final_latents_path=latent,
        )
        service.jobs[source.id] = source
        response = await self.client.post(
            f"/api/v1/jobs/{source.id}/second-sampling",
            json={
                "resolution": "1440p",
                "steps": 1,
                "acceleration": 100,
                "strength": "preserve",
            },
        )
        self.assertEqual(response.status, 202, await response.text())
        child = await response.json()
        self.assertEqual(child["second_sampling"]["resolution"], "1440p")
        self.assertEqual(
            (
                child["second_sampling"]["width"],
                child["second_sampling"]["height"],
            ),
            (2560, 1440),
        )

    async def test_busy_queue_blocks_engine_exit(self) -> None:
        await self.client.put("/api/v1/engine", json={"engine": "original"})
        service = self.app["job_service"]
        spec = GenerationSpec.from_mapping({"prompt": "queued", "seed": 7})
        service.jobs["queued"] = JobRecord(id="queued", spec=spec, status="queued")
        service.pending.append("queued")
        response = await self.client.delete("/api/v1/engine")
        self.assertEqual(response.status, 409)
        self.assertIn("queued", await response.text())

    async def test_workspace_switch_isolates_history_and_storage(self) -> None:
        options = await (await self.client.get("/api/v1/options")).json()
        default_root = self.temporary / "workspace" / "default"
        self.assertEqual(Path(options["workspace"]["current"]["path"]), default_root)
        self.assertTrue(options["workspace"]["switchable"])

        project_a = self.temporary / "creative-project-a"
        response = await self.client.put(
            "/api/v1/workspace", json={"path": str(project_a)}
        )
        self.assertEqual(response.status, 200, await response.text())
        service = self.app["job_service"]
        self.assertEqual(service.data_dir, project_a / ".x-minimax-h3")
        self.assertEqual(service.output_root, project_a / "outputs")
        spec = GenerationSpec.from_mapping({"prompt": "workspace A", "seed": 17})
        service.jobs["workspace-a-job"] = JobRecord(
            id="workspace-a-job", spec=spec, status="failed", error="fixture"
        )
        service.persist(service.jobs["workspace-a-job"])

        project_b = self.temporary / "creative-project-b"
        response = await self.client.put(
            "/api/v1/workspace", json={"path": str(project_b)}
        )
        self.assertEqual(response.status, 200, await response.text())
        self.assertNotIn("workspace-a-job", service.jobs)
        self.assertTrue((project_b / "outputs").is_dir())
        self.assertTrue((project_b / ".x-minimax-h3/checkpoints").is_dir())

        response = await self.client.put(
            "/api/v1/workspace", json={"path": str(project_a)}
        )
        self.assertEqual(response.status, 200, await response.text())
        self.assertIn("workspace-a-job", service.jobs)

    async def test_workspace_cannot_switch_with_loaded_engine(self) -> None:
        await self.client.put("/api/v1/engine", json={"engine": "original"})
        response = await self.client.put(
            "/api/v1/workspace", json={"path": str(self.temporary / "blocked")}
        )
        self.assertEqual(response.status, 409)
        self.assertIn("exit", await response.text())


class TurboApiTest(AioHTTPTestCase):
    async def get_application(self):
        self.temporary = Path(tempfile.mkdtemp(prefix="h3serve-turbo-test-"))
        self.video = self.temporary / "turbo.mp4"
        self.video.write_bytes(b"turbo-video")
        serve_dir = Path(__file__).resolve().parents[1]
        paths = ServicePaths.defaults(self.temporary, data_dir=self.temporary / "data")
        return create_app(
            paths=paths,
            serve_dir=serve_dir,
            backend=FakeBackend(self.video),
            fixed_engine="lora",
        )

    async def asyncTearDown(self) -> None:
        await super().asyncTearDown()
        shutil.rmtree(self.temporary, ignore_errors=True)

    async def test_turbo_process_injects_lora_and_six_step_default(self) -> None:
        response = await self.client.get("/api/v1/options")
        self.assertEqual(response.status, 200)
        options = await response.json()
        self.assertEqual(options["current_engine"], "first_last")
        self.assertEqual(options["current_model_variant"], "lora")
        self.assertEqual(options["defaults"]["quality"], "quality")

        response = await self.client.post(
            "/api/v1/generations", json={"prompt": "Turbo six-step default", "seed": 9}
        )
        self.assertEqual(response.status, 202)
        job = await response.json()
        self.assertEqual(job["request"]["engine"], "lora")
        self.assertEqual(job["request"]["quality"], "quality")


class ReferenceApiTest(AioHTTPTestCase):
    async def get_application(self):
        self.temporary = Path(tempfile.mkdtemp(prefix="h3serve-reference-test-"))
        self.video = self.temporary / "reference.mp4"
        self.video.write_bytes(b"reference-video")
        serve_dir = Path(__file__).resolve().parents[1]
        paths = ServicePaths.defaults(self.temporary, data_dir=self.temporary / "data")
        self.backend = FakeBackend(self.video)
        self.prompt_enhancer = FakePromptEnhancer()
        return create_app(
            paths=paths,
            serve_dir=serve_dir,
            backend=self.backend,
            fixed_engine="reference",
            prompt_enhancer=self.prompt_enhancer,
        )

    async def asyncTearDown(self) -> None:
        await super().asyncTearDown()
        shutil.rmtree(self.temporary, ignore_errors=True)

    async def test_reference_process_requires_and_forwards_reference_images(self) -> None:
        response = await self.client.post(
            "/api/v1/generations", json={"prompt": "missing reference", "seed": 1}
        )
        self.assertEqual(response.status, 400)
        self.assertIn("requires at least one", await response.text())

        form = aiohttp.FormData()
        image_buffer = io.BytesIO()
        Image.new("RGB", (32, 32), (120, 60, 30)).save(image_buffer, format="PNG")
        form.add_field("prompt", "keep the reference identity")
        form.add_field("seed", "2")
        form.add_field("reference_image_resolution", "original")
        form.add_field("reference_video_resolution", "480p")
        form.add_field(
            "reference_image_1", image_buffer.getvalue(),
            filename="identity.png", content_type="image/png",
        )
        response = await self.client.post("/api/v1/generations", data=form)
        self.assertEqual(response.status, 202)
        job = await response.json()
        self.assertEqual(job["request"]["engine"], "reference")
        self.assertEqual(job["request"]["quality"], "balanced")
        self.assertEqual(job["request"]["condition_mode"], "reference")
        self.assertEqual(job["request"]["reference_image_count"], 1)
        self.assertEqual(job["request"]["reference_image_resolution"], "original")
        self.assertEqual(job["request"]["reference_video_resolution"], "480p")

        for _ in range(50):
            state = await (
                await self.client.get(f"/api/v1/jobs/{job['id']}")
            ).json()
            if state["status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(len(self.backend.reference_images), 1)
        self.assertEqual(self.backend.reference_images[0].name, "reference_image_1.png")

    async def test_reference_generation_forwards_one_raw_prompt_without_enhancement(self) -> None:
        prompt = "  subject_definitions:\n<Subject 1> from <Picture 1>.\n  "
        image_buffer = io.BytesIO()
        Image.new("RGB", (32, 32), (10, 20, 30)).save(image_buffer, format="PNG")
        form = aiohttp.FormData()
        form.add_field("prompt", prompt)
        form.add_field(
            "reference_image_1", image_buffer.getvalue(),
            filename="identity.png", content_type="image/png",
        )
        response = await self.client.post("/api/v1/generations", data=form)
        self.assertEqual(response.status, 202)
        job = await response.json()
        for _ in range(50):
            state = await (
                await self.client.get(f"/api/v1/jobs/{job['id']}")
            ).json()
            if state["status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(self.backend.last_spec.prompt, prompt)
        self.assertIsNone(self.prompt_enhancer.request)

    async def test_reference_media_console_policy_is_shared_default(self) -> None:
        response = await self.client.put(
            "/api/v1/settings/reference-media",
            json={"image_resolution": "480p", "video_resolution": "720p"},
        )
        self.assertEqual(response.status, 200)
        policy = await response.json()
        self.assertEqual(policy["image_resolution"], "480p")
        self.assertEqual(policy["video_resolution"], "720p")
        self.assertTrue(policy["preserve_aspect_ratio"])
        self.assertFalse(policy["crop"])
        self.assertFalse(policy["stretch"])
        persisted = json.loads(
            (self.temporary / "data/settings/reference_media.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(persisted, {
            "image_resolution": "480p",
            "video_resolution": "720p",
        })

        options = await (await self.client.get("/api/v1/options")).json()
        self.assertEqual(options["defaults"]["reference_image_resolution"], "480p")
        self.assertEqual(options["defaults"]["reference_video_resolution"], "720p")

        image_buffer = io.BytesIO()
        Image.new("RGB", (48, 32), (30, 90, 120)).save(
            image_buffer, format="PNG"
        )
        form = aiohttp.FormData()
        form.add_field("prompt", "inherit the global reference policy")
        form.add_field(
            "reference_image_1", image_buffer.getvalue(),
            filename="identity.png", content_type="image/png",
        )
        response = await self.client.post("/api/v1/generations", data=form)
        self.assertEqual(response.status, 202)
        job = await response.json()
        self.assertEqual(job["request"]["reference_image_resolution"], "480p")
        self.assertEqual(job["request"]["reference_video_resolution"], "720p")

        bad = await self.client.put(
            "/api/v1/settings/reference-media",
            json={"image_resolution": "1080p"},
        )
        self.assertEqual(bad.status, 400)

    async def test_reference_prompt_enhancement_uses_uppercase_ref2va_contract(self) -> None:
        storyboard = {
            "shots": [{"id": "one", "duration_seconds": 5,
                       "prompt": "女孩参考<Picture 1>走进房间。"}],
            "reference_media": [{
                "kind": "image", "name": "girl.png", "mime_type": "image/png",
                "role": "女孩身份参考",
            }],
        }
        image_buffer = io.BytesIO()
        Image.new("RGB", (16, 16), (80, 40, 120)).save(image_buffer, format="PNG")
        form = aiohttp.FormData()
        form.add_field("storyboard", json.dumps(storyboard, ensure_ascii=False))
        form.add_field(
            "reference_image_1", image_buffer.getvalue(),
            filename="girl.png", content_type="image/png",
        )
        response = await self.client.post(
            "/studio/prompt-enhancements",
            headers={"X-MiMo-API-Key": "mimo-secret"},
            data=form,
        )
        self.assertEqual(response.status, 200, await response.text())
        self.assertEqual(self.prompt_enhancer.request.condition_mode, "REF2VA")

    async def test_reference_video_is_persisted_and_forwarded(self) -> None:
        buffer = io.BytesIO()
        with av.open(buffer, "w", format="mp4") as container:
            stream = container.add_stream("mpeg4", rate=24)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
            for index in range(48):
                pixels = np.full((48, 64, 3), index, dtype=np.uint8)
                for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        form = aiohttp.FormData()
        form.add_field("prompt", "continue <Video 1>")
        form.add_field("seed", "3")
        form.add_field("reference_video_1", buffer.getvalue(), filename="motion.mp4", content_type="video/mp4")
        response = await self.client.post("/api/v1/generations", data=form)
        self.assertEqual(response.status, 202, await response.text())
        self.assertEqual(len(self.backend.reference_videos), 1)
        self.assertEqual(self.backend.reference_videos[0].name, "reference_video_1.mp4")

    async def test_reference_video_is_forwarded_to_mimo_prompt_enhancement(self) -> None:
        buffer = io.BytesIO()
        with av.open(buffer, "w", format="mp4") as container:
            stream = container.add_stream("mpeg4", rate=24)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
            for index in range(48):
                pixels = np.full((48, 64, 3), index, dtype=np.uint8)
                for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        storyboard = {
            "shots": [{"id": "one", "duration_seconds": 2,
                       "prompt": "保持<Video 1>中的骑行运动。"}],
            "reference_media": [{
                "kind": "video", "name": "motion.mp4", "mime_type": "video/mp4",
                "role": "<Video 1> motion and camera reference",
            }],
        }
        form = aiohttp.FormData()
        form.add_field("storyboard", json.dumps(storyboard, ensure_ascii=False))
        form.add_field("reference_video_1", buffer.getvalue(), filename="motion.mp4", content_type="video/mp4")
        response = await self.client.post(
            "/studio/prompt-enhancements",
            headers={"X-MiMo-API-Key": "mimo-secret"}, data=form,
        )
        self.assertEqual(response.status, 200, await response.text())
        self.assertEqual(self.prompt_enhancer.videos[0][0], "<Video 1>")
        self.assertEqual(self.prompt_enhancer.videos[0][1], "video/mp4")

    async def test_reference_audio_is_persisted_and_forwarded(self) -> None:
        buffer = io.BytesIO()
        with av.open(buffer, "w", format="wav") as container:
            stream = container.add_stream("pcm_s16le", rate=48000)
            stream.layout = "mono"
            frame = av.AudioFrame.from_ndarray(
                np.zeros((1, 4800), dtype=np.float32), format="flt", layout="mono"
            )
            frame.sample_rate = 48000
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        form = aiohttp.FormData()
        form.add_field("prompt", "use <Audio 1> as the speaker voice")
        form.add_field("seed", "4")
        form.add_field(
            "reference_audio_1", buffer.getvalue(), filename="voice.wav",
            content_type="audio/wav",
        )
        response = await self.client.post("/api/v1/generations", data=form)
        self.assertEqual(response.status, 202, await response.text())
        job = await response.json()
        self.assertEqual(job["request"]["reference_audio_count"], 1)
        for _ in range(50):
            state = await (await self.client.get(f"/api/v1/jobs/{job['id']}")).json()
            if state["status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(self.backend.reference_audios[0].name, "reference_audio_1.wav")


if __name__ == "__main__":
    unittest.main()
