from __future__ import annotations

import asyncio
import shutil
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from h3serve.backend import NativeBackendManager
from h3serve.contract import GenerationSpec
from h3serve.native_engine import NativeH3Engine, NativeHotH3Engine


class FakePipeline:
    def __init__(self) -> None:
        self.last_request = None
        self.closed = False

    def generate(self, request, *, cancel_check):
        if cancel_check():
            raise AssertionError("test request was unexpectedly cancelled")
        self.last_request = request
        request.output_path.write_bytes(b"native-video")
        return SimpleNamespace(
            result=request.output_path,
            metrics=SimpleNamespace(elapsed_seconds={"denoise": 0.5}),
        )

    def close(self) -> None:
        self.closed = True


class FakeHotSession:
    def __init__(self) -> None:
        self.requests = []
        self.closed = False

    def generate(self, request):
        self.requests.append(request)
        request.output_path.write_bytes(b"hot-native-video")
        return SimpleNamespace(
            output_path=request.output_path,
            phases={"denoise": 0.1},
        )

    def close(self):
        self.closed = True


class FakeCheckpointHotSession(FakeHotSession):
    def generate(self, request):
        from h3serve.native_engine.hot_session import HotSessionCheckpointResult

        self.requests.append(request)
        if request.checkpoint_after_step is not None:
            checkpoint_path = Path(request.checkpoint_state_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"formal-lora-checkpoint")
            preview_path = request.preview_output_path
            if preview_path is not None:
                preview_path.write_bytes(b"lora-preview")
            return HotSessionCheckpointResult(
                checkpoint_path=checkpoint_path,
                preview_path=preview_path,
                completed_steps=request.checkpoint_after_step,
                total_steps=request.steps,
                total_seconds=0.1,
                phases={"denoise": 0.1},
                step_seconds=(0.03,) * request.checkpoint_after_step,
                execution_profile={
                    "joint_acceleration": request.acceleration_plan_summary,
                    "formal_checkpoint": {
                        "completed_steps": request.checkpoint_after_step,
                        "formal_trajectory_mutated": False,
                    },
                },
            )
        request.output_path.write_bytes(b"resumed-lora-video")
        return SimpleNamespace(
            output_path=request.output_path,
            phases={"denoise": 0.1},
            execution_profile={
                "joint_acceleration": request.acceleration_plan_summary,
                "formal_resume": {
                    "formal_prefix_replayed": False,
                },
            },
        )


class FakeHotFactory:
    sparse_attention_available = False

    def __init__(self) -> None:
        self.builds = []
        self.sessions = []

    def build(self, family):
        self.builds.append(family)
        session = FakeHotSession()
        self.sessions.append(session)
        return SimpleNamespace(
            session=session, startup_seconds=0.1, qwen_storage="source"
        )

    def preflight(self, _family):
        return {"ready": True, "checks": {"fake": True}}


class FakeSparseHotFactory(FakeHotFactory):
    sparse_attention_available = True


class FakeV19HotFactory(FakeSparseHotFactory):
    v19_release_enabled = True


class FakeCheckpointHotFactory(FakeSparseHotFactory):
    def build(self, family):
        self.builds.append(family)
        session = FakeCheckpointHotSession()
        self.sessions.append(session)
        return SimpleNamespace(
            session=session, startup_seconds=0.1, qwen_storage="source"
        )


class NativeEngineBoundaryTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = Path(tempfile.mkdtemp(prefix="native-engine-test-"))
        self.pipeline = FakePipeline()
        self.engine = NativeH3Engine(self.pipeline, self.temporary)
        self.manager = NativeBackendManager(self.engine)

    async def asyncTearDown(self) -> None:
        await self.manager.stop()
        shutil.rmtree(self.temporary, ignore_errors=True)

    async def test_original_quality_maps_to_explicit_schedule(self) -> None:
        spec = GenerationSpec.from_mapping({
            "prompt": "fixed native contract",
            "engine": "original",
            "quality": "balanced",
            "duration_seconds": 5,
            "seed": 4404,
        })
        result = await self.manager.generate(
            spec, "job-1", None, None, (), (), (), asyncio.Event()
        )
        self.assertEqual(result.output_path.read_bytes(), b"native-video")
        self.assertEqual(
            self.pipeline.last_request.sampling.actual_step_indices,
            (0, 1, 2, 3, 4, 8, 12, 16, 19),
        )
        self.assertEqual(self.pipeline.last_request.sampling.engine, "original")

    async def test_lora_route_preserves_distilled_step_count(self) -> None:
        spec = GenerationSpec.from_mapping({
            "prompt": "fixed native turbo contract",
            "engine": "lora",
            "quality": "quality",
            "duration_seconds": 5,
            "seed": 8833,
        })
        await self.manager.generate(spec, "job-2", None, None, (), (), (), asyncio.Event())
        sampling = self.pipeline.last_request.sampling
        self.assertEqual(sampling.engine, "lora")
        self.assertEqual(sampling.num_steps, 6)
        self.assertEqual(sampling.sampler, "turbo")
        self.assertEqual(sampling.lora_strength, 1.0)

    def test_hot_engine_only_latency_routes_calibrated_step_presets(self) -> None:
        original_balanced = GenerationSpec.from_mapping({
            "prompt": "balanced",
            "engine": "original",
            "quality": "balanced",
            "seed": 1,
        })
        original_quality = GenerationSpec.from_mapping({
            "prompt": "quality",
            "engine": "original",
            "quality": "quality",
            "seed": 2,
        })
        lora_quality = GenerationSpec.from_mapping({
            "prompt": "lora six",
            "engine": "lora",
            "quality": "quality",
            "seed": 3,
        })
        lora_fast = GenerationSpec.from_mapping({
            "prompt": "lora four",
            "engine": "lora",
            "quality": "fast",
            "seed": 4,
        })
        original_advanced_balanced = GenerationSpec.from_mapping({
            "prompt": "advanced balanced", "engine": "original", "advanced": True,
            "width": 864, "height": 480, "frames": 124,
            "actual_steps": 9, "seed": 5,
        })
        lora_advanced_six = GenerationSpec.from_mapping({
            "prompt": "advanced six", "engine": "lora", "advanced": True,
            "width": 864, "height": 480, "frames": 124,
            "lora_steps": 6, "seed": 6,
        })
        self.assertIsNone(NativeHotH3Engine._request_plan(original_balanced))
        self.assertIsNotNone(NativeHotH3Engine._request_plan(original_quality))
        self.assertIsNone(NativeHotH3Engine._request_plan(lora_quality))
        self.assertIsNotNone(NativeHotH3Engine._request_plan(lora_fast))
        self.assertIsNone(NativeHotH3Engine._request_plan(original_advanced_balanced))
        self.assertIsNone(NativeHotH3Engine._request_plan(lora_advanced_six))

    async def test_hot_engine_reuses_family_session_across_base_lora_base(self) -> None:
        factory = FakeHotFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        try:
            specs = [
                GenerationSpec.from_mapping({
                    "prompt": "base one", "service_family": "first_last",
                    "model_variant": "base", "seed": 1,
                }),
                GenerationSpec.from_mapping({
                    "prompt": "lora", "service_family": "first_last",
                    "model_variant": "lora", "quality": "quality", "seed": 2,
                }),
                GenerationSpec.from_mapping({
                    "prompt": "base two", "service_family": "first_last",
                    "model_variant": "base", "seed": 3,
                }),
            ]
            for index, spec in enumerate(specs):
                await engine.generate(
                    spec, None, None, (), (), (), asyncio.Event(),
                    self.temporary / f"hot-{index}.mp4",
                )
            self.assertEqual(factory.builds, ["first_last"])
            self.assertEqual(
                [request.use_lora for request in factory.sessions[0].requests],
                [False, True, False],
            )
        finally:
            await engine.close()

    async def test_hot_engine_applies_two_control_joint_plan_without_rebuild(self) -> None:
        factory = FakeSparseHotFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        try:
            spec = GenerationSpec.from_mapping({
                "prompt": "joint schedule",
                "engine": "original",
                "mode": "advanced",
                "width": 1280,
                "height": 736,
                "duration_seconds": 15,
                "sampling_steps": 20,
                "acceleration": 100,
                "seed": 82303,
            })
            await engine.generate(
                spec, None, None, (), (), (), asyncio.Event(),
                self.temporary / "joint.mp4",
            )
            request = factory.sessions[0].requests[-1]
            self.assertEqual(request.steps, 20)
            self.assertEqual(
                request.actual_step_indices,
                (0, 1, 2, 3, 4, 8, 12, 15, 18, 19),
            )
            self.assertEqual(len(request.attention_action_schedule), 10 * 50 + 10 * 3)
            self.assertEqual(
                request.acceleration_plan_summary["acceleration"], 100.0
            )
            self.assertEqual(
                request.acceleration_plan_summary["scheduler_family"],
                "h3_int8_frozen_round229",
            )
            self.assertTrue(request.execution_plan.fused_rms_adaln)
            self.assertTrue(request.execution_plan.vae_transformer_block_compile)
            self.assertEqual(factory.builds, ["first_last"])
        finally:
            await engine.close()

    async def test_hot_engine_routes_lora_to_no_forecast_sparse_scheduler(self) -> None:
        # Even when a Base-only V19 bundle is installed, LoRA remains in its
        # own no-forecast scheduling domain.
        factory = FakeV19HotFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        try:
            spec = GenerationSpec.from_mapping({
                "prompt": "LoRA scheduled trajectory",
                "service_family": "first_last",
                "model_variant": "lora",
                "mode": "advanced",
                "width": 864,
                "height": 480,
                "duration_seconds": 5,
                "sampling_steps": 8,
                "acceleration": 50,
                "seed": 82416,
            })
            await engine.generate(
                spec, None, None, (), (), (), asyncio.Event(),
                self.temporary / "lora-joint.mp4",
            )
            request = factory.sessions[0].requests[-1]
            self.assertTrue(request.use_lora)
            self.assertEqual(request.steps, 8)
            self.assertEqual(request.actual_step_indices, tuple(range(8)))
            self.assertEqual(len(request.attention_action_schedule), 8 * 50)
            self.assertIsNone(request.v19_acceleration)
            self.assertEqual(
                request.acceleration_plan_summary["forecast_evaluations"], 0
            )
            self.assertFalse(
                request.acceleration_plan_summary["forecast_allowed"]
            )
            self.assertEqual(
                request.acceleration_plan_summary["scheduler_family"],
                "h3_lora_v1_no_forecast_round229",
            )
            self.assertEqual(
                request.acceleration_plan_summary["model_variant"], "lora"
            )
        finally:
            await engine.close()

    async def test_lora_joint_checkpoint_preview_and_resume_keep_one_schedule(self) -> None:
        factory = FakeCheckpointHotFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        checkpoint_path = self.temporary / "checkpoints" / "lora.pt"
        output_path = self.temporary / "lora-checkpoint.mp4"
        spec = GenerationSpec.from_mapping({
            "prompt": "LoRA resumable scheduled trajectory",
            "service_family": "reference",
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
            "checkpoint_preview": True,
            "checkpoint_preview_steps": 4,
            "seed": 82416,
        })
        try:
            stopped = await engine.generate(
                spec, None, None, (), (), (), asyncio.Event(), output_path,
                checkpoint_path=checkpoint_path,
            )
            self.assertEqual(stopped.completed_steps, 3)
            self.assertEqual(stopped.total_steps, 8)
            self.assertEqual(stopped.checkpoint_path, checkpoint_path)
            first = factory.sessions[0].requests[-1]
            self.assertEqual(first.actual_step_indices, tuple(range(8)))
            self.assertEqual(len(first.attention_action_schedule), 8 * 50)
            self.assertEqual(first.checkpoint_after_step, 3)
            self.assertTrue(first.preview_branch_use_lora)
            self.assertTrue(first.preview_branch_force_dense)
            self.assertEqual(first.preview_branch_steps, 4)

            resumed = await engine.generate(
                spec, None, None, (), (), (), asyncio.Event(), output_path,
                resume_checkpoint_path=checkpoint_path,
            )
            second = factory.sessions[0].requests[-1]
            self.assertEqual(second.formal_resume_state_path, checkpoint_path)
            self.assertIsNone(second.checkpoint_after_step)
            self.assertEqual(
                second.attention_action_schedule,
                first.attention_action_schedule,
            )
            self.assertEqual(second.actual_step_indices, first.actual_step_indices)
            self.assertEqual(
                second.acceleration_plan_summary["scheduler_family"],
                "h3_lora_v1_no_forecast_round229",
            )
            self.assertEqual(resumed.output_path.read_bytes(), b"resumed-lora-video")
        finally:
            await engine.close()

    async def test_hot_engine_defers_v19_routing_until_exact_tokenisation(self) -> None:
        factory = FakeV19HotFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        try:
            spec = GenerationSpec.from_mapping({
                "prompt": "V19 exact token routing",
                "engine": "original",
                "mode": "advanced",
                "width": 1280,
                "height": 736,
                "duration_seconds": 15,
                "sampling_steps": 20,
                "acceleration": 100,
                "seed": 82303,
            })
            await engine.generate(
                spec, None, None, (), (), (), asyncio.Event(),
                self.temporary / "v19.mp4",
            )
            request = factory.sessions[0].requests[-1]
            self.assertEqual(request.actual_step_indices, tuple(range(20)))
            self.assertEqual(request.attention_action_schedule, ())
            self.assertIsNone(request.acceleration_plan_summary)
            self.assertEqual(request.v19_acceleration, 100.0)
            self.assertTrue(request.execution_plan.fused_rms_adaln)
            self.assertTrue(request.execution_plan.vae_transformer_block_compile)
        finally:
            await engine.close()

    async def test_v19_auto_preview_requests_the_standard_actual_anchor(self) -> None:
        factory = FakeV19HotFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        try:
            spec = GenerationSpec.from_mapping({
                "prompt": "V19 preview anchor",
                "engine": "original",
                "mode": "advanced",
                "width": 1280,
                "height": 736,
                "duration_seconds": 15,
                "sampling_steps": 20,
                "acceleration": 100,
                "preview_mode": "auto",
                "preview_step_index": 5,
                "preview_fast_finish": True,
                "checkpoint_preview_steps": 4,
                "checkpoint_preview_resolution": "360p",
                "seed": 82303,
            })
            await engine.generate(
                spec, None, None, (), (), (), asyncio.Event(),
                self.temporary / "v19-preview.mp4",
            )
            request = factory.sessions[0].requests[-1]
            self.assertEqual(request.preview_step_index, 5)
            self.assertIsNotNone(request.preview_output_path)
            self.assertEqual(request.preview_decode_mode, "fast_finish")
            self.assertTrue(request.preview_branch_use_lora)
            self.assertEqual(request.preview_branch_steps, 4)
            self.assertAlmostEqual(request.preview_branch_spatial_scale, 360 / 736)
        finally:
            await engine.close()

    async def test_long_video_candidate_is_exactly_routed_and_does_not_leak(self) -> None:
        factory = FakeSparseHotFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        try:
            with (
                patch.dict(
                    "os.environ", {"H3_NATIVE_LONG_VIDEO_REVIEW": "1"}
                ),
                patch(
                    "h3serve.native_engine.detail_restore.restore_intrame_detail",
                    return_value=SimpleNamespace(elapsed_seconds=8.78),
                ) as detail_restore,
            ):
                eligible = GenerationSpec.from_mapping({
                    "prompt": "eligible long video",
                    "engine": "original",
                    "quality": "quality",
                    "resolution": "720p",
                    "aspect_ratio": "16:9",
                    "duration_seconds": 15,
                    "seed": 82303,
                })
                eligible_result = await engine.generate(
                    eligible, None, None, (), (), (), asyncio.Event(),
                    self.temporary / "eligible.mp4",
                )
                selected = factory.sessions[-1].requests[-1]
                self.assertEqual(
                    (
                        selected.terminal_refinement_initial_width,
                        selected.terminal_refinement_initial_height,
                        selected.terminal_refinement_steps,
                        selected.terminal_refinement_denoise,
                        selected.terminal_refinement_dense_tail_steps,
                    ),
                    (864, 480, 2, 0.025, 1),
                )
                self.assertTrue(
                    selected.execution_plan.long_video_motion_detail_attention
                )
                self.assertTrue(selected.execution_plan.fused_rms_adaln)
                self.assertTrue(
                    selected.execution_plan.vae_transformer_block_compile
                )
                self.assertEqual(
                    selected.execution_plan.dense_qk_quant_gran, "per_warp"
                )
                detail_restore.assert_called_once()
                self.assertEqual(
                    eligible_result.stage_seconds["intrame_detail_restore"], 8.78
                )

                short = GenerationSpec.from_mapping({
                    "prompt": "short request",
                    "engine": "original",
                    "quality": "quality",
                    "resolution": "720p",
                    "aspect_ratio": "16:9",
                    "duration_seconds": 5,
                    "seed": 82304,
                })
                await engine.generate(
                    short, None, None, (), (), (), asyncio.Event(),
                    self.temporary / "short.mp4",
                )
                excluded = factory.sessions[-1].requests[-1]
                self.assertIsNone(excluded.terminal_refinement_initial_width)
                self.assertEqual(excluded.terminal_refinement_steps, 0)
                self.assertFalse(
                    excluded.execution_plan.long_video_motion_detail_attention
                )
        finally:
            await engine.close()


if __name__ == "__main__":
    unittest.main()
