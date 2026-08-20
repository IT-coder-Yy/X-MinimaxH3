"""Service-facing adapter for the in-process native H3 pipeline."""

from __future__ import annotations

import asyncio
import ctypes
import gc
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..contract import GenerationSpec, actual_step_schedule, engine_family, engine_variant
from .pipeline import GenerationInput, NativeH3Pipeline, PipelineCancelled, SamplingConfig


class NativeGenerationCancelled(RuntimeError):
    """A request was cancelled at a safe native-pipeline boundary."""


@dataclass(frozen=True, slots=True)
class NativeGenerationResult:
    runtime_key: str
    elapsed_seconds: float
    output_path: Path
    stage_seconds: dict[str, float]
    inference_plan: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class NativeCheckpointResult:
    runtime_key: str
    elapsed_seconds: float
    checkpoint_path: Path | None
    preview_path: Path | None
    completed_steps: int
    total_steps: int
    stage_seconds: dict[str, float]
    inference_plan: dict[str, Any] | None = None


_ORIGINAL_ACTUAL_INDICES = {
    "fast": (0, 1, 2, 3, 4, 8, 13, 19),
    "balanced": (0, 1, 2, 3, 4, 8, 12, 16, 19),
    "quality": (0, 1, 2, 3, 4, 6, 8, 11, 14, 17, 18, 19),
    "ultra": tuple(range(20)),
}


def _sampling(spec: GenerationSpec) -> SamplingConfig:
    if spec.engine in ("original", "reference"):
        return SamplingConfig(
            engine="original",
            num_steps=20,
            actual_step_indices=_ORIGINAL_ACTUAL_INDICES[spec.quality],
            sampler="res_multistep",
            scheduler="simple",
        )
    return SamplingConfig(
        engine="lora",
        num_steps=int(spec.preset["steps"]),
        actual_step_indices=None,
        sampler="turbo",
        scheduler="simple",
        lora_strength=float(spec.preset["strength"]),
    )


class NativeH3Engine:
    """Map the stable service contract onto one hot in-process H3 pipeline.

    Model construction is intentionally injected.  It keeps the API/queue
    importable on CPU and lets checkpoint loading fail during readiness rather
    than while importing the web application.
    """

    def __init__(
        self,
        pipeline: NativeH3Pipeline,
        output_root: Path,
        *,
        runtime_revision: str = "native-h3-sm89-v1",
    ) -> None:
        self._pipeline = pipeline
        self._output_root = output_root.resolve()
        self._output_root.mkdir(parents=True, exist_ok=True)
        self.runtime_revision = runtime_revision

    async def generate(
        self,
        spec: GenerationSpec,
        first_frame: Path | None,
        last_frame: Path | None,
        reference_images: tuple[Path, ...],
        reference_videos: tuple[Path, ...],
        reference_audios: tuple[Path, ...],
        cancel_event: asyncio.Event,
        output_path: Path,
        progress_callback: Any | None = None,
        preview_ready_callback: Any | None = None,
        preview_decision_wait: Any | None = None,
        checkpoint_path: Path | None = None,
        resume_checkpoint_path: Path | None = None,
    ) -> NativeGenerationResult:
        output_path = output_path.resolve()
        if not output_path.is_relative_to(self._output_root):
            raise ValueError("output_path must stay inside the configured output root")
        if reference_images or reference_videos or reference_audios:
            raise RuntimeError("the compatibility pipeline does not implement Ref2VA")
        request = GenerationInput(
            prompt=spec.prompt,
            width=spec.width,
            height=spec.height,
            num_frames=spec.frames,
            seed=spec.seed,
            sampling=_sampling(spec),
            fps=24,
            first_frame=first_frame,
            last_frame=last_frame,
            output_path=output_path,
        )
        started = time.monotonic()
        if progress_callback is not None:
            progress_callback({"percent": 5, "stage": "generating", "detail": "开始生成"})
        try:
            state = await asyncio.to_thread(
                self._pipeline.generate,
                request,
                cancel_check=cancel_event.is_set,
            )
        except PipelineCancelled as error:
            raise NativeGenerationCancelled(str(error)) from error

        result_path = self._result_path(state.result, output_path)
        return NativeGenerationResult(
            runtime_key=f"{spec.engine}:{self.runtime_revision}",
            elapsed_seconds=round(time.monotonic() - started, 3),
            output_path=result_path,
            stage_seconds=dict(state.metrics.elapsed_seconds),
        )

    def _result_path(self, result: Any, expected: Path) -> Path:
        if isinstance(result, (str, Path)):
            candidate = Path(result).resolve()
        elif isinstance(result, dict) and result.get("output_path"):
            candidate = Path(result["output_path"]).resolve()
        else:
            candidate = expected.resolve()
        if not candidate.is_relative_to(self._output_root):
            raise RuntimeError("native engine returned a path outside output_root")
        if not candidate.is_file():
            raise RuntimeError(f"native engine did not create the expected video: {candidate.name}")
        return candidate

    async def close(self) -> None:
        await asyncio.to_thread(self._pipeline.close)

    @property
    def output_root(self) -> Path:
        return self._output_root

class NativeHotH3Engine:
    """Own one hot family session with a request-level base/LoRA switch."""

    def __init__(self, factory: Any, output_root: Path) -> None:
        self._factory = factory
        self._output_root = output_root.resolve()
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._built = None
        self._engine_name: str | None = None
        self._warm_state: dict[str, Any] = {
            "status": "cold", "engine": None, "startup_seconds": None, "error": None,
        }

    def preflight(self, engine: str) -> dict[str, Any]:
        return self._factory.preflight(engine)

    def _ensure_session(self, engine: str):
        family = engine if engine in ("first_last", "reference") else engine_family(engine)
        if self._built is not None and self._engine_name == family:
            return self._built
        if self._built is not None:
            self._built.session.close()
            self._built = None
            self._engine_name = None
        try:
            self._built = self._factory.build(family)
            self._engine_name = family
            self._warm_state = {
                "status": "ready",
                "engine": family,
                "startup_seconds": round(self._built.startup_seconds, 3),
                "qwen_storage": getattr(self._built, "qwen_storage", "source"),
                "qwen_layer_cache": bool(
                    getattr(self._built, "qwen_layer_cache", False)
                ),
                "v19_release_bundle": getattr(
                    self._built, "v19_release_bundle", None
                ),
                "v19_release_digest": getattr(
                    self._built, "v19_release_digest", None
                ),
                "error": None,
            }
        except Exception as error:
            self._warm_state = {
                "status": "failed", "engine": family,
                "startup_seconds": None, "error": str(error),
            }
            raise
        return self._built

    async def preload(self, engine: str) -> None:
        engine = engine if engine in ("first_last", "reference") else engine_family(engine)
        async with self._lock:
            if self._built is not None and self._engine_name == engine:
                return
            self._warm_state = {
                "status": "loading", "engine": engine,
                "startup_seconds": None, "error": None,
            }
            try:
                await asyncio.to_thread(self._ensure_session, engine)
            except Exception:
                # Readiness exposes the failure. Keep the Web/API process alive
                # so an operator can inspect it or repair files in place.
                return

    @property
    def warm_state(self) -> dict[str, Any]:
        # Health is intentionally public. Do not leak checkpoint paths or
        # loader exception details through it.
        return {
            key: self._warm_state.get(key)
            for key in (
                "status", "engine", "startup_seconds", "qwen_storage",
                "qwen_layer_cache",
            )
        }

    @staticmethod
    def _request_plan(spec: GenerationSpec):
        # Six-step Larry and original 9/11 are the fully calibrated routes.
        # Other exposed presets use the same lossless mechanical plan while
        # retaining their user-selected model behavior; they are deliberately
        # excluded from latency-based routing instead of failing the request.
        approximate_attention = (
            spec.joint_acceleration_enabled and float(spec.acceleration or 0.0) > 0.0
        ) or (spec.advanced and spec.attention_keep_ratio < 1.0)
        calibrated = not spec.joint_acceleration_enabled and not approximate_attention and ((
            spec.engine in ("lora", "reference_lora") and int(spec.preset["steps"]) == 6
        ) or (
            spec.engine in ("original", "reference")
            and int(spec.preset["actual_steps"]) == 9
        ))
        if calibrated:
            return None
        from .planner import ExecutionPlan
        from .runtime import OffloadMode

        plan = ExecutionPlan(
            offload_mode=OffloadMode.BLOCK,
            mlp_chunk_tokens=8192,
            block_buffer_count=2,
            prefetch_depth=1,
            vae_spatial_tile=(288, 288),
        )
        if approximate_attention and not spec.joint_acceleration_enabled:
            plan = replace(
                plan,
                attention_topk=spec.attention_keep_ratio,
                sparse_scope=spec.sparse_scope,
            )
        return plan

    async def generate(
        self,
        spec: GenerationSpec,
        first_frame: Path | None,
        last_frame: Path | None,
        reference_images: tuple[Path, ...],
        reference_videos: tuple[Path, ...],
        reference_audios: tuple[Path, ...],
        cancel_event: asyncio.Event,
        output_path: Path,
        progress_callback: Any | None = None,
        preview_ready_callback: Any | None = None,
        preview_decision_wait: Any | None = None,
        checkpoint_path: Path | None = None,
        resume_checkpoint_path: Path | None = None,
    ) -> NativeGenerationResult | NativeCheckpointResult:
        from .hot_session import (
            HotSessionCancelled,
            HotSessionCheckpointResult,
            HotSessionRequest,
        )
        from .long_video_motion_detail import select_candidate

        joint_plan = None
        # INT8 V19 and distilled LoRA deliberately have separate scheduling
        # domains.  A Base release bundle must never make LoRA borrow a
        # forecast trajectory that was calibrated for the 20-step model.
        use_v19 = bool(
            spec.model_variant == "base"
            and
            spec.joint_acceleration_enabled
            and getattr(self._factory, "v19_release_enabled", False)
        )
        joint_scheduler_id = None
        if spec.joint_acceleration_enabled and not use_v19:
            from .planner import (
                FROZEN_INT8_JOINT_POLICY,
                H3JointAccelerationScheduler,
                H3LoraAccelerationScheduler,
                H3WorkloadAnalyzer,
                JointWorkloadContext,
                LORA_NO_FORECAST_SCHEDULER_ID,
            )

            latent_frames = H3WorkloadAnalyzer.video_latent_frames(spec.frames)
            spatial_tokens = (spec.height // 32) * (spec.width // 32)
            visual_condition_count = (
                int(first_frame is not None)
                + int(last_frame is not None)
                + len(reference_images)
                + len(reference_videos)
            )
            # Exact Qwen tokenisation happens later in the hot session.  Text
            # is <2% of the calibrated packed sequence, so this bounded
            # estimate selects/interpolates the correct shape cost model
            # without performing text encoding twice.
            estimated_text_tokens = max(
                128, min(1024, int(math.ceil(len(spec.prompt) * 0.55)))
            )
            audio_tokens = 2 * round((spec.frames / 24.0) * 40.0)
            workload = JointWorkloadContext(
                packed_tokens=(
                    latent_frames * spatial_tokens
                    + visual_condition_count * spatial_tokens
                    + audio_tokens
                    + estimated_text_tokens
                ),
                condition_count=visual_condition_count,
                service_family=spec.service_family,
                model_variant=spec.model_variant,
            )

            # Research runs can pin a version without changing the two-field
            # public request contract or silently moving the release default.
            # The selected id remains serialized in request telemetry.
            if spec.model_variant == "lora":
                policy_id = (
                    os.environ.get(
                        "H3_NATIVE_RESEARCH_LORA_JOINT_POLICY", ""
                    ).strip()
                    or FROZEN_INT8_JOINT_POLICY
                )
                scheduler = H3LoraAccelerationScheduler(policy_id=policy_id)
                joint_plan = scheduler.plan(
                    int(spec.sampling_steps or 8),
                    float(spec.acceleration or 0.0),
                    workload=workload,
                )
                joint_scheduler_id = LORA_NO_FORECAST_SCHEDULER_ID
            else:
                policy_id = (
                    os.environ.get(
                        "H3_NATIVE_RESEARCH_JOINT_POLICY", ""
                    ).strip()
                    or FROZEN_INT8_JOINT_POLICY
                )
                joint_plan = H3JointAccelerationScheduler(
                    policy_id=policy_id
                ).plan(
                    int(spec.sampling_steps or 20),
                    float(spec.acceleration or 0.0),
                    allow_forecast=True,
                    workload=workload,
                )
                joint_scheduler_id = "h3_int8_frozen_round229"
        sparse_requested = (
            use_v19 and float(spec.acceleration or 0.0) > 0.0
        ) or (
            joint_plan is not None and joint_plan.uses_sparse_attention
        ) or (
            spec.advanced
            and not spec.joint_acceleration_enabled
            and spec.attention_keep_ratio < 1.0
        )
        if sparse_requested and not self._factory.sparse_attention_available:
            raise RuntimeError(
                "sparse attention is not installed for this service; "
                "use acceleration=0 or install the optional SM89 runtime"
            )

        output_path = output_path.resolve()
        if not output_path.is_relative_to(self._output_root):
            raise ValueError("output_path must stay inside the configured output root")
        async with self._lock:
            if cancel_event.is_set():
                raise NativeGenerationCancelled("native H3 generation cancelled")
            built = await asyncio.to_thread(self._ensure_session, spec.service_family)
            if use_v19:
                total_steps = int(spec.sampling_steps or 20)
                actual_steps = tuple(range(total_steps))
            elif joint_plan is not None:
                actual_steps = joint_plan.actual_step_indices
                total_steps = joint_plan.total_steps
            else:
                actual_steps = (
                    actual_step_schedule(int(spec.preset["actual_steps"]))
                    if spec.engine in ("original", "reference")
                    else tuple(range(int(spec.preset["steps"])))
                )
                total_steps = (
                    20
                    if spec.engine in ("original", "reference")
                    else int(spec.preset["steps"])
                )
            candidate = (
                None
                if joint_plan is not None or use_v19
                else select_candidate(
                    spec,
                    first_frame=first_frame,
                    last_frame=last_frame,
                    reference_images=reference_images,
                    reference_videos=reference_videos,
                    reference_audios=reference_audios,
                )
            )
            execution_plan = self._request_plan(spec)
            if joint_plan is not None or use_v19:
                if execution_plan is None:
                    raise RuntimeError(
                        "joint acceleration requires an explicit RTX 4090 execution plan"
                    )
                # The scheduler is allowed to trade only sampler/Attention
                # compute.  Keep the mature Round86/143 mechanical baseline
                # underneath every versioned joint policy so a new control-plane
                # version cannot accidentally benchmark against slower,
                # unfused runtime defaults.
                execution_plan = replace(
                    execution_plan,
                    fused_rms_adaln=True,
                    vae_transformer_block_compile=True,
                )
            if candidate is not None:
                if not self._factory.sparse_attention_available:
                    raise RuntimeError(
                        "the reviewed long-video route requires the pinned "
                        "SM89 sparse-attention runtime"
                    )
                if execution_plan is None:
                    raise RuntimeError(
                        "the reviewed long-video route has no explicit "
                        "RTX 4090 execution plan"
                    )
                execution_plan = replace(
                    execution_plan,
                    fused_rms_adaln=True,
                    dense_qk_quant_gran="per_warp",
                    vae_transformer_block_compile=True,
                    long_video_motion_detail_attention=True,
                )
            preview_step = None
            preview_output = None
            checkpoint_after_step = None
            if spec.execution_mode == "checkpoint" and resume_checkpoint_path is None:
                checkpoint_after_step = spec.checkpoint_step
                if checkpoint_path is None:
                    raise RuntimeError("checkpoint task has no persistence path")
                if spec.checkpoint_preview:
                    preview_step = int(spec.checkpoint_step or 1) - 1
                    preview_output = output_path.with_name(
                        output_path.stem + ".checkpoint-preview.mp4"
                    )
            elif spec.preview_mode != "off":
                if spec.preview_step_index is not None:
                    preview_step = spec.preview_step_index
                elif spec.model_variant == "lora":
                    preview_step = max(1, min(int(spec.preset["steps"]) - 2, int(spec.preset["steps"]) // 2))
                elif use_v19:
                    # V19 owns the actual/forecast schedule only after exact
                    # tokenisation.  Use the standard three-quarter protected
                    # anchor and let the certified selector require it as an
                    # actual evaluation; if no admitted trajectory contains
                    # it, routing fails closed to Dense rather than decoding a
                    # low-quality forecast x0 estimate.
                    preview_step = max(
                        1,
                        min(total_steps - 2, round(total_steps * 0.75)),
                    )
                else:
                    # Select a real-compute anchor around two thirds of the
                    # calibrated actual evaluations, never a forecast point.
                    preview_step = actual_steps[min(len(actual_steps) - 2, (2 * len(actual_steps)) // 3)]
                preview_output = output_path.with_name(output_path.stem + ".preview.mp4")
            preview_scale = 1.0
            fast_preview = (
                (spec.execution_mode == "checkpoint" and spec.checkpoint_preview)
                or (spec.preview_mode != "off" and spec.preview_fast_finish)
            )
            if (
                fast_preview
                and spec.checkpoint_preview_resolution != "source"
            ):
                preview_scale = min(
                    1.0,
                    int(spec.checkpoint_preview_resolution[:-1])
                    / float(min(spec.width, spec.height)),
                )
            request = HotSessionRequest(
                prompt=spec.prompt,
                seed=spec.seed,
                width=spec.width,
                height=spec.height,
                frames=spec.frames,
                fps=24,
                steps=total_steps,
                output_path=output_path,
                actual_step_indices=actual_steps,
                execution_plan=execution_plan,
                attention_action_schedule=(
                    ()
                    if joint_plan is None
                    else tuple(
                        (step, layer, action)
                        for (step, layer), action in sorted(
                            joint_plan.runtime_action_schedule().items()
                        )
                    )
                ),
                attention_online_guard_id=(
                    None if joint_plan is None else joint_plan.online_guard_id
                ),
                attention_online_budget_dense_layers=(
                    0.0
                    if joint_plan is None or joint_plan.online_guard_id is None
                    else joint_plan.online_recovery_reserve_units * 50.0
                ),
                attention_online_rebate_schedule=(
                    () if joint_plan is None else joint_plan.online_rebate_schedule
                ),
                acceleration_plan_summary=(
                    None
                    if joint_plan is None
                    else {
                        **{
                            key: value
                            for key, value in joint_plan.to_dict().items()
                            if key != "attention_decisions"
                        },
                        "scheduler_family": joint_scheduler_id,
                        "model_variant": spec.model_variant,
                    }
                ),
                v19_acceleration=(
                    float(spec.acceleration or 0.0) if use_v19 else None
                ),
                first_frame=first_frame,
                last_frame=last_frame,
                reference_images=reference_images,
                reference_videos=reference_videos,
                reference_audios=reference_audios,
                reference_image_resolution=spec.reference_image_resolution,
                reference_video_resolution=spec.reference_video_resolution,
                cancel_check=cancel_event.is_set,
                progress_callback=progress_callback,
                use_lora=engine_variant(spec.engine) == "lora",
                formal_resume_state_path=resume_checkpoint_path,
                checkpoint_after_step=checkpoint_after_step,
                checkpoint_state_path=(
                    checkpoint_path if checkpoint_after_step is not None else None
                ),
                preview_step_index=preview_step,
                preview_output_path=preview_output,
                preview_decode_mode=(
                    "fast_finish"
                    if fast_preview
                    else "direct_x0"
                ),
                preview_branch_steps=(
                    spec.checkpoint_preview_steps
                    if fast_preview
                    else spec.preview_branch_steps
                ),
                # The ordinary API preview keeps its historical direct-x0
                # path.  Explicit fast-finish clients (ComfyUI/checkpoints)
                # may instead request a disposable LoRA branch and a smaller
                # preview canvas without mutating the formal trajectory.
                preview_branch_spatial_scale=preview_scale,
                preview_branch_force_dense=True,
                preview_branch_use_lora=(
                    fast_preview
                ),
                preview_audio_branch_use_lora=False,
                preview_audio_branch_steps=4,
                preview_audio_branch_spatial_scale=0.65,
                preview_ready_callback=preview_ready_callback,
                preview_decision_wait=(
                    preview_decision_wait if spec.preview_mode == "pause" else None
                ),
                terminal_refinement_initial_width=(
                    candidate.initial_width if candidate is not None else None
                ),
                terminal_refinement_initial_height=(
                    candidate.initial_height if candidate is not None else None
                ),
                terminal_refinement_steps=(
                    candidate.refinement_steps if candidate is not None else 0
                ),
                terminal_refinement_denoise=(
                    candidate.refinement_denoise if candidate is not None else 0.0125
                ),
                terminal_refinement_dense_tail_steps=(
                    candidate.dense_tail_steps if candidate is not None else 1
                ),
            )
            started = time.monotonic()
            try:
                result = await asyncio.to_thread(built.session.generate, request)
            except HotSessionCancelled as error:
                raise NativeGenerationCancelled(str(error)) from error
            if isinstance(result, HotSessionCheckpointResult):
                return NativeCheckpointResult(
                    runtime_key=f"{spec.engine}:native-sm89",
                    elapsed_seconds=round(time.monotonic() - started, 3),
                    checkpoint_path=result.checkpoint_path,
                    preview_path=result.preview_path,
                    completed_steps=result.completed_steps,
                    total_steps=result.total_steps,
                    stage_seconds=dict(result.phases),
                    inference_plan=(
                        getattr(result, "execution_profile", None) or {}
                    ).get("joint_acceleration"),
                )
            phases = dict(result.phases)
            if candidate is not None:
                from .detail_restore import (
                    DetailRestoreCancelled,
                    restore_intrame_detail,
                )

                if progress_callback is not None:
                    progress_callback({
                        "percent": 98,
                        "stage": "detail_restore",
                        "detail": "恢复画面细节",
                    })
                try:
                    restored = await asyncio.to_thread(
                        restore_intrame_detail,
                        result.output_path,
                        expected_width=spec.width,
                        expected_height=spec.height,
                        expected_frames=spec.frames,
                        cancel_check=cancel_event.is_set,
                        preserve_raw=True,
                        parallel_shards=4,
                        fps=24,
                    )
                except DetailRestoreCancelled as error:
                    raise NativeGenerationCancelled(str(error)) from error
                phases["intrame_detail_restore"] = restored.elapsed_seconds
            return NativeGenerationResult(
                runtime_key=f"{spec.engine}:native-sm89",
                elapsed_seconds=round(time.monotonic() - started, 3),
                output_path=result.output_path,
                stage_seconds=phases,
                inference_plan=(
                    getattr(result, "execution_profile", None) or {}
                ).get("joint_acceleration"),
            )

    async def close(self) -> None:
        async with self._lock:
            if self._built is not None:
                await asyncio.to_thread(self._built.session.close)
            self._built = None
            self._engine_name = None
            def release_host_session_pages() -> None:
                gc.collect()
                try:
                    ctypes.CDLL(None).malloc_trim(0)
                except (AttributeError, OSError):
                    pass
            await asyncio.to_thread(release_host_session_pages)
            self._warm_state = {
                "status": "cold", "engine": None,
                "startup_seconds": None, "error": None,
            }

    @property
    def output_root(self) -> Path:
        return self._output_root

    def set_output_root(self, output_root: Path) -> None:
        if self._built is not None:
            raise RuntimeError("cannot switch workspace while an engine is loaded")
        resolved = output_root.resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        self._factory.set_output_root(resolved)
        self._output_root = resolved
