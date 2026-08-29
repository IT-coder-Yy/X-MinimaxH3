"""Service-facing adapter for the in-process native H3 pipeline."""

from __future__ import annotations

import asyncio
import ctypes
import gc
import math
import os
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..contract import (
    GenerationSpec,
    SecondSamplingSpec,
    actual_step_schedule,
    engine_variant,
    launcher_family,
    launcher_vram_profile,
    launcher_weight_tier,
    normalize_launcher,
)
from .pipeline import GenerationInput, NativeH3Pipeline, PipelineCancelled, SamplingConfig
from .adapters.sampling_mux import refinement_sigma_schedule


class NativeGenerationCancelled(RuntimeError):
    """A request was cancelled at a safe native-pipeline boundary."""


@dataclass(frozen=True, slots=True)
class NativeGenerationResult:
    runtime_key: str
    elapsed_seconds: float
    output_path: Path
    stage_seconds: dict[str, float]
    inference_plan: dict[str, Any] | None = None
    final_latents_path: Path | None = None


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


def _public_inference_plan(execution_profile: object) -> dict[str, Any] | None:
    """Keep quality scheduling and physical memory routing auditable.

    Historically the service returned only ``joint_acceleration`` from the
    much larger native execution profile.  That made an ``auto`` memory-mode
    request impossible to inspect after completion.  Preserve the existing
    flat scheduler fields for client compatibility and attach only the small,
    stable physical-route receipt needed by the public product contract.
    """

    if not isinstance(execution_profile, dict):
        return None
    joint = execution_profile.get("joint_acceleration")
    result = dict(joint) if isinstance(joint, dict) else {}
    memory_execution = execution_profile.get("memory_execution")
    if isinstance(memory_execution, dict):
        result["memory_execution"] = dict(memory_execution)
    conditioning_cache = execution_profile.get("qwen_conditioning_cache")
    if isinstance(conditioning_cache, dict):
        # This is a small receipt only; the cached tensors remain private to
        # the latent checkpoint.  Exposing the receipt makes it possible to
        # verify that second sampling did not silently run Qwen again.
        result["qwen_conditioning_cache"] = dict(conditioning_cache)
    return result or None


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
        final_latents_path: Path | None = None,
        second_sampling: SecondSamplingSpec | None = None,
        refinement_latents_path: Path | None = None,
    ) -> NativeGenerationResult:
        output_path = output_path.resolve()
        if not output_path.is_relative_to(self._output_root):
            raise ValueError("output_path must stay inside the configured output root")
        if reference_images or reference_videos or reference_audios:
            raise RuntimeError("the compatibility pipeline does not implement Ref2VA")
        if second_sampling is not None or refinement_latents_path is not None:
            raise RuntimeError("the compatibility pipeline does not implement H3 second sampling")
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
            final_latents_path=None,
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
            "progress_percent": 0.0,
            "progress_stage": "cold",
            "progress_detail": "尚未加载模型",
        }

    def _loading_progress(self, percent: float, stage: str, detail: str) -> None:
        if self._warm_state.get("status") != "loading":
            return
        self._warm_state.update({
            "progress_percent": round(max(0.0, min(100.0, percent)), 1),
            "progress_stage": stage,
            "progress_detail": detail,
        })

    def preflight(self, engine: str) -> dict[str, Any]:
        return self._factory.preflight(engine)

    def _ensure_session(self, engine: str):
        launcher = normalize_launcher(engine)
        family = launcher_family(launcher)
        weight_tier = launcher_weight_tier(launcher)
        vram_profile = launcher_vram_profile(launcher)
        if self._built is not None and self._engine_name == launcher:
            return self._built
        if self._built is not None:
            self._built.session.close()
            self._built = None
            self._engine_name = None
        try:
            configure_progress = getattr(self._factory, "set_progress_callback", None)
            if callable(configure_progress):
                configure_progress(self._loading_progress)
            self._built = self._factory.build(launcher)
            self._engine_name = launcher
            self._warm_state = {
                "status": "ready",
                "engine": family,
                "launcher": launcher,
                "weight_tier": weight_tier,
                "vram_profile": vram_profile,
                "allocator_ceiling_gib": round(
                    float(getattr(self._built, "allocator_ceiling_gib", 0.0)), 3
                ),
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
                "pareto_policy_id": getattr(
                    self._built, "pareto_policy_id", None
                ),
                "pareto_candidate_id": getattr(
                    self._built, "pareto_candidate_id", None
                ),
                "lora_checkpoint": Path(
                    getattr(self._built, "lora_checkpoint", "")
                ).name,
                "lora_profile_id": getattr(
                    self._built, "lora_profile_id", None
                ),
                "lora_display_name": getattr(
                    self._built, "lora_display_name", None
                ),
                "lora_recommended_steps": list(getattr(
                    self._built, "lora_recommended_steps", ()
                )),
                "lora_default_steps": getattr(
                    self._built, "lora_default_steps", None
                ),
                "error": None,
                "progress_percent": 100.0,
                "progress_stage": "ready",
                "progress_detail": "模型引擎已就绪",
            }
        except Exception as error:
            self._warm_state = {
                "status": "failed", "engine": family,
                "launcher": launcher, "weight_tier": weight_tier,
                "vram_profile": vram_profile,
                "startup_seconds": None, "error": str(error),
                "progress_percent": 100.0,
                "progress_stage": "failed",
                "progress_detail": "模型引擎加载失败",
            }
            raise
        finally:
            configure_progress = getattr(self._factory, "set_progress_callback", None)
            if callable(configure_progress):
                configure_progress(None)
        return self._built

    async def preload(self, engine: str) -> None:
        engine = normalize_launcher(engine)
        family = launcher_family(engine)
        weight_tier = launcher_weight_tier(engine)
        vram_profile = launcher_vram_profile(engine)
        async with self._lock:
            if self._built is not None and self._engine_name == engine:
                return
            self._warm_state = {
                "status": "loading", "engine": family,
                "launcher": engine, "weight_tier": weight_tier,
                "vram_profile": vram_profile,
                "startup_seconds": None, "error": None,
                "progress_percent": 1.0,
                "progress_stage": "starting",
                "progress_detail": "开始加载模型引擎",
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
                "qwen_layer_cache", "pareto_policy_id", "pareto_candidate_id",
                "launcher", "weight_tier", "vram_profile",
                "allocator_ceiling_gib",
                "lora_checkpoint",
                "lora_profile_id", "lora_display_name",
                "lora_recommended_steps", "lora_default_steps",
                "progress_percent", "progress_stage", "progress_detail",
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
        final_latents_path: Path | None = None,
        second_sampling: SecondSamplingSpec | None = None,
        refinement_latents_path: Path | None = None,
    ) -> NativeGenerationResult | NativeCheckpointResult:
        from .hot_session import (
            HotSessionCancelled,
            HotSessionCheckpointResult,
            HotSessionRequest,
        )
        from .long_video_motion_detail import select_candidate

        is_second_sampling = second_sampling is not None
        if is_second_sampling != (refinement_latents_path is not None):
            raise ValueError(
                "second_sampling and refinement_latents_path must be supplied together"
            )
        if second_sampling is not None and second_sampling.model_variant != "base":
            raise ValueError("H3 second sampling is fixed to the Base weights")
        joint_plan = None
        second_attention_schedule: tuple[tuple[int, int, str], ...] = ()
        second_plan_summary: dict[str, Any] | None = None
        # INT8 V19 and distilled LoRA deliberately have separate scheduling
        # domains.  A Base release bundle (or isolated Base research overlay)
        # must never make LoRA borrow a forecast trajectory calibrated for the
        # 20-step model.
        use_v19 = bool(
            not is_second_sampling
            and
            spec.model_variant == "base"
            and
            spec.joint_acceleration_enabled
            and getattr(
                self._factory,
                "v19_scheduler_enabled",
                getattr(self._factory, "v19_release_enabled", False),
            )
        )
        joint_scheduler_id = None
        if (spec.joint_acceleration_enabled or is_second_sampling) and not use_v19:
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
            requested_steps = (
                int(second_sampling.steps)
                if second_sampling is not None
                else int(
                    spec.sampling_steps
                    or (
                        self._built.lora_default_steps
                        if spec.model_variant == "lora"
                        else 20
                    )
                )
            )
            requested_acceleration = (
                float(second_sampling.acceleration)
                if second_sampling is not None
                else float(spec.acceleration or 0.0)
            )
            if spec.model_variant == "lora":
                policy_id = (
                    os.environ.get(
                        "H3_NATIVE_RESEARCH_LORA_JOINT_POLICY", ""
                    ).strip()
                    or FROZEN_INT8_JOINT_POLICY
                )
                scheduler = H3LoraAccelerationScheduler(policy_id=policy_id)
                joint_plan = scheduler.plan(
                    max(4, requested_steps) if is_second_sampling else requested_steps,
                    requested_acceleration,
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
                    max(4, requested_steps) if is_second_sampling else requested_steps,
                    requested_acceleration,
                    # UltimateUpscale is a short, low-noise trajectory.  All
                    # solver positions remain real DiT evaluations; the same
                    # acceleration control is projected only onto Attention.
                    allow_forecast=not is_second_sampling,
                    workload=workload,
                )
                joint_scheduler_id = (
                    "h3_second_sampling_exact_attention_v1"
                    if is_second_sampling
                    else "h3_int8_frozen_round229"
                )
            if is_second_sampling and requested_steps < 4:
                # The first-pass optimizer's certified domain starts at four
                # trajectory points.  UltimateUpscale commonly uses one real
                # low-noise evaluation, so project the terminal N rows of the
                # four-step exact-only Attention policy onto N refinement
                # rows.  This never invents Forecast steps and preserves the
                # terminal layer protection learned by the optimizer.
                provisional = joint_plan
                terminal_start = provisional.total_steps - requested_steps
                remapped = []
                for (step, layer), action in sorted(
                    provisional.runtime_action_schedule().items()
                ):
                    if step >= terminal_start:
                        remapped.append((step - terminal_start, layer, action))
                second_attention_schedule = tuple(remapped)
                second_plan_summary = {
                    **{
                        key: value
                        for key, value in provisional.to_dict().items()
                        if key != "attention_decisions"
                    },
                    "schema_version": "h3_second_sampling_attention_projection_v1",
                    "total_steps": requested_steps,
                    "actual_step_indices": list(range(requested_steps)),
                    "forecast_step_indices": [],
                    "actual_evaluations": requested_steps,
                    "forecast_evaluations": 0,
                    "projection_source_steps": provisional.total_steps,
                    "projection_source_terminal_start": terminal_start,
                    "scheduler_family": joint_scheduler_id,
                    "model_variant": spec.model_variant,
                }
                joint_plan = None
        sparse_requested = (
            use_v19 and float(spec.acceleration or 0.0) > 0.0
        ) or (
            joint_plan is not None and joint_plan.uses_sparse_attention
        ) or (
            any(action != "dense" for _step, _layer, action in second_attention_schedule)
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
            built = await asyncio.to_thread(
                self._ensure_session, spec.runtime_launcher
            )
            runtime_config = getattr(built.session, "runtime_config", None)
            allocator_ceiling_gib = (
                float(runtime_config.max_device_bytes) / 1024**3
                if runtime_config is not None
                else {"24gb": 23.25, "16gb": 15.25, "8gb": 7.25}[
                    spec.vram_profile
                ]
            )
            if is_second_sampling:
                assert second_sampling is not None
                total_steps = second_sampling.steps
                actual_steps = tuple(range(total_steps))
            elif use_v19:
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
                if joint_plan is not None or use_v19 or is_second_sampling
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
            if is_second_sampling and execution_plan is None:
                from .planner import ExecutionPlan
                from .runtime import OffloadMode

                execution_plan = ExecutionPlan(
                    offload_mode=OffloadMode.BLOCK,
                    mlp_chunk_tokens=8192,
                    block_buffer_count=2,
                    prefetch_depth=1,
                    vae_spatial_tile=(288, 288),
                )
            if joint_plan is not None or use_v19 or is_second_sampling:
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
            ultimate_plan = None
            if second_sampling is not None:
                from .ultimate_upscale import plan_ultimate_upscale

                ultimate_plan = plan_ultimate_upscale(
                    target_width=second_sampling.width,
                    target_height=second_sampling.height,
                    frames=spec.frames,
                    device_budget_bytes=built.session._device_execution_budget_bytes(),
                    text_tokens=max(
                        128, min(1024, int(math.ceil(len(spec.prompt) * 0.55)))
                    ),
                    condition_count=(
                        int(first_frame is not None)
                        + int(last_frame is not None)
                        + len(reference_images)
                        + len(reference_videos)
                        + len(reference_audios)
                    ),
                    engine=spec.engine,
                    actual_evaluations=second_sampling.steps,
                    requested_mode=second_sampling.memory_mode,
                    weight_tier=built.weight_tier,
                    resource_profile=built.session.runtime_config.resource_profile,
                    allow_spatial_tiles=False,
                    temporal_window_frames=(
                        second_sampling.temporal_window_frames
                    ),
                )
                if not ultimate_plan.full_canvas and len(ultimate_plan.spatial) != 1:
                    raise RuntimeError(
                        "this second-sampling target requires spatial tiles; "
                        "the release executor currently enables the faster full-spatial "
                        "temporal-window executor only"
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
                release_byte_exact_optimizations=True,
                memory_mode=spec.memory_mode,
                attention_action_schedule=(
                    second_attention_schedule
                    if second_attention_schedule
                    else ()
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
                    second_plan_summary
                    if second_plan_summary is not None
                    else None
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
                scheduler_required_actual_step_indices=(
                    (int(spec.checkpoint_step or 1) - 1,)
                    if use_v19 and spec.execution_mode == "checkpoint"
                    else ()
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
                refinement_latents_path=refinement_latents_path,
                refinement_denoise=(
                    None if second_sampling is None else second_sampling.denoise
                ),
                refinement_spatial_mode=(
                    "strict" if second_sampling is None else second_sampling.spatial_mode
                ),
                preserve_refinement_audio=(
                    True if second_sampling is None else second_sampling.preserve_audio
                ),
                save_final_latents_path=final_latents_path,
                conditioning_cache_source_path=(
                    Path(refinement_latents_path).resolve()
                    if second_sampling is not None
                    and refinement_latents_path is not None
                    else None
                ),
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
                if ultimate_plan is not None and not ultimate_plan.full_canvas:
                    from .ultimate_upscale import (
                        append_av_temporal_piece,
                        slice_av_temporal_piece,
                    )
                    import torch

                    def run_temporal_windows():
                        source = torch.load(
                            Path(refinement_latents_path),
                            map_location="cpu",
                            weights_only=True,
                        )
                        source_video = source.get("video")
                        source_audio = source.get("audio")
                        if not isinstance(source_video, torch.Tensor) or source_video.ndim != 5:
                            raise ValueError("second-sampling source has invalid video latent")
                        if not isinstance(source_audio, torch.Tensor) or source_audio.ndim != 4:
                            raise ValueError("second-sampling source has invalid audio latent")
                        if source_video.shape[2] != ultimate_plan.temporal[-1].video_token_stop:
                            raise ValueError("UltimateUpscale plan does not cover source video clock")
                        if source_audio.shape[-1] < ultimate_plan.temporal[-1].audio_token_stop:
                            raise ValueError("UltimateUpscale plan does not cover source audio clock")

                        accumulated_video = None
                        accumulated_audio = None
                        all_phases: dict[str, float] = {}
                        all_steps: list[float] = []
                        window_profiles: list[dict[str, Any]] = []
                        rebuilt_conditioning: dict[str, Any] | None = None
                        peak_allocated = 0.0
                        peak_reserved = 0.0
                        window_count = len(ultimate_plan.temporal)
                        with tempfile.TemporaryDirectory(
                            prefix=".h3-ultimate-",
                            dir=str(output_path.parent),
                        ) as temporary_root:
                            temporary = Path(temporary_root)
                            for index, piece in enumerate(ultimate_plan.temporal):
                                if cancel_event.is_set():
                                    raise HotSessionCancelled(
                                        "native H3 generation cancelled"
                                    )
                                piece_video, piece_audio = slice_av_temporal_piece(
                                    source_video, source_audio, piece
                                )
                                piece_input = temporary / f"window-{index:02d}-source.pt"
                                piece_output = temporary / f"window-{index:02d}-sampled.pt"
                                torch.save(
                                    {
                                        "video": piece_video,
                                        "audio": piece_audio,
                                        "frames": piece.frames,
                                        "fps": request.fps,
                                        "width": source.get("width"),
                                        "height": source.get("height"),
                                        "engine": source.get("engine"),
                                        "seed": request.seed,
                                    },
                                    piece_input,
                                )

                                def piece_progress(event, *, _index=index):
                                    if progress_callback is None:
                                        return
                                    local = float(event.get("percent", 0.0)) / 100.0
                                    progress_callback({
                                        "percent": 8.0 + 76.0 * ((_index + local) / window_count),
                                        "stage": "second_sampling_window",
                                        "detail": (
                                            f"{second_sampling.resolution} 时间窗口 "
                                            f"{_index + 1}/{window_count} · "
                                            f"{event.get('detail', event.get('stage', '执行中'))}"
                                        ),
                                    })

                                piece_request = replace(
                                    request,
                                    frames=piece.frames,
                                    output_path=temporary / f"window-{index:02d}.mp4",
                                    first_frame=(request.first_frame if index == 0 else None),
                                    last_frame=(
                                        request.last_frame
                                        if index == window_count - 1
                                        else None
                                    ),
                                    progress_callback=piece_progress,
                                    refinement_latents_path=piece_input,
                                    save_final_latents_path=piece_output,
                                    internal_video_tokens=(
                                        piece.video_token_stop - piece.video_token_start
                                    ),
                                    internal_audio_tokens=(
                                        piece.audio_token_stop - piece.audio_token_start
                                    ),
                                    latent_only=True,
                                    formal_resume_state_path=None,
                                    checkpoint_after_step=None,
                                    checkpoint_state_path=None,
                                    preview_step_index=None,
                                    preview_output_path=None,
                                    terminal_refinement_initial_width=None,
                                    terminal_refinement_initial_height=None,
                                    terminal_refinement_steps=0,
                                )
                                piece_result = built.session.generate(piece_request)
                                sampled = torch.load(
                                    piece_output, map_location="cpu", weights_only=True
                                )
                                accumulated_video, accumulated_audio = append_av_temporal_piece(
                                    accumulated_video,
                                    accumulated_audio,
                                    sampled["video"],
                                    sampled["audio"],
                                    piece,
                                )
                                for name, seconds in piece_result.phases.items():
                                    all_phases[
                                        f"window_{index + 1:02d}.{name}"
                                    ] = seconds
                                all_steps.extend(piece_result.step_seconds)
                                window_profiles.append(piece_result.execution_profile)
                                latest_conditioning = getattr(
                                    built.session,
                                    "_last_conditioning_cache_payload",
                                    None,
                                )
                                if (
                                    rebuilt_conditioning is None
                                    and isinstance(latest_conditioning, dict)
                                ):
                                    # A legacy source latent has no persisted
                                    # Qwen cache.  The first temporal piece
                                    # necessarily rebuilds it; retain that exact
                                    # host payload so the stitched checkpoint is
                                    # automatically upgraded for future runs.
                                    rebuilt_conditioning = latest_conditioning
                                peak_allocated = max(
                                    peak_allocated, piece_result.peak_allocated_gib
                                )
                                peak_reserved = max(
                                    peak_reserved, piece_result.peak_reserved_gib
                                )
                                del sampled, piece_video, piece_audio

                            if accumulated_video is None or accumulated_audio is None:
                                raise RuntimeError("UltimateUpscale produced no windows")
                            # The upstream algorithm never re-samples audio.  Use
                            # the original full clock byte-for-byte instead of a
                            # numerically equivalent overlap blend.
                            del accumulated_audio
                            accumulated_audio = source_audio
                            stitched_path = (
                                Path(final_latents_path)
                                if final_latents_path is not None
                                else temporary / "stitched-final.pt"
                            )
                            stitched_path.parent.mkdir(parents=True, exist_ok=True)
                            stitched_document = {
                                "video": accumulated_video,
                                "audio": accumulated_audio,
                                "frames": request.frames,
                                "fps": request.fps,
                                "width": request.width,
                                "height": request.height,
                                "engine": source.get("engine"),
                                "seed": request.seed,
                            }
                            source_conditioning = source.get(
                                "qwen_conditioning_cache"
                            )
                            if not isinstance(source_conditioning, dict):
                                source_conditioning = rebuilt_conditioning
                            if isinstance(source_conditioning, dict):
                                stitched_document["qwen_conditioning_cache"] = (
                                    source_conditioning
                                )
                            torch.save(stitched_document, stitched_path)
                            decode_request = replace(
                                request,
                                refinement_latents_path=None,
                                refinement_denoise=None,
                                refinement_spatial_mode="strict",
                                save_final_latents_path=None,
                                internal_video_tokens=None,
                                internal_audio_tokens=None,
                                latent_only=False,
                                # The stitched high-resolution latent is intentionally
                                # decoded through the already validated exact
                                # host-temporal Video-VAE graph.  Reusing the
                                # per-window DiT plan here would materialize a
                                # full FP32 2K clip on GPU and consume the last
                                # ~0.6 GiB of the card for no speed benefit.
                                execution_plan=replace(
                                    request.execution_plan,
                                    vae_spatial_tile=request.execution_plan.vae_spatial_tile,
                                    vae_temporal_tile=6,
                                    vae_tile_batch_size=(
                                        8
                                        if built.vram_profile == "8gb"
                                        else 1
                                    ),
                                ),
                            )
                            decoded = built.session.decode_latent_checkpoint(
                                decode_request, stitched_path
                            )
                            all_phases.update({
                                f"final_decode.{name}": seconds
                                for name, seconds in decoded.phases.items()
                            })
                            return replace(
                                decoded,
                                total_seconds=time.monotonic() - started,
                                phases=all_phases,
                                step_seconds=tuple(all_steps),
                                execution_profile={
                                    **decoded.execution_profile,
                                    **(
                                        {
                                            "qwen_conditioning_cache": dict(
                                                window_profiles[0][
                                                    "qwen_conditioning_cache"
                                                ]
                                            )
                                        }
                                        if window_profiles
                                        and isinstance(
                                            window_profiles[0].get(
                                                "qwen_conditioning_cache"
                                            ),
                                            dict,
                                        )
                                        else {}
                                    ),
                                    "ultimate_upscale_windows": {
                                        "provenance": ultimate_plan.provenance,
                                        "count": window_count,
                                        "full_spatial_canvas": True,
                                        "audio_resampled": False,
                                        "latent_crossfade": True,
                                        "single_final_decode": True,
                                        "window_profiles": window_profiles,
                                    },
                                },
                                peak_allocated_gib=max(
                                    peak_allocated, decoded.peak_allocated_gib
                                ),
                                peak_reserved_gib=max(
                                    peak_reserved, decoded.peak_reserved_gib
                                ),
                            )

                    result = await asyncio.to_thread(run_temporal_windows)
                else:
                    result = await asyncio.to_thread(built.session.generate, request)
            except HotSessionCancelled as error:
                raise NativeGenerationCancelled(str(error)) from error
            if isinstance(result, HotSessionCheckpointResult):
                public_plan = _public_inference_plan(
                    getattr(result, "execution_profile", None)
                )
                if result.peak_allocated_gib > 0.0 or result.peak_reserved_gib > 0.0:
                    public_plan = dict(public_plan or {})
                    public_plan["runtime_memory"] = {
                        "peak_allocated_gib": round(result.peak_allocated_gib, 4),
                        "peak_reserved_gib": round(result.peak_reserved_gib, 4),
                        "allocator_ceiling_gib": round(allocator_ceiling_gib, 4),
                        "vram_profile": spec.vram_profile,
                    }
                return NativeCheckpointResult(
                    runtime_key=(
                        f"{spec.engine}:{spec.weight_tier}:{spec.vram_profile}:native-sm89"
                    ),
                    elapsed_seconds=round(time.monotonic() - started, 3),
                    checkpoint_path=result.checkpoint_path,
                    preview_path=result.preview_path,
                    completed_steps=result.completed_steps,
                    total_steps=result.total_steps,
                    stage_seconds=dict(result.phases),
                    inference_plan=public_plan,
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
            public_plan = _public_inference_plan(
                getattr(result, "execution_profile", None)
            )
            peak_allocated_gib = float(getattr(result, "peak_allocated_gib", 0.0))
            peak_reserved_gib = float(getattr(result, "peak_reserved_gib", 0.0))
            if peak_allocated_gib > 0.0 or peak_reserved_gib > 0.0:
                public_plan = dict(public_plan or {})
                public_plan["runtime_memory"] = {
                    "peak_allocated_gib": round(peak_allocated_gib, 4),
                    "peak_reserved_gib": round(peak_reserved_gib, 4),
                    "allocator_ceiling_gib": round(allocator_ceiling_gib, 4),
                    "vram_profile": spec.vram_profile,
                }
            if ultimate_plan is not None:
                public_plan = dict(public_plan or {})
                public_plan["ultimate_upscale"] = ultimate_plan.telemetry()
            if second_sampling is not None:
                public_plan = dict(public_plan or {})
                public_plan["second_sampling_solver"] = {
                    "model_variant": "base",
                    "sampler": "sa_solver",
                    "scheduler": "simple",
                    "video_shift": 6.0,
                    "audio_shift": 3.0,
                    "steps": second_sampling.steps,
                    "strength": second_sampling.strength,
                    "denoise": second_sampling.denoise,
                    "start_sigma": refinement_sigma_schedule(
                        second_sampling.steps,
                        second_sampling.denoise,
                        6.0,
                    )[0],
                    "forecast_enabled": False,
                }
            return NativeGenerationResult(
                runtime_key=(
                    f"{spec.engine}:{spec.weight_tier}:"
                    f"{spec.vram_profile}:native-sm89"
                ),
                elapsed_seconds=round(time.monotonic() - started, 3),
                output_path=result.output_path,
                stage_seconds=phases,
                inference_plan=public_plan,
                final_latents_path=(
                    final_latents_path
                    if final_latents_path is not None and final_latents_path.is_file()
                    else None
                ),
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
                "launcher": None, "weight_tier": None, "vram_profile": None,
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
