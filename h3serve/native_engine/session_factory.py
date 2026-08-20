"""Production construction of the measured RTX 4090 Native hot session."""

from __future__ import annotations

import importlib
import ctypes
import gc
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from ..memory_policy import HOST_MEMORY_PROFILES, HostMemoryProfile
from .hot_session import NativeT2AVHotSession
from .local_checkpoint_cache import (
    materialize_local_checkpoint,
    materialize_qwen_layer_cache,
    should_localize_checkpoint,
)
from ..contract import engine_family, resolve_engine
from .long_video_motion_detail import candidate_requested


def _generic_sparse_requested() -> bool:
    return (
        os.environ.get("H3_NATIVE_ENABLE_SPARSE", "0") == "1"
        or os.environ.get("H3_NATIVE_REVIEW_SPARSE", "0") == "1"
    )


def _review_sparse_requested() -> bool:
    # Product installs may opt into the pinned SM89 extension explicitly. The
    # legacy review flag remains accepted for existing calibration scripts.
    return (
        _generic_sparse_requested()
        or candidate_requested()
        or (
            _v19_release_bundle_path().is_file()
            and os.environ.get("H3_NATIVE_ENABLE_SPARSE", "auto") != "0"
        )
    )


def _v19_release_bundle_path() -> Path:
    serve_root = Path(__file__).resolve().parents[2]
    configured = os.environ.get("H3_NATIVE_V19_RELEASE_BUNDLE", "").strip()
    return (
        Path(configured).expanduser().resolve()
        if configured
        else (
            serve_root
            / "h3serve/native_engine/planner/evidence/v19_release_bundle.json"
        )
    )


def _review_fused_rms_requested() -> bool:
    return os.environ.get("H3_NATIVE_REVIEW_FUSED_RMS", "0") == "1"


def _prepare_review_sparse_import() -> None:
    build_dir = os.environ.get("H3_NATIVE_SPARGE_BUILD_DIR")
    if build_dir:
        path = Path(build_dir).expanduser().resolve()
        if not path.is_dir():
            raise RuntimeError(f"H3_NATIVE_SPARGE_BUILD_DIR is not a directory: {path}")
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    try:
        importlib.import_module("spas_sage_attn")
    except Exception as error:
        raise RuntimeError(
            "review sparse routing was requested but the pinned SpargeAttention "
            "SM89 extension is unavailable"
        ) from error


@dataclass(frozen=True, slots=True)
class NativeSessionPaths:
    model_root: Path
    minimax_source: Path
    lightx_source: Path
    turbo_curve: Path
    output_root: Path

    def resolved(self) -> "NativeSessionPaths":
        return NativeSessionPaths(
            model_root=self.model_root.resolve(),
            minimax_source=self.minimax_source.resolve(),
            lightx_source=self.lightx_source.resolve(),
            turbo_curve=self.turbo_curve.resolve(),
            output_root=self.output_root.resolve(),
        )


@dataclass(frozen=True, slots=True)
class BuiltNativeSession:
    session: NativeT2AVHotSession
    startup_seconds: float
    startup_tasks: dict[str, float]
    qwen_storage: str = "source"
    qwen_layer_cache: bool = False
    v19_release_bundle: str | None = None
    v19_release_digest: str | None = None


class NativeSessionFactory:
    """Build one real-weight session without importing a UI graph runtime."""

    def __init__(
        self,
        paths: NativeSessionPaths,
        memory_profile: HostMemoryProfile | None = None,
    ) -> None:
        self.paths = paths.resolved()
        self.memory_profile = memory_profile or HOST_MEMORY_PROFILES["fullspeed"]

    def set_memory_profile(self, profile: HostMemoryProfile) -> None:
        self.memory_profile = profile

    def set_output_root(self, output_root: Path) -> None:
        """Retarget future sessions while the owning hot engine is cold."""

        self.paths = NativeSessionPaths(
            model_root=self.paths.model_root,
            minimax_source=self.paths.minimax_source,
            lightx_source=self.paths.lightx_source,
            turbo_curve=self.paths.turbo_curve,
            output_root=output_root.resolve(),
        )

    @property
    def v19_release_enabled(self) -> bool:
        return (
            _v19_release_bundle_path().is_file()
            and os.environ.get("H3_NATIVE_ENABLE_SPARSE", "auto") != "0"
        )

    def preflight(self, engine: str) -> dict[str, object]:
        family = engine_family(engine) if engine not in ("first_last", "reference") else engine
        root = self.paths.model_root
        checks = {
            "base_weight": (
                root / "diffusion_models" / (
                    "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
                    if family == "reference"
                    else "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
                )
            ).is_file(),
            "text_weight": (
                root / "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
            ).is_file(),
            "video_vae_weight": (
                root / "vae/minimax_h3_video_vae_fp16.safetensors"
            ).is_file(),
            "audio_vae_weight": (
                root / "vae/minimax_h3_audio_vae_fp32.safetensors"
            ).is_file(),
            "tokenizer": (self.paths.minimax_source / "tokenizer").is_dir(),
            "video_vae_source": (
                self.paths.minimax_source / "FL2VA/video_vae/klvae.py"
            ).is_file(),
            "audio_vae_config": (
                self.paths.minimax_source / "audio_vae/config.json"
            ).is_file(),
            "lightx_audio_source": (
                self.paths.lightx_source
                / "lightx2v/models/audio_encoders/hf/minimax_h3/audio_vae.py"
            ).is_file(),
        }
        # Both request routes are embedded in one family session.  Readiness is
        # therefore honest only when the base and its hot-switchable adapter
        # assets are present.
        checks.update(
            {
                "lora_weight": (
                    root / "loras/minimax_h3_turbo_v4_step600_ema.safetensors"
                ).is_file(),
                "turbo_curve": self.paths.turbo_curve.is_file(),
            }
        )
        sparse_requested = _review_sparse_requested()
        sparse_available = False
        if sparse_requested:
            try:
                _prepare_review_sparse_import()
                checks["review_sparse_backend"] = True
                sparse_available = True
            except RuntimeError:
                checks["review_sparse_backend"] = False
        configured_v19 = os.environ.get(
            "H3_NATIVE_V19_RELEASE_BUNDLE", ""
        ).strip()
        if configured_v19:
            checks["v19_release_bundle"] = _v19_release_bundle_path().is_file()
        return {
            "ready": all(checks.values()),
            "checks": checks,
            "capabilities": {
                "sparse_attention": sparse_available,
                "v19_certified_frontier": bool(
                    sparse_available and self.v19_release_enabled
                ),
            },
        }

    @property
    def sparse_attention_available(self) -> bool:
        """Whether this process was explicitly built for approximate attention."""

        return _review_sparse_requested()

    def build(self, engine: str) -> BuiltNativeSession:
        import torch

        from .adapters.conditioning_vae import (
            H3AudioVAEAdapter,
            H3VideoVAEAdapter,
            PackedQwen3VLT2AVConditioner,
        )
        from .adapters.real_vae import (
            decode_native_video,
            load_native_audio_vae,
            load_native_video_vae,
        )
        from .adapters.vae_compile import (
            enable_transformer_block_compile,
            prewarm_feed_forward_compile,
            prewarm_transformer_block_compile,
        )
        from .model import (
            SafeTensorSource,
            assemble_full_pruned_dit,
            comfy_kitchen_int8_kernel,
            load_full_silu_curve,
            load_larry_updates_from_safetensors,
            make_joint_action_scheduled_sparge_attention_sm89,
            sage_attention_sm89,
        )
        from .planner import (
            RTX4090Planner,
            review_combined_profiles_2026_08_12,
            review_fused_rms_profiles_2026_08_12,
            review_sparse_profiles_2026_08_12,
            validated_profiles_for_engine,
        )
        from .runtime import ImmutablePinnedModuleResidency, RuntimeConfig
        from .sm89_policy import configure_sm89_runtime

        family = engine_family(engine) if engine not in ("first_last", "reference") else engine
        base_engine = resolve_engine(family, "base")
        lora_engine = resolve_engine(family, "lora")
        preflight = self.preflight(family)
        if not preflight["ready"]:
            missing = ", ".join(
                name for name, ready in preflight["checks"].items() if not ready
            )
            raise RuntimeError(f"native {engine} runtime is incomplete: {missing}")
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
            raise RuntimeError("native H3 release requires one RTX 4090 / SM89 GPU")
        kernel_runtime = configure_sm89_runtime(
            quant_backend="cuda", smoke_test=True
        )
        review_sparse = _review_sparse_requested()
        # A certified V19 bundle contains request-local physical sparse action
        # schedules.  Loading the selector without the matching dispatching
        # backend would silently execute Dense Attention under a sparse plan
        # certificate, invalidating both timing and quality provenance.
        generic_sparse = _generic_sparse_requested() or self.v19_release_enabled
        long_video_review = candidate_requested()
        review_fused_rms = _review_fused_rms_requested()
        if review_sparse:
            _prepare_review_sparse_import()
        v19_selector = None
        v19_bundle_source = None
        v19_bundle_digest = None
        if self.v19_release_enabled:
            from .planner import (
                FIXED_TOPK_ACTION_IMPLEMENTATION,
                ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
                ROUND215_ACTION_IMPLEMENTATION,
                ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
                ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION,
                V19RuntimeSelector,
                build_v19_bootstrap_registry,
                capture_v19_runtime_fingerprint,
                load_v19_release_bundle,
            )

            sparse_package = importlib.import_module("spas_sage_attn")
            sparse_build_dir = Path(sparse_package.__file__).resolve().parent.parent
            serve_root = Path(__file__).resolve().parents[2]
            runtime_fingerprint = capture_v19_runtime_fingerprint(
                serve_root=serve_root,
                sparge_build_dir=sparse_build_dir,
                kernel_runtime=kernel_runtime,
            )
            registry = build_v19_bootstrap_registry(implementation_ids={
                "fixed_topk": FIXED_TOPK_ACTION_IMPLEMENTATION,
                "round215": ROUND215_ACTION_IMPLEMENTATION,
                "round188": ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
                "round228": ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
                "round229": ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION,
            })
            loaded_v19 = load_v19_release_bundle(
                _v19_release_bundle_path(),
                registry=registry,
            )
            v19_selector = V19RuntimeSelector(
                loaded_v19.catalog,
                runtime_digest=runtime_fingerprint.digest,
            )
            v19_bundle_source = str(loaded_v19.source)
            v19_bundle_digest = loaded_v19.bundle_digest
        if long_video_review:
            from .long_video_motion_detail import make_attention_backend

            attention_backend = make_attention_backend()
        elif generic_sparse:
            attention_backend = make_joint_action_scheduled_sparge_attention_sm89()
        else:
            attention_backend = sage_attention_sm89

        root = self.paths.model_root
        base = root / "diffusion_models" / (
            "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
            if family == "reference"
            else "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
        )
        text_source = root / "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
        video_checkpoint = root / "vae/minimax_h3_video_vae_fp16.safetensors"
        audio_checkpoint = root / "vae/minimax_h3_audio_vae_fp32.safetensors"
        lora_checkpoint = root / "loras/minimax_h3_turbo_v4_step600_ema.safetensors"
        task_seconds: dict[str, float] = {}

        def timed(name, operation):
            started = time.perf_counter()
            result = operation()
            task_seconds[name] = time.perf_counter() - started
            return result

        # Compact hosts cannot retain the 13.5+ GiB packed Qwen slab. Under
        # WSL the release model is commonly linked into /mnt/c (9p/DrvFS),
        # where thousands of layer-local tensor reads dominate every new
        # prompt. Keep a byte-identical, reclaimable ext4 disk copy instead.
        # This is storage locality, not a second resident weight cache.
        text = (
            timed(
                "qwen_native_checkpoint",
                lambda: materialize_local_checkpoint(text_source),
            )
            if self.memory_profile.key == "compact"
            else text_source
        )
        qwen_storage = "native_cache" if text != text_source.resolve() else "source"
        qwen_layer_cache = (
            timed(
                "qwen_layer_cache",
                lambda: materialize_qwen_layer_cache(text),
            )
            if self.memory_profile.key == "compact"
            else None
        )
        if self.memory_profile.key == "compact":
            if qwen_storage == "native_cache":
                print(
                    "64GB Qwen storage: Linux native cache ready",
                    flush=True,
                )
            elif should_localize_checkpoint(text_source):
                print(
                    "WARNING: 64GB Qwen storage is still on a WSL cross-drive "
                    "mount; generation remains available but new-prompt latency "
                    "will be higher. Check Linux cache disk space or "
                    "H3_SERVE_LOCAL_MODEL_CACHE.",
                    flush=True,
                )
            if qwen_layer_cache is not None:
                print(
                    "64GB Qwen streaming: execution-ordered layer cache ready",
                    flush=True,
                )
        conditioner = PackedQwen3VLT2AVConditioner(
            text,
            self.paths.minimax_source / "tokenizer",
            cache_pinned_weights=self.memory_profile.cache_qwen_weights,
            layer_cache_dir=qwen_layer_cache,
        )

        def prepare_dit():
            def load_lora_assets():
                return (
                    load_larry_updates_from_safetensors(
                        str(lora_checkpoint),
                        strength=1.0,
                        device="cpu",
                        dtype=torch.bfloat16,
                    ),
                    load_full_silu_curve(str(self.paths.turbo_curve)),
                )

            updates, curve = timed("dit_lora_assets", load_lora_assets)

            def assemble_dit():
                with SafeTensorSource(str(base)) as source:
                    return assemble_full_pruned_dit(
                        source,
                        device="cpu",
                        compute_dtype=torch.bfloat16,
                        int8_kernel=comfy_kitchen_int8_kernel,
                        attention_backend=attention_backend,
                        lora_updates=updates,
                        full_silu_curve=curve,
                    )

            model = timed("dit_graph_assembly", assemble_dit)
            model.eval().requires_grad_(False)
            residency = ImmutablePinnedModuleResidency(
                "transformer", model,
                pin_host_weights=self.memory_profile.pin_model_weights,
                copy_host_weights=self.memory_profile.copy_model_weights,
            )
            timed("dit_pin_host", residency.prepare_host)
            return residency

        def prepare_video():
            model, mean, std = load_native_video_vae(
                self.paths.minimax_source,
                video_checkpoint,
                device="cpu",
                tile_size=288,
                compile_feed_forward=True,
            )
            if long_video_review or generic_sparse:
                # Install both eager and compiled dispatch paths while the
                # model is still on CPU.  The joint scheduler and retained
                # long-video route share this mature mechanical baseline;
                # request-local execution plans still choose the path.
                enable_transformer_block_compile(model)
            residency = ImmutablePinnedModuleResidency(
                "video_vae", model,
                pin_host_weights=self.memory_profile.pin_model_weights,
                copy_host_weights=self.memory_profile.copy_model_weights,
            )
            residency.prepare_host()
            return residency, mean, std

        def prepare_audio():
            model = load_native_audio_vae(
                self.paths.lightx_source,
                self.paths.minimax_source,
                audio_checkpoint,
                device="cpu",
            )
            residency = ImmutablePinnedModuleResidency(
                "audio_vae", model,
                pin_host_weights=self.memory_profile.pin_model_weights,
                copy_host_weights=self.memory_profile.copy_model_weights,
            )
            residency.prepare_host()
            return residency

        started = time.perf_counter()
        if self.memory_profile.parallel_model_build:
            with ThreadPoolExecutor(max_workers=4, thread_name_prefix="h3-native-startup") as pool:
                futures = {
                    pool.submit(timed, "dit_cache", prepare_dit): "dit",
                    pool.submit(timed, "video_vae_cache", prepare_video): "video",
                    pool.submit(timed, "audio_vae_cache", prepare_audio): "audio",
                }
                if self.memory_profile.cache_qwen_weights:
                    futures[
                        pool.submit(timed, "qwen_cache", conditioner.prepare_host_cache)
                    ] = "qwen"
                prepared = {}
                for future in as_completed(futures):
                    prepared[futures[future]] = future.result()
        else:
            # Low-capacity hosts trade cold-start latency for a lower assembly
            # peak. The resulting immutable models and generation math are
            # identical to the parallel path.
            prepared = {
                "dit": timed("dit_cache", prepare_dit),
                "video": timed("video_vae_cache", prepare_video),
                "audio": timed("audio_vae_cache", prepare_audio),
            }
            if self.memory_profile.cache_qwen_weights:
                timed("qwen_cache", conditioner.prepare_host_cache)

        transformer = prepared["dit"]
        video_vae, video_mean, video_std = prepared["video"]
        audio_vae = prepared["audio"]

        def prewarm_video_vae_compile():
            video_vae.move_to("cuda:0", non_blocking=False)
            try:
                if long_video_review or generic_sparse:
                    prewarm_transformer_block_compile(video_vae.value)
                else:
                    prewarm_feed_forward_compile(video_vae.value)
                torch.cuda.synchronize()
            finally:
                video_vae.move_to("cpu", non_blocking=False)
                torch.cuda.empty_cache()

        timed("video_vae_compile_warmup", prewarm_video_vae_compile)

        def release_startup_host_scratch() -> None:
            """Return one-time graph assembly scratch to the OS.

            Live immutable weights already reside in compact registered host
            slabs. CPython can release superseded source tensors while glibc
            still keeps their multi-gigabyte arenas mapped. This trim runs once
            after startup/prewarm and never enters the request path.
            """

            gc.collect()
            try:
                libc = ctypes.CDLL(None)
                malloc_trim = libc.malloc_trim
                malloc_trim.argtypes = [ctypes.c_size_t]
                malloc_trim.restype = ctypes.c_int
                malloc_trim(0)
            except (AttributeError, OSError):
                pass

        timed("release_startup_host_scratch", release_startup_host_scratch)

        def video_decoder(model, latents, frame_count):
            return decode_native_video(
                model,
                latents,
                video_mean,
                video_std,
                frame_count,
                output_dtype="uint8",
            )

        frame_adapter = H3VideoVAEAdapter(
            video_vae.value,
            latents_mean=video_mean.tolist(),
            latents_std=video_std.tolist(),
        )

        def frame_encoder(_model, request):
            if request.reference_images or request.reference_videos:
                return frame_adapter.encode_references(request)
            return frame_adapter.encode_conditioning(request)

        audio_adapter = H3AudioVAEAdapter(
            audio_vae.value,
            latents_mean=audio_vae.value.latents_mean.tolist(),
            latents_std=audio_vae.value.latents_std.tolist(),
        )

        def audio_condition_encoder(_model, request):
            from .adapters.conditioning_vae import prepare_reference_audios

            return tuple(
                audio_adapter.encode(item.waveform.to("cuda:0"))
                for item in prepare_reference_audios(request)
            )

        def audio_decoder(model, latents):
            flattened = latents.permute(0, 2, 1, 3).reshape(
                2, 32, latents.shape[-1]
            )
            with torch.inference_mode():
                return model.decode(flattened, stereo_batch=True, return_cpu=True)

        # The request chooses base or LoRA without rebuilding this graph.  Give
        # the shared planner the measured profiles for both routes.
        profiles = (
            validated_profiles_for_engine(base_engine)
            + validated_profiles_for_engine(lora_engine)
        )
        if generic_sparse:
            profiles += tuple(
                profile
                for profile in review_sparse_profiles_2026_08_12()
                if base_engine in profile.supported_engines
                or lora_engine in profile.supported_engines
                or family == "reference"
            )
        if review_fused_rms:
            profiles += tuple(
                profile
                for profile in review_fused_rms_profiles_2026_08_12()
                if base_engine in profile.supported_engines
                or lora_engine in profile.supported_engines
                or family == "reference"
            )
        if generic_sparse and review_fused_rms:
            profiles += tuple(
                profile
                for profile in review_combined_profiles_2026_08_12()
                if base_engine in profile.supported_engines
                or lora_engine in profile.supported_engines
                or family == "reference"
            )
        session = NativeT2AVHotSession(
            engine=base_engine,
            conditioner=conditioner,
            transformer=transformer,
            video_vae=video_vae,
            audio_vae=audio_vae,
            decode_video=video_decoder,
            decode_audio=audio_decoder,
            encode_video_conditioning=frame_encoder,
            encode_audio_conditioning=audio_condition_encoder,
            output_root=self.paths.output_root,
            runtime_config=RuntimeConfig(),
            planner=RTX4090Planner(
                profiles,
                allow_experimental=generic_sparse or review_fused_rms,
            ),
            attention_backend=attention_backend,
            v19_selector=v19_selector,
        )
        return BuiltNativeSession(
            session=session,
            startup_seconds=time.perf_counter() - started,
            startup_tasks=task_seconds,
            qwen_storage=qwen_storage,
            qwen_layer_cache=qwen_layer_cache is not None,
            v19_release_bundle=v19_bundle_source,
            v19_release_digest=v19_bundle_digest,
        )


__all__ = ["BuiltNativeSession", "NativeSessionFactory", "NativeSessionPaths"]
