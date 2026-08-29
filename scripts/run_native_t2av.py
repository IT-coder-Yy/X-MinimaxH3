#!/usr/bin/env python3
"""Run the standalone H3 T2AV core and emit an MP4 plus timing evidence.

This is the first real-weight integration runner.  The DiT, Qwen conditioner,
scheduler, sampler, and muxer are provided by ``h3serve.native_engine``.  Until
the already-audited Apache VAE graphs are vendored into the release package,
the runner accepts their source roots explicitly; it never starts or imports
ComfyUI.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from h3serve.native_engine.adapters.conditioning_vae import (
    PackedQwen3VLT2AVConditioner,
)
from h3serve.native_engine.adapters.vae_tiling import install_bounded_tile_batching
from h3serve.native_engine.adapters.vae_compile import enable_feed_forward_compile
from h3serve.native_engine.adapters.sampling_mux import (
    AVPrediction,
    AtomicPyAVMuxer,
    ResMultistepAVSampler,
    SamplingPlan,
    TurboAVSampler,
    TurboClockMode,
    simple_sigma_schedule,
)
from h3serve.native_engine.model import (
    SafeTensorSource,
    assemble_full_pruned_dit,
    comfy_kitchen_int8_kernel,
    load_full_silu_curve,
    load_larry_updates_from_safetensors,
    sage_attention_sm89,
)
from h3serve.native_engine.sm89_policy import configure_sm89_runtime


DEFAULT_PROMPT = (
    "A cinematic aerial shot of medieval crusaders marching through deep mud "
    "toward a besieged stone fortress at dawn. Horses, banners and siege "
    "engines move naturally. Volumetric mist, realistic documentary style. "
    "Stereo audio: boots and hooves in mud, armor rattling, distant war drums "
    "and wind. No subtitles, no on-screen text."
)


@contextmanager
def timed_phase(metrics: dict[str, Any], name: str):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        metrics["phases"][name] = {
            "seconds": round(time.perf_counter() - started, 4),
            "peak_allocated_gib": round(
                torch.cuda.max_memory_allocated() / (1024**3), 4
            ),
            "peak_reserved_gib": round(
                torch.cuda.max_memory_reserved() / (1024**3), 4
            ),
        }
        print(name, metrics["phases"][name], flush=True)


def release_cuda(*values: Any) -> None:
    for value in values:
        del value
    gc.collect()
    torch.cuda.empty_cache()


def load_video_vae(
    source_root: Path,
    checkpoint: Path,
    *,
    device: str | torch.device = "cuda",
    tile_size: int | None = None,
    tile_batch_size: int = 1,
    compile_feed_forward: bool = False,
):
    """Load the official Apache fused H3 video VAE without ComfyUI."""

    fl2va_root = source_root / "FL2VA"
    sys.path.insert(0, str(fl2va_root))
    from video_vae.klvae import AutoencoderKLLegacy
    from video_vae.parallel import get_parallel_state

    # The released VAE expects this process-local single-GPU topology even
    # when no torch.distributed process group is created.
    parallel_state = get_parallel_state()
    if not parallel_state:
        parallel_state.update(
            {
                "group_size": 1,
                "group_rank": 0,
                "local_process_group": None,
                "sp_size": 1,
                "sp_rank": 0,
                "sp_enabled": False,
                "sp_process_group": None,
                "tp_size": 1,
                "tp_rank": 0,
            }
        )

    source_dir = fl2va_root / "video_vae" / "source"
    wrapper_config = json.loads(
        (fl2va_root / "video_vae" / "config.json").read_text(encoding="utf-8")
    )
    config = AutoencoderKLLegacy.load_config(str(source_dir))
    resolved_tile_size = (
        int(wrapper_config["vae_tile_size"])
        if tile_size is None
        else int(tile_size)
    )
    if resolved_tile_size < 128 or resolved_tile_size % 16:
        raise ValueError("video VAE tile size must be >= 128 and divisible by 16")
    load_kwargs = {
        "clip_length": int(wrapper_config["vae_clip_length"]),
        "token_drop": int(wrapper_config["vae_token_drop"]),
        "encoder_tiling": int(wrapper_config["vae_encoder_tiling"]),
        "decoder_tiling": int(wrapper_config["vae_decoder_tiling"]),
        "parallel_tiling": int(wrapper_config["vae_parallel_tiling"]),
        "tile_size": resolved_tile_size,
        "tile_overlap_min": int(wrapper_config["vae_tile_overlap_min"]),
        "encoder_parallel": 0,
        "decoder_parallel": 0,
        "chunk_dim": -1,
    }
    model, _ = AutoencoderKLLegacy.from_config(
        config, return_unused_kwargs=True, **load_kwargs
    )
    model.half()
    state = load_file(str(checkpoint))
    latent_mean = state.pop("latents_mean")
    latent_std = state.pop("latents_std")
    model.load_state_dict(state, strict=True)
    del state
    install_bounded_tile_batching(model, tile_batch_size)
    if compile_feed_forward:
        enable_feed_forward_compile(model)
    return model.eval().requires_grad_(False).to(device), latent_mean, latent_std


def decode_video(
    model,
    normalized: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    frame_count: int,
    *,
    output_dtype: str = "float32",
) -> torch.Tensor:
    mean = latent_mean.to(normalized.device).view(1, -1, 1, 1, 1)
    std = latent_std.to(normalized.device).view(1, -1, 1, 1, 1)
    latent = normalized.float() * std + mean
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        decoded = model.decode_base(latent, frame_num=frame_count)
    from h3serve.native_engine.adapters.real_vae import postprocess_native_video

    return postprocess_native_video(decoded, output_dtype=output_dtype)


def load_audio_vae(
    lightx_root: Path,
    source_root: Path,
    checkpoint: Path,
    *,
    device: str | torch.device = "cuda",
):
    """Load Apache audio graph after folding its legacy weight-norm wrappers."""

    # Import only the audited H3 component.  LightX2V's package __init__ eagerly
    # imports every server/runner backend (Cosmos, metrics, imageio, ...), none
    # of which belongs in this standalone inference path.
    if "lightx2v" not in sys.modules:
        package = types.ModuleType("lightx2v")
        package.__path__ = [str(lightx_root / "lightx2v")]
        package.__package__ = "lightx2v"
        sys.modules["lightx2v"] = package
    sys.path.insert(0, str(lightx_root))
    from lightx2v.models.audio_encoders.hf.minimax_h3.audio_vae import (
        MiniMaxH3AudioVAE,
    )

    config = json.loads(
        (source_root / "audio_vae" / "config.json").read_text(encoding="utf-8")
    )
    model = MiniMaxH3AudioVAE(config, device=str(device), cpu_offload=False)
    removed = 0
    for module in model.modules():
        try:
            torch.nn.utils.remove_weight_norm(module)
            removed += 1
        except (AttributeError, ValueError):
            pass
    if removed != 172:
        raise RuntimeError(f"audio VAE weight-norm fold count changed: {removed}")
    state = load_file(str(checkpoint))
    latent_mean = state.pop("latents_mean")
    latent_std = state.pop("latents_std")
    model.load_state_dict(state, strict=True)
    del state
    model.latents_mean.copy_(latent_mean)
    model.latents_std.copy_(latent_std)
    return model.eval().requires_grad_(False).to(device)


def parse_args() -> argparse.Namespace:
    serve_root = Path(__file__).resolve().parents[1]
    main_root = serve_root.parents[1]
    backend_compare = main_root.parent / "backend-compare"
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=4404)
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--engine", choices=("original", "lora"), default="original")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--lora-strength", type=float, default=1.0)
    parser.add_argument(
        "--turbo-audio-transport",
        choices=("shared_video", "dual_shift"),
        default="shared_video",
        help="audio clock representation; shared_video matches current Comfy H3",
    )
    parser.add_argument(
        "--profile-step",
        type=int,
        default=0,
        help="capture Torch profiler key averages for this 1-based DiT step",
    )
    parser.add_argument(
        "--stop-after-denoise",
        action="store_true",
        help="write timing evidence after DiT sampling without loading either VAE",
    )
    parser.add_argument(
        "--debug-step-dir",
        type=Path,
        help="optionally save sampler x/denoised tensors after every evaluation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=serve_root / "runtime" / "outputs" / "native_t2av_480p5s.mp4",
    )
    parser.add_argument("--model-root", type=Path, default=serve_root / "models")
    parser.add_argument("--minimax-source", type=Path, default=main_root / "MiniMax-H3")
    parser.add_argument(
        "--lightx-source",
        type=Path,
        default=backend_compare / "sources" / "LightX2V",
    )
    args = parser.parse_args()
    if args.steps is None:
        args.steps = 6 if args.engine == "lora" else 20
    return args


def main() -> int:
    args = parse_args()
    configure_sm89_runtime(quant_backend="cuda", smoke_test=True)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this integration runner requires one SM89 GPU")
    if args.width % 32 or args.height % 32:
        raise SystemExit("width and height must be multiples of 32")
    if args.frames < 5 or (args.frames - 5) % 17:
        raise SystemExit("frames must satisfy 17*n+5")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output.with_suffix(".timing.json")
    metrics: dict[str, Any] = {
        "status": "running",
        "prompt": args.prompt,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "fps": args.fps,
        "steps": args.steps,
        "engine": args.engine,
        "lora_strength": args.lora_strength if args.engine == "lora" else None,
        "turbo_audio_transport": (
            args.turbo_audio_transport if args.engine == "lora" else None
        ),
        "phases": {},
    }
    started_total = time.perf_counter()
    base = args.model_root / "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    text_checkpoint = args.model_root / "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    video_checkpoint = args.model_root / "vae/minimax_h3_video_vae_fp16.safetensors"
    audio_checkpoint = args.model_root / "vae/minimax_h3_audio_vae_fp32.safetensors"
    lora_checkpoint = args.model_root / "loras/minimax_h3_turbo_v4_step600_ema.safetensors"
    egrid_checkpoint = (
        Path(__file__).resolve().parents[1]
        / "backends/turbo/custom_node/h3_silu_temb_grid.safetensors"
    )

    try:
        conditioner = PackedQwen3VLT2AVConditioner(
            text_checkpoint,
            args.minimax_source / "tokenizer",
        )
        with timed_phase(metrics, "text_encode"):
            encoded = conditioner.encode_prompt(args.prompt)
        context_5120 = encoded.prompt_embeds
        text_tags = encoded.text_token_tags
        del conditioner, encoded
        gc.collect()
        torch.cuda.empty_cache()

        with timed_phase(metrics, "dit_load"):
            lora_updates = None
            full_silu_curve = None
            if args.engine == "lora":
                lora_updates = load_larry_updates_from_safetensors(
                    str(lora_checkpoint),
                    strength=args.lora_strength,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                full_silu_curve = load_full_silu_curve(str(egrid_checkpoint))
            with SafeTensorSource(base) as source:
                dit = assemble_full_pruned_dit(
                    source,
                    device="cuda",
                    compute_dtype=torch.bfloat16,
                    int8_kernel=comfy_kitchen_int8_kernel,
                    attention_backend=sage_attention_sm89,
                    lora_updates=lora_updates,
                    full_silu_curve=full_silu_curve,
                )
            dit.eval().requires_grad_(False)
            with torch.inference_mode():
                context = dit.token_refiner(
                    dit.condition_proj(context_5120[0].to(dit.compute_dtype))
                ).unsqueeze(0)
        del context_5120
        gc.collect()
        torch.cuda.empty_cache()

        duration = args.frames / float(args.fps)
        video_shape = (1, 24, ((args.frames - 5) // 17) * 5 + 2, args.height // 16, args.width // 16)
        audio_shape = (1, 32, 2, round(duration * 40))
        generator = torch.Generator("cpu").manual_seed(args.seed)
        video = torch.randn(video_shape, generator=generator, dtype=torch.float32).cuda()
        audio = torch.randn(audio_shape, generator=generator, dtype=torch.float32).cuda()
        sigmas = simple_sigma_schedule(args.steps, 12.0)
        # FullH3DiT converts the audio velocity onto the video clock through
        # d(sigma_audio)/d(sigma_video), exactly like the accepted Comfy path.
        plan = SamplingPlan(
            sampler="turbo" if args.engine == "lora" else "res_multistep",
            video_sigmas=sigmas,
            audio_sigmas=sigmas,
            actual_step_indices=tuple(range(args.steps)),
            video_shift=12.0,
            audio_shift=3.0,
        )
        layout = None
        step_seconds: list[float] = []
        last_denoised = None

        def predict(video_value, audio_value, clock, *, step_index, is_actual_step):
            nonlocal layout, last_denoised
            if not is_actual_step:
                raise RuntimeError("baseline runner does not forecast denoise steps")
            step_started = time.perf_counter()
            def call_dit():
                return dit(
                    video_value,
                    audio_value,
                    context,
                    torch.tensor([clock.video_sigma], device="cuda"),
                    output_frame_count=args.frames,
                    text_token_tags=text_tags,
                    layout=layout,
                    audio_transport_scale=(
                        4.0
                        if args.engine == "lora"
                        and args.turbo_audio_transport == "shared_video"
                        else None
                    ),
                )

            with torch.inference_mode():
                if args.profile_step == step_index + 1:
                    with torch.profiler.profile(
                        activities=(
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ),
                        record_shapes=True,
                        profile_memory=False,
                    ) as profiler:
                        result = call_dit()
                    table = profiler.key_averages(group_by_input_shape=True).table(
                        sort_by="cuda_time_total", row_limit=80
                    )
                    metrics["profile_step"] = step_index + 1
                    metrics["profile_key_averages"] = table
                    print(table, flush=True)
                else:
                    result = call_dit()
            layout = result.layout
            torch.cuda.synchronize()
            step_seconds.append(time.perf_counter() - step_started)
            print(
                f"denoise step {step_index + 1}/{args.steps}: "
                f"{step_seconds[-1]:.3f}s",
                flush=True,
            )
            sigma = clock.video_sigma
            prediction = AVPrediction(
                video_denoised=video_value - result.video * sigma,
                audio_denoised=audio_value - result.audio * sigma,
            )
            last_denoised = prediction
            return prediction

        def debug_step(index, clock, step_video, step_audio):
            if args.debug_step_dir is None:
                return
            if last_denoised is None:
                raise RuntimeError("sampler callback ran before model prediction")
            args.debug_step_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "x": torch.cat(
                        (step_video.flatten(1), step_audio.flatten(1)), dim=-1
                    ).detach().cpu(),
                    "denoised": torch.cat(
                        (
                            last_denoised.video_denoised.flatten(1),
                            last_denoised.audio_denoised.flatten(1),
                        ),
                        dim=-1,
                    ).detach().cpu(),
                    "sigma": clock.video_sigma,
                    "sigma_next": clock.video_sigma_next,
                },
                args.debug_step_dir / f"native_step_{index:02d}.pt",
            )

        with timed_phase(metrics, "denoise"):
            sampler = (
                TurboAVSampler(
                    TurboClockMode.SHARED_VIDEO
                    if args.turbo_audio_transport == "shared_video"
                    else TurboClockMode.DUAL_SHIFT
                )
                if args.engine == "lora"
                else ResMultistepAVSampler()
            )
            video, audio = sampler.sample(
                video,
                audio,
                plan,
                predict,
                callback=debug_step if args.debug_step_dir is not None else None,
            )
            if (
                args.engine == "lora"
                and args.turbo_audio_transport == "shared_video"
            ):
                # ModelSamplingAV.process_latent_out performs this inverse
                # transport after the sampler reaches sigma zero.
                audio.div_(4.0)
        metrics["denoise_step_seconds"] = [round(value, 4) for value in step_seconds]
        del dit, context, text_tags, layout
        gc.collect()
        torch.cuda.empty_cache()

        if args.stop_after_denoise:
            metrics["status"] = "complete"
            metrics["stopped_after_denoise"] = True
            metrics["total_seconds"] = round(time.perf_counter() - started_total, 4)
            metrics_path.write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
            return 0

        with timed_phase(metrics, "video_vae_load"):
            video_vae, video_mean, video_std = load_video_vae(
                args.minimax_source, video_checkpoint
            )
        with timed_phase(metrics, "video_decode"):
            decoded_video = decode_video(
                video_vae,
                video,
                video_mean,
                video_std,
                args.frames,
                output_dtype="uint8",
            )
        del video_vae, video_mean, video_std, video
        gc.collect()
        torch.cuda.empty_cache()

        with timed_phase(metrics, "audio_vae_load"):
            audio_vae = load_audio_vae(
                args.lightx_source, args.minimax_source, audio_checkpoint
            )
        with timed_phase(metrics, "audio_decode"):
            flattened_audio = audio.permute(0, 2, 1, 3).reshape(
                2, 32, audio.shape[-1]
            )
            with torch.inference_mode():
                decoded_audio = audio_vae.decode(
                    flattened_audio,
                    stereo_batch=True,
                    return_cpu=True,
                )
        del audio_vae, audio, flattened_audio
        gc.collect()
        torch.cuda.empty_cache()

        with timed_phase(metrics, "mux"):
            muxer = AtomicPyAVMuxer(output_root=output.parent)
            mux_result = muxer.write(
                video=decoded_video,
                audio=decoded_audio,
                sample_rate=32000,
                fps=args.fps,
                output_path=output,
            )
        metrics["media"] = mux_result
        metrics["status"] = "complete"
        metrics["total_seconds"] = round(time.perf_counter() - started_total, 4)
        metrics_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        metrics["status"] = "failed"
        metrics["error"] = f"{type(error).__name__}: {error}"
        metrics["total_seconds"] = round(time.perf_counter() - started_total, 4)
        metrics_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
