#!/usr/bin/env python3
"""Decode one generated AV latent twice to isolate Video-VAE compilation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from h3serve.native_engine.adapters.real_vae import (
    decode_native_video,
    load_native_audio_vae,
    load_native_video_vae,
)
from h3serve.native_engine.adapters.sampling_mux import AtomicPyAVMuxer
from h3serve.native_engine.adapters.vae_compile import (
    enable_transformer_block_compile,
    prewarm_transformer_block_compile,
    transformer_block_compile,
)


def parse_args() -> argparse.Namespace:
    serve_root = Path(__file__).resolve().parents[1]
    main_root = serve_root.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prefix", default="same_latent_vae_ab")
    parser.add_argument("--model-root", type=Path, default=serve_root / "models")
    parser.add_argument("--minimax-source", type=Path, default=main_root / "MiniMax-H3")
    parser.add_argument(
        "--lightx-source",
        type=Path,
        default=main_root.parent / "backend-compare/sources/LightX2V",
    )
    return parser.parse_args()


def timed(operation):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = operation()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this audit requires one RTX 4090 / SM89 GPU")
    payload = torch.load(args.latents, map_location="cpu", weights_only=True)
    video_latent = payload["video"]
    audio_latent = payload["audio"]
    frames = int(payload["frames"])
    fps = int(payload["fps"])
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    video_model, mean, std = load_native_video_vae(
        args.minimax_source,
        args.model_root / "vae/minimax_h3_video_vae_fp16.safetensors",
        device="cuda:0",
        tile_size=288,
        tile_batch_size=1,
        compile_feed_forward=False,
    )
    video_latent = video_latent.cuda()
    eager_video, eager_seconds = timed(
        lambda: decode_native_video(
            video_model, video_latent, mean, std, frames, output_dtype="uint8"
        )
    )
    compiled_modules = enable_transformer_block_compile(video_model)
    prewarm_transformer_block_compile(video_model)
    with transformer_block_compile(True):
        compiled_video, compiled_seconds = timed(
            lambda: decode_native_video(
                video_model, video_latent, mean, std, frames, output_dtype="uint8"
            )
        )
    del video_model, video_latent
    torch.cuda.empty_cache()

    audio_model = load_native_audio_vae(
        args.lightx_source,
        args.minimax_source,
        args.model_root / "vae/minimax_h3_audio_vae_fp32.safetensors",
        device="cuda:0",
    )
    audio_latent = audio_latent.cuda()
    flattened = audio_latent.permute(0, 2, 1, 3).reshape(
        2, 32, audio_latent.shape[-1]
    )
    with torch.inference_mode():
        audio, audio_seconds = timed(
            lambda: audio_model.decode(flattened, stereo_batch=True, return_cpu=True)
        )

    eager_path = output_root / f"{args.prefix}_eager.mp4"
    compiled_path = output_root / f"{args.prefix}_blockcompile.mp4"
    muxer = AtomicPyAVMuxer(output_root=output_root)
    eager_probe = muxer.write(
        video=eager_video,
        audio=audio,
        sample_rate=32000,
        fps=fps,
        output_path=eager_path,
    )
    compiled_probe = muxer.write(
        video=compiled_video,
        audio=audio,
        sample_rate=32000,
        fps=fps,
        output_path=compiled_path,
    )
    delta = (compiled_video.to(torch.int16) - eager_video.to(torch.int16)).abs()
    report = {
        "schema_version": 1,
        "source_latents": str(args.latents.resolve()),
        "engine": payload.get("engine"),
        "seed": payload.get("seed"),
        "geometry": [payload.get("width"), payload.get("height"), frames],
        "compiled_modules": compiled_modules,
        "eager_decode_seconds": eager_seconds,
        "compiled_decode_seconds": compiled_seconds,
        "decode_speedup": eager_seconds / compiled_seconds,
        "audio_decode_seconds": audio_seconds,
        "pixel_max_abs_uint8": int(delta.max()),
        "pixel_mean_abs_uint8": float(delta.float().mean()),
        "pixel_equal_fraction": float((delta == 0).float().mean()),
        "eager_output": str(eager_path),
        "compiled_output": str(compiled_path),
        "eager_probe": eager_probe,
        "compiled_probe": compiled_probe,
    }
    report_path = output_root / f"{args.prefix}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
