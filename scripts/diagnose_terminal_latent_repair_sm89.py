#!/usr/bin/env python3
"""Evaluate guarded repairs for a collapsed terminal H3 video latent token."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image

from h3serve.native_engine.adapters.real_vae import (
    decode_native_video,
    load_native_video_vae,
)
from h3serve.native_engine.adapters.vae_compile import (
    enable_feed_forward_compile,
    prewarm_feed_forward_compile,
)
from h3serve.native_engine.terminal_latent_guard import (
    stabilize_terminal_video_latent_,
)


def _arguments() -> argparse.Namespace:
    serve_root = Path(__file__).resolve().parents[1]
    main_root = serve_root.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=288)
    parser.add_argument(
        "--modes",
        default="bottom_half_linear,production_adaptive,bottom_energy_global,bottom_energy_channel,bottom_hold,bottom_linear,full_linear",
    )
    parser.add_argument("--model-root", type=Path, default=serve_root / "models")
    parser.add_argument("--minimax-source", type=Path, default=main_root / "MiniMax-H3")
    return parser.parse_args()


def _vertical_mask(height: int, *, device: torch.device) -> torch.Tensor:
    """Feather from zero at 52% height to one at 76% height."""

    y = torch.linspace(0.0, 1.0, height, device=device, dtype=torch.float32)
    weight = ((y - 0.52) / (0.76 - 0.52)).clamp(0.0, 1.0)
    return weight.view(1, 1, 1, height, 1)


def _repair(normalized: torch.Tensor, mode: str) -> torch.Tensor:
    if normalized.shape[2] < 11:
        raise ValueError("terminal repair needs two earlier matching temporal phases")
    result = normalized.clone()
    if mode == "production_adaptive":
        stabilize_terminal_video_latent_(result)
        return result
    current = result[:, :, -1:]
    phase_new = result[:, :, -6:-5]
    phase_old = result[:, :, -11:-10]
    mask = _vertical_mask(result.shape[-2], device=result.device)
    split = round(result.shape[-2] * 0.52)
    if mode in ("bottom_energy_global", "bottom_energy_channel"):
        current_bottom = current[..., split:, :]
        previous_bottom = phase_new[..., split:, :]
        if mode == "bottom_energy_channel":
            dimensions = (-2, -1)
        else:
            dimensions = (1, 2, 3, 4)
        current_mean = current_bottom.float().mean(dim=dimensions, keepdim=True)
        current_std = current_bottom.float().std(
            dim=dimensions, keepdim=True, unbiased=False
        ).clamp_min(1.0e-6)
        previous_std = previous_bottom.float().std(
            dim=dimensions, keepdim=True, unbiased=False
        )
        gain = (previous_std / current_std).clamp(1.0, 1.5)
        enhanced = current_mean + (current.float() - current_mean) * gain
        current.lerp_(enhanced.to(current.dtype), mask)
        return result
    if mode.endswith("hold"):
        estimate = phase_new
    elif mode.endswith("linear"):
        estimate = phase_new + (phase_new - phase_old)
    else:
        raise ValueError(mode)
    if mode.startswith("full_"):
        current.copy_(estimate)
    elif mode.startswith("bottom_"):
        current.lerp_(estimate, mask)
    elif mode.startswith("bottom_half_"):
        mask = _vertical_mask(result.shape[-2], device=result.device).mul_(0.5)
        current.lerp_(estimate, mask)
    else:
        raise ValueError(mode)
    return result


def _save_contact(output: torch.Tensor, path: Path) -> None:
    images = []
    for index in range(output.shape[2] - 10, output.shape[2]):
        frame = output[0, :, index].permute(1, 2, 0).contiguous().numpy()
        image = Image.fromarray(frame, mode="RGB")
        image.thumbnail((640, 368), Image.Resampling.LANCZOS)
        images.append(image)
    canvas = Image.new("RGB", (images[0].width * 5, images[0].height * 2))
    for index, image in enumerate(images):
        canvas.paste(image, ((index % 5) * image.width, (index // 5) * image.height))
    canvas.save(path)


def main() -> int:
    args = _arguments()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this diagnostic requires one RTX 4090 / SM89 GPU")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.latents, map_location="cpu", weights_only=True)
    normalized = payload["video"].cuda()
    frames = int(payload["frames"])
    model, mean, std = load_native_video_vae(
        args.minimax_source,
        args.model_root / "vae/minimax_h3_video_vae_fp16.safetensors",
        device="cuda:0",
        tile_size=args.tile_size,
        tile_batch_size=1,
        compile_feed_forward=False,
    )
    os.environ["MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT"] = "1"
    enabled = enable_feed_forward_compile(model)
    prewarm_feed_forward_compile(model)
    report = {
        "schema_version": 1,
        "source": str(args.latents.resolve()),
        "frames": frames,
        "feed_forward_compiled_modules": enabled,
        "variants": {},
    }
    modes = tuple(mode.strip() for mode in args.modes.split(",") if mode.strip())
    for mode in modes:
        candidate = _repair(normalized, mode)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = decode_native_video(
            model, candidate, mean, std, frames, output_dtype="uint8"
        )
        torch.cuda.synchronize()
        timing = {
            "seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        }
        report["variants"][mode] = timing
        _save_contact(output, output_root / f"{mode}_tail10.jpg")
        Image.fromarray(
            output[0, :, -1].permute(1, 2, 0).contiguous().numpy(), mode="RGB"
        ).save(output_root / f"{mode}_last.png")
        print(json.dumps({mode: timing}), flush=True)
        del candidate, output
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    del normalized, model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
