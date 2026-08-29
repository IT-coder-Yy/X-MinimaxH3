#!/usr/bin/env python3
"""Test non-causal Video-VAE tail context without rerunning the H3 DiT."""

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
    load_native_video_vae,
    postprocess_native_video,
)
from h3serve.native_engine.adapters.vae_compile import (
    enable_feed_forward_compile,
    prewarm_feed_forward_compile,
)


def _arguments() -> argparse.Namespace:
    serve_root = Path(__file__).resolve().parents[1]
    main_root = serve_root.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=288)
    parser.add_argument("--model-root", type=Path, default=serve_root / "models")
    parser.add_argument("--minimax-source", type=Path, default=main_root / "MiniMax-H3")
    return parser.parse_args()


def _context(normalized: torch.Tensor, mode: str) -> torch.Tensor:
    """Append one 5-token H3 temporal cycle as decoder-only context."""

    if normalized.shape[2] < 10:
        raise ValueError("tail-context probe requires at least ten latent tokens")
    last = normalized[:, :, -5:]
    previous = normalized[:, :, -10:-5]
    if mode == "edge":
        suffix = normalized[:, :, -1:].expand(-1, -1, 5, -1, -1)
    elif mode == "reflect":
        suffix = last.flip(2)
    elif mode == "cycle":
        suffix = last
    elif mode == "cycle_linear":
        suffix = last + (last - previous)
    else:
        raise ValueError(mode)
    return torch.cat((normalized, suffix), dim=2)


def _decode(model, normalized, mean, std, decoded_frames: int, keep_frames: int):
    mean = mean.to(normalized.device).view(1, -1, 1, 1, 1)
    std = std.to(normalized.device).view(1, -1, 1, 1, 1)
    latent = normalized.float() * std + mean
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        decoded = model.decode_base(latent, frame_num=decoded_frames)
    decoded = decoded[:, :, :keep_frames]
    output = postprocess_native_video(decoded, output_dtype="uint8")
    torch.cuda.synchronize()
    return output, {
        "seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }


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
    target_frames = int(payload["frames"])
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
        "target_frames": target_frames,
        "context_tokens": 5,
        "decoded_frames": target_frames + 17,
        "feed_forward_compiled_modules": enabled,
        "variants": {},
    }
    for mode in ("edge", "reflect", "cycle", "cycle_linear"):
        candidate = _context(normalized, mode)
        output, timing = _decode(
            model, candidate, mean, std, target_frames + 17, target_frames
        )
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
