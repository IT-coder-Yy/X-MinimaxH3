#!/usr/bin/env python3
"""Isolate H3 Video-VAE tail corruption on one deterministic 720p latent.

The probe keeps the latent bit-identical while toggling the two project-owned
memory/compile paths.  It reports full-frame and tail-region uint8 deltas and
writes the last frame of every variant for visual inspection.
"""

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


def parse_args() -> argparse.Namespace:
    serve_root = Path(__file__).resolve().parents[1]
    main_root = serve_root.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=736)
    parser.add_argument("--frames", type=int, default=362)
    parser.add_argument("--seed", type=int, default=82303)
    parser.add_argument("--tile-size", type=int, default=288)
    parser.add_argument("--alternate-tile-size", type=int, default=256)
    parser.add_argument(
        "--latents",
        type=Path,
        help="use a saved HotSession final_av_latents.pt instead of synthetic noise",
    )
    parser.add_argument("--model-root", type=Path, default=serve_root / "models")
    parser.add_argument("--minimax-source", type=Path, default=main_root / "MiniMax-H3")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _decode(model, normalized, mean, std, frames: int, *, stream: bool):
    os.environ["MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT"] = "1" if stream else "0"
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = decode_native_video(
        model, normalized, mean, std, frames, output_dtype="uint8"
    )
    torch.cuda.synchronize()
    return output, {
        "seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }


def _delta(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError("VAE variants returned different shapes")

    def stats(left, right):
        delta = (left.to(torch.int16) - right.to(torch.int16)).abs()
        return {
            "max_abs_uint8": int(delta.max()),
            "mean_abs_uint8": float(delta.float().mean()),
            "equal_fraction": float((delta == 0).float().mean()),
        }

    height = reference.shape[-2]
    return {
        "full_strided": stats(
            reference[:, :, ::8, ::8, ::8], candidate[:, :, ::8, ::8, ::8]
        ),
        "tail5_full": stats(reference[:, :, -5:], candidate[:, :, -5:]),
        "tail5_top": stats(
            reference[:, :, -5:, : height // 2],
            candidate[:, :, -5:, : height // 2],
        ),
        "tail5_bottom": stats(
            reference[:, :, -5:, height // 2 :],
            candidate[:, :, -5:, height // 2 :],
        ),
    }


def _save_last_frame(output: torch.Tensor, path: Path) -> None:
    frame = output[0, :, -1].permute(1, 2, 0).contiguous().numpy()
    Image.fromarray(frame, mode="RGB").save(path)


def _save_tail_contact(output: torch.Tensor, path: Path) -> None:
    frames = []
    for index in range(output.shape[2] - 10, output.shape[2]):
        frame = output[0, :, index].permute(1, 2, 0).contiguous().numpy()
        image = Image.fromarray(frame, mode="RGB")
        image.thumbnail((640, 368), Image.Resampling.LANCZOS)
        frames.append(image)
    canvas = Image.new("RGB", (frames[0].width * 5, frames[0].height * 2))
    for index, image in enumerate(frames):
        canvas.paste(image, ((index % 5) * image.width, (index // 5) * image.height))
    canvas.save(path)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this diagnostic requires one RTX 4090 / SM89 GPU")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = args.model_root / "vae/minimax_h3_video_vae_fp16.safetensors"
    model, mean, std = load_native_video_vae(
        args.minimax_source,
        checkpoint,
        device="cuda:0",
        tile_size=args.tile_size,
        tile_batch_size=1,
        compile_feed_forward=False,
    )
    if args.latents is not None:
        payload = torch.load(args.latents, map_location="cpu", weights_only=True)
        normalized = payload["video"].cuda()
        args.frames = int(payload["frames"])
        args.width = int(payload["width"])
        args.height = int(payload["height"])
        args.seed = int(payload["seed"])
    else:
        latent_frames = ((args.frames - 5) // 17) * 5 + 2
        generator = torch.Generator("cpu").manual_seed(args.seed)
        normalized = torch.randn(
            (1, 24, latent_frames, args.height // 16, args.width // 16),
            generator=generator,
            dtype=torch.float32,
        ).cuda()
    if args.width % 16 or args.height % 16 or (args.frames - 5) % 17:
        raise SystemExit("geometry must satisfy the H3 latent grid")

    report = {
        "schema_version": 1,
        "geometry": [args.width, args.height, args.frames],
        "latent_shape": list(normalized.shape),
        "seed": args.seed,
        "tile_size": args.tile_size,
        "source_latents": None if args.latents is None else str(args.latents.resolve()),
        "variants": {},
        "comparisons_to_eager_nonstream": {},
    }

    outputs = {}
    for name, stream in (("eager_nonstream", False), ("eager_stream", True)):
        output, timing = _decode(model, normalized, mean, std, args.frames, stream=stream)
        outputs[name] = output
        report["variants"][name] = timing
        _save_last_frame(output, output_root / f"{name}_last.png")
        _save_tail_contact(output, output_root / f"{name}_tail10.jpg")
        print(json.dumps({name: timing}), flush=True)

    enabled_modules = enable_feed_forward_compile(model)
    prewarm_feed_forward_compile(model)
    report["feed_forward_compiled_modules"] = enabled_modules
    for name, stream in (("compiled_nonstream", False), ("compiled_stream", True)):
        output, timing = _decode(model, normalized, mean, std, args.frames, stream=stream)
        outputs[name] = output
        report["variants"][name] = timing
        _save_last_frame(output, output_root / f"{name}_last.png")
        _save_tail_contact(output, output_root / f"{name}_tail10.jpg")
        print(json.dumps({name: timing}), flush=True)

    if args.alternate_tile_size:
        model.decoder_tile_size = int(args.alternate_tile_size)
        alternate_name = f"compiled_stream_tile{args.alternate_tile_size}"
        alternate, timing = _decode(
            model, normalized, mean, std, args.frames, stream=True
        )
        outputs[alternate_name] = alternate
        report["variants"][alternate_name] = timing
        _save_last_frame(alternate, output_root / f"{alternate_name}_last.png")
        _save_tail_contact(alternate, output_root / f"{alternate_name}_tail10.jpg")
        print(json.dumps({alternate_name: timing}), flush=True)

    reference = outputs["eager_nonstream"]
    for name, output in outputs.items():
        report["comparisons_to_eager_nonstream"][name] = _delta(reference, output)
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)

    del outputs, reference, normalized, model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
