#!/usr/bin/env python3
"""Launch the configurable original-weight backend on RTX 4090."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from schedules import PRESETS, parse_actual_steps, schedule_label


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8221)
    parser.add_argument("--comfy-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--reserve-vram", type=float, default=0.0)
    parser.add_argument("--quality-preset", choices=tuple(PRESETS), default="balanced")
    parser.add_argument("--actual-steps")
    parser.add_argument("--mlp-chunk-tokens", type=int, default=8192)
    parser.add_argument("--long-mlp-chunk-tokens", type=int, default=4096)
    parser.add_argument("--long-sequence-threshold", type=int, default=20000)
    parser.add_argument("--sample-channels", type=int, default=32)
    parser.add_argument("--vae-tile-size", type=int, default=288)
    parser.add_argument("--async-offload-streams", type=int, default=2)
    args, extra = parser.parse_known_args()
    try:
        actual_steps = parse_actual_steps(
            preset=args.quality_preset, custom=args.actual_steps
        )
    except ValueError as error:
        parser.error(str(error))
    if args.async_offload_streams < 1:
        parser.error("async offload streams must be positive")

    import torch
    import comfy_kitchen as ck
    import sageattention._qattn_sm89  # noqa: F401

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("the original engine requires RTX 4090 / SM89")
    ck.disable_backend("cuda")
    ck.enable_backend("triton")
    ck.set_backend_priority(["triton", "eager"])

    comfy = args.comfy_dir.resolve()
    env = os.environ.copy()
    env["H3SERVE_COMFY_DIR"] = str(comfy)
    env["H3SERVE_ACTUAL_STEPS"] = ",".join(map(str, actual_steps))
    env["H3SERVE_SCHEDULE_LABEL"] = schedule_label(actual_steps)
    env["H3SERVE_MLP_CHUNK_TOKENS"] = str(args.mlp_chunk_tokens)
    env["H3SERVE_LONG_MLP_CHUNK_TOKENS"] = str(args.long_mlp_chunk_tokens)
    env["H3SERVE_LONG_SEQUENCE_THRESHOLD"] = str(args.long_sequence_threshold)
    env["H3SERVE_SAMPLE_CHANNELS"] = str(args.sample_channels)
    env["H3SERVE_VAE_TILE_SIZE"] = str(args.vae_tile_size)
    env["H3_SPECTRUM_MAX_CONSECUTIVE_FORECASTS"] = "16"
    env["H3_SPECTRUM_MIN_ACTUAL_AFTER_FORECAST"] = "0"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(HERE), env.get("PYTHONPATH"))))
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    command = [
        sys.executable,
        str(HERE / "entry.py"),
        "--disable-auto-launch",
        "--listen", "127.0.0.1",
        "--port", str(args.port),
        "--reserve-vram", str(args.reserve_vram),
        "--extra-model-paths-config", str(args.model_config.resolve()),
        "--output-directory", str(args.output_directory.resolve()),
        "--disable-all-custom-nodes",
        "--whitelist-custom-nodes", "ComfyUI-Spectrum-MiniMax-H3",
        "--enable-triton-backend",
        "--use-sage-attention",
        "--async-offload", str(args.async_offload_streams),
        *extra,
    ]
    print(
        f"original engine: preset={args.quality_preset} schedule={schedule_label(actual_steps)}",
        flush=True,
    )
    os.chdir(comfy)
    os.execve(sys.executable, command, env)


if __name__ == "__main__":
    main()

