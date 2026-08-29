#!/usr/bin/env python3
"""Launch the Turbo LoRA backend on RTX 4090."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8208)
    parser.add_argument("--comfy-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--reserve-vram", type=float, default=1.5)
    parser.add_argument("--vae-tile-size", type=int, default=288)
    parser.add_argument("--async-offload-streams", type=int, default=4)
    parser.add_argument("--compile-dit-blocks", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune-no-cudagraphs"),
        default="default",
    )
    parser.add_argument("--disable-sage", action="store_true")
    args, extra = parser.parse_known_args()
    if args.vae_tile_size < 256 or args.vae_tile_size % 16:
        parser.error("VAE tile size must be a multiple of 16 and at least 256")
    if args.async_offload_streams < 1:
        parser.error("async offload streams must be positive")

    import torch
    import comfy_kitchen as ck

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("the Turbo engine requires RTX 4090 / SM89")
    ck.enable_backend("cuda")
    state = ck.list_backends().get("cuda", {})
    if not state.get("available") or state.get("disabled"):
        raise SystemExit(f"the Turbo CUDA backend is unavailable: {state}")

    comfy = args.comfy_dir.resolve()
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
        "--whitelist-custom-nodes", "H3Serve-Turbo",
        "--enable-triton-backend",
        "--preview-method", "none",
        "--async-offload", str(args.async_offload_streams),
    ]
    if not args.disable_sage:
        command.append("--use-sage-attention")
    command.extend(extra)

    env = os.environ.copy()
    env["H3SERVE_COMFY_DIR"] = str(comfy)
    env["H3SERVE_TURBO_VAE_TILE_SIZE"] = str(args.vae_tile_size)
    env["H3SERVE_TURBO_COMPILE_DIT_BLOCKS"] = "1" if args.compile_dit_blocks else "0"
    env["H3SERVE_TURBO_COMPILE_MODE"] = args.compile_mode
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(HERE / "vendor"), env.get("PYTHONPATH")
    )))
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.chdir(comfy)
    os.execve(sys.executable, command, env)


if __name__ == "__main__":
    main()
