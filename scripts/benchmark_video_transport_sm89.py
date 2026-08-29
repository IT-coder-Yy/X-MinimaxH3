#!/usr/bin/env python3
"""Audit the exact uint8 Video-VAE host transport on a real RTX 4090."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from h3serve.native_engine.adapters.real_vae import (
    decode_native_video,
    load_native_video_vae,
)
from h3serve.native_engine.adapters.sampling_mux.mux import _video_uint8


def main() -> int:
    serve_root = Path(__file__).resolve().parents[1]
    main_root = serve_root.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=73)
    parser.add_argument(
        "--output",
        type=Path,
        default=serve_root / "runtime/calibration/video_transport_sm89.json",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this audit requires one RTX 4090 / SM89 GPU")
    if args.width % 32 or args.height % 32 or (args.frames - 5) % 17:
        raise SystemExit("invalid H3 video geometry")

    model, mean, std = load_native_video_vae(
        main_root / "MiniMax-H3",
        serve_root / "models/vae/minimax_h3_video_vae_fp16.safetensors",
        device="cuda:0",
        tile_size=288,
        tile_batch_size=1,
        compile_feed_forward=True,
    )
    latent_frames = ((args.frames - 5) // 17) * 5 + 2
    normalized = torch.randn(
        (1, 24, latent_frames, args.height // 16, args.width // 16),
        generator=torch.Generator("cpu").manual_seed(4090),
        dtype=torch.float32,
    ).cuda()

    runs = []
    outputs = {}
    for dtype in ("float32", "uint8"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        output = decode_native_video(
            model, normalized, mean, std, args.frames, output_dtype=dtype
        )
        torch.cuda.synchronize()
        runs.append(
            {
                "output_dtype": dtype,
                "seconds": time.perf_counter() - started,
                "host_bytes": output.numel() * output.element_size(),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            }
        )
        outputs[dtype] = _video_uint8(output)
    delta = np.abs(
        outputs["float32"].astype(np.int16) - outputs["uint8"].astype(np.int16)
    )
    report = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(),
        "geometry": [args.width, args.height, args.frames],
        "runs": runs,
        "byte_exact": bool(np.array_equal(outputs["float32"], outputs["uint8"])),
        "different_bytes": int(np.count_nonzero(delta)),
        "max_byte_delta": int(delta.max(initial=0)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["byte_exact"]:
        raise SystemExit("uint8 transport changed codec input bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
