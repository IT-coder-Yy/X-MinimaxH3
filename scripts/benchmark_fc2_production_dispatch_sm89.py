#!/usr/bin/env python3
"""Compare the current and a forced CUTLASS dispatch at H3's FC2 boundary.

Unlike benchmark_cutlass_configs_h3_sm89.py, this benchmark includes the
production fused SwiGLU + ConvRot row quantizer. It loads one real H3 FC2
weight/scale pair, so the only difference between the two routes is CUTLASS
tile config 13 (the generic heuristic) versus config 0 (the CUDA 13/SM89
candidate).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from safetensors import safe_open

from h3serve.native_engine.sm89_policy import configure_sm89_runtime


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "models/MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors",
    )
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--current-config", type=int, default=13)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=11)
    parser.add_argument(
        "--output",
        type=Path,
        default=root
        / "runtime/calibration/cuda13_optimization/fc2_production_dispatch_sm89.json",
    )
    args = parser.parse_args()
    if args.rows <= 0 or args.warmup < 1 or args.iterations < 3:
        parser.error("rows must be positive, warmup >= 1 and iterations >= 3")
    return args


def elapsed_ms(callable_) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    result = callable_()
    stop.record()
    stop.synchronize()
    del result
    return float(start.elapsed_time(stop))


def summarize(samples: list[float]) -> dict[str, float | list[float]]:
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def main() -> int:
    args = parse_args()
    runtime = configure_sm89_runtime(quant_backend="cuda", smoke_test=True)
    import comfy_kitchen
    import comfy_kitchen.backends.cuda as cuda_backend

    prefix = "blocks.0.mlp.fc2"
    with safe_open(str(args.checkpoint), framework="pt", device="cpu") as checkpoint:
        qweight = checkpoint.get_tensor(f"{prefix}.weight").to("cuda:0")
        weight_scale = checkpoint.get_tensor(f"{prefix}.weight_scale").to("cuda:0")
    out_features, inner = qweight.shape
    raw = torch.randn(
        (args.rows, inner * 2),
        device="cuda:0",
        dtype=torch.bfloat16,
        generator=torch.Generator("cuda:0").manual_seed(82303),
    )
    expanded_scale = (
        weight_scale
        if weight_scale.numel() == out_features
        else weight_scale.expand(out_features).contiguous()
    )
    extension = cuda_backend._C
    wrap = cuda_backend._wrap_for_dlpack

    def current_heuristic() -> torch.Tensor:
        return comfy_kitchen.int8_linear(
            raw,
            qweight,
            weight_scale,
            None,
            torch.bfloat16,
            convrot=True,
            convrot_groupsize=256,
            input_act="swiglu",
        )

    def forced_config0() -> torch.Tensor:
        qx, x_scale = cuda_backend.quantize_int8_rowwise_convrot64(
            raw, 256, input_act="swiglu"
        )
        output = torch.empty(
            (args.rows, out_features), dtype=torch.bfloat16, device="cuda:0"
        )
        supported = extension.cutlass_int8_dequant_config(
            wrap(qx),
            wrap(qweight),
            wrap(x_scale),
            wrap(expanded_scale),
            wrap(output),
            2,
            0,
            torch.cuda.current_stream().cuda_stream,
        )
        if not supported:
            raise RuntimeError("CUTLASS config 0 does not support the H3 FC2 shape")
        return output

    with comfy_kitchen.use_backend("cuda"):
        for _ in range(args.warmup):
            current_heuristic()
            forced_config0()
        torch.cuda.synchronize()

        current_output = current_heuristic()
        candidate_output = forced_config0()
        torch.cuda.synchronize()
        delta = candidate_output.float().sub_(current_output.float()).abs_()
        numerical = {
            "max_abs": float(delta.max()),
            "mean_abs": float(delta.mean()),
            "exact_equal": bool(torch.equal(current_output, candidate_output)),
        }
        del current_output, candidate_output, delta

        current_samples: list[float] = []
        candidate_samples: list[float] = []
        for iteration in range(args.iterations):
            # Reverse order every iteration to reduce temperature/order bias.
            if iteration % 2:
                candidate_samples.append(elapsed_ms(forced_config0))
                current_samples.append(elapsed_ms(current_heuristic))
            else:
                current_samples.append(elapsed_ms(current_heuristic))
                candidate_samples.append(elapsed_ms(forced_config0))

    current = summarize(current_samples)
    candidate = summarize(candidate_samples)
    report = {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "checkpoint": str(args.checkpoint.resolve()),
        "operation": "H3 FC2 fused SwiGLU + ConvRot quant + INT8 GEMM/dequant",
        "shape": [args.rows, out_features, inner],
        "current_production_config": args.current_config,
        "candidate_config": 0,
        "runtime": runtime.to_dict(),
        "numerical": numerical,
        "current": current,
        "candidate": candidate,
        "speedup": current["median_ms"] / candidate["median_ms"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
