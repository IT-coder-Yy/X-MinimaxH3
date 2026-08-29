#!/usr/bin/env python3
"""Autotune existing exact CUTLASS INT8 epilogues on H3 MLP shapes.

This benchmark starts after ConvRot row quantization.  It compares only the
integer GEMM plus scale/dequant epilogue, where all candidates implement the
same arithmetic contract.  The purpose is to check whether Comfy-Kitchen's
generic shape heuristic selects the best already-compiled tile on RTX 4090.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from h3serve.native_engine.sm89_policy import configure_sm89_runtime


SHAPES = (
    ("mlp_fc1", 8192, 28672, 5376),
    ("mlp_fc2", 8192, 5376, 14336),
    ("mlp_fc1_720p15_tail", 1859, 28672, 5376),
    ("mlp_fc2_720p15_tail", 1859, 5376, 14336),
    ("mlp_fc1_720p5_tail", 2103, 28672, 5376),
    ("mlp_fc2_720p5_tail", 2103, 5376, 14336),
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "runtime/calibration/cutlass_h3_sm89_autotune.json",
    )
    args = parser.parse_args()
    if args.iterations < 3:
        parser.error("iterations must be >= 3")
    return args


def main() -> int:
    args = parse_args()
    configure_sm89_runtime(quant_backend="cuda", smoke_test=True)
    import comfy_kitchen.backends.cuda as cuda_backend

    extension = cuda_backend._C
    wrap = cuda_backend._wrap_for_dlpack
    stream = torch.cuda.current_stream().cuda_stream
    generator = torch.Generator("cuda:0").manual_seed(4090)
    report: dict[str, object] = {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(),
        "iterations": args.iterations,
        "runs": [],
    }
    for name, rows, out_features, inner in SHAPES:
        activation = torch.randint(
            -127,
            128,
            (rows, inner),
            dtype=torch.int8,
            device="cuda:0",
            generator=generator,
        )
        weight = torch.randint(
            -127,
            128,
            (out_features, inner),
            dtype=torch.int8,
            device="cuda:0",
            generator=generator,
        )
        activation_scale = torch.rand(
            rows, dtype=torch.float32, device="cuda:0", generator=generator
        )
        weight_scale = torch.rand(
            out_features, dtype=torch.float32, device="cuda:0", generator=generator
        )
        output = torch.empty(
            (rows, out_features), dtype=torch.bfloat16, device="cuda:0"
        )
        reference = None
        shape_runs = []
        for config in range(15):
            elapsed = extension.benchmark_cutlass_int8_dequant_config(
                wrap(activation),
                wrap(weight),
                wrap(activation_scale),
                wrap(weight_scale),
                wrap(output),
                2,
                config,
                args.iterations,
                stream,
            )
            if elapsed < 0:
                shape_runs.append({"config": config, "supported": False})
                continue
            torch.cuda.synchronize()
            extension.cutlass_int8_dequant_config(
                wrap(activation),
                wrap(weight),
                wrap(activation_scale),
                wrap(weight_scale),
                wrap(output),
                2,
                config,
                stream,
            )
            torch.cuda.synchronize()
            candidate = output.clone()
            if reference is None:
                reference = candidate
            delta = candidate.float().sub_(reference.float()).abs_()
            shape_runs.append(
                {
                    "config": config,
                    "supported": True,
                    "mean_ms": elapsed / args.iterations,
                    "max_abs_vs_config0": float(delta.max()),
                    "mean_abs_vs_config0": float(delta.mean()),
                }
            )
            del candidate, delta
        supported = [entry for entry in shape_runs if entry["supported"]]
        fastest = min(supported, key=lambda entry: entry["mean_ms"])
        result = {
            "operation": name,
            "shape": [rows, out_features, inner],
            "fastest_config": fastest["config"],
            "fastest_mean_ms": fastest["mean_ms"],
            "configs": shape_runs,
        }
        report["runs"].append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        del activation, weight, activation_scale, weight_scale, output, reference
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
