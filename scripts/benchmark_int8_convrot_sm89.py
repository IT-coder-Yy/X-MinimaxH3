#!/usr/bin/env python3
"""Compare pinned Comfy-Kitchen CUDA/Triton on exact H3 linear shapes."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from h3serve.native_engine.sm89_policy import configure_sm89_runtime


SHAPES = (
    ("qkv", 5376, 21504, None),
    ("attention_out", 7168, 5376, None),
    ("mlp_fc1", 5376, 28672, None),
    ("mlp_fc2_swiglu", 28672, 5376, "swiglu"),
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, action="append", default=[])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=9)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "runtime/calibration/int8_convrot_sm89.json",
    )
    args = parser.parse_args()
    args.rows = tuple(args.rows or (2048, 8192))
    if any(value <= 0 for value in args.rows) or args.warmup < 1 or args.repeat < 3:
        parser.error("rows/warmup/repeat must be positive; repeat must be >=3")
    return args


def elapsed_ms(operation) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    result = operation()
    stop.record()
    stop.synchronize()
    value = start.elapsed_time(stop)
    del result, start, stop
    return value


def main() -> int:
    args = parse_args()
    configure_sm89_runtime(quant_backend="cuda", smoke_test=True)
    import comfy_kitchen

    generator = torch.Generator("cuda:0").manual_seed(4090)
    report = {"schema_version": 1, "gpu": torch.cuda.get_device_name(), "runs": []}
    for rows in args.rows:
        for name, input_width, output_width, input_act in SHAPES:
            value = torch.randn(
                (rows, input_width),
                device="cuda:0",
                dtype=torch.bfloat16,
                generator=generator,
            )
            effective_input = input_width // 2 if input_act == "swiglu" else input_width
            qweight = torch.randint(
                -127,
                128,
                (output_width, effective_input),
                device="cuda:0",
                dtype=torch.int8,
                generator=generator,
            )
            scale = torch.rand(
                (output_width, 1),
                device="cuda:0",
                dtype=torch.float32,
                generator=generator,
            ) * 0.02

            def invoke():
                return comfy_kitchen.int8_linear(
                    value,
                    qweight,
                    scale,
                    None,
                    torch.bfloat16,
                    convrot=True,
                    convrot_groupsize=256,
                    input_act=input_act,
                )

            outputs = {}
            for backend in ("cuda", "triton"):
                with comfy_kitchen.use_backend(backend):
                    for _ in range(args.warmup):
                        invoke()
                    torch.cuda.synchronize()
                    timings = [elapsed_ms(invoke) for _ in range(args.repeat)]
                    outputs[backend] = invoke().detach()
                    torch.cuda.synchronize()
                report["runs"].append(
                    {
                        "rows": rows,
                        "operation": name,
                        "backend": backend,
                        "median_ms": statistics.median(timings),
                        "min_ms": min(timings),
                        "max_ms": max(timings),
                    }
                )
            delta = (outputs["cuda"].float() - outputs["triton"].float()).abs()
            comparison = {
                "rows": rows,
                "operation": name,
                "backend": "comparison",
                "max_abs": float(delta.max()),
                "mean_abs": float(delta.mean()),
                "cosine": float(
                    torch.nn.functional.cosine_similarity(
                        outputs["cuda"].float().flatten(),
                        outputs["triton"].float().flatten(),
                        dim=0,
                    )
                ),
            }
            report["runs"].append(comparison)
            print(json.dumps(report["runs"][-3:], ensure_ascii=False), flush=True)
            del value, qweight, scale, outputs, delta
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
