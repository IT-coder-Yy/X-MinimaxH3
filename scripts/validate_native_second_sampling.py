#!/usr/bin/env python3
"""Run one real source-generation -> native H3 second-sampling gate.

This is deliberately a product-boundary check: it uses NativeBackendManager,
persists the first pass's clean AV latent, and submits the second pass through
the same method used by the Web/API job service.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import shutil
import time
from pathlib import Path

from h3serve.backend import build_native_backend
from h3serve.config import ServicePaths
from h3serve.contract import GenerationSpec, SecondSamplingSpec
from h3serve.memory_policy import HOST_MEMORY_PROFILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--launcher",
        default="fl2va_int8_24gb",
        choices=(
            "fl2va_int8_24gb", "ref2va_int8_24gb",
            "fl2va_int8_16gb", "ref2va_int8_16gb",
            "fl2va_w4a8_8gb", "ref2va_w4a8_8gb",
        ),
    )
    parser.add_argument("--source-resolution", default="480p")
    parser.add_argument(
        "--source-latents",
        type=Path,
        help="Reuse a retained clean H3 AV checkpoint and skip first generation.",
    )
    parser.add_argument("--target-resolution", default="720p")
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--duration-seconds", type=float, default=5.0)
    parser.add_argument("--source-steps", type=int, default=5)
    parser.add_argument("--source-acceleration", type=float, default=95.0)
    parser.add_argument("--second-steps", type=int, default=1)
    parser.add_argument("--second-acceleration", type=float, default=75.0)
    parser.add_argument("--denoise", type=float, default=0.20)
    parser.add_argument(
        "--repeat-second-sampling",
        type=int,
        default=1,
        help="Run the target pass repeatedly in one hot backend to verify cleanup/reuse.",
    )
    parser.add_argument("--reference-image", action="append", type=Path, default=[])
    parser.add_argument("--reference-audio", action="append", type=Path, default=[])
    parser.add_argument("--memory-mode", choices=("auto", "performance", "low_vram"), default="auto")
    parser.add_argument("--seed", type=int, default=82704)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


PROMPT = (
    "A locked-off documentary shot inside a quiet restoration workshop. "
    "A conservator in blue gloves slowly lifts a small brass key from a tray, "
    "turns it once under a desk lamp, and says in natural Mandarin: "
    "“编号确认，表面没有新的裂纹。” The key remains rigid and unchanged. "
    "Background shelves, tools, and door stay completely still. "
    "Natural room tone, no music."
)


def generation_spec(args: argparse.Namespace, resolution: str) -> GenerationSpec:
    family = "reference" if args.launcher.startswith("ref2va") else "first_last"
    return GenerationSpec.from_mapping({
        "prompt": PROMPT,
        "runtime_launcher": args.launcher,
        "service_family": family,
        "model_variant": "base",
        "quality": "balanced",
        "resolution": resolution,
        "aspect_ratio": args.aspect_ratio,
        "duration_seconds": args.duration_seconds,
        "seed": args.seed,
        "sampling_steps": args.source_steps,
        "acceleration": args.source_acceleration,
        "memory_mode": args.memory_mode,
    })


async def run(args: argparse.Namespace) -> dict[str, object]:
    if args.repeat_second_sampling < 1:
        raise ValueError("repeat-second-sampling must be at least one")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = ServicePaths.defaults(args.release_root)
    paths = type(paths)(**{**paths.__dict__, "output_dir": args.output_dir.resolve()})
    backend = build_native_backend(paths, memory_profile=HOST_MEMORY_PROFILES["fullspeed"])
    reference_images = tuple(path.resolve() for path in args.reference_image)
    reference_audios = tuple(path.resolve() for path in args.reference_audio)
    for path in (*reference_images, *reference_audios):
        if not path.is_file():
            raise FileNotFoundError(path)
    cancel = asyncio.Event()
    started = time.perf_counter()
    try:
        await backend.preload(args.launcher)
        warm_state = backend.warm_state
        source = generation_spec(args, args.source_resolution)
        source_result = None
        if args.source_latents is None:
            source_result = await backend.generate(
                source,
                "second_sampling_source",
                None,
                None,
                reference_images,
                (),
                reference_audios,
                cancel,
            )
            if source_result.final_latents_path is None:
                raise RuntimeError("source generation did not retain a clean AV latent")
            source_latents = source_result.final_latents_path
        else:
            external_source = args.source_latents.resolve()
            if not external_source.is_file():
                raise FileNotFoundError(external_source)
            latent_store = (args.output_dir / ".h3-latents").resolve()
            latent_store.mkdir(parents=True, exist_ok=True)
            source_latents = latent_store / "reused-source.pt"
            shutil.copy2(external_source, source_latents)
        second = SecondSamplingSpec.from_mapping({
            "resolution": args.target_resolution,
            "steps": args.second_steps,
            "acceleration": args.second_acceleration,
            "denoise": args.denoise,
            "memory_mode": args.memory_mode,
        }, source=source)
        target = dataclasses.replace(
            source,
            resolution=second.resolution,
            width=second.width,
            height=second.height,
            advanced=True,
            custom_actual_steps=20,
            sampling_steps=None,
            acceleration=None,
        )
        second_runs = []
        for index in range(args.repeat_second_sampling):
            job_id = (
                "second_sampling_target"
                if args.repeat_second_sampling == 1
                else f"second_sampling_target_{index + 1}"
            )
            second_result = await backend.second_sample(
                target,
                second,
                source_latents,
                job_id,
                None,
                None,
                reference_images,
                (),
                reference_audios,
                cancel,
            )
            second_runs.append({
                "index": index + 1,
                "elapsed_seconds": second_result.elapsed_seconds,
                "output_path": str(second_result.output_path),
                "latents_path": str(second_result.final_latents_path),
                "inference_plan": second_result.inference_plan,
            })
        return {
            "schema_version": "h3_native_second_sampling_e2e_v1",
            "warm_state": warm_state,
            "wall_seconds": time.perf_counter() - started,
            "source": {
                "request": source.to_dict(),
                "reused": args.source_latents is not None,
                "elapsed_seconds": (
                    None if source_result is None else source_result.elapsed_seconds
                ),
                "output_path": (
                    None if source_result is None else str(source_result.output_path)
                ),
                "latents_path": str(source_latents),
                "inference_plan": (
                    None if source_result is None else source_result.inference_plan
                ),
            },
            "second_sampling": {
                "request": second.to_dict(),
                "elapsed_seconds": second_result.elapsed_seconds,
                "output_path": str(second_result.output_path),
                "latents_path": str(second_result.final_latents_path),
                "inference_plan": second_result.inference_plan,
                "reference_images": [str(path) for path in reference_images],
                "reference_audios": [str(path) for path in reference_audios],
                "hot_runs": second_runs,
            },
        }
    finally:
        await backend.stop()


def main() -> None:
    args = parse_args()
    report = asyncio.run(run(args))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    second = report["second_sampling"]
    plan = second["inference_plan"] or {}
    runtime_memory = plan.get("runtime_memory") or {}
    print(json.dumps({
        "status": "passed",
        "report": str(args.report.resolve()),
        "wall_seconds": report["wall_seconds"],
        "output_path": second["output_path"],
        "second_sampling_seconds": second["elapsed_seconds"],
        "hot_run_seconds": [run["elapsed_seconds"] for run in second["hot_runs"]],
        "peak_allocated_gib": runtime_memory.get("peak_allocated_gib"),
        "peak_reserved_gib": runtime_memory.get("peak_reserved_gib"),
        "reference_images": second["reference_images"],
        "reference_audios": second["reference_audios"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
