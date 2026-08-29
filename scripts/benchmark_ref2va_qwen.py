#!/usr/bin/env python3
"""Profile the real Ref2VA Qwen3-VL prefix without loading DiT or VAEs."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from types import SimpleNamespace


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "benchmarks/ref2va_extreme/ref2va_multiref_8s_enhanced_v1.json",
    )
    parser.add_argument("--cache-pinned-weights", action="store_true")
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="override the packed Qwen checkpoint for storage-path A/B",
    )
    parser.add_argument("--disable-vision-cache", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--second-prompt-suffix",
        default="",
        help="append this text from run two onward while reusing the references",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch

    from h3serve.native_engine.adapters.conditioning_vae import (
        PackedQwen3VLT2AVConditioner,
    )

    root = Path(__file__).resolve().parents[1]
    document = json.loads(args.manifest.read_text(encoding="utf-8"))
    scenario = document["scenarios"][0]
    base = args.manifest.resolve().parents[2]

    def paths(key: str) -> tuple[Path, ...]:
        return tuple((base / value).resolve() for value in scenario.get(key, ()))

    request = SimpleNamespace(
        prompt=scenario["prompt"],
        reference_images=paths("reference_images"),
        reference_videos=paths("reference_videos"),
        reference_audios=paths("reference_audios"),
        first_frame=None,
        last_frame=None,
        frames=int(scenario["frames"]),
        num_frames=int(scenario["frames"]),
        width=int(scenario["width"]),
        height=int(scenario["height"]),
        fps=24,
        prepared_reference_images=(),
        prepared_reference_videos=(),
        prepared_reference_audios=(),
    )
    conditioner = PackedQwen3VLT2AVConditioner(
        args.checkpoint or root / "models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        root.parents[1] / "MiniMax-H3/tokenizer",
        cache_pinned_weights=args.cache_pinned_weights,
        cache_vision_features=not args.disable_vision_cache,
        processor_path=root.parents[1] / "MiniMax-H3/processor",
        model_config_path=root.parents[1] / "MiniMax-H3/text_encoder/config.json",
    )
    phases: dict[str, float] = {}

    def timed(name: str, operation):
        started = time.perf_counter()
        result = operation()
        phases[name] = time.perf_counter() - started
        print(f"{name}: {phases[name]:.3f}s", flush=True)
        return result

    if args.cache_pinned_weights:
        timed("startup_host_cache", conditioner.prepare_host_cache)
    runs = []
    original_prompt = request.prompt
    for index in range(args.repeat):
        request.prompt = (
            original_prompt
            if index == 0
            else original_prompt + args.second_prompt_suffix
        )
        prepared = timed(
            f"run{index + 1}_multimodal_prepare_and_vision",
            lambda: conditioner._prepare_multimodal(request),
        )
        if prepared is None:
            raise RuntimeError("expected a multimodal Ref2VA presentation")
        ids, tags, position_ids, vision_mask, vision_embeds, deepstack, keyframe_count = prepared
        result = timed(
            f"run{index + 1}_qwen_50_layers",
            lambda: conditioner._encode_tokens(
                ids,
                tags,
                position_ids=position_ids,
                vision_mask=vision_mask,
                vision_embeds=vision_embeds,
                deepstack=deepstack,
                keyframe_count=keyframe_count,
            ),
        )
        runs.append({
            "embedding_checksum": {
                "sum": float(result.prompt_embeds.double().sum().item()),
                "abs_sum": float(result.prompt_embeds.double().abs().sum().item()),
            },
            "vision_cache_hits": conditioner.vision_cache_hits,
            "vision_cache_misses": conditioner.vision_cache_misses,
            "prompt_changed": request.prompt != original_prompt,
        })
    report = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "cache_pinned_weights": args.cache_pinned_weights,
        "cache_vision_features": not args.disable_vision_cache,
        "phases": phases,
        "token_count": result.token_count,
        "qwen_reported_seconds": result.elapsed_seconds,
        "qwen_peak_allocated_gib": result.peak_allocated_gib,
        "host_cache_gib": conditioner.host_cache_bytes / (1024**3),
        "vision_feature_cache_mib": conditioner.vision_feature_cache_bytes / (1024**2),
        "process_max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / (1024**3),
        "embedding_shape": list(result.prompt_embeds.shape),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
