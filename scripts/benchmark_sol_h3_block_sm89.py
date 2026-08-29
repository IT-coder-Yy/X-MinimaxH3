#!/usr/bin/env python3
"""Compare official Sol-Attn with the current H3 long-sequence backend.

This is an experimental evidence script, not a production dependency.  It
loads one real INT8 ConvRot H3 block at the 720p/15s packed shape, preserves the
entire text/condition/audio prefix as exact KV for every query, and overwrites
the prefix-query output with the accepted dense Sage path.  That boundary is
the minimum meaningful comparison for H3's joint audio-video stream.
"""

from __future__ import annotations

import argparse
import json
import site
import statistics
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "models/diffusion_models/"
            "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
        ),
    )
    parser.add_argument(
        "--sparge-build-dir",
        type=Path,
        default=Path("runtime/extensions/sparge-sm89-py310-torch28-cu12"),
    )
    parser.add_argument(
        "--sol-source",
        type=Path,
        default=Path("../../DIT-knowledge/sources/projects/sana_sol"),
    )
    parser.add_argument(
        "--sol-deps",
        type=Path,
        default=Path("runtime/extensions/sol-attn-deps"),
    )
    parser.add_argument("--block", type=int, default=20)
    parser.add_argument("--topk", type=float, default=0.50)
    parser.add_argument("--text-length", type=int, default=517)
    parser.add_argument("--latent-frames", type=int, default=107)
    parser.add_argument("--latent-height", type=int, default=46)
    parser.add_argument("--latent-width", type=int, default=80)
    parser.add_argument("--audio-frames", type=int, default=603)
    parser.add_argument("--expected-tokens", type=int, default=100163)
    parser.add_argument("--taus", default="0.8,1.0,1.3")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runtime/calibration/workload_routing_round27/"
            "official_sol_real_block20.json"
        ),
    )
    return parser.parse_args()


def measure(operation, reset, *, warmup: int, repeat: int) -> tuple[list[float], int]:
    for _ in range(warmup):
        reset()
        operation()
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples: list[float] = []
    for _ in range(repeat):
        reset()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    return samples, int(torch.cuda.max_memory_allocated())


class OfficialSolH3Backend:
    """Official SM89 Sol kernel with H3's sensitive prefix protected."""

    approximate = True

    def __init__(self, *, tau: float, protected_tokens: int, dense_backend) -> None:
        self.tau = float(tau)
        self.protected_tokens = int(protected_tokens)
        self.dense_backend = dense_backend

    def __call__(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        from sol_attn import sol_attn

        q = query.unsqueeze(0).contiguous().to(torch.bfloat16)
        k = key.unsqueeze(0).contiguous().to(torch.bfloat16)
        v = value.unsqueeze(0).contiguous().to(torch.bfloat16)
        output = sol_attn(
            q,
            k,
            v,
            tau=self.tau,
            thresh_type="diag",
            sink_start=0,
            sink_tokens=self.protected_tokens,
        ).squeeze(0)
        # Sol's sink protects prefix K/V for every query. H3 also needs the
        # audio/text prefix query rows themselves to remain dense.
        output[: self.protected_tokens].copy_(
            self.dense_backend(
                query[: self.protected_tokens], key, value
            )
        )
        return output


def error_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    delta = (reference.float() - candidate.float()).abs()
    return {
        "mean_abs": float(delta.mean().cpu()),
        "max_abs": float(delta.max().cpu()),
        "rmse": float(delta.square().mean().sqrt().cpu()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                reference.float().reshape(1, -1),
                candidate.float().reshape(1, -1),
            ).cpu()
        ),
    }


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this benchmark requires one RTX 4090 / SM89 GPU")
    if not 0 <= args.block < 50:
        raise SystemExit("--block must lie inside [0, 50)")
    try:
        taus = tuple(float(item) for item in args.taus.split(",") if item.strip())
    except ValueError as error:
        raise SystemExit("--taus must be comma-separated numbers") from error
    if not taus or any(value < 0.0 for value in taus):
        raise SystemExit("--taus must contain non-negative values")
    site.addsitedir(str(args.sol_deps.resolve()))
    sys.path.insert(0, str(args.sol_source.resolve()))
    sys.path.insert(
        0, str((args.sol_source / "techniques" / "sparse_backends").resolve())
    )
    sys.path.insert(0, str(args.sparge_build_dir.resolve()))

    from sol_attn import get_sol_attn_backend

    from h3serve.native_engine.model import (
        SafeTensorSource,
        assemble_pruned_block,
        build_fl2va_layout,
        comfy_kitchen_int8_kernel,
        make_split_modality_protected_sparge_attention_sm89,
        sage_attention_sm89,
    )
    from h3serve.native_engine.model.dit import FullH3DiT
    from h3serve.native_engine.model.kernels import (
        attention_layer,
        attention_protected_prefix,
    )
    from h3serve.native_engine.model.layers import rope_frequencies, rope_rotation_table
    from h3serve.native_engine.model.lora import AdaLNCurveRows, interpolate_curve
    from h3serve.native_engine.sm89_policy import configure_sm89_runtime

    configure_sm89_runtime(quant_backend="cuda", smoke_test=True)
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    layout = build_fl2va_layout(
        text_length=args.text_length,
        latent_frames=args.latent_frames,
        latent_height=args.latent_height,
        latent_width=args.latent_width,
        audio_frames=args.audio_frames,
    )
    if layout.sequence_length != args.expected_tokens:
        raise RuntimeError(f"unexpected probe length: {layout.sequence_length}")
    protected = layout.segment("video", last=True).start
    sigma = torch.tensor([0.5], device=device)
    unique_timesteps, segments, _ = FullH3DiT._timestep_plan(
        sigma,
        layout,
        sigma_shift_video=5.0,
        sigma_shift_audio=2.0,
        visual_condition_timestep=0.999,
        audio_condition_timestep=1.0,
        text_token_tags=None,
        device=device,
    )
    split_backend = make_split_modality_protected_sparge_attention_sm89(args.topk)
    with SafeTensorSource(str(args.checkpoint)) as source:
        block = assemble_pruned_block(
            args.block,
            source,
            device=device,
            compute_dtype=torch.bfloat16,
            int8_kernel=comfy_kitchen_int8_kernel,
            attention_backend=split_backend,
        )
        curve_rows = AdaLNCurveRows(
            compressed=interpolate_curve(
                source.tensor("adaln_t_table").to(device), unique_timesteps
            )
        )
        frequencies = rope_rotation_table(
            rope_frequencies(
                layout.position_ids.to(device), source.tensor("rope.inv_freq").to(device)
            ),
            torch.bfloat16,
        )
    block.eval().requires_grad_(False)

    base = torch.randn(
        layout.sequence_length, 5376, device=device, dtype=torch.bfloat16
    )
    working = torch.empty_like(base)

    def reset() -> None:
        working.copy_(base)

    def run(block) -> torch.Tensor:
        with attention_protected_prefix(protected), attention_layer(args.block):
            return block(
                working,
                timestep_rows=curve_rows,
                modulation_segments=segments,
                frequencies=frequencies,
                mlp_chunk_tokens=None,
            )

    reset()
    block.attention.backend = sage_attention_sm89
    reset()
    dense_reference = run(block).clone()
    dense_samples, dense_peak = measure(
        lambda: run(block), reset, warmup=args.warmup, repeat=args.repeat
    )
    dense_ms = statistics.median(dense_samples)

    block.attention.backend = split_backend
    reset()
    reference = run(block).clone()
    split_samples, split_peak = measure(
        lambda: run(block), reset, warmup=args.warmup, repeat=args.repeat
    )
    split_ms = statistics.median(split_samples)
    report = {
        "shape": {
            "tokens": layout.sequence_length,
            "protected_tokens": protected,
            "video_tokens": layout.sequence_length - protected,
            "heads": 56,
            "head_dim": 128,
            "text_length": args.text_length,
            "latent_frames": args.latent_frames,
            "latent_height": args.latent_height,
            "latent_width": args.latent_width,
            "audio_frames": args.audio_frames,
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "block": args.block,
        "official_sol_source": {
            "path": str(args.sol_source.resolve()),
            "backend": get_sol_attn_backend(device),
        },
        "reference": {
            "backend": "split-modality-sparge",
            "topk": args.topk,
            "samples_ms": split_samples,
            "median_ms": split_ms,
            "peak_allocated_gib": split_peak / (1024**3),
            "speedup_vs_dense": dense_ms / split_ms,
            "full_output_vs_dense": error_stats(dense_reference, reference),
        },
        "dense": {
            "backend": "sage-attention-sm89",
            "samples_ms": dense_samples,
            "median_ms": dense_ms,
            "peak_allocated_gib": dense_peak / (1024**3),
        },
        "candidates": {},
    }
    for tau in taus:
        block.attention.backend = OfficialSolH3Backend(
            tau=tau,
            protected_tokens=protected,
            dense_backend=sage_attention_sm89,
        )
        reset()
        candidate = run(block).clone()
        samples, peak = measure(
            lambda block=block: run(block),
            reset,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        median_ms = statistics.median(samples)
        report["candidates"][str(tau)] = {
            "samples_ms": samples,
            "median_ms": median_ms,
            "speedup_vs_split": split_ms / median_ms,
            "peak_allocated_gib": peak / (1024**3),
            "speedup_vs_dense": dense_ms / median_ms,
            "full_output_vs_dense": error_stats(dense_reference, candidate),
            "full_output_vs_split": error_stats(reference, candidate),
            "prefix_output_vs_split": error_stats(
                reference[:protected], candidate[:protected]
            ),
            "video_output_vs_split": error_stats(
                reference[protected:], candidate[protected:]
            ),
        }
        del candidate
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    del dense_reference, reference, block, base, working, frequencies, curve_rows
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
