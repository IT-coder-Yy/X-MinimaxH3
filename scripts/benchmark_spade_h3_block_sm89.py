#!/usr/bin/env python3
"""Benchmark SPADE's SM89 block-sparse kernel inside one real H3 block.

This is a kernel-applicability experiment, not a production backend.  It uses
the upstream SPADE/block_sparse_attn extension, pads H3's packed AV sequence to
the kernel block grid, keeps the whole text/condition/audio prefix as KV
anchors, and restores prefix-query rows with the accepted dense Sage path.
The static diagonal masks deliberately test execution economics only; an
end-to-end candidate must use an H3-calibrated 3D/head-wise selector.
"""

from __future__ import annotations

import argparse
import json
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
        "--spade-source",
        type=Path,
        default=Path("../../DIT-knowledge/sources/projects/dac_spade"),
    )
    parser.add_argument("--block", type=int, default=20)
    parser.add_argument("--topk", type=float, default=0.50)
    parser.add_argument("--keep-ratios", default="0.25,0.35,0.50")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runtime/calibration/workload_routing_round27/"
            "spade_real_block20.json"
        ),
    )
    return parser.parse_args()


def measure(operation, reset, *, warmup: int, repeat: int) -> tuple[list[float], int]:
    for _ in range(warmup):
        reset()
        operation()
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
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


class SpadeStaticH3Backend:
    approximate = True

    def __init__(
        self,
        *,
        keep_ratio: float,
        protected_tokens: int,
        dense_backend,
    ) -> None:
        self.keep_ratio = float(keep_ratio)
        self.protected_tokens = int(protected_tokens)
        self.dense_backend = dense_backend
        self._mask = None
        self._head_types = None

    def _build_mask(self, *, tokens: int, heads: int, device: torch.device):
        q_blocks = (tokens + 127) // 128
        k_blocks = (tokens + 63) // 64
        prefix_k = min(k_blocks, (self.protected_tokens + 63) // 64)
        keep = max(prefix_k, round(k_blocks * self.keep_ratio))
        local_keep = max(0, keep - prefix_k)
        mask = torch.zeros(
            (1, heads, q_blocks, k_blocks), device=device, dtype=torch.bool
        )
        if prefix_k:
            mask[..., :prefix_k] = True
        if local_keep:
            video_k = k_blocks - prefix_k
            offsets = torch.arange(local_keep, device=device)
            # Each 128-query block spans two 64-key blocks.  Head-dependent
            # phase shifts avoid making all heads observe an identical band.
            centers = 2 * torch.arange(q_blocks, device=device)
            half = local_keep // 2
            for head in range(heads):
                phase = (head * max(1, local_keep // 8)) % max(1, video_k)
                indices = (
                    centers[:, None] - half + offsets[None, :] + phase
                ) % max(1, video_k)
                mask[0, head].scatter_(
                    1, indices.add(prefix_k), True
                )
        return mask, torch.ones(heads, device=device, dtype=torch.int32)

    def __call__(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        from block_sparse_attn import block_sparse_attn_func_bnsh

        tokens, heads, dim = query.shape
        padded_tokens = (tokens + 127) // 128 * 128
        pad = padded_tokens - tokens
        if self._mask is None:
            self._mask, self._head_types = self._build_mask(
                tokens=tokens, heads=heads, device=query.device
            )
        q = query.movedim(1, 0).unsqueeze(0).contiguous()
        k = key.movedim(1, 0).unsqueeze(0).contiguous()
        v = value.movedim(1, 0).unsqueeze(0).contiguous()
        if pad:
            q = torch.nn.functional.pad(q, (0, 0, 0, pad))
            k = torch.nn.functional.pad(k, (0, 0, 0, pad))
            v = torch.nn.functional.pad(v, (0, 0, 0, pad))
        output = block_sparse_attn_func_bnsh(
            q,
            k,
            v,
            head_mask_type=self._head_types,
            base_blockmask=self._mask,
            m_block_dim=128,
            n_block_dim=64,
        )[:, :, :tokens].squeeze(0).movedim(0, 1).contiguous()
        output[: self.protected_tokens].copy_(
            self.dense_backend(query[: self.protected_tokens], key, value)
        )
        return output


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this benchmark requires one RTX 4090 / SM89 GPU")
    ratios = tuple(float(item) for item in args.keep_ratios.split(",") if item)
    if not ratios or any(not 0.0 < item <= 1.0 for item in ratios):
        raise SystemExit("--keep-ratios must contain values inside (0, 1]")
    sys.path.insert(0, str((args.spade_source / "python").resolve()))
    sys.path.insert(0, str(args.sparge_build_dir.resolve()))

    from h3serve.native_engine.model import (
        SafeTensorSource,
        assemble_pruned_block,
        build_fl2va_layout,
        comfy_kitchen_int8_kernel,
        make_split_modality_protected_sparge_attention_sm89,
        sage_attention_sm89,
    )
    from h3serve.native_engine.model.dit import FullH3DiT
    from h3serve.native_engine.model.kernels import attention_layer, attention_protected_prefix
    from h3serve.native_engine.model.layers import rope_frequencies, rope_rotation_table
    from h3serve.native_engine.model.lora import AdaLNCurveRows, interpolate_curve
    from h3serve.native_engine.sm89_policy import configure_sm89_runtime

    configure_sm89_runtime(quant_backend="cuda", smoke_test=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    layout = build_fl2va_layout(
        text_length=517,
        latent_frames=107,
        latent_height=46,
        latent_width=80,
        audio_frames=603,
    )
    protected = layout.segment("video", last=True).start
    unique_timesteps, segments, _ = FullH3DiT._timestep_plan(
        torch.tensor([0.5], device=device),
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
            attention_backend=sage_attention_sm89,
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
    base = torch.randn(layout.sequence_length, 5376, device=device, dtype=torch.bfloat16)
    working = torch.empty_like(base)

    def reset():
        working.copy_(base)

    def run():
        with attention_protected_prefix(protected), attention_layer(args.block):
            return block(
                working,
                timestep_rows=curve_rows,
                modulation_segments=segments,
                frequencies=frequencies,
                mlp_chunk_tokens=None,
            )

    report = {
        "shape": {
            "tokens": layout.sequence_length,
            "protected_tokens": protected,
            "heads": 56,
            "head_dim": 128,
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "block": args.block,
        "spade_source": str(args.spade_source.resolve()),
        "mask_note": "static prefix-anchor plus head-shifted diagonal band; kernel economics only",
        "candidates": {},
    }
    block.attention.backend = sage_attention_sm89
    reset()
    dense_output = run().clone()
    samples, peak = measure(run, reset, warmup=args.warmup, repeat=args.repeat)
    dense_ms = statistics.median(samples)
    report["dense"] = {
        "samples_ms": samples,
        "median_ms": dense_ms,
        "peak_allocated_gib": peak / 1024**3,
    }

    block.attention.backend = split_backend
    reset()
    split_output = run().clone()
    samples, peak = measure(run, reset, warmup=args.warmup, repeat=args.repeat)
    split_ms = statistics.median(samples)
    report["split"] = {
        "topk": args.topk,
        "samples_ms": samples,
        "median_ms": split_ms,
        "speedup_vs_dense": dense_ms / split_ms,
        "peak_allocated_gib": peak / 1024**3,
        "output_vs_dense": error_stats(dense_output, split_output),
    }

    for ratio in ratios:
        block.attention.backend = SpadeStaticH3Backend(
            keep_ratio=ratio,
            protected_tokens=protected,
            dense_backend=sage_attention_sm89,
        )
        reset()
        candidate = run().clone()
        samples, peak = measure(run, reset, warmup=args.warmup, repeat=args.repeat)
        median_ms = statistics.median(samples)
        report["candidates"][str(ratio)] = {
            "samples_ms": samples,
            "median_ms": median_ms,
            "speedup_vs_dense": dense_ms / median_ms,
            "speedup_vs_split": split_ms / median_ms,
            "peak_allocated_gib": peak / 1024**3,
            "output_vs_dense": error_stats(dense_output, candidate),
            "output_vs_split": error_stats(split_output, candidate),
        }
        del candidate

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    del dense_output, split_output, block, base, working, frequencies, curve_rows
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
