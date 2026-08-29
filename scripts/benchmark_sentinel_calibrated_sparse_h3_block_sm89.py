#!/usr/bin/env python3
"""Probe dense-sentinel error feedback for aggressive H3 sparse attention."""

from __future__ import annotations

import argparse
import json
import site
import statistics
import sys
from pathlib import Path

import torch

from scripts.benchmark_cdf_budget_h3_block_sm89 import error_stats, measure


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
    parser.add_argument("--block", type=int, default=20)
    parser.add_argument("--aggressive-topks", default="0.35,0.40")
    parser.add_argument("--sentinel-blocks", default="0,4,8,16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runtime/calibration/workload_routing_round29/"
            "sentinel_calibrated_real_block20.json"
        ),
    )
    return parser.parse_args()


class SentinelCalibratedSplitBackend:
    """Aggressive sparse video queries plus online dense-sentinel correction."""

    approximate = True

    def __init__(
        self,
        *,
        topk: float,
        sentinel_blocks: int,
        protected_tokens: int,
        production_backend,
    ) -> None:
        self.topk = float(topk)
        self.sentinel_blocks = int(sentinel_blocks)
        self.protected_tokens = int(protected_tokens)
        self.production = production_backend
        self.last_retained_fraction: torch.Tensor | None = None

    @staticmethod
    def _sentinel_indices(
        video_tokens: int, block_count: int, device: torch.device
    ) -> torch.Tensor:
        query_blocks = (video_tokens + 127) // 128
        block_count = min(block_count, query_blocks)
        blocks = torch.linspace(
            0, query_blocks - 1, steps=block_count, device=device
        ).round().long().unique(sorted=True)
        pieces = [
            torch.arange(
                int(block) * 128,
                min((int(block) + 1) * 128, video_tokens),
                device=device,
                dtype=torch.long,
            )
            for block in blocks
        ]
        return torch.cat(pieces)

    def __call__(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        from einops import rearrange
        from spas_sage_attn import core as sparge_core
        from spas_sage_attn.utils import (
            block_map_lut_triton,
            get_block_map_meansim_fuse_quant,
            hyperparameter_check,
        )

        protected = self.protected_tokens
        k, v_fp8, v_scale, heads, _kv_len, head_dim = self.production._prepare_kv(
            key, value
        )
        prefix = self.production._dense_prefix(
            query[:protected], k, v_fp8, v_scale, head_dim=head_dim
        )
        video_query = query[protected:]
        q = rearrange(video_query, "L H D -> 1 H L D").contiguous().to(
            torch.bfloat16
        )
        key_mean = k.mean(dim=-2, keepdim=True)
        block_map, q8, qs, k8, ks = get_block_map_meansim_fuse_quant(
            q,
            k,
            key_mean,
            BLKQ=128,
            BLKK=64,
            simthreshd1=-0.1,
            cdfthreshd=None,
            topk=self.topk,
            is_causal=False,
        )
        block_map[:, :, :, : (protected + 63) // 64] = True
        self.last_retained_fraction = block_map.float().mean()
        lut, valid = block_map_lut_triton(block_map.contiguous())
        sparse_hnd = torch.empty_like(q)
        pv_threshold = hyperparameter_check(50, heads, query.device)
        sparge_core.qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
            q8,
            k8,
            v_fp8,
            sparse_hnd,
            lut,
            valid,
            pv_threshold,
            qs,
            ks,
            v_scale,
            1,
            0,
            1,
            1.0 / (head_dim**0.5),
            0,
        )
        video = rearrange(sparse_hnd, "1 H L D -> L H D")
        if self.sentinel_blocks:
            indices = self._sentinel_indices(
                video.shape[0], self.sentinel_blocks, video.device
            )
            dense_sentinel = self.production._dense_prefix(
                video_query.index_select(0, indices),
                k,
                v_fp8,
                v_scale,
                head_dim=head_dim,
            )
            sparse_sentinel = video.index_select(0, indices)
            delta = dense_sentinel.float() - sparse_sentinel.float()
            sentinel_query_blocks = (indices // 128).unique(sorted=True)
            corrections: list[torch.Tensor] = []
            offset = 0
            for query_block in sentinel_query_blocks:
                rows = min(
                    128,
                    video.shape[0] - int(query_block) * 128,
                )
                corrections.append(delta[offset : offset + rows].mean(dim=0))
                offset += rows
            total_query_blocks = (video.shape[0] + 127) // 128
            positions = [int(item) for item in sentinel_query_blocks]
            boundaries = [0]
            boundaries.extend(
                (left + right + 1) // 2
                for left, right in zip(positions, positions[1:])
            )
            boundaries.append(total_query_blocks)
            for start_block, stop_block, correction in zip(
                boundaries, boundaries[1:], corrections
            ):
                video[
                    start_block * 128 : min(stop_block * 128, video.shape[0])
                ].add_(correction.to(video.dtype))
        return torch.cat((prefix, video), dim=0)


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this benchmark requires one RTX 4090 / SM89 GPU")
    try:
        topks = tuple(float(item) for item in args.aggressive_topks.split(","))
        sentinel_counts = tuple(int(item) for item in args.sentinel_blocks.split(","))
    except ValueError as error:
        raise SystemExit("topks and sentinel blocks must be comma-separated numbers") from error
    if any(not 0.1 <= item < 0.5 for item in topks):
        raise SystemExit("aggressive topks must lie inside [0.1, 0.5)")
    if any(item < 0 for item in sentinel_counts):
        raise SystemExit("sentinel block counts cannot be negative")

    site.addsitedir(str(args.sparge_build_dir.resolve()))
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
    split = make_split_modality_protected_sparge_attention_sm89(0.50)
    with SafeTensorSource(str(args.checkpoint)) as source:
        block = assemble_pruned_block(
            args.block,
            source,
            device=device,
            compute_dtype=torch.bfloat16,
            int8_kernel=comfy_kitchen_int8_kernel,
            attention_backend=split,
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

    def run() -> torch.Tensor:
        with attention_protected_prefix(protected), attention_layer(args.block):
            return block(
                working,
                timestep_rows=curve_rows,
                modulation_segments=segments,
                frequencies=frequencies,
                mlp_chunk_tokens=None,
            )

    block.attention.backend = sage_attention_sm89
    reset()
    dense = run().clone()
    dense_samples, dense_peak = measure(
        run, reset, warmup=args.warmup, repeat=args.repeat
    )
    dense_ms = statistics.median(dense_samples)
    block.attention.backend = split
    reset()
    accepted = run().clone()
    split_samples, split_peak = measure(
        run, reset, warmup=args.warmup, repeat=args.repeat
    )
    split_ms = statistics.median(split_samples)

    candidates = []
    for topk in topks:
        for sentinel_count in sentinel_counts:
            candidate = SentinelCalibratedSplitBackend(
                topk=topk,
                sentinel_blocks=sentinel_count,
                protected_tokens=protected,
                production_backend=split,
            )
            block.attention.backend = candidate
            reset()
            output = run().clone()
            retained = float(candidate.last_retained_fraction.cpu())
            samples, peak = measure(
                run, reset, warmup=args.warmup, repeat=args.repeat
            )
            median_ms = statistics.median(samples)
            candidates.append(
                {
                    "topk": topk,
                    "sentinel_blocks": sentinel_count,
                    "correction": "nearest_sentinel_query_region_head_channel_mean",
                    "sentinel_query_fraction": (
                        min(sentinel_count * 128, layout.sequence_length - protected)
                        / (layout.sequence_length - protected)
                    ),
                    "retained_block_fraction": retained,
                    "samples_ms": samples,
                    "median_ms": median_ms,
                    "speedup_vs_dense": dense_ms / median_ms,
                    "speedup_vs_split050": split_ms / median_ms,
                    "peak_allocated_gib": peak / (1024**3),
                    "error_vs_dense": error_stats(dense, output),
                    "error_vs_split050": error_stats(accepted, output),
                }
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
        "dense": {
            "median_ms": dense_ms,
            "samples_ms": dense_samples,
            "peak_allocated_gib": dense_peak / (1024**3),
        },
        "accepted_split050": {
            "median_ms": split_ms,
            "samples_ms": split_samples,
            "speedup_vs_dense": dense_ms / split_ms,
            "peak_allocated_gib": split_peak / (1024**3),
            "error_vs_dense": error_stats(dense, accepted),
        },
        "candidates": candidates,
        "boundary": "Single real-weight block with synthetic hidden input; no quality claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
