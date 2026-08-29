#!/usr/bin/env python3
"""Compare fixed Top-K with per-query CDF budgets on one real H3 block."""

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
    parser.add_argument("--block", type=int, default=20)
    parser.add_argument("--topk", type=float, default=0.50)
    parser.add_argument("--cdf-thresholds", default="0.80,0.90,0.95")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runtime/calibration/workload_routing_round28/"
            "cdf_budget_real_block20.json"
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


class CDFSplitBackend:
    """Production split boundary with a per-query cumulative-mass budget."""

    approximate = True

    def __init__(
        self, threshold: float, production_backend, *, protected_tokens: int
    ) -> None:
        self.threshold = float(threshold)
        self.production = production_backend
        self.protected_tokens = int(protected_tokens)
        self.last_retained_fraction: torch.Tensor | None = None

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
        if protected <= 0 or protected >= query.shape[0]:
            raise RuntimeError("CDF probe requires the packed H3 protected prefix")
        k, v_fp8, v_scale, heads, _kv_len, head_dim = self.production._prepare_kv(
            key, value
        )
        prefix_output = self.production._dense_prefix(
            query[:protected], k, v_fp8, v_scale, head_dim=head_dim
        )
        q = rearrange(
            query[protected:], "L H D -> 1 H L D"
        ).contiguous().to(torch.bfloat16)
        key_mean = k.mean(dim=-2, keepdim=True)
        block_map, q8, qs, k8, ks = get_block_map_meansim_fuse_quant(
            q,
            k,
            key_mean,
            BLKQ=128,
            BLKK=64,
            simthreshd1=-0.1,
            cdfthreshd=self.threshold,
            topk=None,
            is_causal=False,
        )
        protected_k_blocks = (protected + 63) // 64
        block_map[:, :, :, :protected_k_blocks] = True
        self.last_retained_fraction = block_map.float().mean()
        lut, valid = block_map_lut_triton(block_map.contiguous())
        video_output = torch.empty_like(q)
        pv_threshold = hyperparameter_check(50, heads, query.device)
        sparge_core.qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
            q8,
            k8,
            v_fp8,
            video_output,
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
        return torch.cat(
            (prefix_output, rearrange(video_output, "1 H L D -> L H D")), dim=0
        )


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this benchmark requires one RTX 4090 / SM89 GPU")
    try:
        thresholds = tuple(
            float(item) for item in args.cdf_thresholds.split(",") if item.strip()
        )
    except ValueError as error:
        raise SystemExit("--cdf-thresholds must be comma-separated floats") from error
    if not thresholds or any(not 0.0 < item < 1.0 for item in thresholds):
        raise SystemExit("CDF thresholds must lie inside (0, 1)")

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
    split = make_split_modality_protected_sparge_attention_sm89(args.topk)
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

    results: dict[str, object] = {}
    block.attention.backend = sage_attention_sm89
    reset()
    dense_output = run().clone()
    samples, peak = measure(run, reset, warmup=args.warmup, repeat=args.repeat)
    dense_ms = statistics.median(samples)
    results["dense"] = {
        "samples_ms": samples,
        "median_ms": dense_ms,
        "peak_allocated_gib": peak / (1024**3),
    }

    block.attention.backend = split
    reset()
    split_output = run().clone()
    samples, peak = measure(run, reset, warmup=args.warmup, repeat=args.repeat)
    split_ms = statistics.median(samples)
    results["fixed_topk"] = {
        "topk": args.topk,
        "samples_ms": samples,
        "median_ms": split_ms,
        "speedup_vs_dense": dense_ms / split_ms,
        "peak_allocated_gib": peak / (1024**3),
        "error_vs_dense": error_stats(dense_output, split_output),
    }

    cdf_results = []
    for threshold in thresholds:
        candidate = CDFSplitBackend(
            threshold, split, protected_tokens=protected
        )
        block.attention.backend = candidate
        reset()
        output = run().clone()
        retained = float(candidate.last_retained_fraction.cpu())
        samples, peak = measure(run, reset, warmup=args.warmup, repeat=args.repeat)
        median_ms = statistics.median(samples)
        cdf_results.append(
            {
                "cdf_threshold": threshold,
                "retained_block_fraction": retained,
                "samples_ms": samples,
                "median_ms": median_ms,
                "speedup_vs_dense": dense_ms / median_ms,
                "speedup_vs_fixed_topk": split_ms / median_ms,
                "peak_allocated_gib": peak / (1024**3),
                "error_vs_dense": error_stats(dense_output, output),
                "error_vs_fixed_topk": error_stats(split_output, output),
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
        "results": results,
        "cdf_candidates": cdf_results,
        "boundary": "Single real-weight block with synthetic hidden input; no quality claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
