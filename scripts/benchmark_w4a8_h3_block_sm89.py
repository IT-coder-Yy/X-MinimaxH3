#!/usr/bin/env python3
"""Real-H3 Block probe for runtime W4A8 projection/MLP execution.

The source checkpoint remains the accepted ConvRot INT8 model.  This probe
derives a temporary W4A8 execution cache from the physical rotated weights and
compares one complete long-sequence H3 block before any full-model conversion.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn


class W4A8Linear(nn.Module):
    def __init__(self, tensors, *, group_size: int, convrot_group_size: int) -> None:
        super().__init__()
        qdata, s_rel, s_channel, correction, codebook = tensors
        self.register_buffer("qdata", qdata)
        self.register_buffer("s_rel", s_rel)
        self.register_buffer("s_channel", s_channel)
        self.register_buffer("correction", correction)
        self.register_buffer("codebook", codebook)
        self.group_size = group_size
        self.convrot_group_size = convrot_group_size

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        from comfy_kitchen import w4a8_int8_linear

        return w4a8_int8_linear(
            value,
            self.qdata,
            self.s_rel,
            self.s_channel,
            codebook=self.codebook,
            correction=self.correction,
            group_size=self.group_size,
            convrot_groupsize=self.convrot_group_size,
            out_dtype=value.dtype,
        )


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
    parser.add_argument("--step", type=int, default=11)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runtime/calibration/workload_routing_round79/"
            "w4a8_real_h3_block.json"
        ),
    )
    return parser.parse_args()


def measure(operation, reset, repeat: int) -> list[float]:
    reset()
    operation()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeat):
        reset()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    return samples


def error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.float()
    cand = candidate.float()
    delta = ref - cand
    return {
        "relative_l1": float(delta.abs().mean() / ref.abs().mean().clamp_min(1e-8)),
        "rmse": float(delta.square().mean().sqrt()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                ref.reshape(1, -1), cand.reshape(1, -1)
            )
        ),
    }


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("requires RTX 4090 / SM89")
    sys.path.insert(0, str(args.sparge_build_dir.resolve()))
    sys.path.insert(0, str(Path("backends/turbo/vendor").resolve()))
    from comfy_kitchen.backends.eager.w4a8_int8 import (
        _quantize_rotated_w4a8_int8_weight,
    )
    from h3serve.native_engine.model import (
        SafeTensorSource,
        SplitModalityProtectedSpargeAttentionBackend,
        assemble_pruned_block,
        build_fl2va_layout,
        comfy_kitchen_int8_kernel,
    )
    from h3serve.native_engine.model.dit import FullH3DiT
    from h3serve.native_engine.model.kernels import (
        attention_layer,
        attention_protected_prefix,
        attention_step,
        attention_video_layout,
    )
    from h3serve.native_engine.model.layers import rope_frequencies, rope_rotation_table
    from h3serve.native_engine.model.lora import AdaLNCurveRows, interpolate_curve
    from h3serve.native_engine.model.quantization import ConvRotInt8Linear
    from h3serve.native_engine.sm89_policy import configure_sm89_runtime

    configure_sm89_runtime(quant_backend="cuda", smoke_test=True)
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
    attention = SplitModalityProtectedSpargeAttentionBackend(
        0.10,
        experimental_minimum_topk=0.0625,
        temporal_correspondence_radius=1,
        temporal_spatial_block_radius=1,
        temporal_global_anchor_stride=8,
    )
    with SafeTensorSource(str(args.checkpoint)) as source:
        block = assemble_pruned_block(
            args.block,
            source,
            device=device,
            compute_dtype=torch.bfloat16,
            int8_kernel=comfy_kitchen_int8_kernel,
            attention_backend=attention,
        ).eval()
        # Match the production hot-session contract.  The fused in-place
        # QK-normalization/RoPE kernel correctly rejects trainable norm
        # parameters even when the caller happens to be in inference_mode.
        block.requires_grad_(False)
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

    torch.manual_seed(4090)
    base = torch.randn(
        layout.sequence_length, 5376, device=device, dtype=torch.bfloat16
    )
    working = torch.empty_like(base)

    def reset() -> None:
        working.copy_(base)

    def run() -> torch.Tensor:
        with (
            attention_protected_prefix(protected),
            attention_video_layout(107, 920),
            attention_step(args.step, 20),
            attention_layer(args.block),
        ):
            return block(
                working,
                timestep_rows=curve_rows,
                modulation_segments=segments,
                frequencies=frequencies,
                mlp_chunk_tokens=None,
            )

    reset()
    reference = run().index_select(
        0, torch.linspace(0, layout.sequence_length - 1, 2048, device=device).long()
    ).cpu()
    w8_samples = measure(run, reset, args.repeat)

    converted: dict[str, float] = {}
    for name, parent, attribute in (
        ("qkv", block.attention, "qkv_proj"),
        ("out", block.attention, "out_proj"),
        ("fc1", block.mlp, "fc1"),
        ("fc2", block.mlp, "fc2"),
    ):
        layer = getattr(parent, attribute)
        if not isinstance(layer, ConvRotInt8Linear):
            raise TypeError(f"{name} is not ConvRot INT8")
        started = time.perf_counter()
        rotated = layer.qweight.cpu().float()
        rotated.mul_(layer.scale.cpu().float().reshape(-1, 1))
        tensors = _quantize_rotated_w4a8_int8_weight(
            rotated,
            group_size=16,
            symmetric=True,
            scale_dtype=torch.float8_e4m3fn,
            codebook=True,
        )
        tensors = tuple(None if item is None else item.to(device) for item in tensors)
        setattr(
            parent,
            attribute,
            W4A8Linear(
                tensors,
                group_size=16,
                convrot_group_size=layer.spec.convrot_groupsize,
            ),
        )
        converted[name] = time.perf_counter() - started
        del layer, rotated, tensors
        torch.cuda.empty_cache()

    reset()
    candidate = run().index_select(
        0, torch.linspace(0, layout.sequence_length - 1, 2048, device=device).long()
    ).cpu()
    w4_samples = measure(run, reset, args.repeat)
    report = {
        "tokens": layout.sequence_length,
        "block": args.block,
        "step": args.step,
        "source_weights": "pruned_int8_convrot",
        "candidate_execution": "derived_symmetric_codebook_w4a8_group16",
        "conversion_seconds": converted,
        "w8a8_ms": w8_samples,
        "w4a8_ms": w4_samples,
        "median_speedup": statistics.median(w8_samples) / statistics.median(w4_samples),
        "sampled_output_error": error(reference, candidate),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
