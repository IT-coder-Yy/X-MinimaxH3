#!/usr/bin/env python3
"""Record whether one real H3 block fits the 1080p/15s packed shape.

Unlike the synthetic attention probe, this includes the 5376-wide residual,
AdaLN/RMS, fused INT8 QKV/out projections and chunked fused MLP.  Each backend
is attempted once and OOM is retained as a first-class result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch

from scripts.profile_h3_1080p15_attention_memory import GIB, NvidiaSmiSampler


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
    parser.add_argument("--block", type=int, default=20)
    parser.add_argument("--text-length", type=int, default=517)
    parser.add_argument("--latent-frames", type=int, default=107)
    parser.add_argument("--latent-height", type=int, default=68)
    parser.add_argument("--latent-width", type=int, default=120)
    parser.add_argument("--audio-frames", type=int, default=603)
    parser.add_argument("--reference-image-count", type=int, default=0)
    parser.add_argument("--reference-image-latent-height", type=int, default=68)
    parser.add_argument("--reference-image-latent-width", type=int, default=120)
    parser.add_argument("--reference-audio-count", type=int, default=0)
    parser.add_argument("--reference-audio-frames", type=int, default=603)
    parser.add_argument("--expected-tokens", type=int, default=220_003)
    parser.add_argument("--mlp-chunk-tokens", type=int, default=8192)
    parser.add_argument("--query-chunk-tokens", type=int, default=32768)
    parser.add_argument("--projection-chunk-tokens", type=int, default=8192)
    parser.add_argument("--split-qkv-outputs", action="store_true")
    parser.add_argument("--single-qknorm-rope", action="store_true")
    parser.add_argument("--parallel-sparse-lut", action="store_true")
    parser.add_argument("--partial-sparse-topk", action="store_true")
    parser.add_argument("--fused-prefix-k-quant", action="store_true")
    parser.add_argument("--fused-query-projection", action="store_true")
    parser.add_argument("--fused-qknorm-hnd-layout", action="store_true")
    parser.add_argument("--direct-nhd-output", action="store_true")
    parser.add_argument("--direct-nhd-kv", action="store_true")
    parser.add_argument("--topk", type=float, default=0.25)
    parser.add_argument(
        "--dense-qk-quant-gran",
        choices=("per_thread", "per_warp"),
        default="per_warp",
    )
    parser.add_argument(
        "--absolute-cap-reference-key-blocks", type=int, default=1565
    )
    parser.add_argument("--absolute-cap-multiplier", type=float, default=1.75)
    parser.add_argument("--minimum-retained-topk-mass", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--equivalence-sample-rows", type=int, default=2048)
    parser.add_argument(
        "--full-output-sha256",
        action="store_true",
        help=(
            "hash every BF16 output byte after timing; adds D2H/hash overhead "
            "outside the reported CUDA interval"
        ),
    )
    parser.add_argument("--hash-chunk-rows", type=int, default=4096)
    parser.add_argument(
        "--kineto-dir",
        type=Path,
        help="optional directory for one post-timing block trace and kernel table",
    )
    parser.add_argument(
        "--stages",
        default="dense,streamed_dense,full_sparse,streamed_sparse",
        help=(
            "dense, streamed_dense, full_sparse, streamed_sparse, "
            "streamed_sparse_split_qkv, streamed_sparse_cap and/or "
            "streamed_sparse_mass_guarded.  The bracketed same-process A/B "
            "uses streamed_sparse_split_qkv_reference_start, "
            "streamed_sparse_split_qkv_candidate and "
            "streamed_sparse_split_qkv_reference_end.  The exact execution "
            "ladder uses full_sparse_reference_start, "
            "streamed_sparse_reference, streamed_sparse_split_qkv_reference, "
            "streamed_sparse_split_qkv_single_qknorm, "
            "streamed_sparse_split_qkv_single_qknorm_parallel_lut and "
            "streamed_sparse_compact_kv, streamed_dense_compact_kv and "
            "full_sparse_reference_end."
        ),
    )
    parser.add_argument(
        "--sparge-build-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "H3_NATIVE_SPARGE_BUILD_DIR",
                "runtime/extensions/sparge-sm89-py310-torch213-cu133",
            )
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runtime/calibration/long_1080p15_20260824/real_block20_probe.json"
        ),
    )
    return parser.parse_args()


def tensor_sha256(value: torch.Tensor, *, chunk_rows: int = 4096) -> str:
    """Hash every logical tensor byte without retaining a second full output."""

    if value.ndim < 1 or value.shape[0] <= 0:
        raise ValueError("full-output hash requires at least one tensor row")
    if chunk_rows <= 0:
        raise ValueError("hash chunk rows must be positive")
    digest = hashlib.sha256()
    for start in range(0, int(value.shape[0]), chunk_rows):
        raw = (
            value[start : start + chunk_rows]
            .detach()
            .contiguous()
            .view(torch.uint8)
            .cpu()
            .numpy()
            .tobytes()
        )
        digest.update(raw)
    return digest.hexdigest()


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if args.hash_chunk_rows <= 0:
        raise SystemExit("--hash-chunk-rows must be positive")
    stages = tuple(item.strip() for item in args.stages.split(",") if item.strip())
    if not stages or any(
        item
        not in (
            "dense",
            "streamed_dense",
            "full_sparse",
            "streamed_sparse",
            "streamed_sparse_split_qkv",
            "streamed_sparse_split_qkv_single_qknorm",
            "streamed_sparse_split_qkv_reference_start",
            "streamed_sparse_split_qkv_candidate",
            "streamed_sparse_split_qkv_reference_end",
            "full_sparse_reference_start",
            "streamed_sparse_reference",
            "streamed_sparse_split_qkv_reference",
            "streamed_sparse_split_qkv_single_qknorm_parallel_lut",
            "streamed_sparse_split_qkv_single_qknorm_helpers_reference_start",
            "streamed_sparse_split_qkv_single_qknorm_helpers_fused_query_projection",
            "streamed_sparse_split_qkv_single_qknorm_helpers_reference_end",
            "streamed_release_stack_reference_start",
            "streamed_release_stack_fused_qknorm_hnd",
            "streamed_release_stack_reference_end",
            "streamed_sparse_compact_kv",
            "full_sparse_reference_end",
            "dense_reference_start",
            "streamed_dense_reference",
            "streamed_dense_split_qkv",
            "streamed_dense_split_qkv_single_qknorm",
            "streamed_dense_compact_kv",
            "streamed_joint_dense_split_qkv_single_qknorm",
            "dense_reference_end",
            "streamed_sparse_cap",
            "streamed_sparse_mass_guarded",
        )
        for item in stages
    ):
        raise SystemExit(
            "--stages contains an unsupported real-block probe"
        )
    if args.query_chunk_tokens < 128 or args.query_chunk_tokens % 128:
        raise SystemExit("--query-chunk-tokens must be a positive multiple of 128")
    if args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("--warmup must be non-negative and --repeat positive")
    if args.equivalence_sample_rows <= 0:
        raise SystemExit("--equivalence-sample-rows must be positive")
    if args.projection_chunk_tokens <= 0:
        raise SystemExit("--projection-chunk-tokens must be positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this probe requires one RTX 4090 / SM89 GPU")
    if not 0 <= args.block < 50:
        raise SystemExit("--block must lie inside [0, 50)")
    if args.absolute_cap_reference_key_blocks <= 0:
        raise SystemExit("--absolute-cap-reference-key-blocks must be positive")
    if not 1.0 <= args.absolute_cap_multiplier <= 4.0:
        raise SystemExit("--absolute-cap-multiplier must lie inside [1, 4]")
    if not 0.0 < args.minimum_retained_topk_mass <= 1.0:
        raise SystemExit("--minimum-retained-topk-mass must lie inside (0, 1]")
    build_dir = args.sparge_build_dir.resolve()
    if not build_dir.is_dir():
        raise SystemExit(f"missing Sparge build: {build_dir}")
    sys.path.insert(0, str(build_dir))

    from h3serve.native_engine.model import (
        SafeTensorSource,
        SplitModalityProtectedSpargeAttentionBackend,
        attention_action_schedule,
        assemble_pruned_block,
        build_fl2va_layout,
        build_ref2va_layout,
        comfy_kitchen_int8_kernel,
        make_joint_action_scheduled_sparge_attention_sm89,
        sage_attention_sm89,
    )
    from h3serve.native_engine.model.dit import FullH3DiT
    from h3serve.native_engine.model.kernels import (
        attention_layer,
        attention_protected_prefix,
        attention_step,
        attention_video_layout,
        dense_qk_quantization,
        long_sequence_query_chunking,
        rms_adaln_fusion,
    )
    from h3serve.native_engine.model.layers import rope_frequencies, rope_rotation_table
    from h3serve.native_engine.model.lora import AdaLNCurveRows, interpolate_curve
    from h3serve.native_engine.sm89_policy import configure_sm89_runtime

    configure_sm89_runtime(quant_backend="cuda", smoke_test=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    if not 0 <= args.reference_image_count <= 9:
        raise SystemExit("--reference-image-count must lie inside [0, 9]")
    if not 0 <= args.reference_audio_count <= 3:
        raise SystemExit("--reference-audio-count must lie inside [0, 3]")
    has_references = bool(
        args.reference_image_count or args.reference_audio_count
    )
    if has_references:
        reference_shapes = (
            (
                1,
                args.reference_image_latent_height,
                args.reference_image_latent_width,
            ),
        ) * args.reference_image_count
        reference_audio_frames = (
            args.reference_audio_frames,
        ) * args.reference_audio_count
        layout = build_ref2va_layout(
            text_length=args.text_length,
            latent_frames=args.latent_frames,
            latent_height=args.latent_height,
            latent_width=args.latent_width,
            audio_frames=args.audio_frames,
            reference_shapes=reference_shapes,
            reference_kinds=("image",) * args.reference_image_count,
            reference_audio_frames=reference_audio_frames,
        )
    else:
        reference_shapes = ()
        reference_audio_frames = ()
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
    frame_tokens = (args.latent_height // 2) * (args.latent_width // 2)
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
    sparse = SplitModalityProtectedSpargeAttentionBackend(
        args.topk,
        experimental_minimum_topk=0.0625,
        temporal_correspondence_radius=1,
        temporal_spatial_block_radius=1,
        temporal_global_anchor_stride=8,
        parallel_long_sequence_lut=args.parallel_sparse_lut,
        partial_long_sequence_topk=args.partial_sparse_topk,
        fused_long_sequence_prefix_k_quant=args.fused_prefix_k_quant,
    )
    # The bracketed A/B deliberately ignores the global experimental flags.
    # Both sides share the same process, checkpoint, hidden state, allocator
    # history and split-QKV execution.  Only the three candidate mechanics
    # below differ, and the trailing reference exposes timing drift.
    sparse_reference = SplitModalityProtectedSpargeAttentionBackend(
        args.topk,
        experimental_minimum_topk=0.0625,
        temporal_correspondence_radius=1,
        temporal_spatial_block_radius=1,
        temporal_global_anchor_stride=8,
    )
    sparse_parallel_lut = SplitModalityProtectedSpargeAttentionBackend(
        args.topk,
        experimental_minimum_topk=0.0625,
        temporal_correspondence_radius=1,
        temporal_spatial_block_radius=1,
        temporal_global_anchor_stride=8,
        parallel_long_sequence_lut=True,
    )
    sparse_candidate = SplitModalityProtectedSpargeAttentionBackend(
        args.topk,
        experimental_minimum_topk=0.0625,
        temporal_correspondence_radius=1,
        temporal_spatial_block_radius=1,
        temporal_global_anchor_stride=8,
        parallel_long_sequence_lut=True,
        partial_long_sequence_topk=True,
        fused_long_sequence_prefix_k_quant=True,
    )
    reference_selected_blocks = math.floor(
        args.topk * args.absolute_cap_reference_key_blocks
    )
    absolute_cap = math.ceil(
        reference_selected_blocks * args.absolute_cap_multiplier
    )
    sparse_absolute_cap = SplitModalityProtectedSpargeAttentionBackend(
        args.topk,
        experimental_minimum_topk=0.0625,
        temporal_correspondence_radius=1,
        temporal_spatial_block_radius=1,
        temporal_global_anchor_stride=8,
        selection_mode="fixed_topk_absolute_cap",
        maximum_selected_key_blocks=absolute_cap,
        parallel_long_sequence_lut=args.parallel_sparse_lut,
        partial_long_sequence_topk=args.partial_sparse_topk,
        fused_long_sequence_prefix_k_quant=args.fused_prefix_k_quant,
    )
    sparse_mass_guarded = SplitModalityProtectedSpargeAttentionBackend(
        args.topk,
        experimental_minimum_topk=0.0625,
        temporal_correspondence_radius=1,
        temporal_spatial_block_radius=1,
        temporal_global_anchor_stride=8,
        selection_mode="fixed_topk_mass_guarded_cap",
        maximum_selected_key_blocks=absolute_cap,
        minimum_retained_topk_mass=args.minimum_retained_topk_mass,
        parallel_long_sequence_lut=args.parallel_sparse_lut,
        partial_long_sequence_topk=args.partial_sparse_topk,
        fused_long_sequence_prefix_k_quant=args.fused_prefix_k_quant,
    )
    joint_scheduled = make_joint_action_scheduled_sparge_attention_sm89()
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
                layout.position_ids.to(device),
                source.tensor("rope.inv_freq").to(device),
            ),
            torch.bfloat16,
        )
    block.eval().requires_grad_(False)
    base = torch.empty(
        layout.sequence_length, 5376, device=device, dtype=torch.bfloat16
    ).normal_().clamp_(-4, 4)
    working = torch.empty_like(base)
    torch.cuda.synchronize()

    def run(
        *,
        query_chunk_tokens: int | None = None,
        split_qkv_outputs: bool = False,
        compact_kv: bool = False,
        single_qknorm_rope: bool = False,
        fused_query_projection: bool = False,
        fused_qknorm_hnd_layout: bool = False,
        direct_nhd_output: bool = False,
        direct_nhd_kv: bool = False,
    ) -> torch.Tensor:
        working.copy_(base)
        with (
            attention_protected_prefix(protected),
            attention_video_layout(args.latent_frames, frame_tokens),
            attention_step(10, 20),
            attention_layer(args.block),
            attention_action_schedule({(10, args.block): "dense"}),
            dense_qk_quantization(args.dense_qk_quant_gran),
            rms_adaln_fusion(True),
            long_sequence_query_chunking(
                query_chunk_tokens,
                projection_chunk_tokens=args.projection_chunk_tokens,
                split_qkv_outputs=split_qkv_outputs,
                compact_kv=compact_kv,
                single_qknorm_rope=single_qknorm_rope,
                fused_query_projection=fused_query_projection,
                fused_qknorm_hnd_layout=fused_qknorm_hnd_layout,
                direct_nhd_output=direct_nhd_output,
                direct_nhd_kv=direct_nhd_kv,
            ),
        ):
            return block(
                working,
                timestep_rows=curve_rows,
                modulation_segments=segments,
                frequencies=frequencies,
                mlp_chunk_tokens=args.mlp_chunk_tokens,
            )

    report: dict[str, object] = {
        "schema_version": "h3_1080p15_real_block_memory_probe_v2",
        "warning": "One synthetic-hidden-state block probe; no video-quality claim.",
        "runtime": {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_total_gib": torch.cuda.get_device_properties(device).total_memory / GIB,
            "checkpoint": str(args.checkpoint.resolve()),
            "sparge_build_dir": str(build_dir),
        },
        "shape": {
            "tokens": layout.sequence_length,
            "protected_tokens": protected,
            "video_tokens": layout.sequence_length - protected,
            "latent_frames": args.latent_frames,
            "latent_height": args.latent_height,
            "latent_width": args.latent_width,
            "frame_tokens": frame_tokens,
            "block": args.block,
            "mlp_chunk_tokens": args.mlp_chunk_tokens,
            "query_chunk_tokens": args.query_chunk_tokens,
            "projection_chunk_tokens": args.projection_chunk_tokens,
            "split_qkv_outputs": args.split_qkv_outputs,
            "single_qknorm_rope": args.single_qknorm_rope,
            "parallel_sparse_lut": args.parallel_sparse_lut,
            "partial_sparse_topk": args.partial_sparse_topk,
            "fused_prefix_k_quant": args.fused_prefix_k_quant,
            "fused_query_projection": args.fused_query_projection,
            "fused_qknorm_hnd_layout": args.fused_qknorm_hnd_layout,
            "direct_nhd_output": args.direct_nhd_output,
            "direct_nhd_kv": args.direct_nhd_kv,
            "equivalence_sample_rows": args.equivalence_sample_rows,
            "full_output_sha256": args.full_output_sha256,
            "hash_chunk_rows": args.hash_chunk_rows,
            "topk": args.topk,
            "dense_qk_quant_gran": args.dense_qk_quant_gran,
            "absolute_cap_reference_key_blocks": (
                args.absolute_cap_reference_key_blocks
            ),
            "absolute_cap_multiplier": args.absolute_cap_multiplier,
            "absolute_cap_selected_blocks": absolute_cap,
            "minimum_retained_topk_mass": args.minimum_retained_topk_mass,
            "layout_family": "ref2va" if has_references else "fl2va",
            "reference_image_count": args.reference_image_count,
            "reference_audio_count": args.reference_audio_count,
        },
        "baseline_allocated_gib": torch.cuda.memory_allocated(device) / GIB,
        "stages": {},
        "sampled_equivalence": {},
        "full_output_equivalence": {},
        "comparisons": {},
    }

    baseline_allocated = int(torch.cuda.memory_allocated(device))
    sample_indices = torch.linspace(
        0,
        layout.sequence_length - 1,
        min(args.equivalence_sample_rows, layout.sequence_length),
        device=device,
    ).round().to(torch.long).unique()
    reference_stage: str | None = None
    reference_sample: torch.Tensor | None = None
    reference_output_sha256: str | None = None
    for stage_name in stages:
        stage_backends = {
            "dense": sage_attention_sm89,
            "streamed_dense": sage_attention_sm89,
            "full_sparse": sparse,
            "streamed_sparse": sparse,
            "streamed_sparse_split_qkv": sparse,
            "streamed_sparse_split_qkv_single_qknorm": sparse_reference,
            "streamed_sparse_split_qkv_reference_start": sparse_reference,
            "streamed_sparse_split_qkv_candidate": sparse_candidate,
            "streamed_sparse_split_qkv_reference_end": sparse_reference,
            "full_sparse_reference_start": sparse_reference,
            "streamed_sparse_reference": sparse_reference,
            "streamed_sparse_split_qkv_reference": sparse_reference,
            "streamed_sparse_split_qkv_single_qknorm_parallel_lut": (
                sparse_parallel_lut
            ),
            "streamed_sparse_split_qkv_single_qknorm_helpers_reference_start": sparse_candidate,
            "streamed_sparse_split_qkv_single_qknorm_helpers_fused_query_projection": sparse_candidate,
            "streamed_sparse_split_qkv_single_qknorm_helpers_reference_end": sparse_candidate,
            "streamed_release_stack_reference_start": sparse_candidate,
            "streamed_release_stack_fused_qknorm_hnd": sparse_candidate,
            "streamed_release_stack_reference_end": sparse_candidate,
            "streamed_sparse_compact_kv": sparse_parallel_lut,
            "full_sparse_reference_end": sparse_reference,
            "dense_reference_start": sage_attention_sm89,
            "streamed_dense_reference": sage_attention_sm89,
            "streamed_dense_split_qkv": sage_attention_sm89,
            "streamed_dense_split_qkv_single_qknorm": sage_attention_sm89,
            "streamed_dense_compact_kv": sage_attention_sm89,
            "streamed_joint_dense_split_qkv_single_qknorm": joint_scheduled,
            "dense_reference_end": sage_attention_sm89,
            "streamed_sparse_cap": sparse_absolute_cap,
            "streamed_sparse_mass_guarded": sparse_mass_guarded,
        }
        block.attention.backend = stage_backends[stage_name]
        stage_split_qkv = (
            args.split_qkv_outputs
            or "split_qkv" in stage_name
            or "compact_kv" in stage_name
        )
        stage_compact_kv = "compact_kv" in stage_name
        stage_single_qknorm = (
            stage_name.startswith("streamed_sparse_split_qkv_single_qknorm")
            or stage_name == "streamed_dense_split_qkv_single_qknorm"
            or stage_name == "streamed_joint_dense_split_qkv_single_qknorm"
            or stage_name.startswith("streamed_release_stack_")
            or stage_compact_kv
            or (
                args.single_qknorm_rope
                and stage_name == "streamed_sparse_split_qkv_candidate"
            )
        )
        stage_fused_query_projection = (
            args.fused_query_projection
            or stage_name
            == "streamed_sparse_split_qkv_single_qknorm_helpers_fused_query_projection"
            or stage_name.startswith("streamed_release_stack_")
        )
        stage_fused_qknorm_hnd_layout = (
            args.fused_qknorm_hnd_layout
            or stage_name == "streamed_release_stack_fused_qknorm_hnd"
        )
        stage_backend_flags = {
            "parallel_sparse_lut": bool(
                getattr(block.attention.backend, "parallel_long_sequence_lut", False)
            ),
            "partial_sparse_topk": bool(
                getattr(block.attention.backend, "partial_long_sequence_topk", False)
            ),
            "fused_prefix_k_quant": bool(
                getattr(
                    block.attention.backend,
                    "fused_long_sequence_prefix_k_quant",
                    False,
                )
            ),
        }
        torch.cuda.empty_cache()
        for _ in range(args.warmup):
            warm = run(
                query_chunk_tokens=(
                    args.query_chunk_tokens
                    if stage_name.startswith("streamed_")
                    else None
                ),
                split_qkv_outputs=stage_split_qkv,
                compact_kv=stage_compact_kv,
                single_qknorm_rope=stage_single_qknorm,
                fused_query_projection=stage_fused_query_projection,
                fused_qknorm_hnd_layout=stage_fused_qknorm_hnd_layout,
                direct_nhd_output=args.direct_nhd_output,
                direct_nhd_kv=args.direct_nhd_kv,
            )
            del warm
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        sampler = NvidiaSmiSampler()
        sampler.start()
        started = time.perf_counter()
        result = None
        try:
            cuda_ms_samples: list[float] = []
            wall_samples: list[float] = []
            for repeat_index in range(args.repeat):
                if result is not None:
                    del result
                    result = None
                start_event = torch.cuda.Event(enable_timing=True)
                stop_event = torch.cuda.Event(enable_timing=True)
                repeat_started = time.perf_counter()
                start_event.record()
                current = run(
                    query_chunk_tokens=(
                        args.query_chunk_tokens
                        if stage_name.startswith("streamed_")
                        else None
                    ),
                    split_qkv_outputs=stage_split_qkv,
                    compact_kv=stage_compact_kv,
                    single_qknorm_rope=stage_single_qknorm,
                    fused_query_projection=stage_fused_query_projection,
                    fused_qknorm_hnd_layout=stage_fused_qknorm_hnd_layout,
                    direct_nhd_output=args.direct_nhd_output,
                    direct_nhd_kv=args.direct_nhd_kv,
                )
                stop_event.record()
                stop_event.synchronize()
                cuda_ms_samples.append(float(start_event.elapsed_time(stop_event)))
                wall_samples.append(time.perf_counter() - repeat_started)
                result = current
            stage: dict[str, object] = {
                "status": "ok",
                "warmup": args.warmup,
                "repeat": args.repeat,
                "split_qkv_outputs": stage_split_qkv,
                "compact_kv": stage_compact_kv,
                "single_qknorm_rope": stage_single_qknorm,
                "fused_query_projection": stage_fused_query_projection,
                "fused_qknorm_hnd_layout": stage_fused_qknorm_hnd_layout,
                "direct_nhd_output": args.direct_nhd_output,
                "direct_nhd_kv": args.direct_nhd_kv,
                "backend_optimization_flags": stage_backend_flags,
                "cuda_ms_samples": cuda_ms_samples,
                "cuda_ms": statistics.median(cuda_ms_samples),
                "wall_seconds_samples": wall_samples,
                "wall_seconds": statistics.median(wall_samples),
                "stage_total_wall_seconds": time.perf_counter() - started,
                "checksum_mean_sample": float(
                    result[:: max(1, args.expected_tokens // 32)].float().mean().cpu()
                ),
            }
            if args.full_output_sha256:
                output_sha256 = tensor_sha256(
                    result,
                    chunk_rows=args.hash_chunk_rows,
                )
                stage["full_output_sha256"] = output_sha256
                if reference_output_sha256 is None:
                    reference_output_sha256 = output_sha256
                report["full_output_equivalence"][stage_name] = {
                    "reference_stage": (
                        stage_name if reference_stage is None else reference_stage
                    ),
                    "sha256": output_sha256,
                    "exact_equal": output_sha256 == reference_output_sha256,
                }
            telemetry = getattr(block.attention.backend, "telemetry", None)
            if telemetry is not None:
                stage["attention_telemetry"] = telemetry()
            sample = result.index_select(0, sample_indices).cpu()
            del result
            result = None
            if args.kineto_dir is not None:
                kineto_dir = args.kineto_dir.resolve()
                kineto_dir.mkdir(parents=True, exist_ok=True)
                trace_path = kineto_dir / f"{stage_name}_trace.json"
                table_path = kineto_dir / f"{stage_name}_kernels.txt"
                with torch.profiler.profile(
                    activities=(
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ),
                    record_shapes=True,
                    profile_memory=False,
                    with_stack=False,
                ) as profiler:
                    with torch.profiler.record_function(
                        f"h3_real_block_{stage_name}"
                    ):
                        profiled_result = run(
                            query_chunk_tokens=(
                                args.query_chunk_tokens
                                if stage_name.startswith("streamed_")
                                else None
                            ),
                            split_qkv_outputs=stage_split_qkv,
                            compact_kv=stage_compact_kv,
                            single_qknorm_rope=stage_single_qknorm,
                            fused_query_projection=stage_fused_query_projection,
                            fused_qknorm_hnd_layout=(
                                stage_fused_qknorm_hnd_layout
                            ),
                            direct_nhd_output=args.direct_nhd_output,
                            direct_nhd_kv=args.direct_nhd_kv,
                        )
                torch.cuda.synchronize()
                profiler.export_chrome_trace(str(trace_path))
                table_path.write_text(
                    profiler.key_averages(group_by_input_shape=True).table(
                        sort_by="self_cuda_time_total", row_limit=200
                    ),
                    encoding="utf-8",
                )
                stage["kineto_trace"] = str(trace_path)
                stage["kineto_table"] = str(table_path)
                del profiled_result
            if reference_sample is None:
                reference_stage = stage_name
                reference_sample = sample
                report["sampled_equivalence"][stage_name] = {
                    "reference_stage": stage_name,
                    "sample_rows": int(sample.shape[0]),
                    "exact_equal": True,
                    "max_abs": 0.0,
                }
            else:
                difference = sample.float() - reference_sample.float()
                absolute = difference.abs()
                reference_float = reference_sample.float()
                prefix_mask = sample_indices.cpu() < protected
                video_mask = ~prefix_mask
                report["sampled_equivalence"][stage_name] = {
                    "reference_stage": reference_stage,
                    "sample_rows": int(sample.shape[0]),
                    "exact_equal": bool(torch.equal(sample, reference_sample)),
                    "max_abs": float(absolute.max().item()),
                    "mean_abs": float(absolute.mean().item()),
                    "relative_l1": float(
                        absolute.mean()
                        / reference_float.abs().mean().clamp_min(1.0e-6)
                    ),
                    "cosine": float(
                        torch.nn.functional.cosine_similarity(
                            sample.float().flatten(),
                            reference_float.flatten(),
                            dim=0,
                        )
                    ),
                    "prefix_mean_abs": (
                        float(absolute[prefix_mask].mean())
                        if bool(prefix_mask.any())
                        else None
                    ),
                    "video_mean_abs": (
                        float(absolute[video_mask].mean())
                        if bool(video_mask.any())
                        else None
                    ),
                }
            del sample
        except torch.OutOfMemoryError as error:
            stage = {
                "status": "oom",
                "wall_seconds": time.perf_counter() - started,
                "error": str(error),
            }
        except Exception as error:
            stage = {
                "status": "error",
                "wall_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        finally:
            sampler.stop()
            del result
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                pass
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            stage["peak_allocated_gib"] = peak_allocated / GIB
            stage["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / GIB
            stage["incremental_peak_allocated_gib"] = (
                peak_allocated - baseline_allocated
            ) / GIB
            stage["nvml"] = sampler.summary()
            report["stages"][stage_name] = stage
            torch.cuda.empty_cache()

    bracket_names = (
        "streamed_sparse_split_qkv_reference_start",
        "streamed_sparse_split_qkv_candidate",
        "streamed_sparse_split_qkv_reference_end",
    )
    if all(
        name in report["stages"]
        and report["stages"][name].get("status") == "ok"
        for name in bracket_names
    ):
        reference_samples = [
            *report["stages"][bracket_names[0]]["cuda_ms_samples"],
            *report["stages"][bracket_names[2]]["cuda_ms_samples"],
        ]
        candidate_samples = report["stages"][bracket_names[1]][
            "cuda_ms_samples"
        ]
        reference_cuda_ms = statistics.median(reference_samples)
        candidate_cuda_ms = statistics.median(candidate_samples)
        candidate_equivalence = report["sampled_equivalence"][bracket_names[1]]
        trailing_equivalence = report["sampled_equivalence"][bracket_names[2]]
        report["comparisons"]["bracketed_exact_stack_candidate"] = {
            "reference_stages": [bracket_names[0], bracket_names[2]],
            "candidate_stage": bracket_names[1],
            "reference_cuda_ms": reference_cuda_ms,
            "candidate_cuda_ms": candidate_cuda_ms,
            "speedup": reference_cuda_ms / candidate_cuda_ms,
            "candidate_sampled_exact_equal": candidate_equivalence["exact_equal"],
            "candidate_sampled_max_abs": candidate_equivalence["max_abs"],
            "trailing_reference_sampled_exact_equal": trailing_equivalence[
                "exact_equal"
            ],
        }

    fused_query_bracket_names = (
        "streamed_sparse_split_qkv_single_qknorm_helpers_reference_start",
        "streamed_sparse_split_qkv_single_qknorm_helpers_fused_query_projection",
        "streamed_sparse_split_qkv_single_qknorm_helpers_reference_end",
    )
    if all(
        name in report["stages"]
        and report["stages"][name].get("status") == "ok"
        for name in fused_query_bracket_names
    ):
        reference_samples = [
            *report["stages"][fused_query_bracket_names[0]]["cuda_ms_samples"],
            *report["stages"][fused_query_bracket_names[2]]["cuda_ms_samples"],
        ]
        candidate_samples = report["stages"][fused_query_bracket_names[1]][
            "cuda_ms_samples"
        ]
        reference_cuda_ms = statistics.median(reference_samples)
        candidate_cuda_ms = statistics.median(candidate_samples)
        candidate_equivalence = report["sampled_equivalence"][
            fused_query_bracket_names[1]
        ]
        candidate_full_equivalence = report["full_output_equivalence"].get(
            fused_query_bracket_names[1]
        )
        trailing_full_equivalence = report["full_output_equivalence"].get(
            fused_query_bracket_names[2]
        )
        report["comparisons"]["bracketed_fused_query_projection"] = {
            "reference_stages": [
                fused_query_bracket_names[0],
                fused_query_bracket_names[2],
            ],
            "candidate_stage": fused_query_bracket_names[1],
            "reference_cuda_ms": reference_cuda_ms,
            "candidate_cuda_ms": candidate_cuda_ms,
            "speedup": reference_cuda_ms / candidate_cuda_ms,
            "candidate_sampled_exact_equal": candidate_equivalence["exact_equal"],
            "candidate_full_output_exact_equal": (
                None
                if candidate_full_equivalence is None
                else candidate_full_equivalence["exact_equal"]
            ),
            "trailing_reference_full_output_exact_equal": (
                None
                if trailing_full_equivalence is None
                else trailing_full_equivalence["exact_equal"]
            ),
        }

    fused_qknorm_hnd_bracket_names = (
        "streamed_release_stack_reference_start",
        "streamed_release_stack_fused_qknorm_hnd",
        "streamed_release_stack_reference_end",
    )
    if all(
        name in report["stages"]
        and report["stages"][name].get("status") == "ok"
        for name in fused_qknorm_hnd_bracket_names
    ):
        reference_samples = [
            *report["stages"][fused_qknorm_hnd_bracket_names[0]][
                "cuda_ms_samples"
            ],
            *report["stages"][fused_qknorm_hnd_bracket_names[2]][
                "cuda_ms_samples"
            ],
        ]
        candidate_samples = report["stages"][
            fused_qknorm_hnd_bracket_names[1]
        ]["cuda_ms_samples"]
        reference_cuda_ms = statistics.median(reference_samples)
        candidate_cuda_ms = statistics.median(candidate_samples)
        candidate_equivalence = report["sampled_equivalence"][
            fused_qknorm_hnd_bracket_names[1]
        ]
        candidate_full_equivalence = report["full_output_equivalence"].get(
            fused_qknorm_hnd_bracket_names[1]
        )
        trailing_full_equivalence = report["full_output_equivalence"].get(
            fused_qknorm_hnd_bracket_names[2]
        )
        report["comparisons"]["bracketed_fused_qknorm_hnd_layout"] = {
            "reference_stages": [
                fused_qknorm_hnd_bracket_names[0],
                fused_qknorm_hnd_bracket_names[2],
            ],
            "candidate_stage": fused_qknorm_hnd_bracket_names[1],
            "reference_cuda_ms": reference_cuda_ms,
            "candidate_cuda_ms": candidate_cuda_ms,
            "speedup": reference_cuda_ms / candidate_cuda_ms,
            "candidate_sampled_exact_equal": candidate_equivalence["exact_equal"],
            "candidate_full_output_exact_equal": (
                None
                if candidate_full_equivalence is None
                else candidate_full_equivalence["exact_equal"]
            ),
            "trailing_reference_full_output_exact_equal": (
                None
                if trailing_full_equivalence is None
                else trailing_full_equivalence["exact_equal"]
            ),
        }

    component_names = (
        "streamed_sparse_split_qkv_reference_start",
        "streamed_sparse_split_qkv",
        "streamed_sparse_split_qkv_reference_end",
    )
    if all(
        name in report["stages"]
        and report["stages"][name].get("status") == "ok"
        for name in component_names
    ):
        reference_samples = [
            *report["stages"][component_names[0]]["cuda_ms_samples"],
            *report["stages"][component_names[2]]["cuda_ms_samples"],
        ]
        candidate_samples = report["stages"][component_names[1]][
            "cuda_ms_samples"
        ]
        reference_cuda_ms = statistics.median(reference_samples)
        candidate_cuda_ms = statistics.median(candidate_samples)
        candidate_equivalence = report["sampled_equivalence"][
            component_names[1]
        ]
        full_equivalence = report["full_output_equivalence"].get(
            component_names[1]
        )
        report["comparisons"]["bracketed_component_candidate"] = {
            "reference_stages": [component_names[0], component_names[2]],
            "candidate_stage": component_names[1],
            "candidate_flags": report["stages"][component_names[1]][
                "backend_optimization_flags"
            ],
            "reference_cuda_ms": reference_cuda_ms,
            "candidate_cuda_ms": candidate_cuda_ms,
            "speedup": reference_cuda_ms / candidate_cuda_ms,
            "candidate_sampled_exact_equal": candidate_equivalence[
                "exact_equal"
            ],
            "candidate_full_output_exact_equal": (
                None if full_equivalence is None else full_equivalence["exact_equal"]
            ),
        }

    exact_ladder_names = (
        "full_sparse_reference_start",
        "streamed_sparse_reference",
        "streamed_sparse_split_qkv_reference",
        "streamed_sparse_split_qkv_single_qknorm",
        "streamed_sparse_split_qkv_single_qknorm_parallel_lut",
        "full_sparse_reference_end",
    )
    if all(
        name in report["stages"]
        and report["stages"][name].get("status") == "ok"
        for name in exact_ladder_names
    ):
        reference_samples = [
            *report["stages"][exact_ladder_names[0]]["cuda_ms_samples"],
            *report["stages"][exact_ladder_names[-1]]["cuda_ms_samples"],
        ]
        reference_cuda_ms = statistics.median(reference_samples)
        ladder: dict[str, object] = {}
        for name in exact_ladder_names[1:-1]:
            candidate_cuda_ms = statistics.median(
                report["stages"][name]["cuda_ms_samples"]
            )
            full_equivalence = report["full_output_equivalence"].get(name)
            ladder[name] = {
                "cuda_ms": candidate_cuda_ms,
                "speedup": reference_cuda_ms / candidate_cuda_ms,
                "sampled_exact_equal": report["sampled_equivalence"][name][
                    "exact_equal"
                ],
                "full_output_exact_equal": (
                    None
                    if full_equivalence is None
                    else full_equivalence["exact_equal"]
                ),
            }
        report["comparisons"]["exact_execution_ladder"] = {
            "reference_stages": [exact_ladder_names[0], exact_ladder_names[-1]],
            "reference_cuda_ms": reference_cuda_ms,
            "candidates": ladder,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
