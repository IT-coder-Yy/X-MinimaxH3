"""Checked-in RTX 4090 profiles backed by dated local measurements."""

from __future__ import annotations

from dataclasses import replace

from ..runtime import OffloadMode
from .contracts import CalibratedProfile, ExecutionPlan, LatencyModel, MemoryModel

_GIB = 1024**3


def validated_original_profiles_2026_08_11() -> tuple[CalibratedProfile, ...]:
    """Return the fail-closed original 9/11 table after Round-7 crossover.

    Block beat or matched Resident end-to-end at both 360p5 and 480p5 while
    reducing allocated peak by about 15 GiB.  The VAE tile is selected from
    visually accepted 256/288 candidates using the exact overlap-grid work.
    """

    block_large = CalibratedProfile(
        profile_id="sm89_original911_block_360p_720p15_r2",
        supported_engines=("original",),
        plan=ExecutionPlan(
            offload_mode=OffloadMode.BLOCK,
            mlp_chunk_tokens=8192,
            block_buffer_count=2,
            prefetch_depth=1,
        ),
        latency=LatencyModel(
            per_packed_token=0.00198026498,
            per_packed_token_squared=2.70884241e-8,
        ),
        memory=MemoryModel(
            base_bytes=int(round(4.7767738144 * _GIB)),
            per_packed_token_bytes=125_093.0,
        ),
        evidence_status="validated",
        max_packed_tokens=105_000,
        max_spatial_tokens=920,
        max_latent_frames=107,
        max_output_pixel_frames=342_000_000,
        allowed_actual_evaluations=(9,),
        allowed_forecast_evaluations=(11,),
        vae_tile_candidates=(256, 288),
        switch_penalty_seconds=0.25,
    )
    return (block_large,)


def validated_profiles_for_engine(engine: str) -> tuple[CalibratedProfile, ...]:
    """Map product engines to their measured compute-family profiles.

    Ref2VA uses the same original/Larry DiT execution mechanics while adding
    packed reference tokens.  Its workload analyzer deliberately reports the
    routing engine as ``reference`` for both dense and LoRA reference layouts,
    so the selected family must be retargeted to that routing identity.  The
    token-aware memory model remains responsible for the additional media.
    """

    if engine in ("original", "reference"):
        profiles = validated_original_profiles_2026_08_11()
    elif engine in ("lora", "reference_lora"):
        profiles = validated_lora_profiles_2026_08_11()
    else:
        raise ValueError(f"unsupported H3 engine for routing: {engine}")
    routing_engine = "reference" if engine in ("reference", "reference_lora") else engine
    return tuple(
        replace(profile, supported_engines=(routing_engine,))
        for profile in profiles
    )


def validated_lora_profiles_2026_08_11() -> tuple[CalibratedProfile, ...]:
    """Larry six-step profile measured from 360p/5s through 720p/15s.

    Block execution was byte-identical to Resident at the 360p and 480p
    anchors while also being marginally faster and using 15--16 GiB less
    allocated memory. The quadratic latency model is a least-squares fit to
    four hot-request measurements; the memory line is shifted upward so none
    of those anchors is underpredicted.
    """

    return (
        CalibratedProfile(
            profile_id="sm89_lora6_block_360p_720p15_r1",
            supported_engines=("lora",),
            plan=ExecutionPlan(
                offload_mode=OffloadMode.BLOCK,
                mlp_chunk_tokens=8192,
                block_buffer_count=2,
                prefetch_depth=1,
            ),
            latency=LatencyModel(
                intercept_seconds=2.23898797,
                per_packed_token=1.33916141e-3,
                per_packed_token_squared=1.79550816e-8,
            ),
            memory=MemoryModel(
                base_bytes=5_160_855_680,
                per_packed_token_bytes=124_894.909,
                conditioned_min_bytes=int(round(10.25 * _GIB)),
            ),
            evidence_status="validated",
            max_packed_tokens=105_000,
            max_spatial_tokens=920,
            max_latent_frames=107,
            max_output_pixel_frames=342_000_000,
            allowed_actual_evaluations=(6,),
            allowed_forecast_evaluations=(0,),
            vae_tile_candidates=(256, 288),
            switch_penalty_seconds=0.25,
        ),
    )


def review_sparse_profiles_2026_08_12() -> tuple[CalibratedProfile, ...]:
    """Quality-gated SM89 sparse candidates; excluded unless explicitly enabled.

    The bounds intentionally describe only generated-video evidence from
    Round 9/10.  In particular, 480p15 is excluded after producing an extra
    spoken phrase, and one-frame conditioning/aspect-ratio variants fail
    closed to the validated dense profiles.
    """

    lora_sparse_plan = ExecutionPlan(
        offload_mode=OffloadMode.BLOCK,
        mlp_chunk_tokens=8192,
        block_buffer_count=2,
        prefetch_depth=1,
        attention_topk=0.50,
    )
    # 67,368-token 720p10 and ~100k-token 720p15 anchors.  This line is used
    # only to rank two feasible profiles inside its narrow measured interval.
    lora_sparse_latency = LatencyModel(
        intercept_seconds=-64.0,
        per_packed_token=3.034e-3,
    )
    lora_sparse_memory = MemoryModel(
        base_bytes=int(round(5.7 * _GIB)),
        per_packed_token_bytes=138_000.0,
        conditioned_min_bytes=int(round(14.0 * _GIB)),
    )
    lora_t2av = CalibratedProfile(
        profile_id="sm89_lora6_sparse050_720landscape_10to15s_review",
        supported_engines=("lora",),
        plan=lora_sparse_plan,
        latency=lora_sparse_latency,
        memory=lora_sparse_memory,
        evidence_status="experimental",
        min_packed_tokens=65_000,
        max_packed_tokens=105_000,
        min_spatial_tokens=920,
        max_spatial_tokens=920,
        min_latent_frames=72,
        max_latent_frames=107,
        min_output_pixel_frames=220_000_000,
        max_output_pixel_frames=342_000_000,
        allowed_condition_counts=(0,),
        allowed_actual_evaluations=(6,),
        allowed_forecast_evaluations=(0,),
        vae_tile_candidates=(256, 288),
    )
    lora_fl2av = CalibratedProfile(
        profile_id="sm89_lora6_sparse050_fl2av_720landscape_10s_review",
        supported_engines=("lora",),
        plan=lora_sparse_plan,
        latency=lora_sparse_latency,
        memory=lora_sparse_memory,
        evidence_status="experimental",
        min_packed_tokens=69_000,
        max_packed_tokens=73_000,
        min_spatial_tokens=920,
        max_spatial_tokens=920,
        min_latent_frames=72,
        max_latent_frames=72,
        min_output_pixel_frames=220_000_000,
        max_output_pixel_frames=235_000_000,
        allowed_condition_counts=(2,),
        allowed_actual_evaluations=(6,),
        allowed_forecast_evaluations=(0,),
        vae_tile_candidates=(256, 288),
    )
    original_sparse = CalibratedProfile(
        profile_id="sm89_original911_sparse075_720landscape_10s_review",
        supported_engines=("original",),
        plan=ExecutionPlan(
            offload_mode=OffloadMode.BLOCK,
            mlp_chunk_tokens=8192,
            block_buffer_count=2,
            prefetch_depth=1,
            attention_topk=0.75,
        ),
        latency=LatencyModel(intercept_seconds=225.0),
        memory=MemoryModel(
            base_bytes=int(round(5.7 * _GIB)),
            per_packed_token_bytes=138_000.0,
        ),
        evidence_status="experimental",
        min_packed_tokens=65_000,
        max_packed_tokens=69_000,
        min_spatial_tokens=920,
        max_spatial_tokens=920,
        min_latent_frames=72,
        max_latent_frames=72,
        min_output_pixel_frames=220_000_000,
        max_output_pixel_frames=235_000_000,
        allowed_condition_counts=(0,),
        allowed_actual_evaluations=(9,),
        allowed_forecast_evaluations=(11,),
        vae_tile_candidates=(256, 288),
    )
    return (lora_t2av, lora_fl2av, original_sparse)


def review_fused_rms_profiles_2026_08_12() -> tuple[CalibratedProfile, ...]:
    """Dense fused RMS/AdaLN candidates with exact measured workload bounds."""

    common = dict(
        supported_engines=("lora",),
        memory=MemoryModel(base_bytes=0),
        evidence_status="experimental",
        allowed_condition_counts=(0,),
        allowed_actual_evaluations=(6,),
        allowed_forecast_evaluations=(0,),
        vae_tile_candidates=(256, 288),
    )
    return (
        CalibratedProfile(
            profile_id="sm89_lora6_rms_480landscape_5s_review",
            plan=ExecutionPlan(
                offload_mode=OffloadMode.BLOCK,
                mlp_chunk_tokens=8192,
                fused_rms_adaln=True,
            ),
            latency=LatencyModel(intercept_seconds=24.069),
            memory=MemoryModel(base_bytes=int(round(6.61 * _GIB))),
            min_packed_tokens=15_614,
            max_packed_tokens=15_614,
            min_spatial_tokens=405,
            max_spatial_tokens=405,
            min_latent_frames=37,
            max_latent_frames=37,
            min_output_pixel_frames=51_425_280,
            max_output_pixel_frames=51_425_280,
            **{key: value for key, value in common.items() if key != "memory"},
        ),
        CalibratedProfile(
            profile_id="sm89_lora6_rms_720landscape_10s_review",
            plan=ExecutionPlan(
                offload_mode=OffloadMode.BLOCK,
                mlp_chunk_tokens=8192,
                fused_rms_adaln=True,
            ),
            latency=LatencyModel(intercept_seconds=164.507),
            memory=MemoryModel(base_bytes=int(round(12.62 * _GIB))),
            min_packed_tokens=67_368,
            max_packed_tokens=67_368,
            min_spatial_tokens=920,
            max_spatial_tokens=920,
            min_latent_frames=72,
            max_latent_frames=72,
            min_output_pixel_frames=228_925_440,
            max_output_pixel_frames=228_925_440,
            **{key: value for key, value in common.items() if key != "memory"},
        ),
        CalibratedProfile(
            profile_id="sm89_lora6_rms_720landscape_15s_review",
            plan=ExecutionPlan(
                offload_mode=OffloadMode.BLOCK,
                mlp_chunk_tokens=8192,
                fused_rms_adaln=True,
                dense_qk_quant_gran="per_warp",
            ),
            latency=LatencyModel(intercept_seconds=305.957),
            memory=MemoryModel(base_bytes=int(round(16.41 * _GIB))),
            min_packed_tokens=100_000,
            max_packed_tokens=100_000,
            min_spatial_tokens=920,
            max_spatial_tokens=920,
            min_latent_frames=107,
            max_latent_frames=107,
            min_output_pixel_frames=341_032_960,
            max_output_pixel_frames=341_032_960,
            **{key: value for key, value in common.items() if key != "memory"},
        ),
    )


def review_combined_profiles_2026_08_12() -> tuple[CalibratedProfile, ...]:
    """Candidates requiring both sparse and fused-RMS authorization."""

    # Round 11 combined 50% sparse attention with request-scoped fused
    # RMSNorm+AdaLN. Keep it on the one measured complex 720p15 prompt shape
    # until Human continuous-playback review expands the quality envelope.
    lora_t2av_720p15_combo = CalibratedProfile(
        profile_id="sm89_lora6_sparse050_rms_720landscape_15s_review",
        supported_engines=("lora",),
        plan=ExecutionPlan(
            offload_mode=OffloadMode.BLOCK,
            mlp_chunk_tokens=8192,
            block_buffer_count=2,
            prefetch_depth=1,
            attention_topk=0.50,
            fused_rms_adaln=True,
            dense_qk_quant_gran="per_warp",
        ),
        latency=LatencyModel(intercept_seconds=226.915),
        memory=MemoryModel(base_bytes=int(round(18.51 * _GIB))),
        evidence_status="experimental",
        min_packed_tokens=100_000,
        max_packed_tokens=100_000,
        min_spatial_tokens=920,
        max_spatial_tokens=920,
        min_latent_frames=107,
        max_latent_frames=107,
        min_output_pixel_frames=341_032_960,
        max_output_pixel_frames=341_032_960,
        allowed_condition_counts=(0,),
        allowed_actual_evaluations=(6,),
        allowed_forecast_evaluations=(0,),
        vae_tile_candidates=(256, 288),
    )
    return (lora_t2av_720p15_combo,)


__all__ = [
    "review_combined_profiles_2026_08_12",
    "review_fused_rms_profiles_2026_08_12",
    "review_sparse_profiles_2026_08_12",
    "validated_lora_profiles_2026_08_11",
    "validated_original_profiles_2026_08_11",
    "validated_profiles_for_engine",
]
