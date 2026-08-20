"""Typed contracts for measured workload routing; no creative controls live here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..runtime.config import OffloadMode

EngineName = Literal["original", "lora", "reference"]


@dataclass(frozen=True, slots=True)
class WorkloadFeatures:
    width: int
    height: int
    frames: int
    fps: int
    text_tokens: int
    condition_count: int
    latent_frames: int
    spatial_tokens: int
    video_tokens: int
    condition_tokens: int
    audio_tokens: int
    packed_tokens: int
    output_pixel_frames: int
    engine: EngineName
    actual_evaluations: int
    forecast_evaluations: int

    @property
    def shape_key(self) -> tuple[int, int, int, int]:
        return (self.width, self.height, self.frames, self.condition_count)


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Internal execution mechanics selected after preset quality authorization.

    Most fields are numerically conservative.  Approximate attention and fused
    normalization are deliberately explicit so an experimental profile cannot
    silently leak into the validated high-fidelity route.
    """

    offload_mode: OffloadMode
    mlp_chunk_tokens: int
    block_buffer_count: int = 2
    prefetch_depth: int = 1
    resident_block_count: int = 0
    vae_spatial_tile: tuple[int, int] | None = None
    vae_temporal_tile: int | None = None
    vae_tile_batch_size: int = 1
    vae_transformer_block_compile: bool = False
    attention_topk: float | None = None
    sparse_scope: Literal["middle_only", "guarded", "full"] = "full"
    fused_rms_adaln: bool = False
    dense_qk_quant_gran: Literal["per_thread", "per_warp"] = "per_thread"
    frame_interleave_stride: int = 1
    frame_interleave_layer_start: int = 0
    frame_interleave_layer_stop: int = 50
    frame_interleave_dense_layers: tuple[int, ...] = ()
    frame_interleave_dense_steps: tuple[int, ...] = ()
    spatial_query_lattice_stride: int = 1
    spatial_query_lattice_layer_start: int = 0
    spatial_query_lattice_layer_stop: int = 50
    spatial_query_lattice_dense_layers: tuple[int, ...] = ()
    spatial_query_lattice_dense_steps: tuple[int, ...] = ()
    mlp_spatial_lattice_stride: int = 1
    mlp_spatial_lattice_layer_start: int = 0
    mlp_spatial_lattice_layer_stop: int = 50
    mlp_spatial_lattice_dense_layers: tuple[int, ...] = ()
    mlp_spatial_lattice_dense_steps: tuple[int, ...] = ()
    mlp_spatial_lattice_detail_fraction: float = 0.0
    segment_cache_layer_start: int = 0
    segment_cache_layer_stop: int = 0
    segment_cache_reuse_steps: tuple[int, ...] = ()
    segment_cache_directional_trust: bool = False
    segment_cache_directional_max_extra: float = 0.35
    segment_cache_directional_min_cosine: float = 0.25
    segment_cache_protected_refresh: bool = False
    segment_cache_active_video_ratio: float = 0.0
    segment_cache_dynamic_video_budget: bool = False
    segment_cache_active_video_min_ratio: float = 0.0
    segment_cache_innovation_risk_coverage: float = 0.80
    segment_cache_innovation_max_relative: float = 4.0
    segment_cache_active_layer_start: int = 0
    segment_cache_active_layer_stop: int = 0
    segment_cache_sequential_layer_groups: bool = False
    segment_cache_sequential_conservative_hold: bool = False
    compile_bucket: str | None = None
    long_video_motion_detail_attention: bool = False

    def __post_init__(self) -> None:
        if self.mlp_chunk_tokens <= 0:
            raise ValueError("mlp_chunk_tokens must be positive")
        if self.offload_mode is OffloadMode.BLOCK and self.block_buffer_count != 2:
            raise ValueError("H3 block offload requires exactly two buffers")
        if self.prefetch_depth not in (0, 1):
            raise ValueError("single-GPU H3 supports prefetch depth 0 or 1")
        if self.resident_block_count < 0:
            raise ValueError("resident_block_count cannot be negative")
        if self.vae_temporal_tile is not None and self.vae_temporal_tile <= 0:
            raise ValueError("vae_temporal_tile must be positive")
        if self.vae_tile_batch_size <= 0:
            raise ValueError("vae_tile_batch_size must be positive")
        if self.attention_topk is not None and not 0.5 <= self.attention_topk <= 1.0:
            raise ValueError("attention_topk must be between 0.5 and 1.0")
        if self.sparse_scope not in ("middle_only", "guarded", "full"):
            raise ValueError("sparse_scope must be middle_only, guarded or full")
        if self.dense_qk_quant_gran not in ("per_thread", "per_warp"):
            raise ValueError("dense_qk_quant_gran must be per_thread or per_warp")
        if self.frame_interleave_stride <= 0:
            raise ValueError("frame_interleave_stride must be positive")
        if not (
            0
            <= self.frame_interleave_layer_start
            <= self.frame_interleave_layer_stop
            <= 50
        ):
            raise ValueError("frame interleave layer range must lie inside [0, 50]")
        if (
            tuple(sorted(set(self.frame_interleave_dense_layers)))
            != self.frame_interleave_dense_layers
        ):
            raise ValueError("frame interleave dense layers must be sorted and unique")
        if any(
            layer < 0 or layer >= 50
            for layer in self.frame_interleave_dense_layers
        ):
            raise ValueError("frame interleave dense layer falls outside [0, 50)")
        if (
            tuple(sorted(set(self.frame_interleave_dense_steps)))
            != self.frame_interleave_dense_steps
        ):
            raise ValueError("frame interleave dense steps must be sorted and unique")
        if any(step < 0 for step in self.frame_interleave_dense_steps):
            raise ValueError("frame interleave dense steps cannot be negative")
        if self.spatial_query_lattice_stride <= 0:
            raise ValueError("spatial Query lattice stride must be positive")
        if not (
            0
            <= self.spatial_query_lattice_layer_start
            <= self.spatial_query_lattice_layer_stop
            <= 50
        ):
            raise ValueError("spatial Query lattice layer range must lie inside [0, 50]")
        if (
            tuple(sorted(set(self.spatial_query_lattice_dense_layers)))
            != self.spatial_query_lattice_dense_layers
        ):
            raise ValueError("spatial Query lattice dense layers must be sorted and unique")
        if any(
            layer < 0 or layer >= 50
            for layer in self.spatial_query_lattice_dense_layers
        ):
            raise ValueError("spatial Query lattice dense layer falls outside [0, 50)")
        if (
            tuple(sorted(set(self.spatial_query_lattice_dense_steps)))
            != self.spatial_query_lattice_dense_steps
        ):
            raise ValueError("spatial Query lattice dense steps must be sorted and unique")
        if any(step < 0 for step in self.spatial_query_lattice_dense_steps):
            raise ValueError("spatial Query lattice dense steps cannot be negative")
        if self.mlp_spatial_lattice_stride <= 0:
            raise ValueError("MLP spatial lattice stride must be positive")
        if not (
            0 <= self.mlp_spatial_lattice_layer_start
            <= self.mlp_spatial_lattice_layer_stop <= 50
        ):
            raise ValueError("MLP spatial lattice layer range must lie inside [0, 50]")
        if tuple(sorted(set(self.mlp_spatial_lattice_dense_layers))) != self.mlp_spatial_lattice_dense_layers:
            raise ValueError("MLP spatial lattice dense layers must be sorted and unique")
        if any(layer < 0 or layer >= 50 for layer in self.mlp_spatial_lattice_dense_layers):
            raise ValueError("MLP spatial lattice dense layer falls outside [0, 50)")
        if tuple(sorted(set(self.mlp_spatial_lattice_dense_steps))) != self.mlp_spatial_lattice_dense_steps:
            raise ValueError("MLP spatial lattice dense steps must be sorted and unique")
        if any(step < 0 for step in self.mlp_spatial_lattice_dense_steps):
            raise ValueError("MLP spatial lattice dense steps cannot be negative")
        if not 0.0 <= self.mlp_spatial_lattice_detail_fraction < 1.0:
            raise ValueError("MLP spatial lattice detail fraction must lie inside [0, 1)")
        if not (
            0 <= self.segment_cache_layer_start <= self.segment_cache_layer_stop <= 50
        ):
            raise ValueError("segment cache layer range must lie inside [0, 50]")
        if bool(self.segment_cache_reuse_steps) != (
            self.segment_cache_layer_start < self.segment_cache_layer_stop
        ):
            raise ValueError(
                "segment cache requires both a non-empty layer range and reuse steps"
            )
        if tuple(sorted(set(self.segment_cache_reuse_steps))) != self.segment_cache_reuse_steps:
            raise ValueError("segment cache reuse steps must be sorted and unique")
        if any(step < 0 for step in self.segment_cache_reuse_steps):
            raise ValueError("segment cache reuse steps cannot be negative")
        if self.segment_cache_directional_trust and not self.segment_cache_reuse_steps:
            raise ValueError("directional trust requires segment cache reuse steps")
        if self.segment_cache_protected_refresh and not self.segment_cache_reuse_steps:
            raise ValueError("protected refresh requires segment cache reuse steps")
        if not 0.0 <= self.segment_cache_active_video_ratio <= 1.0:
            raise ValueError("segment cache active video ratio must lie inside [0, 1]")
        if self.segment_cache_active_video_ratio and not self.segment_cache_protected_refresh:
            raise ValueError("active video routing requires protected refresh")
        if (
            self.segment_cache_dynamic_video_budget
            and not self.segment_cache_active_video_ratio
        ):
            raise ValueError(
                "dynamic video budgeting requires a non-zero maximum ratio"
            )
        if not (
            0.0
            <= self.segment_cache_active_video_min_ratio
            <= self.segment_cache_active_video_ratio
        ):
            raise ValueError(
                "active video minimum ratio must lie inside [0, maximum ratio]"
            )
        if not 0.0 < self.segment_cache_innovation_risk_coverage <= 1.0:
            raise ValueError("innovation risk coverage must lie inside (0, 1]")
        if self.segment_cache_innovation_max_relative <= 0.0:
            raise ValueError("innovation relative-risk limit must be positive")
        if not (
            0
            <= self.segment_cache_active_layer_start
            <= self.segment_cache_active_layer_stop
            <= 50
        ):
            raise ValueError("active video layer range must lie inside [0, 50]")
        has_active_layer_range = (
            self.segment_cache_active_layer_start
            < self.segment_cache_active_layer_stop
        )
        if has_active_layer_range and not self.segment_cache_active_video_ratio:
            raise ValueError("active video layer range requires a non-zero video ratio")
        if has_active_layer_range and not (
            self.segment_cache_layer_start
            <= self.segment_cache_active_layer_start
            < self.segment_cache_active_layer_stop
            <= self.segment_cache_layer_stop
        ):
            raise ValueError(
                "active video layer range must lie inside the segment cache range"
            )
        if self.segment_cache_sequential_layer_groups and not (
            self.segment_cache_protected_refresh
            and self.segment_cache_active_video_ratio
            and has_active_layer_range
        ):
            raise ValueError(
                "sequential layer groups require protected refresh, active video, "
                "and an explicit active layer range"
            )
        if (
            self.segment_cache_sequential_conservative_hold
            and not self.segment_cache_sequential_layer_groups
        ):
            raise ValueError(
                "sequential conservative hold requires sequential layer groups"
            )
        if not 0.0 <= self.segment_cache_directional_max_extra <= 1.0:
            raise ValueError("segment cache directional max extra must lie inside [0, 1]")
        if not -1.0 <= self.segment_cache_directional_min_cosine <= 1.0:
            raise ValueError(
                "segment cache directional minimum cosine must lie inside [-1, 1]"
            )
        if self.vae_spatial_tile is not None:
            if len(self.vae_spatial_tile) != 2:
                raise ValueError("VAE spatial tile must contain height and width")
            if any(
                value < 128 or value % 16 for value in self.vae_spatial_tile
            ):
                raise ValueError(
                    "VAE spatial tile dimensions must be >= 128 and divisible by 16"
                )
            if self.vae_spatial_tile[0] != self.vae_spatial_tile[1]:
                raise ValueError("the current H3 Video-VAE supports square tiles only")


@dataclass(frozen=True, slots=True)
class LatencyModel:
    """Small auditable model fitted from real 4090 measurements."""

    intercept_seconds: float = 0.0
    per_packed_token: float = 0.0
    per_packed_token_squared: float = 0.0
    per_output_pixel_frame: float = 0.0
    per_actual_evaluation: float = 0.0
    per_forecast_evaluation: float = 0.0

    def predict(self, features: WorkloadFeatures) -> float:
        n = float(features.packed_tokens)
        result = (
            self.intercept_seconds
            + self.per_packed_token * n
            + self.per_packed_token_squared * n * n
            + self.per_output_pixel_frame * features.output_pixel_frames
            + self.per_actual_evaluation * features.actual_evaluations
            + self.per_forecast_evaluation * features.forecast_evaluations
        )
        return max(0.0, result)


@dataclass(frozen=True, slots=True)
class MemoryModel:
    base_bytes: int
    per_packed_token_bytes: float = 0.0
    per_output_pixel_frame_bytes: float = 0.0
    conditioned_min_bytes: int = 0

    def predict(self, features: WorkloadFeatures) -> int:
        value = (
            self.base_bytes
            + self.per_packed_token_bytes * features.packed_tokens
            + self.per_output_pixel_frame_bytes * features.output_pixel_frames
        )
        if features.condition_count:
            value = max(value, self.conditioned_min_bytes)
        return max(0, int(round(value)))


@dataclass(frozen=True, slots=True)
class CalibratedProfile:
    profile_id: str
    supported_engines: tuple[EngineName, ...]
    plan: ExecutionPlan
    latency: LatencyModel
    memory: MemoryModel
    evidence_status: Literal["experimental", "validated"] = "experimental"
    min_packed_tokens: int | None = None
    max_packed_tokens: int | None = None
    min_spatial_tokens: int | None = None
    max_spatial_tokens: int | None = None
    min_latent_frames: int | None = None
    max_latent_frames: int | None = None
    min_output_pixel_frames: int | None = None
    max_output_pixel_frames: int | None = None
    allowed_condition_counts: tuple[int, ...] | None = None
    allowed_actual_evaluations: tuple[int, ...] | None = None
    allowed_forecast_evaluations: tuple[int, ...] | None = None
    vae_tile_candidates: tuple[int, ...] | None = None
    switch_penalty_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name, minimum, maximum in (
            ("packed_tokens", self.min_packed_tokens, self.max_packed_tokens),
            ("spatial_tokens", self.min_spatial_tokens, self.max_spatial_tokens),
            ("latent_frames", self.min_latent_frames, self.max_latent_frames),
            (
                "output_pixel_frames",
                self.min_output_pixel_frames,
                self.max_output_pixel_frames,
            ),
        ):
            if minimum is not None and minimum < 0:
                raise ValueError(f"min_{name} cannot be negative")
            if maximum is not None and maximum < 0:
                raise ValueError(f"max_{name} cannot be negative")
            if minimum is not None and maximum is not None and minimum > maximum:
                raise ValueError(f"min_{name} cannot exceed max_{name}")
        if self.allowed_condition_counts is not None and any(
            value < 0 or value > 9 for value in self.allowed_condition_counts
        ):
            raise ValueError("allowed condition counts must be between 0 and 9")
        if self.vae_tile_candidates is not None:
            if self.plan.vae_spatial_tile is not None:
                raise ValueError("fixed and adaptive VAE tile policies cannot coexist")
            if not self.vae_tile_candidates:
                raise ValueError("adaptive VAE tile policy requires candidates")
            if any(
                tile < 128 or tile % 16 for tile in self.vae_tile_candidates
            ):
                raise ValueError("VAE tile candidates must be >=128 and divisible by16")

    def supports(self, features: WorkloadFeatures) -> bool:
        bounds = (
            (features.packed_tokens, self.min_packed_tokens, self.max_packed_tokens),
            (features.spatial_tokens, self.min_spatial_tokens, self.max_spatial_tokens),
            (features.latent_frames, self.min_latent_frames, self.max_latent_frames),
            (
                features.output_pixel_frames,
                self.min_output_pixel_frames,
                self.max_output_pixel_frames,
            ),
        )
        return (
            features.engine in self.supported_engines
            and all(
                (minimum is None or value >= minimum)
                and (maximum is None or value <= maximum)
                for value, minimum, maximum in bounds
            )
            and (
                self.allowed_condition_counts is None
                or features.condition_count in self.allowed_condition_counts
            )
            and (
                self.allowed_actual_evaluations is None
                or features.actual_evaluations in self.allowed_actual_evaluations
            )
            and (
                self.allowed_forecast_evaluations is None
                or features.forecast_evaluations in self.allowed_forecast_evaluations
            )
        )


@dataclass(frozen=True, slots=True)
class RouteDecision:
    profile_id: str
    plan: ExecutionPlan
    predicted_seconds: float
    predicted_peak_bytes: int
    shape_cache_hit: bool
    switched_profile: bool
