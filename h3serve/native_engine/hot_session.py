"""Persistent real-weight T2AV session for one RTX 4090.

This module owns request-to-request lifecycle, not checkpoint construction.
Callers inject already prepared immutable residencies so the same implementation
can serve both the pruned INT8 base route and the Larry LoRA route.
"""

from __future__ import annotations

import gc
import ctypes
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal

import torch
import torch.nn.functional as F

from .adapters.sampling_mux import (
    AVPrediction,
    AtomicPyAVMuxer,
    ResMultistepAVSampler,
    SamplingPlan,
    TurboAVSampler,
    TurboClockMode,
    simple_sigma_schedule,
)
from .forecast import DirectionalForecastController
from .model import (
    AttentionOnlineBudget,
    FrameInterleaveConfig,
    SpatialQueryLatticeConfig,
    attention_action_schedule as attention_action_schedule_context,
    attention_actual_steps,
    attention_online_budget,
    attention_sparsity,
    attention_force_dense,
    attention_step,
    build_h3_block_executor,
    dense_qk_quantization,
    frame_interleave_config,
    spatial_query_lattice_config,
    MLPSpatialLatticeConfig,
    mlp_spatial_lattice_config,
    rms_adaln_fusion,
    long_video_attention,
)
from .planner import (
    ExecutionPlan,
    H3WorkloadAnalyzer,
    RTX4090Planner,
    select_stable_dense_qk_quantization,
)
from .runtime import ImmutablePinnedModuleResidency, OffloadMode, RuntimeConfig
from .segment_cache import (
    CoordinateAlignedSegmentCache,
    SegmentResidualCacheConfig,
)
from .terminal_latent_guard import stabilize_terminal_video_latent_


VideoDecoder = Callable[[Any, torch.Tensor, int], torch.Tensor]
AudioDecoder = Callable[[Any, torch.Tensor], torch.Tensor]
VideoConditionEncoder = Callable[[Any, Any], Any]
AudioConditionEncoder = Callable[[Any, Any], Any]


class HotSessionCancelled(RuntimeError):
    """A queued request was cancelled between safe GPU operations."""


class _HotSessionCheckpointReached(RuntimeError):
    """Internal control flow after a requested sampler checkpoint."""


def resize_refinement_video_latent_spatial(
    latent: torch.Tensor,
    *,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    """Resize only the spatial axes of an H3 video latent.

    H3 stores video latents as ``B,C,T,H,W``.  Flattening ``B*T`` before the
    interpolation makes the important invariant explicit: no information is
    mixed between adjacent latent frames, so the first pass owns the motion
    trajectory while the second pass receives new spatial degrees of freedom.
    """

    if latent.ndim != 5:
        raise ValueError("refinement video latent must have shape B,C,T,H,W")
    if target_height <= 0 or target_width <= 0:
        raise ValueError("target latent height and width must be positive")
    batch, channels, latent_frames, source_height, source_width = latent.shape
    if (source_height, source_width) == (target_height, target_width):
        return latent
    frame_batch = latent.permute(0, 2, 1, 3, 4).reshape(
        batch * latent_frames,
        channels,
        source_height,
        source_width,
    )
    resized = F.interpolate(
        frame_batch.float(),
        size=(target_height, target_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return resized.reshape(
        batch,
        latent_frames,
        channels,
        target_height,
        target_width,
    ).permute(0, 2, 1, 3, 4).contiguous()


def blend_terminal_refinement_detail(
    motion_latent: torch.Tensor,
    refined_latent: torch.Tensor,
    *,
    source_height: int,
    source_width: int,
    low_frequency_gain: float,
    temporal_lowpass: bool = False,
    temporal_outlier_only: bool = False,
) -> torch.Tensor:
    """Keep the base trajectory's motion while retaining refined detail.

    The terminal pass is intentionally allowed to create spatial frequencies
    that did not exist on the motion canvas.  Its low-frequency residual can,
    however, also rewrite object position and coarse geometry.  Decompose that
    residual at the original latent resolution, attenuate only the low band,
    and retain the complete high band.  A gain of one is exactly the former
    behavior; zero anchors all coarse motion to ``motion_latent``.
    """

    if motion_latent.shape != refined_latent.shape or motion_latent.ndim != 5:
        raise ValueError("terminal refinement latents must share B,C,T,H,W shape")
    if not 0.0 <= low_frequency_gain <= 1.0:
        raise ValueError("terminal low-frequency gain must lie inside [0, 1]")
    if temporal_outlier_only and not temporal_lowpass:
        raise ValueError("temporal outlier filtering requires temporal lowpass")
    if low_frequency_gain == 1.0 and not temporal_lowpass:
        return refined_latent
    batch, channels, latent_frames, height, width = refined_latent.shape
    if not (0 < source_height <= height and 0 < source_width <= width):
        raise ValueError("terminal source geometry must fit the refined latent")
    delta = refined_latent.float() - motion_latent.float()
    frame_delta = delta.permute(0, 2, 1, 3, 4).reshape(
        batch * latent_frames, channels, height, width
    )
    low = F.interpolate(
        frame_delta,
        size=(source_height, source_width),
        mode="area",
    )
    low = F.interpolate(
        low,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    detail = frame_delta - low
    if temporal_lowpass and latent_frames > 1:
        low_sequence = low.reshape(
            batch, latent_frames, channels, height, width
        )
        previous = torch.cat(
            (low_sequence[:, :1], low_sequence[:, :-1]), dim=1
        )
        following = torch.cat(
            (low_sequence[:, 1:], low_sequence[:, -1:]), dim=1
        )
        # Binomial [1, 2, 1] filtering preserves temporally constant detail
        # exactly while suppressing a one-frame coarse correction spike.  It
        # has no threshold or content-specific hand tuning.
        smoothed = (previous + 2.0 * low_sequence + following) * 0.25
        if temporal_outlier_only and latent_frames > 2:
            innovation = low_sequence - smoothed
            score = innovation.square().mean(dim=(2, 3, 4)).sqrt()
            median = score.median(dim=1, keepdim=True).values
            mad = (score - median).abs().median(dim=1, keepdim=True).values
            robust_sigma = (1.4826 * mad).clamp_min(1e-6)
            threshold = median + 3.0 * robust_sigma
            # Preserve all in-distribution corrections exactly.  For an
            # outlier, remove only the fraction above the robust 3-sigma
            # envelope instead of replacing the frame wholesale.
            outlier_weight = (
                (score - threshold).clamp_min(0.0) / score.clamp_min(1e-6)
            ).view(batch, latent_frames, 1, 1, 1)
            low_sequence = low_sequence + outlier_weight * (
                smoothed - low_sequence
            )
            low = low_sequence.reshape_as(frame_delta)
        else:
            low = smoothed.reshape_as(frame_delta)
    blended = (
        motion_latent.float().permute(0, 2, 1, 3, 4).reshape_as(frame_delta)
        + detail
        + low_frequency_gain * low
    )
    return blended.reshape(
        batch, latent_frames, channels, height, width
    ).permute(0, 2, 1, 3, 4).contiguous().to(refined_latent.dtype)


def spatial_highpass_noise(
    noise: torch.Tensor,
    *,
    low_height: int,
    low_width: int,
) -> torch.Tensor:
    """Return normalized detail-band noise absent from a low-res trajectory."""

    if noise.ndim != 5:
        raise ValueError("multiscale noise must have shape B,C,T,H,W")
    batch, channels, latent_frames, height, width = noise.shape
    if not (0 < low_height <= height and 0 < low_width <= width):
        raise ValueError("low-resolution noise geometry must fit the target")
    frames = noise.permute(0, 2, 1, 3, 4).reshape(
        batch * latent_frames, channels, height, width
    ).float()
    low = F.interpolate(frames, size=(low_height, low_width), mode="area")
    reconstructed = F.interpolate(
        low,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    highpass = frames - reconstructed
    scale = highpass.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
    highpass = highpass / scale
    return highpass.reshape(
        batch, latent_frames, channels, height, width
    ).permute(0, 2, 1, 3, 4).contiguous()


@dataclass(frozen=True, slots=True)
class HotSessionRequest:
    prompt: str
    seed: int
    width: int
    height: int
    frames: int
    fps: int
    steps: int
    output_path: Path
    actual_step_indices: tuple[int, ...] | None = None
    # Exact request-local optimization: reference rows are constant across
    # denoise steps.  Keep this switch so benchmark v00 can replay the former
    # implementation and v01 can be compared in the same code revision.
    cache_condition_rows: bool = True
    cache_condition_embeddings: bool = False
    cache_reference_latents: bool = True
    mlp_chunk_tokens: int | None = None
    execution_plan: ExecutionPlan | None = None
    # Physical decisions emitted by the two-control joint optimizer.  Tuple
    # storage keeps the frozen request auditable and checkpoint-comparable.
    attention_action_schedule: tuple[tuple[int, int, str], ...] = ()
    attention_online_guard_id: str | None = None
    attention_online_budget_dense_layers: float = 0.0
    attention_online_rebate_schedule: tuple[tuple[int, int], ...] = ()
    acceleration_plan_summary: dict[str, Any] | None = None
    # When present, the exact-token V19 selector owns both the actual/forecast
    # trajectory and the per-cell Attention schedule.  Selection happens only
    # after Qwen tokenisation and reference preprocessing, so a creator prompt
    # is never routed using the former character-count approximation.
    v19_acceleration: float | None = None
    first_frame: Path | None = None
    last_frame: Path | None = None
    reference_images: tuple[Path, ...] = ()
    reference_videos: tuple[Path, ...] = ()
    reference_audios: tuple[Path, ...] = ()
    # Service-side reference-media caps.  They only downscale while preserving
    # the full frame and source aspect ratio; ``original`` skips that cap.
    reference_image_resolution: str = "720p"
    reference_video_resolution: str = "360p"
    prepared_reference_images: tuple[Any, ...] = ()
    prepared_reference_videos: tuple[Any, ...] = ()
    prepared_reference_audios: tuple[Any, ...] = ()
    cancel_check: Callable[[], bool] | None = None
    progress_callback: Callable[[dict[str, Any]], None] | None = None
    use_lora: bool = False
    # Optional second-pass refinement.  ``refinement_latents_path`` points to
    # a clean (sigma=0) AV checkpoint produced by this runtime.  The checkpoint
    # is re-noised on the same rectified-flow clock and sampled over only the
    # final ``steps`` solver intervals.  This is deliberately separate from
    # forecast steps: it is a new low-noise trajectory, not an invalid
    # continuation after the first pass has already reached sigma=0.
    refinement_latents_path: Path | None = None
    refinement_denoise: float | None = None
    refinement_spatial_mode: Literal["strict", "bicubic"] = "strict"
    preserve_refinement_audio: bool = True
    save_final_latents_path: Path | None = None
    # Resume a formally paused noisy sampler state without replaying its
    # prefix.  The resumed request owns a fresh, explicitly rescheduled sigma
    # tail whose length is ``steps``.  This is intended for disposable preview
    # branches, not for pretending that the shortened tail is the formal run.
    sampler_state_path: Path | None = None
    # A product checkpoint resumes the untouched suffix of the original sigma
    # trajectory.  It is intentionally distinct from sampler_state_path,
    # whose tail is rescheduled for disposable research previews.
    formal_resume_state_path: Path | None = None
    checkpoint_after_step: int | None = None
    checkpoint_state_path: Path | None = None
    # Decode one in-trajectory x0 estimate without interrupting the sampler.
    # This supports a creator-facing preview/card-selection experiment while
    # the exact original solver state continues toward the final result.
    preview_step_index: int | None = None
    preview_output_path: Path | None = None
    preview_latents_path: Path | None = None
    # ``direct_x0`` decodes the clean-sample prediction already produced by
    # the selected formal DiT evaluation.  It adds no preview DiT work.
    # ``fast_finish`` retains the research-only disposable solver branch.
    preview_decode_mode: Literal["direct_x0", "fast_finish"] = "direct_x0"
    # Optional comparison branch using only the formal controller's shallow
    # anchor and forecasted tail for N sigma transitions.
    preview_forecast_steps: int = 0
    preview_forecast_output_path: Path | None = None
    preview_branch_steps: int = 2
    preview_branch_actual_step_indices: tuple[int, ...] | None = None
    # Research controls for a more readable early preview.  Spatially reducing
    # only the disposable branch buys several stable solver evaluations for
    # roughly the cost of the former two full-canvas jumps.  The formal latent
    # trajectory always remains at the requested output geometry.
    preview_branch_spatial_scale: float = 1.0
    preview_branch_warm_history: bool = False
    preview_branch_force_dense: bool = False
    preview_branch_use_lora: bool = False
    # Optional audio-only companion branch.  The primary preview branch keeps
    # the requested/full video canvas, while this disposable LoRA branch may
    # use a cheaper video canvas because only its audio latent is retained.
    # This combines the Base branch's more faithful image with the distilled
    # route's substantially more readable early speech without touching the
    # paused formal trajectory.
    preview_audio_branch_use_lora: bool = False
    preview_audio_branch_steps: int = 4
    preview_audio_branch_spatial_scale: float = 0.65
    preview_ready_callback: Callable[[dict[str, Any]], None] | None = None
    preview_decision_wait: Callable[[], str] | None = None
    # Optional in-trajectory spatial transition.  Early solver positions run
    # on a smaller canvas; after ``multiscale_resize_after_step`` the exact
    # state and RES history are lifted to the requested output canvas and the
    # missing spatial noise band is introduced at the current sigma.
    multiscale_initial_width: int | None = None
    multiscale_initial_height: int | None = None
    multiscale_resize_after_step: int | None = None
    multiscale_highpass_strength: float = 1.0
    # Finish the normal low-resolution trajectory at sigma=0, then lift its
    # clean motion latent to the requested canvas and spend a small number of
    # dense low-noise evaluations on spatial detail.  Unlike the experimental
    # in-trajectory resize above, this starts a mathematically valid new
    # rectified-flow trajectory and never interpolates a noisy solver state.
    terminal_refinement_initial_width: int | None = None
    terminal_refinement_initial_height: int | None = None
    terminal_refinement_steps: int = 0
    terminal_refinement_denoise: float = 0.0125
    terminal_refinement_dense_tail_steps: int = 1
    terminal_refinement_low_frequency_gain: float = 1.0
    terminal_refinement_temporal_lowpass: bool = False
    terminal_refinement_temporal_outlier_only: bool = False

    @property
    def num_frames(self) -> int:
        """Compatibility name consumed by the conditioning adapters."""

        return self.frames

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt cannot be empty")
        if self.width % 32 or self.height % 32:
            raise ValueError("width and height must be multiples of 32")
        if self.frames < 5 or (self.frames - 5) % 17:
            raise ValueError("frames must satisfy 17*n+5")
        if self.steps <= 0 or self.fps <= 0:
            raise ValueError("steps and fps must be positive")
        if self.v19_acceleration is not None and (
            not math.isfinite(self.v19_acceleration)
            or not 0.0 <= self.v19_acceleration <= 100.0
        ):
            raise ValueError("V19 acceleration must lie in [0, 100]")
        if self.preview_step_index is None:
            if self.preview_output_path is not None or self.preview_latents_path is not None:
                raise ValueError(
                    "preview output/checkpoint requires preview_step_index"
                )
        else:
            if not 0 <= self.preview_step_index < self.steps:
                raise ValueError("preview_step_index falls outside the sigma schedule")
            if self.preview_output_path is None:
                raise ValueError("preview_step_index requires preview_output_path")
            if self.preview_decode_mode not in ("direct_x0", "fast_finish"):
                raise ValueError(
                    "preview_decode_mode must be direct_x0 or fast_finish"
                )
            if not 0 <= self.preview_forecast_steps <= 6:
                raise ValueError("preview_forecast_steps must be between 0 and 6")
            if (
                self.preview_forecast_steps > 0
                and self.preview_forecast_output_path is None
            ):
                raise ValueError(
                    "preview_forecast_steps requires preview_forecast_output_path"
                )
            if not 1 <= self.preview_branch_steps <= 6:
                raise ValueError("preview_branch_steps must be between 1 and 6")
            if self.preview_branch_actual_step_indices is not None:
                indices = self.preview_branch_actual_step_indices
                if (
                    tuple(sorted(set(indices))) != indices
                    or any(
                        value < 0 or value >= self.preview_branch_steps
                        for value in indices
                    )
                ):
                    raise ValueError(
                        "preview branch actual steps must be sorted, unique and inside the branch"
                    )
            if not 0.4 <= self.preview_branch_spatial_scale <= 1.0:
                raise ValueError(
                    "preview_branch_spatial_scale must be between 0.4 and 1.0"
                )
            if self.preview_audio_branch_use_lora:
                if not 1 <= self.preview_audio_branch_steps <= 6:
                    raise ValueError(
                        "preview_audio_branch_steps must be between 1 and 6"
                    )
                if not 0.4 <= self.preview_audio_branch_spatial_scale <= 1.0:
                    raise ValueError(
                        "preview_audio_branch_spatial_scale must be between 0.4 and 1.0"
                    )
            if self.preview_decision_wait is not None and self.preview_ready_callback is None:
                raise ValueError("preview decision wait requires a ready callback")
        if self.refinement_latents_path is None:
            if self.refinement_denoise is not None:
                raise ValueError(
                    "refinement_denoise requires refinement_latents_path"
                )
            if self.refinement_spatial_mode != "strict":
                raise ValueError(
                    "refinement_spatial_mode requires refinement_latents_path"
                )
        else:
            if not Path(self.refinement_latents_path).is_file():
                raise ValueError(
                    "refinement latent checkpoint does not exist: "
                    f"{self.refinement_latents_path}"
                )
            if self.refinement_denoise is None:
                raise ValueError(
                    "refinement_latents_path requires refinement_denoise"
                )
            if not 0.0 < self.refinement_denoise <= 1.0:
                raise ValueError("refinement_denoise must be in (0, 1]")
            if self.refinement_spatial_mode not in ("strict", "bicubic"):
                raise ValueError(
                    "refinement_spatial_mode must be strict or bicubic"
                )
            if self.actual_step_indices is not None and self.actual_step_indices != tuple(
                range(self.steps)
            ):
                raise ValueError(
                    "second-pass refinement currently requires every step to be actual"
                )
        if self.sampler_state_path is not None:
            if self.refinement_latents_path is not None:
                raise ValueError(
                    "sampler_state_path and refinement_latents_path are mutually exclusive"
                )
            if not Path(self.sampler_state_path).is_file():
                raise ValueError(
                    "sampler state checkpoint does not exist: "
                    f"{self.sampler_state_path}"
                )
        if self.formal_resume_state_path is not None:
            if self.refinement_latents_path is not None or self.sampler_state_path is not None:
                raise ValueError(
                    "formal resume cannot be combined with refinement or preview resume"
                )
            if not Path(self.formal_resume_state_path).is_file():
                raise ValueError(
                    "formal resume checkpoint does not exist: "
                    f"{self.formal_resume_state_path}"
                )
        if self.checkpoint_after_step is None:
            if self.checkpoint_state_path is not None:
                raise ValueError(
                    "checkpoint_state_path requires checkpoint_after_step"
                )
        else:
            if not 1 <= self.checkpoint_after_step < self.steps:
                raise ValueError(
                    "checkpoint_after_step must be before the final solver step"
                )
            if self.checkpoint_state_path is None:
                raise ValueError(
                    "checkpoint_after_step requires checkpoint_state_path"
                )
            if self.formal_resume_state_path is not None:
                raise ValueError("a resumed request cannot create the same breakpoint again")
        if self.mlp_chunk_tokens is not None and self.mlp_chunk_tokens <= 0:
            raise ValueError("mlp_chunk_tokens must be positive when provided")
        if self.execution_plan is not None and self.mlp_chunk_tokens is not None:
            raise ValueError(
                "execution_plan owns MLP chunking; do not also set mlp_chunk_tokens"
            )
        if self.actual_step_indices is not None:
            if not self.actual_step_indices:
                raise ValueError("actual_step_indices cannot be empty")
            if tuple(sorted(set(self.actual_step_indices))) != self.actual_step_indices:
                raise ValueError("actual_step_indices must be sorted and unique")
            if any(index < 0 or index >= self.steps for index in self.actual_step_indices):
                raise ValueError("actual step index falls outside the requested steps")
        if self.attention_action_schedule:
            if tuple(sorted(set(self.attention_action_schedule))) != self.attention_action_schedule:
                raise ValueError(
                    "attention action schedule must be sorted and contain unique cells"
                )
            actual = (
                frozenset(range(self.steps))
                if self.actual_step_indices is None
                else frozenset(self.actual_step_indices)
            )
            forecast = frozenset(range(self.steps)) - actual
            valid_actions = {
                "dense", "sparse_topk_0.5", "sparse_topk_0.25",
                "sparse_topk_0.1", "sparse_topk_0.0625",
                "round215:sparse_topk_0.5",
                "round215:sparse_topk_0.25",
                "round215:sparse_topk_0.1",
                "round215:sparse_topk_0.0625",
                "frontier:sparse_topk_0.5",
                "frontier:sparse_topk_0.25",
                "frontier:sparse_topk_0.1",
                "frontier:sparse_topk_0.0625",
                "fastfrontier:sparse_topk_0.5",
                "fastfrontier:sparse_topk_0.25",
                "fastfrontier:sparse_topk_0.1",
                "fastfrontier:sparse_topk_0.0625",
                "forecastfrontier:sparse_topk_0.5",
                "forecastfrontier:sparse_topk_0.25",
                "forecastfrontier:sparse_topk_0.1",
                "forecastfrontier:sparse_topk_0.0625",
            }
            for step, layer, action in self.attention_action_schedule:
                forecast_anchor = (
                    step in forecast
                    and layer < 3
                    and action == "forecastfrontier:sparse_topk_0.0625"
                )
                if (
                    not 0 <= layer < 50
                    or (step not in actual and not forecast_anchor)
                ):
                    raise ValueError(
                        "attention action cell must target an actual H3 layer or "
                        "a certified forecast anchor layer"
                    )
                if action not in valid_actions:
                    raise ValueError(f"unknown attention action: {action}")
        if self.attention_online_guard_id is None:
            if self.attention_online_budget_dense_layers != 0.0:
                raise ValueError("online Attention budget requires a guard id")
            if self.attention_online_rebate_schedule:
                raise ValueError("online Attention rebate requires a guard id")
        elif (
            not math.isfinite(self.attention_online_budget_dense_layers)
            or self.attention_online_budget_dense_layers <= 0.0
        ):
            raise ValueError("online Attention guard requires a positive finite budget")
        if self.attention_online_rebate_schedule:
            if (
                tuple(sorted(set(self.attention_online_rebate_schedule)))
                != self.attention_online_rebate_schedule
            ):
                raise ValueError(
                    "online Attention rebate schedule must be sorted and unique"
                )
            actual = (
                frozenset(range(self.steps))
                if self.actual_step_indices is None
                else frozenset(self.actual_step_indices)
            )
            if any(
                step not in actual or not 0 <= layer < 50
                for step, layer in self.attention_online_rebate_schedule
            ):
                raise ValueError(
                    "online Attention rebate must target actual H3 cells"
                )
        multiscale_values = (
            self.multiscale_initial_width,
            self.multiscale_initial_height,
            self.multiscale_resize_after_step,
        )
        if any(value is not None for value in multiscale_values):
            if not all(value is not None for value in multiscale_values):
                raise ValueError("multiscale transition requires width, height and step")
            assert self.multiscale_initial_width is not None
            assert self.multiscale_initial_height is not None
            assert self.multiscale_resize_after_step is not None
            if self.multiscale_initial_width % 32 or self.multiscale_initial_height % 32:
                raise ValueError("multiscale initial canvas must be divisible by 32")
            if self.multiscale_initial_width > self.width or self.multiscale_initial_height > self.height:
                raise ValueError("multiscale initial canvas cannot exceed output canvas")
            if not 0 <= self.multiscale_resize_after_step < self.steps - 1:
                raise ValueError("multiscale transition must leave at least one solver step")
            if not 0.0 <= self.multiscale_highpass_strength <= 1.0:
                raise ValueError("multiscale highpass strength must lie inside [0, 1]")
            actual = (
                set(range(self.steps))
                if self.actual_step_indices is None
                else set(self.actual_step_indices)
            )
            if any(
                index not in actual
                for index in range(self.multiscale_resize_after_step + 1, self.steps)
            ):
                raise ValueError("all post-transition solver steps must be actual")
            if self.first_frame is not None or self.last_frame is not None or self.reference_images or self.reference_videos:
                raise ValueError("multiscale transition is currently T2AV-only")
        terminal_values = (
            self.terminal_refinement_initial_width,
            self.terminal_refinement_initial_height,
        )
        if any(value is not None for value in terminal_values) or self.terminal_refinement_steps:
            if not all(value is not None for value in terminal_values):
                raise ValueError(
                    "terminal refinement requires initial width and height"
                )
            assert self.terminal_refinement_initial_width is not None
            assert self.terminal_refinement_initial_height is not None
            if self.terminal_refinement_steps <= 0:
                raise ValueError("terminal refinement steps must be positive")
            if self.terminal_refinement_steps > 3:
                raise ValueError("terminal refinement supports at most three steps")
            if not 1 <= self.terminal_refinement_dense_tail_steps <= self.terminal_refinement_steps:
                raise ValueError(
                    "terminal refinement dense tail must cover between one and all steps"
                )
            if not 0.0 < self.terminal_refinement_denoise <= 1.0:
                raise ValueError("terminal refinement denoise must be in (0, 1]")
            if not 0.0 <= self.terminal_refinement_low_frequency_gain <= 1.0:
                raise ValueError(
                    "terminal refinement low-frequency gain must lie inside [0, 1]"
                )
            if (
                self.terminal_refinement_temporal_outlier_only
                and not self.terminal_refinement_temporal_lowpass
            ):
                raise ValueError(
                    "terminal temporal outlier filtering requires temporal lowpass"
                )
            if (
                self.terminal_refinement_initial_width % 32
                or self.terminal_refinement_initial_height % 32
            ):
                raise ValueError(
                    "terminal refinement initial canvas must be divisible by 32"
                )
            if (
                self.terminal_refinement_initial_width > self.width
                or self.terminal_refinement_initial_height > self.height
            ):
                raise ValueError(
                    "terminal refinement initial canvas cannot exceed output canvas"
                )
            if self.refinement_latents_path is not None:
                raise ValueError(
                    "terminal refinement cannot be combined with checkpoint refinement"
                )
            if any(value is not None for value in multiscale_values):
                raise ValueError(
                    "terminal refinement cannot be combined with an in-trajectory resize"
                )
            if (
                self.first_frame is not None
                or self.last_frame is not None
                or self.reference_images
                or self.reference_videos
            ):
                raise ValueError("terminal refinement is currently T2AV-only")
        elif (
            self.terminal_refinement_low_frequency_gain != 1.0
            or self.terminal_refinement_temporal_lowpass
            or self.terminal_refinement_temporal_outlier_only
        ):
            raise ValueError(
                "terminal refinement low-frequency gain requires terminal refinement"
            )
        for role, path in (
            ("first_frame", self.first_frame),
            ("last_frame", self.last_frame),
        ):
            if path is not None and not Path(path).is_file():
                raise ValueError(f"{role} does not exist: {path}")
        if len(self.reference_images) > 9:
            raise ValueError("Ref2VA accepts at most 9 reference images")
        for path in self.reference_images:
            if not Path(path).is_file():
                raise ValueError(f"reference image does not exist: {path}")
        if len(self.reference_videos) > 3:
            raise ValueError("Ref2VA accepts at most 3 reference videos")
        if len(self.reference_images) + len(self.reference_videos) > 12:
            raise ValueError("Ref2VA accepts at most 12 total reference files")
        for path in self.reference_videos:
            if not Path(path).is_file():
                raise ValueError(f"reference video does not exist: {path}")
        if len(self.reference_audios) > 3:
            raise ValueError("Ref2VA accepts at most 3 reference audios")
        for path in self.reference_audios:
            if not Path(path).is_file():
                raise ValueError(f"reference audio does not exist: {path}")


@dataclass(frozen=True, slots=True)
class HotSessionResult:
    output_path: Path
    total_seconds: float
    phases: dict[str, float]
    step_seconds: tuple[float, ...]
    forecast_profile: dict[str, Any]
    execution_profile: dict[str, Any]
    peak_allocated_gib: float = 0.0


@dataclass(frozen=True, slots=True)
class HotSessionCheckpointResult:
    checkpoint_path: Path | None
    preview_path: Path | None
    completed_steps: int
    total_steps: int
    total_seconds: float
    phases: dict[str, float]
    step_seconds: tuple[float, ...]
    execution_profile: dict[str, Any]
    peak_allocated_gib: float = 0.0


@dataclass(frozen=True, slots=True)
class _ReferenceLatentCacheEntry:
    key: tuple[Any, ...]
    video_latents: tuple[torch.Tensor, ...]
    video_shapes: tuple[tuple[int, int, int], ...]
    video_kinds: tuple[str, ...]
    audio_latents: tuple[torch.Tensor, ...]
    audio_frames: tuple[int, ...]


class NativeT2AVHotSession:
    """Execute independent requests while retaining immutable host weights."""

    def __init__(
        self,
        *,
        engine: Literal["original", "lora", "reference", "reference_lora"],
        conditioner: Any,
        transformer: ImmutablePinnedModuleResidency,
        video_vae: ImmutablePinnedModuleResidency,
        audio_vae: ImmutablePinnedModuleResidency,
        decode_video: VideoDecoder,
        decode_audio: AudioDecoder,
        encode_video_conditioning: VideoConditionEncoder | None = None,
        encode_audio_conditioning: AudioConditionEncoder | None = None,
        output_root: Path,
        turbo_clock_mode: TurboClockMode = TurboClockMode.SHARED_VIDEO,
        debug_step_dir: Path | None = None,
        debug_final_latents_path: Path | None = None,
        runtime_config: RuntimeConfig = RuntimeConfig(),
        planner: RTX4090Planner | None = None,
        attention_backend: Any | None = None,
        v19_selector: Any | None = None,
    ) -> None:
        self.engine = engine
        self.conditioner = conditioner
        self.transformer = transformer
        self.video_vae = video_vae
        self.audio_vae = audio_vae
        self.decode_video = decode_video
        self.decode_audio = decode_audio
        self.encode_video_conditioning = encode_video_conditioning
        self.encode_audio_conditioning = encode_audio_conditioning
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.turbo_clock_mode = turbo_clock_mode
        self.debug_step_dir = (
            None if debug_step_dir is None else debug_step_dir.resolve()
        )
        self.debug_final_latents_path = (
            None
            if debug_final_latents_path is None
            else debug_final_latents_path.resolve()
        )
        self.runtime_config = runtime_config
        self.planner = planner
        self.attention_backend = attention_backend
        self.v19_selector = v19_selector
        # Research-only hook: a factory may share trajectory calibration across
        # requests while the immutable model session stays hot.  Production
        # behavior is unchanged when this remains ``None``.
        self.forecast_controller_factory: Callable[..., Any] | None = None
        # Research-only whole-DiT speculative verifier.  Production remains
        # unchanged unless a benchmark explicitly selects actual solver steps.
        self.self_speculative_verify_steps: tuple[int, ...] = ()
        self.self_speculative_verify_threshold: float = float("inf")
        self._active_block_executor = None
        # Exact one-entry cache for repeated-prompt seed/preset exploration.
        # Only immutable Qwen outputs live on pinned host memory; generated
        # state is never reused between requests.
        self._prompt_cache: tuple[str, torch.Tensor, torch.Tensor] | None = None
        # One exact multimodal cache supports the common creator workflow of
        # keeping prompt/anchors fixed while exploring seeds.  File content,
        # rather than only the path, participates in the key so replacing an
        # uploaded image can never return stale conditioning.
        self._conditioning_cache: tuple[
            tuple[Any, ...], torch.Tensor, torch.Tensor
        ] | None = None
        # One exact reference-pack cache.  Ref2VA creators commonly keep the
        # same characters/props/voices while changing prompt or seed.  Those
        # VAE latents are deterministic and independent of the denoise state,
        # so retaining only the latest pack avoids repeated Video/Audio-VAE
        # encodes without allowing unbounded host-memory growth.
        self._reference_latent_cache: _ReferenceLatentCacheEntry | None = None

    @staticmethod
    def _uses_turbo_sampler(request: HotSessionRequest) -> bool:
        return bool(request.use_lora)

    @property
    def _uses_reference_layout(self) -> bool:
        return self.engine in ("reference", "reference_lora")

    @staticmethod
    def _timed(phases: dict[str, float], name: str, operation: Callable[[], Any]) -> Any:
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = operation()
        torch.cuda.synchronize()
        phases[name] = time.perf_counter() - started
        return result

    @staticmethod
    def _release_device(*, collect_cycles: bool = False) -> None:
        # Normal tensor lifetimes are reference-counted. Full cyclic GC at
        # every DiT/VAE handoff adds online latency without releasing model
        # storage; reserve it for errors and service shutdown.
        if collect_cycles:
            gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def _release_request_host_scratch() -> None:
        """Release completed task buffers without touching hot model slabs."""

        gc.collect()
        # PyTorch's host caching allocator otherwise retains multi-gigabyte
        # long-video staging blocks as /dev/zero mappings after the request.
        host_empty_cache = getattr(torch._C, "_host_emptyCache", None)
        if callable(host_empty_cache):
            host_empty_cache()
        try:
            libc = ctypes.CDLL(None)
            malloc_trim = libc.malloc_trim
            malloc_trim.argtypes = [ctypes.c_size_t]
            malloc_trim.restype = ctypes.c_int
            malloc_trim(0)
        except (AttributeError, OSError):
            pass

    def generate(
        self, request: HotSessionRequest
    ) -> HotSessionResult | HotSessionCheckpointResult:
        """Generate and restore a clean CPU-resident state after any failure."""

        torch.cuda.reset_peak_memory_stats()
        telemetry_before = self._attention_telemetry()
        try:
            result = self._generate_impl(request)
            profile = dict(result.execution_profile)
            profile["attention_backend"] = self._telemetry_delta(
                telemetry_before,
                self._attention_telemetry(),
            )
            result = replace(
                result,
                execution_profile=profile,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / (1024**3),
            )
            self._persist_scheduler_telemetry(result)
            return result
        except BaseException:
            # Do not strand 20+ GiB after a failed kernel/VAE invocation and
            # poison the next queued request.
            for component in (self.transformer, self.video_vae, self.audio_vae):
                try:
                    component.move_to("cpu", non_blocking=False)
                except Exception:
                    pass
            self._clear_block_executor()
            self._release_device(collect_cycles=True)
            raise

    def _attention_telemetry(self) -> dict[str, Any]:
        telemetry = getattr(self.attention_backend, "telemetry", None)
        return dict(telemetry()) if callable(telemetry) else {}

    @classmethod
    def _telemetry_delta(
        cls,
        before: Any,
        after: Any,
        path: tuple[str, ...] = (),
    ) -> Any:
        """Produce one request-local view from cumulative backend counters."""

        if isinstance(before, dict) and isinstance(after, dict):
            return {
                key: cls._telemetry_delta(before.get(key), value, path + (key,))
                for key, value in after.items()
            }
        if (
            isinstance(before, (int, float))
            and not isinstance(before, bool)
            and isinstance(after, (int, float))
            and not isinstance(after, bool)
        ) and path and (
            path[-1].endswith(("calls", "_count", "_heads", "_tokens", "_pairs"))
            or (len(path) > 1 and path[-2] == "action_calls")
        ):
            return after - before
        if isinstance(before, list) and isinstance(after, list):
            return after[len(before):] if after[:len(before)] == before else after
        return after

    @staticmethod
    def _persist_scheduler_telemetry(
        result: HotSessionResult | HotSessionCheckpointResult,
    ) -> None:
        """Persist opt-in research evidence without adding release latency."""

        destination = os.environ.get("H3_NATIVE_SCHEDULER_TELEMETRY_DIR", "").strip()
        if not destination:
            return
        root = Path(destination).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        output = getattr(result, "output_path", None) or getattr(
            result, "checkpoint_path", None
        )
        stem = Path(output).stem if output is not None else f"request-{time.time_ns()}"
        target = root / f"{stem}.scheduler.json"
        temporary = target.with_suffix(target.suffix + ".tmp")
        document = {
            "schema_version": "h3_native_scheduler_runtime_v1",
            "artifact": None if output is None else str(Path(output).resolve()),
            "total_seconds": result.total_seconds,
            "peak_allocated_gib": result.peak_allocated_gib,
            "phases": result.phases,
            "step_seconds": list(result.step_seconds),
            "execution_profile": result.execution_profile,
            "forecast_profile": getattr(result, "forecast_profile", None),
        }
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(target)

    def _clear_block_executor(self) -> None:
        dit = self.transformer.value
        block_stack = getattr(dit, "block_stack", None)
        if block_stack is not None:
            block_stack.clear_block_executor()
        self._active_block_executor = None

    def _analyze_request_features(
        self,
        request: HotSessionRequest,
        *,
        text_tokens: int,
    ):
        all_steps = tuple(range(request.steps))
        actual = (
            all_steps
            if request.actual_step_indices is None
            else request.actual_step_indices
        )
        forecast_count = request.steps - len(actual) if not self._uses_turbo_sampler(request) else 0
        condition_count = (
            len(request.reference_images) + len(request.reference_videos) + len(request.reference_audios)
            if request.reference_images or request.reference_videos or request.reference_audios
            else int(request.first_frame is not None) + int(request.last_frame is not None)
        )
        condition_token_override = None
        if request.reference_images or request.reference_videos or request.reference_audios:
            from .adapters.conditioning_vae.preprocess import prepare_reference_audios, prepare_reference_images, prepare_reference_videos

            image_tokens = sum(
                ((image.height + 31) // 32) * ((image.width + 31) // 32)
                for image in (prepare_reference_images(request) if request.reference_images else ())
            )
            video_tokens = sum(
                (((len(item.frames) - 5) // 17) * 5 + 2)
                * ((item.frames.shape[1] + 31) // 32)
                * ((item.frames.shape[2] + 31) // 32)
                for item in (prepare_reference_videos(request) if request.reference_videos else ())
            )
            audio_tokens = sum(
                int(item.waveform.shape[-1] + 799) // 800 * 2
                for item in (prepare_reference_audios(request) if request.reference_audios else ())
            )
            condition_token_override = image_tokens + video_tokens + audio_tokens
        features = H3WorkloadAnalyzer(fps=request.fps).analyze(
            width=request.width,
            height=request.height,
            frames=request.frames,
            text_tokens=text_tokens,
            condition_count=condition_count,
            engine=("reference" if self._uses_reference_layout else ("lora" if request.use_lora else "original")),
            actual_evaluations=len(actual),
            forecast_evaluations=forecast_count,
            condition_tokens_override=condition_token_override,
        )
        return features

    def _apply_v19_selection(
        self,
        request: HotSessionRequest,
        *,
        text_tokens: int,
    ) -> HotSessionRequest:
        if request.v19_acceleration is None:
            return request
        if self.v19_selector is None:
            # A configured V19 request cannot silently execute a legacy
            # approximate scheduler.  Preserve the request and model ability
            # with the complete Dense trajectory instead.
            dense_request = replace(
                request,
                actual_step_indices=tuple(range(request.steps)),
                attention_action_schedule=(),
                attention_online_guard_id=None,
                attention_online_budget_dense_layers=0.0,
                attention_online_rebate_schedule=(),
                acceleration_plan_summary={
                    "policy_id": "h3_v19_human_aligned_budgeted_adaptive_inference",
                    "accelerated": False,
                    "reason": "v19_release_bundle_unavailable_dense_fallback",
                    "acceleration": request.v19_acceleration,
                },
            )
            dense_request.validate()
            return dense_request
        from .planner import V19WorkloadContext

        # Initial features are used only to obtain the exact packed layout.
        # The selector then replaces the optimizer-owned actual-step schedule.
        features = self._analyze_request_features(
            request,
            text_tokens=text_tokens,
        )
        workload = V19WorkloadContext(
            model_variant="lora" if request.use_lora else "base",
            service_family=(
                "reference" if self._uses_reference_layout else "first_last"
            ),
            packed_tokens=features.packed_tokens,
            condition_count=features.condition_count,
            reference_images=len(request.reference_images),
            reference_audio=len(request.reference_audios),
            reference_videos=len(request.reference_videos),
            device_arch="sm89",
            width=request.width,
            height=request.height,
            frames=request.frames,
            steps=request.steps,
            actual_step_indices=(
                tuple(range(request.steps))
                if request.actual_step_indices is None
                else request.actual_step_indices
            ),
            sampler="turbo" if request.use_lora else "res_multistep",
            scheduler="simple",
        )
        selected = self.v19_selector.select(
            workload=workload,
            acceleration=request.v19_acceleration,
            required_actual_step_indices=(
                ()
                if request.preview_step_index is None
                else (request.preview_step_index,)
            ),
        )
        selected_request = replace(
            request,
            actual_step_indices=selected.actual_step_indices,
            attention_action_schedule=selected.attention_action_schedule,
            attention_online_guard_id=None,
            attention_online_budget_dense_layers=0.0,
            attention_online_rebate_schedule=(),
            acceleration_plan_summary=selected.summary,
        )
        selected_request.validate()
        return selected_request

    def _resolve_execution_plan(
        self,
        request: HotSessionRequest,
        *,
        text_tokens: int,
    ) -> tuple[ExecutionPlan | None, dict[str, Any]]:
        features = self._analyze_request_features(
            request,
            text_tokens=text_tokens,
        )
        feature_profile = {
            "packed_tokens": features.packed_tokens,
            "spatial_tokens": features.spatial_tokens,
            "latent_frames": features.latent_frames,
            "output_pixel_frames": features.output_pixel_frames,
            "condition_count": features.condition_count,
            "actual_evaluations": features.actual_evaluations,
            "forecast_evaluations": features.forecast_evaluations,
        }
        if request.execution_plan is not None:
            requested_qk = request.execution_plan.dense_qk_quant_gran
            effective_qk, qk_override = select_stable_dense_qk_quantization(
                requested_qk,
                packed_tokens=features.packed_tokens,
            )
            plan = (
                request.execution_plan
                if effective_qk == requested_qk
                else replace(
                    request.execution_plan,
                    dense_qk_quant_gran=effective_qk,
                )
            )
            return plan, {
                "source": "explicit",
                "profile_id": None,
                "offload_mode": plan.offload_mode.value,
                "mlp_chunk_tokens": plan.mlp_chunk_tokens,
                "prefetch_depth": plan.prefetch_depth,
                "resident_block_count": plan.resident_block_count,
                "vae_spatial_tile": plan.vae_spatial_tile,
                "vae_transformer_block_compile": plan.vae_transformer_block_compile,
                "attention_topk": plan.attention_topk,
                "fused_rms_adaln": plan.fused_rms_adaln,
                "long_video_motion_detail_attention": (
                    plan.long_video_motion_detail_attention
                ),
                "dense_qk_quant_gran": plan.dense_qk_quant_gran,
                "dense_qk_quant_gran_requested": requested_qk,
                "dense_qk_stability_override": qk_override,
                "frame_interleave_stride": plan.frame_interleave_stride,
                "frame_interleave_layer_start": plan.frame_interleave_layer_start,
                "frame_interleave_layer_stop": plan.frame_interleave_layer_stop,
                "frame_interleave_dense_layers": list(
                    plan.frame_interleave_dense_layers
                ),
                "frame_interleave_dense_steps": list(
                    plan.frame_interleave_dense_steps
                ),
                "spatial_query_lattice_stride": plan.spatial_query_lattice_stride,
                "spatial_query_lattice_layer_start": (
                    plan.spatial_query_lattice_layer_start
                ),
                "spatial_query_lattice_layer_stop": (
                    plan.spatial_query_lattice_layer_stop
                ),
                "spatial_query_lattice_dense_layers": list(
                    plan.spatial_query_lattice_dense_layers
                ),
                "spatial_query_lattice_dense_steps": list(
                    plan.spatial_query_lattice_dense_steps
                ),
                "mlp_spatial_lattice_stride": plan.mlp_spatial_lattice_stride,
                "mlp_spatial_lattice_layer_start": plan.mlp_spatial_lattice_layer_start,
                "mlp_spatial_lattice_layer_stop": plan.mlp_spatial_lattice_layer_stop,
                "mlp_spatial_lattice_dense_layers": list(
                    plan.mlp_spatial_lattice_dense_layers
                ),
                "mlp_spatial_lattice_dense_steps": list(
                    plan.mlp_spatial_lattice_dense_steps
                ),
                "mlp_spatial_lattice_detail_fraction": (
                    plan.mlp_spatial_lattice_detail_fraction
                ),
                "segment_cache_layer_start": plan.segment_cache_layer_start,
                "segment_cache_layer_stop": plan.segment_cache_layer_stop,
                "segment_cache_reuse_steps": list(plan.segment_cache_reuse_steps),
                "segment_cache_directional_trust": (
                    plan.segment_cache_directional_trust
                ),
                "segment_cache_directional_max_extra": (
                    plan.segment_cache_directional_max_extra
                ),
                "segment_cache_directional_min_cosine": (
                    plan.segment_cache_directional_min_cosine
                ),
                "segment_cache_protected_refresh": (
                    plan.segment_cache_protected_refresh
                ),
                "segment_cache_active_video_ratio": (
                    plan.segment_cache_active_video_ratio
                ),
                "segment_cache_dynamic_video_budget": (
                    plan.segment_cache_dynamic_video_budget
                ),
                "segment_cache_active_video_min_ratio": (
                    plan.segment_cache_active_video_min_ratio
                ),
                "segment_cache_innovation_risk_coverage": (
                    plan.segment_cache_innovation_risk_coverage
                ),
                "segment_cache_innovation_max_relative": (
                    plan.segment_cache_innovation_max_relative
                ),
                "segment_cache_active_layer_start": (
                    plan.segment_cache_active_layer_start
                ),
                "segment_cache_active_layer_stop": (
                    plan.segment_cache_active_layer_stop
                ),
                "segment_cache_sequential_layer_groups": (
                    plan.segment_cache_sequential_layer_groups
                ),
                "segment_cache_sequential_conservative_hold": (
                    plan.segment_cache_sequential_conservative_hold
                ),
                **feature_profile,
            }
        if self.planner is None:
            effective_qk, qk_override = select_stable_dense_qk_quantization(
                "per_thread",
                packed_tokens=features.packed_tokens,
            )
            return None, {
                "source": "legacy_default",
                "profile_id": None,
                "offload_mode": OffloadMode.RESIDENT.value,
                "mlp_chunk_tokens": request.mlp_chunk_tokens,
                "prefetch_depth": None,
                "resident_block_count": None,
                "attention_topk": None,
                "vae_transformer_block_compile": False,
                "fused_rms_adaln": False,
                "long_video_motion_detail_attention": False,
                "dense_qk_quant_gran": effective_qk,
                "dense_qk_quant_gran_requested": "per_thread",
                "dense_qk_stability_override": qk_override,
                "frame_interleave_stride": 1,
                **feature_profile,
            }
        free_bytes, _ = torch.cuda.mem_get_info(self.runtime_config.device)
        # cudaMemGetInfo treats PyTorch's inactive cache as unavailable even
        # though subsequent model allocations can reuse it. Counting only raw
        # driver free memory makes the router unnecessarily choose Block mode
        # after Qwen. Add the reserved-but-unallocated cache back, while the
        # planner still applies its independent 1 GiB safety reserve.
        reusable_cache = max(
            0,
            torch.cuda.memory_reserved(self.runtime_config.device)
            - torch.cuda.memory_allocated(self.runtime_config.device),
        )
        effective_free = free_bytes + reusable_cache
        decision = self.planner.select(
            features,
            free_device_bytes=effective_free,
        )
        requested_qk = decision.plan.dense_qk_quant_gran
        effective_qk, qk_override = select_stable_dense_qk_quantization(
            requested_qk,
            packed_tokens=features.packed_tokens,
        )
        plan = (
            decision.plan
            if effective_qk == requested_qk
            else replace(decision.plan, dense_qk_quant_gran=effective_qk)
        )
        return plan, {
            "source": "rtx4090_planner",
            "profile_id": decision.profile_id,
            "offload_mode": plan.offload_mode.value,
            "mlp_chunk_tokens": plan.mlp_chunk_tokens,
            "prefetch_depth": plan.prefetch_depth,
            "resident_block_count": plan.resident_block_count,
            "vae_spatial_tile": plan.vae_spatial_tile,
            "vae_transformer_block_compile": plan.vae_transformer_block_compile,
            "attention_topk": plan.attention_topk,
            "fused_rms_adaln": plan.fused_rms_adaln,
            "long_video_motion_detail_attention": (
                plan.long_video_motion_detail_attention
            ),
            "dense_qk_quant_gran": plan.dense_qk_quant_gran,
            "dense_qk_quant_gran_requested": requested_qk,
            "dense_qk_stability_override": qk_override,
            "frame_interleave_stride": plan.frame_interleave_stride,
            "frame_interleave_layer_start": plan.frame_interleave_layer_start,
            "frame_interleave_layer_stop": plan.frame_interleave_layer_stop,
            "frame_interleave_dense_layers": list(
                plan.frame_interleave_dense_layers
            ),
            "frame_interleave_dense_steps": list(
                plan.frame_interleave_dense_steps
            ),
            "spatial_query_lattice_stride": plan.spatial_query_lattice_stride,
            "spatial_query_lattice_layer_start": (
                plan.spatial_query_lattice_layer_start
            ),
            "spatial_query_lattice_layer_stop": (
                plan.spatial_query_lattice_layer_stop
            ),
            "spatial_query_lattice_dense_layers": list(
                plan.spatial_query_lattice_dense_layers
            ),
            "spatial_query_lattice_dense_steps": list(
                plan.spatial_query_lattice_dense_steps
            ),
            "mlp_spatial_lattice_stride": plan.mlp_spatial_lattice_stride,
            "mlp_spatial_lattice_layer_start": plan.mlp_spatial_lattice_layer_start,
            "mlp_spatial_lattice_layer_stop": plan.mlp_spatial_lattice_layer_stop,
            "mlp_spatial_lattice_dense_layers": list(
                plan.mlp_spatial_lattice_dense_layers
            ),
            "mlp_spatial_lattice_dense_steps": list(
                plan.mlp_spatial_lattice_dense_steps
            ),
            "mlp_spatial_lattice_detail_fraction": (
                plan.mlp_spatial_lattice_detail_fraction
            ),
            "segment_cache_layer_start": plan.segment_cache_layer_start,
            "segment_cache_layer_stop": plan.segment_cache_layer_stop,
            "segment_cache_reuse_steps": list(plan.segment_cache_reuse_steps),
            "segment_cache_directional_trust": plan.segment_cache_directional_trust,
            "segment_cache_directional_max_extra": (
                plan.segment_cache_directional_max_extra
            ),
            "segment_cache_directional_min_cosine": (
                plan.segment_cache_directional_min_cosine
            ),
            "segment_cache_protected_refresh": plan.segment_cache_protected_refresh,
            "segment_cache_active_video_ratio": (
                plan.segment_cache_active_video_ratio
            ),
            "segment_cache_dynamic_video_budget": (
                plan.segment_cache_dynamic_video_budget
            ),
            "segment_cache_active_video_min_ratio": (
                plan.segment_cache_active_video_min_ratio
            ),
            "segment_cache_innovation_risk_coverage": (
                plan.segment_cache_innovation_risk_coverage
            ),
            "segment_cache_innovation_max_relative": (
                plan.segment_cache_innovation_max_relative
            ),
            "segment_cache_active_layer_start": (
                plan.segment_cache_active_layer_start
            ),
            "segment_cache_active_layer_stop": (
                plan.segment_cache_active_layer_stop
            ),
            "segment_cache_sequential_layer_groups": (
                plan.segment_cache_sequential_layer_groups
            ),
            "segment_cache_sequential_conservative_hold": (
                plan.segment_cache_sequential_conservative_hold
            ),
            "predicted_seconds": decision.predicted_seconds,
            "predicted_peak_gib": decision.predicted_peak_bytes / (1024**3),
            **feature_profile,
            "driver_free_gib": free_bytes / (1024**3),
            "reusable_torch_cache_gib": reusable_cache / (1024**3),
            "effective_free_gib": effective_free / (1024**3),
        }

    def _activate_transformer(self, plan: ExecutionPlan | None) -> Any:
        mode = OffloadMode.RESIDENT if plan is None else plan.offload_mode
        self._clear_block_executor()
        if mode is OffloadMode.BLOCK:
            if plan is None or plan.block_buffer_count != 2:
                raise ValueError("H3 block offload requires a two-buffer execution plan")
            block_count = len(self.transformer.value.block_stack.blocks)
            resident_count = plan.resident_block_count
            if resident_count >= block_count:
                raise ValueError(
                    "Block plan must leave at least one transformer block offloaded"
                )
            host_prefixes = tuple(
                f"block_stack.blocks.{index}"
                for index in range(resident_count, block_count)
            )
            self.transformer.move_partition_to_cuda(
                self.runtime_config.device,
                host_module_prefixes=host_prefixes,
            )
            dit = self.transformer.value
            config = replace(
                self.runtime_config,
                offload_mode=OffloadMode.BLOCK,
                block_buffer_count=plan.block_buffer_count,
            )
            executor = build_h3_block_executor(
                dit.block_stack.blocks[resident_count:],
                config,
                prefetch_depth=plan.prefetch_depth,
            )
            dit.block_stack.configure_block_executor(
                executor,
                offload_start=resident_count,
            )
            self._active_block_executor = executor
            return dit
        if mode not in (OffloadMode.RESIDENT, OffloadMode.MODEL):
            raise ValueError(f"unsupported transformer residency mode: {mode}")
        self.transformer.move_to(self.runtime_config.device, non_blocking=True)
        return self.transformer.value

    def _encode_request(
        self, request: HotSessionRequest
    ) -> tuple[torch.Tensor, torch.Tensor]:
        has_frames = bool(request.reference_images or request.reference_videos or request.reference_audios) or request.first_frame is not None or request.last_frame is not None
        conditioning_key = None
        if has_frames:
            def image_key(path: Path | None):
                if path is None:
                    return None
                # API uploads are stored below a job-specific directory.  The
                # same image uploaded for a second seed therefore has a new
                # pathname even though its conditioning is byte-identical.
                # Cache by content (plus role/order and geometry below), not
                # by transient storage location.
                return self._file_content_digest(Path(path))

            conditioning_key = (
                request.prompt,
                image_key(request.first_frame),
                image_key(request.last_frame),
                tuple(image_key(path) for path in request.reference_images),
                tuple(image_key(path) for path in request.reference_videos),
                tuple(image_key(path) for path in request.reference_audios),
                request.width,
                request.height,
                request.frames,
            )
            cached_conditioning = self._conditioning_cache
            if (
                cached_conditioning is not None
                and cached_conditioning[0] == conditioning_key
            ):
                return (
                    cached_conditioning[1].to("cuda:0", non_blocking=True),
                    cached_conditioning[2].to("cuda:0", non_blocking=True),
                )
        cached = self._prompt_cache
        if not has_frames and cached is not None and cached[0] == request.prompt:
            return (
                cached[1].to("cuda:0", non_blocking=True),
                cached[2].to("cuda:0", non_blocking=True),
            )
        encoded = (
            self.conditioner.encode_request(request)
            if has_frames
            else self.conditioner.encode_prompt(request.prompt)
        )
        embeds = encoded.prompt_embeds
        tags = encoded.text_token_tags
        if has_frames:
            self._conditioning_cache = (
                conditioning_key,
                embeds.detach().to("cpu").pin_memory(),
                tags.detach().to("cpu").pin_memory(),
            )
        else:
            self._prompt_cache = (
                request.prompt,
                embeds.detach().to("cpu").pin_memory(),
                tags.detach().to("cpu").pin_memory(),
            )
        return embeds, tags

    @staticmethod
    def _file_content_digest(path: Path) -> bytes:
        return hashlib.sha256(Path(path).resolve().read_bytes()).digest()

    def _reference_latent_key(
        self, request: HotSessionRequest
    ) -> tuple[Any, ...]:
        """Content-address one deterministic Ref2VA VAE conditioning pack."""

        return (
            tuple(self._file_content_digest(path) for path in request.reference_images),
            tuple(self._file_content_digest(path) for path in request.reference_videos),
            tuple(self._file_content_digest(path) for path in request.reference_audios),
            # Reference videos are capped/aligned against the requested frame
            # count.  Image and standalone-audio encodes are geometry agnostic.
            request.frames if request.reference_videos else None,
        )

    def _generate_impl(self, request: HotSessionRequest) -> HotSessionResult:
        request.validate()
        from .model import set_lora_enabled

        def set_active_lora(enabled: bool) -> int:
            """Toggle source modules and any live block-offload device slots."""

            count = set_lora_enabled(self.transformer.value, enabled)
            executor = self._active_block_executor
            if executor is not None:
                for buffer in executor.buffers:
                    module = getattr(buffer, "module", None)
                    if module is not None:
                        count += set_lora_enabled(module, enabled)
            return count

        adapter_count = set_active_lora(request.use_lora)
        if request.use_lora and adapter_count == 0:
            raise RuntimeError("LoRA route requested but the hot family has no adapters")
        preview_lora_requested = (
            request.preview_decode_mode == "fast_finish"
            and (
                request.preview_branch_use_lora
                or request.preview_audio_branch_use_lora
            )
        )
        if preview_lora_requested and adapter_count == 0:
            raise RuntimeError("LoRA preview requested but the hot family has no adapters")
        active_lora_for_predict = bool(request.use_lora)
        request_engine = (
            "reference_lora" if self._uses_reference_layout and request.use_lora
            else "reference" if self._uses_reference_layout
            else "lora" if request.use_lora
            else "original"
        )
        cancel_check = request.cancel_check or (lambda: False)

        def progress(percent: float, stage: str, detail: str) -> None:
            if request.progress_callback is not None:
                request.progress_callback({
                    "percent": percent, "stage": stage, "detail": detail,
                })

        def raise_if_cancelled() -> None:
            if cancel_check():
                raise HotSessionCancelled("native H3 generation cancelled")

        raise_if_cancelled()
        output = request.output_path.resolve()
        if not output.is_relative_to(self.output_root):
            raise ValueError("output_path must stay inside output_root")
        output.parent.mkdir(parents=True, exist_ok=True)
        preview_output = (
            None
            if request.preview_output_path is None
            else Path(request.preview_output_path).resolve()
        )
        if preview_output is not None:
            if not preview_output.is_relative_to(self.output_root):
                raise ValueError("preview_output_path must stay inside output_root")
            preview_output.parent.mkdir(parents=True, exist_ok=True)
        preview_forecast_output = (
            None
            if request.preview_forecast_output_path is None
            else Path(request.preview_forecast_output_path).resolve()
        )
        if preview_forecast_output is not None:
            if not preview_forecast_output.is_relative_to(self.output_root):
                raise ValueError(
                    "preview_forecast_output_path must stay inside output_root"
                )
            preview_forecast_output.parent.mkdir(parents=True, exist_ok=True)
        phases: dict[str, float] = {}
        started_total = time.perf_counter()
        if request.reference_images or request.reference_videos or request.reference_audios:
            from .adapters.conditioning_vae.preprocess import prepare_reference_audios, prepare_reference_images, prepare_reference_videos

            progress(1, "reference_media", "解码参考媒体")
            prepared_images, prepared_videos, prepared_audios = self._timed(
                phases,
                "reference_media_prepare",
                lambda: (
                    prepare_reference_images(request) if request.reference_images else (),
                    prepare_reference_videos(request) if request.reference_videos else (),
                    prepare_reference_audios(request) if request.reference_audios else (),
                ),
            )
            request = replace(
                request,
                prepared_reference_images=prepared_images,
                prepared_reference_videos=prepared_videos,
                prepared_reference_audios=prepared_audios,
            )
            del prepared_images, prepared_videos, prepared_audios
        progress(3, "text", "理解提示词")
        vision_cache_hits_before = int(
            getattr(self.conditioner, "vision_cache_hits", 0)
        )

        context_5120, text_tags = self._timed(
            phases,
            "text_encode",
            lambda: self._encode_request(request),
        )
        progress(12, "text", "提示词编码完成")
        raise_if_cancelled()

        request = self._apply_v19_selection(
            request,
            text_tokens=int(context_5120.shape[-2]),
        )

        execution_plan, execution_profile = self._resolve_execution_plan(
            request,
            text_tokens=int(context_5120.shape[-2]),
        )
        execution_profile.update({
            "qwen_pinned_weight_cache": bool(
                getattr(self.conditioner, "host_cache_ready", False)
            ),
            "qwen_layer_streaming_cache": bool(
                getattr(self.conditioner, "layer_cache_dir", None)
            ),
            "qwen_vision_feature_cache_hit": int(
                getattr(self.conditioner, "vision_cache_hits", 0)
            ) > vision_cache_hits_before,
            "qwen_vision_feature_cache_mib": round(
                int(getattr(self.conditioner, "vision_feature_cache_bytes", 0))
                / (1024**2),
                3,
            ),
        })
        if request.acceleration_plan_summary is not None:
            execution_profile["joint_acceleration"] = dict(
                request.acceleration_plan_summary
            )

        condition_latents_cpu: tuple[torch.Tensor, ...] = ()
        condition_audio_latents_cpu: tuple[torch.Tensor, ...] = ()
        keyframe_indices: tuple[int, ...] = ()
        reference_shapes: tuple[tuple[int, int, int], ...] = ()
        reference_kinds: tuple[str, ...] = ()
        reference_audio_frames: tuple[int, ...] = ()
        if request.reference_images or request.reference_videos or request.reference_audios or request.first_frame is not None or request.last_frame is not None:
            is_reference = bool(request.reference_images or request.reference_videos or request.reference_audios)
            progress(13, "conditioning", "编码参考媒体条件" if is_reference else "编码首尾帧条件")
            if (request.reference_images or request.reference_videos or request.first_frame is not None or request.last_frame is not None) and self.encode_video_conditioning is None:
                raise RuntimeError(
                    "this Native H3 session has no Video-VAE condition encoder"
                )
            cached_reference = None
            reference_cache_key = None
            if is_reference and request.cache_reference_latents:
                reference_cache_key = self._reference_latent_key(request)
                candidate = self._reference_latent_cache
                if candidate is not None and candidate.key == reference_cache_key:
                    cached_reference = candidate
            execution_profile["reference_latent_cache_hit"] = bool(cached_reference)
            if cached_reference is not None:
                condition_latents_cpu = cached_reference.video_latents
                reference_shapes = cached_reference.video_shapes
                reference_kinds = cached_reference.video_kinds
                condition_audio_latents_cpu = cached_reference.audio_latents
                reference_audio_frames = cached_reference.audio_frames
            elif request.reference_images or request.reference_videos or not is_reference:
                self._timed(phases, "condition_vae_h2d", lambda: self.video_vae.move_to("cuda:0", non_blocking=True))
                frame_conditioning = self._timed(
                    phases, "condition_video_encode",
                    lambda: self.encode_video_conditioning(self.video_vae.value, request),
                )
                if is_reference:
                    condition_latents_cpu = tuple(latent.detach().to("cpu") for latent in frame_conditioning.latents)
                    reference_shapes = tuple(frame_conditioning.latent_shapes)
                    reference_kinds = tuple(frame_conditioning.kinds)
                else:
                    condition_latents_cpu = tuple(item.latent.detach().to("cpu") for item in frame_conditioning.keyframes)
                    keyframe_indices = tuple(int(item.semantic_frame_index) for item in frame_conditioning.keyframes)
                self._timed(phases, "condition_vae_evict", lambda: self.video_vae.move_to("cpu", non_blocking=False))
                self._release_device()
            if request.reference_audios and cached_reference is None:
                if self.encode_audio_conditioning is None:
                    raise RuntimeError("this Native H3 session has no Audio-VAE condition encoder")
                self._timed(phases, "condition_audio_vae_h2d", lambda: self.audio_vae.move_to("cuda:0", non_blocking=True))
                audio_conditions = self._timed(
                    phases, "condition_audio_encode",
                    lambda: self.encode_audio_conditioning(self.audio_vae.value, request),
                )
                condition_audio_latents_cpu = tuple(latent.detach().to("cpu") for latent in audio_conditions)
                reference_audio_frames = tuple(int(latent.shape[-1]) for latent in condition_audio_latents_cpu)
                self._timed(phases, "condition_audio_vae_evict", lambda: self.audio_vae.move_to("cpu", non_blocking=False))
                self._release_device()
            if (
                is_reference
                and request.cache_reference_latents
                and cached_reference is None
            ):
                assert reference_cache_key is not None
                self._reference_latent_cache = _ReferenceLatentCacheEntry(
                    key=reference_cache_key,
                    video_latents=condition_latents_cpu,
                    video_shapes=reference_shapes,
                    video_kinds=reference_kinds,
                    audio_latents=condition_audio_latents_cpu,
                    audio_frames=reference_audio_frames,
                )
            request = replace(request, prepared_reference_images=(), prepared_reference_videos=(), prepared_reference_audios=())
            progress(17, "conditioning", "参考媒体条件完成" if is_reference else "首尾帧条件完成")
            raise_if_cancelled()

        progress(18, "denoise", "载入 DiT 计算阶段")
        dit = self._timed(
            phases,
            "dit_h2d",
            lambda: self._activate_transformer(execution_plan),
        )
        with torch.inference_mode():
            context = self._timed(
                phases,
                "condition_projection",
                lambda: dit.token_refiner(
                    dit.condition_proj(context_5120[0].to(dit.compute_dtype))
                ).unsqueeze(0),
            )
        del context_5120
        condition_video_latents = tuple(
            latent.to("cuda:0", non_blocking=False)
            for latent in condition_latents_cpu
        )
        del condition_latents_cpu
        condition_audio_latents = tuple(
            latent.to("cuda:0", non_blocking=False)
            for latent in condition_audio_latents_cpu
        )
        del condition_audio_latents_cpu
        attention_backend = dit.block_stack.blocks[0].attention.backend
        request_attention_schedule = (
            {
                (int(step), int(layer)): str(action)
                for step, layer, action in request.attention_action_schedule
            }
            if request.attention_action_schedule
            else None
        )
        request_online_budget = (
            AttentionOnlineBudget(
                policy_id=request.attention_online_guard_id,
                limit_dense_layers=request.attention_online_budget_dense_layers,
                rebate_schedule=request.attention_online_rebate_schedule,
            )
            if request.attention_online_guard_id is not None
            else None
        )
        selected_attention_topk = (
            None if execution_plan is None else execution_plan.attention_topk
        )
        sparse_scope = (
            "full" if execution_plan is None else execution_plan.sparse_scope
        )

        def step_attention_topk(step_index: int) -> float | None:
            """Apply the user budget only inside its requested quality guard."""

            if selected_attention_topk is None:
                return None
            if sparse_scope == "full":
                return selected_attention_topk
            if sparse_scope == "guarded":
                # The first two and final two solver points retain dense
                # attention because they carry coarse layout and convergence.
                return (
                    None
                    if step_index < 2 or step_index >= request.steps - 2
                    else selected_attention_topk
                )
            # Conservative mode uses sparse attention only in the central
            # half of the trajectory (inclusive start, exclusive stop).
            start = request.steps // 4
            stop = request.steps - start
            return selected_attention_topk if start <= step_index < stop else None
        guard_approximate_math = bool(
            selected_attention_topk is not None
            if getattr(attention_backend, "request_routed", False)
            else getattr(attention_backend, "approximate", False)
        ) or bool(
            execution_plan is not None and execution_plan.fused_rms_adaln
        ) or bool(
            execution_plan is not None
            and execution_plan.frame_interleave_stride > 1
        ) or bool(
            execution_plan is not None
            and execution_plan.spatial_query_lattice_stride > 1
        ) or bool(
            execution_plan is not None
            and execution_plan.segment_cache_reuse_steps
        )

        duration = request.frames / float(request.fps)
        video_shape = (
            1,
            24,
            ((request.frames - 5) // 17) * 5 + 2,
            request.height // 16,
            request.width // 16,
        )
        initial_video_shape = video_shape
        if request.multiscale_initial_width is not None:
            assert request.multiscale_initial_height is not None
            initial_video_shape = (
                video_shape[0],
                video_shape[1],
                video_shape[2],
                request.multiscale_initial_height // 16,
                request.multiscale_initial_width // 16,
            )
        elif request.terminal_refinement_initial_width is not None:
            assert request.terminal_refinement_initial_height is not None
            initial_video_shape = (
                video_shape[0],
                video_shape[1],
                video_shape[2],
                request.terminal_refinement_initial_height // 16,
                request.terminal_refinement_initial_width // 16,
            )
        audio_shape = (1, 32, 2, round(duration * 40))
        generator = torch.Generator("cpu").manual_seed(request.seed)
        video_noise_cpu = torch.randn(
            initial_video_shape, generator=generator, dtype=torch.float32
        )
        audio_noise_cpu = torch.randn(
            audio_shape, generator=generator, dtype=torch.float32
        )
        multiscale_highpass_cpu = None
        if request.multiscale_initial_width is not None:
            target_noise = torch.randn(
                video_shape, generator=generator, dtype=torch.float32
            )
            multiscale_highpass_cpu = spatial_highpass_noise(
                target_noise,
                low_height=initial_video_shape[-2],
                low_width=initial_video_shape[-1],
            )
            del target_noise
        terminal_video_noise_cpu = None
        terminal_audio_noise_cpu = None
        if request.terminal_refinement_initial_width is not None:
            terminal_video_noise_cpu = torch.randn(
                video_shape, generator=generator, dtype=torch.float32
            )
            terminal_audio_noise_cpu = torch.randn(
                audio_shape, generator=generator, dtype=torch.float32
            )
        preserved_refinement_audio = None
        resume_previous_video = None
        resume_previous_audio = None
        resume_previous_video_sigma = None
        resume_previous_audio_sigma = None
        resume_step_offset = 0
        resume_forecast_state = None
        if request.formal_resume_state_path is not None:
            checkpoint = torch.load(
                Path(request.formal_resume_state_path),
                map_location="cpu",
                weights_only=True,
            )
            expected_metadata = {
                "frames": request.frames,
                "fps": request.fps,
                "width": request.width,
                "height": request.height,
                "engine": request_engine,
                "seed": request.seed,
                "steps": request.steps,
                "representation": "formal_sampler_checkpoint_v1",
            }
            for key, expected in expected_metadata.items():
                if checkpoint.get(key) != expected:
                    raise ValueError(
                        "formal checkpoint metadata mismatch for "
                        f"{key}: expected {expected!r}, got {checkpoint.get(key)!r}"
                    )
            if checkpoint.get("use_lora") is not request.use_lora:
                raise ValueError("formal checkpoint model variant does not match the request")
            prompt_digest = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()
            if checkpoint.get("prompt_sha256") != prompt_digest:
                raise ValueError("formal checkpoint prompt does not match the request")
            resume_step_offset = int(checkpoint.get("next_step_index", -1))
            if not 1 <= resume_step_offset < request.steps:
                raise ValueError("formal checkpoint has an invalid next step")
            full_sigmas = simple_sigma_schedule(request.steps, 12.0)
            recorded_sigmas = tuple(float(value) for value in checkpoint.get("sigmas", ()))
            if recorded_sigmas != full_sigmas:
                raise ValueError("formal checkpoint sigma schedule does not match the request")
            recorded_actual = tuple(
                int(value) for value in checkpoint.get("actual_step_indices", ())
            )
            requested_actual = (
                tuple(range(request.steps))
                if request.actual_step_indices is None
                else request.actual_step_indices
            )
            if recorded_actual != requested_actual:
                raise ValueError("formal checkpoint actual-step schedule does not match")
            recorded_attention_schedule = tuple(
                (int(step), int(layer), str(action))
                for step, layer, action in checkpoint.get(
                    "attention_action_schedule", ()
                )
            )
            if recorded_attention_schedule != request.attention_action_schedule:
                raise ValueError(
                    "formal checkpoint attention schedule does not match"
                )
            if checkpoint.get("attention_online_guard_id") != request.attention_online_guard_id:
                raise ValueError("formal checkpoint online guard does not match")
            recorded_rebate_schedule = tuple(
                (int(step), int(layer))
                for step, layer in checkpoint.get(
                    "attention_online_rebate_schedule", ()
                )
            )
            if recorded_rebate_schedule != request.attention_online_rebate_schedule:
                raise ValueError(
                    "formal checkpoint online rebate schedule does not match"
                )
            if not math.isclose(
                float(checkpoint.get("attention_online_budget_dense_layers", 0.0)),
                request.attention_online_budget_dense_layers,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("formal checkpoint online budget does not match")
            recorded_online_state = checkpoint.get("attention_online_runtime_state")
            if request_online_budget is None:
                if recorded_online_state is not None:
                    raise ValueError(
                        "formal checkpoint unexpectedly contains online runtime state"
                    )
            else:
                if not isinstance(recorded_online_state, dict):
                    raise ValueError(
                        "formal checkpoint is missing online runtime state"
                    )
                request_online_budget.restore_checkpoint_state(
                    recorded_online_state.get("budget")
                )
                restore_verifier = getattr(
                    attention_backend,
                    "restore_online_checkpoint_state",
                    None,
                )
                if restore_verifier is None:
                    raise ValueError(
                        "active attention backend cannot restore online state"
                    )
                restore_verifier(
                    request_online_budget,
                    recorded_online_state.get("verifier"),
                )
            expected_resume_video_shape = initial_video_shape
            if (
                request.multiscale_resize_after_step is not None
                and resume_step_offset > request.multiscale_resize_after_step
            ):
                expected_resume_video_shape = video_shape
            for key, expected_shape in (
                ("video", expected_resume_video_shape),
                ("audio", audio_shape),
            ):
                value = checkpoint.get(key)
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
                    raise ValueError(f"formal checkpoint has an invalid {key} latent")
            video = checkpoint["video"].cuda()
            audio = checkpoint["audio"].cuda()
            sigmas = full_sigmas[resume_step_offset:]
            if not request.use_lora:
                for key, expected_shape in (
                    ("previous_video", expected_resume_video_shape),
                    ("previous_audio", audio_shape),
                ):
                    value = checkpoint.get(key)
                    if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
                        raise ValueError(f"formal checkpoint has an invalid {key} latent")
                resume_previous_video = checkpoint["previous_video"].cuda()
                resume_previous_audio = checkpoint["previous_audio"].cuda()
                resume_previous_video_sigma = float(checkpoint["previous_video_sigma"])
                resume_previous_audio_sigma = float(checkpoint["previous_audio_sigma"])
                resume_forecast_state = checkpoint.get("forecast_state")
            execution_profile["formal_resume"] = {
                "checkpoint": str(Path(request.formal_resume_state_path).resolve()),
                "next_step_index": resume_step_offset,
                "remaining_steps": request.steps - resume_step_offset,
                "formal_prefix_replayed": False,
                "sigma_schedule_preserved": True,
            }
            del checkpoint
        elif request.sampler_state_path is not None:
            checkpoint = torch.load(
                Path(request.sampler_state_path),
                map_location="cpu",
                weights_only=True,
            )
            expected_metadata = {
                "frames": request.frames,
                "fps": request.fps,
                "width": request.width,
                "height": request.height,
                "engine": request_engine,
                "representation": "formal_noisy_sampler_state_after_step",
            }
            for key, expected in expected_metadata.items():
                if checkpoint.get(key) != expected:
                    raise ValueError(
                        "sampler state metadata mismatch for "
                        f"{key}: expected {expected!r}, got {checkpoint.get(key)!r}"
                    )
            for key, expected_shape in (
                ("video", video_shape),
                ("audio", audio_shape),
                ("previous_video", video_shape),
                ("previous_audio", audio_shape),
            ):
                value = checkpoint.get(key)
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
                    raise ValueError(f"sampler state has an invalid {key} latent")
            sigma_start = float(checkpoint["sigma_next"])
            if not 0.0 < sigma_start <= 1.0:
                raise ValueError("sampler state sigma_next must be inside (0, 1]")
            sigmas = tuple(
                sigma_start * (1.0 - offset / request.steps)
                for offset in range(request.steps + 1)
            )
            video = checkpoint["video"].cuda()
            audio = checkpoint["audio"].cuda()
            resume_previous_video = checkpoint["previous_video"].cuda()
            resume_previous_audio = checkpoint["previous_audio"].cuda()
            resume_previous_video_sigma = float(checkpoint["previous_video_sigma"])
            resume_previous_audio_sigma = float(checkpoint["previous_audio_sigma"])
            execution_profile["sampler_resume"] = {
                "checkpoint": str(Path(request.sampler_state_path).resolve()),
                "source_step_index": int(checkpoint["step_index"]),
                "source_sigma_next": sigma_start,
                "solver_steps": request.steps,
                "video_sigmas": list(sigmas),
                "formal_prefix_replayed": False,
            }
            del checkpoint
        elif request.refinement_latents_path is None:
            video = video_noise_cpu.cuda()
            audio = audio_noise_cpu.cuda()
            sigmas = simple_sigma_schedule(request.steps, 12.0)
        else:
            checkpoint = torch.load(
                Path(request.refinement_latents_path),
                map_location="cpu",
                weights_only=True,
            )
            expected_metadata = {
                "frames": request.frames,
                "fps": request.fps,
                "engine": request_engine,
            }
            for key, expected in expected_metadata.items():
                if checkpoint.get(key) != expected:
                    raise ValueError(
                        "refinement checkpoint metadata mismatch for "
                        f"{key}: expected {expected!r}, got {checkpoint.get(key)!r}"
                    )
            source_width = checkpoint.get("width")
            source_height = checkpoint.get("height")
            if not isinstance(source_width, int) or not isinstance(
                source_height, int
            ):
                raise ValueError(
                    "refinement checkpoint is missing integer source geometry"
                )
            source_geometry = (source_width, source_height)
            target_geometry = (request.width, request.height)
            if (
                source_geometry != target_geometry
                and request.refinement_spatial_mode == "strict"
            ):
                raise ValueError(
                    "refinement checkpoint metadata mismatch for geometry: "
                    f"expected {target_geometry!r}, got {source_geometry!r}"
                )
            clean_video_cpu = checkpoint.get("video")
            clean_audio_cpu = checkpoint.get("audio")
            source_video_shape = (
                1,
                24,
                video_shape[2],
                source_height // 16,
                source_width // 16,
            )
            if not isinstance(clean_video_cpu, torch.Tensor) or tuple(
                clean_video_cpu.shape
            ) != source_video_shape:
                raise ValueError("refinement checkpoint has an invalid video latent")
            if not isinstance(clean_audio_cpu, torch.Tensor) or tuple(
                clean_audio_cpu.shape
            ) != audio_shape:
                raise ValueError("refinement checkpoint has an invalid audio latent")
            clean_video_cpu = resize_refinement_video_latent_spatial(
                clean_video_cpu,
                target_height=video_shape[-2],
                target_width=video_shape[-1],
            )
            assert request.refinement_denoise is not None
            total_refinement_steps = int(request.steps / request.refinement_denoise)
            if total_refinement_steps < request.steps:
                total_refinement_steps = request.steps
            full_refinement_sigmas = simple_sigma_schedule(
                total_refinement_steps, 12.0
            )
            sigmas = full_refinement_sigmas[-(request.steps + 1) :]
            sigma_start = float(sigmas[0])
            video = (
                sigma_start * video_noise_cpu
                + (1.0 - sigma_start) * clean_video_cpu.float()
            ).cuda()
            audio = (
                sigma_start * audio_noise_cpu
                + (1.0 - sigma_start) * clean_audio_cpu.float()
            ).cuda()
            if request.preserve_refinement_audio:
                preserved_refinement_audio = clean_audio_cpu.cuda()
            execution_profile["refinement"] = {
                "checkpoint": str(Path(request.refinement_latents_path).resolve()),
                "denoise": request.refinement_denoise,
                "solver_steps": request.steps,
                "schedule_total_steps": total_refinement_steps,
                "video_sigmas": list(sigmas),
                "preserve_first_pass_audio": request.preserve_refinement_audio,
                "spatial_mode": request.refinement_spatial_mode,
                "source_geometry": list(source_geometry),
                "target_geometry": list(target_geometry),
            }
            del checkpoint, clean_video_cpu, clean_audio_cpu
        del video_noise_cpu, audio_noise_cpu
        all_steps = tuple(range(request.steps))
        actual_steps = (
            all_steps
            if request.actual_step_indices is None
            else request.actual_step_indices
        )
        if self._uses_turbo_sampler(request) and actual_steps != all_steps:
            raise ValueError("the distilled LoRA route executes every requested step")
        segment_cache = None
        if execution_plan is not None and execution_plan.segment_cache_reuse_steps:
            if not set(execution_plan.segment_cache_reuse_steps).issubset(actual_steps):
                raise ValueError("segment cache reuse steps must be actual DiT steps")
            segment_cache = CoordinateAlignedSegmentCache(
                SegmentResidualCacheConfig(
                    layer_start=execution_plan.segment_cache_layer_start,
                    layer_stop=execution_plan.segment_cache_layer_stop,
                    reuse_steps=execution_plan.segment_cache_reuse_steps,
                    directional_trust=(
                        execution_plan.segment_cache_directional_trust
                    ),
                    directional_max_extra=(
                        execution_plan.segment_cache_directional_max_extra
                    ),
                    directional_min_cosine=(
                        execution_plan.segment_cache_directional_min_cosine
                    ),
                    protected_refresh=(
                        execution_plan.segment_cache_protected_refresh
                    ),
                    active_video_ratio=(
                        execution_plan.segment_cache_active_video_ratio
                    ),
                    dynamic_video_budget=(
                        execution_plan.segment_cache_dynamic_video_budget
                    ),
                    active_video_min_ratio=(
                        execution_plan.segment_cache_active_video_min_ratio
                    ),
                    innovation_risk_coverage=(
                        execution_plan.segment_cache_innovation_risk_coverage
                    ),
                    innovation_max_relative=(
                        execution_plan.segment_cache_innovation_max_relative
                    ),
                    active_layer_start=(
                        execution_plan.segment_cache_active_layer_start
                    ),
                    active_layer_stop=(
                        execution_plan.segment_cache_active_layer_stop
                    ),
                    sequential_layer_groups=(
                        execution_plan.segment_cache_sequential_layer_groups
                    ),
                    sequential_conservative_hold=(
                        execution_plan.segment_cache_sequential_conservative_hold
                    ),
                )
            )
        forecast = None
        forecast_profile_override = None
        if not self._uses_turbo_sampler(request):
            if self.forecast_controller_factory is not None:
                forecast = self.forecast_controller_factory(
                    segment_cache=segment_cache
                )
            elif (
                actual_steps != all_steps
                or request.preview_forecast_steps > 0
                or (
                    request.preview_decode_mode == "fast_finish"
                    and request.preview_branch_actual_step_indices is not None
                    and len(request.preview_branch_actual_step_indices)
                    < request.preview_branch_steps
                )
            ):
                forecast = DirectionalForecastController(
                    actual_steps=actual_steps,
                    segment_cache=segment_cache,
                )
        if segment_cache is not None and forecast is None:
            raise ValueError(
                "the first segment-cache prototype requires the original forecast route"
            )
        if resume_forecast_state is not None:
            if forecast is None:
                raise ValueError(
                    "formal checkpoint contains forecast history but this request has no controller"
                )
            forecast.restore_checkpoint_state(resume_forecast_state)
        remaining_actual_steps = tuple(
            index for index in actual_steps if index >= resume_step_offset
        )
        plan = SamplingPlan(
            sampler="turbo" if self._uses_turbo_sampler(request) else "res_multistep",
            video_sigmas=sigmas,
            audio_sigmas=sigmas,
            actual_step_indices=remaining_actual_steps,
            video_shift=12.0,
            audio_shift=3.0,
            step_index_offset=resume_step_offset,
        )
        layout = None
        step_seconds: list[float] = []
        self_speculative_records: list[dict[str, Any]] = []
        last_denoised: tuple[torch.Tensor, torch.Tensor] | None = None
        preview_latents: tuple[torch.Tensor, torch.Tensor] | None = None
        preview_published = False

        def transition_multiscale(
            index,
            clock,
            step_video,
            step_audio,
            previous_video,
            previous_audio,
        ):
            nonlocal forecast, forecast_profile_override, layout, last_denoised, multiscale_highpass_cpu
            if index != request.multiscale_resize_after_step:
                return step_video, step_audio, previous_video, previous_audio
            target_height, target_width = video_shape[-2:]
            step_video = resize_refinement_video_latent_spatial(
                step_video,
                target_height=target_height,
                target_width=target_width,
            )
            if previous_video is not None:
                previous_video = resize_refinement_video_latent_spatial(
                    previous_video,
                    target_height=target_height,
                    target_width=target_width,
                )
            if multiscale_highpass_cpu is not None:
                detail = multiscale_highpass_cpu.to("cuda:0", non_blocking=False)
                step_video = step_video + (
                    float(clock.video_sigma_next)
                    * request.multiscale_highpass_strength
                    * detail
                )
                del detail
                multiscale_highpass_cpu = None
            if last_denoised is not None:
                last_denoised = (
                    resize_refinement_video_latent_spatial(
                        last_denoised[0],
                        target_height=target_height,
                        target_width=target_width,
                    ),
                    last_denoised[1],
                )
            if forecast is not None:
                forecast_profile_override = forecast.export()
            forecast = None
            layout = None
            execution_profile["multiscale_transition"] = {
                "after_step": index,
                "sigma_next": float(clock.video_sigma_next),
                "source_geometry": [
                    request.multiscale_initial_width,
                    request.multiscale_initial_height,
                ],
                "target_geometry": [request.width, request.height],
                "highpass_strength": request.multiscale_highpass_strength,
                "post_transition_steps": request.steps - index - 1,
            }
            return step_video, step_audio, previous_video, previous_audio

        def predict(video_value, audio_value, clock, *, step_index, is_actual_step):
            nonlocal layout, last_denoised
            raise_if_cancelled()
            step_started = time.perf_counter()
            block_stack_runner = None
            if forecast is not None:

                def block_stack_runner(stack, value, **kwargs):
                    return forecast.run_block_stack(
                        stack,
                        value,
                        step_index=step_index,
                        requested_actual=is_actual_step,
                        **kwargs,
                    )
            interleave = None
            if (
                execution_plan is not None
                and execution_plan.frame_interleave_stride > 1
                and step_index not in execution_plan.frame_interleave_dense_steps
            ):
                interleave = FrameInterleaveConfig(
                    stride=execution_plan.frame_interleave_stride,
                    layer_start=execution_plan.frame_interleave_layer_start,
                    layer_stop=execution_plan.frame_interleave_layer_stop,
                    dense_layers=execution_plan.frame_interleave_dense_layers,
                )
            query_lattice = None
            if (
                execution_plan is not None
                and execution_plan.spatial_query_lattice_stride > 1
                and step_index
                not in execution_plan.spatial_query_lattice_dense_steps
            ):
                query_lattice = SpatialQueryLatticeConfig(
                    stride=execution_plan.spatial_query_lattice_stride,
                    layer_start=(
                        execution_plan.spatial_query_lattice_layer_start
                    ),
                    layer_stop=execution_plan.spatial_query_lattice_layer_stop,
                    dense_layers=(
                        execution_plan.spatial_query_lattice_dense_layers
                    ),
                    phase_offset=step_index,
                )
            mlp_lattice = None
            if (
                execution_plan is not None
                and execution_plan.mlp_spatial_lattice_stride > 1
                and step_index not in execution_plan.mlp_spatial_lattice_dense_steps
            ):
                mlp_lattice = MLPSpatialLatticeConfig(
                    stride=execution_plan.mlp_spatial_lattice_stride,
                    layer_start=execution_plan.mlp_spatial_lattice_layer_start,
                    layer_stop=execution_plan.mlp_spatial_lattice_layer_stop,
                    dense_layers=execution_plan.mlp_spatial_lattice_dense_layers,
                    phase_offset=step_index,
                    detail_fraction=(
                        execution_plan.mlp_spatial_lattice_detail_fraction
                    ),
                )
            def run_dit_once():
                return dit(
                    video_value,
                    audio_value,
                    context,
                    torch.tensor([clock.video_sigma], device="cuda"),
                    output_frame_count=request.frames,
                    text_token_tags=text_tags,
                    condition_video_latents=condition_video_latents,
                    condition_audio_latents=condition_audio_latents,
                    keyframe_indices=keyframe_indices,
                    reference_shapes=reference_shapes,
                    reference_kinds=reference_kinds,
                    reference_audio_frames=reference_audio_frames,
                    condition_seed=(request.seed if self._uses_reference_layout else 42),
                    cache_condition_rows=request.cache_condition_rows,
                    cache_condition_embeddings=request.cache_condition_embeddings,
                    layout=layout,
                    audio_transport_scale=(
                        4.0
                        if active_lora_for_predict
                        and self.turbo_clock_mode is TurboClockMode.SHARED_VIDEO
                        else None
                    ),
                    block_stack_runner=block_stack_runner,
                    mlp_chunk_tokens=(
                        execution_plan.mlp_chunk_tokens
                        if execution_plan is not None
                        else request.mlp_chunk_tokens
                    ),
                )

            with (
                torch.inference_mode(),
                attention_actual_steps(actual_steps),
                attention_step(step_index, request.steps),
                attention_action_schedule_context(request_attention_schedule),
                attention_online_budget(request_online_budget),
                attention_sparsity(step_attention_topk(step_index)),
                frame_interleave_config(interleave),
                spatial_query_lattice_config(query_lattice),
                mlp_spatial_lattice_config(mlp_lattice),
                dense_qk_quantization(
                    str(execution_profile["dense_qk_quant_gran"])
                ),
                rms_adaln_fusion(
                    False
                    if execution_plan is None
                    else execution_plan.fused_rms_adaln
                ),
                long_video_attention(
                    False
                    if execution_plan is None
                    else execution_plan.long_video_motion_detail_attention
                ),
            ):
                verify_whole_dit = (
                    is_actual_step
                    and step_index in self.self_speculative_verify_steps
                )
                if verify_whole_dit and (
                    forecast is None or forecast.segment_cache is not None
                ):
                    raise RuntimeError(
                        "whole-DiT speculative verification requires the "
                        "directional forecast controller without segment cache"
                    )
                history_before = list(forecast.history) if verify_whole_dit else None
                records_before = list(forecast.records) if verify_whole_dit else None
                result = run_dit_once()
                if verify_whole_dit:
                    assert forecast is not None
                    draft_history = list(forecast.history)
                    draft_records = list(forecast.records)
                    forecast.history = history_before
                    forecast.records = records_before
                    with attention_force_dense():
                        exact_result = run_dit_once()

                    def relative_rms(reference, candidate):
                        difference = (
                            (reference.float() - candidate.float())
                            .square()
                            .mean()
                            .sqrt()
                        )
                        scale = (
                            reference.float().square().mean().sqrt().clamp_min(1e-6)
                        )
                        return float((difference / scale).item())

                    video_error = relative_rms(exact_result.video, result.video)
                    audio_error = relative_rms(exact_result.audio, result.audio)
                    rejected = max(video_error, audio_error) > float(
                        self.self_speculative_verify_threshold
                    )
                    self_speculative_records.append(
                        {
                            "step": int(step_index),
                            "video_relative_rms": video_error,
                            "audio_relative_rms": audio_error,
                            "threshold": float(
                                self.self_speculative_verify_threshold
                            ),
                            "decision": "dense_rollback" if rejected else "accept_draft",
                        }
                    )
                    if rejected:
                        result = exact_result
                    else:
                        forecast.history = draft_history
                        forecast.records = draft_records
            if guard_approximate_math:
                for modality, prediction in (
                    ("video", result.video),
                    ("audio", result.audio),
                ):
                    if not bool(torch.isfinite(prediction).all().item()):
                        raise FloatingPointError(
                            f"non-finite {modality} DiT prediction at sampling step "
                            f"{step_index} of {request.steps}"
                        )
            layout = result.layout
            torch.cuda.synchronize()
            step_seconds.append(time.perf_counter() - step_started)
            sigma = clock.video_sigma
            prediction = AVPrediction(
                video_denoised=video_value - result.video * sigma,
                audio_denoised=audio_value - result.audio * sigma,
            )
            last_denoised = (
                prediction.video_denoised,
                prediction.audio_denoised,
            )
            return prediction

        def finish_preview_branch(
            index,
            clock,
            step_video,
            step_audio,
            *,
            branch_steps=None,
            branch_spatial_scale=None,
            branch_warm_history=None,
            branch_force_dense=None,
            branch_use_lora=None,
            branch_forecast_only=False,
            branch_actual_step_indices=None,
        ):
            """Fast-finish a disposable branch without mutating main solver state."""

            nonlocal forecast, layout, last_denoised
            nonlocal condition_video_latents, reference_shapes
            nonlocal active_lora_for_predict
            sigma_start = float(clock.video_sigma_next)
            if sigma_start <= 0.0:
                if last_denoised is None:
                    raise RuntimeError("preview branch has no denoised estimate")
                return last_denoised[0].clone(), last_denoised[1].clone()
            count = int(
                request.preview_branch_steps
                if branch_steps is None else branch_steps
            )
            branch_sigmas = tuple(
                sigma_start * (1.0 - offset / count)
                for offset in range(count + 1)
            )
            use_lora_override = (
                request.preview_branch_use_lora
                if branch_use_lora is None else branch_use_lora
            )
            branch_uses_lora = bool(request.use_lora or use_lora_override)
            requested_branch_actual = (
                request.preview_branch_actual_step_indices
                if branch_actual_step_indices is None
                else branch_actual_step_indices
            )
            branch_actual = (
                ()
                if branch_forecast_only
                else tuple(range(count))
                if requested_branch_actual is None
                else tuple(requested_branch_actual)
            )
            branch_plan = SamplingPlan(
                sampler="turbo" if branch_uses_lora else "res_multistep",
                video_sigmas=branch_sigmas,
                audio_sigmas=branch_sigmas,
                actual_step_indices=branch_actual,
                video_shift=12.0,
                audio_shift=3.0,
            )
            saved_forecast = forecast
            saved_layout = layout
            saved_denoised = last_denoised
            saved_condition_video_latents = condition_video_latents
            saved_reference_shapes = reference_shapes
            saved_lora_mode = active_lora_for_predict
            saved_step_count = len(step_seconds)
            saved_forecast_record_count = (
                None if saved_forecast is None else len(saved_forecast.records)
            )
            branch_uses_forecast = len(branch_actual) < count
            saved_forecast_history = (
                None if saved_forecast is None else list(saved_forecast.history)
            )
            if branch_uses_forecast:
                if saved_forecast is None or len(saved_forecast.history) < 2:
                    raise RuntimeError(
                        "forecast preview requires two formal actual observations"
                    )
                forecast = saved_forecast
            else:
                forecast = None
            try:
                if branch_uses_lora != saved_lora_mode:
                    if set_active_lora(branch_uses_lora) == 0:
                        raise RuntimeError("LoRA preview adapters are unavailable")
                    active_lora_for_predict = branch_uses_lora
                branch_video = step_video.clone()
                branch_audio = step_audio.clone()
                previous_video = None
                previous_audio = None
                previous_video_sigma = None
                previous_audio_sigma = None
                warm_history = (
                    request.preview_branch_warm_history
                    if branch_warm_history is None else branch_warm_history
                )
                if warm_history and saved_denoised is not None:
                    previous_video = saved_denoised[0].clone()
                    previous_audio = saved_denoised[1].clone()
                    previous_video_sigma = float(clock.video_sigma)
                    previous_audio_sigma = float(clock.audio_sigma)

                scale = float(
                    request.preview_branch_spatial_scale
                    if branch_spatial_scale is None else branch_spatial_scale
                )
                if scale < 1.0:
                    source_height = int(branch_video.shape[-2])
                    source_width = int(branch_video.shape[-1])
                    target_height = max(
                        2, int(round(source_height * scale / 2.0)) * 2
                    )
                    target_width = max(
                        2, int(round(source_width * scale / 2.0)) * 2
                    )
                    branch_video = resize_refinement_video_latent_spatial(
                        branch_video,
                        target_height=target_height,
                        target_width=target_width,
                    )
                    if previous_video is not None:
                        previous_video = resize_refinement_video_latent_spatial(
                            previous_video,
                            target_height=target_height,
                            target_width=target_width,
                        )
                    condition_video_latents = tuple(
                        resize_refinement_video_latent_spatial(
                            latent,
                            target_height=max(
                                2,
                                int(round(int(latent.shape[-2]) * scale / 2.0))
                                * 2,
                            ),
                            target_width=max(
                                2,
                                int(round(int(latent.shape[-1]) * scale / 2.0))
                                * 2,
                            ),
                        )
                        for latent in saved_condition_video_latents
                    )
                    if saved_reference_shapes:
                        reference_shapes = tuple(
                            tuple(int(value) for value in latent.shape[-3:])
                            for latent in condition_video_latents
                        )
                    layout = None

                def branch_predict(
                    video_value,
                    audio_value,
                    branch_clock,
                    *,
                    step_index,
                    is_actual_step,
                ):
                    # Approximate-attention policies are indexed by the formal
                    # trajectory, not by this branch's local 0..N counter.
                    routed_step = min(
                        range(request.steps),
                        key=lambda candidate: abs(
                            float(sigmas[candidate])
                            - float(branch_clock.video_sigma)
                        ),
                    )
                    return predict(
                        video_value,
                        audio_value,
                        branch_clock,
                        step_index=routed_step,
                        is_actual_step=is_actual_step,
                    )

                branch_sampler = (
                    TurboAVSampler(self.turbo_clock_mode)
                    if branch_uses_lora else ResMultistepAVSampler()
                )
                force_dense = (
                    request.preview_branch_force_dense
                    if branch_force_dense is None else branch_force_dense
                )
                with attention_force_dense(force_dense):
                    if branch_uses_lora:
                        branch_video, branch_audio = branch_sampler.sample(
                            branch_video,
                            branch_audio,
                            branch_plan,
                            branch_predict,
                            cancel_check=raise_if_cancelled,
                        )
                    else:
                        branch_video, branch_audio = branch_sampler.sample(
                            branch_video,
                            branch_audio,
                            branch_plan,
                            branch_predict,
                            cancel_check=raise_if_cancelled,
                            initial_previous_video=previous_video,
                            initial_previous_audio=previous_audio,
                            initial_previous_video_sigma=previous_video_sigma,
                            initial_previous_audio_sigma=previous_audio_sigma,
                        )
                if (
                    branch_uses_lora
                    and self.turbo_clock_mode is TurboClockMode.SHARED_VIDEO
                ):
                    branch_audio.div_(4.0)
                return branch_video, branch_audio
            finally:
                forecast = saved_forecast
                layout = saved_layout
                last_denoised = saved_denoised
                condition_video_latents = saved_condition_video_latents
                reference_shapes = saved_reference_shapes
                if active_lora_for_predict != saved_lora_mode:
                    set_active_lora(saved_lora_mode)
                    active_lora_for_predict = saved_lora_mode
                if (
                    saved_forecast is not None
                    and saved_forecast_record_count is not None
                ):
                    del saved_forecast.records[saved_forecast_record_count:]
                if saved_forecast is not None and saved_forecast_history is not None:
                    saved_forecast.history = saved_forecast_history
                del step_seconds[saved_step_count:]

        def save_formal_checkpoint(index, clock, step_video, step_audio) -> Path:
            """Atomically persist the exact formal state after one solver step."""

            if request.checkpoint_state_path is None:
                raise RuntimeError("checkpoint path is unavailable")
            checkpoint_path = Path(request.checkpoint_state_path).resolve()
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            online_runtime_state = None
            if request_online_budget is not None:
                checkpoint_verifier = getattr(
                    attention_backend,
                    "online_checkpoint_state",
                    None,
                )
                if checkpoint_verifier is None:
                    raise RuntimeError(
                        "active attention backend cannot checkpoint online state"
                    )
                online_runtime_state = {
                    "budget": request_online_budget.checkpoint_state(),
                    "verifier": checkpoint_verifier(request_online_budget),
                }
            document: dict[str, Any] = {
                "schema_version": 1,
                "representation": "formal_sampler_checkpoint_v1",
                "video": step_video.detach().cpu(),
                "audio": step_audio.detach().cpu(),
                "frames": request.frames,
                "fps": request.fps,
                "width": request.width,
                "height": request.height,
                "engine": request_engine,
                "seed": request.seed,
                "steps": request.steps,
                "step_index": index,
                "next_step_index": index + 1,
                "sigma": float(clock.video_sigma),
                "sigma_next": float(clock.video_sigma_next),
                "sigmas": list(simple_sigma_schedule(request.steps, 12.0)),
                "actual_step_indices": list(actual_steps),
                "attention_action_schedule": list(
                    request.attention_action_schedule
                ),
                "attention_online_guard_id": request.attention_online_guard_id,
                "attention_online_rebate_schedule": list(
                    request.attention_online_rebate_schedule
                ),
                "attention_online_budget_dense_layers": (
                    request.attention_online_budget_dense_layers
                ),
                "attention_online_runtime_state": online_runtime_state,
                "prompt_sha256": hashlib.sha256(
                    request.prompt.encode("utf-8")
                ).hexdigest(),
                "use_lora": request.use_lora,
                "forecast_state": (
                    None if forecast is None else forecast.checkpoint_state()
                ),
            }
            if not request.use_lora:
                if last_denoised is None:
                    raise RuntimeError("RES checkpoint has no denoised history")
                document.update({
                    "previous_video": last_denoised[0].detach().cpu(),
                    "previous_audio": last_denoised[1].detach().cpu(),
                    "previous_video_sigma": float(clock.video_sigma),
                    "previous_audio_sigma": float(clock.audio_sigma),
                })
            torch.save(document, temporary)
            temporary.replace(checkpoint_path)
            execution_profile["formal_checkpoint"] = {
                "checkpoint": str(checkpoint_path),
                "completed_steps": index + 1,
                "total_steps": request.steps,
                "sigma_next": float(clock.video_sigma_next),
                "formal_trajectory_mutated": False,
            }
            return checkpoint_path

        def publish_preview(index, clock, step_video, step_audio):
            """Decode, publish and optionally pause while the main state stays exact."""

            nonlocal dit, preview_latents, preview_published
            if request.preview_decode_mode == "direct_x0":
                if last_denoised is None:
                    raise RuntimeError(
                        "direct x0 preview requested before a DiT prediction exists"
                    )
                preview_video = last_denoised[0].clone()
                preview_audio = last_denoised[1].clone()
            else:
                preview_video, preview_audio = finish_preview_branch(
                    index, clock, step_video, step_audio
                )
            if (
                request.preview_decode_mode == "fast_finish"
                and request.preview_audio_branch_use_lora
            ):
                audio_branch_video, audio_branch_audio = finish_preview_branch(
                    index,
                    clock,
                    step_video,
                    step_audio,
                    branch_steps=request.preview_audio_branch_steps,
                    branch_spatial_scale=request.preview_audio_branch_spatial_scale,
                    branch_warm_history=False,
                    branch_force_dense=True,
                    branch_use_lora=True,
                )
                del audio_branch_video, preview_audio
                preview_audio = audio_branch_audio
            forecast_preview_latents = None
            if request.preview_forecast_steps > 0:
                forecast_video, forecast_audio = finish_preview_branch(
                    index,
                    clock,
                    step_video,
                    step_audio,
                    branch_steps=request.preview_forecast_steps,
                    branch_spatial_scale=1.0,
                    branch_warm_history=True,
                    branch_force_dense=True,
                    branch_use_lora=False,
                    branch_forecast_only=True,
                )
                forecast_preview_latents = (
                    forecast_video.detach().cpu(),
                    forecast_audio.detach().cpu(),
                )
                del forecast_video, forecast_audio
            preview_latents = (
                preview_video.detach().cpu(),
                preview_audio.detach().cpu(),
            )
            del preview_video, preview_audio
            execution_profile["intermediate_preview"] = {
                "step_index": index,
                "completed_sigma_positions": index + 1,
                "sigma": clock.video_sigma,
                "sigma_next": clock.video_sigma_next,
                "representation": (
                    "formal_step_x0_prediction"
                    if request.preview_decode_mode == "direct_x0"
                    else "isolated_fast_finish_branch"
                ),
                "decode_mode": request.preview_decode_mode,
                "branch_actual_steps": (
                    0
                    if request.preview_decode_mode == "direct_x0"
                    else request.preview_branch_steps
                    if request.preview_branch_actual_step_indices is None
                    else len(request.preview_branch_actual_step_indices)
                ),
                "branch_actual_step_indices": (
                    None
                    if request.preview_decode_mode == "direct_x0"
                    else request.preview_branch_actual_step_indices
                ),
                "branch_spatial_scale": request.preview_branch_spatial_scale,
                "branch_warm_history": request.preview_branch_warm_history,
                "branch_force_dense": request.preview_branch_force_dense,
                "branch_use_lora": request.preview_branch_use_lora,
                "audio_branch_use_lora": (
                    request.preview_decode_mode == "fast_finish"
                    and request.preview_audio_branch_use_lora
                ),
                "audio_branch_actual_steps": (
                    request.preview_audio_branch_steps
                    if request.preview_decode_mode == "fast_finish"
                    and request.preview_audio_branch_use_lora else None
                ),
                "audio_branch_spatial_scale": (
                    request.preview_audio_branch_spatial_scale
                    if request.preview_decode_mode == "fast_finish"
                    and request.preview_audio_branch_use_lora else None
                ),
                "video_source": "primary_preview_branch",
                "audio_source": (
                    "formal_step_x0_prediction"
                    if request.preview_decode_mode == "direct_x0"
                    else "lora_companion_branch"
                    if request.preview_audio_branch_use_lora
                    else "primary_preview_branch"
                ),
                "preview_width": int(preview_latents[0].shape[-1]) * 16,
                "preview_height": int(preview_latents[0].shape[-2]) * 16,
                "main_trajectory_mutated": False,
                "forecast_comparison_steps": request.preview_forecast_steps,
                "forecast_comparison_output": (
                    None
                    if preview_forecast_output is None
                    else str(preview_forecast_output)
                ),
            }
            if request.preview_latents_path is not None:
                preview_path = Path(request.preview_latents_path).resolve()
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "video": preview_latents[0], "audio": preview_latents[1],
                        "frames": request.frames, "fps": request.fps,
                        "width": int(preview_latents[0].shape[-1]) * 16,
                        "height": int(preview_latents[0].shape[-2]) * 16,
                        "engine": request_engine, "seed": request.seed,
                        "step_index": index, "sigma": clock.video_sigma,
                        "sigma_next": clock.video_sigma_next,
                        "representation": (
                            "formal_step_x0_prediction"
                            if request.preview_decode_mode == "direct_x0"
                            else "isolated_fast_finish_branch"
                        ),
                    },
                    preview_path,
                )

            # Evict DiT, decode the disposable branch, then restore the exact
            # same graph before the blocked main sampler resumes.
            self.transformer.move_to("cpu", non_blocking=False)
            self._clear_block_executor()
            self._release_device()
            self.video_vae.move_to("cuda:0", non_blocking=True)
            decoded_video = self.decode_video(
                self.video_vae.value, preview_latents[0].to("cuda:0"), request.frames
            )
            decoded_forecast_video = (
                None
                if forecast_preview_latents is None
                else self.decode_video(
                    self.video_vae.value,
                    forecast_preview_latents[0].to("cuda:0"),
                    request.frames,
                )
            )
            self.video_vae.move_to("cpu", non_blocking=False)
            self._release_device()
            self.audio_vae.move_to("cuda:0", non_blocking=True)
            decoded_audio = self.decode_audio(
                self.audio_vae.value, preview_latents[1].to("cuda:0")
            )
            decoded_forecast_audio = (
                None
                if forecast_preview_latents is None
                else self.decode_audio(
                    self.audio_vae.value,
                    forecast_preview_latents[1].to("cuda:0"),
                )
            )
            self.audio_vae.move_to("cpu", non_blocking=False)
            self._release_device()
            assert preview_output is not None
            AtomicPyAVMuxer(output_root=self.output_root).write(
                video=decoded_video, audio=decoded_audio,
                sample_rate=32000, fps=request.fps,
                output_path=preview_output, cancel_check=raise_if_cancelled,
            )
            if forecast_preview_latents is not None:
                assert preview_forecast_output is not None
                assert decoded_forecast_video is not None
                assert decoded_forecast_audio is not None
                AtomicPyAVMuxer(output_root=self.output_root).write(
                    video=decoded_forecast_video,
                    audio=decoded_forecast_audio,
                    sample_rate=32000,
                    fps=request.fps,
                    output_path=preview_forecast_output,
                    cancel_check=raise_if_cancelled,
                )
                del decoded_forecast_video, decoded_forecast_audio
            del decoded_video, decoded_audio
            forecast_preview_latents = None
            preview_latents = None
            preview_published = True
            if request.preview_ready_callback is not None:
                request.preview_ready_callback({
                    "output_path": str(preview_output),
                    "step_index": index,
                    "decode_mode": request.preview_decode_mode,
                    "branch_steps": (
                        0
                        if request.preview_decode_mode == "direct_x0"
                        else request.preview_branch_steps
                    ),
                    "spatial_scale": request.preview_branch_spatial_scale,
                    "audio_branch_use_lora": request.preview_audio_branch_use_lora,
                    "audio_branch_steps": (
                        request.preview_audio_branch_steps
                        if request.preview_audio_branch_use_lora else None
                    ),
                    "width": int(execution_profile["intermediate_preview"]["preview_width"]),
                    "height": int(execution_profile["intermediate_preview"]["preview_height"]),
                    "main_trajectory_mutated": False,
                    "forecast_steps": request.preview_forecast_steps,
                    "forecast_output_path": (
                        None
                        if preview_forecast_output is None
                        else str(preview_forecast_output)
                    ),
                })
            if request.preview_decision_wait is not None:
                decision = request.preview_decision_wait()
                if decision != "continue":
                    raise HotSessionCancelled("preview branch discarded")
            raise_if_cancelled()
            if request.checkpoint_after_step is None:
                dit = self._activate_transformer(execution_plan)

        def debug_step(index, clock, step_video, step_audio):
            progress(
                20 + 58 * (index + 1) / request.steps,
                "denoise",
                f"DiT 去噪 {index + 1}/{request.steps}",
            )
            checkpoint_now = request.checkpoint_after_step == index + 1
            if checkpoint_now:
                save_formal_checkpoint(index, clock, step_video, step_audio)
            if request.preview_step_index == index:
                publish_preview(index, clock, step_video, step_audio)
            if checkpoint_now:
                raise _HotSessionCheckpointReached
            if self.debug_step_dir is None:
                return
            if last_denoised is None:
                raise RuntimeError("debug callback ran before a model prediction")
            self.debug_step_dir.mkdir(parents=True, exist_ok=True)
            denoised_video, denoised_audio = last_denoised
            torch.save(
                {
                    "x": torch.cat(
                        (step_video.flatten(1), step_audio.flatten(1)), dim=-1
                    ).detach().cpu(),
                    "denoised": torch.cat(
                        (
                            denoised_video.flatten(1),
                            denoised_audio.flatten(1),
                        ),
                        dim=-1,
                    ).detach().cpu(),
                    "sigma": clock.video_sigma,
                    "sigma_next": clock.video_sigma_next,
                },
                self.debug_step_dir / f"native_step_{index:02d}.pt",
            )

        def sample():
            sampler = (
                TurboAVSampler(self.turbo_clock_mode)
                if self._uses_turbo_sampler(request)
                else ResMultistepAVSampler()
            )
            result_video, result_audio = sampler.sample(
                video,
                audio,
                plan,
                predict,
                callback=debug_step,
                transition=(
                    transition_multiscale
                    if request.multiscale_resize_after_step is not None
                    else None
                ),
                initial_previous_video=resume_previous_video,
                initial_previous_audio=resume_previous_audio,
                initial_previous_video_sigma=resume_previous_video_sigma,
                initial_previous_audio_sigma=resume_previous_audio_sigma,
            )
            if (
                self._uses_turbo_sampler(request)
                and self.turbo_clock_mode is TurboClockMode.SHARED_VIDEO
            ):
                result_audio.div_(4.0)
            return result_video, result_audio

        torch.cuda.synchronize()
        denoise_started = time.perf_counter()
        try:
            video, audio = sample()
        except _HotSessionCheckpointReached:
            torch.cuda.synchronize()
            phases["denoise"] = time.perf_counter() - denoise_started
            if request_online_budget is not None:
                execution_profile["attention_online_guard"] = (
                    request_online_budget.telemetry()
                )
            del context, text_tags, layout, dit
            del condition_video_latents, condition_audio_latents
            self.transformer.move_to("cpu", non_blocking=False)
            self._clear_block_executor()
            self._release_device()
            self._release_request_host_scratch()
            return HotSessionCheckpointResult(
                checkpoint_path=(
                    None
                    if request.checkpoint_state_path is None
                    else Path(request.checkpoint_state_path).resolve()
                ),
                preview_path=(
                    preview_output if preview_published else None
                ),
                completed_steps=int(request.checkpoint_after_step or 0),
                total_steps=request.steps,
                total_seconds=time.perf_counter() - started_total,
                phases=phases,
                step_seconds=tuple(step_seconds),
                execution_profile=execution_profile,
            )
        torch.cuda.synchronize()
        phases["denoise"] = time.perf_counter() - denoise_started
        if request_online_budget is not None:
            execution_profile["attention_online_guard"] = (
                request_online_budget.telemetry()
            )
        execution_profile["self_speculative_verifier"] = {
            "mode": "whole_dit_draft_verify_rollback",
            "verify_steps": list(self.self_speculative_verify_steps),
            "threshold": (
                self.self_speculative_verify_threshold
                if math.isfinite(self.self_speculative_verify_threshold)
                else None
            ),
            "records": self_speculative_records,
        }
        if request.terminal_refinement_initial_width is not None:
            assert terminal_video_noise_cpu is not None
            assert terminal_audio_noise_cpu is not None
            clean_audio = audio
            video = resize_refinement_video_latent_spatial(
                video,
                target_height=video_shape[-2],
                target_width=video_shape[-1],
            )
            motion_video = video
            total_refinement_steps = int(
                request.terminal_refinement_steps
                / request.terminal_refinement_denoise
            )
            total_refinement_steps = max(
                total_refinement_steps, request.terminal_refinement_steps
            )
            full_refinement_sigmas = simple_sigma_schedule(
                total_refinement_steps, 12.0
            )
            refinement_sigmas = full_refinement_sigmas[
                -(request.terminal_refinement_steps + 1) :
            ]
            sigma_start = float(refinement_sigmas[0])
            video = (
                sigma_start * terminal_video_noise_cpu.to("cuda:0")
                + (1.0 - sigma_start) * video.float()
            )
            refinement_audio = (
                sigma_start * terminal_audio_noise_cpu.to("cuda:0")
                + (1.0 - sigma_start) * clean_audio.float()
            )
            del terminal_video_noise_cpu, terminal_audio_noise_cpu
            if forecast is not None:
                forecast_profile_override = forecast.export()
            forecast = None
            layout = None
            refinement_plan = SamplingPlan(
                sampler="res_multistep",
                video_sigmas=refinement_sigmas,
                audio_sigmas=refinement_sigmas,
                actual_step_indices=tuple(
                    range(request.terminal_refinement_steps)
                ),
                video_shift=12.0,
                audio_shift=3.0,
            )

            def terminal_predict(
                video_value,
                audio_value,
                clock,
                *,
                step_index,
                is_actual_step,
            ):
                # Route the correction through the protected final solver
                # positions.  For the current one-step preset this is step 19,
                # which is dense even when the motion stage uses sparse MTCR.
                routed_index = (
                    request.steps
                    - request.terminal_refinement_steps
                    + step_index
                )
                dense_start = (
                    request.terminal_refinement_steps
                    - request.terminal_refinement_dense_tail_steps
                )
                if step_index >= dense_start:
                    with attention_force_dense():
                        return predict(
                            video_value,
                            audio_value,
                            clock,
                            step_index=routed_index,
                            is_actual_step=True,
                        )
                return predict(
                    video_value,
                    audio_value,
                    clock,
                    step_index=routed_index,
                    is_actual_step=True,
                )

            def refine_terminal():
                return ResMultistepAVSampler().sample(
                    video,
                    refinement_audio,
                    refinement_plan,
                    terminal_predict,
                    cancel_check=raise_if_cancelled,
                )

            video, discarded_audio = self._timed(
                phases, "terminal_refinement", refine_terminal
            )
            del discarded_audio
            video = blend_terminal_refinement_detail(
                motion_video,
                video,
                source_height=initial_video_shape[-2],
                source_width=initial_video_shape[-1],
                low_frequency_gain=(
                    request.terminal_refinement_low_frequency_gain
                ),
                temporal_lowpass=(
                    request.terminal_refinement_temporal_lowpass
                ),
                temporal_outlier_only=(
                    request.terminal_refinement_temporal_outlier_only
                ),
            )
            del motion_video
            audio = clean_audio
            execution_profile["terminal_refinement"] = {
                "source_geometry": [
                    request.terminal_refinement_initial_width,
                    request.terminal_refinement_initial_height,
                ],
                "target_geometry": [request.width, request.height],
                "steps": request.terminal_refinement_steps,
                "dense_tail_steps": request.terminal_refinement_dense_tail_steps,
                "denoise": request.terminal_refinement_denoise,
                "low_frequency_gain": (
                    request.terminal_refinement_low_frequency_gain
                ),
                "temporal_lowpass": (
                    request.terminal_refinement_temporal_lowpass
                ),
                "temporal_outlier_only": (
                    request.terminal_refinement_temporal_outlier_only
                ),
                "video_sigmas": list(refinement_sigmas),
                "attention_route": "recovery_sparse_then_dense_tail",
                "preserve_motion_stage_audio": True,
                "intermediate_decode": False,
            }
        if preserved_refinement_audio is not None:
            del audio
            audio = preserved_refinement_audio
        terminal_latent_guard = stabilize_terminal_video_latent_(video)
        execution_profile["terminal_latent_guard"] = terminal_latent_guard
        final_latents_path = (
            request.save_final_latents_path or self.debug_final_latents_path
        )
        if final_latents_path is not None:
            final_latents_path = Path(final_latents_path).resolve()
            final_latents_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "video": video.detach().cpu(),
                    "audio": audio.detach().cpu(),
                    "frames": request.frames,
                    "fps": request.fps,
                    "width": request.width,
                    "height": request.height,
                    "engine": request_engine,
                    "seed": request.seed,
                },
                final_latents_path,
            )
        del (
            context,
            text_tags,
            layout,
            dit,
            condition_video_latents,
            condition_audio_latents,
        )
        self._timed(
            phases,
            "dit_evict",
            lambda: self.transformer.move_to("cpu", non_blocking=False),
        )
        self._clear_block_executor()
        self._release_device()
        raise_if_cancelled()

        progress(80, "video_decode", "视频解码")
        self._timed(
            phases,
            "video_vae_h2d",
            lambda: self.video_vae.move_to("cuda:0", non_blocking=True),
        )
        if execution_plan is not None and execution_plan.vae_spatial_tile is not None:
            tile_height, tile_width = execution_plan.vae_spatial_tile
            if tile_height != tile_width:
                raise ValueError("the current H3 Video-VAE supports square tiles only")
            video_vae_model = self.video_vae.value
            if not hasattr(video_vae_model, "decoder_tile_size"):
                raise TypeError("the H3 Video-VAE does not expose decoder_tile_size")
            video_vae_model.decoder_tile_size = tile_height
        from .adapters.vae_tiling import configure_vae_tile_batching
        from .adapters.vae_compile import (
            transformer_block_compile,
            transformer_block_compile_ready,
        )

        configure_vae_tile_batching(
            self.video_vae.value,
            1 if execution_plan is None else execution_plan.vae_tile_batch_size,
        )
        compile_vae_block = bool(
            execution_plan is not None
            and execution_plan.vae_transformer_block_compile
        )
        if compile_vae_block and not transformer_block_compile_ready(
            self.video_vae.value
        ):
            raise RuntimeError(
                "execution plan requested VAE TransformerBlock compilation, "
                "but the session did not prebuild the compiled decoder"
            )
        with transformer_block_compile(compile_vae_block):
            decoded_video = self._timed(
                phases,
                "video_decode",
                lambda: self.decode_video(
                    self.video_vae.value, video, request.frames
                ),
            )
            decoded_preview_video = (
                None
                if preview_latents is None or preview_published
                else self._timed(
                    phases,
                    "preview_video_decode",
                    lambda: self.decode_video(
                        self.video_vae.value,
                        preview_latents[0].to("cuda:0"),
                        request.frames,
                    ),
                )
            )
        del video
        self._timed(
            phases,
            "video_vae_evict",
            lambda: self.video_vae.move_to("cpu", non_blocking=False),
        )
        self._release_device()
        self._timed(
            phases,
            "video_host_scratch_release",
            self._release_request_host_scratch,
        )
        raise_if_cancelled()

        progress(94, "audio_decode", "音频解码")
        self._timed(
            phases,
            "audio_vae_h2d",
            lambda: self.audio_vae.move_to("cuda:0", non_blocking=True),
        )
        decoded_audio = self._timed(
            phases,
            "audio_decode",
            lambda: self.decode_audio(self.audio_vae.value, audio),
        )
        decoded_preview_audio = (
            None
            if preview_latents is None or preview_published
            else self._timed(
                phases,
                "preview_audio_decode",
                lambda: self.decode_audio(
                    self.audio_vae.value, preview_latents[1].to("cuda:0")
                ),
            )
        )
        preview_latents = None
        del audio
        self._timed(
            phases,
            "audio_vae_evict",
            lambda: self.audio_vae.move_to("cpu", non_blocking=False),
        )
        self._release_device()
        self._timed(
            phases,
            "audio_host_scratch_release",
            self._release_request_host_scratch,
        )

        progress(98, "mux", "封装音视频")
        def mux():
            return AtomicPyAVMuxer(output_root=self.output_root).write(
                video=decoded_video,
                audio=decoded_audio,
                sample_rate=32000,
                fps=request.fps,
                output_path=output,
                cancel_check=raise_if_cancelled,
            )

        self._timed(phases, "mux", mux)
        if preview_output is not None and not preview_published:
            if decoded_preview_video is None or decoded_preview_audio is None:
                raise RuntimeError("requested intermediate preview was not captured")

            def mux_preview():
                return AtomicPyAVMuxer(output_root=self.output_root).write(
                    video=decoded_preview_video,
                    audio=decoded_preview_audio,
                    sample_rate=32000,
                    fps=request.fps,
                    output_path=preview_output,
                    cancel_check=raise_if_cancelled,
                )

            self._timed(phases, "preview_mux", mux_preview)
            del decoded_preview_video, decoded_preview_audio
        del decoded_video, decoded_audio
        self._timed(
            phases, "host_scratch_release", self._release_request_host_scratch
        )
        return HotSessionResult(
            output_path=output,
            total_seconds=time.perf_counter() - started_total,
            phases=phases,
            step_seconds=tuple(step_seconds),
            forecast_profile=(
                forecast_profile_override
                if forecast_profile_override is not None
                else forecast.export()
                if forecast is not None
                else {
                    "schema_version": 1,
                    "mode": "disabled",
                    "planned_actual_steps": list(actual_steps),
                    "actual_steps": request.steps,
                    "forecast_steps": 0,
                    "records": [],
                }
            ),
            execution_profile=execution_profile,
        )

    def close(self) -> None:
        self._clear_block_executor()
        for component in (self.transformer, self.video_vae, self.audio_vae):
            component.move_to("cpu", non_blocking=False)
        self._reference_latent_cache = None
        self._release_device(collect_cycles=True)


__all__ = [
    "HotSessionCancelled",
    "HotSessionCheckpointResult",
    "HotSessionRequest",
    "HotSessionResult",
    "NativeT2AVHotSession",
]
