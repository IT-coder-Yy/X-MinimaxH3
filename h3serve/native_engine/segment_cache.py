"""Coordinate-aligned residual reuse for a contiguous H3 block segment.

Unlike temporal frame interpolation, this cache never mixes different video
frames or spatial locations.  It predicts one block-segment residual at the
same packed token coordinate from two previous dense observations.  All rows
(text, conditions, audio and video) remain present and the blocks before and
after the cached segment still execute exactly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class SegmentResidualCacheConfig:
    layer_start: int
    layer_stop: int
    reuse_steps: tuple[int, ...]
    transfer_chunk_rows: int = 4096
    directional_trust: bool = False
    directional_sample_channels: int = 32
    directional_max_extra: float = 0.35
    directional_min_cosine: float = 0.25
    protected_refresh: bool = False
    active_video_ratio: float = 0.0
    dynamic_video_budget: bool = False
    active_video_min_ratio: float = 0.0
    innovation_risk_coverage: float = 0.80
    innovation_max_relative: float = 4.0
    active_query_block: int = 128
    active_layer_start: int = 0
    active_layer_stop: int = 0
    sequential_layer_groups: bool = False
    sequential_conservative_hold: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.layer_start < self.layer_stop <= 50:
            raise ValueError("segment cache layer range must lie inside [0, 50]")
        if tuple(sorted(set(self.reuse_steps))) != self.reuse_steps:
            raise ValueError("segment cache reuse steps must be sorted and unique")
        if any(step < 0 for step in self.reuse_steps):
            raise ValueError("segment cache reuse steps cannot be negative")
        if self.transfer_chunk_rows <= 0:
            raise ValueError("segment cache transfer chunk must be positive")
        if self.directional_sample_channels <= 0:
            raise ValueError("directional sample channels must be positive")
        if not 0.0 <= self.directional_max_extra <= 1.0:
            raise ValueError("directional max extra must lie inside [0, 1]")
        if not -1.0 <= self.directional_min_cosine <= 1.0:
            raise ValueError("directional minimum cosine must lie inside [-1, 1]")
        if not 0.0 <= self.active_video_ratio <= 1.0:
            raise ValueError("active video ratio must lie inside [0, 1]")
        if self.active_video_ratio and not self.protected_refresh:
            raise ValueError("active video refresh requires protected refresh")
        if self.dynamic_video_budget and not self.active_video_ratio:
            raise ValueError("dynamic video budgeting requires a non-zero maximum ratio")
        if not 0.0 <= self.active_video_min_ratio <= self.active_video_ratio:
            raise ValueError(
                "active video minimum ratio must lie inside [0, maximum ratio]"
            )
        if not 0.0 < self.innovation_risk_coverage <= 1.0:
            raise ValueError("innovation risk coverage must lie inside (0, 1]")
        if self.innovation_max_relative <= 0.0:
            raise ValueError("innovation relative-risk limit must be positive")
        if self.active_query_block <= 0:
            raise ValueError("active query block must be positive")
        if not 0 <= self.active_layer_start <= self.active_layer_stop <= 50:
            raise ValueError("active video layer range must lie inside [0, 50]")
        has_active_layer_range = self.active_layer_start < self.active_layer_stop
        if has_active_layer_range and not self.active_video_ratio:
            raise ValueError("active video layer range requires a non-zero video ratio")
        if has_active_layer_range and not (
            self.layer_start
            <= self.active_layer_start
            < self.active_layer_stop
            <= self.layer_stop
        ):
            raise ValueError(
                "active video layer range must lie inside the segment cache range"
            )
        if self.sequential_layer_groups and not (
            self.protected_refresh
            and self.active_video_ratio
            and has_active_layer_range
        ):
            raise ValueError(
                "sequential layer groups require protected refresh, active video, "
                "and an explicit active layer range"
            )
        if self.sequential_conservative_hold and not self.sequential_layer_groups:
            raise ValueError(
                "sequential conservative hold requires sequential layer groups"
            )


@dataclass(slots=True)
class _ResidualObservation:
    step_index: int
    input_sample_host: torch.Tensor
    residual_host: torch.Tensor


class CoordinateAlignedSegmentCache:
    """Predict a segment update without changing packed-token identity."""

    def __init__(self, config: SegmentResidualCacheConfig) -> None:
        self.config = config
        self.history: list[_ResidualObservation] = []
        self.group_histories: dict[
            tuple[int, int], list[_ResidualObservation]
        ] = {}
        self.records: list[dict[str, Any]] = []

    def _layer_groups(self) -> tuple[tuple[int, int], ...]:
        boundaries = {
            self.config.layer_start,
            self.config.active_layer_start,
            self.config.active_layer_stop,
            self.config.layer_stop,
        }
        ordered = sorted(
            value
            for value in boundaries
            if self.config.layer_start <= value <= self.config.layer_stop
        )
        return tuple(
            (start, stop)
            for start, stop in zip(ordered, ordered[1:])
            if start < stop
        )

    @staticmethod
    def _run_range(stack, value, *, start: int, stop: int, kwargs):
        if callable(getattr(stack, "run_range", None)):
            return stack.run_range(value, start=start, stop=stop, **kwargs)
        for block in stack.blocks[start:stop]:
            value = block(value, **kwargs)
        return value

    @staticmethod
    def _run_protected_range(
        stack,
        value,
        *,
        start: int,
        stop: int,
        protected_tokens: int,
        active_video_indices: torch.Tensor | None,
        active_video_layer_start: int,
        active_video_layer_stop: int,
        kwargs,
    ):
        runner = getattr(stack, "run_protected_range", None)
        if not callable(runner):
            raise RuntimeError("block stack does not support protected refresh")
        return runner(
            value,
            start=start,
            stop=stop,
            protected_tokens=protected_tokens,
            active_video_indices=active_video_indices,
            active_video_layer_start=active_video_layer_start,
            active_video_layer_stop=active_video_layer_stop,
            **kwargs,
        )

    def _select_active_video_blocks(
        self,
        value: torch.Tensor,
        *,
        protected_tokens: int,
        video_shape: tuple[int, int, int],
        alpha: float,
        history: list[_ResidualObservation] | None = None,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        ratio = self.config.active_video_ratio
        if ratio <= 0.0:
            return None, {}
        observations = self.history if history is None else history
        if not observations:
            raise RuntimeError("active video routing requires one dense observation")
        latent_t, grid_h, grid_w = video_shape
        video_rows = int(value.shape[0]) - protected_tokens
        if video_rows != latent_t * grid_h * grid_w:
            raise ValueError("active video routing shape does not match packed rows")
        channels = int(observations[-1].input_sample_host.shape[1])
        current = value[protected_tokens:, :channels].detach().float()
        older, newer = observations
        previous = newer.input_sample_host[protected_tokens:].to(
            device=value.device, non_blocking=False
        )
        if self.config.dynamic_video_budget:
            old = older.input_sample_host[protected_tokens:].to(
                device=value.device, non_blocking=False
            )
            trajectory = previous - old
            expected = old.lerp(previous, alpha)
            innovation = current - expected
            risk = innovation.square().mean(dim=1)
            relative_innovation = float(
                (
                    innovation.square().mean()
                    / trajectory.square().mean().clamp_min(1.0e-8)
                )
                .detach()
                .cpu()
            )
            innovation_trust_feasible = (
                relative_innovation <= self.config.innovation_max_relative
            )
        else:
            risk = (current - previous).square().mean(dim=1)
            relative_innovation = 0.0
            innovation_trust_feasible = True
        # A small local average prevents isolated numerical spikes from
        # stealing the budget and makes selected query blocks track coherent
        # motion/structure regions rather than individual tokens.
        risk_3d = risk.reshape(1, 1, latent_t, grid_h, grid_w)
        kernel = tuple(3 if size >= 3 else 1 for size in video_shape)
        padding = tuple(size // 2 for size in kernel)
        smoothed = torch.nn.functional.avg_pool3d(
            risk_3d,
            kernel_size=kernel,
            stride=1,
            padding=padding,
            count_include_pad=False,
        ).reshape(-1)
        block = self.config.active_query_block
        block_count = (video_rows + block - 1) // block
        padded = torch.nn.functional.pad(
            smoothed, (0, block_count * block - video_rows), value=0.0
        )
        block_risk = padded.reshape(block_count, block).mean(dim=1)
        maximum_count = max(1, round(block_count * ratio))
        maximum_count = min(maximum_count, block_count)
        minimum_count = max(
            1, round(block_count * self.config.active_video_min_ratio)
        )
        minimum_count = min(minimum_count, maximum_count)
        selected_count = maximum_count
        coverage_achieved = 1.0
        innovation_budget_feasible = True
        risk_baseline = 0.0
        if self.config.dynamic_video_budget:
            risk_baseline_tensor = block_risk.median()
            excess = (block_risk - risk_baseline_tensor).clamp_min(0.0)
            sorted_excess = excess.sort(descending=True).values
            total_excess = sorted_excess.sum()
            risk_baseline = float(risk_baseline_tensor.detach().cpu())
            if float(total_excess.detach().cpu()) > 1.0e-8:
                target = total_excess * self.config.innovation_risk_coverage
                selected_count = int(
                    torch.searchsorted(
                        sorted_excess.cumsum(dim=0), target
                    ).detach().cpu()
                ) + 1
                selected_count = max(minimum_count, selected_count)
                if selected_count > maximum_count:
                    innovation_budget_feasible = False
                    selected_count = maximum_count
                coverage_achieved = float(
                    (sorted_excess[:selected_count].sum() / total_excess)
                    .detach()
                    .cpu()
                )
            else:
                selected_count = minimum_count
        selected_blocks = torch.topk(
            block_risk, k=selected_count, largest=True, sorted=False
        ).indices.sort().values
        relative = torch.cat(
            tuple(
                torch.arange(
                    int(block_index) * block,
                    min((int(block_index) + 1) * block, video_rows),
                    device=value.device,
                    dtype=torch.long,
                )
                for block_index in selected_blocks
            )
        )
        stats = {
            "active_video_ratio": float(relative.numel()) / float(video_rows),
            "active_video_tokens": float(relative.numel()),
            "risk_min": float(block_risk[selected_blocks].min().detach().cpu()),
            "risk_mean": float(block_risk[selected_blocks].mean().detach().cpu()),
            "risk_max": float(block_risk[selected_blocks].max().detach().cpu()),
            "dynamic_video_budget": self.config.dynamic_video_budget,
            "active_video_min_ratio": self.config.active_video_min_ratio,
            "innovation_risk_coverage_target": (
                self.config.innovation_risk_coverage
            ),
            "innovation_risk_coverage_achieved": coverage_achieved,
            "innovation_relative_energy": relative_innovation,
            "innovation_trust_feasible": innovation_trust_feasible,
            "innovation_budget_feasible": innovation_budget_feasible,
            "innovation_risk_baseline": risk_baseline,
        }
        return relative + protected_tokens, stats

    def _observe(
        self,
        *,
        step_index: int,
        before: torch.Tensor,
        after: torch.Tensor,
        history: list[_ResidualObservation] | None = None,
    ) -> None:
        sample_channels = min(
            self.config.directional_sample_channels, int(before.shape[1])
        )
        input_sample = before[:, :sample_channels].detach().to(
            device="cpu", dtype=torch.float32
        ).clone()
        # Reuse the temporary pre-segment clone as the delta buffer.  This
        # avoids allocating a second full [tokens, hidden] CUDA tensor.
        before.neg_().add_(after)
        pin = before.device.type == "cuda"
        host = torch.empty(
            before.shape,
            dtype=before.dtype,
            device="cpu",
            pin_memory=pin,
        )
        host.copy_(before, non_blocking=False)
        target = self.history if history is None else history
        target.append(_ResidualObservation(step_index, input_sample, host))
        del target[:-2]

    def _directional_alpha(
        self,
        value: torch.Tensor,
        *,
        older: _ResidualObservation,
        newer: _ResidualObservation,
        row_start: int = 0,
    ) -> tuple[float, dict[str, float]]:
        """Infer one bounded trust coefficient from the live feature trajectory.

        This intentionally samples channels but retains every packed row.  It
        is only a gate/coefficient estimator; accepted reuse still applies the
        cached residual at the identical token coordinate.
        """

        channels = int(older.input_sample_host.shape[1])
        current = value[row_start:, :channels].detach().float()
        device = current.device
        old = older.input_sample_host[row_start:].to(device=device, non_blocking=False)
        new = newer.input_sample_host[row_start:].to(device=device, non_blocking=False)
        history_delta = new - old
        current_delta = current - new
        dot = (history_delta * current_delta).sum()
        history_energy = history_delta.square().sum().clamp_min(1.0e-8)
        current_energy = current_delta.square().sum().clamp_min(1.0e-8)
        cosine = dot / (history_energy * current_energy).sqrt()
        cosine_value = float(cosine.detach().cpu())
        if cosine_value < self.config.directional_min_cosine:
            raise RuntimeError("segment residual direction is outside the trust region")
        relative_step = (current_energy / history_energy).sqrt()
        relative_value = float(relative_step.detach().cpu())
        confidence = max(0.0, min(1.0, cosine_value)) ** 2
        extra = min(relative_value, self.config.directional_max_extra) * confidence
        alpha = 1.0 + extra
        return alpha, {
            "directional_cosine": cosine_value,
            "directional_relative_step": relative_value,
            "directional_confidence": confidence,
        }

    def _prediction_alpha(
        self,
        value: torch.Tensor,
        step_index: int,
        *,
        row_start: int = 0,
        history: list[_ResidualObservation] | None = None,
    ) -> tuple[float, dict[str, float]]:
        observations = self.history if history is None else history
        if len(observations) < 2:
            raise RuntimeError("two dense segment observations are required")
        older, newer = observations
        stats: dict[str, float] = {}
        span = newer.step_index - older.step_index
        if span <= 0:
            raise RuntimeError("segment cache history is not chronological")
        alpha = float(step_index - older.step_index) / float(span)
        # Preserve the already validated step-index trust region.  The live
        # trajectory gate is an extension for requests outside that region,
        # not a replacement for stable early-step reuse.
        if not 0.0 <= alpha <= 2.0:
            if not self.config.directional_trust:
                raise RuntimeError(
                    "segment residual extrapolation is outside the safe range"
                )
            alpha, stats = self._directional_alpha(
                value, older=older, newer=newer, row_start=row_start
            )
            stats["directional_extended_safe_range"] = 1.0
        return alpha, stats

    def _apply_prediction(
        self,
        value: torch.Tensor,
        *,
        alpha: float,
        row_start: int = 0,
        preserved_indices: torch.Tensor | None = None,
        history: list[_ResidualObservation] | None = None,
    ) -> None:
        observations = self.history if history is None else history
        older, newer = observations
        preserved = (
            None
            if preserved_indices is None
            else value.index_select(0, preserved_indices).clone()
        )
        device = value.device
        chunk = self.config.transfer_chunk_rows
        for start in range(row_start, value.shape[0], chunk):
            stop = min(start + chunk, value.shape[0])
            old = older.residual_host[start:stop].to(
                device=device, non_blocking=device.type == "cuda"
            )
            new = newer.residual_host[start:stop].to(
                device=device, non_blocking=device.type == "cuda"
            )
            old.lerp_(new, alpha)
            value[start:stop].add_(old)
        if preserved is not None:
            value.index_copy_(0, preserved_indices, preserved)

    def _run_dense_layer_groups(
        self,
        stack,
        value: torch.Tensor,
        *,
        step_index: int,
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        for group in self._layer_groups():
            history = self.group_histories.setdefault(group, [])
            before = value.clone()
            value = self._run_range(
                stack, value, start=group[0], stop=group[1], kwargs=kwargs
            )
            self._observe(
                step_index=step_index,
                before=before,
                after=value,
                history=history,
            )
        return value

    def _run_sequential_prediction(
        self,
        stack,
        value: torch.Tensor,
        *,
        step_index: int,
        protected_tokens: int,
        video_shape: tuple[int, int, int],
        kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, float]]:
        """Compose predicted residuals in transformer layer order.

        The previous implementation predicted one whole-segment residual and
        then replaced selected video rows with a later-layer refresh computed
        from the stale segment input.  This path instead predicts each layer
        group in sequence, so the sensitive group sees the state produced by
        all preceding groups and selected rows retain only their true update
        for that group.
        """

        group_records: list[dict[str, Any]] = []
        active_stats: dict[str, float] = {}
        for start, stop in self._layer_groups():
            history = self.group_histories.setdefault((start, stop), [])
            try:
                alpha, trust = self._prediction_alpha(
                    value,
                    step_index,
                    row_start=protected_tokens,
                    history=history,
                )
            except RuntimeError as error:
                if not (
                    self.config.sequential_conservative_hold
                    and "direction is outside the trust region" in str(error)
                ):
                    raise
                # A low-confidence group must not force every other layer to
                # recompute. Reuse the latest same-coordinate residual without
                # extrapolation; this is more conservative than reducing the
                # cosine threshold or extending the trajectory.
                alpha = 1.0
                trust = {
                    "directional_conservative_hold": 1.0,
                    "directional_fallback_reason": str(error),
                }
            is_active_group = (
                self.config.active_layer_start <= start
                and stop <= self.config.active_layer_stop
            )
            active_indices = None
            local_active: dict[str, float] = {}
            if is_active_group:
                active_indices, local_active = self._select_active_video_blocks(
                    value,
                    protected_tokens=protected_tokens,
                    video_shape=video_shape,
                    alpha=alpha,
                    history=history,
                )
                if not local_active.get("innovation_trust_feasible", True):
                    raise RuntimeError(
                        "video innovation is outside the dynamic refresh trust region"
                    )
                if not local_active.get("innovation_budget_feasible", True):
                    raise RuntimeError(
                        "innovation risk is too diffuse for the active-video budget"
                    )
                active_stats = local_active
            value = self._run_protected_range(
                stack,
                value,
                start=start,
                stop=stop,
                protected_tokens=protected_tokens,
                active_video_indices=active_indices,
                active_video_layer_start=start if is_active_group else 0,
                active_video_layer_stop=stop if is_active_group else 0,
                kwargs=kwargs,
            )
            self._apply_prediction(
                value,
                alpha=alpha,
                row_start=protected_tokens,
                preserved_indices=active_indices,
                history=history,
            )
            group_records.append(
                {
                    "layer_start": start,
                    "layer_stop": stop,
                    "alpha": alpha,
                    "active_video_refresh": is_active_group,
                    **trust,
                    **local_active,
                }
            )
        return value, group_records, active_stats

    def run_actual_tail(
        self,
        stack,
        value: torch.Tensor,
        *,
        step_index: int,
        prefix_stop: int,
        protected_tokens: int,
        block_kwargs: dict[str, Any],
        video_shape: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        """Run one actual DiT step, caching or predicting only one segment."""

        if self.config.layer_start < prefix_stop:
            raise ValueError("segment cache overlaps the always-dense anchor prefix")
        if not 0 <= protected_tokens <= value.shape[0]:
            raise ValueError("protected token boundary lies outside packed rows")
        value = self._run_range(
            stack,
            value,
            start=prefix_stop,
            stop=self.config.layer_start,
            kwargs=block_kwargs,
        )
        requested_reuse = step_index in self.config.reuse_steps
        started = time.perf_counter()
        mode = "dense"
        alpha = None
        trust_stats: dict[str, float] = {}
        active_video_indices = None
        active_stats: dict[str, float] = {}
        group_records: list[dict[str, Any]] = []
        fallback_reason = None
        histories_ready = (
            all(
                len(self.group_histories.get(group, ())) >= 2
                for group in self._layer_groups()
            )
            if self.config.sequential_layer_groups
            else len(self.history) >= 2
        )
        if requested_reuse and histories_ready:
            segment_input = value.clone() if self.config.sequential_layer_groups else None
            try:
                row_start = protected_tokens if self.config.protected_refresh else 0
                if self.config.sequential_layer_groups:
                    if video_shape is None:
                        raise RuntimeError(
                            "sequential layer groups require the generated video shape"
                        )
                    value, group_records, active_stats = (
                        self._run_sequential_prediction(
                            stack,
                            value,
                            step_index=step_index,
                            protected_tokens=protected_tokens,
                            video_shape=video_shape,
                            kwargs=block_kwargs,
                        )
                    )
                    alpha = max(
                        float(group["alpha"]) for group in group_records
                    )
                    mode = "predicted"
                else:
                    alpha, trust_stats = self._prediction_alpha(
                        value, step_index, row_start=row_start
                    )
                    if self.config.protected_refresh:
                        if self.config.active_video_ratio and video_shape is None:
                            raise RuntimeError("active video routing requires video shape")
                        active_video_indices, active_stats = (
                            self._select_active_video_blocks(
                                value,
                                protected_tokens=protected_tokens,
                                video_shape=(
                                    (1, 1, value.shape[0] - protected_tokens)
                                    if video_shape is None
                                    else video_shape
                                ),
                                alpha=alpha,
                            )
                        )
                        if not active_stats.get("innovation_trust_feasible", True):
                            raise RuntimeError(
                                "video innovation is outside the dynamic refresh trust region"
                            )
                        if not active_stats.get("innovation_budget_feasible", True):
                            raise RuntimeError(
                                "innovation risk is too diffuse for the active-video budget"
                            )
                        value = self._run_protected_range(
                            stack,
                            value,
                            start=self.config.layer_start,
                            stop=self.config.layer_stop,
                            protected_tokens=protected_tokens,
                            active_video_indices=active_video_indices,
                            active_video_layer_start=(
                                self.config.active_layer_start
                                if self.config.active_layer_start
                                < self.config.active_layer_stop
                                else self.config.layer_start
                            ),
                            active_video_layer_stop=(
                                self.config.active_layer_stop
                                if self.config.active_layer_start
                                < self.config.active_layer_stop
                                else self.config.layer_stop
                            ),
                            kwargs=block_kwargs,
                        )
                    self._apply_prediction(
                        value,
                        alpha=alpha,
                        row_start=row_start,
                        preserved_indices=active_video_indices,
                    )
                    mode = "predicted"
            except RuntimeError as error:
                # Fail closed: an invalid history range executes real blocks.
                fallback_reason = str(error)
                if segment_input is not None:
                    value.copy_(segment_input)
                requested_reuse = False
        if not requested_reuse or mode == "dense":
            if self.config.sequential_layer_groups:
                value = self._run_dense_layer_groups(
                    stack,
                    value,
                    step_index=step_index,
                    kwargs=block_kwargs,
                )
            else:
                before = value.clone()
                value = self._run_range(
                    stack,
                    value,
                    start=self.config.layer_start,
                    stop=self.config.layer_stop,
                    kwargs=block_kwargs,
                )
                self._observe(step_index=step_index, before=before, after=value)
        segment_seconds = time.perf_counter() - started
        self.records.append(
            {
                "step_index": int(step_index),
                "mode": mode,
                "layer_start": self.config.layer_start,
                "layer_stop": self.config.layer_stop,
                "alpha": alpha,
                "protected_refresh": bool(
                    mode == "predicted" and self.config.protected_refresh
                ),
                "protected_tokens": (
                    int(protected_tokens)
                    if mode == "predicted" and self.config.protected_refresh
                    else 0
                ),
                **trust_stats,
                **active_stats,
                "sequential_layer_groups": self.config.sequential_layer_groups,
                "sequential_conservative_hold": (
                    self.config.sequential_conservative_hold
                ),
                "layer_group_records": group_records,
                "fallback_reason": fallback_reason,
                "segment_seconds": segment_seconds,
            }
        )
        return self._run_range(
            stack,
            value,
            start=self.config.layer_stop,
            stop=len(stack.blocks),
            kwargs=block_kwargs,
        )

    def export(self) -> dict[str, Any]:
        cache_bytes = sum(
            item.residual_host.numel() * item.residual_host.element_size()
            for item in self.history
        )
        cache_bytes += sum(
            item.residual_host.numel() * item.residual_host.element_size()
            for history in self.group_histories.values()
            for item in history
        )
        return {
            "schema_version": 1,
            "mode": "coordinate_aligned_segment_residual",
            "layer_start": self.config.layer_start,
            "layer_stop": self.config.layer_stop,
            "reuse_steps": list(self.config.reuse_steps),
            "directional_trust": self.config.directional_trust,
            "directional_max_extra": self.config.directional_max_extra,
            "directional_min_cosine": self.config.directional_min_cosine,
            "protected_refresh": self.config.protected_refresh,
            "active_video_ratio": self.config.active_video_ratio,
            "dynamic_video_budget": self.config.dynamic_video_budget,
            "active_video_min_ratio": self.config.active_video_min_ratio,
            "innovation_risk_coverage": self.config.innovation_risk_coverage,
            "innovation_max_relative": self.config.innovation_max_relative,
            "active_layer_start": (
                self.config.active_layer_start
                if self.config.active_layer_start < self.config.active_layer_stop
                else self.config.layer_start
            ),
            "active_layer_stop": (
                self.config.active_layer_stop
                if self.config.active_layer_start < self.config.active_layer_stop
                else self.config.layer_stop
            ),
            "active_query_block": self.config.active_query_block,
            "sequential_layer_groups": self.config.sequential_layer_groups,
            "sequential_conservative_hold": (
                self.config.sequential_conservative_hold
            ),
            "layer_groups": [list(group) for group in self._layer_groups()],
            "history_bytes": int(cache_bytes),
            "records": list(self.records),
        }


__all__ = ["CoordinateAlignedSegmentCache", "SegmentResidualCacheConfig"]
