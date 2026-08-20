"""Depth-3, locally confidence-damped tail predictor for directional forecast."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from layout import TargetLayout, target_layout


# The production schedule bounds every forecast streak
# to three steps.  The quality mechanism is the depth-3 local predictor below;
# this schedule is its fail-safer operating envelope, not the contribution by
# itself.
DEFAULT_ACTUAL_STEPS = frozenset({0, 1, 2, 3, 4, 8, 12, 16, 19})


@dataclass(slots=True)
class _TailHistory:
    anchor_residual_sample: torch.Tensor
    tail_residual_host: torch.Tensor


def _directional_gamma(
    current: torch.Tensor,
    older: torch.Tensor,
    newer: torch.Tensor,
    *,
    layout: TargetLayout | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return a smooth local extrapolation field in [1, 1.35].

    A previous global magnitude ratio reached its clamp even
    when local motion changed direction, blending stale and future contours.
    directional forecast extrapolates only where current depth-3 motion agrees with the last
    actual motion.  Disagreeing regions fall back to the newest actual tail.
    """

    if current.shape != older.shape or current.shape != newer.shape or current.ndim != 2:
        raise ValueError("directional forecast directional inputs must be matching [rows, channels]")
    history_delta = newer.float() - older.float()
    current_delta = current.float() - newer.float()

    if layout is None:
        dot = (history_delta * current_delta).sum().reshape(1, 1)
        history_energy = history_delta.square().sum().reshape(1, 1)
        current_energy = current_delta.square().sum().reshape(1, 1)
        cosine = dot / (history_energy * current_energy).sqrt().clamp_min(1.0e-8)
        extra = (current_energy / history_energy.clamp_min(1.0e-8)).sqrt().clamp(0.0, 0.35)
        confidence = cosine.clamp(0.0, 1.0).square()
        gamma = 1.0 + extra * confidence
    else:
        if current.shape[0] != layout.video_rows:
            raise ValueError("directional forecast video rows do not match the packed layout")
        shape = (layout.latent_t, layout.grid_h, layout.grid_w, current.shape[1])
        history_delta = history_delta.reshape(shape)
        current_delta = current_delta.reshape(shape)
        dot_rows = (history_delta * current_delta).sum(dim=-1)
        history_rows = history_delta.square().sum(dim=-1)
        current_rows = current_delta.square().sum(dim=-1)
        tile_size = (
            min(4, layout.latent_t),
            min(4, layout.grid_h),
            min(4, layout.grid_w),
        )

        def pool(value: torch.Tensor) -> torch.Tensor:
            return F.adaptive_avg_pool3d(value[None, None], tile_size)

        dot = pool(dot_rows)
        history_energy = pool(history_rows)
        current_energy = pool(current_rows)
        cosine = dot / (history_energy * current_energy).sqrt().clamp_min(1.0e-8)
        extra = (current_energy / history_energy.clamp_min(1.0e-8)).sqrt().clamp(0.0, 0.35)
        confidence = cosine.clamp(0.0, 1.0).square()
        tile_gamma = 1.0 + extra * confidence
        gamma = F.interpolate(
            tile_gamma,
            size=(layout.latent_t, layout.grid_h, layout.grid_w),
            mode="trilinear",
            align_corners=False,
        ).reshape(layout.video_rows, 1)

    values = torch.stack((
        gamma.min(), gamma.mean(), gamma.max(),
        (confidence < 0.25).float().mean(),
    )).detach().cpu().tolist()
    return gamma, {
        "gamma_min": float(values[0]),
        "gamma_mean": float(values[1]),
        "gamma_max": float(values[2]),
        "conservative_fraction": float(values[3]),
    }


class DirectionalForecastController:
    def __init__(
        self,
        *,
        sample_channels: int = 32,
        actual_steps: frozenset[int] | set[int] | None = None,
    ) -> None:
        self.sample_channels = max(4, int(sample_channels))
        self.actual_steps = frozenset(
            DEFAULT_ACTUAL_STEPS if actual_steps is None else actual_steps
        )
        self._indices: dict[tuple[int, str], torch.Tensor] = {}
        self.reset(0, 0)

    def reset(self, run_id: int, total_steps: int) -> None:
        self.run_id = int(run_id)
        self.total_steps = int(total_steps)
        self.history: list[_TailHistory] = []
        self.records: list[dict[str, Any]] = []
        self.cache_release_count = 0

    def needs_cache_release(self) -> bool:
        required = bool(self.records and self.records[-1]["mode"] == "forecast")
        if required:
            self.cache_release_count += 1
        return required

    def sample(self, feature: torch.Tensor) -> torch.Tensor:
        count = min(self.sample_channels, int(feature.shape[1]))
        key = (int(feature.shape[1]), str(feature.device))
        indices = self._indices.get(key)
        if indices is None or indices.device != feature.device or indices.numel() != count:
            indices = torch.arange(count, device=feature.device, dtype=torch.long)
            self._indices[key] = indices
        return feature.detach().index_select(1, indices).clone()

    def should_forecast(self, step_id: int) -> bool:
        return int(step_id) not in self.actual_steps and len(self.history) >= 2

    def predict(
        self,
        step_id: int,
        anchor_full: torch.Tensor,
        input_sample: torch.Tensor,
        anchor_sample: torch.Tensor,
        packed_layout: Any,
    ) -> torch.Tensor:
        started = time.perf_counter()
        info = target_layout(packed_layout)
        current = anchor_sample - input_sample
        older, newer = self.history[-2:]
        split = info.audio_rows
        gamma_audio, audio_stats = _directional_gamma(
            current[:split],
            older.anchor_residual_sample[:split],
            newer.anchor_residual_sample[:split],
        )
        gamma_video, video_stats = _directional_gamma(
            current[split:],
            older.anchor_residual_sample[split:],
            newer.anchor_residual_sample[split:],
            layout=info,
        )
        gamma_rows = torch.cat((
            gamma_audio.expand(split, 1),
            gamma_video,
        ), dim=0).to(dtype=anchor_full.dtype)

        # Reuse the depth-3 anchor as the output buffer and stream histories one
        # at a time. addcmul avoids materialising a full hidden-width blend.
        device = anchor_full.device
        predicted = anchor_full
        old = older.tail_residual_host.to(device=device, non_blocking=True)
        predicted.addcmul_(old, 1.0 - gamma_rows)
        del old
        new = newer.tail_residual_host.to(device=device, non_blocking=True)
        predicted.addcmul_(new, gamma_rows)
        del new
        del gamma_rows

        self.records.append({
            "step_id": int(step_id),
            "mode": "forecast",
            "anchor_depth": 3,
            "audio": audio_stats,
            "video": video_stats,
            "prediction_seconds": time.perf_counter() - started,
        })
        return predicted.unsqueeze(0)

    def observe_actual(
        self,
        step_id: int,
        input_sample: torch.Tensor,
        anchor_sample: torch.Tensor,
        tail_residual_host: torch.Tensor,
    ) -> None:
        residual_sample = anchor_sample - input_sample
        if tail_residual_host.device.type != "cpu" or not tail_residual_host.is_pinned():
            raise ValueError("directional forecast tail history must be pinned CPU memory")
        self.history.append(_TailHistory(residual_sample, tail_residual_host))
        self.history = self.history[-2:]
        self.records.append({"step_id": int(step_id), "mode": "actual", "anchor_depth": 3})

    def export(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "mode": "depth3_local_directional",
            "run_id": self.run_id,
            "total_steps": self.total_steps,
            "planned_actual_steps": sorted(self.actual_steps),
            "actual_steps": sum(r["mode"] == "actual" for r in self.records),
            "forecast_steps": sum(r["mode"] == "forecast" for r in self.records),
            "cache_release_policy": "after_forecast_only",
            "cache_release_count": self.cache_release_count,
            "predictor_math_changed": False,
            "records": list(self.records),
        }
