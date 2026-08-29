"""Native depth-3 directional forecast for the pruned H3 block stack.

The controller mirrors the accepted V7-D mechanism without importing
ComfyUI or Spectrum.  Requested forecast steps execute blocks 0..2, then use
the two latest full-refresh tails to estimate the skipped blocks 3..49.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .model.packed import PackedLayout


LONG_FORECAST_HISTORY_PAGEABLE_BYTES = 2 * 1024**3
LONG_FORECAST_HISTORY_CHUNK_ROWS = 4096
V24_FORECAST_FEEDBACK_POLICY_ID = "v24_request_local_forecast_debt_v1"


@dataclass(frozen=True, slots=True)
class TargetLayout:
    start: int
    stop: int
    audio_rows: int
    video_rows: int
    latent_t: int
    grid_h: int
    grid_w: int


@dataclass(slots=True)
class _TailHistory:
    step_index: int
    anchor_residual_sample: torch.Tensor
    tail_residual_host: torch.Tensor
    streamed: bool = False
    chunk_rows: int = LONG_FORECAST_HISTORY_CHUNK_ROWS


@dataclass(frozen=True, slots=True)
class _ForecastCalibrationPoint:
    step_index: int
    audio_anchor: torch.Tensor
    video_anchor: torch.Tensor
    audio_tail: torch.Tensor
    video_tail: torch.Tensor


def _tensor_storage_bytes(value: torch.Tensor) -> int:
    return int(value.numel() * value.element_size())


def forecast_history_storage_mode(
    *,
    rows: int,
    channels: int,
    element_size: int,
    device_type: str,
) -> str:
    storage_bytes = int(rows) * int(channels) * int(element_size)
    if min(rows, channels, element_size) <= 0:
        raise ValueError("forecast history geometry must be positive")
    if (
        device_type == "cuda"
        and storage_bytes >= LONG_FORECAST_HISTORY_PAGEABLE_BYTES
    ):
        return "pageable_chunked"
    return "pinned_whole" if device_type == "cuda" else "pageable_whole"


def _stream_forecast_history(value: torch.Tensor) -> bool:
    """Keep multi-GiB Forecast history out of CUDA's pinned allocation pool."""

    return forecast_history_storage_mode(
        rows=int(value.shape[0]),
        channels=int(value.numel() // value.shape[0]),
        element_size=int(value.element_size()),
        device_type=value.device.type,
    ) == "pageable_chunked"


def _copy_forecast_history_to_host(
    value: torch.Tensor,
    *,
    streamed: bool,
    chunk_rows: int = LONG_FORECAST_HISTORY_CHUNK_ROWS,
) -> torch.Tensor:
    pin = value.device.type == "cuda" and not streamed
    host = torch.empty(
        value.shape,
        dtype=value.dtype,
        device="cpu",
        pin_memory=pin,
    )
    if not streamed:
        host.copy_(value.detach(), non_blocking=pin)
        return host
    for start in range(0, value.shape[0], chunk_rows):
        stop = min(start + chunk_rows, value.shape[0])
        host[start:stop].copy_(value.detach()[start:stop], non_blocking=False)
    return host


def _apply_forecast_history_terms_(
    predicted: torch.Tensor,
    histories: tuple[_TailHistory, ...],
    coefficients: tuple[torch.Tensor, ...],
) -> bool:
    """Accumulate host Forecast histories without materializing a full GPU tail."""

    if len(histories) != len(coefficients):
        raise ValueError("forecast histories and coefficients must have equal length")
    streamed = any(history.streamed for history in histories)
    device = predicted.device
    if not streamed:
        for history, coefficient in zip(histories, coefficients, strict=True):
            tail = history.tail_residual_host.to(
                device=device,
                non_blocking=device.type == "cuda",
            )
            predicted.addcmul_(tail, coefficient.to(dtype=predicted.dtype))
            del tail
        return False

    chunk_rows = min(history.chunk_rows for history in histories)
    for start in range(0, predicted.shape[0], chunk_rows):
        stop = min(start + chunk_rows, predicted.shape[0])
        predicted_chunk = predicted[start:stop]
        for history, coefficient in zip(histories, coefficients, strict=True):
            tail_chunk = history.tail_residual_host[start:stop].to(
                device=device,
                non_blocking=False,
            )
            predicted_chunk.addcmul_(
                tail_chunk,
                coefficient[start:stop].to(dtype=predicted.dtype),
            )
            del tail_chunk
    return True


def _curvature_predict(
    current: torch.Tensor,
    oldest: _ForecastCalibrationPoint | _TailHistory,
    older: _ForecastCalibrationPoint | _TailHistory,
    newer: _ForecastCalibrationPoint | _TailHistory,
    *,
    modality: str,
    layout: TargetLayout | None,
) -> torch.Tensor:
    """Third-order control-variate forecast of the expensive DiT tail.

    The first three H3 blocks are evaluated at every solver point.  Their
    residual supplies a cheap local clock for the expensive blocks 3..49.
    ``gamma - 1`` is the observed distance beyond the newest full refresh;
    the first difference is the existing V7-D secant and the second
    difference adds curvature without changing any model weight.
    """

    anchor_name = f"{modality}_anchor"
    tail_name = f"{modality}_tail"
    anchor0 = getattr(oldest, anchor_name, None)
    anchor1 = getattr(older, anchor_name, None)
    anchor2 = getattr(newer, anchor_name, None)
    tail0 = getattr(oldest, tail_name, None)
    tail1 = getattr(older, tail_name, None)
    tail2 = getattr(newer, tail_name, None)
    if anchor0 is None:
        # Runtime history stores the two modalities in one tensor.
        raise TypeError("curvature calibration points must expose modality tensors")
    gamma, _ = _directional_gamma(current, anchor1, anchor2, layout=layout)
    distance = gamma.float() - 1.0
    gap01 = float(max(1, older.step_index - oldest.step_index))
    gap12 = float(max(1, newer.step_index - older.step_index))
    dx = distance * gap12
    # Non-uniform backward Newton interpolation.  The full-refresh schedule is
    # irregular, so the familiar equal-step second difference is incorrect.
    slope01 = (tail1.float() - tail0.float()) / gap01
    slope12 = (tail2.float() - tail1.float()) / gap12
    curvature = (
        dx * (dx + gap12) / (gap01 + gap12)
    ).clamp(0.0, 0.30 * gap12)
    return tail2.float() + dx * slope12 + curvature * (slope12 - slope01)


class QualityConstrainedForecastPolicy:
    """Solve the fewest full DiT evaluations inside the accepted 12/8 error.

    One calibration request evaluates every solver point and retains only a
    small, evenly-spaced sample of the expensive tail residual.  The policy
    then simulates all feasible full-refresh schedules.  It selects the
    smallest schedule whose audio and video relative-L1/cosine envelope is no
    worse than the accepted depth-3 12/8 forecast.  This is an offline control
    problem inside one hot model session, not a user-facing step-count sweep.
    """

    accepted_steps = (0, 1, 2, 3, 4, 6, 8, 11, 14, 17, 18, 19)

    def __init__(self, *, step_count: int = 20, sentinel_rows: int = 512) -> None:
        if step_count != 20:
            raise ValueError("quality-constrained forecast is calibrated for 20 steps")
        if sentinel_rows < 128:
            raise ValueError("forecast calibration needs at least 128 sentinel rows")
        self.step_count = int(step_count)
        self.sentinel_rows = int(sentinel_rows)
        self.points: list[_ForecastCalibrationPoint] = []
        self.selected_steps: tuple[int, ...] | None = None
        self.selected_mode = "curvature"
        self.baseline_metrics: dict[str, float] = {}
        self.selected_metrics: dict[str, float] = {}

    @staticmethod
    def _row_indices(rows: int, count: int) -> torch.Tensor:
        chosen = min(rows, count)
        return torch.linspace(0, rows - 1, steps=chosen).round().long().unique()

    def observe(
        self,
        *,
        step_index: int,
        anchor_residual: torch.Tensor,
        tail_residual_host: torch.Tensor,
        layout: TargetLayout,
    ) -> None:
        if step_index != len(self.points):
            raise ValueError("forecast calibration observations must be contiguous")
        split = layout.audio_rows
        audio_indices = self._row_indices(split, min(512, self.sentinel_rows))
        video_indices = self._row_indices(layout.video_rows, self.sentinel_rows)
        channel_count = int(anchor_residual.shape[1])
        if channel_count <= 0:
            raise ValueError("forecast anchor sample has no channels")

        def take(value, start, indices):
            source_indices = indices.to(value.device) + start
            return value.index_select(0, source_indices).detach().float().cpu()

        self.points.append(
            _ForecastCalibrationPoint(
                step_index=step_index,
                audio_anchor=take(anchor_residual, 0, audio_indices),
                video_anchor=take(anchor_residual, split, video_indices),
                audio_tail=take(tail_residual_host[:, :channel_count], 0, audio_indices),
                video_tail=take(
                    tail_residual_host[:, :channel_count], split, video_indices
                ),
            )
        )

    @staticmethod
    def _secant_predict(
        current: torch.Tensor,
        older: _ForecastCalibrationPoint,
        newer: _ForecastCalibrationPoint,
        *,
        modality: str,
    ) -> torch.Tensor:
        anchor1 = getattr(older, f"{modality}_anchor")
        anchor2 = getattr(newer, f"{modality}_anchor")
        tail1 = getattr(older, f"{modality}_tail")
        tail2 = getattr(newer, f"{modality}_tail")
        gamma, _ = _directional_gamma(current, anchor1, anchor2)
        return tail1.float() * (1.0 - gamma.float()) + tail2.float() * gamma.float()

    @staticmethod
    def _error(predicted: torch.Tensor, actual: torch.Tensor) -> tuple[float, float]:
        left, right = predicted.flatten().float(), actual.flatten().float()
        relative_l1 = float(
            ((left - right).abs().sum() / right.abs().sum().clamp_min(1e-8)).item()
        )
        cosine_loss = float(
            (1.0 - F.cosine_similarity(left[None], right[None]).item())
        )
        return relative_l1, max(0.0, cosine_loss)

    def _simulate(
        self, schedule: tuple[int, ...], *, mode: str
    ) -> dict[str, float] | None:
        actual = frozenset(schedule)
        histories: list[_ForecastCalibrationPoint] = []
        errors = {"audio_l1": [], "video_l1": [], "audio_cos": [], "video_cos": []}
        for point in self.points:
            if point.step_index in actual:
                histories.append(point)
                histories = histories[-3:]
                continue
            required = 2 if mode == "secant" else 3
            if len(histories) < required:
                return None
            for modality in ("audio", "video"):
                current = getattr(point, f"{modality}_anchor")
                if mode == "secant":
                    predicted = self._secant_predict(
                        current, histories[-2], histories[-1], modality=modality
                    )
                else:
                    predicted = _curvature_predict(
                        current,
                        histories[-3],
                        histories[-2],
                        histories[-1],
                        modality=modality,
                        layout=None,
                    )
                l1, cosine_loss = self._error(
                    predicted, getattr(point, f"{modality}_tail")
                )
                errors[f"{modality}_l1"].append(l1)
                errors[f"{modality}_cos"].append(cosine_loss)
        result: dict[str, float] = {}
        for modality in ("audio", "video"):
            l1 = errors[f"{modality}_l1"]
            cosine = errors[f"{modality}_cos"]
            result[f"{modality}_l1_sum"] = sum(l1)
            result[f"{modality}_l1_max"] = max(l1, default=0.0)
            result[f"{modality}_cos_sum"] = sum(cosine)
            result[f"{modality}_cos_max"] = max(cosine, default=0.0)
        return result

    @staticmethod
    def _inside(candidate: dict[str, float], limit: dict[str, float]) -> bool:
        # A tiny numeric allowance is only for CPU reduction order.  It is not
        # a quality-relaxation knob.
        return all(candidate[key] <= value * 1.0005 + 1e-6 for key, value in limit.items())

    def finalize(self) -> tuple[int, ...]:
        if self.selected_steps is not None:
            return self.selected_steps
        if len(self.points) != self.step_count:
            raise RuntimeError("forecast calibration did not observe every solver step")
        baseline = self._simulate(self.accepted_steps, mode="secant")
        if baseline is None:
            raise RuntimeError("accepted forecast schedule could not be simulated")
        self.baseline_metrics = baseline

        mandatory = frozenset((0, 1, 2, self.step_count - 2, self.step_count - 1))
        # Backward elimination is deliberately bounded: calibration must be
        # negligible next to a real H3 request.  Starting from the exact
        # 20-step trajectory, remove the safest refresh at each round until no
        # further removal stays inside the accepted 12/8 local-error envelope.
        # This is an automatic schedule solver, not a manual cardinality sweep.
        schedule = tuple(range(self.step_count))
        selected_metrics = self._simulate(schedule, mode="curvature")
        while len(schedule) > len(mandatory):
            feasible: list[tuple[float, tuple[int, ...], dict[str, float]]] = []
            for removed in schedule:
                if removed in mandatory:
                    continue
                candidate_schedule = tuple(
                    index for index in schedule if index != removed
                )
                metrics = self._simulate(candidate_schedule, mode="curvature")
                if metrics is None or not self._inside(metrics, baseline):
                    continue
                risk = max(
                    metrics[key] / max(baseline[key], 1e-8) for key in baseline
                )
                feasible.append((risk, candidate_schedule, metrics))
            if not feasible:
                break
            _, schedule, selected_metrics = min(
                feasible, key=lambda item: (item[0], item[1])
            )
        if schedule == tuple(range(self.step_count)):
            self.selected_steps = self.accepted_steps
            self.selected_mode = "secant"
            self.selected_metrics = baseline
        else:
            self.selected_steps = schedule
            self.selected_metrics = selected_metrics or {}
        return self.selected_steps

    def export(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mode": "quality_constrained_curvature_forecast",
            "calibration_points": len(self.points),
            "accepted_steps": list(self.accepted_steps),
            "selected_steps": (
                None if self.selected_steps is None else list(self.selected_steps)
            ),
            "selected_mode": self.selected_mode,
            "baseline_metrics": dict(self.baseline_metrics),
            "selected_metrics": dict(self.selected_metrics),
        }


def target_layout(layout: PackedLayout) -> TargetLayout:
    audio = layout.segment("audio", last=True)
    video = layout.segment("video", last=True)
    if audio.stop != video.start:
        raise ValueError("forecast requires contiguous [audio | video] target rows")
    _, latent_t, latent_h, latent_w, _, _ = layout.signature
    result = TargetLayout(
        start=audio.start,
        stop=video.stop,
        audio_rows=audio.length,
        video_rows=video.length,
        latent_t=latent_t,
        grid_h=latent_h // 2,
        grid_w=latent_w // 2,
    )
    if result.video_rows != result.latent_t * result.grid_h * result.grid_w:
        raise ValueError("video target rows do not match the packed H3 grid")
    return result


def _directional_gamma(
    current: torch.Tensor,
    older: torch.Tensor,
    newer: torch.Tensor,
    *,
    layout: TargetLayout | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if current.shape != older.shape or current.shape != newer.shape or current.ndim != 2:
        raise ValueError("directional inputs must be matching [rows, channels]")
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
            raise ValueError("video rows do not match the packed H3 grid")
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

    values = torch.stack(
        (
            gamma.min(),
            gamma.mean(),
            gamma.max(),
            (confidence < 0.25).float().mean(),
        )
    ).detach().cpu().tolist()
    return gamma, {
        "gamma_min": float(values[0]),
        "gamma_mean": float(values[1]),
        "gamma_max": float(values[2]),
        "conservative_fraction": float(values[3]),
    }


class DirectionalForecastController:
    """Run full refreshes or the V7-D depth-3 local forecast on one stack."""

    def __init__(
        self,
        *,
        actual_steps: tuple[int, ...],
        sample_channels: int = 32,
        anchor_depth: int = 3,
        segment_cache=None,
    ) -> None:
        if not actual_steps:
            raise ValueError("forecast controller requires actual steps")
        if tuple(sorted(set(actual_steps))) != actual_steps:
            raise ValueError("actual steps must be sorted and unique")
        if anchor_depth <= 0:
            raise ValueError("anchor depth must be positive")
        self.actual_steps = frozenset(actual_steps)
        self.sample_channels = max(4, int(sample_channels))
        self.anchor_depth = int(anchor_depth)
        self.segment_cache = segment_cache
        self._indices: dict[tuple[int, str], torch.Tensor] = {}
        self.history: list[_TailHistory] = []
        self.records: list[dict[str, Any]] = []

    @property
    def history_limit(self) -> int:
        return 2

    def _on_actual_observation(
        self,
        *,
        step_index: int,
        anchor_residual: torch.Tensor,
        tail_residual_host: torch.Tensor,
        layout: TargetLayout,
    ) -> None:
        """Optional observation hook for request-to-request calibration."""

    def _sample(self, feature: torch.Tensor) -> torch.Tensor:
        count = min(self.sample_channels, int(feature.shape[1]))
        key = (int(feature.shape[1]), str(feature.device))
        indices = self._indices.get(key)
        if indices is None or indices.device != feature.device or indices.numel() != count:
            indices = torch.arange(count, device=feature.device, dtype=torch.long)
            self._indices[key] = indices
        return feature.detach().index_select(1, indices).clone()

    def should_forecast(self, step_index: int, requested_actual: bool) -> bool:
        return not requested_actual and len(self.history) >= 2

    def _predict(
        self,
        *,
        step_index: int,
        anchor_full: torch.Tensor,
        input_sample: torch.Tensor,
        anchor_sample: torch.Tensor,
        layout: TargetLayout,
    ) -> torch.Tensor:
        started = time.perf_counter()
        current = anchor_sample - input_sample
        older, newer = self.history[-2:]
        split = layout.audio_rows
        gamma_audio, audio_stats = _directional_gamma(
            current[:split],
            older.anchor_residual_sample[:split],
            newer.anchor_residual_sample[:split],
        )
        gamma_video, video_stats = _directional_gamma(
            current[split:],
            older.anchor_residual_sample[split:],
            newer.anchor_residual_sample[split:],
            layout=layout,
        )
        gamma_rows = torch.cat(
            (gamma_audio.expand(split, 1), gamma_video), dim=0
        ).to(dtype=anchor_full.dtype)

        predicted = anchor_full
        streamed = _apply_forecast_history_terms_(
            predicted,
            (older, newer),
            (1.0 - gamma_rows, gamma_rows),
        )
        del gamma_rows
        self.records.append(
            {
                "step_index": int(step_index),
                "mode": "forecast",
                "anchor_depth": self.anchor_depth,
                "audio": audio_stats,
                "video": video_stats,
                "history_transfer": (
                    "pageable_chunked" if streamed else "pinned_whole"
                ),
                "history_chunk_rows": (
                    LONG_FORECAST_HISTORY_CHUNK_ROWS if streamed else None
                ),
                "prediction_seconds": time.perf_counter() - started,
            }
        )
        return predicted

    def _observe_actual(
        self,
        *,
        step_index: int,
        input_sample: torch.Tensor,
        anchor_sample: torch.Tensor,
        anchor_host: torch.Tensor,
        final: torch.Tensor,
        layout: TargetLayout,
        streamed_history: bool,
    ) -> None:
        started = time.perf_counter()
        pin = final.device.type == "cuda" and not streamed_history
        tail_host = torch.empty(
            final.shape,
            dtype=final.dtype,
            device="cpu",
            pin_memory=pin,
        )
        for start in range(0, final.shape[0], LONG_FORECAST_HISTORY_CHUNK_ROWS):
            stop = min(start + LONG_FORECAST_HISTORY_CHUNK_ROWS, final.shape[0])
            anchor_chunk = anchor_host[start:stop].to(
                device=final.device,
                non_blocking=pin,
            )
            tail_chunk = final[start:stop] - anchor_chunk
            tail_host[start:stop].copy_(
                tail_chunk,
                non_blocking=pin,
            )
            del anchor_chunk, tail_chunk
        anchor_residual = anchor_sample - input_sample
        self._on_actual_observation(
            step_index=step_index,
            anchor_residual=anchor_residual,
            tail_residual_host=tail_host,
            layout=layout,
        )
        self.history.append(_TailHistory(
            step_index,
            anchor_residual,
            tail_host,
            streamed=streamed_history,
        ))
        self.history = self.history[-self.history_limit :]
        self.records.append(
            {
                "step_index": int(step_index),
                "mode": "actual",
                "anchor_depth": self.anchor_depth,
                "history_storage": (
                    "pageable_chunked" if streamed_history else "pinned_whole"
                ),
                "history_bytes": _tensor_storage_bytes(tail_host),
                "history_chunk_rows": (
                    LONG_FORECAST_HISTORY_CHUNK_ROWS
                    if streamed_history
                    else None
                ),
                "observation_seconds": time.perf_counter() - started,
            }
        )

    def run_block_stack(
        self,
        stack: Any,
        value: torch.Tensor,
        *,
        step_index: int,
        requested_actual: bool,
        layout: PackedLayout,
        unique_timesteps: torch.Tensor,
        modulation_segments: tuple[tuple[int, int, int], ...],
        frequencies: torch.Tensor,
        curve_rows: Any,
        mlp_chunk_tokens: int | None = None,
    ) -> torch.Tensor:
        if len(stack.blocks) < self.anchor_depth + 1:
            raise ValueError("forecast requires blocks beyond its anchor depth")
        info = target_layout(layout)
        target = slice(info.start, info.stop)
        input_sample = self._sample(value[target])
        block_kwargs = {
            "timestep_rows": curve_rows,
            "modulation_segments": modulation_segments,
            "frequencies": frequencies,
            "mlp_chunk_tokens": mlp_chunk_tokens,
        }

        if callable(getattr(stack, "run_range", None)):
            value = stack.run_range(
                value, start=0, stop=self.anchor_depth, **block_kwargs
            )
        else:
            for block in stack.blocks[: self.anchor_depth]:
                value = block(value, **block_kwargs)

        anchor = value[target]
        anchor_sample = self._sample(anchor)
        if self.should_forecast(step_index, requested_actual):
            value[target] = self._predict(
                step_index=step_index,
                anchor_full=anchor,
                input_sample=input_sample,
                anchor_sample=anchor_sample,
                layout=info,
            )
            return value
        streamed_history = _stream_forecast_history(anchor)
        anchor_host = _copy_forecast_history_to_host(
            anchor,
            streamed=streamed_history,
        )

        if self.segment_cache is not None:
            value = self.segment_cache.run_actual_tail(
                stack,
                value,
                step_index=step_index,
                prefix_stop=self.anchor_depth,
                protected_tokens=info.start + info.audio_rows,
                video_shape=(info.latent_t, info.grid_h, info.grid_w),
                block_kwargs=block_kwargs,
            )
        elif callable(getattr(stack, "run_range", None)):
            value = stack.run_range(
                value,
                start=self.anchor_depth,
                stop=len(stack.blocks),
                **block_kwargs,
            )
        else:
            for block in stack.blocks[self.anchor_depth :]:
                value = block(value, **block_kwargs)

        self._observe_actual(
            step_index=step_index,
            input_sample=input_sample,
            anchor_sample=anchor_sample,
            anchor_host=anchor_host,
            final=value[target],
            layout=info,
            streamed_history=streamed_history,
        )
        return value

    def export(self) -> dict[str, Any]:
        report = {
            "schema_version": 1,
            "mode": "native_depth3_local_directional",
            "planned_actual_steps": sorted(self.actual_steps),
            "actual_steps": sum(record["mode"] == "actual" for record in self.records),
            "forecast_steps": sum(record["mode"] == "forecast" for record in self.records),
            "records": list(self.records),
        }
        if self.segment_cache is not None:
            report["segment_cache"] = self.segment_cache.export()
        return report

    def checkpoint_state(self) -> dict[str, Any]:
        """Serialize only the state required for an exact later forecast."""

        if self.segment_cache is not None:
            raise RuntimeError(
                "checkpoint/resume is not yet compatible with segment-cache state"
            )
        return {
            "schema_version": 1,
            "history": [
                {
                    "step_index": int(item.step_index),
                    "anchor_residual_sample": item.anchor_residual_sample.detach().cpu(),
                    "tail_residual_host": item.tail_residual_host.detach().cpu(),
                    "streamed": bool(item.streamed),
                    "chunk_rows": int(item.chunk_rows),
                }
                for item in self.history
            ],
            "records": list(self.records),
        }

    def restore_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Restore a state emitted by :meth:`checkpoint_state`."""

        if state.get("schema_version") != 1:
            raise ValueError("unsupported forecast checkpoint schema")
        history = state.get("history")
        records = state.get("records")
        if not isinstance(history, list) or not isinstance(records, list):
            raise ValueError("forecast checkpoint is malformed")
        restored: list[_TailHistory] = []
        for item in history:
            if not isinstance(item, dict):
                raise ValueError("forecast checkpoint history is malformed")
            anchor = item.get("anchor_residual_sample")
            tail = item.get("tail_residual_host")
            if not isinstance(anchor, torch.Tensor) or not isinstance(tail, torch.Tensor):
                raise ValueError("forecast checkpoint history is missing tensors")
            restored.append(
                _TailHistory(
                    int(item["step_index"]),
                    anchor.to(device="cuda", non_blocking=False),
                    tail.contiguous(),
                    streamed=bool(item.get(
                        "streamed",
                        _tensor_storage_bytes(tail)
                        >= LONG_FORECAST_HISTORY_PAGEABLE_BYTES,
                    )),
                    chunk_rows=int(item.get(
                        "chunk_rows",
                        LONG_FORECAST_HISTORY_CHUNK_ROWS,
                    )),
                )
            )
        self.history = restored[-self.history_limit :]
        self.records = list(records)


class ForecastErrorDebtController(DirectionalForecastController):
    """Measure request-local Forecast debt without running a Dense teacher.

    Every planned Actual correction already produces the exact expensive-tail
    residual.  This controller compares that residual with the secant tail the
    normal Forecast path *would* have used at the same point.  Only uniformly
    sampled rows and the existing shallow-anchor channels are inspected, so
    observation adds no DiT block or Attention evaluation.

    The release starts in ``observe_only`` mode.  Once a workload-independent
    null envelope has been calibrated, the same controller can spend a bounded
    token bucket by promoting a future requested Forecast to a Dense Actual.
    That discrete execution decision is driven by accumulated continuous error
    debt, not by prompt semantics, scene labels or named candidate versions.
    """

    def __init__(
        self,
        *,
        actual_steps: tuple[int, ...],
        segment_cache=None,
        sentinel_audio_rows: int = 128,
        sentinel_video_rows: int = 512,
        recovery_enabled: bool = False,
        max_runtime_promotions: int = 0,
        risk_reserve_controller: Any | None = None,
    ) -> None:
        super().__init__(
            actual_steps=actual_steps,
            segment_cache=segment_cache,
        )
        if sentinel_audio_rows <= 0 or sentinel_video_rows <= 0:
            raise ValueError("forecast feedback sentinel counts must be positive")
        if max_runtime_promotions < 0:
            raise ValueError("forecast feedback promotion limit cannot be negative")
        if recovery_enabled and max_runtime_promotions == 0:
            raise ValueError("enabled forecast recovery requires a promotion budget")
        if risk_reserve_controller is not None and recovery_enabled:
            raise ValueError(
                "use either legacy debt recovery or mechanistic risk-reserve control"
            )
        if risk_reserve_controller is not None:
            controller_limit = int(risk_reserve_controller.maximum_promotions)
            if max_runtime_promotions not in (0, controller_limit):
                raise ValueError(
                    "forecast promotion limit disagrees with risk-reserve controller"
                )
            max_runtime_promotions = controller_limit
        self.sentinel_audio_rows = int(sentinel_audio_rows)
        self.sentinel_video_rows = int(sentinel_video_rows)
        self.recovery_enabled = bool(recovery_enabled)
        self.max_runtime_promotions = int(max_runtime_promotions)
        self.risk_reserve_controller = risk_reserve_controller
        self.feedback_records: list[dict[str, Any]] = []
        self._baseline_max: dict[str, float] = {}
        self._baseline_locked = False
        self._forecast_debt = 0.0
        self._runtime_promotions: list[int] = []

    @staticmethod
    def _row_indices(rows: int, count: int) -> torch.Tensor:
        chosen = min(int(rows), int(count))
        return torch.linspace(0, rows - 1, steps=chosen).round().long().unique()

    @staticmethod
    def _take_rows(
        value: torch.Tensor,
        *,
        start: int,
        indices: torch.Tensor,
        channels: int,
    ) -> torch.Tensor:
        source = indices.to(device=value.device) + int(start)
        return (
            value[:, :channels]
            .index_select(0, source)
            .detach()
            .float()
            .cpu()
        )

    @staticmethod
    def _relative_errors(
        predicted: torch.Tensor,
        actual: torch.Tensor,
    ) -> tuple[float, float]:
        left = predicted.flatten().float()
        right = actual.flatten().float()
        relative_l1 = float(
            (
                (left - right).abs().sum()
                / right.abs().sum().clamp_min(1.0e-8)
            ).item()
        )
        cosine_loss = float(
            max(
                0.0,
                1.0
                - F.cosine_similarity(left[None], right[None]).item(),
            )
        )
        return relative_l1, cosine_loss

    def should_forecast(self, step_index: int, requested_actual: bool) -> bool:
        eligible = super().should_forecast(step_index, requested_actual)
        if not eligible:
            return False
        if (
            self.risk_reserve_controller is not None
            and self.risk_reserve_controller.should_promote(step_index)
        ):
            if int(step_index) not in self._runtime_promotions:
                self._runtime_promotions.append(int(step_index))
            return False
        may_recover = (
            self.recovery_enabled
            and len(self._runtime_promotions) < self.max_runtime_promotions
            and self._forecast_debt >= 1.0
        )
        if not may_recover:
            return True
        self._forecast_debt -= 1.0
        self._runtime_promotions.append(int(step_index))
        return False

    def _on_actual_observation(
        self,
        *,
        step_index: int,
        anchor_residual: torch.Tensor,
        tail_residual_host: torch.Tensor,
        layout: TargetLayout,
    ) -> None:
        if len(self.history) < 2:
            self.feedback_records.append({
                "step_index": int(step_index),
                "status": "warming_history",
            })
            return
        older, newer = self.history[-2:]
        horizon = max(1, int(step_index) - int(newer.step_index))
        channels = min(
            int(anchor_residual.shape[1]),
            int(tail_residual_host.shape[1]),
        )
        if channels <= 0:
            raise ValueError("forecast feedback requires sampled channels")
        audio_indices = self._row_indices(
            layout.audio_rows, self.sentinel_audio_rows
        )
        video_indices = self._row_indices(
            layout.video_rows, self.sentinel_video_rows
        )
        metrics: dict[str, float] = {}
        for modality, start, indices in (
            ("audio", 0, audio_indices),
            ("video", layout.audio_rows, video_indices),
        ):
            current_anchor = self._take_rows(
                anchor_residual,
                start=start,
                indices=indices,
                channels=channels,
            )
            older_anchor = self._take_rows(
                older.anchor_residual_sample,
                start=start,
                indices=indices,
                channels=channels,
            )
            newer_anchor = self._take_rows(
                newer.anchor_residual_sample,
                start=start,
                indices=indices,
                channels=channels,
            )
            gamma, _stats = _directional_gamma(
                current_anchor,
                older_anchor,
                newer_anchor,
            )
            older_tail = self._take_rows(
                older.tail_residual_host,
                start=start,
                indices=indices,
                channels=channels,
            )
            newer_tail = self._take_rows(
                newer.tail_residual_host,
                start=start,
                indices=indices,
                channels=channels,
            )
            actual_tail = self._take_rows(
                tail_residual_host,
                start=start,
                indices=indices,
                channels=channels,
            )
            predicted_tail = older_tail.mul(1.0 - gamma).add_(
                newer_tail.mul(gamma)
            )
            relative_l1, cosine_loss = self._relative_errors(
                predicted_tail, actual_tail
            )
            metrics[f"{modality}_relative_l1"] = relative_l1
            metrics[f"{modality}_cosine_loss"] = cosine_loss

        control_l1 = {
            # The observed quantity is already the complete tail error at the
            # current sigma point.  The first GPU calibration showed that an
            # additional horizon-squared divisor suppressed every legitimate
            # correction signal by 7--16x, so horizon remains telemetry rather
            # than an assumed error law.
            key: value
            for key, value in metrics.items()
            if key.endswith("relative_l1")
        }
        is_opening_baseline = horizon == 1 and not self._baseline_locked
        if is_opening_baseline:
            for key, value in control_l1.items():
                self._baseline_max[key] = max(
                    value, self._baseline_max.get(key, 0.0)
                )
        elif horizon > 1:
            self._baseline_locked = True

        risk_ratio = None
        debt_increment = 0.0
        if self._baseline_max and horizon > 1:
            risk_ratio = max(
                control_l1[key] / max(self._baseline_max[key], 1.0e-8)
                for key in control_l1
            )
            debt_increment = max(0.0, float(risk_ratio) - 1.0)
            # Stable corrections discharge stale evidence; anomalous growth
            # accumulates continuously until it can buy one bounded recovery.
            self._forecast_debt = max(
                0.0,
                self._forecast_debt * 0.5 + debt_increment,
            )

        mechanistic_decision = None
        if risk_ratio is not None and self.risk_reserve_controller is not None:
            modality_ratios = {
                modality: (
                    control_l1[f"{modality}_relative_l1"]
                    / max(
                        self._baseline_max[f"{modality}_relative_l1"],
                        1.0e-8,
                    )
                )
                for modality in ("audio", "video")
            }
            decision = self.risk_reserve_controller.observe_actual(
                step_index=step_index,
                audio_risk_ratio=modality_ratios["audio"],
                video_risk_ratio=modality_ratios["video"],
            )
            mechanistic_decision = decision.to_dict()

        self.feedback_records.append({
            "step_index": int(step_index),
            "status": "observed",
            "history_steps": [int(older.step_index), int(newer.step_index)],
            "horizon": horizon,
            "sentinel_audio_rows": int(audio_indices.numel()),
            "sentinel_video_rows": int(video_indices.numel()),
            "sample_channels": channels,
            "metrics": metrics,
            "control_relative_l1": control_l1,
            "horizon_normalization": "none_gpu_calibrated_v1",
            "opening_baseline": is_opening_baseline,
            "baseline_max": dict(sorted(self._baseline_max.items())),
            "risk_ratio": risk_ratio,
            "debt_increment": debt_increment,
            "forecast_debt": self._forecast_debt,
            "mechanistic_runtime_decision": mechanistic_decision,
        })

    def export(self) -> dict[str, Any]:
        report = super().export()
        report.update({
            "mode": "request_local_forecast_error_debt",
            "feedback_policy_id": V24_FORECAST_FEEDBACK_POLICY_ID,
            "feedback_mode": (
                "mechanistic_risk_reserve"
                if self.risk_reserve_controller is not None
                else "bounded_recovery"
                if self.recovery_enabled
                else "observe_only"
            ),
            "adds_teacher_evaluations": False,
            "sentinel_audio_rows": self.sentinel_audio_rows,
            "sentinel_video_rows": self.sentinel_video_rows,
            "max_runtime_promotions": self.max_runtime_promotions,
            "runtime_promotions": list(self._runtime_promotions),
            "forecast_debt": self._forecast_debt,
            "baseline_max": dict(sorted(self._baseline_max.items())),
            "feedback_records": list(self.feedback_records),
            "mechanistic_runtime": (
                None
                if self.risk_reserve_controller is None
                else self.risk_reserve_controller.export()
            ),
        })
        return report

    def checkpoint_state(self) -> dict[str, Any]:
        state = super().checkpoint_state()
        state["schema_version"] = (
            3 if self.risk_reserve_controller is not None else 2
        )
        state["forecast_feedback"] = {
            "policy_id": V24_FORECAST_FEEDBACK_POLICY_ID,
            "recovery_enabled": self.recovery_enabled,
            "max_runtime_promotions": self.max_runtime_promotions,
            "baseline_max": dict(self._baseline_max),
            "baseline_locked": self._baseline_locked,
            "forecast_debt": self._forecast_debt,
            "runtime_promotions": list(self._runtime_promotions),
            "feedback_records": list(self.feedback_records),
            "mechanistic_runtime": (
                None
                if self.risk_reserve_controller is None
                else self.risk_reserve_controller.checkpoint_state()
            ),
        }
        return state

    def restore_checkpoint_state(self, state: dict[str, Any]) -> None:
        schema_version = state.get("schema_version")
        if schema_version not in (2, 3):
            raise ValueError("unsupported forecast feedback checkpoint schema")
        feedback = state.get("forecast_feedback")
        if not isinstance(feedback, dict):
            raise ValueError("forecast feedback checkpoint is missing state")
        if feedback.get("policy_id") != V24_FORECAST_FEEDBACK_POLICY_ID:
            raise ValueError("forecast feedback checkpoint policy mismatch")
        if bool(feedback.get("recovery_enabled")) != self.recovery_enabled:
            raise ValueError("forecast feedback recovery mode mismatch")
        if int(feedback.get("max_runtime_promotions", -1)) != self.max_runtime_promotions:
            raise ValueError("forecast feedback promotion budget mismatch")
        base_state = dict(state)
        base_state["schema_version"] = 1
        base_state.pop("forecast_feedback", None)
        super().restore_checkpoint_state(base_state)
        baseline = feedback.get("baseline_max")
        records = feedback.get("feedback_records")
        promotions = feedback.get("runtime_promotions")
        if (
            not isinstance(baseline, dict)
            or not isinstance(records, list)
            or not isinstance(promotions, list)
        ):
            raise ValueError("forecast feedback checkpoint is malformed")
        restored_baseline = {
            str(key): float(value) for key, value in baseline.items()
        }
        debt = float(feedback.get("forecast_debt", float("nan")))
        restored_promotions = [int(step) for step in promotions]
        if (
            any(
                not math.isfinite(value) or value < 0.0
                for value in restored_baseline.values()
            )
            or not math.isfinite(debt)
            or debt < 0.0
            or len(restored_promotions) > self.max_runtime_promotions
        ):
            raise ValueError("forecast feedback checkpoint contains invalid values")
        self._baseline_max = restored_baseline
        self._baseline_locked = bool(feedback.get("baseline_locked"))
        self._forecast_debt = debt
        self._runtime_promotions = restored_promotions
        self.feedback_records = list(records)
        runtime_state = feedback.get("mechanistic_runtime")
        if schema_version == 3:
            if self.risk_reserve_controller is None or not isinstance(
                runtime_state, dict
            ):
                raise ValueError(
                    "forecast feedback checkpoint requires mechanistic runtime"
                )
            self.risk_reserve_controller.restore_checkpoint_state(runtime_state)
        elif self.risk_reserve_controller is not None:
            raise ValueError(
                "forecast feedback checkpoint omits mechanistic runtime state"
            )


class CalibrationForecastController(DirectionalForecastController):
    """Collect one exact solver trajectory for an adaptive hot-session policy."""

    def __init__(self, *, policy: QualityConstrainedForecastPolicy) -> None:
        super().__init__(actual_steps=tuple(range(policy.step_count)))
        self.policy = policy

    def should_forecast(self, step_index: int, requested_actual: bool) -> bool:
        return False

    def _on_actual_observation(
        self,
        *,
        step_index: int,
        anchor_residual: torch.Tensor,
        tail_residual_host: torch.Tensor,
        layout: TargetLayout,
    ) -> None:
        self.policy.observe(
            step_index=step_index,
            anchor_residual=anchor_residual,
            tail_residual_host=tail_residual_host,
            layout=layout,
        )

    def export(self) -> dict[str, Any]:
        selected = self.policy.finalize()
        report = super().export()
        report.update(
            {
                "mode": "quality_constrained_forecast_calibration",
                "selected_steps": list(selected),
                "adaptive_policy": self.policy.export(),
            }
        )
        return report


class CurvatureForecastController(DirectionalForecastController):
    """Use a bounded second-order tail model selected by calibration."""

    def __init__(
        self,
        *,
        policy: QualityConstrainedForecastPolicy,
        segment_cache=None,
    ) -> None:
        selected = policy.finalize()
        super().__init__(actual_steps=selected, segment_cache=segment_cache)
        self.policy = policy

    @property
    def history_limit(self) -> int:
        return 3

    def should_forecast(self, step_index: int, requested_actual: bool) -> bool:
        required = 2 if self.policy.selected_mode == "secant" else 3
        return step_index not in self.actual_steps and len(self.history) >= required

    def _predict(
        self,
        *,
        step_index: int,
        anchor_full: torch.Tensor,
        input_sample: torch.Tensor,
        anchor_sample: torch.Tensor,
        layout: TargetLayout,
    ) -> torch.Tensor:
        if self.policy.selected_mode == "secant":
            return super()._predict(
                step_index=step_index,
                anchor_full=anchor_full,
                input_sample=input_sample,
                anchor_sample=anchor_sample,
                layout=layout,
            )
        started = time.perf_counter()
        current = anchor_sample - input_sample
        oldest, older, newer = self.history[-3:]
        split = layout.audio_rows
        gamma_audio, audio_stats = _directional_gamma(
            current[:split],
            older.anchor_residual_sample[:split],
            newer.anchor_residual_sample[:split],
        )
        gamma_video, video_stats = _directional_gamma(
            current[split:],
            older.anchor_residual_sample[split:],
            newer.anchor_residual_sample[split:],
            layout=layout,
        )
        distance = torch.cat(
            (gamma_audio.expand(split, 1), gamma_video), dim=0
        ).float() - 1.0
        gap01 = float(max(1, older.step_index - oldest.step_index))
        gap12 = float(max(1, newer.step_index - older.step_index))
        dx = distance * gap12
        curvature = (
            dx * (dx + gap12) / (gap01 + gap12)
        ).clamp(0.0, 0.30 * gap12)
        coefficients = (
            curvature / gap01,
            -distance - curvature * (1.0 / gap12 + 1.0 / gap01),
            1.0 + distance + curvature / gap12,
        )
        predicted = anchor_full
        streamed = _apply_forecast_history_terms_(
            predicted,
            (oldest, older, newer),
            coefficients,
        )
        curvature_values = curvature.detach().float()
        self.records.append(
            {
                "step_index": int(step_index),
                "mode": "forecast",
                "forecast_order": 2,
                "anchor_depth": self.anchor_depth,
                "audio": audio_stats,
                "video": video_stats,
                "curvature_mean": float(curvature_values.mean().cpu()),
                "curvature_max": float(curvature_values.max().cpu()),
                "history_transfer": (
                    "pageable_chunked" if streamed else "pinned_whole"
                ),
                "history_chunk_rows": (
                    LONG_FORECAST_HISTORY_CHUNK_ROWS if streamed else None
                ),
                "prediction_seconds": time.perf_counter() - started,
            }
        )
        return predicted

    def export(self) -> dict[str, Any]:
        report = super().export()
        report.update(
            {
                "mode": "quality_constrained_curvature_forecast",
                "adaptive_policy": self.policy.export(),
            }
        )
        return report


class QualityConstrainedForecastFactory:
    """Share calibration evidence across requests in one immutable hot session."""

    def __init__(self, *, step_count: int = 20, sentinel_rows: int = 512) -> None:
        self.policy = QualityConstrainedForecastPolicy(
            step_count=step_count, sentinel_rows=sentinel_rows
        )
        self.controllers: list[DirectionalForecastController] = []

    def __call__(self, *, segment_cache=None) -> DirectionalForecastController:
        if self.policy.selected_steps is None:
            if self.controllers:
                raise RuntimeError(
                    "calibration request must finish and export before the next request"
                )
            controller: DirectionalForecastController = CalibrationForecastController(
                policy=self.policy
            )
        else:
            controller = CurvatureForecastController(
                policy=self.policy, segment_cache=segment_cache
            )
        self.controllers.append(controller)
        return controller

    def export(self) -> dict[str, Any]:
        return {
            "policy": self.policy.export(),
            "requests": [controller.export() for controller in self.controllers],
        }


__all__ = [
    "CalibrationForecastController",
    "CurvatureForecastController",
    "DirectionalForecastController",
    "ForecastErrorDebtController",
    "QualityConstrainedForecastFactory",
    "QualityConstrainedForecastPolicy",
    "TargetLayout",
    "V24_FORECAST_FEEDBACK_POLICY_ID",
    "target_layout",
]
