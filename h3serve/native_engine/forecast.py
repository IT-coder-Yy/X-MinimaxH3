"""Native depth-3 directional forecast for the pruned H3 block stack.

The controller mirrors the accepted V7-D mechanism without importing
ComfyUI or Spectrum.  Requested forecast steps execute blocks 0..2, then use
the two latest full-refresh tails to estimate the skipped blocks 3..49.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .model.packed import PackedLayout


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


@dataclass(frozen=True, slots=True)
class _ForecastCalibrationPoint:
    step_index: int
    audio_anchor: torch.Tensor
    video_anchor: torch.Tensor
    audio_tail: torch.Tensor
    video_tail: torch.Tensor


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
        device = predicted.device
        old = older.tail_residual_host.to(device=device, non_blocking=device.type == "cuda")
        predicted.addcmul_(old, 1.0 - gamma_rows)
        del old
        new = newer.tail_residual_host.to(device=device, non_blocking=device.type == "cuda")
        predicted.addcmul_(new, gamma_rows)
        del new, gamma_rows
        self.records.append(
            {
                "step_index": int(step_index),
                "mode": "forecast",
                "anchor_depth": self.anchor_depth,
                "audio": audio_stats,
                "video": video_stats,
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
    ) -> None:
        pin = final.device.type == "cuda"
        tail_host = torch.empty(
            final.shape,
            dtype=final.dtype,
            device="cpu",
            pin_memory=pin,
        )
        for start in range(0, final.shape[0], 4096):
            stop = min(start + 4096, final.shape[0])
            anchor_chunk = anchor_host[start:stop].to(
                device=final.device, non_blocking=pin
            )
            tail_chunk = final[start:stop] - anchor_chunk
            tail_host[start:stop].copy_(tail_chunk, non_blocking=pin)
            del anchor_chunk, tail_chunk
        anchor_residual = anchor_sample - input_sample
        self._on_actual_observation(
            step_index=step_index,
            anchor_residual=anchor_residual,
            tail_residual_host=tail_host,
            layout=layout,
        )
        self.history.append(_TailHistory(step_index, anchor_residual, tail_host))
        self.history = self.history[-self.history_limit :]
        self.records.append(
            {
                "step_index": int(step_index),
                "mode": "actual",
                "anchor_depth": self.anchor_depth,
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
        pin = anchor.device.type == "cuda"
        anchor_host = torch.empty(
            anchor.shape,
            dtype=anchor.dtype,
            device="cpu",
            pin_memory=pin,
        )
        anchor_host.copy_(anchor.detach(), non_blocking=pin)

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
                )
            )
        self.history = restored[-self.history_limit :]
        self.records = list(records)


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
        device = predicted.device
        for history, coefficient in zip(
            (oldest, older, newer), coefficients, strict=True
        ):
            tail = history.tail_residual_host.to(
                device=device, non_blocking=device.type == "cuda"
            )
            predicted.addcmul_(tail, coefficient.to(dtype=predicted.dtype))
            del tail
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
    "QualityConstrainedForecastFactory",
    "QualityConstrainedForecastPolicy",
    "TargetLayout",
    "target_layout",
]
