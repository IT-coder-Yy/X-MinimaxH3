"""ComfyUI-independent H3 audio-video sampler interfaces."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol

from .scheduler import SamplingPlan, StepClock


@dataclass(frozen=True, slots=True)
class AVPrediction:
    """Denoised x0 predictions returned by the model/forecast controller."""

    video_denoised: Any
    audio_denoised: Any


class AVPredictor(Protocol):
    def __call__(
        self,
        video: Any,
        audio: Any,
        clock: StepClock,
        *,
        step_index: int,
        is_actual_step: bool,
    ) -> AVPrediction: ...


CancelCheck = Callable[[], None]
StepCallback = Callable[[int, StepClock, Any, Any], None]
StepTransition = Callable[
    [int, StepClock, Any, Any, Any | None, Any | None],
    tuple[Any, Any, Any | None, Any | None],
]


def _euler(sample: Any, denoised: Any, sigma: float, sigma_next: float) -> Any:
    if sigma <= 0.0:
        raise ValueError("Euler source sigma must be positive")
    derivative = (sample - denoised) / sigma
    return sample + derivative * (sigma_next - sigma)


def _phi_values(value: float) -> tuple[float, float]:
    if abs(value) < 1e-7:
        return 1.0, 0.5
    phi1 = math.expm1(value) / value
    return phi1, (phi1 - 1.0) / value


def _res_step(
    sample: Any,
    denoised: Any,
    *,
    sigma: float,
    sigma_next: float,
    previous_sigma: float | None,
    previous_denoised: Any | None,
) -> Any:
    if sigma_next == 0.0 or previous_denoised is None or previous_sigma is None:
        return _euler(sample, denoised, sigma, sigma_next)

    t = -math.log(sigma)
    t_previous = -math.log(previous_sigma)
    t_next = -math.log(sigma_next)
    # eta=0 means the prior step's sigma_down is today's source sigma.
    t_old_down = t
    h = t_next - t
    if h == 0.0:
        raise ValueError("duplicate sigma endpoints are not supported")
    c2 = (t_previous - t_old_down) / h
    phi1, phi2 = _phi_values(-h)
    if c2 == 0.0 or not math.isfinite(c2):
        b1, b2 = phi1, 0.0
    else:
        b2 = phi2 / c2
        b1 = phi1 - b2
    return math.exp(-h) * sample + h * (b1 * denoised + b2 * previous_denoised)


class ResMultistepAVSampler:
    """Second-order RES multistep sampler over synchronized H3 clocks.

    The predictor is responsible for executing either a real DiT step or a
    forecast step according to ``is_actual_step``. Both paths return the same
    x0 contract, so sampler mathematics stays independent from acceleration.
    """

    name = "res_multistep"

    def sample(
        self,
        video: Any,
        audio: Any,
        plan: SamplingPlan,
        predict: AVPredictor,
        *,
        cancel_check: CancelCheck = lambda: None,
        callback: StepCallback | None = None,
        transition: StepTransition | None = None,
        initial_previous_video: Any | None = None,
        initial_previous_audio: Any | None = None,
        initial_previous_video_sigma: float | None = None,
        initial_previous_audio_sigma: float | None = None,
    ) -> tuple[Any, Any]:
        history_values = (
            initial_previous_video,
            initial_previous_audio,
            initial_previous_video_sigma,
            initial_previous_audio_sigma,
        )
        if any(value is not None for value in history_values) and not all(
            value is not None for value in history_values
        ):
            raise ValueError("initial RES history must provide both modalities and sigmas")
        previous_video = initial_previous_video
        previous_audio = initial_previous_audio
        previous_video_sigma = initial_previous_video_sigma
        previous_audio_sigma = initial_previous_audio_sigma
        actual = frozenset(plan.actual_step_indices)

        for index in range(plan.step_count):
            cancel_check()
            clock = plan.clock(index)
            global_index = clock.index
            prediction = predict(
                video,
                audio,
                clock,
                step_index=global_index,
                is_actual_step=global_index in actual,
            )
            next_video = _res_step(
                video,
                prediction.video_denoised,
                sigma=clock.video_sigma,
                sigma_next=clock.video_sigma_next,
                previous_sigma=previous_video_sigma,
                previous_denoised=previous_video,
            )
            next_audio = _res_step(
                audio,
                prediction.audio_denoised,
                sigma=clock.audio_sigma,
                sigma_next=clock.audio_sigma_next,
                previous_sigma=previous_audio_sigma,
                previous_denoised=previous_audio,
            )
            previous_video, previous_audio = (
                prediction.video_denoised,
                prediction.audio_denoised,
            )
            previous_video_sigma = clock.video_sigma
            previous_audio_sigma = clock.audio_sigma
            video, audio = next_video, next_audio
            if transition is not None:
                video, audio, previous_video, previous_audio = transition(
                    global_index,
                    clock,
                    video,
                    audio,
                    previous_video,
                    previous_audio,
                )
            if callback is not None:
                callback(global_index, clock, video, audio)
        return video, audio


class TurboClockMode(str, Enum):
    """How the Turbo predictor represents its audio denoised estimate."""

    DUAL_SHIFT = "dual_shift"
    SHARED_VIDEO = "shared_video"


def _time_shift_sigma(sigma: float, source_shift: float, target_shift: float) -> float:
    base = sigma / (source_shift + sigma * (1.0 - source_shift))
    return target_shift * base / (1.0 + (target_shift - 1.0) * base)


def _time_shift_slope(sigma: float, source_shift: float, target_shift: float) -> float:
    base = sigma / (source_shift + sigma * (1.0 - source_shift))
    numerator = target_shift * (1.0 + (source_shift - 1.0) * base) ** 2
    denominator = source_shift * (1.0 + (target_shift - 1.0) * base) ** 2
    return numerator / denominator


class TurboAVSampler:
    """First-order distilled sampler with explicit audio clock ownership."""

    name = "turbo"

    def __init__(self, clock_mode: TurboClockMode = TurboClockMode.DUAL_SHIFT) -> None:
        self.clock_mode = clock_mode

    def sample(
        self,
        video: Any,
        audio: Any,
        plan: SamplingPlan,
        predict: AVPredictor,
        *,
        cancel_check: CancelCheck = lambda: None,
        callback: StepCallback | None = None,
        transition: StepTransition | None = None,
        initial_previous_video: Any | None = None,
        initial_previous_audio: Any | None = None,
        initial_previous_video_sigma: float | None = None,
        initial_previous_audio_sigma: float | None = None,
    ) -> tuple[Any, Any]:
        if transition is not None or any(
            value is not None
            for value in (
                initial_previous_video,
                initial_previous_audio,
                initial_previous_video_sigma,
                initial_previous_audio_sigma,
            )
        ):
            raise ValueError(
                "Turbo sampler does not support RES history or state transitions"
            )
        for index in range(plan.step_count):
            cancel_check()
            clock = plan.clock(index)
            global_index = clock.index
            prediction = predict(
                video,
                audio,
                clock,
                step_index=global_index,
                is_actual_step=True,
            )
            video_derivative = (video - prediction.video_denoised) / clock.video_sigma
            video = video + (
                clock.video_sigma_next - clock.video_sigma
            ) * video_derivative

            if self.clock_mode is TurboClockMode.SHARED_VIDEO:
                audio_derivative = (audio - prediction.audio_denoised) / clock.video_sigma
                audio = audio + (
                    clock.video_sigma_next - clock.video_sigma
                ) * audio_derivative
            else:
                # Matches the released Turbo adapter's legacy dual-clock
                # contract: model output is differentiated on the video clock,
                # then converted to the audio clock through d sigma_a/d sigma_v.
                slope = _time_shift_slope(
                    max(clock.video_sigma, 1e-6),
                    plan.video_shift,
                    plan.audio_shift,
                )
                audio_derivative = (
                    (audio - prediction.audio_denoised) / clock.video_sigma / slope
                )
                audio_sigma = _time_shift_sigma(
                    clock.video_sigma, plan.video_shift, plan.audio_shift
                )
                audio_sigma_next = _time_shift_sigma(
                    clock.video_sigma_next, plan.video_shift, plan.audio_shift
                )
                audio = audio + (audio_sigma_next - audio_sigma) * audio_derivative
            if callback is not None:
                callback(global_index, clock, video, audio)
        return video, audio


def create_sampler(
    name: str,
    *,
    turbo_clock_mode: TurboClockMode = TurboClockMode.DUAL_SHIFT,
) -> ResMultistepAVSampler | TurboAVSampler:
    if name == ResMultistepAVSampler.name:
        return ResMultistepAVSampler()
    if name == TurboAVSampler.name:
        return TurboAVSampler(turbo_clock_mode)
    raise ValueError(f"unsupported native H3 sampler: {name!r}")
