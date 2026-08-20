"""H3-specific lifecycle stages with narrow model-adapter contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import PipelineState
from .executor import PipelineContext, PipelineStage


def _method(component: Any, name: str):
    method = getattr(component, name, None)
    if not callable(method):
        raise TypeError(f"component {component!r} does not implement {name}()")
    return method


def _move_result_to_host(value: Any) -> Any:
    """Move a tensor-like decoded result off GPU without importing PyTorch."""

    to_method = getattr(value, "to", None)
    if callable(to_method):
        return to_method("cpu", non_blocking=False)
    return value


class ValidateRequestStage(PipelineStage):
    name = "validate"

    def run(self, state: PipelineState, context: PipelineContext) -> None:
        state.request.validate()


class TextConditioningStage(PipelineStage):
    name = "text_conditioning"

    def run(self, state: PipelineState, context: PipelineContext) -> None:
        context.residency.transition(("text_encoder",))
        encoder = context.component("text_encoder")
        state.text_conditioning = _method(encoder, "encode")(state.request)

    def verify_output(self, state: PipelineState) -> None:
        if state.text_conditioning is None:
            raise ValueError("text encoder returned no conditioning")


class FrameConditioningStage(PipelineStage):
    name = "frame_conditioning"

    def run(self, state: PipelineState, context: PipelineContext) -> None:
        if state.request.first_frame is None and state.request.last_frame is None:
            return
        context.residency.transition(("video_vae",))
        video_vae = context.component("video_vae")
        state.frame_conditioning = _method(video_vae, "encode_conditioning")(
            state.request
        )


class PrepareLatentsStage(PipelineStage):
    name = "prepare_latents"

    def run(self, state: PipelineState, context: PipelineContext) -> None:
        # Scheduler storage is small and host-resident; registering it with the
        # residency manager keeps one component lookup path without consuming
        # the GPU budget.
        scheduler = context.component("scheduler")
        prepared = _method(scheduler, "prepare")(state)
        if not isinstance(prepared, dict):
            raise TypeError("scheduler.prepare() must return a dict")
        state.video_latents = prepared.get("video_latents")
        state.audio_latents = prepared.get("audio_latents")
        state.packed_layout = prepared.get("packed_layout")

    def verify_output(self, state: PipelineState) -> None:
        if state.video_latents is None or state.audio_latents is None:
            raise ValueError("scheduler did not prepare both video and audio latents")


class DenoiseStage(PipelineStage):
    name = "denoise"

    def run(self, state: PipelineState, context: PipelineContext) -> None:
        context.residency.transition(("transformer",))
        transformer = context.component("transformer")
        # The adapter must call this hook between sampling steps. Cancellation
        # is cooperative: an in-flight CUDA kernel is allowed to complete.
        denoised = _method(transformer, "denoise")(
            state,
            cancel_check=context.raise_if_cancelled,
        )
        if not isinstance(denoised, dict):
            raise TypeError("transformer.denoise() must return a dict")
        state.video_latents = denoised.get("video_latents")
        state.audio_latents = denoised.get("audio_latents")

    def verify_output(self, state: PipelineState) -> None:
        if state.video_latents is None or state.audio_latents is None:
            raise ValueError("denoiser returned incomplete audio-video latents")


class DecodeVideoStage(PipelineStage):
    name = "decode_video"

    def run(self, state: PipelineState, context: PipelineContext) -> None:
        # This transition is intentionally eviction-first: the transformer may
        # not overlap the video VAE and decoded FP32 frames on a 24 GiB card.
        context.residency.transition(("video_vae",))
        video_vae = context.component("video_vae")
        decoded = _method(video_vae, "decode")(state.video_latents)
        state.decoded_video = _move_result_to_host(decoded)

    def verify_output(self, state: PipelineState) -> None:
        if state.decoded_video is None:
            raise ValueError("video VAE returned no frames")


class DecodeAudioStage(PipelineStage):
    name = "decode_audio"

    def run(self, state: PipelineState, context: PipelineContext) -> None:
        context.residency.transition(("audio_vae",))
        audio_vae = context.component("audio_vae")
        decoded = _method(audio_vae, "decode")(state.audio_latents)
        if isinstance(decoded, tuple) and len(decoded) == 2:
            audio, state.audio_sample_rate = decoded
            state.decoded_audio = _move_result_to_host(audio)
        else:
            state.decoded_audio = _move_result_to_host(decoded)
            state.audio_sample_rate = getattr(audio_vae, "sample_rate", None)

    def verify_output(self, state: PipelineState) -> None:
        if state.decoded_audio is None:
            raise ValueError("audio VAE returned no waveform")


class MuxStage(PipelineStage):
    name = "mux"

    def run(self, state: PipelineState, context: PipelineContext) -> None:
        # Encoding is CPU/I/O work. Free device residency before materializing
        # or retaining an output-sized video tensor.
        context.residency.transition(())
        muxer = context.component("muxer")
        output_path = state.request.output_path
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        state.result = _method(muxer, "write")(
            video=state.decoded_video,
            audio=state.decoded_audio,
            sample_rate=state.audio_sample_rate,
            fps=state.request.fps,
            output_path=output_path,
            cancel_check=context.raise_if_cancelled,
        )

    def verify_output(self, state: PipelineState) -> None:
        if state.result is None:
            raise ValueError("muxer returned no generation result")


def default_h3_stages() -> tuple[PipelineStage, ...]:
    """Return the fixed production ordering for T2AV and first/last-frame AV."""

    return (
        ValidateRequestStage(),
        TextConditioningStage(),
        FrameConditioningStage(),
        PrepareLatentsStage(),
        DenoiseStage(),
        DecodeVideoStage(),
        DecodeAudioStage(),
        MuxStage(),
    )
