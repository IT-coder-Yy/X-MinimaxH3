"""Exact H3 shape accounting used before selecting an execution strategy."""

from __future__ import annotations

from typing import Literal

from .contracts import WorkloadFeatures


class H3WorkloadAnalyzer:
    """Convert a normalized H3 request into model-relevant workload features.

    The equations match the native scheduler and packed-layout implementation:
    video VAE spatial scale 16, DiT spatial patch 2x2, temporal grid 17*n+5,
    latent temporal grid 5*n+2, and 40 stereo audio latent rows per second.
    """

    def __init__(self, *, fps: int = 24, audio_latent_hz: int = 40) -> None:
        if fps <= 0 or audio_latent_hz <= 0:
            raise ValueError("fps and audio_latent_hz must be positive")
        self.fps = int(fps)
        self.audio_latent_hz = int(audio_latent_hz)

    @staticmethod
    def video_latent_frames(frames: int) -> int:
        if frames < 5 or (frames - 5) % 17:
            raise ValueError("H3 frames must satisfy 17*n+5")
        return ((frames - 5) // 17) * 5 + 2

    def analyze(
        self,
        *,
        width: int,
        height: int,
        frames: int,
        text_tokens: int,
        condition_count: int,
        engine: Literal["original", "lora", "reference"],
        actual_evaluations: int,
        forecast_evaluations: int = 0,
        condition_tokens_override: int | None = None,
        latent_frames_override: int | None = None,
        audio_frames_override: int | None = None,
    ) -> WorkloadFeatures:
        if width <= 0 or height <= 0 or width % 32 or height % 32:
            raise ValueError("H3 width and height must be positive multiples of 32")
        if text_tokens <= 0:
            raise ValueError("text_tokens must be positive")
        # Ref2VA publicly accepts 9 images + 3 videos + 3 audios.  Condition
        # token accounting uses their real encoded sizes through the override;
        # the count itself must therefore cover the complete 15-item contract.
        if not 0 <= condition_count <= 15:
            raise ValueError("condition_count must be between 0 and 15")
        if engine not in ("original", "lora", "reference"):
            raise ValueError(f"unsupported H3 engine: {engine}")
        if actual_evaluations <= 0 or forecast_evaluations < 0:
            raise ValueError("invalid evaluation counts")
        if engine == "lora" and forecast_evaluations:
            raise ValueError("the distilled LoRA route cannot use forecast evaluations")

        latent_frames = (
            self.video_latent_frames(frames)
            if latent_frames_override is None
            else int(latent_frames_override)
        )
        if latent_frames <= 0:
            raise ValueError("latent_frames_override must be positive")
        spatial_tokens = (height // 32) * (width // 32)
        video_tokens = latent_frames * spatial_tokens
        condition_tokens = (
            condition_count * spatial_tokens
            if condition_tokens_override is None
            else int(condition_tokens_override)
        )
        if condition_tokens < 0:
            raise ValueError("condition_tokens_override cannot be negative")
        audio_frames = (
            round((frames / self.fps) * self.audio_latent_hz)
            if audio_frames_override is None
            else int(audio_frames_override)
        )
        if audio_frames <= 0:
            raise ValueError("audio_frames_override must be positive")
        audio_tokens = 2 * audio_frames
        packed_tokens = text_tokens + video_tokens + condition_tokens + audio_tokens
        return WorkloadFeatures(
            width=width,
            height=height,
            frames=frames,
            fps=self.fps,
            text_tokens=text_tokens,
            condition_count=condition_count,
            latent_frames=latent_frames,
            spatial_tokens=spatial_tokens,
            video_tokens=video_tokens,
            condition_tokens=condition_tokens,
            audio_tokens=audio_tokens,
            packed_tokens=packed_tokens,
            output_pixel_frames=width * height * frames,
            engine=engine,
            actual_evaluations=actual_evaluations,
            forecast_evaluations=forecast_evaluations,
        )
