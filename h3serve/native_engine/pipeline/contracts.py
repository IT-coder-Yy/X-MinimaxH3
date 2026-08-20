"""Request and in-flight state contracts for the native H3 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Sampling controls shared by the high-fidelity and accelerated engines."""

    engine: Literal["original", "lora"] = "original"
    num_steps: int = 20
    actual_step_indices: tuple[int, ...] | None = tuple(range(20))
    sampler: Literal["res_multistep", "turbo"] = "res_multistep"
    scheduler: Literal["simple"] = "simple"
    lora_strength: float = 0.0

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.engine == "original" and self.sampler != "res_multistep":
            raise ValueError("original engine requires the res_multistep sampler")
        if self.engine == "lora" and self.sampler != "turbo":
            raise ValueError("lora engine requires the turbo sampler")
        if self.engine == "lora" and self.actual_step_indices is not None:
            raise ValueError("lora engine executes its complete distilled schedule")
        if self.engine == "original" and self.actual_step_indices is None:
            raise ValueError("original engine requires explicit actual-step indices")
        if self.engine == "original" and self.lora_strength != 0.0:
            raise ValueError("original engine cannot apply a LoRA")
        if self.engine == "lora" and self.lora_strength <= 0.0:
            raise ValueError("lora engine requires a positive LoRA strength")
        if self.actual_step_indices is not None:
            if not self.actual_step_indices:
                raise ValueError("actual_step_indices cannot be empty")
            if tuple(sorted(set(self.actual_step_indices))) != self.actual_step_indices:
                raise ValueError("actual_step_indices must be sorted and unique")
            if self.actual_step_indices[0] < 0 or self.actual_step_indices[-1] >= self.num_steps:
                raise ValueError("actual_step_indices must fall inside the sampling schedule")


@dataclass(frozen=True, slots=True)
class GenerationInput:
    """One H3 request after public resolution/duration normalization."""

    prompt: str
    width: int
    height: int
    num_frames: int
    seed: int
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    fps: int = 24
    first_frame: Path | None = None
    last_frame: Path | None = None
    output_path: Path | None = None
    batch_size: int = 1

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt cannot be empty")
        if self.batch_size != 1:
            raise ValueError("native RTX 4090 pipeline supports batch_size=1 only")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.width % 32 or self.height % 32:
            raise ValueError("H3 width and height must be multiples of 32")
        if self.num_frames < 5 or (self.num_frames - 5) % 17:
            raise ValueError("H3 num_frames must satisfy 17*n+5")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        for role, path in (("first_frame", self.first_frame), ("last_frame", self.last_frame)):
            if path is not None and not path.is_file():
                raise ValueError(f"{role} does not exist: {path}")


@dataclass(slots=True)
class StageMetrics:
    elapsed_seconds: dict[str, float] = field(default_factory=dict)
    residency_after_stage: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineState:
    """Declared mutable state passed from one stage to the next."""

    request: GenerationInput
    text_conditioning: Any = None
    frame_conditioning: Any = None
    packed_layout: Any = None
    video_latents: Any = None
    audio_latents: Any = None
    decoded_video: Any = None
    decoded_audio: Any = None
    audio_sample_rate: int | None = None
    result: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: StageMetrics = field(default_factory=StageMetrics)
