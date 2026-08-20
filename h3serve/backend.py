"""Backend-neutral Web/API boundary for the in-process Native H3 engine."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ServicePaths
from .contract import GenerationSpec


class BackendError(RuntimeError):
    pass


class JobCancelled(BackendError):
    pass


@dataclass(frozen=True)
class GenerationResult:
    runtime_key: str
    elapsed_seconds: float
    output_path: Path
    inference_plan: dict[str, Any] | None = None


@dataclass(frozen=True)
class CheckpointResult:
    runtime_key: str
    elapsed_seconds: float
    checkpoint_path: Path | None
    preview_path: Path | None
    completed_steps: int
    total_steps: int
    inference_plan: dict[str, Any] | None = None


class NativeBackendManager:
    """Keep persistence/job identifiers outside the model runtime."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.key: str | None = None

    def preflight(self, engine: str) -> dict[str, Any]:
        check = getattr(self.engine, "preflight", None)
        if callable(check):
            return check(engine)
        return {"ready": True, "checks": {"native_engine": True}}

    async def preload(self, engine: str) -> None:
        preload = getattr(self.engine, "preload", None)
        if callable(preload):
            await preload(engine)

    @property
    def warm_state(self) -> dict[str, Any]:
        return dict(getattr(self.engine, "warm_state", {"status": "unsupported"}))

    async def generate(
        self,
        spec: GenerationSpec,
        job_id: str,
        first_frame: Path | None,
        last_frame: Path | None,
        reference_images: tuple[Path, ...],
        reference_videos: tuple[Path, ...],
        reference_audios: tuple[Path, ...],
        cancel_event: asyncio.Event,
        progress_callback: Any | None = None,
        preview_ready_callback: Any | None = None,
        preview_decision_wait: Any | None = None,
        checkpoint_path: Path | None = None,
        resume_checkpoint_path: Path | None = None,
    ) -> GenerationResult | CheckpointResult:
        from .native_engine.engine import (
            NativeCheckpointResult,
            NativeGenerationCancelled,
        )

        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", job_id).strip("._")
        if not safe_name:
            raise BackendError("job id does not contain a safe output name")
        output_path = self.engine.output_root / f"{safe_name[:128]}.mp4"
        try:
            result = await self.engine.generate(
                spec, first_frame, last_frame, reference_images, reference_videos, reference_audios, cancel_event, output_path,
                progress_callback=progress_callback,
                preview_ready_callback=preview_ready_callback,
                preview_decision_wait=preview_decision_wait,
                checkpoint_path=checkpoint_path,
                resume_checkpoint_path=resume_checkpoint_path,
            )
        except NativeGenerationCancelled as error:
            raise JobCancelled(str(error)) from error
        self.key = result.runtime_key
        if isinstance(result, NativeCheckpointResult):
            return CheckpointResult(
                runtime_key=result.runtime_key,
                elapsed_seconds=result.elapsed_seconds,
                checkpoint_path=result.checkpoint_path,
                preview_path=result.preview_path,
                completed_steps=result.completed_steps,
                total_steps=result.total_steps,
                inference_plan=getattr(result, "inference_plan", None),
            )
        return GenerationResult(
            runtime_key=result.runtime_key,
            elapsed_seconds=result.elapsed_seconds,
            output_path=result.output_path,
            inference_plan=getattr(result, "inference_plan", None),
        )

    async def stop(self) -> None:
        await self.engine.close()
        self.key = None

    def configure_memory_profile(self, profile: Any) -> None:
        factory = getattr(self.engine, "_factory", None)
        configure = getattr(factory, "set_memory_profile", None)
        if not callable(configure):
            raise BackendError("native backend does not support memory-profile changes")
        configure(profile)

    def configure_output_root(self, output_root: Path) -> None:
        configure = getattr(self.engine, "set_output_root", None)
        if not callable(configure):
            raise BackendError("native backend does not support workspace changes")
        configure(output_root)


def build_native_backend(paths: ServicePaths, *, memory_profile: Any = None) -> NativeBackendManager:
    """Construct the production backend without loading large weights yet."""

    from .native_engine import NativeHotH3Engine
    from .native_engine.session_factory import NativeSessionFactory, NativeSessionPaths

    factory = NativeSessionFactory(
        NativeSessionPaths(
            model_root=paths.model_dir,
            minimax_source=paths.minimax_source_dir,
            lightx_source=paths.lightx_source_dir,
            turbo_curve=paths.turbo_curve_path,
            output_root=paths.output_dir,
        ),
        memory_profile=memory_profile,
    )
    return NativeBackendManager(
        NativeHotH3Engine(factory, output_root=paths.output_dir)
    )


__all__ = [
    "BackendError",
    "CheckpointResult",
    "GenerationResult",
    "JobCancelled",
    "NativeBackendManager",
    "build_native_backend",
]
