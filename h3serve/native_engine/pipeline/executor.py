"""Small stage executor independent from FastVideo and ComfyUI."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Iterable

from ..runtime.residency import ResidencyManager
from .contracts import GenerationInput, PipelineState


class StageError(RuntimeError):
    """Wrap a stage failure with its stable stage name."""


class PipelineCancelled(RuntimeError):
    """A request was cancelled at a safe pipeline or denoise boundary."""


@dataclass(slots=True)
class PipelineContext:
    residency: ResidencyManager
    cancel_check: Callable[[], bool]

    def component(self, name: str) -> Any:
        return self.residency.component(name)

    def raise_if_cancelled(self) -> None:
        if self.cancel_check():
            raise PipelineCancelled("native H3 generation cancelled")


class PipelineStage(ABC):
    """One deterministic verb in the H3 generation lifecycle."""

    name: str

    def verify_input(self, state: PipelineState) -> None:
        return None

    def verify_output(self, state: PipelineState) -> None:
        return None

    @abstractmethod
    def run(self, state: PipelineState, context: PipelineContext) -> None:
        """Mutate only declared :class:`PipelineState` fields."""

    def __call__(self, state: PipelineState, context: PipelineContext) -> None:
        self.verify_input(state)
        started = perf_counter()
        try:
            self.run(state, context)
            self.verify_output(state)
        except PipelineCancelled:
            raise
        except Exception as exc:
            raise StageError(f"native pipeline stage '{self.name}' failed: {exc}") from exc
        finally:
            state.metrics.elapsed_seconds[self.name] = perf_counter() - started
            state.metrics.residency_after_stage[self.name] = tuple(
                sorted(context.residency.active_names)
            )


class NativeH3Pipeline:
    """Composable batch-one H3 audio-video inference pipeline."""

    def __init__(
        self,
        stages: Iterable[PipelineStage],
        residency: ResidencyManager,
    ) -> None:
        self._stages = tuple(stages)
        if not self._stages:
            raise ValueError("native pipeline requires at least one stage")
        names = [stage.name for stage in self._stages]
        if len(names) != len(set(names)):
            raise ValueError("native pipeline stage names must be unique")
        self._residency = residency

    @property
    def stages(self) -> tuple[PipelineStage, ...]:
        return self._stages

    def generate(
        self,
        request: GenerationInput,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> PipelineState:
        state = PipelineState(request=request)
        context = PipelineContext(
            residency=self._residency,
            cancel_check=cancel_check or (lambda: False),
        )
        try:
            for stage in self._stages:
                context.raise_if_cancelled()
                stage(state, context)
            return state
        except Exception:
            # A failed request must not strand a 20+ GiB component on device.
            self._residency.release_all()
            raise

    def close(self) -> None:
        self._residency.release_all()
