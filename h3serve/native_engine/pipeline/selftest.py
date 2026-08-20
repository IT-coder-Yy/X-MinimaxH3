"""Dependency-free CPU smoke test for native lifecycle and block scheduling."""

from __future__ import annotations

from typing import Any

from ..runtime import (
    DoubleBufferBlockExecutor,
    HostComponentResidency,
    ResidencyManager,
    RuntimeConfig,
    TorchModuleResidency,
    create_stream_coordinator,
)
from .contracts import GenerationInput
from .executor import NativeH3Pipeline, PipelineCancelled
from .stages import default_h3_stages


class _FakeComponent:
    sample_rate = 48_000

    def to(self, device: str, *, non_blocking: bool = False) -> "_FakeComponent":
        return self

    def encode(self, request: GenerationInput) -> str:
        return request.prompt

    def prepare(self, state: Any) -> dict[str, Any]:
        return {"video_latents": 1, "audio_latents": 2, "packed_layout": ()}

    def denoise(self, state: Any, *, cancel_check: Any) -> dict[str, Any]:
        cancel_check()
        return {
            "video_latents": state.video_latents + 10,
            "audio_latents": state.audio_latents + 20,
        }

    def decode(self, latent: int) -> int:
        return latent * 2

    def write(self, **values: Any) -> dict[str, Any]:
        return values


class _FakeBuffer:
    value: int

    def load_from(
        self,
        source_block: int,
        *,
        block_index: int,
        non_blocking: bool,
    ) -> None:
        self.value = source_block


def run() -> None:
    config = RuntimeConfig.cpu_test()
    residency = ResidencyManager(config)
    for name in ("text_encoder", "video_vae", "transformer", "audio_vae"):
        residency.register(TorchModuleResidency(name, _FakeComponent(), 0))
    residency.register(HostComponentResidency("scheduler", _FakeComponent()))
    residency.register(HostComponentResidency("muxer", _FakeComponent()))

    pipeline = NativeH3Pipeline(default_h3_stages(), residency)
    request = GenerationInput(
        prompt="native pipeline smoke",
        width=864,
        height=480,
        num_frames=73,
        seed=4090,
    )
    state = pipeline.generate(request)
    assert state.decoded_video == 22
    assert state.decoded_audio == 44
    assert not residency.active_names

    calls = 0

    def cancel_during_denoise() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 6

    try:
        pipeline.generate(request, cancel_check=cancel_during_denoise)
    except PipelineCancelled:
        pass
    else:
        raise AssertionError("cooperative cancellation was not propagated")
    assert not residency.active_names

    executor = DoubleBufferBlockExecutor(
        [_FakeBuffer(), _FakeBuffer()],
        create_stream_coordinator(config),
    )
    visits: list[tuple[int, int]] = []

    def run_block(
        index: int,
        block: _FakeBuffer,
        hidden: int,
        shared: Any,
    ) -> int:
        visits.append((index, block.value))
        return hidden + block.value

    assert executor.run([1, 2, 3, 4], 0, run_block) == 10
    assert visits == [(0, 1), (1, 2), (2, 3), (3, 4)]


if __name__ == "__main__":
    run()
    print("native runtime/pipeline CPU self-test: PASS")
