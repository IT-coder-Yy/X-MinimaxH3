"""Typed runtime policy for the supported RTX 4090 deployment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OffloadMode(str, Enum):
    """Transformer residency strategy.

    ``BLOCK`` is the expected 24 GiB production mode. ``MODEL`` is useful for
    smaller future checkpoints, while ``RESIDENT`` is primarily a diagnostic
    mode and is not expected to fit the current H3 deployment checkpoint.
    """

    BLOCK = "block"
    MODEL = "model"
    RESIDENT = "resident"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Hardware and memory policy fixed at service startup.

    Per-request creative controls deliberately do not live here.  The native
    service targets one RTX 4090, one generation at a time, and batch size one.
    """

    device: str = "cuda:0"
    expected_compute_capability: tuple[int, int] = (8, 9)
    max_device_bytes: int = 23 * 1024**3
    batch_size: int = 1
    offload_mode: OffloadMode = OffloadMode.BLOCK
    block_buffer_count: int = 2
    pin_host_weights: bool = True
    copy_stream_priority: int = 0
    compute_stream_priority: int = -1
    clear_cache_on_phase_transition: bool = False
    retain_block_buffers_between_requests: bool = True
    weight_tier: str | None = None
    resource_profile: str | None = None

    def __post_init__(self) -> None:
        if self.batch_size != 1:
            raise ValueError("the RTX 4090 runtime supports batch_size=1 only")
        if self.block_buffer_count != 2:
            raise ValueError("block offload requires exactly two device buffers")
        if self.max_device_bytes <= 0:
            raise ValueError("max_device_bytes must be positive")
        if self.device != "cpu" and not self.device.startswith("cuda:"):
            raise ValueError("device must be 'cpu' or an explicit CUDA device such as 'cuda:0'")

    @classmethod
    def for_cuda_device(
        cls,
        *,
        weight_tier: str,
        provisioned_limit_gib: float,
        backend_profile: str,
    ) -> "RuntimeConfig":
        """Build the device policy for one release resource backend.

        The allocator ceiling follows the backend's planner budget, which
        already retains headroom for the CUDA context and custom-kernel
        workspaces on top of the provisioned VRAM profile.
        """

        from ...deployment_profiles import get_resource_backend

        backend = get_resource_backend(backend_profile, weight_tier=weight_tier)
        return cls(
            device="cuda:0",
            expected_compute_capability=(8, 9),
            max_device_bytes=int(round(backend.planner_budget_gib * 1024**3)),
            resource_profile=backend.profile_id,
            weight_tier=backend.weight_tier,
        )

    @classmethod
    def cpu_test(cls) -> "RuntimeConfig":
        """Return a no-CUDA configuration for unit tests and adapter bring-up."""

        return cls(
            device="cpu",
            expected_compute_capability=(0, 0),
            max_device_bytes=8 * 1024**3,
            pin_host_weights=False,
            clear_cache_on_phase_transition=False,
        )
