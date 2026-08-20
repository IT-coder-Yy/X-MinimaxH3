"""Production model adapters for H3 double-buffered block execution."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from ..runtime import DoubleBufferBlockExecutor, RuntimeConfig, create_stream_coordinator


def _state_tensors(module: nn.Module) -> dict[str, torch.Tensor]:
    # keep_vars exposes the registered storage that execution actually reads.
    return dict(module.state_dict(keep_vars=True))


class TorchModuleBlockBuffer:
    """One preallocated device block whose tensors are overwritten in-place."""

    def __init__(self, module: nn.Module, *, device: str) -> None:
        self.module = module
        self.device = str(device)
        self._targets = _state_tensors(module)
        if not self._targets:
            raise ValueError("H3 block buffer cannot be empty")

    @classmethod
    def from_source(cls, source: nn.Module, *, device: str) -> "TorchModuleBlockBuffer":
        # All H3 blocks share one graph/schema. deepcopy preserves Python-side
        # quant specs and injected kernel callables while allocating independent
        # registered tensor storage for the device slot.
        module = copy.deepcopy(source).eval().requires_grad_(False)
        module.to(device)
        return cls(module, device=device)

    @property
    def registered_bytes(self) -> int:
        seen: set[int] = set()
        total = 0
        for tensor in self._targets.values():
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            total += int(tensor.numel()) * int(tensor.element_size())
        return total

    def validate_source(self, source_block: Any) -> None:
        if not isinstance(source_block, nn.Module):
            raise TypeError("H3 block source must be a torch.nn.Module")
        source = _state_tensors(source_block)
        if source.keys() != self._targets.keys():
            missing = sorted(self._targets.keys() - source.keys())
            extra = sorted(source.keys() - self._targets.keys())
            raise ValueError(
                f"H3 block schema mismatch; missing={missing[:4]}, extra={extra[:4]}"
            )
        for name, target in self._targets.items():
            value = source[name]
            if target.shape != value.shape or target.dtype != value.dtype:
                raise ValueError(
                    f"H3 block tensor mismatch for {name}: "
                    f"{tuple(value.shape)}/{value.dtype} != "
                    f"{tuple(target.shape)}/{target.dtype}"
                )

    def load_from(
        self,
        source_block: Any,
        *,
        block_index: int,
        non_blocking: bool,
    ) -> None:
        del block_index
        self.validate_source(source_block)
        source = _state_tensors(source_block)
        with torch.no_grad():
            for name, target in self._targets.items():
                target.copy_(source[name], non_blocking=non_blocking)


def build_h3_block_executor(
    source_blocks: Sequence[nn.Module],
    config: RuntimeConfig,
    *,
    prefetch_depth: int = 1,
) -> DoubleBufferBlockExecutor:
    if not source_blocks:
        raise ValueError("cannot build H3 block executor without source blocks")
    first = source_blocks[0]
    buffers = tuple(
        TorchModuleBlockBuffer.from_source(first, device=config.device)
        for _ in range(config.block_buffer_count)
    )
    for source in source_blocks:
        buffers[0].validate_source(source)
    return DoubleBufferBlockExecutor(
        buffers,
        create_stream_coordinator(config),
        overlap_copy_compute=prefetch_depth == 1,
    )


__all__ = ["TorchModuleBlockBuffer", "build_h3_block_executor"]
