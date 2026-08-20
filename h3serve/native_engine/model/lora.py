"""Runtime LoRA adapters for H3, including the pruned AdaLN curve bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class LowRankUpdate:
    down: torch.Tensor
    up: torch.Tensor
    strength: float = 1.0
    alpha: float | None = None

    def __post_init__(self) -> None:
        if self.down.ndim != 2 or self.up.ndim != 2:
            raise ValueError("LoRA tensors must be matrices")
        if self.up.shape[1] != self.down.shape[0]:
            raise ValueError("LoRA rank mismatch")

    @property
    def scale(self) -> float:
        rank = int(self.down.shape[0])
        return self.strength * (1.0 if self.alpha is None else self.alpha / rank)

    def apply(self, value: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        down = self.down.to(device=value.device, dtype=value.dtype)
        up = self.up.to(device=value.device, dtype=value.dtype)
        low_rank = F.linear(value, down)
        output_2d = base_output.reshape(-1, base_output.shape[-1])
        low_rank_2d = low_rank.reshape(-1, low_rank.shape[-1])
        if torch.is_grad_enabled() and (
            output_2d.requires_grad or low_rank_2d.requires_grad or up.requires_grad
        ):
            return (
                output_2d + self.scale * low_rank_2d.mm(up.T)
            ).reshape_as(base_output)
        torch.addmm(
            output_2d,
            low_rank_2d,
            up.T,
            beta=1.0,
            alpha=self.scale,
            out=output_2d,
        )
        return base_output


@dataclass(frozen=True, slots=True)
class AdaLNCurveRows:
    """Curve coordinates interpolated once and shared by all 50 blocks."""

    compressed: torch.Tensor
    full_silu: torch.Tensor | None = None


class RuntimeLoRALinear(nn.Module):
    """Preserve a quantized base path and add the LoRA in activation space."""

    def __init__(self, base: nn.Module, update: LowRankUpdate) -> None:
        super().__init__()
        self.base = base
        self.update = ResidentLowRankUpdate(update)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.update.apply(value, self.base(value))

    def set_lora_enabled(self, enabled: bool) -> None:
        self.update.enabled = bool(enabled)


class ResidentLowRankUpdate(nn.Module):
    """A LoRA pair that follows its owning module across residency changes.

    ``LowRankUpdate`` is the immutable checkpoint record.  Keeping that record
    directly on a module leaves its tensors invisible to ``Module.to()`` and
    would copy all 259 Larry pairs from CPU inside every forward after an
    offload.  Registering the pair as buffers makes one explicit phase transfer
    authoritative and keeps hot inference free of hidden PCIe traffic.
    """

    def __init__(
        self, update: LowRankUpdate, *, allow_dtype_conversion: bool = False
    ) -> None:
        super().__init__()
        self.register_buffer("down", update.down)
        self.register_buffer("up", update.up)
        self.strength = float(update.strength)
        self.alpha = update.alpha
        self.allow_dtype_conversion = allow_dtype_conversion
        # This flag is request-local execution policy, not model state.  It is
        # deliberately a Python boolean rather than a tensor/buffer so toggling
        # the adapter never changes checkpoint storage or triggers a transfer.
        self.enabled = True

    @property
    def scale(self) -> float:
        rank = int(self.down.shape[0])
        return self.strength * (
            1.0 if self.alpha is None else float(self.alpha) / rank
        )

    def apply(self, value: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return base_output
        if self.down.device != value.device or self.up.device != value.device:
            raise RuntimeError(
                "LoRA residency mismatch: weights must move with their owning module"
            )
        dtype_mismatch = self.down.dtype != value.dtype or self.up.dtype != value.dtype
        if dtype_mismatch and not self.allow_dtype_conversion:
            raise RuntimeError(
                "LoRA dtype mismatch: per-forward weight conversion is forbidden"
            )
        # AdaLN is the one documented exception: its base is an FP32 island
        # while Larry stores the low-rank checkpoint in BF16.  Converting one
        # active block at a time costs less peak memory than retaining all 51
        # promoted updates on a 24GB card.
        down = self.down.to(dtype=value.dtype) if dtype_mismatch else self.down
        up = self.up.to(dtype=value.dtype) if dtype_mismatch else self.up
        low_rank = F.linear(value, down)
        output_2d = base_output.reshape(-1, base_output.shape[-1])
        low_rank_2d = low_rank.reshape(-1, low_rank.shape[-1])
        if torch.is_grad_enabled() and (
            output_2d.requires_grad or low_rank_2d.requires_grad or up.requires_grad
        ):
            return (
                output_2d + self.scale * low_rank_2d.mm(up.T)
            ).reshape_as(base_output)
        torch.addmm(
            output_2d,
            low_rank_2d,
            up.T,
            beta=1.0,
            alpha=self.scale,
            out=output_2d,
        )
        return base_output


def load_larry_updates(
    state: Mapping[str, torch.Tensor], *, strength: float = 1.0
) -> dict[str, LowRankUpdate]:
    """Map Larry's native H3 names directly; no Comfy key-map is required."""

    suffix = ".lora_A.weight"
    updates: dict[str, LowRankUpdate] = {}
    for key, down in state.items():
        if not key.endswith(suffix):
            continue
        module_name = key[: -len(suffix)]
        up_key = f"{module_name}.lora_B.weight"
        if up_key not in state:
            raise KeyError(f"missing paired Larry tensor {up_key!r}")
        alpha_value = state.get(f"{module_name}.alpha")
        alpha = None if alpha_value is None else float(alpha_value.item())
        updates[module_name] = LowRankUpdate(
            down=down,
            up=state[up_key],
            strength=strength,
            alpha=alpha,
        )
    return updates


def load_larry_updates_from_safetensors(
    path: str,
    *,
    strength: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, LowRankUpdate]:
    """Load the 259 native-name pairs without materializing unrelated tensors."""

    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("safetensors is required to load Larry LoRA") from error
    updates: dict[str, LowRankUpdate] = {}
    suffix = ".lora_A.weight"
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        for key in sorted(keys):
            if not key.endswith(suffix):
                continue
            module_name = key[: -len(suffix)]
            up_key = f"{module_name}.lora_B.weight"
            if up_key not in keys:
                raise KeyError(f"missing paired Larry tensor {up_key!r}")
            alpha_key = f"{module_name}.alpha"
            alpha = (
                float(checkpoint.get_tensor(alpha_key).item())
                if alpha_key in keys
                else None
            )
            down = checkpoint.get_tensor(key)
            up = checkpoint.get_tensor(up_key)
            if device is not None or dtype is not None:
                down = down.to(device=device or down.device, dtype=dtype or down.dtype)
                up = up.to(device=device or up.device, dtype=dtype or up.dtype)
            updates[module_name] = LowRankUpdate(
                down=down,
                up=up,
                strength=strength,
                alpha=alpha,
            )
    return updates


def interpolate_curve(table: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    """Linear interpolation over a `[grid, features]` table at t in [0,1]."""

    if table.ndim != 2 or table.shape[0] < 2:
        raise ValueError("curve table must be [grid>=2, features]")
    position = timesteps.float().clamp(0.0, 1.0) * (table.shape[0] - 1)
    lower = position.floor().long().clamp(max=table.shape[0] - 2)
    fraction = (position - lower).unsqueeze(-1)
    # Larry's bundled full SiLU grid is BF16, while its reference adapter
    # deliberately interpolates grid endpoints in FP32 before casting the
    # selected rows for the low-rank projection.  This also avoids mixed-dtype
    # torch.lerp rejection on recent PyTorch.
    return torch.lerp(
        table[lower].float(), table[lower + 1].float(), fraction.float()
    )


class PrunedCurveAdaLN(nn.Module):
    """Project the compressed base curve and optionally restore Larry AdaLN.

    The pruned checkpoint's base linear consumes 8-D coordinates interpolated
    from ``adaln_t_table``. Larry's AdaLN LoRA consumes the original 2688-D
    ``silu(time_embedding)`` curve. Both tables are therefore needed for Turbo;
    applying the 2688-D update to the 8-D base weight is mathematically invalid.
    """

    def __init__(
        self,
        base_linear: nn.Module,
        compressed_curve: torch.Tensor | None,
        *,
        full_silu_curve: torch.Tensor | None = None,
        lora_update: LowRankUpdate | None = None,
    ) -> None:
        super().__init__()
        if compressed_curve is not None and compressed_curve.ndim != 2:
            raise ValueError("compressed AdaLN curve must be rank two")
        if compressed_curve is not None and (full_silu_curve is None) != (lora_update is None):
            raise ValueError("full_silu_curve and lora_update must be supplied together")
        self.base_linear = base_linear
        self.register_buffer("compressed_curve", compressed_curve)
        self.register_buffer("full_silu_curve", full_silu_curve)
        self.lora_update = (
            None
            if lora_update is None
            else ResidentLowRankUpdate(lora_update, allow_dtype_conversion=True)
        )

    def forward(self, timesteps: torch.Tensor | AdaLNCurveRows) -> torch.Tensor:
        if isinstance(timesteps, AdaLNCurveRows):
            compressed = timesteps.compressed
            full = timesteps.full_silu
        else:
            if self.compressed_curve is None:
                raise ValueError("precomputed AdaLNCurveRows are required")
            compressed = interpolate_curve(
                self.compressed_curve.to(timesteps.device), timesteps
            )
            full = None
        compressed = compressed.to(
            dtype=next(self.base_linear.parameters(), compressed).dtype
        )
        result = self.base_linear(compressed)
        if self.lora_update is not None and self.lora_update.enabled:
            if full is None:
                if self.full_silu_curve is None or not isinstance(timesteps, torch.Tensor):
                    raise ValueError("full SiLU curve rows are required for AdaLN LoRA")
                full = interpolate_curve(
                    self.full_silu_curve.to(timesteps.device), timesteps
                )
            full = full.to(result.dtype)
            result = self.lora_update.apply(full, result)
        return result


def set_lora_enabled(module: nn.Module, enabled: bool) -> int:
    """Enable/disable every resident adapter without rebuilding the base graph.

    The return value is the number of LoRA pairs affected and is useful for
    startup/preflight assertions.  Block-offload device slots are rebuilt for
    each request from these source modules, so their Python-side switch is
    copied together with the graph before any block executes.
    """

    count = 0
    for child in module.modules():
        if isinstance(child, ResidentLowRankUpdate):
            child.enabled = bool(enabled)
            count += 1
    return count
