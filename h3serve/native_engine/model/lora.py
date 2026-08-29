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
class SlicedLowRankUpdate:
    """Independent LoRA projections that target slices of one fused linear.

    Diffusers stores MiniMax H3 Q/K/V adapters as three separate rank-128
    projections, while the native runtime deliberately keeps the checkpoint's
    fused QKV base matrix.  A block-diagonal merge would be exact but would
    triple the expensive output GEMM and materialize more than a GiB of zeros.
    Retaining the three updates and writing each into its own output slice is
    both mathematically exact and faithful to the upstream execution cost.
    """

    slices: tuple[tuple[int, int, LowRankUpdate], ...]

    def __post_init__(self) -> None:
        previous_stop = 0
        for start, stop, update in self.slices:
            if not 0 <= start < stop or start < previous_stop:
                raise ValueError("LoRA output slices must be ordered and disjoint")
            if update.up.shape[0] != stop - start:
                raise ValueError("LoRA output slice width does not match update")
            previous_stop = stop


@dataclass(frozen=True, slots=True)
class AdaLNCurveRows:
    """Curve coordinates interpolated once and shared by all 50 blocks."""

    compressed: torch.Tensor
    full_silu: torch.Tensor | None = None


class RuntimeLoRALinear(nn.Module):
    """Preserve a quantized base path and add the LoRA in activation space."""

    def __init__(
        self,
        base: nn.Module,
        update: LowRankUpdate | SlicedLowRankUpdate,
    ) -> None:
        super().__init__()
        self.base = base
        self.update = (
            ResidentSlicedLowRankUpdate(update)
            if isinstance(update, SlicedLowRankUpdate)
            else ResidentLowRankUpdate(update)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.update.apply(value, self.base(value))

    def forward_output_slice(
        self, value: torch.Tensor, start: int, stop: int
    ) -> torch.Tensor:
        """Preserve compact Q/K/V projection without dropping the adapter."""

        base_slice = getattr(self.base, "forward_output_slice", None)
        if not callable(base_slice):
            raise RuntimeError(
                "the quantized LoRA base does not support output-row slicing"
            )
        return self.update.apply_output_slice(
            value,
            base_slice(value, start, stop),
            start,
            stop,
        )

    def prepare_output_slices(self, value: torch.Tensor):
        """Prepare only the quantized base input; LoRA math stays unchanged."""

        prepare = getattr(self.base, "prepare_output_slices", None)
        if not callable(prepare):
            raise RuntimeError(
                "the quantized LoRA base does not support prepared output slices"
            )
        return prepare(value)

    def forward_prepared_output_slice(
        self,
        value: torch.Tensor,
        prepared,
        row_start: int,
        row_stop: int,
        output_start: int,
        output_stop: int,
    ) -> torch.Tensor:
        """Reuse base activation quantization without altering LoRA GEMMs."""

        project = getattr(self.base, "forward_prepared_output_slice", None)
        if not callable(project):
            raise RuntimeError(
                "the quantized LoRA base does not support prepared output slices"
            )
        value_rows = value[int(row_start):int(row_stop)]
        base_output = project(
            value,
            prepared,
            row_start,
            row_stop,
            output_start,
            output_stop,
        )
        return self.update.apply_output_slice(
            value_rows,
            base_output,
            output_start,
            output_stop,
        )

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

    def apply_output_slice(
        self,
        value: torch.Tensor,
        base_output: torch.Tensor,
        start: int,
        stop: int,
    ) -> torch.Tensor:
        """Add only the requested contiguous output rows of one LoRA pair."""

        if not self.enabled:
            return base_output
        start, stop = int(start), int(stop)
        if not 0 <= start < stop <= self.up.shape[0]:
            raise ValueError("LoRA output slice lies outside the update")
        if base_output.shape[-1] != stop - start:
            raise ValueError("LoRA output slice does not match the base projection")
        if self.down.device != value.device or self.up.device != value.device:
            raise RuntimeError(
                "LoRA residency mismatch: weights must move with their owning module"
            )
        dtype_mismatch = (
            self.down.dtype != value.dtype or self.up.dtype != value.dtype
        )
        if dtype_mismatch and not self.allow_dtype_conversion:
            raise RuntimeError(
                "LoRA dtype mismatch: per-forward weight conversion is forbidden"
            )
        down = self.down.to(dtype=value.dtype) if dtype_mismatch else self.down
        up = self.up[start:stop]
        up = up.to(dtype=value.dtype) if dtype_mismatch else up
        low_rank = F.linear(value, down)
        output_2d = base_output.reshape(-1, base_output.shape[-1])
        low_rank_2d = low_rank.reshape(-1, low_rank.shape[-1])
        if torch.is_grad_enabled() and (
            output_2d.requires_grad
            or low_rank_2d.requires_grad
            or up.requires_grad
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


class ResidentSlicedLowRankUpdate(nn.Module):
    """Resident form of separate Q/K/V LoRAs over one fused projection."""

    def __init__(self, update: SlicedLowRankUpdate) -> None:
        super().__init__()
        self.slices = tuple((start, stop) for start, stop, _ in update.slices)
        self.updates = nn.ModuleList(
            ResidentLowRankUpdate(item) for _, _, item in update.slices
        )

    @property
    def enabled(self) -> bool:
        return any(update.enabled for update in self.updates)

    @enabled.setter
    def enabled(self, enabled: bool) -> None:
        for update in self.updates:
            update.enabled = bool(enabled)

    def apply(self, value: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        for (start, stop), update in zip(self.slices, self.updates, strict=True):
            target = base_output[..., start:stop]
            base_output[..., start:stop] = update.apply(value, target)
        return base_output

    def apply_output_slice(
        self,
        value: torch.Tensor,
        base_output: torch.Tensor,
        start: int,
        stop: int,
    ) -> torch.Tensor:
        start, stop = int(start), int(stop)
        if base_output.shape[-1] != stop - start:
            raise ValueError("LoRA output slice does not match the base projection")
        for (global_start, global_stop), update in zip(
            self.slices, self.updates, strict=True
        ):
            overlap_start = max(start, global_start)
            overlap_stop = min(stop, global_stop)
            if overlap_start >= overlap_stop:
                continue
            output_start = overlap_start - start
            output_stop = overlap_stop - start
            update_start = overlap_start - global_start
            update_stop = overlap_stop - global_start
            target = base_output[..., output_start:output_stop]
            base_output[..., output_start:output_stop] = update.apply_output_slice(
                value,
                target,
                update_start,
                update_stop,
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


def _load_lightx2v_updates_from_checkpoint(
    checkpoint,
    *,
    strength: float,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> dict[str, LowRankUpdate | SlicedLowRankUpdate]:
    keys = set(checkpoint.keys())
    metadata = checkpoint.metadata() or {}
    try:
        alpha = float(metadata.get("alpha", "8"))
    except (TypeError, ValueError) as error:
        raise ValueError("LightX2V LoRA metadata has an invalid alpha") from error
    a_suffix = ".lora_A.default.weight"
    b_suffix = ".lora_B.default.weight"
    raw_modules = sorted(key[: -len(a_suffix)] for key in keys if key.endswith(a_suffix))
    if len(raw_modules) != 312:
        raise ValueError(f"LightX2V LoRA pair count {len(raw_modules)} != 312")

    def pair(module_name: str) -> LowRankUpdate:
        a_key = f"{module_name}{a_suffix}"
        b_key = f"{module_name}{b_suffix}"
        if a_key not in keys or b_key not in keys:
            raise KeyError(f"missing paired LightX2V tensors for {module_name!r}")
        down = checkpoint.get_tensor(a_key)
        up = checkpoint.get_tensor(b_key)
        if device is not None or dtype is not None:
            down = down.to(device=device or down.device, dtype=dtype or down.dtype)
            up = up.to(device=device or up.device, dtype=dtype or up.dtype)
        return LowRankUpdate(
            down=down,
            up=up,
            strength=strength,
            alpha=alpha,
        )

    updates: dict[str, LowRankUpdate | SlicedLowRankUpdate] = {}

    def convert_stack(source_prefix: str, target_prefix: str, count: int) -> None:
        for index in range(count):
            source = f"{source_prefix}.{index}"
            target = f"{target_prefix}.{index}"
            q = pair(f"{source}.attn.to_q")
            k = pair(f"{source}.attn.to_k")
            v = pair(f"{source}.attn.to_v")
            q_width = int(q.up.shape[0])
            k_width = int(k.up.shape[0])
            v_width = int(v.up.shape[0])
            updates[f"{target}.attn.qkv_proj"] = SlicedLowRankUpdate((
                (0, q_width, q),
                (q_width, q_width + k_width, k),
                (q_width + k_width, q_width + k_width + v_width, v),
            ))
            updates[f"{target}.attn.out_proj"] = pair(f"{source}.attn.to_out.0")
            updates[f"{target}.mlp.fc1"] = pair(f"{source}.ff.net.0.proj")
            updates[f"{target}.mlp.fc2"] = pair(f"{source}.ff.net.2")

    convert_stack("transformer_blocks", "blocks", 50)
    convert_stack("token_refiner.refiner_blocks", "token_refiner.blocks", 2)
    consumed = 50 * 6 + 2 * 6
    if consumed != len(raw_modules) or len(updates) != 208:
        raise RuntimeError("LightX2V LoRA conversion did not consume the full checkpoint")
    return updates


def load_h3_updates_from_safetensors(
    path: str,
    *,
    strength: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, LowRankUpdate | SlicedLowRankUpdate]:
    """Load either native-name or LightX2V Diffusers-format H3 adapters."""

    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("safetensors is required to load H3 LoRA") from error
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        if any(key.endswith(".lora_A.default.weight") for key in keys):
            return _load_lightx2v_updates_from_checkpoint(
                checkpoint,
                strength=strength,
                device=device,
                dtype=dtype,
            )
    return load_larry_updates_from_safetensors(
        path,
        strength=strength,
        device=device,
        dtype=dtype,
    )


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
