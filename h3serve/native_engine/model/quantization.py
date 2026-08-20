"""Comfy checkpoint quantization contract without a ComfyUI dependency.

The checkpoint parser is a clean compatibility implementation. Optimized CUDA
execution is deliberately injected as a callable, so this model package stays
importable on CPU and does not depend on a particular kernel distribution.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

Int8Kernel = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, int],
    torch.Tensor,
]


def comfy_kitchen_int8_kernel(
    value: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    convrot_group_size: int,
) -> torch.Tensor:
    """Lazy Apache Comfy-Kitchen binding; this does not import ComfyUI."""

    try:
        from comfy_kitchen import int8_linear
    except ImportError as error:
        raise RuntimeError("comfy-kitchen is required for the production INT8 path") from error
    return int8_linear(
        value,
        qweight,
        scale,
        bias,
        value.dtype,
        convrot=bool(convrot_group_size),
        convrot_groupsize=convrot_group_size or 256,
    )


@dataclass(frozen=True, slots=True)
class ComfyQuantSpec:
    format: str
    convrot: bool = False
    convrot_groupsize: int = 256

    @classmethod
    def decode(cls, raw: object) -> "ComfyQuantSpec":
        if isinstance(raw, torch.Tensor):
            if raw.device.type != "cpu":
                raw = raw.cpu()
            payload = bytes(raw.to(torch.uint8).tolist())
        elif isinstance(raw, (bytes, bytearray, memoryview)):
            payload = bytes(raw)
        elif isinstance(raw, str):
            payload = raw.encode("utf-8")
        elif isinstance(raw, Mapping):
            data = dict(raw)
            return cls._from_mapping(data)
        else:
            raise TypeError(f"unsupported comfy_quant payload: {type(raw).__name__}")
        return cls._from_mapping(json.loads(payload.decode("utf-8")))

    @classmethod
    def _from_mapping(cls, data: Mapping[str, object]) -> "ComfyQuantSpec":
        params = data.get("params", {})
        if not isinstance(params, Mapping):
            params = {}
        return cls(
            format=str(data.get("format", "")),
            convrot=bool(data.get("convrot", params.get("convrot", False))),
            convrot_groupsize=int(
                data.get("convrot_groupsize", params.get("convrot_groupsize", 256))
            ),
        )

    def validate_supported(self) -> None:
        if self.format != "int8_tensorwise":
            raise ValueError(f"native core currently supports int8_tensorwise, got {self.format!r}")
        size = self.convrot_groupsize
        if self.convrot and (size < 4 or size & (size - 1) or size.bit_length() % 2 == 0):
            raise ValueError("ConvRot group size must be a power of four")


def groupwise_hadamard(value: torch.Tensor, group_size: int) -> torch.Tensor:
    """Normalized regular H4-Kronecker transform used by ConvRot weights."""

    if group_size < 4 or group_size & (group_size - 1) or group_size.bit_length() % 2 == 0:
        raise ValueError("group_size must be a power of four")
    if value.shape[-1] % group_size:
        raise ValueError("last dimension must be divisible by group_size")
    h4 = value.new_tensor(
        ((1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1))
    )
    matrix = h4
    width = 4
    while width < group_size:
        matrix = torch.kron(matrix, h4)
        width *= 4
    matrix = matrix * (1.0 / math.sqrt(group_size))
    grouped = value.reshape(-1, value.shape[-1] // group_size, group_size)
    return torch.matmul(grouped, matrix).reshape_as(value)


class ConvRotInt8Linear(nn.Module):
    """W8A8-compatible linear with an exact, slow CPU/reference fallback.

    Stored ConvRot weights are already rotated along the input dimension.
    Therefore the online activation rotation is mandatory. Omitting it produces
    numerically invalid H3 output, not merely a lower-performance fallback.
    """

    def __init__(
        self,
        qweight: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        spec: ComfyQuantSpec,
        kernel: Int8Kernel | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        spec.validate_supported()
        if qweight.ndim != 2 or qweight.dtype != torch.int8:
            raise ValueError("qweight must be a rank-two int8 tensor")
        if scale.numel() not in (1, qweight.shape[0]):
            raise ValueError("INT8 scale must be scalar or one value per output row")
        if bias is not None and bias.numel() != qweight.shape[0]:
            raise ValueError("bias does not match qweight output rows")
        if spec.convrot and qweight.shape[1] % spec.convrot_groupsize:
            raise ValueError("qweight input width is not divisible by ConvRot group size")
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias)
        self.spec = spec
        self.kernel = kernel
        self.output_dtype = output_dtype

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, torch.Tensor],
        prefix: str,
        *,
        kernel: Int8Kernel | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> "ConvRotInt8Linear":
        quant_key = f"{prefix}.comfy_quant"
        try:
            spec = ComfyQuantSpec.decode(state[quant_key])
            qweight = state[f"{prefix}.weight"]
            scale = state[f"{prefix}.weight_scale"]
        except KeyError as error:
            raise KeyError(f"incomplete quantized layer {prefix!r}: missing {error.args[0]!r}") from error
        return cls(
            qweight,
            scale,
            state.get(f"{prefix}.bias"),
            spec=spec,
            kernel=kernel,
            output_dtype=output_dtype,
        )

    @property
    def in_features(self) -> int:
        return int(self.qweight.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.qweight.shape[0])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.in_features:
            raise ValueError("input width does not match quantized weight")
        if self.kernel is not None and value.is_cuda:
            return self.kernel(
                value,
                self.qweight,
                self.scale,
                self.bias,
                self.spec.convrot_groupsize if self.spec.convrot else 0,
            )
        rotated = (
            groupwise_hadamard(value.float(), self.spec.convrot_groupsize)
            if self.spec.convrot
            else value.float()
        )
        scale = self.scale.float().reshape(-1, 1)
        weight = self.qweight.float() * scale
        bias = None if self.bias is None else self.bias.float()
        result = F.linear(rotated, weight, bias)
        return result.to(self.output_dtype or value.dtype)


def inspect_quantized_layers(state: Mapping[str, torch.Tensor]) -> dict[str, ComfyQuantSpec]:
    suffix = ".comfy_quant"
    return {
        key[: -len(suffix)]: ComfyQuantSpec.decode(value)
        for key, value in state.items()
        if key.endswith(suffix)
    }
