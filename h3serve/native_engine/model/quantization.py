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
W4A8Kernel = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        int,
        int,
        torch.dtype,
    ],
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


def comfy_kitchen_w4a8_kernel(
    value: torch.Tensor,
    qdata: torch.Tensor,
    s_rel: torch.Tensor,
    s_channel: torch.Tensor,
    codebook: torch.Tensor | None,
    correction: torch.Tensor | None,
    bias: torch.Tensor | None,
    group_size: int,
    convrot_group_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Lazy release-owned Comfy-Kitchen W4A8 binding."""

    try:
        from comfy_kitchen import w4a8_int8_linear
    except ImportError as error:
        raise RuntimeError(
            "the release-owned comfy-kitchen W4A8 runtime is required"
        ) from error
    return w4a8_int8_linear(
        value,
        qdata,
        s_rel,
        s_channel,
        codebook=codebook,
        correction=correction,
        bias=bias,
        group_size=group_size,
        convrot_groupsize=convrot_group_size,
        out_dtype=output_dtype,
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


@dataclass(frozen=True, slots=True)
class PreparedConvRotInt8Input:
    """One exact ConvRot/row-INT8 activation shared by output-row GEMMs."""

    qdata: torch.Tensor
    scale: torch.Tensor
    rows: int
    width: int
    dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class W4A8QuantSpec:
    """Physical layout recorded by the MiniMax-H3 mixed W4A8 checkpoints."""

    format: str = "asym_w4a8_int8"
    group_size: int = 16
    convrot: bool = True
    convrot_groupsize: int = 256

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "W4A8QuantSpec":
        return cls(
            format=str(data.get("format", "")),
            group_size=int(data.get("group_size", 16)),
            convrot=bool(data.get("convrot", True)),
            convrot_groupsize=int(data.get("convrot_groupsize", 256)),
        )

    def validate_supported(self) -> None:
        if self.format != "asym_w4a8_int8":
            raise ValueError(
                "native core currently supports asym_w4a8_int8, "
                f"got {self.format!r}"
            )
        if self.group_size < 4 or (
            16 % self.group_size != 0 and self.group_size % 16 != 0
        ):
            raise ValueError(
                "W4A8 group size must be >=4 and divide 16 or be a multiple of 16"
            )
        if not self.convrot:
            raise ValueError("the H3 W4A8 checkpoint requires ConvRot")
        size = self.convrot_groupsize
        if size < 4 or size & (size - 1) or size.bit_length() % 2 == 0:
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

    def _forward_parameters(
        self,
        value: torch.Tensor,
        qweight: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if value.shape[-1] != self.in_features:
            raise ValueError("input width does not match quantized weight")
        if self.kernel is not None and value.is_cuda:
            return self.kernel(
                value,
                qweight,
                scale,
                bias,
                self.spec.convrot_groupsize if self.spec.convrot else 0,
            )
        rotated = (
            groupwise_hadamard(value.float(), self.spec.convrot_groupsize)
            if self.spec.convrot
            else value.float()
        )
        scale = scale.float().reshape(-1, 1)
        weight = qweight.float() * scale
        bias = None if bias is None else bias.float()
        result = F.linear(rotated, weight, bias)
        return result.to(self.output_dtype or value.dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self._forward_parameters(
            value, self.qweight, self.scale, self.bias
        )

    def forward_output_slice(
        self, value: torch.Tensor, start: int, stop: int
    ) -> torch.Tensor:
        """Project a contiguous output-row range without changing its math.

        ConvRot transforms only the input dimension, and tensorwise INT8
        scales are either scalar or independent per output row.  A contiguous
        Q/K/V range can therefore use the same production kernel with a row
        view of the stored checkpoint tensors.  This is an execution view,
        not a new or modified model weight.
        """

        start, stop = int(start), int(stop)
        if not 0 <= start < stop <= self.out_features:
            raise ValueError("quantized output slice lies outside the layer")
        scale = (
            self.scale
            if self.scale.numel() == 1
            else self.scale[start:stop]
        )
        bias = None if self.bias is None else self.bias[start:stop]
        return self._forward_parameters(
            value,
            self.qweight[start:stop],
            scale,
            bias,
        )

    def prepare_output_slices(
        self, value: torch.Tensor
    ) -> PreparedConvRotInt8Input:
        """Quantize one 2-D CUDA activation once for later output slices.

        The long H3 Attention executor consumes K/V first and Query later, but
        all three checkpoint rows see the identical normalized hidden state.
        ConvRot and dynamic row scales depend only on those input rows, so this
        physical cache removes repeated work without sharing an approximation
        or changing the quantized values seen by either GEMM.
        """

        if self.kernel is not comfy_kitchen_int8_kernel:
            raise RuntimeError("prepared output slices require the audited CUDA INT8 kernel")
        if not value.is_cuda or value.ndim != 2:
            raise RuntimeError("prepared output slices require a rank-two CUDA activation")
        if not self.spec.convrot or self.spec.convrot_groupsize != 256:
            raise RuntimeError("prepared output slices require ConvRot group size 256")
        if value.shape[1] != self.in_features:
            raise ValueError("input width does not match quantized weight")
        if not 256 <= self.in_features <= 16_384 or self.in_features % 256:
            raise RuntimeError("prepared output slice width is outside the fused CUDA contract")
        from comfy_kitchen.backends.cuda import quantize_int8_rowwise_convrot64

        contiguous = value if value.is_contiguous() else value.contiguous()
        qdata, scale = quantize_int8_rowwise_convrot64(contiguous, 256)
        return PreparedConvRotInt8Input(
            qdata=qdata,
            scale=scale,
            rows=int(value.shape[0]),
            width=int(value.shape[1]),
            dtype=value.dtype,
        )

    def forward_prepared_output_slice(
        self,
        value: torch.Tensor,
        prepared: PreparedConvRotInt8Input,
        row_start: int,
        row_stop: int,
        output_start: int,
        output_stop: int,
    ) -> torch.Tensor:
        """Project one row/output rectangle from a shared quantized input."""

        row_start, row_stop = int(row_start), int(row_stop)
        output_start, output_stop = int(output_start), int(output_stop)
        if prepared.width != self.in_features:
            raise ValueError("prepared activation width does not match quantized weight")
        if value.ndim != 2 or value.shape != (prepared.rows, prepared.width):
            raise ValueError("source activation does not match its prepared INT8 input")
        if not 0 <= row_start < row_stop <= prepared.rows:
            raise ValueError("prepared activation row slice lies outside the input")
        if not 0 <= output_start < output_stop <= self.out_features:
            raise ValueError("quantized output slice lies outside the layer")
        scale = (
            self.scale
            if self.scale.numel() == 1
            else self.scale[output_start:output_stop]
        )
        bias = (
            None
            if self.bias is None
            else self.bias[output_start:output_stop]
        )
        from comfy_kitchen.backends.cuda import int8_linear_prequantized

        return int8_linear_prequantized(
            prepared.qdata[row_start:row_stop],
            prepared.scale[row_start:row_stop],
            self.qweight[output_start:output_stop],
            scale,
            bias,
            self.output_dtype or prepared.dtype,
        )


class ConvRotW4A8Linear(nn.Module):
    """Grouped packed-W4 weights with W8 activations and mandatory ConvRot.

    The checkpoint stores two four-bit codes in each signed byte.  Group-local
    FP8 scales and a channel FP32 scale reconstruct the INT8 grid consumed by
    the production CUDA GEMM.  LoRA remains a separate activation-space delta,
    so this module never mutates or merges the packed base weight.
    """

    def __init__(
        self,
        qdata: torch.Tensor,
        s_rel: torch.Tensor,
        s_channel: torch.Tensor,
        *,
        codebook: torch.Tensor | None = None,
        correction: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        spec: W4A8QuantSpec = W4A8QuantSpec(),
        kernel: W4A8Kernel | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        spec.validate_supported()
        if qdata.ndim != 2 or qdata.dtype != torch.int8:
            raise ValueError("packed W4 weight must be a rank-two int8 tensor")
        out_features, packed_width = qdata.shape
        in_features = packed_width * 2
        if in_features % 16 or in_features % spec.group_size:
            raise ValueError("W4A8 input width is incompatible with its group size")
        if in_features % spec.convrot_groupsize:
            raise ValueError("W4A8 input width is incompatible with ConvRot")
        groups = in_features // spec.group_size
        if tuple(s_rel.shape) != (out_features, groups):
            raise ValueError(
                f"s_rel must be {(out_features, groups)}, got {tuple(s_rel.shape)}"
            )
        if tuple(s_channel.shape) != (out_features,):
            raise ValueError(
                f"s_channel must be {(out_features,)}, got {tuple(s_channel.shape)}"
            )
        if codebook is not None and tuple(codebook.shape) != (16,):
            raise ValueError("W4A8 codebook must contain 16 entries")
        if correction is not None and tuple(correction.shape) != (
            groups,
            out_features,
        ):
            raise ValueError(
                "W4A8 correction must be [groups, out_features]"
            )
        if bias is not None and bias.numel() != out_features:
            raise ValueError("bias does not match W4A8 output rows")
        self.register_buffer("qdata", qdata)
        self.register_buffer("s_rel", s_rel)
        self.register_buffer("s_channel", s_channel)
        self.register_buffer("codebook", codebook)
        self.register_buffer("correction", correction)
        self.register_buffer("bias", bias)
        self.spec = spec
        self.kernel = kernel
        self.output_dtype = output_dtype

    @property
    def in_features(self) -> int:
        return int(self.qdata.shape[1]) * 2

    @property
    def out_features(self) -> int:
        return int(self.qdata.shape[0])

    def _reference_weight(
        self,
        qdata: torch.Tensor,
        s_rel: torch.Tensor,
        s_channel: torch.Tensor,
        codebook: torch.Tensor | None,
        correction: torch.Tensor | None,
    ) -> torch.Tensor:
        """Decode the physical rotated weight for CPU tests and diagnostics."""

        packed = qdata.to(torch.int32) & 0xFF
        codes = torch.empty(
            qdata.shape[0], qdata.shape[1] * 2,
            device=qdata.device, dtype=torch.int64,
        )
        codes[:, 0::2] = packed & 0xF
        codes[:, 1::2] = (packed >> 4) & 0xF
        values = (
            codes.float() - 8.0
            if codebook is None
            else codebook.float()[codes]
        )
        groups = self.in_features // self.spec.group_size
        int8_grid = (
            values.view(qdata.shape[0], groups, self.spec.group_size)
            * s_rel.float().unsqueeze(-1)
        ).round().clamp_(-127, 127)
        weight = int8_grid * s_channel.float().view(-1, 1, 1)
        if correction is not None:
            weight = weight + correction.float().T.unsqueeze(-1)
        return weight.reshape(qdata.shape[0], self.in_features)

    def _forward_parameters(
        self,
        value: torch.Tensor,
        qdata: torch.Tensor,
        s_rel: torch.Tensor,
        s_channel: torch.Tensor,
        codebook: torch.Tensor | None,
        correction: torch.Tensor | None,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if value.shape[-1] != self.in_features:
            raise ValueError("input width does not match packed W4A8 weight")
        output_dtype = self.output_dtype or value.dtype
        if self.kernel is not None and value.is_cuda:
            return self.kernel(
                value,
                qdata,
                s_rel,
                s_channel,
                codebook,
                correction,
                bias,
                self.spec.group_size,
                self.spec.convrot_groupsize,
                output_dtype,
            )
        rotated = groupwise_hadamard(
            value.float(), self.spec.convrot_groupsize
        )
        weight = self._reference_weight(
            qdata, s_rel, s_channel, codebook, correction
        )
        result = F.linear(
            rotated,
            weight.float(),
            None if bias is None else bias.float(),
        )
        return result.to(output_dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self._forward_parameters(
            value,
            self.qdata,
            self.s_rel,
            self.s_channel,
            self.codebook,
            self.correction,
            self.bias,
        )

    def forward_output_slice(
        self, value: torch.Tensor, start: int, stop: int
    ) -> torch.Tensor:
        start, stop = int(start), int(stop)
        if not 0 <= start < stop <= self.out_features:
            raise ValueError("W4A8 output slice lies outside the layer")
        return self._forward_parameters(
            value,
            self.qdata[start:stop],
            self.s_rel[start:stop],
            self.s_channel[start:stop],
            self.codebook,
            (
                None
                if self.correction is None
                else self.correction[:, start:stop]
            ),
            None if self.bias is None else self.bias[start:stop],
        )


def inspect_quantized_layers(state: Mapping[str, torch.Tensor]) -> dict[str, ComfyQuantSpec]:
    suffix = ".comfy_quant"
    return {
        key[: -len(suffix)]: ComfyQuantSpec.decode(value)
        for key, value in state.items()
        if key.endswith(suffix)
    }
