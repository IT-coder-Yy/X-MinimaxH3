"""Exact single-sided partial QK-Norm/RoPE for the measured SM89 runtime.

Comfy Kitchen's public single-tensor wrapper currently exposes only full-width
RoPE, while its CUDA shared object already exports the common launcher with a
``has_k`` switch and an explicit ``rot_dim``.  The long H3 split-QKV path owns
only Q *or* K at a time; calling the paired public wrapper therefore allocates
and transforms a same-sized dummy tensor.

This adapter calls that already-loaded launcher with ``has_k=False``.  It adds
no new numerical implementation: the exact same compiled kernel template,
reduction order, BF16 rounding and current CUDA stream are used.  The internal
ABI is pinned and every unsupported layout fails closed to the established
paired path in :mod:`h3serve.native_engine.model.layers`.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import torch


# The interactive environment currently imports 0.2.26 directly, while the
# reproducible Linux runtime mirror resolves its pinned wheel as 0.2.28.  Both
# expose the same launcher ABI and are covered by the physical CUDA test.
_SUPPORTED_COMFY_KITCHEN_VERSIONS = frozenset({"0.2.26", "0.2.28"})
_DTYPE_CODES = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}


@lru_cache(maxsize=1)
def _load_single_launcher():
    try:
        installed = version("comfy-kitchen")
    except PackageNotFoundError:
        return None
    if installed not in _SUPPORTED_COMFY_KITCHEN_VERSIONS:
        return None
    try:
        from comfy_kitchen.backends.cuda import _C

        library = ctypes.CDLL(_C.__file__)
        launcher = library.launch_rms_rope_kernel
    except (AttributeError, ImportError, OSError):
        return None

    pointer = ctypes.c_void_p
    integer = ctypes.c_int64
    # Seven pointers; 32 int64 shape/stride values; epsilon; three dtype
    # codes; has_k/split_half; current cudaStream_t.
    launcher.argtypes = (
        [pointer] * 7
        + [integer] * 32
        + [ctypes.c_float]
        + [ctypes.c_int] * 3
        + [ctypes.c_bool] * 2
        + [pointer]
    )
    launcher.restype = None
    # Keep the CDLL alive for the complete process lifetime.
    return library, launcher


def _try_apply_single_qknorm_rope_out(
    value: torch.Tensor,
    output: torch.Tensor,
    *,
    weight: torch.Tensor,
    frequencies: torch.Tensor,
    eps: float,
) -> bool:
    """Apply the exact single-side kernel into an arbitrary logical NHD view.

    The accepted H3 layout is ``value=[tokens, heads, head_dim]`` and
    ``frequencies=[1|B, 1|tokens, 1|heads, rot_dim/2, 2, 2]``.  No copy,
    reshape allocation or synchronization is introduced.  ``output`` has the
    same logical NHD shape but may carry HND-backed strides; the pinned CUDA
    launcher already owns distinct input/output strides, allowing QK-Norm and
    RoPE to land directly in Attention's physical layout.
    """

    # CPU/unsupported callers must fail closed before importing any optional
    # process-global Comfy-Kitchen installation. The SM89 runtime policy owns
    # selection and hash validation of that module for real CUDA execution.
    if not value.is_cuda or not output.is_cuda:
        return False
    loaded = _load_single_launcher()
    if loaded is None:
        return False
    if (
        value.ndim != 3
        or output.ndim != 3
        or frequencies.ndim != 6
        or weight.ndim != 1
    ):
        return False
    if value.dtype not in (torch.float16, torch.bfloat16):
        return False
    if frequencies.dtype not in _DTYPE_CODES or weight.dtype not in _DTYPE_CODES:
        return False
    if (
        output.shape != value.shape
        or output.dtype != value.dtype
        or output.device != value.device
        or value.device != frequencies.device
        or value.device != weight.device
    ):
        return False
    tokens, heads, head_dim = (int(size) for size in value.shape)
    rotate_width = int(frequencies.shape[-3]) * 2
    if (
        tokens <= 0
        or heads <= 0
        or head_dim < 32
        or head_dim % 32
        or rotate_width <= 0
        or rotate_width > head_dim
        or rotate_width % 4
        or frequencies.shape[-2:] != (2, 2)
        or int(weight.shape[0]) != head_dim
        or int(weight.stride(0)) != 1
        or int(value.stride(-1)) != 1
        or int(output.stride(-1)) != 1
    ):
        return False
    frequency_prefix = tuple(int(size) for size in frequencies.shape[:3])
    if (
        frequency_prefix[0] != 1
        or frequency_prefix[1] not in (1, tokens)
        or frequency_prefix[2] not in (1, heads)
    ):
        return False
    try:
        if torch.cuda.is_current_stream_capturing():
            return False
    except RuntimeError:
        return False

    _library, launcher = loaded
    query = value.unsqueeze(0)
    output_query = output.unsqueeze(0)
    query_strides = tuple(int(stride) for stride in query.stride())
    output_query_strides = tuple(
        int(stride) for stride in output_query.stride()
    )
    frequency_strides = tuple(int(stride) for stride in frequencies.stride())
    null_strides = (0, 0, 0, 0)
    pointer = ctypes.c_void_p

    def address(tensor: torch.Tensor | None) -> ctypes.c_void_p:
        return pointer(0 if tensor is None else int(tensor.data_ptr()))

    arguments = (
        address(query),
        address(None),
        address(frequencies),
        address(weight),
        address(None),
        address(output_query),
        address(None),
        *tuple(int(size) for size in query.shape[:3]),
        head_dim,
        rotate_width,
        *frequency_prefix,
        *query_strides,
        *null_strides,
        *output_query_strides,
        *null_strides,
        *frequency_strides,
        int(weight.stride(0)),
        0,
        float(eps),
        _DTYPE_CODES[value.dtype],
        _DTYPE_CODES[frequencies.dtype],
        _DTYPE_CODES[weight.dtype],
        False,
        True,
        pointer(int(torch.cuda.current_stream(value.device).cuda_stream)),
    )
    if len(arguments) != 46:
        return False
    launcher(*arguments)
    return True


def try_apply_single_qknorm_rope_(
    value: torch.Tensor,
    *,
    weight: torch.Tensor,
    frequencies: torch.Tensor,
    eps: float,
) -> bool:
    """Apply the established launcher in place, or return ``False``."""

    return _try_apply_single_qknorm_rope_out(
        value,
        value,
        weight=weight,
        frequencies=frequencies,
        eps=eps,
    )


def try_apply_single_qknorm_rope_to_hnd(
    value: torch.Tensor,
    output_nhd_view: torch.Tensor,
    *,
    weight: torch.Tensor,
    frequencies: torch.Tensor,
    eps: float,
) -> bool:
    """Write logical NHD input directly into an HND-backed NHD view."""

    if output_nhd_view.is_contiguous():
        return False
    tokens, heads, head_dim = value.shape
    strides = tuple(int(item) for item in output_nhd_view.stride())
    if (
        strides[0] != head_dim
        or strides[1] < tokens * head_dim
        or strides[1] % head_dim
        or strides[2] != 1
    ):
        return False
    return _try_apply_single_qknorm_rope_out(
        value,
        output_nhd_view,
        weight=weight,
        frequencies=frequencies,
        eps=eps,
    )


__all__ = [
    "try_apply_single_qknorm_rope_",
    "try_apply_single_qknorm_rope_to_hnd",
]
