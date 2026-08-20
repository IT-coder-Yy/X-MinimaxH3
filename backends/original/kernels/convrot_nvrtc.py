"""Driver-compatible SM89 SwiGLU + ConvRot + row-INT8 quantizer.

The installed comfy-kitchen CUDA wheel was compiled for a CUDA runtime newer
than the host driver.  This module deliberately compiles only the preprocessing
kernel with PyTorch's CUDA 12.6 NVRTC and loads the PTX through the current CUDA
driver.  It does not replace comfy-kitchen or modify the shared environment.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import torch


_CUDA_SOURCE = r"""
#include <cuda_bf16.h>
#include <cuda_fp16.h>

template <typename T> __device__ __forceinline__ float as_float(T value);
template <> __device__ __forceinline__ float as_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}
template <> __device__ __forceinline__ float as_float<__half>(__half value) {
    return __half2float(value);
}

template <typename T> __device__ __forceinline__ T from_float(float value);
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}
template <> __device__ __forceinline__ __half from_float<__half>(float value) {
    return __float2half_rn(value);
}

template <typename T> __device__ __forceinline__ float dtype_round(float value) {
    return as_float<T>(from_float<T>(value));
}

__device__ __forceinline__ float warp_max(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

template <int NUM_WARPS>
__device__ __forceinline__ float block_max(float value, float* warp_values, float* result) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_max(value);
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();
    if (warp == 0) {
        float total = lane < NUM_WARPS ? warp_values[lane] : 0.0f;
        total = warp_max(total);
        if (lane == 0) *result = total;
    }
    __syncthreads();
    return *result;
}

template <int STRIDE>
__device__ __forceinline__ void h4_stage(const float* src, float* dst, int lane) {
    const int base = (lane % STRIDE) + (lane / STRIDE) * (4 * STRIDE);
    const float x0 = src[base];
    const float x1 = src[base + STRIDE];
    const float x2 = src[base + 2 * STRIDE];
    const float x3 = src[base + 3 * STRIDE];
    dst[base] = 0.5f * ( x0 + x1 + x2 - x3);
    dst[base + STRIDE] = 0.5f * ( x0 + x1 - x2 + x3);
    dst[base + 2 * STRIDE] = 0.5f * ( x0 - x1 + x2 + x3);
    dst[base + 3 * STRIDE] = 0.5f * (-x0 + x1 + x2 + x3);
}

template <typename T, int BLOCK_THREADS, bool SWIGLU>
__device__ void convrot_quantize(
    const T* __restrict__ x,
    signed char* __restrict__ q,
    float* __restrict__ scales,
    int K)
{
    constexpr int GROUP_THREADS = 64;
    constexpr int GROUP = 256;
    constexpr int GROUPS_IN_FLIGHT = BLOCK_THREADS / GROUP_THREADS;
    constexpr int WARPS = BLOCK_THREADS / 32;

    extern __shared__ float smem[];
    float* row_buf = smem;
    float* scratch = smem + K;
    __shared__ float warp_values[WARPS];
    __shared__ float row_max;

    const int row = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const int sub = tid / GROUP_THREADS;
    const int lane = tid % GROUP_THREADS;
    const int groups = K / GROUP;
    const long long input_row = static_cast<long long>(row) * (SWIGLU ? 2 * K : K);
    const long long output_row = static_cast<long long>(row) * K;
    float* buf0 = scratch + sub * (2 * GROUP);
    float* buf1 = buf0 + GROUP;
    float local_max = 0.0f;

    const int iterations = (groups + GROUPS_IN_FLIGHT - 1) / GROUPS_IN_FLIGHT;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const int group = iteration * GROUPS_IN_FLIGHT + sub;
        const bool active = group < groups;
        const int base = lane * 4;
        const int column = group * GROUP + base;
        float values[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (active && SWIGLU) {
                const float gate = as_float<T>(x[input_row + column + i]);
                const float up = as_float<T>(x[input_row + K + column + i]);
                // Match eager torch SiLU followed by the in-place multiply:
                // each operation stores to the activation dtype before the next.
                const float silu = dtype_round<T>(gate / (1.0f + expf(-gate)));
                values[i] = dtype_round<T>(silu * up);
            } else if (active) {
                values[i] = as_float<T>(x[input_row + column + i]);
            } else {
                values[i] = 0.0f;
            }
        }
        buf1[base] = 0.5f * ( values[0] + values[1] + values[2] - values[3]);
        buf1[base + 1] = 0.5f * ( values[0] + values[1] - values[2] + values[3]);
        buf1[base + 2] = 0.5f * ( values[0] - values[1] + values[2] + values[3]);
        buf1[base + 3] = 0.5f * (-values[0] + values[1] + values[2] + values[3]);
        __syncthreads();

        h4_stage<4>(buf1, buf0, lane);
        __syncthreads();
        h4_stage<16>(buf0, buf1, lane);
        __syncthreads();

        if (active) {
            const int fbase = (lane % 64) + (lane / 64) * 256;
            const float x0 = buf1[fbase];
            const float x1 = buf1[fbase + 64];
            const float x2 = buf1[fbase + 128];
            const float x3 = buf1[fbase + 192];
            // The reference batched matmul stores the rotated activation in T.
            const float y0 = dtype_round<T>(0.5f * ( x0 + x1 + x2 - x3));
            const float y1 = dtype_round<T>(0.5f * ( x0 + x1 - x2 + x3));
            const float y2 = dtype_round<T>(0.5f * ( x0 - x1 + x2 + x3));
            const float y3 = dtype_round<T>(0.5f * (-x0 + x1 + x2 + x3));
            const int out = group * GROUP + fbase;
            row_buf[out] = y0;
            row_buf[out + 64] = y1;
            row_buf[out + 128] = y2;
            row_buf[out + 192] = y3;
            local_max = fmaxf(local_max, fmaxf(fmaxf(fabsf(y0), fabsf(y1)),
                                                fmaxf(fabsf(y2), fabsf(y3))));
        }
        __syncthreads();
    }

    const float maximum = block_max<WARPS>(local_max, warp_values, &row_max);
    const float scale = fmaxf(maximum * (1.0f / 127.0f), 1.0e-30f);
    if (tid == 0) scales[row] = scale;
    const float rounded_scale = dtype_round<T>(scale);
    for (int col = tid; col < K; col += BLOCK_THREADS) {
        const float normalized = dtype_round<T>(row_buf[col] / rounded_scale);
        float quantized = nearbyintf(normalized);
        quantized = fminf(127.0f, fmaxf(-128.0f, quantized));
        q[output_row + col] = static_cast<signed char>(quantized);
    }
}

extern "C" __global__ void swiglu_convrot_q_bf16(
    const __nv_bfloat16* x, signed char* q, float* scales, int K) {
    convrot_quantize<__nv_bfloat16, 1024, true>(x, q, scales, K);
}

extern "C" __global__ void swiglu_convrot_q_fp16(
    const __half* x, signed char* q, float* scales, int K) {
    convrot_quantize<__half, 1024, true>(x, q, scales, K);
}

extern "C" __global__ void convrot_q_bf16(
    const __nv_bfloat16* x, signed char* q, float* scales, int K) {
    convrot_quantize<__nv_bfloat16, 1024, false>(x, q, scales, K);
}

extern "C" __global__ void convrot_q_fp16(
    const __half* x, signed char* q, float* scales, int K) {
    convrot_quantize<__half, 1024, false>(x, q, scales, K);
}
"""


_MODULE = None
_KERNELS: dict[tuple[torch.dtype, bool], object] = {}


def _find_cuda12_component(component: str, filename: str) -> Path:
    for entry in map(Path, sys.path):
        candidate = entry / "nvidia" / component / filename
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"CUDA 12.6 component not found: nvidia/{component}/{filename}")


def _compile_ptx() -> bytes:
    nvrtc_path = _find_cuda12_component("cuda_nvrtc", "lib/libnvrtc.so.12")
    include_dir = _find_cuda12_component("cuda_runtime", "include/cuda_runtime.h").parent
    nvrtc = ctypes.CDLL(str(nvrtc_path))
    nvrtc.nvrtcGetErrorString.restype = ctypes.c_char_p

    def check(code: int) -> None:
        if code:
            raise RuntimeError(nvrtc.nvrtcGetErrorString(code).decode("utf-8"))

    program = ctypes.c_void_p()
    source = _CUDA_SOURCE.encode("utf-8")
    check(nvrtc.nvrtcCreateProgram(
        ctypes.byref(program), source, b"convrot.cu", 0, None, None
    ))
    options = [
        b"--gpu-architecture=compute_89",
        b"--std=c++17",
        b"--use_fast_math",
        f"-I{include_dir}".encode("utf-8"),
    ]
    option_array = (ctypes.c_char_p * len(options))(*options)
    result = nvrtc.nvrtcCompileProgram(program, len(options), option_array)
    if result:
        size = ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
        log = ctypes.create_string_buffer(size.value)
        nvrtc.nvrtcGetProgramLog(program, log)
        raise RuntimeError("ConvRot NVRTC compilation failed:\n" + log.value.decode("utf-8"))
    size = ctypes.c_size_t()
    check(nvrtc.nvrtcGetPTXSize(program, ctypes.byref(size)))
    ptx = ctypes.create_string_buffer(size.value)
    check(nvrtc.nvrtcGetPTX(program, ptx))
    check(nvrtc.nvrtcDestroyProgram(ctypes.byref(program)))
    return ptx.raw


def _load() -> None:
    global _MODULE, _KERNELS
    if _MODULE is not None:
        return
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError("ConvRot kernel requires an SM89 CUDA device")
    from torch.cuda._utils import _cuda_load_module, _get_cuda_library

    _MODULE = _cuda_load_module(_compile_ptx())
    _KERNELS = {
        (torch.bfloat16, True): _MODULE.swiglu_convrot_q_bf16,
        (torch.float16, True): _MODULE.swiglu_convrot_q_fp16,
        (torch.bfloat16, False): _MODULE.convrot_q_bf16,
        (torch.float16, False): _MODULE.convrot_q_fp16,
    }
    driver = _get_cuda_library()
    driver.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8.
    for kernel in _KERNELS.values():
        result = driver.cuFuncSetAttribute(kernel.func, 8, 96 * 1024)
        if result != 0:
            raise RuntimeError(f"cuFuncSetAttribute failed with CUDA error {result}")


def warmup_module() -> None:
    """Compile and load the isolated module without launching a workload."""
    _load()


def _fused_convrot_row_quant(
    x: torch.Tensor, *, swiglu: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    width_factor = 2 if swiglu else 1
    if x.ndim != 2 or x.shape[1] % (256 * width_factor):
        label = "[M, 2*K]" if swiglu else "[M, K]"
        raise ValueError(f"expected {label} with K divisible by 256, got {tuple(x.shape)}")
    if x.dtype not in (torch.bfloat16, torch.float16) or not x.is_cuda:
        raise ValueError("ConvRot supports CUDA BF16/FP16 inputs only")
    if not x.is_contiguous():
        x = x.contiguous()
    _load()
    rows, raw_cols = x.shape
    cols = raw_cols // width_factor
    if cols > 14336:
        raise ValueError(f"K={cols} exceeds the validated shared-memory contract")
    q = torch.empty((rows, cols), dtype=torch.int8, device=x.device)
    scales = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    # 1024 threads use 16 concurrent 256-value groups: K + 16*2*256 floats.
    shared_mem = (cols + 8192) * 4
    _KERNELS[(x.dtype, swiglu)](
        grid=(rows, 1, 1),
        block=(1024, 1, 1),
        args=[x, q, scales, cols],
        shared_mem=shared_mem,
    )
    return q, scales


def fused_swiglu_convrot_row_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a contiguous ``[M, 2*K]`` paired SwiGLU activation."""
    return _fused_convrot_row_quant(x, swiglu=True)


def fused_convrot_row_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a contiguous ``[M, K]`` activation after groupwise ConvRot."""
    return _fused_convrot_row_quant(x, swiglu=False)
