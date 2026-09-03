#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SageAttention / GPU 分层 A/B 诊断脚本（中文报告）。

默认阶段：
  copy -> memory_pattern -> int32_alu -> fp32_gemm -> tf32_gemm
  -> fp16_gemm -> bf16_gemm -> sdpa_math -> sdpa_flash
  -> sage_fp16 -> sage_fp8

关键点：
- CPU 固定输入 + CPU float32 reference
- 每个 GPU/stage/run 都是全新的 Python 子进程
- 父进程负责物理 GPU 映射，子进程只使用 cuda:0
- CUDA_LAUNCH_BLOCKING=1
- 默认只读采集温度、功耗、频率、利用率和限频原因；绝不修改 GPU 设置
- memory_pattern 按物理 GPU 只运行一次，覆盖尽可能多的空闲显存
- 可用 --compute-sanitizer 对指定 kernel 阶段执行 memcheck
- 输出中文 Markdown 诊断章节；默认更新已有 sage_gpu_diagnosis.md，并保留第一轮原始内容

示例：
  python diagnose_sage_gpu_cn.py --control-gpu 1 --target-gpu 4 --runs 3
  python diagnose_sage_gpu_cn.py --target-only --target-gpu 4 --runs 3
  python diagnose_sage_gpu_cn.py --control-gpu 1 --target-gpu 4 --runs 3 --size-sweep
  python diagnose_sage_gpu_cn.py --control-gpu 1 --target-gpu 4 --runs 3 \
    --compute-sanitizer --timeout 900
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

DEVICE_CONFIG = "device_wide"
DEVICE_STAGES = ("memory_pattern",)
GEMM_STAGES = ("fp32_gemm", "tf32_gemm", "fp16_gemm", "bf16_gemm")
STAGES = (
    "copy",
    "memory_pattern",
    "int32_alu",
    "fp32_gemm",
    "tf32_gemm",
    "fp16_gemm",
    "bf16_gemm",
    "sdpa_math",
    "sdpa_flash",
    "sage_fp16",
    "sage_fp8",
)
STAGE_CN = {
    "copy": "CPU↔GPU 拷贝完整性",
    "memory_pattern": "大范围显存模式校验",
    "int32_alu": "INT32 逐元 ALU",
    "fp32_gemm": "FP32 GEMM（关闭 TF32）",
    "tf32_gemm": "TF32 GEMM",
    "fp16_gemm": "FP16 GEMM",
    "bf16_gemm": "BF16 GEMM",
    "sdpa_math": "SDPA Math",
    "sdpa_flash": "SDPA Flash/Fused",
    "sage_fp16": "SageAttention FP16",
    "sage_fp8": "SageAttention FP8",
}

REPORT_NAME = "sage_gpu_diagnosis.md"
UPDATE_BEGIN = "<!-- SAGE_GPU_CN_V2_BEGIN -->"
UPDATE_END = "<!-- SAGE_GPU_CN_V2_END -->"
DEFAULT_SEED = 42
DEFAULT_GEMM_SIZE = 2048
DEFAULT_TIMEOUT = 900
DEFAULT_MEMORY_FRACTION = 0.80
DEFAULT_MEMORY_RESERVE_MIB = 1024
DEFAULT_MEMORY_CHUNK_MIB = 32
DEFAULT_MONITOR_INTERVAL = 0.5
DEFAULT_OBSERVATION_SECONDS = 2.0
DEFAULT_SPATIAL_SAMPLES = 8
DEFAULT_SPATIAL_TILE = 16

# A/B 诊断阈值：故意比上一版更严格，尤其是 SDPA。
FP32_MAE_LIMIT = 0.005
FP32_MAX_LIMIT = 0.10
TF32_MAE_LIMIT = 0.10
TF32_MAX_LIMIT = 2.0
FP16_MAE_LIMIT = 0.10
FP16_MAX_LIMIT = 2.0
GEMM_MAE_LIMIT = 1.0
GEMM_MAX_LIMIT = 20.0
SDPA_MAE_LIMIT = 0.005
SDPA_MAX_LIMIT = 0.05
SAGE_MAE_LIMIT = 0.01
SAGE_MAX_LIMIT = 0.10


class BackendUnsupported(RuntimeError):
    pass


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def print_json(obj: dict[str, Any]) -> None:
    print(json.dumps(obj, ensure_ascii=False, allow_nan=False), flush=True)


def nvidia_info(gpu: int) -> str:
    try:
        p = subprocess.run([
            "nvidia-smi", "-i", str(gpu),
            "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,temperature.gpu,power.draw,pstate",
            "--format=csv,noheader"
        ], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15, check=False)
        return p.stdout.strip()
    except Exception as exc:
        return f"nvidia-smi 查询失败: {exc!r}"


def kernel_log() -> str:
    for cmd in (["dmesg", "-T"], ["journalctl", "-k", "-n", "500", "--no-pager"]):
        try:
            p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               timeout=15, check=False)
            if p.returncode != 0:
                continue
            lines = [x for x in p.stdout.splitlines()
                     if "NVRM" in x or "Xid" in x or "nvidia" in x.lower()]
            return "\n".join(lines[-80:]) if lines else "(未发现 NVIDIA/Xid 相关内核日志)"
        except Exception:
            pass
    return "(无法读取内核日志，可能是权限不足)"


def cpu_attention_reference(q, k, v):
    import torch
    qf, kf, vf = q.float(), k.float(), v.float()
    scores = torch.matmul(qf, kf.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    return torch.matmul(torch.softmax(scores, dim=-1), vf).contiguous()


def create_bundle(path: Path, batch: int, heads: int, seq: int, dim: int,
                  gemm_size: int, seed: int) -> None:
    import torch
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    shape = (batch, heads, seq, dim)
    q = torch.randn(shape, generator=g).to(torch.bfloat16)
    k = torch.randn(shape, generator=g).to(torch.bfloat16)
    v = torch.randn(shape, generator=g).to(torch.bfloat16)
    attn_ref = cpu_attention_reference(q, k, v)

    a = torch.randn((gemm_size, gemm_size), generator=g).to(torch.bfloat16)
    b = torch.randn((gemm_size, gemm_size), generator=g).to(torch.bfloat16)
    gemm_ref = torch.matmul(a.float(), b.float()).contiguous()

    # 用有界整数表达式单独覆盖普通 ALU/寄存器/数据通路。
    # 中间值不会溢出 int32，CPU/GPU 应该逐位一致。
    int_x = torch.randint(-2048, 2049, (gemm_size, gemm_size), generator=g,
                          dtype=torch.int32)
    int_y = torch.randint(-2048, 2049, (gemm_size, gemm_size), generator=g,
                          dtype=torch.int32)
    int_ref = ((int_x * 17 + int_y * 31) ^ (int_x << 3)).contiguous()

    torch.save({
        "q": q.contiguous(), "k": k.contiguous(), "v": v.contiguous(),
        "attn_ref": attn_ref,
        "a": a.contiguous(), "b": b.contiguous(), "gemm_ref": gemm_ref,
        "int_x": int_x.contiguous(), "int_y": int_y.contiguous(),
        "int_ref": int_ref,
    }, path)


def roundtrip(gpu_tensor, cpu_tensor) -> dict[str, Any]:
    import torch
    back = gpu_tensor.detach().cpu()
    exact = torch.equal(back, cpu_tensor)
    if exact:
        return {"exact": True, "mismatch_count": 0, "max_abs_diff": 0.0}
    diff = (back.float() - cpu_tensor.float()).abs()
    return {
        "exact": False,
        "mismatch_count": int((back != cpu_tensor).sum().item()),
        "max_abs_diff": safe_float(diff.max().item()),
    }


def flat_to_index(flat_index: int, shape: tuple[int, ...]) -> list[int]:
    coords: list[int] = []
    for size in reversed(shape):
        coords.append(flat_index % size)
        flat_index //= size
    return list(reversed(coords))


def scalar_bits(tensor, flat_index: int) -> Optional[str]:
    """返回一个标量的原始位模式，用于判断是数值偏差还是疑似 bit flip。"""
    import torch
    value = tensor.detach().cpu().contiguous().reshape(-1)[flat_index:flat_index + 1]
    views = {
        torch.float16: (torch.int16, 16),
        torch.bfloat16: (torch.int16, 16),
        torch.float32: (torch.int32, 32),
        torch.float64: (torch.int64, 64),
        torch.int32: (torch.int32, 32),
        torch.int64: (torch.int64, 64),
        torch.uint8: (torch.uint8, 8),
    }
    spec = views.get(value.dtype)
    if spec is None:
        return None
    view_dtype, width = spec
    raw = int(value.view(view_dtype).item()) & ((1 << width) - 1)
    return f"0x{raw:0{width // 4}x}"


def spatial_error_analysis(output_cpu, reference_cpu, bad_mask, abs_diff,
                           element_limit: float, sample_limit: int,
                           tile_size: int) -> dict[str, Any]:
    """提取严重错误坐标、位模式和 tile 聚集情况，不保存整张差分图。"""
    import torch
    shape = tuple(int(x) for x in output_cpu.shape)
    flat_bad = bad_mask.reshape(-1)
    bad_count = int(torch.count_nonzero(flat_bad).item())
    result: dict[str, Any] = {
        "shape": list(shape),
        "element_limit": safe_float(element_limit),
        "severe_count": bad_count,
        "severe_rate": safe_float(bad_count / max(1, flat_bad.numel())),
        "first_severe": None,
        "top_errors": [],
        "top_tiles": [],
        "tile_size": tile_size,
    }
    if bad_count == 0:
        return result

    # argmax 避免对大面积错误生成巨大的 nonzero 坐标张量。
    first_flat = int(flat_bad.to(torch.uint8).argmax().item())
    result["first_severe"] = flat_to_index(first_flat, shape)

    ranked = torch.nan_to_num(abs_diff.reshape(-1), nan=math.inf,
                              posinf=math.inf, neginf=math.inf)
    top_count = min(sample_limit, ranked.numel())
    _, top_indices = torch.topk(ranked, k=top_count)
    reference_typed = reference_cpu.to(dtype=output_cpu.dtype).contiguous()
    output_flat = output_cpu.reshape(-1)
    ref_float = reference_cpu.float().reshape(-1)
    diff_flat = abs_diff.reshape(-1)
    samples = []
    for flat in top_indices.tolist():
        if not bool(flat_bad[flat].item()):
            continue
        out_bits = scalar_bits(output_cpu, flat)
        ref_bits = scalar_bits(reference_typed, flat)
        xor_bits = None
        if out_bits and ref_bits:
            xor_bits = hex(int(out_bits, 16) ^ int(ref_bits, 16))
        samples.append({
            "index": flat_to_index(int(flat), shape),
            "output": safe_float(output_flat[flat].float().item()),
            "reference": safe_float(ref_float[flat].item()),
            "abs_error": safe_float(diff_flat[flat].item()),
            "output_bits": out_bits,
            "reference_bits": ref_bits,
            "xor_bits": xor_bits,
        })
    result["top_errors"] = samples

    # 把多维输出压成 [rows, last_dim]，再统计 tile 中严重错误数。
    cols = shape[-1] if shape else 1
    rows = max(1, flat_bad.numel() // cols)
    bad_2d = flat_bad.reshape(rows, cols)
    padded_rows = ((rows + tile_size - 1) // tile_size) * tile_size
    padded_cols = ((cols + tile_size - 1) // tile_size) * tile_size
    padded = torch.zeros((padded_rows, padded_cols), dtype=torch.int32)
    padded[:rows, :cols] = bad_2d
    tile_counts = padded.reshape(
        padded_rows // tile_size, tile_size,
        padded_cols // tile_size, tile_size,
    ).sum(dim=(1, 3))
    nonzero_tiles = int(torch.count_nonzero(tile_counts).item())
    if nonzero_tiles:
        count = min(sample_limit, nonzero_tiles)
        values, indices = torch.topk(tile_counts.reshape(-1), k=count)
        tile_cols = tile_counts.shape[1]
        result["top_tiles"] = [
            {
                "tile": [int(i) // tile_cols, int(i) % tile_cols],
                "count": int(v),
            }
            for v, i in zip(values.tolist(), indices.tolist()) if v > 0
        ]
    return result


def compare_output(output, reference_cpu, mae_limit: float, max_limit: float,
                   sample_limit: int, tile_size: int) -> dict[str, Any]:
    import torch
    output_cpu = output.detach().cpu().contiguous()
    out = output_cpu.float()
    ref = reference_cpu.float()
    finite = bool(torch.isfinite(out).all().item())
    nan_count = int(torch.isnan(out).sum().item())
    inf_count = int(torch.isinf(out).sum().item())
    diff = (out - ref).abs()
    bad_mask = (~torch.isfinite(out)) | (diff > max_limit)
    spatial = spatial_error_analysis(output_cpu, reference_cpu, bad_mask, diff,
                                     max_limit, sample_limit, tile_size)
    if not finite:
        return {
            "finite": False, "nan_count": nan_count, "inf_count": inf_count,
            "mae": None, "max_error": None, "rmse": None, "pass": False,
            "spatial": spatial,
        }
    mae = float(diff.mean().item())
    max_error = float(diff.max().item())
    rmse = float(torch.sqrt(torch.mean((out - ref) ** 2)).item())
    return {
        "finite": True, "nan_count": 0, "inf_count": 0,
        "mae": safe_float(mae), "max_error": safe_float(max_error),
        "rmse": safe_float(rmse),
        "pass": bool(mae <= mae_limit and max_error <= max_limit),
        "spatial": spatial,
    }


def compare_exact_output(output, reference_cpu, sample_limit: int,
                         tile_size: int) -> dict[str, Any]:
    import torch
    output_cpu = output.detach().cpu().contiguous()
    mismatch = output_cpu != reference_cpu
    diff = (output_cpu.to(torch.int64) - reference_cpu.to(torch.int64)).abs().float()
    spatial = spatial_error_analysis(output_cpu, reference_cpu, mismatch, diff,
                                     0.0, sample_limit, tile_size)
    mismatch_count = int(torch.count_nonzero(mismatch).item())
    return {
        "finite": True,
        "nan_count": 0,
        "inf_count": 0,
        "mae": safe_float(diff.mean().item()),
        "max_error": safe_float(diff.max().item()),
        "mismatch_count": mismatch_count,
        "pass": mismatch_count == 0,
        "spatial": spatial,
    }


def make_memory_pattern(name: str, start: int, length: int, seed: int):
    """在 CPU 上生成可重放的 byte pattern，避免用被测 GPU 生成期望值。"""
    import torch
    if name == "zero":
        return torch.zeros(length, dtype=torch.uint8)
    if name == "ff":
        return torch.full((length,), 0xFF, dtype=torch.uint8)
    if name == "aa55":
        out = torch.empty(length, dtype=torch.uint8)
        first = 0 if start % 2 == 0 else 1
        out[first::2] = 0xAA
        out[1 - first::2] = 0x55
        return out
    if name == "address":
        byte_offset = start % 4
        first_word = start // 4
        word_count = (byte_offset + length + 3) // 4
        words = torch.arange(first_word, first_word + word_count,
                             dtype=torch.int64).to(torch.int32)
        return words.view(torch.uint8)[byte_offset:byte_offset + length].clone()
    if name == "random":
        generator = torch.Generator(device="cpu")
        chunk_seed = (seed ^ (start * 0x9E3779B1)) & ((1 << 63) - 1)
        generator.manual_seed(chunk_seed)
        return torch.randint(0, 256, (length,), generator=generator,
                             dtype=torch.uint8)
    raise ValueError(f"未知显存 pattern: {name}")


def memory_mismatch_samples(actual, expected, global_start: int,
                            sample_limit: int) -> tuple[int, list[dict[str, int]]]:
    """统计错误但不为大面积损坏物化全量坐标。"""
    import torch
    mismatch = actual != expected
    count = int(torch.count_nonzero(mismatch).item())
    if count == 0 or sample_limit <= 0:
        return count, []
    samples: list[dict[str, int]] = []
    scan_block = 1 << 20
    for block_start in range(0, mismatch.numel(), scan_block):
        block = mismatch[block_start:block_start + scan_block]
        if not bool(block.any().item()):
            continue
        indices = torch.nonzero(block, as_tuple=False).reshape(-1)
        for local in indices[:sample_limit - len(samples)].tolist():
            offset = block_start + int(local)
            samples.append({
                "byte_offset": global_start + offset,
                "expected": int(expected[offset].item()),
                "actual": int(actual[offset].item()),
                "xor": int(expected[offset].item()) ^ int(actual[offset].item()),
            })
        if len(samples) >= sample_limit:
            break
    return count, samples


def run_memory_pattern(args: argparse.Namespace, result: dict[str, Any], device) -> int:
    import torch
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    reserve_bytes = int(args.memory_reserve_mib * 1024 * 1024)
    usable_bytes = max(0, int(free_bytes) - reserve_bytes)
    requested_bytes = int(usable_bytes * args.memory_fraction)
    chunk_bytes = max(4, int(args.memory_chunk_mib * 1024 * 1024))
    requested_bytes -= requested_bytes % 4
    if requested_bytes < 4:
        raise RuntimeError(
            f"可用显存不足: free={free_bytes}, reserve={reserve_bytes}"
        )

    allocated_bytes = requested_bytes
    buffer = None
    allocation_attempts: list[int] = []
    while allocated_bytes >= min(chunk_bytes, requested_bytes):
        allocation_attempts.append(allocated_bytes)
        try:
            buffer = torch.empty(allocated_bytes, dtype=torch.uint8, device=device)
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            allocated_bytes = int(allocated_bytes * 0.90)
            allocated_bytes -= allocated_bytes % 4
    if buffer is None:
        raise RuntimeError(f"无法分配显存测试缓冲区，尝试={allocation_attempts}")

    chunk_bytes = min(chunk_bytes, allocated_bytes)
    pattern_results: list[dict[str, Any]] = []
    overall_pass = True
    try:
        for pattern_name in args.memory_patterns:
            # 先写完整个缓冲区，再从头校验，避免只做立即往返拷贝。
            for start in range(0, allocated_bytes, chunk_bytes):
                length = min(chunk_bytes, allocated_bytes - start)
                expected = make_memory_pattern(pattern_name, start, length, args.seed)
                buffer[start:start + length].copy_(expected)
            torch.cuda.synchronize()

            mismatch_count = 0
            samples: list[dict[str, int]] = []
            for start in range(0, allocated_bytes, chunk_bytes):
                length = min(chunk_bytes, allocated_bytes - start)
                expected = make_memory_pattern(pattern_name, start, length, args.seed)
                actual = buffer[start:start + length].cpu()
                count, found = memory_mismatch_samples(
                    actual, expected, start,
                    max(0, args.spatial_samples - len(samples)),
                )
                mismatch_count += count
                samples.extend(found)
            passed = mismatch_count == 0
            overall_pass = overall_pass and passed
            pattern_results.append({
                "pattern": pattern_name,
                "pass": passed,
                "mismatch_count": mismatch_count,
                "samples": samples,
            })
    finally:
        del buffer
        torch.cuda.empty_cache()

    result.update({
        "status": "PASS" if overall_pass else "FAIL",
        "pass": overall_pass,
        "memory_test": {
            "free_bytes_before": int(free_bytes),
            "total_bytes": int(total_bytes),
            "requested_bytes": requested_bytes,
            "allocated_bytes": allocated_bytes,
            "coverage_of_total": safe_float(allocated_bytes / max(1, int(total_bytes))),
            "chunk_bytes": chunk_bytes,
            "allocation_attempts": allocation_attempts,
            "patterns": pattern_results,
        },
    })
    print_json(result)
    return 0 if overall_pass else 13


def sdpa_backend(q, k, v, backend_name: str):
    import torch
    import torch.nn.functional as F
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        backend = SDPBackend.MATH if backend_name == "math" else SDPBackend.FLASH_ATTENTION
        try:
            with sdpa_kernel(backend):
                return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        except RuntimeError as exc:
            txt = str(exc).lower()
            if "no available kernel" in txt or "not supported" in txt or "not compiled" in txt:
                raise BackendUnsupported(str(exc)) from exc
            raise
    except ImportError:
        pass

    try:
        if backend_name == "math":
            ctx = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True,
                                                 enable_mem_efficient=False)
        else:
            ctx = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False,
                                                 enable_mem_efficient=False)
        try:
            with ctx:
                return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        except RuntimeError as exc:
            txt = str(exc).lower()
            if "no available kernel" in txt or "not supported" in txt or "not compiled" in txt:
                raise BackendUnsupported(str(exc)) from exc
            raise
    except AttributeError as exc:
        raise BackendUnsupported("当前 PyTorch 无法强制选择 SDPA backend") from exc


def call_sage(func, q, k, v):
    kwargs: dict[str, Any] = {}
    try:
        params = inspect.signature(func).parameters
    except Exception:
        params = {}
    if "tensor_layout" in params:
        kwargs["tensor_layout"] = "HND"
    if "is_causal" in params:
        kwargs["is_causal"] = False
    if "sm89" in params:
        import torch
        kwargs["sm89"] = torch.cuda.get_device_capability(0) == (8, 9)
    out = func(q, k, v, **kwargs)
    return out[0] if isinstance(out, tuple) else out


def run_observed_kernel(call, minimum_seconds: float):
    """保证短 kernel 有足够长的只读遥测观察窗口。"""
    import torch
    started = time.monotonic()
    iterations = 0
    output = None
    while iterations == 0 or time.monotonic() - started < minimum_seconds:
        output = call()
        torch.cuda.synchronize()
        iterations += 1
    elapsed = time.monotonic() - started
    return output, iterations, elapsed


def child_main(args: argparse.Namespace) -> int:
    result = {"stage": args.stage, "status": "CRASH", "pass": False,
              "error": "", "device": "", "capability": None}
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() == False")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(f"子进程应只看到 1 张 GPU，当前={torch.cuda.device_count()}")
        result["device"] = torch.cuda.get_device_name(0)
        result["capability"] = list(torch.cuda.get_device_capability(0))
        result["torch_version"] = torch.__version__
        result["torch_cuda"] = torch.version.cuda

        device = torch.device("cuda:0")
        if args.stage == "memory_pattern":
            return run_memory_pattern(args, result, device)

        bundle = torch.load(args.bundle, map_location="cpu")
        q_cpu, k_cpu, v_cpu = bundle["q"], bundle["k"], bundle["v"]
        q, k, v = q_cpu.to(device), k_cpu.to(device), v_cpu.to(device)
        torch.cuda.synchronize()

        pre = {"q": roundtrip(q, q_cpu), "k": roundtrip(k, k_cpu), "v": roundtrip(v, v_cpu)}
        result["input_precheck"] = pre
        if args.stage != "copy" and not all(x["exact"] for x in pre.values()):
            result.update(status="FAIL", error="计算前 Q/K/V round-trip 已不一致")
            print_json(result)
            return 11

        if args.stage == "copy":
            failures = []
            for i in range(args.copy_loops):
                q2, k2, v2 = q_cpu.to(device), k_cpu.to(device), v_cpu.to(device)
                torch.cuda.synchronize()
                m = {"q": roundtrip(q2, q_cpu), "k": roundtrip(k2, k_cpu), "v": roundtrip(v2, v_cpu)}
                if not all(x["exact"] for x in m.values()):
                    failures.append({"loop": i + 1, **m})
                del q2, k2, v2
            a_cpu, b_cpu = bundle["a"], bundle["b"]
            a, b = a_cpu.to(device), b_cpu.to(device)
            torch.cuda.synchronize()
            ab = {"a": roundtrip(a, a_cpu), "b": roundtrip(b, b_cpu)}
            passed = (not failures and all(x["exact"] for x in pre.values())
                      and all(x["exact"] for x in ab.values()))
            result.update({"status": "PASS" if passed else "FAIL", "pass": passed,
                           "copy_failures": failures, "gemm_input_copy": ab})
            print_json(result)
            return 0 if passed else 10

        if args.stage in GEMM_STAGES:
            a_cpu, b_cpu, ref = bundle["a"], bundle["b"], bundle["gemm_ref"]
            dtype_by_stage = {
                "fp32_gemm": torch.float32,
                "tf32_gemm": torch.float32,
                "fp16_gemm": torch.float16,
                "bf16_gemm": torch.bfloat16,
            }
            limits = {
                "fp32_gemm": (args.fp32_mae_limit, args.fp32_max_limit),
                "tf32_gemm": (args.tf32_mae_limit, args.tf32_max_limit),
                "fp16_gemm": (args.fp16_mae_limit, args.fp16_max_limit),
                "bf16_gemm": (args.gemm_mae_limit, args.gemm_max_limit),
            }
            dtype = dtype_by_stage[args.stage]
            a_typed, b_typed = a_cpu.to(dtype), b_cpu.to(dtype)
            a, b = a_typed.to(device), b_typed.to(device)
            torch.cuda.synchronize()

            old_tf32 = torch.backends.cuda.matmul.allow_tf32
            old_precision = torch.get_float32_matmul_precision()
            try:
                if args.stage == "fp32_gemm":
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.set_float32_matmul_precision("highest")
                elif args.stage == "tf32_gemm":
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.set_float32_matmul_precision("high")
                out, iterations, observed_seconds = run_observed_kernel(
                    lambda: torch.matmul(a, b), args.observation_seconds,
                )
            finally:
                torch.backends.cuda.matmul.allow_tf32 = old_tf32
                torch.set_float32_matmul_precision(old_precision)

            mae_limit, max_limit = limits[args.stage]
            metrics = compare_output(
                out, ref, mae_limit, max_limit,
                args.spatial_samples, args.spatial_tile,
            )
            post = {"a": roundtrip(a, a_typed), "b": roundtrip(b, b_typed)}
            passed = metrics["pass"] and all(x["exact"] for x in post.values())
            result.update({"metrics": metrics, "input_postcheck": post,
                           "compute_path": {
                               "dtype": str(dtype),
                               "tf32_requested": args.stage == "tf32_gemm",
                               "tf32_forced_off": args.stage == "fp32_gemm",
                           },
                           "observation": {
                               "iterations": iterations,
                               "seconds": safe_float(observed_seconds),
                           },
                           "status": "PASS" if passed else "FAIL", "pass": passed})
            if not all(x["exact"] for x in post.values()):
                result["error"] = f"{args.stage} 后输入 A/B round-trip 发生变化"
            print_json(result)
            return 0 if passed else 12

        if args.stage == "int32_alu":
            x_cpu, y_cpu, ref = bundle["int_x"], bundle["int_y"], bundle["int_ref"]
            x, y = x_cpu.to(device), y_cpu.to(device)
            torch.cuda.synchronize()
            out, iterations, observed_seconds = run_observed_kernel(
                lambda: (x * 17 + y * 31) ^ (x << 3),
                args.observation_seconds,
            )
            metrics = compare_exact_output(
                out, ref, args.spatial_samples, args.spatial_tile,
            )
            post = {"x": roundtrip(x, x_cpu), "y": roundtrip(y, y_cpu)}
            passed = metrics["pass"] and all(v["exact"] for v in post.values())
            result.update({
                "metrics": metrics,
                "input_postcheck": post,
                "compute_path": {"dtype": "torch.int32", "exact": True},
                "observation": {
                    "iterations": iterations,
                    "seconds": safe_float(observed_seconds),
                },
                "status": "PASS" if passed else "FAIL",
                "pass": passed,
            })
            if not all(v["exact"] for v in post.values()):
                result["error"] = "INT32 ALU 后输入 X/Y round-trip 发生变化"
            print_json(result)
            return 0 if passed else 12

        ref = bundle["attn_ref"]
        if args.stage == "sdpa_math":
            out, iterations, observed_seconds = run_observed_kernel(
                lambda: sdpa_backend(q, k, v, "math"),
                args.observation_seconds,
            )
            metrics = compare_output(
                out, ref, args.sdpa_mae_limit, args.sdpa_max_limit,
                args.spatial_samples, args.spatial_tile,
            )
        elif args.stage == "sdpa_flash":
            try:
                out, iterations, observed_seconds = run_observed_kernel(
                    lambda: sdpa_backend(q, k, v, "flash"),
                    args.observation_seconds,
                )
            except BackendUnsupported as exc:
                result.update(status="SKIP", error=str(exc))
                print_json(result)
                return 0
            metrics = compare_output(
                out, ref, args.sdpa_mae_limit, args.sdpa_max_limit,
                args.spatial_samples, args.spatial_tile,
            )
        elif args.stage == "sage_fp16":
            try:
                from sageattention import sageattn_qk_int8_pv_fp16_cuda as func
            except Exception as exc:
                result.update(status="SKIP", error=f"Sage FP16 函数不可用: {exc}")
                print_json(result)
                return 0
            result["sage_signature"] = str(inspect.signature(func))
            out, iterations, observed_seconds = run_observed_kernel(
                lambda: call_sage(func, q, k, v),
                args.observation_seconds,
            )
            metrics = compare_output(
                out, ref, args.sage_mae_limit, args.sage_max_limit,
                args.spatial_samples, args.spatial_tile,
            )
        elif args.stage == "sage_fp8":
            try:
                from sageattention import sageattn_qk_int8_pv_fp8_cuda as func
            except Exception as exc:
                result.update(status="SKIP", error=f"Sage FP8 函数不可用: {exc}")
                print_json(result)
                return 0
            result["sage_signature"] = str(inspect.signature(func))
            out, iterations, observed_seconds = run_observed_kernel(
                lambda: call_sage(func, q, k, v),
                args.observation_seconds,
            )
            metrics = compare_output(
                out, ref, args.sage_mae_limit, args.sage_max_limit,
                args.spatial_samples, args.spatial_tile,
            )
        else:
            raise RuntimeError(f"未知 stage: {args.stage}")

        post = {"q": roundtrip(q, q_cpu), "k": roundtrip(k, k_cpu), "v": roundtrip(v, v_cpu)}
        passed = metrics["pass"] and all(x["exact"] for x in post.values())
        result.update({"metrics": metrics, "input_postcheck": post,
                       "observation": {
                           "iterations": iterations,
                           "seconds": safe_float(observed_seconds),
                       },
                       "status": "PASS" if passed else "FAIL", "pass": passed})
        if not all(x["exact"] for x in post.values()):
            result["error"] = "kernel 执行后 Q/K/V round-trip 发生变化"
        print_json(result)
        return 0 if passed else 12

    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        result.update({"status": "CRASH", "pass": False, "error": f"{type(exc).__name__}: {exc}"})
        try:
            print_json(result)
        except Exception:
            pass
        return 99


@dataclass
class ParentResult:
    gpu: int
    config: str
    stage: str
    run: int
    status: str
    passed: bool
    rc: int
    payload: dict[str, Any]
    stderr: str


def parse_json(stdout: str) -> Optional[dict[str, Any]]:
    for line in reversed(stdout.splitlines()):
        try:
            v = json.loads(line.strip())
            if isinstance(v, dict):
                return v
        except Exception:
            pass
    return None


def query_gpu_telemetry(gpu: int) -> dict[str, Any]:
    """只读查询物理 GPU；字段不受支持时自动降级。"""
    field_sets = [
        (
            "timestamp,index,uuid,pstate,temperature.gpu,temperature.memory,"
            "power.draw,power.limit,clocks.current.graphics,clocks.current.memory,"
            "utilization.gpu,utilization.memory,clocks_throttle_reasons.active,"
            "pci.link.gen.current,pci.link.width.current"
        ).split(","),
        (
            "timestamp,index,uuid,pstate,temperature.gpu,power.draw,power.limit,"
            "clocks.current.graphics,clocks.current.memory,utilization.gpu,"
            "utilization.memory,pci.link.gen.current,pci.link.width.current"
        ).split(","),
    ]
    errors: list[str] = []
    for fields in field_sets:
        try:
            p = subprocess.run(
                [
                    "nvidia-smi", "-i", str(gpu),
                    "--query-gpu=" + ",".join(fields),
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=3,
                check=False,
            )
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        if p.returncode != 0 or not p.stdout.strip():
            errors.append(p.stdout.strip()[:300] or f"nvidia-smi rc={p.returncode}")
            continue
        values = [x.strip() for x in p.stdout.strip().splitlines()[-1].split(",")]
        if len(values) != len(fields):
            errors.append(f"遥测字段数不匹配: {p.stdout.strip()[:300]}")
            continue
        return dict(zip(fields, values))
    return {"error": " | ".join(errors)[:1000] or "nvidia-smi 遥测失败"}


def telemetry_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", str(value))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def summarize_telemetry(samples: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    valid = [x for x in samples if "error" not in x]
    summary: dict[str, Any] = {
        "sample_count": len(valid),
        "query_error_count": len(samples) - len(valid),
        "elapsed_seconds": safe_float(elapsed),
    }
    if not valid:
        summary["error"] = next((x.get("error") for x in samples if x.get("error")),
                                "未获取到遥测样本")
        return summary

    numeric_fields = (
        "temperature.gpu", "temperature.memory", "power.draw", "power.limit",
        "clocks.current.graphics", "clocks.current.memory",
        "utilization.gpu", "utilization.memory",
        "pci.link.gen.current", "pci.link.width.current",
    )
    for field in numeric_fields:
        values = [telemetry_number(x.get(field)) for x in valid]
        finite_values = [x for x in values if x is not None and math.isfinite(x)]
        if finite_values:
            summary[field] = {
                "min": min(finite_values),
                "max": max(finite_values),
                "median": statistics.median(finite_values),
            }
    summary["pstates"] = sorted({x.get("pstate", "") for x in valid if x.get("pstate")})
    summary["uuids"] = sorted({x.get("uuid", "") for x in valid if x.get("uuid")})
    reasons = {
        x.get("clocks_throttle_reasons.active", "") for x in valid
        if x.get("clocks_throttle_reasons.active") not in (None, "", "0x0000000000000000")
    }
    summary["active_throttle_reasons"] = sorted(reasons)
    summary["first_sample"] = valid[0]
    summary["last_sample"] = valid[-1]
    return summary


def run_process_with_monitor(cmd: list[str], env: dict[str, str], gpu: int,
                             timeout: float, monitor: bool,
                             interval: float) -> tuple[int, str, str, bool, dict[str, Any]]:
    started = time.monotonic()
    samples: list[dict[str, Any]] = []
    process = subprocess.Popen(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if monitor:
        samples.append(query_gpu_telemetry(gpu))
    timed_out = False
    stdout = ""
    stderr = ""
    while True:
        elapsed = time.monotonic() - started
        remaining = timeout - elapsed
        if remaining <= 0:
            timed_out = True
            process.kill()
            stdout, stderr = process.communicate()
            break
        try:
            wait_for = min(interval if monitor else remaining, remaining)
            stdout, stderr = process.communicate(timeout=wait_for)
            break
        except subprocess.TimeoutExpired:
            if monitor:
                samples.append(query_gpu_telemetry(gpu))
    if monitor:
        samples.append(query_gpu_telemetry(gpu))
    elapsed = time.monotonic() - started
    return (
        process.returncode if process.returncode is not None else 124,
        stdout,
        stderr,
        timed_out,
        summarize_telemetry(samples, elapsed),
    )


def sanitizer_error_count(output: str) -> Optional[int]:
    matches = re.findall(r"ERROR SUMMARY:\s*(\d+)\s+errors?", output)
    return int(matches[-1]) if matches else None


def run_child(args, script: Path, bundle: Path, gpu: int, config: str,
              stage: str, run_index: int) -> ParentResult:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    cmd = [
        sys.executable, str(script), "--child", "--stage", stage, "--bundle", str(bundle),
        "--copy-loops", str(args.copy_loops),
        "--memory-fraction", str(args.memory_fraction),
        "--memory-reserve-mib", str(args.memory_reserve_mib),
        "--memory-chunk-mib", str(args.memory_chunk_mib),
        "--memory-patterns", *args.memory_patterns,
        "--spatial-samples", str(args.spatial_samples),
        "--spatial-tile", str(args.spatial_tile),
        "--observation-seconds", str(args.observation_seconds),
        "--fp32-mae-limit", str(args.fp32_mae_limit),
        "--fp32-max-limit", str(args.fp32_max_limit),
        "--tf32-mae-limit", str(args.tf32_mae_limit),
        "--tf32-max-limit", str(args.tf32_max_limit),
        "--fp16-mae-limit", str(args.fp16_mae_limit),
        "--fp16-max-limit", str(args.fp16_max_limit),
        "--gemm-mae-limit", str(args.gemm_mae_limit), "--gemm-max-limit", str(args.gemm_max_limit),
        "--sdpa-mae-limit", str(args.sdpa_mae_limit), "--sdpa-max-limit", str(args.sdpa_max_limit),
        "--sage-mae-limit", str(args.sage_mae_limit), "--sage-max-limit", str(args.sage_max_limit),
    ]
    use_sanitizer = args.compute_sanitizer and stage in args.sanitizer_stages
    if use_sanitizer:
        executable = shutil.which("compute-sanitizer")
        if not executable:
            return ParentResult(
                gpu, config, stage, run_index, "CRASH", False, 127,
                {"error": "启用了 --compute-sanitizer，但 PATH 中找不到 compute-sanitizer"},
                "",
            )
        cmd = [
            executable,
            "--tool", "memcheck",
            "--error-exitcode", str(args.sanitizer_error_exitcode),
            "--target-processes", "all",
            *cmd,
        ]

    rc, stdout, stderr, timed_out, telemetry = run_process_with_monitor(
        cmd, env, gpu, args.timeout, args.monitor, args.monitor_interval,
    )
    if timed_out:
        return ParentResult(gpu, config, stage, run_index, "TIMEOUT", False, 124,
                            {"error": f"子进程超过 {args.timeout}s 超时",
                             "telemetry": telemetry}, stderr[-5000:])

    payload = parse_json(stdout) or {
        "error": "子进程未返回 JSON",
        "stdout_tail": stdout[-1000:],
    }
    payload["telemetry"] = telemetry
    status = str(payload.get("status", "CRASH" if rc else "UNKNOWN"))
    passed = bool(payload.get("pass", False))
    if use_sanitizer:
        count = sanitizer_error_count(stdout + "\n" + stderr)
        payload["sanitizer"] = {
            "enabled": True,
            "tool": "memcheck",
            "error_count": count,
            "returncode": rc,
            "output_tail": (stdout + "\n" + stderr)[-5000:],
        }
        if ((count is not None and count > 0)
                or rc == args.sanitizer_error_exitcode):
            status = "FAIL"
            passed = False
            payload["status"] = status
            payload["pass"] = False
            count_text = str(count) if count is not None else "未解析数量的"
            payload["error"] = f"Compute Sanitizer 发现 {count_text} 内存访问错误"
    return ParentResult(gpu, config, stage, run_index, status, passed,
                        rc, payload, stderr[-5000:])


def metric(row: ParentResult, key: str) -> Optional[float]:
    try:
        v = row.payload.get("metrics", {}).get(key)
        return None if v is None else float(v)
    except Exception:
        return None


def stats(results: list[ParentResult], gpu: int, config: str, stage: str) -> dict[str, Any]:
    rows = [r for r in results if r.gpu == gpu and r.config == config and r.stage == stage]
    effective = [r for r in rows if r.status != "SKIP"]
    maes = [metric(r, "mae") for r in effective]
    maxes = [metric(r, "max_error") for r in effective]
    maes = [x for x in maes if x is not None and math.isfinite(x)]
    maxes = [x for x in maxes if x is not None and math.isfinite(x)]
    return {
        "runs": len(rows),
        "effective": len(effective),
        "pass": sum(r.passed for r in effective),
        "fail": sum(r.status == "FAIL" for r in effective),
        "crash": sum(r.status == "CRASH" for r in effective),
        "timeout": sum(r.status == "TIMEOUT" for r in effective),
        "skip": sum(r.status == "SKIP" for r in rows),
        "median_mae": statistics.median(maes) if maes else None,
        "median_max": statistics.median(maxes) if maxes else None,
    }


def all_pass(s: dict[str, Any]) -> bool:
    return s["effective"] > 0 and s["pass"] == s["effective"]


def has_failure(s: dict[str, Any]) -> bool:
    return s["effective"] > 0 and s["pass"] < s["effective"]


def target_only_interpretation(results: list[ParentResult], target: int,
                               config: str) -> list[str]:
    S = {stage: stats(results, target, config, stage) for stage in STAGES}
    memory = stats(results, target, DEVICE_CONFIG, "memory_pattern")
    out: list[str] = []

    if has_failure(memory):
        out.append(
            f"- **GPU{target} 大范围显存 pattern 校验失败。** "
            "优先怀疑显存颗粒、显存控制器或显存频率/供电稳定性。"
        )
    if has_failure(S["copy"]):
        out.append(
            f"- **GPU{target} 在 copy 层就出现异常。** "
            "怀疑显存/内存控制器/传输路径/卡状态。"
        )
    if has_failure(S["int32_alu"]):
        out.append(
            f"- GPU{target} 的 **INT32 逐元 ALU 失败**；故障不限于 "
            "Tensor Core，应扩大到普通 SM/ALU/寄存器/缓存。"
        )
    low_precision_failures = [
        stage for stage in ("tf32_gemm", "fp16_gemm", "bf16_gemm")
        if has_failure(S[stage])
    ]
    if (all_pass(S["fp32_gemm"])
            and all_pass(S["int32_alu"])
            and low_precision_failures):
        out.append(
            f"- GPU{target} 的 FP32（关 TF32）和 INT32 正常，但 "
            + "、".join(f"`{x}`" for x in low_precision_failures)
            + " 失败；故障更偏向 Tensor Core/低精度数据通路。"
        )
    if all_pass(S["copy"]) and has_failure(S["bf16_gemm"]):
        out.append(
            f"- GPU{target} 的 copy 正常，但 **BF16 GEMM 相对 CPU 参考失败**；"
            "问题已超出 SageAttention，进入普通 GEMM/Tensor Core/计算层。"
        )
    failed = [
        stage for stage in STAGES if stage not in DEVICE_STAGES
        and has_failure(S[stage])
    ]
    if failed:
        out.append(
            f"- **单卡绝对正确性证据：** GPU{target} 相对固定 CPU reference "
            "失败：" + "、".join(f"`{x}`" for x in failed) + "。"
        )
    sanitizer_errors = [
        row for row in results if row.gpu == target
        and (row.payload.get("sanitizer", {}).get("error_count") or 0) > 0
    ]
    if sanitizer_errors:
        out.append(
            f"- Compute Sanitizer 在 GPU{target} 上发现内存访问错误；"
            "还需要后续用 GPU1 或另一张健康卡运行同一 kernel，"
            "才能区分软件越界与卡特异损坏。"
        )
    if not out:
        out.append(
            f"- GPU{target} 在本次已执行的单卡绝对正确性测试中未发现失败。"
        )
    out.append(
        "- **单卡模式边界：** 可以确认是否偏离 CPU reference，"
        "但不会生成本轮同时期 GPU1/GPU4 A/B 结论。"
    )
    return out


def interpretation(results: list[ParentResult], control: int, target: int,
                   config: str, target_only: bool = False) -> list[str]:
    if target_only:
        return target_only_interpretation(results, target, config)
    S = {stage: {"c": stats(results, control, config, stage),
                 "t": stats(results, target, config, stage)} for stage in STAGES}
    memory = {
        "c": stats(results, control, DEVICE_CONFIG, "memory_pattern"),
        "t": stats(results, target, DEVICE_CONFIG, "memory_pattern"),
    }
    out: list[str] = []

    if all_pass(memory["c"]) and has_failure(memory["t"]):
        out.append(
            f"- **GPU{target} 大范围显存 pattern 校验失败。** "
            "优先怀疑显存颗粒、显存控制器或显存频率/供电稳定性。"
        )
    if all_pass(S["copy"]["c"]) and has_failure(S["copy"]["t"]):
        out.append(f"- **GPU{target} 在 copy 层就出现异常。** 更应怀疑显存/内存控制器/传输路径/卡状态，但不能定位具体显存颗粒。")
    if all_pass(S["int32_alu"]["c"]) and has_failure(S["int32_alu"]["t"]):
        out.append(
            f"- GPU{target} 的 **INT32 逐元 ALU 也失败**；故障不限于 Tensor Core，"
            "应扩大到普通 SM/ALU/寄存器/缓存或整体稳定性。"
        )
    low_precision_failures = [
        stage for stage in ("tf32_gemm", "fp16_gemm", "bf16_gemm")
        if all_pass(S[stage]["c"]) and has_failure(S[stage]["t"])
    ]
    if (all_pass(S["fp32_gemm"]["t"])
            and all_pass(S["int32_alu"]["t"])
            and low_precision_failures):
        out.append(
            f"- GPU{target} 的 FP32（关 TF32）和 INT32 正常，但 "
            + "、".join(f"`{x}`" for x in low_precision_failures)
            + " 失败；故障更偏向 Tensor Core/低精度数据通路。"
        )
    if all_pass(S["copy"]["t"]) and all_pass(S["bf16_gemm"]["c"]) and has_failure(S["bf16_gemm"]["t"]):
        out.append(f"- GPU{target} 的 copy 正常，但 **BF16 GEMM 失败**；问题已经超出 SageAttention，进入普通 GEMM/Tensor Core/计算执行层。")
    if all_pass(S["bf16_gemm"]["t"]) and all_pass(S["sdpa_math"]["c"]) and has_failure(S["sdpa_math"]["t"]):
        out.append(f"- GPU{target} 的 BF16 GEMM 正常，但 **SDPA Math 失败**；普通 attention 数学路径即可触发异常。")
    if all_pass(S["sdpa_math"]["t"]) and all_pass(S["sdpa_flash"]["c"]) and has_failure(S["sdpa_flash"]["t"]):
        out.append(f"- GPU{target} 的 SDPA Math 正常，但 **SDPA Flash/Fused 失败**；触发范围明显偏向 fused attention / Tensor Core / shared memory / register / SM 路径。")
    if all_pass(S["sdpa_math"]["t"]) and all_pass(S["sage_fp16"]["c"]) and has_failure(S["sage_fp16"]["t"]):
        out.append(f"- GPU{target} 的普通 SDPA 仍可工作，但 **Sage FP16 失败**；说明 Sage/custom-kernel 共用路径即可触发，不是 FP8 独有。")
    if all_pass(S["sage_fp16"]["t"]) and all_pass(S["sage_fp8"]["c"]) and has_failure(S["sage_fp8"]["t"]):
        out.append(f"- GPU{target} 的 Sage FP16 正常，只有 **Sage FP8 失败**；这才最明显指向 Sage FP8 专用路径。")

    # 数值倍率提示
    for stage in (*GEMM_STAGES, "sdpa_math", "sdpa_flash", "sage_fp16", "sage_fp8"):
        c, t = S[stage]["c"], S[stage]["t"]
        if c["median_mae"] and t["median_mae"] and t["median_mae"] / c["median_mae"] >= 5:
            out.append(f"- `{stage}` 中位 MAE：GPU{target} 约为 GPU{control} 的 **{t['median_mae']/c['median_mae']:.1f} 倍**。")
        if c["median_max"] and t["median_max"] and t["median_max"] / c["median_max"] >= 10:
            out.append(f"- `{stage}` 中位最大误差：GPU{target} 约为 GPU{control} 的 **{t['median_max']/c['median_max']:.1f} 倍**。")

    for stage in STAGES:
        if stage in DEVICE_STAGES:
            continue
        if has_failure(S[stage]["c"]) and has_failure(S[stage]["t"]):
            out.append(f"- **两张卡都在 `{stage}` 失败。** 应优先排查软件 build、kernel 版本或 shape 支持，而不是直接归因 GPU{target}。")

    card_specific = [
        x for x in STAGES if x not in DEVICE_STAGES
        and all_pass(S[x]["c"]) and has_failure(S[x]["t"])
    ]
    if all_pass(memory["c"]) and has_failure(memory["t"]):
        card_specific.insert(0, "memory_pattern")
    if card_specific:
        out.append(f"- **卡特异性证据：** GPU{control} 正常，而 GPU{target} 在相同输入/软件下失败：" + "、".join(f"`{x}`" for x in card_specific) + "。")
    if not out:
        out.append("- 当前结果没有形成足够清晰的分层模式，建议增加 run 数或执行 `--size-sweep`。")
    out.append("- **结论边界：** 本脚本能定位从哪类执行路径开始异常，但不能单独证明具体坏的是哪颗显存、哪个 SM/Tensor Core、register file、shared memory 或 memory controller。")
    return out


def telemetry_stat(payload: dict[str, Any], field: str,
                   statistic: str = "max") -> Optional[float]:
    try:
        return safe_float(payload["telemetry"][field][statistic])
    except (KeyError, TypeError, ValueError):
        return None


def spatial_field(metrics: dict[str, Any], field: str) -> Any:
    try:
        return metrics["spatial"].get(field)
    except (KeyError, TypeError):
        return None


def markdown_cell(value: Any, limit: int = 220) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value).replace("|", "\\|").replace("\n", " ")[:limit]


def append_run_details(lines: list[str], rows: list[ParentResult]) -> None:
    detailed = []
    for row in rows:
        metrics = row.payload.get("metrics", {})
        spatial = metrics.get("spatial", {})
        sanitizer = row.payload.get("sanitizer")
        if spatial.get("severe_count", 0) or sanitizer:
            detailed.append((row, spatial, sanitizer))
    if not detailed:
        return
    lines += ["### 错误坐标、位模式与 Sanitizer 详情", ""]
    for row, spatial, sanitizer in detailed:
        lines += [
            f"#### GPU {row.gpu} | {STAGE_CN[row.stage]} | Run {row.run}", "",
        ]
        if spatial.get("severe_count", 0):
            details = {
                "shape": spatial.get("shape"),
                "element_limit": spatial.get("element_limit"),
                "severe_count": spatial.get("severe_count"),
                "severe_rate": spatial.get("severe_rate"),
                "first_severe": spatial.get("first_severe"),
                "top_tiles": spatial.get("top_tiles"),
                "top_errors": spatial.get("top_errors"),
            }
            lines += ["```json", json.dumps(details, ensure_ascii=False, indent=2), "```", ""]
        if sanitizer:
            lines += ["```text", str(sanitizer.get("output_tail", "")), "```", ""]


def telemetry_range(payload: dict[str, Any], field: str,
                    suffix: str = "") -> str:
    low = telemetry_stat(payload, field, "min")
    high = telemetry_stat(payload, field, "max")
    if low is None:
        return "-"
    return f"{fmt(low)}..{fmt(high)}{suffix}"


def append_telemetry_table(lines: list[str], rows: list[ParentResult]) -> None:
    if not any(row.payload.get("telemetry") for row in rows):
        return
    lines += [
        "### 只读 GPU 遥测汇总", "",
        "| GPU | 阶段 | Run | 样本/秒 | P-State | 核心温度 | 显存温度 | 功耗 | 核心频率 | 显存频率 | GPU 利用率 | PCIe | 限频原因 |",
        "|---:|---|---:|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        telemetry = row.payload.get("telemetry", {})
        sample_text = (
            f"{telemetry.get('sample_count', 0)} / "
            f"{fmt(telemetry.get('elapsed_seconds'))}"
        )
        pcie_gen = telemetry_stat(row.payload, "pci.link.gen.current")
        pcie_width = telemetry_stat(row.payload, "pci.link.width.current")
        pcie = "-" if pcie_gen is None else f"Gen{fmt(pcie_gen)} x{fmt(pcie_width)}"
        lines.append(
            f"| {row.gpu} | {STAGE_CN[row.stage]} | {row.run} | {sample_text} | "
            f"{markdown_cell(telemetry.get('pstates'))} | "
            f"{telemetry_range(row.payload, 'temperature.gpu', '°C')} | "
            f"{telemetry_range(row.payload, 'temperature.memory', '°C')} | "
            f"{telemetry_range(row.payload, 'power.draw', 'W')} | "
            f"{telemetry_range(row.payload, 'clocks.current.graphics', 'MHz')} | "
            f"{telemetry_range(row.payload, 'clocks.current.memory', 'MHz')} | "
            f"{telemetry_range(row.payload, 'utilization.gpu', '%')} | {pcie} | "
            f"{markdown_cell(telemetry.get('active_throttle_reasons'))} |"
        )
    lines.append("")


def write_report(path: Path, results: list[ParentResult], configs: list[tuple],
                 control: int, target: int, infos: dict[int, str],
                 before: str, after: str, args: argparse.Namespace) -> None:
    identity_lines = (
        [
            "- 运行模式：**仅目标 GPU（无本轮实时对照）**",
            f"- 目标物理 GPU：`GPU {target}`",
        ]
        if args.target_only
        else [
            f"- 健康对照物理 GPU：`GPU {control}`",
            f"- 目标物理 GPU：`GPU {target}`",
        ]
    )
    gpus = (target,) if args.target_only else (control, target)
    lines = [
        "# SageAttention / GPU 分层诊断报告", "",
        f"- 生成时间：`{now_text()}`",
        *identity_lines,
        f"- 每计算阶段独立运行次数：`{args.runs}`",
        "- 大范围显存模式校验：每张物理 GPU 只运行 1 次",
        "- 每个 GPU / stage / run：**全新 Python 子进程**",
        "- 子进程：`CUDA_LAUNCH_BLOCKING=1`，且只看到 `cuda:0`",
        f"- 只读 GPU 遥测：`{'enabled' if args.monitor else 'disabled'}`",
        f"- 每个计算阶段的最小遥测观察窗：`{args.observation_seconds}s`",
        f"- Compute Sanitizer：`{'enabled' if args.compute_sanitizer else 'disabled'}`", "",
        "## 一、GPU 信息", "",
    ]
    for gpu, info in infos.items():
        lines += [f"### GPU {gpu}", "", "```text", info or "(无数据)", "```", ""]

    if "memory_pattern" in args.stages:
        lines += [
            "## 二、设备级大范围显存校验", "",
            f"- 目标分配比例：扣除 `{args.memory_reserve_mib} MiB` 后的空闲显存的 `{args.memory_fraction:.0%}`",
            f"- 分块大小：`{args.memory_chunk_mib} MiB`",
            "- Pattern：" + "、".join(f"`{x}`" for x in args.memory_patterns), "",
            "| GPU | 状态 | 实际覆盖 | 总显存覆盖率 | Pattern 结果 | 峰值核心温度 | 峰值显存温度 | 峰值功耗 |",
            "|---:|---|---:|---:|---|---:|---:|---:|",
        ]
        memory_rows = [r for r in results if r.config == DEVICE_CONFIG]
        for row in memory_rows:
            memory = row.payload.get("memory_test", {})
            patterns = memory.get("patterns", [])
            pattern_text = ", ".join(
                f"{x.get('pattern')}={'PASS' if x.get('pass') else 'FAIL'}"
                f"({x.get('mismatch_count', 0)})" for x in patterns
            ) or "-"
            lines.append(
                f"| {row.gpu} | {row.status} | {memory.get('allocated_bytes', 0) / (1024 ** 3):.2f} GiB | "
                f"{fmt(memory.get('coverage_of_total'))} | {markdown_cell(pattern_text)} | "
                f"{fmt(telemetry_stat(row.payload, 'temperature.gpu'))} | "
                f"{fmt(telemetry_stat(row.payload, 'temperature.memory'))} | "
                f"{fmt(telemetry_stat(row.payload, 'power.draw'))} W |"
            )
        lines.append("")
        for row in memory_rows:
            failures = [
                x for x in row.payload.get("memory_test", {}).get("patterns", [])
                if not x.get("pass")
            ]
            if failures:
                lines += [
                    f"### GPU {row.gpu} 显存错误样本", "",
                    "```json", json.dumps(failures, ensure_ascii=False, indent=2), "```", "",
                ]

    selected_compute_stages = [x for x in args.stages if x not in DEVICE_STAGES]
    for config_name, batch, heads, seq, dim, gemm_size in configs:
        lines += [
            f"## 三、计算测试配置：{config_name}", "",
            f"- Attention：`[B={batch}, H={heads}, N={seq}, D={dim}]`，输入 `bfloat16`",
            "- Attention Reference：CPU float32 `softmax(QK^T/sqrt(D)) @ V`",
            f"- BF16 GEMM：`[{gemm_size},{gemm_size}] @ [{gemm_size},{gemm_size}]`",
            "- GEMM Reference：CPU float32 matmul", "",
            "### 2.1 汇总", "",
            "| GPU | 阶段 | PASS | FAIL | CRASH | TIMEOUT | SKIP | 中位 MAE | 中位最大误差 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for gpu in gpus:
            for stage in selected_compute_stages:
                s = stats(results, gpu, config_name, stage)
                lines.append(f"| {gpu} | {STAGE_CN[stage]} | {s['pass']} | {s['fail']} | {s['crash']} | {s['timeout']} | {s['skip']} | {fmt(s['median_mae'])} | {fmt(s['median_max'])} |")

        lines += ["", "### 2.2 自动诊断", ""] + interpretation(
            results, control, target, config_name, args.target_only,
        )
        lines += ["", "### 2.3 每次独立运行", "",
                  "| GPU | 阶段 | Run | 状态 | RC | 观察循环/秒 | MAE | 最大误差 | 严重异常数 | 首个坐标 | Top tile | 峰值温度 | 峰值功耗 | 核心频率范围 | 限频原因 | 错误摘要 |",
                  "|---:|---|---:|---|---:|---|---:|---:|---:|---|---|---:|---:|---|---|---|"]
        config_rows = [r for r in results if r.config == config_name]
        for r in config_rows:
            m = r.payload.get("metrics", {})
            err = str(r.payload.get("error", "") or "")
            if not err and r.stderr and r.status in ("CRASH", "TIMEOUT"):
                ps = r.stderr.splitlines()
                if ps:
                    err = ps[-1]
            top_tiles = spatial_field(m, "top_tiles") or []
            top_tile = top_tiles[0] if top_tiles else None
            clock_min = telemetry_stat(r.payload, "clocks.current.graphics", "min")
            clock_max = telemetry_stat(r.payload, "clocks.current.graphics", "max")
            clock_range = "-" if clock_min is None else f"{fmt(clock_min)}..{fmt(clock_max)} MHz"
            throttle = r.payload.get("telemetry", {}).get("active_throttle_reasons", [])
            observation = r.payload.get("observation", {})
            observation_text = (
                f"{observation.get('iterations', '-')} / "
                f"{fmt(observation.get('seconds'))}"
            )
            lines.append(
                f"| {r.gpu} | {STAGE_CN[r.stage]} | {r.run} | {r.status} | {r.rc} | "
                f"{observation_text} | {fmt(m.get('mae'))} | "
                f"{fmt(m.get('max_error'))} | "
                f"{fmt(spatial_field(m, 'severe_count'))} | "
                f"{markdown_cell(spatial_field(m, 'first_severe'))} | "
                f"{markdown_cell(top_tile)} | "
                f"{fmt(telemetry_stat(r.payload, 'temperature.gpu'))} | "
                f"{fmt(telemetry_stat(r.payload, 'power.draw'))} W | "
                f"{clock_range} | {markdown_cell(throttle)} | {markdown_cell(err)} |"
            )
        lines.append("")
        append_run_details(lines, config_rows)
        append_telemetry_table(lines, config_rows)

    lines += [
        "## 四、测试前 NVIDIA/Xid 内核日志快照", "", "```text", before, "```", "",
        "## 五、测试后 NVIDIA/Xid 内核日志快照", "", "```text", after, "```", "",
        "## 六、结果阅读顺序", "",
        "1. **memory_pattern 失败**：优先显存颗粒/显存控制器/显存频率或供电稳定性。",
        "2. **INT32 也失败**：不是低精度 Tensor Core 独有，扩大到普通 SM/ALU/寄存器/缓存。",
        "3. **FP32/INT32 正常，TF32/FP16/BF16 失败**：偏向 Tensor Core/低精度数据通路。",
        "4. **copy 正常、BF16 GEMM 失败**：问题进入普通 GEMM/Tensor Core/计算层。",
        "5. **SDPA 正常、Sage 失败**：再结合 Compute Sanitizer 判断 custom kernel 是否越界。", "",
        "### 重要限制", "",
        "- 本报告用于缩小故障层级，不等于定位具体物理坏点。",
        "- 遥测全程只读，本脚本不会修改频率、电压或功耗上限。",
        "- 若健康对照卡也失败，应优先排查软件 build / kernel / shape 支持。",
        "- 若目标卡在固定输入、独立进程下结果仍波动明显，这是额外的卡稳定性异常信号。", "",
    ]
    if args.target_only:
        lines += [
            "- 本轮未访问健康对照卡；单卡失败表示偏离 CPU reference，"
            "不应写成同时期的卡特异 A/B 结论。", "",
        ]
    new_section = "\n".join(lines).rstrip() + "\n"

    # 默认更新同一个历史报告，但不覆盖第一轮原始内容。
    # 第一次使用新版脚本时：
    #   1) 若旧报告已存在，先自动生成一次备份；
    #   2) 把新版中文诊断追加到旧报告末尾。
    # 后续再次运行新版脚本时，只替换 SAGE_GPU_CN_V2 标记之间的区域，
    # 因此第一轮旧报告始终保持原样。
    if path.exists():
        original = path.read_text(encoding="utf-8")

        backup = path.with_name(path.stem + ".before_cn_v2.md")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")

        if UPDATE_BEGIN in original and UPDATE_END in original:
            start = original.index(UPDATE_BEGIN)
            end = original.index(UPDATE_END, start) + len(UPDATE_END)
            merged = (
                original[:start].rstrip()
                + "\n\n"
                + UPDATE_BEGIN
                + "\n"
                + new_section
                + UPDATE_END
                + original[end:]
            )
        else:
            merged = (
                original.rstrip()
                + "\n\n---\n\n"
                + UPDATE_BEGIN
                + "\n"
                + new_section
                + UPDATE_END
                + "\n"
            )

        path.write_text(merged, encoding="utf-8")
    else:
        path.write_text(
            UPDATE_BEGIN
            + "\n"
            + new_section
            + UPDATE_END
            + "\n",
            encoding="utf-8",
        )


def parent_main(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve()
    report = Path(args.report).resolve()
    if not args.target_only and args.control_gpu == args.target_gpu:
        print("ERROR: control GPU 和 target GPU 不能相同", file=sys.stderr)
        return 2
    if args.runs <= 0 or args.copy_loops <= 0:
        print("ERROR: --runs 和 --copy-loops 必须大于 0", file=sys.stderr)
        return 2
    if not 0 < args.memory_fraction <= 1:
        print("ERROR: --memory-fraction 必须在 (0, 1] 内", file=sys.stderr)
        return 2
    if args.memory_reserve_mib < 0 or args.memory_chunk_mib <= 0:
        print("ERROR: 显存 reserve 不能为负，chunk 必须大于 0", file=sys.stderr)
        return 2
    if args.monitor_interval <= 0 or args.observation_seconds < 0:
        print(
            "ERROR: --monitor-interval 必须大于 0，"
            "--observation-seconds 不能为负",
            file=sys.stderr,
        )
        return 2
    if args.spatial_samples < 0 or args.spatial_tile <= 0:
        print("ERROR: --spatial-samples 不能为负，--spatial-tile 必须大于 0", file=sys.stderr)
        return 2
    if args.compute_sanitizer and shutil.which("compute-sanitizer") is None:
        print("ERROR: 已要求 Compute Sanitizer，但 PATH 中找不到 compute-sanitizer", file=sys.stderr)
        return 2

    if args.size_sweep:
        configs = [(f"seq{n}_dim128", args.batch, args.heads, n, 128, args.gemm_size)
                   for n in (64, 128, 256, 512, 1024)]
    else:
        configs = [(f"seq{args.seq}_dim{args.dim}", args.batch, args.heads,
                    args.seq, args.dim, args.gemm_size)]

    print("=== SageAttention / GPU 分层诊断 ===")
    if args.target_only:
        print("运行模式:          仅目标 GPU（不访问对照卡）")
    else:
        print(f"健康对照物理 GPU: {args.control_gpu}")
    print(f"目标物理 GPU:     {args.target_gpu}")
    print(f"Runs/stage:        {args.runs}")
    print("Stages:            " + ", ".join(args.stages))
    print(f"GEMM/INT32 size:   {args.gemm_size}")
    print(f"只读遥测:          {'开启' if args.monitor else '关闭'}")
    print(f"Compute Sanitizer: {'开启' if args.compute_sanitizer else '关闭'}")
    print(f"诊断报告（原文件内更新）: {report}")
    print("每个 GPU/stage/run 都是全新 Python 子进程。\n")

    gpus = (args.target_gpu,) if args.target_only else (
        args.control_gpu, args.target_gpu,
    )
    infos = {gpu: nvidia_info(gpu) for gpu in gpus}
    before = kernel_log()
    results: list[ParentResult] = []

    with tempfile.TemporaryDirectory(prefix="sage_gpu_diag_cn_") as td:
        td = Path(td)
        if "memory_pattern" in args.stages:
            print("\n=== 设备级大范围显存 pattern 校验（每张卡 1 次）===")
            for gpu in gpus:
                print(f"  物理 GPU {gpu}: memory_pattern")
                r = run_child(
                    args, script, td / "unused-memory-bundle.pt", gpu,
                    DEVICE_CONFIG, "memory_pattern", 1,
                )
                results.append(r)
                memory = r.payload.get("memory_test", {})
                print(
                    f"    {r.status:<7} rc={r.rc:<3} "
                    f"covered={memory.get('allocated_bytes', 0) / (1024 ** 3):.2f} GiB"
                )
                if r.payload.get("error"):
                    print("      " + str(r.payload["error"])[:260])

        compute_stages = [x for x in args.stages if x not in DEVICE_STAGES]
        if compute_stages:
            for config_name, batch, heads, seq, dim, gemm_size in configs:
                bundle = td / f"{config_name}.pt"
                print(f"\n准备 CPU 固定输入/reference: B={batch} H={heads} N={seq} D={dim}, GEMM={gemm_size}")
                create_bundle(bundle, batch, heads, seq, dim, gemm_size, args.seed)
                for gpu in gpus:
                    print(f"\n--- 物理 GPU {gpu} | {config_name} ---")
                    for stage in compute_stages:
                        print(f"  Stage: {stage} ({STAGE_CN[stage]})")
                        for i in range(1, args.runs + 1):
                            r = run_child(args, script, bundle, gpu, config_name, stage, i)
                            results.append(r)
                            print(f"    run {i:02d}: {r.status:<7} rc={r.rc:<3} mae={fmt(metric(r,'mae')):>10} max={fmt(metric(r,'max_error')):>10}")
                            if r.status in ("FAIL", "CRASH", "TIMEOUT") and r.payload.get("error"):
                                print("      " + str(r.payload["error"])[:260])

    after = kernel_log()
    write_report(report, results, configs, args.control_gpu, args.target_gpu,
                 infos, before, after, args)

    print("\n" + "=" * 80)
    print(f"诊断报告已更新: {report}")
    backup = report.with_name(report.stem + ".before_cn_v2.md")
    if backup.exists():
        print(f"第一轮报告自动备份: {backup}")
    summary_config = configs[-1][0] if args.size_sweep else configs[0][0]
    print("\n自动诊断摘要：")
    for line in interpretation(
        results, args.control_gpu, args.target_gpu,
        summary_config, args.target_only,
    ):
        print(line)
    print("=" * 80)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SageAttention / GPU 分层 A/B 诊断")
    p.add_argument("--control-gpu", type=int, default=1, help="健康对照物理 GPU")
    p.add_argument("--target-gpu", type=int, default=4, help="疑似异常物理 GPU")
    p.add_argument("--target-only", action="store_true",
                   help="仅访问 target GPU，不查询也不创建 control GPU 上下文")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--gemm-size", type=int, default=DEFAULT_GEMM_SIZE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    p.add_argument("--copy-loops", type=int, default=5)
    p.add_argument("--memory-fraction", type=float, default=DEFAULT_MEMORY_FRACTION,
                   help="显存测试使用扣除 reserve 后空闲显存的比例")
    p.add_argument("--memory-reserve-mib", type=int, default=DEFAULT_MEMORY_RESERVE_MIB,
                   help="显存测试为驱动/桌面/其他程序保留的 MiB")
    p.add_argument("--memory-chunk-mib", type=int, default=DEFAULT_MEMORY_CHUNK_MIB,
                   help="CPU↔GPU pattern 写入/校验分块大小")
    p.add_argument("--memory-patterns", nargs="+",
                   choices=("zero", "ff", "aa55", "address", "random"),
                   default=["zero", "ff", "aa55", "address", "random"])
    p.add_argument("--spatial-samples", type=int, default=DEFAULT_SPATIAL_SAMPLES,
                   help="每次运行保留的错误坐标/位模式样本数")
    p.add_argument("--spatial-tile", type=int, default=DEFAULT_SPATIAL_TILE,
                   help="错误空间聚类的 tile 边长")
    p.add_argument("--no-monitor", dest="monitor", action="store_false",
                   help="关闭只读 nvidia-smi 遥测（默认开启）")
    p.set_defaults(monitor=True)
    p.add_argument("--monitor-interval", type=float, default=DEFAULT_MONITOR_INTERVAL,
                   help="只读 GPU 遥测采样间隔（秒）")
    p.add_argument("--observation-seconds", type=float,
                   default=DEFAULT_OBSERVATION_SECONDS,
                   help="短计算 kernel 为只读遥测保持的最小运行窗口")
    p.add_argument("--compute-sanitizer", action="store_true",
                   help="用 Compute Sanitizer memcheck 包裹指定阶段")
    p.add_argument("--sanitizer-stages", nargs="+", choices=STAGES,
                   default=["sage_fp16", "sage_fp8"])
    p.add_argument("--sanitizer-error-exitcode", type=int, default=97,
                   help=argparse.SUPPRESS)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--size-sweep", action="store_true")
    p.add_argument("--report", default=REPORT_NAME)
    p.add_argument("--fp32-mae-limit", type=float, default=FP32_MAE_LIMIT)
    p.add_argument("--fp32-max-limit", type=float, default=FP32_MAX_LIMIT)
    p.add_argument("--tf32-mae-limit", type=float, default=TF32_MAE_LIMIT)
    p.add_argument("--tf32-max-limit", type=float, default=TF32_MAX_LIMIT)
    p.add_argument("--fp16-mae-limit", type=float, default=FP16_MAE_LIMIT)
    p.add_argument("--fp16-max-limit", type=float, default=FP16_MAX_LIMIT)
    p.add_argument("--gemm-mae-limit", type=float, default=GEMM_MAE_LIMIT)
    p.add_argument("--gemm-max-limit", type=float, default=GEMM_MAX_LIMIT)
    p.add_argument("--sdpa-mae-limit", type=float, default=SDPA_MAE_LIMIT)
    p.add_argument("--sdpa-max-limit", type=float, default=SDPA_MAX_LIMIT)
    p.add_argument("--sage-mae-limit", type=float, default=SAGE_MAE_LIMIT)
    p.add_argument("--sage-max-limit", type=float, default=SAGE_MAX_LIMIT)
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--stage", choices=STAGES, help=argparse.SUPPRESS)
    p.add_argument("--bundle", help=argparse.SUPPRESS)
    return p


def main() -> int:
    p = build_parser()
    args = p.parse_args()
    if args.child:
        if not args.stage or not args.bundle:
            p.error("--child 需要 --stage 和 --bundle")
        return child_main(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
