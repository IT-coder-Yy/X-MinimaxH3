#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SageAttention / GPU 分层 A/B 诊断脚本（中文报告）。

默认阶段：
  copy -> bf16_gemm -> sdpa_math -> sdpa_flash -> sage_fp16 -> sage_fp8

关键点：
- CPU 固定输入 + CPU float32 reference
- 每个 GPU/stage/run 都是全新的 Python 子进程
- 父进程负责物理 GPU 映射，子进程只使用 cuda:0
- CUDA_LAUNCH_BLOCKING=1
- 输出中文 Markdown 诊断章节；默认更新已有 sage_gpu_diagnosis.md，并保留第一轮原始内容

示例：
  python diagnose_sage_gpu_cn.py --control-gpu 1 --target-gpu 4 --runs 3
  python diagnose_sage_gpu_cn.py --control-gpu 1 --target-gpu 4 --runs 3 --size-sweep
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

STAGES = ("copy", "bf16_gemm", "sdpa_math", "sdpa_flash", "sage_fp16", "sage_fp8")
STAGE_CN = {
    "copy": "CPU↔GPU 拷贝完整性",
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
DEFAULT_TIMEOUT = 240

# A/B 诊断阈值：故意比上一版更严格，尤其是 SDPA。
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

    torch.save({
        "q": q.contiguous(), "k": k.contiguous(), "v": v.contiguous(),
        "attn_ref": attn_ref,
        "a": a.contiguous(), "b": b.contiguous(), "gemm_ref": gemm_ref,
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


def compare_output(output, reference_cpu, mae_limit: float, max_limit: float) -> dict[str, Any]:
    import torch
    out = output.detach().float().cpu()
    ref = reference_cpu.float()
    finite = bool(torch.isfinite(out).all().item())
    nan_count = int(torch.isnan(out).sum().item())
    inf_count = int(torch.isinf(out).sum().item())
    if not finite:
        return {"finite": False, "nan_count": nan_count, "inf_count": inf_count,
                "mae": None, "max_error": None, "rmse": None, "pass": False}
    diff = (out - ref).abs()
    mae = float(diff.mean().item())
    max_error = float(diff.max().item())
    rmse = float(torch.sqrt(torch.mean((out - ref) ** 2)).item())
    return {
        "finite": True, "nan_count": 0, "inf_count": 0,
        "mae": safe_float(mae), "max_error": safe_float(max_error),
        "rmse": safe_float(rmse),
        "pass": bool(mae <= mae_limit and max_error <= max_limit),
    }


def sdpa_backend(q, k, v, backend_name: str):
    import torch
    import torch.nn.functional as F
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
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

        bundle = torch.load(args.bundle, map_location="cpu")
        device = torch.device("cuda:0")
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

        if args.stage == "bf16_gemm":
            a_cpu, b_cpu, ref = bundle["a"], bundle["b"], bundle["gemm_ref"]
            a, b = a_cpu.to(device), b_cpu.to(device)
            torch.cuda.synchronize()
            out = torch.matmul(a, b)
            torch.cuda.synchronize()
            metrics = compare_output(out, ref, args.gemm_mae_limit, args.gemm_max_limit)
            post = {"a": roundtrip(a, a_cpu), "b": roundtrip(b, b_cpu)}
            passed = metrics["pass"] and all(x["exact"] for x in post.values())
            result.update({"metrics": metrics, "input_postcheck": post,
                           "status": "PASS" if passed else "FAIL", "pass": passed})
            if not all(x["exact"] for x in post.values()):
                result["error"] = "BF16 GEMM 后输入 A/B round-trip 发生变化"
            print_json(result)
            return 0 if passed else 12

        ref = bundle["attn_ref"]
        if args.stage == "sdpa_math":
            out = sdpa_backend(q, k, v, "math")
            torch.cuda.synchronize()
            metrics = compare_output(out, ref, args.sdpa_mae_limit, args.sdpa_max_limit)
        elif args.stage == "sdpa_flash":
            try:
                out = sdpa_backend(q, k, v, "flash")
            except BackendUnsupported as exc:
                result.update(status="SKIP", error=str(exc))
                print_json(result)
                return 0
            torch.cuda.synchronize()
            metrics = compare_output(out, ref, args.sdpa_mae_limit, args.sdpa_max_limit)
        elif args.stage == "sage_fp16":
            try:
                from sageattention import sageattn_qk_int8_pv_fp16_cuda as func
            except Exception as exc:
                result.update(status="SKIP", error=f"Sage FP16 函数不可用: {exc}")
                print_json(result)
                return 0
            result["sage_signature"] = str(inspect.signature(func))
            out = call_sage(func, q, k, v)
            torch.cuda.synchronize()
            metrics = compare_output(out, ref, args.sage_mae_limit, args.sage_max_limit)
        elif args.stage == "sage_fp8":
            try:
                from sageattention import sageattn_qk_int8_pv_fp8_cuda as func
            except Exception as exc:
                result.update(status="SKIP", error=f"Sage FP8 函数不可用: {exc}")
                print_json(result)
                return 0
            result["sage_signature"] = str(inspect.signature(func))
            out = call_sage(func, q, k, v)
            torch.cuda.synchronize()
            metrics = compare_output(out, ref, args.sage_mae_limit, args.sage_max_limit)
        else:
            raise RuntimeError(f"未知 stage: {args.stage}")

        post = {"q": roundtrip(q, q_cpu), "k": roundtrip(k, k_cpu), "v": roundtrip(v, v_cpu)}
        passed = metrics["pass"] and all(x["exact"] for x in post.values())
        result.update({"metrics": metrics, "input_postcheck": post,
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


def run_child(args, script: Path, bundle: Path, gpu: int, config: str,
              stage: str, run_index: int) -> ParentResult:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    cmd = [
        sys.executable, str(script), "--child", "--stage", stage, "--bundle", str(bundle),
        "--copy-loops", str(args.copy_loops),
        "--gemm-mae-limit", str(args.gemm_mae_limit), "--gemm-max-limit", str(args.gemm_max_limit),
        "--sdpa-mae-limit", str(args.sdpa_mae_limit), "--sdpa-max-limit", str(args.sdpa_max_limit),
        "--sage-mae-limit", str(args.sage_mae_limit), "--sage-max-limit", str(args.sage_max_limit),
    ]
    try:
        p = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=args.timeout, check=False)
        payload = parse_json(p.stdout) or {"error": "子进程未返回 JSON", "stdout_tail": p.stdout[-1000:]}
        status = str(payload.get("status", "CRASH" if p.returncode else "UNKNOWN"))
        passed = bool(payload.get("pass", False))
        return ParentResult(gpu, config, stage, run_index, status, passed,
                            p.returncode, payload, p.stderr[-5000:])
    except subprocess.TimeoutExpired as exc:
        return ParentResult(gpu, config, stage, run_index, "TIMEOUT", False, 124,
                            {"error": f"子进程超过 {args.timeout}s 超时"},
                            exc.stderr[-5000:] if isinstance(exc.stderr, str) else "")


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


def interpretation(results: list[ParentResult], control: int, target: int, config: str) -> list[str]:
    S = {stage: {"c": stats(results, control, config, stage),
                 "t": stats(results, target, config, stage)} for stage in STAGES}
    out: list[str] = []

    if all_pass(S["copy"]["c"]) and has_failure(S["copy"]["t"]):
        out.append(f"- **GPU{target} 在 copy 层就出现异常。** 更应怀疑显存/内存控制器/传输路径/卡状态，但不能定位具体显存颗粒。")
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
    for stage in ("bf16_gemm", "sdpa_math", "sdpa_flash", "sage_fp16", "sage_fp8"):
        c, t = S[stage]["c"], S[stage]["t"]
        if c["median_mae"] and t["median_mae"] and t["median_mae"] / c["median_mae"] >= 5:
            out.append(f"- `{stage}` 中位 MAE：GPU{target} 约为 GPU{control} 的 **{t['median_mae']/c['median_mae']:.1f} 倍**。")
        if c["median_max"] and t["median_max"] and t["median_max"] / c["median_max"] >= 10:
            out.append(f"- `{stage}` 中位最大误差：GPU{target} 约为 GPU{control} 的 **{t['median_max']/c['median_max']:.1f} 倍**。")

    for stage in STAGES:
        if has_failure(S[stage]["c"]) and has_failure(S[stage]["t"]):
            out.append(f"- **两张卡都在 `{stage}` 失败。** 应优先排查软件 build、kernel 版本或 shape 支持，而不是直接归因 GPU{target}。")

    card_specific = [x for x in STAGES if all_pass(S[x]["c"]) and has_failure(S[x]["t"])]
    if card_specific:
        out.append(f"- **卡特异性证据：** GPU{control} 正常，而 GPU{target} 在相同输入/软件下失败：" + "、".join(f"`{x}`" for x in card_specific) + "。")
    if not out:
        out.append("- 当前结果没有形成足够清晰的分层模式，建议增加 run 数或执行 `--size-sweep`。")
    out.append("- **结论边界：** 本脚本能定位从哪类执行路径开始异常，但不能单独证明具体坏的是哪颗显存、哪个 SM/Tensor Core、register file、shared memory 或 memory controller。")
    return out


def write_report(path: Path, results: list[ParentResult], configs: list[tuple],
                 control: int, target: int, runs: int, infos: dict[int, str],
                 before: str, after: str) -> None:
    lines = [
        "# SageAttention / GPU 分层诊断报告", "",
        f"- 生成时间：`{now_text()}`",
        f"- 健康对照物理 GPU：`GPU {control}`",
        f"- 目标物理 GPU：`GPU {target}`",
        f"- 每阶段独立运行次数：`{runs}`",
        "- 每个 GPU / stage / run：**全新 Python 子进程**",
        "- 子进程：`CUDA_LAUNCH_BLOCKING=1`，且只看到 `cuda:0`", "",
        "## 一、GPU 信息", "",
    ]
    for gpu, info in infos.items():
        lines += [f"### GPU {gpu}", "", "```text", info or "(无数据)", "```", ""]

    for config_name, batch, heads, seq, dim, gemm_size in configs:
        lines += [
            f"## 二、测试配置：{config_name}", "",
            f"- Attention：`[B={batch}, H={heads}, N={seq}, D={dim}]`，输入 `bfloat16`",
            "- Attention Reference：CPU float32 `softmax(QK^T/sqrt(D)) @ V`",
            f"- BF16 GEMM：`[{gemm_size},{gemm_size}] @ [{gemm_size},{gemm_size}]`",
            "- GEMM Reference：CPU float32 matmul", "",
            "### 2.1 汇总", "",
            "| GPU | 阶段 | PASS | FAIL | CRASH | TIMEOUT | SKIP | 中位 MAE | 中位最大误差 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for gpu in (control, target):
            for stage in STAGES:
                s = stats(results, gpu, config_name, stage)
                lines.append(f"| {gpu} | {STAGE_CN[stage]} | {s['pass']} | {s['fail']} | {s['crash']} | {s['timeout']} | {s['skip']} | {fmt(s['median_mae'])} | {fmt(s['median_max'])} |")

        lines += ["", "### 2.2 自动诊断", ""] + interpretation(results, control, target, config_name)
        lines += ["", "### 2.3 每次独立运行", "",
                  "| GPU | 阶段 | Run | 状态 | RC | MAE | 最大误差 | Finite | NaN | Inf | 错误摘要 |",
                  "|---:|---|---:|---|---:|---:|---:|---|---:|---:|---|"]
        for r in results:
            if r.config != config_name:
                continue
            m = r.payload.get("metrics", {})
            err = str(r.payload.get("error", "") or "")
            if not err and r.stderr and r.status in ("CRASH", "TIMEOUT"):
                ps = r.stderr.splitlines()
                if ps:
                    err = ps[-1]
            err = err.replace("|", "\\|").replace("\n", " ")[:220]
            lines.append(f"| {r.gpu} | {STAGE_CN[r.stage]} | {r.run} | {r.status} | {r.rc} | {fmt(m.get('mae'))} | {fmt(m.get('max_error'))} | {m.get('finite','-')} | {m.get('nan_count','-')} | {m.get('inf_count','-')} | {err or '-'} |")
        lines.append("")

    lines += [
        "## 三、测试前 NVIDIA/Xid 内核日志快照", "", "```text", before, "```", "",
        "## 四、测试后 NVIDIA/Xid 内核日志快照", "", "```text", after, "```", "",
        "## 五、结果阅读顺序", "",
        "1. **copy 失败**：优先怀疑显存/内存控制器/传输路径/卡状态。",
        "2. **copy 正常、BF16 GEMM 失败**：问题进入普通 GEMM/Tensor Core/计算层。",
        "3. **BF16 GEMM 正常、SDPA Math 失败**：普通 attention 数学路径即可触发。",
        "4. **SDPA Math 正常、SDPA Flash 失败**：重点转向 fused attention / Tensor Core / shared memory / register / SM。",
        "5. **SDPA 正常、Sage FP16 失败**：Sage/custom-kernel 共用路径即可触发，不是 FP8 独有。",
        "6. **Sage FP16 正常、仅 Sage FP8 失败**：才最明显指向 FP8 专用路径。", "",
        "### 重要限制", "",
        "- 本报告用于缩小故障层级，不等于定位具体物理坏点。",
        "- 若健康对照卡也失败，应优先排查软件 build / kernel / shape 支持。",
        "- 若目标卡在固定输入、独立进程下结果仍波动明显，这是额外的卡稳定性异常信号。", "",
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
    if args.control_gpu == args.target_gpu:
        print("ERROR: control GPU 和 target GPU 不能相同", file=sys.stderr)
        return 2

    if args.size_sweep:
        configs = [(f"seq{n}_dim128", args.batch, args.heads, n, 128, args.gemm_size)
                   for n in (64, 128, 256, 512, 1024)]
    else:
        configs = [(f"seq{args.seq}_dim{args.dim}", args.batch, args.heads,
                    args.seq, args.dim, args.gemm_size)]

    print("=== SageAttention / GPU 分层诊断 ===")
    print(f"健康对照物理 GPU: {args.control_gpu}")
    print(f"目标物理 GPU:     {args.target_gpu}")
    print(f"Runs/stage:        {args.runs}")
    print("Stages:            " + ", ".join(args.stages))
    print(f"BF16 GEMM size:    {args.gemm_size}")
    print(f"诊断报告（原文件内更新）: {report}")
    print("每个 GPU/stage/run 都是全新 Python 子进程。\n")

    infos = {args.control_gpu: nvidia_info(args.control_gpu),
             args.target_gpu: nvidia_info(args.target_gpu)}
    before = kernel_log()
    results: list[ParentResult] = []

    with tempfile.TemporaryDirectory(prefix="sage_gpu_diag_cn_") as td:
        td = Path(td)
        for config_name, batch, heads, seq, dim, gemm_size in configs:
            bundle = td / f"{config_name}.pt"
            print(f"\n准备 CPU 固定输入/reference: B={batch} H={heads} N={seq} D={dim}, GEMM={gemm_size}")
            create_bundle(bundle, batch, heads, seq, dim, gemm_size, args.seed)
            for gpu in (args.control_gpu, args.target_gpu):
                print(f"\n--- 物理 GPU {gpu} | {config_name} ---")
                for stage in args.stages:
                    print(f"  Stage: {stage} ({STAGE_CN[stage]})")
                    for i in range(1, args.runs + 1):
                        r = run_child(args, script, bundle, gpu, config_name, stage, i)
                        results.append(r)
                        print(f"    run {i:02d}: {r.status:<7} rc={r.rc:<3} mae={fmt(metric(r,'mae')):>10} max={fmt(metric(r,'max_error')):>10}")
                        if r.status in ("FAIL", "CRASH", "TIMEOUT") and r.payload.get("error"):
                            print("      " + str(r.payload["error"])[:260])

    after = kernel_log()
    write_report(report, results, configs, args.control_gpu, args.target_gpu,
                 args.runs, infos, before, after)

    print("\n" + "=" * 80)
    print(f"诊断报告已更新: {report}")
    backup = report.with_name(report.stem + ".before_cn_v2.md")
    if backup.exists():
        print(f"第一轮报告自动备份: {backup}")
    summary_config = configs[-1][0] if args.size_sweep else configs[0][0]
    print("\n自动诊断摘要：")
    for line in interpretation(results, args.control_gpu, args.target_gpu, summary_config):
        print(line)
    print("=" * 80)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SageAttention / GPU 分层 A/B 诊断")
    p.add_argument("--control-gpu", type=int, default=1, help="健康对照物理 GPU")
    p.add_argument("--target-gpu", type=int, default=4, help="疑似异常物理 GPU")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--gemm-size", type=int, default=DEFAULT_GEMM_SIZE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    p.add_argument("--copy-loops", type=int, default=5)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--size-sweep", action="store_true")
    p.add_argument("--report", default=REPORT_NAME)
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
