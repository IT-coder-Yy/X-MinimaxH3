#!/usr/bin/env python3

"""定位 GPU4 的 warp illegal instruction：最小算子、同二进制 A/B、完整日志。

默认顺序：
  1. bf16_mean：只运行 K 的 BF16 mean；
  2. bf16_sub：使用 CPU 生成的正确 mean，只运行 K - mean；
  3. triton_quant：传入已中心化的 K，跳过 K - mean，直接运行 SageAttention
     的 per-thread INT8 Triton 量化 kernel。

父进程只生成 CPU 输入、启动子进程和采集只读 nvidia-smi 数据。每个
GPU/stage/run 都使用全新子进程；子进程通过 CUDA_VISIBLE_DEVICES 只看到
一张卡。GPU1 和 GPU4 共用同一个 Triton cache，以便复用同一份 sm_89 产物。

示例：
  # 先只跑 GPU4（不访问 GPU1）
  python diagnose_warp_illegal_instruction.py --target-only --target-gpu 4

  # GPU1/GPU4 A/B，并用当前 PATH 中的 Compute Sanitizer 包裹三个阶段
  python diagnose_warp_illegal_instruction.py \
    --control-gpu 1 --target-gpu 4 --compute-sanitizer

注意：若要匹配 CUDA Toolkit 12.9，应在启动本脚本前确保 CUDA_HOME 和 PATH
指向 CUDA 12.9；本脚本不会静默切换到其他版本的 Compute Sanitizer。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STAGES = ("bf16_mean", "bf16_sub", "triton_quant")
STAGE_CN = {
    "bf16_mean": "BF16 Mean（归约）",
    "bf16_sub": "BF16 K-Mean（广播减法）",
    "triton_quant": "SageAttention Triton INT8 量化",
}
RESULT_PREFIX = "WARP_DIAG_RESULT="
DEFAULT_REPORT = "tests/diagnose_gpu/warp_illegal_instruction_diagnosis.md"
SANITIZER_ERROR_EXITCODE = 97


@dataclass
class RunResult:
    gpu: int
    stage: str
    run: int
    status: str
    returncode: int
    payload: dict[str, Any]
    stdout_log: str
    stderr_log: str
    telemetry: dict[str, Any]
    sanitizer: dict[str, Any]
    cache_digest: str


def json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return str(value)


def emit_result(payload: dict[str, Any]) -> None:
    print(
        RESULT_PREFIX
        + json.dumps(json_safe(payload), ensure_ascii=False, sort_keys=True)
    )


def tensor_sha256(tensor: Any) -> str:
    raw = tensor.detach().cpu().contiguous().view(-1).view(__import__("torch").uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def tensor_metrics(output: Any, reference: Any) -> dict[str, Any]:
    import torch

    out = output.detach().cpu()
    ref = reference.detach().cpu()
    diff = (out.float() - ref.float()).abs()
    finite = bool(torch.isfinite(out.float()).all().item())
    exact = bool(torch.equal(out, ref))
    mismatch_count = int(
        torch.count_nonzero(out.view(torch.int16) != ref.view(torch.int16)).item()
    )
    return {
        "finite": finite,
        "exact": exact,
        "mismatch_count": mismatch_count,
        "mae": float(diff.mean().item()),
        "max_error": float(diff.max().item()),
        "output_sha256": tensor_sha256(out),
        "reference_sha256": tensor_sha256(ref),
    }


def runtime_info(torch_module: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": str(torch_module.__version__),
        "torch_cuda": str(torch_module.version.cuda),
    }
    try:
        import triton

        info["triton"] = str(triton.__version__)
    except (ImportError, AttributeError) as exc:
        info["triton"] = f"unavailable: {exc}"
    try:
        import importlib.metadata

        info["sageattention"] = importlib.metadata.version("sageattention")
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        info["sageattention"] = f"unknown: {exc}"
    try:
        info["device"] = torch_module.cuda.get_device_name(0)
        info["capability"] = list(torch_module.cuda.get_device_capability(0))
    except (RuntimeError, AssertionError) as exc:
        info["device_error"] = str(exc)
    return info


def child_main(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {
        "gpu": args.physical_gpu,
        "stage": args.stage,
        "status": "CRASH",
        "pass": False,
    }
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("当前子进程不可见 CUDA GPU")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                f"子进程应只看到一张 GPU，实际看到 {torch.cuda.device_count()} 张"
            )

        payload["runtime"] = runtime_info(torch)
        bundle = torch.load(args.bundle, map_location="cpu")
        q_cpu = bundle["q"]
        k_cpu = bundle["k"]
        km_cpu = bundle["km_ref"]
        centered_k_cpu = bundle["centered_k_ref"]
        device = torch.device("cuda:0")

        q = q_cpu.to(device)
        k = k_cpu.to(device)
        torch.cuda.synchronize()

        input_roundtrip = {
            "q": tensor_sha256(q) == tensor_sha256(q_cpu),
            "k": tensor_sha256(k) == tensor_sha256(k_cpu),
        }
        payload["input_roundtrip"] = input_roundtrip
        if not all(input_roundtrip.values()):
            raise RuntimeError("Q/K 输入传输到 GPU 后哈希不一致")

        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()

        if args.stage == "bf16_mean":
            output = k.mean(dim=2, keepdim=True)
            torch.cuda.synchronize()
            metrics = tensor_metrics(output, km_cpu)
            passed = metrics["finite"] and metrics["max_error"] <= args.mean_max_error
            payload.update(
                {
                    "status": "PASS" if passed else "FAIL",
                    "pass": passed,
                    "metrics": metrics,
                }
            )

        elif args.stage == "bf16_sub":
            # mean 在 CPU 上生成后再上传，只隔离 GPU 的广播减法。
            km = km_cpu.to(device)
            torch.cuda.synchronize()
            output = k - km
            torch.cuda.synchronize()
            metrics = tensor_metrics(output, centered_k_cpu)
            passed = metrics["finite"] and metrics["exact"]
            payload.update(
                {
                    "status": "PASS" if passed else "FAIL",
                    "pass": passed,
                    "metrics": metrics,
                }
            )

        elif args.stage == "triton_quant":
            # 使用 CPU 生成的中心化 K，并传 km=None，明确跳过 k = k - km。
            centered_k = centered_k_cpu.to(device)
            torch.cuda.synchronize()
            from sageattention.triton.quant_per_thread import per_thread_int8

            q_int8, q_scale, k_int8, k_scale = per_thread_int8(
                q,
                centered_k,
                km=None,
                tensor_layout="HND",
                BLKQ=128,
                WARPQ=32,
                BLKK=64,
                WARPK=64,
            )
            torch.cuda.synchronize()

            q_scale_cpu = q_scale.detach().cpu()
            k_scale_cpu = k_scale.detach().cpu()
            finite = bool(
                torch.isfinite(q_scale_cpu).all().item()
                and torch.isfinite(k_scale_cpu).all().item()
            )
            output_hashes = {
                "q_int8": tensor_sha256(q_int8),
                "q_scale": tensor_sha256(q_scale_cpu),
                "k_int8": tensor_sha256(k_int8),
                "k_scale": tensor_sha256(k_scale_cpu),
            }
            passed = finite
            payload.update(
                {
                    "status": "PASS" if passed else "FAIL",
                    "pass": passed,
                    "metrics": {
                        "finite_scales": finite,
                        "q_scale_min": float(q_scale_cpu.min().item()),
                        "q_scale_max": float(q_scale_cpu.max().item()),
                        "k_scale_min": float(k_scale_cpu.min().item()),
                        "k_scale_max": float(k_scale_cpu.max().item()),
                        "output_sha256": output_hashes,
                    },
                }
            )
        else:
            raise ValueError(f"未知 stage：{args.stage}")

        payload["elapsed_seconds"] = time.monotonic() - started
        payload["cuda_memory"] = {
            "allocated_mib": torch.cuda.memory_allocated(device) / 2**20,
            "reserved_mib": torch.cuda.memory_reserved(device) / 2**20,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        }
        emit_result(payload)
        return 0 if payload["pass"] else 12

    except Exception as exc:  # noqa: BLE001 - 子进程边界必须把 CUDA 异常写入报告
        payload.update(
            {
                "status": "CRASH",
                "pass": False,
                "exception_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        emit_result(payload)
        traceback.print_exc(file=sys.stderr)
        return 20


def create_bundle(path: Path, args: argparse.Namespace) -> None:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    shape = (args.batch, args.heads, args.seq, args.dim)
    q = torch.randn(shape, generator=generator, dtype=torch.float32).to(torch.bfloat16)
    k = torch.randn(shape, generator=generator, dtype=torch.float32).to(torch.bfloat16)

    # CPU float32 归约后只舍入一次到 BF16，作为固定输入和参考。
    km_ref = k.float().mean(dim=2, keepdim=True).to(torch.bfloat16)
    centered_k_ref = (k.float() - km_ref.float()).to(torch.bfloat16)
    torch.save(
        {
            "q": q,
            "k": k,
            "km_ref": km_ref,
            "centered_k_ref": centered_k_ref,
            "shape": shape,
            "seed": args.seed,
        },
        path,
    )


def parse_result(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            try:
                value = json.loads(line[len(RESULT_PREFIX) :])
                return value if isinstance(value, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


def first_matching_lines(
    text: str, patterns: tuple[str, ...], limit: int = 20
) -> list[str]:
    compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    matches: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        normalized = line.strip()
        if normalized and any(pattern.search(normalized) for pattern in compiled):
            if normalized not in seen:
                matches.append(normalized)
                seen.add(normalized)
            if len(matches) >= limit:
                break
    return matches


def classify_sanitizer(text: str, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False}

    summaries = re.findall(r"ERROR SUMMARY:\s*(\d+)\s+errors?", text, re.IGNORECASE)
    memory_lines = first_matching_lines(
        text,
        (
            r"Invalid\s+__(?:global|shared|local|constant)__\s+(?:read|write)",
            r"out\s+of\s+bounds",
            r"misaligned\s+(?:address|access)",
        ),
    )
    hardware_lines = first_matching_lines(
        text,
        (
            r"Hardware exception",
            r"Warp .*?(?:Exception|Illegal|Invalid|Fault|Overflow)",
            r"illegal instruction",
            r"invalid program counter|invalid pc",
        ),
    )
    api_lines = first_matching_lines(
        text,
        (
            r"Program hit error",
            r"API call .* returned error",
            r"cudaError[A-Za-z]+",
        ),
    )
    return {
        "enabled": True,
        "summary_errors": int(summaries[-1]) if summaries else None,
        "memory_access_errors": memory_lines,
        "hardware_exceptions": hardware_lines,
        "cuda_api_errors": api_lines,
        "has_precise_memory_error": bool(memory_lines),
        "has_hardware_exception": bool(hardware_lines),
    }


def query_telemetry(gpu: int) -> dict[str, Any]:
    fields = (
        "index,uuid,name,memory.used,temperature.gpu,power.draw,pstate,"
        "clocks.sm,clocks.mem,utilization.gpu"
    )
    command = [
        "nvidia-smi",
        "-i",
        str(gpu),
        f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "error": str(exc)}
    if completed.returncode:
        return {"ok": False, "error": completed.stderr.strip()}
    values = [part.strip() for part in completed.stdout.strip().split(",")]
    names = fields.split(",")
    return {
        "ok": len(values) == len(names),
        "values": dict(zip(names, values)),
        "raw": completed.stdout.strip(),
    }


def summarize_telemetry(samples: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [sample["values"] for sample in samples if sample.get("ok")]
    result: dict[str, Any] = {"sample_count": len(valid)}
    if not valid:
        errors = [sample.get("error") for sample in samples if sample.get("error")]
        result["errors"] = list(dict.fromkeys(errors))[:5]
        return result

    def numeric(field: str) -> list[float]:
        output: list[float] = []
        for sample in valid:
            try:
                output.append(float(sample[field]))
            except (KeyError, TypeError, ValueError):
                pass
        return output

    for field, label in (
        ("memory.used", "peak_memory_used_mib"),
        ("temperature.gpu", "peak_temperature_c"),
        ("power.draw", "peak_power_w"),
        ("clocks.sm", "min_sm_clock_mhz"),
        ("clocks.mem", "min_memory_clock_mhz"),
        ("utilization.gpu", "peak_utilization_percent"),
    ):
        values = numeric(field)
        if values:
            result[label] = min(values) if label.startswith("min_") else max(values)
    result["pstates"] = sorted({sample.get("pstate", "") for sample in valid})
    return result


def run_process(
    command: list[str],
    env: dict[str, str],
    gpu: int,
    timeout: int,
    monitor_interval: float,
) -> tuple[int, str, str, bool, dict[str, Any]]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    started = time.monotonic()
    samples: list[dict[str, Any]] = [query_telemetry(gpu)]
    timed_out = False
    stdout = ""
    stderr = ""
    while True:
        elapsed = time.monotonic() - started
        if elapsed >= timeout:
            timed_out = True
            process.kill()
            stdout, stderr = process.communicate()
            break
        try:
            stdout, stderr = process.communicate(
                timeout=min(monitor_interval, max(0.05, timeout - elapsed))
            )
            break
        except subprocess.TimeoutExpired:
            samples.append(query_telemetry(gpu))
    samples.append(query_telemetry(gpu))
    return (
        process.returncode if process.returncode is not None else 124,
        stdout,
        stderr,
        timed_out,
        summarize_telemetry(samples),
    )


def hash_cache_binaries(cache_dir: Path) -> str:
    suffixes = {".cubin", ".ptx", ".so", ".bin", ".fatbin"}
    records: list[str] = []
    if cache_dir.exists():
        for path in sorted(cache_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in suffixes:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                records.append(f"{path.relative_to(cache_dir)}\0{digest}")
    if not records:
        return ""
    return hashlib.sha256("\n".join(records).encode()).hexdigest()


def sanitizer_path(args: argparse.Namespace) -> str | None:
    if not args.compute_sanitizer:
        return None
    executable = shutil.which("compute-sanitizer")
    if not executable:
        raise RuntimeError(
            "PATH 中找不到 compute-sanitizer；请先让 CUDA_HOME/PATH 指向真实部署 Toolkit"
        )
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        expected = Path(cuda_home).resolve() / "bin" / "compute-sanitizer"
        if not expected.is_file():
            raise RuntimeError(f"CUDA_HOME 下找不到 Compute Sanitizer：{expected}")
        if Path(executable).resolve() != expected.resolve():
            raise RuntimeError(
                "PATH 中的 Compute Sanitizer 与 CUDA_HOME 不一致："
                f"PATH={Path(executable).resolve()}，CUDA_HOME={expected.resolve()}"
            )
        return str(expected)
    return executable


def child_command(
    args: argparse.Namespace,
    script: Path,
    bundle: Path,
    gpu: int,
    stage: str,
    sanitizer: str | None,
    coredump_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(script),
        "--child",
        "--physical-gpu",
        str(gpu),
        "--stage",
        stage,
        "--bundle",
        str(bundle),
        "--mean-max-error",
        str(args.mean_max_error),
    ]
    if sanitizer and stage in args.sanitizer_stages:
        command = [
            sanitizer,
            "--tool",
            "memcheck",
            "--target-processes",
            "all",
            "--report-api-errors",
            "all",
            "--show-backtrace",
            "yes",
            "--error-exitcode",
            str(SANITIZER_ERROR_EXITCODE),
            *command,
        ]
        if args.generate_coredump:
            command[1:1] = [
                "--generate-coredump",
                "yes",
                "--coredump-name",
                str(coredump_path),
            ]
    return command


def run_one(
    args: argparse.Namespace,
    script: Path,
    bundle: Path,
    artifact_dir: Path,
    cache_dir: Path,
    gpu: int,
    stage: str,
    run_index: int,
    sanitizer: str | None,
) -> RunResult:
    stem = f"gpu{gpu}_{stage}_run{run_index:02d}"
    stdout_path = artifact_dir / f"{stem}.stdout.log"
    stderr_path = artifact_dir / f"{stem}.stderr.log"
    coredump_path = artifact_dir / f"{stem}.nvcudmp"
    use_sanitizer = sanitizer is not None and stage in args.sanitizer_stages
    command = child_command(args, script, bundle, gpu, stage, sanitizer, coredump_path)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    env["TRITON_CACHE_DIR"] = str(cache_dir)

    returncode, stdout, stderr, timed_out, telemetry = run_process(
        command, env, gpu, args.timeout, args.monitor_interval
    )
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    combined = stdout + "\n" + stderr
    sanitizer_info = classify_sanitizer(combined, use_sanitizer)
    payload = parse_result(stdout)

    if timed_out:
        status = "TIMEOUT"
        payload = {"status": status, "pass": False, "error": f"超过 {args.timeout}s"}
    elif not payload:
        status = "CRASH"
        payload = {
            "status": status,
            "pass": False,
            "error": "子进程没有输出结构化结果；请查看完整日志",
        }
    else:
        status = str(payload.get("status", "CRASH"))

    summary_errors = sanitizer_info.get("summary_errors")
    if use_sanitizer and (
        returncode == SANITIZER_ERROR_EXITCODE
        or (isinstance(summary_errors, int) and summary_errors > 0)
    ):
        status = "FAIL"
        payload["status"] = status
        payload["pass"] = False

    return RunResult(
        gpu=gpu,
        stage=stage,
        run=run_index,
        status=status,
        returncode=returncode,
        payload=payload,
        stdout_log=str(stdout_path),
        stderr_log=str(stderr_path),
        telemetry=telemetry,
        sanitizer=sanitizer_info,
        cache_digest=hash_cache_binaries(cache_dir),
    )


def command_version(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"
    text = (completed.stdout + "\n" + completed.stderr).strip()
    return (
        text if completed.returncode == 0 else f"error({completed.returncode}): {text}"
    )


def output_digest(payload: dict[str, Any]) -> str:
    value = payload.get("metrics", {}).get("output_sha256", "")
    if isinstance(value, dict):
        joined = "\n".join(f"{key}:{value[key]}" for key in sorted(value))
        return hashlib.sha256(joined.encode()).hexdigest()
    return str(value)


def short(value: Any, length: int = 12) -> str:
    text = str(value) if value not in (None, "") else "-"
    return text[:length]


def md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def write_report(
    path: Path,
    args: argparse.Namespace,
    results: list[RunResult],
    selected_gpus: list[int],
    artifact_dir: Path,
    cache_dir: Path,
    sanitizer: str | None,
) -> None:
    runtime_rows: list[str] = []
    for gpu in selected_gpus:
        row = next((item for item in results if item.gpu == gpu), None)
        runtime = row.payload.get("runtime", {}) if row else {}
        runtime_rows.append(
            f"| {gpu} | {runtime.get('device', '-')} | {runtime.get('capability', '-')} | "
            f"{runtime.get('torch', '-')} | {runtime.get('torch_cuda', '-')} | "
            f"{runtime.get('triton', '-')} | {runtime.get('sageattention', '-')} |"
        )

    lines = [
        "# GPU Warp Illegal Instruction 最小化诊断",
        "",
        f"- 生成时间：`{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- 模式：`{'GPU4 only' if args.target_only else 'GPU1/GPU4 A/B'}`",
        f"- 物理 GPU：`{', '.join(map(str, selected_gpus))}`",
        f"- 阶段：`{', '.join(args.stages)}`",
        f"- 每阶段运行：`{args.runs}` 次，全新子进程",
        f"- 输入：`B={args.batch}, H={args.heads}, N={args.seq}, D={args.dim}, BF16`",
        "- CUDA_LAUNCH_BLOCKING：`1`",
        f"- Triton cache：`{cache_dir}`（A/B 共用）",
        f"- Compute Sanitizer：`{sanitizer or 'disabled'}`",
        f"- 完整日志目录：`{artifact_dir}`",
        "",
        "## 环境",
        "",
        "```text",
        f"Python: {sys.executable}",
        f"CUDA_HOME: {os.environ.get('CUDA_HOME', '-')}",
        f"nvcc:\n{command_version(['nvcc', '--version'])}",
        f"compute-sanitizer:\n{command_version([sanitizer, '--version']) if sanitizer else 'disabled'}",
        "```",
        "",
        "### 子进程运行时",
        "",
        "| 物理GPU | 设备 | Capability | PyTorch | CUDA Runtime | Triton | SageAttention |",
        "|---:|---|---|---|---|---|---|",
        *runtime_rows,
        "",
        "## 结果",
        "",
        "| GPU | 阶段 | Run | 状态 | RC | Exact | MAE | Max error | 输出哈希 | Sanitizer汇总 | 精确内存错误 | 硬件异常 | 峰值PyTorch显存 | nvidia-smi峰值 |",
        "|---:|---|---:|---|---:|---|---:|---:|---|---:|---|---|---:|---:|",
    ]
    for row in results:
        metrics = row.payload.get("metrics", {})
        cuda_memory = row.payload.get("cuda_memory", {})
        sanitizer_info = row.sanitizer
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.gpu),
                    STAGE_CN[row.stage],
                    str(row.run),
                    row.status,
                    str(row.returncode),
                    str(metrics.get("exact", "-")),
                    short(metrics.get("mae")),
                    short(metrics.get("max_error")),
                    short(output_digest(row.payload)),
                    str(sanitizer_info.get("summary_errors", "-")),
                    "YES" if sanitizer_info.get("has_precise_memory_error") else "NO",
                    "YES" if sanitizer_info.get("has_hardware_exception") else "NO",
                    short(cuda_memory.get("peak_allocated_mib")),
                    short(row.telemetry.get("peak_memory_used_mib")),
                ]
            )
            + " |"
        )

    if not args.target_only:
        lines += ["", "## GPU1/GPU4 同输入结果对照", ""]
        lines += [
            "| 阶段 | Run | GPU1状态 | GPU4状态 | 输出哈希一致 | Triton二进制集合哈希一致 |",
            "|---|---:|---|---|---|---|",
        ]
        indexed = {(row.gpu, row.stage, row.run): row for row in results}
        for stage in args.stages:
            for run_index in range(1, args.runs + 1):
                control = indexed.get((args.control_gpu, stage, run_index))
                target = indexed.get((args.target_gpu, stage, run_index))
                if not control or not target:
                    continue
                left = output_digest(control.payload)
                right = output_digest(target.payload)
                output_same = "YES" if left and left == right else "NO/UNKNOWN"
                cache_same = (
                    "YES"
                    if control.cache_digest
                    and target.cache_digest
                    and control.cache_digest == target.cache_digest
                    else "NO/UNKNOWN"
                )
                lines.append(
                    f"| {STAGE_CN[stage]} | {run_index} | {control.status} | "
                    f"{target.status} | {output_same} | {cache_same} |"
                )

    lines += ["", "## Sanitizer 分类与完整日志", ""]
    for row in results:
        if not row.sanitizer.get("enabled"):
            continue
        lines += [
            f"### GPU{row.gpu} / {row.stage} / Run {row.run}",
            "",
            f"- Summary errors：`{row.sanitizer.get('summary_errors')}`",
            f"- stdout：`{row.stdout_log}`",
            f"- stderr：`{row.stderr_log}`",
            f"- 精确内存访问错误：`{len(row.sanitizer.get('memory_access_errors', []))}` 类/行",
            f"- 硬件异常：`{len(row.sanitizer.get('hardware_exceptions', []))}` 类/行",
            f"- CUDA API错误：`{len(row.sanitizer.get('cuda_api_errors', []))}` 类/行",
            "",
        ]
        for title, key in (
            ("精确内存访问错误", "memory_access_errors"),
            ("硬件异常", "hardware_exceptions"),
            ("CUDA API错误", "cuda_api_errors"),
        ):
            values = row.sanitizer.get(key, [])
            if values:
                lines += [f"**{title}**", "", "```text"]
                lines += [md_escape(value) for value in values]
                lines += ["```", ""]

    lines += [
        "## 判读规则",
        "",
        "1. GPU4 在 `bf16_mean` 失败、GPU1 通过：优先缩小到 BF16 归约/普通计算路径。",
        "2. `bf16_mean` 通过但 GPU4 在 `bf16_sub` 失败：优先缩小到普通 BF16 广播减法；此时尚未进入 Triton 量化 kernel。",
        "3. 前两项都通过，仅 GPU4 在 `triton_quant` 失败：再调查 Triton 量化 kernel、生成的 cubin 和具体异常 PC。",
        "4. 两张卡在同一 Sanitizer 下报同类异常：优先排查 Toolkit、驱动、Triton 与 Sanitizer 组合。",
        "5. A/B 使用相同缓存且二进制哈希相同，只有 GPU4 失败：优先转向 GPU4 本体、插槽或供电路径。",
        "",
        "说明：Sanitizer 的 `ERROR SUMMARY` 可以包含精确内存错误、硬件异常和 CUDA API错误；本报告分别分类，不把它们统一称为内存错误。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def self_test() -> int:
    sample = """
========= Invalid __global__ read of size 4 bytes
========= Hardware exception encountered: Warp Illegal Instruction
========= Program hit error 700 on CUDA API call to cudaDeviceSynchronize
========= ERROR SUMMARY: 3 errors
"""
    parsed = classify_sanitizer(sample, True)
    assert parsed["summary_errors"] == 3
    assert parsed["has_precise_memory_error"]
    assert parsed["has_hardware_exception"]
    assert parsed["cuda_api_errors"]
    assert parse_result(RESULT_PREFIX + '{"status":"PASS"}')["status"] == "PASS"
    print("self-test PASS")
    return 0


def parent_main(args: argparse.Namespace) -> int:
    if args.self_test:
        return self_test()
    if args.runs < 1:
        raise ValueError("--runs 必须至少为 1")
    if args.control_gpu == args.target_gpu and not args.target_only:
        raise ValueError("A/B 模式下 control GPU 和 target GPU 不能相同")
    if args.monitor_interval <= 0:
        raise ValueError("--monitor-interval 必须大于 0")

    script = Path(__file__).resolve()
    report = Path(args.report).resolve()
    artifact_dir = (
        Path(args.artifact_dir).resolve()
        if args.artifact_dir
        else report.with_name(report.stem + "_artifacts")
    )
    cache_dir = (
        Path(args.triton_cache_dir).resolve()
        if args.triton_cache_dir
        else artifact_dir / "triton-cache"
    )
    selected_gpus = (
        [args.target_gpu] if args.target_only else [args.control_gpu, args.target_gpu]
    )

    if args.dry_run:
        print("DRY RUN：不会导入 PyTorch，也不会访问 GPU")
        print(f"GPU：{selected_gpus}")
        print(f"Stages：{list(args.stages)}")
        print(f"Report：{report}")
        print(f"Artifacts：{artifact_dir}")
        print(f"Triton cache：{cache_dir}")
        print(
            f"Compute Sanitizer：{'enabled' if args.compute_sanitizer else 'disabled'}"
        )
        return 0

    sanitizer = sanitizer_path(args)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    bundle = artifact_dir / "fixed_cpu_inputs.pt"
    create_bundle(bundle, args)

    print("=== GPU Warp Illegal Instruction 最小化诊断 ===")
    print(
        f"模式：{'仅 GPU' + str(args.target_gpu) if args.target_only else 'GPU' + str(args.control_gpu) + '/GPU' + str(args.target_gpu) + ' A/B'}"
    )
    print(f"阶段：{', '.join(args.stages)}")
    print(f"完整日志：{artifact_dir}")
    print(f"共享 Triton cache：{cache_dir}")
    print(f"Compute Sanitizer：{sanitizer or 'disabled'}")

    results: list[RunResult] = []
    # 每个 stage 都先跑健康对照，再跑目标卡；目标卡复用健康卡生成的缓存。
    for stage in args.stages:
        for gpu in selected_gpus:
            for run_index in range(1, args.runs + 1):
                print(f"GPU{gpu} / {stage} / run {run_index} ...", flush=True)
                row = run_one(
                    args,
                    script,
                    bundle,
                    artifact_dir,
                    cache_dir,
                    gpu,
                    stage,
                    run_index,
                    sanitizer,
                )
                results.append(row)
                sanitizer_summary = row.sanitizer.get("summary_errors", "-")
                print(
                    f"  {row.status} rc={row.returncode} sanitizer_errors={sanitizer_summary}",
                    flush=True,
                )

    write_report(
        report,
        args,
        results,
        selected_gpus,
        artifact_dir,
        cache_dir,
        sanitizer,
    )
    print(f"报告：{report}")
    failed = [row for row in results if row.status != "PASS"]
    return 12 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="最小化定位 GPU warp illegal instruction，并对 GPU1/GPU4 做同二进制 A/B"
    )
    parser.add_argument("--control-gpu", type=int, default=1, help="健康对照物理 GPU")
    parser.add_argument("--target-gpu", type=int, default=4, help="疑似异常物理 GPU")
    parser.add_argument(
        "--target-only",
        action="store_true",
        help="只访问 target GPU，不查询或初始化 control GPU",
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="每个 GPU/stage 的独立进程次数"
    )
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--mean-max-error",
        type=float,
        default=0.02,
        help="GPU BF16 mean 相对固定 CPU reference 的最大允许误差",
    )
    parser.add_argument("--compute-sanitizer", action="store_true")
    parser.add_argument(
        "--sanitizer-stages", nargs="+", choices=STAGES, default=list(STAGES)
    )
    parser.add_argument(
        "--generate-coredump",
        action="store_true",
        help="让 Compute Sanitizer 在检测到错误时生成 GPU coredump",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--monitor-interval", type=float, default=0.25)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--artifact-dir")
    parser.add_argument("--triton-cache-dir")
    parser.add_argument("--dry-run", action="store_true", help="只显示计划，不访问 GPU")
    parser.add_argument("--self-test", action="store_true", help="只测试日志解析器")

    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--physical-gpu", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--stage", choices=STAGES, help=argparse.SUPPRESS)
    parser.add_argument("--bundle", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.child:
            if args.physical_gpu < 0 or not args.stage or not args.bundle:
                parser.error("子进程缺少 --physical-gpu/--stage/--bundle")
            return child_main(args)
        return parent_main(args)
    except Exception as exc:  # noqa: BLE001 - CLI 边界统一转为可读错误
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
