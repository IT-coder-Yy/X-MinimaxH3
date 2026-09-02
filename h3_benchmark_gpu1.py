#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
X-MinimaxH3 single-GPU benchmark (physical GPU 1 only).

Default benchmark:
  Prompt: A small dog sitting in front of a computer, typing code.
  Acceleration: 50
  Sampling steps: 20
  Aspect ratio: 16:9
  Repeats: 3 per case
  Warm-up: 1 x 480p / 5s (not counted)

Cases:
  480p / 5s
  480p / 10s
  720p / 5s
  720p / 10s
  1080p / 5s

Output:
  ./benchmark_report_gpu1.md

Resume behavior:
  If OOM kills the H3 service, the script saves the current report and exits.
  Restart the H3 service manually, then run:
      python h3_benchmark_gpu1.py --resume
  Completed runs are not repeated.

Important:
  - The benchmark never starts/restarts/stops the H3 service.
  - It never downloads generated videos.
  - If a case OOMs once, remaining repeats for that case are skipped and the
    benchmark moves to the next case (if the service is still alive).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional


GPU_INDEX = 1
REPORT_NAME = "benchmark_report_gpu1.md"

PROMPT = "A small dog sitting in front of a computer, typing code."
ACCELERATION = 50
SAMPLING_STEPS = 20
ASPECT_RATIO = "16:9"
MODEL_VARIANT = "base"
SEED = 12345
REPEATS = 3

CASES = [
    ("480p", 5),
    ("480p", 10),
    ("720p", 5),
    ("720p", 10),
    ("1080p", 5),
]

WARMUP_CASE = ("480p", 5)

POLL_INTERVAL_S = 1.0
GPU_SAMPLE_MS = 200
READY_WAIT_S = 120
CASE_TIMEOUT_S = 3600
OOM_RECOVERY_WAIT_S = 30

OOM_PATTERNS = (
    "out of memory",
    "cuda oom",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
    "alloc failed",
    "cannot allocate memory",
    "memoryerror",
)

STATE_BEGIN = "<!-- H3_BENCHMARK_STATE_V1"
STATE_END = "H3_BENCHMARK_STATE_V1 -->"


@dataclass
class RunRecord:
    case_id: str
    resolution: str
    duration_seconds: int
    run_index: int
    status: str
    job_id: str
    extra_vram_gib: Optional[float]
    generation_seconds: Optional[float]
    error: str
    started_at: str
    finished_at: str


class ApiError(RuntimeError):
    def __init__(self, code: int, body: str):
        self.code = code
        self.body = body
        super().__init__(f"HTTP {code}: {body[:1000]}")


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def is_oom(value: Any) -> bool:
    text = str(value or "").lower()
    return any(p in text for p in OOM_PATTERNS)


def run_command(args: list[str], timeout: float = 15.0) -> str:
    proc = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    return proc.stdout.strip()


def parse_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result

    pattern = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)\s*$")
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line)
        if not match:
            continue
        key = match.group(1)
        value = match.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        result[key] = value
    return result


def ensure_ascii_path(path: Path) -> None:
    try:
        str(path).encode("ascii")
    except UnicodeEncodeError:
        raise RuntimeError(f"Report path must contain ASCII characters only: {path}")


def query_gpu_memory_mib() -> float:
    out = run_command(
        [
            "nvidia-smi",
            "-i",
            str(GPU_INDEX),
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    return float(out.splitlines()[-1].strip())


def gpu_info() -> str:
    return run_command(
        [
            "nvidia-smi",
            "-i",
            str(GPU_INDEX),
            "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,"
            "utilization.gpu,power.draw,temperature.gpu,pstate",
            "--format=csv,noheader",
        ]
    )


def gpu_processes() -> str:
    return run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader",
        ],
        timeout=10,
    )


class GpuMonitor:
    def __init__(self) -> None:
        self.baseline_mib: Optional[float] = None
        self.samples_mib: list[float] = []
        self.proc: Optional[subprocess.Popen[str]] = None
        self.thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()

    def start(self) -> None:
        self.baseline_mib = query_gpu_memory_mib()
        self.proc = subprocess.Popen(
            [
                "nvidia-smi",
                "-i",
                str(GPU_INDEX),
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-lms",
                str(GPU_SAMPLE_MS),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self) -> None:
        assert self.proc is not None
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            if self.stop_flag.is_set():
                break
            try:
                self.samples_mib.append(float(line.strip()))
            except Exception:
                pass

    def stop(self) -> tuple[Optional[float], Optional[float]]:
        self.stop_flag.set()
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        if self.thread is not None:
            self.thread.join(timeout=2)

        baseline = self.baseline_mib
        peak = max(self.samples_mib) if self.samples_mib else baseline
        return baseline, peak


class H3Client:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[dict[str, Any]] = None,
        auth: bool = True,
        timeout: float = 20.0,
    ) -> Any:
        url = self.base_url + path
        headers = {"Accept": "application/json"}

        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        if auth and self.api_key:
            headers["X-API-Key"] = self.api_key

        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
                if not raw:
                    return {}
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return {"_raw": raw}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ApiError(exc.code, body)
        except urllib.error.URLError as exc:
            raise RuntimeError(f"API unreachable: {exc}")

    def http_status(self, path: str, timeout: float = 5.0) -> int:
        try:
            request = urllib.request.Request(self.base_url + path, method="GET")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status
        except urllib.error.HTTPError as exc:
            return exc.code
        except Exception:
            return 0

    def health_ok(self) -> bool:
        return self.http_status("/healthz") == 200

    def ready_ok(self) -> bool:
        return self.http_status("/readyz") == 200

    def wait_ready(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.ready_ok():
                return True
            time.sleep(2)
        return False

    def options(self) -> dict[str, Any]:
        value = self._request("GET", "/api/v1/options")
        return value if isinstance(value, dict) else {}

    def create_generation(self, payload: dict[str, Any]) -> dict[str, Any]:
        value = self._request(
            "POST",
            "/api/v1/generations",
            payload=payload,
            timeout=30,
        )
        if not isinstance(value, dict):
            raise RuntimeError(f"Unexpected generation response: {value!r}")
        return value

    def get_job(self, job_id: str) -> dict[str, Any]:
        value = self._request("GET", f"/api/v1/jobs/{job_id}", timeout=15)
        if not isinstance(value, dict):
            raise RuntimeError(f"Unexpected job response: {value!r}")
        return value


def get_job_error(job: dict[str, Any]) -> str:
    value = job.get("error")
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def mib_to_gib(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(value / 1024.0, 3)


def format_num(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def config_object() -> dict[str, Any]:
    return {
        "gpu_index": GPU_INDEX,
        "prompt": PROMPT,
        "acceleration": ACCELERATION,
        "sampling_steps": SAMPLING_STEPS,
        "aspect_ratio": ASPECT_RATIO,
        "model_variant": MODEL_VARIANT,
        "seed": SEED,
        "repeats": REPEATS,
        "cases": [list(c) for c in CASES],
        "warmup_case": list(WARMUP_CASE),
    }


def new_state(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": 1,
        "config": config_object(),
        "metadata": metadata,
        "warmup_done": False,
        "warmup_status": "not_run",
        "records": [],
        "case_oom": {},
        "created_at": now_text(),
        "updated_at": now_text(),
    }


def load_state(report_path: Path) -> dict[str, Any]:
    text = report_path.read_text(encoding="utf-8")
    start = text.find(STATE_BEGIN)
    end = text.find(STATE_END)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(
            f"Existing report does not contain resumable state: {report_path}"
        )
    payload = text[start + len(STATE_BEGIN):end].strip()
    return json.loads(payload)


def validate_resume_state(state: dict[str, Any]) -> None:
    old = state.get("config")
    current = config_object()
    if old != current:
        raise RuntimeError(
            "Benchmark configuration differs from the existing report. "
            "Do not resume with different cases/prompt/steps/acceleration."
        )


def records_from_state(state: dict[str, Any]) -> list[RunRecord]:
    records: list[RunRecord] = []
    for item in state.get("records", []):
        records.append(RunRecord(**item))
    return records


def median_success(
    records: list[RunRecord],
    field: str,
) -> Optional[float]:
    values: list[float] = []
    for record in records:
        if record.status != "succeeded":
            continue
        value = getattr(record, field)
        if value is not None:
            values.append(float(value))
    return statistics.median(values) if values else None


def write_report(path: Path, state: dict[str, Any]) -> None:
    records = records_from_state(state)
    metadata = state.get("metadata", {})

    lines: list[str] = []
    lines.append("# X-MinimaxH3 Deployment Benchmark - GPU1")
    lines.append("")
    lines.append("## Test configuration")
    lines.append("")
    lines.append(f"- Physical GPU: **GPU {GPU_INDEX} only**")
    lines.append(f"- Prompt: `{PROMPT}`")
    lines.append(f"- Sampling steps: `{SAMPLING_STEPS}`")
    lines.append(f"- Acceleration: `{ACCELERATION}`")
    lines.append(f"- Aspect ratio: `{ASPECT_RATIO}`")
    lines.append(f"- Model variant: `{MODEL_VARIANT}`")
    lines.append(f"- Seed: `{SEED}`")
    lines.append(f"- Repeats per case: `{REPEATS}`")
    lines.append(f"- Warm-up: `{WARMUP_CASE[0]} / {WARMUP_CASE[1]}s`, not counted")
    lines.append(f"- Sparse setting: `{metadata.get('sparse_env', '-')}`")
    lines.append(f"- Launcher: `{metadata.get('launcher', '-')}`")
    lines.append(f"- Engine: `{metadata.get('engine', '-')}`")
    lines.append("")
    lines.append("Generation time is measured from `POST /api/v1/generations` until the job reaches a terminal state.")
    lines.append("Extra VRAM is measured as `peak GPU1 memory.used - memory.used immediately before submission`.")
    lines.append("")
    lines.append("## GPU environment")
    lines.append("")
    lines.append("```text")
    lines.append(str(metadata.get("gpu_info", "-")))
    lines.append("```")
    lines.append("")
    lines.append("GPU compute processes captured before the benchmark:")
    lines.append("")
    lines.append("```text")
    lines.append(str(metadata.get("gpu_processes", "-")))
    lines.append("```")
    lines.append("")
    lines.append("## Warm-up")
    lines.append("")
    lines.append(f"- Status: `{state.get('warmup_status', 'unknown')}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Resolution | Duration | Successful runs | OOM | "
        "Median extra VRAM (GiB) | Median generation time (s) | Job IDs |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---|")

    for resolution, duration in CASES:
        case_id = f"{resolution}_{duration}s"
        case_records = [r for r in records if r.case_id == case_id]
        successes = [r for r in case_records if r.status == "succeeded"]
        oom_count = sum(1 for r in case_records if r.status == "oom")
        extra = median_success(case_records, "extra_vram_gib")
        seconds = median_success(case_records, "generation_seconds")
        job_ids = ", ".join(r.job_id for r in case_records if r.job_id) or "-"
        lines.append(
            f"| {resolution} | {duration}s | {len(successes)}/{REPEATS} | "
            f"{oom_count} | {format_num(extra, 3)} | {format_num(seconds, 2)} | {job_ids} |"
        )

    lines.append("")
    lines.append("## Run details")
    lines.append("")
    lines.append(
        "| Case | Run | Status | Extra VRAM (GiB) | Generation time (s) | Job ID | Error |"
    )
    lines.append("|---|---:|---|---:|---:|---|---|")

    for record in records:
        error = (record.error or "").replace("|", "\\|").replace("\n", " ")
        if len(error) > 220:
            error = error[:217] + "..."
        lines.append(
            f"| {record.case_id} | {record.run_index} | {record.status} | "
            f"{format_num(record.extra_vram_gib, 3)} | "
            f"{format_num(record.generation_seconds, 2)} | "
            f"{record.job_id or '-'} | {error or '-'} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Tests run sequentially; there is no multi-GPU or concurrent generation.")
    lines.append("- If one run of a case OOMs, remaining repeats of that case are skipped.")
    lines.append("- If OOM kills the H3 service, the report is saved first and the benchmark exits.")
    lines.append("- After manually restarting the H3 service, run the same script with `--resume`.")
    lines.append("- Completed runs are not repeated when resuming.")
    lines.append("- No generated video is downloaded or copied by this benchmark.")
    lines.append("- If another process changes its GPU1 memory usage during a run, the extra-VRAM number can be affected.")
    lines.append("")
    lines.append(f"Last updated: `{now_text()}`")
    lines.append("")
    lines.append(STATE_BEGIN)
    state["updated_at"] = now_text()
    lines.append(json.dumps(state, ensure_ascii=False, separators=(",", ":")))
    lines.append(STATE_END)
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def record_exists(
    state: dict[str, Any],
    case_id: str,
    run_index: int,
) -> bool:
    for item in state.get("records", []):
        if item.get("case_id") == case_id and int(item.get("run_index", -1)) == run_index:
            return True
    return False


def append_record(state: dict[str, Any], record: RunRecord) -> None:
    state.setdefault("records", []).append(asdict(record))


def payload_for(resolution: str, duration_seconds: int) -> dict[str, Any]:
    return {
        "prompt": PROMPT,
        "model_variant": MODEL_VARIANT,
        "mode": "preset",
        "resolution": resolution,
        "aspect_ratio": ASPECT_RATIO,
        "duration_seconds": duration_seconds,
        "sampling_steps": SAMPLING_STEPS,
        "acceleration": ACCELERATION,
        "seed": SEED,
    }


def run_generation(
    client: H3Client,
    resolution: str,
    duration_seconds: int,
    run_index: int,
) -> RunRecord:
    case_id = f"{resolution}_{duration_seconds}s"
    started_at = now_text()
    started_perf = time.perf_counter()
    job_id = ""
    status = "unknown"
    error = ""
    generation_seconds: Optional[float] = None

    monitor = GpuMonitor()
    monitor.start()

    try:
        try:
            created = client.create_generation(payload_for(resolution, duration_seconds))
        except Exception as exc:
            generation_seconds = time.perf_counter() - started_perf
            status = "oom" if is_oom(exc) else "create_failed"
            error = str(exc)
            return finish_record(
                monitor, case_id, resolution, duration_seconds, run_index,
                status, job_id, error, generation_seconds, started_at
            )

        job_id = str(created.get("id") or created.get("job_id") or "")
        if not job_id:
            generation_seconds = time.perf_counter() - started_perf
            return finish_record(
                monitor, case_id, resolution, duration_seconds, run_index,
                "create_failed", "", f"No job id returned: {created!r}",
                generation_seconds, started_at
            )

        deadline = time.monotonic() + CASE_TIMEOUT_S
        poll_failures = 0
        last_log = 0.0

        while time.monotonic() < deadline:
            try:
                job = client.get_job(job_id)
                poll_failures = 0
            except Exception as exc:
                poll_failures += 1

                # If the whole service disappeared, stop this run promptly.
                if not client.health_ok():
                    status = "service_down"
                    error = f"Service became unavailable while job {job_id} was running: {exc}"
                    break

                if poll_failures >= 10:
                    status = "poll_failed"
                    error = f"Job polling failed 10 times: {exc}"
                    break

                time.sleep(2)
                continue

            job_status = str(job.get("status", "")).lower()
            progress = job.get("progress") or {}

            if time.perf_counter() - last_log >= 5:
                print(
                    f"  job={job_id[:12]} status={job_status} "
                    f"progress={progress.get('percent', '-')} "
                    f"stage={progress.get('stage', '-')}"
                )
                last_log = time.perf_counter()

            if job_status == "succeeded":
                status = "succeeded"
                break

            if job_status in ("failed", "cancelled"):
                error = get_job_error(job)
                status = "oom" if is_oom(error) else job_status
                break

            if job_status in ("checkpointed", "awaiting_preview"):
                status = job_status
                error = f"Unexpected interactive job state: {job_status}"
                break

            time.sleep(POLL_INTERVAL_S)
        else:
            status = "timeout"
            error = f"Case exceeded {CASE_TIMEOUT_S}s"

        generation_seconds = time.perf_counter() - started_perf
        return finish_record(
            monitor, case_id, resolution, duration_seconds, run_index,
            status, job_id, error, generation_seconds, started_at
        )
    finally:
        if monitor.proc is not None and monitor.proc.poll() is None:
            try:
                monitor.stop()
            except Exception:
                pass


def finish_record(
    monitor: GpuMonitor,
    case_id: str,
    resolution: str,
    duration_seconds: int,
    run_index: int,
    status: str,
    job_id: str,
    error: str,
    generation_seconds: Optional[float],
    started_at: str,
) -> RunRecord:
    baseline_mib, peak_mib = monitor.stop()

    extra_gib: Optional[float] = None
    if baseline_mib is not None and peak_mib is not None:
        extra_gib = round(max(0.0, peak_mib - baseline_mib) / 1024.0, 3)

    record = RunRecord(
        case_id=case_id,
        resolution=resolution,
        duration_seconds=duration_seconds,
        run_index=run_index,
        status=status,
        job_id=job_id,
        extra_vram_gib=extra_gib,
        generation_seconds=(
            None if generation_seconds is None else round(generation_seconds, 3)
        ),
        error=error,
        started_at=started_at,
        finished_at=now_text(),
    )

    print(
        f"  => {record.status} | extra VRAM={format_num(record.extra_vram_gib, 3)} GiB "
        f"| time={format_num(record.generation_seconds, 2)} s "
        f"| job={record.job_id or '-'}"
    )
    if record.error:
        print("  error:", record.error[:500])

    return record


def run_warmup(client: H3Client) -> tuple[str, bool]:
    resolution, duration = WARMUP_CASE
    print(f"\n=== Warm-up: {resolution} / {duration}s (not counted) ===")
    record = run_generation(client, resolution, duration, 0)

    if record.status == "succeeded":
        return "succeeded", True

    if record.status == "oom":
        return "oom", client.health_ok()

    if record.status == "service_down":
        return "service_down", False

    return record.status, client.health_ok()


def wait_after_oom(client: H3Client) -> bool:
    print(
        f"\nOOM recorded. Checking service health for up to {OOM_RECOVERY_WAIT_S}s..."
    )
    deadline = time.monotonic() + OOM_RECOVERY_WAIT_S
    while time.monotonic() < deadline:
        if client.health_ok() and client.ready_ok():
            print("Service is still healthy and ready. Continuing with the next case.")
            return True
        time.sleep(2)
    return False


def stop_for_manual_restart(report_path: Path) -> int:
    print()
    print("=" * 78)
    print("H3 SERVICE IS NOT READY AFTER THE FAILURE.")
    print(f"Current benchmark state has been saved to: {report_path}")
    print()
    print("Please manually restart X-MinimaxH3, wait until /readyz returns HTTP 200,")
    print("then continue from the saved checkpoint with:")
    print()
    print("    runtime/venv/bin/python h3_benchmark_gpu1.py --resume")
    print()
    print("Completed runs will NOT be repeated.")
    print("=" * 78)
    return 75


def main() -> int:
    parser = argparse.ArgumentParser(
        description="X-MinimaxH3 benchmark on physical GPU1 only"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from ./benchmark_report_gpu1.md",
    )
    args = parser.parse_args()

    cwd = Path.cwd().resolve()
    report_path = cwd / REPORT_NAME
    ensure_ascii_path(report_path)

    env_path = cwd / ".env"
    env = parse_env(env_path)

    if env.get("CUDA_VISIBLE_DEVICES") != "1":
        print(
            "ERROR: This benchmark only allows physical GPU1.\n"
            f"Expected {env_path} to contain: export CUDA_VISIBLE_DEVICES=1\n"
            f"Current value: {env.get('CUDA_VISIBLE_DEVICES')!r}",
            file=sys.stderr,
        )
        return 2

    port = env.get("H3_SERVE_PORT", "21900")
    api_key = env.get("H3_SERVE_API_KEY", "")
    if not api_key:
        print(
            f"ERROR: H3_SERVE_API_KEY was not found in {env_path}",
            file=sys.stderr,
        )
        return 2

    base_url = f"http://127.0.0.1:{port}"
    client = H3Client(base_url, api_key)

    print("=== X-MinimaxH3 deployment benchmark ===")
    print("Working directory:", cwd)
    print("Report:", report_path)
    print("Physical GPU:", GPU_INDEX)
    print("Prompt:", PROMPT)
    print("Acceleration:", ACCELERATION)
    print("Sampling steps:", SAMPLING_STEPS)
    print("Cases:", CASES)
    print("Repeats:", REPEATS)

    if not client.health_ok():
        print(
            "ERROR: /healthz is not HTTP 200. Start/restart the H3 service first.",
            file=sys.stderr,
        )
        return 3

    if not client.wait_ready(READY_WAIT_S):
        print(
            f"ERROR: /readyz did not become HTTP 200 within {READY_WAIT_S}s.",
            file=sys.stderr,
        )
        return 3

    try:
        options = client.options()
    except Exception as exc:
        print(f"WARNING: Could not read /api/v1/options: {exc}")
        options = {}

    launcher = str(
        options.get("current_launcher")
        or options.get("active_launcher")
        or "unknown"
    )
    engine = str(
        options.get("current_engine")
        or options.get("active_engine")
        or "unknown"
    )

    if args.resume:
        if not report_path.exists():
            print(
                f"ERROR: Cannot resume because {report_path} does not exist.",
                file=sys.stderr,
            )
            return 4
        state = load_state(report_path)
        validate_resume_state(state)
        print("Resuming existing benchmark report.")
    else:
        if report_path.exists():
            print(
                f"ERROR: {report_path} already exists.\n"
                "Use --resume to continue it, or rename/remove the old report first.",
                file=sys.stderr,
            )
            return 4

        metadata = {
            "launcher": launcher,
            "engine": engine,
            "sparse_env": env.get("H3_NATIVE_ENABLE_SPARSE", "(unset)"),
            "gpu_info": gpu_info(),
            "gpu_processes": gpu_processes(),
            "base_url": base_url,
        }
        state = new_state(metadata)
        write_report(report_path, state)

    # Exactly one warm-up for this benchmark state.
    if not state.get("warmup_done", False):
        warmup_status, service_alive = run_warmup(client)
        state["warmup_status"] = warmup_status
        state["warmup_done"] = True
        write_report(report_path, state)

        if warmup_status != "succeeded":
            print(f"Warm-up failed with status: {warmup_status}")
            if not service_alive:
                return stop_for_manual_restart(report_path)
            print("Warm-up failed even though the service is still alive. Benchmark stopped.")
            return 5

        time.sleep(3)

    for resolution, duration in CASES:
        case_id = f"{resolution}_{duration}s"

        # Once a case OOMs, never retry the remaining repeats for that case.
        if state.get("case_oom", {}).get(case_id):
            print(f"\nSKIP {case_id}: this case already OOMed earlier.")
            continue

        print(f"\n=== Case: {resolution} / {duration}s ===")

        for run_index in range(1, REPEATS + 1):
            if record_exists(state, case_id, run_index):
                print(f"  SKIP run {run_index}: already completed.")
                continue

            if not client.ready_ok():
                print("  Service is not ready before this run.")
                write_report(report_path, state)
                return stop_for_manual_restart(report_path)

            print(f"\n  Run {run_index}/{REPEATS}")
            record = run_generation(
                client,
                resolution=resolution,
                duration_seconds=duration,
                run_index=run_index,
            )
            append_record(state, record)

            if record.status == "oom":
                state.setdefault("case_oom", {})[case_id] = True

            write_report(report_path, state)
            print(f"  Report updated: {report_path}")

            if record.status == "oom":
                # User requested: record OOM and continue with the next CASE,
                # not the next repeat of the same case.
                if not wait_after_oom(client):
                    write_report(report_path, state)
                    return stop_for_manual_restart(report_path)
                print(f"  Skipping remaining repeats of {case_id} after OOM.")
                break

            if record.status == "service_down":
                write_report(report_path, state)
                return stop_for_manual_restart(report_path)

            # Any non-OOM job failure is recorded, then continue.
            time.sleep(3)

    write_report(report_path, state)

    print()
    print("=" * 78)
    print("Benchmark finished.")
    print(f"Report: {report_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
