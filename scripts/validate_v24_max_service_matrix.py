#!/usr/bin/env python3
"""Run the V24 public service matrix at the 1080p x 15 s boundary.

This is a capability and integration stress test, not a quality comparison.
The deliberately low step counts keep the four-route release check bounded
while still exercising real weights, conditioning, VAE decode, audio muxing,
the V24 selector, and Base/LoRA hot switching in each service family.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from h3serve.config import ServicePaths
from h3serve.contract import GenerationSpec
from h3serve.native_engine import NativeHotH3Engine
from h3serve.native_engine.session_factory import NativeSessionFactory, NativeSessionPaths


RELEASE_ROOT = Path(__file__).resolve().parents[1]
ROUTES = (
    ("fl2va_base", "original", 5),
    ("fl2va_lora", "lora", 4),
    ("ref2va_base", "reference", 5),
    ("ref2va_lora", "reference_lora", 4),
)

FL2VA_PROMPT = (
    "明亮整洁的客厅里，一个深蓝色双肩书包摆在圆形木桌上。"
    "女孩走近桌子，检查书包前袋并把书包背到肩上；年轻男人走近，"
    "俯身帮她整理肩带。女孩问：\"叔叔，这个书包是送给我的吗？\""
    "男人回答：\"是的，背上试试看。\"单一连续镜头缓慢拉远，"
    "保留自然室内环境声、拉链声、脚步声和衣物摩擦声，不要背景音乐。"
)
REF2VA_PROMPT = (
    "15秒写实电影短片。清晨的校园入口，女孩背着深蓝色书包等待上学。"
    "年轻男人走近，轻拍她的肩膀并帮她调整书包肩带。男人说："
    "\"准备好了吗？今天上学别忘带东西。\"女孩回答："
    "\"都带好了，我自己可以背。\"随后两人自然地一起走进校园。"
    "镜头从书包细节平稳拉远到两人的背影，保留自然环境声，不要背景音乐。"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


class GpuSampler:
    """Low-rate external GPU sampling concurrent with the timed inference."""

    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return self.summary()

    def _run(self) -> None:
        command = (
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        )
        while not self._stop.is_set():
            sampled_at = time.time()
            try:
                output = subprocess.check_output(
                    command, text=True, timeout=5, stderr=subprocess.DEVNULL
                ).strip().splitlines()[0]
                utilization, memory_mib, power_w = (
                    float(part.strip()) for part in output.split(",")
                )
                self.samples.append({
                    "time": sampled_at,
                    "utilization_percent": utilization,
                    "memory_mib": memory_mib,
                    "power_w": power_w,
                })
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self._stop.wait(self.interval_seconds)

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
        return round(ordered[index], 3)

    def summary(self) -> dict[str, Any]:
        power = [sample["power_w"] for sample in self.samples]
        utilization = [sample["utilization_percent"] for sample in self.samples]
        memory = [sample["memory_mib"] for sample in self.samples]
        active_power = [
            sample["power_w"]
            for sample in self.samples
            if sample["utilization_percent"] >= 50.0
        ]
        return {
            "sample_count": len(self.samples),
            "interval_seconds": self.interval_seconds,
            "power_w_mean_all": round(statistics.fmean(power), 3) if power else None,
            "power_w_mean_active": (
                round(statistics.fmean(active_power), 3) if active_power else None
            ),
            "power_w_p50": self._percentile(power, 0.50),
            "power_w_p90": self._percentile(power, 0.90),
            "power_w_max": round(max(power), 3) if power else None,
            "utilization_percent_mean": (
                round(statistics.fmean(utilization), 3) if utilization else None
            ),
            "utilization_percent_p90": self._percentile(utilization, 0.90),
            "memory_mib_max": round(max(memory), 3) if memory else None,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--acceleration", type=float, default=75.0)
    parser.add_argument("--seed", type=int, default=82416)
    parser.add_argument(
        "--routes",
        nargs="*",
        choices=[item[0] for item in ROUTES],
        default=[item[0] for item in ROUTES],
    )
    return parser.parse_args()


def _required_inputs(input_root: Path) -> dict[str, Any]:
    values = {
        "first_frame": input_root / "first_frame.png",
        "last_frame": input_root / "last_frame.png",
        "reference_images": tuple(
            input_root / f"reference_image_{index}.png" for index in range(1, 4)
        ),
        "reference_audios": tuple(
            input_root / f"reference_audio_{index}.wav" for index in range(1, 3)
        ),
    }
    paths = (
        values["first_frame"],
        values["last_frame"],
        *values["reference_images"],
        *values["reference_audios"],
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("missing release validation input(s): " + ", ".join(missing))
    return values


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _progress(route_id: str):
    last: tuple[Any, Any] = (None, None)

    def callback(payload: dict[str, Any]) -> None:
        nonlocal last
        percent = payload.get("percent")
        stage = payload.get("stage")
        bucket = None if percent is None else int(float(percent) // 5) * 5
        marker = (stage, bucket)
        if marker != last:
            print(
                json.dumps({
                    "event": "progress",
                    "route": route_id,
                    "stage": stage,
                    "percent": percent,
                    "detail": payload.get("detail"),
                }, ensure_ascii=False),
                flush=True,
            )
            last = marker

    return callback


async def _main() -> int:
    args = _parse_args()
    inputs = _required_inputs(args.input_root.resolve())
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    paths = ServicePaths.defaults(RELEASE_ROOT)
    factory = NativeSessionFactory(NativeSessionPaths(
        model_root=paths.model_dir,
        minimax_source=paths.minimax_source_dir,
        lightx_source=paths.lightx_source_dir,
        turbo_curve=paths.turbo_curve_path,
        output_root=output_root,
    ))
    engine = NativeHotH3Engine(factory, output_root=output_root)
    requested = set(args.routes)
    selected = [item for item in ROUTES if item[0] in requested]
    if not selected:
        raise SystemExit("at least one route must be selected")
    report: dict[str, Any] = {
        "schema_version": "v24_max_service_matrix_real_e2e_v1",
        "purpose": "release capability smoke; low-step outputs are not quality candidates",
        "started_at_unix": time.time(),
        "release_version": __import__("h3serve").__version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "public_boundary": {
            "resolution": "1080p",
            "aspect_ratio": "16:9",
            "duration_seconds": 15,
            "width": 1920,
            "height": 1088,
            "frames": 362,
            "acceleration": args.acceleration,
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_HOME",
                "H3_NATIVE_ENABLE_SPARSE",
                "H3_NATIVE_PARETO_V24",
                "H3_NATIVE_SPARGE_BUILD_DIR",
                "H3_SERVE_MINIMAX_SOURCE",
                "H3_SERVE_LIGHTX_SOURCE",
                "H3_SERVE_LOCAL_MODEL_CACHE",
            )
        },
        "inputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (
                inputs["first_frame"],
                inputs["last_frame"],
                *inputs["reference_images"],
                *inputs["reference_audios"],
            )
        },
        "routes": [],
    }
    _write_report(args.report.resolve(), report)
    current_family: str | None = None
    failures = 0
    try:
        for route_id, public_engine, steps in selected:
            family = "reference" if route_id.startswith("ref2va") else "first_last"
            if family != current_family:
                preload_started = time.monotonic()
                print(json.dumps({"event": "preload_start", "family": family}), flush=True)
                await engine.preload(family)
                preload_seconds = round(time.monotonic() - preload_started, 3)
                warm_state = engine.warm_state
                print(json.dumps({
                    "event": "preload_end",
                    "family": family,
                    "seconds": preload_seconds,
                    "warm_state": warm_state,
                }, ensure_ascii=False), flush=True)
                if warm_state.get("status") != "ready":
                    raise RuntimeError(f"failed to preload {family}: {warm_state}")
                current_family = family
            else:
                preload_seconds = 0.0
                warm_state = engine.warm_state

            spec = GenerationSpec.from_mapping({
                "prompt": REF2VA_PROMPT if family == "reference" else FL2VA_PROMPT,
                "engine": public_engine,
                "resolution": "1080p",
                "aspect_ratio": "16:9",
                "duration_seconds": 15,
                "sampling_steps": steps,
                "acceleration": args.acceleration,
                "seed": args.seed,
            })
            output_path = output_root / (
                f"v24_final_{route_id}_1080p15_{steps}steps_"
                f"accel{args.acceleration:g}_seed{args.seed}.mp4"
            )
            route_report: dict[str, Any] = {
                "route_id": route_id,
                "status": "running",
                "preflight": factory.preflight(public_engine),
                "preload_seconds": preload_seconds,
                "warm_state": warm_state,
                "spec": spec.to_dict(include_execution=True),
                "output_file": output_path.name,
                "started_at_unix": time.time(),
            }
            report["routes"].append(route_report)
            _write_report(args.report.resolve(), report)
            sampler = GpuSampler()
            sampler.start()
            wall_started = time.monotonic()
            print(json.dumps({
                "event": "route_start",
                "route": route_id,
                "engine": public_engine,
                "steps": steps,
                "acceleration": args.acceleration,
            }), flush=True)
            try:
                result = await engine.generate(
                    spec=spec,
                    first_frame=(inputs["first_frame"] if family == "first_last" else None),
                    last_frame=(inputs["last_frame"] if family == "first_last" else None),
                    reference_images=(inputs["reference_images"] if family == "reference" else ()),
                    reference_videos=(),
                    reference_audios=(inputs["reference_audios"] if family == "reference" else ()),
                    cancel_event=asyncio.Event(),
                    output_path=output_path,
                    progress_callback=_progress(route_id),
                )
                route_report.update({
                    "status": "succeeded",
                    "wall_seconds": round(time.monotonic() - wall_started, 3),
                    "generation_elapsed_seconds": result.elapsed_seconds,
                    "runtime_key": result.runtime_key,
                    "stage_seconds": result.stage_seconds,
                    "inference_plan": result.inference_plan,
                    "output_bytes": result.output_path.stat().st_size,
                    "output_sha256": _sha256(result.output_path),
                })
            except Exception as error:
                failures += 1
                route_report.update({
                    "status": "failed",
                    "wall_seconds": round(time.monotonic() - wall_started, 3),
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
                print(json.dumps({
                    "event": "route_failed",
                    "route": route_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }, ensure_ascii=False), flush=True)
            finally:
                route_report["gpu"] = sampler.stop()
                route_report["completed_at_unix"] = time.time()
                _write_report(args.report.resolve(), report)
            print(json.dumps({
                "event": "route_end",
                "route": route_id,
                "status": route_report["status"],
                "wall_seconds": route_report.get("wall_seconds"),
                "gpu": route_report["gpu"],
            }, ensure_ascii=False), flush=True)
    finally:
        close_started = time.monotonic()
        await engine.close()
        report["close_seconds"] = round(time.monotonic() - close_started, 3)
        report["completed_at_unix"] = time.time()
        all_succeeded = (
            len(report["routes"]) == len(selected)
            and all(route.get("status") == "succeeded" for route in report["routes"])
        )
        report["status"] = (
            "succeeded" if failures == 0 and all_succeeded else "failed"
        )
        _write_report(args.report.resolve(), report)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
