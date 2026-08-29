#!/usr/bin/env python3
"""Exercise all four V24 routes through the real HTTP queue boundary.

The server is started separately in unified-console mode.  This client enters
each service family, submits Base then LoRA with the public two-control
contract, waits for the real job, downloads the result through the public
video endpoint, and records only portable release evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import aiohttp


ROUTES = (
    ("fl2va_base", "first_last", "base", 5),
    ("fl2va_lora", "first_last", "lora", 4),
    ("ref2va_base", "reference", "base", 5),
    ("ref2va_lora", "reference", "lora", 4),
)

FL2VA_PROMPT = (
    "室内固定镜头，女孩把桌上的蓝色书包背到肩上，男人帮她整理肩带。"
    "女孩问：\"这个书包是给我的吗？\"男人回答：\"是的。\""
    "动作连续自然，保留脚步声和衣物摩擦声，不要背景音乐。"
)
REF2VA_PROMPT = (
    "保持参考人物身份和声音。校园入口的固定镜头中，女孩背着蓝色书包，"
    "男人帮她调整肩带。男人说：\"准备好了吗？\"女孩回答：\"准备好了。\""
    "动作连续自然，保留环境声，不要背景音乐。"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18090")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--acceleration", type=float, default=75.0)
    parser.add_argument("--seed", type=int, default=82417)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    return parser.parse_args()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


async def _json(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    **kwargs: Any,
) -> dict[str, Any]:
    async with session.request(method, url, **kwargs) as response:
        text = await response.text()
        if response.status >= 400:
            raise RuntimeError(f"{method} {url} -> {response.status}: {text[:500]}")
        try:
            return json.loads(text)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{method} {url} returned non-JSON") from error


def _form(
    *,
    family: str,
    variant: str,
    steps: int,
    acceleration: float,
    seed: int,
    inputs: dict[str, Any],
) -> aiohttp.FormData:
    form = aiohttp.FormData()
    values = {
        "prompt": REF2VA_PROMPT if family == "reference" else FL2VA_PROMPT,
        "service_family": family,
        "model_variant": variant,
        "resolution": "360p",
        "aspect_ratio": "16:9",
        "duration_seconds": "1",
        "sampling_steps": str(steps),
        "acceleration": str(acceleration),
        "seed": str(seed),
        "preview_mode": "off",
    }
    for name, value in values.items():
        form.add_field(name, value)
    if family == "first_last":
        for role in ("first_frame", "last_frame"):
            path = inputs[role]
            form.add_field(
                role,
                path.read_bytes(),
                filename=path.name,
                content_type="image/png",
            )
    else:
        for index, path in enumerate(inputs["reference_images"], 1):
            form.add_field(
                f"reference_image_{index}",
                path.read_bytes(),
                filename=path.name,
                content_type="image/png",
            )
        for index, path in enumerate(inputs["reference_audios"], 1):
            form.add_field(
                f"reference_audio_{index}",
                path.read_bytes(),
                filename=path.name,
                content_type="audio/wav",
            )
    return form


async def _wait_job(
    session: aiohttp.ClientSession,
    base_url: str,
    job_id: str,
    poll_seconds: float,
) -> dict[str, Any]:
    last_marker: tuple[Any, Any] | None = None
    while True:
        job = await _json(session, "GET", f"{base_url}/api/v1/jobs/{job_id}")
        progress = job.get("progress") or {}
        marker = (job.get("status"), progress.get("stage"))
        if marker != last_marker:
            print(json.dumps({
                "event": "job_progress",
                "job_id": job_id,
                "status": job.get("status"),
                "stage": progress.get("stage"),
                "percent": progress.get("percent"),
            }, ensure_ascii=False), flush=True)
            last_marker = marker
        if job.get("status") in {"succeeded", "failed", "cancelled"}:
            return job
        await asyncio.sleep(poll_seconds)


def _inputs(root: Path) -> dict[str, Any]:
    result = {
        "first_frame": root / "first_frame.png",
        "last_frame": root / "last_frame.png",
        "reference_images": tuple(
            root / f"reference_image_{index}.png" for index in range(1, 4)
        ),
        "reference_audios": tuple(
            root / f"reference_audio_{index}.wav" for index in range(1, 3)
        ),
    }
    paths = (
        result["first_frame"],
        result["last_frame"],
        *result["reference_images"],
        *result["reference_audios"],
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("missing input(s): " + ", ".join(missing))
    return result


async def _main() -> int:
    args = _parse_args()
    base_url = args.base_url.rstrip("/")
    inputs = _inputs(args.input_root.resolve())
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.report.resolve()
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    report: dict[str, Any] = {
        "schema_version": "v24_http_real_service_matrix_v1",
        "purpose": "real HTTP + queue + download integration smoke; not quality candidates",
        "started_at_unix": time.time(),
        "base_url": base_url,
        "public_request": {
            "resolution": "360p",
            "aspect_ratio": "16:9",
            "duration_seconds": 1,
            "acceleration": args.acceleration,
        },
        "inputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in (
                inputs["first_frame"],
                inputs["last_frame"],
                *inputs["reference_images"],
                *inputs["reference_audios"],
            )
        },
        "routes": [],
    }
    _write_report(report_path, report)
    timeout = aiohttp.ClientTimeout(total=900, connect=30, sock_read=900)
    failures = 0
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        health = await _json(session, "GET", f"{base_url}/healthz")
        options = await _json(session, "GET", f"{base_url}/api/v1/options")
        openapi = await _json(session, "GET", f"{base_url}/openapi.json")
        report["server"] = {
            "version": health.get("version"),
            "initial_status": health.get("status"),
            "initial_active_service_family": health.get("active_service_family"),
            "deployment_mode": options.get("deployment_mode"),
            "public_service_families": sorted((options.get("service_families") or {}).keys()),
            "sampling_steps": (options.get("advanced_limits") or {}).get("sampling_steps"),
            "acceleration": (options.get("advanced_limits") or {}).get("acceleration"),
            "openapi_sha256": _sha256_bytes(
                json.dumps(openapi, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ),
        }
        _write_report(report_path, report)
        current_family: str | None = None
        try:
            for route_id, family, variant, steps in ROUTES:
                if family != current_family:
                    select_started = time.monotonic()
                    selected = await _json(
                        session,
                        "PUT",
                        f"{base_url}/api/v1/engine",
                        json={"service_family": family, "model_variant": variant},
                    )
                    switch_seconds = round(time.monotonic() - select_started, 3)
                    ready = await _json(session, "GET", f"{base_url}/readyz")
                    if ready.get("status") != "ready":
                        raise RuntimeError(f"{family} readiness failed: {ready}")
                    current_family = family
                else:
                    selected = {"changed": False, "active_engine": family}
                    switch_seconds = 0.0
                    ready = await _json(session, "GET", f"{base_url}/readyz")

                route: dict[str, Any] = {
                    "route_id": route_id,
                    "service_family": family,
                    "model_variant": variant,
                    "sampling_steps": steps,
                    "acceleration": args.acceleration,
                    "engine_switch_seconds": switch_seconds,
                    "engine_selection": selected,
                    "readiness": ready,
                    "status": "submitting",
                    "started_at_unix": time.time(),
                }
                report["routes"].append(route)
                _write_report(report_path, report)
                submit_started = time.monotonic()
                submitted = await _json(
                    session,
                    "POST",
                    f"{base_url}/api/v1/generations",
                    data=_form(
                        family=family,
                        variant=variant,
                        steps=steps,
                        acceleration=args.acceleration,
                        seed=args.seed,
                        inputs=inputs,
                    ),
                )
                route["submit_seconds"] = round(time.monotonic() - submit_started, 3)
                route["job_id"] = submitted["id"]
                route["accepted_request"] = submitted.get("request")
                route["status"] = "queued"
                _write_report(report_path, report)
                job = await _wait_job(
                    session, base_url, submitted["id"], args.poll_seconds
                )
                route["job"] = job
                route["status"] = job.get("status")
                if route["status"] != "succeeded":
                    failures += 1
                    _write_report(report_path, report)
                    continue
                executed_plan = job.get("inference_plan") or {}
                if variant == "base":
                    if executed_plan.get("policy_id") != (
                        "h3_pareto_v24_human_knee_continuous_deployment_v3"
                    ):
                        raise RuntimeError(
                            f"{route_id} did not execute the V24 Base policy"
                        )
                elif (
                    executed_plan.get("scheduler_family")
                    != "h3_lora_v1_no_forecast_round229"
                    or executed_plan.get("forecast_step_indices") != []
                ):
                    raise RuntimeError(
                        f"{route_id} did not preserve the no-Forecast LoRA contract"
                    )
                async with session.get(
                    f"{base_url}/api/v1/jobs/{submitted['id']}/video"
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(
                            f"video download failed: {response.status} {await response.text()}"
                        )
                    content = await response.read()
                    route["download_content_type"] = response.headers.get("Content-Type")
                output_file = f"{route_id}_{submitted['id']}.mp4"
                output_path = output_root / output_file
                output_path.write_bytes(content)
                route.update({
                    "download_file": output_file,
                    "download_bytes": len(content),
                    "download_sha256": _sha256_bytes(content),
                    "completed_at_unix": time.time(),
                })
                _write_report(report_path, report)
                print(json.dumps({
                    "event": "route_end",
                    "route": route_id,
                    "job_id": submitted["id"],
                    "elapsed_seconds": job.get("elapsed_seconds"),
                    "download_bytes": len(content),
                }), flush=True)
        finally:
            try:
                report["engine_exit"] = await _json(
                    session, "DELETE", f"{base_url}/api/v1/engine"
                )
            except Exception as error:
                report["engine_exit"] = {"error": str(error)}

    all_succeeded = (
        len(report["routes"]) == len(ROUTES)
        and all(route.get("status") == "succeeded" for route in report["routes"])
    )
    report["status"] = "succeeded" if failures == 0 and all_succeeded else "failed"
    report["completed_at_unix"] = time.time()
    _write_report(report_path, report)
    return 0 if report["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
