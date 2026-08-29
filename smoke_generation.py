#!/usr/bin/env python3
"""Run one real queued generation without requiring a browser."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from h3serve.app import JobService
from h3serve.backend import BackendManager
from h3serve.config import ServicePaths
from h3serve.contract import GenerationSpec


DEFAULT_PROMPT = (
    "A cinematic medium shot of an artisan working beside a warm forge in a medieval "
    "stone workshop. The artisan turns naturally toward the camera and says in Mandarin: "
    "火候正好，可以开始了。 Stable anatomy, coherent motion, realistic firelight, clear "
    "synchronized dialogue and workshop ambience, no text, no subtitles, no logos."
)


async def run(args: argparse.Namespace) -> dict:
    serve_dir = Path(__file__).resolve().parent
    release_root = (args.release_root or serve_dir).resolve()
    data_dir = (args.data_dir or serve_dir / "data").resolve()
    spec = GenerationSpec.from_mapping({
        "prompt": args.prompt,
        "engine": args.engine,
        "quality": args.quality,
        "resolution": args.resolution,
        "aspect_ratio": args.aspect_ratio,
        "duration_seconds": args.duration_seconds,
        "seed": args.seed,
    })
    paths = ServicePaths.defaults(release_root, data_dir=data_dir)
    backend = BackendManager(paths)
    service = JobService(data_dir, backend)
    await service.start()
    try:
        job = await service.submit(spec, {})
        last_status = None
        last_reported = 0.0
        while job.status not in {"succeeded", "failed", "cancelled"}:
            now = time.monotonic()
            if job.status != last_status or now - last_reported >= 30:
                print(json.dumps({
                    "id": job.id,
                    "status": job.status,
                    "queue_position": service.queue_position(job.id),
                    "runtime_key": job.runtime_key,
                }, ensure_ascii=False), flush=True)
                last_status = job.status
                last_reported = now
            await asyncio.sleep(2)
        result = service.serialize(job)
        if job.output_path is not None:
            result["output_path"] = str(job.output_path)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        if job.status != "succeeded":
            raise RuntimeError(job.error or f"generation ended with {job.status}")
        return result
    finally:
        await service.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--engine", choices=("original", "lora"), default="original")
    parser.add_argument(
        "--quality", choices=("fast", "balanced", "quality", "ultra"), default="balanced"
    )
    parser.add_argument(
        "--resolution", choices=("360p", "480p", "720p", "1080p"),
        default="480p",
    )
    parser.add_argument(
        "--aspect-ratio", choices=("1:1", "4:3", "3:4", "16:9", "9:16"), default="16:9"
    )
    parser.add_argument("--duration-seconds", type=float, default=5)
    parser.add_argument("--seed", type=int, default=9901)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
