#!/usr/bin/env python3
"""Launch one immutable Ref2VA candidate from the checked-in registry."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


SERVE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = SERVE_ROOT / "benchmarks/ref2va_extreme/candidates.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--execute", action="store_true",
        help="run the candidate; without this flag only print the command",
    )
    parser.add_argument(
        "--output-name",
        help="optional new directory name for a retry; existing directories are never reused",
    )
    return parser.parse_args()


def load_candidate(registry: Path, version: str) -> tuple[dict, dict]:
    document = json.loads(registry.read_text(encoding="utf-8"))
    matches = [item for item in document["candidates"] if item["version"] == version]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate candidate version: {version}")
    return document, matches[0]


def build_command(document: dict, candidate: dict, output_root: Path) -> list[str]:
    invariant = document["invariants"]
    command = [
        sys.executable,
        str(SERVE_ROOT / "scripts/benchmark_native_hot_session.py"),
        "--engine", "reference",
        "--scenario-manifest", str(
            SERVE_ROOT / "benchmarks/ref2va_extreme" / document["anchor"]
        ),
        "--label-prefix", candidate["version"],
        "--repeat", str(candidate.get("repeat", 1)),
        "--actual-steps", ",".join(str(value) for value in candidate["actual_steps"]),
        "--offload-mode", invariant["offload_mode"],
        "--prefetch-depth", str(invariant["prefetch_depth"]),
        "--mlp-chunk-tokens", str(invariant["mlp_chunk_tokens"]),
        "--vae-tile-size", str(invariant["vae_tile_size"]),
        "--vae-compile-feed-forward",
        "--memory-profile", invariant["memory_profile"],
        "--output-root", str(output_root),
    ]
    if not candidate["cache_condition_rows"]:
        command.append("--disable-condition-row-cache")
    if not candidate["cache_reference_latents"]:
        command.append("--disable-reference-latent-cache")
    if candidate.get("cache_condition_embeddings", False):
        command.append("--cache-condition-embeddings")
    return command


def main() -> int:
    args = parse_args()
    registry = args.registry.resolve()
    document, candidate = load_candidate(registry, args.version)
    output_name = args.output_name or candidate["version"]
    output_root = SERVE_ROOT / "runtime/outputs/ref2va_extreme" / output_name
    command = build_command(document, candidate, output_root)
    print(shlex.join(command), flush=True)
    if not args.execute:
        return 0
    if output_root.exists():
        raise FileExistsError(
            f"candidate output already exists; choose --output-name: {output_root}"
        )
    output_root.mkdir(parents=True)
    launch = {
        "schema_version": 1,
        "candidate": candidate,
        "registry": str(registry),
        "command": shlex.join(command),
        "status": "running",
    }
    launch_path = output_root / "launch.json"
    launch_path.write_text(
        json.dumps(launch, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(SERVE_ROOT)
            if not existing_pythonpath
            else f"{SERVE_ROOT}{os.pathsep}{existing_pythonpath}"
        )
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment.setdefault("TMPDIR", str(SERVE_ROOT / "runtime/tmp"))
        subprocess.run(
            command,
            cwd=SERVE_ROOT,
            env=environment,
            check=True,
        )
    except BaseException as error:
        launch["status"] = "failed"
        launch["error"] = f"{type(error).__name__}: {error}"
        raise
    else:
        launch["status"] = "complete"
    finally:
        launch_path.write_text(
            json.dumps(launch, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
